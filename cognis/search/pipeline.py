"""
Hybrid RRF search pipeline for Cognis.

Core accuracy engine — ported from production provider.py:1996-2062.
Combines two-stage Matryoshka vector search + BM25 + immediate recall
with Reciprocal Rank Fusion (70% vector + 30% BM25, k=10).
"""

import math
import time
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

from cognis.config import CognisConfig
from cognis.models import Memory
from cognis.stores.qdrant_store import QdrantLocalStore
from cognis.stores.sqlite_store import SQLiteStore
from cognis.embeddings.base import BaseEmbedder
from cognis.search.temporal import parse_temporal_query, calculate_temporal_relevance
from cognis.utils import now_utc

logger = logging.getLogger(__name__)


class HybridSearchPipeline:
    """
    Hybrid search pipeline with RRF fusion.

    Pipeline (ported from production):
    1. Generate query embeddings (768D + 256D)
    2. Two-stage Matryoshka vector search (256D shortlist -> 768D rerank)
    3. BM25 keyword search via SQLite FTS5
    4. Immediate recall (raw messages in 256D)
    5. Merge + deduplicate
    6. RRF fusion: 70% vector + 30% BM25 (k=10)
    7. Recency boost: 0.25 * exp(-age_seconds / 120)
    8. Temporal boosting: 0.6 * rrf + 0.4 * temporal (if temporal query)
    9. Content dedup + query echo filtering
    """

    def __init__(
        self,
        qdrant_store: QdrantLocalStore,
        sqlite_store: SQLiteStore,
        embedder: BaseEmbedder,
        config: CognisConfig,
    ):
        self._qdrant = qdrant_store
        self._sqlite = sqlite_store
        self._embedder = embedder
        self._config = config

    def search(
        self,
        query: str,
        owner_id: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 10,
        include_historical: bool = False,
    ) -> List[Dict[str, Any]]:
        t_start = time.time()

        # 0. Parse temporal intent
        is_temporal, query_date, window_days = parse_temporal_query(query)

        # 1. Generate query embeddings (both dimensions)
        t_emb = time.time()
        query_embedding = self._embedder.embed_query(query)
        emb_full = query_embedding.get(self._config.embedding_full_dim)
        emb_small = query_embedding.get(self._config.embedding_small_dim)
        logger.debug("Embedding: %.3fs", time.time() - t_emb)

        if not emb_full or not emb_small:
            logger.error("Embedding failed for query: %s", query[:50])
            return []

        candidate_limit = limit * 3

        # 2. Two-stage vector search (extracted memories — global to owner, no session filter)
        t_vec = time.time()
        vector_results = self._qdrant.search_two_stage(
            query_full=emb_full,
            query_small=emb_small,
            owner_id=owner_id,
            agent_id=agent_id,
            session_id=None,  # memories are global to owner
            shortlist_limit=self._config.shortlist_size,
            limit=candidate_limit,
            include_historical=include_historical,
            global_memory=True,
        )
        logger.debug("Vector search: %d results in %.3fs", len(vector_results), time.time() - t_vec)

        # 3. BM25 text search (extracted memories — global to owner, no session filter)
        t_bm25 = time.time()
        bm25_results = self._sqlite.text_search(
            query=query,
            owner_id=owner_id,
            agent_id=agent_id,
            session_id=None,  # memories are global to owner
            limit=candidate_limit,
            include_historical=include_historical,
        )
        logger.debug("BM25 search: %d results in %.3fs", len(bm25_results), time.time() - t_bm25)

        # 4. Immediate recall (raw messages — session-scoped)
        t_imm = time.time()
        immediate_results = self._qdrant.search_immediate(
            query_256d=emb_small,
            owner_id=owner_id,
            session_id=session_id,  # messages are session-scoped
            limit=50,
        )
        logger.debug("Immediate recall: %d results in %.3fs", len(immediate_results), time.time() - t_imm)

        # 5. Filter by similarity threshold
        min_sim = self._config.similarity_threshold
        filtered_vector = [(m, s) for m, s in vector_results if s >= min_sim]
        # Ensure minimum results (ported from provider.py:1850-1860)
        if len(filtered_vector) < min(limit, 10) and vector_results:
            filtered_vector = vector_results[:min(limit, 10)]

        # 6. Merge all results into unified dict
        all_memories: Dict[str, Memory] = {}
        vector_scores: Dict[str, float] = {}
        for mem, score in filtered_vector:
            all_memories[mem.memory_id] = mem
            vector_scores[mem.memory_id] = score

        bm25_scores: Dict[str, float] = {}
        for mem, score in bm25_results:
            bm25_scores[mem.memory_id] = min(1.0, score / 10.0)
            if mem.memory_id not in all_memories:
                all_memories[mem.memory_id] = mem

        immediate_scores: Dict[str, float] = {}
        for msg in immediate_results:
            msg_id = msg.get("memory_id", "")
            if msg_id:
                immediate_scores[msg_id] = msg.get("score", 0.0)
                if msg_id not in all_memories:
                    # Create pseudo-memory for immediate recall
                    pseudo = Memory(
                        memory_id=msg_id,
                        content=msg.get("content", ""),
                        owner_id=owner_id,
                        session_id=session_id,
                    )
                    all_memories[msg_id] = pseudo

        if not all_memories:
            return []

        # 7. RRF Fusion (ported verbatim from provider.py:1996-2062)
        k = self._config.rrf_k  # 10

        # Build rank maps
        vector_rank = {m.memory_id: i + 1 for i, (m, _) in enumerate(filtered_vector)}
        bm25_rank = {m.memory_id: i + 1 for i, (m, _) in enumerate(bm25_results)}
        immediate_rank = {msg["memory_id"]: i + 1 for i, msg in enumerate(immediate_results) if msg.get("memory_id")}

        _now = now_utc()
        scored_memories = []

        for memory_id, memory in all_memories.items():
            # RRF score from vector (normalized to 0-1 range)
            v_rank = vector_rank.get(memory_id, candidate_limit + 1)
            rrf_vector = (1.0 / (k + v_rank)) * (k + 1)

            # RRF score from BM25 (normalized to 0-1 range)
            b_rank = bm25_rank.get(memory_id, candidate_limit + 1)
            rrf_bm25 = (1.0 / (k + b_rank)) * (k + 1)

            # Immediate recall bonus (if not already in vector results)
            immediate_bonus = 0.0
            if memory_id in immediate_rank and memory_id not in vector_rank:
                i_rank = immediate_rank[memory_id]
                immediate_bonus = (1.0 / (k + i_rank)) * (k + 1) * self._config.vector_weight

            # Combined RRF (70% vector + 30% BM25)
            rrf_score = (
                self._config.vector_weight * rrf_vector
                + self._config.bm25_weight * rrf_bm25
                + immediate_bonus
            )

            # Recency boost: 0.25 * exp(-age_seconds / 120)
            recency_boost = 0.0
            if memory.created_at:
                created = memory.created_at
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                age_seconds = max(0, (_now - created).total_seconds())
                recency_boost = self._config.recency_boost_weight * math.exp(
                    -age_seconds / self._config.recency_half_life_seconds
                )

            # Temporal boosting (for explicit temporal queries)
            temporal_score = 1.0
            if is_temporal and query_date:
                temporal_score = calculate_temporal_relevance(
                    memory, query_date=query_date, window_days=window_days,
                )
                ranking_score = 0.6 * rrf_score + 0.4 * temporal_score + recency_boost
            else:
                ranking_score = rrf_score + recency_boost

            # Display score: use actual similarity if available (more interpretable)
            display_score = vector_scores.get(
                memory_id,
                bm25_scores.get(memory_id, immediate_scores.get(memory_id, rrf_score)),
            )

            scored_memories.append((memory, ranking_score, temporal_score, display_score))

        # 8. Sort by ranking score
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        # 9. Prefer extracted memories over raw messages, then dedup
        # Split into extracted facts vs raw messages
        extracted = [(m, r, t, d) for m, r, t, d in scored_memories if m.memory_id not in immediate_scores or m.memory_id in vector_rank]
        raw_msgs = [(m, r, t, d) for m, r, t, d in scored_memories if m.memory_id in immediate_scores and m.memory_id not in vector_rank]
        # Extracted first, raw messages fill remaining slots
        ordered = extracted + raw_msgs

        query_norm = query.strip().lower()
        seen = set()
        deduped = []
        for mem, rank_s, temp_s, disp_s in ordered:
            content_key = mem.content.strip().lower()[:100]
            if content_key == query_norm:
                continue
            if content_key and content_key not in seen:
                seen.add(content_key)
                deduped.append((mem, disp_s))

        # 10. Return top-k
        top = deduped[:limit]

        logger.debug(
            "Search complete: %d results in %.3fs (vec=%d, bm25=%d, imm=%d)",
            len(top), time.time() - t_start,
            len(vector_results), len(bm25_results), len(immediate_results),
        )

        return [{**mem.to_dict(), "score": round(score, 4)} for mem, score in top]
