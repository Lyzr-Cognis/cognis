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
from cognis.rerank import RemoteReranker
from cognis.utils import now_utc

logger = logging.getLogger(__name__)


def _is_history_query(query: str) -> bool:
    """Conservative history-intent detector; deleted versions stay excluded."""
    lowered = query.lower()
    return any(term in lowered for term in (
        "previous", "before", "history", "journey", "changes", "all the",
        "were", "used to", "changed", "evolution", "over time", "different",
        "past", "earlier", "originally", "first", "initially", "started", "began",
    ))


def _is_global_query(query: str) -> bool:
    """Summarization / whole-history intent: needs breadth, not precision."""
    lowered = query.lower()
    return any(term in lowered for term in (
        "summarize", "summarise", "summary", "overview", "recap",
        "comprehensive", "overall", "everything about", "all about",
        "tell me everything", "big picture",
    ))


def _is_ordering_query(query: str) -> bool:
    """Event-ordering intent: chronological presentation helps reconstruction."""
    lowered = query.lower()
    return any(term in lowered for term in (
        "order of", "in what order", "sequence of", "chronolog", "timeline",
        "what happened first", "which came first", "which happened first",
        "first to last", "order did",
    ))


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


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
        self._reranker = (
            RemoteReranker(config.remote_rerank_url, timeout_seconds=config.remote_rerank_timeout)
            if config.rerank_provider == "remote" and config.remote_rerank_url else None
        )

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

        # 0. Parse temporal and historical intent. Explicit history queries include old versions.
        include_historical = include_historical or _is_history_query(query)
        is_temporal, query_date, window_days = parse_temporal_query(query)
        is_global = _is_global_query(query)
        is_ordering = _is_ordering_query(query)
        if is_global:
            # Top-k precision starves summarization (the LLM sees ~10% of the
            # story); widen the serving budget and let breadth win.
            limit = max(limit, self._config.global_query_limit)

        # 1. Generate query embeddings (both dimensions)
        t_emb = time.time()
        query_embedding = self._embedder.embed_query(query)
        emb_full = query_embedding.get(self._config.embedding_full_dim)
        emb_small = query_embedding.get(self._config.embedding_small_dim)
        logger.debug("Embedding: %.3fs", time.time() - t_emb)

        if not emb_full or not emb_small:
            logger.error("Embedding failed for query: %s", query[:50])
            return []

        candidate_limit = max(limit * 3, 30)

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
            bridge_ttl_hours=self._config.immediate_recall_ttl_hours,
        )
        logger.debug("Immediate recall: %d results in %.3fs", len(immediate_results), time.time() - t_imm)

        # 4b. Raw-evidence recall (v3 dual-tier): verbatim messages across ALL
        # sessions, processed included. Extraction compresses; verbatim turns
        # carry the specifics that information-extraction, instruction- and
        # preference-following questions need.
        evidence_results: List[Dict[str, Any]] = []
        if self._config.enable_raw_evidence:
            t_ev = time.time()
            evidence_results = self._qdrant.search_raw_evidence(
                query_256d=emb_small,
                owner_id=owner_id,
                agent_id=agent_id,
                limit=self._config.raw_evidence_limit,
            )
            logger.debug("Raw evidence: %d results in %.3fs", len(evidence_results), time.time() - t_ev)

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

        evidence_scores: Dict[str, float] = {}
        for msg in evidence_results:
            msg_id = msg.get("memory_id", "")
            if not msg_id:
                continue
            evidence_scores[msg_id] = msg.get("score", 0.0)
            if msg_id not in all_memories:
                pseudo = Memory(
                    memory_id=msg_id,
                    content=msg.get("content", ""),
                    owner_id=owner_id,
                    session_id=msg.get("session_id"),
                )
                created = _parse_iso(msg.get("created_at"))
                if created:
                    pseudo.created_at = created
                all_memories[msg_id] = pseudo

        if not all_memories:
            return []

        # 7. RRF Fusion (ported verbatim from provider.py:1996-2062)
        k = self._config.rrf_k  # 10

        # Build rank maps
        vector_rank = {m.memory_id: i + 1 for i, (m, _) in enumerate(filtered_vector)}
        bm25_rank = {m.memory_id: i + 1 for i, (m, _) in enumerate(bm25_results)}
        immediate_rank = {msg["memory_id"]: i + 1 for i, msg in enumerate(immediate_results) if msg.get("memory_id")}
        evidence_rank = {msg["memory_id"]: i + 1 for i, msg in enumerate(evidence_results) if msg.get("memory_id")}

        _now = now_utc()
        scored_memories = []

        for memory_id, memory in all_memories.items():
            # RRF score from vector (normalized to 0-1 range)
            v_rank = vector_rank.get(memory_id, candidate_limit + 1)
            rrf_vector = (1.0 / (k + v_rank)) * (k + 1)

            # RRF score from BM25 (normalized to 0-1 range)
            b_rank = bm25_rank.get(memory_id, candidate_limit + 1)
            rrf_bm25 = (1.0 / (k + b_rank)) * (k + 1)

            # Immediate recall / raw evidence bonus (if not already in vector results)
            immediate_bonus = 0.0
            if memory_id in immediate_rank and memory_id not in vector_rank:
                i_rank = immediate_rank[memory_id]
                immediate_bonus = (1.0 / (k + i_rank)) * (k + 1) * self._config.vector_weight
            if memory_id in evidence_rank and memory_id not in vector_rank:
                e_rank = evidence_rank[memory_id]
                immediate_bonus = max(
                    immediate_bonus,
                    (1.0 / (k + e_rank)) * (k + 1) * self._config.vector_weight,
                )

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

        # 8. Sort by RRF-derived ranking score, then optionally rerank its candidate pool.
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        rerank_scores: Dict[str, float] = {}
        if self._reranker:
            rerank_candidates = scored_memories[:candidate_limit]
            scores = self._reranker.rerank(query, [m.content for m, _, _, _ in rerank_candidates])
            if scores is not None:
                rerank_scores = {m.memory_id: score for (m, _, _, _), score in zip(rerank_candidates, scores)}
                scored_memories = sorted(
                    rerank_candidates,
                    key=lambda item: rerank_scores[item[0].memory_id],
                    reverse=True,
                )
                if self._config.enable_relevance_gate and scored_memories:
                    # Noise-floor + relative gate: absolute rerank scores do not
                    # transfer across corpora (oblique queries score relevant docs
                    # ~1e-4 while true noise sits ~1e-5), so gate on the *shape*
                    # of the score distribution instead of a fixed cutoff.
                    top_score = rerank_scores[scored_memories[0][0].memory_id]
                    if top_score < self._config.relevance_gate_noise_floor:
                        scored_memories = []  # nothing plausibly relevant
                    elif not is_global:
                        # Global queries keep breadth; only noise-floor applies.
                        keep_at = top_score * self._config.relevance_gate_relative_frac
                        kept = [
                            item for item in scored_memories
                            if rerank_scores[item[0].memory_id] >= keep_at
                        ]
                        # Min-keep floor: recall-style queries score the best
                        # memory 10-100x above supporting evidence; the relative
                        # gate alone trims multi-evidence contexts to 1-3 items
                        # and collapses recall. scored_memories is already in
                        # rerank order, so keep the top min_keep.
                        min_keep = self._config.relevance_gate_min_keep
                        if len(kept) < min_keep:
                            kept = scored_memories[:min_keep]
                        scored_memories = kept

        # 9. Order for serving. With cross-encoder scores, keep pure rerank
        # order so facts and verbatim evidence interleave by relevance.
        # Without one, prefer extracted memories over raw messages.
        raw_ids = set(immediate_scores) | set(evidence_scores)
        if rerank_scores:
            ordered = scored_memories
        else:
            extracted = [(m, r, t, d) for m, r, t, d in scored_memories if m.memory_id not in raw_ids or m.memory_id in vector_rank]
            raw_msgs = [(m, r, t, d) for m, r, t, d in scored_memories if m.memory_id in raw_ids and m.memory_id not in vector_rank]
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
        if is_global or is_ordering:
            # Chronological presentation: summaries and event-ordering answers
            # reconstruct a storyline; similarity order actively hurts both.
            def _mem_time(item):
                mem = item[0]
                stamp = getattr(mem, "event_time", None) or mem.created_at
                if stamp is None:
                    return datetime.max.replace(tzinfo=timezone.utc)
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=timezone.utc)
                return stamp
            top.sort(key=_mem_time)

        logger.debug(
            "Search complete: %d results in %.3fs (vec=%d, bm25=%d, imm=%d)",
            len(top), time.time() - t_start,
            len(vector_results), len(bm25_results), len(immediate_results),
        )

        serialized = []
        for mem, score in top:
            result = {**mem.to_dict(), "score": round(score, 4)}
            if mem.memory_id in rerank_scores:
                result["rerank_score"] = rerank_scores[mem.memory_id]
            if mem.memory_id in evidence_scores and mem.memory_id not in vector_rank:
                result["type"] = "raw_evidence"
                if self._config.raw_evidence_window:
                    window = self._sqlite.get_message_window(
                        mem.memory_id, owner_id, self._config.raw_evidence_window,
                    )
                    if window:
                        result["session_snippet"] = window
            if self._config.enable_chain_attachment and self._config.chain_max_ancestors:
                history = self._sqlite.get_ancestor_chain(
                    mem.memory_id, owner_id, self._config.chain_max_ancestors,
                )
                if history:
                    result["history"] = [
                        {
                            "content": ancestor.content,
                            "version": ancestor.version,
                            "valid_from": ancestor.valid_from.isoformat() if ancestor.valid_from else None,
                            "valid_until": ancestor.valid_until.isoformat() if ancestor.valid_until else None,
                        }
                        for ancestor in history
                    ]
            if self._config.enable_conflict_attachment and mem.memory_id not in raw_ids:
                conflicts = self._sqlite.get_conflicts(mem.memory_id, owner_id)
                if conflicts:
                    result["conflicts"] = [
                        {
                            "memory_id": conflict.memory_id,
                            "content": conflict.content,
                            "created_at": conflict.created_at.isoformat() if conflict.created_at else None,
                        }
                        for conflict in conflicts
                    ]
            serialized.append(result)
        return serialized
