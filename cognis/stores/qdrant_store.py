"""
Qdrant local-mode vector store for Cognis.

Uses QdrantClient(path=...) for in-process, file-backed storage.
Manages two collections:
- memories_full (768D): Accurate retrieval
- memories_small (256D): Fast shortlisting + immediate recall messages

Two-stage Matryoshka search ported from production provider.py:1635-1726.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple

from cognis.models import Memory
from cognis.utils import qdrant_uuid

logger = logging.getLogger(__name__)


class QdrantLocalStore:
    """Qdrant local-mode vector store with two-stage Matryoshka search."""

    def __init__(
        self,
        path: str,
        collection_full: str = "memories_full",
        collection_small: str = "memories_small",
        full_dim: int = 768,
        small_dim: int = 256,
    ):
        self._path = path
        self._collection_full = collection_full
        self._collection_small = collection_small
        self._full_dim = full_dim
        self._small_dim = small_dim
        self._client = None

    @property
    def collection_full(self) -> str:
        return self._collection_full

    @property
    def collection_small(self) -> str:
        return self._collection_small

    def connect(self) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import VectorParams, Distance

        self._client = QdrantClient(path=self._path)

        for name, dim in [
            (self._collection_full, self._full_dim),
            (self._collection_small, self._small_dim),
        ]:
            if not self._client.collection_exists(name):
                self._client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
                logger.info("Created collection %s (dim=%d)", name, dim)
            self._ensure_indexes(name)

    def _ensure_indexes(self, collection: str) -> None:
        from qdrant_client.models import PayloadSchemaType

        keyword_fields = [
            "owner_id", "memory_id", "session_id", "agent_id",
            "status", "category", "type",
        ]
        for f in keyword_fields:
            try:
                self._client.create_payload_index(
                    collection_name=collection,
                    field_name=f,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass
        for f in ["is_current", "processed"]:
            try:
                self._client.create_payload_index(
                    collection_name=collection,
                    field_name=f,
                    field_schema=PayloadSchemaType.BOOL,
                )
            except Exception:
                pass

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    # ── Upsert ───────────────────────────────────────────────────────────

    def upsert(
        self,
        memory: Memory,
        embedding_full: List[float],
        embedding_small: List[float],
    ) -> None:
        from qdrant_client.models import PointStruct

        payload = memory.to_qdrant_payload()

        # 768D collection
        self._client.upsert(
            collection_name=self._collection_full,
            points=[PointStruct(
                id=memory.qdrant_uuid_768d,
                vector=embedding_full,
                payload=payload,
            )],
        )

        # 256D collection
        self._client.upsert(
            collection_name=self._collection_small,
            points=[PointStruct(
                id=memory.qdrant_uuid_256d,
                vector=embedding_small,
                payload=payload,
            )],
        )

    def upsert_batch(
        self,
        memories: List[Memory],
        embeddings_full: List[List[float]],
        embeddings_small: List[List[float]],
    ) -> None:
        from qdrant_client.models import PointStruct

        points_full = []
        points_small = []
        for mem, emb_f, emb_s in zip(memories, embeddings_full, embeddings_small):
            payload = mem.to_qdrant_payload()
            points_full.append(PointStruct(id=mem.qdrant_uuid_768d, vector=emb_f, payload=payload))
            points_small.append(PointStruct(id=mem.qdrant_uuid_256d, vector=emb_s, payload=payload))

        if points_full:
            self._client.upsert(collection_name=self._collection_full, points=points_full)
        if points_small:
            self._client.upsert(collection_name=self._collection_small, points=points_small)

    # ── Immediate Recall (raw messages in 256D) ─────────────────────────

    def upsert_immediate_message(
        self,
        message_id: str,
        content: str,
        embedding_256d: List[float],
        owner_id: str,
        session_id: str,
        agent_id: Optional[str] = None,
        role: str = "user",
    ) -> None:
        from qdrant_client.models import PointStruct

        point_id = qdrant_uuid(message_id, "256d")
        payload = {
            "memory_id": message_id,
            "content": content,
            "owner_id": owner_id,
            "session_id": session_id,
            "agent_id": agent_id,
            "type": "message",
            "role": role,
            "processed": False,
            "is_current": True,
        }
        self._client.upsert(
            collection_name=self._collection_small,
            points=[PointStruct(id=point_id, vector=embedding_256d, payload=payload)],
        )

    def upsert_immediate_messages_batch(
        self,
        message_ids: List[str],
        contents: List[str],
        embeddings_256d: List[List[float]],
        owner_id: str,
        session_id: str,
        agent_id: Optional[str] = None,
        roles: Optional[List[str]] = None,
    ) -> None:
        from qdrant_client.models import PointStruct

        if roles is None:
            roles = ["user"] * len(message_ids)

        points = []
        for msg_id, content, emb, role in zip(message_ids, contents, embeddings_256d, roles):
            point_id = qdrant_uuid(msg_id, "256d")
            payload = {
                "memory_id": msg_id,
                "content": content,
                "owner_id": owner_id,
                "session_id": session_id,
                "agent_id": agent_id,
                "type": "message",
                "role": role,
                "processed": False,
                "is_current": True,
            }
            points.append(PointStruct(id=point_id, vector=emb, payload=payload))

        if points:
            self._client.upsert(collection_name=self._collection_small, points=points)

    # ── Search ───────────────────────────────────────────────────────────

    def _build_filter(
        self,
        owner_id: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        include_historical: bool = False,
        exclude_messages: bool = True,
        only_messages: bool = False,
    ):
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        conditions = [FieldCondition(key="owner_id", match=MatchValue(value=owner_id))]

        if not include_historical:
            conditions.append(FieldCondition(key="is_current", match=MatchValue(value=True)))

        if agent_id:
            conditions.append(FieldCondition(key="agent_id", match=MatchValue(value=agent_id)))

        if session_id:
            conditions.append(FieldCondition(key="session_id", match=MatchValue(value=session_id)))

        if exclude_messages:
            conditions.append(FieldCondition(key="type", match=MatchValue(value="memory")))

        if only_messages:
            conditions.append(FieldCondition(key="type", match=MatchValue(value="message")))

        return Filter(must=conditions)

    def search(
        self,
        query_vector: List[float],
        owner_id: str,
        collection: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 10,
        include_historical: bool = False,
        global_memory: bool = False,
    ) -> List[Tuple[Memory, float]]:
        """Search a single collection."""
        qf = self._build_filter(
            owner_id=owner_id,
            agent_id=agent_id,
            session_id=session_id if not global_memory else None,
            include_historical=include_historical,
            exclude_messages=(collection == self._collection_small),
        )

        results = self._client.query_points(
            collection_name=collection,
            query=query_vector,
            query_filter=qf,
            limit=limit,
            with_payload=True,
        )

        points = results.points if hasattr(results, "points") else []
        return [
            (Memory.from_qdrant_payload(p.payload, p.score), p.score)
            for p in points
        ]

    def search_two_stage(
        self,
        query_full: List[float],
        query_small: List[float],
        owner_id: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        shortlist_limit: int = 200,
        limit: int = 30,
        include_historical: bool = False,
        global_memory: bool = False,
    ) -> List[Tuple[Memory, float]]:
        """
        Two-stage Matryoshka search (ported from provider.py:1635-1726):
        1. Fast 256D shortlisting
        2. Accurate 768D re-ranking on shortlisted IDs
        """
        from qdrant_client.models import Filter, FieldCondition, MatchAny

        # Stage 1: Fast 256D shortlisting
        t0 = time.time()
        shortlist = self.search(
            query_vector=query_small,
            owner_id=owner_id,
            collection=self._collection_small,
            agent_id=agent_id,
            session_id=session_id if not global_memory else None,
            limit=shortlist_limit,
            include_historical=include_historical,
            global_memory=global_memory,
        )
        logger.debug("256D shortlist: %d results in %.3fs", len(shortlist), time.time() - t0)

        if not shortlist:
            # Fallback to 768D direct search
            return self.search(
                query_vector=query_full,
                owner_id=owner_id,
                collection=self._collection_full,
                agent_id=agent_id,
                session_id=session_id if not global_memory else None,
                limit=limit,
                include_historical=include_historical,
                global_memory=global_memory,
            )

        # Stage 2: Re-rank with 768D
        shortlist_ids = [m.memory_id for m, _ in shortlist]

        t1 = time.time()
        base_filter = self._build_filter(
            owner_id=owner_id,
            agent_id=agent_id,
            session_id=session_id if not global_memory else None,
            include_historical=include_historical,
            exclude_messages=False,
        )
        # Add MatchAny filter on memory_id
        rerank_filter = Filter(must=[
            *base_filter.must,
            FieldCondition(key="memory_id", match=MatchAny(any=shortlist_ids)),
        ])

        rerank_results = self._client.query_points(
            collection_name=self._collection_full,
            query=query_full,
            query_filter=rerank_filter,
            limit=limit,
            with_payload=True,
        )

        points = rerank_results.points if hasattr(rerank_results, "points") else []
        results = []
        reranked_ids = set()
        for p in points:
            mem = Memory.from_qdrant_payload(p.payload, p.score)
            results.append((mem, p.score))
            reranked_ids.add(mem.memory_id)

        # Merge back 256D-only items not found in 768D
        for mem, score_256d in shortlist:
            if mem.memory_id not in reranked_ids:
                results.append((mem, score_256d))

        results.sort(key=lambda x: x[1], reverse=True)
        logger.debug("768D rerank: %d reranked + %d 256D-only in %.3fs",
                      len(reranked_ids), len(results) - len(reranked_ids), time.time() - t1)

        return results

    def search_immediate(
        self,
        query_256d: List[float],
        owner_id: str,
        session_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Search immediate recall (raw messages in 256D)."""
        qf = self._build_filter(
            owner_id=owner_id,
            session_id=session_id,
            exclude_messages=False,
            only_messages=True,
        )

        results = self._client.query_points(
            collection_name=self._collection_small,
            query=query_256d,
            query_filter=qf,
            limit=limit,
            with_payload=True,
        )

        points = results.points if hasattr(results, "points") else []
        return [
            {
                "memory_id": p.payload.get("memory_id", ""),
                "content": p.payload.get("content", ""),
                "score": p.score,
                "role": p.payload.get("role", "user"),
            }
            for p in points
        ]

    # ── Delete / Update ──────────────────────────────────────────────────

    def delete(self, memory_id: str) -> None:
        from qdrant_client.models import PointIdsList

        for collection in [self._collection_full, self._collection_small]:
            suffixes = ["768d", "256d"] if collection == self._collection_full else ["256d"]
            point_ids = [qdrant_uuid(memory_id, s) for s in suffixes]
            try:
                self._client.delete(
                    collection_name=collection,
                    points_selector=PointIdsList(points=point_ids),
                )
            except Exception:
                pass

    def update_payload(self, memory_id: str, updates: Dict[str, Any]) -> None:
        from qdrant_client.models import PointIdsList

        for collection, suffix in [
            (self._collection_full, "768d"),
            (self._collection_small, "256d"),
        ]:
            point_id = qdrant_uuid(memory_id, suffix)
            try:
                self._client.set_payload(
                    collection_name=collection,
                    payload=updates,
                    points=[point_id],
                )
            except Exception:
                pass

    def clear(self, owner_id: str, session_id: Optional[str] = None) -> None:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        conditions = [FieldCondition(key="owner_id", match=MatchValue(value=owner_id))]
        if session_id:
            conditions.append(FieldCondition(key="session_id", match=MatchValue(value=session_id)))

        qf = Filter(must=conditions)
        for collection in [self._collection_full, self._collection_small]:
            try:
                self._client.delete(collection_name=collection, points_selector=qf)
            except Exception:
                pass
