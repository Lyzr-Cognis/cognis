"""
Cognis — Lightweight, local-first memory for LLM agents.

Session management matches the hosted version:
- owner_id, agent_id, session_id — at least one required
- Extracted memories are global to (owner_id, agent_id)
- Raw messages are scoped to (owner_id, agent_id, session_id)
- Search returns global memories + session-scoped messages
- get_context reads short-term from session, long-term globally

Usage:
    from cognis import Cognis

    memory = Cognis(gemini_api_key="...", owner_id="user_123")
    memory.add([{"role": "user", "content": "My name is Alice"}])
    results = memory.search("What is my name?")
    context = memory.get_context([{"role": "user", "content": "Hi"}])
    memory.close()
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from cognis.config import CognisConfig
from cognis.models import Memory
from cognis.stores.sqlite_store import SQLiteStore
from cognis.stores.qdrant_store import QdrantLocalStore
from cognis.embeddings.gemini import GeminiEmbedder
from cognis.search.pipeline import HybridSearchPipeline
from cognis.extraction.extractor import SyncFactExtractor
from cognis.utils import generate_session_id, now_utc

logger = logging.getLogger(__name__)


def _require_at_least_one(**kwargs):
    """Validate at least one of the given kwargs is non-None."""
    if not any(v for v in kwargs.values()):
        names = ", ".join(kwargs.keys())
        raise ValueError(f"At least one of ({names}) is required")


class Cognis:
    """
    Lightweight, local-first memory for LLM agents.

    Session model (matches hosted version):
    - owner_id: Identifies the user who owns the memories
    - agent_id: Identifies which agent context the memories belong to
    - session_id: Identifies the conversation session

    At least one of (owner_id, agent_id, session_id) must be provided.

    Scoping:
    - Extracted memories: scoped to (owner_id, agent_id) — persist across sessions
    - Raw messages: scoped to (owner_id, agent_id, session_id) — session-local
    - Search: returns global memories + current session messages
    """

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        data_dir: str = "~/.cognis",
        owner_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        config: Optional[CognisConfig] = None,
    ):
        _require_at_least_one(owner_id=owner_id, agent_id=agent_id, session_id=session_id)

        self._config = config or CognisConfig()
        self._owner_id = owner_id
        self._agent_id = agent_id
        self._session_id = session_id or generate_session_id()

        # Resolve data directory
        data_path = Path(data_dir).expanduser()
        data_path.mkdir(parents=True, exist_ok=True)

        # Initialize stores
        self._sqlite = SQLiteStore(str(data_path / "cognis.db"))
        self._sqlite.connect()

        self._qdrant = QdrantLocalStore(
            path=str(data_path / "qdrant"),
            collection_full=self._config.qdrant_collection_full,
            collection_small=self._config.qdrant_collection_small,
            full_dim=self._config.embedding_full_dim,
            small_dim=self._config.embedding_small_dim,
        )
        self._qdrant.connect()

        # Initialize embedder
        api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self._embedder = GeminiEmbedder(
            api_key=api_key,
            model=self._config.embedding_model,
            full_dim=self._config.embedding_full_dim,
            small_dim=self._config.embedding_small_dim,
        )

        # Initialize search pipeline
        self._search_pipeline = HybridSearchPipeline(
            qdrant_store=self._qdrant,
            sqlite_store=self._sqlite,
            embedder=self._embedder,
            config=self._config,
        )

        # Initialize extractor
        self._extractor = SyncFactExtractor(
            sqlite_store=self._sqlite,
            qdrant_store=self._qdrant,
            embedder=self._embedder,
            config=self._config,
            api_key=api_key,
        )

        logger.info("Cognis initialized (data_dir=%s, owner=%s, agent=%s, session=%s)",
                     data_path, self._owner_id, self._agent_id, self._session_id)

    # ── Core API ─────────────────────────────────────────────────────────

    def add(
        self,
        messages: List[Dict[str, str]],
        owner_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Add messages and extract memories.

        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
            owner_id: Override owner (defaults to instance owner_id)
            agent_id: Override agent (defaults to instance agent_id)
            session_id: Override session (defaults to instance session_id)
        """
        oid = owner_id or self._owner_id
        aid = agent_id or self._agent_id
        sid = session_id or self._session_id
        _require_at_least_one(owner_id=oid, agent_id=aid, session_id=sid)

        # Store raw messages in SQLite
        msg_ids = self._sqlite.store_messages(
            messages=messages,
            owner_id=oid,
            session_id=sid,
            agent_id=aid,
        )

        # Store in Qdrant immediate recall (256D, session-scoped)
        if self._config.enable_immediate_recall:
            contents = [m.get("content", "") for m in messages]
            roles = [m.get("role", "user") for m in messages]
            embeddings = []
            for c in contents:
                if c.strip():
                    emb = self._embedder.embed_document(c)
                    emb_small = emb.get(self._config.embedding_small_dim)
                    embeddings.append(emb_small if emb_small else [0.0] * self._config.embedding_small_dim)
                else:
                    embeddings.append([0.0] * self._config.embedding_small_dim)

            self._qdrant.upsert_immediate_messages_batch(
                message_ids=msg_ids,
                contents=contents,
                embeddings_256d=embeddings,
                owner_id=oid,
                session_id=sid,
                agent_id=aid,
                roles=roles,
            )

        # Extract facts synchronously (memories are global to owner+agent)
        new_memories = self._extractor.extract_and_store(
            owner_id=oid,
            session_id=sid,
            agent_id=aid,
            messages=messages,
        )

        return {
            "success": True,
            "message": f"Extracted {len(new_memories)} memories from {len(messages)} messages",
            "session_message_count": len(msg_ids),
            "memories": [m.to_dict() for m in new_memories],
        }

    def search(
        self,
        query: str,
        limit: int = 10,
        owner_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search memories using hybrid RRF pipeline.

        Returns dict matching hosted SearchResponse:
        {"success": True, "results": [...], "count": N, "query": "..."}
        """
        oid = owner_id or self._owner_id
        aid = agent_id or self._agent_id
        sid = session_id or self._session_id

        results = self._search_pipeline.search(
            query=query,
            owner_id=oid,
            agent_id=aid,
            session_id=sid,
            limit=limit,
        )
        return {
            "success": True,
            "results": results,
            "count": len(results),
            "query": query,
        }

    def get(self, memory_id: str, owner_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a specific memory by ID.

        Returns dict matching hosted MemoryDetailResponse:
        {"success": True, "memory": {...}} or {"success": False, "memory": None}
        """
        oid = owner_id or self._owner_id
        mem = self._sqlite.get_memory(memory_id, oid)
        if mem:
            return {"success": True, "memory": mem.to_dict()}
        return {"success": False, "memory": None}

    def get_all(
        self,
        limit: int = 100,
        offset: int = 0,
        owner_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        include_historical: bool = False,
    ) -> Dict[str, Any]:
        """
        Get all memories for the owner+agent.

        Returns dict matching hosted MemoriesListResponse:
        {"success": True, "memories": [...], "total": N, "limit": N, "offset": N}
        """
        oid = owner_id or self._owner_id
        aid = agent_id or self._agent_id
        memories = self._sqlite.get_memories(
            owner_id=oid,
            agent_id=aid,
            limit=limit,
            offset=offset,
            include_historical=include_historical,
        )
        total = self._sqlite.count_memories(oid)
        return {
            "success": True,
            "memories": [m.to_dict() for m in memories],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    def delete(self, memory_id: str, owner_id: Optional[str] = None) -> Dict[str, Any]:
        """Delete a specific memory. Returns {"success": True/False, "message": "..."}."""
        oid = owner_id or self._owner_id
        deleted = self._sqlite.delete_memory(memory_id, oid)
        if deleted:
            self._qdrant.delete(memory_id)
            return {"success": True, "message": f"Memory {memory_id} deleted"}
        return {"success": False, "message": f"Memory {memory_id} not found"}

    def get_context(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        max_short_term: int = 30,
        include_long_term: bool = True,
        owner_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get conversation context (short-term messages + long-term memories).

        Short-term: recent messages from the session (session-scoped).
        Long-term: semantic search across all memories (owner+agent global).
        """
        oid = owner_id or self._owner_id
        aid = agent_id or self._agent_id
        sid = session_id or self._session_id

        # Short-term: recent messages from this session
        short_term = self._sqlite.get_recent_messages(
            owner_id=oid,
            session_id=sid,
            limit=max_short_term,
        )

        # Long-term: semantic search on last user message (global)
        long_term = []
        if include_long_term:
            query = None
            if messages:
                for m in reversed(messages):
                    if m.get("role") == "user" and m.get("content", "").strip():
                        query = m["content"]
                        break
            if not query and short_term:
                for m in reversed(short_term):
                    if m.get("role") == "user" and m.get("content", "").strip():
                        query = m["content"]
                        break

            if query:
                long_term = self._search_pipeline.search(
                    query=query,
                    owner_id=oid,
                    agent_id=aid,
                    session_id=sid,
                    limit=10,
                )

        lt_context = ""
        if long_term:
            facts = [m["content"] for m in long_term if m.get("content")]
            if facts:
                lt_context = "Relevant memories:\n" + "\n".join(f"- {f}" for f in facts)

        return {
            "short_term": short_term,
            "long_term": long_term,
            "short_term_count": len(short_term),
            "long_term_count": len(long_term),
            "context_string": lt_context,
        }

    def clear(
        self,
        owner_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Clear memories for owner (optionally scoped to session)."""
        oid = owner_id or self._owner_id
        count = self._sqlite.clear(oid, session_id)
        self._qdrant.clear(oid, session_id)
        return {"success": True, "message": f"Deleted {count} memories"}

    def count(self, owner_id: Optional[str] = None) -> int:
        """Count current memories for this owner."""
        return self._sqlite.count_memories(owner_id or self._owner_id)

    # ── Session Management ───────────────────────────────────────────────

    def set_session(self, session_id: str) -> None:
        self._session_id = session_id

    def set_owner(self, owner_id: str) -> None:
        self._owner_id = owner_id

    def set_agent(self, agent_id: str) -> None:
        self._agent_id = agent_id

    def new_session(self) -> str:
        self._session_id = generate_session_id()
        return self._session_id

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def owner_id(self) -> Optional[str]:
        return self._owner_id

    @property
    def agent_id(self) -> Optional[str]:
        return self._agent_id

    # ── Lifecycle ────────────────────────────────────────────────────────

    def close(self) -> None:
        self._sqlite.close()
        self._qdrant.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        count = self._sqlite.count_memories(self._owner_id)
        return f"Cognis(owner={self._owner_id}, agent={self._agent_id}, session={self._session_id}, memories={count})"
