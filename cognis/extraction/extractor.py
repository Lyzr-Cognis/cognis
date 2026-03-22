"""
Synchronous fact extraction pipeline for Cognis.

Replaces the production Lambda/SQS async pipeline with in-process extraction.
Uses LiteLLM for LLM calls (supports any provider).

Pipeline:
1. Get unprocessed messages
2. Extract facts via LLM (USER_MEMORY_EXTRACTION_PROMPT)
3. Find similar existing memories via vector search
4. Decide operations via LLM (UPDATE_MEMORY_PROMPT): ADD/UPDATE/DELETE/NONE
5. Process operations (store new, mark old as historical)
6. Mark messages as processed
"""

import json
import re
import logging
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from cognis.config import CognisConfig
from cognis.models import Memory, MemoryMetadata
from cognis.stores.sqlite_store import SQLiteStore
from cognis.stores.qdrant_store import QdrantLocalStore
from cognis.embeddings.base import BaseEmbedder
from cognis.extraction.prompts import USER_MEMORY_EXTRACTION_PROMPT, UPDATE_MEMORY_PROMPT
from cognis.utils import generate_memory_id, now_utc

logger = logging.getLogger(__name__)


class SyncFactExtractor:
    """Synchronous fact extraction via LiteLLM."""

    def __init__(
        self,
        sqlite_store: SQLiteStore,
        qdrant_store: QdrantLocalStore,
        embedder: BaseEmbedder,
        config: CognisConfig,
        api_key: Optional[str] = None,
    ):
        self._sqlite = sqlite_store
        self._qdrant = qdrant_store
        self._embedder = embedder
        self._config = config

        # LiteLLM reads API keys from env vars automatically
        # (OPENAI_API_KEY for gpt-*, GEMINI_API_KEY for gemini/*)
        pass

    def extract_and_store(
        self,
        owner_id: str,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> List[Memory]:
        """
        Extract facts from messages and store them.

        Args:
            owner_id: Memory owner
            session_id: Current session
            agent_id: Current agent
            messages: Raw messages (if None, reads unprocessed from SQLite)

        Returns:
            List of newly created Memory objects
        """
        # Get messages
        if messages is None and session_id:
            raw = self._sqlite.get_unprocessed_messages(session_id, owner_id)
            messages = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in raw]

        if not messages:
            return []

        # Format messages for extraction
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if not content.strip():
                continue
            if role == "user":
                parts.append(f"[USER] {content}")
            else:
                parts.append(f"[ASSISTANT] {content}")
        user_content = "\n".join(parts)

        if not user_content.strip():
            if session_id:
                self._sqlite.mark_messages_processed(session_id)
            return []

        # Step 1: Extract facts via LLM
        facts = self._extract_facts(user_content)
        if not facts:
            if session_id:
                self._sqlite.mark_messages_processed(session_id)
            return []

        # Step 2: Find similar existing memories
        similar = self._find_similar_memories(user_content, owner_id, agent_id)

        # Step 3: Decide operations (ADD/UPDATE/DELETE/NONE)
        operations = self._decide_operations(facts, similar)

        # Step 4: Process operations
        new_memories = []
        for op in operations:
            event = op.get("event", "").upper()
            text = op.get("text", "")
            op_id = op.get("id", "new")

            if event == "ADD" and text.strip():
                memory = self._create_and_store(text, owner_id, session_id, agent_id)
                if memory:
                    new_memories.append(memory)

            elif event == "UPDATE" and text.strip() and op_id != "new":
                # Mark old as historical
                self._sqlite.mark_historical(op_id, owner_id)
                self._qdrant.update_payload(op_id, {"is_current": False, "status": "historical"})
                # Get old version number
                old = self._sqlite.get_memory(op_id, owner_id)
                version = (old.version + 1) if old else 1
                memory = self._create_and_store(
                    text, owner_id, session_id, agent_id,
                    replaces_id=op_id, version=version,
                )
                if memory:
                    new_memories.append(memory)

            elif event == "DELETE" and op_id != "new":
                self._sqlite.mark_historical(op_id, owner_id)
                self._qdrant.update_payload(op_id, {"is_current": False, "status": "historical"})

        # Step 5: Mark messages as processed
        if session_id:
            self._sqlite.mark_messages_processed(session_id)

        logger.info("Extracted %d new memories from %d messages", len(new_memories), len(messages))
        return new_memories

    def _llm_call(self, prompt: str) -> str:
        """Make an LLM call via LiteLLM."""
        import litellm

        response = litellm.completion(
            model=self._config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    def _extract_facts(self, content: str) -> List[str]:
        """Call LLM to extract facts from content."""
        now = datetime.now(timezone.utc)
        next_month = now.replace(month=now.month % 12 + 1) if now.month < 12 else now.replace(year=now.year + 1, month=1)

        prompt = USER_MEMORY_EXTRACTION_PROMPT.format(
            current_date=now.strftime("%Y-%m-%d"),
            user_message=content,
            next_month=next_month.strftime("%B %Y"),
        )

        try:
            text = self._llm_call(prompt)
            return self._parse_facts_json(text)
        except Exception as e:
            logger.warning("Fact extraction failed: %s", e)
            return []

    def _find_similar_memories(
        self,
        content: str,
        owner_id: str,
        agent_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Find top-K similar existing memories for dedup/update decisions."""
        try:
            emb = self._embedder.embed_query(content)
            emb_small = emb.get(self._config.embedding_small_dim)
            if not emb_small:
                return []

            results = self._qdrant.search(
                query_vector=emb_small,
                owner_id=owner_id,
                collection=self._qdrant.collection_small,
                agent_id=agent_id,
                limit=self._config.unified_operation_top_k,
            )
            return [
                {"id": mem.memory_id, "text": mem.content, "score": score}
                for mem, score in results
            ]
        except Exception as e:
            logger.warning("Similar memory search failed: %s", e)
            return []

    def _decide_operations(
        self,
        facts: List[str],
        similar: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Call LLM to decide ADD/UPDATE/DELETE/NONE for each fact."""
        if not similar:
            # No existing memories — all facts are ADDs
            return [{"id": "new", "text": f, "event": "ADD"} for f in facts]

        existing_str = json.dumps(
            [{"id": s["id"], "text": s["text"]} for s in similar],
            indent=2,
        )
        facts_str = json.dumps(facts)

        prompt = UPDATE_MEMORY_PROMPT.format(
            existing_memory=existing_str,
            new_facts=facts_str,
        )

        try:
            text = self._llm_call(prompt)
            return self._parse_operations_json(text)
        except Exception as e:
            logger.warning("Operation decision failed: %s, adding all as new", e)
            return [{"id": "new", "text": f, "event": "ADD"} for f in facts]

    def _categorize_fact(self, fact: str) -> str:
        """Categorize a fact into one of the 13 categories using simple keyword matching."""
        fact_lower = fact.lower()

        rules = [
            ("identity", ["name is", "years old", "age", "born", "gender", "nationality"]),
            ("relationships", ["friend", "brother", "sister", "mother", "father", "family", "wife", "husband", "partner", "pet", "dog", "cat"]),
            ("work_career", ["work", "job", "engineer", "company", "colleague", "career", "profession", "staff", "team", "building", "developer"]),
            ("learning", ["study", "school", "university", "degree", "course", "learn", "skill", "language", "certification"]),
            ("wellness", ["health", "doctor", "exercise", "diet", "vegetarian", "vegan", "allergy", "medical", "fitness"]),
            ("lifestyle", ["routine", "habit", "sleep", "wake", "commute", "moved", "lives in", "relocated"]),
            ("interests", ["love", "enjoy", "hobby", "play", "watch", "fan of", "cricket", "football", "music", "game", "sport"]),
            ("preferences", ["prefer", "favorite", "like", "dislike", "hate", "food", "dosa", "pizza", "cuisine"]),
            ("plans_goals", ["plan", "goal", "want to", "intend", "trip", "visit", "aim", "aspire", "dream"]),
            ("experiences", ["visited", "went to", "travelled", "achieved", "won", "completed", "milestone"]),
            ("opinions", ["think", "believe", "feel that", "opinion", "view"]),
            ("context", ["currently", "right now", "today", "this week", "working on"]),
        ]

        for category, keywords in rules:
            if any(kw in fact_lower for kw in keywords):
                return category
        return "misc"

    def _create_and_store(
        self,
        content: str,
        owner_id: str,
        session_id: Optional[str],
        agent_id: Optional[str],
        replaces_id: Optional[str] = None,
        version: int = 1,
    ) -> Optional[Memory]:
        """Create a Memory, embed it, and store in both SQLite and Qdrant."""
        try:
            emb_result = self._embedder.embed_document(content)
            emb_full = emb_result.get(self._config.embedding_full_dim)
            emb_small = emb_result.get(self._config.embedding_small_dim)
            if not emb_full or not emb_small:
                return None

            category = self._categorize_fact(content)

            memory = Memory(
                memory_id=generate_memory_id(),
                content=content,
                owner_id=owner_id,
                agent_id=agent_id,
                session_id=session_id,
                replaces_id=replaces_id,
                version=version,
                metadata=MemoryMetadata(category=category, scope="user"),
            )

            self._sqlite.store_memory(memory)
            self._qdrant.upsert(memory, emb_full, emb_small)
            return memory

        except Exception as e:
            logger.warning("Failed to create memory: %s", e)
            return None

    @staticmethod
    def _parse_facts_json(text: str) -> List[str]:
        """Parse LLM response for facts list."""
        text = text.strip()
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        # Try direct JSON parse
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "facts" in data:
                return [f for f in data["facts"] if isinstance(f, str) and f.strip()]
        except json.JSONDecodeError:
            pass
        # Try to find JSON object in text
        try:
            match = re.search(r'\{.*"facts"\s*:\s*\[.*?\]\s*\}', text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return [f for f in data.get("facts", []) if isinstance(f, str) and f.strip()]
        except (json.JSONDecodeError, AttributeError):
            pass
        return []

    @staticmethod
    def _parse_operations_json(text: str) -> List[Dict[str, Any]]:
        """Parse LLM response for memory operations."""
        text = text.strip()
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "memory" in data:
                return [op for op in data["memory"] if isinstance(op, dict)]
        except json.JSONDecodeError:
            pass
        try:
            match = re.search(r'\{.*"memory"\s*:\s*\[.*?\]\s*\}', text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return [op for op in data.get("memory", []) if isinstance(op, dict)]
        except (json.JSONDecodeError, AttributeError):
            pass
        return []
