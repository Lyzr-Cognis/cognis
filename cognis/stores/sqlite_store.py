"""
SQLite document store with FTS5 BM25 search.

Replaces MongoDB (document store) + OpenSearch (BM25) from production.
Uses SQLite FTS5 for keyword search with Porter stemming.
"""

import sqlite3
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from cognis.models import Memory, MemoryMetadata, MemoryStatus
from cognis.utils import generate_message_id, now_utc, parse_iso_timestamp

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT UNIQUE NOT NULL,
    content TEXT NOT NULL,
    original_content TEXT,
    owner_id TEXT NOT NULL,
    agent_id TEXT,
    session_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    event_time TEXT,
    status TEXT DEFAULT 'current',
    is_current INTEGER DEFAULT 1,
    replaces_id TEXT,
    version INTEGER DEFAULT 1,
    salience_score REAL DEFAULT 0.5,
    decay_score REAL DEFAULT 1.0,
    access_count INTEGER DEFAULT 0,
    category TEXT,
    sector TEXT DEFAULT 'semantic',
    scope TEXT DEFAULT 'user',
    importance REAL DEFAULT 0.5,
    confidence REAL DEFAULT 0.8,
    source_role TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    memory_id UNINDEXED,
    owner_id UNINDEXED,
    agent_id UNINDEXED,
    session_id UNINDEXED,
    is_current UNINDEXED,
    content='memories',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

-- Triggers to keep FTS5 in sync
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, memory_id, owner_id, agent_id, session_id, is_current)
    VALUES (new.rowid, new.content, new.memory_id, new.owner_id, new.agent_id, new.session_id, new.is_current);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, memory_id, owner_id, agent_id, session_id, is_current)
    VALUES ('delete', old.rowid, old.content, old.memory_id, old.owner_id, old.agent_id, old.session_id, old.is_current);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, memory_id, owner_id, agent_id, session_id, is_current)
    VALUES ('delete', old.rowid, old.content, old.memory_id, old.owner_id, old.agent_id, old.session_id, old.is_current);
    INSERT INTO memories_fts(rowid, content, memory_id, owner_id, agent_id, session_id, is_current)
    VALUES (new.rowid, new.content, new.memory_id, new.owner_id, new.agent_id, new.session_id, new.is_current);
END;

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT UNIQUE,
    owner_id TEXT NOT NULL,
    agent_id TEXT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TEXT,
    processed INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_owner ON memories(owner_id);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(owner_id, session_id);
CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(owner_id, agent_id);
CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(owner_id, is_current);
CREATE INDEX IF NOT EXISTS idx_memories_replaces ON memories(replaces_id);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_messages_unprocessed ON messages(session_id, processed);
"""


class SQLiteStore:
    """SQLite-backed document store with FTS5 BM25 search."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Memory CRUD ──────────────────────────────────────────────────────

    def store_memory(self, memory: Memory) -> str:
        row = memory.to_sqlite_row()
        cols = ", ".join(row.keys())
        placeholders = ", ".join("?" for _ in row)
        self._conn.execute(
            f"INSERT OR REPLACE INTO memories ({cols}) VALUES ({placeholders})",
            list(row.values()),
        )
        self._conn.commit()
        return memory.memory_id

    def store_memories(self, memories: List[Memory]) -> List[str]:
        ids = []
        for m in memories:
            ids.append(self.store_memory(m))
        return ids

    def get_memory(self, memory_id: str, owner_id: str) -> Optional[Memory]:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE memory_id = ? AND owner_id = ?",
            (memory_id, owner_id),
        ).fetchone()
        if row is None:
            return None
        return Memory.from_sqlite_row(dict(row))

    def get_memories(
        self,
        owner_id: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        include_historical: bool = False,
    ) -> List[Memory]:
        sql = "SELECT * FROM memories WHERE owner_id = ?"
        params: list = [owner_id]

        if not include_historical:
            sql += " AND is_current = 1"
        if agent_id:
            sql += " AND agent_id = ?"
            params.append(agent_id)
        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)

        sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self._conn.execute(sql, params).fetchall()
        return [Memory.from_sqlite_row(dict(r)) for r in rows]

    def update_memory(self, memory_id: str, owner_id: str, updates: Dict[str, Any]) -> bool:
        updates["updated_at"] = now_utc().isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [memory_id, owner_id]
        result = self._conn.execute(
            f"UPDATE memories SET {set_clause} WHERE memory_id = ? AND owner_id = ?",
            values,
        )
        self._conn.commit()
        return result.rowcount > 0

    def delete_memory(self, memory_id: str, owner_id: str) -> bool:
        result = self._conn.execute(
            "DELETE FROM memories WHERE memory_id = ? AND owner_id = ?",
            (memory_id, owner_id),
        )
        self._conn.commit()
        return result.rowcount > 0

    def mark_historical(self, memory_id: str, owner_id: str) -> bool:
        return self.update_memory(memory_id, owner_id, {
            "is_current": 0,
            "status": MemoryStatus.HISTORICAL.value,
        })

    # ── BM25 Text Search (FTS5) ─────────────────────────────────────────

    def text_search(
        self,
        query: str,
        owner_id: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 10,
        include_historical: bool = False,
    ) -> List[Tuple[Memory, float]]:
        """BM25 keyword search via SQLite FTS5."""
        # Build FTS5 query — quote each word for safety
        words = [w.strip() for w in query.split() if len(w.strip()) > 1]
        if not words:
            return []
        fts_query = " OR ".join(f'"{w}"' for w in words)

        sql = """
            SELECT m.*, bm25(memories_fts) as bm25_score
            FROM memories_fts f
            JOIN memories m ON f.rowid = m.rowid
            WHERE memories_fts MATCH ?
              AND f.owner_id = ?
        """
        params: list = [fts_query, owner_id]

        if not include_historical:
            sql += " AND f.is_current = 1"
        if agent_id:
            sql += " AND f.agent_id = ?"
            params.append(agent_id)
        if session_id:
            sql += " AND f.session_id = ?"
            params.append(session_id)

        sql += " ORDER BY bm25(memories_fts) LIMIT ?"
        params.append(limit)

        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            logger.warning("FTS5 query failed for: %s", query)
            return []

        results = []
        for row in rows:
            d = dict(row)
            score = abs(d.pop("bm25_score", 0.0))  # FTS5 bm25() returns negative
            memory = Memory.from_sqlite_row(d)
            results.append((memory, score))
        return results

    # ── Message Operations ───────────────────────────────────────────────

    def store_messages(
        self,
        messages: List[Dict[str, str]],
        owner_id: str,
        session_id: str,
        agent_id: Optional[str] = None,
    ) -> List[str]:
        ids = []
        ts = now_utc().isoformat()
        for msg in messages:
            msg_id = generate_message_id()
            self._conn.execute(
                """INSERT INTO messages (message_id, owner_id, agent_id, session_id, role, content, timestamp, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (msg_id, owner_id, agent_id, session_id, msg.get("role", "user"), msg.get("content", ""), ts, ts),
            )
            ids.append(msg_id)
        self._conn.commit()
        return ids

    def get_messages(
        self,
        owner_id: str,
        session_id: str,
        limit: Optional[int] = None,
        include_processed: bool = True,
    ) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM messages WHERE owner_id = ? AND session_id = ?"
        params: list = [owner_id, session_id]
        if not include_processed:
            sql += " AND processed = 0"
        sql += " ORDER BY created_at ASC"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_unprocessed_messages(
        self,
        session_id: str,
        owner_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM messages WHERE session_id = ? AND processed = 0"
        params: list = [session_id]
        if owner_id:
            sql += " AND owner_id = ?"
            params.append(owner_id)
        sql += " ORDER BY created_at ASC"
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def mark_messages_processed(self, session_id: str) -> None:
        self._conn.execute(
            "UPDATE messages SET processed = 1 WHERE session_id = ? AND processed = 0",
            (session_id,),
        )
        self._conn.commit()

    def get_recent_messages(
        self,
        owner_id: str,
        session_id: str,
        limit: int = 30,
    ) -> List[Dict[str, str]]:
        """Get recent messages formatted for LLM context."""
        rows = self._conn.execute(
            """SELECT role, content FROM messages
               WHERE owner_id = ? AND session_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (owner_id, session_id, limit),
        ).fetchall()
        # Reverse to chronological order
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def clear(self, owner_id: str, session_id: Optional[str] = None) -> int:
        if session_id:
            r1 = self._conn.execute(
                "DELETE FROM memories WHERE owner_id = ? AND session_id = ?",
                (owner_id, session_id),
            )
            self._conn.execute(
                "DELETE FROM messages WHERE owner_id = ? AND session_id = ?",
                (owner_id, session_id),
            )
        else:
            r1 = self._conn.execute("DELETE FROM memories WHERE owner_id = ?", (owner_id,))
            self._conn.execute("DELETE FROM messages WHERE owner_id = ?", (owner_id,))
        self._conn.commit()
        return r1.rowcount

    def count_memories(self, owner_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM memories WHERE owner_id = ? AND is_current = 1",
            (owner_id,),
        ).fetchone()
        return row["cnt"] if row else 0
