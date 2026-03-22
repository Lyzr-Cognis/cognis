"""
Utility functions for Cognis memory system.
"""

import uuid
import hashlib
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from functools import lru_cache


def generate_memory_id() -> str:
    """Generate a user-friendly memory ID in mem_xxx format."""
    return f"mem_{uuid.uuid4().hex[:12]}"


def generate_message_id() -> str:
    """Generate a message ID in msg_xxx format."""
    return f"msg_{uuid.uuid4().hex[:12]}"


def generate_session_id() -> str:
    """Generate a session ID."""
    return f"ses_{uuid.uuid4().hex[:12]}"


def now_utc() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


def ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Ensure datetime is UTC-aware. Returns None if input is None."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def parse_iso_timestamp(value: Any) -> Optional[datetime]:
    """Parse ISO 8601 timestamp string to datetime. Returns now_utc() if unparseable."""
    if value is None:
        return now_utc()
    if isinstance(value, datetime):
        return ensure_utc(value)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return ensure_utc(dt)
        except (ValueError, TypeError):
            return now_utc()
    return now_utc()


def text_hash(text: str) -> str:
    """Generate MD5 hash of text for dedup checks."""
    return hashlib.md5(text.encode()).hexdigest()


# UUID namespace for deterministic Qdrant point IDs
_UUID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def qdrant_uuid(memory_id: str, suffix: str = "") -> str:
    """Generate deterministic UUID for Qdrant from memory_id + suffix."""
    name = f"{memory_id}:{suffix}" if suffix else memory_id
    return str(uuid.uuid5(_UUID_NAMESPACE, name))


class LRUCache:
    """Simple thread-safe LRU cache with TTL."""

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 300.0):
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._cache: Dict[str, tuple] = {}  # key -> (value, timestamp)

    def get(self, key: str) -> Optional[Any]:
        entry = self._cache.get(key)
        if entry is None:
            return None
        value, ts = entry
        if time.time() - ts > self._ttl:
            del self._cache[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        if len(self._cache) >= self._max_size:
            # Evict oldest entry
            oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
        self._cache[key] = (value, time.time())

    def clear(self) -> None:
        self._cache.clear()
