"""
Memory data model for Cognis.

Simplified from production cognis — same ID scheme (mem_xxx),
same versioning, same Qdrant UUID generation.
Drops MongoDB/OpenSearch-specific serialization.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from enum import Enum

from cognis.utils import generate_memory_id, now_utc, qdrant_uuid, parse_iso_timestamp


class MemoryStatus(Enum):
    CURRENT = "current"
    HISTORICAL = "historical"
    DELETED = "deleted"


@dataclass
class MemoryMetadata:
    category: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    sector: str = "semantic"
    scope: str = "user"
    importance: float = 0.5
    confidence: float = 0.8
    source_role: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "categories": self.categories,
            "sector": self.sector,
            "scope": self.scope,
            "source_role": self.source_role,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryMetadata":
        return cls(
            category=data.get("category"),
            categories=data.get("categories", []),
            sector=data.get("sector", "semantic"),
            scope=data.get("scope", "user"),
            importance=data.get("importance", 0.5),
            confidence=data.get("confidence", 0.8),
            source_role=data.get("source_role"),
        )


@dataclass
class Memory:
    """Core Memory entity."""

    memory_id: str = field(default_factory=generate_memory_id)
    content: str = ""
    original_content: Optional[List[str]] = None
    owner_id: str = ""
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    created_at: datetime = field(default_factory=now_utc)
    updated_at: datetime = field(default_factory=now_utc)
    event_time: Optional[datetime] = None
    status: MemoryStatus = MemoryStatus.CURRENT
    is_current: bool = True
    replaces_id: Optional[str] = None
    version: int = 1
    salience_score: float = 0.5
    decay_score: float = 1.0
    access_count: int = 0
    metadata: MemoryMetadata = field(default_factory=MemoryMetadata)

    def __post_init__(self):
        if isinstance(self.original_content, str):
            self.original_content = [self.original_content]

    @property
    def qdrant_uuid_256d(self) -> str:
        return qdrant_uuid(self.memory_id, "256d")

    @property
    def qdrant_uuid_768d(self) -> str:
        return qdrant_uuid(self.memory_id, "768d")

    def to_dict(self, exclude_internal: bool = True) -> Dict[str, Any]:
        """
        Convert to dict matching hosted platform MemoryItemModel schema.

        Args:
            exclude_internal: When True, omits internal fields (status, replaces_id,
                salience_score, decay_score, original_content) — matches hosted search/get behavior.
        """
        result = {
            "id": self.memory_id,
            "memory_id": self.memory_id,
            "type": "memory",
            "content": self.content,
            "owner_id": self.owner_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "version": self.version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "event_time": self.event_time.isoformat() if self.event_time else None,
            "metadata": self.metadata.to_dict(),
        }
        if not exclude_internal:
            result["status"] = self.status.value
            result["replaces_id"] = self.replaces_id
            result["salience_score"] = self.salience_score
            result["decay_score"] = self.decay_score
            result["original_content"] = self.original_content
        return result

    def to_qdrant_payload(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "owner_id": self.owner_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "event_time": self.event_time.isoformat() if self.event_time else None,
            "is_current": self.is_current,
            "status": self.status.value,
            "replaces_id": self.replaces_id,
            "version": self.version,
            "salience_score": self.salience_score,
            "category": self.metadata.category,
            "sector": self.metadata.sector,
            "scope": self.metadata.scope,
            "original_content": self.original_content,
            "type": "memory",
            "processed": True,
        }

    def to_sqlite_row(self) -> Dict[str, Any]:
        """Flatten for SQLite storage."""
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "original_content": ",".join(self.original_content) if self.original_content else None,
            "owner_id": self.owner_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "event_time": self.event_time.isoformat() if self.event_time else None,
            "status": self.status.value,
            "is_current": 1 if self.is_current else 0,
            "replaces_id": self.replaces_id,
            "version": self.version,
            "salience_score": self.salience_score,
            "decay_score": self.decay_score,
            "access_count": self.access_count,
            "category": self.metadata.category,
            "sector": self.metadata.sector,
            "scope": self.metadata.scope,
            "importance": self.metadata.importance,
            "confidence": self.metadata.confidence,
            "source_role": self.metadata.source_role,
        }

    @classmethod
    def from_sqlite_row(cls, row: Dict[str, Any]) -> "Memory":
        status_str = row.get("status", "current")
        try:
            status = MemoryStatus(status_str)
        except ValueError:
            status = MemoryStatus.CURRENT

        original = row.get("original_content")
        if isinstance(original, str) and original:
            original = original.split(",")
        else:
            original = None

        metadata = MemoryMetadata(
            category=row.get("category"),
            sector=row.get("sector", "semantic"),
            scope=row.get("scope", "user"),
            importance=row.get("importance", 0.5),
            confidence=row.get("confidence", 0.8),
            source_role=row.get("source_role"),
        )

        return cls(
            memory_id=row.get("memory_id", ""),
            content=row.get("content", ""),
            original_content=original,
            owner_id=row.get("owner_id", ""),
            agent_id=row.get("agent_id"),
            session_id=row.get("session_id"),
            created_at=parse_iso_timestamp(row.get("created_at")),
            updated_at=parse_iso_timestamp(row.get("updated_at")),
            event_time=parse_iso_timestamp(row.get("event_time")) if row.get("event_time") else None,
            status=status,
            is_current=bool(row.get("is_current", 1)),
            replaces_id=row.get("replaces_id"),
            version=row.get("version", 1),
            salience_score=row.get("salience_score", 0.5),
            decay_score=row.get("decay_score", 1.0),
            access_count=row.get("access_count", 0),
            metadata=metadata,
        )

    @classmethod
    def from_qdrant_payload(cls, payload: Dict[str, Any], score: float = 0.0) -> "Memory":
        status_str = payload.get("status", "current")
        try:
            status = MemoryStatus(status_str)
        except ValueError:
            status = MemoryStatus.CURRENT

        metadata = MemoryMetadata(
            category=payload.get("category"),
            sector=payload.get("sector", "semantic"),
            scope=payload.get("scope", "user"),
        )

        return cls(
            memory_id=payload.get("memory_id", ""),
            content=payload.get("content", ""),
            original_content=payload.get("original_content"),
            owner_id=payload.get("owner_id", ""),
            agent_id=payload.get("agent_id"),
            session_id=payload.get("session_id"),
            created_at=parse_iso_timestamp(payload.get("created_at")),
            updated_at=parse_iso_timestamp(payload.get("updated_at")),
            event_time=parse_iso_timestamp(payload.get("event_time")) if payload.get("event_time") else None,
            is_current=payload.get("is_current", True),
            status=status,
            replaces_id=payload.get("replaces_id"),
            version=payload.get("version", 1),
            salience_score=payload.get("salience_score", score),
            metadata=metadata,
        )

    def __repr__(self) -> str:
        preview = self.content[:30] + "..." if len(self.content) > 30 else self.content
        return f'Memory(id={self.memory_id}, content="{preview}")'
