"""Tests for preserving incompatible current memories as linked contradictions."""

from __future__ import annotations

import pytest

from cognis.config import CognisConfig
from cognis.extraction.extractor import SyncFactExtractor
from cognis.models import Memory, MemoryStatus
from cognis.stores.sqlite_store import SQLiteStore


class _Embedder:
    def embed_document(self, content):
        return {3: [0.1, 0.2, 0.3], 2: [0.1, 0.2]}


class _QdrantRecorder:
    def __init__(self):
        self.payload_updates = {}

    def upsert(self, memory, full_embedding, small_embedding):
        pass

    def update_payload(self, memory_id, updates):
        self.payload_updates.setdefault(memory_id, {}).update(updates)

    def mark_messages_processed(self, message_ids):
        pass


@pytest.fixture
def extractor(tmp_path):
    store = SQLiteStore(str(tmp_path / "memories.db"))
    store.connect()
    qdrant = _QdrantRecorder()
    config = CognisConfig(embedding_full_dim=3, embedding_small_dim=2)
    result = SyncFactExtractor(store, qdrant, _Embedder(), config)
    yield result, store, qdrant
    store.close()


def _extract_with_operations(monkeypatch, extractor, operations):
    fact_extractor, _, _ = extractor
    monkeypatch.setattr(fact_extractor, "_extract_facts", lambda content, reference_time=None: ["ignored by mocked operations"])
    monkeypatch.setattr(fact_extractor, "_find_similar_memories", lambda content, owner_id, agent_id: [])
    monkeypatch.setattr(fact_extractor, "_decide_operations", lambda facts, similar: operations)
    return fact_extractor.extract_and_store(
        owner_id="owner", agent_id="agent", messages=[{"role": "user", "content": "New statement"}]
    )


def test_contradiction_keeps_memories_current_and_links_both_sides(monkeypatch, extractor):
    fact_extractor, store, qdrant = extractor
    old = Memory(memory_id="old-memory", content="User has a dog", owner_id="owner")
    store.store_memory(old)

    link_calls = []
    original_link_conflict = store.link_conflict

    def record_link(*args):
        link_calls.append(args)
        return original_link_conflict(*args)

    monkeypatch.setattr(store, "link_conflict", record_link)
    created = _extract_with_operations(
        monkeypatch,
        extractor,
        [{"id": old.memory_id, "text": "User does not have a dog", "event": "CONTRADICT"}],
    )

    assert len(created) == 1
    new = store.get_memory(created[0].memory_id, "owner")
    persisted_old = store.get_memory(old.memory_id, "owner")
    assert new.status is MemoryStatus.CURRENT
    assert new.is_current is True
    assert persisted_old.status is MemoryStatus.CURRENT
    assert persisted_old.is_current is True
    assert link_calls == [(new.memory_id, old.memory_id, "owner")]
    assert [memory.memory_id for memory in store.get_conflicts(new.memory_id, "owner")] == [old.memory_id]
    assert [memory.memory_id for memory in store.get_conflicts(old.memory_id, "owner")] == [new.memory_id]
    assert new.conflicts_with == [old.memory_id]
    assert persisted_old.conflicts_with == [new.memory_id]
    assert qdrant.payload_updates[new.memory_id]["conflicts_with"] == [old.memory_id]
    assert qdrant.payload_updates[old.memory_id]["conflicts_with"] == [new.memory_id]


def test_update_still_marks_replaced_memory_historical(monkeypatch, extractor):
    _, store, _ = extractor
    old = Memory(memory_id="old-memory", content="User works at Google", owner_id="owner")
    store.store_memory(old)

    created = _extract_with_operations(
        monkeypatch,
        extractor,
        [
            {"id": "new", "text": "ignored unknown event", "event": "UNRECOGNIZED"},
            {"id": old.memory_id, "text": "User works at Lyzr AI", "event": "UPDATE"},
        ],
    )

    assert len(created) == 1
    persisted_old = store.get_memory(old.memory_id, "owner")
    assert persisted_old.status is MemoryStatus.HISTORICAL
    assert persisted_old.is_current is False
    assert created[0].status is MemoryStatus.CURRENT
    assert created[0].replaces_id == old.memory_id
