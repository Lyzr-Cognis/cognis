"""Tests for SQLite and Qdrant local stores."""

import os
import tempfile
import pytest

from cognis.models import Memory
from cognis.stores.sqlite_store import SQLiteStore
from cognis.stores.qdrant_store import QdrantLocalStore


class TestSQLiteStore:
    @pytest.fixture
    def store(self):
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStore(os.path.join(td, "test.db"))
            s.connect()
            yield s
            s.close()

    def test_store_and_get(self, store):
        mem = Memory(content="User loves pizza", owner_id="u1")
        store.store_memory(mem)
        got = store.get_memory(mem.memory_id, "u1")
        assert got is not None
        assert got.content == "User loves pizza"

    def test_store_batch(self, store):
        mems = [
            Memory(content="Fact 1", owner_id="u1"),
            Memory(content="Fact 2", owner_id="u1"),
            Memory(content="Fact 3", owner_id="u1"),
        ]
        ids = store.store_memories(mems)
        assert len(ids) == 3
        assert store.count_memories("u1") == 3

    def test_get_nonexistent(self, store):
        assert store.get_memory("mem_nope", "u1") is None

    def test_get_memories_with_filters(self, store):
        store.store_memory(Memory(content="F1", owner_id="u1", agent_id="a1", session_id="s1"))
        store.store_memory(Memory(content="F2", owner_id="u1", agent_id="a1", session_id="s2"))
        store.store_memory(Memory(content="F3", owner_id="u2", agent_id="a1", session_id="s1"))

        u1 = store.get_memories("u1")
        assert len(u1) == 2

        s1 = store.get_memories("u1", session_id="s1")
        assert len(s1) == 1
        assert s1[0].content == "F1"

    def test_bm25_search(self, store):
        store.store_memory(Memory(content="User loves pizza", owner_id="u1"))
        store.store_memory(Memory(content="User works at Google", owner_id="u1"))
        store.store_memory(Memory(content="User hates rain", owner_id="u1"))

        results = store.text_search("pizza", "u1")
        assert len(results) >= 1
        assert results[0][0].content == "User loves pizza"
        assert results[0][1] > 0  # BM25 score

    def test_bm25_multi_word(self, store):
        store.store_memory(Memory(content="User works at Google as engineer", owner_id="u1"))
        results = store.text_search("Google engineer", "u1")
        assert len(results) >= 1

    def test_mark_historical(self, store):
        mem = Memory(content="Old fact", owner_id="u1")
        store.store_memory(mem)
        store.mark_historical(mem.memory_id, "u1")
        current = store.get_memories("u1", include_historical=False)
        assert len(current) == 0
        all_inc = store.get_memories("u1", include_historical=True)
        assert len(all_inc) == 1

    def test_delete(self, store):
        mem = Memory(content="To delete", owner_id="u1")
        store.store_memory(mem)
        assert store.delete_memory(mem.memory_id, "u1") is True
        assert store.get_memory(mem.memory_id, "u1") is None

    def test_messages(self, store):
        store.store_messages(
            [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}],
            owner_id="u1", session_id="s1",
        )
        msgs = store.get_messages("u1", "s1")
        assert len(msgs) == 2

        unproc = store.get_unprocessed_messages("s1", "u1")
        assert len(unproc) == 2

        store.mark_messages_processed("s1")
        assert len(store.get_unprocessed_messages("s1", "u1")) == 0

    def test_clear(self, store):
        store.store_memory(Memory(content="F1", owner_id="u1"))
        store.store_memory(Memory(content="F2", owner_id="u1"))
        count = store.clear("u1")
        assert count == 2
        assert store.count_memories("u1") == 0


class TestQdrantLocalStore:
    @pytest.fixture
    def store(self):
        with tempfile.TemporaryDirectory() as td:
            s = QdrantLocalStore(path=os.path.join(td, "qdrant"), full_dim=4, small_dim=2)
            s.connect()
            yield s
            s.close()

    def test_upsert_and_search(self, store):
        mem = Memory(content="User loves pizza", owner_id="u1")
        store.upsert(mem, [0.1, 0.2, 0.3, 0.4], [0.5, 0.6])
        results = store.search([0.5, 0.6], "u1", store.collection_small, limit=5)
        assert len(results) >= 1
        assert results[0][0].content == "User loves pizza"

    def test_two_stage_search(self, store):
        mem1 = Memory(content="Pizza fact", owner_id="u1")
        mem2 = Memory(content="Work fact", owner_id="u1")
        store.upsert(mem1, [0.1, 0.2, 0.3, 0.4], [0.5, 0.6])
        store.upsert(mem2, [0.4, 0.3, 0.2, 0.1], [0.7, 0.8])

        results = store.search_two_stage(
            query_full=[0.1, 0.2, 0.3, 0.4], query_small=[0.5, 0.6],
            owner_id="u1", limit=5,
        )
        assert len(results) >= 1

    def test_immediate_recall(self, store):
        store.upsert_immediate_message("msg_1", "Hello world", [0.3, 0.4], "u1", "s1")
        results = store.search_immediate([0.3, 0.4], "u1", "s1", limit=5)
        assert len(results) >= 1
        assert results[0]["content"] == "Hello world"

    def test_delete(self, store):
        mem = Memory(content="To delete", owner_id="u1")
        store.upsert(mem, [0.1, 0.2, 0.3, 0.4], [0.5, 0.6])
        store.delete(mem.memory_id)
        results = store.search([0.5, 0.6], "u1", store.collection_small, limit=5)
        # After delete, should not find the memory
        found = any(r[0].memory_id == mem.memory_id for r in results)
        assert not found

    def test_owner_isolation(self, store):
        mem1 = Memory(content="User1 fact", owner_id="u1")
        mem2 = Memory(content="User2 fact", owner_id="u2")
        store.upsert(mem1, [0.1, 0.2, 0.3, 0.4], [0.5, 0.6])
        store.upsert(mem2, [0.1, 0.2, 0.3, 0.4], [0.5, 0.6])

        u1_results = store.search([0.5, 0.6], "u1", store.collection_small, limit=5)
        assert all(r[0].owner_id == "u1" for r in u1_results)
