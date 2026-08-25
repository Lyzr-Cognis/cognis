"""Focused tests for raw evidence, message windows, and conflicts."""

import os
import tempfile

from cognis.models import Memory
from cognis.stores.qdrant_store import QdrantLocalStore
from cognis.stores.sqlite_store import SQLiteStore


def test_raw_evidence_includes_processed_messages_across_sessions():
    with tempfile.TemporaryDirectory() as directory:
        store = QdrantLocalStore(path=os.path.join(directory, "qdrant"), full_dim=4, small_dim=2)
        store.connect()
        try:
            store.upsert_immediate_message("msg_first", "first", [1.0, 0.0], "owner", "session-a", "agent")
            store.upsert_immediate_message("msg_second", "second", [0.8, 0.2], "owner", "session-b", "agent")
            store.mark_messages_processed(["msg_first"])
            store.upsert_immediate_message("msg_other_agent", "other", [1.0, 0.0], "owner", "session-c", "other-agent")

            evidence = store.search_raw_evidence([1.0, 0.0], "owner", agent_id="agent", limit=5)

            assert {item["memory_id"] for item in evidence} == {"msg_first", "msg_second"}
            assert all(set(item) == {"memory_id", "content", "score", "role"} for item in evidence)
            assert evidence == sorted(evidence, key=lambda item: item["score"], reverse=True)
        finally:
            store.close()


def test_message_window_is_chronological_and_handles_missing_message():
    with tempfile.TemporaryDirectory() as directory:
        store = SQLiteStore(os.path.join(directory, "store.db"))
        store.connect()
        try:
            ids = store.store_messages(
                [
                    {"role": "user", "content": "one"},
                    {"role": "assistant", "content": "two"},
                    {"role": "user", "content": "three"},
                    {"role": "assistant", "content": "four"},
                    {"role": "user", "content": "five"},
                ],
                owner_id="owner",
                session_id="session",
            )
            store.store_messages([{"role": "user", "content": "other"}], "owner", "other-session")

            window = store.get_message_window(ids[2], "owner", radius=1)

            assert [message["message_id"] for message in window] == ids[1:4]
            assert store.get_message_window("missing", "owner", radius=1) == []
        finally:
            store.close()


def test_conflicts_survive_reopen_and_exclude_non_current_memories():
    with tempfile.TemporaryDirectory() as directory:
        db_path = os.path.join(directory, "store.db")
        store = SQLiteStore(db_path)
        store.connect()
        first = Memory(content="first", owner_id="owner")
        current = Memory(content="current", owner_id="owner")
        historical = Memory(content="historical", owner_id="owner")
        store.store_memories([first, current, historical])
        store.link_conflict(first.memory_id, current.memory_id, "owner")
        store.link_conflict(first.memory_id, historical.memory_id, "owner")
        store.mark_historical(historical.memory_id, "owner")
        store.close()

        reopened = SQLiteStore(db_path)
        reopened.connect()
        try:
            assert [memory.memory_id for memory in reopened.get_conflicts(first.memory_id, "owner")] == [current.memory_id]
            assert [memory.memory_id for memory in reopened.get_conflicts(current.memory_id, "owner")] == [first.memory_id]
        finally:
            reopened.close()


def test_memory_conflicts_with_round_trips_through_serializers():
    memory = Memory(content="fact", owner_id="owner", conflicts_with=["mem_a", "mem_b"])

    assert memory.to_dict()["conflicts_with"] == ["mem_a", "mem_b"]
    assert Memory.from_sqlite_row(memory.to_sqlite_row()).conflicts_with == ["mem_a", "mem_b"]
    assert Memory.from_qdrant_payload(memory.to_qdrant_payload()).conflicts_with == ["mem_a", "mem_b"]
