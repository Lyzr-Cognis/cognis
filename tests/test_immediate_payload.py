from cognis.stores.qdrant_store import QdrantLocalStore


class _Client:
    def __init__(self):
        self.points = []

    def upsert(self, *, collection_name, points):
        self.points.extend(points)


def test_immediate_batch_points_keep_bridge_metadata():
    store = object.__new__(QdrantLocalStore)
    store._client = _Client()
    store._collection_small = "small"

    store.upsert_immediate_messages_batch(
        message_ids=["msg-1"],
        contents=["remember this"],
        embeddings_256d=[[0.1]],
        owner_id="owner",
        session_id="session",
    )

    payload = store._client.points[0].payload
    assert payload["processed"] is False
    assert payload["status"] == "current"
    assert payload["created_at"]


def test_history_intent_recognizes_change_over_time():
    from cognis.search.pipeline import _is_history_query

    assert _is_history_query("How has the preference changed over time?")
