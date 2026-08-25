"""Focused lifecycle, history, and reranking tests without network dependencies."""

import json
from datetime import datetime, timezone

from cognis.config import CognisConfig
from cognis.extraction.extractor import SyncFactExtractor
from cognis.models import Memory, MemoryStatus
from cognis.rerank import RemoteReranker
from cognis.search.pipeline import HybridSearchPipeline
from cognis.stores.sqlite_store import SQLiteStore
from cognis.stores.qdrant_store import QdrantLocalStore


class StubEmbedder:
    def embed_query(self, _text):
        return {768: [0.1] * 768, 256: [0.1] * 256}

    embed_document = embed_query


class StubQdrant:
    collection_small = "small"

    def __init__(self, memories=None):
        self.memories = memories or []
        self.payload_updates = []
        self.historical_flags = []

    def search_two_stage(self, **kwargs):
        self.historical_flags.append(kwargs["include_historical"])
        return [(m, 0.9) for m in self.memories if kwargs["include_historical"] or m.is_current]

    def search(self, **_kwargs):
        return []

    def search_immediate(self, **_kwargs):
        return []

    def update_payload(self, memory_id, updates):
        self.payload_updates.append((memory_id, updates))

    def upsert(self, *_args):
        pass

    def mark_messages_processed(self, _ids):
        pass


class FixedReranker:
    def __init__(self, scores):
        self.scores = scores

    def rerank(self, _query, _documents):
        return self.scores


def make_store(tmp_path):
    store = SQLiteStore(str(tmp_path / "cognis.db"))
    store.connect()
    return store


def test_deleted_memory_is_hidden_from_listing_and_text_search(tmp_path):
    store = make_store(tmp_path)
    memory = Memory(memory_id="mem_deleted", content="orchid knowledge", owner_id="owner")
    store.store_memory(memory)

    assert store.mark_deleted(memory.memory_id, "owner")
    deleted = store.get_memory(memory.memory_id, "owner")
    assert deleted.status is MemoryStatus.DELETED
    assert deleted.valid_until is not None
    assert store.get_memories("owner", include_historical=True) == []
    assert store.text_search("orchid", "owner", include_historical=True) == []
    store.close()


def test_update_closes_old_version_and_sets_fact_event_time(tmp_path):
    store = make_store(tmp_path)
    old = Memory(memory_id="mem_old", content="User lives in Paris", owner_id="owner")
    store.store_memory(old)
    qdrant = StubQdrant()
    config = CognisConfig(embedding_full_dim=768, embedding_small_dim=256)
    extractor = SyncFactExtractor(store, qdrant, StubEmbedder(), config)
    extractor._extract_facts = lambda _content, reference_time=None: ["User moved to Berlin on January 3, 2024"]
    extractor._find_similar_memories = lambda *_args: [{"id": old.memory_id, "text": old.content}]
    extractor._decide_operations = lambda *_args: [{"id": old.memory_id, "text": "User moved to Berlin on January 3, 2024", "event": "UPDATE"}]

    created = extractor.extract_and_store("owner", session_id="session", messages=[{"role": "user", "content": "update"}])
    closed = store.get_memory(old.memory_id, "owner")
    assert closed.status is MemoryStatus.HISTORICAL
    assert not closed.is_current and closed.valid_until is not None
    assert qdrant.payload_updates[0][1]["valid_until"] == closed.valid_until.isoformat()
    assert created[0].event_time == datetime(2024, 1, 3, tzinfo=timezone.utc)
    assert created[0].valid_from is not None
    store.close()


def test_history_query_includes_historical_and_attaches_ancestors(tmp_path):
    store = make_store(tmp_path)
    old = Memory(memory_id="mem_old", content="User lived in Paris", owner_id="owner")
    store.store_memory(old)
    store.mark_historical(old.memory_id, "owner")
    current = Memory(memory_id="mem_new", content="User lives in Berlin", owner_id="owner", replaces_id=old.memory_id, version=2)
    store.store_memory(current)
    qdrant = StubQdrant([old, current])
    config = CognisConfig(embedding_full_dim=768, embedding_small_dim=256, chain_max_ancestors=2)
    pipeline = HybridSearchPipeline(qdrant, store, StubEmbedder(), config)

    results = pipeline.search("What is the history of where the user lived?", "owner", limit=5)
    assert qdrant.historical_flags == [True]
    current_result = next(result for result in results if result["memory_id"] == "mem_new")
    old_version = store.get_memory("mem_old", "owner")
    assert current_result["history"][0] == {
        "content": "User lived in Paris",
        "version": 1,
        "valid_from": old_version.valid_from.isoformat(),
        "valid_until": old_version.valid_until.isoformat(),
    }
    assert current_result["status"] == "current"
    assert current_result["replaces_id"] == "mem_old"
    assert "valid_from" in current_result and "valid_until" in current_result
    store.close()


def test_remote_relevance_gate_trims_relative_tail(tmp_path):
    store = make_store(tmp_path)
    first = Memory(memory_id="mem_first", content="relevant fact", owner_id="owner")
    second = Memory(memory_id="mem_second", content="irrelevant fact", owner_id="owner")
    qdrant = StubQdrant([first, second])
    config = CognisConfig(embedding_full_dim=768, embedding_small_dim=256, rerank_provider="remote", remote_rerank_url="https://unused", enable_relevance_gate=True, relevance_gate_relative_frac=0.05, relevance_gate_min_keep=1)
    pipeline = HybridSearchPipeline(qdrant, store, StubEmbedder(), config)
    pipeline._reranker = FixedReranker([0.9, 0.02])

    results = pipeline.search("fact", "owner", limit=5)
    assert [result["memory_id"] for result in results] == ["mem_first"]
    assert results[0]["rerank_score"] == 0.9
    store.close()


def test_remote_relevance_gate_min_keep_preserves_supporting_evidence(tmp_path):
    store = make_store(tmp_path)
    memories = [
        Memory(memory_id=f"mem_{i}", content=f"fact {i}", owner_id="owner")
        for i in range(5)
    ]
    qdrant = StubQdrant(memories)
    config = CognisConfig(embedding_full_dim=768, embedding_small_dim=256, rerank_provider="remote", remote_rerank_url="https://unused", enable_relevance_gate=True, relevance_gate_min_keep=4)
    pipeline = HybridSearchPipeline(qdrant, store, StubEmbedder(), config)
    # Recall regime: top scores high, supporting evidence 10-100x lower; the
    # relative gate alone would keep only mem_0.
    pipeline._reranker = FixedReranker([0.9, 0.01, 0.008, 0.005, 0.003])

    results = pipeline.search("fact", "owner", limit=5)
    assert [result["memory_id"] for result in results] == ["mem_0", "mem_1", "mem_2", "mem_3"]
    store.close()


def test_remote_relevance_gate_keeps_oblique_low_scores_above_noise_floor(tmp_path):
    store = make_store(tmp_path)
    first = Memory(memory_id="mem_first", content="relevant fact", owner_id="owner")
    second = Memory(memory_id="mem_second", content="also relevant", owner_id="owner")
    qdrant = StubQdrant([first, second])
    config = CognisConfig(embedding_full_dim=768, embedding_small_dim=256, rerank_provider="remote", remote_rerank_url="https://unused", enable_relevance_gate=True)
    pipeline = HybridSearchPipeline(qdrant, store, StubEmbedder(), config)
    # Oblique query: relevant docs score ~1e-4, far below any absolute cutoff.
    pipeline._reranker = FixedReranker([3e-4, 1e-4])

    results = pipeline.search("fact", "owner", limit=5)
    assert [result["memory_id"] for result in results] == ["mem_first", "mem_second"]
    store.close()


def test_remote_relevance_gate_drops_all_below_noise_floor(tmp_path):
    store = make_store(tmp_path)
    first = Memory(memory_id="mem_first", content="noise", owner_id="owner")
    qdrant = StubQdrant([first])
    config = CognisConfig(embedding_full_dim=768, embedding_small_dim=256, rerank_provider="remote", remote_rerank_url="https://unused", enable_relevance_gate=True)
    pipeline = HybridSearchPipeline(qdrant, store, StubEmbedder(), config)
    pipeline._reranker = FixedReranker([1e-5])

    assert pipeline.search("fact", "owner", limit=5) == []
    store.close()


def test_immediate_recall_requires_a_session():
    store = object.__new__(QdrantLocalStore)
    assert store.search_immediate([0.1], owner_id="owner", session_id=None) == []


def test_remote_reranker_posts_bge_payload_to_rerank_endpoint(monkeypatch):
    captured = {}

    class Response:
        def read(self):
            return b'{"results": [{"index": 1, "relevance_score": 0.2}, {"index": 0, "score": 0.8}]}'

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = request.data
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("cognis.rerank.urlopen", fake_urlopen)
    reranker = RemoteReranker("https://rerank.example")

    assert reranker.rerank("question", ["first", "second"]) == [0.8, 0.2]
    assert captured["url"] == "https://rerank.example/rerank"
    # Dual schema: local reranker_server reads "documents", HF TEI reads "texts".
    assert json.loads(captured["body"]) == {
        "query": "question",
        "documents": ["first", "second"],
        "texts": ["first", "second"],
    }


class _Point:
    def __init__(self, payload, score=0.9):
        self.payload = payload
        self.score = score


class _QueryResult:
    def __init__(self, points):
        self.points = points


class _ImmediateClient:
    def __init__(self, points):
        self.points = points

    def query_points(self, **_kwargs):
        return _QueryResult(self.points)


def test_qdrant_hides_deleted_and_processed_raw_messages_in_same_session():
    from cognis.utils import now_utc

    store = object.__new__(QdrantLocalStore)
    query_filter = store._build_filter("owner", include_historical=True)
    assert any(condition.key == "status" and condition.match.value == "deleted" for condition in query_filter.must_not)

    store._collection_small = "small"
    store._client = _ImmediateClient([
        _Point({
            "memory_id": "gym-message", "content": "My gym membership is at Equinox on 5th Avenue.",
            "processed": True, "created_at": now_utc().isoformat(), "status": "current",
        }),
        _Point({
            "memory_id": "cancellation-message", "content": "Forget my gym membership.",
            "processed": False, "created_at": now_utc().isoformat(), "status": "current",
        }),
        _Point({
            "memory_id": "deleted-message", "content": "must not serve", "processed": False,
            "created_at": now_utc().isoformat(), "status": "deleted",
        }),
    ])
    results = store.search_immediate([0.1], "owner", session_id="session", bridge_ttl_hours=1)
    assert [result["memory_id"] for result in results] == ["cancellation-message"]
    assert all("Equinox" not in result["content"] for result in results)
