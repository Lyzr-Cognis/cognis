"""v3 dual-tier serving: raw-evidence fusion, global/ordering routes, conflict attachment."""

from datetime import datetime, timezone, timedelta

from cognis.config import CognisConfig
from cognis.models import Memory
from cognis.search.pipeline import HybridSearchPipeline

from tests.test_lifecycle import StubEmbedder, StubQdrant, FixedReranker, make_store


class EvidenceQdrant(StubQdrant):
    def __init__(self, memories=None, evidence=None):
        super().__init__(memories)
        self.evidence = evidence or []
        self.evidence_calls = []

    def search_raw_evidence(self, **kwargs):
        self.evidence_calls.append(kwargs)
        return self.evidence


def _config(**overrides):
    defaults = dict(embedding_full_dim=768, embedding_small_dim=256)
    defaults.update(overrides)
    return CognisConfig(**defaults)


def test_raw_evidence_disabled_by_default(tmp_path):
    store = make_store(tmp_path)
    qdrant = EvidenceQdrant(
        [Memory(memory_id="mem_fact", content="User works at Acme", owner_id="owner")],
        evidence=[{"memory_id": "raw_1", "content": "verbatim turn", "score": 0.8, "session_id": "s1", "created_at": None}],
    )
    pipeline = HybridSearchPipeline(qdrant, store, StubEmbedder(), _config())

    results = pipeline.search("where does the user work", "owner", limit=5)
    assert qdrant.evidence_calls == []
    assert [r["memory_id"] for r in results] == ["mem_fact"]
    store.close()


def test_raw_evidence_fused_across_sessions(tmp_path):
    store = make_store(tmp_path)
    fact = Memory(memory_id="mem_fact", content="User works at Acme", owner_id="owner")
    qdrant = EvidenceQdrant(
        [fact],
        evidence=[
            {"memory_id": "raw_1", "content": "I started at Acme as a staff engineer on the payments team", "score": 0.8, "session_id": "old_session", "created_at": "2024-01-05T10:00:00+00:00"},
        ],
    )
    config = _config(enable_raw_evidence=True, raw_evidence_window=0)
    pipeline = HybridSearchPipeline(qdrant, store, StubEmbedder(), config)

    results = pipeline.search("what role does the user have at Acme", "owner", limit=5)
    by_id = {r["memory_id"]: r for r in results}
    assert "raw_1" in by_id, "verbatim evidence from another session must be served"
    assert by_id["raw_1"]["type"] == "raw_evidence"
    assert "mem_fact" in by_id
    # searched without a session: session-scoped immediate recall alone would return nothing
    assert qdrant.evidence_calls[0]["owner_id"] == "owner"
    store.close()


def test_rerank_interleaves_evidence_above_weak_facts(tmp_path):
    store = make_store(tmp_path)
    fact = Memory(memory_id="mem_fact", content="User works at Acme", owner_id="owner")
    qdrant = EvidenceQdrant(
        [fact],
        evidence=[{"memory_id": "raw_1", "content": "detailed verbatim answer", "score": 0.9, "session_id": "s1", "created_at": None}],
    )
    config = _config(enable_raw_evidence=True, raw_evidence_window=0, rerank_provider="remote", remote_rerank_url="https://unused")
    pipeline = HybridSearchPipeline(qdrant, store, StubEmbedder(), config)
    # RRF-ordered candidates: [raw_1 (evidence bonus, no recency), mem_fact].
    # Cross-encoder scores raw_1 far above the fact; the old extracted-first
    # reorder would still serve mem_fact first — rerank order must win.
    pipeline._reranker = FixedReranker([0.95, 0.05])

    results = pipeline.search("specific detail question", "owner", limit=5)
    assert [r["memory_id"] for r in results] == ["raw_1", "mem_fact"]
    store.close()


def test_global_query_widens_limit_and_orders_chronologically(tmp_path):
    store = make_store(tmp_path)
    base = datetime(2024, 3, 1, tzinfo=timezone.utc)
    memories = [
        Memory(memory_id=f"mem_{i}", content=f"milestone {i}", owner_id="owner", created_at=base + timedelta(days=i))
        for i in (2, 0, 1)
    ]
    qdrant = EvidenceQdrant(memories)
    config = _config(global_query_limit=10)
    pipeline = HybridSearchPipeline(qdrant, store, StubEmbedder(), config)

    results = pipeline.search("give me a comprehensive summary of my project", "owner", limit=1)
    assert len(results) == 3, "global queries must not be starved by the caller limit"
    assert [r["memory_id"] for r in results] == ["mem_0", "mem_1", "mem_2"]
    store.close()


def test_ordering_query_orders_chronologically(tmp_path):
    store = make_store(tmp_path)
    base = datetime(2024, 3, 1, tzinfo=timezone.utc)
    memories = [
        Memory(memory_id=f"mem_{i}", content=f"event {i}", owner_id="owner", created_at=base + timedelta(days=i))
        for i in (1, 0)
    ]
    qdrant = EvidenceQdrant(memories)
    pipeline = HybridSearchPipeline(qdrant, store, StubEmbedder(), _config())

    results = pipeline.search("in what order did the events happen", "owner", limit=5)
    assert [r["memory_id"] for r in results] == ["mem_0", "mem_1"]
    store.close()


def test_global_query_skips_relative_gate_trim(tmp_path):
    store = make_store(tmp_path)
    memories = [
        Memory(memory_id=f"mem_{i}", content=f"aspect {i}", owner_id="owner")
        for i in range(4)
    ]
    qdrant = EvidenceQdrant(memories)
    config = _config(rerank_provider="remote", remote_rerank_url="https://unused", enable_relevance_gate=True, relevance_gate_min_keep=1)
    pipeline = HybridSearchPipeline(qdrant, store, StubEmbedder(), config)
    pipeline._reranker = FixedReranker([0.9, 0.001, 0.001, 0.001])

    results = pipeline.search("summarize everything about my work", "owner", limit=10)
    assert len(results) == 4, "breadth wins for summarization; relative trim must not apply"
    store.close()


def test_conflicts_attached_to_served_memory(tmp_path):
    store = make_store(tmp_path)
    claim = Memory(memory_id="mem_claim", content="User has never used Flask-Login", owner_id="owner")
    counter = Memory(memory_id="mem_counter", content="Flask-Login v0.6.2 is integrated in the user's app", owner_id="owner")
    store.store_memory(claim)
    store.store_memory(counter)
    store.link_conflict("mem_claim", "mem_counter", "owner")
    qdrant = EvidenceQdrant([claim, counter])
    pipeline = HybridSearchPipeline(qdrant, store, StubEmbedder(), _config())

    results = pipeline.search("does the user use Flask-Login", "owner", limit=5)
    by_id = {r["memory_id"]: r for r in results}
    assert [c["memory_id"] for c in by_id["mem_claim"]["conflicts"]] == ["mem_counter"]
    assert [c["memory_id"] for c in by_id["mem_counter"]["conflicts"]] == ["mem_claim"]
    store.close()
