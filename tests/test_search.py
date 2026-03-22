"""Tests for search pipeline — hybrid RRF, relevance, latency."""

import time
import pytest


class TestSearchFormat:
    def test_returns_hosted_format(self, seeded_memory):
        resp = seeded_memory.search("user name")
        assert resp["success"] is True
        assert isinstance(resp["results"], list)
        assert "count" in resp
        assert "query" in resp
        assert resp["query"] == "user name"

    def test_results_have_fields(self, seeded_memory):
        resp = seeded_memory.search("user name", limit=3)
        r = resp["results"][0]
        assert "id" in r
        assert "content" in r
        assert "score" in r
        assert "type" in r
        assert "metadata" in r
        assert "category" in r["metadata"]

    def test_respects_limit(self, seeded_memory):
        resp = seeded_memory.search("user", limit=2)
        assert len(resp["results"]) <= 2


class TestSearchRelevance:
    @pytest.mark.parametrize("query,keywords", [
        ("What is the user's name?", ["parshva", "name"]),
        ("Where does the user work?", ["lyzr", "engineer", "work"]),
        ("What sport does the user play?", ["cricket"]),
        ("What food does the user like?", ["dosa", "south indian", "food"]),
        ("What is the user working on?", ["memory", "agent", "building"]),
        ("Who is the user a fan of?", ["virat", "kohli", "fan", "cricket", "huge", "playing"]),
    ])
    def test_relevance(self, seeded_memory, query, keywords):
        resp = seeded_memory.search(query, limit=5)
        all_content = " ".join(r["content"].lower() for r in resp["results"])
        assert any(kw in all_content for kw in keywords), (
            f"None of {keywords} found in search results for '{query}'"
        )


class TestSearchLatency:
    def test_avg_latency_under_3s(self, seeded_memory):
        queries = ["user name", "what food", "what sport", "where work"]
        latencies = []
        for q in queries:
            t = time.time()
            seeded_memory.search(q, limit=5)
            latencies.append((time.time() - t) * 1000)
        avg = sum(latencies) / len(latencies)
        assert avg < 3000, f"Avg latency {avg:.0f}ms exceeds 3s"
