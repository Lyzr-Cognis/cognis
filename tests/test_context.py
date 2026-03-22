"""Tests for get_context — short-term + long-term retrieval."""


class TestGetContext:
    def test_returns_structure(self, seeded_memory):
        ctx = seeded_memory.get_context([{"role": "user", "content": "Tell me about myself"}])
        assert "short_term" in ctx
        assert "long_term" in ctx
        assert "short_term_count" in ctx
        assert "long_term_count" in ctx
        assert "context_string" in ctx

    def test_short_term_present(self, seeded_memory):
        ctx = seeded_memory.get_context()
        assert ctx["short_term_count"] > 0

    def test_long_term_present(self, seeded_memory):
        ctx = seeded_memory.get_context([{"role": "user", "content": "What sport do I play?"}])
        assert ctx["long_term_count"] > 0

    def test_context_string_has_content(self, seeded_memory):
        ctx = seeded_memory.get_context([{"role": "user", "content": "What is my name?"}])
        assert len(ctx["context_string"]) > 0

    def test_disable_long_term(self, seeded_memory):
        ctx = seeded_memory.get_context(
            [{"role": "user", "content": "hello"}],
            include_long_term=False,
        )
        assert ctx["long_term_count"] == 0
