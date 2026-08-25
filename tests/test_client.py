"""
Tests for CognisClient (hosted mode).

All HTTP is mocked with httpx.MockTransport — no network access.
"""

import json

import httpx
import pytest

from cognis.client import CognisClient, DEFAULT_BASE_URL
from cognis.exceptions import (
    CognisAPIError,
    CognisAuthenticationError,
    CognisConnectionError,
    CognisPermissionError,
)


def make_client(handler, **kwargs):
    """Build a CognisClient with a MockTransport handler."""
    kwargs.setdefault("api_key", "test-key")
    kwargs.setdefault("owner_id", "user_1")
    return CognisClient(transport=httpx.MockTransport(handler), **kwargs)


def capture(response_json=None, status_code=200):
    """Return (requests_list, handler) where handler records each request."""
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status_code, json=response_json if response_json is not None else {"success": True})

    return requests, handler


def body(request: httpx.Request) -> dict:
    return json.loads(request.content.decode())


# ── Construction / auth ─────────────────────────────────────────────────


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("LYZR_API_KEY", raising=False)
    with pytest.raises(CognisAuthenticationError, match="LYZR_API_KEY"):
        CognisClient(owner_id="user_1")


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("LYZR_API_KEY", "env-key")
    requests, handler = capture()
    client = CognisClient(owner_id="user_1", transport=httpx.MockTransport(handler))
    client.health()
    assert requests[0].headers["x-api-key"] == "env-key"


def test_api_key_param_wins_over_env(monkeypatch):
    monkeypatch.setenv("LYZR_API_KEY", "env-key")
    requests, handler = capture()
    client = make_client(handler, api_key="param-key")
    client.health()
    assert requests[0].headers["x-api-key"] == "param-key"


def test_requires_at_least_one_id():
    with pytest.raises(ValueError, match="At least one"):
        CognisClient(api_key="k")


def test_x_api_key_header_on_every_request():
    requests, handler = capture({"success": True, "results": [], "count": 0, "query": "q"})
    client = make_client(handler)
    client.search("q")
    client.health()
    assert all(r.headers["x-api-key"] == "test-key" for r in requests)


# ── Base URL resolution ─────────────────────────────────────────────────


def test_default_base_url(monkeypatch):
    monkeypatch.delenv("BASE_MEMORY_URL", raising=False)
    requests, handler = capture()
    client = make_client(handler)
    assert client.base_url == DEFAULT_BASE_URL == "https://memory.studio.lyzr.ai"
    client.health()
    assert str(requests[0].url) == "https://memory.studio.lyzr.ai/health"


def test_base_url_from_env(monkeypatch):
    monkeypatch.setenv("BASE_MEMORY_URL", "https://memory.onprem.example.com/")
    client = make_client(capture()[1])
    assert client.base_url == "https://memory.onprem.example.com"


def test_base_url_param_wins_over_env(monkeypatch):
    monkeypatch.setenv("BASE_MEMORY_URL", "https://env.example.com")
    client = make_client(capture()[1], base_url="https://param.example.com/")
    assert client.base_url == "https://param.example.com"


# ── Endpoint / payload mapping ──────────────────────────────────────────


def test_add_posts_messages():
    requests, handler = capture({"success": True, "message": "ok", "session_message_count": 2})
    client = make_client(handler, agent_id="agent_1", session_id="ses_1")
    resp = client.add([{"role": "user", "content": "hi"}])
    req = requests[0]
    assert req.method == "POST"
    assert req.url.path == "/v1/memories"
    payload = body(req)
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["owner_id"] == "user_1"
    assert payload["agent_id"] == "agent_1"
    assert payload["session_id"] == "ses_1"
    assert payload["provider_type"] == "cognis"
    assert resp["success"] is True


def test_add_per_call_overrides():
    requests, handler = capture()
    client = make_client(handler)
    client.add([{"role": "user", "content": "hi"}], owner_id="user_2", agent_id="agent_9")
    payload = body(requests[0])
    assert payload["owner_id"] == "user_2"
    assert payload["agent_id"] == "agent_9"


def test_search_defaults_to_cross_session():
    requests, handler = capture({"success": True, "results": [], "count": 0, "query": "q"})
    client = make_client(handler)
    client.search("q", limit=5)
    req = requests[0]
    assert req.method == "POST"
    assert req.url.path == "/v1/memories/search"
    payload = body(req)
    assert payload["cross_session"] is True
    assert payload["limit"] == 5


def test_search_explicit_session_scopes():
    requests, handler = capture({"success": True, "results": [], "count": 0, "query": "q"})
    client = make_client(handler)
    client.search("q", session_id="ses_x")
    payload = body(requests[0])
    assert payload["session_id"] == "ses_x"
    assert payload["cross_session"] is False


def test_get_memory():
    requests, handler = capture({"success": True, "memory": {"id": "mem_1", "content": "c"}})
    client = make_client(handler)
    resp = client.get("mem_1")
    req = requests[0]
    assert req.method == "GET"
    assert req.url.path == "/v1/memories/mem_1"
    assert req.url.params["owner_id"] == "user_1"
    assert resp["memory"]["id"] == "mem_1"


def test_get_all_and_count():
    requests, handler = capture(
        {"success": True, "memories": [], "total": 42, "limit": 1, "offset": 0}
    )
    client = make_client(handler)
    resp = client.get_all(limit=10, offset=5)
    req = requests[0]
    assert req.method == "GET"
    assert req.url.path == "/v1/memories"
    assert req.url.params["limit"] == "10"
    assert req.url.params["offset"] == "5"
    assert resp["total"] == 42

    assert client.count() == 42
    assert requests[1].url.params["limit"] == "1"


def test_delete_memory():
    requests, handler = capture({"success": True, "message": "deleted"})
    client = make_client(handler)
    client.delete("mem_1")
    req = requests[0]
    assert req.method == "DELETE"
    assert req.url.path == "/v1/memories/mem_1"
    assert req.url.params["owner_id"] == "user_1"


def test_update_memory():
    requests, handler = capture({"success": True})
    client = make_client(handler)
    client.update("mem_1", content="new", metadata={"category": "identity"})
    req = requests[0]
    assert req.method == "PATCH"
    assert req.url.path == "/v1/memories/mem_1"
    payload = body(req)
    assert payload["content"] == "new"
    assert payload["metadata"] == {"category": "identity"}
    assert "is_current" not in payload  # None fields are dropped


def test_get_context_synthesizes_context_string():
    requests, handler = capture({
        "success": True,
        "context": [
            {"role": "system", "content": "Relevant memories: Alice works at Google"},
            {"role": "user", "content": "hi"},
        ],
        "has_long_term_memory": True,
        "short_term_count": 1,
        "long_term_count": 1,
    })
    client = make_client(handler, session_id="ses_1")
    resp = client.get_context([{"role": "user", "content": "who am I?"}])
    req = requests[0]
    assert req.url.path == "/v1/memories/context"
    payload = body(req)
    assert payload["current_messages"] == [{"role": "user", "content": "who am I?"}]
    assert payload["max_short_term_messages"] == 30
    assert payload["enable_long_term_memory"] is True
    assert "system: Relevant memories" in resp["context_string"]
    assert "user: hi" in resp["context_string"]


def test_clear_defaults_to_cross_session():
    requests, handler = capture({"success": True, "message": "cleared"})
    client = make_client(handler)
    client.clear()
    req = requests[0]
    assert req.method == "DELETE"
    assert req.url.path == "/v1/memories/session"
    payload = body(req)
    assert payload["cross_session"] is True
    assert "session_id" not in payload


def test_clear_specific_session():
    requests, handler = capture({"success": True, "message": "cleared"})
    client = make_client(handler)
    client.clear(session_id="ses_x")
    payload = body(requests[0])
    assert payload["session_id"] == "ses_x"
    assert payload["cross_session"] is False


# ── Error mapping ────────────────────────────────────────────────────────


def error_client(status_code, detail):
    def handler(request):
        return httpx.Response(status_code, json={"detail": detail})
    return make_client(handler)


def test_403_invalid_key_maps_to_auth_error():
    client = error_client(403, "Invalid API key")
    with pytest.raises(CognisAuthenticationError, match="Invalid API key"):
        client.search("q")


def test_403_missing_permission_maps_to_permission_error():
    client = error_client(403, "Missing permission: memory:write")
    with pytest.raises(CognisPermissionError, match="memory:write"):
        client.delete("mem_1")


def test_401_maps_to_auth_error():
    client = error_client(401, "Unauthorized")
    with pytest.raises(CognisAuthenticationError):
        client.search("q")


def test_500_maps_to_api_error():
    client = error_client(500, "boom")
    with pytest.raises(CognisAPIError) as exc_info:
        client.search("q")
    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "boom"


def test_503_maps_to_api_error_with_retry_hint():
    client = error_client(503, "Auth cache unavailable")
    with pytest.raises(CognisAPIError, match="Retry shortly"):
        client.search("q")


def test_transport_error_maps_to_connection_error():
    def handler(request):
        raise httpx.ConnectError("connection refused")
    client = make_client(handler)
    with pytest.raises(CognisConnectionError, match="Could not reach"):
        client.search("q")


# ── Session management parity ────────────────────────────────────────────


def test_session_management():
    client = make_client(capture()[1], session_id="ses_1")
    assert client.session_id == "ses_1"
    client.set_session("ses_2")
    assert client.session_id == "ses_2"
    new = client.new_session()
    assert new.startswith("ses_") and client.session_id == new
    client.set_owner("user_9")
    client.set_agent("agent_9")
    assert client.owner_id == "user_9"
    assert client.agent_id == "agent_9"


def test_context_manager_closes():
    with make_client(capture()[1]) as client:
        assert "CognisClient" in repr(client)
    assert client._http.is_closed
