"""
Tests for configurable client auth: bearer tokens, token providers,
tenant headers, custom API-key header, and the OktaClientCredentials
client-credentials flow. All HTTP is mocked via httpx.MockTransport.
"""


import httpx
import pytest

from cognis import CognisClient, OktaClientCredentials
from cognis.exceptions import CognisAPIError, CognisAuthenticationError


def make_client(handler, **kwargs):
    kwargs.setdefault("owner_id", "user_1")
    return CognisClient(transport=httpx.MockTransport(handler), **kwargs)


def capture(response_json=None, status_code=200):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status_code, json=response_json or {"success": True})

    return requests, handler


# ── CognisClient: bearer token ───────────────────────────────────────────


def test_token_param_sends_bearer_no_api_key():
    requests, handler = capture()
    client = make_client(handler, token="tok-1")
    client.health()
    assert requests[0].headers["Authorization"] == "Bearer tok-1"
    assert "x-api-key" not in requests[0].headers


def test_token_from_env(monkeypatch):
    monkeypatch.delenv("LYZR_API_KEY", raising=False)
    monkeypatch.setenv("LYZR_MEMORY_TOKEN", "env-tok")
    requests, handler = capture()
    client = make_client(handler)
    client.health()
    assert requests[0].headers["Authorization"] == "Bearer env-tok"


def test_token_and_token_provider_conflict():
    with pytest.raises(ValueError, match="not both"):
        CognisClient(owner_id="u", token="t", token_provider=lambda: "t2")


def test_token_provider_called_per_request():
    tokens = iter(["tok-a", "tok-b"])
    requests, handler = capture()
    client = make_client(handler, token_provider=lambda: next(tokens))
    client.health()
    client.health()
    assert requests[0].headers["Authorization"] == "Bearer tok-a"
    assert requests[1].headers["Authorization"] == "Bearer tok-b"


# ── CognisClient: 401 refresh-once semantics ─────────────────────────────


class FakeProvider:
    def __init__(self, tokens):
        self._tokens = iter(tokens)
        self.invalidations = 0

    def __call__(self):
        return next(self._tokens)

    def invalidate(self):
        self.invalidations += 1


def test_401_with_provider_refreshes_once_and_retries():
    provider = FakeProvider(["stale", "fresh"])
    requests = []

    def handler(request):
        requests.append(request)
        if request.headers["Authorization"] == "Bearer stale":
            return httpx.Response(401, json={"detail": "token expired"})
        return httpx.Response(200, json={"success": True})

    client = make_client(handler, token_provider=provider)
    assert client.health() == {"success": True}
    assert len(requests) == 2
    assert provider.invalidations == 1
    assert requests[1].headers["Authorization"] == "Bearer fresh"


def test_401_twice_with_provider_raises_after_single_retry():
    provider = FakeProvider(["t1", "t2"])
    requests, handler = capture({"detail": "nope"}, status_code=401)
    client = make_client(handler, token_provider=provider)
    with pytest.raises(CognisAuthenticationError):
        client.health()
    assert len(requests) == 2
    assert provider.invalidations == 1


def test_401_without_provider_does_not_retry():
    requests, handler = capture({"detail": "bad key"}, status_code=401)
    client = make_client(handler, api_key="k")
    with pytest.raises(CognisAuthenticationError):
        client.health()
    assert len(requests) == 1


# ── CognisClient: API-key header + tenant headers ────────────────────────


def test_custom_api_key_header():
    requests, handler = capture()
    client = make_client(handler, api_key="pep-key", api_key_header="x-pepgenx-apikey")
    client.health()
    assert requests[0].headers["x-pepgenx-apikey"] == "pep-key"
    assert "x-api-key" not in requests[0].headers


def test_tenant_headers_on_every_request():
    requests, handler = capture(
        {"success": True, "results": [], "count": 0, "query": "q"}
    )
    client = make_client(
        handler,
        api_key="k",
        tenant={"team_id": "T1", "project_id": "P1", "user_id": "049000001"},
    )
    client.health()
    client.search("q")
    for r in requests:
        assert r.headers["team_id"] == "T1"
        assert r.headers["project_id"] == "P1"
        assert r.headers["user_id"] == "049000001"


def test_tenant_header_rename_map():
    requests, handler = capture()
    client = make_client(
        handler,
        api_key="k",
        tenant={"org_id": "T1"},
        tenant_headers={"org_id": "team_id"},
    )
    client.health()
    assert requests[0].headers["team_id"] == "T1"
    assert "org_id" not in requests[0].headers


def test_composite_sends_all_three_factors():
    requests, handler = capture()
    client = make_client(
        handler,
        token="okta-jwt",
        api_key="pep-key",
        api_key_header="x-pepgenx-apikey",
        tenant={"team_id": "T1", "project_id": "P1"},
    )
    client.health()
    r = requests[0]
    assert r.headers["Authorization"] == "Bearer okta-jwt"
    assert r.headers["x-pepgenx-apikey"] == "pep-key"
    assert r.headers["team_id"] == "T1"
    assert r.headers["project_id"] == "P1"


def test_base_url_new_env_wins_over_legacy(monkeypatch):
    monkeypatch.setenv("LYZR_MEMORY_BASE_URL", "https://new.example.com")
    monkeypatch.setenv("BASE_MEMORY_URL", "https://legacy.example.com")
    client = make_client(capture()[1], api_key="k")
    assert client.base_url == "https://new.example.com"


# ── OktaClientCredentials ────────────────────────────────────────────────


def okta_transport(responses, requests=None):
    """MockTransport returning queued responses; records requests."""
    queue = list(responses)

    def handler(request):
        if requests is not None:
            requests.append(request)
        status, body = queue.pop(0) if len(queue) > 1 else queue[0]
        return httpx.Response(status, json=body)

    return httpx.MockTransport(handler)


def token_response(token="tok-1", expires_in=3600):
    return (200, {"access_token": token, "expires_in": expires_in, "token_type": "Bearer"})


def test_okta_mints_token_with_basic_auth_and_scopes():
    requests = []
    provider = OktaClientCredentials(
        issuer="https://acme.okta.com/oauth2/aus123/",
        client_id="cid",
        client_secret="csec",
        scopes=["memory.read", "memory.write"],
        transport=okta_transport([token_response()], requests),
    )
    assert provider() == "tok-1"
    req = requests[0]
    assert str(req.url) == "https://acme.okta.com/oauth2/aus123/v1/token"
    assert req.headers["Authorization"].startswith("Basic ")
    form = dict(
        pair.split("=", 1) for pair in req.content.decode().split("&")
    )
    assert form["grant_type"] == "client_credentials"
    assert form["scope"] == "memory.read+memory.write"


def test_okta_explicit_token_url_overrides_issuer():
    requests = []
    provider = OktaClientCredentials(
        token_url="https://idp.example.com/oauth/token",
        client_id="cid",
        client_secret="csec",
        transport=okta_transport([token_response()], requests),
    )
    provider()
    assert str(requests[0].url) == "https://idp.example.com/oauth/token"


def test_okta_caches_until_near_expiry():
    clock = [1000.0]
    requests = []
    provider = OktaClientCredentials(
        issuer="https://acme.okta.com/oauth2/aus123",
        client_id="cid",
        client_secret="csec",
        transport=okta_transport(
            [token_response("tok-1", expires_in=300), token_response("tok-2")],
            requests,
        ),
        now=lambda: clock[0],
    )
    assert provider() == "tok-1"
    assert provider() == "tok-1"  # cached, no second HTTP call
    assert len(requests) == 1
    clock[0] = 1000.0 + 300 - 29  # within the 30s expiry margin
    assert provider() == "tok-2"
    assert len(requests) == 2


def test_okta_invalidate_forces_refetch():
    requests = []
    provider = OktaClientCredentials(
        issuer="https://acme.okta.com/oauth2/aus123",
        client_id="cid",
        client_secret="csec",
        transport=okta_transport(
            [token_response("tok-1"), token_response("tok-2")], requests
        ),
    )
    assert provider() == "tok-1"
    provider.invalidate()
    assert provider() == "tok-2"
    assert len(requests) == 2


def test_okta_error_response_raises_auth_error():
    provider = OktaClientCredentials(
        issuer="https://acme.okta.com/oauth2/aus123",
        client_id="cid",
        client_secret="wrong",
        transport=okta_transport([(401, {"error": "invalid_client"})]),
    )
    with pytest.raises(CognisAuthenticationError, match="401"):
        provider()


def test_okta_missing_access_token_raises():
    provider = OktaClientCredentials(
        issuer="https://acme.okta.com/oauth2/aus123",
        client_id="cid",
        client_secret="csec",
        transport=okta_transport([(200, {"token_type": "Bearer"})]),
    )
    with pytest.raises(CognisAuthenticationError, match="no access_token"):
        provider()


def test_okta_requires_issuer_or_token_url():
    with pytest.raises(ValueError, match="issuer or token_url"):
        OktaClientCredentials(client_id="cid", client_secret="csec")


def test_okta_requires_client_credentials():
    with pytest.raises(ValueError, match="client_id and client_secret"):
        OktaClientCredentials(issuer="https://acme.okta.com/oauth2/aus123")


def test_okta_end_to_end_with_client():
    """OktaClientCredentials plugged into CognisClient as token_provider."""
    okta_requests = []
    provider = OktaClientCredentials(
        issuer="https://acme.okta.com/oauth2/aus123",
        client_id="cid",
        client_secret="csec",
        transport=okta_transport([token_response("okta-tok")], okta_requests),
    )
    api_requests, handler = capture()
    client = make_client(handler, token_provider=provider)
    client.health()
    client.health()
    assert all(
        r.headers["Authorization"] == "Bearer okta-tok" for r in api_requests
    )
    assert len(okta_requests) == 1  # token cached across API calls
