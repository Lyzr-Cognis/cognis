"""
Bearer-token providers for CognisClient.

A token provider is any zero-argument callable returning a bearer access
token string. CognisClient calls it before each request and, on a 401,
invalidates it once (if it has an ``invalidate()`` method) and retries.

OktaClientCredentials implements the OAuth2 client-credentials flow against
Okta or any OIDC-compliant IdP (Entra, Auth0, Keycloak) and caches the token
until 30 seconds before expiry — the standard way for a service or agent
workload to authenticate to a lyzr-memory deployment running in
``oidc_jwt`` or ``composite`` auth mode.

Usage:
    from cognis import CognisClient, OktaClientCredentials

    okta = OktaClientCredentials(
        issuer="https://acme.okta.com/oauth2/aus123",
        client_id="0oa...",
        client_secret="...",
        scopes=["memory.read", "memory.write"],
    )
    m = CognisClient(token_provider=okta, owner_id="user_123")
"""

import threading
import time
from typing import Callable, Optional, Sequence

import httpx

from cognis.exceptions import CognisAuthenticationError, CognisConnectionError

# Refresh this many seconds before the token actually expires.
_EXPIRY_MARGIN_SECONDS = 30.0


class OktaClientCredentials:
    """OAuth2 client-credentials token provider with expiry-aware caching.

    Args:
        issuer: Authorization-server issuer URL; the token endpoint is
            derived as ``<issuer>/v1/token`` (the Okta convention).
        token_url: Explicit token endpoint; overrides derivation from
            ``issuer`` (use for IdPs with a different endpoint layout).
        client_id / client_secret: The service app's credentials
            (sent as HTTP Basic auth, per RFC 6749 client_secret_basic).
        scopes: OAuth scopes to request (e.g. ["memory.read", "memory.write"]).
        timeout: Token-request timeout in seconds.
        transport: Custom httpx transport (testing only).
        now: Clock override (testing only).

    Thread-safe: concurrent callers share one cached token; only one
    refresh runs at a time.
    """

    def __init__(
        self,
        issuer: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        token_url: Optional[str] = None,
        timeout: float = 10.0,
        transport: Optional[httpx.BaseTransport] = None,
        now: Callable[[], float] = time.time,
    ):
        if not token_url:
            if not issuer:
                raise ValueError("Either issuer or token_url is required")
            token_url = issuer.rstrip("/") + "/v1/token"
        if not client_id or not client_secret:
            raise ValueError("client_id and client_secret are required")

        self._token_url = token_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._scopes = list(scopes or [])
        self._now = now
        self._http = httpx.Client(timeout=timeout, transport=transport)

        self._lock = threading.Lock()
        self._token: Optional[str] = None
        self._expires_at: float = 0.0

    def __call__(self) -> str:
        """Return a valid access token, refreshing if missing or near expiry."""
        with self._lock:
            if self._token and self._now() < self._expires_at:
                return self._token
            self._refresh_locked()
            return self._token

    def invalidate(self) -> None:
        """Drop the cached token; the next call fetches a fresh one."""
        with self._lock:
            self._token = None
            self._expires_at = 0.0

    def _refresh_locked(self) -> None:
        data = {"grant_type": "client_credentials"}
        if self._scopes:
            data["scope"] = " ".join(self._scopes)

        try:
            response = self._http.post(
                self._token_url,
                auth=(self._client_id, self._client_secret),
                data=data,
                headers={"Accept": "application/json"},
            )
        except httpx.TransportError as e:
            raise CognisConnectionError(
                f"Could not reach token endpoint {self._token_url}: {e}"
            ) from e

        if response.status_code != 200:
            raise CognisAuthenticationError(
                f"Token request to {self._token_url} failed "
                f"({response.status_code}): {response.text[:200]}"
            )

        payload = response.json()
        token = payload.get("access_token")
        if not token:
            raise CognisAuthenticationError(
                f"Token endpoint {self._token_url} returned no access_token"
            )
        self._token = token
        expires_in = float(payload.get("expires_in") or 3600)
        self._expires_at = self._now() + expires_in - _EXPIRY_MARGIN_SECONDS

    def close(self) -> None:
        self._http.close()

    def __repr__(self) -> str:
        return (
            f"OktaClientCredentials(token_url={self._token_url}, "
            f"client_id={self._client_id})"
        )
