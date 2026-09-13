"""
CognisClient — hosted-mode client for the Lyzr memory service.

Talks to a hosted lyzr-memory deployment over HTTPS with the same method
surface as the local `Cognis` class. Authentication is configurable to match
the service's pluggable auth modes:

- API key (default): sent in the `api_key_header` (default `x-api-key`) —
  Lyzr Studio keys, standalone `lm_...` keys, or PepGenX onboarding keys
  (`api_key_header="x-pepgenx-apikey"`).
- Bearer token: a static `token` or a `token_provider` callable (e.g.
  `OktaClientCredentials`) sent as `Authorization: Bearer ...` — for
  deployments validating Okta / any OIDC JWTs. On a 401 the provider is
  refreshed once and the request retried.
- Tenant headers: `tenant={"team_id": ..., "project_id": ..., "user_id": ...}`
  sent on every request — required by composite (PepGenX-style) deployments.

These combine: composite mode sends bearer + API key + tenant headers.
Authorization (org/project scoping and RBAC permissions such as
`memory:write`) is always enforced server-side.

Configuration resolution order (param → env var → default):
- api_key:  api_key param → $LYZR_API_KEY
- token:    token param   → $LYZR_MEMORY_TOKEN
- base_url: base_url param → $LYZR_MEMORY_BASE_URL → $BASE_MEMORY_URL
            → https://memory.studio.lyzr.ai
At least one credential (api_key, token, or token_provider) is required.

Usage:
    from cognis import CognisClient

    # API key (default)
    m = CognisClient(api_key="lyzr-...", owner_id="user_123")

    # Okta client-credentials (OIDC bearer)
    from cognis import OktaClientCredentials
    okta = OktaClientCredentials(issuer="https://acme.okta.com/oauth2/aus123",
                                 client_id="0oa...", client_secret="...",
                                 scopes=["memory.read", "memory.write"])
    m = CognisClient(token_provider=okta, owner_id="user_123")

    # PepGenX composite (bearer + API key + tenant headers)
    m = CognisClient(token_provider=okta, api_key="pep-...",
                     api_key_header="x-pepgenx-apikey",
                     tenant={"team_id": "T1", "project_id": "P1",
                             "user_id": "049000001"},
                     owner_id="user_123")

    m.add([{"role": "user", "content": "My name is Alice"}])
    results = m.search("What is my name?")
    m.close()
"""

import logging
import os
from typing import Any, Callable, Dict, List, Mapping, Optional

import httpx

from cognis.exceptions import (
    CognisAPIError,
    CognisAuthenticationError,
    CognisConnectionError,
    CognisPermissionError,
)
from cognis.utils import generate_session_id

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://memory.studio.lyzr.ai"
BASE_URL_ENV = "LYZR_MEMORY_BASE_URL"
LEGACY_BASE_URL_ENV = "BASE_MEMORY_URL"
API_KEY_ENV = "LYZR_API_KEY"
TOKEN_ENV = "LYZR_MEMORY_TOKEN"
DEFAULT_API_KEY_HEADER = "x-api-key"
PROVIDER_TYPE = "cognis"


def _require_at_least_one(**kwargs):
    """Validate at least one of the given kwargs is non-None."""
    if not any(v for v in kwargs.values()):
        names = ", ".join(kwargs.keys())
        raise ValueError(f"At least one of ({names}) is required")


def _resolve_base_url(base_url: Optional[str]) -> str:
    url = (
        base_url
        or os.environ.get(BASE_URL_ENV)
        or os.environ.get(LEGACY_BASE_URL_ENV)
        or DEFAULT_BASE_URL
    )
    return url.rstrip("/")


class CognisClient:
    """
    Hosted memory client for the Lyzr memory service.

    Same session model as the local `Cognis` class:
    - owner_id, agent_id, session_id — at least one required
    - Extracted memories are global to (owner_id, agent_id)
    - Raw messages are scoped to (owner_id, agent_id, session_id)

    All data is additionally isolated to the organization (and optional
    project) resolved from the credential server-side — an API key's binding
    or a bearer token's claims — so two orgs using the same owner_id never
    see each other's memories.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        owner_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        timeout: float = 30.0,
        transport: Optional[httpx.BaseTransport] = None,
        api_key_header: str = DEFAULT_API_KEY_HEADER,
        token: Optional[str] = None,
        token_provider: Optional[Callable[[], str]] = None,
        tenant: Optional[Mapping[str, str]] = None,
        tenant_headers: Optional[Mapping[str, str]] = None,
    ):
        """
        Args:
            api_key: API key (default: $LYZR_API_KEY)
            base_url: Memory service URL (default: $LYZR_MEMORY_BASE_URL,
                then $BASE_MEMORY_URL, then https://memory.studio.lyzr.ai)
            owner_id: Memory owner identifier
            agent_id: Agent identifier
            session_id: Session identifier (auto-generated if omitted)
            timeout: Request timeout in seconds
            transport: Custom httpx transport (testing only)
            api_key_header: Header name carrying the API key
                (default "x-api-key"; PepGenX uses "x-pepgenx-apikey")
            token: Static bearer token (default: $LYZR_MEMORY_TOKEN);
                sent as `Authorization: Bearer ...`
            token_provider: Zero-argument callable returning a bearer token,
                called per request (e.g. OktaClientCredentials); mutually
                exclusive with `token`. A 401 response triggers exactly one
                `invalidate()` + retry when the provider supports it.
            tenant: Tenant headers sent on every request, e.g.
                {"team_id": "T1", "project_id": "P1", "user_id": "049..."}
            tenant_headers: Optional rename map from `tenant` keys to wire
                header names (default: keys are used verbatim)
        """
        _require_at_least_one(owner_id=owner_id, agent_id=agent_id, session_id=session_id)

        if token and token_provider:
            raise ValueError("Pass either token or token_provider, not both")

        self._api_key = api_key or os.environ.get(API_KEY_ENV)
        self._token = token or os.environ.get(TOKEN_ENV)
        self._token_provider = token_provider
        if not (self._api_key or self._token or self._token_provider):
            raise CognisAuthenticationError(
                "No credentials provided. Pass api_key= (or set the "
                f"{API_KEY_ENV} environment variable), token= (or set "
                f"{TOKEN_ENV}), or token_provider=."
            )

        self._base_url = _resolve_base_url(base_url)
        self._owner_id = owner_id
        self._agent_id = agent_id
        self._session_id = session_id or generate_session_id()

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers[api_key_header] = self._api_key
        if self._token and not self._token_provider:
            headers["Authorization"] = f"Bearer {self._token}"
        for key, value in (tenant or {}).items():
            if value is not None:
                headers[(tenant_headers or {}).get(key, key)] = str(value)

        self._http = httpx.Client(
            base_url=self._base_url,
            timeout=timeout,
            headers=headers,
            transport=transport,
        )

        logger.info(
            "CognisClient initialized (base_url=%s, owner=%s, agent=%s, session=%s)",
            self._base_url, self._owner_id, self._agent_id, self._session_id,
        )

    # ── HTTP plumbing ────────────────────────────────────────────────────

    def _send(
        self,
        method: str,
        path: str,
        json: Optional[Dict[str, Any]],
        params: Optional[Dict[str, Any]],
    ) -> httpx.Response:
        headers = None
        if self._token_provider is not None:
            headers = {"Authorization": f"Bearer {self._token_provider()}"}
        try:
            return self._http.request(
                method, path, json=json, params=params, headers=headers
            )
        except httpx.TransportError as e:
            raise CognisConnectionError(
                f"Could not reach memory service at {self._base_url}: {e}"
            ) from e

    def _request(
        self,
        method: str,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if params:
            params = {k: v for k, v in params.items() if v is not None}
        if json:
            json = {k: v for k, v in json.items() if v is not None}

        response = self._send(method, path, json, params)

        # An expired/revoked bearer token earns exactly one refresh + retry.
        if response.status_code == 401 and self._token_provider is not None:
            invalidate = getattr(self._token_provider, "invalidate", None)
            if callable(invalidate):
                invalidate()
            response = self._send(method, path, json, params)

        if response.status_code >= 400:
            self._raise_for_status(response)
        return response.json()

    def _raise_for_status(self, response: httpx.Response) -> None:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        detail = detail if isinstance(detail, str) else str(detail)
        status = response.status_code

        if status == 401:
            raise CognisAuthenticationError(f"Authentication failed ({status}): {detail}")
        if status == 403:
            if detail.startswith("Missing permission"):
                raise CognisPermissionError(
                    f"Permission denied ({status}): {detail}. Your API key's policy "
                    "needs this permission (e.g. memory:write) granted in Lyzr Studio."
                )
            raise CognisAuthenticationError(f"Authentication failed ({status}): {detail}")
        if status == 503:
            raise CognisAPIError(
                f"Memory service temporarily unavailable ({status}): {detail}. Retry shortly.",
                status_code=status,
                detail=detail,
            )
        raise CognisAPIError(
            f"Memory service error ({status}): {detail}",
            status_code=status,
            detail=detail,
        )

    # ── Core API ─────────────────────────────────────────────────────────

    def add(
        self,
        messages: List[Dict[str, str]],
        owner_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        sync_extraction: bool = False,
        instructions: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Add messages; the hosted service extracts memories from them.

        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
            owner_id: Override owner (defaults to instance owner_id)
            agent_id: Override agent (defaults to instance agent_id)
            session_id: Override session (defaults to instance session_id)
            sync_extraction: Wait for fact extraction to finish before returning
            instructions: Custom extraction instructions for the server

        Returns dict matching hosted MemoryResponse:
        {"success": True, "message": "...", "session_message_count": N}
        """
        oid = owner_id or self._owner_id
        aid = agent_id or self._agent_id
        sid = session_id or self._session_id
        _require_at_least_one(owner_id=oid, agent_id=aid, session_id=sid)

        return self._request("POST", "/v1/memories", json={
            "messages": messages,
            "owner_id": oid,
            "agent_id": aid,
            "session_id": sid,
            "provider_type": PROVIDER_TYPE,
            "sync_extraction": sync_extraction,
            "instructions": instructions,
        })

    def search(
        self,
        query: str,
        limit: int = 10,
        owner_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        include_historical: bool = False,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search memories on the hosted service.

        Matches local semantics: extracted memories are global to
        (owner_id, agent_id), so search spans all sessions unless an explicit
        session_id is passed to scope it down.

        Returns dict matching hosted SearchResponse:
        {"success": True, "results": [...], "count": N, "query": "..."}
        """
        return self._request("POST", "/v1/memories/search", json={
            "query": query,
            "limit": limit,
            "owner_id": owner_id or self._owner_id,
            "agent_id": agent_id or self._agent_id,
            "session_id": session_id or self._session_id,
            "cross_session": session_id is None,
            "provider_type": PROVIDER_TYPE,
            "include_historical": include_historical,
            "category": category,
        })

    def get(self, memory_id: str, owner_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a specific memory by ID.

        Returns dict matching hosted MemoryDetailResponse:
        {"success": True, "memory": {...}}
        """
        return self._request("GET", f"/v1/memories/{memory_id}", params={
            "owner_id": owner_id or self._owner_id,
            "provider_type": PROVIDER_TYPE,
        })

    def get_all(
        self,
        limit: int = 100,
        offset: int = 0,
        owner_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        include_historical: bool = False,
    ) -> Dict[str, Any]:
        """
        List memories for the owner/agent.

        Returns dict matching hosted MemoriesListResponse:
        {"success": True, "memories": [...], "total": N, "limit": N, "offset": N}
        """
        return self._request("GET", "/v1/memories", params={
            "owner_id": owner_id or self._owner_id,
            "agent_id": agent_id or self._agent_id,
            "limit": limit,
            "offset": offset,
            "include_historical": include_historical,
            "cross_session": True,
            "provider_type": PROVIDER_TYPE,
        })

    def delete(self, memory_id: str, owner_id: Optional[str] = None) -> Dict[str, Any]:
        """Delete a specific memory. Requires the memory:write permission."""
        return self._request("DELETE", f"/v1/memories/{memory_id}", params={
            "owner_id": owner_id or self._owner_id,
            "provider_type": PROVIDER_TYPE,
        })

    def update(
        self,
        memory_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_current: Optional[bool] = None,
        owner_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Update a memory's content, metadata, or current/historical status.
        Requires the memory:write permission. Updating content triggers
        re-embedding server-side.
        """
        return self._request("PATCH", f"/v1/memories/{memory_id}", json={
            "owner_id": owner_id or self._owner_id,
            "content": content,
            "metadata": metadata,
            "is_current": is_current,
            "provider_type": PROVIDER_TYPE,
        })

    def get_context(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        max_short_term: int = 30,
        include_long_term: bool = True,
        owner_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        cross_session: bool = False,
    ) -> Dict[str, Any]:
        """
        Get conversation context (short-term messages + long-term memories).

        Returns the hosted ContextResponse plus a computed "context_string":
        {"success", "context": [messages...], "has_long_term_memory",
         "short_term_count", "long_term_count", "context_string"}
        """
        oid = owner_id or self._owner_id
        aid = agent_id or self._agent_id
        sid = session_id or self._session_id
        _require_at_least_one(owner_id=oid, agent_id=aid, session_id=sid)

        result = self._request("POST", "/v1/memories/context", json={
            "owner_id": oid,
            "agent_id": aid,
            "session_id": sid,
            "current_messages": messages or [],
            "provider_type": PROVIDER_TYPE,
            "max_short_term_messages": max_short_term,
            "enable_long_term_memory": include_long_term,
            "cross_session": cross_session,
        })

        context_msgs = result.get("context") or []
        result["context_string"] = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in context_msgs
        )
        return result

    def clear(
        self,
        owner_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Clear memories for the owner. With session_id, clears only that
        session; without it, clears across all sessions (cross_session).
        Requires the memory:write permission.
        """
        oid = owner_id or self._owner_id
        return self._request("DELETE", "/v1/memories/session", json={
            "owner_id": oid,
            "agent_id": agent_id or self._agent_id,
            "session_id": session_id,
            "provider_type": PROVIDER_TYPE,
            "cross_session": session_id is None,
        })

    def count(self, owner_id: Optional[str] = None) -> int:
        """Count current memories for this owner."""
        result = self.get_all(limit=1, owner_id=owner_id)
        return int(result.get("total", 0))

    def health(self) -> Dict[str, Any]:
        """Check the hosted service's health endpoint (no auth required)."""
        return self._request("GET", "/health")

    # ── Session Management ───────────────────────────────────────────────

    def set_session(self, session_id: str) -> None:
        self._session_id = session_id

    def set_owner(self, owner_id: str) -> None:
        self._owner_id = owner_id

    def set_agent(self, agent_id: str) -> None:
        self._agent_id = agent_id

    def new_session(self) -> str:
        self._session_id = generate_session_id()
        return self._session_id

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def owner_id(self) -> Optional[str]:
        return self._owner_id

    @property
    def agent_id(self) -> Optional[str]:
        return self._agent_id

    @property
    def base_url(self) -> str:
        return self._base_url

    # ── Lifecycle ────────────────────────────────────────────────────────

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        return (
            f"CognisClient(base_url={self._base_url}, owner={self._owner_id}, "
            f"agent={self._agent_id}, session={self._session_id})"
        )
