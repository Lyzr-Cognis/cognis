"""
CognisClient — hosted-mode client for the Lyzr memory service.

Talks to a hosted lyzr-memory deployment over HTTPS with the same method
surface as the local `Cognis` class. Authentication uses a Lyzr Studio API
key sent as the `x-api-key` header; authorization (org scoping and RBAC
permissions such as `memory:write`) is enforced server-side.

Configuration resolution order (param → env var → default):
- api_key:  api_key param → $LYZR_API_KEY            (required)
- base_url: base_url param → $BASE_MEMORY_URL → https://memory.studio.lyzr.ai

Usage:
    from cognis import CognisClient

    m = CognisClient(api_key="lyzr-...", owner_id="user_123")
    m.add([{"role": "user", "content": "My name is Alice"}])
    results = m.search("What is my name?")
    m.close()
"""

import logging
import os
from typing import Any, Dict, List, Optional

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
BASE_URL_ENV = "BASE_MEMORY_URL"
API_KEY_ENV = "LYZR_API_KEY"
PROVIDER_TYPE = "cognis"


def _require_at_least_one(**kwargs):
    """Validate at least one of the given kwargs is non-None."""
    if not any(v for v in kwargs.values()):
        names = ", ".join(kwargs.keys())
        raise ValueError(f"At least one of ({names}) is required")


def _resolve_base_url(base_url: Optional[str]) -> str:
    url = base_url or os.environ.get(BASE_URL_ENV) or DEFAULT_BASE_URL
    return url.rstrip("/")


class CognisClient:
    """
    Hosted memory client for the Lyzr memory service.

    Same session model as the local `Cognis` class:
    - owner_id, agent_id, session_id — at least one required
    - Extracted memories are global to (owner_id, agent_id)
    - Raw messages are scoped to (owner_id, agent_id, session_id)

    All data is additionally isolated to the organization that owns the API
    key — the server derives the org from the key, so two orgs using the same
    owner_id never see each other's memories.
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
    ):
        """
        Args:
            api_key: Lyzr Studio API key (default: $LYZR_API_KEY)
            base_url: Memory service URL (default: $BASE_MEMORY_URL, then
                https://memory.studio.lyzr.ai)
            owner_id: Memory owner identifier
            agent_id: Agent identifier
            session_id: Session identifier (auto-generated if omitted)
            timeout: Request timeout in seconds
            transport: Custom httpx transport (testing only)
        """
        _require_at_least_one(owner_id=owner_id, agent_id=agent_id, session_id=session_id)

        self._api_key = api_key or os.environ.get(API_KEY_ENV)
        if not self._api_key:
            raise CognisAuthenticationError(
                f"No API key provided. Pass api_key= or set the {API_KEY_ENV} "
                "environment variable with your Lyzr Studio API key."
            )

        self._base_url = _resolve_base_url(base_url)
        self._owner_id = owner_id
        self._agent_id = agent_id
        self._session_id = session_id or generate_session_id()

        self._http = httpx.Client(
            base_url=self._base_url,
            timeout=timeout,
            headers={"x-api-key": self._api_key, "Content-Type": "application/json"},
            transport=transport,
        )

        logger.info(
            "CognisClient initialized (base_url=%s, owner=%s, agent=%s, session=%s)",
            self._base_url, self._owner_id, self._agent_id, self._session_id,
        )

    # ── HTTP plumbing ────────────────────────────────────────────────────

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

        try:
            response = self._http.request(method, path, json=json, params=params)
        except httpx.TransportError as e:
            raise CognisConnectionError(
                f"Could not reach memory service at {self._base_url}: {e}"
            ) from e

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
