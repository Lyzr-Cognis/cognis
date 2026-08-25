"""
Exceptions for the Cognis hosted client.

Raised by CognisClient when talking to a hosted lyzr-memory deployment
(default: https://memory.studio.lyzr.ai).
"""

from typing import Optional


class CognisError(Exception):
    """Base exception for all Cognis errors."""


class CognisAuthenticationError(CognisError):
    """Authentication failed: missing, invalid, or unauthorized API key.

    Raised client-side when no API key can be resolved, and server-side on
    HTTP 401/403 responses whose detail indicates an API-key problem
    (e.g. "API key missing", "Invalid API key",
    "API key has no associated organization").
    """


class CognisPermissionError(CognisError):
    """Authorization failed: the API key is valid but lacks an RBAC permission.

    Raised on HTTP 403 responses with a "Missing permission: ..." detail —
    delete/update/clear require the `memory:write` permission on the key's
    policy when the server enforces RBAC.
    """


class CognisAPIError(CognisError):
    """The hosted service returned an unexpected error response."""

    def __init__(self, message: str, status_code: Optional[int] = None, detail: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


class CognisConnectionError(CognisError):
    """Could not reach the hosted service (DNS, refused connection, timeout)."""
