"""
Cognis — Lightweight memory for LLM agents. Local-first or hosted.

pip install lyzr-cognis

Local mode (in-process, zero infrastructure):
    from cognis import Cognis

    memory = Cognis(gemini_api_key="...", owner_id="user_123")
    memory.add([{"role": "user", "content": "My name is Alice"}])
    results = memory.search("What is my name?")

Hosted mode (Lyzr memory service; API key, OIDC bearer token, or both):
    from cognis import CognisClient

    memory = CognisClient(api_key="lyzr-...", owner_id="user_123")
    memory.add([{"role": "user", "content": "My name is Alice"}])
    results = memory.search("What is my name?")

Okta / OIDC client-credentials:
    from cognis import CognisClient, OktaClientCredentials

    okta = OktaClientCredentials(issuer="https://acme.okta.com/oauth2/aus123",
                                 client_id="0oa...", client_secret="...",
                                 scopes=["memory.read", "memory.write"])
    memory = CognisClient(token_provider=okta, owner_id="user_123")
"""

from cognis.memory import Cognis
from cognis.config import CognisConfig
from cognis.client import CognisClient
from cognis.auth import OktaClientCredentials
from cognis.exceptions import (
    CognisError,
    CognisAuthenticationError,
    CognisPermissionError,
    CognisAPIError,
    CognisConnectionError,
)

__all__ = [
    "Cognis",
    "CognisConfig",
    "CognisClient",
    "OktaClientCredentials",
    "CognisError",
    "CognisAuthenticationError",
    "CognisPermissionError",
    "CognisAPIError",
    "CognisConnectionError",
]
__version__ = "1.1.0"
