"""
Cognis — Lightweight memory for LLM agents. Local-first or hosted.

pip install lyzr-cognis

Local mode (in-process, zero infrastructure):
    from cognis import Cognis

    memory = Cognis(gemini_api_key="...", owner_id="user_123")
    memory.add([{"role": "user", "content": "My name is Alice"}])
    results = memory.search("What is my name?")

Hosted mode (Lyzr memory service, authenticated with your Lyzr Studio API key):
    from cognis import CognisClient

    memory = CognisClient(api_key="lyzr-...", owner_id="user_123")
    memory.add([{"role": "user", "content": "My name is Alice"}])
    results = memory.search("What is my name?")
"""

from cognis.memory import Cognis
from cognis.config import CognisConfig
from cognis.client import CognisClient
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
    "CognisError",
    "CognisAuthenticationError",
    "CognisPermissionError",
    "CognisAPIError",
    "CognisConnectionError",
]
__version__ = "1.0.0"
