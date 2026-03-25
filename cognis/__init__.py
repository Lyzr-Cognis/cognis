"""
Cognis — Lightweight, local-first memory for LLM agents.

pip install lyzr-cognis

Usage:
    from cognis import Cognis, CognisConfig

    memory = Cognis(gemini_api_key="...", owner_id="user_123")
    memory.add([{"role": "user", "content": "My name is Alice"}])
    results = memory.search("What is my name?")
"""

from cognis.memory import Cognis
from cognis.config import CognisConfig

__all__ = ["Cognis", "CognisConfig"]
__version__ = "0.1.0"
