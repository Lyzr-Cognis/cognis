"""Shared fixtures for Cognis tests."""

import os
import tempfile
import warnings

import pytest

warnings.filterwarnings("ignore")

from dotenv import load_dotenv

_env = os.path.join(os.path.dirname(__file__), "..", "..", "lyzr-memory", ".env")
if os.path.exists(_env):
    load_dotenv(_env)


@pytest.fixture
def data_dir():
    with tempfile.TemporaryDirectory() as td:
        yield td


@pytest.fixture
def memory(data_dir):
    from cognis import Cognis

    m = Cognis(data_dir=data_dir, owner_id="test_user", agent_id="test_agent")
    yield m
    m.close()


@pytest.fixture(scope="module")
def seeded_memory():
    """Module-scoped memory pre-loaded with conversations. Created once, reused across tests."""
    from cognis import Cognis

    with tempfile.TemporaryDirectory() as td:
        m = Cognis(data_dir=td, owner_id="test_user", agent_id="test_agent")
        m.add([
            {"role": "user", "content": "Hi! My name is Parshva and I am a software engineer at Lyzr AI."},
            {"role": "assistant", "content": "Nice to meet you, Parshva!"},
            {"role": "user", "content": "I love playing cricket on weekends and I am a huge fan of Virat Kohli."},
        ])
        m.add([
            {"role": "user", "content": "I recently moved to Bangalore and I prefer South Indian food, especially dosas."},
            {"role": "user", "content": "I am working on building a memory system for AI agents. My goal is to make it lightweight and fast."},
        ])
        yield m
        m.close()


@pytest.fixture(scope="module")
def openai_client():
    from openai import OpenAI
    return OpenAI()
