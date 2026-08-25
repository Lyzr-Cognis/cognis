"""
Test get_context — short-term messages + long-term memory retrieval.

Run: .venv/bin/python examples/test_context_retrieval.py
"""

import tempfile
from cognis import Cognis

with tempfile.TemporaryDirectory() as td:
    m = Cognis(data_dir=td, owner_id="user_1")

    # Build up some memory
    m.add([
        {"role": "user", "content": "My name is Parshva and I'm a software engineer at Lyzr AI."},
        {"role": "assistant", "content": "Nice to meet you, Parshva!"},
        {"role": "user", "content": "I love cricket and I'm a fan of Virat Kohli."},
        {"role": "user", "content": "I recently moved to Bangalore. I prefer South Indian food."},
    ])

    print(f"Stored {m.count()} memories\n")

    # Get context for a new query
    ctx = m.get_context([{"role": "user", "content": "Tell me everything you know about me"}])

    print(f"Short-term messages: {ctx['short_term_count']}")
    for msg in ctx["short_term"]:
        print(f"  [{msg['role']}] {msg['content'][:60]}")
    print()

    print(f"Long-term memories: {ctx['long_term_count']}")
    for mem in ctx["long_term"]:
        print(f"  - {mem['content']}")
    print()

    print("Context string (ready for LLM):")
    print(ctx["context_string"])

    m.close()
    print("\nDone!")
