"""
Test multi-user isolation — each owner's memories are separate.
Uses a single Cognis instance with owner_id overrides per call.

Run: .venv/bin/python examples/test_multi_user.py
"""

import tempfile
from cognis import Cognis

with tempfile.TemporaryDirectory() as td:
    # Single instance, switch owners per call
    m = Cognis(data_dir=td, owner_id="alice")

    # Alice's memories
    m.add([{"role": "user", "content": "My name is Alice and I'm a data scientist."}], owner_id="alice")

    # Bob's memories (override owner_id)
    m.add([{"role": "user", "content": "My name is Bob and I'm a product manager."}], owner_id="bob")

    print(f"Alice memories: {m.count(owner_id='alice')}")
    for mem in m.get_all(owner_id="alice")["memories"]:
        print(f"  - {mem['content']}")

    print(f"\nBob memories: {m.count(owner_id='bob')}")
    for mem in m.get_all(owner_id="bob")["memories"]:
        print(f"  - {mem['content']}")

    # Alice should NOT see Bob's memories in search
    alice_resp = m.search("product manager", limit=5, owner_id="alice")
    print(f"\nAlice search 'product manager': {alice_resp['count']} results")
    for r in alice_resp["results"]:
        print(f"  -> {r['content']}")

    bob_resp = m.search("data scientist", limit=5, owner_id="bob")
    print(f"\nBob search 'data scientist': {bob_resp['count']} results")
    for r in bob_resp["results"]:
        print(f"  -> {r['content']}")

    m.close()
    print("\nDone!")
