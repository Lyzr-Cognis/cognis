"""
Test delete single memory and clear all.

Run: .venv/bin/python examples/test_delete_and_clear.py
"""

import tempfile
from cognis import Cognis

with tempfile.TemporaryDirectory() as td:
    m = Cognis(data_dir=td, owner_id="user_1")

    # Add
    m.add([
        {"role": "user", "content": "I love pizza."},
        {"role": "user", "content": "I work at Google."},
        {"role": "user", "content": "My cat's name is Whiskers."},
    ])
    print(f"After add: {m.count()} memories")
    for mem in m.get_all()["memories"]:
        print(f"  [{mem['memory_id']}] {mem['content']}")

    # Delete one
    target = m.get_all()["memories"][0]
    print(f"\nDeleting: {target['memory_id']} ({target['content']})")
    resp = m.delete(target["memory_id"])
    print(f"  delete(): {resp}")
    print(f"  get() after delete: {m.get(target['memory_id'])}")
    print(f"  Remaining: {m.count()} memories")

    # Clear all
    print("\nClearing all...")
    resp = m.clear()
    print(f"  clear(): {resp}")
    print(f"  Remaining: {m.count()}")
    assert m.count() == 0, "Clear failed!"
    print("  PASS: All cleared")

    m.close()
    print("\nDone!")
