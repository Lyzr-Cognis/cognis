"""
Test session handling — separate sessions, session switching.

Run: .venv/bin/python examples/test_session_handling.py
"""

import tempfile
from cognis import Cognis

with tempfile.TemporaryDirectory() as td:
    m = Cognis(data_dir=td, owner_id="user_1")

    # Session 1
    s1 = m.session_id
    print(f"Session 1: {s1}")
    m.add([{"role": "user", "content": "I'm working on a Python project today."}])

    # Session 2
    s2 = m.new_session()
    print(f"Session 2: {s2}")
    m.add([{"role": "user", "content": "Now I'm cooking pasta for dinner."}])

    print(f"\nAll memories across sessions ({m.count()}):")
    for mem in m.get_all()["memories"]:
        print(f"  [{mem.get('session_id', '?')[:12]}] {mem['content']}")

    # Search — memories are global, should find across sessions
    print(f"\nSearch 'cooking':")
    resp = m.search("cooking", limit=3)
    for r in resp["results"]:
        print(f"  -> {r['content']}")

    # Messages per session
    msgs_s1 = m._sqlite.get_messages(m.owner_id, s1)
    msgs_s2 = m._sqlite.get_messages(m.owner_id, s2)
    print(f"\nSession 1 messages: {len(msgs_s1)}")
    print(f"Session 2 messages: {len(msgs_s2)}")

    m.close()
    print("\nDone!")
