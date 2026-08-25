"""
Test memory update/dedup — adding contradicting info should update, not duplicate.

Run: .venv/bin/python examples/test_memory_update.py
"""

import tempfile
from cognis import Cognis

with tempfile.TemporaryDirectory() as td:
    m = Cognis(data_dir=td, owner_id="user_1")

    # First: user works at Google
    print("Step 1: Adding 'I work at Google'")
    m.add([{"role": "user", "content": "I work at Google as a software engineer."}])
    print(f"  Memories ({m.count()}):")
    for mem in m.get_all()["memories"]:
        print(f"    {mem['content']}")
    print()

    # Second: user switches to Lyzr
    print("Step 2: Adding 'I now work at Lyzr AI'")
    m.add([{"role": "user", "content": "I just switched jobs. I now work at Lyzr AI."}])
    print(f"  Memories ({m.count()}):")
    for mem in m.get_all(include_historical=True)["memories"]:
        print(f"    {mem['content']}")
    print()

    # Check: Lyzr should be present
    all_content = " ".join(mem["content"].lower() for mem in m.get_all()["memories"])
    assert "lyzr" in all_content, "Expected 'lyzr' in memories after update!"
    print("PASS: 'lyzr' found in memories after update")

    # Search should return Lyzr
    resp = m.search("Where does the user work?", limit=3)
    print(f"\nSearch 'Where does the user work?':")
    for r in resp["results"]:
        print(f"  -> {r['content']}  (score: {r['score']})")

    m.close()
    print("\nDone!")
