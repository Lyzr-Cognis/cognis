"""
Basic add + search flow using the Cognis SDK.

Run: .venv/bin/python examples/test_basic_add_search.py
"""

import tempfile
from cognis import Cognis

with tempfile.TemporaryDirectory() as td:
    m = Cognis(data_dir=td, owner_id="user_1")

    # Add
    result = m.add([
        {"role": "user", "content": "My name is Daftari and I work at Lyzr AI as a member of technical staff"},
        {"role": "user", "content": "I love watching Formula 1 and I'm a huge fan of Ferrari"},
    ])
    print(f"Added: {result['message']}")
    print()

    # List all
    print("All memories:")
    for mem in m.get_all()["memories"]:
        cat = mem["metadata"]["category"]
        print(f"  [{cat:<12}] {mem['content']}")
    print(f"Total: {m.count()}")
    print()

    # Search
    queries = ["What is the user's name?", "What sport does user watch?", "What is user's job title?", "What is favorite team?"]
    for q in queries:
        resp = m.search(q, limit=5)
        print(f"Q: {q}")
        for r in resp["results"][:2]:
            print(f"  -> {r['content']}  (score: {r['score']})")
        print()
    
    m.close()
    print("Done!")
