"""
Benchmark search latency — measures per-query timing breakdown.

Run: .venv/bin/python examples/test_search_latency.py
"""

import time
import tempfile
from cognis import Cognis

with tempfile.TemporaryDirectory() as td:
    m = Cognis(data_dir=td, owner_id="bench_user")

    # Seed with memories
    print("Seeding memories...")
    m.add([
        {"role": "user", "content": "My name is Parshva. I work at Lyzr AI as a software engineer."},
        {"role": "user", "content": "I love playing cricket and I'm a fan of Virat Kohli."},
        {"role": "user", "content": "I live in Bangalore and prefer South Indian food like dosas."},
        {"role": "user", "content": "I'm building a lightweight memory system for AI agents."},
    ])
    print(f"Seeded {m.count()} memories\n")

    queries = [
        "What is the user's name?",
        "Where does the user work?",
        "What sport does the user play?",
        "What food does the user like?",
        "What is the user building?",
        "Where does the user live?",
        "Who is the user's favorite cricketer?",
        "What is the user's profession?",
    ]

    print(f"{'Query':<45} {'Latency':>8}")
    print("-" * 55)

    latencies = []
    for q in queries:
        t = time.time()
        resp = m.search(q, limit=5)
        lat = (time.time() - t) * 1000
        latencies.append(lat)
        hits = resp["results"]
        top = hits[0]["content"][:35] if hits else "(none)"
        print(f"{q:<45} {lat:>6.0f}ms  -> {top}")

    print("-" * 55)
    print(f"{'Avg':<45} {sum(latencies)/len(latencies):>6.0f}ms")
    print(f"{'Min':<45} {min(latencies):>6.0f}ms")
    print(f"{'Max':<45} {max(latencies):>6.0f}ms")

    # Cached queries (should be near-instant)
    print("\nCached queries (repeat):")
    for q in queries[:3]:
        t = time.time()
        m.search(q, limit=5)
        lat = (time.time() - t) * 1000
        print(f"  {q:<45} {lat:>6.0f}ms")

    print("\nAll memories:")
    for mem in m.get_all()["memories"]:
        cat = mem["metadata"]["category"]
        print(f"  [{cat:<12}] {mem['content']}")

    m.close()
    print("\nDone!")
