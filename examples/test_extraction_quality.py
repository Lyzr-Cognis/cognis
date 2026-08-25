"""
Test extraction quality with GPT-4.1-mini as LLM judge.

Feeds conversations, extracts memories, judges each extraction.
Run: .venv/bin/python examples/test_extraction_quality.py
"""

import json
import tempfile
from openai import OpenAI
from cognis import Cognis

JUDGE = OpenAI()
JUDGE_MODEL = "gpt-4.1-mini"


def judge_extraction(conversation, expected_facts, extracted_memories):
    prompt = f"""Grade whether extracted memories capture expected facts.

CONVERSATION: {conversation}
EXPECTED: {json.dumps(expected_facts)}
EXTRACTED: {json.dumps(extracted_memories)}

Grade A=all captured, B=some missing, C=most missing.
Return JSON: {{"grade": "A/B/C", "missing": [], "reason": "..."}}"""

    resp = JUDGE.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content.strip())


test_cases = [
    {
        "name": "Identity",
        "messages": [{"role": "user", "content": "My name is Parshva, I'm 28, and I work at Lyzr AI."}],
        "expected": ["User's name is Parshva", "User is 28 years old", "User works at Lyzr AI"],
    },
    {
        "name": "Preferences",
        "messages": [{"role": "user", "content": "I love cricket, I'm a fan of Virat Kohli, and I enjoy sci-fi novels."}],
        "expected": ["User loves cricket", "User is a fan of Virat Kohli", "User enjoys sci-fi novels"],
    },
    {
        "name": "Food & Location",
        "messages": [{"role": "user", "content": "I moved to Bangalore and I prefer South Indian food, especially dosas."}],
        "expected": ["User lives in Bangalore", "User prefers South Indian food", "User likes dosas"],
    },
    {
        "name": "Professional",
        "messages": [{"role": "user", "content": "I'm building a memory system for AI agents using Python and FastAPI."}],
        "expected": ["User is building a memory system", "User uses Python", "User uses FastAPI"],
    },
    {
        "name": "Multi-turn",
        "messages": [
            {"role": "user", "content": "I'm planning a trip to Japan."},
            {"role": "assistant", "content": "Which cities?"},
            {"role": "user", "content": "Tokyo and Kyoto. I want to try ramen."},
        ],
        "expected": ["User planning trip to Japan", "User visiting Tokyo", "User visiting Kyoto", "User wants ramen"],
    },
    {
        "name": "Greetings (should extract nothing)",
        "messages": [
            {"role": "user", "content": "Hello! How are you?"},
            {"role": "user", "content": "Thanks, just saying hi!"},
        ],
        "expected": [],
    },
]

print("=" * 60)
print("EXTRACTION QUALITY TEST (GPT-4.1-mini judge)")
print("=" * 60)

grades = {"A": 0, "B": 0, "C": 0}

for tc in test_cases:
    with tempfile.TemporaryDirectory() as td:
        m = Cognis(data_dir=td, owner_id="test")
        m.add(tc["messages"])
        extracted = [mem["content"] for mem in m.get_all()["memories"]]
        m.close()

    print(f"\n[{tc['name']}]")
    print(f"  Extracted: {extracted}")

    if not tc["expected"]:
        grade = "A" if len(extracted) <= 1 else "B"
        print(f"  Grade: {grade} (expected 0-1 facts, got {len(extracted)})")
        grades[grade] += 1
        continue

    v = judge_extraction(
        "\n".join(f"[{msg['role']}] {msg['content']}" for msg in tc["messages"]),
        tc["expected"],
        extracted,
    )
    print(f"  Grade: {v['grade']} | Missing: {v.get('missing',[])} | {v['reason']}")
    grades[v["grade"]] += 1

print("\n" + "=" * 60)
total = sum(grades.values())
print(f"Results: A={grades['A']} B={grades['B']} C={grades['C']}  ({grades['A']}/{total} perfect)")
print("=" * 60)
