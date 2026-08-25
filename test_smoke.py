"""
Cognis Smoke Test — end-to-end validation with GPT-4.1-mini LLM-as-judge.

Run: .venv/bin/python test_smoke.py
"""

import json
import time
import tempfile
import warnings
import os
import sys

warnings.filterwarnings("ignore")

from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), "..", "lyzr-memory", ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)

from openai import OpenAI
from cognis import Cognis

PASS = 0
FAIL = 0
LLM_JUDGE = OpenAI()
JUDGE_MODEL = "gpt-4.1-mini"


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def llm_judge(question: str, expected_answer: str, retrieved_memories: list[dict]) -> dict:
    """
    Use GPT-4.1-mini to grade whether retrieved memories can answer the question.
    Returns {"grade": "A"|"B"|"C", "reason": "..."}
      A = CORRECT — memories contain enough info to answer
      B = INCORRECT — memories are present but wrong/irrelevant
      C = NOT_ATTEMPTED — no useful memories retrieved
    """
    memories_text = "\n".join(
        f"- {m['content']} (score: {m.get('score', 'N/A')})" for m in retrieved_memories
    )

    prompt = f"""You are a strict memory retrieval evaluator. Grade whether the retrieved memories
can correctly answer the question.

QUESTION: {question}
EXPECTED ANSWER: {expected_answer}

RETRIEVED MEMORIES:
{memories_text}

GRADING:
- Grade A (CORRECT): The retrieved memories contain the information needed to answer the question correctly.
  The expected answer's key facts must be present in at least one memory.
- Grade B (INCORRECT): Memories were retrieved but they do NOT contain the right answer.
- Grade C (NOT_ATTEMPTED): No relevant memories were retrieved at all.

Return ONLY valid JSON:
{{"grade": "A or B or C", "reason": "brief explanation"}}"""

    response = LLM_JUDGE.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    text = response.choices[0].message.content.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"grade": "C", "reason": f"Judge parse error: {text[:100]}"}


def main():
    global PASS, FAIL

    with tempfile.TemporaryDirectory() as td:
        print("=" * 70)
        print("  COGNIS SMOKE TEST  (LLM Judge: gpt-4.1-mini)")
        print("=" * 70)
        print()

        # ── 1. Initialization ────────────────────────────────────────
        print("[1] Initialization")
        t0 = time.time()
        memory = Cognis(data_dir=td, owner_id="test_user", agent_id="test_agent")
        init_time = time.time() - t0
        check("Cognis initializes", memory is not None)
        check("Init < 5s", init_time < 5, f"took {init_time:.1f}s")
        check("Count starts at 0", memory.count() == 0)
        print()

        # ── 2. Add conversation 1 ───────────────────────────────────
        print("[2] Add conversation 1 (identity + interests)")
        t1 = time.time()
        r1 = memory.add([
            {"role": "user", "content": "Hi! My name is Parshva and I am a software engineer at Lyzr AI."},
            {"role": "assistant", "content": "Nice to meet you, Parshva! What do you like to do?"},
            {"role": "user", "content": "I love playing cricket on weekends and I am a huge fan of Virat Kohli."},
        ])
        print(f"  Extracted {r1['memory_count']} memories in {time.time()-t1:.1f}s")
        check("message_count = 3", r1["message_count"] == 3)
        check("Extracted >= 3 memories", r1["memory_count"] >= 3, f"got {r1['memory_count']}")
        check("IDs are mem_ format", all(mid.startswith("mem_") for mid in r1["memory_ids"]))
        print()

        # ── 3. Add conversation 2 ───────────────────────────────────
        print("[3] Add conversation 2 (location + food + work)")
        t2 = time.time()
        r2 = memory.add([
            {"role": "user", "content": "I recently moved to Bangalore and I prefer South Indian food, especially dosas."},
            {"role": "user", "content": "I am working on building a memory system for AI agents. My goal is to make it lightweight and fast."},
        ])
        print(f"  Extracted {r2['memory_count']} memories in {time.time()-t2:.1f}s")
        check("Extracted >= 2 memories", r2["memory_count"] >= 2, f"got {r2['memory_count']}")
        print()

        # ── 4. All memories ──────────────────────────────────────────
        print("[4] All extracted memories")
        all_mem = memory.get_all()
        total = memory.count()
        check("count() matches get_all()", total == len(all_mem))
        check("Total >= 5 memories", total >= 5, f"got {total}")
        for m in all_mem:
            print(f"    {m['content']}")
        print(f"  Total: {total}")
        print()

        # ── 5. Get single memory ─────────────────────────────────────
        print("[5] Get / Get non-existent")
        first_id = all_mem[0]["memory_id"]
        got = memory.get(first_id)
        check("get() returns memory", got is not None)
        check("Content matches", got["content"] == all_mem[0]["content"])
        check("get(bad_id) returns None", memory.get("mem_doesnotexist") is None)
        print()

        # ── 6. Search with LLM Judge ─────────────────────────────────
        print("[6] Search quality (GPT-4.1-mini judge)")
        print("-" * 70)

        qa_pairs = [
            ("What is the user's name?", "Parshva"),
            ("Where does the user work?", "Lyzr AI"),
            ("What is the user's profession?", "Software engineer"),
            ("What sport does the user play?", "Cricket"),
            ("What food does the user prefer?", "South Indian food, dosas"),
            ("Where does the user live?", "Bangalore"),
            ("What is the user building?", "A memory system for AI agents"),
            ("Who is the user a fan of?", "Virat Kohli"),
        ]

        grades = {"A": 0, "B": 0, "C": 0}
        latencies = []

        for question, expected in qa_pairs:
            t = time.time()
            results = memory.search(question, limit=5)
            lat = (time.time() - t) * 1000
            latencies.append(lat)

            verdict = llm_judge(question, expected, results)
            grade = verdict.get("grade", "C")
            reason = verdict.get("reason", "")
            grades[grade] = grades.get(grade, 0) + 1

            grade_icon = {"A": "CORRECT", "B": "WRONG", "C": "MISSING"}.get(grade, "?")
            check(
                f"Q: '{question[:45]}' -> {grade_icon}",
                grade == "A",
                reason,
            )
            top = results[0]["content"][:55] if results else "(none)"
            print(f"    Top result: {top}  ({lat:.0f}ms)")

        avg_lat = sum(latencies) / len(latencies)
        accuracy = grades["A"] / len(qa_pairs) * 100

        print()
        print(f"  Judge results:  A={grades['A']}  B={grades['B']}  C={grades['C']}")
        print(f"  Accuracy: {accuracy:.0f}% ({grades['A']}/{len(qa_pairs)})")
        print(f"  Avg search latency: {avg_lat:.0f}ms")
        print()

        # ── 7. Get context ───────────────────────────────────────────
        print("[7] Get context")
        ctx = memory.get_context([{"role": "user", "content": "Tell me about myself"}])
        check("short_term_count > 0", ctx["short_term_count"] > 0)
        check("long_term_count > 0", ctx["long_term_count"] > 0)
        check("context_string not empty", len(ctx["context_string"]) > 0)
        print(f"  Short-term: {ctx['short_term_count']} msgs, Long-term: {ctx['long_term_count']} mems")
        print()

        # ── 8. Delete ────────────────────────────────────────────────
        print("[8] Delete")
        del_id = all_mem[0]["memory_id"]
        count_before = memory.count()
        deleted = memory.delete(del_id)
        check("delete() returns True", deleted)
        check("Count decreased", memory.count() == count_before - 1)
        check("get() returns None", memory.get(del_id) is None)
        print()

        # ── 9. Session management ────────────────────────────────────
        print("[9] Sessions")
        old = memory.session_id
        new = memory.new_session()
        check("new_session() returns ses_", new.startswith("ses_"))
        check("Session changed", new != old)
        print()

        # ── 10. Clear ────────────────────────────────────────────────
        print("[10] Clear")
        memory.clear()
        check("All cleared", memory.count() == 0)
        print()

        memory.close()

        # ── Summary ──────────────────────────────────────────────────
        print("=" * 70)
        print(f"  RESULTS: {PASS} passed, {FAIL} failed out of {PASS + FAIL}")
        print(f"  JUDGE ACCURACY: {accuracy:.0f}% ({grades['A']}/{len(qa_pairs)})")
        print(f"  AVG SEARCH LATENCY: {avg_lat:.0f}ms")
        print("=" * 70)

        if FAIL > 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
