"""
Tests for memory extraction quality — GPT-4.1-mini as LLM judge.

Each test feeds a conversation, extracts memories, then asks GPT-4.1-mini
to grade whether the extracted facts correctly capture expected information.

Grading:
  A = All expected facts captured
  B = Some expected facts missing
  C = Most expected facts missing or extraction failed
"""

import json
import pytest

JUDGE_MODEL = "gpt-4.1-mini"


def _judge(client, conversation: str, expected: list[str], extracted: list[str]) -> dict:
    """Grade extraction quality via GPT-4.1-mini."""
    prompt = f"""Grade whether extracted memories capture expected facts.

CONVERSATION: {conversation}
EXPECTED: {json.dumps(expected)}
EXTRACTED: {json.dumps(extracted)}

Grade A=all captured, B=some missing, C=most missing.
Return JSON: {{"grade": "A/B/C", "missing": [], "reason": "..."}}"""

    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    try:
        return json.loads(resp.choices[0].message.content.strip())
    except json.JSONDecodeError:
        return {"grade": "C", "missing": [], "reason": "parse error"}


class TestExtractionIdentity:
    def test_name_and_job(self, memory, openai_client):
        memory.add([{"role": "user", "content": "My name is Parshva and I am a software engineer at Lyzr AI."}])
        extracted = [m["content"] for m in memory.get_all()["memories"]]

        v = _judge(openai_client,
                   "My name is Parshva and I am a software engineer at Lyzr AI.",
                   ["User's name is Parshva", "User works at Lyzr AI", "User is a software engineer"],
                   extracted)
        print(f"\n  Grade: {v['grade']} | Missing: {v.get('missing',[])} | {v['reason']}")
        assert v["grade"] == "A", f"Got {v['grade']}: {v['reason']}"

    def test_age_and_location(self, memory, openai_client):
        memory.add([{"role": "user", "content": "I am 28 years old and I live in Bangalore."}])
        extracted = [m["content"] for m in memory.get_all()["memories"]]

        v = _judge(openai_client,
                   "I am 28 years old and I live in Bangalore.",
                   ["User is 28 years old", "User lives in Bangalore"],
                   extracted)
        print(f"\n  Grade: {v['grade']} | Missing: {v.get('missing',[])} | {v['reason']}")
        assert v["grade"] == "A", f"Got {v['grade']}: {v['reason']}"


class TestExtractionPreferences:
    def test_hobbies_and_fandom(self, memory, openai_client):
        memory.add([{"role": "user", "content": "I love playing cricket on weekends. I'm a huge fan of Virat Kohli. I also enjoy reading sci-fi novels."}])
        extracted = [m["content"] for m in memory.get_all()["memories"]]

        v = _judge(openai_client,
                   "I love playing cricket on weekends. I'm a huge fan of Virat Kohli. I also enjoy reading sci-fi novels.",
                   ["User loves playing cricket", "User is a fan of Virat Kohli", "User enjoys reading sci-fi novels"],
                   extracted)
        print(f"\n  Grade: {v['grade']} | Missing: {v.get('missing',[])} | {v['reason']}")
        assert v["grade"] == "A", f"Got {v['grade']}: {v['reason']}"

    def test_food_preferences(self, memory, openai_client):
        memory.add([{"role": "user", "content": "I prefer South Indian food, especially dosas and idli. I'm vegetarian."}])
        extracted = [m["content"] for m in memory.get_all()["memories"]]

        v = _judge(openai_client,
                   "I prefer South Indian food, especially dosas and idli. I'm vegetarian.",
                   ["User prefers South Indian food", "User likes dosas", "User is vegetarian"],
                   extracted)
        print(f"\n  Grade: {v['grade']} | Missing: {v.get('missing',[])} | {v['reason']}")
        assert v["grade"] in ("A", "B"), f"Got {v['grade']}: {v['reason']}"


class TestExtractionProfessional:
    def test_project_and_goals(self, memory, openai_client):
        memory.add([{"role": "user", "content": "I am building a memory system for AI agents. I use Python and FastAPI. My goal is to make it lightweight and open source."}])
        extracted = [m["content"] for m in memory.get_all()["memories"]]

        v = _judge(openai_client,
                   "I am building a memory system for AI agents. I use Python and FastAPI. My goal is to make it lightweight and open source.",
                   ["User is building a memory system for AI agents", "User uses Python", "User uses FastAPI", "User wants to make it lightweight and open source"],
                   extracted)
        print(f"\n  Grade: {v['grade']} | Missing: {v.get('missing',[])} | {v['reason']}")
        assert v["grade"] in ("A", "B"), f"Got {v['grade']}: {v['reason']}"


class TestExtractionMultiTurn:
    def test_multi_turn_travel(self, memory, openai_client):
        msgs = [
            {"role": "user", "content": "I'm planning a trip to Japan next month."},
            {"role": "assistant", "content": "Which cities?"},
            {"role": "user", "content": "Tokyo and Kyoto. I want to try authentic ramen and visit temples."},
        ]
        memory.add(msgs)
        extracted = [m["content"] for m in memory.get_all()["memories"]]
        conv = "\n".join(f"[{m['role']}] {m['content']}" for m in msgs)

        v = _judge(openai_client, conv,
                   ["User is planning a trip to Japan", "User wants to visit Tokyo", "User wants to visit Kyoto", "User wants to try ramen"],
                   extracted)
        print(f"\n  Grade: {v['grade']} | Missing: {v.get('missing',[])} | {v['reason']}")
        assert v["grade"] in ("A", "B"), f"Got {v['grade']}: {v['reason']}"


class TestExtractionEdgeCases:
    def test_no_facts_from_greetings(self, memory, openai_client):
        memory.add([
            {"role": "user", "content": "Hello! How are you?"},
            {"role": "user", "content": "Thanks, just checking in!"},
        ])
        extracted = [m["content"] for m in memory.get_all()["memories"]]
        assert len(extracted) <= 1, f"Extracted {len(extracted)} facts from greetings: {extracted}"

    def test_update_replaces_old(self, memory, openai_client):
        memory.add([{"role": "user", "content": "I work at Google as a software engineer."}])
        memory.add([{"role": "user", "content": "I just switched jobs. I now work at Lyzr AI."}])
        all_mem = [m["content"] for m in memory.get_all()["memories"]]
        all_text = " ".join(all_mem).lower()
        assert "lyzr" in all_text, f"Expected 'lyzr' in memories after update: {all_mem}"
