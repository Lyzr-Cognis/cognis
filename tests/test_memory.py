"""Tests for core Cognis memory class — add, get, delete, clear, sessions."""


class TestAdd:
    def test_add_returns_hosted_format(self, memory):
        result = memory.add([{"role": "user", "content": "My name is Alice and I work at Google."}])
        assert result["success"] is True
        assert "session_message_count" in result
        assert isinstance(result["memories"], list)
        assert len(result["memories"]) >= 1
        assert result["memories"][0]["id"].startswith("mem_")

    def test_add_multiple(self, memory):
        result = memory.add([
            {"role": "user", "content": "I live in New York."},
            {"role": "user", "content": "I enjoy running marathons."},
        ])
        assert result["success"] is True
        assert result["session_message_count"] == 2
        assert len(result["memories"]) >= 2


class TestGet:
    def test_get_existing(self, seeded_memory):
        all_resp = seeded_memory.get_all()
        first = all_resp["memories"][0]
        got = seeded_memory.get(first["memory_id"])
        assert got["success"] is True
        assert got["memory"]["content"] == first["content"]

    def test_get_nonexistent(self, seeded_memory):
        got = seeded_memory.get("mem_doesnotexist")
        assert got["success"] is False
        assert got["memory"] is None

    def test_get_all_format(self, seeded_memory):
        resp = seeded_memory.get_all()
        assert resp["success"] is True
        assert isinstance(resp["memories"], list)
        assert len(resp["memories"]) >= 5
        assert resp["total"] >= 5
        assert "limit" in resp
        assert "offset" in resp


class TestDelete:
    def test_delete_returns_format(self, seeded_memory):
        first_id = seeded_memory.get_all()["memories"][0]["memory_id"]
        resp = seeded_memory.delete(first_id)
        assert resp["success"] is True

    def test_delete_nonexistent(self, seeded_memory):
        resp = seeded_memory.delete("mem_doesnotexist")
        assert resp["success"] is False


class TestSessions:
    def test_new_session(self, memory):
        old = memory.session_id
        new = memory.new_session()
        assert new.startswith("ses_")
        assert new != old
        assert memory.session_id == new

    def test_set_owner(self, memory):
        memory.set_owner("new_owner")
        assert memory.owner_id == "new_owner"
