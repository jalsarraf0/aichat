"""
Tests for the conversation persistence feature.

Sections:
  1. Pure unit tests  — no services needed; always run
  2. Source inspection — grep for required symbols/strings; always run
  3. DB integration   — probe the live aichat-database service; skipped when offline
"""
from __future__ import annotations

import sys
import uuid

import httpx
import pytest

sys.path.insert(0, "src")


# ---------------------------------------------------------------------------
# DB availability helper
# ---------------------------------------------------------------------------

def _is_database_up() -> bool:
    try:
        r = httpx.get("http://localhost:8091/health", timeout=2)
        if r.status_code != 200:
            return False
        data = r.json()
        return "articles" in data and "cached_pages" in data
    except Exception:
        return False


_DB_UP = _is_database_up()

skip_db = pytest.mark.skipif(not _DB_UP, reason="aichat-database not reachable (Docker not running?)")


# ===========================================================================
# 1. Pure unit tests (no services)
# ===========================================================================

class TestConversationStoreImportable:
    def test_conversation_store_importable(self):
        from aichat.tools.conversation_store import ConversationStoreTool
        tool = ConversationStoreTool()
        assert hasattr(tool, "create_session")
        assert hasattr(tool, "store_turn")
        assert hasattr(tool, "search_turns")
        assert hasattr(tool, "list_sessions")
        assert hasattr(tool, "get_session")
        assert hasattr(tool, "update_title")

    def test_all_methods_are_coroutines(self):
        import inspect
        from aichat.tools.conversation_store import ConversationStoreTool
        tool = ConversationStoreTool()
        for method_name in ("create_session", "store_turn", "search_turns",
                            "list_sessions", "get_session", "update_title"):
            assert inspect.iscoroutinefunction(getattr(tool, method_name)), \
                f"{method_name} must be async"


class TestStateNewFields:
    def test_state_has_session_id(self):
        from aichat.state import AppState
        s = AppState()
        assert s.session_id == ""

    def test_state_has_rag_flag(self):
        from aichat.state import AppState
        s = AppState()
        assert s.rag_context_enabled is True

    def test_max_tool_calls_per_turn_is_6(self):
        from aichat.state import AppState
        s = AppState()
        assert s.max_tool_calls_per_turn == 6


class TestManagerConvIntegration:
    def test_conv_in_manager(self):
        from aichat.tools.manager import ToolManager
        from aichat.tools.conversation_store import ConversationStoreTool
        tm = ToolManager()
        assert hasattr(tm, "conv")
        assert isinstance(tm.conv, ConversationStoreTool)

    def test_toolname_conv_search_exists(self):
        from aichat.tools.manager import ToolName
        assert hasattr(ToolName, "CONV_SEARCH_HISTORY")
        assert ToolName.CONV_SEARCH_HISTORY.value == "conv_search_history"

    def test_run_conv_search_history_exists(self):
        import inspect
        from aichat.tools.manager import ToolManager
        tm = ToolManager()
        assert hasattr(tm, "run_conv_search_history")
        assert inspect.iscoroutinefunction(tm.run_conv_search_history)


# ===========================================================================
# 2. Source inspection tests (no services, grep the source files)
# ===========================================================================

class TestDatabaseAppSource:
    def _read(self, relpath: str) -> str:
        import pathlib
        root = pathlib.Path(__file__).parent.parent
        return (root / relpath).read_text()

    def test_conv_session_table_in_database_app(self):
        src = self._read("docker/database/app.py")
        assert "conversation_sessions" in src, \
            "conversation_sessions table not found in docker/database/app.py"

    def test_conv_turns_table_in_database_app(self):
        src = self._read("docker/database/app.py")
        assert "conversation_turns" in src, \
            "conversation_turns table not found in docker/database/app.py"

    def test_conv_endpoints_in_database_app(self):
        src = self._read("docker/database/app.py")
        required_routes = [
            "/conversations/sessions",
            "/conversations/turns",
            "/conversations/search",
        ]
        for route in required_routes:
            assert route in src, f"Route '{route}' not found in docker/database/app.py"


class TestAppSource:
    def _read(self) -> str:
        import pathlib
        root = pathlib.Path(__file__).parent.parent
        return (root / "src/aichat/app.py").read_text()

    def test_auto_save_turn_in_app(self):
        assert "_auto_save_turn" in self._read(), \
            "_auto_save_turn method not found in app.py"

    def test_fetch_rag_context_in_app(self):
        assert "_fetch_rag_context" in self._read(), \
            "_fetch_rag_context method not found in app.py"

    def test_history_command_in_app(self):
        src = self._read()
        assert "_handle_history_command" in src, \
            "_handle_history_command not found in app.py"
        assert '"/history"' in src or "startswith(\"/history\")" in src, \
            "/history command dispatch not found in app.py"

    def test_sessions_command_in_app(self):
        src = self._read()
        assert "_handle_sessions_command" in src, \
            "_handle_sessions_command not found in app.py"

    def test_context_command_in_app(self):
        src = self._read()
        assert "_handle_context_toggle" in src, \
            "_handle_context_toggle not found in app.py"

    def test_llm_messages_is_async(self):
        src = self._read()
        assert "async def _llm_messages" in src, \
            "_llm_messages must be declared async in app.py"

    def test_status_throttle_in_app(self):
        src = self._read()
        assert "_STATUS_CACHE_TTL" in src, \
            "Status throttle (_STATUS_CACHE_TTL) not found in app.py"
        assert "_last_status_ts" in src, \
            "_last_status_ts not found in app.py"

    def test_session_refresh_throttle_in_app(self):
        src = self._read()
        assert "_last_refresh_ts" in src, \
            "Session refresh throttle (_last_refresh_ts) not found in app.py"

    def test_auto_generate_title_in_app(self):
        assert "_auto_generate_title" in self._read(), \
            "_auto_generate_title not found in app.py"

    def test_session_uuid_set_in_new_chat(self):
        src = self._read()
        assert "session_id" in src and "_uuid.uuid4" in src, \
            "session_id UUID generation not found in app.py"


# ===========================================================================
# 3. DB integration tests (require live aichat-database service)
# ===========================================================================

@skip_db
class TestConversationDBIntegration:
    """Integration tests hitting the real aichat-database service at :8091."""

    BASE = "http://localhost:8091"
    TIMEOUT = 5.0

    def _post(self, path: str, **kwargs) -> dict:
        r = httpx.post(f"{self.BASE}{path}", timeout=self.TIMEOUT, **kwargs)
        r.raise_for_status()
        return r.json()

    def _get(self, path: str, **kwargs) -> dict:
        r = httpx.get(f"{self.BASE}{path}", timeout=self.TIMEOUT, **kwargs)
        r.raise_for_status()
        return r.json()

    def _patch(self, path: str, **kwargs) -> dict:
        r = httpx.patch(f"{self.BASE}{path}", timeout=self.TIMEOUT, **kwargs)
        r.raise_for_status()
        return r.json()

    def test_create_session(self):
        sid = f"test-{uuid.uuid4()}"
        data = self._post("/conversations/sessions",
                          json={"session_id": sid, "title": "Test session", "model": "test-model"})
        assert data["status"] in ("created", "exists")
        assert data["session_id"] == sid

    def test_create_session_idempotent(self):
        sid = f"test-{uuid.uuid4()}"
        d1 = self._post("/conversations/sessions", json={"session_id": sid})
        d2 = self._post("/conversations/sessions", json={"session_id": sid})
        assert d1["status"] == "created"
        assert d2["status"] == "exists"
        assert d1["session_id"] == d2["session_id"] == sid

    def test_store_turn_basic(self):
        sid = f"test-{uuid.uuid4()}"
        self._post("/conversations/sessions", json={"session_id": sid})
        data = self._post("/conversations/turns",
                          json={"session_id": sid, "role": "user",
                                "content": "Hello world", "turn_index": 0})
        assert data["status"] == "stored"
        assert data["turn_id"] is not None

    def test_store_turn_with_embedding(self):
        sid = f"test-{uuid.uuid4()}"
        self._post("/conversations/sessions", json={"session_id": sid})
        embedding = [0.1] * 64
        data = self._post("/conversations/turns",
                          json={"session_id": sid, "role": "user",
                                "content": "With embedding", "turn_index": 0,
                                "embedding": embedding})
        assert data["status"] == "stored"

    def test_search_turns_by_embedding(self):
        sid = f"test-{uuid.uuid4()}"
        self._post("/conversations/sessions", json={"session_id": sid})
        # Store two turns with different embeddings
        emb_a = [1.0] + [0.0] * 63   # close to query
        emb_b = [0.0] * 63 + [1.0]   # far from query
        self._post("/conversations/turns",
                   json={"session_id": sid, "role": "user",
                         "content": "Close match", "turn_index": 0, "embedding": emb_a})
        self._post("/conversations/turns",
                   json={"session_id": sid, "role": "assistant",
                         "content": "Far match", "turn_index": 1, "embedding": emb_b})
        query = [1.0] + [0.0] * 63
        result = self._post("/conversations/search",
                            json={"embedding": query, "limit": 2})
        assert "results" in result
        results = result["results"]
        assert len(results) >= 1
        # First result should be the closer one
        assert results[0]["content"] == "Close match"

    def test_search_excludes_current_session(self):
        marker = str(uuid.uuid4())  # unique per test run to avoid cross-run contamination
        sid_current = f"test-{uuid.uuid4()}"
        sid_other = f"test-{uuid.uuid4()}"
        self._post("/conversations/sessions", json={"session_id": sid_current})
        self._post("/conversations/sessions", json={"session_id": sid_other})
        emb = [1.0] + [0.0] * 63
        content_current = f"Current-{marker}"
        content_other = f"Other-{marker}"
        # Store in current session (should be excluded)
        self._post("/conversations/turns",
                   json={"session_id": sid_current, "role": "user",
                         "content": content_current, "turn_index": 0, "embedding": emb})
        # Store in other session (should appear)
        self._post("/conversations/turns",
                   json={"session_id": sid_other, "role": "user",
                         "content": content_other, "turn_index": 0, "embedding": emb})
        result = self._post("/conversations/search",
                            json={"embedding": emb, "limit": 100,
                                  "exclude_session": sid_current})
        contents = [r["content"] for r in result["results"]]
        # Other session turn must appear; current session turn must NOT appear
        assert content_other in contents, f"Expected '{content_other}' in results: {contents}"
        # Verify that turns with sid_current are excluded from results
        session_ids = [r["session_id"] for r in result["results"]]
        assert sid_current not in session_ids, f"sid_current should be excluded but found in: {session_ids}"

    def test_list_sessions(self):
        sid = f"test-{uuid.uuid4()}"
        self._post("/conversations/sessions", json={"session_id": sid, "title": "Listed"})
        data = self._get("/conversations/sessions", params={"limit": 50})
        assert "sessions" in data
        sids = [s["session_id"] for s in data["sessions"]]
        assert sid in sids

    def test_get_session_with_turns(self):
        sid = f"test-{uuid.uuid4()}"
        self._post("/conversations/sessions", json={"session_id": sid, "title": "Detail test"})
        self._post("/conversations/turns",
                   json={"session_id": sid, "role": "user",
                         "content": "First turn", "turn_index": 0})
        self._post("/conversations/turns",
                   json={"session_id": sid, "role": "assistant",
                         "content": "Second turn", "turn_index": 1})
        data = self._get(f"/conversations/sessions/{sid}")
        assert data["session_id"] == sid
        assert data["title"] == "Detail test"
        assert "turns" in data
        roles = [t["role"] for t in data["turns"]]
        assert "user" in roles
        assert "assistant" in roles

    def test_update_session_title(self):
        sid = f"test-{uuid.uuid4()}"
        self._post("/conversations/sessions", json={"session_id": sid, "title": "Old title"})
        upd = self._patch(f"/conversations/sessions/{sid}/title",
                          json={"title": "New shiny title"})
        assert upd["status"] == "updated"
        data = self._get(f"/conversations/sessions/{sid}")
        assert data["title"] == "New shiny title"

    @pytest.mark.asyncio
    async def test_conversation_store_tool_create_session(self):
        from aichat.tools.conversation_store import ConversationStoreTool
        tool = ConversationStoreTool()
        sid = f"test-{uuid.uuid4()}"
        result = await tool.create_session(sid, title="OOP test", model="test")
        assert result.get("session_id") == sid

    @pytest.mark.asyncio
    async def test_conversation_store_tool_store_and_search(self):
        from aichat.tools.conversation_store import ConversationStoreTool
        tool = ConversationStoreTool()
        sid = f"test-{uuid.uuid4()}"
        await tool.create_session(sid)
        emb = [1.0] + [0.0] * 63
        result = await tool.store_turn(sid, "user", "OOP store test", turn_index=0, embedding=emb)
        assert result.get("status") == "stored"
        # Search should find it (not excluding this session)
        results = await tool.search_turns(emb, limit=5)
        contents = [r["content"] for r in results]
        assert "OOP store test" in contents

    @pytest.mark.asyncio
    async def test_conversation_store_tool_list_sessions(self):
        from aichat.tools.conversation_store import ConversationStoreTool
        tool = ConversationStoreTool()
        sid = f"test-{uuid.uuid4()}"
        await tool.create_session(sid, title="List OOP")
        sessions = await tool.list_sessions(50)
        sids = [s["session_id"] for s in sessions]
        assert sid in sids

    @pytest.mark.asyncio
    async def test_conversation_store_tool_update_title(self):
        from aichat.tools.conversation_store import ConversationStoreTool
        tool = ConversationStoreTool()
        sid = f"test-{uuid.uuid4()}"
        await tool.create_session(sid, title="Before")
        upd = await tool.update_title(sid, "After")
        assert upd.get("status") == "updated"
        data = await tool.get_session(sid)
        assert data.get("title") == "After"
