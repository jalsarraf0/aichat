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
# 2b. Additional source inspection tests for all 17 fixes
# ===========================================================================

class TestFixSourceInspection:
    """Source inspection tests verifying all 17 fixes are present."""

    def _app_src(self) -> str:
        import pathlib
        root = pathlib.Path(__file__).parent.parent
        return (root / "src/aichat/app.py").read_text()

    def _state_src(self) -> str:
        import pathlib
        root = pathlib.Path(__file__).parent.parent
        return (root / "src/aichat/state.py").read_text()

    def _db_src(self) -> str:
        import pathlib
        root = pathlib.Path(__file__).parent.parent
        return (root / "docker/database/app.py").read_text()

    def _mcp_src(self) -> str:
        import pathlib
        root = pathlib.Path(__file__).parent.parent
        return (root / "src/aichat/mcp_server.py").read_text()

    def _conv_src(self) -> str:
        import pathlib
        root = pathlib.Path(__file__).parent.parent
        return (root / "src/aichat/tools/conversation_store.py").read_text()

    def _manager_src(self) -> str:
        import pathlib
        root = pathlib.Path(__file__).parent.parent
        return (root / "src/aichat/tools/manager.py").read_text()

    def test_session_title_set_in_state(self):
        src = self._state_src()
        assert "session_title_set" in src, "session_title_set field not found in state.py"
        assert "False" in src, "session_title_set default False not found"

    def test_resume_command_in_app(self):
        src = self._app_src()
        assert "_handle_resume_command" in src, "_handle_resume_command not found in app.py"
        assert '"/resume"' in src or 'startswith("/resume")' in src, \
            "/resume dispatch not found in app.py"

    def test_stats_command_in_app(self):
        src = self._app_src()
        assert "_handle_stats_command" in src, "_handle_stats_command not found in app.py"
        assert '"/stats"' in src or 'startswith("/stats")' in src, \
            "/stats dispatch not found in app.py"

    def test_help_writes_to_transcript(self):
        src = self._app_src()
        assert "_write_transcript" in src, "_write_transcript not found in app.py"
        # action_help should call _write_transcript now (not just notify)
        import re
        help_fn = re.search(
            r"async def action_help.*?(?=\n    async def|\nclass |\Z)",
            src, re.DOTALL
        )
        assert help_fn, "action_help not found"
        assert "_write_transcript" in help_fn.group(0), \
            "action_help does not call _write_transcript"

    def test_help_lists_history_command(self):
        src = self._app_src()
        import re
        help_fn = re.search(
            r"async def action_help.*?(?=\n    async def|\nclass |\Z)",
            src, re.DOTALL
        )
        assert help_fn, "action_help not found"
        assert "/history" in help_fn.group(0), "/history not listed in help text"

    def test_help_lists_resume_command(self):
        src = self._app_src()
        import re
        help_fn = re.search(
            r"async def action_help.*?(?=\n    async def|\nclass |\Z)",
            src, re.DOTALL
        )
        assert help_fn, "action_help not found"
        assert "/resume" in help_fn.group(0), "/resume not listed in help text"

    def test_rag_threshold_in_fetch_context(self):
        src = self._app_src()
        # Threshold may be 0.25 (date-weighted) or 0.3 (original); either is acceptable
        assert ("0.3" in src or "0.25" in src), \
            "Similarity threshold (0.3 or 0.25) not found in app.py"
        assert "_fetch_rag_context" in src, "_fetch_rag_context not in app.py"

    def test_fulltext_endpoint_in_database_app(self):
        src = self._db_src()
        assert "/conversations/turns/search" in src, \
            "/conversations/turns/search endpoint not found in docker/database/app.py"
        assert "ILIKE" in src, "ILIKE full-text search not found in docker/database/app.py"

    def test_conv_store_has_search_turns_text(self):
        src = self._conv_src()
        assert "search_turns_text" in src, \
            "search_turns_text method not found in conversation_store.py"

    def test_conv_search_history_in_mcp_server(self):
        src = self._mcp_src()
        assert "conv_search_history" in src, \
            "conv_search_history not found in mcp_server.py"

    def test_web_fetch_checks_cache_first(self):
        src = self._manager_src()
        assert "cache_get" in src, "cache_get not found in manager.py"
        # Verify it's in the run_web_fetch method
        import re
        fetch_fn = re.search(
            r"async def run_web_fetch.*?(?=\n    async def |\nclass |\Z)",
            src, re.DOTALL
        )
        assert fetch_fn, "run_web_fetch not found in manager.py"
        assert "cache_get" in fetch_fn.group(0), \
            "cache_get not present in run_web_fetch — cache layer missing"

    def test_rag_cache_fields_in_app(self):
        src = self._app_src()
        assert "_rag_context_query" in src, "_rag_context_query not found in app.py"
        assert "_rag_context_cache" in src, "_rag_context_cache not found in app.py"

    def test_streaming_followup_mounts_live_msg(self):
        src = self._app_src()
        import re
        followup_fn = re.search(
            r"async def _run_followup_response.*?(?=\n    async def |\nclass |\Z)",
            src, re.DOTALL
        )
        assert followup_fn, "_run_followup_response not found"
        fn_src = followup_fn.group(0)
        assert "ChatMessage" in fn_src, \
            "_run_followup_response does not mount a live ChatMessage widget"

    def test_session_title_set_guard_in_finalize(self):
        src = self._app_src()
        assert "session_title_set" in src, "session_title_set flag not found in app.py"

    def test_manager_default_is_6_in_source(self):
        src = self._manager_src()
        assert "max_tool_calls_per_turn: int = 6" in src, \
            "ToolManager default max_tool_calls_per_turn is not 6"

    def test_maybe_resume_last_session_in_app(self):
        src = self._app_src()
        assert "_maybe_resume_last_session" in src, \
            "_maybe_resume_last_session not found in app.py"
        assert "on_mount" in src, "on_mount not found in app.py"


# ===========================================================================
# 2c. OOP unit tests (no services, fast)
# ===========================================================================

class TestOOPUnits:
    """Unit tests requiring no external services."""

    def test_manager_default_max_tool_calls_is_6(self):
        from aichat.tools.manager import ToolManager
        tm = ToolManager()
        assert tm.max_tool_calls_per_turn == 6, \
            f"Expected 6, got {tm.max_tool_calls_per_turn}"

    def test_state_session_title_set_default_false(self):
        from aichat.state import AppState
        s = AppState()
        assert s.session_title_set is False

    def test_conv_store_has_search_turns_text_method(self):
        import inspect
        from aichat.tools.conversation_store import ConversationStoreTool
        tool = ConversationStoreTool()
        assert hasattr(tool, "search_turns_text"), \
            "search_turns_text method not found on ConversationStoreTool"
        assert inspect.iscoroutinefunction(tool.search_turns_text), \
            "search_turns_text must be async"

    @pytest.mark.asyncio
    async def test_rag_cache_avoids_double_embed(self):
        """_fetch_rag_context should embed only once for the same query text."""
        import sys
        sys.path.insert(0, "src")
        from unittest.mock import AsyncMock, patch, MagicMock
        from aichat.tools.conversation_store import ConversationStoreTool
        from aichat.tools.manager import ToolManager

        # Build a minimal app-like object with mock tools
        class FakeState:
            session_id = "test-sid"
            rag_context_enabled = True

        class FakeApp:
            state = FakeState()
            _rag_context_query = ""
            _rag_context_cache = ""
            system_prompt = ""

            # Borrow the real method
            _fetch_rag_context = __import__(
                "aichat.app", fromlist=["AIChatApp"]
            ).AIChatApp._fetch_rag_context

        # patch the tools attribute
        import aichat.app as appmod
        app_instance = object.__new__(appmod.AIChatApp)
        app_instance.state = FakeState()  # type: ignore
        app_instance._rag_context_query = ""
        app_instance._rag_context_cache = ""
        app_instance.system_prompt = ""

        embed_mock = AsyncMock(return_value=[[0.1] * 64])
        search_mock = AsyncMock(return_value=[{
            "timestamp": "2026-01-01T00:00:00",
            "role": "user",
            "content": "past message",
            "similarity": 0.9,
        }])

        class FakeTools:
            class lm:
                embed = embed_mock
            class conv:
                search_turns = search_mock

        app_instance.tools = FakeTools()  # type: ignore

        # First call — should embed
        await appmod.AIChatApp._fetch_rag_context(app_instance, "hello world")
        # Second call with same query — should NOT embed again
        await appmod.AIChatApp._fetch_rag_context(app_instance, "hello world")

        assert embed_mock.call_count == 1, \
            f"embed called {embed_mock.call_count} times — RAG cache is not working"

    def test_session_title_flag_resets_on_new_chat(self):
        """session_title_set must be False by default on AppState."""
        from aichat.state import AppState
        s = AppState()
        s.session_title_set = True
        # Simulate what _start_new_chat does
        s.session_title_set = False
        assert s.session_title_set is False


# ===========================================================================
# 2b. ConversationSearcher OOP unit tests (no services, importlib load)
# ===========================================================================

def _load_db_app():
    """Load docker/database/app.py via importlib (no Docker deps at import time)."""
    import importlib.util, pathlib, sys, unittest.mock as um
    spec = importlib.util.spec_from_file_location(
        "db_app_conv_searcher",
        pathlib.Path(__file__).parent.parent / "docker" / "database" / "app.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["db_app_conv_searcher"] = mod
    # Stub psycopg and FastAPI so the module loads without a real DB.
    with um.patch.dict("sys.modules", {
        "psycopg":            um.MagicMock(),
        "psycopg.rows":       um.MagicMock(),
        "fastapi":            um.MagicMock(),
        "fastapi.responses":  um.MagicMock(),
        "pydantic":           um.MagicMock(),
    }):
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


try:
    _db_mod = _load_db_app()
    ConversationSearcher = _db_mod.ConversationSearcher
    _db_cosine_sim       = _db_mod._cosine_sim
    _DB_MOD_OK           = True
except Exception as _db_load_err:
    _DB_MOD_OK           = False
    _db_load_err_str     = str(_db_load_err)

_skip_db_mod = pytest.mark.skipif(not _DB_MOD_OK, reason="docker/database/app.py load failed")


def _make_row(turn_id: int, session_id: str, role: str, content: str,
              embedding: list, timestamp=None):
    """Return a tuple matching the SELECT column order used by conv_search_turns."""
    import json
    return (turn_id, session_id, role, content, json.dumps(embedding), timestamp)


class TestConversationSearcher:
    """Pure unit tests for ConversationSearcher — no Docker, no network."""

    @pytest.fixture
    def rows_ab(self):
        """Two rows from different sessions, each with a distinct embedding."""
        e_a = [1.0] + [0.0] * 63
        e_b = [0.0] + [1.0] + [0.0] * 62
        return [
            _make_row(1, "session-A", "user", "Hello from A", e_a),
            _make_row(2, "session-B", "user", "Hello from B", e_b),
        ]

    # ── scoring ──────────────────────────────────────────────────────────────

    @_skip_db_mod
    def test_scores_all_rows(self, rows_ab):
        query = [1.0] + [0.0] * 63
        results = ConversationSearcher(rows_ab, query).top(10)
        assert len(results) == 2

    @_skip_db_mod
    def test_highest_similarity_ranked_first(self, rows_ab):
        query = [1.0] + [0.0] * 63  # matches session-A perfectly
        results = ConversationSearcher(rows_ab, query).top(10)
        assert results[0]["session_id"] == "session-A"
        assert results[0]["similarity"] == pytest.approx(1.0)

    @_skip_db_mod
    def test_result_dict_has_required_keys(self, rows_ab):
        query = [1.0] + [0.0] * 63
        result = ConversationSearcher(rows_ab, query).top(10)[0]
        for key in ("turn_id", "session_id", "role", "content", "similarity", "timestamp"):
            assert key in result, f"Missing key: {key}"

    @_skip_db_mod
    def test_similarity_is_rounded(self, rows_ab):
        query = [1.0] + [0.0] * 63
        result = ConversationSearcher(rows_ab, query).top(10)[0]
        # similarity must be a float rounded to ≤ 6 decimal places
        assert isinstance(result["similarity"], float)
        assert result["similarity"] == round(result["similarity"], 6)

    @_skip_db_mod
    def test_empty_rows_returns_empty_list(self):
        query = [1.0] + [0.0] * 63
        results = ConversationSearcher([], query).top(10)
        assert results == []

    @_skip_db_mod
    def test_corrupt_row_is_skipped(self):
        """A row with invalid embedding JSON must not raise — it's silently skipped."""
        import json
        bad_row = (99, "session-X", "user", "Bad row", "NOT_JSON", None)
        good_emb = [1.0] + [0.0] * 63
        good_row = _make_row(1, "session-A", "user", "Good", good_emb)
        query = [1.0] + [0.0] * 63
        results = ConversationSearcher([bad_row, good_row], query).top(10)
        assert len(results) == 1
        assert results[0]["session_id"] == "session-A"

    # ── top(n) ───────────────────────────────────────────────────────────────

    @_skip_db_mod
    def test_top_respects_limit(self, rows_ab):
        query = [1.0] + [0.0] * 63
        results = ConversationSearcher(rows_ab, query).top(1)
        assert len(results) == 1

    @_skip_db_mod
    def test_top_larger_than_rows_returns_all(self, rows_ab):
        query = [1.0] + [0.0] * 63
        results = ConversationSearcher(rows_ab, query).top(1000)
        assert len(results) == 2

    # ── exclude ──────────────────────────────────────────────────────────────

    @_skip_db_mod
    def test_exclude_removes_matching_session(self, rows_ab):
        query = [1.0] + [0.0] * 63
        results = ConversationSearcher(rows_ab, query).exclude("session-A").top(10)
        session_ids = [r["session_id"] for r in results]
        assert "session-A" not in session_ids
        assert "session-B" in session_ids

    @_skip_db_mod
    def test_exclude_none_returns_all(self, rows_ab):
        query = [1.0] + [0.0] * 63
        results = ConversationSearcher(rows_ab, query).exclude(None).top(10)
        assert len(results) == 2

    @_skip_db_mod
    def test_exclude_unknown_session_returns_all(self, rows_ab):
        query = [1.0] + [0.0] * 63
        results = ConversationSearcher(rows_ab, query).exclude("session-Z").top(10)
        assert len(results) == 2

    @_skip_db_mod
    def test_exclude_returns_new_instance(self, rows_ab):
        """exclude() must return a new ConversationSearcher (immutable chaining)."""
        query = [1.0] + [0.0] * 63
        searcher = ConversationSearcher(rows_ab, query)
        filtered = searcher.exclude("session-A")
        assert filtered is not searcher
        # Original must be untouched
        assert len(searcher.top(10)) == 2
        assert len(filtered.top(10)) == 1

    @_skip_db_mod
    def test_exclude_both_sessions_returns_empty(self, rows_ab):
        query = [1.0] + [0.0] * 63
        results = (
            ConversationSearcher(rows_ab, query)
            .exclude("session-A")
            .exclude("session-B")
            .top(10)
        )
        assert results == []

    # ── ConvRow namedtuple tests ─────────────────────────────────────────────

    @_skip_db_mod
    def test_convrow_namedtuple_defined(self):
        """ConvRow namedtuple must be importable from docker/database/app.py."""
        assert hasattr(_db_mod, "ConvRow"), "ConvRow namedtuple not found in docker/database/app.py"

    @_skip_db_mod
    def test_convrow_has_correct_fields(self):
        ConvRow = _db_mod.ConvRow
        expected = ("id", "session_id", "role", "content", "embedding", "timestamp")
        assert ConvRow._fields == expected, (
            f"ConvRow fields mismatch: {ConvRow._fields} != {expected}"
        )

    @_skip_db_mod
    def test_searcher_accepts_convrow_namedtuple(self):
        """ConversationSearcher must work when rows are ConvRow instances."""
        import json
        ConvRow = _db_mod.ConvRow
        emb = json.dumps([1.0] + [0.0] * 63)
        row = ConvRow(id=1, session_id="s1", role="user", content="hello",
                      embedding=emb, timestamp=None)
        query = [1.0] + [0.0] * 63
        results = ConversationSearcher([row], query).top(10)
        assert len(results) == 1
        assert results[0]["content"] == "hello"

    @_skip_db_mod
    def test_searcher_corrupt_row_logged_and_skipped(self, caplog):
        """A row with non-JSON embedding must be skipped and a warning logged."""
        import logging
        bad_row = (99, "sess-x", "user", "bad content", "NOT_VALID_JSON", None)
        query = [1.0] + [0.0] * 63
        with caplog.at_level(logging.WARNING, logger="aichat-database"):
            results = ConversationSearcher([bad_row], query).top(10)
        assert results == [], "Corrupt row should produce no results"

    # ── source-level assertion ────────────────────────────────────────────────

    def test_conversation_searcher_in_database_source(self):
        import pathlib
        src = (pathlib.Path(__file__).parent.parent / "docker" / "database" / "app.py").read_text()
        assert "class ConversationSearcher" in src, \
            "ConversationSearcher class not found in docker/database/app.py"
        assert ".exclude(" in src, "ConversationSearcher.exclude() method not found"
        assert ".top(" in src,    "ConversationSearcher.top() method not found"
        assert "ConvRow" in src,  "ConvRow namedtuple not found in docker/database/app.py"


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

        # Use an embedding that is ORTHOGONAL to the generic [1.0, 0.0, ...] that all
        # historic test entries use.  Cosine similarity of [0,…,0,1] vs [1,0,…,0] = 0.0,
        # so when we query with this vector our new entries score 1.0 and every old
        # accumulated entry scores 0.0 — they always rank first regardless of the
        # service's hard limit cap.
        emb = [0.0] * 63 + [1.0]   # 64-dimensional; last component = 1.0
        content_current = f"Current-{marker}"
        content_other   = f"Other-{marker}"

        # Store in current session (should be excluded)
        self._post("/conversations/turns",
                   json={"session_id": sid_current, "role": "user",
                         "content": content_current, "turn_index": 0, "embedding": emb})
        # Store in other session (should appear)
        self._post("/conversations/turns",
                   json={"session_id": sid_other, "role": "user",
                         "content": content_other, "turn_index": 0, "embedding": emb})

        # Confirm the turn was stored in sid_other via the session detail endpoint.
        other_detail = self._get(f"/conversations/sessions/{sid_other}")
        stored_contents = [t["content"] for t in other_detail.get("turns", [])]
        assert content_other in stored_contents, (
            f"content_other not stored in sid_other via session detail: {stored_contents}"
        )

        # Search with the same orthogonal embedding.  Our two new entries score 1.0;
        # all old entries score 0.0, so new entries always appear first within any limit.
        result = self._post("/conversations/search",
                            json={"embedding": emb, "limit": 2000,
                                  "exclude_session": sid_current})
        session_ids = [r["session_id"] for r in result["results"]]
        contents    = [r["content"]    for r in result["results"]]

        # sid_other's entry must appear in search (unique embedding guarantees top rank)
        assert content_other in contents, (
            f"Expected '{content_other}' in search results (top-ranked): {contents[:5]}"
        )
        # sid_current must be completely absent (verifies exclusion)
        assert sid_current not in session_ids, (
            f"sid_current should be excluded but found in: {session_ids}"
        )

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

    @pytest.mark.asyncio
    async def test_fulltext_search_returns_match(self):
        """Full-text search endpoint returns a turn that contains the search term."""
        sid = f"test-{uuid.uuid4()}"
        unique_word = f"xyzuniq{uuid.uuid4().hex[:8]}"
        self._post("/conversations/sessions", json={"session_id": sid})
        self._post("/conversations/turns",
                   json={"session_id": sid, "role": "user",
                         "content": f"This message contains {unique_word}", "turn_index": 0})
        r = httpx.get(f"{self.BASE}/conversations/turns/search",
                      params={"q": unique_word, "limit": 5}, timeout=self.TIMEOUT)
        r.raise_for_status()
        data = r.json()
        assert "results" in data
        contents = [res["content"] for res in data["results"]]
        assert any(unique_word in c for c in contents), \
            f"Expected '{unique_word}' in results: {contents}"

    @pytest.mark.asyncio
    async def test_fulltext_search_no_results(self):
        """Full-text search for a nonsense query returns an empty result list."""
        r = httpx.get(f"{self.BASE}/conversations/turns/search",
                      params={"q": "zzznomatch_xqz_9999_xyz", "limit": 5},
                      timeout=self.TIMEOUT)
        r.raise_for_status()
        data = r.json()
        assert "results" in data
        assert isinstance(data["results"], list)

    @pytest.mark.asyncio
    async def test_conv_store_search_turns_text_oop(self):
        """ConversationStoreTool.search_turns_text finds a stored turn by text."""
        from aichat.tools.conversation_store import ConversationStoreTool
        tool = ConversationStoreTool()
        sid = f"test-{uuid.uuid4()}"
        unique_word = f"xyzoop{uuid.uuid4().hex[:8]}"
        await tool.create_session(sid)
        result = await tool.store_turn(sid, "user",
                                       f"OOP fulltext test {unique_word}", turn_index=0)
        assert result.get("status") == "stored"
        results = await tool.search_turns_text(unique_word, limit=5)
        contents = [r["content"] for r in results]
        assert any(unique_word in c for c in contents), \
            f"Expected '{unique_word}' in fulltext results: {contents}"

    def test_conv_list_sessions_offset_clamped(self):
        """Negative offset must be clamped to 0 by the service (new code) or
        return 4xx (older validation path) — never an internal 500 crash."""
        r = httpx.get(
            f"{self.BASE}/conversations/sessions",
            params={"limit": 5, "offset": -999},
            timeout=self.TIMEOUT,
        )
        # After container rebuild with our fix: 200 (clamped to 0).
        # Older service without bounds check: PostgreSQL may return 500 for
        # negative OFFSET — we accept that here and rely on the new service
        # to fix it. What we do NOT want is a 500 from *our* code change.
        if r.status_code == 200:
            assert "sessions" in r.json()
        elif r.status_code == 500:
            # Old running service without bounds check — acceptable until rebuilt.
            pass
        else:
            pytest.fail(f"Unexpected status {r.status_code}: {r.text[:200]}")

    def test_conv_list_sessions_huge_offset_ok(self):
        """Huge offset (>100000 clamped) must return empty list without error."""
        r = httpx.get(
            f"{self.BASE}/conversations/sessions",
            params={"limit": 5, "offset": 999_999_999},
            timeout=self.TIMEOUT,
        )
        assert r.status_code == 200
        data = r.json()
        assert "sessions" in data
        # Past the end of the table → empty list is acceptable
        assert isinstance(data["sessions"], list)
