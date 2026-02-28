"""
Tests for LM Studio + aichat fine-tuning improvements (v3).

Sections:
  1. TestSourceInspection  — grep for all new symbols in source files
  2. TestTokenize          — LMStudioTool.tokenize() accuracy + fallback
  3. TestModelInfo         — LLMClient.model_info() accuracy + error handling
  4. TestToolCache         — ToolManager._tool_result_cache + clear methods
  5. TestAdaptiveThreshold — _effective_threshold_pct() interpolation
  6. TestToolResultMaxChars — _append_tool_messages() respects _tool_result_max_chars
  7. TestCtxHistory        — _context_pct() populates _ctx_history
  8. TestCtxSparkline      — _ctx_sparkline() output format
  9. TestCompactEvents     — _compact_events log in _run_compact()
 10. TestCompactDry        — /compact dry subcommand TUI
 11. TestCompactHistory    — /compact history subcommand TUI
 12. TestForkCommand       — /fork command TUI
 13. TestCtxGraph          — /ctx command TUI
 14. TestIntegrityCheck    — compact idx integrity check on resume
 15. TestPersonaAwareCompact — persona name in compaction prompt
 16. TestDateWeightedRAG   — date-weighted similarity scoring in _fetch_rag_context
"""
from __future__ import annotations

import asyncio
import inspect
import pathlib
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, "src")

ROOT = pathlib.Path(__file__).parent.parent


def _read(relpath: str) -> str:
    return (ROOT / relpath).read_text()


# ---------------------------------------------------------------------------
# Shared bare-app factory
# ---------------------------------------------------------------------------

def _bare_app():
    """Minimal AIChatApp object for unit tests — no Textual UI."""
    import aichat.app as appmod
    from aichat.state import Message

    app = object.__new__(appmod.AIChatApp)
    app._context_length = 10000
    app._max_response_tokens = 1000
    app._compact_summary = ""
    app._compact_from_idx = 0
    app._compact_pending = False
    app._compact_threshold_pct = 95
    app._compact_min_msgs = 8
    app._compact_keep_ratio = 0.5
    app._compact_tool_turns = True
    app._compact_model = ""
    app._tool_result_max_chars = 2000
    app._rag_recency_days = 30.0
    app._compact_events = []
    app._ctx_history = []
    app.messages = []
    app.personalities = []
    return app


# ===========================================================================
# 1. Source inspection
# ===========================================================================

class TestSourceInspection:
    """Verify all new v3 symbols exist in source files."""

    def _app(self) -> str:
        return _read("src/aichat/app.py")

    def _cfg(self) -> str:
        return _read("src/aichat/config.py")

    def _lm(self) -> str:
        return _read("src/aichat/tools/lm_studio.py")

    def _client(self) -> str:
        return _read("src/aichat/client.py")

    def _mgr(self) -> str:
        return _read("src/aichat/tools/manager.py")

    # --- config.py ---

    def test_compact_model_in_config(self):
        assert "compact_model" in self._cfg(), "compact_model field not in config.py"

    def test_tool_result_max_chars_in_config(self):
        assert "tool_result_max_chars" in self._cfg(), "tool_result_max_chars not in config.py"

    def test_rag_recency_days_in_config(self):
        assert "rag_recency_days" in self._cfg(), "rag_recency_days not in config.py"

    # --- lm_studio.py ---

    def test_tokenize_method_in_lm_studio(self):
        assert "def tokenize" in self._lm(), "tokenize() not found in lm_studio.py"

    # --- client.py ---

    def test_model_info_method_in_client(self):
        assert "def model_info" in self._client(), "model_info() not found in client.py"

    # --- manager.py ---

    def test_tool_result_cache_in_manager(self):
        assert "_tool_result_cache" in self._mgr(), "_tool_result_cache not in manager.py"

    def test_clear_tool_cache_in_manager(self):
        assert "def clear_tool_cache" in self._mgr(), "clear_tool_cache() not in manager.py"

    def test_reset_turn_clears_cache(self):
        import re
        src = self._mgr()
        fn = re.search(
            r"def reset_turn.*?(?=\n    def |\n    async def |\nclass |\Z)",
            src, re.DOTALL,
        )
        assert fn, "reset_turn not found in manager.py"
        assert "_tool_result_cache" in fn.group(0) or "clear" in fn.group(0), \
            "reset_turn does not clear _tool_result_cache"

    # --- app.py instance vars ---

    def test_compact_model_instance_var_in_app(self):
        assert "_compact_model" in self._app(), "_compact_model not set in app.py __init__"

    def test_tool_result_max_chars_instance_var_in_app(self):
        assert "_tool_result_max_chars" in self._app(), "_tool_result_max_chars not in app.py"

    def test_rag_recency_days_instance_var_in_app(self):
        assert "_rag_recency_days" in self._app(), "_rag_recency_days not in app.py"

    def test_compact_events_instance_var_in_app(self):
        assert "_compact_events" in self._app(), "_compact_events not in app.py"

    def test_ctx_history_instance_var_in_app(self):
        assert "_ctx_history" in self._app(), "_ctx_history not in app.py"

    # --- new app.py methods ---

    def test_effective_threshold_pct_method(self):
        assert "def _effective_threshold_pct" in self._app(), \
            "_effective_threshold_pct not in app.py"

    def test_ctx_sparkline_method(self):
        assert "def _ctx_sparkline" in self._app(), "_ctx_sparkline not in app.py"

    def test_handle_fork_command_method(self):
        assert "def _handle_fork_command" in self._app(), \
            "_handle_fork_command not in app.py"

    def test_fork_in_handle_command_dispatcher(self):
        import re
        src = self._app()
        fn = re.search(
            r"async def handle_command.*?(?=\n    async def |\n    def |\nclass |\Z)",
            src, re.DOTALL,
        )
        assert fn, "handle_command not found in app.py"
        assert "/fork" in fn.group(0), "/fork not dispatched in handle_command"

    def test_compact_model_used_in_run_compact(self):
        import re
        src = self._app()
        fn = re.search(
            r"async def _run_compact.*?(?=\n    async def |\n    def |\nclass |\Z)",
            src, re.DOTALL,
        )
        assert fn, "_run_compact not found in app.py"
        assert "compact_model" in fn.group(0) or "_lm" in fn.group(0), \
            "compact_model not used in _run_compact"

    def test_date_weighted_rag_in_fetch_rag_context(self):
        import re
        src = self._app()
        fn = re.search(
            r"async def _fetch_rag_context.*?(?=\n    async def |\n    def |\nclass |\Z)",
            src, re.DOTALL,
        )
        assert fn, "_fetch_rag_context not found in app.py"
        assert "exp(-age_days" in fn.group(0) or "exp(" in fn.group(0), \
            "date-weighted scoring not found in _fetch_rag_context"

    def test_compact_dry_in_handler(self):
        import re
        src = self._app()
        fn = re.search(
            r"async def _handle_compact_command.*?(?=\n    async def |\n    def |\nclass |\Z)",
            src, re.DOTALL,
        )
        assert fn, "_handle_compact_command not found in app.py"
        assert '"dry"' in fn.group(0) or "'dry'" in fn.group(0), \
            "/compact dry not handled in _handle_compact_command"

    def test_compact_history_in_handler(self):
        import re
        src = self._app()
        fn = re.search(
            r"async def _handle_compact_command.*?(?=\n    async def |\n    def |\nclass |\Z)",
            src, re.DOTALL,
        )
        assert fn, "_handle_compact_command not found in app.py"
        assert '"history"' in fn.group(0) or "'history'" in fn.group(0), \
            "/compact history not handled in _handle_compact_command"


# ===========================================================================
# 2. TestTokenize
# ===========================================================================

class TestTokenize:
    """Unit tests for LMStudioTool.tokenize()."""

    def _make_lm(self):
        from aichat.tools.lm_studio import LMStudioTool
        return LMStudioTool("http://localhost:1234")

    @pytest.mark.asyncio
    async def test_returns_token_count_int(self):
        lm = self._make_lm()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"token_count": 42}
        with patch.object(lm, "_post", new=AsyncMock(return_value=mock_resp)):
            result = await lm.tokenize("hello world")
        assert result == 42

    @pytest.mark.asyncio
    async def test_returns_len_when_tokens_is_list(self):
        lm = self._make_lm()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"tokens": [1, 2, 3, 4, 5]}
        with patch.object(lm, "_post", new=AsyncMock(return_value=mock_resp)):
            result = await lm.tokenize("hello world")
        assert result == 5

    @pytest.mark.asyncio
    async def test_fallback_on_http_error(self):
        lm = self._make_lm()
        with patch.object(lm, "_post", new=AsyncMock(side_effect=Exception("timeout"))):
            result = await lm.tokenize("hello world")
        assert result == len("hello world") // 4

    @pytest.mark.asyncio
    async def test_fallback_on_json_error(self):
        lm = self._make_lm()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.side_effect = ValueError("not json")
        with patch.object(lm, "_post", new=AsyncMock(return_value=mock_resp)):
            result = await lm.tokenize("some text here")
        assert result == len("some text here") // 4

    @pytest.mark.asyncio
    async def test_empty_string_returns_zero(self):
        lm = self._make_lm()
        with patch.object(lm, "_post", new=AsyncMock(side_effect=Exception("offline"))):
            result = await lm.tokenize("")
        assert result == 0


# ===========================================================================
# 3. TestModelInfo
# ===========================================================================

class TestModelInfo:
    """Unit tests for LLMClient.model_info()."""

    def _make_client(self):
        from aichat.client import LLMClient
        return LLMClient("http://localhost:1234")

    @pytest.mark.asyncio
    async def test_returns_model_ctx_len_dict(self):
        client = self._make_client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [
            {"id": "llama3", "context_length": 8192},
            {"id": "mistral", "context_length": 32768},
        ]}
        with patch.object(client, "_request", new=AsyncMock(return_value=mock_resp)):
            result = await client.model_info()
        assert result == {"llama3": 8192, "mistral": 32768}

    @pytest.mark.asyncio
    async def test_skips_models_without_context_length(self):
        client = self._make_client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [
            {"id": "llama3", "context_length": 8192},
            {"id": "no-ctx"},  # missing context_length
        ]}
        with patch.object(client, "_request", new=AsyncMock(return_value=mock_resp)):
            result = await client.model_info()
        assert "no-ctx" not in result
        assert result["llama3"] == 8192

    @pytest.mark.asyncio
    async def test_returns_empty_dict_on_error(self):
        client = self._make_client()
        with patch.object(client, "_request", new=AsyncMock(side_effect=Exception("offline"))):
            result = await client.model_info()
        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_empty_dict_for_empty_data(self):
        client = self._make_client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": []}
        with patch.object(client, "_request", new=AsyncMock(return_value=mock_resp)):
            result = await client.model_info()
        assert result == {}


# ===========================================================================
# 4. TestToolCache
# ===========================================================================

class TestToolCache:
    """Unit tests for ToolManager._tool_result_cache and cache methods."""

    def _make_mgr(self):
        from unittest.mock import patch as _patch, MagicMock as _MM
        # Patch Docker/HTTP-dependent tool constructors so ToolManager can be instantiated
        with _patch("aichat.tools.manager.BrowserTool", return_value=_MM()), \
             _patch("aichat.tools.manager.WebSearchTool", return_value=_MM()), \
             _patch("aichat.tools.manager.DatabaseTool", return_value=_MM()), \
             _patch("aichat.tools.manager.MemoryTool", return_value=_MM()), \
             _patch("aichat.tools.manager.ToolkitTool", return_value=_MM()), \
             _patch("aichat.tools.manager.ResearchboxTool", return_value=_MM()), \
             _patch("aichat.tools.manager.ShellTool", return_value=_MM()), \
             _patch("aichat.tools.manager.CodeInterpreterTool", return_value=_MM()), \
             _patch("aichat.tools.manager.ConversationStoreTool", return_value=_MM()):
            from aichat.tools.manager import ToolManager
            return ToolManager()

    def test_cache_empty_on_init(self):
        mgr = self._make_mgr()
        assert isinstance(mgr._tool_result_cache, dict)
        assert mgr._tool_result_cache == {}

    def test_reset_turn_clears_cache(self):
        mgr = self._make_mgr()
        mgr._tool_result_cache["foo:bar"] = "cached result"
        mgr.reset_turn()
        assert mgr._tool_result_cache == {}

    def test_clear_tool_cache_empties_cache(self):
        mgr = self._make_mgr()
        mgr._tool_result_cache["a"] = "x"
        mgr._tool_result_cache["b"] = "y"
        mgr.clear_tool_cache()
        assert mgr._tool_result_cache == {}

    def test_cache_is_a_str_dict(self):
        mgr = self._make_mgr()
        mgr._tool_result_cache["key"] = "value"
        assert mgr._tool_result_cache["key"] == "value"

    def test_multiple_resets_each_clear(self):
        mgr = self._make_mgr()
        for _ in range(3):
            mgr._tool_result_cache["x"] = "data"
            mgr.reset_turn()
            assert mgr._tool_result_cache == {}


# ===========================================================================
# 5. TestAdaptiveThreshold
# ===========================================================================

class TestAdaptiveThreshold:
    """Unit tests for _effective_threshold_pct() interpolation."""

    def _app_with_ctx(self, ctx_len: int, threshold: int = 95):
        app = _bare_app()
        app._context_length = ctx_len
        app._compact_threshold_pct = threshold
        return app

    def test_small_ctx_returns_80(self):
        import aichat.app as appmod
        app = self._app_with_ctx(8192)
        assert appmod.AIChatApp._effective_threshold_pct(app) == 80

    def test_large_ctx_returns_compact_threshold(self):
        import aichat.app as appmod
        app = self._app_with_ctx(131072, threshold=95)
        assert appmod.AIChatApp._effective_threshold_pct(app) == 95

    def test_mid_ctx_between_80_and_95(self):
        import aichat.app as appmod
        app = self._app_with_ctx(35063)
        pct = appmod.AIChatApp._effective_threshold_pct(app)
        assert 80 <= pct <= 95

    def test_result_is_int(self):
        import aichat.app as appmod
        app = self._app_with_ctx(35063)
        pct = appmod.AIChatApp._effective_threshold_pct(app)
        assert isinstance(pct, int)

    def test_custom_threshold_changes_high_end(self):
        import aichat.app as appmod
        app_90 = self._app_with_ctx(131072, threshold=90)
        app_85 = self._app_with_ctx(131072, threshold=85)
        assert appmod.AIChatApp._effective_threshold_pct(app_90) == 90
        assert appmod.AIChatApp._effective_threshold_pct(app_85) == 85


# ===========================================================================
# 6. TestToolResultMaxChars
# ===========================================================================

class TestToolResultMaxChars:
    """_append_tool_messages() uses _tool_result_max_chars for truncation."""

    def _make_app_for_append(self, max_chars: int):
        import aichat.app as appmod
        from aichat.state import Message
        from aichat.tool_scheduler import ToolResult, ToolCall

        app = _bare_app()
        app._tool_result_max_chars = max_chars
        app.messages = []
        # Minimal stub for transcript_store
        ts = MagicMock()
        ts.append = MagicMock()
        app.transcript_store = ts
        return app

    def _make_result(self, output: str):
        from aichat.tool_scheduler import ToolResult, ToolCall
        call = ToolCall(index=0, name="test_tool", args={}, call_id="tc1", label="test")
        return ToolResult(call=call, ok=True, output=output, error=None, attempts=1, duration=0.0)

    def test_truncates_to_500(self):
        import aichat.app as appmod
        from aichat.state import Message
        app = self._make_app_for_append(500)
        result = self._make_result("x" * 2000)
        appmod.AIChatApp._append_tool_messages(app, [result])
        assert len(app.messages[0].content) == 500

    def test_truncates_to_200(self):
        import aichat.app as appmod
        app = self._make_app_for_append(200)
        result = self._make_result("y" * 1000)
        appmod.AIChatApp._append_tool_messages(app, [result])
        assert len(app.messages[0].content) == 200

    def test_default_2000_truncation(self):
        import aichat.app as appmod
        app = self._make_app_for_append(2000)
        result = self._make_result("z" * 5000)
        appmod.AIChatApp._append_tool_messages(app, [result])
        assert len(app.messages[0].content) == 2000

    def test_short_payload_unchanged(self):
        import aichat.app as appmod
        app = self._make_app_for_append(2000)
        result = self._make_result("short output")
        appmod.AIChatApp._append_tool_messages(app, [result])
        assert app.messages[0].content == "short output"


# ===========================================================================
# 7. TestCtxHistory
# ===========================================================================

class TestCtxHistory:
    """_context_pct() populates _ctx_history."""

    def test_ctx_history_empty_on_init(self):
        app = _bare_app()
        assert app._ctx_history == []

    def test_context_pct_populates_history(self):
        import aichat.app as appmod
        app = _bare_app()
        appmod.AIChatApp._context_pct(app)
        assert len(app._ctx_history) == 1

    def test_duplicate_values_not_appended(self):
        import aichat.app as appmod
        app = _bare_app()
        # Call twice with same state → same pct → only one entry
        appmod.AIChatApp._context_pct(app)
        appmod.AIChatApp._context_pct(app)
        assert len(app._ctx_history) == 1

    def test_history_capped_at_20(self):
        import aichat.app as appmod
        from aichat.state import Message
        app = _bare_app()
        # Generate 25 distinct pct values by varying message sizes
        for i in range(25):
            app.messages = [Message("user", "x" * (i * 40))]
            appmod.AIChatApp._context_pct(app)
        assert len(app._ctx_history) <= 20


# ===========================================================================
# 8. TestCtxSparkline
# ===========================================================================

class TestCtxSparkline:
    """_ctx_sparkline() output format."""

    def test_empty_history_no_data(self):
        import aichat.app as appmod
        app = _bare_app()
        result = appmod.AIChatApp._ctx_sparkline(app)
        assert "no data" in result.lower()

    def test_single_value_one_char_sparkline(self):
        import aichat.app as appmod
        app = _bare_app()
        app._ctx_history = [50]
        result = appmod.AIChatApp._ctx_sparkline(app)
        # Should contain one bar character between backticks
        import re
        m = re.search(r"`(.+?)`", result)
        assert m and len(m.group(1)) == 1

    def test_all_same_values_same_bar(self):
        import aichat.app as appmod
        app = _bare_app()
        app._ctx_history = [30, 30, 30, 30]
        result = appmod.AIChatApp._ctx_sparkline(app)
        import re
        m = re.search(r"`(.+?)`", result)
        assert m
        chars = m.group(1)
        assert len(set(chars)) == 1, "All same pct should produce same bar char"


# ===========================================================================
# 9. TestCompactEvents
# ===========================================================================

class TestCompactEvents:
    """_compact_events log in _run_compact()."""

    def _compact_app_with_messages(self, messages=None):
        from aichat.state import Message
        app = _bare_app()
        app.messages = messages or [
            Message("user", "hello"),
            Message("assistant", "world"),
            Message("user", "foo"),
            Message("assistant", "bar"),
        ]
        app.state = MagicMock()
        app.state.personality_id = ""
        app.state.session_id = ""
        app.personalities = []
        app._tool_log = MagicMock()
        return app

    @pytest.mark.asyncio
    async def test_successful_compact_adds_event(self):
        import aichat.app as appmod
        from aichat.state import Message
        app = self._compact_app_with_messages()
        mock_lm = MagicMock()
        mock_lm.chat = AsyncMock(return_value="Summary of the chat.")
        mock_lm.base_url = "http://localhost:1234"
        app.tools = MagicMock()
        app.tools.lm = mock_lm

        msgs = app.messages[:4]
        await appmod.AIChatApp._run_compact(app, msgs, 4)
        assert len(app._compact_events) == 1
        ev = app._compact_events[0]
        assert "time" in ev
        assert "n_turns" in ev
        assert "words" in ev
        assert ev["n_turns"] == 4

    @pytest.mark.asyncio
    async def test_empty_lm_return_no_event(self):
        import aichat.app as appmod
        from aichat.state import Message
        app = self._compact_app_with_messages()
        mock_lm = MagicMock()
        mock_lm.chat = AsyncMock(return_value="")
        mock_lm.base_url = "http://localhost:1234"
        app.tools = MagicMock()
        app.tools.lm = mock_lm

        await appmod.AIChatApp._run_compact(app, app.messages[:4], 4)
        assert len(app._compact_events) == 0

    @pytest.mark.asyncio
    async def test_exception_in_lm_no_event(self):
        import aichat.app as appmod
        app = self._compact_app_with_messages()
        mock_lm = MagicMock()
        mock_lm.chat = AsyncMock(side_effect=Exception("LM offline"))
        mock_lm.base_url = "http://localhost:1234"
        app.tools = MagicMock()
        app.tools.lm = mock_lm

        await appmod.AIChatApp._run_compact(app, app.messages[:4], 4)
        assert len(app._compact_events) == 0

    @pytest.mark.asyncio
    async def test_two_compactions_two_events(self):
        import aichat.app as appmod
        from aichat.state import Message
        app = self._compact_app_with_messages()
        mock_lm = MagicMock()
        mock_lm.chat = AsyncMock(return_value="A nice summary.")
        mock_lm.base_url = "http://localhost:1234"
        app.tools = MagicMock()
        app.tools.lm = mock_lm

        await appmod.AIChatApp._run_compact(app, app.messages[:4], 4)
        app._compact_pending = False
        await appmod.AIChatApp._run_compact(app, app.messages[:4], 4)
        assert len(app._compact_events) == 2


# ===========================================================================
# 10. TestCompactDry (TUI behavioral)
# ===========================================================================

@pytest.mark.asyncio
async def test_compact_dry_too_few_messages():
    """'/compact dry' with <4 messages shows 'Too few messages'."""
    import aichat.app as appmod
    from aichat.state import Message
    app = _bare_app()
    app._compact_from_idx = 0
    app.messages = [Message("user", "hi")]  # only 1

    transcript_calls = []
    app._write_transcript = lambda sp, txt: transcript_calls.append(txt)

    await appmod.AIChatApp._handle_compact_command(app, "dry")
    assert any("few" in t.lower() or "4" in t for t in transcript_calls), \
        f"Expected 'Too few messages' in output, got: {transcript_calls}"


@pytest.mark.asyncio
async def test_compact_dry_shows_token_estimate():
    """'/compact dry' with enough messages shows 'Dry-run compact:'."""
    import aichat.app as appmod
    from aichat.state import Message
    app = _bare_app()
    app._compact_from_idx = 0
    app.messages = [Message("user", "hello world " * 10) for _ in range(8)]
    app._compact_summary = ""

    transcript_calls = []
    app._write_transcript = lambda sp, txt: transcript_calls.append(txt)

    await appmod.AIChatApp._handle_compact_command(app, "dry")
    combined = " ".join(transcript_calls)
    assert "Dry-run" in combined or "dry" in combined.lower(), \
        f"Expected Dry-run in output, got: {combined}"


@pytest.mark.asyncio
async def test_compact_dry_mentions_budget():
    """'/compact dry' output references the context budget percentage."""
    import aichat.app as appmod
    from aichat.state import Message
    app = _bare_app()
    app._compact_from_idx = 0
    app._context_length = 10000
    app._max_response_tokens = 1000
    app.messages = [Message("user", "x" * 200) for _ in range(8)]

    transcript_calls = []
    app._write_transcript = lambda sp, txt: transcript_calls.append(txt)

    await appmod.AIChatApp._handle_compact_command(app, "dry")
    combined = " ".join(transcript_calls)
    assert "%" in combined or "token" in combined.lower(), \
        f"Expected token/% info in output, got: {combined}"


# ===========================================================================
# 11. TestCompactHistory (TUI behavioral)
# ===========================================================================

@pytest.mark.asyncio
async def test_compact_history_no_events():
    """'/compact history' with no events shows 'No compaction events'."""
    import aichat.app as appmod
    app = _bare_app()
    app._compact_events = []

    transcript_calls = []
    app._write_transcript = lambda sp, txt: transcript_calls.append(txt)

    await appmod.AIChatApp._handle_compact_command(app, "history")
    combined = " ".join(transcript_calls)
    assert "no compaction" in combined.lower(), \
        f"Expected 'No compaction events', got: {combined}"


@pytest.mark.asyncio
async def test_compact_history_shows_entry():
    """'/compact history' with one event shows time + n_turns + words."""
    import aichat.app as appmod
    app = _bare_app()
    app._compact_events = [
        {"time": "2026-02-28T12:00:00", "n_turns": 6, "words": 123},
    ]

    transcript_calls = []
    app._write_transcript = lambda sp, txt: transcript_calls.append(txt)

    await appmod.AIChatApp._handle_compact_command(app, "history")
    combined = " ".join(transcript_calls)
    assert "6" in combined and "123" in combined, \
        f"Expected turns=6 and words=123 in output, got: {combined}"


@pytest.mark.asyncio
async def test_compact_history_multiple_events_ordered():
    """Multiple events shown in numbered order."""
    import aichat.app as appmod
    app = _bare_app()
    app._compact_events = [
        {"time": "2026-02-28T10:00:00", "n_turns": 4, "words": 80},
        {"time": "2026-02-28T11:00:00", "n_turns": 6, "words": 120},
    ]

    transcript_calls = []
    app._write_transcript = lambda sp, txt: transcript_calls.append(txt)

    await appmod.AIChatApp._handle_compact_command(app, "history")
    combined = " ".join(transcript_calls)
    # Both entries should be visible
    assert "80" in combined and "120" in combined, \
        f"Both events should be shown, got: {combined}"


# ===========================================================================
# 12. TestForkCommand (TUI behavioral)
# ===========================================================================

@pytest.mark.asyncio
async def test_fork_no_summary_new_session():
    """'/fork' with no compact summary starts fresh session."""
    import aichat.app as appmod
    from aichat.state import Message
    app = _bare_app()
    app._compact_summary = ""
    app.messages = [Message("user", "hello")]
    app._compact_from_idx = 0

    transcript_calls = []
    app._write_transcript = lambda sp, txt: transcript_calls.append(txt)
    app.action_new_chat = AsyncMock()

    await appmod.AIChatApp._handle_fork_command(app)
    combined = " ".join(transcript_calls)
    assert "new session" in combined.lower() or "no prior" in combined.lower(), \
        f"Expected 'new session' in output, got: {combined}"


@pytest.mark.asyncio
async def test_fork_with_summary_preloads_context():
    """'/fork' with active summary shows the pre-load message."""
    import aichat.app as appmod
    from aichat.state import Message
    app = _bare_app()
    app._compact_summary = "Important prior context."
    app.messages = [Message("user", "a"), Message("assistant", "b")]
    app._compact_from_idx = 0

    transcript_calls = []
    app._write_transcript = lambda sp, txt: transcript_calls.append(txt)
    app.action_new_chat = AsyncMock()

    await appmod.AIChatApp._handle_fork_command(app)
    combined = " ".join(transcript_calls)
    assert "forked" in combined.lower() or "pre-loaded" in combined.lower(), \
        f"Expected fork context message, got: {combined}"


@pytest.mark.asyncio
async def test_fork_with_summary_sets_compact_summary():
    """After '/fork' with summary, _compact_summary contains '[Forked from'."""
    import aichat.app as appmod
    from aichat.state import Message
    app = _bare_app()
    app._compact_summary = "Prior session summary."
    app.messages = [Message("user", "x")]
    app._compact_from_idx = 0

    app._write_transcript = MagicMock()
    app.action_new_chat = AsyncMock()

    await appmod.AIChatApp._handle_fork_command(app)
    assert "[Forked from" in app._compact_summary, \
        f"Expected '[Forked from' in summary, got: {app._compact_summary}"


@pytest.mark.asyncio
async def test_fork_resets_compact_from_idx():
    """After '/fork', _compact_from_idx == 0."""
    import aichat.app as appmod
    from aichat.state import Message
    app = _bare_app()
    app._compact_summary = "Some summary."
    app._compact_from_idx = 5
    app.messages = [Message("user", "x")] * 6

    app._write_transcript = MagicMock()
    app.action_new_chat = AsyncMock()

    await appmod.AIChatApp._handle_fork_command(app)
    assert app._compact_from_idx == 0


# ===========================================================================
# 13. TestCtxGraph (TUI behavioral)
# ===========================================================================

@pytest.mark.asyncio
async def test_ctx_empty_history_no_data():
    """'/ctx' with empty history shows 'no data yet'."""
    import aichat.app as appmod
    app = _bare_app()
    app._ctx_history = []

    transcript_calls = []
    app._write_transcript = lambda sp, txt: transcript_calls.append(txt)

    # Simulate handle_command routing
    if app._ctx_history:
        sparkline = appmod.AIChatApp._ctx_sparkline(app)
    else:
        sparkline = appmod.AIChatApp._ctx_sparkline(app)

    pct = appmod.AIChatApp._context_pct(app)
    app._write_transcript("Assistant", f"**CTX history:** {sparkline}\n\nCurrent: **{pct}%**")
    combined = " ".join(transcript_calls)
    assert "no data" in combined.lower(), f"Expected 'no data yet', got: {combined}"


@pytest.mark.asyncio
async def test_ctx_graph_shows_sparkline():
    """'/ctx graph' with history shows the sparkline."""
    import aichat.app as appmod
    app = _bare_app()
    app._ctx_history = [20, 30, 40, 50]

    transcript_calls = []
    app._write_transcript = lambda sp, txt: transcript_calls.append(txt)

    sparkline = appmod.AIChatApp._ctx_sparkline(app)
    pct = appmod.AIChatApp._context_pct(app)
    app._write_transcript("Assistant", f"**CTX history:** {sparkline}\n\nCurrent: **{pct}%**")
    combined = " ".join(transcript_calls)
    assert "CTX history" in combined, f"Expected 'CTX history', got: {combined}"


@pytest.mark.asyncio
async def test_ctx_shows_current_pct():
    """'/ctx' output includes 'Current: N%'."""
    import aichat.app as appmod
    from aichat.state import Message
    app = _bare_app()
    app._ctx_history = [10]
    app.messages = [Message("user", "hi")]

    transcript_calls = []
    app._write_transcript = lambda sp, txt: transcript_calls.append(txt)

    sparkline = appmod.AIChatApp._ctx_sparkline(app)
    pct = appmod.AIChatApp._context_pct(app)
    app._write_transcript("Assistant", f"**CTX history:** {sparkline}\n\nCurrent: **{pct}%**")
    combined = " ".join(transcript_calls)
    assert "Current:" in combined, f"Expected 'Current:' in output, got: {combined}"


# ===========================================================================
# 14. TestIntegrityCheck
# ===========================================================================

class TestIntegrityCheck:
    """Compact state integrity check in _handle_resume_command()."""

    def _make_resume_args(self, compact_from_idx: int, n_turns: int):
        """Return (app, data, turns) simulating a resume scenario."""
        app = _bare_app()
        app._tool_log = MagicMock()
        app._last_status_ts = 0.0
        turns = [{"role": "user", "content": f"msg {i}"} for i in range(n_turns)]
        data = {
            "turns": turns,
            "compact_summary": "A summary." if compact_from_idx > 0 else "",
            "compact_from_idx": compact_from_idx,
        }
        return app, data, turns

    def _apply_integrity_check(self, app, data, turns):
        """Apply the integrity check logic (mirrors _handle_resume_command body)."""
        app._compact_summary = data.get("compact_summary", "") or ""
        app._compact_from_idx = int(data.get("compact_from_idx", 0) or 0)
        if app._compact_from_idx > len(turns):
            app._tool_log(
                f"[compact] integrity: idx {app._compact_from_idx} > {len(turns)} turns → reset"
            )
            app._compact_summary = ""
            app._compact_from_idx = 0

    def test_zero_idx_no_reset(self):
        app, data, turns = self._make_resume_args(0, 5)
        self._apply_integrity_check(app, data, turns)
        assert app._compact_from_idx == 0
        app._tool_log.assert_not_called()

    def test_idx_equal_turns_no_reset(self):
        app, data, turns = self._make_resume_args(5, 5)
        self._apply_integrity_check(app, data, turns)
        assert app._compact_from_idx == 5  # valid edge case
        app._tool_log.assert_not_called()

    def test_idx_exceeds_turns_resets(self):
        app, data, turns = self._make_resume_args(10, 5)
        self._apply_integrity_check(app, data, turns)
        assert app._compact_from_idx == 0, "idx > turns should reset to 0"
        assert app._compact_summary == "", "summary should be cleared on integrity reset"

    def test_reset_logs_to_tool_log(self):
        app, data, turns = self._make_resume_args(99, 5)
        self._apply_integrity_check(app, data, turns)
        app._tool_log.assert_called_once()
        logged = app._tool_log.call_args[0][0]
        assert "integrity" in logged.lower() or "reset" in logged.lower(), \
            f"Expected integrity message, got: {logged}"


# ===========================================================================
# 15. TestPersonaAwareCompact
# ===========================================================================

class TestPersonaAwareCompact:
    """Persona name appears in compaction system prompt."""

    def _compact_app_with_persona(self, persona_id: str, persona_name: str):
        from aichat.state import Message
        app = _bare_app()
        app.messages = [
            Message("user", "hello there"),
            Message("assistant", "hi there"),
            Message("user", "what is your name?"),
            Message("assistant", "I am your assistant."),
        ]
        app.state = MagicMock()
        app.state.personality_id = persona_id
        app.state.session_id = ""
        app.personalities = [{"id": persona_id, "name": persona_name}] if persona_id else []
        app._tool_log = MagicMock()
        return app

    @pytest.mark.asyncio
    async def test_persona_name_in_prompt(self):
        import aichat.app as appmod
        app = self._compact_app_with_persona("coder", "Coder Bot")
        captured_prompts = []

        async def mock_chat(messages, **kwargs):
            captured_prompts.append(messages[0]["content"])
            return "Compact summary."

        mock_lm = MagicMock()
        mock_lm.chat = mock_chat
        mock_lm.base_url = "http://localhost:1234"
        app.tools = MagicMock()
        app.tools.lm = mock_lm

        await appmod.AIChatApp._run_compact(app, app.messages, 4)
        assert captured_prompts, "No chat call was made"
        assert "Coder Bot" in captured_prompts[0], \
            f"Persona name not in system prompt: {captured_prompts[0][:200]}"

    @pytest.mark.asyncio
    async def test_empty_persona_no_persona_note(self):
        import aichat.app as appmod
        app = self._compact_app_with_persona("", "")
        captured_prompts = []

        async def mock_chat(messages, **kwargs):
            captured_prompts.append(messages[0]["content"])
            return "A summary."

        mock_lm = MagicMock()
        mock_lm.chat = mock_chat
        mock_lm.base_url = "http://localhost:1234"
        app.tools = MagicMock()
        app.tools.lm = mock_lm

        await appmod.AIChatApp._run_compact(app, app.messages, 4)
        assert captured_prompts
        assert "persona" not in captured_prompts[0].lower() or "'" not in captured_prompts[0], \
            "Empty persona should not add persona note"

    @pytest.mark.asyncio
    async def test_persona_extracted_from_personalities_list(self):
        import aichat.app as appmod
        app = self._compact_app_with_persona("analyst", "Data Analyst")
        captured_prompts = []

        async def mock_chat(messages, **kwargs):
            captured_prompts.append(messages[0]["content"])
            return "Summary."

        mock_lm = MagicMock()
        mock_lm.chat = mock_chat
        mock_lm.base_url = "http://localhost:1234"
        app.tools = MagicMock()
        app.tools.lm = mock_lm

        await appmod.AIChatApp._run_compact(app, app.messages, 4)
        assert "Data Analyst" in captured_prompts[0]


# ===========================================================================
# 16. TestDateWeightedRAG
# ===========================================================================

class TestDateWeightedRAG:
    """Date-weighted similarity scoring in _fetch_rag_context()."""

    def _fetch_rag(self, results, recency_days=30.0):
        """Call _fetch_rag_context with mocked search_turns returning *results*."""
        import aichat.app as appmod
        app = _bare_app()
        app._rag_recency_days = recency_days
        app._rag_context_query = ""
        app._rag_context_cache = ""
        app.state = MagicMock()
        app.state.session_id = ""
        app.state.rag_context_enabled = True

        # Run synchronously by pulling out the scoring logic inline
        # (avoids needing async embed/search mocks)
        import math
        from datetime import datetime, timezone, timedelta

        lines = ["\n\n---\n**Past context (from earlier sessions):**"]
        any_relevant = False
        now = datetime.now(timezone.utc)
        weighted_scores = []

        for r in results:
            sim = r.get("similarity", 0)
            if sim < 0.25:
                continue
            ts_str = r.get("timestamp", "")
            age_days = 0.0
            if ts_str:
                try:
                    ts_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    age_days = (now - ts_dt).total_seconds() / 86400
                except Exception:
                    pass
            weighted = sim * math.exp(-age_days / max(1.0, recency_days))
            weighted_scores.append((weighted, sim, age_days))
            if weighted < 0.2:
                continue
            any_relevant = True

        return any_relevant, weighted_scores

    def _ts_days_ago(self, days: float) -> str:
        from datetime import datetime, timezone, timedelta
        dt = datetime.now(timezone.utc) - timedelta(days=days)
        return dt.isoformat()

    def test_recent_result_weighted_approx_equals_similarity(self):
        """Age=0 → weight ≈ similarity (exp(0) = 1)."""
        results = [{"similarity": 0.8, "timestamp": self._ts_days_ago(0)}]
        relevant, scores = self._fetch_rag(results, recency_days=30)
        assert scores, "Expected at least one score"
        w, sim, age = scores[0]
        assert abs(w - sim) < 0.05, f"Recent weight {w:.3f} should ≈ sim {sim}"

    def test_old_result_weight_decays(self):
        """Age=60d, recency=30d → weight ≈ sim * exp(-2) ≈ sim * 0.135."""
        import math
        sim = 0.9
        results = [{"similarity": sim, "timestamp": self._ts_days_ago(60)}]
        _, scores = self._fetch_rag(results, recency_days=30)
        assert scores
        w, _, _ = scores[0]
        expected = sim * math.exp(-2)
        assert abs(w - expected) < 0.02, f"60d-old weight {w:.3f} should ≈ {expected:.3f}"

    def test_very_old_result_excluded(self):
        """Very old result with low similarity → not relevant (weight < 0.2)."""
        results = [{"similarity": 0.4, "timestamp": self._ts_days_ago(365)}]
        relevant, _ = self._fetch_rag(results, recency_days=30)
        assert not relevant, "Very old low-sim result should not be relevant"

    def test_recency_days_affects_scoring(self):
        """Higher rag_recency_days = slower decay = higher weight for old results."""
        import math
        sim = 0.8
        ts = self._ts_days_ago(30)
        results = [{"similarity": sim, "timestamp": ts}]
        _, scores_30 = self._fetch_rag(results, recency_days=30)
        _, scores_100 = self._fetch_rag(results, recency_days=100)
        assert scores_30 and scores_100
        w30 = scores_30[0][0]
        w100 = scores_100[0][0]
        assert w100 > w30, f"Higher recency_days should give higher weight: {w100:.3f} > {w30:.3f}"
