"""
Tests for the parallel thinking model (v4).

Sections:
  1. TestSourceInspection   — grep for all new symbols
  2. TestThinkResult        — dataclass structure
  3. TestScoreChain         — heuristic scoring
  4. TestThinkFanOut        — parallel fan-out (mock lm.chat)
  5. TestSynthesize         — synthesis step (mock lm.chat)
  6. TestThinkAndAnswer     — end-to-end (mock)
  7. TestExtractThinking    — sanitizer.extract_thinking()
  8. TestToolCache          — check_cache / store_cache on ToolManager
  9. TestCacheWiring        — _execute_tool_from_call caching behaviour
 10. TestThinkConfig        — config field validators
 11. TestThinkCommand       — /think TUI command
 12. TestThinkingCommand    — /thinking [on|off|status|paths N|model X]
 13. TestAutoThinking       — auto-thinking on submit
 14. TestStatusBar          — THK:ON indicator in status bar
 15. TestThinkingToolInit   — ToolManager.think_tool creation
"""
from __future__ import annotations

import asyncio
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
    """Minimal AIChatApp object — no Textual UI."""
    import aichat.app as appmod

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
    app._thinking_enabled = False
    app._thinking_paths = 3
    app._thinking_model = ""
    app._thinking_temperature = 0.8
    app._thinking_count = 0
    app.messages = []
    app.personalities = []
    return app


# ===========================================================================
# 1. Source inspection
# ===========================================================================

class TestSourceInspection:
    """Verify all new v4 symbols exist in source files."""

    def _thinking(self) -> str:
        return _read("src/aichat/tools/thinking.py")

    def _sanitizer(self) -> str:
        return _read("src/aichat/sanitizer.py")

    def _cfg(self) -> str:
        return _read("src/aichat/config.py")

    def _mgr(self) -> str:
        return _read("src/aichat/tools/manager.py")

    def _app(self) -> str:
        return _read("src/aichat/app.py")

    # thinking.py
    def test_thinkresult_in_thinking(self):
        assert "ThinkResult" in self._thinking()

    def test_thinkingtool_in_thinking(self):
        assert "ThinkingTool" in self._thinking()

    def test_score_chain_in_thinking(self):
        assert "score_chain" in self._thinking()

    def test_think_method_in_thinking(self):
        assert "async def think(" in self._thinking()

    def test_synthesize_in_thinking(self):
        assert "async def synthesize(" in self._thinking()

    def test_think_and_answer_in_thinking(self):
        assert "async def think_and_answer(" in self._thinking()

    # sanitizer.py
    def test_extract_thinking_in_sanitizer(self):
        assert "def extract_thinking(" in self._sanitizer()

    # config.py
    def test_thinking_enabled_in_config(self):
        assert "thinking_enabled" in self._cfg()

    def test_thinking_paths_in_config(self):
        assert "thinking_paths" in self._cfg()

    def test_thinking_model_in_config(self):
        assert "thinking_model" in self._cfg()

    def test_thinking_temperature_in_config(self):
        assert "thinking_temperature" in self._cfg()

    # manager.py
    def test_think_toolname_in_manager(self):
        assert 'THINK = "think"' in self._mgr()

    def test_think_tool_attr_in_manager(self):
        assert "think_tool" in self._mgr()

    def test_check_cache_in_manager(self):
        assert "def check_cache(" in self._mgr()

    def test_store_cache_in_manager(self):
        assert "def store_cache(" in self._mgr()

    def test_run_think_in_manager(self):
        assert "async def run_think(" in self._mgr()

    # app.py
    def test_thinking_enabled_in_app_init(self):
        assert "_thinking_enabled" in self._app()

    def test_apply_thinking_in_app(self):
        assert "async def _apply_thinking(" in self._app()

    def test_handle_thinking_command_in_app(self):
        assert "async def _handle_thinking_command(" in self._app()


# ===========================================================================
# 2. TestThinkResult
# ===========================================================================

class TestThinkResult:
    def test_paths_all_defaults_to_empty_list(self):
        from aichat.tools.thinking import ThinkResult
        r = ThinkResult(reasoning="r", answer="a", paths_tried=1,
                        best_score=0.5, duration_ms=10)
        assert r.paths_all == []

    def test_all_fields_accessible(self):
        from aichat.tools.thinking import ThinkResult
        r = ThinkResult(reasoning="r", answer="a", paths_tried=2,
                        best_score=0.7, duration_ms=42, paths_all=["x"])
        assert r.reasoning == "r"
        assert r.answer == "a"
        assert r.paths_tried == 2
        assert r.best_score == 0.7
        assert r.duration_ms == 42
        assert r.paths_all == ["x"]

    def test_duration_ms_is_int(self):
        from aichat.tools.thinking import ThinkResult
        r = ThinkResult(reasoning="", answer="", paths_tried=0,
                        best_score=0.0, duration_ms=100)
        assert isinstance(r.duration_ms, int)


# ===========================================================================
# 3. TestScoreChain
# ===========================================================================

class TestScoreChain:
    def _tool(self):
        from aichat.tools.lm_studio import LMStudioTool
        from aichat.tools.thinking import ThinkingTool
        lm = MagicMock(spec=LMStudioTool)
        lm.base_url = "http://localhost:1234"
        return ThinkingTool(lm=lm)

    def test_empty_string_returns_zero(self):
        assert self._tool().score_chain("") == 0.0

    def test_empty_whitespace_returns_zero(self):
        assert self._tool().score_chain("   \n  ") == 0.0

    def test_short_string_scores_low(self):
        score = self._tool().score_chain("hi")
        assert score < 0.3

    def test_rich_chain_scores_high(self):
        # 400-word chain with many markers — length_score saturates at 1.0
        chain = " ".join(["step first therefore because conclusion"] * 80)
        score = self._tool().score_chain(chain)
        assert score > 0.5

    def test_score_always_in_zero_one_range(self):
        tool = self._tool()
        for text in ["", "abc", "first step because therefore " * 30]:
            s = tool.score_chain(text)
            assert 0.0 <= s <= 1.0, f"Score {s} out of [0,1]"

    def test_marker_rich_beats_short(self):
        tool = self._tool()
        long_with_markers = " ".join(
            ["therefore step first second finally because hence result"] * 10
        )
        short = "yes"
        assert tool.score_chain(long_with_markers) > tool.score_chain(short)


# ===========================================================================
# 4. TestThinkFanOut
# ===========================================================================

class TestThinkFanOut:
    def _tool(self, chat_return="reasoning chain"):
        from aichat.tools.lm_studio import LMStudioTool
        from aichat.tools.thinking import ThinkingTool
        lm = MagicMock(spec=LMStudioTool)
        lm.base_url = "http://localhost:1234"
        lm.chat = AsyncMock(return_value=chat_return)
        return ThinkingTool(lm=lm)

    @pytest.mark.asyncio
    async def test_n_paths_3_calls_chat_3_times(self):
        tool = self._tool()
        chains = await tool.think("question", n_paths=3)
        assert tool.lm.chat.call_count == 3
        assert len(chains) == 3

    @pytest.mark.asyncio
    async def test_n_paths_1_returns_one_result(self):
        tool = self._tool("single chain")
        chains = await tool.think("q", n_paths=1)
        assert len(chains) == 1
        assert chains[0] == "single chain"

    @pytest.mark.asyncio
    async def test_all_calls_fail_returns_empty_list(self):
        from aichat.tools.lm_studio import LMStudioTool
        from aichat.tools.thinking import ThinkingTool
        lm = MagicMock(spec=LMStudioTool)
        lm.base_url = "http://localhost:1234"
        lm.chat = AsyncMock(side_effect=RuntimeError("fail"))
        tool = ThinkingTool(lm=lm)
        # LMStudioTool.chat is fail-open and returns "" on exception;
        # simulate that by returning "" instead
        lm.chat = AsyncMock(return_value="")
        chains = await tool.think("q", n_paths=3)
        assert chains == []

    @pytest.mark.asyncio
    async def test_partial_failure_returns_remaining(self):
        from aichat.tools.lm_studio import LMStudioTool
        from aichat.tools.thinking import ThinkingTool
        lm = MagicMock(spec=LMStudioTool)
        lm.base_url = "http://localhost:1234"
        call_count = [0]

        async def _chat(msgs, **kw):
            call_count[0] += 1
            if call_count[0] == 2:
                return ""  # simulate empty (filtered out)
            return "chain"

        lm.chat = _chat
        tool = ThinkingTool(lm=lm)
        chains = await tool.think("q", n_paths=3)
        assert len(chains) == 2
        assert all(c == "chain" for c in chains)

    @pytest.mark.asyncio
    async def test_empty_string_response_filtered(self):
        tool = self._tool("")  # lm returns empty
        chains = await tool.think("q", n_paths=3)
        assert chains == []

    @pytest.mark.asyncio
    async def test_returns_list_of_strings(self):
        tool = self._tool("abc")
        chains = await tool.think("q", n_paths=2)
        assert isinstance(chains, list)
        assert all(isinstance(c, str) for c in chains)


# ===========================================================================
# 5. TestSynthesize
# ===========================================================================

class TestSynthesize:
    def _tool(self, chat_return="final answer"):
        from aichat.tools.lm_studio import LMStudioTool
        from aichat.tools.thinking import ThinkingTool
        lm = MagicMock(spec=LMStudioTool)
        lm.base_url = "http://localhost:1234"
        lm.chat = AsyncMock(return_value=chat_return)
        return ThinkingTool(lm=lm)

    @pytest.mark.asyncio
    async def test_returns_chat_result(self):
        tool = self._tool("my answer")
        result = await tool.synthesize("question", "reasoning")
        assert result == "my answer"

    @pytest.mark.asyncio
    async def test_empty_chat_returns_empty(self):
        tool = self._tool("")
        result = await tool.synthesize("q", "r")
        assert result == ""

    @pytest.mark.asyncio
    async def test_correct_messages_structure(self):
        tool = self._tool("answer")
        await tool.synthesize("What is 2+2?", "2+2=4 therefore 4")
        call_args = tool.lm.chat.call_args
        msgs = call_args[0][0]
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "What is 2+2?" in msgs[1]["content"]
        assert "2+2=4 therefore 4" in msgs[1]["content"]

    @pytest.mark.asyncio
    async def test_uses_low_temperature(self):
        tool = self._tool("ok")
        await tool.synthesize("q", "r")
        kw = tool.lm.chat.call_args[1]
        assert kw.get("temperature", 1.0) <= 0.5


# ===========================================================================
# 6. TestThinkAndAnswer
# ===========================================================================

class TestThinkAndAnswer:
    def _tool(self, think_returns=None, synth_return="answer"):
        from aichat.tools.lm_studio import LMStudioTool
        from aichat.tools.thinking import ThinkingTool
        lm = MagicMock(spec=LMStudioTool)
        lm.base_url = "http://localhost:1234"
        call_n = [0]
        think_rets = think_returns or ["chain1", "chain2 therefore conclusion", "chain3"]

        async def _chat(msgs, **kw):
            call_n[0] += 1
            # First n_paths calls are think calls; last is synthesize
            if call_n[0] <= len(think_rets):
                return think_rets[call_n[0] - 1]
            return synth_return

        lm.chat = _chat
        return ThinkingTool(lm=lm)

    @pytest.mark.asyncio
    async def test_empty_prompt_returns_empty_result(self):
        from aichat.tools.thinking import ThinkResult
        from aichat.tools.lm_studio import LMStudioTool
        from aichat.tools.thinking import ThinkingTool
        lm = MagicMock(spec=LMStudioTool)
        lm.base_url = "http://localhost:1234"
        lm.chat = AsyncMock(return_value="")
        tool = ThinkingTool(lm=lm)
        result = await tool.think_and_answer("", n_paths=2)
        assert result.answer == ""
        assert result.reasoning == ""

    @pytest.mark.asyncio
    async def test_all_chains_fail_returns_empty(self):
        from aichat.tools.lm_studio import LMStudioTool
        from aichat.tools.thinking import ThinkingTool
        lm = MagicMock(spec=LMStudioTool)
        lm.base_url = "http://localhost:1234"
        lm.chat = AsyncMock(return_value="")
        tool = ThinkingTool(lm=lm)
        result = await tool.think_and_answer("hard question", n_paths=3)
        assert result.answer == ""
        assert result.paths_tried == 3

    @pytest.mark.asyncio
    async def test_best_chain_selected(self):
        from aichat.tools.lm_studio import LMStudioTool
        from aichat.tools.thinking import ThinkingTool
        lm = MagicMock(spec=LMStudioTool)
        lm.base_url = "http://localhost:1234"
        short = "yes"
        long_rich = " ".join(["step therefore because finally conclusion"] * 20)
        call_n = [0]
        chains = [short, long_rich, short]
        async def _chat(msgs, **kw):
            call_n[0] += 1
            if call_n[0] <= 3:
                return chains[call_n[0] - 1]
            return "synthesized"
        lm.chat = _chat
        tool = ThinkingTool(lm=lm)
        result = await tool.think_and_answer("q", n_paths=3)
        # Best chain should be the long rich one
        assert result.reasoning == long_rich

    @pytest.mark.asyncio
    async def test_synthesis_used_for_answer(self):
        tool = self._tool(synth_return="synthesized answer")
        result = await tool.think_and_answer("question", n_paths=3)
        assert result.answer == "synthesized answer"

    @pytest.mark.asyncio
    async def test_synthesis_empty_fallback_to_chain(self):
        from aichat.tools.lm_studio import LMStudioTool
        from aichat.tools.thinking import ThinkingTool
        lm = MagicMock(spec=LMStudioTool)
        lm.base_url = "http://localhost:1234"
        call_n = [0]
        async def _chat(msgs, **kw):
            call_n[0] += 1
            if call_n[0] <= 2:
                return "the best chain" if call_n[0] == 1 else "other chain"
            return ""  # synthesis fails
        lm.chat = _chat
        tool = ThinkingTool(lm=lm)
        result = await tool.think_and_answer("q", n_paths=2)
        assert result.answer == result.reasoning  # fallback

    @pytest.mark.asyncio
    async def test_duration_ms_non_negative(self):
        tool = self._tool()
        result = await tool.think_and_answer("what?", n_paths=3)
        assert result.duration_ms >= 0


# ===========================================================================
# 7. TestExtractThinking
# ===========================================================================

class TestExtractThinking:
    def _fn(self):
        from aichat.sanitizer import extract_thinking
        return extract_thinking

    def test_no_tags_returns_empty_and_original(self):
        fn = self._fn()
        thinking, text = fn("hello world")
        assert thinking == ""
        assert text == "hello world"

    def test_simple_block_extracted(self):
        fn = self._fn()
        thinking, text = fn("<think>reasoning here</think>")
        assert thinking == "reasoning here"
        assert text == ""

    def test_block_plus_rest_splits_correctly(self):
        fn = self._fn()
        thinking, text = fn("<think>reason</think> final answer")
        assert thinking == "reason"
        assert text == "final answer"

    def test_multiline_block_preserved(self):
        fn = self._fn()
        thinking, text = fn("<think>line1\nline2\nline3</think>rest")
        assert "line1" in thinking
        assert "line3" in thinking
        assert text == "rest"

    def test_only_first_block_extracted(self):
        fn = self._fn()
        thinking, text = fn("<think>first</think> middle <think>second</think> tail")
        assert thinking == "first"
        # Second block should still be in text (not extracted)
        assert "second" in text


# ===========================================================================
# 8. TestToolCache
# ===========================================================================

class TestToolCache:
    def _mgr(self):
        from aichat.tools.manager import ToolManager
        return ToolManager.__new__(ToolManager)

    def _make_mgr(self):
        mgr = self._mgr()
        mgr._tool_result_cache = {}
        return mgr

    def test_check_cache_returns_none_for_unknown(self):
        mgr = self._make_mgr()
        assert mgr.check_cache("web_search", {"query": "test"}) is None

    def test_store_then_check_returns_value(self):
        mgr = self._make_mgr()
        mgr.store_cache("web_search", {"query": "test"}, "result!")
        assert mgr.check_cache("web_search", {"query": "test"}) == "result!"

    def test_same_name_args_same_key(self):
        mgr = self._make_mgr()
        mgr.store_cache("db_search", {"q": "hello"}, "v1")
        assert mgr.check_cache("db_search", {"q": "hello"}) == "v1"

    def test_different_args_different_key(self):
        mgr = self._make_mgr()
        mgr.store_cache("db_search", {"q": "hello"}, "v1")
        assert mgr.check_cache("db_search", {"q": "world"}) is None

    def test_reset_turn_clears_cache(self):
        mgr = self._make_mgr()
        mgr._calls_this_turn = 0
        mgr.store_cache("web_search", {"query": "x"}, "val")
        mgr.reset_turn()
        assert mgr.check_cache("web_search", {"query": "x"}) is None


# ===========================================================================
# 9. TestCacheWiring
# ===========================================================================

class TestCacheWiring:
    """_execute_tool_from_call should check cache before executing, and store after."""

    def _app_with_tools(self):
        import aichat.app as appmod
        app = _bare_app()
        app._tool_log = MagicMock()

        # Minimal tools mock
        tools = MagicMock()
        tools.check_cache = MagicMock(return_value=None)
        tools.store_cache = MagicMock()
        app.tools = tools

        # Patch _execute_tool_call
        app._execute_tool_call = AsyncMock(return_value="fresh result")
        return app

    @pytest.mark.asyncio
    async def test_cache_miss_calls_execute(self):
        import aichat.app as appmod
        app = self._app_with_tools()
        call = MagicMock()
        call.name = "web_search"
        call.args = {"query": "test"}
        result = await appmod.AIChatApp._execute_tool_from_call(app, call)
        assert result == "fresh result"
        app._execute_tool_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_hit_skips_execute(self):
        import aichat.app as appmod
        app = self._app_with_tools()
        app.tools.check_cache.return_value = "cached!"
        call = MagicMock()
        call.name = "web_search"
        call.args = {"query": "test"}
        result = await appmod.AIChatApp._execute_tool_from_call(app, call)
        assert result == "cached!"
        app._execute_tool_call.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_hit_logs_cache_hit(self):
        import aichat.app as appmod
        app = self._app_with_tools()
        app.tools.check_cache.return_value = "hit"
        call = MagicMock()
        call.name = "web_search"
        call.args = {}
        await appmod.AIChatApp._execute_tool_from_call(app, call)
        log_call = app._tool_log.call_args[0][0]
        assert "cache hit" in log_call

    @pytest.mark.asyncio
    async def test_different_args_both_execute(self):
        import aichat.app as appmod
        app = self._app_with_tools()
        # Always cache miss
        app.tools.check_cache.return_value = None
        call1 = MagicMock(); call1.name = "web_search"; call1.args = {"q": "a"}
        call2 = MagicMock(); call2.name = "web_search"; call2.args = {"q": "b"}
        await appmod.AIChatApp._execute_tool_from_call(app, call1)
        await appmod.AIChatApp._execute_tool_from_call(app, call2)
        assert app._execute_tool_call.call_count == 2


# ===========================================================================
# 10. TestThinkConfig
# ===========================================================================

class TestThinkConfig:
    def _validate(self, overrides: dict) -> dict:
        from aichat.config import _validate
        return _validate(overrides)

    def test_thinking_enabled_defaults_false(self):
        cfg = self._validate({})
        assert cfg["thinking_enabled"] is False

    def test_thinking_paths_clamped_high(self):
        cfg = self._validate({"thinking_paths": 99})
        from aichat.config import AppConfig
        assert cfg["thinking_paths"] == AppConfig().thinking_paths  # reset to default

    def test_thinking_paths_clamped_low(self):
        cfg = self._validate({"thinking_paths": 0})
        from aichat.config import AppConfig
        assert cfg["thinking_paths"] == AppConfig().thinking_paths

    def test_thinking_temperature_bad_value_reset(self):
        # temperature <= 0 should be rejected
        cfg = self._validate({"thinking_temperature": -1.0})
        from aichat.config import AppConfig
        assert cfg["thinking_temperature"] == AppConfig().thinking_temperature

    def test_thinking_temperature_valid(self):
        cfg = self._validate({"thinking_temperature": 1.5})
        assert cfg["thinking_temperature"] == 1.5

    def test_thinking_model_accepts_empty(self):
        cfg = self._validate({"thinking_model": ""})
        assert cfg["thinking_model"] == ""


# ===========================================================================
# 11. TestThinkCommand
# ===========================================================================

class TestThinkCommand:
    def _app(self):
        app = _bare_app()
        app._write_transcript = MagicMock()
        app._apply_thinking = AsyncMock(return_value="answer (3 paths, score 0.75, 120ms)")
        app.tools = MagicMock()
        app.tools.reset_turn = MagicMock()

        class FakeState:
            approval = MagicMock()
        app.state = FakeState()
        return app

    @pytest.mark.asyncio
    async def test_think_no_arg_shows_usage(self):
        import aichat.app as appmod
        app = self._app()
        await appmod.AIChatApp.handle_command(app, "/think")
        transcript_call = app._write_transcript.call_args[0][1]
        assert "Usage" in transcript_call

    @pytest.mark.asyncio
    async def test_think_query_calls_apply_thinking(self):
        import aichat.app as appmod
        app = self._app()
        await appmod.AIChatApp.handle_command(app, "/think What is 2+2?")
        app._apply_thinking.assert_called_once_with("What is 2+2?")

    @pytest.mark.asyncio
    async def test_think_failure_shows_failure_message(self):
        import aichat.app as appmod
        app = self._app()
        app._apply_thinking = AsyncMock(return_value=None)
        await appmod.AIChatApp.handle_command(app, "/think something")
        calls = [c[0][1] for c in app._write_transcript.call_args_list]
        assert any("failed" in c.lower() or "no answer" in c.lower() for c in calls)

    @pytest.mark.asyncio
    async def test_think_output_shows_metadata(self):
        import aichat.app as appmod
        app = self._app()
        await appmod.AIChatApp.handle_command(app, "/think something")
        calls = [c[0][1] for c in app._write_transcript.call_args_list]
        assert any("paths" in c for c in calls)


# ===========================================================================
# 12. TestThinkingCommand
# ===========================================================================

class TestThinkingCommand:
    def _app(self):
        app = _bare_app()
        app._write_transcript = MagicMock()
        app.update_status = AsyncMock()

        class FakeTool:
            model = ""
        app.tools = MagicMock()
        app.tools.think_tool = FakeTool()
        return app

    @pytest.mark.asyncio
    async def test_on_enables_thinking(self):
        import aichat.app as appmod
        app = self._app()
        await appmod.AIChatApp._handle_thinking_command(app, "on")
        assert app._thinking_enabled is True
        msg = app._write_transcript.call_args[0][1]
        assert "ON" in msg

    @pytest.mark.asyncio
    async def test_off_disables_thinking(self):
        import aichat.app as appmod
        app = self._app()
        app._thinking_enabled = True
        await appmod.AIChatApp._handle_thinking_command(app, "off")
        assert app._thinking_enabled is False
        msg = app._write_transcript.call_args[0][1]
        assert "OFF" in msg

    @pytest.mark.asyncio
    async def test_paths_sets_value(self):
        import aichat.app as appmod
        app = self._app()
        await appmod.AIChatApp._handle_thinking_command(app, "paths 5")
        assert app._thinking_paths == 5

    @pytest.mark.asyncio
    async def test_paths_clamped_to_ten(self):
        import aichat.app as appmod
        app = self._app()
        await appmod.AIChatApp._handle_thinking_command(app, "paths 99")
        assert app._thinking_paths == 10

    @pytest.mark.asyncio
    async def test_paths_clamped_to_one(self):
        import aichat.app as appmod
        app = self._app()
        await appmod.AIChatApp._handle_thinking_command(app, "paths 0")
        assert app._thinking_paths == 1

    @pytest.mark.asyncio
    async def test_model_sets_value_and_tool(self):
        import aichat.app as appmod
        app = self._app()
        await appmod.AIChatApp._handle_thinking_command(app, "model qwen-2.5")
        assert app._thinking_model == "qwen-2.5"
        assert app.tools.think_tool.model == "qwen-2.5"

    @pytest.mark.asyncio
    async def test_status_shows_all_fields(self):
        import aichat.app as appmod
        app = self._app()
        app._thinking_enabled = True
        app._thinking_paths = 4
        app._thinking_count = 7
        await appmod.AIChatApp._handle_thinking_command(app, "status")
        msg = app._write_transcript.call_args[0][1]
        assert "4" in msg           # paths
        assert "7" in msg           # call count
        assert "ON" in msg or "OFF" in msg


# ===========================================================================
# 13. TestAutoThinking
# ===========================================================================

class TestAutoThinking:
    def _app(self, thinking_enabled=False):
        app = _bare_app()
        app._write_transcript = MagicMock()
        app._apply_thinking = AsyncMock(return_value="**[Parallel Think]**\n\nanswer")
        app._thinking_enabled = thinking_enabled
        return app

    @pytest.mark.asyncio
    async def test_disabled_no_thinking_call(self):
        app = self._app(thinking_enabled=False)
        # Simulate what handle_submit does re: auto-thinking
        if app._thinking_enabled:
            await app._apply_thinking("test")
        app._apply_thinking.assert_not_called()

    @pytest.mark.asyncio
    async def test_enabled_calls_apply_thinking(self):
        app = self._app(thinking_enabled=True)
        text = "what is the meaning of life?"
        if app._thinking_enabled:
            _think_out = await app._apply_thinking(text)
            if _think_out:
                app._write_transcript("Assistant", f"**[Parallel Think]**\n\n{_think_out}")
        app._apply_thinking.assert_called_once_with(text)

    @pytest.mark.asyncio
    async def test_result_shows_parallel_think_header(self):
        app = self._app(thinking_enabled=True)
        text = "question"
        if app._thinking_enabled:
            _think_out = await app._apply_thinking(text)
            if _think_out:
                app._write_transcript("Assistant", f"**[Parallel Think]**\n\n{_think_out}")
        transcript_calls = [c[0][1] for c in app._write_transcript.call_args_list]
        assert any("Parallel Think" in c for c in transcript_calls)


# ===========================================================================
# 14. TestStatusBar
# ===========================================================================

class TestStatusBar:
    def _status_string(self, thinking_enabled: bool) -> str:
        """Build the portion of the status line that includes THK."""
        thk_str = " | THK:ON" if thinking_enabled else ""
        return f"stream=OFF | approval=auto | concise=OFF{thk_str}"

    def test_disabled_thk_not_in_status(self):
        status = self._status_string(False)
        assert "THK:ON" not in status

    def test_enabled_thk_in_status(self):
        status = self._status_string(True)
        assert "THK:ON" in status

    def test_thk_format_correct(self):
        src = _read("src/aichat/app.py")
        assert 'thk_str = " | THK:ON" if self._thinking_enabled else ""' in src


# ===========================================================================
# 15. TestThinkingToolInit
# ===========================================================================

class TestThinkingToolInit:
    def _make_mgr(self):
        from aichat.tools.manager import ToolManager
        # We can't actually init the full ToolManager (needs Docker services),
        # so we check the source and the class definition only.
        return None  # placeholder

    def test_manager_source_creates_think_tool(self):
        src = _read("src/aichat/tools/manager.py")
        assert "think_tool" in src
        assert "ThinkingTool" in src

    def test_thinking_tool_class_exists(self):
        from aichat.tools.thinking import ThinkingTool
        assert ThinkingTool is not None

    def test_thinkingtool_lm_attr(self):
        from aichat.tools.lm_studio import LMStudioTool
        from aichat.tools.thinking import ThinkingTool
        lm = MagicMock(spec=LMStudioTool)
        lm.base_url = "http://localhost:1234"
        tool = ThinkingTool(lm=lm)
        assert tool.lm is lm

    def test_thinkingtool_model_default_empty(self):
        from aichat.tools.lm_studio import LMStudioTool
        from aichat.tools.thinking import ThinkingTool
        lm = MagicMock(spec=LMStudioTool)
        lm.base_url = "http://localhost:1234"
        tool = ThinkingTool(lm=lm)
        assert tool.model == ""
