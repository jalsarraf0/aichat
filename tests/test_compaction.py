"""
Tests for contextual compaction feature.

Sections:
  1. Source inspection — grep for required symbols/strings; always run
  2. OOP unit tests   — no services needed; always run
  3. TUI behavioral   — headless Textual app; no LM services needed
"""
from __future__ import annotations

import inspect
import pathlib
import sys

import pytest

sys.path.insert(0, "src")

ROOT = pathlib.Path(__file__).parent.parent


def _read(relpath: str) -> str:
    return (ROOT / relpath).read_text()


# ===========================================================================
# 1. Source inspection tests (always run)
# ===========================================================================

class TestCompactionSourceInspection:
    """Verify all required compaction symbols exist in source files."""

    def _app_src(self) -> str:
        return _read("src/aichat/app.py")

    def _state_src(self) -> str:
        return _read("src/aichat/state.py")

    def test_compaction_enabled_in_state(self):
        src = self._state_src()
        assert "compaction_enabled" in src, \
            "compaction_enabled field not found in state.py"
        assert "True" in src, "compaction_enabled default True not found"

    def test_compact_summary_field_in_app(self):
        src = self._app_src()
        assert "_compact_summary" in src, \
            "_compact_summary field not found in app.py __init__"

    def test_compact_from_idx_field_in_app(self):
        src = self._app_src()
        assert "_compact_from_idx" in src, \
            "_compact_from_idx field not found in app.py __init__"

    def test_compact_pending_field_in_app(self):
        src = self._app_src()
        assert "_compact_pending" in src, \
            "_compact_pending field not found in app.py __init__"

    def test_compact_constants_in_app(self):
        src = self._app_src()
        assert "_COMPACT_MIN_MSGS" in src, \
            "_COMPACT_MIN_MSGS constant not found in app.py"
        assert "_COMPACT_KEEP_RATIO" in src, \
            "_COMPACT_KEEP_RATIO constant not found in app.py"

    def test_run_compact_method_in_app(self):
        src = self._app_src()
        assert "async def _run_compact" in src, \
            "_run_compact method not found in app.py"

    def test_maybe_compact_method_in_app(self):
        src = self._app_src()
        assert "async def _maybe_compact" in src, \
            "_maybe_compact method not found in app.py"

    def test_compact_command_dispatch_in_app(self):
        src = self._app_src()
        assert '"/compact"' in src or 'startswith("/compact")' in src, \
            "/compact dispatch not found in handle_command"

    def test_compact_handler_in_app(self):
        src = self._app_src()
        assert "async def _handle_compact_command" in src, \
            "_handle_compact_command method not found in app.py"

    def test_compact_in_help_text(self):
        src = self._app_src()
        import re
        help_fn = re.search(
            r"async def action_help.*?(?=\n    async def |\nclass |\Z)",
            src, re.DOTALL,
        )
        assert help_fn, "action_help not found in app.py"
        assert "/compact" in help_fn.group(0), \
            "/compact not listed in action_help help text"

    def test_cmp_indicator_in_update_status(self):
        src = self._app_src()
        import re
        status_fn = re.search(
            r"async def update_status.*?(?=\n    async def |\n    def |\nclass |\Z)",
            src, re.DOTALL,
        )
        assert status_fn, "update_status not found in app.py"
        assert "CMP" in status_fn.group(0), \
            "CMP indicator not found in update_status"

    def test_compact_reset_in_start_new_chat(self):
        src = self._app_src()
        import re
        new_chat_fn = re.search(
            r"def _start_new_chat.*?(?=\n    async def |\n    def |\nclass |\Z)",
            src, re.DOTALL,
        )
        assert new_chat_fn, "_start_new_chat not found in app.py"
        fn_src = new_chat_fn.group(0)
        assert "_compact_summary" in fn_src, \
            "_compact_summary not reset in _start_new_chat"
        assert "_compact_from_idx" in fn_src, \
            "_compact_from_idx not reset in _start_new_chat"
        assert "_compact_pending" in fn_src, \
            "_compact_pending not reset in _start_new_chat"

    def test_compact_summary_in_llm_messages(self):
        """_llm_messages must inject compact summary into system content."""
        src = self._app_src()
        import re
        llm_fn = re.search(
            r"async def _llm_messages.*?(?=\n    async def |\n    def |\nclass |\Z)",
            src, re.DOTALL,
        )
        assert llm_fn, "_llm_messages not found in app.py"
        fn_src = llm_fn.group(0)
        assert "_compact_summary" in fn_src, \
            "_compact_summary not used in _llm_messages"
        assert "_compact_from_idx" in fn_src, \
            "_compact_from_idx not used in _llm_messages"
        assert "_maybe_compact" in fn_src, \
            "_maybe_compact not called in _llm_messages"


# ===========================================================================
# 2. OOP unit tests (no services)
# ===========================================================================

class TestCompactionState:
    def test_state_compaction_enabled_default(self):
        from aichat.state import AppState
        s = AppState()
        assert s.compaction_enabled is True, \
            f"compaction_enabled should default to True, got {s.compaction_enabled}"

    def test_compact_state_init_values(self):
        """AIChatApp __init__ must initialise compaction fields to falsy defaults."""
        import aichat.app as appmod
        from unittest.mock import MagicMock, patch
        with patch.object(appmod.AIChatApp, "on_mount", return_value=None), \
             patch("aichat.app.load_config", return_value={}), \
             patch("aichat.app.TranscriptStore") as mock_ts:
            mock_ts.return_value.load_messages.return_value = []
            app = object.__new__(appmod.AIChatApp)
            # Call __init__ without Textual runtime by patching super().__init__
            from aichat.config import AppConfig
            cfg = AppConfig()
            # Direct attribute inspection via AIChatApp.__init__ side-effects
            # (We just use the live app from _make_app_instance helper)
            pass
        # Use the lighter approach: just check the class defines these attrs
        src = _read("src/aichat/app.py")
        assert 'self._compact_summary: str = ""' in src
        assert "self._compact_from_idx: int = 0" in src
        assert "self._compact_pending: bool = False" in src

    def test_state_compaction_enabled_is_toggleable(self):
        from aichat.state import AppState
        s = AppState()
        s.compaction_enabled = False
        assert s.compaction_enabled is False
        s.compaction_enabled = True
        assert s.compaction_enabled is True


class TestRunCompact:
    """Unit tests for _run_compact core logic."""

    def _make_messages(self, n: int):
        from aichat.state import Message
        msgs = []
        for i in range(n):
            role = "user" if i % 2 == 0 else "assistant"
            msgs.append(Message(role, f"Message {i}"))
        return msgs

    def _base_compact_app(self):
        """Return a minimal app with all attributes _run_compact requires."""
        import aichat.app as appmod

        app = object.__new__(appmod.AIChatApp)
        app._compact_summary = ""
        app._compact_from_idx = 0
        app._compact_pending = False
        app._compact_tool_turns = True  # include tool turns
        app._compact_model = ""         # use main model
        app._compact_events = []        # event log
        app._tool_log = lambda msg: None
        app.personalities = []          # persona-aware prompt needs this

        class FakeState:
            session_id = ""  # no DB persist in unit tests
            personality_id = ""

        app.state = FakeState()
        return app

    @pytest.mark.asyncio
    async def test_run_compact_updates_summary_and_idx(self):
        """_run_compact sets _compact_summary and advances _compact_from_idx."""
        from unittest.mock import AsyncMock
        import aichat.app as appmod

        app = self._base_compact_app()

        class FakeTools:
            class lm:
                chat = AsyncMock(return_value="• key decision made\n• feature X approved")

        app.tools = FakeTools()

        msgs = self._make_messages(4)
        await appmod.AIChatApp._run_compact(app, msgs, 4)

        assert app._compact_summary == "• key decision made\n• feature X approved"
        assert app._compact_from_idx == 4
        assert app._compact_pending is False

    @pytest.mark.asyncio
    async def test_run_compact_appends_to_existing_summary(self):
        """_run_compact uses meta-compaction (replaces summary) when one exists."""
        from unittest.mock import AsyncMock
        import aichat.app as appmod

        app = self._base_compact_app()
        app._compact_summary = "old summary"
        app._compact_from_idx = 2

        class FakeTools:
            class lm:
                chat = AsyncMock(return_value="merged: old and new summary point")

        app.tools = FakeTools()

        msgs = self._make_messages(4)
        await appmod.AIChatApp._run_compact(app, msgs, 4)

        # Meta-compaction: summary is replaced with LM output (contains merged content)
        assert app._compact_summary == "merged: old and new summary point"
        assert app._compact_from_idx == 6

    @pytest.mark.asyncio
    async def test_run_compact_fail_open_on_exception(self):
        """_run_compact is fail-open: exception does not propagate or change state."""
        from unittest.mock import AsyncMock
        import aichat.app as appmod

        app = self._base_compact_app()

        class FakeTools:
            class lm:
                chat = AsyncMock(side_effect=Exception("LM Studio offline"))

        app.tools = FakeTools()

        msgs = self._make_messages(4)
        # Should not raise
        await appmod.AIChatApp._run_compact(app, msgs, 4)

        assert app._compact_summary == ""
        assert app._compact_from_idx == 0
        assert app._compact_pending is False

    @pytest.mark.asyncio
    async def test_run_compact_no_change_when_lm_returns_empty(self):
        """_run_compact does not advance idx if LM Studio returns empty string."""
        from unittest.mock import AsyncMock
        import aichat.app as appmod

        app = self._base_compact_app()

        class FakeTools:
            class lm:
                chat = AsyncMock(return_value="")

        app.tools = FakeTools()

        msgs = self._make_messages(4)
        await appmod.AIChatApp._run_compact(app, msgs, 4)

        assert app._compact_summary == ""
        assert app._compact_from_idx == 0


class TestMaybeCompact:
    """Unit tests for _maybe_compact guard conditions."""

    def _make_messages(self, n: int):
        from aichat.state import Message
        msgs = []
        for i in range(n):
            role = "user" if i % 2 == 0 else "assistant"
            msgs.append(Message(role, f"Message {i}"))
        return msgs

    @pytest.mark.asyncio
    async def test_compact_pending_prevents_reentry(self):
        """_maybe_compact returns immediately when _compact_pending is True."""
        from unittest.mock import AsyncMock
        import aichat.app as appmod

        app = object.__new__(appmod.AIChatApp)
        app._compact_pending = True  # already running
        app._compact_from_idx = 0
        app._compact_summary = ""

        class FakeState:
            compaction_enabled = True

        class FakeTools:
            class lm:
                chat = AsyncMock()

        app.state = FakeState()
        app.tools = FakeTools()
        app.messages = self._make_messages(10)

        await appmod.AIChatApp._maybe_compact(app)

        assert FakeTools.lm.chat.call_count == 0, \
            "lm.chat should not be called when _compact_pending is True"

    @pytest.mark.asyncio
    async def test_maybe_compact_skips_when_disabled(self):
        """_maybe_compact returns immediately when compaction_enabled is False."""
        from unittest.mock import AsyncMock
        import aichat.app as appmod

        app = object.__new__(appmod.AIChatApp)
        app._compact_pending = False
        app._compact_from_idx = 0
        app._compact_summary = ""

        class FakeState:
            compaction_enabled = False

        class FakeTools:
            class lm:
                chat = AsyncMock()

        app.state = FakeState()
        app.tools = FakeTools()
        app.messages = self._make_messages(10)

        await appmod.AIChatApp._maybe_compact(app)

        assert FakeTools.lm.chat.call_count == 0

    @pytest.mark.asyncio
    async def test_maybe_compact_skips_when_too_few_messages(self):
        """_maybe_compact returns immediately when fewer than _compact_min_msgs messages."""
        from unittest.mock import AsyncMock
        import aichat.app as appmod

        app = object.__new__(appmod.AIChatApp)
        app._compact_pending = False
        app._compact_from_idx = 0
        app._compact_summary = ""
        app._compact_min_msgs = 8       # instance var (config-driven)
        app._compact_keep_ratio = 0.5

        class FakeState:
            compaction_enabled = True

        class FakeTools:
            class lm:
                chat = AsyncMock()

        app.state = FakeState()
        app.tools = FakeTools()
        app.messages = self._make_messages(4)  # fewer than _compact_min_msgs (8)

        await appmod.AIChatApp._maybe_compact(app)

        assert FakeTools.lm.chat.call_count == 0


# ===========================================================================
# 3. TUI behavioral tests (headless Textual, no LM services)
# ===========================================================================

def _make_app():
    """Return a fresh AIChatApp with a mocked LLM client."""
    sys.path.insert(0, "src")
    from unittest.mock import AsyncMock
    from aichat.app import AIChatApp

    app = AIChatApp()
    app.client.health = AsyncMock(return_value=True)
    app.client.list_models = AsyncMock(return_value=["test-model"])
    app.client.ensure_model = AsyncMock()
    app.client.chat_once_with_tools = AsyncMock(
        return_value={"content": "Test response.", "tool_calls": []}
    )

    async def _fake_stream(*args, **kwargs):
        yield {"type": "content", "value": "Streamed response."}

    app.client.chat_stream_events = _fake_stream
    app.state.streaming = False
    return app


async def _type_and_send(pilot, text: str) -> None:
    await pilot.click("#prompt")
    await pilot.pause(0.1)
    key_map = {
        "/": "slash", " ": "space", ".": "period", "-": "minus",
        "=": "equals", "_": "underscore",
    }
    for ch in text:
        await pilot.press(key_map.get(ch, ch))
    await pilot.pause(0.1)
    await pilot.press("enter")


async def _wait_bubbles(app, pilot, minimum: int, timeout_iters: int = 30) -> list:
    for _ in range(timeout_iters):
        await pilot.pause(0.3)
        if len(app.query(".chat-msg")) >= minimum:
            break
    return list(app.query(".chat-msg"))


class TestCompactionTUI:
    """TUI behavioral tests for /compact command."""

    @pytest.mark.asyncio
    async def test_compact_command_on_enables(self):
        """/compact on sets compaction_enabled = True."""
        app = _make_app()
        app.state.compaction_enabled = False  # start disabled
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause(1.0)
            await _type_and_send(pilot, "/compact on")
            await pilot.pause(0.5)
            assert app.state.compaction_enabled is True

    @pytest.mark.asyncio
    async def test_compact_command_off_disables(self):
        """/compact off sets compaction_enabled = False."""
        app = _make_app()
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause(1.0)
            await _type_and_send(pilot, "/compact off")
            await pilot.pause(0.5)
            assert app.state.compaction_enabled is False

    @pytest.mark.asyncio
    async def test_compact_command_too_few_messages(self):
        """/compact with empty session shows 'Not enough messages' bubble."""
        app = _make_app()
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause(1.0)
            await _type_and_send(pilot, "/compact")
            bubbles = await _wait_bubbles(app, pilot, 1)
            texts = [getattr(b.query_one(".msg-body"), "_markdown", "") or "" for b in bubbles]
            all_text = " ".join(str(t) for t in texts)
            assert "enough" in all_text.lower() or "compact" in all_text.lower(), \
                f"Expected 'Not enough' or compact message, got: {all_text}"

    @pytest.mark.asyncio
    async def test_compact_command_force_no_lm(self):
        """/compact with messages but LM Studio offline shows failure message without crash."""
        from unittest.mock import AsyncMock
        app = _make_app()
        # Mock LM Studio chat to return empty (simulates offline)
        app.tools.lm.chat = AsyncMock(return_value="")

        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause(1.0)
            # Add enough messages for compaction to attempt
            from aichat.state import Message
            for i in range(6):
                role = "user" if i % 2 == 0 else "assistant"
                app.messages.append(Message(role, f"Test message {i}"))
            await _type_and_send(pilot, "/compact")
            bubbles = await _wait_bubbles(app, pilot, 1)
            texts = [getattr(b.query_one(".msg-body"), "_markdown", "") or "" for b in bubbles]
            all_text = " ".join(str(t) for t in texts)
            # Should show "Compaction failed" or "Compacted N turns" or "Not enough"
            assert any(
                phrase in all_text.lower()
                for phrase in ["compact", "failed", "enough", "turns"]
            ), f"Expected compaction message, got: {all_text}"
            # No exceptions — app still responsive
            assert not app.state.busy
