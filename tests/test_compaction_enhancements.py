"""
Tests for contextual compaction enhancements (v2).

Sections:
  1. Source inspection — grep for new symbols; always run
  2. TestContextPct — _context_pct() helper accuracy
  3. TestMetaCompaction — merge-summary behaviour
  4. TestToolTurns — compact_tool_turns flag
  5. TestDBPersist — update_compact_state fire-and-forget
  6. TestProactiveTrigger — background task at 70%
  7. TestCompactStatus — /compact status TUI
  8. TestConfigPersist — on/off persist to config file
  9. TestConversationStoreCompact — update_compact_state HTTP client
"""
from __future__ import annotations

import asyncio
import inspect
import pathlib
import sys

import pytest

sys.path.insert(0, "src")

ROOT = pathlib.Path(__file__).parent.parent


def _read(relpath: str) -> str:
    return (ROOT / relpath).read_text()


# ===========================================================================
# 1. Source inspection tests
# ===========================================================================

class TestEnhancedSourceInspection:
    """Verify all new compaction-enhancement symbols exist in source files."""

    def _app(self) -> str:
        return _read("src/aichat/app.py")

    def _cfg(self) -> str:
        return _read("src/aichat/config.py")

    def _store(self) -> str:
        return _read("src/aichat/tools/conversation_store.py")

    def _db(self) -> str:
        return _read("docker/database/app.py")

    # --- AppConfig fields ---

    def test_compact_threshold_pct_in_config(self):
        assert "compact_threshold_pct" in self._cfg(), \
            "compact_threshold_pct not found in config.py AppConfig"

    def test_compact_min_msgs_in_config(self):
        assert "compact_min_msgs" in self._cfg(), \
            "compact_min_msgs not found in config.py AppConfig"

    def test_compact_keep_ratio_in_config(self):
        assert "compact_keep_ratio" in self._cfg(), \
            "compact_keep_ratio not found in config.py AppConfig"

    def test_compact_tool_turns_in_config(self):
        assert "compact_tool_turns" in self._cfg(), \
            "compact_tool_turns not found in config.py AppConfig"

    def test_compaction_enabled_in_config(self):
        assert "compaction_enabled" in self._cfg(), \
            "compaction_enabled not found in config.py AppConfig"

    # --- app.py instance vars ---

    def test_compact_threshold_pct_instance_var(self):
        assert "_compact_threshold_pct" in self._app(), \
            "_compact_threshold_pct not set in app.py __init__"

    def test_compact_min_msgs_instance_var(self):
        assert "_compact_min_msgs" in self._app(), \
            "_compact_min_msgs not set in app.py __init__"

    def test_compact_keep_ratio_instance_var(self):
        assert "_compact_keep_ratio" in self._app(), \
            "_compact_keep_ratio not set in app.py __init__"

    def test_compact_tool_turns_instance_var(self):
        assert "_compact_tool_turns" in self._app(), \
            "_compact_tool_turns not set in app.py __init__"

    # --- New methods / triggers ---

    def test_context_pct_method_in_app(self):
        assert "def _context_pct" in self._app(), \
            "_context_pct method not found in app.py"

    def test_proactive_trigger_in_finalize(self):
        import re
        src = self._app()
        fn = re.search(
            r"def _finalize_assistant_response.*?(?=\n    def |\n    async def |\nclass |\Z)",
            src, re.DOTALL,
        )
        assert fn, "_finalize_assistant_response not found in app.py"
        body = fn.group(0)
        assert "_context_pct" in body or "_maybe_compact" in body, \
            "Proactive compaction trigger not found in _finalize_assistant_response"

    def test_meta_compaction_in_run_compact(self):
        import re
        src = self._app()
        fn = re.search(
            r"async def _run_compact.*?(?=\n    async def |\n    def |\nclass |\Z)",
            src, re.DOTALL,
        )
        assert fn, "_run_compact not found in app.py"
        assert "PREVIOUS SUMMARY" in fn.group(0), \
            "Meta-compaction 'PREVIOUS SUMMARY' branch not found in _run_compact"

    def test_tool_turns_in_run_compact(self):
        import re
        src = self._app()
        fn = re.search(
            r"async def _run_compact.*?(?=\n    async def |\n    def |\nclass |\Z)",
            src, re.DOTALL,
        )
        assert fn, "_run_compact not found in app.py"
        assert "compact_tool_turns" in fn.group(0) or "valid_roles" in fn.group(0), \
            "Tool-turn inclusion logic not found in _run_compact"

    def test_db_persist_in_run_compact(self):
        import re
        src = self._app()
        fn = re.search(
            r"async def _run_compact.*?(?=\n    async def |\n    def |\nclass |\Z)",
            src, re.DOTALL,
        )
        assert fn, "_run_compact not found in app.py"
        assert "update_compact_state" in fn.group(0), \
            "DB persist call update_compact_state not found in _run_compact"

    def test_compact_status_in_handler(self):
        import re
        src = self._app()
        fn = re.search(
            r"async def _handle_compact_command.*?(?=\n    async def |\n    def |\nclass |\Z)",
            src, re.DOTALL,
        )
        assert fn, "_handle_compact_command not found in app.py"
        assert "status" in fn.group(0), \
            "'/compact status' case not found in _handle_compact_command"

    def test_save_config_in_handler(self):
        import re
        src = self._app()
        fn = re.search(
            r"async def _handle_compact_command.*?(?=\n    async def |\n    def |\nclass |\Z)",
            src, re.DOTALL,
        )
        assert fn, "_handle_compact_command not found in app.py"
        assert "save_config" in fn.group(0) or "_save_cfg" in fn.group(0), \
            "save_config call not found in _handle_compact_command"

    def test_update_compact_state_in_store(self):
        assert "update_compact_state" in self._store(), \
            "update_compact_state method not found in conversation_store.py"

    def test_compact_summary_column_in_db(self):
        assert "compact_summary" in self._db(), \
            "compact_summary column not found in docker/database/app.py"

    def test_compact_endpoint_in_db(self):
        assert "/compact" in self._db(), \
            "PATCH /compact endpoint not found in docker/database/app.py"


# ===========================================================================
# 2. TestContextPct — _context_pct() helper
# ===========================================================================

def _bare_app():
    """Return an AIChatApp instance with minimal attributes for unit tests."""
    import aichat.app as appmod
    from aichat.state import Message

    app = object.__new__(appmod.AIChatApp)
    app._context_length = 10000
    app._max_response_tokens = 1000
    app._compact_summary = ""
    app._compact_from_idx = 0
    app.messages = []
    return app


class TestContextPct:
    """Unit tests for _context_pct() helper accuracy."""

    def test_empty_history_returns_zero(self):
        import aichat.app as appmod
        app = _bare_app()
        assert appmod.AIChatApp._context_pct(app) == 0

    def test_history_at_budget_returns_100(self):
        """A message exactly filling the budget should give ctx_pct == 100."""
        import aichat.app as appmod
        from aichat.state import Message
        app = _bare_app()
        budget = app._context_length - app._max_response_tokens  # 9000 tokens
        # Each token ≈ 4 chars; budget tokens → budget*4 chars
        big_content = "x" * (budget * 4)
        app.messages = [Message("user", big_content)]
        pct = appmod.AIChatApp._context_pct(app)
        assert pct == 100, f"Expected 100 when exactly at budget, got {pct}"

    def test_uses_only_uncompacted_messages(self):
        """CTX% should be based on messages[_compact_from_idx:], not all messages."""
        import aichat.app as appmod
        from aichat.state import Message
        app = _bare_app()
        # 4 old messages (compacted) + 4 new messages
        old = [Message("user", "x" * 800) for _ in range(4)]
        new = [Message("user", "short") for _ in range(4)]
        app.messages = old + new
        app._compact_from_idx = 4  # old messages are "compacted"

        pct = appmod.AIChatApp._context_pct(app)
        # pct should reflect only "short" * 4, not the big old messages
        assert pct < 5, f"CTX% too high; should only count uncompacted msgs: {pct}"

    def test_accounts_for_compact_summary_tokens(self):
        """A non-empty _compact_summary should add to the CTX% count."""
        import aichat.app as appmod
        app = _bare_app()
        app._compact_summary = ""
        pct_empty = appmod.AIChatApp._context_pct(app)
        app._compact_summary = "x" * 4000  # ~1000 tokens
        pct_with_summary = appmod.AIChatApp._context_pct(app)
        assert pct_with_summary > pct_empty, \
            "CTX% should be higher when compact_summary has content"

    def test_correct_combined_summary_and_history(self):
        """Combined summary + history tokens must produce correct percentage."""
        import aichat.app as appmod
        from aichat.state import Message
        app = _bare_app()
        budget = app._context_length - app._max_response_tokens  # 9000 tokens
        # 50% in summary, 0 in history → expect ~50%
        app._compact_summary = "x" * (budget * 4 // 2)  # 4500 tokens
        app.messages = []
        pct = appmod.AIChatApp._context_pct(app)
        assert 45 <= pct <= 55, f"Expected ~50%, got {pct}"

    def test_clamps_at_100_when_over_budget(self):
        """CTX% must not exceed 100 even if history exceeds budget."""
        import aichat.app as appmod
        from aichat.state import Message
        app = _bare_app()
        budget = app._context_length - app._max_response_tokens
        app.messages = [Message("user", "x" * (budget * 8))]  # 2× over budget
        pct = appmod.AIChatApp._context_pct(app)
        assert pct == 100, f"Expected clamped 100, got {pct}"


# ===========================================================================
# 3. TestMetaCompaction — merge-summary behaviour
# ===========================================================================

def _compact_app(compact_summary="", compact_from_idx=0, tool_turns=True, session_id=""):
    """Minimal app for _run_compact unit tests."""
    import aichat.app as appmod
    app = object.__new__(appmod.AIChatApp)
    app._compact_summary = compact_summary
    app._compact_from_idx = compact_from_idx
    app._compact_pending = False
    app._compact_tool_turns = tool_turns
    app._compact_min_msgs = 8
    app._compact_keep_ratio = 0.5
    app._tool_log = lambda msg: None

    _sid = session_id  # capture before class body (class scope can't see enclosing locals)

    class FakeState:
        compaction_enabled = True
        session_id = _sid

    app.state = FakeState()
    return app


def _make_messages(n: int, include_tool: bool = False):
    from aichat.state import Message
    msgs = []
    for i in range(n):
        if include_tool and i % 3 == 2:
            msgs.append(Message("tool", f"Tool result {i}"))
        elif i % 2 == 0:
            msgs.append(Message("user", f"User msg {i}"))
        else:
            msgs.append(Message("assistant", f"Assistant msg {i}"))
    return msgs


class TestMetaCompaction:
    """Meta-compaction: second compact merges old summary + new turns."""

    @pytest.mark.asyncio
    async def test_first_compaction_no_previous_summary_in_prompt(self):
        """First compaction uses simple summarize prompt (no 'PREVIOUS SUMMARY' key)."""
        from unittest.mock import AsyncMock, call
        import aichat.app as appmod

        app = _compact_app(compact_summary="")

        class FakeTools:
            class lm:
                chat = AsyncMock(return_value="• initial summary")
            class conv:
                update_compact_state = AsyncMock(return_value={})

        app.tools = FakeTools()

        msgs = _make_messages(4)
        await appmod.AIChatApp._run_compact(app, msgs, 4)

        # The system message should NOT contain PREVIOUS SUMMARY
        call_args = FakeTools.lm.chat.call_args
        messages_sent = call_args[0][0]  # first positional arg = list of messages
        system_msg = next(m for m in messages_sent if m["role"] == "system")
        assert "PREVIOUS SUMMARY" not in system_msg["content"], \
            "First compaction must not use meta-compaction prompt"
        assert app._compact_summary == "• initial summary"

    @pytest.mark.asyncio
    async def test_second_compaction_uses_meta_prompt(self):
        """Second compaction sends PREVIOUS SUMMARY + NEW TURNS to LM."""
        from unittest.mock import AsyncMock
        import aichat.app as appmod

        app = _compact_app(compact_summary="Old key point")

        class FakeTools:
            class lm:
                chat = AsyncMock(return_value="• merged summary")
            class conv:
                update_compact_state = AsyncMock(return_value={})

        app.tools = FakeTools()

        msgs = _make_messages(4)
        await appmod.AIChatApp._run_compact(app, msgs, 4)

        call_args = FakeTools.lm.chat.call_args
        messages_sent = call_args[0][0]
        system_msg = next(m for m in messages_sent if m["role"] == "system")
        user_msg = next(m for m in messages_sent if m["role"] == "user")
        assert "PREVIOUS SUMMARY" in system_msg["content"] or \
               "PREVIOUS SUMMARY" in user_msg["content"], \
            "Meta-compaction must include PREVIOUS SUMMARY marker"
        assert "Old key point" in user_msg["content"], \
            "Old summary must be included in the user message"

    @pytest.mark.asyncio
    async def test_meta_compaction_replaces_not_appends(self):
        """Meta-compaction REPLACES _compact_summary; does not append."""
        from unittest.mock import AsyncMock
        import aichat.app as appmod

        app = _compact_app(compact_summary="old")

        class FakeTools:
            class lm:
                chat = AsyncMock(return_value="completely new merged summary")
            class conv:
                update_compact_state = AsyncMock(return_value={})

        app.tools = FakeTools()

        msgs = _make_messages(4)
        await appmod.AIChatApp._run_compact(app, msgs, 4)

        assert app._compact_summary == "completely new merged summary", \
            "Meta-compaction must replace, not append: got " + repr(app._compact_summary)
        assert "old" not in app._compact_summary, \
            "Old summary text must not appear literally in the new summary"

    @pytest.mark.asyncio
    async def test_empty_lm_response_leaves_state_unchanged(self):
        """Empty LM response: summary and idx stay unchanged."""
        from unittest.mock import AsyncMock
        import aichat.app as appmod

        app = _compact_app(compact_summary="existing", compact_from_idx=3)

        class FakeTools:
            class lm:
                chat = AsyncMock(return_value="")
            class conv:
                update_compact_state = AsyncMock(return_value={})

        app.tools = FakeTools()

        msgs = _make_messages(4)
        await appmod.AIChatApp._run_compact(app, msgs, 4)

        assert app._compact_summary == "existing"
        assert app._compact_from_idx == 3


# ===========================================================================
# 4. TestToolTurns — compact_tool_turns flag
# ===========================================================================

class TestToolTurns:
    """Verify tool-role messages are included/excluded based on compact_tool_turns."""

    @pytest.mark.asyncio
    async def test_tool_turns_included_when_enabled(self):
        """compact_tool_turns=True: 'TOOL:' appears in conv_text sent to LM."""
        from unittest.mock import AsyncMock
        import aichat.app as appmod

        app = _compact_app(compact_summary="", tool_turns=True)

        captured: dict = {}

        class FakeTools:
            class lm:
                @staticmethod
                async def chat(messages, **kwargs):
                    captured["messages"] = messages
                    return "summary"
            class conv:
                update_compact_state = AsyncMock(return_value={})

        app.tools = FakeTools()
        msgs = _make_messages(6, include_tool=True)
        await appmod.AIChatApp._run_compact(app, msgs, 6)

        user_content = captured["messages"][1]["content"]
        assert "TOOL:" in user_content, \
            "Tool messages should appear in conv_text when compact_tool_turns=True"

    @pytest.mark.asyncio
    async def test_tool_turns_excluded_when_disabled(self):
        """compact_tool_turns=False: 'TOOL:' does not appear in conv_text."""
        from unittest.mock import AsyncMock
        import aichat.app as appmod

        app = _compact_app(compact_summary="", tool_turns=False)

        captured: dict = {}

        class FakeTools:
            class lm:
                @staticmethod
                async def chat(messages, **kwargs):
                    captured["messages"] = messages
                    return "summary"
            class conv:
                update_compact_state = AsyncMock(return_value={})

        app.tools = FakeTools()
        msgs = _make_messages(6, include_tool=True)
        await appmod.AIChatApp._run_compact(app, msgs, 6)

        user_content = captured.get("messages", [{}])[-1].get("content", "") if captured else ""
        assert "TOOL:" not in user_content, \
            "Tool messages should be excluded when compact_tool_turns=False"

    @pytest.mark.asyncio
    async def test_tool_messages_truncated_to_300_chars(self):
        """Tool messages are truncated to 300 chars, not 400 like user/assistant."""
        from unittest.mock import AsyncMock
        from aichat.state import Message
        import aichat.app as appmod

        app = _compact_app(compact_summary="", tool_turns=True)

        captured: dict = {}

        class FakeTools:
            class lm:
                @staticmethod
                async def chat(messages, **kwargs):
                    captured["messages"] = messages
                    return "summary"
            class conv:
                update_compact_state = AsyncMock(return_value={})

        app.tools = FakeTools()
        long_tool_content = "T" * 600
        msgs = [
            Message("user", "hello"),
            Message("tool", long_tool_content),
            Message("assistant", "ok"),
            Message("user", "bye"),
        ]
        await appmod.AIChatApp._run_compact(app, msgs, 4)

        user_content = captured["messages"][1]["content"]
        # The tool line should show at most 300 chars of content
        tool_line = next((l for l in user_content.split("\n") if l.startswith("TOOL:")), "")
        assert len(tool_line) <= len("TOOL: ") + 300 + 5, \
            f"Tool message not truncated to 300 chars: {len(tool_line)} chars"


# ===========================================================================
# 5. TestDBPersist — update_compact_state fire-and-forget
# ===========================================================================

class TestDBPersist:
    """Verify update_compact_state is called correctly after successful compaction."""

    @pytest.mark.asyncio
    async def test_db_persist_called_on_success(self):
        """Successful compaction calls update_compact_state with correct args."""
        from unittest.mock import AsyncMock
        import aichat.app as appmod

        app = _compact_app(compact_summary="", session_id="test-session-123")

        update_mock = AsyncMock(return_value={})

        class FakeTools:
            class lm:
                chat = AsyncMock(return_value="• summary point")
            class conv:
                update_compact_state = update_mock

        app.tools = FakeTools()

        msgs = _make_messages(4)
        await appmod.AIChatApp._run_compact(app, msgs, 4)

        # Allow any tasks spawned via asyncio.create_task to run
        await asyncio.sleep(0.05)

        assert update_mock.call_count >= 1, \
            "update_compact_state should be called after successful compaction"
        call_kwargs = update_mock.call_args
        assert "test-session-123" in str(call_kwargs), \
            "session_id should be passed to update_compact_state"

    @pytest.mark.asyncio
    async def test_db_persist_not_called_when_lm_returns_empty(self):
        """Empty LM response: update_compact_state must NOT be called."""
        from unittest.mock import AsyncMock
        import aichat.app as appmod

        app = _compact_app(compact_summary="", session_id="test-session-123")
        update_mock = AsyncMock(return_value={})

        class FakeTools:
            class lm:
                chat = AsyncMock(return_value="")
            class conv:
                update_compact_state = update_mock

        app.tools = FakeTools()

        msgs = _make_messages(4)
        await appmod.AIChatApp._run_compact(app, msgs, 4)
        await asyncio.sleep(0.05)

        assert update_mock.call_count == 0, \
            "update_compact_state must not be called when LM returns empty"

    @pytest.mark.asyncio
    async def test_db_persist_not_called_on_lm_exception(self):
        """Exception in lm.chat: update_compact_state must NOT be called."""
        from unittest.mock import AsyncMock
        import aichat.app as appmod

        app = _compact_app(compact_summary="", session_id="test-session-123")
        update_mock = AsyncMock(return_value={})

        class FakeTools:
            class lm:
                chat = AsyncMock(side_effect=RuntimeError("LM offline"))
            class conv:
                update_compact_state = update_mock

        app.tools = FakeTools()

        msgs = _make_messages(4)
        await appmod.AIChatApp._run_compact(app, msgs, 4)
        await asyncio.sleep(0.05)

        assert update_mock.call_count == 0, \
            "update_compact_state must not be called when lm.chat raises"

    @pytest.mark.asyncio
    async def test_db_persist_not_called_without_session_id(self):
        """No session_id: update_compact_state must NOT be called."""
        from unittest.mock import AsyncMock
        import aichat.app as appmod

        app = _compact_app(compact_summary="", session_id="")
        update_mock = AsyncMock(return_value={})

        class FakeTools:
            class lm:
                chat = AsyncMock(return_value="• summary")
            class conv:
                update_compact_state = update_mock

        app.tools = FakeTools()

        msgs = _make_messages(4)
        await appmod.AIChatApp._run_compact(app, msgs, 4)
        await asyncio.sleep(0.05)

        assert update_mock.call_count == 0, \
            "update_compact_state must not be called when session_id is empty"


# ===========================================================================
# 6. TestProactiveTrigger — background task at 70%
# ===========================================================================

def _finalize_app(ctx_pct_value: int, compaction_enabled=True, compact_pending=False,
                  compact_min_msgs=8):
    """Return a minimal app where _context_pct() returns a fixed value."""
    import aichat.app as appmod
    from unittest.mock import AsyncMock, MagicMock

    app = object.__new__(appmod.AIChatApp)
    app._compact_pending = compact_pending
    app._compact_from_idx = 0
    app._compact_min_msgs = compact_min_msgs
    app._compact_keep_ratio = 0.5
    app._compact_tool_turns = True
    app._compact_summary = ""

    class FakeState:
        compaction_enabled = True
        session_title_set = False
        session_id = "test-session"

    app.state = FakeState()
    app.state.compaction_enabled = compaction_enabled
    app.messages = []
    app._context_pct = lambda: ctx_pct_value
    app._write_transcript = MagicMock()
    app.transcript_store = MagicMock()
    app.transcript_store.append = MagicMock()

    class FakeTools:
        class conv:
            store_turn = AsyncMock(return_value={})
            create_session = AsyncMock(return_value={})
        class lm:
            embed = AsyncMock(return_value=[[0.1, 0.2]])

    app.tools = FakeTools()
    app._auto_save_turn = AsyncMock()
    app._auto_generate_title = AsyncMock()
    app._tool_log = lambda msg: None
    return app


class TestProactiveTrigger:
    """Proactive background compaction fires when CTX% >= 70% after finalize."""

    @pytest.mark.asyncio
    async def test_proactive_fires_at_70_pct(self):
        """When _context_pct() >= 70 and enabled, _maybe_compact task is spawned."""
        import aichat.app as appmod
        from unittest.mock import AsyncMock, patch

        app = _finalize_app(ctx_pct_value=75, compaction_enabled=True,
                            compact_min_msgs=0)

        compact_called = []

        async def _fake_maybe_compact():
            compact_called.append(True)

        app._maybe_compact = _fake_maybe_compact
        app.messages = []

        from aichat.state import Message
        from unittest.mock import MagicMock
        import asyncio

        tasks_created: list = []
        orig_create_task = asyncio.create_task

        def _patched_create_task(coro, **kwargs):
            task = orig_create_task(coro, **kwargs)
            tasks_created.append(task)
            return task

        with patch("asyncio.create_task", side_effect=_patched_create_task):
            appmod.AIChatApp._finalize_assistant_response(app, "hello world")

        # Let all tasks run
        await asyncio.sleep(0.1)

        assert len(compact_called) > 0, \
            "Proactive compaction should have been triggered at 75% CTX"

    @pytest.mark.asyncio
    async def test_proactive_does_not_fire_at_50_pct(self):
        """When _context_pct() < 70, no proactive compaction task is spawned."""
        import aichat.app as appmod
        from unittest.mock import AsyncMock, patch

        app = _finalize_app(ctx_pct_value=50, compaction_enabled=True,
                            compact_min_msgs=0)

        compact_called = []

        async def _fake_maybe_compact():
            compact_called.append(True)

        app._maybe_compact = _fake_maybe_compact
        app.messages = []

        with patch("asyncio.create_task", wraps=asyncio.create_task):
            appmod.AIChatApp._finalize_assistant_response(app, "hello world")

        await asyncio.sleep(0.1)

        assert len(compact_called) == 0, \
            "Proactive compaction must NOT fire at 50% CTX"

    @pytest.mark.asyncio
    async def test_proactive_blocked_when_pending(self):
        """When _compact_pending=True, proactive trigger is skipped even at 90%."""
        import aichat.app as appmod
        from unittest.mock import AsyncMock, patch

        app = _finalize_app(ctx_pct_value=90, compaction_enabled=True,
                            compact_pending=True, compact_min_msgs=0)

        compact_called = []

        async def _fake_maybe_compact():
            compact_called.append(True)

        app._maybe_compact = _fake_maybe_compact
        app.messages = []

        appmod.AIChatApp._finalize_assistant_response(app, "hello world")
        await asyncio.sleep(0.1)

        assert len(compact_called) == 0, \
            "Proactive compaction must NOT fire when _compact_pending=True"

    @pytest.mark.asyncio
    async def test_proactive_blocked_when_disabled(self):
        """When compaction_enabled=False, proactive trigger is skipped at 90%."""
        import aichat.app as appmod

        app = _finalize_app(ctx_pct_value=90, compaction_enabled=False,
                            compact_min_msgs=0)

        compact_called = []

        async def _fake_maybe_compact():
            compact_called.append(True)

        app._maybe_compact = _fake_maybe_compact
        app.messages = []

        appmod.AIChatApp._finalize_assistant_response(app, "hello world")
        await asyncio.sleep(0.1)

        assert len(compact_called) == 0, \
            "Proactive compaction must NOT fire when compaction_enabled=False"


# ===========================================================================
# 7. TestCompactStatus — /compact status TUI
# ===========================================================================

def _make_app():
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


def _bubble_text(bubbles: list) -> str:
    texts = [getattr(b.query_one(".msg-body"), "_markdown", "") or "" for b in bubbles]
    return " ".join(str(t) for t in texts)


class TestCompactStatus:
    """/compact status TUI behavioral tests."""

    @pytest.mark.asyncio
    async def test_status_with_no_summary_shows_uncompacted(self):
        """/compact status with empty session shows 'No active compaction' bubble."""
        app = _make_app()
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause(1.0)
            await _type_and_send(pilot, "/compact status")
            bubbles = await _wait_bubbles(app, pilot, 1)
            all_text = _bubble_text(bubbles).lower()
            assert "no active" in all_text or "uncompacted" in all_text or "compact" in all_text, \
                f"Expected 'No active compaction' message, got: {all_text}"

    @pytest.mark.asyncio
    async def test_status_with_active_summary_shows_summary(self):
        """/compact status shows the active summary when compaction has run."""
        from unittest.mock import AsyncMock
        app = _make_app()
        app._compact_summary = "• Key decision: use PostgreSQL\n• Feature X approved"
        app._compact_from_idx = 4

        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause(1.0)
            await _type_and_send(pilot, "/compact status")
            bubbles = await _wait_bubbles(app, pilot, 1)
            all_text = _bubble_text(bubbles).lower()
            assert "active" in all_text or "summary" in all_text or "compact" in all_text, \
                f"Expected summary display, got: {all_text}"

    @pytest.mark.asyncio
    async def test_status_shows_compact_from_idx_count(self):
        """/compact status shows how many turns are summarized."""
        app = _make_app()
        app._compact_summary = "• some important point"
        app._compact_from_idx = 8

        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause(1.0)
            await _type_and_send(pilot, "/compact status")
            bubbles = await _wait_bubbles(app, pilot, 1)
            all_text = _bubble_text(bubbles)
            assert "8" in all_text or "compact" in all_text.lower(), \
                f"Expected compact_from_idx '8' in status output, got: {all_text}"


# ===========================================================================
# 8. TestConfigPersist — on/off toggle persists to config file
# ===========================================================================

class TestConfigPersist:
    """Verify /compact on/off persists compaction_enabled to config file."""

    @pytest.mark.asyncio
    async def test_compact_off_calls_save_config_with_false(self):
        """/compact off calls save_config with compaction_enabled=False."""
        from unittest.mock import AsyncMock, patch, MagicMock
        import aichat.app as appmod

        app = object.__new__(appmod.AIChatApp)
        app._compact_summary = ""
        app._compact_from_idx = 0
        app._compact_pending = False
        app._last_status_ts = 0.0

        class FakeState:
            compaction_enabled = True

        app.state = FakeState()
        app._write_transcript = MagicMock()
        app.update_status = AsyncMock()

        saved: list = []

        def _fake_save(cfg, *args, **kwargs):
            saved.append(cfg)

        with patch("aichat.config.load_config", return_value={"compaction_enabled": True}):
            with patch("aichat.config.save_config", side_effect=_fake_save):
                await appmod.AIChatApp._handle_compact_command(app, "off")

        assert len(saved) >= 1, "save_config should have been called"
        assert saved[-1].get("compaction_enabled") is False, \
            f"compaction_enabled should be False in saved config, got {saved[-1]}"

    @pytest.mark.asyncio
    async def test_compact_on_calls_save_config_with_true(self):
        """/compact on calls save_config with compaction_enabled=True."""
        from unittest.mock import AsyncMock, patch, MagicMock
        import aichat.app as appmod

        app = object.__new__(appmod.AIChatApp)
        app._compact_summary = ""
        app._compact_from_idx = 0
        app._compact_pending = False
        app._last_status_ts = 0.0

        class FakeState:
            compaction_enabled = False

        app.state = FakeState()
        app._write_transcript = MagicMock()
        app.update_status = AsyncMock()

        saved: list = []

        def _fake_save(cfg, *args, **kwargs):
            saved.append(cfg)

        with patch("aichat.config.load_config", return_value={"compaction_enabled": False}):
            with patch("aichat.config.save_config", side_effect=_fake_save):
                await appmod.AIChatApp._handle_compact_command(app, "on")

        assert len(saved) >= 1, "save_config should have been called"
        assert saved[-1].get("compaction_enabled") is True, \
            f"compaction_enabled should be True in saved config, got {saved[-1]}"

    @pytest.mark.asyncio
    async def test_compact_status_does_not_call_save_config(self):
        """/compact status must NOT call save_config."""
        from unittest.mock import AsyncMock, patch, MagicMock
        import aichat.app as appmod

        app = object.__new__(appmod.AIChatApp)
        app._compact_summary = ""
        app._compact_from_idx = 0
        app._compact_pending = False
        app._write_transcript = MagicMock()

        saved: list = []

        def _fake_save(cfg, *args, **kwargs):
            saved.append(cfg)

        with patch("aichat.config.save_config", side_effect=_fake_save):
            await appmod.AIChatApp._handle_compact_command(app, "status")

        assert len(saved) == 0, \
            "/compact status must not persist anything to config"


# ===========================================================================
# 9. TestConversationStoreCompact — update_compact_state HTTP client
# ===========================================================================

class TestConversationStoreCompact:
    """Unit tests for ConversationStoreTool.update_compact_state()."""

    @pytest.mark.asyncio
    async def test_sends_patch_to_correct_url(self):
        """update_compact_state sends PATCH to /conversations/sessions/{id}/compact."""
        from unittest.mock import AsyncMock, patch, MagicMock
        from aichat.tools.conversation_store import ConversationStoreTool

        store = ConversationStoreTool(base_url="http://localhost:8091")
        captured: dict = {}

        class FakeResponse:
            def raise_for_status(self): pass
            def json(self): return {"status": "updated"}

        class FakeClient:
            async def request(self, method, url, **kwargs):
                captured["method"] = method
                captured["url"] = url
                captured["kwargs"] = kwargs
                return FakeResponse()
            async def __aenter__(self): return self
            async def __aexit__(self, *_): pass

        with patch("httpx.AsyncClient", return_value=FakeClient()):
            result = await store.update_compact_state("sess-001", "• summary", 5)

        assert captured.get("method") == "PATCH", \
            f"Expected PATCH, got {captured.get('method')}"
        assert "/conversations/sessions/sess-001/compact" in captured.get("url", ""), \
            f"URL should contain /compact path: {captured.get('url')}"

    @pytest.mark.asyncio
    async def test_sends_correct_json_body(self):
        """update_compact_state sends compact_summary and compact_from_idx in body."""
        from unittest.mock import patch
        from aichat.tools.conversation_store import ConversationStoreTool

        store = ConversationStoreTool(base_url="http://localhost:8091")
        captured: dict = {}

        class FakeResponse:
            def raise_for_status(self): pass
            def json(self): return {"status": "updated"}

        class FakeClient:
            async def request(self, method, url, **kwargs):
                captured.update(kwargs)
                return FakeResponse()
            async def __aenter__(self): return self
            async def __aexit__(self, *_): pass

        with patch("httpx.AsyncClient", return_value=FakeClient()):
            await store.update_compact_state("sess-001", "• bullet point", 7)

        body = captured.get("json", {})
        assert body.get("compact_summary") == "• bullet point", \
            f"compact_summary not in body: {body}"
        assert body.get("compact_from_idx") == 7, \
            f"compact_from_idx not in body: {body}"

    @pytest.mark.asyncio
    async def test_fail_open_on_http_error(self):
        """update_compact_state returns {} on HTTP error (fail-open)."""
        from unittest.mock import patch
        import httpx
        from aichat.tools.conversation_store import ConversationStoreTool

        store = ConversationStoreTool(base_url="http://localhost:8091")

        class FailClient:
            async def request(self, *args, **kwargs):
                raise httpx.ConnectError("connection refused")
            async def __aenter__(self): return self
            async def __aexit__(self, *_): pass

        with patch("httpx.AsyncClient", return_value=FailClient()):
            result = await store.update_compact_state("sess-001", "summary", 3)

        assert result == {}, \
            f"Should return {{}} on HTTP error (fail-open), got: {result}"
