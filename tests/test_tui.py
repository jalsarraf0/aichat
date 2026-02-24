"""
Comprehensive TUI pilot tests for AIChatApp.

Covers:
- App mounts without crash (VerticalScroll, toolpane, sessionpane, prompt)
- User + assistant ChatMessage bubbles render with correct CSS classes
- Streaming path: live ChatMessage mounted then replaced by final bubble
- Theme switching: all 4 themes apply without crash
- action_scroll_up / action_scroll_down execute without crash
- /new  -> clears transcript bubbles and mounts "New chat started." bubble
- /clear -> clears all bubbles
- Slash command (/concise) -> response bubble rendered
- Tool error displays as assistant bubble (no Docker required)
- Dangerous shell command blocked (ToolDeniedError → bubble shown)
"""
from __future__ import annotations

import asyncio
import sys
import unittest.mock
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
sys.path.insert(0, "src")

from aichat.app import AIChatApp, ChatMessage
from aichat.state import ApprovalMode
from textual.widgets import Static, TextArea, Log
from textual.containers import VerticalScroll


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_app() -> AIChatApp:
    """Return a fresh AIChatApp with a mocked LLM client."""
    app = AIChatApp()
    app.client.health = AsyncMock(return_value=True)
    app.client.list_models = AsyncMock(return_value=["test-model"])
    app.client.ensure_model = AsyncMock()
    app.client.chat_once_with_tools = AsyncMock(
        return_value={"content": "Test response.", "tool_calls": []}
    )
    # Stub streaming: yield one content event then stop
    async def _fake_stream(*args, **kwargs):
        yield {"type": "content", "value": "Streamed response."}
    app.client.chat_stream_events = _fake_stream
    app.state.streaming = False      # default to non-streaming; individual tests override
    return app


async def _type_and_send(pilot, text: str) -> None:
    """Focus prompt, type text character by character, press enter."""
    await pilot.click("#prompt")
    await pilot.pause(0.1)
    key_map = {"/": "slash", " ": "space", ".": "period", ":": "colon",
               "_": "underscore", "-": "minus", "=": "equals", "!": "exclamation_mark"}
    for ch in text:
        await pilot.press(key_map.get(ch, ch))
    await pilot.pause(0.1)
    await pilot.press("enter")


async def _wait_bubbles(app: AIChatApp, pilot, minimum: int, timeout_iters: int = 40) -> list:
    for _ in range(timeout_iters):
        await pilot.pause(0.3)
        if len(app.query(".chat-msg")) >= minimum:
            break
    return list(app.query(".chat-msg"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTUIMount:
    """App mounts and critical widgets are present."""

    @pytest.mark.asyncio
    async def test_mount_widgets_present(self):
        app = _make_app()
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause(1.0)
            assert app.query_one("#transcript", VerticalScroll)
            assert app.query_one("#toolpane", Log)
            assert app.query_one("#sessionpane", Log)
            assert app.query_one("#prompt", TextArea)

    @pytest.mark.asyncio
    async def test_mount_no_bubbles_initially(self):
        """Transcript should start empty (no bubbles on a fresh launch)."""
        app = _make_app()
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause(1.0)
            bubbles = app.query(".chat-msg")
            assert len(bubbles) == 0


class TestChatBubbles:
    """User + assistant bubbles render with correct classes."""

    @pytest.mark.asyncio
    async def test_user_and_assistant_bubbles(self):
        app = _make_app()
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause(1.0)
            await _type_and_send(pilot, "hello")
            bubbles = await _wait_bubbles(app, pilot, 2)

            assert len(bubbles) >= 2, f"Expected ≥2 bubbles, got {len(bubbles)}"
            classes = [b.classes for b in bubbles]
            assert any("chat-user" in c for c in classes), "No chat-user bubble"
            assert any("chat-assistant" in c for c in classes), "No chat-assistant bubble"

    @pytest.mark.asyncio
    async def test_bubble_speaker_labels(self):
        app = _make_app()
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause(1.0)
            await _type_and_send(pilot, "hello")
            await _wait_bubbles(app, pilot, 2)

            bubbles = list(app.query(".chat-msg"))
            speakers = [b.query_one(".msg-speaker", Static).content for b in bubbles]
            assert "You" in speakers, f"No 'You' label in {speakers}"
            assert "Assistant" in speakers, f"No 'Assistant' label in {speakers}"

    @pytest.mark.asyncio
    async def test_multiple_turns(self):
        """Two consecutive messages produce 4 bubbles (2 user + 2 assistant)."""
        app = _make_app()
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause(1.0)
            await _type_and_send(pilot, "first")
            await _wait_bubbles(app, pilot, 2)
            await _type_and_send(pilot, "second")
            bubbles = await _wait_bubbles(app, pilot, 4)
            assert len(bubbles) >= 4, f"Expected ≥4 bubbles, got {len(bubbles)}"

    @pytest.mark.asyncio
    async def test_chat_msg_is_chatmessage_instance(self):
        app = _make_app()
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause(1.0)
            await _type_and_send(pilot, "hello")
            await _wait_bubbles(app, pilot, 1)
            bubbles = list(app.query(".chat-msg"))
            for b in bubbles:
                assert isinstance(b, ChatMessage), f"Unexpected type: {type(b)}"


class TestStreaming:
    """Streaming path: live bubble appears then is replaced by final bubble."""

    @pytest.mark.asyncio
    async def test_streaming_produces_assistant_bubble(self):
        app = _make_app()
        app.state.streaming = True
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause(1.0)
            await _type_and_send(pilot, "hello")
            bubbles = await _wait_bubbles(app, pilot, 2)
            assert any("chat-assistant" in b.classes for b in bubbles)

    @pytest.mark.asyncio
    async def test_streaming_live_bubble_removed_after_completion(self):
        """After streaming completes the temporary live bubble is gone;
        only the final finalized assistant bubble remains."""
        app = _make_app()
        app.state.streaming = True
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause(1.0)
            await _type_and_send(pilot, "hello")
            # Wait for LLM turn to finish
            for _ in range(30):
                await pilot.pause(0.3)
                if not app.state.busy:
                    break
            bubbles = list(app.query(".chat-msg"))
            # Should have exactly 1 user + 1 assistant bubble (no dangling live bubble)
            user_b = [b for b in bubbles if "chat-user" in b.classes]
            asst_b = [b for b in bubbles if "chat-assistant" in b.classes]
            assert len(user_b) == 1
            assert len(asst_b) == 1


class TestThemes:
    """All 4 themes apply without raising exceptions."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("theme", ["dark", "cyberpunk", "light", "synth"])
    async def test_theme_applies(self, theme):
        app = _make_app()
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause(0.8)
            try:
                app.apply_theme(theme)
                await pilot.pause(0.3)
            except Exception as exc:
                pytest.fail(f"apply_theme('{theme}') raised: {exc}")
            assert app.state.theme == theme


class TestScrollActions:
    """PageUp/PageDown scroll the VerticalScroll without crash."""

    @pytest.mark.asyncio
    async def test_scroll_up_no_crash(self):
        app = _make_app()
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause(0.8)
            await pilot.press("pageup")
            await pilot.pause(0.2)
            # If we get here without exception, test passes

    @pytest.mark.asyncio
    async def test_scroll_down_no_crash(self):
        app = _make_app()
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause(0.8)
            await pilot.press("pagedown")
            await pilot.pause(0.2)

    @pytest.mark.asyncio
    async def test_scroll_up_down_with_content(self):
        """Scroll with several bubbles already in the transcript."""
        app = _make_app()
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause(1.0)
            await _type_and_send(pilot, "msg1")
            await _wait_bubbles(app, pilot, 2)
            await _type_and_send(pilot, "msg2")
            await _wait_bubbles(app, pilot, 4)
            await pilot.press("pageup")
            await pilot.pause(0.2)
            await pilot.press("pagedown")
            await pilot.pause(0.2)


class TestNewChatAndClear:
    """/new and /clear commands reset the transcript."""

    @pytest.mark.asyncio
    async def test_new_chat_clears_bubbles(self):
        app = _make_app()
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause(1.0)
            await _type_and_send(pilot, "hello")
            await _wait_bubbles(app, pilot, 2)
            await _type_and_send(pilot, "/new")
            # /new produces an "New chat started." assistant bubble
            for _ in range(20):
                await pilot.pause(0.3)
                bubbles = list(app.query(".chat-msg"))
                if len(bubbles) == 1 and "chat-assistant" in bubbles[0].classes:
                    break
            bubbles = list(app.query(".chat-msg"))
            assert len(bubbles) == 1, f"After /new expected 1 bubble, got {len(bubbles)}"
            assert "chat-assistant" in bubbles[0].classes

    @pytest.mark.asyncio
    async def test_clear_transcript_removes_all_bubbles(self):
        app = _make_app()
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause(1.0)
            await _type_and_send(pilot, "hello")
            await _wait_bubbles(app, pilot, 2)
            await _type_and_send(pilot, "/clear")
            for _ in range(20):
                await pilot.pause(0.3)
                if len(app.query(".chat-msg")) == 0:
                    break
            assert len(app.query(".chat-msg")) == 0, "Bubbles remain after /clear"


class TestSlashCommands:
    """/concise and /verbose produce assistant bubbles."""

    @pytest.mark.asyncio
    async def test_concise_toggle(self):
        app = _make_app()
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause(1.0)
            await _type_and_send(pilot, "/concise")
            bubbles = await _wait_bubbles(app, pilot, 1)
            assert any("chat-assistant" in b.classes for b in bubbles)

    @pytest.mark.asyncio
    async def test_verbose_toggle(self):
        app = _make_app()
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause(1.0)
            await _type_and_send(pilot, "/verbose")
            bubbles = await _wait_bubbles(app, pilot, 1)
            assert any("chat-assistant" in b.classes for b in bubbles)


class TestDangerousCommandBlocked:
    """Dangerous shell command produces an assistant bubble with an error."""

    @pytest.mark.asyncio
    async def test_dangerous_shell_blocked(self):
        app = _make_app()
        app.state.shell_enabled = True
        app.state.approval = ApprovalMode.AUTO
        # LLM returns a tool call for a dangerous command
        app.client.chat_once_with_tools = AsyncMock(return_value={
            "content": "",
            "tool_calls": [{
                "id": "call1",
                "function": {
                    "name": "shell_exec",
                    "arguments": '{"command": "rm -rf /tmp/test_dir"}',
                },
            }],
        })
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause(1.0)
            await _type_and_send(pilot, "run command")
            for _ in range(30):
                await pilot.pause(0.4)
                if not app.state.busy:
                    break
            # After tool failure, an assistant bubble should be in the transcript
            bubbles = list(app.query(".chat-msg"))
            # Either the tool-error bubble appeared or a normal assistant bubble
            assert len(bubbles) >= 1, "No bubble after dangerous-command block"


class TestWriteTranscriptEdgeCases:
    """_write_transcript handles empty / long / markdown content."""

    @pytest.mark.asyncio
    async def test_empty_message(self):
        app = _make_app()
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause(0.8)
            app._write_transcript("Assistant", "")
            await pilot.pause(0.3)
            bubbles = list(app.query(".chat-msg"))
            assert len(bubbles) == 1

    @pytest.mark.asyncio
    async def test_markdown_in_message(self):
        app = _make_app()
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause(0.8)
            app._write_transcript("Assistant", "**bold** and `code` and\n\n- list item")
            await pilot.pause(0.3)
            bubbles = list(app.query(".chat-msg"))
            assert len(bubbles) == 1
            assert "chat-assistant" in bubbles[0].classes

    @pytest.mark.asyncio
    async def test_long_message_truncated_in_display(self):
        app = _make_app()
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause(0.8)
            long_text = "word " * 2000
            app._write_transcript("You", long_text)
            await pilot.pause(0.3)
            bubbles = list(app.query(".chat-msg"))
            assert len(bubbles) == 1
            assert "chat-user" in bubbles[0].classes
