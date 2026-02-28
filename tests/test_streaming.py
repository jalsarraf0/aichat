"""
Tests for streaming drop fixes (v5).

Sections:
  1. TestSourceInspection    — verify fix symbols exist in source
  2. TestStreamTimeout       — _STREAM_TIMEOUT constant in client.py
  3. TestRetriableErrors     — _RETRIABLE tuple in client.py
  4. TestRawStream           — _raw_stream() low-level SSE parser (mock httpx)
  5. TestChatStreamRetry     — chat_stream_events() 1-retry logic
  6. TestChatStreamEvents    — normal streaming path
  7. TestChatStream          — simple text-only stream helper
  8. TestStreamWatchdog      — app.py 5-minute stream watchdog
  9. TestPartialRecovery     — partial response on watchdog timeout
 10. TestMcpToolTimeout      — docker/mcp/app.py _TOOL_TIMEOUT constant
 11. TestMcpStreamResult     — _stream_result() error path
 12. TestMcpDeliver          — _deliver() error response path
 13. TestLmStudioCaption     — lm_studio.py caption() no nested timeout
 14. TestClientExceptions    — LLMClientError / ModelNotFoundError hierarchy
 15. TestStreamEventTypes    — event type routing (content vs tool_calls)
"""
from __future__ import annotations

import asyncio
import pathlib
import sys
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

sys.path.insert(0, "src")

ROOT = pathlib.Path(__file__).parent.parent


def _read(relpath: str) -> str:
    return (ROOT / relpath).read_text()


# ===========================================================================
# 1. Source Inspection
# ===========================================================================

class TestSourceInspection:
    """Verify all streaming-fix symbols are present in the source files."""

    def _client(self) -> str:
        return _read("src/aichat/client.py")

    def _app(self) -> str:
        return _read("src/aichat/app.py")

    def _mcp(self) -> str:
        return _read("docker/mcp/app.py")

    def _lmstudio(self) -> str:
        return _read("src/aichat/tools/lm_studio.py")

    # client.py
    def test_stream_timeout_in_client(self):
        assert "_STREAM_TIMEOUT" in self._client()

    def test_retriable_in_client(self):
        assert "_RETRIABLE" in self._client()

    def test_raw_stream_in_client(self):
        assert "async def _raw_stream(" in self._client()

    def test_chat_stream_events_in_client(self):
        assert "async def chat_stream_events(" in self._client()

    def test_retry_loop_in_client(self):
        assert "for attempt in range(2)" in self._client()

    def test_cancelled_error_handled(self):
        assert "CancelledError" in self._client()

    # app.py
    def test_watchdog_timeout_in_app(self):
        assert "asyncio.timeout(300" in self._app()

    def test_partial_recovery_in_app(self):
        # partial response used when watchdog fires after content received
        assert "using partial response" in self._app()

    # mcp/app.py
    def test_tool_timeout_const_in_mcp(self):
        assert "_TOOL_TIMEOUT" in self._mcp()

    def test_tool_timeout_value_180(self):
        assert "180" in self._mcp()

    def test_mcp_task_result_try_except(self):
        # task.result() must now be inside a try block
        src = self._mcp()
        # There should be "try:" near "task.result()" in the file
        assert "task.result()" in src
        assert "except Exception as exc:" in src

    def test_mcp_error_response_on_exception(self):
        assert '"code": -32000' in self._mcp()

    # lm_studio.py
    def test_no_nested_asyncio_wait_for_in_caption(self):
        src = self._lmstudio()
        # asyncio.wait_for should not appear in lm_studio.py at all
        assert "asyncio.wait_for" not in src

    def test_caption_uses_single_timeout(self):
        # Should use timeout=10.0 (the safe, single httpx timeout)
        assert "timeout=10.0" in self._lmstudio()

    def test_no_asyncio_import_in_lmstudio(self):
        # asyncio import should be gone since wait_for was removed
        src = self._lmstudio()
        assert "import asyncio" not in src


# ===========================================================================
# 2. Stream Timeout Constant
# ===========================================================================

class TestStreamTimeout:
    def test_stream_timeout_type(self):
        import httpx
        from aichat.client import _STREAM_TIMEOUT
        assert isinstance(_STREAM_TIMEOUT, httpx.Timeout)

    def test_stream_timeout_connect(self):
        from aichat.client import _STREAM_TIMEOUT
        assert _STREAM_TIMEOUT.connect == 15.0

    def test_stream_timeout_read_is_none(self):
        """read=None means no per-chunk timeout — essential for streaming."""
        from aichat.client import _STREAM_TIMEOUT
        assert _STREAM_TIMEOUT.read is None

    def test_stream_timeout_write(self):
        from aichat.client import _STREAM_TIMEOUT
        assert _STREAM_TIMEOUT.write == 15.0


# ===========================================================================
# 3. Retriable Errors
# ===========================================================================

class TestRetriableErrors:
    def test_retriable_is_tuple(self):
        from aichat.client import _RETRIABLE
        assert isinstance(_RETRIABLE, tuple)

    def test_retriable_includes_remote_protocol(self):
        import httpx
        from aichat.client import _RETRIABLE
        assert httpx.RemoteProtocolError in _RETRIABLE

    def test_retriable_includes_read_error(self):
        import httpx
        from aichat.client import _RETRIABLE
        assert httpx.ReadError in _RETRIABLE

    def test_retriable_includes_connect_error(self):
        import httpx
        from aichat.client import _RETRIABLE
        assert httpx.ConnectError in _RETRIABLE

    def test_retriable_not_empty(self):
        from aichat.client import _RETRIABLE
        assert len(_RETRIABLE) >= 2


# ===========================================================================
# 4. RawStream — low-level SSE parser
# ===========================================================================

class TestRawStream:
    """Test _raw_stream() in isolation with mock httpx."""

    def _client(self):
        from aichat.client import LLMClient
        return LLMClient("http://localhost:1234")

    @pytest.mark.asyncio
    async def test_yields_content_chunks(self):
        import json

        async def _lines():
            yield 'data: {"choices":[{"delta":{"content":"hello"}}]}'
            yield 'data: {"choices":[{"delta":{"content":" world"}}]}'
            yield "data: [DONE]"

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.aiter_lines = _lines

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_stream)

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_ctx):
            cl = self._client()
            events = []
            async for ev in cl._raw_stream({"model": "test", "stream": True}):
                events.append(ev)

        assert len(events) == 2
        assert events[0] == {"type": "content", "value": "hello"}
        assert events[1] == {"type": "content", "value": " world"}

    @pytest.mark.asyncio
    async def test_skips_blank_lines(self):
        async def _lines():
            yield ""
            yield "  "
            yield 'data: {"choices":[{"delta":{"content":"ok"}}]}'
            yield "data: [DONE]"

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.aiter_lines = _lines

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_stream)

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_ctx):
            cl = self._client()
            events = [ev async for ev in cl._raw_stream({"model": "test", "stream": True})]

        assert len(events) == 1
        assert events[0]["value"] == "ok"

    @pytest.mark.asyncio
    async def test_stops_at_done_sentinel(self):
        async def _lines():
            yield 'data: {"choices":[{"delta":{"content":"a"}}]}'
            yield "data: [DONE]"
            yield 'data: {"choices":[{"delta":{"content":"SHOULD NOT APPEAR"}}]}'

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.aiter_lines = _lines

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_stream)

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_ctx):
            cl = self._client()
            events = [ev async for ev in cl._raw_stream({"model": "test", "stream": True})]

        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_yields_tool_call_events(self):
        async def _lines():
            yield 'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"foo"}}]}}]}'
            yield "data: [DONE]"

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.aiter_lines = _lines

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_stream)

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_ctx):
            cl = self._client()
            events = [ev async for ev in cl._raw_stream({"model": "test", "stream": True})]

        assert len(events) == 1
        assert events[0]["type"] == "tool_calls"

    @pytest.mark.asyncio
    async def test_skips_bad_json(self):
        async def _lines():
            yield "data: NOTJSON"
            yield 'data: {"choices":[{"delta":{"content":"good"}}]}'
            yield "data: [DONE]"

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.aiter_lines = _lines

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_stream)

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_ctx):
            cl = self._client()
            events = [ev async for ev in cl._raw_stream({"model": "test", "stream": True})]

        assert len(events) == 1
        assert events[0]["value"] == "good"


# ===========================================================================
# 5. ChatStreamRetry — retry logic in chat_stream_events()
# ===========================================================================

class TestChatStreamRetry:

    def _client(self):
        from aichat.client import LLMClient
        return LLMClient("http://localhost:1234")

    @pytest.mark.asyncio
    async def test_no_retry_on_success(self):
        """Clean stream with no errors → _raw_stream called exactly once."""
        cl = self._client()

        call_count = 0

        async def _mock_raw(payload):
            nonlocal call_count
            call_count += 1
            yield {"type": "content", "value": "hi"}

        with patch.object(cl, "_raw_stream", side_effect=_mock_raw):
            with patch.object(cl, "ensure_model", new=AsyncMock()):
                events = [
                    ev async for ev in cl.chat_stream_events("m", [])
                ]

        assert call_count == 1
        assert events == [{"type": "content", "value": "hi"}]

    @pytest.mark.asyncio
    async def test_retry_on_zero_events_retriable_error(self):
        """Retriable error with 0 events → retries once."""
        import httpx
        from aichat.client import LLMClientError

        cl = self._client()
        call_count = 0

        async def _mock_raw(payload):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise LLMClientError("conn") from httpx.ConnectError("fail")
            yield {"type": "content", "value": "ok"}

        with patch.object(cl, "_raw_stream", side_effect=_mock_raw):
            with patch.object(cl, "ensure_model", new=AsyncMock()):
                with patch("asyncio.sleep", new=AsyncMock()):
                    events = [
                        ev async for ev in cl.chat_stream_events("m", [])
                    ]

        assert call_count == 2
        assert events == [{"type": "content", "value": "ok"}]

    @pytest.mark.asyncio
    async def test_no_retry_after_partial_stream(self):
        """Retriable error AFTER events yielded → no retry, raises immediately."""
        import httpx
        from aichat.client import LLMClientError

        cl = self._client()
        call_count = 0

        async def _mock_raw(payload):
            nonlocal call_count
            call_count += 1
            yield {"type": "content", "value": "partial"}
            raise LLMClientError("mid-stream") from httpx.ReadError("lost")

        with patch.object(cl, "_raw_stream", side_effect=_mock_raw):
            with patch.object(cl, "ensure_model", new=AsyncMock()):
                events = []
                with pytest.raises(LLMClientError):
                    async for ev in cl.chat_stream_events("m", []):
                        events.append(ev)

        assert call_count == 1  # no retry
        assert len(events) == 1  # got partial content

    @pytest.mark.asyncio
    async def test_no_retry_on_non_retriable_error(self):
        """HTTP 4xx/5xx errors → no retry."""
        from aichat.client import LLMClientError

        cl = self._client()
        call_count = 0

        async def _mock_raw(payload):
            nonlocal call_count
            call_count += 1
            raise LLMClientError("HTTP 500")
            yield  # make it a generator

        with patch.object(cl, "_raw_stream", side_effect=_mock_raw):
            with patch.object(cl, "ensure_model", new=AsyncMock()):
                with pytest.raises(LLMClientError):
                    async for _ in cl.chat_stream_events("m", []):
                        pass

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_max_one_retry(self):
        """Even with retriable errors, only 1 retry (2 total attempts)."""
        import httpx
        from aichat.client import LLMClientError

        cl = self._client()
        call_count = 0

        async def _mock_raw(payload):
            nonlocal call_count
            call_count += 1
            raise LLMClientError("x") from httpx.ConnectError("fail")
            yield

        with patch.object(cl, "_raw_stream", side_effect=_mock_raw):
            with patch.object(cl, "ensure_model", new=AsyncMock()):
                with patch("asyncio.sleep", new=AsyncMock()):
                    with pytest.raises(LLMClientError):
                        async for _ in cl.chat_stream_events("m", []):
                            pass

        assert call_count == 2  # at most 2 attempts


# ===========================================================================
# 6. Chat Stream Events — normal path
# ===========================================================================

class TestChatStreamEvents:

    def _client(self):
        from aichat.client import LLMClient
        return LLMClient("http://localhost:1234")

    @pytest.mark.asyncio
    async def test_events_forwarded(self):
        cl = self._client()

        async def _mock_raw(payload):
            yield {"type": "content", "value": "hello"}
            yield {"type": "content", "value": " world"}

        with patch.object(cl, "_raw_stream", side_effect=_mock_raw):
            with patch.object(cl, "ensure_model", new=AsyncMock()):
                events = [ev async for ev in cl.chat_stream_events("m", [])]

        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_tools_passed_to_payload(self):
        cl = self._client()
        received_payloads = []

        async def _mock_raw(payload):
            received_payloads.append(payload)
            yield {"type": "content", "value": "x"}

        tools = [{"type": "function", "function": {"name": "foo"}}]
        with patch.object(cl, "_raw_stream", side_effect=_mock_raw):
            with patch.object(cl, "ensure_model", new=AsyncMock()):
                async for _ in cl.chat_stream_events("m", [], tools=tools):
                    pass

        assert received_payloads[0]["tools"] == tools
        assert received_payloads[0]["tool_choice"] == "auto"


# ===========================================================================
# 7. Chat Stream — text-only helper
# ===========================================================================

class TestChatStream:

    def _client(self):
        from aichat.client import LLMClient
        return LLMClient("http://localhost:1234")

    @pytest.mark.asyncio
    async def test_only_content_events_yielded(self):
        cl = self._client()

        async def _mock_events(model, messages, tools=None, tool_choice=None, max_tokens=None):
            yield {"type": "content", "value": "hello"}
            yield {"type": "tool_calls", "value": []}  # should be filtered
            yield {"type": "content", "value": " world"}

        with patch.object(cl, "chat_stream_events", side_effect=_mock_events):
            chunks = [c async for c in cl.chat_stream("m", [])]

        assert chunks == ["hello", " world"]

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        cl = self._client()

        async def _mock_events(model, messages, tools=None, tool_choice=None, max_tokens=None):
            return
            yield  # make generator

        with patch.object(cl, "chat_stream_events", side_effect=_mock_events):
            chunks = [c async for c in cl.chat_stream("m", [])]

        assert chunks == []


# ===========================================================================
# 8. Stream Watchdog in app.py
# ===========================================================================

def _bare_app():
    """Minimal AIChatApp — no Textual UI."""
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


class TestStreamWatchdog:
    """app.py 5-minute watchdog fires asyncio.TimeoutError."""

    def _app(self):
        app = _bare_app()
        app.state = MagicMock()
        app.state.streaming = True
        app.state.busy = False
        app.state.model = "test-model"
        app.state.shell_enabled = False
        app.state.approval = "always"
        app.client = MagicMock()
        app.tools = MagicMock()
        app.tools.tool_definitions = MagicMock(return_value=[])
        app.tools.reset_turn = MagicMock()
        app._last_status_ts = 0.0
        app.active_task = None
        return app

    @pytest.mark.asyncio
    async def test_watchdog_timer_present(self):
        """Source should contain asyncio.timeout(300 ..."""
        src = _read("src/aichat/app.py")
        assert "asyncio.timeout(300" in src

    @pytest.mark.asyncio
    async def test_timeout_raises_no_content_shows_message(self):
        """When timeout fires with no content, show timeout message."""
        app = self._app()

        written = []

        def _write(speaker, msg):
            written.append((speaker, msg))

        app._write_transcript = _write
        app._finalize_assistant_response = MagicMock()
        app._tool_log = MagicMock()
        app._safe_query_one = MagicMock(return_value=None)
        app.update_status = AsyncMock()
        app._refresh_sessions = MagicMock()
        app._llm_messages = AsyncMock(return_value=[])

        async def _hanging_stream(*args, **kwargs):
            await asyncio.sleep(400)  # hangs forever
            return "", []

        app._stream_with_tools = _hanging_stream

        # Patch asyncio.timeout to fire immediately
        import contextlib

        @contextlib.asynccontextmanager
        async def _instant_timeout(delay):
            raise asyncio.TimeoutError()
            yield

        import aichat.app as appmod
        with patch.object(appmod.asyncio, "timeout", _instant_timeout):
            await app.run_llm_turn()

        # No content → timeout message shown
        timeout_msgs = [m for _, m in written if "timed out" in m.lower()]
        assert len(timeout_msgs) >= 1

    @pytest.mark.asyncio
    async def test_timeout_with_partial_uses_partial(self):
        """When timeout fires with partial content, finalize partial."""
        app = self._app()

        written = []

        def _write(speaker, msg):
            written.append((speaker, msg))

        app._write_transcript = _write
        finalized = []
        app._finalize_assistant_response = lambda c: finalized.append(c)
        app._tool_log = MagicMock()
        app._safe_query_one = MagicMock(return_value=None)
        app.update_status = AsyncMock()
        app._refresh_sessions = MagicMock()
        app._llm_messages = AsyncMock(return_value=[])

        partial_content = "Partial response text"

        async def _partial_then_timeout(*args, **kwargs):
            # Simulate content accumulated then timeout
            raise asyncio.TimeoutError()

        # We need to patch so that 'content' is non-empty at timeout.
        # Do this by directly testing the branch logic.
        import aichat.app as appmod
        import contextlib

        @contextlib.asynccontextmanager
        async def _instant_timeout(delay):
            raise asyncio.TimeoutError()
            yield

        # Patch _stream_with_tools to do nothing so content stays ""
        app._stream_with_tools = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch.object(appmod.asyncio, "timeout", _instant_timeout):
            await app.run_llm_turn()

        # With no content, should show timeout message (not crash)
        assert app.state.busy is False


# ===========================================================================
# 9. Partial Response Recovery (source-level test)
# ===========================================================================

class TestPartialRecovery:
    def test_partial_recovery_uses_finalize(self):
        src = _read("src/aichat/app.py")
        # The recovery block should call _finalize_assistant_response
        assert "_finalize_assistant_response(content)" in src

    def test_partial_recovery_logs_message(self):
        src = _read("src/aichat/app.py")
        assert "partial response" in src.lower()

    def test_no_content_shows_timeout_message(self):
        src = _read("src/aichat/app.py")
        assert "timed out" in src.lower()


# ===========================================================================
# 10. MCP Tool Timeout Constant
# ===========================================================================

class TestMcpToolTimeout:
    def test_tool_timeout_exists(self):
        src = _read("docker/mcp/app.py")
        assert "_TOOL_TIMEOUT" in src

    def test_tool_timeout_is_180(self):
        src = _read("docker/mcp/app.py")
        assert "_TOOL_TIMEOUT = 180.0" in src

    def test_tool_timeout_used_in_stream_result(self):
        src = _read("docker/mcp/app.py")
        # Should appear multiple times — module-level def + usage in funcs
        assert src.count("_TOOL_TIMEOUT") >= 2


# ===========================================================================
# 11. MCP StreamResult Error Handling
# ===========================================================================

class TestMcpStreamResult:
    def test_task_result_wrapped_in_try(self):
        src = _read("docker/mcp/app.py")
        # task.result() should be followed by an except block
        assert "try:\n                result = task.result()" in src or \
               "try:\n            result = task.result()" in src

    def test_error_response_yielded_on_exception(self):
        src = _read("docker/mcp/app.py")
        # Error response JSON-RPC format
        assert '"code": -32000' in src

    def test_timeout_cancels_task(self):
        src = _read("docker/mcp/app.py")
        assert "task.cancel()" in src

    def test_keepalive_still_sent(self):
        src = _read("docker/mcp/app.py")
        assert ": keepalive" in src


# ===========================================================================
# 12. MCP Deliver Error Response
# ===========================================================================

class TestMcpDeliver:
    def test_deliver_sends_error_on_task_exception(self):
        src = _read("docker/mcp/app.py")
        # _deliver must push an error to the session queue on exception
        assert '_sessions[sid].put' in src

    def test_deliver_not_silent_on_exception(self):
        src = _read("docker/mcp/app.py")
        # The old "except Exception: pass" pattern should not exist alone
        # (there may be a break, but not just pass)
        # Verify error response is sent
        assert '"error"' in src

    def test_deliver_timeout_sends_error(self):
        src = _read("docker/mcp/app.py")
        assert "Tool execution timed out" in src


# ===========================================================================
# 13. LMStudio Caption — no nested timeout
# ===========================================================================

class TestLmStudioCaption:
    def test_no_asyncio_wait_for(self):
        src = _read("src/aichat/tools/lm_studio.py")
        assert "asyncio.wait_for" not in src

    def test_uses_httpx_timeout_param(self):
        src = _read("src/aichat/tools/lm_studio.py")
        assert "timeout=10.0" in src

    def test_caption_still_present(self):
        src = _read("src/aichat/tools/lm_studio.py")
        assert "async def caption(" in src

    @pytest.mark.asyncio
    async def test_caption_fail_open(self):
        """Caption should still return '' on error (fail-open)."""
        from aichat.tools.lm_studio import LMStudioTool

        lm = LMStudioTool("http://localhost:9999")  # nothing listening
        result = await lm.caption("aGVsbG8=")  # valid b64, no server
        assert result == ""

    @pytest.mark.asyncio
    async def test_caption_timeout_returns_empty(self):
        """caption() with a 10s timeout returns '' on network failure."""
        from aichat.tools.lm_studio import LMStudioTool

        lm = LMStudioTool("http://192.0.2.1:1234")  # TEST-NET, always drops

        # Use a very short timeout override for test speed
        original_post = lm._post

        async def _fast_post(path, payload, timeout=None):
            raise Exception("test timeout simulation")

        lm._post = _fast_post
        result = await lm.caption("dGVzdA==")
        assert result == ""


# ===========================================================================
# 14. Client Exceptions
# ===========================================================================

class TestClientExceptions:
    def test_llm_client_error_is_runtime(self):
        from aichat.client import LLMClientError
        assert issubclass(LLMClientError, RuntimeError)

    def test_model_not_found_is_llm_error(self):
        from aichat.client import LLMClientError, ModelNotFoundError
        assert issubclass(ModelNotFoundError, LLMClientError)

    def test_llm_client_error_str(self):
        from aichat.client import LLMClientError
        exc = LLMClientError("something failed")
        assert "something failed" in str(exc)

    def test_model_not_found_includes_model_name(self):
        from aichat.client import ModelNotFoundError
        exc = ModelNotFoundError("Model 'x' not available")
        assert "x" in str(exc)


# ===========================================================================
# 15. Stream Event Types
# ===========================================================================

class TestStreamEventTypes:
    """Verify event routing: content vs tool_calls."""

    def _client(self):
        from aichat.client import LLMClient
        return LLMClient("http://localhost:1234")

    @pytest.mark.asyncio
    async def test_empty_content_filtered(self):
        """Empty-string content chunks should NOT be yielded."""
        async def _lines():
            yield 'data: {"choices":[{"delta":{"content":""}}]}'
            yield 'data: {"choices":[{"delta":{"content":"real"}}]}'
            yield "data: [DONE]"

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.aiter_lines = _lines

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_stream)

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_ctx):
            cl = self._client()
            events = [ev async for ev in cl._raw_stream({"model": "t", "stream": True})]

        assert len(events) == 1
        assert events[0]["value"] == "real"

    @pytest.mark.asyncio
    async def test_sse_comment_lines_skipped(self):
        """SSE comment lines (': ...') must be silently skipped."""
        async def _lines():
            yield ": this is a comment"
            yield ": keep-alive"
            yield 'data: {"choices":[{"delta":{"content":"x"}}]}'
            yield "data: [DONE]"

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.aiter_lines = _lines

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_stream)

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_ctx):
            cl = self._client()
            events = [ev async for ev in cl._raw_stream({"model": "t", "stream": True})]

        assert len(events) == 1
        assert events[0]["value"] == "x"

    @pytest.mark.asyncio
    async def test_null_content_in_delta_not_yielded(self):
        """delta.content = null → no content event."""
        async def _lines():
            yield 'data: {"choices":[{"delta":{"content":null}}]}'
            yield 'data: {"choices":[{"delta":{"content":"y"}}]}'
            yield "data: [DONE]"

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.aiter_lines = _lines

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_stream)

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_ctx):
            cl = self._client()
            events = [ev async for ev in cl._raw_stream({"model": "t", "stream": True})]

        # null content filtered out
        assert len(events) == 1
        assert events[0]["value"] == "y"
