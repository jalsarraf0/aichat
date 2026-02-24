"""
End-to-end tests for all tool paths.

Service tests probe the real Docker endpoints (8091-8095) and skip
gracefully when a service is not reachable.

TUI tests run the full Textual app in headless mode with:
  - mocked LLM client (no real model needed)
  - real or mocked tool backends depending on availability
"""
from __future__ import annotations

import asyncio
import json
import sys
from unittest.mock import AsyncMock, patch

import httpx
import pytest

sys.path.insert(0, "src")

from aichat.app import AIChatApp, ChatMessage
from aichat.state import ApprovalMode
from aichat.tool_args import parse_tool_args
from aichat.tools.manager import ToolManager, ToolDeniedError
from textual.containers import VerticalScroll
from textual.widgets import Static


# ---------------------------------------------------------------------------
# Service availability helpers
# ---------------------------------------------------------------------------

SERVICES = {
    "fetch":       "http://localhost:8093/health",
    "memory":      "http://localhost:8094/health",
    "rssfeed":     "http://localhost:8091/health",
    # researchbox has no /health — probe its actual search endpoint
    "researchbox": "http://localhost:8092/search-feeds?topic=test",
    "toolkit":     "http://localhost:8095/health",
}


def _is_up(url: str) -> bool:
    try:
        r = httpx.get(url, timeout=2)
        return r.status_code < 500
    except Exception:
        return False


SERVICE_STATUS: dict[str, bool] = {
    name: _is_up(url) for name, url in SERVICES.items()
}


def skip_if_down(service: str):
    return pytest.mark.skipif(
        not SERVICE_STATUS.get(service, False),
        reason=f"{service} service not reachable (Docker not running?)",
    )


# ---------------------------------------------------------------------------
# Unit: tool argument parsing (XML format fix)
# ---------------------------------------------------------------------------

class TestXmlArgParsingE2E:
    """Confirm parse_tool_args handles all formats the LLM might emit."""

    def test_json_format(self):
        args, err = parse_tool_args("rss_latest", '{"topic": "Iran"}')
        assert err is None
        assert args["topic"] == "Iran"

    def test_xml_format_single(self):
        args, err = parse_tool_args(
            "rss_latest",
            "<arg_key>topic</arg_key><arg_value>Iran%20News</arg_value>",
        )
        assert err is None
        assert args["topic"] == "Iran News"

    def test_xml_format_multi(self):
        raw = (
            "<arg_key>url</arg_key><arg_value>https://example.com</arg_value>"
            "<arg_key>max_chars</arg_key><arg_value>2000</arg_value>"
        )
        args, err = parse_tool_args("web_fetch", raw)
        assert err is None
        assert args["url"] == "https://example.com"
        assert args["max_chars"] == 2000

    def test_yaml_format(self):
        args, err = parse_tool_args("rss_latest", "topic: AI news")
        assert err is None
        assert args["topic"] == "AI news"

    def test_shell_yaml_block(self):
        args, err = parse_tool_args("shell_exec", "command: |\n  echo hi\n  echo bye")
        assert err is None
        assert "echo hi" in args["command"]


# ---------------------------------------------------------------------------
# Service health probes (real Docker)
# ---------------------------------------------------------------------------

class TestServiceHealth:
    @skip_if_down("fetch")
    def test_fetch_health(self):
        r = httpx.get("http://localhost:8093/health", timeout=3)
        assert r.status_code == 200

    @skip_if_down("memory")
    def test_memory_health(self):
        r = httpx.get("http://localhost:8094/health", timeout=3)
        assert r.status_code == 200

    @skip_if_down("rssfeed")
    def test_rssfeed_health(self):
        r = httpx.get("http://localhost:8091/health", timeout=3)
        assert r.status_code == 200

    @skip_if_down("researchbox")
    def test_researchbox_health(self):
        # researchbox has no /health endpoint; probe the search endpoint instead
        r = httpx.get("http://localhost:8092/search-feeds", params={"topic": "test"}, timeout=3)
        assert r.status_code < 500

    @skip_if_down("toolkit")
    def test_toolkit_health(self):
        r = httpx.get("http://localhost:8095/health", timeout=3)
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Live service integration (real HTTP calls)
# ---------------------------------------------------------------------------

class TestFetchToolLive:
    @skip_if_down("fetch")
    @pytest.mark.asyncio
    async def test_fetch_example_com(self):
        from aichat.tools.fetch import FetchTool
        tool = FetchTool()
        result = await tool.fetch_url("https://example.com", max_chars=500)
        assert "text" in result
        assert len(result["text"]) > 0
        assert "example" in result["text"].lower()

    @skip_if_down("fetch")
    @pytest.mark.asyncio
    async def test_fetch_truncation(self):
        from aichat.tools.fetch import FetchTool
        tool = FetchTool()
        result = await tool.fetch_url("https://example.com", max_chars=50)
        assert result.get("truncated") is True or len(result.get("text", "")) <= 100


class TestMemoryToolLive:
    @skip_if_down("memory")
    @pytest.mark.asyncio
    async def test_store_and_recall(self):
        from aichat.tools.memory import MemoryTool
        tool = MemoryTool()
        key = "test_e2e_key"
        await tool.store(key, "hello from e2e test")
        result = await tool.recall(key)
        assert "hello from e2e test" in json.dumps(result)

    @skip_if_down("memory")
    @pytest.mark.asyncio
    async def test_recall_all_keys(self):
        from aichat.tools.memory import MemoryTool
        tool = MemoryTool()
        result = await tool.recall("")
        assert isinstance(result, dict)


class TestRSSToolLive:
    @skip_if_down("rssfeed")
    @pytest.mark.asyncio
    async def test_news_latest_returns_dict(self):
        from aichat.tools.rss import RSSTool
        tool = RSSTool()
        result = await tool.news_latest("technology")
        assert isinstance(result, dict)


class TestResearchboxToolLive:
    @skip_if_down("researchbox")
    @pytest.mark.asyncio
    async def test_search_feeds_returns_dict(self):
        from aichat.tools.researchbox import ResearchboxTool
        tool = ResearchboxTool()
        result = await tool.rb_search_feeds("technology")
        assert isinstance(result, dict)


class TestToolkitLive:
    @skip_if_down("toolkit")
    @pytest.mark.asyncio
    async def test_list_tools(self):
        from aichat.tools.toolkit import ToolkitTool
        tool = ToolkitTool()
        tools = await tool.list_tools()
        assert isinstance(tools, list)

    @skip_if_down("toolkit")
    @pytest.mark.asyncio
    async def test_register_and_call_tool(self):
        from aichat.tools.toolkit import ToolkitTool
        tool = ToolkitTool()
        await tool.register_tool(
            "e2e_test_tool",
            "Returns a fixed greeting for e2e tests.",
            {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
            'return f"Hello {kwargs[\'name\']} from e2e!"',
        )
        result = await tool.call_tool("e2e_test_tool", {"name": "world"})
        assert "Hello world" in result.get("result", "")
        # Cleanup
        await tool.delete_tool("e2e_test_tool")


# ---------------------------------------------------------------------------
# ToolManager unit tests (mocked backends)
# ---------------------------------------------------------------------------

class TestToolManagerDispatch:
    """Verify ToolManager routes every tool name to the right backend."""

    @pytest.mark.asyncio
    async def test_run_web_fetch_mocked(self):
        mgr = ToolManager()
        with patch.object(mgr.fetch, "fetch_url", new=AsyncMock(
            return_value={"text": "Example Domain", "truncated": False, "char_count": 14}
        )):
            result = await mgr.run_web_fetch(
                "https://example.com", 4000, ApprovalMode.AUTO, None
            )
            assert "Example Domain" in json.dumps(result)

    @pytest.mark.asyncio
    async def test_run_memory_store_mocked(self):
        mgr = ToolManager()
        with patch.object(mgr.memory, "store", new=AsyncMock(
            return_value={"status": "ok"}
        )):
            result = await mgr.run_memory_store("k", "v", ApprovalMode.AUTO, None)
            assert result == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_run_memory_recall_mocked(self):
        mgr = ToolManager()
        with patch.object(mgr.memory, "recall", new=AsyncMock(
            return_value={"value": "stored value"}
        )):
            result = await mgr.run_memory_recall("k", ApprovalMode.AUTO, None)
            assert result["value"] == "stored value"

    @pytest.mark.asyncio
    async def test_run_rss_mocked(self):
        mgr = ToolManager()
        with patch.object(mgr.rss, "news_latest", new=AsyncMock(
            return_value={"items": [{"title": "AI News"}]}
        )):
            result = await mgr.run_rss("AI", ApprovalMode.AUTO, None)
            assert result["items"][0]["title"] == "AI News"

    @pytest.mark.asyncio
    async def test_run_researchbox_mocked(self):
        mgr = ToolManager()
        with patch.object(mgr.researchbox, "rb_search_feeds", new=AsyncMock(
            return_value={"feeds": ["https://feeds.example.com/rss"]}
        )):
            result = await mgr.run_researchbox("AI", ApprovalMode.AUTO, None)
            assert "feeds" in result

    @pytest.mark.asyncio
    async def test_tool_denied_mode(self):
        mgr = ToolManager()
        with pytest.raises(ToolDeniedError):
            await mgr.run_rss("topic", ApprovalMode.DENY, None)

    @pytest.mark.asyncio
    async def test_dangerous_shell_blocked_regardless_of_mode(self):
        mgr = ToolManager()
        with pytest.raises(ToolDeniedError, match="destructive"):
            await mgr.run_shell(
                "rm -rf /etc/passwd", ApprovalMode.AUTO, None
            )

    @pytest.mark.asyncio
    async def test_create_tool_mocked(self):
        mgr = ToolManager()
        with patch.object(mgr.toolkit, "register_tool", new=AsyncMock(
            return_value={"status": "created"}
        )), patch.object(mgr, "refresh_custom_tools", new=AsyncMock()):
            result = await mgr.run_create_tool(
                "my_tool", "desc", {}, "return 'hi'", ApprovalMode.AUTO, None
            )
            assert result["status"] == "created"

    @pytest.mark.asyncio
    async def test_list_custom_tools_mocked(self):
        mgr = ToolManager()
        with patch.object(mgr.toolkit, "list_tools", new=AsyncMock(
            return_value=[{"name": "my_tool", "description": "desc", "parameters": {}}]
        )), patch.object(mgr, "refresh_custom_tools", new=AsyncMock()):
            result = await mgr.run_list_custom_tools(ApprovalMode.AUTO, None)
            assert result[0]["name"] == "my_tool"

    @pytest.mark.asyncio
    async def test_delete_custom_tool_mocked(self):
        mgr = ToolManager()
        with patch.object(mgr.toolkit, "delete_tool", new=AsyncMock(
            return_value={"status": "deleted"}
        )), patch.object(mgr, "refresh_custom_tools", new=AsyncMock()):
            result = await mgr.run_delete_custom_tool("my_tool", ApprovalMode.AUTO, None)
            assert result["status"] == "deleted"


# ---------------------------------------------------------------------------
# TUI integration: every tool dispatch path exercised via mocked LLM
# ---------------------------------------------------------------------------

def _make_app() -> AIChatApp:
    app = AIChatApp()
    app.client.health = AsyncMock(return_value=True)
    app.client.list_models = AsyncMock(return_value=["test-model"])
    app.client.ensure_model = AsyncMock()
    app.client.chat_once_with_tools = AsyncMock(
        return_value={"content": "Done.", "tool_calls": []}
    )
    async def _fake_stream(*a, **kw):
        yield {"type": "content", "value": "Streamed."}
    app.client.chat_stream_events = _fake_stream
    app.state.streaming = False
    return app


async def _type_and_send(pilot, text: str) -> None:
    await pilot.click("#prompt")
    await pilot.pause(0.1)
    key_map = {"/": "slash", " ": "space", ".": "period", ":": "colon",
               "_": "underscore", "-": "minus"}
    for ch in text:
        await pilot.press(key_map.get(ch, ch))
    await pilot.pause(0.1)
    await pilot.press("enter")


async def _wait_bubble(app, pilot, minimum=1, iters=40):
    for _ in range(iters):
        await pilot.pause(0.3)
        if len(app.query(".chat-msg")) >= minimum:
            break
    return list(app.query(".chat-msg"))


class TestTUIToolDispatch:
    """Each tool slash-command produces a result bubble without crashing."""

    @pytest.mark.asyncio
    async def test_slash_fetch_shows_bubble(self):
        app = _make_app()
        with patch("aichat.tools.fetch.FetchTool.fetch_url", new=AsyncMock(
            return_value={"text": "Example page content", "truncated": False, "char_count": 20}
        )):
            async with app.run_test(size=(180, 50)) as pilot:
                await pilot.pause(1.0)
                await _type_and_send(pilot, "/fetch https://example.com")
                bubbles = await _wait_bubble(app, pilot)
                assert any("chat-assistant" in b.classes for b in bubbles)

    @pytest.mark.asyncio
    async def test_slash_memory_store_shows_bubble(self):
        app = _make_app()
        with patch("aichat.tools.memory.MemoryTool.store", new=AsyncMock(
            return_value={"status": "ok"}
        )):
            async with app.run_test(size=(180, 50)) as pilot:
                await pilot.pause(1.0)
                await _type_and_send(pilot, "/memory store mykey myvalue")
                bubbles = await _wait_bubble(app, pilot)
                assert any("chat-assistant" in b.classes for b in bubbles)

    @pytest.mark.asyncio
    async def test_slash_memory_recall_shows_bubble(self):
        app = _make_app()
        with patch("aichat.tools.memory.MemoryTool.recall", new=AsyncMock(
            return_value={"value": "myvalue"}
        )):
            async with app.run_test(size=(180, 50)) as pilot:
                await pilot.pause(1.0)
                await _type_and_send(pilot, "/memory recall mykey")
                bubbles = await _wait_bubble(app, pilot)
                assert any("chat-assistant" in b.classes for b in bubbles)

    @pytest.mark.asyncio
    async def test_slash_rss_shows_bubble(self):
        app = _make_app()
        with patch("aichat.tools.rss.RSSTool.news_latest", new=AsyncMock(
            return_value={"items": [{"title": "Tech Story"}]}
        )):
            async with app.run_test(size=(180, 50)) as pilot:
                await pilot.pause(1.0)
                await _type_and_send(pilot, "/rss technology")
                bubbles = await _wait_bubble(app, pilot)
                assert any("chat-assistant" in b.classes for b in bubbles)

    @pytest.mark.asyncio
    async def test_slash_researchbox_shows_bubble(self):
        app = _make_app()
        with patch("aichat.tools.researchbox.ResearchboxTool.rb_search_feeds", new=AsyncMock(
            return_value={"feeds": ["https://feeds.example.com/rss"]}
        )):
            async with app.run_test(size=(180, 50)) as pilot:
                await pilot.pause(1.0)
                await _type_and_send(pilot, "/researchbox technology")
                bubbles = await _wait_bubble(app, pilot)
                assert any("chat-assistant" in b.classes for b in bubbles)

    @pytest.mark.asyncio
    async def test_slash_tool_list_shows_bubble(self):
        app = _make_app()
        with patch("aichat.tools.toolkit.ToolkitTool.list_tools", new=AsyncMock(
            return_value=[]
        )), patch.object(app.tools, "refresh_custom_tools", new=AsyncMock()):
            async with app.run_test(size=(180, 50)) as pilot:
                await pilot.pause(1.0)
                await _type_and_send(pilot, "/tool list")
                bubbles = await _wait_bubble(app, pilot)
                assert any("chat-assistant" in b.classes for b in bubbles)

    @pytest.mark.asyncio
    async def test_llm_tool_call_web_fetch_roundtrip(self):
        """LLM returns a web_fetch tool call → tool runs → followup response → bubble."""
        app = _make_app()
        call_count = 0

        async def _chat_with_tools(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "content": "",
                    "tool_calls": [{
                        "id": "c1",
                        "function": {
                            "name": "web_fetch",
                            "arguments": '{"url":"https://example.com","max_chars":500}',
                        },
                    }],
                }
            return {"content": "The page title is: Example Domain.", "tool_calls": []}

        app.client.chat_once_with_tools = _chat_with_tools
        with patch("aichat.tools.fetch.FetchTool.fetch_url", new=AsyncMock(
            return_value={"text": "Example Domain page", "truncated": False, "char_count": 19}
        )):
            async with app.run_test(size=(180, 50)) as pilot:
                await pilot.pause(1.0)
                await _type_and_send(pilot, "what is on example.com")
                for _ in range(50):
                    await pilot.pause(0.4)
                    if not app.state.busy:
                        break
                bubbles = list(app.query(".chat-msg"))
                assert any("chat-assistant" in b.classes for b in bubbles)

    @pytest.mark.asyncio
    async def test_llm_tool_call_memory_roundtrip(self):
        """LLM returns a memory_store tool call → stored → followup → bubble."""
        app = _make_app()
        call_count = 0

        async def _chat_with_tools(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "content": "",
                    "tool_calls": [{
                        "id": "c1",
                        "function": {
                            "name": "memory_store",
                            "arguments": '{"key":"user_name","value":"Alice"}',
                        },
                    }],
                }
            return {"content": "I've remembered your name.", "tool_calls": []}

        app.client.chat_once_with_tools = _chat_with_tools
        with patch("aichat.tools.memory.MemoryTool.store", new=AsyncMock(
            return_value={"status": "ok"}
        )):
            async with app.run_test(size=(180, 50)) as pilot:
                await pilot.pause(1.0)
                await _type_and_send(pilot, "remember my name is Alice")
                for _ in range(50):
                    await pilot.pause(0.4)
                    if not app.state.busy:
                        break
                bubbles = list(app.query(".chat-msg"))
                assert any("chat-assistant" in b.classes for b in bubbles)

    @pytest.mark.asyncio
    async def test_xml_arg_format_tool_call_handled(self):
        """Tool call with XML-format arguments is parsed and executed correctly."""
        app = _make_app()
        _call_count = 0

        async def _chat_with_tools_xml(*a, **kw):
            nonlocal _call_count
            _call_count += 1
            if _call_count == 1:
                return {
                    "content": "",
                    "tool_calls": [{
                        "id": "c1",
                        "function": {
                            "name": "rss_latest",
                            "arguments": (
                                "<arg_key>topic</arg_key>"
                                "<arg_value>Iran%20News</arg_value>"
                            ),
                        },
                    }],
                }
            return {"content": "Iran news summary.", "tool_calls": []}

        app.client.chat_once_with_tools = _chat_with_tools_xml

        with patch("aichat.tools.rss.RSSTool.news_latest", new=AsyncMock(
            return_value={"items": [{"title": "Iran story"}]}
        )):
            async with app.run_test(size=(180, 50)) as pilot:
                await pilot.pause(1.0)
                await _type_and_send(pilot, "Iran news")
                for _ in range(50):
                    await pilot.pause(0.4)
                    if not app.state.busy:
                        break
                bubbles = list(app.query(".chat-msg"))
                assert any("chat-assistant" in b.classes for b in bubbles)

    @pytest.mark.asyncio
    async def test_tool_error_does_not_crash_tui(self):
        """A failing tool surfaces an error bubble without crashing the app."""
        app = _make_app()
        app.client.chat_once_with_tools = AsyncMock(return_value={
            "content": "",
            "tool_calls": [{
                "id": "c1",
                "function": {
                    "name": "web_fetch",
                    "arguments": '{"url":"https://example.com"}',
                },
            }],
        })
        with patch("aichat.tools.fetch.FetchTool.fetch_url", side_effect=Exception("service down")):
            async with app.run_test(size=(180, 50)) as pilot:
                await pilot.pause(1.0)
                await _type_and_send(pilot, "fetch something")
                for _ in range(40):
                    await pilot.pause(0.4)
                    if not app.state.busy:
                        break
                bubbles = list(app.query(".chat-msg"))
                assert len(bubbles) >= 1

    @pytest.mark.asyncio
    async def test_transcript_no_xml_tags_in_bubbles(self):
        """Even if the LLM returns XML arg content, bubbles must not contain raw tags."""
        app = _make_app()
        # LLM returns XML-style junk in the content field
        app.client.chat_once_with_tools = AsyncMock(return_value={
            "content": (
                "rss_latest topic</arg_key> Iran%20News</arg_value>"
            ),
            "tool_calls": [],
        })
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause(1.0)
            await _type_and_send(pilot, "Iran news")
            for _ in range(30):
                await pilot.pause(0.4)
                if not app.state.busy:
                    break
            bubbles = list(app.query(".chat-msg"))
            # Find the assistant bubble body text
            for b in bubbles:
                if "chat-assistant" in b.classes:
                    body = b.query_one(".msg-body")
                    body_src = getattr(body, "_markdown", None) or ""
                    assert "</arg_key>" not in body_src
                    assert "</arg_value>" not in body_src


# ---------------------------------------------------------------------------
# Browser tool — unit tests (always run, mocked)
# ---------------------------------------------------------------------------

class TestBrowserToolUnit:
    """BrowserTool dispatch through ToolManager with all I/O mocked."""

    @pytest.mark.asyncio
    async def test_run_browser_navigate_mocked(self):
        mgr = ToolManager()
        mock_result = {"title": "Example Domain", "url": "https://example.com", "content": "Hello"}
        with patch.object(mgr.browser, "navigate", new=AsyncMock(return_value=mock_result)):
            result = await mgr.run_browser(
                "navigate", ApprovalMode.AUTO, None, url="https://example.com"
            )
        assert result["title"] == "Example Domain"

    @pytest.mark.asyncio
    async def test_run_browser_read_mocked(self):
        mgr = ToolManager()
        mock_result = {"title": "My Page", "url": "https://example.com", "content": "page text"}
        with patch.object(mgr.browser, "read", new=AsyncMock(return_value=mock_result)):
            result = await mgr.run_browser("read", ApprovalMode.AUTO, None)
        assert result["content"] == "page text"

    @pytest.mark.asyncio
    async def test_run_browser_click_mocked(self):
        mgr = ToolManager()
        mock_result = {"url": "https://example.com/page2", "content": "clicked page"}
        with patch.object(mgr.browser, "click", new=AsyncMock(return_value=mock_result)):
            result = await mgr.run_browser("click", ApprovalMode.AUTO, None, selector="#btn")
        assert "url" in result

    @pytest.mark.asyncio
    async def test_run_browser_fill_mocked(self):
        mgr = ToolManager()
        with patch.object(mgr.browser, "fill", new=AsyncMock(return_value={"ok": True})):
            result = await mgr.run_browser(
                "fill", ApprovalMode.AUTO, None, selector="#input", value="hello"
            )
        assert result["ok"] is True

    @pytest.mark.asyncio
    async def test_run_browser_eval_mocked(self):
        mgr = ToolManager()
        with patch.object(mgr.browser, "eval_js", new=AsyncMock(return_value={"result": "42"})):
            result = await mgr.run_browser(
                "eval", ApprovalMode.AUTO, None, code="1 + 41"
            )
        assert result["result"] == "42"

    @pytest.mark.asyncio
    async def test_run_browser_screenshot_mocked(self):
        mgr = ToolManager()
        mock_result = {
            "path": "/workspace/screenshot_test.png",
            "title": "Test",
            "url": "https://example.com",
            "host_path": "/docker/human_browser/workspace/screenshot_test.png",
        }
        with patch.object(mgr.browser, "screenshot", new=AsyncMock(return_value=mock_result)):
            result = await mgr.run_browser(
                "screenshot", ApprovalMode.AUTO, None, url="https://example.com"
            )
        assert "path" in result

    @pytest.mark.asyncio
    async def test_run_browser_missing_url_returns_error(self):
        mgr = ToolManager()
        result = await mgr.run_browser("navigate", ApprovalMode.AUTO, None)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_run_browser_unknown_action_returns_error(self):
        mgr = ToolManager()
        result = await mgr.run_browser("teleport", ApprovalMode.AUTO, None)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_run_browser_denied_raises(self):
        mgr = ToolManager()
        with pytest.raises(ToolDeniedError):
            await mgr.run_browser("navigate", ApprovalMode.DENY, None, url="https://x.com")

    def test_browser_tool_definition_present(self):
        """browser tool must appear in the tool definitions list."""
        mgr = ToolManager()
        defs = mgr.tool_definitions(shell_enabled=True)
        names = [d["function"]["name"] for d in defs]
        assert "browser" in names, f"'browser' not found in tool definitions: {names}"

    def test_browser_tool_has_required_fields(self):
        mgr = ToolManager()
        defs = mgr.tool_definitions(shell_enabled=False)
        browser_def = next(d for d in defs if d["function"]["name"] == "browser")
        params = browser_def["function"]["parameters"]["properties"]
        assert "action" in params
        assert "url" in params
        assert "selector" in params
        assert "value" in params
        assert "code" in params


# ---------------------------------------------------------------------------
# Browser tool — live e2e smoke test (skipped if container not running)
# ---------------------------------------------------------------------------

def _human_browser_running() -> bool:
    try:
        import subprocess, json as _json
        r = subprocess.run(
            ["docker", "inspect", "human_browser"],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return False
        data = _json.loads(r.stdout)[0]
        return bool(data.get("State", {}).get("Running"))
    except Exception:
        return False


_BROWSER_UP = _human_browser_running()
skip_no_browser = pytest.mark.skipif(
    not _BROWSER_UP,
    reason="human_browser container not running",
)


class TestBrowserLive:
    """Live end-to-end tests against the real human_browser container."""

    @skip_no_browser
    @pytest.mark.asyncio
    async def test_navigate_example_com(self):
        from aichat.tools.browser import BrowserTool
        tool = BrowserTool()
        result = await tool.navigate("https://example.com")
        assert "title" in result
        assert "content" in result
        assert "example" in result["title"].lower() or "example" in result["content"].lower()

    @skip_no_browser
    @pytest.mark.asyncio
    async def test_read_after_navigate(self):
        from aichat.tools.browser import BrowserTool
        tool = BrowserTool()
        await tool.navigate("https://example.com")
        result = await tool.read()
        assert "url" in result
        assert "content" in result

    @skip_no_browser
    @pytest.mark.asyncio
    async def test_eval_js(self):
        from aichat.tools.browser import BrowserTool
        tool = BrowserTool()
        await tool.navigate("https://example.com")
        result = await tool.eval_js("document.title")
        assert "result" in result
        assert result["result"]  # non-empty

    @skip_no_browser
    @pytest.mark.asyncio
    async def test_screenshot_saves_file(self):
        import os
        from aichat.tools.browser import BrowserTool
        tool = BrowserTool()
        result = await tool.screenshot("https://example.com")
        assert "path" in result
        # Check the file was written into the workspace
        host_path = result.get("host_path", "")
        if host_path and os.path.exists(host_path):
            assert os.path.getsize(host_path) > 0

    @skip_no_browser
    @pytest.mark.asyncio
    async def test_manager_run_browser_live(self):
        """Full stack: ToolManager → BrowserTool → real container."""
        mgr = ToolManager()
        result = await mgr.run_browser(
            "navigate", ApprovalMode.AUTO, None, url="https://example.com"
        )
        assert "error" not in result, f"Unexpected error: {result.get('error')}"
        assert result.get("title") or result.get("content")


# ---------------------------------------------------------------------------
# Browser tool — TUI integration (mocked browser, real TUI)
# ---------------------------------------------------------------------------

class TestBrowserTUIDispatch:
    """LLM returns browser tool call → dispatched → result bubble appears."""

    @pytest.mark.asyncio
    async def test_browser_navigate_tui_roundtrip(self):
        app = _make_app()
        call_count = 0

        async def _chat_with_browser(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "content": "",
                    "tool_calls": [{
                        "id": "b1",
                        "function": {
                            "name": "browser",
                            "arguments": '{"action":"navigate","url":"https://example.com"}',
                        },
                    }],
                }
            return {"content": "The page title is Example Domain.", "tool_calls": []}

        app.client.chat_once_with_tools = _chat_with_browser

        with patch.object(
            app.tools.browser, "navigate",
            new=AsyncMock(return_value={
                "title": "Example Domain",
                "url": "https://example.com",
                "content": "This domain is for illustrative examples.",
            }),
        ):
            async with app.run_test(size=(180, 50)) as pilot:
                await pilot.pause(1.0)
                await _type_and_send(pilot, "go to example.com")
                for _ in range(50):
                    await pilot.pause(0.4)
                    if not app.state.busy:
                        break
                bubbles = list(app.query(".chat-msg"))
                assert any("chat-assistant" in b.classes for b in bubbles)

    @pytest.mark.asyncio
    async def test_browser_error_does_not_crash_tui(self):
        """A browser tool failure surfaces gracefully without crashing the app."""
        app = _make_app()
        app.client.chat_once_with_tools = AsyncMock(return_value={
            "content": "",
            "tool_calls": [{
                "id": "b2",
                "function": {
                    "name": "browser",
                    "arguments": '{"action":"navigate","url":"https://example.com"}',
                },
            }],
        })
        with patch.object(
            app.tools.browser, "navigate",
            new=AsyncMock(side_effect=RuntimeError("Container not running")),
        ):
            async with app.run_test(size=(180, 50)) as pilot:
                await pilot.pause(1.0)
                await _type_and_send(pilot, "browse")
                for _ in range(40):
                    await pilot.pause(0.4)
                    if not app.state.busy:
                        break
                bubbles = list(app.query(".chat-msg"))
                assert len(bubbles) >= 1
