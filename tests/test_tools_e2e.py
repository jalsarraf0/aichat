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
    "memory":      "http://localhost:8094/health",
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


def _is_database_up() -> bool:
    """Check that the *new* aichat-database service (not old rssfeed) is running."""
    try:
        r = httpx.get("http://localhost:8091/health", timeout=2)
        if r.status_code != 200:
            return False
        # New service returns {"ok": true, "articles": N, "cached_pages": N}
        # Old rssfeed returned {"ok": true, "last_purge_at": ...}
        data = r.json()
        return "articles" in data and "cached_pages" in data
    except Exception:
        return False


SERVICE_STATUS: dict[str, bool] = {
    name: _is_up(url) for name, url in SERVICES.items()
}
SERVICE_STATUS["database"] = _is_database_up()


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
        args, err = parse_tool_args("db_search", '{"topic": "Iran"}')
        assert err is None
        assert args["topic"] == "Iran"

    def test_xml_format_single(self):
        args, err = parse_tool_args(
            "db_search",
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
        args, err = parse_tool_args("db_search", "topic: AI news")
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
    @skip_if_down("database")
    def test_database_health(self):
        r = httpx.get("http://localhost:8091/health", timeout=3)
        assert r.status_code == 200
        data = r.json()
        assert data.get("ok") is True

    @skip_if_down("memory")
    def test_memory_health(self):
        r = httpx.get("http://localhost:8094/health", timeout=3)
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

class TestDatabaseToolLive:
    @skip_if_down("database")
    @pytest.mark.asyncio
    async def test_store_and_search_article(self):
        from aichat.tools.database import DatabaseTool
        tool = DatabaseTool()
        result = await tool.store_article(
            url="https://example.com/e2e-test",
            title="E2E Test Article",
            content="This is an e2e test article about technology.",
            topic="e2e_test",
        )
        assert result.get("status") == "stored"
        search = await tool.search_articles(topic="e2e_test")
        urls = [a["url"] for a in search.get("articles", [])]
        assert "https://example.com/e2e-test" in urls

    @skip_if_down("database")
    @pytest.mark.asyncio
    async def test_cache_store_and_get(self):
        from aichat.tools.database import DatabaseTool
        tool = DatabaseTool()
        await tool.cache_store(
            url="https://example.com/cache-test",
            content="Cached page content",
            title="Cache Test",
        )
        result = await tool.cache_get("https://example.com/cache-test")
        assert result.get("found") is True
        assert "Cached page content" in result.get("content", "")

    @skip_if_down("database")
    @pytest.mark.asyncio
    async def test_cache_check_missing(self):
        from aichat.tools.database import DatabaseTool
        tool = DatabaseTool()
        result = await tool.cache_check("https://this-url-does-not-exist-in-cache.example.com")
        assert result.get("cached") is False


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
        """web_fetch is now routed through the browser (human_browser / a12fdfeaaf78)."""
        mgr = ToolManager()
        with patch.object(mgr.browser, "navigate", new=AsyncMock(
            return_value={"title": "Example Domain", "url": "https://example.com", "content": "Example Domain"}
        )):
            result = await mgr.run_web_fetch(
                "https://example.com", 4000, ApprovalMode.AUTO, None
            )
            assert "Example Domain" in result.get("text", "")

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
    async def test_run_db_store_article_mocked(self):
        mgr = ToolManager()
        with patch.object(mgr.db, "store_article", new=AsyncMock(
            return_value={"status": "stored", "url": "https://example.com"}
        )):
            result = await mgr.run_db_store_article(
                "https://example.com", "Title", "Content", "tech",
                ApprovalMode.AUTO, None,
            )
            assert result["status"] == "stored"

    @pytest.mark.asyncio
    async def test_run_db_search_mocked(self):
        mgr = ToolManager()
        with patch.object(mgr.db, "search_articles", new=AsyncMock(
            return_value={"articles": [{"url": "https://example.com", "title": "Title"}]}
        )):
            result = await mgr.run_db_search("tech", "", ApprovalMode.AUTO, None)
            assert "articles" in result

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
            await mgr.run_researchbox("topic", ApprovalMode.DENY, None)

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

    @pytest.mark.asyncio
    async def test_run_web_search_tier1_mocked(self):
        """web_search tier 1: browser returns content → used directly."""
        mgr = ToolManager()
        with patch.object(mgr.search_tool, "search", new=AsyncMock(
            return_value={
                "query": "python asyncio",
                "tier": 1,
                "tier_name": "browser (human-like)",
                "url": "https://duckduckgo.com/?q=python+asyncio",
                "content": "Python asyncio documentation...",
            }
        )):
            result = await mgr.run_web_search("python asyncio", 4000, ApprovalMode.AUTO, None)
            assert result["tier"] == 1
            assert "asyncio" in result["content"]

    @pytest.mark.asyncio
    async def test_run_web_search_fallback_to_tier2(self):
        """web_search: browser fails → tier 2 httpx result returned."""
        mgr = ToolManager()
        with patch.object(mgr.search_tool, "search", new=AsyncMock(
            return_value={
                "query": "python asyncio",
                "tier": 2,
                "tier_name": "httpx (programmatic)",
                "url": "https://html.duckduckgo.com/html/?q=python+asyncio",
                "content": "asyncio results...",
            }
        )):
            result = await mgr.run_web_search("python asyncio", 4000, ApprovalMode.AUTO, None)
            assert result["tier"] == 2

    @pytest.mark.asyncio
    async def test_run_web_search_denied_raises(self):
        mgr = ToolManager()
        with pytest.raises(ToolDeniedError):
            await mgr.run_web_search("query", 4000, ApprovalMode.DENY, None)

    def test_web_search_tool_definition_present(self):
        mgr = ToolManager()
        defs = mgr.tool_definitions(shell_enabled=False)
        names = [d["function"]["name"] for d in defs]
        assert "web_search" in names, f"'web_search' not in tool definitions: {names}"


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
    async def test_slash_search_shows_bubble(self):
        app = _make_app()
        with patch.object(app.tools.search_tool, "search", new=AsyncMock(
            return_value={
                "query": "python asyncio",
                "tier": 1, "tier_name": "browser (human-like)",
                "url": "https://duckduckgo.com/", "content": "Python asyncio docs",
            }
        )):
            async with app.run_test(size=(180, 50)) as pilot:
                await pilot.pause(1.0)
                await _type_and_send(pilot, "/search python asyncio")
                bubbles = await _wait_bubble(app, pilot)
                assert any("chat-assistant" in b.classes for b in bubbles)

    @pytest.mark.asyncio
    async def test_slash_fetch_shows_bubble(self):
        # /fetch now routes through the browser (human_browser / a12fdfeaaf78)
        app = _make_app()
        with patch.object(app.tools.browser, "navigate", new=AsyncMock(
            return_value={"title": "Example Domain", "url": "https://example.com", "content": "Example page content"}
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
        with patch.object(app.tools.db, "search_articles", new=AsyncMock(
            return_value={"articles": [{"url": "https://t.com", "title": "Tech Story"}]}
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
    async def test_llm_tool_call_web_search_roundtrip(self):
        """LLM returns a web_search tool call → search runs → followup response → bubble."""
        app = _make_app()
        call_count = 0

        async def _chat_with_search(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "content": "",
                    "tool_calls": [{
                        "id": "s1",
                        "function": {
                            "name": "web_search",
                            "arguments": '{"query":"python asyncio tutorial","max_chars":500}',
                        },
                    }],
                }
            return {"content": "Here are some asyncio tutorials.", "tool_calls": []}

        app.client.chat_once_with_tools = _chat_with_search
        with patch.object(app.tools.search_tool, "search", new=AsyncMock(
            return_value={
                "query": "python asyncio tutorial",
                "tier": 1, "tier_name": "browser (human-like)",
                "url": "https://duckduckgo.com/", "content": "Python asyncio tutorial results",
            }
        )):
            async with app.run_test(size=(180, 50)) as pilot:
                await pilot.pause(1.0)
                await _type_and_send(pilot, "find asyncio tutorials")
                for _ in range(50):
                    await pilot.pause(0.4)
                    if not app.state.busy:
                        break
                bubbles = list(app.query(".chat-msg"))
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
        with patch.object(app.tools.browser, "navigate", new=AsyncMock(
            return_value={"title": "Example Domain", "url": "https://example.com", "content": "Example Domain page"}
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
                            "name": "db_search",
                            "arguments": (
                                "<arg_key>topic</arg_key>"
                                "<arg_value>Iran%20News</arg_value>"
                            ),
                        },
                    }],
                }
            return {"content": "Iran news summary.", "tool_calls": []}

        app.client.chat_once_with_tools = _chat_with_tools_xml

        with patch.object(app.tools.db, "search_articles", new=AsyncMock(
            return_value={"articles": [{"title": "Iran story", "url": "https://example.com"}]}
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
        with patch.object(app.tools.browser, "navigate", side_effect=Exception("browser down")):
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
                "db_search topic</arg_key> Iran%20News</arg_value>"
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
    async def test_screenshot_auto_saves_to_db(self):
        """Screenshot action on ToolManager auto-saves to DB when host_path is set."""
        mgr = ToolManager()
        mock_result = {
            "path": "/workspace/screenshot_test.png",
            "title": "Example Domain",
            "url": "https://example.com",
            "host_path": "/docker/human_browser/workspace/screenshot_test.png",
        }
        db_store_calls = []

        async def _fake_store_image(**kwargs):
            db_store_calls.append(kwargs)
            return {"ok": True, "id": 1}

        with patch.object(mgr.browser, "screenshot", new=AsyncMock(return_value=mock_result)):
            with patch.object(mgr.db, "store_image", side_effect=_fake_store_image):
                result = await mgr.run_browser(
                    "screenshot", ApprovalMode.AUTO, None, url="https://example.com"
                )
        assert "path" in result
        assert len(db_store_calls) == 1
        assert db_store_calls[0]["host_path"] == "/docker/human_browser/workspace/screenshot_test.png"

    @pytest.mark.asyncio
    async def test_screenshot_db_failure_does_not_raise(self):
        """DB save failure after screenshot must not propagate — screenshot still returned."""
        mgr = ToolManager()
        mock_result = {
            "path": "/workspace/screenshot_test.png",
            "title": "Test",
            "url": "https://example.com",
            "host_path": "/docker/human_browser/workspace/screenshot_test.png",
        }
        with patch.object(mgr.browser, "screenshot", new=AsyncMock(return_value=mock_result)):
            with patch.object(mgr.db, "store_image", side_effect=RuntimeError("DB down")):
                result = await mgr.run_browser(
                    "screenshot", ApprovalMode.AUTO, None, url="https://example.com"
                )
        assert "path" in result  # screenshot still returned despite DB error

    @pytest.mark.asyncio
    async def test_run_db_store_image_mocked(self):
        """run_db_store_image delegates to DatabaseTool.store_image."""
        mgr = ToolManager()
        with patch.object(mgr.db, "store_image", new=AsyncMock(return_value={"ok": True, "id": 7})):
            result = await mgr.run_db_store_image(
                url="https://example.com",
                host_path="/docker/human_browser/workspace/test.png",
                alt_text="Test screenshot",
                mode=ApprovalMode.AUTO,
                confirmer=None,
            )
        assert result.get("ok") is True

    @pytest.mark.asyncio
    async def test_run_db_list_images_mocked(self):
        """run_db_list_images returns images list from DatabaseTool."""
        mgr = ToolManager()
        fake_images = [
            {"id": 1, "url": "https://example.com", "host_path": "/docker/x/test.png",
             "alt_text": "Test", "stored_at": "2026-02-24T12:00:00"},
        ]
        with patch.object(mgr.db, "list_images", new=AsyncMock(return_value={"images": fake_images})):
            result = await mgr.run_db_list_images(limit=10, mode=ApprovalMode.AUTO, confirmer=None)
        assert "images" in result
        assert len(result["images"]) == 1

    @pytest.mark.asyncio
    async def test_run_db_store_image_denied(self):
        mgr = ToolManager()
        with pytest.raises(ToolDeniedError):
            await mgr.run_db_store_image(
                url="https://example.com", host_path="", alt_text="",
                mode=ApprovalMode.DENY, confirmer=None,
            )

    @pytest.mark.asyncio
    async def test_run_db_list_images_denied(self):
        mgr = ToolManager()
        with pytest.raises(ToolDeniedError):
            await mgr.run_db_list_images(limit=5, mode=ApprovalMode.DENY, confirmer=None)

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
        assert "find_text" in params, "browser tool schema missing 'find_text' parameter"

    @pytest.mark.asyncio
    async def test_run_browser_screenshot_passes_find_text(self):
        """run_browser(screenshot, find_text=...) forwards find_text to BrowserTool."""
        mgr = ToolManager()
        captured_kwargs: list[dict] = []

        async def _fake_screenshot(url=None, find_text=None):
            captured_kwargs.append({"url": url, "find_text": find_text})
            return {
                "path": "/workspace/screenshot_test.png",
                "title": "Test",
                "url": url or "https://example.com",
                "host_path": "/docker/human_browser/workspace/screenshot_test.png",
                "clipped": bool(find_text),
            }

        with patch.object(mgr.browser, "screenshot", side_effect=_fake_screenshot):
            with patch.object(mgr.db, "store_image", new=AsyncMock(return_value={"ok": True})):
                result = await mgr.run_browser(
                    "screenshot", ApprovalMode.AUTO, None,
                    url="https://example.com", find_text="Introduction",
                )
        assert "path" in result
        assert len(captured_kwargs) == 1
        assert captured_kwargs[0]["find_text"] == "Introduction"


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


# ---------------------------------------------------------------------------
# WebSearchTool — unit tests (mocked)
# ---------------------------------------------------------------------------

class TestWebSearchToolUnit:
    """WebSearchTool tiered fallback logic, all I/O mocked."""

    @pytest.mark.asyncio
    async def test_tier1_success(self):
        from aichat.tools.browser import BrowserTool
        from aichat.tools.search import WebSearchTool
        browser = BrowserTool()
        tool = WebSearchTool(browser)
        with patch.object(browser, "search", new=AsyncMock(
            return_value={"query": "test", "url": "https://ddg.com", "content": "result text"}
        )):
            result = await tool.search("test")
        assert result["tier"] == 1
        assert result["content"] == "result text"

    @pytest.mark.asyncio
    async def test_tier1_fails_falls_back_to_tier2(self):
        from aichat.tools.browser import BrowserTool
        from aichat.tools.search import WebSearchTool
        browser = BrowserTool()
        tool = WebSearchTool(browser)
        with patch.object(browser, "search", side_effect=RuntimeError("browser down")):
            with patch.object(tool, "_tier2_httpx", new=AsyncMock(
                return_value={"url": "https://ddg.com", "content": "tier2 results"}
            )):
                result = await tool.search("test")
        assert result["tier"] == 2

    @pytest.mark.asyncio
    async def test_tier1_and_2_fail_falls_back_to_tier3(self):
        from aichat.tools.browser import BrowserTool
        from aichat.tools.search import WebSearchTool
        browser = BrowserTool()
        tool = WebSearchTool(browser)
        with patch.object(browser, "search", side_effect=RuntimeError("browser down")):
            with patch.object(tool, "_tier2_httpx", side_effect=RuntimeError("httpx down")):
                with patch.object(tool, "_tier3_api", new=AsyncMock(
                    return_value={"url": "https://lite.ddg.com", "content": "tier3 results"}
                )):
                    result = await tool.search("test")
        assert result["tier"] == 3

    @pytest.mark.asyncio
    async def test_all_tiers_fail_returns_error(self):
        from aichat.tools.browser import BrowserTool
        from aichat.tools.search import WebSearchTool
        browser = BrowserTool()
        tool = WebSearchTool(browser)
        with patch.object(browser, "search", side_effect=RuntimeError("browser down")):
            with patch.object(tool, "_tier2_httpx", side_effect=RuntimeError("httpx down")):
                with patch.object(tool, "_tier3_api", side_effect=RuntimeError("api down")):
                    result = await tool.search("test")
        assert result["tier"] == 0
        assert "error" in result

    @pytest.mark.asyncio
    async def test_max_chars_truncation(self):
        from aichat.tools.browser import BrowserTool
        from aichat.tools.search import WebSearchTool
        browser = BrowserTool()
        tool = WebSearchTool(browser)
        long_content = "x" * 10000
        with patch.object(browser, "search", new=AsyncMock(
            return_value={"query": "test", "url": "https://ddg.com", "content": long_content}
        )):
            result = await tool.search("test", max_chars=500)
        assert len(result["content"]) == 500

    @pytest.mark.asyncio
    async def test_browser_search_error_field_triggers_fallback(self):
        """If browser returns {"error": ...} with no content, tier 1 fails gracefully."""
        from aichat.tools.browser import BrowserTool
        from aichat.tools.search import WebSearchTool
        browser = BrowserTool()
        tool = WebSearchTool(browser)
        with patch.object(browser, "search", new=AsyncMock(
            return_value={"error": "page load failed", "content": "", "query": "test"}
        )):
            with patch.object(tool, "_tier2_httpx", new=AsyncMock(
                return_value={"url": "https://ddg.com", "content": "tier2 fallback content"}
            )):
                result = await tool.search("test")
        assert result["tier"] == 2


# ---------------------------------------------------------------------------
# WebSearchTool — live e2e (skipped if browser not running)
# ---------------------------------------------------------------------------

class TestWebSearchLive:
    @skip_no_browser
    @pytest.mark.asyncio
    async def test_live_search_returns_content(self):
        """Tier 1 live: real browser navigates DuckDuckGo for 'python asyncio'."""
        from aichat.tools.browser import BrowserTool
        from aichat.tools.search import WebSearchTool
        browser = BrowserTool()
        tool = WebSearchTool(browser)
        result = await tool.search("python asyncio", max_chars=2000)
        assert result.get("tier", 0) >= 1, "Expected at least one tier to succeed"
        assert result.get("content"), "Expected non-empty search results"


# ---------------------------------------------------------------------------
# MCP server — screenshot and image rendering tests
# ---------------------------------------------------------------------------

class TestMCPScreenshot:
    """MCP server returns proper content-block lists for screenshot and image tools."""

    @pytest.mark.asyncio
    async def test_mcp_screenshot_returns_text_block(self):
        """browser screenshot via MCP returns at least a text content block."""
        import aichat.mcp_server as mcp
        mock_result = {
            "path": "/workspace/screenshot.png",
            "title": "Example",
            "url": "https://example.com",
            "host_path": "/docker/human_browser/workspace/screenshot.png",
        }
        mgr = mcp._get_manager()
        with patch.object(mgr.browser, "screenshot", new=AsyncMock(return_value=mock_result)):
            with patch.object(mgr.db, "store_image", new=AsyncMock(return_value={"ok": True})):
                blocks = await mcp._call_tool("browser", {"action": "screenshot", "url": "https://example.com"})
        assert isinstance(blocks, list)
        assert len(blocks) >= 1
        assert blocks[0]["type"] == "text"
        assert "screenshot" in blocks[0]["text"].lower() or "file" in blocks[0]["text"].lower()

    @pytest.mark.asyncio
    async def test_mcp_screenshot_with_file_includes_image_block(self, tmp_path):
        """If the host_path file exists, MCP adds an image content block."""
        import aichat.mcp_server as mcp
        # Create a tiny valid PNG (1×1 pixel)
        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
            b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        img_file = tmp_path / "screenshot.png"
        img_file.write_bytes(png_bytes)
        mock_result = {
            "path": "/workspace/screenshot.png",
            "title": "Example",
            "url": "https://example.com",
            "host_path": str(img_file),
        }
        mgr = mcp._get_manager()
        with patch.object(mgr.browser, "screenshot", new=AsyncMock(return_value=mock_result)):
            with patch.object(mgr.db, "store_image", new=AsyncMock(return_value={"ok": True})):
                blocks = await mcp._call_tool("browser", {"action": "screenshot", "url": "https://example.com"})
        assert any(b["type"] == "image" for b in blocks), f"Expected image block in: {blocks}"
        img_block = next(b for b in blocks if b["type"] == "image")
        assert img_block["mimeType"] == "image/png"
        assert len(img_block["data"]) > 0

    @pytest.mark.asyncio
    async def test_mcp_db_list_images_no_images(self):
        """db_list_images with empty DB returns text 'No screenshots stored'."""
        import aichat.mcp_server as mcp
        mgr = mcp._get_manager()
        with patch.object(mgr.db, "list_images", new=AsyncMock(return_value={"images": []})):
            blocks = await mcp._call_tool("db_list_images", {})
        assert blocks[0]["type"] == "text"
        assert "no screenshots" in blocks[0]["text"].lower()

    @pytest.mark.asyncio
    async def test_mcp_db_store_image(self):
        """db_store_image tool returns JSON ok response."""
        import aichat.mcp_server as mcp
        mgr = mcp._get_manager()
        with patch.object(mgr.db, "store_image", new=AsyncMock(return_value={"ok": True, "id": 42})):
            blocks = await mcp._call_tool(
                "db_store_image",
                {"url": "https://example.com", "host_path": "/docker/x.png", "alt_text": "Test"},
            )
        assert blocks[0]["type"] == "text"
        data = json.loads(blocks[0]["text"])
        assert data.get("ok") is True

    @pytest.mark.asyncio
    async def test_mcp_tools_call_returns_content_blocks(self):
        """tools/call handler passes content_blocks directly (not wrapped in text)."""
        import aichat.mcp_server as mcp
        captured: list[dict] = []
        original_write = mcp._write

        def _capture(obj):
            captured.append(obj)

        mgr = mcp._get_manager()
        with patch.object(mgr.db, "list_images", new=AsyncMock(return_value={"images": []})):
            with patch.object(mcp, "_write", side_effect=_capture):
                await mcp._handle(json.dumps({
                    "jsonrpc": "2.0",
                    "id": 99,
                    "method": "tools/call",
                    "params": {"name": "db_list_images", "arguments": {}},
                }))

        assert len(captured) == 1
        result = captured[0]["result"]
        # content must be a list of blocks, not a string
        assert isinstance(result["content"], list)
        assert result["content"][0]["type"] == "text"

    @pytest.mark.asyncio
    async def test_mcp_web_search_returns_text_block(self):
        """web_search tool in stdio MCP server returns a text content block."""
        import aichat.mcp_server as mcp
        mgr = mcp._get_manager()
        mock_search_result = {
            "query": "python asyncio",
            "tier": 1,
            "tier_name": "browser",
            "url": "https://duckduckgo.com/?q=python+asyncio",
            "content": "Python asyncio documentation and examples.",
        }
        with patch.object(mgr, "run_web_search", new=AsyncMock(return_value=mock_search_result)):
            blocks = await mcp._call_tool("web_search", {"query": "python asyncio"})
        assert isinstance(blocks, list)
        assert blocks[0]["type"] == "text"
        assert "asyncio" in blocks[0]["text"].lower() or "browser" in blocks[0]["text"].lower()

    @pytest.mark.asyncio
    async def test_mcp_screenshot_find_text_passed(self):
        """screenshot tool with find_text passes it through to run_browser."""
        import aichat.mcp_server as mcp
        mgr = mcp._get_manager()
        captured_kwargs: list[dict] = []

        async def _fake_run_browser(action, mode, confirmer, url=None, find_text=None, **kw):
            captured_kwargs.append({"action": action, "url": url, "find_text": find_text})
            return {
                "path": "/workspace/s.png",
                "title": "Test",
                "url": url or "",
                "host_path": "",
                "clipped": bool(find_text),
            }

        with patch.object(mgr, "run_browser", side_effect=_fake_run_browser):
            await mcp._call_tool("screenshot", {"url": "https://example.com", "find_text": "API"})
        assert len(captured_kwargs) == 1
        assert captured_kwargs[0]["find_text"] == "API"

    def test_mcp_stdio_tool_schemas_complete(self):
        """stdio MCP server exposes all expected tools."""
        import aichat.mcp_server as mcp
        names = {t["name"] for t in mcp._TOOL_SCHEMAS}
        expected = {
            "browser", "screenshot", "fetch_image", "screenshot_search",
            "web_fetch", "web_search", "memory_store", "memory_recall",
            "db_store_article", "db_search", "db_cache_store", "db_cache_get",
            "db_store_image", "db_list_images", "researchbox_search", "researchbox_push",
            "shell_exec", "create_tool", "list_custom_tools", "delete_custom_tool",
        }
        missing = expected - names
        assert not missing, f"stdio MCP missing tools: {missing}"
