from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
from collections.abc import Awaitable, Callable
from enum import Enum

from ..state import ApprovalMode
from .browser import BrowserTool
from .database import DatabaseTool
from .memory import MemoryTool
from .researchbox import ResearchboxTool
from .search import WebSearchTool
from .shell import ShellTool
from .toolkit import ToolkitTool


class ToolDeniedError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Dangerous-command blocklist (shell safety)
# ---------------------------------------------------------------------------

_DANGEROUS_RE = re.compile(
    r"""(
        # rm with recursive+force targeting root, home, or critical paths
        \brm\b[^|&;#\n]*-[a-zA-Z]*[rf][a-zA-Z]*[rf][a-zA-Z]*\s+(/[^/]|~/?[^/]|/\*)
        |\brm\b[^|&;#\n]*-[a-zA-Z]*f[a-zA-Z]*r[a-zA-Z]*\s+(/[^/]|~/?[^/]|/\*)
        # fork bomb
        |:\(\)\s*\{[^}]*:\s*\|[^}]*:[^}]*&[^}]*\}[^;]*;
        # mkfs / wipefs / dd to raw disk
        |\bmkfs\b
        |\bwipefs\b
        |\bdd\b[^|&;\n]*\bof\s*=\s*/dev/[a-z]
        # overwrite MBR/disk
        |\bdd\b[^|&;\n]*\bif\s*=\s*/dev/zero[^|&;\n]*\bof\s*=\s*/dev/[a-z]
        # chmod/chown 777 or world-writable on root or /etc
        |\bchmod\b[^|&;\n]*(777|[ao][+\-=]w)[^|&;\n]*(/\s*$|/etc|/usr|/bin|/sbin|/lib|/boot)
        # truncate / shred critical paths
        |\bshred\b[^|&;\n]*(/(bin|sbin|lib|boot|etc|usr)\b|/dev/)
        # kill -9 1 (kill PID 1 = init)
        |\bkill\s+-9\s+1\b
        # Python/Perl one-liner writing to /dev/sda etc.
        |\bopen\s*\(\s*['"]/dev/[a-z]
    )""",
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)


def _is_dangerous(command: str) -> bool:
    """Return True if the command matches a known-dangerous pattern."""
    return bool(_DANGEROUS_RE.search(command))


class ToolName(str, Enum):
    SHELL = "shell"
    RESEARCHBOX = "researchbox"
    RESEARCHBOX_PUSH = "researchbox_push"
    WEB_FETCH = "web_fetch"
    MEMORY_STORE = "memory_store"
    MEMORY_RECALL = "memory_recall"
    CREATE_TOOL = "create_tool"
    LIST_CUSTOM_TOOLS = "list_custom_tools"
    DELETE_CUSTOM_TOOL = "delete_custom_tool"
    WEB_SEARCH = "web_search"
    BROWSER = "browser"
    DB_STORE_ARTICLE = "db_store_article"
    DB_SEARCH = "db_search"
    DB_CACHE_STORE = "db_cache_store"
    DB_CACHE_GET = "db_cache_get"
    DB_STORE_IMAGE = "db_store_image"
    DB_LIST_IMAGES = "db_list_images"


class ToolManager:
    def __init__(self, max_tool_calls_per_turn: int = 1) -> None:
        self.shell = ShellTool()
        self.researchbox = ResearchboxTool()
        self.memory = MemoryTool()
        self.toolkit = ToolkitTool()
        # human_browser container (a12fdfeaaf78) — default for ALL web operations.
        self.browser = BrowserTool()
        # Tiered search: browser (human-like) → httpx → DDG lite API.
        self.search_tool = WebSearchTool(self.browser)
        # PostgreSQL-backed storage/cache (replaces rssfeed + fetch services).
        self.db = DatabaseTool()
        self.max_tool_calls_per_turn = max_tool_calls_per_turn
        self._calls_this_turn = 0
        # name → {description, parameters} for custom tools loaded from toolkit
        self._custom_tools: dict[str, dict] = {}

    def reset_turn(self) -> None:
        self._calls_this_turn = 0

    # ------------------------------------------------------------------
    # Custom tool registry
    # ------------------------------------------------------------------

    async def refresh_custom_tools(self) -> None:
        """Reload custom tool metadata from the toolkit service."""
        try:
            tools = await self.toolkit.list_tools()
            self._custom_tools = {
                t["name"]: {"description": t["description"], "parameters": t["parameters"]}
                for t in tools
            }
        except Exception:
            # Toolkit service unavailable — keep last known state
            pass

    def is_custom_tool(self, name: str) -> bool:
        return name in self._custom_tools

    # ------------------------------------------------------------------
    # Approval gate
    # ------------------------------------------------------------------

    async def _check_approval(
        self,
        mode: ApprovalMode,
        tool: str,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> None:
        if self._calls_this_turn >= self.max_tool_calls_per_turn:
            raise ToolDeniedError("Tool call limit reached for current turn")
        if mode == ApprovalMode.DENY:
            raise ToolDeniedError("Tool execution denied by current approval mode")
        if mode == ApprovalMode.ASK:
            if confirmer is None or not await confirmer(tool):
                raise ToolDeniedError(f"Tool '{tool}' rejected by user")
        self._calls_this_turn += 1

    # ------------------------------------------------------------------
    # Built-in tools
    # ------------------------------------------------------------------

    async def run_shell(
        self,
        command: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
        cwd: str | None = None,
    ) -> tuple[str, str | None]:
        if _is_dangerous(command):
            raise ToolDeniedError(f"Blocked: potentially destructive command refused for safety.")
        await self._check_approval(mode, ToolName.SHELL.value, confirmer)
        exit_code, output, new_cwd = await self._run_shell_process(command, cwd=cwd)
        trimmed = output.strip()
        if exit_code != 0:
            if trimmed:
                return f"{trimmed}\n(exit {exit_code})", new_cwd
            return f"(exit {exit_code})", new_cwd
        return trimmed, new_cwd

    async def run_shell_stream(
        self,
        command: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
        *,
        cwd: str | None = None,
        on_output: Callable[[str], None] | None = None,
    ) -> tuple[int, str, str | None]:
        if _is_dangerous(command):
            raise ToolDeniedError(f"Blocked: potentially destructive command refused for safety.")
        await self._check_approval(mode, ToolName.SHELL.value, confirmer)
        return await self._run_shell_process(command, cwd=cwd, on_output=on_output)

    async def _run_shell_process(
        self,
        command: str,
        *,
        cwd: str | None = None,
        on_output: Callable[[str], None] | None = None,
    ) -> tuple[int, str, str | None]:
        marker = "__AICHAT_CWD__"
        command = self._ensure_non_interactive_sudo(command)
        wrapped = self._wrap_command_with_pwd(command, marker)
        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-lc",
            wrapped,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
            env=os.environ.copy(),
        )
        output_chunks: list[str] = []
        buffer = ""
        cwd_value: str | None = None
        marker_token = f"\n{marker}"
        assert proc.stdout is not None
        while True:
            data = await proc.stdout.read(1024)
            if not data:
                break
            text = data.decode(errors="replace")
            buffer += text
            if marker_token in buffer:
                before, after = buffer.split(marker_token, 1)
                if before:
                    output_chunks.append(before)
                    if on_output:
                        on_output(before)
                cwd_value = after.strip() or None
                buffer = ""
                continue
            if len(buffer) > len(marker_token):
                safe = buffer[:-len(marker_token)]
                if safe:
                    output_chunks.append(safe)
                    if on_output:
                        on_output(safe)
                buffer = buffer[-len(marker_token):]
        exit_code = await proc.wait()
        if buffer and cwd_value is None:
            output_chunks.append(buffer)
            if on_output:
                on_output(buffer)
        return exit_code, "".join(output_chunks).strip(), cwd_value

    def _ensure_non_interactive_sudo(self, command: str) -> str:
        stripped = command.lstrip()
        if not stripped.startswith("sudo "):
            return command
        try:
            parts = shlex.split(command)
        except ValueError:
            return command
        if not parts or parts[0] != "sudo":
            return command
        if "-n" in parts or "--non-interactive" in parts:
            return command
        parts.insert(1, "-n")
        return shlex.join(parts)

    def _wrap_command_with_pwd(self, command: str, marker: str) -> str:
        if not command.strip():
            return command
        return (
            f"{command}\n"
            "status=$?\n"
            f"printf '\\n{marker}%s' \"$PWD\"\n"
            "exit $status"
        )

    async def run_researchbox(
        self,
        topic: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        await self._check_approval(mode, ToolName.RESEARCHBOX.value, confirmer)
        return await self.researchbox.rb_search_feeds(topic)

    async def run_researchbox_push(
        self,
        feed_url: str,
        topic: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        await self._check_approval(mode, ToolName.RESEARCHBOX_PUSH.value, confirmer)
        return await self.researchbox.rb_push_feeds(feed_url, topic)

    async def run_web_fetch(
        self,
        url: str,
        max_chars: int,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        """Fetch a URL using the human_browser container (a12fdfeaaf78).

        Returns the same shape as the old aichat-fetch service so all callers
        remain compatible: {"text": ..., "truncated": ..., "char_count": ...}.
        """
        await self._check_approval(mode, ToolName.WEB_FETCH.value, confirmer)
        result = await self.browser.navigate(url)
        content = result.get("content", "")
        truncated = False
        if len(content) > max_chars:
            content = content[:max_chars]
            truncated = True
        return {
            "url": result.get("url", url),
            "title": result.get("title", ""),
            "text": content,
            "char_count": len(content),
            "truncated": truncated,
        }

    async def run_web_search(
        self,
        query: str,
        max_chars: int,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        """Search the web using tiered strategy: browser → httpx → DDG lite API.

        Returns {query, tier, tier_name, url, content[, error]}.
        """
        await self._check_approval(mode, ToolName.WEB_SEARCH.value, confirmer)
        max_chars = max(500, min(max_chars, 16000))
        return await self.search_tool.search(query, max_chars=max_chars)

    async def run_memory_store(
        self,
        key: str,
        value: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        await self._check_approval(mode, ToolName.MEMORY_STORE.value, confirmer)
        return await self.memory.store(key, value)

    async def run_memory_recall(
        self,
        key: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        await self._check_approval(mode, ToolName.MEMORY_RECALL.value, confirmer)
        return await self.memory.recall(key)

    # ------------------------------------------------------------------
    # Database tools (PostgreSQL storage / web cache)
    # ------------------------------------------------------------------

    async def run_db_store_article(
        self,
        url: str,
        title: str,
        content: str,
        topic: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        await self._check_approval(mode, ToolName.DB_STORE_ARTICLE.value, confirmer)
        return await self.db.store_article(url, title=title, content=content, topic=topic)

    async def run_db_search(
        self,
        topic: str,
        q: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        await self._check_approval(mode, ToolName.DB_SEARCH.value, confirmer)
        return await self.db.search_articles(topic=topic or None, q=q or None)

    async def run_db_cache_store(
        self,
        url: str,
        content: str,
        title: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        await self._check_approval(mode, ToolName.DB_CACHE_STORE.value, confirmer)
        return await self.db.cache_store(url, content, title=title or None)

    async def run_db_cache_get(
        self,
        url: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        await self._check_approval(mode, ToolName.DB_CACHE_GET.value, confirmer)
        return await self.db.cache_get(url)

    async def run_db_store_image(
        self,
        url: str,
        host_path: str,
        alt_text: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        await self._check_approval(mode, ToolName.DB_STORE_IMAGE.value, confirmer)
        return await self.db.store_image(url=url, host_path=host_path or None, alt_text=alt_text or None)

    async def run_db_list_images(
        self,
        limit: int,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        await self._check_approval(mode, ToolName.DB_LIST_IMAGES.value, confirmer)
        return await self.db.list_images(limit=limit)

    # ------------------------------------------------------------------
    # Toolkit meta-tools (create / list / delete / call custom tools)
    # ------------------------------------------------------------------

    async def run_create_tool(
        self,
        tool_name: str,
        description: str,
        parameters: dict,
        code: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        await self._check_approval(mode, ToolName.CREATE_TOOL.value, confirmer)
        result = await self.toolkit.register_tool(tool_name, description, parameters, code)
        await self.refresh_custom_tools()
        return result

    async def run_list_custom_tools(
        self,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> list[dict]:
        await self._check_approval(mode, ToolName.LIST_CUSTOM_TOOLS.value, confirmer)
        tools = await self.toolkit.list_tools()
        await self.refresh_custom_tools()
        return tools

    async def run_delete_custom_tool(
        self,
        tool_name: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        await self._check_approval(mode, ToolName.DELETE_CUSTOM_TOOL.value, confirmer)
        result = await self.toolkit.delete_tool(tool_name)
        await self.refresh_custom_tools()
        return result

    async def run_custom_tool(
        self,
        tool_name: str,
        params: dict,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        await self._check_approval(mode, tool_name, confirmer)
        return await self.toolkit.call_tool(tool_name, params)

    async def run_browser(
        self,
        action: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
        url: str | None = None,
        selector: str | None = None,
        value: str | None = None,
        code: str | None = None,
    ) -> dict:
        await self._check_approval(mode, ToolName.BROWSER.value, confirmer)
        try:
            if action == "navigate":
                if not url:
                    return {"error": "url is required for navigate"}
                return await self.browser.navigate(url)
            if action == "screenshot":
                result = await self.browser.screenshot(url)
                # Auto-persist screenshot metadata to the image database
                host_path = result.get("host_path", "")
                if host_path and not result.get("error"):
                    try:
                        page_url = result.get("url") or url or host_path
                        alt = f"Screenshot of {result.get('title', page_url)}"
                        await self.db.store_image(
                            url=page_url,
                            host_path=host_path,
                            alt_text=alt,
                        )
                    except Exception:
                        pass  # Never fail the screenshot because DB is down
                return result
            if action == "click":
                if not selector:
                    return {"error": "selector is required for click"}
                return await self.browser.click(selector)
            if action == "fill":
                if not selector or value is None:
                    return {"error": "selector and value are required for fill"}
                return await self.browser.fill(selector, value)
            if action == "read":
                return await self.browser.read()
            if action == "eval":
                if not code:
                    return {"error": "code is required for eval"}
                return await self.browser.eval_js(code)
            return {
                "error": (
                    f"Unknown action '{action}'. "
                    "Valid: navigate, read, screenshot, click, fill, eval"
                )
            }
        except Exception as exc:
            return {"error": str(exc)}

    def active_sessions(self) -> list[str]:
        return [f"shell:{sid}" for sid in self.shell.sessions]

    # ------------------------------------------------------------------
    # Tool definitions (sent to LLM each turn)
    # ------------------------------------------------------------------

    def tool_definitions(self, shell_enabled: bool) -> list[dict[str, object]]:
        tools: list[dict[str, object]] = [
            # ----------------------------------------------------------
            # Web search — tiered: browser (human-like) → httpx → DDG lite
            # ----------------------------------------------------------
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": (
                        "Search the web for a query using a tiered strategy. "
                        "Tier 1 (preferred): opens a real Chromium browser (human_browser / a12fdfeaaf78), "
                        "navigates to DuckDuckGo, types the query and presses Enter — exactly as a human would. "
                        "If the browser is unavailable or times out, falls back to a direct HTTP fetch "
                        "(Tier 2), then to the DuckDuckGo lite API (Tier 3). "
                        "Returns the search results page text and which tier was used."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query.",
                            },
                            "max_chars": {
                                "type": "integer",
                                "description": "Maximum characters to return (default 4000, max 16000).",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            # ----------------------------------------------------------
            # Web browsing — all web operations go through human_browser
            # ----------------------------------------------------------
            {
                "type": "function",
                "function": {
                    "name": "web_fetch",
                    "description": (
                        "Fetch a web page using the real Chromium browser "
                        "(human_browser container, ID a12fdfeaaf78) and return its "
                        "readable text content. Use this to read documentation, "
                        "articles, GitHub files, or any URL."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The full URL to fetch (http/https).",
                            },
                            "max_chars": {
                                "type": "integer",
                                "description": "Maximum characters to return (default 4000, max 16000).",
                            },
                        },
                        "required": ["url"],
                    },
                },
            },
            # ----------------------------------------------------------
            # RSS/feed discovery (researchbox)
            # ----------------------------------------------------------
            {
                "type": "function",
                "function": {
                    "name": "researchbox_search",
                    "description": "Search for RSS feed sources for a topic via the docker researchbox service.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "Topic to search for feeds.",
                            }
                        },
                        "required": ["topic"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "researchbox_push",
                    "description": (
                        "Fetch an RSS feed and store its articles in the PostgreSQL "
                        "database for later retrieval and comparison."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "feed_url": {
                                "type": "string",
                                "description": "RSS feed URL to ingest.",
                            },
                            "topic": {
                                "type": "string",
                                "description": "Topic label to store articles under.",
                            },
                        },
                        "required": ["feed_url", "topic"],
                    },
                },
            },
            # ----------------------------------------------------------
            # PostgreSQL storage / cache
            # ----------------------------------------------------------
            {
                "type": "function",
                "function": {
                    "name": "db_store_article",
                    "description": (
                        "Store an article (URL, title, content, topic) in the "
                        "PostgreSQL database for long-term retrieval and comparison "
                        "with future results."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url":     {"type": "string", "description": "Article URL."},
                            "title":   {"type": "string", "description": "Article title."},
                            "content": {"type": "string", "description": "Article text content."},
                            "topic":   {"type": "string", "description": "Topic label."},
                        },
                        "required": ["url"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "db_search",
                    "description": (
                        "Search previously stored articles in the PostgreSQL database. "
                        "Use to compare new web results against what was stored before."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "Filter by topic (optional).",
                            },
                            "q": {
                                "type": "string",
                                "description": "Full-text search query (optional).",
                            },
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "db_cache_store",
                    "description": (
                        "Cache a web page's content in PostgreSQL so it can be retrieved "
                        "later without re-fetching."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url":     {"type": "string", "description": "Page URL."},
                            "content": {"type": "string", "description": "Page content to cache."},
                            "title":   {"type": "string", "description": "Page title (optional)."},
                        },
                        "required": ["url", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "db_cache_get",
                    "description": (
                        "Retrieve a previously cached web page from PostgreSQL. "
                        "Returns {found: false} if the URL has not been cached yet."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "Page URL to look up."},
                        },
                        "required": ["url"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "db_store_image",
                    "description": (
                        "Save an image or screenshot to the PostgreSQL image registry. "
                        "Use after taking a screenshot or downloading an image to record it "
                        "permanently with its source URL and description."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url":       {"type": "string", "description": "Source URL the image was captured from."},
                            "host_path": {"type": "string", "description": "File path on the host (e.g. /docker/human_browser/workspace/screenshot.png)."},
                            "alt_text":  {"type": "string", "description": "Description or caption for the image."},
                        },
                        "required": ["url"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "db_list_images",
                    "description": (
                        "List screenshots and images previously saved to the PostgreSQL database. "
                        "Returns host file paths so the user can open them."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of images to return (default 20).",
                            },
                        },
                        "required": [],
                    },
                },
            },
            # ----------------------------------------------------------
            # Memory
            # ----------------------------------------------------------
            {
                "type": "function",
                "function": {
                    "name": "memory_store",
                    "description": (
                        "Store a note or fact in persistent memory for recall later. "
                        "Storing the same key overwrites the previous value."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "key":   {"type": "string", "description": "Short label for this memory entry."},
                            "value": {"type": "string", "description": "Content to remember."},
                        },
                        "required": ["key", "value"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "memory_recall",
                    "description": (
                        "Retrieve notes from persistent memory. "
                        "Provide a key to look up a specific entry, or leave empty to list all stored keys."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "key": {
                                "type": "string",
                                "description": "Key to look up (omit or empty string to list all).",
                            }
                        },
                        "required": [],
                    },
                },
            },
            # ----------------------------------------------------------
            # Toolkit meta-tools
            # ----------------------------------------------------------
            {
                "type": "function",
                "function": {
                    "name": "create_tool",
                    "description": (
                        "Create a new persistent custom tool that runs in an isolated Docker container. "
                        "The tool is saved to disk and immediately available for use — in this session "
                        "and all future sessions. Use this whenever you need a capability not covered "
                        "by the built-in tools. You can make HTTP calls (httpx), process data, parse "
                        "HTML, call APIs, run shell commands (subprocess/asyncio.create_subprocess_exec), "
                        "and read files from the user's repos at /data/repos. Tools persist across restarts."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tool_name": {
                                "type": "string",
                                "description": "Snake_case identifier (e.g. 'search_wikipedia', 'get_stock_price').",
                            },
                            "description": {
                                "type": "string",
                                "description": "What the tool does — shown to you when deciding which tool to call.",
                            },
                            "parameters_schema": {
                                "type": "object",
                                "description": (
                                    "JSON Schema object for the tool's inputs. Example: "
                                    '{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}'
                                ),
                            },
                            "code": {
                                "type": "string",
                                "description": (
                                    "Python implementation — the body of `async def run(**kwargs) -> str:`. "
                                    "Available: asyncio, json, re, os, math, datetime, pathlib, shlex, subprocess, httpx. "
                                    "User git repos are at /data/repos (Path('/data/repos/reponame')). "
                                    "Run shell commands with asyncio.create_subprocess_exec or subprocess.run. "
                                    "Access parameters via kwargs. Must return a string."
                                ),
                            },
                        },
                        "required": ["tool_name", "description", "parameters_schema", "code"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_custom_tools",
                    "description": "List all custom tools you have created, with their names, descriptions, and parameter schemas.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_custom_tool",
                    "description": "Permanently delete a custom tool you previously created.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tool_name": {
                                "type": "string",
                                "description": "Name of the tool to delete.",
                            }
                        },
                        "required": ["tool_name"],
                    },
                },
            },
        ]

        # Browser — human_browser container (ID a12fdfeaaf78)
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "browser",
                    "description": (
                        "Control a real Chromium browser running in the human_browser "
                        "Docker container (ID a12fdfeaaf78) via Playwright. "
                        "This is the default for ALL web-related operations. "
                        "The browser keeps its state between calls (same session), so you "
                        "can navigate to a page, then click a button, then read the result. "
                        "Actions: "
                        "navigate — go to a URL and return page title + text; "
                        "read — return the current page title + text without navigating; "
                        "screenshot — capture the page as a PNG saved to /workspace (returns file path); "
                        "click — click a CSS selector; "
                        "fill — type a value into a CSS selector input; "
                        "eval — run a JavaScript expression and return its result."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["navigate", "read", "screenshot", "click", "fill", "eval"],
                                "description": "Which browser action to perform.",
                            },
                            "url": {
                                "type": "string",
                                "description": "URL to navigate to (navigate / screenshot).",
                            },
                            "selector": {
                                "type": "string",
                                "description": "CSS selector for click / fill.",
                            },
                            "value": {
                                "type": "string",
                                "description": "Text to type into the element (fill only).",
                            },
                            "code": {
                                "type": "string",
                                "description": "JavaScript expression to evaluate (eval only).",
                            },
                        },
                        "required": ["action"],
                    },
                },
            }
        )

        if shell_enabled:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "shell_exec",
                        "description": "Run a shell command on the host machine.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "command": {
                                    "type": "string",
                                    "description": "Shell command to execute.",
                                }
                            },
                            "required": ["command"],
                        },
                    },
                }
            )

        # Append any currently registered custom tools so the LLM can call them
        for name, meta in self._custom_tools.items():
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": meta.get("description", "Custom tool."),
                        "parameters": meta.get(
                            "parameters",
                            {"type": "object", "properties": {}, "required": []},
                        ),
                    },
                }
            )

        return tools
