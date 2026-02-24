"""
MCP (Model Context Protocol) server for aichat.

Exposes the full aichat tool suite — browser (human_browser / a12fdfeaaf78),
web_fetch, memory, database storage/cache, researchbox, and shell — as MCP
tools that any MCP-compatible client (LM Studio, Claude Desktop, etc.) can use.

Protocol: JSON-RPC 2.0 over stdio (one JSON object per line).

Usage
-----
Run directly:
    python -m aichat.mcp_server

Or via the CLI:
    aichat mcp

LM Studio mcp_servers.json entry
---------------------------------
{
  "mcpServers": {
    "aichat": {
      "command": "/home/<user>/.local/share/aichat/venv/bin/python",
      "args": ["-m", "aichat.mcp_server"],
      "env": {}
    }
  }
}
"""
from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from .state import ApprovalMode
from .tools.manager import ToolManager, ToolDeniedError

# All tool calls run in AUTO mode when invoked via MCP.
_APPROVAL = ApprovalMode.AUTO
_manager: ToolManager | None = None


def _get_manager() -> ToolManager:
    global _manager
    if _manager is None:
        _manager = ToolManager(max_tool_calls_per_turn=999)
    return _manager


# ---------------------------------------------------------------------------
# Tool schema registry — one entry per exposed tool
# ---------------------------------------------------------------------------

_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "browser",
        "description": (
            "Control a real Chromium browser in the human_browser Docker container "
            "(ID a12fdfeaaf78). Actions: navigate, read, screenshot, click, fill, eval."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["navigate", "read", "screenshot", "click", "fill", "eval"],
                    "description": "Which browser action to perform.",
                },
                "url":      {"type": "string", "description": "URL for navigate/screenshot."},
                "selector": {"type": "string", "description": "CSS selector for click/fill."},
                "value":    {"type": "string", "description": "Text to type (fill only)."},
                "code":     {"type": "string", "description": "JavaScript to evaluate (eval only)."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "web_fetch",
        "description": (
            "Fetch a web page using the real Chromium browser (human_browser / a12fdfeaaf78) "
            "and return its readable text. Use for documentation, articles, or any URL."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url":       {"type": "string", "description": "Full URL to fetch (http/https)."},
                "max_chars": {"type": "integer", "description": "Max chars to return (default 4000)."},
            },
            "required": ["url"],
        },
    },
    {
        "name": "memory_store",
        "description": "Store a key-value note in persistent memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "key":   {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "memory_recall",
        "description": "Recall a note from persistent memory (omit key to list all).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Key to look up, or empty to list all."},
            },
            "required": [],
        },
    },
    {
        "name": "db_store_article",
        "description": "Store an article (url, title, content, topic) in PostgreSQL.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url":     {"type": "string"},
                "title":   {"type": "string"},
                "content": {"type": "string"},
                "topic":   {"type": "string"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "db_search",
        "description": "Search stored articles in PostgreSQL by topic and/or full-text query.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Filter by topic (optional)."},
                "q":     {"type": "string", "description": "Full-text search query (optional)."},
            },
            "required": [],
        },
    },
    {
        "name": "db_cache_store",
        "description": "Cache a web page's content in PostgreSQL.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url":     {"type": "string"},
                "content": {"type": "string"},
                "title":   {"type": "string"},
            },
            "required": ["url", "content"],
        },
    },
    {
        "name": "db_cache_get",
        "description": "Retrieve a cached web page from PostgreSQL.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "researchbox_search",
        "description": "Discover RSS feed URLs for a topic.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
            },
            "required": ["topic"],
        },
    },
    {
        "name": "researchbox_push",
        "description": "Fetch an RSS feed and store its articles in PostgreSQL.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "feed_url": {"type": "string"},
                "topic":    {"type": "string"},
            },
            "required": ["feed_url", "topic"],
        },
    },
    {
        "name": "shell_exec",
        "description": "Run a shell command on the host machine.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
            },
            "required": ["command"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

async def _call_tool(name: str, arguments: dict[str, Any]) -> str:
    mgr = _get_manager()
    try:
        if name == "browser":
            result = await mgr.run_browser(
                action=str(arguments.get("action", "")),
                mode=_APPROVAL,
                confirmer=None,
                url=arguments.get("url"),
                selector=arguments.get("selector"),
                value=arguments.get("value"),
                code=arguments.get("code"),
            )
            return json.dumps(result, ensure_ascii=False)

        if name == "web_fetch":
            url = str(arguments.get("url", "")).strip()
            max_chars = int(arguments.get("max_chars", 4000))
            max_chars = max(500, min(max_chars, 16000))
            result = await mgr.run_web_fetch(url, max_chars, _APPROVAL, None)
            text = result.get("text", "")
            truncated = result.get("truncated", False)
            suffix = "\n...[truncated]" if truncated else ""
            return f"{text}{suffix}" if text else "(no content)"

        if name == "memory_store":
            result = await mgr.run_memory_store(
                str(arguments.get("key", "")),
                str(arguments.get("value", "")),
                _APPROVAL,
                None,
            )
            return json.dumps(result, ensure_ascii=False)

        if name == "memory_recall":
            result = await mgr.run_memory_recall(
                str(arguments.get("key", "")),
                _APPROVAL,
                None,
            )
            return json.dumps(result, ensure_ascii=False)

        if name == "db_store_article":
            result = await mgr.run_db_store_article(
                str(arguments.get("url", "")),
                str(arguments.get("title", "")),
                str(arguments.get("content", "")),
                str(arguments.get("topic", "")),
                _APPROVAL,
                None,
            )
            return json.dumps(result, ensure_ascii=False)

        if name == "db_search":
            result = await mgr.run_db_search(
                str(arguments.get("topic", "")),
                str(arguments.get("q", "")),
                _APPROVAL,
                None,
            )
            return json.dumps(result, ensure_ascii=False)

        if name == "db_cache_store":
            result = await mgr.run_db_cache_store(
                str(arguments.get("url", "")),
                str(arguments.get("content", "")),
                str(arguments.get("title", "")),
                _APPROVAL,
                None,
            )
            return json.dumps(result, ensure_ascii=False)

        if name == "db_cache_get":
            result = await mgr.run_db_cache_get(
                str(arguments.get("url", "")),
                _APPROVAL,
                None,
            )
            return json.dumps(result, ensure_ascii=False)

        if name == "researchbox_search":
            result = await mgr.run_researchbox(
                str(arguments.get("topic", "")),
                _APPROVAL,
                None,
            )
            return json.dumps(result, ensure_ascii=False)

        if name == "researchbox_push":
            result = await mgr.run_researchbox_push(
                str(arguments.get("feed_url", "")),
                str(arguments.get("topic", "")),
                _APPROVAL,
                None,
            )
            return json.dumps(result, ensure_ascii=False)

        if name == "shell_exec":
            output, _ = await mgr.run_shell(
                str(arguments.get("command", "")),
                _APPROVAL,
                None,
            )
            return output or "(no output)"

        return f"Unknown tool: {name}"

    except ToolDeniedError as exc:
        return f"Tool denied: {exc}"
    except Exception as exc:
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 helpers
# ---------------------------------------------------------------------------

def _ok(request_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _err(request_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _write(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main request handler
# ---------------------------------------------------------------------------

async def _handle(line: str) -> None:
    try:
        req = json.loads(line)
    except json.JSONDecodeError:
        _write(_err(None, -32700, "Parse error"))
        return

    req_id = req.get("id")
    method = req.get("method", "")
    params = req.get("params") or {}

    if method == "initialize":
        _write(_ok(req_id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {
                "name": "aichat",
                "version": "1.0.0",
            },
        }))

    elif method == "notifications/initialized":
        # Notification — no response needed
        pass

    elif method == "tools/list":
        _write(_ok(req_id, {"tools": _TOOL_SCHEMAS}))

    elif method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments") or {}
        content_text = await _call_tool(tool_name, arguments)
        _write(_ok(req_id, {
            "content": [{"type": "text", "text": content_text}],
            "isError": False,
        }))

    elif method == "ping":
        _write(_ok(req_id, {}))

    else:
        if req_id is not None:
            _write(_err(req_id, -32601, f"Method not found: {method}"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def _run() -> None:
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        try:
            line_bytes = await reader.readline()
        except Exception:
            break
        if not line_bytes:
            break
        line = line_bytes.decode(errors="replace").strip()
        if line:
            await _handle(line)


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
