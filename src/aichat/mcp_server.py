"""
MCP (Model Context Protocol) server for aichat.

Exposes the full aichat tool suite — browser (human_browser), web_fetch,
memory, database storage/cache, researchbox, and shell — as MCP tools that
any MCP-compatible client (LM Studio, Claude Desktop, etc.) can use.

Protocol: JSON-RPC 2.0 over stdio (one JSON object per line).

Screenshot responses include an inline base64-encoded PNG image block so
clients that support the MCP image content type (e.g. LM Studio) can render
the screenshot directly.

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
import base64
import json
import os
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
            "(ID a12fdfeaaf78). Actions: navigate, read, screenshot, click, fill, eval. "
            "The browser keeps state between calls so you can navigate, click, fill, and read "
            "in sequence. Use 'find_text' with screenshot to zoom into a specific page section."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["navigate", "read", "screenshot", "click", "fill", "eval"],
                    "description": "Which browser action to perform.",
                },
                "url":       {"type": "string", "description": "URL for navigate/screenshot."},
                "selector":  {"type": "string", "description": "CSS selector for click/fill."},
                "value":     {"type": "string", "description": "Text to type (fill only)."},
                "code":      {"type": "string", "description": "JavaScript to evaluate (eval only)."},
                "find_text": {
                    "type": "string",
                    "description": (
                        "Optional, screenshot only. Scroll to the first occurrence of this text "
                        "and clip the screenshot to show just that region."
                    ),
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "screenshot",
        "description": (
            "Take a screenshot of any URL using the real Chromium browser and return it as an "
            "inline image. The screenshot is automatically saved to the PostgreSQL image registry. "
            "Use 'find_text' to zoom into a specific section of the page."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL to screenshot (http/https)."},
                "find_text": {
                    "type": "string",
                    "description": (
                        "Optional. A word or phrase to search for on the page. "
                        "The screenshot will be clipped to show just the matching region."
                    ),
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "fetch_image",
        "description": (
            "Download an image directly from a URL (jpg, png, gif, webp, etc.), save it to "
            "disk, and return it as an inline rendered image. Use this when the user gives a "
            "direct image URL and wants to view or save it."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Direct URL to the image file."},
            },
            "required": ["url"],
        },
    },
    {
        "name": "screenshot_search",
        "description": (
            "Search the web for a topic or query, screenshot the most relevant result pages, "
            "save them to the PostgreSQL image registry, and return all screenshots inline. "
            "Use this when the user asks to 'find a picture of X', 'show me X', or wants to "
            "visually browse search results. Makes a best-effort: returns whatever succeeds."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Topic or query to search and screenshot."},
                "max_results": {
                    "type": "integer",
                    "description": "Number of result pages to screenshot (default 3, max 5).",
                },
            },
            "required": ["query"],
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
        "name": "db_store_image",
        "description": (
            "Save a screenshot or image to the PostgreSQL image registry. "
            "Records the host file path, source URL, and a description."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url":       {"type": "string", "description": "Source URL the image was captured from."},
                "host_path": {"type": "string", "description": "Host file path (e.g. /docker/human_browser/workspace/screenshot.png)."},
                "alt_text":  {"type": "string", "description": "Description or caption."},
            },
            "required": ["url"],
        },
    },
    {
        "name": "db_list_images",
        "description": "List screenshots and images saved in the PostgreSQL database, with their host file paths.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max images to return (default 20)."},
            },
            "required": [],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the web for a query using a tiered strategy: real Chromium browser "
            "(Tier 1, human-like DuckDuckGo search) → direct httpx fetch (Tier 2) → "
            "DuckDuckGo lite API (Tier 3). Returns search results page text."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query":     {"type": "string", "description": "The search query."},
                "max_chars": {"type": "integer", "description": "Max chars to return (default 4000, max 16000)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "create_tool",
        "description": (
            "Create a new persistent custom tool that runs in the aichat-toolkit Docker "
            "sandbox. The tool is saved to disk and immediately available for use in this "
            "and all future sessions. Tools can make HTTP calls (httpx), process data, "
            "call APIs, run shell commands (subprocess), and read user repos at /data/repos."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": "Snake_case identifier (e.g. 'get_stock_price').",
                },
                "description": {
                    "type": "string",
                    "description": "What the tool does.",
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
                        "Available: asyncio, json, re, os, math, datetime, pathlib, shlex, "
                        "subprocess, httpx. Must return a string."
                    ),
                },
            },
            "required": ["tool_name", "description", "parameters_schema", "code"],
        },
    },
    {
        "name": "list_custom_tools",
        "description": "List all custom tools previously created, with names, descriptions, and parameter schemas.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "delete_custom_tool",
        "description": "Permanently delete a custom tool you previously created.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": "Name of the tool to delete.",
                },
            },
            "required": ["tool_name"],
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
# Tool dispatch — returns list of MCP content blocks
# ---------------------------------------------------------------------------

def _image_content_blocks(host_path: str, text: str) -> list[dict[str, Any]]:
    """Build MCP content list with a text summary and inline base64 PNG image."""
    blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]
    if host_path and os.path.isfile(host_path):
        try:
            with open(host_path, "rb") as fh:
                b64 = base64.standard_b64encode(fh.read()).decode("ascii")
            blocks.append({"type": "image", "data": b64, "mimeType": "image/png"})
        except Exception:
            pass
    return blocks


async def _call_tool(name: str, arguments: dict[str, Any]) -> list[dict[str, Any]]:
    """Dispatch a tool call and return a list of MCP content blocks."""
    mgr = _get_manager()

    def _text(s: str) -> list[dict[str, Any]]:
        return [{"type": "text", "text": s}]

    try:
        if name == "browser":
            action = str(arguments.get("action", ""))
            find_text = str(arguments["find_text"]).strip() if arguments.get("find_text") else None
            result = await mgr.run_browser(
                action=action,
                mode=_APPROVAL,
                confirmer=None,
                url=arguments.get("url"),
                selector=arguments.get("selector"),
                value=arguments.get("value"),
                code=arguments.get("code"),
                find_text=find_text,
            )
            if action == "screenshot":
                host_path = result.get("host_path", "")
                error = result.get("error", "")
                if error and not host_path:
                    return _text(f"Screenshot failed: {error}")
                page = result.get("title", "") or result.get("url", "")
                summary = (
                    f"Screenshot saved.\n"
                    f"File: {host_path}\n"
                    f"Page: {page}"
                )
                return _image_content_blocks(host_path, summary)
            return _text(json.dumps(result, ensure_ascii=False))

        if name == "screenshot":
            url = str(arguments.get("url", "")).strip()
            if not url:
                return _text("screenshot: 'url' is required")
            find_text = str(arguments["find_text"]).strip() if arguments.get("find_text") else None
            result = await mgr.run_browser(
                action="screenshot",
                mode=_APPROVAL,
                confirmer=None,
                url=url,
                find_text=find_text,
            )
            host_path = result.get("host_path", "")
            error = result.get("error", "")
            if error and not host_path:
                return _text(f"Screenshot failed: {error}")
            page = result.get("title", "") or result.get("url", url)
            clipped = result.get("clipped", False)
            clip_note = f"\nZoomed to: '{find_text}'" if clipped and find_text else ""
            summary = (
                f"Screenshot of: {page}\n"
                f"URL: {url}{clip_note}\n"
                f"File: {host_path}"
            )
            return _image_content_blocks(host_path, summary)

        if name == "fetch_image":
            url = str(arguments.get("url", "")).strip()
            if not url:
                return _text("fetch_image: 'url' is required")
            result = await mgr.run_fetch_image(url, _APPROVAL, None)
            error = result.get("error", "")
            if error:
                return _text(f"fetch_image failed: {error}")
            host_path = result.get("host_path", "")
            content_type = result.get("content_type", "image/jpeg")
            size = result.get("size", 0)
            summary = (
                f"Image saved.\n"
                f"Source: {url}\n"
                f"File: {host_path}\n"
                f"Type: {content_type}  Size: {size:,} bytes"
            )
            return _image_content_blocks(host_path, summary)

        if name == "screenshot_search":
            query = str(arguments.get("query", "")).strip()
            if not query:
                return _text("screenshot_search: 'query' is required")
            max_results = min(int(arguments.get("max_results", 3)), 5)
            result = await mgr.run_screenshot_search(query, max_results, _APPROVAL, None)
            error = result.get("error", "")
            screenshots = result.get("screenshots", [])
            if error and not screenshots:
                return _text(f"Screenshot search failed: {error}")
            blocks: list[dict[str, Any]] = [{"type": "text", "text": f"Visual search: '{query}' — {len(screenshots)} result(s)\n"}]
            for shot in screenshots:
                host_path = shot.get("host_path", "")
                url_s = shot.get("url", "")
                title = shot.get("title", "")
                err = shot.get("error", "")
                if err and not host_path:
                    blocks.append({"type": "text", "text": f"Failed: {url_s} — {err}"})
                else:
                    summary = f"{title or url_s}\n{url_s}\nFile: {host_path}"
                    blocks.extend(_image_content_blocks(host_path, summary))
            return blocks

        if name == "web_fetch":
            url = str(arguments.get("url", "")).strip()
            max_chars = int(arguments.get("max_chars", 4000))
            max_chars = max(500, min(max_chars, 16000))
            result = await mgr.run_web_fetch(url, max_chars, _APPROVAL, None)
            text = result.get("text", "")
            truncated = result.get("truncated", False)
            suffix = "\n...[truncated]" if truncated else ""
            return _text(f"{text}{suffix}" if text else "(no content)")

        if name == "memory_store":
            result = await mgr.run_memory_store(
                str(arguments.get("key", "")),
                str(arguments.get("value", "")),
                _APPROVAL, None,
            )
            return _text(json.dumps(result, ensure_ascii=False))

        if name == "memory_recall":
            result = await mgr.run_memory_recall(
                str(arguments.get("key", "")),
                _APPROVAL, None,
            )
            return _text(json.dumps(result, ensure_ascii=False))

        if name == "db_store_article":
            result = await mgr.run_db_store_article(
                str(arguments.get("url", "")),
                str(arguments.get("title", "")),
                str(arguments.get("content", "")),
                str(arguments.get("topic", "")),
                _APPROVAL, None,
            )
            return _text(json.dumps(result, ensure_ascii=False))

        if name == "db_search":
            result = await mgr.run_db_search(
                str(arguments.get("topic", "")),
                str(arguments.get("q", "")),
                _APPROVAL, None,
            )
            return _text(json.dumps(result, ensure_ascii=False))

        if name == "db_cache_store":
            result = await mgr.run_db_cache_store(
                str(arguments.get("url", "")),
                str(arguments.get("content", "")),
                str(arguments.get("title", "")),
                _APPROVAL, None,
            )
            return _text(json.dumps(result, ensure_ascii=False))

        if name == "db_cache_get":
            result = await mgr.run_db_cache_get(
                str(arguments.get("url", "")),
                _APPROVAL, None,
            )
            return _text(json.dumps(result, ensure_ascii=False))

        if name == "db_store_image":
            result = await mgr.run_db_store_image(
                str(arguments.get("url", "")),
                str(arguments.get("host_path", "")),
                str(arguments.get("alt_text", "")),
                _APPROVAL, None,
            )
            return _text(json.dumps(result, ensure_ascii=False))

        if name == "db_list_images":
            limit = int(arguments.get("limit", 20))
            result = await mgr.run_db_list_images(limit, _APPROVAL, None)
            images = result.get("images", [])
            if not images:
                return _text("No screenshots stored yet.")
            lines = [f"Stored screenshots ({len(images)}):"]
            for img in images:
                hp = img.get("host_path") or img.get("url", "")
                alt = img.get("alt_text", "")
                ts = (img.get("stored_at", "")[:19]).replace("T", " ") if img.get("stored_at") else ""
                lines.append(f"  {hp}" + (f"  [{alt}]" if alt else "") + (f"  {ts}" if ts else ""))
            # Include the most recent image inline if it exists
            most_recent = images[0].get("host_path", "") if images else ""
            return _image_content_blocks(most_recent, "\n".join(lines))

        if name == "researchbox_search":
            result = await mgr.run_researchbox(
                str(arguments.get("topic", "")),
                _APPROVAL, None,
            )
            return _text(json.dumps(result, ensure_ascii=False))

        if name == "researchbox_push":
            result = await mgr.run_researchbox_push(
                str(arguments.get("feed_url", "")),
                str(arguments.get("topic", "")),
                _APPROVAL, None,
            )
            return _text(json.dumps(result, ensure_ascii=False))

        if name == "web_search":
            query = str(arguments.get("query", "")).strip()
            max_chars = int(arguments.get("max_chars", 4000))
            max_chars = max(500, min(max_chars, 16000))
            result = await mgr.run_web_search(query, max_chars, _APPROVAL, None)
            content = result.get("content", "")
            tier = result.get("tier_name", "")
            header = f"[{tier}]\n\n" if tier else ""
            return _text(f"{header}{content}" if content else "(no results)")

        if name == "create_tool":
            tool_name = str(arguments.get("tool_name", "")).strip()
            description = str(arguments.get("description", "")).strip()
            parameters_schema = arguments.get("parameters_schema", {})
            code = str(arguments.get("code", "")).strip()
            result = await mgr.run_create_tool(
                tool_name, description, parameters_schema, code, _APPROVAL, None,
            )
            return _text(json.dumps(result, ensure_ascii=False))

        if name == "list_custom_tools":
            tools = await mgr.run_list_custom_tools(_APPROVAL, None)
            return _text(json.dumps({"tools": tools}, ensure_ascii=False))

        if name == "delete_custom_tool":
            tool_name = str(arguments.get("tool_name", "")).strip()
            result = await mgr.run_delete_custom_tool(tool_name, _APPROVAL, None)
            return _text(json.dumps(result, ensure_ascii=False))

        if name == "shell_exec":
            output, _ = await mgr.run_shell(
                str(arguments.get("command", "")),
                _APPROVAL, None,
            )
            return _text(output or "(no output)")

        return _text(f"Unknown tool: {name}")

    except ToolDeniedError as exc:
        return _text(f"Tool denied: {exc}")
    except Exception as exc:
        return _text(f"Error: {exc}")


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
        content_blocks = await _call_tool(tool_name, arguments)
        _write(_ok(req_id, {
            "content": content_blocks,
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
