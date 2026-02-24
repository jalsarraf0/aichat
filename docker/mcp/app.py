"""
aichat MCP HTTP/SSE server — exposes all aichat tools to remote MCP clients
(LM Studio, Claude Desktop, Cursor, etc.) over the network.

MCP SSE transport (spec 2024-11-05):
  GET  /sse           — client connects here; receives an 'endpoint' event
                        pointing to /messages?sessionId=<id>
  POST /messages      — client sends JSON-RPC requests here
  GET  /health        — health probe

The server listens on 0.0.0.0:8096 so it is reachable from other machines
on the same network.

LM Studio mcp_servers.json entry (on the localhost machine):
  {
    "mcpServers": {
      "aichat": {
        "url": "http://<THIS_MACHINE_IP>:8096/sse"
      }
    }
  }

Screenshot support
------------------
The human_browser container must be connected to this container's Docker
network (the install script does this automatically).  Screenshots are written
to /workspace inside human_browser and are bind-mounted read-only into this
container at /browser-workspace.  The result is sent to LM Studio as an inline
base64-encoded PNG image block so it renders directly in the chat.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import uuid
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

app = FastAPI(title="aichat-mcp")

# Base URLs for aichat backend services (running on the same Docker network).
DATABASE_URL = os.environ.get("DATABASE_URL", "http://aichat-database:8091")
MEMORY_URL    = os.environ.get("MEMORY_URL",   "http://aichat-memory:8094")
RESEARCH_URL  = os.environ.get("RESEARCH_URL", "http://aichat-researchbox:8092")
# human_browser browser-server API — reachable after install connects it to this network.
BROWSER_URL   = os.environ.get("BROWSER_URL",  "http://human_browser:7081")
# Screenshot PNGs are bind-mounted from /docker/human_browser/workspace on the host.
BROWSER_WORKSPACE = os.environ.get("BROWSER_WORKSPACE", "/browser-workspace")

# Active SSE sessions: session_id -> asyncio.Queue
_sessions: dict[str, asyncio.Queue] = {}


# ---------------------------------------------------------------------------
# Tool schemas exposed to MCP clients
# ---------------------------------------------------------------------------

_TOOLS: list[dict[str, Any]] = [
    {
        "name": "screenshot",
        "description": (
            "Take a screenshot of any URL using the real Chromium browser and return it as an "
            "inline image rendered directly in the chat. The screenshot is also saved to the "
            "PostgreSQL image registry for later retrieval. "
            "Use this whenever you want to visually inspect a web page."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL to screenshot (http/https)."},
            },
            "required": ["url"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the web using a tiered strategy: "
            "Tier 2 — direct httpx HTTP fetch of DuckDuckGo HTML; "
            "Tier 3 — DuckDuckGo lite API. Returns results text."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query":     {"type": "string", "description": "Search query."},
                "max_chars": {"type": "integer", "description": "Max chars to return (default 4000)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "web_fetch",
        "description": (
            "Fetch a web page and return its readable text. "
            "Checks the PostgreSQL cache first; falls back to a live httpx fetch."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url":       {"type": "string", "description": "Full URL to fetch."},
                "max_chars": {"type": "integer", "description": "Max chars to return (default 4000)."},
            },
            "required": ["url"],
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
                "topic": {"type": "string"},
                "q":     {"type": "string"},
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
        "name": "db_store_image",
        "description": "Save a screenshot or image reference to the PostgreSQL image registry.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url":       {"type": "string", "description": "Source URL the image was captured from."},
                "host_path": {"type": "string", "description": "Host file path of the image."},
                "alt_text":  {"type": "string", "description": "Description or caption."},
            },
            "required": ["url"],
        },
    },
    {
        "name": "db_list_images",
        "description": "List screenshots and images saved in the PostgreSQL database.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max images to return (default 20)."},
            },
            "required": [],
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
                "key": {"type": "string"},
            },
            "required": [],
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
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text(s: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": s}]


def _image_blocks(container_path: str, summary: str) -> list[dict[str, Any]]:
    """Return MCP content blocks: text summary + inline base64 PNG (if readable)."""
    blocks: list[dict[str, Any]] = [{"type": "text", "text": summary}]
    # container_path is e.g. /workspace/screenshot.png inside human_browser.
    # The workspace is bind-mounted at BROWSER_WORKSPACE inside this container.
    if container_path:
        filename = os.path.basename(container_path)
        local_path = os.path.join(BROWSER_WORKSPACE, filename)
        if os.path.isfile(local_path):
            try:
                with open(local_path, "rb") as fh:
                    b64 = base64.standard_b64encode(fh.read()).decode("ascii")
                blocks.append({"type": "image", "data": b64, "mimeType": "image/png"})
            except Exception:
                pass
    return blocks


# ---------------------------------------------------------------------------
# Tool dispatch (HTTP calls to sibling Docker services)
# ---------------------------------------------------------------------------

async def _call_tool(name: str, args: dict[str, Any]) -> list[dict[str, Any]]:
    """Dispatch a tool call and return a list of MCP content blocks."""
    async with httpx.AsyncClient(timeout=60) as c:
        try:
            # ----------------------------------------------------------------
            if name == "screenshot":
                url = str(args.get("url", "")).strip()
                if not url:
                    return _text("screenshot: 'url' is required")
                try:
                    r = await c.post(f"{BROWSER_URL}/screenshot", json={"url": url})
                    data = r.json()
                except Exception as exc:
                    return _text(f"Screenshot failed (browser unreachable): {exc}")
                error = data.get("error", "")
                container_path = data.get("path", "")  # /workspace/screenshot.png
                if error and not container_path:
                    return _text(f"Screenshot failed: {error}")
                # host_path for DB storage (matches the bind-mount source on the host)
                filename = os.path.basename(container_path) if container_path else ""
                host_path = f"/docker/human_browser/workspace/{filename}" if filename else ""
                page = data.get("title", "") or url
                # Auto-save to database
                try:
                    await c.post(f"{DATABASE_URL}/images/store", json={
                        "url": url,
                        "host_path": host_path,
                        "alt_text": f"Screenshot of {page}",
                    })
                except Exception:
                    pass  # never fail the screenshot because DB is down
                summary = (
                    f"Screenshot of: {page}\n"
                    f"URL: {url}\n"
                    f"File: {host_path}"
                )
                return _image_blocks(container_path, summary)

            # ----------------------------------------------------------------
            if name == "web_search":
                query = str(args.get("query", "")).strip()
                max_chars = int(args.get("max_chars", 4000))
                max_chars = max(500, min(max_chars, 16000))
                # Tier 2: DuckDuckGo HTML
                try:
                    r = await c.get(
                        f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}",
                        headers={"User-Agent": "Mozilla/5.0"},
                        follow_redirects=True,
                    )
                    text = re.sub(r"<[^>]+>", " ", r.text)
                    text = re.sub(r"\s+", " ", text).strip()[:max_chars]
                    return _text(f"[Search via httpx (tier 2)]\n\n{text}")
                except Exception:
                    pass
                # Tier 3: DDG lite
                try:
                    r = await c.get(
                        f"https://lite.duckduckgo.com/lite/?q={query.replace(' ', '+')}",
                        headers={"User-Agent": "Mozilla/5.0"},
                        follow_redirects=True,
                    )
                    text = re.sub(r"<[^>]+>", " ", r.text)
                    text = re.sub(r"\s+", " ", text).strip()[:max_chars]
                    return _text(f"[Search via DDG lite (tier 3)]\n\n{text}")
                except Exception as exc:
                    return _text(f"web_search failed: {exc}")

            # ----------------------------------------------------------------
            if name == "web_fetch":
                url = str(args.get("url", "")).strip()
                max_chars = int(args.get("max_chars", 4000))
                max_chars = max(500, min(max_chars, 16000))
                # Check cache first
                try:
                    cache_r = await c.get(f"{DATABASE_URL}/cache/get", params={"url": url})
                    if cache_r.status_code == 200:
                        data = cache_r.json()
                        if data.get("found"):
                            return _text(f"[cached] {data.get('content', '')[:max_chars]}")
                except Exception:
                    pass
                # Fetch live
                r = await c.get(url, headers={"User-Agent": "Mozilla/5.0"}, follow_redirects=True)
                text = r.text[:max_chars]
                try:
                    await c.post(f"{DATABASE_URL}/cache/store", json={"url": url, "content": text})
                except Exception:
                    pass
                return _text(text)

            # ----------------------------------------------------------------
            if name == "db_store_article":
                r = await c.post(f"{DATABASE_URL}/articles/store", json=args)
                return _text(json.dumps(r.json()))

            if name == "db_search":
                r = await c.get(f"{DATABASE_URL}/articles/search", params=args)
                return _text(json.dumps(r.json()))

            if name == "db_cache_store":
                r = await c.post(f"{DATABASE_URL}/cache/store", json=args)
                return _text(json.dumps(r.json()))

            if name == "db_cache_get":
                r = await c.get(f"{DATABASE_URL}/cache/get", params={"url": args.get("url", "")})
                return _text(json.dumps(r.json()))

            # ----------------------------------------------------------------
            if name == "db_store_image":
                r = await c.post(f"{DATABASE_URL}/images/store", json={
                    "url":       args.get("url", ""),
                    "host_path": args.get("host_path", ""),
                    "alt_text":  args.get("alt_text", ""),
                })
                return _text(json.dumps(r.json()))

            if name == "db_list_images":
                limit = int(args.get("limit", 20))
                r = await c.get(f"{DATABASE_URL}/images/list", params={"limit": limit})
                data = r.json()
                images = data.get("images", [])
                if not images:
                    return _text("No screenshots stored yet.")
                lines = [f"Stored screenshots ({len(images)}):"]
                for img in images:
                    hp = img.get("host_path") or img.get("url", "")
                    alt = img.get("alt_text", "")
                    ts = img.get("stored_at", "")[:19].replace("T", " ")
                    lines.append(f"  {hp}" + (f"  [{alt}]" if alt else "") + (f"  {ts}" if ts else ""))
                # Inline the most recent image
                most_recent = images[0].get("path", "") or ""  # container path if available
                return _image_blocks(most_recent, "\n".join(lines))

            # ----------------------------------------------------------------
            if name == "memory_store":
                r = await c.post(f"{MEMORY_URL}/store", json={"key": args["key"], "value": args["value"]})
                return _text(json.dumps(r.json()))

            if name == "memory_recall":
                r = await c.get(f"{MEMORY_URL}/recall", params={"key": args.get("key", "")})
                return _text(json.dumps(r.json()))

            if name == "researchbox_search":
                r = await c.get(f"{RESEARCH_URL}/search-feeds", params={"topic": args.get("topic", "")})
                return _text(json.dumps(r.json()))

            if name == "researchbox_push":
                r = await c.post(f"{RESEARCH_URL}/push-feed", json=args)
                return _text(json.dumps(r.json()))

            return _text(f"Unknown tool: {name}")

        except Exception as exc:
            return _text(f"Error calling {name}: {exc}")


# ---------------------------------------------------------------------------
# JSON-RPC dispatch
# ---------------------------------------------------------------------------

async def _handle_rpc(req: dict[str, Any]) -> dict[str, Any] | None:
    """Process one JSON-RPC request; return response dict or None for notifications."""
    req_id = req.get("id")
    method = req.get("method", "")
    params = req.get("params") or {}

    def ok(result: Any) -> dict:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def err(code: int, msg: str) -> dict:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": msg}}

    if method == "initialize":
        return ok({
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "aichat", "version": "1.0.0"},
        })

    if method in ("notifications/initialized", "initialized"):
        return None  # notification — no response

    if method == "tools/list":
        return ok({"tools": _TOOLS})

    if method == "tools/call":
        tool_name  = params.get("name", "")
        arguments  = params.get("arguments") or {}
        content_blocks = await _call_tool(tool_name, arguments)
        return ok({
            "content": content_blocks,
            "isError": False,
        })

    if method == "ping":
        return ok({})

    if req_id is not None:
        return err(-32601, f"Method not found: {method}")
    return None


# ---------------------------------------------------------------------------
# SSE endpoint — MCP clients connect here
# ---------------------------------------------------------------------------

@app.get("/sse")
async def sse_connect(request: Request) -> StreamingResponse:
    session_id = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    _sessions[session_id] = queue

    async def event_stream():
        # Tell the client where to POST its JSON-RPC requests
        messages_endpoint = f"/messages?sessionId={session_id}"
        yield f"event: endpoint\ndata: {messages_endpoint}\n\n"
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"event: message\ndata: {json.dumps(msg)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            _sessions.pop(session_id, None)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# Messages endpoint — MCP clients POST JSON-RPC requests here
# ---------------------------------------------------------------------------

@app.post("/messages")
async def messages(request: Request, sessionId: str = "") -> Response:
    body = await request.body()
    try:
        rpc = json.loads(body)
    except json.JSONDecodeError:
        return Response(
            content=json.dumps({"jsonrpc": "2.0", "id": None,
                                "error": {"code": -32700, "message": "Parse error"}}),
            media_type="application/json",
            status_code=400,
        )

    response = await _handle_rpc(rpc)
    if response is not None and sessionId in _sessions:
        await _sessions[sessionId].put(response)

    return Response(content="", status_code=202)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"ok": True, "sessions": len(_sessions), "tools": len(_TOOLS)}
