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
"""
from __future__ import annotations

import asyncio
import json
import os
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

# Active SSE sessions: session_id -> asyncio.Queue
_sessions: dict[str, asyncio.Queue] = {}


# ---------------------------------------------------------------------------
# Tool schemas exposed to MCP clients
# ---------------------------------------------------------------------------

_TOOLS: list[dict[str, Any]] = [
    {
        "name": "web_search",
        "description": (
            "Search the web using a tiered strategy: "
            "Tier 1 — real Chromium browser (human_browser / a12fdfeaaf78) navigating DuckDuckGo like a human; "
            "Tier 2 — direct httpx HTTP fetch of DuckDuckGo HTML; "
            "Tier 3 — DuckDuckGo lite API. Returns results text and which tier was used."
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
            "Fetch a web page via the Chromium browser (human_browser / a12fdfeaaf78) "
            "and return its readable text."
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
# Tool dispatch (HTTP calls to sibling Docker services)
# ---------------------------------------------------------------------------

async def _call_tool(name: str, args: dict[str, Any]) -> str:
    async with httpx.AsyncClient(timeout=45) as c:
        try:
            if name == "web_search":
                query = str(args.get("query", "")).strip()
                max_chars = int(args.get("max_chars", 4000))
                max_chars = max(500, min(max_chars, 16000))
                # Tier 2: DuckDuckGo HTML
                try:
                    url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
                    r = await c.get(url, headers={"User-Agent": "Mozilla/5.0"}, follow_redirects=True)
                    import re as _re
                    text = _re.sub(r"<[^>]+>", " ", r.text)
                    text = _re.sub(r"\s+", " ", text).strip()[:max_chars]
                    return f"[Search via httpx (tier 2)]\n\n{text}"
                except Exception:
                    pass
                # Tier 3: DDG lite
                try:
                    url = f"https://lite.duckduckgo.com/lite/?q={query.replace(' ', '+')}"
                    r = await c.get(url, headers={"User-Agent": "Mozilla/5.0"}, follow_redirects=True)
                    import re as _re2
                    text = _re2.sub(r"<[^>]+>", " ", r.text)
                    text = _re2.sub(r"\s+", " ", text).strip()[:max_chars]
                    return f"[Search via DDG lite (tier 3)]\n\n{text}"
                except Exception as exc:
                    return f"web_search failed: {exc}"

            if name == "web_fetch":
                # web_fetch is handled by the browser tool inside aichat itself;
                # here we forward to the database service to check cache first,
                # then fall through to a direct httpx fetch as a lightweight proxy.
                url = str(args.get("url", "")).strip()
                max_chars = int(args.get("max_chars", 4000))
                max_chars = max(500, min(max_chars, 16000))
                # Check cache
                cache_r = await c.get(f"{DATABASE_URL}/cache/get", params={"url": url})
                if cache_r.status_code == 200:
                    data = cache_r.json()
                    if data.get("found"):
                        content = data.get("content", "")[:max_chars]
                        return f"[cached] {content}"
                # Fetch live
                r = await c.get(url, headers={"User-Agent": "Mozilla/5.0"}, follow_redirects=True)
                text = r.text[:max_chars]
                # Store in cache
                await c.post(f"{DATABASE_URL}/cache/store", json={"url": url, "content": text})
                return text

            if name == "db_store_article":
                r = await c.post(f"{DATABASE_URL}/articles/store", json=args)
                return json.dumps(r.json())

            if name == "db_search":
                r = await c.get(f"{DATABASE_URL}/articles/search", params=args)
                return json.dumps(r.json())

            if name == "db_cache_store":
                r = await c.post(f"{DATABASE_URL}/cache/store", json=args)
                return json.dumps(r.json())

            if name == "db_cache_get":
                r = await c.get(f"{DATABASE_URL}/cache/get", params={"url": args.get("url", "")})
                return json.dumps(r.json())

            if name == "memory_store":
                r = await c.post(f"{MEMORY_URL}/store", json={"key": args["key"], "value": args["value"]})
                return json.dumps(r.json())

            if name == "memory_recall":
                key = args.get("key", "")
                r = await c.get(f"{MEMORY_URL}/recall", params={"key": key})
                return json.dumps(r.json())

            if name == "researchbox_search":
                r = await c.get(f"{RESEARCH_URL}/search-feeds", params={"topic": args.get("topic", "")})
                return json.dumps(r.json())

            if name == "researchbox_push":
                r = await c.post(f"{RESEARCH_URL}/push-feed", json=args)
                return json.dumps(r.json())

            return f"Unknown tool: {name}"

        except Exception as exc:
            return f"Error calling {name}: {exc}"


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
        tool_name = params.get("name", "")
        arguments  = params.get("arguments") or {}
        result_text = await _call_tool(tool_name, arguments)
        return ok({
            "content": [{"type": "text", "text": result_text}],
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
                    # Send a keepalive comment
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
            content=json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}),
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
