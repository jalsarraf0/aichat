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
import random
import re
import uuid
from datetime import datetime
from typing import Any
from urllib.parse import unquote as _url_unquote

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

try:
    from PIL import Image as _PilImage
    import io as _io
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

app = FastAPI(title="aichat-mcp")

# Allow all origins so LM Studio (Electron/WebView2) can connect without CORS issues.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Base URLs for aichat backend services (running on the same Docker network).
DATABASE_URL = os.environ.get("DATABASE_URL", "http://aichat-database:8091")
MEMORY_URL    = os.environ.get("MEMORY_URL",   "http://aichat-memory:8094")
RESEARCH_URL  = os.environ.get("RESEARCH_URL", "http://aichat-researchbox:8092")
TOOLKIT_URL   = os.environ.get("TOOLKIT_URL",  "http://aichat-toolkit:8095")
# human_browser browser-server API — reachable after install connects it to this network.
BROWSER_URL   = os.environ.get("BROWSER_URL",  "http://human_browser:7081")
# Screenshot PNGs are bind-mounted from /docker/human_browser/workspace on the host.
BROWSER_WORKSPACE = os.environ.get("BROWSER_WORKSPACE", "/browser-workspace")

# Active SSE sessions: session_id -> asyncio.Queue
_sessions: dict[str, asyncio.Queue] = {}

# Realistic browser headers — used for all outbound httpx requests to reduce
# bot-detection and rate-limit exposure.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

# ---------------------------------------------------------------------------
# Tool schemas exposed to MCP clients
# ---------------------------------------------------------------------------

_TOOLS: list[dict[str, Any]] = [
    {
        "name": "screenshot",
        "description": (
            "Take a screenshot of a specific URL using the real Chromium browser and return it "
            "as an inline image rendered directly in the chat. "
            "The screenshot is also saved to the PostgreSQL image registry. "
            "IMPORTANT: Always use the exact URL from the user's request or from a prior "
            "web_search result. Never substitute a placeholder like example.com. "
            "If the user asks to screenshot a topic or site name rather than a full URL, "
            "call web_search first to find the correct URL, then call screenshot with that URL. "
            "Use 'find_text' to zoom in on a specific section — the browser will scroll to the "
            "first occurrence of that text and clip the screenshot to show just that region."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": (
                        "The exact URL to screenshot (must start with http:// or https://). "
                        "Do NOT use example.com or any placeholder — use the real URL from "
                        "the user's message or from a web_search result."
                    ),
                },
                "find_text": {
                    "type": "string",
                    "description": (
                        "Optional. A word or phrase to search for on the page. "
                        "The screenshot will be clipped to the section containing this text, "
                        "zooming in on the relevant content instead of the full viewport."
                    ),
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "fetch_image",
        "description": (
            "Download an image directly from a URL (jpg, png, gif, webp, etc.) and return it "
            "as an inline rendered image. Also saves metadata to the PostgreSQL image registry. "
            "Use this when the user provides a direct image URL and wants to view or save it, "
            "rather than screenshotting a full web page."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Direct URL to the image file (http/https)."},
            },
            "required": ["url"],
        },
    },
    {
        "name": "screenshot_search",
        "description": (
            "Search the web for a topic, screenshot the top result pages, and return all "
            "screenshots inline as rendered images. Also saves each to the PostgreSQL image "
            "registry. Use this when the user asks to 'find a picture of X', 'show me X', "
            "or wants to visually browse search results for a query. "
            "Makes a best-effort: returns whatever screenshots succeed even if some pages fail."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Topic or query to search and screenshot."},
                "max_results": {
                    "type": "integer",
                    "description": "Number of result pages to screenshot (default 3, max 3).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the web and return result text. Use this to find the correct URL for a "
            "site or topic before calling screenshot. "
            "Tier 2 — direct httpx fetch of DuckDuckGo HTML; "
            "Tier 3 — DuckDuckGo lite API fallback."
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
    {
        "name": "browser",
        "description": (
            "Control a real Chromium browser running in the human_browser Docker container "
            "via Playwright. The browser keeps its state between calls (same live session), "
            "so you can navigate to a page, click a button, fill a form, then read the result. "
            "Actions: "
            "navigate — go to a URL and return page title + readable text; "
            "read — return the current page title + text without navigating; "
            "click — click a CSS selector and return updated content; "
            "fill — type text into a CSS selector input; "
            "eval — run a JavaScript expression and return its result."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["navigate", "read", "click", "fill", "eval"],
                    "description": "Which browser action to perform.",
                },
                "url": {
                    "type": "string",
                    "description": "URL to navigate to (navigate action only).",
                },
                "selector": {
                    "type": "string",
                    "description": "CSS selector for click / fill actions.",
                },
                "value": {
                    "type": "string",
                    "description": "Text to type into the element (fill action only).",
                },
                "code": {
                    "type": "string",
                    "description": "JavaScript expression to evaluate (eval action only).",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "create_tool",
        "description": (
            "Create a new persistent custom tool that runs inside the aichat-toolkit Docker "
            "sandbox. The tool is saved to disk and immediately available — in this session "
            "and all future sessions. Use this when you need a capability not covered by "
            "built-in tools. Tools can make HTTP calls (httpx), process data, call APIs, "
            "run shell commands (subprocess/asyncio.create_subprocess_exec), and read files "
            "from user repos at /data/repos."
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
                    "description": "What the tool does — shown when deciding which tool to call.",
                },
                "parameters": {
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
                        "subprocess, httpx. Access parameters via kwargs. Must return a string."
                    ),
                },
            },
            "required": ["tool_name", "description", "code"],
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
        "name": "call_custom_tool",
        "description": (
            "Call a previously created custom tool by name. "
            "Use list_custom_tools first to see available tool names and their parameter schemas."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": "Name of the custom tool to call.",
                },
                "params": {
                    "type": "object",
                    "description": "Parameters to pass to the tool (must match its schema).",
                },
            },
            "required": ["tool_name"],
        },
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text(s: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": s}]


def _image_blocks(container_path: str, summary: str) -> list[dict[str, Any]]:
    """Return MCP content blocks: text summary + inline base64 image (compressed for LM Studio)."""
    blocks: list[dict[str, Any]] = [{"type": "text", "text": summary}]
    if not container_path:
        return blocks
    # container_path is e.g. /workspace/screenshot.png inside human_browser.
    # The workspace is bind-mounted at BROWSER_WORKSPACE inside this container.
    filename = os.path.basename(container_path)
    local_path = os.path.join(BROWSER_WORKSPACE, filename)
    if not os.path.isfile(local_path):
        return blocks
    try:
        if _HAS_PIL:
            # Resize to max 1280×1024 and compress as JPEG — large PNGs can exceed
            # LM Studio's MCP message size limit and fail to render.
            with _PilImage.open(local_path) as img:
                img = img.convert("RGB")  # strip alpha channel (JPEG doesn't support it)
                if img.width > 1280 or img.height > 1024:
                    img.thumbnail((1280, 1024), _PilImage.LANCZOS)
                buf = _io.BytesIO()
                img.save(buf, format="JPEG", quality=82)
            b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
            mime = "image/jpeg"
        else:
            with open(local_path, "rb") as fh:
                b64 = base64.standard_b64encode(fh.read()).decode("ascii")
            mime = "image/png"
        blocks.append({"type": "image", "data": b64, "mimeType": mime})
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
                find_text = str(args.get("find_text", "")).strip() or None
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                container_path = f"/workspace/screenshot_{ts}.png"
                shot_req: dict = {"url": url, "path": container_path}
                if find_text:
                    shot_req["find_text"] = find_text
                try:
                    r = await c.post(f"{BROWSER_URL}/screenshot",
                                     json=shot_req, timeout=20)
                    data = r.json()
                except Exception as exc:
                    return _text(f"Screenshot failed (browser unreachable): {exc}")
                error = data.get("error", "")
                image_urls = data.get("image_urls", [])
                container_path = data.get("path", container_path)
                filename = os.path.basename(container_path)
                host_path = f"/docker/human_browser/workspace/{filename}"
                local_path = os.path.join(BROWSER_WORKSPACE, filename)
                page_title = data.get("title", "") or url
                clipped = data.get("clipped", False)
                clip_note = f"\nZoomed to: '{find_text}'" if clipped and find_text else ""
                summary = (
                    f"Screenshot of: {page_title}\n"
                    f"URL: {url}{clip_note}\n"
                    f"File: {host_path}"
                )
                # Happy path — screenshot file was written
                if os.path.isfile(local_path):
                    try:
                        await c.post(f"{DATABASE_URL}/images/store", json={
                            "url": url,
                            "host_path": host_path,
                            "alt_text": f"Screenshot of {page_title}",
                        })
                    except Exception:
                        pass
                    return _image_blocks(container_path, summary)
                # Screenshot file missing — browser was blocked or crashed.
                # Try fetching a real image from the page DOM (browser v2+ returns image_urls).
                img_hdrs = {
                    "User-Agent": _BROWSER_HEADERS["User-Agent"],
                    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
                }
                for img_url in image_urls[:3]:
                    try:
                        ir = await c.get(img_url, headers=img_hdrs,
                                         follow_redirects=True, timeout=15)
                        if ir.status_code == 200:
                            ct = ir.headers.get("content-type", "image/jpeg").split(";")[0].strip()
                            b64 = base64.standard_b64encode(ir.content).decode("ascii")
                            fallback_summary = (
                                f"Screenshot of: {page_title}\n"
                                f"URL: {url}\n"
                                f"(screenshot blocked — showing page image)"
                            )
                            return [
                                {"type": "text", "text": fallback_summary},
                                {"type": "image", "data": b64, "mimeType": ct},
                            ]
                    except Exception:
                        continue
                return _text(f"Screenshot failed: {error or 'unknown error'}. URL: {url}")

            # ----------------------------------------------------------------
            if name == "fetch_image":
                url = str(args.get("url", "")).strip()
                if not url:
                    return _text("fetch_image: 'url' is required")
                img_fetch_headers = {
                    "User-Agent": _BROWSER_HEADERS["User-Agent"],
                    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
                    "Accept-Language": _BROWSER_HEADERS["Accept-Language"],
                    "Accept-Encoding": _BROWSER_HEADERS["Accept-Encoding"],
                    "DNT": "1",
                }
                last_exc: Exception | None = None
                content_type = "image/jpeg"
                img_data = b""
                for attempt in range(2):
                    try:
                        r = await c.get(url, headers=img_fetch_headers, follow_redirects=True)
                        if r.status_code == 429 and attempt == 0:
                            retry_after = min(int(r.headers.get("retry-after", "15")), 30)
                            await asyncio.sleep(retry_after)
                            continue
                        r.raise_for_status()
                        content_type = r.headers.get("content-type", "image/jpeg").split(";")[0].strip()
                        img_data = r.content
                        break
                    except Exception as exc:
                        last_exc = exc
                        if attempt == 0:
                            await asyncio.sleep(3)
                            continue
                else:
                    return _text(f"fetch_image failed: {last_exc}")
                # Derive host_path for DB metadata (workspace is writable via host bind-mount)
                raw_name = url.split("?")[0].split("/")[-1] or "image"
                if "." not in raw_name:
                    ext = {"image/jpeg": ".jpg", "image/png": ".png",
                           "image/gif": ".gif", "image/webp": ".webp"}.get(content_type, ".jpg")
                    raw_name += ext
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"img_{ts}_{raw_name}"
                host_path = f"/docker/human_browser/workspace/{filename}"
                # Save metadata to DB
                try:
                    await c.post(f"{DATABASE_URL}/images/store", json={
                        "url": url,
                        "host_path": host_path,
                        "alt_text": f"Image from {url}",
                    })
                except Exception:
                    pass
                # Return inline base64 image
                b64 = base64.standard_b64encode(img_data).decode("ascii")
                summary = (
                    f"Image from: {url}\n"
                    f"Type: {content_type}  Size: {len(img_data):,} bytes\n"
                    f"File: {host_path}"
                )
                return [
                    {"type": "text", "text": summary},
                    {"type": "image", "data": b64, "mimeType": content_type},
                ]

            # ----------------------------------------------------------------
            if name == "screenshot_search":
                query = str(args.get("query", "")).strip()
                if not query:
                    return _text("screenshot_search: 'query' is required")
                max_results = max(1, min(int(args.get("max_results", 3)), 3))

                # Search DuckDuckGo HTML for result URLs (realistic headers)
                try:
                    r = await c.get(
                        f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}",
                        headers=_BROWSER_HEADERS,
                        follow_redirects=True,
                    )
                    html = r.text
                except Exception as exc:
                    return _text(f"Search failed: {exc}")

                # Tier 1: DDG uddg= redirect params (stable HTML endpoint format)
                _DDG_HOSTS = ('duckduckgo.com', 'ddg.gg', 'duck.co')
                raw = re.findall(r'uddg=(https?%3A[^&"\'>\s]+)', html)
                seen_u: set[str] = set()
                urls: list[str] = []
                for encoded in raw:
                    decoded = _url_unquote(encoded)
                    # Skip DDG-internal URLs that get double-encoded into uddg=
                    if any(d in decoded for d in _DDG_HOSTS):
                        continue
                    if decoded not in seen_u:
                        seen_u.add(decoded)
                        urls.append(decoded)
                urls = urls[:max_results]

                # Tier 2: direct href links (fallback if DDG changed format or rate-limited)
                if not urls:
                    href_raw = re.findall(r'href=["\']?(https?://[^"\'>\s]+)', html)
                    urls = list(dict.fromkeys(
                        u for u in href_raw
                        if not any(d in u for d in _DDG_HOSTS)
                    ))[:max_results]

                # Tier 3: browser search + DOM eval (Chromium w/ anti-detection, most reliable)
                if not urls:
                    try:
                        await asyncio.wait_for(
                            c.post(f"{BROWSER_URL}/search", json={"query": query}),
                            timeout=35.0,
                        )
                        ev = await c.post(f"{BROWSER_URL}/eval", json={"code": r"""
                            JSON.stringify(
                                Array.from(document.links)
                                    .map(a => {
                                        try {
                                            const u = new URL(a.href);
                                            if (u.hostname === 'duckduckgo.com' && u.pathname === '/l/')
                                                return u.searchParams.get('uddg') || null;
                                            if (u.hostname !== 'duckduckgo.com' && u.hostname !== 'duck.co')
                                                return a.href;
                                            return null;
                                        } catch(e) { return null; }
                                    })
                                    .filter(u => u && u.startsWith('http'))
                                    .filter((u, i, arr) => arr.indexOf(u) === i)
                                    .slice(0, 5)
                            )
                        """}, timeout=10)
                        extracted = json.loads(ev.json().get("result", "[]"))
                        urls = [u for u in extracted if u][:max_results]
                    except Exception:
                        pass

                if not urls:
                    return _text(f"No URLs found in search results for: {query}")

                blocks: list[dict[str, Any]] = [
                    {"type": "text", "text": f"Visual search: '{query}' — screenshotting {len(urls)} result(s)...\n"}
                ]
                img_hdrs = {
                    "User-Agent": _BROWSER_HEADERS["User-Agent"],
                    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
                }
                # 24-second budget for all screenshots — fits within LM Studio's timeout.
                _deadline = asyncio.get_event_loop().time() + 24.0
                for i, url in enumerate(urls):
                    remaining = _deadline - asyncio.get_event_loop().time()
                    if remaining < 3:
                        blocks.append({"type": "text", "text": f"(time budget reached — stopped at {i} of {len(urls)} results)"})
                        break
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{i}"
                    cp = f"/workspace/screenshot_{ts}.png"
                    try:
                        sr = await c.post(f"{BROWSER_URL}/screenshot",
                                          json={"url": url, "path": cp},
                                          timeout=min(15.0, remaining - 2.0))
                        data = sr.json()
                    except Exception as exc:
                        blocks.append({"type": "text", "text": f"Failed to screenshot {url}: {exc}"})
                        continue
                    err = data.get("error", "")
                    s_image_urls = data.get("image_urls", [])
                    container_path = data.get("path", cp)
                    filename = os.path.basename(container_path)
                    host_path = f"/docker/human_browser/workspace/{filename}"
                    local_path = os.path.join(BROWSER_WORKSPACE, filename)
                    page_title = data.get("title", "") or url
                    summary = f"{page_title}\n{url}\nFile: {host_path}"
                    if os.path.isfile(local_path):
                        try:
                            await c.post(f"{DATABASE_URL}/images/store", json={
                                "url": url,
                                "host_path": host_path,
                                "alt_text": f"Search: '{query}' — {page_title}",
                            })
                        except Exception:
                            pass
                        blocks.extend(_image_blocks(container_path, summary))
                    else:
                        # Screenshot failed — try image_urls fallback
                        fetched = False
                        for img_url in s_image_urls[:3]:
                            try:
                                ir = await c.get(img_url, headers=img_hdrs,
                                                 follow_redirects=True, timeout=15)
                                if ir.status_code == 200:
                                    ct = ir.headers.get("content-type", "image/jpeg").split(";")[0].strip()
                                    b64 = base64.standard_b64encode(ir.content).decode("ascii")
                                    fb_summary = f"{page_title}\n{url}\n(screenshot blocked — showing page image)"
                                    blocks.extend([
                                        {"type": "text", "text": fb_summary},
                                        {"type": "image", "data": b64, "mimeType": ct},
                                    ])
                                    fetched = True
                                    break
                            except Exception:
                                continue
                        if not fetched:
                            blocks.append({"type": "text", "text": f"Failed: {url} — {err or 'screenshot unavailable'}"})
                return blocks

            # ----------------------------------------------------------------
            if name == "web_search":
                query = str(args.get("query", "")).strip()
                max_chars = int(args.get("max_chars", 4000))
                max_chars = max(500, min(max_chars, 16000))
                # Tier 1: DuckDuckGo HTML via httpx (fast, reliable, bot-resilient with brotlicffi)
                try:
                    r = await c.get(
                        f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}",
                        headers=_BROWSER_HEADERS,
                        follow_redirects=True,
                    )
                    text = re.sub(r"<[^>]+>", " ", r.text)
                    text = re.sub(r"\s+", " ", text).strip()[:max_chars]
                    return _text(f"[Search results]\n\n{text}")
                except Exception:
                    pass
                # Tier 3: DDG lite (realistic browser headers)
                try:
                    r = await c.get(
                        f"https://lite.duckduckgo.com/lite/?q={query.replace(' ', '+')}",
                        headers=_BROWSER_HEADERS,
                        follow_redirects=True,
                    )
                    text = re.sub(r"<[^>]+>", " ", r.text)
                    text = re.sub(r"\s+", " ", text).strip()[:max_chars]
                    return _text(f"[Search results (lite)]\n\n{text}")
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
                            cached_text = data.get("content", "")
                            # Re-cache items may contain raw HTML from old behavior — strip if needed.
                            # Use `>?` so truncated tags (no closing >) are also removed.
                            if cached_text.lstrip().startswith("<"):
                                cached_text = re.sub(r"<[^>]*>?", " ", cached_text)
                                cached_text = re.sub(r"\s+", " ", cached_text).strip()
                            if len(cached_text) > 50:
                                return _text(f"[cached] {cached_text[:max_chars]}")
                            # Too short (stripped HTML left only a title/nav) — fall through to live fetch
                except Exception:
                    pass
                # Fetch via browser (renders JS, returns clean text, handles SSL)
                text = ""
                try:
                    nav_r = await asyncio.wait_for(
                        c.post(f"{BROWSER_URL}/navigate", json={"url": url}),
                        timeout=20.0,
                    )
                    nav_data = nav_r.json()
                    text = nav_data.get("content", "")
                    if text:
                        title = nav_data.get("title", "")
                        final_url = nav_data.get("url", url)
                        header = f"Title: {title}\nURL: {final_url}\n\n" if title else ""
                        text = (header + text)[:max_chars]
                except Exception:
                    pass
                # Fallback: httpx + strip tags
                if not text:
                    try:
                        r = await c.get(url, headers=_BROWSER_HEADERS, follow_redirects=True)
                        raw = r.text
                        text = re.sub(r"<[^>]+>", " ", raw)
                        text = re.sub(r"\s+", " ", text).strip()[:max_chars]
                    except Exception as exc:
                        return _text(f"web_fetch failed: {exc}")
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
                # Inline the most recent image — derive container path from host_path basename
                hp0 = images[0].get("host_path", "") or ""
                most_recent = f"/workspace/{os.path.basename(hp0)}" if hp0 else ""
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

            # ----------------------------------------------------------------
            if name == "browser":
                action = str(args.get("action", "")).strip()
                if not action:
                    return _text("browser: 'action' is required")
                if action == "navigate":
                    url = str(args.get("url", "")).strip()
                    if not url:
                        return _text("browser navigate: 'url' is required")
                    try:
                        nav_r = await asyncio.wait_for(
                            c.post(f"{BROWSER_URL}/navigate", json={"url": url}),
                            timeout=20.0,
                        )
                        data = nav_r.json()
                        content = data.get("content", "")
                        title = data.get("title", "")
                        final_url = data.get("url", url)
                        header = f"Title: {title}\nURL: {final_url}\n\n" if title else ""
                        return _text((header + content)[:8000])
                    except Exception as exc:
                        return _text(f"browser navigate failed: {exc}")
                if action == "read":
                    try:
                        read_r = await c.get(f"{BROWSER_URL}/read", timeout=10.0)
                        data = read_r.json()
                        content = data.get("content", "")
                        title = data.get("title", "")
                        header = f"Title: {title}\n\n" if title else ""
                        return _text((header + content)[:8000])
                    except Exception as exc:
                        return _text(f"browser read failed: {exc}")
                if action == "click":
                    selector = str(args.get("selector", "")).strip()
                    if not selector:
                        return _text("browser click: 'selector' is required")
                    try:
                        click_r = await c.post(
                            f"{BROWSER_URL}/click", json={"selector": selector}, timeout=10.0
                        )
                        data = click_r.json()
                        return _text(data.get("content", "Clicked."))
                    except Exception as exc:
                        return _text(f"browser click failed: {exc}")
                if action == "fill":
                    selector = str(args.get("selector", "")).strip()
                    value = str(args.get("value", ""))
                    if not selector:
                        return _text("browser fill: 'selector' is required")
                    try:
                        fill_r = await c.post(
                            f"{BROWSER_URL}/fill",
                            json={"selector": selector, "value": value},
                            timeout=10.0,
                        )
                        data = fill_r.json()
                        return _text(data.get("content", "Filled."))
                    except Exception as exc:
                        return _text(f"browser fill failed: {exc}")
                if action == "eval":
                    code = str(args.get("code", "")).strip()
                    if not code:
                        return _text("browser eval: 'code' is required")
                    try:
                        eval_r = await c.post(
                            f"{BROWSER_URL}/eval", json={"code": code}, timeout=10.0
                        )
                        data = eval_r.json()
                        return _text(str(data.get("result", "")))
                    except Exception as exc:
                        return _text(f"browser eval failed: {exc}")
                return _text(f"browser: unknown action '{action}'")

            # ----------------------------------------------------------------
            if name == "create_tool":
                tool_name = str(args.get("tool_name", "")).strip()
                description = str(args.get("description", "")).strip()
                parameters = args.get("parameters", {})
                code = str(args.get("code", "")).strip()
                if not tool_name or not description or not code:
                    return _text("create_tool: 'tool_name', 'description', and 'code' are required")
                r = await c.post(
                    f"{TOOLKIT_URL}/register",
                    json={
                        "tool_name": tool_name,
                        "description": description,
                        "parameters": parameters,
                        "code": code,
                    },
                    timeout=10.0,
                )
                return _text(json.dumps(r.json()))

            if name == "list_custom_tools":
                r = await c.get(f"{TOOLKIT_URL}/tools", timeout=10.0)
                return _text(json.dumps(r.json()))

            if name == "delete_custom_tool":
                tool_name = str(args.get("tool_name", "")).strip()
                if not tool_name:
                    return _text("delete_custom_tool: 'tool_name' is required")
                r = await c.delete(f"{TOOLKIT_URL}/tool/{tool_name}", timeout=10.0)
                return _text(json.dumps(r.json()))

            if name == "call_custom_tool":
                tool_name = str(args.get("tool_name", "")).strip()
                params = args.get("params", {})
                if not tool_name:
                    return _text("call_custom_tool: 'tool_name' is required")
                r = await c.post(
                    f"{TOOLKIT_URL}/call/{tool_name}",
                    json={"params": params},
                    timeout=30.0,
                )
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


_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
    "Access-Control-Allow-Origin": "*",
}


# ---------------------------------------------------------------------------
# MCP 2025-03-26 Streamable HTTP transport  (preferred by LM Studio 0.3.6+)
# POST /mcp — single endpoint; responds with JSON or SSE depending on Accept
# GET  /mcp — SSE stream, same session handshake as /sse
# ---------------------------------------------------------------------------

@app.post("/mcp")
async def mcp_post(request: Request) -> Response:
    """
    Streamable HTTP transport (MCP 2025-03-26).
    Clients that prefer a single endpoint send JSON-RPC here.
    If the client sent Accept: text/event-stream we stream back; otherwise JSON.
    """
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

    accept = request.headers.get("accept", "")
    if "text/event-stream" in accept:
        # Non-blocking: stream keepalives while the tool runs, yield result when done.
        async def _stream_result(rpc: dict):
            task = asyncio.create_task(_handle_rpc(rpc))
            while not task.done():
                yield ": keepalive\n\n"
                await asyncio.sleep(5.0)
            result = task.result()
            if result is not None:
                yield f"event: message\ndata: {json.dumps(result)}\n\n"
        return StreamingResponse(_stream_result(rpc), media_type="text/event-stream",
                                 headers=_SSE_HEADERS)

    # JSON (non-SSE) path — synchronous; used only by non-LM-Studio clients.
    response = await _handle_rpc(rpc)
    if response is None:
        return Response(content="", status_code=202)
    return Response(content=json.dumps(response), media_type="application/json")


@app.get("/mcp")
async def mcp_sse(request: Request) -> StreamingResponse:
    """GET /mcp — SSE stream, same as GET /sse (alias for the Streamable HTTP spec)."""
    return await sse_connect(request)


# ---------------------------------------------------------------------------
# Legacy SSE transport (MCP 2024-11-05) — kept for backward compatibility
# ---------------------------------------------------------------------------

@app.get("/sse")
async def sse_connect(request: Request) -> StreamingResponse:
    session_id = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    _sessions[session_id] = queue

    async def event_stream():
        yield f"event: endpoint\ndata: /messages?sessionId={session_id}\n\n"
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=5.0)
                    yield f"event: message\ndata: {json.dumps(msg)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            _sessions.pop(session_id, None)

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers=_SSE_HEADERS)


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

    async def _deliver(rpc: dict, sid: str) -> None:
        try:
            task = asyncio.create_task(_handle_rpc(rpc))
            req_id = rpc.get("id")
            # Send MCP progress notifications every 5 s while the tool runs.
            # These are real data: events on the SSE stream — keeps LM Studio's
            # internal tool-call timer alive even for slow tools (screenshot, etc.).
            while not task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
                except asyncio.TimeoutError:
                    if sid in _sessions and req_id is not None:
                        await _sessions[sid].put({
                            "jsonrpc": "2.0",
                            "method": "notifications/progress",
                            "params": {
                                "progressToken": str(req_id),
                                "progress": 0,
                                "total": 1,
                            },
                        })
            response = task.result()
            if response is not None and sid in _sessions:
                await _sessions[sid].put(response)
        except Exception:
            pass

    asyncio.create_task(_deliver(rpc, sessionId))
    return Response(content="", status_code=202)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "sessions": len(_sessions),
        "tools": len(_TOOLS),
        "transports": ["POST /mcp (streamable-http)", "GET /sse (sse)", "GET /mcp (sse-alias)"],
    }
