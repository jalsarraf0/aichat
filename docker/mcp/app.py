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
import collections
import json
import os
import random
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from urllib.parse import unquote as _url_unquote

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

try:
    from PIL import (
        Image as _PilImage,
        ImageEnhance as _ImageEnhance,
        ImageFilter as _ImageFilter,
        ImageChops as _ImageChops,
        ImageDraw as _ImageDraw,
        ImageOps as _ImageOps,
    )
    import io as _io
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    import cv2 as _cv2
    import numpy as _np
    _HAS_CV2 = True
except ImportError:
    _cv2 = None  # type: ignore[assignment]
    _np  = None  # type: ignore[assignment]
    _HAS_CV2 = False

import textwrap as _textwrap
import time as _time

# ---------------------------------------------------------------------------
# Perceptual hash helpers — pure PIL, no extra packages
# ---------------------------------------------------------------------------

def _dhash(img: "_PilImage.Image") -> str:
    """64-bit difference hash of a PIL Image → 16-char hex string."""
    if not _HAS_PIL:
        return ""
    try:
        gray = img.convert("L").resize((9, 8), _PilImage.LANCZOS)
        px   = list(gray.getdata())
        bits = sum(
            1 << i for i in range(64)
            if px[i % 8 + (i // 8) * 9] > px[i % 8 + (i // 8) * 9 + 1]
        )
        return f"{bits:016x}"
    except Exception:
        return ""


def _hamming(h1: str, h2: str) -> int:
    """Bit-level Hamming distance between two 16-hex-char dHashes."""
    if not h1 or not h2 or len(h1) != 16 or len(h2) != 16:
        return 64
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")


async def _vision_confirm(
    b64: str, subject: str, hc: "httpx.AsyncClient",
    phash: str = "",
) -> "tuple[bool, str, float]":
    """Ask the loaded multimodal model whether the image shows *subject*.

    Returns (is_match, description, confidence).  Fails open on any error so
    image_search never breaks when no vision model is loaded in LM Studio.
    Uses IMAGE_GEN_BASE_URL/v1/chat/completions with 8-second timeout.
    Set IMAGE_VISION_CONFIRM=false to disable (skip confirmation, keep all images).

    phash: if provided, VisionCache is checked first (skips GPU call on cache hit)
    and the result is stored in VisionCache after a live LM Studio call.
    ModelRegistry is used for a fast bail-out when no model is loaded.
    """
    env_flag = os.environ.get("IMAGE_VISION_CONFIRM", "true").lower()
    if env_flag not in ("1", "true", "yes"):
        return True, "", 0.6
    # 1. In-memory cache check — free, instant, no GPU call
    if phash:
        cached = _vision_cache.get(phash)
        if cached is not None:
            return cached
    # 2. ModelRegistry fast bail-out — avoids the full 8 s timeout when no model is loaded
    if not await ModelRegistry.get().is_available(hc):
        return True, "", 0.6   # fail-open (same as before)
    prompt = (
        f"Describe this image in one short sentence. "
        f"Then answer: does it clearly show '{subject}'? "
        f"Format: DESCRIPTION: <text> | MATCH: YES or NO"
    )
    payload: dict = {
        "messages": [{"role": "user", "content": [
            {"type": "text",      "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
        "max_tokens": 80,
        "temperature": 0.0,
    }
    model = IMAGE_GEN_MODEL.strip()
    if model:
        payload["model"] = model
    try:
        r = await asyncio.wait_for(
            hc.post(f"{IMAGE_GEN_BASE_URL}/v1/chat/completions", json=payload),
            timeout=8.0,
        )
        _choices = r.json().get("choices", [])
        text = (_choices[0]["message"]["content"].strip() if _choices else "")
        desc  = ""
        if "DESCRIPTION:" in text:
            desc = text.split("DESCRIPTION:")[1].split("|")[0].strip()
        match = "MATCH: YES" in text.upper() or text.upper().strip().endswith("YES")
        conf  = 0.9 if match else 0.2
        result: tuple[bool, str, float] = (match, desc, conf)
    except Exception:
        result = (True, "", 0.6)   # fail-open
    # 3. Store in VisionCache for future calls
    if phash:
        _vision_cache.put(phash, result)
    return result


app = FastAPI(title="aichat-mcp")

# Allow all origins so LM Studio (Electron/WebView2) can connect without CORS issues.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Base URLs for aichat backend services (running on the same Docker network).
# (exception handler is registered after _report_error is defined, below)
DATABASE_URL = os.environ.get("DATABASE_URL", "http://aichat-database:8091")
MEMORY_URL    = os.environ.get("MEMORY_URL",   "http://aichat-memory:8094")
RESEARCH_URL  = os.environ.get("RESEARCH_URL", "http://aichat-researchbox:8092")
TOOLKIT_URL   = os.environ.get("TOOLKIT_URL",  "http://aichat-toolkit:8095")
GRAPH_URL     = os.environ.get("GRAPH_URL",   "http://aichat-graph:8098")
VECTOR_URL    = os.environ.get("VECTOR_URL",  "http://aichat-vector:6333")
VIDEO_URL     = os.environ.get("VIDEO_URL",   "http://aichat-video:8099")
OCR_URL       = os.environ.get("OCR_URL",     "http://aichat-ocr:8100")
DOCS_URL      = os.environ.get("DOCS_URL",    "http://aichat-docs:8101")
PLANNER_URL   = os.environ.get("PLANNER_URL", "http://aichat-planner:8102")
# human_browser browser-server API — reachable after install connects it to this network.
BROWSER_URL   = os.environ.get("BROWSER_URL",  "http://human_browser:7081")
# Screenshot PNGs are bind-mounted from /docker/human_browser/workspace on the host.
BROWSER_WORKSPACE = os.environ.get("BROWSER_WORKSPACE", "/browser-workspace")
# Image generation — LM Studio OpenAI-compatible image API (or any compatible backend).
IMAGE_GEN_BASE_URL = os.environ.get("IMAGE_GEN_BASE_URL", "http://host.docker.internal:1234")
IMAGE_GEN_MODEL    = os.environ.get("IMAGE_GEN_MODEL", "")

# Max seconds a single tool call may run before it is cancelled.
_TOOL_TIMEOUT = 180.0

# Active SSE sessions: session_id -> asyncio.Queue
_sessions: dict[str, asyncio.Queue] = {}

# ---------------------------------------------------------------------------
# Logging + error reporting
# ---------------------------------------------------------------------------

import logging as _logging

_log = _logging.getLogger("aichat-mcp")
_logging.basicConfig(
    level=_logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_SERVICE_NAME = "aichat-mcp"


async def _report_error(message: str, detail: str | None = None) -> None:
    """Fire-and-forget: send an error entry to aichat-database."""
    try:
        async with httpx.AsyncClient(timeout=5) as _c:
            await _c.post(
                f"{DATABASE_URL}/errors/log",
                json={"service": _SERVICE_NAME, "level": "ERROR",
                      "message": message, "detail": detail},
            )
    except Exception:
        pass  # never let error reporting crash the MCP server


@app.exception_handler(Exception)
async def global_exception_handler(request: "Request", exc: Exception) -> "Response":
    from fastapi.responses import JSONResponse as _JSONResponse
    message = str(exc)
    detail = f"{request.method} {request.url.path}"
    _log.error("Unhandled error [%s %s]: %s", request.method, request.url.path, exc, exc_info=True)
    asyncio.create_task(_report_error(message, detail))
    return _JSONResponse(status_code=500, content={"error": message})


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
            "first occurrence of that text and clip the screenshot to show just that region. "
            "Use 'find_image' to precisely capture a specific <img> element on the page — match "
            "by src/alt substring (e.g. 'logo', 'hero') or 1-based index (e.g. '2', '#3')."
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
                "find_image": {
                    "type": "string",
                    "description": (
                        "Optional. Match an <img> element by src/alt substring or 1-based index "
                        "(e.g. 'logo', 'hero', '2', '#3'). The screenshot is tightly cropped "
                        "to that image element only. Mutually exclusive with find_text."
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
        "name": "browser_save_images",
        "description": (
            "Download specific image URLs using the real Chromium browser's live session — "
            "exactly like a human right-clicking 'Save Image As'. Because it uses the browser's "
            "cookies, auth tokens, and referrer headers it succeeds on auth-gated images, "
            "CDN-protected images, session-bound content, and lazy-loaded images that a plain "
            "HTTP download would fail on. "
            "Pass urls as a JSON array of image URLs (or a comma-separated string). "
            "Files are saved to the browser workspace and returned as inline images."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "urls": {
                    "description": (
                        "Image URLs to download. Accepts a JSON array of strings, or a single "
                        "comma-separated string. Get these from browser list_images_detail or "
                        "from page source."
                    ),
                },
                "prefix": {
                    "type": "string",
                    "description": "Filename prefix for saved files (default 'image').",
                },
                "max": {
                    "type": "integer",
                    "description": "Maximum number of images to download (default 20, max 50).",
                },
            },
            "required": ["urls"],
        },
    },
    {
        "name": "browser_download_page_images",
        "description": (
            "Navigate to a URL (or use the current browser page) and download all visible "
            "<img> elements using the browser's live session — the same as a human saving "
            "every image on a page. Optionally filter by a substring in the image src or alt. "
            "Uses cookies, auth headers, and referrer so auth-gated or CDN-protected images "
            "are downloaded successfully. "
            "Returns saved file paths and inline previews of the first few images."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": (
                        "Optional. Navigate to this URL first, then download images. "
                        "If omitted, downloads images from the current browser page."
                    ),
                },
                "filter": {
                    "type": "string",
                    "description": (
                        "Optional substring to match against image src or alt text. "
                        "Only images containing this string are downloaded. "
                        "Examples: 'product', 'hero', '.jpg', 'thumbnail'."
                    ),
                },
                "prefix": {
                    "type": "string",
                    "description": "Filename prefix for saved files (default 'image').",
                },
                "max": {
                    "type": "integer",
                    "description": "Maximum number of images to download (default 20, max 50).",
                },
            },
            "required": [],
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
                    "description": "Number of result pages to screenshot (default 3, max 5).",
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
                "topic":        {"type": "string"},
                "q":            {"type": "string"},
                "limit":        {"type": "integer", "description": "Max results (default 20)."},
                "offset":       {"type": "integer", "description": "Skip first N results for pagination (default 0)."},
                "summary_only": {"type": "boolean", "description": "Truncate content to 300 chars for compact results (default false)."},
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
                "key":         {"type": "string"},
                "value":       {"type": "string"},
                "ttl_seconds": {
                    "type": "integer",
                    "description": "Optional. Entry expires after this many seconds (omit for permanent).",
                },
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
                "key":     {"type": "string", "description": "Exact key to look up (optional)."},
                "pattern": {
                    "type": "string",
                    "description": (
                        "Optional SQL LIKE pattern to match multiple keys "
                        "(e.g. 'whatsapp:%'). Used when key is empty."
                    ),
                },
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
            "eval — run a JavaScript expression and return its result; "
            "screenshot_element — take a precise screenshot cropped to a CSS selector element; "
            "list_images_detail — list all <img> elements with src, alt, dimensions, and viewport info."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["navigate", "read", "click", "fill", "eval",
                             "screenshot_element", "list_images_detail"],
                    "description": "Which browser action to perform.",
                },
                "url": {
                    "type": "string",
                    "description": "URL to navigate to (navigate action only).",
                },
                "selector": {
                    "type": "string",
                    "description": (
                        "CSS selector for click / fill / screenshot_element actions. "
                        "For screenshot_element, use any valid CSS selector, e.g. 'img.hero', "
                        "'#logo', 'article img:first-child'."
                    ),
                },
                "value": {
                    "type": "string",
                    "description": "Text to type into the element (fill action only).",
                },
                "code": {
                    "type": "string",
                    "description": "JavaScript expression to evaluate (eval action only).",
                },
                "pad": {
                    "type": "integer",
                    "description": (
                        "Padding in pixels around the element for screenshot_element (default 20)."
                    ),
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
    {
        "name": "get_errors",
        "description": (
            "Query the structured error log stored in PostgreSQL. "
            "Returns recent application errors from all aichat services "
            "(aichat-mcp, aichat-memory, aichat-toolkit, aichat-researchbox, etc.). "
            "Use this to diagnose failures, check service health history, or audit errors."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of error entries to return (default 50, max 200).",
                },
                "service": {
                    "type": "string",
                    "description": "Filter by service name (e.g. 'aichat-memory'). Omit to see all services.",
                },
            },
            "required": [],
        },
    },
    # ── Image manipulation tools ────────────────────────────────────────────
    {
        "name": "image_crop",
        "description": (
            "Crop a saved screenshot or image to a specific pixel region and return it inline. "
            "Use this after screenshot to isolate a panel, chart, button, or any rectangular area. "
            "Step 1 in the zoom-scan pipeline: crop → zoom → scan."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Filename (e.g. 'screenshot_20260224_120000.png'), "
                        "container path (/workspace/…), or host path (/docker/human_browser/workspace/…)."
                    ),
                },
                "left":   {"type": "integer", "description": "Left edge in pixels (default 0)."},
                "top":    {"type": "integer", "description": "Top edge in pixels (default 0)."},
                "right":  {"type": "integer", "description": "Right edge in pixels (default: image width)."},
                "bottom": {"type": "integer", "description": "Bottom edge in pixels (default: image height)."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "image_zoom",
        "description": (
            "Zoom into a region of a saved screenshot by a scale factor and return it inline. "
            "Optionally crop first (left/top/right/bottom) then scale. Scale 2.0 = 200%. "
            "Step 2 in the zoom-scan pipeline: crop → zoom → scan."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path":   {"type": "string", "description": "Image filename or path (same formats as image_crop)."},
                "scale":  {"type": "number",  "description": "Zoom factor: 2.0 = 200%, max 8.0 (default 2.0)."},
                "left":   {"type": "integer", "description": "Crop left before zooming (default 0)."},
                "top":    {"type": "integer", "description": "Crop top before zooming (default 0)."},
                "right":  {"type": "integer", "description": "Crop right before zooming (default: image width)."},
                "bottom": {"type": "integer", "description": "Crop bottom before zooming (default: image height)."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "image_scan",
        "description": (
            "Prepare a screenshot region for text reading by the vision model. "
            "Converts to greyscale, boosts contrast 2.5×, sharpens 3×, scales up small regions, "
            "and returns the enhanced image inline so the model can read any text in it. "
            "Step 3 in the zoom-scan pipeline: crop → zoom → scan."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path":   {"type": "string", "description": "Image filename or path (same formats as image_crop)."},
                "left":   {"type": "integer", "description": "Scan region left (default 0 = whole image)."},
                "top":    {"type": "integer", "description": "Scan region top (default 0)."},
                "right":  {"type": "integer", "description": "Scan region right (default: image width)."},
                "bottom": {"type": "integer", "description": "Scan region bottom (default: image height)."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "image_enhance",
        "description": (
            "Apply visual enhancements to a saved screenshot — adjust contrast, sharpness, and brightness, "
            "or convert to greyscale. Returns the enhanced image inline. "
            "Useful for making dark screenshots, blurry text, or low-contrast charts easier to read."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path":       {"type": "string",  "description": "Image filename or path."},
                "contrast":   {"type": "number",  "description": "Contrast multiplier 0.5–4.0 (default 1.5)."},
                "sharpness":  {"type": "number",  "description": "Sharpness multiplier 0.5–4.0 (default 1.5)."},
                "brightness": {"type": "number",  "description": "Brightness multiplier 0.5–3.0 (default 1.0)."},
                "grayscale":  {"type": "boolean", "description": "Convert to greyscale before enhancing (default false)."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "image_stitch",
        "description": (
            "Combine two or more saved screenshots side-by-side (horizontal) or stacked (vertical). "
            "Useful for comparing before/after states, merging multi-panel pages, or building "
            "a composite view of several cropped regions."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of image filenames or paths to combine (in order, 2–8 images).",
                },
                "direction": {
                    "type": "string",
                    "enum": ["horizontal", "vertical"],
                    "description": "Stack images left-to-right ('horizontal') or top-to-bottom ('vertical'). Default: vertical.",
                },
                "gap": {
                    "type": "integer",
                    "description": "Pixel gap between images (default 0).",
                },
            },
            "required": ["paths"],
        },
    },
    {
        "name": "image_diff",
        "description": (
            "Show a pixel-level visual diff between two screenshots. "
            "Changed pixels are highlighted in red on a white background so it is easy to spot "
            "what changed between two page states. Useful after interactions (click, scroll, fill)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path_a": {"type": "string", "description": "First image (before state) — filename or path."},
                "path_b": {"type": "string", "description": "Second image (after state) — filename or path."},
                "amplify": {
                    "type": "number",
                    "description": "Multiply pixel differences by this factor so subtle changes are visible (default 3.0).",
                },
            },
            "required": ["path_a", "path_b"],
        },
    },
    {
        "name": "image_annotate",
        "description": (
            "Draw bounding boxes and labels on a screenshot to highlight regions of interest. "
            "Useful for marking UI elements, errors, or areas to fix before sending to the user."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Image filename or path to annotate."},
                "boxes": {
                    "type": "array",
                    "description": (
                        "List of bounding box objects: each has 'left', 'top', 'right', 'bottom' (pixels), "
                        "optional 'label' (string), optional 'color' (hex e.g. '#FF0000', default red)."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "left":   {"type": "integer"},
                            "top":    {"type": "integer"},
                            "right":  {"type": "integer"},
                            "bottom": {"type": "integer"},
                            "label":  {"type": "string"},
                            "color":  {"type": "string"},
                        },
                        "required": ["left", "top", "right", "bottom"],
                    },
                },
                "outline_width": {
                    "type": "integer",
                    "description": "Thickness of the bounding box outline in pixels (default 3).",
                },
            },
            "required": ["path", "boxes"],
        },
    },
    {
        "name": "page_extract",
        "description": (
            "Extract structured data from the current browser page: links, headings, tables, images, "
            "and meta tags — all in one call. Navigate first with browser(action='navigate', url=...) "
            "then call this to get a structured view without a full screenshot."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "include": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["links", "headings", "tables", "images", "meta", "text"]},
                    "description": "Which data types to extract (default: all). Options: links, headings, tables, images, meta, text.",
                },
                "max_links": {"type": "integer", "description": "Max number of links to return (default 50)."},
                "max_text":  {"type": "integer", "description": "Max characters of body text to return (default 3000)."},
            },
            "required": [],
        },
    },
    {
        "name": "extract_article",
        "description": (
            "Fetch a URL and extract a clean, readable article (title, byline, publish date, and body text) "
            "stripping all ads, nav, and boilerplate. Returns plain text ready for the model to summarise. "
            "Much faster than screenshot for text-heavy pages."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url":       {"type": "string",  "description": "The article URL to fetch and extract."},
                "max_chars": {"type": "integer", "description": "Truncate article body to this many characters (default 8000)."},
            },
            "required": ["url"],
        },
    },
    {
        "name": "page_scrape",
        "description": (
            "Navigate to a URL, scroll through the full page in viewport-sized steps waiting for lazy-loaded "
            "content to appear (infinite-scroll feeds, JS widgets, lazy images), then extract the complete "
            "rendered text from the final DOM state. "
            "Unlike web_fetch or extract_article (which grab text immediately after load), page_scrape "
            "simulates a human scrolling to the bottom, so content that only renders on scroll is captured. "
            "Returns the full page text plus scroll statistics (steps taken, whether the page grew, final height). "
            "Use max_scrolls to control depth (default 10, max 30). "
            "Set include_links=true to also return all hyperlinks from the final DOM."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to navigate to before scraping. Omit to scrape the current browser page.",
                },
                "max_scrolls": {
                    "type": "integer",
                    "description": "Maximum scroll steps (default 10, max 30). Each step is ~one viewport height.",
                },
                "wait_ms": {
                    "type": "integer",
                    "description": "Milliseconds to wait after each scroll step for lazy content to load (default 500).",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return from the page text (default 16000).",
                },
                "include_links": {
                    "type": "boolean",
                    "description": "If true, also return all hyperlinks found in the final DOM (default false).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "page_images",
        "description": (
            "Extract every image URL from a web page: <img> src, highest-resolution srcset variant, "
            "data-src/data-lazy-src lazy-load attributes, <picture><source> responsive images, "
            "og:image and twitter:image meta tags, inline CSS background-image, and JSON-LD "
            "schema.org image/logo/thumbnail fields. Scrolls the page first to trigger lazy loaders. "
            "Returns a deduplicated list (up to 150) with source type and alt text. "
            "Use with fetch_image or browser_save_images to render/download the results."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Page URL to navigate to and extract images from.",
                },
                "scroll": {
                    "type": "boolean",
                    "description": "Scroll before extracting to trigger lazy-loaded images (default true).",
                },
                "max_scrolls": {
                    "type": "integer",
                    "description": "Number of scroll steps to trigger lazy loaders (default 3).",
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "image_search",
        "description": (
            "Search for any image by text description and return MULTIPLE images rendered inline. "
            "Uses query expansion (game shorthands, artwork variant), collects candidates from "
            "multiple pages and query variants, enforces domain diversity (max 2 per domain), "
            "fetches in parallel, and deduplicates via seen-URL cache so repeated calls return "
            "DIFFERENT images. Use 'offset' to paginate to the next batch. "
            "Use for character art, outfit skins, product photos, logos, screenshots, or any "
            "visual content. Example: 'Klukai GFL2 Cerulean Breaker outfit', 'Eiffel Tower night'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language description of the image to find.",
                },
                "count": {
                    "type": "integer",
                    "description": "Number of images to return (default 4, max 20).",
                },
                "offset": {
                    "type": "integer",
                    "description": "Skip first N ranked candidates for pagination (default 0).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "bulk_screenshot",
        "description": (
            "Take screenshots of multiple URLs in parallel and return them all inline. "
            "Useful for monitoring a list of pages, comparing several sites, or batch-capturing search results. "
            "Returns up to 6 screenshots; for larger batches use repeated screenshot calls."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of URLs to screenshot (max 6).",
                },
            },
            "required": ["urls"],
        },
    },
    {
        "name": "scroll_screenshot",
        "description": (
            "Capture a full-page screenshot by scrolling through the page and stitching viewport captures together. "
            "Useful for long pages where a single viewport screenshot misses content. "
            "Navigate to the page first, then call this. Returns one tall composite image."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Page URL to capture. If omitted, captures the currently loaded page.",
                },
                "max_scrolls": {
                    "type": "integer",
                    "description": "Maximum number of viewport scrolls (default 5, max 10).",
                },
                "scroll_overlap": {
                    "type": "integer",
                    "description": "Pixel overlap between consecutive captures to avoid gaps (default 100).",
                },
            },
            "required": [],
        },
    },
    # ── Image generation (LM Studio / OpenAI-compatible) ────────────────────
    {
        "name": "image_generate",
        "description": (
            "Generate an image from a text prompt using LM Studio's OpenAI-compatible image API "
            "(requires an image generation model such as FLUX or SDXL to be loaded in LM Studio). "
            "The generated image is returned inline and saved to the workspace so it can be passed "
            "directly into image_crop, image_zoom, image_scan, image_annotate, image_stitch, etc. "
            "Pipeline: image_generate → image_upscale → image_scan (to read text in generated image) "
            "or image_generate × 2 → image_stitch → image_diff (style comparison)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Text description of the image to generate. Be detailed and specific.",
                },
                "negative_prompt": {
                    "type": "string",
                    "description": "What to avoid in the image (e.g. 'blurry, low quality, watermark').",
                },
                "model": {
                    "type": "string",
                    "description": "Model name as shown in LM Studio (optional; uses IMAGE_GEN_MODEL env var or server default).",
                },
                "size": {
                    "type": "string",
                    "description": "Image dimensions WxH, e.g. '512x512', '768x768', '1024x1024' (default '512x512').",
                },
                "n": {
                    "type": "integer",
                    "description": "Number of images to generate 1–4 (default 1).",
                },
                "steps": {
                    "type": "integer",
                    "description": "Inference steps — more steps = higher quality, slower (optional, provider-specific).",
                },
                "guidance_scale": {
                    "type": "number",
                    "description": "Prompt adherence strength 1–20. Higher = follows prompt more strictly (optional).",
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed for reproducible results (optional, -1 = random).",
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "image_edit",
        "description": (
            "Edit or remix an existing image using a text prompt (img2img / inpainting). "
            "Loads an image from the workspace, applies the prompt-guided transformation, "
            "and returns the result inline. "
            "Pipeline: screenshot → image_edit('make it look like a watercolor') → image_diff(original, edited). "
            "Note: requires LM Studio to support /v1/images/edits — support varies by model."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Source image filename or path (from workspace). Same formats as image_crop.",
                },
                "prompt": {
                    "type": "string",
                    "description": "Editing instruction, e.g. 'make it look like an oil painting' or 'change the sky to night'.",
                },
                "negative_prompt": {
                    "type": "string",
                    "description": "What to avoid in the result (optional).",
                },
                "model": {
                    "type": "string",
                    "description": "Model name to use (optional).",
                },
                "size": {
                    "type": "string",
                    "description": "Output size, e.g. '512x512' (optional, default matches source).",
                },
                "strength": {
                    "type": "number",
                    "description": "How much to transform 0.0–1.0. Lower = more like original, higher = more creative (default 0.75).",
                },
                "n": {
                    "type": "integer",
                    "description": "Number of variations to generate (default 1).",
                },
            },
            "required": ["path", "prompt"],
        },
    },
    {
        "name": "image_remix",
        "description": (
            "Style-transfer / creative remix of an existing workspace image using GPU AI (LM Studio). "
            "Differs from image_edit: image_remix is for creative transformation (style, mood, medium) "
            "while image_edit is for content edits (remove object, change sky). "
            "Examples: 'anime style', 'oil painting', 'cyberpunk neon', 'make it dark mode', "
            "'pixel art', 'watercolor sketch'. "
            "Uses /v1/images/edits with a moderate strength to preserve structure. "
            "Requires an image-generation model in LM Studio; returns a friendly error if none is loaded. "
            "Returns up to 4 inline image variations."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Source image filename or path in the workspace.",
                },
                "prompt": {
                    "type": "string",
                    "description": "Style / transformation description, e.g. 'anime style' or 'oil painting on canvas'.",
                },
                "strength": {
                    "type": "number",
                    "description": "Transformation strength 0.1–1.0 (default 0.65). Lower = more faithful to source, higher = more creative.",
                },
                "n": {
                    "type": "integer",
                    "description": "Number of variations to return, 1–4 (default 1).",
                },
            },
            "required": ["path", "prompt"],
        },
    },
    {
        "name": "image_upscale",
        "description": (
            "Upscale a saved image. GPU AI (via LM Studio) is the primary path — "
            "supports both NVIDIA and Intel Arc GPUs. "
            "CPU LANCZOS is the fallback only when no GPU model is available. "
            "Works on any image in the workspace including generated images, screenshots, or crops. "
            "Set gpu=false to force CPU-only (LANCZOS + optional sharpening). "
            "Pipeline: image_generate → image_upscale(scale=4) → image_scan (read fine text in generated image). "
            "Also useful for making small thumbnails or cropped regions large enough for the vision model to read."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Image filename or path to upscale (same formats as image_crop).",
                },
                "scale": {
                    "type": "number",
                    "description": "CPU LANCZOS upscale factor: 2.0 = 200%, max 8.0 (default 2.0). Ignored when GPU path succeeds.",
                },
                "sharpen": {
                    "type": "boolean",
                    "description": "Apply a sharpening pass after CPU LANCZOS upscaling (default true). Ignored when GPU path succeeds.",
                },
                "gpu": {
                    "type": "boolean",
                    "description": (
                        "Default unset/true = try GPU first (NVIDIA + Intel Arc via LM Studio). "
                        "Set gpu=false to force CPU LANCZOS only (no LM Studio call)."
                    ),
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "tts",
        "description": (
            "Convert text to speech using LM Studio's /v1/audio/speech endpoint. "
            "Saves the audio file to the workspace and returns the file path. "
            "Requires a TTS-capable model loaded in LM Studio. "
            "Supports multiple voices and output formats."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to convert to speech.",
                },
                "voice": {
                    "type": "string",
                    "description": "Voice name (default 'alloy'). Options: alloy, echo, fable, onyx, nova, shimmer.",
                },
                "speed": {
                    "type": "number",
                    "description": "Speech speed multiplier 0.25–4.0 (default 1.0).",
                },
                "format": {
                    "type": "string",
                    "description": "Output format: mp3, opus, aac, flac, wav (default mp3).",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "embed_store",
        "description": (
            "Embed a piece of text using LM Studio's /v1/embeddings endpoint and store it in the "
            "PostgreSQL database for later semantic search. "
            "Use a meaningful key (e.g. URL or doc ID) and optional topic for filtering. "
            "Pipeline: embed_store → embed_search to find semantically similar documents."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Unique identifier for this document (URL, hash, or descriptive ID).",
                },
                "content": {
                    "type": "string",
                    "description": "The text to embed and store (up to ~8000 chars).",
                },
                "topic": {
                    "type": "string",
                    "description": "Optional topic tag for filtering (e.g. 'science', 'news').",
                },
            },
            "required": ["key", "content"],
        },
    },
    {
        "name": "embed_search",
        "description": (
            "Embed a query string and find the most semantically similar documents previously stored "
            "via embed_store. Uses cosine similarity on LM Studio embeddings. "
            "Returns ranked results with similarity scores."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to embed and match against stored documents.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 5).",
                },
                "topic": {
                    "type": "string",
                    "description": "Filter results to a specific topic tag (optional).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "code_run",
        "description": (
            "Execute arbitrary Python code in a sandboxed subprocess with a 30-second timeout. "
            "Returns stdout, stderr, exit code, and duration. "
            "Optionally installs pip packages before running. "
            "GPU support: when code references torch/tensorflow/cuda, a DEVICE variable is "
            "automatically injected ('cuda', 'mps', or 'cpu') — use model.to(DEVICE) without "
            "any setup. Pre-installed packages: numpy, scipy, opencv-python-headless (cv2), "
            "Pillow, httpx. Larger GPU frameworks (torch, tensorflow) can be installed via packages param. "
            "Use for data analysis, GPU tensor operations, image processing, or testing code snippets."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source code to execute.",
                },
                "packages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of pip packages to install before running (e.g. ['requests', 'pandas']).",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Wall-clock timeout in seconds (default 30, max 120).",
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "smart_summarize",
        "description": (
            "Summarize text using LM Studio's chat completion endpoint. "
            "Supports 3 styles: brief (2-3 sentences), detailed (full paragraph), bullets (markdown list). "
            "Pass the raw text directly (already fetched from web_fetch or db_cache_get)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Text to summarize (up to ~8000 chars; longer content is truncated).",
                },
                "style": {
                    "type": "string",
                    "enum": ["brief", "detailed", "bullets"],
                    "description": "Summary style: brief (2-3 sentences), detailed (paragraph), bullets (markdown list).",
                },
                "max_words": {
                    "type": "integer",
                    "description": "Approximate target word count for the summary (optional).",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "image_caption",
        "description": (
            "Describe an image in detail using LM Studio's vision model. "
            "Pass the image as a base64-encoded JPEG string. "
            "Returns a detailed description including subject, colors, style, and notable elements. "
            "Pipeline: fetch_image → image_caption (describe) or screenshot → image_caption."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "b64": {
                    "type": "string",
                    "description": "Base64-encoded JPEG image data (without data: URI prefix).",
                },
                "detail_level": {
                    "type": "string",
                    "enum": ["brief", "detailed"],
                    "description": "brief = one sentence, detailed = full description with colors/mood/style (default detailed).",
                },
            },
            "required": ["b64"],
        },
    },
    {
        "name": "structured_extract",
        "description": (
            "Extract structured JSON data from text using LM Studio's json_object response mode. "
            "Provide the text to parse and a JSON Schema describing the desired output shape. "
            "Uses response_format: json_object for strict JSON output. "
            "Ideal for pulling structured facts from articles, product pages, or research text."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Text to extract structured data from (up to ~6000 chars).",
                },
                "schema_json": {
                    "type": "string",
                    "description": 'JSON Schema string describing the output shape, e.g. \'{"type":"object","properties":{"title":{"type":"string"}}}\'.',
                },
                "instructions": {
                    "type": "string",
                    "description": "Additional extraction instructions or context (optional).",
                },
            },
            "required": ["content", "schema_json"],
        },
    },
    # ── Knowledge Graph ──────────────────────────────────────────────────────
    {
        "name": "graph_add_node",
        "description": (
            "Add or update a node in the knowledge graph. "
            "Nodes have an ID, a list of labels (categories), and arbitrary JSON properties. "
            "Use this to build a persistent knowledge base of entities and concepts. "
            "Pipeline: graph_add_node → graph_add_edge to connect nodes → graph_query to explore."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "id":         {"type": "string", "description": "Unique node ID (e.g. 'person:alice', 'concept:AI')."},
                "labels":     {"type": "array",  "items": {"type": "string"},
                               "description": "Category labels (e.g. ['Person', 'Researcher'])."},
                "properties": {"type": "object", "description": "Arbitrary key-value metadata."},
            },
            "required": ["id"],
        },
    },
    {
        "name": "graph_add_edge",
        "description": (
            "Add a directed edge between two nodes in the knowledge graph. "
            "Edges have a from_id, to_id, type (relationship label), and optional properties. "
            "Example: graph_add_edge(from_id='person:alice', to_id='org:mit', type='works_at')."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "from_id":    {"type": "string", "description": "Source node ID."},
                "to_id":      {"type": "string", "description": "Target node ID."},
                "type":       {"type": "string", "description": "Relationship type (e.g. 'knows', 'works_at', 'related_to')."},
                "properties": {"type": "object", "description": "Optional edge metadata."},
            },
            "required": ["from_id", "to_id"],
        },
    },
    {
        "name": "graph_query",
        "description": (
            "Get a node and all its connected neighbors from the knowledge graph. "
            "Returns the node's properties plus its outgoing and incoming edges. "
            "Use this to explore what is connected to a known entity."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Node ID to look up."},
            },
            "required": ["id"],
        },
    },
    {
        "name": "graph_path",
        "description": (
            "Find the shortest path between two nodes in the knowledge graph using BFS. "
            "Returns the list of node IDs forming the path, or null if no path exists. "
            "Useful for discovering indirect relationships between entities."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "from_id": {"type": "string", "description": "Starting node ID."},
                "to_id":   {"type": "string", "description": "Target node ID."},
            },
            "required": ["from_id", "to_id"],
        },
    },
    {
        "name": "graph_search",
        "description": (
            "Search the knowledge graph for nodes matching a label and/or property values. "
            "Returns a list of matching nodes with their properties. "
            "Example: graph_search(label='Person', properties={'city': 'Seattle'})."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "label":      {"type": "string", "description": "Node label to filter by (substring match)."},
                "properties": {"type": "object", "description": "Property key-value pairs to match exactly."},
                "limit":      {"type": "integer", "description": "Max results (default 50)."},
            },
        },
    },
    # ── Vector Store (Qdrant) ────────────────────────────────────────────────
    {
        "name": "vector_store",
        "description": (
            "Store a text embedding in the Qdrant vector database for later semantic search. "
            "Text is embedded via LM Studio (/v1/embeddings) and stored with its metadata. "
            "Supports multiple collections (namespaces) for organizing different knowledge domains. "
            "Pipeline: vector_store → vector_search to retrieve similar content."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text":       {"type": "string", "description": "Text to embed and store."},
                "id":         {"type": "string", "description": "Unique identifier for this entry."},
                "collection": {"type": "string", "description": "Qdrant collection name (default: 'default')."},
                "metadata":   {"type": "object", "description": "Arbitrary metadata stored alongside the vector."},
            },
            "required": ["text", "id"],
        },
    },
    {
        "name": "vector_search",
        "description": (
            "Semantic search in the Qdrant vector database. "
            "Embeds the query text and finds the most similar stored entries. "
            "Returns top_k results with scores and metadata. "
            "More powerful than embed_search: supports multi-collection and metadata filtering."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query":      {"type": "string",  "description": "Search query text."},
                "collection": {"type": "string",  "description": "Qdrant collection to search (default: 'default')."},
                "top_k":      {"type": "integer", "description": "Number of results (default 5)."},
                "filter":     {"type": "object",  "description": "Optional Qdrant payload filter object."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "vector_delete",
        "description": "Delete a vector entry from Qdrant by its ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id":         {"type": "string", "description": "Entry ID to delete."},
                "collection": {"type": "string", "description": "Collection name (default: 'default')."},
            },
            "required": ["id"],
        },
    },
    {
        "name": "vector_collections",
        "description": "List all Qdrant collections (vector namespaces) and their sizes.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    # ── Video Analysis ───────────────────────────────────────────────────────
    {
        "name": "video_info",
        "description": (
            "Get metadata from a video URL: duration, frame rate, resolution, codec, file size. "
            "Supports HTTP/HTTPS video URLs (mp4, webm, mkv, etc.). "
            "Use this before video_frames to know the video length and choose a good interval."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "HTTP/HTTPS URL of the video file."},
            },
            "required": ["url"],
        },
    },
    {
        "name": "video_frames",
        "description": (
            "Extract frames from a video at regular intervals and save them to the workspace. "
            "Returns a list of frame file paths with their timestamps. "
            "Frames can be passed to screenshot, image_crop, ocr_image, or image_caption for analysis. "
            "Pipeline: video_frames → [frame paths] → image_caption or ocr_image."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url":          {"type": "string",  "description": "HTTP/HTTPS URL of the video."},
                "interval_sec": {"type": "number",  "description": "Seconds between frames (default 5.0)."},
                "max_frames":   {"type": "integer", "description": "Maximum frames to extract (default 20, max 100)."},
            },
            "required": ["url"],
        },
    },
    {
        "name": "video_thumbnail",
        "description": (
            "Extract a single frame from a video at a specific timestamp. "
            "Returns the frame as an inline image block. "
            "Use to preview video content or get a representative image for a specific moment."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url":           {"type": "string", "description": "HTTP/HTTPS URL of the video."},
                "timestamp_sec": {"type": "number", "description": "Timestamp in seconds (default 0.0)."},
            },
            "required": ["url"],
        },
    },
    # ── OCR ─────────────────────────────────────────────────────────────────
    {
        "name": "ocr_image",
        "description": (
            "Extract text from an image in the workspace using Tesseract OCR. "
            "More accurate than LLM vision for dense text (receipts, scanned docs, charts with labels). "
            "Pass a workspace file path (e.g. from screenshot, video_frames, image_scan). "
            "Pipeline: screenshot → ocr_image to read text from any web page or document scan."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace file path (e.g. '/workspace/screenshot_20260301.png')."},
                "lang": {"type": "string", "description": "Tesseract language code (default 'eng'). Use 'eng+fra' for multi-language."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "ocr_pdf",
        "description": (
            "Extract text from a PDF file using Tesseract OCR (rasterizes pages at 300 DPI). "
            "Better than pdfminer for scanned PDFs that have no embedded text. "
            "Provide the PDF path from the workspace. Returns per-page text and a combined full_text."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path":  {"type": "string",  "description": "Workspace path to the PDF file."},
                "pages": {"type": "array", "items": {"type": "integer"},
                          "description": "Specific 1-based page numbers to OCR (omit for all pages)."},
                "lang":  {"type": "string", "description": "Tesseract language code (default 'eng')."},
            },
            "required": ["path"],
        },
    },
    # ── Document Ingestion ───────────────────────────────────────────────────
    {
        "name": "docs_ingest",
        "description": (
            "Convert a document (PDF, DOCX, XLSX, PPTX, HTML, MD, TXT) to clean Markdown. "
            "Accepts a URL (downloaded automatically) or a workspace file path. "
            "Extracts headings, paragraphs, and tables in normalized Markdown format. "
            "Pipeline: docs_ingest → smart_summarize to summarize any document in one step."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url":      {"type": "string", "description": "HTTP/HTTPS URL of the document (alternative to path)."},
                "path":     {"type": "string", "description": "Workspace file path (alternative to url)."},
                "filename": {"type": "string", "description": "Filename with extension — needed for format detection when using path."},
            },
        },
    },
    {
        "name": "docs_extract_tables",
        "description": (
            "Extract all tables from a document as structured JSON. "
            "Each table has a title, headers list, and rows list. "
            "Supports PDF, DOCX, XLSX, HTML. "
            "Use when you need the data from tables in a structured format for analysis."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url":      {"type": "string", "description": "HTTP/HTTPS URL of the document."},
                "path":     {"type": "string", "description": "Workspace file path."},
                "filename": {"type": "string", "description": "Filename with extension."},
            },
        },
    },
    # ── Task Planner ─────────────────────────────────────────────────────────
    {
        "name": "plan_create_task",
        "description": (
            "Create a task in the persistent task planner with optional dependencies. "
            "A task is 'ready' when all its depends_on tasks are done. "
            "Use this to break complex goals into trackable sub-tasks with a dependency graph. "
            "Pipeline: plan_create_task (multiple, with depends_on) → plan_list_tasks → plan_complete_task."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "title":       {"type": "string",  "description": "Short task title."},
                "description": {"type": "string",  "description": "Detailed task description."},
                "depends_on":  {"type": "array", "items": {"type": "string"},
                                "description": "Task IDs that must complete before this task is ready."},
                "priority":    {"type": "integer", "description": "Priority (higher = more urgent, default 0)."},
                "due_at":      {"type": "string",  "description": "ISO 8601 due date (optional)."},
                "metadata":    {"type": "object",  "description": "Arbitrary metadata (e.g. assignee, tags)."},
            },
            "required": ["title"],
        },
    },
    {
        "name": "plan_get_task",
        "description": "Get the status and details of a task by its ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Task ID returned by plan_create_task."},
            },
            "required": ["id"],
        },
    },
    {
        "name": "plan_complete_task",
        "description": "Mark a task as done. Unlocks any tasks that depend on this one.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Task ID to mark as done."},
            },
            "required": ["id"],
        },
    },
    {
        "name": "plan_fail_task",
        "description": "Mark a task as failed with an optional reason.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id":     {"type": "string", "description": "Task ID to mark as failed."},
                "detail": {"type": "string", "description": "Failure reason or error message."},
            },
            "required": ["id"],
        },
    },
    {
        "name": "plan_list_tasks",
        "description": (
            "List tasks in the planner. Filter by status or leave blank for all tasks. "
            "Use status='ready' to find tasks that can be started now (all dependencies satisfied)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {"type": "string",
                           "description": "Filter: pending | ready | in_progress | done | failed | cancelled. Omit for all."},
                "limit":  {"type": "integer", "description": "Max results (default 50)."},
            },
        },
    },
    {
        "name": "plan_delete_task",
        "description": "Delete a task from the planner.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Task ID to delete."},
            },
            "required": ["id"],
        },
    },
    {
        "name": "orchestrate",
        "description": (
            "Execute a multi-step workflow of MCP tools with automatic parallelism and "
            "dependency management. Steps without dependencies run in parallel via "
            "asyncio.gather(); steps with 'depends_on' run after their prerequisites complete. "
            "Earlier step outputs can be injected into later step arguments using "
            "{step_id.result} placeholders. Returns a structured report with every step's "
            "result and timing. "
            "Use this to chain browser + research + summarise + store operations in a single "
            "call, dramatically reducing the number of LLM turns needed for complex tasks."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "description": "Ordered list of workflow steps to execute.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": (
                                    "Unique step identifier (letters, digits, underscores). "
                                    "Used in depends_on references and {id.result} placeholders."
                                ),
                            },
                            "tool": {
                                "type": "string",
                                "description": (
                                    "MCP tool name to call "
                                    "(e.g. 'screenshot', 'web_search', 'smart_summarize')."
                                ),
                            },
                            "args": {
                                "type": "object",
                                "description": (
                                    "Arguments for the tool. String values may contain "
                                    "{id.result} to embed a prior step's text output."
                                ),
                            },
                            "depends_on": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Step IDs that must complete before this step runs. "
                                    "Omit or leave empty for parallel execution with other "
                                    "independent steps."
                                ),
                            },
                            "label": {
                                "type": "string",
                                "description": (
                                    "Human-readable label shown in the result report. "
                                    "Defaults to the step id if omitted."
                                ),
                            },
                        },
                        "required": ["id", "tool", "args"],
                    },
                },
                "stop_on_error": {
                    "type": "boolean",
                    "description": (
                        "If true, abort all remaining steps when any step fails. "
                        "Default: false (continue executing independent steps)."
                    ),
                },
            },
            "required": ["steps"],
        },
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text(s: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": s}]


def _image_blocks(container_path: str, summary: str) -> list[dict[str, Any]]:
    """Return MCP content blocks: text summary + inline base64 image (compressed for LM Studio).
    Delegates to ImageRenderer.encode_path() which enforces the payload size cap."""
    return _renderer.encode_path(container_path, summary)


# ---------------------------------------------------------------------------
# Orchestration — WorkflowStep / WorkflowResult / WorkflowExecutor
# ---------------------------------------------------------------------------

@dataclass
class WorkflowStep:
    """A single step in an orchestrated workflow."""
    id: str                              # unique identifier; used in depends_on + {id.result}
    tool: str                            # MCP tool name to call
    args: dict                           # tool arguments (strings may use {id.result} placeholders)
    depends_on: list[str] = field(default_factory=list)  # step IDs that must finish first
    label: str = ""                      # human-readable label for the result report


@dataclass
class WorkflowResult:
    """Result from a single executed workflow step."""
    step_id: str
    label: str
    tool: str
    result: str        # first text block returned by the tool
    ok: bool           # False if the tool raised or returned an error
    duration_ms: int


class WorkflowExecutor:
    """Execute a DAG of WorkflowSteps with automatic parallelism.

    Steps with no pending dependencies are gathered and run concurrently
    via asyncio.gather(). Steps that depend on earlier steps receive their
    results via {step_id.result} interpolation in arg string values.
    """

    def __init__(self, steps: list[WorkflowStep], stop_on_error: bool = False) -> None:
        self._steps = steps
        self._stop_on_error = stop_on_error

    async def run(self) -> list[WorkflowResult]:
        """Execute all steps respecting dependencies; return results in execution order."""
        waves = self._build_waves()
        completed: dict[str, WorkflowResult] = {}
        results: list[WorkflowResult] = []
        for wave in waves:
            if self._stop_on_error and any(not r.ok for r in results):
                break
            wave_results = await asyncio.gather(
                *[self._run_step(s, completed) for s in wave]
            )
            for r in wave_results:
                completed[r.step_id] = r
                results.append(r)
        return results

    async def _run_step(
        self, step: WorkflowStep, completed: dict[str, WorkflowResult]
    ) -> WorkflowResult:
        """Interpolate args, call the tool, return a WorkflowResult."""
        label = step.label or step.id
        t0 = _time.monotonic()
        try:
            interpolated = self._interpolate(step.args, completed)
            content = await _call_tool(step.tool, interpolated)
            text = next((b["text"] for b in content if b.get("type") == "text"), "")
            ok = not (
                text.startswith("Error ")
                or text.startswith("Unknown tool")
                or text.startswith(f"Step '{step.id}' raised")
            )
        except Exception as exc:
            text = f"Step '{step.id}' raised: {exc}"
            ok = False
        ms = int((_time.monotonic() - t0) * 1000)
        return WorkflowResult(step.id, label, step.tool, text, ok, ms)

    def _build_waves(self) -> list[list[WorkflowStep]]:
        """Topological sort via Kahn's algorithm → ordered execution waves.

        Raises ValueError on cycle or unknown dependency reference.
        """
        by_id: dict[str, WorkflowStep] = {}
        for s in self._steps:
            if s.id in by_id:
                raise ValueError(f"duplicate step id: '{s.id}'")
            by_id[s.id] = s

        # Validate all depends_on references
        for s in self._steps:
            for dep in s.depends_on:
                if dep not in by_id:
                    raise ValueError(
                        f"step '{s.id}' depends on unknown step '{dep}'"
                    )

        # Kahn's algorithm
        in_degree: dict[str, int] = {s.id: len(s.depends_on) for s in self._steps}
        queue: list[str] = [sid for sid, d in in_degree.items() if d == 0]
        waves: list[list[WorkflowStep]] = []
        visited = 0

        while queue:
            wave = [by_id[sid] for sid in queue]
            waves.append(wave)
            visited += len(queue)
            next_queue: list[str] = []
            for sid in queue:
                for s in self._steps:
                    if sid in s.depends_on:
                        in_degree[s.id] -= 1
                        if in_degree[s.id] == 0:
                            next_queue.append(s.id)
            queue = next_queue

        if visited != len(self._steps):
            raise ValueError("cycle detected in workflow dependencies")
        return waves

    def _interpolate(self, args: dict, completed: dict[str, WorkflowResult]) -> dict:
        """Recursively replace {step_id.result} in string arg values."""
        out: dict = {}
        for k, v in args.items():
            if isinstance(v, str):
                for step_id, res in completed.items():
                    v = v.replace(f"{{{step_id}.result}}", res.result)
                out[k] = v
            elif isinstance(v, dict):
                out[k] = self._interpolate(v, completed)
            else:
                out[k] = v
        return out

    @staticmethod
    def _format_report(results: list[WorkflowResult]) -> str:
        """Format a human-readable multi-line report of all step results."""
        lines = ["## Workflow Results\n"]
        for i, r in enumerate(results, 1):
            status = "OK" if r.ok else "FAILED"
            preview = r.result[:500] + ("..." if len(r.result) > 500 else "")
            lines.append(
                f"### Step {i}: {r.label} [{status}] ({r.duration_ms} ms)\n"
                f"Tool: `{r.tool}`\n\n"
                f"{preview}\n"
            )
        total_ms = sum(r.duration_ms for r in results)
        ok_count = sum(1 for r in results if r.ok)
        lines.append(
            f"---\n**{ok_count}/{len(results)} steps succeeded** | "
            f"total wall-time: {total_ms} ms"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Image manipulation helpers (PIL-based, no external OCR required)
# ---------------------------------------------------------------------------

def _resolve_image_path(path: str) -> str | None:
    """
    Accept any of:
      - bare filename            →  BROWSER_WORKSPACE/<filename>
      - /workspace/<filename>    →  BROWSER_WORKSPACE/<filename>  (human_browser container path)
      - /docker/human_browser/workspace/<filename>  →  BROWSER_WORKSPACE/<filename>
      - any other absolute path  →  used as-is if it exists
    Returns a readable local path or None if not found.
    """
    if not path:
        return None
    name = os.path.basename(path)
    # Bare filename or known prefix → remap to our bind-mount
    if "/" not in path or path.startswith("/workspace/") or path.startswith("/docker/human_browser/workspace/"):
        candidate = os.path.join(BROWSER_WORKSPACE, name)
        return candidate if os.path.isfile(candidate) else None
    return path if os.path.isfile(path) else None


def _pil_to_blocks(
    img: "_PilImage.Image",
    summary: str,
    quality: int = 85,
    save_prefix: str | None = None,
) -> list[dict[str, Any]]:
    """Encode a PIL Image as an inline JPEG MCP block (delegates to ImageRenderer)."""
    return _renderer.encode(img, summary, save_prefix=save_prefix, quality=quality)


# ---------------------------------------------------------------------------
# ImageRenderer — OOP image encoding pipeline for LM Studio inline rendering
# ---------------------------------------------------------------------------

# LM Studio silently drops (shows "external image") when the MCP payload exceeds
# its internal message size limit.  All encoding paths go through ImageRenderer,
# which enforces a hard byte-count cap by stepping down JPEG quality.
_MAX_INLINE_BYTES: int = 3_000_000   # 3 MB raw  ≈  4 MB base64 — safe for LM Studio


class ImageRenderer:
    """
    OOP wrapper that encodes PIL images, workspace paths, and raw HTTP bytes
    as inline MCP image content blocks, always honouring LM Studio's payload cap.

    Usage
    -----
    renderer = ImageRenderer()
    blocks = renderer.encode(pil_img, "Screenshot of …")
    blocks = renderer.encode_path("/workspace/shot.png", "Summary text")
    blocks = renderer.encode_url_bytes(raw_bytes, "image/jpeg", "Fetched from …")
    """

    MAX_BYTES: int = _MAX_INLINE_BYTES
    # Quality ladder: try highest first, step down until payload fits.
    _QUALITY_LADDER: tuple[int, ...] = (85, 75, 65, 50)

    # ── private helpers ──────────────────────────────────────────────────────

    def _compress_to_limit(self, img: "_PilImage.Image", min_quality: int = 85) -> bytes:
        """JPEG-compress img, reducing quality until payload < MAX_BYTES.

        Starts from min_quality (caller's preference) and steps down through
        standard rungs until the payload fits within MAX_BYTES.
        """
        # Build a descending ladder starting at the caller's quality preference
        _RUNGS = (75, 65, 50)
        ladder = [min_quality] + [q for q in _RUNGS if q < min_quality]
        for q in ladder:
            buf = _io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=q)
            raw = buf.getvalue()
            if len(raw) <= self.MAX_BYTES:
                return raw
        # Absolute last resort: quality=40
        buf = _io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=40)
        return buf.getvalue()

    def _fit(
        self,
        img: "_PilImage.Image",
        max_w: int = 1280,
        max_h: int = 1024,
    ) -> "_PilImage.Image":
        """Downscale img to fit within max_w × max_h, preserving aspect ratio.
        When max_w/max_h are raised (e.g. for upscale output), the image passes
        through unchanged unless it exceeds those larger bounds."""
        if img.width > max_w or img.height > max_h:
            img = img.copy()
            img.thumbnail((max_w, max_h), _PilImage.LANCZOS)
        return img

    # ── public API ───────────────────────────────────────────────────────────

    def encode(
        self,
        img: "_PilImage.Image",
        summary: str,
        save_prefix: str | None = None,
        quality: int = 85,
        max_w: int = 1280,
        max_h: int = 1024,
    ) -> list[dict[str, Any]]:
        """
        Encode a PIL Image → [text_block, image_block], guaranteed to fit
        within MAX_BYTES.  Optionally saves the compressed JPEG to BROWSER_WORKSPACE.
        quality is the preferred JPEG quality (85–95 recommended); the encoder
        steps down automatically if the payload would exceed MAX_BYTES.
        max_w / max_h control the pixel-dimension cap before compression
        (default 1280×1024 for chat; raise to 4096×4096 for upscale output).
        """
        img = self._fit(img.convert("RGB"), max_w=max_w, max_h=max_h)
        raw = self._compress_to_limit(img, min_quality=max(40, min(quality, 95)))
        if save_prefix and os.path.isdir(BROWSER_WORKSPACE):
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"{save_prefix}_{ts}.jpg"
                with open(os.path.join(BROWSER_WORKSPACE, fname), "wb") as fh:
                    fh.write(raw)
                summary += f"\n→ Saved as: {fname}  (pass this as 'path' in the next pipeline step)"
            except OSError:
                pass  # workspace may be read-only; inline image is still returned
        b64 = base64.standard_b64encode(raw).decode("ascii")
        return [
            {"type": "text",  "text": summary},
            {"type": "image", "data": b64, "mimeType": "image/jpeg"},
        ]

    def encode_path(self, container_path: str, summary: str) -> list[dict[str, Any]]:
        """
        Load an image from BROWSER_WORKSPACE (given a container path or bare filename)
        and return inline MCP blocks.  Returns text-only if the file is missing.
        Replaces the old _image_blocks() function.
        """
        blocks: list[dict[str, Any]] = [{"type": "text", "text": summary}]
        if not container_path:
            return blocks
        fname = os.path.basename(container_path)
        local_path = os.path.join(BROWSER_WORKSPACE, fname)
        if not os.path.isfile(local_path):
            return blocks
        try:
            if _HAS_PIL:
                with _PilImage.open(local_path) as img:
                    # Reuse encode() but skip the text block (already in blocks[0])
                    encoded = self.encode(img, "")
                    blocks.extend(b for b in encoded if b.get("type") == "image")
            else:
                with open(local_path, "rb") as fh:
                    raw = fh.read()
                # Cap payload: LM Studio silently drops images that exceed the limit.
                if len(raw) > self.MAX_BYTES:
                    raw = raw[: self.MAX_BYTES]  # truncate as last resort
                b64 = base64.standard_b64encode(raw).decode("ascii")
                blocks.append({"type": "image", "data": b64, "mimeType": "image/png"})
        except Exception:
            pass
        return blocks

    def encode_url_bytes(
        self,
        raw: bytes,
        content_type: str,
        summary: str,
    ) -> list[dict[str, Any]]:
        """
        Compress raw HTTP image bytes → inline MCP blocks.
        Used by fetch_image so that large external images are always compressed
        before being sent to LM Studio (fixes the "external image" display bug).
        Falls back to raw if PIL is unavailable and the payload is small enough.
        """
        if _HAS_PIL:
            try:
                with _PilImage.open(_io.BytesIO(raw)) as img:
                    return self.encode(_ImageOps.exif_transpose(img).convert("RGB"), summary)
            except Exception:
                pass  # corrupt / unrecognised format — try raw fallback below
        # PIL unavailable or image unreadable — send raw only if it fits
        if len(raw) <= self.MAX_BYTES:
            b64 = base64.standard_b64encode(raw).decode("ascii")
            return [
                {"type": "text",  "text": summary},
                {"type": "image", "data": b64, "mimeType": content_type},
            ]
        return [{"type": "text",
                 "text": summary + "\n⚠ Image too large to render inline (PIL unavailable)."}]


_renderer = ImageRenderer()   # module-level singleton used by all tool handlers


# ---------------------------------------------------------------------------
# ImageRenderingPolicy — guarantees every image tool produces an image block
# ---------------------------------------------------------------------------

class ImageRenderingPolicy:
    """Guarantees every image-returning MCP tool produces at least one image block.

    LM Studio silently shows 'external image' when a tool response contains only
    text blocks where an inline image was expected.  Call ``enforce()`` on any
    content-block list from an image-producing tool to guarantee rendering.

    Usage
    -----
    content = await _call_tool(name, args)
    if ImageRenderingPolicy.is_image_tool(name):
        content = ImageRenderingPolicy.enforce(content)

    Three-step escalation inside enforce():
    1. Image block already present  → return unchanged.
    2. ``fallback_bytes`` provided  → compress via _renderer and embed.
    3. Still missing                → append a small dark JPEG placeholder so
                                      LM Studio always renders something visual.
    """

    # Tools that MUST always return at least one inline image block when called
    # from LM Studio (i.e. via _handle_rpc → tools/call).
    IMAGE_TOOLS: frozenset[str] = frozenset({
        "screenshot", "fetch_image",
        "image_generate", "image_edit", "image_remix",
        "image_crop", "image_zoom", "image_scan", "image_enhance",
        "image_stitch", "image_diff", "image_annotate", "image_upscale",
        "bulk_screenshot", "scroll_screenshot", "screenshot_search",
        "browser_save_images", "browser_download_page_images",
        "image_search", "db_list_images",
    })

    @staticmethod
    def has_image(blocks: list[dict[str, Any]]) -> bool:
        """Return True if any block in blocks is an image block."""
        return any(b.get("type") == "image" for b in blocks)

    @classmethod
    def is_image_tool(cls, name: str) -> bool:
        """Return True if this tool is expected to always return an image block."""
        return name in cls.IMAGE_TOOLS

    @classmethod
    def enforce(
        cls,
        blocks: list[dict[str, Any]],
        fallback_bytes: bytes = b"",
        content_type: str = "image/jpeg",
    ) -> list[dict[str, Any]]:
        """Ensure ``blocks`` contains at least one image block.

        Parameters
        ----------
        blocks:
            Content blocks returned by a tool handler.
        fallback_bytes:
            Optional raw image bytes (e.g. downloaded from the web) to use
            as a last-chance source if no image block is present.
        content_type:
            MIME type of ``fallback_bytes`` (default ``image/jpeg``).
        """
        if cls.has_image(blocks):
            return blocks

        # Step 2 — try to produce an image from raw bytes via _renderer
        if fallback_bytes:
            summary = cls._first_text(blocks)
            extra = _renderer.encode_url_bytes(fallback_bytes, content_type, summary)
            if cls.has_image(extra):
                text_blocks = [b for b in blocks if b.get("type") == "text"]
                img_blocks  = [b for b in extra  if b.get("type") == "image"]
                return text_blocks + img_blocks

        # Step 3 — placeholder image so LM Studio never shows 'external image'
        ph = cls._placeholder()
        if ph is not None:
            return blocks + [ph]
        return blocks  # PIL unavailable; best-effort text-only

    @staticmethod
    def _first_text(blocks: list[dict[str, Any]]) -> str:
        """Return text from the first text block, or empty string."""
        return next((b["text"] for b in blocks if b.get("type") == "text"), "")

    @staticmethod
    def _placeholder() -> dict[str, str] | None:
        """Generate a minimal dark-grey JPEG indicating image was unavailable.

        Returns None if PIL is not available (caller handles gracefully).
        """
        if not _HAS_PIL:
            return None
        buf = _io.BytesIO()
        img = _PilImage.new("RGB", (400, 100), color=(28, 28, 28))
        img.save(buf, "JPEG", quality=60)
        return {
            "type":     "image",
            "data":     base64.standard_b64encode(buf.getvalue()).decode("ascii"),
            "mimeType": "image/jpeg",
        }


# ---------------------------------------------------------------------------
# GpuDetector / GpuUpscaler — GPU-accelerated upscaling via LM Studio
# ---------------------------------------------------------------------------

class GpuDetector:
    """
    Detect available GPU hardware (NVIDIA and Intel Arc) without requiring
    PyTorch or other heavy frameworks — uses lightweight OS-level probes only.
    Results are cached after the first call.
    Only used by GpuUpscaler (image_upscale tool); not wired to other tools.
    """

    _cache: dict[str, str] | None = None   # {"vendor": "nvidia"|"intel"|"none", "name": str}

    @classmethod
    def detect(cls) -> dict[str, str]:
        """Return {"vendor": "nvidia"|"intel"|"none", "name": <human readable>}."""
        if cls._cache is not None:
            return cls._cache
        cls._cache = cls._probe()
        return cls._cache

    @classmethod
    def _probe(cls) -> dict[str, str]:
        import subprocess as _sp

        # ── NVIDIA: check nvidia-smi or /dev/nvidia0 ──────────────────────
        try:
            out = _sp.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                timeout=4, stderr=_sp.DEVNULL,
            ).decode().strip().splitlines()
            if out:
                return {"vendor": "nvidia", "name": out[0]}
        except Exception:
            pass
        if os.path.exists("/dev/nvidia0"):
            return {"vendor": "nvidia", "name": "NVIDIA (device node)"}

        # ── Intel Arc / Intel GPU: check /dev/dri or vainfo ───────────────
        try:
            # vainfo reports the VA-API driver; Intel iHD = Arc/Iris/UHD
            out = _sp.check_output(
                ["vainfo", "--display", "drm"],
                timeout=4, stderr=_sp.DEVNULL,
            ).decode()
            if "iHD" in out or "Intel" in out or "i965" in out:
                # Extract driver name line if present
                for ln in out.splitlines():
                    if "Driver version" in ln or "vainfo: Driver" in ln:
                        return {"vendor": "intel", "name": ln.strip()}
                return {"vendor": "intel", "name": "Intel GPU (vainfo)"}
        except Exception:
            pass
        # Fallback: render node exists → likely Intel integrated or Arc
        try:
            dri = os.listdir("/dev/dri")
            renders = [d for d in dri if d.startswith("renderD")]
            if renders:
                return {"vendor": "intel", "name": f"Intel GPU (/dev/dri/{renders[0]})"}
        except Exception:
            pass

        # ── Check env var override (useful in Docker with GPU passthrough) ─
        if os.environ.get("NVIDIA_VISIBLE_DEVICES", "").strip() not in ("", "void"):
            return {"vendor": "nvidia", "name": "NVIDIA (env NVIDIA_VISIBLE_DEVICES)"}
        if os.environ.get("INTEL_GPU", "").strip() == "1":
            return {"vendor": "intel", "name": "Intel GPU (env INTEL_GPU=1)"}

        return {"vendor": "none", "name": "No GPU detected"}

    @classmethod
    def available(cls) -> bool:
        return cls.detect()["vendor"] != "none"

    @classmethod
    def vendor(cls) -> str:
        return cls.detect()["vendor"]

    @classmethod
    def name(cls) -> str:
        return cls.detect()["name"]


class GpuUpscaler:
    """
    AI-powered image upscaling via LM Studio's /v1/images/edits endpoint.
    Supports both NVIDIA and Intel Arc GPUs (LM Studio handles the acceleration).
    Only used by the image_upscale tool.

    Falls back gracefully: if no GPU is detected or the LM Studio call fails,
    returns None so the caller can fall through to LANCZOS upscaling.
    """

    _PROMPT = "upscale to high resolution, sharpen fine details, enhance clarity"
    _STRENGTH = "0.35"   # low strength = preserve structure, enhance quality

    def __init__(self, base_url: str, model: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        self.model    = model

    async def upscale(
        self,
        img: "_PilImage.Image",
        client: "httpx.AsyncClient",
    ) -> "_PilImage.Image | None":
        """
        Send img to LM Studio /v1/images/edits for AI upscaling.
        Returns the upscaled PIL Image on success, None on any failure.
        """
        if not _HAS_PIL:
            return None
        # Encode input image as JPEG for the multipart upload
        in_buf = _io.BytesIO()
        img.convert("RGB").save(in_buf, format="JPEG", quality=92)
        in_buf.seek(0)

        form: dict[str, str] = {
            "prompt": self._PROMPT,
            "n": "1",
            "response_format": "b64_json",
            "strength": self._STRENGTH,
        }
        if self.model:
            form["model"] = self.model

        try:
            resp = await asyncio.wait_for(
                client.post(
                    f"{self.base_url}/v1/images/edits",
                    files={"image": ("input.jpg", in_buf, "image/jpeg")},
                    data=form,
                    timeout=90.0,
                ),
                timeout=95.0,
            )
            if resp.status_code != 200:
                return None
            b64_out = resp.json()["data"][0]["b64_json"]
            return _PilImage.open(_io.BytesIO(base64.b64decode(b64_out))).convert("RGB")
        except Exception:
            return None

    def gpu_label(self) -> str:
        """Human-readable GPU label for the summary text."""
        info = GpuDetector.detect()
        return info["name"]


# ---------------------------------------------------------------------------
# GpuImageProcessor — OpenCV/CUDA-accelerated image ops, PIL fallback
# ---------------------------------------------------------------------------

class GpuImageProcessor:
    """
    GPU-accelerated image operations for the aichat image pipeline tools.

    Uses OpenCV (cv2) with CUDA support when available; otherwise falls back
    transparently to PIL/Pillow.  All public methods accept and return PIL
    Images so existing tool code needs minimal changes.

    Call GpuImageProcessor.backend() to see which engine is active.
    """

    # Class-level engine detection (set once at class definition time)
    _CUDA_OK: bool = False   # cv2 built with CUDA and a GPU is present
    _CV2_OK:  bool = _HAS_CV2

    if _HAS_CV2:
        try:
            _CUDA_OK = _cv2.cuda.getCudaEnabledDeviceCount() > 0  # type: ignore[union-attr]
        except Exception:
            _CUDA_OK = False

    # ── internal conversion helpers ─────────────────────────────────────────

    @staticmethod
    def _to_np(img: "_PilImage.Image") -> "_np.ndarray":
        """PIL Image → BGR numpy array (OpenCV native format)."""
        rgb = _np.array(img.convert("RGB"))
        return _cv2.cvtColor(rgb, _cv2.COLOR_RGB2BGR)  # type: ignore[union-attr]

    @staticmethod
    def _to_pil(arr: "_np.ndarray") -> "_PilImage.Image":
        """BGR numpy array → PIL Image."""
        rgb = _cv2.cvtColor(arr, _cv2.COLOR_BGR2RGB)  # type: ignore[union-attr]
        return _PilImage.fromarray(rgb)

    # ── public API ───────────────────────────────────────────────────────────

    @classmethod
    def backend(cls) -> str:
        """Return the active image processing engine name."""
        if cls._CUDA_OK:
            return "opencv-cuda"
        if cls._CV2_OK:
            return "opencv-cpu"
        return "pillow"

    @classmethod
    def resize(
        cls,
        img: "_PilImage.Image",
        w: int,
        h: int,
    ) -> "_PilImage.Image":
        """High-quality resize to w×h.  LANCZOS4 via cv2, LANCZOS via PIL."""
        if cls._CV2_OK:
            arr = cls._to_np(img)
            resized = _cv2.resize(arr, (w, h), interpolation=_cv2.INTER_LANCZOS4)  # type: ignore[union-attr]
            return cls._to_pil(resized)
        copy = img.copy()
        copy.thumbnail((w, h), _PilImage.LANCZOS)
        return copy

    @classmethod
    def sharpen(
        cls,
        img: "_PilImage.Image",
        radius: float = 0.5,
        percent: int = 80,
        threshold: int = 2,
    ) -> "_PilImage.Image":
        """Unsharp-mask sharpening.  cv2 GaussianBlur kernel, PIL UnsharpMask fallback."""
        if cls._CV2_OK:
            arr = cls._to_np(img)
            blur = _cv2.GaussianBlur(arr, (0, 0), max(0.1, radius * 4))  # type: ignore[union-attr]
            amount = percent / 100.0
            sharpened = _cv2.addWeighted(arr, 1.0 + amount, blur, -amount, 0)  # type: ignore[union-attr]
            return cls._to_pil(sharpened)
        return img.filter(_ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))

    @classmethod
    def enhance_contrast(
        cls,
        img: "_PilImage.Image",
        factor: float = 2.5,
    ) -> "_PilImage.Image":
        """Contrast enhancement.  cv2 convertScaleAbs, PIL ImageEnhance fallback."""
        if cls._CV2_OK:
            arr = cls._to_np(img)
            # Linear contrast stretch: new = clip(alpha*old + beta)
            alpha = max(0.5, min(factor, 5.0))
            enhanced = _cv2.convertScaleAbs(arr, alpha=alpha, beta=0)  # type: ignore[union-attr]
            return cls._to_pil(enhanced)
        return _ImageEnhance.Contrast(img).enhance(factor)

    @classmethod
    def enhance_sharpness(
        cls,
        img: "_PilImage.Image",
        factor: float = 1.4,
    ) -> "_PilImage.Image":
        """Sharpness enhancement.  cv2 Laplacian kernel, PIL ImageEnhance fallback."""
        if cls._CV2_OK:
            arr = cls._to_np(img)
            # Scale kernel weight by sharpness factor
            k = max(0.0, factor - 1.0)
            kernel = _np.array([[0, -k, 0], [-k, 1 + 4 * k, -k], [0, -k, 0]], dtype=_np.float32)  # type: ignore[union-attr]
            sharpened = _cv2.filter2D(arr, -1, kernel)  # type: ignore[union-attr]
            return cls._to_pil(sharpened)
        return _ImageEnhance.Sharpness(img).enhance(factor)

    @classmethod
    def diff(
        cls,
        img1: "_PilImage.Image",
        img2: "_PilImage.Image",
    ) -> "_PilImage.Image":
        """Pixel-wise absolute difference.  cv2.absdiff, PIL ImageChops fallback."""
        if cls._CV2_OK:
            a1 = cls._to_np(img1)
            a2 = cls._to_np(img2)
            # Resize img2 to match img1 if sizes differ
            if a1.shape != a2.shape:
                a2 = _cv2.resize(a2, (a1.shape[1], a1.shape[0]), interpolation=_cv2.INTER_LANCZOS4)  # type: ignore[union-attr]
            d = _cv2.absdiff(a1, a2)  # type: ignore[union-attr]
            return cls._to_pil(d)
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, _PilImage.LANCZOS)
        return _ImageChops.difference(img1, img2)

    @classmethod
    def to_grayscale(cls, img: "_PilImage.Image") -> "_PilImage.Image":
        """Convert to grayscale.  cv2.cvtColor or PIL convert('L')."""
        if cls._CV2_OK:
            arr = cls._to_np(img)
            gray = _cv2.cvtColor(arr, _cv2.COLOR_BGR2GRAY)  # type: ignore[union-attr]
            return _PilImage.fromarray(gray)
        return img.convert("L")

    @classmethod
    def annotate(
        cls,
        img: "_PilImage.Image",
        boxes: "list[tuple[int,int,int,int]]",
        labels: "list[str]",
        color: "tuple[int,int,int]" = (255, 80, 0),
        thickness: int = 3,
    ) -> "_PilImage.Image":
        """Draw bounding boxes + labels.  cv2.rectangle/putText, PIL ImageDraw fallback."""
        if cls._CV2_OK:
            arr = cls._to_np(img)
            bgr = (color[2], color[1], color[0])  # RGB → BGR
            for (x1, y1, x2, y2), label in zip(boxes, labels):
                _cv2.rectangle(arr, (x1, y1), (x2, y2), bgr, thickness)  # type: ignore[union-attr]
                if label:
                    _cv2.putText(  # type: ignore[union-attr]
                        arr, label, (x1, max(y1 - 6, 0)),
                        _cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2, _cv2.LINE_AA,  # type: ignore[union-attr]
                    )
            return cls._to_pil(arr)
        # PIL fallback
        draw = _ImageDraw.Draw(img)
        for (x1, y1, x2, y2), label in zip(boxes, labels):
            draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
            if label:
                draw.text((x1, max(y1 - 14, 0)), label, fill=color)
        return img


_GpuImg = GpuImageProcessor()   # module-level singleton


# ---------------------------------------------------------------------------
# ModelRegistry — /v1/models probe with TTL cache
# ---------------------------------------------------------------------------

class ModelRegistry:
    """
    Probes LM Studio's /v1/models endpoint (30 s TTL) so tools can make
    adaptive decisions: skip GPU calls when no model is loaded, surface
    helpful errors instead of 90 s timeouts, route to the right model.

    Usage: await ModelRegistry.get().is_available(client)
    """

    _TTL:      float = 30.0
    _instance: "ModelRegistry | None" = None

    def __init__(self) -> None:
        self._models:      list[dict[str, Any]] = []
        self._last_probe:  float = 0.0
        self._probe_ok:    bool  = False

    @classmethod
    def get(cls) -> "ModelRegistry":
        """Return the process-level singleton."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def invalidate(self) -> None:
        """Force a fresh probe on the next access."""
        self._last_probe = 0.0

    async def _refresh(self, client: httpx.AsyncClient) -> None:
        """Probe /v1/models; update cache.  Silently swallows all errors."""
        try:
            r = await asyncio.wait_for(
                client.get(f"{IMAGE_GEN_BASE_URL}/v1/models"),
                timeout=4.0,
            )
            if r.status_code == 200:
                data = r.json()
                self._models     = data.get("data", [])
                self._probe_ok   = True
                self._last_probe = _time.monotonic()
                return
        except Exception:
            pass
        self._models     = []
        self._probe_ok   = False
        self._last_probe = _time.monotonic()

    async def _ensure_fresh(self, client: httpx.AsyncClient) -> None:
        if _time.monotonic() - self._last_probe >= self._TTL:
            await self._refresh(client)

    async def models(self, client: httpx.AsyncClient) -> list[dict[str, Any]]:
        """Return cached model list, refreshing if the TTL has expired."""
        await self._ensure_fresh(client)
        return list(self._models)

    async def is_available(self, client: httpx.AsyncClient) -> bool:
        """True if LM Studio responded with at least one model."""
        await self._ensure_fresh(client)
        return self._probe_ok and bool(self._models)

    async def has_vision(self, client: httpx.AsyncClient) -> bool:
        """True if any loaded model supports vision (multimodal)."""
        for m in await self.models(client):
            mid = (m.get("id") or "").lower()
            mtype = (m.get("type") or "").lower()
            if "vision" in mid or "vlm" in mid or "vision" in mtype or "multimodal" in mtype:
                return True
        return False

    async def has_image_gen(self, client: httpx.AsyncClient) -> bool:
        """True if any image-generation model is loaded."""
        for m in await self.models(client):
            mid  = (m.get("id") or "").lower()
            mtype = (m.get("type") or "").lower()
            if any(k in mid for k in ("flux", "sdxl", "stable-diffusion", "sd-", "img2img")):
                return True
            if "image" in mtype or "diffusion" in mtype:
                return True
        # If IMAGE_GEN_MODEL env is explicitly set, trust the user
        if IMAGE_GEN_MODEL.strip():
            return self._probe_ok
        return False

    async def best_chat_model(self, client: httpx.AsyncClient) -> str:
        """Return IMAGE_GEN_MODEL if set, else first available chat model, else ''."""
        if IMAGE_GEN_MODEL.strip():
            return IMAGE_GEN_MODEL.strip()
        for m in await self.models(client):
            mtype = (m.get("type") or "").lower()
            if "llm" in mtype or "chat" in mtype or not mtype:
                return m.get("id", "")
        return ""


# ---------------------------------------------------------------------------
# VisionCache — in-memory phash → vision result (LRU, MAX_SIZE=500)
# ---------------------------------------------------------------------------

class VisionCache:
    """
    LRU in-memory cache mapping image perceptual hashes (phash) to vision
    confirmation results.  Eliminates redundant LM Studio /v1/chat/completions
    calls for images that have already been confirmed or rejected.

    Capacity: 500 entries; least-recently-used entry evicted on overflow.

    LRU semantics
    -------------
    * ``get()`` — cache hit does NOT promote the entry (read-only).
    * ``put()`` — on re-insert, the entry is moved to the *back* of the
      eviction queue so frequently-updated hashes are never spuriously evicted.
    * Uses ``collections.deque`` for O(1) popleft on every eviction.
    """

    _MAX_SIZE: int = 500

    def __init__(self) -> None:
        self._cache: dict[str, tuple[bool, str, float]] = {}
        self._order: collections.deque[str] = collections.deque()  # LRU order

    def get(self, phash: str) -> tuple[bool, str, float] | None:
        """Return cached (is_match, desc, conf) for phash, or None if absent."""
        if not phash:
            return None
        return self._cache.get(phash)

    def put(self, phash: str, result: tuple[bool, str, float]) -> None:
        """Store result; on re-insert move to back (LRU); evict LRU on overflow."""
        if not phash:
            return
        if phash in self._cache:
            # Re-insert: promote to most-recently-used position.
            try:
                self._order.remove(phash)
            except ValueError:
                pass  # defensive: deque and cache can't be out of sync, but be safe
        else:
            # New entry: evict LRU if at capacity.
            if len(self._cache) >= self._MAX_SIZE:
                oldest = self._order.popleft()   # O(1) via deque
                self._cache.pop(oldest, None)
        self._order.append(phash)
        self._cache[phash] = result

    def size(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()
        self._order.clear()


_vision_cache = VisionCache()   # module-level singleton


# ---------------------------------------------------------------------------
# GpuCodeRuntime — auto-inject GPU device detection into code_run payloads
# ---------------------------------------------------------------------------

class GpuCodeRuntime:
    """
    Prepends a GPU device detection preamble to user-supplied code when the
    code references torch, tensorflow, or CUDA keywords.  This gives the
    ``code_run`` tool a ``DEVICE`` variable (``'cuda'`` | ``'mps'`` | ``'cpu'``)
    that works automatically regardless of the host GPU type.

    Pre-installed packages (available without pip install):
      numpy, scipy, opencv-python-headless (cv2), Pillow, httpx
    """

    _PREINSTALLED: frozenset[str] = frozenset({"numpy", "scipy", "cv2", "PIL", "httpx"})
    _GPU_TRIGGERS: frozenset[str] = frozenset({"torch", "tensorflow", "tf.", "cuda", ".to(", "device"})

    _PREAMBLE = _textwrap.dedent("""\
        # ── GPU device auto-detection (injected by aichat GpuCodeRuntime) ──────
        _device = "cpu"
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _device = "cuda"
            elif getattr(getattr(_torch, "backends", None), "mps", None) and _torch.backends.mps.is_available():
                _device = "mps"
        except ImportError:
            pass
        DEVICE = _device   # use this in your code: model.to(DEVICE)
        # ───────────────────────────────────────────────────────────────────────
    """)

    @classmethod
    def needs_device_injection(cls, code: str) -> bool:
        """True if the code mentions GPU-related keywords."""
        return any(kw in code for kw in cls._GPU_TRIGGERS)

    @classmethod
    def prepare(cls, code: str) -> str:
        """Return code with the DEVICE preamble prepended when GPU triggers detected."""
        if cls.needs_device_injection(code):
            return cls._PREAMBLE + "\n" + code
        return code

    @classmethod
    def available_packages(cls) -> list[str]:
        """Return importable GPU-related packages in the current Python env."""
        candidates = ["torch", "tensorflow", "cv2", "numpy", "scipy", "cupy", "jax"]
        available: list[str] = []
        import importlib
        for pkg in candidates:
            try:
                importlib.import_module(pkg)
                available.append(pkg)
            except ImportError:
                pass
        return available


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
                find_text  = str(args.get("find_text",  "")).strip() or None
                find_image = str(args.get("find_image", "")).strip() or None
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                container_path = f"/workspace/screenshot_{ts}.png"
                shot_req: dict = {"url": url, "path": container_path}
                if find_text:
                    shot_req["find_text"] = find_text
                elif find_image:
                    shot_req["find_image"] = find_image
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
                image_meta = data.get("image_meta", {})
                if clipped and find_image:
                    src_hint = image_meta.get("src", find_image)
                    nat_w = image_meta.get("natural_w", 0)
                    nat_h = image_meta.get("natural_h", 0)
                    dim_note = f" ({nat_w}×{nat_h} natural)" if nat_w and nat_h else ""
                    clip_note = f"\nImage: '{src_hint}'{dim_note}"
                elif clipped and find_text:
                    clip_note = f"\nZoomed to: '{find_text}'"
                else:
                    clip_note = ""
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
                            fallback_summary = (
                                f"Screenshot of: {page_title}\n"
                                f"URL: {url}\n"
                                f"(screenshot blocked — showing page image)"
                            )
                            return _renderer.encode_url_bytes(ir.content, ct, fallback_summary)
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
                # Return inline base64 image — always compress via ImageRenderer so
                # large PNGs/WebPs never exceed LM Studio's MCP payload cap.
                summary = (
                    f"Image from: {url}\n"
                    f"Type: {content_type}  Size: {len(img_data):,} bytes\n"
                    f"File: {host_path}"
                )
                return _renderer.encode_url_bytes(img_data, content_type, summary)

            # ----------------------------------------------------------------
            if name == "screenshot_search":
                query = str(args.get("query", "")).strip()
                if not query:
                    return _text("screenshot_search: 'query' is required")
                max_results = max(1, min(int(args.get("max_results", 3)), 5))

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
                _deadline = asyncio.get_running_loop().time() + 24.0
                for i, url in enumerate(urls):
                    remaining = _deadline - asyncio.get_running_loop().time()
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
                search_params: dict = {}
                if args.get("topic"):
                    search_params["topic"] = str(args["topic"])
                if args.get("q"):
                    search_params["q"] = str(args["q"])
                try:
                    search_params["limit"] = max(1, min(200, int(args.get("limit", 20))))
                    search_params["offset"] = max(0, int(args.get("offset", 0)))
                except (ValueError, TypeError) as exc:
                    return _text(f"db_search: 'limit' and 'offset' must be integers — {exc}")
                if args.get("summary_only"):
                    search_params["summary_only"] = "true"
                r = await c.get(f"{DATABASE_URL}/articles/search", params=search_params)
                return _text(json.dumps(r.json()))

            if name == "db_cache_store":
                cache_url_cs = str(args.get("url", "")).strip()
                cache_content_cs = str(args.get("content", "")).strip()
                if not cache_url_cs:
                    return _text("db_cache_store: 'url' is required")
                if not cache_content_cs:
                    return _text("db_cache_store: 'content' is required")
                cache_payload_cs: dict = {"url": cache_url_cs, "content": cache_content_cs}
                if args.get("title"):
                    cache_payload_cs["title"] = str(args["title"])
                r = await c.post(f"{DATABASE_URL}/cache/store", json=cache_payload_cs)
                return _text(json.dumps(r.json()))

            if name == "db_cache_get":
                cache_url_cg = str(args.get("url", "")).strip()
                if not cache_url_cg:
                    return _text("db_cache_get: 'url' is required")
                r = await c.get(f"{DATABASE_URL}/cache/get", params={"url": cache_url_cg})
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
                key_ms = str(args.get("key", "")).strip()
                val_ms = str(args.get("value", "")).strip()
                if not key_ms:
                    return _text("memory_store: 'key' is required")
                if not val_ms:
                    return _text("memory_store: 'value' is required")
                payload: dict = {"key": key_ms, "value": val_ms}
                if args.get("ttl_seconds"):
                    try:
                        payload["ttl_seconds"] = int(args["ttl_seconds"])
                    except (ValueError, TypeError):
                        return _text("memory_store: 'ttl_seconds' must be an integer")
                r = await c.post(f"{MEMORY_URL}/store", json=payload)
                return _text(json.dumps(r.json()))

            if name == "memory_recall":
                params: dict = {}
                if args.get("key"):
                    params["key"] = str(args["key"])
                if args.get("pattern"):
                    params["pattern"] = str(args["pattern"])
                r = await c.get(f"{MEMORY_URL}/recall", params=params)
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
                if action == "screenshot_element":
                    selector = str(args.get("selector", "")).strip()
                    if not selector:
                        return _text("browser screenshot_element: 'selector' is required")
                    pad = int(args.get("pad", 20))
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    container_path = f"/workspace/element_{ts}.png"
                    try:
                        el_r = await c.post(
                            f"{BROWSER_URL}/screenshot_element",
                            json={"selector": selector, "path": container_path, "pad": pad},
                            timeout=20.0,
                        )
                        data = el_r.json()
                    except Exception as exc:
                        return _text(f"browser screenshot_element failed: {exc}")
                    err = data.get("error", "")
                    if err:
                        return _text(f"browser screenshot_element: {err}")
                    saved_path = data.get("path", container_path)
                    filename = os.path.basename(saved_path)
                    local_path = os.path.join(BROWSER_WORKSPACE, filename)
                    host_path = f"/docker/human_browser/workspace/{filename}"
                    bbox = data.get("bbox", {})
                    bbox_note = (
                        f"  bbox: x={bbox.get('x',0):.0f}, y={bbox.get('y',0):.0f}, "
                        f"w={bbox.get('width',0):.0f}, h={bbox.get('height',0):.0f}"
                        if bbox else ""
                    )
                    summary = (
                        f"Element screenshot: {selector}\n"
                        f"File: {host_path}{bbox_note}"
                    )
                    if os.path.isfile(local_path):
                        return _image_blocks(saved_path, summary)
                    return _text(f"screenshot_element: file not found at {local_path}")
                if action == "list_images_detail":
                    try:
                        imgs_r = await c.get(f"{BROWSER_URL}/images", timeout=10.0)
                        imgs_data = imgs_r.json()
                    except Exception as exc:
                        return _text(f"browser list_images_detail failed: {exc}")
                    images = imgs_data.get("images", imgs_data) if isinstance(imgs_data, dict) else imgs_data
                    if not images:
                        return _text("No images found on the current page.")
                    lines = ["Images on current page:"]
                    for img in images:
                        idx  = img.get("index", "?")
                        src  = img.get("src", "")
                        alt  = img.get("alt", "")
                        rw   = img.get("rendered_w", 0)
                        rh   = img.get("rendered_h", 0)
                        nw   = img.get("natural_w", 0)
                        nh   = img.get("natural_h", 0)
                        vis  = "✓" if img.get("visible") else "✗"
                        vp   = "in-viewport" if img.get("in_viewport") else "off-screen"
                        dim  = f"{rw}×{rh}px rendered" + (f" ({nw}×{nh} natural)" if nw else "")
                        lines.append(
                            f"  [{idx}] {vis} {vp}  {dim}\n"
                            f"       src: {src[:120]}\n"
                            f"       alt: {alt[:80]}"
                        )
                    return _text("\n".join(lines))
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

            # ----------------------------------------------------------------
            if name == "get_errors":
                limit = max(1, min(int(args.get("limit", 50)), 200))
                params: dict = {"limit": limit}
                svc = str(args.get("service", "")).strip()
                if svc:
                    params["service"] = svc
                r = await c.get(f"{DATABASE_URL}/errors/recent", params=params, timeout=10.0)
                data = r.json()
                errors = data.get("errors", [])
                if not errors:
                    return _text("No errors logged yet." + (f" (service={svc})" if svc else ""))
                lines = [f"Recent errors ({len(errors)}):"]
                for e in errors:
                    ts = str(e.get("logged_at", ""))[:19].replace("T", " ")
                    lines.append(
                        f"  [{ts}] [{e.get('level', '?')}] {e.get('service', '?')}: {e.get('message', '')}"
                        + (f"\n    detail: {e['detail']}" if e.get("detail") else "")
                    )
                return _text("\n".join(lines))

            # ----------------------------------------------------------------
            # Image manipulation tools (PIL — no HTTP calls needed)
            # ----------------------------------------------------------------
            if name in ("image_crop", "image_zoom", "image_scan", "image_enhance"):
                if not _HAS_PIL:
                    return _text(f"{name}: Pillow is not installed in this container.")
                path = str(args.get("path", "")).strip()
                local = _resolve_image_path(path)
                if not local:
                    return _text(f"{name}: image not found — tried '{path}' in {BROWSER_WORKSPACE}")

                with _PilImage.open(local) as _src:
                    img = _ImageOps.exif_transpose(_src.copy())

                w, h = img.size
                try:
                    left   = int(args.get("left",   0))
                    top    = int(args.get("top",    0))
                    right  = int(args.get("right",  w))
                    bottom = int(args.get("bottom", h))
                except (ValueError, TypeError) as _e:
                    return _text(f"{name}: invalid coordinate — {_e}")
                # Clamp to image bounds and make positive
                if right  <= 0: right  = w
                if bottom <= 0: bottom = h
                right  = min(right,  w)
                bottom = min(bottom, h)
                box = (max(0, left), max(0, top), right, bottom)

                if name == "image_crop":
                    if box[2] <= box[0] or box[3] <= box[1]:
                        return _text(f"image_crop: invalid box {box} for {w}×{h} image")
                    result = img.crop(box)
                    rw, rh = result.size
                    summary = (
                        f"Cropped ({box[0]},{box[1]}) → ({box[2]},{box[3]})\n"
                        f"Original: {w}×{h}  Result: {rw}×{rh}  Source: {os.path.basename(path)}"
                    )
                    return _pil_to_blocks(result, summary, save_prefix="cropped")

                if name == "image_zoom":
                    scale = max(1.1, min(float(args.get("scale", 2.0)), 8.0))
                    region = img.crop(box) if box != (0, 0, w, h) else img
                    rw, rh = region.size
                    zoomed = _GpuImg.resize(region, int(rw * scale), int(rh * scale))
                    zw, zh = zoomed.size
                    summary = (
                        f"Zoomed {scale:.1f}× from ({box[0]},{box[1]})→({box[2]},{box[3]})\n"
                        f"Region: {rw}×{rh}  Output: {zw}×{zh}  Source: {os.path.basename(path)}\n"
                        f"Backend: {_GpuImg.backend()}"
                    )
                    return _pil_to_blocks(zoomed, summary, quality=92, save_prefix="zoomed")

                if name == "image_scan":
                    region = img.crop(box) if box != (0, 0, w, h) else img
                    rw, rh = region.size
                    # Auto-upscale small regions so fine text is legible for the vision model
                    if rw < 800:
                        up = max(2, 800 // max(rw, 1))
                        region = _GpuImg.resize(region, rw * up, rh * up)
                        rw, rh = region.size
                    # Greyscale → contrast boost → sharpen → unsharp mask (GPU-accelerated)
                    region = _GpuImg.to_grayscale(region)
                    region = _GpuImg.enhance_contrast(region.convert("RGB"), 2.5)
                    region = _GpuImg.enhance_sharpness(region, 3.0)
                    region = _GpuImg.sharpen(region, radius=1, percent=150, threshold=3)
                    summary = (
                        f"Scan-enhanced for text reading — greyscale + contrast×2.5 + sharpen×3.0\n"
                        f"Region: ({box[0]},{box[1]})→({box[2]},{box[3]})  Output: {rw}×{rh}\n"
                        f"Source: {os.path.basename(path)}  Backend: {_GpuImg.backend()}\n"
                        f"Read all text visible in this image."
                    )
                    return _pil_to_blocks(region, summary, quality=95, save_prefix="scanned")

                if name == "image_enhance":
                    contrast   = max(0.5, min(float(args.get("contrast",   1.5)), 4.0))
                    sharpness  = max(0.5, min(float(args.get("sharpness",  1.5)), 4.0))
                    brightness = max(0.5, min(float(args.get("brightness", 1.0)), 3.0))
                    grayscale  = bool(args.get("grayscale", False))
                    result = img.copy()
                    if grayscale:
                        result = result.convert("L").convert("RGB")
                    result = _GpuImg.enhance_contrast(result, contrast)
                    result = _GpuImg.enhance_sharpness(result, sharpness)
                    result = _ImageEnhance.Brightness(result).enhance(brightness)  # PIL only
                    summary = (
                        f"Enhanced: contrast={contrast:.1f} sharpness={sharpness:.1f} "
                        f"brightness={brightness:.1f}"
                        + (" grayscale" if grayscale else "") + "\n"
                        f"Size: {w}×{h}  Source: {os.path.basename(path)}\n"
                        f"Backend: {_GpuImg.backend()}"
                    )
                    return _pil_to_blocks(result, summary, save_prefix="enhanced")

            # ----------------------------------------------------------------
            # image_stitch — combine multiple images into one canvas
            # ----------------------------------------------------------------
            if name == "image_stitch":
                if not _HAS_PIL:
                    return _text("image_stitch: Pillow is not installed.")
                paths = [str(p) for p in args.get("paths", [])]
                if len(paths) < 2:
                    return _text("image_stitch: at least 2 paths are required")
                paths = paths[:8]
                direction = str(args.get("direction", "vertical")).lower()
                try:
                    gap = max(0, int(args.get("gap", 0)))
                except (ValueError, TypeError):
                    gap = 0
                images: list["_PilImage.Image"] = []
                for p in paths:
                    loc = _resolve_image_path(p)
                    if not loc:
                        return _text(f"image_stitch: image not found — '{p}'")
                    with _PilImage.open(loc) as im:
                        images.append(_ImageOps.exif_transpose(im.convert("RGB").copy()))
                if direction == "horizontal":
                    total_w = sum(im.width for im in images) + gap * (len(images) - 1)
                    max_h   = max(im.height for im in images)
                    canvas  = _PilImage.new("RGB", (total_w, max_h), (255, 255, 255))
                    x = 0
                    for im in images:
                        # Center shorter images vertically within the canvas
                        y_off = (max_h - im.height) // 2
                        canvas.paste(im, (x, y_off))
                        x += im.width + gap
                else:
                    max_w   = max(im.width for im in images)
                    total_h = sum(im.height for im in images) + gap * (len(images) - 1)
                    canvas  = _PilImage.new("RGB", (max_w, total_h), (255, 255, 255))
                    y = 0
                    for im in images:
                        # Center narrower images horizontally within the canvas
                        x_off = (max_w - im.width) // 2
                        canvas.paste(im, (x_off, y))
                        y += im.height + gap
                summary = (
                    f"Stitched {len(images)} images ({direction})\n"
                    f"Output size: {canvas.width}×{canvas.height}"
                )
                return _pil_to_blocks(canvas, summary, save_prefix="stitched")

            # ----------------------------------------------------------------
            # image_diff — highlight pixel-level differences between two images
            # ----------------------------------------------------------------
            if name == "image_diff":
                if not _HAS_PIL:
                    return _text("image_diff: Pillow is not installed.")
                path_a = str(args.get("path_a", "")).strip()
                path_b = str(args.get("path_b", "")).strip()
                loc_a = _resolve_image_path(path_a)
                loc_b = _resolve_image_path(path_b)
                if not loc_a:
                    return _text(f"image_diff: path_a not found — '{path_a}'")
                if not loc_b:
                    return _text(f"image_diff: path_b not found — '{path_b}'")
                try:
                    amplify = max(1.0, min(float(args.get("amplify", 3.0)), 10.0))
                except (ValueError, TypeError):
                    amplify = 3.0
                with _PilImage.open(loc_a) as ia:
                    img_a = _ImageOps.exif_transpose(ia.convert("RGB").copy())
                with _PilImage.open(loc_b) as ib:
                    img_b = _ImageOps.exif_transpose(ib.convert("RGB").copy())
                diff = _GpuImg.diff(img_a, img_b)   # handles size mismatch internally
                diff_l = diff.convert("L")
                diff_l = _ImageEnhance.Brightness(diff_l).enhance(amplify)
                white = _PilImage.new("RGB", img_a.size, (255, 255, 255))
                red   = _PilImage.new("RGB", img_a.size, (220, 30, 30))
                result = _PilImage.composite(red, white, diff_l)
                summary = (
                    f"Pixel diff: {os.path.basename(path_a)} vs {os.path.basename(path_b)}\n"
                    f"Amplify: {amplify:.1f}×  Size: {img_a.width}×{img_a.height}\n"
                    f"Backend: {_GpuImg.backend()}  Red pixels = changed regions."
                )
                return _pil_to_blocks(result, summary, save_prefix="diff")

            # ----------------------------------------------------------------
            # image_annotate — draw bounding boxes and labels on a screenshot
            # ----------------------------------------------------------------
            if name == "image_annotate":
                if not _HAS_PIL:
                    return _text("image_annotate: Pillow is not installed.")
                path = str(args.get("path", "")).strip()
                boxes = args.get("boxes", [])
                if not path:
                    return _text("image_annotate: 'path' is required")
                if not boxes:
                    return _text("image_annotate: 'boxes' list is required")
                loc = _resolve_image_path(path)
                if not loc:
                    return _text(f"image_annotate: image not found — '{path}'")
                try:
                    outline_width = max(1, int(args.get("outline_width", 3)))
                except (ValueError, TypeError):
                    outline_width = 3
                with _PilImage.open(loc) as src:
                    img = _ImageOps.exif_transpose(src.convert("RGB").copy())
                # Parse boxes into tuples for GpuImageProcessor.annotate()
                box_tuples: list[tuple[int, int, int, int]] = []
                box_labels: list[str] = []
                box_colors: list[tuple[int, int, int]] = []
                for box in boxes:
                    try:
                        bx_l = int(box.get("left",   0))
                        bx_t = int(box.get("top",    0))
                        bx_r = int(box.get("right",  img.width))
                        bx_b = int(box.get("bottom", img.height))
                    except (ValueError, TypeError):
                        continue  # skip malformed box rather than crash
                    box_tuples.append((bx_l, bx_t, bx_r, bx_b))
                    box_labels.append(str(box.get("label", "")))
                    raw_col = str(box.get("color", "#FF3333")).lstrip("#")
                    try:
                        r = int(raw_col[0:2], 16)
                        g = int(raw_col[2:4], 16)
                        b = int(raw_col[4:6], 16)
                        box_colors.append((r, g, b))
                    except Exception:
                        box_colors.append((255, 51, 51))
                # Use the first box color for all (GpuImageProcessor uses a single color param)
                # For multi-color support, fall back to PIL draw loop
                if len(set(str(c) for c in box_colors)) == 1 or _HAS_CV2:
                    primary_color = box_colors[0] if box_colors else (255, 51, 51)
                    img = _GpuImg.annotate(img, box_tuples, box_labels,
                                           color=primary_color, thickness=outline_width)
                else:
                    # Multiple distinct colors — use PIL for per-box color control
                    draw = _ImageDraw.Draw(img)
                    for (bx_l, bx_t, bx_r, bx_b), label, col in zip(box_tuples, box_labels, box_colors):
                        for i in range(outline_width):
                            draw.rectangle([bx_l - i, bx_t - i, bx_r + i, bx_b + i], outline=col)
                        if label:
                            tx, ty = bx_l + 2, max(0, bx_t - 18)
                            draw.rectangle([tx - 2, ty - 2, tx + len(label) * 7 + 2, ty + 16], fill=col)
                            draw.text((tx, ty), label, fill="white")
                n = len(boxes)
                summary = (
                    f"Annotated {n} bounding box{'es' if n != 1 else ''} on {os.path.basename(path)}\n"
                    f"Size: {img.width}×{img.height}  Backend: {_GpuImg.backend()}"
                )
                return _pil_to_blocks(img, summary, save_prefix="annotated")

            # ----------------------------------------------------------------
            # page_extract — JS-based structured data extraction from live page
            # ----------------------------------------------------------------
            if name == "page_extract":
                include = list(args.get("include") or ["links", "headings", "tables", "images", "meta", "text"])
                max_links = max(1, int(args.get("max_links", 50)))
                max_text  = max(100, int(args.get("max_text", 3000)))
                js = (
                    "(function(){"
                    "var out={};"
                    "var inc=" + json.dumps(include) + ";"
                    "function has(k){return inc.indexOf(k)!==-1;}"
                    "if(has('meta')){"
                    " var m={};document.querySelectorAll('meta[name],meta[property]').forEach(function(el){"
                    "  var k=el.getAttribute('name')||el.getAttribute('property');"
                    "  if(k)m[k]=el.getAttribute('content')||'';"
                    " });out.meta=m;out.title=document.title;}"
                    "if(has('headings')){"
                    " out.headings=Array.from(document.querySelectorAll('h1,h2,h3,h4')).slice(0,50).map(function(h){"
                    "  return{tag:h.tagName,text:h.innerText.trim()};});}"
                    "if(has('links')){"
                    " out.links=Array.from(document.querySelectorAll('a[href]')).slice(0," + str(max_links) + ").map(function(a){"
                    "  return{text:a.innerText.trim().slice(0,120),href:a.href};});}"
                    "if(has('images')){"
                    " out.images=Array.from(document.querySelectorAll('img[src]')).slice(0,30).map(function(i){"
                    "  return{src:i.src,alt:i.alt||''};});}"
                    "if(has('tables')){"
                    " out.tables=Array.from(document.querySelectorAll('table')).slice(0,5).map(function(tbl){"
                    "  return Array.from(tbl.querySelectorAll('tr')).slice(0,20).map(function(tr){"
                    "   return Array.from(tr.querySelectorAll('th,td')).map(function(td){return td.innerText.trim();});});});}"
                    "if(has('text')){"
                    " out.text=(document.body?document.body.innerText:'').slice(0," + str(max_text) + ");}"
                    "return JSON.stringify(out);})()"
                )
                try:
                    eval_r = await c.post(f"{BROWSER_URL}/eval", json={"code": js}, timeout=15.0)
                    raw = eval_r.json().get("result", "{}")
                    data = json.loads(raw) if isinstance(raw, str) else (raw or {})
                    lines: list[str] = []
                    if data.get("title"):
                        lines.append(f"Title: {data['title']}")
                    for k, v in list((data.get("meta") or {}).items())[:10]:
                        lines.append(f"  meta[{k}]: {str(v)[:120]}")
                    if "headings" in data:
                        lines.append(f"\nHeadings ({len(data['headings'])}):")
                        for hd in data["headings"]:
                            lines.append(f"  {hd['tag']}: {hd['text'][:120]}")
                    if "links" in data:
                        lines.append(f"\nLinks ({len(data['links'])}):")
                        for lk in data["links"]:
                            lines.append(f"  [{lk['text'][:60]}] → {lk['href'][:120]}")
                    if "images" in data:
                        lines.append(f"\nImages ({len(data['images'])}):")
                        for im in data["images"][:10]:
                            lines.append(f"  {im['src'][:100]}  alt='{im['alt'][:60]}'")
                    if "tables" in data:
                        lines.append(f"\nTables ({len(data['tables'])}):")
                        for ti, tbl in enumerate(data["tables"]):
                            lines.append(f"  Table {ti+1} ({len(tbl)} rows):")
                            for row in tbl[:5]:
                                lines.append("    | " + " | ".join(str(cell)[:30] for cell in row))
                    if "text" in data:
                        lines.append(f"\nText excerpt:\n{data['text'][:1500]}")
                    return _text("\n".join(lines) or "No data extracted.")
                except Exception as exc:
                    return _text(f"page_extract failed: {exc}")

            # ----------------------------------------------------------------
            # extract_article — clean readable article text from a URL
            # ----------------------------------------------------------------
            if name == "extract_article":
                url = str(args.get("url", "")).strip()
                if not url:
                    return _text("extract_article: 'url' is required")
                max_chars = max(500, int(args.get("max_chars", 8000)))
                try:
                    nav_r = await asyncio.wait_for(
                        c.post(f"{BROWSER_URL}/navigate", json={"url": url}),
                        timeout=25.0,
                    )
                    data = nav_r.json()
                    title     = data.get("title", "")
                    content   = data.get("content", "")
                    final_url = data.get("url", url)
                    clean_lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
                    clean = "\n".join(clean_lines)[:max_chars]
                    header = f"Title: {title}\nURL:   {final_url}\n\n" if title else f"URL: {final_url}\n\n"
                    return _text(header + clean)
                except Exception as exc:
                    return _text(f"extract_article failed: {exc}")

            # ----------------------------------------------------------------
            # page_scrape — scroll + lazy-load + full DOM text extraction
            # ----------------------------------------------------------------
            if name == "page_scrape":
                url         = str(args.get("url", "")).strip()
                max_scrolls = max(1, min(int(args.get("max_scrolls", 10)), 30))
                wait_ms     = max(100, min(int(args.get("wait_ms", 500)), 3000))
                max_chars   = max(500, min(int(args.get("max_chars", 16000)), 64000))
                include_links = bool(args.get("include_links", False))
                payload: dict = {
                    "max_scrolls": max_scrolls,
                    "wait_ms":     wait_ms,
                    "max_chars":   max_chars,
                    "include_links": include_links,
                }
                if url:
                    payload["url"] = url
                try:
                    timeout_s = max_scrolls * (wait_ms / 1000.0) + 30.0
                    scrape_r = await asyncio.wait_for(
                        c.post(f"{BROWSER_URL}/scrape", json=payload),
                        timeout=timeout_s,
                    )
                    data = scrape_r.json()
                except Exception as exc:
                    return _text(f"page_scrape: browser unreachable — {exc}")
                if data.get("error"):
                    return _text(f"page_scrape error: {data['error']}")
                title   = data.get("title", "")
                content = data.get("content", "")
                final_url = data.get("url", url)
                steps   = data.get("scroll_steps", 0)
                grew    = data.get("content_grew_on_scroll", False)
                height  = data.get("final_page_height", 0)
                chars   = data.get("char_count", len(content))
                header = (
                    f"Title: {title}\nURL: {final_url}\n"
                    f"Scrolled: {steps} steps | Page height: {height}px"
                    + (" | lazy content grew" if grew else "")
                    + f" | {chars} chars extracted\n\n"
                )
                lines = [l.strip() for l in content.splitlines() if l.strip()]
                body = "\n".join(lines)
                result_text = header + body
                if include_links and data.get("links"):
                    link_lines = [f"[{lk.get('text','')[:60]}] → {lk.get('href','')[:120]}"
                                  for lk in data["links"][:100]]
                    result_text += f"\n\nLinks ({len(data['links'])}):\n" + "\n".join(link_lines)
                return _text(result_text)

            # ----------------------------------------------------------------
            # page_images — comprehensive image URL extraction from live page
            # ----------------------------------------------------------------
            if name == "page_images":
                url         = str(args.get("url", "")).strip()
                scroll      = bool(args.get("scroll", True))
                max_scrolls = max(1, min(int(args.get("max_scrolls", 3)), 20))
                if not url:
                    return _text("page_images: 'url' is required")
                payload = {"url": url, "scroll": scroll, "max_scrolls": max_scrolls}
                timeout_s = max_scrolls * 2.0 + 30.0
                try:
                    pi_r = await asyncio.wait_for(
                        c.post(f"{BROWSER_URL}/page_images", json=payload),
                        timeout=timeout_s,
                    )
                    data = pi_r.json()
                except Exception as exc:
                    return _text(f"page_images: browser unreachable — {exc}")
                if data.get("error"):
                    return _text(f"page_images error: {data['error']}")
                imgs   = data.get("images", [])
                count  = data.get("count", len(imgs))
                title  = data.get("title", "")
                final_url = data.get("url", url)
                lines = [f"Found {count} images on: {final_url}  ({title})"]
                for img in imgs:
                    line = f"[{img.get('type','?')}] {img['url']}"
                    if img.get("alt"):
                        line += f"  alt={img['alt']!r}"
                    if img.get("natural_w"):
                        line += f"  {img['natural_w']}×{img['natural_h']}"
                    elif img.get("srcset_width"):
                        line += f"  {img['srcset_width']}w"
                    lines.append(line)
                return _text("\n".join(lines))

            # ----------------------------------------------------------------
            # bulk_screenshot — parallel screenshots of multiple URLs
            # ----------------------------------------------------------------
            if name == "bulk_screenshot":
                urls = [str(u).strip() for u in args.get("urls", []) if str(u).strip()]
                if not urls:
                    return _text("bulk_screenshot: 'urls' list is required")
                urls = urls[:6]

                async def _single_shot(shot_url: str, idx: int) -> list[dict[str, Any]]:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    container_path = f"/workspace/bulk_{idx}_{ts}.png"
                    try:
                        shot_r = await c.post(
                            f"{BROWSER_URL}/screenshot",
                            json={"url": shot_url, "path": container_path},
                            timeout=25.0,
                        )
                        shot_data = shot_r.json()
                        cp    = shot_data.get("path", container_path)
                        title = shot_data.get("title", shot_url)
                        fname = os.path.basename(cp)
                        lp    = os.path.join(BROWSER_WORKSPACE, fname)
                        summary = f"[{idx+1}/{len(urls)}] {title}\n{shot_url}\nFile: {fname}"
                        if os.path.isfile(lp):
                            return _image_blocks(cp, summary)
                        return _text(f"[{idx+1}] {shot_url} — screenshot missing: {shot_data.get('error', 'unknown')}")
                    except Exception as exc:
                        return _text(f"[{idx+1}] {shot_url} — failed: {exc}")

                tasks = [_single_shot(u, i) for i, u in enumerate(urls)]
                results = await asyncio.gather(*tasks)
                combined: list[dict[str, Any]] = []
                for blocks in results:
                    combined.extend(blocks)
                return combined

            # ----------------------------------------------------------------
            # scroll_screenshot — full-page capture via scroll + stitch
            # ----------------------------------------------------------------
            if name == "scroll_screenshot":
                if not _HAS_PIL:
                    return _text("scroll_screenshot: Pillow is not installed in this container.")
                url = str(args.get("url", "")).strip() or None
                max_scrolls = max(1, min(int(args.get("max_scrolls", 5)), 10))
                overlap     = max(0, int(args.get("scroll_overlap", 100)))
                if url:
                    try:
                        await asyncio.wait_for(
                            c.post(f"{BROWSER_URL}/navigate", json={"url": url}),
                            timeout=20.0,
                        )
                    except Exception as exc:
                        return _text(f"scroll_screenshot: navigate failed: {exc}")
                try:
                    h_r  = await c.post(f"{BROWSER_URL}/eval",
                                        json={"code": "document.documentElement.scrollHeight"},
                                        timeout=5.0)
                    page_h = int(h_r.json().get("result", 0) or 0)
                    vp_r = await c.post(f"{BROWSER_URL}/eval",
                                        json={"code": "window.innerHeight"},
                                        timeout=5.0)
                    vp_h = int(vp_r.json().get("result", 800) or 800)
                except Exception:
                    page_h, vp_h = 0, 800
                step = max(vp_h - overlap, 100)
                frames: list["_PilImage.Image"] = []
                ts_base = datetime.now().strftime("%Y%m%d_%H%M%S")
                for i in range(max_scrolls):
                    scroll_y = i * step
                    if page_h > 0 and scroll_y >= page_h:
                        break
                    try:
                        await c.post(f"{BROWSER_URL}/eval",
                                     json={"code": f"window.scrollTo(0, {scroll_y})"},
                                     timeout=5.0)
                        await asyncio.sleep(0.3)
                        container_path = f"/workspace/scroll_{i}_{ts_base}.png"
                        shot_r = await c.post(f"{BROWSER_URL}/screenshot",
                                              json={"path": container_path},
                                              timeout=15.0)
                        cp    = shot_r.json().get("path", container_path)
                        fname = os.path.basename(cp)
                        lp    = os.path.join(BROWSER_WORKSPACE, fname)
                        if os.path.isfile(lp):
                            with _PilImage.open(lp) as fr:
                                frames.append(_ImageOps.exif_transpose(fr).convert("RGB").copy())
                    except Exception:
                        break
                if not frames:
                    return _text("scroll_screenshot: no frames captured.")
                total_h = sum(f.height for f in frames)
                max_w   = max(f.width  for f in frames)
                canvas  = _PilImage.new("RGB", (max_w, total_h), (255, 255, 255))
                y = 0
                for fr in frames:
                    canvas.paste(fr, (0, y))
                    y += fr.height
                summary = (
                    f"Full-page scroll screenshot: {len(frames)} frames stitched\n"
                    f"Canvas: {canvas.width}×{canvas.height}"
                    + (f"  URL: {url}" if url else "")
                )
                return _pil_to_blocks(canvas, summary, quality=85, save_prefix="fullpage")

            # ----------------------------------------------------------------
            # image_generate — text-to-image via LM Studio / OpenAI-compatible API
            # ----------------------------------------------------------------
            if name == "image_generate":
                prompt = str(args.get("prompt", "")).strip()
                if not prompt:
                    return _text("image_generate: 'prompt' is required")
                # Fast model-availability check — avoids a 180 s timeout when nothing is loaded
                if not await ModelRegistry.get().has_image_gen(c):
                    loaded = [m.get("id","?") for m in await ModelRegistry.get().models(c)]
                    loaded_str = ", ".join(loaded) if loaded else "none"
                    return _text(
                        f"image_generate: no image-generation model is loaded in LM Studio.\n"
                        f"Currently loaded: {loaded_str}\n"
                        "→ Load FLUX, SDXL, or another diffusion model in LM Studio and try again."
                    )
                model = str(args.get("model", IMAGE_GEN_MODEL)).strip() or None
                size  = str(args.get("size", "512x512")).strip()
                n     = max(1, min(int(args.get("n", 1)), 4))
                neg   = str(args.get("negative_prompt", "")).strip() or None
                steps = args.get("steps")
                gs    = args.get("guidance_scale")
                seed  = args.get("seed")
                payload: dict = {"prompt": prompt, "n": n, "size": size, "response_format": "b64_json"}
                if model:        payload["model"]                = model
                if neg:          payload["negative_prompt"]       = neg
                if steps:        payload["num_inference_steps"]   = int(steps)
                if gs is not None: payload["guidance_scale"]      = float(gs)
                if seed is not None and int(seed) >= 0: payload["seed"] = int(seed)
                gen_url = f"{IMAGE_GEN_BASE_URL}/v1/images/generations"
                try:
                    r = await c.post(gen_url, json=payload, timeout=180.0)
                    r.raise_for_status()
                    data = r.json()
                except Exception as exc:
                    return _text(
                        f"image_generate failed: {exc}\n"
                        f"Endpoint: {gen_url}\n"
                        "→ Make sure an image generation model (FLUX, SDXL, etc.) is loaded in LM Studio.\n"
                        "→ Or set IMAGE_GEN_BASE_URL to point at your Automatic1111/ComfyUI instance."
                    )
                images = data.get("data", [])
                if not images:
                    err = data.get("error", {})
                    msg = err.get("message", str(data)) if isinstance(err, dict) else str(err)
                    return _text(f"image_generate: no images returned — {msg}")
                blocks: list[dict[str, Any]] = []
                for i, img_data in enumerate(images):
                    b64 = img_data.get("b64_json", "")
                    img_url = img_data.get("url", "")
                    revised = img_data.get("revised_prompt", "")
                    save_note = ""
                    if b64 and os.path.isdir(BROWSER_WORKSPACE):
                        try:
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                            fname = f"generated_{i}_{ts}.jpg"
                            lp = os.path.join(BROWSER_WORKSPACE, fname)
                            if _HAS_PIL:
                                with _PilImage.open(_io.BytesIO(base64.b64decode(b64))) as gi:
                                    gi.convert("RGB").save(lp, format="JPEG", quality=95)
                            else:
                                with open(lp, "wb") as fh:
                                    fh.write(base64.b64decode(b64))
                            save_note = f"\n→ Saved as: {fname}  (pass as 'path' in image_crop/zoom/upscale/annotate)"
                        except Exception:
                            pass
                    summary = f"Generated image {i+1}/{n}\nPrompt: {prompt[:200]}"
                    if revised: summary += f"\nRevised: {revised[:150]}"
                    summary += save_note
                    if b64:
                        if _HAS_PIL:
                            try:
                                with _PilImage.open(_io.BytesIO(base64.b64decode(b64))) as gi:
                                    gi = gi.convert("RGB")
                                    if gi.width > 1280 or gi.height > 1280:
                                        gi.thumbnail((1280, 1280), _PilImage.LANCZOS)
                                    buf = _io.BytesIO()
                                    gi.save(buf, format="JPEG", quality=90)
                                b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
                                mime = "image/jpeg"
                            except Exception:
                                mime = "image/png"
                        else:
                            mime = "image/png"
                        blocks.extend([{"type": "text", "text": summary},
                                        {"type": "image", "data": b64, "mimeType": mime}])
                    elif img_url:
                        blocks.append({"type": "text", "text": f"{summary}\nImage URL: {img_url}"})
                return blocks if blocks else _text("image_generate: no image data in response")

            # ----------------------------------------------------------------
            # image_edit — img2img / inpainting via /v1/images/edits
            # ----------------------------------------------------------------
            if name == "image_edit":
                path   = str(args.get("path", "")).strip()
                prompt = str(args.get("prompt", "")).strip()
                if not path:   return _text("image_edit: 'path' is required")
                if not prompt: return _text("image_edit: 'prompt' is required")
                local = _resolve_image_path(path)
                if not local:
                    return _text(f"image_edit: image not found — '{path}' in {BROWSER_WORKSPACE}")
                # Fast model check before spending time on file I/O + 90 s timeout
                if not await ModelRegistry.get().has_image_gen(c):
                    loaded = [m.get("id","?") for m in await ModelRegistry.get().models(c)]
                    loaded_str = ", ".join(loaded) if loaded else "none"
                    return _text(
                        f"image_edit: no image-generation model is loaded in LM Studio.\n"
                        f"Currently loaded: {loaded_str}\n"
                        "→ Load FLUX, SDXL, or another diffusion model and try again."
                    )
                model    = str(args.get("model", IMAGE_GEN_MODEL)).strip() or None
                neg      = str(args.get("negative_prompt", "")).strip() or None
                strength = max(0.0, min(float(args.get("strength", 0.75)), 1.0))
                n        = max(1, min(int(args.get("n", 1)), 4))
                size     = str(args.get("size", "")).strip() or None
                ext  = os.path.splitext(local)[1].lower()
                mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                            ".png": "image/png", ".webp": "image/webp"}
                img_mime = mime_map.get(ext, "image/jpeg")
                with open(local, "rb") as fh:
                    img_bytes = fh.read()
                edit_url = f"{IMAGE_GEN_BASE_URL}/v1/images/edits"
                form: dict = {"prompt": prompt, "n": str(n), "response_format": "b64_json",
                               "strength": str(strength)}
                if model:       form["model"]            = model
                if neg:         form["negative_prompt"]   = neg
                if size:        form["size"]              = size
                try:
                    r = await c.post(edit_url,
                                     files={"image": (os.path.basename(local), img_bytes, img_mime)},
                                     data=form, timeout=180.0)
                    if r.status_code in (404, 405, 422):
                        # Fallback: JSON body with base64-encoded image
                        b64_in = base64.standard_b64encode(img_bytes).decode("ascii")
                        json_pl: dict = {"prompt": prompt, "n": n, "response_format": "b64_json",
                                          "strength": strength,
                                          "image": f"data:{img_mime};base64,{b64_in}"}
                        if model: json_pl["model"] = model
                        if neg:   json_pl["negative_prompt"] = neg
                        if size:  json_pl["size"] = size
                        r = await c.post(edit_url, json=json_pl, timeout=180.0)
                    r.raise_for_status()
                    data = r.json()
                except Exception as exc:
                    return _text(
                        f"image_edit failed: {exc}\n"
                        f"Endpoint: {edit_url}\n"
                        "→ img2img support varies by model. Make sure a compatible model is loaded.\n"
                        "→ For full img2img support, point IMAGE_GEN_BASE_URL at Automatic1111/ComfyUI."
                    )
                images = data.get("data", [])
                if not images:
                    err = data.get("error", {})
                    msg = err.get("message", str(data)) if isinstance(err, dict) else str(err)
                    return _text(f"image_edit: no images returned — {msg}")
                blocks_e: list[dict[str, Any]] = []
                for i, img_data in enumerate(images):
                    b64 = img_data.get("b64_json", "")
                    revised = img_data.get("revised_prompt", "")
                    save_note = ""
                    if b64 and os.path.isdir(BROWSER_WORKSPACE):
                        try:
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                            fname = f"edited_{i}_{ts}.jpg"
                            lp = os.path.join(BROWSER_WORKSPACE, fname)
                            if _HAS_PIL:
                                with _PilImage.open(_io.BytesIO(base64.b64decode(b64))) as ei:
                                    ei.convert("RGB").save(lp, format="JPEG", quality=95)
                            else:
                                with open(lp, "wb") as fh:
                                    fh.write(base64.b64decode(b64))
                            save_note = f"\n→ Saved as: {fname}"
                        except Exception:
                            pass
                    summary = (f"Edited image {i+1}/{n}  Source: {os.path.basename(path)}\n"
                               f"Prompt: {prompt[:200]}  strength={strength:.2f}")
                    if revised: summary += f"\nRevised: {revised[:150]}"
                    summary += save_note
                    if b64:
                        if _HAS_PIL:
                            try:
                                with _PilImage.open(_io.BytesIO(base64.b64decode(b64))) as ei:
                                    ei = ei.convert("RGB")
                                    if ei.width > 1280 or ei.height > 1280:
                                        ei.thumbnail((1280, 1280), _PilImage.LANCZOS)
                                    buf = _io.BytesIO()
                                    ei.save(buf, format="JPEG", quality=90)
                                b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
                                mime = "image/jpeg"
                            except Exception:
                                mime = "image/png"
                        else:
                            mime = "image/png"
                        blocks_e.extend([{"type": "text", "text": summary},
                                          {"type": "image", "data": b64, "mimeType": mime}])
                return blocks_e if blocks_e else _text("image_edit: no image data in response")

            # ----------------------------------------------------------------
            # image_remix — GPU AI style transfer via /v1/images/edits
            #               OOP: ModelRegistry (model check) + ImageRenderer (encode)
            # ----------------------------------------------------------------
            if name == "image_remix":
                if not _HAS_PIL:
                    return _text("image_remix: Pillow is not installed.")
                path_rm   = str(args.get("path", "")).strip()
                prompt_rm = str(args.get("prompt", "")).strip()
                if not path_rm:
                    return _text("image_remix: 'path' is required")
                if not prompt_rm:
                    return _text("image_remix: 'prompt' is required")
                local_rm = _resolve_image_path(path_rm)
                if not local_rm:
                    return _text(f"image_remix: image not found — '{path_rm}' in {BROWSER_WORKSPACE}")
                strength_rm = max(0.1, min(float(args.get("strength", 0.65)), 1.0))
                n_rm        = max(1, min(int(args.get("n", 1)), 4))
                # Fast model check — friendly error instead of 90 s timeout
                if not await ModelRegistry.get().has_image_gen(c):
                    loaded_rm = [m.get("id","?") for m in await ModelRegistry.get().models(c)]
                    return _text(
                        f"image_remix: no image-generation model loaded in LM Studio.\n"
                        f"Currently loaded: {', '.join(loaded_rm) or 'none'}\n"
                        "→ Load FLUX, SDXL, or another diffusion model and try again."
                    )
                # Encode source image as JPEG for multipart upload
                with _PilImage.open(local_rm) as src_rm:
                    img_rm = _ImageOps.exif_transpose(src_rm).convert("RGB")
                in_buf_rm = _io.BytesIO()
                img_rm.save(in_buf_rm, format="JPEG", quality=92)
                in_buf_rm.seek(0)
                rm_model  = IMAGE_GEN_MODEL.strip() or None
                form_rm: dict[str, str] = {
                    "prompt":          prompt_rm,
                    "n":               str(n_rm),
                    "response_format": "b64_json",
                    "strength":        str(strength_rm),
                }
                if rm_model:
                    form_rm["model"] = rm_model
                try:
                    resp_rm = await asyncio.wait_for(
                        c.post(
                            f"{IMAGE_GEN_BASE_URL}/v1/images/edits",
                            files={"image": ("source.jpg", in_buf_rm, "image/jpeg")},
                            data=form_rm,
                            timeout=120.0,
                        ),
                        timeout=125.0,
                    )
                    resp_rm.raise_for_status()
                    data_rm = resp_rm.json()
                except Exception as exc_rm:
                    return _text(
                        f"image_remix failed: {exc_rm}\n"
                        "→ Ensure an image-generation model supports /v1/images/edits in LM Studio."
                    )
                blocks_rm: list[dict[str, Any]] = []
                src_name_rm = os.path.basename(path_rm)
                for idx_rm, item_rm in enumerate(data_rm.get("data", [])):
                    b64_rm = item_rm.get("b64_json", "")
                    if not b64_rm:
                        continue
                    try:
                        ri_rm = _PilImage.open(_io.BytesIO(base64.b64decode(b64_rm))).convert("RGB")
                        summary_rm = (
                            f"Remix [{idx_rm+1}/{n_rm}]: '{prompt_rm[:80]}'\n"
                            f"Source: {src_name_rm}  {img_rm.width}×{img_rm.height}"
                            f"  strength={strength_rm:.2f}  GPU: {GpuDetector.name()}"
                        )
                        blocks_rm.extend(_renderer.encode(ri_rm, summary_rm, save_prefix="remix"))
                    except Exception:
                        continue
                return blocks_rm if blocks_rm else _text(
                    f"image_remix: no image data in response for prompt '{prompt_rm[:60]}'"
                )

            # ----------------------------------------------------------------
            # image_upscale — GPU AI primary (NVIDIA / Intel Arc via LM Studio);
            #                  CPU LANCZOS is fallback only if GPU is unavailable
            #                  or the model call fails.
            #                  OOP via GpuDetector + GpuUpscaler + ImageRenderer.
            # ----------------------------------------------------------------
            if name == "image_upscale":
                if not _HAS_PIL:
                    return _text("image_upscale: Pillow is not installed.")
                path = str(args.get("path", "")).strip()
                if not path:
                    return _text("image_upscale: 'path' is required")
                local = _resolve_image_path(path)
                if not local:
                    return _text(f"image_upscale: image not found — '{path}' in {BROWSER_WORKSPACE}")
                try:
                    scale = max(1.1, min(float(args.get("scale", 2.0)), 8.0))
                except (ValueError, TypeError):
                    scale = 2.0
                # sharpen: accept bool or string "false"/"true"
                _sharpen_raw = args.get("sharpen", True)
                sharpen = not (str(_sharpen_raw).lower() in ("false", "0", "no"))
                # gpu param: None/unset → auto (GPU primary), False → force CPU only
                gpu_arg = args.get("gpu", None)
                force_cpu = (gpu_arg is False or str(gpu_arg).lower() == "false")
                with _PilImage.open(local) as src:
                    img = _ImageOps.exif_transpose(src).convert("RGB")
                ow, oh = img.size
                # ── PRIMARY: GPU AI upscale (NVIDIA + Intel Arc via LM Studio) ──
                if not force_cpu and IMAGE_GEN_BASE_URL:
                    gpu_info  = GpuDetector.detect()
                    # Fast check: skip 90 s timeout if no image-gen model is loaded
                    _has_gen  = await ModelRegistry.get().has_image_gen(c)
                    upscaler  = GpuUpscaler(IMAGE_GEN_BASE_URL, IMAGE_GEN_MODEL)
                    ai_result = await upscaler.upscale(img, c) if _has_gen else None
                    if ai_result is not None:
                        ai_w, ai_h = ai_result.size
                        summary = (
                            f"GPU AI upscale ({gpu_info['name']})\n"
                            f"  {ow}×{oh} → {ai_w}×{ai_h}\n"
                            f"Source: {os.path.basename(path)}"
                        )
                        return _renderer.encode(ai_result, summary, save_prefix="upscaled",
                                                max_w=4096, max_h=4096)
                    # GPU call failed — warn and fall through to CPU LANCZOS
                    cpu_note = (
                        f"⚠ GPU AI upscale failed (no model loaded or LM Studio unreachable). "
                        f"GPU detected: {gpu_info['name']}. "
                        f"Falling back to CPU LANCZOS.\n"
                    )
                else:
                    cpu_note = ""
                # ── FALLBACK (CPU): LANCZOS ──────────────────────────────────────
                # Safety cap: clamp scale so no output dimension exceeds 8192 px
                max_dim = max(ow, oh)
                if max_dim * scale > 8192:
                    scale = 8192 / max_dim
                    capped = True
                else:
                    capped = False
                nw, nh = int(ow * scale), int(oh * scale)
                upscaled = img.resize((nw, nh), _PilImage.LANCZOS)
                if sharpen:
                    upscaled = _ImageEnhance.Sharpness(upscaled).enhance(1.4)
                    upscaled = upscaled.filter(_ImageFilter.UnsharpMask(radius=0.5, percent=80, threshold=2))
                summary = (
                    cpu_note
                    + f"CPU LANCZOS upscale {scale:.2f}×  {ow}×{oh} → {nw}×{nh}"
                    + (" + sharpen" if sharpen else "")
                    + (" [scale capped to 8192px]" if capped else "")
                    + f"\nSource: {os.path.basename(path)}"
                )
                return _renderer.encode(upscaled, summary, save_prefix="upscaled",
                                        max_w=4096, max_h=4096)

            # ----------------------------------------------------------------
            if name == "browser_save_images":
                raw_urls = args.get("urls", [])
                if isinstance(raw_urls, str):
                    raw_urls = [u.strip() for u in raw_urls.split(",") if u.strip()]
                if not raw_urls:
                    return _text("browser_save_images: 'urls' is required (list or comma-separated string)")
                prefix = str(args.get("prefix", "image")).strip() or "image"
                max_imgs = int(args.get("max", 20))
                try:
                    save_r = await c.post(
                        f"{BROWSER_URL}/save_images",
                        json={"urls": raw_urls, "prefix": prefix, "max": max_imgs},
                        timeout=120.0,
                    )
                    save_data = save_r.json()
                except Exception as exc:
                    return _text(f"browser_save_images: browser unreachable — {exc}")
                saved = save_data.get("saved", [])
                errors = save_data.get("errors", [])
                if not saved and errors:
                    errs = "; ".join(e.get("error", "?") for e in errors[:3])
                    return _text(f"browser_save_images: all downloads failed — {errs}")
                blocks: list = []
                lines = [f"Downloaded {len(saved)} image(s)" +
                         (f"  ({len(errors)} failed)" if errors else "")]
                for item in saved:
                    fname = os.path.basename(item.get("path", ""))
                    host_path = f"/docker/human_browser/workspace/{fname}"
                    local_path = os.path.join(BROWSER_WORKSPACE, fname)
                    size_kb = item.get("size", 0) // 1024
                    lines.append(f"  [{item.get('index','?')}] {fname}  {size_kb} KB  {item.get('url','')[:80]}")
                    if os.path.isfile(local_path) and len(blocks) < 6:
                        blocks.extend(_image_blocks(item["path"], f"[{item.get('index','?')}] {fname}"))
                blocks.insert(0, {"type": "text", "text": "\n".join(lines)})
                return blocks

            # ----------------------------------------------------------------
            if name == "browser_download_page_images":
                url = str(args.get("url", "")).strip() or None
                if url:
                    try:
                        await c.post(f"{BROWSER_URL}/navigate", json={"url": url}, timeout=20.0)
                    except Exception as exc:
                        return _text(f"browser_download_page_images: navigate failed — {exc}")
                filter_q = str(args.get("filter", "")).strip() or None
                prefix = str(args.get("prefix", "image")).strip() or "image"
                max_imgs = int(args.get("max", 20))
                payload: dict = {"max": max_imgs, "prefix": prefix}
                if filter_q:
                    payload["filter"] = filter_q
                try:
                    dl_r = await c.post(
                        f"{BROWSER_URL}/download_page_images",
                        json=payload,
                        timeout=120.0,
                    )
                    dl_data = dl_r.json()
                except Exception as exc:
                    return _text(f"browser_download_page_images: browser unreachable — {exc}")
                saved = dl_data.get("saved", [])
                errors = dl_data.get("errors", [])
                applied_filter = dl_data.get("filter")
                filter_note = f" (filter: '{applied_filter}')" if applied_filter else ""
                if not saved and errors:
                    errs = "; ".join(e.get("error", "?") for e in errors[:3])
                    return _text(f"browser_download_page_images: all downloads failed{filter_note} — {errs}")
                if not saved:
                    return _text(f"browser_download_page_images: no images found on page{filter_note}")
                blocks = []
                lines = [
                    f"Downloaded {len(saved)} image(s){filter_note}" +
                    (f"  ({len(errors)} failed)" if errors else "")
                ]
                for item in saved:
                    fname = os.path.basename(item.get("path", ""))
                    local_path = os.path.join(BROWSER_WORKSPACE, fname)
                    size_kb = item.get("size", 0) // 1024
                    lines.append(f"  [{item.get('index','?')}] {fname}  {size_kb} KB  {item.get('url','')[:80]}")
                    if os.path.isfile(local_path) and len(blocks) < 9:
                        blocks.extend(_image_blocks(item["path"], f"[{item.get('index','?')}] {fname}"))
                blocks.insert(0, {"type": "text", "text": "\n".join(lines)})
                return blocks

            # ----------------------------------------------------------------
            # image_search — find images by query; return multiple inline
            # ----------------------------------------------------------------
            if name == "image_search":
                query      = str(args.get("query", "")).strip()
                img_count  = max(1, min(int(args.get("count", 4)), 20))
                img_offset = max(0, int(args.get("offset", 0)))
                if not query:
                    return _text("image_search: 'query' is required")

                import hashlib as _hl2
                import json as _js2
                from urllib.parse import quote_plus as _qp, urlparse as _up2, \
                    parse_qs as _pqs2, unquote as _uq2
                qwords = {w for w in query.lower().split() if len(w) > 2}

                _GOOD_D  = ("wikia.nocookie.net", "imgur.com", "redd.it",
                            "prydwen.gg", "fandom.com", "iopwiki.com", "cdn.")
                _SKIP_P  = ("/16px-", "/25px-", "/32px-", "/48px-", "favicon",
                            "logo", "icon", "avatar", "pixel.gif", "button",
                            "ytimg.com", "yt3.ggpht")
                _SKIP_T1 = ("youtube.com", "youtu.be", "vimeo.com", "dailymotion.com",
                            "twitch.tv", "pinterest.com", "instagram.com",
                            "deviantart.com", "artstation.com")

                # ── Seen-URL dedup via memory service ─────────────────────────
                _qhash        = _hl2.md5(query.lower().encode()).hexdigest()[:16]
                _mem_key      = f"imgsr:{_qhash}"
                _seen_urls:   set[str]  = set()
                _seen_hashes: list[str] = []   # intra-call phash dedup
                _url_to_hash: dict[str, str] = {}
                try:
                    mem_r = await asyncio.wait_for(
                        c.get(f"{MEMORY_URL}/recall", params={"key": _mem_key}),
                        timeout=5.0,
                    )
                    rdata = mem_r.json()
                    entries = rdata.get("entries") or []
                    if rdata.get("found") and entries:
                        _seen_urls = set(_js2.loads(entries[0]["value"]))
                except Exception:
                    pass

                _norm_subject = query.lower().strip()

                # ── Query expansion ────────────────────────────────────────────
                _EXPANSIONS = {
                    "gfl2": "Girls Frontline 2", "gfl": "Girls Frontline",
                    "hsr":  "Honkai Star Rail",  "gi":  "Genshin Impact",
                    "hk":   "Honkai Impact",
                }

                def _expand_queries(q: str) -> list[str]:
                    variants = [q]
                    q_lower  = q.lower()
                    for short, full in _EXPANSIONS.items():
                        if short in q_lower and full.lower() not in q_lower:
                            variants.append(q.replace(short, full).replace(short.upper(), full))
                            break
                    if "artwork" not in q_lower and "art" not in q_lower:
                        variants.append(q + " artwork")
                    return variants[:3]

                def _unwrap_thumb(url: str) -> str:
                    if "/images/thumb/" in url:
                        no_thumb = url.replace("/images/thumb/", "/images/", 1)
                        m = re.search(r"/\d+px-[^?]*", no_thumb)
                        if m:
                            return no_thumb[:m.start()]
                    return url

                def _img_referer(url: str) -> str:
                    u = url.lower()
                    if "pbs.twimg.com" in u or "media.twimg.com" in u:
                        return "https://twitter.com/"
                    if "wikia.nocookie.net" in u or "static.fandom.com" in u:
                        return "https://www.fandom.com/"
                    if "iopwiki.com" in u:
                        return "https://iopwiki.com/"
                    if "prydwen.gg" in u:
                        return "https://www.prydwen.gg/"
                    return "https://www.google.com/"

                def _score(img: dict) -> int:
                    url_l = img.get("url", "").lower()
                    alt_l = img.get("alt", "").lower()
                    if any(p in url_l for p in _SKIP_P):
                        return -999
                    s  = sum(2 for w in qwords if w in url_l)
                    s += sum(5 for w in qwords if w in alt_l)
                    if "pbs.twimg.com" in url_l or "media.twimg.com" in url_l:
                        s += 10
                    else:
                        s += sum(6 for d in _GOOD_D if d in url_l)
                    if img.get("type") in ("srcset", "picture"):
                        s += 2
                    nw = img.get("natural_w", 0)
                    if nw >= 500:
                        s += 4
                    elif nw >= 300:
                        s += 2
                    return s

                def _domain_of(url: str) -> str:
                    try:
                        h = _up2(url).hostname or ""
                        return h.removeprefix("www.")
                    except Exception:
                        return url

                def _apply_domain_cap(candidates: list[dict], max_per_domain: int = 2) -> list[dict]:
                    counts: dict[str, int] = {}
                    result = []
                    for cand in candidates:
                        d = _domain_of(cand.get("url", ""))
                        if counts.get(d, 0) < max_per_domain:
                            result.append(cand)
                            counts[d] = counts.get(d, 0) + 1
                    return result

                async def _fetch_render(url: str) -> list[dict] | None:
                    """Fetch, resize, phash-dedup, and encode one image URL."""
                    if not url or not url.startswith("http"):
                        return None
                    url  = _unwrap_thumb(url)
                    hdrs = {
                        "User-Agent":      _BROWSER_HEADERS["User-Agent"],
                        "Accept":          "image/webp,image/apng,image/*,*/*;q=0.8",
                        "Accept-Language": _BROWSER_HEADERS["Accept-Language"],
                        "Referer":         _img_referer(url),
                    }
                    try:
                        img_r = await asyncio.wait_for(
                            c.get(url, headers=hdrs, follow_redirects=True),
                            timeout=12.0,
                        )
                        if img_r.status_code != 200:
                            return None
                        ct = img_r.headers.get("content-type", "").split(";")[0].strip()
                        if not ct.startswith("image/"):
                            return None
                        raw = img_r.content
                        if len(raw) < 10_240:
                            return None
                        if _HAS_PIL:
                            try:
                                with _PilImage.open(_io.BytesIO(raw)) as pil:
                                    if pil.width < 150 or pil.height < 150:
                                        return None
                                    pil = pil.convert("RGB")
                                    # dHash intra-call dedup: skip visually identical images
                                    img_hash = _dhash(pil)
                                    if img_hash and any(
                                        _hamming(img_hash, h) < 8 for h in _seen_hashes
                                    ):
                                        return None
                                    if img_hash:
                                        _seen_hashes.append(img_hash)
                                        _url_to_hash[url] = img_hash
                                    if pil.width > 1280 or pil.height > 1024:
                                        pil.thumbnail((1280, 1024), _PilImage.LANCZOS)
                                    buf = _io.BytesIO()
                                    pil.save(buf, format="JPEG", quality=85)
                                    raw, ct = buf.getvalue(), "image/jpeg"
                            except Exception:
                                pass
                        b64 = base64.standard_b64encode(raw).decode("ascii")
                        return [
                            {"type": "text",  "text": f"Image: {url}\nQuery: {query}"},
                            {"type": "image", "data": b64, "mimeType": ct},
                        ]
                    except Exception:
                        return None

                # ── DB-first: return confirmed images if we have enough cached ───
                try:
                    db_r = await asyncio.wait_for(
                        c.get(f"{DATABASE_URL}/images/search",
                              params={"subject": _norm_subject, "limit": img_count * 2}),
                        timeout=3.0,
                    )
                    db_imgs = db_r.json().get("images", [])
                    if len(db_imgs) >= img_count:
                        db_blocks: list[dict] = []
                        for dbi in db_imgs:
                            fb = await _fetch_render(dbi["url"])
                            if fb:
                                db_blocks.extend(fb)
                            if len(db_blocks) >= img_count * 2:
                                break
                        if db_blocks:
                            return db_blocks
                except Exception:
                    pass

                # ── Collect ALL candidates across all query variants + both tiers ──
                all_candidates: list[dict] = []
                _ddg_hosts = ("duckduckgo.com", "ddg.gg", "duck.co")

                # Tier 1: DDG web search → page_images on top article pages
                for variant_q in _expand_queries(query):
                    try:
                        sr = await asyncio.wait_for(
                            c.get(
                                f"https://html.duckduckgo.com/html/?q={_qp(variant_q)}&kp=-2",
                                headers=_BROWSER_HEADERS,
                                follow_redirects=True,
                            ),
                            timeout=12.0,
                        )
                        raw_enc = re.findall(r'uddg=(https?%3A[^&"\'>\s]+)', sr.text)
                        t1_urls: list[str] = []
                        t1_seen: set[str]  = set()
                        for enc in raw_enc:
                            dec = _url_unquote(enc)
                            if any(d in dec for d in _ddg_hosts):
                                continue
                            if dec not in t1_seen:
                                t1_seen.add(dec)
                                t1_urls.append(dec)
                        t1_urls = [u for u in t1_urls if not any(d in u for d in _SKIP_T1)]
                        for page_url in t1_urls[:5]:
                            try:
                                pi_r = await asyncio.wait_for(
                                    c.post(f"{BROWSER_URL}/page_images",
                                           json={"url": page_url, "scroll": True, "max_scrolls": 1}),
                                    timeout=35.0,
                                )
                                page_imgs = pi_r.json().get("images", [])
                                if len(page_imgs) < 3:
                                    continue
                                all_candidates.extend(page_imgs)
                            except Exception:
                                continue
                    except Exception:
                        pass

                # Tier 2: DDG image-search page → decode proxy URLs
                try:
                    t2_url = f"https://duckduckgo.com/?q={_qp(query)}&iax=images&ia=images&kp=-2"
                    pi2_r = await asyncio.wait_for(
                        c.post(f"{BROWSER_URL}/page_images",
                               json={"url": t2_url, "scroll": True, "max_scrolls": 3}),
                        timeout=45.0,
                    )
                    for img2 in pi2_r.json().get("images", []):
                        u2 = img2.get("url", "")
                        if "external-content.duckduckgo.com" in u2:
                            try:
                                params2 = _pqs2(_up2(u2).query)
                                orig2   = _uq2(params2.get("u", [""])[0])
                                if orig2.startswith("http"):
                                    all_candidates.append({**img2, "url": orig2})
                                    continue
                            except Exception:
                                pass
                        all_candidates.append(img2)
                except Exception:
                    pass

                # Tier 3: Bing Images — different corpus from DDG, different CDNs
                try:
                    bing_url = f"https://www.bing.com/images/search?q={_qp(query)}&form=HDRSC2&first=1"
                    pi3_r = await asyncio.wait_for(
                        c.post(f"{BROWSER_URL}/page_images",
                               json={"url": bing_url, "scroll": True, "max_scrolls": 2}),
                        timeout=45.0,
                    )
                    for img3 in pi3_r.json().get("images", []):
                        u3 = img3.get("url", "")
                        # Bing wraps images in /th?id=... proxy URLs — skip thumbnails
                        if "bing.com/th" in u3 or "tse1.mm.bing" in u3:
                            continue
                        all_candidates.append(img3)
                except Exception:
                    pass

                # ── Score, dedup seen URLs + intra-call dupes, domain-cap, offset ──
                _intra_seen: set[str] = set()
                fresh: list[dict] = []
                for cand in all_candidates:
                    u = _unwrap_thumb(cand.get("url", ""))
                    if _score(cand) >= 0 and u not in _seen_urls and u not in _intra_seen:
                        fresh.append(cand)
                        _intra_seen.add(u)
                sorted_cands = _apply_domain_cap(
                    sorted(fresh, key=_score, reverse=True), max_per_domain=2
                )
                sorted_cands = sorted_cands[img_offset:]
                fetch_pool   = sorted_cands[:img_count * 3]

                if not fetch_pool:
                    return _text(
                        f"image_search: no image found for '{query}'.\n"
                        "Try: page_images on a specific URL, or refine the query."
                    )

                fetch_results = await asyncio.gather(
                    *[_fetch_render(cand["url"]) for cand in fetch_pool],
                    return_exceptions=True,
                )

                # ── Vision confirm pass (best-effort, parallel) ────────────────
                async def _maybe_confirm(
                    blocks: list[dict] | None,
                    url: str,
                    img_hash: str,
                ) -> list[dict] | None:
                    if not blocks:
                        return None
                    img_b64 = next(
                        (b["data"] for b in blocks if b.get("type") == "image"), ""
                    )
                    if not img_b64:
                        return blocks
                    is_match, desc, conf = await _vision_confirm(img_b64, query, c, phash=img_hash)
                    if not is_match:
                        return None
                    # Fire-and-forget: persist confirmed image to DB
                    asyncio.create_task(c.post(
                        f"{DATABASE_URL}/images/store",
                        json={
                            "url": url, "subject": _norm_subject,
                            "description": desc, "phash": img_hash,
                            "quality_score": conf,
                        },
                    ))
                    return blocks

                confirmed_results = await asyncio.gather(
                    *[
                        _maybe_confirm(
                            res if not isinstance(res, Exception) else None,
                            _unwrap_thumb(cand["url"]),
                            _url_to_hash.get(_unwrap_thumb(cand["url"]), ""),
                        )
                        for cand, res in zip(fetch_pool, fetch_results)
                    ],
                    return_exceptions=True,
                )

                output_blocks: list[dict] = []
                returned_urls: list[str]  = []
                for cand, res in zip(fetch_pool, confirmed_results):
                    if isinstance(res, Exception) or res is None:
                        continue
                    output_blocks.extend(res)
                    returned_urls.append(_unwrap_thumb(cand["url"]))
                    if len(returned_urls) >= img_count:
                        break

                # ── Persist seen URLs for next call ────────────────────────────
                if returned_urls:
                    try:
                        merged = list(_seen_urls | set(returned_urls))
                        await asyncio.wait_for(
                            c.post(f"{MEMORY_URL}/store",
                                   json={"key": _mem_key, "value": _js2.dumps(merged),
                                         "ttl_seconds": 3600}),
                            timeout=5.0,
                        )
                    except Exception:
                        pass

                if output_blocks:
                    return output_blocks
                return _text(
                    f"image_search: no image found for '{query}'.\n"
                    "Try: page_images on a specific URL, or refine the query."
                )

            # ----------------------------------------------------------------
            # tts — text-to-speech via LM Studio /v1/audio/speech
            # ----------------------------------------------------------------
            if name == "tts":
                import tempfile as _tmpfile, os as _os_tts
                text_in  = str(args.get("text", "")).strip()
                if not text_in:
                    return _text("tts: 'text' is required")
                voice  = str(args.get("voice", "alloy")).strip() or "alloy"
                speed  = max(0.25, min(4.0, float(args.get("speed", 1.0))))
                fmt    = str(args.get("format", "mp3")).strip() or "mp3"
                payload_tts: dict = {
                    "input": text_in,
                    "voice": voice,
                    "speed": speed,
                    "response_format": fmt,
                }
                if IMAGE_GEN_MODEL:
                    payload_tts["model"] = IMAGE_GEN_MODEL
                async with httpx.AsyncClient(timeout=60.0) as hc_tts:
                    try:
                        r_tts = await hc_tts.post(
                            f"{IMAGE_GEN_BASE_URL}/v1/audio/speech",
                            json=payload_tts,
                        )
                        if r_tts.status_code != 200:
                            return _text(
                                f"tts: LM Studio returned {r_tts.status_code}. "
                                "Make sure a TTS-capable model is loaded."
                            )
                        audio_bytes = r_tts.content
                    except Exception as exc:
                        return _text(f"tts: request failed — {exc}")
                # Save to workspace
                ts_tts  = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname   = f"tts_{ts_tts}.{fmt}"
                ws_path = _os_tts.path.join(BROWSER_WORKSPACE, fname)
                try:
                    with open(ws_path, "wb") as fh:
                        fh.write(audio_bytes)
                    return _text(
                        f"tts: saved {len(audio_bytes):,} bytes → {ws_path}\n"
                        f"voice={voice} speed={speed} format={fmt}"
                    )
                except Exception as exc:
                    return _text(f"tts: audio generated ({len(audio_bytes):,} bytes) but could not save: {exc}")

            # ----------------------------------------------------------------
            # embed_store — embed text + persist to DB
            # ----------------------------------------------------------------
            if name == "embed_store":
                import json as _js_emb
                key_e    = str(args.get("key", "")).strip()
                content_e = str(args.get("content", "")).strip()
                topic_e  = str(args.get("topic", "")).strip()
                if not key_e:     return _text("embed_store: 'key' is required")
                if not content_e: return _text("embed_store: 'content' is required")
                emb_payload = {"input": [content_e]}
                if IMAGE_GEN_MODEL:
                    emb_payload["model"] = IMAGE_GEN_MODEL
                async with httpx.AsyncClient(timeout=30.0) as hc_emb:
                    try:
                        r_emb = await hc_emb.post(
                            f"{IMAGE_GEN_BASE_URL}/v1/embeddings", json=emb_payload
                        )
                        r_emb.raise_for_status()
                        emb_data = r_emb.json().get("data", [])
                        if not emb_data:
                            return _text("embed_store: LM Studio returned empty embedding data — is an embedding model loaded?")
                        embedding = emb_data[0].get("embedding", [])
                        if not embedding:
                            return _text("embed_store: LM Studio returned empty embedding vector")
                    except Exception as exc:
                        return _text(f"embed_store: LM Studio embedding failed — {exc}")
                    # Store in DB
                    try:
                        r_db = await hc_emb.post(
                            f"{DATABASE_URL}/embeddings/store",
                            json={"key": key_e, "content": content_e,
                                  "embedding": embedding,
                                  "model": IMAGE_GEN_MODEL, "topic": topic_e},
                        )
                        r_db.raise_for_status()
                        return _text(
                            f"embed_store: stored key='{key_e}' "
                            f"dims={len(embedding)} topic='{topic_e}'"
                        )
                    except Exception as exc:
                        return _text(f"embed_store: DB store failed — {exc}")

            # ----------------------------------------------------------------
            # embed_search — semantic search over stored embeddings
            # ----------------------------------------------------------------
            if name == "embed_search":
                import json as _js_es
                query_es  = str(args.get("query", "")).strip()
                try:
                    limit_es = max(1, min(20, int(args.get("limit", 5))))
                except (ValueError, TypeError):
                    return _text("embed_search: 'limit' must be an integer")
                topic_es  = str(args.get("topic", "")).strip()
                if not query_es: return _text("embed_search: 'query' is required")
                emb_payload_s = {"input": [query_es]}
                if IMAGE_GEN_MODEL:
                    emb_payload_s["model"] = IMAGE_GEN_MODEL
                async with httpx.AsyncClient(timeout=30.0) as hc_es:
                    try:
                        r_es = await hc_es.post(
                            f"{IMAGE_GEN_BASE_URL}/v1/embeddings", json=emb_payload_s
                        )
                        r_es.raise_for_status()
                        es_data = r_es.json().get("data", [])
                        if not es_data:
                            return _text("embed_search: LM Studio returned empty embedding data — is an embedding model loaded?")
                        q_vec = es_data[0].get("embedding", [])
                        if not q_vec:
                            return _text("embed_search: LM Studio returned empty embedding vector")
                    except Exception as exc:
                        return _text(f"embed_search: LM Studio embedding failed — {exc}")
                    body_es: dict = {"embedding": q_vec, "limit": limit_es}
                    if topic_es:
                        body_es["topic"] = topic_es
                    try:
                        r_db_es = await hc_es.post(
                            f"{DATABASE_URL}/embeddings/search", json=body_es
                        )
                        r_db_es.raise_for_status()
                        results_es = r_db_es.json().get("results", [])
                    except Exception as exc:
                        return _text(f"embed_search: DB search failed — {exc}")
                if not results_es:
                    return _text(f"embed_search: no results found for '{query_es}'")
                lines = [f"embed_search: {len(results_es)} result(s) for '{query_es}'\n"]
                for i, r in enumerate(results_es, 1):
                    lines.append(
                        f"{i}. [{r.get('similarity', 0):.3f}] key={r.get('key','')} "
                        f"topic={r.get('topic','')}\n   {r.get('content','')[:200]}"
                    )
                return _text("\n".join(lines))

            # ----------------------------------------------------------------
            # code_run — sandboxed Python subprocess execution
            # ----------------------------------------------------------------
            if name == "code_run":
                import tempfile as _tmpfile_cr, os as _os_cr, time as _time_cr
                code_cr    = str(args.get("code", "")).strip()
                pkgs_cr    = args.get("packages") or []
                timeout_cr = max(1, min(120, int(args.get("timeout", 30))))
                if not code_cr:
                    return _text("code_run: 'code' is required")
                # Auto-inject DEVICE variable for torch/tensorflow/cuda code
                code_cr = GpuCodeRuntime.prepare(code_cr)
                install_log_cr: list[str] = []
                if pkgs_cr:
                    for pkg in pkgs_cr:
                        pkg = str(pkg).strip()
                        if not pkg:
                            continue
                        try:
                            proc_pip = await asyncio.create_subprocess_exec(
                                "python3", "-m", "pip", "install", "--quiet", pkg,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.STDOUT,
                            )
                            out_pip, _ = await asyncio.wait_for(
                                proc_pip.communicate(), timeout=15.0
                            )
                            install_log_cr.append(
                                f"pip install {pkg}: exit {proc_pip.returncode}"
                            )
                        except asyncio.TimeoutError:
                            install_log_cr.append(f"pip install {pkg}: timed out")
                        except Exception as exc:
                            install_log_cr.append(f"pip install {pkg}: {exc}")
                # Write code to tempfile
                import sys as _sys_cr
                with _tmpfile_cr.NamedTemporaryFile(
                    suffix=".py", mode="w", encoding="utf-8", delete=False
                ) as _tf:
                    _tf.write(code_cr)
                    tmp_path_cr = _tf.name
                t0_cr = _time_cr.monotonic()
                try:
                    proc_cr = await asyncio.create_subprocess_exec(
                        _sys_cr.executable, tmp_path_cr,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    try:
                        stdout_b, stderr_b = await asyncio.wait_for(
                            proc_cr.communicate(), timeout=float(timeout_cr)
                        )
                        exit_code_cr = proc_cr.returncode
                    except asyncio.TimeoutError:
                        proc_cr.kill()
                        await proc_cr.wait()
                        return _text(
                            f"code_run: timed out after {timeout_cr}s\n"
                            + ("\nInstall log:\n" + "\n".join(install_log_cr) if install_log_cr else "")
                        )
                    duration_ms = int((_time_cr.monotonic() - t0_cr) * 1000)
                    stdout_s = stdout_b.decode(errors="replace")
                    stderr_s = stderr_b.decode(errors="replace")
                    parts = [f"code_run: exit={exit_code_cr} duration={duration_ms}ms"]
                    if stdout_s.strip():
                        parts.append(f"stdout:\n{stdout_s.strip()}")
                    if stderr_s.strip():
                        parts.append(f"stderr:\n{stderr_s.strip()}")
                    if install_log_cr:
                        parts.append("install_log:\n" + "\n".join(install_log_cr))
                    return _text("\n\n".join(parts))
                finally:
                    try:
                        _os_cr.unlink(tmp_path_cr)
                    except OSError:
                        pass

            # ----------------------------------------------------------------
            # smart_summarize — LLM summarization of text
            # ----------------------------------------------------------------
            if name == "smart_summarize":
                content_ss = str(args.get("content", "")).strip()
                style_ss   = str(args.get("style", "brief")).strip() or "brief"
                max_words  = args.get("max_words")
                if not content_ss:
                    return _text("smart_summarize: 'content' is required")
                text_ss = content_ss[:8000]
                if style_ss == "bullets":
                    instruction_ss = "Summarize the following text as a concise markdown bullet list."
                elif style_ss == "detailed":
                    instruction_ss = "Write a detailed, comprehensive summary of the following text."
                else:
                    instruction_ss = "Summarize the following text in 2–3 sentences."
                if max_words:
                    instruction_ss += f" Aim for approximately {int(max_words)} words."
                msgs_ss = [
                    {"role": "system", "content": "You are a helpful summarizer. Be concise and accurate."},
                    {"role": "user",   "content": f"{instruction_ss}\n\n---\n{text_ss}"},
                ]
                payload_ss: dict = {"messages": msgs_ss, "max_tokens": 600, "temperature": 0.3}
                if IMAGE_GEN_MODEL:
                    payload_ss["model"] = IMAGE_GEN_MODEL
                async with httpx.AsyncClient(timeout=30.0) as hc_ss:
                    try:
                        r_ss = await hc_ss.post(
                            f"{IMAGE_GEN_BASE_URL}/v1/chat/completions", json=payload_ss
                        )
                        r_ss.raise_for_status()
                        _ch_ss = r_ss.json().get("choices", [])
                        summary = (_ch_ss[0]["message"]["content"].strip() if _ch_ss else "")
                        return _text(summary if summary else "smart_summarize: empty response from model")
                    except Exception as exc:
                        return _text(
                            f"smart_summarize: LM Studio request failed — {exc}\n"
                            "Make sure a chat model is loaded in LM Studio."
                        )

            # ----------------------------------------------------------------
            # image_caption — vision model image description
            # ----------------------------------------------------------------
            if name == "image_caption":
                b64_ic     = str(args.get("b64", "")).strip()
                detail_ic  = str(args.get("detail_level", "detailed")).strip() or "detailed"
                if not b64_ic:
                    return _text("image_caption: 'b64' is required (base64-encoded JPEG)")
                if detail_ic == "brief":
                    prompt_ic = "Describe this image in one concise sentence."
                else:
                    prompt_ic = (
                        "Describe this image in detail. Include: the main subject, "
                        "background, colors, style, mood, and any notable elements. "
                        "Be specific and vivid."
                    )
                msgs_ic = [{"role": "user", "content": [
                    {"type": "text",      "text": prompt_ic},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_ic}"}},
                ]}]
                payload_ic: dict = {
                    "messages": msgs_ic,
                    "max_tokens": 300 if detail_ic == "detailed" else 80,
                    "temperature": 0.3,
                }
                if IMAGE_GEN_MODEL:
                    payload_ic["model"] = IMAGE_GEN_MODEL
                async with httpx.AsyncClient(timeout=30.0) as hc_ic:
                    try:
                        r_ic = await asyncio.wait_for(
                            hc_ic.post(
                                f"{IMAGE_GEN_BASE_URL}/v1/chat/completions", json=payload_ic
                            ),
                            timeout=12.0,
                        )
                        r_ic.raise_for_status()
                        _ch_ic = r_ic.json().get("choices", [])
                        caption = (_ch_ic[0]["message"]["content"].strip() if _ch_ic else "")
                        return _text(caption if caption else "image_caption: empty response from vision model")
                    except Exception as exc:
                        return _text(
                            f"image_caption: LM Studio vision request failed — {exc}\n"
                            "Make sure a vision-capable model is loaded in LM Studio."
                        )

            # ----------------------------------------------------------------
            # structured_extract — JSON extraction with json_object mode
            # ----------------------------------------------------------------
            if name == "structured_extract":
                import json as _js_se
                content_se = str(args.get("content", "")).strip()
                schema_se  = str(args.get("schema_json", "")).strip()
                extra_se   = str(args.get("instructions", "")).strip()
                if not content_se:
                    return _text("structured_extract: 'content' is required")
                text_se    = content_se[:6000]
                schema_hint = f"\n\nExtract data matching this schema:\n{schema_se}" if schema_se else ""
                extra_hint  = f"\n\nAdditional instructions: {extra_se}" if extra_se else ""
                msgs_se = [
                    {"role": "system", "content": (
                        "You are a structured data extractor. Extract information from the provided "
                        "text and return it as valid JSON matching the requested schema. "
                        "Return ONLY the JSON object, no extra text."
                    )},
                    {"role": "user", "content": f"Extract from this text:{schema_hint}{extra_hint}\n\n---\n{text_se}"},
                ]
                payload_se: dict = {
                    "messages": msgs_se,
                    "max_tokens": 800,
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"},
                }
                if IMAGE_GEN_MODEL:
                    payload_se["model"] = IMAGE_GEN_MODEL
                async with httpx.AsyncClient(timeout=30.0) as hc_se:
                    try:
                        r_se = await hc_se.post(
                            f"{IMAGE_GEN_BASE_URL}/v1/chat/completions", json=payload_se
                        )
                        r_se.raise_for_status()
                        _ch_se = r_se.json().get("choices", [])
                        raw_se = (_ch_se[0]["message"]["content"].strip() if _ch_se else "")
                        try:
                            parsed = _js_se.loads(raw_se)
                            return _text(_js_se.dumps(parsed, indent=2))
                        except _js_se.JSONDecodeError:
                            return _text(f"structured_extract: model returned non-JSON:\n{raw_se}")
                    except Exception as exc:
                        return _text(
                            f"structured_extract: LM Studio request failed — {exc}\n"
                            "Make sure a chat model is loaded (ideally one that supports json_object mode)."
                        )

            # ----------------------------------------------------------------
            elif name == "orchestrate":
                raw_steps = args.get("steps", [])
                stop_on_error = bool(args.get("stop_on_error", False))
                if not isinstance(raw_steps, list) or not raw_steps:
                    return _text("orchestrate: 'steps' must be a non-empty list")
                try:
                    steps = [
                        WorkflowStep(
                            id=str(s["id"]),
                            tool=str(s["tool"]),
                            args=dict(s.get("args", {})),
                            depends_on=list(s.get("depends_on", [])),
                            label=str(s.get("label", s["id"])),
                        )
                        for s in raw_steps
                    ]
                except (KeyError, TypeError) as exc:
                    return _text(f"orchestrate: invalid step definition — {exc}")
                try:
                    executor = WorkflowExecutor(steps, stop_on_error=stop_on_error)
                    results = await executor.run()
                except ValueError as exc:
                    return _text(f"orchestrate: workflow error — {exc}")
                return _text(WorkflowExecutor._format_report(results))

            # ----------------------------------------------------------------
            # Graph tools
            # ----------------------------------------------------------------
            if name == "graph_add_node":
                node_id    = str(args.get("id", "")).strip()
                labels     = list(args.get("labels", []))
                properties = dict(args.get("properties", {}))
                if not node_id:
                    return _text("graph_add_node: 'id' is required")
                r = await c.post(f"{GRAPH_URL}/nodes/add",
                                 json={"id": node_id, "labels": labels, "properties": properties},
                                 timeout=10)
                d = r.json()
                return _text(f"Node added: {d.get('added')} (labels={d.get('labels')})")

            if name == "graph_add_edge":
                from_id    = str(args.get("from_id", "")).strip()
                to_id      = str(args.get("to_id",   "")).strip()
                etype      = str(args.get("type", "related")).strip() or "related"
                properties = dict(args.get("properties", {}))
                if not from_id or not to_id:
                    return _text("graph_add_edge: 'from_id' and 'to_id' are required")
                r = await c.post(f"{GRAPH_URL}/edges/add",
                                 json={"from_id": from_id, "to_id": to_id,
                                       "type": etype, "properties": properties},
                                 timeout=10)
                d = r.json()
                return _text(f"Edge added: {d.get('from_id')} -[{etype}]-> {d.get('to_id')} (id={d.get('added')})")

            if name == "graph_query":
                node_id = str(args.get("id", "")).strip()
                if not node_id:
                    return _text("graph_query: 'id' is required")
                r = await c.get(f"{GRAPH_URL}/nodes/{node_id}/neighbors", timeout=10)
                if r.status_code == 404:
                    return _text(f"graph_query: node '{node_id}' not found")
                d = r.json()
                neighbors = d.get("neighbors", [])
                if not neighbors:
                    return _text(f"Node '{node_id}' has no outgoing neighbors.")
                lines = [f"Neighbors of '{node_id}' ({len(neighbors)}):"]
                for nb in neighbors:
                    props = nb.get("properties", {})
                    lines.append(f"  → {nb['id']} [{nb.get('edge_type','')}]"
                                 + (f" {props}" if props else ""))
                return _text("\n".join(lines))

            if name == "graph_path":
                from_id = str(args.get("from_id", "")).strip()
                to_id   = str(args.get("to_id",   "")).strip()
                if not from_id or not to_id:
                    return _text("graph_path: 'from_id' and 'to_id' are required")
                r = await c.post(f"{GRAPH_URL}/path",
                                 json={"from_id": from_id, "to_id": to_id}, timeout=15)
                d = r.json()
                path = d.get("path")
                if not path:
                    return _text(f"No path found from '{from_id}' to '{to_id}'.")
                length = d.get("length", len(path) - 1)
                return _text(f"Path ({length} hops): {' → '.join(path)}")

            if name == "graph_search":
                label      = str(args.get("label", "")).strip()
                properties = dict(args.get("properties", {}))
                limit_g    = int(args.get("limit", 50))
                r = await c.post(f"{GRAPH_URL}/search",
                                 json={"label": label, "properties": properties, "limit": limit_g},
                                 timeout=10)
                d = r.json()
                results_g = d.get("results", [])
                if not results_g:
                    return _text(f"graph_search: no nodes found (label={label!r})")
                lines = [f"Found {len(results_g)} node(s):"]
                for node in results_g:
                    lbl = ",".join(node.get("labels", []))
                    props = node.get("properties", {})
                    lines.append(f"  {node['id']} [{lbl}] {props}")
                return _text("\n".join(lines))

            # ----------------------------------------------------------------
            # Vector tools (Qdrant)
            # ----------------------------------------------------------------
            if name == "vector_store":
                text_v      = str(args.get("text", "")).strip()
                vid         = str(args.get("id", "")).strip()
                collection  = str(args.get("collection", "default")).strip() or "default"
                metadata_v  = dict(args.get("metadata", {}))
                if not text_v:
                    return _text("vector_store: 'text' is required")
                # Qdrant requires UUID or unsigned integer point IDs
                # Use a deterministic UUID v5 from the user-supplied id string
                import uuid as _uuid_v
                if not vid:
                    vid_key = _uuid_v.uuid4().hex
                else:
                    vid_key = vid
                # Convert to UUID (v5, namespace=DNS) for Qdrant compatibility
                qdrant_id = str(_uuid_v.uuid5(_uuid_v.NAMESPACE_DNS, vid_key))
                # Step 1: embed via LM Studio
                try:
                    embed_payload = {"input": text_v}
                    if IMAGE_GEN_MODEL:
                        embed_payload["model"] = IMAGE_GEN_MODEL
                    async with httpx.AsyncClient(timeout=30) as hc_v:
                        r_embed = await hc_v.post(
                            f"{IMAGE_GEN_BASE_URL}/v1/embeddings", json=embed_payload
                        )
                        r_embed.raise_for_status()
                        embed_data = r_embed.json().get("data", [])
                        if not embed_data:
                            return _text("vector_store: LM Studio returned no embeddings")
                        vector = embed_data[0]["embedding"]
                        dim    = len(vector)
                except Exception as exc:
                    return _text(f"vector_store: embedding failed — {exc}")
                # Step 2: ensure collection exists
                try:
                    rc = await c.get(f"{VECTOR_URL}/collections/{collection}", timeout=5)
                    if rc.status_code == 404:
                        await c.put(
                            f"{VECTOR_URL}/collections/{collection}",
                            json={"vectors": {"size": dim, "distance": "Cosine"}},
                            timeout=10,
                        )
                except Exception:
                    pass  # best-effort collection creation
                # Step 3: upsert point (store user id in payload for delete by id_key)
                payload_meta = {"text": text_v[:500], "id_key": vid_key, **metadata_v}
                try:
                    ru = await c.put(
                        f"{VECTOR_URL}/collections/{collection}/points",
                        json={"points": [{"id": qdrant_id, "vector": vector, "payload": payload_meta}]},
                        timeout=15,
                    )
                    ru.raise_for_status()
                except Exception as exc:
                    return _text(f"vector_store: Qdrant upsert failed — {exc}")
                return _text(f"Stored vector: id={vid_key} (qdrant_id={qdrant_id}) collection={collection} dim={dim}")

            if name == "vector_search":
                query_v    = str(args.get("query", "")).strip()
                collection = str(args.get("collection", "default")).strip() or "default"
                top_k      = max(1, int(args.get("top_k", 5)))
                filter_v   = args.get("filter")
                if not query_v:
                    return _text("vector_search: 'query' is required")
                # Embed query
                try:
                    ep = {"input": query_v}
                    if IMAGE_GEN_MODEL:
                        ep["model"] = IMAGE_GEN_MODEL
                    async with httpx.AsyncClient(timeout=30) as hc_vs:
                        re = await hc_vs.post(f"{IMAGE_GEN_BASE_URL}/v1/embeddings", json=ep)
                        re.raise_for_status()
                        ed = re.json().get("data", [])
                        if not ed:
                            return _text("vector_search: embedding returned empty")
                        qvec = ed[0]["embedding"]
                except Exception as exc:
                    return _text(f"vector_search: embedding failed — {exc}")
                # Search Qdrant
                search_body: dict = {"vector": qvec, "limit": top_k, "with_payload": True}
                if filter_v:
                    search_body["filter"] = filter_v
                try:
                    rs = await c.post(
                        f"{VECTOR_URL}/collections/{collection}/points/search",
                        json=search_body, timeout=10,
                    )
                    rs.raise_for_status()
                    hits = rs.json().get("result", [])
                except Exception as exc:
                    return _text(f"vector_search: Qdrant search failed — {exc}")
                if not hits:
                    return _text(f"vector_search: no results in collection '{collection}'")
                lines = [f"Top {len(hits)} results from '{collection}':"]
                for h in hits:
                    score   = round(h.get("score", 0), 4)
                    payload = h.get("payload", {})
                    text_preview = str(payload.get("text", ""))[:120]
                    lines.append(f"  [{score}] id={h.get('id')} — {text_preview}")
                return _text("\n".join(lines))

            if name == "vector_delete":
                vid_d      = str(args.get("id", "")).strip()
                collection = str(args.get("collection", "default")).strip() or "default"
                if not vid_d:
                    return _text("vector_delete: 'id' is required")
                # Delete by payload id_key filter (since Qdrant IDs are UUIDs internally)
                import uuid as _uuid_d
                qdrant_id_d = str(_uuid_d.uuid5(_uuid_d.NAMESPACE_DNS, vid_d))
                try:
                    rd = await c.post(
                        f"{VECTOR_URL}/collections/{collection}/points/delete",
                        json={"points": [qdrant_id_d]}, timeout=10,
                    )
                    rd.raise_for_status()
                except Exception as exc:
                    return _text(f"vector_delete: failed — {exc}")
                return _text(f"Deleted vector id={vid_d} from collection '{collection}'")

            if name == "vector_collections":
                try:
                    rc = await c.get(f"{VECTOR_URL}/collections", timeout=10)
                    rc.raise_for_status()
                    colls = rc.json().get("result", {}).get("collections", [])
                except Exception as exc:
                    return _text(f"vector_collections: failed — {exc}")
                if not colls:
                    return _text("No Qdrant collections found. Use vector_store to create one.")
                lines = ["Qdrant collections:"]
                for col in colls:
                    lines.append(f"  {col.get('name')} — vectors: {col.get('vectors_count', '?')}")
                return _text("\n".join(lines))

            # ----------------------------------------------------------------
            # Video tools
            # ----------------------------------------------------------------
            if name == "video_info":
                url_vi = str(args.get("url", "")).strip()
                if not url_vi:
                    return _text("video_info: 'url' is required")
                try:
                    r_vi = await c.post(f"{VIDEO_URL}/info", json={"url": url_vi}, timeout=90)
                    r_vi.raise_for_status()
                    d_vi = r_vi.json()
                except Exception as exc:
                    return _text(f"video_info: failed — {exc}")
                return _text(
                    f"Video info for {url_vi}:\n"
                    f"  Duration: {d_vi.get('duration_s')}s  FPS: {d_vi.get('fps')}\n"
                    f"  Resolution: {d_vi.get('width')}×{d_vi.get('height')}  Codec: {d_vi.get('codec')}\n"
                    f"  Format: {d_vi.get('format')}  Size: {d_vi.get('size_mb')} MB"
                )

            if name == "video_frames":
                url_vf       = str(args.get("url", "")).strip()
                interval_vf  = float(args.get("interval_sec", 5.0))
                max_frames_vf = int(args.get("max_frames", 20))
                if not url_vf:
                    return _text("video_frames: 'url' is required")
                try:
                    r_vf = await c.post(
                        f"{VIDEO_URL}/frames",
                        json={"url": url_vf, "interval_sec": interval_vf,
                              "max_frames": max_frames_vf},
                        timeout=180,
                    )
                    r_vf.raise_for_status()
                    d_vf = r_vf.json()
                except Exception as exc:
                    return _text(f"video_frames: failed — {exc}")
                frames = d_vf.get("frames", [])
                lines = [f"Extracted {len(frames)} frames from {url_vf}:"]
                for fr in frames:
                    lines.append(f"  [{fr.get('timestamp_s')}s] {fr.get('path')}")
                return _text("\n".join(lines))

            if name == "video_thumbnail":
                url_vt  = str(args.get("url", "")).strip()
                ts_vt   = float(args.get("timestamp_sec", 0.0))
                if not url_vt:
                    return _text("video_thumbnail: 'url' is required")
                try:
                    r_vt = await c.post(
                        f"{VIDEO_URL}/thumbnail",
                        json={"url": url_vt, "timestamp_sec": ts_vt},
                        timeout=90,
                    )
                    r_vt.raise_for_status()
                    d_vt = r_vt.json()
                except Exception as exc:
                    return _text(f"video_thumbnail: failed — {exc}")
                b64_vt = d_vt.get("b64", "")
                if not b64_vt:
                    return _text(f"video_thumbnail: no image returned")
                import base64 as _b64_vt_mod
                raw_vt = _b64_vt_mod.b64decode(b64_vt)
                summary_vt = (f"Video thumbnail at {ts_vt}s — "
                              f"{d_vt.get('width')}×{d_vt.get('height')} from {url_vt}")
                return _renderer.encode_url_bytes(raw_vt, "image/png", summary_vt)

            # ----------------------------------------------------------------
            # OCR tools
            # ----------------------------------------------------------------
            if name == "ocr_image":
                path_oi = str(args.get("path", "")).strip()
                lang_oi = str(args.get("lang", "eng")).strip() or "eng"
                if not path_oi:
                    return _text("ocr_image: 'path' is required")
                local_oi = _resolve_image_path(path_oi)
                if not local_oi:
                    return _text(f"ocr_image: file not found — {path_oi}")
                import base64 as _b64_oi
                b64_oi = _b64_oi.standard_b64encode(
                    open(local_oi, "rb").read()
                ).decode("ascii")
                try:
                    r_oi = await c.post(
                        f"{OCR_URL}/ocr",
                        json={"b64": b64_oi, "lang": lang_oi},
                        timeout=60,
                    )
                    r_oi.raise_for_status()
                    d_oi = r_oi.json()
                except Exception as exc:
                    return _text(f"ocr_image: OCR service failed — {exc}")
                text_oi = d_oi.get("text", "")
                words_oi = d_oi.get("word_count", 0)
                return _text(f"OCR result ({words_oi} words):\n\n{text_oi}")

            if name == "ocr_pdf":
                path_op = str(args.get("path", "")).strip()
                lang_op = str(args.get("lang", "eng")).strip() or "eng"
                pages_op = list(args.get("pages", []))
                if not path_op:
                    return _text("ocr_pdf: 'path' is required")
                local_op = _resolve_image_path(path_op)
                if not local_op:
                    return _text(f"ocr_pdf: file not found — {path_op}")
                import base64 as _b64_op
                b64_op = _b64_op.standard_b64encode(
                    open(local_op, "rb").read()
                ).decode("ascii")
                try:
                    r_op = await c.post(
                        f"{OCR_URL}/ocr/pdf",
                        json={"b64_pdf": b64_op, "lang": lang_op, "pages": pages_op or None},
                        timeout=120,
                    )
                    r_op.raise_for_status()
                    d_op = r_op.json()
                except Exception as exc:
                    return _text(f"ocr_pdf: OCR service failed — {exc}")
                total_w = d_op.get("word_count", 0)
                page_count = d_op.get("page_count", 0)
                full_text = d_op.get("full_text", "")
                return _text(f"PDF OCR: {page_count} pages, {total_w} words\n\n{full_text}")

            # ----------------------------------------------------------------
            # Docs tools
            # ----------------------------------------------------------------
            if name in ("docs_ingest", "docs_extract_tables"):
                url_di   = str(args.get("url",  "")).strip()
                path_di  = str(args.get("path", "")).strip()
                fname_di = str(args.get("filename", "")).strip()

                if url_di:
                    # Service fetches URL directly
                    try:
                        if name == "docs_ingest":
                            r_di = await c.post(f"{DOCS_URL}/ingest/url",
                                                json={"url": url_di}, timeout=60)
                        else:
                            # For tables from URL: ingest first then return tables
                            r_di = await c.post(f"{DOCS_URL}/ingest/url",
                                                json={"url": url_di}, timeout=60)
                        r_di.raise_for_status()
                        d_di = r_di.json()
                    except Exception as exc:
                        return _text(f"{name}: docs service failed — {exc}")
                elif path_di:
                    local_di = _resolve_image_path(path_di)
                    if not local_di:
                        return _text(f"{name}: file not found — {path_di}")
                    if not fname_di:
                        fname_di = os.path.basename(local_di)
                    import base64 as _b64_di
                    b64_di = _b64_di.standard_b64encode(
                        open(local_di, "rb").read()
                    ).decode("ascii")
                    try:
                        endpoint_di = "/ingest" if name == "docs_ingest" else "/tables"
                        r_di = await c.post(
                            f"{DOCS_URL}{endpoint_di}",
                            json={"b64": b64_di, "filename": fname_di},
                            timeout=60,
                        )
                        r_di.raise_for_status()
                        d_di = r_di.json()
                    except Exception as exc:
                        return _text(f"{name}: docs service failed — {exc}")
                else:
                    return _text(f"{name}: 'url' or 'path' is required")

                if name == "docs_ingest":
                    md_di  = d_di.get("markdown", "")
                    title  = d_di.get("title", "")
                    words  = d_di.get("word_count", 0)
                    tables = d_di.get("tables_found", 0)
                    return _text(f"# {title}\n\n_{words} words, {tables} tables_\n\n{md_di}")
                else:
                    tables_di = d_di.get("tables", [])
                    if not tables_di:
                        return _text("No tables found in document.")
                    import json as _jdi
                    return _text(_jdi.dumps(tables_di, indent=2))

            # ----------------------------------------------------------------
            # Planner tools
            # ----------------------------------------------------------------
            if name == "plan_create_task":
                title_pt = str(args.get("title", "")).strip()
                if not title_pt:
                    return _text("plan_create_task: 'title' is required")
                body_pt = {
                    "title":       title_pt,
                    "description": str(args.get("description", "")),
                    "depends_on":  list(args.get("depends_on", [])),
                    "priority":    int(args.get("priority", 0)),
                    "metadata":    dict(args.get("metadata", {})),
                }
                if args.get("due_at"):
                    body_pt["due_at"] = str(args["due_at"])
                try:
                    r_pt = await c.post(f"{PLANNER_URL}/tasks", json=body_pt, timeout=10)
                    r_pt.raise_for_status()
                    d_pt = r_pt.json()
                except Exception as exc:
                    return _text(f"plan_create_task: failed — {exc}")
                return _text(
                    f"Task created: id={d_pt.get('id')} title={d_pt.get('title')!r} "
                    f"status={d_pt.get('status')} priority={d_pt.get('priority')} "
                    f"depends_on={d_pt.get('depends_on')}"
                )

            if name == "plan_get_task":
                tid_pg = str(args.get("id", "")).strip()
                if not tid_pg:
                    return _text("plan_get_task: 'id' is required")
                try:
                    r_pg = await c.get(f"{PLANNER_URL}/tasks/{tid_pg}", timeout=10)
                    if r_pg.status_code == 404:
                        return _text(f"plan_get_task: task '{tid_pg}' not found")
                    r_pg.raise_for_status()
                    d_pg = r_pg.json()
                except Exception as exc:
                    return _text(f"plan_get_task: failed — {exc}")
                return _text(
                    f"Task {d_pg['id']}: {d_pg['title']!r}\n"
                    f"  Status: {d_pg['status']}  Priority: {d_pg['priority']}\n"
                    f"  Description: {d_pg.get('description','')}\n"
                    f"  Depends on: {d_pg.get('depends_on',[])}\n"
                    f"  Created: {d_pg.get('created_at','')}  Updated: {d_pg.get('updated_at','')}"
                )

            if name == "plan_complete_task":
                tid_pc = str(args.get("id", "")).strip()
                if not tid_pc:
                    return _text("plan_complete_task: 'id' is required")
                try:
                    r_pc = await c.post(f"{PLANNER_URL}/tasks/{tid_pc}/complete", timeout=10)
                    if r_pc.status_code == 404:
                        return _text(f"plan_complete_task: task '{tid_pc}' not found")
                    r_pc.raise_for_status()
                except Exception as exc:
                    return _text(f"plan_complete_task: failed — {exc}")
                return _text(f"Task {tid_pc} marked as done.")

            if name == "plan_fail_task":
                tid_pf = str(args.get("id", "")).strip()
                detail_pf = str(args.get("detail", "")).strip()
                if not tid_pf:
                    return _text("plan_fail_task: 'id' is required")
                try:
                    r_pf = await c.post(f"{PLANNER_URL}/tasks/{tid_pf}/fail",
                                        json={"detail": detail_pf}, timeout=10)
                    if r_pf.status_code == 404:
                        return _text(f"plan_fail_task: task '{tid_pf}' not found")
                    r_pf.raise_for_status()
                except Exception as exc:
                    return _text(f"plan_fail_task: failed — {exc}")
                return _text(f"Task {tid_pf} marked as failed. Reason: {detail_pf or '(none)'}")

            if name == "plan_list_tasks":
                status_pl = str(args.get("status", "")).strip()
                limit_pl  = max(1, int(args.get("limit", 50)))
                try:
                    params_pl: dict = {"limit": limit_pl}
                    if status_pl:
                        params_pl["status"] = status_pl
                    r_pl = await c.get(f"{PLANNER_URL}/tasks", params=params_pl, timeout=10)
                    r_pl.raise_for_status()
                    d_pl = r_pl.json()
                except Exception as exc:
                    return _text(f"plan_list_tasks: failed — {exc}")
                tasks_pl = d_pl.get("tasks", [])
                total_pl = d_pl.get("total", len(tasks_pl))
                if not tasks_pl:
                    return _text(f"No tasks found" + (f" with status='{status_pl}'" if status_pl else "") + ".")
                lines_pl = [f"Tasks ({len(tasks_pl)}/{total_pl}" +
                            (f", status={status_pl}" if status_pl else "") + "):"]
                for t_pl in tasks_pl:
                    dep_pl = f" [deps: {t_pl['depends_on']}]" if t_pl.get("depends_on") else ""
                    lines_pl.append(f"  [{t_pl['status']}] {t_pl['id']}: {t_pl['title']!r}{dep_pl}")
                return _text("\n".join(lines_pl))

            if name == "plan_delete_task":
                tid_pd = str(args.get("id", "")).strip()
                if not tid_pd:
                    return _text("plan_delete_task: 'id' is required")
                try:
                    r_pd = await c.delete(f"{PLANNER_URL}/tasks/{tid_pd}", timeout=10)
                    if r_pd.status_code == 404:
                        return _text(f"plan_delete_task: task '{tid_pd}' not found")
                    r_pd.raise_for_status()
                except Exception as exc:
                    return _text(f"plan_delete_task: failed — {exc}")
                return _text(f"Task {tid_pd} deleted.")

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
        # Echo back the client's requested version if we support it; otherwise
        # fall back to 2024-11-05.  LM Studio 0.3.6+ sends "2025-03-26" and
        # will only enable image rendering when the agreed version matches.
        client_ver = params.get("protocolVersion", "2024-11-05")
        agreed_ver = client_ver if client_ver in {"2024-11-05", "2025-03-26"} else "2024-11-05"
        return ok({
            "protocolVersion": agreed_ver,
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
        if ImageRenderingPolicy.is_image_tool(tool_name):
            content_blocks = ImageRenderingPolicy.enforce(content_blocks)
        # Propagate isError so MCP clients can distinguish tool errors from
        # successful empty results.  A block is considered an error if it is
        # text-only AND its text starts with "Error" or "Unknown tool".
        is_error = (
            bool(content_blocks)
            and not any(b.get("type") == "image" for b in content_blocks)
            and all(b.get("type") == "text" for b in content_blocks)
            and any(
                b.get("text", "").startswith(("Error", "Unknown tool"))
                for b in content_blocks
            )
        )
        return ok({
            "content": content_blocks,
            "isError": is_error,
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

    method = rpc.get("method", "")
    accept = request.headers.get("accept", "")

    # MCP 2025-03-26 §3.3: server MUST return Mcp-Session-Id on initialize so
    # that LM Studio can enable image rendering and session tracking.
    extra_headers: dict[str, str] = {}
    if method == "initialize":
        extra_headers["Mcp-Session-Id"] = str(uuid.uuid4())

    if "text/event-stream" in accept:
        # Non-blocking: stream keepalives while the tool runs, yield result when done.
        async def _stream_result(rpc: dict):
            task = asyncio.create_task(_handle_rpc(rpc))
            elapsed = 0.0
            while not task.done():
                yield ": keepalive\n\n"
                await asyncio.sleep(5.0)
                elapsed += 5.0
                if elapsed >= _TOOL_TIMEOUT:
                    task.cancel()
                    req_id = rpc.get("id")
                    error_resp = {
                        "jsonrpc": "2.0", "id": req_id,
                        "error": {"code": -32000, "message": "Tool execution timed out"},
                    }
                    yield f"event: message\ndata: {json.dumps(error_resp)}\n\n"
                    return
            try:
                result = task.result()
            except Exception as exc:
                req_id = rpc.get("id")
                error_resp = {
                    "jsonrpc": "2.0", "id": req_id,
                    "error": {"code": -32000, "message": str(exc)},
                }
                yield f"event: message\ndata: {json.dumps(error_resp)}\n\n"
                return
            if result is not None:
                yield f"event: message\ndata: {json.dumps(result)}\n\n"
        return StreamingResponse(_stream_result(rpc), media_type="text/event-stream",
                                 headers={**_SSE_HEADERS, **extra_headers})

    # JSON (non-SSE) path — synchronous; used only by non-LM-Studio clients.
    response = await _handle_rpc(rpc)
    if response is None:
        return Response(content="", status_code=202, headers=extra_headers)
    return Response(content=json.dumps(response), media_type="application/json",
                    headers=extra_headers)


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
        task = asyncio.create_task(_handle_rpc(rpc))
        req_id = rpc.get("id")
        elapsed = 0.0
        # Send MCP progress notifications every 5 s while the tool runs.
        # These are real data: events on the SSE stream — keeps LM Studio's
        # internal tool-call timer alive even for slow tools (screenshot, etc.).
        while not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
            except asyncio.TimeoutError:
                elapsed += 5.0
                if elapsed >= _TOOL_TIMEOUT:
                    task.cancel()
                    if sid in _sessions and req_id is not None:
                        await _sessions[sid].put({
                            "jsonrpc": "2.0", "id": req_id,
                            "error": {"code": -32000, "message": "Tool execution timed out"},
                        })
                    return
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
            except Exception:
                break
        try:
            response = task.result()
        except Exception as exc:
            if sid in _sessions and req_id is not None:
                await _sessions[sid].put({
                    "jsonrpc": "2.0", "id": req_id,
                    "error": {"code": -32000, "message": str(exc)},
                })
            return
        if response is not None and sid in _sessions:
            await _sessions[sid].put(response)

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
