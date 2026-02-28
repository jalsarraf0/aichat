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
    b64: str, subject: str, hc: "httpx.AsyncClient"
) -> "tuple[bool, str, float]":
    """Ask the loaded multimodal model whether the image shows *subject*.

    Returns (is_match, description, confidence).  Fails open on any error so
    image_search never breaks when no vision model is loaded in LM Studio.
    Uses IMAGE_GEN_BASE_URL/v1/chat/completions with 8-second timeout.
    Set IMAGE_VISION_CONFIRM=false to disable (skip confirmation, keep all images).
    """
    env_flag = os.environ.get("IMAGE_VISION_CONFIRM", "true").lower()
    if env_flag not in ("1", "true", "yes"):
        return True, "", 0.6
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
        text = r.json()["choices"][0]["message"]["content"].strip()
        desc  = ""
        if "DESCRIPTION:" in text:
            desc = text.split("DESCRIPTION:")[1].split("|")[0].strip()
        match = "MATCH: YES" in text.upper() or text.upper().strip().endswith("YES")
        conf  = 0.9 if match else 0.2
        return match, desc, conf
    except Exception:
        return True, "", 0.6   # fail-open


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
# human_browser browser-server API — reachable after install connects it to this network.
BROWSER_URL   = os.environ.get("BROWSER_URL",  "http://human_browser:7081")
# Screenshot PNGs are bind-mounted from /docker/human_browser/workspace on the host.
BROWSER_WORKSPACE = os.environ.get("BROWSER_WORKSPACE", "/browser-workspace")
# Image generation — LM Studio OpenAI-compatible image API (or any compatible backend).
IMAGE_GEN_BASE_URL = os.environ.get("IMAGE_GEN_BASE_URL", "http://host.docker.internal:1234")
IMAGE_GEN_MODEL    = os.environ.get("IMAGE_GEN_MODEL", "")

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
        "name": "image_upscale",
        "description": (
            "Upscale a saved image using high-quality PIL LANCZOS resampling (no AI model required). "
            "Works on any image in the workspace including generated images, screenshots, or crops. "
            "Optionally applies a sharpening pass after upscaling. "
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
                    "description": "Upscale factor: 2.0 = 200%, max 8.0 (default 2.0).",
                },
                "sharpen": {
                    "type": "boolean",
                    "description": "Apply a sharpening pass after upscaling to recover edge crispness (default true).",
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
            "Use for data analysis, calculations, file manipulation, or testing code snippets."
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
    quality: int = 90,
    save_prefix: str | None = None,
) -> list[dict[str, Any]]:
    """
    Encode a PIL Image as an inline JPEG MCP block.
    If save_prefix is given, also write the result to BROWSER_WORKSPACE so the
    filename can be used as 'path' in the next pipeline step.
    """
    if save_prefix and os.path.isdir(BROWSER_WORKSPACE):
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{save_prefix}_{ts}.jpg"
            local_path = os.path.join(BROWSER_WORKSPACE, filename)
            img.convert("RGB").save(local_path, format="JPEG", quality=quality)
            summary += f"\n→ Saved as: {filename}  (pass this as 'path' in the next pipeline step)"
        except OSError:
            pass  # workspace may be read-only; inline image is still returned
    buf = _io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return [
        {"type": "text",  "text": summary},
        {"type": "image", "data": b64, "mimeType": "image/jpeg"},
    ]


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
                search_params: dict = {}
                if args.get("topic"):
                    search_params["topic"] = str(args["topic"])
                if args.get("q"):
                    search_params["q"] = str(args["q"])
                search_params["limit"] = int(args.get("limit", 20))
                search_params["offset"] = int(args.get("offset", 0))
                if args.get("summary_only"):
                    search_params["summary_only"] = "true"
                r = await c.get(f"{DATABASE_URL}/articles/search", params=search_params)
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
                payload: dict = {"key": args["key"], "value": args["value"]}
                if args.get("ttl_seconds"):
                    payload["ttl_seconds"] = int(args["ttl_seconds"])
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
                        f"  [{ts}] [{e['level']}] {e['service']}: {e['message']}"
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
                    img = _src.copy()

                w, h = img.size
                left   = int(args.get("left",   0))
                top    = int(args.get("top",    0))
                right  = int(args.get("right",  w))
                bottom = int(args.get("bottom", h))
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
                    zoomed = region.resize((int(rw * scale), int(rh * scale)), _PilImage.LANCZOS)
                    zw, zh = zoomed.size
                    summary = (
                        f"Zoomed {scale:.1f}× from ({box[0]},{box[1]})→({box[2]},{box[3]})\n"
                        f"Region: {rw}×{rh}  Output: {zw}×{zh}  Source: {os.path.basename(path)}"
                    )
                    return _pil_to_blocks(zoomed, summary, quality=92, save_prefix="zoomed")

                if name == "image_scan":
                    region = img.crop(box) if box != (0, 0, w, h) else img
                    rw, rh = region.size
                    # Auto-upscale small regions so fine text is legible for the vision model
                    if rw < 800:
                        up = max(2, 800 // max(rw, 1))
                        region = region.resize((rw * up, rh * up), _PilImage.LANCZOS)
                        rw, rh = region.size
                    # Greyscale → contrast boost → sharpen → unsharp mask
                    region = region.convert("L")
                    region = _ImageEnhance.Contrast(region).enhance(2.5)
                    region = _ImageEnhance.Sharpness(region).enhance(3.0)
                    region = region.filter(_ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
                    summary = (
                        f"Scan-enhanced for text reading — greyscale + contrast×2.5 + sharpen×3.0\n"
                        f"Region: ({box[0]},{box[1]})→({box[2]},{box[3]})  Output: {rw}×{rh}\n"
                        f"Source: {os.path.basename(path)}\n"
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
                    result = _ImageEnhance.Contrast(result).enhance(contrast)
                    result = _ImageEnhance.Sharpness(result).enhance(sharpness)
                    result = _ImageEnhance.Brightness(result).enhance(brightness)
                    summary = (
                        f"Enhanced: contrast={contrast:.1f} sharpness={sharpness:.1f} "
                        f"brightness={brightness:.1f}"
                        + (" grayscale" if grayscale else "") + "\n"
                        f"Size: {w}×{h}  Source: {os.path.basename(path)}"
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
                gap = max(0, int(args.get("gap", 0)))
                images: list["_PilImage.Image"] = []
                for p in paths:
                    loc = _resolve_image_path(p)
                    if not loc:
                        return _text(f"image_stitch: image not found — '{p}'")
                    with _PilImage.open(loc) as im:
                        images.append(im.convert("RGB").copy())
                if direction == "horizontal":
                    total_w = sum(im.width for im in images) + gap * (len(images) - 1)
                    max_h   = max(im.height for im in images)
                    canvas  = _PilImage.new("RGB", (total_w, max_h), (255, 255, 255))
                    x = 0
                    for im in images:
                        canvas.paste(im, (x, 0))
                        x += im.width + gap
                else:
                    max_w   = max(im.width for im in images)
                    total_h = sum(im.height for im in images) + gap * (len(images) - 1)
                    canvas  = _PilImage.new("RGB", (max_w, total_h), (255, 255, 255))
                    y = 0
                    for im in images:
                        canvas.paste(im, (0, y))
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
                amplify = max(1.0, min(float(args.get("amplify", 3.0)), 10.0))
                with _PilImage.open(loc_a) as ia:
                    img_a = ia.convert("RGB").copy()
                with _PilImage.open(loc_b) as ib:
                    img_b = ib.convert("RGB").copy()
                if img_a.size != img_b.size:
                    img_b = img_b.resize(img_a.size, _PilImage.LANCZOS)
                diff = _ImageChops.difference(img_a, img_b)
                diff_l = diff.convert("L")
                diff_l = _ImageEnhance.Brightness(diff_l).enhance(amplify)
                white = _PilImage.new("RGB", img_a.size, (255, 255, 255))
                red   = _PilImage.new("RGB", img_a.size, (220, 30, 30))
                result = _PilImage.composite(red, white, diff_l)
                summary = (
                    f"Pixel diff: {os.path.basename(path_a)} vs {os.path.basename(path_b)}\n"
                    f"Amplify: {amplify:.1f}×  Size: {img_a.width}×{img_a.height}\n"
                    f"Red pixels = changed regions."
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
                outline_width = max(1, int(args.get("outline_width", 3)))
                with _PilImage.open(loc) as src:
                    img = src.convert("RGB").copy()
                draw = _ImageDraw.Draw(img)
                for box in boxes:
                    bx_l = int(box.get("left",   0))
                    bx_t = int(box.get("top",    0))
                    bx_r = int(box.get("right",  img.width))
                    bx_b = int(box.get("bottom", img.height))
                    color = str(box.get("color", "#FF3333"))
                    label = str(box.get("label", ""))
                    for i in range(outline_width):
                        draw.rectangle([bx_l - i, bx_t - i, bx_r + i, bx_b + i], outline=color)
                    if label:
                        tx, ty = bx_l + 2, max(0, bx_t - 18)
                        draw.rectangle([tx - 2, ty - 2, tx + len(label) * 7 + 2, ty + 16], fill=color)
                        draw.text((tx, ty), label, fill="white")
                n = len(boxes)
                summary = (
                    f"Annotated {n} bounding box{'es' if n != 1 else ''} on {os.path.basename(path)}\n"
                    f"Size: {img.width}×{img.height}"
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
                                frames.append(fr.convert("RGB").copy())
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
            # image_upscale — high-quality LANCZOS upscale (no AI model required)
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
                scale   = max(1.1, min(float(args.get("scale", 2.0)), 8.0))
                sharpen = bool(args.get("sharpen", True))
                with _PilImage.open(local) as src:
                    img = _ImageOps.exif_transpose(src).convert("RGB")
                ow, oh = img.size
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
                    f"Upscaled {scale:.2f}×  {ow}×{oh} → {nw}×{nh}"
                    + (" + sharpen" if sharpen else "")
                    + (" [scale capped to 8192px]" if capped else "")
                    + f"\nSource: {os.path.basename(path)}"
                )
                return _pil_to_blocks(upscaled, summary, quality=92, save_prefix="upscaled")

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
                    if rdata.get("found"):
                        _seen_urls = set(_js2.loads(rdata["entries"][0]["value"]))
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
                    is_match, desc, conf = await _vision_confirm(img_b64, query, c)
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
                        embedding = r_emb.json()["data"][0]["embedding"]
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
                limit_es  = max(1, min(20, int(args.get("limit", 5))))
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
                        q_vec = r_es.json()["data"][0]["embedding"]
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
                        summary = r_ss.json()["choices"][0]["message"]["content"].strip()
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
                        caption = r_ic.json()["choices"][0]["message"]["content"].strip()
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
                        raw_se = r_se.json()["choices"][0]["message"]["content"].strip()
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
            while not task.done():
                yield ": keepalive\n\n"
                await asyncio.sleep(5.0)
            result = task.result()
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
