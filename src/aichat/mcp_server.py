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
            "(ID a12fdfeaaf78). Actions: navigate, read, screenshot, click, fill, eval, "
            "screenshot_element, list_images_detail. "
            "The browser keeps state between calls so you can navigate, click, fill, and read "
            "in sequence. Use 'find_text' with screenshot to zoom into a specific page section. "
            "Use screenshot_element to precisely crop a single element. "
            "Use list_images_detail to see all images with src, alt, dimensions, and viewport info."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["navigate", "read", "screenshot", "click", "fill", "eval",
                             "screenshot_element", "list_images_detail"],
                    "description": "Which browser action to perform.",
                },
                "url":       {"type": "string", "description": "URL for navigate/screenshot."},
                "selector":  {"type": "string", "description": "CSS selector for click/fill/screenshot_element."},
                "value":     {"type": "string", "description": "Text to type (fill only)."},
                "code":      {"type": "string", "description": "JavaScript to evaluate (eval only)."},
                "find_text": {
                    "type": "string",
                    "description": (
                        "Optional, screenshot only. Scroll to the first occurrence of this text "
                        "and clip the screenshot to show just that region."
                    ),
                },
                "find_image": {
                    "type": "string",
                    "description": (
                        "Optional, screenshot only. Match an <img> by src/alt substring or "
                        "1-based index (e.g. 'logo', '#2'). Crops to that image. "
                        "Mutually exclusive with find_text."
                    ),
                },
                "pad": {
                    "type": "integer",
                    "description": "Padding in px around element for screenshot_element (default 20).",
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
            "Use 'find_text' to zoom into a specific section of the page. "
            "Use 'find_image' to precisely crop a specific <img> element by src/alt substring "
            "or 1-based index (e.g. 'logo', 'hero', '2', '#3')."
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
                "find_image": {
                    "type": "string",
                    "description": (
                        "Optional. Match an <img> element by src/alt substring or 1-based index. "
                        "The screenshot is tightly cropped to that image. "
                        "Mutually exclusive with find_text."
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
        "name": "browser_save_images",
        "description": (
            "Download specific image URLs using the real Chromium browser's live session — "
            "exactly like a human right-clicking 'Save Image As'. Uses the browser's cookies, "
            "auth tokens, and referrer headers so auth-gated images, CDN-protected images, and "
            "session-bound content are downloaded successfully. "
            "Returns saved file paths and inline image previews."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "urls": {
                    "description": "Image URLs to download (list of strings or comma-separated string).",
                },
                "prefix": {"type": "string", "description": "Filename prefix (default 'image')."},
                "max":    {"type": "integer", "description": "Max images to download (default 20)."},
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
            "Auth-gated, CDN-protected, and session-bound images all succeed because the browser "
            "session's cookies and headers are used."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url":    {"type": "string",  "description": "Optional. Navigate here first. If omitted, uses current page."},
                "filter": {"type": "string",  "description": "Optional src/alt substring filter (e.g. 'product', 'hero')."},
                "prefix": {"type": "string",  "description": "Filename prefix (default 'image')."},
                "max":    {"type": "integer", "description": "Max images to download (default 20)."},
            },
            "required": [],
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
                "topic":        {"type": "string", "description": "Filter by topic (optional)."},
                "q":            {"type": "string", "description": "Full-text search query (optional)."},
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
    # ── Image manipulation ──────────────────────────────────────────────────
    {
        "name": "image_crop",
        "description": (
            "Crop a saved screenshot or image to a specific pixel region and return it inline. "
            "Step 1 in the zoom-scan pipeline: crop → zoom → scan."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path":   {"type": "string",  "description": "Filename or full path of the screenshot."},
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
            "Crop first (left/top/right/bottom) then scale. Step 2 in the zoom-scan pipeline."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path":   {"type": "string",  "description": "Filename or full path of the screenshot."},
                "scale":  {"type": "number",  "description": "Zoom factor 1.1–8.0 (default 2.0)."},
                "left":   {"type": "integer", "description": "Crop left before zooming (default 0)."},
                "top":    {"type": "integer", "description": "Crop top before zooming (default 0)."},
                "right":  {"type": "integer", "description": "Crop right before zooming (default: width)."},
                "bottom": {"type": "integer", "description": "Crop bottom before zooming (default: height)."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "image_scan",
        "description": (
            "Prepare a screenshot region for text reading: greyscale + contrast×2.5 + sharpen×3.0 "
            "+ auto-upscale. Returns enhanced image inline for the vision model to read. "
            "Step 3 in the zoom-scan pipeline."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path":   {"type": "string",  "description": "Filename or full path of the screenshot."},
                "left":   {"type": "integer", "description": "Region left (default 0)."},
                "top":    {"type": "integer", "description": "Region top (default 0)."},
                "right":  {"type": "integer", "description": "Region right (default: width)."},
                "bottom": {"type": "integer", "description": "Region bottom (default: height)."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "image_enhance",
        "description": (
            "Adjust contrast, sharpness, brightness, or convert to greyscale. "
            "Returns the enhanced image inline."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path":       {"type": "string",  "description": "Filename or full path of the screenshot."},
                "contrast":   {"type": "number",  "description": "0.5–4.0 (default 1.5)."},
                "sharpness":  {"type": "number",  "description": "0.5–4.0 (default 1.5)."},
                "brightness": {"type": "number",  "description": "0.5–3.0 (default 1.0)."},
                "grayscale":  {"type": "boolean", "description": "Convert to greyscale first (default false)."},
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
                "gap": {"type": "integer", "description": "Pixel gap between images (default 0)."},
            },
            "required": ["paths"],
        },
    },
    {
        "name": "image_diff",
        "description": (
            "Show a pixel-level visual diff between two screenshots. "
            "Changed pixels are highlighted in red on a white background."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path_a":   {"type": "string", "description": "First image (before state) — filename or path."},
                "path_b":   {"type": "string", "description": "Second image (after state) — filename or path."},
                "amplify":  {"type": "number", "description": "Difference amplification factor (default 3.0)."},
            },
            "required": ["path_a", "path_b"],
        },
    },
    {
        "name": "image_annotate",
        "description": (
            "Draw bounding boxes and labels on a screenshot to highlight regions of interest."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Image filename or path to annotate."},
                "boxes": {
                    "type": "array",
                    "description": "List of {left, top, right, bottom, label?, color?} objects.",
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
                "outline_width": {"type": "integer", "description": "Box outline thickness (default 3)."},
            },
            "required": ["path", "boxes"],
        },
    },
    {
        "name": "page_extract",
        "description": (
            "Extract structured data from the current browser page: links, headings, tables, images, "
            "and meta tags. Navigate first, then call this."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "include": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["links", "headings", "tables", "images", "meta", "text"]},
                    "description": "Which data types to extract (default: all).",
                },
                "max_links": {"type": "integer", "description": "Max links to return (default 50)."},
                "max_text":  {"type": "integer", "description": "Max chars of body text (default 3000)."},
            },
            "required": [],
        },
    },
    {
        "name": "extract_article",
        "description": (
            "Fetch a URL and extract a clean readable article: title + body text, stripped of ads and nav."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url":       {"type": "string",  "description": "The article URL to fetch and extract."},
                "max_chars": {"type": "integer", "description": "Truncate body to this many characters (default 8000)."},
            },
            "required": ["url"],
        },
    },
    {
        "name": "page_scrape",
        "description": (
            "Navigate to a URL, scroll through the full page in viewport-sized steps waiting for "
            "lazy-loaded content to appear (infinite-scroll feeds, JS widgets, lazy images), then "
            "extract the complete rendered text from the final DOM state. "
            "Unlike web_fetch (grabs text immediately after load), page_scrape simulates a human "
            "scrolling to the bottom so content that only renders on scroll is captured. "
            "Returns full page text plus scroll stats (steps, whether the page grew, final height). "
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
                    "description": "Maximum scroll steps (default 10, max 30).",
                },
                "wait_ms": {
                    "type": "integer",
                    "description": "Milliseconds to wait after each scroll for lazy content (default 500).",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return (default 16000).",
                },
                "include_links": {
                    "type": "boolean",
                    "description": "If true, also return all hyperlinks from the final DOM.",
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
            "schema.org image/logo/thumbnail fields. Scrolls first to trigger lazy loaders. "
            "Returns a deduplicated list (up to 150) with source type and alt text. "
            "Feed the URLs to fetch_image or browser_save_images to render/download them."
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
            "Take screenshots of multiple URLs in parallel and return them all inline (max 6 URLs)."
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
            "Capture a full-page screenshot by scrolling the page and stitching viewport captures. "
            "Navigate to the page first, then call this."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url":            {"type": "string",  "description": "Page URL to capture (or omit to capture current page)."},
                "max_scrolls":    {"type": "integer", "description": "Max scroll steps (default 5, max 10)."},
                "scroll_overlap": {"type": "integer", "description": "Pixel overlap between captures (default 100)."},
            },
            "required": [],
        },
    },
    # ── Image generation ────────────────────────────────────────────────────
    {
        "name": "image_generate",
        "description": (
            "Generate an image from a text prompt using LM Studio's OpenAI-compatible image API "
            "(requires an image generation model such as FLUX or SDXL to be loaded). "
            "The result is saved to the workspace for chaining into image_crop/zoom/scan/upscale/annotate."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt":          {"type": "string",  "description": "Detailed text description of the image."},
                "negative_prompt": {"type": "string",  "description": "What to avoid (optional)."},
                "model":           {"type": "string",  "description": "Model name in LM Studio (optional)."},
                "size":            {"type": "string",  "description": "WxH e.g. '512x512', '1024x1024' (default '512x512')."},
                "n":               {"type": "integer", "description": "Number of images 1–4 (default 1)."},
                "steps":           {"type": "integer", "description": "Inference steps (optional)."},
                "guidance_scale":  {"type": "number",  "description": "Prompt strength 1–20 (optional)."},
                "seed":            {"type": "integer", "description": "Seed for reproducibility (optional)."},
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "image_edit",
        "description": (
            "Edit or remix an existing image using a text prompt (img2img). "
            "Source image must be in the workspace. "
            "Pipeline: screenshot → image_edit('make it look like a painting') → image_diff(original, edited)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path":            {"type": "string", "description": "Source image filename or path."},
                "prompt":          {"type": "string", "description": "Editing instruction."},
                "negative_prompt": {"type": "string", "description": "What to avoid (optional)."},
                "model":           {"type": "string", "description": "Model name (optional)."},
                "size":            {"type": "string", "description": "Output size (optional)."},
                "strength":        {"type": "number", "description": "0.0–1.0 — how much to change. Default 0.75."},
                "n":               {"type": "integer", "description": "Number of variations (default 1)."},
            },
            "required": ["path", "prompt"],
        },
    },
    {
        "name": "image_upscale",
        "description": (
            "Upscale an image using high-quality PIL LANCZOS resampling (no AI model required). "
            "Pipeline: image_generate → image_upscale(scale=4) → image_scan (read fine text)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path":    {"type": "string",  "description": "Image filename or path."},
                "scale":   {"type": "number",  "description": "Scale factor 1.1–8.0 (default 2.0)."},
                "sharpen": {"type": "boolean", "description": "Sharpen after upscale (default true)."},
            },
            "required": ["path"],
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
    {
        "name": "conv_search_history",
        "description": (
            "Search past conversation turns by semantic similarity using LM Studio embeddings. "
            "Returns the most relevant past exchanges from previous sessions. "
            "Requires LM Studio to be running for embedding generation."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find semantically similar past turns.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 5).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "server_time",
        "description": (
            "Returns the current server date and time with timezone information. "
            "Use this tool whenever you need to know the current time, date, day of week, "
            "or need to reason about time-sensitive information. "
            "Always call this tool before answering questions about 'today', 'now', 'current date', etc."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": (
                        "IANA timezone name (e.g. 'America/New_York', 'Europe/London', 'Asia/Tokyo'). "
                        "Defaults to the server's local timezone if omitted."
                    ),
                },
            },
            "required": [],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool dispatch — returns list of MCP content blocks
# ---------------------------------------------------------------------------

def _image_content_blocks(host_path: str, text: str) -> list[dict[str, Any]]:
    """Build MCP content list with a text summary and inline base64 image."""
    blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]
    if host_path and os.path.isfile(host_path):
        try:
            ext = os.path.splitext(host_path)[1].lower()
            mime = {
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".webp": "image/webp",
                ".gif": "image/gif",
            }.get(ext, "image/png")
            with open(host_path, "rb") as fh:
                b64 = base64.standard_b64encode(fh.read()).decode("ascii")
            blocks.append({"type": "image", "data": b64, "mimeType": mime})
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
            find_text  = str(arguments["find_text"]).strip()  if arguments.get("find_text")  else None
            find_image = str(arguments["find_image"]).strip() if arguments.get("find_image") else None
            pad = int(arguments.get("pad", 20))
            result = await mgr.run_browser(
                action=action,
                mode=_APPROVAL,
                confirmer=None,
                url=arguments.get("url"),
                selector=arguments.get("selector"),
                value=arguments.get("value"),
                code=arguments.get("code"),
                find_text=find_text,
                find_image=find_image,
                pad=pad,
            )
            if action == "screenshot":
                host_path = result.get("host_path", "")
                error = result.get("error", "")
                if error and not host_path:
                    return _text(f"Screenshot failed: {error}")
                page = result.get("title", "") or result.get("url", "")
                clipped = result.get("clipped", False)
                image_meta = result.get("image_meta", {})
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
                    f"Screenshot saved.\n"
                    f"File: {host_path}\n"
                    f"Page: {page}{clip_note}"
                )
                return _image_content_blocks(host_path, summary)
            if action == "screenshot_element":
                host_path = result.get("host_path", "")
                error = result.get("error", "")
                if error:
                    return _text(f"screenshot_element failed: {error}")
                bbox = result.get("bbox", {})
                bbox_note = (
                    f"  bbox: x={bbox.get('x',0):.0f}, y={bbox.get('y',0):.0f}, "
                    f"w={bbox.get('width',0):.0f}, h={bbox.get('height',0):.0f}"
                    if bbox else ""
                )
                summary = f"Element screenshot: {arguments.get('selector','')}\nFile: {host_path}{bbox_note}"
                return _image_content_blocks(host_path, summary)
            if action == "list_images_detail":
                images = result.get("images", result) if isinstance(result, dict) else result
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
            return _text(json.dumps(result, ensure_ascii=False))

        if name == "screenshot":
            url = str(arguments.get("url") or "").strip()
            if not url:
                return _text("screenshot: 'url' is required")
            find_text  = str(arguments["find_text"]).strip()  if arguments.get("find_text")  else None
            find_image = str(arguments["find_image"]).strip() if arguments.get("find_image") else None
            result = await mgr.run_browser(
                action="screenshot",
                mode=_APPROVAL,
                confirmer=None,
                url=url,
                find_text=find_text,
                find_image=find_image,
            )
            host_path = result.get("host_path", "")
            error = result.get("error", "")
            if error and not host_path:
                return _text(f"Screenshot failed: {error}")
            page = result.get("title", "") or result.get("url", url)
            clipped = result.get("clipped", False)
            image_meta = result.get("image_meta", {})
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
                f"Screenshot of: {page}\n"
                f"URL: {url}{clip_note}\n"
                f"File: {host_path}"
            )
            return _image_content_blocks(host_path, summary)

        if name == "fetch_image":
            url = str(arguments.get("url") or "").strip()
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
            query = str(arguments.get("query") or "").strip()
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
            url = str(arguments.get("url") or "").strip()
            if not url:
                return _text("web_fetch: 'url' is required")
            max_chars = int(arguments.get("max_chars", 4000))
            max_chars = max(500, min(max_chars, 16000))
            result = await mgr.run_web_fetch(url, max_chars, _APPROVAL, None)
            text = result.get("text", "")
            truncated = result.get("truncated", False)
            suffix = "\n...[truncated]" if truncated else ""
            return _text(f"{text}{suffix}" if text else "(no content)")

        if name == "memory_store":
            ttl = arguments.get("ttl_seconds")
            result = await mgr.run_memory_store(
                str(arguments.get("key", "")),
                str(arguments.get("value", "")),
                _APPROVAL, None,
                ttl_seconds=int(ttl) if ttl is not None else None,
            )
            return _text(json.dumps(result, ensure_ascii=False))

        if name == "memory_recall":
            result = await mgr.run_memory_recall(
                str(arguments.get("key", "")),
                _APPROVAL, None,
                pattern=str(arguments.get("pattern", "")),
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
                limit=int(arguments.get("limit", 20)),
                offset=int(arguments.get("offset", 0)),
                summary_only=bool(arguments.get("summary_only", False)),
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
            query = str(arguments.get("query") or "").strip()
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

        if name == "call_custom_tool":
            tool_name = str(arguments.get("tool_name", "")).strip()
            params = arguments.get("params") or {}
            if not tool_name:
                return _text("call_custom_tool: 'tool_name' is required")
            result = await mgr.run_custom_tool(tool_name, params, _APPROVAL, None)
            return _text(json.dumps(result, ensure_ascii=False))

        if name in ("image_crop", "image_zoom", "image_scan", "image_enhance"):
            try:
                from PIL import Image as _PI, ImageEnhance as _IE, ImageFilter as _IF
                import io as _io_mod
            except ImportError:
                return _text(f"{name}: Pillow is not installed. Run: pip install Pillow")

            _WORKSPACE = "/docker/human_browser/workspace"
            raw_path = str(arguments.get("path", "")).strip()

            def _resolve(p: str) -> str | None:
                basename = os.path.basename(p)
                if "/" not in p or p.startswith("/workspace/") or p.startswith("/docker/human_browser/workspace/"):
                    c = os.path.join(_WORKSPACE, basename)
                    return c if os.path.isfile(c) else None
                return p if os.path.isfile(p) else None

            local = _resolve(raw_path)
            if not local:
                return _text(f"{name}: image not found — tried '{raw_path}' in {_WORKSPACE}")

            with _PI.open(local) as _s:
                img = _s.copy()
            w, h = img.size
            left   = int(arguments.get("left",   0))
            top    = int(arguments.get("top",    0))
            right  = int(arguments.get("right",  w))
            bottom = int(arguments.get("bottom", h))
            if right  <= 0: right  = w
            if bottom <= 0: bottom = h
            right, bottom = min(right, w), min(bottom, h)
            box = (max(0, left), max(0, top), right, bottom)

            def _encode(pil_img, quality: int = 90) -> str:
                buf = _io_mod.BytesIO()
                pil_img.convert("RGB").save(buf, format="JPEG", quality=quality)
                return base64.standard_b64encode(buf.getvalue()).decode("ascii")

            if name == "image_crop":
                result = img.crop(box)
                rw, rh = result.size
                summary = f"Cropped ({box[0]},{box[1]})→({box[2]},{box[3]}) | orig {w}×{h} | result {rw}×{rh}"
                return [{"type": "text", "text": summary},
                        {"type": "image", "data": _encode(result), "mimeType": "image/jpeg"}]

            if name == "image_zoom":
                scale = max(1.1, min(float(arguments.get("scale", 2.0)), 8.0))
                region = img.crop(box) if box != (0, 0, w, h) else img
                rw, rh = region.size
                zoomed = region.resize((int(rw * scale), int(rh * scale)), _PI.LANCZOS)
                summary = f"Zoomed {scale:.1f}× | region {rw}×{rh} | output {zoomed.width}×{zoomed.height}"
                return [{"type": "text", "text": summary},
                        {"type": "image", "data": _encode(zoomed, 92), "mimeType": "image/jpeg"}]

            if name == "image_scan":
                region = img.crop(box) if box != (0, 0, w, h) else img
                rw, rh = region.size
                if rw < 800:
                    up = max(2, 800 // max(rw, 1))
                    region = region.resize((rw * up, rh * up), _PI.LANCZOS)
                    rw, rh = region.size
                region = region.convert("L")
                region = _IE.Contrast(region).enhance(2.5)
                region = _IE.Sharpness(region).enhance(3.0)
                region = region.filter(_IF.UnsharpMask(radius=1, percent=150, threshold=3))
                summary = (
                    f"Scan-enhanced (greyscale + contrast×2.5 + sharpen×3.0) | {rw}×{rh}\n"
                    f"Read all text visible in this image."
                )
                return [{"type": "text", "text": summary},
                        {"type": "image", "data": _encode(region, 95), "mimeType": "image/jpeg"}]

            if name == "image_enhance":
                contrast   = max(0.5, min(float(arguments.get("contrast",   1.5)), 4.0))
                sharpness  = max(0.5, min(float(arguments.get("sharpness",  1.5)), 4.0))
                brightness = max(0.5, min(float(arguments.get("brightness", 1.0)), 3.0))
                grayscale  = bool(arguments.get("grayscale", False))
                result = img.copy()
                if grayscale:
                    result = result.convert("L").convert("RGB")
                result = _IE.Contrast(result).enhance(contrast)
                result = _IE.Sharpness(result).enhance(sharpness)
                result = _IE.Brightness(result).enhance(brightness)
                summary = (
                    f"Enhanced: contrast={contrast:.1f} sharpness={sharpness:.1f} "
                    f"brightness={brightness:.1f}" + (" greyscale" if grayscale else "")
                )
                return [{"type": "text", "text": summary},
                        {"type": "image", "data": _encode(result), "mimeType": "image/jpeg"}]

        if name in ("image_stitch", "image_diff", "image_annotate"):
            try:
                from PIL import (
                    Image as _PI,
                    ImageEnhance as _IE,
                    ImageChops as _IC,
                    ImageDraw as _ID,
                )
                import io as _io_mod
            except ImportError:
                return _text(f"{name}: Pillow is not installed. Run: pip install Pillow")
            _WORKSPACE = "/docker/human_browser/workspace"

            def _resolve(p: str) -> str | None:
                basename = os.path.basename(p)
                if "/" not in p or p.startswith("/workspace/") or p.startswith("/docker/human_browser/workspace/"):
                    c = os.path.join(_WORKSPACE, basename)
                    return c if os.path.isfile(c) else None
                return p if os.path.isfile(p) else None

            def _encode(pil_img, quality: int = 90) -> str:
                buf = _io_mod.BytesIO()
                pil_img.convert("RGB").save(buf, format="JPEG", quality=quality)
                return base64.standard_b64encode(buf.getvalue()).decode("ascii")

            def _pil_blocks(pil_img, summary: str, quality: int = 90) -> list[dict[str, Any]]:
                return [{"type": "text", "text": summary},
                        {"type": "image", "data": _encode(pil_img, quality), "mimeType": "image/jpeg"}]

            if name == "image_stitch":
                paths = [str(p) for p in arguments.get("paths", [])]
                if len(paths) < 2:
                    return _text("image_stitch: at least 2 paths are required")
                paths = paths[:8]
                direction = str(arguments.get("direction", "vertical")).lower()
                gap = max(0, int(arguments.get("gap", 0)))
                images: list = []
                for p in paths:
                    loc = _resolve(p)
                    if not loc:
                        return _text(f"image_stitch: image not found — '{p}'")
                    with _PI.open(loc) as im:
                        images.append(im.convert("RGB").copy())
                if direction == "horizontal":
                    tw = sum(im.width for im in images) + gap * (len(images) - 1)
                    mh = max(im.height for im in images)
                    canvas = _PI.new("RGB", (tw, mh), (255, 255, 255))
                    x = 0
                    for im in images:
                        canvas.paste(im, (x, 0))
                        x += im.width + gap
                else:
                    mw = max(im.width for im in images)
                    th = sum(im.height for im in images) + gap * (len(images) - 1)
                    canvas = _PI.new("RGB", (mw, th), (255, 255, 255))
                    y = 0
                    for im in images:
                        canvas.paste(im, (0, y))
                        y += im.height + gap
                summary = f"Stitched {len(images)} images ({direction}) → {canvas.width}×{canvas.height}"
                return _pil_blocks(canvas, summary, quality=85)

            if name == "image_diff":
                path_a = str(arguments.get("path_a", "")).strip()
                path_b = str(arguments.get("path_b", "")).strip()
                loc_a = _resolve(path_a)
                loc_b = _resolve(path_b)
                if not loc_a:
                    return _text(f"image_diff: path_a not found — '{path_a}'")
                if not loc_b:
                    return _text(f"image_diff: path_b not found — '{path_b}'")
                amplify = max(1.0, min(float(arguments.get("amplify", 3.0)), 10.0))
                with _PI.open(loc_a) as ia:
                    img_a = ia.convert("RGB").copy()
                with _PI.open(loc_b) as ib:
                    img_b = ib.convert("RGB").copy()
                if img_a.size != img_b.size:
                    img_b = img_b.resize(img_a.size, _PI.LANCZOS)
                diff = _IC.difference(img_a, img_b)
                diff_l = diff.convert("L")
                diff_l = _IE.Brightness(diff_l).enhance(amplify)
                white = _PI.new("RGB", img_a.size, (255, 255, 255))
                red   = _PI.new("RGB", img_a.size, (220, 30, 30))
                result = _PI.composite(red, white, diff_l)
                summary = (
                    f"Pixel diff: {os.path.basename(path_a)} vs {os.path.basename(path_b)}\n"
                    f"Amplify: {amplify:.1f}×  Size: {img_a.width}×{img_a.height} — red = changed"
                )
                return _pil_blocks(result, summary)

            if name == "image_annotate":
                path = str(arguments.get("path", "")).strip()
                boxes = arguments.get("boxes", [])
                if not path:
                    return _text("image_annotate: 'path' is required")
                if not boxes:
                    return _text("image_annotate: 'boxes' list is required")
                loc = _resolve(path)
                if not loc:
                    return _text(f"image_annotate: image not found — '{path}'")
                outline_width = max(1, int(arguments.get("outline_width", 3)))
                with _PI.open(loc) as src:
                    img = src.convert("RGB").copy()
                draw = _ID.Draw(img)
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
                return _pil_blocks(img, summary)

        if name == "page_extract":
            include = list(arguments.get("include") or ["links", "headings", "tables", "images", "meta", "text"])
            max_links = max(1, int(arguments.get("max_links", 50)))
            max_text  = max(100, int(arguments.get("max_text", 3000)))
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
            result = await mgr.run_browser(
                action="eval", mode=_APPROVAL, confirmer=None, code=js,
            )
            raw = result.get("result", "{}")
            data: dict = json.loads(raw) if isinstance(raw, str) else (raw or {})
            lines_out: list[str] = []
            if data.get("title"):
                lines_out.append(f"Title: {data['title']}")
            for k, v in list((data.get("meta") or {}).items())[:10]:
                lines_out.append(f"  meta[{k}]: {str(v)[:120]}")
            if "headings" in data:
                lines_out.append(f"\nHeadings ({len(data['headings'])}):")
                for hd in data["headings"]:
                    lines_out.append(f"  {hd['tag']}: {hd['text'][:120]}")
            if "links" in data:
                lines_out.append(f"\nLinks ({len(data['links'])}):")
                for lk in data["links"]:
                    lines_out.append(f"  [{lk['text'][:60]}] → {lk['href'][:120]}")
            if "images" in data:
                lines_out.append(f"\nImages ({len(data['images'])}):")
                for im in data["images"][:10]:
                    lines_out.append(f"  {im['src'][:100]}  alt='{im['alt'][:60]}'")
            if "tables" in data:
                lines_out.append(f"\nTables ({len(data['tables'])}):")
                for ti, tbl in enumerate(data["tables"]):
                    lines_out.append(f"  Table {ti+1} ({len(tbl)} rows):")
                    for row in tbl[:5]:
                        lines_out.append("    | " + " | ".join(str(cell)[:30] for cell in row))
            if "text" in data:
                lines_out.append(f"\nText excerpt:\n{data['text'][:1500]}")
            return _text("\n".join(lines_out) or "No data extracted.")

        if name == "extract_article":
            url = str(arguments.get("url") or "").strip()
            if not url:
                return _text("extract_article: 'url' is required")
            max_chars = max(500, min(int(arguments.get("max_chars", 8000)), 64000))
            result = await mgr.run_web_fetch(url, max_chars, _APPROVAL, None)
            text_out = result.get("text", "")
            clean_lines = [ln.strip() for ln in text_out.splitlines() if ln.strip()]
            clean = "\n".join(clean_lines)[:max_chars]
            title = result.get("title", "")
            final_url = result.get("url", url)
            header = f"Title: {title}\nURL:   {final_url}\n\n" if title else f"URL: {final_url}\n\n"
            return _text(header + clean)

        if name == "page_scrape":
            result = await mgr.run_page_scrape(
                url=str(arguments.get("url") or ""),
                mode=_APPROVAL, confirmer=None,
                max_scrolls=max(1, min(int(arguments.get("max_scrolls", 10)), 30)),
                wait_ms=max(100, min(int(arguments.get("wait_ms", 500)), 3000)),
                max_chars=max(500, min(int(arguments.get("max_chars", 16000)), 64000)),
                include_links=bool(arguments.get("include_links", False)),
            )
            if result.get("error"):
                return _text(f"page_scrape error: {result['error']}")
            title     = result.get("title", "")
            content   = result.get("content", "")
            final_url = result.get("url", "")
            steps     = result.get("scroll_steps", 0)
            grew      = result.get("content_grew_on_scroll", False)
            height    = result.get("final_page_height", 0)
            chars     = result.get("char_count", len(content))
            header = (
                f"Title: {title}\nURL: {final_url}\n"
                f"Scrolled: {steps} steps | Page height: {height}px"
                + (" | lazy content grew" if grew else "")
                + f" | {chars} chars extracted\n\n"
            )
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            body = "\n".join(lines)
            result_text = header + body
            if arguments.get("include_links") and result.get("links"):
                link_lines = [f"[{lk.get('text','')[:60]}] → {lk.get('href','')[:120]}"
                              for lk in result["links"][:100]]
                result_text += f"\n\nLinks ({len(result['links'])}):\n" + "\n".join(link_lines)
            return _text(result_text)

        if name == "page_images":
            url         = str(arguments.get("url", "")).strip()
            scroll      = bool(arguments.get("scroll", True))
            max_scrolls = max(1, min(int(arguments.get("max_scrolls", 3)), 20))
            if not url:
                return _text("page_images: 'url' is required")
            result = await mgr.run_page_images(
                url=url, mode=_APPROVAL, confirmer=None,
                scroll=scroll, max_scrolls=max_scrolls,
            )
            if result.get("error"):
                return _text(f"page_images error: {result['error']}")
            imgs      = result.get("images", [])
            count     = result.get("count", len(imgs))
            title     = result.get("title", "")
            final_url = result.get("url", url)
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

        if name == "image_search":
            query      = str(arguments.get("query", "")).strip()
            img_count  = max(1, min(int(arguments.get("count", 4)), 20))
            img_offset = max(0, int(arguments.get("offset", 0)))
            if not query:
                return _text("image_search: 'query' is required")
            return await mgr.run_image_search(
                query=query, count=img_count, mode=_APPROVAL, confirmer=None,
                offset=img_offset,
            )

        if name == "bulk_screenshot":
            urls = [str(u).strip() for u in arguments.get("urls", []) if str(u).strip()]
            if not urls:
                return _text("bulk_screenshot: 'urls' list is required")
            urls = urls[:6]

            async def _shot(shot_url: str, idx: int) -> list[dict[str, Any]]:
                result = await mgr.run_browser(
                    action="screenshot", mode=_APPROVAL, confirmer=None, url=shot_url,
                )
                host_path = result.get("host_path", "")
                error = result.get("error", "")
                title = result.get("title", shot_url)
                summary = f"[{idx+1}/{len(urls)}] {title}\n{shot_url}\nFile: {host_path}"
                if host_path and os.path.isfile(host_path):
                    return _image_content_blocks(host_path, summary)
                return _text(f"[{idx+1}] {shot_url} — failed: {error or 'screenshot missing'}")

            tasks = [_shot(u, i) for i, u in enumerate(urls)]
            results_bulk = await asyncio.gather(*tasks)
            combined_bulk: list[dict[str, Any]] = []
            for blk in results_bulk:
                combined_bulk.extend(blk)
            return combined_bulk

        if name == "scroll_screenshot":
            try:
                from PIL import Image as _PI_sc
                import io as _io_sc
            except ImportError:
                return _text("scroll_screenshot: Pillow is not installed. Run: pip install Pillow")
            url = str(arguments.get("url", "")).strip() or None
            max_scrolls = max(1, min(int(arguments.get("max_scrolls", 5)), 10))
            overlap     = max(0, int(arguments.get("scroll_overlap", 100)))
            _WORKSPACE_SC = "/docker/human_browser/workspace"
            if url:
                await mgr.run_browser(action="navigate", mode=_APPROVAL, confirmer=None, url=url)
            # Get page/viewport height
            try:
                ph_r = await mgr.run_browser(action="eval", mode=_APPROVAL, confirmer=None,
                                              code="document.documentElement.scrollHeight")
                page_h = int(ph_r.get("result", 0) or 0)
                vh_r = await mgr.run_browser(action="eval", mode=_APPROVAL, confirmer=None,
                                              code="window.innerHeight")
                vp_h = int(vh_r.get("result", 800) or 800)
            except Exception:
                page_h, vp_h = 0, 800
            step = max(vp_h - overlap, 100)
            frames: list = []
            for i in range(max_scrolls):
                scroll_y = i * step
                if page_h > 0 and scroll_y >= page_h:
                    break
                try:
                    await mgr.run_browser(action="eval", mode=_APPROVAL, confirmer=None,
                                          code=f"window.scrollTo(0, {scroll_y})")
                    await asyncio.sleep(0.3)
                    shot_r = await mgr.run_browser(action="screenshot", mode=_APPROVAL, confirmer=None)
                    hp = shot_r.get("host_path", "")
                    if hp and os.path.isfile(hp):
                        with _PI_sc.open(hp) as fr:
                            frames.append(fr.convert("RGB").copy())
                except Exception:
                    break
            if not frames:
                return _text("scroll_screenshot: no frames captured.")
            total_h = sum(f.height for f in frames)
            max_w   = max(f.width  for f in frames)
            canvas  = _PI_sc.new("RGB", (max_w, total_h), (255, 255, 255))
            y = 0
            for fr in frames:
                canvas.paste(fr, (0, y))
                y += fr.height
            buf_sc = _io_sc.BytesIO()
            canvas.convert("RGB").save(buf_sc, format="JPEG", quality=85)
            b64_sc = base64.standard_b64encode(buf_sc.getvalue()).decode("ascii")
            summary = (
                f"Full-page scroll screenshot: {len(frames)} frames stitched\n"
                f"Canvas: {canvas.width}×{canvas.height}" + (f"  URL: {url}" if url else "")
            )
            return [{"type": "text", "text": summary},
                    {"type": "image", "data": b64_sc, "mimeType": "image/jpeg"}]

        if name in ("image_generate", "image_edit"):
            import httpx as _httpx
            _IMG_BASE = os.environ.get(
                "IMAGE_GEN_BASE_URL",
                os.environ.get("LM_STUDIO_URL", "http://localhost:1234"),
            )
            _IMG_MODEL = os.environ.get("IMAGE_GEN_MODEL", "")
            _WORKSPACE_IG = "/docker/human_browser/workspace"

            try:
                from PIL import Image as _PI_ig
                import io as _io_ig
                _PIL_OK = True
            except ImportError:
                _PIL_OK = False

            def _save_gen(b64: str, prefix: str) -> str:
                """Save base64 image to workspace; return filename or ''."""
                import datetime as _dt
                ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"{prefix}_0_{ts}.jpg"
                lp = os.path.join(_WORKSPACE_IG, fname)
                try:
                    raw = base64.b64decode(b64)
                    if _PIL_OK:
                        with _PI_ig.open(_io_ig.BytesIO(raw)) as gi:
                            gi.convert("RGB").save(lp, format="JPEG", quality=95)
                    else:
                        with open(lp, "wb") as fh:
                            fh.write(raw)
                    return fname
                except Exception:
                    return ""

            def _resize_b64(b64: str) -> tuple[str, str]:
                """Resize to max 1280px, return (b64_jpeg, mime)."""
                if not _PIL_OK:
                    return b64, "image/png"
                try:
                    with _PI_ig.open(_io_ig.BytesIO(base64.b64decode(b64))) as gi:
                        gi = gi.convert("RGB")
                        if gi.width > 1280 or gi.height > 1280:
                            gi.thumbnail((1280, 1280), _PI_ig.LANCZOS)
                        buf = _io_ig.BytesIO()
                        gi.save(buf, format="JPEG", quality=90)
                    return base64.standard_b64encode(buf.getvalue()).decode("ascii"), "image/jpeg"
                except Exception:
                    return b64, "image/png"

            if name == "image_generate":
                prompt = str(arguments.get("prompt", "")).strip()
                if not prompt:
                    return _text("image_generate: 'prompt' is required")
                model = str(arguments.get("model", _IMG_MODEL)).strip() or None
                size  = str(arguments.get("size", "512x512")).strip()
                n     = max(1, min(int(arguments.get("n", 1)), 4))
                neg   = str(arguments.get("negative_prompt", "")).strip() or None
                steps = arguments.get("steps")
                gs    = arguments.get("guidance_scale")
                seed  = arguments.get("seed")
                payload: dict = {"prompt": prompt, "n": n, "size": size, "response_format": "b64_json"}
                if model:                 payload["model"]               = model
                if neg:                   payload["negative_prompt"]      = neg
                if steps:                 payload["num_inference_steps"]  = int(steps)
                if gs is not None:        payload["guidance_scale"]       = float(gs)
                if seed is not None and int(seed) >= 0: payload["seed"]  = int(seed)
                gen_url = f"{_IMG_BASE}/v1/images/generations"
                try:
                    async with _httpx.AsyncClient(timeout=180.0) as hc:
                        r = await hc.post(gen_url, json=payload)
                        r.raise_for_status()
                        data = r.json()
                except Exception as exc:
                    return _text(
                        f"image_generate failed: {exc}\n"
                        f"Endpoint: {gen_url}\n"
                        "→ Load an image generation model (FLUX, SDXL) in LM Studio, or\n"
                        "→ set IMAGE_GEN_BASE_URL to your Automatic1111/ComfyUI endpoint."
                    )
                images = data.get("data", [])
                if not images:
                    err = data.get("error", {})
                    msg = err.get("message", str(data)) if isinstance(err, dict) else str(err)
                    return _text(f"image_generate: no images returned — {msg}")
                blocks_g: list[dict[str, Any]] = []
                for i, img_data in enumerate(images):
                    b64 = img_data.get("b64_json", "")
                    img_url = img_data.get("url", "")
                    revised = img_data.get("revised_prompt", "")
                    save_note = ""
                    if b64 and os.path.isdir(_WORKSPACE_IG):
                        fname = _save_gen(b64, f"generated_{i}")
                        if fname:
                            save_note = f"\n→ Saved as: {fname}  (use as 'path' in image_crop/upscale/annotate)"
                    summary = f"Generated image {i+1}/{n}\nPrompt: {prompt[:200]}"
                    if revised:    summary += f"\nRevised: {revised[:150]}"
                    summary += save_note
                    if b64:
                        b64r, mime = _resize_b64(b64)
                        blocks_g.extend([{"type": "text", "text": summary},
                                          {"type": "image", "data": b64r, "mimeType": mime}])
                    elif img_url:
                        blocks_g.append({"type": "text", "text": f"{summary}\nURL: {img_url}"})
                return blocks_g if blocks_g else _text("image_generate: no image data in response")

            if name == "image_edit":
                path   = str(arguments.get("path", "")).strip()
                prompt = str(arguments.get("prompt", "")).strip()
                if not path:   return _text("image_edit: 'path' is required")
                if not prompt: return _text("image_edit: 'prompt' is required")
                _W2 = "/docker/human_browser/workspace"
                basename2 = os.path.basename(path)
                if "/" not in path or path.startswith("/workspace/") or path.startswith("/docker/human_browser/workspace/"):
                    local2 = os.path.join(_W2, basename2) if os.path.isfile(os.path.join(_W2, basename2)) else None
                else:
                    local2 = path if os.path.isfile(path) else None
                if not local2:
                    return _text(f"image_edit: image not found — '{path}'")
                model    = str(arguments.get("model", _IMG_MODEL)).strip() or None
                neg      = str(arguments.get("negative_prompt", "")).strip() or None
                strength = max(0.0, min(float(arguments.get("strength", 0.75)), 1.0))
                n2       = max(1, min(int(arguments.get("n", 1)), 4))
                size2    = str(arguments.get("size", "")).strip() or None
                ext2 = os.path.splitext(local2)[1].lower()
                mime2 = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                         ".webp": "image/webp"}.get(ext2, "image/jpeg")
                with open(local2, "rb") as fh:
                    img_bytes2 = fh.read()
                edit_url = f"{_IMG_BASE}/v1/images/edits"
                form2: dict = {"prompt": prompt, "n": str(n2), "response_format": "b64_json",
                               "strength": str(strength)}
                if model:  form2["model"] = model
                if neg:    form2["negative_prompt"] = neg
                if size2:  form2["size"] = size2
                try:
                    async with _httpx.AsyncClient(timeout=180.0) as hc2:
                        r2 = await hc2.post(edit_url,
                                            files={"image": (basename2, img_bytes2, mime2)},
                                            data=form2)
                        if r2.status_code in (404, 405, 422):
                            b64_in = base64.standard_b64encode(img_bytes2).decode("ascii")
                            jp: dict = {"prompt": prompt, "n": n2, "response_format": "b64_json",
                                        "strength": strength, "image": f"data:{mime2};base64,{b64_in}"}
                            if model:  jp["model"] = model
                            if neg:    jp["negative_prompt"] = neg
                            if size2:  jp["size"] = size2
                            r2 = await hc2.post(edit_url, json=jp)
                        r2.raise_for_status()
                        data2 = r2.json()
                except Exception as exc:
                    return _text(
                        f"image_edit failed: {exc}\n"
                        f"Endpoint: {edit_url}\n"
                        "→ img2img support varies by model. Try Automatic1111 for full support."
                    )
                images2 = data2.get("data", [])
                if not images2:
                    err2 = data2.get("error", {})
                    msg2 = err2.get("message", str(data2)) if isinstance(err2, dict) else str(err2)
                    return _text(f"image_edit: no images returned — {msg2}")
                blocks_e2: list[dict[str, Any]] = []
                for i, img_data in enumerate(images2):
                    b64e = img_data.get("b64_json", "")
                    revised2 = img_data.get("revised_prompt", "")
                    save_note2 = ""
                    if b64e and os.path.isdir(_W2):
                        fname2 = _save_gen(b64e, f"edited_{i}")
                        if fname2: save_note2 = f"\n→ Saved as: {fname2}"
                    summary2 = (f"Edited image {i+1}/{n2}  Source: {basename2}\n"
                                f"Prompt: {prompt[:200]}  strength={strength:.2f}")
                    if revised2: summary2 += f"\nRevised: {revised2[:150]}"
                    summary2 += save_note2
                    if b64e:
                        b64r2, mime_r2 = _resize_b64(b64e)
                        blocks_e2.extend([{"type": "text", "text": summary2},
                                           {"type": "image", "data": b64r2, "mimeType": mime_r2}])
                return blocks_e2 if blocks_e2 else _text("image_edit: no image data in response")

        if name == "image_upscale":
            try:
                from PIL import Image as _PI_up, ImageEnhance as _IE_up, \
                                ImageFilter as _IF_up, ImageOps as _IO_up
                import io as _io_up
            except ImportError:
                return _text("image_upscale: Pillow is not installed. Run: pip install Pillow")
            _WU = "/docker/human_browser/workspace"
            path = str(arguments.get("path", "")).strip()
            if not path:
                return _text("image_upscale: 'path' is required")
            bname_u = os.path.basename(path)
            if "/" not in path or path.startswith("/workspace/") or path.startswith("/docker/human_browser/workspace/"):
                local_u = os.path.join(_WU, bname_u) if os.path.isfile(os.path.join(_WU, bname_u)) else None
            else:
                local_u = path if os.path.isfile(path) else None
            if not local_u:
                return _text(f"image_upscale: image not found — '{path}'")
            scale_u   = max(1.1, min(float(arguments.get("scale", 2.0)), 8.0))
            sharpen_u = bool(arguments.get("sharpen", True))
            with _PI_up.open(local_u) as src_u:
                img_u = _IO_up.exif_transpose(src_u).convert("RGB")
            ow_u, oh_u = img_u.size
            # Safety cap: clamp scale so no output dimension exceeds 8192 px
            max_dim_u = max(ow_u, oh_u)
            if max_dim_u * scale_u > 8192:
                scale_u = 8192 / max_dim_u
                capped_u = True
            else:
                capped_u = False
            nw_u, nh_u = int(ow_u * scale_u), int(oh_u * scale_u)
            up_u = img_u.resize((nw_u, nh_u), _PI_up.LANCZOS)
            if sharpen_u:
                up_u = _IE_up.Sharpness(up_u).enhance(1.4)
                up_u = up_u.filter(_IF_up.UnsharpMask(radius=0.5, percent=80, threshold=2))
            buf_u = _io_up.BytesIO()
            up_u.convert("RGB").save(buf_u, format="JPEG", quality=92)
            b64_u = base64.standard_b64encode(buf_u.getvalue()).decode("ascii")
            summary_u = (
                f"Upscaled {scale_u:.2f}×  {ow_u}×{oh_u} → {nw_u}×{nh_u}"
                + (" + sharpen" if sharpen_u else "")
                + (" [scale capped to 8192px]" if capped_u else "")
                + f"\nSource: {bname_u}"
            )
            # Save to workspace for pipeline chaining
            import datetime as _dt_u
            ts_u = _dt_u.datetime.now().strftime("%Y%m%d_%H%M%S")
            fname_u = f"upscaled_{ts_u}.jpg"
            try:
                up_u.save(os.path.join(_WU, fname_u), format="JPEG", quality=92)
                summary_u += f"\n→ Saved as: {fname_u}"
            except Exception:
                pass
            return [{"type": "text", "text": summary_u},
                    {"type": "image", "data": b64_u, "mimeType": "image/jpeg"}]

        if name == "shell_exec":
            output, _ = await mgr.run_shell(
                str(arguments.get("command", "")),
                _APPROVAL, None,
            )
            return _text(output or "(no output)")

        if name == "get_errors":
            limit = max(1, min(int(arguments.get("limit", 50)), 200))
            service = str(arguments.get("service", "")).strip()
            result = await mgr.run_get_errors(limit, service, _APPROVAL, None)
            errors = result.get("errors", [])
            if not errors:
                return _text("No errors logged yet." + (f" (service={service})" if service else ""))
            lines = [f"Recent errors ({len(errors)}):"]
            for e in errors:
                ts = str(e.get("logged_at", ""))[:19].replace("T", " ")
                lines.append(
                    f"  [{ts}] [{e['level']}] {e['service']}: {e['message']}"
                    + (f"\n    detail: {e['detail']}" if e.get("detail") else "")
                )
            return _text("\n".join(lines))

        if name == "browser_save_images":
            raw_urls = arguments.get("urls", [])
            if isinstance(raw_urls, str):
                raw_urls = [u.strip() for u in raw_urls.split(",") if u.strip()]
            if not raw_urls:
                return _text("browser_save_images: 'urls' is required")
            prefix = str(arguments.get("prefix", "image")).strip() or "image"
            max_imgs = int(arguments.get("max", 20))
            result = await mgr.run_browser(
                "save_images", _APPROVAL, None,
                image_urls=raw_urls, image_prefix=prefix, max_images=max_imgs,
            )
            saved = result.get("saved", [])
            errors = result.get("errors", [])
            if not saved and errors:
                errs = "; ".join(e.get("error", "?") for e in errors[:3])
                return _text(f"browser_save_images: all downloads failed — {errs}")
            blocks: list = []
            lines = [f"Downloaded {len(saved)} image(s)" +
                     (f"  ({len(errors)} failed)" if errors else "")]
            for item in saved:
                hp = item.get("host_path", "")
                size_kb = item.get("size", 0) // 1024
                lines.append(f"  [{item.get('index','?')}] {os.path.basename(hp)}  {size_kb} KB")
                if hp and os.path.isfile(hp) and len(blocks) < 10:
                    blocks.extend(_image_content_blocks(hp, os.path.basename(hp)))
            blocks.insert(0, _text("\n".join(lines))[0])
            return blocks

        if name == "browser_download_page_images":
            url = str(arguments.get("url", "")).strip() or None
            if url:
                await mgr.run_browser("navigate", _APPROVAL, None, url=url)
            prefix = str(arguments.get("prefix", "image")).strip() or "image"
            max_imgs = int(arguments.get("max", 20))
            filter_q = str(arguments.get("filter", "")).strip() or None
            result = await mgr.run_browser(
                "download_page_images", _APPROVAL, None,
                filter_query=filter_q, max_images=max_imgs, image_prefix=prefix,
            )
            saved = result.get("saved", [])
            errors = result.get("errors", [])
            applied_filter = result.get("filter")
            filter_note = f" (filter: '{applied_filter}')" if applied_filter else ""
            if not saved:
                no_img_msg = f"No images downloaded{filter_note}"
                if errors:
                    errs = "; ".join(e.get("error", "?") for e in errors[:3])
                    no_img_msg += f" — {errs}"
                return _text(no_img_msg)
            blocks = []
            lines = [f"Downloaded {len(saved)} image(s){filter_note}" +
                     (f"  ({len(errors)} failed)" if errors else "")]
            for item in saved:
                hp = item.get("host_path", "")
                size_kb = item.get("size", 0) // 1024
                lines.append(f"  [{item.get('index','?')}] {os.path.basename(hp)}  {size_kb} KB")
                if hp and os.path.isfile(hp) and len(blocks) < 10:
                    blocks.extend(_image_content_blocks(hp, os.path.basename(hp)))
            blocks.insert(0, _text("\n".join(lines))[0])
            return blocks

        if name == "conv_search_history":
            query = str(arguments.get("query", "")).strip()
            limit = int(arguments.get("limit", 5))
            if not query:
                return _text("conv_search_history: 'query' is required")
            results = await mgr.run_conv_search_history(query, limit, _APPROVAL, None)
            if not results:
                return _text("No matching past conversation turns found.")
            lines = [f"Past turns matching '{query}' ({len(results)} results):"]
            for r in results:
                ts = str(r.get("timestamp", ""))[:10]
                role = r.get("role", "")
                snippet = str(r.get("content", ""))[:200].replace("\n", " ")
                sim = round(r.get("similarity", 0), 3)
                lines.append(f"  [{ts}] {role} (sim={sim}): {snippet}")
            return _text("\n".join(lines))

        if name == "server_time":
            from datetime import datetime, timezone
            import zoneinfo
            tz_name = str(arguments.get("timezone", "")).strip()
            try:
                if tz_name:
                    tz = zoneinfo.ZoneInfo(tz_name)
                else:
                    tz = None  # use local timezone
            except (zoneinfo.ZoneInfoNotFoundError, KeyError):
                return _text(f"server_time: unknown timezone '{tz_name}'")
            now = datetime.now(tz=tz or None)
            utc_now = datetime.now(timezone.utc)
            offset = now.strftime("%z") or "local"
            return _text(
                f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')} ({offset})\n"
                f"UTC time:     {utc_now.strftime('%Y-%m-%d %H:%M:%S')} (UTC)\n"
                f"Day of week:  {now.strftime('%A')}\n"
                f"Timezone:     {tz_name or 'server local'}\n"
                f"Unix epoch:   {int(utc_now.timestamp())}"
            )

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
        client_ver = params.get("protocolVersion", "2024-11-05")
        agreed_ver = client_ver if client_ver in {"2024-11-05", "2025-03-26"} else "2024-11-05"
        _write(_ok(req_id, {
            "protocolVersion": agreed_ver,
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
    loop = asyncio.get_running_loop()
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
