"""Shared helper functions and constants for the MCP gateway."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Browser-like headers for outbound HTTP requests
# ---------------------------------------------------------------------------

BROWSER_HEADERS: dict[str, str] = {
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
# Service URLs (from environment)
# ---------------------------------------------------------------------------

DATABASE_URL       = os.environ.get("DATABASE_URL",       "http://aichat-data:8091")
MEMORY_URL         = os.environ.get("MEMORY_URL",         "http://aichat-data:8091/memory")
RESEARCH_URL       = os.environ.get("RESEARCH_URL",       "http://aichat-data:8091/research")
TOOLKIT_URL        = os.environ.get("TOOLKIT_URL",        "http://aichat-sandbox:8095")
GRAPH_URL          = os.environ.get("GRAPH_URL",          "http://aichat-data:8091/graph")
VECTOR_URL         = os.environ.get("VECTOR_URL",         "http://aichat-vector:6333")
VIDEO_URL          = os.environ.get("VIDEO_URL",          "http://aichat-vision:8099")
OCR_URL            = os.environ.get("OCR_URL",            "http://aichat-vision:8099/ocr")
DOCS_URL           = os.environ.get("DOCS_URL",           "http://aichat-docs:8101")
PLANNER_URL        = os.environ.get("PLANNER_URL",        "http://aichat-data:8091/planner")
PDF_URL            = os.environ.get("PDF_URL",            "http://aichat-docs:8101/pdf")
JOB_URL            = os.environ.get("JOB_URL",            "http://aichat-data:8091/jobs")
SEARXNG_URL        = os.environ.get("SEARXNG_URL",        "http://aichat-searxng:8080")
BROWSER_URL        = os.environ.get("BROWSER_URL",        "http://human_browser:7081")
JUPYTER_URL        = os.environ.get("JUPYTER_URL",        "http://aichat-jupyter:8098")
BROWSER_WORKSPACE  = Path(os.environ.get("BROWSER_WORKSPACE", "/browser-workspace"))
IMAGE_GEN_BASE_URL = os.environ.get("IMAGE_GEN_BASE_URL", "http://192.168.50.2:1234")
IMAGE_GEN_MODEL    = os.environ.get("IMAGE_GEN_MODEL",    "")
EMBED_MODEL        = os.environ.get("EMBED_MODEL",        "")
MINIO_URL          = os.environ.get("MINIO_URL",          "http://aichat-minio:9002")
MINIO_ACCESS       = os.environ.get("MINIO_ROOT_USER",    "minioadmin")
MINIO_SECRET       = os.environ.get("MINIO_ROOT_PASSWORD", "")
CLIP_URL           = os.environ.get("CLIP_URL",           "http://aichat-vision:8099/clip")
BROWSER_AUTO_URL   = os.environ.get("BROWSER_AUTO_URL",   "http://aichat-browser:8104")
DETECT_URL         = os.environ.get("DETECT_URL",         "http://aichat-vision:8099/detect")

# Jupyter API key (optional)
JUPYTER_API_KEY    = os.environ.get("JUPYTER_API_KEY",     "")


# ---------------------------------------------------------------------------
# MCP content block helpers
# ---------------------------------------------------------------------------

def text_block(s: str) -> list[dict[str, Any]]:
    """Return a single MCP text content block."""
    return [{"type": "text", "text": s}]


def json_or_err(r: Any, tool: str) -> list[dict[str, Any]]:
    """Return .json() as text, or an error message if status >= 400."""
    if r.status_code >= 400:
        return text_block(f"{tool}: upstream returned {r.status_code} — {r.text[:300]}")
    try:
        return text_block(json.dumps(r.json()))
    except Exception:
        return text_block(f"{tool}: upstream returned {r.status_code} (non-JSON)")


def image_content_block(data_b64: str, mime: str = "image/png") -> dict[str, Any]:
    """Return an MCP image content block."""
    return {"type": "image", "data": data_b64, "mimeType": mime}
