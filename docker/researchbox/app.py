"""aichat-researchbox: RSS feed discovery and article ingestion service.

/search-feeds  — suggest feed URLs for a topic (no external deps).
/push-feed     — fetch a feed and store its articles in aichat-database.
"""
from __future__ import annotations

import asyncio
import logging
import os

import feedparser
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("aichat-researchbox")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# aichat-database REST endpoint — used for both article storage and error logging.
DB_API = os.environ.get("DATABASE_URL", "http://aichat-database:8091")
_SERVICE_NAME = "aichat-researchbox"

app = FastAPI()


# ---------------------------------------------------------------------------
# Error reporting
# ---------------------------------------------------------------------------

async def _report_error(message: str, detail: str | None = None) -> None:
    """Fire-and-forget: send an error entry to aichat-database."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(
                f"{DB_API}/errors/log",
                json={"service": _SERVICE_NAME, "level": "ERROR",
                      "message": message, "detail": detail},
            )
    except Exception:
        pass  # never let error reporting crash the service


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    message = str(exc)
    detail = f"{request.method} {request.url.path}"
    log.error("Unhandled error [%s %s]: %s", request.method, request.url.path, exc, exc_info=True)
    asyncio.create_task(_report_error(message, detail))
    return JSONResponse(status_code=500, content={"error": message})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/search-feeds")
def search_feeds(topic: str) -> dict:
    return {
        "topic": topic,
        "feeds": [
            f"https://news.google.com/rss/search?q={topic}",
            f"https://hnrss.org/newest?q={topic}",
        ],
    }


@app.post("/push-feed")
async def push_feed(payload: dict) -> dict:
    topic    = str(payload.get("topic", "")).strip()
    feed_url = str(payload.get("feed_url", "")).strip()
    if not topic or not feed_url:
        return {"error": "topic and feed_url are required", "inserted": 0, "failed": 0}
    from urllib.parse import urlparse as _urlparse
    if _urlparse(feed_url).scheme not in ("http", "https"):
        return {"error": "feed_url must use http or https", "inserted": 0, "failed": 0}

    # feedparser.parse is synchronous; run in a thread to avoid blocking the event loop
    loop = asyncio.get_running_loop()
    try:
        parsed = await asyncio.wait_for(
            loop.run_in_executor(None, feedparser.parse, feed_url),
            timeout=20.0,
        )
    except asyncio.TimeoutError:
        return {"topic": topic, "feed_url": feed_url, "inserted": 0, "failed": 0,
                "errors": [{"url": feed_url, "error": "feed fetch timed out after 20s"}]}

    items = [
        {
            "title": e.get("title", "untitled"),
            "url": e.get("link", feed_url),
        }
        for e in getattr(parsed, "entries", [])[:20]
    ]

    stored = 0
    failed = 0
    errors: list[dict] = []
    async with httpx.AsyncClient(timeout=30) as c:
        for item in items:
            try:
                r = await c.post(
                    f"{DB_API}/articles/store",
                    json={"url": item["url"], "title": item["title"], "topic": topic},
                )
                r.raise_for_status()
                stored += 1
            except Exception as exc:
                failed += 1
                errors.append({"url": item["url"], "error": str(exc)})

    log.info("push_feed %s → %d inserted, %d failed", feed_url, stored, failed)
    result: dict = {"topic": topic, "feed_url": feed_url, "inserted": stored, "failed": failed}
    if errors:
        result["errors"] = errors
    return result
