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
    topic = payload["topic"]
    feed_url = payload["feed_url"]
    parsed = feedparser.parse(feed_url)
    items = [
        {
            "title": e.get("title", "untitled"),
            "url": e.get("link", feed_url),
        }
        for e in parsed.entries[:20]
    ]
    stored = 0
    async with httpx.AsyncClient(timeout=30) as c:
        for item in items:
            try:
                r = await c.post(
                    f"{DB_API}/articles/store",
                    json={
                        "url": item["url"],
                        "title": item["title"],
                        "topic": topic,
                    },
                )
                r.raise_for_status()
                stored += 1
            except Exception:
                pass
    return {"topic": topic, "feed_url": feed_url, "inserted": stored}
