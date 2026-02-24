"""aichat-researchbox: RSS feed discovery and article ingestion service.

/search-feeds  — suggest feed URLs for a topic (no external deps).
/push-feed     — fetch a feed and store its articles in aichat-database.
"""
from __future__ import annotations

import os

import feedparser
import httpx
from fastapi import FastAPI

app = FastAPI()

# aichat-database REST endpoint (set via DATABASE_URL env var).
DB_API = os.environ.get("DATABASE_URL", "http://localhost:8091")


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
