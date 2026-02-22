from __future__ import annotations

import httpx


class ResearchboxTool:
    def __init__(self, base_url: str = "http://localhost:8092") -> None:
        self.base_url = base_url.rstrip("/")

    async def rb_search_feeds(self, topic: str) -> dict:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.get(f"{self.base_url}/search-feeds", params={"topic": topic})
            r.raise_for_status()
            return r.json()

    async def rb_push_feeds(self, feed_url: str, topic: str) -> dict:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(f"{self.base_url}/push-feed", json={"feed_url": feed_url, "topic": topic})
            r.raise_for_status()
            return r.json()
