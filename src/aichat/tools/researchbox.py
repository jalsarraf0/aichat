from __future__ import annotations

import httpx


class ResearchBoxError(RuntimeError):
    pass


class ResearchboxTool:
    def __init__(self, base_url: str = "http://localhost:8092") -> None:
        self.base_url = base_url.rstrip("/")

    async def _request(self, method: str, path: str, **kwargs: object) -> dict:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.request(method, f"{self.base_url}{path}", **kwargs)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as exc:
            raise ResearchBoxError(f"ResearchBox request failed for {path}: {exc}") from exc

    async def rb_search_feeds(self, topic: str) -> dict:
        return await self._request("GET", "/search-feeds", params={"topic": topic})

    async def rb_push_feeds(self, feed_url: str, topic: str) -> dict:
        return await self._request("POST", "/push-feed", json={"feed_url": feed_url, "topic": topic})
