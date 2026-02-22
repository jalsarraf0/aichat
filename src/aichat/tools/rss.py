from __future__ import annotations

import httpx


class RSSTool:
    def __init__(self, base_url: str = "http://localhost:8091") -> None:
        self.base_url = base_url.rstrip("/")

    async def news_list_sources(self, topic: str) -> dict:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(f"{self.base_url}/sources", params={"topic": topic})
            r.raise_for_status()
            return r.json()

    async def news_refresh(self, topic: str) -> dict:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(f"{self.base_url}/refresh", json={"topic": topic})
            r.raise_for_status()
            return r.json()

    async def news_latest(self, topic: str) -> dict:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(f"{self.base_url}/latest", params={"topic": topic})
            r.raise_for_status()
            return r.json()
