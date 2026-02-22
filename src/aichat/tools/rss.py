from __future__ import annotations

import httpx

from .errors import ToolRequestError, is_retryable_status


class RSSToolError(ToolRequestError):
    pass


class RSSTool:
    def __init__(self, base_url: str = "http://localhost:8091") -> None:
        self.base_url = base_url.rstrip("/")

    async def _request(self, method: str, path: str, **kwargs: object) -> dict:
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.request(method, f"{self.base_url}{path}", **kwargs)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as exc:
            status = None
            retryable = False
            if isinstance(exc, httpx.HTTPStatusError):
                status = exc.response.status_code
                retryable = is_retryable_status(status)
            elif isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
                retryable = True
            raise RSSToolError(
                f"RSS request failed for {path}: {exc}",
                status_code=status,
                retryable=retryable,
            ) from exc

    async def news_list_sources(self, topic: str) -> dict:
        return await self._request("GET", "/sources", params={"topic": topic})

    async def news_refresh(self, topic: str) -> dict:
        return await self._request("POST", "/refresh", json={"topic": topic})

    async def news_latest(self, topic: str) -> dict:
        return await self._request("GET", "/latest", params={"topic": topic})
