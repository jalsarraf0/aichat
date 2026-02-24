from __future__ import annotations

import httpx

from .errors import ToolRequestError, is_retryable_status


class FetchToolError(ToolRequestError):
    pass


class FetchTool:
    def __init__(self, base_url: str = "http://localhost:8093") -> None:
        self.base_url = base_url.rstrip("/")

    async def _request(self, method: str, path: str, **kwargs: object) -> dict:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
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
            raise FetchToolError(
                f"Fetch request failed for {path}: {exc}",
                status_code=status,
                retryable=retryable,
            ) from exc

    async def fetch_url(self, url: str, max_chars: int = 4000) -> dict:
        return await self._request("POST", "/fetch", json={"url": url, "max_chars": max_chars})
