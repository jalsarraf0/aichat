from __future__ import annotations

import httpx

from .errors import ToolRequestError, is_retryable_status


class MemoryToolError(ToolRequestError):
    pass


class MemoryTool:
    def __init__(self, base_url: str = "http://localhost:8094") -> None:
        self.base_url = base_url.rstrip("/")

    async def _request(self, method: str, path: str, **kwargs: object) -> dict:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
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
            raise MemoryToolError(
                f"Memory request failed for {path}: {exc}",
                status_code=status,
                retryable=retryable,
            ) from exc

    async def store(self, key: str, value: str) -> dict:
        return await self._request("POST", "/store", json={"key": key, "value": value})

    async def recall(self, key: str = "") -> dict:
        return await self._request("GET", "/recall", params={"key": key})

    async def delete(self, key: str) -> dict:
        return await self._request("DELETE", "/delete", params={"key": key})

    async def clear(self) -> dict:
        return await self._request("DELETE", "/clear")
