from __future__ import annotations

import httpx

from .errors import ToolRequestError, is_retryable_status


class ToolkitError(ToolRequestError):
    pass


class ToolkitTool:
    def __init__(self, base_url: str = "http://localhost:8095") -> None:
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
            raise ToolkitError(
                f"Toolkit request failed for {path}: {exc}",
                status_code=status,
                retryable=retryable,
            ) from exc

    async def health(self) -> bool:
        try:
            result = await self._request("GET", "/health")
            return result.get("status") == "ok"
        except Exception:
            return False

    async def list_tools(self) -> list[dict]:
        result = await self._request("GET", "/tools")
        return result.get("tools", [])

    async def call_tool(self, tool_name: str, params: dict) -> dict:
        return await self._request("POST", f"/call/{tool_name}", json={"params": params})

    async def register_tool(
        self,
        tool_name: str,
        description: str,
        parameters: dict,
        code: str,
    ) -> dict:
        return await self._request(
            "POST",
            "/register",
            json={
                "tool_name": tool_name,
                "description": description,
                "parameters": parameters,
                "code": code,
            },
        )

    async def delete_tool(self, tool_name: str) -> dict:
        return await self._request("DELETE", f"/tool/{tool_name}")
