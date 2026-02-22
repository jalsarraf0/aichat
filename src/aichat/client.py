from __future__ import annotations

import json
from typing import AsyncIterator

import httpx


class LLMClientError(RuntimeError):
    pass


class ModelNotFoundError(LLMClientError):
    pass


class LLMClient:
    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def _request(self, method: str, path: str, **kwargs: object) -> httpx.Response:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.request(method, f"{self.base_url}{path}", **kwargs)
                response.raise_for_status()
                return response
        except httpx.TimeoutException as exc:
            raise LLMClientError(f"Request timed out: {path}") from exc
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:300]
            raise LLMClientError(f"HTTP {exc.response.status_code} on {path}: {body}") from exc
        except httpx.HTTPError as exc:
            raise LLMClientError(f"Network error on {path}: {exc}") from exc

    async def health(self) -> bool:
        try:
            await self._request("GET", "/v1/models")
            return True
        except LLMClientError:
            return False

    async def list_models(self) -> list[str]:
        response = await self._request("GET", "/v1/models")
        payload = response.json()
        models = [m.get("id", "") for m in payload.get("data", []) if isinstance(m, dict)]
        return [m for m in models if m]

    async def ensure_model(self, model: str) -> None:
        models = await self.list_models()
        if models and model not in models:
            raise ModelNotFoundError(f"Model '{model}' not available. Available: {', '.join(models)}")

    async def chat_once(self, model: str, messages: list[dict[str, str]]) -> str:
        await self.ensure_model(model)
        payload = {"model": model, "messages": messages, "stream": False}
        response = await self._request("POST", "/v1/chat/completions", json=payload)
        try:
            return response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError("Malformed chat response") from exc

    async def chat_stream(self, model: str, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        await self.ensure_model(model)
        payload = {"model": model, "messages": messages, "stream": True}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", f"{self.base_url}/v1/chat/completions", json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            parsed = json.loads(data)
                            chunk = parsed.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        except json.JSONDecodeError:
                            chunk = data
                        if chunk:
                            yield chunk
        except httpx.TimeoutException as exc:
            raise LLMClientError("Streaming timed out") from exc
        except httpx.HTTPStatusError as exc:
            raise LLMClientError(f"Streaming failed with HTTP {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            raise LLMClientError(f"Streaming network error: {exc}") from exc
