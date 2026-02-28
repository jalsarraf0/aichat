from __future__ import annotations

import json
from typing import Any, AsyncIterator

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

    async def model_info(self) -> dict[str, int]:
        """Return ``{model_id: context_length}`` from ``/v1/models``.

        LM Studio includes ``context_length`` in each model entry.  Returns an
        empty dict on any failure (fail-open) so callers can always safely check
        ``info.get(model_id, 0)``.
        """
        try:
            response = await self._request("GET", "/v1/models")
            data = response.json().get("data", [])
            return {
                m["id"]: int(m["context_length"])
                for m in data
                if isinstance(m, dict) and m.get("id") and m.get("context_length")
            }
        except Exception:
            return {}

    async def ensure_model(self, model: str) -> None:
        models = await self.list_models()
        if models and model not in models:
            raise ModelNotFoundError(f"Model '{model}' not available. Available: {', '.join(models)}")

    async def chat_once(self, model: str, messages: list[dict[str, Any]], max_tokens: int | None = None) -> str:
        response = await self.chat_once_with_tools(model, messages, max_tokens=max_tokens)
        return response.get("content", "")

    async def chat_once_with_tools(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, object]] | None = None,
        tool_choice: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        await self.ensure_model(model)
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        if max_tokens:
            payload["max_tokens"] = max_tokens
        response = await self._request("POST", "/v1/chat/completions", json=payload)
        try:
            message = response.json()["choices"][0]["message"]
            return {
                "content": message.get("content") or "",
                "tool_calls": message.get("tool_calls") or [],
                "raw": message,
            }
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise LLMClientError("Malformed chat response") from exc

    async def chat_stream_events(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, object]] | None = None,
        tool_choice: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        await self.ensure_model(model)
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": True}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        if max_tokens:
            payload["max_tokens"] = max_tokens
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
                        except json.JSONDecodeError:
                            continue
                        delta = parsed.get("choices", [{}])[0].get("delta", {})
                        if "content" in delta and delta["content"]:
                            yield {"type": "content", "value": delta["content"]}
                        if "tool_calls" in delta and delta["tool_calls"]:
                            yield {"type": "tool_calls", "value": delta["tool_calls"]}
        except httpx.TimeoutException as exc:
            raise LLMClientError("Streaming timed out") from exc
        except httpx.HTTPStatusError as exc:
            raise LLMClientError(f"Streaming failed with HTTP {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            raise LLMClientError(f"Streaming network error: {exc}") from exc

    async def chat_stream(self, model: str, messages: list[dict[str, Any]], max_tokens: int | None = None) -> AsyncIterator[str]:
        async for event in self.chat_stream_events(model, messages, max_tokens=max_tokens):
            if event.get("type") == "content":
                yield event.get("value", "")
