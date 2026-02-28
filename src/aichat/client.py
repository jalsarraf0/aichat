from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

import httpx


class LLMClientError(RuntimeError):
    pass


class ModelNotFoundError(LLMClientError):
    pass


# Separate timeouts for streaming vs non-streaming requests.
# Streaming uses read=None so that individual chunk gaps are not limited
# (the model may pause between tokens), while connect/write stay short.
# Non-streaming uses a fixed 60 s read.
_STREAM_TIMEOUT = httpx.Timeout(connect=15.0, read=None, write=15.0, pool=5.0)

# Transient errors that warrant a single automatic retry (connection-level only).
_RETRIABLE = (
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.ConnectError,
    httpx.LocalProtocolError,
)


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

    async def _raw_stream(
        self, payload: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """Low-level SSE streaming iterator (no retry logic).

        Parses ``data: …`` lines from LM Studio's OpenAI-compatible stream.
        SSE comment lines (``": …"``) and blank lines are silently skipped per spec.
        Raises :class:`LLMClientError` on transport failures.
        """
        try:
            async with httpx.AsyncClient(timeout=_STREAM_TIMEOUT) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        # Skip blank lines and SSE comment/keepalive lines.
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            return
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
            raise LLMClientError(
                f"Streaming failed with HTTP {exc.response.status_code}"
            ) from exc
        except asyncio.CancelledError:
            raise  # never swallow cancellation
        except asyncio.TimeoutError as exc:
            raise LLMClientError("Streaming cancelled by asyncio timeout") from exc
        except _RETRIABLE as exc:
            raise LLMClientError(f"Streaming network error: {exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMClientError(f"Streaming network error: {exc}") from exc

    async def chat_stream_events(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, object]] | None = None,
        tool_choice: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream chat-completion events with one automatic retry on transient errors.

        A retry is attempted only if the connection failed *before* any event was
        delivered, so the caller never sees partial duplicate content.

        Yields dicts with keys ``type`` (``"content"`` or ``"tool_calls"``) and
        ``value``.  Raises :class:`LLMClientError` on unrecoverable failure.
        """
        await self.ensure_model(model)
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": True}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        if max_tokens:
            payload["max_tokens"] = max_tokens

        events_yielded = 0
        last_exc: Exception | None = None

        for attempt in range(2):  # up to 1 retry
            if attempt > 0:
                # Only retry if we received nothing — avoids duplicate content.
                if events_yielded > 0:
                    break
                await asyncio.sleep(0.5)

            try:
                async for event in self._raw_stream(payload):
                    events_yielded += 1
                    yield event
                return  # clean completion

            except LLMClientError as exc:
                last_exc = exc
                # Only retry on connection-level errors (not 4xx/5xx/timeout).
                cause = exc.__cause__
                if not isinstance(cause, _RETRIABLE):
                    raise
                if events_yielded > 0:
                    raise  # partial stream — don't retry
                # Will retry on next loop iteration.

        if last_exc is not None:
            raise last_exc

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        async for event in self.chat_stream_events(model, messages, max_tokens=max_tokens):
            if event.get("type") == "content":
                yield event.get("value", "")
