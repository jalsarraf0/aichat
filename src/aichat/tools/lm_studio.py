"""LMStudioTool — thin async client for LM Studio's OpenAI-compatible API.

Wraps: TTS (/v1/audio/speech), embeddings (/v1/embeddings), chat completions
(/v1/chat/completions) for plain text, vision, summarization, and structured
JSON extraction.

All methods are fail-open: they catch every exception and return a sentinel
value (empty string, empty list, empty dict) so callers never crash.
"""
from __future__ import annotations

import base64
import json
import math
import os
from typing import Any

import httpx


class LMStudioError(RuntimeError):
    pass


class LMStudioTool:
    """Async client for LM Studio's OpenAI-compatible inference endpoints.

    Parameters
    ----------
    base_url:
        Root URL of the LM Studio server, e.g. ``http://192.168.50.2:1234``.
    model:
        Default model name to send with every request.  Leave empty to let the
        server use whichever model is currently loaded.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:1234",
        model: str = "",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model    = model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _payload(self, extra: dict) -> dict:
        p: dict[str, Any] = {**extra}
        if self.model:
            p.setdefault("model", self.model)
        return p

    async def _post(self, path: str, body: dict, timeout: float = 30.0) -> httpx.Response:
        async with httpx.AsyncClient(timeout=timeout) as hc:
            return await hc.post(f"{self.base_url}{path}", json=body)

    # ------------------------------------------------------------------
    # Text-to-speech
    # ------------------------------------------------------------------

    async def tts(
        self,
        text: str,
        voice: str = "alloy",
        speed: float = 1.0,
        fmt: str = "mp3",
    ) -> bytes:
        """Convert *text* to audio bytes via ``/v1/audio/speech``.

        Returns raw audio bytes (MP3/WAV/FLAC) or empty bytes on failure.
        """
        try:
            payload = self._payload({
                "input": text,
                "voice": voice,
                "speed": max(0.25, min(4.0, speed)),
                "response_format": fmt,
            })
            r = await self._post("/v1/audio/speech", payload, timeout=60.0)
            if r.status_code == 200:
                return r.content
        except Exception:
            pass
        return b""

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings via ``/v1/embeddings``.

        Returns a list of float vectors (one per input text).  Returns an
        empty list on failure.
        """
        if not texts:
            return []
        try:
            payload = self._payload({"input": texts})
            r = await self._post("/v1/embeddings", payload)
            r.raise_for_status()
            data = r.json().get("data", [])
            # Sort by index to preserve input order
            data.sort(key=lambda d: d.get("index", 0))
            return [d["embedding"] for d in data]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Token counting
    # ------------------------------------------------------------------

    async def tokenize(self, text: str) -> int:
        """Count tokens in *text* via ``/v1/tokenize``.

        Falls back to ``len(text) // 4`` on any failure (fail-open).
        """
        try:
            payload = self._payload({"text": text})
            r = await self._post("/v1/tokenize", payload)
            r.raise_for_status()
            data = r.json()
            tokens = data.get("token_count") or data.get("tokens")
            if isinstance(tokens, int):
                return tokens
            if isinstance(tokens, list):
                return len(tokens)
        except Exception:
            pass
        return max(0, len(text) // 4)

    # ------------------------------------------------------------------
    # Chat completions
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict],
        max_tokens: int = 500,
        temperature: float = 0.7,
        json_mode: bool = False,
    ) -> str:
        """Send a chat completion request and return the assistant reply text.

        Returns empty string on failure.
        """
        try:
            payload: dict = self._payload({
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            })
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            r = await self._post("/v1/chat/completions", payload)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Vision / image caption
    # ------------------------------------------------------------------

    async def caption(
        self,
        b64: str,
        detail_level: str = "detailed",
    ) -> str:
        """Ask the loaded vision model to describe an image (base64-encoded JPEG).

        *detail_level* is ``"brief"`` (one sentence) or ``"detailed"`` (full paragraph).
        Returns the caption string, or empty string on failure (fail-open).
        """
        try:
            if detail_level == "brief":
                prompt = "Describe this image in one concise sentence."
            else:
                prompt = (
                    "Describe this image in detail. Include: the main subject, "
                    "background, colors, style, mood, and any notable elements. "
                    "Be specific and vivid."
                )
            messages = [{"role": "user", "content": [
                {"type": "text",      "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]}]
            payload: dict = self._payload({
                "messages": messages,
                "max_tokens": 300 if detail_level == "detailed" else 80,
                "temperature": 0.3,
            })
            r = await self._post("/v1/chat/completions", payload, timeout=10.0)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Smart summarize
    # ------------------------------------------------------------------

    async def summarize(
        self,
        content: str,
        style: str = "brief",
        max_words: int | None = None,
    ) -> str:
        """Summarize *content* (pre-fetched text) using a chat completion.

        Parameters
        ----------
        content:
            Raw text to summarize (up to ~8000 chars; truncated internally).
        style:
            ``"brief"`` (2-3 sentences), ``"detailed"`` (full paragraph),
            or ``"bullets"`` (markdown bullet list).
        max_words:
            Optional word-count hint appended to the prompt.
        Returns the summary text, or empty string on failure.
        """
        try:
            text = content[:8000]
            if style == "bullets":
                instruction = "Summarize the following text as a concise markdown bullet list."
            elif style == "detailed":
                instruction = "Write a detailed, comprehensive summary of the following text."
            else:
                instruction = "Summarize the following text in 2–3 sentences."
            if max_words:
                instruction += f" Aim for approximately {max_words} words."
            messages = [
                {"role": "system", "content": "You are a helpful summarizer. Be concise and accurate."},
                {"role": "user",   "content": f"{instruction}\n\n---\n{text}"},
            ]
            return await self.chat(messages, max_tokens=600, temperature=0.3)
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Structured JSON extraction
    # ------------------------------------------------------------------

    async def extract(
        self,
        content: str,
        schema_json: str,
        instructions: str = "",
    ) -> dict:
        """Extract structured data from *content* matching *schema_json*.

        Uses ``response_format: json_object`` to guarantee JSON output.
        Returns the parsed dict, or empty dict on failure.
        """
        try:
            text = content[:6000]
            schema_hint = f"\n\nExtract data matching this schema:\n{schema_json}" if schema_json else ""
            extra = f"\n\nAdditional instructions: {instructions}" if instructions else ""
            messages = [
                {"role": "system", "content": (
                    "You are a structured data extractor. Extract information from the provided "
                    "text and return it as valid JSON matching the requested schema. "
                    "Return ONLY the JSON object, no extra text."
                )},
                {"role": "user", "content": f"Extract from this text:{schema_hint}{extra}\n\n---\n{text}"},
            ]
            raw = await self.chat(messages, max_tokens=800, temperature=0.0, json_mode=True)
            return json.loads(raw) if raw else {}
        except Exception:
            return {}


# ---------------------------------------------------------------------------
# Cosine similarity utility (pure Python, no numpy)
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return cosine similarity between two float vectors.  Range: -1.0 to 1.0."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)
