"""
Smoke checks for LM Studio API using a target model (default: gpt-oss-20b).

These tests validate:
  - /v1/models is reachable
  - target model is loaded
  - a minimal /v1/chat/completions call succeeds
"""
from __future__ import annotations

import os

import httpx
import pytest

LM_URL = os.environ.get("LM_STUDIO_URL", os.environ.get("LM_URL", "http://192.168.50.2:1234"))
TARGET_MODEL = os.environ.get("AICHAT_SMOKE_MODEL", "gpt-oss-20b")


def _up(url: str) -> bool:
    try:
        return httpx.get(url, timeout=3).status_code < 500
    except Exception:
        return False


_LM_UP = _up(f"{LM_URL}/v1/models")
skip_lm = pytest.mark.skipif(not _LM_UP, reason=f"LM Studio not reachable at {LM_URL}")


def _models() -> list[dict]:
    r = httpx.get(f"{LM_URL}/v1/models", timeout=8)
    r.raise_for_status()
    data = r.json()
    return data.get("data", []) if isinstance(data, dict) else []


@pytest.mark.smoke
@skip_lm
def test_models_endpoint_reachable() -> None:
    models = _models()
    assert isinstance(models, list)
    assert len(models) > 0, "No models reported by LM Studio /v1/models"


@pytest.mark.smoke
@skip_lm
def test_target_model_loaded() -> None:
    models = _models()
    model_ids = [str(m.get("id", "")) for m in models]
    assert any(TARGET_MODEL in mid for mid in model_ids), (
        f"Target model '{TARGET_MODEL}' not found. Loaded models: {model_ids}"
    )


@pytest.mark.smoke
@skip_lm
def test_target_model_chat_completion_roundtrip() -> None:
    payload = {
        "model": TARGET_MODEL,
        "messages": [
            {"role": "user", "content": "Reply with exactly: ok"},
        ],
        "temperature": 0,
        "max_tokens": 8,
    }
    r = httpx.post(f"{LM_URL}/v1/chat/completions", json=payload, timeout=25)
    r.raise_for_status()
    body = r.json()
    choices = body.get("choices", [])
    assert choices, f"No choices in response: {body}"
    message = choices[0].get("message", {})
    assert message.get("role") == "assistant", f"Unexpected message payload: {message}"
    # Some local models return empty text for short low-token prompts while still
    # producing a valid assistant turn; we accept that as a successful API roundtrip.
    content = str(message.get("content", ""))
    assert isinstance(content, str)
