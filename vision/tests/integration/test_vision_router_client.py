"""Integration tests for VisionRouterClient.

Requires a live vision-router instance.  Set VISION_ROUTER_URL to the
base URL (e.g. ``http://192.168.50.2:8090``).  All tests are skipped
automatically when the variable is not set.
"""
from __future__ import annotations

import base64
import io
import os

import pytest
from PIL import Image

pytestmark = pytest.mark.skipif(
    not os.getenv("VISION_ROUTER_URL"),
    reason="Requires VISION_ROUTER_URL env var pointing to a live vision-router instance",
)


def _make_test_jpeg(width: int = 640, height: int = 480, color: tuple[int, int, int] = (128, 64, 32)) -> bytes:
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return buf.getvalue()


def _make_test_b64_jpeg(**kwargs: object) -> str:
    return base64.b64encode(_make_test_jpeg(**kwargs)).decode()  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_vision_router_health():
    """Health endpoint should return ok."""
    from app.clients.router import VisionRouterClient  # type: ignore[import]
    from app.config import VisionRouterConfig  # type: ignore[import]

    cfg = VisionRouterConfig(url=os.environ["VISION_ROUTER_URL"])
    client = VisionRouterClient(cfg)
    result = await client.health()
    assert isinstance(result, dict)
    assert result.get("status") in ("ok", "healthy")


@pytest.mark.asyncio
async def test_detect_objects_returns_list():
    """detect_objects should return a list of detected objects."""
    from app.clients.router import VisionRouterClient  # type: ignore[import]
    from app.config import VisionRouterConfig  # type: ignore[import]

    cfg = VisionRouterConfig(url=os.environ["VISION_ROUTER_URL"])
    client = VisionRouterClient(cfg)
    b64 = _make_test_b64_jpeg()
    result = await client.detect_objects(b64, min_confidence=0.1, max_results=20)
    assert isinstance(result, dict)
    assert "objects" in result or "detections" in result or "result" in result


@pytest.mark.asyncio
async def test_classify_returns_labels():
    """classify should return a list of labels with confidence scores."""
    from app.clients.router import VisionRouterClient  # type: ignore[import]
    from app.config import VisionRouterConfig  # type: ignore[import]

    cfg = VisionRouterConfig(url=os.environ["VISION_ROUTER_URL"])
    client = VisionRouterClient(cfg)
    b64 = _make_test_b64_jpeg()
    result = await client.classify(b64, top_k=5)
    assert isinstance(result, dict)
    assert "labels" in result or "classifications" in result or "result" in result


@pytest.mark.asyncio
async def test_embed_returns_vector():
    """embed should return a float vector."""
    from app.clients.router import VisionRouterClient  # type: ignore[import]
    from app.config import VisionRouterConfig  # type: ignore[import]

    cfg = VisionRouterConfig(url=os.environ["VISION_ROUTER_URL"])
    client = VisionRouterClient(cfg)
    b64 = _make_test_b64_jpeg()
    result = await client.embed(b64, model="clip_vit_b32", normalize=True)
    assert isinstance(result, dict)
    embeddings = result.get("embeddings") or result.get("embedding") or result.get("result")
    assert embeddings is not None
    if isinstance(embeddings, list):
        assert len(embeddings) > 0
        assert all(isinstance(x, (int, float)) for x in embeddings)


@pytest.mark.asyncio
async def test_detect_clothing_returns_items():
    """detect_clothing should return clothing item predictions."""
    from app.clients.router import VisionRouterClient  # type: ignore[import]
    from app.config import VisionRouterConfig  # type: ignore[import]

    cfg = VisionRouterConfig(url=os.environ["VISION_ROUTER_URL"])
    client = VisionRouterClient(cfg)
    b64 = _make_test_b64_jpeg()
    result = await client.detect_clothing(b64, min_confidence=0.05, top_k=5)
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_analyze_returns_comprehensive_result():
    """analyze should return results from multiple backends."""
    from app.clients.router import VisionRouterClient  # type: ignore[import]
    from app.config import VisionRouterConfig  # type: ignore[import]

    cfg = VisionRouterConfig(url=os.environ["VISION_ROUTER_URL"])
    client = VisionRouterClient(cfg)
    b64 = _make_test_b64_jpeg()
    result = await client.analyze(
        b64,
        include_objects=True,
        include_classification=True,
        include_clothing=False,
        include_embeddings=False,
    )
    assert isinstance(result, dict)
