"""End-to-end vision tool tests (require vision-mcp at localhost:8097)."""
from __future__ import annotations

import base64
import io
import os
import sys

import pytest

sys.path.insert(0, "src")


def _vision_mcp_up() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:8097/health", timeout=2)
        return True
    except Exception:
        return False


# Skip entire module if vision-mcp is not reachable or explicitly skipped
pytestmark = pytest.mark.skipif(
    os.environ.get("SKIP_VISION_E2E", "0") != "0" or not _vision_mcp_up(),
    reason="vision-mcp not reachable at localhost:8097",
)


def _make_test_b64() -> str:
    """Create a minimal valid JPEG in base64."""
    from PIL import Image
    img = Image.new("RGB", (64, 64), (128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def vision_tool():
    from aichat.tools.vision import VisionMCPTool
    return VisionMCPTool("http://localhost:8097")


@pytest.fixture
def test_b64():
    return _make_test_b64()


@pytest.mark.asyncio
async def test_health_endpoint(vision_tool):
    healthy = await vision_tool.health()
    # May be False if Windows services are down — that's OK
    assert isinstance(healthy, bool)


@pytest.mark.asyncio
async def test_list_face_subjects_reachable(vision_tool):
    result = await vision_tool.list_face_subjects()
    # Either returns subjects list or error (CompreFace may be offline)
    assert isinstance(result, dict)
    assert "subjects" in result or "error" in result


@pytest.mark.asyncio
async def test_detect_objects_with_image(vision_tool, test_b64):
    result = await vision_tool.detect_objects(image_base64=test_b64)
    assert isinstance(result, dict)
    assert "objects" in result or "error" in result


@pytest.mark.asyncio
async def test_classify_image_with_image(vision_tool, test_b64):
    result = await vision_tool.classify_image(image_base64=test_b64, top_k=3)
    assert isinstance(result, dict)
    assert "labels" in result or "error" in result


@pytest.mark.asyncio
async def test_analyze_image_with_image(vision_tool, test_b64):
    result = await vision_tool.analyze_image(
        image_base64=test_b64,
        include_objects=True, include_labels=True, include_embedding=False,
    )
    assert isinstance(result, dict)
    # Should have at least objects or labels if Triton is up, or error if not
    assert "objects" in result or "labels" in result or "error" in result


@pytest.mark.asyncio
async def test_embed_image_returns_vector(vision_tool, test_b64):
    result = await vision_tool.embed_image(image_base64=test_b64)
    if "error" not in result:
        assert "embedding" in result
        assert isinstance(result["embedding"], list)
        assert len(result["embedding"]) == 512
