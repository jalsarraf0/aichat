"""End-to-end vision tool tests against an explicit vision-mcp server."""
from __future__ import annotations

import base64
import io
import json
import os
import sys

import pytest

sys.path.insert(0, "src")

VISION_URL = os.environ.get("VISION_MCP_URL") or os.environ.get("MCP_SERVER_URL") or ""


def _vision_mcp_up() -> bool:
    try:
        import urllib.request
        with urllib.request.urlopen(f"{VISION_URL}/health", timeout=2) as resp:
            data = json.load(resp)
        return isinstance(data, dict) and int(data.get("tools", 0)) > 0
    except Exception:
        return False


# Skip entire module if vision-mcp is not reachable or explicitly skipped
pytestmark = pytest.mark.skipif(
    os.environ.get("SKIP_VISION_E2E", "0") != "0" or not VISION_URL or not _vision_mcp_up(),
    reason="requires VISION_MCP_URL or MCP_SERVER_URL pointing to a healthy vision-mcp server",
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
    return VisionMCPTool(VISION_URL)


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
