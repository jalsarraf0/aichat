from __future__ import annotations

import base64
import importlib.util
import os
import sys
import uuid
from pathlib import Path

import httpx
import pytest

MCP_URL = "http://localhost:8096"
WORKSPACE = "/docker/human_browser/workspace"
_REPO = Path(__file__).parent.parent


def _load_mcp():
    mod_name = "mcp_app_face_recognition"
    spec = importlib.util.spec_from_file_location(mod_name, _REPO / "docker" / "mcp" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _up(url: str) -> bool:
    try:
        return httpx.get(url, timeout=3).status_code < 500
    except Exception:
        return False


try:
    _mcp = _load_mcp()
    _TOOLS = _mcp._TOOLS
    _LOAD_OK = True
except Exception:
    _LOAD_OK = False
    _TOOLS = []

skip_load = pytest.mark.skipif(not _LOAD_OK, reason="docker/mcp/app.py failed to load")
skip_mcp = pytest.mark.skipif(not _up(f"{MCP_URL}/health"), reason="aichat-mcp not running")
skip_ws = pytest.mark.skipif(not os.path.isdir(WORKSPACE), reason="browser workspace not mounted")


def _tool(name: str) -> dict:
    for tool in _TOOLS:
        if tool.get("name") == name:
            return tool
    return {}


def _mcp_call(name: str, arguments: dict, timeout: float = 40.0) -> dict:
    r = httpx.post(
        f"{MCP_URL}/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def _text_from(content: list[dict]) -> str:
    return "\n".join(block.get("text", "") for block in content if block.get("type") == "text")


def _has_image(content: list[dict]) -> bool:
    return any(block.get("type") == "image" for block in content)


@skip_load
class TestFaceRecognizeSchema:
    def test_face_recognize_advertised(self):
        names = {t["name"] for t in _TOOLS}
        assert "face_recognize" in names

    def test_face_recognize_schema_required_path(self):
        schema = _tool("face_recognize").get("inputSchema", {})
        assert "path" in schema.get("required", [])

    def test_face_recognize_schema_has_reference_path(self):
        props = _tool("face_recognize").get("inputSchema", {}).get("properties", {})
        assert "reference_path" in props


@pytest.fixture
def test_face_image_path() -> str:
    fname = f"face_tool_test_{uuid.uuid4().hex[:8]}.png"
    path = os.path.join(WORKSPACE, fname)
    try:
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (320, 320), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        # Simple synthetic face-like drawing. Detection may or may not trigger,
        # but the tool should process the image and return structured output.
        draw.ellipse([80, 60, 240, 220], outline=(30, 30, 30), width=4)
        draw.ellipse([125, 120, 145, 140], fill=(30, 30, 30))
        draw.ellipse([175, 120, 195, 140], fill=(30, 30, 30))
        draw.arc([120, 130, 200, 190], start=20, end=160, fill=(30, 30, 30), width=4)
        img.save(path, format="PNG")
    except Exception:
        # 1x1 transparent PNG fallback.
        raw = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+yL0YAAAAASUVORK5CYII="
        )
        with open(path, "wb") as fh:
            fh.write(raw)
    yield fname
    try:
        os.unlink(path)
    except Exception:
        pass


@pytest.mark.smoke
@skip_mcp
@skip_ws
class TestFaceRecognizeE2E:
    def test_face_recognize_returns_image_and_text(self, test_face_image_path):
        data = _mcp_call("face_recognize", {"path": test_face_image_path})
        content = data.get("result", {}).get("content", [])
        text = _text_from(content)
        assert _has_image(content), "face_recognize should return an inline image block"
        assert text.strip(), "face_recognize should return summary text"
        assert "face" in text.lower() or "opencv" in text.lower()

    def test_face_recognize_reference_mode(self, test_face_image_path):
        data = _mcp_call(
            "face_recognize",
            {
                "path": test_face_image_path,
                "reference_path": test_face_image_path,
                "match_threshold": 0.8,
            },
        )
        content = data.get("result", {}).get("content", [])
        text = _text_from(content).lower()
        assert _has_image(content), "reference mode should return an inline image block"
        assert text.strip(), "reference mode should return summary text"
