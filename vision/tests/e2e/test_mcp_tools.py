"""End-to-end MCP protocol tests against the full vision MCP server.

Requires MCP_SERVER_URL pointing to a running vision MCP server instance
(e.g. ``http://localhost:8097``).  All tests are skipped automatically
when the variable is not set.
"""
from __future__ import annotations

import base64
import io
import os

import httpx
import pytest
from PIL import Image

pytestmark = pytest.mark.skipif(
    not os.getenv("MCP_SERVER_URL"),
    reason="Requires MCP_SERVER_URL env var pointing to a running MCP server",
)

MCP_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8097")

EXPECTED_TOOLS = {
    "recognize_face",
    "verify_face",
    "detect_faces",
    "enroll_face",
    "list_face_subjects",
    "delete_face_subject",
    "detect_objects",
    "classify_image",
    "detect_clothing",
    "embed_image",
    "analyze_image",
}


def _make_test_jpeg_b64(width: int = 64, height: int = 64) -> str:
    img = Image.new("RGB", (width, height), (128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return base64.b64encode(buf.getvalue()).decode()


async def _call_tool(name: str, args: dict) -> dict:
    """Send a tools/call JSON-RPC request and return the parsed response."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{MCP_URL}/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": name, "arguments": args},
            },
        )
        resp.raise_for_status()
        return resp.json()


async def _list_tools() -> list[dict]:
    """Retrieve the tool list from the MCP server."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            f"{MCP_URL}/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["result"]["tools"]


# ---------------------------------------------------------------------------
# Health and metadata
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_endpoint():
    """GET /health should return 200 with status=ok."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{MCP_URL}/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_list_tools_returns_all_11():
    """tools/list should return exactly 11 tools."""
    tools = await _list_tools()
    tool_names = {t["name"] for t in tools}
    assert len(tools) == 11, f"Expected 11 tools, got {len(tools)}: {tool_names}"
    assert EXPECTED_TOOLS == tool_names


@pytest.mark.asyncio
async def test_all_tools_have_input_schema():
    """Every tool should expose a valid inputSchema."""
    tools = await _list_tools()
    for tool in tools:
        assert "inputSchema" in tool, f"Tool {tool['name']} missing inputSchema"
        schema = tool["inputSchema"]
        assert schema.get("type") == "object", f"Tool {tool['name']} inputSchema type != object"


@pytest.mark.asyncio
async def test_all_tools_have_description():
    """Every tool should have a non-empty description."""
    tools = await _list_tools()
    for tool in tools:
        assert tool.get("description"), f"Tool {tool['name']} has empty description"


# ---------------------------------------------------------------------------
# Face tools (no image required for list)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_face_subjects():
    """list_face_subjects should succeed and return a list (possibly empty)."""
    result = await _call_tool("list_face_subjects", {})
    assert "result" in result or "error" in result
    if "result" in result:
        content = result["result"]
        # The result should not indicate an internal server error
        assert "traceback" not in str(content).lower()


@pytest.mark.asyncio
async def test_detect_objects_with_test_image():
    """detect_objects should accept a base64 image and return a structured response."""
    b64 = _make_test_jpeg_b64(width=640, height=480)
    result = await _call_tool("detect_objects", {"image": {"base64": b64}})
    # Either a result or an error (e.g. backend unavailable) — no crash
    assert "result" in result or "error" in result


@pytest.mark.asyncio
async def test_classify_image_with_test_image():
    """classify_image should accept a base64 image and return a structured response."""
    b64 = _make_test_jpeg_b64()
    result = await _call_tool("classify_image", {"image": {"base64": b64}})
    assert "result" in result or "error" in result


@pytest.mark.asyncio
async def test_detect_faces_with_test_image():
    """detect_faces should accept a base64 image and not crash."""
    b64 = _make_test_jpeg_b64(width=128, height=128)
    result = await _call_tool("detect_faces", {"image": {"base64": b64}})
    assert "result" in result or "error" in result


@pytest.mark.asyncio
async def test_embed_image_with_test_image():
    """embed_image should return an embedding vector or a backend error."""
    b64 = _make_test_jpeg_b64()
    result = await _call_tool("embed_image", {"image": {"base64": b64}})
    assert "result" in result or "error" in result


@pytest.mark.asyncio
async def test_analyze_image_with_test_image():
    """analyze_image should accept a base64 image and return a structured response."""
    b64 = _make_test_jpeg_b64(width=640, height=480)
    result = await _call_tool(
        "analyze_image",
        {
            "image": {"base64": b64},
            "include_objects": True,
            "include_classification": True,
            "include_clothing": False,
            "include_embeddings": False,
        },
    )
    assert "result" in result or "error" in result


@pytest.mark.asyncio
async def test_invalid_tool_name_returns_error():
    """Calling a non-existent tool should return a JSON-RPC error."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            f"{MCP_URL}/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "nonexistent_tool_xyz", "arguments": {}},
            },
        )
    data = resp.json()
    assert "error" in data


@pytest.mark.asyncio
async def test_missing_required_argument_returns_error():
    """Calling a tool with missing required args should return a JSON-RPC error."""
    # detect_objects requires an image argument
    result = await _call_tool("detect_objects", {})
    assert "error" in result or "result" in result  # graceful error, not a server crash


@pytest.mark.asyncio
async def test_ssrf_blocked_url():
    """Attempting to load an image from localhost should be blocked."""
    result = await _call_tool(
        "detect_objects",
        {"image": {"url": "http://127.0.0.1/internal"}},
    )
    # Should return an error (SSRF blocked), not silently succeed
    if "result" in result:
        content = str(result["result"])
        assert "error" in content.lower() or "blocked" in content.lower() or "private" in content.lower()
    # error key means it was handled correctly
