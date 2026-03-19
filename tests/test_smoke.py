"""
Smoke tests for all aichat Docker services.

These tests require the Docker stack to be running and are skipped automatically
when the services are not reachable.  Run them with:

    pytest tests/test_smoke.py -v -m smoke

Or alongside the full suite:

    pytest -v -m "smoke or not smoke"

Consolidated architecture (as of refactor):
  aichat-data    :8091  — database, memory, graph, planner, research, jobs
  aichat-vision  :8099  — video + OCR
  aichat-docs    :8101  — document ingestor + PDF operations
  aichat-sandbox :8095  — isolated code execution
  aichat-mcp     :8096  — MCP gateway
  aichat-whatsapp:8097  — WhatsApp bot (optional)
"""
from __future__ import annotations

import os

import pytest
import httpx


# ---------------------------------------------------------------------------
# Service definitions (consolidated architecture)
# URLs can be overridden via env vars so CI can run tests inside the Docker
# compose network (using service hostnames) without host port bindings.
# ---------------------------------------------------------------------------

_DATA_URL    = os.environ.get("DATA_URL",    "http://localhost:8091")
_VISION_URL  = os.environ.get("VISION_URL",  "http://localhost:8099")
_DOCS_URL    = os.environ.get("DOCS_URL",    "http://localhost:8101")
_SANDBOX_URL = os.environ.get("SANDBOX_URL", "http://localhost:8095")
_MCP_URL     = os.environ.get("MCP_URL",     "http://localhost:8096")
_WHATSAPP_URL = os.environ.get("WHATSAPP_URL", "http://localhost:8097")
_JUPYTER_URL  = os.environ.get("JUPYTER_URL",  "http://localhost:8098")

_SERVICES = {
    "aichat-data":     f"{_DATA_URL}/health",
    "aichat-vision":   f"{_VISION_URL}/health",
    "aichat-docs":     f"{_DOCS_URL}/health",
    "aichat-sandbox":  f"{_SANDBOX_URL}/health",
    "aichat-mcp":      f"{_MCP_URL}/health",
    "aichat-whatsapp": f"{_WHATSAPP_URL}/health",
    "aichat-jupyter":  f"{_JUPYTER_URL}/health",
}

# Timeout for each health-check request (seconds).
_TIMEOUT = 5.0


def _is_reachable(url: str) -> bool:
    """Return True if the service at *url* responds with HTTP 2xx."""
    try:
        r = httpx.get(url, timeout=_TIMEOUT, follow_redirects=True)
        return r.status_code < 300
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Parametrised smoke tests
# ---------------------------------------------------------------------------

@pytest.mark.smoke
@pytest.mark.parametrize("service,url", list(_SERVICES.items()))
def test_service_health(service: str, url: str) -> None:
    """Each service must return HTTP 200 from its /health endpoint."""
    if not _is_reachable(url):
        pytest.skip(f"{service} not reachable at {url}")
    r = httpx.get(url, timeout=_TIMEOUT, follow_redirects=True)
    assert r.status_code == 200, f"{service} health returned {r.status_code}: {r.text}"
    body = r.json()
    # All services should return a JSON object
    assert isinstance(body, dict), f"{service} health response is not a JSON object: {body}"


@pytest.mark.smoke
def test_data_health_includes_services() -> None:
    """aichat-data /health must report status ok and all sub-service states."""
    url = _SERVICES["aichat-data"]
    if not _is_reachable(url):
        pytest.skip("aichat-data not reachable")
    r = httpx.get(url, timeout=_TIMEOUT)
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "ok", f"Unexpected status in data health: {body}"
    assert "services" in body, f"Missing 'services' in data health: {body}"
    for svc in ("postgres", "memory", "graph", "planner", "jobs"):
        assert svc in body["services"], f"Sub-service '{svc}' missing from data health: {body}"


@pytest.mark.smoke
def test_mcp_health_includes_tools() -> None:
    """aichat-mcp /health must report the number of registered tools."""
    url = _SERVICES["aichat-mcp"]
    if not _is_reachable(url):
        pytest.skip("aichat-mcp not reachable")
    r = httpx.get(url, timeout=_TIMEOUT)
    assert r.status_code == 200
    body = r.json()
    assert "tools" in body, f"Missing 'tools' in MCP health: {body}"
    assert body["tools"] == 16, f"Expected 16 mega-tools, got {body['tools']}"


@pytest.mark.smoke
def test_mcp_tools_list_via_jsonrpc() -> None:
    """MCP server must respond to tools/list JSON-RPC request with ≥20 tools."""
    base_url = _MCP_URL
    if not _is_reachable(f"{base_url}/health"):
        pytest.skip("aichat-mcp not reachable")
    rpc_req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {},
    }
    r = httpx.post(f"{base_url}/mcp", json=rpc_req, timeout=_TIMEOUT)
    assert r.status_code == 200
    body = r.json()
    tools = body.get("result", {}).get("tools", [])
    assert len(tools) == 16, f"Expected 16 mega-tools from MCP, got {len(tools)}"
    tool_names = {t["name"] for t in tools}
    for expected in ("web", "browser", "image", "document", "media", "data",
                     "memory", "knowledge", "vector", "code", "custom_tools",
                     "planner", "jobs", "research", "think", "system"):
        assert expected in tool_names, f"Mega-tool '{expected}' missing from MCP tools list"


@pytest.mark.smoke
def test_errors_endpoint_accessible() -> None:
    """aichat-data /errors/recent must return a valid response."""
    url = f"{_DATA_URL}/errors/recent"
    if not _is_reachable(_SERVICES["aichat-data"]):
        pytest.skip("aichat-data not reachable")
    r = httpx.get(url, timeout=_TIMEOUT, params={"limit": 10})
    assert r.status_code == 200
    body = r.json()
    assert "errors" in body, f"Missing 'errors' key in /errors/recent response: {body}"
    assert isinstance(body["errors"], list), "'errors' must be a list"


@pytest.mark.smoke
def test_error_log_roundtrip() -> None:
    """POST /errors/log then GET /errors/recent must return the logged entry."""
    base = _DATA_URL
    if not _is_reachable(f"{base}/health"):
        pytest.skip("aichat-data not reachable")

    marker = "smoke-test-marker-aichat"
    post_r = httpx.post(
        f"{base}/errors/log",
        json={"service": "smoke-test", "level": "INFO", "message": marker},
        timeout=_TIMEOUT,
    )
    assert post_r.status_code == 200, f"POST /errors/log failed: {post_r.text}"

    get_r = httpx.get(
        f"{base}/errors/recent",
        params={"service": "smoke-test", "limit": 5},
        timeout=_TIMEOUT,
    )
    assert get_r.status_code == 200
    errors = get_r.json().get("errors", [])
    messages = [e["message"] for e in errors]
    assert marker in messages, f"Logged entry not found in recent errors: {messages}"


@pytest.mark.smoke
def test_memory_store_recall_roundtrip() -> None:
    """POST /memory/store then GET /memory/recall must return the stored value."""
    base = _DATA_URL
    if not _is_reachable(f"{base}/health"):
        pytest.skip("aichat-data not reachable")

    key = "_smoke_test_key"
    value = "smoke-test-value-aichat"

    post_r = httpx.post(f"{base}/memory/store", json={"key": key, "value": value}, timeout=_TIMEOUT)
    assert post_r.status_code == 200

    get_r = httpx.get(f"{base}/memory/recall", params={"key": key}, timeout=_TIMEOUT)
    assert get_r.status_code == 200
    body = get_r.json()
    assert body.get("found"), f"Stored key not found: {body}"
    assert body["entries"][0]["value"] == value


@pytest.mark.smoke
def test_sandbox_tools_list() -> None:
    """aichat-sandbox GET /tools must return a valid tools list."""
    base = _SANDBOX_URL
    if not _is_reachable(f"{base}/health"):
        pytest.skip("aichat-sandbox not reachable")

    r = httpx.get(f"{base}/tools", timeout=_TIMEOUT)
    assert r.status_code == 200
    body = r.json()
    assert "tools" in body, f"Missing 'tools' key: {body}"
    assert isinstance(body["tools"], list)


@pytest.mark.smoke
def test_research_search_feeds() -> None:
    """aichat-data /research/search-feeds must return feed URLs for a topic."""
    base = _DATA_URL
    if not _is_reachable(f"{base}/health"):
        pytest.skip("aichat-data not reachable")

    r = httpx.get(f"{base}/research/search-feeds", params={"topic": "python"}, timeout=_TIMEOUT)
    assert r.status_code == 200
    body = r.json()
    assert "feeds" in body, f"Missing 'feeds' key: {body}"
    assert len(body["feeds"]) > 0, "No feeds returned"


def _make_tiny_png() -> bytes:
    """Generate a valid 4×4 white RGB PNG using stdlib only (no Pillow)."""
    import struct, zlib as _zl

    def _chunk(tag: bytes, data: bytes) -> bytes:
        payload = tag + data
        return struct.pack(">I", len(data)) + payload + struct.pack(">I", _zl.crc32(payload) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    # IHDR: width=4, height=4, bit_depth=8, color_type=2 (RGB)
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 0))
    # Raw scanlines: filter_byte(0) + 4 white pixels per row
    raw = b"".join(b"\x00" + b"\xff\xff\xff" * 4 for _ in range(4))
    idat = _chunk(b"IDAT", _zl.compress(raw))
    iend = _chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


@pytest.mark.smoke
def test_ocr_vision_direct() -> None:
    """aichat-vision POST /ocr must accept a valid PNG and return OCR fields.

    Regression test: ensures the /ocr route exists (not /ocr/ocr).
    """
    base_url = _VISION_URL
    if not _is_reachable(f"{base_url}/health"):
        pytest.skip("aichat-vision not reachable")

    import base64
    b64 = base64.standard_b64encode(_make_tiny_png()).decode()
    r = httpx.post(f"{base_url}/ocr", json={"b64": b64, "lang": "eng"}, timeout=30)
    assert r.status_code == 200, f"POST /ocr returned {r.status_code}: {r.text}"
    body = r.json()
    assert "text" in body, f"Missing 'text' key in OCR response: {body}"
    assert "word_count" in body, f"Missing 'word_count' key in OCR response: {body}"


@pytest.mark.smoke
def test_ocr_image_via_mcp_jsonrpc() -> None:
    """ocr_image MCP tool must call /ocr (not /ocr/ocr) and return OCR text.

    Regression test for the double-prefix bug where OCR_URL already contained
    /ocr but the handler appended /ocr again → /ocr/ocr (404 Not Found).
    Uses _resolve_image_path's /docker/human_browser/workspace/ alias so the
    path works both from the test host and inside the MCP container.
    """
    base_url = _MCP_URL
    if not _is_reachable(f"{base_url}/health"):
        pytest.skip("aichat-mcp not reachable")
    if not _is_reachable(f"{_VISION_URL}/ocr/health"):
        pytest.skip("aichat-vision OCR not reachable")

    import glob
    # _resolve_image_path inside MCP remaps /docker/human_browser/workspace/* → BROWSER_WORKSPACE
    host_ws = "/docker/human_browser/workspace"
    images = sorted(glob.glob(f"{host_ws}/*.jpg") + glob.glob(f"{host_ws}/*.png"))
    if not images:
        pytest.skip("No images in /docker/human_browser/workspace to OCR")

    rpc = {
        "jsonrpc": "2.0",
        "id": 99,
        "method": "tools/call",
        "params": {
            "name": "ocr_image",
            "arguments": {"path": images[0], "lang": "eng"},
        },
    }
    r = httpx.post(f"{base_url}/mcp", json=rpc, timeout=60)
    assert r.status_code == 200, f"MCP returned {r.status_code}"
    body = r.json()
    content = body.get("result", {}).get("content", [{}])
    text = content[0].get("text", "") if content else ""
    assert "OCR result" in text, f"Expected 'OCR result' in response, got: {text[:300]}"
    assert "404" not in text, f"OCR still returning 404 — routing bug: {text[:300]}"


@pytest.mark.smoke
def test_jupyter_health() -> None:
    """aichat-jupyter /health must return status ok."""
    url = _SERVICES["aichat-jupyter"]
    if not _is_reachable(url):
        pytest.skip("aichat-jupyter not reachable")
    r = httpx.get(url, timeout=_TIMEOUT)
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "ok"
    assert body.get("service") == "aichat-jupyter"


@pytest.mark.smoke
def test_jupyter_exec_via_mcp() -> None:
    """jupyter_exec MCP tool must execute Python code and return stdout."""
    base_url = _MCP_URL
    if not _is_reachable(f"{base_url}/health"):
        pytest.skip("aichat-mcp not reachable")

    rpc = {
        "jsonrpc": "2.0",
        "id": 200,
        "method": "tools/call",
        "params": {
            "name": "jupyter_exec",
            "arguments": {"code": "print('smoke-jupyter-ok')"},
        },
    }
    r = httpx.post(f"{base_url}/mcp", json=rpc, timeout=30)
    assert r.status_code == 200
    body = r.json()
    content = body.get("result", {}).get("content", [{}])
    text = content[0].get("text", "") if content else ""
    assert "smoke-jupyter-ok" in text, f"Expected stdout output, got: {text[:300]}"
