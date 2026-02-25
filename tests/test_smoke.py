"""
Smoke tests for all aichat Docker services.

These tests require the Docker stack to be running and are skipped automatically
when the services are not reachable.  Run them with:

    pytest tests/test_smoke.py -v -m smoke

Or alongside the full suite:

    pytest -v -m "smoke or not smoke"
"""
from __future__ import annotations

import pytest
import httpx


# ---------------------------------------------------------------------------
# Service definitions
# ---------------------------------------------------------------------------

_SERVICES = {
    "aichat-database":    "http://localhost:8091/health",
    "aichat-researchbox": "http://localhost:8092/health",
    "aichat-memory":      "http://localhost:8094/health",
    "aichat-toolkit":     "http://localhost:8095/health",
    "aichat-mcp":         "http://localhost:8096/health",
    "aichat-whatsapp":    "http://localhost:8097/health",
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
def test_database_health_includes_counts() -> None:
    """aichat-database /health must include article and page-cache counts."""
    url = _SERVICES["aichat-database"]
    if not _is_reachable(url):
        pytest.skip("aichat-database not reachable")
    r = httpx.get(url, timeout=_TIMEOUT)
    assert r.status_code == 200
    body = r.json()
    assert "articles" in body, f"Missing 'articles' in database health: {body}"
    assert "cached_pages" in body, f"Missing 'cached_pages' in database health: {body}"


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
    assert body["tools"] >= 20, f"Expected ≥20 tools, got {body['tools']}"


@pytest.mark.smoke
def test_mcp_tools_list_via_jsonrpc() -> None:
    """MCP server must respond to tools/list JSON-RPC request with ≥20 tools."""
    base_url = "http://localhost:8096"
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
    assert len(tools) >= 20, f"Expected ≥20 tools from MCP, got {len(tools)}"
    tool_names = {t["name"] for t in tools}
    for expected in ("screenshot", "web_search", "memory_store", "get_errors"):
        assert expected in tool_names, f"Tool '{expected}' missing from MCP tools list"


@pytest.mark.smoke
def test_errors_endpoint_accessible() -> None:
    """aichat-database /errors/recent must return a valid response."""
    url = "http://localhost:8091/errors/recent"
    if not _is_reachable(_SERVICES["aichat-database"]):
        pytest.skip("aichat-database not reachable")
    r = httpx.get(url, timeout=_TIMEOUT, params={"limit": 10})
    assert r.status_code == 200
    body = r.json()
    assert "errors" in body, f"Missing 'errors' key in /errors/recent response: {body}"
    assert isinstance(body["errors"], list), "'errors' must be a list"


@pytest.mark.smoke
def test_error_log_roundtrip() -> None:
    """POST /errors/log then GET /errors/recent must return the logged entry."""
    base = "http://localhost:8091"
    if not _is_reachable(f"{base}/health"):
        pytest.skip("aichat-database not reachable")

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
    """POST /store then GET /recall must return the stored value."""
    base = "http://localhost:8094"
    if not _is_reachable(f"{base}/health"):
        pytest.skip("aichat-memory not reachable")

    key = "_smoke_test_key"
    value = "smoke-test-value-aichat"

    post_r = httpx.post(f"{base}/store", json={"key": key, "value": value}, timeout=_TIMEOUT)
    assert post_r.status_code == 200

    get_r = httpx.get(f"{base}/recall", params={"key": key}, timeout=_TIMEOUT)
    assert get_r.status_code == 200
    body = get_r.json()
    assert body.get("found"), f"Stored key not found: {body}"
    assert body["entries"][0]["value"] == value


@pytest.mark.smoke
def test_toolkit_tools_list() -> None:
    """aichat-toolkit GET /tools must return a valid tools list."""
    base = "http://localhost:8095"
    if not _is_reachable(f"{base}/health"):
        pytest.skip("aichat-toolkit not reachable")

    r = httpx.get(f"{base}/tools", timeout=_TIMEOUT)
    assert r.status_code == 200
    body = r.json()
    assert "tools" in body, f"Missing 'tools' key: {body}"
    assert isinstance(body["tools"], list)


@pytest.mark.smoke
def test_researchbox_search_feeds() -> None:
    """aichat-researchbox GET /search-feeds must return feed URLs for a topic."""
    base = "http://localhost:8092"
    if not _is_reachable(f"{base}/health"):
        pytest.skip("aichat-researchbox not reachable")

    r = httpx.get(f"{base}/search-feeds", params={"topic": "python"}, timeout=_TIMEOUT)
    assert r.status_code == 200
    body = r.json()
    assert "feeds" in body, f"Missing 'feeds' key: {body}"
    assert len(body["feeds"]) > 0, "No feeds returned"
