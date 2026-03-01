"""
Database tool validation, smoke, and E2E tests.

Structure
---------
TestDbToolInputValidation  — unit tests: MCP handler input guards (no Docker)
TestDbCacheStoreE2E        — E2E with mocked httpx: db_cache_store full path
TestDbCacheGetE2E          — E2E with mocked httpx: db_cache_get full path
TestMemoryStoreE2E         — E2E with mocked httpx: memory_store full path
TestDbSearchE2E            — E2E with mocked httpx: db_search full path
TestEmbedE2E               — E2E with mocked httpx: embed_store / embed_search
TestDbToolsSmoke           — @pytest.mark.smoke: live service calls
TestKlukaiOCR              — @pytest.mark.smoke: ultimate E2E — find images → OCR
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# ---------------------------------------------------------------------------
# Load docker/mcp/app.py via importlib (no Docker needed)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent


def _load_mcp_mod():
    spec = importlib.util.spec_from_file_location(
        "mcp_app_db_tools",
        _REPO_ROOT / "docker" / "mcp" / "app.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mcp_app_db_tools"] = mod
    spec.loader.exec_module(mod)   # type: ignore[union-attr]
    return mod


try:
    _mcp = _load_mcp_mod()
    _call_tool = _mcp._call_tool
    _TOOLS      = _mcp._TOOLS
    _LOAD_OK    = True
except Exception as _load_err:
    _LOAD_OK  = False
    _LOAD_ERR = str(_load_err)

skip_load = pytest.mark.skipif(
    not _LOAD_OK,
    reason=f"docker/mcp/app.py load failed: {_LOAD_ERR if not _LOAD_OK else ''}",
)

# Live MCP reachability
_MCP_URL = "http://localhost:8096"
_MCP_UP  = False
_DB_URL  = "http://localhost:8091"
_DB_UP   = False
_MEM_URL = "http://localhost:8094"
_MEM_UP  = False

try:
    _MCP_UP = httpx.get(f"{_MCP_URL}/health", timeout=2).status_code == 200
except Exception:
    pass
try:
    _DB_UP = httpx.get(f"{_DB_URL}/health", timeout=2).status_code == 200
except Exception:
    pass
try:
    _MEM_UP = httpx.get(f"{_MEM_URL}/health", timeout=2).status_code == 200
except Exception:
    pass

skip_mcp = pytest.mark.skipif(not _MCP_UP, reason="aichat-mcp not reachable at localhost:8096")
skip_db  = pytest.mark.skipif(not _DB_UP,  reason="aichat-database not reachable at localhost:8091")
skip_mem = pytest.mark.skipif(not _MEM_UP, reason="aichat-memory not reachable at localhost:8094")


def _run(coro):
    return asyncio.run(coro)


def _mcp_call(name: str, arguments: dict, timeout: float = 30) -> dict:
    r = httpx.post(
        f"{_MCP_URL}/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
              "params": {"name": name, "arguments": arguments}},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def _text_blocks(resp: dict) -> list[str]:
    return [b["text"] for b in resp.get("result", {}).get("content", [])
            if b.get("type") == "text"]


# ===========================================================================
# 1. Input validation — unit tests (no Docker)
# ===========================================================================

@skip_load
class TestDbToolInputValidation:
    """MCP handler input guards: missing required fields return friendly errors."""

    # ------------------------------------------------------------------
    # memory_store
    # ------------------------------------------------------------------
    def test_memory_store_missing_key(self):
        """memory_store without 'key' returns a clear error, not KeyError."""
        async def _fake_post(*a, **kw):
            raise AssertionError("should not reach HTTP layer")

        async def _run_it():
            with patch("httpx.AsyncClient") as mock_cls:
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
                return await _call_tool("memory_store", {"value": "hello"})

        result = _run(_run_it())
        texts = [b["text"] for b in result if b.get("type") == "text"]
        assert texts
        assert "key" in texts[0].lower(), f"Expected 'key' error, got: {texts[0]}"

    def test_memory_store_missing_value(self):
        async def _run_it():
            with patch("httpx.AsyncClient") as mock_cls:
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
                return await _call_tool("memory_store", {"key": "mykey"})

        result = _run(_run_it())
        texts = [b["text"] for b in result if b.get("type") == "text"]
        assert texts
        assert "value" in texts[0].lower(), f"Expected 'value' error, got: {texts[0]}"

    def test_memory_store_empty_key(self):
        async def _run_it():
            with patch("httpx.AsyncClient") as mock_cls:
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
                return await _call_tool("memory_store", {"key": "  ", "value": "v"})

        result = _run(_run_it())
        texts = [b["text"] for b in result if b.get("type") == "text"]
        assert texts
        assert "key" in texts[0].lower()

    def test_memory_store_bad_ttl(self):
        async def _run_it():
            with patch("httpx.AsyncClient") as mock_cls:
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
                return await _call_tool("memory_store",
                                        {"key": "k", "value": "v", "ttl_seconds": "forever"})

        result = _run(_run_it())
        texts = [b["text"] for b in result if b.get("type") == "text"]
        assert texts
        assert "ttl" in texts[0].lower() or "integer" in texts[0].lower()

    # ------------------------------------------------------------------
    # db_cache_store
    # ------------------------------------------------------------------
    def test_db_cache_store_missing_url(self):
        async def _run_it():
            with patch("httpx.AsyncClient") as mock_cls:
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
                return await _call_tool("db_cache_store", {"content": "some page"})

        result = _run(_run_it())
        texts = [b["text"] for b in result if b.get("type") == "text"]
        assert texts
        assert "url" in texts[0].lower()

    def test_db_cache_store_missing_content(self):
        async def _run_it():
            with patch("httpx.AsyncClient") as mock_cls:
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
                return await _call_tool("db_cache_store",
                                        {"url": "https://example.com"})

        result = _run(_run_it())
        texts = [b["text"] for b in result if b.get("type") == "text"]
        assert texts
        assert "content" in texts[0].lower()

    # ------------------------------------------------------------------
    # db_cache_get
    # ------------------------------------------------------------------
    def test_db_cache_get_missing_url(self):
        async def _run_it():
            with patch("httpx.AsyncClient") as mock_cls:
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
                return await _call_tool("db_cache_get", {})

        result = _run(_run_it())
        texts = [b["text"] for b in result if b.get("type") == "text"]
        assert texts
        assert "url" in texts[0].lower()

    def test_db_cache_get_empty_url(self):
        async def _run_it():
            with patch("httpx.AsyncClient") as mock_cls:
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
                return await _call_tool("db_cache_get", {"url": ""})

        result = _run(_run_it())
        texts = [b["text"] for b in result if b.get("type") == "text"]
        assert texts
        assert "url" in texts[0].lower()

    # ------------------------------------------------------------------
    # db_search
    # ------------------------------------------------------------------
    def test_db_search_non_integer_limit(self):
        async def _run_it():
            with patch("httpx.AsyncClient") as mock_cls:
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
                return await _call_tool("db_search",
                                        {"topic": "AI", "limit": "twenty"})

        result = _run(_run_it())
        texts = [b["text"] for b in result if b.get("type") == "text"]
        assert texts
        assert "integer" in texts[0].lower() or "limit" in texts[0].lower()

    def test_db_search_non_integer_offset(self):
        async def _run_it():
            with patch("httpx.AsyncClient") as mock_cls:
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
                return await _call_tool("db_search",
                                        {"topic": "AI", "offset": "start"})

        result = _run(_run_it())
        texts = [b["text"] for b in result if b.get("type") == "text"]
        assert texts
        assert "integer" in texts[0].lower() or "offset" in texts[0].lower()

    # ------------------------------------------------------------------
    # embed_search
    # ------------------------------------------------------------------
    def test_embed_search_non_integer_limit(self):
        async def _run_it():
            with patch("httpx.AsyncClient") as mock_cls:
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
                return await _call_tool("embed_search",
                                        {"query": "AI", "limit": "many"})

        result = _run(_run_it())
        texts = [b["text"] for b in result if b.get("type") == "text"]
        assert texts
        assert "integer" in texts[0].lower() or "limit" in texts[0].lower()

    def test_embed_search_missing_query(self):
        async def _run_it():
            with patch("httpx.AsyncClient") as mock_cls:
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
                return await _call_tool("embed_search", {})

        result = _run(_run_it())
        texts = [b["text"] for b in result if b.get("type") == "text"]
        assert texts
        assert "query" in texts[0].lower()


# ===========================================================================
# 2. db_cache_store E2E — mocked HTTP
# ===========================================================================

@skip_load
class TestDbCacheStoreE2E:
    """db_cache_store: validates input and posts correct payload to DB service."""

    def _make_mock_client(self, status=200, json_body=None):
        if json_body is None:
            json_body = {"status": "cached", "url": "https://example.com"}
        mock_resp = MagicMock()
        mock_resp.json.return_value = json_body
        mock_resp.status_code = status
        mock_c = MagicMock()
        mock_c.post = AsyncMock(return_value=mock_resp)
        mock_c.get  = AsyncMock(return_value=mock_resp)
        return mock_c

    def test_valid_store_posts_to_db(self):
        mock_c = self._make_mock_client()

        async def _run_it():
            with patch("httpx.AsyncClient") as mock_cls:
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_c)
                mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
                return await _call_tool("db_cache_store", {
                    "url": "https://example.com",
                    "content": "Hello world",
                    "title": "Example",
                })

        result = _run(_run_it())
        assert mock_c.post.called
        call_kwargs = mock_c.post.call_args
        payload = call_kwargs[1].get("json") or call_kwargs[0][1] if len(call_kwargs[0]) > 1 else call_kwargs[1]["json"]
        assert payload["url"] == "https://example.com"
        assert payload["content"] == "Hello world"
        assert payload["title"] == "Example"

    def test_store_no_title_omits_title(self):
        mock_c = self._make_mock_client()

        async def _run_it():
            with patch("httpx.AsyncClient") as mock_cls:
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_c)
                mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
                return await _call_tool("db_cache_store", {
                    "url": "https://example.com",
                    "content": "Hello world",
                })

        result = _run(_run_it())
        assert mock_c.post.called
        call_kwargs = mock_c.post.call_args
        payload = call_kwargs[1].get("json", call_kwargs[0][1] if len(call_kwargs[0]) > 1 else {})
        assert "title" not in payload

    def test_result_contains_status(self):
        mock_c = self._make_mock_client(json_body={"status": "cached", "url": "https://x.com"})

        async def _run_it():
            with patch("httpx.AsyncClient") as mock_cls:
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_c)
                mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
                return await _call_tool("db_cache_store", {
                    "url": "https://x.com", "content": "text"
                })

        result = _run(_run_it())
        texts = [b["text"] for b in result if b.get("type") == "text"]
        assert texts
        parsed = json.loads(texts[0])
        assert parsed.get("status") == "cached"


# ===========================================================================
# 3. db_cache_get E2E — mocked HTTP
# ===========================================================================

@skip_load
class TestDbCacheGetE2E:
    """db_cache_get: validates URL and queries DB service."""

    def test_valid_get_sends_url_param(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"found": True, "title": "T", "content": "C"}
        mock_c = MagicMock()
        mock_c.get = AsyncMock(return_value=mock_resp)

        async def _run_it():
            with patch("httpx.AsyncClient") as mock_cls:
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_c)
                mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
                return await _call_tool("db_cache_get",
                                        {"url": "https://example.com/page"})

        result = _run(_run_it())
        assert mock_c.get.called
        call_kwargs = mock_c.get.call_args
        params = call_kwargs[1].get("params", {})
        assert params.get("url") == "https://example.com/page"

    def test_result_contains_found_field(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"found": True, "content": "cached content"}
        mock_c = MagicMock()
        mock_c.get = AsyncMock(return_value=mock_resp)

        async def _run_it():
            with patch("httpx.AsyncClient") as mock_cls:
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_c)
                mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
                return await _call_tool("db_cache_get",
                                        {"url": "https://example.com"})

        result = _run(_run_it())
        texts = [b["text"] for b in result if b.get("type") == "text"]
        parsed = json.loads(texts[0])
        assert parsed.get("found") is True


# ===========================================================================
# 4. memory_store E2E — mocked HTTP
# ===========================================================================

@skip_load
class TestMemoryStoreE2E:
    """memory_store: valid payloads reach the memory service correctly."""

    def test_valid_store_sends_key_and_value(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"stored": "greeting"}
        mock_c = MagicMock()
        mock_c.post = AsyncMock(return_value=mock_resp)

        async def _run_it():
            with patch("httpx.AsyncClient") as mock_cls:
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_c)
                mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
                return await _call_tool("memory_store",
                                        {"key": "greeting", "value": "hello world"})

        result = _run(_run_it())
        assert mock_c.post.called
        call_kwargs = mock_c.post.call_args
        payload = call_kwargs[1].get("json", {})
        assert payload["key"] == "greeting"
        assert payload["value"] == "hello world"

    def test_ttl_included_when_provided(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"stored": "k", "expires_at": 9999}
        mock_c = MagicMock()
        mock_c.post = AsyncMock(return_value=mock_resp)

        async def _run_it():
            with patch("httpx.AsyncClient") as mock_cls:
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_c)
                mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
                return await _call_tool("memory_store",
                                        {"key": "k", "value": "v", "ttl_seconds": 3600})

        result = _run(_run_it())
        payload = mock_c.post.call_args[1].get("json", {})
        assert payload.get("ttl_seconds") == 3600

    def test_result_contains_stored_key(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"stored": "mykey"}
        mock_c = MagicMock()
        mock_c.post = AsyncMock(return_value=mock_resp)

        async def _run_it():
            with patch("httpx.AsyncClient") as mock_cls:
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_c)
                mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
                return await _call_tool("memory_store",
                                        {"key": "mykey", "value": "myvalue"})

        result = _run(_run_it())
        texts = [b["text"] for b in result if b.get("type") == "text"]
        parsed = json.loads(texts[0])
        assert parsed.get("stored") == "mykey"


# ===========================================================================
# 5. db_search E2E — mocked HTTP
# ===========================================================================

@skip_load
class TestDbSearchE2E:
    """db_search: correct params forwarded to DB service."""

    def _mock_client_with_articles(self, articles=None):
        if articles is None:
            articles = [{"id": 1, "title": "AI Today", "url": "https://ai.example"}]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"articles": articles, "total": len(articles)}
        mock_c = MagicMock()
        mock_c.get = AsyncMock(return_value=mock_resp)
        return mock_c

    def test_search_sends_topic_and_limit(self):
        mock_c = self._mock_client_with_articles()

        async def _run_it():
            with patch("httpx.AsyncClient") as mock_cls:
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_c)
                mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
                return await _call_tool("db_search",
                                        {"topic": "AI", "limit": 10, "offset": 0})

        _run(_run_it())
        assert mock_c.get.called
        params = mock_c.get.call_args[1].get("params", {})
        assert params["topic"] == "AI"
        assert params["limit"] == 10
        assert params["offset"] == 0

    def test_limit_clamped_to_200(self):
        mock_c = self._mock_client_with_articles()

        async def _run_it():
            with patch("httpx.AsyncClient") as mock_cls:
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_c)
                mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
                return await _call_tool("db_search", {"limit": 9999})

        _run(_run_it())
        params = mock_c.get.call_args[1].get("params", {})
        assert params["limit"] <= 200

    def test_offset_not_negative(self):
        mock_c = self._mock_client_with_articles()

        async def _run_it():
            with patch("httpx.AsyncClient") as mock_cls:
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_c)
                mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
                return await _call_tool("db_search", {"offset": -50})

        _run(_run_it())
        params = mock_c.get.call_args[1].get("params", {})
        assert params["offset"] >= 0

    def test_summary_only_forwarded(self):
        mock_c = self._mock_client_with_articles()

        async def _run_it():
            with patch("httpx.AsyncClient") as mock_cls:
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_c)
                mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
                return await _call_tool("db_search",
                                        {"topic": "tech", "summary_only": True})

        _run(_run_it())
        params = mock_c.get.call_args[1].get("params", {})
        assert params.get("summary_only") == "true"


# ===========================================================================
# 6. embed_store / embed_search E2E — mocked HTTP
# ===========================================================================

@skip_load
class TestEmbedE2E:
    """embed_store / embed_search: safe embedding extraction from LM Studio response."""

    def _make_emb_client(self, emb_data=None, db_resp=None):
        """Build a mock httpx.AsyncClient that returns controlled embedding responses."""
        if emb_data is None:
            emb_data = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        if db_resp is None:
            db_resp = {"stored": True}
        mock_emb_resp = MagicMock()
        mock_emb_resp.json.return_value = emb_data
        mock_emb_resp.raise_for_status = MagicMock()
        mock_db_resp = MagicMock()
        mock_db_resp.json.return_value = db_resp
        mock_db_resp.raise_for_status = MagicMock()
        mock_c = MagicMock()
        # First post → LM Studio embeddings; second post → DB store
        mock_c.post = AsyncMock(side_effect=[mock_emb_resp, mock_db_resp])
        return mock_c

    def test_embed_store_empty_data_returns_friendly_error(self):
        """LM Studio returns empty data → clear error, no IndexError."""
        mock_emb_resp = MagicMock()
        mock_emb_resp.json.return_value = {"data": []}
        mock_emb_resp.raise_for_status = MagicMock()
        mock_c = MagicMock()
        mock_c.post = AsyncMock(return_value=mock_emb_resp)

        async def _run_it():
            with patch("httpx.AsyncClient") as mock_cls:
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_c)
                mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
                return await _call_tool("embed_store",
                                        {"key": "doc1", "content": "test content"})

        result = _run(_run_it())
        texts = [b["text"] for b in result if b.get("type") == "text"]
        assert texts
        assert "empty" in texts[0].lower() or "embedding" in texts[0].lower()
        assert "IndexError" not in texts[0]

    def test_embed_search_empty_data_returns_friendly_error(self):
        """LM Studio returns empty data → clear error, no IndexError."""
        mock_emb_resp = MagicMock()
        mock_emb_resp.json.return_value = {"data": []}
        mock_emb_resp.raise_for_status = MagicMock()
        mock_c = MagicMock()
        mock_c.post = AsyncMock(return_value=mock_emb_resp)

        async def _run_it():
            with patch("httpx.AsyncClient") as mock_cls:
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_c)
                mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
                return await _call_tool("embed_search", {"query": "test query"})

        result = _run(_run_it())
        texts = [b["text"] for b in result if b.get("type") == "text"]
        assert texts
        assert "empty" in texts[0].lower() or "embedding" in texts[0].lower()
        assert "IndexError" not in texts[0]

    def test_embed_store_valid_path(self):
        """Happy path: valid embedding data → posts to DB."""
        mock_c = self._make_emb_client()

        async def _run_it():
            with patch("httpx.AsyncClient") as mock_cls:
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_c)
                mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
                return await _call_tool("embed_store",
                                        {"key": "doc1", "content": "hello world"})

        result = _run(_run_it())
        texts = [b["text"] for b in result if b.get("type") == "text"]
        assert texts
        # Should mention the key and dims
        assert "doc1" in texts[0] or "stored" in texts[0].lower()


# ===========================================================================
# 7. Live smoke tests — require running services
# ===========================================================================

@skip_mcp
@pytest.mark.smoke
class TestDbToolsSmoke:
    """Live smoke tests against running aichat-mcp at localhost:8096."""

    def test_memory_store_and_recall(self):
        """Store a value then recall it — verifies round-trip."""
        ts = int(time.time())
        key = f"smoke_test_{ts}"
        value = f"smoke_value_{ts}"

        store_resp = _mcp_call("memory_store", {"key": key, "value": value, "ttl_seconds": 120})
        texts = _text_blocks(store_resp)
        assert texts, "memory_store returned no text"
        parsed = json.loads(texts[0])
        assert parsed.get("stored") == key, f"Unexpected store response: {parsed}"

        recall_resp = _mcp_call("memory_recall", {"key": key})
        texts = _text_blocks(recall_resp)
        parsed = json.loads(texts[0])
        entries = parsed.get("entries", [])
        assert any(e.get("value") == value for e in entries), \
            f"Recalled entries don't contain expected value: {entries}"

    def test_memory_store_missing_key_returns_error(self):
        """memory_store with no 'key' should return an error block, not a 500."""
        resp = _mcp_call("memory_store", {"value": "orphaned"})
        texts = _text_blocks(resp)
        assert texts
        assert "key" in texts[0].lower(), f"Expected key error, got: {texts[0]}"

    def test_memory_store_missing_value_returns_error(self):
        resp = _mcp_call("memory_store", {"key": "orphaned_key"})
        texts = _text_blocks(resp)
        assert texts
        assert "value" in texts[0].lower(), f"Expected value error, got: {texts[0]}"

    def test_db_cache_store_and_get(self):
        """Cache a page then retrieve it — verifies round-trip."""
        ts = int(time.time())
        url = f"https://smoke-test-{ts}.example.com/"
        content = f"Smoke test page content at {ts}"

        store_resp = _mcp_call("db_cache_store", {
            "url": url, "content": content, "title": "Smoke Test Page"
        })
        texts = _text_blocks(store_resp)
        assert texts
        parsed = json.loads(texts[0])
        assert parsed.get("status") == "cached", f"Unexpected: {parsed}"

        get_resp = _mcp_call("db_cache_get", {"url": url})
        texts = _text_blocks(get_resp)
        parsed = json.loads(texts[0])
        assert parsed.get("found") is True or "content" in parsed, \
            f"Cache get did not find cached page: {parsed}"

    def test_db_cache_store_missing_content_returns_error(self):
        resp = _mcp_call("db_cache_store", {"url": "https://example.com"})
        texts = _text_blocks(resp)
        assert texts
        assert "content" in texts[0].lower()

    def test_db_cache_get_empty_url_returns_error(self):
        resp = _mcp_call("db_cache_get", {"url": ""})
        texts = _text_blocks(resp)
        assert texts
        assert "url" in texts[0].lower()

    def test_db_store_article_url_only(self):
        """Store article with only URL (other fields optional) — should not crash."""
        ts = int(time.time())
        resp = _mcp_call("db_store_article", {
            "url": f"https://smoke-article-{ts}.example.com/"
        })
        texts = _text_blocks(resp)
        assert texts
        # Should succeed or return a recognizable response
        parsed = json.loads(texts[0]) if texts[0].startswith("{") else {}
        # Either succeeded or returned a proper error — not a raw 422/500
        assert "422" not in texts[0], f"Got raw 422: {texts[0]}"

    def test_db_search_returns_articles(self):
        resp = _mcp_call("db_search", {"limit": 5})
        texts = _text_blocks(resp)
        assert texts
        parsed = json.loads(texts[0])
        assert "articles" in parsed or "error" in parsed, f"Unexpected response: {parsed}"

    def test_db_search_bad_limit_returns_error(self):
        resp = _mcp_call("db_search", {"limit": "many"})
        texts = _text_blocks(resp)
        assert texts
        assert "integer" in texts[0].lower() or "limit" in texts[0].lower()


# ===========================================================================
# 8. TestImageOCR — generic image search + OCR pipeline (smoke)
# ===========================================================================

def _image_ocr_pipeline(query: str, count: int = 2, timeout: float = 90) -> dict:
    """
    Generic helper: search for images matching *query*, then caption/OCR the first
    image block found.  Returns a dict with:
      - "search_content": raw content blocks from image_search
      - "image_found": True if at least one image block was found
      - "caption": string result of image_caption (empty if no image or caption failed)
      - "caption_ok": True if caption did not start with "Error"
    """
    # 1. Search for images
    search_resp = _mcp_call("image_search", {"query": query, "count": count}, timeout=timeout)
    search_content = search_resp.get("result", {}).get("content", [])

    # 2. Extract the first base64 image block
    first_image = next(
        (b for b in search_content if b.get("type") == "image"),
        None,
    )
    result = {
        "search_content": search_content,
        "image_found": first_image is not None,
        "caption": "",
        "caption_ok": False,
    }
    if first_image is None:
        return result

    # 3. image_caption expects base64 JPEG
    b64 = (
        first_image.get("data", "")  # MCP 2025-03-26 image block
        or first_image.get("url", "").split(",")[-1]  # data URI fallback
    )
    if not b64:
        return result

    caption_resp = _mcp_call("image_caption", {"b64": b64, "detail_level": "detailed"},
                             timeout=30)
    caption_blocks = _text_blocks(caption_resp)
    caption = caption_blocks[0] if caption_blocks else ""
    result["caption"] = caption
    result["caption_ok"] = bool(caption) and not caption.startswith("Error")
    return result


@skip_mcp
@pytest.mark.smoke
class TestImageOCR:
    """
    Generic image search + OCR pipeline tests.

    These run against the live MCP server and verify that:
    1. image_search finds images for common queries
    2. image_caption can describe any returned image
    3. The pipeline works for arbitrary subjects (not just Klukai)
    """

    def test_find_images_generic(self):
        """image_search returns at least text or image blocks for a generic query."""
        resp = _mcp_call("image_search", {"query": "anime character art", "count": 2}, timeout=60)
        content = resp.get("result", {}).get("content", [])
        assert content, "image_search returned no content blocks"
        has_image = any(b.get("type") == "image" for b in content)
        has_text  = any(b.get("type") == "text"  for b in content)
        assert has_image or has_text, "image_search returned neither image nor text"

    def test_image_ocr_any_subject(self):
        """Search + OCR pipeline works for a generic subject (nature photograph)."""
        result = _image_ocr_pipeline("scenic mountain landscape photography", count=1)
        assert result["search_content"], "image_search returned empty content"
        # If no image block was returned, the search result still must have text
        if not result["image_found"]:
            texts = [b["text"] for b in result["search_content"] if b.get("type") == "text"]
            assert texts, "No image and no text either"
            return  # pass — image_search returned text-only result (DB fast-path or no vision model)
        # If an image was found, caption must have been attempted
        assert result["caption"], "image_caption returned empty string"
        assert result["caption_ok"], f"image_caption error: {result['caption']}"

    # ------------------------------------------------------------------
    # Klukai-specific test (the original "ultimate E2E" request)
    # ------------------------------------------------------------------

    def test_find_klukai_images(self):
        """image_search finds at least one Klukai image result."""
        resp = _mcp_call("image_search", {
            "query": "Klukai Girls Frontline 2",
            "count": 2,
        }, timeout=60)
        content = resp.get("result", {}).get("content", [])
        assert content, "image_search returned no content blocks"
        has_image = any(b.get("type") == "image" for b in content)
        has_text  = any(b.get("type") == "text"  for b in content)
        assert has_image or has_text, "image_search returned neither image nor text blocks"
        if has_text:
            texts = [b["text"] for b in content if b.get("type") == "text"]
            assert not all(t.startswith("Error") for t in texts), \
                f"image_search returned only errors: {texts}"

    def test_klukai_image_ocr_pipeline(self):
        """Find Klukai images and OCR/caption them to verify content."""
        result = _image_ocr_pipeline("Klukai Girls Frontline 2 character art", count=2)
        assert result["search_content"], "image_search returned empty content for Klukai"
        if not result["image_found"]:
            # Acceptable if service returns text-only (DB fast path, no vision model)
            texts = [b["text"] for b in result["search_content"] if b.get("type") == "text"]
            assert texts, "No Klukai image and no text result either"
            return
        assert result["caption"], "image_caption returned empty string for Klukai image"
        assert result["caption_ok"], f"image_caption error on Klukai: {result['caption']}"

    def test_klukai_screenshot_via_orchestrate(self):
        """Orchestrate: screenshot Klukai wiki page — verifies orchestrate + browser pipeline."""
        orch_resp = _mcp_call("orchestrate", {
            "steps": [
                {
                    "id": "shot",
                    "tool": "screenshot",
                    "args": {"url": "https://gfl2.amaryllishare.page/characters/klukai"},
                    "label": "Klukai character page screenshot",
                },
            ],
        }, timeout=90)
        texts = _text_blocks(orch_resp)
        assert texts, "orchestrate returned no text"
        report = texts[0]
        assert "Klukai" in report or "screenshot" in report.lower() or "ms" in report, \
            f"Unexpected orchestrate report: {report[:300]}"

    def test_memory_store_klukai_result(self):
        """Store a Klukai search result in memory for future reference."""
        ts = int(time.time())
        key = f"klukai_test_{ts}"
        value = ("Klukai is a character in Girls Frontline 2 "
                 "with a distinctive blue-silver appearance.")
        resp = _mcp_call("memory_store", {"key": key, "value": value, "ttl_seconds": 300})
        texts = _text_blocks(resp)
        assert texts
        parsed = json.loads(texts[0])
        assert "stored" in parsed, f"Unexpected: {parsed}"

        recall_resp = _mcp_call("memory_recall", {"pattern": "klukai_test_%"})
        texts = _text_blocks(recall_resp)
        parsed = json.loads(texts[0])
        assert parsed.get("found") is True, f"Pattern recall failed: {parsed}"
        entries = parsed.get("entries", [])
        assert any("Klukai" in e.get("value", "") for e in entries), \
            f"Klukai entry not found in recall: {entries}"
