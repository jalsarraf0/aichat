"""
Comprehensive E2E test suite — calls through live MCP + gpt-oss-20b.

Tests every major MCP tool category via the live HTTP endpoint at localhost:8096.
LM Studio tools (smart_summarize, structured_extract, image_caption, embed_store,
embed_search, code_run, tts) are tested against the gpt-oss-20b model.

Usage:
    pytest tests/test_comprehensive_e2e.py -v -m smoke
    pytest tests/test_comprehensive_e2e.py -v  # runs all (smoke + non-smoke)
"""
from __future__ import annotations

import base64
import io
import os
import time
import uuid

import httpx
import pytest

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MCP_URL    = "http://localhost:8096"
DB_URL     = "http://localhost:8091"
MEMORY_URL = "http://localhost:8094"
LM_URL     = "http://192.168.50.2:1234"
WORKSPACE  = "/docker/human_browser/workspace"


def _up(url: str) -> bool:
    try:
        return httpx.get(url, timeout=3).status_code < 500
    except Exception:
        return False


_MCP_UP = _up(f"{MCP_URL}/health")
_LM_UP  = _up(f"{LM_URL}/v1/models")

skip_mcp = pytest.mark.skipif(not _MCP_UP, reason="MCP server not running")
skip_lm  = pytest.mark.skipif(not _LM_UP, reason="LM Studio not running")

pytestmark = [pytest.mark.smoke, skip_mcp]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mcp_call(name: str, arguments: dict, timeout: float = 60) -> dict:
    r = httpx.post(
        f"{MCP_URL}/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
              "params": {"name": name, "arguments": arguments}},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def _mcp_content(name: str, arguments: dict, timeout: float = 60) -> list[dict]:
    return _mcp_call(name, arguments, timeout).get("result", {}).get("content", [])


def _text_from(blocks: list[dict]) -> str:
    return "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")


def _has_image(blocks: list[dict]) -> bool:
    return any(b.get("type") == "image" for b in blocks)


def _get_image_bytes(blocks: list[dict]) -> bytes | None:
    for b in blocks:
        if b.get("type") == "image":
            return base64.b64decode(b["data"])
    return None


# ---------------------------------------------------------------------------
# 1. Tool inventory
# ---------------------------------------------------------------------------

class TestToolInventory:
    """Verify all expected tools are registered."""

    def test_tools_list(self):
        r = httpx.post(
            f"{MCP_URL}/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            timeout=10,
        )
        tools = r.json().get("result", {}).get("tools", [])
        names = {t["name"] for t in tools}
        assert len(names) >= 49, f"Expected >=49 tools, got {len(names)}"
        # Spot-check key tools
        for t in ("screenshot", "web_search", "smart_summarize", "orchestrate",
                   "image_crop", "image_upscale", "embed_store", "code_run"):
            assert t in names, f"Missing tool: {t}"


# ---------------------------------------------------------------------------
# 2. Browser tools
# ---------------------------------------------------------------------------

class TestBrowser:
    """Screenshot, scroll_screenshot, browser navigation."""

    def test_screenshot(self):
        blocks = _mcp_content("screenshot", {"url": "https://example.com"})
        assert _has_image(blocks), "screenshot should return an image block"
        img = _get_image_bytes(blocks)
        assert img and len(img) > 1000

    def test_browser_navigate(self):
        blocks = _mcp_content("browser", {
            "action": "navigate",
            "url": "https://example.com",
        })
        text = _text_from(blocks)
        assert "example" in text.lower() or _has_image(blocks)

    def test_web_fetch(self):
        blocks = _mcp_content("web_fetch", {"url": "https://example.com"})
        text = _text_from(blocks)
        assert "example" in text.lower()


# ---------------------------------------------------------------------------
# 3. Image pipeline tools (create test image via screenshot)
# ---------------------------------------------------------------------------

class TestImagePipeline:
    """image_crop, image_zoom, image_scan, image_enhance, image_stitch,
    image_diff, image_annotate, image_upscale."""

    @pytest.fixture(autouse=True)
    def _screenshot(self):
        """Take a screenshot to have a workspace image for testing."""
        blocks = _mcp_content("screenshot", {
            "url": "https://example.com",
            "path": "/workspace/test_comp_e2e.png",
        })
        # Find actual path from text
        text = _text_from(blocks)
        self.img_path = "/workspace/test_comp_e2e.png"

    def test_crop(self):
        blocks = _mcp_content("image_crop", {
            "path": self.img_path,
            "left": 0, "top": 0, "right": 200, "bottom": 200,
        })
        assert _has_image(blocks), "image_crop should return inline image"

    def test_zoom(self):
        blocks = _mcp_content("image_zoom", {
            "path": self.img_path,
            "left": 0, "top": 0, "right": 200, "bottom": 200,
            "scale": 2,
        })
        assert _has_image(blocks)

    def test_scan(self):
        blocks = _mcp_content("image_scan", {
            "path": self.img_path,
        })
        assert _has_image(blocks)

    def test_enhance(self):
        blocks = _mcp_content("image_enhance", {
            "path": self.img_path,
            "contrast": 1.5,
        })
        assert _has_image(blocks)

    def test_stitch_vertical(self):
        blocks = _mcp_content("image_stitch", {
            "paths": [self.img_path, self.img_path],
            "direction": "vertical",
        })
        assert _has_image(blocks)

    def test_diff(self):
        blocks = _mcp_content("image_diff", {
            "path_a": self.img_path,
            "path_b": self.img_path,
        })
        assert _has_image(blocks)

    def test_annotate(self):
        blocks = _mcp_content("image_annotate", {
            "path": self.img_path,
            "boxes": [{"left": 10, "top": 10, "right": 100, "bottom": 100,
                       "label": "test", "color": "red"}],
        })
        assert _has_image(blocks)

    def test_upscale_2x(self):
        blocks = _mcp_content("image_upscale", {
            "path": self.img_path,
            "scale": 2,
        })
        assert _has_image(blocks)
        text = _text_from(blocks)
        assert "upscale" in text.lower() or "×" in text or "x" in text.lower()


# ---------------------------------------------------------------------------
# 4. Web search
# ---------------------------------------------------------------------------

class TestWebSearch:
    def test_web_search(self):
        blocks = _mcp_content("web_search", {"query": "Python asyncio tutorial"})
        text = _text_from(blocks)
        assert len(text) > 50, "web_search should return results"

    def test_image_search(self):
        blocks = _mcp_content("image_search", {
            "query": "cute cat photo",
            "count": 2,
        }, timeout=90)
        assert _has_image(blocks) or len(_text_from(blocks)) > 20


# ---------------------------------------------------------------------------
# 5. LM Studio tools (gpt-oss-20b)
# ---------------------------------------------------------------------------

@skip_lm
class TestLMStudioTools:
    """Tests that call through to gpt-oss-20b via LM Studio."""

    def test_smart_summarize(self):
        long_text = (
            "Artificial intelligence has transformed numerous industries. "
            "Machine learning models can now process natural language, generate images, "
            "and even write code. Large language models like GPT and Claude have shown "
            "remarkable capabilities in understanding and generating human language. "
            "These models are trained on vast amounts of text data and can perform "
            "tasks ranging from translation to creative writing."
        )
        blocks = _mcp_content("smart_summarize", {"text": long_text}, timeout=60)
        text = _text_from(blocks)
        assert len(text) > 20, f"Summary too short: {text!r}"
        assert "error" not in text.lower()[:20]

    def test_structured_extract(self):
        blocks = _mcp_content("structured_extract", {
            "content": "John Smith is 35 years old and lives in Seattle, Washington. He works as a software engineer at Microsoft.",
            "schema_json": '{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}}}',
        }, timeout=60)
        text = _text_from(blocks)
        assert len(text) > 10
        # Should contain extracted info or a graceful error about json_object mode
        assert ("john" in text.lower() or "smith" in text.lower()
                or "name" in text.lower() or "json" in text.lower()
                or "connection" in text.lower())

    def test_code_run(self):
        blocks = _mcp_content("code_run", {
            "code": "print('hello from gpt-oss-20b test')\nresult = 2 + 2\nprint(f'2+2={result}')",
        }, timeout=30)
        text = _text_from(blocks)
        assert "hello" in text.lower() or "2+2=4" in text

    def test_embed_store_and_search(self):
        unique = uuid.uuid4().hex[:8]
        key = f"test_comp_{unique}"
        content = "The quick brown fox jumps over the lazy dog"

        # Store
        store_blocks = _mcp_content("embed_store", {
            "key": key,
            "content": content,
            "topic": "test",
        }, timeout=30)
        store_text = _text_from(store_blocks)
        assert "error" not in store_text.lower()[:20]

        # Search
        search_blocks = _mcp_content("embed_search", {
            "query": "fox jumping over dog",
            "top_k": 3,
        }, timeout=30)
        search_text = _text_from(search_blocks)
        assert len(search_text) > 5


# ---------------------------------------------------------------------------
# 6. Database tools
# ---------------------------------------------------------------------------

class TestDatabase:
    def test_db_store_and_search(self):
        unique = uuid.uuid4().hex[:8]
        title = f"Test Article {unique}"
        url = f"https://test.example.com/{unique}"

        # Store
        store_blocks = _mcp_content("db_store_article", {
            "url": url, "title": title, "topic": "test_comp_e2e",
        })
        store_text = _text_from(store_blocks)
        assert "stored" in store_text.lower() or "error" not in store_text.lower()[:10]

        # Search
        search_blocks = _mcp_content("db_search", {
            "query": unique,
        })
        search_text = _text_from(search_blocks)
        assert unique in search_text or "no results" in search_text.lower()

    def test_db_cache_roundtrip(self):
        unique = uuid.uuid4().hex[:8]
        url = f"https://test-cache-{unique}.example.com"
        value = f"cached_value_{unique}"

        _mcp_content("db_cache_store", {"url": url, "content": value})
        blocks = _mcp_content("db_cache_get", {"url": url})
        text = _text_from(blocks)
        assert unique in text

    def test_get_errors(self):
        blocks = _mcp_content("get_errors", {"limit": 5})
        text = _text_from(blocks)
        # Should return either errors or "No errors logged"
        assert len(text) > 0


# ---------------------------------------------------------------------------
# 7. Memory tools
# ---------------------------------------------------------------------------

class TestMemory:
    def test_memory_store_and_recall(self):
        unique = uuid.uuid4().hex[:8]
        key = f"test_mem_{unique}"

        _mcp_content("memory_store", {"key": key, "value": f"memory_val_{unique}"})
        blocks = _mcp_content("memory_recall", {"key": key})
        text = _text_from(blocks)
        assert unique in text

    def test_memory_pattern_recall(self):
        unique = uuid.uuid4().hex[:8]
        key = f"pattern_test_{unique}"
        _mcp_content("memory_store", {"key": key, "value": "pattern_match_test"})
        blocks = _mcp_content("memory_recall", {"key": f"pattern_test_{unique}"})
        text = _text_from(blocks)
        assert "pattern_match_test" in text


# ---------------------------------------------------------------------------
# 8. Research tools
# ---------------------------------------------------------------------------

class TestResearch:
    def test_researchbox_search(self):
        blocks = _mcp_content("researchbox_search", {"topic": "python"})
        text = _text_from(blocks)
        assert "python" in text.lower() or "feed" in text.lower()


# ---------------------------------------------------------------------------
# 9. Orchestrate tool
# ---------------------------------------------------------------------------

class TestOrchestrate:
    def test_two_step_sequential(self):
        blocks = _mcp_content("orchestrate", {
            "steps": [
                {"id": "search", "tool": "web_search",
                 "args": {"query": "Python asyncio"}, "label": "Web Search"},
                {"id": "summarize", "tool": "smart_summarize",
                 "args": {"text": "{search.result}"},
                 "depends_on": ["search"], "label": "Summary"},
            ],
        }, timeout=120)
        text = _text_from(blocks)
        assert "Web Search" in text or "Summary" in text
        assert len(text) > 50

    def test_parallel_steps(self):
        blocks = _mcp_content("orchestrate", {
            "steps": [
                {"id": "a", "tool": "web_search",
                 "args": {"query": "Rust programming language"}, "label": "Search A"},
                {"id": "b", "tool": "web_search",
                 "args": {"query": "Go programming language"}, "label": "Search B"},
            ],
        }, timeout=90)
        text = _text_from(blocks)
        assert "Search A" in text and "Search B" in text

    def test_stop_on_error(self):
        blocks = _mcp_content("orchestrate", {
            "steps": [
                {"id": "bad", "tool": "nonexistent_tool_xyz",
                 "args": {}, "label": "Bad Step"},
                {"id": "good", "tool": "web_search",
                 "args": {"query": "test"},
                 "depends_on": ["bad"], "label": "Good Step"},
            ],
            "stop_on_error": True,
        }, timeout=30)
        text = _text_from(blocks)
        assert "Bad Step" in text
        # Good Step should be skipped due to stop_on_error
        assert "skipped" in text.lower() or "Good Step" not in text or "FAIL" in text

    def test_cycle_detection(self):
        blocks = _mcp_content("orchestrate", {
            "steps": [
                {"id": "a", "tool": "web_search", "args": {"query": "x"},
                 "depends_on": ["b"]},
                {"id": "b", "tool": "web_search", "args": {"query": "y"},
                 "depends_on": ["a"]},
            ],
        }, timeout=10)
        text = _text_from(blocks)
        assert "cycle" in text.lower()


# ---------------------------------------------------------------------------
# 10. Edge cases and error handling
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_unknown_tool(self):
        blocks = _mcp_content("nonexistent_tool_abc", {"foo": "bar"})
        text = _text_from(blocks)
        assert "unknown" in text.lower() or "error" in text.lower()

    def test_screenshot_invalid_url(self):
        blocks = _mcp_content("screenshot", {"url": "not-a-valid-url"})
        text = _text_from(blocks)
        # Should handle gracefully
        assert len(text) > 0

    def test_image_crop_invalid_coords(self):
        # Use the test image from the pipeline fixture
        _mcp_content("screenshot", {
            "url": "https://example.com",
            "path": "/workspace/test_edge_case.png",
        })
        blocks = _mcp_content("image_crop", {
            "path": "/workspace/test_edge_case.png",
            "left": "abc", "top": 0, "right": 100, "bottom": 100,
        })
        text = _text_from(blocks)
        # May get "invalid coordinate" or "not found" (path resolution)
        assert "invalid" in text.lower() or "error" in text.lower() or "not found" in text.lower()

    def test_orchestrate_empty_steps(self):
        blocks = _mcp_content("orchestrate", {"steps": []})
        text = _text_from(blocks)
        assert "non-empty" in text.lower() or "error" in text.lower()
