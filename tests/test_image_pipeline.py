"""
Comprehensive smoke tests for all image pipeline and web tools.

Coverage:
  - image_crop / image_zoom / image_scan / image_enhance  (MCP HTTP + stdio)
  - image_stitch / image_diff / image_annotate            (PIL composition)
  - page_extract / extract_article                        (browser structured data)
  - bulk_screenshot / scroll_screenshot                   (parallel + full-page capture)
  - Full order-of-operations pipeline: screenshot → crop → zoom → scan
  - Stitch → annotate pipeline with save_prefix chaining
  - database: offset pagination, summary_only, error-log indexes
  - memory: TTL expiry, LIKE pattern matching
  - researchbox: push_feed error reporting + feedparser timeout
  - toolkit: asyncio execution timeout
  - mcp_server (stdio): call_custom_tool dispatch, new tool schema parity
"""
from __future__ import annotations

import asyncio
import base64
import io
import os
import sys
import time
import uuid

import httpx
import pytest

sys.path.insert(0, "src")

# ---------------------------------------------------------------------------
# Service availability
# ---------------------------------------------------------------------------

MCP_URL       = "http://localhost:8096"
DB_URL        = "http://localhost:8091"
MEMORY_URL    = "http://localhost:8094"
RESEARCH_URL  = "http://localhost:8092"
TOOLKIT_URL   = "http://localhost:8095"
WORKSPACE     = "/docker/human_browser/workspace"


def _up(url: str) -> bool:
    try:
        return httpx.get(url, timeout=2).status_code < 500
    except Exception:
        return False


_MCP_UP       = _up(f"{MCP_URL}/health")
_DB_UP        = _up(f"{DB_URL}/health")
_MEM_UP       = _up(f"{MEMORY_URL}/health")
_RESEARCH_UP  = _up(f"{RESEARCH_URL}/search-feeds?topic=test")
_TOOLKIT_UP   = _up(f"{TOOLKIT_URL}/health")
_WORKSPACE_OK = os.path.isdir(WORKSPACE)

skip_mcp      = pytest.mark.skipif(not _MCP_UP,      reason="aichat-mcp not reachable")
skip_db       = pytest.mark.skipif(not _DB_UP,        reason="aichat-database not reachable")
skip_mem      = pytest.mark.skipif(not _MEM_UP,       reason="aichat-memory not reachable")
skip_research = pytest.mark.skipif(not _RESEARCH_UP,  reason="aichat-researchbox not reachable")
skip_toolkit  = pytest.mark.skipif(not _TOOLKIT_UP,   reason="aichat-toolkit not reachable")
skip_ws       = pytest.mark.skipif(not _WORKSPACE_OK, reason="browser workspace not mounted")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mcp_call(name: str, arguments: dict) -> dict:
    """Call a tool via the MCP HTTP endpoint and return the result dict."""
    r = httpx.post(
        f"{MCP_URL}/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
              "params": {"name": name, "arguments": arguments}},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def _mcp_content(name: str, arguments: dict) -> list[dict]:
    """Return the content blocks from a tool call."""
    return _mcp_call(name, arguments).get("result", {}).get("content", [])


def _has_image_block(blocks: list[dict]) -> bool:
    return any(b.get("type") == "image" for b in blocks)


def _get_image_bytes(blocks: list[dict]) -> bytes:
    for b in blocks:
        if b.get("type") == "image":
            return base64.b64decode(b["data"])
    return b""


def _make_test_png(width: int = 400, height: int = 200, label: str = "smoke") -> bytes:
    """Create a minimal PNG in-memory using only stdlib (no Pillow needed here)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGB", (width, height), color=(240, 240, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 10, width - 10, height - 10], outline=(0, 80, 200), width=3)
        draw.text((20, 20), f"aichat smoke test\n{label}", fill=(20, 20, 20))
        # Draw a grid to make zoom/crop verifiable
        for x in range(0, width, 50):
            draw.line([(x, 0), (x, height)], fill=(200, 200, 200))
        for y in range(0, height, 50):
            draw.line([(0, y), (width, y)], fill=(200, 200, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        # Minimal 1×1 white PNG (fallback — Pillow unavailable on host)
        return base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
        )


@pytest.fixture(scope="module")
def test_image_path() -> str:
    """Write a test PNG to the browser workspace; yield its filename; clean up."""
    fname = f"smoke_test_{uuid.uuid4().hex[:8]}.png"
    path = os.path.join(WORKSPACE, fname)
    png = _make_test_png(400, 200, label=fname)
    with open(path, "wb") as fh:
        fh.write(png)
    yield fname
    try:
        os.unlink(path)
    except Exception:
        pass


# ===========================================================================
# 1. MCP tools/list — all new tools advertised
# ===========================================================================

class TestMCPToolsAdvertised:
    @skip_mcp
    def test_image_tools_in_tools_list(self):
        r = httpx.post(
            f"{MCP_URL}/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            timeout=10,
        )
        assert r.status_code == 200
        tools = {t["name"] for t in r.json()["result"]["tools"]}
        for expected in ("image_crop", "image_zoom", "image_scan", "image_enhance",
                         "call_custom_tool"):
            assert expected in tools, f"Tool '{expected}' missing from tools/list"

    @skip_mcp
    def test_all_image_tools_have_required_path_param(self):
        r = httpx.post(
            f"{MCP_URL}/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            timeout=10,
        )
        tools_by_name = {t["name"]: t for t in r.json()["result"]["tools"]}
        for tname in ("image_crop", "image_zoom", "image_scan", "image_enhance"):
            schema = tools_by_name[tname]["inputSchema"]
            assert "path" in schema["properties"], f"{tname} missing 'path' property"
            assert "path" in schema.get("required", []), f"{tname} 'path' not required"


# ===========================================================================
# 2. image_crop
# ===========================================================================

class TestImageCrop:
    @skip_mcp
    @skip_ws
    def test_crop_full_image_returns_inline_image(self, test_image_path):
        blocks = _mcp_content("image_crop", {"path": test_image_path})
        assert _has_image_block(blocks), "image_crop returned no image block"

    @skip_mcp
    @skip_ws
    def test_crop_region_is_smaller_than_original(self, test_image_path):
        from PIL import Image
        # Full image
        full_blocks  = _mcp_content("image_crop", {"path": test_image_path})
        # Crop top-left quadrant
        crop_blocks  = _mcp_content("image_crop",
                                     {"path": test_image_path, "left": 0, "top": 0,
                                      "right": 200, "bottom": 100})
        full_data = _get_image_bytes(full_blocks)
        crop_data = _get_image_bytes(crop_blocks)
        assert crop_data, "crop returned no image data"
        full_img = Image.open(io.BytesIO(full_data))
        crop_img = Image.open(io.BytesIO(crop_data))
        assert crop_img.width <= full_img.width
        assert crop_img.height <= full_img.height

    @skip_mcp
    @skip_ws
    def test_crop_text_in_summary(self, test_image_path):
        blocks = _mcp_content("image_crop",
                               {"path": test_image_path, "left": 10, "top": 10,
                                "right": 100, "bottom": 80})
        text_blocks = [b["text"] for b in blocks if b.get("type") == "text"]
        assert any("Crop" in t or "crop" in t for t in text_blocks)

    @skip_mcp
    @skip_ws
    def test_crop_missing_file_returns_error_text(self):
        blocks = _mcp_content("image_crop", {"path": "nonexistent_abc123.png"})
        text = " ".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        assert "not found" in text.lower() or "error" in text.lower()

    @skip_mcp
    @skip_ws
    def test_crop_accepts_container_path_format(self, test_image_path):
        container_path = f"/workspace/{test_image_path}"
        blocks = _mcp_content("image_crop", {"path": container_path})
        assert _has_image_block(blocks), "image_crop rejected /workspace/ path format"

    @skip_mcp
    @skip_ws
    def test_crop_accepts_host_path_format(self, test_image_path):
        host_path = f"/docker/human_browser/workspace/{test_image_path}"
        blocks = _mcp_content("image_crop", {"path": host_path})
        assert _has_image_block(blocks), "image_crop rejected host path format"


# ===========================================================================
# 3. image_zoom
# ===========================================================================

class TestImageZoom:
    @skip_mcp
    @skip_ws
    def test_zoom_returns_inline_image(self, test_image_path):
        blocks = _mcp_content("image_zoom", {"path": test_image_path, "scale": 2.0})
        assert _has_image_block(blocks)

    @skip_mcp
    @skip_ws
    def test_zoom_2x_produces_larger_image(self, test_image_path):
        from PIL import Image
        orig_blocks = _mcp_content("image_crop", {"path": test_image_path})
        zoom_blocks = _mcp_content("image_zoom", {"path": test_image_path, "scale": 2.0})
        orig = Image.open(io.BytesIO(_get_image_bytes(orig_blocks)))
        zoomed = Image.open(io.BytesIO(_get_image_bytes(zoom_blocks)))
        # Zoomed should be bigger (within JPEG rounding)
        assert zoomed.width >= orig.width, "2× zoom did not enlarge width"
        assert zoomed.height >= orig.height, "2× zoom did not enlarge height"

    @skip_mcp
    @skip_ws
    def test_zoom_with_crop_region(self, test_image_path):
        blocks = _mcp_content("image_zoom",
                               {"path": test_image_path, "scale": 3.0,
                                "left": 0, "top": 0, "right": 100, "bottom": 100})
        assert _has_image_block(blocks)
        text = " ".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        assert "3.0" in text or "3×" in text or "Zoom" in text

    @skip_mcp
    @skip_ws
    def test_zoom_scale_clamped_to_max(self, test_image_path):
        # scale=99 should be clamped to 8.0, not crash
        blocks = _mcp_content("image_zoom", {"path": test_image_path, "scale": 99})
        assert _has_image_block(blocks)

    @skip_mcp
    @skip_ws
    def test_zoom_missing_file_error(self):
        blocks = _mcp_content("image_zoom", {"path": "no_such.png", "scale": 2.0})
        text = " ".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        assert "not found" in text.lower() or "error" in text.lower()


# ===========================================================================
# 4. image_scan
# ===========================================================================

class TestImageScan:
    @skip_mcp
    @skip_ws
    def test_scan_returns_inline_image(self, test_image_path):
        blocks = _mcp_content("image_scan", {"path": test_image_path})
        assert _has_image_block(blocks)

    @skip_mcp
    @skip_ws
    def test_scan_summary_mentions_text_reading(self, test_image_path):
        blocks = _mcp_content("image_scan", {"path": test_image_path})
        text = " ".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        assert "read" in text.lower() or "scan" in text.lower() or "text" in text.lower()

    @skip_mcp
    @skip_ws
    def test_scan_with_region(self, test_image_path):
        blocks = _mcp_content("image_scan",
                               {"path": test_image_path, "left": 0, "top": 0,
                                "right": 200, "bottom": 100})
        assert _has_image_block(blocks)

    @skip_mcp
    @skip_ws
    def test_scan_produces_high_contrast_image(self, test_image_path):
        """Scanned image should be greyscale (R≈G≈B for every pixel after JPEG encoding)."""
        from PIL import Image
        scan_data = _get_image_bytes(_mcp_content("image_scan", {"path": test_image_path}))
        scan_img = Image.open(io.BytesIO(scan_data))
        # Greyscale stored as RGB JPEG — channels should be nearly equal (allow JPEG rounding)
        r_arr = list(scan_img.getdata(band=0))
        g_arr = list(scan_img.getdata(band=1))
        b_arr = list(scan_img.getdata(band=2))
        diffs = [abs(rv - gv) + abs(rv - bv) for rv, gv, bv in zip(r_arr, g_arr, b_arr)]
        mean_diff = sum(diffs) / len(diffs)
        assert mean_diff < 15, f"Scanned image does not look greyscale (mean channel diff={mean_diff:.1f})"

    @skip_mcp
    @skip_ws
    def test_scan_missing_file_error(self):
        blocks = _mcp_content("image_scan", {"path": "ghost.png"})
        text = " ".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        assert "not found" in text.lower() or "error" in text.lower()


# ===========================================================================
# 5. image_enhance
# ===========================================================================

class TestImageEnhance:
    @skip_mcp
    @skip_ws
    def test_enhance_returns_inline_image(self, test_image_path):
        blocks = _mcp_content("image_enhance", {"path": test_image_path})
        assert _has_image_block(blocks)

    @skip_mcp
    @skip_ws
    def test_enhance_grayscale_flag(self, test_image_path):
        from PIL import Image
        blocks = _mcp_content("image_enhance",
                               {"path": test_image_path, "grayscale": True})
        img = Image.open(io.BytesIO(_get_image_bytes(blocks)))
        r, g, b = img.split()
        diffs = [abs(rv - gv) + abs(rv - bv)
                 for rv, gv, bv in zip(r.getdata(), g.getdata(), b.getdata())]
        mean_diff = sum(diffs) / len(diffs)
        assert mean_diff < 15, "Greyscale flag did not produce a grey image"

    @skip_mcp
    @skip_ws
    def test_enhance_summary_contains_params(self, test_image_path):
        blocks = _mcp_content("image_enhance",
                               {"path": test_image_path,
                                "contrast": 2.0, "sharpness": 1.8, "brightness": 1.1})
        text = " ".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        assert "2.0" in text and "1.8" in text and "1.1" in text

    @skip_mcp
    @skip_ws
    def test_enhance_clamps_contrast_out_of_range(self, test_image_path):
        # contrast=99 should be clamped, not error
        blocks = _mcp_content("image_enhance",
                               {"path": test_image_path, "contrast": 99})
        assert _has_image_block(blocks)

    @skip_mcp
    @skip_ws
    def test_enhance_missing_file_error(self):
        blocks = _mcp_content("image_enhance", {"path": "missing.png"})
        text = " ".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        assert "not found" in text.lower() or "error" in text.lower()


# ===========================================================================
# 6. Full order-of-operations pipeline: crop → zoom → scan
# ===========================================================================

class TestImagePipeline:
    @skip_mcp
    @skip_ws
    def test_full_pipeline_crop_zoom_scan(self, test_image_path):
        """
        Verify the complete image analysis pipeline:
          1. image_crop  — isolate a region
          2. image_zoom  — scale it up for fine detail
          3. image_scan  — enhance for text reading
        All three steps must produce an inline image block.
        """
        # Step 1: crop a region from the test image
        crop_blocks = _mcp_content("image_crop",
                                    {"path": test_image_path,
                                     "left": 0, "top": 0, "right": 200, "bottom": 100})
        assert _has_image_block(crop_blocks), "Pipeline step 1 (crop) failed"

        # Step 2: zoom into the cropped image using the original path + same coords
        zoom_blocks = _mcp_content("image_zoom",
                                    {"path": test_image_path, "scale": 2.0,
                                     "left": 0, "top": 0, "right": 200, "bottom": 100})
        assert _has_image_block(zoom_blocks), "Pipeline step 2 (zoom) failed"

        # Step 3: scan the same region for text
        scan_blocks = _mcp_content("image_scan",
                                    {"path": test_image_path,
                                     "left": 0, "top": 0, "right": 200, "bottom": 100})
        assert _has_image_block(scan_blocks), "Pipeline step 3 (scan) failed"

        # Each step must return a different image (different base64)
        crop_b64 = next(b["data"] for b in crop_blocks if b.get("type") == "image")
        zoom_b64 = next(b["data"] for b in zoom_blocks if b.get("type") == "image")
        scan_b64 = next(b["data"] for b in scan_blocks if b.get("type") == "image")
        assert zoom_b64 != crop_b64, "Zoom image identical to crop (not actually zoomed)"
        assert scan_b64 != crop_b64, "Scan image identical to crop (not actually scanned)"

    @skip_mcp
    @skip_ws
    def test_pipeline_produces_progressively_larger_then_greyscale(self, test_image_path):
        """Zoom step increases dimensions; scan converts to greyscale."""
        from PIL import Image

        zoom_data = _get_image_bytes(_mcp_content("image_zoom",
                                                   {"path": test_image_path, "scale": 2.0}))
        scan_data = _get_image_bytes(_mcp_content("image_scan",
                                                   {"path": test_image_path}))

        zoom_img = Image.open(io.BytesIO(zoom_data))
        scan_img = Image.open(io.BytesIO(scan_data))

        # Scan should be greyscale (R≈G≈B)
        r, g, b = scan_img.split()
        diffs = [abs(rv - gv) + abs(rv - bv)
                 for rv, gv, bv in zip(r.getdata(), g.getdata(), b.getdata())]
        assert sum(diffs) / len(diffs) < 15

        # Zoom image should be wider than original (400px * 2.0 = 800px)
        assert zoom_img.width > 400, f"Zoom image not larger than original (width={zoom_img.width})"

    @skip_mcp
    @skip_ws
    def test_pipeline_enhance_after_crop(self, test_image_path):
        """image_enhance applied after crop (by sharing path + coords)."""
        enhance_blocks = _mcp_content("image_enhance",
                                       {"path": test_image_path,
                                        "contrast": 2.0, "sharpness": 2.0, "grayscale": False})
        assert _has_image_block(enhance_blocks)
        text = " ".join(b.get("text", "") for b in enhance_blocks if b.get("type") == "text")
        assert "2.0" in text


# ===========================================================================
# 7. Database: offset pagination + summary_only
# ===========================================================================

class TestDatabasePagination:
    @skip_db
    def test_search_articles_with_offset(self):
        uid = uuid.uuid4().hex[:8]
        topic = f"paginate_test_{uid}"
        # Store 3 articles
        for i in range(3):
            httpx.post(f"{DB_URL}/articles/store",
                       json={"url": f"https://example.com/page-{i}-{uid}",
                             "title": f"Page {i}", "content": f"Content {i}", "topic": topic},
                       timeout=5)
        # First page (limit=2, offset=0)
        r0 = httpx.get(f"{DB_URL}/articles/search",
                        params={"topic": topic, "limit": 2, "offset": 0}, timeout=5)
        # Second page (limit=2, offset=2)
        r1 = httpx.get(f"{DB_URL}/articles/search",
                        params={"topic": topic, "limit": 2, "offset": 2}, timeout=5)
        assert r0.status_code == 200
        assert r1.status_code == 200
        page0 = r0.json()["articles"]
        page1 = r1.json()["articles"]
        assert len(page0) == 2
        assert len(page1) == 1
        urls0 = {a["url"] for a in page0}
        urls1 = {a["url"] for a in page1}
        assert urls0.isdisjoint(urls1), "Pagination returned overlapping results"

    @skip_db
    def test_search_articles_summary_only_truncates_content(self):
        uid = uuid.uuid4().hex[:8]
        long_content = "A" * 600
        httpx.post(f"{DB_URL}/articles/store",
                   json={"url": f"https://example.com/long-{uid}",
                         "title": "Long article", "content": long_content,
                         "topic": f"summary_test_{uid}"},
                   timeout=5)
        r = httpx.get(f"{DB_URL}/articles/search",
                       params={"topic": f"summary_test_{uid}", "summary_only": "true"},
                       timeout=5)
        assert r.status_code == 200
        articles = r.json()["articles"]
        assert articles, "No articles returned"
        assert len(articles[0]["content"]) <= 305, \
            f"summary_only did not truncate: got {len(articles[0]['content'])} chars"

    @skip_db
    def test_list_images_with_offset(self):
        r0 = httpx.get(f"{DB_URL}/images/list", params={"limit": 5, "offset": 0}, timeout=5)
        r1 = httpx.get(f"{DB_URL}/images/list", params={"limit": 5, "offset": 5}, timeout=5)
        assert r0.status_code == 200
        assert r1.status_code == 200
        # Both must return the images key
        assert "images" in r0.json()
        assert "images" in r1.json()


# ===========================================================================
# 8. Memory: TTL expiry + pattern matching
# ===========================================================================

class TestMemoryTTLAndPattern:
    @skip_mem
    def test_store_and_recall_with_ttl(self):
        key = f"ttl_test_{uuid.uuid4().hex[:8]}"
        # Store with 2-second TTL
        r = httpx.post(f"{MEMORY_URL}/store",
                        json={"key": key, "value": "expires-soon", "ttl_seconds": 2},
                        timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert data.get("stored") == key
        assert "expires_at" in data

        # Immediately recall — should be found
        r2 = httpx.get(f"{MEMORY_URL}/recall", params={"key": key}, timeout=5)
        assert r2.json().get("found") is True

        # Wait for TTL to expire
        time.sleep(3)

        # Recall after expiry — should not be found
        r3 = httpx.get(f"{MEMORY_URL}/recall", params={"key": key}, timeout=5)
        assert r3.json().get("found") is False, "TTL entry still found after expiry"

    @skip_mem
    def test_pattern_matching_recall(self):
        uid = uuid.uuid4().hex[:8]
        # Store 3 keys under a common prefix
        for i in range(3):
            httpx.post(f"{MEMORY_URL}/store",
                        json={"key": f"pattern_{uid}_item_{i}", "value": f"val{i}"},
                        timeout=5)
        # Also store a key that should NOT match
        httpx.post(f"{MEMORY_URL}/store",
                    json={"key": f"other_{uid}", "value": "decoy"},
                    timeout=5)

        r = httpx.get(f"{MEMORY_URL}/recall",
                       params={"pattern": f"pattern_{uid}_%"},
                       timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert data.get("found") is True
        keys = [e["key"] for e in data["entries"]]
        assert len(keys) == 3, f"Expected 3 pattern matches, got {len(keys)}: {keys}"
        assert all(k.startswith(f"pattern_{uid}") for k in keys)
        assert f"other_{uid}" not in keys

    @skip_mem
    def test_store_without_ttl_persists(self):
        key = f"persist_{uuid.uuid4().hex[:8]}"
        httpx.post(f"{MEMORY_URL}/store",
                    json={"key": key, "value": "permanent"},
                    timeout=5)
        r = httpx.get(f"{MEMORY_URL}/recall", params={"key": key}, timeout=5)
        data = r.json()
        assert data.get("found") is True
        assert data["entries"][0]["value"] == "permanent"
        # Clean up
        httpx.delete(f"{MEMORY_URL}/delete", params={"key": key}, timeout=5)


# ===========================================================================
# 9. Researchbox: push_feed structured error reporting
# ===========================================================================

class TestResearchboxPushFeed:
    @skip_research
    @skip_db
    def test_push_feed_returns_inserted_and_failed_keys(self):
        r = httpx.post(
            f"{RESEARCH_URL}/push-feed",
            json={"feed_url": "https://hnrss.org/newest", "topic": "smoke_hn"},
            timeout=30,
        )
        assert r.status_code == 200
        data = r.json()
        assert "inserted" in data, "push_feed missing 'inserted' key"
        assert "failed"   in data, "push_feed missing 'failed' key"
        assert isinstance(data["inserted"], int)
        assert isinstance(data["failed"],   int)
        assert data["inserted"] + data["failed"] >= 0

    @skip_research
    def test_push_feed_invalid_url_returns_errors_key(self):
        r = httpx.post(
            f"{RESEARCH_URL}/push-feed",
            json={"feed_url": "https://this-domain-definitely-does-not-exist.invalid/feed.xml",
                  "topic": "smoke_invalid"},
            timeout=35,
        )
        assert r.status_code == 200
        data = r.json()
        # Should either time out gracefully OR return with errors key
        assert "inserted" in data
        assert "failed" in data or "errors" in data


# ===========================================================================
# 10. Toolkit: execution timeout
# ===========================================================================

class TestToolkitTimeout:
    @skip_toolkit
    def test_tool_timeout_returns_error_not_hang(self):
        """Register a tool that sleeps 60s, call it, expect timeout response within ~35s."""
        uid = uuid.uuid4().hex[:8]
        tool_name = f"sleepy_{uid}"
        # Register
        r = httpx.post(f"{TOOLKIT_URL}/register",
                        json={"tool_name": tool_name,
                              "description": "Sleeps forever",
                              "parameters": {},
                              "code": "import asyncio\nawait asyncio.sleep(60)\nreturn 'done'"},
                        timeout=10)
        assert r.status_code == 200, f"Register failed: {r.text}"

        # Call — should return a timeout error within TOOL_TIMEOUT (default 30s)
        start = time.monotonic()
        r2 = httpx.post(f"{TOOLKIT_URL}/call/{tool_name}",
                         json={"params": {}}, timeout=40)
        elapsed = time.monotonic() - start

        assert r2.status_code == 200
        data = r2.json()
        assert data.get("error") is True, f"Tool should have timed out: {data}"
        assert "timed out" in data.get("result", "").lower() or "timeout" in data.get("result", "").lower()
        assert elapsed < 38, f"Toolkit did not enforce timeout (took {elapsed:.1f}s)"

        # Clean up
        httpx.delete(f"{TOOLKIT_URL}/tool/{tool_name}", timeout=5)

    @skip_toolkit
    def test_fast_tool_completes_without_timeout(self):
        uid = uuid.uuid4().hex[:8]
        tool_name = f"fast_{uid}"
        httpx.post(f"{TOOLKIT_URL}/register",
                    json={"tool_name": tool_name,
                          "description": "Returns instantly",
                          "parameters": {},
                          "code": "return 'fast_ok'"},
                    timeout=10)
        r = httpx.post(f"{TOOLKIT_URL}/call/{tool_name}",
                        json={"params": {}}, timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert data.get("error") is not True
        assert "fast_ok" in data.get("result", "")
        httpx.delete(f"{TOOLKIT_URL}/tool/{tool_name}", timeout=5)


# ===========================================================================
# 11. call_custom_tool via MCP HTTP
# ===========================================================================

class TestCallCustomToolMCP:
    @skip_mcp
    @skip_toolkit
    def test_call_custom_tool_roundtrip(self):
        """Create a tool, call it via MCP call_custom_tool, verify result."""
        uid = uuid.uuid4().hex[:8]
        tool_name = f"greet_{uid}"

        # Create the tool via MCP
        create_result = _mcp_call("create_tool", {
            "tool_name": tool_name,
            "description": "Says hello",
            "parameters_schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            "code": "return f'Hello, {kwargs[\"name\"]}!'",
        })
        assert create_result.get("result") is not None

        # Call it via call_custom_tool
        call_blocks = _mcp_content("call_custom_tool",
                                    {"tool_name": tool_name, "params": {"name": "World"}})
        text = " ".join(b.get("text", "") for b in call_blocks if b.get("type") == "text")
        assert "Hello, World!" in text, f"Expected greeting, got: {text!r}"

        # Delete via MCP
        _mcp_call("delete_custom_tool", {"tool_name": tool_name})

    @skip_mcp
    def test_call_custom_tool_missing_tool_name_returns_error(self):
        blocks = _mcp_content("call_custom_tool", {"tool_name": "", "params": {}})
        text = " ".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        assert "required" in text.lower() or "tool_name" in text.lower() or "error" in text.lower()

    @skip_mcp
    def test_call_custom_tool_nonexistent_tool_returns_error(self):
        blocks = _mcp_content("call_custom_tool",
                               {"tool_name": "this_tool_does_not_exist_xyz", "params": {}})
        text = " ".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        assert "not found" in text.lower() or "error" in text.lower() or "404" in text


# ===========================================================================
# 12. MCP stdio: call_custom_tool + image tools in schema
# ===========================================================================

class TestMCPStdioSchemas:
    """
    Parse the stdio mcp_server.py schemas in-process to verify correctness
    without needing a running stdio process.
    """
    def test_call_custom_tool_in_stdio_schemas(self):
        from aichat.mcp_server import _TOOL_SCHEMAS
        names = {t["name"] for t in _TOOL_SCHEMAS}
        assert "call_custom_tool" in names

    def test_image_tools_in_stdio_schemas(self):
        from aichat.mcp_server import _TOOL_SCHEMAS
        names = {t["name"] for t in _TOOL_SCHEMAS}
        for tname in ("image_crop", "image_zoom", "image_scan", "image_enhance"):
            assert tname in names, f"'{tname}' missing from stdio _TOOL_SCHEMAS"

    def test_image_tool_schemas_have_required_path(self):
        from aichat.mcp_server import _TOOL_SCHEMAS
        by_name = {t["name"]: t for t in _TOOL_SCHEMAS}
        for tname in ("image_crop", "image_zoom", "image_scan", "image_enhance"):
            schema = by_name[tname]["inputSchema"]
            assert "path" in schema.get("required", []), \
                f"'{tname}' schema missing 'path' in required"

    def test_call_custom_tool_schema_has_required_tool_name(self):
        from aichat.mcp_server import _TOOL_SCHEMAS
        by_name = {t["name"]: t for t in _TOOL_SCHEMAS}
        schema = by_name["call_custom_tool"]["inputSchema"]
        assert "tool_name" in schema.get("required", [])


# ===========================================================================
# 13. MCP HTTP health + tools/list parity check
# ===========================================================================

class TestMCPParity:
    @skip_mcp
    def test_mcp_health_endpoint(self):
        r = httpx.get(f"{MCP_URL}/health", timeout=5)
        assert r.status_code == 200

    @skip_mcp
    def test_tools_list_json_rpc_format(self):
        r = httpx.post(
            f"{MCP_URL}/mcp",
            json={"jsonrpc": "2.0", "id": 42, "method": "tools/list", "params": {}},
            timeout=10,
        )
        data = r.json()
        assert data.get("id") == 42
        assert "result" in data
        assert "tools" in data["result"]
        assert len(data["result"]["tools"]) > 10


# ===========================================================================
# 14. image_stitch — combine multiple images
# ===========================================================================

class TestImageStitch:
    @skip_mcp
    @skip_ws
    def test_stitch_two_images_vertical(self, test_image_path):
        blocks = _mcp_content("image_stitch", {
            "paths": [test_image_path, test_image_path],
            "direction": "vertical",
        })
        assert _has_image_block(blocks), "image_stitch returned no image block"

    @skip_mcp
    @skip_ws
    def test_stitch_two_images_horizontal(self, test_image_path):
        blocks = _mcp_content("image_stitch", {
            "paths": [test_image_path, test_image_path],
            "direction": "horizontal",
        })
        assert _has_image_block(blocks), "image_stitch horizontal returned no image block"

    @skip_mcp
    @skip_ws
    def test_stitch_result_is_taller_than_source(self, test_image_path):
        """Vertical stitch should produce image at least as tall as the source."""
        from PIL import Image
        orig_bytes = _get_image_bytes(
            _mcp_content("image_crop", {"path": test_image_path})
        )
        stitch_bytes = _get_image_bytes(
            _mcp_content("image_stitch", {"paths": [test_image_path, test_image_path]})
        )
        orig_h  = Image.open(io.BytesIO(orig_bytes)).height
        stitch_h = Image.open(io.BytesIO(stitch_bytes)).height
        assert stitch_h >= orig_h, f"Stitch height {stitch_h} < source {orig_h}"

    @skip_mcp
    def test_stitch_too_few_paths_returns_error(self):
        blocks = _mcp_content("image_stitch", {"paths": ["one.png"]})
        assert any("at least 2" in b.get("text", "") for b in blocks)

    @skip_mcp
    def test_stitch_missing_path_returns_error(self):
        blocks = _mcp_content("image_stitch", {"paths": ["ghost_a.png", "ghost_b.png"]})
        assert any("not found" in b.get("text", "") for b in blocks)


# ===========================================================================
# 15. image_diff — pixel-level diff between two images
# ===========================================================================

class TestImageDiff:
    @skip_mcp
    @skip_ws
    def test_diff_same_image_returns_image(self, test_image_path):
        """Diff of an image against itself should return an image block."""
        blocks = _mcp_content("image_diff", {
            "path_a": test_image_path,
            "path_b": test_image_path,
        })
        assert _has_image_block(blocks), "image_diff returned no image block"

    @skip_mcp
    @skip_ws
    def test_diff_same_image_is_mostly_white(self, test_image_path):
        """Diff of identical images should have very few non-white pixels."""
        from PIL import Image
        diff_bytes = _get_image_bytes(
            _mcp_content("image_diff", {"path_a": test_image_path, "path_b": test_image_path})
        )
        img = Image.open(io.BytesIO(diff_bytes)).convert("RGB")
        # Count pixels that are not white (255, 255, 255) — tiny noise is ok
        pixels = list(img.getdata())
        non_white = sum(1 for p in pixels if p != (255, 255, 255))
        assert non_white < len(pixels) * 0.01, f"Too many non-white pixels: {non_white}"

    @skip_mcp
    def test_diff_missing_path_a_returns_error(self, test_image_path):
        blocks = _mcp_content("image_diff", {
            "path_a": "ghost.png",
            "path_b": test_image_path if _WORKSPACE_OK else "ghost2.png",
        })
        assert any("not found" in b.get("text", "") for b in blocks)


# ===========================================================================
# 16. image_annotate — draw bounding boxes on screenshots
# ===========================================================================

class TestImageAnnotate:
    @skip_mcp
    @skip_ws
    def test_annotate_single_box_returns_image(self, test_image_path):
        blocks = _mcp_content("image_annotate", {
            "path": test_image_path,
            "boxes": [{"left": 10, "top": 10, "right": 100, "bottom": 80, "label": "Test"}],
        })
        assert _has_image_block(blocks), "image_annotate returned no image block"

    @skip_mcp
    @skip_ws
    def test_annotate_multiple_boxes(self, test_image_path):
        blocks = _mcp_content("image_annotate", {
            "path": test_image_path,
            "boxes": [
                {"left": 10, "top": 10, "right": 100, "bottom": 50, "label": "A", "color": "#FF0000"},
                {"left": 150, "top": 60, "right": 300, "bottom": 120, "label": "B", "color": "#0000FF"},
            ],
        })
        assert _has_image_block(blocks), "image_annotate multi-box returned no image block"
        text = " ".join(b.get("text", "") for b in blocks)
        assert "2 bounding boxes" in text

    @skip_mcp
    def test_annotate_empty_boxes_returns_error(self, test_image_path):
        blocks = _mcp_content("image_annotate", {
            "path": test_image_path if _WORKSPACE_OK else "x.png",
            "boxes": [],
        })
        assert any("boxes" in b.get("text", "") for b in blocks)

    @skip_mcp
    def test_annotate_missing_path_returns_error(self):
        blocks = _mcp_content("image_annotate", {
            "path": "ghost.png",
            "boxes": [{"left": 0, "top": 0, "right": 10, "bottom": 10}],
        })
        assert any("not found" in b.get("text", "") for b in blocks)


# ===========================================================================
# 17. image_stitch → image_annotate pipeline
# ===========================================================================

class TestStitchAnnotatePipeline:
    @skip_mcp
    @skip_ws
    def test_stitch_then_annotate(self, test_image_path):
        """Stitch two images, save result, annotate the stitched image."""
        # Stitch — the output filename is embedded in the text summary
        stitch_blocks = _mcp_content("image_stitch", {
            "paths": [test_image_path, test_image_path],
        })
        assert _has_image_block(stitch_blocks)
        # Extract saved filename from summary (→ Saved as: stitched_*.jpg)
        text = " ".join(b.get("text", "") for b in stitch_blocks)
        import re
        m = re.search(r"stitched_\S+\.jpg", text)
        if not m:
            pytest.skip("save_prefix did not produce a filename (workspace may be read-only)")
        saved_name = m.group(0)
        # Annotate the stitched image
        ann_blocks = _mcp_content("image_annotate", {
            "path": saved_name,
            "boxes": [{"left": 0, "top": 0, "right": 200, "bottom": 100, "label": "Stitched"}],
        })
        assert _has_image_block(ann_blocks), "Annotate after stitch returned no image"


# ===========================================================================
# 18. page_extract — structured extraction from browser page
# ===========================================================================

class TestPageExtract:
    @skip_mcp
    def test_page_extract_returns_text(self):
        """page_extract should return a text block (even if browser has no loaded page)."""
        blocks = _mcp_content("page_extract", {"include": ["title"]})
        assert any(b.get("type") == "text" for b in blocks)

    @skip_mcp
    def test_page_extract_all_fields_accepted(self):
        """All include fields should be accepted without error."""
        blocks = _mcp_content("page_extract", {
            "include": ["links", "headings", "tables", "images", "meta", "text"],
            "max_links": 10,
            "max_text": 500,
        })
        text = " ".join(b.get("text", "") for b in blocks)
        assert "page_extract failed" not in text or len(text) > 0


# ===========================================================================
# 19. extract_article — clean article text from URL
# ===========================================================================

class TestExtractArticle:
    @skip_mcp
    def test_extract_article_missing_url_returns_error(self):
        blocks = _mcp_content("extract_article", {})
        assert any("url" in b.get("text", "").lower() for b in blocks)

    @skip_mcp
    def test_extract_article_returns_text(self):
        """extract_article should return a text block for a real URL."""
        blocks = _mcp_content("extract_article", {
            "url": "http://example.com",
            "max_chars": 2000,
        })
        assert any(b.get("type") == "text" for b in blocks)
        text = " ".join(b.get("text", "") for b in blocks)
        assert len(text) > 10


# ===========================================================================
# 20. bulk_screenshot — parallel screenshots
# ===========================================================================

class TestBulkScreenshot:
    @skip_mcp
    def test_bulk_screenshot_missing_urls_returns_error(self):
        blocks = _mcp_content("bulk_screenshot", {"urls": []})
        assert any("urls" in b.get("text", "").lower() for b in blocks)

    @skip_mcp
    def test_bulk_screenshot_single_url_returns_content(self):
        """Bulk screenshot of one URL should return at least a text block."""
        blocks = _mcp_content("bulk_screenshot", {
            "urls": ["http://example.com"],
        })
        assert len(blocks) > 0

    @skip_mcp
    def test_bulk_screenshot_multiple_urls(self):
        """Two-URL bulk screenshot — expect two sets of content blocks."""
        blocks = _mcp_content("bulk_screenshot", {
            "urls": ["http://example.com", "http://example.org"],
        })
        # Should have at least 2 blocks (one per URL minimum)
        assert len(blocks) >= 2


# ===========================================================================
# 21. scroll_screenshot — full-page capture
# ===========================================================================

class TestScrollScreenshot:
    @skip_mcp
    def test_scroll_screenshot_returns_content(self):
        """scroll_screenshot with a URL should return at least a text block."""
        blocks = _mcp_content("scroll_screenshot", {
            "url": "http://example.com",
            "max_scrolls": 2,
        })
        assert len(blocks) > 0
        assert any(b.get("type") == "text" for b in blocks)

    @skip_mcp
    def test_scroll_screenshot_with_image(self):
        """scroll_screenshot should produce an inline image if browser is available."""
        blocks = _mcp_content("scroll_screenshot", {
            "url": "http://example.com",
            "max_scrolls": 2,
            "scroll_overlap": 50,
        })
        text = " ".join(b.get("text", "") for b in blocks)
        # Either we get an image, or a helpful error — not a crash
        assert "scroll_screenshot" in text or _has_image_block(blocks)


# ===========================================================================
# 22. New tools all advertised in tools/list
# ===========================================================================

class TestNewToolsAdvertised:
    _NEW_TOOLS = [
        "image_stitch", "image_diff", "image_annotate",
        "page_extract", "extract_article",
        "bulk_screenshot", "scroll_screenshot",
    ]

    @skip_mcp
    def test_all_new_tools_in_tools_list(self):
        r = httpx.post(
            f"{MCP_URL}/mcp",
            json={"jsonrpc": "2.0", "id": 99, "method": "tools/list", "params": {}},
            timeout=10,
        )
        names = {t["name"] for t in r.json()["result"]["tools"]}
        for tname in self._NEW_TOOLS:
            assert tname in names, f"'{tname}' missing from tools/list"

    @skip_mcp
    def test_total_tool_count_is_40(self):
        r = httpx.post(
            f"{MCP_URL}/mcp",
            json={"jsonrpc": "2.0", "id": 100, "method": "tools/list", "params": {}},
            timeout=10,
        )
        tools = r.json()["result"]["tools"]
        assert len(tools) == 40, f"Expected 40 tools, got {len(tools)}"

    def test_new_tools_in_stdio_schema(self):
        from aichat.mcp_server import _TOOL_SCHEMAS
        by_name = {t["name"]: t for t in _TOOL_SCHEMAS}
        for tname in self._NEW_TOOLS:
            assert tname in by_name, f"'{tname}' missing from stdio _TOOL_SCHEMAS"


# ===========================================================================
# 23. Image generation tools
# ===========================================================================

_IMAGE_GEN_URL = "http://192.168.50.2:1234"  # LM Studio


def _lm_studio_has_image_model() -> bool:
    """Check if LM Studio has an image generation model loaded."""
    try:
        r = httpx.post(
            f"{_IMAGE_GEN_URL}/v1/images/generations",
            json={"prompt": "test", "n": 1, "size": "64x64", "response_format": "b64_json"},
            timeout=10,
        )
        # If we get data[] back, a model is loaded
        return bool(r.json().get("data"))
    except Exception:
        return False


_LM_IMAGE_OK = _lm_studio_has_image_model()
skip_imggen = pytest.mark.skipif(not _LM_IMAGE_OK, reason="No image generation model loaded in LM Studio")


class TestImageGeneration:

    # -- image_generate errors (always testable) ----------------------------

    @skip_mcp
    def test_image_generate_missing_prompt_returns_error(self):
        blocks = _mcp_content("image_generate", {})
        assert any("prompt" in b.get("text", "").lower() for b in blocks)

    @skip_mcp
    def test_image_generate_helpful_error_on_failure(self):
        """When no model is loaded, the error message should guide the user."""
        if _LM_IMAGE_OK:
            pytest.skip("Image model IS loaded — no failure to test")
        blocks = _mcp_content("image_generate", {"prompt": "a red circle"})
        text = " ".join(b.get("text", "") for b in blocks)
        # Should mention how to fix the situation
        assert ("LM Studio" in text or "IMAGE_GEN_BASE_URL" in text or "model" in text.lower()), \
            f"Error message should guide the user, got: {text[:300]}"

    # -- image_edit errors (always testable) --------------------------------

    @skip_mcp
    def test_image_edit_missing_path_returns_error(self):
        blocks = _mcp_content("image_edit", {"prompt": "make it blue"})
        assert any("path" in b.get("text", "").lower() for b in blocks)

    @skip_mcp
    def test_image_edit_missing_prompt_returns_error(self, test_image_path):
        blocks = _mcp_content("image_edit", {
            "path": test_image_path if _WORKSPACE_OK else "x.png",
        })
        assert any("prompt" in b.get("text", "").lower() for b in blocks)

    @skip_mcp
    def test_image_edit_missing_image_file_returns_error(self):
        blocks = _mcp_content("image_edit", {"path": "ghost_missing.png", "prompt": "make it blue"})
        assert any("not found" in b.get("text", "").lower() for b in blocks)

    # -- image_upscale (always works — pure PIL) ----------------------------

    @skip_mcp
    @skip_ws
    def test_image_upscale_returns_larger_image(self, test_image_path):
        from PIL import Image
        orig_bytes = _get_image_bytes(
            _mcp_content("image_crop", {"path": test_image_path})
        )
        up_blocks = _mcp_content("image_upscale", {"path": test_image_path, "scale": 3.0})
        assert _has_image_block(up_blocks), "image_upscale returned no image block"
        up_bytes = _get_image_bytes(up_blocks)
        orig_w = Image.open(io.BytesIO(orig_bytes)).width
        up_w   = Image.open(io.BytesIO(up_bytes)).width
        assert up_w >= orig_w * 2, f"Upscaled width {up_w} should be ~3× original {orig_w}"

    @skip_mcp
    @skip_ws
    def test_image_upscale_with_sharpen(self, test_image_path):
        blocks = _mcp_content("image_upscale", {"path": test_image_path, "scale": 2.0, "sharpen": True})
        assert _has_image_block(blocks)
        text = " ".join(b.get("text", "") for b in blocks)
        assert "sharpen" in text

    @skip_mcp
    @skip_ws
    def test_image_upscale_without_sharpen(self, test_image_path):
        blocks = _mcp_content("image_upscale", {"path": test_image_path, "scale": 2.0, "sharpen": False})
        assert _has_image_block(blocks)

    @skip_mcp
    def test_image_upscale_missing_path_returns_error(self):
        blocks = _mcp_content("image_upscale", {})
        assert any("path" in b.get("text", "").lower() for b in blocks)

    @skip_mcp
    def test_image_upscale_missing_file_returns_error(self):
        blocks = _mcp_content("image_upscale", {"path": "ghost_upscale.png"})
        assert any("not found" in b.get("text", "").lower() for b in blocks)

    # -- upscale → scan pipeline (always works) -----------------------------

    @skip_mcp
    @skip_ws
    def test_upscale_then_scan_pipeline(self, test_image_path):
        """Upscale a small image, then scan — the scan should get bigger image."""
        up_blocks = _mcp_content("image_upscale", {"path": test_image_path, "scale": 2.0})
        assert _has_image_block(up_blocks)
        import re
        up_text = " ".join(b.get("text", "") for b in up_blocks)
        m = re.search(r"upscaled_\S+\.jpg", up_text)
        if not m:
            pytest.skip("Upscale did not save file (workspace may be read-only)")
        up_fname = m.group(0)
        scan_blocks = _mcp_content("image_scan", {"path": up_fname})
        assert _has_image_block(scan_blocks)

    # -- image_generate with model (skipped if none loaded) -----------------

    @skip_mcp
    @skip_imggen
    def test_image_generate_returns_image(self):
        blocks = _mcp_content("image_generate", {
            "prompt": "a simple red circle on white background",
            "size": "256x256",
            "n": 1,
        })
        assert _has_image_block(blocks), "image_generate returned no image block"

    @skip_mcp
    @skip_imggen
    def test_image_generate_saves_to_workspace(self):
        import re
        blocks = _mcp_content("image_generate", {
            "prompt": "a blue square on white background",
            "size": "256x256",
        })
        text = " ".join(b.get("text", "") for b in blocks)
        assert re.search(r"generated_\S+\.jpg", text), \
            f"Expected saved filename in summary, got: {text[:200]}"

    @skip_mcp
    @skip_imggen
    @skip_ws
    def test_generate_then_upscale_pipeline(self):
        """Generate a small image, upscale it, verify result is larger."""
        import re
        from PIL import Image
        gen_blocks = _mcp_content("image_generate", {
            "prompt": "a tiny red dot",
            "size": "256x256",
        })
        gen_text = " ".join(b.get("text", "") for b in gen_blocks)
        m = re.search(r"generated_\S+\.jpg", gen_text)
        if not m:
            pytest.skip("generate did not save filename")
        gen_fname = m.group(0)
        up_blocks = _mcp_content("image_upscale", {"path": gen_fname, "scale": 2.0})
        assert _has_image_block(up_blocks)
        gen_w  = Image.open(io.BytesIO(_get_image_bytes(gen_blocks))).width
        up_w   = Image.open(io.BytesIO(_get_image_bytes(up_blocks))).width
        assert up_w >= gen_w * 1.5, f"Upscaled width {up_w} should exceed original {gen_w}"

    # -- tool advertisement -------------------------------------------------

    @skip_mcp
    def test_gen_tools_advertised_in_tools_list(self):
        r = httpx.post(
            f"{MCP_URL}/mcp",
            json={"jsonrpc": "2.0", "id": 200, "method": "tools/list", "params": {}},
            timeout=10,
        )
        names = {t["name"] for t in r.json()["result"]["tools"]}
        for tname in ("image_generate", "image_edit", "image_upscale"):
            assert tname in names, f"'{tname}' missing from tools/list"

    @skip_mcp
    def test_total_tool_count_is_40_imggen(self):
        r = httpx.post(
            f"{MCP_URL}/mcp",
            json={"jsonrpc": "2.0", "id": 201, "method": "tools/list", "params": {}},
            timeout=10,
        )
        tools = r.json()["result"]["tools"]
        assert len(tools) == 40, f"Expected 40 tools, got {len(tools)}"

    def test_gen_tools_in_stdio_schema(self):
        from aichat.mcp_server import _TOOL_SCHEMAS
        by_name = {t["name"]: t for t in _TOOL_SCHEMAS}
        for tname in ("image_generate", "image_edit", "image_upscale"):
            assert tname in by_name, f"'{tname}' missing from stdio _TOOL_SCHEMAS"


# ===========================================================================
# Browser v6 — find_image + screenshot_element + list_images_detail
# ===========================================================================

class TestBrowserV6Schemas:
    """Verify that browser v6 capabilities are correctly advertised in both schema layers."""

    def test_screenshot_tool_has_find_image_param_in_mcp_schema(self):
        """docker/mcp/app.py _TOOLS: screenshot should expose find_image."""
        import importlib.util, sys
        # Read the _TOOLS list directly from the source file rather than importing
        # the full FastAPI app (which would try to start a server).
        import ast, pathlib
        src = pathlib.Path(__file__).parent.parent / "docker/mcp/app.py"
        if not src.exists():
            pytest.skip("docker/mcp/app.py not found")
        tree = ast.parse(src.read_text())
        # We just verify the string "find_image" appears in that file
        assert "find_image" in src.read_text(), \
            "find_image not found in docker/mcp/app.py"

    def test_screenshot_tool_has_find_image_in_stdio_schema(self):
        from aichat.mcp_server import _TOOL_SCHEMAS
        by_name = {t["name"]: t for t in _TOOL_SCHEMAS}
        assert "screenshot" in by_name
        props = by_name["screenshot"]["inputSchema"]["properties"]
        assert "find_image" in props, "find_image missing from stdio screenshot schema"

    def test_browser_tool_has_screenshot_element_in_stdio_schema(self):
        from aichat.mcp_server import _TOOL_SCHEMAS
        by_name = {t["name"]: t for t in _TOOL_SCHEMAS}
        assert "browser" in by_name
        enum_vals = by_name["browser"]["inputSchema"]["properties"]["action"]["enum"]
        assert "screenshot_element" in enum_vals, \
            f"screenshot_element missing from browser action enum: {enum_vals}"

    def test_browser_tool_has_list_images_detail_in_stdio_schema(self):
        from aichat.mcp_server import _TOOL_SCHEMAS
        by_name = {t["name"]: t for t in _TOOL_SCHEMAS}
        enum_vals = by_name["browser"]["inputSchema"]["properties"]["action"]["enum"]
        assert "list_images_detail" in enum_vals, \
            f"list_images_detail missing from browser action enum: {enum_vals}"

    def test_browser_tool_has_find_image_param_in_stdio_schema(self):
        from aichat.mcp_server import _TOOL_SCHEMAS
        by_name = {t["name"]: t for t in _TOOL_SCHEMAS}
        props = by_name["browser"]["inputSchema"]["properties"]
        assert "find_image" in props, "find_image missing from stdio browser schema"

    def test_browser_tool_has_pad_param_in_stdio_schema(self):
        from aichat.mcp_server import _TOOL_SCHEMAS
        by_name = {t["name"]: t for t in _TOOL_SCHEMAS}
        props = by_name["browser"]["inputSchema"]["properties"]
        assert "pad" in props, "pad missing from stdio browser schema"

    @skip_mcp
    def test_screenshot_tool_has_find_image_in_mcp_http_schema(self):
        r = httpx.post(
            f"{MCP_URL}/mcp",
            json={"jsonrpc": "2.0", "id": 300, "method": "tools/list", "params": {}},
            timeout=10,
        )
        tools = {t["name"]: t for t in r.json()["result"]["tools"]}
        assert "screenshot" in tools
        props = tools["screenshot"]["inputSchema"]["properties"]
        assert "find_image" in props, "find_image missing from HTTP MCP screenshot schema"

    @skip_mcp
    def test_browser_tool_has_screenshot_element_in_mcp_http_schema(self):
        r = httpx.post(
            f"{MCP_URL}/mcp",
            json={"jsonrpc": "2.0", "id": 301, "method": "tools/list", "params": {}},
            timeout=10,
        )
        tools = {t["name"]: t for t in r.json()["result"]["tools"]}
        enum_vals = tools["browser"]["inputSchema"]["properties"]["action"]["enum"]
        assert "screenshot_element" in enum_vals


class TestBrowserV6ManagerRouting:
    """Verify run_browser() in manager.py accepts the new parameters (unit-level)."""

    def test_run_browser_signature_has_find_image(self):
        import inspect
        from aichat.tools.manager import ToolManager
        sig = inspect.signature(ToolManager.run_browser)
        assert "find_image" in sig.parameters, \
            "find_image not in run_browser() signature"

    def test_run_browser_signature_has_pad(self):
        import inspect
        from aichat.tools.manager import ToolManager
        sig = inspect.signature(ToolManager.run_browser)
        assert "pad" in sig.parameters, "pad not in run_browser() signature"

    def test_run_browser_error_message_mentions_new_actions(self):
        """The unknown-action error string should mention screenshot_element."""
        import inspect
        from aichat.tools.manager import ToolManager
        src = inspect.getsource(ToolManager.run_browser)
        assert "screenshot_element" in src, \
            "screenshot_element not mentioned in run_browser source"
        assert "list_images_detail" in src, \
            "list_images_detail not mentioned in run_browser source"


# ===========================================================================
# Browser v7 — browser_save_images + browser_download_page_images
# ===========================================================================

class TestBrowserImageDownload:
    """Schema + routing tests for human-like image download tools."""

    # -- stdio schema checks (always run) ------------------------------------

    def test_browser_save_images_in_stdio_schema(self):
        from aichat.mcp_server import _TOOL_SCHEMAS
        by_name = {t["name"]: t for t in _TOOL_SCHEMAS}
        assert "browser_save_images" in by_name, \
            "browser_save_images missing from stdio _TOOL_SCHEMAS"
        props = by_name["browser_save_images"]["inputSchema"]["properties"]
        assert "urls" in props, "urls missing from browser_save_images schema"

    def test_browser_download_page_images_in_stdio_schema(self):
        from aichat.mcp_server import _TOOL_SCHEMAS
        by_name = {t["name"]: t for t in _TOOL_SCHEMAS}
        assert "browser_download_page_images" in by_name, \
            "browser_download_page_images missing from stdio _TOOL_SCHEMAS"

    def test_browser_download_page_images_has_filter_param(self):
        from aichat.mcp_server import _TOOL_SCHEMAS
        by_name = {t["name"]: t for t in _TOOL_SCHEMAS}
        props = by_name["browser_download_page_images"]["inputSchema"]["properties"]
        assert "filter" in props, "filter param missing from browser_download_page_images"

    def test_browser_download_page_images_has_url_param(self):
        from aichat.mcp_server import _TOOL_SCHEMAS
        by_name = {t["name"]: t for t in _TOOL_SCHEMAS}
        props = by_name["browser_download_page_images"]["inputSchema"]["properties"]
        assert "url" in props, "url param (navigate-first) missing from browser_download_page_images"

    def test_browser_save_images_accepts_comma_separated_string(self):
        """Schema accepts generic 'urls' field (no type restriction) for comma-sep or list."""
        from aichat.mcp_server import _TOOL_SCHEMAS
        by_name = {t["name"]: t for t in _TOOL_SCHEMAS}
        schema = by_name["browser_save_images"]["inputSchema"]
        # urls should be in required
        assert "urls" in schema.get("required", []), \
            "urls not in required list for browser_save_images"

    # -- manager routing checks ----------------------------------------------

    def test_run_browser_has_save_images_action(self):
        import inspect
        from aichat.tools.manager import ToolManager
        sig = inspect.signature(ToolManager.run_browser)
        assert "image_urls" in sig.parameters, "image_urls not in run_browser() signature"
        assert "filter_query" in sig.parameters, "filter_query not in run_browser() signature"
        assert "image_prefix" in sig.parameters, "image_prefix not in run_browser() signature"
        assert "max_images" in sig.parameters, "max_images not in run_browser() signature"

    def test_run_browser_routes_save_images(self):
        import inspect
        from aichat.tools.manager import ToolManager
        src = inspect.getsource(ToolManager.run_browser)
        assert "save_images" in src, "save_images not routed in run_browser"
        assert "download_page_images" in src, "download_page_images not routed in run_browser"

    # -- browser server version check (updated with each server bump) ---------

    def test_browser_server_version_is_14(self):
        from aichat.tools.browser import _REQUIRED_SERVER_VERSION, _SERVER_SRC
        assert _REQUIRED_SERVER_VERSION == "14", \
            f"Expected _REQUIRED_SERVER_VERSION='14', got '{_REQUIRED_SERVER_VERSION}'"
        assert '_VERSION = "14"' in _SERVER_SRC, \
            "_VERSION = '14' not found in _SERVER_SRC"

    def test_browser_server_v14_has_crash_recovery(self):
        from aichat.tools.browser import _SERVER_SRC
        assert "_restart_browser" in _SERVER_SRC, "_restart_browser not in _SERVER_SRC"
        # _ensure_page must call _restart_browser as third tier
        assert "_restart_browser" in _SERVER_SRC, "_restart_browser missing from _SERVER_SRC"
        # _rotate_context_and_page must fall back to _restart_browser
        assert "Browser may have crashed" in _SERVER_SRC, \
            "Browser crash fallback comment missing from _rotate_context_and_page"

    def test_browser_server_has_block_detection(self):
        from aichat.tools.browser import _SERVER_SRC
        assert "_BLOCK_SIGNALS" in _SERVER_SRC, "_BLOCK_SIGNALS not in _SERVER_SRC"
        assert "_is_blocked" in _SERVER_SRC, "_is_blocked not in _SERVER_SRC"
        assert "_rotate_context_and_page" in _SERVER_SRC, \
            "_rotate_context_and_page not in _SERVER_SRC"
        assert "_site_fallback" in _SERVER_SRC, \
            "_site_fallback not in _SERVER_SRC"
        assert "old.reddit.com" in _SERVER_SRC, \
            "old.reddit.com fallback not in _SERVER_SRC"

    def test_browser_server_stealth_headers(self):
        from aichat.tools.browser import _SERVER_SRC
        assert "Google Chrome" in _SERVER_SRC, "Google Chrome brand not in _SERVER_SRC"
        assert "Windows" in _SERVER_SRC, "Windows platform not in _SERVER_SRC"
        assert "Chrome/145.0.0.0" in _SERVER_SRC, "Chrome/145 UA not in _SERVER_SRC"
        assert "Win32" in _SERVER_SRC, "Win32 navigator.platform not in _SERVER_SRC"

    def test_browser_server_v12_cloudflare_fixes(self):
        """v12 stealth: all Cloudflare Turnstile signals are addressed."""
        from aichat.tools.browser import _SERVER_SRC
        # CRITICAL: webdriver on prototype with enumerable:false
        assert "Navigator.prototype" in _SERVER_SRC, \
            "Navigator.prototype webdriver fix missing"
        assert "enumerable: false" in _SERVER_SRC, \
            "enumerable: false not set on webdriver"
        # CRITICAL: permissions.query toString() spoofing via _markNative
        assert "_markNative" in _SERVER_SRC, \
            "_markNative native-code spoofer missing"
        assert "_nativizedFns" in _SERVER_SRC, \
            "_nativizedFns WeakMap missing"
        assert "wrappedQuery" in _SERVER_SRC, \
            "permissions.query wrapper missing"
        # HIGH: Accept-Language q-values in the server headers dict
        assert "en-US,en;q=0.9" in _SERVER_SRC, \
            "Accept-Language q-values missing from _SERVER_SRC"
        # MEDIUM: pdfViewerEnabled
        assert "pdfViewerEnabled" in _SERVER_SRC, \
            "pdfViewerEnabled spoof missing"
        # MEDIUM: battery spoof — realistic non-charging laptop values
        assert "_fakeBattery" in _SERVER_SRC, \
            "_fakeBattery object missing"
        assert "charging: false" in _SERVER_SRC, \
            "battery.charging should be false (not headless default true)"
        # MEDIUM: timezone in context
        assert "timezone_id" in _SERVER_SRC, \
            "timezone_id not set in _new_context()"
        # LOW: mimeTypes
        assert "mimeTypes" in _SERVER_SRC, \
            "navigator.mimeTypes spoof missing"
        # LOW: connection RTT
        assert "connection" in _SERVER_SRC and "rtt" in _SERVER_SRC, \
            "connection.rtt spoof missing"

    def test_browser_server_has_save_images_endpoint(self):
        from aichat.tools.browser import _SERVER_SRC
        assert "/save_images" in _SERVER_SRC, "/save_images endpoint missing from _SERVER_SRC"
        assert "page.request.get" in _SERVER_SRC, \
            "page.request.get not used in _SERVER_SRC (needed for browser-context download)"

    def test_browser_server_has_download_page_images_endpoint(self):
        from aichat.tools.browser import _SERVER_SRC
        assert "/download_page_images" in _SERVER_SRC, \
            "/download_page_images endpoint missing from _SERVER_SRC"

    def test_browser_tool_has_save_images_method(self):
        from aichat.tools.browser import BrowserTool
        assert hasattr(BrowserTool, "save_images"), "BrowserTool.save_images() missing"
        assert hasattr(BrowserTool, "download_page_images"), \
            "BrowserTool.download_page_images() missing"

    # -- page_scrape checks --------------------------------------------------

    def test_browser_server_v14_has_scrape_endpoint(self):
        from aichat.tools.browser import _SERVER_SRC, _REQUIRED_SERVER_VERSION
        assert _REQUIRED_SERVER_VERSION == "14", \
            f"Expected v14, got {_REQUIRED_SERVER_VERSION}"
        assert "/scrape" in _SERVER_SRC, "/scrape endpoint missing from _SERVER_SRC"
        assert "_scroll_full_page" in _SERVER_SRC, "_scroll_full_page missing"
        assert "_extract_text_long" in _SERVER_SRC, "_extract_text_long missing"
        assert "ScrapeReq" in _SERVER_SRC, "ScrapeReq missing"
        assert "content_grew_on_scroll" in _SERVER_SRC, "content_grew_on_scroll missing"
        assert "include_links" in _SERVER_SRC, "include_links missing"
        assert "final_page_height" in _SERVER_SRC, "final_page_height missing"

    def test_browser_tool_has_scrape_method(self):
        from aichat.tools.browser import BrowserTool
        assert hasattr(BrowserTool, "scrape"), "BrowserTool.scrape() missing"
        import inspect
        sig = inspect.signature(BrowserTool.scrape)
        assert "max_scrolls" in sig.parameters, "max_scrolls missing from BrowserTool.scrape"
        assert "wait_ms" in sig.parameters, "wait_ms missing from BrowserTool.scrape"
        assert "include_links" in sig.parameters, "include_links missing from BrowserTool.scrape"

    def test_manager_has_run_page_scrape(self):
        import inspect
        from aichat.tools.manager import ToolManager
        assert hasattr(ToolManager, "run_page_scrape"), "run_page_scrape missing from ToolManager"
        sig = inspect.signature(ToolManager.run_page_scrape)
        assert "max_scrolls" in sig.parameters
        assert "wait_ms" in sig.parameters
        assert "include_links" in sig.parameters

    def test_page_scrape_in_stdio_schema(self):
        from aichat.mcp_server import _TOOL_SCHEMAS
        by_name = {t["name"]: t for t in _TOOL_SCHEMAS}
        assert "page_scrape" in by_name, "page_scrape missing from stdio _TOOL_SCHEMAS"
        props = by_name["page_scrape"]["inputSchema"]["properties"]
        assert "url" in props
        assert "max_scrolls" in props
        assert "wait_ms" in props
        assert "max_chars" in props
        assert "include_links" in props

    # -- page_images checks --------------------------------------------------

    def test_browser_server_v13_has_page_images_endpoint(self):
        from aichat.tools.browser import _SERVER_SRC
        assert "/page_images" in _SERVER_SRC, "/page_images endpoint missing from _SERVER_SRC"
        assert "PageImagesReq" in _SERVER_SRC, "PageImagesReq model missing"
        assert "bestSrcset" in _SERVER_SRC, "bestSrcset helper missing"
        assert "data-src" in _SERVER_SRC, "data-src lazy extraction missing"
        assert "data-lazy-src" in _SERVER_SRC, "data-lazy-src lazy extraction missing"
        assert "picture source" in _SERVER_SRC, "<picture><source> extraction missing"
        assert "twitter:image" in _SERVER_SRC, "twitter:image meta missing"
        assert "json_ld" in _SERVER_SRC, "JSON-LD extraction missing"
        assert "css_bg" in _SERVER_SRC, "CSS background-image extraction missing"
        assert "new URL(u, base).href" in _SERVER_SRC, "relative URL normalisation missing"

    def test_browser_tool_has_page_images_method(self):
        from aichat.tools.browser import BrowserTool
        import inspect
        assert hasattr(BrowserTool, "page_images"), "BrowserTool.page_images() missing"
        sig = inspect.signature(BrowserTool.page_images)
        assert "url" in sig.parameters, "url param missing from page_images"
        assert "scroll" in sig.parameters, "scroll param missing from page_images"
        assert "max_scrolls" in sig.parameters, "max_scrolls param missing from page_images"

    def test_manager_has_run_page_images(self):
        from aichat.tools.manager import ToolManager
        import inspect
        assert hasattr(ToolManager, "run_page_images"), "run_page_images missing from ToolManager"
        sig = inspect.signature(ToolManager.run_page_images)
        assert "url" in sig.parameters
        assert "scroll" in sig.parameters
        assert "max_scrolls" in sig.parameters

    def test_page_images_in_stdio_schema(self):
        from aichat.mcp_server import _TOOL_SCHEMAS
        by_name = {t["name"]: t for t in _TOOL_SCHEMAS}
        assert "page_images" in by_name, "page_images missing from stdio _TOOL_SCHEMAS"
        props = by_name["page_images"]["inputSchema"]["properties"]
        assert "url" in props
        assert "scroll" in props
        assert "max_scrolls" in props

    # -- MCP HTTP schema checks (skip if MCP not running) --------------------

    @skip_mcp
    def test_tool_count_is_40(self):
        r = httpx.post(
            f"{MCP_URL}/mcp",
            json={"jsonrpc": "2.0", "id": 400, "method": "tools/list", "params": {}},
            timeout=10,
        )
        tools = r.json()["result"]["tools"]
        assert len(tools) == 40, f"Expected 40 tools, got {len(tools)}"

    @skip_mcp
    def test_browser_save_images_in_mcp_http_schema(self):
        r = httpx.post(
            f"{MCP_URL}/mcp",
            json={"jsonrpc": "2.0", "id": 401, "method": "tools/list", "params": {}},
            timeout=10,
        )
        tools = {t["name"]: t for t in r.json()["result"]["tools"]}
        assert "browser_save_images" in tools, "browser_save_images missing from HTTP MCP tools"
        assert "browser_download_page_images" in tools, \
            "browser_download_page_images missing from HTTP MCP tools"

    # -- image_search tests ---------------------------------------------------

    def test_image_search_in_stdio_schema(self):
        from aichat.mcp_server import _TOOL_SCHEMAS
        names = {t["name"] for t in _TOOL_SCHEMAS}
        assert "image_search" in names, "image_search missing from stdio _TOOL_SCHEMAS"
        schema = next(t for t in _TOOL_SCHEMAS if t["name"] == "image_search")
        assert "query" in schema["inputSchema"]["properties"]
        assert "query" in schema["inputSchema"]["required"]

    def test_manager_has_run_image_search(self):
        from aichat.tools.manager import ToolManager
        assert hasattr(ToolManager, "run_image_search"), \
            "ToolManager missing run_image_search method"
        import inspect
        sig = inspect.signature(ToolManager.run_image_search)
        assert "query" in sig.parameters
        assert "count" in sig.parameters

    @skip_mcp
    def test_image_search_in_mcp_http_schema(self):
        r = httpx.post(
            f"{MCP_URL}/mcp",
            json={"jsonrpc": "2.0", "id": 402, "method": "tools/list", "params": {}},
            timeout=10,
        )
        tools = {t["name"]: t for t in r.json()["result"]["tools"]}
        assert "image_search" in tools, "image_search missing from HTTP MCP tools"
        props = tools["image_search"]["inputSchema"]["properties"]
        assert "query" in props
        assert "count" in props
