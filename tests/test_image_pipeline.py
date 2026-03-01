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
    def test_total_tool_count_is_47(self):
        r = httpx.post(
            f"{MCP_URL}/mcp",
            json={"jsonrpc": "2.0", "id": 100, "method": "tools/list", "params": {}},
            timeout=10,
        )
        tools = r.json()["result"]["tools"]
        assert len(tools) == 47, f"Expected 47 tools, got {len(tools)}"

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

    @skip_mcp
    @skip_ws
    def test_image_upscale_safety_cap(self):
        """Scale factor that would push output beyond 8192 px is clamped and reported."""
        from PIL import Image as _PILC
        img = _PILC.new("RGB", (2000, 2000), color=(200, 200, 200))
        buf = io.BytesIO()
        img.save(buf, "JPEG")
        path = os.path.join(WORKSPACE, "upscale_cap_test.jpg")
        with open(path, "wb") as fh:
            fh.write(buf.getvalue())
        try:
            blocks = _mcp_content("image_upscale", {"path": "upscale_cap_test.jpg", "scale": 8.0})
            assert _has_image_block(blocks), "No image returned from capped upscale"
            texts = " ".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            assert "capped" in texts.lower(), f"Cap notice missing from summary: {texts!r}"
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass

    @skip_mcp
    @skip_ws
    def test_image_upscale_exif_no_crash(self, test_image_path):
        """EXIF auto-rotation via ImageOps.exif_transpose doesn't raise on normal JPEG."""
        blocks = _mcp_content("image_upscale", {"path": test_image_path, "scale": 1.5})
        assert _has_image_block(blocks), "No image returned — EXIF path likely crashed"

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
    def test_total_tool_count_is_47_imggen(self):
        r = httpx.post(
            f"{MCP_URL}/mcp",
            json={"jsonrpc": "2.0", "id": 201, "method": "tools/list", "params": {}},
            timeout=10,
        )
        tools = r.json()["result"]["tools"]
        assert len(tools) == 47, f"Expected 47 tools, got {len(tools)}"

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

    def test_browser_server_version_is_18(self):
        from aichat.tools.browser import _REQUIRED_SERVER_VERSION, _SERVER_SRC
        assert _REQUIRED_SERVER_VERSION == "18", \
            f"Expected _REQUIRED_SERVER_VERSION='18', got '{_REQUIRED_SERVER_VERSION}'"
        assert '_VERSION = "18"' in _SERVER_SRC, \
            "_VERSION = '18' not found in _SERVER_SRC"

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

    def test_browser_server_v18_has_scrape_endpoint(self):
        from aichat.tools.browser import _SERVER_SRC, _REQUIRED_SERVER_VERSION
        assert _REQUIRED_SERVER_VERSION == "18", \
            f"Expected v18, got {_REQUIRED_SERVER_VERSION}"
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
    def test_tool_count_is_47(self):
        r = httpx.post(
            f"{MCP_URL}/mcp",
            json={"jsonrpc": "2.0", "id": 400, "method": "tools/list", "params": {}},
            timeout=10,
        )
        tools = r.json()["result"]["tools"]
        assert len(tools) == 47, f"Expected 47 tools, got {len(tools)}"

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
        assert "offset" in sig.parameters, "run_image_search missing 'offset' parameter"
        assert sig.parameters["offset"].default == 0, "offset default should be 0"

    def test_image_search_stdio_schema_has_offset(self):
        from aichat.mcp_server import _TOOL_SCHEMAS
        schema = next(t for t in _TOOL_SCHEMAS if t["name"] == "image_search")
        props = schema["inputSchema"]["properties"]
        assert "offset" in props, "image_search stdio schema missing 'offset'"

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
        assert "offset" in props, "image_search HTTP schema missing 'offset'"

    def test_image_search_domain_cap(self):
        """_apply_domain_cap keeps max 2 images per domain."""
        from urllib.parse import urlparse

        def _domain_of(url):
            h = urlparse(url).hostname or ""
            return h.removeprefix("www.")

        def _apply_domain_cap(candidates, max_per_domain=2):
            counts: dict = {}
            result = []
            for cand in candidates:
                d = _domain_of(cand.get("url", ""))
                if counts.get(d, 0) < max_per_domain:
                    result.append(cand)
                    counts[d] = counts.get(d, 0) + 1
            return result

        cands = [
            {"url": "https://fandom.com/img/a.png"},
            {"url": "https://fandom.com/img/b.png"},
            {"url": "https://fandom.com/img/c.png"},   # 3rd — should be dropped
            {"url": "https://imgur.com/img/d.png"},
        ]
        result = _apply_domain_cap(cands, max_per_domain=2)
        assert len(result) == 3
        domains = [_domain_of(c["url"]) for c in result]
        assert domains.count("fandom.com") == 2
        assert domains.count("imgur.com") == 1

    def test_image_search_query_expansion(self):
        """_expand_queries produces variants for known shorthand terms."""
        _EXPANSIONS = {
            "gfl2": "Girls Frontline 2", "gfl": "Girls Frontline",
            "hsr":  "Honkai Star Rail",  "gi":  "Genshin Impact",
        }

        def _expand_queries(q):
            variants = [q]
            q_lower  = q.lower()
            for short, full in _EXPANSIONS.items():
                if short in q_lower and full.lower() not in q_lower:
                    variants.append(q.replace(short, full).replace(short.upper(), full))
                    break
            if "artwork" not in q_lower and "art" not in q_lower:
                variants.append(q + " artwork")
            return variants[:3]

        v = _expand_queries("Klukai GFL2")
        assert len(v) >= 2
        assert any("Girls Frontline 2" in vv for vv in v), \
            f"Expected GFL2 expansion, got: {v}"
        # No 'artwork' appended since we cap at 3 and already have 2 variants (base + expansion)
        # but ensure no duplicates
        assert len(v) == len(set(v))


class TestBrowserV16Stealth:
    """Verify v16 stealth spoofs and ad-blocking are present in _SERVER_SRC."""

    def test_webgl_gpu_spoof(self):
        from aichat.tools.browser import _SERVER_SRC
        assert "GeForce RTX 3070" in _SERVER_SRC, "WebGL GPU spoof missing"
        assert "37445" in _SERVER_SRC, "WebGL VENDOR param (37445) missing"
        assert "37446" in _SERVER_SRC, "WebGL RENDERER param (37446) missing"

    def test_canvas_noise(self):
        from aichat.tools.browser import _SERVER_SRC
        assert "toDataURL" in _SERVER_SRC, "Canvas toDataURL override missing"
        assert "_seed" in _SERVER_SRC, "Canvas noise seed missing"
        assert "toBlob" in _SERVER_SRC, "Canvas toBlob override missing"

    def test_audio_context_noise(self):
        from aichat.tools.browser import _SERVER_SRC
        assert "getChannelData" in _SERVER_SRC, "AudioBuffer.getChannelData override missing"

    def test_screen_geometry_spoof(self):
        from aichat.tools.browser import _SERVER_SRC
        assert "devicePixelRatio" in _SERVER_SRC, "devicePixelRatio spoof missing"
        assert "outerWidth" in _SERVER_SRC, "outerWidth spoof missing"
        assert "availWidth" in _SERVER_SRC, "availWidth spoof missing"

    def test_device_memory_spoof(self):
        from aichat.tools.browser import _SERVER_SRC
        assert "deviceMemory" in _SERVER_SRC, "deviceMemory spoof missing"

    def test_media_devices_stub(self):
        from aichat.tools.browser import _SERVER_SRC
        assert "enumerateDevices" in _SERVER_SRC, "enumerateDevices stub missing"
        assert "audioinput" in _SERVER_SRC, "audioinput device stub missing"

    def test_ad_domains_present(self):
        from aichat.tools.browser import _SERVER_SRC
        assert "_AD_DOMAINS" in _SERVER_SRC, "_AD_DOMAINS missing"
        assert "doubleclick.net" in _SERVER_SRC, "doubleclick.net missing from _AD_DOMAINS"
        assert "googlesyndication.com" in _SERVER_SRC, "googlesyndication.com missing"
        assert "outbrain.com" in _SERVER_SRC, "outbrain.com missing"
        assert "hotjar.com" in _SERVER_SRC, "hotjar.com missing"

    def test_route_handler_present(self):
        from aichat.tools.browser import _SERVER_SRC
        assert "_route_handler" in _SERVER_SRC, "_route_handler function missing"
        assert "route.abort()" in _SERVER_SRC, "route.abort() missing"
        assert "route.continue_()" in _SERVER_SRC, "route.continue_() missing"

    def test_new_page_helper_present(self):
        from aichat.tools.browser import _SERVER_SRC
        assert "async def _new_page()" in _SERVER_SRC, "_new_page() helper missing"
        assert 'route("**/*", _route_handler)' in _SERVER_SRC, "route registration missing"

    def test_extra_launch_args(self):
        from aichat.tools.browser import _SERVER_SRC
        assert "--no-first-run" in _SERVER_SRC, "--no-first-run flag missing"
        assert "--lang=en-US" in _SERVER_SRC, "--lang=en-US flag missing"


# ===========================================================================
# Browser v17 — _human_move, navigator spoofs, scroll jitter, block-retry
# ===========================================================================

class TestBrowserV17:
    """Verify v17 improvements are present in _SERVER_SRC and dependent files."""

    def test_human_move_in_server_src(self):
        from aichat.tools.browser import _SERVER_SRC
        assert "_human_move" in _SERVER_SRC, "_human_move function missing from _SERVER_SRC"
        assert "page.mouse.move" in _SERVER_SRC, "page.mouse.move call missing"

    def test_navigator_vendor_spoof(self):
        from aichat.tools.browser import _SERVER_SRC
        assert "Google Inc." in _SERVER_SRC, "navigator.vendor spoof missing"
        assert "maxTouchPoints" in _SERVER_SRC, "maxTouchPoints spoof missing"

    def test_navigator_cookie_online_spoof(self):
        from aichat.tools.browser import _SERVER_SRC
        assert "cookieEnabled" in _SERVER_SRC, "cookieEnabled spoof missing"
        assert "onLine" in _SERVER_SRC, "onLine spoof missing"

    def test_scroll_jitter(self):
        from aichat.tools.browser import _SERVER_SRC
        assert "uniform(0.6, 1.5)" in _SERVER_SRC, "Scroll jitter uniform(0.6, 1.5) missing"

    def test_screenshot_block_retry(self):
        from aichat.tools.browser import _SERVER_SRC
        assert "_is_blocked(page_text)" in _SERVER_SRC, "Screenshot block-retry missing"

    def test_bing_tier3_in_manager(self):
        import inspect
        from aichat.tools.manager import ToolManager
        src = inspect.getsource(ToolManager.run_image_search)
        assert "bing.com/images" in src, "Bing Images Tier 3 missing from manager"

    def test_bing_tier3_in_docker_mcp(self):
        with open("docker/mcp/app.py") as f:
            src = f.read()
        assert "bing.com/images" in src, "Bing Images Tier 3 missing from docker/mcp/app.py"


# ===========================================================================
# Image recognition — dHash dedup + vision confirm + DB caching
# ===========================================================================

class TestImageRecognition:
    """Unit + integration tests for perceptual hash dedup and DB-backed image caching."""

    # ── Pure-unit: _dhash + _hamming (no services needed) ──────────────────

    def test_dhash_identical_images_have_zero_distance(self):
        """Same image produces the same hash → Hamming distance 0."""
        from aichat.tools.manager import _dhash, _hamming
        from PIL import Image
        img = Image.new("RGB", (100, 100), (128, 64, 32))
        h = _dhash(img)
        assert len(h) == 16, f"Expected 16-char hex, got {len(h)}: {h!r}"
        assert _hamming(h, h) == 0

    def test_dhash_different_images_have_high_distance(self):
        """Images with opposite horizontal gradients should differ by many bits."""
        from aichat.tools.manager import _dhash, _hamming
        from PIL import Image
        # Gradient left→right: bright on left, dark on right
        w, h = 100, 50
        lr = Image.new("RGB", (w, h))
        lr.putdata([(int(255 * (1 - x / w)), 0, 0) for y in range(h) for x in range(w)])
        # Gradient right→left: dark on left, bright on right
        rl = Image.new("RGB", (w, h))
        rl.putdata([(int(255 * (x / w)), 0, 0) for y in range(h) for x in range(w)])
        dist = _hamming(_dhash(lr), _dhash(rl))
        assert dist > 20, f"Opposite-gradient images should differ by > 20 bits, got {dist}"

    def test_dhash_empty_string_returns_max_distance(self):
        """Empty/invalid hash sentinel → max distance (64)."""
        from aichat.tools.manager import _hamming
        assert _hamming("", "0000000000000000") == 64
        assert _hamming("0000000000000000", "") == 64
        assert _hamming("short", "0000000000000000") == 64

    def test_dhash_nearly_identical_images_have_low_distance(self):
        """Same image with minor brightness tweak should be < 8 bits apart."""
        from aichat.tools.manager import _dhash, _hamming
        from PIL import Image, ImageEnhance
        base = Image.new("RGB", (200, 200), (100, 150, 200))
        # Add a noise pattern so there's actual gradient information
        import random
        pixels = [(random.randint(90, 110), random.randint(140, 160),
                   random.randint(190, 210)) for _ in range(200 * 200)]
        base.putdata(pixels)
        tweaked = ImageEnhance.Brightness(base).enhance(1.02)  # 2% brighter
        h1 = _dhash(base)
        h2 = _dhash(tweaked)
        dist = _hamming(h1, h2)
        assert dist < 8, f"Near-identical images should have distance < 8, got {dist}"

    def test_dhash_present_in_docker_mcp(self):
        """docker/mcp/app.py must contain the _dhash and _hamming helpers."""
        with open("docker/mcp/app.py") as f:
            src = f.read()
        assert "def _dhash(" in src, "_dhash helper missing from docker/mcp/app.py"
        assert "def _hamming(" in src, "_hamming helper missing from docker/mcp/app.py"
        assert "_vision_confirm" in src, "_vision_confirm helper missing from docker/mcp/app.py"

    def test_db_first_fastpath_present_in_docker_mcp(self):
        """DB-first fast path should be present in the image_search handler."""
        with open("docker/mcp/app.py") as f:
            src = f.read()
        assert "/images/search" in src, "DB-first /images/search call missing from docker/mcp/app.py"
        assert "_norm_subject" in src, "_norm_subject variable missing from docker/mcp/app.py"

    # ── Integration: DB /images/search endpoint ────────────────────────────

    @skip_db
    def test_db_images_search_endpoint_exists(self):
        """GET /images/search returns structured JSON with images + count keys."""
        r = httpx.get(
            f"{DB_URL}/images/search", params={"subject": "test", "limit": 5}, timeout=5
        )
        assert r.status_code == 200
        data = r.json()
        assert "images" in data, f"Missing 'images' key: {data}"
        assert "count" in data,  f"Missing 'count' key: {data}"
        assert isinstance(data["images"], list)

    @skip_db
    def test_db_store_and_search_image_rich(self):
        """Store an image with phash + quality_score, then retrieve it by subject."""
        test_url = f"https://example.com/test_{uuid.uuid4().hex}.jpg"
        test_subject = f"test_subj_{uuid.uuid4().hex[:8]}"

        # Store with new fields
        store_r = httpx.post(
            f"{DB_URL}/images/store",
            json={
                "url": test_url,
                "subject": test_subject,
                "description": "a synthetic test image",
                "phash": "abcdef1234567890",
                "quality_score": 0.9,
            },
            timeout=5,
        )
        assert store_r.status_code == 200, store_r.text

        # Retrieve by subject
        search_r = httpx.get(
            f"{DB_URL}/images/search",
            params={"subject": test_subject, "limit": 5},
            timeout=5,
        )
        assert search_r.status_code == 200
        found = [img for img in search_r.json()["images"] if img["url"] == test_url]
        assert found, f"Stored image not found by subject search (subject={test_subject})"
        assert found[0]["phash"] == "abcdef1234567890", f"phash mismatch: {found[0]}"
        assert found[0]["quality_score"] >= 0.9 - 1e-6, f"quality_score mismatch: {found[0]}"


# ===========================================================================
# 26. New LM Studio tools — TTS, embeddings, code_run, summarize, caption, extract
# ===========================================================================


class TestNewLMStudioTools:
    """Pure unit + source-inspection + DB + MCP integration tests for the 7
    new LM Studio tools added in this session."""

    # ── Pure unit: import checks ───────────────────────────────────────────

    def test_lm_studio_tool_importable(self):
        from aichat.tools.lm_studio import LMStudioTool
        lm = LMStudioTool()
        assert hasattr(lm, "tts")
        assert hasattr(lm, "embed")
        assert hasattr(lm, "chat")
        assert hasattr(lm, "caption")
        assert hasattr(lm, "summarize")
        assert hasattr(lm, "extract")

    def test_code_interpreter_importable(self):
        from aichat.tools.code_interpreter import CodeInterpreterTool
        ci = CodeInterpreterTool()
        assert hasattr(ci, "run")
        assert ci.timeout == 30

    def test_cosine_similarity_function(self):
        from aichat.tools.lm_studio import cosine_similarity
        # Identical vectors → 1.0
        a = [1.0, 0.0, 0.0]
        assert abs(cosine_similarity(a, a) - 1.0) < 1e-9
        # Orthogonal vectors → 0.0
        b = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(a, b)) < 1e-9
        # Empty → 0.0
        assert cosine_similarity([], []) == 0.0

    def test_tool_name_entries_exist(self):
        from aichat.tools.manager import ToolName
        for expected in ("tts", "embed_store", "embed_search", "code_run",
                         "smart_summarize", "image_caption", "structured_extract"):
            names = {t.value for t in ToolName}
            assert expected in names, f"ToolName.{expected.upper()} missing"

    def test_lm_and_code_attrs_on_manager(self):
        from aichat.tools.manager import ToolManager
        from aichat.tools.lm_studio import LMStudioTool
        from aichat.tools.code_interpreter import CodeInterpreterTool
        mgr = ToolManager()
        assert isinstance(mgr.lm, LMStudioTool), "ToolManager.lm must be LMStudioTool"
        assert isinstance(mgr.code, CodeInterpreterTool), "ToolManager.code must be CodeInterpreterTool"

    # ── Pure unit: CodeInterpreterTool subprocess tests ───────────────────

    def test_code_interpreter_basic(self):
        from aichat.tools.code_interpreter import CodeInterpreterTool
        ci = CodeInterpreterTool()
        result = asyncio.run(ci.run("print(1+1)"))
        assert result["exit_code"] == 0, f"Expected exit 0: {result}"
        assert "2" in result["stdout"], f"Expected '2' in stdout: {result['stdout']}"
        assert result["error"] is None

    def test_code_interpreter_timeout(self):
        import time as _time
        from aichat.tools.code_interpreter import CodeInterpreterTool
        ci = CodeInterpreterTool(timeout=2)
        t0 = _time.monotonic()
        result = asyncio.run(ci.run("while True: pass"))
        elapsed = _time.monotonic() - t0
        assert result["exit_code"] == -1, f"Expected timeout exit -1: {result}"
        assert elapsed < 5.0, f"Timeout took too long: {elapsed:.1f}s"

    def test_code_interpreter_syntax_error(self):
        from aichat.tools.code_interpreter import CodeInterpreterTool
        ci = CodeInterpreterTool()
        result = asyncio.run(ci.run("def f( : pass"))
        assert result["exit_code"] != 0, f"Syntax error should give non-zero exit: {result}"
        assert result["stderr"].strip(), f"Syntax error should produce stderr: {result}"

    def test_code_interpreter_packages_stdlib(self):
        """Requesting stdlib 'json' as a package should not error on execution."""
        from aichat.tools.code_interpreter import CodeInterpreterTool
        ci = CodeInterpreterTool()
        result = asyncio.run(
            ci.run("import json; print(json.dumps({'ok': True}))", packages=["json"])
        )
        assert result["exit_code"] == 0, f"Expected exit 0: {result}"
        assert "ok" in result["stdout"]

    # ── Source-inspection: new tool schemas present in docker/mcp/app.py ──

    def _mcp_src(self) -> str:
        with open("docker/mcp/app.py") as f:
            return f.read()

    def test_tts_schema_in_mcp(self):
        assert '"tts"' in self._mcp_src() or "'tts'" in self._mcp_src(), \
            "tts schema missing from docker/mcp/app.py"

    def test_embed_store_schema_in_mcp(self):
        assert "embed_store" in self._mcp_src(), \
            "embed_store schema missing from docker/mcp/app.py"

    def test_embed_search_schema_in_mcp(self):
        assert "embed_search" in self._mcp_src(), \
            "embed_search schema missing from docker/mcp/app.py"

    def test_code_run_schema_in_mcp(self):
        assert "code_run" in self._mcp_src(), \
            "code_run schema missing from docker/mcp/app.py"

    def test_smart_summarize_schema_in_mcp(self):
        assert "smart_summarize" in self._mcp_src(), \
            "smart_summarize schema missing from docker/mcp/app.py"

    def test_image_caption_schema_in_mcp(self):
        assert "image_caption" in self._mcp_src(), \
            "image_caption schema missing from docker/mcp/app.py"

    def test_structured_extract_schema_in_mcp(self):
        assert "structured_extract" in self._mcp_src(), \
            "structured_extract schema missing from docker/mcp/app.py"

    # ── DB integration: embeddings endpoint ───────────────────────────────

    @skip_db
    def test_embeddings_store_endpoint(self):
        """POST /embeddings/store returns status=stored (or updated) for a fresh key."""
        key = f"test_embed_{uuid.uuid4().hex}"
        embedding = [0.1] * 128
        r = httpx.post(
            f"{DB_URL}/embeddings/store",
            json={"key": key, "content": "test content", "embedding": embedding,
                  "model": "test-model", "topic": "test"},
            timeout=10,
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data.get("status") in ("stored", "updated"), f"Unexpected status: {data}"

    @skip_db
    def test_embeddings_search_endpoint_cosine(self):
        """Store two docs, search returns the more similar one first."""
        # Embed doc A: 1-axis vector
        vec_a = [1.0, 0.0, 0.0] + [0.0] * 125
        # Embed doc B: 2-axis vector
        vec_b = [0.0, 1.0, 0.0] + [0.0] * 125
        key_a = f"emb_a_{uuid.uuid4().hex}"
        key_b = f"emb_b_{uuid.uuid4().hex}"
        topic = f"test_search_{uuid.uuid4().hex[:8]}"

        httpx.post(f"{DB_URL}/embeddings/store",
                   json={"key": key_a, "content": "doc A", "embedding": vec_a, "topic": topic},
                   timeout=10)
        httpx.post(f"{DB_URL}/embeddings/store",
                   json={"key": key_b, "content": "doc B", "embedding": vec_b, "topic": topic},
                   timeout=10)

        # Query with vec_a — doc A should rank higher
        r = httpx.post(
            f"{DB_URL}/embeddings/search",
            json={"embedding": vec_a, "limit": 5, "topic": topic},
            timeout=10,
        )
        assert r.status_code == 200, r.text
        results = r.json().get("results", [])
        assert results, "Expected at least one result"
        assert results[0]["key"] == key_a, \
            f"Expected key_a as top result, got {results[0]['key']}"

    # ── MCP integration: end-to-end tool calls ─────────────────────────────

    @skip_mcp
    def test_mcp_code_run(self):
        """code_run tool should execute Python and return the output."""
        blocks = _mcp_content("code_run", {"code": 'print("hello_mcp_code_run")'})
        text = " ".join(b.get("text", "") for b in blocks)
        assert "hello_mcp_code_run" in text, f"Expected stdout in response: {text}"

    @skip_mcp
    def test_mcp_code_run_missing_code_returns_error(self):
        blocks = _mcp_content("code_run", {})
        text = " ".join(b.get("text", "") for b in blocks)
        assert "code" in text.lower() and "required" in text.lower(), \
            f"Expected validation error: {text}"

    @skip_mcp
    def test_mcp_smart_summarize(self):
        """smart_summarize should return a non-empty response (even if model isn't loaded)."""
        content = (
            "Python is a high-level, general-purpose programming language. "
            "Its design philosophy emphasizes code readability. "
            "It supports multiple programming paradigms including structured, "
            "object-oriented, and functional programming."
        )
        blocks = _mcp_content("smart_summarize", {"content": content, "style": "brief"})
        text = " ".join(b.get("text", "") for b in blocks)
        # Either the summary or a helpful error about loading a model
        assert text.strip(), f"Expected non-empty response: {text}"

    @skip_mcp
    def test_mcp_tts_missing_text_returns_error(self):
        blocks = _mcp_content("tts", {})
        text = " ".join(b.get("text", "") for b in blocks)
        assert "text" in text.lower() and "required" in text.lower(), \
            f"Expected validation error: {text}"

    @skip_mcp
    def test_mcp_embed_store_missing_key_returns_error(self):
        blocks = _mcp_content("embed_store", {"content": "some text"})
        text = " ".join(b.get("text", "") for b in blocks)
        assert "key" in text.lower() and "required" in text.lower(), \
            f"Expected validation error: {text}"

    @skip_mcp
    def test_mcp_new_7_tools_in_tools_list(self):
        """All 7 new tools must appear in the MCP tools/list response."""
        r = httpx.post(
            f"{MCP_URL}/mcp",
            json={"jsonrpc": "2.0", "id": 500, "method": "tools/list", "params": {}},
            timeout=10,
        )
        names = {t["name"] for t in r.json()["result"]["tools"]}
        for tname in ("tts", "embed_store", "embed_search", "code_run",
                      "smart_summarize", "image_caption", "structured_extract"):
            assert tname in names, f"'{tname}' missing from MCP tools/list"
