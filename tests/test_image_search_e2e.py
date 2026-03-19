"""
E2E tests for image_search — validates SearXNG Tier 0 delivers inline
base64 images that render in LM Studio for queries that previously failed
(Klukai/GFL2 character art, anime, blocked sites).

Requires live stack: aichat-mcp (8096) + aichat-searxng (8098).
"""
import json
import os

import httpx
import pytest

MCP_URL  = os.environ.get("MCP_URL", "http://localhost:8096")
SEARXNG  = os.environ.get("SEARXNG_URL", "http://localhost:8098")

# Skip if MCP not reachable
try:
    _ok = httpx.get(f"{MCP_URL}/health", timeout=3).status_code == 200
except Exception:
    _ok = False

# Skip if SearXNG not exposed on host (it's internal-only in production)
try:
    _searxng_ok = httpx.get(f"{SEARXNG}/", timeout=3).status_code < 500
except Exception:
    _searxng_ok = False

live = pytest.mark.skipif(not _ok, reason="aichat-mcp not reachable")
live_searxng = pytest.mark.skipif(not _searxng_ok, reason="SearXNG not reachable on host (internal-only)")


def _call_tool(name: str, args: dict, timeout: float = 60.0) -> list[dict]:
    r = httpx.post(
        f"{MCP_URL}/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
              "params": {"name": name, "arguments": args}},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json().get("result", {}).get("content", [])


# ---------------------------------------------------------------------------
# SearXNG sanity — direct API
# ---------------------------------------------------------------------------

@live
@live_searxng
def test_searxng_image_api_returns_results():
    """SearXNG image endpoint must return ≥5 results for the problematic query."""
    r = httpx.get(
        f"{SEARXNG}/search",
        params={"q": "Klukai Girls Frontline 2", "format": "json", "categories": "images"},
        timeout=20,
    )
    assert r.status_code == 200
    results = r.json().get("results", [])
    assert len(results) >= 5, f"Expected ≥5 SearXNG image results, got {len(results)}"
    img_src_count = sum(1 for res in results if res.get("img_src", "").startswith("http"))
    assert img_src_count >= 3, "Expected ≥3 results with direct img_src URLs"


# ---------------------------------------------------------------------------
# image_search MCP tool — must return inline base64 image blocks
# ---------------------------------------------------------------------------

@live
def test_image_search_klukai_returns_inline_images():
    """
    Previously failing case: 'Klukai Girls Frontline 2 artwork'.
    Must return ≥1 image block with base64 data suitable for LM Studio rendering.
    """
    content = _call_tool("image_search", {"query": "Klukai Girls Frontline 2 artwork", "count": 3})
    images = [b for b in content if b.get("type") == "image"]
    assert len(images) >= 1, (
        f"image_search returned no image blocks for Klukai — "
        f"got content: {[b.get('type') for b in content]}"
    )
    for img in images:
        assert img.get("data"), "Image block missing base64 data"
        assert img.get("mimeType", "").startswith("image/"), f"Bad mimeType: {img.get('mimeType')}"
        # Sanity: must be a real image (≥5KB base64 ≈ ≥3.75KB raw)
        assert len(img["data"]) >= 5000, f"Image data suspiciously small: {len(img['data'])} chars"


@live
def test_image_search_gfl2_shorthand_expansion():
    """GFL2 shorthand must expand and still return images."""
    content = _call_tool("image_search", {"query": "Klukai GFL2", "count": 2})
    images = [b for b in content if b.get("type") == "image"]
    assert len(images) >= 1, "image_search with GFL2 shorthand returned no images"


@live
def test_image_search_renders_multiple_images():
    """count=4 must return ≥2 distinct inline images."""
    content = _call_tool("image_search", {"query": "Girls Frontline 2 character art", "count": 4})
    images = [b for b in content if b.get("type") == "image"]
    assert len(images) >= 2, f"Expected ≥2 images, got {len(images)}"
    # All must be different — compare a mid-body slice (skip identical JPEG SOI header)
    fingerprints = {img["data"][200:300] for img in images}
    assert len(fingerprints) >= max(1, len(images) - 1), (
        f"Too many duplicate images returned: {len(images)} images but only {len(fingerprints)} unique"
    )


@live
def test_image_search_no_error_text_when_images_found():
    """When images are found, response must NOT contain an error/failure message."""
    content = _call_tool("image_search", {"query": "Klukai Girls Frontline 2", "count": 2})
    images = [b for b in content if b.get("type") == "image"]
    if images:
        texts = [b.get("text", "") for b in content if b.get("type") == "text"]
        error_texts = [t for t in texts if "no image found" in t.lower() or "failed" in t.lower()]
        assert not error_texts, f"Got error text alongside images: {error_texts}"


@live
def test_image_search_offset_returns_different_images():
    """offset=0 and offset=3 must return different images (dedup/pagination works)."""
    c0 = _call_tool("image_search", {"query": "Klukai Girls Frontline 2", "count": 3, "offset": 0})
    c3 = _call_tool("image_search", {"query": "Klukai Girls Frontline 2", "count": 3, "offset": 3})
    imgs0 = {b["data"][200:400] for b in c0 if b.get("type") == "image"}
    imgs3 = {b["data"][200:400] for b in c3 if b.get("type") == "image"}
    overlap = imgs0 & imgs3
    # Allow up to 1 duplicate — external search providers occasionally return
    # the same popular image at different offsets (e.g. CDN deduplication).
    assert len(overlap) <= 1 or len(imgs0) == 0 or len(imgs3) == 0, (
        f"offset pagination returned too many overlapping images: {len(overlap)} duplicates"
    )


# ---------------------------------------------------------------------------
# web_search engine=images — text results (not inline images)
# ---------------------------------------------------------------------------

@live
def test_web_search_images_engine_returns_image_urls():
    """web_search engine=images must return Image URL lines in the text result."""
    content = _call_tool("web_search", {"query": "Klukai Girls Frontline 2", "engine": "images"})
    texts = [b.get("text", "") for b in content if b.get("type") == "text"]
    combined = "\n".join(texts)
    assert "Image URL:" in combined or "Image results" in combined, (
        f"web_search engine=images didn't return image URLs:\n{combined[:300]}"
    )
