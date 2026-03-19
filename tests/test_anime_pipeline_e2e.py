"""End-to-end tests for the anime image pipeline, MinIO, and CLIP embedding.

These tests require the full aichat stack to be running:
  docker compose -f docker-compose.yml -f docker-compose.ports.yml up -d

Tests that require external API keys (Pixiv, SauceNAO) are skipped if
the keys are not configured.
"""
from __future__ import annotations

import json
import os

import httpx
import pytest

# ---------------------------------------------------------------------------
# Service URLs — auto-discover from Docker or fall back to localhost
# ---------------------------------------------------------------------------

MCP_URL = os.environ.get("MCP_URL", "http://localhost:8096")
VISION_URL = os.environ.get("VISION_URL", "http://localhost:8099")
MINIO_URL = os.environ.get("MINIO_URL", "http://localhost:9002")


def _mcp_call(tool_name: str, arguments: dict) -> dict:
    """Call an MCP tool via JSON-RPC and return the result dict."""
    with httpx.Client(timeout=90) as c:
        r = c.post(
            f"{MCP_URL}/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            },
        )
        r.raise_for_status()
        return r.json().get("result", {})


def _result_text(result: dict) -> str:
    """Extract all text content from an MCP result."""
    return "\n".join(
        blk.get("text", "") for blk in result.get("content", [])
        if blk.get("type") == "text"
    )


def _result_images(result: dict) -> list[dict]:
    """Extract image blocks from an MCP result."""
    return [blk for blk in result.get("content", []) if blk.get("type") == "image"]


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------

class TestHealthChecks:
    def test_mcp_health(self):
        with httpx.Client(timeout=10) as c:
            r = c.get(f"{MCP_URL}/health")
            r.raise_for_status()
            data = r.json()
            assert data.get("ok") is True
            assert data.get("tools", 0) == 16  # 16 mega-tools

    def test_clip_health(self):
        with httpx.Client(timeout=10) as c:
            r = c.get(f"{VISION_URL}/clip/health")
            r.raise_for_status()
            data = r.json()
            assert data["status"] == "ok"
            assert data["model"] == "clip-vit-b32"
            assert data["model_exists"] is True

    def test_minio_health(self):
        with httpx.Client(timeout=10) as c:
            r = c.get(f"{MINIO_URL}/minio/health/live")
            assert r.status_code == 200


# ---------------------------------------------------------------------------
# CLIP embedding
# ---------------------------------------------------------------------------

class TestCLIPEmbed:
    def test_clip_embed_returns_512dim(self):
        """CLIP ViT-B/32 visual encoder produces 512-dim L2-normalized vectors."""
        import base64
        from io import BytesIO
        from PIL import Image

        # Generate a small test image
        img = Image.new("RGB", (224, 224), color=(100, 150, 200))
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        with httpx.Client(timeout=30) as c:
            r = c.post(
                f"{VISION_URL}/clip/embed",
                json={"image_base64": b64},
            )
            r.raise_for_status()
            data = r.json()

        assert data["model"] == "clip-vit-b32"
        assert data["dim"] == 512
        assert len(data["embedding"]) == 512
        # Check L2 normalization (norm should be ~1.0)
        norm = sum(x**2 for x in data["embedding"]) ** 0.5
        assert abs(norm - 1.0) < 0.01

    def test_clip_embed_different_images_different_vectors(self):
        """Different images should produce different embeddings."""
        import base64
        from io import BytesIO
        from PIL import Image

        embeddings = []
        for color in [(255, 0, 0), (0, 255, 0), (0, 0, 255)]:
            img = Image.new("RGB", (224, 224), color=color)
            buf = BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            with httpx.Client(timeout=30) as c:
                r = c.post(f"{VISION_URL}/clip/embed", json={"image_base64": b64})
                r.raise_for_status()
                embeddings.append(r.json()["embedding"])

        # Red, Green, Blue should have different embeddings
        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            return dot  # already L2-normalized, so dot product = cosine similarity

        assert cosine(embeddings[0], embeddings[1]) < 0.99  # red vs green
        assert cosine(embeddings[0], embeddings[2]) < 0.99  # red vs blue


# ---------------------------------------------------------------------------
# Tool discovery
# ---------------------------------------------------------------------------

class TestToolDiscovery:
    def test_list_tools_all_categories(self):
        result = _mcp_call("list_tools_by_category", {})
        text = _result_text(result)
        # Anime category removed (paid APIs). Check for real categories.
        assert "image" in text.lower()
        assert "web" in text.lower() or "search" in text.lower()

    def test_list_tools_anime_category(self):
        pytest.skip("anime/danbooru/saucenao tools removed — no paid APIs allowed")

    def test_list_tools_search_keyword(self):
        result = _mcp_call("list_tools_by_category", {"search": "image"})
        text = _result_text(result)
        assert "image" in text.lower()


# ---------------------------------------------------------------------------
# Anime search
# ---------------------------------------------------------------------------

class TestAnimeSearch:
    def test_danbooru_search_returns_images(self):
        pytest.skip("anime_search removed — Danbooru requires paid API key")

    def test_searxng_search_returns_results(self):
        # SearXNG-based image search is now image.search in mega-tool
        result = _mcp_call("image", {"action": "search", "query": "miku vocaloid", "count": 2})
        text = _result_text(result)
        assert len(text) > 0


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class TestAnimePipeline:
    def test_pipeline_stores_and_embeds(self):
        pytest.skip("anime_pipeline removed — Danbooru/SauceNAO require paid API keys")


# ---------------------------------------------------------------------------
# SauceNAO (optional — skip if no API key)
# ---------------------------------------------------------------------------

class TestSauceNAO:
    def test_saucenao_requires_key(self):
        result = _mcp_call("saucenao_search", {
            "image_url": "https://example.com/test.jpg",
        })
        text = _result_text(result)
        # Should either work (key set) or report "not configured"
        assert "saucenao" in text.lower()
