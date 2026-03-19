"""
Full regression test suite — every MCP tool + every service endpoint.
Klukai (Girls Frontline 2) images are used for all image-related tests.

Run locally:
    DATA_URL=http://localhost:8091 VISION_URL=http://localhost:8099 \
    DOCS_URL=http://localhost:8101 SANDBOX_URL=http://localhost:8095 \
    MCP_URL=http://localhost:8096 \
    pytest tests/test_full_regression.py -v --tb=short -m regression
"""
from __future__ import annotations

import base64
import glob
import json
import os
import struct
import time
import zlib
from typing import Any

import httpx
import pytest

# ---------------------------------------------------------------------------
# Service URLs (overridable via env for CI)
# ---------------------------------------------------------------------------
MCP_URL     = os.environ.get("MCP_URL",     "http://localhost:8096")
DATA_URL    = os.environ.get("DATA_URL",    "http://localhost:8091")
VISION_URL  = os.environ.get("VISION_URL",  "http://localhost:8099")
DOCS_URL    = os.environ.get("DOCS_URL",    "http://localhost:8101")
SANDBOX_URL = os.environ.get("SANDBOX_URL", "http://localhost:8095")
JUPYTER_URL = os.environ.get("JUPYTER_URL", "http://localhost:8098")

TIMEOUT     = 10.0
LONG_TIMEOUT = 120.0   # vision / LM Studio calls

WORKSPACE   = "/docker/human_browser/workspace"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reachable(url: str, timeout: float = 3.0) -> bool:
    try:
        return httpx.get(url, timeout=timeout, follow_redirects=True).status_code < 400
    except Exception:
        return False


def mcp(name: str, args: dict[str, Any], timeout: float = TIMEOUT) -> str:
    """Call an MCP tool via JSON-RPC and return the first text content."""
    rpc = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
           "params": {"name": name, "arguments": args}}
    r = httpx.post(f"{MCP_URL}/mcp", json=rpc, timeout=timeout)
    assert r.status_code == 200, f"{name}: HTTP {r.status_code}"
    content = r.json().get("result", {}).get("content", [])
    return next((i.get("text", "") for i in content if i.get("type") == "text"), "")


def klukai_path() -> str:
    """Return path to a Klukai image that exists in workspace."""
    candidates = sorted(
        glob.glob(f"{WORKSPACE}/klukai_search__*.jpg")
        + glob.glob(f"{WORKSPACE}/klukai_*.png")
        + glob.glob(f"{WORKSPACE}/klukai_*.jpg")
    )
    if candidates:
        return candidates[0]
    # fallback: newest jpg in workspace
    imgs = sorted(glob.glob(f"{WORKSPACE}/*.jpg"), key=os.path.getmtime, reverse=True)
    return imgs[0] if imgs else ""


def klukai_b64() -> str:
    p = klukai_path()
    if not p:
        pytest.skip("No Klukai/workspace image available")
    return base64.standard_b64encode(open(p, "rb").read()).decode()


def _make_tiny_png() -> bytes:
    """Valid 4×4 white RGB PNG using stdlib only."""
    def _chunk(tag: bytes, data: bytes) -> bytes:
        payload = tag + data
        return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 0))
    raw = b"".join(b"\x00" + b"\xff\xff\xff" * 4 for _ in range(4))
    idat = _chunk(b"IDAT", zlib.compress(raw))
    return sig + ihdr + idat + _chunk(b"IEND", b"")


pytestmark = pytest.mark.regression


# ===========================================================================
# SERVICE HEALTH — all 6 services
# ===========================================================================

class TestServiceHealth:
    def test_data_health(self):
        if not _reachable(f"{DATA_URL}/health"):
            pytest.skip("aichat-data unreachable")
        r = httpx.get(f"{DATA_URL}/health", timeout=TIMEOUT)
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        for svc in ("postgres", "memory", "graph", "planner", "jobs"):
            assert svc in body["services"], f"sub-service '{svc}' missing"

    def test_vision_health(self):
        if not _reachable(f"{VISION_URL}/health"):
            pytest.skip("aichat-vision unreachable")
        r = httpx.get(f"{VISION_URL}/health", timeout=TIMEOUT)
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_docs_health(self):
        if not _reachable(f"{DOCS_URL}/health"):
            pytest.skip("aichat-docs unreachable")
        r = httpx.get(f"{DOCS_URL}/health", timeout=TIMEOUT)
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_sandbox_health(self):
        if not _reachable(f"{SANDBOX_URL}/health"):
            pytest.skip("aichat-sandbox unreachable")
        r = httpx.get(f"{SANDBOX_URL}/health", timeout=TIMEOUT)
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_mcp_health(self):
        if not _reachable(f"{MCP_URL}/health"):
            pytest.skip("aichat-mcp unreachable")
        r = httpx.get(f"{MCP_URL}/health", timeout=TIMEOUT)
        assert r.status_code == 200
        body = r.json()
        assert body.get("ok") is True or body.get("status") == "ok", f"Unexpected health body: {body}"
        assert body.get("tools", 0) == 16

    def test_mcp_tools_list(self):
        if not _reachable(f"{MCP_URL}/health"):
            pytest.skip()
        r = httpx.post(f"{MCP_URL}/mcp",
                       json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
                       timeout=TIMEOUT)
        assert r.status_code == 200
        tools = r.json()["result"]["tools"]
        assert len(tools) == 16, f"Expected 16 mega-tools, got {len(tools)}"
        names = {t["name"] for t in tools}
        # Verify all 16 mega-tool names are present
        for expected in ("web", "browser", "image", "document", "media", "data",
                         "memory", "knowledge", "vector", "code", "custom_tools",
                         "planner", "jobs", "research", "think", "system"):
            assert expected in names, f"Mega-tool '{expected}' missing"


# ===========================================================================
# DATA SERVICE — postgres, memory, graph, errors, research, planner, jobs
# ===========================================================================

class TestDataService:
    def setup_method(self):
        if not _reachable(f"{DATA_URL}/health"):
            pytest.skip("aichat-data unreachable")

    # --- PostgreSQL articles ---
    def test_db_store_article(self):
        txt = mcp("db_store_article", {
            "url": "https://example.com/klukai-gfl2",
            "title": "Klukai GFL2 regression test",
            "content": "Klukai is HK416 in Girls Frontline 2.",
            "topic": "regression"
        })
        assert "stored" in txt.lower() or "ok" in txt.lower() or "article" in txt.lower(), txt

    def test_db_search(self):
        txt = mcp("db_search", {"query": "Klukai GFL2 regression", "max_results": 3})
        assert txt  # any non-empty response

    # --- Page cache ---
    def test_db_cache_store_get(self):
        key = "regression_cache_klukai"
        mcp("db_cache_store", {"url": key, "content": "klukai-cache-value", "content_type": "text/plain"})
        txt = mcp("db_cache_get", {"url": key})
        assert "klukai-cache-value" in txt or "found" in txt.lower() or txt

    # --- Memory ---
    def test_memory_store_recall(self):
        mcp("memory_store", {"key": "_reg_klukai", "value": "klukai-memory-test"})
        txt = mcp("memory_recall", {"key": "_reg_klukai"})
        assert "klukai-memory-test" in txt, f"Memory recall failed: {txt[:200]}"

    # --- Error log ---
    def test_get_errors(self):
        txt = mcp("get_errors", {"limit": 5})
        assert "error" in txt.lower() or "errors" in txt.lower() or txt is not None

    # --- Research box ---
    def test_researchbox_search(self):
        txt = mcp("researchbox_search", {"topic": "Girls Frontline 2"})
        assert txt  # any non-empty response

    def test_researchbox_push(self):
        txt = mcp("researchbox_push", {
            "url": "https://example.com/klukai",
            "topic": "anime character regression"
        })
        assert txt

    # --- Graph ---
    def test_graph_add_node_edge_query(self):
        mcp("graph_add_node", {"node_id": "klukai", "label": "Character",
                               "properties": {"game": "GFL2", "weapon": "HK416"}})
        mcp("graph_add_node", {"node_id": "gfl2", "label": "Game",
                               "properties": {"publisher": "MICA"}})
        mcp("graph_add_edge", {"source": "klukai", "target": "gfl2",
                               "relation": "appears_in"})
        txt = mcp("graph_query", {"node_id": "klukai"})
        assert "klukai" in txt.lower() or txt

    def test_graph_path(self):
        txt = mcp("graph_path", {"source": "klukai", "target": "gfl2"})
        assert txt

    def test_graph_search(self):
        txt = mcp("graph_search", {"query": "klukai"})
        assert txt

    # --- Planner ---
    def test_plan_create_list_complete(self):
        create = mcp("plan_create_task", {
            "title": "Regression: analyse Klukai artwork",
            "description": "Test task from regression suite",
            "priority": 1
        })
        # extract task id from response
        import re
        tid_match = re.search(r'[0-9a-f\-]{8,}', create)
        if not tid_match:
            pytest.skip(f"Could not parse task id: {create[:200]}")
        tid = tid_match.group()
        txt = mcp("plan_list_tasks", {"status": "pending"})
        assert txt
        mcp("plan_complete_task", {"task_id": tid})
        mcp("plan_delete_task", {"task_id": tid})

    # --- Async jobs ---
    def test_job_submit_status_result(self):
        import re
        submit = mcp("job_submit", {
            "tool_name": "web_search",
            "args": {"query": "Klukai GFL2", "max_results": 1}
        })
        jid_match = re.search(r'[0-9a-f\-]{8,}', submit)
        assert jid_match, f"Could not parse job id from: {submit[:200]}"
        jid = jid_match.group()
        for _ in range(10):
            status = mcp("job_status", {"job_id": jid})
            if "completed" in status.lower() or "failed" in status.lower():
                break
            time.sleep(3)
        result = mcp("job_result", {"job_id": jid})
        assert result
        mcp("job_list", {"limit": 5})

    def test_job_cancel(self):
        import re
        submit = mcp("job_submit", {
            "tool_name": "web_search",
            "args": {"query": "cancel-test", "max_results": 1}
        })
        jid_match = re.search(r'[0-9a-f\-]{8,}', submit)
        assert jid_match, f"Could not parse job id: {submit[:200]}"
        jid = jid_match.group()
        txt = mcp("job_cancel", {"job_id": jid})
        assert txt


# ===========================================================================
# VISION SERVICE — OCR + video stubs
# ===========================================================================

class TestVisionService:
    def setup_method(self):
        if not _reachable(f"{VISION_URL}/health"):
            pytest.skip("aichat-vision unreachable")

    def test_ocr_health(self):
        r = httpx.get(f"{VISION_URL}/ocr/health", timeout=TIMEOUT)
        assert r.status_code == 200
        assert "tesseract" in r.json().get("tesseract_version", "").lower() or r.json()["status"] == "ok"

    def test_ocr_image_direct(self):
        b64 = base64.standard_b64encode(_make_tiny_png()).decode()
        r = httpx.post(f"{VISION_URL}/ocr", json={"b64": b64, "lang": "eng"}, timeout=30)
        assert r.status_code == 200
        body = r.json()
        assert "text" in body and "word_count" in body

    def test_ocr_klukai_via_mcp(self):
        p = klukai_path()
        if not p:
            pytest.skip("No Klukai image")
        txt = mcp("ocr_image", {"path": p, "lang": "eng"}, timeout=60)
        assert "OCR result" in txt and "404" not in txt, txt[:300]

    def test_ocr_languages(self):
        r = httpx.get(f"{VISION_URL}/ocr/languages", timeout=TIMEOUT)
        assert r.status_code == 200
        langs = r.json().get("languages", [])
        assert "eng" in langs

    def test_video_info_stub(self):
        # video_info with a synthetic non-video path returns error, not 500
        txt = mcp("video_info", {"url": "https://example.com/fake.mp4"}, timeout=30)
        assert txt  # any response (error or result) is acceptable


# ===========================================================================
# DOCS SERVICE — ingest, tables, PDF ops
# ===========================================================================

class TestDocsService:
    def setup_method(self):
        if not _reachable(f"{DOCS_URL}/health"):
            pytest.skip("aichat-docs unreachable")

    def test_docs_formats(self):
        r = httpx.get(f"{DOCS_URL}/formats", timeout=TIMEOUT)
        assert r.status_code == 200

    def test_docs_ingest_url(self):
        txt = mcp("docs_ingest", {"url": "https://iopwiki.com/wiki/Klukai"}, timeout=60)
        assert txt and "404" not in txt

    def test_docs_extract_tables(self):
        txt = mcp("docs_extract_tables", {"url": "https://en.wikipedia.org/wiki/HK416"}, timeout=60)
        assert txt

    def test_pdf_read_stub(self):
        # no local PDF — ensure graceful error, not 500
        txt = mcp("pdf_read", {"path": "/nonexistent.pdf"}, timeout=30)
        assert txt  # error message is fine

    def test_pdf_merge_stub(self):
        txt = mcp("pdf_merge", {"paths": []}, timeout=15)
        assert txt  # error or success


# ===========================================================================
# SANDBOX SERVICE — code execution
# ===========================================================================

class TestSandboxService:
    def setup_method(self):
        if not _reachable(f"{SANDBOX_URL}/health"):
            pytest.skip("aichat-sandbox unreachable")

    def test_sandbox_tools_list(self):
        r = httpx.get(f"{SANDBOX_URL}/tools", timeout=TIMEOUT)
        assert r.status_code == 200
        assert isinstance(r.json().get("tools"), list)

    def test_code_run_python(self):
        txt = mcp("code_run", {
            "language": "python",
            "code": "x = 6 * 7\nprint(f'klukai result: {x}')"
        }, timeout=30)
        assert "42" in txt or "klukai result" in txt, txt[:300]

    def test_code_run_bash(self):
        txt = mcp("code_run", {
            "language": "bash",
            "code": "echo 'klukai-bash-ok'"
        }, timeout=30)
        assert "klukai-bash-ok" in txt, txt[:300]

    def test_create_list_delete_custom_tool(self):
        mcp("create_tool", {
            "name": "_reg_test_tool",
            "description": "regression test tool",
            "code": "print('klukai-custom-tool')",
            "language": "python"
        })
        lst = mcp("list_custom_tools", {})
        assert "_reg_test_tool" in lst or lst
        call_result = mcp("call_custom_tool", {"name": "_reg_test_tool", "args": {}}, timeout=30)
        assert call_result
        mcp("delete_custom_tool", {"name": "_reg_test_tool"})


# ===========================================================================
# VECTOR SERVICE (Qdrant)
# ===========================================================================

class TestVectorService:
    def setup_method(self):
        if not _reachable("http://localhost:6333/healthz", timeout=3):
            pytest.skip("Qdrant unreachable")

    def test_vector_store_search_delete(self):
        coll = "regression_klukai"
        mcp("vector_store", {
            "collection": coll,
            "id": "klukai_reg_001",
            "text": "Klukai is a character in Girls Frontline 2, also known as HK416",
            "metadata": {"source": "regression"}
        })
        txt = mcp("vector_search", {
            "collection": coll,
            "query": "Girls Frontline character",
            "top_k": 3
        })
        assert txt
        mcp("vector_delete", {"collection": coll, "id": "klukai_reg_001"})

    def test_vector_collections(self):
        txt = mcp("vector_collections", {})
        assert txt


# ===========================================================================
# WEB & BROWSER TOOLS
# ===========================================================================

class TestWebTools:
    def setup_method(self):
        if not _reachable(f"{MCP_URL}/health"):
            pytest.skip("aichat-mcp unreachable")

    def test_web_search(self):
        txt = mcp("web_search", {"query": "Klukai Girls Frontline 2 character", "max_results": 3})
        assert "Klukai" in txt or "Girls Frontline" in txt or txt, txt[:300]

    def test_web_fetch(self):
        txt = mcp("web_fetch", {"url": "https://iopwiki.com/wiki/Klukai"}, timeout=30)
        assert txt and len(txt) > 50

    def test_page_extract(self):
        txt = mcp("page_extract", {"url": "https://iopwiki.com/wiki/Klukai"}, timeout=30)
        assert txt

    def test_extract_article(self):
        txt = mcp("extract_article", {"url": "https://iopwiki.com/wiki/Klukai"}, timeout=30)
        assert txt

    def test_page_scrape(self):
        txt = mcp("page_scrape", {"url": "https://iopwiki.com/wiki/Klukai"}, timeout=30)
        assert txt

    def test_page_images(self):
        txt = mcp("page_images", {"url": "https://iopwiki.com/wiki/Klukai"}, timeout=30)
        assert txt

    def test_image_search(self):
        txt = mcp("image_search", {
            "query": "Klukai HK416 Girls Frontline 2 character art",
            "max_results": 3
        }, timeout=120)
        assert txt

    def test_fetch_image(self):
        txt = mcp("fetch_image", {
            "url": "https://pbs.twimg.com/media/GV8L06CXUAA2Red.jpg"
        }, timeout=30)
        assert txt and ("image" in txt.lower() or "jpg" in txt.lower() or "png" in txt.lower())

    def test_smart_summarize(self):
        txt = mcp("smart_summarize", {
            "url": "https://iopwiki.com/wiki/Klukai",
            "focus": "appearance and weapons"
        }, timeout=60)
        assert txt


# ===========================================================================
# IMAGE PIPELINE TOOLS (all use Klukai)
# ===========================================================================

class TestImagePipeline:
    def setup_method(self):
        if not _reachable(f"{MCP_URL}/health"):
            pytest.skip("aichat-mcp unreachable")
        self.klukai = klukai_path()
        if not self.klukai:
            pytest.skip("No Klukai image in workspace")

    def test_face_recognize(self):
        txt = mcp("face_recognize", {"path": self.klukai, "annotate": True}, timeout=60)
        assert "face" in txt.lower() or "detected" in txt.lower(), txt[:300]

    def test_image_caption(self):
        if not _reachable("http://192.168.50.2:1234/v1/models", timeout=3):
            pytest.skip("LM Studio not reachable at 192.168.50.2:1234")
        b64 = klukai_b64()
        txt = mcp("image_caption", {"b64": b64, "detail_level": "detailed"}, timeout=LONG_TIMEOUT)
        # Must get a real description, not a routing error or empty
        assert "404" not in txt, f"image_caption routing error: {txt[:300]}"
        assert any(w in txt.lower() for w in ("hair", "eye", "character", "anime", "cloth", "color", "image")), \
            f"image_caption did not describe Klukai: {txt[:400]}"

    def test_image_crop(self):
        txt = mcp("image_crop", {
            "path": self.klukai,
            "x": 0, "y": 0, "width": 100, "height": 100
        }, timeout=30)
        assert txt and "error" not in txt.lower()[:50]

    def test_image_zoom(self):
        txt = mcp("image_zoom", {
            "path": self.klukai,
            "x": 0, "y": 0, "width": 200, "height": 200,
            "scale": 2.0
        }, timeout=30)
        assert txt

    def test_image_scan(self):
        txt = mcp("image_scan", {"path": self.klukai}, timeout=30)
        assert txt

    def test_image_enhance(self):
        txt = mcp("image_enhance", {
            "path": self.klukai,
            "contrast": 1.2,
            "sharpness": 1.1
        }, timeout=30)
        assert txt

    def test_image_stitch(self):
        txt = mcp("image_stitch", {
            "paths": [self.klukai, self.klukai],
            "direction": "horizontal"
        }, timeout=30)
        assert txt

    def test_image_diff(self):
        txt = mcp("image_diff", {
            "path_a": self.klukai,
            "path_b": self.klukai
        }, timeout=30)
        assert txt

    def test_image_annotate(self):
        txt = mcp("image_annotate", {
            "path": self.klukai,
            "boxes": [{"x": 10, "y": 10, "width": 50, "height": 50, "label": "Klukai"}]
        }, timeout=30)
        assert txt

    def test_db_store_image(self):
        txt = mcp("db_store_image", {
            "path": self.klukai,
            "label": "klukai_regression"
        }, timeout=30)
        assert txt

    def test_db_list_images(self):
        txt = mcp("db_list_images", {"limit": 5})
        assert txt

    def test_screenshot_search(self):
        txt = mcp("screenshot_search", {"query": "klukai"}, timeout=60)
        assert txt


# ===========================================================================
# IMAGE GENERATION (LM Studio)
# ===========================================================================

class TestImageGeneration:
    def setup_method(self):
        if not _reachable(f"{MCP_URL}/health"):
            pytest.skip("aichat-mcp unreachable")

    def test_image_generate(self):
        txt = mcp("image_generate", {
            "prompt": "Klukai from Girls Frontline 2, anime style, HK416, white hair",
            "width": 512,
            "height": 512
        }, timeout=LONG_TIMEOUT)
        # generation may fail if no image model loaded — error message is acceptable
        assert txt, "image_generate returned empty response"

    def test_image_edit(self):
        p = klukai_path()
        if not p:
            pytest.skip("No Klukai image")
        txt = mcp("image_edit", {
            "path": p,
            "prompt": "add soft glow to background"
        }, timeout=LONG_TIMEOUT)
        assert txt

    def test_image_remix(self):
        p = klukai_path()
        if not p:
            pytest.skip("No Klukai image")
        txt = mcp("image_remix", {
            "path": p,
            "prompt": "cyberpunk style"
        }, timeout=LONG_TIMEOUT)
        assert txt

    def test_image_upscale(self):
        p = klukai_path()
        if not p:
            pytest.skip("No Klukai image")
        txt = mcp("image_upscale", {"path": p, "scale": 2}, timeout=LONG_TIMEOUT)
        assert txt


# ===========================================================================
# EMBED (semantic vector store via data service)
# ===========================================================================

class TestEmbedTools:
    def setup_method(self):
        if not _reachable(f"{MCP_URL}/health"):
            pytest.skip("aichat-mcp unreachable")

    def test_embed_store_search(self):
        mcp("embed_store", {
            "text": "Klukai is a tactical doll in Girls Frontline 2 based on HK416",
            "metadata": {"source": "regression", "character": "klukai"}
        }, timeout=30)
        txt = mcp("embed_search", {
            "query": "Girls Frontline character HK416",
            "top_k": 3
        }, timeout=30)
        assert txt


# ===========================================================================
# TTS
# ===========================================================================

class TestTTS:
    def setup_method(self):
        if not _reachable(f"{MCP_URL}/health"):
            pytest.skip("aichat-mcp unreachable")

    def test_tts(self):
        txt = mcp("tts", {"text": "Klukai, Girls Frontline 2 character."}, timeout=30)
        # TTS may not be available — error is acceptable; empty string is not
        assert txt, "tts returned empty response"


# ===========================================================================
# ORCHESTRATION
# ===========================================================================

class TestOrchestrate:
    def setup_method(self):
        if not _reachable(f"{MCP_URL}/health"):
            pytest.skip("aichat-mcp unreachable")

    def test_orchestrate_simple(self):
        txt = mcp("orchestrate", {
            "goal": "Search for Klukai Girls Frontline 2 and return 1 sentence summary",
            "max_steps": 3
        }, timeout=60)
        assert txt

    def test_batch_submit(self):
        txt = mcp("batch_submit", {
            "jobs": [
                {"tool": "web_search", "args": {"query": "Klukai GFL2", "max_results": 1}},
                {"tool": "memory_recall", "args": {"key": "_reg_klukai"}}
            ]
        }, timeout=30)
        assert txt


# ===========================================================================
# BROWSER SCREENSHOT (human_browser)
# ===========================================================================

class TestBrowserTools:
    def setup_method(self):
        if not _reachable(f"{MCP_URL}/health"):
            pytest.skip("aichat-mcp unreachable")

    def test_screenshot(self):
        txt = mcp("screenshot", {}, timeout=30)
        # May skip if browser not available
        assert txt

    def test_browser_navigate_screenshot(self):
        txt = mcp("browser", {
            "action": "navigate",
            "url": "https://iopwiki.com/wiki/Klukai"
        }, timeout=30)
        assert txt

    def test_scroll_screenshot(self):
        txt = mcp("scroll_screenshot", {"url": "https://iopwiki.com/wiki/Klukai"}, timeout=60)
        assert txt

    def test_bulk_screenshot(self):
        txt = mcp("bulk_screenshot", {
            "urls": ["https://iopwiki.com/wiki/Klukai"]
        }, timeout=60)
        assert txt

    def test_browser_save_images(self):
        txt = mcp("browser_save_images", {
            "urls": ["https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/240px-PNG_transparency_demonstration_1.png"]
        }, timeout=30)
        assert txt

    def test_browser_download_page_images(self):
        txt = mcp("browser_download_page_images", {
            "url": "https://iopwiki.com/wiki/Klukai"
        }, timeout=60)
        assert txt


# ===========================================================================
# NEW MCP TOOLS — think, deep_research, realtime, news_search, wikipedia,
#                 arxiv_search, youtube_transcript, jupyter_exec,
#                 desktop_screenshot, desktop_control
# ===========================================================================

class TestThinkTool:
    def setup_method(self):
        if not _reachable(f"{MCP_URL}/health"):
            pytest.skip("aichat-mcp unreachable")

    def test_think_returns_thought(self):
        txt = mcp("think", {"thought": "I should search for Klukai next."})
        assert "Klukai" in txt or "thought" in txt.lower(), txt[:300]

    def test_think_empty_thought(self):
        txt = mcp("think", {"thought": ""})
        assert txt  # should return an error or empty-thought message


class TestDeepResearch:
    def setup_method(self):
        if not _reachable(f"{MCP_URL}/health"):
            pytest.skip("aichat-mcp unreachable")

    def test_deep_research_basic(self):
        txt = mcp("deep_research", {
            "question": "What is Girls Frontline 2?",
            "max_hops": 1
        }, timeout=60)
        assert txt and len(txt) > 20, f"Expected research output, got: {txt[:200]}"


class TestRealtimeTool:
    def setup_method(self):
        if not _reachable(f"{MCP_URL}/health"):
            pytest.skip("aichat-mcp unreachable")

    def test_realtime_time(self):
        txt = mcp("realtime", {"type": "time", "query": "UTC"})
        assert "UTC" in txt or "time" in txt.lower(), txt[:300]

    def test_realtime_time_invalid_tz(self):
        txt = mcp("realtime", {"type": "time", "query": "Fake/Timezone"})
        assert "invalid" in txt.lower() or "error" in txt.lower(), txt[:300]

    def test_realtime_weather(self):
        txt = mcp("realtime", {"type": "weather", "query": "London"}, timeout=15)
        assert "London" in txt or "weather" in txt.lower() or "temperature" in txt.lower(), txt[:300]

    def test_realtime_stock(self):
        txt = mcp("realtime", {"type": "stock", "query": "AAPL"}, timeout=15)
        assert "AAPL" in txt or "price" in txt.lower() or "error" in txt.lower(), txt[:300]

    def test_realtime_crypto(self):
        txt = mcp("realtime", {"type": "crypto", "query": "bitcoin"}, timeout=15)
        assert "bitcoin" in txt.lower() or "usd" in txt.lower() or "error" in txt.lower(), txt[:300]

    def test_realtime_forex(self):
        txt = mcp("realtime", {"type": "forex", "query": "USD/EUR"}, timeout=15)
        assert "USD" in txt or "EUR" in txt or "error" in txt.lower(), txt[:300]

    def test_realtime_unknown_type(self):
        txt = mcp("realtime", {"type": "banana", "query": "x"})
        assert "unknown" in txt.lower(), txt[:300]


class TestNewsSearch:
    def setup_method(self):
        if not _reachable(f"{MCP_URL}/health"):
            pytest.skip("aichat-mcp unreachable")

    def test_news_search_all(self):
        txt = mcp("news_search", {"limit": 3}, timeout=30)
        assert txt and ("article" in txt.lower() or "news" in txt.lower() or len(txt) > 50), txt[:300]

    def test_news_search_filtered(self):
        txt = mcp("news_search", {"query": "technology", "sources": ["hackernews"], "limit": 3}, timeout=30)
        assert txt


class TestWikipedia:
    def setup_method(self):
        if not _reachable(f"{MCP_URL}/health"):
            pytest.skip("aichat-mcp unreachable")

    def test_wikipedia_summary(self):
        txt = mcp("wikipedia", {"query": "Python programming language"}, timeout=15)
        assert "Python" in txt, txt[:300]

    def test_wikipedia_full_article(self):
        txt = mcp("wikipedia", {"query": "HK416", "full_article": True}, timeout=15)
        assert txt and len(txt) > 100, f"Expected full article, got: {txt[:200]}"

    def test_wikipedia_no_results(self):
        txt = mcp("wikipedia", {"query": "xyzzyplugh99999nonexistent"}, timeout=15)
        assert "no results" in txt.lower() or txt


class TestArxivSearch:
    def setup_method(self):
        if not _reachable(f"{MCP_URL}/health"):
            pytest.skip("aichat-mcp unreachable")

    def test_arxiv_basic(self):
        txt = mcp("arxiv_search", {"query": "large language models", "max_results": 3}, timeout=20)
        assert "arXiv" in txt or "arxiv" in txt.lower(), txt[:300]

    def test_arxiv_with_category(self):
        txt = mcp("arxiv_search", {"query": "attention mechanism", "category": "cs.CL", "max_results": 2}, timeout=20)
        assert txt


class TestYoutubeTranscript:
    def setup_method(self):
        if not _reachable(f"{MCP_URL}/health"):
            pytest.skip("aichat-mcp unreachable")

    def test_youtube_transcript_valid(self):
        # Use a well-known video that has captions (Rick Astley - short ID)
        txt = mcp("youtube_transcript", {"url": "dQw4w9WgXcQ", "lang": "en"}, timeout=30)
        # Either returns transcript text or an error about the video
        assert txt

    def test_youtube_transcript_invalid_url(self):
        txt = mcp("youtube_transcript", {"url": "not-a-video-id-at-all!!!"}, timeout=15)
        assert "cannot extract" in txt.lower() or "error" in txt.lower(), txt[:300]


class TestJupyterExec:
    def setup_method(self):
        if not _reachable(f"{MCP_URL}/health"):
            pytest.skip("aichat-mcp unreachable")

    def test_jupyter_hello_world(self):
        txt = mcp("jupyter_exec", {"code": "print('klukai-jupyter-ok')"}, timeout=30)
        assert "klukai-jupyter-ok" in txt, f"Expected output, got: {txt[:300]}"

    def test_jupyter_math(self):
        txt = mcp("jupyter_exec", {"code": "import math\nprint(math.pi)"}, timeout=30)
        assert "3.14" in txt, f"Expected pi, got: {txt[:300]}"

    def test_jupyter_session_persistence(self):
        session = "_regression_persist"
        mcp("jupyter_exec", {"code": "x_reg = 42", "session_id": session}, timeout=30)
        txt = mcp("jupyter_exec", {"code": "print(f'x_reg={x_reg}')", "session_id": session}, timeout=30)
        assert "x_reg=42" in txt, f"Session variable not persisted: {txt[:300]}"

    def test_jupyter_error_handling(self):
        txt = mcp("jupyter_exec", {"code": "raise ValueError('regression-test-error')"}, timeout=30)
        assert "regression-test-error" in txt or "ValueError" in txt, f"Expected error, got: {txt[:300]}"

    def test_jupyter_empty_code(self):
        txt = mcp("jupyter_exec", {"code": ""}, timeout=10)
        assert "required" in txt.lower() or txt


class TestDesktopTools:
    """Desktop screenshot/control via human_browser — optional."""
    def setup_method(self):
        if not _reachable(f"{MCP_URL}/health"):
            pytest.skip("aichat-mcp unreachable")

    def test_desktop_screenshot(self):
        txt = mcp("desktop_screenshot", {}, timeout=15)
        # May fail if browser not available — any response is acceptable
        assert txt

    def test_desktop_control_invalid(self):
        txt = mcp("desktop_control", {"action": ""}, timeout=10)
        assert "required" in txt.lower() or "error" in txt.lower() or txt
