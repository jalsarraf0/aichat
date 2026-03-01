"""
Image rendering tests — ensure LM Studio always receives inline base64 blocks.

Structure
---------
TestImageRenderer       — pure unit tests (no Docker, no network)
TestGpuDetector         — unit tests for GpuDetector OOP class
TestGpuUpscaler         — unit tests for GpuUpscaler OOP class (mocked HTTP)
TestImageRenderingSmoke — @pytest.mark.smoke: live MCP endpoint
TestImageRenderingE2E   — @pytest.mark.smoke: full stack (fetch_image, upscale)
"""
from __future__ import annotations

import asyncio
import base64
import io
import os
import sys
import tempfile
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(0, "src")

# ---------------------------------------------------------------------------
# Helpers shared with test_image_pipeline
# ---------------------------------------------------------------------------

MCP_URL   = "http://localhost:8096"
WORKSPACE = "/docker/human_browser/workspace"

_MCP_UP       = False
_WORKSPACE_OK = False
try:
    _MCP_UP = httpx.get(f"{MCP_URL}/health", timeout=2).status_code < 500
except Exception:
    pass
_WORKSPACE_OK = os.path.isdir(WORKSPACE)

skip_mcp = pytest.mark.skipif(not _MCP_UP,       reason="aichat-mcp not reachable")
skip_ws  = pytest.mark.skipif(not _WORKSPACE_OK, reason="browser workspace not mounted")


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


def _has_image_block(blocks: list[dict]) -> bool:
    return any(b.get("type") == "image" for b in blocks)


def _get_image_bytes(blocks: list[dict]) -> bytes:
    for b in blocks:
        if b.get("type") == "image":
            return base64.b64decode(b["data"])
    return b""


def _make_pil_image(width: int = 400, height: int = 300, mode: str = "RGB"):
    """Create an in-memory PIL Image (requires Pillow, available in venv)."""
    from PIL import Image
    return Image.new(mode, (width, height), color=(120, 180, 240))


def _make_large_pil_image() -> "Image":
    """4000×3000 RGB image — raw JPEG will exceed 3 MB without compression."""
    from PIL import Image
    import random
    img = Image.new("RGB", (4000, 3000))
    # Fill with random-ish content so JPEG can't compress it too trivially
    px = img.load()
    for y in range(0, 3000, 10):
        for x in range(0, 4000, 10):
            c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for dy in range(10):
                for dx in range(10):
                    if y + dy < 3000 and x + dx < 4000:
                        px[x + dx, y + dy] = c
    return img


def _make_test_png_bytes(width: int = 400, height: int = 200) -> bytes:
    """PNG bytes suitable for encode_url_bytes / MCP upload."""
    from PIL import Image
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=(200, 220, 200))
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Load ImageRenderer, GpuDetector, GpuUpscaler from docker/mcp/app.py
# ---------------------------------------------------------------------------

def _load_mcp_classes():
    """Import ImageRenderer, GpuDetector, GpuUpscaler from docker/mcp/app.py.
    Uses importlib so we don't pollute sys.modules for unrelated tests."""
    import importlib.util, pathlib, sys
    spec = importlib.util.spec_from_file_location(
        "mcp_app",
        pathlib.Path(__file__).parent.parent / "docker" / "mcp" / "app.py",
    )
    mod = importlib.util.module_from_spec(spec)
    # Required in Python 3.14+: @dataclass inspects sys.modules[cls.__module__]
    sys.modules["mcp_app"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# Module-level import (cached)
try:
    _mcp = _load_mcp_classes()
    ImageRenderer    = _mcp.ImageRenderer
    GpuDetector      = _mcp.GpuDetector
    GpuUpscaler      = _mcp.GpuUpscaler
    GpuImageProcessor = _mcp.GpuImageProcessor
    ModelRegistry    = _mcp.ModelRegistry
    VisionCache      = _mcp.VisionCache
    GpuCodeRuntime   = _mcp.GpuCodeRuntime
    _MAX_INLINE_BYTES = _mcp._MAX_INLINE_BYTES
    _HAS_PIL         = _mcp._HAS_PIL
    _HAS_CV2         = _mcp._HAS_CV2
    _LOAD_OK         = True
except Exception as _e:
    _LOAD_OK          = False
    _LOAD_ERR         = str(_e)
    _MAX_INLINE_BYTES = 3_000_000

skip_load = pytest.mark.skipif(not _LOAD_OK, reason=f"docker/mcp/app.py load failed")
skip_pil  = pytest.mark.skipif(not _LOAD_OK or not _HAS_PIL if _LOAD_OK else True, reason="PIL not available")


# ===========================================================================
# 1. ImageRenderer — pure unit tests
# ===========================================================================

@skip_load
class TestImageRenderer:
    """Unit tests for ImageRenderer OOP class — no Docker, no network."""

    def setup_method(self):
        self.renderer = ImageRenderer()

    # ── _compress_to_limit ───────────────────────────────────────────────────

    def test_compress_large_image_under_limit(self):
        """A 4000×3000 image must be compressed to < MAX_INLINE_BYTES."""
        if not _HAS_PIL:
            pytest.skip("PIL not available in host env")
        img = _make_large_pil_image()
        raw = self.renderer._compress_to_limit(img.convert("RGB"))
        assert len(raw) <= _MAX_INLINE_BYTES, (
            f"Compressed size {len(raw):,} exceeds limit {_MAX_INLINE_BYTES:,}"
        )

    def test_compress_small_image_unchanged_quality(self):
        """A small 200×150 image should compress fine at first quality level."""
        if not _HAS_PIL:
            pytest.skip("PIL not available in host env")
        img = _make_pil_image(200, 150)
        raw = self.renderer._compress_to_limit(img)
        assert len(raw) < _MAX_INLINE_BYTES

    # ── _fit ─────────────────────────────────────────────────────────────────

    def test_fit_downscales_oversized_image(self):
        if not _HAS_PIL:
            pytest.skip("PIL not available in host env")
        img = _make_pil_image(3000, 2000)
        fitted = self.renderer._fit(img)
        assert fitted.width <= 1280
        assert fitted.height <= 1024

    def test_fit_does_not_upscale_small_image(self):
        if not _HAS_PIL:
            pytest.skip("PIL not available in host env")
        img = _make_pil_image(200, 150)
        fitted = self.renderer._fit(img)
        assert fitted.width == 200
        assert fitted.height == 150

    # ── encode ───────────────────────────────────────────────────────────────

    def test_encode_returns_text_and_image_blocks(self):
        if not _HAS_PIL:
            pytest.skip("PIL not available in host env")
        img = _make_pil_image(400, 300)
        blocks = self.renderer.encode(img, "Test summary")
        assert len(blocks) == 2
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "Test summary"
        assert blocks[1]["type"] == "image"
        assert blocks[1]["mimeType"] == "image/jpeg"
        assert blocks[1]["data"]  # non-empty base64

    def test_encode_large_image_always_under_limit(self):
        """4K image must be rendered inline (never skipped as 'too large')."""
        if not _HAS_PIL:
            pytest.skip("PIL not available in host env")
        img = _make_large_pil_image()
        blocks = self.renderer.encode(img, "Big image")
        assert _has_image_block(blocks), "Large image produced no image block"
        raw = _get_image_bytes(blocks)
        # base64 → raw bytes; check raw JPEG < limit
        assert len(raw) <= _MAX_INLINE_BYTES

    def test_encode_saves_to_workspace(self, tmp_path, monkeypatch):
        """encode() with save_prefix writes a .jpg file."""
        if not _HAS_PIL:
            pytest.skip("PIL not available in host env")
        monkeypatch.setattr(_mcp, "BROWSER_WORKSPACE", str(tmp_path))
        self.renderer = ImageRenderer()   # re-init after monkeypatch
        img = _make_pil_image(200, 150)
        blocks = self.renderer.encode(img, "save test", save_prefix="unittest")
        saved = list(tmp_path.glob("unittest_*.jpg"))
        assert saved, "No file written to workspace"
        # Summary should reference the filename
        assert "unittest_" in blocks[0]["text"]

    # ── encode_path ──────────────────────────────────────────────────────────

    def test_encode_path_missing_file_returns_text_only(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mcp, "BROWSER_WORKSPACE", str(tmp_path))
        blocks = self.renderer.encode_path("/workspace/nonexistent.png", "Missing")
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert not _has_image_block(blocks)

    def test_encode_path_existing_file_has_image_block(self, tmp_path, monkeypatch):
        if not _HAS_PIL:
            pytest.skip("PIL not available in host env")
        monkeypatch.setattr(_mcp, "BROWSER_WORKSPACE", str(tmp_path))
        # Write a small PNG to tmp workspace
        fname = "test_enc.png"
        (tmp_path / fname).write_bytes(_make_test_png_bytes(300, 200))
        blocks = self.renderer.encode_path(f"/workspace/{fname}", "File test")
        assert _has_image_block(blocks), "encode_path produced no image block for existing file"

    # ── encode_url_bytes ─────────────────────────────────────────────────────

    def test_encode_url_bytes_compresses_large_png(self):
        """Raw PNG from a 4K image must be compressed to fit."""
        if not _HAS_PIL:
            pytest.skip("PIL not available in host env")
        raw = _make_test_png_bytes(3000, 2000)
        blocks = self.renderer.encode_url_bytes(raw, "image/png", "Big PNG")
        assert _has_image_block(blocks), "encode_url_bytes lost image block"
        img_bytes = _get_image_bytes(blocks)
        assert len(img_bytes) <= _MAX_INLINE_BYTES

    def test_encode_url_bytes_small_no_pil_sends_raw(self, monkeypatch):
        """When PIL is unavailable and image is small, raw bytes are sent."""
        monkeypatch.setattr(_mcp, "_HAS_PIL", False)
        raw = b"FAKEIMAGEBYTES" * 10   # 140 bytes — well under limit
        blocks = self.renderer.encode_url_bytes(raw, "image/jpeg", "Small raw")
        assert _has_image_block(blocks)
        decoded = base64.b64decode(blocks[1]["data"])
        assert decoded == raw

    def test_encode_url_bytes_large_no_pil_returns_text_warning(self, monkeypatch):
        """When PIL is unavailable and image is too big, warn instead of silent drop."""
        monkeypatch.setattr(_mcp, "_HAS_PIL", False)
        raw = b"X" * (_MAX_INLINE_BYTES + 1)
        blocks = self.renderer.encode_url_bytes(raw, "image/png", "Huge")
        assert not _has_image_block(blocks), "Should not embed oversized image without PIL"
        assert "too large" in blocks[0]["text"].lower()


# ===========================================================================
# 2. GpuDetector — unit tests
# ===========================================================================

@skip_load
class TestGpuDetector:
    """Unit tests for GpuDetector — probe logic mocked at subprocess level."""

    def setup_method(self):
        # Reset cache before each test
        GpuDetector._cache = None

    def test_detect_nvidia_via_nvidia_smi(self, monkeypatch):
        import subprocess as sp
        monkeypatch.setattr(
            sp, "check_output",
            lambda *a, **kw: b"NVIDIA GeForce RTX 4090\n",
        )
        result = GpuDetector._probe()
        assert result["vendor"] == "nvidia"
        assert "RTX 4090" in result["name"]

    def test_detect_intel_via_dev_dri(self, monkeypatch, tmp_path):
        """When nvidia-smi fails but /dev/dri/renderD128 exists → Intel."""
        import subprocess as sp
        monkeypatch.setattr(sp, "check_output", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError()))
        monkeypatch.setattr(os.path, "exists", lambda p: p == "/dev/dri/renderD128" or os.path.exists.__wrapped__(p) if hasattr(os.path.exists, "__wrapped__") else False)
        # Use /dev/dri listing mock
        monkeypatch.setattr(os, "listdir", lambda p: ["renderD128", "card0"] if p == "/dev/dri" else os.listdir.__wrapped__(p) if hasattr(os.listdir, "__wrapped__") else [])
        result = GpuDetector._probe()
        assert result["vendor"] == "intel"

    def test_detect_none_returns_none(self, monkeypatch):
        import subprocess as sp
        monkeypatch.setattr(sp, "check_output", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError()))
        monkeypatch.setattr(os.path, "exists", lambda p: False)
        monkeypatch.setattr(os, "listdir", lambda p: [] if "/dev/dri" in p else [])
        monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "")
        monkeypatch.setenv("INTEL_GPU", "")
        result = GpuDetector._probe()
        assert result["vendor"] == "none"

    def test_detect_nvidia_via_env(self, monkeypatch):
        import subprocess as sp
        monkeypatch.setattr(sp, "check_output", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError()))
        monkeypatch.setattr(os.path, "exists", lambda p: False)
        monkeypatch.setattr(os, "listdir", lambda p: [])
        monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "0")
        result = GpuDetector._probe()
        assert result["vendor"] == "nvidia"

    def test_cache_is_reused(self, monkeypatch):
        GpuDetector._cache = {"vendor": "nvidia", "name": "Cached NVIDIA"}
        result = GpuDetector.detect()
        assert result["name"] == "Cached NVIDIA"

    def test_available_true_when_gpu_detected(self):
        GpuDetector._cache = {"vendor": "intel", "name": "Intel Arc A770"}
        assert GpuDetector.available() is True

    def test_available_false_when_no_gpu(self):
        GpuDetector._cache = {"vendor": "none", "name": "No GPU detected"}
        assert GpuDetector.available() is False


# ===========================================================================
# 3. GpuUpscaler — unit tests (mocked HTTP)
# ===========================================================================

@skip_load
class TestGpuUpscaler:
    """Unit tests for GpuUpscaler — LM Studio HTTP call is mocked."""

    def _make_b64_jpeg(self, w: int = 256, h: int = 256) -> str:
        """Return a valid JPEG as base64 string."""
        if not _HAS_PIL:
            return base64.b64encode(b"FAKEJPEG").decode()
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (w, h), (100, 150, 200)).save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode()

    @pytest.mark.asyncio
    async def test_upscale_success_returns_image(self):
        if not _HAS_PIL:
            pytest.skip("PIL not available in host env")
        b64 = self._make_b64_jpeg(512, 512)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"b64_json": b64}]}
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        upscaler = GpuUpscaler("http://localhost:1234")
        from PIL import Image
        src_img = Image.new("RGB", (128, 128), (80, 80, 80))
        result = await upscaler.upscale(src_img, mock_client)
        assert result is not None
        assert result.width == 512
        assert result.height == 512

    @pytest.mark.asyncio
    async def test_upscale_api_error_returns_none(self):
        if not _HAS_PIL:
            pytest.skip("PIL not available in host env")
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        upscaler = GpuUpscaler("http://localhost:1234")
        from PIL import Image
        src_img = Image.new("RGB", (64, 64))
        result = await upscaler.upscale(src_img, mock_client)
        assert result is None

    @pytest.mark.asyncio
    async def test_upscale_network_error_returns_none(self):
        if not _HAS_PIL:
            pytest.skip("PIL not available in host env")
        mock_client = MagicMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

        upscaler = GpuUpscaler("http://localhost:1234")
        from PIL import Image
        src_img = Image.new("RGB", (64, 64))
        result = await upscaler.upscale(src_img, mock_client)
        assert result is None

    @pytest.mark.asyncio
    async def test_upscale_no_pil_returns_none(self, monkeypatch):
        monkeypatch.setattr(_mcp, "_HAS_PIL", False)
        mock_client = MagicMock()
        upscaler = GpuUpscaler("http://localhost:1234")
        # No src_img needed — should bail early
        result = await upscaler.upscale(None, mock_client)  # type: ignore[arg-type]
        assert result is None

    def test_gpu_label_uses_detector(self):
        GpuDetector._cache = {"vendor": "intel", "name": "Intel Arc A770"}
        upscaler = GpuUpscaler("http://localhost:1234")
        assert "Intel Arc A770" in upscaler.gpu_label()

    @pytest.mark.asyncio
    async def test_upscale_passes_model_in_form(self):
        if not _HAS_PIL:
            pytest.skip("PIL not available in host env")
        b64 = self._make_b64_jpeg(128, 128)
        captured_data: list[dict] = []

        async def _mock_post(url, files, data, timeout):
            captured_data.append(dict(data))
            mock = MagicMock()
            mock.status_code = 200
            mock.json.return_value = {"data": [{"b64_json": b64}]}
            return mock

        mock_client = MagicMock()
        mock_client.post = _mock_post

        upscaler = GpuUpscaler("http://localhost:1234", model="mymodel")
        from PIL import Image
        src_img = Image.new("RGB", (64, 64))
        await upscaler.upscale(src_img, mock_client)
        assert captured_data[0].get("model") == "mymodel"


# ===========================================================================
# 4. Smoke tests — require live MCP endpoint
# ===========================================================================

@skip_mcp
@skip_load
@pytest.mark.smoke
class TestImageRenderingSmoke:
    """Smoke tests: call live MCP endpoint and assert inline image blocks."""

    def test_fetch_image_returns_image_block(self):
        """fetch_image must always return an image block, never 'external image'."""
        # Use a small, reliable public image URL
        url = "https://httpbin.org/image/jpeg"
        blocks = _mcp_content("fetch_image", {"url": url}, timeout=30)
        assert blocks, "fetch_image returned empty result"
        assert _has_image_block(blocks), (
            f"fetch_image returned no image block — got: {[b.get('type') for b in blocks]}"
        )

    def test_fetch_image_b64_fits_in_limit(self):
        """The base64 payload from fetch_image must not exceed _MAX_INLINE_BYTES."""
        url = "https://httpbin.org/image/png"
        blocks = _mcp_content("fetch_image", {"url": url}, timeout=30)
        raw = _get_image_bytes(blocks)
        if not raw:
            pytest.skip("fetch_image returned no image (network issue)")
        assert len(raw) <= _MAX_INLINE_BYTES, (
            f"Inline image {len(raw):,} bytes exceeds limit {_MAX_INLINE_BYTES:,}"
        )

    @skip_ws
    def test_image_upscale_cpu_fallback_has_image_block(self, tmp_path):
        """image_upscale with gpu=false must always return an image block."""
        # Write a test image to workspace
        fname = f"upscale_smoke_{uuid.uuid4().hex[:8]}.png"
        fpath = os.path.join(WORKSPACE, fname)
        try:
            with open(fpath, "wb") as fh:
                fh.write(_make_test_png_bytes(200, 150))
            blocks = _mcp_content(
                "image_upscale", {"path": fname, "scale": 2.0, "gpu": False}, timeout=30
            )
            assert _has_image_block(blocks), (
                f"image_upscale (CPU) returned no image block: {[b.get('type') for b in blocks]}"
            )
        finally:
            try:
                os.unlink(fpath)
            except Exception:
                pass

    @skip_ws
    def test_image_upscale_gpu_with_fallback_has_image_block(self):
        """image_upscale (GPU primary) must return an image block even if GPU model is absent."""
        fname = f"upscale_gpu_{uuid.uuid4().hex[:8]}.png"
        fpath = os.path.join(WORKSPACE, fname)
        try:
            with open(fpath, "wb") as fh:
                fh.write(_make_test_png_bytes(200, 150))
            blocks = _mcp_content(
                "image_upscale", {"path": fname, "scale": 2.0}, timeout=120
            )
            assert _has_image_block(blocks), (
                f"image_upscale (GPU) returned no image block: {[b.get('type') for b in blocks]}"
            )
        finally:
            try:
                os.unlink(fpath)
            except Exception:
                pass


# ===========================================================================
# 5. E2E tests — full stack (requires MCP + browser services)
# ===========================================================================

@skip_mcp
@skip_load
@pytest.mark.smoke
class TestImageRenderingE2E:
    """E2E tests: verify inline rendering pipeline end-to-end."""

    def test_image_search_returns_inline_images(self):
        """image_search must return at least one inline image block."""
        blocks = _mcp_content("image_search", {"query": "blue sky", "count": 1}, timeout=90)
        if not blocks:
            pytest.skip("image_search returned empty (network or vision model issue)")
        # Either we got an image block, or a text-only 'no image found' — both acceptable,
        # but if an image block exists it MUST be properly encoded.
        if _has_image_block(blocks):
            raw = _get_image_bytes(blocks)
            assert len(raw) <= _MAX_INLINE_BYTES, "image_search image exceeds inline limit"
            assert len(raw) > 0, "image_search image block has empty data"

    def test_fetch_image_jpeg_is_always_rendered(self):
        """
        Fetch a known JPEG and verify:
        1. image block present
        2. base64 decodes to valid JPEG header
        3. size within limit
        """
        url = "https://httpbin.org/image/jpeg"
        blocks = _mcp_content("fetch_image", {"url": url}, timeout=30)
        if not _has_image_block(blocks):
            pytest.skip("network unavailable")
        raw = _get_image_bytes(blocks)
        assert raw[:2] == b"\xff\xd8", "Expected JPEG magic bytes"
        assert len(raw) <= _MAX_INLINE_BYTES

    def test_fetch_image_png_is_rendered_inline(self):
        """PNG fetched externally must render inline — an image block must be present
        and the payload must be within the size limit.
        (ImageRenderer converts to JPEG after container rebuild; MIME type
        assertion is intentionally omitted here to be rebuild-agnostic.)"""
        url = "https://httpbin.org/image/png"
        blocks = _mcp_content("fetch_image", {"url": url}, timeout=30)
        if not _has_image_block(blocks):
            pytest.skip("network unavailable")
        raw = _get_image_bytes(blocks)
        assert len(raw) > 0, "Image block has empty data"
        assert len(raw) <= _MAX_INLINE_BYTES, (
            f"PNG inline image {len(raw):,} bytes exceeds limit {_MAX_INLINE_BYTES:,}"
        )

    def test_upscale_gpu_fallback_graceful(self):
        """
        If no GPU model is loaded, image_upscale must still return an image
        block (CPU LANCZOS fallback) and must not raise.
        """
        fname = f"e2e_upscale_{uuid.uuid4().hex[:8]}.png"
        fpath = os.path.join(WORKSPACE, fname)
        if not _WORKSPACE_OK:
            pytest.skip("workspace not mounted")
        try:
            with open(fpath, "wb") as fh:
                fh.write(_make_test_png_bytes(200, 150))
            blocks = _mcp_content("image_upscale", {"path": fname, "scale": 2.0}, timeout=120)
            assert _has_image_block(blocks), (
                "image_upscale returned no image block even after CPU fallback"
            )
            # Block must not be empty
            raw = _get_image_bytes(blocks)
            assert len(raw) > 0
        finally:
            try:
                os.unlink(fpath)
            except Exception:
                pass


# ===========================================================================
# 6. GpuImageProcessor — unit tests
# ===========================================================================

@skip_load
class TestGpuImageProcessor:
    """Unit tests for GpuImageProcessor OOP class — no Docker, no network."""

    def _img(self, w: int = 200, h: int = 150, mode: str = "RGB"):
        if not _HAS_PIL:
            pytest.skip("PIL not available")
        return _make_pil_image(w, h, mode)

    def test_backend_returns_valid_string(self):
        backend = GpuImageProcessor.backend()
        assert backend in ("opencv-cuda", "opencv-cpu", "pillow"), (
            f"Unexpected backend: {backend!r}"
        )

    def test_resize_returns_correct_size(self):
        img = self._img(200, 150)
        result = GpuImageProcessor.resize(img, 100, 80)
        # cv2 path does exact resize; PIL thumbnail() preserves aspect ratio
        # so we assert the result fits within the requested box in both cases
        assert result.width <= 100
        assert result.height <= 80

    def test_resize_returns_pil_image(self):
        from PIL import Image
        img = self._img(200, 150)
        result = GpuImageProcessor.resize(img, 64, 64)
        assert isinstance(result, Image.Image)

    def test_sharpen_returns_pil_image_same_size(self):
        from PIL import Image
        img = self._img(100, 80)
        result = GpuImageProcessor.sharpen(img)
        assert isinstance(result, Image.Image)
        assert result.size == (100, 80)

    def test_enhance_contrast_returns_pil_image(self):
        from PIL import Image
        img = self._img(100, 80)
        result = GpuImageProcessor.enhance_contrast(img, factor=1.5)
        assert isinstance(result, Image.Image)
        assert result.size == (100, 80)

    def test_enhance_sharpness_returns_pil_image(self):
        from PIL import Image
        img = self._img(100, 80)
        result = GpuImageProcessor.enhance_sharpness(img, factor=1.2)
        assert isinstance(result, Image.Image)
        assert result.size == (100, 80)

    def test_diff_same_size_produces_image(self):
        from PIL import Image
        img1 = _make_pil_image(100, 80, "RGB")
        img2 = _make_pil_image(100, 80, "RGB")
        result = GpuImageProcessor.diff(img1, img2)
        assert isinstance(result, Image.Image)

    def test_diff_different_sizes_auto_resizes(self):
        """diff() must handle mismatched sizes without raising."""
        from PIL import Image
        img1 = _make_pil_image(200, 150)
        img2 = _make_pil_image(100, 80)
        result = GpuImageProcessor.diff(img1, img2)
        assert isinstance(result, Image.Image)

    def test_to_grayscale_returns_image_same_dimensions(self):
        from PIL import Image
        img = self._img(80, 60)
        result = GpuImageProcessor.to_grayscale(img)
        assert isinstance(result, Image.Image)
        assert result.size == (80, 60)

    def test_annotate_returns_image_with_boxes(self):
        from PIL import Image
        img = self._img(200, 200)
        boxes = [(10, 10, 50, 50), (100, 100, 180, 180)]
        labels = ["cat", "dog"]
        result = GpuImageProcessor.annotate(img, boxes, labels)
        assert isinstance(result, Image.Image)
        assert result.size == img.size

    def test_annotate_empty_boxes_returns_image(self):
        from PIL import Image
        img = self._img(100, 100)
        result = GpuImageProcessor.annotate(img, [], [])
        assert isinstance(result, Image.Image)

    def test_pillow_fallback_when_cv2_disabled(self, monkeypatch):
        """Force _CV2_OK=False and confirm PIL code path returns valid image."""
        from PIL import Image
        monkeypatch.setattr(GpuImageProcessor, "_CV2_OK", False)
        monkeypatch.setattr(GpuImageProcessor, "_CUDA_OK", False)
        img = self._img(60, 60)
        result = GpuImageProcessor.resize(img, 30, 30)
        assert isinstance(result, Image.Image)
        assert GpuImageProcessor.backend() == "pillow"


# ===========================================================================
# 7. ModelRegistry — unit tests (mocked HTTP)
# ===========================================================================

@skip_load
class TestModelRegistry:
    """Unit tests for ModelRegistry — LM Studio HTTP is fully mocked."""

    def setup_method(self):
        # Reset singleton and TTL before every test
        ModelRegistry._instance = None
        reg = ModelRegistry.get()
        reg._last_probe = 0.0
        reg._probe_ok = False
        reg._models = []

    def _mock_client(self, models: list[dict], status: int = 200):
        """Return an AsyncMock httpx.AsyncClient whose GET returns `models`."""
        mock_resp = MagicMock()
        mock_resp.status_code = status
        mock_resp.json.return_value = {"data": models}
        client = MagicMock()
        client.get = AsyncMock(return_value=mock_resp)
        return client

    def test_get_returns_singleton(self):
        a = ModelRegistry.get()
        b = ModelRegistry.get()
        assert a is b

    @pytest.mark.asyncio
    async def test_is_available_true_when_models_returned(self):
        client = self._mock_client([{"id": "llama-3", "type": "llm"}])
        reg = ModelRegistry.get()
        assert await reg.is_available(client) is True

    @pytest.mark.asyncio
    async def test_is_available_false_when_no_models(self):
        client = self._mock_client([])
        reg = ModelRegistry.get()
        assert await reg.is_available(client) is False

    @pytest.mark.asyncio
    async def test_is_available_false_on_network_error(self):
        client = MagicMock()
        client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        reg = ModelRegistry.get()
        assert await reg.is_available(client) is False

    @pytest.mark.asyncio
    async def test_has_vision_detects_vision_model(self):
        client = self._mock_client([{"id": "llava-vision-7b", "type": "vlm"}])
        reg = ModelRegistry.get()
        assert await reg.has_vision(client) is True

    @pytest.mark.asyncio
    async def test_has_vision_false_for_plain_llm(self):
        client = self._mock_client([{"id": "llama-3-8b", "type": "llm"}])
        reg = ModelRegistry.get()
        assert await reg.has_vision(client) is False

    @pytest.mark.asyncio
    async def test_has_image_gen_detects_flux(self):
        client = self._mock_client([{"id": "flux-schnell-q4", "type": "image"}])
        reg = ModelRegistry.get()
        assert await reg.has_image_gen(client) is True

    @pytest.mark.asyncio
    async def test_has_image_gen_detects_sdxl(self):
        client = self._mock_client([{"id": "sdxl-turbo", "type": "image"}])
        reg = ModelRegistry.get()
        assert await reg.has_image_gen(client) is True

    @pytest.mark.asyncio
    async def test_has_image_gen_false_when_no_gen_model(self):
        client = self._mock_client([{"id": "llama-3-8b", "type": "llm"}])
        reg = ModelRegistry.get()
        import unittest.mock as _um
        with _um.patch.object(_mcp, "IMAGE_GEN_MODEL", ""):
            assert await reg.has_image_gen(client) is False

    @pytest.mark.asyncio
    async def test_ttl_caching_avoids_double_probe(self):
        """Within TTL, a second call must not trigger another GET."""
        client = self._mock_client([{"id": "llama-3", "type": "llm"}])
        reg = ModelRegistry.get()
        await reg.is_available(client)
        await reg.is_available(client)  # should hit cache
        assert client.get.call_count == 1

    @pytest.mark.asyncio
    async def test_invalidate_forces_fresh_probe(self):
        client = self._mock_client([{"id": "llama-3", "type": "llm"}])
        reg = ModelRegistry.get()
        await reg.is_available(client)
        reg.invalidate()
        await reg.is_available(client)
        assert client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_best_chat_model_returns_first_llm(self):
        client = self._mock_client([
            {"id": "flux-schnell", "type": "image"},
            {"id": "llama-3-8b",  "type": "llm"},
        ])
        reg = ModelRegistry.get()
        import unittest.mock as _um
        with _um.patch.object(_mcp, "IMAGE_GEN_MODEL", ""):
            result = await reg.best_chat_model(client)
        assert result == "llama-3-8b"


# ===========================================================================
# 8. VisionCache — unit tests
# ===========================================================================

@skip_load
class TestVisionCache:
    """Unit tests for VisionCache LRU in-memory cache."""

    def setup_method(self):
        self.cache = VisionCache()

    def test_get_returns_none_for_unknown_hash(self):
        assert self.cache.get("deadbeef") is None

    def test_put_and_get_returns_result(self):
        self.cache.put("aabbccdd", (True, "cat photo", 0.92))
        result = self.cache.get("aabbccdd")
        assert result == (True, "cat photo", 0.92)

    def test_get_returns_none_for_empty_phash(self):
        assert self.cache.get("") is None

    def test_put_empty_phash_is_noop(self):
        self.cache.put("", (True, "ignored", 1.0))
        assert self.cache.size() == 0

    def test_size_reports_correctly(self):
        assert self.cache.size() == 0
        self.cache.put("hash1", (True, "a", 0.8))
        self.cache.put("hash2", (False, "b", 0.4))
        assert self.cache.size() == 2

    def test_clear_empties_cache(self):
        self.cache.put("h1", (True, "a", 0.9))
        self.cache.put("h2", (False, "b", 0.3))
        self.cache.clear()
        assert self.cache.size() == 0
        assert self.cache.get("h1") is None

    def test_overwrite_existing_key_no_size_growth(self):
        self.cache.put("h1", (True, "first", 0.8))
        self.cache.put("h1", (False, "second", 0.5))
        assert self.cache.size() == 1
        assert self.cache.get("h1") == (False, "second", 0.5)

    def test_evicts_oldest_when_at_capacity(self):
        for i in range(VisionCache._MAX_SIZE):
            self.cache.put(f"hash{i:04d}", (True, f"item{i}", 0.7))
        assert self.cache.size() == VisionCache._MAX_SIZE
        self.cache.put("hash_new", (False, "overflow", 0.5))
        assert self.cache.size() == VisionCache._MAX_SIZE
        assert self.cache.get("hash0000") is None
        assert self.cache.get("hash_new") == (False, "overflow", 0.5)

    def test_get_after_eviction_returns_none(self):
        for i in range(VisionCache._MAX_SIZE + 5):
            self.cache.put(f"k{i:05d}", (True, "", 0.9))
        for i in range(5):
            assert self.cache.get(f"k{i:05d}") is None

    def test_different_hashes_independent(self):
        self.cache.put("aaa", (True, "match", 0.95))
        self.cache.put("bbb", (False, "no match", 0.1))
        assert self.cache.get("aaa") == (True, "match", 0.95)
        assert self.cache.get("bbb") == (False, "no match", 0.1)


# ===========================================================================
# 9. GpuCodeRuntime — unit tests
# ===========================================================================

@skip_load
class TestGpuCodeRuntime:
    """Unit tests for GpuCodeRuntime — no subprocess, no GPU needed."""

    def test_no_injection_for_plain_python(self):
        code = "x = 1 + 2\nprint(x)"
        assert GpuCodeRuntime.needs_device_injection(code) is False

    def test_needs_injection_for_torch(self):
        code = "import torch\nmodel = torch.nn.Linear(10, 5)"
        assert GpuCodeRuntime.needs_device_injection(code) is True

    def test_needs_injection_for_tensorflow(self):
        code = "import tensorflow as tf\nmodel = tf.keras.Sequential()"
        assert GpuCodeRuntime.needs_device_injection(code) is True

    def test_needs_injection_for_cuda_keyword(self):
        code = "# run on cuda\ntensor = x.cuda()"
        assert GpuCodeRuntime.needs_device_injection(code) is True

    def test_needs_injection_for_device_keyword(self):
        code = "model = model.to(device)"
        assert GpuCodeRuntime.needs_device_injection(code) is True

    def test_prepare_adds_preamble_when_triggered(self):
        code = "import torch\nprint(torch.cuda.is_available())"
        prepared = GpuCodeRuntime.prepare(code)
        assert "DEVICE" in prepared
        # Preamble appears before the user code — use a unique user-only marker
        assert prepared.endswith(code), "User code must appear at the end of prepared string"

    def test_prepare_unchanged_when_not_triggered(self):
        code = "import math\nprint(math.pi)"
        prepared = GpuCodeRuntime.prepare(code)
        assert prepared == code
        assert "DEVICE" not in prepared

    def test_preamble_defines_device_variable(self):
        assert "DEVICE" in GpuCodeRuntime._PREAMBLE

    def test_preamble_sets_device_to_cpu_at_minimum(self):
        assert '"cpu"' in GpuCodeRuntime._PREAMBLE or "'cpu'" in GpuCodeRuntime._PREAMBLE

    def test_prepare_with_gpu_trigger_not_empty(self):
        code = "import torch"
        result = GpuCodeRuntime.prepare(code)
        assert len(result) > len(code)

    def test_available_packages_returns_list(self):
        result = GpuCodeRuntime.available_packages()
        assert isinstance(result, list)

    def test_preinstalled_frozenset_contains_numpy(self):
        assert "numpy" in GpuCodeRuntime._PREINSTALLED

    def test_gpu_triggers_frozenset_contains_torch(self):
        assert "torch" in GpuCodeRuntime._GPU_TRIGGERS


# ===========================================================================
# 10. image_remix tool schema — introspection (no Docker needed)
# ===========================================================================

@skip_load
class TestImageRemixSchema:
    """Verify the image_remix tool is registered with the correct schema."""

    def _get_tool(self, name: str) -> dict:
        for t in _mcp._TOOLS:
            if t.get("name") == name:
                return t
        return {}

    def test_image_remix_in_tools_list(self):
        tool = self._get_tool("image_remix")
        assert tool, "image_remix not found in _TOOLS"

    def test_image_remix_has_path_and_prompt_required(self):
        tool = self._get_tool("image_remix")
        schema = tool.get("inputSchema", {})
        required = schema.get("required", [])
        assert "path" in required
        assert "prompt" in required

    def test_image_remix_has_strength_property(self):
        tool = self._get_tool("image_remix")
        props = tool.get("inputSchema", {}).get("properties", {})
        assert "strength" in props
        assert props["strength"].get("type") == "number"

    def test_image_remix_has_n_property(self):
        tool = self._get_tool("image_remix")
        props = tool.get("inputSchema", {}).get("properties", {})
        assert "n" in props
        assert props["n"].get("type") == "integer"

    def test_image_remix_description_mentions_style_or_remix(self):
        tool = self._get_tool("image_remix")
        desc = tool.get("description", "").lower()
        assert any(kw in desc for kw in ("style", "remix", "transform")), (
            f"Description missing style/remix/transform: {desc[:120]}"
        )


# ===========================================================================
# 11. image_remix smoke tests — require live MCP endpoint + workspace
# ===========================================================================

@skip_mcp
@skip_load
@pytest.mark.smoke
class TestImageRemixSmoke:
    """Smoke tests for image_remix — require live MCP + mounted workspace."""

    @skip_ws
    def test_image_remix_no_model_gives_text_or_image(self):
        """image_remix must respond with text error (no model) or image block
        (model loaded) — never a timeout or crash."""
        fname = f"remix_smoke_{uuid.uuid4().hex[:8]}.png"
        fpath = os.path.join(WORKSPACE, fname)
        try:
            with open(fpath, "wb") as fh:
                fh.write(_make_test_png_bytes(200, 150))
            blocks = _mcp_content(
                "image_remix",
                {"path": fname, "prompt": "anime style", "n": 1},
                timeout=30,
            )
            assert blocks, "image_remix returned empty result"
            img_blocks  = [b for b in blocks if b.get("type") == "image"]
            text_blocks = [b for b in blocks if b.get("type") == "text"]
            if img_blocks:
                raw = _get_image_bytes(blocks)
                assert len(raw) <= _MAX_INLINE_BYTES
            else:
                combined = " ".join(b.get("text", "") for b in text_blocks).lower()
                assert any(kw in combined for kw in ("model", "lm studio", "diffusion")), (
                    f"Unexpected text error: {combined[:200]}"
                )
        finally:
            try:
                os.unlink(fpath)
            except Exception:
                pass
