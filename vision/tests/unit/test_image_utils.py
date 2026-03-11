"""Unit tests for vision/mcp-server/app/utils/image.py."""
from __future__ import annotations

import asyncio
import base64
import sys
from pathlib import Path

import pytest

# Minimal valid image headers for magic-byte tests
MINIMAL_JPEG = b"\xff\xd8\xff" + b"\x00" * 100
MINIMAL_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
MINIMAL_WEBP_RIFF = b"RIFF" + b"\x00" * 100
MINIMAL_GIF = b"GIF89a" + b"\x00" * 100
MINIMAL_BMP = b"BM" + b"\x00" * 100

_repo_root = Path(__file__).parent.parent.parent.parent  # .../aichat
_mcp_root_str = str(_repo_root / "vision" / "mcp-server")
_router_root_str = str(_repo_root / "vision" / "services" / "vision-router")


# ---------------------------------------------------------------------------
# Helper: build an ImageSource-like object (no full package needed)
# ---------------------------------------------------------------------------

def _make_source(*, url=None, base64_=None, file_path=None):
    """Create a minimal ImageSource stub compatible with load_image_bytes."""
    class _Stub:
        pass
    s = _Stub()
    s.url = url
    s.base64 = base64_
    s.file_path = file_path
    return s


def _run(coro):
    return asyncio.run(coro)


def _load_image_module():
    """Load app.utils.image from mcp-server, handling sys.path ordering."""
    for key in list(sys.modules.keys()):
        if key == "app" or key.startswith("app."):
            del sys.modules[key]

    path_backup = sys.path[:]
    sys.path = [p for p in sys.path if p != _router_root_str]
    if _mcp_root_str not in sys.path:
        sys.path.insert(0, _mcp_root_str)
    try:
        from app.utils.image import (  # type: ignore[import]
            MAX_BYTES,
            _detect_mime,
            _load_from_base64,
            _validate_url,
            load_image_bytes,
        )
        return _detect_mime, _load_from_base64, _validate_url, load_image_bytes, MAX_BYTES
    finally:
        sys.path[:] = path_backup


# ---------------------------------------------------------------------------
# MIME detection tests
# ---------------------------------------------------------------------------

class TestDetectMime:
    def test_jpeg_magic(self):
        _detect_mime, *_ = _load_image_module()
        assert _detect_mime(MINIMAL_JPEG) == "image/jpeg"

    def test_png_magic(self):
        _detect_mime, *_ = _load_image_module()
        assert _detect_mime(MINIMAL_PNG) == "image/png"

    def test_webp_riff_magic(self):
        _detect_mime, *_ = _load_image_module()
        assert _detect_mime(MINIMAL_WEBP_RIFF) == "image/webp"

    def test_gif_magic(self):
        _detect_mime, *_ = _load_image_module()
        assert _detect_mime(MINIMAL_GIF) == "image/gif"

    def test_bmp_magic(self):
        _detect_mime, *_ = _load_image_module()
        assert _detect_mime(MINIMAL_BMP) == "image/bmp"

    def test_unknown_returns_none(self):
        _detect_mime, *_ = _load_image_module()
        assert _detect_mime(b"\x00\x01\x02\x03") is None

    def test_empty_returns_none(self):
        _detect_mime, *_ = _load_image_module()
        assert _detect_mime(b"") is None


# ---------------------------------------------------------------------------
# Base64 loading tests
# ---------------------------------------------------------------------------

class TestLoadFromBase64:
    def test_plain_base64_jpeg(self):
        _, _load_from_base64, _, _, _ = _load_image_module()
        b64 = base64.b64encode(MINIMAL_JPEG).decode()
        raw, mime = _load_from_base64(b64)
        assert raw == MINIMAL_JPEG
        assert mime == "image/jpeg"

    def test_data_uri_prefix_jpeg(self):
        _, _load_from_base64, _, _, _ = _load_image_module()
        b64 = "data:image/jpeg;base64," + base64.b64encode(MINIMAL_JPEG).decode()
        raw, mime = _load_from_base64(b64)
        assert raw == MINIMAL_JPEG
        assert mime == "image/jpeg"

    def test_data_uri_prefix_png(self):
        _, _load_from_base64, _, _, _ = _load_image_module()
        b64 = "data:image/png;base64," + base64.b64encode(MINIMAL_PNG).decode()
        raw, mime = _load_from_base64(b64)
        assert raw == MINIMAL_PNG
        assert mime == "image/png"

    def test_invalid_base64_raises(self):
        _, _load_from_base64, _, _, _ = _load_image_module()
        with pytest.raises(ValueError, match="Invalid base64"):
            _load_from_base64("not-valid-base64!!!!")

    def test_mime_overridden_by_magic_bytes(self):
        _, _load_from_base64, _, _, _ = _load_image_module()
        b64 = "data:image/png;base64," + base64.b64encode(MINIMAL_JPEG).decode()
        raw, mime = _load_from_base64(b64)
        assert mime == "image/jpeg"

    def test_too_large_raises(self):
        _, _load_from_base64, _, _, MAX_BYTES = _load_image_module()
        big = b"\xff\xd8\xff" + b"\x00" * (MAX_BYTES + 1)
        b64 = base64.b64encode(big).decode()
        with pytest.raises(ValueError, match="too large"):
            _load_from_base64(b64)


# ---------------------------------------------------------------------------
# URL validation tests
# ---------------------------------------------------------------------------

class TestValidateUrl:
    def test_valid_https_url(self):
        _, _, _validate_url, _, _ = _load_image_module()
        _validate_url("https://example.com/image.jpg")  # Should not raise

    def test_valid_http_url(self):
        _, _, _validate_url, _, _ = _load_image_module()
        _validate_url("http://example.com/image.jpg")

    def test_blocks_localhost_hostname(self):
        _, _, _validate_url, _, _ = _load_image_module()
        with pytest.raises(ValueError, match="blocked"):
            _validate_url("http://localhost/image.jpg")

    def test_blocks_loopback_ip(self):
        _, _, _validate_url, _, _ = _load_image_module()
        with pytest.raises(ValueError, match="private"):
            _validate_url("http://127.0.0.1/image.jpg")

    def test_blocks_private_ip_10x(self):
        _, _, _validate_url, _, _ = _load_image_module()
        with pytest.raises(ValueError, match="private"):
            _validate_url("http://10.0.0.1/image.jpg")

    def test_blocks_private_ip_192_168(self):
        _, _, _validate_url, _, _ = _load_image_module()
        with pytest.raises(ValueError, match="private"):
            _validate_url("http://192.168.1.100/image.jpg")

    def test_blocks_private_ip_172_16(self):
        _, _, _validate_url, _, _ = _load_image_module()
        with pytest.raises(ValueError, match="private"):
            _validate_url("http://172.16.0.1/image.jpg")

    def test_blocks_ftp_scheme(self):
        _, _, _validate_url, _, _ = _load_image_module()
        with pytest.raises(ValueError, match="http/https"):
            _validate_url("ftp://example.com/image.jpg")

    def test_blocks_file_scheme(self):
        _, _, _validate_url, _, _ = _load_image_module()
        with pytest.raises(ValueError, match="http/https"):
            _validate_url("file:///etc/passwd")

    def test_blocks_metadata_endpoint(self):
        _, _, _validate_url, _, _ = _load_image_module()
        with pytest.raises(ValueError, match="blocked"):
            _validate_url("http://metadata.google.internal/computeMetadata/v1/")


# ---------------------------------------------------------------------------
# File path validation tests (via load_image_bytes)
# ---------------------------------------------------------------------------

class TestLoadFromFile:
    def test_path_outside_allowed_dirs_raises(self):
        _, _, _, load_image_bytes, _ = _load_image_module()
        src = _make_source(file_path="/etc/passwd")
        with pytest.raises(PermissionError, match="outside allowed"):
            _run(load_image_bytes(src))

    def test_path_traversal_raises(self):
        _, _, _, load_image_bytes, _ = _load_image_module()
        src = _make_source(file_path="/workspace/../etc/shadow")
        with pytest.raises(PermissionError, match="outside allowed"):
            _run(load_image_bytes(src))

    def test_nonexistent_allowed_path_raises(self):
        _, _, _, load_image_bytes, _ = _load_image_module()
        src = _make_source(file_path="/tmp/vision_test_nonexistent_12345.jpg")
        with pytest.raises(FileNotFoundError):
            _run(load_image_bytes(src))

    def test_tmp_path_allowed(self):
        _, _, _, load_image_bytes, _ = _load_image_module()
        img_bytes = MINIMAL_JPEG
        target = Path("/tmp/vision_unit_test_load.jpg")
        target.write_bytes(img_bytes)
        try:
            src = _make_source(file_path=str(target))
            raw, mime = _run(load_image_bytes(src))
            assert raw == img_bytes
        finally:
            target.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# load_image_bytes dispatch tests
# ---------------------------------------------------------------------------

class TestLoadImageBytesDispatch:
    def test_dispatches_base64(self):
        _, _, _, load_image_bytes, _ = _load_image_module()
        b64 = base64.b64encode(MINIMAL_JPEG).decode()
        src = _make_source(base64_=b64)
        raw, mime = _run(load_image_bytes(src))
        assert raw == MINIMAL_JPEG
        assert mime == "image/jpeg"

    def test_no_source_raises(self):
        _, _, _, load_image_bytes, _ = _load_image_module()
        src = _make_source()
        with pytest.raises(ValueError, match="No image source"):
            _run(load_image_bytes(src))

    def test_url_disabled_raises(self):
        _, _, _, load_image_bytes, _ = _load_image_module()
        src = _make_source(url="https://example.com/img.jpg")
        with pytest.raises(ValueError, match="URL input is disabled"):
            _run(load_image_bytes(src, allow_url=False))
