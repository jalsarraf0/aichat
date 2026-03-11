"""Image loading utilities for vision MCP server."""
from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from ..models import ImageSource

logger = logging.getLogger(__name__)

# Maximum dimensions before we downscale (safety valve)
MAX_DIMENSION = 4096
# Supported MIME types
SUPPORTED_MIMES = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"}
# URL fetch timeout
URL_TIMEOUT = 20.0
# Max size 25MB raw
MAX_BYTES = 25 * 1024 * 1024


async def load_image_bytes(source: "ImageSource", allow_url: bool = True) -> tuple[bytes, str]:
    """
    Load image bytes from a source (url, base64, or file_path).

    Returns:
        (bytes, mime_type) tuple

    Raises:
        ValueError: If source is invalid or image cannot be loaded
        PermissionError: If file path is outside allowed directories
    """
    if source.base64:
        return _load_from_base64(source.base64)
    elif source.file_path:
        return await _load_from_file(source.file_path)
    elif source.url:
        if not allow_url:
            raise ValueError("URL input is disabled in server configuration")
        return await _load_from_url(source.url)
    else:
        raise ValueError("No image source provided")


def _load_from_base64(data: str) -> tuple[bytes, str]:
    """Decode base64 image data."""
    # Strip data URI prefix if present
    if "," in data and data.startswith("data:"):
        header, data = data.split(",", 1)
        mime = header.split(":")[1].split(";")[0].lower()
    else:
        mime = "image/jpeg"  # assume JPEG if no header

    try:
        raw = base64.b64decode(data)
    except Exception as exc:
        raise ValueError(f"Invalid base64 data: {exc}") from exc

    if len(raw) > MAX_BYTES:
        raise ValueError(f"Image too large: {len(raw)//1024//1024}MB (max {MAX_BYTES//1024//1024}MB)")

    detected_mime = _detect_mime(raw)
    if detected_mime:
        mime = detected_mime

    return raw, mime


async def _load_from_file(path: str) -> tuple[bytes, str]:
    """Load image from a local file path (blocking I/O run in executor)."""
    p = Path(path).resolve()

    # Safety: only allow absolute paths within /workspace or /tmp
    allowed_prefixes = [Path("/workspace"), Path("/tmp"), Path("/data")]
    if not any(str(p).startswith(str(prefix)) for prefix in allowed_prefixes):
        raise PermissionError(
            f"File path {path!r} is outside allowed directories "
            f"({', '.join(str(x) for x in allowed_prefixes)})"
        )

    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not p.is_file():
        raise ValueError(f"Path is not a file: {path}")

    stat = p.stat()
    if stat.st_size > MAX_BYTES:
        raise ValueError(f"File too large: {stat.st_size//1024//1024}MB")

    raw = await asyncio.get_event_loop().run_in_executor(None, p.read_bytes)
    mime = _detect_mime(raw) or "image/jpeg"
    return raw, mime


async def _load_from_url(url: str) -> tuple[bytes, str]:
    """Fetch image from a URL with safety checks."""
    _validate_url(url)

    headers = {
        "User-Agent": "VisionMCP/1.0 (image-fetch)",
        "Accept": "image/*,*/*;q=0.8",
    }

    try:
        async with httpx.AsyncClient(timeout=URL_TIMEOUT, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise ValueError(f"URL fetch failed: HTTP {exc.response.status_code}") from exc
    except httpx.RequestError as exc:
        raise ValueError(f"URL fetch error: {exc}") from exc

    raw = response.content
    if len(raw) > MAX_BYTES:
        raise ValueError(f"Remote image too large: {len(raw)//1024//1024}MB")

    content_type = response.headers.get("content-type", "").split(";")[0].lower().strip()
    if not content_type.startswith("image/"):
        detected = _detect_mime(raw)
        if not detected:
            raise ValueError(f"URL did not return an image (content-type: {content_type!r})")
        content_type = detected

    return raw, content_type


def _validate_url(url: str) -> None:
    """Validate URL safety."""
    import ipaddress
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Only http/https URLs are allowed (got {parsed.scheme!r})")

    hostname = parsed.hostname or ""

    # Block private/loopback addresses
    try:
        addr = ipaddress.ip_address(hostname)
        if addr.is_loopback or addr.is_private or addr.is_link_local:
            raise ValueError(f"URL hostname resolves to a private address: {hostname}")
    except ValueError as exc:
        if "private" in str(exc) or "loopback" in str(exc):
            raise
        # Hostname is a DNS name, not an IP — that's fine

    blocked_hosts = {"localhost", "metadata.google.internal"}
    if hostname.lower() in blocked_hosts:
        raise ValueError(f"URL hostname is blocked: {hostname}")


def _detect_mime(raw: bytes) -> str | None:
    """Detect image MIME type from magic bytes."""
    if raw[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw[:4] in (b"RIFF", b"WEBP"):
        return "image/webp"
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if raw[:2] == b"BM":
        return "image/bmp"
    return None


def image_to_base64(raw: bytes) -> str:
    """Encode image bytes as base64."""
    return base64.b64encode(raw).decode("ascii")
