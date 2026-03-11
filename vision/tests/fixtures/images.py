"""Test image fixture helpers for vision stack tests."""
from __future__ import annotations

import base64
import io

from PIL import Image


def make_test_jpeg(width: int = 64, height: int = 64, color: tuple[int, int, int] = (128, 64, 32)) -> bytes:
    """Create a minimal valid JPEG in memory."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_test_png(width: int = 64, height: int = 64, color: tuple[int, int, int] = (64, 128, 32)) -> bytes:
    """Create a minimal valid PNG in memory."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_test_b64_jpeg(**kwargs: object) -> str:
    """Return a base64-encoded test JPEG string."""
    return base64.b64encode(make_test_jpeg(**kwargs)).decode()  # type: ignore[arg-type]


def make_test_b64_png(**kwargs: object) -> str:
    """Return a base64-encoded test PNG string."""
    return base64.b64encode(make_test_png(**kwargs)).decode()  # type: ignore[arg-type]


# Regression fixture: a person silhouette (solid blue rectangle — stands in for a face)
REGRESSION_FACE_B64: str = make_test_b64_jpeg(width=128, height=128, color=(100, 100, 200))
REGRESSION_SCENE_B64: str = make_test_b64_jpeg(width=640, height=480, color=(50, 100, 50))
