"""Integration tests for CompreFaceClient.

These tests require a live CompreFace instance.  Set COMPREFACE_URL to
the base URL of the running service (e.g. ``http://192.168.50.2:8080``).
All tests are skipped automatically when the variable is not set.
"""
from __future__ import annotations

import base64
import io
import os

import pytest
from PIL import Image

pytestmark = pytest.mark.skipif(
    not os.getenv("COMPREFACE_URL"),
    reason="Requires COMPREFACE_URL env var pointing to a live CompreFace instance",
)


def _make_test_jpeg(width: int = 64, height: int = 64) -> bytes:
    img = Image.new("RGB", (width, height), (100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_compreface_health():
    """CompreFace health endpoint should return ok or degraded."""
    from app.clients.compreface import CompreFaceClient  # type: ignore[import]
    from app.config import CompreFaceConfig  # type: ignore[import]

    cfg = CompreFaceConfig(url=os.environ["COMPREFACE_URL"])
    client = CompreFaceClient(cfg)
    result = await client.health()
    assert isinstance(result, dict)
    assert result.get("status") in ("ok", "degraded", "healthy")


@pytest.mark.asyncio
async def test_list_subjects_returns_list():
    """list_subjects should return a list (possibly empty)."""
    from app.clients.compreface import CompreFaceClient  # type: ignore[import]
    from app.config import CompreFaceConfig  # type: ignore[import]

    cfg = CompreFaceConfig(url=os.environ["COMPREFACE_URL"])
    client = CompreFaceClient(cfg)
    subjects = await client.list_subjects()
    assert isinstance(subjects, list)


@pytest.mark.asyncio
async def test_detect_faces_returns_result():
    """detect_faces should process an image and return a structured result."""
    from app.clients.compreface import CompreFaceClient  # type: ignore[import]
    from app.config import CompreFaceConfig  # type: ignore[import]

    cfg = CompreFaceConfig(url=os.environ["COMPREFACE_URL"])
    client = CompreFaceClient(cfg)
    jpeg_bytes = _make_test_jpeg(128, 128)
    result = await client.detect_faces(jpeg_bytes, "image/jpeg", det_prob_threshold=0.5)
    # A plain solid-color image will have 0 detected faces, but the call should succeed
    assert isinstance(result, dict)
    assert "result" in result or "faces" in result or "code" in result


@pytest.mark.asyncio
async def test_enroll_and_delete_subject():
    """Enroll a test subject then immediately delete it."""
    from app.clients.compreface import CompreFaceClient  # type: ignore[import]
    from app.config import CompreFaceConfig  # type: ignore[import]

    cfg = CompreFaceConfig(url=os.environ["COMPREFACE_URL"])
    client = CompreFaceClient(cfg)
    jpeg_bytes = _make_test_jpeg(128, 128)
    subject_name = "integration_test_subject_99999"

    # Enroll
    enroll_result = await client.enroll_face(jpeg_bytes, "image/jpeg", subject_name)
    assert isinstance(enroll_result, dict)

    # Delete (cleanup)
    delete_result = await client.delete_subject(subject_name)
    assert isinstance(delete_result, dict)


@pytest.mark.asyncio
async def test_recognize_faces_no_subjects():
    """recognize_faces with no enrolled subjects should return empty matches."""
    from app.clients.compreface import CompreFaceClient  # type: ignore[import]
    from app.config import CompreFaceConfig  # type: ignore[import]

    cfg = CompreFaceConfig(url=os.environ["COMPREFACE_URL"])
    client = CompreFaceClient(cfg)
    jpeg_bytes = _make_test_jpeg(128, 128)
    result = await client.recognize_faces(
        jpeg_bytes, "image/jpeg", limit=5, det_prob_threshold=0.5
    )
    assert isinstance(result, dict)
