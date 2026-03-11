"""MCP tool handlers for face operations."""
from __future__ import annotations

import logging
from typing import Any

from ..backends import get_backend_health
from ..clients.compreface import CompreFaceClient
from ..config import get_settings
from ..models import (
    DeleteSubjectResult,
    DetectFacesRequest,
    EnrollFaceRequest,
    RecognizeFaceRequest,
    VerifyFaceRequest,
)
from ..utils.image import load_image_bytes

logger = logging.getLogger(__name__)


async def _resolve_compreface_client() -> CompreFaceClient:
    """Return a CompreFaceClient pointed at whichever backend is available.

    Tries the primary RTX host first. If unreachable and fallback is enabled,
    returns a client pointed at the local CPU-mode CompreFace container.
    """
    settings = get_settings()
    health = get_backend_health()

    if await health.compreface_primary_ok():
        return CompreFaceClient(settings.compreface)

    if settings.fallback.enabled:
        logger.info("CompreFace primary down — routing to local fallback")
        return CompreFaceClient(settings.fallback_compreface_config())

    # Primary down, fallback disabled — let the caller handle the error.
    raise RuntimeError(
        "Primary CompreFace is unreachable and local fallback is disabled "
        "(set VISION_FALLBACK__ENABLED=true to enable it)"
    )


async def recognize_face(args: dict[str, Any]) -> dict[str, Any]:
    """Identify faces in an image against enrolled subjects."""
    req = RecognizeFaceRequest.model_validate(args)
    settings = get_settings()

    if not settings.enable_compreface:
        return {"error": "Face recognition is disabled", "code": "DISABLED"}

    raw, mime = await load_image_bytes(req.image, allow_url=settings.enable_url_input)
    client = await _resolve_compreface_client()
    result = await client.recognize(
        raw, mime,
        limit=req.limit,
        det_prob_threshold=settings.compreface.det_prob_threshold,
        min_confidence=req.min_confidence,
    )
    return result.model_dump()


async def verify_face(args: dict[str, Any]) -> dict[str, Any]:
    """Verify if two images contain the same person."""
    req = VerifyFaceRequest.model_validate(args)
    settings = get_settings()

    if not settings.enable_compreface:
        return {"error": "Face verification is disabled", "code": "DISABLED"}

    raw_a, mime_a = await load_image_bytes(req.image_a, allow_url=settings.enable_url_input)
    raw_b, mime_b = await load_image_bytes(req.image_b, allow_url=settings.enable_url_input)
    client = await _resolve_compreface_client()
    result = await client.verify(raw_a, mime_a, raw_b, mime_b)
    return result.model_dump()


async def detect_faces(args: dict[str, Any]) -> dict[str, Any]:
    """Detect all faces in an image without recognition."""
    req = DetectFacesRequest.model_validate(args)
    settings = get_settings()

    if not settings.enable_compreface:
        return {"error": "Face detection is disabled", "code": "DISABLED"}

    raw, mime = await load_image_bytes(req.image, allow_url=settings.enable_url_input)
    client = await _resolve_compreface_client()
    result = await client.detect(
        raw, mime,
        det_prob_threshold=req.min_confidence,
        return_landmarks=req.return_landmarks,
    )
    return result.model_dump()


async def enroll_face(args: dict[str, Any]) -> dict[str, Any]:
    """Enroll a face under a named subject for future recognition."""
    req = EnrollFaceRequest.model_validate(args)
    settings = get_settings()

    if not settings.enable_compreface:
        return {"error": "Face enrollment is disabled", "code": "DISABLED"}

    raw, mime = await load_image_bytes(req.image, allow_url=settings.enable_url_input)
    client = await _resolve_compreface_client()
    result = await client.enroll(raw, mime, req.subject_name)
    return result.model_dump()


async def list_face_subjects(args: dict[str, Any]) -> dict[str, Any]:
    """List all enrolled face subjects."""
    settings = get_settings()
    if not settings.enable_compreface:
        return {"error": "CompreFace is disabled", "code": "DISABLED"}
    client = await _resolve_compreface_client()
    result = await client.list_subjects()
    return result.model_dump()


async def delete_face_subject(args: dict[str, Any]) -> dict[str, Any]:
    """Delete all faces for a subject."""
    subject = args.get("subject_name", "")
    if not subject:
        return {"error": "subject_name is required", "code": "VALIDATION_ERROR"}

    settings = get_settings()
    if not settings.enable_compreface:
        return {"error": "CompreFace is disabled", "code": "DISABLED"}

    client = await _resolve_compreface_client()
    deleted = await client.delete_subject(subject)
    from urllib.parse import urlparse

    from ..models import BackendInfo
    result = DeleteSubjectResult(
        subject=subject,
        deleted=deleted,
        backend=BackendInfo(name="compreface", host=urlparse(settings.compreface.url).netloc),
    )
    return result.model_dump()
