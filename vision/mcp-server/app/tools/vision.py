"""MCP tool handlers for general vision operations."""
from __future__ import annotations

import logging
from typing import Any

from ..backends import get_backend_health
from ..clients.router import VisionRouterClient
from ..config import get_settings
from ..models import (
    AnalyzeImageRequest,
    ClassifyImageRequest,
    DetectClothingRequest,
    DetectObjectsRequest,
    EmbedImageRequest,
    TagImageRequest,
)
from ..utils.image import image_to_base64, load_image_bytes

logger = logging.getLogger(__name__)


def _make_router_client() -> VisionRouterClient:
    return VisionRouterClient(get_settings().router)


async def _triton_available() -> bool:
    """Return True if Triton (via vision-router) is up, respecting the feature flag."""
    settings = get_settings()
    if not settings.enable_triton:
        return False
    return await get_backend_health().triton_ok()


_TRITON_UNAVAILABLE = {"error": "Triton inference backend is unreachable", "code": "UNAVAILABLE"}


async def detect_objects(args: dict[str, Any]) -> dict[str, Any]:
    """Detect and locate objects in an image using YOLOv8."""
    req = DetectObjectsRequest.model_validate(args)
    settings = get_settings()

    if not await _triton_available():
        return _TRITON_UNAVAILABLE

    raw, mime = await load_image_bytes(req.image, allow_url=settings.enable_url_input)
    b64 = image_to_base64(raw, mime)
    client = _make_router_client()
    result = await client.detect_objects(
        b64, mime,
        min_confidence=req.min_confidence,
        max_results=req.max_results,
        classes=req.classes,
    )
    return result.model_dump()


async def classify_image(args: dict[str, Any]) -> dict[str, Any]:
    """Classify an image using EfficientNet-B0 (ImageNet classes)."""
    req = ClassifyImageRequest.model_validate(args)
    settings = get_settings()

    if not await _triton_available():
        return _TRITON_UNAVAILABLE

    raw, mime = await load_image_bytes(req.image, allow_url=settings.enable_url_input)
    b64 = image_to_base64(raw, mime)
    client = _make_router_client()
    result = await client.classify_image(b64, mime, top_k=req.top_k, min_confidence=req.min_confidence)
    return result.model_dump()


async def detect_clothing(args: dict[str, Any]) -> dict[str, Any]:
    """Detect and classify clothing items using FashionCLIP."""
    req = DetectClothingRequest.model_validate(args)
    settings = get_settings()

    if not await _triton_available():
        return _TRITON_UNAVAILABLE

    raw, mime = await load_image_bytes(req.image, allow_url=settings.enable_url_input)
    b64 = image_to_base64(raw, mime)
    client = _make_router_client()
    result = await client.detect_clothing(b64, mime, min_confidence=req.min_confidence, top_k=req.top_k)
    return result.model_dump()


async def embed_image(args: dict[str, Any]) -> dict[str, Any]:
    """Generate a dense embedding vector for an image using CLIP ViT-B/32."""
    req = EmbedImageRequest.model_validate(args)
    settings = get_settings()

    if not settings.enable_embeddings:
        return {"error": "Image embeddings are disabled", "code": "DISABLED"}
    if not await _triton_available():
        return _TRITON_UNAVAILABLE

    raw, mime = await load_image_bytes(req.image, allow_url=settings.enable_url_input)
    b64 = image_to_base64(raw, mime)
    client = _make_router_client()
    result = await client.embed_image(b64, mime, model=req.model, normalize=req.normalize)
    return result.model_dump()


async def tag_image(args: dict[str, Any]) -> dict[str, Any]:
    """Tag an anime/illustration image with Danbooru-style labels using WD ViT Tagger."""
    req = TagImageRequest.model_validate(args)
    settings = get_settings()

    if not await _triton_available():
        return _TRITON_UNAVAILABLE

    raw, mime = await load_image_bytes(req.image, allow_url=settings.enable_url_input)
    b64 = image_to_base64(raw, mime)
    client = _make_router_client()
    result = await client.tag_image(
        b64, mime,
        general_threshold=req.general_threshold,
        character_threshold=req.character_threshold,
    )
    return result.model_dump()


async def analyze_image(args: dict[str, Any]) -> dict[str, Any]:
    """
    Run a multi-model analysis pipeline on an image.

    Combines object detection, classification, optional clothing detection,
    and optional embeddings into a single structured response.
    Suitable for pattern recognition and visual anomaly analysis.
    """
    req = AnalyzeImageRequest.model_validate(args)
    settings = get_settings()

    if not await _triton_available():
        return _TRITON_UNAVAILABLE

    raw, mime = await load_image_bytes(req.image, allow_url=settings.enable_url_input)
    b64 = image_to_base64(raw, mime)
    client = _make_router_client()
    result = await client.analyze(
        b64, mime,
        include_objects=req.include_objects,
        include_classification=req.include_classification,
        include_clothing=req.include_clothing,
        include_embeddings=req.include_embeddings and settings.enable_embeddings,
        object_confidence=req.object_confidence,
        classification_top_k=req.classification_top_k,
    )
    return result.model_dump()
