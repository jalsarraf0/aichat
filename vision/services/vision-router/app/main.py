"""Vision Router FastAPI application.

Provides REST endpoints that bridge the vision MCP server to the Triton
Inference Server.  Each request preprocesses the image, dispatches
inference asynchronously to Triton, then postprocesses and returns a
structured response.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from .config import RouterConfig, get_config
from .inference import TritonInferenceClient
from .models import (
    AnalyzeRequest,
    AnalyzeResponse,
    ClassifyRequest,
    ClassifyResponse,
    DetectClothingRequest,
    DetectClothingResponse,
    DetectObjectsRequest,
    DetectObjectsResponse,
    EmbedRequest,
    EmbedResponse,
    TagRequest,
    TagResponse,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level metrics store
# ---------------------------------------------------------------------------

_metrics: dict[str, Any] = {
    "request_count": defaultdict(int),   # endpoint → count
    "error_count": defaultdict(int),     # endpoint → count
    "total_inference_ms": defaultdict(float),  # endpoint → sum
}

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def _record(endpoint: str, inference_ms: float, error: bool = False) -> None:
    """Update in-memory metrics for ``endpoint``."""
    _metrics["request_count"][endpoint] += 1
    _metrics["total_inference_ms"][endpoint] += inference_ms
    if error:
        _metrics["error_count"][endpoint] += 1


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Verify Triton connectivity on startup; clean up on shutdown."""
    cfg: RouterConfig = get_config()
    logging.basicConfig(
        level=cfg.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger.info("Vision Router starting up — Triton URL: %s", cfg.triton.url)

    client = TritonInferenceClient(url=cfg.triton.url, timeout=cfg.triton.timeout_s)

    # Probe Triton health endpoint
    triton_ok = False
    try:
        async with httpx.AsyncClient(timeout=10.0) as hc:
            resp = await hc.get(f"{cfg.triton.url}/v2/health/ready")
        triton_ok = resp.status_code == 200
    except Exception as exc:
        logger.warning("Triton health check failed at startup: %s", exc)

    if triton_ok:
        logger.info("Triton is ready.")
    else:
        logger.warning(
            "Triton is not ready at startup — endpoints will return 503 until it recovers."
        )

    app.state.triton_client = client
    app.state.config = cfg

    yield

    logger.info("Vision Router shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Vision Router",
    description=(
        "Bridges the aichat vision MCP server to NVIDIA Triton Inference Server. "
        "Supports object detection (YOLOv8), image classification (EfficientNet-B0), "
        "image embedding (CLIP ViT-B/32), and clothing detection (FashionCLIP)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

async def verify_api_key(
    request: Request,
    api_key: str | None = Depends(_API_KEY_HEADER),
) -> None:
    """Validate the X-API-Key header when the router is configured with a key.

    Raises:
        HTTPException 401: If a key is configured and the header is missing or wrong.
    """
    cfg: RouterConfig = request.app.state.config
    if not cfg.api_key:
        return  # auth disabled
    if api_key != cfg.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


# ---------------------------------------------------------------------------
# Triton client accessor
# ---------------------------------------------------------------------------

def _client(request: Request) -> TritonInferenceClient:
    return request.app.state.triton_client


# ---------------------------------------------------------------------------
# POST /v1/detect-objects
# ---------------------------------------------------------------------------

@app.post(
    "/v1/detect-objects",
    response_model=DetectObjectsResponse,
    summary="Object detection with YOLOv8",
    dependencies=[Depends(verify_api_key)],
)
async def detect_objects(
    body: DetectObjectsRequest,
    request: Request,
) -> DetectObjectsResponse:
    """Detect objects in the provided image using the YOLOv8n model.

    Returns bounding boxes, class labels, and per-detection confidence scores.
    """
    cfg: RouterConfig = request.app.state.config
    endpoint = "detect-objects"

    try:
        detections, inf_ms = await _client(request).detect_objects(
            image_b64=body.image_b64,
            model_name=cfg.yolo_model,
            confidence_threshold=body.confidence_threshold,
            nms_threshold=body.nms_threshold,
            max_detections=body.max_detections,
        )
    except ValueError as exc:
        _record(endpoint, 0.0, error=True)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:
        _record(endpoint, 0.0, error=True)
        logger.exception("Triton error in detect-objects: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Inference failed: {exc}",
        ) from exc

    _record(endpoint, inf_ms)

    from .models import BoundingBox, DetectedObject  # local import to avoid circular
    objects = [
        DetectedObject(
            label=d["label"],
            class_id=d["class_id"],
            confidence=d["confidence"],
            box=BoundingBox(**d["box"]),
        )
        for d in detections
    ]
    return DetectObjectsResponse(
        objects=objects,
        count=len(objects),
        model=cfg.yolo_model,
        inference_ms=inf_ms,
    )


# ---------------------------------------------------------------------------
# POST /v1/classify
# ---------------------------------------------------------------------------

@app.post(
    "/v1/classify",
    response_model=ClassifyResponse,
    summary="Image classification with EfficientNet-B0",
    dependencies=[Depends(verify_api_key)],
)
async def classify_image(
    body: ClassifyRequest,
    request: Request,
) -> ClassifyResponse:
    """Classify the provided image using EfficientNet-B0 (ImageNet 1000 classes).

    Returns the top-k predicted labels sorted by confidence descending.
    """
    cfg: RouterConfig = request.app.state.config
    endpoint = "classify"

    try:
        labels, inf_ms = await _client(request).classify_image(
            image_b64=body.image_b64,
            model_name=cfg.efficientnet_model,
            top_k=body.top_k,
        )
    except ValueError as exc:
        _record(endpoint, 0.0, error=True)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:
        _record(endpoint, 0.0, error=True)
        logger.exception("Triton error in classify: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Inference failed: {exc}",
        ) from exc

    _record(endpoint, inf_ms)

    from .models import TopLabel
    return ClassifyResponse(
        labels=[TopLabel(**lbl) for lbl in labels],
        model=cfg.efficientnet_model,
        inference_ms=inf_ms,
    )


# ---------------------------------------------------------------------------
# POST /v1/detect-clothing
# ---------------------------------------------------------------------------

@app.post(
    "/v1/detect-clothing",
    response_model=DetectClothingResponse,
    summary="Clothing detection with FashionCLIP",
    dependencies=[Depends(verify_api_key)],
)
async def detect_clothing(
    body: DetectClothingRequest,
    request: Request,
) -> DetectClothingResponse:
    """Detect clothing and fashion items using zero-shot FashionCLIP inference.

    Returns detected clothing categories with confidence scores.
    """
    cfg: RouterConfig = request.app.state.config
    endpoint = "detect-clothing"

    try:
        items, inf_ms = await _client(request).detect_clothing(
            image_b64=body.image_b64,
            model_name=cfg.fashion_clip_model,
            confidence_threshold=body.confidence_threshold,
        )
    except ValueError as exc:
        _record(endpoint, 0.0, error=True)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:
        _record(endpoint, 0.0, error=True)
        logger.exception("Triton error in detect-clothing: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Inference failed: {exc}",
        ) from exc

    _record(endpoint, inf_ms)

    from .models import ClothingItem
    clothing_items = [
        ClothingItem(
            category=it["category"],
            confidence=it["confidence"],
            box=it["box"],
        )
        for it in items
    ]
    return DetectClothingResponse(
        items=clothing_items,
        count=len(clothing_items),
        model=cfg.fashion_clip_model,
        inference_ms=inf_ms,
    )


# ---------------------------------------------------------------------------
# POST /v1/embed
# ---------------------------------------------------------------------------

@app.post(
    "/v1/embed",
    response_model=EmbedResponse,
    summary="Image embedding with CLIP ViT-B/32",
    dependencies=[Depends(verify_api_key)],
)
async def embed_image(
    body: EmbedRequest,
    request: Request,
) -> EmbedResponse:
    """Generate a 512-dimensional CLIP image embedding.

    The embedding can be used for similarity search, clustering, or as
    features for downstream models.
    """
    cfg: RouterConfig = request.app.state.config
    endpoint = "embed"

    try:
        embedding, inf_ms = await _client(request).embed_image(
            image_b64=body.image_b64,
            model_name=cfg.clip_model,
            normalize=body.normalize,
        )
    except ValueError as exc:
        _record(endpoint, 0.0, error=True)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:
        _record(endpoint, 0.0, error=True)
        logger.exception("Triton error in embed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Inference failed: {exc}",
        ) from exc

    _record(endpoint, inf_ms)

    return EmbedResponse(
        embedding=embedding,
        dimension=len(embedding),
        model=cfg.clip_model,
        inference_ms=inf_ms,
    )


# ---------------------------------------------------------------------------
# POST /v1/analyze
# ---------------------------------------------------------------------------

@app.post(
    "/v1/analyze",
    response_model=AnalyzeResponse,
    summary="Multi-task image analysis",
    dependencies=[Depends(verify_api_key)],
)
async def analyze_image(
    body: AnalyzeRequest,
    request: Request,
) -> AnalyzeResponse:
    """Run multiple vision tasks concurrently on a single image.

    Tasks that are requested run in parallel via ``asyncio.gather``.
    The response includes results only for the tasks that were requested.
    """
    cfg: RouterConfig = request.app.state.config
    endpoint = "analyze"
    t_wall = time.perf_counter()

    client = _client(request)
    tasks: list[asyncio.Task[Any]] = []
    task_keys: list[str] = []

    if body.include_objects:
        tasks.append(
            asyncio.ensure_future(
                client.detect_objects(
                    image_b64=body.image_b64,
                    model_name=cfg.yolo_model,
                    confidence_threshold=body.confidence_threshold,
                )
            )
        )
        task_keys.append("objects")

    if body.include_labels:
        tasks.append(
            asyncio.ensure_future(
                client.classify_image(
                    image_b64=body.image_b64,
                    model_name=cfg.efficientnet_model,
                    top_k=body.top_k,
                )
            )
        )
        task_keys.append("labels")

    if body.include_embedding:
        tasks.append(
            asyncio.ensure_future(
                client.embed_image(
                    image_b64=body.image_b64,
                    model_name=cfg.clip_model,
                    normalize=True,
                )
            )
        )
        task_keys.append("embedding")

    if not tasks:
        return AnalyzeResponse(inference_ms=0.0)

    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as exc:
        _record(endpoint, 0.0, error=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Inference failed: {exc}",
        ) from exc

    wall_ms = (time.perf_counter() - t_wall) * 1000.0

    from .models import BoundingBox, ClothingItem, DetectedObject, TopLabel

    objects_out = None
    labels_out = None
    embedding_out = None
    model_versions: dict[str, str] = {}
    has_error = False

    for key, result in zip(task_keys, results):
        if isinstance(result, Exception):
            logger.error("Task '%s' failed in /v1/analyze: %s", key, result)
            has_error = True
            continue

        if key == "objects":
            detections, _ = result
            objects_out = [
                DetectedObject(
                    label=d["label"],
                    class_id=d["class_id"],
                    confidence=d["confidence"],
                    box=BoundingBox(**d["box"]),
                )
                for d in detections
            ]
            model_versions[cfg.yolo_model] = ""

        elif key == "labels":
            label_dicts, _ = result
            labels_out = [TopLabel(**lbl) for lbl in label_dicts]
            model_versions[cfg.efficientnet_model] = ""

        elif key == "embedding":
            vec, _ = result
            embedding_out = vec
            model_versions[cfg.clip_model] = ""

    if has_error and objects_out is None and labels_out is None and embedding_out is None:
        _record(endpoint, wall_ms, error=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="All requested inference tasks failed.",
        )

    _record(endpoint, wall_ms, error=has_error)

    return AnalyzeResponse(
        objects=objects_out,
        labels=labels_out,
        embedding=embedding_out,
        model_versions=model_versions,
        inference_ms=wall_ms,
    )


# ---------------------------------------------------------------------------
# POST /v1/tag
# ---------------------------------------------------------------------------

@app.post(
    "/v1/tag",
    response_model=TagResponse,
    summary="Anime/illustration tagging with WD ViT Tagger",
    dependencies=[Depends(verify_api_key)],
)
async def tag_image(
    body: TagRequest,
    request: Request,
) -> TagResponse:
    """Tag an image with Danbooru-style labels using WD ViT Large Tagger v3.

    Returns three buckets: character names, general content/style tags, and
    rating (safe / questionable / explicit).  Useful for anime and illustration
    content where face-recognition models do not apply.
    """
    cfg: RouterConfig = request.app.state.config
    endpoint = "tag"

    try:
        buckets, inf_ms = await _client(request).tag_image(
            image_b64=body.image_b64,
            model_name=cfg.wd_tagger_model,
            general_threshold=body.general_threshold,
            character_threshold=body.character_threshold,
        )
    except ValueError as exc:
        _record(endpoint, 0.0, error=True)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except RuntimeError as exc:
        _record(endpoint, 0.0, error=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception as exc:
        _record(endpoint, 0.0, error=True)
        logger.exception("Triton error in tag: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Inference failed: {exc}",
        ) from exc

    _record(endpoint, inf_ms)

    from .models import Tag
    return TagResponse(
        characters=[Tag(**t) for t in buckets["characters"]],
        general=[Tag(**t) for t in buckets["general"]],
        rating=[Tag(**t) for t in buckets["rating"]],
        model=cfg.wd_tagger_model,
        inference_ms=inf_ms,
    )


# ---------------------------------------------------------------------------
# GET /v1/health
# ---------------------------------------------------------------------------

@app.get(
    "/v1/health",
    summary="Health check",
)
async def health(request: Request) -> JSONResponse:
    """Return service health and per-model readiness status."""
    cfg: RouterConfig = request.app.state.config
    client: TritonInferenceClient = _client(request)

    model_names = [
        cfg.yolo_model,
        cfg.efficientnet_model,
        cfg.clip_model,
        cfg.fashion_clip_model,
        cfg.wd_tagger_model,
    ]

    readiness = await asyncio.gather(
        *[client.check_model_ready(m) for m in model_names],
        return_exceptions=True,
    )

    models_status: dict[str, str] = {}
    triton_status = "ok"
    for name, ready in zip(model_names, readiness):
        if isinstance(ready, Exception):
            models_status[name] = "error"
            triton_status = "degraded"
        elif ready:
            models_status[name] = "ready"
        else:
            models_status[name] = "not_ready"
            triton_status = "degraded"

    http_status = status.HTTP_200_OK if triton_status == "ok" else status.HTTP_206_PARTIAL_CONTENT

    return JSONResponse(
        status_code=http_status,
        content={
            "status": "ok",
            "triton": triton_status,
            "models": models_status,
        },
    )


# ---------------------------------------------------------------------------
# GET /v1/metrics
# ---------------------------------------------------------------------------

@app.get(
    "/v1/metrics",
    summary="Request metrics",
)
async def metrics() -> JSONResponse:
    """Return aggregated request counts, error counts, and average latency."""
    result: dict[str, Any] = {}
    all_endpoints = set(_metrics["request_count"].keys()) | set(_metrics["error_count"].keys())

    for ep in sorted(all_endpoints):
        req_count = _metrics["request_count"][ep]
        err_count = _metrics["error_count"][ep]
        total_ms = _metrics["total_inference_ms"][ep]
        avg_ms = total_ms / req_count if req_count > 0 else 0.0
        result[ep] = {
            "request_count": req_count,
            "error_count": err_count,
            "avg_inference_ms": round(avg_ms, 2),
        }

    return JSONResponse(content={"endpoints": result})


# ---------------------------------------------------------------------------
# Generic exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for unexpected server errors."""
    logger.exception("Unhandled exception on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error."},
    )
