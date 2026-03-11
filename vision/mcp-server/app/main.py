"""Vision MCP Server — FastAPI application entry point.

Exposes the MCP JSON-RPC 2.0 protocol over:
  - POST /mcp   (streamable-HTTP transport, preferred)
  - GET  /sse   (legacy SSE transport)
  - GET  /health
  - GET  /metrics
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .backends import get_backend_health
from .clients.compreface import CompreFaceClient
from .clients.router import VisionRouterClient
from .config import get_settings
from .tools.face import (
    delete_face_subject,
    detect_faces,
    enroll_face,
    list_face_subjects,
    recognize_face,
    verify_face,
)
from .tools.vision import (
    analyze_image,
    classify_image,
    detect_clothing,
    detect_objects,
    embed_image,
    tag_image,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-8s %(name)-30s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOLS: dict[str, dict[str, Any]] = {
    "recognize_face": {
        "name": "recognize_face",
        "description": (
            "Identify faces in an image by comparing them against enrolled subjects in CompreFace. "
            "Returns matching subjects with similarity scores, bounding boxes, and optional attributes "
            "(age, gender, emotion)."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["image"],
            "properties": {
                "image": {
                    "type": "object",
                    "description": "Input image. Provide exactly one of: url, base64, or file_path.",
                    "properties": {
                        "url": {"type": "string", "description": "Public URL of the image"},
                        "base64": {"type": "string", "description": "Base64-encoded image bytes"},
                        "file_path": {"type": "string", "description": "Server-local absolute file path"},
                    },
                },
                "subject_filter": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Restrict recognition to these subjects only",
                },
                "min_confidence": {
                    "type": "number",
                    "default": 0.7,
                    "description": "Minimum similarity score to return a match (0-1)",
                },
                "limit": {
                    "type": "integer",
                    "default": 5,
                    "description": "Maximum number of matches per detected face",
                },
            },
        },
    },
    "verify_face": {
        "name": "verify_face",
        "description": (
            "Verify whether two images contain the same person. "
            "Returns a boolean verdict and similarity score."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["image_a", "image_b"],
            "properties": {
                "image_a": {"type": "object", "description": "First image (same format as recognize_face.image)"},
                "image_b": {"type": "object", "description": "Second image"},
                "min_similarity": {"type": "number", "default": 0.85, "description": "Similarity threshold for verified=true"},
            },
        },
    },
    "detect_faces": {
        "name": "detect_faces",
        "description": (
            "Detect all faces in an image without recognition. "
            "Returns bounding boxes and optional attributes (age, gender, emotion, landmarks)."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["image"],
            "properties": {
                "image": {"type": "object", "description": "Input image"},
                "min_confidence": {"type": "number", "default": 0.7},
                "return_landmarks": {"type": "boolean", "default": False},
            },
        },
    },
    "enroll_face": {
        "name": "enroll_face",
        "description": "Enroll a face image under a named subject for future recognition.",
        "inputSchema": {
            "type": "object",
            "required": ["image", "subject_name"],
            "properties": {
                "image": {"type": "object", "description": "Image containing a clear frontal face"},
                "subject_name": {"type": "string", "description": "Name to enroll this face under"},
            },
        },
    },
    "list_face_subjects": {
        "name": "list_face_subjects",
        "description": "List all enrolled face subjects in CompreFace.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    "delete_face_subject": {
        "name": "delete_face_subject",
        "description": "Delete all face images for a subject from CompreFace.",
        "inputSchema": {
            "type": "object",
            "required": ["subject_name"],
            "properties": {
                "subject_name": {"type": "string", "description": "Subject to delete"},
            },
        },
    },
    "detect_objects": {
        "name": "detect_objects",
        "description": (
            "Detect and locate objects in an image using YOLOv8n on the RTX 3090. "
            "Returns labels, confidence scores, and bounding boxes for up to 80 COCO classes."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["image"],
            "properties": {
                "image": {"type": "object", "description": "Input image"},
                "min_confidence": {"type": "number", "default": 0.4},
                "max_results": {"type": "integer", "default": 20},
                "classes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter to these class names only",
                },
            },
        },
    },
    "classify_image": {
        "name": "classify_image",
        "description": (
            "Classify an image into ImageNet categories using EfficientNet-B0 on the RTX 3090. "
            "Returns top-k labels with confidence scores."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["image"],
            "properties": {
                "image": {"type": "object", "description": "Input image"},
                "top_k": {"type": "integer", "default": 5},
                "min_confidence": {"type": "number", "default": 0.01},
            },
        },
    },
    "detect_clothing": {
        "name": "detect_clothing",
        "description": (
            "Detect and classify clothing items in an image using FashionCLIP on the RTX 3090. "
            "Returns clothing categories (e.g. jacket, jeans, sneakers) with confidence scores and optional color attributes."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["image"],
            "properties": {
                "image": {"type": "object", "description": "Input image"},
                "min_confidence": {"type": "number", "default": 0.15},
                "top_k": {"type": "integer", "default": 5},
            },
        },
    },
    "embed_image": {
        "name": "embed_image",
        "description": (
            "Generate a dense embedding vector for an image using CLIP ViT-B/32 on the RTX 3090. "
            "Useful for similarity search, clustering, and downstream ML tasks."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["image"],
            "properties": {
                "image": {"type": "object", "description": "Input image"},
                "model": {"type": "string", "default": "clip_vit_b32"},
                "normalize": {"type": "boolean", "default": True},
            },
        },
    },
    "analyze_image": {
        "name": "analyze_image",
        "description": (
            "Run a full multi-model vision analysis pipeline on an image. "
            "Combines object detection, image classification, optional clothing detection, "
            "and optional embeddings into one structured response. Suitable for visual pattern recognition "
            "and anomaly-friendly image analysis."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["image"],
            "properties": {
                "image": {"type": "object", "description": "Input image"},
                "include_objects": {"type": "boolean", "default": True},
                "include_classification": {"type": "boolean", "default": True},
                "include_clothing": {"type": "boolean", "default": False},
                "include_embeddings": {"type": "boolean", "default": False},
                "object_confidence": {"type": "number", "default": 0.4},
                "classification_top_k": {"type": "integer", "default": 3},
            },
        },
    },
    "tag_image": {
        "name": "tag_image",
        "description": (
            "Tag an anime or illustration image with Danbooru-style labels using WD ViT Large Tagger v3. "
            "Returns three buckets: character names (e.g. 'klukai_(girls'_frontline_2)'), "
            "general content/style tags, and rating (safe/questionable/explicit). "
            "Use this instead of face recognition for anime-style artwork."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["image"],
            "properties": {
                "image": {"type": "object", "description": "Input image. Provide exactly one of: url, base64, or file_path.",
                    "properties": {
                        "url": {"type": "string", "description": "Public URL of the image"},
                        "base64": {"type": "string", "description": "Base64-encoded image bytes"},
                        "file_path": {"type": "string", "description": "Server-local absolute file path"},
                    },
                },
                "general_threshold": {"type": "number", "default": 0.35, "description": "Minimum confidence for general tags (0–1)"},
                "character_threshold": {"type": "number", "default": 0.85, "description": "Minimum confidence for character tags (0–1)"},
            },
        },
    },
}

TOOL_HANDLERS: dict[str, Any] = {
    "recognize_face": recognize_face,
    "verify_face": verify_face,
    "detect_faces": detect_faces,
    "enroll_face": enroll_face,
    "list_face_subjects": list_face_subjects,
    "delete_face_subject": delete_face_subject,
    "detect_objects": detect_objects,
    "classify_image": classify_image,
    "detect_clothing": detect_clothing,
    "embed_image": embed_image,
    "analyze_image": analyze_image,
    "tag_image": tag_image,
}

# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger.info("=== Vision MCP Server starting ===")
    logger.info("CompreFace: %s (enabled=%s)", settings.compreface.url, settings.enable_compreface)
    logger.info("VisionRouter: %s (enabled=%s)", settings.router.url, settings.enable_triton)
    logger.info("Tools: %d registered", len(TOOLS))

    # Startup diagnostics
    if settings.enable_compreface:
        cf = CompreFaceClient(settings.compreface)
        h = await cf.health()
        logger.info("CompreFace health: %s", h)

    if settings.enable_triton:
        vr = VisionRouterClient(settings.router)
        h = await vr.health()
        logger.info("VisionRouter health: %s", h)

    logger.info("=== Vision MCP Server ready ===")
    yield
    logger.info("Vision MCP Server shutting down")


app = FastAPI(
    title="Vision MCP Server",
    version="1.0.0",
    description="GPU-accelerated vision tools via MCP protocol",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.server.cors_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# MCP JSON-RPC dispatcher
# ---------------------------------------------------------------------------

async def _dispatch(method: str, params: dict[str, Any], request_id: str) -> dict[str, Any]:
    """Dispatch a JSON-RPC method to the appropriate handler."""
    if method == "initialize":
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "vision-mcp", "version": "1.0.0"},
        }

    if method == "tools/list":
        return {"tools": list(TOOLS.values())}

    if method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})

        handler = TOOL_HANDLERS.get(tool_name)
        if not handler:
            raise ValueError(f"Unknown tool: {tool_name!r}")

        t0 = time.perf_counter()
        try:
            result = await handler(tool_args)
        except (ValueError, TypeError) as exc:
            return {
                "content": [{"type": "text", "text": json.dumps({"error": str(exc), "code": "VALIDATION_ERROR"})}],
                "isError": True,
            }
        except RuntimeError as exc:
            logger.error("Tool %s runtime error: %s", tool_name, exc)
            return {
                "content": [{"type": "text", "text": json.dumps({"error": str(exc), "code": "BACKEND_ERROR"})}],
                "isError": True,
            }
        except Exception as exc:
            logger.exception("Tool %s unexpected error", tool_name)
            return {
                "content": [{"type": "text", "text": json.dumps({"error": "Internal server error", "code": "INTERNAL_ERROR"})}],
                "isError": True,
            }

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("tool=%s request_id=%s elapsed_ms=%.1f", tool_name, request_id, elapsed_ms)
        return {"content": [{"type": "text", "text": json.dumps(result, default=str)}]}

    if method == "ping":
        return {}

    raise ValueError(f"Unknown method: {method!r}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> JSONResponse:
    settings = get_settings()
    bh = get_backend_health()
    checks: dict[str, Any] = {"status": "ok", "tools": len(TOOLS)}

    if settings.enable_compreface:
        cf_primary_ok = await bh.compreface_primary_ok()
        checks["compreface_primary"] = "ok" if cf_primary_ok else "unreachable"
        if not cf_primary_ok and settings.fallback.enabled:
            cf_local = CompreFaceClient(settings.fallback_compreface_config())
            checks["compreface_fallback"] = await cf_local.health()
        elif cf_primary_ok:
            cf = CompreFaceClient(settings.compreface)
            checks["compreface"] = await cf.health()

    if settings.enable_triton:
        triton_ok = await bh.triton_ok()
        checks["vision_router"] = "ok" if triton_ok else "unreachable"

    # Degraded if any enabled backend is down with no fallback covering it.
    cf_degraded = (
        settings.enable_compreface
        and not await bh.compreface_primary_ok()
        and not settings.fallback.enabled
    )
    triton_degraded = settings.enable_triton and not await bh.triton_ok()
    checks["status"] = "degraded" if (cf_degraded or triton_degraded) else "ok"
    return JSONResponse(checks)


@app.get("/metrics")
async def metrics() -> JSONResponse:
    """Minimal Prometheus-compatible text metrics."""
    return JSONResponse({
        "tools_registered": len(TOOLS),
        "server": "vision-mcp/1.0.0",
    })


@app.post("/mcp")
async def mcp_http(request: Request) -> Response:
    """Streamable HTTP MCP transport (preferred)."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}, status_code=400)

    rpc_id = body.get("id")
    method = body.get("method", "")
    params = body.get("params", {})

    if body.get("jsonrpc") != "2.0":
        return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32600, "message": "Invalid Request"}}, status_code=400)

    try:
        result = await _dispatch(method, params, request_id)
        return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "result": result})
    except ValueError as exc:
        return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32601, "message": str(exc)}}, status_code=404)
    except Exception as exc:
        logger.exception("MCP dispatch error for method=%s", method)
        return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32603, "message": "Internal error"}}, status_code=500)


# SSE session store (minimal, for tool calls over SSE)
_sse_sessions: dict[str, asyncio.Queue] = {}


@app.get("/sse")
async def mcp_sse(request: Request) -> StreamingResponse:
    """Legacy SSE MCP transport."""
    session_id = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    _sse_sessions[session_id] = queue

    async def event_stream() -> AsyncIterator[str]:
        # Send endpoint event
        yield f"event: endpoint\ndata: /messages?sessionId={session_id}\n\n"
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(msg)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            _sse_sessions.pop(session_id, None)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/messages")
async def mcp_sse_messages(request: Request) -> JSONResponse:
    """SSE message endpoint."""
    session_id = request.query_params.get("sessionId", "")
    queue = _sse_sessions.get(session_id)
    if not queue:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Parse error"}, status_code=400)

    rpc_id = body.get("id")
    method = body.get("method", "")
    params = body.get("params", {})
    request_id = str(uuid.uuid4())

    try:
        result = await _dispatch(method, params, request_id)
        await queue.put({"jsonrpc": "2.0", "id": rpc_id, "result": result})
    except Exception as exc:
        await queue.put({"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32603, "message": str(exc)}})

    return JSONResponse({}, status_code=202)
