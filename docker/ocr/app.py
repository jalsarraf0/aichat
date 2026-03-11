"""aichat-ocr: Tesseract OCR + pdf2image for image and PDF text extraction.

Endpoints:
  POST /ocr            — OCR from base64 image
  POST /ocr/path       — OCR from workspace file path
  POST /ocr/boxes      — OCR words + bounding boxes from base64 image
  POST /ocr/path/boxes — OCR words + bounding boxes from workspace file path
  POST /ocr/pdf        — OCR a PDF (all pages or specific pages)
  GET  /languages      — list installed Tesseract language codes
  GET  /health         — service health + tesseract version
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import pytesseract
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("aichat-ocr")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace")).resolve()
DB_API = os.environ.get("DATABASE_URL", "http://aichat-database:8091")
_SERVICE = "aichat-ocr"

app = FastAPI()


# ---------------------------------------------------------------------------
# Error reporting
# ---------------------------------------------------------------------------

async def _report_error(message: str, detail: str | None = None) -> None:
    try:
        import httpx

        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(
                f"{DB_API}/errors/log",
                json={
                    "service": _SERVICE,
                    "level": "ERROR",
                    "message": message,
                    "detail": detail,
                },
            )
    except Exception:
        pass


@app.exception_handler(Exception)
async def _global_exc(request: Request, exc: Exception) -> JSONResponse:
    msg = str(exc)
    log.error("Unhandled [%s %s]: %s", request.method, request.url.path, msg, exc_info=True)
    asyncio.create_task(_report_error(msg, f"{request.method} {request.url.path}"))
    return JSONResponse(status_code=500, content={"error": msg})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tesseract_version() -> str:
    try:
        r = subprocess.run(["tesseract", "--version"], capture_output=True, text=True, timeout=5)
        first = (r.stdout or r.stderr).splitlines()[0]
        return first.replace("tesseract ", "").strip()
    except Exception:
        return "unknown"


def _installed_langs() -> list[str]:
    try:
        r = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True, timeout=5)
        lines = (r.stdout or r.stderr).splitlines()
        return [ln.strip() for ln in lines if ln.strip() and "List" not in ln]
    except Exception:
        return ["eng"]


def _do_ocr(img: Image.Image, lang: str) -> dict[str, Any]:
    text = pytesseract.image_to_string(img, lang=lang)
    words = [w for w in text.split() if w]
    return {"text": text.strip(), "word_count": len(words)}


def _do_ocr_boxes(img: Image.Image, lang: str) -> dict[str, Any]:
    """Return OCR text plus per-token bounding boxes.

    Coordinates are pixel-based in the input image coordinate system.
    """
    data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
    n = len(data.get("text", []))
    boxes: list[dict[str, Any]] = []
    full_tokens: list[str] = []
    for i in range(n):
        token = str(data.get("text", [""])[i] or "").strip()
        if not token:
            continue
        conf_raw = str(data.get("conf", ["-1"])[i] or "-1").strip()
        try:
            conf = float(conf_raw)
        except Exception:
            conf = -1.0
        left = int(data.get("left", [0])[i] or 0)
        top = int(data.get("top", [0])[i] or 0)
        width = int(data.get("width", [0])[i] or 0)
        height = int(data.get("height", [0])[i] or 0)
        right = left + max(0, width)
        bottom = top + max(0, height)
        item = {
            "index": len(boxes),
            "text": token,
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
            "conf": conf,
            "line_num": int(data.get("line_num", [0])[i] or 0),
            "block_num": int(data.get("block_num", [0])[i] or 0),
            "par_num": int(data.get("par_num", [0])[i] or 0),
        }
        boxes.append(item)
        full_tokens.append(token)

    # Keep full text normalized from OCR primary path for consistency.
    text = pytesseract.image_to_string(img, lang=lang).strip()
    return {
        "text": text,
        "word_count": len(full_tokens),
        "boxes": boxes,
    }


def _resolve_workspace_file(path: str) -> Path:
    raw = (path or "").strip()
    if not raw:
        raise HTTPException(status_code=422, detail="'path' is required")

    p = Path(raw)
    if not p.is_absolute():
        p = (WORKSPACE / p).resolve()
    else:
        p = p.resolve()

    try:
        p.relative_to(WORKSPACE)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Access denied: path outside workspace") from exc

    if p.is_file():
        return p

    alt = (WORKSPACE / p.name).resolve()
    if alt.is_file():
        return alt

    raise HTTPException(status_code=404, detail=f"File not found: {path}")


def _decode_image_b64(b64: str) -> Image.Image:
    try:
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid image data: {exc}") from exc


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "tesseract_version": _tesseract_version(),
        "languages": _installed_langs(),
    }


@app.get("/languages")
def languages() -> dict[str, Any]:
    return {"languages": _installed_langs()}


@app.post("/ocr")
def ocr_image(payload: dict[str, Any]) -> dict[str, Any]:
    b64 = str(payload.get("b64", "")).strip()
    lang = str(payload.get("lang", "eng")).strip() or "eng"
    if not b64:
        raise HTTPException(status_code=422, detail="'b64' is required (base64-encoded image)")
    img = _decode_image_b64(b64)
    result = _do_ocr(img, lang)
    log.info("ocr_image lang=%s words=%d", lang, result["word_count"])
    return result


@app.post("/ocr/boxes")
def ocr_image_boxes(payload: dict[str, Any]) -> dict[str, Any]:
    b64 = str(payload.get("b64", "")).strip()
    lang = str(payload.get("lang", "eng")).strip() or "eng"
    if not b64:
        raise HTTPException(status_code=422, detail="'b64' is required (base64-encoded image)")
    img = _decode_image_b64(b64)
    result = _do_ocr_boxes(img, lang)
    log.info("ocr_image_boxes lang=%s boxes=%d", lang, len(result.get("boxes", [])))
    return result


@app.post("/ocr/path")
def ocr_path(payload: dict[str, Any]) -> dict[str, Any]:
    path = str(payload.get("path", "")).strip()
    lang = str(payload.get("lang", "eng")).strip() or "eng"
    full = _resolve_workspace_file(path)
    try:
        img = Image.open(str(full)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot open image: {exc}") from exc
    result = _do_ocr(img, lang)
    log.info("ocr_path %s lang=%s words=%d", path, lang, result["word_count"])
    return result


@app.post("/ocr/path/boxes")
def ocr_path_boxes(payload: dict[str, Any]) -> dict[str, Any]:
    path = str(payload.get("path", "")).strip()
    lang = str(payload.get("lang", "eng")).strip() or "eng"
    full = _resolve_workspace_file(path)
    try:
        img = Image.open(str(full)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot open image: {exc}") from exc
    result = _do_ocr_boxes(img, lang)
    log.info("ocr_path_boxes %s lang=%s boxes=%d", path, lang, len(result.get("boxes", [])))
    return result


@app.post("/ocr/pdf")
def ocr_pdf(payload: dict[str, Any]) -> dict[str, Any]:
    b64_pdf = str(payload.get("b64_pdf", "")).strip()
    lang = str(payload.get("lang", "eng")).strip() or "eng"
    pages = payload.get("pages")  # None means all pages

    if not b64_pdf:
        raise HTTPException(status_code=422, detail="'b64_pdf' is required")

    try:
        pdf_bytes = base64.b64decode(b64_pdf)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid base64: {exc}") from exc

    try:
        from pdf2image import convert_from_bytes

        pil_pages = convert_from_bytes(pdf_bytes, dpi=300)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF rasterization failed: {exc}") from exc

    if pages:
        page_indices = [p - 1 for p in pages if 0 < p <= len(pil_pages)]
        pil_pages = [pil_pages[i] for i in page_indices if i < len(pil_pages)]

    results = []
    full_text_parts = []
    for i, pil_img in enumerate(pil_pages):
        page_result = _do_ocr(pil_img.convert("RGB"), lang)
        results.append({
            "page": i + 1,
            "text": page_result["text"],
            "word_count": page_result["word_count"],
        })
        full_text_parts.append(page_result["text"])

    full_text = "\n\n---\n\n".join(full_text_parts)
    log.info("ocr_pdf lang=%s pages=%d", lang, len(results))
    return {
        "pages": results,
        "page_count": len(results),
        "full_text": full_text,
        "word_count": sum(r["word_count"] for r in results),
    }
