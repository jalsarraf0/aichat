"""aichat-ocr: Tesseract OCR + pdf2image for image and PDF text extraction.

Endpoints:
  POST /ocr       — OCR from base64 image
  POST /ocr/path  — OCR from workspace file path
  POST /ocr/pdf   — OCR a PDF (all pages or specific pages)
  GET  /languages — list installed Tesseract language codes
  GET  /health    — service health + tesseract version
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import subprocess

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

WORKSPACE = os.environ.get("WORKSPACE", "/workspace")
DB_API    = os.environ.get("DATABASE_URL", "http://aichat-database:8091")
_SERVICE  = "aichat-ocr"

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
                json={"service": _SERVICE, "level": "ERROR",
                      "message": message, "detail": detail},
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
        r = subprocess.run(
            ["tesseract", "--list-langs"], capture_output=True, text=True, timeout=5
        )
        lines = (r.stdout or r.stderr).splitlines()
        return [ln.strip() for ln in lines if ln.strip() and "List" not in ln]
    except Exception:
        return ["eng"]


def _do_ocr(img: Image.Image, lang: str) -> dict:
    text = pytesseract.image_to_string(img, lang=lang)
    words = [w for w in text.split() if w]
    return {"text": text.strip(), "word_count": len(words)}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "tesseract_version": _tesseract_version(),
        "languages": _installed_langs(),
    }


@app.get("/languages")
def languages() -> dict:
    return {"languages": _installed_langs()}


@app.post("/ocr")
def ocr_image(payload: dict) -> dict:
    b64  = str(payload.get("b64", "")).strip()
    lang = str(payload.get("lang", "eng")).strip() or "eng"
    if not b64:
        raise HTTPException(status_code=422, detail="'b64' is required (base64-encoded image)")
    try:
        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid image data: {exc}") from exc
    result = _do_ocr(img, lang)
    log.info("ocr_image lang=%s words=%d", lang, result["word_count"])
    return result


@app.post("/ocr/path")
def ocr_path(payload: dict) -> dict:
    path = str(payload.get("path", "")).strip()
    lang = str(payload.get("lang", "eng")).strip() or "eng"
    if not path:
        raise HTTPException(status_code=422, detail="'path' is required")

    # Resolve path — allow /workspace prefix
    if not os.path.isabs(path):
        full = os.path.join(WORKSPACE, path)
    else:
        full = path

    if not os.path.isfile(full):
        # Try stripping /workspace and re-joining
        basename = os.path.basename(full)
        alt = os.path.join(WORKSPACE, basename)
        if os.path.isfile(alt):
            full = alt
        else:
            raise HTTPException(status_code=404, detail=f"File not found: {path}")

    try:
        img = Image.open(full).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot open image: {exc}") from exc

    result = _do_ocr(img, lang)
    log.info("ocr_path %s lang=%s words=%d", path, lang, result["word_count"])
    return result


@app.post("/ocr/pdf")
def ocr_pdf(payload: dict) -> dict:
    b64_pdf = str(payload.get("b64_pdf", "")).strip()
    lang    = str(payload.get("lang", "eng")).strip() or "eng"
    pages   = payload.get("pages")  # None means all pages

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
        # pages is 1-based list e.g. [1, 2, 3]
        page_indices = [p - 1 for p in pages if 0 < p <= len(pil_pages)]
        pil_pages = [pil_pages[i] for i in page_indices if i < len(pil_pages)]

    results = []
    full_text_parts = []
    for i, pil_img in enumerate(pil_pages):
        page_result = _do_ocr(pil_img.convert("RGB"), lang)
        results.append({"page": i + 1, "text": page_result["text"],
                         "word_count": page_result["word_count"]})
        full_text_parts.append(page_result["text"])

    full_text = "\n\n---\n\n".join(full_text_parts)
    log.info("ocr_pdf lang=%s pages=%d", lang, len(results))
    return {
        "pages":      results,
        "page_count": len(results),
        "full_text":  full_text,
        "word_count": sum(r["word_count"] for r in results),
    }
