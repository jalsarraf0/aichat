"""aichat-vision: Consolidated video analysis and OCR service.

Merges aichat-video (port 8099) and aichat-ocr (port 8100) into a single
FastAPI application. The video routes are at the root; OCR routes are at /ocr/*.

Routes:
  /health          -- aggregate health (ffmpeg + tesseract versions)
  /info            -- video metadata (duration, fps, resolution, codec)
  /frames          -- extract frames at intervals
  /thumbnail       -- single frame as base64 PNG
  /ocr/health      -- OCR sub-health
  /ocr             -- OCR from base64 image
  /ocr/path        -- OCR from workspace file path
  /ocr/boxes       -- OCR words + bounding boxes from base64 image
  /ocr/path/boxes  -- OCR words + bounding boxes from workspace file
  /ocr/pdf         -- OCR a PDF (all pages or specific pages)
  /ocr/languages   -- list installed Tesseract language codes
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pytesseract
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("aichat-vision")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WORKSPACE     = Path(os.environ.get("WORKSPACE", "/workspace"))
DB_API        = os.environ.get("DATABASE_URL", "http://aichat-data:8091")
_INTEL_GPU    = os.environ.get("INTEL_GPU", "").strip() == "1"
_VAAPI_DEVICE = os.environ.get("INTEL_VAAPI_DEVICE", "/dev/dri/renderD128")

app = FastAPI(title="aichat-vision")


# ---------------------------------------------------------------------------
# Error reporting
# ---------------------------------------------------------------------------

async def _report_error(message: str, detail: str | None = None) -> None:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(
                f"{DB_API}/errors/log",
                json={"service": "aichat-vision", "level": "ERROR",
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

# ===========================================================================
# Video helpers
# ===========================================================================

def _ffmpeg_version() -> str:
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
        first = r.stdout.splitlines()[0] if r.stdout else ""
        return first.replace("ffmpeg version ", "").split(" ")[0]
    except Exception:
        return "unknown"


def _gpu_mode() -> str:
    if _INTEL_GPU and os.path.exists(_VAAPI_DEVICE):
        return "intel-vaapi"
    return "cpu"


def _ffmpeg_hwaccel_args() -> list[str]:
    if _gpu_mode() == "intel-vaapi":
        return ["-hwaccel", "auto"]
    return []


def _ffprobe(url_or_path: str) -> dict:
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_streams", "-show_format", url_or_path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {r.stderr[:300]}")
    return json.loads(r.stdout)


def _parse_video_info(probe: dict) -> dict:
    fmt     = probe.get("format", {})
    streams = probe.get("streams", [])
    video   = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio   = next((s for s in streams if s.get("codec_type") == "audio"), {})
    duration = 0.0
    try:
        duration = float(fmt.get("duration") or video.get("duration", 0))
    except (ValueError, TypeError):
        pass
    fps = 0.0
    try:
        num, den = video.get("r_frame_rate", "0/1").split("/")
        fps = round(float(num) / float(den), 3) if float(den) else 0.0
    except Exception:
        pass
    size_bytes = 0
    try:
        size_bytes = int(fmt.get("size", 0))
    except (ValueError, TypeError):
        pass
    return {
        "duration_s":  round(duration, 3),
        "fps":         fps,
        "width":       int(video.get("width", 0)),
        "height":      int(video.get("height", 0)),
        "codec":       video.get("codec_name", ""),
        "audio_codec": audio.get("codec_name", ""),
        "format":      fmt.get("format_name", ""),
        "size_mb":     round(size_bytes / 1_048_576, 3),
    }


async def _download_to_tmp(url: str) -> str:
    suffix = Path(url.split("?")[0]).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.close()
    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                with open(tmp.name, "wb") as f:
                    async for chunk in resp.aiter_bytes(65536):
                        f.write(chunk)
    except Exception:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise
    return tmp.name


def _is_remote(url: str) -> bool:
    return url.startswith("http://") or url.startswith("https://")


# ===========================================================================
# Root video routes
# ===========================================================================

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "ffmpeg_version": _ffmpeg_version(),
        "gpu_mode": _gpu_mode(),
        "tesseract_version": _tesseract_version(),
        "tesseract_languages": _installed_langs(),
    }


@app.post("/info")
async def video_info(payload: dict) -> dict:
    url = str(payload.get("url", "")).strip()
    if not url:
        raise HTTPException(status_code=422, detail="'url' is required")
    tmp_path = None
    try:
        if _is_remote(url):
            tmp_path = await _download_to_tmp(url)
            target = tmp_path
        else:
            target = url
        probe = _ffprobe(target)
        return {"url": url, **_parse_video_info(probe)}
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not fetch video — upstream returned {exc.response.status_code}: {url}",
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"video_info failed: {exc}")
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


@app.post("/frames")
async def extract_frames(payload: dict) -> dict:
    url          = str(payload.get("url", "")).strip()
    interval_sec = float(payload.get("interval_sec", 5.0))
    max_frames   = max(1, min(int(payload.get("max_frames", 20)), 100))
    if not url:
        raise HTTPException(status_code=422, detail="'url' is required")
    if interval_sec <= 0:
        interval_sec = 5.0
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    ts_base  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir  = WORKSPACE / f"video_frames_{ts_base}"
    out_dir.mkdir(exist_ok=True)
    tmp_path = None
    try:
        if _is_remote(url):
            tmp_path = await _download_to_tmp(url)
            target = tmp_path
        else:
            target = url
        out_pattern = str(out_dir / "frame_%04d.png")
        cmd = ["ffmpeg", *_ffmpeg_hwaccel_args(), "-i", target,
               "-vf", f"fps=1/{interval_sec}", "-frames:v", str(max_frames),
               "-q:v", "2", out_pattern, "-y"]
        r = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: subprocess.run(cmd, capture_output=True, text=True, timeout=120),
        )
        if r.returncode != 0 and _ffmpeg_hwaccel_args():
            cpu_cmd = ["ffmpeg", "-i", target, "-vf", f"fps=1/{interval_sec}",
                       "-frames:v", str(max_frames), "-q:v", "2", out_pattern, "-y"]
            r = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: subprocess.run(cpu_cmd, capture_output=True, text=True, timeout=120),
            )
        if r.returncode != 0 and not list(out_dir.iterdir()):
            raise RuntimeError(f"ffmpeg failed: {r.stderr[:400]}")
        frames = [
            {"path": str(f), "timestamp_s": round(i * interval_sec, 3), "filename": f.name}
            for i, f in enumerate(sorted(out_dir.iterdir()))
        ]
        return {"url": url, "interval_sec": interval_sec, "frames": frames,
                "count": len(frames), "output_dir": str(out_dir)}
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not fetch video — upstream returned {exc.response.status_code}: {url}",
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"video_frames failed: {exc}")
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


@app.post("/thumbnail")
async def get_thumbnail(payload: dict) -> dict:
    url           = str(payload.get("url", "")).strip()
    timestamp_sec = float(payload.get("timestamp_sec", 0.0))
    if not url:
        raise HTTPException(status_code=422, detail="'url' is required")
    tmp_path  = None
    thumb_tmp = None
    try:
        if _is_remote(url):
            tmp_path = await _download_to_tmp(url)
            target = tmp_path
        else:
            target = url
        thumb_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        thumb_tmp.close()
        cmd = ["ffmpeg", "-ss", str(timestamp_sec), *_ffmpeg_hwaccel_args(),
               "-i", target, "-vframes", "1", "-q:v", "2", thumb_tmp.name, "-y"]
        r = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: subprocess.run(cmd, capture_output=True, text=True, timeout=30),
        )
        if (r.returncode != 0 or not os.path.getsize(thumb_tmp.name)) \
                and _ffmpeg_hwaccel_args():
            cpu_cmd = ["ffmpeg", "-ss", str(timestamp_sec), "-i", target,
                       "-vframes", "1", "-q:v", "2", thumb_tmp.name, "-y"]
            r = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: subprocess.run(cpu_cmd, capture_output=True, text=True, timeout=30),
            )
        if r.returncode != 0 or not os.path.getsize(thumb_tmp.name):
            raise RuntimeError(f"ffmpeg thumbnail failed: {r.stderr[:300]}")
        raw = Path(thumb_tmp.name).read_bytes()
        b64 = base64.standard_b64encode(raw).decode("ascii")
        probe = _ffprobe(thumb_tmp.name)
        info  = _parse_video_info(probe)
        return {"url": url, "timestamp_sec": timestamp_sec, "b64": b64,
                "width": info["width"], "height": info["height"], "mime_type": "image/png"}
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not fetch video — upstream returned {exc.response.status_code}: {url}",
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"video_thumbnail failed: {exc}")
    finally:
        for p in [tmp_path, thumb_tmp.name if thumb_tmp else None]:
            if p:
                try:
                    os.unlink(p)
                except Exception:
                    pass

# ===========================================================================
# /ocr - Tesseract OCR  (was aichat-ocr:8100)
# ===========================================================================

ocr_router = APIRouter(prefix="/ocr", tags=["ocr"])


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
    data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
    n = len(data.get("text", []))
    boxes: list[dict[str, Any]] = []
    for i in range(n):
        token = str(data.get("text", [""])[i] or "").strip()
        if not token:
            continue
        try:
            conf = float(str(data.get("conf", ["-1"])[i] or "-1").strip())
        except Exception:
            conf = -1.0
        left   = int(data.get("left",   [0])[i] or 0)
        top    = int(data.get("top",    [0])[i] or 0)
        width  = int(data.get("width",  [0])[i] or 0)
        height = int(data.get("height", [0])[i] or 0)
        boxes.append({
            "index": len(boxes), "text": token,
            "left": left, "top": top, "right": left + max(0, width),
            "bottom": top + max(0, height), "conf": conf,
            "line_num": int(data.get("line_num", [0])[i] or 0),
            "block_num": int(data.get("block_num", [0])[i] or 0),
        })
    text = pytesseract.image_to_string(img, lang=lang).strip()
    return {"text": text, "word_count": len(boxes), "boxes": boxes}


def _resolve_workspace_file(path: str) -> Path:
    """Resolve a user-supplied path to an absolute path strictly inside WORKSPACE.

    Symlinks are NOT followed so that a symlink pointing outside WORKSPACE cannot
    bypass the containment check (eliminates TOCTOU race).  The resolved path must
    be a regular file; the fallback-by-basename logic has been removed because it
    allowed bypassing the containment check via path traversal.
    """
    raw = (path or "").strip()
    if not raw:
        raise HTTPException(status_code=422, detail="'path' is required")
    p = Path(raw)
    workspace_resolved = WORKSPACE.resolve()
    # Build candidate without following symlinks
    if p.is_absolute():
        candidate = p
    else:
        candidate = WORKSPACE / p
    # Normalize without resolving symlinks to avoid TOCTOU
    try:
        candidate = Path(os.path.normpath(candidate))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid path: {exc}") from exc
    # Enforce containment
    try:
        candidate.relative_to(workspace_resolved)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Access denied: path outside workspace") from exc
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    return candidate


def _decode_b64_image(b64: str) -> Image.Image:
    try:
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid image data: {exc}") from exc


@ocr_router.get("/health")
def ocr_health() -> dict[str, Any]:
    return {"status": "ok", "tesseract_version": _tesseract_version(),
            "languages": _installed_langs()}


@ocr_router.get("/languages")
def ocr_languages() -> dict[str, Any]:
    return {"languages": _installed_langs()}


@ocr_router.post("")
def ocr_image(payload: dict[str, Any]) -> dict[str, Any]:
    b64  = str(payload.get("b64", "")).strip()
    lang = str(payload.get("lang", "eng")).strip() or "eng"
    if not b64:
        raise HTTPException(status_code=422, detail="'b64' is required (base64-encoded image)")
    img = _decode_b64_image(b64)
    result = _do_ocr(img, lang)
    log.info("ocr lang=%s words=%d", lang, result["word_count"])
    return result


@ocr_router.post("/boxes")
def ocr_image_boxes(payload: dict[str, Any]) -> dict[str, Any]:
    b64  = str(payload.get("b64", "")).strip()
    lang = str(payload.get("lang", "eng")).strip() or "eng"
    if not b64:
        raise HTTPException(status_code=422, detail="'b64' is required")
    img = _decode_b64_image(b64)
    return _do_ocr_boxes(img, lang)


@ocr_router.post("/path")
def ocr_path(payload: dict[str, Any]) -> dict[str, Any]:
    path = str(payload.get("path", "")).strip()
    lang = str(payload.get("lang", "eng")).strip() or "eng"
    full = _resolve_workspace_file(path)
    try:
        img = Image.open(str(full)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot open image: {exc}") from exc
    return _do_ocr(img, lang)


@ocr_router.post("/path/boxes")
def ocr_path_boxes(payload: dict[str, Any]) -> dict[str, Any]:
    path = str(payload.get("path", "")).strip()
    lang = str(payload.get("lang", "eng")).strip() or "eng"
    full = _resolve_workspace_file(path)
    try:
        img = Image.open(str(full)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot open image: {exc}") from exc
    return _do_ocr_boxes(img, lang)


@ocr_router.post("/pdf")
def ocr_pdf(payload: dict[str, Any]) -> dict[str, Any]:
    b64_pdf = str(payload.get("b64_pdf", "")).strip()
    lang    = str(payload.get("lang", "eng")).strip() or "eng"
    pages   = payload.get("pages")
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
        results.append({"page": i + 1, "text": page_result["text"],
                        "word_count": page_result["word_count"]})
        full_text_parts.append(page_result["text"])
    full_text = "\n\n---\n\n".join(full_text_parts)
    log.info("ocr_pdf lang=%s pages=%d", lang, len(results))
    return {"pages": results, "page_count": len(results), "full_text": full_text,
            "word_count": sum(r["word_count"] for r in results)}


app.include_router(ocr_router)
