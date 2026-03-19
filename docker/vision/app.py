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


@app.on_event("startup")
def _log_gpu_status() -> None:
    mode = _gpu_mode()
    log.info("GPU mode: %s  (INTEL_GPU=%s, device=%s exists=%s)",
             mode, _INTEL_GPU, _VAAPI_DEVICE, os.path.exists(_VAAPI_DEVICE))
    if mode == "intel-vaapi":
        profiles = _vaapi_profiles()
        log.info("VA-API profiles found: %d", len(profiles))
        encode = [p for p in profiles if "EncSlice" in p or "EncPicture" in p]
        if encode:
            log.info("VA-API HW encode capable: %s",
                     ", ".join(sorted({p.split(":")[0].strip() for p in encode})))
        else:
            log.warning("VA-API device present but no encode profiles found — "
                        "check intel-media-va-driver installation")
    else:
        log.info("Running in CPU-only mode (no VA-API)")


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


def _vaapi_profiles() -> list[str]:
    """Return VA-API profiles reported by vainfo inside the container."""
    try:
        r = subprocess.run(
            ["vainfo", "--display", "drm", "--device", _VAAPI_DEVICE],
            capture_output=True, text=True, timeout=5,
        )
        output = r.stdout + r.stderr
        return [
            ln.strip() for ln in output.splitlines()
            if "VAProfile" in ln and "VAEntrypoint" in ln
        ]
    except Exception:
        return []


def _ffmpeg_hwaccel_args() -> list[str]:
    """VA-API decode args — frames auto-transfer to system memory for filters."""
    if _gpu_mode() == "intel-vaapi":
        return ["-hwaccel", "vaapi", "-vaapi_device", _VAAPI_DEVICE]
    return []


def _ffmpeg_full_hwaccel_args() -> list[str]:
    """VA-API decode args keeping frames on GPU (for transcode → encode)."""
    if _gpu_mode() == "intel-vaapi":
        return [
            "-hwaccel", "vaapi",
            "-hwaccel_device", _VAAPI_DEVICE,
            "-hwaccel_output_format", "vaapi",
        ]
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
    mode = _gpu_mode()
    result: dict[str, Any] = {
        "status": "ok",
        "ffmpeg_version": _ffmpeg_version(),
        "gpu_mode": mode,
        "tesseract_version": _tesseract_version(),
        "tesseract_languages": _installed_langs(),
    }
    if mode == "intel-vaapi":
        profiles = _vaapi_profiles()
        result["vaapi_device"] = _VAAPI_DEVICE
        result["vaapi_profile_count"] = len(profiles)
        # Summarise encode-capable codecs
        encode_profiles = [p for p in profiles if "EncSlice" in p or "EncPicture" in p]
        result["vaapi_hw_encoders"] = sorted({
            p.split(":")[0].strip().replace("VAProfile", "")
            for p in encode_profiles
        })
    return result


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

_VAAPI_ENCODERS: dict[str, str] = {
    "h264":  "h264_vaapi",
    "hevc":  "hevc_vaapi",
    "h265":  "hevc_vaapi",
    "vp9":   "vp9_vaapi",
    "av1":   "av1_vaapi",
    "mjpeg": "mjpeg_vaapi",
}


@app.post("/transcode")
async def transcode(payload: dict) -> dict:
    """Hardware-accelerated transcode using Intel Arc VA-API encode.

    Input: {"url": "...", "codec": "h264"|"hevc"|"av1"|"vp9",
            "bitrate": "5M", "width": 0, "height": 0,
            "filename": "output.mp4"}
    Output: {"path": "/workspace/...", "codec": ..., "gpu_accelerated": true}
    """
    url      = str(payload.get("url", "")).strip()
    codec    = str(payload.get("codec", "h264")).strip().lower()
    bitrate  = str(payload.get("bitrate", "5M")).strip()
    width    = int(payload.get("width", 0))
    height   = int(payload.get("height", 0))
    filename = str(payload.get("filename", "")).strip()
    if not url:
        raise HTTPException(status_code=422, detail="'url' is required")

    hw_encoder = _VAAPI_ENCODERS.get(codec)
    use_gpu = _gpu_mode() == "intel-vaapi" and hw_encoder is not None

    WORKSPACE.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ext = ".webm" if codec == "vp9" else ".mp4"
    if not filename:
        filename = f"transcode_{ts_str}{ext}"
    out_path = WORKSPACE / filename

    tmp_path = None
    try:
        if _is_remote(url):
            tmp_path = await _download_to_tmp(url)
            target = tmp_path
        else:
            target = url

        if use_gpu:
            # Full VA-API pipeline: decode on GPU → optional scale on GPU → encode on GPU
            vf_parts: list[str] = []
            if width > 0 or height > 0:
                sw = width if width > 0 else -2
                sh = height if height > 0 else -2
                vf_parts.append(f"scale_vaapi=w={sw}:h={sh}")
            vf_arg = ",".join(vf_parts) if vf_parts else ""

            cmd = [
                "ffmpeg",
                *_ffmpeg_full_hwaccel_args(),
                "-i", target,
            ]
            if vf_arg:
                cmd += ["-vf", vf_arg]
            cmd += [
                "-c:v", hw_encoder,
                "-b:v", bitrate,
                "-c:a", "copy",
                str(out_path), "-y",
            ]
        else:
            # CPU fallback
            sw_encoder = {"hevc": "libx265", "h265": "libx265", "av1": "libsvtav1",
                          "vp9": "libvpx-vp9"}.get(codec, "libx264")
            cmd = ["ffmpeg", "-i", target]
            if width > 0 or height > 0:
                sw = width if width > 0 else -2
                sh = height if height > 0 else -2
                cmd += ["-vf", f"scale={sw}:{sh}"]
            cmd += [
                "-c:v", sw_encoder,
                "-b:v", bitrate,
                "-c:a", "copy",
                str(out_path), "-y",
            ]

        log.info("transcode cmd: %s", " ".join(cmd))
        r = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: subprocess.run(cmd, capture_output=True, text=True, timeout=600),
        )

        # GPU fallback to CPU on encode failure
        if r.returncode != 0 and use_gpu:
            log.warning("VA-API encode failed, falling back to CPU: %s", r.stderr[:300])
            sw_encoder = {"hevc": "libx265", "h265": "libx265", "av1": "libsvtav1",
                          "vp9": "libvpx-vp9"}.get(codec, "libx264")
            cpu_cmd = ["ffmpeg", "-i", target]
            if width > 0 or height > 0:
                sw = width if width > 0 else -2
                sh = height if height > 0 else -2
                cpu_cmd += ["-vf", f"scale={sw}:{sh}"]
            cpu_cmd += ["-c:v", sw_encoder, "-b:v", bitrate,
                        "-c:a", "copy", str(out_path), "-y"]
            r = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: subprocess.run(cpu_cmd, capture_output=True, text=True, timeout=600),
            )
            use_gpu = False

        if r.returncode != 0:
            raise RuntimeError(f"transcode failed: {r.stderr[:400]}")

        # Get output info
        probe = _ffprobe(str(out_path))
        info = _parse_video_info(probe)
        return {
            "url": url,
            "path": str(out_path),
            "filename": filename,
            "codec": codec,
            "gpu_accelerated": use_gpu,
            **info,
        }
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not fetch video — upstream returned {exc.response.status_code}: {url}",
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"transcode failed: {exc}")
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
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


# ===========================================================================
# /clip - CLIP ViT-B/32 image embedding  (768-dim visual vectors)
# ===========================================================================

clip_router = APIRouter(prefix="/clip", tags=["clip"])

# Lazy-loaded ONNX session (loads ~350MB model on first request)
_clip_session: Any = None
_CLIP_MODEL_PATH = "/app/models/clip_vit_b32_visual.onnx"
# CLIP normalization constants (not ImageNet — specific to CLIP training data)
_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def _get_clip_session() -> Any:
    """Lazy-load the CLIP ONNX session on first use."""
    global _clip_session
    if _clip_session is not None:
        return _clip_session
    import onnxruntime as ort  # noqa: E402
    providers = ["CPUExecutionProvider"]
    # Prefer OpenVINO (Intel Arc) if available
    available = ort.get_available_providers()
    if "OpenVINOExecutionProvider" in available:
        providers.insert(0, "OpenVINOExecutionProvider")
    _clip_session = ort.InferenceSession(_CLIP_MODEL_PATH, providers=providers)
    log.info("CLIP ONNX session loaded: providers=%s, model=%s",
             _clip_session.get_providers(), _CLIP_MODEL_PATH)
    return _clip_session


def _preprocess_clip(img: Image.Image) -> Any:
    """Resize to 224x224, normalize with CLIP constants, return CHW float32 numpy array."""
    import numpy as np  # noqa: E402
    img_rgb = img.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    arr = np.array(img_rgb, dtype=np.float32) / 255.0  # HWC [0,1]
    # Normalize per-channel
    for c in range(3):
        arr[:, :, c] = (arr[:, :, c] - _CLIP_MEAN[c]) / _CLIP_STD[c]
    # HWC → CHW → NCHW
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    arr = np.expand_dims(arr, axis=0)    # NCHW batch=1
    return arr


@clip_router.get("/health")
def clip_health() -> dict[str, Any]:
    model_exists = os.path.isfile(_CLIP_MODEL_PATH)
    return {
        "status": "ok" if model_exists else "model_missing",
        "model": "clip-vit-b32",
        "model_path": _CLIP_MODEL_PATH,
        "model_exists": model_exists,
        "embedding_dim": 512,
    }


@clip_router.post("/embed")
async def clip_embed(payload: dict[str, Any]) -> dict[str, Any]:
    """Generate a 768-dim CLIP embedding for an image.

    Input: {"image_base64": "..."} or {"image_url": "..."}
    Output: {"embedding": [768 floats], "model": "clip-vit-b32", "dim": 768}
    """
    import numpy as np  # noqa: E402

    b64 = str(payload.get("image_base64", "")).strip()
    url = str(payload.get("image_url", "")).strip()

    if not b64 and not url:
        raise HTTPException(status_code=422, detail="Provide 'image_base64' or 'image_url'")

    # Load image
    if b64:
        img = _decode_b64_image(b64)
    else:
        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                r = await client.get(url)
                r.raise_for_status()
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Failed to fetch image: {exc}") from exc

    # Preprocess and run inference
    try:
        input_tensor = _preprocess_clip(img)
        session = _get_clip_session()
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_tensor})
        embedding = outputs[0][0]  # shape: (768,)
        # L2-normalize
        norm = float(np.linalg.norm(embedding))
        if norm > 0:
            embedding = embedding / norm
        embedding_list = embedding.tolist()
    except Exception as exc:
        log.error("CLIP inference failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"CLIP inference failed: {exc}") from exc

    return {
        "embedding": embedding_list,
        "model": "clip-vit-b32",
        "dim": len(embedding_list),
    }


app.include_router(clip_router)


# ===========================================================================
# /detect - YOLOv8n object and human detection
# ===========================================================================

detect_router = APIRouter(prefix="/detect", tags=["detect"])

_yolo_session: Any = None
_YOLO_MODEL_PATH = "/app/models/yolov8n.onnx"

# COCO 80-class names
_COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def _get_yolo_session() -> Any:
    """Lazy-load the YOLOv8n ONNX session."""
    global _yolo_session
    if _yolo_session is not None:
        return _yolo_session
    import onnxruntime as ort
    _yolo_session = ort.InferenceSession(_YOLO_MODEL_PATH, providers=["CPUExecutionProvider"])
    log.info("YOLOv8n ONNX session loaded: %s", _YOLO_MODEL_PATH)
    return _yolo_session


def _preprocess_yolo(img: Image.Image) -> tuple[Any, float, float]:
    """Resize to 640x640 with letterboxing, normalize, return (NCHW tensor, x_scale, y_scale)."""
    import numpy as np
    import cv2

    orig_w, orig_h = img.size
    target = 640

    # Letterbox resize (maintain aspect ratio)
    scale = min(target / orig_w, target / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)

    arr = np.array(img.convert("RGB"))
    arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to 640x640
    padded = np.full((target, target, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = arr

    # Normalize to [0, 1] and CHW
    tensor = padded.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))  # HWC → CHW
    tensor = np.expand_dims(tensor, 0)  # NCHW

    return tensor, scale, scale


def _postprocess_yolo(
    output: Any, score_thresh: float, iou_thresh: float,
    x_scale: float, y_scale: float, orig_w: int, orig_h: int,
) -> list[dict]:
    """Parse YOLOv8 output [1, 84, 8400] → list of detections."""
    import numpy as np

    # YOLOv8 output shape: [1, 84, 8400] — 84 = 4 (box) + 80 (classes)
    preds = output[0]  # [84, 8400]
    if preds.shape[0] == 84:
        preds = preds.T  # [8400, 84]

    boxes = preds[:, :4]  # cx, cy, w, h
    scores = preds[:, 4:]  # 80 class scores

    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    # Filter by confidence
    mask = confidences > score_thresh
    boxes = boxes[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    if len(boxes) == 0:
        return []

    # Convert cx, cy, w, h → x1, y1, x2, y2
    x1 = (boxes[:, 0] - boxes[:, 2] / 2) / x_scale
    y1 = (boxes[:, 1] - boxes[:, 3] / 2) / y_scale
    x2 = (boxes[:, 0] + boxes[:, 2] / 2) / x_scale
    y2 = (boxes[:, 1] + boxes[:, 3] / 2) / y_scale

    # Clip to image bounds
    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)

    # NMS (simple greedy)
    order = np.argsort(-confidences)
    keep: list[int] = []
    suppressed = set()
    for i in order:
        if int(i) in suppressed:
            continue
        keep.append(int(i))
        for j in order:
            if int(j) in suppressed or int(j) == int(i):
                continue
            if class_ids[i] != class_ids[j]:
                continue
            # IoU
            ix1 = max(x1[i], x1[j])
            iy1 = max(y1[i], y1[j])
            ix2 = min(x2[i], x2[j])
            iy2 = min(y2[i], y2[j])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_i = (x2[i] - x1[i]) * (y2[i] - y1[i])
            area_j = (x2[j] - x1[j]) * (y2[j] - y1[j])
            union = area_i + area_j - inter
            if union > 0 and inter / union > iou_thresh:
                suppressed.add(int(j))

    results = []
    for idx in keep:
        cls_name = _COCO_CLASSES[class_ids[idx]] if class_ids[idx] < len(_COCO_CLASSES) else f"class_{class_ids[idx]}"
        results.append({
            "class": cls_name,
            "class_id": int(class_ids[idx]),
            "confidence": round(float(confidences[idx]), 4),
            "bbox": {
                "x1": round(float(x1[idx])),
                "y1": round(float(y1[idx])),
                "x2": round(float(x2[idx])),
                "y2": round(float(y2[idx])),
            },
        })

    return sorted(results, key=lambda d: d["confidence"], reverse=True)


@detect_router.get("/health")
def detect_health() -> dict[str, Any]:
    return {
        "status": "ok" if os.path.isfile(_YOLO_MODEL_PATH) else "model_missing",
        "model": "yolov8n",
        "model_path": _YOLO_MODEL_PATH,
        "classes": len(_COCO_CLASSES),
    }


@detect_router.post("/objects")
async def detect_objects(payload: dict[str, Any]) -> dict[str, Any]:
    """Detect objects in an image using YOLOv8n (80 COCO classes).

    Input: {"image_base64": "..."} or {"image_url": "..."}
    Optional: {"confidence": 0.25, "iou_threshold": 0.45, "classes": ["person", "car"]}
    Output: {"detections": [{class, confidence, bbox}, ...]}
    """
    b64 = str(payload.get("image_base64", "")).strip()
    url = str(payload.get("image_url", "")).strip()
    conf_thresh = float(payload.get("confidence", 0.25))
    iou_thresh = float(payload.get("iou_threshold", 0.45))
    filter_classes = payload.get("classes", [])

    if not b64 and not url:
        raise HTTPException(422, "Provide 'image_base64' or 'image_url'")

    if b64:
        img = _decode_b64_image(b64)
    else:
        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                r = await client.get(url)
                r.raise_for_status()
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception as exc:
            raise HTTPException(422, f"Failed to fetch image: {exc}") from exc

    orig_w, orig_h = img.size
    tensor, x_scale, y_scale = _preprocess_yolo(img)

    session = _get_yolo_session()
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: tensor})

    detections = _postprocess_yolo(outputs[0], conf_thresh, iou_thresh,
                                   x_scale, y_scale, orig_w, orig_h)

    # Filter by class if requested
    if filter_classes:
        filter_set = {c.lower() for c in filter_classes}
        detections = [d for d in detections if d["class"] in filter_set]

    # Summary
    class_counts: dict[str, int] = {}
    for d in detections:
        class_counts[d["class"]] = class_counts.get(d["class"], 0) + 1

    return {
        "detections": detections,
        "count": len(detections),
        "classes_found": class_counts,
        "image_size": {"width": orig_w, "height": orig_h},
    }


@detect_router.post("/humans")
async def detect_humans(payload: dict[str, Any]) -> dict[str, Any]:
    """Detect humans/people in an image. Shortcut for detect_objects with class=person.

    Input: {"image_base64": "..."} or {"image_url": "..."}
    Optional: {"confidence": 0.3}
    Output: {"people": [{confidence, bbox}, ...], "count": N}
    """
    payload["classes"] = ["person"]
    if "confidence" not in payload:
        payload["confidence"] = 0.3
    result = await detect_objects(payload)

    people = [
        {"confidence": d["confidence"], "bbox": d["bbox"]}
        for d in result["detections"]
    ]
    return {
        "people": people,
        "count": len(people),
        "image_size": result["image_size"],
    }


app.include_router(detect_router)
