"""aichat-video: FFmpeg-based video analysis and frame extraction.

Endpoints:
  POST /info       — get video metadata (duration, fps, resolution, codec)
  POST /frames     — extract frames at regular intervals
  POST /thumbnail  — get a single frame as base64 PNG
  GET  /health     — service health + ffmpeg version
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("aichat-video")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
DB_API    = os.environ.get("DATABASE_URL", "http://aichat-database:8091")
_SERVICE  = "aichat-video"
_INTEL_GPU = os.environ.get("INTEL_GPU", "").strip() == "1"
_VAAPI_DEVICE = os.environ.get("INTEL_VAAPI_DEVICE", "/dev/dri/renderD128")

app = FastAPI()


# ---------------------------------------------------------------------------
# Error reporting
# ---------------------------------------------------------------------------

async def _report_error(message: str, detail: str | None = None) -> None:
    try:
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
# FFmpeg helpers
# ---------------------------------------------------------------------------

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
    # Keep this conservative and portable: ffmpeg auto-selects a supported HW
    # decoder when available, and falls back to software for unsupported codecs.
    if _gpu_mode() == "intel-vaapi":
        return ["-hwaccel", "auto"]
    return []


def _ffprobe(url_or_path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        url_or_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {r.stderr[:300]}")
    return json.loads(r.stdout)


def _parse_info(probe: dict) -> dict:
    fmt  = probe.get("format", {})
    streams = probe.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})

    duration = 0.0
    try:
        duration = float(fmt.get("duration") or video.get("duration", 0))
    except (ValueError, TypeError):
        pass

    fps = 0.0
    try:
        r_frame_rate = video.get("r_frame_rate", "0/1")
        num, den = r_frame_rate.split("/")
        fps = round(float(num) / float(den), 3) if float(den) else 0.0
    except Exception:
        pass

    size_bytes = 0
    try:
        size_bytes = int(fmt.get("size", 0))
    except (ValueError, TypeError):
        pass

    return {
        "duration_s": round(duration, 3),
        "fps":        fps,
        "width":      int(video.get("width",  0)),
        "height":     int(video.get("height", 0)),
        "codec":      video.get("codec_name", ""),
        "audio_codec": audio.get("codec_name", ""),
        "format":     fmt.get("format_name", ""),
        "size_mb":    round(size_bytes / 1_048_576, 3),
    }


async def _download_to_tmp(url: str) -> str:
    """Download a remote URL to a temp file. Returns temp file path."""
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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "ffmpeg_version": _ffmpeg_version(),
        "gpu_mode": _gpu_mode(),
        "vaapi_device": _VAAPI_DEVICE if _gpu_mode() == "intel-vaapi" else "",
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
        info  = _parse_info(probe)
        return {"url": url, **info}
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
        cmd = [
            "ffmpeg", *_ffmpeg_hwaccel_args(), "-i", target,
            "-vf", f"fps=1/{interval_sec}",
            "-frames:v", str(max_frames),
            "-q:v", "2",
            out_pattern,
            "-y",
        ]
        r = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: subprocess.run(cmd, capture_output=True, text=True, timeout=120),
        )
        if r.returncode != 0 and _ffmpeg_hwaccel_args():
            # Retry once on CPU path if the selected hardware decoder rejects
            # this codec/container combination.
            cpu_cmd = [
                "ffmpeg", "-i", target,
                "-vf", f"fps=1/{interval_sec}",
                "-frames:v", str(max_frames),
                "-q:v", "2",
                out_pattern,
                "-y",
            ]
            r = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: subprocess.run(cpu_cmd, capture_output=True, text=True, timeout=120),
            )
        if r.returncode != 0 and not list(out_dir.iterdir()):
            raise RuntimeError(f"ffmpeg failed: {r.stderr[:400]}")

        frames = []
        for i, f in enumerate(sorted(out_dir.iterdir())):
            frames.append({
                "path":        str(f),
                "timestamp_s": round(i * interval_sec, 3),
                "filename":    f.name,
            })

        return {
            "url":          url,
            "interval_sec": interval_sec,
            "frames":       frames,
            "count":        len(frames),
            "output_dir":   str(out_dir),
        }
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

        cmd = [
            "ffmpeg",
            "-ss", str(timestamp_sec),
            *_ffmpeg_hwaccel_args(),
            "-i", target,
            "-vframes", "1",
            "-q:v", "2",
            thumb_tmp.name,
            "-y",
        ]
        r = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: subprocess.run(cmd, capture_output=True, text=True, timeout=30),
        )
        if (r.returncode != 0 or not os.path.getsize(thumb_tmp.name)) and _ffmpeg_hwaccel_args():
            cpu_cmd = [
                "ffmpeg",
                "-ss", str(timestamp_sec),
                "-i", target,
                "-vframes", "1",
                "-q:v", "2",
                thumb_tmp.name,
                "-y",
            ]
            r = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: subprocess.run(cpu_cmd, capture_output=True, text=True, timeout=30),
            )
        if r.returncode != 0 or not os.path.getsize(thumb_tmp.name):
            raise RuntimeError(f"ffmpeg thumbnail failed: {r.stderr[:300]}")

        raw = Path(thumb_tmp.name).read_bytes()
        b64 = base64.standard_b64encode(raw).decode("ascii")

        # Get dimensions
        probe = _ffprobe(thumb_tmp.name)
        info  = _parse_info(probe)

        return {
            "url":           url,
            "timestamp_sec": timestamp_sec,
            "b64":           b64,
            "width":         info["width"],
            "height":        info["height"],
            "mime_type":     "image/png",
        }
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        if thumb_tmp:
            try:
                os.unlink(thumb_tmp.name)
            except Exception:
                pass
