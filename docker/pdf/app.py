"""aichat-pdf: precise PDF/image read/edit service with workspace-safe IO.

Endpoints:
  GET  /health      — service health + GPU/OpenCL status
  POST /read        — extract text from PDF or image (text/OCR/auto)
  POST /edit        — deterministic editing operations for PDF and images
  POST /fill-form   — fill AcroForm fields (PDF only), optional flatten
  POST /merge       — merge multiple PDFs
  POST /split       — split a PDF by page ranges
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import fitz
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

try:
    import cv2 as _cv2
except Exception:  # pragma: no cover - optional accel path
    _cv2 = None  # type: ignore[assignment]

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - dependency should exist in image
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("aichat-pdf")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace")).resolve()
DB_API = os.environ.get("DATABASE_URL", "http://aichat-database:8091")
OCR_URL = os.environ.get("OCR_URL", "http://aichat-ocr:8100")
_INTEL_GPU = os.environ.get("INTEL_GPU", "").strip() == "1"
_SERVICE = "aichat-pdf"

_HOST_WORKSPACE_PREFIX = "/docker/human_browser/workspace"
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}

# Token regex for OCR phrase matching in image text operations.
_TOKEN_RE = re.compile(r"[a-z0-9]+")

app = FastAPI()


# ---------------------------------------------------------------------------
# Error reporting
# ---------------------------------------------------------------------------

async def _report_error(message: str, detail: str | None = None) -> None:
    try:
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
# Generic path and type helpers
# ---------------------------------------------------------------------------

def _to_workspace_path(path: str) -> str:
    p = (path or "").strip()
    if p.startswith(_HOST_WORKSPACE_PREFIX + "/"):
        tail = p[len(_HOST_WORKSPACE_PREFIX) + 1 :]
        return f"/workspace/{tail}"
    return p


def _ensure_within_workspace(path: Path) -> None:
    try:
        path.relative_to(WORKSPACE)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Access denied: path outside workspace") from exc


def _resolve_input_path(path: str) -> Path:
    raw = _to_workspace_path(path)
    if not raw:
        raise HTTPException(status_code=422, detail="'path' is required")

    p = Path(raw)
    if not p.is_absolute():
        p = WORKSPACE / p
    p = p.resolve()
    _ensure_within_workspace(p)

    if p.is_file():
        return p

    alt = (WORKSPACE / p.name).resolve()
    if alt.is_file():
        return alt

    raise HTTPException(status_code=404, detail=f"File not found: {path}")


def _input_kind(source: Path) -> str:
    ext = source.suffix.lower()
    if ext in _IMAGE_EXTS:
        return "image"
    return "pdf"


def _resolve_output_path(
    output_path: str,
    source: Path,
    suffix: str,
    *,
    kind: str,
    default_ext: str,
) -> Path:
    raw = _to_workspace_path(output_path)
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = WORKSPACE / p
    else:
        p = source.with_name(f"{source.stem}{suffix}{default_ext}")

    p = p.resolve()
    _ensure_within_workspace(p)

    # Add default extension if missing.
    if not p.suffix:
        p = p.with_suffix(default_ext)

    ext = p.suffix.lower()
    if kind == "pdf" and ext != ".pdf":
        raise HTTPException(status_code=422, detail="Output path for PDF operations must end with .pdf")
    if kind == "image" and ext not in _IMAGE_EXTS:
        raise HTTPException(
            status_code=422,
            detail=f"Output path extension must be one of: {', '.join(sorted(_IMAGE_EXTS))}",
        )

    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _resolve_output_dir(output_dir: str, source: Path) -> Path:
    raw = _to_workspace_path(output_dir)
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = WORKSPACE / p
    else:
        p = source.parent
    p = p.resolve()
    _ensure_within_workspace(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _container_path(path: Path) -> str:
    rel = path.resolve().relative_to(WORKSPACE).as_posix()
    return f"/workspace/{rel}"


def _host_path(path: Path) -> str:
    rel = path.resolve().relative_to(WORKSPACE).as_posix()
    return f"{_HOST_WORKSPACE_PREFIX}/{rel}"


def _parse_pages(page_count: int, pages: Any) -> list[int]:
    if not pages:
        return list(range(page_count))
    if not isinstance(pages, list):
        raise HTTPException(status_code=422, detail="'pages' must be a list of 1-based page numbers")

    out: list[int] = []
    seen: set[int] = set()
    for item in pages:
        try:
            page_num = int(item)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Invalid page number: {item!r}") from exc
        if page_num < 1 or page_num > page_count:
            raise HTTPException(status_code=422, detail=f"Page out of range: {page_num} (1..{page_count})")
        idx = page_num - 1
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def _parse_image_pages(pages: Any) -> list[int]:
    # Image inputs are single-frame for editing semantics (GIF first frame only).
    if not pages:
        return [0]
    parsed = _parse_pages(1, pages)
    return parsed if parsed else [0]


def _parse_operation_pages(op: dict[str, Any], page_count: int, default_all: bool) -> list[int]:
    if op.get("pages") is not None:
        return _parse_pages(page_count, op.get("pages"))
    if op.get("page") is not None:
        return _parse_pages(page_count, [op.get("page")])
    if default_all:
        return list(range(page_count))
    return [0] if page_count > 0 else []


def _auth_doc(doc: fitz.Document, password: str | None) -> None:
    if doc.needs_pass:
        if not password:
            raise HTTPException(status_code=401, detail="Password required for encrypted PDF")
        ok = bool(doc.authenticate(password))
        if not ok:
            raise HTTPException(status_code=401, detail="Invalid PDF password")


def _assert_pdf_input(source: Path, feature: str) -> None:
    if _input_kind(source) != "pdf":
        raise HTTPException(status_code=422, detail=f"{feature} supports PDF input only")


# ---------------------------------------------------------------------------
# Shared color/text helpers
# ---------------------------------------------------------------------------

def _color(value: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if isinstance(value, str) and value.startswith("#") and len(value) in (7, 9):
        try:
            r = int(value[1:3], 16) / 255.0
            g = int(value[3:5], 16) / 255.0
            b = int(value[5:7], 16) / 255.0
            return (r, g, b)
        except Exception:
            return default
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        out: list[float] = []
        for v in value[:3]:
            try:
                x = float(v)
            except Exception:
                x = 0.0
            out.append(min(1.0, max(0.0, x)))
        return (out[0], out[1], out[2])
    return default


def _color255(value: Any, default: tuple[int, int, int]) -> tuple[int, int, int]:
    c = _color(value, (default[0] / 255.0, default[1] / 255.0, default[2] / 255.0))
    return (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))


def _rect(value: Any) -> fitz.Rect:
    if isinstance(value, dict):
        keys = ("x0", "y0", "x1", "y1")
        if not all(k in value for k in keys):
            raise HTTPException(status_code=422, detail="Rect dict must include x0,y0,x1,y1")
        return fitz.Rect(float(value["x0"]), float(value["y0"]), float(value["x1"]), float(value["y1"]))
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        return fitz.Rect(float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    raise HTTPException(status_code=422, detail="Rect must be [x0,y0,x1,y1] or {x0,y0,x1,y1}")


def _img_box(value: Any, width: int, height: int) -> tuple[int, int, int, int]:
    if isinstance(value, dict):
        if all(k in value for k in ("left", "top", "right", "bottom")):
            x0 = int(float(value["left"]))
            y0 = int(float(value["top"]))
            x1 = int(float(value["right"]))
            y1 = int(float(value["bottom"]))
        elif all(k in value for k in ("x0", "y0", "x1", "y1")):
            x0 = int(float(value["x0"]))
            y0 = int(float(value["y0"]))
            x1 = int(float(value["x1"]))
            y1 = int(float(value["y1"]))
        else:
            raise HTTPException(status_code=422, detail="Image rect dict must include x0,y0,x1,y1 or left,top,right,bottom")
    elif isinstance(value, (list, tuple)) and len(value) >= 4:
        x0 = int(float(value[0]))
        y0 = int(float(value[1]))
        x1 = int(float(value[2]))
        y1 = int(float(value[3]))
    else:
        raise HTTPException(status_code=422, detail="Image rect must be [x0,y0,x1,y1] or {x0,y0,x1,y1}")

    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(1, min(width, x1))
    y1 = max(1, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        raise HTTPException(status_code=422, detail="Image rect has invalid bounds")
    return (x0, y0, x1, y1)


def _insert_text_fit_pdf(
    page: fitz.Page,
    box: fitz.Rect,
    text: str,
    preferred_size: float,
    min_size: float,
    color: tuple[float, float, float],
    font_name: str,
    align: int,
) -> float:
    size = float(preferred_size)
    while size >= min_size:
        shape = page.new_shape()
        spare = shape.insert_textbox(
            box,
            text,
            fontsize=size,
            fontname=font_name,
            color=color,
            align=align,
        )
        if spare >= 0:
            shape.commit(overlay=True)
            return size
        size = round(size - 0.5, 2)
    page.insert_textbox(
        box,
        text,
        fontsize=min_size,
        fontname=font_name,
        color=color,
        align=align,
        overlay=True,
    )
    return min_size


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------

def _extract_text_page(doc: fitz.Document, page_index: int) -> str:
    return doc.load_page(page_index).get_text("text").strip()


def _extract_text(doc: fitz.Document, page_indices: list[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx in page_indices:
        out.append({"page": idx + 1, "text": _extract_text_page(doc, idx), "source": "text"})
    return out


def _needs_ocr(text: str) -> bool:
    return len((text or "").strip()) < 40


async def _ocr_pdf_pages(path: Path, pages: list[int], lang: str) -> dict[int, str]:
    if not pages:
        return {}
    with open(path, "rb") as fh:
        b64_pdf = base64.standard_b64encode(fh.read()).decode("ascii")
    payload = {
        "b64_pdf": b64_pdf,
        "pages": [p + 1 for p in pages],
        "lang": lang,
    }
    try:
        async with httpx.AsyncClient(timeout=240) as client:
            r = await client.post(f"{OCR_URL}/ocr/pdf", json=payload)
            r.raise_for_status()
            data = r.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OCR service failed: {exc}") from exc

    out: dict[int, str] = {}
    for item in data.get("pages", []):
        try:
            idx = int(item.get("page", 0)) - 1
        except Exception:
            continue
        if idx >= 0:
            out[idx] = str(item.get("text", ""))
    return out


async def _ocr_image_text(path: Path, lang: str) -> dict[str, Any]:
    payload = {"path": _container_path(path), "lang": lang}
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{OCR_URL}/ocr/path", json=payload)
            r.raise_for_status()
            data = r.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OCR service failed: {exc}") from exc
    return {
        "text": str(data.get("text", "")),
        "word_count": int(data.get("word_count", 0) or 0),
    }


async def _ocr_image_boxes(path: Path, lang: str) -> dict[str, Any]:
    payload = {"path": _container_path(path), "lang": lang}
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{OCR_URL}/ocr/path/boxes", json=payload)
            r.raise_for_status()
            data = r.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OCR boxes service failed: {exc}") from exc

    boxes: list[dict[str, Any]] = []
    for item in data.get("boxes", []):
        try:
            boxes.append(
                {
                    "text": str(item.get("text", "")),
                    "left": int(item.get("left", 0)),
                    "top": int(item.get("top", 0)),
                    "right": int(item.get("right", 0)),
                    "bottom": int(item.get("bottom", 0)),
                    "conf": float(item.get("conf", -1.0) or -1.0),
                }
            )
        except Exception:
            continue
    return {
        "text": str(data.get("text", "")),
        "word_count": int(data.get("word_count", 0) or 0),
        "boxes": boxes,
    }


def _image_to_ocr_b64(image: Any) -> str:
    _ensure_pillow()
    buf = io.BytesIO()
    # OCR consumes RGB raster data; PNG avoids repeated lossy JPEG transcoding.
    image.convert("RGB").save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


async def _ocr_image_boxes_from_image(image: Any, lang: str) -> dict[str, Any]:
    payload = {"b64": _image_to_ocr_b64(image), "lang": lang}
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{OCR_URL}/ocr/boxes", json=payload)
            r.raise_for_status()
            data = r.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OCR boxes service failed: {exc}") from exc

    boxes: list[dict[str, Any]] = []
    for item in data.get("boxes", []):
        try:
            boxes.append(
                {
                    "text": str(item.get("text", "")),
                    "left": int(item.get("left", 0)),
                    "top": int(item.get("top", 0)),
                    "right": int(item.get("right", 0)),
                    "bottom": int(item.get("bottom", 0)),
                    "conf": float(item.get("conf", -1.0) or -1.0),
                }
            )
        except Exception:
            continue
    return {
        "text": str(data.get("text", "")),
        "word_count": int(data.get("word_count", 0) or 0),
        "boxes": boxes,
    }


# ---------------------------------------------------------------------------
# GPU info
# ---------------------------------------------------------------------------

def _gpu_info() -> dict[str, Any]:
    have_dri = os.path.isdir("/dev/dri")
    info: dict[str, Any] = {
        "gpu_mode": "cpu",
        "opencl_enabled": False,
        "dri_present": have_dri,
        "cv2": bool(_cv2 is not None),
        "cv2_version": "",
    }
    if _cv2 is None:
        if have_dri:
            info["gpu_mode"] = "dri-present"
        return info

    try:
        info["cv2_version"] = str(getattr(_cv2, "__version__", ""))
    except Exception:
        pass

    try:
        have_opencl = bool(_cv2.ocl.haveOpenCL())
        if have_opencl and (_INTEL_GPU or have_dri):
            _cv2.ocl.setUseOpenCL(True)
        use_opencl = bool(_cv2.ocl.useOpenCL()) if have_opencl else False
        info["opencl_enabled"] = use_opencl
        if use_opencl:
            info["gpu_mode"] = "opencv-opencl"
        elif have_dri:
            info["gpu_mode"] = "dri-present"
    except Exception:
        if have_dri:
            info["gpu_mode"] = "dri-present"

    return info


# ---------------------------------------------------------------------------
# PDF edit operations
# ---------------------------------------------------------------------------

def _op_replace_text_pdf(doc: fitz.Document, op: dict[str, Any]) -> dict[str, Any]:
    find = str(op.get("find", "")).strip()
    replace = str(op.get("replace", "")).strip()
    if not find:
        raise HTTPException(status_code=422, detail="replace_text requires non-empty 'find'")

    page_indices = _parse_operation_pages(op, doc.page_count, default_all=True)
    fill = _color(op.get("fill_color"), (1.0, 1.0, 1.0))
    color = _color(op.get("text_color"), (0.0, 0.0, 0.0))
    font_name = str(op.get("font_name", "helv"))
    font_size = float(op.get("font_size", 0.0) or 0.0)
    min_size = float(op.get("min_font_size", 5.0) or 5.0)
    align = int(op.get("align", 0) or 0)
    max_replacements = int(op.get("max_replacements", 0) or 0)

    replaced = 0
    pages_touched: list[int] = []

    for idx in page_indices:
        if max_replacements > 0 and replaced >= max_replacements:
            break

        page = doc.load_page(idx)
        rects = page.search_for(find)
        if not rects:
            continue

        if max_replacements > 0:
            remaining = max_replacements - replaced
            rects = rects[:remaining]

        for r in rects:
            page.add_redact_annot(r, fill=fill)
        page.apply_redactions()

        for r in rects:
            pref = font_size if font_size > 0 else max(min(r.height * 0.82, 28.0), min_size)
            _insert_text_fit_pdf(
                page,
                r,
                replace,
                preferred_size=pref,
                min_size=min_size,
                color=color,
                font_name=font_name,
                align=align,
            )

        replaced += len(rects)
        pages_touched.append(idx + 1)

    return {
        "operation": "replace_text",
        "find": find,
        "replace": replace,
        "replaced": replaced,
        "pages": pages_touched,
    }


def _op_redact_text_pdf(doc: fitz.Document, op: dict[str, Any]) -> dict[str, Any]:
    needle = str(op.get("text", op.get("find", ""))).strip()
    if not needle:
        raise HTTPException(status_code=422, detail="redact_text requires non-empty 'text'")

    page_indices = _parse_operation_pages(op, doc.page_count, default_all=True)
    fill = _color(op.get("fill_color"), (0.0, 0.0, 0.0))

    redacted = 0
    pages_touched: list[int] = []

    for idx in page_indices:
        page = doc.load_page(idx)
        rects = page.search_for(needle)
        if not rects:
            continue
        for r in rects:
            page.add_redact_annot(r, fill=fill)
        page.apply_redactions()
        redacted += len(rects)
        pages_touched.append(idx + 1)

    return {
        "operation": "redact_text",
        "text": needle,
        "redacted": redacted,
        "pages": pages_touched,
    }


def _op_insert_text_pdf(doc: fitz.Document, op: dict[str, Any]) -> dict[str, Any]:
    text = str(op.get("text", "")).strip()
    if not text:
        raise HTTPException(status_code=422, detail="insert_text requires non-empty 'text'")

    page_indices = _parse_operation_pages(op, doc.page_count, default_all=False)
    if not page_indices:
        raise HTTPException(status_code=422, detail="insert_text: no target pages")

    color = _color(op.get("text_color"), (0.0, 0.0, 0.0))
    font_name = str(op.get("font_name", "helv"))
    font_size = float(op.get("font_size", 12.0) or 12.0)
    align = int(op.get("align", 0) or 0)

    inserted = 0
    rect_val = op.get("rect")
    point_val = op.get("point")
    x = float(op.get("x", 36.0) or 36.0)
    y = float(op.get("y", 72.0) or 72.0)

    for idx in page_indices:
        page = doc.load_page(idx)
        if rect_val is not None:
            box = _rect(rect_val)
            _insert_text_fit_pdf(
                page,
                box,
                text,
                preferred_size=font_size,
                min_size=max(5.0, min(font_size, 5.0)),
                color=color,
                font_name=font_name,
                align=align,
            )
        else:
            if isinstance(point_val, (list, tuple)) and len(point_val) >= 2:
                px = float(point_val[0])
                py = float(point_val[1])
            else:
                px, py = x, y
            page.insert_text(
                fitz.Point(px, py),
                text,
                fontsize=font_size,
                fontname=font_name,
                color=color,
                overlay=True,
            )
        inserted += 1

    return {
        "operation": "insert_text",
        "inserted": inserted,
        "pages": [p + 1 for p in page_indices],
    }


def _op_annotate_pdf(doc: fitz.Document, op: dict[str, Any]) -> dict[str, Any]:
    ann_type = str(op.get("type", "highlight")).strip().lower()
    content = str(op.get("content", "")).strip()
    color = _color(op.get("color"), (1.0, 1.0, 0.0))

    page_indices = _parse_operation_pages(op, doc.page_count, default_all=False)
    if not page_indices:
        raise HTTPException(status_code=422, detail="annotate: no target pages")

    source_text = str(op.get("text", "")).strip()
    rect_val = op.get("rect")
    point_val = op.get("point")

    created = 0
    for idx in page_indices:
        page = doc.load_page(idx)

        rects: list[fitz.Rect] = []
        if source_text:
            rects = page.search_for(source_text)
        elif rect_val is not None:
            rects = [_rect(rect_val)]

        if ann_type in {"highlight", "underline", "strikeout", "squiggly", "rectangle"}:
            if not rects:
                continue
            for r in rects:
                if ann_type == "highlight":
                    annot = page.add_highlight_annot(r)
                elif ann_type == "underline":
                    annot = page.add_underline_annot(r)
                elif ann_type == "strikeout":
                    annot = page.add_strikeout_annot(r)
                elif ann_type == "squiggly":
                    annot = page.add_squiggly_annot(r)
                else:
                    annot = page.add_rect_annot(r)
                if content:
                    annot.set_info(content=content, title="aichat-pdf")
                annot.set_colors(stroke=color)
                annot.update()
                created += 1
            continue

        if ann_type == "text":
            if isinstance(point_val, (list, tuple)) and len(point_val) >= 2:
                x = float(point_val[0])
                y = float(point_val[1])
            elif rect_val is not None:
                rc = _rect(rect_val)
                x = float(rc.x0)
                y = float(rc.y0)
            else:
                x = float(op.get("x", 36.0) or 36.0)
                y = float(op.get("y", 72.0) or 72.0)
            annot = page.add_text_annot(fitz.Point(x, y), content or "Note")
            annot.set_colors(stroke=color)
            annot.update()
            created += 1
            continue

        raise HTTPException(status_code=422, detail=f"annotate: unsupported type '{ann_type}'")

    return {
        "operation": "annotate",
        "type": ann_type,
        "created": created,
        "pages": [p + 1 for p in page_indices],
    }


def _op_rotate_page_pdf(doc: fitz.Document, op: dict[str, Any]) -> dict[str, Any]:
    degrees = int(op.get("degrees", 0) or 0)
    if degrees % 90 != 0:
        raise HTTPException(status_code=422, detail="rotate_page requires 'degrees' to be a multiple of 90")

    page_indices = _parse_operation_pages(op, doc.page_count, default_all=False)
    if not page_indices:
        raise HTTPException(status_code=422, detail="rotate_page: no target pages")

    for idx in page_indices:
        page = doc.load_page(idx)
        page.set_rotation((page.rotation + degrees) % 360)

    return {
        "operation": "rotate_page",
        "degrees": degrees,
        "pages": [p + 1 for p in page_indices],
    }


def _op_reorder_pages_pdf(doc: fitz.Document, op: dict[str, Any]) -> tuple[fitz.Document, dict[str, Any]]:
    order = op.get("order")
    if not isinstance(order, list) or not order:
        raise HTTPException(status_code=422, detail="reorder_pages requires non-empty 'order' list")

    try:
        order_int = [int(x) for x in order]
    except Exception as exc:
        raise HTTPException(status_code=422, detail="reorder_pages: 'order' must contain integers") from exc

    page_count = doc.page_count
    expected = set(range(1, page_count + 1))
    if set(order_int) != expected or len(order_int) != page_count:
        raise HTTPException(
            status_code=422,
            detail=f"reorder_pages requires each page exactly once (1..{page_count})",
        )

    new_doc = fitz.open()
    for page_num in order_int:
        new_doc.insert_pdf(doc, from_page=page_num - 1, to_page=page_num - 1)

    doc.close()
    return new_doc, {
        "operation": "reorder_pages",
        "order": order_int,
    }


def _op_delete_pages_pdf(doc: fitz.Document, op: dict[str, Any]) -> dict[str, Any]:
    page_indices = _parse_operation_pages(op, doc.page_count, default_all=False)
    if not page_indices:
        raise HTTPException(status_code=422, detail="delete_pages: no target pages")

    for idx in sorted(set(page_indices), reverse=True):
        doc.delete_page(idx)

    if doc.page_count == 0:
        raise HTTPException(status_code=422, detail="delete_pages would remove all pages")

    return {
        "operation": "delete_pages",
        "deleted": len(set(page_indices)),
        "pages": [p + 1 for p in sorted(set(page_indices))],
        "remaining_pages": doc.page_count,
    }


# ---------------------------------------------------------------------------
# Image operation helpers
# ---------------------------------------------------------------------------

def _ensure_pillow() -> None:
    if Image is None or ImageDraw is None or ImageFont is None:
        raise HTTPException(status_code=500, detail="Pillow is not installed in aichat-pdf")


def _normalize_token(token: str) -> str:
    return "".join(_TOKEN_RE.findall((token or "").lower()))


def _phrase_tokens(text: str) -> list[str]:
    out = []
    for tok in (text or "").split():
        n = _normalize_token(tok)
        if n:
            out.append(n)
    return out


def _token_stream_from_boxes(boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stream: list[dict[str, Any]] = []
    for box in boxes:
        raw = str(box.get("text", ""))
        toks = _TOKEN_RE.findall(raw.lower())
        if not toks:
            continue
        for t in toks:
            stream.append({"token": t, "box": box})
    return stream


def _match_phrase_boxes(boxes: list[dict[str, Any]], phrase: str) -> list[tuple[int, int, int, int]]:
    target = _phrase_tokens(phrase)
    if not target:
        return []
    stream = _token_stream_from_boxes(boxes)
    if not stream:
        return []

    matches: list[tuple[int, int, int, int]] = []
    n = len(target)
    seen: set[tuple[int, int, int, int]] = set()
    for i in range(0, len(stream) - n + 1):
        seg = stream[i : i + n]
        if [s["token"] for s in seg] != target:
            continue
        left = min(int(s["box"].get("left", 0)) for s in seg)
        top = min(int(s["box"].get("top", 0)) for s in seg)
        right = max(int(s["box"].get("right", 0)) for s in seg)
        bottom = max(int(s["box"].get("bottom", 0)) for s in seg)
        rect = (left, top, right, bottom)
        if rect not in seen:
            seen.add(rect)
            matches.append(rect)
    return matches


def _load_font(size: int) -> Any:
    _ensure_pillow()
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    )
    for c in candidates:
        try:
            return ImageFont.truetype(c, size=size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _measure_text(draw: Any, text: str, font: Any, align: str) -> tuple[int, int]:
    try:
        bbox = draw.multiline_textbbox((0, 0), text, font=font, align=align, spacing=4)
        return (max(1, int(bbox[2] - bbox[0])), max(1, int(bbox[3] - bbox[1])))
    except Exception:
        try:
            return draw.multiline_textsize(text, font=font, spacing=4)
        except Exception:
            return (max(1, len(text) * 7), 12)


def _draw_text_fit_image(
    image: Any,
    box: tuple[int, int, int, int],
    text: str,
    preferred_size: float,
    min_size: float,
    fill: tuple[int, int, int],
    align: int,
) -> float:
    _ensure_pillow()
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = box
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    align_map = {0: "left", 1: "center", 2: "right"}
    align_name = align_map.get(int(align), "left")

    size = max(float(min_size), float(preferred_size))
    while size >= float(min_size):
        font = _load_font(max(1, int(round(size))))
        tw, th = _measure_text(draw, text, font, align_name)
        if tw <= bw and th <= bh:
            if align_name == "left":
                tx = x0
            elif align_name == "center":
                tx = x0 + (bw - tw) / 2.0
            else:
                tx = x1 - tw
            ty = y0 + max(0, (bh - th) / 2.0)
            draw.multiline_text((tx, ty), text, font=font, fill=fill, align=align_name, spacing=4)
            return size
        size -= 1.0

    font = _load_font(max(1, int(round(min_size))))
    draw.multiline_text((x0, y0), text, font=font, fill=fill, align=align_name, spacing=4)
    return float(min_size)


def _draw_squiggly(draw: Any, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box
    baseline = y1 - 2
    amp = max(1, int((y1 - y0) / 10))
    step = 6
    points: list[tuple[float, float]] = []
    direction = 1
    x = float(x0)
    while x <= float(x1):
        points.append((x, baseline + amp * direction))
        direction *= -1
        x += step
    if len(points) >= 2:
        draw.line(points, fill=color, width=2)


def _open_image_for_edit(path: Path) -> Any:
    _ensure_pillow()
    with Image.open(path) as img:
        try:
            if getattr(img, "is_animated", False):
                img.seek(0)
        except Exception:
            pass
        # Work in RGBA so highlight/text overlays are consistent across formats.
        return img.convert("RGBA")


def _save_image(path: Path, image: Any) -> None:
    _ensure_pillow()
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg", ".bmp"}:
        image.convert("RGB").save(path)
        return
    if ext == ".gif":
        image.convert("P", palette=Image.Palette.ADAPTIVE).save(path, save_all=False)
        return
    if ext in {".png", ".webp", ".tif", ".tiff"}:
        image.save(path)
        return
    raise HTTPException(status_code=422, detail=f"Unsupported output image extension: {ext}")


async def _verify_image_output(output_path: Path, operations: list[dict[str, Any]], lang: str) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    try:
        ocr = await _ocr_image_text(output_path, lang=lang)
        text_blob = str(ocr.get("text", ""))
        word_count = int(ocr.get("word_count", 0) or 0)
    except Exception as exc:
        return {"ok": False, "checks": [], "error": f"image verification OCR failed: {exc}"}

    ok = True
    for op in operations:
        name = str(op.get("op", "")).strip().lower()
        if name == "replace_text":
            find = str(op.get("find", "")).strip()
            repl = str(op.get("replace", "")).strip()
            c = {
                "operation": "replace_text",
                "find": find,
                "replace": repl,
                "replacement_found": bool(repl and repl.lower() in text_blob.lower()),
                "original_still_present": bool(find and find.lower() in text_blob.lower()),
                "ocr_word_count": word_count,
            }
            if repl and not c["replacement_found"]:
                ok = False
            checks.append(c)
        elif name == "redact_text":
            needle = str(op.get("text", op.get("find", ""))).strip()
            c = {
                "operation": "redact_text",
                "text": needle,
                "text_still_present": bool(needle and needle.lower() in text_blob.lower()),
                "ocr_word_count": word_count,
            }
            if needle and c["text_still_present"]:
                ok = False
            checks.append(c)

    return {"ok": ok, "checks": checks}


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def _extract_all_text_pdf(doc: fitz.Document) -> str:
    return "\n".join(doc.load_page(i).get_text("text") for i in range(doc.page_count))


def _verify_pdf_output(output_path: Path, operations: list[dict[str, Any]]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    try:
        out_doc = fitz.open(str(output_path))
    except Exception as exc:
        return {"ok": False, "checks": [], "error": f"verification open failed: {exc}"}

    text_blob = _extract_all_text_pdf(out_doc)
    out_doc.close()

    ok = True
    for op in operations:
        op_name = str(op.get("op", "")).strip().lower()
        if op_name == "replace_text":
            find = str(op.get("find", "")).strip()
            repl = str(op.get("replace", "")).strip()
            c = {
                "operation": "replace_text",
                "find": find,
                "replace": repl,
                "replacement_found": bool(repl and repl in text_blob),
                "original_still_present": bool(find and find in text_blob),
            }
            if repl and not c["replacement_found"]:
                ok = False
            checks.append(c)
        elif op_name == "redact_text":
            needle = str(op.get("text", op.get("find", ""))).strip()
            c = {
                "operation": "redact_text",
                "text": needle,
                "text_still_present": bool(needle and needle in text_blob),
            }
            if needle and c["text_still_present"]:
                ok = False
            checks.append(c)

    return {"ok": ok, "checks": checks}


# ---------------------------------------------------------------------------
# PDF form helpers
# ---------------------------------------------------------------------------

def _iter_widgets(page: fitz.Page) -> list[Any]:
    widgets = []
    try:
        for w in page.widgets():
            widgets.append(w)
    except Exception:
        w = page.first_widget
        while w:
            widgets.append(w)
            w = w.next
    return widgets


def _flatten_widget_text(page: fitz.Page, widget: Any) -> None:
    value = str(getattr(widget, "field_value", "") or "").strip()
    if not value:
        return
    rect = fitz.Rect(widget.rect)
    _insert_text_fit_pdf(
        page,
        rect,
        value,
        preferred_size=max(8.0, min(rect.height * 0.8, 14.0)),
        min_size=6.0,
        color=(0.0, 0.0, 0.0),
        font_name="helv",
        align=0,
    )


def _parse_ranges(ranges: Any, page_count: int) -> list[tuple[int, int]]:
    if not isinstance(ranges, list) or not ranges:
        raise HTTPException(status_code=422, detail="'ranges' must be a non-empty list")

    out: list[tuple[int, int]] = []
    for item in ranges:
        if isinstance(item, str):
            parts = item.split("-", 1)
            if len(parts) != 2:
                raise HTTPException(status_code=422, detail=f"Invalid range: {item!r}")
            start, end = int(parts[0]), int(parts[1])
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            start, end = int(item[0]), int(item[1])
        elif isinstance(item, dict) and "start" in item and "end" in item:
            start, end = int(item["start"]), int(item["end"])
        else:
            raise HTTPException(status_code=422, detail=f"Invalid range format: {item!r}")

        if start < 1 or end < 1 or start > page_count or end > page_count or start > end:
            raise HTTPException(status_code=422, detail=f"Range out of bounds: {start}-{end} for 1..{page_count}")
        out.append((start - 1, end - 1))

    return out


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, Any]:
    gpu = _gpu_info()
    return {
        "status": "ok",
        "service": _SERVICE,
        "workspace": str(WORKSPACE),
        "gpu_mode": gpu.get("gpu_mode", "cpu"),
        "opencl_enabled": bool(gpu.get("opencl_enabled", False)),
        "versions": {
            "pymupdf": getattr(fitz, "VersionBind", "unknown"),
            "cv2": gpu.get("cv2_version", ""),
        },
    }


@app.post("/read")
async def read_document(payload: dict[str, Any]) -> dict[str, Any]:
    path = str(payload.get("path", "")).strip()
    mode = str(payload.get("mode", "auto")).strip().lower() or "auto"
    pages_raw = payload.get("pages")
    lang = str(payload.get("lang", "eng")).strip() or "eng"
    password = payload.get("password")
    password = str(password) if password is not None else None

    if mode not in {"text", "ocr", "auto"}:
        raise HTTPException(status_code=422, detail="mode must be one of: text, ocr, auto")

    source = _resolve_input_path(path)
    kind = _input_kind(source)
    t0 = time.perf_counter()

    if kind == "image":
        _ensure_pillow()
        _parse_image_pages(pages_raw)
        ocr = await _ocr_image_text(source, lang=lang)
        text = str(ocr.get("text", ""))
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "input_kind": "image",
            "path": _container_path(source),
            "host_path": _host_path(source),
            "mode": mode,
            "effective_mode": "ocr",
            "lang": lang,
            "page_count": 1,
            "pages": [{"page": 1, "text": text, "source": "ocr"}],
            "text": text,
            "ocr_used": True,
            "timings_ms": {"total": elapsed_ms},
        }

    # PDF path
    try:
        doc = fitz.open(str(source))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot open PDF: {exc}") from exc

    _auth_doc(doc, password)
    page_indices = _parse_pages(doc.page_count, pages_raw)
    text_pages = _extract_text(doc, page_indices)

    ocr_used = False
    effective_mode = mode
    if mode == "text":
        pages = text_pages
    elif mode == "ocr":
        ocr_map = await _ocr_pdf_pages(source, page_indices, lang)
        pages = [{"page": idx + 1, "text": ocr_map.get(idx, ""), "source": "ocr"} for idx in page_indices]
        ocr_used = True
    else:
        fallback = [item["page"] - 1 for item in text_pages if _needs_ocr(item["text"])]
        ocr_map = await _ocr_pdf_pages(source, fallback, lang) if fallback else {}
        pages = []
        for item in text_pages:
            idx = item["page"] - 1
            if idx in ocr_map:
                pages.append({"page": item["page"], "text": ocr_map[idx], "source": "ocr"})
            else:
                pages.append(item)
        ocr_used = bool(ocr_map)
        if ocr_used:
            effective_mode = "auto+ocr"

    doc.close()
    full_text = "\n\n---\n\n".join(item["text"] for item in pages if item["text"])
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "input_kind": "pdf",
        "path": _container_path(source),
        "host_path": _host_path(source),
        "mode": mode,
        "effective_mode": effective_mode,
        "lang": lang,
        "page_count": len(pages),
        "pages": pages,
        "text": full_text,
        "ocr_used": ocr_used,
        "timings_ms": {"total": elapsed_ms},
    }


@app.post("/edit")
async def edit_document(payload: dict[str, Any]) -> dict[str, Any]:
    path = str(payload.get("path", "")).strip()
    output_path = str(payload.get("output_path", "")).strip()
    operations = payload.get("operations")
    verify = bool(payload.get("verify", True))
    lang = str(payload.get("lang", "eng")).strip() or "eng"
    password = payload.get("password")
    password = str(password) if password is not None else None

    if not isinstance(operations, list) or not operations:
        raise HTTPException(status_code=422, detail="'operations' must be a non-empty list")

    source = _resolve_input_path(path)
    kind = _input_kind(source)
    default_ext = source.suffix.lower() if kind == "image" else ".pdf"
    out = _resolve_output_path(output_path, source, suffix="_edited", kind=kind, default_ext=default_ext)

    if kind == "image":
        _ensure_pillow()
        image = _open_image_for_edit(source)
        warnings: list[str] = []
        op_results: list[dict[str, Any]] = []

        for idx, raw_op in enumerate(operations, start=1):
            if not isinstance(raw_op, dict):
                raise HTTPException(status_code=422, detail=f"operation #{idx} must be an object")
            op_name = str(raw_op.get("op", "")).strip().lower()
            if not op_name:
                raise HTTPException(status_code=422, detail=f"operation #{idx}: missing 'op'")

            # Enforce single-frame semantics for image operations.
            _parse_operation_pages(raw_op, 1, default_all=True)

            width, height = image.size
            draw = ImageDraw.Draw(image, "RGBA")

            if op_name in ("replace_text", "redact_text"):
                needle = str(raw_op.get("find", raw_op.get("text", ""))).strip()
                if not needle:
                    raise HTTPException(status_code=422, detail=f"{op_name} requires non-empty 'find' or 'text'")

                ocr_lang = str(raw_op.get("lang", lang)).strip() or lang
                ocr = await _ocr_image_boxes_from_image(image, lang=ocr_lang)
                boxes = _match_phrase_boxes(ocr.get("boxes", []), needle)
                if not boxes:
                    op_results.append(
                        {
                            "operation": op_name,
                            "text": needle,
                            "matched": 0,
                            "warning": "No OCR phrase match found",
                            "pages": [1],
                        }
                    )
                    continue

                fill_rgb = _color255(raw_op.get("fill_color"), (255, 255, 255) if op_name == "replace_text" else (0, 0, 0))
                for b in boxes:
                    draw.rectangle(b, fill=(fill_rgb[0], fill_rgb[1], fill_rgb[2], 255))

                if op_name == "replace_text":
                    repl = str(raw_op.get("replace", "")).strip()
                    text_rgb = _color255(raw_op.get("text_color"), (0, 0, 0))
                    font_size = float(raw_op.get("font_size", 0.0) or 0.0)
                    min_size = float(raw_op.get("min_font_size", 8.0) or 8.0)
                    align = int(raw_op.get("align", 0) or 0)
                    for b in boxes:
                        preferred = font_size if font_size > 0 else max(12.0, min((b[3] - b[1]) * 0.85, 48.0))
                        _draw_text_fit_image(image, b, repl, preferred, min_size, text_rgb, align)
                    op_results.append(
                        {
                            "operation": "replace_text",
                            "find": needle,
                            "replace": repl,
                            "replaced": len(boxes),
                            "pages": [1],
                        }
                    )
                else:
                    op_results.append(
                        {
                            "operation": "redact_text",
                            "text": needle,
                            "redacted": len(boxes),
                            "pages": [1],
                        }
                    )
                continue

            if op_name == "insert_text":
                text = str(raw_op.get("text", "")).strip()
                if not text:
                    raise HTTPException(status_code=422, detail="insert_text requires non-empty 'text'")
                text_rgb = _color255(raw_op.get("text_color"), (0, 0, 0))
                font_size = float(raw_op.get("font_size", 24.0) or 24.0)
                align = int(raw_op.get("align", 0) or 0)
                rect_val = raw_op.get("rect")
                point_val = raw_op.get("point")
                if rect_val is not None:
                    box = _img_box(rect_val, width, height)
                else:
                    if isinstance(point_val, (list, tuple)) and len(point_val) >= 2:
                        px = int(float(point_val[0]))
                        py = int(float(point_val[1]))
                    else:
                        px = int(float(raw_op.get("x", 36) or 36))
                        py = int(float(raw_op.get("y", 72) or 72))
                    box = (
                        max(0, px),
                        max(0, py),
                        min(width, max(px + int(max(50, len(text) * font_size * 0.6)), px + 50)),
                        min(height, max(py + int(max(20, font_size * 1.5)), py + 20)),
                    )
                _draw_text_fit_image(
                    image,
                    box,
                    text,
                    preferred_size=font_size,
                    min_size=max(8.0, min(font_size, 8.0)),
                    fill=text_rgb,
                    align=align,
                )
                op_results.append({"operation": "insert_text", "inserted": 1, "pages": [1]})
                continue

            if op_name == "annotate":
                ann_type = str(raw_op.get("type", "highlight")).strip().lower()
                content = str(raw_op.get("content", "")).strip()
                color = _color255(raw_op.get("color"), (255, 255, 0))

                target_boxes: list[tuple[int, int, int, int]] = []
                source_text = str(raw_op.get("text", "")).strip()
                if source_text:
                    ocr_lang = str(raw_op.get("lang", lang)).strip() or lang
                    ocr = await _ocr_image_boxes_from_image(image, lang=ocr_lang)
                    target_boxes = _match_phrase_boxes(ocr.get("boxes", []), source_text)
                elif raw_op.get("rect") is not None:
                    target_boxes = [_img_box(raw_op.get("rect"), width, height)]

                if ann_type == "text":
                    point_val = raw_op.get("point")
                    if isinstance(point_val, (list, tuple)) and len(point_val) >= 2:
                        px = int(float(point_val[0]))
                        py = int(float(point_val[1]))
                    elif target_boxes:
                        px, py = target_boxes[0][0], target_boxes[0][1]
                    else:
                        px = int(float(raw_op.get("x", 36) or 36))
                        py = int(float(raw_op.get("y", 72) or 72))
                    text_to_draw = content or "Note"
                    _draw_text_fit_image(
                        image,
                        (max(0, px), max(0, py), min(width, px + 300), min(height, py + 80)),
                        text_to_draw,
                        preferred_size=float(raw_op.get("font_size", 20.0) or 20.0),
                        min_size=8.0,
                        fill=color,
                        align=0,
                    )
                    op_results.append({"operation": "annotate", "type": "text", "created": 1, "pages": [1]})
                    continue

                if not target_boxes:
                    op_results.append(
                        {
                            "operation": "annotate",
                            "type": ann_type,
                            "created": 0,
                            "warning": "No target region found",
                            "pages": [1],
                        }
                    )
                    continue

                created = 0
                for b in target_boxes:
                    if ann_type == "rectangle":
                        draw.rectangle(b, outline=(color[0], color[1], color[2], 255), width=2)
                    elif ann_type == "highlight":
                        draw.rectangle(b, fill=(color[0], color[1], color[2], 90), outline=(color[0], color[1], color[2], 200), width=1)
                    elif ann_type == "underline":
                        draw.line([(b[0], b[3] - 1), (b[2], b[3] - 1)], fill=(color[0], color[1], color[2], 255), width=2)
                    elif ann_type == "strikeout":
                        mid = b[1] + (b[3] - b[1]) // 2
                        draw.line([(b[0], mid), (b[2], mid)], fill=(color[0], color[1], color[2], 255), width=2)
                    elif ann_type == "squiggly":
                        _draw_squiggly(draw, b, color)
                    else:
                        raise HTTPException(status_code=422, detail=f"annotate: unsupported type '{ann_type}'")
                    created += 1
                op_results.append({"operation": "annotate", "type": ann_type, "created": created, "pages": [1]})
                continue

            if op_name == "rotate_page":
                degrees = int(raw_op.get("degrees", 0) or 0)
                if degrees % 90 != 0:
                    raise HTTPException(status_code=422, detail="rotate_page requires 'degrees' to be a multiple of 90")
                # Keep behavior aligned with PDF: positive = clockwise.
                image = image.rotate(-degrees, expand=True)
                op_results.append({"operation": "rotate_page", "degrees": degrees, "pages": [1]})
                continue

            if op_name in ("reorder_pages", "delete_pages"):
                raise HTTPException(status_code=422, detail=f"{op_name} is PDF-only and unsupported for image input")

            raise HTTPException(status_code=422, detail=f"Unsupported edit operation: '{op_name}'")

        _save_image(out, image)

        verification = (
            await _verify_image_output(out, operations, lang=lang)
            if verify
            else {"ok": True, "checks": [], "skipped": True}
        )

        return {
            "input_kind": "image",
            "output_path": _container_path(out),
            "host_path": _host_path(out),
            "operations_applied": op_results,
            "verification": verification,
            "warnings": warnings,
        }

    # PDF edit path
    try:
        doc = fitz.open(str(source))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot open PDF: {exc}") from exc

    _auth_doc(doc, password)

    op_results: list[dict[str, Any]] = []
    for idx, raw_op in enumerate(operations, start=1):
        if not isinstance(raw_op, dict):
            raise HTTPException(status_code=422, detail=f"operation #{idx} must be an object")
        op_name = str(raw_op.get("op", "")).strip().lower()
        if not op_name:
            raise HTTPException(status_code=422, detail=f"operation #{idx}: missing 'op'")

        if op_name == "replace_text":
            op_results.append(_op_replace_text_pdf(doc, raw_op))
        elif op_name == "redact_text":
            op_results.append(_op_redact_text_pdf(doc, raw_op))
        elif op_name == "annotate":
            op_results.append(_op_annotate_pdf(doc, raw_op))
        elif op_name == "insert_text":
            op_results.append(_op_insert_text_pdf(doc, raw_op))
        elif op_name == "rotate_page":
            op_results.append(_op_rotate_page_pdf(doc, raw_op))
        elif op_name == "reorder_pages":
            doc, result = _op_reorder_pages_pdf(doc, raw_op)
            op_results.append(result)
        elif op_name == "delete_pages":
            op_results.append(_op_delete_pages_pdf(doc, raw_op))
        else:
            raise HTTPException(status_code=422, detail=f"Unsupported edit operation: '{op_name}'")

    try:
        doc.save(str(out), garbage=4, deflate=True)
    except Exception as exc:
        doc.close()
        raise HTTPException(status_code=500, detail=f"Failed to save edited PDF: {exc}") from exc
    doc.close()

    verification = _verify_pdf_output(out, operations) if verify else {"ok": True, "checks": [], "skipped": True}

    return {
        "input_kind": "pdf",
        "output_path": _container_path(out),
        "host_path": _host_path(out),
        "operations_applied": op_results,
        "verification": verification,
        "warnings": [],
    }


@app.post("/fill-form")
async def fill_form(payload: dict[str, Any]) -> dict[str, Any]:
    path = str(payload.get("path", "")).strip()
    output_path = str(payload.get("output_path", "")).strip()
    flatten = bool(payload.get("flatten", False))
    fields = payload.get("fields") or {}
    password = payload.get("password")
    password = str(password) if password is not None else None

    if not isinstance(fields, dict) or not fields:
        raise HTTPException(status_code=422, detail="'fields' must be a non-empty object")

    source = _resolve_input_path(path)
    _assert_pdf_input(source, "fill-form")
    out = _resolve_output_path(output_path, source, suffix="_filled", kind="pdf", default_ext=".pdf")

    try:
        doc = fitz.open(str(source))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot open PDF: {exc}") from exc

    _auth_doc(doc, password)

    written: set[str] = set()
    warnings: list[str] = []

    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        widgets = _iter_widgets(page)
        for widget in widgets:
            name = str(getattr(widget, "field_name", "") or "").strip()
            if not name or name not in fields:
                continue
            raw_val = fields[name]
            if isinstance(raw_val, bool):
                value = "Yes" if raw_val else "Off"
            else:
                value = str(raw_val)
            try:
                widget.field_value = value
                widget.update()
                written.add(name)
            except Exception as exc:
                warnings.append(f"field '{name}' update failed on page {page_index + 1}: {exc}")

    if flatten:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            widgets = _iter_widgets(page)
            for widget in widgets:
                name = str(getattr(widget, "field_name", "") or "").strip()
                if name and name in written:
                    try:
                        _flatten_widget_text(page, widget)
                    except Exception as exc:
                        warnings.append(f"field '{name}' flatten draw failed on page {page_index + 1}: {exc}")
                try:
                    page.delete_widget(widget)
                except Exception:
                    pass

    try:
        doc.save(str(out), garbage=4, deflate=True)
    except Exception as exc:
        doc.close()
        raise HTTPException(status_code=500, detail=f"Failed to save filled PDF: {exc}") from exc
    doc.close()

    return {
        "input_kind": "pdf",
        "output_path": _container_path(out),
        "host_path": _host_path(out),
        "fields_written": sorted(written),
        "flattened": flatten,
        "warnings": warnings,
    }


@app.post("/merge")
async def merge_pdfs(payload: dict[str, Any]) -> dict[str, Any]:
    paths = payload.get("paths")
    output_path = str(payload.get("output_path", "")).strip()
    passwords = payload.get("passwords") or {}

    if not isinstance(paths, list) or len(paths) < 2:
        raise HTTPException(status_code=422, detail="'paths' must be a list with at least 2 PDFs")

    resolved: list[Path] = []
    for p in paths:
        src = _resolve_input_path(str(p))
        _assert_pdf_input(src, "merge")
        resolved.append(src)

    ts = int(time.time())
    base_source = resolved[0]
    out = _resolve_output_path(
        output_path,
        base_source.with_name(f"merged_{ts}.pdf"),
        suffix="",
        kind="pdf",
        default_ext=".pdf",
    )

    merged = fitz.open()
    try:
        for src in resolved:
            doc = fitz.open(str(src))
            pwd = ""
            if isinstance(passwords, dict):
                pwd = str(passwords.get(str(src), passwords.get(_container_path(src), "")) or "")
            if doc.needs_pass:
                if not pwd:
                    raise HTTPException(status_code=401, detail=f"Password required for encrypted PDF: {src.name}")
                if not doc.authenticate(pwd):
                    raise HTTPException(status_code=401, detail=f"Invalid password for PDF: {src.name}")
            merged.insert_pdf(doc)
            doc.close()

        merged.save(str(out), garbage=4, deflate=True)
    finally:
        merged.close()

    page_count = 0
    with fitz.open(str(out)) as merged_doc:
        page_count = merged_doc.page_count

    return {
        "input_kind": "pdf",
        "output_path": _container_path(out),
        "host_path": _host_path(out),
        "inputs": [_container_path(p) for p in resolved],
        "page_count": page_count,
    }


@app.post("/split")
async def split_pdf(payload: dict[str, Any]) -> dict[str, Any]:
    path = str(payload.get("path", "")).strip()
    ranges = payload.get("ranges")
    prefix = str(payload.get("prefix", "")).strip()
    output_dir = str(payload.get("output_dir", "")).strip()
    password = payload.get("password")
    password = str(password) if password is not None else None

    source = _resolve_input_path(path)
    _assert_pdf_input(source, "split")

    try:
        doc = fitz.open(str(source))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot open PDF: {exc}") from exc

    _auth_doc(doc, password)
    spans = _parse_ranges(ranges, doc.page_count)
    out_dir = _resolve_output_dir(output_dir, source)

    stem_prefix = prefix or f"{source.stem}_part"
    outputs: list[dict[str, Any]] = []

    for i, (start, end) in enumerate(spans, start=1):
        part = fitz.open()
        part.insert_pdf(doc, from_page=start, to_page=end)
        name = f"{stem_prefix}_{i:02d}_{start + 1}-{end + 1}.pdf"
        out = (out_dir / name).resolve()
        _ensure_within_workspace(out)
        part.save(str(out), garbage=4, deflate=True)
        part.close()
        outputs.append(
            {
                "path": _container_path(out),
                "host_path": _host_path(out),
                "range": [start + 1, end + 1],
                "pages": end - start + 1,
            }
        )

    doc.close()

    return {
        "input_kind": "pdf",
        "input_path": _container_path(source),
        "outputs": outputs,
        "count": len(outputs),
    }
