"""Image preprocessing utilities for Triton inference models."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image


def resize_for_wd_tagger(img: Image.Image, target: int = 448) -> np.ndarray:
    """Prepare an image for WD ViT Tagger inference.

    Pads the image to a square with a white background, then resizes to
    ``target × target`` using bicubic resampling.  Unlike the other models in
    this stack, WD tagger expects raw pixel values in [0, 255] (no
    normalisation) in NHWC layout.

    Args:
        img:    PIL Image in RGB mode.
        target: Target square resolution (default 448).

    Returns:
        ``np.ndarray`` of shape ``(1, target, target, 3)``, dtype ``float32``,
        values in ``[0, 255]``.
    """
    w, h = img.size
    max_dim = max(w, h)
    canvas = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    canvas.paste(img, ((max_dim - w) // 2, (max_dim - h) // 2))
    resized = canvas.resize((target, target), Image.Resampling.BICUBIC)
    arr = np.asarray(resized, dtype=np.float32)  # (H, W, 3) values 0–255
    return arr[np.newaxis, ...]  # (1, H, W, 3)


def decode_b64_image(b64: str) -> Image.Image:
    """Decode a base64-encoded image string into a PIL Image.

    Accepts raw base64 or data-URI strings of the form
    ``data:image/<fmt>;base64,<data>``.  The image is converted to RGB
    so all downstream pipelines receive consistent 3-channel input.

    Args:
        b64: Base64-encoded image bytes, optionally prefixed with a
             data-URI header.

    Returns:
        A PIL Image in RGB mode.

    Raises:
        ValueError: If the string cannot be decoded as a valid image.
    """
    # Strip optional data-URI prefix (e.g. "data:image/jpeg;base64,")
    if "," in b64:
        b64 = b64.split(",", 1)[1]

    try:
        raw_bytes = base64.b64decode(b64)
    except Exception as exc:
        raise ValueError(f"Invalid base64 encoding: {exc}") from exc

    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img.load()  # force decode so we catch corrupt images early
    except Exception as exc:
        raise ValueError(f"Cannot decode image data: {exc}") from exc

    return img.convert("RGB")


def resize_for_yolo(
    img: Image.Image,
    target: int = 640,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize an image to ``target × target`` with letterboxing for YOLOv8.

    The image is scaled so the longest side equals ``target``, then padded
    symmetrically with gray (114/255) to make it square.  This matches the
    standard YOLOv8 preprocessing pipeline.

    Args:
        img:    PIL Image in RGB mode.
        target: Target square resolution (default 640).

    Returns:
        A 3-tuple:
        - ``chw_array`` – ``np.ndarray`` of shape ``(3, target, target)``,
          dtype ``float32``, values in ``[0, 1]``.
        - ``scale`` – float scaling factor applied to the original image
          (used to map boxes back to original coordinates).
        - ``(pad_x, pad_y)`` – pixels of padding added on each side
          horizontally and vertically respectively.
    """
    orig_w, orig_h = img.size
    scale = min(target / orig_w, target / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Create gray canvas
    canvas = Image.new("RGB", (target, target), (114, 114, 114))
    pad_x = (target - new_w) // 2
    pad_y = (target - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))

    # HWC → CHW float32 [0, 1]
    arr = np.asarray(canvas, dtype=np.float32) / 255.0  # (H, W, 3)
    chw = arr.transpose(2, 0, 1)  # (3, H, W)

    return chw, scale, (pad_x, pad_y)


def resize_for_efficientnet(img: Image.Image, target: int = 224) -> np.ndarray:
    """Prepare an image for EfficientNet-B0 inference.

    Applies the standard ImageNet preprocessing pipeline:
    1. Resize shortest side to ``target`` preserving aspect ratio.
    2. Center-crop to ``target × target``.
    3. Normalise with ImageNet mean and standard deviation.

    Args:
        img:    PIL Image in RGB mode.
        target: Output spatial resolution (default 224).

    Returns:
        ``np.ndarray`` of shape ``(1, 3, target, target)``, dtype
        ``float32``, in NCHW layout, normalised to ImageNet statistics.
    """
    # Resize so shorter side == target
    orig_w, orig_h = img.size
    if orig_w < orig_h:
        new_w = target
        new_h = int(round(orig_h * target / orig_w))
    else:
        new_h = target
        new_w = int(round(orig_w * target / orig_h))

    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Center crop
    left = (new_w - target) // 2
    top = (new_h - target) // 2
    cropped = resized.crop((left, top, left + target, top + target))

    arr = np.asarray(cropped, dtype=np.float32) / 255.0  # (H, W, 3) in [0,1]

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std  # (H, W, 3)

    chw = arr.transpose(2, 0, 1)  # (3, H, W)
    nchw = chw[np.newaxis, ...]   # (1, 3, H, W)

    return nchw


def resize_for_clip(img: Image.Image, target: int = 224) -> np.ndarray:
    """Prepare an image for CLIP ViT-B/32 inference.

    Applies the standard OpenAI CLIP preprocessing pipeline:
    1. Resize shortest side to ``target`` preserving aspect ratio.
    2. Center-crop to ``target × target``.
    3. Normalise with CLIP-specific mean and standard deviation.

    Args:
        img:    PIL Image in RGB mode.
        target: Output spatial resolution (default 224).

    Returns:
        ``np.ndarray`` of shape ``(1, 3, target, target)``, dtype
        ``float32``, in NCHW layout, normalised to CLIP statistics.
    """
    # Resize so shorter side == target (bicubic matches original CLIP)
    orig_w, orig_h = img.size
    if orig_w < orig_h:
        new_w = target
        new_h = int(round(orig_h * target / orig_w))
    else:
        new_h = target
        new_w = int(round(orig_w * target / orig_h))

    resized = img.resize((new_w, new_h), Image.Resampling.BICUBIC)

    # Center crop
    left = (new_w - target) // 2
    top = (new_h - target) // 2
    cropped = resized.crop((left, top, left + target, top + target))

    arr = np.asarray(cropped, dtype=np.float32) / 255.0  # (H, W, 3) in [0,1]

    # CLIP normalisation constants
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    arr = (arr - mean) / std  # (H, W, 3)

    chw = arr.transpose(2, 0, 1)  # (3, H, W)
    nchw = chw[np.newaxis, ...]   # (1, 3, H, W)

    return nchw
