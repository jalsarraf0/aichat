"""Unit tests for vision/services/vision-router/app/preprocessing.py."""
from __future__ import annotations

import base64
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Load preprocessing module explicitly from vision-router to avoid
# collision with the mcp-server's app package.
# ---------------------------------------------------------------------------

_repo_root = Path(__file__).parent.parent.parent.parent  # .../aichat
_preprocessing_path = _repo_root / "vision" / "services" / "vision-router" / "app" / "preprocessing.py"

_spec = importlib.util.spec_from_file_location("vision_router_preprocessing", _preprocessing_path)
assert _spec is not None and _spec.loader is not None, f"Cannot find {_preprocessing_path}"
_preprocessing = importlib.util.module_from_spec(_spec)
sys.modules["vision_router_preprocessing"] = _preprocessing
_spec.loader.exec_module(_preprocessing)  # type: ignore[union-attr]

decode_b64_image = _preprocessing.decode_b64_image
resize_for_yolo = _preprocessing.resize_for_yolo
resize_for_efficientnet = _preprocessing.resize_for_efficientnet
resize_for_clip = _preprocessing.resize_for_clip


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pil(width: int = 8, height: int = 8, color: tuple[int, int, int] = (200, 100, 50)) -> Image.Image:
    """Create a tiny solid-color RGB PIL image."""
    return Image.new("RGB", (width, height), color)


def _make_b64_jpeg(width: int = 8, height: int = 8, color: tuple[int, int, int] = (200, 100, 50)) -> str:
    img = _make_pil(width, height, color)
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# decode_b64_image
# ---------------------------------------------------------------------------

class TestDecodeB64Image:
    def test_plain_base64_jpeg(self):
        b64 = _make_b64_jpeg()
        img = decode_b64_image(b64)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_data_uri_jpeg(self):
        b64 = "data:image/jpeg;base64," + _make_b64_jpeg()
        img = decode_b64_image(b64)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_image_dimensions_preserved(self):
        b64 = _make_b64_jpeg(width=32, height=16)
        img = decode_b64_image(b64)
        # JPEG may slightly shift dimensions due to chroma subsampling alignment
        assert img.size == (32, 16)

    def test_converts_to_rgb(self):
        """Even a grayscale PNG input should come out RGB."""
        gray_img = Image.new("L", (8, 8), 128)
        buf = io.BytesIO()
        gray_img.save(buf, "PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        img = decode_b64_image(b64)
        assert img.mode == "RGB"

    def test_invalid_base64_raises(self):
        with pytest.raises(ValueError, match="Invalid base64"):
            decode_b64_image("not-valid-base64!!!!")

    def test_corrupt_image_raises(self):
        garbage = base64.b64encode(b"\x00" * 50).decode()
        with pytest.raises(ValueError, match="Cannot decode"):
            decode_b64_image(garbage)


# ---------------------------------------------------------------------------
# resize_for_yolo
# ---------------------------------------------------------------------------

class TestResizeForYolo:
    def test_output_shape(self):
        img = _make_pil(640, 480)
        chw, scale, (pad_x, pad_y) = resize_for_yolo(img, target=640)
        assert chw.shape == (3, 640, 640)

    def test_output_dtype(self):
        img = _make_pil(64, 64)
        chw, scale, _ = resize_for_yolo(img, target=640)
        assert chw.dtype == np.float32

    def test_output_range(self):
        """All pixel values should be in [0, 1]."""
        img = _make_pil(100, 100)
        chw, scale, _ = resize_for_yolo(img, target=640)
        assert float(chw.min()) >= 0.0
        assert float(chw.max()) <= 1.0

    def test_scale_factor_landscape(self):
        """For a 640x480 image scaled to 640, scale = 640/640 = 1.0."""
        img = _make_pil(640, 480)
        _, scale, _ = resize_for_yolo(img, target=640)
        assert scale == pytest.approx(1.0)

    def test_scale_factor_portrait(self):
        """For a 480x640 portrait image scaled to 640, scale = 640/640 = 1.0."""
        img = _make_pil(480, 640)
        _, scale, _ = resize_for_yolo(img, target=640)
        assert scale == pytest.approx(1.0)

    def test_scale_factor_small_image(self):
        """For an 8x8 image scaled to 640, scale = 640/8 = 80."""
        img = _make_pil(8, 8)
        _, scale, _ = resize_for_yolo(img, target=640)
        assert scale == pytest.approx(80.0)

    def test_padding_is_nonnegative(self):
        img = _make_pil(100, 60)
        _, _, (pad_x, pad_y) = resize_for_yolo(img, target=640)
        assert pad_x >= 0
        assert pad_y >= 0

    def test_custom_target(self):
        img = _make_pil(8, 8)
        chw, _, _ = resize_for_yolo(img, target=320)
        assert chw.shape == (3, 320, 320)

    def test_gray_padding_value(self):
        """Padding area should be approximately 114/255 ≈ 0.447."""
        # Use a tall narrow image so there will be left/right padding
        img = _make_pil(8, 640, color=(0, 0, 0))
        chw, _, (pad_x, _) = resize_for_yolo(img, target=640)
        if pad_x > 0:
            left_col_mean = float(chw[:, :, 0].mean())
            expected = 114.0 / 255.0
            assert left_col_mean == pytest.approx(expected, abs=0.05)


# ---------------------------------------------------------------------------
# resize_for_efficientnet
# ---------------------------------------------------------------------------

class TestResizeForEfficientnet:
    def test_output_shape(self):
        img = _make_pil(256, 256)
        arr = resize_for_efficientnet(img, target=224)
        assert arr.shape == (1, 3, 224, 224)

    def test_output_dtype(self):
        img = _make_pil(256, 256)
        arr = resize_for_efficientnet(img)
        assert arr.dtype == np.float32

    def test_nchw_layout(self):
        """Batch dim first, then channels."""
        img = _make_pil(256, 256)
        arr = resize_for_efficientnet(img)
        assert arr.ndim == 4
        assert arr.shape[0] == 1  # batch
        assert arr.shape[1] == 3  # channels

    def test_imagenet_normalization_stats(self):
        """After ImageNet normalization, mean should be near 0 for a 'average' image."""
        # ImageNet mean in [0,1]: [0.485, 0.456, 0.406]
        mean_color = (int(0.485 * 255), int(0.456 * 255), int(0.406 * 255))
        img = Image.new("RGB", (256, 256), mean_color)
        arr = resize_for_efficientnet(img)
        # After normalization the per-channel means should be close to 0
        for c in range(3):
            chan_mean = float(arr[0, c, :, :].mean())
            assert abs(chan_mean) < 0.1, f"Channel {c} mean {chan_mean} too far from 0"

    def test_custom_target(self):
        img = _make_pil(300, 300)
        arr = resize_for_efficientnet(img, target=128)
        assert arr.shape == (1, 3, 128, 128)

    def test_rectangular_input(self):
        img = _make_pil(320, 200)
        arr = resize_for_efficientnet(img)
        assert arr.shape == (1, 3, 224, 224)


# ---------------------------------------------------------------------------
# resize_for_clip
# ---------------------------------------------------------------------------

class TestResizeForClip:
    def test_output_shape(self):
        img = _make_pil(256, 256)
        arr = resize_for_clip(img, target=224)
        assert arr.shape == (1, 3, 224, 224)

    def test_output_dtype(self):
        img = _make_pil(256, 256)
        arr = resize_for_clip(img)
        assert arr.dtype == np.float32

    def test_nchw_layout(self):
        img = _make_pil(256, 256)
        arr = resize_for_clip(img)
        assert arr.ndim == 4
        assert arr.shape[0] == 1
        assert arr.shape[1] == 3

    def test_clip_normalization_produces_finite_values(self):
        img = _make_pil(64, 64)
        arr = resize_for_clip(img)
        assert np.all(np.isfinite(arr))

    def test_different_from_imagenet_normalization(self):
        """CLIP and EfficientNet normalizations differ — arrays should not be identical."""
        img = _make_pil(256, 256, color=(128, 128, 128))
        arr_clip = resize_for_clip(img)
        arr_effnet = resize_for_efficientnet(img)
        assert not np.allclose(arr_clip, arr_effnet), "CLIP and EfficientNet arrays should differ"

    def test_rectangular_input(self):
        img = _make_pil(320, 200)
        arr = resize_for_clip(img)
        assert arr.shape == (1, 3, 224, 224)
