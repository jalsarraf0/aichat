"""Unit tests for vision/mcp-server/app/models.py."""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

_repo_root = Path(__file__).parent.parent.parent.parent  # .../aichat
_mcp_root_str = str(_repo_root / "vision" / "mcp-server")
_router_root_str = str(_repo_root / "vision" / "services" / "vision-router")


def _load_mcp_models():
    """Load app.models from mcp-server, handling sys.path ordering."""
    for key in list(sys.modules.keys()):
        if key == "app" or key.startswith("app."):
            del sys.modules[key]

    path_backup = sys.path[:]
    sys.path = [p for p in sys.path if p != _router_root_str]
    if _mcp_root_str not in sys.path:
        sys.path.insert(0, _mcp_root_str)
    try:
        import app.models as _m  # type: ignore[import]
        return _m
    finally:
        sys.path[:] = path_backup


# ---------------------------------------------------------------------------
# ImageSource
# ---------------------------------------------------------------------------

class TestImageSource:
    def test_exactly_one_source_base64(self):
        m = _load_mcp_models()
        src = m.ImageSource(base64=base64.b64encode(b"fake").decode())
        assert src.base64 is not None
        assert src.url is None
        assert src.file_path is None

    def test_exactly_one_source_url(self):
        m = _load_mcp_models()
        src = m.ImageSource(url="https://example.com/img.jpg")
        assert src.url == "https://example.com/img.jpg"

    def test_exactly_one_source_file_path(self):
        m = _load_mcp_models()
        src = m.ImageSource(file_path="/workspace/image.jpg")
        assert src.file_path == "/workspace/image.jpg"

    def test_no_source_raises(self):
        m = _load_mcp_models()
        with pytest.raises(ValidationError, match="Exactly one"):
            m.ImageSource()

    def test_multiple_sources_raises(self):
        m = _load_mcp_models()
        with pytest.raises(ValidationError, match="Exactly one"):
            m.ImageSource(
                url="https://example.com/img.jpg",
                base64=base64.b64encode(b"fake").decode(),
            )

    def test_all_three_sources_raises(self):
        m = _load_mcp_models()
        with pytest.raises(ValidationError, match="Exactly one"):
            m.ImageSource(
                url="https://example.com/img.jpg",
                base64=base64.b64encode(b"fake").decode(),
                file_path="/workspace/image.jpg",
            )

    def test_invalid_base64_raises(self):
        m = _load_mcp_models()
        with pytest.raises(ValidationError):
            m.ImageSource(base64="not-valid-base64!!!!")

    def test_valid_base64_accepted(self):
        m = _load_mcp_models()
        valid = base64.b64encode(b"\xff\xd8\xff" + b"\x00" * 10).decode()
        src = m.ImageSource(base64=valid)
        assert src.base64 == valid

    def test_none_base64_accepted(self):
        m = _load_mcp_models()
        src = m.ImageSource(url="https://example.com/img.jpg")
        assert src.base64 is None


# ---------------------------------------------------------------------------
# BoundingBox
# ---------------------------------------------------------------------------

class TestBoundingBox:
    def test_valid_bounding_box(self):
        m = _load_mcp_models()
        bb = m.BoundingBox(x_min=10, y_min=20, x_max=100, y_max=200, confidence=0.95)
        assert bb.x_min == 10
        assert bb.confidence == pytest.approx(0.95)

    def test_zero_coordinates_valid(self):
        m = _load_mcp_models()
        bb = m.BoundingBox(x_min=0, y_min=0, x_max=0, y_max=0)
        assert bb.x_min == 0

    def test_negative_coordinate_raises(self):
        m = _load_mcp_models()
        with pytest.raises(ValidationError):
            m.BoundingBox(x_min=-1, y_min=0, x_max=100, y_max=100)

    def test_confidence_default(self):
        m = _load_mcp_models()
        bb = m.BoundingBox(x_min=0, y_min=0, x_max=50, y_max=50)
        assert bb.confidence == 1.0

    def test_confidence_above_one_raises(self):
        m = _load_mcp_models()
        with pytest.raises(ValidationError):
            m.BoundingBox(x_min=0, y_min=0, x_max=50, y_max=50, confidence=1.1)

    def test_confidence_below_zero_raises(self):
        m = _load_mcp_models()
        with pytest.raises(ValidationError):
            m.BoundingBox(x_min=0, y_min=0, x_max=50, y_max=50, confidence=-0.1)


# ---------------------------------------------------------------------------
# RecognizeFaceResult serialization
# ---------------------------------------------------------------------------

class TestRecognizeFaceResult:
    def _make(self):
        m = _load_mcp_models()
        box = m.BoundingBox(x_min=10, y_min=20, x_max=80, y_max=90, confidence=0.99)
        match = m.FaceMatch(subject="alice", similarity=0.93)
        face = m.RecognizedFace(box=box, matches=[match])
        timing = m.TimingInfo(total_ms=42.5, backend_ms=35.0)
        backend = m.BackendInfo(name="compreface", host="192.168.50.2")
        return m.RecognizeFaceResult(faces=[face], count=1, timing=timing, backend=backend)

    def test_serializes_to_dict(self):
        result = self._make()
        d = result.model_dump()
        assert d["count"] == 1
        assert len(d["faces"]) == 1
        assert d["faces"][0]["matches"][0]["subject"] == "alice"

    def test_serializes_to_json(self):
        result = self._make()
        j = result.model_dump_json()
        parsed = json.loads(j)
        assert parsed["count"] == 1
        assert parsed["timing"]["total_ms"] == pytest.approx(42.5)

    def test_backend_info_preserved(self):
        result = self._make()
        d = result.model_dump()
        assert d["backend"]["name"] == "compreface"
        assert d["backend"]["host"] == "192.168.50.2"

    def test_empty_faces_list(self):
        m = _load_mcp_models()
        result = m.RecognizeFaceResult(
            faces=[],
            count=0,
            timing=m.TimingInfo(total_ms=5.0),
            backend=m.BackendInfo(name="compreface"),
        )
        assert result.count == 0
        assert result.faces == []


# ---------------------------------------------------------------------------
# DetectObjectsRequest defaults
# ---------------------------------------------------------------------------

class TestDetectObjectsRequest:
    def test_default_min_confidence(self):
        m = _load_mcp_models()
        req = m.DetectObjectsRequest(image={"url": "https://example.com/img.jpg"})
        assert req.min_confidence == pytest.approx(0.4)

    def test_default_max_results(self):
        m = _load_mcp_models()
        req = m.DetectObjectsRequest(image={"url": "https://example.com/img.jpg"})
        assert req.max_results == 20

    def test_default_classes_none(self):
        m = _load_mcp_models()
        req = m.DetectObjectsRequest(image={"url": "https://example.com/img.jpg"})
        assert req.classes is None

    def test_custom_confidence(self):
        m = _load_mcp_models()
        req = m.DetectObjectsRequest(
            image={"url": "https://example.com/img.jpg"},
            min_confidence=0.7,
        )
        assert req.min_confidence == pytest.approx(0.7)

    def test_max_results_validation(self):
        m = _load_mcp_models()
        with pytest.raises(ValidationError):
            m.DetectObjectsRequest(
                image={"url": "https://example.com/img.jpg"},
                max_results=0,
            )

    def test_classes_filter(self):
        m = _load_mcp_models()
        req = m.DetectObjectsRequest(
            image={"url": "https://example.com/img.jpg"},
            classes=["person", "car"],
        )
        assert req.classes == ["person", "car"]


# ---------------------------------------------------------------------------
# Other model smoke tests
# ---------------------------------------------------------------------------

class TestOtherModels:
    def test_embed_image_request_default_model(self):
        m = _load_mcp_models()
        req = m.EmbedImageRequest(image={"url": "https://example.com/img.jpg"})
        assert req.model == "clip_vit_b32"
        assert req.normalize is True

    def test_classify_image_request_defaults(self):
        m = _load_mcp_models()
        req = m.ClassifyImageRequest(image={"url": "https://example.com/img.jpg"})
        assert req.top_k == 5
        assert req.min_confidence == pytest.approx(0.01)

    def test_analyze_image_request_defaults(self):
        m = _load_mcp_models()
        req = m.AnalyzeImageRequest(image={"url": "https://example.com/img.jpg"})
        assert req.include_objects is True
        assert req.include_classification is True
        assert req.include_clothing is False
        assert req.include_embeddings is False

    def test_enroll_face_request_subject_name_required(self):
        m = _load_mcp_models()
        with pytest.raises(ValidationError):
            m.EnrollFaceRequest(image={"url": "https://example.com/img.jpg"}, subject_name="")

    def test_label_confidence_bounds(self):
        m = _load_mcp_models()
        with pytest.raises(ValidationError):
            m.Label(name="cat", confidence=1.5)
        with pytest.raises(ValidationError):
            m.Label(name="cat", confidence=-0.1)
        label = m.Label(name="dog", confidence=0.9)
        assert label.name == "dog"
