"""Pydantic request/response models for the vision MCP tools."""
from __future__ import annotations

import base64
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Shared input types
# ---------------------------------------------------------------------------

class ImageSource(BaseModel):
    """Input image — exactly one source must be provided."""
    url: str | None = Field(None, description="Public URL to fetch the image from")
    base64: str | None = Field(None, description="Base64-encoded image bytes")
    file_path: str | None = Field(None, description="Server-local absolute file path")

    @model_validator(mode="after")
    def exactly_one_source(self) -> "ImageSource":
        sources = [x for x in (self.url, self.base64, self.file_path) if x]
        if len(sources) != 1:
            raise ValueError("Exactly one of url, base64, or file_path must be provided")
        return self

    @field_validator("base64")
    @classmethod
    def validate_base64(cls, v: str | None) -> str | None:
        if v is None:
            return v
        try:
            base64.b64decode(v, validate=True)
        except Exception:
            raise ValueError("base64 must be valid base64-encoded data")
        return v


# ---------------------------------------------------------------------------
# Shared result primitives
# ---------------------------------------------------------------------------

class BoundingBox(BaseModel):
    x_min: float = Field(..., ge=0)
    y_min: float = Field(..., ge=0)
    x_max: float = Field(..., ge=0)
    y_max: float = Field(..., ge=0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class Label(BaseModel):
    name: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class TimingInfo(BaseModel):
    total_ms: float
    backend_ms: float | None = None
    preprocess_ms: float | None = None
    postprocess_ms: float | None = None


class BackendInfo(BaseModel):
    name: str
    host: str | None = None
    model: str | None = None


# ---------------------------------------------------------------------------
# Face recognition
# ---------------------------------------------------------------------------

class RecognizeFaceRequest(BaseModel):
    image: ImageSource
    subject_filter: list[str] | None = None
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    limit: int = Field(default=5, ge=1, le=50)


class FaceMatch(BaseModel):
    subject: str
    similarity: float = Field(..., ge=0.0, le=1.0)


class RecognizedFace(BaseModel):
    box: BoundingBox
    matches: list[FaceMatch]
    age: dict[str, Any] | None = None
    gender: dict[str, Any] | None = None
    emotion: list[dict[str, Any]] | None = None
    mask: dict[str, Any] | None = None


class RecognizeFaceResult(BaseModel):
    faces: list[RecognizedFace]
    count: int
    timing: TimingInfo
    backend: BackendInfo


# ---------------------------------------------------------------------------
# Face verification
# ---------------------------------------------------------------------------

class VerifyFaceRequest(BaseModel):
    image_a: ImageSource
    image_b: ImageSource
    min_similarity: float = Field(default=0.85, ge=0.0, le=1.0)


class VerifyFaceResult(BaseModel):
    verified: bool
    similarity: float = Field(..., ge=0.0, le=1.0)
    subject_a_face_count: int
    subject_b_face_count: int
    timing: TimingInfo
    backend: BackendInfo


# ---------------------------------------------------------------------------
# Face detection
# ---------------------------------------------------------------------------

class DetectFacesRequest(BaseModel):
    image: ImageSource
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    return_landmarks: bool = False


class DetectedFace(BaseModel):
    box: BoundingBox
    landmarks: dict[str, list[float]] | None = None
    age: dict[str, Any] | None = None
    gender: dict[str, Any] | None = None
    emotion: list[dict[str, Any]] | None = None


class DetectFacesResult(BaseModel):
    faces: list[DetectedFace]
    count: int
    timing: TimingInfo
    backend: BackendInfo


# ---------------------------------------------------------------------------
# Face enrollment
# ---------------------------------------------------------------------------

class EnrollFaceRequest(BaseModel):
    image: ImageSource
    subject_name: str = Field(..., min_length=1, max_length=200)


class EnrollFaceResult(BaseModel):
    subject: str
    image_id: str
    timing: TimingInfo
    backend: BackendInfo


class ListSubjectsResult(BaseModel):
    subjects: list[str]
    count: int
    backend: BackendInfo


class DeleteSubjectResult(BaseModel):
    subject: str
    deleted: bool
    backend: BackendInfo


# ---------------------------------------------------------------------------
# Object detection
# ---------------------------------------------------------------------------

class DetectObjectsRequest(BaseModel):
    image: ImageSource
    min_confidence: float = Field(default=0.4, ge=0.0, le=1.0)
    max_results: int = Field(default=20, ge=1, le=100)
    classes: list[str] | None = None


class DetectedObject(BaseModel):
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    box: BoundingBox


class DetectObjectsResult(BaseModel):
    objects: list[DetectedObject]
    count: int
    timing: TimingInfo
    backend: BackendInfo


# ---------------------------------------------------------------------------
# Image classification
# ---------------------------------------------------------------------------

class ClassifyImageRequest(BaseModel):
    image: ImageSource
    top_k: int = Field(default=5, ge=1, le=50)
    min_confidence: float = Field(default=0.01, ge=0.0, le=1.0)


class ClassifyImageResult(BaseModel):
    labels: list[Label]
    top_label: str
    top_confidence: float
    timing: TimingInfo
    backend: BackendInfo


# ---------------------------------------------------------------------------
# Clothing detection
# ---------------------------------------------------------------------------

class DetectClothingRequest(BaseModel):
    image: ImageSource
    min_confidence: float = Field(default=0.15, ge=0.0, le=1.0)
    top_k: int = Field(default=5, ge=1, le=20)


class ClothingItem(BaseModel):
    category: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    attributes: dict[str, str] = Field(default_factory=dict)
    box: BoundingBox | None = None


class DetectClothingResult(BaseModel):
    items: list[ClothingItem]
    count: int
    dominant_category: str | None = None
    timing: TimingInfo
    backend: BackendInfo


# ---------------------------------------------------------------------------
# Image analysis
# ---------------------------------------------------------------------------

class AnalyzeImageRequest(BaseModel):
    image: ImageSource
    include_objects: bool = True
    include_classification: bool = True
    include_clothing: bool = False
    include_embeddings: bool = False
    object_confidence: float = Field(default=0.4, ge=0.0, le=1.0)
    classification_top_k: int = Field(default=3, ge=1, le=20)


class ImageAnalysisResult(BaseModel):
    objects: list[DetectedObject] | None = None
    classification: list[Label] | None = None
    clothing: list[ClothingItem] | None = None
    embeddings: list[float] | None = None
    summary: str
    timing: TimingInfo
    backend: BackendInfo


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class EmbedImageRequest(BaseModel):
    image: ImageSource
    model: str = Field(default="clip_vit_b32", description="Embedding model to use")
    normalize: bool = True


class EmbedImageResult(BaseModel):
    embeddings: list[float]
    dim: int
    model: str
    normalized: bool
    timing: TimingInfo
    backend: BackendInfo


# ---------------------------------------------------------------------------
# Anime/illustration tagging
# ---------------------------------------------------------------------------

class TagImageRequest(BaseModel):
    image: ImageSource
    general_threshold: float = Field(default=0.35, ge=0.0, le=1.0, description="Minimum confidence for general tags")
    character_threshold: float = Field(default=0.85, ge=0.0, le=1.0, description="Minimum confidence for character tags")


class Tag(BaseModel):
    tag: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class TagImageResult(BaseModel):
    characters: list[Tag]
    general: list[Tag]
    rating: list[Tag]
    timing: TimingInfo
    backend: BackendInfo


# ---------------------------------------------------------------------------
# Error model
# ---------------------------------------------------------------------------

class VisionError(BaseModel):
    error: str
    code: str
    details: dict[str, Any] | None = None
    backend: str | None = None
    timing: TimingInfo | None = None
