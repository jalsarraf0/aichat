"""Pydantic request and response models for the vision router API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class InferRequest(BaseModel):
    """Base request model carrying a base64-encoded image."""

    image_b64: str = Field(..., description="Base64-encoded image bytes (JPEG/PNG). Data-URI prefix is accepted.")


class BoundingBox(BaseModel):
    """Axis-aligned bounding box with associated detection confidence."""

    x_min: float = Field(..., description="Left edge of the box in pixel coordinates.")
    y_min: float = Field(..., description="Top edge of the box in pixel coordinates.")
    x_max: float = Field(..., description="Right edge of the box in pixel coordinates.")
    y_max: float = Field(..., description="Bottom edge of the box in pixel coordinates.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence for this box.")


class DetectedObject(BaseModel):
    """A single detected object returned by the object detection endpoint."""

    label: str = Field(..., description="Human-readable class label.")
    class_id: int = Field(..., ge=0, description="Numeric class index.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence.")
    box: BoundingBox = Field(..., description="Bounding box in original image pixel coordinates.")


class DetectObjectsRequest(InferRequest):
    """Request body for POST /v1/detect-objects."""

    confidence_threshold: float = Field(0.25, ge=0.0, le=1.0, description="Minimum confidence to keep a detection.")
    nms_threshold: float = Field(0.45, ge=0.0, le=1.0, description="IoU threshold for non-maximum suppression.")
    max_detections: int = Field(100, ge=1, le=1000, description="Maximum number of detections to return.")


class DetectObjectsResponse(BaseModel):
    """Response from POST /v1/detect-objects."""

    objects: list[DetectedObject] = Field(..., description="List of detected objects.")
    count: int = Field(..., ge=0, description="Number of detections returned.")
    model: str = Field(..., description="Triton model name used for inference.")
    inference_ms: float = Field(..., ge=0.0, description="End-to-end inference latency in milliseconds.")


class TopLabel(BaseModel):
    """A single classification result with label and confidence."""

    label: str = Field(..., description="Human-readable class label.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Softmax probability for this label.")


class ClassifyRequest(InferRequest):
    """Request body for POST /v1/classify."""

    top_k: int = Field(5, ge=1, le=1000, description="Number of top-scoring labels to return.")


class ClassifyResponse(BaseModel):
    """Response from POST /v1/classify."""

    labels: list[TopLabel] = Field(..., description="Top-k predicted class labels, sorted by confidence descending.")
    model: str = Field(..., description="Triton model name used for inference.")
    inference_ms: float = Field(..., ge=0.0, description="End-to-end inference latency in milliseconds.")


class ClothingItem(BaseModel):
    """A single clothing/fashion item detected by FashionCLIP."""

    category: str = Field(..., description="Clothing category label.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Similarity score for this category.")
    box: BoundingBox | None = Field(None, description="Bounding box if localization was performed (None for zero-shot).")


class DetectClothingRequest(InferRequest):
    """Request body for POST /v1/detect-clothing."""

    confidence_threshold: float = Field(0.3, ge=0.0, le=1.0, description="Minimum similarity score to include a category.")


class DetectClothingResponse(BaseModel):
    """Response from POST /v1/detect-clothing."""

    items: list[ClothingItem] = Field(..., description="Detected clothing items sorted by confidence descending.")
    count: int = Field(..., ge=0, description="Number of clothing items returned.")
    model: str = Field(..., description="Triton model name used for inference.")
    inference_ms: float = Field(..., ge=0.0, description="End-to-end inference latency in milliseconds.")


class EmbedRequest(InferRequest):
    """Request body for POST /v1/embed."""

    normalize: bool = Field(True, description="If True, normalize the embedding to unit length.")


class EmbedResponse(BaseModel):
    """Response from POST /v1/embed."""

    embedding: list[float] = Field(..., description="Image feature embedding vector.")
    dimension: int = Field(..., ge=1, description="Dimensionality of the embedding vector.")
    model: str = Field(..., description="Triton model name used for inference.")
    inference_ms: float = Field(..., ge=0.0, description="End-to-end inference latency in milliseconds.")


class Tag(BaseModel):
    """A single tag result from the WD tagger."""

    tag: str = Field(..., description="Danbooru-style tag name.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Sigmoid probability from the model.")


class TagRequest(InferRequest):
    """Request body for POST /v1/tag."""

    general_threshold: float = Field(0.35, ge=0.0, le=1.0, description="Minimum confidence for general tags.")
    character_threshold: float = Field(0.85, ge=0.0, le=1.0, description="Minimum confidence for character tags.")


class TagResponse(BaseModel):
    """Response from POST /v1/tag."""

    characters: list[Tag] = Field(..., description="Matched character tags, sorted by confidence.")
    general: list[Tag] = Field(..., description="General content/style tags, sorted by confidence.")
    rating: list[Tag] = Field(..., description="Rating tags (safe/questionable/explicit), sorted by confidence.")
    model: str = Field(..., description="Triton model name used for inference.")
    inference_ms: float = Field(..., ge=0.0, description="End-to-end inference latency in milliseconds.")


class AnalyzeRequest(InferRequest):
    """Request body for POST /v1/analyze. Runs multiple vision tasks in parallel."""

    include_objects: bool = Field(True, description="Run YOLO object detection.")
    include_labels: bool = Field(True, description="Run EfficientNet image classification.")
    include_embedding: bool = Field(False, description="Run CLIP embedding extraction.")
    confidence_threshold: float = Field(0.25, ge=0.0, le=1.0, description="Confidence threshold for object detection.")
    top_k: int = Field(5, ge=1, le=1000, description="Top-k labels to return from classification.")


class AnalyzeResponse(BaseModel):
    """Response from POST /v1/analyze."""

    objects: list[DetectedObject] | None = Field(None, description="Detected objects (None if not requested).")
    labels: list[TopLabel] | None = Field(None, description="Classification labels (None if not requested).")
    embedding: list[float] | None = Field(None, description="Image embedding (None if not requested).")
    model_versions: dict[str, str] = Field(default_factory=dict, description="Map of model name to version string used.")
    inference_ms: float = Field(..., ge=0.0, description="Total wall-clock latency for all inference calls in milliseconds.")
