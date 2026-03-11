"""Vision Router configuration using pydantic-settings."""

from functools import lru_cache

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TritonConfig(BaseModel):
    """Triton Inference Server connection settings."""

    url: str = "http://localhost:8000"
    """HTTP endpoint for Triton REST API."""

    grpc_url: str = "localhost:8001"
    """gRPC endpoint for Triton (reserved for future use)."""

    timeout_s: float = 30.0
    """Request timeout in seconds."""


class RouterConfig(BaseSettings):
    """Top-level configuration for the vision router service."""

    model_config = SettingsConfigDict(
        env_prefix="ROUTER_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    triton: TritonConfig = Field(default_factory=TritonConfig)
    """Triton server connection settings."""

    yolo_model: str = "yolov8n"
    """Triton model name for YOLO object detection."""

    efficientnet_model: str = "efficientnet_b0"
    """Triton model name for EfficientNet classification."""

    clip_model: str = "clip_vit_b32"
    """Triton model name for CLIP image embedding."""

    fashion_clip_model: str = "fashion_clip"
    """Triton model name for FashionCLIP clothing detection."""

    wd_tagger_model: str = "wd_tagger"
    """Triton model name for WD ViT Tagger anime/illustration tagging."""

    host: str = "0.0.0.0"
    """Host address to bind the HTTP server."""

    port: int = 8090
    """Port to bind the HTTP server."""

    workers: int = 2
    """Number of uvicorn worker processes."""

    max_file_size_mb: float = 20.0
    """Maximum allowed image upload size in megabytes."""

    api_key: str = ""
    """Optional API key for authentication. Empty string disables auth."""

    log_level: str = "INFO"
    """Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)."""


@lru_cache(maxsize=1)
def get_config() -> RouterConfig:
    """Return a cached singleton RouterConfig instance."""
    return RouterConfig()
