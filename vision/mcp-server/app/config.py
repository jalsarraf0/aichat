"""Centralised configuration for the vision MCP server."""
from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class CompreFaceConfig(BaseModel):
    url: str = "http://192.168.50.2:8080"
    api_key: str = ""
    recognition_api_key: str = ""
    detection_api_key: str = ""
    verification_api_key: str = ""
    timeout_s: float = 30.0
    max_file_size_mb: float = 10.0
    det_prob_threshold: float = 0.8
    similarity_threshold: float = 0.85
    limit: int = 5


class VisionRouterConfig(BaseModel):
    url: str = "http://192.168.50.2:8090"
    api_key: str = ""
    timeout_s: float = 60.0
    max_file_size_mb: float = 20.0


class FallbackConfig(BaseModel):
    """Local CPU-mode CompreFace used when the primary RTX host is unreachable.

    Triton-backed tools (object detection, classification, clothing, embeddings)
    have no local fallback — they return a clear UNAVAILABLE error instead.
    """
    enabled: bool = True
    compreface_url: str = "http://aichat-compreface-local:8080"
    # Leave empty to reuse the primary API key; set explicitly if the local
    # CompreFace instance was configured with a different key.
    compreface_api_key: str = ""
    health_check_interval_s: float = 30.0


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8097
    workers: int = 2
    max_upload_mb: float = 20.0
    allow_url_input: bool = True
    url_allowlist: list[str] = []
    cors_origins: list[str] = ["*"]
    request_id_header: str = "X-Request-ID"
    log_level: str = "INFO"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VISION_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    compreface: CompreFaceConfig = Field(default_factory=CompreFaceConfig)
    router: VisionRouterConfig = Field(default_factory=VisionRouterConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    fallback: FallbackConfig = Field(default_factory=FallbackConfig)

    # Feature flags
    enable_compreface: bool = True
    enable_triton: bool = True
    enable_url_input: bool = True
    enable_embeddings: bool = True

    def fallback_compreface_config(self) -> CompreFaceConfig:
        """Return a CompreFaceConfig pointed at the local fallback instance."""
        key = self.fallback.compreface_api_key or self.compreface.api_key
        return CompreFaceConfig(
            url=self.fallback.compreface_url,
            api_key=key,
            recognition_api_key=key,
            detection_api_key=key,
            verification_api_key=key,
            timeout_s=self.compreface.timeout_s,
            max_file_size_mb=self.compreface.max_file_size_mb,
            det_prob_threshold=self.compreface.det_prob_threshold,
            similarity_threshold=self.compreface.similarity_threshold,
            limit=self.compreface.limit,
        )

    @model_validator(mode="after")
    def _set_compreface_keys(self) -> Settings:
        """If individual API keys are not set, derive from master key."""
        if self.compreface.api_key:
            for field in ("recognition_api_key", "detection_api_key", "verification_api_key"):
                if not getattr(self.compreface, field):
                    object.__setattr__(self.compreface, field, self.compreface.api_key)
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
