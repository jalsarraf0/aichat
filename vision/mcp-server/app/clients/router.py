"""Vision Router HTTP client for Triton-based inference."""
from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from ..config import VisionRouterConfig
from ..models import (
    BackendInfo,
    BoundingBox,
    ClassifyImageResult,
    ClothingItem,
    DetectClothingResult,
    DetectObjectsResult,
    DetectedObject,
    EmbedImageResult,
    ImageAnalysisResult,
    Label,
    Tag,
    TagImageResult,
    TimingInfo,
)

logger = logging.getLogger(__name__)


def _backend_info(cfg: VisionRouterConfig, model: str = "") -> BackendInfo:
    from urllib.parse import urlparse
    host = urlparse(cfg.url).netloc
    return BackendInfo(name="triton", host=host, model=model)


def _box_from_dict(d: dict[str, Any]) -> BoundingBox:
    return BoundingBox(
        x_min=d.get("x_min", d.get("x1", 0.0)),
        y_min=d.get("y_min", d.get("y1", 0.0)),
        x_max=d.get("x_max", d.get("x2", 0.0)),
        y_max=d.get("y_max", d.get("y2", 0.0)),
        confidence=d.get("confidence", d.get("score", 1.0)),
    )


class VisionRouterClient:
    """
    HTTP client for the vision router service running on the RTX 3090 host.
    The router handles all Triton inference and pre/post-processing.
    """

    def __init__(self, cfg: VisionRouterConfig) -> None:
        self._cfg = cfg
        self._base = cfg.url.rstrip("/")
        self._headers = {"X-API-Key": cfg.api_key} if cfg.api_key else {}

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=self._cfg.timeout_s,
            headers=self._headers,
        )

    async def detect_objects(
        self,
        image_b64: str,
        mime: str,
        *,
        min_confidence: float = 0.4,
        max_results: int = 20,
        classes: list[str] | None = None,
    ) -> DetectObjectsResult:
        t0 = time.perf_counter()
        payload: dict[str, Any] = {
            "image": image_b64,
            "mime": mime,
            "min_confidence": min_confidence,
            "max_results": max_results,
        }
        if classes:
            payload["classes"] = classes

        try:
            async with self._client() as c:
                resp = await c.post(f"{self._base}/v1/detect-objects", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"Vision router detect-objects failed: HTTP {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"Vision router connection error: {exc}") from exc

        objects = [
            DetectedObject(
                label=o["label"],
                confidence=o["confidence"],
                box=_box_from_dict(o["box"]),
            )
            for o in data.get("objects", [])
        ]
        total_ms = (time.perf_counter() - t0) * 1000
        return DetectObjectsResult(
            objects=objects,
            count=len(objects),
            timing=TimingInfo(
                total_ms=round(total_ms, 2),
                backend_ms=data.get("timing", {}).get("inference_ms"),
            ),
            backend=_backend_info(self._cfg, "yolov8n"),
        )

    async def classify_image(
        self,
        image_b64: str,
        mime: str,
        *,
        top_k: int = 5,
        min_confidence: float = 0.01,
    ) -> ClassifyImageResult:
        t0 = time.perf_counter()
        payload = {"image": image_b64, "mime": mime, "top_k": top_k, "min_confidence": min_confidence}

        try:
            async with self._client() as c:
                resp = await c.post(f"{self._base}/v1/classify", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"Vision router classify failed: HTTP {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"Vision router connection error: {exc}") from exc

        labels = [Label(name=l["name"], confidence=l["confidence"]) for l in data.get("labels", [])]
        top = labels[0] if labels else Label(name="unknown", confidence=0.0)
        total_ms = (time.perf_counter() - t0) * 1000
        return ClassifyImageResult(
            labels=labels,
            top_label=top.name,
            top_confidence=top.confidence,
            timing=TimingInfo(total_ms=round(total_ms, 2), backend_ms=data.get("timing", {}).get("inference_ms")),
            backend=_backend_info(self._cfg, "efficientnet_b0"),
        )

    async def detect_clothing(
        self,
        image_b64: str,
        mime: str,
        *,
        min_confidence: float = 0.15,
        top_k: int = 5,
    ) -> DetectClothingResult:
        t0 = time.perf_counter()
        payload = {"image": image_b64, "mime": mime, "min_confidence": min_confidence, "top_k": top_k}

        try:
            async with self._client() as c:
                resp = await c.post(f"{self._base}/v1/detect-clothing", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"Vision router detect-clothing failed: HTTP {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"Vision router connection error: {exc}") from exc

        items = [
            ClothingItem(
                category=item["category"],
                confidence=item["confidence"],
                attributes=item.get("attributes", {}),
                box=_box_from_dict(item["box"]) if item.get("box") else None,
            )
            for item in data.get("items", [])
        ]
        total_ms = (time.perf_counter() - t0) * 1000
        return DetectClothingResult(
            items=items,
            count=len(items),
            dominant_category=data.get("dominant_category"),
            timing=TimingInfo(total_ms=round(total_ms, 2), backend_ms=data.get("timing", {}).get("inference_ms")),
            backend=_backend_info(self._cfg, "fashion_clip"),
        )

    async def embed_image(
        self,
        image_b64: str,
        mime: str,
        *,
        model: str = "clip_vit_b32",
        normalize: bool = True,
    ) -> EmbedImageResult:
        t0 = time.perf_counter()
        payload = {"image": image_b64, "mime": mime, "model": model, "normalize": normalize}

        try:
            async with self._client() as c:
                resp = await c.post(f"{self._base}/v1/embed", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"Vision router embed failed: HTTP {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"Vision router connection error: {exc}") from exc

        embeddings = data.get("embeddings", [])
        total_ms = (time.perf_counter() - t0) * 1000
        return EmbedImageResult(
            embeddings=embeddings,
            dim=len(embeddings),
            model=data.get("model", model),
            normalized=data.get("normalized", normalize),
            timing=TimingInfo(total_ms=round(total_ms, 2), backend_ms=data.get("timing", {}).get("inference_ms")),
            backend=_backend_info(self._cfg, model),
        )

    async def analyze(
        self,
        image_b64: str,
        mime: str,
        *,
        include_objects: bool = True,
        include_classification: bool = True,
        include_clothing: bool = False,
        include_embeddings: bool = False,
        object_confidence: float = 0.4,
        classification_top_k: int = 3,
    ) -> ImageAnalysisResult:
        t0 = time.perf_counter()
        payload = {
            "image": image_b64,
            "mime": mime,
            "include_objects": include_objects,
            "include_classification": include_classification,
            "include_clothing": include_clothing,
            "include_embeddings": include_embeddings,
            "object_confidence": object_confidence,
            "classification_top_k": classification_top_k,
        }

        try:
            async with self._client() as c:
                resp = await c.post(f"{self._base}/v1/analyze", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"Vision router analyze failed: HTTP {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"Vision router connection error: {exc}") from exc

        objects = None
        if data.get("objects") is not None:
            objects = [
                DetectedObject(label=o["label"], confidence=o["confidence"], box=_box_from_dict(o["box"]))
                for o in data["objects"]
            ]

        classification = None
        if data.get("classification") is not None:
            classification = [Label(name=l["name"], confidence=l["confidence"]) for l in data["classification"]]

        clothing = None
        if data.get("clothing") is not None:
            clothing = [
                ClothingItem(
                    category=item["category"],
                    confidence=item["confidence"],
                    attributes=item.get("attributes", {}),
                    box=_box_from_dict(item["box"]) if item.get("box") else None,
                )
                for item in data["clothing"]
            ]

        total_ms = (time.perf_counter() - t0) * 1000
        return ImageAnalysisResult(
            objects=objects,
            classification=classification,
            clothing=clothing,
            embeddings=data.get("embeddings"),
            summary=data.get("summary", "Analysis complete"),
            timing=TimingInfo(total_ms=round(total_ms, 2), backend_ms=data.get("timing", {}).get("total_ms")),
            backend=_backend_info(self._cfg, "multi"),
        )

    async def tag_image(
        self,
        image_b64: str,
        mime: str,
        *,
        general_threshold: float = 0.35,
        character_threshold: float = 0.85,
    ) -> TagImageResult:
        t0 = time.perf_counter()
        payload = {
            "image": image_b64,
            "mime": mime,
            "general_threshold": general_threshold,
            "character_threshold": character_threshold,
        }
        try:
            async with self._client() as c:
                resp = await c.post(f"{self._base}/v1/tag", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"Vision router tag failed: HTTP {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"Vision router connection error: {exc}") from exc

        total_ms = (time.perf_counter() - t0) * 1000
        return TagImageResult(
            characters=[Tag(tag=t["tag"], confidence=t["confidence"]) for t in data.get("characters", [])],
            general=[Tag(tag=t["tag"], confidence=t["confidence"]) for t in data.get("general", [])],
            rating=[Tag(tag=t["tag"], confidence=t["confidence"]) for t in data.get("rating", [])],
            timing=TimingInfo(total_ms=round(total_ms, 2), backend_ms=data.get("inference_ms")),
            backend=_backend_info(self._cfg, "wd_tagger"),
        )

    async def health(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                resp = await c.get(f"{self._base}/v1/health")
                if resp.status_code == 200:
                    return resp.json()
                return {"status": "degraded", "code": resp.status_code}
        except Exception as exc:
            return {"status": "unreachable", "error": str(exc)}
