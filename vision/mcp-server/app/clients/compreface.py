"""CompreFace REST API client."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from ..config import CompreFaceConfig

_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 0.5  # seconds


async def _post_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    files: dict | None = None,
    json: dict | None = None,
) -> httpx.Response:
    """POST with exponential-backoff retry on transient connection errors.

    Retries up to ``_MAX_RETRIES`` times on ``httpx.RequestError`` (connection
    refused, DNS failure, read timeout, etc.).  HTTP 4xx/5xx errors are NOT
    retried — those are caller-handled business errors.
    """
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = await client.post(
                url,
                params=params,
                headers=headers,
                files=files,
                json=json,
            )
            return resp
        except httpx.RequestError as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BACKOFF_BASE * (2 ** attempt)
                logging.getLogger(__name__).warning(
                    "CompreFace request error (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, _MAX_RETRIES, exc, delay,
                )
                await asyncio.sleep(delay)
    raise last_exc  # type: ignore[misc]
from ..models import (
    BackendInfo,
    BoundingBox,
    DetectedFace,
    DetectFacesResult,
    EnrollFaceResult,
    FaceMatch,
    ListSubjectsResult,
    RecognizedFace,
    RecognizeFaceResult,
    TimingInfo,
    VerifyFaceResult,
)

logger = logging.getLogger(__name__)


def _box_from_cf(cf_box: dict[str, Any]) -> BoundingBox:
    """Convert CompreFace box format to our BoundingBox."""
    return BoundingBox(
        x_min=cf_box.get("x_min", 0.0),
        y_min=cf_box.get("y_min", 0.0),
        x_max=cf_box.get("x_max", 0.0),
        y_max=cf_box.get("y_max", 0.0),
        confidence=cf_box.get("probability", 1.0),
    )


def _backend_info(cfg: CompreFaceConfig) -> BackendInfo:
    from urllib.parse import urlparse
    host = urlparse(cfg.url).netloc
    return BackendInfo(name="compreface", host=host, model="facenet512")


class CompreFaceClient:
    """
    Thin async wrapper around the CompreFace REST API.

    Uses separate API keys for recognition, detection, and verification services
    as required by CompreFace's per-service authorization model.
    """

    def __init__(self, cfg: CompreFaceConfig) -> None:
        self._cfg = cfg
        self._base = cfg.url.rstrip("/")

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------

    async def recognize(
        self,
        image_bytes: bytes,
        mime: str,
        *,
        limit: int = 5,
        det_prob_threshold: float = 0.8,
        min_confidence: float = 0.7,
    ) -> RecognizeFaceResult:
        t0 = time.perf_counter()

        endpoint = f"{self._base}/api/v1/recognition/recognize"
        params = {
            "limit": limit,
            "det_prob_threshold": det_prob_threshold,
            "face_plugins": "age,gender,emotion,mask",
        }
        headers = {"x-api-key": self._cfg.recognition_api_key}

        try:
            async with httpx.AsyncClient(timeout=self._cfg.timeout_s) as client:
                t_backend = time.perf_counter()
                resp = await _post_with_retry(
                    client, endpoint,
                    params=params,
                    headers=headers,
                    files={"file": ("image", image_bytes, mime)},
                )
                backend_ms = (time.perf_counter() - t_backend) * 1000
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:300]
            logger.error(
                "CompreFace recognize HTTP %s: %s", exc.response.status_code, body
            )
            raise RuntimeError(
                f"CompreFace recognition failed: HTTP {exc.response.status_code} — {body}"
            ) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"CompreFace recognition connection error: {exc}") from exc

        faces: list[RecognizedFace] = []
        for result in data.get("result", []):
            raw_subjects = result.get("subjects", [])
            matches = [
                FaceMatch(subject=s["subject"], similarity=s["similarity"])
                for s in raw_subjects
                if s.get("similarity", 0) >= min_confidence
            ]
            faces.append(RecognizedFace(
                box=_box_from_cf(result.get("box", {})),
                matches=matches,
                age=result.get("age"),
                gender=result.get("gender"),
                emotion=result.get("emotion"),
                mask=result.get("mask"),
            ))

        total_ms = (time.perf_counter() - t0) * 1000
        return RecognizeFaceResult(
            faces=faces,
            count=len(faces),
            timing=TimingInfo(total_ms=round(total_ms, 2), backend_ms=round(backend_ms, 2)),
            backend=_backend_info(self._cfg),
        )

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    async def verify(
        self,
        image_a: bytes,
        mime_a: str,
        image_b: bytes,
        mime_b: str,
        *,
        det_prob_threshold: float = 0.8,
    ) -> VerifyFaceResult:
        t0 = time.perf_counter()

        endpoint = f"{self._base}/api/v1/verification/verify"
        params = {"det_prob_threshold": det_prob_threshold}
        headers = {"x-api-key": self._cfg.verification_api_key}

        try:
            async with httpx.AsyncClient(timeout=self._cfg.timeout_s) as client:
                t_backend = time.perf_counter()
                resp = await _post_with_retry(
                    client, endpoint,
                    params=params,
                    headers=headers,
                    files={
                        "source_image": ("source", image_a, mime_a),
                        "target_image": ("target", image_b, mime_b),
                    },
                )
                backend_ms = (time.perf_counter() - t_backend) * 1000
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"CompreFace verification failed: HTTP {exc.response.status_code} — {exc.response.text[:200]}"
            ) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"CompreFace verification connection error: {exc}") from exc

        result = data.get("result", [{}])[0] if data.get("result") else {}
        similarity_list = result.get("similarity", 0.0)

        # CompreFace may return list of similarities; take the max
        if isinstance(similarity_list, list):
            similarity = max(similarity_list) if similarity_list else 0.0
        else:
            similarity = float(similarity_list)

        total_ms = (time.perf_counter() - t0) * 1000
        return VerifyFaceResult(
            verified=similarity >= self._cfg.similarity_threshold,
            similarity=round(similarity, 4),
            subject_a_face_count=len(data.get("result", [])),
            subject_b_face_count=1,
            timing=TimingInfo(total_ms=round(total_ms, 2), backend_ms=round(backend_ms, 2)),
            backend=_backend_info(self._cfg),
        )

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    async def detect(
        self,
        image_bytes: bytes,
        mime: str,
        *,
        det_prob_threshold: float = 0.8,
        return_landmarks: bool = False,
    ) -> DetectFacesResult:
        t0 = time.perf_counter()

        endpoint = f"{self._base}/api/v1/detection/detect"
        plugins = "age,gender,emotion"
        if return_landmarks:
            plugins += ",landmarks"
        params = {"det_prob_threshold": det_prob_threshold, "face_plugins": plugins}
        headers = {"x-api-key": self._cfg.detection_api_key}

        try:
            async with httpx.AsyncClient(timeout=self._cfg.timeout_s) as client:
                t_backend = time.perf_counter()
                resp = await _post_with_retry(
                    client, endpoint,
                    params=params,
                    headers=headers,
                    files={"file": ("image", image_bytes, mime)},
                )
                backend_ms = (time.perf_counter() - t_backend) * 1000
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"CompreFace detection failed: HTTP {exc.response.status_code} — {exc.response.text[:200]}"
            ) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"CompreFace detection connection error: {exc}") from exc

        faces: list[DetectedFace] = []
        for result in data.get("result", []):
            lm = result.get("landmarks")
            faces.append(DetectedFace(
                box=_box_from_cf(result.get("box", {})),
                landmarks=lm if return_landmarks else None,
                age=result.get("age"),
                gender=result.get("gender"),
                emotion=result.get("emotion"),
            ))

        total_ms = (time.perf_counter() - t0) * 1000
        return DetectFacesResult(
            faces=faces,
            count=len(faces),
            timing=TimingInfo(total_ms=round(total_ms, 2), backend_ms=round(backend_ms, 2)),
            backend=_backend_info(self._cfg),
        )

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------

    async def enroll(
        self,
        image_bytes: bytes,
        mime: str,
        subject: str,
    ) -> EnrollFaceResult:
        t0 = time.perf_counter()

        endpoint = f"{self._base}/api/v1/recognition/faces"
        params = {"subject": subject, "det_prob_threshold": self._cfg.det_prob_threshold}
        headers = {"x-api-key": self._cfg.recognition_api_key}

        try:
            async with httpx.AsyncClient(timeout=self._cfg.timeout_s) as client:
                resp = await _post_with_retry(
                    client, endpoint,
                    params=params,
                    headers=headers,
                    files={"file": ("image", image_bytes, mime)},
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"CompreFace enrollment failed: HTTP {exc.response.status_code} — {exc.response.text[:200]}"
            ) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"CompreFace enrollment connection error: {exc}") from exc

        total_ms = (time.perf_counter() - t0) * 1000
        return EnrollFaceResult(
            subject=data.get("subject", subject),
            image_id=data.get("image_id", ""),
            timing=TimingInfo(total_ms=round(total_ms, 2)),
            backend=_backend_info(self._cfg),
        )

    # ------------------------------------------------------------------
    # Subject management
    # ------------------------------------------------------------------

    async def list_subjects(self) -> ListSubjectsResult:
        endpoint = f"{self._base}/api/v1/recognition/subjects"
        headers = {"x-api-key": self._cfg.recognition_api_key}
        try:
            async with httpx.AsyncClient(timeout=self._cfg.timeout_s) as client:
                resp = await client.get(endpoint, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"CompreFace list subjects failed: HTTP {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"CompreFace list subjects connection error: {exc}") from exc

        subjects = data.get("subjects", [])
        return ListSubjectsResult(
            subjects=subjects,
            count=len(subjects),
            backend=_backend_info(self._cfg),
        )

    async def delete_subject(self, subject: str) -> bool:
        endpoint = f"{self._base}/api/v1/recognition/subjects/{subject}"
        headers = {"x-api-key": self._cfg.recognition_api_key}
        try:
            async with httpx.AsyncClient(timeout=self._cfg.timeout_s) as client:
                resp = await client.delete(endpoint, headers=headers)
                if resp.status_code == 404:
                    return False
                resp.raise_for_status()
                return True
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"CompreFace delete subject failed: HTTP {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"CompreFace delete subject connection error: {exc}") from exc

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base}/actuator/health")
                # 200 = healthy, 401/403 = up but auth required (also healthy from
                # a connectivity standpoint), anything else = degraded.
                up = resp.status_code in (200, 401, 403) or (
                    resp.status_code == 400 and b"x-api-key" in resp.content
                )
                return {"status": "ok" if up else "degraded", "code": resp.status_code}
        except Exception as exc:
            return {"status": "unreachable", "error": str(exc)}
