"""Client for the vision-mcp JSON-RPC 2.0 service."""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_VISION_URL = "http://localhost:8097"


class VisionMCPTool:
    """Thin async wrapper around the vision-mcp JSON-RPC endpoint."""

    def __init__(self, base_url: str = _DEFAULT_VISION_URL, timeout: float = 60.0) -> None:
        self._mcp_url = base_url.rstrip("/") + "/mcp"
        self._health_url = base_url.rstrip("/") + "/health"
        self._timeout = timeout

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Invoke a vision-mcp tool via JSON-RPC 2.0.

        Returns the parsed tool result dict, or {"error": "..."} on failure.
        """
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(self._mcp_url, json=payload)
                resp.raise_for_status()
                try:
                    data = resp.json()
                except ValueError:
                    content_type = resp.headers.get("content-type", "unknown")
                    logger.warning(
                        "vision-mcp returned non-JSON response (%s): %s",
                        content_type,
                        resp.text[:200],
                    )
                    return {"error": "vision-mcp returned non-JSON response"}
        except httpx.HTTPStatusError as exc:
            logger.warning("vision-mcp HTTP error %s: %s", exc.response.status_code, exc.response.text[:200])
            return {"error": f"vision-mcp HTTP {exc.response.status_code}"}
        except httpx.RequestError as exc:
            logger.warning("vision-mcp connection error: %s", exc)
            return {"error": f"vision-mcp unreachable: {exc}"}

        if "error" in data:
            err = data["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            return {"error": msg}

        # Extract text from content array
        content_list = data.get("result", {}).get("content", [])
        text = " ".join(c.get("text", "") for c in content_list if c.get("type") == "text").strip()
        if not text:
            return {}
        try:
            return json.loads(text)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            return {"result": text}

    async def health(self) -> bool:
        """Return True if the vision-mcp server is reachable and healthy."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(self._health_url)
                resp.raise_for_status()
                data = resp.json()
                return isinstance(data, dict) and int(data.get("tools", 0)) > 0
        except Exception:
            return False

    def _image_source(
        self,
        *,
        image_url: str | None = None,
        image_base64: str | None = None,
        image_file: str | None = None,
    ) -> dict[str, Any]:
        """Build an ImageSource dict from whichever field is provided."""
        if image_url:
            return {"url": image_url}
        if image_base64:
            return {"base64": image_base64}
        if image_file:
            return {"file_path": image_file}
        return {}

    # ------------------------------------------------------------------
    # Face tools
    # ------------------------------------------------------------------

    async def recognize_face(
        self,
        image_url: str | None = None,
        image_base64: str | None = None,
        image_file: str | None = None,
        limit: int = 5,
        min_confidence: float = 0.7,
    ) -> dict[str, Any]:
        """Identify faces in an image against enrolled subjects."""
        return await self.call("recognize_face", {
            "image": self._image_source(image_url=image_url, image_base64=image_base64, image_file=image_file),
            "limit": limit,
            "min_confidence": min_confidence,
        })

    async def verify_face(
        self,
        image_a_url: str | None = None,
        image_b_url: str | None = None,
        image_a_base64: str | None = None,
        image_b_base64: str | None = None,
    ) -> dict[str, Any]:
        """Verify whether two images contain the same person."""
        return await self.call("verify_face", {
            "image_a": self._image_source(image_url=image_a_url, image_base64=image_a_base64),
            "image_b": self._image_source(image_url=image_b_url, image_base64=image_b_base64),
        })

    async def detect_faces(
        self,
        image_url: str | None = None,
        image_base64: str | None = None,
        image_file: str | None = None,
        min_confidence: float = 0.8,
        return_landmarks: bool = False,
    ) -> dict[str, Any]:
        """Detect all faces in an image without recognition."""
        return await self.call("detect_faces", {
            "image": self._image_source(image_url=image_url, image_base64=image_base64, image_file=image_file),
            "min_confidence": min_confidence,
            "return_landmarks": return_landmarks,
        })

    async def enroll_face(
        self,
        subject_name: str,
        image_url: str | None = None,
        image_base64: str | None = None,
        image_file: str | None = None,
    ) -> dict[str, Any]:
        """Enroll a face under a named subject for future recognition."""
        return await self.call("enroll_face", {
            "image": self._image_source(image_url=image_url, image_base64=image_base64, image_file=image_file),
            "subject_name": subject_name,
        })

    async def list_face_subjects(self) -> dict[str, Any]:
        """List all enrolled face subjects."""
        return await self.call("list_face_subjects", {})

    async def delete_face_subject(self, subject_name: str) -> dict[str, Any]:
        """Delete all face enrollments for a subject."""
        return await self.call("delete_face_subject", {"subject_name": subject_name})

    # ------------------------------------------------------------------
    # Vision / Triton tools
    # ------------------------------------------------------------------

    async def detect_objects(
        self,
        image_url: str | None = None,
        image_base64: str | None = None,
        image_file: str | None = None,
        confidence_threshold: float = 0.25,
    ) -> dict[str, Any]:
        """Detect objects in an image using YOLOv8 (COCO 80 classes)."""
        return await self.call("detect_objects", {
            "image": self._image_source(image_url=image_url, image_base64=image_base64, image_file=image_file),
            "confidence_threshold": confidence_threshold,
        })

    async def classify_image(
        self,
        image_url: str | None = None,
        image_base64: str | None = None,
        image_file: str | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Classify an image into ImageNet categories using EfficientNet-B0."""
        return await self.call("classify_image", {
            "image": self._image_source(image_url=image_url, image_base64=image_base64, image_file=image_file),
            "top_k": top_k,
        })

    async def detect_clothing(
        self,
        image_url: str | None = None,
        image_base64: str | None = None,
        image_file: str | None = None,
        confidence_threshold: float = 0.3,
    ) -> dict[str, Any]:
        """Detect clothing items in an image using FashionCLIP zero-shot."""
        return await self.call("detect_clothing", {
            "image": self._image_source(image_url=image_url, image_base64=image_base64, image_file=image_file),
            "confidence_threshold": confidence_threshold,
        })

    async def embed_image(
        self,
        image_url: str | None = None,
        image_base64: str | None = None,
        image_file: str | None = None,
        normalize: bool = True,
    ) -> dict[str, Any]:
        """Embed an image into a 512-dim CLIP vector."""
        return await self.call("embed_image", {
            "image": self._image_source(image_url=image_url, image_base64=image_base64, image_file=image_file),
            "normalize": normalize,
        })

    async def analyze_image(
        self,
        image_url: str | None = None,
        image_base64: str | None = None,
        image_file: str | None = None,
        include_objects: bool = True,
        include_labels: bool = True,
        include_embedding: bool = False,
        confidence_threshold: float = 0.25,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Full scene analysis: objects + labels + optional embedding."""
        return await self.call("analyze_image", {
            "image": self._image_source(image_url=image_url, image_base64=image_base64, image_file=image_file),
            "include_objects": include_objects,
            "include_labels": include_labels,
            "include_embedding": include_embedding,
            "confidence_threshold": confidence_threshold,
            "top_k": top_k,
        })
