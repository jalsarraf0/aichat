"""Client for the aichat-database PostgreSQL storage/cache service."""
from __future__ import annotations

from typing import Optional

import httpx

from .errors import ToolRequestError, is_retryable_status


class DatabaseToolError(ToolRequestError):
    pass


class DatabaseTool:
    def __init__(self, base_url: str = "http://localhost:8091") -> None:
        self.base_url = base_url.rstrip("/")

    async def _request(self, method: str, path: str, **kwargs: object) -> dict:
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.request(method, f"{self.base_url}{path}", **kwargs)
                response.raise_for_status()
                return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            status = None
            retryable = False
            if isinstance(exc, httpx.HTTPStatusError):
                status = exc.response.status_code
                retryable = is_retryable_status(status)
            elif isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
                retryable = True
            raise DatabaseToolError(
                f"Database request failed for {path}: {exc}",
                status_code=status,
                retryable=retryable,
            ) from exc

    async def store_article(
        self,
        url: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> dict:
        return await self._request(
            "POST",
            "/articles/store",
            json={"url": url, "title": title, "content": content, "topic": topic},
        )

    async def search_articles(
        self,
        topic: Optional[str] = None,
        q: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
        summary_only: bool = False,
    ) -> dict:
        params: dict[str, object] = {"limit": limit, "offset": offset}
        if topic:
            params["topic"] = topic
        if q:
            params["q"] = q
        if summary_only:
            params["summary_only"] = "true"
        return await self._request("GET", "/articles/search", params=params)

    async def store_image(
        self,
        url: str,
        host_path: Optional[str] = None,
        alt_text: Optional[str] = None,
    ) -> dict:
        return await self._request(
            "POST",
            "/images/store",
            json={"url": url, "host_path": host_path, "alt_text": alt_text},
        )

    async def store_image_rich(
        self,
        url: str,
        *,
        host_path: Optional[str] = None,
        alt_text: Optional[str] = None,
        subject: Optional[str] = None,
        description: Optional[str] = None,
        phash: Optional[str] = None,
        quality_score: Optional[float] = None,
    ) -> dict:
        """Store an image with vision-recognition metadata (subject, phash, quality_score)."""
        return await self._request(
            "POST",
            "/images/store",
            json={
                "url": url, "host_path": host_path, "alt_text": alt_text,
                "subject": subject, "description": description,
                "phash": phash, "quality_score": quality_score,
            },
        )

    async def search_images(self, subject: str, limit: int = 20) -> dict:
        """Search previously confirmed images by subject or description text."""
        return await self._request(
            "GET", "/images/search", params={"subject": subject, "limit": limit}
        )

    async def cache_store(self, url: str, content: str, title: Optional[str] = None) -> dict:
        return await self._request(
            "POST",
            "/cache/store",
            json={"url": url, "content": content, "title": title},
        )

    async def cache_get(self, url: str) -> dict:
        return await self._request("GET", "/cache/get", params={"url": url})

    async def cache_check(self, url: str) -> dict:
        return await self._request("GET", "/cache/check", params={"url": url})

    async def list_images(self, limit: int = 20) -> dict:
        return await self._request("GET", "/images/list", params={"limit": limit})

    async def get_errors(
        self,
        limit: int = 50,
        service: Optional[str] = None,
    ) -> dict:
        params: dict[str, object] = {"limit": limit}
        if service:
            params["service"] = service
        return await self._request("GET", "/errors/recent", params=params)

    async def store_embedding(
        self,
        key: str,
        content: str,
        embedding: list,
        model: str = "",
        topic: str = "",
    ) -> dict:
        """Store a text embedding (float array) keyed by *key* (URL or hash)."""
        return await self._request(
            "POST",
            "/embeddings/store",
            json={"key": key, "content": content, "embedding": embedding,
                  "model": model, "topic": topic},
        )

    async def search_by_embedding(
        self,
        embedding: list,
        limit: int = 5,
        topic: str = "",
    ) -> dict:
        """Return the top-N most semantically similar stored documents."""
        body: dict = {"embedding": embedding, "limit": limit}
        if topic:
            body["topic"] = topic
        return await self._request("POST", "/embeddings/search", json=body)
