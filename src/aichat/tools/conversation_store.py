"""Client for persistent conversation storage via aichat-database."""
from __future__ import annotations

import httpx


class ConversationStoreTool:
    """Persistent conversation store backed by PostgreSQL via aichat-database.

    All methods fail-open: on any exception, return {} or [] so the caller
    is never blocked by a DB connectivity issue.
    """

    def __init__(self, base_url: str = "http://localhost:8091") -> None:
        self.base_url = base_url.rstrip("/")

    async def _request(self, method: str, path: str, **kwargs: object) -> dict | list:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.request(method, f"{self.base_url}{path}", **kwargs)
                response.raise_for_status()
                return response.json()
        except Exception:
            return {}

    async def create_session(self, session_id: str, title: str = "", model: str = "") -> dict:
        result = await self._request(
            "POST",
            "/conversations/sessions",
            json={"session_id": session_id, "title": title, "model": model},
        )
        return result if isinstance(result, dict) else {}

    async def store_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        turn_index: int = 0,
        embedding: list[float] | None = None,
    ) -> dict:
        payload: dict = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "turn_index": turn_index,
        }
        if embedding is not None:
            payload["embedding"] = embedding
        result = await self._request("POST", "/conversations/turns", json=payload)
        return result if isinstance(result, dict) else {}

    async def search_turns(
        self,
        embedding: list[float],
        limit: int = 5,
        exclude_session: str = "",
    ) -> list[dict]:
        payload: dict = {"embedding": embedding, "limit": limit}
        if exclude_session:
            payload["exclude_session"] = exclude_session
        result = await self._request("POST", "/conversations/search", json=payload)
        if isinstance(result, dict):
            return result.get("results", [])
        return []

    async def list_sessions(self, limit: int = 20) -> list[dict]:
        result = await self._request(
            "GET", "/conversations/sessions", params={"limit": limit}
        )
        if isinstance(result, dict):
            return result.get("sessions", [])
        return []

    async def get_session(self, session_id: str, limit: int = 200) -> dict:
        result = await self._request(
            "GET",
            f"/conversations/sessions/{session_id}",
            params={"limit": limit},
        )
        return result if isinstance(result, dict) else {}

    async def update_title(self, session_id: str, title: str) -> dict:
        result = await self._request(
            "PATCH",
            f"/conversations/sessions/{session_id}/title",
            json={"title": title},
        )
        return result if isinstance(result, dict) else {}

    async def update_compact_state(
        self, session_id: str, compact_summary: str, compact_from_idx: int
    ) -> dict:
        """Persist compaction overlay state for a session (fail-open)."""
        result = await self._request(
            "PATCH",
            f"/conversations/sessions/{session_id}/compact",
            json={"compact_summary": compact_summary, "compact_from_idx": compact_from_idx},
        )
        return result if isinstance(result, dict) else {}

    async def search_turns_text(self, query: str, limit: int = 8) -> list[dict]:
        """Full-text (ILIKE) search on conversation_turns — fallback when embeddings unavailable."""
        result = await self._request(
            "GET",
            "/conversations/turns/search",
            params={"q": query, "limit": limit},
        )
        if isinstance(result, dict):
            return result.get("results", [])
        return []
