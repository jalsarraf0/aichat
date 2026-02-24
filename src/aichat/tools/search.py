"""
WebSearchTool — tiered web search for aichat.

Tier 1: human_browser container (Chromium / Playwright via a12fdfeaaf78)
        Navigates to DuckDuckGo like a real user: fills the search box,
        presses Enter, waits for results, extracts text.  Timeout: 30 s.

Tier 2: httpx programmatic — direct HTTP fetch of DuckDuckGo HTML.
        Strips HTML tags and returns the raw text.  Timeout: 15 s.

Tier 3: DuckDuckGo lite endpoint — plain-text-friendly fallback.
        Timeout: 10 s.
"""
from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from .browser import BrowserTool

_DDG_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

_TIER1_TIMEOUT = 30.0
_TIER2_TIMEOUT = 15.0
_TIER3_TIMEOUT = 10.0


def _strip_html(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class WebSearchTool:
    """
    Three-tier web search:
      1. Browser (a12fdfeaaf78 / Chromium) — human-like, most reliable
      2. httpx DuckDuckGo HTML — programmatic fallback
      3. DuckDuckGo lite — minimal plaintext fallback
    """

    def __init__(self, browser: "BrowserTool") -> None:
        self._browser = browser

    async def search(self, query: str, max_chars: int = 4000) -> dict:
        """Run tiered search. Returns {query, tier, tier_name, url, content[, error]}."""
        # ------------------------------------------------------------------
        # Tier 1 — human_browser (a12fdfeaaf78)
        # ------------------------------------------------------------------
        try:
            result = await asyncio.wait_for(
                self._tier1_browser(query),
                timeout=_TIER1_TIMEOUT,
            )
            if result.get("content"):
                return self._make_result(query, 1, "browser (human-like)", result, max_chars)
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Tier 2 — httpx DuckDuckGo HTML
        # ------------------------------------------------------------------
        try:
            result = await asyncio.wait_for(
                self._tier2_httpx(query),
                timeout=_TIER2_TIMEOUT,
            )
            if result.get("content"):
                return self._make_result(query, 2, "httpx (programmatic)", result, max_chars)
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Tier 3 — DuckDuckGo lite API
        # ------------------------------------------------------------------
        try:
            result = await asyncio.wait_for(
                self._tier3_api(query),
                timeout=_TIER3_TIMEOUT,
            )
            if result.get("content"):
                return self._make_result(query, 3, "DDG lite", result, max_chars)
        except Exception:
            pass

        return {
            "query": query,
            "tier": 0,
            "tier_name": "none",
            "error": "All search tiers failed",
            "content": "",
            "url": "",
        }

    # ------------------------------------------------------------------
    # Tier implementations
    # ------------------------------------------------------------------

    async def _tier1_browser(self, query: str) -> dict:
        result = await self._browser.search(query)
        if result.get("error"):
            raise RuntimeError(result["error"])
        return result

    async def _tier2_httpx(self, query: str) -> dict:
        url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        async with httpx.AsyncClient(
            headers=_DDG_HEADERS,
            follow_redirects=True,
            timeout=_TIER2_TIMEOUT,
        ) as c:
            r = await c.get(url)
            r.raise_for_status()
            return {"url": url, "content": _strip_html(r.text)}

    async def _tier3_api(self, query: str) -> dict:
        url = f"https://lite.duckduckgo.com/lite/?q={query.replace(' ', '+')}"
        async with httpx.AsyncClient(
            headers=_DDG_HEADERS,
            follow_redirects=True,
            timeout=_TIER3_TIMEOUT,
        ) as c:
            r = await c.get(url)
            r.raise_for_status()
            return {"url": url, "content": _strip_html(r.text)}

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _make_result(query: str, tier: int, tier_name: str, raw: dict, max_chars: int) -> dict:
        content = raw.get("content", "")
        if len(content) > max_chars:
            content = content[:max_chars]
        return {
            "query": query,
            "tier": tier,
            "tier_name": tier_name,
            "url": raw.get("url", ""),
            "content": content,
        }
