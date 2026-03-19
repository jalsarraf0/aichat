"""
WebSearchTool — tiered web search for aichat.

Tier 1: human_browser container (Chromium / Playwright via a12fdfeaaf78)
        Navigates to DuckDuckGo like a real user: fills the search box,
        presses Enter, waits for results, extracts text.  Timeout: 30 s.

Tier 2: httpx programmatic — direct HTTP fetch of DuckDuckGo HTML.
        Strips HTML tags and returns the raw text.  Timeout: 15 s.

Tier 3: DuckDuckGo lite endpoint — plain-text-friendly fallback.
        Timeout: 10 s.

Race strategy (Tier 1 + 2):
  Tier 1 and Tier 2 are launched simultaneously.  Whichever returns
  non-empty content first wins; the other is cancelled.  Tier 3 is only
  tried if both Tier 1 and Tier 2 fail entirely.

  Typical result: ~3-5 s (Tier 2 wins over the network before the
  browser finishes rendering).  Worst case drops from ~55 s to ~15 s.
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
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Upgrade-Insecure-Requests": "1",
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
    Three-tier web search with Tier 1 + Tier 2 parallel racing:

      Race simultaneously:
        1. Browser (a12fdfeaaf78 / Chromium) — human-like, most reliable
        2. httpx DuckDuckGo HTML — programmatic, typically faster
      → First to return non-empty content wins; loser is cancelled.
      Fallback (only if both above fail):
        3. DuckDuckGo lite — minimal plaintext fallback
    """

    def __init__(self, browser: "BrowserTool") -> None:
        self._browser = browser

    async def search(self, query: str, max_chars: int = 4000) -> dict:
        """Race Tier 1 (browser) + Tier 2 (httpx) in parallel.

        First to return non-empty content wins; the other is cancelled.
        Falls through to Tier 3 (DDG lite) only if both fail.

        Returns {query, tier, tier_name, url, content[, error]}.
        """
        t1 = asyncio.create_task(self._run_tier1(query))
        t2 = asyncio.create_task(self._run_tier2(query))

        winner_tier = 0
        winner_name = ""
        winner_raw: dict | None = None
        pending: set = {t1, t2}

        while pending and winner_raw is None:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                if task.cancelled():
                    continue
                if task.exception() is not None:
                    continue
                tier, name, raw = task.result()
                if raw.get("content"):
                    winner_tier, winner_name, winner_raw = tier, name, raw
                    break

        # Cancel and drain whatever is still running (the loser, or both on failure)
        for task in pending:
            task.cancel()
        await asyncio.gather(t1, t2, return_exceptions=True)

        if winner_raw is not None:
            return self._make_result(query, winner_tier, winner_name, winner_raw, max_chars)

        # ── Tier 3 — DDG lite (only reached if Tier 1 + 2 both failed) ──────
        try:
            result = await asyncio.wait_for(
                self._tier3_api(query), timeout=_TIER3_TIMEOUT
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
    # Tier runners — wrap tier implementations with timeout + metadata
    # ------------------------------------------------------------------

    async def _run_tier1(self, query: str) -> tuple[int, str, dict]:
        """Run Tier 1 (browser) with timeout. Returns (tier_num, tier_name, raw_dict)."""
        result = await asyncio.wait_for(
            self._tier1_browser(query), timeout=_TIER1_TIMEOUT
        )
        return 1, "browser (human-like)", result

    async def _run_tier2(self, query: str) -> tuple[int, str, dict]:
        """Run Tier 2 (httpx) with timeout. Returns (tier_num, tier_name, raw_dict)."""
        result = await asyncio.wait_for(
            self._tier2_httpx(query), timeout=_TIER2_TIMEOUT
        )
        return 2, "httpx (programmatic)", result

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
