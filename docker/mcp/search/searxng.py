"""SearXNG multi-instance search with failover and HTML fallback.

The local self-hosted aichat-searxng container is used as the LAST fallback.
Public SearXNG instances are tried first, shuffled each call to distribute load.

This module is self-contained — it manages its own instance pool, cooldown
tracking, and HTML result parsing.  The aichat-searxng container configuration
(settings.yml, limiter.toml) is NOT modified by this code.
"""
from __future__ import annotations

import html as _html
import random
import re
import time
from typing import Any

from helpers import BROWSER_HEADERS, SEARXNG_URL

# ---------------------------------------------------------------------------
# Public SearXNG instances — primary search backends
# Source: https://searx.space/
# ---------------------------------------------------------------------------

PUBLIC_SEARXNG_INSTANCES: list[str] = [
    "https://priv.au",
    "https://search.freestater.org",
    "https://etsi.me",
    "https://copp.gg",
    "https://search.femboy.ad",
    "https://search.unredacted.org",
    "https://searx.party",
    "https://search.rhscz.eu",
    "https://search.internetsucks.net",
    "https://ooglester.com",
    "https://search.abohiccups.com",
]

# Instance health tracking — failed instances are skipped during cooldown
_fail_until: dict[str, float] = {}
_COOLDOWN = 300.0        # 5 min cooldown for hard-failed instances
_429_COOLDOWN = 60.0     # 1 min cooldown for rate-limited instances


# ---------------------------------------------------------------------------
# Endpoint selection
# ---------------------------------------------------------------------------

def searxng_endpoints() -> list[str]:
    """Return SearXNG endpoints in priority order: public (shuffled) first, local last."""
    now = time.monotonic()
    healthy_public = [
        u for u in PUBLIC_SEARXNG_INSTANCES
        if _fail_until.get(u, 0.0) <= now
    ]
    random.shuffle(healthy_public)
    endpoints = list(healthy_public)
    if SEARXNG_URL:
        endpoints.append(SEARXNG_URL)
    return endpoints


# ---------------------------------------------------------------------------
# HTML result parsing (fallback when JSON API is blocked)
# ---------------------------------------------------------------------------

def extract_searxng_html_results(html_text: str, max_results: int = 12) -> list[dict]:
    """Parse SearXNG HTML search results into structured dicts."""
    results: list[dict] = []
    seen: set[str] = set()

    # Pattern 1: h3/h4 anchors (standard web results)
    for m in re.finditer(
        r'<h[34][^>]*>\s*<a[^>]+href="(https?://[^"]+)"[^>]*>(.*?)</a>',
        html_text, flags=re.IGNORECASE | re.DOTALL,
    ):
        url = _html.unescape(m.group(1))
        title = re.sub(r"<[^>]+>", " ", m.group(2))
        title = _html.unescape(re.sub(r"\s+", " ", title).strip())
        if url in seen or not url.startswith("http"):
            continue
        seen.add(url)
        results.append({"url": url, "title": title})
        if len(results) >= max_results:
            return results

    # Pattern 2: result class anchors
    for m in re.finditer(
        r'<a[^>]+class="[^"]*result[^"]*"[^>]+href="(https?://[^"]+)"[^>]*>(.*?)</a>',
        html_text, flags=re.IGNORECASE | re.DOTALL,
    ):
        url = _html.unescape(m.group(1))
        title = re.sub(r"<[^>]+>", " ", m.group(2))
        title = _html.unescape(re.sub(r"\s+", " ", title).strip())
        if url in seen or not url.startswith("http"):
            continue
        seen.add(url)
        results.append({"url": url, "title": title})
        if len(results) >= max_results:
            return results

    # Pattern 3: any external link with text >= 8 chars (broadest fallback)
    for m in re.finditer(
        r'<a[^>]+href="(https?://[^"]+)"[^>]*>([^<]{8,})</a>',
        html_text, flags=re.IGNORECASE,
    ):
        url = _html.unescape(m.group(1))
        title = _html.unescape(m.group(2).strip())
        if url in seen or not url.startswith("http"):
            continue
        for skip in ("searx", "searxng", "github.com/searxng", "/preferences", "/about"):
            if skip in url.lower():
                break
        else:
            seen.add(url)
            results.append({"url": url, "title": title})
            if len(results) >= max_results:
                return results

    return results


# ---------------------------------------------------------------------------
# Single-instance query
# ---------------------------------------------------------------------------

async def query_one(
    client: Any,
    base_url: str,
    params: dict[str, str],
) -> dict | None:
    """Try a single SearXNG instance; return parsed JSON or None on failure.

    Strategy:
      1. Try JSON API (format=json).
      2. On 429 (rate limit), try HTML and parse results from the page.
      3. On other errors, mark instance as failed with cooldown.
    """
    try:
        r = await client.get(
            f"{base_url}/search",
            params=params,
            headers={**BROWSER_HEADERS, "Accept": "application/json"},
            timeout=12.0,
        )
        if r.status_code == 200:
            try:
                data = r.json()
                if data.get("results"):
                    return data
            except Exception:
                pass
        elif r.status_code == 429:
            pass  # try HTML below
        else:
            _fail_until[base_url] = time.monotonic() + _COOLDOWN
            return None
    except Exception:
        _fail_until[base_url] = time.monotonic() + _COOLDOWN
        return None

    # HTML parsing fallback
    try:
        html_params = {k: v for k, v in params.items() if k != "format"}
        r = await client.get(
            f"{base_url}/search",
            params=html_params,
            headers=BROWSER_HEADERS,
            timeout=12.0,
        )
        if r.status_code == 200 and len(r.text) > 500:
            parsed = extract_searxng_html_results(r.text)
            if parsed:
                return {"results": parsed}
        elif r.status_code == 429:
            _fail_until[base_url] = time.monotonic() + _429_COOLDOWN
            return None
        else:
            _fail_until[base_url] = time.monotonic() + _COOLDOWN
            return None
    except Exception:
        _fail_until[base_url] = time.monotonic() + _429_COOLDOWN
        return None

    return None


# ---------------------------------------------------------------------------
# Public API — multi-instance web search
# ---------------------------------------------------------------------------

async def searxng_search(
    client: Any,
    query: str,
    *,
    engines: str = "",
    categories: str = "general",
    max_results: int = 10,
) -> list[tuple[str, str]]:
    """Query SearXNG with multi-instance failover; return [(url, title)].

    Tries public instances first (shuffled), then local self-hosted.
    """
    params: dict[str, str] = {
        "q": query,
        "format": "json",
        "safesearch": "0",
        "language": "en",
    }
    if engines:
        params["engines"] = engines
    if categories:
        params["categories"] = categories

    for base_url in searxng_endpoints():
        data = await query_one(client, base_url, params)
        if data is None:
            continue
        results = data.get("results") or []
        links: list[tuple[str, str]] = []
        seen: set[str] = set()
        for res in results:
            url = str(res.get("url") or "").strip()
            title = str(res.get("title") or url).strip()
            if not url.startswith("http") or url in seen:
                continue
            links.append((url, title))
            seen.add(url)
            if len(links) >= max_results:
                break
        if links:
            return links
    return []


async def searxng_image_search(
    client: Any,
    query: str,
    *,
    max_results: int = 30,
) -> list[dict]:
    """Query SearXNG image search with multi-instance failover.

    Returns list of image candidate dicts: {url, page, alt, natural_w, type}.
    """
    params: dict[str, str] = {
        "q": query,
        "format": "json",
        "safesearch": "0",
        "language": "en",
        "categories": "images",
    }

    for base_url in searxng_endpoints():
        data = await query_one(client, base_url, params)
        if data is None:
            continue
        results = data.get("results") or []
        candidates: list[dict] = []
        seen: set[str] = set()
        for res in results:
            img_src = str(res.get("img_src") or "").strip()
            if not img_src.startswith("http") or img_src in seen:
                continue
            candidates.append({
                "url":       img_src,
                "page":      str(res.get("url") or ""),
                "alt":       str(res.get("title") or ""),
                "natural_w": 0,
                "type":      "searxng",
            })
            seen.add(img_src)
            if len(candidates) >= max_results:
                break
        if candidates:
            return candidates
    return []
