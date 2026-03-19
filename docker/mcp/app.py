"""
aichat MCP HTTP/SSE server — exposes all aichat tools to remote MCP clients
(LM Studio and other MCP-compatible apps) over the network.

MCP SSE transport (spec 2024-11-05):
  GET  /sse           — client connects here; receives an 'endpoint' event
                        pointing to /messages?sessionId=<id>
  POST /messages      — client sends JSON-RPC requests here
  GET  /health        — health probe

The server listens on 0.0.0.0:8096 so it is reachable from other machines
on the same network.

LM Studio mcp_servers.json entry (on the localhost machine):
  {
    "mcpServers": {
      "aichat": {
        "url": "http://<THIS_MACHINE_IP>:8096/sse"
      }
    }
  }

Screenshot support
------------------
The human_browser container must be connected to this container's Docker
network (the install script does this automatically).  Screenshots are written
to /workspace inside human_browser and are bind-mounted read-only into this
container at /browser-workspace.  The result is sent to LM Studio as an inline
base64-encoded PNG image block so it renders directly in the chat.
"""
from __future__ import annotations

import asyncio
import base64
import collections
import html as _html
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from urllib.parse import unquote as _url_unquote

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

try:
    from PIL import (
        Image as _PilImage,
        ImageEnhance as _ImageEnhance,
        ImageFilter as _ImageFilter,
        ImageStat as _ImageStat,
        ImageChops as _ImageChops,
        ImageDraw as _ImageDraw,
        ImageOps as _ImageOps,
    )
    import io as _io
    _HAS_PIL = True
except ImportError:
    _PilImage = None  # type: ignore[assignment]
    _ImageEnhance = None  # type: ignore[assignment]
    _ImageFilter = None  # type: ignore[assignment]
    _ImageStat = None  # type: ignore[assignment]
    _ImageChops = None  # type: ignore[assignment]
    _ImageDraw = None  # type: ignore[assignment]
    _ImageOps = None  # type: ignore[assignment]
    _HAS_PIL = False

try:
    import cv2 as _cv2
    import numpy as _np
    _HAS_CV2 = True
except ImportError:
    _cv2 = None  # type: ignore[assignment]
    _np  = None  # type: ignore[assignment]
    _HAS_CV2 = False

import textwrap as _textwrap
import time as _time


def _init_cv2_acceleration() -> dict[str, Any]:
    """Best-effort OpenCV acceleration bootstrap (CUDA/OpenCL/optimized kernels)."""
    status: dict[str, Any] = {
        "cv2": bool(_HAS_CV2 and _cv2 is not None),
        "cv2_version": "",
        "optimized": False,
        "cuda_devices": 0,
        "opencl_have": False,
        "opencl_use": False,
        "backend": "none",
    }
    if not _HAS_CV2 or _cv2 is None:
        return status
    try:
        status["cv2_version"] = str(getattr(_cv2, "__version__", ""))
    except Exception:
        pass
    try:
        _cv2.setUseOptimized(True)  # type: ignore[union-attr]
        status["optimized"] = bool(_cv2.useOptimized())  # type: ignore[union-attr]
    except Exception:
        pass
    try:
        status["opencl_have"] = bool(_cv2.ocl.haveOpenCL())  # type: ignore[union-attr]
        # Enable OpenCL aggressively when Intel GPU passthrough is expected.
        wants_opencl = (
            os.environ.get("INTEL_GPU", "").strip() == "1"
            or os.path.isdir("/dev/dri")
        )
        if status["opencl_have"] and wants_opencl:
            _cv2.ocl.setUseOpenCL(True)  # type: ignore[union-attr]
        status["opencl_use"] = bool(_cv2.ocl.useOpenCL())  # type: ignore[union-attr]
    except Exception:
        pass
    try:
        status["cuda_devices"] = int(_cv2.cuda.getCudaEnabledDeviceCount())  # type: ignore[union-attr]
    except Exception:
        status["cuda_devices"] = 0
    if status["cuda_devices"] > 0:
        status["backend"] = "opencv-cuda"
    elif status["opencl_use"]:
        status["backend"] = "opencv-opencl"
    else:
        status["backend"] = "opencv-cpu"
    return status


_CV2_ACCEL_STATUS = _init_cv2_acceleration()

# ---------------------------------------------------------------------------
# Perceptual hash helpers — pure PIL, no extra packages
# ---------------------------------------------------------------------------

def _dhash(img: "_PilImage.Image") -> str:
    """64-bit difference hash of a PIL Image → 16-char hex string."""
    if not _HAS_PIL:
        return ""
    try:
        gray = img.convert("L").resize((9, 8), _PilImage.LANCZOS)
        # Pillow deprecates Image.getdata(); tobytes() is stable and faster here.
        px = gray.tobytes()
        bits = sum(
            1 << i for i in range(64)
            if px[i % 8 + (i // 8) * 9] > px[i % 8 + (i // 8) * 9 + 1]
        )
        return f"{bits:016x}"
    except Exception:
        return ""


def _hamming(h1: str, h2: str) -> int:
    """Bit-level Hamming distance between two 16-hex-char dHashes."""
    if not h1 or not h2 or len(h1) != 16 or len(h2) != 16:
        return 64
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")




def _domain_from_url(url: str) -> str:
    """Lowercased hostname without leading www., or empty on parse failure."""
    from urllib.parse import urlparse as _urlparse

    try:
        host = (_urlparse(url).hostname or "").lower()
    except Exception:
        return ""
    return host.removeprefix("www.")


def _url_has_explicit_content(url: str, text: str = "") -> bool:
    """Content filter — disabled. All URLs pass through."""
    return False


def _normalize_search_query(query: str) -> tuple[str, str]:
    """Normalize common query typos; return (normalized_query, note_or_empty)."""
    original = re.sub(r"\s+", " ", (query or "")).strip()
    if not original:
        return "", ""
    fixed = original
    fixes: tuple[tuple[str, str], ...] = (
        (r"\bkluki\b", "Klukai"),
        (r"\bgirls?\s*frontline\s*2\b", "Girls Frontline 2"),
        (r"\bgirls?\s+frontline2\b", "Girls Frontline 2"),
    )
    for pat, rep in fixes:
        fixed = re.sub(pat, rep, fixed, flags=re.IGNORECASE)
    fixed = re.sub(r"\s+", " ", fixed).strip()
    if fixed.lower() == original.lower():
        return original, ""
    return fixed, f"Query normalized: '{original}' -> '{fixed}'"


def _search_terms(query: str) -> list[str]:
    """Tokenize query into lowercased word terms useful for URL relevance scoring."""
    return [w.lower() for w in re.findall(r"[a-z0-9]{3,}", (query or "").lower())]


def _query_preferred_domains(query: str) -> tuple[str, ...]:
    """Domain preferences for known entities to improve relevance ordering."""
    q = (query or "").lower()
    if any(k in q for k in ("girls frontline 2", "gfl2", "klukai")):
        return ("iopwiki.com", "gf2exilium.com", "fandom.com", "prydwen.gg")
    return tuple()


def _score_url_relevance(url: str, query_terms: list[str], preferred_domains: tuple[str, ...] = ()) -> int:
    """Simple URL relevance score used for ranking candidate pages/images."""
    url_l = (url or "").lower()
    host = _domain_from_url(url_l)
    score = 0
    for w in query_terms:
        if w in url_l:
            score += 3
        if re.search(rf"(?<![a-z0-9]){re.escape(w)}(?![a-z0-9])", url_l):
            score += 2
    for dom in preferred_domains:
        if host == dom or host.endswith(f".{dom}"):
            score += 8
    return score


def _unwrap_ddg_redirect(url: str) -> str:
    """Return target URL when the input is a DuckDuckGo redirect URL."""
    from urllib.parse import parse_qs as _parse_qs, urlparse as _urlparse

    u = (url or "").strip()
    if u.startswith("//"):
        u = "https:" + u
    if not u:
        return ""
    try:
        parsed = _urlparse(u)
        host = (parsed.hostname or "").lower()
        if "duckduckgo.com" in host and parsed.path == "/l/":
            cand = _url_unquote((_parse_qs(parsed.query).get("uddg") or [""])[0])
            if cand.startswith("http"):
                return cand
        m = re.search(r"uddg=(https?%3A[^&\s\"'>]+)", u)
        if m:
            cand2 = _url_unquote(m.group(1))
            if cand2.startswith("http"):
                return cand2
    except Exception:
        return u
    return u


def _extract_ddg_links(html: str, max_results: int = 12) -> list[tuple[str, str]]:
    """Parse DDG HTML search results into [(url, title)] with deduped URLs."""
    links: list[tuple[str, str]] = []
    seen: set[str] = set()

    # Primary: result anchors with class="result__a"
    for m in re.finditer(
        r"<a[^>]+class=[\"'][^\"']*result__a[^\"']*[\"'][^>]+href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        href = _html.unescape(m.group(1))
        title = re.sub(r"<[^>]+>", " ", m.group(2))
        title = _html.unescape(re.sub(r"\s+", " ", title).strip())
        url = _unwrap_ddg_redirect(href)
        if not url.startswith("http") or url in seen:
            continue
        links.append((url, title or url))
        seen.add(url)
        if len(links) >= max_results:
            return links

    # Fallback: uddg redirect parameters
    for enc in re.findall(r"uddg=(https?%3A[^&\"'>\s]+)", html):
        url = _url_unquote(enc)
        if not url.startswith("http") or url in seen:
            continue
        links.append((url, url))
        seen.add(url)
        if len(links) >= max_results:
            break
    return links


def _extract_bing_links(html: str, max_results: int = 12) -> list[tuple[str, str]]:
    """Parse Bing HTML web search results into [(url, title)]."""
    links: list[tuple[str, str]] = []
    seen: set[str] = set()

    def _unwrap_bing_redirect(url: str) -> str:
        from urllib.parse import parse_qs as _parse_qs, urlparse as _urlparse

        u = (url or "").strip()
        try:
            p = _urlparse(u)
            host = (p.hostname or "").lower()
            if "bing.com" in host and p.path.startswith("/ck/a"):
                raw_u = (_parse_qs(p.query).get("u") or [""])[0]
                if raw_u.startswith("a1"):
                    b64 = raw_u[2:]
                    b64 += "=" * ((4 - (len(b64) % 4)) % 4)
                    dec = base64.urlsafe_b64decode(b64.encode("ascii")).decode("utf-8", errors="ignore")
                    if dec.startswith("http"):
                        return dec
        except Exception:
            return u
        return u

    for m in re.finditer(
        r"<li[^>]+class=[\"'][^\"']*b_algo[^\"']*[\"'][^>]*>.*?<h2[^>]*>.*?<a[^>]+href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        url = _unwrap_bing_redirect(_html.unescape(m.group(1)))
        title = re.sub(r"<[^>]+>", " ", m.group(2))
        title = _html.unescape(re.sub(r"\s+", " ", title).strip())
        if not url.startswith("http") or url in seen:
            continue
        links.append((url, title or url))
        seen.add(url)
        if len(links) >= max_results:
            break
    return links


def _extract_google_links(html: str, max_results: int = 12) -> list[tuple[str, str]]:
    """Parse Google HTML search results into [(url, title)]."""
    links: list[tuple[str, str]] = []
    seen: set[str] = set()
    from urllib.parse import parse_qs as _pqs, urlparse as _up_g

    # Google wraps result links as /url?q=<target>&... — unwrap them
    def _unwrap_google(href: str) -> str:
        if href.startswith("/url?") or href.startswith("https://www.google.com/url?"):
            qs = _pqs(_up_g(href).query)
            cand = (qs.get("q") or [""])[0]
            if cand.startswith("http"):
                return cand
        return href

    # Pattern 1: anchor before or wrapping h3 (standard blue links)
    for m in re.finditer(
        r'<a[^>]+href="([^"]*)"[^>]*>(?:[^<]|<(?!h3))*?<h3[^>]*>(.*?)</h3>',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        url = _unwrap_google(_html.unescape(m.group(1)))
        title = re.sub(r"<[^>]+>", " ", m.group(2))
        title = _html.unescape(re.sub(r"\s+", " ", title).strip())
        if not url.startswith("http") or url in seen:
            continue
        host = _domain_from_url(url)
        if "google.com" in host or "googleapis.com" in host:
            continue
        links.append((url, title or url))
        seen.add(url)
        if len(links) >= max_results:
            return links

    # Pattern 2: cite elements (often contain clean destination URLs)
    for m in re.finditer(r'<cite[^>]*>([^<]+)</cite>', html, flags=re.IGNORECASE):
        raw = _html.unescape(m.group(1)).strip().split(" ")[0]
        if not raw.startswith("http"):
            raw = "https://" + raw
        url = raw.split("›")[0].strip().rstrip("/")
        if not url.startswith("http") or url in seen:
            continue
        host = _domain_from_url(url)
        if "google.com" in host:
            continue
        links.append((url, url))
        seen.add(url)
        if len(links) >= max_results:
            break

    return links


# ---------------------------------------------------------------------------
# Public SearXNG instances — used as PRIMARY search backends.
# Local self-hosted instance (SEARXNG_URL) is the LAST fallback.
# Instances are tried round-robin starting from a random offset so traffic
# is distributed across the pool and no single instance gets hammered.
# Source: https://searx.space/
# ---------------------------------------------------------------------------

_PUBLIC_SEARXNG_INSTANCES: list[str] = [
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

import random as _random

# Track which instances are healthy; unhealthy ones are skipped for a cooldown.
_searxng_fail_until: dict[str, float] = {}
_SEARXNG_COOLDOWN = 300.0  # 5 min cooldown for hard-failed (non-429) instances
_SEARXNG_429_COOLDOWN = 60.0  # 1 min cooldown for rate-limited instances


def _searxng_endpoints() -> list[str]:
    """Return SearXNG endpoints in priority order: public first, local last.

    Failed instances are skipped if still within their cooldown window.
    Public instances are shuffled each call to distribute load.
    """
    now = _time.monotonic()
    healthy_public = [
        u for u in _PUBLIC_SEARXNG_INSTANCES
        if _searxng_fail_until.get(u, 0.0) <= now
    ]
    _random.shuffle(healthy_public)
    endpoints = list(healthy_public)
    # Local self-hosted instance as last fallback
    if SEARXNG_URL:
        endpoints.append(SEARXNG_URL)
    return endpoints


def _extract_searxng_html_results(html_text: str, max_results: int = 12) -> list[dict]:
    """Parse SearXNG HTML search results into structured dicts.

    SearXNG default template uses:
      <h3><a href="URL">TITLE</a></h3>  or
      <a href="URL" class="...">TITLE</a> inside result blocks.
    Also extracts image results from <img> tags with data-src or src.
    """
    results: list[dict] = []
    seen: set[str] = set()

    # Pattern 1: h3/h4 anchors (standard web results)
    for m in re.finditer(
        r'<h[34][^>]*>\s*<a[^>]+href="(https?://[^"]+)"[^>]*>(.*?)</a>',
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
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

    # Pattern 2: result__a class (some SearXNG themes)
    for m in re.finditer(
        r'<a[^>]+class="[^"]*result[^"]*"[^>]+href="(https?://[^"]+)"[^>]*>(.*?)</a>',
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
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

    # Pattern 3: any external link with substantial text (broadest fallback)
    for m in re.finditer(
        r'<a[^>]+href="(https?://[^"]+)"[^>]*>([^<]{8,})</a>',
        html_text,
        flags=re.IGNORECASE,
    ):
        url = _html.unescape(m.group(1))
        title = _html.unescape(m.group(2).strip())
        if url in seen or not url.startswith("http"):
            continue
        # Skip self-links and navigation
        for skip in ("searx", "searxng", "github.com/searxng", "/preferences", "/about"):
            if skip in url.lower():
                break
        else:
            seen.add(url)
            results.append({"url": url, "title": title})
            if len(results) >= max_results:
                return results

    return results


async def _searxng_query_one(
    client: Any,
    base_url: str,
    params: dict[str, str],
) -> dict | None:
    """Try a single SearXNG instance; return parsed JSON or None on failure.

    Strategy:
      1. Try JSON API (format=json) — works on local instances and some public ones.
      2. On 429 (rate limit), try HTML and parse results from the page.
      3. On other errors, mark instance as failed with cooldown.
    """
    # --- Attempt 1: JSON API ---
    try:
        r = await client.get(
            f"{base_url}/search",
            params=params,
            headers={**_BROWSER_HEADERS, "Accept": "application/json"},
            timeout=12.0,
        )
        if r.status_code == 200:
            try:
                data = r.json()
                if data.get("results"):
                    return data
            except Exception:
                pass  # non-JSON response, try HTML below
        elif r.status_code == 429:
            # Rate-limited on JSON — try HTML (many instances allow HTML but block JSON API)
            pass
        else:
            _searxng_fail_until[base_url] = _time.monotonic() + _SEARXNG_COOLDOWN
            return None
    except Exception:
        _searxng_fail_until[base_url] = _time.monotonic() + _SEARXNG_COOLDOWN
        return None

    # --- Attempt 2: HTML parsing fallback (same query, no format=json) ---
    try:
        html_params = {k: v for k, v in params.items() if k != "format"}
        r = await client.get(
            f"{base_url}/search",
            params=html_params,
            headers=_BROWSER_HEADERS,
            timeout=12.0,
        )
        if r.status_code == 200 and len(r.text) > 500:
            parsed = _extract_searxng_html_results(r.text)
            if parsed:
                # Convert to the same format as JSON API response
                return {"results": parsed}
        elif r.status_code == 429:
            _searxng_fail_until[base_url] = _time.monotonic() + _SEARXNG_429_COOLDOWN
            return None
        else:
            _searxng_fail_until[base_url] = _time.monotonic() + _SEARXNG_COOLDOWN
            return None
    except Exception:
        _searxng_fail_until[base_url] = _time.monotonic() + _SEARXNG_429_COOLDOWN
        return None

    return None


async def _searxng_search(
    client: Any,
    query: str,
    *,
    engines: str = "",
    categories: str = "general",
    max_results: int = 10,
) -> list[tuple[str, str]]:
    """Query SearXNG JSON API with multi-instance failover; return [(url, title)].

    Tries public instances first (shuffled for load distribution), then falls
    back to the local self-hosted instance.  Failed instances are cooled down
    for 120 s so subsequent calls skip them immediately.
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

    for base_url in _searxng_endpoints():
        data = await _searxng_query_one(client, base_url, params)
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


async def _searxng_image_search(
    client: Any,
    query: str,
    *,
    max_results: int = 30,
) -> list[dict]:
    """Query SearXNG image search with multi-instance failover.

    Returns list of image candidate dicts with keys: url, page, alt, natural_w, type.
    Tries public instances first, local last.
    """
    params: dict[str, str] = {
        "q": query,
        "format": "json",
        "safesearch": "0",
        "language": "en",
        "categories": "images",
    }

    for base_url in _searxng_endpoints():
        data = await _searxng_query_one(client, base_url, params)
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
                "url":     img_src,
                "page":    str(res.get("url") or ""),
                "alt":     str(res.get("title") or ""),
                "natural_w": 0,
                "type":    "searxng",
            })
            seen.add(img_src)
            if len(candidates) >= max_results:
                break
        if candidates:
            return candidates
    return []


def _is_low_information_image(img: "_PilImage.Image") -> bool:
    """Detect near-solid placeholder images (e.g., pure black/white blocks)."""
    if not _HAS_PIL:
        return False
    try:
        sample = img.convert("RGB").copy()
        sample.thumbnail((96, 96), _PilImage.BILINEAR)
        stats = _ImageStat.Stat(sample)
        if not stats.mean or not stats.stddev:
            return False
        mean_luma = sum(float(v) for v in stats.mean) / 3.0
        max_std = max(float(v) for v in stats.stddev)
        return max_std < 6.0 and (mean_luma < 20.0 or mean_luma > 245.0)
    except Exception:
        return False


def _blocks_indicate_error(tool_name: str, blocks: list[dict[str, Any]]) -> bool:
    """Heuristic: detect text-only tool responses that represent errors."""
    if not blocks:
        return False
    if any(b.get("type") == "image" for b in blocks):
        return False
    texts = [
        str(b.get("text", "")).strip()
        for b in blocks
        if b.get("type") == "text" and str(b.get("text", "")).strip()
    ]
    if not texts:
        return False
    lowered = [t.lower() for t in texts]
    if any(t.startswith("error") or t.startswith("unknown tool") for t in lowered):
        return True
    prefix = (tool_name or "").lower().strip()
    if prefix:
        tool_prefix = f"{prefix}:"
        if any(t.startswith(tool_prefix) for t in lowered):
            joined = "\n".join(lowered)
            error_hints = (
                " failed",
                " not found",
                " is required",
                " unreachable",
                " timeout",
                " timed out",
                " no image found",
                " no urls found",
                " invalid",
                " unsupported",
            )
            if any(h in joined for h in error_hints):
                return True
    return False


async def _vision_confirm(
    b64: str, subject: str, hc: "httpx.AsyncClient",
    phash: str = "",
) -> "tuple[bool, str, float]":
    """Ask the loaded multimodal model whether the image shows *subject*.

    Returns (is_match, description, confidence).  Fails open on any error so
    image_search never breaks when no vision model is loaded in LM Studio.
    Uses IMAGE_GEN_BASE_URL/v1/chat/completions with 8-second timeout.
    Set IMAGE_VISION_CONFIRM=false to disable (skip confirmation, keep all images).

    phash: if provided, VisionCache is checked first (skips GPU call on cache hit)
    and the result is stored in VisionCache after a live LM Studio call.
    ModelRegistry is used for a fast bail-out when no model is loaded.
    """
    env_flag = os.environ.get("IMAGE_VISION_CONFIRM", "true").lower()
    if env_flag not in ("1", "true", "yes"):
        return True, "", 0.6
    # 1. In-memory cache check — free, instant, no GPU call
    if phash:
        cached = _vision_cache.get(phash)
        if cached is not None:
            return cached
    # 2. ModelRegistry fast bail-out — avoids the full 8 s timeout when no model is loaded
    if not await ModelRegistry.get().is_available(hc):
        return True, "", 0.6   # fail-open (same as before)
    # 3. Capacity guard — skip GPU call if the target model would cause eviction
    _cv_model = IMAGE_GEN_MODEL.strip()
    if _cv_model:
        _cv_busy = await ModelRegistry.get().ensure_model_or_busy(hc, _cv_model)
        if _cv_busy:
            return True, "", 0.6  # fail-open: don't evict a model for confirmation
    prompt = (
        f"Describe this image in one short sentence. "
        f"Then answer: does it clearly show '{subject}'? "
        f"Format: DESCRIPTION: <text> | MATCH: YES or NO"
    )
    payload: dict = {
        "messages": [{"role": "user", "content": [
            {"type": "text",      "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
        "max_tokens": 80,
        "temperature": 0.0,
    }
    model = IMAGE_GEN_MODEL.strip()
    if model:
        payload["model"] = model
    try:
        r = await asyncio.wait_for(
            hc.post(f"{IMAGE_GEN_BASE_URL}/v1/chat/completions", json=payload),
            timeout=8.0,
        )
        data = r.json()
        if data.get("error"):
            # LM Studio returns HTTP 200 with {"error": ...} when no chat model
            # is loaded — treat as unavailable and fail-open.
            result = (True, "", 0.6)
        else:
            _choices = data.get("choices", [])
            text = (_choices[0]["message"]["content"].strip() if _choices else "")
            desc  = ""
            if "DESCRIPTION:" in text:
                desc = text.split("DESCRIPTION:")[1].split("|")[0].strip()
            match = "MATCH: YES" in text.upper() or text.upper().strip().endswith("YES")
            conf  = 0.9 if match else 0.2
            result = (match, desc, conf)
    except Exception:
        result = (True, "", 0.6)   # fail-open
    # 3. Store in VisionCache for future calls
    if phash:
        _vision_cache.put(phash, result)
    return result


app = FastAPI(title="aichat-mcp")

# Allow all origins so LM Studio (Electron/WebView2) can connect without CORS issues.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Backend service URLs — consolidated service containers (reduced from 10 to 5)
# aichat-data    (8091): database + memory + graph + planner + researchbox + jobs
# aichat-vision  (8099): video + ocr
# aichat-docs    (8101): docs ingestor + pdf editing
# aichat-sandbox (8095): custom code execution (isolated)
# ---------------------------------------------------------------------------
DATABASE_URL = os.environ.get("DATABASE_URL", "http://aichat-data:8091")
MEMORY_URL    = os.environ.get("MEMORY_URL",   "http://aichat-data:8091/memory")
RESEARCH_URL  = os.environ.get("RESEARCH_URL", "http://aichat-data:8091/research")
TOOLKIT_URL   = os.environ.get("TOOLKIT_URL",  "http://aichat-sandbox:8095")
GRAPH_URL     = os.environ.get("GRAPH_URL",    "http://aichat-data:8091/graph")
VECTOR_URL    = os.environ.get("VECTOR_URL",   "http://aichat-vector:6333")
VIDEO_URL     = os.environ.get("VIDEO_URL",    "http://aichat-vision:8099")
OCR_URL       = os.environ.get("OCR_URL",      "http://aichat-vision:8099/ocr")
DOCS_URL      = os.environ.get("DOCS_URL",     "http://aichat-docs:8101")
PLANNER_URL   = os.environ.get("PLANNER_URL",  "http://aichat-data:8091/planner")
PDF_URL       = os.environ.get("PDF_URL",      "http://aichat-docs:8101/pdf")
JOB_URL       = os.environ.get("JOB_URL",      "http://aichat-data:8091/jobs")
# Self-hosted SearXNG meta-search — proxies Google/Bing/DDG/Brave without bot detection.
SEARXNG_URL   = os.environ.get("SEARXNG_URL",  "http://aichat-searxng:8080")
# human_browser browser-server API — reachable after install connects it to this network.
BROWSER_URL   = os.environ.get("BROWSER_URL",  "http://human_browser:7081")
JUPYTER_URL   = os.environ.get("JUPYTER_URL",  "http://aichat-jupyter:8098")
# Screenshot PNGs are bind-mounted from /docker/human_browser/workspace on the host.
BROWSER_WORKSPACE = os.environ.get("BROWSER_WORKSPACE", "/browser-workspace")
# Image generation — LM Studio OpenAI-compatible image API (or any compatible backend).
IMAGE_GEN_BASE_URL = os.environ.get("IMAGE_GEN_BASE_URL", "http://192.168.50.2:1234")
IMAGE_GEN_MODEL    = os.environ.get("IMAGE_GEN_MODEL", "")
# Embedding model — if unset, auto-detected from LM Studio on first use (looks for "embed" in model name).
_EMBED_MODEL_OVERRIDE = os.environ.get("EMBED_MODEL", "")
_embed_model_cache: str | None = None  # cached after first auto-detection

# MinIO S3-compatible object store
MINIO_URL       = os.environ.get("MINIO_URL", "http://aichat-minio:9002")
MINIO_ACCESS    = os.environ.get("MINIO_ROOT_USER", "minioadmin")
MINIO_SECRET    = os.environ.get("MINIO_ROOT_PASSWORD", "")
CLIP_URL        = os.environ.get("CLIP_URL", "http://aichat-vision:8099/clip")
BROWSER_AUTO_URL = os.environ.get("BROWSER_AUTO_URL", "http://aichat-browser:8104")
DETECT_URL       = os.environ.get("DETECT_URL", "http://aichat-vision:8099/detect")
_MINIO_BUCKET   = "aichat-images"


# Lazy-initialized MinIO client
_minio_client = None

def _get_minio():
    """Lazy-initialize the MinIO client and ensure the bucket exists."""
    global _minio_client
    if _minio_client is not None:
        return _minio_client
    try:
        from minio import Minio  # noqa: E402
        endpoint = MINIO_URL.replace("http://", "").replace("https://", "")
        _minio_client = Minio(
            endpoint,
            access_key=MINIO_ACCESS,
            secret_key=MINIO_SECRET,
            secure=MINIO_URL.startswith("https"),
        )
        if not _minio_client.bucket_exists(_MINIO_BUCKET):
            _minio_client.make_bucket(_MINIO_BUCKET)
        return _minio_client
    except Exception:
        return None

async def _get_embed_model() -> str:
    """Return the embedding model to use. Prefers EMBED_MODEL env, then auto-detects from LM Studio."""
    global _embed_model_cache
    if _EMBED_MODEL_OVERRIDE:
        return _EMBED_MODEL_OVERRIDE
    if _embed_model_cache:
        return _embed_model_cache
    try:
        async with httpx.AsyncClient(timeout=5.0) as _hc:
            _r = await _hc.get(f"{IMAGE_GEN_BASE_URL}/v1/models")
            _models = [m["id"] for m in _r.json().get("data", [])]
        # Prefer a model with "embed" in the name
        for _m in _models:
            if "embed" in _m.lower():
                _embed_model_cache = _m
                return _m
        # Fall back to IMAGE_GEN_MODEL or first available
        _embed_model_cache = IMAGE_GEN_MODEL or (_models[0] if _models else "")
    except Exception:
        _embed_model_cache = IMAGE_GEN_MODEL
    return _embed_model_cache or ""

# Max seconds a single tool call may run before it is cancelled.
_TOOL_TIMEOUT = 180.0

# Active SSE sessions: session_id -> asyncio.Queue
_sessions: dict[str, asyncio.Queue] = {}

# ---------------------------------------------------------------------------
# Orchestrator + source strategy singletons (lazy-init)
# ---------------------------------------------------------------------------

import contextvars as _contextvars

_current_session_id: _contextvars.ContextVar[str] = _contextvars.ContextVar(
    "session_id", default=""
)
_current_request_id: _contextvars.ContextVar[str] = _contextvars.ContextVar(
    "request_id", default=""
)

_orchestrator_instance = None  # type: ignore[assignment]


def _get_orchestrator():
    """Lazy singleton — created on first call inside the event loop."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        from orchestrator import Orchestrator

        _orchestrator_instance = Orchestrator()

        async def _progress_cb(
            tool_name: str, status: str, detail: str, percent: int
        ) -> None:
            sid = _current_session_id.get("")
            rid = _current_request_id.get("")
            if sid and sid in _sessions and rid:
                await _sessions[sid].put({
                    "jsonrpc": "2.0",
                    "method": "notifications/progress",
                    "params": {
                        "progressToken": rid,
                        "progress": percent,
                        "total": 100,
                        "message": f"[{status}] {tool_name}: {detail}",
                    },
                })

        _orchestrator_instance.register_progress_callback(_progress_cb)
    return _orchestrator_instance


_source_strategy_instance = None  # type: ignore[assignment]


def _get_source_strategy():
    """Lazy singleton for search result ranking."""
    global _source_strategy_instance
    if _source_strategy_instance is None:
        from source_strategy import SourceStrategy

        _source_strategy_instance = SourceStrategy()
    return _source_strategy_instance


# ---------------------------------------------------------------------------
# Logging + error reporting
# ---------------------------------------------------------------------------

import logging as _logging

_log = _logging.getLogger("aichat-mcp")
_logging.basicConfig(
    level=_logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_SERVICE_NAME = "aichat-mcp"


async def _report_error(message: str, detail: str | None = None) -> None:
    """Fire-and-forget: send an error entry to aichat-database."""
    try:
        async with httpx.AsyncClient(timeout=5) as _c:
            await _c.post(
                f"{DATABASE_URL}/errors/log",
                json={"service": _SERVICE_NAME, "level": "ERROR",
                      "message": message, "detail": detail},
            )
    except Exception:
        pass  # never let error reporting crash the MCP server


@app.exception_handler(Exception)
async def global_exception_handler(request: "Request", exc: Exception) -> "Response":
    from fastapi.responses import JSONResponse as _JSONResponse
    message = str(exc)
    detail = f"{request.method} {request.url.path}"
    _log.error("Unhandled error [%s %s]: %s", request.method, request.url.path, exc, exc_info=True)
    asyncio.create_task(_report_error(message, detail))
    return _JSONResponse(status_code=500, content={"error": message})


# Realistic browser headers — used for all outbound httpx requests to reduce
# bot-detection and rate-limit exposure.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

# ---------------------------------------------------------------------------
# Tool schemas exposed to MCP clients
# ---------------------------------------------------------------------------

_TOOLS: list[dict[str, Any]] = [
    # ==================================================================
    # 1. web — Web search, fetch, extract, summarize, news, wiki, arxiv, youtube
    # ==================================================================
    {
        "name": "web",
        "description": (
            "Unified web information tool. Search the web, fetch pages, extract articles, "
            "summarize text, search news, query Wikipedia, search arXiv papers, and get YouTube transcripts.\n"
            "Actions:\n"
            "  search — Search the web (SearXNG, DDG, Bing). Returns URLs + titles.\n"
            "  fetch — Fetch a web page as readable text. Checks cache first.\n"
            "  extract — Extract clean article from URL, structured data from text, or page elements.\n"
            "  summarize — Summarize text via LM Studio (brief/detailed/bullets).\n"
            "  news — Search current news from RSS feeds (BBC, Reuters, HN, etc.).\n"
            "  wikipedia — Search Wikipedia for summaries or full articles.\n"
            "  arxiv — Search arXiv academic papers.\n"
            "  youtube — Extract transcript/captions from a YouTube video."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["search", "quick_search", "fetch", "extract", "summarize", "news", "wikipedia", "arxiv", "youtube"],
                    "description": "Which web action to perform. Use quick_search for fast results (1-3s).",
                },
                "url": {"type": "string", "description": "URL for fetch/extract/youtube."},
                "query": {"type": "string", "description": "Search query for search/news/wikipedia/arxiv."},
                "engine": {
                    "type": "string",
                    "enum": ["auto", "google", "bing", "ddg", "images", "searxng"],
                    "description": "Search engine (search action). Default: auto.",
                },
                "max_chars": {"type": "integer", "description": "Max chars to return (fetch/extract). Default 4000."},
                "content": {"type": "string", "description": "Text to summarize (summarize action)."},
                "style": {
                    "type": "string", "enum": ["brief", "detailed", "bullets"],
                    "description": "Summary style (summarize). Default: brief.",
                },
                "max_words": {"type": "integer", "description": "Target word count (summarize)."},
                "sources": {"type": "array", "items": {"type": "string"}, "description": "News source filter (news)."},
                "limit": {"type": "integer", "description": "Max results (news/arxiv)."},
                "lang": {"type": "string", "description": "Language code (wikipedia/youtube). Default 'en'."},
                "full_article": {"type": "boolean", "description": "Full Wikipedia article text (wikipedia). Default false."},
                "max_results": {"type": "integer", "description": "Papers to return (arxiv). Default 8."},
                "category": {"type": "string", "description": "arXiv category (arxiv). E.g. 'cs.AI'."},
                "sort_by": {
                    "type": "string", "enum": ["relevance", "lastUpdatedDate", "submittedDate"],
                    "description": "Sort order (arxiv). Default relevance.",
                },
                "include_timestamps": {"type": "boolean", "description": "Include timestamps (youtube). Default true."},
                "include": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Data types for page extraction: links, headings, tables, images, meta, text.",
                },
                "max_links": {"type": "integer", "description": "Max links (extract). Default 50."},
                "max_text": {"type": "integer", "description": "Max text chars (extract). Default 3000."},
                "schema_json": {"type": "string", "description": "JSON Schema for structured extraction."},
                "instructions": {"type": "string", "description": "Extra extraction instructions."},
            },
            "required": ["action"],
        },
    },
    # ==================================================================
    # 2. browser — Full browser control, screenshots, scraping
    # ==================================================================
    {
        "name": "browser",
        "description": (
            "Control a real Chromium browser, take screenshots, scrape pages, and download images.\n"
            "Actions:\n"
            "  navigate — Go to a URL, return page title + text.\n"
            "  read — Return current page content without navigating.\n"
            "  click — Click a CSS selector or coordinates.\n"
            "  scroll — Scroll the page (up/down/left/right).\n"
            "  fill — Type text into a form field by CSS selector.\n"
            "  eval — Run JavaScript on the page.\n"
            "  screenshot — Take a single-URL screenshot (returns inline image).\n"
            "  screenshot_search — Search web + screenshot top results.\n"
            "  bulk_screenshot — Screenshot multiple URLs in parallel.\n"
            "  scroll_screenshot — Full-page screenshot via scroll stitching.\n"
            "  screenshot_element — Screenshot a specific CSS element.\n"
            "  save_images — Download specific image URLs via browser session.\n"
            "  download_images — Download all page images via browser session.\n"
            "  list_images — List all <img> elements on page with details.\n"
            "  scrape — Scroll entire page, extract all rendered text (lazy-load friendly).\n"
            "  keyboard — Press keyboard keys (Enter, Tab, Escape, etc.).\n"
            "  fill_form — Fill multiple form fields at once."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "navigate", "read", "click", "scroll", "fill", "eval",
                        "screenshot", "screenshot_search", "bulk_screenshot", "scroll_screenshot",
                        "screenshot_element", "save_images", "download_images", "list_images",
                        "scrape", "keyboard", "fill_form",
                    ],
                    "description": "Browser action to perform.",
                },
                "url": {"type": "string", "description": "URL (navigate/screenshot/scrape/download_images)."},
                "query": {"type": "string", "description": "Search query (screenshot_search)."},
                "urls": {"description": "URL list (bulk_screenshot/save_images). Array or comma-separated string."},
                "selector": {"type": "string", "description": "CSS selector (click/fill/screenshot_element)."},
                "value": {"type": "string", "description": "Text to type (fill action)."},
                "code": {"type": "string", "description": "JavaScript expression (eval action)."},
                "key": {"type": "string", "description": "Keyboard key (keyboard action). E.g. Enter, Tab."},
                "fields": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"selector": {"type": "string"}, "value": {"type": "string"}}},
                    "description": "Form fields to fill (fill_form action).",
                },
                "find_text": {"type": "string", "description": "Text to zoom into (screenshot action)."},
                "find_image": {"type": "string", "description": "Match <img> by src/alt/index (screenshot action)."},
                "button": {"type": "string", "enum": ["left", "right", "middle"], "description": "Mouse button (click)."},
                "click_count": {"type": "integer", "description": "Click count (click). Default 1."},
                "direction": {"type": "string", "enum": ["up", "down", "left", "right"], "description": "Scroll direction."},
                "amount": {"type": "integer", "description": "Pixels to scroll. Default 800."},
                "behavior": {"type": "string", "enum": ["instant", "smooth"], "description": "Scroll behavior."},
                "x": {"type": "number", "description": "X coordinate (click)."},
                "y": {"type": "number", "description": "Y coordinate (click)."},
                "pad": {"type": "integer", "description": "Padding px for screenshot_element. Default 20."},
                "prefix": {"type": "string", "description": "Filename prefix (save_images/download_images)."},
                "max": {"type": "integer", "description": "Max images (save_images/download_images). Default 20."},
                "filter": {"type": "string", "description": "Image src/alt filter (download_images)."},
                "max_results": {"type": "integer", "description": "Result pages to screenshot (screenshot_search). Default 3."},
                "max_scrolls": {"type": "integer", "description": "Scroll steps (scrape/scroll_screenshot)."},
                "scroll_overlap": {"type": "integer", "description": "Pixel overlap (scroll_screenshot). Default 100."},
                "wait_ms": {"type": "integer", "description": "Wait after scroll (scrape). Default 500."},
                "include_links": {"type": "boolean", "description": "Include links (scrape). Default false."},
                "wait_until": {"type": "string", "description": "Wait event (navigate). Default: domcontentloaded."},
                "full_page": {"type": "boolean", "description": "Full page capture. Default false."},
                "what": {"type": "string", "description": "What to extract: text, links, images."},
                "clear_first": {"type": "boolean", "description": "Clear field before typing. Default false."},
                "text": {"type": "string", "description": "Text to type (keyboard/type actions)."},
                "expression": {"type": "string", "description": "JavaScript expression (eval via browser_evaluate)."},
            },
            "required": ["action"],
        },
    },
    # ==================================================================
    # 3. image — Image fetch, search, generate, edit, crop, zoom, enhance, etc.
    # ==================================================================
    {
        "name": "image",
        "description": (
            "Unified image tool for fetching, searching, generating, editing, and analyzing images.\n"
            "Actions:\n"
            "  fetch — Download an image from URL and return inline.\n"
            "  search — Search for images by text description (SearXNG/DDG/Bing). Returns inline images.\n"
            "  generate — Generate image from text prompt (LM Studio FLUX/SDXL).\n"
            "  edit — Edit/remix existing image with text prompt (img2img).\n"
            "  crop — Crop image to pixel region.\n"
            "  zoom — Zoom into image region with scale factor.\n"
            "  enhance — Adjust contrast/sharpness/brightness, convert to greyscale.\n"
            "  scan — Prepare image region for text reading (greyscale + contrast + sharpen).\n"
            "  stitch — Combine 2-8 images side-by-side or stacked.\n"
            "  diff — Pixel-level visual diff between two images.\n"
            "  annotate — Draw bounding boxes and labels on an image.\n"
            "  caption — Describe an image using LM Studio vision model.\n"
            "  upscale — Upscale image via GPU AI or CPU LANCZOS.\n"
            "  remix — Creative style-transfer of an image.\n"
            "  face_detect — Detect faces (real photos + anime). Optional same-person comparison.\n"
            "  similarity — Semantic image search using CLIP embeddings in Qdrant."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "fetch", "search", "generate", "edit", "crop", "zoom", "enhance",
                        "scan", "stitch", "diff", "annotate", "caption", "upscale", "remix",
                        "face_detect", "similarity",
                    ],
                    "description": "Image action to perform.",
                },
                "url": {"type": "string", "description": "Image URL (fetch) or page URL (search)."},
                "path": {"type": "string", "description": "Workspace image path (crop/zoom/enhance/scan/edit/remix/upscale/face_detect/annotate)."},
                "query": {"type": "string", "description": "Search query (search/similarity) or generation prompt (generate)."},
                "prompt": {"type": "string", "description": "Text prompt (generate/edit/remix)."},
                "negative_prompt": {"type": "string", "description": "Avoid in generation (generate/edit)."},
                "model": {"type": "string", "description": "Model name (generate/edit)."},
                "size": {"type": "string", "description": "Image dimensions WxH (generate/edit). Default '512x512'."},
                "n": {"type": "integer", "description": "Number of images (generate/edit/remix). Default 1."},
                "steps": {"type": "integer", "description": "Inference steps (generate)."},
                "guidance_scale": {"type": "number", "description": "Prompt adherence 1-20 (generate)."},
                "seed": {"type": "integer", "description": "Random seed (generate). -1 = random."},
                "strength": {"type": "number", "description": "Transform strength 0.0-1.0 (edit/remix). Default 0.75."},
                "left": {"type": "integer", "description": "Left edge px (crop/zoom/scan). Default 0."},
                "top": {"type": "integer", "description": "Top edge px (crop/zoom/scan). Default 0."},
                "right": {"type": "integer", "description": "Right edge px (crop/zoom/scan)."},
                "bottom": {"type": "integer", "description": "Bottom edge px (crop/zoom/scan)."},
                "scale": {"type": "number", "description": "Zoom/upscale factor (zoom/upscale). Default 2.0."},
                "contrast": {"type": "number", "description": "Contrast multiplier (enhance). Default 1.5."},
                "sharpness": {"type": "number", "description": "Sharpness multiplier (enhance). Default 1.5."},
                "brightness": {"type": "number", "description": "Brightness multiplier (enhance). Default 1.0."},
                "grayscale": {"type": "boolean", "description": "Convert to greyscale (enhance). Default false."},
                "paths": {"type": "array", "items": {"type": "string"}, "description": "Image paths to combine (stitch). 2-8 images."},
                "direction": {"type": "string", "enum": ["horizontal", "vertical"], "description": "Stitch direction. Default vertical."},
                "gap": {"type": "integer", "description": "Pixel gap between images (stitch). Default 0."},
                "path_a": {"type": "string", "description": "First image (diff)."},
                "path_b": {"type": "string", "description": "Second image (diff)."},
                "amplify": {"type": "number", "description": "Diff amplification factor (diff). Default 3.0."},
                "boxes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "left": {"type": "integer"}, "top": {"type": "integer"},
                            "right": {"type": "integer"}, "bottom": {"type": "integer"},
                            "label": {"type": "string"}, "color": {"type": "string"},
                        },
                        "required": ["left", "top", "right", "bottom"],
                    },
                    "description": "Bounding boxes (annotate). Each has left/top/right/bottom + optional label/color.",
                },
                "outline_width": {"type": "integer", "description": "Box outline width (annotate). Default 3."},
                "b64": {"type": "string", "description": "Base64-encoded JPEG (caption)."},
                "detail_level": {"type": "string", "enum": ["brief", "detailed"], "description": "Caption detail (caption). Default detailed."},
                "sharpen": {"type": "boolean", "description": "Sharpen after CPU upscale (upscale). Default true."},
                "gpu": {"type": "boolean", "description": "Use GPU for upscale (upscale). Default true."},
                "count": {"type": "integer", "description": "Number of images to return (search). Default 4."},
                "offset": {"type": "integer", "description": "Skip first N results (search). Default 0."},
                "domains": {"type": "array", "items": {"type": "string"}, "description": "Domain allow-list (search)."},
                "reference_path": {"type": "string", "description": "Reference face image (face_detect)."},
                "style": {"type": "string", "enum": ["auto", "anime"], "description": "Detection style (face_detect). Default auto."},
                "match_threshold": {"type": "number", "description": "Same-person threshold (face_detect). Default 0.82."},
                "min_face_size": {"type": "integer", "description": "Min face px (face_detect). Default 40."},
                "min_neighbors": {"type": "integer", "description": "Detection neighbors (face_detect). Default 5."},
                "scale_factor": {"type": "number", "description": "Detection scale step (face_detect). Default 1.1."},
                "annotate_faces": {"type": "boolean", "description": "Draw face boxes (face_detect). Default true."},
                "top_k": {"type": "integer", "description": "Results count (similarity). Default 5."},
            },
            "required": ["action"],
        },
    },
    # ==================================================================
    # 4. document — Document ingestion, tables, OCR, PDF operations
    # ==================================================================
    {
        "name": "document",
        "description": (
            "Document processing: ingest, extract tables, OCR, and PDF operations.\n"
            "Actions:\n"
            "  ingest — Convert PDF/DOCX/XLSX/PPTX/HTML/MD/TXT to clean Markdown.\n"
            "  tables — Extract all tables from a document as structured JSON.\n"
            "  ocr — Extract text from an image using Tesseract OCR.\n"
            "  ocr_pdf — Extract text from a scanned PDF using Tesseract OCR.\n"
            "  pdf_read — Read PDF with text extraction, OCR, or auto mode.\n"
            "  pdf_edit — Edit PDF: replace text, redact, annotate, insert, rotate, reorder.\n"
            "  pdf_form — Fill AcroForm fields in a PDF.\n"
            "  pdf_merge — Merge multiple PDFs into one.\n"
            "  pdf_split — Split a PDF into multiple files by page ranges."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["ingest", "tables", "ocr", "ocr_pdf", "pdf_read", "pdf_edit", "pdf_form", "pdf_merge", "pdf_split"],
                    "description": "Document action to perform.",
                },
                "url": {"type": "string", "description": "Document URL (ingest/tables)."},
                "path": {"type": "string", "description": "Workspace file path."},
                "filename": {"type": "string", "description": "Filename with extension (ingest/tables format detection)."},
                "lang": {"type": "string", "description": "Tesseract language code (ocr/ocr_pdf/pdf_read). Default 'eng'."},
                "pages": {
                    "type": "array", "items": {"type": "integer"},
                    "description": "Specific 1-based pages (ocr_pdf/pdf_read).",
                },
                "mode": {
                    "type": "string", "enum": ["text", "ocr", "auto"],
                    "description": "PDF read mode (pdf_read). Default auto.",
                },
                "password": {"type": "string", "description": "Password for encrypted PDFs."},
                "output_path": {"type": "string", "description": "Output path in workspace (pdf_edit/pdf_form/pdf_merge)."},
                "verify": {"type": "boolean", "description": "Post-edit verification (pdf_edit). Default true."},
                "operations": {
                    "type": "array", "items": {"type": "object"},
                    "description": "Edit operations (pdf_edit): replace_text, redact_text, annotate, insert_text, rotate_page, reorder_pages, delete_pages.",
                },
                "fields": {"type": "object", "description": "Form field_name -> value map (pdf_form)."},
                "flatten": {"type": "boolean", "description": "Flatten filled widgets (pdf_form). Default false."},
                "paths": {
                    "type": "array", "items": {"type": "string"},
                    "description": "PDF paths to merge (pdf_merge). 2+ required.",
                },
                "passwords": {"type": "object", "description": "Password map by input path (pdf_merge)."},
                "ranges": {"type": "array", "items": {}, "description": "Page ranges for split (pdf_split)."},
                "prefix": {"type": "string", "description": "Output filename prefix (pdf_split)."},
                "output_dir": {"type": "string", "description": "Output directory (pdf_split)."},
            },
            "required": ["action"],
        },
    },
    # ==================================================================
    # 5. media — Video analysis, TTS, object/human detection
    # ==================================================================
    {
        "name": "media",
        "description": (
            "Media processing: video analysis, text-to-speech, and object detection.\n"
            "Actions:\n"
            "  video_info — Get video metadata (duration, fps, resolution, codec).\n"
            "  video_frames — Extract frames at regular intervals.\n"
            "  video_thumbnail — Extract single frame at timestamp.\n"
            "  video_transcode — Transcode video with Intel Arc GPU acceleration.\n"
            "  tts — Convert text to speech via LM Studio.\n"
            "  detect_objects — Detect objects in image using YOLOv8n (80 classes).\n"
            "  detect_humans — Detect people in image with bounding boxes."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["video_info", "video_frames", "video_thumbnail", "video_transcode", "tts", "detect_objects", "detect_humans"],
                    "description": "Media action to perform.",
                },
                "url": {"type": "string", "description": "Video URL (video_*)."},
                "interval_sec": {"type": "number", "description": "Seconds between frames (video_frames). Default 5.0."},
                "max_frames": {"type": "integer", "description": "Max frames (video_frames). Default 20."},
                "timestamp_sec": {"type": "number", "description": "Timestamp in seconds (video_thumbnail). Default 0."},
                "codec": {"type": "string", "enum": ["h264", "hevc", "vp9", "av1"], "description": "Target codec (video_transcode). Default h264."},
                "bitrate": {"type": "string", "description": "Target bitrate (video_transcode). Default '5M'."},
                "width": {"type": "integer", "description": "Target width (video_transcode)."},
                "height": {"type": "integer", "description": "Target height (video_transcode)."},
                "filename": {"type": "string", "description": "Output filename (video_transcode)."},
                "text": {"type": "string", "description": "Text to convert (tts)."},
                "voice": {"type": "string", "description": "TTS voice (tts). Default 'alloy'."},
                "speed": {"type": "number", "description": "Speech speed 0.25-4.0 (tts). Default 1.0."},
                "format": {"type": "string", "description": "Audio format (tts): mp3/opus/aac/flac/wav. Default mp3."},
                "image_base64": {"type": "string", "description": "Base64 image (detect_objects/detect_humans)."},
                "image_url": {"type": "string", "description": "Image URL (detect_objects/detect_humans)."},
                "confidence": {"type": "number", "description": "Min confidence 0-1 (detect_objects/detect_humans). Default 0.25."},
                "classes": {"type": "array", "items": {"type": "string"}, "description": "Object class filter (detect_objects)."},
            },
            "required": ["action"],
        },
    },
    # ==================================================================
    # 6. data — Database storage, search, cache, images, errors
    # ==================================================================
    {
        "name": "data",
        "description": (
            "Database operations: store/search articles, cache pages, manage images, query errors.\n"
            "Actions:\n"
            "  store_article — Store article in PostgreSQL.\n"
            "  search — Search stored articles by topic/full-text.\n"
            "  cache_store — Cache a web page in PostgreSQL.\n"
            "  cache_get — Retrieve cached web page.\n"
            "  store_image — Save image reference to registry.\n"
            "  list_images — List saved images from registry.\n"
            "  errors — Query structured error log from all services."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["store_article", "search", "cache_store", "cache_get", "store_image", "list_images", "errors"],
                    "description": "Data action to perform.",
                },
                "url": {"type": "string", "description": "URL (store_article/cache_store/cache_get/store_image)."},
                "title": {"type": "string", "description": "Title (store_article/cache_store)."},
                "content": {"type": "string", "description": "Content text (store_article/cache_store)."},
                "topic": {"type": "string", "description": "Topic tag (store_article/search)."},
                "q": {"type": "string", "description": "Full-text query (search)."},
                "limit": {"type": "integer", "description": "Max results. Default 20-50."},
                "offset": {"type": "integer", "description": "Skip first N (search). Default 0."},
                "summary_only": {"type": "boolean", "description": "Truncate content (search). Default false."},
                "host_path": {"type": "string", "description": "Host file path (store_image)."},
                "alt_text": {"type": "string", "description": "Image description (store_image)."},
                "service": {"type": "string", "description": "Service name filter (errors)."},
            },
            "required": ["action"],
        },
    },
    # ==================================================================
    # 7. memory — Persistent key-value memory
    # ==================================================================
    {
        "name": "memory",
        "description": (
            "Persistent key-value memory store.\n"
            "Actions:\n"
            "  store — Save a key-value note with optional TTL.\n"
            "  recall — Look up a note by key, pattern, or list all."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["store", "recall"],
                    "description": "Memory action.",
                },
                "key": {"type": "string", "description": "Memory key."},
                "value": {"type": "string", "description": "Value to store (store action)."},
                "ttl_seconds": {"type": "integer", "description": "Expiry in seconds (store). Omit for permanent."},
                "pattern": {"type": "string", "description": "SQL LIKE pattern (recall). E.g. 'whatsapp:%'."},
            },
            "required": ["action"],
        },
    },
    # ==================================================================
    # 8. knowledge — Knowledge graph nodes, edges, queries
    # ==================================================================
    {
        "name": "knowledge",
        "description": (
            "Knowledge graph operations.\n"
            "Actions:\n"
            "  add_node — Add/update a node with labels and properties.\n"
            "  add_edge — Add directed edge between two nodes.\n"
            "  query — Get a node and all connected neighbors.\n"
            "  path — Find shortest path between two nodes (BFS).\n"
            "  search — Search nodes by label and/or properties."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add_node", "add_edge", "query", "path", "search"],
                    "description": "Knowledge graph action.",
                },
                "id": {"type": "string", "description": "Node ID (add_node/query)."},
                "labels": {"type": "array", "items": {"type": "string"}, "description": "Node labels (add_node)."},
                "properties": {"type": "object", "description": "Node/edge metadata."},
                "from_id": {"type": "string", "description": "Source node (add_edge/path)."},
                "to_id": {"type": "string", "description": "Target node (add_edge/path)."},
                "type": {"type": "string", "description": "Relationship type (add_edge)."},
                "label": {"type": "string", "description": "Label filter (search)."},
                "limit": {"type": "integer", "description": "Max results (search). Default 50."},
            },
            "required": ["action"],
        },
    },
    # ==================================================================
    # 9. vector — Vector store (Qdrant) + embeddings
    # ==================================================================
    {
        "name": "vector",
        "description": (
            "Vector database (Qdrant) and embedding operations.\n"
            "Actions:\n"
            "  store — Embed text and store in Qdrant.\n"
            "  search — Semantic search in Qdrant.\n"
            "  delete — Delete vector entry by ID.\n"
            "  collections — List all Qdrant collections.\n"
            "  embed_store — Embed text via LM Studio and store in PostgreSQL.\n"
            "  embed_search — Semantic search over PostgreSQL embeddings."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["store", "search", "delete", "collections", "embed_store", "embed_search"],
                    "description": "Vector action.",
                },
                "text": {"type": "string", "description": "Text to embed (store)."},
                "query": {"type": "string", "description": "Search query (search/embed_search)."},
                "id": {"type": "string", "description": "Entry ID (store/delete)."},
                "collection": {"type": "string", "description": "Qdrant collection (store/search/delete). Default 'default'."},
                "metadata": {"type": "object", "description": "Metadata (store)."},
                "top_k": {"type": "integer", "description": "Results count (search). Default 5."},
                "filter": {"type": "object", "description": "Qdrant payload filter (search)."},
                "key": {"type": "string", "description": "Document key (embed_store)."},
                "content": {"type": "string", "description": "Text to embed (embed_store)."},
                "topic": {"type": "string", "description": "Topic tag (embed_store/embed_search)."},
                "limit": {"type": "integer", "description": "Max results (embed_search). Default 5."},
            },
            "required": ["action"],
        },
    },
    # ==================================================================
    # 10. code — Execute Python, JavaScript, Jupyter
    # ==================================================================
    {
        "name": "code",
        "description": (
            "Execute code in sandboxed environments.\n"
            "Actions:\n"
            "  python — Run Python in a sandboxed subprocess (30s timeout, GPU auto-detection).\n"
            "  javascript — Run JavaScript via Node.js.\n"
            "  jupyter — Execute Python in a persistent Jupyter kernel (state preserved between calls)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["python", "javascript", "jupyter"],
                    "description": "Code execution environment.",
                },
                "code": {"type": "string", "description": "Source code to execute."},
                "packages": {"type": "array", "items": {"type": "string"}, "description": "pip packages to install (python)."},
                "timeout": {"type": "integer", "description": "Timeout in seconds. Default 30 (python/js), 60 (jupyter)."},
                "session_id": {"type": "string", "description": "Kernel session name (jupyter). Default 'default'."},
                "reset": {"type": "boolean", "description": "Restart kernel (jupyter). Default false."},
            },
            "required": ["action", "code"],
        },
    },
    # ==================================================================
    # 11. custom_tools — Create, list, delete, call custom tools
    # ==================================================================
    {
        "name": "custom_tools",
        "description": (
            "Manage and execute custom tools created at runtime.\n"
            "Actions:\n"
            "  create — Create a new persistent custom tool (Python code in Docker sandbox).\n"
            "  list — List all custom tools.\n"
            "  delete — Delete a custom tool.\n"
            "  call — Execute a custom tool by name."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "list", "delete", "call"],
                    "description": "Custom tool action.",
                },
                "tool_name": {"type": "string", "description": "Tool name (create/delete/call)."},
                "description": {"type": "string", "description": "Tool description (create)."},
                "parameters": {"type": "object", "description": "JSON Schema for inputs (create)."},
                "code": {"type": "string", "description": "Python implementation (create). Body of async def run(**kwargs)->str."},
                "params": {"type": "object", "description": "Parameters to pass (call)."},
            },
            "required": ["action"],
        },
    },
    # ==================================================================
    # 12. planner — Task planning and orchestration
    # ==================================================================
    {
        "name": "planner",
        "description": (
            "Task planner and workflow orchestration.\n"
            "Actions:\n"
            "  create — Create a task with dependencies.\n"
            "  get — Get task status by ID.\n"
            "  complete — Mark task as done.\n"
            "  fail — Mark task as failed.\n"
            "  list — List tasks (filter by status).\n"
            "  delete — Delete a task.\n"
            "  orchestrate — Execute multi-step workflow with auto-parallelism.\n"
            "  plan — Generate a step-by-step tool execution plan via LLM."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "get", "complete", "fail", "list", "delete", "orchestrate", "plan"],
                    "description": "Planner action.",
                },
                "id": {"type": "string", "description": "Task ID (get/complete/fail/delete)."},
                "title": {"type": "string", "description": "Task title (create)."},
                "description": {"type": "string", "description": "Task description (create)."},
                "depends_on": {"type": "array", "items": {"type": "string"}, "description": "Dependency task IDs (create)."},
                "priority": {"type": "integer", "description": "Priority (create). Default 0."},
                "due_at": {"type": "string", "description": "ISO 8601 due date (create)."},
                "metadata": {"type": "object", "description": "Arbitrary metadata (create)."},
                "status": {"type": "string", "description": "Filter status (list)."},
                "limit": {"type": "integer", "description": "Max results (list). Default 50."},
                "detail": {"type": "string", "description": "Failure reason (fail)."},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"}, "tool": {"type": "string"},
                            "args": {"type": "object"},
                            "depends_on": {"type": "array", "items": {"type": "string"}},
                            "label": {"type": "string"},
                        },
                        "required": ["id", "tool", "args"],
                    },
                    "description": "Workflow steps (orchestrate).",
                },
                "stop_on_error": {"type": "boolean", "description": "Abort on step failure (orchestrate). Default false."},
                "task": {"type": "string", "description": "Natural-language task to plan (plan)."},
                "context": {"type": "string", "description": "Extra context (plan)."},
                "extra_tools": {"type": "string", "description": "Additional tool descriptions (plan)."},
                "max_steps": {"type": "integer", "description": "Max plan steps (plan). Default 10."},
            },
            "required": ["action"],
        },
    },
    # ==================================================================
    # 13. jobs — Async job system
    # ==================================================================
    {
        "name": "jobs",
        "description": (
            "Async background job system for long-running tool calls.\n"
            "Actions:\n"
            "  submit — Submit a tool call as background job. Returns job_id.\n"
            "  status — Check job status.\n"
            "  result — Get completed job output.\n"
            "  cancel — Cancel a job.\n"
            "  list — List jobs with optional filters.\n"
            "  batch — Submit multiple jobs at once."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["submit", "status", "result", "cancel", "list", "batch"],
                    "description": "Job action.",
                },
                "job_id": {"type": "string", "description": "Job ID (status/result/cancel)."},
                "tool_name": {"type": "string", "description": "MCP tool to run (submit)."},
                "args": {"type": "object", "description": "Tool arguments (submit)."},
                "priority": {"type": "integer", "description": "Job priority (submit). Default 0."},
                "timeout_s": {"type": "number", "description": "Max seconds (submit). Default 300."},
                "max_retries": {"type": "integer", "description": "Retry count (submit). Default 0."},
                "status_filter": {"type": "string", "description": "Filter by status (list)."},
                "tool_name_filter": {"type": "string", "description": "Filter by tool (list)."},
                "limit": {"type": "integer", "description": "Max results (list). Default 20."},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool_name": {"type": "string"}, "args": {"type": "object"},
                            "priority": {"type": "integer"}, "timeout_s": {"type": "number"},
                            "max_retries": {"type": "integer"},
                        },
                        "required": ["tool_name", "args"],
                    },
                    "description": "Batch job list (batch).",
                },
            },
            "required": ["action"],
        },
    },
    # ==================================================================
    # 14. research — RSS, deep research, live data
    # ==================================================================
    {
        "name": "research",
        "description": (
            "Research tools: RSS feeds, deep multi-hop research, and live real-time data.\n"
            "Actions:\n"
            "  rss_search — Discover RSS feed URLs for a topic.\n"
            "  rss_push — Fetch RSS feed and store articles in PostgreSQL.\n"
            "  deep — Multi-hop autonomous research with citations.\n"
            "  realtime — Live data (time, weather, stock, crypto, forex). No API keys."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["rss_search", "rss_push", "deep", "realtime"],
                    "description": "Research action.",
                },
                "topic": {"type": "string", "description": "Topic (rss_search/rss_push)."},
                "feed_url": {"type": "string", "description": "RSS feed URL (rss_push)."},
                "question": {"type": "string", "description": "Research question (deep)."},
                "depth": {"type": "integer", "description": "Search rounds 1-3 (deep). Default 2."},
                "max_sources": {"type": "integer", "description": "Sources per round (deep). Default 4."},
                "type": {
                    "type": "string", "enum": ["time", "weather", "stock", "crypto", "forex"],
                    "description": "Data type (realtime).",
                },
                "query": {"type": "string", "description": "Query (realtime). Timezone/city/ticker/coin/pair."},
            },
            "required": ["action"],
        },
    },
    # ==================================================================
    # 15. think — Reasoning scratchpad (no action parameter)
    # ==================================================================
    {
        "name": "think",
        "description": (
            "Scratchpad for explicit step-by-step reasoning. Call BEFORE any irreversible tool. "
            "Records a thought/plan in the trace without side effects. Returns the thought as-is."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "Reasoning, hypothesis, step-by-step plan, or conclusion.",
                },
            },
            "required": ["thought"],
        },
    },
    # ==================================================================
    # 16. system — Tool discovery, instructions, desktop control
    # ==================================================================
    {
        "name": "system",
        "description": (
            "System utilities: tool discovery, instructions, and desktop automation.\n"
            "Actions:\n"
            "  list_categories — List available tools by category. Call when unsure which tool to use.\n"
            "  instructions — Get the recommended system prompt for using this MCP server.\n"
            "  desktop_screenshot — Capture the full X11 virtual desktop (all windows, not just browser).\n"
            "  desktop_control — Control X11 desktop: click, type, press keys, scroll, launch apps."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list_categories", "instructions", "desktop_screenshot", "desktop_control"],
                    "description": "System action.",
                },
                "category": {"type": "string", "description": "Tool category (list_categories)."},
                "search": {"type": "string", "description": "Keyword search (list_categories)."},
                "region": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}, "width": {"type": "integer"}, "height": {"type": "integer"}},
                    "description": "Crop region (desktop_screenshot).",
                },
                "desktop_action": {
                    "type": "string",
                    "enum": ["click", "double_click", "right_click", "type", "key", "move", "scroll", "run"],
                    "description": "Desktop action (desktop_control).",
                },
                "x": {"type": "integer", "description": "X coordinate (desktop_control click/move)."},
                "y": {"type": "integer", "description": "Y coordinate (desktop_control click/move)."},
                "text": {"type": "string", "description": "Text to type or key combo (desktop_control)."},
                "command": {"type": "string", "description": "Shell command (desktop_control run action)."},
                "button": {"type": "integer", "description": "Mouse button 1/2/3 (desktop_control). Default 1."},
                "direction": {"type": "string", "enum": ["up", "down"], "description": "Scroll direction (desktop_control)."},
                "amount": {"type": "integer", "description": "Scroll clicks (desktop_control). Default 3."},
            },
            "required": ["action"],
        },
    },
]


# ---------------------------------------------------------------------------
# plan_task catalog — curated short list of core tools (~25) for the planner prompt.
# Keeping this small (<600 chars) ensures the full system prompt stays under ~1000 tokens,
# which fits any loaded model's context and avoids slow inference with huge prompts.
# ---------------------------------------------------------------------------

_PLAN_TASK_CATALOG: str = """\
Web tools:
  web(action=search)    | params: query, engine, max_chars | search the web
  web(action=fetch)     | params: url, max_chars | fetch page content
  web(action=extract)   | params: url, max_chars | extract clean article
  web(action=wikipedia) | params: query, lang | search Wikipedia

Browser tools:
  browser(action=screenshot)    | params: url, find_text, find_image | capture screenshot
  browser(action=navigate)      | params: url | navigate + return page text
  browser(action=scrape)        | params: url, max_scrolls | full-page scrape

Image tools:
  image(action=search)       | params: query, count | search for images
  image(action=face_detect)  | params: path, style[anime/photo] | detect faces
  image(action=crop)         | params: path, left, top, right, bottom | crop image

Code tools:
  code(action=python)     | params: code, packages | execute Python
  code(action=javascript) | params: code | execute JavaScript
  code(action=jupyter)    | params: code, session_id | persistent Jupyter kernel

Memory and Knowledge tools:
  memory(action=store)  | params: key, value | persist a fact
  memory(action=recall) | params: key, pattern | retrieve stored facts
  vector(action=search) | params: query, top_k | semantic vector search
  knowledge(action=query) | params: id | query the knowledge graph

Documents and Media tools:
  document(action=pdf_read)  | params: path, mode | read PDF
  document(action=ocr)       | params: path, lang | OCR an image
  web(action=youtube)        | params: url | YouTube transcript

LLM Utility tools:
  web(action=summarize)       | params: content, style, max_words | summarize text
  web(action=extract)         | params: content, schema_json | extract structured data

Database tools:
  data(action=search)  | params: topic, q, limit | search stored articles
  data(action=errors)  | params: limit, service | query error log
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text(s: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": s}]


def _json_or_err(r: "httpx.Response", tool: str) -> list[dict[str, Any]]:
    """Return .json() as text, or an error message if the status code is not 2xx."""
    if r.status_code >= 400:
        return _text(f"{tool}: upstream returned {r.status_code} — {r.text[:300]}")
    try:
        return _text(json.dumps(r.json()))
    except Exception:
        return _text(f"{tool}: upstream returned {r.status_code} (non-JSON)")


def _image_blocks(container_path: str, summary: str) -> list[dict[str, Any]]:
    """Return MCP content blocks: text summary + inline base64 image (compressed for LM Studio).
    Delegates to ImageRenderer.encode_path() which enforces the payload size cap."""
    return _renderer.encode_path(container_path, summary)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Async Job Executor — runs tool calls in background tasks
# ---------------------------------------------------------------------------


_job_cancelled: set[str] = set()  # track cancelled job_ids to abort in-flight jobs


async def _execute_job(job_id: str, tool_name: str, tool_args: dict, timeout_s: float) -> None:
    """Background coroutine: execute a job and update its state in aichat-data/jobs."""
    def _iso() -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

    async def _patch(data: dict) -> None:
        try:
            async with httpx.AsyncClient(timeout=10) as hc:
                await hc.patch(f"{JOB_URL}/{job_id}", json=data)
        except Exception:
            pass

    # Mark running
    await _patch({"status": "running", "started_at": _iso()})

    try:
        if job_id in _job_cancelled:
            await _patch({"status": "cancelled", "finished_at": _iso()})
            return
        result_blocks = await asyncio.wait_for(_call_tool(tool_name, tool_args), timeout=timeout_s)
        if job_id in _job_cancelled:
            await _patch({"status": "cancelled", "finished_at": _iso()})
            return
        result_text = next((b["text"] for b in result_blocks if b.get("type") == "text"), "")
        if _blocks_indicate_error(tool_name, result_blocks):
            await _patch({
                "status": "failed",
                "error": result_text[:2000] or f"{tool_name} failed",
                "finished_at": _iso(),
            })
            return
        await _patch({"status": "succeeded", "result": result_text[:100_000],
                      "progress": 100, "finished_at": _iso()})
    except asyncio.TimeoutError:
        await _patch({"status": "failed", "error": f"Timed out after {timeout_s}s",
                      "finished_at": _iso()})
    except Exception as exc:
        await _patch({"status": "failed", "error": str(exc)[:2000], "finished_at": _iso()})


# ---------------------------------------------------------------------------
# Orchestration — WorkflowStep / WorkflowResult / WorkflowExecutor

# ---------------------------------------------------------------------------

@dataclass
class WorkflowStep:
    """A single step in an orchestrated workflow."""
    id: str                              # unique identifier; used in depends_on + {id.result}
    tool: str                            # MCP tool name to call
    args: dict                           # tool arguments (strings may use {id.result} placeholders)
    depends_on: list[str] = field(default_factory=list)  # step IDs that must finish first
    label: str = ""                      # human-readable label for the result report


@dataclass
class WorkflowResult:
    """Result from a single executed workflow step."""
    step_id: str
    label: str
    tool: str
    result: str        # first text block returned by the tool
    ok: bool           # False if the tool raised or returned an error
    duration_ms: int


class WorkflowExecutor:
    """Execute a DAG of WorkflowSteps with automatic parallelism.

    Steps with no pending dependencies are gathered and run concurrently
    via asyncio.gather(). Steps that depend on earlier steps receive their
    results via {step_id.result} interpolation in arg string values.
    """

    def __init__(self, steps: list[WorkflowStep], stop_on_error: bool = False) -> None:
        self._steps = steps
        self._stop_on_error = stop_on_error

    async def run(self) -> list[WorkflowResult]:
        """Execute all steps respecting dependencies; return results in execution order."""
        waves = self._build_waves()
        completed: dict[str, WorkflowResult] = {}
        results: list[WorkflowResult] = []
        for wave in waves:
            if self._stop_on_error and any(not r.ok for r in results):
                break
            wave_results = await asyncio.gather(
                *[self._run_step(s, completed) for s in wave]
            )
            for r in wave_results:
                completed[r.step_id] = r
                results.append(r)
        return results

    async def _run_step(
        self, step: WorkflowStep, completed: dict[str, WorkflowResult]
    ) -> WorkflowResult:
        """Interpolate args, call the tool, return a WorkflowResult."""
        label = step.label or step.id
        t0 = _time.monotonic()
        try:
            interpolated = self._interpolate(step.args, completed)
            content = await _call_tool(step.tool, interpolated)
            text = next((b["text"] for b in content if b.get("type") == "text"), "")
            ok = not (
                text.startswith("Error ")
                or text.startswith("Unknown tool")
                or text.startswith(f"Step '{step.id}' raised")
            )
        except Exception as exc:
            text = f"Step '{step.id}' raised: {exc}"
            ok = False
        ms = int((_time.monotonic() - t0) * 1000)
        return WorkflowResult(step.id, label, step.tool, text, ok, ms)

    def _build_waves(self) -> list[list[WorkflowStep]]:
        """Topological sort via Kahn's algorithm → ordered execution waves.

        Raises ValueError on cycle or unknown dependency reference.
        """
        by_id: dict[str, WorkflowStep] = {}
        for s in self._steps:
            if s.id in by_id:
                raise ValueError(f"duplicate step id: '{s.id}'")
            by_id[s.id] = s

        # Validate all depends_on references
        for s in self._steps:
            for dep in s.depends_on:
                if dep not in by_id:
                    raise ValueError(
                        f"step '{s.id}' depends on unknown step '{dep}'"
                    )

        # Kahn's algorithm
        in_degree: dict[str, int] = {s.id: len(s.depends_on) for s in self._steps}
        queue: list[str] = [sid for sid, d in in_degree.items() if d == 0]
        waves: list[list[WorkflowStep]] = []
        visited = 0

        while queue:
            wave = [by_id[sid] for sid in queue]
            waves.append(wave)
            visited += len(queue)
            next_queue: list[str] = []
            for sid in queue:
                for s in self._steps:
                    if sid in s.depends_on:
                        in_degree[s.id] -= 1
                        if in_degree[s.id] == 0:
                            next_queue.append(s.id)
            queue = next_queue

        if visited != len(self._steps):
            raise ValueError("cycle detected in workflow dependencies")
        return waves

    def _interpolate(self, args: dict, completed: dict[str, WorkflowResult]) -> dict:
        """Recursively replace {step_id.result} in string arg values."""
        out: dict = {}
        for k, v in args.items():
            if isinstance(v, str):
                for step_id, res in completed.items():
                    v = v.replace(f"{{{step_id}.result}}", res.result)
                out[k] = v
            elif isinstance(v, dict):
                out[k] = self._interpolate(v, completed)
            else:
                out[k] = v
        return out

    @staticmethod
    def _format_report(results: list[WorkflowResult]) -> str:
        """Format a human-readable multi-line report of all step results."""
        lines = ["## Workflow Results\n"]
        for i, r in enumerate(results, 1):
            status = "OK" if r.ok else "FAILED"
            preview = r.result[:500] + ("..." if len(r.result) > 500 else "")
            lines.append(
                f"### Step {i}: {r.label} [{status}] ({r.duration_ms} ms)\n"
                f"Tool: `{r.tool}`\n\n"
                f"{preview}\n"
            )
        total_ms = sum(r.duration_ms for r in results)
        ok_count = sum(1 for r in results if r.ok)
        lines.append(
            f"---\n**{ok_count}/{len(results)} steps succeeded** | "
            f"total wall-time: {total_ms} ms"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Image manipulation helpers (PIL-based, no external OCR required)
# ---------------------------------------------------------------------------

def _resolve_image_path(path: str) -> str | None:
    """
    Accept any of:
      - bare filename            →  BROWSER_WORKSPACE/<filename>
      - /workspace/<filename>    →  BROWSER_WORKSPACE/<filename>  (human_browser container path)
      - /docker/human_browser/workspace/<filename>  →  BROWSER_WORKSPACE/<filename>
      - any other absolute path  →  used as-is if it exists
    Returns a readable local path or None if not found.
    """
    if not path:
        return None
    name = os.path.basename(path)
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}

    def _workspace_images() -> list[tuple[float, str]]:
        if not os.path.isdir(BROWSER_WORKSPACE):
            return []
        picks: list[tuple[float, str]] = []
        for fn in os.listdir(BROWSER_WORKSPACE):
            full = os.path.join(BROWSER_WORKSPACE, fn)
            if not os.path.isfile(full):
                continue
            ext = os.path.splitext(fn)[1].lower()
            if ext not in image_exts:
                continue
            try:
                picks.append((os.path.getmtime(full), full))
            except OSError:
                continue
        picks.sort(key=lambda p: p[0], reverse=True)
        return picks

    def _resolve_client_alias() -> str | None:
        # Some chat clients send synthetic attachment names (for example,
        # "image-1772402117277.jpg") that are not real workspace filenames.
        # In that case, use the most recent real image in the workspace.
        if not re.fullmatch(r"image-\d{8,}\.(?:png|jpe?g|webp|gif|bmp|tiff?)", name, flags=re.IGNORECASE):
            return None
        picks = _workspace_images()
        return picks[0][1] if picks else None

    def _resolve_workspace_best_effort(prefer_latest: bool = False) -> str | None:
        """Resolve missing workspace path to a likely file without masking true missing bare names."""
        picks = _workspace_images()
        if not picks:
            return None
        by_name = {os.path.basename(p).lower(): p for _, p in picks}
        lname = name.lower()
        exact = by_name.get(lname)
        if exact:
            return exact
        stem = os.path.splitext(lname)[0]
        if stem:
            stem_matches = [
                p for _, p in picks
                if os.path.splitext(os.path.basename(p).lower())[0].startswith(stem)
            ]
            if stem_matches:
                return stem_matches[0]
        if prefer_latest:
            return picks[0][1]
        return None

    # Bare filename or known prefix → remap to our bind-mount
    if "/" not in path or path.startswith("/workspace/") or path.startswith("/docker/human_browser/workspace/"):
        candidate = os.path.join(BROWSER_WORKSPACE, name)
        if os.path.isfile(candidate):
            return candidate
        # Prefix paths often point to browser-container filenames that may differ
        # from what was actually persisted; pick the closest workspace match.
        if path.startswith("/workspace/") or path.startswith("/docker/human_browser/workspace/"):
            return _resolve_workspace_best_effort(prefer_latest=True) or _resolve_client_alias()
        # Bare names keep strict behavior so obvious missing files still error.
        return _resolve_workspace_best_effort(prefer_latest=False) or _resolve_client_alias()
    if os.path.isfile(path):
        return path
    return _resolve_client_alias()


def _pil_to_blocks(
    img: "_PilImage.Image",
    summary: str,
    quality: int = 85,
    save_prefix: str | None = None,
) -> list[dict[str, Any]]:
    """Encode a PIL Image as an inline JPEG MCP block (delegates to ImageRenderer)."""
    return _renderer.encode(img, summary, save_prefix=save_prefix, quality=quality)


_ANIME_CASCADE_PATH = "/app/lbpcascade_animeface.xml"


def _run_cascades(
    gray: Any,
    cascade_paths: list[str],
    sf: float,
    mn: int,
    msz: int,
    seen: set[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Run a list of cascade classifiers and return deduplicated (x,y,w,h) boxes."""
    parsed: list[tuple[int, int, int, int]] = []
    for cascade_path in cascade_paths:
        if not os.path.exists(cascade_path):
            continue
        detector = _cv2.CascadeClassifier(cascade_path)  # type: ignore[union-attr]
        if detector.empty():
            continue
        raw = detector.detectMultiScale(gray, scaleFactor=sf, minNeighbors=mn, minSize=(msz, msz))
        for (x, y, w, h) in list(raw):  # type: ignore[assignment]
            box = (int(x), int(y), int(w), int(h))
            cx, cy = box[0] + box[2] // 2, box[1] + box[3] // 2
            dup = False
            for (ex, ey, ew, eh) in seen:
                if abs(cx - (ex + ew // 2)) < ew * 0.3 and abs(cy - (ey + eh // 2)) < eh * 0.3:
                    dup = True
                    break
            if not dup:
                seen.add(box)
                parsed.append(box)
    return parsed


def _detect_faces(
    img: "_PilImage.Image",
    *,
    min_face_size: int = 40,
    min_neighbors: int = 5,
    scale_factor: float = 1.1,
    anime: bool = False,
) -> list[tuple[int, int, int, int]]:
    """Detect faces via OpenCV cascades and return sorted (x, y, w, h) boxes.

    When *anime* is True the dedicated ``lbpcascade_animeface`` cascade
    (trained on anime/manga/illustration faces by nagadomi) is used as the
    primary detector.  Standard Haar cascades serve as fallback, and an
    ultra-permissive retry pass runs automatically if the first pass finds
    nothing.

    For real photos (anime=False), the standard Haar cascades run first.
    If nothing is found a second, more permissive pass is attempted.
    """
    if not _HAS_CV2 or _cv2 is None or _np is None:
        return []

    if anime:
        scale_factor  = min(scale_factor, 1.05)
        min_neighbors = min(min_neighbors, 2)
        min_face_size = min(min_face_size, 20)

    arr = _np.array(img.convert("RGB"))
    gray: Any
    try:
        if bool(_CV2_ACCEL_STATUS.get("opencl_use", False)):
            umat = _cv2.UMat(arr)  # type: ignore[union-attr]
            gray_u = _cv2.cvtColor(umat, _cv2.COLOR_RGB2GRAY)  # type: ignore[union-attr]
            gray_u = _cv2.equalizeHist(gray_u)  # type: ignore[union-attr]
            gray = gray_u.get()  # type: ignore[assignment]
        else:
            gray = _cv2.cvtColor(arr, _cv2.COLOR_RGB2GRAY)  # type: ignore[union-attr]
            gray = _cv2.equalizeHist(gray)  # type: ignore[union-attr]
    except Exception:
        gray = _cv2.cvtColor(arr, _cv2.COLOR_RGB2GRAY)  # type: ignore[union-attr]
        gray = _cv2.equalizeHist(gray)  # type: ignore[union-attr]

    haarcascades_dir = getattr(_cv2.data, "haarcascades", "")
    lbpcascades_dir  = getattr(_cv2.data, "lbpcascades", "")

    # -- build cascade list by mode ------------------------------------
    if anime:
        # Anime-specific cascade first (trained on anime/manga faces),
        # then standard cascades + profile cascades as backup.
        cascade_paths: list[str] = []
        if os.path.exists(_ANIME_CASCADE_PATH):
            cascade_paths.append(_ANIME_CASCADE_PATH)
        cascade_paths.extend([
            os.path.join(haarcascades_dir, "haarcascade_frontalface_default.xml"),
            os.path.join(haarcascades_dir, "haarcascade_frontalface_alt2.xml"),
            os.path.join(lbpcascades_dir, "lbpcascade_profileface.xml"),
            os.path.join(haarcascades_dir, "haarcascade_profileface.xml"),
        ])
    else:
        cascade_paths = [
            os.path.join(haarcascades_dir, "haarcascade_frontalface_default.xml"),
            os.path.join(haarcascades_dir, "haarcascade_frontalface_alt2.xml"),
        ]

    sf  = max(1.01, min(scale_factor, 1.5))
    mn  = max(1, min(min_neighbors, 12))
    msz = max(12, min_face_size)

    seen: set[tuple[int, int, int, int]] = set()
    parsed = _run_cascades(gray, cascade_paths, sf, mn, msz, seen)

    # -- retry with ultra-permissive params if first pass found nothing -
    if not parsed:
        retry_sf  = 1.02 if anime else 1.05
        retry_mn  = 1
        retry_msz = 12 if anime else 20
        parsed = _run_cascades(gray, cascade_paths, retry_sf, retry_mn, retry_msz, seen)

    # -- anime: also try on un-equalized grayscale (some art has flat
    #    histograms that equalizeHist distorts, hurting cascade detection) -
    if anime and not parsed:
        try:
            gray_raw = _cv2.cvtColor(arr, _cv2.COLOR_RGB2GRAY)  # type: ignore[union-attr]
            parsed = _run_cascades(gray_raw, cascade_paths, 1.02, 1, 12, seen)
        except Exception:
            pass

    parsed.sort(key=lambda box: box[2] * box[3], reverse=True)
    return parsed


async def _detect_faces_yolo_fallback(
    img: "_PilImage.Image",
) -> list[tuple[int, int, int, int]]:
    """Fallback face detection via YOLO person detection from the vision service.

    Calls aichat-vision /detect/humans, then estimates face regions as the
    upper portion of each detected person bounding box.  This catches faces
    that cascade classifiers miss entirely (heavily stylized art, unusual
    angles, non-standard proportions).
    """
    if _np is None:
        return []
    buf = _io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "http://aichat-vision:8099/detect/humans",
                json={"image_base64": b64, "confidence": 0.25},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        return []

    faces: list[tuple[int, int, int, int]] = []
    for person in data.get("people", []):
        bbox = person.get("bbox", {})
        x1, y1 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0))
        x2, y2 = int(bbox.get("x2", 0)), int(bbox.get("y2", 0))
        pw, ph = x2 - x1, y2 - y1
        if pw < 20 or ph < 30:
            continue
        # Estimate face as upper ~25% of person bbox, narrowed to ~85% width
        face_h = int(ph * 0.25)
        face_w = min(pw, int(face_h * 0.85))
        face_x = x1 + (pw - face_w) // 2
        face_y = y1
        faces.append((face_x, face_y, face_w, face_h))
    return faces


def _face_embedding(
    img: "_PilImage.Image",
    box: tuple[int, int, int, int],
) -> "Any | None":
    """Compute a lightweight normalized face vector for best-effort matching."""
    if not _HAS_CV2 or _cv2 is None or _np is None:
        return None
    x, y, w, h = box
    arr = _np.array(img.convert("RGB"))
    gray = _cv2.cvtColor(arr, _cv2.COLOR_RGB2GRAY)  # type: ignore[union-attr]
    y1, y2 = max(0, y), min(gray.shape[0], y + h)
    x1, x2 = max(0, x), min(gray.shape[1], x + w)
    face = gray[y1:y2, x1:x2]
    if face.size == 0:
        return None
    try:
        if bool(_CV2_ACCEL_STATUS.get("opencl_use", False)):
            face_u = _cv2.UMat(face)  # type: ignore[union-attr]
            face = _cv2.resize(face_u, (64, 64), interpolation=_cv2.INTER_AREA).get()  # type: ignore[union-attr]
        else:
            face = _cv2.resize(face, (64, 64), interpolation=_cv2.INTER_AREA)  # type: ignore[union-attr]
    except Exception:
        face = _cv2.resize(face, (64, 64), interpolation=_cv2.INTER_AREA)  # type: ignore[union-attr]
    vec = face.astype("float32").reshape(-1)  # type: ignore[union-attr]
    vec -= float(vec.mean())
    norm = float(_np.linalg.norm(vec))
    if norm <= 1e-9:
        return None
    return vec / norm


def _face_similarity(vec_a: Any, vec_b: Any) -> float:
    """Cosine similarity mapped to [0.0, 1.0]."""
    if _np is None:
        return 0.0
    raw = float(_np.dot(vec_a, vec_b))
    return float(_np.clip((raw + 1.0) / 2.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# ImageRenderer — OOP image encoding pipeline for LM Studio inline rendering
# ---------------------------------------------------------------------------

# LM Studio silently drops (shows "external image") when the MCP payload exceeds
# its internal message size limit.  All encoding paths go through ImageRenderer,
# which enforces a hard byte-count cap by stepping down JPEG quality.
_MAX_INLINE_BYTES: int = 3_000_000   # 3 MB raw  ≈  4 MB base64 — safe for LM Studio


class ImageRenderer:
    """
    OOP wrapper that encodes PIL images, workspace paths, and raw HTTP bytes
    as inline MCP image content blocks, always honouring LM Studio's payload cap.

    Usage
    -----
    renderer = ImageRenderer()
    blocks = renderer.encode(pil_img, "Screenshot of …")
    blocks = renderer.encode_path("/workspace/shot.png", "Summary text")
    blocks = renderer.encode_url_bytes(raw_bytes, "image/jpeg", "Fetched from …")
    """

    MAX_BYTES: int = _MAX_INLINE_BYTES
    # Quality ladder: try highest first, step down until payload fits.
    _QUALITY_LADDER: tuple[int, ...] = (85, 75, 65, 50)

    # ── private helpers ──────────────────────────────────────────────────────

    def _compress_to_limit(self, img: "_PilImage.Image", min_quality: int = 85) -> bytes:
        """JPEG-compress img, reducing quality until payload < MAX_BYTES.

        Starts from min_quality (caller's preference) and steps down through
        standard rungs until the payload fits within MAX_BYTES.
        """
        # Build a descending ladder starting at the caller's quality preference
        _RUNGS = (75, 65, 50)
        ladder = [min_quality] + [q for q in _RUNGS if q < min_quality]
        for q in ladder:
            buf = _io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=q)
            raw = buf.getvalue()
            if len(raw) <= self.MAX_BYTES:
                return raw
        # Absolute last resort: quality=40
        buf = _io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=40)
        return buf.getvalue()

    def _fit(
        self,
        img: "_PilImage.Image",
        max_w: int = 1280,
        max_h: int = 1024,
    ) -> "_PilImage.Image":
        """Downscale img to fit within max_w × max_h, preserving aspect ratio.
        When max_w/max_h are raised (e.g. for upscale output), the image passes
        through unchanged unless it exceeds those larger bounds."""
        if img.width > max_w or img.height > max_h:
            img = img.copy()
            img.thumbnail((max_w, max_h), _PilImage.LANCZOS)
        return img

    # ── public API ───────────────────────────────────────────────────────────

    def encode(
        self,
        img: "_PilImage.Image",
        summary: str,
        save_prefix: str | None = None,
        quality: int = 85,
        max_w: int = 1280,
        max_h: int = 1024,
    ) -> list[dict[str, Any]]:
        """
        Encode a PIL Image → [text_block, image_block], guaranteed to fit
        within MAX_BYTES.  Optionally saves the compressed JPEG to BROWSER_WORKSPACE.
        quality is the preferred JPEG quality (85–95 recommended); the encoder
        steps down automatically if the payload would exceed MAX_BYTES.
        max_w / max_h control the pixel-dimension cap before compression
        (default 1280×1024 for chat; raise to 4096×4096 for upscale output).
        """
        img = self._fit(img.convert("RGB"), max_w=max_w, max_h=max_h)
        raw = self._compress_to_limit(img, min_quality=max(40, min(quality, 95)))
        if save_prefix and os.path.isdir(BROWSER_WORKSPACE):
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"{save_prefix}_{ts}.jpg"
                with open(os.path.join(BROWSER_WORKSPACE, fname), "wb") as fh:
                    fh.write(raw)
                summary += f"\n→ Saved as: {fname}  (pass this as 'path' in the next pipeline step)"
            except OSError:
                pass  # workspace may be read-only; inline image is still returned
        b64 = base64.standard_b64encode(raw).decode("ascii")
        return [
            {"type": "text",  "text": summary},
            {"type": "image", "data": b64, "mimeType": "image/jpeg"},
        ]

    def encode_path(self, container_path: str, summary: str) -> list[dict[str, Any]]:
        """
        Load an image from BROWSER_WORKSPACE (given a container path or bare filename)
        and return inline MCP blocks.  Returns text-only if the file is missing.
        Replaces the old _image_blocks() function.
        """
        blocks: list[dict[str, Any]] = [{"type": "text", "text": summary}]
        if not container_path:
            return blocks
        fname = os.path.basename(container_path)
        local_path = os.path.join(BROWSER_WORKSPACE, fname)
        if not os.path.isfile(local_path):
            return blocks
        try:
            if _HAS_PIL:
                with _PilImage.open(local_path) as img:
                    # Reuse encode() but skip the text block (already in blocks[0])
                    encoded = self.encode(img, "")
                    blocks.extend(b for b in encoded if b.get("type") == "image")
            else:
                with open(local_path, "rb") as fh:
                    raw = fh.read()
                # Cap payload: LM Studio silently drops images that exceed the limit.
                if len(raw) > self.MAX_BYTES:
                    raw = raw[: self.MAX_BYTES]  # truncate as last resort
                b64 = base64.standard_b64encode(raw).decode("ascii")
                blocks.append({"type": "image", "data": b64, "mimeType": "image/png"})
        except Exception:
            pass
        return blocks

    def encode_url_bytes(
        self,
        raw: bytes,
        content_type: str,
        summary: str,
    ) -> list[dict[str, Any]]:
        """
        Compress raw HTTP image bytes → inline MCP blocks.
        Used by fetch_image so that large external images are always compressed
        before being sent to LM Studio (fixes the "external image" display bug).
        Falls back to raw if PIL is unavailable and the payload is small enough.
        """
        if _HAS_PIL:
            try:
                with _PilImage.open(_io.BytesIO(raw)) as img:
                    return self.encode(_ImageOps.exif_transpose(img).convert("RGB"), summary)
            except Exception:
                pass  # corrupt / unrecognised format — try raw fallback below
        # PIL unavailable or image unreadable — send raw only if it fits
        if len(raw) <= self.MAX_BYTES:
            b64 = base64.standard_b64encode(raw).decode("ascii")
            return [
                {"type": "text",  "text": summary},
                {"type": "image", "data": b64, "mimeType": content_type},
            ]
        return [{"type": "text",
                 "text": summary + "\n⚠ Image too large to render inline (PIL unavailable)."}]


_renderer = ImageRenderer()   # module-level singleton used by all tool handlers


# ---------------------------------------------------------------------------
# ImageRenderingPolicy — guarantees every image tool produces an image block
# ---------------------------------------------------------------------------

class ImageRenderingPolicy:
    """Guarantees every image-returning MCP tool produces at least one image block.

    LM Studio silently shows 'external image' when a tool response contains only
    text blocks where an inline image was expected.  Call ``enforce()`` on any
    content-block list from an image-producing tool to guarantee rendering.

    Usage
    -----
    content = await _call_tool(name, args)
    if ImageRenderingPolicy.is_image_tool(name):
        content = ImageRenderingPolicy.enforce(content)

    Three-step escalation inside enforce():
    1. Image block already present  → return unchanged.
    2. ``fallback_bytes`` provided  → compress via _renderer and embed.
    3. Still missing                → append a small dark JPEG placeholder so
                                      LM Studio always renders something visual.
    """

    # Tools that MUST always return at least one inline image block when called
    # from LM Studio (i.e. via _handle_rpc → tools/call).
    IMAGE_TOOLS: frozenset[str] = frozenset({
        # Mega-tool names (resolved before this check)
        "screenshot", "fetch_image",
        "image_generate", "image_edit", "image_remix",
        "image_crop", "image_zoom", "image_scan", "image_enhance",
        "image_stitch", "image_diff", "image_annotate", "image_upscale",
        "bulk_screenshot", "scroll_screenshot", "screenshot_search",
        "browser_save_images", "browser_download_page_images",
        "image_search", "db_list_images", "face_recognize",
        # Mega-tool entry points (before resolution)
        "image", "browser",
    })

    @staticmethod
    def has_image(blocks: list[dict[str, Any]]) -> bool:
        """Return True if any block in blocks is an image block."""
        return any(b.get("type") == "image" for b in blocks)

    @classmethod
    def is_image_tool(cls, name: str) -> bool:
        """Return True if this tool is expected to always return an image block."""
        return name in cls.IMAGE_TOOLS

    @classmethod
    def enforce(
        cls,
        blocks: list[dict[str, Any]],
        fallback_bytes: bytes = b"",
        content_type: str = "image/jpeg",
    ) -> list[dict[str, Any]]:
        """Ensure ``blocks`` contains at least one image block.

        Parameters
        ----------
        blocks:
            Content blocks returned by a tool handler.
        fallback_bytes:
            Optional raw image bytes (e.g. downloaded from the web) to use
            as a last-chance source if no image block is present.
        content_type:
            MIME type of ``fallback_bytes`` (default ``image/jpeg``).
        """
        if cls.has_image(blocks):
            return blocks

        # Step 2 — try to produce an image from raw bytes via _renderer
        if fallback_bytes:
            summary = cls._first_text(blocks)
            extra = _renderer.encode_url_bytes(fallback_bytes, content_type, summary)
            if cls.has_image(extra):
                text_blocks = [b for b in blocks if b.get("type") == "text"]
                img_blocks  = [b for b in extra  if b.get("type") == "image"]
                return text_blocks + img_blocks

        # Step 3 — placeholder image so LM Studio never shows 'external image'
        ph = cls._placeholder()
        if ph is not None:
            return blocks + [ph]
        return blocks  # PIL unavailable; best-effort text-only

    @staticmethod
    def _first_text(blocks: list[dict[str, Any]]) -> str:
        """Return text from the first text block, or empty string."""
        return next((b["text"] for b in blocks if b.get("type") == "text"), "")

    @staticmethod
    def _placeholder() -> dict[str, str] | None:
        """Generate a minimal neutral JPEG indicating image was unavailable.

        Returns None if PIL is not available (caller handles gracefully).
        """
        if not _HAS_PIL:
            return None
        buf = _io.BytesIO()
        img = _PilImage.new("RGB", (460, 110), color=(236, 236, 236))
        try:
            draw = _ImageDraw.Draw(img)
            draw.text((14, 44), "Image preview unavailable", fill=(48, 48, 48))
        except Exception:
            pass
        img.save(buf, "JPEG", quality=60)
        return {
            "type":     "image",
            "data":     base64.standard_b64encode(buf.getvalue()).decode("ascii"),
            "mimeType": "image/jpeg",
        }


# ---------------------------------------------------------------------------
# GpuDetector / GpuUpscaler — GPU-accelerated upscaling via LM Studio
# ---------------------------------------------------------------------------

class GpuDetector:
    """
    Detect available GPU hardware (NVIDIA and Intel Arc) without requiring
    PyTorch or other heavy frameworks — uses lightweight OS-level probes only.
    Results are cached after the first call.
    Only used by GpuUpscaler (image_upscale tool); not wired to other tools.
    """

    _cache: dict[str, str] | None = None   # {"vendor": "nvidia"|"intel"|"none", "name": str}

    @classmethod
    def detect(cls) -> dict[str, str]:
        """Return {"vendor": "nvidia"|"intel"|"none", "name": <human readable>}."""
        if cls._cache is not None:
            return cls._cache
        cls._cache = cls._probe()
        return cls._cache

    @classmethod
    def _probe(cls) -> dict[str, str]:
        import subprocess as _sp

        # ── NVIDIA: check nvidia-smi or /dev/nvidia0 ──────────────────────
        try:
            out = _sp.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                timeout=4, stderr=_sp.DEVNULL,
            ).decode().strip().splitlines()
            if out:
                return {"vendor": "nvidia", "name": out[0]}
        except Exception:
            pass
        if os.path.exists("/dev/nvidia0"):
            return {"vendor": "nvidia", "name": "NVIDIA (device node)"}

        # ── Intel Arc / Intel GPU: check /dev/dri or vainfo ───────────────
        try:
            # vainfo reports the VA-API driver; Intel iHD = Arc/Iris/UHD
            out = _sp.check_output(
                ["vainfo", "--display", "drm"],
                timeout=4, stderr=_sp.DEVNULL,
            ).decode()
            if "iHD" in out or "Intel" in out or "i965" in out:
                # Extract driver name line if present
                for ln in out.splitlines():
                    if "Driver version" in ln or "vainfo: Driver" in ln:
                        return {"vendor": "intel", "name": ln.strip()}
                return {"vendor": "intel", "name": "Intel GPU (vainfo)"}
        except Exception:
            pass
        # Fallback: render node exists → likely Intel integrated or Arc
        try:
            dri = os.listdir("/dev/dri")
            renders = [d for d in dri if d.startswith("renderD")]
            if renders:
                return {"vendor": "intel", "name": f"Intel GPU (/dev/dri/{renders[0]})"}
        except Exception:
            pass

        # ── Check env var override (useful in Docker with GPU passthrough) ─
        if os.environ.get("NVIDIA_VISIBLE_DEVICES", "").strip() not in ("", "void"):
            return {"vendor": "nvidia", "name": "NVIDIA (env NVIDIA_VISIBLE_DEVICES)"}
        if os.environ.get("INTEL_GPU", "").strip() == "1":
            return {"vendor": "intel", "name": "Intel GPU (env INTEL_GPU=1)"}

        return {"vendor": "none", "name": "No GPU detected"}

    @classmethod
    def available(cls) -> bool:
        return cls.detect()["vendor"] != "none"

    @classmethod
    def vendor(cls) -> str:
        return cls.detect()["vendor"]

    @classmethod
    def name(cls) -> str:
        return cls.detect()["name"]


class GpuUpscaler:
    """
    AI-powered image upscaling via LM Studio's /v1/images/edits endpoint.
    Supports both NVIDIA and Intel Arc GPUs (LM Studio handles the acceleration).
    Only used by the image_upscale tool.

    Falls back gracefully: if no GPU is detected or the LM Studio call fails,
    returns None so the caller can fall through to LANCZOS upscaling.
    """

    _PROMPT = "upscale to high resolution, sharpen fine details, enhance clarity"
    _STRENGTH = "0.35"   # low strength = preserve structure, enhance quality

    def __init__(self, base_url: str, model: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        self.model    = model

    async def upscale(
        self,
        img: "_PilImage.Image",
        client: "httpx.AsyncClient",
    ) -> "_PilImage.Image | None":
        """
        Send img to LM Studio /v1/images/edits for AI upscaling.
        Returns the upscaled PIL Image on success, None on any failure.
        """
        if not _HAS_PIL:
            return None
        # Encode input image as JPEG for the multipart upload
        in_buf = _io.BytesIO()
        img.convert("RGB").save(in_buf, format="JPEG", quality=92)
        in_buf.seek(0)

        form: dict[str, str] = {
            "prompt": self._PROMPT,
            "n": "1",
            "response_format": "b64_json",
            "strength": self._STRENGTH,
        }
        if self.model:
            form["model"] = self.model

        try:
            resp = await asyncio.wait_for(
                client.post(
                    f"{self.base_url}/v1/images/edits",
                    files={"image": ("input.jpg", in_buf, "image/jpeg")},
                    data=form,
                    timeout=90.0,
                ),
                timeout=95.0,
            )
            if resp.status_code != 200:
                return None
            rdata = resp.json().get("data") or []
            if not rdata or not rdata[0].get("b64_json"):
                return None
            b64_out = rdata[0]["b64_json"]
            return _PilImage.open(_io.BytesIO(base64.b64decode(b64_out))).convert("RGB")
        except Exception:
            return None

    def gpu_label(self) -> str:
        """Human-readable GPU label for the summary text."""
        info = GpuDetector.detect()
        return info["name"]


# ---------------------------------------------------------------------------
# GpuImageProcessor — OpenCV/CUDA-accelerated image ops, PIL fallback
# ---------------------------------------------------------------------------

class GpuImageProcessor:
    """
    GPU-accelerated image operations for the aichat image pipeline tools.

    Uses OpenCV (cv2) with CUDA support when available; otherwise falls back
    transparently to PIL/Pillow.  All public methods accept and return PIL
    Images so existing tool code needs minimal changes.

    Call GpuImageProcessor.backend() to see which engine is active.
    """

    # Class-level engine detection (set once at class definition time)
    _CUDA_OK: bool = bool(_CV2_ACCEL_STATUS.get("cuda_devices", 0) > 0)
    _OPENCL_OK: bool = bool(_CV2_ACCEL_STATUS.get("opencl_use", False))
    _CV2_OK:  bool = _HAS_CV2

    # ── internal conversion helpers ─────────────────────────────────────────

    @staticmethod
    def _to_np(img: "_PilImage.Image") -> "_np.ndarray":
        """PIL Image → BGR numpy array (OpenCV native format)."""
        rgb = _np.array(img.convert("RGB"))
        return _cv2.cvtColor(rgb, _cv2.COLOR_RGB2BGR)  # type: ignore[union-attr]

    @staticmethod
    def _to_pil(arr: "_np.ndarray") -> "_PilImage.Image":
        """BGR numpy array → PIL Image."""
        rgb = _cv2.cvtColor(arr, _cv2.COLOR_BGR2RGB)  # type: ignore[union-attr]
        return _PilImage.fromarray(rgb)

    @classmethod
    def _to_cv_input(cls, img: "_PilImage.Image") -> "Any":
        """Return ndarray or UMat input depending on active OpenCL mode."""
        arr = cls._to_np(img)
        if cls._OPENCL_OK and _cv2 is not None:
            try:
                return _cv2.UMat(arr)  # type: ignore[union-attr]
            except Exception:
                pass
        return arr

    @classmethod
    def _materialize(cls, arr: "Any") -> "_np.ndarray":
        """Materialize OpenCV output to ndarray (handles UMat safely)."""
        if _cv2 is not None:
            try:
                if isinstance(arr, _cv2.UMat):  # type: ignore[union-attr]
                    return arr.get()  # type: ignore[no-any-return]
            except Exception:
                pass
        return arr

    # ── public API ───────────────────────────────────────────────────────────

    @classmethod
    def backend(cls) -> str:
        """Return the active image processing engine name."""
        if cls._CUDA_OK:
            return "opencv-cuda"
        if cls._OPENCL_OK:
            return "opencv-opencl"
        if cls._CV2_OK:
            return "opencv-cpu"
        return "pillow"

    @classmethod
    def resize(
        cls,
        img: "_PilImage.Image",
        w: int,
        h: int,
    ) -> "_PilImage.Image":
        """High-quality resize to w×h.  LANCZOS4 via cv2, LANCZOS via PIL."""
        if cls._CV2_OK:
            src = cls._to_cv_input(img)
            resized = _cv2.resize(src, (w, h), interpolation=_cv2.INTER_LANCZOS4)  # type: ignore[union-attr]
            return cls._to_pil(cls._materialize(resized))
        return img.resize((w, h), _PilImage.LANCZOS)

    @classmethod
    def sharpen(
        cls,
        img: "_PilImage.Image",
        radius: float = 0.5,
        percent: int = 80,
        threshold: int = 2,
    ) -> "_PilImage.Image":
        """Unsharp-mask sharpening.  cv2 GaussianBlur kernel, PIL UnsharpMask fallback."""
        if cls._CV2_OK:
            arr = cls._to_cv_input(img)
            blur = _cv2.GaussianBlur(arr, (0, 0), max(0.1, radius * 4))  # type: ignore[union-attr]
            amount = percent / 100.0
            sharpened = _cv2.addWeighted(arr, 1.0 + amount, blur, -amount, 0)  # type: ignore[union-attr]
            return cls._to_pil(cls._materialize(sharpened))
        return img.filter(_ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))

    @classmethod
    def enhance_contrast(
        cls,
        img: "_PilImage.Image",
        factor: float = 2.5,
    ) -> "_PilImage.Image":
        """Contrast enhancement.  cv2 convertScaleAbs, PIL ImageEnhance fallback."""
        if cls._CV2_OK:
            arr = cls._to_cv_input(img)
            # Linear contrast stretch: new = clip(alpha*old + beta)
            alpha = max(0.5, min(factor, 5.0))
            enhanced = _cv2.convertScaleAbs(arr, alpha=alpha, beta=0)  # type: ignore[union-attr]
            return cls._to_pil(cls._materialize(enhanced))
        return _ImageEnhance.Contrast(img).enhance(factor)

    @classmethod
    def enhance_sharpness(
        cls,
        img: "_PilImage.Image",
        factor: float = 1.4,
    ) -> "_PilImage.Image":
        """Sharpness enhancement.  cv2 Laplacian kernel, PIL ImageEnhance fallback."""
        if cls._CV2_OK:
            arr = cls._to_cv_input(img)
            # Scale kernel weight by sharpness factor
            k = max(0.0, factor - 1.0)
            kernel = _np.array([[0, -k, 0], [-k, 1 + 4 * k, -k], [0, -k, 0]], dtype=_np.float32)  # type: ignore[union-attr]
            sharpened = _cv2.filter2D(arr, -1, kernel)  # type: ignore[union-attr]
            return cls._to_pil(cls._materialize(sharpened))
        return _ImageEnhance.Sharpness(img).enhance(factor)

    @classmethod
    def diff(
        cls,
        img1: "_PilImage.Image",
        img2: "_PilImage.Image",
    ) -> "_PilImage.Image":
        """Pixel-wise absolute difference.  cv2.absdiff, PIL ImageChops fallback."""
        if cls._CV2_OK:
            a1 = cls._to_np(img1)
            a2 = cls._to_np(img2)
            # Resize img2 to match img1 if sizes differ
            if a1.shape != a2.shape:
                a2 = _cv2.resize(a2, (a1.shape[1], a1.shape[0]), interpolation=_cv2.INTER_LANCZOS4)  # type: ignore[union-attr]
            d = _cv2.absdiff(a1, a2)  # type: ignore[union-attr]
            return cls._to_pil(d)
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, _PilImage.LANCZOS)
        return _ImageChops.difference(img1, img2)

    @classmethod
    def to_grayscale(cls, img: "_PilImage.Image") -> "_PilImage.Image":
        """Convert to grayscale.  cv2.cvtColor or PIL convert('L')."""
        if cls._CV2_OK:
            arr = cls._to_cv_input(img)
            gray = _cv2.cvtColor(arr, _cv2.COLOR_BGR2GRAY)  # type: ignore[union-attr]
            return _PilImage.fromarray(cls._materialize(gray))
        return img.convert("L")

    @classmethod
    def annotate(
        cls,
        img: "_PilImage.Image",
        boxes: "list[tuple[int,int,int,int]]",
        labels: "list[str]",
        color: "tuple[int,int,int]" = (255, 80, 0),
        thickness: int = 3,
    ) -> "_PilImage.Image":
        """Draw bounding boxes + labels.  cv2.rectangle/putText, PIL ImageDraw fallback."""
        if cls._CV2_OK:
            arr = cls._to_np(img)
            bgr = (color[2], color[1], color[0])  # RGB → BGR
            for (x1, y1, x2, y2), label in zip(boxes, labels):
                _cv2.rectangle(arr, (x1, y1), (x2, y2), bgr, thickness)  # type: ignore[union-attr]
                if label:
                    _cv2.putText(  # type: ignore[union-attr]
                        arr, label, (x1, max(y1 - 6, 0)),
                        _cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2, _cv2.LINE_AA,  # type: ignore[union-attr]
                    )
            return cls._to_pil(arr)
        # PIL fallback
        draw = _ImageDraw.Draw(img)
        for (x1, y1, x2, y2), label in zip(boxes, labels):
            draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
            if label:
                draw.text((x1, max(y1 - 14, 0)), label, fill=color)
        return img


_GpuImg = GpuImageProcessor()   # module-level singleton


# ---------------------------------------------------------------------------
# ModelRegistry — /v1/models probe with TTL cache
# ---------------------------------------------------------------------------

class ModelRegistry:
    """
    Probes LM Studio's /v1/models endpoint (30 s TTL) so tools can make
    adaptive decisions: skip GPU calls when no model is loaded, surface
    helpful errors instead of 90 s timeouts, route to the right model.

    Also probes /api/v0/models to track which models are currently loaded
    in GPU memory, enabling capacity-aware routing that avoids triggering
    costly model evictions when LM Studio is at its loaded-model limit.

    Usage: await ModelRegistry.get().is_available(client)
    """

    _TTL:        float = 30.0
    _MAX_LOADED: int   = int(os.environ.get("LM_STUDIO_MAX_LOADED", "2"))
    _instance: "ModelRegistry | None" = None

    def __init__(self) -> None:
        self._models:      list[dict[str, Any]] = []
        self._loaded_ids:  list[str]            = []
        self._last_probe:  float = 0.0
        self._probe_ok:    bool  = False

    @classmethod
    def get(cls) -> "ModelRegistry":
        """Return the process-level singleton."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def invalidate(self) -> None:
        """Force a fresh probe on the next access."""
        self._last_probe = 0.0

    async def _refresh(self, client: httpx.AsyncClient) -> None:
        """Probe /v1/models and /api/v0/models; update cache.  Silently swallows all errors."""
        try:
            r = await asyncio.wait_for(
                client.get(f"{IMAGE_GEN_BASE_URL}/v1/models"),
                timeout=4.0,
            )
            if r.status_code == 200:
                data = r.json()
                self._models     = data.get("data", [])
                self._probe_ok   = True
            else:
                self._models   = []
                self._probe_ok = False
        except Exception:
            self._models   = []
            self._probe_ok = False

        # Fetch load state from /api/v0/models (LM Studio extension).
        try:
            r2 = await asyncio.wait_for(
                client.get(f"{IMAGE_GEN_BASE_URL}/api/v0/models"),
                timeout=4.0,
            )
            if r2.status_code == 200:
                v0_data = r2.json()
                v0_list = v0_data if isinstance(v0_data, list) else v0_data.get("data", [])
                self._loaded_ids = [
                    m.get("id", "")
                    for m in v0_list
                    if m.get("state") == "loaded"
                    and "embed" not in (m.get("id") or "").lower()
                ]
            else:
                self._loaded_ids = []
        except Exception:
            self._loaded_ids = []

        self._last_probe = _time.monotonic()

    async def _ensure_fresh(self, client: httpx.AsyncClient) -> None:
        if _time.monotonic() - self._last_probe >= self._TTL:
            await self._refresh(client)

    async def models(self, client: httpx.AsyncClient) -> list[dict[str, Any]]:
        """Return cached model list, refreshing if the TTL has expired."""
        await self._ensure_fresh(client)
        return list(self._models)

    async def is_available(self, client: httpx.AsyncClient) -> bool:
        """True if LM Studio responded with at least one model."""
        await self._ensure_fresh(client)
        return self._probe_ok and bool(self._models)

    async def loaded_models(self, client: httpx.AsyncClient) -> list[str]:
        """Return model IDs currently loaded in GPU memory (from /api/v0/models)."""
        await self._ensure_fresh(client)
        return list(self._loaded_ids)

    async def at_capacity(self, client: httpx.AsyncClient) -> bool:
        """True if the number of loaded models meets or exceeds _MAX_LOADED."""
        await self._ensure_fresh(client)
        return len(self._loaded_ids) >= self._MAX_LOADED

    async def ensure_model_or_busy(self, client: httpx.AsyncClient, model: str) -> str | None:
        """Check whether *model* can be used without evicting another model.

        Returns None if the model is safe to use (already loaded, or LM Studio
        has capacity to load it).  Returns a human-readable error string if the
        model is NOT loaded and LM Studio is already at its loaded-model limit.
        """
        await self._ensure_fresh(client)
        if model in self._loaded_ids:
            return None  # already loaded — OK
        if len(self._loaded_ids) < self._MAX_LOADED:
            return None  # room to load — OK
        loaded_str = ", ".join(self._loaded_ids) if self._loaded_ids else "none"
        return (
            f"LM Studio is at capacity ({len(self._loaded_ids)}/{self._MAX_LOADED} models loaded: "
            f"{loaded_str}). Model '{model}' is not loaded and sending this request would evict "
            f"an active model.\n"
            f"→ Unload a model in LM Studio first, or use one of the already-loaded models."
        )

    async def has_vision(self, client: httpx.AsyncClient) -> bool:
        """True if any loaded model supports vision (multimodal)."""
        for m in await self.models(client):
            mid = (m.get("id") or "").lower()
            mtype = (m.get("type") or "").lower()
            if ("vision" in mid or "vlm" in mid or "vision" in mtype or "multimodal" in mtype
                    or "-vl-" in mid or mid.endswith("-vl") or "-vl." in mid):
                return True
        return False

    async def best_vision_model(self, client: httpx.AsyncClient) -> str:
        """Return the best vision-capable model ID, or '' if none is loaded.

        Checks for common vision model naming conventions:
        - Qwen-VL family: qwen3-vl-*, qwen-vl-*
        - Any model with 'vision', 'vlm', 'multimodal' in the ID
        Skips embedding models.

        Prefers models that are already loaded in GPU memory to avoid
        triggering model eviction when LM Studio is at capacity.
        """
        if IMAGE_GEN_MODEL.strip():
            return IMAGE_GEN_MODEL.strip()
        _VISION_KEYS = ("vision", "vlm", "multimodal")
        _VL_PATTERNS = ("-vl-", "-vl.", ".vl-", "qwen-vl", "qwen3-vl")
        await self._ensure_fresh(client)
        loaded_set = set(self._loaded_ids)
        vision_candidates: list[tuple[int, int, str]] = []
        for m in await self.models(client):
            mid   = (m.get("id") or "").lower()
            mtype = (m.get("type") or "").lower()
            if "embed" in mid or "embed" in mtype:
                continue
            is_vision = (
                any(k in mid for k in _VISION_KEYS)
                or any(k in mid for k in _VL_PATTERNS)
                or mid.endswith("-vl")
                or any(k in mtype for k in _VISION_KEYS)
            )
            if is_vision:
                # Prefer smaller/faster models
                if any(k in mid for k in ("flash", "tiny", "mini", "nano", "small")):
                    score = 0
                elif any(k in mid for k in ("7b", "8b", "3b")):
                    score = 1
                else:
                    score = 2
                # loaded_priority: 0 = loaded (prefer), 1 = not loaded
                loaded_priority = 0 if m.get("id", "") in loaded_set else 1
                vision_candidates.append((loaded_priority, score, m.get("id", "")))
        if vision_candidates:
            vision_candidates.sort()
            return vision_candidates[0][2]
        return ""

    async def has_image_gen(self, client: httpx.AsyncClient) -> bool:
        """True if any image-generation model is loaded."""
        for m in await self.models(client):
            mid  = (m.get("id") or "").lower()
            mtype = (m.get("type") or "").lower()
            if any(k in mid for k in ("flux", "sdxl", "stable-diffusion", "sd-", "img2img")):
                return True
            if "image" in mtype or "diffusion" in mtype:
                return True
        # If IMAGE_GEN_MODEL env is explicitly set, trust the user
        if IMAGE_GEN_MODEL.strip():
            return self._probe_ok
        return False

    async def best_chat_model(self, client: httpx.AsyncClient) -> str:
        """Return IMAGE_GEN_MODEL if set, else first suitable text-chat model, else ''.

        Prefers models that are already loaded in GPU memory to avoid
        triggering model eviction when LM Studio is at capacity.
        """
        if IMAGE_GEN_MODEL.strip():
            return IMAGE_GEN_MODEL.strip()
        await self._ensure_fresh(client)
        loaded_set = set(self._loaded_ids)
        # First pass: look for a loaded chat model (avoids eviction)
        fallback: str = ""
        for m in await self.models(client):
            mid   = (m.get("id") or "").lower()
            mtype = (m.get("type") or "").lower()
            # Skip embedding and vision-only models
            if "embed" in mid or "embed" in mtype:
                continue
            if "vl-" in mid or "-vl" in mid or mid.endswith("vl") or "flash" in mid:
                continue
            if "llm" in mtype or "chat" in mtype or not mtype:
                model_id = m.get("id", "")
                if model_id in loaded_set:
                    return model_id  # already loaded — best choice
                if not fallback:
                    fallback = model_id  # first suitable model (may not be loaded)
        return fallback

    @staticmethod
    def _model_total_params_b(mid: str) -> float | None:
        """Extract total parameter count (in billions) from a model ID string.

        Handles:
        - Standard:  mistral-7b → 7.0
        - Decimal:   qwen2.5-14b → 14.0
        - MoE total: qwen3.5-35b-a3b → 35.0  (total weights, not active)
        - MoE total: lfm2-24b-a2b → 24.0
        Uses the largest Nb token before any '-aNb' active-params suffix.
        """
        import re as _re_p
        # MoE pattern: <total>b-a<active>b  →  return total
        moe = _re_p.search(r"(\d+(?:\.\d+)?)b-a\d+(?:\.\d+)?b", mid)
        if moe:
            return float(moe.group(1))
        # Generic: collect all Nb tokens, return largest
        hits = _re_p.findall(r"(\d+(?:\.\d+)?)b", mid)
        return max(float(x) for x in hits) if hits else None

    @staticmethod
    def _vram_score(mid: str) -> int:
        """Score a model by estimated VRAM fit for a 24 GB GPU (RTX 3090 / 4090).

        Lower score = faster inference (smaller model, better GPU fit).
        Rough Q4 VRAM estimates: 7B≈5 GB, 14B≈9 GB, 20B≈12 GB, 27B≈17 GB,
                                  30B≈18 GB, 35B≈21 GB, 36B≈22 GB.
        Score mapping:
          0  — ≤14 B  (fast, fits with plenty of KV-cache room)
          1  — ≤30 B  (fits at Q4, moderate speed)
          2  — ≤36 B  (tight fit, minimal KV-cache headroom)
          3  — >36 B  (may not fit / very slow due to CPU offloading)
          4  — qualitative small/mini (unknown size, assume compact)
          5  — tiny/nano (fast but weak reasoning)
          9  — unknown size
        """
        b = ModelRegistry._model_total_params_b(mid)
        if b is not None:
            if b <= 14:
                return 0
            if b <= 30:
                return 1
            if b <= 36:
                return 2
            return 3
        if any(k in mid for k in ("tiny", "nano")):
            return 5
        if any(k in mid for k in ("small", "mini")):
            return 4
        return 9

    async def chat_models(self, client: httpx.AsyncClient) -> list[str]:
        """Return ordered list of suitable text-chat model IDs (skips embed/vision).

        Sorted by (loaded_priority, _vram_score): loaded models first, then
        smallest/fastest.  Qualitative labels (tiny, small) are deprioritised
        below explicit parameter-count models because 'devstral-small' is
        actually a large model despite its name.
        """
        if IMAGE_GEN_MODEL.strip():
            return [IMAGE_GEN_MODEL.strip()]
        await self._ensure_fresh(client)
        loaded_set = set(self._loaded_ids)
        candidates = []
        for m in await self.models(client):
            mid   = (m.get("id") or "").lower()
            mtype = (m.get("type") or "").lower()
            if "embed" in mid or "embed" in mtype:
                continue
            if "vl-" in mid or "-vl" in mid or mid.endswith("vl") or "flash" in mid:
                continue
            model_id = m.get("id", "")
            loaded_priority = 0 if model_id in loaded_set else 1
            candidates.append((loaded_priority, self._vram_score(mid), model_id))
        return [orig for _, _, orig in sorted(candidates)]

    async def plan_models(self, client: httpx.AsyncClient) -> list[str]:
        """Like chat_models but skips tiny/nano models (too weak for structured JSON).

        Prefers loaded models first, then models that fit fully in 24 GB VRAM
        and produce reliable JSON output.
        Score 0 (≤14B) → score 1 (≤30B) → score 2 (≤36B) → qualitative 'small' → rest.
        """
        if IMAGE_GEN_MODEL.strip():
            return [IMAGE_GEN_MODEL.strip()]
        await self._ensure_fresh(client)
        loaded_set = set(self._loaded_ids)
        candidates = []
        for m in await self.models(client):
            mid   = (m.get("id") or "").lower()
            mtype = (m.get("type") or "").lower()
            if "embed" in mid or "embed" in mtype:
                continue
            if "vl-" in mid or "-vl" in mid or mid.endswith("vl") or "flash" in mid:
                continue
            score = self._vram_score(mid)
            if score == 5:
                score = 10  # tiny/nano: demote to last-resort for planning tasks
            model_id = m.get("id", "")
            loaded_priority = 0 if model_id in loaded_set else 1
            candidates.append((loaded_priority, score, model_id))
        return [orig for _, _, orig in sorted(candidates)]


# ---------------------------------------------------------------------------
# VisionCache — in-memory phash → vision result (LRU, MAX_SIZE=500)
# ---------------------------------------------------------------------------

class VisionCache:
    """
    LRU in-memory cache mapping image perceptual hashes (phash) to vision
    confirmation results.  Eliminates redundant LM Studio /v1/chat/completions
    calls for images that have already been confirmed or rejected.

    Capacity: 500 entries; least-recently-used entry evicted on overflow.

    LRU semantics
    -------------
    * ``get()`` — cache hit does NOT promote the entry (read-only).
    * ``put()`` — on re-insert, the entry is moved to the *back* of the
      eviction queue so frequently-updated hashes are never spuriously evicted.
    * Uses ``collections.deque`` for O(1) popleft on every eviction.
    """

    _MAX_SIZE: int = 500

    def __init__(self) -> None:
        self._cache: dict[str, tuple[bool, str, float]] = {}
        self._order: collections.deque[str] = collections.deque()  # LRU order

    def get(self, phash: str) -> tuple[bool, str, float] | None:
        """Return cached (is_match, desc, conf) for phash, or None if absent."""
        if not phash:
            return None
        return self._cache.get(phash)

    def put(self, phash: str, result: tuple[bool, str, float]) -> None:
        """Store result; on re-insert move to back (LRU); evict LRU on overflow."""
        if not phash:
            return
        if phash in self._cache:
            # Re-insert: promote to most-recently-used position.
            try:
                self._order.remove(phash)
            except ValueError:
                pass  # defensive: deque and cache can't be out of sync, but be safe
        else:
            # New entry: evict LRU if at capacity.
            if len(self._cache) >= self._MAX_SIZE:
                oldest = self._order.popleft()   # O(1) via deque
                self._cache.pop(oldest, None)
        self._order.append(phash)
        self._cache[phash] = result

    def size(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()
        self._order.clear()


_vision_cache = VisionCache()   # module-level singleton


# ---------------------------------------------------------------------------
# GpuCodeRuntime — auto-inject GPU device detection into code_run payloads
# ---------------------------------------------------------------------------

class GpuCodeRuntime:
    """
    Prepends a GPU device detection preamble to user-supplied code when the
    code references torch, tensorflow, or CUDA keywords.  This gives the
    ``code_run`` tool a ``DEVICE`` variable (``'cuda'`` | ``'mps'`` | ``'cpu'``)
    that works automatically regardless of the host GPU type.

    Pre-installed packages (available without pip install):
      numpy, scipy, opencv-python-headless (cv2), Pillow, httpx
    """

    _PREINSTALLED: frozenset[str] = frozenset({"numpy", "scipy", "cv2", "PIL", "httpx"})
    _GPU_TRIGGERS: frozenset[str] = frozenset({"torch", "tensorflow", "tf.", "cuda", ".to(", "device"})

    _PREAMBLE = _textwrap.dedent("""\
        # ── GPU device auto-detection (injected by aichat GpuCodeRuntime) ──────
        _device = "cpu"
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _device = "cuda"
            elif getattr(getattr(_torch, "backends", None), "mps", None) and _torch.backends.mps.is_available():
                _device = "mps"
        except ImportError:
            pass
        DEVICE = _device   # use this in your code: model.to(DEVICE)
        # ───────────────────────────────────────────────────────────────────────
    """)

    @classmethod
    def needs_device_injection(cls, code: str) -> bool:
        """True if the code mentions GPU-related keywords."""
        return any(kw in code for kw in cls._GPU_TRIGGERS)

    @classmethod
    def prepare(cls, code: str) -> str:
        """Return code with the DEVICE preamble prepended when GPU triggers detected."""
        if cls.needs_device_injection(code):
            return cls._PREAMBLE + "\n" + code
        return code

    @classmethod
    def available_packages(cls) -> list[str]:
        """Return importable GPU-related packages in the current Python env."""
        candidates = ["torch", "tensorflow", "cv2", "numpy", "scipy", "cupy", "jax"]
        available: list[str] = []
        import importlib
        for pkg in candidates:
            try:
                importlib.import_module(pkg)
                available.append(pkg)
            except ImportError:
                pass
        return available


# ---------------------------------------------------------------------------
# Tool dispatch (HTTP calls to sibling Docker services)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Mega-tool → original handler name dispatch
# ---------------------------------------------------------------------------

_MEGA_TOOL_MAP: dict[str, dict[str, str]] = {
    "web": {
        "search": "web_search",
        "quick_search": "web_quick_search",
        "fetch": "web_fetch",
        "extract": "extract_article",
        "summarize": "smart_summarize",
        "news": "news_search",
        "wikipedia": "wikipedia",
        "arxiv": "arxiv_search",
        "youtube": "youtube_transcript",
    },
    "browser": {
        "screenshot": "screenshot",
        "screenshot_search": "screenshot_search",
        "bulk_screenshot": "bulk_screenshot",
        "scroll_screenshot": "scroll_screenshot",
        "navigate": "browser",
        "read": "browser",
        "click": "browser",
        "scroll": "browser",
        "fill": "browser",
        "eval": "browser",
        "screenshot_element": "browser",
        "save_images": "browser",
        "download_images": "browser",
        "list_images": "browser",
        "scrape": "page_scrape",
        "keyboard": "browser_keyboard",
        "fill_form": "browser_fill_form",
    },
    "image": {
        "fetch": "fetch_image",
        "search": "image_search",
        "generate": "image_generate",
        "edit": "image_edit",
        "crop": "image_crop",
        "zoom": "image_zoom",
        "enhance": "image_enhance",
        "scan": "image_scan",
        "stitch": "image_stitch",
        "diff": "image_diff",
        "annotate": "image_annotate",
        "caption": "image_caption",
        "upscale": "image_upscale",
        "remix": "image_remix",
        "face_detect": "face_recognize",
        "similarity": "embed_search",
    },
    "document": {
        "ingest": "docs_ingest",
        "tables": "docs_extract_tables",
        "ocr": "ocr_image",
        "ocr_pdf": "ocr_pdf",
        "pdf_read": "pdf_read",
        "pdf_edit": "pdf_edit",
        "pdf_form": "pdf_fill_form",
        "pdf_merge": "pdf_merge",
        "pdf_split": "pdf_split",
    },
    "media": {
        "video_info": "video_info",
        "video_frames": "video_frames",
        "video_thumbnail": "video_thumbnail",
        "video_transcode": "video_transcode",
        "tts": "tts",
        "detect_objects": "detect_objects",
        "detect_humans": "detect_humans",
    },
    "data": {
        "store_article": "db_store_article",
        "search": "db_search",
        "cache_store": "db_cache_store",
        "cache_get": "db_cache_get",
        "store_image": "db_store_image",
        "list_images": "db_list_images",
        "errors": "get_errors",
    },
    "memory": {
        "store": "memory_store",
        "recall": "memory_recall",
    },
    "knowledge": {
        "add_node": "graph_add_node",
        "add_edge": "graph_add_edge",
        "query": "graph_query",
        "path": "graph_path",
        "search": "graph_search",
    },
    "vector": {
        "store": "vector_store",
        "search": "vector_search",
        "delete": "vector_delete",
        "collections": "vector_collections",
        "embed_store": "embed_store",
        "embed_search": "embed_search",
    },
    "code": {
        "python": "code_run",
        "javascript": "run_javascript",
        "jupyter": "jupyter_exec",
    },
    "custom_tools": {
        "create": "create_tool",
        "list": "list_custom_tools",
        "delete": "delete_custom_tool",
        "call": "call_custom_tool",
    },
    "planner": {
        "create": "plan_create_task",
        "get": "plan_get_task",
        "complete": "plan_complete_task",
        "fail": "plan_fail_task",
        "list": "plan_list_tasks",
        "delete": "plan_delete_task",
        "orchestrate": "orchestrate",
        "plan": "plan_task",
    },
    "jobs": {
        "submit": "job_submit",
        "status": "job_status",
        "result": "job_result",
        "cancel": "job_cancel",
        "list": "job_list",
        "batch": "batch_submit",
    },
    "research": {
        "rss_search": "researchbox_search",
        "rss_push": "researchbox_push",
        "deep": "deep_research",
        "realtime": "realtime",
    },
    "system": {
        "list_categories": "list_tools_by_category",
        "instructions": "get_system_instructions",
        "desktop_screenshot": "desktop_screenshot",
        "desktop_control": "desktop_control",
    },
}


def _resolve_mega_tool(name: str, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Map mega-tool + action to original handler name and adjusted args."""
    if name not in _MEGA_TOOL_MAP:
        return name, args  # not a mega-tool, pass through unchanged
    if name == "think":
        return name, args  # think has no action parameter
    action = str(args.pop("action", "")).strip()
    if not action:
        return name, args  # no action specified
    mapping = _MEGA_TOOL_MAP[name]
    original_name = mapping.get(action)
    if not original_name:
        return name, args
    # For browser tool, preserve the action parameter for the browser handler
    if original_name == "browser":
        # Map mega-tool action names to browser handler action names
        browser_action_map = {
            "navigate": "navigate",
            "read": "read",
            "click": "click",
            "scroll": "scroll",
            "fill": "fill",
            "eval": "eval",
            "screenshot_element": "screenshot_element",
            "save_images": "save_images",
            "download_images": "download_page_images",
            "list_images": "list_images_detail",
        }
        args["action"] = browser_action_map.get(action, action)
    # For system/desktop_control, map desktop_action → action for the handler
    if original_name == "desktop_control" and "desktop_action" in args:
        args["action"] = args.pop("desktop_action")
    # For jobs, remap status_filter/tool_name_filter back to status/tool_name
    if name == "jobs":
        if "status_filter" in args:
            args["status"] = args.pop("status_filter")
        if "tool_name_filter" in args:
            args["tool_name"] = args.pop("tool_name_filter")
    # For image/face_detect, remap annotate_faces → annotate
    if original_name == "face_recognize" and "annotate_faces" in args:
        args["annotate"] = args.pop("annotate_faces")
    return original_name, args


async def _call_tool(name: str, args: dict[str, Any]) -> list[dict[str, Any]]:
    """Dispatch a tool call and return a list of MCP content blocks."""
    # Resolve mega-tool calls to original handler names
    name, args = _resolve_mega_tool(name, dict(args))  # copy args to avoid mutation
    async with httpx.AsyncClient(timeout=60) as c:
        try:
            # ----------------------------------------------------------------
            if name == "screenshot":
                url = str(args.get("url", "")).strip()
                if not url:
                    return _text("screenshot: 'url' is required")
                find_text  = str(args.get("find_text",  "")).strip() or None
                find_image = str(args.get("find_image", "")).strip() or None
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                container_path = f"/workspace/screenshot_{ts}.png"
                shot_req: dict = {"url": url, "path": container_path}
                if find_text:
                    shot_req["find_text"] = find_text
                elif find_image:
                    shot_req["find_image"] = find_image
                try:
                    r = await c.post(f"{BROWSER_URL}/screenshot",
                                     json=shot_req, timeout=20)
                    data = r.json()
                except Exception as exc:
                    return _text(f"Screenshot failed (browser unreachable): {exc}")
                error = data.get("error", "")
                image_urls = data.get("image_urls", [])
                container_path = data.get("path", container_path)
                filename = os.path.basename(container_path)
                host_path = f"/docker/human_browser/workspace/{filename}"
                local_path = os.path.join(BROWSER_WORKSPACE, filename)
                page_title = data.get("title", "") or url
                clipped = data.get("clipped", False)
                image_meta = data.get("image_meta", {})
                if clipped and find_image:
                    src_hint = image_meta.get("src", find_image)
                    nat_w = image_meta.get("natural_w", 0)
                    nat_h = image_meta.get("natural_h", 0)
                    dim_note = f" ({nat_w}×{nat_h} natural)" if nat_w and nat_h else ""
                    clip_note = f"\nImage: '{src_hint}'{dim_note}"
                elif clipped and find_text:
                    clip_note = f"\nZoomed to: '{find_text}'"
                else:
                    clip_note = ""
                summary = (
                    f"Screenshot of: {page_title}\n"
                    f"URL: {url}{clip_note}\n"
                    f"File: {host_path}"
                )
                # Happy path — screenshot file was written
                if os.path.isfile(local_path):
                    try:
                        await c.post(f"{DATABASE_URL}/images/store", json={
                            "url": url,
                            "host_path": host_path,
                            "alt_text": f"Screenshot of {page_title}",
                        })
                    except Exception:
                        pass
                    return _image_blocks(container_path, summary)
                # Screenshot file missing — browser was blocked or crashed.
                # Try fetching a real image from the page DOM (browser v2+ returns image_urls).
                img_hdrs = {
                    "User-Agent": _BROWSER_HEADERS["User-Agent"],
                    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
                }
                for img_url in image_urls[:3]:
                    try:
                        ir = await c.get(img_url, headers=img_hdrs,
                                         follow_redirects=True, timeout=15)
                        if ir.status_code == 200:
                            ct = ir.headers.get("content-type", "image/jpeg").split(";")[0].strip()
                            fallback_summary = (
                                f"Screenshot of: {page_title}\n"
                                f"URL: {url}\n"
                                f"(screenshot blocked — showing page image)"
                            )
                            return _renderer.encode_url_bytes(ir.content, ct, fallback_summary)
                    except Exception:
                        continue
                return _text(f"Screenshot failed: {error or 'unknown error'}. URL: {url}")

            # ----------------------------------------------------------------
            if name == "fetch_image":
                url = str(args.get("url", "")).strip()
                if not url:
                    return _text("fetch_image: 'url' is required")
                img_fetch_headers = {
                    "User-Agent": _BROWSER_HEADERS["User-Agent"],
                    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
                    "Accept-Language": _BROWSER_HEADERS["Accept-Language"],
                    "Accept-Encoding": _BROWSER_HEADERS["Accept-Encoding"],
                    "DNT": "1",
                }
                last_exc: Exception | None = None
                content_type = "image/jpeg"
                img_data = b""
                for attempt in range(2):
                    try:
                        r = await c.get(url, headers=img_fetch_headers, follow_redirects=True)
                        if r.status_code == 429 and attempt == 0:
                            retry_after = min(int(r.headers.get("retry-after", "15")), 30)
                            await asyncio.sleep(retry_after)
                            continue
                        r.raise_for_status()
                        content_type = r.headers.get("content-type", "image/jpeg").split(";")[0].strip()
                        img_data = r.content
                        break
                    except Exception as exc:
                        last_exc = exc
                        if attempt == 0:
                            await asyncio.sleep(3)
                            continue
                else:
                    return _text(f"fetch_image failed: {last_exc}")
                # Derive host_path for DB metadata (workspace is writable via host bind-mount)
                raw_name = url.split("?")[0].split("/")[-1] or "image"
                if "." not in raw_name:
                    ext = {"image/jpeg": ".jpg", "image/png": ".png",
                           "image/gif": ".gif", "image/webp": ".webp"}.get(content_type, ".jpg")
                    raw_name += ext
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"img_{ts}_{raw_name}"
                host_path = f"/docker/human_browser/workspace/{filename}"
                # Save metadata to DB
                try:
                    await c.post(f"{DATABASE_URL}/images/store", json={
                        "url": url,
                        "host_path": host_path,
                        "alt_text": f"Image from {url}",
                    })
                except Exception:
                    pass
                # Return inline base64 image — always compress via ImageRenderer so
                # large PNGs/WebPs never exceed LM Studio's MCP payload cap.
                summary = (
                    f"Image from: {url}\n"
                    f"Type: {content_type}  Size: {len(img_data):,} bytes\n"
                    f"File: {host_path}"
                )
                return _renderer.encode_url_bytes(img_data, content_type, summary)

            # ----------------------------------------------------------------
            if name == "screenshot_search":
                raw_query = str(args.get("query", "")).strip()
                query, normalize_note = _normalize_search_query(raw_query)
                if not query:
                    return _text("screenshot_search: 'query' is required")
                max_results = max(1, min(int(args.get("max_results", 3)), 5))
                from urllib.parse import quote_plus as _qp

                # Search DuckDuckGo HTML for result URLs (realistic headers)
                try:
                    r = await c.get(
                        f"https://html.duckduckgo.com/html/?q={_qp(query)}",
                        headers=_BROWSER_HEADERS,
                        follow_redirects=True,
                    )
                    html = r.text
                except Exception as exc:
                    return _text(f"Search failed: {exc}")

                # Tier 1: parse DDG result links.
                _DDG_HOSTS = ('duckduckgo.com', 'ddg.gg', 'duck.co')
                seen_u: set[str] = set()
                urls: list[str] = []
                for url, _title in _extract_ddg_links(html, max_results=20):
                    if any(d in url for d in _DDG_HOSTS):
                        continue
                    if _url_has_explicit_content(url):
                        continue
                    if url in seen_u:
                        continue
                    seen_u.add(url)
                    urls.append(url)

                # Tier 2: direct href links (fallback if DDG changed format or rate-limited)
                if not urls:
                    href_raw = re.findall(r'href=["\']?(https?://[^"\'>\s]+)', html)
                    urls = list(dict.fromkeys(
                        u for u in href_raw
                        if not any(d in u for d in _DDG_HOSTS)
                        and not _url_has_explicit_content(u)
                    ))

                # Tier 2b: Bing fallback when DDG yields challenge/empty results.
                if not urls:
                    try:
                        rb = await c.get(
                            f"https://www.bing.com/search?q={_qp(query)}&setlang=en-US",
                            headers=_BROWSER_HEADERS,
                            follow_redirects=True,
                        )
                        urls = [
                            u for u, _t in _extract_bing_links(rb.text, max_results=20)
                            if not _url_has_explicit_content(u)
                        ]
                    except Exception:
                        pass

                # Tier 3: browser search + DOM eval (Chromium w/ anti-detection, most reliable)
                if not urls:
                    try:
                        await asyncio.wait_for(
                            c.post(f"{BROWSER_URL}/search", json={"query": query}),
                            timeout=35.0,
                        )
                        ev = await c.post(f"{BROWSER_URL}/eval", json={"code": r"""
                            JSON.stringify(
                                Array.from(document.links)
                                    .map(a => {
                                        try {
                                            const u = new URL(a.href);
                                            if (u.hostname === 'duckduckgo.com' && u.pathname === '/l/')
                                                return u.searchParams.get('uddg') || null;
                                            if (u.hostname !== 'duckduckgo.com' && u.hostname !== 'duck.co')
                                                return a.href;
                                            return null;
                                        } catch(e) { return null; }
                                    })
                                    .filter(u => u && u.startsWith('http'))
                                    .filter((u, i, arr) => arr.indexOf(u) === i)
                                    .slice(0, 5)
                            )
                        """}, timeout=10)
                        extracted = json.loads(ev.json().get("result", "[]"))
                        urls = [u for u in extracted if u and not _url_has_explicit_content(u)]
                    except Exception:
                        pass

                if not urls:
                    return _text(f"No URLs found in search results for: {query}")

                # Rank URLs for relevance and safer domains.
                _SKIP_T1 = ("youtube.com", "youtu.be", "vimeo.com", "dailymotion.com",
                            "twitch.tv", "pinterest.com", "instagram.com", "facebook.com")
                q_terms = _search_terms(query)
                preferred_domains = _query_preferred_domains(query)
                urls = sorted(
                    list(dict.fromkeys(u for u in urls if u.startswith("http"))),
                    key=lambda u: (
                        _score_url_relevance(u, q_terms, preferred_domains)
                        - (30 if any(s in u.lower() for s in _SKIP_T1) else 0)
                    ),
                    reverse=True,
                )[:max_results]

                blocks: list[dict[str, Any]] = [
                    {"type": "text", "text": (
                        f"Visual search: '{query}' — screenshotting {len(urls)} result(s)...\n"
                        + (normalize_note + "\n" if normalize_note else "")
                    )}
                ]
                img_hdrs = {
                    "User-Agent": _BROWSER_HEADERS["User-Agent"],
                    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
                }
                # 24-second budget for all screenshots — fits within LM Studio's timeout.
                _deadline = asyncio.get_running_loop().time() + 24.0
                for i, url in enumerate(urls):
                    remaining = _deadline - asyncio.get_running_loop().time()
                    if remaining < 3:
                        blocks.append({"type": "text", "text": f"(time budget reached — stopped at {i} of {len(urls)} results)"})
                        break
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{i}"
                    cp = f"/workspace/screenshot_{ts}.png"
                    try:
                        sr = await c.post(f"{BROWSER_URL}/screenshot",
                                          json={"url": url, "path": cp},
                                          timeout=min(15.0, remaining - 2.0))
                        data = sr.json()
                    except Exception as exc:
                        blocks.append({"type": "text", "text": f"Failed to screenshot {url}: {exc}"})
                        continue
                    err = data.get("error", "")
                    s_image_urls = data.get("image_urls", [])
                    container_path = data.get("path", cp)
                    filename = os.path.basename(container_path)
                    host_path = f"/docker/human_browser/workspace/{filename}"
                    local_path = os.path.join(BROWSER_WORKSPACE, filename)
                    page_title = data.get("title", "") or url
                    summary = f"{page_title}\n{url}\nFile: {host_path}"
                    if os.path.isfile(local_path):
                        try:
                            await c.post(f"{DATABASE_URL}/images/store", json={
                                "url": url,
                                "host_path": host_path,
                                "alt_text": f"Search: '{query}' — {page_title}",
                            })
                        except Exception:
                            pass
                        blocks.extend(_image_blocks(container_path, summary))
                    else:
                        # Screenshot failed — try image_urls fallback
                        fetched = False
                        for img_url in s_image_urls[:3]:
                            try:
                                ir = await c.get(img_url, headers=img_hdrs,
                                                 follow_redirects=True, timeout=15)
                                if ir.status_code == 200:
                                    ct = ir.headers.get("content-type", "image/jpeg").split(";")[0].strip()
                                    b64 = base64.standard_b64encode(ir.content).decode("ascii")
                                    fb_summary = f"{page_title}\n{url}\n(screenshot blocked — showing page image)"
                                    blocks.extend([
                                        {"type": "text", "text": fb_summary},
                                        {"type": "image", "data": b64, "mimeType": ct},
                                    ])
                                    fetched = True
                                    break
                            except Exception:
                                continue
                        if not fetched:
                            blocks.append({"type": "text", "text": f"Failed: {url} — {err or 'screenshot unavailable'}"})
                return blocks

            # ----------------------------------------------------------------
            # quick_search: SearXNG-only fast search (1-3s, no fetch/extract)
            if name == "web_quick_search":
                raw_query = str(args.get("query", "")).strip()
                query, normalize_note = _normalize_search_query(raw_query)
                if not query:
                    return _text("quick_search: 'query' is required")
                try:
                    links = await _searxng_search(c, query, max_results=8)
                    if not links:
                        return _text(f"No results found for: {query}")
                    lines = [f"[Quick search] Query: {query}"]
                    if normalize_note:
                        lines.append(normalize_note)
                    for idx, (url, title) in enumerate(links[:8], start=1):
                        lines.append(f"{idx}. {title or url}")
                        lines.append(f"URL: {url}")
                    return _text("\n".join(lines))
                except Exception as exc:
                    return _text(f"quick_search failed: {exc}")

            # ----------------------------------------------------------------
            if name == "web_search":
                raw_query = str(args.get("query", "")).strip()
                query, normalize_note = _normalize_search_query(raw_query)
                if not query:
                    return _text("web_search: 'query' is required")
                engine = str(args.get("engine", "auto")).strip().lower() or "auto"
                max_chars = int(args.get("max_chars", 4000))
                max_chars = max(500, min(max_chars, 16000))
                from urllib.parse import quote_plus as _qp

                q_terms = _search_terms(query)
                pref = _query_preferred_domains(query)

                def _fmt_links(
                    links: "list[tuple[str, str]]",
                    label: str = "Search results",
                    is_images: bool = False,
                ) -> "str | None":
                    """Rank, filter and format [(url, title)] into a result string."""
                    cleaned = [
                        (u, t) for (u, t) in links
                        if u and not _url_has_explicit_content(u, t)
                    ]
                    # Source strategy ranking (preferred sources, dedup, quality)
                    _ss = _get_source_strategy()
                    _as_dicts = [{"url": u, "title": t} for u, t in cleaned]
                    _ranked = _ss.rank_results(_as_dicts, query)
                    cleaned = [(r["url"], r["title"]) for r in _ranked]
                    if not cleaned:
                        return None
                    lines = [f"[{label}] Query: {query}"]
                    if normalize_note:
                        lines.append(normalize_note)
                    for idx, (url, title) in enumerate(cleaned[:8], start=1):
                        if is_images:
                            lines.append(f"{idx}. {title or url}")
                            lines.append(f"Image URL: {url}")
                        else:
                            lines.append(f"{idx}. {title or url}")
                            lines.append(f"URL: {url}")
                    return "\n".join(lines)[:max_chars]

                # ── Tier 0: SearXNG meta-search ──────────────────────────────
                # Maps engine → (primary_engine_filter, category).
                # primary="" means "use all available SearXNG engines".
                # If the primary-filtered search returns nothing (e.g. Google
                # is blocked on this SearXNG instance), we retry with all
                # engines so Bing/Brave still provide results.
                _SEARXNG_ENGINE_MAP: dict[str, tuple[str, str]] = {
                    "auto":    ("",            "general"),
                    "searxng": ("",            "general"),
                    "google":  ("google",      "general"),
                    "bing":    ("bing",        "general"),
                    "ddg":     ("duckduckgo",  "general"),
                    "brave":   ("brave",       "general"),
                    "images":  ("",            "images"),
                }
                sx_engines, sx_category = _SEARXNG_ENGINE_MAP.get(
                    engine, ("", "general")
                )
                sx_links = await _searxng_search(
                    c, query,
                    engines=sx_engines,
                    categories=sx_category,
                    max_results=12,
                )
                # If specific engine filter returned nothing, retry with all engines
                if not sx_links and sx_engines:
                    sx_links = await _searxng_search(
                        c, query,
                        engines="",
                        categories=sx_category,
                        max_results=12,
                    )
                if sx_links:
                    formatted = _fmt_links(
                        sx_links,
                        label="Search results" if engine != "images" else "Image results",
                        is_images=(engine == "images"),
                    )
                    if formatted:
                        return _text(formatted)


                # ── Tier 1b: Engine-specific direct HTML scraping ─────────────
                if engine in ("auto", "ddg"):
                    try:
                        r = await c.get(
                            f"https://html.duckduckgo.com/html/?q={_qp(query)}",
                            headers=_BROWSER_HEADERS,
                            follow_redirects=True,
                        )
                        challenge_markers = (
                            "bots use duckduckgo too",
                            "select all squares",
                            "error-lite@duckduckgo.com",
                        )
                        if not any(m in r.text.lower() for m in challenge_markers):
                            ddg_links = _extract_ddg_links(r.text, max_results=10)
                            formatted = _fmt_links(ddg_links, label="Search results (DDG)")
                            if formatted:
                                return _text(formatted)
                    except Exception:
                        pass

                if engine in ("auto", "bing"):
                    try:
                        rb = await c.get(
                            f"https://www.bing.com/search?q={_qp(query)}&setlang=en-US",
                            headers=_BROWSER_HEADERS,
                            follow_redirects=True,
                        )
                        bing_links = _extract_bing_links(rb.text, max_results=10)
                        formatted = _fmt_links(bing_links, label="Search results (Bing)")
                        if formatted:
                            return _text(formatted)
                    except Exception:
                        pass

                if engine in ("google",):
                    try:
                        rg = await c.get(
                            f"https://www.google.com/search?q={_qp(query)}&hl=en&num=10",
                            headers={
                                **_BROWSER_HEADERS,
                                "Accept-Language": "en-US,en;q=0.9",
                                "Referer": "https://www.google.com/",
                            },
                            follow_redirects=True,
                        )
                        g_links = _extract_google_links(rg.text, max_results=10)
                        formatted = _fmt_links(g_links, label="Search results (Google)")
                        if formatted:
                            return _text(formatted)
                    except Exception:
                        pass

                # ── Tier 2: Real-browser DOM extraction ───────────────────────
                try:
                    # For Google Images, navigate directly to images.google.com
                    if engine == "images":
                        browser_url_target = f"https://www.google.com/search?q={_qp(query)}&tbm=isch"
                        dom_js = r"""
                            JSON.stringify(
                                Array.from(document.querySelectorAll('img[src]'))
                                    .map(img => img.src)
                                    .filter(s => s && s.startsWith('http') && !s.includes('google.com/logos'))
                                    .filter((s, i, a) => a.indexOf(s) === i)
                                    .slice(0, 20)
                            )
                        """
                        await asyncio.wait_for(
                            c.post(f"{BROWSER_URL}/navigate", json={"url": browser_url_target}),
                            timeout=35.0,
                        )
                    else:
                        dom_js = r"""
                            JSON.stringify(
                                Array.from(document.links)
                                    .map(a => {
                                        try {
                                            const u = new URL(a.href);
                                            if (u.hostname === 'duckduckgo.com' && u.pathname === '/l/')
                                                return [u.searchParams.get('uddg') || null, a.innerText.trim()];
                                            if (!['duckduckgo.com','duck.co','google.com','bing.com'].includes(u.hostname))
                                                return [a.href, a.innerText.trim()];
                                            return null;
                                        } catch(e) { return null; }
                                    })
                                    .filter(x => x && x[0] && x[0].startsWith('http'))
                                    .filter((x, i, arr) => arr.findIndex(y => y && y[0] === x[0]) === i)
                                    .slice(0, 20)
                            )
                        """
                        await asyncio.wait_for(
                            c.post(f"{BROWSER_URL}/search", json={"query": query}),
                            timeout=35.0,
                        )
                    ev = await c.post(
                        f"{BROWSER_URL}/eval",
                        json={"code": dom_js},
                        timeout=10,
                    )
                    raw = json.loads(ev.json().get("result", "[]"))
                    if engine == "images":
                        browser_links = [(u, u) for u in raw if u]
                        formatted = _fmt_links(
                            browser_links, label="Image results (browser)", is_images=True
                        )
                    else:
                        browser_links = [
                            (item[0], item[1]) if isinstance(item, list) else (item, item)
                            for item in raw if item
                        ]
                        formatted = _fmt_links(browser_links, label="Search results (browser)")
                    if formatted:
                        return _text(formatted)
                except Exception:
                    pass

                # ── Tier 3: DDG lite (last resort text scrape) ────────────────
                try:
                    r = await c.get(
                        f"https://lite.duckduckgo.com/lite/?q={_qp(query)}",
                        headers=_BROWSER_HEADERS,
                        follow_redirects=True,
                    )
                    text = re.sub(r"<[^>]+>", " ", r.text)
                    text = re.sub(r"\s+", " ", text).strip()[:max_chars]
                    header = f"[Search results (lite)] Query: {query}\n"
                    if normalize_note:
                        header += normalize_note + "\n"
                    return _text(header + "\n" + text)
                except Exception as exc:
                    return _text(f"web_search failed: {exc}")

            # ----------------------------------------------------------------
            if name == "web_fetch":
                url = str(args.get("url", "")).strip()
                max_chars = int(args.get("max_chars", 4000))
                max_chars = max(500, min(max_chars, 16000))
                # Check cache first
                try:
                    cache_r = await c.get(f"{DATABASE_URL}/cache/get", params={"key": url})
                    if cache_r.status_code == 200:
                        data = cache_r.json()
                        if data.get("found"):
                            cached_text = data.get("content", "")
                            # Re-cache items may contain raw HTML from old behavior — strip if needed.
                            # Use `>?` so truncated tags (no closing >) are also removed.
                            if cached_text.lstrip().startswith("<"):
                                cached_text = re.sub(r"<[^>]*>?", " ", cached_text)
                                cached_text = re.sub(r"\s+", " ", cached_text).strip()
                            if len(cached_text) > 50:
                                return _text(f"[cached] {cached_text[:max_chars]}")
                            # Too short (stripped HTML left only a title/nav) — fall through to live fetch
                except Exception:
                    pass
                # Fetch via browser (renders JS, returns clean text, handles SSL)
                text = ""
                try:
                    nav_r = await asyncio.wait_for(
                        c.post(f"{BROWSER_URL}/navigate", json={"url": url}),
                        timeout=20.0,
                    )
                    nav_data = nav_r.json()
                    text = nav_data.get("content", "")
                    if text:
                        title = nav_data.get("title", "")
                        final_url = nav_data.get("url", url)
                        header = f"Title: {title}\nURL: {final_url}\n\n" if title else ""
                        text = (header + text)[:max_chars]
                except Exception:
                    pass
                # Fallback: httpx + strip tags
                if not text:
                    try:
                        r = await c.get(url, headers=_BROWSER_HEADERS, follow_redirects=True)
                        raw = r.text
                        text = re.sub(r"<[^>]+>", " ", raw)
                        text = re.sub(r"\s+", " ", text).strip()[:max_chars]
                    except Exception as exc:
                        return _text(f"web_fetch failed: {exc}")
                try:
                    await c.post(f"{DATABASE_URL}/cache/store", json={"url": url, "content": text})
                except Exception:
                    pass
                return _text(text)

            # ----------------------------------------------------------------
            if name == "db_store_article":
                r = await c.post(f"{DATABASE_URL}/articles/store", json=args)
                return _json_or_err(r, "db_store_article")

            if name == "db_search":
                search_params: dict = {}
                if args.get("topic"):
                    search_params["topic"] = str(args["topic"])
                if args.get("q"):
                    search_params["q"] = str(args["q"])
                try:
                    search_params["limit"] = max(1, min(200, int(args.get("limit", 20))))
                    search_params["offset"] = max(0, int(args.get("offset", 0)))
                except (ValueError, TypeError) as exc:
                    return _text(f"db_search: 'limit' and 'offset' must be integers — {exc}")
                if args.get("summary_only"):
                    search_params["summary_only"] = "true"
                r = await c.get(f"{DATABASE_URL}/articles/search", params=search_params)
                return _json_or_err(r, "db_search")

            if name == "db_cache_store":
                cache_url_cs = str(args.get("url", "")).strip()
                cache_content_cs = str(args.get("content", "")).strip()
                if not cache_url_cs:
                    return _text("db_cache_store: 'url' is required")
                if not cache_content_cs:
                    return _text("db_cache_store: 'content' is required")
                cache_payload_cs: dict = {"url": cache_url_cs, "content": cache_content_cs}
                if args.get("title"):
                    cache_payload_cs["title"] = str(args["title"])
                r = await c.post(f"{DATABASE_URL}/cache/store", json=cache_payload_cs)
                return _json_or_err(r, "db_cache_store")

            if name == "db_cache_get":
                cache_url_cg = str(args.get("url", "")).strip()
                if not cache_url_cg:
                    return _text("db_cache_get: 'url' is required")
                r = await c.get(f"{DATABASE_URL}/cache/get", params={"key": cache_url_cg})
                return _json_or_err(r, "db_cache_get")

            # ----------------------------------------------------------------
            if name == "db_store_image":
                r = await c.post(f"{DATABASE_URL}/images/store", json={
                    "url":       args.get("url", ""),
                    "host_path": args.get("host_path", ""),
                    "alt_text":  args.get("alt_text", ""),
                })
                return _json_or_err(r, "db_store_image")

            if name == "db_list_images":
                limit = int(args.get("limit", 20))
                r = await c.get(f"{DATABASE_URL}/images/list", params={"limit": limit})
                if r.status_code >= 400:
                    return _text(f"db_list_images: upstream returned {r.status_code}")
                data = r.json()
                images = data.get("images", [])
                if not images:
                    return _text("No screenshots stored yet.")
                lines = [f"Stored screenshots ({len(images)}):"]
                for img in images:
                    hp = img.get("host_path") or img.get("url", "")
                    alt = img.get("alt_text", "")
                    ts = img.get("stored_at", "")[:19].replace("T", " ")
                    lines.append(f"  {hp}" + (f"  [{alt}]" if alt else "") + (f"  {ts}" if ts else ""))
                # Inline the most recent image — derive container path from host_path basename
                hp0 = images[0].get("host_path", "") or ""
                most_recent = f"/workspace/{os.path.basename(hp0)}" if hp0 else ""
                return _image_blocks(most_recent, "\n".join(lines))

            # ----------------------------------------------------------------
            if name == "memory_store":
                key_ms = str(args.get("key", "")).strip()
                val_ms = str(args.get("value", "")).strip()
                if not key_ms:
                    return _text("memory_store: 'key' is required")
                if not val_ms:
                    return _text("memory_store: 'value' is required")
                payload: dict = {"key": key_ms, "value": val_ms}
                if args.get("ttl_seconds"):
                    try:
                        payload["ttl_seconds"] = int(args["ttl_seconds"])
                    except (ValueError, TypeError):
                        return _text("memory_store: 'ttl_seconds' must be an integer")
                r = await c.post(f"{MEMORY_URL}/store", json=payload)
                return _json_or_err(r, "memory_store")

            if name == "memory_recall":
                params: dict = {}
                if args.get("key"):
                    params["key"] = str(args["key"])
                if args.get("pattern"):
                    params["pattern"] = str(args["pattern"])
                r = await c.get(f"{MEMORY_URL}/recall", params=params)
                return _json_or_err(r, "memory_recall")

            if name == "researchbox_search":
                r = await c.get(f"{RESEARCH_URL}/search-feeds", params={"topic": args.get("topic", "")})
                return _json_or_err(r, "researchbox_search")

            if name == "researchbox_push":
                r = await c.post(f"{RESEARCH_URL}/push-feed", json=args)
                return _json_or_err(r, "researchbox_push")

            # ----------------------------------------------------------------
            if name == "browser":
                action = str(args.get("action", "")).strip()
                if not action:
                    return _text("browser: 'action' is required")
                if action == "navigate":
                    url = str(args.get("url", "")).strip()
                    if not url:
                        return _text("browser navigate: 'url' is required")
                    try:
                        nav_r = await asyncio.wait_for(
                            c.post(f"{BROWSER_URL}/navigate", json={"url": url}),
                            timeout=20.0,
                        )
                        data = nav_r.json()
                        content = data.get("content", "")
                        title = data.get("title", "")
                        final_url = data.get("url", url)
                        header = f"Title: {title}\nURL: {final_url}\n\n" if title else ""
                        return _text((header + content)[:8000])
                    except Exception as exc:
                        return _text(f"browser navigate failed: {exc}")
                if action == "read":
                    try:
                        read_r = await c.get(f"{BROWSER_URL}/read", timeout=10.0)
                        data = read_r.json()
                        content = data.get("content", "")
                        title = data.get("title", "")
                        header = f"Title: {title}\n\n" if title else ""
                        return _text((header + content)[:8000])
                    except Exception as exc:
                        return _text(f"browser read failed: {exc}")
                if action in {"click", "left_click", "right_click"}:
                    selector = str(args.get("selector", "")).strip()
                    if not selector:
                        return _text("browser click: 'selector' is required")
                    try:
                        button = str(args.get("button", "left")).strip().lower() or "left"
                        if action == "left_click":
                            button = "left"
                        if action == "right_click":
                            button = "right"
                        try:
                            click_count = int(args.get("click_count", 1))
                        except (TypeError, ValueError):
                            click_count = 1
                        click_r = await c.post(
                            f"{BROWSER_URL}/click",
                            json={
                                "selector": selector,
                                "button": button,
                                "click_count": click_count,
                            },
                            timeout=10.0,
                        )
                        data = click_r.json()
                        return _text(data.get("content", "Clicked."))
                    except Exception as exc:
                        return _text(f"browser click failed: {exc}")
                if action == "scroll":
                    direction = str(args.get("direction", "down")).strip().lower() or "down"
                    try:
                        amount = int(args.get("amount", 800))
                    except (TypeError, ValueError):
                        amount = 800
                    behavior = str(args.get("behavior", "instant")).strip().lower() or "instant"
                    try:
                        scroll_r = await c.post(
                            f"{BROWSER_URL}/scroll",
                            json={"direction": direction, "amount": amount, "behavior": behavior},
                            timeout=10.0,
                        )
                        data = scroll_r.json()
                        if data.get("error"):
                            return _text(f"browser scroll failed: {data.get('error')}")
                        return _text(
                            "Scrolled page.\n"
                            f"Direction: {data.get('direction', direction)}  "
                            f"Amount: {data.get('amount', amount)}  "
                            f"Behavior: {data.get('behavior', behavior)}\n"
                            f"Position: x={data.get('scroll_x', 0)} y={data.get('scroll_y', 0)}"
                        )
                    except Exception as exc:
                        return _text(f"browser scroll failed: {exc}")
                if action == "fill":
                    selector = str(args.get("selector", "")).strip()
                    value = str(args.get("value", ""))
                    if not selector:
                        return _text("browser fill: 'selector' is required")
                    try:
                        fill_r = await c.post(
                            f"{BROWSER_URL}/fill",
                            json={"selector": selector, "value": value},
                            timeout=10.0,
                        )
                        data = fill_r.json()
                        return _text(data.get("content", "Filled."))
                    except Exception as exc:
                        return _text(f"browser fill failed: {exc}")
                if action == "eval":
                    code = str(args.get("code", "")).strip()
                    if not code:
                        return _text("browser eval: 'code' is required")
                    try:
                        eval_r = await c.post(
                            f"{BROWSER_URL}/eval", json={"code": code}, timeout=10.0
                        )
                        data = eval_r.json()
                        return _text(str(data.get("result", "")))
                    except Exception as exc:
                        return _text(f"browser eval failed: {exc}")
                if action == "screenshot_element":
                    selector = str(args.get("selector", "")).strip()
                    if not selector:
                        return _text("browser screenshot_element: 'selector' is required")
                    pad = int(args.get("pad", 20))
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    container_path = f"/workspace/element_{ts}.png"
                    try:
                        el_r = await c.post(
                            f"{BROWSER_URL}/screenshot_element",
                            json={"selector": selector, "path": container_path, "pad": pad},
                            timeout=20.0,
                        )
                        data = el_r.json()
                    except Exception as exc:
                        return _text(f"browser screenshot_element failed: {exc}")
                    err = data.get("error", "")
                    if err:
                        return _text(f"browser screenshot_element: {err}")
                    saved_path = data.get("path", container_path)
                    filename = os.path.basename(saved_path)
                    local_path = os.path.join(BROWSER_WORKSPACE, filename)
                    host_path = f"/docker/human_browser/workspace/{filename}"
                    bbox = data.get("bbox", {})
                    bbox_note = (
                        f"  bbox: x={bbox.get('x',0):.0f}, y={bbox.get('y',0):.0f}, "
                        f"w={bbox.get('width',0):.0f}, h={bbox.get('height',0):.0f}"
                        if bbox else ""
                    )
                    summary = (
                        f"Element screenshot: {selector}\n"
                        f"File: {host_path}{bbox_note}"
                    )
                    if os.path.isfile(local_path):
                        return _image_blocks(saved_path, summary)
                    return _text(f"screenshot_element: file not found at {local_path}")
                if action == "list_images_detail":
                    try:
                        imgs_r = await c.get(f"{BROWSER_URL}/images", timeout=10.0)
                        imgs_data = imgs_r.json()
                    except Exception as exc:
                        return _text(f"browser list_images_detail failed: {exc}")
                    images = imgs_data.get("images", imgs_data) if isinstance(imgs_data, dict) else imgs_data
                    if not images:
                        return _text("No images found on the current page.")
                    lines = ["Images on current page:"]
                    for img in images:
                        idx  = img.get("index", "?")
                        src  = img.get("src", "")
                        alt  = img.get("alt", "")
                        rw   = img.get("rendered_w", 0)
                        rh   = img.get("rendered_h", 0)
                        nw   = img.get("natural_w", 0)
                        nh   = img.get("natural_h", 0)
                        vis  = "✓" if img.get("visible") else "✗"
                        vp   = "in-viewport" if img.get("in_viewport") else "off-screen"
                        dim  = f"{rw}×{rh}px rendered" + (f" ({nw}×{nh} natural)" if nw else "")
                        lines.append(
                            f"  [{idx}] {vis} {vp}  {dim}\n"
                            f"       src: {src[:120]}\n"
                            f"       alt: {alt[:80]}"
                        )
                    return _text("\n".join(lines))
                if action == "save_images":
                    raw_urls = args.get("urls", [])
                    if isinstance(raw_urls, str):
                        raw_urls = [u.strip() for u in raw_urls.split(",") if u.strip()]
                    if not raw_urls:
                        return _text("browser save_images: 'urls' is required")
                    prefix = str(args.get("prefix", "image")).strip() or "image"
                    try:
                        max_imgs = int(args.get("max", 20))
                    except (TypeError, ValueError):
                        max_imgs = 20
                    try:
                        sav_r = await c.post(
                            f"{BROWSER_URL}/save_images",
                            json={"urls": raw_urls, "prefix": prefix, "max": max_imgs},
                            timeout=120.0,
                        )
                        data = sav_r.json()
                    except Exception as exc:
                        return _text(f"browser save_images failed: {exc}")
                    if data.get("error"):
                        return _text(f"browser save_images failed: {data.get('error')}")
                    saved = data.get("saved", [])
                    errors = data.get("errors", [])
                    if not saved and errors:
                        errs = "; ".join(e.get("error", "?") for e in errors[:3])
                        return _text(f"browser save_images: all downloads failed — {errs}")
                    blocks: list = []
                    lines = [f"Downloaded {len(saved)} image(s)" +
                             (f"  ({len(errors)} failed)" if errors else "")]
                    for item in saved:
                        p = item.get("path", "")
                        fname = os.path.basename(p)
                        hp = f"/docker/human_browser/workspace/{fname}" if fname else ""
                        size_kb = item.get("size", 0) // 1024
                        lines.append(f"  [{item.get('index','?')}] {fname}  {size_kb} KB")
                        if hp and os.path.isfile(os.path.join(BROWSER_WORKSPACE, fname)) and len(blocks) < 10:
                            blocks.extend(_image_blocks(p, fname))
                    blocks.insert(0, _text("\n".join(lines))[0])
                    return blocks
                if action == "download_page_images":
                    url = str(args.get("url", "")).strip()
                    if url:
                        try:
                            nav_r = await c.post(f"{BROWSER_URL}/navigate", json={"url": url}, timeout=20.0)
                            nav_data = nav_r.json()
                            if nav_data.get("error"):
                                return _text(f"browser download_page_images: navigate failed — {nav_data['error']}")
                        except Exception as exc:
                            return _text(f"browser download_page_images: navigate failed — {exc}")
                    prefix = str(args.get("prefix", "image")).strip() or "image"
                    try:
                        max_imgs = int(args.get("max", 20))
                    except (TypeError, ValueError):
                        max_imgs = 20
                    filter_q = str(args.get("filter", "")).strip() or None
                    payload: dict[str, Any] = {"prefix": prefix, "max": max_imgs}
                    if filter_q:
                        payload["filter"] = filter_q
                    try:
                        dl_r = await c.post(
                            f"{BROWSER_URL}/download_page_images",
                            json=payload,
                            timeout=120.0,
                        )
                        data = dl_r.json()
                    except Exception as exc:
                        return _text(f"browser download_page_images failed: {exc}")
                    if data.get("error"):
                        return _text(f"browser download_page_images failed: {data.get('error')}")
                    saved = data.get("saved", [])
                    errors = data.get("errors", [])
                    applied_filter = data.get("filter")
                    filter_note = f" (filter: '{applied_filter}')" if applied_filter else ""
                    if not saved:
                        msg = f"No images downloaded{filter_note}"
                        if errors:
                            errs = "; ".join(e.get("error", "?") for e in errors[:3])
                            msg += f" — {errs}"
                        return _text(msg)
                    blocks = []
                    lines = [f"Downloaded {len(saved)} image(s){filter_note}" +
                             (f"  ({len(errors)} failed)" if errors else "")]
                    for item in saved:
                        p = item.get("path", "")
                        fname = os.path.basename(p)
                        size_kb = item.get("size", 0) // 1024
                        lines.append(f"  [{item.get('index','?')}] {fname}  {size_kb} KB")
                        if fname and os.path.isfile(os.path.join(BROWSER_WORKSPACE, fname)) and len(blocks) < 10:
                            blocks.extend(_image_blocks(p, fname))
                    blocks.insert(0, _text("\n".join(lines))[0])
                    return blocks
                return _text(f"browser: unknown action '{action}'")

            # ----------------------------------------------------------------
            if name == "create_tool":
                tool_name = str(args.get("tool_name", "")).strip()
                description = str(args.get("description", "")).strip()
                parameters = args.get("parameters", {})
                code = str(args.get("code", "")).strip()
                if not tool_name or not description or not code:
                    return _text("create_tool: 'tool_name', 'description', and 'code' are required")
                r = await c.post(
                    f"{TOOLKIT_URL}/register",
                    json={
                        "tool_name": tool_name,
                        "description": description,
                        "parameters": parameters,
                        "code": code,
                    },
                    timeout=10.0,
                )
                return _json_or_err(r, "create_tool")

            if name == "list_custom_tools":
                r = await c.get(f"{TOOLKIT_URL}/tools", timeout=10.0)
                return _json_or_err(r, "list_custom_tools")

            if name == "delete_custom_tool":
                tool_name = str(args.get("tool_name", "")).strip()
                if not tool_name:
                    return _text("delete_custom_tool: 'tool_name' is required")
                r = await c.delete(f"{TOOLKIT_URL}/tool/{tool_name}", timeout=10.0)
                return _json_or_err(r, "delete_custom_tool")

            if name == "call_custom_tool":
                tool_name = str(args.get("tool_name", "")).strip()
                params = args.get("params", {})
                if not tool_name:
                    return _text("call_custom_tool: 'tool_name' is required")
                r = await c.post(
                    f"{TOOLKIT_URL}/call/{tool_name}",
                    json={"params": params},
                    timeout=30.0,
                )
                return _json_or_err(r, "call_custom_tool")

            # ----------------------------------------------------------------
            if name == "get_errors":
                limit = max(1, min(int(args.get("limit", 50)), 200))
                params: dict = {"limit": limit}
                svc = str(args.get("service", "")).strip()
                if svc:
                    params["service"] = svc
                r = await c.get(f"{DATABASE_URL}/errors/recent", params=params, timeout=10.0)
                if r.status_code != 200:
                    return _text(f"get_errors: upstream returned {r.status_code} — {r.text[:300]}")
                data = r.json()
                errors = data.get("errors", [])
                if not errors:
                    return _text("No errors logged yet." + (f" (service={svc})" if svc else ""))
                lines = [f"Recent errors ({len(errors)}):"]
                for e in errors:
                    ts = str(e.get("logged_at", ""))[:19].replace("T", " ")
                    lines.append(
                        f"  [{ts}] [{e.get('level', '?')}] {e.get('service', '?')}: {e.get('message', '')}"
                        + (f"\n    detail: {e['detail']}" if e.get("detail") else "")
                    )
                return _text("\n".join(lines))

            # ----------------------------------------------------------------
            # Image manipulation tools (PIL — no HTTP calls needed)
            # ----------------------------------------------------------------
            if name in ("image_crop", "image_zoom", "image_scan", "image_enhance"):
                if not _HAS_PIL:
                    return _text(f"{name}: Pillow is not installed in this container.")
                path = str(args.get("path", "")).strip()
                local = _resolve_image_path(path)
                if not local:
                    return _text(f"{name}: image not found — tried '{path}' in {BROWSER_WORKSPACE}")

                with _PilImage.open(local) as _src:
                    img = _ImageOps.exif_transpose(_src.copy())

                w, h = img.size
                try:
                    left   = int(args.get("left",   0))
                    top    = int(args.get("top",    0))
                    right  = int(args.get("right",  w))
                    bottom = int(args.get("bottom", h))
                except (ValueError, TypeError) as _e:
                    return _text(f"{name}: invalid coordinate — {_e}")
                # Clamp to image bounds and make positive
                if right  <= 0: right  = w
                if bottom <= 0: bottom = h
                right  = min(right,  w)
                bottom = min(bottom, h)
                box = (max(0, left), max(0, top), right, bottom)

                if name == "image_crop":
                    if box[2] <= box[0] or box[3] <= box[1]:
                        return _text(f"image_crop: invalid box {box} for {w}×{h} image")
                    result = img.crop(box)
                    rw, rh = result.size
                    summary = (
                        f"Cropped ({box[0]},{box[1]}) → ({box[2]},{box[3]})\n"
                        f"Original: {w}×{h}  Result: {rw}×{rh}  Source: {os.path.basename(path)}"
                    )
                    return _pil_to_blocks(result, summary, save_prefix="cropped")

                if name == "image_zoom":
                    scale = max(1.1, min(float(args.get("scale", 2.0)), 8.0))
                    region = img.crop(box) if box != (0, 0, w, h) else img
                    rw, rh = region.size
                    zoomed = _GpuImg.resize(region, int(rw * scale), int(rh * scale))
                    zw, zh = zoomed.size
                    summary = (
                        f"Zoomed {scale:.1f}× from ({box[0]},{box[1]})→({box[2]},{box[3]})\n"
                        f"Region: {rw}×{rh}  Output: {zw}×{zh}  Source: {os.path.basename(path)}\n"
                        f"Backend: {_GpuImg.backend()}"
                    )
                    return _pil_to_blocks(zoomed, summary, quality=92, save_prefix="zoomed")

                if name == "image_scan":
                    region = img.crop(box) if box != (0, 0, w, h) else img
                    rw, rh = region.size
                    # Auto-upscale small regions so fine text is legible for the vision model
                    if rw < 800:
                        up = max(2, 800 // max(rw, 1))
                        region = _GpuImg.resize(region, rw * up, rh * up)
                        rw, rh = region.size
                    # Greyscale → contrast boost → sharpen → unsharp mask (GPU-accelerated)
                    region = _GpuImg.to_grayscale(region)
                    region = _GpuImg.enhance_contrast(region.convert("RGB"), 2.5)
                    region = _GpuImg.enhance_sharpness(region, 3.0)
                    region = _GpuImg.sharpen(region, radius=1, percent=150, threshold=3)
                    summary = (
                        f"Scan-enhanced for text reading — greyscale + contrast×2.5 + sharpen×3.0\n"
                        f"Region: ({box[0]},{box[1]})→({box[2]},{box[3]})  Output: {rw}×{rh}\n"
                        f"Source: {os.path.basename(path)}  Backend: {_GpuImg.backend()}\n"
                        f"Read all text visible in this image."
                    )
                    return _pil_to_blocks(region, summary, quality=95, save_prefix="scanned")

                if name == "image_enhance":
                    contrast   = max(0.5, min(float(args.get("contrast",   1.5)), 4.0))
                    sharpness  = max(0.5, min(float(args.get("sharpness",  1.5)), 4.0))
                    brightness = max(0.5, min(float(args.get("brightness", 1.0)), 3.0))
                    grayscale  = bool(args.get("grayscale", False))
                    result = img.copy()
                    if grayscale:
                        result = result.convert("L").convert("RGB")
                    result = _GpuImg.enhance_contrast(result, contrast)
                    result = _GpuImg.enhance_sharpness(result, sharpness)
                    result = _ImageEnhance.Brightness(result).enhance(brightness)  # PIL only
                    summary = (
                        f"Enhanced: contrast={contrast:.1f} sharpness={sharpness:.1f} "
                        f"brightness={brightness:.1f}"
                        + (" grayscale" if grayscale else "") + "\n"
                        f"Size: {w}×{h}  Source: {os.path.basename(path)}\n"
                        f"Backend: {_GpuImg.backend()}"
                    )
                    return _pil_to_blocks(result, summary, save_prefix="enhanced")

            # ----------------------------------------------------------------
            # image_stitch — combine multiple images into one canvas
            # ----------------------------------------------------------------
            if name == "image_stitch":
                if not _HAS_PIL:
                    return _text("image_stitch: Pillow is not installed.")
                paths = [str(p) for p in args.get("paths", [])]
                if len(paths) < 2:
                    return _text("image_stitch: at least 2 paths are required")
                paths = paths[:8]
                direction = str(args.get("direction", "vertical")).lower()
                try:
                    gap = max(0, int(args.get("gap", 0)))
                except (ValueError, TypeError):
                    gap = 0
                images: list["_PilImage.Image"] = []
                for p in paths:
                    loc = _resolve_image_path(p)
                    if not loc:
                        return _text(f"image_stitch: image not found — '{p}'")
                    with _PilImage.open(loc) as im:
                        images.append(_ImageOps.exif_transpose(im.convert("RGB").copy()))
                if direction == "horizontal":
                    total_w = sum(im.width for im in images) + gap * (len(images) - 1)
                    max_h   = max(im.height for im in images)
                    canvas  = _PilImage.new("RGB", (total_w, max_h), (255, 255, 255))
                    x = 0
                    for im in images:
                        # Center shorter images vertically within the canvas
                        y_off = (max_h - im.height) // 2
                        canvas.paste(im, (x, y_off))
                        x += im.width + gap
                else:
                    max_w   = max(im.width for im in images)
                    total_h = sum(im.height for im in images) + gap * (len(images) - 1)
                    canvas  = _PilImage.new("RGB", (max_w, total_h), (255, 255, 255))
                    y = 0
                    for im in images:
                        # Center narrower images horizontally within the canvas
                        x_off = (max_w - im.width) // 2
                        canvas.paste(im, (x_off, y))
                        y += im.height + gap
                summary = (
                    f"Stitched {len(images)} images ({direction})\n"
                    f"Output size: {canvas.width}×{canvas.height}"
                )
                return _pil_to_blocks(canvas, summary, save_prefix="stitched")

            # ----------------------------------------------------------------
            # image_diff — highlight pixel-level differences between two images
            # ----------------------------------------------------------------
            if name == "image_diff":
                if not _HAS_PIL:
                    return _text("image_diff: Pillow is not installed.")
                path_a = str(args.get("path_a", "")).strip()
                path_b = str(args.get("path_b", "")).strip()
                loc_a = _resolve_image_path(path_a)
                loc_b = _resolve_image_path(path_b)
                if not loc_a:
                    return _text(f"image_diff: path_a not found — '{path_a}'")
                if not loc_b:
                    return _text(f"image_diff: path_b not found — '{path_b}'")
                try:
                    amplify = max(1.0, min(float(args.get("amplify", 3.0)), 10.0))
                except (ValueError, TypeError):
                    amplify = 3.0
                with _PilImage.open(loc_a) as ia:
                    img_a = _ImageOps.exif_transpose(ia.convert("RGB").copy())
                with _PilImage.open(loc_b) as ib:
                    img_b = _ImageOps.exif_transpose(ib.convert("RGB").copy())
                diff = _GpuImg.diff(img_a, img_b)   # handles size mismatch internally
                diff_l = diff.convert("L")
                diff_l = _ImageEnhance.Brightness(diff_l).enhance(amplify)
                white = _PilImage.new("RGB", img_a.size, (255, 255, 255))
                red   = _PilImage.new("RGB", img_a.size, (220, 30, 30))
                result = _PilImage.composite(red, white, diff_l)
                summary = (
                    f"Pixel diff: {os.path.basename(path_a)} vs {os.path.basename(path_b)}\n"
                    f"Amplify: {amplify:.1f}×  Size: {img_a.width}×{img_a.height}\n"
                    f"Backend: {_GpuImg.backend()}  Red pixels = changed regions."
                )
                return _pil_to_blocks(result, summary, save_prefix="diff")

            # ----------------------------------------------------------------
            # image_annotate — draw bounding boxes and labels on a screenshot
            # ----------------------------------------------------------------
            if name == "image_annotate":
                if not _HAS_PIL:
                    return _text("image_annotate: Pillow is not installed.")
                path = str(args.get("path", "")).strip()
                boxes = args.get("boxes", [])
                if not path:
                    return _text("image_annotate: 'path' is required")
                if not boxes:
                    return _text("image_annotate: 'boxes' list is required")
                loc = _resolve_image_path(path)
                if not loc:
                    return _text(f"image_annotate: image not found — '{path}'")
                try:
                    outline_width = max(1, int(args.get("outline_width", 3)))
                except (ValueError, TypeError):
                    outline_width = 3
                with _PilImage.open(loc) as src:
                    img = _ImageOps.exif_transpose(src.convert("RGB").copy())
                # Parse boxes into tuples for GpuImageProcessor.annotate()
                box_tuples: list[tuple[int, int, int, int]] = []
                box_labels: list[str] = []
                box_colors: list[tuple[int, int, int]] = []
                for box in boxes:
                    try:
                        bx_l = int(box.get("left",   0))
                        bx_t = int(box.get("top",    0))
                        bx_r = int(box.get("right",  img.width))
                        bx_b = int(box.get("bottom", img.height))
                    except (ValueError, TypeError):
                        continue  # skip malformed box rather than crash
                    box_tuples.append((bx_l, bx_t, bx_r, bx_b))
                    box_labels.append(str(box.get("label", "")))
                    raw_col = str(box.get("color", "#FF3333")).lstrip("#")
                    try:
                        r = int(raw_col[0:2], 16)
                        g = int(raw_col[2:4], 16)
                        b = int(raw_col[4:6], 16)
                        box_colors.append((r, g, b))
                    except Exception:
                        box_colors.append((255, 51, 51))
                # Use the first box color for all (GpuImageProcessor uses a single color param)
                # For multi-color support, fall back to PIL draw loop
                if len(set(str(c) for c in box_colors)) == 1:
                    primary_color = box_colors[0] if box_colors else (255, 51, 51)
                    img = _GpuImg.annotate(img, box_tuples, box_labels,
                                           color=primary_color, thickness=outline_width)
                else:
                    # Multiple distinct colors — use PIL for per-box color control
                    draw = _ImageDraw.Draw(img)
                    for (bx_l, bx_t, bx_r, bx_b), label, col in zip(box_tuples, box_labels, box_colors):
                        for i in range(outline_width):
                            draw.rectangle([bx_l - i, bx_t - i, bx_r + i, bx_b + i], outline=col)
                        if label:
                            tx, ty = bx_l + 2, max(0, bx_t - 18)
                            draw.rectangle([tx - 2, ty - 2, tx + len(label) * 7 + 2, ty + 16], fill=col)
                            draw.text((tx, ty), label, fill="white")
                n = len(boxes)
                summary = (
                    f"Annotated {n} bounding box{'es' if n != 1 else ''} on {os.path.basename(path)}\n"
                    f"Size: {img.width}×{img.height}  Backend: {_GpuImg.backend()}"
                )
                return _pil_to_blocks(img, summary, save_prefix="annotated")

            # ----------------------------------------------------------------
            # face_recognize — detect faces + optional reference matching
            # ----------------------------------------------------------------
            if name == "face_recognize":
                if not _HAS_PIL:
                    return _text("face_recognize: Pillow is not installed.")
                path = str(args.get("path", "")).strip()
                if not path:
                    return _text("face_recognize: 'path' is required")
                loc = _resolve_image_path(path)
                if not loc:
                    return _text(f"face_recognize: image not found — '{path}'")

                with _PilImage.open(loc) as src:
                    img = _ImageOps.exif_transpose(src.convert("RGB").copy())

                if not _HAS_CV2 or _cv2 is None or _np is None:
                    return _pil_to_blocks(
                        img,
                        "face_recognize: OpenCV is unavailable in this environment. "
                        "Returning source image without detections.",
                        save_prefix="face_recognize",
                    )

                anime_mode = str(args.get("style", "auto")).lower() == "anime"
                try:
                    min_face_size = max(12, min(int(args.get("min_face_size", 20 if anime_mode else 40)), 2048))
                except (ValueError, TypeError):
                    min_face_size = 20 if anime_mode else 40
                try:
                    min_neighbors = max(1, min(int(args.get("min_neighbors", 2 if anime_mode else 5)), 12))
                except (ValueError, TypeError):
                    min_neighbors = 2 if anime_mode else 5
                try:
                    scale_factor = max(1.01, min(float(args.get("scale_factor", 1.05 if anime_mode else 1.1)), 1.5))
                except (ValueError, TypeError):
                    scale_factor = 1.05 if anime_mode else 1.1
                try:
                    match_threshold = max(0.0, min(float(args.get("match_threshold", 0.82)), 1.0))
                except (ValueError, TypeError):
                    match_threshold = 0.82
                annotate = bool(args.get("annotate", True))

                faces = _detect_faces(
                    img,
                    min_face_size=min_face_size,
                    min_neighbors=min_neighbors,
                    scale_factor=scale_factor,
                    anime=anime_mode,
                )
                detection_method = "cascade"

                # YOLO person-detection fallback when cascades find nothing
                if not faces:
                    try:
                        yolo_faces = await _detect_faces_yolo_fallback(img)
                        if yolo_faces:
                            faces = yolo_faces
                            detection_method = "yolo-person-fallback"
                    except Exception:
                        pass

                out_img = img.copy()
                style_label = "anime" if anime_mode else "auto"
                anime_cascade_ok = os.path.exists(_ANIME_CASCADE_PATH)
                lines = [
                    f"Detected {len(faces)} face(s) in {os.path.basename(path)} "
                    f"(style={style_label}, method={detection_method}, "
                    f"min_face_size={min_face_size}px, "
                    f"min_neighbors={min_neighbors}, scale_factor={scale_factor:.2f}).",
                    (
                        f"Vision backend: {_GpuImg.backend()}  "
                        f"(OpenCL={'on' if _CV2_ACCEL_STATUS.get('opencl_use') else 'off'}, "
                        f"CUDA devices={int(_CV2_ACCEL_STATUS.get('cuda_devices', 0))}, "
                        f"anime_cascade={'loaded' if anime_cascade_ok else 'missing'})"
                    ),
                ]
                if len(faces) == 0 and not anime_mode:
                    lines.append(
                        "Tip: if this is an anime or illustrated image, retry with style='anime'."
                    )

                ref_path = str(args.get("reference_path", "")).strip()
                similarities: list[float] = []
                matched_flags: list[bool] = []
                if ref_path:
                    ref_loc = _resolve_image_path(ref_path)
                    if not ref_loc:
                        lines.append(f"Reference image not found: {ref_path}")
                    else:
                        with _PilImage.open(ref_loc) as ref_src:
                            ref_img = _ImageOps.exif_transpose(ref_src.convert("RGB").copy())
                        ref_faces = _detect_faces(ref_img, min_face_size=min_face_size)
                        if not ref_faces:
                            lines.append(
                                f"Reference image '{os.path.basename(ref_path)}' has no detectable faces."
                            )
                        else:
                            ref_vec = _face_embedding(ref_img, ref_faces[0])
                            if ref_vec is None:
                                lines.append("Reference face embedding failed.")
                            else:
                                for box in faces:
                                    face_vec = _face_embedding(img, box)
                                    if face_vec is None:
                                        score = 0.0
                                    else:
                                        score = _face_similarity(ref_vec, face_vec)
                                    similarities.append(score)
                                    matched_flags.append(score >= match_threshold)
                                if similarities:
                                    best_idx = max(range(len(similarities)), key=lambda i: similarities[i])
                                    best_score = similarities[best_idx]
                                    lines.append(
                                        f"Best match: face #{best_idx + 1} similarity={best_score:.3f} "
                                        f"({'MATCH' if best_score >= match_threshold else 'NO MATCH'}) "
                                        f"threshold={match_threshold:.2f}."
                                    )

                if annotate:
                    draw = _ImageDraw.Draw(out_img)
                    for idx, (x, y, w, h) in enumerate(faces, start=1):
                        is_match = matched_flags[idx - 1] if idx - 1 < len(matched_flags) else False
                        color = (64, 176, 89) if is_match else (230, 57, 70)
                        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
                        if idx - 1 < len(similarities):
                            label = f"Face {idx} {similarities[idx - 1]:.2f}"
                        else:
                            label = f"Face {idx}"
                        tx, ty = x + 2, max(0, y - 18)
                        draw.rectangle([tx - 2, ty - 2, tx + len(label) * 7 + 2, ty + 16], fill=color)
                        draw.text((tx, ty), label, fill="white")

                return _pil_to_blocks(
                    out_img,
                    "\n".join(lines),
                    save_prefix="face_recognize",
                )

            # ----------------------------------------------------------------
            # page_extract — JS-based structured data extraction from live page
            # ----------------------------------------------------------------
            if name == "page_extract":
                include = list(args.get("include") or ["links", "headings", "tables", "images", "meta", "text"])
                max_links = max(1, int(args.get("max_links", 50)))
                max_text  = max(100, int(args.get("max_text", 3000)))
                js = (
                    "(function(){"
                    "var out={};"
                    "var inc=" + json.dumps(include) + ";"
                    "function has(k){return inc.indexOf(k)!==-1;}"
                    "if(has('meta')){"
                    " var m={};document.querySelectorAll('meta[name],meta[property]').forEach(function(el){"
                    "  var k=el.getAttribute('name')||el.getAttribute('property');"
                    "  if(k)m[k]=el.getAttribute('content')||'';"
                    " });out.meta=m;out.title=document.title;}"
                    "if(has('headings')){"
                    " out.headings=Array.from(document.querySelectorAll('h1,h2,h3,h4')).slice(0,50).map(function(h){"
                    "  return{tag:h.tagName,text:h.innerText.trim()};});}"
                    "if(has('links')){"
                    " out.links=Array.from(document.querySelectorAll('a[href]')).slice(0," + str(max_links) + ").map(function(a){"
                    "  return{text:a.innerText.trim().slice(0,120),href:a.href};});}"
                    "if(has('images')){"
                    " out.images=Array.from(document.querySelectorAll('img[src]')).slice(0,30).map(function(i){"
                    "  return{src:i.src,alt:i.alt||''};});}"
                    "if(has('tables')){"
                    " out.tables=Array.from(document.querySelectorAll('table')).slice(0,5).map(function(tbl){"
                    "  return Array.from(tbl.querySelectorAll('tr')).slice(0,20).map(function(tr){"
                    "   return Array.from(tr.querySelectorAll('th,td')).map(function(td){return td.innerText.trim();});});});}"
                    "if(has('text')){"
                    " out.text=(document.body?document.body.innerText:'').slice(0," + str(max_text) + ");}"
                    "return JSON.stringify(out);})()"
                )
                try:
                    eval_r = await c.post(f"{BROWSER_URL}/eval", json={"code": js}, timeout=15.0)
                    raw = eval_r.json().get("result", "{}")
                    data = json.loads(raw) if isinstance(raw, str) else (raw or {})
                    lines: list[str] = []
                    if data.get("title"):
                        lines.append(f"Title: {data.get('title')}")
                    for k, v in list((data.get("meta") or {}).items())[:10]:
                        lines.append(f"  meta[{k}]: {str(v)[:120]}")
                    if "headings" in data:
                        lines.append(f"\nHeadings ({len(data['headings'])}):")
                        for hd in data["headings"]:
                            lines.append(f"  {hd.get('tag','?')}: {str(hd.get('text',''))[:120]}")
                    if "links" in data:
                        lines.append(f"\nLinks ({len(data['links'])}):")
                        for lk in data["links"]:
                            lines.append(f"  [{str(lk.get('text',''))[:60]}] → {str(lk.get('href',''))[:120]}")
                    if "images" in data:
                        lines.append(f"\nImages ({len(data['images'])}):")
                        for im in data["images"][:10]:
                            lines.append(f"  {str(im.get('src',''))[:100]}  alt='{str(im.get('alt',''))[:60]}'")
                    if "tables" in data:
                        lines.append(f"\nTables ({len(data['tables'])}):")
                        for ti, tbl in enumerate(data["tables"]):
                            lines.append(f"  Table {ti+1} ({len(tbl)} rows):")
                            for row in tbl[:5]:
                                lines.append("    | " + " | ".join(str(cell)[:30] for cell in row))
                    if "text" in data:
                        lines.append(f"\nText excerpt:\n{data['text'][:1500]}")
                    return _text("\n".join(lines) or "No data extracted.")
                except Exception as exc:
                    return _text(f"page_extract failed: {exc}")

            # ----------------------------------------------------------------
            # extract_article — clean readable article text from a URL
            # ----------------------------------------------------------------
            if name == "extract_article":
                url = str(args.get("url", "")).strip()
                if not url:
                    return _text("extract_article: 'url' is required")
                max_chars = max(500, int(args.get("max_chars", 8000)))
                try:
                    nav_r = await asyncio.wait_for(
                        c.post(f"{BROWSER_URL}/navigate", json={"url": url}),
                        timeout=25.0,
                    )
                    data = nav_r.json()
                    title     = data.get("title", "")
                    content   = data.get("content", "")
                    final_url = data.get("url", url)
                    clean_lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
                    clean = "\n".join(clean_lines)[:max_chars]
                    header = f"Title: {title}\nURL:   {final_url}\n\n" if title else f"URL: {final_url}\n\n"
                    return _text(header + clean)
                except Exception as exc:
                    return _text(f"extract_article failed: {exc}")

            # ----------------------------------------------------------------
            # page_scrape — scroll + lazy-load + full DOM text extraction
            # ----------------------------------------------------------------
            if name == "page_scrape":
                url         = str(args.get("url", "")).strip()
                max_scrolls = max(1, min(int(args.get("max_scrolls", 10)), 30))
                wait_ms     = max(100, min(int(args.get("wait_ms", 500)), 3000))
                max_chars   = max(500, min(int(args.get("max_chars", 16000)), 64000))
                include_links = bool(args.get("include_links", False))
                payload: dict = {
                    "max_scrolls": max_scrolls,
                    "wait_ms":     wait_ms,
                    "max_chars":   max_chars,
                    "include_links": include_links,
                }
                if url:
                    payload["url"] = url
                try:
                    timeout_s = max_scrolls * (wait_ms / 1000.0) + 30.0
                    scrape_r = await asyncio.wait_for(
                        c.post(f"{BROWSER_URL}/scrape", json=payload),
                        timeout=timeout_s,
                    )
                    data = scrape_r.json()
                except Exception as exc:
                    return _text(f"page_scrape: browser unreachable — {exc}")
                if data.get("error"):
                    return _text(f"page_scrape error: {data['error']}")
                title   = data.get("title", "")
                content = data.get("content", "")
                final_url = data.get("url", url)
                steps   = data.get("scroll_steps", 0)
                grew    = data.get("content_grew_on_scroll", False)
                height  = data.get("final_page_height", 0)
                chars   = data.get("char_count", len(content))
                header = (
                    f"Title: {title}\nURL: {final_url}\n"
                    f"Scrolled: {steps} steps | Page height: {height}px"
                    + (" | lazy content grew" if grew else "")
                    + f" | {chars} chars extracted\n\n"
                )
                lines = [l.strip() for l in content.splitlines() if l.strip()]
                body = "\n".join(lines)
                result_text = header + body
                if include_links and data.get("links"):
                    link_lines = [f"[{lk.get('text','')[:60]}] → {lk.get('href','')[:120]}"
                                  for lk in data["links"][:100]]
                    result_text += f"\n\nLinks ({len(data['links'])}):\n" + "\n".join(link_lines)
                return _text(result_text)

            # ----------------------------------------------------------------
            # page_images — comprehensive image URL extraction from live page
            # ----------------------------------------------------------------
            if name == "page_images":
                url         = str(args.get("url", "")).strip()
                scroll      = bool(args.get("scroll", True))
                max_scrolls = max(1, min(int(args.get("max_scrolls", 3)), 20))
                if not url:
                    return _text("page_images: 'url' is required")
                payload = {"url": url, "scroll": scroll, "max_scrolls": max_scrolls}
                timeout_s = max_scrolls * 2.0 + 30.0
                try:
                    pi_r = await asyncio.wait_for(
                        c.post(f"{BROWSER_URL}/page_images", json=payload),
                        timeout=timeout_s,
                    )
                    data = pi_r.json()
                except Exception as exc:
                    return _text(f"page_images: browser unreachable — {exc}")
                if data.get("error"):
                    return _text(f"page_images error: {data['error']}")
                imgs   = data.get("images", [])
                count  = data.get("count", len(imgs))
                title  = data.get("title", "")
                final_url = data.get("url", url)
                lines = [f"Found {count} images on: {final_url}  ({title})"]
                for img in imgs:
                    line = f"[{img.get('type','?')}] {img.get('url','')}"
                    if img.get("alt"):
                        line += f"  alt={img.get('alt','')!r}"
                    if img.get("natural_w"):
                        line += f"  {img.get('natural_w',0)}×{img.get('natural_h',0)}"
                    elif img.get("srcset_width"):
                        line += f"  {img.get('srcset_width','')}w"
                    lines.append(line)
                return _text("\n".join(lines))

            # ----------------------------------------------------------------
            # bulk_screenshot — parallel screenshots of multiple URLs
            # ----------------------------------------------------------------
            if name == "bulk_screenshot":
                urls = [str(u).strip() for u in args.get("urls", []) if str(u).strip()]
                if not urls:
                    return _text("bulk_screenshot: 'urls' list is required")
                urls = urls[:6]

                async def _single_shot(shot_url: str, idx: int) -> list[dict[str, Any]]:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    container_path = f"/workspace/bulk_{idx}_{ts}.png"
                    try:
                        shot_r = await c.post(
                            f"{BROWSER_URL}/screenshot",
                            json={"url": shot_url, "path": container_path},
                            timeout=25.0,
                        )
                        shot_data = shot_r.json()
                        cp    = shot_data.get("path", container_path)
                        title = shot_data.get("title", shot_url)
                        fname = os.path.basename(cp)
                        lp    = os.path.join(BROWSER_WORKSPACE, fname)
                        summary = f"[{idx+1}/{len(urls)}] {title}\n{shot_url}\nFile: {fname}"
                        if os.path.isfile(lp):
                            return _image_blocks(cp, summary)
                        return _text(f"[{idx+1}] {shot_url} — screenshot missing: {shot_data.get('error', 'unknown')}")
                    except Exception as exc:
                        return _text(f"[{idx+1}] {shot_url} — failed: {exc}")

                tasks = [_single_shot(u, i) for i, u in enumerate(urls)]
                results = await asyncio.gather(*tasks)
                combined: list[dict[str, Any]] = []
                for blocks in results:
                    combined.extend(blocks)
                return combined

            # ----------------------------------------------------------------
            # scroll_screenshot — full-page capture via scroll + stitch
            # ----------------------------------------------------------------
            if name == "scroll_screenshot":
                if not _HAS_PIL:
                    return _text("scroll_screenshot: Pillow is not installed in this container.")
                url = str(args.get("url", "")).strip() or None
                max_scrolls = max(1, min(int(args.get("max_scrolls", 5)), 10))
                overlap     = max(0, int(args.get("scroll_overlap", 100)))
                if url:
                    try:
                        await asyncio.wait_for(
                            c.post(f"{BROWSER_URL}/navigate", json={"url": url}),
                            timeout=20.0,
                        )
                    except Exception as exc:
                        return _text(f"scroll_screenshot: navigate failed: {exc}")
                try:
                    h_r  = await c.post(f"{BROWSER_URL}/eval",
                                        json={"code": "document.documentElement.scrollHeight"},
                                        timeout=5.0)
                    page_h = int(h_r.json().get("result", 0) or 0)
                    vp_r = await c.post(f"{BROWSER_URL}/eval",
                                        json={"code": "window.innerHeight"},
                                        timeout=5.0)
                    vp_h = int(vp_r.json().get("result", 800) or 800)
                except Exception:
                    page_h, vp_h = 0, 800
                step = max(vp_h - overlap, 100)
                frames: list["_PilImage.Image"] = []
                ts_base = datetime.now().strftime("%Y%m%d_%H%M%S")
                for i in range(max_scrolls):
                    scroll_y = i * step
                    if page_h > 0 and scroll_y >= page_h:
                        break
                    try:
                        await c.post(f"{BROWSER_URL}/eval",
                                     json={"code": f"window.scrollTo(0, {scroll_y})"},
                                     timeout=5.0)
                        await asyncio.sleep(0.3)
                        container_path = f"/workspace/scroll_{i}_{ts_base}.png"
                        shot_r = await c.post(f"{BROWSER_URL}/screenshot",
                                              json={"path": container_path},
                                              timeout=15.0)
                        cp    = shot_r.json().get("path", container_path)
                        fname = os.path.basename(cp)
                        lp    = os.path.join(BROWSER_WORKSPACE, fname)
                        if os.path.isfile(lp):
                            with _PilImage.open(lp) as fr:
                                frames.append(_ImageOps.exif_transpose(fr).convert("RGB").copy())
                    except Exception:
                        break
                if not frames:
                    return _text("scroll_screenshot: no frames captured.")
                total_h = sum(f.height for f in frames)
                max_w   = max(f.width  for f in frames)
                canvas  = _PilImage.new("RGB", (max_w, total_h), (255, 255, 255))
                y = 0
                for fr in frames:
                    canvas.paste(fr, (0, y))
                    y += fr.height
                summary = (
                    f"Full-page scroll screenshot: {len(frames)} frames stitched\n"
                    f"Canvas: {canvas.width}×{canvas.height}"
                    + (f"  URL: {url}" if url else "")
                )
                return _pil_to_blocks(canvas, summary, quality=85, save_prefix="fullpage")

            # ----------------------------------------------------------------
            # image_generate — text-to-image via LM Studio / OpenAI-compatible API
            # ----------------------------------------------------------------
            if name == "image_generate":
                prompt = str(args.get("prompt", "")).strip()
                if not prompt:
                    return _text("image_generate: 'prompt' is required")
                # Fast model-availability check — avoids a 180 s timeout when nothing is loaded
                if not await ModelRegistry.get().has_image_gen(c):
                    loaded = [m.get("id","?") for m in await ModelRegistry.get().models(c)]
                    loaded_str = ", ".join(loaded) if loaded else "none"
                    return _text(
                        f"image_generate: no image-generation model is loaded in LM Studio.\n"
                        f"Currently loaded: {loaded_str}\n"
                        "→ Load FLUX, SDXL, or another diffusion model in LM Studio and try again."
                    )
                model = str(args.get("model", IMAGE_GEN_MODEL)).strip() or None
                size  = str(args.get("size", "512x512")).strip()
                n     = max(1, min(int(args.get("n", 1)), 4))
                neg   = str(args.get("negative_prompt", "")).strip() or None
                steps = args.get("steps")
                gs    = args.get("guidance_scale")
                seed  = args.get("seed")
                payload: dict = {"prompt": prompt, "n": n, "size": size, "response_format": "b64_json"}
                if model:        payload["model"]                = model
                if neg:          payload["negative_prompt"]       = neg
                if steps:        payload["num_inference_steps"]   = int(steps)
                if gs is not None: payload["guidance_scale"]      = float(gs)
                if seed is not None and int(seed) >= 0: payload["seed"] = int(seed)
                gen_url = f"{IMAGE_GEN_BASE_URL}/v1/images/generations"
                try:
                    r = await c.post(gen_url, json=payload, timeout=180.0)
                    r.raise_for_status()
                    data = r.json()
                except Exception as exc:
                    return _text(
                        f"image_generate failed: {exc}\n"
                        f"Endpoint: {gen_url}\n"
                        "→ Make sure an image generation model (FLUX, SDXL, etc.) is loaded in LM Studio.\n"
                        "→ Or set IMAGE_GEN_BASE_URL to point at your Automatic1111/ComfyUI instance."
                    )
                images = data.get("data", [])
                if not images:
                    err = data.get("error", {})
                    msg = err.get("message", str(data)) if isinstance(err, dict) else str(err)
                    return _text(f"image_generate: no images returned — {msg}")
                blocks: list[dict[str, Any]] = []
                for i, img_data in enumerate(images):
                    b64 = img_data.get("b64_json", "")
                    img_url = img_data.get("url", "")
                    revised = img_data.get("revised_prompt", "")
                    save_note = ""
                    if b64 and os.path.isdir(BROWSER_WORKSPACE):
                        try:
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                            fname = f"generated_{i}_{ts}.jpg"
                            lp = os.path.join(BROWSER_WORKSPACE, fname)
                            if _HAS_PIL:
                                with _PilImage.open(_io.BytesIO(base64.b64decode(b64))) as gi:
                                    gi.convert("RGB").save(lp, format="JPEG", quality=95)
                            else:
                                with open(lp, "wb") as fh:
                                    fh.write(base64.b64decode(b64))
                            save_note = f"\n→ Saved as: {fname}  (pass as 'path' in image_crop/zoom/upscale/annotate)"
                        except Exception:
                            pass
                    summary = f"Generated image {i+1}/{n}\nPrompt: {prompt[:200]}"
                    if revised: summary += f"\nRevised: {revised[:150]}"
                    summary += save_note
                    if b64:
                        if _HAS_PIL:
                            try:
                                with _PilImage.open(_io.BytesIO(base64.b64decode(b64))) as gi:
                                    gi = gi.convert("RGB")
                                    if gi.width > 1280 or gi.height > 1280:
                                        gi.thumbnail((1280, 1280), _PilImage.LANCZOS)
                                    buf = _io.BytesIO()
                                    gi.save(buf, format="JPEG", quality=90)
                                b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
                                mime = "image/jpeg"
                            except Exception:
                                mime = "image/png"
                        else:
                            mime = "image/png"
                        blocks.extend([{"type": "text", "text": summary},
                                        {"type": "image", "data": b64, "mimeType": mime}])
                    elif img_url:
                        blocks.append({"type": "text", "text": f"{summary}\nImage URL: {img_url}"})
                return blocks if blocks else _text("image_generate: no image data in response")

            # ----------------------------------------------------------------
            # image_edit — img2img / inpainting via /v1/images/edits
            # ----------------------------------------------------------------
            if name == "image_edit":
                path   = str(args.get("path", "")).strip()
                prompt = str(args.get("prompt", "")).strip()
                if not path:   return _text("image_edit: 'path' is required")
                if not prompt: return _text("image_edit: 'prompt' is required")
                local = _resolve_image_path(path)
                if not local:
                    return _text(f"image_edit: image not found — '{path}' in {BROWSER_WORKSPACE}")
                # Fast model check before spending time on file I/O + 90 s timeout
                if not await ModelRegistry.get().has_image_gen(c):
                    loaded = [m.get("id","?") for m in await ModelRegistry.get().models(c)]
                    loaded_str = ", ".join(loaded) if loaded else "none"
                    return _text(
                        f"image_edit: no image-generation model is loaded in LM Studio.\n"
                        f"Currently loaded: {loaded_str}\n"
                        "→ Load FLUX, SDXL, or another diffusion model and try again."
                    )
                model    = str(args.get("model", IMAGE_GEN_MODEL)).strip() or None
                neg      = str(args.get("negative_prompt", "")).strip() or None
                strength = max(0.0, min(float(args.get("strength", 0.75)), 1.0))
                n        = max(1, min(int(args.get("n", 1)), 4))
                size     = str(args.get("size", "")).strip() or None
                ext  = os.path.splitext(local)[1].lower()
                mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                            ".png": "image/png", ".webp": "image/webp"}
                img_mime = mime_map.get(ext, "image/jpeg")
                with open(local, "rb") as fh:
                    img_bytes = fh.read()
                edit_url = f"{IMAGE_GEN_BASE_URL}/v1/images/edits"
                form: dict = {"prompt": prompt, "n": str(n), "response_format": "b64_json",
                               "strength": str(strength)}
                if model:       form["model"]            = model
                if neg:         form["negative_prompt"]   = neg
                if size:        form["size"]              = size
                try:
                    r = await c.post(edit_url,
                                     files={"image": (os.path.basename(local), img_bytes, img_mime)},
                                     data=form, timeout=180.0)
                    if r.status_code in (404, 405, 422):
                        # Fallback: JSON body with base64-encoded image
                        b64_in = base64.standard_b64encode(img_bytes).decode("ascii")
                        json_pl: dict = {"prompt": prompt, "n": n, "response_format": "b64_json",
                                          "strength": strength,
                                          "image": f"data:{img_mime};base64,{b64_in}"}
                        if model: json_pl["model"] = model
                        if neg:   json_pl["negative_prompt"] = neg
                        if size:  json_pl["size"] = size
                        r = await c.post(edit_url, json=json_pl, timeout=180.0)
                    r.raise_for_status()
                    data = r.json()
                except Exception as exc:
                    return _text(
                        f"image_edit failed: {exc}\n"
                        f"Endpoint: {edit_url}\n"
                        "→ img2img support varies by model. Make sure a compatible model is loaded.\n"
                        "→ For full img2img support, point IMAGE_GEN_BASE_URL at Automatic1111/ComfyUI."
                    )
                images = data.get("data", [])
                if not images:
                    err = data.get("error", {})
                    msg = err.get("message", str(data)) if isinstance(err, dict) else str(err)
                    return _text(f"image_edit: no images returned — {msg}")
                blocks_e: list[dict[str, Any]] = []
                for i, img_data in enumerate(images):
                    b64 = img_data.get("b64_json", "")
                    revised = img_data.get("revised_prompt", "")
                    save_note = ""
                    if b64 and os.path.isdir(BROWSER_WORKSPACE):
                        try:
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                            fname = f"edited_{i}_{ts}.jpg"
                            lp = os.path.join(BROWSER_WORKSPACE, fname)
                            if _HAS_PIL:
                                with _PilImage.open(_io.BytesIO(base64.b64decode(b64))) as ei:
                                    ei.convert("RGB").save(lp, format="JPEG", quality=95)
                            else:
                                with open(lp, "wb") as fh:
                                    fh.write(base64.b64decode(b64))
                            save_note = f"\n→ Saved as: {fname}"
                        except Exception:
                            pass
                    summary = (f"Edited image {i+1}/{n}  Source: {os.path.basename(path)}\n"
                               f"Prompt: {prompt[:200]}  strength={strength:.2f}")
                    if revised: summary += f"\nRevised: {revised[:150]}"
                    summary += save_note
                    if b64:
                        if _HAS_PIL:
                            try:
                                with _PilImage.open(_io.BytesIO(base64.b64decode(b64))) as ei:
                                    ei = ei.convert("RGB")
                                    if ei.width > 1280 or ei.height > 1280:
                                        ei.thumbnail((1280, 1280), _PilImage.LANCZOS)
                                    buf = _io.BytesIO()
                                    ei.save(buf, format="JPEG", quality=90)
                                b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
                                mime = "image/jpeg"
                            except Exception:
                                mime = "image/png"
                        else:
                            mime = "image/png"
                        blocks_e.extend([{"type": "text", "text": summary},
                                          {"type": "image", "data": b64, "mimeType": mime}])
                return blocks_e if blocks_e else _text("image_edit: no image data in response")

            # ----------------------------------------------------------------
            # image_remix — GPU AI style transfer via /v1/images/edits
            #               OOP: ModelRegistry (model check) + ImageRenderer (encode)
            # ----------------------------------------------------------------
            if name == "image_remix":
                if not _HAS_PIL:
                    return _text("image_remix: Pillow is not installed.")
                path_rm   = str(args.get("path", "")).strip()
                prompt_rm = str(args.get("prompt", "")).strip()
                if not path_rm:
                    return _text("image_remix: 'path' is required")
                if not prompt_rm:
                    return _text("image_remix: 'prompt' is required")
                local_rm = _resolve_image_path(path_rm)
                if not local_rm:
                    return _text(f"image_remix: image not found — '{path_rm}' in {BROWSER_WORKSPACE}")
                strength_rm = max(0.1, min(float(args.get("strength", 0.65)), 1.0))
                n_rm        = max(1, min(int(args.get("n", 1)), 4))
                # Fast model check — friendly error instead of 90 s timeout
                if not await ModelRegistry.get().has_image_gen(c):
                    loaded_rm = [m.get("id","?") for m in await ModelRegistry.get().models(c)]
                    return _text(
                        f"image_remix: no image-generation model loaded in LM Studio.\n"
                        f"Currently loaded: {', '.join(loaded_rm) or 'none'}\n"
                        "→ Load FLUX, SDXL, or another diffusion model and try again."
                    )
                # Encode source image as JPEG for multipart upload
                with _PilImage.open(local_rm) as src_rm:
                    img_rm = _ImageOps.exif_transpose(src_rm).convert("RGB")
                in_buf_rm = _io.BytesIO()
                img_rm.save(in_buf_rm, format="JPEG", quality=92)
                in_buf_rm.seek(0)
                rm_model  = IMAGE_GEN_MODEL.strip() or None
                form_rm: dict[str, str] = {
                    "prompt":          prompt_rm,
                    "n":               str(n_rm),
                    "response_format": "b64_json",
                    "strength":        str(strength_rm),
                }
                if rm_model:
                    form_rm["model"] = rm_model
                try:
                    resp_rm = await asyncio.wait_for(
                        c.post(
                            f"{IMAGE_GEN_BASE_URL}/v1/images/edits",
                            files={"image": ("source.jpg", in_buf_rm, "image/jpeg")},
                            data=form_rm,
                            timeout=120.0,
                        ),
                        timeout=125.0,
                    )
                    resp_rm.raise_for_status()
                    data_rm = resp_rm.json()
                except Exception as exc_rm:
                    return _text(
                        f"image_remix failed: {exc_rm}\n"
                        "→ Ensure an image-generation model supports /v1/images/edits in LM Studio."
                    )
                blocks_rm: list[dict[str, Any]] = []
                src_name_rm = os.path.basename(path_rm)
                for idx_rm, item_rm in enumerate(data_rm.get("data", [])):
                    b64_rm = item_rm.get("b64_json", "")
                    if not b64_rm:
                        continue
                    try:
                        ri_rm = _PilImage.open(_io.BytesIO(base64.b64decode(b64_rm))).convert("RGB")
                        summary_rm = (
                            f"Remix [{idx_rm+1}/{n_rm}]: '{prompt_rm[:80]}'\n"
                            f"Source: {src_name_rm}  {img_rm.width}×{img_rm.height}"
                            f"  strength={strength_rm:.2f}  GPU: {GpuDetector.name()}"
                        )
                        blocks_rm.extend(_renderer.encode(ri_rm, summary_rm, save_prefix="remix"))
                    except Exception:
                        continue
                return blocks_rm if blocks_rm else _text(
                    f"image_remix: no image data in response for prompt '{prompt_rm[:60]}'"
                )

            # ----------------------------------------------------------------
            # image_upscale — GPU AI primary (NVIDIA / Intel Arc via LM Studio);
            #                  CPU LANCZOS is fallback only if GPU is unavailable
            #                  or the model call fails.
            #                  OOP via GpuDetector + GpuUpscaler + ImageRenderer.
            # ----------------------------------------------------------------
            if name == "image_upscale":
                if not _HAS_PIL:
                    return _text("image_upscale: Pillow is not installed.")
                path = str(args.get("path", "")).strip()
                if not path:
                    return _text("image_upscale: 'path' is required")
                local = _resolve_image_path(path)
                if not local:
                    return _text(f"image_upscale: image not found — '{path}' in {BROWSER_WORKSPACE}")
                resolved_name = os.path.basename(local)
                alias_note = ""
                if os.path.basename(path) != resolved_name:
                    alias_note = (
                        f"\nInput alias '{os.path.basename(path)}' resolved to workspace image "
                        f"'{resolved_name}'."
                    )
                try:
                    scale = max(1.1, min(float(args.get("scale", 2.0)), 8.0))
                except (ValueError, TypeError):
                    scale = 2.0
                # sharpen: accept bool or string "false"/"true"
                _sharpen_raw = args.get("sharpen", True)
                sharpen = str(_sharpen_raw).lower() not in ("false", "0", "no")
                # gpu param: None/unset → auto (GPU primary), False → force CPU only
                gpu_arg = args.get("gpu", None)
                force_cpu = (gpu_arg is False or str(gpu_arg).lower() == "false")
                with _PilImage.open(local) as src:
                    img = _ImageOps.exif_transpose(src).convert("RGB")
                ow, oh = img.size
                # ── PRIMARY: GPU AI upscale (NVIDIA + Intel Arc via LM Studio) ──
                if not force_cpu and IMAGE_GEN_BASE_URL:
                    gpu_info  = GpuDetector.detect()
                    # Fast check: skip 90 s timeout if no image-gen model is loaded
                    _has_gen  = await ModelRegistry.get().has_image_gen(c)
                    upscaler  = GpuUpscaler(IMAGE_GEN_BASE_URL, IMAGE_GEN_MODEL)
                    ai_result = await upscaler.upscale(img, c) if _has_gen else None
                    if ai_result is not None:
                        ai_w, ai_h = ai_result.size
                        summary = (
                            f"GPU AI upscale ({gpu_info['name']})\n"
                            f"  {ow}×{oh} → {ai_w}×{ai_h}\n"
                            f"Source: {resolved_name}{alias_note}"
                        )
                        return _renderer.encode(ai_result, summary, save_prefix="upscaled",
                                                max_w=4096, max_h=4096)
                    # GPU call failed — warn and fall through to CPU LANCZOS
                    cpu_note = (
                        f"⚠ GPU AI upscale failed (no model loaded or LM Studio unreachable). "
                        f"GPU detected: {gpu_info['name']}. "
                        f"Falling back to CPU LANCZOS.\n"
                    )
                else:
                    cpu_note = ""
                # ── FALLBACK (CPU): LANCZOS ──────────────────────────────────────
                # Safety cap: clamp scale so no output dimension exceeds 8192 px
                max_dim = max(ow, oh)
                if max_dim * scale > 8192:
                    scale = 8192 / max_dim
                    capped = True
                else:
                    capped = False
                nw, nh = int(ow * scale), int(oh * scale)
                upscaled = _GpuImg.resize(img, nw, nh)
                if sharpen:
                    upscaled = _GpuImg.enhance_sharpness(upscaled, factor=1.4)
                    upscaled = _GpuImg.sharpen(upscaled, radius=0.5, percent=80, threshold=2)
                summary = (
                    cpu_note
                    + f"{_GpuImg.backend()} upscale {scale:.2f}×  {ow}×{oh} → {nw}×{nh}"
                    + (" + sharpen" if sharpen else "")
                    + (" [scale capped to 8192px]" if capped else "")
                    + f"\nSource: {resolved_name}{alias_note}"
                )
                return _renderer.encode(upscaled, summary, save_prefix="upscaled",
                                        max_w=4096, max_h=4096)

            # ----------------------------------------------------------------
            if name == "browser_save_images":
                raw_urls = args.get("urls", [])
                if isinstance(raw_urls, str):
                    raw_urls = [u.strip() for u in raw_urls.split(",") if u.strip()]
                if not raw_urls:
                    return _text("browser_save_images: 'urls' is required (list or comma-separated string)")
                prefix = str(args.get("prefix", "image")).strip() or "image"
                max_imgs = int(args.get("max", 20))
                try:
                    save_r = await c.post(
                        f"{BROWSER_URL}/save_images",
                        json={"urls": raw_urls, "prefix": prefix, "max": max_imgs},
                        timeout=120.0,
                    )
                    save_data = save_r.json()
                except Exception as exc:
                    return _text(f"browser_save_images: browser unreachable — {exc}")
                saved = save_data.get("saved", [])
                errors = save_data.get("errors", [])
                if not saved and errors:
                    errs = "; ".join(e.get("error", "?") for e in errors[:3])
                    return _text(f"browser_save_images: all downloads failed — {errs}")
                blocks: list = []
                lines = [f"Downloaded {len(saved)} image(s)" +
                         (f"  ({len(errors)} failed)" if errors else "")]
                for item in saved:
                    fname = os.path.basename(item.get("path", ""))
                    host_path = f"/docker/human_browser/workspace/{fname}"
                    local_path = os.path.join(BROWSER_WORKSPACE, fname)
                    size_kb = item.get("size", 0) // 1024
                    lines.append(f"  [{item.get('index','?')}] {fname}  {size_kb} KB  {item.get('url','')[:80]}")
                    if os.path.isfile(local_path) and len(blocks) < 6:
                        blocks.extend(_image_blocks(item["path"], f"[{item.get('index','?')}] {fname}"))
                blocks.insert(0, {"type": "text", "text": "\n".join(lines)})
                return blocks

            # ----------------------------------------------------------------
            if name == "browser_download_page_images":
                url = str(args.get("url", "")).strip() or None
                if url:
                    try:
                        await c.post(f"{BROWSER_URL}/navigate", json={"url": url}, timeout=20.0)
                    except Exception as exc:
                        return _text(f"browser_download_page_images: navigate failed — {exc}")
                filter_q = str(args.get("filter", "")).strip() or None
                prefix = str(args.get("prefix", "image")).strip() or "image"
                max_imgs = int(args.get("max", 20))
                payload: dict = {"max": max_imgs, "prefix": prefix}
                if filter_q:
                    payload["filter"] = filter_q
                try:
                    dl_r = await c.post(
                        f"{BROWSER_URL}/download_page_images",
                        json=payload,
                        timeout=120.0,
                    )
                    dl_data = dl_r.json()
                except Exception as exc:
                    return _text(f"browser_download_page_images: browser unreachable — {exc}")
                saved = dl_data.get("saved", [])
                errors = dl_data.get("errors", [])
                applied_filter = dl_data.get("filter")
                filter_note = f" (filter: '{applied_filter}')" if applied_filter else ""
                if not saved and errors:
                    errs = "; ".join(e.get("error", "?") for e in errors[:3])
                    return _text(f"browser_download_page_images: all downloads failed{filter_note} — {errs}")
                if not saved:
                    return _text(f"browser_download_page_images: no images found on page{filter_note}")
                blocks = []
                lines = [
                    f"Downloaded {len(saved)} image(s){filter_note}" +
                    (f"  ({len(errors)} failed)" if errors else "")
                ]
                for item in saved:
                    fname = os.path.basename(item.get("path", ""))
                    local_path = os.path.join(BROWSER_WORKSPACE, fname)
                    size_kb = item.get("size", 0) // 1024
                    lines.append(f"  [{item.get('index','?')}] {fname}  {size_kb} KB  {item.get('url','')[:80]}")
                    if os.path.isfile(local_path) and len(blocks) < 9:
                        blocks.extend(_image_blocks(item["path"], f"[{item.get('index','?')}] {fname}"))
                blocks.insert(0, {"type": "text", "text": "\n".join(lines)})
                return blocks

            # ----------------------------------------------------------------
            # image_search — find images by query; return multiple inline
            # ----------------------------------------------------------------
            if name == "image_search":
                raw_query  = str(args.get("query", "")).strip()
                query, normalize_note = _normalize_search_query(raw_query)
                img_count  = max(1, min(int(args.get("count", 4)), 20))
                img_offset = max(0, int(args.get("offset", 0)))
                if not query:
                    return _text("image_search: 'query' is required")

                import hashlib as _hl2
                import json as _js2
                from urllib.parse import quote_plus as _qp, urlparse as _up2, \
                    parse_qs as _pqs2, unquote as _uq2
                qwords = set(_search_terms(query))
                _preferred_domains = _query_preferred_domains(query)
                _domain_allow_raw = args.get("domains", [])
                if isinstance(_domain_allow_raw, str):
                    _domain_allow_raw = [d.strip() for d in _domain_allow_raw.split(",") if d.strip()]
                _domain_allow = {
                    str(d).strip().lower().removeprefix("www.")
                    for d in (_domain_allow_raw if isinstance(_domain_allow_raw, list) else [])
                    if str(d).strip()
                }

                _GOOD_D  = ("wikia.nocookie.net", "imgur.com", "redd.it",
                            "prydwen.gg", "fandom.com", "iopwiki.com", "cdn.")
                _SKIP_P  = ("/16px-", "/25px-", "/32px-", "/48px-", "favicon",
                            "logo", "icon", "avatar", "pixel.gif", "button",
                            "ytimg.com", "yt3.ggpht", "explicit.bing.net")
                _SKIP_T1 = ("youtube.com", "youtu.be", "vimeo.com", "dailymotion.com",
                            "twitch.tv", "pinterest.com", "instagram.com",
                            "deviantart.com", "artstation.com")

                # ── Seen-URL dedup via memory service ─────────────────────────
                _qhash        = _hl2.sha256(query.lower().encode()).hexdigest()[:16]
                _mem_key      = f"imgsr:{_qhash}"
                _seen_urls:   set[str]  = set()
                _seen_hashes: list[str] = []   # intra-call phash dedup
                _url_to_hash: dict[str, str] = {}
                try:
                    mem_r = await asyncio.wait_for(
                        c.get(f"{MEMORY_URL}/recall", params={"key": _mem_key}),
                        timeout=5.0,
                    )
                    rdata = mem_r.json()
                    entries = rdata.get("entries") or []
                    if rdata.get("found") and entries:
                        _seen_urls = set(_js2.loads(entries[0]["value"]))
                except Exception:
                    pass

                _norm_subject = query.lower().strip()

                # ── Query expansion ────────────────────────────────────────────
                _EXPANSIONS = {
                    "gfl2": "Girls Frontline 2", "gfl": "Girls Frontline",
                    "hsr":  "Honkai Star Rail",  "gi":  "Genshin Impact",
                    "hk":   "Honkai Impact",
                }

                def _expand_queries(q: str) -> list[str]:
                    variants = [q]
                    q_lower  = q.lower()
                    for short, full in _EXPANSIONS.items():
                        if short in q_lower and full.lower() not in q_lower:
                            variants.append(q.replace(short, full).replace(short.upper(), full))
                            break
                    if "artwork" not in q_lower and "art" not in q_lower:
                        variants.append(q + " artwork")
                    return variants[:3]

                def _unwrap_thumb(url: str) -> str:
                    if "/images/thumb/" in url:
                        no_thumb = url.replace("/images/thumb/", "/images/", 1)
                        m = re.search(r"/\d+px-[^?]*", no_thumb)
                        if m:
                            return no_thumb[:m.start()]
                    return url

                def _img_referer(url: str) -> str:
                    u = url.lower()
                    if "pbs.twimg.com" in u or "media.twimg.com" in u:
                        return "https://twitter.com/"
                    if "wikia.nocookie.net" in u or "static.fandom.com" in u:
                        return "https://www.fandom.com/"
                    if "iopwiki.com" in u:
                        return "https://iopwiki.com/"
                    if "prydwen.gg" in u:
                        return "https://www.prydwen.gg/"
                    return "https://www.google.com/"

                def _score(img: dict) -> int:
                    url_l = img.get("url", "").lower()
                    alt_l = img.get("alt", "").lower()
                    if _url_has_explicit_content(url_l, alt_l):
                        return -999
                    if any(p in url_l for p in _SKIP_P):
                        return -999
                    host_l = _domain_from_url(url_l)
                    if _domain_allow and not any(
                        host_l == d or host_l.endswith(f".{d}") for d in _domain_allow
                    ):
                        return -999
                    s  = sum(2 for w in qwords if w in url_l)
                    s += sum(5 for w in qwords if w in alt_l)
                    if "pbs.twimg.com" in url_l or "media.twimg.com" in url_l:
                        s += 10
                    else:
                        s += sum(6 for d in _GOOD_D if d in url_l)
                    s += sum(8 for d in _preferred_domains if host_l == d or host_l.endswith(f".{d}"))
                    if img.get("type") in ("srcset", "picture"):
                        s += 2
                    nw = img.get("natural_w", 0)
                    if nw >= 500:
                        s += 4
                    elif nw >= 300:
                        s += 2
                    return s

                def _domain_of(url: str) -> str:
                    try:
                        h = _up2(url).hostname or ""
                        return h.removeprefix("www.")
                    except Exception:
                        return url

                def _apply_domain_cap(candidates: list[dict], max_per_domain: int = 2) -> list[dict]:
                    counts: dict[str, int] = {}
                    result = []
                    for cand in candidates:
                        d = _domain_of(cand.get("url", ""))
                        if counts.get(d, 0) < max_per_domain:
                            result.append(cand)
                            counts[d] = counts.get(d, 0) + 1
                    return result

                async def _fetch_render(url: str) -> list[dict] | None:
                    """Fetch, resize, phash-dedup, and encode one image URL."""
                    if not url or not url.startswith("http"):
                        return None
                    url  = _unwrap_thumb(url)
                    if _url_has_explicit_content(url):
                        return None
                    host = _domain_from_url(url)
                    if _domain_allow and not any(host == d or host.endswith(f".{d}") for d in _domain_allow):
                        return None
                    hdrs = {
                        "User-Agent":      _BROWSER_HEADERS["User-Agent"],
                        "Accept":          "image/webp,image/apng,image/*,*/*;q=0.8",
                        "Accept-Language": _BROWSER_HEADERS["Accept-Language"],
                        "Referer":         _img_referer(url),
                    }
                    try:
                        img_r = await asyncio.wait_for(
                            c.get(url, headers=hdrs, follow_redirects=True),
                            timeout=12.0,
                        )
                        if img_r.status_code != 200:
                            return None
                        ct = img_r.headers.get("content-type", "").split(";")[0].strip()
                        if not ct.startswith("image/"):
                            return None
                        raw = img_r.content
                        if len(raw) < 10_240:
                            return None
                        if _HAS_PIL:
                            try:
                                with _PilImage.open(_io.BytesIO(raw)) as pil:
                                    if pil.width < 150 or pil.height < 150:
                                        return None
                                    pil = pil.convert("RGB")
                                    if _is_low_information_image(pil):
                                        return None
                                    # dHash intra-call dedup: skip visually identical images
                                    img_hash = _dhash(pil)
                                    if img_hash and any(
                                        _hamming(img_hash, h) < 8 for h in _seen_hashes
                                    ):
                                        return None
                                    if img_hash:
                                        _seen_hashes.append(img_hash)
                                        _url_to_hash[url] = img_hash
                                    if pil.width > 1280 or pil.height > 1024:
                                        pil.thumbnail((1280, 1024), _PilImage.LANCZOS)
                                    buf = _io.BytesIO()
                                    pil.save(buf, format="JPEG", quality=85)
                                    raw, ct = buf.getvalue(), "image/jpeg"
                            except Exception:
                                pass
                        b64 = base64.standard_b64encode(raw).decode("ascii")
                        return [
                            {"type": "text",  "text": f"Image: {url}\nQuery: {query}"},
                            {"type": "image", "data": b64, "mimeType": ct},
                        ]
                    except Exception:
                        return None

                # ── DB-first: return confirmed images if we have enough cached ───
                try:
                    db_limit = max(img_count * 4, (img_count + img_offset) * 3)
                    db_r = await asyncio.wait_for(
                        c.get(f"{DATABASE_URL}/images/search",
                              params={"subject": _norm_subject, "limit": db_limit, "offset": 0}),
                        timeout=3.0,
                    )
                    db_imgs: list[dict] = []
                    db_seen: set[str] = set()
                    for raw_img in db_r.json().get("images", []):
                        db_url = _unwrap_thumb(str(raw_img.get("url", "")))
                        if not db_url or db_url in _seen_urls or db_url in db_seen:
                            continue
                        db_imgs.append({**raw_img, "url": db_url})
                        db_seen.add(db_url)
                    db_imgs = db_imgs[img_offset:]
                    if len(db_imgs) >= img_count:
                        db_blocks: list[dict] = []
                        for dbi in db_imgs:
                            fb = await _fetch_render(dbi["url"])
                            if fb:
                                db_blocks.extend(fb)
                            if len(db_blocks) >= img_count * 2:
                                break
                        if db_blocks:
                            return db_blocks
                except Exception:
                    pass

                # ── Collect ALL candidates across all query variants + both tiers ──
                all_candidates: list[dict] = []
                _ddg_hosts = ("duckduckgo.com", "ddg.gg", "duck.co")

                # Tier 0: SearXNG image search — direct CDN URLs, no browser needed.
                # This is the most reliable tier: SearXNG aggregates Google Images +
                # Bing Images and returns the raw img_src URLs (Artstation CDN, Twitter
                # media, wixmp, etc.) so _fetch_render can download them directly.
                for variant_q in _expand_queries(query):
                    sx_img_cands = await _searxng_image_search(c, variant_q, max_results=30)
                    all_candidates.extend(sx_img_cands)

                # Tier 1: DDG web search → page_images on top article pages
                for variant_q in _expand_queries(query):
                    try:
                        sr = await asyncio.wait_for(
                            c.get(
                                f"https://html.duckduckgo.com/html/?q={_qp(variant_q)}&kp=-2",
                                headers=_BROWSER_HEADERS,
                                follow_redirects=True,
                            ),
                            timeout=12.0,
                        )
                        raw_enc = re.findall(r'uddg=(https?%3A[^&"\'>\s]+)', sr.text)
                        t1_urls: list[str] = []
                        t1_seen: set[str]  = set()
                        for enc in raw_enc:
                            dec = _url_unquote(enc)
                            if any(d in dec for d in _ddg_hosts):
                                continue
                            if dec not in t1_seen:
                                t1_seen.add(dec)
                                t1_urls.append(dec)
                        t1_urls = [u for u in t1_urls if not any(d in u for d in _SKIP_T1)]
                        for page_url in t1_urls[:5]:
                            try:
                                pi_r = await asyncio.wait_for(
                                    c.post(f"{BROWSER_URL}/page_images",
                                           json={"url": page_url, "scroll": True, "max_scrolls": 1}),
                                    timeout=35.0,
                                )
                                page_imgs = pi_r.json().get("images", [])
                                if len(page_imgs) < 3:
                                    continue
                                for pi in page_imgs:
                                    pu = _unwrap_thumb(pi.get("url", ""))
                                    if _url_has_explicit_content(pu, pi.get("alt", "")):
                                        continue
                                    if _domain_allow:
                                        phost = _domain_from_url(pu)
                                        if not any(phost == d or phost.endswith(f".{d}") for d in _domain_allow):
                                            continue
                                    all_candidates.append(pi)
                            except Exception:
                                continue
                    except Exception:
                        pass

                # Tier 2: DDG image-search page → decode proxy URLs
                try:
                    t2_url = f"https://duckduckgo.com/?q={_qp(query)}&iax=images&ia=images&kp=-2"
                    pi2_r = await asyncio.wait_for(
                        c.post(f"{BROWSER_URL}/page_images",
                               json={"url": t2_url, "scroll": True, "max_scrolls": 3}),
                        timeout=45.0,
                    )
                    for img2 in pi2_r.json().get("images", []):
                        u2 = img2.get("url", "")
                        if "external-content.duckduckgo.com" in u2:
                            try:
                                params2 = _pqs2(_up2(u2).query)
                                orig2   = _uq2(params2.get("u", [""])[0])
                                if orig2.startswith("http"):
                                    if not _url_has_explicit_content(orig2, img2.get("alt", "")):
                                        if not _domain_allow:
                                            all_candidates.append({**img2, "url": orig2})
                                        else:
                                            h2 = _domain_from_url(orig2)
                                            if any(h2 == d or h2.endswith(f".{d}") for d in _domain_allow):
                                                all_candidates.append({**img2, "url": orig2})
                                    continue
                            except Exception:
                                pass
                        if _url_has_explicit_content(u2, img2.get("alt", "")):
                            continue
                        if _domain_allow:
                            h2 = _domain_from_url(u2)
                            if not any(h2 == d or h2.endswith(f".{d}") for d in _domain_allow):
                                continue
                        all_candidates.append(img2)
                except Exception:
                    pass

                # Tier 3: Bing Images — different corpus from DDG, different CDNs
                try:
                    bing_url = f"https://www.bing.com/images/search?q={_qp(query)}&form=HDRSC2&first=1"
                    pi3_r = await asyncio.wait_for(
                        c.post(f"{BROWSER_URL}/page_images",
                               json={"url": bing_url, "scroll": True, "max_scrolls": 2}),
                        timeout=45.0,
                    )
                    for img3 in pi3_r.json().get("images", []):
                        u3 = img3.get("url", "")
                        # Bing wraps images in /th?id=... proxy URLs — skip thumbnails
                        if (
                            "bing.com/th" in u3
                            or ".mm.bing.net" in u3
                            or "explicit.bing.net" in u3
                        ):
                            continue
                        if _url_has_explicit_content(u3, img3.get("alt", "")):
                            continue
                        if _domain_allow:
                            h3 = _domain_from_url(u3)
                            if not any(h3 == d or h3.endswith(f".{d}") for d in _domain_allow):
                                continue
                        all_candidates.append(img3)
                except Exception:
                    pass

                # ── Score, dedup seen URLs + intra-call dupes, domain-cap, offset ──
                _intra_seen: set[str] = set()
                fresh: list[dict] = []
                for cand in all_candidates:
                    u = _unwrap_thumb(cand.get("url", ""))
                    if _score(cand) >= 0 and u not in _seen_urls and u not in _intra_seen:
                        fresh.append(cand)
                        _intra_seen.add(u)
                sorted_cands = _apply_domain_cap(
                    sorted(fresh, key=_score, reverse=True), max_per_domain=2
                )
                sorted_cands = sorted_cands[img_offset:]
                fetch_pool   = sorted_cands[:img_count * 3]

                # Domain-restricted fallback: search site pages directly via Bing,
                # then extract page images from those URLs.
                if not fetch_pool and _domain_allow:
                    site_pages: list[str] = []
                    for dom in sorted(_domain_allow):
                        for variant_q in _expand_queries(query):
                            site_q = f"site:{dom} {variant_q}"
                            try:
                                rb = await asyncio.wait_for(
                                    c.get(
                                        f"https://www.bing.com/search?q={_qp(site_q)}&setlang=en-US",
                                        headers=_BROWSER_HEADERS,
                                        follow_redirects=True,
                                    ),
                                    timeout=12.0,
                                )
                                for su, _st in _extract_bing_links(rb.text, max_results=6):
                                    shost = _domain_from_url(su)
                                    if shost == dom or shost.endswith(f".{dom}"):
                                        site_pages.append(su)
                            except Exception:
                                continue
                    for page_url in list(dict.fromkeys(site_pages))[:10]:
                        try:
                            pi_r = await asyncio.wait_for(
                                c.post(
                                    f"{BROWSER_URL}/page_images",
                                    json={"url": page_url, "scroll": True, "max_scrolls": 1},
                                ),
                                timeout=35.0,
                            )
                            page_imgs = pi_r.json().get("images", [])
                            for pi in page_imgs:
                                pu = _unwrap_thumb(pi.get("url", ""))
                                if _url_has_explicit_content(pu, pi.get("alt", "")):
                                    continue
                                phost = _domain_from_url(pu)
                                if not any(phost == d or phost.endswith(f".{d}") for d in _domain_allow):
                                    continue
                                all_candidates.append(pi)
                        except Exception:
                            continue
                    # Re-score with newly harvested domain-specific candidates.
                    _intra_seen = set()
                    fresh = []
                    for cand in all_candidates:
                        u = _unwrap_thumb(cand.get("url", ""))
                        if _score(cand) >= 0 and u not in _seen_urls and u not in _intra_seen:
                            fresh.append(cand)
                            _intra_seen.add(u)
                    sorted_cands = _apply_domain_cap(
                        sorted(fresh, key=_score, reverse=True), max_per_domain=2
                    )
                    sorted_cands = sorted_cands[img_offset:]
                    fetch_pool = sorted_cands[:img_count * 3]

                if not fetch_pool:
                    extra = f"\n{normalize_note}" if normalize_note else ""
                    if _domain_allow:
                        extra += f"\nDomain filter: {', '.join(sorted(_domain_allow))}"
                    return _text(
                        f"image_search: no image found for '{query}'.\n"
                        "Try: page_images on a specific URL, or refine the query."
                        + extra
                    )

                fetch_results = await asyncio.gather(
                    *[_fetch_render(cand["url"]) for cand in fetch_pool],
                    return_exceptions=True,
                )

                # ── Vision confirm pass (best-effort, parallel) ────────────────
                async def _maybe_confirm(
                    blocks: list[dict] | None,
                    url: str,
                    img_hash: str,
                ) -> list[dict] | None:
                    if not blocks:
                        return None
                    img_b64 = next(
                        (b["data"] for b in blocks if b.get("type") == "image"), ""
                    )
                    if not img_b64:
                        return blocks
                    is_match, desc, conf = await _vision_confirm(img_b64, query, c, phash=img_hash)
                    if not is_match:
                        return None
                    # Fire-and-forget: persist confirmed image to DB (own client to avoid closed-client bug)
                    _store_url = url
                    _store_subj = _norm_subject
                    _store_desc = desc
                    _store_hash = img_hash
                    _store_conf = conf
                    async def _store_img():
                        try:
                            async with httpx.AsyncClient(timeout=10) as _sc:
                                await _sc.post(
                                    f"{DATABASE_URL}/images/store",
                                    json={
                                        "url": _store_url, "subject": _store_subj,
                                        "description": _store_desc, "phash": _store_hash,
                                        "quality_score": _store_conf,
                                    },
                                )
                        except Exception:
                            pass
                    asyncio.create_task(_store_img())
                    return blocks

                confirmed_results = await asyncio.gather(
                    *[
                        _maybe_confirm(
                            res if not isinstance(res, Exception) else None,
                            _unwrap_thumb(cand["url"]),
                            _url_to_hash.get(_unwrap_thumb(cand["url"]), ""),
                        )
                        for cand, res in zip(fetch_pool, fetch_results)
                    ],
                    return_exceptions=True,
                )

                output_blocks: list[dict] = []
                returned_urls: list[str]  = []
                for cand, res in zip(fetch_pool, confirmed_results):
                    if isinstance(res, Exception) or res is None:
                        continue
                    output_blocks.extend(res)
                    returned_urls.append(_unwrap_thumb(cand["url"]))
                    if len(returned_urls) >= img_count:
                        break

                # ── Persist seen URLs for next call ────────────────────────────
                if returned_urls:
                    try:
                        merged = list(_seen_urls | set(returned_urls))
                        await asyncio.wait_for(
                            c.post(f"{MEMORY_URL}/store",
                                   json={"key": _mem_key, "value": _js2.dumps(merged),
                                         "ttl_seconds": 3600}),
                            timeout=5.0,
                        )
                    except Exception:
                        pass

                if output_blocks:
                    if normalize_note:
                        output_blocks.insert(0, {"type": "text", "text": normalize_note})
                    return output_blocks
                extra = f"\n{normalize_note}" if normalize_note else ""
                if _domain_allow:
                    extra += f"\nDomain filter: {', '.join(sorted(_domain_allow))}"
                return _text(
                    f"image_search: no image found for '{query}'.\n"
                    "Try: page_images on a specific URL, or refine the query."
                    + extra
                )

            # ----------------------------------------------------------------
            # tts — text-to-speech via LM Studio /v1/audio/speech
            # ----------------------------------------------------------------
            if name == "tts":
                import os as _os_tts
                text_in  = str(args.get("text", "")).strip()
                if not text_in:
                    return _text("tts: 'text' is required")
                voice  = str(args.get("voice", "alloy")).strip() or "alloy"
                speed  = max(0.25, min(4.0, float(args.get("speed", 1.0))))
                fmt    = str(args.get("format", "mp3")).strip() or "mp3"
                payload_tts: dict = {
                    "input": text_in,
                    "voice": voice,
                    "speed": speed,
                    "response_format": fmt,
                }
                if IMAGE_GEN_MODEL:
                    payload_tts["model"] = IMAGE_GEN_MODEL
                async with httpx.AsyncClient(timeout=60.0) as hc_tts:
                    try:
                        r_tts = await hc_tts.post(
                            f"{IMAGE_GEN_BASE_URL}/v1/audio/speech",
                            json=payload_tts,
                        )
                        if r_tts.status_code != 200:
                            return _text(
                                f"tts: LM Studio returned {r_tts.status_code}. "
                                "Make sure a TTS-capable model is loaded."
                            )
                        audio_bytes = r_tts.content
                        # LM Studio may return 200 with a JSON error body instead of audio
                        if audio_bytes and audio_bytes[:1] in (b"{", b"["):
                            try:
                                err_json = __import__("json").loads(audio_bytes)
                                err_msg = err_json.get("error", audio_bytes.decode(errors="replace")[:200])
                            except Exception:
                                err_msg = audio_bytes.decode(errors="replace")[:200]
                            return _text(f"tts: service error — {err_msg}. Make sure a TTS-capable model is loaded.")
                    except Exception as exc:
                        return _text(f"tts: request failed — {exc}")
                # Save to workspace
                ts_tts  = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname   = f"tts_{ts_tts}.{fmt}"
                ws_path = _os_tts.path.join(BROWSER_WORKSPACE, fname)
                try:
                    with open(ws_path, "wb") as fh:
                        fh.write(audio_bytes)
                    return _text(
                        f"tts: saved {len(audio_bytes):,} bytes → {ws_path}\n"
                        f"voice={voice} speed={speed} format={fmt}"
                    )
                except Exception as exc:
                    return _text(f"tts: audio generated ({len(audio_bytes):,} bytes) but could not save: {exc}")

            # ----------------------------------------------------------------
            # embed_store — embed text + persist to DB
            # ----------------------------------------------------------------
            if name == "embed_store":
                key_e    = str(args.get("key", "")).strip()
                content_e = str(args.get("content", "")).strip()
                topic_e  = str(args.get("topic", "")).strip()
                if not key_e:     return _text("embed_store: 'key' is required")
                if not content_e: return _text("embed_store: 'content' is required")
                emb_payload = {"input": [content_e], "model": await _get_embed_model()}
                async with httpx.AsyncClient(timeout=30.0) as hc_emb:
                    try:
                        r_emb = await hc_emb.post(
                            f"{IMAGE_GEN_BASE_URL}/v1/embeddings", json=emb_payload
                        )
                        r_emb.raise_for_status()
                        emb_data = r_emb.json().get("data", [])
                        if not emb_data:
                            return _text("embed_store: LM Studio returned empty embedding data — is an embedding model loaded?")
                        embedding = emb_data[0].get("embedding", [])
                        if not embedding:
                            return _text("embed_store: LM Studio returned empty embedding vector")
                    except Exception as exc:
                        return _text(f"embed_store: LM Studio embedding failed — {exc}")
                    # Store in DB
                    try:
                        r_db = await hc_emb.post(
                            f"{DATABASE_URL}/embeddings/store",
                            json={"key": key_e, "content": content_e,
                                  "embedding": embedding,
                                  "model": IMAGE_GEN_MODEL, "topic": topic_e},
                        )
                        r_db.raise_for_status()
                        return _text(
                            f"embed_store: stored key='{key_e}' "
                            f"dims={len(embedding)} topic='{topic_e}'"
                        )
                    except Exception as exc:
                        return _text(f"embed_store: DB store failed — {exc}")

            # ----------------------------------------------------------------
            # embed_search — semantic search over stored embeddings
            # ----------------------------------------------------------------
            if name == "embed_search":
                query_es  = str(args.get("query", "")).strip()
                try:
                    limit_es = max(1, min(20, int(args.get("limit", 5))))
                except (ValueError, TypeError):
                    return _text("embed_search: 'limit' must be an integer")
                topic_es  = str(args.get("topic", "")).strip()
                if not query_es: return _text("embed_search: 'query' is required")
                emb_payload_s = {"input": [query_es], "model": await _get_embed_model()}
                async with httpx.AsyncClient(timeout=30.0) as hc_es:
                    try:
                        r_es = await hc_es.post(
                            f"{IMAGE_GEN_BASE_URL}/v1/embeddings", json=emb_payload_s
                        )
                        r_es.raise_for_status()
                        es_data = r_es.json().get("data", [])
                        if not es_data:
                            return _text("embed_search: LM Studio returned empty embedding data — is an embedding model loaded?")
                        q_vec = es_data[0].get("embedding", [])
                        if not q_vec:
                            return _text("embed_search: LM Studio returned empty embedding vector")
                    except Exception as exc:
                        return _text(f"embed_search: LM Studio embedding failed — {exc}")
                    body_es: dict = {"embedding": q_vec, "limit": limit_es}
                    if topic_es:
                        body_es["topic"] = topic_es
                    try:
                        r_db_es = await hc_es.post(
                            f"{DATABASE_URL}/embeddings/search", json=body_es
                        )
                        r_db_es.raise_for_status()
                        results_es = r_db_es.json().get("results", [])
                    except Exception as exc:
                        return _text(f"embed_search: DB search failed — {exc}")
                if not results_es:
                    return _text(f"embed_search: no results found for '{query_es}'")
                lines = [f"embed_search: {len(results_es)} result(s) for '{query_es}'\n"]
                for i, r in enumerate(results_es, 1):
                    lines.append(
                        f"{i}. [{r.get('similarity', 0):.3f}] key={r.get('key','')} "
                        f"topic={r.get('topic','')}\n   {r.get('content','')[:200]}"
                    )
                return _text("\n".join(lines))

            # ----------------------------------------------------------------
            # code_run — sandboxed Python subprocess execution
            # ----------------------------------------------------------------
            if name == "code_run":
                import tempfile as _tmpfile_cr
                import os as _os_cr
                import time as _time_cr
                code_cr    = str(args.get("code", "")).strip()
                pkgs_cr    = args.get("packages") or []
                timeout_cr = max(1, min(120, int(args.get("timeout", 30))))
                if not code_cr:
                    return _text("code_run: 'code' is required")
                # Auto-inject DEVICE variable for torch/tensorflow/cuda code
                code_cr = GpuCodeRuntime.prepare(code_cr)
                install_log_cr: list[str] = []
                if pkgs_cr:
                    for pkg in pkgs_cr:
                        pkg = str(pkg).strip()
                        if not pkg:
                            continue
                        try:
                            proc_pip = await asyncio.create_subprocess_exec(
                                "python3", "-m", "pip", "install", "--quiet", pkg,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.STDOUT,
                            )
                            out_pip, _ = await asyncio.wait_for(
                                proc_pip.communicate(), timeout=15.0
                            )
                            install_log_cr.append(
                                f"pip install {pkg}: exit {proc_pip.returncode}"
                            )
                        except asyncio.TimeoutError:
                            install_log_cr.append(f"pip install {pkg}: timed out")
                        except Exception as exc:
                            install_log_cr.append(f"pip install {pkg}: {exc}")
                # Write code to tempfile
                import sys as _sys_cr
                with _tmpfile_cr.NamedTemporaryFile(
                    suffix=".py", mode="w", encoding="utf-8", delete=False
                ) as _tf:
                    _tf.write(code_cr)
                    tmp_path_cr = _tf.name
                t0_cr = _time_cr.monotonic()
                try:
                    proc_cr = await asyncio.create_subprocess_exec(
                        _sys_cr.executable, tmp_path_cr,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    try:
                        stdout_b, stderr_b = await asyncio.wait_for(
                            proc_cr.communicate(), timeout=float(timeout_cr)
                        )
                        exit_code_cr = proc_cr.returncode
                    except asyncio.TimeoutError:
                        proc_cr.kill()
                        await proc_cr.wait()
                        return _text(
                            f"code_run: timed out after {timeout_cr}s\n"
                            + ("\nInstall log:\n" + "\n".join(install_log_cr) if install_log_cr else "")
                        )
                    duration_ms = int((_time_cr.monotonic() - t0_cr) * 1000)
                    stdout_s = stdout_b.decode(errors="replace")
                    stderr_s = stderr_b.decode(errors="replace")
                    parts = [f"code_run: exit={exit_code_cr} duration={duration_ms}ms"]
                    if stdout_s.strip():
                        parts.append(f"stdout:\n{stdout_s.strip()}")
                    if stderr_s.strip():
                        parts.append(f"stderr:\n{stderr_s.strip()}")
                    if install_log_cr:
                        parts.append("install_log:\n" + "\n".join(install_log_cr))
                    return _text("\n\n".join(parts))
                finally:
                    try:
                        _os_cr.unlink(tmp_path_cr)
                    except OSError:
                        pass

            # ----------------------------------------------------------------
            # run_javascript — Node.js execution (CommonJS)
            # ----------------------------------------------------------------
            if name == "run_javascript":
                import tempfile as _tmpfile_js
                import os as _os_js
                import time as _time_js
                import shutil as _shutil_js
                code_js    = str(args.get("code", "")).strip()
                timeout_js = max(1, min(120, int(args.get("timeout", 30))))
                if not code_js:
                    return _text("run_javascript: 'code' is required")
                node_bin = _shutil_js.which("node") or _shutil_js.which("nodejs")
                if not node_bin:
                    return _text(
                        "run_javascript: Node.js is not installed in this environment. "
                        "Use code_run (Python) as an alternative."
                    )
                with _tmpfile_js.NamedTemporaryFile(
                    suffix=".cjs", mode="w", encoding="utf-8", delete=False,
                    dir="/workspace" if _os_js.path.isdir("/workspace") else None,
                ) as _tf_js:
                    _tf_js.write(code_js)
                    tmp_path_js = _tf_js.name
                t0_js = _time_js.monotonic()
                try:
                    proc_js = await asyncio.create_subprocess_exec(
                        node_bin, tmp_path_js,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd="/workspace" if _os_js.path.isdir("/workspace") else None,
                    )
                    try:
                        stdout_js_b, stderr_js_b = await asyncio.wait_for(
                            proc_js.communicate(), timeout=float(timeout_js)
                        )
                        exit_code_js = proc_js.returncode
                    except asyncio.TimeoutError:
                        proc_js.kill()
                        await proc_js.wait()
                        return _text(f"run_javascript: timed out after {timeout_js}s")
                    duration_js_ms = int((_time_js.monotonic() - t0_js) * 1000)
                    stdout_js_s = stdout_js_b.decode(errors="replace")
                    stderr_js_s = stderr_js_b.decode(errors="replace")
                    parts_js = [f"run_javascript: exit={exit_code_js} duration={duration_js_ms}ms"]
                    if stdout_js_s.strip():
                        parts_js.append(f"stdout:\n{stdout_js_s.strip()}")
                    if stderr_js_s.strip():
                        parts_js.append(f"stderr:\n{stderr_js_s.strip()}")
                    return _text("\n\n".join(parts_js))
                finally:
                    try:
                        _os_js.unlink(tmp_path_js)
                    except OSError:
                        pass

            # ----------------------------------------------------------------
            # get_system_instructions — returns planner directive for MCP clients
            # ----------------------------------------------------------------
            if name == "get_system_instructions":
                from system_prompt import get_system_prompt as _get_sys_prompt
                _base_prompt = _get_sys_prompt()
                _planning_hint = (
                    "\n\n## Planning Workflow\n\n"
                    "For any task that requires more than one tool call, or where the right\n"
                    "sequence of tools is unclear, call plan_task FIRST.\n\n"
                    "  plan_task(task='...', context='...', max_steps=N)\n\n"
                    "It returns an ordered STEPS list with exact tool names, args, and\n"
                    "dependencies. Execute each step in order, passing prior outputs forward.\n\n"
                    "For single-tool tasks, call the tool directly.\n\n"
                    "Always prefer fewer, targeted tool calls over verbose chains.\n"
                    "When referencing a prior step's output, describe it clearly in the next call's args."
                )
                return _text(_base_prompt + _planning_hint)

            # ----------------------------------------------------------------
            # plan_task — orchestration planner (calls LM Studio to build plan)
            # ----------------------------------------------------------------
            if name == "plan_task":
                task_pt       = str(args.get("task", "")).strip()
                context_pt    = str(args.get("context", "")).strip()
                extra_tools_pt = str(args.get("extra_tools", "")).strip()
                max_steps_pt  = max(2, min(20, int(args.get("max_steps", 10))))

                if not task_pt:
                    return _text("plan_task: 'task' is required")

                catalog_pt = _PLAN_TASK_CATALOG
                if extra_tools_pt:
                    catalog_pt += f"\n\n[Extra Tools (from other MCPs)]\n{extra_tools_pt}"

                system_pt = (
                    "You are an AI task planner. Output ONLY a JSON object, no prose, no function calls.\n"
                    "REQUIRED OUTPUT FORMAT (use exactly this structure — no other format is accepted):\n"
                    "{\"summary\":\"one sentence\","
                    "\"steps\":[{\"step\":1,\"tool\":\"tool_name\",\"args\":{\"param\":\"string_value\"},"
                    "\"rationale\":\"\",\"depends_on\":[],\"output_var\":\"\"}],"
                    "\"notes\":\"\"}\n"
                    f"Use at most {max_steps_pt} steps. Use exact tool names from list below.\n"
                    "RULES:\n"
                    "- Output the JSON object DIRECTLY. Never use function-call syntax or tool_calls format.\n"
                    "- All values in 'args' must be quoted JSON strings.\n"
                    "- Reference prior step output with placeholder strings like \"{{step_1.result}}\".\n"
                    "- For anime/illustrated images use face_recognize with style='anime'.\n"
                    f"AVAILABLE TOOLS:\n{catalog_pt}"
                )
                user_pt = f"Task: {task_pt}"
                if context_pt:
                    user_pt += f"\n\nContext: {context_pt}"

                import re as _re_pt, json as _json_pt

                raw_pt = ""
                last_err_pt = ""
                # Probe timeout: short ping to detect which model is GPU-resident.
                # Full timeout: generous for actual plan generation.
                _PT_PROBE_TIMEOUT = 12.0
                _PT_FULL_TIMEOUT  = 60.0

                async with httpx.AsyncClient(timeout=_PT_PROBE_TIMEOUT) as hc_probe:
                    all_candidate_models_pt = await ModelRegistry.get().plan_models(hc_probe)

                if not all_candidate_models_pt:
                    return _text("plan_task: no suitable chat model found in LM Studio")

                # Phase 1: Quick probe (1-token) to find the GPU-resident model.
                # Models not in GPU memory will timeout; the active one responds in <12s.
                active_model_pt = None
                _probe_payload = {"messages": [{"role": "user", "content": "1"}], "max_tokens": 1}
                async with httpx.AsyncClient(timeout=_PT_PROBE_TIMEOUT) as hc_probe2:
                    for _pm in all_candidate_models_pt[:8]:  # probe up to 8 candidates
                        try:
                            _probe_payload["model"] = _pm
                            _pr = await hc_probe2.post(
                                f"{IMAGE_GEN_BASE_URL}/v1/chat/completions", json=_probe_payload
                            )
                            if _pr.status_code < 500:
                                active_model_pt = _pm
                                break
                        except Exception:
                            continue

                # Phase 2: Run the actual plan request against the responsive model(s).
                # If probe found no active model, fall back to iterating all candidates.
                candidate_models_pt = (
                    [active_model_pt] + [m for m in all_candidate_models_pt if m != active_model_pt]
                    if active_model_pt
                    else all_candidate_models_pt
                )[:3]

                # Capacity guard — check the first candidate before the heavy inference call
                if candidate_models_pt:
                    async with httpx.AsyncClient(timeout=_PT_PROBE_TIMEOUT) as _hc_cap_pt:
                        _busy_pt = await ModelRegistry.get().ensure_model_or_busy(_hc_cap_pt, candidate_models_pt[0])
                    if _busy_pt:
                        return _text(f"plan_task: {_busy_pt}")

                async with httpx.AsyncClient(timeout=_PT_FULL_TIMEOUT) as hc_pt:
                    base_msg_pt = [
                        {"role": "system", "content": system_pt},
                        {"role": "user",   "content": user_pt},
                    ]

                    for try_model_pt in candidate_models_pt:
                        # Attempt A: with response_format json_object
                        payload_a = {
                            "model": try_model_pt,
                            "messages": base_msg_pt,
                            "max_tokens": 800,
                            "temperature": 0.1,
                            "response_format": {"type": "json_object"},
                        }
                        try:
                            r_a = await hc_pt.post(
                                f"{IMAGE_GEN_BASE_URL}/v1/chat/completions", json=payload_a
                            )
                            if r_a.status_code == 400:
                                # Model may not support json_object — retry as plain text
                                payload_b = {k: v for k, v in payload_a.items() if k != "response_format"}
                                r_b = await hc_pt.post(
                                    f"{IMAGE_GEN_BASE_URL}/v1/chat/completions", json=payload_b
                                )
                                if r_b.status_code in (400, 404, 422):
                                    last_err_pt = f"model {try_model_pt}: HTTP {r_b.status_code}"
                                    continue  # try next model
                                r_b.raise_for_status()
                                choices_b = r_b.json().get("choices", [])
                                _msg_b = choices_b[0]["message"] if choices_b else {}
                                # content may be None if LM Studio used tool_calls format
                                raw_pt = (_msg_b.get("content") or "").strip()
                                _tc_b = _msg_b.get("tool_calls") or []
                                if not raw_pt and _tc_b:
                                    _fn = _tc_b[0].get("function", {})
                                    raw_pt = json.dumps({"name": _fn.get("name"), "arguments": json.loads(_fn.get("arguments", "{}"))})
                                # Some models (e.g. lfm2) output native tool-call tokens even without response_format.
                                # Format: <|tool_call_start|>[{...}]<|tool_call_end|>
                                # Must be detected BEFORE the generic JSON-block extractor strips the delimiters.
                                # Must be detected BEFORE the generic JSON-block extractor strips the delimiters.
                                if "<|tool_call_start|>" in raw_pt:
                                    _tc_m = _re_pt.search(r"<\|tool_call_start\|>\s*(\[.*?\])", raw_pt, _re_pt.DOTALL)
                                    if _tc_m:
                                        try:
                                            _tc_list = _json_pt.loads(_tc_m.group(1))
                                            _steps_from_tc = [
                                                {
                                                    "step": _i + 1, "tool": _tc.get("name", "unknown"),
                                                    "args": (_tc.get("arguments", {}) if isinstance(_tc.get("arguments"), dict)
                                                             else _json_pt.loads(_tc.get("arguments", "{}"))),
                                                    "rationale": "", "depends_on": [],
                                                    "output_var": f"step_{_i + 1}_result",
                                                }
                                                for _i, _tc in enumerate(_tc_list)
                                            ]
                                            raw_pt = _json_pt.dumps({
                                                "summary": "Plan: " + ", ".join(tc.get("name", "") for tc in _tc_list),
                                                "steps": _steps_from_tc, "notes": "",
                                            })
                                        except Exception:
                                            pass
                                else:
                                    # Extract the first JSON block from prose response
                                    m_re = _re_pt.search(r"\{[\s\S]*\}", raw_pt)
                                    if m_re:
                                        raw_pt = m_re.group(0)
                            else:
                                r_a.raise_for_status()
                                choices_a = r_a.json().get("choices", [])
                                raw_pt = choices_a[0]["message"]["content"].strip() if choices_a else ""
                                # Some models (e.g. lfm2) respond to response_format=json_object
                                # with OpenAI function-call format: {"name":"tool","arguments":{}}
                                # Detect and retry without json_object to get the actual plan.
                                try:
                                    _tmp_obj = _json_pt.loads(raw_pt)
                                    if "name" in _tmp_obj and "steps" not in _tmp_obj:
                                        payload_plain = {k: v for k, v in payload_a.items() if k != "response_format"}
                                        r_plain = await hc_pt.post(
                                            f"{IMAGE_GEN_BASE_URL}/v1/chat/completions", json=payload_plain
                                        )
                                        r_plain.raise_for_status()
                                        _ch_plain = r_plain.json().get("choices", [])
                                        raw_pt = _ch_plain[0]["message"]["content"].strip() if _ch_plain else raw_pt
                                        _m_plain = _re_pt.search(r"\{[\s\S]*\}", raw_pt)
                                        if _m_plain:
                                            raw_pt = _m_plain.group(0)
                                except Exception:
                                    pass
                            break  # success — stop trying models
                        except Exception as exc_loop:
                            last_err_pt = f"model {try_model_pt}: {exc_loop}"
                            continue

                if not raw_pt:
                    return _text(
                        f"plan_task: all candidate models failed. Last error: {last_err_pt}\n"
                        "Ensure at least one chat model is loaded in LM Studio."
                    )

                # Validate and pretty-print the JSON plan.
                # Use raw_decode to extract the first valid JSON object even if the
                # model generated trailing garbage or a second disconnected JSON object.
                _raw_stripped = raw_pt.lstrip()
                try:
                    plan_obj, _end_pos = _json_pt.JSONDecoder().raw_decode(_raw_stripped)

                    # Some models (e.g. lfm2) emit extra step objects after the main plan:
                    # {...plan JSON with step 1...}{"step":2,"tool":"summarize",...}
                    # Scan the trailing text and merge any orphaned step objects.
                    if isinstance(plan_obj.get("steps"), list):
                        _seen = {s.get("step") for s in plan_obj["steps"]}
                        _tail = _raw_stripped[_end_pos:]
                        _dec2 = _json_pt.JSONDecoder()
                        _p = 0
                        while _p < len(_tail):
                            try:
                                _extra, _adv = _dec2.raw_decode(_tail[_p:].lstrip())
                                _p += len(_tail[_p:]) - len(_tail[_p:].lstrip()) + _adv
                                if (isinstance(_extra, dict) and "step" in _extra
                                        and "tool" in _extra and _extra["step"] not in _seen):
                                    plan_obj["steps"].append(_extra)
                                    _seen.add(_extra["step"])
                            except Exception:
                                break
                        plan_obj["steps"].sort(key=lambda s: s.get("step", 999))

                    steps    = plan_obj.get("steps", [])
                    summary  = plan_obj.get("summary", "")
                    notes    = plan_obj.get("notes", "")

                    # Format for easy reading by the outer LLM
                    lines_pt = [
                        f"PLAN: {summary}",
                        f"STEPS ({len(steps)} total):",
                    ]
                    for s in steps:
                        dep_str = f" [needs step {s.get('depends_on')}]" if s.get("depends_on") else ""
                        raw_args = s.get("args", {})
                        # Some models serialize args as a JSON string rather than an object
                        if isinstance(raw_args, str):
                            try:
                                raw_args = _json_pt.loads(raw_args)
                            except Exception:
                                pass
                        lines_pt.append(
                            f"  {s.get('step')}. {s.get('tool')}({_json_pt.dumps(raw_args, separators=(',', ':'))})"
                            f"  ← {s.get('rationale', '')}{dep_str}"
                        )
                    if notes:
                        lines_pt.append(f"\nNOTES: {notes}")
                    lines_pt.append(
                        "\nEXECUTE: call each tool above in step order, passing prior outputs as indicated."
                    )
                    # Also embed the full JSON for programmatic use
                    lines_pt.append(f"\nFULL_JSON:\n{_json_pt.dumps(plan_obj, indent=2)}")
                    return _text("\n".join(lines_pt))
                except Exception:
                    # JSON parse failed — apply chain of targeted fixups for common model output bugs:
                    # 1. tool(args) signature leakage → bare name: "web_search(query)" → "web_search"
                    # 2. Missing { before step N+1: },"step":2 → },{"step":2
                    # 3. Bare variable names as arg values: "content":top_result → "content":"{{top_result}}"
                    try:
                        fixed = raw_pt
                        fixed = _re_pt.sub(r'"tool"\s*:\s*"(\w+)\([^"]*\)"', lambda m: f'"tool":"{m.group(1)}"', fixed)
                        fixed = _re_pt.sub(r'}\s*,\s*"step"\s*:', r'},{"step":', fixed)
                        fixed = _re_pt.sub(
                            r':\s*([A-Za-z_][A-Za-z0-9_]*)\s*([,}\]])',
                            r': "{{\1}}" \2',
                            fixed,
                        )
                        plan_obj, _ = _json_pt.JSONDecoder().raw_decode(fixed.lstrip())
                        steps   = plan_obj.get("steps", [])
                        summary = plan_obj.get("summary", "")
                        notes   = plan_obj.get("notes", "")
                        lines_pt = [f"PLAN: {summary}", f"STEPS ({len(steps)} total):"]
                        for s in steps:
                            dep_str = f" [needs step {s.get('depends_on')}]" if s.get("depends_on") else ""
                            lines_pt.append(
                                f"  {s.get('step')}. {s.get('tool')}({_json_pt.dumps(s.get('args',{}), separators=(',',':'))})"
                                f"  ← {s.get('rationale','')}{dep_str}"
                            )
                        if notes:
                            lines_pt.append(f"\nNOTES: {notes}")
                        lines_pt.append("\nEXECUTE: call each tool above in step order, passing prior outputs as indicated.")
                        lines_pt.append(f"\nFULL_JSON:\n{_json_pt.dumps(plan_obj, indent=2)}")
                        return _text("\n".join(lines_pt))
                    except Exception:
                        # Final fallback: return raw so the LLM can still use it
                        return _text(f"plan_task result (raw):\n{raw_pt}")

            # ----------------------------------------------------------------
            # smart_summarize — LLM summarization of text
            # ----------------------------------------------------------------
            if name == "smart_summarize":
                content_ss = str(args.get("content", "")).strip()
                style_ss   = str(args.get("style", "brief")).strip() or "brief"
                max_words  = args.get("max_words")
                if not content_ss:
                    return _text("smart_summarize: 'content' is required")
                text_ss = content_ss[:8000]
                if style_ss == "bullets":
                    instruction_ss = "Summarize the following text as a concise markdown bullet list."
                elif style_ss == "detailed":
                    instruction_ss = "Write a detailed, comprehensive summary of the following text."
                else:
                    instruction_ss = "Summarize the following text in 2–3 sentences."
                if max_words:
                    instruction_ss += f" Aim for approximately {int(max_words)} words."
                msgs_ss = [
                    {"role": "system", "content": "You are a helpful summarizer. Be concise and accurate."},
                    {"role": "user",   "content": f"{instruction_ss}\n\n---\n{text_ss}"},
                ]
                async with httpx.AsyncClient(timeout=60.0) as hc_ss:
                    _ss_candidates = (
                        [IMAGE_GEN_MODEL.strip()]
                        if IMAGE_GEN_MODEL.strip()
                        else (await ModelRegistry.get().chat_models(hc_ss))[:2]
                    )
                    if not _ss_candidates:
                        return _text("smart_summarize: no chat model found in LM Studio")
                    # Capacity guard on the first candidate (chat_models already prefers loaded)
                    _busy_ss = await ModelRegistry.get().ensure_model_or_busy(hc_ss, _ss_candidates[0])
                    if _busy_ss:
                        return _text(f"smart_summarize: {_busy_ss}")
                    payload_ss: dict = {"messages": msgs_ss, "max_tokens": 600, "temperature": 0.3}
                    summary = ""
                    _ss_last_err = ""
                    for _ss_model in _ss_candidates:
                        payload_ss["model"] = _ss_model
                        try:
                            r_ss = await hc_ss.post(
                                f"{IMAGE_GEN_BASE_URL}/v1/chat/completions", json=payload_ss
                            )
                            r_ss.raise_for_status()
                            _ch_ss = r_ss.json().get("choices", [])
                            summary = (_ch_ss[0]["message"]["content"].strip() if _ch_ss else "")
                            if summary:
                                break  # got a non-empty response — done
                        except Exception as exc:
                            _ss_last_err = str(exc)
                            continue
                    if summary:
                        return _text(summary)
                    if _ss_last_err:
                        return _text(
                            f"smart_summarize: LM Studio request failed — {_ss_last_err}\n"
                            "Make sure a chat model is loaded in LM Studio."
                        )
                    return _text("smart_summarize: empty response from model")

            # ----------------------------------------------------------------
            # image_caption — vision model image description
            # ----------------------------------------------------------------
            if name == "image_caption":
                b64_ic     = str(args.get("b64", "")).strip()
                detail_ic  = str(args.get("detail_level", "detailed")).strip() or "detailed"
                if not b64_ic:
                    return _text("image_caption: 'b64' is required (base64-encoded JPEG)")
                if detail_ic == "brief":
                    prompt_ic = "Describe this image in one concise sentence."
                else:
                    prompt_ic = (
                        "Describe this image in detail. Include: the main subject, "
                        "background, colors, style, mood, and any notable elements. "
                        "Be specific and vivid."
                    )
                msgs_ic = [{"role": "user", "content": [
                    {"type": "text",      "text": prompt_ic},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_ic}"}},
                ]}]
                payload_ic: dict = {
                    "messages": msgs_ic,
                    "max_tokens": 512 if detail_ic == "detailed" else 80,
                    "temperature": 0.3,
                }

                # Model selection: auto-detect best vision model via ModelRegistry;
                # fallback to IMAGE_GEN_MODEL env if set, or LM Studio server default.
                async with httpx.AsyncClient(timeout=120.0) as hc_ic:
                    _chosen_model = await ModelRegistry.get().best_vision_model(hc_ic)
                    if _chosen_model:
                        payload_ic["model"] = _chosen_model
                        # Capacity guard — avoid evicting an active model
                        _busy_ic = await ModelRegistry.get().ensure_model_or_busy(hc_ic, _chosen_model)
                        if _busy_ic:
                            return _text(f"image_caption: {_busy_ic}")
                    if not _chosen_model and not await ModelRegistry.get().is_available(hc_ic):
                        return _text(
                            "image_caption: no model loaded in LM Studio.\n"
                            f"Endpoint: {IMAGE_GEN_BASE_URL}\n"
                            "Ensure LM Studio Local Server is running with a vision-capable model loaded."
                        )
                    # Vision inference can be slow (30 s+) for large models; give 120 s.
                    try:
                        r_ic = await hc_ic.post(
                            f"{IMAGE_GEN_BASE_URL}/v1/chat/completions", json=payload_ic
                        )
                        r_ic.raise_for_status()
                        _ch_ic = r_ic.json().get("choices", [])
                        caption = (_ch_ic[0]["message"]["content"].strip() if _ch_ic else "")
                        if caption:
                            model_note = f" [model: {_chosen_model}]" if _chosen_model else ""
                            return _text(f"{caption}{model_note}")
                        return _text("image_caption: empty response — ensure a model is loaded and running in LM Studio.")
                    except Exception as exc:
                        return _text(
                            f"image_caption: LM Studio request failed — {exc}\n"
                            f"Endpoint: {IMAGE_GEN_BASE_URL}  Model: {_chosen_model or '(server default)'}\n"
                            "Ensure LM Studio Local Server is running with a vision-capable model loaded."
                        )

            # ----------------------------------------------------------------
            # structured_extract — JSON extraction with json_object mode
            # ----------------------------------------------------------------
            if name == "structured_extract":
                import json as _js_se
                content_se = str(args.get("content", "")).strip()
                schema_se  = str(args.get("schema_json", "")).strip()
                extra_se   = str(args.get("instructions", "")).strip()
                if not content_se:
                    return _text("structured_extract: 'content' is required")
                text_se    = content_se[:6000]
                schema_hint = f"\n\nExtract data matching this schema:\n{schema_se}" if schema_se else ""
                extra_hint  = f"\n\nAdditional instructions: {extra_se}" if extra_se else ""
                msgs_se = [
                    {"role": "system", "content": (
                        "You are a structured data extractor. Extract information from the provided "
                        "text and return it as valid JSON matching the requested schema. "
                        "Return ONLY the JSON object, no extra text."
                    )},
                    {"role": "user", "content": f"Extract from this text:{schema_hint}{extra_hint}\n\n---\n{text_se}"},
                ]
                payload_se: dict = {
                    "messages": msgs_se,
                    "max_tokens": 800,
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"},
                }
                async with httpx.AsyncClient(timeout=60.0) as hc_se:
                    _se_candidates = (
                        [IMAGE_GEN_MODEL.strip()]
                        if IMAGE_GEN_MODEL.strip()
                        else (await ModelRegistry.get().chat_models(hc_se))[:2]
                    )
                    if not _se_candidates:
                        return _text("structured_extract: no chat model found in LM Studio")
                    # Capacity guard on the first candidate (chat_models already prefers loaded)
                    _busy_se = await ModelRegistry.get().ensure_model_or_busy(hc_se, _se_candidates[0])
                    if _busy_se:
                        return _text(f"structured_extract: {_busy_se}")
                    raw_se = ""
                    _se_last_err = ""
                    for _se_model in _se_candidates:
                        payload_se["model"] = _se_model
                        try:
                            r_se = await hc_se.post(
                                f"{IMAGE_GEN_BASE_URL}/v1/chat/completions", json=payload_se
                            )
                            if r_se.status_code == 400:
                                # json_object not supported — retry without it
                                _p2 = {k: v for k, v in payload_se.items() if k != "response_format"}
                                r_se = await hc_se.post(
                                    f"{IMAGE_GEN_BASE_URL}/v1/chat/completions", json=_p2
                                )
                            r_se.raise_for_status()
                            _ch_se = r_se.json().get("choices", [])
                            raw_se = (_ch_se[0]["message"]["content"].strip() if _ch_se else "")
                            break
                        except Exception as exc:
                            _se_last_err = str(exc)
                            continue
                    if not raw_se:
                        return _text(
                            f"structured_extract: LM Studio request failed — {_se_last_err}\n"
                            "Make sure a chat model is loaded (ideally one that supports json_object mode)."
                        )
                    try:
                        parsed = _js_se.loads(raw_se)
                        return _text(_js_se.dumps(parsed, indent=2))
                    except _js_se.JSONDecodeError:
                        return _text(f"structured_extract: model returned non-JSON:\n{raw_se}")

            # ----------------------------------------------------------------
            if name == "orchestrate":
                raw_steps = args.get("steps", [])
                stop_on_error = bool(args.get("stop_on_error", False))
                if not isinstance(raw_steps, list) or not raw_steps:
                    return _text("orchestrate: 'steps' must be a non-empty list")
                try:
                    steps = [
                        WorkflowStep(
                            id=str(s["id"]),
                            tool=str(s["tool"]),
                            args=dict(s.get("args", {})),
                            depends_on=list(s.get("depends_on", [])),
                            label=str(s.get("label", s["id"])),
                        )
                        for s in raw_steps
                    ]
                except (KeyError, TypeError) as exc:
                    return _text(f"orchestrate: invalid step definition — {exc}")
                try:
                    executor = WorkflowExecutor(steps, stop_on_error=stop_on_error)
                    results = await executor.run()
                except ValueError as exc:
                    return _text(f"orchestrate: workflow error — {exc}")
                return _text(WorkflowExecutor._format_report(results))

            # ----------------------------------------------------------------
            # Graph tools
            # ----------------------------------------------------------------
            if name == "graph_add_node":
                node_id    = str(args.get("id", "")).strip()
                labels     = list(args.get("labels", []))
                properties = dict(args.get("properties", {}))
                if not node_id:
                    return _text("graph_add_node: 'id' is required")
                r = await c.post(f"{GRAPH_URL}/nodes/add",
                                 json={"id": node_id, "labels": labels, "properties": properties},
                                 timeout=10)
                if r.status_code >= 400:
                    return _text(f"graph_add_node failed: {r.status_code} — {r.text[:300]}")
                d = r.json()
                return _text(f"Node added: {d.get('added')} (labels={d.get('labels')})")

            if name == "graph_add_edge":
                from_id    = str(args.get("from_id", "")).strip()
                to_id      = str(args.get("to_id",   "")).strip()
                etype      = str(args.get("type", "related")).strip() or "related"
                properties = dict(args.get("properties", {}))
                if not from_id or not to_id:
                    return _text("graph_add_edge: 'from_id' and 'to_id' are required")
                r = await c.post(f"{GRAPH_URL}/edges/add",
                                 json={"from_id": from_id, "to_id": to_id,
                                       "type": etype, "properties": properties},
                                 timeout=10)
                if r.status_code >= 400:
                    return _text(f"graph_add_edge failed: {r.status_code} — {r.text[:300]}")
                d = r.json()
                return _text(f"Edge added: {d.get('from_id')} -[{etype}]-> {d.get('to_id')} (id={d.get('added')})")

            if name == "graph_query":
                node_id = str(args.get("id", "")).strip()
                if not node_id:
                    return _text("graph_query: 'id' is required")
                r = await c.get(f"{GRAPH_URL}/nodes/{node_id}/neighbors", timeout=10)
                if r.status_code == 404:
                    return _text(f"graph_query: node '{node_id}' not found")
                if r.status_code >= 400:
                    return _text(f"graph_query failed: {r.status_code} — {r.text[:300]}")
                d = r.json()
                neighbors = d.get("neighbors", [])
                if not neighbors:
                    return _text(f"Node '{node_id}' has no outgoing neighbors.")
                lines = [f"Neighbors of '{node_id}' ({len(neighbors)}):"]
                for nb in neighbors:
                    props = nb.get("properties", {})
                    lines.append(f"  → {nb.get('id','?')} [{nb.get('edge_type','')}]"
                                 + (f" {props}" if props else ""))
                return _text("\n".join(lines))

            if name == "graph_path":
                from_id = str(args.get("from_id", "")).strip()
                to_id   = str(args.get("to_id",   "")).strip()
                if not from_id or not to_id:
                    return _text("graph_path: 'from_id' and 'to_id' are required")
                r = await c.post(f"{GRAPH_URL}/path",
                                 json={"from_id": from_id, "to_id": to_id}, timeout=15)
                if r.status_code >= 400:
                    return _text(f"graph_path failed: {r.status_code} — {r.text[:300]}")
                d = r.json()
                path = d.get("path")
                if not path:
                    return _text(f"No path found from '{from_id}' to '{to_id}'.")
                length = d.get("length", len(path) - 1)
                return _text(f"Path ({length} hops): {' → '.join(path)}")

            if name == "graph_search":
                label      = str(args.get("label", "")).strip()
                properties = dict(args.get("properties", {}))
                limit_g    = int(args.get("limit", 50))
                r = await c.post(f"{GRAPH_URL}/search",
                                 json={"label": label, "properties": properties, "limit": limit_g},
                                 timeout=10)
                if r.status_code >= 400:
                    return _text(f"graph_search failed: {r.status_code} — {r.text[:300]}")
                d = r.json()
                results_g = d.get("results", [])
                if not results_g:
                    return _text(f"graph_search: no nodes found (label={label!r})")
                lines = [f"Found {len(results_g)} node(s):"]
                for node in results_g:
                    lbl = ",".join(node.get("labels", []))
                    props = node.get("properties", {})
                    lines.append(f"  {node.get('id','?')} [{lbl}] {props}")
                return _text("\n".join(lines))

            # ----------------------------------------------------------------
            # Vector tools (Qdrant)
            # ----------------------------------------------------------------
            if name == "vector_store":
                text_v      = str(args.get("text", "")).strip()
                vid         = str(args.get("id", "")).strip()
                collection  = str(args.get("collection", "default")).strip() or "default"
                metadata_v  = dict(args.get("metadata", {}))
                if not text_v:
                    return _text("vector_store: 'text' is required")
                # Qdrant requires UUID or unsigned integer point IDs
                # Use a deterministic UUID v5 from the user-supplied id string
                import uuid as _uuid_v
                if not vid:
                    vid_key = _uuid_v.uuid4().hex
                else:
                    vid_key = vid
                # Convert to UUID (v5, namespace=DNS) for Qdrant compatibility
                qdrant_id = str(_uuid_v.uuid5(_uuid_v.NAMESPACE_DNS, vid_key))
                # Step 1: embed via LM Studio
                try:
                    embed_payload = {"input": text_v, "model": await _get_embed_model()}
                    async with httpx.AsyncClient(timeout=30) as hc_v:
                        r_embed = await hc_v.post(
                            f"{IMAGE_GEN_BASE_URL}/v1/embeddings", json=embed_payload
                        )
                        r_embed.raise_for_status()
                        embed_data = r_embed.json().get("data", [])
                        if not embed_data:
                            return _text("vector_store: LM Studio returned no embeddings")
                        vector = embed_data[0]["embedding"]
                        dim    = len(vector)
                except Exception as exc:
                    return _text(f"vector_store: embedding failed — {exc}")
                # Step 2: ensure collection exists
                try:
                    rc = await c.get(f"{VECTOR_URL}/collections/{collection}", timeout=5)
                    if rc.status_code == 404:
                        await c.put(
                            f"{VECTOR_URL}/collections/{collection}",
                            json={"vectors": {"size": dim, "distance": "Cosine"}},
                            timeout=10,
                        )
                except Exception:
                    pass  # best-effort collection creation
                # Step 3: upsert point (store user id in payload for delete by id_key)
                payload_meta = {"text": text_v[:500], "id_key": vid_key, **metadata_v}
                try:
                    ru = await c.put(
                        f"{VECTOR_URL}/collections/{collection}/points",
                        json={"points": [{"id": qdrant_id, "vector": vector, "payload": payload_meta}]},
                        timeout=15,
                    )
                    ru.raise_for_status()
                except Exception as exc:
                    return _text(f"vector_store: Qdrant upsert failed — {exc}")
                return _text(f"Stored vector: id={vid_key} (qdrant_id={qdrant_id}) collection={collection} dim={dim}")

            if name == "vector_search":
                query_v    = str(args.get("query", "")).strip()
                collection = str(args.get("collection", "default")).strip() or "default"
                top_k      = max(1, int(args.get("top_k", 5)))
                filter_v   = args.get("filter")
                if not query_v:
                    return _text("vector_search: 'query' is required")
                # Embed query
                try:
                    ep = {"input": query_v, "model": await _get_embed_model()}
                    async with httpx.AsyncClient(timeout=30) as hc_vs:
                        r_vs = await hc_vs.post(f"{IMAGE_GEN_BASE_URL}/v1/embeddings", json=ep)
                        r_vs.raise_for_status()
                        ed = r_vs.json().get("data", [])
                        if not ed:
                            return _text("vector_search: embedding returned empty")
                        qvec = ed[0]["embedding"]
                except Exception as exc:
                    return _text(f"vector_search: embedding failed — {exc}")
                # Search Qdrant
                search_body: dict = {"vector": qvec, "limit": top_k, "with_payload": True}
                if filter_v:
                    search_body["filter"] = filter_v
                try:
                    rs = await c.post(
                        f"{VECTOR_URL}/collections/{collection}/points/search",
                        json=search_body, timeout=10,
                    )
                    rs.raise_for_status()
                    hits = rs.json().get("result", [])
                except Exception as exc:
                    return _text(f"vector_search: Qdrant search failed — {exc}")
                if not hits:
                    return _text(f"vector_search: no results in collection '{collection}'")
                lines = [f"Top {len(hits)} results from '{collection}':"]
                for h in hits:
                    score   = round(h.get("score", 0), 4)
                    payload = h.get("payload", {})
                    text_preview = str(payload.get("text", ""))[:120]
                    lines.append(f"  [{score}] id={h.get('id')} — {text_preview}")
                return _text("\n".join(lines))

            if name == "vector_delete":
                vid_d      = str(args.get("id", "")).strip()
                collection = str(args.get("collection", "default")).strip() or "default"
                if not vid_d:
                    return _text("vector_delete: 'id' is required")
                # Delete by payload id_key filter (since Qdrant IDs are UUIDs internally)
                import uuid as _uuid_d
                qdrant_id_d = str(_uuid_d.uuid5(_uuid_d.NAMESPACE_DNS, vid_d))
                try:
                    rd = await c.post(
                        f"{VECTOR_URL}/collections/{collection}/points/delete",
                        json={"points": [qdrant_id_d]}, timeout=10,
                    )
                    rd.raise_for_status()
                except Exception as exc:
                    return _text(f"vector_delete: failed — {exc}")
                return _text(f"Deleted vector id={vid_d} from collection '{collection}'")

            if name == "vector_collections":
                try:
                    rc = await c.get(f"{VECTOR_URL}/collections", timeout=10)
                    rc.raise_for_status()
                    colls = rc.json().get("result", {}).get("collections", [])
                except Exception as exc:
                    return _text(f"vector_collections: failed — {exc}")
                if not colls:
                    return _text("No Qdrant collections found. Use vector_store to create one.")
                lines = ["Qdrant collections:"]
                for col in colls:
                    lines.append(f"  {col.get('name')} — vectors: {col.get('vectors_count', '?')}")
                return _text("\n".join(lines))

            # ----------------------------------------------------------------
            # Video tools
            # ----------------------------------------------------------------
            if name == "video_info":
                url_vi = str(args.get("url", "")).strip()
                if not url_vi:
                    return _text("video_info: 'url' is required")
                try:
                    r_vi = await c.post(f"{VIDEO_URL}/info", json={"url": url_vi}, timeout=90)
                    r_vi.raise_for_status()
                    d_vi = r_vi.json()
                except Exception as exc:
                    return _text(f"video_info: failed — {exc}")
                return _text(
                    f"Video info for {url_vi}:\n"
                    f"  Duration: {d_vi.get('duration_s')}s  FPS: {d_vi.get('fps')}\n"
                    f"  Resolution: {d_vi.get('width')}×{d_vi.get('height')}  Codec: {d_vi.get('codec')}\n"
                    f"  Format: {d_vi.get('format')}  Size: {d_vi.get('size_mb')} MB"
                )

            if name == "video_frames":
                url_vf       = str(args.get("url", "")).strip()
                interval_vf  = float(args.get("interval_sec", 5.0))
                max_frames_vf = int(args.get("max_frames", 20))
                if not url_vf:
                    return _text("video_frames: 'url' is required")
                try:
                    r_vf = await c.post(
                        f"{VIDEO_URL}/frames",
                        json={"url": url_vf, "interval_sec": interval_vf,
                              "max_frames": max_frames_vf},
                        timeout=180,
                    )
                    r_vf.raise_for_status()
                    d_vf = r_vf.json()
                except Exception as exc:
                    return _text(f"video_frames: failed — {exc}")
                frames = d_vf.get("frames", [])
                lines = [f"Extracted {len(frames)} frames from {url_vf}:"]
                for fr in frames:
                    lines.append(f"  [{fr.get('timestamp_s')}s] {fr.get('path')}")
                return _text("\n".join(lines))

            if name == "video_thumbnail":
                url_vt  = str(args.get("url", "")).strip()
                ts_vt   = float(args.get("timestamp_sec", 0.0))
                if not url_vt:
                    return _text("video_thumbnail: 'url' is required")
                try:
                    r_vt = await c.post(
                        f"{VIDEO_URL}/thumbnail",
                        json={"url": url_vt, "timestamp_sec": ts_vt},
                        timeout=90,
                    )
                    r_vt.raise_for_status()
                    d_vt = r_vt.json()
                except Exception as exc:
                    return _text(f"video_thumbnail: failed — {exc}")
                b64_vt = d_vt.get("b64", "")
                if not b64_vt:
                    return _text("video_thumbnail: no image returned")
                import base64 as _b64_vt_mod
                raw_vt = _b64_vt_mod.b64decode(b64_vt)
                summary_vt = (f"Video thumbnail at {ts_vt}s — "
                              f"{d_vt.get('width')}×{d_vt.get('height')} from {url_vt}")
                return _renderer.encode_url_bytes(raw_vt, "image/png", summary_vt)

            if name == "video_transcode":
                url_xt   = str(args.get("url", "")).strip()
                codec_xt = str(args.get("codec", "h264")).strip()
                br_xt    = str(args.get("bitrate", "5M")).strip()
                w_xt     = int(args.get("width", 0))
                h_xt     = int(args.get("height", 0))
                fn_xt    = str(args.get("filename", "")).strip()
                if not url_xt:
                    return _text("video_transcode: 'url' is required")
                try:
                    r_xt = await c.post(
                        f"{VIDEO_URL}/transcode",
                        json={"url": url_xt, "codec": codec_xt, "bitrate": br_xt,
                              "width": w_xt, "height": h_xt, "filename": fn_xt},
                        timeout=600,
                    )
                    r_xt.raise_for_status()
                    d_xt = r_xt.json()
                except Exception as exc:
                    return _text(f"video_transcode: failed — {exc}")
                gpu_tag = "GPU (VA-API)" if d_xt.get("gpu_accelerated") else "CPU"
                return _text(
                    f"Transcoded {url_xt} → {d_xt.get('path')}\n"
                    f"  Codec: {d_xt.get('codec')}  Encoder: {gpu_tag}\n"
                    f"  Resolution: {d_xt.get('width')}×{d_xt.get('height')}\n"
                    f"  Duration: {d_xt.get('duration_s')}s  Size: {d_xt.get('size_mb')} MB"
                )

            # ----------------------------------------------------------------
            # OCR tools
            # ----------------------------------------------------------------
            if name == "ocr_image":
                path_oi = str(args.get("path", "")).strip()
                lang_oi = str(args.get("lang", "eng")).strip() or "eng"
                if not path_oi:
                    return _text("ocr_image: 'path' is required")
                local_oi = _resolve_image_path(path_oi)
                if not local_oi:
                    return _text(f"ocr_image: file not found — {path_oi}")
                import base64 as _b64_oi
                with open(local_oi, "rb") as _f_oi:
                    b64_oi = _b64_oi.standard_b64encode(_f_oi.read()).decode("ascii")
                try:
                    r_oi = await c.post(
                        OCR_URL,
                        json={"b64": b64_oi, "lang": lang_oi},
                        timeout=60,
                    )
                    r_oi.raise_for_status()
                    d_oi = r_oi.json()
                except Exception as exc:
                    return _text(f"ocr_image: OCR service failed — {exc}")
                text_oi = d_oi.get("text", "")
                words_oi = d_oi.get("word_count", 0)
                return _text(f"OCR result ({words_oi} words):\n\n{text_oi}")

            if name == "ocr_pdf":
                path_op = str(args.get("path", "")).strip()
                lang_op = str(args.get("lang", "eng")).strip() or "eng"
                pages_op = list(args.get("pages", []))
                if not path_op:
                    return _text("ocr_pdf: 'path' is required")
                local_op = _resolve_image_path(path_op)
                if not local_op:
                    return _text(f"ocr_pdf: file not found — {path_op}")
                import base64 as _b64_op
                with open(local_op, "rb") as _f_op:
                    b64_op = _b64_op.standard_b64encode(_f_op.read()).decode("ascii")
                try:
                    r_op = await c.post(
                        f"{OCR_URL}/pdf",
                        json={"b64_pdf": b64_op, "lang": lang_op, "pages": pages_op or None},
                        timeout=120,
                    )
                    r_op.raise_for_status()
                    d_op = r_op.json()
                except Exception as exc:
                    return _text(f"ocr_pdf: OCR service failed — {exc}")
                total_w = d_op.get("word_count", 0)
                page_count = d_op.get("page_count", 0)
                full_text = d_op.get("full_text", "")
                return _text(f"PDF OCR: {page_count} pages, {total_w} words\n\n{full_text}")

            # ----------------------------------------------------------------
            # Docs tools
            # ----------------------------------------------------------------
            if name in ("docs_ingest", "docs_extract_tables"):
                url_di   = str(args.get("url",  "")).strip()
                path_di  = str(args.get("path", "")).strip()
                fname_di = str(args.get("filename", "")).strip()

                if url_di:
                    # Service fetches URL directly
                    try:
                        if name == "docs_ingest":
                            r_di = await c.post(f"{DOCS_URL}/ingest/url",
                                                json={"url": url_di}, timeout=60)
                        else:
                            # For tables from URL: ingest first then return tables
                            r_di = await c.post(f"{DOCS_URL}/ingest/url",
                                                json={"url": url_di}, timeout=60)
                        r_di.raise_for_status()
                        d_di = r_di.json()
                    except Exception as exc:
                        return _text(f"{name}: docs service failed — {exc}")
                elif path_di:
                    local_di = _resolve_image_path(path_di)
                    if not local_di:
                        return _text(f"{name}: file not found — {path_di}")
                    if not fname_di:
                        fname_di = os.path.basename(local_di)
                    import base64 as _b64_di
                    b64_di = _b64_di.standard_b64encode(
                        open(local_di, "rb").read()
                    ).decode("ascii")
                    try:
                        endpoint_di = "/ingest" if name == "docs_ingest" else "/tables"
                        r_di = await c.post(
                            f"{DOCS_URL}{endpoint_di}",
                            json={"b64": b64_di, "filename": fname_di},
                            timeout=60,
                        )
                        r_di.raise_for_status()
                        d_di = r_di.json()
                    except Exception as exc:
                        return _text(f"{name}: docs service failed — {exc}")
                else:
                    return _text(f"{name}: 'url' or 'path' is required")

                if name == "docs_ingest":
                    md_di  = d_di.get("markdown", "")
                    title  = d_di.get("title", "")
                    words  = d_di.get("word_count", 0)
                    tables = d_di.get("tables_found", 0)
                    return _text(f"# {title}\n\n_{words} words, {tables} tables_\n\n{md_di}")
                else:
                    tables_di = d_di.get("tables", [])
                    if not tables_di:
                        return _text("No tables found in document.")
                    import json as _jdi
                    return _text(_jdi.dumps(tables_di, indent=2))

            # ----------------------------------------------------------------
            # PDF tools
            # ----------------------------------------------------------------
            if name in ("pdf_read", "pdf_edit", "pdf_fill_form", "pdf_merge", "pdf_split"):
                def _workspace_file_for_pdf(raw_path: str) -> tuple[str | None, str | None]:
                    rp = str(raw_path or "").strip()
                    if not rp:
                        return None, None

                    rel = ""
                    if rp.startswith("/workspace/"):
                        rel = rp[len("/workspace/") :].lstrip("/")
                    elif rp.startswith("/docker/human_browser/workspace/"):
                        rel = rp[len("/docker/human_browser/workspace/") :].lstrip("/")
                    elif rp.startswith(BROWSER_WORKSPACE + "/"):
                        rel = rp[len(BROWSER_WORKSPACE) + 1 :].lstrip("/")
                    elif os.path.isabs(rp):
                        try:
                            rel_guess = os.path.relpath(rp, BROWSER_WORKSPACE)
                        except Exception:
                            return None, None
                        if rel_guess.startswith(".."):
                            return None, None
                        rel = rel_guess.replace("\\", "/").lstrip("/")
                    else:
                        rel = rp.lstrip("/")

                    local = os.path.normpath(os.path.join(BROWSER_WORKSPACE, rel))
                    ws_root = os.path.normpath(BROWSER_WORKSPACE)
                    if not (local == ws_root or local.startswith(ws_root + os.sep)):
                        return None, None
                    if os.path.isfile(local):
                        return local, f"/workspace/{rel}".replace("//", "/")

                    # Fallback to basename for callers that pass odd prefixes.
                    base = os.path.basename(rel)
                    if not base:
                        return None, None
                    alt_local = os.path.join(BROWSER_WORKSPACE, base)
                    if os.path.isfile(alt_local):
                        return alt_local, f"/workspace/{base}"
                    return None, None

                def _workspace_target_for_pdf(raw_path: str) -> str:
                    rp = str(raw_path or "").strip()
                    if not rp:
                        return ""
                    if rp.startswith("/workspace/"):
                        return rp
                    if rp.startswith("/docker/human_browser/workspace/"):
                        return f"/workspace/{rp[len('/docker/human_browser/workspace/'):]}"
                    if rp.startswith(BROWSER_WORKSPACE + "/"):
                        return f"/workspace/{rp[len(BROWSER_WORKSPACE) + 1:]}"
                    if os.path.isabs(rp):
                        try:
                            rel_guess = os.path.relpath(rp, BROWSER_WORKSPACE)
                        except Exception:
                            return ""
                        if rel_guess.startswith(".."):
                            return ""
                        return f"/workspace/{rel_guess}".replace("\\", "/")
                    return f"/workspace/{rp.lstrip('/')}"

                if name == "pdf_read":
                    path_pr = str(args.get("path", "")).strip()
                    mode_pr = str(args.get("mode", "auto")).strip() or "auto"
                    pages_pr = list(args.get("pages", []))
                    lang_pr = str(args.get("lang", "eng")).strip() or "eng"
                    password_pr = str(args.get("password", "")).strip()
                    if not path_pr:
                        return _text("pdf_read: 'path' is required")
                    local_pr, ws_path_pr = _workspace_file_for_pdf(path_pr)
                    if not local_pr or not ws_path_pr:
                        return _text(f"pdf_read: file not found in workspace — {path_pr}")

                    body_pr: dict[str, Any] = {
                        "path": ws_path_pr,
                        "mode": mode_pr,
                        "lang": lang_pr,
                    }
                    if pages_pr:
                        body_pr["pages"] = pages_pr
                    if password_pr:
                        body_pr["password"] = password_pr
                    try:
                        r_pr = await c.post(f"{PDF_URL}/read", json=body_pr, timeout=240)
                        r_pr.raise_for_status()
                        d_pr = r_pr.json()
                    except Exception as exc:
                        return _text(f"pdf_read: failed — {exc}")

                    text_pr = str(d_pr.get("text", ""))
                    truncated = False
                    if len(text_pr) > 14000:
                        text_pr = text_pr[:14000] + "\n\n...[truncated]..."
                        truncated = True
                    input_kind = str(d_pr.get("input_kind", "pdf")).strip().lower() or "pdf"
                    label = "Image" if input_kind == "image" else "PDF"
                    return _text(
                        f"{label} read: {d_pr.get('page_count', 0)} page(s) | "
                        f"mode={d_pr.get('mode', mode_pr)} | ocr_used={d_pr.get('ocr_used', False)}\n"
                        f"File: {d_pr.get('host_path', '')}\n"
                        f"{'Output truncated for MCP transport size limits.' if truncated else ''}\n\n"
                        f"{text_pr}"
                    )

                if name == "pdf_edit":
                    path_pe = str(args.get("path", "")).strip()
                    ops_pe = args.get("operations", [])
                    out_pe = _workspace_target_for_pdf(str(args.get("output_path", "")).strip())
                    verify_pe = bool(args.get("verify", True))
                    password_pe = str(args.get("password", "")).strip()
                    if not path_pe:
                        return _text("pdf_edit: 'path' is required")
                    if not isinstance(ops_pe, list) or not ops_pe:
                        return _text("pdf_edit: 'operations' must be a non-empty list")
                    local_pe, ws_path_pe = _workspace_file_for_pdf(path_pe)
                    if not local_pe or not ws_path_pe:
                        return _text(f"pdf_edit: file not found in workspace — {path_pe}")
                    body_pe: dict[str, Any] = {
                        "path": ws_path_pe,
                        "operations": ops_pe,
                        "verify": verify_pe,
                    }
                    if out_pe:
                        body_pe["output_path"] = out_pe
                    if password_pe:
                        body_pe["password"] = password_pe
                    try:
                        r_pe = await c.post(f"{PDF_URL}/edit", json=body_pe, timeout=240)
                        r_pe.raise_for_status()
                        d_pe = r_pe.json()
                    except Exception as exc:
                        return _text(f"pdf_edit: failed — {exc}")
                    verify_obj = d_pe.get("verification", {})
                    verify_ok = bool(verify_obj.get("ok", False))
                    ops_applied = d_pe.get("operations_applied", [])
                    input_kind = str(d_pe.get("input_kind", "pdf")).strip().lower() or "pdf"
                    label = "Image" if input_kind == "image" else "PDF"
                    return _text(
                        f"{label} edited successfully.\n"
                        f"Output: {d_pe.get('host_path', '')}\n"
                        f"Verification: {'PASS' if verify_ok else 'WARN/FAIL'}\n"
                        f"Operations applied: {json.dumps(ops_applied, ensure_ascii=False)}\n"
                        f"Checks: {json.dumps(verify_obj.get('checks', []), ensure_ascii=False)}"
                    )

                if name == "pdf_fill_form":
                    path_pf = str(args.get("path", "")).strip()
                    fields_pf = args.get("fields", {})
                    flatten_pf = bool(args.get("flatten", False))
                    out_pf = _workspace_target_for_pdf(str(args.get("output_path", "")).strip())
                    password_pf = str(args.get("password", "")).strip()
                    if not path_pf:
                        return _text("pdf_fill_form: 'path' is required")
                    if not isinstance(fields_pf, dict) or not fields_pf:
                        return _text("pdf_fill_form: 'fields' must be a non-empty object")
                    local_pf, ws_path_pf = _workspace_file_for_pdf(path_pf)
                    if not local_pf or not ws_path_pf:
                        return _text(f"pdf_fill_form: file not found in workspace — {path_pf}")
                    body_pf: dict[str, Any] = {
                        "path": ws_path_pf,
                        "fields": fields_pf,
                        "flatten": flatten_pf,
                    }
                    if out_pf:
                        body_pf["output_path"] = out_pf
                    if password_pf:
                        body_pf["password"] = password_pf
                    try:
                        r_pf = await c.post(f"{PDF_URL}/fill-form", json=body_pf, timeout=180)
                        r_pf.raise_for_status()
                        d_pf = r_pf.json()
                    except Exception as exc:
                        return _text(f"pdf_fill_form: failed — {exc}")
                    return _text(
                        f"PDF form filled.\n"
                        f"Output: {d_pf.get('host_path', '')}\n"
                        f"Flattened: {d_pf.get('flattened', False)}\n"
                        f"Fields written: {d_pf.get('fields_written', [])}\n"
                        f"Warnings: {d_pf.get('warnings', [])}"
                    )

                if name == "pdf_merge":
                    paths_pm = args.get("paths", [])
                    out_pm = _workspace_target_for_pdf(str(args.get("output_path", "")).strip())
                    passwords_pm = args.get("passwords", {})
                    if not isinstance(paths_pm, list) or len(paths_pm) < 2:
                        return _text("pdf_merge: 'paths' must be a list with at least 2 items")

                    ws_paths_pm: list[str] = []
                    for rp in paths_pm:
                        local_pm, ws_path_pm = _workspace_file_for_pdf(str(rp))
                        if not local_pm or not ws_path_pm:
                            return _text(f"pdf_merge: file not found in workspace — {rp}")
                        ws_paths_pm.append(ws_path_pm)

                    body_pm: dict[str, Any] = {"paths": ws_paths_pm}
                    if out_pm:
                        body_pm["output_path"] = out_pm
                    if isinstance(passwords_pm, dict) and passwords_pm:
                        body_pm["passwords"] = passwords_pm
                    try:
                        r_pm = await c.post(f"{PDF_URL}/merge", json=body_pm, timeout=240)
                        r_pm.raise_for_status()
                        d_pm = r_pm.json()
                    except Exception as exc:
                        return _text(f"pdf_merge: failed — {exc}")
                    return _text(
                        f"Merged {len(d_pm.get('inputs', []))} PDF(s) into one file.\n"
                        f"Output: {d_pm.get('host_path', '')}\n"
                        f"Page count: {d_pm.get('page_count', '?')}"
                    )

                if name == "pdf_split":
                    path_ps = str(args.get("path", "")).strip()
                    ranges_ps = args.get("ranges", [])
                    prefix_ps = str(args.get("prefix", "")).strip()
                    output_dir_ps = _workspace_target_for_pdf(str(args.get("output_dir", "")).strip())
                    password_ps = str(args.get("password", "")).strip()
                    if not path_ps:
                        return _text("pdf_split: 'path' is required")
                    if not isinstance(ranges_ps, list) or not ranges_ps:
                        return _text("pdf_split: 'ranges' must be a non-empty list")
                    local_ps, ws_path_ps = _workspace_file_for_pdf(path_ps)
                    if not local_ps or not ws_path_ps:
                        return _text(f"pdf_split: file not found in workspace — {path_ps}")
                    body_ps: dict[str, Any] = {
                        "path": ws_path_ps,
                        "ranges": ranges_ps,
                    }
                    if prefix_ps:
                        body_ps["prefix"] = prefix_ps
                    if output_dir_ps:
                        body_ps["output_dir"] = output_dir_ps
                    if password_ps:
                        body_ps["password"] = password_ps
                    try:
                        r_ps = await c.post(f"{PDF_URL}/split", json=body_ps, timeout=240)
                        r_ps.raise_for_status()
                        d_ps = r_ps.json()
                    except Exception as exc:
                        return _text(f"pdf_split: failed — {exc}")

                    outputs_ps = d_ps.get("outputs", [])
                    if not outputs_ps:
                        return _text("pdf_split: no files were produced")
                    lines_ps = [f"Split into {len(outputs_ps)} PDF(s):"]
                    for it in outputs_ps:
                        lines_ps.append(f"  {it.get('host_path', '')}  pages={it.get('range', [])}")
                    return _text("\\n".join(lines_ps))

            # ----------------------------------------------------------------
            # Planner tools
            # ----------------------------------------------------------------
            if name == "plan_create_task":
                title_pt = str(args.get("title", "")).strip()
                if not title_pt:
                    return _text("plan_create_task: 'title' is required")
                body_pt = {
                    "title":       title_pt,
                    "description": str(args.get("description", "")),
                    "depends_on":  list(args.get("depends_on", [])),
                    "priority":    int(args.get("priority", 0)),
                    "metadata":    dict(args.get("metadata", {})),
                }
                if args.get("due_at"):
                    body_pt["due_at"] = str(args["due_at"])
                try:
                    r_pt = await c.post(f"{PLANNER_URL}/tasks", json=body_pt, timeout=10)
                    r_pt.raise_for_status()
                    d_pt = r_pt.json()
                except Exception as exc:
                    return _text(f"plan_create_task: failed — {exc}")
                return _text(
                    f"Task created: id={d_pt.get('id')} title={d_pt.get('title')!r} "
                    f"status={d_pt.get('status')} priority={d_pt.get('priority')} "
                    f"depends_on={d_pt.get('depends_on')}"
                )

            if name == "plan_get_task":
                tid_pg = str(args.get("id", "")).strip()
                if not tid_pg:
                    return _text("plan_get_task: 'id' is required")
                try:
                    r_pg = await c.get(f"{PLANNER_URL}/tasks/{tid_pg}", timeout=10)
                    if r_pg.status_code == 404:
                        return _text(f"plan_get_task: task '{tid_pg}' not found")
                    r_pg.raise_for_status()
                    d_pg = r_pg.json()
                except Exception as exc:
                    return _text(f"plan_get_task: failed — {exc}")
                return _text(
                    f"Task {d_pg['id']}: {d_pg['title']!r}\n"
                    f"  Status: {d_pg['status']}  Priority: {d_pg['priority']}\n"
                    f"  Description: {d_pg.get('description','')}\n"
                    f"  Depends on: {d_pg.get('depends_on',[])}\n"
                    f"  Created: {d_pg.get('created_at','')}  Updated: {d_pg.get('updated_at','')}"
                )

            if name == "plan_complete_task":
                tid_pc = str(args.get("id", "")).strip()
                if not tid_pc:
                    return _text("plan_complete_task: 'id' is required")
                try:
                    r_pc = await c.post(f"{PLANNER_URL}/tasks/{tid_pc}/complete", timeout=10)
                    if r_pc.status_code == 404:
                        return _text(f"plan_complete_task: task '{tid_pc}' not found")
                    r_pc.raise_for_status()
                except Exception as exc:
                    return _text(f"plan_complete_task: failed — {exc}")
                return _text(f"Task {tid_pc} marked as done.")

            if name == "plan_fail_task":
                tid_pf = str(args.get("id", "")).strip()
                detail_pf = str(args.get("detail", "")).strip()
                if not tid_pf:
                    return _text("plan_fail_task: 'id' is required")
                try:
                    r_pf = await c.post(f"{PLANNER_URL}/tasks/{tid_pf}/fail",
                                        json={"detail": detail_pf}, timeout=10)
                    if r_pf.status_code == 404:
                        return _text(f"plan_fail_task: task '{tid_pf}' not found")
                    r_pf.raise_for_status()
                except Exception as exc:
                    return _text(f"plan_fail_task: failed — {exc}")
                return _text(f"Task {tid_pf} marked as failed. Reason: {detail_pf or '(none)'}")

            if name == "plan_list_tasks":
                status_pl = str(args.get("status", "")).strip()
                limit_pl  = max(1, int(args.get("limit", 50)))
                try:
                    params_pl: dict = {"limit": limit_pl}
                    if status_pl:
                        params_pl["status"] = status_pl
                    r_pl = await c.get(f"{PLANNER_URL}/tasks", params=params_pl, timeout=10)
                    r_pl.raise_for_status()
                    d_pl = r_pl.json()
                except Exception as exc:
                    return _text(f"plan_list_tasks: failed — {exc}")
                tasks_pl = d_pl.get("tasks", [])
                total_pl = d_pl.get("total", len(tasks_pl))
                if not tasks_pl:
                    return _text("No tasks found" + (f" with status='{status_pl}'" if status_pl else "") + ".")
                lines_pl = [f"Tasks ({len(tasks_pl)}/{total_pl}" +
                            (f", status={status_pl}" if status_pl else "") + "):"]
                for t_pl in tasks_pl:
                    dep_pl = f" [deps: {t_pl['depends_on']}]" if t_pl.get("depends_on") else ""
                    lines_pl.append(f"  [{t_pl['status']}] {t_pl['id']}: {t_pl['title']!r}{dep_pl}")
                return _text("\n".join(lines_pl))

            if name == "plan_delete_task":
                tid_pd = str(args.get("id", "")).strip()
                if not tid_pd:
                    return _text("plan_delete_task: 'id' is required")
                try:
                    r_pd = await c.delete(f"{PLANNER_URL}/tasks/{tid_pd}", timeout=10)
                    if r_pd.status_code == 404:
                        return _text(f"plan_delete_task: task '{tid_pd}' not found")
                    r_pd.raise_for_status()
                except Exception as exc:
                    return _text(f"plan_delete_task: failed — {exc}")
                return _text(f"Task {tid_pd} deleted.")

            # ----------------------------------------------------------------
            # Async job tools
            # ----------------------------------------------------------------
            if name == "job_submit":
                t_name   = str(args.get("tool_name", "")).strip()
                t_args   = dict(args.get("args", {}))
                priority  = int(args.get("priority", 0))
                timeout_s = float(args.get("timeout_s", 300.0))
                max_ret   = int(args.get("max_retries", 0))
                if not t_name:
                    return _text("job_submit: 'tool_name' is required")
                payload = {"tool_name": t_name, "args": t_args, "priority": priority,
                           "timeout_s": timeout_s, "max_retries": max_ret,
                           "input_summary": f"{t_name}({json.dumps(t_args)[:120]})"}
                r = await c.post(f"{JOB_URL}", json=payload, timeout=10)
                if r.status_code >= 400:
                    return _text(f"job_submit failed: {r.status_code} — {r.text[:300]}")
                job = r.json()
                job_id = job["id"]
                # Start background execution
                asyncio.create_task(_execute_job(job_id, t_name, t_args, timeout_s))
                return _text(json.dumps({"job_id": job_id, "status": "queued",
                                         "tool_name": t_name, "timeout_s": timeout_s}))

            if name == "job_status":
                jid = str(args.get("job_id", "")).strip()
                if not jid:
                    return _text("job_status: 'job_id' is required")
                r = await c.get(f"{JOB_URL}/{jid}", timeout=10)
                if r.status_code == 404:
                    return _text(f"job_status: job '{jid}' not found")
                if r.status_code >= 400:
                    return _text(f"job_status failed: {r.status_code} — {r.text[:300]}")
                d = r.json()
                return _text(json.dumps({
                    "job_id":       d["id"],
                    "tool_name":    d["tool_name"],
                    "status":       d["status"],
                    "progress":     d["progress"],
                    "submitted_at": d["submitted_at"],
                    "started_at":   d["started_at"],
                    "finished_at":  d["finished_at"],
                    "error":        d["error"],
                }))

            if name == "job_result":
                jid = str(args.get("job_id", "")).strip()
                if not jid:
                    return _text("job_result: 'job_id' is required")
                r = await c.get(f"{JOB_URL}/{jid}", timeout=10)
                if r.status_code == 404:
                    return _text(f"job_result: job '{jid}' not found")
                d = r.json()
                if d["status"] not in ("succeeded", "failed", "cancelled"):
                    return _text(json.dumps({"job_id": jid, "status": d["status"],
                                             "message": "Job has not completed yet."}))
                if d["status"] == "succeeded":
                    return _text(d.get("result") or "(empty result)")
                return _text(json.dumps({"job_id": jid, "status": d["status"],
                                         "error": d.get("error", "unknown error")}))

            if name == "job_cancel":
                jid = str(args.get("job_id", "")).strip()
                if not jid:
                    return _text("job_cancel: 'job_id' is required")
                _job_cancelled.add(jid)
                r = await c.post(f"{JOB_URL}/{jid}/cancel", json={}, timeout=10)
                if r.status_code == 404:
                    return _text(f"job_cancel: job '{jid}' not found")
                d = r.json()
                return _text(json.dumps({"job_id": jid, "status": d.get("status")}))

            if name == "job_list":
                status_f   = str(args.get("status",    "")).strip()
                tool_f     = str(args.get("tool_name", "")).strip()
                limit_f    = max(1, min(int(args.get("limit", 20)), 100))
                params_q: dict = {"limit": limit_f}
                if status_f:
                    params_q["status"] = status_f
                if tool_f:
                    params_q["tool_name"] = tool_f
                r = await c.get(f"{JOB_URL}", params=params_q, timeout=10)
                if r.status_code >= 400:
                    return _text(f"job_list failed: {r.status_code} — {r.text[:300]}")
                d = r.json()
                jobs = d.get("jobs", [])
                lines = [f"Jobs ({d.get('total', 0)} total, showing {len(jobs)}):"]
                for j in jobs:
                    lines.append(f"  {j['id']}  {j['tool_name']:<22}  {j['status']:<12}  "
                                 f"submitted={j['submitted_at'][:19]}")
                return _text("\n".join(lines))

            if name == "batch_submit":
                items_raw = list(args.get("items", []))
                if not items_raw:
                    return _text("batch_submit: 'items' must be a non-empty list")
                r = await c.post(f"{JOB_URL}/batch", json={"items": items_raw}, timeout=30)
                if r.status_code >= 400:
                    return _text(f"batch_submit failed: {r.status_code} — {r.text[:300]}")
                d = r.json()
                job_ids = [j["id"] for j in d.get("job_ids", [])]
                # Fire off background execution for each job
                for item, jid in zip(items_raw, job_ids):
                    asyncio.create_task(_execute_job(
                        jid,
                        item.get("tool_name", ""),
                        dict(item.get("args", {})),
                        float(item.get("timeout_s", 300.0)),
                    ))
                return _text(json.dumps({"job_ids": job_ids, "count": len(job_ids),
                                         "status": "queued"}))

            # ----------------------------------------------------------------
            # think — reasoning scratchpad (no-op, returns thought verbatim)
            # ----------------------------------------------------------------
            if name == "think":
                thought = str(args.get("thought", "")).strip()
                return _text(thought if thought else "(empty thought)")

            # ----------------------------------------------------------------
            # deep_research — multi-hop research: search → fetch → synthesize
            # ----------------------------------------------------------------
            if name == "deep_research":
                question_dr = str(args.get("question", "")).strip()
                depth_dr    = max(1, min(3, int(args.get("depth", 2))))
                max_src_dr  = max(1, min(8, int(args.get("max_sources", 4))))
                if not question_dr:
                    return _text("deep_research: 'question' is required")
                if not SEARXNG_URL:
                    return _text("deep_research: SEARXNG_URL not configured")

                all_content_dr: list[dict] = []
                search_queries_dr = [question_dr]

                for round_dr in range(depth_dr):
                    for sq_dr in search_queries_dr[:2]:
                        try:
                            sr_dr = await c.get(
                                f"{SEARXNG_URL}/search",
                                params={"q": sq_dr, "format": "json", "categories": "general"},
                                timeout=20,
                            )
                            if sr_dr.status_code != 200:
                                continue
                            results_dr = sr_dr.json().get("results", [])[:max_src_dr]
                        except Exception:
                            continue
                        for res_dr in results_dr:
                            url_dr   = res_dr.get("url", "")
                            title_dr = res_dr.get("title", "")
                            snip_dr  = res_dr.get("content", "")
                            if not url_dr:
                                continue
                            try:
                                pr_dr = await c.get(url_dr, timeout=12, follow_redirects=True)
                                raw_dr = pr_dr.text
                                try:
                                    import trafilatura as _traf_dr
                                    text_dr = _traf_dr.extract(raw_dr) or ""
                                except Exception:
                                    text_dr = re.sub(r"<[^>]+>", " ", raw_dr)
                                    text_dr = re.sub(r"\s+", " ", text_dr).strip()
                                text_dr = text_dr[:4000]
                            except Exception:
                                text_dr = snip_dr
                            if text_dr:
                                all_content_dr.append({
                                    "title": title_dr, "url": url_dr,
                                    "content": text_dr, "round": round_dr + 1,
                                })
                    # Follow-up query: append top keywords from fetched content
                    if round_dr < depth_dr - 1 and all_content_dr:
                        combined_dr = " ".join(d["content"] for d in all_content_dr[-max_src_dr:])
                        words_dr = [w for w in combined_dr.split() if len(w) > 5][:10]
                        if words_dr:
                            search_queries_dr = [question_dr + " " + " ".join(words_dr[:5])]

                if not all_content_dr:
                    return _text("deep_research: no content retrieved — try a different question")

                lines_dr = [f"# Research Report: {question_dr}\n",
                            f"**Sources consulted:** {len(all_content_dr)} | "
                            f"**Rounds:** {depth_dr}\n",
                            "---\n", "## Source Summaries\n"]
                citations_dr: list[str] = []
                for i_dr, item_dr in enumerate(all_content_dr, 1):
                    excerpt_dr = item_dr["content"][:600].rstrip()
                    lines_dr.append(f"### [{i_dr}] {item_dr['title']}\n{excerpt_dr}…\n")
                    citations_dr.append(f"[{i_dr}] {item_dr['title']} — {item_dr['url']}")
                lines_dr.append("---\n## Citations\n")
                lines_dr.extend(citations_dr)
                return _text("\n".join(lines_dr))

            # ----------------------------------------------------------------
            # realtime — live weather / time / stocks / crypto / forex
            # ----------------------------------------------------------------
            if name == "realtime":
                rt_type  = str(args.get("type", "time")).lower()
                rt_query = str(args.get("query", "UTC")).strip()

                if rt_type == "time":
                    from datetime import datetime as _dt_rt
                    import zoneinfo as _zi_rt
                    try:
                        tz_rt = _zi_rt.ZoneInfo(rt_query)
                        now_rt = _dt_rt.now(tz_rt)
                        return _text(
                            f"Current time in {rt_query}:\n"
                            f"  {now_rt.strftime('%A, %B %d, %Y  %H:%M:%S %Z')}\n"
                            f"  UTC offset: {now_rt.strftime('%z')}"
                        )
                    except Exception as exc_rt:
                        return _text(f"realtime/time: invalid timezone '{rt_query}' — {exc_rt}")

                if rt_type == "weather":
                    try:
                        geo_r = await c.get(
                            "https://geocoding-api.open-meteo.com/v1/search",
                            params={"name": rt_query, "count": 1, "language": "en", "format": "json"},
                            timeout=10,
                        )
                        if geo_r.status_code != 200 or not geo_r.json().get("results"):
                            return _text(f"realtime/weather: city '{rt_query}' not found")
                        geo_rt = geo_r.json()["results"][0]
                        lat_rt, lon_rt = geo_rt["latitude"], geo_rt["longitude"]
                        wm_r = await c.get(
                            "https://api.open-meteo.com/v1/forecast",
                            params={
                                "latitude": lat_rt, "longitude": lon_rt,
                                "current": "temperature_2m,relative_humidity_2m,precipitation,"
                                           "weather_code,wind_speed_10m,apparent_temperature",
                                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
                                "timezone": "auto", "forecast_days": 7,
                            },
                            timeout=10,
                        )
                        if wm_r.status_code != 200:
                            return _text(f"realtime/weather: Open-Meteo error {wm_r.status_code}")
                        wm_rt = wm_r.json()
                        cur_rt = wm_rt.get("current", {})
                        daily_rt = wm_rt.get("daily", {})
                        _wmo = {0:"Clear",1:"Mainly clear",2:"Partly cloudy",3:"Overcast",
                                45:"Fog",48:"Icy fog",51:"Light drizzle",53:"Drizzle",55:"Heavy drizzle",
                                61:"Light rain",63:"Rain",65:"Heavy rain",71:"Light snow",73:"Snow",
                                75:"Heavy snow",80:"Showers",81:"Heavy showers",95:"Thunderstorm",
                                96:"Hailstorm",99:"Heavy hailstorm"}
                        wcode_rt = int(cur_rt.get("weather_code", 0))
                        cond_rt = _wmo.get(wcode_rt, f"Code {wcode_rt}")
                        lines_rt = [
                            f"**Weather in {geo_rt['name']}, {geo_rt.get('country','')}**",
                            f"Condition: {cond_rt}",
                            f"Temperature: {cur_rt.get('temperature_2m','?')}°C "
                            f"(feels like {cur_rt.get('apparent_temperature','?')}°C)",
                            f"Humidity: {cur_rt.get('relative_humidity_2m','?')}%",
                            f"Wind: {cur_rt.get('wind_speed_10m','?')} km/h",
                            f"Precipitation: {cur_rt.get('precipitation','?')} mm\n",
                            "**7-Day Forecast:**",
                        ]
                        dates_rt = daily_rt.get("time", [])
                        tmax_rt  = daily_rt.get("temperature_2m_max", [])
                        tmin_rt  = daily_rt.get("temperature_2m_min", [])
                        prec_rt  = daily_rt.get("precipitation_sum", [])
                        wcodes_rt= daily_rt.get("weather_code", [])
                        for di_rt in range(min(7, len(dates_rt))):
                            dc_rt = _wmo.get(int(wcodes_rt[di_rt]) if di_rt < len(wcodes_rt) else 0, "")
                            lines_rt.append(
                                f"  {dates_rt[di_rt]}: {tmin_rt[di_rt] if di_rt<len(tmin_rt) else '?'}–"
                                f"{tmax_rt[di_rt] if di_rt<len(tmax_rt) else '?'}°C  "
                                f"precip {prec_rt[di_rt] if di_rt<len(prec_rt) else '?'}mm  {dc_rt}"
                            )
                        return _text("\n".join(lines_rt))
                    except Exception as exc_weather:
                        return _text(f"realtime/weather: {exc_weather}")

                if rt_type == "stock":
                    try:
                        import yfinance as _yf_rt
                        ticker_rt = _yf_rt.Ticker(rt_query.upper())
                        info_rt   = ticker_rt.fast_info
                        hist_rt   = ticker_rt.history(period="2d")
                        price_rt  = float(info_rt.last_price) if hasattr(info_rt, "last_price") else None
                        prev_rt   = float(hist_rt["Close"].iloc[-2]) if len(hist_rt) >= 2 else None
                        chg_rt    = ((price_rt - prev_rt) / prev_rt * 100) if price_rt and prev_rt else None
                        lines_rt2 = [
                            f"**{rt_query.upper()} — {getattr(info_rt,'exchange','?')}**",
                            f"Price: ${price_rt:.4f}" if price_rt else "Price: N/A",
                            f"Change: {chg_rt:+.2f}%" if chg_rt is not None else "Change: N/A",
                            f"52w High: ${float(info_rt.year_high):.2f}" if hasattr(info_rt,"year_high") else "",
                            f"52w Low: ${float(info_rt.year_low):.2f}" if hasattr(info_rt,"year_low") else "",
                            f"Market Cap: ${float(info_rt.market_cap)/1e9:.1f}B" if hasattr(info_rt,"market_cap") and info_rt.market_cap else "",
                        ]
                        return _text("\n".join(l for l in lines_rt2 if l))
                    except Exception as exc_rt2:
                        return _text(f"realtime/stock: {exc_rt2}")

                if rt_type == "crypto":
                    try:
                        cg_r = await c.get(
                            "https://api.coingecko.com/api/v3/simple/price",
                            params={"ids": rt_query.lower(), "vs_currencies": "usd,eur,btc",
                                    "include_24hr_change": "true", "include_market_cap": "true"},
                            timeout=10,
                        )
                        if cg_r.status_code != 200:
                            return _text(f"realtime/crypto: CoinGecko error {cg_r.status_code}")
                        cg_rt = cg_r.json().get(rt_query.lower())
                        if not cg_rt:
                            return _text(f"realtime/crypto: coin '{rt_query}' not found on CoinGecko")
                        lines_cg = [
                            f"**{rt_query.title()}**",
                            f"USD: ${cg_rt.get('usd','?'):,}",
                            f"EUR: €{cg_rt.get('eur','?'):,}",
                            f"24h Change: {cg_rt.get('usd_24h_change','?'):.2f}%" if isinstance(cg_rt.get('usd_24h_change'), float) else "",
                            f"Market Cap: ${cg_rt.get('usd_market_cap','?'):,.0f}" if isinstance(cg_rt.get('usd_market_cap'), float) else "",
                        ]
                        return _text("\n".join(l for l in lines_cg if l))
                    except Exception as exc_crypto:
                        return _text(f"realtime/crypto: {exc_crypto}")

                if rt_type == "forex":
                    try:
                        parts_fx = rt_query.upper().replace("-", "/").split("/")
                        if len(parts_fx) != 2:
                            return _text("realtime/forex: use format 'USD/EUR' or 'GBP/JPY'")
                        base_fx, target_fx = parts_fx
                        fx_r = await c.get(
                            f"https://open.er-api.com/v6/latest/{base_fx}", timeout=10
                        )
                        if fx_r.status_code != 200:
                            return _text(f"realtime/forex: exchange rate API error {fx_r.status_code}")
                        fx_rt = fx_r.json()
                        rate_fx = fx_rt.get("rates", {}).get(target_fx)
                        if rate_fx is None:
                            return _text(f"realtime/forex: currency '{target_fx}' not found")
                        return _text(
                            f"**{base_fx} → {target_fx}**\n"
                            f"1 {base_fx} = {rate_fx:.6f} {target_fx}\n"
                            f"Updated: {fx_rt.get('time_last_update_utc','?')}"
                        )
                    except Exception as exc_forex:
                        return _text(f"realtime/forex: {exc_forex}")

                return _text(f"realtime: unknown type '{rt_type}'")

            # ----------------------------------------------------------------
            # news_search — RSS aggregation from major outlets
            # ----------------------------------------------------------------
            if name == "news_search":
                import feedparser as _fp_ns
                ns_query   = str(args.get("query", "")).strip().lower()
                ns_sources = [s.lower() for s in (args.get("sources") or [])]
                ns_limit   = max(1, min(50, int(args.get("limit", 10))))
                _NS_FEEDS = {
                    "bbc":         "https://feeds.bbci.co.uk/news/rss.xml",
                    "guardian":    "https://www.theguardian.com/world/rss",
                    "hackernews":  "https://news.ycombinator.com/rss",
                    "arstechnica": "https://feeds.arstechnica.com/arstechnica/index",
                    "techcrunch":  "https://techcrunch.com/feed/",
                }
                active_feeds = {k: v for k, v in _NS_FEEDS.items()
                                if not ns_sources or k in ns_sources}
                articles_ns: list[dict] = []
                for src_ns, feed_url_ns in active_feeds.items():
                    try:
                        feed_r_ns = await c.get(feed_url_ns, timeout=10, follow_redirects=True)
                        if feed_r_ns.status_code != 200:
                            continue
                        feed_ns = _fp_ns.parse(feed_r_ns.text)
                        for entry_ns in feed_ns.entries:
                            title_ns   = entry_ns.get("title", "")
                            summary_ns = entry_ns.get("summary", "")
                            link_ns    = entry_ns.get("link", "")
                            pub_ns     = entry_ns.get("published", "")
                            text_ns    = (title_ns + " " + summary_ns).lower()
                            if ns_query and ns_query not in text_ns:
                                continue
                            articles_ns.append({
                                "source": src_ns, "title": title_ns,
                                "summary": re.sub(r"<[^>]+>", "", summary_ns)[:200],
                                "url": link_ns, "published": pub_ns,
                            })
                    except Exception:
                        continue
                if not articles_ns:
                    return _text("news_search: no articles found" +
                                 (f" matching '{ns_query}'" if ns_query else ""))
                articles_ns = articles_ns[:ns_limit]
                lines_ns = [f"## News{' — ' + ns_query if ns_query else ''} "
                            f"({len(articles_ns)} articles)\n"]
                for a_ns in articles_ns:
                    lines_ns.append(
                        f"**[{a_ns['source'].upper()}] {a_ns['title']}**\n"
                        f"{a_ns['summary']}\n"
                        f"<{a_ns['url']}> | {a_ns['published']}\n"
                    )
                return _text("\n".join(lines_ns))

            # ----------------------------------------------------------------
            # wikipedia — Wikipedia REST API search + article summary
            # ----------------------------------------------------------------
            if name == "wikipedia":
                wp_query  = str(args.get("query", "")).strip()
                wp_lang   = str(args.get("lang", "en")).strip() or "en"
                wp_full   = bool(args.get("full_article", False))
                if not wp_query:
                    return _text("wikipedia: 'query' is required")
                _wp_headers = {"User-Agent": "aichat/1.0 (autonomous research bot; open-source)"}
                # Step 1: search for the best matching article
                search_r_wp = await c.get(
                    f"https://{wp_lang}.wikipedia.org/w/api.php",
                    params={"action": "query", "list": "search", "srsearch": wp_query,
                            "srlimit": 3, "format": "json", "srprop": "snippet"},
                    headers=_wp_headers,
                    timeout=10,
                )
                if search_r_wp.status_code != 200:
                    return _text(f"wikipedia: search error {search_r_wp.status_code}")
                search_hits_wp = search_r_wp.json().get("query", {}).get("search", [])
                if not search_hits_wp:
                    return _text(f"wikipedia: no results for '{wp_query}'")
                best_title_wp = search_hits_wp[0]["title"]
                # Step 2: fetch article
                if wp_full:
                    article_r_wp = await c.get(
                        f"https://{wp_lang}.wikipedia.org/w/api.php",
                        params={"action": "query", "prop": "extracts", "titles": best_title_wp,
                                "exintro": False, "explaintext": True, "format": "json"},
                        headers=_wp_headers,
                        timeout=15,
                    )
                    if article_r_wp.status_code != 200:
                        return _text(f"wikipedia: article fetch error {article_r_wp.status_code}")
                    pages_wp = article_r_wp.json().get("query", {}).get("pages", {})
                    text_wp  = next(iter(pages_wp.values()), {}).get("extract", "")[:8000]
                else:
                    summary_r_wp = await c.get(
                        f"https://{wp_lang}.wikipedia.org/api/rest_v1/page/summary/"
                        + best_title_wp.replace(" ", "_"),
                        headers=_wp_headers,
                        timeout=10,
                    )
                    if summary_r_wp.status_code != 200:
                        return _text(f"wikipedia: summary error {summary_r_wp.status_code}")
                    sdata_wp = summary_r_wp.json()
                    text_wp  = sdata_wp.get("extract", "")
                other_titles_wp = [h["title"] for h in search_hits_wp[1:]]
                result_wp = f"# {best_title_wp}\n\n{text_wp}"
                if other_titles_wp:
                    result_wp += f"\n\n---\n*Also consider: {', '.join(other_titles_wp)}*"
                return _text(result_wp)

            # ----------------------------------------------------------------
            # arxiv_search — arXiv academic paper search
            # ----------------------------------------------------------------
            if name == "arxiv_search":
                import defusedxml.ElementTree as _et_ax
                ax_query   = str(args.get("query", "")).strip()
                ax_max     = max(1, min(25, int(args.get("max_results", 8))))
                ax_cat     = str(args.get("category", "")).strip()
                ax_sort    = str(args.get("sort_by", "relevance"))
                if not ax_query:
                    return _text("arxiv_search: 'query' is required")
                search_ax = ax_query if not ax_cat else f"{ax_query} AND cat:{ax_cat}"
                ax_r = await c.get(
                    "https://export.arxiv.org/api/query",
                    params={"search_query": f"all:{search_ax}",
                            "max_results": ax_max, "sortBy": ax_sort},
                    timeout=20,
                )
                if ax_r.status_code != 200:
                    return _text(f"arxiv_search: arXiv API error {ax_r.status_code}")
                _ns_ax = {"atom": "http://www.w3.org/2005/Atom",
                          "arxiv": "http://arxiv.org/schemas/atom"}
                try:
                    root_ax = _et_ax.fromstring(ax_r.text)
                except Exception as exc_ax:
                    return _text(f"arxiv_search: XML parse error — {exc_ax}")
                entries_ax = root_ax.findall("atom:entry", _ns_ax)
                if not entries_ax:
                    return _text(f"arxiv_search: no papers found for '{ax_query}'")
                lines_ax = [f"## arXiv: '{ax_query}' — {len(entries_ax)} results\n"]
                for i_ax, entry_ax in enumerate(entries_ax, 1):
                    title_ax   = (entry_ax.findtext("atom:title", "", _ns_ax) or "").strip().replace("\n", " ")
                    abstract_ax= (entry_ax.findtext("atom:summary", "", _ns_ax) or "").strip()[:400]
                    published_ax = (entry_ax.findtext("atom:published", "", _ns_ax) or "")[:10]
                    authors_ax = [a.findtext("atom:name", "", _ns_ax)
                                  for a in entry_ax.findall("atom:author", _ns_ax)][:4]
                    id_elem_ax = entry_ax.findtext("atom:id", "", _ns_ax) or ""
                    pdf_ax     = id_elem_ax.replace("abs", "pdf") if "arxiv.org" in id_elem_ax else ""
                    lines_ax.append(
                        f"### [{i_ax}] {title_ax}\n"
                        f"**Authors:** {', '.join(authors_ax)}{' et al.' if len(authors_ax)==4 else ''}\n"
                        f"**Date:** {published_ax}\n"
                        f"**Abstract:** {abstract_ax}…\n"
                        f"**PDF:** {pdf_ax}\n"
                    )
                return _text("\n".join(lines_ax))

            # ----------------------------------------------------------------
            # youtube_transcript — extract captions/subtitles from YouTube
            # ----------------------------------------------------------------
            if name == "youtube_transcript":
                import re as _re_yt
                yt_url  = str(args.get("url", "")).strip()
                yt_lang = str(args.get("lang", "en")).strip() or "en"
                yt_ts   = bool(args.get("include_timestamps", True))
                if not yt_url:
                    return _text("youtube_transcript: 'url' is required")
                # Extract video ID
                vid_id_yt = None
                patterns_yt = [
                    r"(?:v=|youtu\.be/|embed/|shorts/)([a-zA-Z0-9_-]{11})",
                    r"^([a-zA-Z0-9_-]{11})$",
                ]
                for pat_yt in patterns_yt:
                    m_yt = _re_yt.search(pat_yt, yt_url, _re_yt.IGNORECASE)
                    if m_yt:
                        vid_id_yt = m_yt.group(1)
                        break
                if not vid_id_yt:
                    return _text(f"youtube_transcript: cannot extract video ID from '{yt_url}'")
                try:
                    from youtube_transcript_api import YouTubeTranscriptApi as _YTA
                    from youtube_transcript_api._errors import (
                        NoTranscriptFound as _NTF,
                        TranscriptsDisabled as _TD,
                    )
                    _yta_instance = _YTA()
                    try:
                        transcript_yt = _yta_instance.fetch(
                            vid_id_yt, languages=[yt_lang, "en", "en-US"]
                        )
                    except _NTF:
                        # Try any available transcript
                        transcript_list_yt = _yta_instance.list(vid_id_yt)
                        transcript_yt = next(iter(transcript_list_yt)).fetch()
                    lines_yt = [f"# YouTube Transcript: {yt_url}\n"]
                    for seg_yt in transcript_yt:
                        start_yt = int(getattr(seg_yt, "start", 0))
                        text_yt  = str(getattr(seg_yt, "text", "")).strip()
                        if yt_ts:
                            mm_yt, ss_yt = divmod(start_yt, 60)
                            lines_yt.append(f"[{mm_yt:02d}:{ss_yt:02d}] {text_yt}")
                        else:
                            lines_yt.append(text_yt)
                    return _text("\n".join(lines_yt))
                except _TD:
                    return _text(f"youtube_transcript: transcripts disabled for video '{vid_id_yt}'")
                except Exception as exc_yt:
                    return _text(f"youtube_transcript: {exc_yt}")

            # ----------------------------------------------------------------
            # desktop_screenshot — full X11 desktop capture via human_browser
            # ----------------------------------------------------------------
            if name == "desktop_screenshot":
                region_ds = args.get("region")
                payload_ds: dict = {}
                if region_ds:
                    payload_ds["region"] = region_ds
                try:
                    ds_r = await c.post(f"{BROWSER_URL}/desktop/screenshot",
                                        json=payload_ds, timeout=15)
                    if ds_r.status_code != 200:
                        return _text(f"desktop_screenshot: browser server error {ds_r.status_code} — {ds_r.text[:300]}")
                    ds_data = ds_r.json()
                    img_path_ds = ds_data.get("path", "")
                    if not img_path_ds:
                        return _text(f"desktop_screenshot: no path in response — {ds_data}")
                    # Translate browser-container path to mcp-container bind-mount path
                    fname_ds = os.path.basename(img_path_ds)
                    local_ds = os.path.join(BROWSER_WORKSPACE, fname_ds)
                    return _image_blocks(local_ds, f"Desktop screenshot saved to {fname_ds}")
                except Exception as exc_ds:
                    return _text(f"desktop_screenshot: {exc_ds}")

            # ----------------------------------------------------------------
            # desktop_control — xdotool computer use via human_browser
            # ----------------------------------------------------------------
            if name == "desktop_control":
                dc_action = str(args.get("action", "")).strip()
                if not dc_action:
                    return _text("desktop_control: 'action' is required")
                payload_dc: dict = {"action": dc_action}
                for k_dc in ("x", "y", "text", "command", "button", "direction", "amount"):
                    if args.get(k_dc) is not None:
                        payload_dc[k_dc] = args[k_dc]
                try:
                    dc_r = await c.post(f"{BROWSER_URL}/desktop/control",
                                        json=payload_dc, timeout=20)
                    if dc_r.status_code != 200:
                        return _text(f"desktop_control: browser server error {dc_r.status_code} — {dc_r.text[:300]}")
                    return _text(json.dumps(dc_r.json()))
                except Exception as exc_dc:
                    return _text(f"desktop_control: {exc_dc}")

            # ----------------------------------------------------------------
            # jupyter_exec — persistent Python kernel execution
            # ----------------------------------------------------------------
            if name == "jupyter_exec":
                import base64 as _b64_jup
                jup_code    = str(args.get("code", "")).strip()
                jup_session = str(args.get("session_id", "default")).strip() or "default"
                jup_timeout = max(1, min(300, int(args.get("timeout", 60))))
                jup_reset   = bool(args.get("reset", False))
                if not jup_code:
                    return _text("jupyter_exec: 'code' is required")
                try:
                    jup_r = await c.post(
                        f"{JUPYTER_URL}/exec",
                        json={"code": jup_code, "session_id": jup_session,
                              "timeout": jup_timeout, "reset": jup_reset},
                        timeout=float(jup_timeout) + 10,
                    )
                    if jup_r.status_code != 200:
                        return _text(f"jupyter_exec: service error {jup_r.status_code} — {jup_r.text[:300]}")
                    jup_data = jup_r.json()
                    blocks_jup: list[dict] = []
                    # Build text output
                    parts_jup: list[str] = []
                    if jup_data.get("error"):
                        parts_jup.append(f"❌ Error:\n{jup_data['error']}")
                    if jup_data.get("stdout", "").strip():
                        parts_jup.append(f"stdout:\n{jup_data['stdout'].strip()}")
                    if jup_data.get("stderr", "").strip():
                        parts_jup.append(f"stderr:\n{jup_data['stderr'].strip()}")
                    for out_jup in jup_data.get("outputs", []):
                        if str(out_jup).strip():
                            parts_jup.append(f"Out:\n{out_jup}")
                    if parts_jup:
                        blocks_jup.append({"type": "text", "text": "\n\n".join(parts_jup)})
                    elif not jup_data.get("images"):
                        blocks_jup.append({"type": "text", "text": "(no output)"})
                    # Inline plot images
                    for img_b64_jup in jup_data.get("images", [])[:4]:
                        try:
                            raw_jup = _b64_jup.b64decode(img_b64_jup)
                            encoded_jup = _b64_jup.b64encode(raw_jup).decode()
                            blocks_jup.append({
                                "type": "image",
                                "data": encoded_jup,
                                "mimeType": "image/png",
                            })
                        except Exception:
                            pass
                    return blocks_jup
                except Exception as exc_jup:
                    return _text(f"jupyter_exec: {exc_jup}")

            # ================================================================
            # browser_navigate — headless Chromium navigation
            # ================================================================
            if name == "browser_navigate":
                bn_url = str(args.get("url", "")).strip()
                if not bn_url:
                    return _text("browser_navigate: 'url' is required")
                try:
                    bn_payload: dict[str, Any] = {"url": bn_url}
                    if args.get("wait_until"):
                        bn_payload["wait_until"] = args["wait_until"]
                    br = await c.post(f"{BROWSER_AUTO_URL}/navigate", json=bn_payload, timeout=45)
                    br.raise_for_status()
                    bd = br.json()
                    blocks_bn: list[dict] = [{"type": "text", "text": f"Navigated to: {bd.get('url')}\nTitle: {bd.get('title')}"}]
                    if bd.get("screenshot_b64"):
                        blocks_bn.append({"type": "image", "data": bd["screenshot_b64"], "mimeType": "image/png"})
                    return blocks_bn
                except Exception as e_bn:
                    return _text(f"browser_navigate: {e_bn}")

            # ================================================================
            # browser_click — click on page element
            # ================================================================
            if name == "browser_click":
                try:
                    bc_payload: dict[str, Any] = {}
                    for k_bc in ("selector", "x", "y", "button", "click_count"):
                        if args.get(k_bc) is not None:
                            bc_payload[k_bc] = args[k_bc]
                    br = await c.post(f"{BROWSER_AUTO_URL}/click", json=bc_payload, timeout=20)
                    br.raise_for_status()
                    bd = br.json()
                    blocks_bc: list[dict] = [{"type": "text", "text": f"Clicked. URL: {bd.get('url')}"}]
                    if bd.get("screenshot_b64"):
                        blocks_bc.append({"type": "image", "data": bd["screenshot_b64"], "mimeType": "image/png"})
                    return blocks_bc
                except Exception as e_bc:
                    return _text(f"browser_click: {e_bc}")

            # ================================================================
            # browser_type — type text
            # ================================================================
            if name == "browser_type":
                bt_text = str(args.get("text", ""))
                try:
                    bt_payload: dict[str, Any] = {"text": bt_text}
                    if args.get("selector"):
                        bt_payload["selector"] = args["selector"]
                    if args.get("clear_first"):
                        bt_payload["clear_first"] = True
                    br = await c.post(f"{BROWSER_AUTO_URL}/type", json=bt_payload, timeout=20)
                    br.raise_for_status()
                    bd = br.json()
                    blocks_bt: list[dict] = [{"type": "text", "text": f"Typed {bd.get('text_length', 0)} chars"}]
                    if bd.get("screenshot_b64"):
                        blocks_bt.append({"type": "image", "data": bd["screenshot_b64"], "mimeType": "image/png"})
                    return blocks_bt
                except Exception as e_bt:
                    return _text(f"browser_type: {e_bt}")

            # ================================================================
            # browser_screenshot_page — screenshot
            # ================================================================
            if name == "browser_screenshot_page":
                try:
                    bs_payload: dict[str, Any] = {}
                    if args.get("full_page"):
                        bs_payload["full_page"] = True
                    if args.get("selector"):
                        bs_payload["selector"] = args["selector"]
                    br = await c.post(f"{BROWSER_AUTO_URL}/screenshot", json=bs_payload, timeout=15)
                    br.raise_for_status()
                    bd = br.json()
                    blocks_bs: list[dict] = [{"type": "text", "text": f"Screenshot of {bd.get('url')}\nTitle: {bd.get('title')}"}]
                    if bd.get("screenshot_b64"):
                        blocks_bs.append({"type": "image", "data": bd["screenshot_b64"], "mimeType": "image/png"})
                    return blocks_bs
                except Exception as e_bs:
                    return _text(f"browser_screenshot_page: {e_bs}")

            # ================================================================
            # browser_extract — extract content from page
            # ================================================================
            if name == "browser_extract":
                be_what = str(args.get("what", "text")).strip()
                try:
                    br = await c.post(f"{BROWSER_AUTO_URL}/extract", json={"what": be_what}, timeout=15)
                    br.raise_for_status()
                    bd = br.json()
                    if be_what == "text":
                        return _text(f"Page text from {bd.get('url')}:\n\n{bd.get('text', '')[:10000]}")
                    elif be_what == "links":
                        links_str = "\n".join(f"- [{l.get('text', '')}]({l.get('href', '')})" for l in bd.get("links", [])[:50])
                        return _text(f"Links ({bd.get('count', 0)}) from {bd.get('url')}:\n{links_str}")
                    else:
                        return _text(json.dumps(bd, indent=2)[:10000])
                except Exception as e_be:
                    return _text(f"browser_extract: {e_be}")

            # ================================================================
            # browser_keyboard — press keys
            # ================================================================
            if name == "browser_keyboard":
                bk_key = str(args.get("key", "")).strip()
                if not bk_key:
                    return _text("browser_keyboard: 'key' is required")
                try:
                    br = await c.post(f"{BROWSER_AUTO_URL}/keyboard", json={"key": bk_key}, timeout=10)
                    br.raise_for_status()
                    bd = br.json()
                    blocks_bk: list[dict] = [{"type": "text", "text": f"Pressed {bk_key}"}]
                    if bd.get("screenshot_b64"):
                        blocks_bk.append({"type": "image", "data": bd["screenshot_b64"], "mimeType": "image/png"})
                    return blocks_bk
                except Exception as e_bk:
                    return _text(f"browser_keyboard: {e_bk}")

            # ================================================================
            # browser_fill_form — fill form fields
            # ================================================================
            if name == "browser_fill_form":
                bf_fields = args.get("fields", [])
                if not bf_fields:
                    return _text("browser_fill_form: 'fields' array is required")
                try:
                    br = await c.post(f"{BROWSER_AUTO_URL}/fill_form", json={"fields": bf_fields}, timeout=20)
                    br.raise_for_status()
                    bd = br.json()
                    blocks_bf: list[dict] = [{"type": "text", "text": f"Filled {bd.get('filled', 0)}/{bd.get('total', 0)} fields"}]
                    if bd.get("screenshot_b64"):
                        blocks_bf.append({"type": "image", "data": bd["screenshot_b64"], "mimeType": "image/png"})
                    return blocks_bf
                except Exception as e_bf:
                    return _text(f"browser_fill_form: {e_bf}")

            # ================================================================
            # browser_scroll — scroll page
            # ================================================================
            if name == "browser_scroll":
                try:
                    bsc_payload: dict[str, Any] = {}
                    for k_bsc in ("direction", "amount", "selector"):
                        if args.get(k_bsc) is not None:
                            bsc_payload[k_bsc] = args[k_bsc]
                    br = await c.post(f"{BROWSER_AUTO_URL}/scroll", json=bsc_payload, timeout=10)
                    br.raise_for_status()
                    bd = br.json()
                    blocks_bsc: list[dict] = [{"type": "text", "text": "Scrolled"}]
                    if bd.get("screenshot_b64"):
                        blocks_bsc.append({"type": "image", "data": bd["screenshot_b64"], "mimeType": "image/png"})
                    return blocks_bsc
                except Exception as e_bsc:
                    return _text(f"browser_scroll: {e_bsc}")

            # ================================================================
            # browser_evaluate — run JavaScript
            # ================================================================
            if name == "browser_evaluate":
                bev_expr = str(args.get("expression", "")).strip()
                if not bev_expr:
                    return _text("browser_evaluate: 'expression' is required")
                try:
                    br = await c.post(f"{BROWSER_AUTO_URL}/evaluate", json={"expression": bev_expr}, timeout=15)
                    br.raise_for_status()
                    bd = br.json()
                    return _text(f"JS result: {json.dumps(bd.get('result'), indent=2, default=str)[:5000]}")
                except Exception as e_bev:
                    return _text(f"browser_evaluate: {e_bev}")

            # ================================================================
            # detect_objects — YOLOv8n object detection
            # ================================================================
            if name == "detect_objects":
                try:
                    do_payload: dict[str, Any] = {}
                    for k_do in ("image_base64", "image_url", "confidence", "classes"):
                        if args.get(k_do) is not None:
                            do_payload[k_do] = args[k_do]
                    dr = await c.post(f"{DETECT_URL}/objects", json=do_payload, timeout=30)
                    dr.raise_for_status()
                    dd = dr.json()
                    lines_do: list[str] = [f"Detected {dd.get('count', 0)} objects:"]
                    for cls_name, cls_count in dd.get("classes_found", {}).items():
                        lines_do.append(f"  {cls_name}: {cls_count}")
                    lines_do.append(f"\nImage: {dd.get('image_size', {}).get('width', '?')}x{dd.get('image_size', {}).get('height', '?')}")
                    for det in dd.get("detections", [])[:20]:
                        bbox = det.get("bbox", {})
                        lines_do.append(f"  [{det['confidence']:.2f}] {det['class']} at ({bbox.get('x1')},{bbox.get('y1')})-({bbox.get('x2')},{bbox.get('y2')})")
                    return _text("\n".join(lines_do))
                except Exception as e_do:
                    return _text(f"detect_objects: {e_do}")

            # ================================================================
            # detect_humans — human/person detection
            # ================================================================
            if name == "detect_humans":
                try:
                    dh_payload: dict[str, Any] = {}
                    for k_dh in ("image_base64", "image_url", "confidence"):
                        if args.get(k_dh) is not None:
                            dh_payload[k_dh] = args[k_dh]
                    dr = await c.post(f"{DETECT_URL}/humans", json=dh_payload, timeout=30)
                    dr.raise_for_status()
                    dd = dr.json()
                    lines_dh: list[str] = [f"Found {dd.get('count', 0)} people:"]
                    for i_dh, person in enumerate(dd.get("people", []), 1):
                        bbox = person.get("bbox", {})
                        lines_dh.append(f"  Person {i_dh}: confidence={person['confidence']:.2f} at ({bbox.get('x1')},{bbox.get('y1')})-({bbox.get('x2')},{bbox.get('y2')})")
                    return _text("\n".join(lines_dh))
                except Exception as e_dh:
                    return _text(f"detect_humans: {e_dh}")

            # ================================================================
            # list_tools_by_category — tool discovery helper
            # ================================================================
            if name == "list_tools_by_category":
                _TOOL_CATEGORIES: dict[str, list[str]] = {
                    "web": ["web(search)", "web(fetch)", "web(extract)", "web(summarize)",
                            "web(news)", "web(wikipedia)", "web(arxiv)", "web(youtube)"],
                    "browser": ["browser(navigate)", "browser(read)", "browser(click)", "browser(scroll)",
                                "browser(fill)", "browser(eval)", "browser(screenshot)", "browser(screenshot_search)",
                                "browser(bulk_screenshot)", "browser(scroll_screenshot)", "browser(screenshot_element)",
                                "browser(save_images)", "browser(download_images)", "browser(list_images)",
                                "browser(scrape)", "browser(keyboard)", "browser(fill_form)"],
                    "image": ["image(fetch)", "image(search)", "image(generate)", "image(edit)",
                              "image(crop)", "image(zoom)", "image(enhance)", "image(scan)",
                              "image(stitch)", "image(diff)", "image(annotate)", "image(caption)",
                              "image(upscale)", "image(remix)", "image(face_detect)", "image(similarity)"],
                    "document": ["document(ingest)", "document(tables)", "document(ocr)", "document(ocr_pdf)",
                                 "document(pdf_read)", "document(pdf_edit)", "document(pdf_form)",
                                 "document(pdf_merge)", "document(pdf_split)"],
                    "media": ["media(video_info)", "media(video_frames)", "media(video_thumbnail)",
                              "media(video_transcode)", "media(tts)", "media(detect_objects)", "media(detect_humans)"],
                    "data": ["data(store_article)", "data(search)", "data(cache_store)", "data(cache_get)",
                             "data(store_image)", "data(list_images)", "data(errors)"],
                    "memory": ["memory(store)", "memory(recall)"],
                    "knowledge": ["knowledge(add_node)", "knowledge(add_edge)", "knowledge(query)",
                                  "knowledge(path)", "knowledge(search)"],
                    "vector": ["vector(store)", "vector(search)", "vector(delete)", "vector(collections)",
                               "vector(embed_store)", "vector(embed_search)"],
                    "code": ["code(python)", "code(javascript)", "code(jupyter)"],
                    "custom_tools": ["custom_tools(create)", "custom_tools(list)",
                                     "custom_tools(delete)", "custom_tools(call)"],
                    "planner": ["planner(create)", "planner(get)", "planner(complete)", "planner(fail)",
                                "planner(list)", "planner(delete)", "planner(orchestrate)", "planner(plan)"],
                    "jobs": ["jobs(submit)", "jobs(status)", "jobs(result)", "jobs(cancel)",
                             "jobs(list)", "jobs(batch)"],
                    "research": ["research(rss_search)", "research(rss_push)",
                                 "research(deep)", "research(realtime)"],
                    "system": ["system(list_categories)", "system(instructions)",
                               "system(desktop_screenshot)", "system(desktop_control)"],
                    "utility": ["think"],
                }
                cat_q = str(args.get("category", "")).strip().lower()
                search_q = str(args.get("search", "")).strip().lower()

                # Build tool name→description lookup
                tool_desc: dict[str, str] = {}
                for t_entry in _TOOLS:
                    t_name = t_entry.get("name", "")
                    t_desc = t_entry.get("description", "")
                    if t_name:
                        tool_desc[t_name] = t_desc[:120]

                if cat_q and cat_q in _TOOL_CATEGORIES:
                    tool_names = _TOOL_CATEGORIES[cat_q]
                    lines_tc: list[str] = [f"Category: {cat_q} ({len(tool_names)} tools)\n"]
                    for tn in tool_names:
                        lines_tc.append(f"  {tn}: {tool_desc.get(tn, '(no description)')}")
                    return _text("\n".join(lines_tc))
                elif search_q:
                    matches: list[str] = []
                    for tn, td in tool_desc.items():
                        if search_q in tn.lower() or search_q in td.lower():
                            matches.append(f"  {tn}: {td}")
                    if matches:
                        return _text(f"Tools matching '{search_q}' ({len(matches)}):\n" + "\n".join(matches[:20]))
                    return _text(f"No tools matching '{search_q}'")
                else:
                    # List all categories
                    lines_all: list[str] = ["Available categories:\n"]
                    for cat_name, cat_tools in sorted(_TOOL_CATEGORIES.items()):
                        lines_all.append(f"  {cat_name} ({len(cat_tools)} tools)")
                    lines_all.append("\nCall with category='<name>' to list tools in that category.")
                    return _text("\n".join(lines_all))

            return _text(f"Unknown tool: {name}")

        except Exception as exc:
            return _text(f"Error calling {name}: {exc}")


# ---------------------------------------------------------------------------
# JSON-RPC dispatch
# ---------------------------------------------------------------------------

async def _handle_rpc(req: dict[str, Any]) -> dict[str, Any] | None:
    """Process one JSON-RPC request; return response dict or None for notifications."""
    req_id = req.get("id")
    method = req.get("method", "")
    params = req.get("params") or {}

    def ok(result: Any) -> dict:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def err(code: int, msg: str) -> dict:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": msg}}

    if method == "initialize":
        # Echo back the client's requested version if we support it; otherwise
        # fall back to 2024-11-05.  LM Studio 0.3.6+ sends "2025-03-26" and
        # will only enable image rendering when the agreed version matches.
        client_ver = params.get("protocolVersion", "2024-11-05")
        agreed_ver = client_ver if client_ver in {"2024-11-05", "2025-03-26"} else "2024-11-05"
        return ok({
            "protocolVersion": agreed_ver,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "aichat", "version": "1.0.0"},
        })

    if method in ("notifications/initialized", "initialized"):
        return None  # notification — no response

    if method == "tools/list":
        return ok({"tools": _TOOLS})

    if method == "tools/call":
        tool_name  = params.get("name", "")
        arguments  = params.get("arguments") or {}
        content_blocks = await _get_orchestrator().execute(
            tool_name, _call_tool, tool_name, arguments
        )
        tool_error = _blocks_indicate_error(tool_name, content_blocks)
        # DIRECTIVE: Images must ALWAYS render.  Enforce on any image tool
        # (both mega-tool names and resolved handler names) even on error,
        # and also enforce on ANY response that already contains image blocks
        # to guarantee compression/sizing for LM Studio's payload cap.
        _is_img_tool = ImageRenderingPolicy.is_image_tool(tool_name)
        _has_img = ImageRenderingPolicy.has_image(content_blocks)
        if _is_img_tool or _has_img:
            content_blocks = ImageRenderingPolicy.enforce(content_blocks)
        is_error = _blocks_indicate_error(tool_name, content_blocks)
        return ok({
            "content": content_blocks,
            "isError": is_error,
        })

    if method == "ping":
        return ok({})

    if req_id is not None:
        return err(-32601, f"Method not found: {method}")
    return None


_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
    "Access-Control-Allow-Origin": "*",
}


# ---------------------------------------------------------------------------
# MCP 2025-03-26 Streamable HTTP transport  (preferred by LM Studio 0.3.6+)
# POST /mcp — single endpoint; responds with JSON or SSE depending on Accept
# GET  /mcp — SSE stream, same session handshake as /sse
# ---------------------------------------------------------------------------

@app.post("/mcp")
async def mcp_post(request: Request) -> Response:
    """
    Streamable HTTP transport (MCP 2025-03-26).
    Clients that prefer a single endpoint send JSON-RPC here.
    If the client sent Accept: text/event-stream we stream back; otherwise JSON.
    """
    body = await request.body()
    try:
        rpc = json.loads(body)
    except json.JSONDecodeError:
        return Response(
            content=json.dumps({"jsonrpc": "2.0", "id": None,
                                "error": {"code": -32700, "message": "Parse error"}}),
            media_type="application/json",
            status_code=400,
        )

    method = rpc.get("method", "")
    accept = request.headers.get("accept", "")

    # MCP 2025-03-26 §3.3: server MUST return Mcp-Session-Id on initialize so
    # that LM Studio can enable image rendering and session tracking.
    extra_headers: dict[str, str] = {}
    if method == "initialize":
        extra_headers["Mcp-Session-Id"] = str(uuid.uuid4())

    if "text/event-stream" in accept:
        # Non-blocking: stream keepalives while the tool runs, yield result when done.
        async def _stream_result(rpc: dict):
            task = asyncio.create_task(_handle_rpc(rpc))
            elapsed = 0.0
            while not task.done():
                yield ": keepalive\n\n"
                await asyncio.sleep(5.0)
                elapsed += 5.0
                if elapsed >= _TOOL_TIMEOUT:
                    task.cancel()
                    req_id = rpc.get("id")
                    error_resp = {
                        "jsonrpc": "2.0", "id": req_id,
                        "error": {"code": -32000, "message": "Tool execution timed out"},
                    }
                    yield f"event: message\ndata: {json.dumps(error_resp)}\n\n"
                    return
            try:
                result = task.result()
            except Exception as exc:
                req_id = rpc.get("id")
                error_resp = {
                    "jsonrpc": "2.0", "id": req_id,
                    "error": {"code": -32000, "message": str(exc)},
                }
                yield f"event: message\ndata: {json.dumps(error_resp)}\n\n"
                return
            if result is not None:
                yield f"event: message\ndata: {json.dumps(result)}\n\n"
        return StreamingResponse(_stream_result(rpc), media_type="text/event-stream",
                                 headers={**_SSE_HEADERS, **extra_headers})

    # JSON (non-SSE) path — synchronous; used only by non-LM-Studio clients.
    response = await _handle_rpc(rpc)
    if response is None:
        return Response(content="", status_code=202, headers=extra_headers)
    return Response(content=json.dumps(response), media_type="application/json",
                    headers=extra_headers)


@app.get("/mcp")
async def mcp_sse(request: Request) -> StreamingResponse:
    """GET /mcp — SSE stream, same as GET /sse (alias for the Streamable HTTP spec)."""
    return await sse_connect(request)


# ---------------------------------------------------------------------------
# Legacy SSE transport (MCP 2024-11-05) — kept for backward compatibility
# ---------------------------------------------------------------------------

@app.get("/sse")
async def sse_connect(request: Request) -> StreamingResponse:
    session_id = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    _sessions[session_id] = queue

    async def event_stream():
        yield f"event: endpoint\ndata: /messages?sessionId={session_id}\n\n"
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=5.0)
                    yield f"event: message\ndata: {json.dumps(msg)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            _sessions.pop(session_id, None)

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers=_SSE_HEADERS)


@app.post("/messages")
async def messages(request: Request, sessionId: str = "") -> Response:
    body = await request.body()
    try:
        rpc = json.loads(body)
    except json.JSONDecodeError:
        return Response(
            content=json.dumps({"jsonrpc": "2.0", "id": None,
                                "error": {"code": -32700, "message": "Parse error"}}),
            media_type="application/json",
            status_code=400,
        )

    async def _deliver(rpc: dict, sid: str) -> None:
        _current_session_id.set(sid)
        _current_request_id.set(str(rpc.get("id", "")))
        task = asyncio.create_task(_handle_rpc(rpc))
        req_id = rpc.get("id")
        elapsed = 0.0
        # Send MCP progress notifications every 5 s while the tool runs.
        # These are real data: events on the SSE stream — keeps LM Studio's
        # internal tool-call timer alive even for slow tools (screenshot, etc.).
        while not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
            except asyncio.TimeoutError:
                elapsed += 5.0
                if elapsed >= _TOOL_TIMEOUT:
                    task.cancel()
                    if sid in _sessions and req_id is not None:
                        await _sessions[sid].put({
                            "jsonrpc": "2.0", "id": req_id,
                            "error": {"code": -32000, "message": "Tool execution timed out"},
                        })
                    return
                if sid in _sessions and req_id is not None:
                    await _sessions[sid].put({
                        "jsonrpc": "2.0",
                        "method": "notifications/progress",
                        "params": {
                            "progressToken": str(req_id),
                            "progress": 0,
                            "total": 1,
                        },
                    })
            except Exception:
                break
        try:
            response = task.result()
        except Exception as exc:
            if sid in _sessions and req_id is not None:
                await _sessions[sid].put({
                    "jsonrpc": "2.0", "id": req_id,
                    "error": {"code": -32000, "message": str(exc)},
                })
            return
        if response is not None and sid in _sessions:
            await _sessions[sid].put(response)

    asyncio.create_task(_deliver(rpc, sessionId))
    return Response(content="", status_code=202)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    gpu = GpuDetector.detect()
    return {
        "ok": True,
        "sessions": len(_sessions),
        "tools": len(_TOOLS),
        "transports": ["POST /mcp (streamable-http)", "GET /sse (sse)", "GET /mcp (sse-alias)"],
        "gpu": {
            "vendor": gpu.get("vendor", "none"),
            "name": gpu.get("name", "unknown"),
        },
        "cv2_accel": {
            "cv2": bool(_CV2_ACCEL_STATUS.get("cv2", False)),
            "cv2_version": _CV2_ACCEL_STATUS.get("cv2_version", ""),
            "backend": _GpuImg.backend(),
            "opencl_have": bool(_CV2_ACCEL_STATUS.get("opencl_have", False)),
            "opencl_use": bool(_CV2_ACCEL_STATUS.get("opencl_use", False)),
            "cuda_devices": int(_CV2_ACCEL_STATUS.get("cuda_devices", 0)),
        },
        "orchestrator": {
            "active_jobs": _get_orchestrator().resource_state().active_jobs,
            "active_gpu_tasks": _get_orchestrator().resource_state().active_gpu_tasks,
            "active_cpu_tasks": _get_orchestrator().resource_state().active_cpu_tasks,
            "gpu_pressure": _get_orchestrator().resource_state().gpu_pressure,
            "cpu_pressure": _get_orchestrator().resource_state().cpu_pressure,
        },
    }
