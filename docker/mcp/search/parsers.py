"""HTML link extractors for DDG, Bing, Google search result pages.

Used as Tier 1b fallback when SearXNG JSON parsing is unavailable.
"""
from __future__ import annotations

import base64
import html as _html
import re
from urllib.parse import parse_qs, unquote as url_unquote, urlparse

from search.normalize import domain_from_url


# ---------------------------------------------------------------------------
# DuckDuckGo
# ---------------------------------------------------------------------------

def unwrap_ddg_redirect(url: str) -> str:
    """Return target URL when the input is a DuckDuckGo redirect URL."""
    u = (url or "").strip()
    if u.startswith("//"):
        u = "https:" + u
    if not u:
        return ""
    try:
        parsed = urlparse(u)
        host = (parsed.hostname or "").lower()
        if "duckduckgo.com" in host and parsed.path == "/l/":
            cand = url_unquote((parse_qs(parsed.query).get("uddg") or [""])[0])
            if cand.startswith("http"):
                return cand
        m = re.search(r"uddg=(https?%3A[^&\s\"'>]+)", u)
        if m:
            cand2 = url_unquote(m.group(1))
            if cand2.startswith("http"):
                return cand2
    except Exception:
        return u
    return u


def extract_ddg_links(html_text: str, max_results: int = 12) -> list[tuple[str, str]]:
    """Parse DDG HTML search results into [(url, title)] with deduped URLs."""
    links: list[tuple[str, str]] = []
    seen: set[str] = set()

    for m in re.finditer(
        r"<a[^>]+class=[\"'][^\"']*result__a[^\"']*[\"'][^>]+href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>",
        html_text, flags=re.IGNORECASE | re.DOTALL,
    ):
        href = _html.unescape(m.group(1))
        title = re.sub(r"<[^>]+>", " ", m.group(2))
        title = _html.unescape(re.sub(r"\s+", " ", title).strip())
        url = unwrap_ddg_redirect(href)
        if not url.startswith("http") or url in seen:
            continue
        links.append((url, title or url))
        seen.add(url)
        if len(links) >= max_results:
            return links

    for enc in re.findall(r"uddg=(https?%3A[^&\"'>\s]+)", html_text):
        url = url_unquote(enc)
        if not url.startswith("http") or url in seen:
            continue
        links.append((url, url))
        seen.add(url)
        if len(links) >= max_results:
            break
    return links


# ---------------------------------------------------------------------------
# Bing
# ---------------------------------------------------------------------------

def _unwrap_bing_redirect(url: str) -> str:
    u = (url or "").strip()
    try:
        p = urlparse(u)
        host = (p.hostname or "").lower()
        if "bing.com" in host and p.path.startswith("/ck/a"):
            raw_u = (parse_qs(p.query).get("u") or [""])[0]
            if raw_u.startswith("a1"):
                b64 = raw_u[2:]
                b64 += "=" * ((4 - (len(b64) % 4)) % 4)
                dec = base64.urlsafe_b64decode(b64.encode("ascii")).decode("utf-8", errors="ignore")
                if dec.startswith("http"):
                    return dec
    except Exception:
        return u
    return u


def extract_bing_links(html_text: str, max_results: int = 12) -> list[tuple[str, str]]:
    """Parse Bing HTML web search results into [(url, title)]."""
    links: list[tuple[str, str]] = []
    seen: set[str] = set()

    for m in re.finditer(
        r"<li[^>]+class=[\"'][^\"']*b_algo[^\"']*[\"'][^>]*>.*?<h2[^>]*>.*?"
        r"<a[^>]+href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>",
        html_text, flags=re.IGNORECASE | re.DOTALL,
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


# ---------------------------------------------------------------------------
# Google
# ---------------------------------------------------------------------------

def extract_google_links(html_text: str, max_results: int = 12) -> list[tuple[str, str]]:
    """Parse Google HTML search results into [(url, title)]."""
    links: list[tuple[str, str]] = []
    seen: set[str] = set()

    def _unwrap_google(href: str) -> str:
        if href.startswith("/url?") or href.startswith("https://www.google.com/url?"):
            qs = parse_qs(urlparse(href).query)
            cand = (qs.get("q") or [""])[0]
            if cand.startswith("http"):
                return cand
        return href

    for m in re.finditer(
        r'<a[^>]+href="([^"]*)"[^>]*>(?:[^<]|<(?!h3))*?<h3[^>]*>(.*?)</h3>',
        html_text, flags=re.IGNORECASE | re.DOTALL,
    ):
        url = _unwrap_google(_html.unescape(m.group(1)))
        title = re.sub(r"<[^>]+>", " ", m.group(2))
        title = _html.unescape(re.sub(r"\s+", " ", title).strip())
        if not url.startswith("http") or url in seen:
            continue
        host = domain_from_url(url)
        if "google.com" in host or "googleapis.com" in host:
            continue
        links.append((url, title or url))
        seen.add(url)
        if len(links) >= max_results:
            return links

    for m in re.finditer(r'<cite[^>]*>([^<]+)</cite>', html_text, flags=re.IGNORECASE):
        raw = _html.unescape(m.group(1)).strip().split(" ")[0]
        if not raw.startswith("http"):
            raw = "https://" + raw
        url = raw.split("\u203a")[0].strip().rstrip("/")
        if not url.startswith("http") or url in seen:
            continue
        host = domain_from_url(url)
        if "google.com" in host:
            continue
        links.append((url, url))
        seen.add(url)
        if len(links) >= max_results:
            break

    return links
