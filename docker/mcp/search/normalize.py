"""Query normalization, URL scoring, and content filtering for search tools."""
from __future__ import annotations

import re
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


def domain_from_url(url: str) -> str:
    """Lowercased hostname without leading www., or empty on parse failure."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return ""
    return host.removeprefix("www.")


def url_has_explicit_content(url: str, text: str = "") -> bool:  # noqa: ARG001
    """Content filter stub — always passes."""
    return False


# ---------------------------------------------------------------------------
# Query normalization
# ---------------------------------------------------------------------------

def normalize_search_query(query: str) -> tuple[str, str]:
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


def search_terms(query: str) -> list[str]:
    """Tokenize query into lowercased word terms for URL relevance scoring."""
    return [w.lower() for w in re.findall(r"[a-z0-9]{3,}", (query or "").lower())]


def query_preferred_domains(query: str) -> tuple[str, ...]:
    """Domain preferences for known entities to improve relevance ordering."""
    q = (query or "").lower()
    if any(k in q for k in ("girls frontline 2", "gfl2", "klukai")):
        return ("iopwiki.com", "gf2exilium.com", "fandom.com", "prydwen.gg")
    return ()


def score_url_relevance(
    url: str,
    query_terms: list[str],
    preferred_domains: tuple[str, ...] = (),
) -> int:
    """Simple URL relevance score used for ranking candidate pages/images."""
    url_l = (url or "").lower()
    host = domain_from_url(url_l)
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
