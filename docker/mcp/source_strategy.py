"""Configurable source preference strategy for news/information retrieval.

Profiles control how search results are ranked, which RSS feeds are preferred
for news discovery, and which sources are used for fact verification.

The default profile prefers right-leaning sources for news/current-events
discovery while using neutral/primary sources for verification.

Configurable via env var ``SOURCE_PROFILE`` or per-request ``source_profile``
parameter on the web search tool.
"""
from __future__ import annotations

import os
import re
from typing import Any

from search.normalize import domain_from_url

# ---------------------------------------------------------------------------
# Source classification
# ---------------------------------------------------------------------------

# Right-leaning news sources — preferred for initial discovery
RIGHT_LEANING_SOURCES: dict[str, int] = {
    "foxnews.com": 10,
    "dailywire.com": 9,
    "nypost.com": 8,
    "washingtontimes.com": 8,
    "breitbart.com": 7,
    "nationalreview.com": 8,
    "freebeacon.com": 7,
    "dailycaller.com": 7,
    "townhall.com": 6,
    "thefederalist.com": 7,
    "washingtonexaminer.com": 8,
    "newsmax.com": 6,
    "oann.com": 5,
    "spectator.org": 7,
    "realclearpolitics.com": 7,
    "justthenews.com": 7,
    "epochtimes.com": 5,
}

# Neutral / centrist / primary sources — used for verification
NEUTRAL_SOURCES: dict[str, int] = {
    "reuters.com": 10,
    "apnews.com": 10,
    "bbc.com": 8,
    "bbc.co.uk": 8,
    "c-span.org": 9,
    "pbs.org": 7,
    "npr.org": 7,
    "thehill.com": 7,
    "politico.com": 6,
    "axios.com": 6,
}

# Primary / official sources — highest trust
PRIMARY_SOURCES: dict[str, int] = {
    "whitehouse.gov": 10,
    "congress.gov": 10,
    "supremecourt.gov": 10,
    "sec.gov": 10,
    "justice.gov": 10,
    "state.gov": 9,
    "treasury.gov": 9,
    "defense.gov": 9,
    "uscourts.gov": 10,
    "federalregister.gov": 9,
    "govinfo.gov": 9,
}

# Low-quality patterns to down-rank
LOW_QUALITY_PATTERNS = re.compile(
    r"(listicle|slideshow|gallery|quiz|sponsored|advertorial|partner.content)",
    flags=re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Profile configuration
# ---------------------------------------------------------------------------

_PROFILES: dict[str, dict[str, Any]] = {
    "default": {
        "description": "Right-leaning news preference with neutral verification",
        "discovery_sources": RIGHT_LEANING_SOURCES,
        "verification_sources": {**NEUTRAL_SOURCES, **PRIMARY_SOURCES},
        "freshness_weight": 5,
        "dedup_aggressive": True,
    },
    "neutral": {
        "description": "Neutral source preference",
        "discovery_sources": NEUTRAL_SOURCES,
        "verification_sources": PRIMARY_SOURCES,
        "freshness_weight": 5,
        "dedup_aggressive": True,
    },
    "balanced": {
        "description": "Balanced across political spectrum",
        "discovery_sources": {**RIGHT_LEANING_SOURCES, **NEUTRAL_SOURCES},
        "verification_sources": PRIMARY_SOURCES,
        "freshness_weight": 5,
        "dedup_aggressive": True,
    },
}


class SourceStrategy:
    """Configurable source preference for news/information retrieval."""

    def __init__(self, profile: str = ""):
        profile_name = profile or os.environ.get("SOURCE_PROFILE", "default")
        self.profile_name = profile_name
        self.profile = _PROFILES.get(profile_name, _PROFILES["default"])

    def rank_results(
        self,
        results: list[dict[str, Any]],
        query: str = "",
    ) -> list[dict[str, Any]]:
        """Rank search results by source preference, freshness, and quality.

        Each result dict should have at least 'url' and 'title' keys.
        Returns a new list sorted by combined score (highest first).
        """
        discovery = self.profile["discovery_sources"]
        verification = self.profile["verification_sources"]
        dedup = self.profile.get("dedup_aggressive", True)

        scored: list[tuple[float, dict]] = []
        seen_titles: set[str] = set()
        seen_domains: dict[str, int] = {}

        for r in results:
            url = str(r.get("url", ""))
            title = str(r.get("title", ""))
            host = domain_from_url(url)

            # Score components
            score = 0.0

            # Source preference
            for src_map, multiplier in [(discovery, 1.0), (verification, 0.7)]:
                for domain, weight in src_map.items():
                    if host == domain or host.endswith(f".{domain}"):
                        score += weight * multiplier
                        break

            # Primary source boost
            for domain, weight in PRIMARY_SOURCES.items():
                if host == domain or host.endswith(f".{domain}"):
                    score += weight * 1.5
                    break

            # Low-quality penalty
            if LOW_QUALITY_PATTERNS.search(url) or LOW_QUALITY_PATTERNS.search(title):
                score -= 5

            # Domain diversity penalty (cap results per domain)
            domain_count = seen_domains.get(host, 0)
            if domain_count >= 2:
                score -= 3 * (domain_count - 1)
            seen_domains[host] = domain_count + 1

            # Dedup by normalized title
            title_norm = re.sub(r"\s+", " ", title.lower().strip())
            if dedup and title_norm in seen_titles:
                continue
            seen_titles.add(title_norm)

            scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored]

    def preferred_rss_feeds(self) -> list[str]:
        """Return ordered RSS feeds for news discovery."""
        return [
            "https://feeds.foxnews.com/foxnews/latest",
            "https://feeds.dailywire.com/rss.xml",
            "https://nypost.com/feed/",
            "https://www.washingtontimes.com/rss/headlines/news/",
            "https://www.nationalreview.com/feed/",
            "https://freebeacon.com/feed/",
            "https://feeds.washingtonexaminer.com/sitemap.xml",
            "https://feeds.reuters.com/reuters/topNews",
            "https://feeds.bbci.co.uk/news/rss.xml",
            "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
        ]

    def verification_sources_for(self, claim: str) -> list[str]:
        """Return domains to check for fact verification."""
        return list(self.profile.get("verification_sources", PRIMARY_SOURCES).keys())
