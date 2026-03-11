"""Backend availability tracker with TTL-cached health probes.

Maintains a lazy, cached view of whether the primary CompreFace and Triton
(via vision-router) backends are reachable. Probes are only fired when the
cached result is stale, so there is no background task and no per-request
overhead beyond a monotonic clock comparison.

Usage:
    from .backends import get_backend_health

    health = get_backend_health()
    if await health.compreface_primary_ok():
        client = CompreFaceClient(settings.compreface)
    else:
        client = CompreFaceClient(settings.fallback_compreface_config())
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# How long a probe result is considered fresh before re-probing.
_DEFAULT_TTL_S = 30.0
# Timeout for individual health-check HTTP calls.
_PROBE_TIMEOUT_S = 3.0


class BackendHealth:
    """Cached health state for the two remote backends."""

    def __init__(self, ttl_s: float = _DEFAULT_TTL_S) -> None:
        self._ttl_s = ttl_s

        self._cf_ok: bool | None = None
        self._cf_last: float = 0.0
        self._cf_lock: asyncio.Lock | None = None

        self._triton_ok: bool | None = None
        self._triton_last: float = 0.0
        self._triton_lock: asyncio.Lock | None = None

    # ------------------------------------------------------------------
    # Lock helpers (created lazily inside the running event loop)
    # ------------------------------------------------------------------

    def _get_cf_lock(self) -> asyncio.Lock:
        if self._cf_lock is None:
            self._cf_lock = asyncio.Lock()
        return self._cf_lock

    def _get_triton_lock(self) -> asyncio.Lock:
        if self._triton_lock is None:
            self._triton_lock = asyncio.Lock()
        return self._triton_lock

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def compreface_primary_ok(self) -> bool:
        """Return True if primary CompreFace is reachable."""
        from .config import get_settings
        now = time.monotonic()
        if self._cf_ok is not None and now - self._cf_last < self._ttl_s:
            return self._cf_ok

        async with self._get_cf_lock():
            # Re-check inside lock to avoid redundant probes under concurrency.
            if self._cf_ok is not None and time.monotonic() - self._cf_last < self._ttl_s:
                return self._cf_ok
            url = get_settings().compreface.url
            result = await self._probe_http(f"{url.rstrip('/')}/actuator/health", label="compreface-primary")
            self._cf_ok = result
            self._cf_last = time.monotonic()
            if not result:
                logger.warning("Primary CompreFace unreachable at %s — will use local fallback", url)
            return self._cf_ok

    async def triton_ok(self) -> bool:
        """Return True if vision-router (Triton proxy) is reachable."""
        from .config import get_settings
        now = time.monotonic()
        if self._triton_ok is not None and now - self._triton_last < self._ttl_s:
            return self._triton_ok

        async with self._get_triton_lock():
            if self._triton_ok is not None and time.monotonic() - self._triton_last < self._ttl_s:
                return self._triton_ok
            url = get_settings().router.url
            result = await self._probe_http(f"{url.rstrip('/')}/v1/health", label="vision-router")
            self._triton_ok = result
            self._triton_last = time.monotonic()
            if not result:
                logger.warning("Vision-router (Triton) unreachable at %s — Triton tools unavailable", url)
            return self._triton_ok

    def invalidate(self) -> None:
        """Force both cached states to expire on next access."""
        self._cf_last = 0.0
        self._triton_last = 0.0

    async def status(self) -> dict[str, Any]:
        """Return a dict suitable for the /health endpoint."""
        cf = await self.compreface_primary_ok()
        tr = await self.triton_ok()
        return {
            "compreface_primary": "ok" if cf else "unreachable",
            "triton": "ok" if tr else "unreachable",
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    async def _probe_http(url: str, *, label: str) -> bool:
        try:
            async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_S) as c:
                resp = await c.get(url)
                # 200 = healthy; 400/401/403 with "x-api-key" body means CompreFace
                # is up but auth is required — still counts as reachable.
                if resp.status_code == 200:
                    return True
                if resp.status_code in (400, 401, 403) and b"x-api-key" in resp.content:
                    return True
                return False
        except Exception as exc:
            logger.debug("Health probe failed for %s (%s): %s", label, url, exc)
            return False


# Module-level singleton — shared across all requests within a process.
_backend_health = BackendHealth()


def get_backend_health() -> BackendHealth:
    return _backend_health
