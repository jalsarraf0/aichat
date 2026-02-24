"""
BrowserTool — controls a real Chromium browser running in the human_browser
Docker container via a thin Playwright/FastAPI server that is deployed on
first use and kept alive for the session.

Architecture
------------
* First call: deploys browser_server.py into /workspace inside the container
  (using `docker cp`), then starts it with `docker exec -d ... uvicorn ...`.
* Subsequent calls: HTTP POST to http://<container-bridge-ip>:7081/<action>.
* The server persists across aichat restarts (the container keeps running).
* Version check: if the running server is an older version it is killed and
  redeployed automatically so _SERVER_SRC changes take effect on next use.
* Fallback: if the browser server fails, httpx fetches the URL directly.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime

import httpx

BROWSER_CONTAINER = "human_browser"
BROWSER_PORT = 7081
_STARTUP_TIMEOUT = 35  # seconds to wait for uvicorn to come up

# When _ensure_server() finds a running server whose /health returns a
# different version it kills it and redeploys the current _SERVER_SRC.
_REQUIRED_SERVER_VERSION = "5"

# ---------------------------------------------------------------------------
# FastAPI Playwright server — injected into the container at first use
# ---------------------------------------------------------------------------
# NOTE: this string is written verbatim into the container; it must not
# contain anything that would break when piped through docker cp.

_SERVER_SRC = '''\
"""Browser automation server — auto-deployed by aichat BrowserTool. v5"""
from __future__ import annotations

import asyncio
import random as _random
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from playwright.async_api import async_playwright

_VERSION = "5"
_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
# Extended stealth: hide automation signals, spoof navigator properties,
# restore chrome runtime so sites don\'t detect a bare Chromium shell.
_STEALTH_JS = """
Object.defineProperty(navigator, \'webdriver\', {get: () => undefined});
Object.defineProperty(navigator, \'plugins\', {get: () => [1, 2, 3, 4, 5]});
Object.defineProperty(navigator, \'languages\', {get: () => [\'en-US\', \'en\']});
window.chrome = {runtime: {}};
try {
    const origQuery = window.navigator.permissions.query;
    window.navigator.permissions.query = (p) =>
        p.name === \'notifications\'
        ? Promise.resolve({state: Notification.permission})
        : origQuery(p);
} catch(_) {}
"""

# Pool of common desktop viewport sizes — picked randomly each new context
# to avoid the fixed-1920×1080 fingerprint.
_VIEWPORTS = [
    {"width": 1366, "height": 768},
    {"width": 1440, "height": 900},
    {"width": 1920, "height": 1080},
    {"width": 1280, "height": 720},
    {"width": 1536, "height": 864},
]

_pw = _browser = _context = _page = None
_page_lock = None  # asyncio.Lock — created lazily inside first async call


async def _new_context():
    global _context
    vp = _random.choice(_VIEWPORTS)
    _context = await _browser.new_context(
        user_agent=_UA,
        viewport=vp,
        locale="en-US",
    )
    await _context.add_init_script(_STEALTH_JS)
    return _context


_CHROMIUM_PATHS = [
    # Playwright-managed headless Chromium (preferred — matches playwright version)
    None,
    # System Chromium fallback (available in the human_browser base image)
    "/usr/bin/chromium",
    "/usr/lib/chromium/chromium",
    "/usr/bin/chromium-browser",
    "/usr/bin/google-chrome",
]

_LAUNCH_ARGS = [
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--disable-blink-features=AutomationControlled",
    "--disable-infobars",
    "--window-size=1920,1080",
]


@asynccontextmanager
async def lifespan(app):
    global _pw, _browser, _context, _page
    _pw = await async_playwright().start()
    last_exc: Exception | None = None
    for exe in _CHROMIUM_PATHS:
        try:
            kwargs = {"headless": True, "args": _LAUNCH_ARGS}
            if exe:
                kwargs["executable_path"] = exe
            _browser = await _pw.chromium.launch(**kwargs)
            break
        except Exception as exc:
            last_exc = exc
    else:
        await _pw.stop()
        raise RuntimeError(f"Could not launch Chromium: {last_exc}")
    await _new_context()
    _page = await _context.new_page()
    yield
    await _browser.close()
    await _pw.stop()


app = FastAPI(lifespan=lifespan)


@app.exception_handler(Exception)
async def _global_exc(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=200,
        content={"error": str(exc), "content": "", "title": "", "url": ""},
    )


async def _ensure_page():
    """Return a healthy page, recreating it (and context if needed) after a crash."""
    global _context, _page, _page_lock
    if _page_lock is None:
        _page_lock = asyncio.Lock()
    async with _page_lock:
        try:
            if _page is not None and not _page.is_closed():
                return _page
        except Exception:
            pass
        # Page is gone — recreate
        try:
            if _context is None or _context.is_closed():
                await _new_context()
            _page = await _context.new_page()
        except Exception:
            await _new_context()
            _page = await _context.new_page()
        return _page


async def _extract_text(page) -> str:
    js = """() => {
        const b = document.body;
        if (!b) return \'\' ;
        b.querySelectorAll(
            \'script,style,nav,footer,header,aside,noscript\'
        ).forEach(e => e.remove());
        return b.innerText.trim().slice(0, 8000);
    }"""
    try:
        return await page.evaluate(js)
    except Exception:
        return ""


async def _get_image_urls(page) -> list:
    """Extract direct http(s) image URLs from the current page DOM."""
    js = """() => {
        const seen = new Set();
        const result = [];
        for (const img of document.querySelectorAll(\'img[src]\')) {
            const src = img.src || \'\' ;
            if (src.startsWith(\'http\') && !seen.has(src)) {
                seen.add(src);
                result.push(src);
            }
        }
        for (const el of document.querySelectorAll(\'[srcset]\')) {
            const srcset = el.getAttribute(\'srcset\') || \'\' ;
            for (const part of srcset.split(\',\')) {
                const tokens = part.trim().split(\' \').filter(function(t) {
                    return t.length > 0;
                });
                const url = tokens[0] || \'\' ;
                if (url.startsWith(\'http\') && !seen.has(url)) {
                    seen.add(url);
                    result.push(url);
                }
            }
        }
        const og = document.querySelector(\'meta[property="og:image"]\');
        if (og) {
            const c = og.getAttribute(\'content\') || \'\' ;
            if (c.startsWith(\'http\') && !seen.has(c)) {
                seen.add(c);
                result.push(c);
            }
        }
        return result.slice(0, 30);
    }"""
    try:
        return await page.evaluate(js)
    except Exception:
        return []


async def _safe_goto(page, url: str) -> None:
    """Navigate to url. \'load\' is fast and sufficient; networkidle can stall 30s+."""
    for wait, ms in ((\"load\", 12000), (\"domcontentloaded\", 5000)):
        try:
            await page.goto(url, wait_until=wait, timeout=ms)
            return
        except Exception:
            if page.is_closed():
                raise
            pass


@app.get("/health")
async def health():
    return {"ok": True, "version": _VERSION}


class NavReq(BaseModel):
    url: str
    wait: str = "networkidle"


@app.post("/navigate")
async def navigate(req: NavReq):
    try:
        page = await _ensure_page()
        await _safe_goto(page, req.url)
        return {
            "title": await page.title(),
            "url": page.url,
            "content": await _extract_text(page),
        }
    except Exception as exc:
        try:
            await _ensure_page()
        except Exception:
            pass
        return {"error": str(exc), "content": "", "title": "", "url": req.url}


class ClickReq(BaseModel):
    selector: str


@app.post("/click")
async def click(req: ClickReq):
    try:
        page = await _ensure_page()
        await page.click(req.selector, timeout=10000)
        try:
            await page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            pass
        return {"url": page.url, "content": await _extract_text(page)}
    except Exception as exc:
        return {"error": str(exc), "content": "", "url": ""}


class FillReq(BaseModel):
    selector: str
    value: str


@app.post("/fill")
async def fill(req: FillReq):
    try:
        page = await _ensure_page()
        await page.fill(req.selector, req.value)
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@app.get("/read")
async def read():
    try:
        page = await _ensure_page()
        return {
            "title": await page.title(),
            "url": page.url,
            "content": await _extract_text(page),
        }
    except Exception as exc:
        return {"error": str(exc), "content": "", "title": "", "url": ""}


class EvalReq(BaseModel):
    code: str


@app.post("/eval")
async def evaluate(req: EvalReq):
    try:
        page = await _ensure_page()
        result = await page.evaluate(req.code)
        return {"result": str(result) if result is not None else "null"}
    except Exception as exc:
        return {"error": str(exc), "result": "null"}


class ScreenshotReq(BaseModel):
    url: Optional[str] = None
    path: str = "/workspace/screenshot.png"
    find_text: Optional[str] = None   # scroll to first occurrence of this text before screenshotting


@app.post("/screenshot")
async def screenshot(req: ScreenshotReq):
    page = await _ensure_page()
    nav_error = None
    if req.url:
        try:
            await _safe_goto(page, req.url)
        except Exception as exc:
            nav_error = str(exc)
            # Page may have crashed — get a fresh one and retry
            try:
                page = await _ensure_page()
                await _safe_goto(page, req.url)
                nav_error = None
            except Exception as exc2:
                nav_error = str(exc2)
    try:
        # Brief human-like pause before screenshot (reduced for speed)
        await asyncio.sleep(_random.uniform(0.2, 0.5))

        clipped = False
        if req.find_text:
            # Zoom: find the element containing the text, scroll to it, then clip
            # the screenshot to show just that region with context padding.
            try:
                import json as _json
                found = await page.evaluate(
                    """(searchText) => {
                        const walker = document.createTreeWalker(
                            document.body, NodeFilter.SHOW_TEXT, null, false);
                        const lower = searchText.toLowerCase();
                        let node;
                        while ((node = walker.nextNode())) {
                            if (node.nodeValue && node.nodeValue.toLowerCase().includes(lower)) {
                                const el = node.parentElement;
                                el.scrollIntoView({behavior: \'instant\', block: \'center\'});
                                const r = el.getBoundingClientRect();
                                return {x: r.left, y: r.top, w: r.width, h: r.height,
                                        found: true};
                            }
                        }
                        return {found: false};
                    }""",
                    req.find_text,
                )
                await asyncio.sleep(_random.uniform(0.1, 0.2))
                if found.get("found"):
                    pad = 60
                    vp = page.viewport_size or {"width": 1280, "height": 900}
                    x = max(0, found["x"] - pad)
                    y = max(0, found["y"] - pad)
                    w = min(vp["width"] - x, max(found["w"] + 2 * pad, 400))
                    h = min(vp["height"] - y, max(found["h"] + 2 * pad, 200))
                    await page.screenshot(
                        path=req.path,
                        clip={"x": x, "y": y, "width": w, "height": h},
                    )
                    clipped = True
                else:
                    # Text not found — fall through to normal scroll+screenshot
                    scroll_px = _random.randint(80, 350)
                    await page.evaluate(f"window.scrollBy(0, {scroll_px})")
                    await asyncio.sleep(_random.uniform(0.1, 0.2))
                    await page.screenshot(path=req.path, full_page=False)
            except Exception:
                # Fallback to normal screenshot on any error
                await page.screenshot(path=req.path, full_page=False)
        else:
            scroll_px = _random.randint(80, 350)
            try:
                await page.evaluate(f"window.scrollBy(0, {scroll_px})")
                await asyncio.sleep(_random.uniform(0.1, 0.2))
            except Exception:
                pass
            await page.screenshot(path=req.path, full_page=False)

        result = {"path": req.path, "title": await page.title(), "url": page.url,
                  "clipped": clipped}
        if nav_error:
            result["nav_error"] = nav_error
        return result
    except Exception as exc:
        # Screenshot itself failed — extract page image URLs as fallback info
        img_urls: list = []
        try:
            if not page.is_closed():
                img_urls = await _get_image_urls(page)
        except Exception:
            pass
        return {
            "error": str(exc),
            "path": req.path,
            "title": "",
            "url": page.url if not page.is_closed() else "",
            "image_urls": img_urls,
        }


@app.get("/images")
async def list_images():
    """Extract direct image URLs from the currently loaded page."""
    try:
        page = await _ensure_page()
        urls = await _get_image_urls(page)
        return {"urls": urls}
    except Exception as exc:
        return {"urls": [], "error": str(exc)}


class SearchReq(BaseModel):
    query: str
    engine: str = "duckduckgo"


@app.post("/search")
async def search(req: SearchReq):
    """Human-like search: go to DDG homepage, fill search box, submit, wait for results URL."""
    try:
        page = await _ensure_page()
        await _safe_goto(page, "https://duckduckgo.com")
        # Fill the search box (instant, no typing delay — DDG homepage allows this)
        await page.fill("input[name=\'q\']", req.query)
        await asyncio.sleep(_random.uniform(0.2, 0.5))
        await page.keyboard.press("Enter")
        # Wait for the URL to change to a search results URL (not networkidle which stalls)
        try:
            await page.wait_for_url(
                lambda u: "q=" in u and "duckduckgo.com" in u and u != "https://duckduckgo.com/",
                timeout=8000,
            )
            await page.wait_for_load_state("domcontentloaded", timeout=5000)
        except Exception:
            await asyncio.sleep(2)  # fallback: just wait a moment
        content = await _extract_text(page)
        return {
            "query": req.query,
            "url": page.url,
            "content": content,
        }
    except Exception as exc:
        return {"error": str(exc), "query": req.query, "content": ""}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7081, log_level="warning")
'''


# ---------------------------------------------------------------------------
# BrowserTool
# ---------------------------------------------------------------------------


class BrowserTool:
    def __init__(self) -> None:
        self._server_url: str | None = None

    # ------------------------------------------------------------------
    # Server lifecycle helpers
    # ------------------------------------------------------------------

    def _container_ip(self) -> str:
        """Return the bridge IP of the human_browser container."""
        r = subprocess.run(
            ["docker", "inspect", BROWSER_CONTAINER],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0 or not r.stdout.strip():
            raise RuntimeError(
                f"Container '{BROWSER_CONTAINER}' not found. "
                "Start it with: docker start human_browser"
            )
        data = json.loads(r.stdout)[0]
        if not data.get("State", {}).get("Running"):
            raise RuntimeError(
                f"Container '{BROWSER_CONTAINER}' is not running. "
                "Start it with: docker start human_browser"
            )
        networks = data.get("NetworkSettings", {}).get("Networks", {})
        for net in networks.values():
            ip = net.get("IPAddress", "")
            if ip:
                return ip
        raise RuntimeError(
            f"Container '{BROWSER_CONTAINER}' has no routable IP address."
        )

    def _deploy_server(self) -> None:
        """Write server script into the container via docker cp."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(_SERVER_SRC)
            tmp = f.name
        try:
            subprocess.run(
                ["docker", "cp", tmp, f"{BROWSER_CONTAINER}:/workspace/browser_server.py"],
                check=True,
                capture_output=True,
            )
        finally:
            os.unlink(tmp)

    def _start_server_daemon(self) -> None:
        """Start uvicorn inside the container in the background."""
        subprocess.run(
            [
                "docker", "exec", "-d", BROWSER_CONTAINER,
                "/opt/ai-venv/bin/python3", "/workspace/browser_server.py",
            ],
            check=True,
            capture_output=True,
        )

    def _kill_old_server(self) -> None:
        """Kill any running browser server process inside the container."""
        subprocess.run(
            ["docker", "exec", BROWSER_CONTAINER, "pkill", "-f", "browser_server.py"],
            capture_output=True,
        )  # ignore return code — process may not be running

    async def _ensure_server(self) -> str:
        """Return the server base URL, starting (or upgrading) the server if needed."""
        # Fast path: cached URL
        if self._server_url:
            try:
                async with httpx.AsyncClient(timeout=2.0) as c:
                    r = await c.get(f"{self._server_url}/health")
                    if r.status_code == 200:
                        if r.json().get("version") == _REQUIRED_SERVER_VERSION:
                            return self._server_url
                        # Version mismatch — kill stale server and redeploy
                        self._kill_old_server()
                        self._server_url = None
                        await asyncio.sleep(1.5)
            except Exception:
                self._server_url = None

        ip = self._container_ip()
        url = f"http://{ip}:{BROWSER_PORT}"

        # Try an already-running server (previous session)
        try:
            async with httpx.AsyncClient(timeout=2.0) as c:
                r = await c.get(f"{url}/health")
                if r.status_code == 200:
                    if r.json().get("version") == _REQUIRED_SERVER_VERSION:
                        self._server_url = url
                        return url
                    # Stale version — kill and redeploy
                    self._kill_old_server()
                    await asyncio.sleep(1.5)
        except Exception:
            pass

        # Deploy + start
        self._deploy_server()
        self._start_server_daemon()

        # Wait for the server to become healthy
        deadline = asyncio.get_event_loop().time() + _STARTUP_TIMEOUT
        while asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(1.0)
            try:
                async with httpx.AsyncClient(timeout=2.0) as c:
                    r = await c.get(f"{url}/health")
                    if r.status_code == 200:
                        self._server_url = url
                        return url
            except Exception:
                pass

        raise RuntimeError(
            "Browser server did not respond within "
            f"{_STARTUP_TIMEOUT}s — check `docker logs {BROWSER_CONTAINER}`"
        )

    # ------------------------------------------------------------------
    # Fallback: plain httpx fetch when browser is unavailable
    # ------------------------------------------------------------------

    async def _httpx_fetch(self, url: str) -> dict:
        """Direct HTTP fetch without Playwright — used as a last resort."""
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,*/*;q=0.9",
        }
        try:
            async with httpx.AsyncClient(timeout=20.0, follow_redirects=True, headers=headers) as c:
                r = await c.get(url)
            raw = r.text
            text = re.sub(r"<[^>]+>", " ", raw)
            text = re.sub(r"\s{3,}", "\n\n", text).strip()[:8000]
            return {"url": str(r.url), "title": url, "content": text, "fallback": True}
        except Exception as exc:
            return {"url": url, "title": "", "content": "", "error": str(exc), "fallback": True}

    # ------------------------------------------------------------------
    # Browser actions
    # ------------------------------------------------------------------

    async def navigate(self, url: str) -> dict:
        """Navigate to url; retries after re-deploying the server on failure, then falls back to httpx."""
        for attempt in range(2):
            try:
                base = await self._ensure_server()
                async with httpx.AsyncClient(timeout=45.0) as c:
                    r = await c.post(f"{base}/navigate", json={"url": url})
                if r.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"Browser server returned {r.status_code}",
                        request=r.request,
                        response=r,
                    )
                return r.json()
            except httpx.HTTPStatusError:
                if attempt == 0:
                    self._server_url = None
                    try:
                        self._deploy_server()
                        self._start_server_daemon()
                        await asyncio.sleep(3.0)
                    except Exception:
                        pass
                    continue
                break
            except Exception:
                if attempt == 0:
                    self._server_url = None
                    continue
                break
        return await self._httpx_fetch(url)

    async def screenshot(self, url: str | None = None, find_text: str | None = None) -> dict:
        base = await self._ensure_server()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"/workspace/screenshot_{ts}.png"
        payload: dict = {"path": path}
        if url:
            payload["url"] = url
        if find_text:
            payload["find_text"] = find_text
        async with httpx.AsyncClient(timeout=45.0) as c:
            r = await c.post(f"{base}/screenshot", json=payload)
        r.raise_for_status()
        data = r.json()
        # Expose the host-side path (workspace is bind-mounted)
        data["host_path"] = f"/docker/human_browser/workspace/screenshot_{ts}.png"
        return data

    async def list_images(self) -> dict:
        """Get image URLs from the current page in the browser."""
        base = await self._ensure_server()
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(f"{base}/images")
        r.raise_for_status()
        return r.json()

    async def click(self, selector: str) -> dict:
        base = await self._ensure_server()
        async with httpx.AsyncClient(timeout=20.0) as c:
            r = await c.post(f"{base}/click", json={"selector": selector})
        r.raise_for_status()
        return r.json()

    async def fill(self, selector: str, value: str) -> dict:
        base = await self._ensure_server()
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.post(f"{base}/fill", json={"selector": selector, "value": value})
        r.raise_for_status()
        return r.json()

    async def read(self) -> dict:
        base = await self._ensure_server()
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(f"{base}/read")
        r.raise_for_status()
        return r.json()

    async def eval_js(self, code: str) -> dict:
        base = await self._ensure_server()
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.post(f"{base}/eval", json={"code": code})
        r.raise_for_status()
        return r.json()

    async def search(self, query: str) -> dict:
        """Human-like search: go to DuckDuckGo, type query, press Enter, return results."""
        base = await self._ensure_server()
        async with httpx.AsyncClient(timeout=45.0) as c:
            r = await c.post(f"{base}/search", json={"query": query})
        r.raise_for_status()
        return r.json()
