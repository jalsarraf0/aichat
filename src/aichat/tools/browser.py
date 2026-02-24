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
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
from datetime import datetime

import httpx

BROWSER_CONTAINER = "human_browser"
BROWSER_PORT = 7081
_STARTUP_TIMEOUT = 35  # seconds to wait for uvicorn to come up

# ---------------------------------------------------------------------------
# FastAPI Playwright server — injected into the container at first use
# ---------------------------------------------------------------------------
# NOTE: this string is written verbatim into the container; it must not
# contain anything that would break when piped through docker cp.

_SERVER_SRC = '''\
"""Browser automation server — auto-deployed by aichat BrowserTool."""
from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from playwright.async_api import async_playwright

_pw = _browser = _page = None


@asynccontextmanager
async def lifespan(app):
    global _pw, _browser, _page
    _pw = await async_playwright().start()
    _browser = await _pw.chromium.launch(
        headless=True,
        args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
    )
    _page = await _browser.new_page()
    yield
    await _browser.close()
    await _pw.stop()


app = FastAPI(lifespan=lifespan)


async def _extract_text(page) -> str:
    js = """() => {
        const b = document.body;
        if (!b) return '';
        b.querySelectorAll(
            'script,style,nav,footer,header,aside,noscript'
        ).forEach(e => e.remove());
        return b.innerText.trim().slice(0, 8000);
    }"""
    try:
        return await page.evaluate(js)
    except Exception:
        return ""


@app.get("/health")
async def health():
    return {"ok": True}


class NavReq(BaseModel):
    url: str
    wait: str = "networkidle"


@app.post("/navigate")
async def navigate(req: NavReq):
    try:
        await _page.goto(req.url, wait_until=req.wait, timeout=30000)
    except Exception:
        await _page.goto(req.url, wait_until="load", timeout=30000)
    return {
        "title": await _page.title(),
        "url": _page.url,
        "content": await _extract_text(_page),
    }


class ClickReq(BaseModel):
    selector: str


@app.post("/click")
async def click(req: ClickReq):
    await _page.click(req.selector, timeout=10000)
    try:
        await _page.wait_for_load_state("networkidle", timeout=10000)
    except Exception:
        pass
    return {"url": _page.url, "content": await _extract_text(_page)}


class FillReq(BaseModel):
    selector: str
    value: str


@app.post("/fill")
async def fill(req: FillReq):
    await _page.fill(req.selector, req.value)
    return {"ok": True}


@app.get("/read")
async def read():
    return {
        "title": await _page.title(),
        "url": _page.url,
        "content": await _extract_text(_page),
    }


class EvalReq(BaseModel):
    code: str


@app.post("/eval")
async def evaluate(req: EvalReq):
    result = await _page.evaluate(req.code)
    return {"result": str(result) if result is not None else "null"}


class ScreenshotReq(BaseModel):
    url: Optional[str] = None
    path: str = "/workspace/screenshot.png"


@app.post("/screenshot")
async def screenshot(req: ScreenshotReq):
    if req.url:
        try:
            await _page.goto(req.url, wait_until="networkidle", timeout=30000)
        except Exception:
            await _page.goto(req.url, wait_until="load", timeout=30000)
    await _page.screenshot(path=req.path, full_page=False)
    return {"path": req.path, "title": await _page.title(), "url": _page.url}


class SearchReq(BaseModel):
    query: str
    engine: str = "duckduckgo"


@app.post("/search")
async def search(req: SearchReq):
    """Human-like search: navigate to DuckDuckGo, type query, submit, return results."""
    try:
        try:
            await _page.goto("https://duckduckgo.com", wait_until="networkidle", timeout=20000)
        except Exception:
            await _page.goto("https://duckduckgo.com", wait_until="load", timeout=20000)
        await _page.fill("input[name='q']", req.query)
        await _page.keyboard.press("Enter")
        try:
            await _page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            await _page.wait_for_load_state("load", timeout=10000)
        content = await _extract_text(_page)
        return {
            "query": req.query,
            "url": _page.url,
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

    async def _ensure_server(self) -> str:
        """Return the server base URL, starting the server if needed."""
        # Fast path: cached URL
        if self._server_url:
            try:
                async with httpx.AsyncClient(timeout=2.0) as c:
                    r = await c.get(f"{self._server_url}/health")
                    if r.status_code == 200:
                        return self._server_url
            except Exception:
                self._server_url = None

        ip = self._container_ip()
        url = f"http://{ip}:{BROWSER_PORT}"

        # Try an already-running server (previous session)
        try:
            async with httpx.AsyncClient(timeout=2.0) as c:
                r = await c.get(f"{url}/health")
                if r.status_code == 200:
                    self._server_url = url
                    return url
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
    # Browser actions
    # ------------------------------------------------------------------

    async def navigate(self, url: str) -> dict:
        base = await self._ensure_server()
        async with httpx.AsyncClient(timeout=45.0) as c:
            r = await c.post(f"{base}/navigate", json={"url": url})
        r.raise_for_status()
        return r.json()

    async def screenshot(self, url: str | None = None) -> dict:
        base = await self._ensure_server()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"/workspace/screenshot_{ts}.png"
        payload: dict = {"path": path}
        if url:
            payload["url"] = url
        async with httpx.AsyncClient(timeout=45.0) as c:
            r = await c.post(f"{base}/screenshot", json=payload)
        r.raise_for_status()
        data = r.json()
        # Expose the host-side path (workspace is bind-mounted)
        data["host_path"] = f"/docker/human_browser/workspace/screenshot_{ts}.png"
        return data

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
