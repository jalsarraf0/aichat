"""aichat-browser: Headless Chromium browser automation for agentic MCP use.

Provides human-like browsing with full keyboard, mouse, and screenshot support
via Playwright. Designed for agentic workflows — navigate, interact, extract.

Routes:
  /health          — service health + browser status
  /navigate        — go to URL, return page info + screenshot
  /click           — click at coordinates or CSS selector
  /type            — type text into focused element or selector
  /screenshot      — capture current page screenshot
  /scroll          — scroll page by amount or to element
  /evaluate        — execute JavaScript on page
  /extract         — extract text/links/images from page
  /fill_form       — fill form fields by selector
  /keyboard        — press keyboard keys (Enter, Tab, Escape, etc.)
  /mouse           — mouse actions (move, click, drag)
  /wait            — wait for selector/navigation/timeout
  /tabs            — manage browser tabs (list, switch, close, new)
  /cookies         — get/set/clear cookies
  /pdf             — save page as PDF
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from playwright.async_api import async_playwright, Browser, BrowserContext, Page

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("aichat-browser")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_VIEWPORT_W = int(os.environ.get("VIEWPORT_WIDTH", "1920"))
_VIEWPORT_H = int(os.environ.get("VIEWPORT_HEIGHT", "1080"))
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

app = FastAPI(title="aichat-browser")

# ---------------------------------------------------------------------------
# Browser lifecycle
# ---------------------------------------------------------------------------
_pw = None
_browser: Browser | None = None
_context: BrowserContext | None = None
_page: Page | None = None


async def _ensure_browser() -> Page:
    """Lazy-initialize Playwright browser and return the active page."""
    global _pw, _browser, _context, _page
    if _page is not None and not _page.is_closed():
        return _page
    if _pw is None:
        _pw = await async_playwright().start()
    if _browser is None or not _browser.is_connected():
        _browser = await _pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-software-rasterizer",
            ],
        )
    if _context is None:
        _context = await _browser.new_context(
            viewport={"width": _VIEWPORT_W, "height": _VIEWPORT_H},
            user_agent=_USER_AGENT,
            locale="en-US",
            timezone_id="America/New_York",
            ignore_https_errors=True,
        )
    if _page is None or _page.is_closed():
        _page = await _context.new_page()
    return _page


async def _take_screenshot(page: Page, full_page: bool = False) -> str:
    """Take a screenshot and return as base64 PNG."""
    raw = await page.screenshot(full_page=full_page, type="png")
    return base64.standard_b64encode(raw).decode("ascii")


@app.on_event("shutdown")
async def _shutdown():
    global _browser, _pw
    if _browser:
        await _browser.close()
    if _pw:
        await _pw.stop()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> dict:
    browser_ok = _browser is not None and _browser.is_connected()
    page_url = ""
    if _page and not _page.is_closed():
        page_url = _page.url
    return {
        "status": "ok",
        "browser_connected": browser_ok,
        "current_url": page_url,
        "viewport": f"{_VIEWPORT_W}x{_VIEWPORT_H}",
    }


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
@app.post("/navigate")
async def navigate(payload: dict[str, Any]) -> dict:
    """Navigate to a URL. Returns page title, URL, and screenshot."""
    url = str(payload.get("url", "")).strip()
    if not url:
        raise HTTPException(422, "'url' is required")
    wait_until = str(payload.get("wait_until", "domcontentloaded")).strip()
    timeout_ms = int(payload.get("timeout_ms", 30000))

    page = await _ensure_browser()
    try:
        await page.goto(url, wait_until=wait_until, timeout=timeout_ms)
    except Exception as exc:
        raise HTTPException(422, f"Navigation failed: {exc}") from exc

    screenshot = await _take_screenshot(page)
    return {
        "url": page.url,
        "title": await page.title(),
        "screenshot_b64": screenshot,
    }


# ---------------------------------------------------------------------------
# Click
# ---------------------------------------------------------------------------
@app.post("/click")
async def click(payload: dict[str, Any]) -> dict:
    """Click at coordinates (x, y) or on a CSS selector."""
    selector = str(payload.get("selector", "")).strip()
    x = payload.get("x")
    y = payload.get("y")
    button = str(payload.get("button", "left")).strip()
    click_count = int(payload.get("click_count", 1))

    page = await _ensure_browser()
    try:
        if selector:
            await page.click(selector, button=button, click_count=click_count, timeout=10000)
        elif x is not None and y is not None:
            await page.mouse.click(float(x), float(y), button=button, click_count=click_count)
        else:
            raise HTTPException(422, "Provide 'selector' or both 'x' and 'y'")
    except Exception as exc:
        raise HTTPException(422, f"Click failed: {exc}") from exc

    await page.wait_for_timeout(500)  # brief settle
    screenshot = await _take_screenshot(page)
    return {"action": "click", "screenshot_b64": screenshot, "url": page.url}


# ---------------------------------------------------------------------------
# Type
# ---------------------------------------------------------------------------
@app.post("/type")
async def type_text(payload: dict[str, Any]) -> dict:
    """Type text into the focused element or a CSS selector."""
    text = str(payload.get("text", ""))
    selector = str(payload.get("selector", "")).strip()
    clear_first = bool(payload.get("clear_first", False))
    delay_ms = int(payload.get("delay_ms", 50))

    page = await _ensure_browser()
    try:
        if selector:
            if clear_first:
                await page.fill(selector, "")
            await page.type(selector, text, delay=delay_ms)
        else:
            await page.keyboard.type(text, delay=delay_ms)
    except Exception as exc:
        raise HTTPException(422, f"Type failed: {exc}") from exc

    screenshot = await _take_screenshot(page)
    return {"action": "type", "text_length": len(text), "screenshot_b64": screenshot}


# ---------------------------------------------------------------------------
# Screenshot
# ---------------------------------------------------------------------------
@app.post("/screenshot")
async def screenshot(payload: dict[str, Any]) -> dict:
    """Capture screenshot of the current page."""
    full_page = bool(payload.get("full_page", False))
    selector = str(payload.get("selector", "")).strip()

    page = await _ensure_browser()
    if selector:
        try:
            element = page.locator(selector)
            raw = await element.screenshot(type="png")
            b64 = base64.standard_b64encode(raw).decode("ascii")
        except Exception as exc:
            raise HTTPException(422, f"Element screenshot failed: {exc}") from exc
    else:
        b64 = await _take_screenshot(page, full_page=full_page)

    return {
        "screenshot_b64": b64,
        "url": page.url,
        "title": await page.title(),
    }


# ---------------------------------------------------------------------------
# Scroll
# ---------------------------------------------------------------------------
@app.post("/scroll")
async def scroll(payload: dict[str, Any]) -> dict:
    """Scroll the page by pixels or to a selector."""
    selector = str(payload.get("selector", "")).strip()
    direction = str(payload.get("direction", "down")).strip()
    amount = int(payload.get("amount", 500))

    page = await _ensure_browser()
    try:
        if selector:
            await page.locator(selector).scroll_into_view_if_needed(timeout=5000)
        else:
            delta = amount if direction == "down" else -amount
            await page.mouse.wheel(0, delta)
    except Exception as exc:
        raise HTTPException(422, f"Scroll failed: {exc}") from exc

    await page.wait_for_timeout(300)
    screenshot = await _take_screenshot(page)
    return {"action": "scroll", "screenshot_b64": screenshot, "url": page.url}


# ---------------------------------------------------------------------------
# Evaluate JavaScript
# ---------------------------------------------------------------------------
@app.post("/evaluate")
async def evaluate(payload: dict[str, Any]) -> dict:
    """Execute JavaScript on the page and return the result."""
    expression = str(payload.get("expression", "")).strip()
    if not expression:
        raise HTTPException(422, "'expression' is required")

    page = await _ensure_browser()
    try:
        result = await page.evaluate(expression)
    except Exception as exc:
        raise HTTPException(422, f"JS evaluation failed: {exc}") from exc

    return {"result": result, "url": page.url}


# ---------------------------------------------------------------------------
# Extract page content
# ---------------------------------------------------------------------------
@app.post("/extract")
async def extract(payload: dict[str, Any]) -> dict:
    """Extract text, links, or images from the current page."""
    what = str(payload.get("what", "text")).strip()

    page = await _ensure_browser()

    if what == "text":
        text = await page.inner_text("body")
        return {"text": text[:50000], "url": page.url, "title": await page.title()}
    elif what == "links":
        links = await page.evaluate("""
            () => Array.from(document.querySelectorAll('a[href]')).map(a => ({
                text: a.innerText.trim().slice(0, 100),
                href: a.href
            })).filter(l => l.href.startsWith('http')).slice(0, 100)
        """)
        return {"links": links, "count": len(links), "url": page.url}
    elif what == "images":
        images = await page.evaluate("""
            () => Array.from(document.querySelectorAll('img[src]')).map(img => ({
                src: img.src,
                alt: (img.alt || '').slice(0, 100),
                width: img.naturalWidth,
                height: img.naturalHeight
            })).filter(i => i.src.startsWith('http')).slice(0, 50)
        """)
        return {"images": images, "count": len(images), "url": page.url}
    else:
        raise HTTPException(422, f"Unknown extract type: {what}. Use: text, links, images")


# ---------------------------------------------------------------------------
# Keyboard
# ---------------------------------------------------------------------------
@app.post("/keyboard")
async def keyboard(payload: dict[str, Any]) -> dict:
    """Press keyboard keys. Supports: Enter, Tab, Escape, Backspace, ArrowDown, etc."""
    key = str(payload.get("key", "")).strip()
    if not key:
        raise HTTPException(422, "'key' is required (e.g. Enter, Tab, Escape, ArrowDown)")

    page = await _ensure_browser()
    try:
        await page.keyboard.press(key)
    except Exception as exc:
        raise HTTPException(422, f"Keyboard press failed: {exc}") from exc

    await page.wait_for_timeout(300)
    screenshot = await _take_screenshot(page)
    return {"action": "keyboard", "key": key, "screenshot_b64": screenshot}


# ---------------------------------------------------------------------------
# Mouse
# ---------------------------------------------------------------------------
@app.post("/mouse")
async def mouse(payload: dict[str, Any]) -> dict:
    """Mouse actions: move, click, dblclick, drag."""
    action = str(payload.get("action", "click")).strip()
    x = float(payload.get("x", 0))
    y = float(payload.get("y", 0))

    page = await _ensure_browser()
    try:
        if action == "move":
            await page.mouse.move(x, y)
        elif action == "click":
            await page.mouse.click(x, y)
        elif action == "dblclick":
            await page.mouse.dblclick(x, y)
        elif action == "drag":
            to_x = float(payload.get("to_x", x))
            to_y = float(payload.get("to_y", y))
            await page.mouse.move(x, y)
            await page.mouse.down()
            await page.mouse.move(to_x, to_y)
            await page.mouse.up()
        else:
            raise HTTPException(422, f"Unknown mouse action: {action}")
    except Exception as exc:
        raise HTTPException(422, f"Mouse action failed: {exc}") from exc

    await page.wait_for_timeout(300)
    screenshot = await _take_screenshot(page)
    return {"action": f"mouse.{action}", "x": x, "y": y, "screenshot_b64": screenshot}


# ---------------------------------------------------------------------------
# Wait
# ---------------------------------------------------------------------------
@app.post("/wait")
async def wait(payload: dict[str, Any]) -> dict:
    """Wait for a selector, navigation, or timeout."""
    selector = str(payload.get("selector", "")).strip()
    timeout_ms = int(payload.get("timeout_ms", 10000))
    state = str(payload.get("state", "visible")).strip()

    page = await _ensure_browser()
    try:
        if selector:
            await page.wait_for_selector(selector, state=state, timeout=timeout_ms)
        else:
            await page.wait_for_timeout(min(timeout_ms, 30000))
    except Exception as exc:
        raise HTTPException(422, f"Wait failed: {exc}") from exc

    screenshot = await _take_screenshot(page)
    return {"action": "wait", "screenshot_b64": screenshot, "url": page.url}


# ---------------------------------------------------------------------------
# Form filling
# ---------------------------------------------------------------------------
@app.post("/fill_form")
async def fill_form(payload: dict[str, Any]) -> dict:
    """Fill form fields. fields: [{selector, value}]"""
    fields = payload.get("fields", [])
    if not fields:
        raise HTTPException(422, "'fields' array is required: [{selector, value}]")

    page = await _ensure_browser()
    filled = 0
    for field in fields:
        sel = str(field.get("selector", "")).strip()
        val = str(field.get("value", ""))
        if not sel:
            continue
        try:
            await page.fill(sel, val)
            filled += 1
        except Exception:
            # Try type fallback for non-input elements
            try:
                await page.click(sel)
                await page.keyboard.type(val, delay=30)
                filled += 1
            except Exception:
                pass

    screenshot = await _take_screenshot(page)
    return {"filled": filled, "total": len(fields), "screenshot_b64": screenshot}


# ---------------------------------------------------------------------------
# Tab management
# ---------------------------------------------------------------------------
@app.post("/tabs")
async def tabs(payload: dict[str, Any]) -> dict:
    """Manage tabs: list, new, switch, close."""
    global _page
    action = str(payload.get("action", "list")).strip()

    page = await _ensure_browser()

    if action == "list":
        pages = _context.pages if _context else []
        tab_list = [{"index": i, "url": p.url, "title": await p.title()}
                    for i, p in enumerate(pages)]
        return {"tabs": tab_list, "active": next(
            (i for i, p in enumerate(pages) if p == _page), 0
        )}
    elif action == "new":
        url = str(payload.get("url", "about:blank")).strip()
        _page = await _context.new_page()
        if url != "about:blank":
            await _page.goto(url, wait_until="domcontentloaded", timeout=30000)
        screenshot = await _take_screenshot(_page)
        return {"action": "new_tab", "url": _page.url, "screenshot_b64": screenshot}
    elif action == "switch":
        index = int(payload.get("index", 0))
        pages = _context.pages if _context else []
        if 0 <= index < len(pages):
            _page = pages[index]
            await _page.bring_to_front()
            screenshot = await _take_screenshot(_page)
            return {"action": "switch", "index": index, "url": _page.url,
                    "screenshot_b64": screenshot}
        raise HTTPException(422, f"Tab index {index} out of range (0-{len(pages)-1})")
    elif action == "close":
        pages = _context.pages if _context else []
        if len(pages) > 1:
            await _page.close()
            _page = _context.pages[-1] if _context and _context.pages else None
            if _page:
                screenshot = await _take_screenshot(_page)
                return {"action": "close", "url": _page.url, "screenshot_b64": screenshot}
        return {"action": "close", "error": "Cannot close last tab"}
    else:
        raise HTTPException(422, f"Unknown tab action: {action}")


# ---------------------------------------------------------------------------
# Cookies
# ---------------------------------------------------------------------------
@app.post("/cookies")
async def cookies(payload: dict[str, Any]) -> dict:
    """Get, set, or clear cookies."""
    action = str(payload.get("action", "get")).strip()

    if action == "get":
        if _context:
            all_cookies = await _context.cookies()
            return {"cookies": all_cookies[:50], "count": len(all_cookies)}
        return {"cookies": [], "count": 0}
    elif action == "set":
        cookie = payload.get("cookie", {})
        if not cookie.get("name") or not cookie.get("value"):
            raise HTTPException(422, "'cookie' must have 'name' and 'value'")
        if _context:
            await _context.add_cookies([cookie])
        return {"action": "set", "name": cookie["name"]}
    elif action == "clear":
        if _context:
            await _context.clear_cookies()
        return {"action": "clear"}
    else:
        raise HTTPException(422, f"Unknown cookie action: {action}")


# ---------------------------------------------------------------------------
# PDF export
# ---------------------------------------------------------------------------
@app.post("/pdf")
async def pdf_export(payload: dict[str, Any]) -> dict:
    """Save the current page as PDF (base64-encoded)."""
    page = await _ensure_browser()
    try:
        raw = await page.pdf(
            format=str(payload.get("format", "A4")),
            print_background=True,
        )
        b64 = base64.standard_b64encode(raw).decode("ascii")
        return {"pdf_b64": b64, "size_bytes": len(raw), "url": page.url}
    except Exception as exc:
        raise HTTPException(422, f"PDF export failed: {exc}") from exc
