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
_REQUIRED_SERVER_VERSION = "19"


class BrowserGpuConfig:
    """Host-side mirror of the BrowserGpuConfig embedded in _SERVER_SRC.

    Identical logic — probes /dev/dri and INTEL_GPU env var to decide
    whether Chromium gets hardware-accelerated launch args (--use-gl=egl,
    VA-API) or the safe --disable-gpu fallback.

    This host-side copy exists so tests can import and exercise the class
    without exec-ing the full _SERVER_SRC string (which requires playwright).
    """

    _BASE_ARGS: list[str] = [
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-blink-features=AutomationControlled",
        "--disable-infobars",
        "--window-size=1920,1080",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-default-apps",
        "--use-mock-keychain",
        "--disable-features=TranslateUI",
        "--lang=en-US",
    ]

    _GPU_ARGS: list[str] = [
        "--use-gl=egl",
        "--enable-features=VaapiVideoDecoder,VaapiVideoEncoder,CanvasOopRasterization",
        "--enable-gpu-rasterization",
        "--enable-zero-copy",
        "--ignore-gpu-blocklist",
        "--disable-software-rasterizer",
    ]

    @classmethod
    def _has_dri(cls) -> bool:
        """True if /dev/dri/renderD* is accessible."""
        try:
            return any(f.startswith("renderD") for f in os.listdir("/dev/dri"))
        except (FileNotFoundError, PermissionError, OSError):
            return False

    @classmethod
    def _intel_gpu_env(cls) -> bool:
        return os.environ.get("INTEL_GPU", "") == "1"

    @classmethod
    def gpu_available(cls) -> bool:
        return cls._has_dri() or cls._intel_gpu_env()

    @classmethod
    def launch_args(cls) -> list[str]:
        """Return the full Chromium arg list for the current environment."""
        args = list(cls._BASE_ARGS)
        if cls.gpu_available():
            args.extend(cls._GPU_ARGS)
        else:
            args.append("--disable-gpu")
        return args

    @classmethod
    def info(cls) -> dict:
        return {
            "gpu_available": cls.gpu_available(),
            "dri_accessible": cls._has_dri(),
            "intel_gpu_env": cls._intel_gpu_env(),
        }

# ---------------------------------------------------------------------------
# FastAPI Playwright server — injected into the container at first use
# ---------------------------------------------------------------------------
# NOTE: this string is written verbatim into the container; it must not
# contain anything that would break when piped through docker cp.

_SERVER_SRC = '''\
"""Browser automation server — auto-deployed by aichat BrowserTool. v19"""
from __future__ import annotations

import asyncio
import os as _os
import random as _random
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from playwright.async_api import async_playwright

_VERSION = "19"


class BrowserGpuConfig:
    """Detects Intel/NVIDIA GPU inside the container and returns Chromium launch
    args for hardware-accelerated rasterization + VA-API video decode.
    Gracefully falls back to --disable-gpu when /dev/dri is not mapped into
    this container (the safe default for the external human_browser container)."""

    _BASE_ARGS: list = [
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-blink-features=AutomationControlled",
        "--disable-infobars",
        "--window-size=1920,1080",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-default-apps",
        "--use-mock-keychain",
        "--disable-features=TranslateUI",
        "--lang=en-US",
    ]

    _GPU_ARGS: list = [
        "--use-gl=egl",
        "--enable-features=VaapiVideoDecoder,VaapiVideoEncoder,CanvasOopRasterization",
        "--enable-gpu-rasterization",
        "--enable-zero-copy",
        "--ignore-gpu-blocklist",
        "--disable-software-rasterizer",
    ]

    @classmethod
    def _has_dri(cls) -> bool:
        """True if /dev/dri/renderD* is accessible inside this container."""
        try:
            return any(f.startswith("renderD") for f in _os.listdir("/dev/dri"))
        except (FileNotFoundError, PermissionError, OSError):
            return False

    @classmethod
    def _intel_gpu_env(cls) -> bool:
        return _os.environ.get("INTEL_GPU", "") == "1"

    @classmethod
    def gpu_available(cls) -> bool:
        return cls._has_dri() or cls._intel_gpu_env()

    @classmethod
    def launch_args(cls) -> list:
        """Return the full Chromium arg list for the current environment."""
        args = list(cls._BASE_ARGS)
        if cls.gpu_available():
            args.extend(cls._GPU_ARGS)
        else:
            args.append("--disable-gpu")
        return args

    @classmethod
    def info(cls) -> dict:
        return {
            "gpu_available": cls.gpu_available(),
            "dri_accessible": cls._has_dri(),
            "intel_gpu_env": cls._intel_gpu_env(),
        }

# UA matches the actual Playwright Chromium version (145) to avoid the
# trivially detectable Chrome/124 vs Sec-Ch-Ua:v="145" mismatch.
# Windows platform is used because it represents ~70% of real desktop Chrome
# users and avoids the Linux-headless server fingerprint.
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
)

# Sec-Ch-Ua headers to inject — must use "Google Chrome" brand (not
# "HeadlessChrome" which Playwright emits and bots always check for).
_SEC_CH_UA_HEADERS = {
    "Sec-Ch-Ua": \'"Not:A-Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"\',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": \'"Windows"\',
    "Accept-Language": "en-US,en;q=0.9",
}

# Extended stealth v12: eliminate every signal Cloudflare Turnstile checks.
# Fixes: webdriver enumerable, permissions toString, Accept-Language q-values,
# pdfViewerEnabled, battery spoof, mimeTypes, connection RTT, timezone.
_STEALTH_JS = """
// 0. Native-code spoofer: wrapper functions appear as [native code] to toString()
const _origFnToString = Function.prototype.toString;
const _nativizedFns = new WeakMap();
Object.defineProperty(Function.prototype, \'toString\', {
    value: function toString() {
        if (_nativizedFns.has(this)) {
            return \'function \' + _nativizedFns.get(this) + \'() { [native code] }\';
        }
        return _origFnToString.call(this);
    },
    writable: true,
    configurable: true,
});
function _markNative(fn, name) {
    _nativizedFns.set(fn, name !== undefined ? name : (fn.name || \'\'));
    return fn;
}

// 1. Remove webdriver flag on Navigator.prototype with enumerable:false
// (instance-level override still shows enumerable:true on the prototype)
Object.defineProperty(Navigator.prototype, \'webdriver\', {
    get: () => undefined,
    enumerable: false,
    configurable: true,
});

// 2. Realistic plugins (Chrome PDF Plugin + Viewer + Native Client)
const _pluginData = [
    {name: \'Chrome PDF Plugin\', filename: \'internal-pdf-viewer\', description: \'Portable Document Format\'},
    {name: \'Chrome PDF Viewer\',  filename: \'mhjfbmdgcfjbbpaeojofohoefgiehjai\', description: \'\'},
    {name: \'Native Client\',      filename: \'internal-nacl-plugin\',            description: \'\'},
];
const _plugins = Object.create(PluginArray.prototype);
_pluginData.forEach((d, i) => {
    const p = Object.create(Plugin.prototype);
    Object.defineProperty(p, \'name\',        {value: d.name});
    Object.defineProperty(p, \'filename\',    {value: d.filename});
    Object.defineProperty(p, \'description\', {value: d.description});
    Object.defineProperty(p, \'length\',      {value: 0});
    _plugins[i] = p;
    _plugins[d.name] = p;
});
Object.defineProperty(_plugins, \'length\', {value: _pluginData.length});
Object.defineProperty(navigator, \'plugins\', {get: () => _plugins});

// 3. MimeTypes matching the PDF plugins (absent in headless)
try {
    const _mimeData = [
        {type: \'application/pdf\', description: \'Portable Document Format\', suffix: \'pdf\'},
        {type: \'text/pdf\',        description: \'Portable Document Format\', suffix: \'pdf\'},
    ];
    const _mimes = Object.create(MimeTypeArray.prototype);
    _mimeData.forEach((d, i) => {
        const m = Object.create(MimeType.prototype);
        Object.defineProperty(m, \'type\',        {value: d.type});
        Object.defineProperty(m, \'description\', {value: d.description});
        Object.defineProperty(m, \'suffixes\',    {value: d.suffix});
        _mimes[i] = m;
        _mimes[d.type] = m;
    });
    Object.defineProperty(_mimes, \'length\', {value: _mimeData.length});
    Object.defineProperty(navigator, \'mimeTypes\', {get: () => _mimes});
} catch(_) {}

// 4. Languages, platform, concurrency, pdfViewerEnabled (true since Chrome 108)
Object.defineProperty(navigator, \'languages\',          {get: () => [\'en-US\', \'en\']});
Object.defineProperty(navigator, \'platform\',           {get: () => \'Win32\'});
Object.defineProperty(navigator, \'hardwareConcurrency\', {get: () => 8});
Object.defineProperty(navigator, \'pdfViewerEnabled\',   {get: () => true});

// 5. Chrome runtime stub (required by many sites)
window.chrome = {
    runtime: {
        id: undefined,
        connect: () => {},
        sendMessage: () => {},
        onMessage: {addListener: () => {}, removeListener: () => {}},
    },
    loadTimes: () => ({}),
    csi: () => ({}),
    app: {},
};

// 6. Permissions — wrap query() and mark as [native code] so toString() check passes
try {
    const origQuery = window.navigator.permissions.query;
    const wrappedQuery = _markNative(function query(perm) {
        if (perm && perm.name === \'notifications\') {
            return Promise.resolve({state: Notification.permission});
        }
        return origQuery.call(window.navigator.permissions, perm);
    }, \'query\');
    Object.defineProperty(window.navigator.permissions, \'query\', {
        value: wrappedQuery,
        writable: true,
        configurable: true,
    });
} catch(_) {}

// 7. Battery API — realistic laptop (not always-full always-charging headless)
try {
    if (\'getBattery\' in navigator) {
        const _fakeBattery = {
            charging: false,
            chargingTime: Infinity,
            dischargingTime: 14400,
            level: 0.72,
            addEventListener:    () => {},
            removeEventListener: () => {},
            dispatchEvent:       () => true,
        };
        navigator.getBattery = _markNative(async () => _fakeBattery, \'getBattery\');
    }
} catch(_) {}

// 8. Network connection — non-zero RTT (datacenter default of 0 is suspicious)
try {
    if (navigator.connection) {
        Object.defineProperty(navigator.connection, \'rtt\',
            {get: () => 100, configurable: true});
        Object.defineProperty(navigator.connection, \'effectiveType\',
            {get: () => \'4g\', configurable: true});
    }
} catch(_) {}

// 9. Canvas — per-session pixel noise defeats exact-hash fingerprinting.
// Seed is fixed per page-load so repeated toDataURL() calls return same value.
(function() {
    const _seed = (Math.floor(Math.random() * 254) + 1) & 0xff;
    const _oTDU  = HTMLCanvasElement.prototype.toDataURL;
    const _oBlob = HTMLCanvasElement.prototype.toBlob;
    const _oGID  = CanvasRenderingContext2D.prototype.getImageData;
    function _addNoise(canvas) {
        const ctx2d = canvas.getContext(\'2d\');
        if (!ctx2d || canvas.width < 2 || canvas.height < 2) return;
        const d = _oGID.call(ctx2d, 0, 0, 1, 1);
        d.data[0] = (d.data[0] ^ _seed) & 0xff;
        ctx2d.putImageData(d, 0, 0);
    }
    HTMLCanvasElement.prototype.toDataURL = _markNative(function(t, q) {
        _addNoise(this); return _oTDU.call(this, t, q);
    }, \'toDataURL\');
    HTMLCanvasElement.prototype.toBlob = _markNative(function(cb, t, q) {
        _addNoise(this); return _oBlob.call(this, cb, t, q);
    }, \'toBlob\');
    CanvasRenderingContext2D.prototype.getImageData = _markNative(function(x, y, w, h) {
        const d = _oGID.call(this, x, y, w, h);
        if (d.data.length > 0) d.data[0] = (d.data[0] ^ _seed) & 0xff;
        return d;
    }, \'getImageData\');
})();

// 10. WebGL — hide SwiftShader/Mesa (headless) with realistic NVIDIA GPU strings.
// SwiftShader RENDERER is the single most reliable headless signal bot detectors use.
(function() {
    function _patchGL(GL) {
        const _orig = GL.prototype.getParameter;
        GL.prototype.getParameter = _markNative(function(p) {
            if (p === 37445) return \'Google Inc. (NVIDIA)\';
            if (p === 37446) return \'ANGLE (NVIDIA, NVIDIA GeForce RTX 3070 Direct3D11 vs_5_0 ps_5_0, D3D11)\';
            return _orig.call(this, p);
        }, \'getParameter\');
    }
    try { if (window.WebGLRenderingContext)  _patchGL(WebGLRenderingContext);  } catch(_) {}
    try { if (window.WebGL2RenderingContext) _patchGL(WebGL2RenderingContext); } catch(_) {}
})();

// 11. AudioContext — add imperceptible per-call noise to prevent sample-hash fingerprinting.
(function() {
    try {
        const _origGCD = AudioBuffer.prototype.getChannelData;
        AudioBuffer.prototype.getChannelData = _markNative(function(ch) {
            const d = _origGCD.call(this, ch);
            if (d.length > 0) d[0] += 1e-8 * (Math.random() - 0.5);
            return d;
        }, \'getChannelData\');
    } catch(_) {}
})();

// 12. Screen & window geometry — match configured viewport, avoid 0×0 headless defaults.
(function() {
    const _sw = window.screen.width  || 1920;
    const _sh = window.screen.height || 1080;
    const _defs = [
        [screen,  \'colorDepth\',      24],
        [screen,  \'pixelDepth\',      24],
        [screen,  \'availWidth\',      _sw],
        [screen,  \'availHeight\',     _sh - 40],
        [window,  \'devicePixelRatio\', 1],
        [window,  \'outerWidth\',      _sw],
        [window,  \'outerHeight\',     _sh],
    ];
    for (const [obj, prop, val] of _defs) {
        try { Object.defineProperty(obj, prop, {get: () => val, configurable: true}); } catch(_) {}
    }
})();

// 13. deviceMemory — 8 GB desktop default (absence or 1 GB is a headless signal).
try { Object.defineProperty(navigator, \'deviceMemory\', {get: () => 8, configurable: true}); } catch(_) {}

// 14. Media devices — realistic audio-only stub (empty list is a headless signal).
(function() {
    try {
        if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
            navigator.mediaDevices.enumerateDevices = _markNative(async function() {
                return [
                    {deviceId: \'default\', kind: \'audioinput\',  label: \'\', groupId: \'default\'},
                    {deviceId: \'default\', kind: \'audiooutput\', label: \'\', groupId: \'default\'},
                ];
            }, \'enumerateDevices\');
        }
    } catch(_) {}
})();

// 15. Remaining navigator properties and window.name
try { Object.defineProperty(navigator,\'vendor\',        {get:()=>\'Google Inc.\',configurable:true}); } catch(_) {}
try { Object.defineProperty(navigator,\'maxTouchPoints\',{get:()=>0,           configurable:true}); } catch(_) {}
try { Object.defineProperty(navigator,\'cookieEnabled\', {get:()=>true,         configurable:true}); } catch(_) {}
try { Object.defineProperty(navigator,\'onLine\',        {get:()=>true,         configurable:true}); } catch(_) {}
try { if (!window.name) window.name = \'\'; } catch(_) {}
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
        timezone_id="America/New_York",
        extra_http_headers=_SEC_CH_UA_HEADERS,
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

# BrowserGpuConfig.launch_args() detects /dev/dri at startup and returns
# GPU-accelerated args (--use-gl=egl, VA-API) when hardware is available,
# or --disable-gpu when this container has no DRI device passthrough.
_LAUNCH_ARGS = BrowserGpuConfig.launch_args()


async def _restart_browser():
    """Restart the entire Playwright browser after a crash (WebSocket closed, OOM, etc.)."""
    global _browser, _context, _page
    for obj in (_page, _context, _browser):
        try:
            if obj is not None:
                await obj.close()
        except Exception:
            pass
    _page = _context = _browser = None
    last_exc = None
    for exe in _CHROMIUM_PATHS:
        try:
            kwargs = {"headless": True, "args": _LAUNCH_ARGS}
            if exe:
                kwargs["executable_path"] = exe
            _browser = await _pw.chromium.launch(**kwargs)
            break
        except Exception as exc:
            last_exc = exc
    if _browser is None:
        raise RuntimeError(f"Browser restart failed: {last_exc}")
    await _new_context()
    await _new_page()


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
    await _new_page()
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
    """Return a healthy page, recreating context or browser if needed after a crash."""
    global _context, _page, _page_lock
    if _page_lock is None:
        _page_lock = asyncio.Lock()
    async with _page_lock:
        # 1. Existing page still alive?
        try:
            if _page is not None and not _page.is_closed():
                return _page
        except Exception:
            pass
        # 2. Recreate page (and context if closed) within existing browser
        try:
            if _context is None or _context.is_closed():
                await _new_context()
            _page = await _new_page()
            return _page
        except Exception:
            pass
        # 3. Browser itself has crashed (WebSocket closed, OOM) — restart entirely
        await _restart_browser()
        return _page


async def _extract_text(page) -> str:
    js = """() => {
        const b = document.body;
        if (!b) return '' ;
        const clone = b.cloneNode(true);
        clone.querySelectorAll(
            'script,style,nav,footer,header,aside,noscript'
        ).forEach(e => e.remove());
        return clone.innerText.trim().slice(0, 8000);
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
    """Navigate to url. 'load' is fast and sufficient; networkidle can stall 30s+."""
    last_exc: Exception | None = None
    for wait, ms in (("load", 12000), ("domcontentloaded", 5000)):
        try:
            await page.goto(url, wait_until=wait, timeout=ms)
            return
        except Exception as exc:
            last_exc = exc
            if page.is_closed():
                raise
    if last_exc is not None:
        raise last_exc


# Phrases that indicate the browser session was blocked by bot-detection.
# When detected in the page text, the navigate handler rotates the context
# (fresh cookies/fingerprint) and retries once automatically.
_BLOCK_SIGNALS = [
    "you\'ve been blocked",
    "access denied",
    "enable javascript and cookies",
    "checking your browser",
    "please enable cookies",
    "cloudflare ray id",
    "attention required",
    "ddos protection by cloudflare",
    "please wait while we verify",
    "your ip address has been",
    "unusual traffic from your",
    "sorry, humans only",
    "are you a robot",
    "challenge-platform",
    # captcha providers
    "hcaptcha.com/captcha",
    "recaptcha/api.js",
    "g-recaptcha",
    "cf-turnstile",
    "press and hold",
    "verify you are human",
    "complete the captcha",
    "this site is protected by hcaptcha",
    "bot protection by",
    "ddos-guard",
    "just a moment",
]

# Sites known to aggressively fingerprint — always rotate context before visiting
_ALWAYS_ROTATE_DOMAINS = (
    "twitter.com", "x.com",
    "reddit.com", "old.reddit.com",
    "pinterest.com",
    "instagram.com",
)

# Ad networks and trackers to block via route interception.
# Requests to URLs containing any of these strings are aborted before they load,
# removing ad iframes, banners, and tracking pixels from screenshots.
_AD_DOMAINS = {
    "doubleclick.net", "googlesyndication.com", "googletagmanager.com",
    "google-analytics.com", "googleadservices.com", "googleoptimize.com",
    "adnxs.com", "adsafeprotected.com", "criteo.com", "rubiconproject.com",
    "pubmatic.com", "openx.net", "casalemedia.com", "advertising.com",
    "adtech.de", "scorecardresearch.com", "quantserve.com",
    "facebook.net", "connect.facebook.net",
    "ads.twitter.com", "static.ads-twitter.com",
    "amazon-adsystem.com", "moatads.com", "outbrain.com", "taboola.com",
    "yieldmo.com", "sharethrough.com", "33across.com", "appnexus.com",
    "bidswitch.net", "turn.com", "rlcdn.com", "smaato.net",
    "bat.bing.com", "hotjar.com", "mouseflow.com", "fullstory.com",
    "logrocket.com", "chartbeat.com", "newrelic.com", "nr-data.net",
    "amazon-adsystem.com", "media.net", "revcontent.com", "mgid.com",
    "disqusads.com", "cdn.jsdelivr.net/npm/adsbygoogle",
}


async def _route_handler(route):
    """Abort ad/tracker requests; pass everything else through."""
    try:
        if any(d in route.request.url for d in _AD_DOMAINS):
            await route.abort()
        else:
            await route.continue_()
    except Exception:
        try:
            await route.continue_()
        except Exception:
            pass


async def _new_page():
    """Create a new page with ad-blocking routes pre-installed."""
    global _page
    _page = await _context.new_page()
    await _page.route("**/*", _route_handler)
    return _page


async def _human_move(page):
    """Simulate brief human-like mouse movement — required by PerimeterX / DataDome
    which check that at least one mousemove event fires before interaction."""
    try:
        vp = page.viewport_size or {"width": 1280, "height": 720}
        w, h = vp["width"], vp["height"]
        x = w // 2 + _random.randint(-180, 180)
        y = h // 3 + _random.randint(-60, 60)
        await page.mouse.move(x, y)
        for _ in range(_random.randint(2, 5)):
            await asyncio.sleep(_random.uniform(0.04, 0.14))
            x = max(40, min(w - 40, x + _random.randint(-120, 120)))
            y = max(40, min(h - 40, y + _random.randint(-80, 160)))
            await page.mouse.move(x, y)
        await asyncio.sleep(_random.uniform(0.05, 0.15))
    except Exception:
        pass


def _is_blocked(text: str) -> bool:
    low = text.lower()
    return any(sig in low for sig in _BLOCK_SIGNALS)


async def _rotate_context_and_page() -> object:
    """Close the current browser context and open a fresh one with new cookies/state."""
    global _context, _page
    try:
        if _context and not _context.is_closed():
            await _context.close()
    except Exception:
        pass
    try:
        await _new_context()
        await _new_page()
    except Exception:
        # Browser may have crashed — restart it entirely
        await _restart_browser()
    return _page


@app.get("/health")
async def health():
    return {"ok": True, "version": _VERSION, "gpu": BrowserGpuConfig.info()}


class NavReq(BaseModel):
    url: str
    wait: str = "networkidle"


def _site_fallback(url: str) -> str | None:
    """Return an alternative URL for sites known to block datacenter IPs."""
    import re as _re
    # new Reddit → old Reddit (much lighter anti-bot stance)
    m = _re.match(r\'https?://(?:www\\.)?reddit\\.com(.*)\', url)
    if m:
        return "https://old.reddit.com" + m.group(1)
    return None


@app.post("/navigate")
async def navigate(req: NavReq):
    try:
        page = await _ensure_page()
        await _safe_goto(page, req.url)
        await _human_move(page)
        content = await _extract_text(page)
        # Auto-rotate context and retry once if bot-detection is triggered
        if _is_blocked(content):
            # 1st try: rotate context (fresh cookies) and retry same URL
            page = await _rotate_context_and_page()
            await _safe_goto(page, req.url)
            content = await _extract_text(page)
        # 2nd try: if still blocked, try a known-good fallback URL
        if _is_blocked(content):
            fallback = _site_fallback(req.url)
            if fallback:
                await _safe_goto(page, fallback)
                content = await _extract_text(page)
        return {
            "title": await page.title(),
            "url": page.url,
            "content": content,
        }
    except Exception as exc:
        try:
            await _ensure_page()
        except Exception:
            pass
        return {"error": str(exc), "content": "", "title": "", "url": req.url}


class ClickReq(BaseModel):
    selector: str
    button: str = "left"
    click_count: int = 1


@app.post("/click")
async def click(req: ClickReq):
    try:
        button = (req.button or "left").strip().lower()
        if button not in {"left", "right", "middle"}:
            return {"error": f"invalid click button: {button}", "content": "", "url": ""}
        click_count = max(1, min(int(req.click_count), 3))
        page = await _ensure_page()
        await page.click(
            req.selector,
            timeout=10000,
            button=button,
            click_count=click_count,
        )
        try:
            await page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            pass
        return {
            "url": page.url,
            "content": await _extract_text(page),
            "button": button,
            "click_count": click_count,
        }
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


class ScrollReq(BaseModel):
    direction: str = "down"
    amount: int = 800
    behavior: str = "instant"


@app.post("/scroll")
async def scroll(req: ScrollReq):
    try:
        page = await _ensure_page()
        direction = (req.direction or "down").strip().lower()
        amount = max(1, min(int(req.amount), 20000))
        dx = 0
        dy = 0
        if direction == "down":
            dy = amount
        elif direction == "up":
            dy = -amount
        elif direction == "right":
            dx = amount
        elif direction == "left":
            dx = -amount
        else:
            return {"error": f"invalid scroll direction: {direction}", "url": page.url}
        behavior = (req.behavior or "instant").strip().lower()
        if behavior not in {"instant", "smooth"}:
            behavior = "instant"
        await page.evaluate(
            """(p) => {
                window.scrollBy({left: p.dx, top: p.dy, behavior: p.behavior});
                return {
                    x: window.scrollX || window.pageXOffset || 0,
                    y: window.scrollY || window.pageYOffset || 0
                };
            }""",
            {"dx": dx, "dy": dy, "behavior": behavior},
        )
        await asyncio.sleep(0.2)
        pos = await page.evaluate(
            "() => ({x: window.scrollX || window.pageXOffset || 0, y: window.scrollY || window.pageYOffset || 0})"
        )
        return {
            "ok": True,
            "direction": direction,
            "amount": amount,
            "behavior": behavior,
            "scroll_x": int(pos.get("x", 0)),
            "scroll_y": int(pos.get("y", 0)),
            "url": page.url,
            "content": await _extract_text(page),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "content": "", "url": ""}


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
    find_image: Optional[str] = None  # match <img> by src/alt substring OR 1-based index (e.g. "logo", "2", "#3")


@app.post("/screenshot")
async def screenshot(req: ScreenshotReq):
    page = await _ensure_page()
    nav_error = None
    if req.url:
        try:
            await _safe_goto(page, req.url)
            await _human_move(page)
        except Exception as exc:
            nav_error = str(exc)
            # Page may have crashed — get a fresh one and retry
            try:
                page = await _ensure_page()
                await _safe_goto(page, req.url)
                await _human_move(page)
                nav_error = None
            except Exception as exc2:
                nav_error = str(exc2)
    try:
        # Brief human-like pause before screenshot (reduced for speed)
        await asyncio.sleep(_random.uniform(0.2, 0.5))

        clipped = False
        image_meta: dict = {}
        if req.find_image:
            # Precise image capture: find an <img> element by src/alt pattern or 1-based index,
            # scroll it to center, and clip the screenshot to its exact bounding box.
            try:
                import json as _json
                found_img = await page.evaluate(
                    """(query) => {
                        const imgs = Array.from(document.querySelectorAll(\'img\'));
                        let target = null;
                        // 1-based index: "#3" or "3"
                        const idxM = query.match(/^#?(\\d+)$/);
                        if (idxM) {
                            const idx = parseInt(idxM[1]) - 1;
                            target = imgs[idx] || null;
                        } else {
                            const lower = query.toLowerCase();
                            for (const img of imgs) {
                                const src = (img.src || \'\').toLowerCase();
                                const alt = (img.alt || \'\').toLowerCase();
                                if (src.includes(lower) || alt.includes(lower)) {
                                    target = img;
                                    break;
                                }
                            }
                        }
                        if (!target) return {found: false};
                        target.scrollIntoView({behavior: \'instant\', block: \'center\'});
                        const r = target.getBoundingClientRect();
                        return {found: true, x: r.left, y: r.top, w: r.width, h: r.height,
                                src: target.src, alt: target.alt,
                                nw: target.naturalWidth, nh: target.naturalHeight};
                    }""",
                    req.find_image,
                )
                await asyncio.sleep(_random.uniform(0.1, 0.2))
                if found_img.get("found") and found_img.get("w", 0) > 0:
                    pad = 8  # tight padding for precise image capture
                    vp = page.viewport_size or {"width": 1280, "height": 900}
                    x = max(0, found_img["x"] - pad)
                    y = max(0, found_img["y"] - pad)
                    w = min(vp["width"] - x, found_img["w"] + 2 * pad)
                    h = min(vp["height"] - y, found_img["h"] + 2 * pad)
                    await page.screenshot(
                        path=req.path,
                        clip={"x": x, "y": y, "width": w, "height": h},
                    )
                    clipped = True
                    image_meta = {
                        "src": found_img.get("src", ""),
                        "alt": found_img.get("alt", ""),
                        "natural_width":  found_img.get("nw", 0),
                        "natural_height": found_img.get("nh", 0),
                    }
                else:
                    # Image not found — fall through to normal screenshot
                    await page.screenshot(path=req.path, full_page=False)
            except Exception:
                await page.screenshot(path=req.path, full_page=False)

        elif req.find_text:
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
            # If page is a bot-check/block page, rotate context and retry once
            page_text = await _extract_text(page)
            if _is_blocked(page_text):
                page = await _rotate_context_and_page()
                if req.url:
                    await _safe_goto(page, req.url)
                    await _human_move(page)
                await page.screenshot(path=req.path, full_page=False)

        result = {"path": req.path, "title": await page.title(), "url": page.url,
                  "clipped": clipped}
        if image_meta:
            result["image_meta"] = image_meta
        if nav_error:
            result["nav_error"] = nav_error
        return result
    except Exception as exc:
        # Screenshot failed — attempt browser recovery then return fallback info
        img_urls: list = []
        try:
            page = await _ensure_page()
            if not page.is_closed():
                img_urls = await _get_image_urls(page)
        except Exception:
            pass
        return {
            "error": str(exc),
            "path": req.path,
            "title": "",
            "url": "",
            "image_urls": img_urls,
        }


class ElementScreenshotReq(BaseModel):
    selector: str
    path: str = "/workspace/element.png"
    pad: int = 20  # padding around the element in pixels


@app.post("/screenshot_element")
async def screenshot_element(req: ElementScreenshotReq):
    """Screenshot a single page element identified by CSS selector."""
    try:
        page = await _ensure_page()
        locator = page.locator(req.selector).first
        bbox = await locator.bounding_box()
        if not bbox:
            return {"error": f"Element \'{req.selector}\' not found or has no bounding box"}
        await locator.scroll_into_view_if_needed()
        await asyncio.sleep(_random.uniform(0.1, 0.2))
        vp = page.viewport_size or {"width": 1280, "height": 900}
        x = max(0, bbox["x"] - req.pad)
        y = max(0, bbox["y"] - req.pad)
        w = min(vp["width"]  - x, bbox["width"]  + 2 * req.pad)
        h = min(vp["height"] - y, bbox["height"] + 2 * req.pad)
        if w <= 0 or h <= 0:
            return {"error": f"Element \'{req.selector}\' has zero-size bounding box"}
        await page.screenshot(path=req.path, clip={"x": x, "y": y, "width": w, "height": h})
        return {
            "path": req.path,
            "selector": req.selector,
            "clip": {"x": x, "y": y, "width": w, "height": h},
            "title": await page.title(),
            "url": page.url,
        }
    except Exception as exc:
        return {"error": str(exc), "path": req.path}


@app.get("/images")
async def list_images():
    """Return metadata for all visible <img> elements on the current page."""
    try:
        page = await _ensure_page()
        data = await page.evaluate(
            """() => {
                const imgs = Array.from(document.querySelectorAll(\'img\'));
                return imgs.slice(0, 30).map(function(img, i) {
                    const r = img.getBoundingClientRect();
                    return {
                        index:          i + 1,
                        src:            img.src || \'\',
                        alt:            img.alt || \'\',
                        rendered_w:     Math.round(r.width),
                        rendered_h:     Math.round(r.height),
                        natural_w:      img.naturalWidth  || 0,
                        natural_h:      img.naturalHeight || 0,
                        visible:        r.width > 1 && r.height > 1,
                        in_viewport:    r.top < window.innerHeight && r.bottom > 0
                            && r.left < window.innerWidth && r.right > 0
                    };
                });
            }"""
        )
        return {"images": data, "count": len(data)}
    except Exception as exc:
        return {"images": [], "count": 0, "error": str(exc)}


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


_MIME_EXT: dict = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/svg+xml": ".svg",
    "image/tiff": ".tiff",
    "image/avif": ".avif",
}


class SaveImagesReq(BaseModel):
    urls: list
    prefix: str = "image"
    max: int = 20


class DownloadPageImagesReq(BaseModel):
    filter: Optional[str] = None
    max: int = 20
    prefix: str = "image"


async def _save_url_via_browser(page, url: str, path: str) -> dict:
    """Download a single image URL using the browser\'s session (cookies, auth, referrer)."""
    import os as _os
    try:
        resp = await page.request.get(url, timeout=30000)
        if not resp.ok:
            return {"url": url, "error": f"HTTP {resp.status}"}
        body = await resp.body()
        ct = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
        ext = _MIME_EXT.get(ct, ".jpg")
        # Replace any placeholder extension in path with the real one
        base, _ = _os.path.splitext(path)
        final_path = base + ext
        _os.makedirs(_os.path.dirname(final_path), exist_ok=True)
        with open(final_path, "wb") as f:
            f.write(body)
        return {
            "url": url,
            "path": final_path,
            "mime": ct,
            "size": len(body),
        }
    except Exception as exc:
        return {"url": url, "error": str(exc)}


@app.post("/save_images")
async def save_images(req: SaveImagesReq):
    """Download a list of image URLs using the browser\'s live session (cookies, auth, referrer)."""
    import datetime as _dt, os as _os
    page = await _ensure_page()
    urls = req.urls[:min(req.max, 50)]
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    saved = []
    errors = []
    for idx, url in enumerate(urls):
        path = f"/workspace/{req.prefix}_{idx + 1}_{ts}.tmp"
        result = await _save_url_via_browser(page, url, path)
        if "error" in result:
            errors.append(result)
        else:
            saved.append({**result, "index": idx + 1})
    return {"saved": saved, "errors": errors, "count": len(saved)}


@app.post("/download_page_images")
async def download_page_images(req: DownloadPageImagesReq):
    """Download all (or filtered) <img> elements from the current page using browser session."""
    page = await _ensure_page()
    # Collect all img srcs + alts from DOM
    img_data = await page.evaluate("""() => {
        const imgs = Array.from(document.querySelectorAll(\'img[src]\'));
        return imgs.slice(0, 100).map(img => ({src: img.src || \'\', alt: img.alt || \'\'}));
    }""")
    # Filter by src/alt substring if requested
    if req.filter:
        lower = req.filter.lower()
        img_data = [i for i in img_data
                    if lower in i["src"].lower() or lower in i["alt"].lower()]
    # Deduplicate by src
    seen = set()
    urls = []
    for item in img_data:
        src = item["src"]
        if src and src not in seen and src.startswith("http"):
            seen.add(src)
            urls.append(src)
    urls = urls[:min(req.max, 50)]
    # Reuse save logic
    import datetime as _dt
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    saved = []
    errors = []
    for idx, url in enumerate(urls):
        path = f"/workspace/{req.prefix}_{idx + 1}_{ts}.tmp"
        result = await _save_url_via_browser(page, url, path)
        if "error" in result:
            errors.append(result)
        else:
            saved.append({**result, "index": idx + 1})
    return {"saved": saved, "errors": errors, "count": len(saved), "filter": req.filter}


class ScrapeReq(BaseModel):
    url: str = ""
    max_scrolls: int = 10
    wait_ms: int = 500
    max_chars: int = 16000
    include_links: bool = False


async def _scroll_full_page(page, max_scrolls: int, wait_ms_s: float) -> dict:
    """Scroll through the page in viewport-sized increments, pausing after each
    step so lazy-loaded content (images, infinite-scroll items, JS widgets) has
    time to appear.  Returns scroll stats including whether the page grew."""
    try:
        vp_h = int(await page.evaluate("window.innerHeight") or 800)
    except Exception:
        vp_h = 800
    step = max(vp_h - 100, 200)   # 100 px overlap keeps edge lazy-loaders alive
    scroll_y, prev_h, grew, steps = 0, 0, False, 0
    for i in range(max_scrolls):
        try:
            await page.evaluate(f"window.scrollTo(0, {scroll_y})")
            await asyncio.sleep(wait_ms_s * _random.uniform(0.6, 1.5))
            cur_h = int(await page.evaluate("document.documentElement.scrollHeight") or 0)
        except Exception:
            break
        if cur_h > prev_h and i > 0:
            grew = True          # infinite-scroll: new content arrived
        prev_h = cur_h
        steps = i + 1
        if scroll_y >= cur_h:
            break                # reached the bottom
        scroll_y += step
    # Return to top so any subsequent screenshot or read starts at the header
    try:
        await page.evaluate("window.scrollTo(0, 0)")
        await asyncio.sleep(0.2)
    except Exception:
        pass
    return {"steps": steps, "content_grew": grew, "final_height": prev_h}


async def _extract_text_long(page, max_chars: int = 16000) -> str:
    """Extract all readable text from the fully-rendered DOM.
    Prefers <main>/<article> if present (avoids nav/footer noise).
    max_chars is configurable — pass 0 for no limit (returns all text)."""
    limit = max_chars if max_chars > 0 else 999999
    js = (
        "() => {"
        "const b = document.body;"
        "if (!b) return \'\';"
        "b.querySelectorAll(\'script,style,noscript,iframe\').forEach(e => e.remove());"
        "const m = b.querySelector(\'main,article\');"
        "return (m || b).innerText.trim().slice(0," + str(limit) + ");"
        "}"
    )
    try:
        return await page.evaluate(js)
    except Exception:
        return ""


@app.post("/scrape")
async def scrape(req: ScrapeReq):
    """Navigate (optional), scroll through the full page waiting for lazy loads,
    then extract complete rendered text from the final DOM state."""
    try:
        page = await _ensure_page()

        # Navigate if a URL was supplied; reuse block-detection + fallback logic
        if req.url:
            await _safe_goto(page, req.url)
            quick = await _extract_text(page)
            if _is_blocked(quick):
                page = await _rotate_context_and_page()
                await _safe_goto(page, req.url)
                quick = await _extract_text(page)
            if _is_blocked(quick):
                fb = _site_fallback(req.url)
                if fb:
                    await _safe_goto(page, fb)

        # Scroll through the whole page so lazy content renders
        stats = await _scroll_full_page(page, req.max_scrolls, req.wait_ms / 1000.0)

        # Full text from the final (completely rendered) DOM
        content = await _extract_text_long(page, req.max_chars)

        result: dict = {
            "title": await page.title(),
            "url": page.url,
            "content": content,
            "char_count": len(content),
            "scroll_steps": stats["steps"],
            "content_grew_on_scroll": stats["content_grew"],
            "final_page_height": stats["final_height"],
        }

        if req.include_links:
            try:
                links = await page.evaluate(
                    "() => Array.from(document.querySelectorAll(\'a[href]\'))"
                    ".slice(0,200)"
                    ".map(a => ({text: a.innerText.trim().slice(0,100), href: a.href}))"
                    ".filter(l => l.href.startsWith(\'http\'))"
                )
                result["links"] = links
            except Exception:
                result["links"] = []

        return result

    except Exception as exc:
        try:
            await _ensure_page()
        except Exception:
            pass
        return {
            "error": str(exc), "content": "", "title": "", "url": req.url,
            "scroll_steps": 0, "content_grew_on_scroll": False, "final_page_height": 0,
        }


class PageImagesReq(BaseModel):
    url: str = ""
    scroll: bool = True
    max_scrolls: int = 3   # fewer needed for URL discovery vs full-text extraction


@app.post("/page_images")
async def page_images_endpoint(req: PageImagesReq):
    """Navigate (optional), scroll to trigger lazy loaders, then extract ALL
    image URLs: img src, highest-res srcset, data-src/lazy variants, <picture>
    sources, og:image, twitter:image, inline CSS background-image, JSON-LD."""
    try:
        page = await _ensure_page()
        if req.url:
            # Pre-rotate context for sites that aggressively fingerprint browsers
            if any(d in req.url for d in _ALWAYS_ROTATE_DOMAINS):
                page = await _rotate_context_and_page()
            await _safe_goto(page, req.url)
            quick = await _extract_text(page)
            if _is_blocked(quick):
                page = await _rotate_context_and_page()
                await _safe_goto(page, req.url)
                quick = await _extract_text(page)
            if _is_blocked(quick):
                fb = _site_fallback(req.url)
                if fb:
                    await _safe_goto(page, fb)
        if req.scroll:
            await _scroll_full_page(page, req.max_scrolls, 0.4)
        images = await page.evaluate("""
() => {
    const base = document.location.href;
    function abs(u) {
        if (!u) return \'\';
        try { return new URL(u, base).href; } catch(e) { return \'\'; }
    }
    const seen = new Set();
    const result = [];
    function add(rawUrl, type, meta) {
        const u = abs(rawUrl);
        if (!u || !u.startsWith(\'http\') || seen.has(u)) return;
        seen.add(u);
        result.push(Object.assign({url: u, type: type}, meta));
    }
    function bestSrcset(srcset) {
        let best = \'\', bestW = 0;
        for (const part of (srcset || \'\').split(\',\')) {
            const tok = part.trim().split(/\\s+/);
            if (!tok[0]) continue;
            const w = tok[1] ? (parseFloat(tok[1]) || 0) : 0;
            if (w > bestW || !best) { best = tok[0]; bestW = w; }
        }
        return {url: best, w: bestW};
    }
    // 1. <img> src + data-src lazy variants + highest-res srcset
    for (const img of document.querySelectorAll(\'img\')) {
        const src = img.getAttribute(\'src\') || \'\';
        if (src) add(src, \'img\', {alt: img.alt||\'\'});
        for (const attr of [\'data-src\',\'data-lazy-src\',\'data-original\',\'data-lazy\',\'data-echo\']) {
            const v = img.getAttribute(attr) || \'\';
            if (v) add(v, \'lazy\', {alt: img.alt||\'\'});
        }
        const ss = img.getAttribute(\'srcset\') || img.getAttribute(\'data-srcset\') || \'\';
        if (ss) { const b = bestSrcset(ss); if (b.url) add(b.url, \'srcset\', {srcset_width: b.w}); }
    }
    // 2. <picture><source> — pick highest descriptor per element
    for (const src of document.querySelectorAll(\'picture source[srcset]\')) {
        const b = bestSrcset(src.getAttribute(\'srcset\') || \'\');
        if (b.url) add(b.url, \'picture\', {srcset_width: b.w});
    }
    // 3. og:image / twitter:image meta tags
    for (const m of document.querySelectorAll(\'meta[property="og:image"],meta[name="twitter:image"],meta[name="twitter:image:src"]\')) {
        const c = m.getAttribute(\'content\') || \'\';
        if (c) add(c, \'og_meta\', {});
    }
    // 4. Inline CSS style attribute background-image
    for (const el of document.querySelectorAll(\'[style*="background"]\')) {
        const bg = el.style.backgroundImage || \'\';
        const m = bg.match(/url\\(["\'\\s]*(https?:[^"\'\\)\\s]+)["\'\\s]*\\)/);
        if (m) add(m[1], \'css_bg\', {});
    }
    // 5. JSON-LD schema.org image / logo / thumbnail
    for (const s of document.querySelectorAll(\'script[type="application/ld+json"]\')) {
        try {
            const d = JSON.parse(s.textContent || \'{}\');
            const fields = [d.image, d.logo, d.thumbnail].flat().filter(Boolean);
            for (const field of fields) {
                const u = typeof field === \'string\' ? field : (field.url||field.contentUrl||field["@id"]||\'\');
                if (u) add(u, \'json_ld\', {});
            }
        } catch(e) {}
    }
    return result.slice(0, 150);
}
""")
        return {
            "images": images,
            "count": len(images),
            "title": await page.title(),
            "url": page.url,
        }
    except Exception as exc:
        try:
            await _ensure_page()
        except Exception:
            pass
        return {"error": str(exc), "images": [], "count": 0, "title": "", "url": req.url}


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
        deadline = asyncio.get_running_loop().time() + _STARTUP_TIMEOUT
        while asyncio.get_running_loop().time() < deadline:
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

    async def screenshot(
        self,
        url: str | None = None,
        find_text: str | None = None,
        find_image: str | None = None,
    ) -> dict:
        base = await self._ensure_server()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"/workspace/screenshot_{ts}.png"
        payload: dict = {"path": path}
        if url:
            payload["url"] = url
        if find_text:
            payload["find_text"] = find_text
        if find_image:
            payload["find_image"] = find_image
        async with httpx.AsyncClient(timeout=45.0) as c:
            r = await c.post(f"{base}/screenshot", json=payload)
        r.raise_for_status()
        data = r.json()
        # Expose the host-side path (workspace is bind-mounted)
        data["host_path"] = f"/docker/human_browser/workspace/screenshot_{ts}.png"
        return data

    async def screenshot_element(self, selector: str, pad: int = 20) -> dict:
        """Screenshot a single DOM element by CSS selector."""
        base = await self._ensure_server()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"/workspace/element_{ts}.png"
        async with httpx.AsyncClient(timeout=20.0) as c:
            r = await c.post(f"{base}/screenshot_element",
                             json={"selector": selector, "path": path, "pad": pad})
        r.raise_for_status()
        data = r.json()
        data["host_path"] = f"/docker/human_browser/workspace/element_{ts}.png"
        return data

    async def list_images(self) -> dict:
        """Get image metadata (index, src, alt, dimensions, visibility) for the current page."""
        base = await self._ensure_server()
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(f"{base}/images")
        r.raise_for_status()
        return r.json()

    async def save_images(
        self,
        urls: list[str],
        prefix: str = "image",
        max_images: int = 20,
    ) -> dict:
        """Download specific image URLs using the browser's authenticated session.

        Uses Playwright page.request.get() — shares cookies, auth tokens, and referrer
        with the live browser page, exactly like a human right-clicking 'Save Image As'.
        """
        base = await self._ensure_server()
        async with httpx.AsyncClient(timeout=120.0) as c:
            r = await c.post(
                f"{base}/save_images",
                json={"urls": urls[:max_images], "prefix": prefix, "max": max_images},
            )
        r.raise_for_status()
        data = r.json()
        for item in data.get("saved", []):
            fname = os.path.basename(item.get("path", ""))
            if fname:
                item["host_path"] = f"/docker/human_browser/workspace/{fname}"
        return data

    async def download_page_images(
        self,
        filter_query: str | None = None,
        max_images: int = 20,
        prefix: str = "image",
    ) -> dict:
        """Download all (or filtered) images from the current page via browser session."""
        base = await self._ensure_server()
        payload: dict = {"max": max_images, "prefix": prefix}
        if filter_query:
            payload["filter"] = filter_query
        async with httpx.AsyncClient(timeout=120.0) as c:
            r = await c.post(f"{base}/download_page_images", json=payload)
        r.raise_for_status()
        data = r.json()
        for item in data.get("saved", []):
            fname = os.path.basename(item.get("path", ""))
            if fname:
                item["host_path"] = f"/docker/human_browser/workspace/{fname}"
        return data

    async def scrape(
        self,
        url: str = "",
        max_scrolls: int = 10,
        wait_ms: int = 500,
        max_chars: int = 16000,
        include_links: bool = False,
    ) -> dict:
        """Navigate (optional), scroll through the full page triggering lazy loads,
        and return the complete rendered text from the final DOM state."""
        base = await self._ensure_server()
        payload: dict = {
            "url": url,
            "max_scrolls": max_scrolls,
            "wait_ms": wait_ms,
            "max_chars": max_chars,
            "include_links": include_links,
        }
        # Scraping can be slow on long pages — generous timeout
        async with httpx.AsyncClient(timeout=max_scrolls * 3 + 30) as c:
            r = await c.post(f"{base}/scrape", json=payload)
        r.raise_for_status()
        return r.json()

    async def click(
        self,
        selector: str,
        button: str = "left",
        click_count: int = 1,
    ) -> dict:
        base = await self._ensure_server()
        async with httpx.AsyncClient(timeout=20.0) as c:
            r = await c.post(
                f"{base}/click",
                json={
                    "selector": selector,
                    "button": button,
                    "click_count": click_count,
                },
            )
        r.raise_for_status()
        return r.json()

    async def scroll(
        self,
        direction: str = "down",
        amount: int = 800,
        behavior: str = "instant",
    ) -> dict:
        base = await self._ensure_server()
        async with httpx.AsyncClient(timeout=20.0) as c:
            r = await c.post(
                f"{base}/scroll",
                json={
                    "direction": direction,
                    "amount": amount,
                    "behavior": behavior,
                },
            )
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

    async def page_images(
        self,
        url: str = "",
        scroll: bool = True,
        max_scrolls: int = 3,
    ) -> dict:
        """Extract all image URLs from the current page (or navigate first).
        Returns a deduplicated list with source type (img/srcset/lazy/picture/og_meta/css_bg/json_ld)."""
        base = await self._ensure_server()
        payload = {"url": url, "scroll": scroll, "max_scrolls": max_scrolls}
        timeout = max_scrolls * 2.0 + 30.0
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.post(f"{base}/page_images", json=payload)
        r.raise_for_status()
        return r.json()
