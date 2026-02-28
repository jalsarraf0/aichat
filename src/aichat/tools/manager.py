from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
from collections.abc import Awaitable, Callable
from enum import Enum

from ..state import ApprovalMode
from .browser import BrowserTool
from .code_interpreter import CodeInterpreterTool
from .conversation_store import ConversationStoreTool
from .database import DatabaseTool
from .lm_studio import LMStudioTool
from .memory import MemoryTool
from .researchbox import ResearchboxTool
from .search import WebSearchTool
from .shell import ShellTool
from .toolkit import ToolkitTool


class ToolDeniedError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Dangerous-command blocklist (shell safety)
# ---------------------------------------------------------------------------

_DANGEROUS_RE = re.compile(
    r"""(
        # rm with recursive+force targeting root, home, or critical paths
        \brm\b[^|&;#\n]*-[a-zA-Z]*[rf][a-zA-Z]*[rf][a-zA-Z]*\s+(/[^/]|~/?[^/]|/\*)
        |\brm\b[^|&;#\n]*-[a-zA-Z]*f[a-zA-Z]*r[a-zA-Z]*\s+(/[^/]|~/?[^/]|/\*)
        # fork bomb
        |:\(\)\s*\{[^}]*:\s*\|[^}]*:[^}]*&[^}]*\}[^;]*;
        # mkfs / wipefs / dd to raw disk
        |\bmkfs\b
        |\bwipefs\b
        |\bdd\b[^|&;\n]*\bof\s*=\s*/dev/[a-z]
        # overwrite MBR/disk
        |\bdd\b[^|&;\n]*\bif\s*=\s*/dev/zero[^|&;\n]*\bof\s*=\s*/dev/[a-z]
        # chmod/chown 777 or world-writable on root or /etc
        |\bchmod\b[^|&;\n]*(777|[ao][+\-=]w)[^|&;\n]*(/\s*$|/etc|/usr|/bin|/sbin|/lib|/boot)
        # truncate / shred critical paths
        |\bshred\b[^|&;\n]*(/(bin|sbin|lib|boot|etc|usr)\b|/dev/)
        # kill -9 1 (kill PID 1 = init)
        |\bkill\s+-9\s+1\b
        # Python/Perl one-liner writing to /dev/sda etc.
        |\bopen\s*\(\s*['"]/dev/[a-z]
    )""",
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)


def _is_dangerous(command: str) -> bool:
    """Return True if the command matches a known-dangerous pattern."""
    return bool(_DANGEROUS_RE.search(command))


# Realistic browser headers for all outbound HTTP requests — reduces bot detection
# and rate-limit exposure compared to minimal "Mozilla/5.0" user-agent strings.
_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "DNT": "1",
    "Upgrade-Insecure-Requests": "1",
}
_IMG_FETCH_HEADERS = {
    **_FETCH_HEADERS,
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
}

# LM Studio endpoint for vision confirmation (same env-var as image generation)
_LM_STUDIO_URL = os.environ.get(
    "IMAGE_GEN_BASE_URL", os.environ.get("LM_STUDIO_URL", "http://localhost:1234")
)


def _dhash(img: object) -> str:
    """64-bit difference hash of a PIL Image → 16-char hex string (pure PIL, no extra packages)."""
    try:
        gray = img.convert("L").resize((9, 8), 3)   # 3 = BICUBIC (int; avoids Resampling enum)
        px   = list(gray.getdata())
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


class ToolName(str, Enum):
    SHELL = "shell"
    RESEARCHBOX = "researchbox"
    RESEARCHBOX_PUSH = "researchbox_push"
    WEB_FETCH = "web_fetch"
    MEMORY_STORE = "memory_store"
    MEMORY_RECALL = "memory_recall"
    CREATE_TOOL = "create_tool"
    LIST_CUSTOM_TOOLS = "list_custom_tools"
    DELETE_CUSTOM_TOOL = "delete_custom_tool"
    WEB_SEARCH = "web_search"
    BROWSER = "browser"
    DB_STORE_ARTICLE = "db_store_article"
    DB_SEARCH = "db_search"
    DB_CACHE_STORE = "db_cache_store"
    DB_CACHE_GET = "db_cache_get"
    DB_STORE_IMAGE = "db_store_image"
    DB_LIST_IMAGES = "db_list_images"
    SCREENSHOT_SEARCH = "screenshot_search"
    FETCH_IMAGE = "fetch_image"
    CALL_CUSTOM_TOOL = "call_custom_tool"
    GET_ERRORS = "get_errors"
    TTS = "tts"
    EMBED_STORE = "embed_store"
    EMBED_SEARCH = "embed_search"
    CODE_RUN = "code_run"
    SMART_SUMMARIZE = "smart_summarize"
    IMAGE_CAPTION = "image_caption"
    STRUCTURED_EXTRACT = "structured_extract"
    CONV_SEARCH_HISTORY = "conv_search_history"


class ToolManager:
    def __init__(self, max_tool_calls_per_turn: int = 6) -> None:
        self.shell = ShellTool()
        self.researchbox = ResearchboxTool()
        self.memory = MemoryTool()
        self.toolkit = ToolkitTool()
        # human_browser container (a12fdfeaaf78) — default for ALL web operations.
        self.browser = BrowserTool()
        # Tiered search: browser (human-like) → httpx → DDG lite API.
        self.search_tool = WebSearchTool(self.browser)
        # PostgreSQL-backed storage/cache (replaces rssfeed + fetch services).
        self.db = DatabaseTool()
        # LM Studio OpenAI-compatible inference client (TTS, embeddings, chat, vision).
        self.lm = LMStudioTool(_LM_STUDIO_URL)
        # Sandboxed Python subprocess executor.
        self.code = CodeInterpreterTool()
        # Persistent conversation store (PostgreSQL via aichat-database).
        self.conv = ConversationStoreTool()
        self.max_tool_calls_per_turn = max_tool_calls_per_turn
        self._calls_this_turn = 0
        # name → {description, parameters} for custom tools loaded from toolkit
        self._custom_tools: dict[str, dict] = {}

    def reset_turn(self) -> None:
        self._calls_this_turn = 0

    # ------------------------------------------------------------------
    # Custom tool registry
    # ------------------------------------------------------------------

    async def refresh_custom_tools(self) -> None:
        """Reload custom tool metadata from the toolkit service."""
        try:
            tools = await self.toolkit.list_tools()
            self._custom_tools = {
                t["name"]: {"description": t["description"], "parameters": t["parameters"]}
                for t in tools
            }
        except Exception:
            # Toolkit service unavailable — keep last known state
            pass

    def is_custom_tool(self, name: str) -> bool:
        return name in self._custom_tools

    # ------------------------------------------------------------------
    # Approval gate
    # ------------------------------------------------------------------

    async def _check_approval(
        self,
        mode: ApprovalMode,
        tool: str,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> None:
        if self._calls_this_turn >= self.max_tool_calls_per_turn:
            raise ToolDeniedError("Tool call limit reached for current turn")
        if mode == ApprovalMode.DENY:
            raise ToolDeniedError("Tool execution denied by current approval mode")
        if mode == ApprovalMode.ASK:
            if confirmer is None or not await confirmer(tool):
                raise ToolDeniedError(f"Tool '{tool}' rejected by user")
        self._calls_this_turn += 1

    # ------------------------------------------------------------------
    # Built-in tools
    # ------------------------------------------------------------------

    async def run_shell(
        self,
        command: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
        cwd: str | None = None,
    ) -> tuple[str, str | None]:
        if _is_dangerous(command):
            raise ToolDeniedError(f"Blocked: potentially destructive command refused for safety.")
        await self._check_approval(mode, ToolName.SHELL.value, confirmer)
        exit_code, output, new_cwd = await self._run_shell_process(command, cwd=cwd)
        trimmed = output.strip()
        if exit_code != 0:
            if trimmed:
                return f"{trimmed}\n(exit {exit_code})", new_cwd
            return f"(exit {exit_code})", new_cwd
        return trimmed, new_cwd

    async def run_shell_stream(
        self,
        command: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
        *,
        cwd: str | None = None,
        on_output: Callable[[str], None] | None = None,
    ) -> tuple[int, str, str | None]:
        if _is_dangerous(command):
            raise ToolDeniedError(f"Blocked: potentially destructive command refused for safety.")
        await self._check_approval(mode, ToolName.SHELL.value, confirmer)
        return await self._run_shell_process(command, cwd=cwd, on_output=on_output)

    async def _run_shell_process(
        self,
        command: str,
        *,
        cwd: str | None = None,
        on_output: Callable[[str], None] | None = None,
    ) -> tuple[int, str, str | None]:
        marker = "__AICHAT_CWD__"
        command = self._ensure_non_interactive_sudo(command)
        wrapped = self._wrap_command_with_pwd(command, marker)
        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-lc",
            wrapped,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
            env=os.environ.copy(),
        )
        output_chunks: list[str] = []
        buffer = ""
        cwd_value: str | None = None
        marker_token = f"\n{marker}"
        assert proc.stdout is not None
        while True:
            data = await proc.stdout.read(1024)
            if not data:
                break
            text = data.decode(errors="replace")
            buffer += text
            if marker_token in buffer:
                before, after = buffer.split(marker_token, 1)
                if before:
                    output_chunks.append(before)
                    if on_output:
                        on_output(before)
                cwd_value = after.strip() or None
                buffer = ""
                continue
            if len(buffer) > len(marker_token):
                safe = buffer[:-len(marker_token)]
                if safe:
                    output_chunks.append(safe)
                    if on_output:
                        on_output(safe)
                buffer = buffer[-len(marker_token):]
        exit_code = await proc.wait()
        if buffer and cwd_value is None:
            output_chunks.append(buffer)
            if on_output:
                on_output(buffer)
        return exit_code, "".join(output_chunks).strip(), cwd_value

    def _ensure_non_interactive_sudo(self, command: str) -> str:
        stripped = command.lstrip()
        if not stripped.startswith("sudo "):
            return command
        try:
            parts = shlex.split(command)
        except ValueError:
            return command
        if not parts or parts[0] != "sudo":
            return command
        if "-n" in parts or "--non-interactive" in parts:
            return command
        parts.insert(1, "-n")
        return shlex.join(parts)

    def _wrap_command_with_pwd(self, command: str, marker: str) -> str:
        if not command.strip():
            return command
        return (
            f"{command}\n"
            "status=$?\n"
            f"printf '\\n{marker}%s' \"$PWD\"\n"
            "exit $status"
        )

    async def run_researchbox(
        self,
        topic: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        await self._check_approval(mode, ToolName.RESEARCHBOX.value, confirmer)
        return await self.researchbox.rb_search_feeds(topic)

    async def run_researchbox_push(
        self,
        feed_url: str,
        topic: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        await self._check_approval(mode, ToolName.RESEARCHBOX_PUSH.value, confirmer)
        return await self.researchbox.rb_push_feeds(feed_url, topic)

    async def run_web_fetch(
        self,
        url: str,
        max_chars: int,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        """Fetch a URL using the human_browser container (a12fdfeaaf78).

        Checks the DB page cache first; falls back to live browser fetch and
        fires a background cache-store on fresh results.

        Returns the same shape as the old aichat-fetch service so all callers
        remain compatible: {"text": ..., "truncated": ..., "char_count": ...}.
        """
        await self._check_approval(mode, ToolName.WEB_FETCH.value, confirmer)
        # --- DB cache fast path ---
        try:
            cached = await self.db.cache_get(url)
            if cached.get("found") and cached.get("content"):
                raw = cached["content"]
                truncated = len(raw) > max_chars
                return {
                    "url": url,
                    "title": cached.get("title", ""),
                    "text": raw[:max_chars],
                    "char_count": min(len(raw), max_chars),
                    "truncated": truncated,
                    "cached": True,
                }
        except Exception:
            pass
        # --- Live fetch ---
        result = await self.browser.navigate(url)
        content = result.get("content", "")
        if content:
            asyncio.create_task(
                self.db.cache_store(url, content, title=result.get("title", "") or None)
            )
        truncated = len(content) > max_chars
        return {
            "url": result.get("url", url),
            "title": result.get("title", ""),
            "text": content[:max_chars],
            "char_count": min(len(content), max_chars),
            "truncated": truncated,
            "cached": False,
        }

    async def run_page_scrape(
        self,
        url: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
        max_scrolls: int = 10,
        wait_ms: int = 500,
        max_chars: int = 16000,
        include_links: bool = False,
    ) -> dict:
        """Navigate, scroll through the full page triggering lazy loads, and return
        the complete rendered text from the final DOM state."""
        await self._check_approval(mode, ToolName.BROWSER.value, confirmer)
        return await self.browser.scrape(
            url=url,
            max_scrolls=max_scrolls,
            wait_ms=wait_ms,
            max_chars=max_chars,
            include_links=include_links,
        )

    async def run_page_images(
        self,
        url: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
        scroll: bool = True,
        max_scrolls: int = 3,
    ) -> dict:
        """Extract all image URLs from a page: src, srcset (highest-res), data-src,
        picture sources, og:image, twitter:image, CSS bg, JSON-LD."""
        await self._check_approval(mode, ToolName.BROWSER.value, confirmer)
        return await self.browser.page_images(url=url, scroll=scroll, max_scrolls=max_scrolls)

    async def run_image_search(
        self,
        query: str,
        count: int,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
        offset: int = 0,
    ) -> list[dict]:
        """Search for images by description; return up to `count` images rendered inline.

        Two-tier strategy with query expansion, domain diversity, parallel fetch,
        and seen-URL deduplication via the memory service:
          Tier 1 — DDG web search (kp=-2) → page_images on top article pages.
          Tier 2 — DDG Images page → decode proxy URLs.
        All candidates collected before scoring; fetched in parallel; up to `count` returned.
        """
        await self._check_approval(mode, ToolName.BROWSER.value, confirmer)
        import asyncio as _asyncio
        import base64 as _b64
        import hashlib as _hl
        import json as _js
        import re as _re
        import httpx as _httpx
        from urllib.parse import (
            unquote as _uq,
            urlparse as _up,
            parse_qs as _pqs,
            quote_plus as _qp,
        )

        count  = max(1, min(count if count else 4, 20))
        offset = max(0, offset)
        qwords = {w for w in query.lower().split() if len(w) > 2}
        _GOOD_D  = ("wikia.nocookie.net", "imgur.com", "redd.it",
                    "prydwen.gg", "fandom.com", "iopwiki.com", "cdn.")
        _SKIP_P  = ("/16px-", "/25px-", "/32px-", "/48px-", "favicon",
                    "logo", "icon", "avatar", "pixel.gif", "button",
                    "ytimg.com", "yt3.ggpht")
        _SKIP_T1 = ("youtube.com", "youtu.be", "vimeo.com", "dailymotion.com",
                    "twitch.tv", "pinterest.com", "instagram.com",
                    "deviantart.com", "artstation.com")
        _UA = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )

        # ── Seen-URL dedup via memory service (TTL 3600 s) ───────────────────
        _qhash        = _hl.md5(query.lower().encode()).hexdigest()[:16]
        _mem_key      = f"imgsr:{_qhash}"
        _seen_urls:   set[str]  = set()
        _seen_hashes: list[str] = []   # intra-call phash dedup
        _url_to_hash: dict[str, str] = {}
        _norm_subject = query.lower().strip()
        try:
            mem_r = await self.memory.recall(key=_mem_key)
            if mem_r.get("found"):
                _seen_urls = set(_js.loads(mem_r["entries"][0]["value"]))
        except Exception:
            pass

        # ── Query expansion ───────────────────────────────────────────────────
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
                m = _re.search(r"/\d+px-[^?]*", no_thumb)
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
            if any(p in url_l for p in _SKIP_P):
                return -999
            s  = sum(2 for w in qwords if w in url_l)
            s += sum(5 for w in qwords if w in alt_l)
            if "pbs.twimg.com" in url_l or "media.twimg.com" in url_l:
                s += 10
            else:
                s += sum(6 for d in _GOOD_D if d in url_l)
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
                h = _up(url).hostname or ""
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

        # ── Collect ALL candidates across all query variants + both tiers ─────
        all_candidates: list[dict] = []
        _ddg_hosts = ("duckduckgo.com", "ddg.gg", "duck.co")

        # Tier 1: DDG web search → page_images on top article pages
        for variant_q in _expand_queries(query):
            try:
                async with _httpx.AsyncClient(timeout=12.0) as hc:
                    sr = await hc.get(
                        f"https://html.duckduckgo.com/html/?q={_qp(variant_q)}&kp=-2",
                        headers={"User-Agent": _UA},
                        follow_redirects=True,
                    )
                raw_enc = _re.findall(r'uddg=(https?%3A[^&"\'>\s]+)', sr.text)
                t1_urls: list[str] = []
                t1_seen: set[str] = set()
                for enc in raw_enc:
                    dec = _uq(enc)
                    if any(d in dec for d in _ddg_hosts):
                        continue
                    if dec not in t1_seen:
                        t1_seen.add(dec)
                        t1_urls.append(dec)
                t1_urls = [u for u in t1_urls if not any(d in u for d in _SKIP_T1)]
                for page_url in t1_urls[:5]:
                    try:
                        pi = await self.run_page_images(
                            page_url, mode, confirmer, scroll=True, max_scrolls=1
                        )
                        page_imgs = pi.get("images", [])
                        if len(page_imgs) < 3:   # auth wall / stub — skip
                            continue
                        all_candidates.extend(page_imgs)
                    except Exception:
                        continue
            except Exception:
                pass

        # Tier 2: DDG image-search page → decode proxy URLs
        try:
            t2_url = f"https://duckduckgo.com/?q={_qp(query)}&iax=images&ia=images&kp=-2"
            pi2 = await self.run_page_images(
                t2_url, mode, confirmer, scroll=True, max_scrolls=3
            )
            for img2 in pi2.get("images", []):
                u2 = img2.get("url", "")
                if "external-content.duckduckgo.com" in u2:
                    try:
                        params2 = _pqs(_up(u2).query)
                        orig2   = _uq(params2.get("u", [""])[0])
                        if orig2.startswith("http"):
                            all_candidates.append({**img2, "url": orig2})
                            continue
                    except Exception:
                        pass
                all_candidates.append(img2)
        except Exception:
            pass

        # Tier 3: Bing Images — different corpus from DDG, different CDNs
        try:
            bing_url = f"https://www.bing.com/images/search?q={_qp(query)}&form=HDRSC2&first=1"
            pi3 = await self.run_page_images(
                bing_url, mode, confirmer, scroll=True, max_scrolls=2
            )
            for img3 in pi3.get("images", []):
                u3 = img3.get("url", "")
                # Bing wraps images in /th?id=... proxy URLs — skip thumbnails
                if "bing.com/th" in u3 or "tse1.mm.bing" in u3:
                    continue
                all_candidates.append(img3)
        except Exception:
            pass

        # ── Score, dedup seen URLs + intra-call dupes, domain-cap, offset ───
        _intra_seen: set[str] = set()
        fresh: list[dict] = []
        for c in all_candidates:
            u = _unwrap_thumb(c.get("url", ""))
            if _score(c) >= 0 and u not in _seen_urls and u not in _intra_seen:
                fresh.append(c)
                _intra_seen.add(u)
        sorted_cands = _apply_domain_cap(
            sorted(fresh, key=_score, reverse=True), max_per_domain=2
        )
        sorted_cands = sorted_cands[offset:]
        fetch_pool   = sorted_cands[:count * 3]   # fetch up to 3× as fallback supply

        if not fetch_pool:
            return [{"type": "text",
                     "text": (f"image_search: no image found for '{query}'.\n"
                              "Try: page_images on a specific URL, or refine the query.")}]

        # ── DB-first: return confirmed images if we have enough cached ─────────
        async def _fetch_render_simple(
            url: str, hc: _httpx.AsyncClient
        ) -> list[dict] | None:
            """Minimal fetch+encode without side-effects — for DB-cache fast path."""
            if not url or not url.startswith("http"):
                return None
            try:
                r0 = await hc.get(
                    url,
                    headers={"User-Agent": _UA, "Accept": "image/webp,image/apng,image/*,*/*;q=0.8"},
                    follow_redirects=True,
                )
                if r0.status_code != 200:
                    return None
                ct0 = r0.headers.get("content-type", "").split(";")[0].strip()
                if not ct0.startswith("image/"):
                    return None
                raw0 = r0.content
                if len(raw0) < 10_240:
                    return None
                b64_0 = _b64.standard_b64encode(raw0).decode("ascii")
                return [
                    {"type": "text",  "text": f"Image: {url}\nQuery: {query}"},
                    {"type": "image", "data": b64_0, "mimeType": ct0},
                ]
            except Exception:
                return None

        try:
            db_data = await self.db.search_images(_norm_subject, limit=count * 2)
            db_imgs = db_data.get("images", [])
            if len(db_imgs) >= count:
                async with _httpx.AsyncClient(timeout=12.0) as hc_db:
                    db_blocks: list[dict] = []
                    for dbi in db_imgs:
                        fb = await _fetch_render_simple(dbi["url"], hc_db)
                        if fb:
                            db_blocks.extend(fb)
                        if len(db_blocks) >= count * 2:
                            break
                    if db_blocks:
                        return db_blocks
        except Exception:
            pass

        # ── Parallel image fetch via shared AsyncClient ───────────────────────
        async def _fetch_render(url: str, hc: _httpx.AsyncClient) -> list[dict] | None:
            """Fetch, resize, phash-dedup, and encode one image URL."""
            if not url or not url.startswith("http"):
                return None
            url  = _unwrap_thumb(url)
            hdrs = {
                "User-Agent": _UA,
                "Accept":     "image/webp,image/apng,image/*,*/*;q=0.8",
                "Referer":    _img_referer(url),
            }
            try:
                r = await hc.get(url, headers=hdrs, follow_redirects=True)
                if r.status_code != 200:
                    return None
                ct = r.headers.get("content-type", "").split(";")[0].strip()
                if not ct.startswith("image/"):
                    return None
                raw = r.content
                if len(raw) < 10_240:
                    return None
                try:
                    from PIL import Image as _PIL
                    import io as _bio
                    with _PIL.open(_bio.BytesIO(raw)) as pil:
                        if pil.width < 150 or pil.height < 150:
                            return None
                        pil = pil.convert("RGB")
                        # dHash intra-call dedup
                        img_hash = _dhash(pil)
                        if img_hash and any(
                            _hamming(img_hash, h) < 8 for h in _seen_hashes
                        ):
                            return None
                        if img_hash:
                            _seen_hashes.append(img_hash)
                            _url_to_hash[url] = img_hash
                        if pil.width > 1280 or pil.height > 1024:
                            pil.thumbnail((1280, 1024), _PIL.LANCZOS)
                        buf = _bio.BytesIO()
                        pil.save(buf, format="JPEG", quality=85)
                        raw, ct = buf.getvalue(), "image/jpeg"
                except Exception:
                    pass
                b64 = _b64.standard_b64encode(raw).decode("ascii")
                return [
                    {"type": "text",  "text": f"Image: {url}\nQuery: {query}"},
                    {"type": "image", "data": b64, "mimeType": ct},
                ]
            except Exception:
                return None

        async with _httpx.AsyncClient(timeout=12.0) as hc:
            fetch_results = await _asyncio.gather(
                *[_fetch_render(c["url"], hc) for c in fetch_pool],
                return_exceptions=True,
            )

        # ── Vision confirm pass (best-effort, parallel) ────────────────────────
        async def _vision_confirm_mgr(
            b64: str, subject: str
        ) -> "tuple[bool, str, float]":
            env_flag = os.environ.get("IMAGE_VISION_CONFIRM", "true").lower()
            if env_flag not in ("1", "true", "yes"):
                return True, "", 0.6
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
            try:
                async with _httpx.AsyncClient(timeout=9.0) as _vc:
                    r2 = await _asyncio.wait_for(
                        _vc.post(f"{_LM_STUDIO_URL}/v1/chat/completions", json=payload),
                        timeout=8.0,
                    )
                text = r2.json()["choices"][0]["message"]["content"].strip()
                desc  = ""
                if "DESCRIPTION:" in text:
                    desc = text.split("DESCRIPTION:")[1].split("|")[0].strip()
                match = "MATCH: YES" in text.upper() or text.upper().strip().endswith("YES")
                return match, desc, 0.9 if match else 0.2
            except Exception:
                return True, "", 0.6   # fail-open

        async def _maybe_confirm(
            blocks: list[dict] | None, url: str, img_hash: str
        ) -> list[dict] | None:
            if not blocks:
                return None
            img_b64 = next(
                (b["data"] for b in blocks if b.get("type") == "image"), ""
            )
            if not img_b64:
                return blocks
            is_match, desc, conf = await _vision_confirm_mgr(img_b64, query)
            if not is_match:
                return None
            # Fire-and-forget: persist confirmed image to DB
            _asyncio.ensure_future(
                self.db.store_image_rich(
                    url, subject=_norm_subject, description=desc,
                    phash=img_hash, quality_score=conf,
                )
            )
            return blocks

        confirmed_results = await _asyncio.gather(
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

        output_blocks:  list[dict] = []
        returned_urls:  list[str]  = []
        for cand, res in zip(fetch_pool, confirmed_results):
            if isinstance(res, Exception) or res is None:
                continue
            output_blocks.extend(res)
            returned_urls.append(_unwrap_thumb(cand["url"]))
            if len(returned_urls) >= count:
                break

        # ── Persist seen URLs for dedup on next call ──────────────────────────
        if returned_urls:
            try:
                merged = list(_seen_urls | set(returned_urls))
                await self.memory.store(_mem_key, _js.dumps(merged), ttl_seconds=3600)
            except Exception:
                pass

        if output_blocks:
            return output_blocks
        return [{"type": "text",
                 "text": (f"image_search: no image found for '{query}'.\n"
                          "Try: page_images on a specific URL, or refine the query.")}]

    async def run_web_search(
        self,
        query: str,
        max_chars: int,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        """Search the web using tiered strategy: browser → httpx → DDG lite API.

        Returns {query, tier, tier_name, url, content[, error]}.
        """
        await self._check_approval(mode, ToolName.WEB_SEARCH.value, confirmer)
        max_chars = max(500, min(max_chars, 16000))
        return await self.search_tool.search(query, max_chars=max_chars)

    async def run_memory_store(
        self,
        key: str,
        value: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
        ttl_seconds: int | None = None,
    ) -> dict:
        await self._check_approval(mode, ToolName.MEMORY_STORE.value, confirmer)
        return await self.memory.store(key, value, ttl_seconds=ttl_seconds)

    async def run_memory_recall(
        self,
        key: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
        pattern: str = "",
    ) -> dict:
        await self._check_approval(mode, ToolName.MEMORY_RECALL.value, confirmer)
        return await self.memory.recall(key, pattern=pattern)

    # ------------------------------------------------------------------
    # Database tools (PostgreSQL storage / web cache)
    # ------------------------------------------------------------------

    async def run_db_store_article(
        self,
        url: str,
        title: str,
        content: str,
        topic: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        await self._check_approval(mode, ToolName.DB_STORE_ARTICLE.value, confirmer)
        return await self.db.store_article(url, title=title, content=content, topic=topic)

    async def run_db_search(
        self,
        topic: str,
        q: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
        limit: int = 20,
        offset: int = 0,
        summary_only: bool = False,
    ) -> dict:
        await self._check_approval(mode, ToolName.DB_SEARCH.value, confirmer)
        return await self.db.search_articles(
            topic=topic or None, q=q or None,
            limit=limit, offset=offset, summary_only=summary_only,
        )

    async def run_db_cache_store(
        self,
        url: str,
        content: str,
        title: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        await self._check_approval(mode, ToolName.DB_CACHE_STORE.value, confirmer)
        return await self.db.cache_store(url, content, title=title or None)

    async def run_db_cache_get(
        self,
        url: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        await self._check_approval(mode, ToolName.DB_CACHE_GET.value, confirmer)
        return await self.db.cache_get(url)

    async def run_db_store_image(
        self,
        url: str,
        host_path: str,
        alt_text: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        await self._check_approval(mode, ToolName.DB_STORE_IMAGE.value, confirmer)
        return await self.db.store_image(url=url, host_path=host_path or None, alt_text=alt_text or None)

    async def run_db_list_images(
        self,
        limit: int,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        await self._check_approval(mode, ToolName.DB_LIST_IMAGES.value, confirmer)
        return await self.db.list_images(limit=limit)

    async def run_get_errors(
        self,
        limit: int,
        service: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        await self._check_approval(mode, ToolName.GET_ERRORS.value, confirmer)
        return await self.db.get_errors(limit=limit, service=service or None)

    # ------------------------------------------------------------------
    # LM Studio / code interpreter tools
    # ------------------------------------------------------------------

    async def run_tts(
        self,
        text: str,
        voice: str,
        speed: float,
        fmt: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        """Text-to-speech via LM Studio /v1/audio/speech. Returns audio bytes."""
        await self._check_approval(mode, ToolName.TTS.value, confirmer)
        audio = await self.lm.tts(text, voice=voice or "alloy", speed=speed or 1.0, fmt=fmt or "mp3")
        return {"bytes": len(audio), "audio": audio}

    async def run_embed_store(
        self,
        key: str,
        content: str,
        topic: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        """Embed text via LM Studio and store in DB. Returns stored key."""
        await self._check_approval(mode, ToolName.EMBED_STORE.value, confirmer)
        vecs = await self.lm.embed([content])
        if not vecs:
            return {"status": "error", "detail": "LM Studio returned no embeddings"}
        return await self.db.store_embedding(key=key, content=content, embedding=vecs[0], topic=topic or "")

    async def run_embed_search(
        self,
        query: str,
        limit: int,
        topic: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        """Embed query text and search DB for semantically similar documents."""
        await self._check_approval(mode, ToolName.EMBED_SEARCH.value, confirmer)
        vecs = await self.lm.embed([query])
        if not vecs:
            return {"results": [], "detail": "LM Studio returned no embeddings"}
        return await self.db.search_by_embedding(embedding=vecs[0], limit=limit or 5, topic=topic or "")

    async def run_code_run(
        self,
        code: str,
        packages: list[str] | None,
        timeout: int,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        """Run Python code in a sandboxed subprocess. Returns stdout/stderr/exit_code."""
        await self._check_approval(mode, ToolName.CODE_RUN.value, confirmer)
        return await self.code.run(code, packages=packages or None, timeout=timeout or None)

    async def run_smart_summarize(
        self,
        content: str,
        style: str,
        max_words: int | None,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> str:
        """Summarize text using LM Studio chat completions."""
        await self._check_approval(mode, ToolName.SMART_SUMMARIZE.value, confirmer)
        return await self.lm.summarize(content, style=style or "brief", max_words=max_words or None)

    async def run_image_caption(
        self,
        b64: str,
        detail_level: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> str:
        """Describe an image using LM Studio vision model."""
        await self._check_approval(mode, ToolName.IMAGE_CAPTION.value, confirmer)
        return await self.lm.caption(b64, detail_level=detail_level or "detailed")

    async def run_structured_extract(
        self,
        content: str,
        schema_json: str,
        instructions: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        """Extract structured JSON from text using LM Studio json_object mode."""
        await self._check_approval(mode, ToolName.STRUCTURED_EXTRACT.value, confirmer)
        return await self.lm.extract(content, schema_json=schema_json or "", instructions=instructions or "")

    async def run_conv_search_history(
        self,
        query: str,
        limit: int,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> list[dict]:
        """Search past conversation turns by semantic similarity."""
        await self._check_approval(mode, ToolName.CONV_SEARCH_HISTORY.value, confirmer)
        vecs = await self.lm.embed([query])
        if not vecs:
            return []
        return await self.conv.search_turns(vecs[0], limit=limit or 5)

    async def run_fetch_image(
        self,
        url: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        """Download an image URL and save it to the workspace; returns host_path + metadata."""
        import httpx as _httpx
        from datetime import datetime as _dt
        import os as _os

        await self._check_approval(mode, ToolName.FETCH_IMAGE.value, confirmer)

        # Derive a clean filename from the URL
        raw_name = url.split("?")[0].split("/")[-1] or "image"
        if "." not in raw_name:
            raw_name += ".jpg"
        ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        filename = f"img_{ts}_{raw_name}"
        host_path = f"/docker/human_browser/workspace/{filename}"

        last_exc: Exception | None = None
        content_type = "image/jpeg"
        data = b""
        for attempt in range(2):
            try:
                async with _httpx.AsyncClient(timeout=30, follow_redirects=True) as c:
                    r = await c.get(url, headers=_IMG_FETCH_HEADERS)
                    if r.status_code == 429 and attempt == 0:
                        retry_after = min(int(r.headers.get("retry-after", "15")), 30)
                        await asyncio.sleep(retry_after)
                        continue
                    r.raise_for_status()
                    content_type = r.headers.get("content-type", "image/jpeg").split(";")[0].strip()
                    data = r.content
                break
            except Exception as exc:
                last_exc = exc
                if attempt == 0:
                    await asyncio.sleep(3)
                    continue
        else:
            return {"url": url, "error": str(last_exc)}

        _os.makedirs(_os.path.dirname(host_path), exist_ok=True)
        with open(host_path, "wb") as fh:
            fh.write(data)

        try:
            await self.db.store_image(url=url, host_path=host_path,
                                      alt_text=f"Image from {url}")
        except Exception:
            pass

        return {
            "url": url,
            "host_path": host_path,
            "content_type": content_type,
            "size": len(data),
        }

    async def _fetch_image_from_urls(
        self,
        image_urls: list[str],
        page_url: str,
        query: str,
        httpx_module,
    ) -> dict | None:
        """Try each image URL in order; download first success and save to workspace."""
        import os as _os
        from datetime import datetime as _dt

        for img_url in image_urls[:3]:
            try:
                async with httpx_module.AsyncClient(timeout=15, follow_redirects=True) as c:
                    r = await c.get(img_url, headers=_IMG_FETCH_HEADERS)
                    r.raise_for_status()
                    ts = _dt.now().strftime("%Y%m%d_%H%M%S")
                    ext = img_url.split("?")[0].rsplit(".", 1)[-1][:5] or "jpg"
                    filename = f"img_{ts}.{ext}"
                    host_path = f"/docker/human_browser/workspace/{filename}"
                    _os.makedirs(_os.path.dirname(host_path), exist_ok=True)
                    with open(host_path, "wb") as fh:
                        fh.write(r.content)
                    try:
                        alt = f"Search: '{query}' — image from {page_url}"
                        await self.db.store_image(url=img_url, host_path=host_path, alt_text=alt)
                    except Exception:
                        pass
                    return {
                        "url": page_url,
                        "host_path": host_path,
                        "image_url": img_url,
                        "content_type": r.headers.get("content-type", "image/jpeg"),
                    }
            except Exception:
                continue
        return None

    async def run_screenshot_search(
        self,
        query: str,
        max_results: int,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        """Search DDG for query, screenshot top result pages, save to DB, return list."""
        import httpx as _httpx
        import re as _re

        await self._check_approval(mode, ToolName.SCREENSHOT_SEARCH.value, confirmer)
        max_results = max(1, min(max_results, 5))

        # Search DuckDuckGo HTML for result URLs (full browser headers to avoid rate-limiting)
        html = ""
        try:
            async with _httpx.AsyncClient(timeout=20, follow_redirects=True) as c:
                r = await c.get(
                    f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}",
                    headers=_FETCH_HEADERS,
                )
            html = r.text
        except Exception:
            pass

        # Tier 1: DDG uddg= redirect params (stable HTML endpoint format)
        from urllib.parse import unquote as _unquote
        raw = _re.findall(r'uddg=(https?%3A[^&"\'>\s]+)', html)
        seen: set[str] = set()
        urls: list[str] = []
        for encoded in raw:
            decoded = _unquote(encoded)
            if decoded not in seen:
                seen.add(decoded)
                urls.append(decoded)
        urls = urls[:max_results]

        # Tier 2: direct href links (fallback if DDG changed format or rate-limited)
        if not urls and html:
            href_raw = _re.findall(r'href=["\']?(https?://[^"\'>\s]+)', html)
            urls = list(dict.fromkeys(
                u for u in href_raw
                if not any(d in u for d in ('duckduckgo.com', 'ddg.gg', 'duck.co'))
            ))[:max_results]

        # Tier 3: browser search + DOM eval (Chromium w/ anti-detection, most reliable)
        if not urls:
            try:
                await self.browser.search(query)
                ev_result = await self.browser.eval_js(r"""
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
                """)
                extracted = json.loads(ev_result.get("result", "[]"))
                urls = [u for u in extracted if u][:max_results]
            except Exception:
                pass

        if not urls:
            return {"error": "No URLs found in search results.", "query": query, "screenshots": []}

        screenshots: list[dict] = []
        for url in urls:
            try:
                result = await self.browser.screenshot(url)
                host_path = result.get("host_path", "")
                if host_path and not result.get("error"):
                    try:
                        page_url = result.get("url") or url
                        alt = f"Search: '{query}' — {result.get('title', page_url)}"
                        await self.db.store_image(url=page_url, host_path=host_path, alt_text=alt)
                    except Exception:
                        pass
                    screenshots.append(result)
                elif result.get("image_urls"):
                    # Screenshot failed but browser loaded the page — try fetching an image directly
                    fallback = await self._fetch_image_from_urls(
                        result["image_urls"], url, query, _httpx
                    )
                    screenshots.append(fallback or result)
                else:
                    screenshots.append(result)
            except Exception as exc:
                screenshots.append({"url": url, "error": str(exc)})

        return {"query": query, "urls": urls, "screenshots": screenshots}

    # ------------------------------------------------------------------
    # Toolkit meta-tools (create / list / delete / call custom tools)
    # ------------------------------------------------------------------

    async def run_create_tool(
        self,
        tool_name: str,
        description: str,
        parameters: dict,
        code: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        await self._check_approval(mode, ToolName.CREATE_TOOL.value, confirmer)
        result = await self.toolkit.register_tool(tool_name, description, parameters, code)
        await self.refresh_custom_tools()
        return result

    async def run_list_custom_tools(
        self,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> list[dict]:
        await self._check_approval(mode, ToolName.LIST_CUSTOM_TOOLS.value, confirmer)
        tools = await self.toolkit.list_tools()
        await self.refresh_custom_tools()
        return tools

    async def run_delete_custom_tool(
        self,
        tool_name: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        await self._check_approval(mode, ToolName.DELETE_CUSTOM_TOOL.value, confirmer)
        result = await self.toolkit.delete_tool(tool_name)
        await self.refresh_custom_tools()
        return result

    async def run_custom_tool(
        self,
        tool_name: str,
        params: dict,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        await self._check_approval(mode, tool_name, confirmer)
        return await self.toolkit.call_tool(tool_name, params)

    async def run_browser(
        self,
        action: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
        url: str | None = None,
        selector: str | None = None,
        value: str | None = None,
        code: str | None = None,
        find_text: str | None = None,
        find_image: str | None = None,
        pad: int = 20,
        image_urls: list[str] | None = None,
        filter_query: str | None = None,
        image_prefix: str = "image",
        max_images: int = 20,
    ) -> dict:
        await self._check_approval(mode, ToolName.BROWSER.value, confirmer)
        try:
            if action == "navigate":
                if not url:
                    return {"error": "url is required for navigate"}
                return await self.browser.navigate(url)
            if action == "screenshot":
                result = await self.browser.screenshot(url, find_text=find_text,
                                                        find_image=find_image)
                # Auto-persist screenshot metadata to the image database
                host_path = result.get("host_path", "")
                if host_path and not result.get("error"):
                    try:
                        page_url = result.get("url") or url or host_path
                        alt = f"Screenshot of {result.get('title', page_url)}"
                        await self.db.store_image(
                            url=page_url,
                            host_path=host_path,
                            alt_text=alt,
                        )
                    except Exception:
                        pass  # Never fail the screenshot because DB is down
                return result
            if action == "click":
                if not selector:
                    return {"error": "selector is required for click"}
                return await self.browser.click(selector)
            if action == "fill":
                if not selector or value is None:
                    return {"error": "selector and value are required for fill"}
                return await self.browser.fill(selector, value)
            if action == "read":
                return await self.browser.read()
            if action == "eval":
                if not code:
                    return {"error": "code is required for eval"}
                return await self.browser.eval_js(code)
            if action == "screenshot_element":
                if not selector:
                    return {"error": "selector is required for screenshot_element"}
                return await self.browser.screenshot_element(selector, pad=pad)
            if action == "list_images_detail":
                return await self.browser.list_images()
            if action == "save_images":
                if not image_urls:
                    return {"error": "image_urls required for save_images"}
                return await self.browser.save_images(
                    image_urls, prefix=image_prefix, max_images=max_images
                )
            if action == "download_page_images":
                return await self.browser.download_page_images(
                    filter_query=filter_query, max_images=max_images, prefix=image_prefix
                )
            return {
                "error": (
                    f"Unknown action '{action}'. "
                    "Valid: navigate, read, screenshot, click, fill, eval, "
                    "screenshot_element, list_images_detail, save_images, download_page_images"
                )
            }
        except Exception as exc:
            return {"error": str(exc)}

    def active_sessions(self) -> list[str]:
        return [f"shell:{sid}" for sid in self.shell.sessions]

    # ------------------------------------------------------------------
    # Tool definitions (sent to LLM each turn)
    # ------------------------------------------------------------------

    def tool_definitions(self, shell_enabled: bool) -> list[dict[str, object]]:
        tools: list[dict[str, object]] = [
            # ----------------------------------------------------------
            # Web search — tiered: browser (human-like) → httpx → DDG lite
            # ----------------------------------------------------------
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": (
                        "Search the web for a query using a tiered strategy. "
                        "Tier 1 (preferred): opens a real Chromium browser (human_browser / a12fdfeaaf78), "
                        "navigates to DuckDuckGo, types the query and presses Enter — exactly as a human would. "
                        "If the browser is unavailable or times out, falls back to a direct HTTP fetch "
                        "(Tier 2), then to the DuckDuckGo lite API (Tier 3). "
                        "Returns the search results page text and which tier was used."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query.",
                            },
                            "max_chars": {
                                "type": "integer",
                                "description": "Maximum characters to return (default 4000, max 16000).",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            # ----------------------------------------------------------
            # Web browsing — all web operations go through human_browser
            # ----------------------------------------------------------
            {
                "type": "function",
                "function": {
                    "name": "web_fetch",
                    "description": (
                        "Fetch a web page using the real Chromium browser "
                        "(human_browser container, ID a12fdfeaaf78) and return its "
                        "readable text content. Use this to read documentation, "
                        "articles, GitHub files, or any URL."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The full URL to fetch (http/https).",
                            },
                            "max_chars": {
                                "type": "integer",
                                "description": "Maximum characters to return (default 4000, max 16000).",
                            },
                        },
                        "required": ["url"],
                    },
                },
            },
            # ----------------------------------------------------------
            # RSS/feed discovery (researchbox)
            # ----------------------------------------------------------
            {
                "type": "function",
                "function": {
                    "name": "researchbox_search",
                    "description": "Search for RSS feed sources for a topic via the docker researchbox service.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "Topic to search for feeds.",
                            }
                        },
                        "required": ["topic"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "researchbox_push",
                    "description": (
                        "Fetch an RSS feed and store its articles in the PostgreSQL "
                        "database for later retrieval and comparison."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "feed_url": {
                                "type": "string",
                                "description": "RSS feed URL to ingest.",
                            },
                            "topic": {
                                "type": "string",
                                "description": "Topic label to store articles under.",
                            },
                        },
                        "required": ["feed_url", "topic"],
                    },
                },
            },
            # ----------------------------------------------------------
            # PostgreSQL storage / cache
            # ----------------------------------------------------------
            {
                "type": "function",
                "function": {
                    "name": "db_store_article",
                    "description": (
                        "Store an article (URL, title, content, topic) in the "
                        "PostgreSQL database for long-term retrieval and comparison "
                        "with future results."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url":     {"type": "string", "description": "Article URL."},
                            "title":   {"type": "string", "description": "Article title."},
                            "content": {"type": "string", "description": "Article text content."},
                            "topic":   {"type": "string", "description": "Topic label."},
                        },
                        "required": ["url"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "db_search",
                    "description": (
                        "Search previously stored articles in the PostgreSQL database. "
                        "Use to compare new web results against what was stored before."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "Filter by topic (optional).",
                            },
                            "q": {
                                "type": "string",
                                "description": "Full-text search query (optional).",
                            },
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "db_cache_store",
                    "description": (
                        "Cache a web page's content in PostgreSQL so it can be retrieved "
                        "later without re-fetching."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url":     {"type": "string", "description": "Page URL."},
                            "content": {"type": "string", "description": "Page content to cache."},
                            "title":   {"type": "string", "description": "Page title (optional)."},
                        },
                        "required": ["url", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "db_cache_get",
                    "description": (
                        "Retrieve a previously cached web page from PostgreSQL. "
                        "Returns {found: false} if the URL has not been cached yet."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "Page URL to look up."},
                        },
                        "required": ["url"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "db_store_image",
                    "description": (
                        "Save an image or screenshot to the PostgreSQL image registry. "
                        "Use after taking a screenshot or downloading an image to record it "
                        "permanently with its source URL and description."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url":       {"type": "string", "description": "Source URL the image was captured from."},
                            "host_path": {"type": "string", "description": "File path on the host (e.g. /docker/human_browser/workspace/screenshot.png)."},
                            "alt_text":  {"type": "string", "description": "Description or caption for the image."},
                        },
                        "required": ["url"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "db_list_images",
                    "description": (
                        "List screenshots and images previously saved to the PostgreSQL database. "
                        "Returns host file paths so the user can open them."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of images to return (default 20).",
                            },
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_image",
                    "description": (
                        "Download an image directly from a URL (jpg, png, gif, webp, etc.) "
                        "and save it to disk. Returns the file path so the user can open it. "
                        "Use this when the user provides a direct image URL and wants to view "
                        "or save it, rather than screenshotting a whole web page."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "Direct URL to the image file (http/https).",
                            },
                        },
                        "required": ["url"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "screenshot_search",
                    "description": (
                        "Search the web for a topic, screenshot the top result pages, "
                        "and tell the user exactly where each screenshot file was saved. "
                        "Use this when the user says 'find a picture of X', 'show me X', "
                        "'screenshot search results for X', or wants to visually browse a topic. "
                        "Best-effort: returns whatever screenshots succeed even if some fail."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The topic or query to search and screenshot.",
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "How many result pages to screenshot (default 3, max 5).",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            # ----------------------------------------------------------
            # Memory
            # ----------------------------------------------------------
            {
                "type": "function",
                "function": {
                    "name": "memory_store",
                    "description": (
                        "Store a note or fact in persistent memory for recall later. "
                        "Storing the same key overwrites the previous value."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "key":   {"type": "string", "description": "Short label for this memory entry."},
                            "value": {"type": "string", "description": "Content to remember."},
                        },
                        "required": ["key", "value"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "memory_recall",
                    "description": (
                        "Retrieve notes from persistent memory. "
                        "Provide a key to look up a specific entry, or leave empty to list all stored keys."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "key": {
                                "type": "string",
                                "description": "Key to look up (omit or empty string to list all).",
                            }
                        },
                        "required": [],
                    },
                },
            },
            # ----------------------------------------------------------
            # Toolkit meta-tools
            # ----------------------------------------------------------
            {
                "type": "function",
                "function": {
                    "name": "create_tool",
                    "description": (
                        "Create a new persistent custom tool that runs in an isolated Docker container. "
                        "The tool is saved to disk and immediately available for use — in this session "
                        "and all future sessions. Use this whenever you need a capability not covered "
                        "by the built-in tools. You can make HTTP calls (httpx), process data, parse "
                        "HTML, call APIs, run shell commands (subprocess/asyncio.create_subprocess_exec), "
                        "and read files from the user's repos at /data/repos. Tools persist across restarts."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tool_name": {
                                "type": "string",
                                "description": "Snake_case identifier (e.g. 'search_wikipedia', 'get_stock_price').",
                            },
                            "description": {
                                "type": "string",
                                "description": "What the tool does — shown to you when deciding which tool to call.",
                            },
                            "parameters_schema": {
                                "type": "object",
                                "description": (
                                    "JSON Schema object for the tool's inputs. Example: "
                                    '{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}'
                                ),
                            },
                            "code": {
                                "type": "string",
                                "description": (
                                    "Python implementation — the body of `async def run(**kwargs) -> str:`. "
                                    "Available: asyncio, json, re, os, math, datetime, pathlib, shlex, subprocess, httpx. "
                                    "User git repos are at /data/repos (Path('/data/repos/reponame')). "
                                    "Run shell commands with asyncio.create_subprocess_exec or subprocess.run. "
                                    "Access parameters via kwargs. Must return a string."
                                ),
                            },
                        },
                        "required": ["tool_name", "description", "parameters_schema", "code"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_custom_tools",
                    "description": "List all custom tools you have created, with their names, descriptions, and parameter schemas.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_custom_tool",
                    "description": "Permanently delete a custom tool you previously created.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tool_name": {
                                "type": "string",
                                "description": "Name of the tool to delete.",
                            }
                        },
                        "required": ["tool_name"],
                    },
                },
            },
        ]

        # Browser — human_browser container (ID a12fdfeaaf78)
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "browser",
                    "description": (
                        "Control a real Chromium browser running in the human_browser "
                        "Docker container (ID a12fdfeaaf78) via Playwright. "
                        "This is the default for ALL web-related operations. "
                        "The browser keeps its state between calls (same session), so you "
                        "can navigate to a page, then click a button, then read the result. "
                        "Actions: "
                        "navigate — go to a URL and return page title + text; "
                        "read — return the current page title + text without navigating; "
                        "screenshot — capture the page as a PNG saved to /workspace (returns file path); "
                        "click — click a CSS selector; "
                        "fill — type a value into a CSS selector input; "
                        "eval — run a JavaScript expression and return its result."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["navigate", "read", "screenshot", "click", "fill", "eval"],
                                "description": "Which browser action to perform.",
                            },
                            "url": {
                                "type": "string",
                                "description": "URL to navigate to (navigate / screenshot).",
                            },
                            "selector": {
                                "type": "string",
                                "description": "CSS selector for click / fill.",
                            },
                            "value": {
                                "type": "string",
                                "description": "Text to type into the element (fill only).",
                            },
                            "code": {
                                "type": "string",
                                "description": "JavaScript expression to evaluate (eval only).",
                            },
                            "find_text": {
                                "type": "string",
                                "description": (
                                    "Optional, screenshot only. A word or phrase to search for "
                                    "on the page — the screenshot will be zoomed/clipped to show "
                                    "just the region containing this text."
                                ),
                            },
                        },
                        "required": ["action"],
                    },
                },
            }
        )

        if shell_enabled:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "shell_exec",
                        "description": "Run a shell command on the host machine.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "command": {
                                    "type": "string",
                                    "description": "Shell command to execute.",
                                }
                            },
                            "required": ["command"],
                        },
                    },
                }
            )

        # Append any currently registered custom tools so the LLM can call them
        for name, meta in self._custom_tools.items():
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": meta.get("description", "Custom tool."),
                        "parameters": meta.get(
                            "parameters",
                            {"type": "object", "properties": {}, "required": []},
                        ),
                    },
                }
            )

        return tools
