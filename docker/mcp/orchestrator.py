"""First-class orchestrator for intent classification, tool routing, and concurrency.

The orchestrator sits between the MCP transport layer and tool handlers.
It classifies intents, manages bounded concurrency, tracks resource pressure,
and provides progress streaming to Dartboard clients.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable

log = logging.getLogger("aichat-mcp.orchestrator")


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

class TaskIntent(str, Enum):
    """High-level intent categories for tool routing decisions."""
    INFO_RETRIEVAL = "info_retrieval"       # web_search, web_fetch, news, wiki
    RESEARCH       = "research"             # deep_research, realtime, arxiv
    DOCUMENT       = "document_analysis"    # ingest, ocr, pdf_*
    IMAGE_CV       = "image_cv"             # detect, clip, face, enhance
    CODE_SYSTEM    = "code_system"          # code_run, jupyter, shell
    MEMORY         = "memory"               # store, recall, compact
    MIXED          = "mixed"                # orchestrate, plan_task


# Tool name → intent mapping
_INTENT_MAP: dict[str, TaskIntent] = {
    # Info retrieval
    "web_search": TaskIntent.INFO_RETRIEVAL,
    "web_fetch": TaskIntent.INFO_RETRIEVAL,
    "news_search": TaskIntent.INFO_RETRIEVAL,
    "wikipedia": TaskIntent.INFO_RETRIEVAL,
    "youtube_transcript": TaskIntent.INFO_RETRIEVAL,
    "extract_article": TaskIntent.INFO_RETRIEVAL,
    # Research
    "deep_research": TaskIntent.RESEARCH,
    "realtime": TaskIntent.RESEARCH,
    "arxiv_search": TaskIntent.RESEARCH,
    "researchbox_search": TaskIntent.RESEARCH,
    "researchbox_push": TaskIntent.RESEARCH,
    # Document
    "docs_ingest": TaskIntent.DOCUMENT,
    "docs_extract_tables": TaskIntent.DOCUMENT,
    "ocr_image": TaskIntent.DOCUMENT,
    "ocr_pdf": TaskIntent.DOCUMENT,
    "pdf_read": TaskIntent.DOCUMENT,
    "pdf_edit": TaskIntent.DOCUMENT,
    "pdf_fill_form": TaskIntent.DOCUMENT,
    "pdf_merge": TaskIntent.DOCUMENT,
    "pdf_split": TaskIntent.DOCUMENT,
    # Image / CV
    "fetch_image": TaskIntent.IMAGE_CV,
    "image_search": TaskIntent.IMAGE_CV,
    "image_generate": TaskIntent.IMAGE_CV,
    "image_edit": TaskIntent.IMAGE_CV,
    "image_crop": TaskIntent.IMAGE_CV,
    "image_zoom": TaskIntent.IMAGE_CV,
    "image_enhance": TaskIntent.IMAGE_CV,
    "image_scan": TaskIntent.IMAGE_CV,
    "image_stitch": TaskIntent.IMAGE_CV,
    "image_diff": TaskIntent.IMAGE_CV,
    "image_annotate": TaskIntent.IMAGE_CV,
    "image_caption": TaskIntent.IMAGE_CV,
    "image_upscale": TaskIntent.IMAGE_CV,
    "image_remix": TaskIntent.IMAGE_CV,
    "face_recognize": TaskIntent.IMAGE_CV,
    "embed_search": TaskIntent.IMAGE_CV,
    "detect_objects": TaskIntent.IMAGE_CV,
    "detect_humans": TaskIntent.IMAGE_CV,
    # Code / system
    "code_run": TaskIntent.CODE_SYSTEM,
    "run_javascript": TaskIntent.CODE_SYSTEM,
    "jupyter_exec": TaskIntent.CODE_SYSTEM,
    "create_tool": TaskIntent.CODE_SYSTEM,
    "list_custom_tools": TaskIntent.CODE_SYSTEM,
    "delete_custom_tool": TaskIntent.CODE_SYSTEM,
    "call_custom_tool": TaskIntent.CODE_SYSTEM,
    # Memory
    "memory_store": TaskIntent.MEMORY,
    "memory_recall": TaskIntent.MEMORY,
    "memory_compact": TaskIntent.MEMORY,
    "graph_add_node": TaskIntent.MEMORY,
    "graph_add_edge": TaskIntent.MEMORY,
    "graph_query": TaskIntent.MEMORY,
    "graph_path": TaskIntent.MEMORY,
    "graph_search": TaskIntent.MEMORY,
    "vector_store": TaskIntent.MEMORY,
    "vector_search": TaskIntent.MEMORY,
    "vector_delete": TaskIntent.MEMORY,
    "vector_collections": TaskIntent.MEMORY,
    "embed_store": TaskIntent.MEMORY,
    "db_store_article": TaskIntent.MEMORY,
    "db_search": TaskIntent.MEMORY,
    "db_cache_store": TaskIntent.MEMORY,
    "db_cache_get": TaskIntent.MEMORY,
    "db_store_image": TaskIntent.MEMORY,
    "db_list_images": TaskIntent.MEMORY,
    "get_errors": TaskIntent.MEMORY,
    # Mixed
    "orchestrate": TaskIntent.MIXED,
    "plan_task": TaskIntent.MIXED,
}


# ---------------------------------------------------------------------------
# Concurrency limits per intent type
# ---------------------------------------------------------------------------

_CONCURRENCY_LIMITS: dict[TaskIntent, int] = {
    TaskIntent.INFO_RETRIEVAL: 4,
    TaskIntent.RESEARCH: 3,
    TaskIntent.DOCUMENT: 2,
    TaskIntent.IMAGE_CV: 1,  # GPU-bound
    TaskIntent.CODE_SYSTEM: 1,  # isolation
    TaskIntent.MEMORY: 8,  # lightweight DB ops
    TaskIntent.MIXED: 2,
}


# ---------------------------------------------------------------------------
# Resource state tracking
# ---------------------------------------------------------------------------

@dataclass
class ResourceState:
    """Current resource pressure metrics."""
    active_jobs: int = 0
    active_gpu_tasks: int = 0
    active_cpu_tasks: int = 0
    queue_depth: int = 0
    last_check: float = 0.0

    @property
    def gpu_pressure(self) -> bool:
        return self.active_gpu_tasks >= 2

    @property
    def cpu_pressure(self) -> bool:
        return self.active_cpu_tasks >= 4

    @property
    def under_pressure(self) -> bool:
        return self.gpu_pressure or self.cpu_pressure


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """First-class orchestrator for intent classification, routing, and concurrency."""

    def __init__(self) -> None:
        self._semaphores: dict[TaskIntent, asyncio.Semaphore] = {
            intent: asyncio.Semaphore(limit)
            for intent, limit in _CONCURRENCY_LIMITS.items()
        }
        self._resources = ResourceState()
        self._progress_callbacks: list[Callable] = []

    def classify_intent(self, tool_name: str) -> TaskIntent:
        """Classify a tool name into a TaskIntent category."""
        return _INTENT_MAP.get(tool_name, TaskIntent.MIXED)

    def resource_state(self) -> ResourceState:
        """Return current resource pressure state."""
        return self._resources

    def register_progress_callback(self, cb: Callable) -> None:
        """Register a callback for progress streaming to clients."""
        self._progress_callbacks.append(cb)

    async def notify_progress(
        self,
        tool_name: str,
        status: str,
        detail: str = "",
        percent: int = 0,
    ) -> None:
        """Send progress notification to all registered callbacks."""
        for cb in self._progress_callbacks:
            try:
                await cb(tool_name=tool_name, status=status,
                         detail=detail, percent=percent)
            except Exception:
                pass

    async def execute(
        self,
        tool_name: str,
        handler: Callable[..., Awaitable[list[dict]]],
        *args: Any,
        **kwargs: Any,
    ) -> list[dict]:
        """Execute a tool handler with bounded concurrency and resource governance.

        Acquires the intent-specific semaphore before running the handler.
        Tracks resource usage for GPU/CPU-bound tasks.
        """
        intent = self.classify_intent(tool_name)
        sem = self._semaphores.get(intent, self._semaphores[TaskIntent.MIXED])

        is_gpu = intent == TaskIntent.IMAGE_CV
        is_cpu = intent == TaskIntent.DOCUMENT

        await self.notify_progress(tool_name, "queued", f"Waiting for {intent.value} slot")

        async with sem:
            if is_gpu:
                self._resources.active_gpu_tasks += 1
            if is_cpu:
                self._resources.active_cpu_tasks += 1
            self._resources.active_jobs += 1

            try:
                await self.notify_progress(tool_name, "running", f"Executing {tool_name}")
                result = await handler(*args, **kwargs)
                await self.notify_progress(tool_name, "completed",
                                           f"{tool_name} completed", percent=100)
                return result
            except Exception as exc:
                await self.notify_progress(tool_name, "failed", str(exc)[:200])
                raise
            finally:
                self._resources.active_jobs -= 1
                if is_gpu:
                    self._resources.active_gpu_tasks -= 1
                if is_cpu:
                    self._resources.active_cpu_tasks -= 1

    async def execute_batch(
        self,
        tasks: list[tuple[str, Callable, tuple, dict]],
        max_concurrent: int = 4,
    ) -> list[list[dict]]:
        """Execute multiple tool calls with bounded concurrency.

        Each task is (tool_name, handler, args, kwargs).
        Returns results in the same order as input tasks.
        """
        sem = asyncio.Semaphore(max_concurrent)
        results: list[list[dict] | None] = [None] * len(tasks)

        async def _run(idx: int, name: str, handler: Callable, args: tuple, kwargs: dict) -> None:
            async with sem:
                try:
                    results[idx] = await self.execute(name, handler, *args, **kwargs)
                except Exception as exc:
                    from helpers import text_block
                    results[idx] = text_block(f"{name} failed: {exc}")

        await asyncio.gather(*[
            _run(i, name, handler, args, kwargs)
            for i, (name, handler, args, kwargs) in enumerate(tasks)
        ])

        return [r or [] for r in results]
