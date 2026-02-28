"""Parallel chain-of-thought reasoning engine.

ThinkingTool fans out N independent reasoning chains in parallel via LM Studio,
scores them heuristically, and synthesises a final answer from the best chain.
All public methods are fail-open: they return sentinel values on error.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .lm_studio import LMStudioTool


@dataclass
class ThinkResult:
    """Result of a parallel thinking run."""

    reasoning: str       # best reasoning chain (highest score)
    answer: str          # synthesised final answer
    paths_tried: int     # number of chains that returned non-empty
    best_score: float    # heuristic score of winning chain (0–1)
    duration_ms: int     # wall-clock time in ms
    paths_all: list[str] = field(default_factory=list)  # all chains for inspection


class ThinkingTool:
    """Parallel chain-of-thought reasoning via LM Studio chat completions.

    Usage::

        tool = ThinkingTool(lm=lm_studio_instance)
        result = await tool.think_and_answer("What is 17 * 23?", n_paths=3)
        print(result.answer)
    """

    _REASONING_MARKERS = (
        "therefore", "thus", "because", "however", "step", "first", "second",
        "finally", "conclusion", "reason", "hence", "result", "so ",
    )

    def __init__(self, lm: "LMStudioTool", model: str = "") -> None:
        self.lm = lm
        self.model = model  # empty = use lm's own model

    # ------------------------------------------------------------------
    # Heuristic scoring
    # ------------------------------------------------------------------

    def score_chain(self, chain: str) -> float:
        """Heuristic score for a reasoning chain in [0.0, 1.0].

        Combines: length (prefer 300-word chains), reasoning markers,
        and conclusion signal in the last 100 chars.
        """
        if not chain.strip():
            return 0.0
        words = len(chain.split())
        length_score = min(1.0, words / 300)
        lower = chain.lower()
        marker_count = sum(1 for m in self._REASONING_MARKERS if m in lower)
        marker_score = min(1.0, marker_count / 5)
        ending = lower[-100:]
        conclusion_score = 0.2 if any(
            w in ending for w in ("therefore", "thus", "conclusion", "answer", "so")
        ) else 0.0
        return 0.5 * length_score + 0.3 * marker_score + 0.2 * conclusion_score

    # ------------------------------------------------------------------
    # Step 1: fan-out N parallel reasoning chains
    # ------------------------------------------------------------------

    async def think(
        self,
        prompt: str,
        n_paths: int = 3,
        temperature: float = 0.8,
    ) -> list[str]:
        """Fan out *n_paths* independent reasoning chains in parallel.

        Returns a list of non-empty chain strings.  May be shorter than
        *n_paths* if some LM calls fail (fail-open).
        """
        from .lm_studio import LMStudioTool as _LMT

        lm = _LMT(self.lm.base_url, self.model) if self.model else self.lm
        msgs = [
            {
                "role": "system",
                "content": (
                    "You are a careful reasoning assistant. Think through the problem "
                    "step by step before concluding. Show your reasoning process."
                ),
            },
            {
                "role": "user",
                "content": f"Think through this carefully:\n\n{prompt}",
            },
        ]
        tasks = [
            lm.chat(msgs, max_tokens=600, temperature=temperature)
            for _ in range(n_paths)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, str) and r.strip()]

    # ------------------------------------------------------------------
    # Step 2: synthesise final answer from best chain
    # ------------------------------------------------------------------

    async def synthesize(self, question: str, reasoning: str) -> str:
        """Generate a concise final answer given the best reasoning chain.

        Returns empty string on failure (fail-open).
        """
        msgs = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Given the reasoning below, "
                    "provide a clear, concise final answer."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\nReasoning:\n{reasoning}\n\n"
                    "Based on this reasoning, provide a clear final answer:"
                ),
            },
        ]
        return await self.lm.chat(msgs, max_tokens=400, temperature=0.3)

    # ------------------------------------------------------------------
    # End-to-end: fan-out → score → synthesise
    # ------------------------------------------------------------------

    async def think_and_answer(
        self,
        question: str,
        n_paths: int = 3,
        temperature: float = 0.8,
    ) -> ThinkResult:
        """Full parallel thinking pipeline.

        1. Fan out *n_paths* independent reasoning chains in parallel.
        2. Score each chain heuristically.
        3. Synthesise a final answer from the best chain.

        Returns a :class:`ThinkResult`.  On total failure returns an empty
        result (fail-open — callers should check ``result.answer``).
        """
        t0 = time.monotonic()
        chains = await self.think(question, n_paths, temperature)
        if not chains:
            return ThinkResult(
                reasoning="",
                answer="",
                paths_tried=n_paths,
                best_score=0.0,
                duration_ms=0,
                paths_all=[],
            )
        scored = sorted(((self.score_chain(c), c) for c in chains), reverse=True)
        best_score, best_chain = scored[0]
        answer = await self.synthesize(question, best_chain)
        if not answer:
            answer = best_chain  # fail-open: fall back to raw reasoning
        return ThinkResult(
            reasoning=best_chain,
            answer=answer,
            paths_tried=len(chains),
            best_score=best_score,
            duration_ms=int((time.monotonic() - t0) * 1000),
            paths_all=chains,
        )
