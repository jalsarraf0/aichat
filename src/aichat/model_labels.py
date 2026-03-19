from __future__ import annotations

import re
from collections.abc import Iterable


def model_capability_emojis(name: str) -> list[str]:
    """Infer capability badges from model name patterns."""
    lowered = name.lower()
    emojis: list[str] = []
    # Vision / VLM detection (highest priority)
    if re.search(r"\bvl\b|-vl-|vision|4v|4\.6v|vlm", lowered):
        emojis.append("\U0001f441\ufe0f")  # eye
    # Reasoning / thinking detection
    if re.search(r"reason|think|qwen3\.5|phi-4|magistral|ministral|glm", lowered):
        emojis.append("\U0001f9e0")  # brain
    # Uncensored / unrestricted
    if "dolphin" in lowered or "uncensored" in lowered:
        emojis.append("\U0001f513")  # unlocked
    # Tool use (most LLMs, skip for embeddings)
    if "embed" not in lowered:
        emojis.append("\U0001f527")  # wrench
    # Embedding-only
    if "embed" in lowered:
        emojis.append("\U0001f9f2")  # magnet
    return emojis


def decorate_model_label(name: str) -> str:
    """Add capability emoji badges to a model name for display."""
    emojis = model_capability_emojis(name)
    if not emojis:
        return name
    return f"{name}  {' '.join(emojis)}"


def model_options(models: Iterable[str]) -> list[tuple[str, str]]:
    """Build (display_label, raw_name) pairs for a model picker."""
    seen: set[str] = set()
    options: list[tuple[str, str]] = []
    for model in models:
        if not isinstance(model, str):
            continue
        model_name = model.strip()
        if not model_name or model_name in seen:
            continue
        seen.add(model_name)
        options.append((decorate_model_label(model_name), model_name))
    return options
