from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SanitizedResponse:
    text: str
    structured_hidden: bool


_THINK_RE = re.compile(r"(?is)<think>.*?</think>")
_ANALYSIS_RE = re.compile(r"(?is)<analysis>.*?</analysis>")
_LEADING_TAG_RE = re.compile(r"^\s*(<[^>\n]+>\s*)+", re.MULTILINE)


def sanitize_response(text: str) -> SanitizedResponse:
    raw = text or ""
    cleaned = _THINK_RE.sub("", raw)
    cleaned = _ANALYSIS_RE.sub("", cleaned)
    cleaned = cleaned.replace("</think>", "").replace("<think>", "")
    cleaned = _LEADING_TAG_RE.sub("", cleaned)
    lines = [line.rstrip() for line in cleaned.splitlines()]
    cleaned = "\n".join(lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    structured_hidden = _looks_structured(cleaned)
    return SanitizedResponse(text=cleaned, structured_hidden=structured_hidden)


def _looks_structured(text: str) -> bool:
    if not text.strip():
        return False
    stripped = text.lstrip()
    lowered = stripped.lower()
    if lowered.startswith(("{", "[", "<tool", "<think", "<json", "<xml")):
        return True
    if re.match(r"^```(json|yaml|xml)", lowered):
        return True
    brace_count = stripped.count("{") + stripped.count("}")
    if brace_count >= 6:
        return True
    return False
