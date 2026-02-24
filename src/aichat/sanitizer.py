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
# GLM-4 and similar models wrap final answers in box tokens.
_BOX_RE = re.compile(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", re.DOTALL)
# Generic pipe-delimited special tokens, e.g. <|im_start|>, <|eot_id|>
_PIPE_TOKEN_RE = re.compile(r"<\|[a-z_]+\|>")


def sanitize_response(text: str) -> SanitizedResponse:
    raw = text or ""
    cleaned = _THINK_RE.sub("", raw)
    cleaned = _ANALYSIS_RE.sub("", cleaned)
    cleaned = cleaned.replace("</think>", "").replace("<think>", "")
    # Unwrap GLM-4 box tokens — keep the content inside.
    cleaned = _BOX_RE.sub(lambda m: m.group(1), cleaned)
    # Strip leftover pipe-delimited special tokens.
    cleaned = _PIPE_TOKEN_RE.sub("", cleaned)
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
