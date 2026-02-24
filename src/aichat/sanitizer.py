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
# Complete inline tool-call blobs some models emit inside message content.
_INLINE_TOOL_BLOB_RE = re.compile(
    r"<(?:tool_call|function_calls?|invoke)\b[^>]*>.*?</(?:tool_call|function_calls?|invoke)>",
    re.IGNORECASE | re.DOTALL,
)
# Any opening or closing XML arg/tool tag (stray or half-stripped).
_ARG_TAG_RE = re.compile(
    r"</?(?:arg_key|arg_value|tool_call|tool_name|tool_input|"
    r"function_calls?|invoke|parameters?)(?:\s[^>]*)?>",
    re.IGNORECASE,
)
# Residual closing arg tags that betray a half-stripped tool call.
_RESIDUAL_ARG_CLOSE_RE = re.compile(
    r"</(?:arg_key|arg_value|tool_call|function_calls?|invoke)>",
    re.IGNORECASE,
)


def sanitize_response(text: str) -> SanitizedResponse:
    raw = text or ""
    cleaned = _THINK_RE.sub("", raw)
    cleaned = _ANALYSIS_RE.sub("", cleaned)
    cleaned = cleaned.replace("</think>", "").replace("<think>", "")
    # Unwrap GLM-4 box tokens -- keep the content inside.
    cleaned = _BOX_RE.sub(lambda m: m.group(1), cleaned)
    # Strip leftover pipe-delimited special tokens.
    cleaned = _PIPE_TOKEN_RE.sub("", cleaned)
    # Remove complete inline tool-call blobs before other processing.
    cleaned = _INLINE_TOOL_BLOB_RE.sub("", cleaned)
    # Check for residual arg-close tags BEFORE stripping — their presence
    # betrays a half-stripped tool call that should be hidden.
    had_residual_arg_tags = bool(_RESIDUAL_ARG_CLOSE_RE.search(cleaned))
    # Strip remaining XML arg/tool tags (opening and closing).
    cleaned = _ARG_TAG_RE.sub("", cleaned)
    cleaned = _LEADING_TAG_RE.sub("", cleaned)
    lines = [line.rstrip() for line in cleaned.splitlines()]
    cleaned = "\n".join(lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    structured_hidden = had_residual_arg_tags or _looks_structured(cleaned)
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
    # Residual closing arg/tool tags mean a half-stripped tool-call leaked through.
    if _RESIDUAL_ARG_CLOSE_RE.search(stripped):
        return True
    return False
