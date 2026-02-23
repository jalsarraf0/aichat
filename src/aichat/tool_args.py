from __future__ import annotations

import json
import re

import yaml


def parse_tool_args(name: str, args_text: str) -> tuple[dict[str, object], str | None]:
    if not args_text:
        return {}, None
    cleaned = args_text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.splitlines()[1:-1]).strip()
    parsed: object
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        if name == "shell_exec" and "\n" in cleaned and "command" in cleaned:
            command = _extract_shell_command(cleaned)
            if command:
                return {"command": command}, None
        try:
            parsed = yaml.safe_load(cleaned)
        except Exception as exc:  # noqa: BLE001
            if name == "shell_exec" and cleaned:
                command = _extract_shell_command(cleaned)
                if command:
                    return {"command": command}, None
                return {"command": cleaned}, None
            return {}, f"invalid tool arguments: {exc}"
    if isinstance(parsed, dict):
        if name == "shell_exec":
            command = parsed.get("command")
            if isinstance(command, str) and "\n" in cleaned and "\n" not in command:
                extracted = _extract_shell_command(cleaned)
                if extracted:
                    return {"command": extracted}, None
        return parsed, None
    if name == "shell_exec" and isinstance(parsed, str) and parsed.strip():
        return {"command": parsed.strip()}, None
    if name == "shell_exec" and cleaned:
        return {"command": cleaned}, None
    return {}, "invalid tool arguments: expected object payload"


def _extract_shell_command(text: str) -> str | None:
    raw = text.strip()
    if not raw:
        return None
    # YAML-style "command: |" or "command: <inline>"
    if raw.lstrip().lower().startswith("command:"):
        lines = raw.splitlines()
        first = lines[0]
        inline = first.split(":", 1)[1].strip()
        rest = lines[1:]
        if inline and inline not in {"|", ">"} and not rest:
            return inline
        if rest:
            indents = [len(line) - len(line.lstrip(" ")) for line in rest if line.strip()]
            min_indent = min(indents) if indents else 0
            lines = [line[min_indent:] if len(line) >= min_indent else line for line in rest]
            return "\n".join(lines).rstrip()
        if inline:
            return inline
    # JSON-ish fallback: extract value after "command":
    match = re.search(r"([\"']?command[\"']?)\s*:\s*(.*)", raw, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    tail = match.group(2).strip()
    # Drop trailing braces/commas.
    tail = tail.rstrip()
    while tail and tail[-1] in "}\n\r\t ,":
        tail = tail[:-1].rstrip()
    # Strip matching quotes if present.
    if tail.startswith(("\"", "'")):
        quote = tail[0]
        if tail.endswith(quote):
            tail = tail[1:-1]
        else:
            tail = tail[1:]
    return tail.strip() or None
