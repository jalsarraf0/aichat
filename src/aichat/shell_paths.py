from __future__ import annotations

import os
import re
import shlex
from pathlib import Path


_PROJECT_PATTERNS = [
    re.compile(
        r"\b(?:create|make|mkdir|start)\b\s+(?:a\s+)?(?:new\s+)?(?:project|folder|directory|dir)\b.*?"
        r"\b(?:called|named)\b\s+(?P<name>\"[^\"]+\"|'[^']+'|[A-Za-z0-9._-]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:create|make)\b\s+(?P<name>[A-Za-z0-9._-]+)\s+(?:project|folder|directory|dir)\b",
        re.IGNORECASE,
    ),
]


def detect_project_name(text: str) -> str | None:
    for pattern in _PROJECT_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        name = match.group("name").strip()
        if name.startswith(("\"", "'")) and name.endswith(("\"", "'")):
            name = name[1:-1].strip()
        if name:
            return name
    return None


def rewrite_cd_commands(command: str, project_root: Path, project_path: Path) -> str:
    if not command:
        return command
    root = _safe_resolve(project_root)
    locked = _safe_resolve(project_path)
    lines: list[str] = []
    for line in command.splitlines():
        if not line.strip():
            lines.append(line)
            continue
        try:
            parts = shlex.split(line, posix=True)
        except ValueError:
            lines.append(line)
            continue
        changed = False
        i = 0
        while i < len(parts):
            if parts[i] not in {"cd", "pushd"}:
                i += 1
                continue
            j = i + 1
            while j < len(parts) and parts[j].startswith("-"):
                if parts[j] == "--":
                    j += 1
                    break
                j += 1
            if j >= len(parts):
                i = j + 1
                continue
            target = parts[j]
            if target == "-":
                i = j + 1
                continue
            expanded = os.path.expandvars(target)
            if expanded.startswith("~"):
                expanded = os.path.expanduser(expanded)
            target_path = Path(expanded)
            if target_path.is_absolute() and _is_under(target_path, root) and not _same_path(target_path, locked):
                parts[j] = str(project_path)
                changed = True
            i = j + 1
        if changed:
            line = shlex.join(parts)
        lines.append(line)
    return "\n".join(lines)


def _safe_resolve(path: Path) -> Path:
    try:
        return path.expanduser().resolve()
    except OSError:
        return path.expanduser()


def _is_under(path: Path, root: Path) -> bool:
    path_res = _safe_resolve(path)
    root_res = _safe_resolve(root)
    root_str = str(root_res)
    path_str = str(path_res)
    return path_str == root_str or path_str.startswith(root_str + os.sep)


def _same_path(a: Path, b: Path) -> bool:
    return _safe_resolve(a) == _safe_resolve(b)
