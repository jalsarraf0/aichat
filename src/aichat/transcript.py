from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
import shutil

from .state import Message


def _default_transcript_path() -> Path:
    configured = os.environ.get("AICHAT_TRANSCRIPT_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".local" / "share" / "aichat" / "transcript.jsonl"


def _ensure_writable_transcript_path(path: Path) -> Path:
    candidate = path.expanduser()
    try:
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.touch(exist_ok=True)
        with candidate.open("a", encoding="utf-8"):
            pass
        return candidate
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "aichat" / "transcript.jsonl"
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fallback.touch(exist_ok=True)
        return fallback


class TranscriptStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = _ensure_writable_transcript_path(path or _default_transcript_path())

    def append(self, message: Message) -> None:
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "role": message.role,
            "content": message.content,
            "full_content": message.full_content or message.content,
            "metadata": message.metadata,
        }
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    def has_content(self) -> bool:
        return self.path.exists() and self.path.stat().st_size > 0

    def clear(self) -> None:
        if self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def archive_to(self, target_dir: Path) -> Path | None:
        if not self.has_content():
            return None
        target_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        dest = target_dir / f"aichat-{timestamp}.jsonl"
        shutil.copy2(self.path, dest)
        return dest

    def load_messages(self) -> list[Message]:
        if not self.path.exists():
            return []
        rows: list[Message] = []
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            record = json.loads(raw)
            rows.append(
                Message(
                    role=record.get("role", "assistant"),
                    content=record.get("content", ""),
                    full_content=record.get("full_content"),
                    metadata=record.get("metadata", {}),
                )
            )
        return rows

    def search(self, query: str) -> list[dict[str, str]]:
        query_lower = query.lower().strip()
        if not query_lower or not self.path.exists():
            return []
        results: list[dict[str, str]] = []
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            record = json.loads(raw)
            text = (record.get("full_content") or record.get("content") or "")
            if query_lower in text.lower():
                results.append(
                    {
                        "timestamp": record.get("timestamp", ""),
                        "role": record.get("role", ""),
                        "content": text,
                    }
                )
        return results

    def export_markdown(self, out_path: Path) -> None:
        lines = ["# AIChat Transcript", ""]
        for message in self.load_messages():
            lines.append(f"## {message.role}")
            lines.append(message.full_content or message.content)
            lines.append("")
        out_path.write_text("\n".join(lines), encoding="utf-8")
