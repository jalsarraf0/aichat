from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import shutil

from .state import Message


class TranscriptStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (Path.home() / ".local" / "share" / "aichat" / "transcript.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)

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
