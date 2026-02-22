from __future__ import annotations

from pathlib import Path
import json
from .state import Message


class TranscriptStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (Path.home() / ".local" / "share" / "aichat" / "transcript.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, message: Message) -> None:
        row = {
            "role": message.role,
            "content": message.content,
            "full_content": message.full_content or message.content,
            "metadata": message.metadata,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def export_markdown(self, out_path: Path) -> None:
        lines = ["# AIChat Export", ""]
        if self.path.exists():
            for raw in self.path.read_text(encoding="utf-8").splitlines():
                row = json.loads(raw)
                lines.append(f"## {row['role']}")
                lines.append(row.get("full_content") or row.get("content", ""))
                lines.append("")
        out_path.write_text("\n".join(lines), encoding="utf-8")
