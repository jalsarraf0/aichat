from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ApprovalMode(str, Enum):
    DENY = "DENY"
    ASK = "ASK"
    AUTO = "AUTO"

    def cycle(self) -> "ApprovalMode":
        order = [ApprovalMode.DENY, ApprovalMode.ASK, ApprovalMode.AUTO]
        return order[(order.index(self) + 1) % len(order)]


@dataclass(slots=True)
class Message:
    role: str
    content: str
    full_content: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_chat_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.full_content or self.content}
        if self.role == "assistant" and "tool_calls" in self.metadata:
            payload["tool_calls"] = self.metadata["tool_calls"]
        if self.role == "tool":
            tool_call_id = self.metadata.get("tool_call_id")
            if tool_call_id:
                payload["tool_call_id"] = tool_call_id
        return payload


@dataclass(slots=True)
class AppState:
    model: str = "local-model"
    base_url: str = "http://localhost:1234"
    approval: ApprovalMode = ApprovalMode.ASK
    streaming: bool = True
    theme: str = "cyberpunk"
    allow_host_shell: bool = True
    busy: bool = False
    cwd: str = "."
    max_tool_calls_per_turn: int = 1
    tool_calls_this_turn: int = 0
