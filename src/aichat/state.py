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


@dataclass
class Message:
    role: str
    content: str
    full_content: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AppState:
    model: str = "local-model"
    base_url: str = "http://localhost:1234"
    approval: ApprovalMode = ApprovalMode.ASK
    streaming: bool = True
    mode: str = "chat"
    theme: str = "cyberpunk"
    busy: bool = False
    tool_calls_this_turn: int = 0
