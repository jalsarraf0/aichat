from __future__ import annotations

from .shell import ShellTool
from .rss import RSSTool
from .researchbox import ResearchboxTool
from ..state import ApprovalMode


class ToolManager:
    def __init__(self) -> None:
        self.shell = ShellTool()
        self.rss = RSSTool()
        self.researchbox = ResearchboxTool()
        self.max_tool_calls_per_turn = 1

    def check_approval(self, mode: ApprovalMode) -> bool:
        return mode != ApprovalMode.DENY
