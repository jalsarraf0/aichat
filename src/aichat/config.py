from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .state import ApprovalMode

LM_STUDIO_BASE_URL = "http://localhost:1234"

CONFIG_PATH = Path.home() / ".config" / "aichat" / "config.yml"


@dataclass(frozen=True)
class AppConfig:
    base_url: str = "http://localhost:1234"
    model: str = "local-model"
    theme: str = "cyberpunk"
    approval: str = ApprovalMode.ASK.value
    allow_host_shell: bool = False


def _validate(cfg: dict[str, Any]) -> dict[str, Any]:
    defaults = AppConfig().__dict__.copy()
    merged = {**defaults, **cfg}
    # Enforce LM Studio endpoint only.
    merged["base_url"] = LM_STUDIO_BASE_URL
    if not isinstance(merged["model"], str) or not merged["model"].strip():
        merged["model"] = defaults["model"]
    if not isinstance(merged["theme"], str) or not merged["theme"].strip():
        merged["theme"] = defaults["theme"]
    if merged["approval"] not in {m.value for m in ApprovalMode}:
        merged["approval"] = defaults["approval"]
    merged["allow_host_shell"] = bool(merged.get("allow_host_shell", defaults["allow_host_shell"]))
    return merged


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        cfg = _validate({})
        save_config(cfg, path)
        return cfg

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = _validate(raw if isinstance(raw, dict) else {})
    if cfg != raw:
        save_config(cfg, path)
    return cfg


def save_config(cfg: dict[str, Any], path: Path = CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    validated = _validate(cfg)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(yaml.safe_dump(validated, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)
