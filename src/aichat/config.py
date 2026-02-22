from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .state import ApprovalMode
from .personalities import DEFAULT_PERSONALITY_ID, default_personalities, normalize_personalities

LM_STUDIO_BASE_URL = "http://localhost:1234"

CONFIG_PATH = Path.home() / ".config" / "aichat" / "config.yml"


@dataclass(frozen=True)
class AppConfig:
    base_url: str = "http://localhost:1234"
    model: str = "local-model"
    theme: str = "cyberpunk"
    approval: str = ApprovalMode.ASK.value
    concise_mode: bool = False
    shell_enabled: bool = True
    active_personality: str = DEFAULT_PERSONALITY_ID
    personalities: list[dict[str, str]] = field(default_factory=default_personalities)
    config_version: int = 4


def _validate(cfg: dict[str, Any]) -> dict[str, Any]:
    defaults = AppConfig().__dict__.copy()
    merged = {**defaults, **cfg}
    raw_version = cfg.get("config_version", 1)
    if isinstance(raw_version, (int, str)) and str(raw_version).isdigit():
        cfg_version = int(raw_version)
    else:
        cfg_version = 1
    # Enforce LM Studio endpoint only.
    merged["base_url"] = LM_STUDIO_BASE_URL
    if not isinstance(merged["model"], str) or not merged["model"].strip():
        merged["model"] = defaults["model"]
    if not isinstance(merged["theme"], str) or not merged["theme"].strip():
        merged["theme"] = defaults["theme"]
    if merged["approval"] not in {m.value for m in ApprovalMode}:
        merged["approval"] = defaults["approval"]
    if cfg_version < 2:
        merged["concise_mode"] = defaults["concise_mode"]
    else:
        merged["concise_mode"] = bool(merged.get("concise_mode", defaults["concise_mode"]))
    if cfg_version < 3:
        merged["shell_enabled"] = defaults["shell_enabled"]
    else:
        merged["shell_enabled"] = bool(
            merged.get("shell_enabled", merged.get("allow_host_shell", defaults["shell_enabled"]))
        )
    if cfg_version < 4:
        merged["personalities"] = defaults["personalities"]
        merged["active_personality"] = defaults["active_personality"]
    else:
        merged["personalities"] = normalize_personalities(merged.get("personalities"), defaults["personalities"])
        merged["active_personality"] = str(merged.get("active_personality") or defaults["active_personality"])
    active = merged["active_personality"]
    ids = {p.get("id") for p in merged["personalities"] if isinstance(p, dict)}
    if active not in ids:
        merged["active_personality"] = defaults["active_personality"]
    merged["config_version"] = defaults["config_version"]
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
