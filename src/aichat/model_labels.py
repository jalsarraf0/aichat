from __future__ import annotations

from collections.abc import Iterable


def model_capability_emojis(name: str) -> list[str]:
    lowered = name.lower()
    emojis: list[str] = []
    rules: list[tuple[str, str]] = [
        ("vision", "🖼️"),
        ("vl", "🖼️"),
        ("image", "🖼️"),
        ("multimodal", "🖼️"),
        ("code", "💻"),
        ("coder", "💻"),
        ("codex", "💻"),
        ("tool", "🛠️"),
        ("function", "🛠️"),
        ("agent", "🛠️"),
        ("math", "🧮"),
        ("reason", "🧠"),
        ("chat", "💬"),
        ("instruct", "💬"),
        ("assistant", "💬"),
        ("embed", "🧲"),
        ("embedding", "🧲"),
        ("audio", "🎧"),
        ("speech", "🎧"),
        ("tts", "🎧"),
        ("asr", "🎧"),
    ]
    for token, emoji in rules:
        if token in lowered and emoji not in emojis:
            emojis.append(emoji)
    return emojis


def decorate_model_label(name: str) -> str:
    emojis = model_capability_emojis(name)
    if not emojis:
        return name
    return f"{name} {' '.join(emojis)}"


def model_options(models: Iterable[str]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    options: list[tuple[str, str]] = []
    for model in models:
        if not isinstance(model, str):
            continue
        model_name = model.strip()
        if not model_name or model_name in seen:
            continue
        seen.add(model_name)
        options.append((decorate_model_label(model_name), model_name))
    return options
