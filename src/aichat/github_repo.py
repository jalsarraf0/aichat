from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class RepoCreateResult:
    ok: bool
    message: str


def iter_ssh_key_candidates(ssh_dir: Path) -> list[Path]:
    preferred = ["id_ed25519", "id_rsa", "id_ecdsa", "id_dsa"]
    candidates: list[Path] = []
    for name in preferred:
        path = ssh_dir / name
        if path.is_file():
            candidates.append(path)
    for path in sorted(ssh_dir.glob("id_*")):
        if path.name.endswith(".pub"):
            continue
        if path in candidates:
            continue
        if path.is_file():
            candidates.append(path)
    return candidates


def probe_ssh_key(
    key_path: Path,
    *,
    runner: Callable = subprocess.run,
) -> bool:
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-i",
        str(key_path),
        "-T",
        "git@github.com",
    ]
    try:
        result = runner(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        return False
    output = (result.stdout or "") + (result.stderr or "")
    lowered = output.lower()
    return "successfully authenticated" in lowered


def select_working_ssh_key(
    ssh_dir: Path | None = None,
    *,
    runner: Callable = subprocess.run,
) -> Path | None:
    ssh_dir = ssh_dir or Path.home() / ".ssh"
    for candidate in iter_ssh_key_candidates(ssh_dir):
        if probe_ssh_key(candidate, runner=runner):
            return candidate
    return None


def gh_authenticated(*, runner: Callable = subprocess.run) -> bool:
    try:
        result = runner(["gh", "auth", "status"], capture_output=True, text=True)
    except FileNotFoundError:
        return False
    return result.returncode == 0


def repo_create_and_push(
    *,
    owner: str | None,
    visibility: str,
    remote: str,
    runner: Callable = subprocess.run,
    log: Callable[[str], None] | None = None,
) -> RepoCreateResult:
    def _log(message: str) -> None:
        if log:
            log(message)

    ssh_key = select_working_ssh_key(runner=runner)
    if ssh_key is None:
        return RepoCreateResult(
            ok=False,
            message="No working SSH key found. Add a GitHub SSH key and try again.",
        )

    if shutil.which("gh") is None:
        return RepoCreateResult(
            ok=False,
            message="Error: GitHub CLI not installed. Install gh and run: gh auth login.",
        )
    if not gh_authenticated(runner=runner):
        return RepoCreateResult(
            ok=False,
            message="Error: GitHub CLI not authenticated. Run: gh auth login (then re-run: aichat repo create).",
        )

    git_ssh_command = f"ssh -i {ssh_key} -o IdentitiesOnly=yes"
    env = os.environ.copy()
    env["GIT_SSH_COMMAND"] = git_ssh_command

    repo_name = "aichat"
    visibility_flag = "--private" if visibility == "private" else "--public"
    create_cmd = [
        "gh",
        "repo",
        "create",
        repo_name,
        visibility_flag,
        "--source",
        ".",
        "--remote",
        remote,
        "--push",
        "--confirm",
    ]
    if owner:
        create_cmd.extend(["--owner", owner])
    _log(f"Using SSH key: {ssh_key}")
    create_result = runner(create_cmd, capture_output=True, text=True, env=env)
    if create_result.returncode != 0:
        details = (create_result.stderr or create_result.stdout or "").strip()
        return RepoCreateResult(
            ok=False,
            message=f"GitHub repo creation failed. {details}",
        )

    owner_name = owner or _resolve_owner(runner=runner, env=env)
    if not owner_name:
        return RepoCreateResult(ok=False, message="Unable to determine GitHub owner name.")
    ssh_url = f"git@github.com:{owner_name}/{repo_name}.git"

    runner(["git", "remote", "set-url", remote, ssh_url], capture_output=True, text=True)
    branch = _current_branch(runner=runner)
    push_result = runner(
        ["git", "push", "-u", remote, branch],
        capture_output=True,
        text=True,
        env=env,
    )
    if push_result.returncode != 0:
        details = (push_result.stderr or push_result.stdout or "").strip()
        return RepoCreateResult(ok=False, message=f"Git push failed. {details}")

    return RepoCreateResult(ok=True, message=f"Repo created and pushed to {ssh_url}.")


def _resolve_owner(*, runner: Callable, env: dict[str, str]) -> str | None:
    result = runner(["gh", "api", "user", "-q", ".login"], capture_output=True, text=True, env=env)
    if result.returncode != 0:
        return None
    return (result.stdout or "").strip() or None


def _current_branch(*, runner: Callable) -> str:
    result = runner(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True)
    return (result.stdout or "").strip() or "main"
