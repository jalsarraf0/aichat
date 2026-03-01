from __future__ import annotations

import argparse
import sys

from .app import main as app_main
from .github_repo import repo_create_and_push
from .mcp_server import main as mcp_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aichat",
        description="Codex-like Textual TUI with research and operator tooling.",
    )
    subparsers = parser.add_subparsers(dest="command")

    repo_parser = subparsers.add_parser("repo", help="GitHub repository helpers")
    repo_sub = repo_parser.add_subparsers(dest="repo_command", required=True)
    create_parser = repo_sub.add_parser("create", help="Create and push the GitHub repo via SSH")
    create_parser.add_argument("--owner", help="GitHub owner/org for the repo")
    create_visibility = create_parser.add_mutually_exclusive_group()
    create_visibility.add_argument("--private", action="store_true", help="Create a private repository (default)")
    create_visibility.add_argument("--public", action="store_true", help="Create a public repository")
    create_parser.add_argument("--remote", default="origin", help="Git remote name (default: origin)")
    create_parser.set_defaults(func=repo_create_command)

    gh_parser = subparsers.add_parser("github", help="GitHub shortcuts")
    gh_sub = gh_parser.add_subparsers(dest="gh_command", required=True)
    gh_init = gh_sub.add_parser("init", help="Alias for: aichat repo create")
    gh_init.add_argument("--owner", help="GitHub owner/org for the repo")
    gh_visibility = gh_init.add_mutually_exclusive_group()
    gh_visibility.add_argument("--private", action="store_true", help="Create a private repository (default)")
    gh_visibility.add_argument("--public", action="store_true", help="Create a public repository")
    gh_init.add_argument("--remote", default="origin", help="Git remote name (default: origin)")
    gh_init.set_defaults(func=repo_create_command)

    subparsers.add_parser(
        "mcp",
        help=(
            "Run the MCP (Model Context Protocol) server over stdio. "
            "Hook this up to LM Studio or any MCP client."
        ),
    )

    return parser


def repo_create_command(args: argparse.Namespace) -> int:
    visibility = "public" if getattr(args, "public", False) else "private"
    result = repo_create_and_push(
        owner=getattr(args, "owner", None),
        visibility=visibility,
        remote=getattr(args, "remote", "origin"),
    )
    if result.ok:
        print(result.message)
        return 0
    print(result.message)
    return 1


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.command:
        app_main()
        return
    if args.command == "mcp":
        mcp_main()
        return
    if hasattr(args, "func"):
        sys.exit(args.func(args))
    parser.print_help()
    sys.exit(2)


if __name__ == "__main__":
    main()
