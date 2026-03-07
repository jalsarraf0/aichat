#!/usr/bin/env python3
"""
Generate lmstudio-mcp.json from the live MCP gateway.

Usage:
    python3 gen_lmstudio_json.py --mcp-url http://localhost:8096 --out lmstudio-mcp.json
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def fetch_tools(mcp_url: str) -> list[dict]:
    req = urllib.request.Request(
        f"{mcp_url}/mcp",
        data=json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {},
        }).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = json.loads(resp.read())
    tools = body.get("result", {}).get("tools", [])
    if not tools:
        raise RuntimeError(f"No tools returned. Response: {body}")
    return tools


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate lmstudio-mcp.json")
    parser.add_argument("--mcp-url", default="http://localhost:8096", help="MCP gateway base URL")
    parser.add_argument("--out", default="lmstudio-mcp.json", help="Output file path")
    args = parser.parse_args()

    print(f"Fetching tools from {args.mcp_url}/mcp ...", file=sys.stderr)
    tools = fetch_tools(args.mcp_url)

    manifest = {
        "mcpServers": {
            "aichat": {
                "url": f"{args.mcp_url}/sse",
                "transport": "sse",
                "description": (
                    f"aichat MCP platform — {len(tools)} tools for browser, vision, "
                    "docs, memory, graph, vector, jobs, and more."
                ),
                "tools": sorted(tools, key=lambda t: t["name"]),
            }
        }
    }

    with open(args.out, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Written {len(tools)} tools → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
