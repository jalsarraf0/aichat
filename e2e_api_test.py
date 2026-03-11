"""
End-to-end API test: GPT-4o drives all MCP tools in an agentic loop.

Usage:
    OPENAI_API_KEY=sk-... python e2e_api_test.py

Outputs:
    - Live tool call log to stdout
    - e2e_issues.json  — every tool call that returned an error or raised
    - e2e_results.log  — full run transcript
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx
from openai import OpenAI

MCP_URL  = "http://localhost:8096"
MODEL    = "gpt-4o"
PROMPT   = (
    "use all available tools and test them. "
    "for images use 'Find images of Klukai from Girls Frontline 2' "
    "anything else can be random and of your choice. "
    "make it as difficult as possible but doable for the MCP server to "
    "thoroughly test everything"
)
MAX_ITER     = 60
TOOL_TIMEOUT = 180

# ---------------------------------------------------------------------------
# MCP helpers
# ---------------------------------------------------------------------------

def get_mcp_tools() -> list[dict]:
    """Fetch tool list from MCP server and convert to OpenAI function format."""
    r = httpx.post(
        f"{MCP_URL}/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        timeout=10,
    )
    r.raise_for_status()
    tools = r.json()["result"]["tools"]
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["inputSchema"],
            },
        }
        for t in tools
    ]


def call_mcp_tool(name: str, args: dict) -> str:
    """Call MCP tool and return a text summary of the result."""
    r = httpx.post(
        f"{MCP_URL}/mcp",
        json={
            "jsonrpc": "2.0", "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": args},
        },
        timeout=TOOL_TIMEOUT,
    )
    r.raise_for_status()
    result  = r.json().get("result", {})
    content = result.get("content", [])

    parts = []
    for item in content:
        if item.get("type") == "text":
            parts.append(item["text"])
        elif item.get("type") == "image":
            parts.append(f"[image: {len(item.get('data',''))} b64 chars]")

    return "\n".join(parts) if parts else "(empty response)"


# ---------------------------------------------------------------------------
# Issue tracker
# ---------------------------------------------------------------------------

issues: list[dict] = []
transcript: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    transcript.append(msg)


def record_issue(tool: str, args: dict, error: str, source: str = "tool_response") -> None:
    issue = {"tool": tool, "args": {k: str(v)[:200] for k, v in args.items()}, "error": error[:400], "source": source}
    issues.append(issue)
    log(f"  [ISSUE] {tool}: {error[:200]}")


# ---------------------------------------------------------------------------
# Error pattern detection
# ---------------------------------------------------------------------------

_ERROR_PATTERNS = (
    "traceback (most recent",
    "exception:",
    "500 internal server error",
    "404 not found",
    "422 unprocessable",
    "connection refused",
    "timed out",
    "no module named",
)


def _has_error(text: str) -> bool:
    low = text.lower()
    return any(p in low for p in _ERROR_PATTERNS)


# ---------------------------------------------------------------------------
# Main agentic loop
# ---------------------------------------------------------------------------

def run() -> list[dict]:
    log("=== aichat E2E API Test ===")
    log(f"Model : {MODEL}")
    log(f"Target: {MCP_URL}")
    log("")

    log("Fetching MCP tool list...")
    tools = get_mcp_tools()
    tool_names_all = {t["function"]["name"] for t in tools}
    log(f"Found {len(tools)} tools\n")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "You are a thorough QA engineer testing an MCP server. "
                "Call as many distinct tools as possible with realistic, varied inputs. "
                "For image searches and browser use: 'Klukai from Girls Frontline 2'. "
                "For code execution: write non-trivial Python (data analysis, math, plots). "
                "For documents: use real URLs. For memory/graph: use meaningful data. "
                "Keep going until you have tested every tool at least once. "
                "Always report what worked and what failed."
            ),
        },
        {"role": "user", "content": PROMPT},
    ]

    tools_called: list[str] = []
    iteration = 0

    while iteration < MAX_ITER:
        iteration += 1
        log(f"\n--- Iteration {iteration} ---")

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=4096,
                temperature=0.3,
            )
        except Exception as exc:
            log(f"[API ERROR] {exc}")
            record_issue("openai_api", {}, str(exc), source="api")
            break

        choice  = response.choices[0]
        message = choice.message

        # Collect assistant text
        if message.content:
            log(f"[GPT-4o]: {message.content[:600]}")

        messages.append(message)  # type: ignore[arg-type]

        if choice.finish_reason == "stop":
            log("\n[GPT-4o] Finished (stop)")
            break

        if choice.finish_reason != "tool_calls" or not message.tool_calls:
            log(f"[GPT-4o] Unexpected finish_reason: {choice.finish_reason}")
            break

        # --- Execute tool calls ---
        for tc in message.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            tools_called.append(name)
            preview = json.dumps(args)[:120]
            log(f"  [CALL] {name}({preview})")

            try:
                result_text = call_mcp_tool(name, args)

                if _has_error(result_text):
                    record_issue(name, args, result_text[:400])

                preview_out = result_text[:200].replace("\n", " ")
                log(f"    -> {preview_out}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_text[:3000],  # cap for context window
                })

            except httpx.HTTPStatusError as exc:
                err = f"HTTP {exc.response.status_code}: {exc.response.text[:300]}"
                record_issue(name, args, err, source="http_error")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": f"Error: {err}",
                })

            except Exception as exc:
                err = str(exc)
                record_issue(name, args, err, source="exception")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": f"Error: {err}",
                })

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    log("\n" + "=" * 60)
    log(f"ITERATIONS  : {iteration}")
    log(f"TOOLS CALLED: {len(tools_called)} calls, {len(set(tools_called))} distinct")
    log(f"ISSUES FOUND: {len(issues)}")

    uncalled = sorted(tool_names_all - set(tools_called))

    if issues:
        log("\nIssues:")
        for i, iss in enumerate(issues, 1):
            log(f"  {i}. [{iss['tool']}] ({iss['source']}) {iss['error'][:200]}")

    if uncalled:
        log(f"\nTools not exercised ({len(uncalled)}): {', '.join(uncalled)}")
    else:
        log("\nAll tools exercised.")

    Path("e2e_issues.json").write_text(json.dumps(issues, indent=2))
    Path("e2e_results.log").write_text("\n".join(transcript))
    log("\nArtifacts: e2e_issues.json, e2e_results.log")

    return issues


if __name__ == "__main__":
    result = run()
    sys.exit(1 if result else 0)
