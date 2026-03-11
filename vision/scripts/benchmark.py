#!/usr/bin/env python3
"""Benchmark inference latency for all vision MCP tools.

Usage:
    python vision/scripts/benchmark.py [--url URL] [--iterations N] [--tool TOOL]

Example:
    MCP_SERVER_URL=http://localhost:8097 python vision/scripts/benchmark.py
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import statistics
import time

import httpx
from PIL import Image


DEFAULT_URL = "http://localhost:8097"
DEFAULT_ITERATIONS = 10


def make_b64_image(width: int = 640, height: int = 480, color: tuple[int, int, int] = (128, 64, 32)) -> str:
    """Create a test JPEG and return it as base64."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return base64.b64encode(buf.getvalue()).decode()


async def benchmark_tool(
    client: httpx.AsyncClient,
    mcp_url: str,
    name: str,
    args: dict,
    n: int = DEFAULT_ITERATIONS,
) -> dict:
    """Run a single tool N times and return latency statistics."""
    latencies: list[float] = []
    errors = 0

    for _ in range(n):
        t0 = time.perf_counter()
        try:
            resp = await client.post(
                f"{mcp_url}/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": args},
                },
            )
            resp.raise_for_status()
        except Exception:
            errors += 1
            continue
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

    if latencies:
        sorted_lat = sorted(latencies)
        p50 = statistics.median(latencies)
        p95_idx = max(0, int(len(latencies) * 0.95) - 1)
        p95 = sorted_lat[p95_idx]
        p99_idx = max(0, int(len(latencies) * 0.99) - 1)
        p99 = sorted_lat[p99_idx]
        return {
            "tool": name,
            "n": n,
            "errors": errors,
            "p50_ms": round(p50, 1),
            "p95_ms": round(p95, 1),
            "p99_ms": round(p99, 1),
            "min_ms": round(sorted_lat[0], 1),
            "max_ms": round(sorted_lat[-1], 1),
        }
    else:
        return {"tool": name, "n": n, "errors": errors, "all_failed": True}


def print_result(stat: dict) -> None:
    if stat.get("all_failed"):
        print(f"  {stat['tool']:30s}  ALL FAILED ({stat['n']} iterations)")
    else:
        print(
            f"  {stat['tool']:30s}  "
            f"p50={stat['p50_ms']:6.0f}ms  "
            f"p95={stat['p95_ms']:6.0f}ms  "
            f"p99={stat['p99_ms']:6.0f}ms  "
            f"min={stat['min_ms']:6.0f}ms  "
            f"max={stat['max_ms']:6.0f}ms  "
            f"err={stat['errors']}/{stat['n']}"
        )


async def main(mcp_url: str, iterations: int, tool_filter: str | None) -> None:
    b64_small = make_b64_image(64, 64)
    b64_medium = make_b64_image(224, 224)
    b64_large = make_b64_image(640, 480)

    image_small = {"image": {"base64": b64_small}}
    image_medium = {"image": {"base64": b64_medium}}
    image_large = {"image": {"base64": b64_large}}

    all_tools: list[tuple[str, dict]] = [
        # Face tools (small image for speed)
        ("list_face_subjects", {}),
        ("detect_faces", image_small),
        ("recognize_face", image_small),
        ("verify_face", {"image_a": {"base64": b64_small}, "image_b": {"base64": b64_small}}),
        # Vision tools (medium/large images matching typical use)
        ("detect_objects", image_large),
        ("classify_image", image_medium),
        ("detect_clothing", image_medium),
        ("embed_image", image_medium),
        ("analyze_image", {**image_large, "include_clothing": False, "include_embeddings": False}),
    ]

    if tool_filter:
        tools = [(name, args) for name, args in all_tools if tool_filter in name]
        if not tools:
            print(f"No tools matched filter: {tool_filter!r}")
            return
    else:
        tools = all_tools

    print(f"\nVision MCP Inference Benchmark")
    print(f"  Target:     {mcp_url}")
    print(f"  Iterations: {iterations}")
    print(f"  Tools:      {len(tools)}")
    print("-" * 90)

    results: list[dict] = []
    async with httpx.AsyncClient(timeout=120) as client:
        for tool_name, args in tools:
            stat = await benchmark_tool(client, mcp_url, tool_name, args, iterations)
            print_result(stat)
            results.append(stat)

    print("-" * 90)

    # Write JSON report
    report_path = "vision/benchmark-report.json"
    with open(report_path, "w") as f:
        json.dump({"url": mcp_url, "iterations": iterations, "results": results}, f, indent=2)
    print(f"\nFull report written to {report_path}")


def _parse_args() -> argparse.Namespace:
    import os

    parser = argparse.ArgumentParser(description="Vision MCP latency benchmark")
    parser.add_argument(
        "--url",
        default=os.getenv("MCP_SERVER_URL", DEFAULT_URL),
        help=f"MCP server URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"Number of iterations per tool (default: {DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--tool",
        default=None,
        help="Filter tools by name substring",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(main(args.url, args.iterations, args.tool))
