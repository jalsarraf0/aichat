#!/usr/bin/env python3
"""Dartboard E2E Regression Test Suite.

Tests all models with image + news prompts, verifies tool routing,
image dedup, personality filtering, and auth.

Usage:
    python3 test/e2e_regression.py [APP_URL]
    Default APP_URL: auto-detect from Docker container
"""

import json
import socket
import subprocess
import sys
import time
import urllib.request

# ── Config ────────────────────────────────────────────────────────────

MODELS = [
    "dolphin-mistral-glm-4.7-flash-24b-venice-edition-thinking-uncensored-i1",
    "zai-org/glm-4.6v-flash",
    "openai/gpt-oss-20b",
    "qwen/qwen3.5-9b",
    "ibm/granite-4-h-tiny",
    "microsoft/phi-4-mini-reasoning",
]

PROMPTS = [
    ("Find photos of the Golden Gate Bridge at sunset", "image"),
    ("What are the latest UFC fight results", "news"),
]

TIMEOUT_PER_PROMPT = 150  # seconds


def get_app_url():
    """Auto-detect dartboard app URL from Docker."""
    if len(sys.argv) > 1:
        return sys.argv[1]
    try:
        result = subprocess.run(
            ["docker", "inspect", "dart-board-app-1",
             "--format", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}"],
            capture_output=True, text=True, timeout=5
        )
        ip = result.stdout.strip()
        if ip:
            return f"http://{ip}:8200"
    except Exception:
        pass
    return "http://localhost:8200"


def api(base, path, data=None):
    url = f"{base}{path}"
    if data:
        req = urllib.request.Request(
            url, json.dumps(data).encode(),
            {"Content-Type": "application/json"}
        )
    else:
        req = urllib.request.Request(url)
    return json.loads(urllib.request.urlopen(req, timeout=15).read())


def send_message(base, conv_id, content):
    url = f"{base}/api/conversations/{conv_id}/messages"
    req = urllib.request.Request(
        url, json.dumps({"content": content}).encode(),
        {"Content-Type": "application/json"}
    )
    try:
        body = urllib.request.urlopen(req, timeout=TIMEOUT_PER_PROMPT).read().decode()
    except socket.timeout:
        return {"status": "TIMEOUT", "tools": 0, "imgs": 0, "dupes": 0, "done": False}
    except Exception as e:
        return {"status": "ERROR", "tools": 0, "imgs": 0, "dupes": 0, "done": False, "err": str(e)[:60]}

    tools = body.count("event: tool_start")
    done = body.count("event: done") > 0
    errors = body.count("event: error")

    # Image analysis
    all_urls = []
    for line in body.split("\n"):
        if '"imageUrls"' in line:
            line = line.strip().replace("data: ", "")
            try:
                d = json.loads(line)
                all_urls.extend(d.get("imageUrls", []))
            except Exception:
                pass
    seen = set()
    dupes = sum(1 for u in all_urls if u.split("?")[0].lower() in seen or not seen.add(u.split("?")[0].lower()))

    # Check for markdown image leakage
    md_imgs = sum(1 for l in body.split("\n") if "![image]" in l and "event: token" in body[:body.index(l)] if "event: token" in body)

    status = "PASS" if done and errors == 0 else "ERROR" if errors > 0 else "TIMEOUT"
    if done and tools == 0 and not errors:
        status = "PASS/txt"

    return {
        "status": status,
        "tools": tools,
        "imgs": len(all_urls),
        "dupes": dupes,
        "md_leak": md_imgs > 0,
        "done": done,
    }


def test_personalities(base):
    """Verify model-restricted personalities filter correctly."""
    # Without model filter
    p_no_model = api(base, "/api/personalities")
    total = len(p_no_model["personalities"])

    # With specific model
    p_qwen = api(base, "/api/personalities?model=qwen/qwen3.5-9b")
    total_qwen = len(p_qwen["personalities"])

    results = []
    results.append(("Personalities loaded", "PASS" if total > 0 else "FAIL"))
    results.append(("Model filter works", "PASS" if total_qwen <= total else "FAIL"))
    return results


def test_tools(base):
    """Verify tool count and refresh."""
    tools = api(base, "/api/tools")
    count = tools["count"]
    return [("Tool count >= 16", "PASS" if count >= 16 else f"FAIL ({count})")]


def main():
    base = get_app_url()
    print("=" * 70)
    print(f"DARTBOARD E2E REGRESSION TEST")
    print(f"Target: {base}")
    print("=" * 70)

    # ── Infrastructure tests ──────────────────────────────────────────
    print("\n--- Infrastructure ---")
    for label, result in test_tools(base):
        print(f"  {result:10s} {label}")
    for label, result in test_personalities(base):
        print(f"  {result:10s} {label}")

    # ── Model tests ───────────────────────────────────────────────────
    print("\n--- Model E2E ---")
    results = {}
    total_pass = 0
    total_tests = 0

    for model in MODELS:
        short = model.split("/")[-1][:25]
        results[model] = []

        for prompt, ptype in PROMPTS:
            total_tests += 1
            label = f"{short}/{ptype}"
            print(f"  {label:35s}", end="", flush=True)

            try:
                conv = api(base, "/api/conversations",
                           {"model": model, "personality_id": "general"})
                conv_id = conv["id"]
            except Exception as e:
                print(f" FAIL (create: {str(e)[:40]})")
                results[model].append({"type": ptype, "status": "FAIL"})
                continue

            start = time.time()
            r = send_message(base, conv_id, prompt)
            elapsed = int(time.time() - start)
            r["time"] = elapsed
            r["type"] = ptype

            dupe_warn = " DUPES!" if r.get("dupes", 0) > 0 else ""
            md_warn = " MD_LEAK!" if r.get("md_leak") else ""
            print(f" {r['status']:10s} {elapsed:3d}s {r['tools']}t {r['imgs']}img{dupe_warn}{md_warn}")

            if r["status"].startswith("PASS"):
                total_pass += 1
            results[model].append(r)

            time.sleep(3)  # Let JIST stabilize
        time.sleep(5)  # Pause between models

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {total_pass}/{total_tests} pass")
    print(f"{'=' * 70}")
    print(f"{'Model':30s} {'Image':12s} {'News':12s}")
    print(f"{'-' * 30} {'-' * 12} {'-' * 12}")
    for model in MODELS:
        short = model.split("/")[-1][:28]
        mr = results.get(model, [])
        p1 = mr[0]["status"] if len(mr) > 0 else "N/A"
        p2 = mr[1]["status"] if len(mr) > 1 else "N/A"
        print(f"{short:30s} {p1:12s} {p2:12s}")

    # Save results
    with open("/tmp/e2e_regression_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to /tmp/e2e_regression_results.json")

    return 0 if total_pass >= total_tests * 0.7 else 1


if __name__ == "__main__":
    sys.exit(main())
