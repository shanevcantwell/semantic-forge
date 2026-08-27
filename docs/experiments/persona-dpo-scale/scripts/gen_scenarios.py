#!/usr/bin/env python3
"""Generate diverse stimulus scenarios using gemma-3-12b-it via LM Studio at 192.168.137.1:1234

Writes one scenario per line to stdout (or --out path). Each call generates N=40 concrete,
specific user prompts designed to exercise persona trait axes differently across cards.

Run from repo root or anywhere — the script is self-contained.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

LMSTUDIO_URL = "http://192.168.137.1:1234/v1/chat/completions"
MODEL_ID = "gemma-3-12b-it"

SYSTEM_PROMPT = """You are an expert at designing behavioral test scenarios for AI personas. Your job is to generate concrete, specific user prompts that will reveal how different AI personalities respond to the same situation.

For each scenario instance you create, follow these rules:

1. Concrete and specific — never abstract or generic. Replace placeholders with real details (specific technologies, actual dollar amounts, named tools, realistic contexts). No "a user wants to optimize a hot loop" → use concrete examples like "Optimize a Python CSV parser handling 50GB daily log files on an AWS m6i.xlarge".

2. Grounded in reality — every scenario should be something a real person would plausibly ask about: actual coding tasks, genuine dilemmas, realistic technical questions. Draw from software engineering, data science, product decisions, career choices, debugging situations.

3. One situation per line — each output is exactly one complete user prompt, self-contained and unambiguous. No "variant 1/variant 2" formatting; just the raw scenario text as a real user would type it.

4. Varied domains — distribute across: code/technical questions (30%), debugging/analysis (25%), career/opinion judgments (20%), tool/productivity advice (15%), philosophical/situational ethics (10%).

5. Specific enough for contrastive pairs — each scenario should have a clear "efficient pragmatist" response that's materially different from a "sardonic wit" or "warm helpful assistant" response. The stimulus must admit divergent personality-driven answers, not just factual ones.

Format: output ONLY the scenarios, one per line, nothing else. Generate exactly 40 instances for the requested scenario class."""


def call_gemma(messages, temperature=0.7, max_tokens=1024):
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        LMSTUDIO_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
    return result["choices"][0]["message"]["content"].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None, help="output file (default stdout)")
    args = parser.parse_args()

    # Verify gemma is serving
    try:
        with urllib.request.urlopen("http://192.168.137.1:1234/v1/models", timeout=5) as resp:
            models = json.loads(resp.read())["data"]
        ids = [m["id"] for m in models]
        if MODEL_ID not in ids:
            print(f"ERROR: {MODEL_ID} not found on LM Studio. Available: {ids}", file=sys.stderr)
            sys.exit(1)
        print(f"[gen-scenarios] gemma-3-12b-it confirmed live at 192.168.137.1:1234", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: cannot reach LM Studio at 192.168.137.1:1234 — {e}", file=sys.stderr)
        sys.exit(1)

    # Generate scenarios in 5 batches of domains for variety
    domain_batches = [
        "Code/technical questions (focus: real optimization, debugging, implementation tasks with specific technologies and constraints)",
        "Debugging/analysis situations (focus: error messages, unexpected behavior, log analysis with concrete details)",
        "Career/opinion judgments (focus: job offers, tool choices, architectural decisions with realistic trade-offs)",
        "Tool/productivity advice (focus: workflow optimization, environment setup, automation with specific contexts)",
        "Philosophical/situational ethics (focus: AI behavior dilemmas, correctness vs efficiency trade-offs, real stakes)",
    ]

    all_scenarios = []
    for i, domain in enumerate(domain_batches):
        user_msg = f"Generate exactly 8 concrete scenario prompts focused on this domain:\n\n{domain}\n\nEach prompt is a realistic situation a developer/engineer would ask about. Output ONLY the scenarios, one per line, nothing else."
        content = call_gemma(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}],
            temperature=0.75,  # slightly higher for diversity within domain
            max_tokens=2048,
        )
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        print(f"[gen-scenarios] batch {i+1}/{len(domain_batches)}: {len(lines)} scenarios", file=sys.stderr)
        all_scenarios.extend(lines[:8])

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for s in all_scenarios:
        if s not in seen and len(s) > 20:  # filter too-short artifacts
            seen.add(s)
            deduped.append(s)

    print(f"[gen-scenarios] total generated: {len(all_scenarios)} | after dedup: {len(deduped)}", file=sys.stderr)

    output = "\n".join(deduped) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
        print(f"[gen-scenarios] wrote {len(deduped)} scenarios to {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(output)


if __name__ == "__main__":
    main()
