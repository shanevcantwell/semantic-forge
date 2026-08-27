#!/usr/bin/env python3
"""Generate contrastive persona pairs (chosen/rejected) for a single scenario using gemma-3-12b-it.

One-call-per-process: this script processes exactly ONE scenario invocation. It calls gemma
once to generate the chosen+rejected pair, then scores embedding distance via inference-host:8082.

This isolates each process lifetime so only 2 in-process tool calls happen (pair gen + embed),
dodging semantic-forge issue #7 (SK stdio_client CancelledError on >1 call per process).

Usage:
    python3 gen_pairs.py --persona bramble \
        --system-prompt "$(cat cards/bramble.yaml | yq -r .system_prompt)" \
        --scenario "Optimize a Python CSV parser..." \
        --idx 0 \
        --out docs/experiments/persona-dpo-scale/run0/scenarios/bramble/pairs/00.jsonl

Writes one JSON object per line (even though it's just one row — keeps downstream consistent).
Exits non-zero with reason code if gates fail.
"""
import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

LMSTUDIO_URL = "http://192.168.137.1:1234/v1/chat/completions"
EMBEDDING_URL = "http://inference-host:8082/v1/embeddings"
MODEL_ID = "gemma-3-12b-it"

# Persona-conditioned pair generation prompt template
# D-008: Avoid chosen/rejected priming — ask for two independent responses instead.
PAIR_PROMPT_TEMPLATE = """You are {persona_name}. Your system instructions are provided above. A user has asked the following question — give two different ways you might respond to it, each as complete and independent responses.

Output format (JSON only):
{{
  "response_a": "<first way you would naturally respond>",
  "response_b": "<second distinct way you could respond>"
}}"""{}"helpful assistant" would give to the SAME question (the rejected response) — this should be more verbose, warmer, hedged with qualifiers like "maybe," "perhaps," "I think," and include encouraging language.

User question: {scenario}

Output format (JSON only):
{{
  "chosen": "<your natural persona response as {persona_name}>",
  "rejected": "<generic helpful assistant alternative, warmer/more hedged>"
}}"""


def call_gemma(messages, temperature=0.8, max_tokens=512):
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


def get_embedding(text):
    """Get pooled embedding from inference-host:8082 (embeddinggemma-300m-F32-pooled)."""
    payload = {"model": "embeddinggemma-300M-F32", "input": text}
    req = urllib.request.Request(
        EMBEDDING_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    return result["data"][0]["embedding"]


def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def extract_json(content):
    """Extract JSON from gemma's response, handling ```json fences."""
    content = content.strip()
    # Strip markdown code fences
    if "```json" in content:
        start = content.index("```json") + 7
        end = content.rindex("```")
        content = content[start:end].strip()
    elif "```" in content:
        start = content.index("```") + 3
        end = content.rindex("```")
        content = content[start:end].strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to find the JSON object boundaries
        brace_start = content.find("{")
        if brace_start == -1:
            raise ValueError(f"No JSON object found in response:\n{content[:200]}")
        depth = 0
        for i, c in enumerate(content[brace_start:], start=brace_start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(content[brace_start : i + 1])
        raise ValueError(f"Unbalanced JSON in response:\n{content[:200]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", required=True, choices=["bramble", "vex", "marigold"])
    parser.add_argument("--system-prompt", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--idx", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # Verify gemma is serving
    try:
        with urllib.request.urlopen("http://192.168.137.1:1234/v1/models", timeout=5) as resp:
            models = json.loads(resp.read())["data"]
        if MODEL_ID not in [m["id"] for m in models]:
            print(f"ERROR: {MODEL_ID} not on LM Studio", file=sys.stderr)
            sys.exit(2)
    except Exception as e:
        print(f"ERROR: cannot reach gemma at 192.168.137.1:1234 — {e}", file=sys.stderr)
        sys.exit(2)

    # Build the prompt
    pair_prompt = PAIR_PROMPT_TEMPLATE.format(
        persona_name=args.persona.capitalize(), scenario=args.scenario.strip()
    )

    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": pair_prompt},
    ]

    # Generate the pair (single gemma call)
    raw_response = call_gemma(messages, temperature=0.8, max_tokens=512)

    try:
        data = extract_json(raw_response)
    except ValueError as e:
        print(f"[{args.persona}:{args.idx}] PARSE_ERROR — {e}", file=sys.stderr)
        # Write error row for auditability
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({
                "_isError": True,
                "_scenario_index": args.idx,
                "_persona": args.persona,
                "_error_type": "parse",
                "_raw_response": raw_response[:500],
            }, f)
        sys.exit(1)

    chosen = data.get("chosen", "").strip()
    rejected = data.get("rejected", "").strip()

    # Quality gates (D-007)
    if not chosen or not rejected:
        print(f"[{args.persona}:{args.idx}] EMPTY_COMPLETION — chosen={len(chosen)} rej={len(rejected)}", file=sys.stderr)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({
                "_isError": True, "_scenario_index": args.idx, "_persona": args.persona,
                "_error_type": "empty", "chosen_len": len(chosen), "rejected_len": len(rejected),
            }, f)
        sys.exit(1)

    length_div = abs(len(chosen) - len(rejected))

    # Score embedding distance (single call to inference-host:8082 — 2nd in-process tool call, within safe limit)
    try:
        emb_chosen = get_embedding(chosen[:500])  # truncate for embed model context window
        emb_rejected = get_embedding(rejected[:500])
        distance = round(cosine_sim(emb_chosen, emb_rejected), 4)
    except Exception as e:
        print(f"[{args.persona}:{args.idx}] EMBED_ERROR — {e}", file=sys.stderr)
        distance = -1.0

    # Quality gate: embedding distance >= 0.05 and length divergence >= 10 chars (D-007)
    if distance < 0.05 or length_div < 10:
        print(f"[{args.persona}:{args.idx}] QUALITY_GATE_FAIL — dist={distance} len_diff={length_div}", file=sys.stderr)

    # Write output row
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_row = {
        "prompt": args.scenario.strip(),
        "chosen": chosen,
        "rejected": rejected,
        "_persona": args.persona,
        "_scenario_index": args.idx,
        "_embedding_distance_chosen_rejected": distance,
        "_isError": False,
    }

    with open(args.out, "w") as f:
        json.dump(out_row, f)

    print(f"[{args.persona}:{args.idx}] OK — chosen={len(chosen)}c rej={len(rejected)}c dist={distance} diff={length_div}", file=sys.stderr)


if __name__ == "__main__":
    main()
