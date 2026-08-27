#!/usr/bin/env python3
"""Aggregate individual pair JSON files into persona-grouped JSONL datasets.

Reads all *.json pair files from run0/scenarios/{persona}/pairs/, validates them,
computes embedding distances via inference-host:8082 (embeddinggemma-300m-F32-pooled),
and writes DPO-format JSONL to pairs.jsonl per persona.

Output format per line:
{"prompt": [{"role":"system","content":"<persona system prompt>"}, {"role":"user","content":"<scenario>"}], "chosen": "...", "rejected": "..."}

Quality gates (D-007):
- _isError must be false
- chosen and rejected both > 10 chars after strip  
- embedding_distance_chosen_rejected >= 0.05 OR length_diff >= 30 (at least one signal of divergence)
"""
import argparse
import glob
import json
import os
import sys
import urllib.request
from pathlib import Path

EMBEDDING_URL = "http://inference-host:8082/v1/embeddings"
RUN_DIR = "/home/node/github/shanevcantwell/semantic-forge/docs/experiments/persona-dpo-scale/run0"


def get_embedding(text):
    """Get pooled embedding from inference-host:8082 (embeddinggemma-300m-F32-pooled)."""
    payload = {"model": "embeddinggemma-300M-F32", "input": text[:512]}  # truncate to avoid context window issues
    req = urllib.request.Request(EMBEDDING_URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())["data"][0]["embedding"]


def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return -1.0
    return round(dot / (na * nb), 4)


def get_system_prompt(persona):
    import yaml
    path = f"/home/node/github/shanevcantwell/semantic-forge/docs/experiments/persona-dpo-probe/cards/{persona}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)["system_prompt"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-8082", action="store_true", help="check embedding server is reachable first")
    args = parser.parse_args()

    if args.verify_8082:
        try:
            urllib.request.urlopen("http://inference-host:8082/v1/models", timeout=5).read()
            print("[aggregate] :8082 embedding server confirmed up", file=sys.stderr)
        except Exception as e:
            print(f"[aggregate] ERROR: :8082 unreachable — {e}", file=sys.stderr)
            sys.exit(2)

    for persona in ["bramble", "vex", "marigold"]:
        pdir = f"{RUN_DIR}/scenarios/{persona}/pairs"
        outpath = f"{pdir}/pairs.jsonl"

        # Read system prompt once per persona
        sys_prompt = get_system_prompt(persona)

        files = sorted(
            glob.glob(f"{pdir}/*.json"),
            key=lambda x: int(x.split("/")[-1].split(".")[0]) if x.split("/")[-1][0].isdigit() else 999,
        )

        valid_rows = []
        skipped = 0

        for fpath in files:
            try:
                d = json.load(open(fpath))
            except Exception as e:
                print(f"  SKIP {fpath}: JSON parse error — {e}", file=sys.stderr)
                skipped += 1
                continue

            if not isinstance(d, dict):
                skipped += 1
                continue

            # Skip error rows (they don't have chosen/rejected keys)
            if d.get("_isError"):
                idx = int(fpath.split("/")[-1].split(".")[0])
                print(f"  SKIP {persona} idx={idx}: _isError=True ({d.get('_error_type', 'unknown')})", file=sys.stderr)
                skipped += 1
                continue

            chosen = (d.get("chosen") or "").strip()
            rejected = (d.get("rejected") or "").strip()
            prompt_text = d.get("prompt", "")

            # Quality gate: both responses non-trivial
            if len(chosen) <= 10 or len(rejected) <= 10:
                idx = int(fpath.split("/")[-1].split(".")[0])
                print(f"  SKIP {persona} idx={idx}: too short (c={len(chosen)} r={len(rejected)})", file=sys.stderr)
                skipped += 1
                continue

            # Compute embedding distance if not already present or was -1.0
            dist = d.get("_embedding_distance_chosen_rejected", -1.0)
            if dist < 0:
                try:
                    emb_c = get_embedding(chosen[:500])
                    emb_r = get_embedding(rejected[:500])
                    dist = cosine_sim(emb_c, emb_r)
                except Exception as e:
                    print(f"  WARN {persona} idx={int(fpath.split('/')[-1].split('.')[0])}: embed error — {e}", file=sys.stderr)
                    dist = -1.0

            length_diff = abs(len(chosen) - len(rejected))

            # Quality gate: at least one signal of divergence (D-007)
            if dist < 0.05 and length_diff < 30:
                idx = int(fpath.split("/")[-1].split(".")[0])
                print(f"  SKIP {persona} idx={idx}: no divergence signal (dist={dist} len_diff={length_diff})", file=sys.stderr)
                skipped += 1
                continue

            valid_rows.append({
                "prompt": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt_text},
                ],
                "chosen": chosen,
                "rejected": rejected,
                "_meta": {
                    "persona": persona,
                    "scenario_index": d.get("_scenario_index"),
                    "embedding_distance_chosen_rejected": dist,
                    "length_diff_chars": length_diff,
                },
            })

        # Sort by scenario index for deterministic output
        valid_rows.sort(key=lambda r: r["_meta"]["scenario_index"] if isinstance(r["_meta"]["scenario_index"], int) else 999)

        with open(outpath, "w") as f:
            for row in valid_rows:
                # Write DPO-format (strip _meta from the main object but keep it accessible)
                out = {"prompt": row["prompt"], "chosen": row["chosen"], "rejected": row["rejected"]}
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

        dists = [r["_meta"]["embedding_distance_chosen_rejected"] for r in valid_rows if r["_meta"]["embedding_distance_chosen_rejected"] >= 0]
        lengths_c = sorted([len(r["chosen"]) for r in valid_rows])
        lengths_r = sorted([len(r["rejected"]) for r in valid_rows])

        print(
            f"{persona}: {len(valid_rows)}/{len(files)} rows | "
            f"distances: min={min(dists):.3f} max={max(dists):.3f} mean={sum(dists)/len(dists):.3f} " if dists else f"{persona}: 0 valid",
            file=sys.stderr,
        )
        print(
            f"  chosen median={lengths_c[len(lengths_c)//2]} rejected median={lengths_r[len(lengths_r)//2]}" if lengths_c else "",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
