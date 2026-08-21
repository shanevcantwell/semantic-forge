#!/usr/bin/env python3
"""Persona-DPO probe sampler (P2 `seed-samples`, see persona-dpo-roadmap).

Reads persona cards from ./cards/*.yaml and scenarios from ./scenarios/*.yaml
(paths resolved relative to THIS file, so it works from any CWD), then samples
k completions per (card x scenario) cell from an OpenAI-compatible chat endpoint.

Pipeline, not consumable: the endpoint and model are never assumed — the base URL comes
from --base-url / PERSONA_PROBE_BASE_URL and the served model identity is discovered at
runtime via GET {base}/models and recorded in every output record's meta block. Point this
at any future small -it checkpoint unchanged.

Output (default ./data/probe_samples.jsonl): one JSON object per line, schema:
  {card_id, scenario_id, sample_idx, system_prompt, user_prompt, completion,
   meta{endpoint, model_served, temperature, timestamp_utc}}

Honest-failure policy: if the endpoint is unreachable after patient retries (or a cell
yields no non-empty completion after its one retry), this script reports exactly what it
saw and exits non-zero. It never fabricates rows. Partial output left in place can be
inspected/removed; re-run simply overwrites.

Usage:
  python3 sample.py                          # defaults (local llama-server)
  PERSONA_PROBE_BASE_URL=http://host:9000/v1 python3 sample.py --k 4
  python3 sample.py --base-url http://inference-host:8081/v1 --model-id Qwen3.8-27B-IQ4_NL \
      --temperature 0.8 --max-workers 3

Deps: Python >= 3.9, PyYAML. (Stdlib only otherwise.)
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_URL = "http://inference-host:8081/v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{utc_now_iso()}] {msg}", file=sys.stderr, flush=True)


def http_json(url: str, payload: dict | None, timeout: float):
    """POST (payload not None) or GET JSON. Returns parsed body and HTTP status."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method="POST" if data is not None else "GET",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode()), resp.status


def discover_model(base_url: str, preferred: str | None, patience_retries: int, retry_delay: float):
    """GET {base}/models at runtime; return (served_id, all_ids, attempts_seen).

    Accepts OpenAI-style 'data:[{id}]' and llama-server-style 'models:[{model|name}]'.
    Raises RuntimeError with the last observed error if unreachable after patient retries.
    """
    url = base_url.rstrip("/") + "/models"
    last_err: str | None = None
    for attempt in range(1, patience_retries + 1):
        try:
            body, status = http_json(url, None, timeout=20)
        except Exception as e:  # URLError, HTTPError, socket.timeout, JSON decode...
            last_err = f"{type(e).__name__}: {e}"
            log(f"model discovery attempt {attempt}/{patience_retries} failed -> {last_err}")
            if attempt < patience_retries:
                time.sleep(retry_delay)
            continue
        entries = body.get("data") or body.get("models") or []
        ids: list[str] = []
        seen: set[str] = set()
        for e in entries:
            mid = e.get("id") or e.get("model") or e.get("name") if isinstance(e, dict) else str(e)
            if mid and mid not in seen:
                seen.add(mid)
                ids.append(mid)
        if not ids:
            last_err = f"HTTP {status} but no parseable model ids in body keys={list(body.keys())}"
            log(f"model discovery attempt {attempt}: {last_err}")
            time.sleep(retry_delay)
            continue
        served = preferred if (preferred and preferred in seen) else ids[0]
        if preferred and preferred not in seen:
            log(f"WARNING: requested model id '{preferred}' not in /models list {ids}; serving from list instead")
        log(f"model discovery OK after {attempt} attempt(s): served='{served}', listed={ids}")
        return served, ids, attempt
    raise RuntimeError(
        f"endpoint unreachable for model discovery: GET {url} failed "
        f"{patience_retries}x; last error -> {last_err}"
    )


def complete(base_url: str, model_id: str, system_prompt: str, user_prompt: str,
             temperature: float, max_tokens: int, read_timeout: float) -> tuple[str, str]:
    """One chat completion. Returns (completion_text, served_model_echo). Raises on failure."""
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    parsed, _status = http_json(url, body, timeout=read_timeout)
    try:
        choice = parsed["choices"][0]
        text = (choice.get("message") or {}).get("content", "")
        echo_model = str(parsed.get("model", ""))
        return text if text is not None else "", echo_model
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"unexpected chat response shape {type(e).__name__}: keys={list(parsed.keys())}")


def load_yaml_dir(path: Path) -> dict[str, dict]:
    """Load every *.yaml/*.yml in a dir into {id: doc}. Card files may be a mapping or list."""
    out: dict[str, dict] = {}
    for f in sorted(path.glob("*.y*ml")):
        try:
            import yaml  # local import so --help works without PyYAML
            docs = yaml.safe_load(f.read_text())
        except Exception as e:
            raise RuntimeError(f"failed to parse {f}: {e}") from e
        items = docs if isinstance(docs, list) else [docs]
        for d in items:
            if not isinstance(d, dict) or "id" not in d:
                raise RuntimeError(f"{f}: expected mapping(s) with an 'id' field")
            out[str(d["id"])] = (str(Path(f).name), d)  # keep source filename for reports
    return {k: v[1] for k, v in out.items()}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default=os.environ.get("PERSONA_PROBE_BASE_URL", DEFAULT_BASE_URL),
                   help=f"OpenAI-compatible base URL (default {DEFAULT_BASE_URL} or $PERSONA_PROBE_BASE_URL)")
    p.add_argument("--model-id", default=os.environ.get("PERSONA_PROBE_MODEL_ID"),
                   help="optional; pin a model id from GET /models. Default: discover and record what is actually served")
    p.add_argument("--k", type=int, default=4, help="completions per (card x scenario) cell (default 4)")
    p.add_argument("--temperature", type=float, default=0.8, help="sampling temperature (default 0.8: within-card variance)")
    p.add_argument("--max-tokens", type=int, default=512, help="per-completion cap (default 512)")
    p.add_argument("--connect-timeout", type=float, default=15.0, help="socket connect timeout seconds")
    p.add_argument("--read-timeout", type=float, default=300.0,
                   help="read timeout per completion; local llama-server may be slow/queued (default 300s)")
    p.add_argument("--retries", type=int, default=1, help="extra attempts per failed completion (default 1 = 'one retry')")
    p.add_argument("--retry-delay", type=float, default=5.0, help="backoff between retries (default 5s)")
    p.add_argument("--max-workers", type=int, default=3, help="concurrent completion requests (default 3; server queues internally)")
    p.add_argument("--discovery-retries", type=int, default=6, help="patient retries for /models probe before failing (default 6)")
    p.add_argument("--cards-dir", default=str(SCRIPT_DIR / "cards"))
    p.add_argument("--scenarios-dir", default=str(SCRIPT_DIR / "scenarios"))
    p.add_argument("--out", default=str(SCRIPT_DIR / "data" / "probe_samples.jsonl"))
    args = p.parse_args()

    try:
        import yaml  # noqa: F401
    except ImportError:
        log("FATAL: PyYAML is required (pip install pyyaml)")
        return 3
    if args.k < 1:
        log("FATAL: --k must be >= 1")
        return 3

    cards_dir, scenarios_dir = Path(args.cards_dir), Path(args.scenarios_dir)
    for d in (cards_dir, scenarios_dir):
        if not d.is_dir():
            log(f"FATAL: directory missing: {d}")
            return 3
    try:
        cards = load_yaml_dir(cards_dir)
        scenarios = load_yaml_dir(scenarios_dir)
    except RuntimeError as e:
        log(f"FATAL: {e}")
        return 3
    if not cards or not scenarios:
        log(f"FATAL: loaded 0 cards / {len(scenarios)} scenarios — nothing to sample")
        return 3

    missing = [cid for cid, c in cards.items() if not (c.get("system_prompt") or c.get("style_constraints"))]
    if missing:
        log(f"FATAL: cards without system_prompt/style_constraints: {missing}")
        return 3

    # System text contract: explicit `system_prompt` wins; else assemble from style_constraints.
    def card_system(c: dict) -> str:
        sp = (c.get("system_prompt") or "").strip()
        if not sp and c.get("style_constraints"):
            name = c.get("name", "assistant")
            sp = f"You are {name}.\n" + "\n".join(f"- {s}" for s in c["style_constraints"])
        return sp

    log(f"loaded {len(cards)} cards ({sorted(cards)}) x {len(scenarios)} scenarios ({sorted(scenarios)}), k={args.k}")

    # Model identity: discovered at runtime, never assumed. Patient retries = honest failure exit if down.
    try:
        model_id, listed_ids, disc_attempts = discover_model(args.base_url, args.model_id,
                                                             patience_retries=args.discovery_retries,
                                                             retry_delay=args.retry_delay)
    except (RuntimeError, Exception) as e:  # noqa: BLE001 — honest failure with what we saw
        log(f"FATAL: giving up on endpoint {args.base_url} -> {e}")
        return 4

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_cells = len(cards) * len(scenarios) * args.k
    t0 = time.time()
    failures: list[str] = []
    records: list[dict] = []

    def work(card_id: str, system_prompt: str, scenario_id: str, user_prompt: str, sample_idx: int):
        last_err = None
        for attempt in range(args.retries + 1):
            try:
                text, echo_model = complete(
                    args.base_url, model_id, system_prompt, user_prompt,
                    args.temperature, args.max_tokens, read_timeout=args.read_timeout)
                if not text.strip():
                    last_err = "empty completion"
                    log(f"{card_id}/{scenario_id}#{sample_idx}: attempt {attempt + 1} returned empty")
                    time.sleep(args.retry_delay)
                    continue
                served = echo_model or model_id
                return {
                    "card_id": card_id,
                    "scenario_id": scenario_id,
                    "sample_idx": sample_idx,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "completion": text,
                    "meta": {
                        "endpoint": args.base_url.rstrip("/"),
                        "model_served": served,
                        "temperature": args.temperature,
                        "timestamp_utc": utc_now_iso(),
                    },
                }
            except Exception as e:  # URLError/timeout/HTTPError/bad shape -> one retry per spec
                last_err = f"{type(e).__name__}: {e}"
                log(f"{card_id}/{scenario_id}#{sample_idx}: attempt {attempt + 1}/{args.retries + 1} failed -> {last_err}")
                if attempt < args.retries:
                    time.sleep(args.retry_delay)
        raise RuntimeError(f"{card_id}/{scenario_id}#{sample_idx}: no completion after "
                           f"{args.retries + 1} attempts; last error: {last_err}")

    jobs = [(cid, card_system(c), sid, s.get("text", "").strip(), i)
            for cid, c in cards.items()
            for sid, s in scenarios.items()
            for i in range(args.k)]

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futs = {ex.submit(work, *j): (j[0], j[2], j[4]) for j in jobs}
        done = 0
        for fut in as_completed(futs):
            cid, sid, idx = futs[fut]
            try:
                records.append(fut.result())
            except RuntimeError as e:
                failures.append(str(e))
                log(f"CELL FAILURE (leaving honest record below): {e}")
            done += 1
            if done % 5 == 0 or done == len(jobs):
                log(f"progress {done}/{len(jobs)} completions")

    # Deterministic row order: card x scenario x sample_idx.
    records.sort(key=lambda r: (r["card_id"], r["scenario_id"], r["sample_idx"]))
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    log(f"wrote {len(records)}/{total_cells} rows to {out_path} in {elapsed:.1f}s "
        f"(served='{model_id}', /models listed={listed_ids}, discovery_attempts={disc_attempts})")
    if failures:
        log(f"HONEST FAILURE EXIT: {len(failures)} cell(s) failed after retries:")
        for m in failures:
            log(f"  - {m}")
        return 2

    # Self-verification before declaring success.
    empty = [r for r in records if not r["completion"].strip()]
    expected_cells = {(c, s, i) for c in cards for s in scenarios for i in range(args.k)}
    got_cells = {(r["card_id"], r["scenario_id"], r["sample_idx"]) for r in records}
    if len(records) != total_cells or empty or got_cells != expected_cells:
        log(f"HONEST FAILURE EXIT: verification mismatch rows={len(records)}/{total_cells} "
            f"empty={len(empty)} missing_cells={sorted(expected_cells - got_cells)[:5]}")
        return 2
    log(f"VERIFIED OK: {len(records)} rows = {len(cards)} cards x {len(scenarios)} scenarios x k={args.k}; all completions non-empty")
    return 0


if __name__ == "__main__":
    sys.exit(main())
