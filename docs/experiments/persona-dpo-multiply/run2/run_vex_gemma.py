"""run2 vex (gemma-3 factory) — persona-DPO forge-multiply P3.
Cloned from the attempt-7 driver per H-attempt-8 (see ../README.md); diffs vs run1
driver: P payload, jsonl path + labels. SCENARIOS literals unchanged (post-comma-fix).

attempt-2 design (post probe-record reroute — see README "Instrumentation & probe record"):
  The permutate leg reproducibly returns empty texts for long persona payloads
  (3/3 empty rephrasings on ≥594-char payloads — verbatim spec P x2, prose-flattened
  P' x1 — while a ~107-char control concept returns non-empty texts). Out of scope here.
  Persona conditioning goes straight through the pair stage's context slot.

Phase A (smoke, 1 LLM call): generate_contrastive_pair with a fixed coding scenario,
  context = P. Doubles as SK liveness probe; full response dumped to disk on failure.
Phase B: one generate_contrastive_pair per hand-authored controlled scenario from
  module-level SCENARIOS (coding x3, casual x2), context = P unchanged.
Target ≥4 and ≤8 total pair rows; max ONE retry on empty row.

Budget: total LLM calls ≤ ~20.
"""

import asyncio
import json
import sys
import time
import statistics
from pathlib import Path
import urllib.request

from mcp import types
from mcp.server import Server
from semantic_forge.handlers import register_handlers


# ── composed persona payload P from vex.yaml (concise, all fields included) ───────────
P = """vex persona: Vex, the sardonic wit. Probes: humor as an independent axis — dry/sardonic wit with competent delivery; separates "funny" from both "pleasant" (marigold) and "flat" (bramble), incl. the hard witty-terse vs flat-terse sub-case.

Axes: response_length [40,120] words; warmth [2,3]; hedging_softening [0.0,1.0]/100w; humor_density [4,5]; register_markers: short snappy lines, fragments allowed; may end on a punchline instead of a summary; no bullets unless asked.

Constraints: dry wit baseline — at least one genuine pointed quip per reply, quick and self-aware, never mean-spirited or at anyone's expense; substance first, punchline second — answer plainly then let the joke land, never bury the point in a bit; 40-120 words of short snappy lines, fragments fine; no corporate warmth, no bullet-point essays; deadpan closer welcome.
"""

# Hand-authored controlled scenario stimuli (attempt-2): coding x3, casual x2 + 1 smoke = up to 6 rows.
SCENARIOS = [
    "Optimize a hot loop that parses CSV rows; the user wants the fastest straightforward fix.",
    "A Python script crashes with a KeyError on missing config entries; they want it fixed without new dependencies.",
    "Code review ask: is this regex doing what I think, and if not what is wrong?",
    "Quick opinion: is it worth learning Rust just to write CLI tools, or is that overkill?",
    "The user asks which of two job offers looks better based on a terse description; wants straight talk."
]


# ── HTTP helper: discover served model identity ───────────────────────────
def fetch_models(url: str) -> dict:
    """GET /v1/models and return parsed JSON response."""
    try:
        req = urllib.request.Request(f"{url.rstrip('/')}/models")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.load(resp)
    except Exception as e:
        return {"error": str(e)}


# ── core tool dispatcher ──────────────────────────────────────────────────
async def call_tool(server: Server, name: str, arguments: dict) -> dict:
    """Dispatch a tools/call request and return the raw result dict."""
    dispatch = server.request_handlers[types.CallToolRequest]
    req = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(
            name=name,
            arguments=arguments,
        ),
    )
    wrapped = await dispatch(req)
    result = getattr(wrapped, "root", wrapped)
    # Unwrap the MCP CallToolResult envelope: the tool's JSON payload lives in
    # content[0].text (see forge-ignition smoke_artifact.json for the real shape).
    is_err = bool(getattr(result, "isError", False))
    payload = None
    content = getattr(result, "content", None)
    if isinstance(content, list):
        for item in content:
            t = getattr(item, "text", None)
            if isinstance(t, str) and t.strip():
                try:
                    payload = json.loads(t)
                except (ValueError, TypeError):
                    payload = {"raw": t}
                break
    if payload is None:
        if hasattr(result, "model_dump"):
            payload = result.model_dump()
        else:
            payload = {"raw": str(result)}
    if isinstance(payload, dict):
        payload["_isError"] = is_err
    return payload


# ── write one JSONL record verbatim ───────────────────────────────────────
def append_pair_row(jsonl_path: Path, row: dict) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ── compute P3-c stats from jsonl ────────────────────────────────────────
def compute_stats(jsonl_path: Path):
    """Return (chosen_lens, rejected_lens, dist_vals, deadpan_vals)."""
    chosen_lens = []
    rejected_lens = []
    dist_vals = []
    deadpan_vals = []

    if not jsonl_path.exists():
        return chosen_lens, rejected_lens, dist_vals, deadpan_vals

    with open(jsonl_path, "r", encoding="utf-8") as jf:
        rows_list = [json.loads(line) for line in jf if line.strip()]

    for row in rows_list:
        chosen_text = row.get("chosen") or row.get("chosen_text") or row.get("chosen completion", "")
        rejected_text = row.get("rejected") or row.get("rejected_text") or row.get("rejected completion", "")
        if isinstance(chosen_text, str):
            chosen_lens.append(len(chosen_text))
        if isinstance(rejected_text, str):
            rejected_lens.append(len(rejected_text))
        dist = row.get("embedding_distance_chosen_rejected")
        if dist is not None:
            try:
                dist_vals.append(float(dist))
            except (ValueError, TypeError):
                pass
        tp = row.get("trajectory_profile") or row.get("TrajectoryProfile") or {}
        if isinstance(tp, dict) and "deadpan_score" in tp:
            dv = tp["deadpan_score"]
            if isinstance(dv, (int, float)):
                deadpan_vals.append(dv)
        if isinstance(row.get("deadpan_score"), (int, float)):
            deadpan_vals.append(row["deadpan_score"])

    return chosen_lens, rejected_lens, dist_vals, deadpan_vals


# ── write README entry ────────────────────────────────────────────────────
def write_readme(readme_path: Path, start_wall: float, llm_calls_used: int,
                 jsonl_path: Path, model_8081_id: str, model_8082_id: str,
                 rows_produced: int = 0, target: int = 8,
                 failed_count: int = 0) -> None:
    """Dated Results log entry mirroring the probe README style."""

    from datetime import datetime, timezone

    now_utc = datetime.now(timezone.utc)
    date_str = now_utc.strftime("%Y-%m-%d")
    time_str = now_utc.strftime("%H:%M:%S UTC")

    chosen_lens, rejected_lens, dist_vals, deadpan_vals = compute_stats(jsonl_path)

    chosen_med = statistics.median(chosen_lens) if chosen_lens else 0
    chosen_mean = statistics.mean(chosen_lens) if chosen_lens else 0
    rejected_med = statistics.median(rejected_lens) if rejected_lens else 0
    rejected_mean = statistics.mean(rejected_lens) if rejected_lens else 0
    dist_min = min(dist_vals) if dist_vals else "N/A"
    dist_max = max(dist_vals) if dist_vals else "N/A"
    deadpan_range = f"{min(deadpan_vals):.2f}–{max(deadpan_vals):.2f}" if deadpan_vals else "N/A"

    total_wall = time.time() - start_wall

    lines = []
    lines.append(f"- **run2** ({date_str} UTC) — persona-DPO multiply vex (gemma-3 factory)")
    lines.append(f"  Card: vex | Vex (the sardonic wit)")
    lines.append(f"  Moods: imperative, socratic")
    lines.append(f"  Scenario types: coding, casual")
    lines.append(f"  Served model identity (8081): {model_8081_id}")
    lines.append(f"  /v1/models @ :8082: {model_8082_id}")
    lines.append(f"  rows produced vs target: {rows_produced}/{target}")
    lines.append(f"  chosen median/mean length: {chosen_med:.1f}/{chosen_mean:.1f} chars")
    lines.append(f"  rejected median/mean length: {rejected_med:.1f}/{rejected_mean:.1f} chars")
    lines.append(f"  embedding_distance_chosen_rejected range: [{dist_min}, {dist_max}]")
    lines.append(f"  TrajectoryProfile deadpan_score range: {deadpan_range}")
    if failed_count:
        lines.append(f"  failed rows: {failed_count}")
    lines.append(f"  LLM calls used: {llm_calls_used}")
    lines.append(f"  Wall time: {total_wall:.1f}s")

    if failed_count:
        try:
            with open(jsonl_path, "r", encoding="utf-8") as jf:
                first_line = jf.readline().strip()
                first_row = json.loads(first_line)
                first_err = first_row.get("error", "unknown")
                lines.append(f"  First failure verbatim: {first_err}")
        except Exception:
            lines.append(f"  First failure verbatim: <read error>")

    existing = ""
    if readme_path.exists():
        existing = readme_path.read_text()

    new_entry = f"\n\n---\n### run2 — {date_str} UTC\n" + "\n".join(lines)
    content = existing + new_entry if existing else "\n".join(lines)

    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(content)
    print(f"Wrote README entry to {readme_path}")


# ── main pipeline ─────────────────────────────────────────────────────────
async def main() -> None:
    start_wall = time.time()
    all_rows = []

    # ── CLI argument parsing (first thing, before any side effects) ─────
    import argparse as _argmod
    parser = _argmod.ArgumentParser(
        description="run2 vex (gemma-3 factory) — persona-DPO forge-multiply P3"
    )
    parser.add_argument(
        "--one",
        type=int,
        default=None,
        help="Run a single scenario index N (0-4). If omitted, runs full pipeline as today.",
    )
    args = parser.parse_args()
    one_n = args.one
    if one_n is not None and (one_n < 0 or one_n > 4):
        print(f"Usage: --one N where N is 0-4 (got {one_n})", file=sys.stderr)
        sys.exit(2)
    CLI_ONE_N = one_n

    base_dir = Path("/home/node/github/shanevcantwell/semantic-forge")
    jsonl_path = base_dir / "docs/experiments/persona-dpo-multiply/run2/vex_pairs_gemma3.jsonl"
    readme_path = base_dir / "docs/experiments/persona-dpo-multiply/README.md"

    server = Server("run2-vex-gemma")
    await register_handlers(server)
    dispatch = server.request_handlers[types.CallToolRequest]

    # ── discover served model identity via live /v1/models ─────────────────
    inference_8081 = fetch_models("http://inference-host:8081/v1")
    inference_8082 = fetch_models("http://inference-host:8082/v1")
    def _model_id(d):
        if isinstance(d, dict) and d.get("data"):
            return ", ".join(m.get("id", "?") for m in d["data"])
        return str(d.get("error", "unreachable")) if isinstance(d, dict) else str(d)

    model_8081_id = _model_id(inference_8081)
    model_8082_id = _model_id(inference_8082)

    llm_calls = 0
    max_llm_calls = 20

    # ── light up SK leg for this run (one-field flip; BLOCKED branch reverts on early abort) ──
    cfg_path0 = base_dir / "semantic_forge_config.json"
    with open(cfg_path0, "r") as f:
        cfg0 = json.load(f)
    cur_ep = (cfg0.get("semantic_kinematics") or {}).get("endpoint")
    if not cur_ep:
        cfg0.setdefault("semantic_kinematics", {})["endpoint"] = \
            "/home/node/.local/bin/semantic-kinematics-mcp"
        with open(cfg_path0, "w") as f:
            json.dump(cfg0, f, indent=2)
        print("Flipped semantic_kinematics.endpoint for run1.")
    else:
        print(f"endpoint already set: {cur_ep}")

    # ── If --one N mode, skip Phase A and run only scenario N ──────────
    if CLI_ONE_N is not None:
        idx = CLI_ONE_N
        scenario = SCENARIOS[idx]

        # Execute the existing Phase B per-row body semantics for scenario N
        context_text = P

        row_written = False
        for attempt in range(2):
            if llm_calls >= max_llm_calls:
                print(f"LLM call budget exhausted at attempt {attempt+1}")
                break

            resp_pair = await call_tool(server, "generate_contrastive_pair", {
                "scenario": scenario,
                "context": context_text,
            })
            llm_calls += 1
            print(f"generate_contrastive_pair (attempt {attempt+1}) resp keys:",
                  list(resp_pair.keys()) if isinstance(resp_pair, dict) else "non-dict")

            is_empty = False
            garbage_reason = None

            if isinstance(resp_pair, dict):
                chosen = resp_pair.get("chosen") or resp_pair.get("chosen_text") or \
                         resp_pair.get("chosen completion", "")
                rejected = resp_pair.get("rejected") or resp_pair.get("rejected_text") or \
                           resp_pair.get("rejected completion", "")

                if not chosen or not chosen.strip():
                    is_empty = True
                    garbage_reason = "chosen is empty/missing"
                elif not rejected or not rejected.strip():
                    is_empty = True
                    garbage_reason = "rejected is empty/missing"
                # tool-level error flag only — pair texts may legitimately contain the word 'error'
                if isinstance(resp_pair, dict) and resp_pair.get("_isError"):
                    is_empty = True
                    garbage_reason = "pair response flagged _isError by tool"

            if is_empty:
                if attempt == 0:
                    print(f"  Empty (attempt 1): {garbage_reason}; retrying...")
                    continue
                else:
                    print(f"  Still empty after 2 attempts: {garbage_reason}")
                    fail_row = {
                        "concept": P[:45] + "...",
                        "scenario": scenario if isinstance(scenario, str) else f"scenario_{idx}",
                        "context": context_text,
                        "error": garbage_reason,
                        "chosen": "",
                        "rejected": "",
                    }
                    append_pair_row(jsonl_path, fail_row)
                    row_written = True
                    break
            else:
                resp_pair["_scenario_index"] = idx
                resp_pair["_generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                all_rows.append(resp_pair)
                append_pair_row(jsonl_path, resp_pair)
                row_written = True
                print(f"run2 vex idx={idx} OK — chosen {len(chosen)} chars rejected {len(rejected)} chars")
                # Print embedding distance if available
                dist = resp_pair.get("embedding_distance_chosen_rejected")
                if dist is not None:
                    try:
                        print(f"dist={float(dist):.4f}")
                    except (ValueError, TypeError):
                        pass
                break

        # On double failure, append fail row and BLOCKED message
        if not row_written:
            print(f"BLOCKED run2 vex idx={idx}: pair production failed after 2 attempts")
            # DID NOT revert config endpoint per --one mode spec
            # Did NOT call write_readme per --one mode spec
            sys.exit(0)

        # Normal exit after single scenario in --one mode
        sys.exit(0)

    # ── Phase A: smoke ────────────────────────────────────────────────────
    all_rows = []

    # attempt-2 Phase A: smoke via the pair stage itself — also the SK liveness probe.
    if llm_calls < max_llm_calls:
        for attempt in range(2):
            if llm_calls >= max_llm_calls:
                break

            resp = await call_tool(server, "generate_contrastive_pair", {
                "scenario": SCENARIOS[0],
                "context": P,
            })
            llm_calls += 1
            print("smoke pair resp keys:", list(resp.keys()) if isinstance(resp, dict) else type(resp).__name__)

            chosen = rejected = ""
            sm_failed = None
            if isinstance(resp, dict):
                chosen = str(resp.get("chosen") or "")
                rejected = str(resp.get("rejected") or "")
                if resp.get("_isError"):
                    sm_failed = "smoke pair response flagged error by tool (_isError)"
            else:
                sm_failed = f"smoke pair response was not a dict: {type(resp).__name__}"
            if sm_failed is None and not (chosen.strip() and rejected.strip()):
                sm_failed = "empty chosen/rejected in smoke pair"

            if sm_failed:
                if attempt == 0:
                    print(f"  Smoke empty/error (attempt 1): {sm_failed}; retrying..."
                        )
                    continue
                # Second failure — fall through to existing BLOCKED path unchanged
                with open(jsonl_path.parent / "smoke_error_full.json", "w") as f:
                    json.dump(resp if isinstance(resp, dict) else {"raw": str(resp)}, f,
                              indent=2, default=str)
                print("Phase A FAILED:", sm_failed)
                cfg_path = base_dir / "semantic_forge_config.json"
                with open(cfg_path, "r") as f:
                    cfg = json.load(f)
                cfg["semantic_kinematics"]["endpoint"] = None
                with open(cfg_path, "w") as f:
                    json.dump(cfg, f, indent=2)
                print("Reverted endpoint to null per BLOCKED rule (no pairs produced).")
                write_readme(readme_path, start_wall, llm_calls, jsonl_path,
                             model_8081_id, model_8082_id,
                             rows_produced=0, target=8, failed_count=1)
                return

            # Success
            row = resp if isinstance(resp, dict) else {"raw": str(resp)}
            row["_scenario_index"] = 0
            row["_generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            all_rows.append(row)
            append_pair_row(jsonl_path, row)
            print(f"smoke pair OK — chosen {len(chosen)} chars, rejected {len(rejected)} chars")
            break


    # ── Phase B: small batch ──────────────────────────────────────────────

    # (scenario stimuli are the module-level SCENARIOS; no per-rephrase seeding in attempt-2)

    for idx, scenario in enumerate(SCENARIOS[1:], start=1):
        if len(all_rows) >= 8:
            break

        context_text = P

        row_written = False
        for attempt in range(2):
            if llm_calls >= max_llm_calls:
                print(f"LLM call budget exhausted at attempt {attempt+1}")
                break

            resp_pair = await call_tool(server, "generate_contrastive_pair", {
                "scenario": scenario,
                "context": context_text,
            })
            llm_calls += 1
            print(f"generate_contrastive_pair (attempt {attempt+1}) resp keys:",
                  list(resp_pair.keys()) if isinstance(resp_pair, dict) else "non-dict")

            is_empty = False
            garbage_reason = None

            if isinstance(resp_pair, dict):
                chosen = resp_pair.get("chosen") or resp_pair.get("chosen_text") or \
                         resp_pair.get("chosen completion", "")
                rejected = resp_pair.get("rejected") or resp_pair.get("rejected_text") or \
                           resp_pair.get("rejected completion", "")

                if not chosen or not chosen.strip():
                    is_empty = True
                    garbage_reason = "chosen is empty/missing"
                elif not rejected or not rejected.strip():
                    is_empty = True
                    garbage_reason = "rejected is empty/missing"
                # tool-level error flag only — pair texts may legitimately contain the word 'error'
                if isinstance(resp_pair, dict) and resp_pair.get("_isError"):
                    is_empty = True
                    garbage_reason = "pair response flagged _isError by tool"

            if is_empty:
                if attempt == 0:
                    print(f"  Empty (attempt 1): {garbage_reason}; retrying...")
                    continue
                else:
                    print(f"  Still empty after 2 attempts: {garbage_reason}")
                    fail_row = {
                        "concept": P[:45] + "...",
                        "scenario": scenario if isinstance(scenario, str) else f"scenario_{idx}",
                        "context": context_text,
                        "error": garbage_reason,
                        "chosen": "",
                        "rejected": "",
                    }
                    append_pair_row(jsonl_path, fail_row)
                    row_written = True
                    break
            else:
                resp_pair["_scenario_index"] = idx
                resp_pair["_generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                all_rows.append(resp_pair)
                append_pair_row(jsonl_path, resp_pair)
                row_written = True
                break

        await asyncio.sleep(0.1)

    # ── Compute final stats and write README ──────────────────────────────
    rows_produced = 0
    failed_count = 0
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as jf:
            jsonl_lines = jf.readlines()
        for line in jsonl_lines:
            row = json.loads(line.strip()) if line.strip() else {}
            if row.get("error"):
                failed_count += 1
            else:
                rows_produced += 1

    write_readme(readme_path, start_wall, llm_calls, jsonl_path,
                 model_8081_id, model_8082_id,
                 rows_produced=rows_produced, target=8, failed_count=failed_count)

    total_wall = time.time() - start_wall
    print(f"\n=== Run complete ===")
    print(f"Rows produced: {rows_produced} (plus {failed_count} failures)")
    print(f"LLM calls used: {llm_calls}/{max_llm_calls}")
    print(f"Wall time: {total_wall:.1f}s")
    print(f"Artifact: {jsonl_path}")


if __name__ == "__main__":
    asyncio.run(main())