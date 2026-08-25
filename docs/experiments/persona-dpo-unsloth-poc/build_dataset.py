#!/usr/bin/env python
"""Build bramble DPO v0 dataset from the run2 gemma pair rows. Deterministic; no model calls.

Sources (all verbatim, see README "Data construction rules"):
  system   <- cards/bramble.yaml `system_prompt:` block
  user     <- the row's LOGGED `prompt` field verbatim (uniform rule for all rows).
              Provenance note: forge's pair stage is structured generation — the response schema's
              `prompt` key is model-filled (`result.get("prompt", scenario)`, handlers.py), so the logged
              text is what the instrument scored/coherency-checks, not necessarily verbatim driver input.
              Rows 0–3 match their driver SCENARIOS literal exactly (identity echo); row 4 is a
              model-materialized expansion of an abstract scenario driver literal (see README provenance note).
              Driver literals are AST-extracted and cross-checked per row; the outcome is recorded as
              `prompt_source` in the provenance file — never silently substituted.
  chosen/rejected <- run2/bramble_pairs_gemma3.jsonl rows verbatim

Outputs: dataset/bramble_dpo_v0.jsonl + dataset/provenance_bramble_v0.json (both tracked; .gitignore override).
Any provenance drift exits non-zero with the mismatch printed — never silently substitutes.
"""
import ast
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

POC = Path(__file__).resolve().parent          # .../docs/experiments/persona-dpo-unsloth-poc
ROOT = POC.parents[2]                          # repo root (semantic-forge)
CARD = ROOT / "docs/experiments/persona-dpo-probe/cards/bramble.yaml"
DRIVER = ROOT / "docs/experiments/persona-dpo-multiply/run2/run_bramble_gemma.py"
ROWS = ROOT / "docs/experiments/persona-dpo-multiply/run2/bramble_pairs_gemma3.jsonl"

import yaml  # pyyaml is present in the studio venv (unsloth CLI dep)


def main() -> None:
    card = yaml.safe_load(CARD.read_text(encoding="utf-8"))
    system = card["system_prompt"]
    assert system.startswith("You are Bramble"), "card drift: bramble.yaml system_prompt changed"

    tree = ast.parse(DRIVER.read_text(encoding="utf-8"))
    scenarios = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "SCENARIOS" for t in node.targets
        ):
            scenarios = ast.literal_eval(node.value)
    assert scenarios is not None and len(scenarios) == 5, (
        f"driver drift: module-level SCENARIOS literal missing or wrong length ({len(scenarios)})"
    )

    raw_lines = [l for l in ROWS.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [json.loads(l) for l in raw_lines]
    assert len(rows) == 5, f"row count drift: expected 5, got {len(rows)}"
    line_no = {id(x): i + 1 for i, x in enumerate(rows)}

    out, prov_rows = [], []
    seen = set()
    for r in sorted(rows, key=lambda x: int(x["_scenario_index"])):
        si = int(r["_scenario_index"])
        assert si not in seen and 0 <= si < 5, f"row {si}: index out of range/duplicate"
        seen.add(si)

        logged = (r.get("prompt") or "").strip()
        assert logged, f"row {si}: empty logged prompt field — unusable"
        if scenarios[si] == logged:
            user_text, psrc = logged, "driver_literal_match"
        else:
            user_text, psrc = logged, "logged_field_expansion"
        assert not r.get("_isError"), f"row {si}: _isError is true — unusable"
        assert r["chosen"].strip() and r["rejected"].strip(), f"row {si}: empty completion(s)"

        out.append(
            {
                "prompt": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_text},
                ],
                "chosen": r["chosen"],
                "rejected": r["rejected"],
            }
        )
        prov_rows.append(
            {
                "scenario_index": si,
                "run2_jsonl_line": line_no[id(r)],
                "generated_at": r.get("_generated_at"),
                "prompt_source": psrc,
                **({"driver_literal": scenarios[si]} if psrc == "logged_field_expansion" else {}),
                "chars": {"chosen": len(r["chosen"]), "rejected": len(r["rejected"])},
            }
        )

    ds = POC / "dataset"
    ds.mkdir(exist_ok=True)
    (ds / "bramble_dpo_v0.jsonl").write_text(
        "\n".join(json.dumps(o, ensure_ascii=False) for o in out) + "\n", encoding="utf-8"
    )

    prov = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "card": str(CARD.relative_to(ROOT)),
        "system_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
        "driver": str(DRIVER.relative_to(ROOT)),
        "rows_source": str(ROWS.relative_to(ROOT)),
        "n_rows": len(out),
        "provenance": prov_rows,
    }
    (ds / "provenance_bramble_v0.json").write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")

    cc = [len(o["chosen"]) for o in out]
    rc = [len(o["rejected"]) for o in out]
    n_expansion = sum(1 for p_ in prov_rows if p_["prompt_source"] == "logged_field_expansion")
    print(
        f"OK {len(out)} rows -> dataset/bramble_dpo_v0.jsonl | "
        f"system_sha256={prov['system_sha256'][:12]}… | chosen chars min..max = {min(cc)}..{max(cc)} | "
        f"rejected chars min..max = {min(rc)}..{max(rc)} | prompt_source: "
        f"{len(prov_rows)-n_expansion} literal-match, {n_expansion} logged-field expansion"
    )


if __name__ == "__main__":
    main()
