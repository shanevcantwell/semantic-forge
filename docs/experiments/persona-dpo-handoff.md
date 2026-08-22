# Persona-DPO — Handoff for Clean Context

- **Handle:** `persona-dpo-handoff` (cross-reference by handle, not title)
- **Written:** 2026-08-22 ~02:5x UTC (≈ 2026-08-21 ~21:5x US/Mountain — the operator's clock; container runs UTC)
- **Close anchor:** commit `1a199ea` on `main`, pushed to origin (`github.com/shanevcantwell/semantic-forge`)

## If you are a cold reader starting here

This effort probes the space of *personalities* that DPO-style preference selection can separate
out on small instruction-tuned models, via a synthetic-data pipeline. This repo (semantic-forge)
is the working + tracking venue per operator direction; the factory itself is also this repo's
package. Phases **P1 `forge-ignition` and P2 `seed-samples` are closed with verdicts** (dated
in the roadmap). The next executable phase is **P3 `forge-multiply`**; it does not start until
the operator gives go — ask, don't auto-start after a context break.

Read first: `docs/experiments/persona-dpo-roadmap.md` (authoritative phase tracker: framing,
standing constraints, checkpoint-vocabulary/lifecycle correction, P1–P6 with verdicts and H0s).
Everything below is operational context the roadmap deliberately doesn't carry.

## Durable pointers

| what | where |
|---|---|
| Roadmap (phases, verdicts, open decisions) | `docs/experiments/persona-dpo-roadmap.md` |
| Persona cards (bramble / vex / marigold — DPO-tractable trait axes) | `docs/experiments/persona-dpo-probe/cards/*.yaml` |
| Scenarios s01–s05 | `docs/experiments/persona-dpo-probe/scenarios/` |
| Reusable sampler (env/CLI-configurable; default max_tokens=512 — **use 2048 for this model family**) | `docs/experiments/persona-dpo-probe/sample.py` |
| Dataset: 60 rows = 3 cards × 5 scenarios × k=4, run3 @ max_tokens=2048 | `docs/experiments/persona-dpo-probe/data/probe_samples.jsonl` |
| Archived contrast: 34-row run2 @ 512 (CoT budget-exhaustion regime) + per-cell results log | same data dir; dated entries in the probe tree's `README.md` |
| Forge smoke rerun + artifact (`permutate_phrasing`, rephraser-only stage) | `docs/experiments/forge-ignition/{rerun_smoke.py,smoke_artifact.json}` |
| Runnable venv (Python 3.12.3; gitignored) | repo `.venv/` — activate with `source .venv/bin/activate` |

## Standing operational constraints (do not relearn these the hard way)

1. **One background agent at a time against inference-host:8081.** The llama-server there runs
   `--unified-kv` with a finite total KV pool; two concurrent agent sessions once killed both
   lanes mid-run ("Context size has been exceeded"). Serialize dispatches: launch one, wait for
   its completion notification, then the next. This was an explicit operator limit "for now".
2. **Subagent model tier:** detail-bearing contracts (file:symbol recon, debugging) run on no-spec
   / `llauncher-i9` (= whatever 8081 serves — Qwen3.8-27B-IQ4_NL at handoff time). Tiny models are
   *subjects or targets of study*, never workers: a qwen3-vl-4b probe (shane-pc-lmstudio,
   192.168.137.1) produced structurally-plausible but symbol-level-unverified recon — treat such
   output as leads to verify, not fact.
3. **Every worker contract must carry output discipline** (cap reads/greps/logs at ~50 lines or
   byte caps): the KV pool is a shared scarce resource; long unbounded reads are how lanes die.
4. **Endpoints:** chat `http://inference-host:8081/v1` (OpenAI-compatible; api key: any non-empty
   string, "not-used" works); embeddings `http://inference-host:8082/v1`, model
   `embeddinggemma-300M-F32-pooled`, 768-dim. **Live `/v1/models` is the ground truth for served
   identity — never assume which checkpoint answers**; llaunch status metadata was stale (listed
   :8082 as `-nonpooled` while the live API served `-pooled`; unresolved at handoff).
5. **Never touch `docs/absurdism/`** in this repo — operator material, untracked by design; exclude
   from every commit scope.
6. **No second large checkpoint on inference-host while :8081 serves.** Operator direction 2026-08-22:
   specifically `huihui-ai_Qwen3-Coder-Next-abliterated-Q4_K_L` (confirmed present in the live llaunch
   inventory that day) would saturate the RTX-8000 pool alongside an in-flight server. The remote branch
   `origin/qwen3-coder-next` is named for this model — do not infer git-work meaning from that name; its 2
   commits (6859b7a, 258af90) stay unverified and operator-deferred, no destructive git ops without an
   explicit ask. Consequence for P3/P4: plans must reuse already-running servers (:8081 chat / :8082
   embeddings); any new endpoint needs its own explicit go.

## Parked / open (deliberately, not lost)

- **frontier_advisor critique of the roadmap:** operator asked for it; the tool/service was not
  resolvable from this vantage (no MCP tool matching "frontier"; no agent file under
  `~/.pi/agent/agents/`). Parked until the operator points at where it lives.
- **Roadmap open decisions** (DPO-select semantics a/b; P6 judge strategy) stay deferred by design
  until P3/P5 data exists — do not close them early or bake assumptions into P3.
- llaunch inventory notes: ~24 models on inference-host, most stopped; P4 target selection is over
  **HF safetensors checkpoints** (e.g., `google/gemma-4-E4B-it` / `-qat`), not llaunch GGUF names —
  see roadmap vocabulary section. The RTX-3090 box for Unsloth work is 192.168.137.1 (P5).

## Known rough edges (flagged, unfixed; operator's call)

- llaunch status vs live API name drift on :8082 (see constraint 4); the "local" llaunch node is
  registered but unreachable at localhost:8765 and pollutes every all-node query.
- semantic-forge remote prints a branch-protection note ("changes must be made through a PR") yet
  accepted direct pushes as of handoff — future git lanes should expect that noise, escalate only
  if a push is actually denied.
- llaunch uptime display unreliable (reports "up 0s" persistently); trust PIDs and live API over it.

## Resume plan (waterfall; operator paces)

1. Confirm go for **P3 `forge-multiply`** with the operator. Inputs are all committed: 60-row
   dataset + cards/scenarios/sampler above; forge runs from `.venv`. P3's H0 lives in the roadmap —
   write any pre-run predictions into its P3 section before observing results (lab-notebook floor).
2. Expected P3 friction to check early: forge's native axis is behavioral/structural (Grammatical
   Mood Multiplier); persona conditioning enters via prompt slots, and the SK diversity leg has
   never been exercised (`semantic_kinematics.endpoint` currently `null` in
   `semantic_forge_config.json`) — first full-stage run will light that path for the first time.
3. P4–P6 proceed per roadmap; Unsloth on the 3090 box is human-driven (scripted, not automated).
