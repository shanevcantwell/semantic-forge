# Persona-DPO — Handoff for Clean Context (edition 2)

- **Handle:** `persona-dpo-handoff` (cross-reference by handle, not title)
- **Written:** 2026-08-23 ≈21:3x UTC (≈ 15:3x US/Mountain) — external-ground time at authoring (container + GitHub API + pc LM Studio host all agree); see clock note under rough edges
- **Close anchor:** `1b02707` = last pushed state at writing time ("docs(handoff): tighten Written line to real date phrasing" — an edition-1 polish committed between editions; expected provenance, not drift). This edition ships in the bank commit that follows it (subject begins "persona-dpo P3: bramble run1 data close"); verify head before acting. Edition 1 lived at commits through `ed2417d`.

## If you are a cold reader starting here

Same effort as edition 1: probe the space of *personalities* that DPO-style preference selection can
separate out on small instruction-tuned models, via synthetic data; semantic-forge is working + tracking
venue and factory. **P1/P2 closed with verdicts** (roadmap). P3 `forge-multiply`: **the bramble run1
dataset phase is COMPLETE** — 5 rows, scenarios {0..4} ×1 each, chosen<rejected in 5/5, all SK
embedding distances real (P3-b lit-up), attempt-6 crash root-caused and worked around via a driver
`--one N` process-isolation mode; full provenance in the multiply README. **vex + marigold replication
runs are NOT started** — that is an open operator fork. Do not auto-start anything after a context
break: ask.

Read first, in order: `docs/experiments/persona-dpo-roadmap.md` (phase tracker, P3-c prediction,
checkpoint vocabulary) → `docs/experiments/persona-dpo-multiply/README.md` (attempt log, H-blocks with
pre-run predictions, composite-stimulus annotation, attempt-7 record) → this doc.

## State deltas since edition 1 (absorb these before acting)

1. **Gate-A repair committed (`5839a98`).** `semantic_forge_config.json` now carries per-request
   `extra_body: {chat_template_kwargs:{enable_thinking:false}}` for the :8081 target client; forge's
   `InferenceBackend.extra_body` passes it through (config.py/llm.py +4 lines each). This is the fix
   for Qwen3 thinking-mode burning the token budget — root cause of attempts 0–2 (evidence in the
   multiply README, H-think P1/P2 CONFIRMED). Do not remove. Parse base rate post-fix on single-sample
   calls: 6/6 first-attempt success so far (N small; bounded by README record). Operator's standing:
   Qwen3.8 stays as the "fast" factory for this phase shape — a swap is a decision, not a default.
2. **SK stdio leg is LIVE and data-producing** (`5839a98` endpoint flip + same-hour spacy repair).
   `semantic_kinematics.endpoint = /home/node/.local/bin/semantic-kinematics-mcp`; launcher shebang
   pins **system python 3.12**; `en_core_web_sm 3.8.0` installed user-site
   (`~/.local/lib/python3.12/site-packages`) with operator-approved `--break-system-packages`.
   **Container rebuild ⇒ re-install required** (wheel: explosion/spacy-models release
   `en_core_web_sm-3.8.0-py3-none-any`; 764-vocab load verified). Do NOT create a venv for SK — it
   would not be used by the launcher.
3. **Evidence-trackability:** repo-wide `.gitignore` ignores `*.log`/`*.jsonl`; a nested override at
   `docs/experiments/persona-dpo-multiply/run1/.gitignore` (committed `8e0b096`) re-includes them for
   that dir. Everything else in the repo still needs explicit `git add -f`.
4. **Subagent lane directive (operator, 2026-08-23 session):** workers dispatch pinned to
   `shane-pc-lmstudio/lmstudio-pc` (RTX-3090 box 192.168.137.1:1234, LM Studio UI). The alias serves
   whatever model is loaded in the UI — at writing time nemotron-3.5-lightning-30b-a3b; a dispatch
   failing "No models loaded"/"Model unloaded" means ask the operator to load one (lms load), don't
   retry-loop. Known pc-lane behavior: report fidelity degrades on long multi-part contracts — this
   session saw out-of-scope demo runs, a self-revert via `git checkout --`, and summaries contradicting
   disk. Standing mitigation: tight single-purpose contracts with halt rules + the orchestrator verifies
   EVERY worker claim against git/disk readbacks before banking anything (trust-but-verify is load-bearing here).
5. **Forge defect triaged, unfixed:** any SECOND consecutive in-process tool call through the SK stdio
   client gets CancelledError from its teardown task (`integrations.py:181 model_status`, evidence in
   attempt-6 log + README record). Workaround proven 4/4: driver `--one N` mode = one process per
   scenario. **File an issue; fix integrations.py before P4-scale multi-call batching** — this is a named
   precondition, not optional cleanup.
6. **:8082 identity drift (edition-1 rough edge) RESOLVED:** live API and llaunch both read
   `embeddinggemma-300M-F32-pooled`; the stale label was llaunch metadata at the time.

## Durable pointers

| what | where |
|---|---|
| Roadmap (phases, P3-c prediction, checkpoint vocabulary/lifecycle) | `docs/experiments/persona-dpo-roadmap.md` |
| Multiply run log: attempts 0–7, H-blocks, annotation, verdict blocks | `docs/experiments/persona-dpo-multiply/README.md` |
| **Bramble dataset artifact (5 rows, {0..4} ×1)** | `run1/bramble_pairs_run1.jsonl` (trackable via nested override) |
| Tainted archives — plumbing-only rows with provenance in README | `run1/bramble_row_attempt4_composite_stimulus.jsonl` · `run1/bramble_row_attempt7_demo_idx0.jsonl` |
| Per-attempt logs (1–6 + attempt-7 idx 1–4) | `run1/run_attempt*.log` |
| Driver with committed Gate-A-era fixes + `--one N` mode | `run1/run_bramble_run1.py` |
| Persona cards (bramble/vex/marigold, shared 5-axis schema) | `docs/experiments/persona-dpo-probe/cards/*.yaml` |
| Scenarios s01–s05 · reusable sampler (use max_tokens=2048 for this family) | `docs/experiments/persona-dpo-probe/scenarios/`, `.../sample.py` |
| P2 60-row probe dataset + run3 results log | `docs/experiments/persona-dpo-probe/data/probe_samples.jsonl` + probe README |
| Runnable venv (Python 3.12.3; gitignored) | repo `.venv/` (`source .venv/bin/activate`) |

## Standing operational constraints (carried from edition 1 unless marked changed)

1. **Serialize heavy consumers of inference-host:8081** — llama-server runs `--unified-kv`, finite
   pool; two concurrent agent sessions once killed both lanes ("Context size has been exceeded").
   pc-lane subagents themselves don't touch :8081, but any two things making forge-style API calls at
   the same time (e.g., two experiment runs) are still forbidden. Explicit operator limit "for now".
2. **(CHANGED)** Detail-bearing worker contracts pin to `shane-pc-lmstudio/lmstudio-pc` per delta 4 —
   supersedes edition-1's no-spec/llauncher-i9 guidance for subagent labor; tiny models remain
   subjects/targets, never workers.
3. Every worker contract carries output discipline (cap reads/greps/logs ~50 lines or byte caps) and a
   named budget/halt rule.
4. Endpoints: chat `http://inference-host:8081/v1` (identity per live API; config carries thinking-off),
   embeddings `http://inference-host:8082/v1` pooled 768-dim. **Live `/v1/models` is ground truth for
   served identity** — it caught the edition-1 drift and remains the read to take before any run.
5. Never touch `docs/absurdism/` — operator material, untracked by design; exclude from every commit.
6. No second large checkpoint on inference-host while :8081 serves (operator direction 2026-08-22;
   re-routed swaps need explicit go). `origin/qwen3-coder-next` remains operator-deferred, no destructive
   git ops without an ask.

## Parked / open (deliberate)

- **frontier_advisor critique of the roadmap: CLOSED** — operator direction 2026-08-22 "nothing to do
  with this task." Do not re-offer as a rescope option.
- Roadmap open decisions (DPO-select semantics a/b; P6 judge strategy) stay deferred by design — but
  note the data state moved: **bramble-only** multiply data exists; both decisions want the full
  triangle before closing, not vibes.
- Issue board spine: #5 wire loop (filters into keep/discard gate — "SK live populates distance data,
  nothing discards" recon note stands), #6 permutate diversity leg dead `_get_embedding` call, and the
  NEW SK-teardown defect from delta 5 to be filed.

## Resume plan (waterfall; operator paces)

0. Ground checks first: `git log --oneline -3` + status vs close anchor above; both `/v1/models`
   identities live; pc lane liveness (`curl :1234/v1/chat/completions`, model `lmstudio-pc`, "pong");
   SK leg under system python (`python3 -c "import spacy; spacy.load('en_core_web_sm')"`).
1. **The operator fork:** (a) vex + marigold replication — same `--one N` protocol, fresh jsonl/README
   entries per card (~5 calls/card); on completion the P3-c triangle verdict becomes testable against
   the roadmap's pre-banked prediction; or (b) hold at bramble pending a factory swap decision or the
   integrations.py repair. If (a): write the H-block(s) into the multiply README **before** running —
   lab-notebook floor, non-negotiable in this repo.
2. File the SK-teardown issue whenever convenient; it gates P4-scale multi-call runs.
3. P4 target selection waits on triangle data + roadmap decision 1/2 resolved by measurement.

## Known rough edges (carried)

- llaunch "local" node registered but unreachable at localhost:8765 — pollutes every all-node query.
- semantic-forge remote prints a branch-protection note yet accepts direct pushes; escalate only if a
  push is actually denied.
- llaunch uptime display unreliable ("up 0s"); trust PIDs and live APIs over it.
- (NEW) pc lane model load state is operator-managed UI state — dispatch failures there are ask-the-operator, not spin-retry.
- (NEW 2026-08-23) **In-session clock-regime disagreement:** an early `date -u` read this session returned ≈11:19 UTC; at authoring time the container, GitHub API Date header, and pc LM Studio host epoch all agree on ≈21:3x UTC — either a long real-time gap between turns or a host-clock step (mechanism unadjudicated from inside). Consequence: trust day-level dating and intra-regime ordering (all attempt `_generated_at` stamps are internally consistent); distrust fine-grained cross-turn diffs. External ground reachable through proxy: `curl -sI https://api.github.com | grep -i '^date'`.
