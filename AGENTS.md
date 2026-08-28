# semantic-forge — repo instructions

**This repo is the behavioral-fine-tuning data-generation toolkit.** MCP-first, implementing the Grammatical Mood Multiplier methodology for generating synthetic DPO/ORPO training data that reinforces healthy model behaviors through structural diversity rather than punishment-based alignment.

## Pointer-first (opinion locality — do not restate canon, point to it)

Cross-repo doctrine lives at:
- **Operating constitution** (procedural-first, pointer-based transfer, durable emission, signals, honest-failure, Hard Never-Dos) — `~/github/shanevcantwell/operating-doctrine/pi/AGENTS.md` (also mirrored at `~/AGENTS.md`)
- **Orchestrator doctrine** (decompose-and-delegate, dispatch defaults, persona) — `~/github/shanevcantwell/operating-doctrine/pi/SYSTEM.md`
- **Ground physics** (data-plane invariants + development-plane disciplines) — `~/github/shanevcantwell/operating-doctrine/ground-physics/GROUND_PHYSICS.md`
- **Doctrine canon + pointer-map** — `~/github/shanevcantwell/operating-doctrine/decisions/ADR-CON-0001-*`

Repo-local:
- **Persona-dpo experiment roadmap** — `docs/experiments/persona-dpo-roadmap.md` (stable handle: `persona-dpo-roadmap`)
- **Repo-local ADRs** — `docs/adrs/proposed/` (series prefix: `ADR-001-{slug}`)
- **Lab-notebook experiments** — `docs/experiments/{handle}/` — one dir per experiment, one per operator direction

## What this repo does

- Generates **contrastive pairs** (chosen/rejected) for DPO/ORPO fine-tuning from behavioral concepts (temporal trust, uncertainty acknowledgment, reasoning before action, anti-sycophancy, etc.)
- MCP tool surface (`permutate_phrasing`, `generate_scenario`, `generate_contrastive_pair`, `score_completion`, `validate_diversity`, `validate_trajectory`, `build_dataset`, `dataset_stats`)
- CogSec adversarial judge for structural-manipulation scoring (not correctness)
- Integrates with `semantic-kinematics-mcp` for embedding diversity validation and `prompt-prix` for cross-model evaluation
- Currently active line: **persona-dpo synthetic personality separation** (roadmap above)

## Working here

- **Lab-notebook floor.** Hypotheses written before observation; null results banked with the same care as confirmations. Each experiment is a durable record, not a transient window.
- **Experiments live in `docs/experiments/{handle}/`.** One subdirectory per operator direction; `README.md` is the experiment log, `*.py` scripts are the runnable record.
- **Handoff files** — `docs/experiments/persona-dpo-handoff.md` carries cross-session continuity for the active persona-dpo line.
- **Conservative write.** This is research code under active experiment — changes carry their decision record, never silent rewrites of the lab floor.
- **No ADRs in `docs/ADRs/`** (that path doesn't exist here). Repo-local ADRs live at `docs/adrs/proposed/` with series prefix `ADR-001-{slug}`. Cross-repo constitution ADRs (`ADR-CON-*`) live at `operating-doctrine/decisions/` per ADR-CON-0009.

## Live caveats a cold session must know

- **unsloth\_compiled\_cache/** and **run\_bramble\_poc.log** are untracked on main. The cache dir is build output — not code. Should be gitignored or pruned.
- **No local feature branches.** Only `main` is local; remote-tracking refs for `docs/adr-namespace-migration`, `docs/adr-valid-friction-sk-mcp2`, `qwen3-coder-next`, `qwen35-27b` exist but local branches do not.
- **Single experiment line active.** persona-dpo P5-machinery PoC (commit 64ffe71 banks env deltas from HF egress recipe + `:8081` identity event). Prior P1–P4 are closed or superseded; see the roadmap for closure records.
- **Two open issues** that aren't on the experiment line: #6 (permutate\_phrasing dead sk\_client call) and #7 (SK stdio client lifecycle CancelledError) — both are framework bugs, not experiment work.

## Inference topology (host x runtime matrix)

| | i9 (`inference-host`) | pc (`192.168.137.1` / shane-pc) |
|---|---|---|
| **lmstudio** | *(unused)* | `shane-pc-lmstudio/lmstudio-pc` — model served via LM Studio UI; worker contracts pin here |
| **llauncher** | `:8081` chat + `:8082` embeddings (identity per live `/v1/models`) | gemma-3-12b-it-IQ4_NL on llauncher-pc at `shane-pc:8081` |

- **Unsloth training lane** runs in Docker on Windows at `shane-pc:8000` — reachable over the network, not local to this container.
- **GPU precision matters:** shane-pc RTX 3090 (Ampere) has native BF16; inference-host RTX 8000 (Turing) is FP16-only. QLoRA/DPO training runs on shane-pc for proper mixed-precision paths — do not route training workloads to inference-host.
- **Serialize heavy consumers of :8081** — llama-server uses `--unified-kv`, finite pool; two concurrent agent sessions can kill both lanes ("Context size has been exceeded"). One background agent at a time.

## Session state continuity

- `docs/experiments/persona-dpo-handoff.md` is the primary cold-resume record for the active experiment line — read it before acting on persona-dpo work.
- Local feature branches are not used; all work happens on `main`. Remote-tracking refs exist but should not be checked out without operator direction.
