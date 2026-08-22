# Persona-DPO Synthetic Data Pipeline — Experiment Roadmap

- **Handle:** `persona-dpo-roadmap` (stable; cross-reference this handle, not the title)
- **Repo of record:** semantic-forge (working + tracking venue per operator direction, 2026-08-21)
- **Started:** 2026-08-21 (UTC)
- **Framing:** Probe the space of *personalities* that DPO-style preference selection can
  separate out on small instruction-tuned models. Synthetic data is the instrument:
  persona-conditioned prompts → sampled responses from tiny -it checkpoints →
  chosen/rejected pairs → selection among personality flavors.

## Standing constraints (invariant for all phases)

1. **Pipelines, not consumables.** Every endpoint and model choice is pluggable via
   config/env (`base_url` + model id). Nothing hardcoded to a specific checkpoint or a
   service that may not exist at re-run time.
2. **Target model deliberately TBD** ("a future question"). Early phases probe the space;
   selection decisions are deferred until phase-3 data exists, then made on measurements,
   not vibes.
3. Lab-notebook floor: hypotheses (H0) written *before* observing results; null results
   banked with the same care as confirmations.

## Checkpoint vocabulary & lifecycle (corrected 2026-08-21)

"Training a quant" collapses three different things; this record keeps them separate:

- **Quantized inference artifact** — llama.cpp k-quants (e.g., GGUF `Q4_K_M`): weights
  packed in blocks with per-block scales, no autograd path through PyTorch, PEFT cannot
  attach adapters to it. *Export target / serving form only*, not a training input.
  (llama.cpp's experimental fine-tuning over GGUF is not part of this pipeline.)
- **QLoRA** — the base model loads frozen via bitsandbytes NF4 quantization (dequantized
  on the fly in the forward pass); LoRA/DPO adapters train in bf16 *on top*. The thing
  being trained is never quantized; only the frozen base is. Input: HF safetensors
  (`-it`), or Unsloth's pre-quantized bnb-4bit form of the same weights (NF4 step done
  ahead). Worked example at start: `google/gemma-4-E4B-it` /
  `unsloth/gemma-4-E4B-it-bnb-4bit`.
- **QAT checkpoint** (`gemma-4-E4B-it-qat`) — bf16 safetensors trained with fake
  quantization in the loop so they degrade less when converted to Q4. They go through the
  ordinary pipeline; a different *starting checkpoint*, reasonable when the final
  deliverable is GGUF.

Lifecycle line for P4→P6: **HF safetensors (`-it`, optionally `-qat`) → QLoRA/DPO in
Unsloth → merge adapter → export GGUF → serve on llauncher.** The GGUF entries in the
llaunch inventory are where a model *ends up* after training, not where it starts.

## Phases

### P1 `forge-ignition` — get semantic-forge running  *(done 2026-08-21)*
End state: repo installs in a venv, config points at live OpenAI-compatible endpoints
(chat on inference-host:8081, embeddings on :8082), and the smallest meaningful pipeline
stage executes end-to-end producing one real artifact.
- **H0 (pre-run):** prior recon's claims about structure (mcp.py / dataset.py / config.py /
  CogSec filtering) are roughly accurate; the repo can be pointed at an OpenAI-compatible
  endpoint with a small provider-agnostic change rather than its documented ollama/vLLM
  backends only.
- **Predicted friction:** unrun-dusty code — import/config drift since last commit (2026-06);
  backend config may not accept our endpoints natively.
- **Verdict (2026-08-21):** H0 held — structure real, venv import clean on first check,
  and one smoke stage (`permutate_phrasing`, rephraser-only) ran end-to-end against live
  endpoints producing a scored artifact. Boundary: the semantic_kinematics diversity leg is
  gated off by `endpoint=null` (by-design no-op); full-stage exercise comes with P3.
  Rerun script + artifact preserved under `docs/experiments/forge-ignition/`. Note: live
  :8082 reports itself as `-pooled` via its API while the llaunch agent's status listed it
  as `-nonpooled` — record trusts the live call; metadata drift flagged to operator.

### P2 `seed-samples` — authoring-oracle probe  *(done 2026-08-21)*
Design 2–3 DPO-tractable persona cards + 3–5 personality-surfacing scenarios; sample k=4
completions per (card × scenario) from the llauncher-i9 endpoint (model identity recorded
per run via `/v1/models`, never assumed). Reusable sampler script — it will later be
pointed at tiny -it checkpoints unchanged. Record: `docs/experiments/persona-dpo-probe/`.
- **Verdict (2026-08-21):** run3 closed the sweep 60/60 @ max_tokens=2048; run2's
  deterministic card-specific empties at 512 resolved as CoT budget exhaustion
  (H0-rerun supported, both regimes archived). Register divergence quantified on survivors:
  median length bramble ~133 / vex ~425 / marigold ~646 chars; no chain-of-thought leakage
  into content. Full per-cell data + dated results log in the probe README.

### P3 `forge-multiply` — multiply through semantic-forge  *(pending)*
Feed P2 seeds into the factory's rephrase/scenario/pair stages to confirm it generalizes
from its native behavioral/structural axis (Grammatical Mood Multiplier) to persona
conditioning with prompt-slot changes, not a structural rewrite.
- **H0:** pair generation generalizes to persona conditioning without core edits; what is
  missing is the persona slot + a persona-relevant scoring signal, both additive.

### P4 `target-selection` — pick QLoRA/DPO target(s) for RTX-3090  *(pending)*
Selection is over **HF safetensors checkpoints**, not llaunch inventory names — the
inventory entries (e.g., `gemma-4-E4B.Q4_K_M`) are post-export serving forms, where a
checkpoint lands after P5 rather than what P4 picks. Candidate example pair at start:
`google/gemma-4-E4B-it`, with the `-qat` variant as the sensible starting checkpoint when
the deliverable is GGUF (see vocabulary above). The 3090 box is 192.168.137.1; decision
made *after* P3 on what pair data actually separates.

### P5 `unsloth-manual` — run QLoRA/DPO via Unsloth, human-driven  *(pending)*
Scripted but not automated: manual runs on the RTX-3090 box against P3 pair data with a
P4 target, following the lifecycle line above exactly — HF safetensors (or Unsloth's
bnb-4bit form) → QLoRA/DPO in bf16 over the frozen NF4 base → merge adapter → export
GGUF → llauncher serves it for evaluation.

### P6 `eval-loop` — our model runs + evaluates unsloth experiments  *(pending)*
llauncher-i9 endpoint (inference-host:8081) serves fine-tuned checkpoints; persona-axis
scoring selects among flavors. Judge strategy is an open decision from the design
discussion: LLM judge (forge's CogSec pattern retargeted to persona fidelity),
deterministic semantic-kinematics axis z-scores, or hybrid (axes first cut, judge breaks
ties). Resolved when P5 checkpoints exist and there is something to score.

## Open decisions (from 2026-08-21 design discussion; deferred by design)

1. **"DPO-selecting" semantics:** (a) generate pairs → fine-tune small -it bases →
   select among resulting flavors, vs (b) no fine-tuning — pairwise tournament
   (Bradley-Terry/Elo over persona-conditioned outputs of the existing model) to find which
   personas a tiny model can actually realize. Phases 1–3 feed both; they fork at P4/P6.
2. **Judge strategy** as above (P6).

## Cross-references

- Experiment record, probe data + sampler: `docs/experiments/persona-dpo-probe/` (P2)
- Related repos: llamagotchi-pet-mcp (existing persona text — `PET_SYSTEM_PROMPT` in its
  react runner; seed material), semantic-kinematics-mcp (measurement primitives for the P6
  judge axis-signal option), thought-vault-integration (`embedding_bridge.py` as reference
  llama-server embeddings client).
- Handoff for clean-context resume (P1/P2 closed): `docs/experiments/persona-dpo-handoff.md`
