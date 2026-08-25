# Persona-DPO P5-machinery PoC — bramble DPO on gemma-3-12b-it (Unsloth, in-container)

- **Handle:** `persona-dpo-unsloth-poc` (cross-reference by handle, not title)
- **Formed:** 2026-08-24 UTC (predictions below pre-date any training observation — lab-notebook floor)
- **Cross-references:** `persona-dpo-roadmap` P4/P5 · multiply README attempt-8 record (bramble 5/5 gemma rows, commit `d53a791`) · handoff `persona-dpo-handoff` deltas-since-edition-2 · issue #7 (SK teardown — data phase parked)

## Framing & scope

Operator direction 2026-08-24: "let's try bramble as proof of concept" — read as: the
**complete bramble dataset is the PoC payload for the Unsloth training lane** just provisioned in-container.
Prove the P5 *training leg* end-to-end: pair data → QLoRA-DPO on gemma-3-12b-it (4-bit base) →
banked adapter artifact, with the serving infrastructure untouched.

**Scope boundary (GATE-AXES):** this PoC certifies the *training axis only*. Adapter→merged-bf16→GGUF
export is a **separate toolchain leg** (llama.cpp conversion; PEFT v0.20 moved its merge API off
`PeftModel` — `merge_and_unload` lives on the tuner layer in this install) and gets its own probe once
the adapter artifact exists. A green training run certifies nothing about export, and vice versa.

**Target selection note (named scoped deviation):** roadmap P4 defers target pick to post-triangle
measurement — that deferral stands for *flavor* selection across cards. This PoC deliberately uses the
factory lineage (`google/gemma-3-12b-it`, the model family serving `inference-host:8081` as IQ4_NL) so a
future merged artifact is directly comparable on the same lane, and because bramble's rows are its data.

**Why now:** vex/marigold data production is parked on #7 + the operator fork (single-shot re-entry vs
repair-first). The training machinery is the larger unproven risk; proving it on 5 rows costs nothing
the data phase needs, and de-risks P4/P5 before more pair calls are spent.

## Lane ground (read this turn, pre-run)

- Unsloth venv `~/github/Unsloth/.venv` (clone HEAD `0998656`, installed as source): torch
  2.10.0+cu128 (CUDA build 12.8), unsloth 2026.8.19, transformers 5.5.0, peft 0.20.0, trl 0.24.0,
  bitsandbytes 0.50.1; `torch.cuda.is_available()` True → Quadro RTX 8000 visible in-container (i9 host GPU).
- API shape in this version: `FastLanguageModel.from_pretrained(...)` + `get_peft_model(...)`; DPO runs on
  **plain TRL** (`DPOTrainer`/`DPOConfig`) — `PatchDPOTrainer()` is a no-op stub here. Dataset row format per
  repo tests: `{"prompt","chosen","rejected"}` (chat-format prompt supported by current TRL).
- Base weights source: `unsloth/gemma-3-12b-it-bnb-4bit` on HF (public; verified via HF API this turn) — the
  roadmap lifecycle's "Unsloth pre-quantized bnb-4bit form". **Corrected post-resolution (E-disc):** the loader
  takes that name *directly* to the pre-quant repo (7.26 GiB tensors, `ALLOW_PREQUANTIZED_MODELS` default) — no
  runtime NF4 of the 25 GiB base; `from_pretrained(...)` returns a `(model, tokenizer)` tuple in this version.
- VRAM at formation: 34,232 MiB free / 49,152 (llama-servers resident ≈14 GB — `:8081` gemma IQ4_NL holds
  12.5 GB incl. its unified-KV pool; **`:8081` is also pi's own serving lane — it must not be stopped to make room**).
- Disk: `/home/node` at 98% (20 G free) and `/tmp` on the same overlay → **HF cache pinned to
  `/mnt/storage/hf-cache`, artifacts under `/mnt/storage/persona-dpo-poc/`** (432 G free). Unsloth's own
  readonly-redirect fallback targets /tmp — do not let it fire.

## Environment resolution (pre-run deltas E1–E6, supersedes venv/disk lines above)

Operator steer 2026-08-24: stop one-off dependency surgery; align once against the repo's declared set.
All steps below are uv-managed package operations with cited grounding — **zero site-packages edits** in the
effective environment (a hand patch made to the retired `.venv` is documented history, not state).

- **E1:** effective env = official installer venv `~/.unsloth/studio/unsloth_studio` (operator ran
  `./install.sh`; Python 3.13.14, torch 2.11.0+cu130, uv-managed, no pip). The hand-built
  `~/github/Unsloth/.venv` (torch cu128) is **retired** for this experiment.
- **E2:** aligned to the repo's own pins (`studio/backend/requirements/extras-no-deps.txt`, applied `--no-deps`
  per that file's comment): trl 0.24.0 → **0.23.1**, peft 0.20.0 → **0.18.1** (file comments: peft 0.19 causes
  export subprocess shutdown issues).
- **E3:** mergekit==0.1.4 + its two declared-but-absent deps (immutables==0.21, scipy) installed by manifest diff.
  Grounding: TRL's DPO import path hard-imports mergekit at module level
  (`trl/trainer/callbacks.py → mergekit_utils`), yet mergekit is **undeclared in both trl and unsloth metadata**
  — upstream packaging gap; 0.1.4 is the only modern release (PyPI releases end at it).
- **E4:** pydantic 2.13.4 → **2.10.6** (+ core 2.27.2). mergekit's module-level
  `pydantic.create_model(... Task[torch.Tensor] ...)` fails schema generation under 2.13.x; its declared range is
  `~=2.10.6` and upstream `main` (checked) still carries the unguarded call — no released fix exists, so the
  floor of that range = the only version matrix with grounding. Probe: `import unsloth` stays green afterwards
  (Unsloth stack integrity under the downgrade verified, not assumed).
- **E5:** disk plan revised at gate time: `/mnt/storage` is mounted **read-only** in-container (432 G free but
  unusable — rw re-mount would be a host-side operator call; not needed here). Cache → `~/hf-cache`, artifacts →
  `~/persona-dpo-poc/`, on the /home/node overlay: 23 G available vs ≈8.5 G required (7.26 GiB download + adapter
  + later GGUF export headroom).
- **E6:** load order is load-bearing — `import unsloth` *before* any TRL trainer import:
  `unsloth/_gpu_init.py:243` runs `fix_trl_vllm_ascend()` (documented fix for transformers-5 tuple returns from
  `_is_package_available`; guarded by repo drift test `tests/test_import_fixes_drift.py`) and the RL replacements
  patch TRL's DPO collators. The training path is plain `trl.DPOTrainer` *as patched at import*
  (introspected: `DPOConfig` stays stock trl; `DPOTrainer` resolves to `UnslothDPOTrainer`, standard signature).
- **E7:** two execution-shape findings from free pre-flight probes (no training observation involved):
  (a) passing `bf16=True` fails transformers 5.5's static validation under torch 2.11+cu130 even though
  `torch.cuda.is_bf16_supported()` is True and a live bf16 CUDA op succeeds — the check chain misfires on this pair,
  so DPOConfig omits the flag; QLoRA precision rides on the checkpoint's own `bnb_4bit_compute_dtype`.
  (b) run with CWD = experiment dir, free of stale `unsloth_compiled_cache/` dirs: the import layer materializes a
  fresh cache in-cwd (gitignored here); other cwds carried old caches from earlier forge runs and routed imports
  through them.
- **E8:** HF blob egress via squid 403s the xet domains (`cas-server.xethub.hf.co`, `us.aws.cdn.hf.co`) while
  direct container egress to them is open; with proxy vars unset for the process + `HF_HUB_DISABLE_XET=1`, the
  client takes huggingface.co → signed AWS-CDN bridge (302) and streams. Pre-run weight prefetch already landed
  under this recipe: 7.4 GiB in ~/hf-cache (6.3 min; readback of both safetensors + tokenizer present). The train
  script needs no network for the model now — these vars are the pin for any future fetch (stage-2, export leg,
  fresh machines).
- **E9:** factory identity event (ground truth at run-prep time): `:8081` live `/v1/models` reports
  `Qwen3.8-27B-IQ4_NL` (llauncher label agreed this time — not metadata drift; PID cross-check:
  nvidia-smi's 35 GiB consumer == llauncher's :8081 server). P-coexist baseline is re-grounded to the identity
  live at run start: whatever serves `:8081`/`:8082` before training must serve identically after. Operator-factory
  call, surfaced; not acted on.
- **Probe readback (post-resolution, pre-training):** `import unsloth` + `from trl import DPOConfig, DPOTrainer`
  clean; RTX 8000 visible from PyTorch (CUDA build 13.0); 45.4 GiB VRAM free at last check (serving lane
  currently idle — P-coexist verifies identity at run end regardless).
- **Serving coexistence still absolute:** `:8081` is pi's own head; it must not be stopped to make room.

## Data construction rules (deterministic; `build_dataset.py`)

Source rows: `persona-dpo-multiply/run2/bramble_pairs_gemma3.jsonl` (5 rows, scenario indices {0..4} ×1).
Per row i: `system` ← `cards/bramble.yaml` `system_prompt:` block **verbatim** (the product conditioning —
what sample.py sent in P2; forge's compressed P payload is the instrument slot, not target conditioning);
`user` ← **the row's logged `prompt` field verbatim** (uniform rule for all rows); `chosen`/`rejected` ←
row fields verbatim. Artifact: `dataset/bramble_dpo_v0.jsonl` + provenance per row (card, scenario_index,
run2 jsonl line, `prompt_source`). Nested `.gitignore` override tracks the jsonl/run logs (run1/run2 pattern).

**Provenance note — row 4 conditioning is model-materialized (forensically verified pre-run):** forge's pair
stage is structured generation whose response schema carries a *model-filled* `prompt` key (`result.get(
"prompt", scenario)`, `semantic_forge/handlers.py`). Rows 0–3: logged prompt == driver SCENARIOS literal
(identity echo — sourcing rule makes no difference). Row 4: the driver literal is an abstract description
("which of two job offers looks better…"); gemma materialized concrete offer detail in its own response and
the completions reference exactly that detail, so [user turn → chosen/rejected] is a self-contained coherent
pair. Row 4 stays under the uniform rule (logged field), tagged `prompt_source=logged_field_expansion`:
scenario diversity (opinion/judgment axis) is precisely what this dataset otherwise lacks, and every scenario
is synthetic by construction — row 4's concrete facts simply live inside its own logged text. The builder
cross-checks all rows against AST-extracted driver literals and tags each outcome, so drift lands as labeled
provenance, not silent substitution.

## H0 & pre-run predictions (before any training observation)

**H0:** Unsloth QLoRA-DPO on gemma-3-12b-it (4-bit base), 5 bramble preference pairs, runs to completion
in-container alongside the serving llama-servers and yields a banked adapter showing DPO movement.

- **P-load:** model + tokenizer load under 4-bit without OOM; trainable params < 2% of total (LoRA-only);
  download+load wall < ~15 min.
- **P-train:** ≥3 epochs / 15 optimizer steps complete with per-step logged losses, no NaN/traceback in the
  tail, and final-step loss < first-step loss (signal present at lr 1e-6; N=5 — a flat trajectory is banked
  as "machinery green, signal unproven at this budget", not repaired en route).
- **P-margin** (added pre-run; supersedes loss-only readout): per-row Δ = mean-token logprob(chosen) −
  mean-token logprob(rejected) on the student policy, measured before and after the update on the same 5 rows.
  Prediction: aggregate mean Δ increases AND ≥3 of 5 rows improve. A flat/decreased margin is banked as
  "update did not move in-sample preference" — a null, recorded with its reasoning, not repaired en route.
- **P-artifact:** adapter dir under `/mnt/storage/persona-dpo-poc/` with weights < ~200 MB, re-loadable via
  `PeftModel.from_pretrained` (readback in the same run before exit).
- **P-coexist:** `:8081` and `:8082` still serving live `/v1/models` at run end — no OOM cascade into the
  serving lane.

## Budget & halt rules (typed exits)

- Hyperparameters pinned: max_seq_length 2048; LoRA r=16, alpha=32, target_modules all-linear, dropout 0;
  DPO beta 0.1, lr 1e-6, epochs 3, batch 1/grad-accum 1, logging every step, no intermediate checkpoints.
- VRAM gate: proceed only if ≥ 25 GB free at script start (measured in-script via torch, printed). Else exit 2.
- Disk gate (E5): start requires ≥10 G free on /home/node; post-download assert ≥6 G before the first optimizer
  step. Else exit 3 with readback.
- OOM / traceback in train log → **HALT**, report last 40 lines as-is. One retry of the same command max;
  never improvise API shape changes mid-run — an unexpected kwarg rejection is a HALT with readback (the
  v0.20 merge-API shift means my prior API memory is unreliable here).
- Download failure → one retry, then **BLOCKED** (network/proxy) with last log lines.
- Disk-gate breach (see above) → stop, print free-space readback, do not continue into training.

## Export leg (named follow-up — NOT this run)

Probe: first choice in this version = native CLI: `unsloth export <checkpoint> <out> --format gguf
--quantization q4_k_m` (ExportBackend; verified in-tree at `unsloth_cli/commands/export.py`, formats:
merged-16bit / merged-4bit / gguf / lora). Fallback only if the CLI path fails: PEFT tuner-layer merge +
llama.cpp `convert_hf_to_gguf.py` (+ `gguf` pip pkg). Note for that fallback: this venv has peft 0.18.1 (E2), so
re-ground the merge API on 0.18 before use — no prior memory of it is banked here.
Serving the artifact (replacing/supplanting `:8081`) is an operator decision — that lane is pi's own head.
