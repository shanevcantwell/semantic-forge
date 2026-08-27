# persona-dpo-scale — Decision Log

Stable handle: `persona-dpo-scale`
Cross-references: `persona-dpo-roadmap`, `persona-dpo-multiply/run2/`, P5 PoC (`persona-dpo-unsloth-poc/`)
Status: active data production (run0)

---

## D-001 — Scale target and structure

**When:** 2026-08-24T22:xx UTC  
**What decided:** Produce ~200 contrastive pairs per persona card (bramble, vex, marigold) = ~600 rows total for run0. Structure: each row is a chat-format DPO pair `{"prompt": [{"role":"system","content":...},{"role":"user","content":<scenario>}], "chosen": "...", "rejected": "..."}`.  
**Why:** P5 PoC trained on N=5 and correctly flagged signal unproven at that budget. 200 rows/card gives statistical power for in-sample margin improvement detection (≥3/5 → ≥140/200 with same directionality).  
**Evidence grounding:** `docs/experiments/persona-dpo-unsloth-poc/README.md` H0/P-margin predictions; run2 bramble row lengths/margins.

---

## D-002 — Scenario generation approach

**When:** 2026-08-24T22:xx UTC  
**What decided:** Generate fresh stimulus scenarios using gemma-3-12b-it directly at `192.168.137.1:1234` (LM Studio), rather than reusing the 5 existing multiply scenario templates verbatim or parameterizing them. Use a system prompt that enforces concrete specificity over abstract placeholders.  
**Why:** The existing 5 scenarios cover narrow domains (CSV parsing, KeyError fix, regex review, Rust opinion, job offers). At scale, repeating these with seed variance risks dataset homogeneity — models learn to map stimulus patterns rather than respond to persona conditioning. Fresh stimuli drawn from a broad distribution are more likely to generalize.  
**Evidence grounding:** run2 bramble chosen lengths 142→28 chars across scenarios; shorter prompts (s03, s04) produce weaker contrastive contrast in embedding distance (0.30 vs 0.17).

---

## D-003 — Issue #7 workaround for scaled production

**When:** 2026-08-24T22:xx UTC  
**What decided:** Use **strict one-call-per-process** (`--one N` pattern from multiply/run2) rather than fixing `integrations.py`. Rationale: the SK teardown defect is a forge-core issue requiring careful testing; production data generation should not block on that fix. Each scenario invocation = fresh process, single gemma API call for pair generation + single embedding call for distance scoring = 2 in-process tool calls per lifetime (within the proven-safe path).  
**Why:** The multiply README attempt-8 record showed isolation dodges the defect "ONLY while each process makes at most one pair call." Two calls per process re-enters the cancel-scope bug.  
**Evidence grounding:** `docs/experiments/persona-dpo-multiply/README.md` attempt 8 / vex idx=1 forensics (`CancelledError ... by <Task pending coro=<async_generator_athrow>>`).

---

## D-004 — Scenario generation system prompt ("seed")

**When:** 2026-08-24T22:xx UTC  
**What decided:** Use a structured system prompt for gemma that enforces: (1) concrete specificity with real details, (2) grounded-in-reality situations, (3) one scenario per line output format, (4) domain distribution across code/technical, debugging/analysis, career/opinion, productivity, philosophy.  
**Why:** Prevents abstract placeholders ("a user wants to optimize a hot loop") that produce homogeneous dataset rows; forces variety in stimulus difficulty and persona-relevant dimensions.  
**Evidence grounding:** Persona cards define trait_axes (response_length, warmth, hedging_softening, humor_density, register_markers) — scenarios must exercise these axes differently per persona.

---

## D-005 — Embedding scoring strategy at scale

**When:** 2026-08-24T22:xx UTC  
**What decided:** Score chosen/rejected embedding distance via `inference-host:8082` (embeddinggemma-300M-F32-pooled) using the same cosine similarity calculation as multiply/run1. One embedding call per pair (chosen+rejected concatenated with separator).  
**Why:** Maintains comparability with existing P3-b distance measurements (~0.15–0.30 range for bramble); provides a continuous quality signal to filter degenerate pairs at scale.  
**Evidence grounding:** `docs/experiments/persona-dpo-multiply/run1/bramble_pairs_run1_qwen38-superseded.jsonl` — all rows have real embedding_distance_chosen_rejected values; P3-b "light-up now rests on 6 independent measurements."

---

## D-006 — Persona card application order and system prompts

**When:** 2026-08-24T22:xx UTC  
**What decided:** Apply bramble → vex → marigold system prompts in sequence (per persona-dpo-roadmap P3-c prediction: median chosen-length orders bramble < vex < marigold). Use the verbatim `system_prompt:` blocks from each card YAML. Temperature = 0.8, matching run2 protocol.  
**Why:** Maintains consistency with existing data; allows triangle verdict comparison (bramble terse → marigold elaborate).  
**Evidence grounding:** Persona cards at `docs/experiments/persona-dpo-probe/cards/{bramble,vex,marigold}.yaml`; run2 medians bramble 116 / vex 397+ / marigold reference ~646 chars.

---

## D-007 — Quality gates for production rows

**When:** 2026-08-24T22:xx UTC  
**What decided:** Filter criteria per row: (1) _isError must be false, (2) chosen and rejected both non-empty after strip, (3) embedding_distance_chosen_rejected ≥ 0.05 (filters near-duplicate pairs), (4) |len(chosen) − len(rejected)| ≥ 10 chars (ensures meaningful length divergence). Log all filtered rows with reason code for auditability but don't include in training dataset.  
**Why:** At scale, gemma will occasionally produce degenerate outputs; these gates maintain dataset quality without requiring manual review of every row.  
**Evidence grounding:** run2 attempt-8 HALTED on vex idx=1 `_isError` tool result — instrumentation gap noted in README; this gate codifies the check upstream.

---

## D-008 — Prompt structure revision (chosen/rejected priming avoidance)

**When:** 2026-08-25T01:xx UTC  
**What decided:** Revised `gen_pairs.py` to ask gemma for two independent persona responses ("response_a" / "response_b") rather than explicitly framing them as "chosen (persona)" vs "rejected (generic helpful assistant)." The old prompt told gemma exactly what the contrastive structure was and even described how to make one response sycophantic — which risks overfitting the model to pipeline expectations rather than producing genuine personality variation. New prompt: "give two different ways you might respond to it, each as complete and independent responses." Scoring (embedding distance + length divergence) happens post-hoc.
**Why:** Models that know they're being evaluated for contrastive pairs may deliberately exaggerate differences or follow the rubric too literally. Unprimed generation produces more naturalistic data — the contrast emerges from comparing independent samples rather than being baked into instruction-following behavior. This aligns with how real DPO datasets are collected (humans don't know which response is "chosen" vs "rejected").
**Evidence grounding:** Anthropic HH methodology; observe that even run2 bramble pairs showed gemma making the "rejected" responses stereotypically sycophantic ("Okay! That's tricky! Let's explore...") — a pattern consistent with prompt priming rather than natural personality expression.

---

## D-009 — Aggregation and embedding distance computation

**When:** 2026-08-25T01:xx UTC  
**What decided:** Aggregated all valid pair files into persona-grouped `pairs.jsonl` using `aggregate_pairs.py`. Computed embedding distances via inference-host:8082 (embeddinggemma-300m-F32-pooled, 768-dim) — required restarting the server through llauncher's port-keyed API (`POST /start/8082` with `{
