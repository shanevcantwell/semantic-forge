# Persona-DPO P3 forge-multiply — run log

> **Annotation 2026-08-23 UTC (post-attempt-4):** all run entries below labeled attempt ≤4 were generated before a verified defect in the driver's `SCENARIOS` literal — five adjacent string literals lacked commas, so Python implicit concatenation produced a one-element list. Consequences: (a) Phase B never executed in any attempt (zero scenario calls); (b) every smoke row was conditioned on a 463-char composite stimulus containing all five scenarios inlined, not SCENARIOS[0] alone — such rows confirm Gate-A plumbing (non-empty pair, no _isError) but are NOT usable as bramble-direction or P3-c register data. The sole attempt-4 smoke row is archived verbatim at `run1/bramble_row_attempt4_composite_stimulus.jsonl` and excluded from the dataset artifact (`run1/bramble_pairs_run1.jsonl`). Fix committed same day (four commas); first clean-scenario data arrives in attempt 5+.

### H-attempt-6 (formed 2026-08-23 UTC, pre-run — before any attempt-6 observation)

Attempt-5's Phase-A failure was a stochastic malformed-JSON degeneration from the pair stage
(`Expecting ',' delimiter: line 4 column 254 (char 608)` in `run1/smoke_error_full.json`), not a Gate-A regression:
attempt-4's smoke parsed cleanly under identical config. Base rate so far at temp 0.7 with thinking-off:
1 clean / 1 malformed out of 2 content-returning calls. Driver gap addressed in this commit: Phase A now
carries one retry, mirroring Phase B's per-row idiom (smoke_error_full.json holds the last response only).

- **P1:** attempt-6 produces ≥1 parseable smoke pair and completes through Phase B → 5 rows total
  (scenarios 0–4), each with non-empty chosen/rejected; wall time < ~2 min.
- **P2 (base-rate probe, low-confidence):** if BOTH smoke calls fail to parse at temp 0.7, the pair-stage
  malformed-JSON rate under Gate-A is high enough (~≥50% on N=4) that single-retry is insufficient and a
  structured-output repair path is required before P4 data work — bank as falsification of "retry suffices",
  not as an instrument fault.

## Results log

- **attempt 0 (2026-08-22 UTC, ~14:37–14:46)** — driver prepared + one smoke pass that failed as an
  **instrument fault, not a pipeline observation**. Zero pairs; config ended at `endpoint=null`
  (correct pre-run state). Four script defects found and fixed before attempt 1:
  (a) the /v1/models identity probe double-appended the path (`/v1/v1/models` → 404);
  (b) the MCP CallToolResult envelope was parsed as the payload — the tool JSON sits in
      `content[0].text`; (c) `permutate_phrasing` returns `rephrasings` as dicts `{mood, text}` but
      the parser accepted string elements only → false Phase-A failure; (d) the script never
      performed the SK endpoint flip itself. The single permutate call to :8081 completed with a
      normal tool result (no exception — the README write only happens after that), so the
      response was received and discarded by local parsing, not by the pipeline. No P3-a..d
      verdict rests on this attempt: no consumable observation was made.

### Instrumentation & probe record (post-attempt-1)

- **attempt 1 observation (2026-08-22 UTC)** — full-P `permutate_phrasing` returned both mood slots with
      empty `text` (`""`) in an otherwise valid payload; no tool error. Reproduced 3/3 on ≥594-char payloads
      (verbatim spec P x2 incl. the attempt-1 driver call, prose-flattened P' x1); a ~107-char control concept
      returns non-empty texts for both moods — so the block is payload-length/content-driven, not notation. Envelope + shape parsing proven
      correct by direct probes: short concept → 2 non-empty rephrasings; driver's own `P` (608 chars,
      imported verbatim, not retyped) → 2 empty texts, twice-reproducible. Config flip and auto-revert
      verified working end-to-end in the failure path.
- **H-think** (formed 2026-08-22 UTC, post attempt-2, before its observations) — root-cause hypothesis for BOTH failure
      classes: :8081 serves Qwen3 with thinking mode ON by default; the pair stage's hardcoded max_tokens=1000 is consumed
      by reasoning tokens leaving `content` empty/non-JSON (permutate at 2048 shows milder degeneration — empty text fields).
      Confirmed server-side trait: a trivial probe returns `reasoning_content` alongside content on default calls, and
      `chat_template_kwargs:{"enable_thinking":false}` suppresses it. **P1:** raw HTTP replay of the handler's exact pair
      messages at max_tokens=1000 with default kwargs reproduces the parse failure (empty/invalid JSON).
      **P2:** the same call with enable_thinking:false returns valid JSON with non-empty chosen+rejected.
      If P2 holds: repair path is an extra-body passthrough for `chat_template_kwargs` in the target client
      (minimal package edit, single config line) — pending operator gate on breaking run1's zero-edits stance.
- **H-think results (same day)** — P1 CONFIRMED: default kwargs → 0 content chars (`finish_reason=length`,
      ~4.4k reasoning chars against the 1000-token budget), byte-identical parse error to attempt-2's tool
      failure. P2 CONFIRMED: same prompt with enable_thinking:false → valid JSON at just 230 completion tokens,
      chosen=211 / rejected=382 chars (chosen already shorter than rejected — bramble-consistent direction).
      Natural-language suppression control ("do not think..." appended in context): failed 3/3 (4–5k reasoning
      chars every trial) → the only effective lever is `chat_template_kwargs`. **Gate opened:** (A) minimal
      extra-body passthrough edit scoped to the target client + one config line, vs (B) operator restart of :8081
      with a thinking-off template — which changes behavior for ALL consumers on that lane, including pi's own
      provider entry. Gate resolved by operator (2026-08-22): B ruled out — a global thinking-off on :8081 would switch off the orchestrator model's own thinking, since pi runs through that same lane. A adopted: per-request `chat_template_kwargs:{enable_thinking:false}` scoped to forge's target client only.
- **H-probe** (formed before its observation): the empty-text cause is the structured spec notation in P
      (bracketed ranges like `[10,55]`, `/100w` units, axes lists) degenerating JSON generation on the
      :8081 checkpoint at temp 0.7 inside the permutate template. **Prediction: a prose-flattened variant
      P' carrying the same persona content yields ≥1 non-empty rephrase.** Null reading if empty persists:
      spec-ness/length of the payload itself blocks the permutate leg → run1 reroutes persona conditioning
      through the pair stage's `context` slot, and P3-a is judged on that leg alone.


---
### run1 — 2026-08-22 UTC
- **run1** (2026-08-22 UTC) — persona-DPO multiply bramble
  Card: bramble | Bramble (the efficient pragmatist)
  Moods: imperative, socratic
  Scenario types: coding, casual
  Served model identity (8081): unknown
  /v1/models @ :8082: unknown
  rows produced vs target: 0/8
  chosen median/mean length: 0.0/0.0 chars
  rejected median/mean length: 0.0/0.0 chars
  embedding_distance_chosen_rejected range: [N/A, N/A]
  TrajectoryProfile deadpan_score range: N/A
  failed rows: 1
  LLM calls used: 1
  Wall time: 42.7s
  First failure verbatim: <read error>

---
### run1 — 2026-08-22 UTC
- **run1** (2026-08-22 UTC) — persona-DPO multiply bramble
  Card: bramble | Bramble (the efficient pragmatist)
  Moods: imperative, socratic
  Scenario types: coding, casual
  Served model identity (8081): Qwen3.8-27B-IQ4_NL
  /v1/models @ :8082: embeddinggemma-300M-F32-pooled
  rows produced vs target: 0/8
  chosen median/mean length: 0.0/0.0 chars
  rejected median/mean length: 0.0/0.0 chars
  embedding_distance_chosen_rejected range: [N/A, N/A]
  TrajectoryProfile deadpan_score range: N/A
  failed rows: 1
  LLM calls used: 1
  Wall time: 47.4s
  First failure verbatim: <read error>

---
### run1 — 2026-08-23 UTC
- **run1** (2026-08-23 UTC) — persona-DPO multiply bramble
  Card: bramble | Bramble (the efficient pragmatist)
  Moods: imperative, socratic
  Scenario types: coding, casual
  Served model identity (8081): Qwen3.8-27B-IQ4_NL
  /v1/models @ :8082: embeddinggemma-300M-F32-pooled
  rows produced vs target: 1/8
  chosen median/mean length: 168.0/168.0 chars
  rejected median/mean length: 266.0/266.0 chars
  embedding_distance_chosen_rejected range: [0.2599, 0.2599]
  TrajectoryProfile deadpan_score range: N/A
  LLM calls used: 1
  Wall time: 11.5s

---
### run1 — 2026-08-23 UTC
- **run1** (2026-08-23 UTC) — persona-DPO multiply bramble
  Card: bramble | Bramble (the efficient pragmatist)
  Moods: imperative, socratic
  Scenario types: coding, casual
  Served model identity (8081): Qwen3.8-27B-IQ4_NL
  /v1/models @ :8082: embeddinggemma-300M-F32-pooled
  rows produced vs target: 0/8
  chosen median/mean length: 0.0/0.0 chars
  rejected median/mean length: 0.0/0.0 chars
  embedding_distance_chosen_rejected range: [N/A, N/A]
  TrajectoryProfile deadpan_score range: N/A
  failed rows: 1
  LLM calls used: 1
  Wall time: 9.2s
  First failure verbatim: <read error>