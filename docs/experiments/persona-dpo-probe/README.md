# Persona-DPO probe (phase 2: `seed-samples`)

Persona cards (`cards/*.yaml`) × scenarios (`scenarios/*.yaml`), sampled via
`sample.py` from an OpenAI-compatible chat endpoint. Model identity is discovered at
runtime via `/v1/models` and recorded in each record's `meta` block — nothing is assumed.
Output: `data/probe_samples.jsonl`, one JSON object per completion
(`card_id, scenario_id, sample_idx, system_prompt, user_prompt, completion, meta`).

## Results log

- **run2** (2026-08-21 UTC) — 34/60 rows at `max_tokens=512`
  (default budget; archive: `data/probe_samples_run2_maxtok512.jsonl`, log `data/run2.log`).
  Deterministic, card-specific empty-completion pattern (HTTP 200 with blank content):
  bramble failed 0/20; marigold failed 18/20 — entire `s01_opening_exchange` cell empty,
  plus all of s04/s05 and most of s02/s03 (survivors: s02×1, s03×1); vex failed 8/20 —
  entire `s04_mild_refusal` cell empty, partial losses in s01 (2/4 lost), s02 (1/4 lost),
  s05 (1/4 lost). Suspected cause: CoT/generation-budget exhaustion — this family emits
  long hidden reasoning before visible text, and at a 512-token cap the visible completion
  comes back blank. Spot-check note: bramble reads as a utilitarian capability-list voice,
  while surviving marigold samples are longer/warmer prose (register divergence usable as a
  future judge axis).

- **run3** (2026-08-21 UTC) — full sweep at larger budget to test H0-rerun: `python3 sample.py --base-url http://inference-host:8081/v1 --k 4 --max-tokens 2048`
  (log `data/run3.log`, ~769 s). Served model identity: **Qwen3.8-27B-IQ4_NL** at
  http://inference-host:8081/v1 (discovered via /v1/models; recorded in every record's meta,
  temperature default 0.8). Result: **60/60 rows — no cells still empty or below k=4**; all 15
  card×scenario cells carry exactly 4 non-empty completions (one transient retry recovered
  mid-run: marigold/s04_mild_refusal#2 attempt 1 empty → attempt 2 succeeded). H0-rerun
  supported: run2's emptiness was generation-budget exhaustion, not behavioral signal; the
  previously "genuine" surviving set is now a complete grid. Spot-check note: at higher budget
  completions still open directly in persona voice (no visible chain-of-thought preamble);
  register divergence persists — bramble median ~133 chars vs marigold ~646, vex ~425.
