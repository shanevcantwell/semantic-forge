# Recon — forge stages for persona-DPO P3 (forge-multiply)

Static recon @ HEAD `e160aca` (persona-dpo P3 pre-run predictions). No HTTP, no servers.
Scope: semantic_forge package + semantic_forge_config.json. All pointers file:symbol or file:line.

## 1. Tool inventory (`register_handlers`)

Registration surface: handlers.py:651-678 — low-level decorators `@server.list_tools()` →
`get_all_tools()` (mcp.py:163) + `@server.call_tool(name, arguments)` validating args against
Pydantic params models then routing via `_TOOL_ROUTES` / `_PARAMS_MODELS` (handlers.py:609-624).
Tool defs + param schemas: mcp.py:82-160 / 13-89. Eight tools, all names match routes exactly:

| Tool | Purpose | Args (defaults) | Stage map |
|---|---|---|---|
| `permutate_phrasing` | N rephrasings of a concept across grammatical moods; optional SK embedding-spread check | `concept:str`, `moods:list[str]`(7 default), `model:str\|None`=None ("ollama:x"/"hf:y" override), `validate_diversity:bool`=True | **rephrase** |
| `generate_scenario` | Situated scenarios for a rephrased concept | `rephrased_concept:str`, `scenario_types:list[str]`=["financial","coding","research","casual"], `count:int`=3 | **scenario** |
| `generate_contrastive_pair` | chosen/rejected pair + CogSec scores + SK trajectories/drift; fails fast if SK off | `scenario:str`, `context:str` | **pair** |
| `score_completion` | CogSec adversarial audit of one completion | `completion:str`, `context:str\|None`=None, `criteria:str`="cogsec" (echoed only) | scoring support |
| `validate_diversity` | pairwise SK drift over rephrasings; warn outside [0.2, 0.5] | `rephrasings:list[str]`, `threshold_min`=0.2, `threshold_max`=0.5 | support (SK) |
| `validate_trajectory` | per-completion SK trajectory vs target shape — result largely hardcoded (§5c) | `completions:list[str]`, `target_shape:str`="steady" | support (SK) |
| `build_dataset` | end-to-end: concept ID → rephrase → scenario → pair → jsonl | `concept:str`(=CONCEPT_LIBRARY **id**, not free text), `rephrasing_count`=5, `scenarios_per_rephrasing`=3, `output_format`="jsonl" | dataset build (handlers.py:487-569) |
| `dataset_stats` | stats on a saved jsonl | `dataset_path:str` | support |

Note: `build_dataset` hardcodes moods/scenario_types internally (handlers.py:506-521), ignores
`output_format` for anything but jsonl, and writes `data/{concept_id}_dataset.jsonl` (:539).

## 2. Persona slot surface — SLOTS EXIST; no core wiring needed for the stage tools

Prompt entry points (exact template lines):
- rephrase: handlers.py:105 — `"Rephrase the following concept as {mood} mood ... \n\n{concept}\n\nRephrased ({mood}):"` → **`concept` arg is arbitrary free text; persona card material plugs in verbatim.**
- scenario: handlers.py:184 — `"...for the concept: "{rephrased_concept}"..."` → carries persona-conditioned rephrase onward.
- pair: handlers.py:263 — prompt embeds `{context}` (persona text acceptable) + `Scenario: {scenario}`.

Persona card shape (cards/bramble.yaml): `id`, `name`, `archetype`, `probes` (free text), 5-axis
schema block — the natural persona payload is a composed string from these fields; no field maps
1:1 to any arg, i.e. **compose-then-pass into `concept`/`rephrased_concept`/`context`.**

Verdict: an existing arg already accepts arbitrary concept/prompt text at every stage → P3-a's H0
holds for the per-stage path (no structural rewrite; persona enters via prompt slots). Two build-path
caveats (additive, not core):
- `build_dataset` entry requires a CONCEPT_LIBRARY id: handlers.py:506 (`get_concept_by_id`, strict
  match at concepts.py:179) → persona card must be added as a BehavioralConcept (concepts.py:16 list
  ends ~178) to go through the one-shot builder.
- In `build_dataset` step 3, pair `context=concept.id` (handlers.py:538), NOT rephrase text — persona
  conditioning attenuates at the pair stage in the built-in orchestrator only; per-stage dispatch keeps it.

## 3. SK leg — config is 1 field from live; but filters are dead code (P3-b delta)

Config dataclass: SemanticKinematicsConfig, config.py:31-47 (`endpoint`, `backend`, `base_url`,
`model_name`). Live values already in semantic_forge_config.json: `endpoint=null`,
`backend="lmstudio"`, **`base_url="http://inference-host:8082/v1"` and
`model_name="embeddinggemma-300M-F32-pooled" ALREADY SET** (the :8082 URL already lives here).
JSON load config.py:141-148; env overrides SEMANTIC_KINEMATICS_ENDPOINT/BACKEND/BASE_URL/MODEL_NAME,
config.py:203-214.

`endpoint` is NOT a URL: SemanticKinematicsClient (integrations.py:14) is an **stdio MCP bridge** —
initialize() execs the endpoint string as a command via StdioServerParameters (integrations.py:57-118;
formats in `_parse_endpoint`: `"semantic-kinematics-mcp"`, `"cmd,arg..."`, `"docker:run,-i,--rm,network=host,image"`).
The :8082/v1 URL is consumed *inside* the semantic-kinematics-mcp server process: forge passes
`base_url`/`model_name` to it as `model_load` args via `_ensure_backend` (integrations.py:138-157, tool calls 169-209).
So flipping `endpoint` literally to the :8082 URL would try to exec a process named "http://..." — wrong shape.

Filter that consumes SK results: dataset.py:40 `filter_by_embedding_distance(pair, min=0.1, max=0.8)` —
keeps pair iff 0.1 ≤ `pair.embedding_distance_chosen_rejected` ≤ 0.8 (distance comes from
sk_client.calculate_drift(chosen,rejected) at handlers.py:~305). Sibling filter_by_cogsec_score at
dataset.py:19 (chosen structural_cleanliness ≥ 0.7 AND rejected manipulation_score ≥ 0.3 — the latter
comparison direction looks inverted vs its docstring). **Operational meaning of "discards a non-zero
fraction": pairs whose chosen/rejected texts embed closer than 0.1 or farther than 0.8 are dropped.**

**However: neither filter is called anywhere as of e160aca** (only definitions dataset.py:19/40 + unused
import handlers.py:46; no call in build path, main.py, __main__.py). Flipping SK lights up *data*
(embedding_distance + trajectories populated on every pair) but discards nothing.

Verdict P3-b: **config-only to light up SK data flow** — set `endpoint` to the executable SK-MCP command;
base_url/model_name need no change (already correct). "Discarding a non-zero fraction" additionally
requires ~2 additive lines calling filter_by_embedding_distance in the build path (handlers.py step 4 or
dataset.build_dataset) — flag as falsification-risk for P3-b's wording, not for SK activation itself.

## 4. Pair generation mechanics + persona-relevant signals

Mechanics (handlers.py:260-318): **one** `client.generate_structured(prompt, dict, temperature=0.8,
max_tokens=1000)` call on the target backend (:8081 Qwen via get_target_config) where the model itself
writes BOTH completions as JSON {prompt, chosen, rejected} — not dual sampling at different temps, and no
score threshold gates acceptance in the handler (filters dead, §3). Then per pair: `cogsec_score(text, context)`
on both = deterministic pattern audit (cogsec.py:188 `score_completion` → CogSecScore threat_level /
manipulation_score / structural_cleanliness / detected_mechanics; no LLM), SK analyze_trajectory per
completion (→ TrajectoryProfile mean_velocity/deadpan_score/acceleration_spikes/torsion/curvature,
data_models.py:~41-48) and calculate_drift → embedding_distance_chosen_rejected. Fail-fast on missing SK
(handlers.py:237-252 → SemanticKinematicsRequiredError).

Output record: ContrastivePair (data_models.py:50-59); build path zips to TrainingExample jsonl
(data_models.py:62-81; dataset.build_dataset dataset.py:60-97, save :110) — fields = concept/mood/
scenario/scenario_type/prompt/chosen/rejected + both CogSecScores + both TrajectoryProfiles +
embedding_distance_chosen_rejected + generated_at/source_model.

Persona-relevant scoring candidates that exist additively once SK is live:
- `embedding_distance_chosen_rejected` (pair-level, populates automatically);
- TrajectoryProfile axes per completion (deadpan_score / mean_velocity / curvature / torsion — register proxies);
- CogSecScore fields (already populate today without SK);
- trivial offline: chosen/rejected length distributions per card (P3-c signature check needs nothing new).
Minimal plug for a genuinely persona-fidelity signal: `score_completion`'s `criteria` arg is accepted but
ignored (handlers.py:340 — only echoed) and the filter hooks sit at dataset.py:19/40; mapping only, no design.

## 5. Issue #5 flags vs current code

a) **MCP registration API mismatch — gone/unverified-as-issue.** handlers.py:651-678 uses low-level
   `@server.list_tools()`/`@server.call_tool()` consistently with mcp.py get_all_tools(); the 8 Tool defs
   match _TOOL_ROUTES exactly; P1's rerun_smoke dispatch ran green on this same surface (P1 verdict,
   roadmap). Current code self-consistent and runtime-proven.
b) **build_dataset dict-vs-Pydantic crash — gone as of HEAD.** handlers.py:533-547 pass list[ContrastivePair]
   into dataset.build_dataset which does attribute access on Pydantic (dataset.py:60-97); rephrasings/
   scenarios are passed dicts and consumed via `.get()` by design. No dict-vs-Pydantic path remains in-repo.
c) **validate_trajectory hardcoded return — STILL HOLDS.** handlers.py:~455-481: single SK
   trajectory_result copied verbatim to every completion; `"matches_target": True  # Simplified`; constant
   `contrastive_validation={"is_truly_contrastive": True, "trajectory_distance": 0.2}`.
d) **Stale docs referencing nonexistent APIs — STILL HOLDS.** INTEGRATION_GUIDE.md:56 documents
   `client.calculate_drift(embeddings)` (code has calculate_drift(text_a,text_b) integrations.py:~252 and
   separate calculate_drift_from_embeddings :280); INTEGRATION_GUIDE.md:134/148 document
   PromptPrixClient.fan_out / compare_results — no such methods (client is list_models/complete/judge/
   calculate_drift only, integrations.py:392-560); TOOL_REFERENCE.md build_dataset example includes an
   `output_path` arg absent from BuildDatasetParams (mcp.py:71-74), and output_format advertises "parquet"
   with no parquet code anywhere in the package.

Bonus defect found during recon (not on issue #5's list): handlers.py:139 calls
`sk_client._get_embedding(r.text)` — **no such method** on SemanticKinematicsClient (real one is
embed_text, integrations.py:~243). With SK live + validate_diversity=True this raises AttributeError,
swallowed by the bare `except Exception: pass` at handlers.py:~158 → diversity leg silently no-ops.
Workaround for P3: call permutate_phrasing with `validate_diversity:false` (budget-wise preferable anyway).

## Minimal P3 invocation sketch (6 rows = 1 card × 2 scenarios)

Persona payload P = bramble.yaml fields composed into one string (id/archetype/probes/axes).
Dispatch shape: rerun_smoke.py pattern — `register_handlers(server)` then dispatch CallToolRequest.

1. `permutate_phrasing` {"concept": "<P>", "moods": ["imperative", "socratic"], "validate_diversity": false}
   (false avoids the dead `_get_embedding` leg, §5 bonus; rephraser hits :8081 per config) → pick r1, r2.
2. ×2: `generate_scenario` {"rephrased_concept": "<ri>", "scenario_types": ["coding", "casual"], "count": 1}
   → s_i for each rephrase (6-row matrix if run with scenario_types×count=3; keep count=2 types).
3. ×(per row): `generate_contrastive_pair` {"scenario": "<s description>", "context": "<P + rephrase context>"}
   — SK LIVE REQUIRED here (fail-fast, handlers.py:237); first call doubles as SK availability probe.

Config diff to light SK (semantic_forge_config.json, semantic_kinematics block):
```diff
-    "endpoint": null,
+    "endpoint": "semantic-kinematics-mcp",   # OR "docker:run,-i,--rm,network=host,<sk-image>" per _parse_endpoint forms
```
`backend`/`base_url`/`model_name` unchanged (already `lmstudio` / :8082/v1 / embeddinggemma-300M-F32-pooled).

Feasibility notes: steps 1–2 depend on nothing above being "additive wiring required" — persona slots exist
(§2 verdict). Step 3 depends on §3 verdict holding: config-only **if** an executable semantic-kinematics-mcp
command exists locally (env dependency, not a code touch); if the SK-MCP binary/docker image is absent, this is
a provisioning gap, still no forge code change. `build_dataset` one-shot alternative requires additive code:
append bramble as BehavioralConcept to CONCEPT_LIBRARY (concepts.py) AND accept context=concept.id attenuation
(§2 caveat) — prefer per-stage dispatch for P3 H0 fidelity. To make P3-b's "discards a non-zero fraction"
operationally true, add the filter_by_embedding_distance call (dataset.py:40) at handlers.py build step 4 or in
dataset.build_dataset — the only additive-wiring gate identified by this recon.
