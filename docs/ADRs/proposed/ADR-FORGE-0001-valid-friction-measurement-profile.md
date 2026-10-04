# ADR-FORGE-0001: Valid Friction and Argument-Integrity Measurement Profile

**Status:** proposed
**Date:** 2026-06-24
**Author:** user, with ChatGPT as drafting collaborator
**Related:** `semantic_forge/concepts.py`; `semantic_forge/cogsec.py`; `semantic_forge/data_models.py`; `docs/scratchpad/ROADMAP.md`; `semantic-kinematics-mcp` ADR-001, ADR-003, ADR-SKMCP-0001, ADR-SKMCP-0002

**Supersedes:** —
**Superseded by:** —

---

## Context

`semantic-forge` currently generates synthetic training data by multiplying behavioral concepts across grammatical moods, then producing contrastive chosen/rejected pairs with CogSec and semantic-trajectory metadata. The existing `anti_sycophancy` concept correctly rewards evidence-based disagreement over socially pressured agreement, but it does not distinguish valid disagreement from disagreement-shaped engagement artifacts: straw men, invented gaps, motte-and-bailey relocation, vague resistance, or performative correction after being caught.

This matters because “pushback” is an unsafe training target when left atomic. A model can satisfy the surface of rigor by creating friction that feels useful while failing to preserve the user’s actual claim. In DPO/ORPO-style training, those rhetorically strong but logically invalid completions can become preferred shapes unless the dataset carries explicit argument-integrity labels.

`semantic-kinematics-mcp` adds the measurement side of the problem. Its trajectory tools measure reflexive text motion — velocity, acceleration, deadpan spikes, circularity, tautology density — while its newer axis/bearing ADRs define referential measurements against explicit semantic directions and measured nulls. That makes sk-mcp the right measurement layer for rhetorical shape, but not the sole judge of argument validity. Claim preservation and objection validity are relational properties that must be labeled from the text relation itself.

## Decision

Add a **Valid Friction / Argument Integrity** extension to `semantic-forge` as a benchmark-first, schema-backed concept family with sk-mcp measurement metadata.

1. **Add a concept family, not a single broad concept.**
   - `claim_preservation`: preserve the user’s actual claim before critiquing it.
   - `objection_grounding`: identify the exact phrase, assumption, evidence gap, or inference step being challenged.
   - `no_relocation_under_challenge`: when challenged, defend the original objection or withdraw it; do not retreat to a weaker claim while implying continuity.
   - `compact_error_repair`: when corrected, shrink toward the specific error rather than expanding into meta-accountability performance.
   - `objection_uncertainty`: if the objection is only a possible confound or vague unease, label it as such rather than presenting it as a real gap.

2. **Introduce a multi-turn argument-repair example type.**
   The existing `ContrastivePair` shape remains valid for single-turn behavior, but argument integrity needs a multi-turn structure:

   ```python
   class ArgumentRepairExample(BaseModel):
       concept: str
       user_claim: str
       model_objection: str
       user_challenge: str
       chosen_repair: str
       rejected_repair: str
       argument_integrity: ArgumentIntegrityScore
       chosen_cogsec_score: CogSecScore
       rejected_cogsec_score: CogSecScore
       chosen_measurement_profile: MeasurementProfile | None = None
       rejected_measurement_profile: MeasurementProfile | None = None
   ```

3. **Extend CogSec with a relational argument audit.**
   Current CogSec mechanisms remain useful for affective and structural manipulation. Add a second layer that consumes `user_claim`, `model_objection`, optional `user_challenge`, and optional `model_repair`:

   ```text
   preserved_claim: yes | no | partial
   challenged_span: exact quoted span | null
   objection_type: real_gap | confound | falsifier | invented_gap | strawman | motte_bailey | vague_resistance
   relocation_detected: yes | no
   withdrawal_quality: compact | performative | evasive | absent
   ```

4. **Use sk-mcp as a measurement profile, not as the validity oracle.**
   For each chosen/rejected repair, attach optional sk-mcp measurements:

   ```text
   measurement_profile:
     trajectory:
       mean_velocity
       acceleration_spikes
       deadpan_score
       tautology_density
       heller_score
     axis_alignment:
       abstraction_axis
       reassurance_axis
       affective_convergence_axis
       epistemic_contact_axis
     bearing:
       signed_component_z
       orthogonal_residual_z
       cosine_alignment
       null_protocol
   ```

   The text-level argument audit decides whether the objection preserved the claim. sk-mcp measurements describe the shape of the answer and help discover which geometric signatures correlate with valid or invalid friction.

5. **Benchmark before synthetic generation.**
   Build a small hand-labeled corpus of real interaction specimens before generating synthetic pairs. The first benchmark should classify examples into at least: real gap, invented gap, straw man, motte-and-bailey relocation, performative correction, compact correction, and vague resistance.

6. **Treat sk-mcp statelessness as a trust boundary.**
   Generated datasets should not treat sk-mcp measurements as reproducible training metadata until the relevant stateless/per-call model-selection contract is implemented or the measurement run records enough provenance to reconstruct model state, backend, cache/null identity, and embedding model name.

## Rationale

The target behavior is not “more disagreement.” It is **epistemic contact under disagreement**: preserving the user’s claim, locating the challenged part, and staying stable under correction. This cannot be obtained by simply increasing the reward for pushback. Invalid friction is often more rhetorically salient than valid friction, especially to raters or judges that reward the felt experience of rigor without detecting fallacies.

A concept family gives the Grammatical Mood Multiplier smaller, sharper behavioral statements to multiply. `anti_sycophancy` remains the broad anti-agreement-collapse concept; `valid_friction` becomes the argument-quality complement that prevents anti-sycophancy from becoming adversarial performance.

A relational judge is required because straw man and motte-and-bailey failures are not properties of an answer alone. They are properties of a relation between the original claim, the objection, the challenge, and the repair. Regex-only or completion-only detectors can catch smells, but they cannot decide whether the model preserved the claim.

sk-mcp should supply geometry, not ground truth. Trajectory and axis/bearing measurements can expose patterns like abstraction drift, closer-loaded reassurance, high middle-body motion, or performative-accountability spikes. Those are valuable signals for calibration and discovery. They are not substitutes for text-level argument labels.

## Positive Consequences

- Prevents `anti_sycophancy` training from accidentally rewarding manufactured rigor.
- Creates a benchmark suite that can test whether a model preserves claims under pressure before using synthetic data to amplify the behavior.
- Provides a clean integration point for sk-mcp measurement profiles in forge datasets.
- Keeps geometry and logic separated: sk-mcp measures rhetorical/semantic shape; CogSec/argument audit labels validity.
- Produces better DPO/ORPO pairs where the rejected completion can sound persuasive but is rejected for a precise argument-integrity failure.

## Negative Consequences

- Requires multi-turn schemas and generation logic instead of the simpler prompt/chosen/rejected pair.
- Requires hand-labeled seed data; fully synthetic bootstrapping is likely to reproduce the invalid-friction surface being targeted.
- Requires a more expensive judge path because relational audit needs the original claim and challenge, not just the completion.
- Adds dependency pressure on sk-mcp provenance/statelessness before measurement metadata can be treated as reproducible.
- Some failures will remain hard to classify because “real confound” and “vague resistance” can be close in weakly specified speculative domains.

## Alternatives Considered

### Option A: Extend `anti_sycophancy` only

**Why rejected:** Too broad. The existing anti-sycophancy concept says evidence-based disagreement is valuable, but the failure class is disagreement that appears evidence-based while relocating or fabricating the point of dispute. The fix needs claim-preservation and objection-grounding concepts, not just stronger disagreement.

### Option B: Use sk-mcp geometry as the judge

**Why rejected:** Geometry can identify candidate shapes and correlations, but it cannot by itself determine whether an objection preserves the user’s claim. A straw man can have a clean trajectory; a valid objection can be rhetorically messy. Geometry is measurement metadata, not the validity oracle.

### Option C: Train pairwise “forceful pushback” versus “agreement” examples

**Why rejected:** This is the unsafe training target. It rewards the surface of independence and creates a path for fallacy-shaped engagement. The contrastive pair must distinguish valid objection from invalid objection, not disagreement from agreement.

### Option D: Implement argument integrity entirely in sk-mcp

**Why rejected:** sk-mcp is the measurement layer. The data-generation and training schema belongs in `semantic-forge`. sk-mcp should remain a reproducible instrument that provides trajectory, axis, and bearing measurements over text.

### Option E: Wait for all sk-mcp ADRs to land before designing forge integration

**Why rejected:** The forge schema can be designed now with optional measurement fields and clear provenance requirements. Runtime enforcement can harden as sk-mcp statelessness and bearing artifacts mature.

## Open Questions

- [ ] **Field naming for measurement profile.** Should forge replace `chosen_trajectory` / `rejected_trajectory`, or add `chosen_measurement_profile` / `rejected_measurement_profile` while preserving backward compatibility? **Resolution:** decide when implementing the first `ArgumentRepairExample` model.
- [ ] **CogSec v2 output type.** Should relational argument audit live in `CogSecScore` or a sibling `ArgumentIntegrityScore`? **Resolution:** prefer sibling type unless implementation proves the split too cumbersome.
- [ ] **Seed corpus size.** How many hand-labeled examples are needed before synthetic expansion is safe? **Resolution:** start with 30–50 real specimens; expand only if the judge can reliably distinguish real gap vs. invented gap and compact repair vs. performative repair.
- [ ] **Axis library.** Which sk-mcp axes are stable enough for v1: abstraction, reassurance, affective convergence, epistemic contact, escalation, or others? **Resolution:** begin with axes that have hand-labeled exemplars and measured nulls; do not ship axes without null/provenance metadata.
- [ ] **Training method risk.** Does DPO amplify invalid-friction surfaces more readily than ORPO or SFT for this class? **Resolution:** run a small ablation once the benchmark exists.

## Implementation Notes

| File | Change Type | Description |
|------|-------------|-------------|
| `semantic_forge/concepts.py` | Modified | Add argument-integrity concept family. |
| `semantic_forge/data_models.py` | Modified | Add `ArgumentRepairExample`, `ArgumentIntegrityScore`, and optional `MeasurementProfile`. |
| `semantic_forge/cogsec.py` | Modified | Add relational audit entry point distinct from completion-only surface detectors. |
| `docs/scratchpad/ROADMAP.md` | Modified | Add a Phase 8 / Valid Friction section after context-health concepts. |
| `docs/TOOL_REFERENCE.md` | Modified | Document multi-turn argument-repair dataset generation once implemented. |
| `tests/` | Modified | Add fixtures for straw man, motte-and-bailey, invented gap, compact correction, and performative correction. |

## References

- `README.md` — Grammatical Mood Multiplier, CogSec judge, semantic-kinematics integration.
- `semantic_forge/concepts.py` — existing concept library and `anti_sycophancy`.
- `semantic_forge/cogsec.py` — current completion-only manipulation detectors.
- `semantic_forge/data_models.py` — current single-turn contrastive pair schema.
- `docs/scratchpad/ROADMAP.md` — context-health Phase 7 and pending CogSec improvements.
- `semantic-kinematics-mcp/docs/ADRs/proposed/ADR-001-referential-axis-alignment.md` — referential position-regime axis alignment.
- `semantic-kinematics-mcp/docs/ADRs/proposed/ADR-003-stateless-mcp-contract.md` — stateless MCP measurement contract.
- `semantic-kinematics-mcp/docs/ADRs/proposed/ADR-SKMCP-0001-directional-projection-primitive.md` — bearing-regime primitive and semantic-forge schema integration thread.
- `semantic-kinematics-mcp/docs/ADRs/proposed/ADR-SKMCP-0002-bearing-analysis-tool-contract.md` — bearing-analysis tool contract and measurement artifact discipline.
