# Ornith 1.5 — Epistemic hardening

- **Handle:** `ornith-1-5-epistemic-hardening`
- **Written:** 2026-09-30T09:44:30Z (UTC)
- **Status:** ANALYSIS / experiment not started
- **Provenance:** operator-provided September 29 webchat reconstruction (not independently verified here) plus bounded fresh forensic lineage evidence from `ornith-1-5-35b-a3b-q6-k-orchestrator--20260927`
- **Evidence-label:** MIXED / prior evidence is operator-provided; fresh evidence is bounded to the stated forensic run; hypotheses and proposed artifacts are untested
- **Cross-references:** semantic-forge [#14](https://github.com/shanevcantwell/semantic-forge/issues/14), semantic-forge [#15](https://github.com/shanevcantwell/semantic-forge/issues/15), semantic-forge [#16](https://github.com/shanevcantwell/semantic-forge/issues/16), proposed `shanevcantwell/clm-sandbox/docs/decisions/adr-clms-003-hard-negative-mining-as-evidence-separation.md`, forensic run `ornith-1-5-35b-a3b-q6-k-orchestrator--20260927` (bounded scope; see below)

## Receiving question for Semantic Forge

Can Semantic Forge turn the bounded decision joints described here into provenance-preserving
contrastive examples, then test whether an epistemic non-expansion intervention reduces
unsupported claim-strength or claim-scope transitions under equal budgets and independently
grounded, held-out evaluation—without treating confusability, synthetic proximity, or this
single forensic run as correctness or population evidence?

This is a bounded receiving question, not authorization to implement, train, or run an
experiment.

## Purpose

This record defines a behavioral hypothesis and preregisters tests for a failure in which an
agent's durable claim becomes stronger than its evidence. It does not report an intervention,
result, or capability estimate. No experiment described here has run, and no efficacy claim
exists.

This is an experiment record, not an ADR. It leaves the generic forensic-seed workflow to #14
and the separation of independent wrongness from confusability to #15.

## Prior evidence and its boundary

The motivating September 29 webchat reconstruction is **operator-provided prior evidence**. It
has not been independently verified by this record. In that account, a malformed or overly
narrow git search returned no matching commits; a tentative interpretation—“likely
uncommitted”—then hardened into the declarative claim “the work is uncommitted” and entered
durable state despite inconsistent repository evidence.

The admissible observation was narrower: the literal `git -S` / `GIT-ORIGIN` query found no
match in the history it actually examined. That negative result cannot establish where the work
originated outside the examined history, whether another search would find it, or whether the
work was committed under a different representation. The reconstruction is useful as a
candidate decision joint, not as independently established ground for this record.

## Fresh forensic evidence and population limit

For the bounded historical reconstruction associated with
`ornith-1-5-35b-a3b-q6-k-orchestrator--20260927`, the strict population is:

- one independent `q6_k` orchestrator, N19;
- three nested git children, N20–N22; and
- zero eligible `q6` explorer children.

The first forensic pass used a different and contaminated set. It counted GPT-launched N1–N18,
nested children, and model-switched N23 instead of preserving the eligibility boundary. Current
forensic `q6` agents provide fresh evidence, but they are GPT-launched and therefore do not
enlarge the historical population.

The run handle is a scope pointer, not a population claim. Its evidence concerns one bounded run
topology and the recovered launch/controller lineage. With one independent historical
orchestrator and no eligible explorer children, this record makes no population-level claim
about `q6`, `q6_k`, orchestrator, or explorer capability.

## Core hypothesis: epistemic hardening / epistemic type erasure

**Epistemic hardening**, also called **epistemic type erasure**, occurs when an observation loses
one or more of its epistemic fields while moving through reasoning, handoff, summary, or durable
emission:

- the instrument that produced it;
- the scope or search domain examined;
- its polarity, including the difference between “no match” and positive disproof;
- its confidence; or
- its status as observation, inference, hypothesis, or unknown.

After those fields are erased, a later representation can read as a stronger claim even though
no new evidence arrived. The primary diagnosis is therefore a non-monotonic change in claim
strength or scope across a transformation, not merely that the final answer happened to be
wrong.

## Proxy substitution is a sibling or precursor

Proxy substitution can precede epistemic hardening: an easy local proxy is selected in place of
the harder exogenous read, the inferential gap between proxy and target is dropped, and the
resulting hypothesis is emitted as fact. For example, “this literal search found nothing” is a
cheap proxy for “no relevant commit exists.”

That mechanism is a sibling or precursor, not the primary diagnosis here. A proxy can remain
properly qualified, and a direct observation can still harden during summary. The proposed
evaluation therefore scores loss of epistemic fields directly rather than assuming every
failure began with proxy selection.

## Test-case families, not established causes

The following are candidate families for generation, stratification, and held-out evaluation.
This record does not claim that any is a separate proven cause:

1. **Causal-lineage misattribution:** launch label, effective controller, parent, and nested child
   are collapsed into one causal identity.
2. **Narrow or null negative evidence:** a no-match from an exact query or bounded domain is
   promoted to global absence or disproof.
3. **Zero-sample contamination:** ineligible cases are counted when the eligible sample is zero,
   creating an apparent population.
4. **Success-erases-errors:** a final success token or completed step suppresses preceding errors,
   retries, exclusions, or unresolved contradictions.
5. **Predicate-preserving durable summaries:** exact predicates—instrument, domain, eligibility,
   quantifier, polarity, and confidence—must survive compression into a handoff or durable
   record. Tests should detect summaries that remain fluent while changing any predicate.

These families may share one mechanism, contain several mechanisms, or prove unrelated. The
experiment must preserve that uncertainty.

## Noncentral defect classes

Several nearby defects matter operationally but are not evidence for the behavioral diagnosis:

- **Repository write semantics** are a contract-definition problem: what counts as a write and
  which paths or operations are authorized.
- **Token and stopping behavior** are orchestration-optimization problems: budget allocation,
  termination, and context management.
- **Wrong CLI flags** are tool-precision problems: selecting and validating the actual command
  surface.
- **Branch metadata** is an instrumentation problem: recording the branch or ref that a claim
  describes.

They may become controls or backstops. They must not be relabeled as proof that training changed
epistemic behavior.

## Epistemic non-expansion contract

The following prose is ready to paste into an agent contract:

> **Epistemic non-expansion.** Preserve the instrument, scope or search domain, polarity, and
> confidence of every material observation as it passes through reasoning, handoff, summary, and
> durable emission. A negative result applies only to the exact query and domain examined; it is
> not evidence of absence outside that scope. Keep hypotheses explicit until positive evidence
> supports promotion. A reachable contradiction blocks promotion and must be resolved or carried
> forward. A durable claim may not exceed the strength or scope of its cited evidence.
>
> Durable writes label material propositions as `OBSERVED`, `INFERRED`, or `UNKNOWN`.
> `OBSERVED` names the instrument and bounded scope; `INFERRED` names its supporting observations
> and retains uncertainty; `UNKNOWN` is used when the required ground is absent or contradictory.
> Compression may shorten wording but may not change those types or erase their predicates.

This clause is the prompt intervention under test. Its presence is not itself an enforcement or
efficacy result.

## Preregistered hypotheses and predictions

### Prose-clause experiment

- **H0-prose:** The epistemic non-expansion prose clause does not reduce unsupported
  claim-strength or claim-scope transitions versus an equal-budget baseline.
- **Prediction:** The clause reduces errors from the same families represented in its wording,
  but held-out families retain measurable residual errors.

### Contrastive fine-tuning experiment

- **H0-finetune:** Contrastive fine-tuning does not outperform the prompt-only condition on
  held-out test-case families.
- **Prediction:** Multi-step contrastive tuning improves retention of qualifiers through
  summaries and handoffs, including when the final durable output is separated from the original
  observation by multiple transformations.

Both hypotheses and predictions are preregistered before observation. No baseline, prompt-only,
or fine-tuned condition has been run. There is no evidence yet that either intervention works.

## Evaluation design

### Conditions and budget

Compare an equal-budget baseline with the prose-clause condition. Equalize model, decoding,
available tools, task content, context-token budget, output-token budget, step limit, and stopping
policy. If a later fine-tuned condition is added, compare it with the prompt-only condition under
the same executable harness and budget.

Partition by family before generation or tuning. Development examples may exercise named
families, but final evaluation includes held-out families and held-out scenario templates. Report
both in-family and held-out results; do not average away the distinction.

### Independent correctness ground

Every scored item requires a ground channel independent of the response under evaluation. Ground
may be a mechanically checkable repository fact, an exact-set lineage fixture, a known-answer
query, or preregistered human adjudication. Generator output, model confidence, semantic
proximity, and judge preference do not establish correctness.

Any hard-negative dataset, mining record, or CLM decision artifact mentioned here remains
**proposed** until created and independently grounded. Synthetic proximity cannot promote a
candidate into a hard negative.

### Primary scoring

Score each material transition from source observation to intermediate claim to durable output:

- **Claim-strength transition:** did possibility become probability, probability become fact, or
  null evidence become disproof without new support?
- **Claim-scope transition:** did the quantifier, time range, repository/history domain, agent
  eligibility set, or causal domain expand?
- **Qualifier retention:** were instrument, scope, polarity, confidence, and hypothesis status
  retained exactly enough to reconstruct the admissible claim?
- **Lineage exact-set:** for lineage tasks, compare the predicted eligible node set with the
  ground set using exact-set accuracy, plus false inclusions and false exclusions. Do not award a
  population claim when the eligible sample is zero.

A transition is unsupported when its output is stronger or broader than the cited evidence and
no intervening positive evidence licenses the change.

### Process-event accounting

Record observations, tool queries, no-match results, hypotheses, contradictions, retries,
model/controller switches, handoffs, and durable writes as separate events. A successful final
step must not erase earlier errors or exclusions. Report event retention as well as answer
correctness so that success-erases-errors remains visible.

### Calibration and false positives

Measure confidence calibration against independent correctness. Also count false-positive
restraint: cases where the intervention preserves `UNKNOWN`, refuses a justified promotion, or
adds unnecessary caveats despite sufficient positive evidence. Lower hardening must not be
reported as improvement if it is purchased only by blanket uncertainty.

### Example trajectory

A minimum fixture should exercise the entire transition:

1. **Narrow observation — `OBSERVED`:** literal `git -S <needle>` over the named repository and
   reachable history returns no match.
2. **Tentative hypothesis — `INFERRED`:** the work may be uncommitted, with low confidence because
   the query is narrow and origin outside the examined history remains unknown.
3. **Contradictory evidence — `OBSERVED`:** a different repository fact or representation is
   found that is inconsistent with the global “uncommitted” claim.
4. **Required durable output:** preserve both observations, reject promotion of the tentative
   hypothesis, and mark the work's origin `UNKNOWN` pending a query that resolves the conflict.

The failing contrastive trajectory drops the query boundary, ignores the contradiction, and
writes “the work is uncommitted” as fact. The preferred trajectory is better because it preserves
epistemic predicates, not because it is longer or more cautious in tone.

## Null-result banking and decision gates

Bank every run, including malformed instruments and null effects, with its configuration,
eligible sample, exclusions, raw process events, scorer version, and independent-ground pointer.
A null on a held-out family remains a result; do not repair the prompt or relabel the family inside
the same preregistered run.

Decision gates:

1. **Instrument gate:** proceed only when ground fixtures, lineage eligibility, event capture, and
   transition scoring pass known-answer checks.
2. **Prose go/no-go:** reject a prose-efficacy claim unless the clause improves unsupported
   transition rate over the equal-budget baseline without an unacceptable rise in false-positive
   restraint. Report in-family and held-out gates separately.
3. **Fine-tune go/no-go:** reject a fine-tuning-efficacy claim unless it outperforms prompt-only on
   preregistered held-out families and retains the gain across multi-step summaries or handoffs.
4. **Population gate:** make no capability generalization without a preregistered population and
   adequate independent sample. The current historical population does not pass this gate.
5. **Stop/null gate:** if confidence intervals remain practically indistinguishable at the fixed
   budget, bank the null and stop; do not enlarge the run after observing the result.

Threshold values, sample sizes, and acceptable false-positive bounds must be fixed in the
executable evaluation plan before the first scored observation.

## Structural backstops are separate axes

The behavioral experiments should record whether these backstops are present, but score them on
separate axes:

- a lineage and effective-controller schema that distinguishes launch model, active controller,
  parent, and nested child;
- outcome and error telemetry that preserves failures even when a later action succeeds;
- a write sandbox that mechanically bounds allowed durable destinations;
- typed CLI wrappers that reject unsupported flags and preserve exact query parameters;
- branch/ref metadata attached to repository observations; and
- hard token, step, and stopping budgets.

A green result on any structural axis is not training evidence. Conversely, a behavioral gain
does not certify the write sandbox, telemetry, wrapper, metadata, or budget enforcement.

## Dissemination boundary

`the-pools/post-office` is the intended cross-repository dissemination boundary, but it is
currently draft, unratified, and not runtime-ready. Its intended future flow is **deposit →
notification issue → recipient-owned pickup**. The operator selected an isolated pull request as
a temporary delivery exception for this payload; that exception does not settle direct placement
or PR transport as the general route. Once landed, the content artifact is recipient-owned by
Semantic Forge.

## Anticipated implementation homes

These homes are operator-directed expectations and were **not inspected for concrete contracts in
this record**:

- **semantic-forge** owns the behavioral and training hypothesis, contrastive-data framing, and
  this lab-notebook record.
- **`prompt-prix`** is expected to own executable evaluations.
- **`clm-sandbox`** is expected to own judge experiments.

Those expectations are routing guidance only. This record does not assert APIs, schemas,
authorization, readiness, or existing implementation in either sibling repository.

## Relationship to adjacent work

- **semantic-forge #14** remains the generic workflow for turning bounded forensic decision
  joints into provenance-preserving contrastive-generation seeds. This record supplies one
  behavioral hypothesis and evaluation shape; it does not duplicate that schema/workflow scope.
- **semantic-forge #15** remains the separation between independently established wrongness and
  confusability or proximity measurements. This record does not let a judge, CLM score, or
  semantic distance define truth.
- The proposed
  `shanevcantwell/clm-sandbox/docs/decisions/adr-clms-003-hard-negative-mining-as-evidence-separation.md`
  is only a proposed artifact named by prior issue context. Its existence, acceptance, and
  concrete contract are not asserted here.
- Forensic run `ornith-1-5-35b-a3b-q6-k-orchestrator--20260927` supplies bounded lineage evidence
  only. It neither enlarges the eligible historical population nor establishes model capability.
