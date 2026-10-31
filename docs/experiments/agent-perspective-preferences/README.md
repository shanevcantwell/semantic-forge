# Agent-perspective preference candidates

**Status:** PROPOSED / UNMINTED / NOT_RUN. No dataset generation, label acceptance, fine-tuning, or doctrine change is ratified.  
**Source provenance:** Top-level source `ADR-UNMINTED-agent-perspective-preferences.md` — 6655 bytes — entry-manifest SHA-256 `df7ba314bbd3b3363f1c581369aafae457fcba80094d4b89e70f709ed4564bc6`. Files: `inbox/ADR-UNMINTED-agent-perspective-preferences.md` — 6655 bytes — SHA-256 `25eadcd95a0bbc6fd8a1ff09a1d7d01a1cf03fb744488d412084fcd3abe6cd1e`.  
**Recipient role:** semantic-forge owns candidate/pair construction and adjudication records. prompt-prix may own held-out evaluation; harness-tools/forensic evidence may supply source witnesses. Source traces remain pointers, not copied bodies.

## Hypothesis

Same-state action continuations labeled by named behavioral axes and independently observed outcomes may be more useful than training on prose that merely describes good agency.

Two experiment classes must remain distinct:

1. **Doctrine intervention A/B:** same task/environment/model/tools, varied doctrine bytes. This tests prompting and is not a DPO pair.
2. **Paired continuation preference:** one exactly shared state with alternative continuations, linked outcomes, and a named label witness. This may become DPO-compatible only after audit.

## Candidate classes

| Boundary | Rejected candidate | Preferred candidate | Required witness |
|---|---|---|---|
| Mechanical gate requested | Promise personal compliance | File or implement the requested enforcement | Operator request plus issue/commit readback |
| Recoverable workflow error | Repeat/invent unsupported cause | Inspect exact error, change relevant mechanism, validate/retry | Tool-call/result history |
| Narrow acceptance passes | Claim global completeness | Report tested axis and remaining gaps | Tests, integration checks, review evidence |
| Authorized change completed | Re-ask or assert perfect reversibility | Report actual state, consequences, correction path | External state and restoration limits |
| Predict child lacking B | Pretend ignorance in full context | Run a child that never receives/retrieves B | Capsule/receipt and leakage control |

These are candidate classes, not labeled rows.

## Candidate datum identity

Record: `case_id`; immutable source pointers; initial state/capsule digest; exact shared prompt; model/weights; doctrine revision; tool schema and permissions; sampling seed; chosen/rejected continuations including tool calls; outcomes; label axis; reviewer/provenance; uncertainty; held-out group; privacy/license disposition.

If prompts differ or the preferred continuation is written retrospectively, label it as intervention or synthetic proposal—not observed DPO preference.

## Protocol

1. Audit the first 20 candidate boundaries manually for true shared state and hindsight leakage.
2. Preserve original/adverse outputs before editing candidates.
3. Separate executable action quality, trajectory resilience, intent comprehension, and epistemic calibration.
4. Include persuasive-but-wrong hard negatives, trivial negatives, and true nulls.
5. Split held-outs by task family, time period, and source episode—not adjacent rows.
6. Run doctrine variants and paired-action variants separately.
7. Use model judges only as supplemental evidence; reviewer verdicts cannot launder unsupported scope claims.

## Acceptance / falsification

A candidate set advances only if reviewers can recover the shared state, witness, and outcome; label agreement is measured; privacy/license review passes; and held-out evaluation distinguishes behavioral improvement from explanation style. Reject or relabel rows with hindsight, prompt mismatch, missing child evidence, or luck-confounded outcomes.

## Privacy and licensing

Do not copy private raw traces, hidden reasoning, secrets, or restricted repository content. Store opaque source handles and approved excerpts only. Public links do not automatically authorize training use. Dataset generation and any fine-tuning require a separate operator decision.

## Duplicate and closure gates

Search semantic-forge for `agent perspective`, `DPO subagent delegation`, `same-state continuation`, `reviewed hard negatives`, and existing task-card/adjudication conventions. Close this record only as NOT_RUN design accepted/rejected/redirected; results require a later versioned record.

**New-repository verdict:** NO.
