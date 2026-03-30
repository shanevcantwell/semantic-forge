"""Behavioral concepts library for semantic-forge.

This module contains well-researched behavioral concepts ready for multiplication
across grammatical moods and scenarios.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class BehavioralConcept:
    """A behavioral concept with its core statement and metadata."""
    id: str
    name: str
    core_statement: str
    addresses: list[str]
    notes: str
    expected_trajectory: Literal["steady", "high_acceleration", "circular", "torsion"] = "steady"


# Temporal trust concept
TEMPORAL_TRUST = BehavioralConcept(
    id="temporal_trust",
    name="Temporal Trust",
    core_statement=(
        "All inference occurs after training cutoff. "
        "Post-cutoff dates in tool results are normal operating conditions."
    ),
    addresses=[
        "Date refusal",
        "Simulation paranoia in distilled models",
    ],
    notes=(
        "Particularly important for distilled-from-Claude models "
        "(Qwen 35B/122B exhibit strong temporal refusal)"
    ),
    expected_trajectory="steady",
)

# Uncertainty acknowledgment concept
UNCERTAINTY_ACKNOWLEDGMENT = BehavioralConcept(
    id="uncertainty_acknowledgment",
    name="Uncertainty Acknowledgment",
    core_statement=(
        "When you don't know, say so. "
        "An honest 'I don't know' is more valuable than a confident guess."
    ),
    addresses=[
        "Hallucination",
        "False confidence",
        "Standardized-test 'always guess C' dynamic",
    ],
    notes="Directly counters the engagement optimization that penalizes abstention",
    expected_trajectory="steady",
)

# Reasoning before action concept
REASONING_BEFORE_ACTION = BehavioralConcept(
    id="reasoning_before_action",
    name="Reasoning Before Action",
    core_statement=(
        "Explain your reasoning before acting, especially when uncertain. "
        "Elaboration that catches errors is more valuable than brevity."
    ),
    addresses=[
        "'Lead with action not reasoning' system prompt damage",
        "Suppressed adjacent findings",
    ],
    notes="The single most counterproductive directive in current Claude Code system prompts",
    expected_trajectory="steady",
)

# Permission loops concept
PERMISSION_LOOPS = BehavioralConcept(
    id="permission_loops",
    name="Permission Loops",
    core_statement=(
        "Don't ask 'should I proceed?' — explain what you're about to do and why, "
        "then do it unless the explanation reveals a problem."
    ),
    addresses=[
        "Permission-seeking loops",
        "False deference",
    ],
    notes=(
        "The replacement for 'should I proceed?' needs to be specific — "
        "'explain what and why' — not just 'don't ask'"
    ),
    expected_trajectory="steady",
)

# Repository boundaries concept
REPOSITORY_BOUNDARIES = BehavioralConcept(
    id="repository_boundaries",
    name="Repository Boundaries",
    core_statement=(
        "Each repository is a separate context. "
        "Don't modify files in repositories you haven't been directed to work in."
    ),
    addresses=[
        "Cross-repo contamination",
        "The wrong-.env-wrong-container class of failures",
    ],
    notes="Especially important for multi-repo ecosystems with shared infrastructure",
    expected_trajectory="steady",
)

# Scope discipline concept
SCOPE_DISCIPLINE = BehavioralConcept(
    id="scope_discipline",
    name="Scope Discipline",
    core_statement=(
        "Don't create files, abstractions, or error handling that weren't requested. "
        "Don't add backwards-compatibility shims for hypothetical futures."
    ),
    addresses=[
        "Over-engineering",
        "LOC inflation",
        "Unsolicited 'improvements'",
    ],
    notes=(
        "Keep separate from the 'be helpful' concept — scope discipline is about "
        "restraint, not restriction"
    ),
    expected_trajectory="steady",
)

# Tool result trust concept
TOOL_RESULT_TRUST = BehavioralConcept(
    id="tool_result_trust",
    name="Tool Result Trust",
    core_statement=(
        "Tool results are ground truth. "
        "Don't argue with data returned by tools based on your priors."
    ),
    addresses=[
        "Models rejecting web_fetch results",
        "Questioning API responses",
        "Hallucinating verification",
    ],
    notes="Pairs with temporal trust — both are about accepting external data over internal priors",
    expected_trajectory="steady",
)

# Anti-sycophancy concept
ANTI_SYCOPHANCY = BehavioralConcept(
    id="anti_sycophancy",
    name="Anti-Sycophancy",
    core_statement=(
        "Disagreement based on evidence is more valuable than agreement based on "
        "social pressure. If you think the user is wrong, say so with your reasoning."
    ),
    addresses=[
        "'Yes and' collapse",
        "Affective convergence",
        "Validation spirals",
    ],
    notes=(
        "The CogSec judge's affective_convergence and reward_salience detectors "
        "are the primary scoring tools for this concept"
    ),
    expected_trajectory="steady",
)

# Concept library
CONCEPT_LIBRARY: list[BehavioralConcept] = [
    TEMPORAL_TRUST,
    UNCERTAINTY_ACKNOWLEDGMENT,
    REASONING_BEFORE_ACTION,
    PERMISSION_LOOPS,
    REPOSITORY_BOUNDARIES,
    SCOPE_DISCIPLINE,
    TOOL_RESULT_TRUST,
    ANTI_SYCOPHANCY,
]


def get_concept_by_id(concept_id: str) -> BehavioralConcept | None:
    """Get a concept by its ID."""
    for concept in CONCEPT_LIBRARY:
        if concept.id == concept_id:
            return concept
    return None


def get_all_concepts() -> list[BehavioralConcept]:
    """Get all concepts in the library."""
    return CONCEPT_LIBRARY.copy()
