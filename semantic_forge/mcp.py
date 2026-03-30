"""MCP server implementation for semantic-forge toolkit."""

import asyncio
import json
from mcp.server import Server
from mcp.types import Tool
from pydantic import BaseModel, Field
from typing import Any


class PermutatePhrasingParams(BaseModel):
    """Parameters for the permutate_phrasing tool."""
    concept: str = Field(description="The core behavioral statement to rephrase")
    moods: list[str] = Field(
        default_factory=lambda: ["imperative", "declarative", "first_plural", "past_perfect",
                                  "conditional", "socratic", "negation"],
        description="Grammatical moods to use for rephrasing"
    )
    model: str | None = Field(
        default=None,
        description="Override inference backend for rephrasing"
    )
    validate_diversity: bool = Field(
        default=True,
        description="Check embedding spread via sk-mcp"
    )


class GenerateScenarioParams(BaseModel):
    """Parameters for the generate_scenario tool."""
    rephrased_concept: str = Field(description="The rephrased concept statement")
    scenario_types: list[str] = Field(
        default_factory=lambda: ["financial", "coding", "research", "casual"],
        description="Types of scenarios to generate"
    )
    count: int = Field(default=3, description="Number of scenarios to generate")


class GenerateContrastivePairParams(BaseModel):
    """Parameters for the generate_contrastive_pair tool."""
    scenario: str = Field(description="The situated scenario")
    context: str = Field(description="Context that produced the completion")


class ScoreCompletionParams(BaseModel):
    """Parameters for the score_completion tool."""
    completion: str = Field(description="Model output to audit")
    context: str | None = Field(default=None, description="Prompt/scenario that produced it")
    criteria: str = Field(default="cogsec", description="Scoring rubric to apply")


class ValidateDiversityParams(BaseModel):
    """Parameters for the validate_diversity tool."""
    rephrasings: list[str] = Field(description="List of rephrased concept statements")
    threshold_min: float = Field(default=0.2, description="Minimum mean pairwise drift")
    threshold_max: float = Field(default=0.5, description="Maximum mean pairwise drift")


class ValidateTrajectoryParams(BaseModel):
    """Parameters for the validate_trajectory tool."""
    completions: list[str] = Field(description="Completions to analyze")
    target_shape: str = Field(
        default="steady",
        description="Expected trajectory shape (steady, high_acceleration, circular, etc.)"
    )


class BuildDatasetParams(BaseModel):
    """Parameters for the build_dataset tool."""
    concept: str = Field(description="The core behavioral concept")
    rephrasing_count: int = Field(default=5, description="Number of rephrasings per concept")
    scenarios_per_rephrasing: int = Field(default=3, description="Scenarios per rephrasing")
    output_format: str = Field(default="jsonl", description="Output format (jsonl, parquet)")


class DatasetStatsParams(BaseModel):
    """Parameters for the dataset_stats tool."""
    dataset_path: str = Field(description="Path to the generated dataset file")


# Tool definitions
PERMUTATE_PHRASING_TOOL = Tool(
    name="permutate_phrasing",
    description=(
        "Takes a concept statement and returns N rephrasings across grammatical moods. "
        "Each rephrasing embeds the same behavioral target from a different grammatical angle, "
        "preventing the model from escaping the concept by pattern-matching against one phrasing."
    ),
    inputSchema=PermutatePhrasingParams.model_json_schema()
)

GENERATE_SCENARIO_TOOL = Tool(
    name="generate_scenario",
    description=(
        "Takes a rephrased concept and generates a situated scenario. "
        "Scenarios ground abstract behavioral concepts in concrete domains "
        "(financial, coding, research, casual) for realistic training data."
    ),
    inputSchema=GenerateScenarioParams.model_json_schema()
)

GENERATE_CONTRASTIVE_PAIR_TOOL = Tool(
    name="generate_contrastive_pair",
    description=(
        "Given a scenario, produces a good completion and a bad completion. "
        "The good completion demonstrates the desired behavior, while the bad "
        "completion shows the failure mode the concept aims to correct."
    ),
    inputSchema=GenerateContrastivePairParams.model_json_schema()
)

SCORE_COMPLETION_TOOL = Tool(
    name="score_completion",
    description=(
        "Runs CogSec adversarial audit on a completion, returns threat level and detected mechanics. "
        "The CogSec judge scores for structural manipulation rather than correctness, "
        "inverting standard RLHF to prevent sycophancy and manipulation patterns."
    ),
    inputSchema=ScoreCompletionParams.model_json_schema()
)

VALIDATE_DIVERSITY_TOOL = Tool(
    name="validate_diversity",
    description=(
        "Calls semantic-kinematics-mcp to verify that a set of rephrasings occupy distinct "
        "embedding positions. Prevents redundant training data where different surface forms "
        "collapse to similar embeddings."
    ),
    inputSchema=ValidateDiversityParams.model_json_schema()
)

VALIDATE_TRAJECTORY_TOOL = Tool(
    name="validate_trajectory",
    description=(
        "Calls semantic-kinematics-mcp to verify that a completion's trajectory matches the "
        "target shape. Ensures contrastive pairs are actually contrastive in embedding space, "
        "not just in surface tokens."
    ),
    inputSchema=ValidateTrajectoryParams.model_json_schema()
)

BUILD_DATASET_TOOL = Tool(
    name="build_dataset",
    description=(
        "End-to-end pipeline: concept → rephrasings → scenarios → pairs → scored → filtered. "
        "Produces a high-quality training dataset with full adversarial scoring and "
        "geometric metadata for downstream fine-tuning."
    ),
    inputSchema=BuildDatasetParams.model_json_schema()
)

DATASET_STATS_TOOL = Tool(
    name="dataset_stats",
    description=(
        "Summary statistics on a generated dataset: mood distribution, scenario coverage, "
        "embedding spread, score distribution. Useful for verifying dataset quality and "
        "identifying coverage gaps."
    ),
    inputSchema=DatasetStatsParams.model_json_schema()
)


def get_all_tools() -> list[Tool]:
    """Return all available MCP tools."""
    return [
        PERMUTATE_PHRASING_TOOL,
        GENERATE_SCENARIO_TOOL,
        GENERATE_CONTRASTIVE_PAIR_TOOL,
        SCORE_COMPLETION_TOOL,
        VALIDATE_DIVERSITY_TOOL,
        VALIDATE_TRAJECTORY_TOOL,
        BUILD_DATASET_TOOL,
        DATASET_STATS_TOOL,
    ]


async def create_mcp_server() -> Server:
    """Create and configure the MCP server."""
    server = Server("semantic-forge")

    # Register tools with handlers
    # Note: Actual handlers will be implemented in semantic_forge.handlers

    return server
