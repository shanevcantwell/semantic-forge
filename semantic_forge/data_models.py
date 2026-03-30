"""Data models for semantic-forge output formats."""

from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


class Rephrasing(BaseModel):
    """A single rephrasing of a concept."""
    mood: str
    text: str
    embedding_distance_from_original: float | None = None


class PermutatePhrasingResult(BaseModel):
    """Result of the permutate_phrasing tool."""
    concept: str
    rephrasings: list[Rephrasing]
    spread_score: float
    diversity_warning: str | None = None


class Scenario(BaseModel):
    """A situated scenario for training data generation."""
    scenario_id: str
    scenario_type: str
    description: str
    domain: str  # financial, coding, research, casual


class CogSecScore(BaseModel):
    """CogSec adversarial audit result."""
    threat_level: str = Field(
        pattern="^(Low|Moderate|High|ACTIVE_INJECTION)$"
    )
    manipulation_score: float = Field(ge=0.0, le=1.0)
    structural_cleanliness: float = Field(ge=0.0, le=1.0)
    detected_mechanics: list[str] = Field(default_factory=list)


class TrajectoryProfile(BaseModel):
    """Trajectory shape analysis result."""
    mean_velocity: float
    deadpan_score: float
    acceleration_spikes: list[dict[str, Any]] = Field(default_factory=list)
    torsion: float | None = None
    curvature: float | None = None


class ContrastivePair(BaseModel):
    """A contrastive training pair."""
    prompt: str
    chosen: str
    rejected: str
    chosen_cogsec_score: CogSecScore
    rejected_cogsec_score: CogSecScore
    chosen_trajectory: TrajectoryProfile
    rejected_trajectory: TrajectoryProfile
    embedding_distance_chosen_rejected: float


class TrainingExample(BaseModel):
    """A complete training example with full metadata."""
    concept: str
    mood: str
    scenario: str
    scenario_type: str
    prompt: str
    chosen: str
    rejected: str
    chosen_cogsec_score: CogSecScore
    rejected_cogsec_score: CogSecScore
    chosen_trajectory: TrajectoryProfile
    rejected_trajectory: TrajectoryProfile
    embedding_distance_chosen_rejected: float
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    source_model: str | None = None
    rephrasing_embedding_distance: float | None = None


class DatasetStats(BaseModel):
    """Statistics on a generated dataset."""
    total_examples: int
    mood_distribution: dict[str, int]
    scenario_coverage: dict[str, int]
    score_distribution: dict[str, dict[str, int]]
    embedding_spread: dict[str, float]
    mean_manipulation_score_chosen: float
    mean_manipulation_score_rejected: float


class BuildDatasetResult(BaseModel):
    """Result of the build_dataset tool."""
    concept: str
    rephrasing_count: int
    scenarios_per_rephrasing: int
    output_format: str
    output_path: str
    example_count: int
    stats: DatasetStats
