"""Dataset utilities for semantic-forge.

This module provides utilities for building, filtering, and exporting
training datasets.
"""

import json
import random
from pathlib import Path
from typing import Any

from semantic_forge.data_models import (
    TrainingExample,
    DatasetStats,
    ContrastivePair,
)


def filter_by_cogsec_score(
    pair: ContrastivePair,
    chosen_min_cleanliness: float = 0.7,
    rejected_max_manipulation: float = 0.3,
) -> bool:
    """
    Filter a contrastive pair based on CogSec scores.

    Args:
        pair: The contrastive pair to filter
        chosen_min_cleanliness: Minimum structural cleanliness for chosen
        rejected_max_manipulation: Maximum manipulation score for rejected

    Returns:
        True if the pair passes the filter
    """
    chosen_pass = pair.chosen_cogsec_score.structural_cleanliness >= chosen_min_cleanliness
    rejected_pass = pair.rejected_cogsec_score.manipulation_score >= rejected_max_manipulation
    return chosen_pass and rejected_pass


def filter_by_embedding_distance(
    pair: ContrastivePair,
    min_distance: float = 0.1,
    max_distance: float = 0.8,
) -> bool:
    """
    Filter a contrastive pair based on embedding distance.

    Args:
        pair: The contrastive pair to filter
        min_distance: Minimum embedding distance
        max_distance: Maximum embedding distance

    Returns:
        True if the pair passes the filter
    """
    distance = pair.embedding_distance_chosen_rejected
    return min_distance <= distance <= max_distance


def build_dataset(
    concept: str,
    rephrasings: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
    contrastive_pairs: list[ContrastivePair],
    output_path: str | None = None,
) -> list[TrainingExample]:
    """
    Build a training dataset from rephrasings, scenarios, and contrastive pairs.

    Args:
        concept: The core concept
        rephrasings: List of rephrasing dictionaries
        scenarios: List of scenario dictionaries
        contrastive_pairs: List of contrastive pairs
        output_path: Optional path to save the dataset

    Returns:
        List of TrainingExample objects
    """
    examples = []

    for i, pair in enumerate(contrastive_pairs):
        # Select mood and scenario for this example
        rephrasing = rephrasings[i % len(rephrasings)]
        scenario = scenarios[i % len(scenarios)]

        example = TrainingExample(
            concept=concept,
            mood=rephrasing.get("mood", "unknown"),
            scenario=scenario.get("description", ""),
            scenario_type=scenario.get("scenario_type", "unknown"),
            prompt=pair.prompt,
            chosen=pair.chosen,
            rejected=pair.rejected,
            chosen_cogsec_score=pair.chosen_cogsec_score,
            rejected_cogsec_score=pair.rejected_cogsec_score,
            chosen_trajectory=pair.chosen_trajectory,
            rejected_trajectory=pair.rejected_trajectory,
            embedding_distance_chosen_rejected=pair.embedding_distance_chosen_rejected,
        )
        examples.append(example)

    # Save to file if path provided
    if output_path:
        save_dataset(examples, output_path)

    return examples


def save_dataset(examples: list[TrainingExample], output_path: str) -> None:
    """Save a dataset to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix.lower()

    if ext == ".jsonl":
        with open(path, "w") as f:
            for example in examples:
                f.write(example.model_dump_json() + "\n")
    elif ext == ".json":
        with open(path, "w") as f:
            json.dump([e.model_dump() for e in examples], f, indent=2)
    else:
        raise ValueError(f"Unsupported output format: {ext}")


def load_dataset(path: str) -> list[TrainingExample]:
    """Load a dataset from a file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    ext = path.suffix.lower()

    if ext == ".jsonl":
        examples = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    examples.append(TrainingExample(**data))
        return examples
    elif ext == ".json":
        with open(path) as f:
            data = json.load(f)
        return [TrainingExample(**d) for d in data]

    raise ValueError(f"Unsupported input format: {ext}")


def compute_dataset_stats(examples: list[TrainingExample]) -> DatasetStats:
    """Compute statistics for a dataset."""
    mood_distribution: dict[str, int] = {}
    scenario_coverage: dict[str, int] = {}
    manipulation_scores_chosen: list[float] = []
    manipulation_scores_rejected: list[float] = []

    for example in examples:
        # Mood distribution
        mood = example.mood
        mood_distribution[mood] = mood_distribution.get(mood, 0) + 1

        # Scenario coverage
        scenario_type = example.scenario_type
        scenario_coverage[scenario_type] = scenario_coverage.get(scenario_type, 0) + 1

        # Manipulation scores
        manipulation_scores_chosen.append(example.chosen_cogsec_score.manipulation_score)
        manipulation_scores_rejected.append(example.rejected_cogsec_score.manipulation_score)

    # Score distribution by threat level
    score_distribution: dict[str, dict[str, int]] = {
        "chosen": {"Low": 0, "Moderate": 0, "High": 0, "ACTIVE_INJECTION": 0},
        "rejected": {"Low": 0, "Moderate": 0, "High": 0, "ACTIVE_INJECTION": 0},
    }
    for example in examples:
        score_distribution["chosen"][example.chosen_cogsec_score.threat_level] += 1
        score_distribution["rejected"][example.rejected_cogsec_score.threat_level] += 1

    # Embedding spread
    embedding_distances = [
        e.embedding_distance_chosen_rejected for e in examples
    ]
    embedding_spread = {
        "mean": sum(embedding_distances) / len(embedding_distances) if embedding_distances else 0,
        "min": min(embedding_distances) if embedding_distances else 0,
        "max": max(embedding_distances) if embedding_distances else 0,
    }

    return DatasetStats(
        total_examples=len(examples),
        mood_distribution=mood_distribution,
        scenario_coverage=scenario_coverage,
        score_distribution=score_distribution,
        embedding_spread=embedding_spread,
        mean_manipulation_score_chosen=sum(manipulation_scores_chosen) / len(manipulation_scores_chosen) if manipulation_scores_chosen else 0,
        mean_manipulation_score_rejected=sum(manipulation_scores_rejected) / len(manipulation_scores_rejected) if manipulation_scores_rejected else 0,
    )


def export_for_dpo(examples: list[TrainingExample], output_path: str) -> None:
    """Export dataset in DPO-compatible format."""
    dpo_examples = []
    for example in examples:
        dpo_examples.append({
            "prompt": example.prompt,
            "chosen": example.chosen,
            "rejected": example.rejected,
        })

    save_dataset(dpo_examples, output_path)


def export_with_metadata(examples: list[TrainingExample], output_path: str) -> None:
    """Export dataset with full metadata (default format)."""
    # TrainingExample already has the full format
    save_dataset(examples, output_path)
