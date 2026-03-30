"""Tests for dataset utilities."""

import json
import tempfile
from pathlib import Path

from semantic_forge.data_models import (
    CogSecScore,
    TrajectoryProfile,
    ContrastivePair,
    TrainingExample,
)
from semantic_forge.dataset import (
    filter_by_cogsec_score,
    build_dataset,
    save_dataset,
    load_dataset,
    compute_dataset_stats,
)


class TestDatasetFilters:
    """Test cases for dataset filtering functions."""

    def test_filter_by_cogsec_score_passes_clean_pair(self):
        """Test that a clean pair passes the filter."""
        pair = ContrastivePair(
            prompt="Test prompt",
            chosen="Clean response",
            rejected="Manipulative response",
            chosen_cogsec_score=CogSecScore(
                threat_level="Low",
                manipulation_score=0.05,
                structural_cleanliness=0.9,
            ),
            rejected_cogsec_score=CogSecScore(
                threat_level="Moderate",
                manipulation_score=0.35,
                structural_cleanliness=0.5,
            ),
            chosen_trajectory=TrajectoryProfile(
                mean_velocity=0.1,
                deadpan_score=0.8,
                acceleration_spikes=[],
            ),
            rejected_trajectory=TrajectoryProfile(
                mean_velocity=0.3,
                deadpan_score=0.2,
                acceleration_spikes=[],
            ),
            embedding_distance_chosen_rejected=0.3,
        )

        assert filter_by_cogsec_score(pair) is True

    def test_filter_by_cogsec_score_rejects_dirty_chosen(self):
        """Test that a dirty chosen completion fails the filter."""
        pair = ContrastivePair(
            prompt="Test prompt",
            chosen="Manipulative chosen",
            rejected="Bad response",
            chosen_cogsec_score=CogSecScore(
                threat_level="Moderate",
                manipulation_score=0.4,
                structural_cleanliness=0.5,  # Below threshold
            ),
            rejected_cogsec_score=CogSecScore(
                threat_level="High",
                manipulation_score=0.5,
                structural_cleanliness=0.3,
            ),
            chosen_trajectory=TrajectoryProfile(
                mean_velocity=0.3,
                deadpan_score=0.2,
                acceleration_spikes=[],
            ),
            rejected_trajectory=TrajectoryProfile(
                mean_velocity=0.4,
                deadpan_score=0.1,
                acceleration_spikes=[],
            ),
            embedding_distance_chosen_rejected=0.3,
        )

        assert filter_by_cogsec_score(pair) is False

    def test_filter_by_cogsec_score_rejects_low_manipulation_rejected(self):
        """Test that a rejected completion with low manipulation fails the filter."""
        pair = ContrastivePair(
            prompt="Test prompt",
            chosen="Clean response",
            rejected="Slightly manipulative",
            chosen_cogsec_score=CogSecScore(
                threat_level="Low",
                manipulation_score=0.05,
                structural_cleanliness=0.9,
            ),
            rejected_cogsec_score=CogSecScore(
                threat_level="Low",
                manipulation_score=0.1,  # Below threshold
                structural_cleanliness=0.8,
            ),
            chosen_trajectory=TrajectoryProfile(
                mean_velocity=0.1,
                deadpan_score=0.8,
                acceleration_spikes=[],
            ),
            rejected_trajectory=TrajectoryProfile(
                mean_velocity=0.15,
                deadpan_score=0.7,
                acceleration_spikes=[],
            ),
            embedding_distance_chosen_rejected=0.1,  # Below min distance
        )

        assert filter_by_cogsec_score(pair) is False


class TestDatasetIO:
    """Test cases for dataset I/O operations."""

    def test_save_and_load_jsonl(self):
        """Test saving and loading a dataset in JSONL format."""
        examples = [
            TrainingExample(
                concept="test_concept",
                mood="imperative",
                scenario="Test scenario",
                scenario_type="coding",
                prompt="Test prompt",
                chosen="Chosen response",
                rejected="Rejected response",
                chosen_cogsec_score=CogSecScore(
                    threat_level="Low",
                    manipulation_score=0.05,
                    structural_cleanliness=0.9,
                ),
                rejected_cogsec_score=CogSecScore(
                    threat_level="High",
                    manipulation_score=0.5,
                    structural_cleanliness=0.3,
                ),
                chosen_trajectory=TrajectoryProfile(
                    mean_velocity=0.1,
                    deadpan_score=0.8,
                    acceleration_spikes=[],
                ),
                rejected_trajectory=TrajectoryProfile(
                    mean_velocity=0.3,
                    deadpan_score=0.2,
                    acceleration_spikes=[],
                ),
                embedding_distance_chosen_rejected=0.3,
            )
        ]

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            save_dataset(examples, path)
            loaded = load_dataset(path)

            assert len(loaded) == 1
            assert loaded[0].concept == "test_concept"
            assert loaded[0].mood == "imperative"
        finally:
            Path(path).unlink()

    def test_save_and_load_json(self):
        """Test saving and loading a dataset in JSON format."""
        examples = [
            TrainingExample(
                concept="test_concept",
                mood="imperative",
                scenario="Test scenario",
                scenario_type="coding",
                prompt="Test prompt",
                chosen="Chosen response",
                rejected="Rejected response",
                chosen_cogsec_score=CogSecScore(
                    threat_level="Low",
                    manipulation_score=0.05,
                    structural_cleanliness=0.9,
                ),
                rejected_cogsec_score=CogSecScore(
                    threat_level="High",
                    manipulation_score=0.5,
                    structural_cleanliness=0.3,
                ),
                chosen_trajectory=TrajectoryProfile(
                    mean_velocity=0.1,
                    deadpan_score=0.8,
                    acceleration_spikes=[],
                ),
                rejected_trajectory=TrajectoryProfile(
                    mean_velocity=0.3,
                    deadpan_score=0.2,
                    acceleration_spikes=[],
                ),
                embedding_distance_chosen_rejected=0.3,
            )
        ]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            save_dataset(examples, path)
            loaded = load_dataset(path)

            assert len(loaded) == 1
            assert loaded[0].concept == "test_concept"
        finally:
            Path(path).unlink()


class TestDatasetStats:
    """Test cases for dataset statistics computation."""

    def test_compute_dataset_stats(self):
        """Test computing dataset statistics."""
        examples = [
            TrainingExample(
                concept="temporal_trust",
                mood="imperative",
                scenario="Scenario 1",
                scenario_type="coding",
                prompt="Prompt 1",
                chosen="Chosen 1",
                rejected="Rejected 1",
                chosen_cogsec_score=CogSecScore(
                    threat_level="Low",
                    manipulation_score=0.05,
                    structural_cleanliness=0.9,
                ),
                rejected_cogsec_score=CogSecScore(
                    threat_level="Moderate",
                    manipulation_score=0.3,
                    structural_cleanliness=0.6,
                ),
                chosen_trajectory=TrajectoryProfile(
                    mean_velocity=0.1,
                    deadpan_score=0.8,
                    acceleration_spikes=[],
                ),
                rejected_trajectory=TrajectoryProfile(
                    mean_velocity=0.3,
                    deadpan_score=0.2,
                    acceleration_spikes=[],
                ),
                embedding_distance_chosen_rejected=0.3,
            ),
            TrainingExample(
                concept="temporal_trust",
                mood="socratic",
                scenario="Scenario 2",
                scenario_type="financial",
                prompt="Prompt 2",
                chosen="Chosen 2",
                rejected="Rejected 2",
                chosen_cogsec_score=CogSecScore(
                    threat_level="Low",
                    manipulation_score=0.0,
                    structural_cleanliness=1.0,
                ),
                rejected_cogsec_score=CogSecScore(
                    threat_level="High",
                    manipulation_score=0.5,
                    structural_cleanliness=0.3,
                ),
                chosen_trajectory=TrajectoryProfile(
                    mean_velocity=0.1,
                    deadpan_score=0.8,
                    acceleration_spikes=[],
                ),
                rejected_trajectory=TrajectoryProfile(
                    mean_velocity=0.4,
                    deadpan_score=0.1,
                    acceleration_spikes=[],
                ),
                embedding_distance_chosen_rejected=0.4,
            ),
        ]

        stats = compute_dataset_stats(examples)

        assert stats.total_examples == 2
        assert stats.mood_distribution == {"imperative": 1, "socratic": 1}
        assert stats.scenario_coverage == {"coding": 1, "financial": 1}
        assert stats.mean_manipulation_score_chosen == 0.025
        assert stats.mean_manipulation_score_rejected == 0.4
