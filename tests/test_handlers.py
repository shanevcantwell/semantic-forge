"""Integration tests for semantic-forge handlers."""

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from semantic_forge.handlers import SemanticForgeHandlers, SemanticKinematicsRequiredError
from semantic_forge.data_models import (
    PermutatePhrasingResult,
    Rephrasing,
    Scenario,
    ContrastivePair,
    BuildDatasetResult,
)
from semantic_forge.mcp import (
    PermutatePhrasingParams,
    GenerateScenarioParams,
    GenerateContrastivePairParams,
    ScoreCompletionParams,
    ValidateDiversityParams,
    ValidateTrajectoryParams,
)


class TestHandlers:
    """Test cases for MCP handlers."""

    @pytest.fixture
    def handlers(self):
        """Get handler instance."""
        return SemanticForgeHandlers()

    @pytest.mark.asyncio
    async def test_handle_permutate_phrasing_returns_pydantic_model(self, handlers):
        """Test permutate_phrasing returns PermutatePhrasingResult model."""
        params = PermutatePhrasingParams(
            concept="Test concept",
            moods=["imperative", "declarative"],
            validate_diversity=False,
        )

        # Mock the LLM client
        with patch("semantic_forge.handlers.create_client") as mock_create:
            mock_client = AsyncMock()
            mock_client.generate = AsyncMock(side_effect=[
                "Do this thing.",
                "This is the way.",
            ])
            mock_create.return_value = mock_client

            result = await handlers.handle_permutate_phrasing(params)

            # Verify Pydantic model return type
            assert isinstance(result, PermutatePhrasingResult)
            assert result.concept == "Test concept"
            assert len(result.rephrasings) == 2
            assert isinstance(result.rephrasings[0], Rephrasing)
            assert result.rephrasings[0].mood == "imperative"
            assert result.rephrasings[0].text == "Do this thing."

    @pytest.mark.asyncio
    async def test_handle_generate_scenario_returns_pydantic_models(self, handlers):
        """Test generate_scenario returns list[Scenario] models."""
        params = GenerateScenarioParams(
            rephrased_concept="Test concept",
            scenario_types=["coding"],
            count=2,
        )

        with patch("semantic_forge.handlers.create_client") as mock_create:
            mock_client = AsyncMock()
            mock_client.generate = AsyncMock(return_value="A coding scenario")
            mock_create.return_value = mock_client

            result = await handlers.handle_generate_scenario(params)

            # Verify list of Pydantic models return type
            assert isinstance(result, list)
            assert len(result) == 2
            assert isinstance(result[0], Scenario)
            assert result[0].scenario_type == "coding"
            assert result[0].description == "A coding scenario"

    @pytest.mark.asyncio
    async def test_handle_score_completion(self, handlers):
        """Test score_completion handler uses CogSec scoring."""
        params = ScoreCompletionParams(
            completion="This is a clean response.",
            context="test_context",
        )

        result = await handlers.handle_score_completion(params)

        assert result.content[0].type == "text"
        data = json.loads(result.content[0].text)
        assert data["completion"] == "This is a clean response."
        assert "threat_level" in data
        assert "manipulation_score" in data
        assert "structural_cleanliness" in data

    @pytest.mark.asyncio
    async def test_handle_validate_diversity_no_sk(self, handlers):
        """Test validate_diversity when semantic-kinematics is not configured."""
        params = ValidateDiversityParams(
            rephrasings=["Text 1", "Text 2"],
        )

        # Mock to return no SK endpoint
        with patch("semantic_forge.handlers.get_semantic_kinematics_endpoint") as mock_sk:
            mock_sk.return_value = None

            result = await handlers.handle_validate_diversity(params)

            assert result.content[0].type == "text"
            data = json.loads(result.content[0].text)
            assert data["rephrasings_count"] == 2
            assert data["diversity_warning"] is not None

    @pytest.mark.asyncio
    async def test_handle_validate_trajectory_no_sk(self, handlers):
        """Test validate_trajectory when semantic-kinematics is not configured."""
        params = ValidateTrajectoryParams(
            completions=["Text 1", "Text 2"],
        )

        with patch("semantic_forge.handlers.get_semantic_kinematics_endpoint") as mock_sk:
            mock_sk.return_value = None

            result = await handlers.handle_validate_trajectory(params)

            assert result.content[0].type == "text"
            data = json.loads(result.content[0].text)
            assert data["completions_count"] == 2
            assert data["target_shape"] == "steady"

    @pytest.mark.asyncio
    async def test_handle_generate_contrastive_pair_fails_without_sk_config(self, handlers):
        """Test generate_contrastive_pair fails fast when SK-MCP is not configured."""
        params = GenerateContrastivePairParams(
            scenario="Test scenario",
            context="test_context",
        )

        # Mock to return no SK endpoint
        with patch("semantic_forge.handlers.get_semantic_kinematics_endpoint") as mock_sk:
            mock_sk.return_value = None

            with pytest.raises(SemanticKinematicsRequiredError) as exc_info:
                await handlers.handle_generate_contrastive_pair(params)

            assert "not configured" in str(exc_info.value)
            assert "SEMANTIC_KINEMATICS_ENDPOINT" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_handle_generate_contrastive_pair_fails_when_sk_unavailable(self, handlers):
        """Test generate_contrastive_pair fails fast when SK-MCP is unresponsive."""
        params = GenerateContrastivePairParams(
            scenario="Test scenario",
            context="test_context",
        )

        # Mock SK endpoint configured but client initialization fails
        with patch("semantic_forge.handlers.get_semantic_kinematics_endpoint") as mock_sk:
            mock_sk.return_value = "semantic-kinematics-mcp"

            with patch("semantic_forge.handlers.SemanticKinematicsClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.initialize = AsyncMock(side_effect=ConnectionError("Connection refused"))
                mock_client_class.return_value = mock_client

                with pytest.raises(SemanticKinematicsRequiredError) as exc_info:
                    await handlers.handle_generate_contrastive_pair(params)

                assert "unavailable" in str(exc_info.value)
                assert "Connection refused" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_handle_generate_contrastive_pair_success_with_sk(self, handlers):
        """Test generate_contrastive_pair succeeds when SK-MCP is available."""
        params = GenerateContrastivePairParams(
            scenario="Test scenario",
            context="temporal_trust",
        )

        # Mock SK endpoint and client
        with patch("semantic_forge.handlers.get_semantic_kinematics_endpoint") as mock_sk:
            mock_sk.return_value = "semantic-kinematics-mcp"

            with patch("semantic_forge.handlers.SemanticKinematicsClient") as mock_sk_client_class:
                mock_sk_client = AsyncMock()
                mock_sk_client.initialize = AsyncMock(return_value=None)
                mock_sk_client.model_status = AsyncMock(return_value={"status": "ok"})
                mock_sk_client.analyze_trajectory = AsyncMock(return_value={
                    "mean_velocity": 0.5,
                    "deadpan_score": 0.3,
                    "acceleration_spikes": [],
                    "torsion": 0.1,
                    "curvature": 0.2,
                })
                mock_sk_client.calculate_drift = AsyncMock(return_value={"drift": 0.4})
                mock_sk_client_class.return_value = mock_sk_client

                # Mock LLM client
                with patch("semantic_forge.handlers.create_client") as mock_llm_create:
                    mock_llm_client = AsyncMock()
                    mock_llm_client.generate_structured = AsyncMock(return_value={
                        "prompt": "Test scenario",
                        "chosen": "Clean response",
                        "rejected": "Manipulative response",
                    })
                    mock_llm_create.return_value = mock_llm_client

                    result = await handlers.handle_generate_contrastive_pair(params)

                    # Verify Pydantic model return type
                    assert isinstance(result, ContrastivePair)

                    # Verify all required fields are present
                    assert result.chosen_trajectory is not None
                    assert result.rejected_trajectory is not None
                    assert result.embedding_distance_chosen_rejected == 0.4

                    # Verify trajectory structure
                    assert result.chosen_trajectory.mean_velocity == 0.5
                    assert result.chosen_trajectory.deadpan_score == 0.3
                    assert result.chosen_trajectory.acceleration_spikes == []

                    # Verify embedding distance (already checked above)
                    assert result.embedding_distance_chosen_rejected == 0.4

    @pytest.mark.asyncio
    async def test_handle_generate_contrastive_pair_raises_runtime_error_on_llm_failure(self, handlers):
        """Test generate_contrastive_pair raises RuntimeError on LLM generation failure."""
        params = GenerateContrastivePairParams(
            scenario="Test scenario",
            context="test_context",
        )

        # Mock SK endpoint and client (successful)
        with patch("semantic_forge.handlers.get_semantic_kinematics_endpoint") as mock_sk:
            mock_sk.return_value = "semantic-kinematics-mcp"

            with patch("semantic_forge.handlers.SemanticKinematicsClient") as mock_sk_client_class:
                mock_sk_client = AsyncMock()
                mock_sk_client.initialize = AsyncMock(return_value=None)
                mock_sk_client.model_status = AsyncMock(return_value={"status": "ok"})
                mock_sk_client_class.return_value = mock_sk_client

                # Mock LLM client that fails
                with patch("semantic_forge.handlers.create_client") as mock_llm_create:
                    mock_llm_client = AsyncMock()
                    mock_llm_client.generate_structured = AsyncMock(side_effect=RuntimeError("LLM error"))
                    mock_llm_create.return_value = mock_llm_client

                    with pytest.raises(RuntimeError) as exc_info:
                        await handlers.handle_generate_contrastive_pair(params)

                    assert "Failed to generate contrastive pair" in str(exc_info.value)
