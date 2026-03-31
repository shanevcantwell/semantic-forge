"""Integration tests for semantic-forge handlers."""

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from semantic_forge.handlers import SemanticForgeHandlers
from semantic_forge.mcp import (
    PermutatePhrasingParams,
    GenerateScenarioParams,
    GenerateContrastivePairParams,
    ScoreCompletionParams,
    ValidateDiversityParams,
    ValidateTrajectoryParams,
    BuildDatasetParams,
    DatasetStatsParams,
)


class TestHandlers:
    """Test cases for MCP handlers."""

    @pytest.fixture
    def handlers(self):
        """Get handler instance."""
        return SemanticForgeHandlers()

    @pytest.mark.asyncio
    async def test_handle_permutate_phrasing_basic(self, handlers):
        """Test basic permutate_phrasing handler functionality."""
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

            assert result.content[0].type == "text"
            data = json.loads(result.content[0].text)
            assert data["concept"] == "Test concept"
            assert len(data["rephrasings"]) == 2
            assert data["rephrasings"][0]["mood"] == "imperative"
            assert data["rephrasings"][0]["text"] == "Do this thing."

    @pytest.mark.asyncio
    async def test_handle_generate_scenario_basic(self, handlers):
        """Test basic generate_scenario handler functionality."""
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

            assert result.content[0].type == "text"
            data = json.loads(result.content[0].text)
            assert data["rephrased_concept"] == "Test concept"
            assert len(data["scenarios"]) == 2
            assert data["scenarios"][0]["scenario_type"] == "coding"

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
