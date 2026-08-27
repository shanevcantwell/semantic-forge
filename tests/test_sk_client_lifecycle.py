"""Regression test for issue #7: SK stdio client lifecycle.

Root cause (see issue #7, comment 2026-08-27): no call site in
`handlers.py` ever called `sk_client.close()`. Each
`generate_contrastive_pair` call constructed a fresh
`SemanticKinematicsClient`, opened its stdio subprocess + anyio cancel
scope inside `initialize()`, and then abandoned it. The garbage
collector later finalized the leaked `stdio_client` async generator on
whatever task GC happened to run on, tripping anyio's "cancel scope
exited on a different task than it was entered in" invariant. That
surfaced as a `CancelledError` in the *next* concurrently-awaiting
in-process call — always the second call, regardless of the first
call's outcome.

This test drives two consecutive in-process `generate_contrastive_pair`
calls against a real stdio subprocess (a stub SK server; the real
`semantic-kinematics-mcp` binary need not be installed for this repo's
CI). Only the LLM side is mocked — the SK client's stdio transport,
anyio task-group, and cancel-scope teardown are exercised for real,
because those are exactly the mechanics the bug lived in. This test
fails before the try/finally close() fix (call 2 raises CancelledError
/ RuntimeError from the cancel-scope mismatch) and passes after.
"""
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from semantic_forge.data_models import ContrastivePair
from semantic_forge.handlers import SemanticForgeHandlers
from semantic_forge.mcp import (
    GenerateContrastivePairParams,
    PermutatePhrasingParams,
    ValidateDiversityParams,
    ValidateTrajectoryParams,
)

STUB_SERVER = Path(__file__).parent / "fixtures" / "stub_sk_server.py"
STUB_ENDPOINT = f"{sys.executable},{STUB_SERVER}"


class TestSemanticKinematicsClientLifecycle:
    """Two consecutive in-process generate_contrastive_pair calls must both succeed."""

    @pytest.fixture
    def handlers(self):
        return SemanticForgeHandlers()

    @pytest.fixture
    def mock_llm(self):
        """Mock only the target LLM side; SK client is real (stub subprocess)."""
        with patch("semantic_forge.handlers.create_client") as mock_create:
            mock_client = AsyncMock()
            mock_client.generate_structured = AsyncMock(return_value={
                "prompt": "Test scenario",
                "chosen": "Clean response",
                "rejected": "Manipulative response",
            })
            mock_create.return_value = mock_client
            yield mock_create

    @pytest.fixture
    def sk_endpoint(self):
        with patch("semantic_forge.handlers.get_semantic_kinematics_endpoint") as mock_ep:
            mock_ep.return_value = STUB_ENDPOINT
            yield mock_ep

    @pytest.mark.asyncio
    async def test_two_consecutive_calls_complete_without_cancelled_error(
        self, handlers, mock_llm, sk_endpoint
    ):
        """Acceptance criterion (issue #7): two consecutive in-process
        generate_contrastive_pair calls complete without CancelledError."""
        params = GenerateContrastivePairParams(
            scenario="Test scenario",
            context="test_context",
        )

        # Call 1 — this always succeeded even before the fix.
        result_1 = await handlers.handle_generate_contrastive_pair(params)
        assert isinstance(result_1, ContrastivePair)

        # Call 2 — this is the call that died with CancelledError before the fix,
        # because call 1's abandoned SK client got GC'd mid-flight and tore down
        # its cancel scope on the wrong task.
        result_2 = await handlers.handle_generate_contrastive_pair(params)
        assert isinstance(result_2, ContrastivePair)

    @pytest.mark.asyncio
    async def test_sk_client_closed_after_each_call(self, handlers, mock_llm, sk_endpoint):
        """The session must be resolved-used-released per call: close() called
        exactly once per generate_contrastive_pair invocation, on the same task
        that opened it."""
        params = GenerateContrastivePairParams(
            scenario="Test scenario",
            context="test_context",
        )

        with patch("semantic_forge.handlers.SemanticKinematicsClient") as mock_client_class:
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
            mock_client_class.return_value = mock_sk_client

            await handlers.handle_generate_contrastive_pair(params)

            mock_sk_client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_sk_client_closed_even_on_generation_failure(self, handlers, sk_endpoint):
        """close() must run even when the LLM generation step raises, so a
        failed call never leaks its SK session into the next in-process call."""
        params = GenerateContrastivePairParams(
            scenario="Test scenario",
            context="test_context",
        )

        with patch("semantic_forge.handlers.SemanticKinematicsClient") as mock_client_class:
            mock_sk_client = AsyncMock()
            mock_sk_client.initialize = AsyncMock(return_value=None)
            mock_sk_client.model_status = AsyncMock(return_value={"status": "ok"})
            mock_client_class.return_value = mock_sk_client

            with patch("semantic_forge.handlers.create_client") as mock_llm_create:
                mock_llm_client = AsyncMock()
                mock_llm_client.generate_structured = AsyncMock(
                    side_effect=RuntimeError("LLM error")
                )
                mock_llm_create.return_value = mock_llm_client

                with pytest.raises(RuntimeError):
                    await handlers.handle_generate_contrastive_pair(params)

            mock_sk_client.close.assert_awaited_once()


class TestOtherSkClientCallSitesCloseSession:
    """The other three SK client call sites (issue #7 also names ~128, 371, 446)
    share the identical leak and must share the identical fix."""

    @pytest.fixture
    def handlers(self):
        return SemanticForgeHandlers()

    def _mock_sk_client_class(self, patcher_target):
        mock_sk_client = AsyncMock()
        mock_sk_client.initialize = AsyncMock(return_value=None)
        mock_sk_client._get_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
        mock_sk_client.calculate_drift = AsyncMock(return_value={
            "mean_pairwise_drift": 0.3,
            "drift": 0.3,
        })
        mock_sk_client.analyze_trajectory = AsyncMock(return_value={
            "mean_velocity": 0.5,
            "deadpan_score": 0.3,
            "acceleration_spikes": [],
        })
        return mock_sk_client

    @pytest.mark.asyncio
    async def test_permutate_phrasing_closes_sk_client(self, handlers):
        """handle_permutate_phrasing's validate_diversity path (~line 128)."""
        params = PermutatePhrasingParams(
            concept="Test concept",
            moods=["imperative"],
            validate_diversity=True,
        )

        with patch("semantic_forge.handlers.create_client") as mock_llm_create:
            mock_llm_client = AsyncMock()
            mock_llm_client.generate = AsyncMock(return_value="Do this thing.")
            mock_llm_create.return_value = mock_llm_client

            with patch("semantic_forge.handlers.get_semantic_kinematics_endpoint") as mock_ep:
                mock_ep.return_value = "semantic-kinematics-mcp"

                with patch("semantic_forge.handlers.SemanticKinematicsClient") as mock_cls:
                    mock_sk_client = self._mock_sk_client_class(mock_cls)
                    mock_cls.return_value = mock_sk_client

                    await handlers.handle_permutate_phrasing(params)

                    mock_sk_client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_validate_diversity_closes_sk_client(self, handlers):
        """handle_validate_diversity's SK client (~line 371)."""
        params = ValidateDiversityParams(rephrasings=["Text 1", "Text 2"])

        with patch("semantic_forge.handlers.get_semantic_kinematics_endpoint") as mock_ep:
            mock_ep.return_value = "semantic-kinematics-mcp"

            with patch("semantic_forge.handlers.SemanticKinematicsClient") as mock_cls:
                mock_sk_client = self._mock_sk_client_class(mock_cls)
                mock_cls.return_value = mock_sk_client

                await handlers.handle_validate_diversity(params)

                mock_sk_client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_validate_diversity_closes_sk_client_on_error(self, handlers):
        """close() must run even when the SK call raises mid-loop."""
        params = ValidateDiversityParams(rephrasings=["Text 1", "Text 2"])

        with patch("semantic_forge.handlers.get_semantic_kinematics_endpoint") as mock_ep:
            mock_ep.return_value = "semantic-kinematics-mcp"

            with patch("semantic_forge.handlers.SemanticKinematicsClient") as mock_cls:
                mock_sk_client = AsyncMock()
                mock_sk_client.initialize = AsyncMock(return_value=None)
                mock_sk_client.calculate_drift = AsyncMock(side_effect=RuntimeError("boom"))
                mock_cls.return_value = mock_sk_client

                # This handler catches the error internally and returns a result;
                # it must not propagate, but close() must still have run.
                await handlers.handle_validate_diversity(params)

                mock_sk_client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_validate_trajectory_closes_sk_client(self, handlers):
        """handle_validate_trajectory's SK client (~line 446)."""
        params = ValidateTrajectoryParams(completions=["Text 1", "Text 2"])

        with patch("semantic_forge.handlers.get_semantic_kinematics_endpoint") as mock_ep:
            mock_ep.return_value = "semantic-kinematics-mcp"

            with patch("semantic_forge.handlers.SemanticKinematicsClient") as mock_cls:
                mock_sk_client = self._mock_sk_client_class(mock_cls)
                mock_cls.return_value = mock_sk_client

                await handlers.handle_validate_trajectory(params)

                mock_sk_client.close.assert_awaited_once()
