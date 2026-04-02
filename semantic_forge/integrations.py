"""Integration helpers for external MCP tools.

This module provides integration with semantic-kinematics-mcp and prompt-prix.
"""

import contextlib
import asyncio
import json
from typing import Any
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp import ClientSession


class SemanticKinematicsClient:
    """Client for semantic-kinematics-mcp integration.

    Connects to semantic-kinematics-mcp via stdio MCP protocol to perform:
    - Embedding generation (embed_text)
    - Semantic drift calculation (calculate_drift)
    - Trajectory analysis (analyze_trajectory)
    - Trajectory comparison (compare_trajectories)
    - Backend management (model_status, model_load, model_unload)

    TODO: Add backend configuration to __init__ for runtime backend switching
    """

    def __init__(
        self,
        endpoint: str | None = None,
        backend: str | None = None,  # TODO: Use this to request specific backend at runtime
        base_url: str | None = None,  # TODO: LM Studio API URL (e.g., http://localhost:1234/v1)
        model_name: str | None = None,  # TODO: Model name for API backends
    ):
        """
        Initialize the semantic-kinematics client.

        Args:
            endpoint: Optional MCP server endpoint (command). Defaults to
                      'semantic-kinematics-mcp' if not provided.
            backend: Optional backend to use (nv_embed, lmstudio, sentence_transformers).
                     TODO: Implement backend switching via model_load tool
            base_url: API URL for lmstudio backend.
                      TODO: Pass to model_load when backend=lmstudio
            model_name: Model name for API backends.
                        TODO: Pass to model_load when using lmstudio or sentence_transformers
        """
        self.endpoint = endpoint or "semantic-kinematics-mcp"
        self.backend_config = {
            "backend": backend,
            "base_url": base_url,
            "model_name": model_name,
        }
        self._initialized = False
        self._session: ClientSession | None = None
        self._exit_stack: contextlib.AsyncExitStack | None = None

    async def initialize(self) -> bool:
        """Initialize connection to semantic-kinematics-mcp via stdio.

        Uses AsyncExitStack to manage the lifecycle of stdio transport and session.

        Uses self.endpoint as the command to execute. Supports:
        - Direct command: "semantic-kinematics-mcp"
        - Docker command: "docker" with args like ["run", "-i", "--rm", "semantic-kinematics-mcp"]

        Returns:
            True if initialization succeeds

        Raises:
            RuntimeError: If initialization fails
        """
        if self._initialized:
            return True

        try:
            # Create exit stack for managing async contexts
            self._exit_stack = contextlib.AsyncExitStack()
            await self._exit_stack.__aenter__()

            # Parse endpoint
            command, args, env = self._parse_endpoint(self.endpoint)
            params = StdioServerParameters(
                command=command,
                args=args,
                env=env,
            )

            # Create stdio transport and enter the context
            stdio_context = stdio_client(params)
            read, write = await self._exit_stack.enter_async_context(stdio_context)

            # Create and enter session
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(
                    read_stream=read,
                    write_stream=write,
                )
            )

            # Complete MCP handshake
            await self._session.initialize()

            # Ensure correct backend is loaded
            await self._ensure_backend()

            self._initialized = True
            return True

        except Exception as e:
            # If initialization fails, clean up the exit stack
            if self._exit_stack:
                await self._exit_stack.aclose()
            raise RuntimeError(f"Failed to initialize semantic-kinematics-mcp: {e}")

    def _parse_endpoint(self, endpoint: str) -> tuple[str, list[str], dict | None]:
        """Parse endpoint string into command, args, and env.

        Supports formats:
        - Simple command: "semantic-kinematics-mcp"
        - Docker with image: "docker:semantic-kinematics-mcp"
        - Full docker: "docker:run,-i,--rm,network=host,image-name"

        Returns:
            Tuple of (command, args, env)
        """
        if endpoint.startswith("docker:"):
            # Docker format
            docker_args = endpoint[7:].split(",")
            command = "docker"
            # If first arg is "run", use docker_args as-is; otherwise prepend defaults
            if docker_args and docker_args[0] == "run":
                args = docker_args
            else:
                # Simple docker:image-name format
                args = ["run", "-i", "--rm"] + docker_args
            env = None
        elif "," in endpoint:
            # Command with args: "command,arg1,arg2"
            parts = endpoint.split(",")
            command = parts[0]
            args = parts[1:]
            env = None
        else:
            # Simple command
            command = endpoint
            args = []
            env = None

        return command, args, env

    async def _ensure_backend(self) -> None:
        """Ensure the configured backend is loaded.

        MVP implementation: Calls model_load with backend_config if backend is specified.

        TODO: Add error handling for failed model_load
        TODO: Check current status before switching to avoid unnecessary reloads
        TODO: Graceful fallback if backend unavailable
        """
        if not self.backend_config.get("backend"):
            # No backend configured - rely on SK-MCP .env configuration
            return

        # Build load args from config
        load_args = {
            k: v for k, v in self.backend_config.items() if v is not None
        }

        # TODO: Add error handling - what if model_load fails?
        # For MVP, let errors propagate
        await self._call_tool("model_load", load_args)

    async def model_status(self) -> dict[str, Any]:
        """Check current embedding backend status.

        Returns:
            Dictionary with backend, model_name, is_loaded, dimensions, cache_size

        TODO: Add this to the public API docs
        """
        return await self._call_tool("model_status", {})

    async def model_load(
        self,
        backend: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        """Load or switch to a specific embedding backend.

        Args:
            backend: Backend type (nv_embed, lmstudio, sentence_transformers)
            base_url: API URL for lmstudio backend
            model_name: Model name for API backends

        Returns:
            Dictionary with status, backend, model_name, is_loaded

        TODO: Add this to the public API docs
        """
        load_args = {}
        if backend:
            load_args["backend"] = backend
        if base_url:
            load_args["base_url"] = base_url
        if model_name:
            load_args["model_name"] = model_name

        return await self._call_tool("model_load", load_args)

    async def model_unload(self, clear_cache: bool = True) -> dict[str, Any]:
        """Unload the current embedding backend.

        Args:
            clear_cache: Also clear the embedding cache

        Returns:
            Dictionary with status and cache_entries_cleared

        TODO: Add this to the public API docs
        """
        return await self._call_tool("model_unload", {"clear_cache": clear_cache})

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call an MCP tool and return parsed JSON result."""
        if not self._session:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        try:
            result = await self._session.call_tool(name, arguments)

            # Extract text content from result
            if result.content:
                for content in result.content:
                    if content.type == "text":
                        return json.loads(content.text)

            return {"error": "No text content in response"}

        except Exception as e:
            return {"error": str(e)}

    async def embed_text(
        self, text: str, full_vector: bool = False, model: str | None = None
    ) -> dict[str, Any]:
        """Get embedding vector for text.

        Args:
            text: Text to embed
            full_vector: Return full embedding vector (can be large)
            model: Embedding model to use

        Returns:
            Dictionary with embedding_preview or embedding field
        """
        return await self._call_tool(
            "embed_text",
            {
                "text": text,
                "full_vector": full_vector,
                "model": model or "nomic-embed-text-v1.5",
            },
        )

    async def calculate_drift(self, text_a: str, text_b: str) -> dict[str, Any]:
        """Calculate semantic drift (cosine distance) between two texts.

        Args:
            text_a: First text
            text_b: Second text

        Returns:
            Dictionary with drift (0.0-2.0) and interpretation
        """
        return await self._call_tool(
            "calculate_drift",
            {"text_a": text_a, "text_b": text_b},
        )

    async def calculate_drift_from_embeddings(
        self, embeddings: list[list[float]]
    ) -> dict[str, Any]:
        """Calculate pairwise drift between multiple embeddings.

        Args:
            embeddings: List of embedding vectors

        Returns:
            Dictionary with drift statistics
        """
        if len(embeddings) < 2:
            return {
                "mean_pairwise_drift": 0.0,
                "min_drift": 0.0,
                "max_drift": 0.0,
                "drift_matrix": [],
            }

        # Calculate pairwise drifts
        drift_matrix = []
        drifts = []

        for i in range(len(embeddings)):
            row = []
            for j in range(len(embeddings)):
                if i == j:
                    drift = 0.0
                elif j > i:
                    # Get drift for this pair
                    # Note: We can't call embed_text here, so we need to compute
                    # cosine distance from the embeddings directly
                    import numpy as np

                    v1 = np.array(embeddings[i])
                    v2 = np.array(embeddings[j])
                    norm1 = np.linalg.norm(v1)
                    norm2 = np.linalg.norm(v2)

                    if norm1 < 1e-10 or norm2 < 1e-10:
                        drift = 0.0
                    else:
                        cosine = np.dot(v1, v2) / (norm1 * norm2)
                        # Convert cosine to drift (cosine range [-1,1] -> drift range [0,2])
                        drift = float(np.arccos(np.clip(cosine, -1.0, 1.0)) / np.pi)

                    drifts.append(drift)
                    row.append(round(drift, 4))
                else:
                    row.append(drift_matrix[j][i])
            drift_matrix.append(row)

        return {
            "mean_pairwise_drift": round(sum(drifts) / len(drifts), 4) if drifts else 0.0,
            "min_drift": round(min(drifts), 4) if drifts else 0.0,
            "max_drift": round(max(drifts), 4) if drifts else 0.0,
            "drift_matrix": drift_matrix,
        }

    async def analyze_trajectory(
        self, text: str, acceleration_threshold: float = 0.3, include_sentences: bool = False
    ) -> dict[str, Any]:
        """Analyze semantic trajectory of a text passage.

        Args:
            text: Text passage to analyze (needs 2+ sentences)
            acceleration_threshold: Threshold for acceleration spikes
            include_sentences: Include sentence breakdown in output

        Returns:
            Dictionary with trajectory metrics
        """
        return await self._call_tool(
            "analyze_trajectory",
            {
                "text": text,
                "acceleration_threshold": acceleration_threshold,
                "include_sentences": include_sentences,
            },
        )

    async def compare_trajectories(
        self, golden_text: str, synthetic_text: str, acceleration_threshold: float = 0.3
    ) -> dict[str, Any]:
        """Compare two trajectory profiles.

        Args:
            golden_text: Reference passage (target structure)
            synthetic_text: Passage to compare against the reference
            acceleration_threshold: Threshold for acceleration spikes

        Returns:
            Dictionary with comparison results including fitness_score
        """
        return await self._call_tool(
            "compare_trajectories",
            {
                "golden_text": golden_text,
                "synthetic_text": synthetic_text,
                "acceleration_threshold": acceleration_threshold,
            },
        )

    async def close(self) -> None:
        """Close the MCP session and clean up resources."""
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
        self._session = None
        self._initialized = False


class PromptPrixClient:
    """Client for prompt-prix integration (fan-out evaluation).

    Connects to prompt-prix-mcp via stdio MCP protocol to perform:
    - Model completion (complete)
    - LLM-as-judge evaluation (judge)
    - Prompt variant generation (generate_variants)
    - Drift analysis (calculate_drift)
    - Trajectory analysis (analyze_trajectory)
    """

    def __init__(self, endpoint: str | None = None):
        """
        Initialize the prompt-prix client.

        Args:
            endpoint: Optional MCP server endpoint (command). Defaults to
                      'prompt-prix-mcp' if not provided.
        """
        self.endpoint = endpoint or "prompt-prix-mcp"
        self._initialized = False
        self._session: ClientSession | None = None
        self._read: Any = None
        self._write: Any = None

    async def initialize(self) -> bool:
        """Initialize connection to prompt-prix via stdio."""
        if self._initialized:
            return True

        try:
            # Create stdio client parameters
            params = StdioServerParameters(
                command="prompt-prix-mcp",
                args=[],
                env=None,
            )

            # Create stdio transport
            transport = stdio_client(params)
            self._read, self._write = await transport

            # Create session
            self._session = ClientSession(
                read_from=self._read,
                write_to=self._write,
            )

            # Initialize session
            await self._session.__aenter__()

            self._initialized = True
            return True

        except Exception as e:
            raise RuntimeError(f"Failed to initialize prompt-prix-mcp: {e}")

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call an MCP tool and return parsed JSON result."""
        if not self._session:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        try:
            result = await self._session.call_tool(name, arguments)

            # Extract text content from result
            if result.content:
                for content in result.content:
                    if content.type == "text":
                        return json.loads(content.text)

            return {"error": "No text content in response"}

        except Exception as e:
            return {"error": str(e)}

    async def list_models(self) -> list[str]:
        """List available models across all configured servers.

        Returns:
            List of model IDs
        """
        result = await self._call_tool("list_models", {})
        return result.get("models", [])

    async def complete(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout_seconds: int = 300,
        tools: list[dict] | None = None,
        seed: int | None = None,
        response_format: dict | None = None,
    ) -> str:
        """Get completion from a model.

        Args:
            model_id: Model to use
            messages: OpenAI chat format messages
            temperature: Generation temperature
            max_tokens: Response length limit
            timeout_seconds: Request timeout
            tools: Tool definitions for tool-use
            seed: Reproducibility seed
            response_format: Structured output schema

        Returns:
            The complete response text
        """
        result = await self._call_tool(
            "complete",
            {
                "model_id": model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout_seconds": timeout_seconds,
                "tools": tools,
                "seed": seed,
                "response_format": response_format,
            },
        )

        if "error" in result:
            raise RuntimeError(result["error"])

        return result

    async def judge(
        self,
        response: str,
        criteria: str,
        judge_model: str,
        temperature: float = 0.1,
        max_tokens: int = 256,
        timeout_seconds: int = 60,
    ) -> dict[str, Any]:
        """LLM-as-judge evaluation.

        Args:
            response: Model response to evaluate
            criteria: Natural language pass/fail criteria
            judge_model: Model to use as judge
            temperature: Judge temperature (low for consistency)
            max_tokens: Judge response length limit
            timeout_seconds: Request timeout

        Returns:
            Dictionary with pass, reason, score, and raw_response
        """
        return await self._call_tool(
            "judge",
            {
                "response": response,
                "criteria": criteria,
                "judge_model": judge_model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout_seconds": timeout_seconds,
            },
        )

    async def calculate_drift(self, text_a: str, text_b: str) -> float:
        """Calculate cosine distance between two texts via embedding.

        Args:
            text_a: First text
            text_b: Second text

        Returns:
            Cosine distance (0.0-2.0)
        """
        result = await self._call_tool(
            "calculate_drift",
            {"text_a": text_a, "text_b": text_b},
        )

        if "error" in result:
            raise RuntimeError(result["error"])

        return result.get("drift", 0.0)

    async def analyze_variants(
        self, variants: dict[str, str], baseline_label: str = "imperative", constraint_name: str = "unnamed"
    ) -> dict[str, Any]:
        """Analyze embedding distances between prompt variants.

        Args:
            variants: Label -> prompt text mapping
            baseline_label: Which variant is the baseline
            constraint_name: Label for the constraint set

        Returns:
            Dictionary with drift analysis between variants
        """
        return await self._call_tool(
            "analyze_variants",
            {
                "variants": variants,
                "baseline_label": baseline_label,
                "constraint_name": constraint_name,
            },
        )

    async def generate_variants(
        self,
        baseline: str,
        model_id: str,
        dimensions: list[str] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
        timeout_seconds: int = 60,
    ) -> dict[str, Any]:
        """Generate grammatical variants of a prompt constraint.

        Args:
            baseline: Imperative constraint to rephrase
            model_id: Model for generation
            dimensions: Grammatical dimensions (mood, voice, person, frame)
            temperature: Generation temperature
            max_tokens: Response length limit
            timeout_seconds: Request timeout

        Returns:
            Dictionary with baseline and variants mapping
        """
        return await self._call_tool(
            "generate_variants",
            {
                "baseline": baseline,
                "model_id": model_id,
                "dimensions": dimensions or ["mood", "voice", "person", "frame"],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout_seconds": timeout_seconds,
            },
        )

    async def analyze_trajectory(
        self, text: str, acceleration_threshold: float = 0.3, include_sentences: bool = False
    ) -> dict[str, Any]:
        """Analyze semantic velocity and acceleration profile of a text passage.

        Args:
            text: Text passage (needs 2+ sentences)
            acceleration_threshold: Threshold for flagging spikes
            include_sentences: Include sentence breakdown in output

        Returns:
            Dictionary with trajectory metrics
        """
        return await self._call_tool(
            "analyze_trajectory",
            {
                "text": text,
                "acceleration_threshold": acceleration_threshold,
                "include_sentences": include_sentences,
            },
        )

    async def compare_trajectories(
        self, golden_text: str, synthetic_text: str, acceleration_threshold: float = 0.3
    ) -> dict[str, Any]:
        """Compare trajectory profile of synthetic text against golden reference.

        Args:
            golden_text: Reference passage (target structure)
            synthetic_text: Model-generated passage to evaluate
            acceleration_threshold: Threshold for acceleration spikes

        Returns:
            Dictionary with fitness_score and comparison metrics
        """
        return await self._call_tool(
            "compare_trajectories",
            {
                "golden_text": golden_text,
                "synthetic_text": synthetic_text,
                "acceleration_threshold": acceleration_threshold,
            },
        )

    async def close(self) -> None:
        """Close the MCP session."""
        if self._session:
            await self._session.__aexit__(None, None, None)
            self._initialized = False
            self._session = None


async def create_semantic_kinematics_client(
    endpoint: str | None = None,
) -> SemanticKinematicsClient:
    """Create and initialize a semantic-kinematics client."""
    client = SemanticKinematicsClient(endpoint)
    await client.initialize()
    return client


async def create_prompt_prix_client(
    endpoint: str | None = None,
) -> PromptPrixClient:
    """Create and initialize a prompt-prix client."""
    client = PromptPrixClient(endpoint)
    await client.initialize()
    return client
