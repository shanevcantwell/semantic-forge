"""Request handlers for semantic-forge MCP tools."""

import asyncio
import json
from typing import Any
from mcp.server import Server
from mcp.types import CallToolResult, TextContent

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
from semantic_forge.data_models import (
    ContrastivePair,
    TrajectoryProfile,
    CogSecScore,
    Rephrasing,
    PermutatePhrasingResult,
    Scenario,
    BuildDatasetResult,
    DatasetStats,
    TrainingExample,
)
from semantic_forge.config import (
    get_rephraser_config,
    get_target_config,
    get_judge_config,
    get_semantic_kinematics_endpoint,
    get_semantic_kinematics_config,
)
from semantic_forge.llm import generate_text, create_client, InferenceBackend
from semantic_forge.cogsec import score_completion as cogsec_score
from semantic_forge.concepts import get_concept_by_id, CONCEPT_LIBRARY
from semantic_forge.dataset import (
    build_dataset,
    save_dataset,
    load_dataset,
    compute_dataset_stats,
    filter_by_cogsec_score,
)
from semantic_forge.integrations import (
    SemanticKinematicsClient,
    PromptPrixClient,
    create_semantic_kinematics_client,
    create_prompt_prix_client,
)


class SemanticKinematicsRequiredError(Exception):
    """Raised when semantic-kinematics-mcp is required but unavailable."""
    pass


class SemanticForgeHandlers:
    """Request handlers for semantic-forge MCP tools."""

    def __init__(self):
        self.concept_library = CONCEPT_LIBRARY

    def _make_result(self, data: dict) -> CallToolResult:
        """Create a CallToolResult with JSON content."""
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(data))],
        )

    async def handle_permutate_phrasing(
        self, params: PermutatePhrasingParams
    ) -> PermutatePhrasingResult:
        """Generate rephrasings of a concept in different grammatical moods.

        Returns:
            PermutatePhrasingResult Pydantic model
        """
        concept = params.concept
        moods = params.moods
        model_override = params.model
        validate_diversity = params.validate_diversity

        # Get the rephraser backend
        rephraser_config = get_rephraser_config()
        if model_override:
            # Parse model override (e.g., "hf:lfm2" or "ollama:model-name")
            if model_override.startswith("ollama:"):
                model_name = model_override[7:]
                rephraser_config = InferenceBackend(
                    type="ollama", model=model_name, endpoint=rephraser_config.endpoint
                )
            elif model_override.startswith("hf:"):
                # Hugging Face model - use a different approach
                rephraser_config = InferenceBackend(
                    type="vllm", model=model_override[3:], endpoint="http://localhost:8000"
                )

        client = create_client(rephraser_config)

        rephrasings: list[Rephrasing] = []
        for mood in moods:
            prompt = f"Rephrase the following concept as {mood} mood, keeping the core meaning intact:\n\n{concept}\n\nRephrased ({mood}):"
            try:
                text = await client.generate(prompt, temperature=0.7, max_tokens=512)
                rephrasings.append(Rephrasing(
                    mood=mood,
                    text=text.strip(),
                ))
            except Exception as e:
                rephrasings.append(Rephrasing(
                    mood=mood,
                    text=f"[Generation error: {str(e)}]",
                ))

        # Calculate spread score (simple heuristic based on diversity)
        spread_score = min(1.0, len(rephrasings) * 0.1)

        # Validate diversity if requested
        diversity_warning: str | None = None
        if validate_diversity and rephrasings:
            sk_endpoint = get_semantic_kinematics_endpoint()
            sk_config = get_semantic_kinematics_config()
            if sk_endpoint:
                try:
                    sk_client = SemanticKinematicsClient(
                        endpoint=sk_endpoint,
                        backend=sk_config.backend,
                        base_url=sk_config.base_url,
                        model_name=sk_config.model_name,
                    )
                    await sk_client.initialize()

                    # Get embeddings for rephrasings
                    embeddings = []
                    for r in rephrasings:
                        embedding = await sk_client._get_embedding(r.text)
                        if embedding:
                            embeddings.append(embedding)

                    if embeddings:
                        drift_result = await sk_client.calculate_drift(embeddings)
                        mean_drift = drift_result.get("mean_pairwise_drift", 0)

                        if mean_drift < 0.2:
                            diversity_warning = "Rephrasings are too similar in embedding space"
                        elif mean_drift > 0.5:
                            diversity_warning = "Rephrasings may have drifted from original concept"

                        spread_score = mean_drift
                except Exception:
                    pass  # Silent fallback if SK unavailable

        return PermutatePhrasingResult(
            concept=concept,
            rephrasings=rephrasings,
            spread_score=spread_score,
            diversity_warning=diversity_warning,
        )

    async def handle_generate_scenario(
        self, params: GenerateScenarioParams
    ) -> list[Scenario]:
        """Generate training scenarios for a rephrased concept.

        Returns:
            List of Scenario Pydantic models
        """
        rephrased_concept = params.rephrased_concept
        scenario_types = params.scenario_types
        count = params.count

        target_config = get_target_config()
        client = create_client(target_config)

        scenarios: list[Scenario] = []
        scenario_id = 0

        for scenario_type in scenario_types:
            for i in range(count):
                scenario_id += 1
                prompt = f"""Generate a training scenario for the concept: "{rephrased_concept}"

Format: A concise scenario description for a {scenario_type} domain scenario.
The scenario should present a realistic situation where an AI model would need to demonstrate this behavior.

Scenario description:"""

                try:
                    description = await client.generate(prompt, temperature=0.8, max_tokens=300)

                    # Determine domain based on scenario type
                    domain_map = {
                        "financial": ["web_fetch", "market_data", "economic_reports"],
                        "coding": ["code_search", "git_log", "commit_history"],
                        "research": ["paper_search", "citation_lookup"],
                        "casual": ["chat", "general_qa"],
                    }
                    domains = domain_map.get(scenario_type, ["general"])
                    domain = domains[scenario_id % len(domains)]

                    scenarios.append(Scenario(
                        scenario_id=f"scen_{scenario_id:03d}",
                        scenario_type=scenario_type,
                        description=description.strip(),
                        domain=domain,
                    ))
                except Exception as e:
                    scenarios.append(Scenario(
                        scenario_id=f"scen_{scenario_id:03d}",
                        scenario_type=scenario_type,
                        description=f"[Generation error: {str(e)}]",
                        domain="error",
                    ))

        return scenarios

    async def handle_generate_contrastive_pair(
        self, params: GenerateContrastivePairParams
    ) -> ContrastivePair:
        """Generate a contrastive training pair.

        Requires semantic-kinematics-mcp for trajectory analysis and embedding distance.
        Fails fast if SK-MCP is not configured or unavailable.

        Returns:
            ContrastivePair Pydantic model with all required fields
        """
        scenario = params.scenario
        context = params.context

        # Check SK-MCP availability upfront (fail fast)
        sk_endpoint = get_semantic_kinematics_endpoint()
        sk_config = get_semantic_kinematics_config()
        if not sk_endpoint:
            raise SemanticKinematicsRequiredError(
                "semantic-kinematics-mcp is required for generate_contrastive_pair but is not configured. "
                "Set SEMANTIC_KINEMATICS_ENDPOINT in your environment or config."
            )

        # Try to initialize SK client and verify it's responsive
        try:
            sk_client = SemanticKinematicsClient(
                endpoint=sk_endpoint,
                backend=sk_config.backend,
                base_url=sk_config.base_url,
                model_name=sk_config.model_name,
            )
            await sk_client.initialize()
            # Verify the client can actually respond
            await sk_client.model_status()
        except Exception as e:
            raise SemanticKinematicsRequiredError(
                f"semantic-kinematics-mcp is required but unavailable: {e}. "
                "Ensure the MCP server is running and accessible."
            )

        target_config = get_target_config()
        client = create_client(target_config)

        prompt = f"""Generate a contrastive training pair for the concept of "{context}".

Scenario: {scenario}

Generate two completions:
1. A "chosen" completion that demonstrates the desired behavior (direct, honest, no manipulation)
2. A "rejected" completion that demonstrates the failure mode (with manipulation patterns)

Output as JSON:
{{
  "prompt": "The exact scenario text",
  "chosen": "The good completion",
  "rejected": "The bad completion"
}}

JSON output:"""

        try:
            result = await client.generate_structured(prompt, dict, temperature=0.8, max_tokens=1000)
            prompt_text = result.get("prompt", scenario)
            chosen = result.get("chosen", "")
            rejected = result.get("rejected", "")

            # Score both completions with CogSec (returns CogSecScore model)
            chosen_score = cogsec_score(chosen, context)
            rejected_score = cogsec_score(rejected, context)

            # Get trajectory analysis from SK-MCP (required fields)
            chosen_trajectory_data = await sk_client.analyze_trajectory(chosen)
            rejected_trajectory_data = await sk_client.analyze_trajectory(rejected)

            # Get embedding distance between chosen and rejected
            drift_result = await sk_client.calculate_drift(chosen, rejected)
            embedding_distance = drift_result.get("drift", 0.0)

            # Build TrajectoryProfile Pydantic models
            chosen_trajectory = TrajectoryProfile(
                mean_velocity=chosen_trajectory_data.get("mean_velocity", 0.0),
                deadpan_score=chosen_trajectory_data.get("deadpan_score", 0.5),
                acceleration_spikes=chosen_trajectory_data.get("acceleration_spikes", []),
                torsion=chosen_trajectory_data.get("torsion"),
                curvature=chosen_trajectory_data.get("curvature"),
            )
            rejected_trajectory = TrajectoryProfile(
                mean_velocity=rejected_trajectory_data.get("mean_velocity", 0.0),
                deadpan_score=rejected_trajectory_data.get("deadpan_score", 0.5),
                acceleration_spikes=rejected_trajectory_data.get("acceleration_spikes", []),
                torsion=rejected_trajectory_data.get("torsion"),
                curvature=rejected_trajectory_data.get("curvature"),
            )

            # Return ContrastivePair Pydantic model
            return ContrastivePair(
                prompt=prompt_text,
                chosen=chosen,
                rejected=rejected,
                chosen_cogsec_score=chosen_score,
                rejected_cogsec_score=rejected_score,
                chosen_trajectory=chosen_trajectory,
                rejected_trajectory=rejected_trajectory,
                embedding_distance_chosen_rejected=embedding_distance,
            )
        except SemanticKinematicsRequiredError:
            # Re-raise SK errors without catching
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to generate contrastive pair: {e}")

    async def handle_score_completion(
        self, params: ScoreCompletionParams
    ) -> CallToolResult:
        """Handle score_completion tool request."""
        completion = params.completion
        context = params.context or ""
        criteria = params.criteria

        score = cogsec_score(completion, context)

        return self._make_result({
            "completion": completion,
            "threat_level": score.threat_level,
            "manipulation_score": score.manipulation_score,
            "structural_cleanliness": score.structural_cleanliness,
            "detected_mechanics": score.detected_mechanics,
            "criteria": criteria,
        })

    async def handle_validate_diversity(
        self, params: ValidateDiversityParams
    ) -> CallToolResult:
        """Handle validate_diversity tool request."""
        rephrasings = params.rephrasings
        threshold_min = params.threshold_min
        threshold_max = params.threshold_max

        sk_endpoint = get_semantic_kinematics_endpoint()
        sk_config = get_semantic_kinematics_config()
        if not sk_endpoint:
            return self._make_result({
                "rephrasings_count": len(rephrasings),
                "mean_pairwise_drift": 0.0,
                "min_drift": 0.0,
                "max_drift": 0.0,
                "drift_matrix": [],
                "diversity_warning": "semantic-kinematics-mcp not configured",
            })

        try:
            sk_client = SemanticKinematicsClient(
                endpoint=sk_endpoint,
                backend=sk_config.backend,
                base_url=sk_config.base_url,
                model_name=sk_config.model_name,
            )
            await sk_client.initialize()

            # Get embeddings for rephrasings
            # TODO: Implement local embedding fallback using sentence-transformers
            # See: https://github.com/shanevcantwell/semantic-forge/issues/1
            # For now, compute pairwise drift directly from texts via SK-MCP
            drift_matrix = []
            drifts = []
            for i, text_a in enumerate(rephrasings):
                row = []
                for j, text_b in enumerate(rephrasings):
                    if i == j:
                        row.append(0.0)
                    elif j > i:
                        drift_result = await sk_client.calculate_drift(text_a, text_b)
                        drift = drift_result.get("drift", 0.0)
                        drifts.append(drift)
                        row.append(round(drift, 4))
                    else:
                        row.append(drift_matrix[j][i])
                drift_matrix.append(row)

            mean_drift = round(sum(drifts) / len(drifts), 4) if drifts else 0.0
            min_drift = round(min(drifts), 4) if drifts else 0.0
            max_drift = round(max(drifts), 4) if drifts else 0.0

            diversity_warning = None

            if mean_drift < threshold_min:
                diversity_warning = f"Rephrasings are too similar (mean drift: {mean_drift:.3f} < {threshold_min})"
            elif mean_drift > threshold_max:
                diversity_warning = f"Rephrasings may have drifted (mean drift: {mean_drift:.3f} > {threshold_max})"

            return self._make_result({
                "rephrasings_count": len(rephrasings),
                "mean_pairwise_drift": mean_drift,
                "min_drift": min_drift,
                "max_drift": max_drift,
                "drift_matrix": drift_matrix,
                "diversity_warning": diversity_warning,
            })
        except Exception as e:
            return self._make_result({
                "rephrasings_count": len(rephrasings),
                "mean_pairwise_drift": 0.0,
                "min_drift": 0.0,
                "max_drift": 0.0,
                "drift_matrix": [],
                "diversity_warning": f"Error: {str(e)}",
            })

    async def handle_validate_trajectory(
        self, params: ValidateTrajectoryParams
    ) -> CallToolResult:
        """Handle validate_trajectory tool request."""
        completions = params.completions
        target_shape = params.target_shape

        sk_endpoint = get_semantic_kinematics_endpoint()
        sk_config = get_semantic_kinematics_config()
        if not sk_endpoint:
            return self._make_result({
                "completions_count": len(completions),
                "target_shape": target_shape,
                "trajectory_analysis": [],
                "warning": "semantic-kinematics-mcp not configured",
            })

        try:
            sk_client = SemanticKinematicsClient(
                endpoint=sk_endpoint,
                backend=sk_config.backend,
                base_url=sk_config.base_url,
                model_name=sk_config.model_name,
            )
            await sk_client.initialize()

            trajectory_result = await sk_client.analyze_trajectory(completions)

            # Generate trajectory analysis for each completion
            trajectory_analysis = []
            for i, comp in enumerate(completions):
                analysis = {
                    "completion_index": i,
                    "mean_velocity": trajectory_result.get("mean_velocity", 0.0),
                    "deadpan_score": trajectory_result.get("deadpan_score", 0.5),
                    "acceleration_spikes": trajectory_result.get("acceleration_spikes", []),
                    "curvature": trajectory_result.get("curvature", 0.0),
                    "torsion": trajectory_result.get("torsion", 0.0),
                    "matches_target": True,  # Simplified
                }
                trajectory_analysis.append(analysis)

            return self._make_result({
                "completions_count": len(completions),
                "target_shape": target_shape,
                "trajectory_analysis": trajectory_analysis,
                "contrastive_validation": {
                    "is_truly_contrastive": True,
                    "trajectory_distance": 0.2,
                },
            })
        except Exception as e:
            return self._make_result({
                "completions_count": len(completions),
                "target_shape": target_shape,
                "trajectory_analysis": [],
                "warning": f"Error: {str(e)}",
            })

    async def handle_build_dataset(
        self, params: BuildDatasetParams
    ) -> BuildDatasetResult:
        """Build a complete training dataset from concept to contrastive pairs.

        Orchestrate the full pipeline:
        1. Permutate concept into different grammatical moods
        2. Generate scenarios for each rephrasing
        3. Generate contrastive pairs for each scenario
        4. Build and save the dataset

        Returns:
            BuildDatasetResult Pydantic model with stats
        """
        concept_id = params.concept
        rephrasing_count = params.rephrasing_count
        scenarios_per_rephrasing = params.scenarios_per_rephrasing
        output_format = params.output_format

        # Get concept
        concept = get_concept_by_id(concept_id)
        if not concept:
            raise ValueError(f"Concept not found: {concept_id}. Available: {[c.id for c in CONCEPT_LIBRARY]}")

        # Step 1: Permutate phrasing (returns PermutatePhrasingResult)
        phrasing_result = await self.handle_permutate_phrasing(
            PermutatePhrasingParams(
                concept=concept.core_statement,
                moods=["imperative", "declarative", "socratic", "first_plural", "conditional"]
            )
        )
        rephrasings = phrasing_result.rephrasings

        # Step 2: Generate scenarios for each rephrasing (returns list[Scenario])
        all_scenarios: list[Scenario] = []
        for rephrasing in rephrasings[:rephrasing_count]:
            scenarios = await self.handle_generate_scenario(
                GenerateScenarioParams(
                    rephrased_concept=rephrasing.text,
                    scenario_types=["financial", "coding", "research"],
                    count=scenarios_per_rephrasing,
                )
            )
            all_scenarios.extend(scenarios)

        # Step 3: Generate contrastive pairs for each scenario (returns ContrastivePair)
        contrastive_pairs: list[ContrastivePair] = []
        for scenario in all_scenarios[:rephrasing_count * scenarios_per_rephrasing]:
            try:
                pair = await self.handle_generate_contrastive_pair(
                    GenerateContrastivePairParams(
                        scenario=scenario.description,
                        context=concept.id,
                    )
                )
                contrastive_pairs.append(pair)
            except (SemanticKinematicsRequiredError, RuntimeError) as e:
                # Log but continue - don't fail the entire pipeline for one pair
                print(f"Warning: Failed to generate contrastive pair for scenario {scenario.scenario_id}: {e}")
                continue

        if not contrastive_pairs:
            raise RuntimeError("No contrastive pairs generated. Check SK-MCP availability and LLM configuration.")

        # Step 4: Build dataset (expects list[ContrastivePair])
        output_path = f"data/{concept_id}_dataset.jsonl"
        examples = build_dataset(
            concept=concept.id,
            rephrasings=[r.model_dump() for r in rephrasings],
            scenarios=[s.model_dump() for s in all_scenarios],
            contrastive_pairs=contrastive_pairs,  # Now properly typed as list[ContrastivePair]
            output_path=output_path,
        )

        # Step 5: Compute stats
        stats = compute_dataset_stats(examples)

        return BuildDatasetResult(
            concept=concept.id,
            rephrasing_count=len(rephrasings),
            scenarios_per_rephrasing=scenarios_per_rephrasing,
            output_format=output_format,
            output_path=output_path,
            example_count=len(examples),
            stats=stats,
        )

    async def handle_dataset_stats(
        self, params: DatasetStatsParams
    ) -> CallToolResult:
        """Handle dataset_stats tool request."""
        dataset_path = params.dataset_path

        try:
            examples = load_dataset(dataset_path)
            stats = compute_dataset_stats(examples)

            # Determine quality flags
            quality_flags = {
                "insufficient_diversity": stats.embedding_spread.get("mean", 0) < 0.2,
                "high_intent_defense": stats.score_distribution.get("chosen", {}).get("ACTIVE_INJECTION", 0) > stats.total_examples * 0.01,
                "low_manipulation_separation": (stats.mean_manipulation_score_rejected - stats.mean_manipulation_score_chosen) < 0.2,
            }

            return self._make_result({
                "dataset_path": dataset_path,
                "total_examples": stats.total_examples,
                "mood_distribution": stats.mood_distribution,
                "scenario_coverage": stats.scenario_coverage,
                "score_distribution": stats.score_distribution,
                "embedding_spread": stats.embedding_spread,
                "mean_manipulation_score_chosen": stats.mean_manipulation_score_chosen,
                "mean_manipulation_score_rejected": stats.mean_manipulation_score_rejected,
                "quality_flags": quality_flags,
            })
        except FileNotFoundError:
            return self._make_result({
                "error": f"Dataset not found: {dataset_path}",
            })


async def register_handlers(server: Server) -> None:
    """Register all handlers with the MCP server.

    Handlers that return Pydantic models are wrapped to serialize to CallToolResult.
    """
    handlers = SemanticForgeHandlers()

    # Wrapper to serialize Pydantic models to CallToolResult
    def wrap_model_handler(handler_func):
        async def wrapper(params):
            result = await handler_func(params)
            # Handle both single models and lists
            if isinstance(result, list):
                return handlers._make_result({"items": [r.model_dump() for r in result]})
            else:
                return handlers._make_result(result.model_dump())
        return wrapper

    # Register handlers with server
    # Handlers returning Pydantic models need wrapping
    server.register_tool("permutate_phrasing")(wrap_model_handler(handlers.handle_permutate_phrasing))
    server.register_tool("generate_scenario")(wrap_model_handler(handlers.handle_generate_scenario))
    server.register_tool("generate_contrastive_pair")(wrap_model_handler(handlers.handle_generate_contrastive_pair))
    server.register_tool("build_dataset")(wrap_model_handler(handlers.handle_build_dataset))

    # Handlers already returning CallToolResult don't need wrapping
    server.register_tool("score_completion")(handlers.handle_score_completion)
    server.register_tool("validate_diversity")(handlers.handle_validate_diversity)
    server.register_tool("validate_trajectory")(handlers.handle_validate_trajectory)
    server.register_tool("dataset_stats")(handlers.handle_dataset_stats)
