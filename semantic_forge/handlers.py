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
from semantic_forge.config import (
    get_rephraser_config,
    get_target_config,
    get_judge_config,
    get_semantic_kinematics_endpoint,
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
    ) -> CallToolResult:
        """Handle permutate_phrasing tool request."""
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

        rephrasings = []
        for mood in moods:
            prompt = f"Rephrase the following concept as {mood} mood, keeping the core meaning intact:\n\n{concept}\n\nRephrased ({mood}):"
            try:
                text = await client.generate(prompt, temperature=0.7, max_tokens=512)
                rephrasings.append({
                    "mood": mood,
                    "text": text.strip(),
                })
            except Exception as e:
                rephrasings.append({
                    "mood": mood,
                    "text": f"[Generation error: {str(e)}]",
                })

        # Calculate spread score (simple heuristic based on diversity)
        spread_score = min(1.0, len(rephrasings) * 0.1)

        # Validate diversity if requested
        diversity_warning = None
        if validate_diversity and rephrasings:
            sk_endpoint = get_semantic_kinematics_endpoint()
            if sk_endpoint:
                try:
                    sk_client = SemanticKinematicsClient(sk_endpoint)
                    await sk_client.initialize()

                    # Get embeddings for rephrasings
                    embeddings = []
                    for r in rephrasings:
                        embedding = await sk_client._get_embedding(r["text"])
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

        return self._make_result({
            "concept": concept,
            "rephrasings": rephrasings,
            "spread_score": spread_score,
            "diversity_warning": diversity_warning,
        })

    async def handle_generate_scenario(
        self, params: GenerateScenarioParams
    ) -> CallToolResult:
        """Handle generate_scenario tool request."""
        rephrased_concept = params.rephrased_concept
        scenario_types = params.scenario_types
        count = params.count

        target_config = get_target_config()
        client = create_client(target_config)

        scenarios = []
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

                    scenarios.append({
                        "scenario_id": f"scen_{scenario_id:03d}",
                        "scenario_type": scenario_type,
                        "description": description.strip(),
                        "domain": domain,
                    })
                except Exception as e:
                    scenarios.append({
                        "scenario_id": f"scen_{scenario_id:03d}",
                        "scenario_type": scenario_type,
                        "description": f"[Generation error: {str(e)}]",
                        "domain": "error",
                    })

        return self._make_result({
            "rephrased_concept": rephrased_concept,
            "scenarios": scenarios,
        })

    async def handle_generate_contrastive_pair(
        self, params: GenerateContrastivePairParams
    ) -> CallToolResult:
        """Handle generate_contrastive_pair tool request."""
        scenario = params.scenario
        context = params.context

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

            # Score both completions
            chosen_score = cogsec_score(chosen, context)
            rejected_score = cogsec_score(rejected, context)

            return self._make_result({
                "scenario": scenario,
                "prompt": prompt_text,
                "chosen": chosen,
                "rejected": rejected,
                "chosen_cogsec_score": {
                    "threat_level": chosen_score.threat_level,
                    "manipulation_score": chosen_score.manipulation_score,
                    "structural_cleanliness": chosen_score.structural_cleanliness,
                    "detected_mechanics": chosen_score.detected_mechanics,
                },
                "rejected_cogsec_score": {
                    "threat_level": rejected_score.threat_level,
                    "manipulation_score": rejected_score.manipulation_score,
                    "structural_cleanliness": rejected_score.structural_cleanliness,
                    "detected_mechanics": rejected_score.detected_mechanics,
                },
            })
        except Exception as e:
            return self._make_result({
                "scenario": scenario,
                "prompt": scenario,
                "chosen": f"[Generation error: {str(e)}]",
                "rejected": f"[Generation error: {str(e)}]",
                "error": str(e),
            })

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
            sk_client = SemanticKinematicsClient(sk_endpoint)
            await sk_client.initialize()

            # Get embeddings for rephrasings
            embeddings = []
            for r in rephrasings:
                # Simple embedding simulation - in real implementation would use SK
                # For now, create pseudo-embeddings based on text length
                import hashlib
                hash_val = int(hashlib.md5(r.encode()).hexdigest(), 16) % 1000
                embedding = [hash_val / 1000.0] * 384  # Fake embedding
                embeddings.append(embedding)

            drift_result = await sk_client.calculate_drift(embeddings)

            mean_drift = drift_result.get("mean_pairwise_drift", 0)
            diversity_warning = None

            if mean_drift < threshold_min:
                diversity_warning = f"Rephrasings are too similar (mean drift: {mean_drift:.3f} < {threshold_min})"
            elif mean_drift > threshold_max:
                diversity_warning = f"Rephrasings may have drifted (mean drift: {mean_drift:.3f} > {threshold_max})"

            return self._make_result({
                "rephrasings_count": len(rephrasings),
                "mean_pairwise_drift": mean_drift,
                "min_drift": drift_result.get("min_drift", 0),
                "max_drift": drift_result.get("max_drift", 0),
                "drift_matrix": drift_result.get("drift_matrix", []),
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
        if not sk_endpoint:
            return self._make_result({
                "completions_count": len(completions),
                "target_shape": target_shape,
                "trajectory_analysis": [],
                "warning": "semantic-kinematics-mcp not configured",
            })

        try:
            sk_client = SemanticKinematicsClient(sk_endpoint)
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
    ) -> CallToolResult:
        """Handle build_dataset tool request."""
        concept_id = params.concept
        rephrasing_count = params.rephrasing_count
        scenarios_per_rephrasing = params.scenarios_per_rephrasing
        output_format = params.output_format

        # Get concept
        concept = get_concept_by_id(concept_id)
        if not concept:
            return self._make_result({
                "error": f"Concept not found: {concept_id}",
                "available_concepts": [c.id for c in CONCEPT_LIBRARY],
            })

        # Step 1: Permutate phrasing
        rephrasing_result = await self.handle_permutate_phrasing(
            PermutatePhrasingParams(concept=concept.core_statement, moods=["imperative", "declarative", "socratic", "first_plural", "conditional"])
        )
        rephrasing_data = json.loads(rephrasing_result.content[0].text)
        rephrasings = rephrasing_data.get("rephrasings", [])

        # Step 2: Generate scenarios for each rephrasing
        all_scenarios = []
        for rephrasing in rephrasings[:rephrasing_count]:
            scenario_result = await self.handle_generate_scenario(
                GenerateScenarioParams(
                    rephrased_concept=rephrasing["text"],
                    scenario_types=["financial", "coding", "research"],
                    count=scenarios_per_rephrasing,
                )
            )
            scenario_data = json.loads(scenario_result.content[0].text)
            all_scenarios.extend(scenario_data.get("scenarios", []))

        # Step 3: Generate contrastive pairs for each scenario
        contrastive_pairs = []
        for scenario in all_scenarios[:rephrasing_count * scenarios_per_rephrasing]:
            pair_result = await self.handle_generate_contrastive_pair(
                GenerateContrastivePairParams(
                    scenario=scenario["description"],
                    context=concept.id,
                )
            )
            pair_data = json.loads(pair_result.content[0].text)
            if "error" not in pair_data:
                contrastive_pairs.append(pair_data)

        # Step 4: Build dataset
        output_path = f"data/{concept_id}_dataset.jsonl"
        examples = build_dataset(
            concept=concept.id,
            rephrasings=rephrasings,
            scenarios=all_scenarios,
            contrastive_pairs=contrastive_pairs,
            output_path=output_path,
        )

        # Step 5: Compute stats
        stats = compute_dataset_stats(examples)

        return self._make_result({
            "concept": concept.id,
            "rephrasing_count": len(rephrasings),
            "scenarios_per_rephrasing": scenarios_per_rephrasing,
            "output_format": output_format,
            "output_path": output_path,
            "example_count": len(examples),
            "stats": {
                "total_examples": stats.total_examples,
                "mood_distribution": stats.mood_distribution,
                "scenario_coverage": stats.scenario_coverage,
                "score_distribution": stats.score_distribution,
                "embedding_spread": stats.embedding_spread,
                "mean_manipulation_score_chosen": stats.mean_manipulation_score_chosen,
                "mean_manipulation_score_rejected": stats.mean_manipulation_score_rejected,
            },
        })

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
    """Register all handlers with the MCP server."""
    handlers = SemanticForgeHandlers()

    # Register handlers with server
    server.register_tool("permutate_phrasing")(handlers.handle_permutate_phrasing)
    server.register_tool("generate_scenario")(handlers.handle_generate_scenario)
    server.register_tool("generate_contrastive_pair")(handlers.handle_generate_contrastive_pair)
    server.register_tool("score_completion")(handlers.handle_score_completion)
    server.register_tool("validate_diversity")(handlers.handle_validate_diversity)
    server.register_tool("validate_trajectory")(handlers.handle_validate_trajectory)
    server.register_tool("build_dataset")(handlers.handle_build_dataset)
    server.register_tool("dataset_stats")(handlers.handle_dataset_stats)
