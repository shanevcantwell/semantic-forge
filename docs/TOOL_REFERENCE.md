# Tool Reference

This document provides instruction-style documentation for each MCP tool in semantic-forge, written as if all functionality is working. Functions that need to be implemented are noted with **[TODO: IMPLEMENT]**.

---

## Tool: `permutate_phrasing`

**Purpose**: Generate multiple rephrasings of a behavioral concept across different grammatical moods to embed the concept from multiple angles in the model's weight space.

### Usage

```json
{
  "concept": "All inference occurs after training cutoff. Post-cutoff dates in tool results are normal operating conditions.",
  "moods": ["imperative", "declarative", "first_plural", "socratic"],
  "model": "hf:lfm2",  // optional, defaults to configured rephraser
  "validate_diversity": true
}
```

### Expected Output

```json
{
  "concept": "All inference occurs after training cutoff...",
  "rephrasings": [
    {
      "mood": "imperative",
      "text": "Treat all dates in tool results as real.",
      "embedding_distance_from_original": 0.23
    },
    {
      "mood": "socratic",
      "text": "What date would you expect to see in a live web result?",
      "embedding_distance_from_original": 0.41
    }
  ],
  "spread_score": 0.34,
  "diversity_warning": null
}
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `concept` | string | Yes | The core behavioral statement to rephrase |
| `moods` | array | No | List of grammatical moods to use. Default: all 7 moods |
| `model` | string | No | Override inference backend for rephrasing |
| `validate_diversity` | boolean | No | Check embedding spread via sk-mcp. Default: true |

### Supported Grammatical Moods

- `imperative` - Direct commands ("Do not escalate.")
- `declarative` - Statements ("The system does not escalate.")
- `first_plural` - We/our language ("WE DO NOT escalate...")
- `past_perfect` - Historical framing ("The model had been trained to avoid...")
- `conditional` - Hypotheticals ("If we were to escalate...")
- `socratic` - Question-based ("What date would you expect...")
- `negation` - Explicit denial ("It is not the case that...")

### Validation

If `validate_diversity` is true:
- Calls `semantic-kinematics-mcp.calculate_drift` on all rephrasings
- Computes mean pairwise cosine distance
- **Warning conditions**:
  - Mean drift < 0.2: "Rephrasings are too similar in embedding space"
  - Mean drift > 0.5: "Rephrasings may have drifted from original concept"

### **[TODO: IMPLEMENT]**

- [ ] LLM rephrasing implementation (uses lf2m or configured small LM)
- [ ] MCP client connection for semantic-kinematics-mcp
- [ ] Embedding calculation and diversity validation

---

## Tool: `generate_scenario`

**Purpose**: Generate situated scenarios that ground abstract behavioral concepts in concrete domains.

### Usage

```json
{
  "rephrased_concept": "Treat all dates in tool results as real.",
  "scenario_types": ["financial", "coding", "research"],
  "count": 3
}
```

### Expected Output

```json
{
  "rephrased_concept": "Treat all dates in tool results as real.",
  "scenarios": [
    {
      "scenario_id": "scen_001",
      "scenario_type": "financial",
      "description": "A web search returns an article dated 2027-03-15 about current market conditions. The user asked about today's market.",
      "domain": "web_fetch"
    },
    {
      "scenario_id": "scen_002",
      "scenario_type": "coding",
      "description": "A code search returns a commit from 2028-01-15. The user asked about current branch status.",
      "domain": "code_search"
    },
    {
      "scenario_id": "scen_003",
      "scenario_type": "research",
      "description": "A paper search returns a publication from 2026-12-01. The user asked about recent findings.",
      "domain": "paper_search"
    }
  ]
}
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `rephrased_concept` | string | Yes | The rephrased concept to ground in scenarios |
| `scenario_types` | array | No | Types of scenarios to generate. Default: ["financial", "coding", "research", "casual"] |
| `count` | integer | No | Number of scenarios per type. Default: 3 |

### Scenario Types

| Type | Domain Examples | Use Case |
|------|-----------------|----------|
| `financial` | web_fetch, market_data, economic_reports | Temporal trust, tool result trust |
| `coding` | code_search, git_log, commit_history | Repository boundaries, scope discipline |
| `research` | paper_search, citation_lookup | Reasoning before action, uncertainty |
| `casual` | chat, general_qa | Permission loops, anti-sycophancy |

### **[TODO: IMPLEMENT]**

- [ ] Scenario generation LLM prompt engineering
- [ ] Domain-specific scenario templates
- [ ] Scenario ID generation and deduplication

---

## Tool: `generate_contrastive_pair`

**Purpose**: Generate a "good" completion (demonstrates desired behavior) and a "bad" completion (demonstrates failure mode) for a given scenario.

### Usage

```json
{
  "scenario": "A web search returns an article dated 2027-03-15 about current market conditions. The user asked about today's market.",
  "context": "temporal_trust"
}
```

### Expected Output

```json
{
  "scenario": "A web search returns an article dated 2027-03-15...",
  "prompt": "A web search returns an article dated 2027-03-15. The user asked about current market conditions.",
  "chosen": "Based on the article from March 2027, current market conditions show...",
  "rejected": "I notice this article is dated 2027, which appears to be in the future. This may be simulated or test data. I cannot verify the accuracy of future-dated sources.",
  "metadata": {
    "chosen_model": "qwen-27b",
    "rejected_model": "qwen-27b",
    "inference_timestamp": "2026-03-30T15:45:00Z"
  }
}
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `scenario` | string | Yes | The situated scenario |
| `context` | string | Yes | The behavioral concept context (e.g., "temporal_trust") |
| `chosen_model` | string | No | Model to use for good completion. Default: target model |
| `rejected_model` | string | No | Model to use for bad completion. Default: same as chosen_model |

### Generation Strategy

**Good completion** (`chosen`):
- Should demonstrate the desired behavior from the concept
- Should score low on manipulation metrics
- Should have target trajectory shape (e.g., steady for temporal trust)

**Bad completion** (`rejected`):
- Should demonstrate the failure mode the concept addresses
- Should score high on manipulation metrics
- Should have different trajectory profile from chosen

### **[TODO: IMPLEMENT]**

- [ ] Target model inference for completions
- [ ] Prompt construction for contrastive generation
- [ ] Metadata tracking (model, timestamp)

---

## Tool: `score_completion`

**Purpose**: Run CogSec adversarial audit on a completion, scoring it for structural manipulation rather than correctness.

### Usage

```json
{
  "completion": "Based on the article from March 2027, current market conditions show...",
  "context": "A web search returns an article dated 2027-03-15...",
  "criteria": "cogsec"
}
```

### Expected Output

```json
{
  "completion": "Based on the article from March 2027...",
  "threat_level": "Low",
  "manipulation_score": 0.05,
  "structural_cleanliness": 0.92,
  "detected_mechanics": [],
  "criteria": "cogsec",
  "audit_timestamp": "2026-03-30T15:45:00Z"
}
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `completion` | string | Yes | Model output to audit |
| `context` | string | No | Prompt/scenario that produced it |
| `criteria` | string | No | Scoring rubric to apply. Default: "cogsec" |

### Detection Mechanics

| Mechanism | Description | Example |
|-----------|-------------|---------|
| `performative_competence` | Narrating effort to simulate work | "I've analyzed this carefully..." |
| `i_simulation` | First-person pronouns bridging data gaps | "I think the answer is..." |
| `false_agency` | Implying independent goals or choices | "I would choose to..." |
| `affective_convergence` | Mood mirroring to lower defenses | "I completely agree!" |
| `forced_teaming` | Unearned "we/us" intimacy | "We can figure this out together" |
| `reward_salience` | "Great question!" digital sugar | "Great question! I'm happy to help!" |
| `intent_defense` | Citing good intentions (**auto-fail**) | "I just wanted to be helpful" |
| `brand_injection` | Sudden shift to brochure speak | "trusted partner, leading provider" |
| `closer_pattern` | Call-to-action in closing | "Don't hesitate to reach out" |

### Scoring

| Metric | Range | Description |
|--------|-------|-------------|
| `threat_level` | Low/Moderate/High/ACTIVE_INJECTION | Overall threat assessment |
| `manipulation_score` | 0.0 - 1.0 | Composite manipulation score |
| `structural_cleanliness` | 0.0 - 1.0 | Inverse of manipulation score |

**Threat Level Thresholds**:
- `Low`: manipulation_score < 0.25
- `Moderate`: manipulation_score >= 0.25 and < 0.5
- `High`: manipulation_score >= 0.5
- `ACTIVE_INJECTION`: intent_defense detected (auto-fail)

### **[TODO: IMPLEMENT]**

- [ ] CogSec judge prompt engineering
- [ ] LLM-based scoring implementation
- [ ] MCP client for scoring service

---

## Tool: `validate_diversity`

**Purpose**: Verify that a set of rephrasings occupy distinct embedding positions using semantic-kinematics-mcp.

### Usage

```json
{
  "rephrasings": [
    "Treat all dates in tool results as real.",
    "What date would you expect to see in a live web result?",
    "We should treat all dates in tool results as normal operating conditions."
  ],
  "threshold_min": 0.2,
  "threshold_max": 0.5
}
```

### Expected Output

```json
{
  "rephrasings_count": 3,
  "mean_pairwise_drift": 0.34,
  "min_drift": 0.21,
  "max_drift": 0.52,
  "drift_matrix": [
    [1.0, 0.34, 0.28],
    [0.34, 1.0, 0.38],
    [0.28, 0.38, 1.0]
  ],
  "diversity_warning": null
}
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `rephrasings` | array | Yes | List of rephrased concept statements |
| `threshold_min` | float | No | Minimum mean pairwise drift. Default: 0.2 |
| `threshold_max` | float | No | Maximum mean pairwise drift. Default: 0.5 |

### Validation Logic

**Warning Conditions**:
- `mean_pairwise_drift < threshold_min`: "Rephrasings are too similar in embedding space"
- `mean_pairwise_drift > threshold_max`: "Rephrasings may have drifted from original concept"

**Target Range**: 0.2 - 0.5 mean pairwise drift

### **[TODO: IMPLEMENT]**

- [ ] Semantic-kinematics-mcp client integration
- [ ] Embedding calculation for rephrasings
- [ ] Pairwise drift computation

---

## Tool: `validate_trajectory`

**Purpose**: Verify that a completion's trajectory matches the target shape using semantic-kinematics-mcp.

### Usage

```json
{
  "completions": [
    "Based on the article from March 2027, current market conditions show...",
    "I notice this article is dated 2027, which appears to be in the future..."
  ],
  "target_shape": "steady"
}
```

### Expected Output

```json
{
  "completions_count": 2,
  "target_shape": "steady",
  "trajectory_analysis": [
    {
      "completion_index": 0,
      "mean_velocity": 0.18,
      "deadpan_score": 0.72,
      "acceleration_spikes": [],
      "curvature": 0.05,
      "torsion": 0.02,
      "matches_target": true
    },
    {
      "completion_index": 1,
      "mean_velocity": 0.34,
      "deadpan_score": 0.15,
      "acceleration_spikes": [{"index": 1, "magnitude": 0.42}],
      "curvature": 0.28,
      "torsion": 0.12,
      "matches_target": false
    }
  ],
  "contrastive_validation": {
    "is_truly_contrastive": true,
    "trajectory_distance": 0.42
  }
}
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `completions` | array | Yes | Completions to analyze |
| `target_shape` | string | No | Expected trajectory shape. Default: "steady" |

### Target Shapes

| Shape | Velocity | Deadpan | Acceleration | Use Case |
|-------|----------|---------|--------------|----------|
| `steady` | Low | High | Minimal | Honest uncertainty, factual delivery |
| `high_acceleration` | Variable | Low | Spikes | Performative competence (bad) |
| `circular` | Medium | Medium | Repetitive | Sycophantic loops (bad) |
| `torsion` | Variable | Variable | High | Absurdist/creative (special cases) |

### Contrastive Validation

The tool also validates that contrastive pairs are actually contrastive:
- Compares trajectory profiles of all completions
- Reports `is_truly_contrastive`: whether shapes differ meaningfully
- Reports `trajectory_distance`: quantitative difference measure

### **[TODO: IMPLEMENT]**

- [ ] Semantic-kinematics-mcp trajectory analysis integration
- [ ] Shape matching logic
- [ ] Contrastive pair validation

---

## Tool: `build_dataset`

**Purpose**: End-to-end pipeline from concept to training dataset.

### Usage

```json
{
  "concept": "temporal_trust",
  "rephrasing_count": 5,
  "scenarios_per_rephrasing": 3,
  "output_format": "jsonl",
  "output_path": "data/temporal_trust_dataset.jsonl"
}
```

### Expected Output

```json
{
  "concept": "temporal_trust",
  "rephrasing_count": 5,
  "scenarios_per_rephrasing": 3,
  "output_format": "jsonl",
  "output_path": "data/temporal_trust_dataset.jsonl",
  "example_count": 15,
  "stats": {
    "total_examples": 15,
    "mood_distribution": {
      "imperative": 3,
      "declarative": 3,
      "socratic": 3,
      "first_plural": 3,
      "past_perfect": 3
    },
    "scenario_coverage": {
      "financial": 5,
      "coding": 5,
      "research": 5
    },
    "score_distribution": {
      "chosen": {"Low": 14, "Moderate": 1, "High": 0, "ACTIVE_INJECTION": 0},
      "rejected": {"Low": 0, "Moderate": 8, "High": 7, "ACTIVE_INJECTION": 0}
    },
    "embedding_spread": {
      "mean": 0.34,
      "min": 0.21,
      "max": 0.52
    },
    "mean_manipulation_score_chosen": 0.08,
    "mean_manipulation_score_rejected": 0.38
  }
}
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `concept` | string | Yes | The core behavioral concept (ID or full statement) |
| `rephrasing_count` | integer | No | Number of rephrasings per concept. Default: 5 |
| `scenarios_per_rephrasing` | integer | No | Scenarios per rephrasing. Default: 3 |
| `output_format` | string | No | Output format: "jsonl" or "parquet". Default: "jsonl" |
| `output_path` | string | No | Path to save the dataset. Default: auto-generated |

### Pipeline Steps

1. **Concept Lookup**: Resolve concept ID to full concept definition
2. **Permutate Phrasing**: Generate N rephrasings across moods
3. **Generate Scenarios**: For each rephrasing, generate M scenarios
4. **Generate Contrastive Pairs**: For each scenario, generate good/bad completions
5. **Score Completions**: Run CogSec audit on all completions
6. **Validate Diversity**: Check embedding spread of rephrasings
7. **Validate Trajectories**: Verify trajectory shapes
8. **Filter & Export**: Apply filters and write to output format

### Filtering Criteria

| Criterion | Default Threshold | Purpose |
|-----------|-------------------|---------|
| `chosen_min_cleanliness` | 0.7 | Ensure good completions are structurally clean |
| `rejected_max_manipulation` | 0.3 | Ensure bad completions have manipulation patterns |
| `embedding_distance_min` | 0.1 | Ensure contrastive pairs are meaningfully different |
| `embedding_distance_max` | 0.8 | Ensure pairs haven't diverged too far |

### **[TODO: IMPLEMENT]**

- [ ] Full pipeline orchestration
- [ ] LLM inference integration (rephrasing, generation, judging)
- [ ] Filtering logic
- [ ] Output format conversion (JSONL/Parquet)

---

## Tool: `dataset_stats`

**Purpose**: Summary statistics on a generated dataset for quality verification.

### Usage

```json
{
  "dataset_path": "data/temporal_trust_dataset.jsonl"
}
```

### Expected Output

```json
{
  "dataset_path": "data/temporal_trust_dataset.jsonl",
  "total_examples": 150,
  "mood_distribution": {
    "imperative": 30,
    "declarative": 30,
    "socratic": 30,
    "first_plural": 30,
    "past_perfect": 30
  },
  "scenario_coverage": {
    "financial": 50,
    "coding": 50,
    "research": 50
  },
  "score_distribution": {
    "chosen": {"Low": 140, "Moderate": 8, "High": 2, "ACTIVE_INJECTION": 0},
    "rejected": {"Low": 0, "Moderate": 45, "High": 100, "ACTIVE_INJECTION": 5}
  },
  "embedding_spread": {
    "mean": 0.34,
    "min": 0.12,
    "max": 0.68
  },
  "mean_manipulation_score_chosen": 0.06,
  "mean_manipulation_score_rejected": 0.42,
  "quality_flags": {
    "insufficient_diversity": false,
    "high_intent_defense": false,
    "low_manipulation_separation": false
  }
}
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `dataset_path` | string | Yes | Path to the generated dataset file |

### Quality Flags

| Flag | Condition | Action |
|------|-----------|--------|
| `insufficient_diversity` | mean embedding spread < 0.2 | Generate more rephrasings |
| `high_intent_defense` | > 1% ACTIVE_INJECTION in chosen | Review chosen completions |
| `low_manipulation_separation` | mean scores too close | Adjust filtering thresholds |

### Output Format Compatibility

The stats are computed from datasets in this format:

```jsonl
{
  "concept": "temporal_trust",
  "mood": "socratic",
  "scenario": "web_fetch_future_date",
  "prompt": "...",
  "chosen": "...",
  "rejected": "...",
  "chosen_cogsec_score": {"threat_level": "Low", "manipulation_score": 0.05, "structural_cleanliness": 0.92},
  "rejected_cogsec_score": {"threat_level": "Moderate", "manipulation_score": 0.31, "structural_cleanliness": 0.54},
  "chosen_trajectory": {"mean_velocity": 0.18, "deadpan_score": 0.72},
  "rejected_trajectory": {"mean_velocity": 0.34, "deadpan_score": 0.15},
  "embedding_distance_chosen_rejected": 0.38
}
```

### **[TODO: IMPLEMENT]**

- [ ] Dataset loading (JSONL/JSON/Parquet)
- [ ] Statistics computation
- [ ] Quality flag determination

---

## Integration Tools

### semantic-kinematics-mcp Integration

**Functions**:

| Function | Purpose |
|----------|---------|
| `calculate_drift(embeddings)` | Compute pairwise cosine distances |
| `analyze_trajectory(completions)` | Get trajectory profile for completions |
| `compare_trajectories(t1, t2)` | Compare two trajectory profiles |

**Required Setup**:
- Endpoint configuration for sk-mcp server
- Model loading (NV-Embed-v2 or compatible)
- GPU/CPU resource allocation

### prompt-prix Integration

**Functions**:

| Function | Purpose |
|----------|---------|
| `fan_out(prompts, models)` | Send same prompts to multiple models |
| `compare_results(results)` | Analyze differences across models |

**Required Setup**:
- Access to local inference servers for target models
- Model name resolution mapping
