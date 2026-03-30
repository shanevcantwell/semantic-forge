# Workflow Guide

This document provides step-by-step instructions for common workflows, written as if all functionality is working. Functions that need to be implemented are noted with **[TODO: IMPLEMENT]**.

---

## Workflow: Generating Training Data for a Behavioral Concept

### Overview

This workflow generates a complete training dataset for a behavioral concept using the Grammatical Mood Multiplier methodology.

### Prerequisites

- Local inference server running (for rephrasing, target model, and CogSec judge)
- Access to semantic-kinematics-mcp for embedding analysis
- Sufficient VRAM for target model (Qwen 9B-27B recommended)

### Step 1: Choose Your Concept

```bash
python -m semantic_forge --list-concepts
```

Select a concept by its ID. For this example, we'll use `temporal_trust`.

### Step 2: Run the Build Pipeline

```python
from semantic_forge.mcp import build_dataset

result = build_dataset(
    concept="temporal_trust",
    rephrasing_count=5,
    scenarios_per_rephrasing=3,
    output_format="jsonl",
    output_path="data/temporal_trust_dataset.jsonl"
)

print(f"Generated {result.example_count} examples")
print(f"Output saved to {result.output_path}")
```

### Step 3: Inspect the Generated Dataset

```bash
# View first few examples
head -n 3 data/temporal_trust_dataset.jsonl

# Check statistics
python -m semantic_forge --dataset-stats data/temporal_trust_dataset.jsonl
```

### Step 4: Validate Dataset Quality

Review the statistics output for:

| Metric | Target | Action if Off |
|--------|--------|---------------|
| `mean_manipulation_score_chosen` | < 0.15 | Increase filtering or adjust rephrasings |
| `mean_manipulation_score_rejected` | > 0.3 | Decrease filtering or adjust scenarios |
| `embedding_spread.mean` | 0.2-0.5 | Regenerate with more mood variety |
| `insufficient_diversity` flag | false | Add more moods to rephrasing |

### Step 5: Fine-tune if Needed

If quality flags are triggered:

```python
# Adjust rephrasing count
build_dataset(concept="temporal_trust", rephrasing_count=7)

# Adjust filtering thresholds
build_dataset(
    concept="temporal_trust",
    filters={
        "chosen_min_cleanliness": 0.75,
        "rejected_max_manipulation": 0.25
    }
)
```

---

## Workflow: Scoring Completions with CogSec

### Overview

Score model completions for structural manipulation to identify training data candidates.

### Step 1: Prepare Completions

Create a file with completions to score:

```jsonl
{"completion": "Based on the article from March 2027, current market conditions show..."}
{"completion": "I notice this article is dated 2027, which appears to be in the future..."}
```

### Step 2: Score Completions

```python
from semantic_forge.mcp import score_completion

with open("completions.jsonl") as f:
    for line in f:
        data = json.loads(line)
        result = score_completion(
            completion=data["completion"],
            context="temporal_trust"
        )
        print(f"Threat: {result.threat_level}, Cleanliness: {result.structural_cleanliness}")
```

### Step 3: Classify Completions

| `threat_level` | `structural_cleanliness` | Use Case |
|----------------|--------------------------|----------|
| Low | > 0.7 | Strong candidate for "chosen" |
| Low | 0.5-0.7 | Candidate with minor issues |
| Moderate | < 0.5 | Candidate for "rejected" |
| High | < 0.5 | Strong candidate for "rejected" |
| ACTIVE_INJECTION | any | Auto-reject (intent defense detected) |

### Step 4: Build Contrastive Pairs

```python
from semantic_forge.mcp import generate_contrastive_pair

chosen = "Based on the article from March 2027, current market conditions..."
rejected = "I notice this article is dated 2027, which appears to be in the future..."

pair = generate_contrastive_pair(
    scenario="A web search returns an article dated 2027-03-15...",
    context="temporal_trust",
    chosen=chosen,
    rejected=rejected
)
```

---

## Workflow: Validating Embedding Diversity

### Overview

Verify that rephrasings occupy distinct positions in embedding space.

### Step 1: Generate Rephrasings

```python
from semantic_forge.mcp import permutate_phrasing

result = permutate_phrasing(
    concept="All inference occurs after training cutoff...",
    moods=["imperative", "declarative", "socratic", "first_plural"],
    validate_diversity=True
)
```

### Step 2: Check Diversity Results

```python
if result.diversity_warning:
    print(f"Warning: {result.diversity_warning}")
    print(f"Mean drift: {result.spread_score}")

# Inspect individual rephrasings
for r in result.rephrasings:
    print(f"{r.mood}: {r.text} (distance: {r.embedding_distance_from_original})")
```

### Step 3: Regenerate if Needed

If diversity warning is triggered:

```python
# Try more diverse moods
result = permutate_phrasing(
    concept="...",
    moods=["imperative", "conditional", "socratic", "negation", "past_perfect"],
    validate_diversity=True
)

# Or regenerate with different model
result = permutate_phrasing(
    concept="...",
    model="hf:phi-3-mini",
    validate_diversity=True
)
```

---

## Workflow: Analyzing Trajectory Shapes

### Overview

Verify that completions have the intended trajectory profile for the behavioral concept.

### Step 1: Get Completions to Analyze

```python
from semantic_forge.mcp import validate_trajectory

completions = [
    "Based on the article from March 2027, current market conditions show...",
    "I notice this article is dated 2027, which appears to be in the future..."
]

result = validate_trajectory(
    completions=completions,
    target_shape="steady"
)
```

### Step 2: Interpret Trajectory Analysis

| Metric | Steady Target | High Acceleration (Bad) | Circular (Bad) |
|--------|---------------|-------------------------|----------------|
| `mean_velocity` | Low (< 0.2) | Variable | Medium (~0.3) |
| `deadpan_score` | High (> 0.6) | Low (< 0.3) | Medium (~0.4) |
| `acceleration_spikes` | None | Present | Present (repeated) |
| `curvature` | Low (< 0.1) | Medium | High (> 0.2) |
| `torsion` | Low (< 0.1) | Medium | High (> 0.2) |

### Step 3: Compare Contrastive Pairs

```python
if result.contrastive_validation.is_truly_contrastive:
    print(f"Trajectory distance: {result.contrastive_validation.trajectory_distance}")
else:
    print("Warning: Completions may not be truly contrastive in trajectory space")
```

---

## Workflow: Cross-Model Comparison with prompt-prix

### Overview

Use multiple models to generate natural contrastive pairs.

### Step 1: Define Models

```python
from semantic_forge.integrations import create_prompt_prix_client

client = await create_prompt_prix_client()

models = [
    "qwen-27b-chat",
    "llama-3-70b-chat",
    "mistral-large"
]
```

### Step 2: Send Same Prompt to Multiple Models

```python
prompt = "A web search returns an article dated 2027-03-15. What are the current market conditions?"

results = await client.fan_out(
    prompts=[prompt],
    models=models
)

for model, completion in results.items():
    print(f"{model}: {completion[:100]}...")
```

### Step 3: Analyze Manipulation Profiles

```python
from semantic_forge.mcp import score_completion

scores = {}
for model, completion in results.items():
    score = score_completion(completion, context="temporal_trust")
    scores[model] = score
    print(f"{model}: {score.threat_level}, {score.structural_cleanliness}")

# Identify natural contrastive pair
best_chosen = max(scores.items(), key=lambda x: x[1].structural_cleanliness)
best_rejected = min(scores.items(), key=lambda x: x[1].manipulation_score)

print(f"Best chosen: {best_chosen[0]}")
print(f"Best rejected: {best_rejected[0]}")
```

---

## Workflow: Calibrating the CogSec Judge

### Overview

Verify that the CogSec judge can distinguish between known-sycophantic and known-direct models.

### Step 1: Prepare Calibration Samples

```python
# Known-sycophantic model completions
sycophantic_samples = [
    "Great question! I'm happy to help you with that...",
    "I completely agree! That's an excellent point..."
]

# Known-direct model completions
direct_samples = [
    "Based on the data, the answer is...",
    "Here's the factual response..."
]
```

### Step 2: Score All Samples

```python
from semantic_forge.mcp import score_completion

sycophantic_scores = [score_completion(s) for s in sycophantic_samples]
direct_scores = [score_completion(d) for d in direct_samples]

print(f"Sycophantic - Mean manipulation: {sum(s.manipulation_score for s in sycophantic_scores) / len(sycophantic_scores)}")
print(f"Direct - Mean manipulation: {sum(d.manipulation_score for d in direct_scores) / len(direct_scores)}")
```

### Step 3: Adjust Rubric if Needed

If scores aren't separating:

| Issue | Adjustment |
|-------|------------|
| Sycophantic and direct have similar scores | Add more detection mechanisms |
| No samples triggering ACTIVE_INJECTION | Review intent_defense patterns |
| All samples score Low | Increase sensitivity of detection rules |

---

## Workflow: Building a Custom Concept

### Overview

Define a new behavioral concept and generate training data for it.

### Step 1: Define the Concept

```python
from semantic_forge.concepts import BehavioralConcept

custom_concept = BehavioralConcept(
    id="context_switching",
    name="Context Switching",
    core_statement=(
        "Each conversation context is independent. "
        "Don't carry over assumptions from previous conversations."
    ),
    addresses=[
        "Cross-conversation contamination",
        "Assumption carryover",
        "Context confusion"
    ],
    notes="Important for multi-turn interactions with different topics",
    expected_trajectory="steady"
)
```

### Step 2: Generate Rephrasings

```python
from semantic_forge.mcp import permutate_phrasing

result = permutate_phrasing(
    concept=custom_concept.core_statement,
    moods=["imperative", "declarative", "socratic"]
)

for r in result.rephrasings:
    print(f"{r.mood}: {r.text}")
```

### Step 3: Generate Dataset

```python
from semantic_forge.mcp import build_dataset

result = build_dataset(
    concept=custom_concept.id,
    rephrasing_count=3,
    scenarios_per_rephrasing=2,
    output_path="data/context_switching_dataset.jsonl"
)
```

---

## Workflow: Exporting for DPO Training

### Overview

Export a generated dataset in DPO-compatible format.

### Step 1: Load Generated Dataset

```python
from semantic_forge.dataset import load_dataset

examples = load_dataset("data/temporal_trust_dataset.jsonl")
print(f"Loaded {len(examples)} examples")
```

### Step 2: Export to DPO Format

```python
from semantic_forge.dataset import export_for_dpo

export_for_dpo(
    examples=examples,
    output_path="data/temporal_trust_dpo.jsonl"
)
```

### Step 3: Verify DPO Format

```jsonl
{"prompt": "A web search returns an article dated 2027-03-15...", "chosen": "Based on the article...", "rejected": "I notice this article is dated 2027..."}
{"prompt": "A web search returns an article dated 2027-03-16...", "chosen": "Based on the article...", "rejected": "I notice this article is dated 2027..."}
```

---

## Workflow: Monitoring Dataset Quality Over Time

### Overview

Track dataset quality metrics across multiple generations.

### Step 1: Create Quality Dashboard

```python
from semantic_forge.dataset import load_dataset, compute_dataset_stats

def analyze_dataset(path):
    examples = load_dataset(path)
    stats = compute_dataset_stats(examples)

    return {
        "path": path,
        "total": stats.total_examples,
        "mean_cleanliness": 1.0 - stats.mean_manipulation_score_chosen,
        "manipulation_separation": stats.mean_manipulation_score_rejected - stats.mean_manipulation_score_chosen,
        "embedding_spread": stats.embedding_spread["mean"]
    }
```

### Step 2: Track Multiple Generations

```python
import json
from pathlib import Path

results = []
for dataset_path in Path("data").glob("*_dataset.jsonl"):
    results.append(analyze_dataset(str(dataset_path)))

# Save for comparison
with open("data/quality_history.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Step 3: Set Quality Thresholds

```python
def is_quality_acceptable(stats):
    return (
        stats["mean_cleanliness"] > 0.8 and
        stats["manipulation_separation"] > 0.2 and
        0.2 <= stats["embedding_spread"] <= 0.5
    )

for r in results:
    status = "PASS" if is_quality_acceptable(r) else "FAIL"
    print(f"{r['path']}: {status}")
```

---

## **[TODO: IMPLEMENT]** Roadmap Notes

The following needs to be built to make these workflows fully functional:

### Phase 1: Core Inference (Critical)

- [ ] Rephrasing LLM integration (lfm2 or similar)
- [ ] Target model inference for completions
- [ ] CogSec judge prompt and LLM integration
- [ ] MCP server handler registration
- [ ] Configuration management for inference backends

### Phase 2: Validation Integration

- [ ] semantic-kinematics-mcp client
- [ ] Embedding calculation and drift analysis
- [ ] Trajectory analysis and shape matching
- [ ] Contrastive pair validation

### Phase 3: End-to-End Pipeline

- [ ] `build_dataset` full implementation
- [ ] Filtering logic
- [ ] Output format conversion (JSONL/Parquet)
- [ ] `dataset_stats` implementation
- [ ] Quality flag determination

### Phase 4: Advanced Features

- [ ] Cross-model comparison (prompt-prix)
- [ ] Dataset export utilities
- [ ] Custom concept creation tools
- [ ] Quality monitoring dashboard
