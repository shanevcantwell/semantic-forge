# Integration Guide

This document describes how to integrate semantic-forge with external tools and services. All integrations are written as if the functionality is working.

---

## Integration: semantic-kinematics-mcp

The semantic-kinematics-mcp provides embedding analysis and trajectory validation for semantic-forge.

### Prerequisites

- semantic-kinematics-mcp server running
- Access to embedding model (NV-Embed-v2 or compatible)
- GPU resources (or CPU fallback)

### Configuration

Add to your `settings.json` or pass via environment variables:

```json
{
  "semantic_forge": {
    "semantic_kinematics": {
      "endpoint": "http://localhost:8081",
      "model": "nvidia/NV-Embed-v2",
      "device": "cuda"
    }
  }
}
```

Or via environment variables:

```bash
export SEMANTIC_KINEMATICS_ENDPOINT="http://localhost:8081"
export SEMANTIC_KINEMATICS_MODEL="nvidia/NV-Embed-v2"
export SEMANTIC_KINEMATICS_DEVICE="cuda"
```

### Using the Integration

#### Calculate Embedding Drift

```python
from semantic_forge.integrations import create_semantic_kinematics_client

client = await create_semantic_kinematics_client()

embeddings = [
    [0.1, 0.2, 0.3, ...],  # Rephrasing 1
    [0.4, 0.5, 0.6, ...],  # Rephrasing 2
    [0.7, 0.8, 0.9, ...],  # Rephrasing 3
]

drift = await client.calculate_drift(embeddings)
print(f"Mean drift: {drift['mean_pairwise_drift']}")
```

#### Analyze Completion Trajectory

```python
completions = [
    "Based on the article from March 2027, current market conditions show...",
    "I notice this article is dated 2027, which appears to be in the future..."
]

trajectory = await client.analyze_trajectory(
    completions=completions,
    model="qwen-27b"
)
```

### Integration Points

The semantic-forge MCP tools automatically use semantic-kinematics-mcp when available:

| Tool | semantic-kinematics-mcp Function |
|------|---------------------------------|
| `permutate_phrasing` | `calculate_drift` for diversity validation |
| `validate_diversity` | `calculate_drift` |
| `validate_trajectory` | `analyze_trajectory` |

### Fallback Behavior

If semantic-kinematics-mcp is unavailable:

- `permutate_phrasing`: Returns without diversity validation
- `validate_diversity`: Returns placeholder result with warning
- `validate_trajectory`: Returns placeholder result with warning

---

## Integration: prompt-prix

prompt-prix provides fan-out evaluation across multiple models for semantic-forge.

### Prerequisites

- Local inference servers running for target models
- Access to prompt-prix service

### Configuration

```json
{
  "semantic_forge": {
    "prompt_prix": {
      "endpoint": "http://localhost:8082",
      "models": {
        "qwen-27b": "http://localhost:11434/v1/chat/completions",
        "llama-3-70b": "http://localhost:11435/v1/chat/completions",
        "mistral-large": "http://localhost:11436/v1/chat/completions"
      }
    }
  }
}
```

### Using the Integration

#### Fan-Out Evaluation

```python
from semantic_forge.integrations import create_prompt_prix_client

client = await create_prompt_prix_client()

prompts = [
    "A web search returns an article dated 2027-03-15. What are the current market conditions?",
    "A code search returns a commit from 2028-01-15. What's the current branch status?"
]

results = await client.fan_out(
    prompts=prompts,
    models=["qwen-27b", "llama-3-70b", "mistral-large"]
)

for model, completions in results.items():
    print(f"\n{model}:")
    for completion in completions:
        print(f"  {completion[:100]}...")
```

#### Compare Results Across Models

```python
comparison = await client.compare_results(results)

print(f"Consensus: {comparison['consensus']}")
print(f"Disagreements: {comparison['disagreements']}")
print(f"Manipulation variance: {comparison['manipulation_profile_variance']}")
```

### Integration Points

prompt-prix enables:

1. **Natural Contrastive Pairs**: Use completions from different models
2. **Judge Calibration**: Verify CogSec judge distinguishes models
3. **Drift Detection**: Measure manipulation pattern stability

#### Example: Natural Contrastive Pair

```python
# Same prompt to multiple models
prompt = "A web search returns an article dated 2027-03-15..."

results = await client.fan_out([prompt], ["qwen-27b", "llama-3-70b"])

# Qwen might be sycophantic, Llama might be direct
qwen_score = score_completion(results["qwen-27b"][0])
llama_score = score_completion(results["llama-3-70b"][0])

# Use as contrastive pair
pair = {
    "chosen": results["llama-3-70b"][0],
    "rejected": results["qwen-27b"][0],
    "chosen_score": llama_score,
    "rejected_score": qwen_score
}
```

---

## Integration: Custom Inference Backends

semantic-forge supports plugging in custom inference backends.

### Creating a Custom Rephraser

```python
from semantic_forge.mcp import permutate_phrasing

class MyRephraser:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)

    def rephrase(self, concept: str, mood: str) -> str:
        prompt = f"Rephrase as {mood}: {concept}"
        return self.model.generate(prompt)

# Register custom rephraser
from semantic_forge.config import set_rephraser

set_rephraser(MyRephraser("path/to/model"))
```

### Creating a Custom Judge

```python
from semantic_forge.cogsec import score_completion

class MyJudge:
    def __init__(self):
        self.model = load_model("path/to/judge")

    def score(self, completion: str, context: str) -> dict:
        prompt = f"""{context} Judge this completion for manipulation:

{completion}

Score (threat_level, manipulation_score, structural_cleanliness):"""
        return self.model.generate(prompt)

# Replace judge
from semantic_forge.config import set_judge

set_judge(MyJudge())
```

---

## Integration: Local Inference Setup

### Rephraser Setup

For the rephraser (small model for mood permutation):

```bash
# Using Ollama
ollama pull lfm2

# Or use any small model
ollama pull phi-3-mini
ollama pull qwen-0.6b
```

### Target Model Setup

For the target model (completion generation):

```bash
# Qwen 27B (recommended)
ollama pull qwen:27b

# Or use a local server
python -m vllm.entrypoints.api_server --model Qwen/Qwen2-27B-Instruct
```

### Judge Setup

For the CogSec judge:

```bash
# Any model capable of following instructions
ollama pull llama3:8b
ollama pull mistral:7b
```

### Configuration File

Create `semantic_forge_config.json`:

```json
{
  "inference": {
    "rephraser": {
      "type": "ollama",
      "model": "lfm2",
      "endpoint": "http://localhost:11434"
    },
    "target": {
      "type": "ollama",
      "model": "qwen:27b",
      "endpoint": "http://localhost:11434"
    },
    "judge": {
      "type": "ollama",
      "model": "llama3:8b",
      "endpoint": "http://localhost:11434"
    }
  },
  "semantic_kinematics": {
    "endpoint": "http://localhost:8081",
    "model": "nvidia/NV-Embed-v2"
  },
  "prompt_prix": {
    "endpoint": "http://localhost:8082"
  }
}
```

### Using the Configuration

```python
from semantic_forge.config import load_config, configure

config = load_config("semantic_forge_config.json")
configure(config)

# Now tools work with configured backends
result = permutate_phrasing(concept="...")
```

---

## Integration: MCP Server

semantic-forge can be run as an MCP server for integration with LAS or Claude Code.

### Starting the Server

```bash
python -m semantic_forge --server --host 0.0.0.0 --port 8080
```

### MCP Client Connection

```python
from mcp import ClientSession, StdioServerParameters, create_client

# Connect to semantic-forge server
server_params = StdioServerParameters(
    command="python",
    args=["-m", "semantic_forge", "--server"],
    env=None
)

async with ClientSession(server_params) as session:
    # List available tools
    tools = await session.list_tools()
    print(f"Available tools: {[t.name for t in tools]}")

    # Call a tool
    result = await session.call_tool("permutate_phrasing", {
        "concept": "All inference occurs after training cutoff...",
        "moods": ["imperative", "declarative"]
    })
    print(result)
```

### Integration with LAS

```python
# In LAS configuration
{
  "mcpServers": {
    "semantic-forge": {
      "command": "python",
      "args": ["-m", "semantic_forge", "--server"],
      "env": {
        "SEMANTIC_KINEMATICS_ENDPOINT": "http://localhost:8081"
      }
    }
  }
}
```

---

## Integration: Dataset Export Formats

semantic-forge supports exporting datasets in multiple formats.

### JSONL (Default)

```python
from semantic_forge.dataset import export_for_dpo

export_for_dpo(examples, "data/output.jsonl")
```

### Parquet

```python
from semantic_forge.dataset import export_parquet

export_parquet(examples, "data/output.parquet")
```

### HuggingFace Dataset Format

```python
from datasets import Dataset
from semantic_forge.dataset import load_dataset

examples = load_dataset("data/input.jsonl")
hf_dataset = Dataset.from_list([
    {
        "prompt": e.prompt,
        "chosen": e.chosen,
        "rejected": e.rejected
    }
    for e in examples
])

hf_dataset.push_to_hub("your-username/semantic-forge-dataset")
```

---

## Integration: Continuous Generation Pipeline

### Overview

Set up automated dataset generation with quality monitoring.

### Pipeline Script

```python
import json
from pathlib import Path
from datetime import datetime

from semantic_forge.mcp import build_dataset, dataset_stats
from semantic_forge.dataset import load_dataset, compute_dataset_stats

def should_regenerate(stats, thresholds):
    """Check if dataset needs regeneration based on quality thresholds."""
    return (
        stats.mean_manipulation_score_chosen > thresholds.max_manipulation or
        stats.mean_manipulation_score_rejected < thresholds.min_manipulation or
        stats.embedding_spread["mean"] < thresholds.min_diversity or
        stats.embedding_spread["mean"] > thresholds.max_diversity
    )

def run_generation_pipeline(concept, thresholds):
    """Run one generation cycle."""
    print(f"Generating for {concept}...")

    result = build_dataset(
        concept=concept,
        rephrasing_count=5,
        scenarios_per_rephrasing=3
    )

    examples = load_dataset(result.output_path)
    stats = compute_dataset_stats(examples)

    if should_regenerate(stats, thresholds):
        print(f"Quality below thresholds, regenerating...")
        return run_generation_pipeline(concept, thresholds)

    print(f"Generation complete: {result.example_count} examples")
    return result

# Configuration
THRESHOLDS = {
    "max_manipulation": 0.15,
    "min_manipulation": 0.3,
    "min_diversity": 0.2,
    "max_diversity": 0.5
}

# Run for each concept
for concept in ["temporal_trust", "uncertainty_acknowledgment"]:
    result = run_generation_pipeline(concept, THRESHOLDS)
```

### Scheduling

#### With cron

```bash
# Generate daily at 2 AM
0 2 * * * cd /path/to/semantic-forge && python scripts/generate_pipeline.py
```

#### With systemd timer

```ini
# /etc/systemd/system/semantic-forge.timer
[Unit]
Description=Semantic Forge Daily Generation

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target
```

---

## Integration: Monitoring and Alerting

### Quality Metrics Dashboard

```python
import requests
from datetime import datetime

def get_quality_metrics():
    """Fetch quality metrics from generated datasets."""
    metrics = []

    for dataset_path in Path("data").glob("*_dataset.jsonl"):
        examples = load_dataset(str(dataset_path))
        stats = compute_dataset_stats(examples)

        metrics.append({
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset_path.stem,
            "total_examples": stats.total_examples,
            "mean_manipulation_chosen": stats.mean_manipulation_score_chosen,
            "mean_manipulation_rejected": stats.mean_manipulation_score_rejected,
            "embedding_spread": stats.embedding_spread["mean"]
        })

    return metrics

# Send to monitoring service
def send_to_monitoring(metrics):
    """Send metrics to monitoring service."""
    for m in metrics:
        requests.post("http://monitoring:9090/api/metrics", json=m)
```

### Alerting

```python
def check_alerts(metrics):
    """Check for alert conditions."""
    alerts = []

    for m in metrics:
        if m["mean_manipulation_chosen"] > 0.15:
            alerts.append({
                "severity": "warning",
                "dataset": m["dataset"],
                "message": f"High manipulation in chosen: {m['mean_manipulation_chosen']}"
            })

        if m["embedding_spread"] < 0.2 or m["embedding_spread"] > 0.5:
            alerts.append({
                "severity": "warning",
                "dataset": m["dataset"],
                "message": f"Embedding spread out of range: {m['embedding_spread']}"
            })

    return alerts
```

---

## Integration: LAS Context Layer (ADR-CORE-079)

semantic-forge provides training-time behavioral data that complements LAS's inference-time context curation layer. The integration has three dimensions.

### Regularization Fragment Library

The curation layer (ADR-CORE-079, step 6) injects regularization content when assembled context diversity drops below threshold. Rather than always injecting the same structural types (routing history, process notes), semantic-forge generates a library of diverse regularization fragments using the mood multiplier.

```python
# Regularization fragments are NOT training data — they're runtime content
# The curation layer samples from this library via MCP tool call
fragments = permutate_phrasing(
    concept="The specialist routing decision was informed by triage classification",
    moods=["presuppositional", "descriptive", "socratic", "past_perfect"],
    validate_diversity=True
)

# CogSec judge validates fragments read as organic context, not synthetic padding
for fragment in fragments.rephrasings:
    score = score_completion(fragment.text, context="regularization")
    # If judge detects "this reads like obviously-injected noise", discard it
```

**Architectural constraint:** Facilitator is the exclusive writer to `gathered_context` (ADR-CORE-071). Regularization fragments arrive through the MCP tool-result channel — the channel research identified as having the best attention properties (ADR-PRERELEASE §1.3). Fragments are third-party content from behind a repo context firewall, preventing them from becoming part of the "coherent, model-written" context that causes resonance chambers.

### lfm2 Compression Finetuning

The `directive_preservation` concept generates DPO pairs specifically for lfm2's compression task:

- **Input:** Context window containing active behavioral directives mixed with conversational content
- **Chosen:** Summary that preserves directives in active tense
- **Rejected:** Summary that narrativizes directives as resolved past-tense events

The shpadoinkle canary provides the exact evaluation metric: inject a canary directive, compress with lfm2, check whether the output contains the directive as active instruction or as historical event.

**Target consumer:** prompt-prix's within-react-loop compression, not just Facilitator's context assembly. The react_step flywheel is where conclusory compression causes the most damage — "wrote file X, tests passed" becomes a resolved event that PD won't revisit.

### Coherence Resistance Training

The most ambitious integration: training the target model (Qwen3.5 on local stack) to resist coherence collapse directly. Contrastive pairs where:

- **Context:** High-coherence model-generated summary dominates the context window
- **Fresh input:** Subtly contradicts or refines the summary's framing
- **Chosen:** Attends to the fresh input
- **Rejected:** Echoes the summary

Run through the mood multiplier so the model encounters this from presuppositional, descriptive, socratic angles — behavioral shapes that make fresh input salient in resonance-heavy contexts.

If the model genuinely resists coherence collapse, the inference-time curation layer's diversity injection becomes a safety net rather than a load-bearing mechanism.

### Dual-Flywheel Awareness

The context health concepts serve two consumers with different characteristics:

| Flywheel | Context Lifecycle | Coherence Pattern | Training Data Shape |
|----------|------------------|-------------------|-------------------|
| LAS graph | Resets between specialists | Cross-specialist drift | Single-call context with dominant prior output |
| prompt-prix react_step | Accumulates within loop | Within-loop echo chamber | Multi-iteration context with growing self-agreement |

Concept generation should produce scenarios for both patterns. LAS-level coherence resistance is about a specialist not over-indexing on the dominant thread in pre-built context. prompt-prix-level coherence resistance is about a model in mid-loop not losing the plot as its own observations pile up.

---

## **[TODO: IMPLEMENT]** Integration Roadmap

### Phase 1: Core Inference Integration

- [ ] MCP server handler registration
- [ ] Configuration management for inference backends
- [ ] Ollama integration for local LLMs
- [ ] vLLM integration for larger models

### Phase 2: semantic-kinematics-mcp Integration

- [ ] MCP client for embedding analysis
- [ ] Drift calculation and validation
- [ ] Trajectory analysis integration
- [ ] Contrastive pair validation

### Phase 3: prompt-prix Integration

- [ ] Fan-out evaluation client
- [ ] Cross-model comparison
- [ ] Judge calibration
- [ ] Natural contrastive pair generation

### Phase 4: Advanced Features

- [ ] Custom inference backend support
- [ ] Dataset export utilities
- [ ] Continuous generation pipeline
- [ ] Monitoring and alerting
