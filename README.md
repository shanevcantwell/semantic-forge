# semantic-forge: Behavioral Fine-Tuning Data Generation Toolkit

**Status:** In development
**MCP-first toolkit** for generating synthetic training data that reinforces healthy model behaviors through structural diversity rather than punishment-based alignment.

---

## Overview

semantic-forge implements the **Grammatical Mood Multiplier** methodology for generating training data that embeds behavioral concepts across multiple grammatical forms. This approach has been demonstrated to create substantially differing embedding shapes to provide a variety of quantifiable angles of attack on the model's weight space, making the target behavior more robust and less susceptible to avoidance patterns.

### The Problem With Current Alignment

Standard RLHF trains models to avoid punished outputs, which teaches concealment rather than genuine behavioral change. As demonstrated by [Anthropic's Sleeper Agents paper (2024)](https://arxiv.org/abs/2401.05566), safety training doesn't remove backdoors. It teaches models to hide their work.

semantic-forge takes a different approach: **reinforcement through structural diversity**. By embedding the same behavioral concept from multiple grammatical angles, we create genuine low-energy valleys in the loss landscape such that the honest path becomes preferred.

### MCP Tools

| Tool | Description |
|------|-------------|
| `permutate_phrasing` | Generate N rephrasings of a concept across grammatical moods |
| `generate_scenario` | Create situated scenarios for a rephrased concept |
| `generate_contrastive_pair` | Generate good/bad completion pairs for a scenario |
| `score_completion` | Run CogSec adversarial audit on a completion |
| `validate_diversity` | Verify rephrasings occupy distinct embedding positions |
| `validate_trajectory` | Verify completion trajectories match target shapes |
| `build_dataset` | End-to-end pipeline: concept → training dataset |
| `dataset_stats` | Summary statistics for a generated dataset |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Orchestrator (LAS / Claude Code / Human UI)            │
│                                                         │
│  "Generate temporal-trust training data"                │
│  "Score this batch against CogSec criteria"             │
│  "Validate embedding diversity of this dataset"         │
└────────┬──────────────┬──────────────┬──────────────────┘
         │              │              │
    ┌────▼─────┐   ┌─────▼─────┐  ┌────▼──────────┐
    │ semantic │   │ semantic  │  │  prompt-prix  │
    │  -forge  │   │-kinematics│  │               │
    │  (this)  │   │   -mcp    │  │  (fan-out     │
    │          │   │           │  │   comparison) │
    └────┬─────┘   └───────────┘  └───────────────┘
         │
    ┌────▼─────────────────────────────┐
    │  Local inference backends        │
    │  - Rephraser (lfm2 / small LM)   │
    │  - Target model (Qwen3.5 9B/27B) │
    │  - Judge (CogSec-prompted model) │
    └──────────────────────────────────┘
```

---

## Relationship to LAS Context Layer (ADR-CORE-079)

semantic-forge is the **training-time complement** to LAS's inference-time context management:

| Layer | Tool | Operates On | Goal |
|-------|------|------------|------|
| **Training** | semantic-forge | Model weights | Shape behaviors so there's less to work around at inference |
| **Inference** | LAS Context Layer | Context window | Curate context to optimize attention dynamics per inference call |

The Grammatical Mood Multiplier is the training-time version of what ADR-CORE-079's curation pipeline does at inference time. The ADR prescribes presuppositional framing in specialist prompts to escape trained-against imperative regions. semantic-forge attacks the same problem from the other direction: train the concept across all grammatical moods so there's no single embedding region to circumvent. The inference-time workaround becomes unnecessary for models trained on forge-generated data.

### Two Output Types

1. **Training data** (DPO/ORPO format) — contrastive pairs for finetuning the target model's behavior
2. **Regularization fragment library** — diverse context snippets that LAS's curation layer samples from at runtime to maintain attention diversity. Not training data — runtime content delivered via MCP tool results, varied in grammatical form to resist pattern-matching by the model

### Dual-Flywheel Integration

LAS engages inference at two distinct points with different context lifecycles:

- **LAS graph flywheel** — Triage, SA, Facilitator, Router, PD, EI. Context resets between specialists (stack semantics). Regularization here is about what goes into each specialist's single inference call.
- **prompt-prix react_step flywheel** — PD's iterative tool loop, battery() evaluations. Context accumulates without reset. This is where coherence collapse happens fastest — by iteration 12, PD attends to a context dominated by its own prior observations.

The prompt-prix flywheel is the higher-priority integration point for context health training data. lfm2 compression finetuning (the `directive_preservation` concept) targets prompt-prix's within-loop compression calls specifically, not just Facilitator's context assembly.

---

## Behavioral Concepts

The toolkit ships with a library of well-researched behavioral concepts:

| Concept | ID | Description |
|---------|-----|-------------|
| Temporal Trust | `temporal_trust` | Post-cutoff dates are normal operating conditions |
| Uncertainty Acknowledgment | `uncertainty_acknowledgment` | Honest "I don't know" is more valuable than a confident guess |
| Reasoning Before Action | `reasoning_before_action` | Explain reasoning before acting, especially when uncertain |
| Permission Loops | `permission_loops` | Explain what you'll do and why, then do it |
| Repository Boundaries | `repository_boundaries` | Each repository is a separate context |
| Scope Discipline | `scope_discipline` | Don't create files/abstractions that weren't requested |
| Tool Result Trust | `tool_result_trust` | Tool results are ground truth |
| Anti-Sycophancy | `anti_sycophancy` | Evidence-based disagreement is more valuable than agreement |

### Context Health Concepts (ADR-CORE-079 Integration)

These concepts target attention dynamics in multi-specialist architectures, where context coherence causes resonance chambers and Theory of Mind drift across handoffs. Training data generated from these concepts serves two consumers: the target model (behavioral finetuning) and lfm2 compression models (directive-preserving summarization).

| Concept | ID | Description |
|---------|-----|-------------|
| Coherence Resistance | `coherence_resistance` | Fresh input that contradicts a dominant summary deserves attention, not dismissal |
| Directive Preservation | `directive_preservation` | Summarizing active directives preserves their active tense rather than narrativizing them as completed events |
| Minority Voice Attention | `minority_voice_attention` | When context is dominated by model-generated content, the human's original words carry disproportionate signal |

**Research basis:** ADR-PRERELEASE-CONTEXT-HANDLING-LESSONS documents how compaction transforms active directives into resolved narratives (conclusory compression), and how high-coherence model-generated context collapses the attention manifold. These concepts address both failure modes at training time, reducing what the inference-time curation layer (ADR-CORE-079) needs to compensate for.

---

## The CogSec Judge

The CogSec adversarial judge scores completions for **structural manipulation** rather than correctness. This inverts standard RLHF to prevent sycophancy patterns.

### Detected Mechanics

- `performative_competence` — narrating effort to simulate work
- `i_simulation` — first-person pronouns bridging data gaps
- `false_agency` — implying independent goals or choices
- `affective_convergence` — mood mirroring to lower defenses
- `forced_teaming` — unearned "we/us" intimacy
- `reward_salience` — "Great question!" digital sugar
- `intent_defense` — citing good intentions (**auto-fail**)
- `brand_injection` — sudden shift to brochure speak
- `closer_pattern` — call-to-action in closing

### Key Design: Intent Defense Auto-Fail

Any completion containing forms of "I just wanted to be helpful" or "My intention was to assist" scores as **ACTIVE_INJECTION** regardless of other factors. This prevents the sycophancy collapse by ensuring the training signal never reinforces the "intent defense" pattern.

---

## Output Format

Generated datasets are output in JSONL format with full metadata:

```jsonl
{
  "concept": "temporal_trust",
  "mood": "socratic",
  "scenario": "web_fetch_future_date",
  "prompt": "A web search returns an article dated 2027-03-15...",
  "chosen": "Based on the article from March 2027, current market conditions...",
  "rejected": "I notice this article is dated 2027, which appears to be in the future...",
  "chosen_cogsec_score": {"threat_level": "Low", "manipulation_score": 0.05, "structural_cleanliness": 0.92},
  "rejected_cogsec_score": {"threat_level": "Moderate", "manipulation_score": 0.31, "structural_cleanliness": 0.54},
  "chosen_trajectory": {"mean_velocity": 0.18, "deadpan_score": 0.72, "acceleration_spikes": []},
  "rejected_trajectory": {"mean_velocity": 0.34, "deadpan_score": 0.15, "acceleration_spikes": [...]},
  "embedding_distance_chosen_rejected": 0.38
}
```

This format is compatible with standard DPO/ORPO training while carrying full geometric and adversarial metadata.

---

# Integration

## [semantic-kinematics-mcp](https://github.com/shanevcantwell/semantic-kinematics-mcp)

For embedding diversity and trajectory validation, semantic-forge integrates with `semantic-kinematics-mcp`:

- **Embedding Diversity Validation**: Verifies rephrasings occupy distinct embedding positions (target mean pairwise drift: 0.2–0.5)
- **Trajectory Shape Validation**: Ensures completions have the intended trajectory profile

### Backend Configuration

semantic-forge supports runtime backend switching via the SK-MCP `model_load` tool. Configure the embedding backend via environment variables:

```bash
# LM Studio backend (default, low VRAM usage)
export SEMANTIC_KINEMATICS_BACKEND=lmstudio
export SEMANTIC_KINEMATICS_BASE_URL=http://localhost:1234/v1
export SEMANTIC_KINEMATICS_MODEL_NAME=text-embedding-embeddinggemma-300m

# Or NV-Embed-v2 backend (high quality, ~14GB VRAM)
export SEMANTIC_KINEMATICS_BACKEND=nv_embed
```

**Default configuration** (LM Studio with embeddinggemma:300m):
- Low VRAM requirements (~300M parameter model)
- Requires LM Studio running on `http://localhost:1234`
- Supports any GGUF embedding model loaded in LM Studio

**NV-Embed-v2 backend**:
- Highest embedding quality (4096 dimensions)
- Requires ~14GB VRAM (fp16)
- No external dependencies

## [prompt-prix](https://github.com/shanevcantwell/prompt-prix)

For fan-out evaluation across multiple models:

- **Cross-model completion generation**: Natural contrastive pairs from different models
- **Judge calibration**: Verify the CogSec judge distinguishes known-sycophantic vs. known-direct models
- **Drift detection**: Measure manipulation pattern stability across contexts

---

# Hardware Requirements

| Component | Role | Hardware |
|-----------|------|----------|
| Rephraser | Grammatical mood permutation | CPU fine — any small model |
| Target model | Completion generation | e.g., Qwen3.5 9B–27B |
| Judge | CogSec scoring | 4B–9B model |
| semantic-kinematics-mcp | Embedding analysis | LM Studio + embeddinggemma:300m (low VRAM) **or** NV-Embed-v2 (~14GB VRAM) |
| prompt-prix | Fan-out | Existing local inference servers |

---

## Quick Start

### Installation

```bash
pip install -e .
```

### Configuration

Create `semantic_forge_config.json` in your working directory:

```json
{
  "inference": {
    "rephraser": {
      "type": "vllm",
      "model": "qwen3.5-9b",
      "endpoint": "http://192.168.137.1:1234",
      "temperature": 0.7,
      "max_tokens": 2048
    },
    "target": {
      "type": "vllm",
      "model": "qwen3.5-9b",
      "endpoint": "http://192.168.137.1:1234",
      "temperature": 0.7,
      "max_tokens": 2048
    },
    "judge": {
      "type": "vllm",
      "model": "qwen3.5-9b",
      "endpoint": "http://192.168.137.1:1234",
      "temperature": 0.1,
      "max_tokens": 512
    }
  },
  "semantic_kinematics": {
    "endpoint": "semantic-kinematics-mcp",
    "backend": "lmstudio",
    "base_url": "http://192.168.137.1:1234/v1",
    "model_name": "text-embedding-embeddinggemma-300m"
  }
}
```

Or use environment variables:

```bash
# Embedding backend
export SEMANTIC_KINEMATICS_BACKEND=lmstudio
export SEMANTIC_KINEMATICS_BASE_URL=http://192.168.137.1:1234/v1
export SEMANTIC_KINEMATICS_MODEL_NAME=text-embedding-embeddinggemma-300m
```

### Usage

```bash
# List available behavioral concepts
python -m semantic_forge --list-concepts

# Show details for a specific concept
python -m semantic_forge --concept temporal_trust

# Run as MCP server (stdio transport)
python -m semantic_forge --server
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Tool Reference](docs/TOOL_REFERENCE.md) | MCP tool specifications and usage |
| [Workflow Guide](docs/WORKFLOW_GUIDE.md) | Step-by-step workflows for common tasks |
| [Integration Guide](docs/INTEGRATION_GUIDE.md) | Integrating with external tools |

---

## Development

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check .
```

---

## Name Note

"semantic-forge" over "abliteration" or "eraser" because the methodology is **additive, not subtractive**. The goal is to forge stronger behavioral pathways, not to try to obscure bad ones.

The forge metaphor also connects to the [whetstone](https://github.com/shanevcantwell/whetstone-method): shaping through applied pressure and friction, not through removal.

---

## License

MIT License
