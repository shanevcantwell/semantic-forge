"""Configuration management for semantic-forge.

Supports loading configuration from JSON files and environment variables.
"""

import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class InferenceBackend:
    """Configuration for an inference backend.

    type is one of: ollama, vllm, openai (generic OpenAI-compatible endpoint).
    For the openai type, `endpoint` is a base URL (with or without trailing /v1)
    and `model` is the model id served at that endpoint. Both are overridable
    per role via environment variables (see _load_from_env).
    """
    type: str = "ollama"  # ollama, vllm, or openai
    model: str = ""
    endpoint: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 2048
    api_key: Optional[str] = None  # Bearer key; openai backend falls back to env/placeholder


@dataclass
class SemanticKinematicsConfig:
    """Configuration for semantic-kinematics-mcp integration.

    Supports runtime backend switching via MCP model_load tool.
    Default backend: lmstudio with embeddinggemma:300m for low VRAM usage.
    """
    endpoint: Optional[str] = None
    # Backend switching via MCP interface
    backend: Optional[str] = "lmstudio"  # "lmstudio", "nv_embed", "sentence_transformers"
    base_url: Optional[str] = "http://localhost:1234/v1"  # LM Studio API URL
    model_name: Optional[str] = "text-embedding-embeddinggemma-300m"  # Model for API backends
    # Legacy fields for NV-Embed backend (kept for backwards compatibility)
    model: str = "nvidia/NV-Embed-v2"
    device: str = "cuda"


@dataclass
class PromptPrixConfig:
    """Configuration for prompt-prix integration."""
    endpoint: Optional[str] = None


@dataclass
class SemanticForgeConfig:
    """Main configuration for semantic-forge."""
    inference: dict = field(default_factory=lambda: {
        "rephraser": InferenceBackend(type="ollama", model="lfm2"),
        "target": InferenceBackend(type="ollama", model="qwen:27b"),
        "judge": InferenceBackend(type="ollama", model="llama3:8b"),
    })
    semantic_kinematics: SemanticKinematicsConfig = field(default_factory=SemanticKinematicsConfig)
    prompt_prix: PromptPrixConfig = field(default_factory=PromptPrixConfig)


# Global configuration instance
_config: Optional[SemanticForgeConfig] = None


def get_config() -> SemanticForgeConfig:
    """Get the global configuration, loading it if necessary."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: SemanticForgeConfig) -> None:
    """Set the global configuration."""
    global _config
    _config = config


def load_config(path: Optional[str] = None) -> SemanticForgeConfig:
    """
    Load configuration from a file or environment variables.

    Args:
        path: Optional path to config file. If None, tries:
              1. semantic_forge_config.json in current directory
              2. ~/.semantic-forge/config.json

    Returns:
        SemanticForgeConfig instance
    """
    config = SemanticForgeConfig()

    # Try to load from file
    config_path = _find_config_path(path)
    if config_path and os.path.exists(config_path):
        config = _load_from_file(config_path, config)

    # Override with environment variables
    config = _load_from_env(config)

    return config


def _find_config_path(path: Optional[str] = None) -> Optional[str]:
    """Find the configuration file path."""
    if path:
        return path

    # Try current directory
    local_path = Path("semantic_forge_config.json")
    if local_path.exists():
        return str(local_path)

    # Try home directory
    home_path = Path.home() / ".semantic-forge" / "config.json"
    if home_path.exists():
        return str(home_path)

    return None


def _load_from_file(path: str, base_config: SemanticForgeConfig) -> SemanticForgeConfig:
    """Load configuration from a JSON file."""
    with open(path) as f:
        data = json.load(f)

    config = base_config

    if "inference" in data:
        if "rephraser" in data["inference"]:
            config.inference["rephraser"] = _dict_to_backend(data["inference"]["rephraser"])
        if "target" in data["inference"]:
            config.inference["target"] = _dict_to_backend(data["inference"]["target"])
        if "judge" in data["inference"]:
            config.inference["judge"] = _dict_to_backend(data["inference"]["judge"])

    if "semantic_kinematics" in data:
        sk_data = data["semantic_kinematics"]
        config.semantic_kinematics = SemanticKinematicsConfig(
            endpoint=sk_data.get("endpoint"),
            backend=sk_data.get("backend"),
            base_url=sk_data.get("base_url"),
            model_name=sk_data.get("model_name"),
            model=sk_data.get("model", "nvidia/NV-Embed-v2"),
            device=sk_data.get("device", "cuda"),
        )

    if "prompt_prix" in data:
        pp_data = data.get("prompt_prix", {})
        config.prompt_prix = PromptPrixConfig(endpoint=pp_data.get("endpoint"))

    return config


def _dict_to_backend(data: dict) -> InferenceBackend:
    """Convert a dictionary to an InferenceBackend."""
    return InferenceBackend(
        type=data.get("type", "ollama"),
        model=data.get("model", ""),
        endpoint=data.get("endpoint", "http://localhost:11434"),
        temperature=data.get("temperature", 0.7),
        max_tokens=data.get("max_tokens", 2048),
        api_key=data.get("api_key"),
    )


# Environment variable names used to override per-role inference settings.
# SEMANTIC_FORGE_<ROLE>_BASE_URL / _MODEL with ROLE in {REPHRASER, TARGET, JUDGE}.
_INFER_ROLES = ("rephraser", "target", "judge")


def _load_from_env(base_config: SemanticForgeConfig) -> SemanticForgeConfig:
    """Load configuration from environment variables."""
    config = base_config

    # Ollama endpoints
    if os.environ.get("OLLAMA_ENDPOINT"):
        for key in config.inference:
            if isinstance(config.inference[key], InferenceBackend):
                config.inference[key].endpoint = os.environ["OLLAMA_ENDPOINT"]

    # Per-role overrides (keep endpoints + model ids pluggable without editing JSON)
    for role in _INFER_ROLES:
        backend = config.inference.get(role)
        if not isinstance(backend, InferenceBackend):
            continue
        base_url = os.environ.get(f"SEMANTIC_FORGE_{role.upper()}_BASE_URL")
        model = os.environ.get(f"SEMANTIC_FORGE_{role.upper()}_MODEL")
        api_key = os.environ.get("SEMANTIC_FORGE_API_KEY")
        if base_url:
            backend.endpoint = base_url
        if model:
            backend.model = model
        if api_key and not backend.api_key:
            backend.api_key = api_key

    # Semantic kinematics backend config
    if os.environ.get("SEMANTIC_KINEMATICS_ENDPOINT"):
        config.semantic_kinematics.endpoint = os.environ["SEMANTIC_KINEMATICS_ENDPOINT"]
    if os.environ.get("SEMANTIC_KINEMATICS_BACKEND"):
        config.semantic_kinematics.backend = os.environ["SEMANTIC_KINEMATICS_BACKEND"]
    if os.environ.get("SEMANTIC_KINEMATICS_BASE_URL"):
        config.semantic_kinematics.base_url = os.environ["SEMANTIC_KINEMATICS_BASE_URL"]
    if os.environ.get("SEMANTIC_KINEMATICS_MODEL_NAME"):
        config.semantic_kinematics.model_name = os.environ["SEMANTIC_KINEMATICS_MODEL_NAME"]
    # Legacy env vars (kept for backwards compatibility)
    if os.environ.get("SEMANTIC_KINEMATICS_MODEL"):
        config.semantic_kinematics.model = os.environ["SEMANTIC_KINEMATICS_MODEL"]
    if os.environ.get("SEMANTIC_KINEMATICS_DEVICE"):
        config.semantic_kinematics.device = os.environ["SEMANTIC_KINEMATICS_DEVICE"]

    # Prompt prix
    if os.environ.get("PROMPT_PRIX_ENDPOINT"):
        config.prompt_prix.endpoint = os.environ["PROMPT_PRIX_ENDPOINT"]

    return config


def get_rephraser_config() -> InferenceBackend:
    """Get the rephrasing backend configuration."""
    return get_config().inference["rephraser"]


def get_target_config() -> InferenceBackend:
    """Get the target model backend configuration."""
    return get_config().inference["target"]


def get_judge_config() -> InferenceBackend:
    """Get the CogSec judge backend configuration."""
    return get_config().inference["judge"]


def get_semantic_kinematics_endpoint() -> Optional[str]:
    """Get the semantic-kinematics-mcp endpoint."""
    return get_config().semantic_kinematics.endpoint


def get_semantic_kinematics_config() -> SemanticKinematicsConfig:
    """Get the full semantic-kinematics configuration.

    TODO: Use this to pass backend config to SemanticKinematicsClient
    """
    return get_config().semantic_kinematics


def get_prompt_prix_endpoint() -> Optional[str]:
    """Get the prompt-prix endpoint."""
    return get_config().prompt_prix.endpoint
