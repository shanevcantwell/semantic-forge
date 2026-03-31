"""LLM client abstractions for semantic-forge.

Supports multiple inference backends including Ollama and vLLM.
"""

import json
import httpx
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from semantic_forge.config import InferenceBackend


class LLMClient:
    """Base class for LLM clients."""

    def __init__(self, backend: InferenceBackend):
        self.backend = backend

    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text from the model."""
        raise NotImplementedError

    async def generate_structured(
        self,
        prompt: str,
        response_format: type,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """Generate structured output from the model."""
        raise NotImplementedError


class OllamaClient(LLMClient):
    """Client for Ollama inference server."""

    def __init__(self, backend: InferenceBackend):
        super().__init__(backend)
        self.endpoint = backend.endpoint.rstrip("/")

    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text using Ollama API."""
        url = f"{self.endpoint}/api/generate"
        payload = {
            "model": self.backend.model,
            "prompt": prompt,
            "temperature": temperature or self.backend.temperature,
            "stream": False,
        }

        if max_tokens:
            payload["options"] = {"num_predict": max_tokens}

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")

    async def generate_structured(
        self,
        prompt: str,
        response_format: type,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """Generate structured output using Ollama's JSON mode."""
        url = f"{self.endpoint}/api/generate"
        payload = {
            "model": self.backend.model,
            "prompt": f"{prompt}\n\nReturn your response as valid JSON only, without any markdown formatting or explanation.",
            "temperature": temperature or self.backend.temperature,
            "stream": False,
            "format": "json",
        }

        if max_tokens:
            payload["options"] = {"num_predict": max_tokens}

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            json_str = result.get("response", "")
            return json.loads(json_str)


class VLLMClient(LLMClient):
    """Client for vLLM inference server."""

    def __init__(self, backend: InferenceBackend):
        super().__init__(backend)
        self.endpoint = backend.endpoint.rstrip("/")

    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text using vLLM API."""
        url = f"{self.endpoint}/v1/chat/completions"

        payload = {
            "model": self.backend.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature or self.backend.temperature,
            "max_tokens": max_tokens or self.backend.max_tokens,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]

    async def generate_structured(
        self,
        prompt: str,
        response_format: type,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """Generate structured output using vLLM."""
        url = f"{self.endpoint}/v1/chat/completions"

        # Create a JSON schema for the response
        payload = {
            "model": self.backend.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that returns valid JSON. Do not use markdown formatting.",
                },
                {"role": "user", "content": f"{prompt}\n\nReturn your response as valid JSON only."},
            ],
            "temperature": temperature or self.backend.temperature,
            "max_tokens": max_tokens or self.backend.max_tokens,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            json_str = result["choices"][0]["message"]["content"]
            # Try to parse JSON, handling potential markdown formatting
            json_str = _extract_json(json_str)
            return json.loads(json_str)


def _extract_json(text: str) -> str:
    """Extract JSON from text that may contain markdown formatting."""
    # Try to find JSON in backticks
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()

    # Try to find JSON object directly
    if "{" in text:
        # Find the first { and last }
        start = text.find("{")
        # Simple heuristic - find the last }
        end = text.rfind("}")
        if start < end:
            return text[start : end + 1]

    return text


def create_client(backend: InferenceBackend) -> LLMClient:
    """Create an LLM client based on backend type."""
    if backend.type == "ollama":
        return OllamaClient(backend)
    elif backend.type == "vllm":
        return VLLMClient(backend)
    else:
        raise ValueError(f"Unknown backend type: {backend.type}")


async def generate_text(
    backend: InferenceBackend,
    prompt: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """Convenience function to generate text with the specified backend."""
    client = create_client(backend)
    return await client.generate(prompt, temperature, max_tokens)


async def generate_structured_output(
    backend: InferenceBackend,
    prompt: str,
    response_format: type,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Any:
    """Convenience function to generate structured output with the specified backend."""
    client = create_client(backend)
    return await client.generate_structured(prompt, response_format, temperature, max_tokens)
