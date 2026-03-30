"""Utility functions for semantic-forge."""

import random
from typing import TypeVar

T = TypeVar("T")


def sample_with_min_distance(items: list[T], n: int, min_distance: float = 0.1) -> list[T]:
    """
    Sample items ensuring minimum distance between selected items.

    Args:
        items: List of items to sample from
        n: Number of items to sample
        min_distance: Minimum distance between selected items

    Returns:
        List of sampled items
    """
    if n >= len(items):
        return items.copy()

    # Simple implementation: shuffle and take first n
    # For more sophisticated distance-based sampling, would need
    # a distance function for the item type
    shuffled = items.copy()
    random.shuffle(shuffled)
    return shuffled[:n]


def chunk_text(text: str, max_length: int = 1000) -> list[str]:
    """
    Chunk text into smaller pieces.

    Args:
        text: The text to chunk
        max_length: Maximum length of each chunk

    Returns:
        List of text chunks
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    words = text.split()
    current_chunk = []

    for word in words:
        if len(" ".join(current_chunk + [word])) <= max_length:
            current_chunk.append(word)
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def merge_scores(scores: list[dict]) -> dict:
    """
    Merge multiple score dictionaries into a summary.

    Args:
        scores: List of score dictionaries

    Returns:
        Merged score dictionary with aggregated values
    """
    if not scores:
        return {}

    # Count occurrences of each threat level
    threat_counts = {}
    manipulation_sum = 0.0
    cleanliness_sum = 0.0

    for score in scores:
        threat = score.get("threat_level", "Unknown")
        threat_counts[threat] = threat_counts.get(threat, 0) + 1
        manipulation_sum += score.get("manipulation_score", 0)
        cleanliness_sum += score.get("structural_cleanliness", 0)

    n = len(scores)
    return {
        "total": n,
        "threat_distribution": threat_counts,
        "mean_manipulation_score": manipulation_sum / n,
        "mean_cleanliness": cleanliness_sum / n,
    }
