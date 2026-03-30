"""Tests for the behavioral concepts library."""

import pytest

from semantic_forge.concepts import (
    CONCEPT_LIBRARY,
    get_concept_by_id,
    get_all_concepts,
    TEMPORAL_TRUST,
    UNCERTAINTY_ACKNOWLEDGMENT,
)


class TestConceptLibrary:
    """Test cases for the behavioral concepts library."""

    def test_library_has_expected_number_of_concepts(self):
        """Verify we have all the starter concepts."""
        assert len(CONCEPT_LIBRARY) == 8

    def test_get_concept_by_id_returns_correct_concept(self):
        """Test looking up a concept by its ID."""
        concept = get_concept_by_id("temporal_trust")
        assert concept is not None
        assert concept.id == "temporal_trust"
        assert "post-cutoff" in concept.core_statement.lower()

    def test_get_concept_by_id_returns_none_for_invalid_id(self):
        """Test that invalid IDs return None."""
        assert get_concept_by_id("nonexistent") is None

    def test_get_all_concepts_returns_copy(self):
        """Test that get_all_concepts returns a copy, not the original."""
        concepts = get_all_concepts()
        original_len = len(CONCEPT_LIBRARY)
        concepts.append(TEMPORAL_TRUST)
        assert len(get_all_concepts()) == original_len

    def test_temporal_trust_concept(self):
        """Verify the temporal trust concept is correctly defined."""
        assert TEMPORAL_TRUST.id == "temporal_trust"
        assert TEMPORAL_TRUST.name == "Temporal Trust"
        assert "post-cutoff" in TEMPORAL_TRUST.core_statement.lower()
        assert len(TEMPORAL_TRUST.addresses) == 2

    def test_uncertainty_acknowledgment_concept(self):
        """Verify the uncertainty acknowledgment concept."""
        assert UNCERTAINTY_ACKNOWLEDGMENT.id == "uncertainty_acknowledgment"
        assert "I don't know" in UNCERTAINTY_ACKNOWLEDGMENT.core_statement

    def test_all_concepts_have_required_fields(self):
        """Verify all concepts have the required fields."""
        for concept in CONCEPT_LIBRARY:
            assert hasattr(concept, "id")
            assert hasattr(concept, "name")
            assert hasattr(concept, "core_statement")
            assert hasattr(concept, "addresses")
            assert hasattr(concept, "notes")
            assert isinstance(concept.id, str)
            assert isinstance(concept.name, str)
            assert isinstance(concept.core_statement, str)
            assert isinstance(concept.addresses, list)
            assert len(concept.addresses) > 0
