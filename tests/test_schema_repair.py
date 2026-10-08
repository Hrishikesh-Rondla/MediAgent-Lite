"""Tests for schema repair / clamp-before-validate behavior.

These tests verify that when a local LLM hallucinates out-of-range or
wrong-type values for probability fields, the Pydantic clamp validators
coerce them to [0, 1] rather than crashing the application.
"""

import pytest

from mediagent_lite.schemas.report import DiagnosisReport


class TestClampValidators:

    def _make_report(self, **overrides):
        defaults = {
            "diagnosis": "Test Dx",
            "rank": 1,
            "confidence": 0.5,
            "retrieval_similarity": 0.5,
            "weighted_evidence_score": 0.5,
            "concordance": 0.5,
        }
        defaults.update(overrides)
        return DiagnosisReport(**defaults)

    def test_concordance_integer_3_clamped_to_1(self):
        """LLM returning concordance=3 (rank-style) must not crash."""
        r = self._make_report(concordance=3)
        assert r.concordance == 1.0

    def test_concordance_integer_2_clamped_to_1(self):
        r = self._make_report(concordance=2)
        assert r.concordance == 1.0

    def test_concordance_negative_clamped_to_0(self):
        r = self._make_report(concordance=-0.5)
        assert r.concordance == 0.0

    def test_confidence_above_1_clamped(self):
        r = self._make_report(confidence=1.5)
        assert r.confidence == 1.0

    def test_confidence_below_0_clamped(self):
        r = self._make_report(confidence=-0.3)
        assert r.confidence == 0.0

    def test_retrieval_similarity_above_1_clamped(self):
        r = self._make_report(retrieval_similarity=2.7)
        assert r.retrieval_similarity == 1.0

    def test_weighted_evidence_score_string_coerced_to_zero(self):
        """LLM returning 'strong' for a float field must return 0.0, not crash."""
        r = self._make_report(weighted_evidence_score="strong")
        assert r.weighted_evidence_score == 0.0

    def test_valid_values_pass_through_unchanged(self):
        r = self._make_report(confidence=0.72, retrieval_similarity=0.55,
                               weighted_evidence_score=0.40, concordance=1.0)
        assert r.confidence == pytest.approx(0.72)
        assert r.retrieval_similarity == pytest.approx(0.55)
        assert r.weighted_evidence_score == pytest.approx(0.40)
        assert r.concordance == pytest.approx(1.0)

    def test_zero_values_valid(self):
        """A diagnosis with no RAG or evidence support must be representable."""
        r = self._make_report(confidence=0.0, retrieval_similarity=0.0,
                               weighted_evidence_score=0.0, concordance=0.0)
        assert r.confidence == 0.0
