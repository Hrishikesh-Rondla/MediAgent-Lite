"""Tests for all Pydantic v2 schema validation.

These tests have zero external dependencies (no LLM, no network, no vector DB).
They verify that schema validation catches bad data at the boundary.
"""

import pytest
from pydantic import ValidationError

from mediagent_lite.schemas.clinical import StructuredCase, Symptom, Lab, Imaging
from mediagent_lite.schemas.hypothesis import (
    DiagnosisHypothesis,
    HypothesisList,
    SupportingChunk,
)
from mediagent_lite.schemas.evidence import (
    EvidenceBundle,
    EvidenceRecord,
    PublicationType,
    PubMedAbstract,
)
from mediagent_lite.schemas.report import (
    ConflictFlag,
    DiagnosisReport,
    FusedReport,
)


# ── StructuredCase ────────────────────────────────────────────────────────────

class TestStructuredCase:
    def test_minimal_valid_with_complaint(self):
        case = StructuredCase(chief_complaint="chest pain")
        assert case.chief_complaint == "chest pain"
        assert case.symptoms == []
        assert case.labs == []

    def test_minimal_valid_with_symptom(self):
        case = StructuredCase(symptoms=[Symptom(name="dyspnea")])
        assert len(case.symptoms) == 1

    def test_rejects_empty_case(self):
        """Crucial: empty case must fail at the schema level."""
        with pytest.raises(ValidationError, match="chief_complaint or one symptom"):
            StructuredCase()

    def test_full_case(self):
        case = StructuredCase(
            age=65,
            sex="F",
            chief_complaint="shortness of breath",
            symptoms=[
                Symptom(name="dyspnea", severity="severe", onset="sudden"),
                Symptom(name="leg swelling", anatomical_region="bilateral lower extremities"),
            ],
            labs=[Lab(name="BNP", value="850 pg/mL", interpretation="high")],
            imaging=[Imaging(modality="Chest X-ray", finding="bilateral pleural effusions")],
            vitals={"BP": "155/95", "HR": "105", "O2Sat": "88%"},
            past_medical_history=["Hypertension", "Type 2 Diabetes"],
        )
        assert case.age == 65
        assert len(case.symptoms) == 2
        assert case.labs[0].interpretation == "high"

    def test_icd10_defaults_empty(self):
        case = StructuredCase(chief_complaint="fever")
        assert case.icd10_codes == []


# ── Hypothesis Schemas ────────────────────────────────────────────────────────

class TestHypothesis:
    def test_supporting_chunk_score_bounds(self):
        chunk = SupportingChunk(
            chunk_id="c1",
            text_span="relevant excerpt",
            similarity_score=0.75,
        )
        assert chunk.similarity_score == 0.75

    def test_invalid_similarity_score_above_1(self):
        with pytest.raises(ValidationError):
            SupportingChunk(chunk_id="c1", text_span="x", similarity_score=1.5)

    def test_invalid_similarity_score_below_0(self):
        with pytest.raises(ValidationError):
            SupportingChunk(chunk_id="c1", text_span="x", similarity_score=-0.1)

    def test_hypothesis_rank_must_be_positive(self):
        with pytest.raises(ValidationError):
            DiagnosisHypothesis(
                diagnosis="Test",
                rank=0,
                reasoning="test",
            )

    def test_hypothesis_list_requires_at_least_one(self):
        with pytest.raises(ValidationError):
            HypothesisList(hypotheses=[])

    def test_valid_hypothesis_list(self, sample_hypothesis_list):
        assert len(sample_hypothesis_list.hypotheses) == 1
        assert sample_hypothesis_list.hypotheses[0].rank == 1


# ── Evidence Schemas ──────────────────────────────────────────────────────────

class TestEvidence:
    def test_publication_type_enum_values(self):
        assert PublicationType.GUIDELINE.value == "guideline"
        assert PublicationType.RCT.value == "rct"

    def test_evidence_record_weight_bounds(self):
        """Evidence weight must stay in [0, 1]."""
        with pytest.raises(ValidationError):
            EvidenceRecord(
                hypothesis="MI",
                pmid="12345",
                title="Test",
                evidence_weight=1.5,  # invalid
                classified_type=PublicationType.RCT,
            )

    def test_valid_evidence_bundle(self):
        bundle = EvidenceBundle(
            records=[
                EvidenceRecord(
                    hypothesis="MI",
                    pmid="12345",
                    title="RCT on aspirin in AMI",
                    evidence_weight=0.8,
                    classified_type=PublicationType.RCT,
                    relevance_summary="Aspirin reduces MI mortality.",
                )
            ],
            weighted_scores={"Acute Myocardial Infarction": 0.72},
        )
        assert bundle.records[0].evidence_weight == 0.8
        assert bundle.weighted_scores["Acute Myocardial Infarction"] == 0.72


# ── Report Schemas ────────────────────────────────────────────────────────────

class TestReport:
    def test_confidence_clamped(self):
        """Floating-point drift above 1.0 should be clamped, not rejected."""
        report = DiagnosisReport(
            diagnosis="MI",
            rank=1,
            confidence=1.0000001,  # floating point drift
            retrieval_similarity=1.0,
            weighted_evidence_score=1.0,
            concordance=1.0,
        )
        assert report.confidence == 1.0

    def test_confidence_clamped_below_zero(self):
        report = DiagnosisReport(
            diagnosis="MI",
            rank=1,
            confidence=-0.001,
            retrieval_similarity=0.0,
            weighted_evidence_score=0.0,
            concordance=0.0,
        )
        assert report.confidence == 0.0

    def test_fused_report_requires_diagnoses(self):
        with pytest.raises(ValidationError):
            FusedReport(diagnoses=[], top_confidence=0.5, iteration=0)

    def test_valid_fused_report(self, sample_hypothesis_list):
        report = FusedReport(
            diagnoses=[
                DiagnosisReport(
                    diagnosis="Acute MI",
                    rank=1,
                    confidence=0.78,
                    retrieval_similarity=0.82,
                    weighted_evidence_score=0.75,
                    concordance=1.0,
                    supporting_chunk_ids=["chunk_001"],
                    pubmed_ids=["12345"],
                )
            ],
            top_confidence=0.78,
            iteration=1,
            summary="Strong evidence for Acute MI.",
        )
        assert report.top_confidence == 0.78
        assert len(report.diagnoses) == 1


# ── Settings ──────────────────────────────────────────────────────────────────

class TestSettings:
    def test_settings_load(self):
        from mediagent_lite.config.settings import get_settings
        settings = get_settings()
        assert settings.top_k == 5
        assert settings.tau == 0.65
        assert settings.max_iter == 3

    def test_settings_confidence_weights_sum_to_one(self):
        from mediagent_lite.config.settings import get_settings
        settings = get_settings()
        w1, w2, w3 = settings.confidence_weights
        assert abs(w1 + w2 + w3 - 1.0) < 1e-9, f"Weights sum to {w1+w2+w3}, not 1.0"

    def test_settings_llm_role_fallback(self):
        from mediagent_lite.config.settings import get_settings
        settings = get_settings()
        # Unknown role falls back to default
        cfg = settings.get_llm_config("nonexistent_role")
        assert cfg["provider"] == "gemini"

    def test_settings_llm_role_override(self):
        from mediagent_lite.config.settings import get_settings
        settings = get_settings()
        cfg = settings.get_llm_config("evidence_scanner")
        assert cfg["provider"] == "groq"
        assert cfg["model"] == "llama-3.3-70b-versatile"


# ── LLM Factory ───────────────────────────────────────────────────────────────

class TestLLMFactory:
    def test_fake_llm_in_test_mode(self):
        """In test mode (MEDIAGENT_TEST_MODE=true), always returns FakeLLM."""
        from mediagent_lite.llm_factory import get_llm, FakeLLM
        llm = get_llm("clarifier")
        assert isinstance(llm, FakeLLM)

    def test_fake_llm_invoke(self):
        from mediagent_lite.llm_factory import FakeLLM
        llm = FakeLLM(responses=['{"key": "value"}'])
        result = llm.invoke("test prompt")
        assert result.content == '{"key": "value"}'

    def test_fake_llm_cycles_responses(self):
        from mediagent_lite.llm_factory import FakeLLM
        llm = FakeLLM(responses=["response_1", "response_2"])
        r1 = llm.invoke("a")
        r2 = llm.invoke("b")
        r3 = llm.invoke("c")  # cycles back
        assert r1.content == "response_1"
        assert r2.content == "response_2"
        assert r3.content == "response_1"
