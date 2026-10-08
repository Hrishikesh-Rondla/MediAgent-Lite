"""Pytest configuration and shared fixtures."""

import os
import pytest

# Force test mode globally — no LLM API calls in tests
os.environ["MEDIAGENT_TEST_MODE"] = "true"


@pytest.fixture
def fake_llm():
    """A FakeLLM that returns a minimal valid JSON response."""
    from mediagent_lite.llm_factory import FakeLLM
    return FakeLLM(responses=["{}"])


@pytest.fixture
def sample_structured_case():
    """A minimal valid StructuredCase for testing."""
    from mediagent_lite.schemas.clinical import StructuredCase, Symptom
    return StructuredCase(
        age=45,
        sex="M",
        chief_complaint="chest pain",
        symptoms=[
            Symptom(
                name="chest pain",
                anatomical_region="left chest",
                duration="2 hours",
                severity="severe",
                onset="sudden",
            )
        ],
        vitals={"BP": "150/95", "HR": "110", "O2Sat": "94%"},
    )


@pytest.fixture
def sample_hypothesis_list():
    """A valid HypothesisList for testing."""
    from mediagent_lite.schemas.hypothesis import HypothesisList, DiagnosisHypothesis, SupportingChunk
    return HypothesisList(
        hypotheses=[
            DiagnosisHypothesis(
                diagnosis="Acute Myocardial Infarction",
                icd10_code="I21.9",
                rank=1,
                reasoning="Sudden severe chest pain with elevated HR in middle-aged male.",
                supporting_chunks=[
                    SupportingChunk(
                        chunk_id="chunk_001",
                        text_span="Sudden onset chest pain is the hallmark of acute MI",
                        similarity_score=0.82,
                    )
                ],
            )
        ]
    )
