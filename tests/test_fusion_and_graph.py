"""Tests for Fusion Agent and overall Graph logic."""

import pytest
from unittest.mock import patch

from mediagent_lite.agents.fusion import FusionAgent
from mediagent_lite.schemas.report import FusedReport, DiagnosisReport, ConflictFlag
from mediagent_lite.schemas.hypothesis import HypothesisList, DiagnosisHypothesis, SupportingChunk
from mediagent_lite.schemas.evidence import EvidenceBundle, EvidenceRecord, PublicationType
from mediagent_lite.graph.orchestrator import evaluate_confidence

FAKE_FUSION_RESPONSE = """
{
    "diagnoses": [
        {
            "diagnosis": "Test Disease",
            "rank": 1,
            "confidence": 0.9,
            "retrieval_similarity": 0.8,
            "weighted_evidence_score": 0.8,
            "concordance": 1.0,
            "explanation": "Fits perfectly.",
            "has_conflict": false
        }
    ],
    "conflicts": [],
    "top_confidence": 0.9,
    "iteration": 0
}
"""

@patch("mediagent_lite.llm_factory.FakeLLM.invoke")
def test_fusion_process_and_math(mock_invoke, sample_structured_case):
    from langchain_core.messages import AIMessage
    mock_invoke.return_value = AIMessage(content=FAKE_FUSION_RESPONSE)
    
    # Setup inputs
    chunk = SupportingChunk(chunk_id="1", text_span="test", similarity_score=0.8) # s_rag = 0.8
    hyp = DiagnosisHypothesis(diagnosis="Test Disease", rank=1, reasoning="test", supporting_chunks=[chunk])
    hyp_list = HypothesisList(hypotheses=[hyp])
    
    ev_record = EvidenceRecord(hypothesis="Test Disease", pmid="123", title="test", classified_type=PublicationType.RCT, evidence_weight=0.8) # s_ev = 0.8
    bundle = EvidenceBundle(records=[ev_record], weighted_scores={"Test Disease": 0.8})
    
    agent = FusionAgent()
    # Mock settings to avoid file reads
    agent.settings._config["confidence"] = {"w1": 0.4, "w2": 0.4, "w3": 0.2, "tau": 0.65}
    
    report = agent.process(sample_structured_case, hyp_list, [bundle])
    
    # Math: (0.4 * 0.8) + (0.4 * 0.8) + (0.2 * 1.0 [concordance])
    # 0.32 + 0.32 + 0.2 = 0.84
    
    assert report.diagnoses[0].diagnosis == "Test Disease"
    assert abs(report.diagnoses[0].confidence - 0.84) < 0.01

def test_evaluate_confidence_edge():
    # Test END condition (high confidence)
    report = FusedReport(diagnoses=[DiagnosisReport(diagnosis="A", rank=1, confidence=0.9, retrieval_similarity=0.9, weighted_evidence_score=0.9, concordance=1.0)], iteration=0, top_confidence=0.9)
    state_end = {"final_report": report, "iteration": 0}
    assert evaluate_confidence(state_end) == "end"
    
    # Test OPTIMIZE condition (low confidence)
    report_low = FusedReport(diagnoses=[DiagnosisReport(diagnosis="A", rank=1, confidence=0.4, retrieval_similarity=0.4, weighted_evidence_score=0.4, concordance=0.5)], iteration=0, top_confidence=0.4)
    state_opt = {"final_report": report_low, "iteration": 0}
    assert evaluate_confidence(state_opt) == "optimize"
    
    # Test END condition (max iterations reached despite low confidence)
    state_max = {"final_report": report_low, "iteration": 3}
    assert evaluate_confidence(state_max) == "end"
