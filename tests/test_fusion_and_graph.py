"""Tests for Fusion Agent and overall Graph logic."""

from unittest.mock import patch

from mediagent_lite.agents.fusion import FusionAgent
from mediagent_lite.graph.orchestrator import evaluate_confidence
from mediagent_lite.schemas.evidence import EvidenceBundle, EvidenceRecord, PublicationType
from mediagent_lite.schemas.hypothesis import DiagnosisHypothesis, HypothesisList, SupportingChunk
from mediagent_lite.schemas.report import DiagnosisReport, FusedReport

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


def test_deterministic_confidence_math():
    """Proves the exact paper formula and overrides LLM values.
    
    C = 0.35 * S_rag + 0.40 * S_ev + 0.25 * S_concordance
    """
    agent = FusionAgent()
    # Force specific weights for the test to avoid config.yaml changes breaking it
    agent.settings._config["confidence"] = {"w1": 0.35, "w2": 0.40, "w3": 0.25, "tau": 0.65}
    
    # LLM hallucinated confidence = 0.99 (should be overwritten)
    # LLM hallucinated concordance = 0.1 (should be overwritten)
    report = DiagnosisReport(
        diagnosis="TestDx",
        rank=1,
        confidence=0.99,
        concordance=0.1,
        retrieval_similarity=0.1,
        weighted_evidence_score=0.1,
    )
    
    # RAG Hypothesis: max supporting chunk similarity = 0.8
    rag_hyp = {
        "diagnosis": "TestDx",
        "supporting_chunks": [{"similarity_score": 0.5}, {"similarity_score": 0.8}]
    }
    
    # PubMed Evidence: RCT (0.8) + Case Report (0.3) = 1.1 -> clamped to 1.0
    ev_bundles = [
        EvidenceBundle(
            records=[
                EvidenceRecord(hypothesis="TestDx", pmid="1", title="A", classified_type=PublicationType.RCT, evidence_weight=0.8),
                EvidenceRecord(hypothesis="TestDx", pmid="2", title="B", classified_type=PublicationType.CASE_REPORT, evidence_weight=0.3),
            ],
            weighted_scores={"TestDx": 1.1} # This is pre-calculated by Evidence Scanner
        )
    ]
    
    final_conf = agent._calculate_confidence(report, rag_hyp, ev_bundles)
    
    # Hand-calculation:
    # S_rag = 0.8
    # S_ev = min(1.0, 1.1) = 1.0
    # S_concordance: both S_rag and S_ev > 0.3 -> 1.0
    # C = 0.35 * 0.8 + 0.40 * 1.0 + 0.25 * 1.0 = 0.28 + 0.40 + 0.25 = 0.93
    
    import math
    assert math.isclose(final_conf, 0.93, rel_tol=1e-5)
    # Prove the LLM's hallucinated values were overwritten
    assert math.isclose(report.confidence, 0.93, rel_tol=1e-5)
    assert report.concordance == 1.0
    assert report.retrieval_similarity == 0.8
    assert report.weighted_evidence_score == 1.0
