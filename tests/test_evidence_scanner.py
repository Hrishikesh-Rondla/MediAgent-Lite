"""Tests for EvidenceScanner."""

from unittest.mock import patch, MagicMock

from mediagent_lite.agents.evidence_scanner import EvidenceScanner
from mediagent_lite.schemas.hypothesis import DiagnosisHypothesis
from mediagent_lite.schemas.evidence import EvidenceBundle

FAKE_EXTRACT_RESPONSE = """
{
    "records": [
        {
            "hypothesis": "Acute Myocardial Infarction",
            "pmid": "123",
            "title": "Test Title",
            "evidence_weight": 0.8,
            "classified_type": "rct"
        }
    ],
    "weighted_scores": {
        "Acute Myocardial Infarction": 0.8
    }
}
"""

@patch("mediagent_lite.tools.pubmed_tool.PubMedTool._run")
@patch("mediagent_lite.llm_factory.FakeLLM.invoke")
def test_evidence_scanner_process(mock_invoke, mock_tool_run, sample_structured_case):
    # Setup LLM mocks
    # First call is query generation, second is extraction
    from langchain_core.messages import AIMessage
    mock_invoke.side_effect = [
        AIMessage(content="myocardial infarction AND aspirin"),
        AIMessage(content=FAKE_EXTRACT_RESPONSE)
    ]
    
    # Setup Tool mock
    mock_tool_run.return_value = "PMID: 123\nTitle: Test\nAbstract: aspirin improves survival"
    
    hypothesis = DiagnosisHypothesis(diagnosis="Acute Myocardial Infarction", rank=1, reasoning="", supporting_chunks=[])
    agent = EvidenceScanner()
    
    bundle = agent.process(sample_structured_case, hypothesis)
    
    assert isinstance(bundle, EvidenceBundle)
    assert "Acute Myocardial Infarction" in bundle.weighted_scores
    assert len(bundle.records) == 1
    assert bundle.records[0].evidence_weight == 0.8
    
    # Verify tool was called with the generated query
    mock_tool_run.assert_called_once()
    assert "myocardial infarction" in mock_tool_run.call_args[1]["query"]
