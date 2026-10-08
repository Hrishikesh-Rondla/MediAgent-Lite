"""Tests for EvidenceScanner.

FakeLLM is used via MEDIAGENT_TEST_MODE=true (set in conftest.py).
PubMedTool._run and the extract_chain are mocked to avoid real network calls.
"""

from unittest.mock import MagicMock, patch

from mediagent_lite.agents.evidence_scanner import EvidenceScanner
from mediagent_lite.schemas.evidence import EvidenceBundle, EvidenceRecord, PublicationType
from mediagent_lite.schemas.hypothesis import DiagnosisHypothesis


def _make_bundle():
    return EvidenceBundle(
        records=[
            EvidenceRecord(
                hypothesis="Acute Myocardial Infarction",
                pmid="123",
                title="Test Title",
                evidence_weight=0.8,
                classified_type=PublicationType.RCT,
            )
        ],
        weighted_scores={"Acute Myocardial Infarction": 0.8},
    )


@patch("mediagent_lite.tools.pubmed_tool.PubMedTool._run")
def test_evidence_scanner_process(mock_tool_run, sample_structured_case):
    """EvidenceScanner returns a valid EvidenceBundle with evidence_weight
    applied from the config-driven hierarchy (not from the LLM)."""
    mock_tool_run.return_value = "PMID: 123\nTitle: Test\nAbstract: aspirin improves survival"

    hypothesis = DiagnosisHypothesis(
        diagnosis="Acute Myocardial Infarction", rank=1, reasoning="", supporting_chunks=[]
    )
    agent = EvidenceScanner()

    # Mock the extraction chain directly so we bypass LLM structured-output parsing
    agent.extract_chain = MagicMock()
    agent.extract_chain.invoke.return_value = _make_bundle()

    bundle = agent.process(sample_structured_case, hypothesis)

    assert isinstance(bundle, EvidenceBundle)
    assert "Acute Myocardial Infarction" in bundle.weighted_scores
    assert len(bundle.records) == 1
    # Weight must come from config hierarchy (rct=0.8), not LLM
    assert bundle.records[0].evidence_weight == 0.8

    # Query must use programmatic construction (diagnosis + symptom names)
    call_kwargs = mock_tool_run.call_args[1]
    query = call_kwargs["query"]
    assert "Acute Myocardial Infarction" in query
    assert "chest pain" in query


@patch("mediagent_lite.tools.pubmed_tool.PubMedTool._run")
def test_evidence_scanner_no_results(mock_tool_run, sample_structured_case):
    """When PubMed returns no results, EvidenceScanner returns an empty bundle without crashing."""
    mock_tool_run.return_value = "No results found for query: test"

    hypothesis = DiagnosisHypothesis(
        diagnosis="Rare Disease", rank=1, reasoning="", supporting_chunks=[]
    )
    agent = EvidenceScanner()
    bundle = agent.process(sample_structured_case, hypothesis)

    assert isinstance(bundle, EvidenceBundle)
    assert len(bundle.records) == 0
