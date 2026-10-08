"""Tests for SymptomRAGAnalyzer."""

import pytest
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from mediagent_lite.agents.rag_analyzer import SymptomRAGAnalyzer
from mediagent_lite.schemas.hypothesis import HypothesisList

FAKE_RAG_RESPONSE = """
{
    "hypotheses": [
        {
            "diagnosis": "Acute Myocardial Infarction",
            "rank": 1,
            "reasoning": "Matches symptoms perfectly.",
            "supporting_chunks": [
                {
                    "chunk_id": "doc123",
                    "text_span": "chest pain indicates MI",
                    "similarity_score": 0.85
                }
            ]
        },
        {
            "diagnosis": "Hallucinated Disease",
            "rank": 2,
            "reasoning": "I made this up.",
            "supporting_chunks": []
        }
    ]
}
"""

@patch("mediagent_lite.rag.retriever.QdrantClient")
@patch("mediagent_lite.rag.retriever.ClinicalRetriever.retrieve")
@patch("mediagent_lite.llm_factory.FakeLLM.invoke")
def test_rag_analyzer_process(mock_invoke, mock_retrieve, mock_qdrant, sample_structured_case):
    from langchain_core.messages import AIMessage
    mock_invoke.return_value = AIMessage(content=FAKE_RAG_RESPONSE)
    
    # Mock retrieval results
    mock_doc = Document(page_content="chest pain indicates MI", metadata={"qdrant_id": "doc123"})
    mock_retrieve.return_value = [(mock_doc, 0.85)]
    
    agent = SymptomRAGAnalyzer()
    
    # Process
    result = agent.process(sample_structured_case, config_flags={})
    
    assert isinstance(result, HypothesisList)
    
    # Check filtering logic: the hallucinated disease (no chunks) should be dropped
    assert len(result.hypotheses) == 1
    assert result.hypotheses[0].diagnosis == "Acute Myocardial Infarction"
    assert result.hypotheses[0].supporting_chunks[0].chunk_id == "doc123"
    
    # Check query building
    mock_retrieve.assert_called_once()
    query_used = mock_retrieve.call_args[0][0]
    assert "chest pain" in query_used
    assert "left chest" in query_used

@patch("mediagent_lite.rag.retriever.QdrantClient")
@patch("mediagent_lite.rag.retriever.ClinicalRetriever.retrieve")
def test_rag_analyzer_empty_retrieval(mock_retrieve, mock_qdrant, sample_structured_case):
    """Test behavior when Qdrant returns nothing."""
    mock_retrieve.return_value = []  # Empty results
    
    agent = SymptomRAGAnalyzer()
    
    # Process should short-circuit and not call LLM
    result = agent.process(sample_structured_case, config_flags={})
    
    assert len(result.hypotheses) == 1
    assert "Retrieval Failed" in result.hypotheses[0].diagnosis
    assert result.hypotheses[0].supporting_chunks[0].chunk_id == "none"
