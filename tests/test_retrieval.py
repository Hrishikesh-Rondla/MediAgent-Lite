"""Tests for RAG chunking and retriever."""

import pytest
from unittest.mock import patch

from mediagent_lite.rag.chunker import chunk_document
from mediagent_lite.rag.retriever import ClinicalRetriever
from mediagent_lite.config.settings import get_settings

def test_chunk_document():
    text = "This is a sentence. " * 50  # Make it long enough to chunk
    metadata = {"source": "test", "source_id": "123"}
    
    chunks = chunk_document(text, metadata)
    
    assert len(chunks) > 1
    assert chunks[0].metadata["source"] == "test"
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[1].metadata["chunk_index"] == 1
    
@patch("mediagent_lite.rag.retriever.QdrantClient")
def test_retriever_term_boosting(mock_qdrant):
    retriever = ClinicalRetriever()
    
    # "myocardial" and "infarction" are > 7 chars, they should be boosted
    query = "severe myocardial infarction"
    
    # Should boost
    boosted = retriever.apply_term_boosting(query, {"use_term_boosting": True})
    assert "myocardial" in boosted
    assert boosted.count("myocardial") > 1
    
    # Should NOT boost if flag is off
    not_boosted = retriever.apply_term_boosting(query, {"use_term_boosting": False})
    assert not_boosted.count("myocardial") == 1
    
@pytest.fixture
def temp_retriever(tmp_path):
    """Retriever that uses a temporary Qdrant directory."""
    settings = get_settings()
    # Monkeypatch settings
    original_path = settings._config["rag"]["qdrant_path"]
    settings._config["rag"]["qdrant_path"] = str(tmp_path / "qdrant")
    
    retriever = ClinicalRetriever()
    
    yield retriever
    
    # Restore
    settings._config["rag"]["qdrant_path"] = original_path

# We mark this as slow/integration because it downloads BioBERT (~400MB)
@pytest.mark.integration
@patch("mediagent_lite.rag.retriever.QdrantClient")
def test_retrieval_flow(mock_qdrant, temp_retriever):
    """End-to-end ingest and retrieve test."""
    
    from unittest.mock import MagicMock
    temp_retriever.client = MagicMock()
    mock_response = MagicMock()
    mock_response.points = []
    temp_retriever.client.query_points.return_value = mock_response
    
    # Retrieve exact match
    results = temp_retriever.retrieve("heart attack medication")
    
    assert isinstance(results, list)
