"""Tests for PubMedTool."""

from unittest.mock import patch

from mediagent_lite.tools.pubmed_tool import PubMedTool
from mediagent_lite.schemas.evidence import PubMedAbstract, PublicationType

@patch("mediagent_lite.rag.pubmed_fetcher.PubMedFetcher.search_pmids")
@patch("mediagent_lite.rag.pubmed_fetcher.PubMedFetcher.fetch_abstracts")
def test_pubmed_tool_run(mock_fetch, mock_search):
    # Setup mocks
    mock_search.return_value = ["123"]
    mock_abstract = PubMedAbstract(
        pmid="123",
        title="Test Title",
        abstract="Test abstract content.",
        year=2023,
        publication_types=["Journal Article"],
        classified_type=PublicationType.OTHER
    )
    mock_fetch.return_value = [mock_abstract]
    
    tool = PubMedTool()
    result = tool._run("test query", retmax=1)
    
    assert "PMID: 123" in result
    assert "Test Title" in result
    assert "Test abstract content." in result
    
@patch("mediagent_lite.rag.pubmed_fetcher.PubMedFetcher.search_pmids")
def test_pubmed_tool_empty(mock_search):
    mock_search.return_value = []
    
    tool = PubMedTool()
    result = tool._run("test query", retmax=1)
    
    assert "No results found" in result
