"""Tests for PubMed fetcher and parser."""

import os
import pytest
from unittest.mock import patch, MagicMock

from mediagent_lite.rag.pubmed_fetcher import PubMedFetcher
from mediagent_lite.schemas.evidence import PublicationType

# Sample XML mimicking PubMed efetch response
SAMPLE_XML = """<?xml version="1.0" ?>
<!DOCTYPE PubmedArticleSet PUBLIC "-//NLM//DTD PubMedArticle, 1st January 2024//EN" "https://dtd.nlm.nih.gov/ncbi/pubmed/out/pubmed_240101.dtd">
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">12345</PMID>
      <Article>
        <ArticleTitle>Aspirin for Myocardial Infarction.</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Aspirin is good.</AbstractText>
          <AbstractText Label="RESULTS">It works.</AbstractText>
        </Abstract>
        <PublicationTypeList>
          <PublicationType UI="D016449">Randomized Controlled Trial</PublicationType>
        </PublicationTypeList>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""

@pytest.fixture
def pubmed_fetcher(tmp_path):
    """Fetcher with temp cache dir."""
    fetcher = PubMedFetcher()
    fetcher.cache_dir = tmp_path
    return fetcher

def test_parse_xml(pubmed_fetcher):
    abstracts = pubmed_fetcher._parse_xml(SAMPLE_XML)
    
    assert len(abstracts) == 1
    assert abstracts[0].pmid == "12345"
    assert abstracts[0].title == "Aspirin for Myocardial Infarction."
    assert "BACKGROUND: Aspirin is good." in abstracts[0].abstract
    assert abstracts[0].classified_type == PublicationType.RCT
    assert "Randomized Controlled Trial" in abstracts[0].publication_types

def test_classify_publication_type(pubmed_fetcher):
    assert pubmed_fetcher._classify_publication_type(["Practice Guideline"]) == PublicationType.GUIDELINE
    assert pubmed_fetcher._classify_publication_type(["Meta-Analysis"]) == PublicationType.META_ANALYSIS
    assert pubmed_fetcher._classify_publication_type(["Case Reports"]) == PublicationType.CASE_REPORT
    assert pubmed_fetcher._classify_publication_type(["Journal Article", "Review"]) == PublicationType.OTHER

@patch("mediagent_lite.rag.pubmed_fetcher.requests.get")
def test_search_pmids_api_call(mock_get, pubmed_fetcher):
    # Setup mock
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"esearchresult": {"idlist": ["111", "222"]}}
    mock_get.return_value = mock_response
    
    # First call hits API
    pmids = pubmed_fetcher.search_pmids("test query", retmax=2)
    assert pmids == ["111", "222"]
    assert mock_get.call_count == 1
    
    # Second call hits cache (mock not called again)
    pmids_cached = pubmed_fetcher.search_pmids("test query", retmax=2)
    assert pmids_cached == ["111", "222"]
    assert mock_get.call_count == 1
