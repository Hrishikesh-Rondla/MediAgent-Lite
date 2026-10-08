"""LangChain Tool wrapper for PubMed Fetcher."""

from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from mediagent_lite.rag.pubmed_fetcher import PubMedFetcher
from mediagent_lite.schemas.evidence import PubMedAbstract


class PubMedSearchInput(BaseModel):
    """Input for PubMed search tool."""
    query: str = Field(description="The search query for PubMed (e.g., 'acute myocardial infarction AND troponin')")
    retmax: int = Field(default=3, description="Maximum number of abstracts to return")


class PubMedTool(BaseTool):
    """Tool to search PubMed and return formatted abstracts."""
    
    name: str = "search_pubmed"
    description: str = "Search PubMed for medical literature. Returns titles, publication types, and abstracts."
    args_schema: Type[BaseModel] = PubMedSearchInput
    
    # We initialize the fetcher once per tool instance to reuse the session/cache
    fetcher: PubMedFetcher = None
    
    def __init__(self):
        super().__init__()
        self.fetcher = PubMedFetcher()
        
    def _run(self, query: str, retmax: int = 3) -> str:
        """Execute the tool."""
        try:
            pmids = self.fetcher.search_pmids(query, retmax=retmax)
            if not pmids:
                return f"No results found for query: {query}"
                
            abstracts: list[PubMedAbstract] = self.fetcher.fetch_abstracts(pmids)
            
            # Format nicely for the LLM
            formatted_results = []
            for a in abstracts:
                pub_types = ", ".join(a.publication_types) if a.publication_types else "Unknown"
                entry = (
                    f"PMID: {a.pmid}\n"
                    f"Title: {a.title}\n"
                    f"Year: {a.year}\n"
                    f"Publication Types: {pub_types} (Classified as: {a.classified_type.value})\n"
                    f"Abstract:\n{a.abstract}\n"
                    f"{'-'*40}"
                )
                formatted_results.append(entry)
                
            return "\n\n".join(formatted_results)
            
        except Exception as e:
            return f"Error occurred while searching PubMed: {str(e)}"
