"""PubMed E-utilities client with caching and rate limiting."""

import hashlib
import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from mediagent_lite.config.settings import get_settings
from mediagent_lite.schemas.evidence import PublicationType, PubMedAbstract


class PubMedRateLimitError(Exception):
    pass


class PubMedFetcher:
    """Client for NCBI E-utilities."""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self):
        self.settings = get_settings()
        self.email = os.getenv("NCBI_EMAIL")
        self.api_key = os.getenv("NCBI_API_KEY")
        
        self.cache_dir = Path(self.settings.pubmed["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.rate_limit_delay = self.settings.pubmed["rate_limit_delay"]
        self.last_request_time = 0.0

    def _wait_for_rate_limit(self):
        """Ensure we don't exceed NCBI rate limits."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def _get_cache_path(self, prefix: str, query: str) -> Path:
        """Generate a deterministic cache file path."""
        hashed = hashlib.md5(query.encode()).hexdigest()
        return self.cache_dir / f"{prefix}_{hashed}.json"

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((requests.RequestException, PubMedRateLimitError))
    )
    def _make_request(self, endpoint: str, params: dict) -> requests.Response:
        """Make an HTTP request with rate limiting and retry."""
        self._wait_for_rate_limit()
        
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
            
        url = f"{self.BASE_URL}/{endpoint}"
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 429:
            raise PubMedRateLimitError("NCBI Rate limit exceeded")
            
        response.raise_for_status()
        return response

    def search_pmids(self, query: str, retmax: int = None) -> list[str]:
        """Search PubMed and return a list of PMIDs."""
        if not retmax:
            retmax = self.settings.pubmed["retmax"]
            
        cache_path = self._get_cache_path("search", f"{query}_{retmax}")
        if cache_path.exists():
            with open(cache_path, "r") as f:
                return json.load(f)

        params = {
            "db": "pubmed",
            "term": query,
            "retmax": retmax,
            "retmode": "json",
            "sort": "relevance"
        }
        
        response = self._make_request("esearch.fcgi", params)
        data = response.json()
        pmids = data.get("esearchresult", {}).get("idlist", [])
        
        with open(cache_path, "w") as f:
            json.dump(pmids, f)
            
        return pmids

    def fetch_abstracts(self, pmids: list[str]) -> list[PubMedAbstract]:
        """Fetch XML details for PMIDs and parse into PubMedAbstract objects."""
        if not pmids:
            return []
            
        # To avoid URI Too Long HTTP errors (max ~2000 chars), we batch the requests
        batch_size = 100
        all_abstracts = []
        
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]
            pmid_str = ",".join(batch)
            cache_path = self._get_cache_path("fetch", pmid_str)
            
            if cache_path.exists():
                try:
                    with open(cache_path, "r") as f:
                        data = json.load(f)
                        all_abstracts.extend([PubMedAbstract.model_validate(item) for item in data])
                        continue
                except Exception:
                    pass
                    
            params = {
                "db": "pubmed",
                "id": pmid_str,
                "retmode": "xml",
            }
            
            response = self._make_request("efetch.fcgi", params)
            batch_abstracts = self._parse_xml(response.text)
            
            with open(cache_path, "w") as f:
                json.dump([a.model_dump() for a in batch_abstracts], f)
                
            all_abstracts.extend(batch_abstracts)
            
        return all_abstracts

    def _parse_xml(self, xml_text: str) -> list[PubMedAbstract]:
        """Parse PubMed efetch XML into objects."""
        abstracts = []
        try:
            root = ET.fromstring(xml_text)
            for article in root.findall(".//PubmedArticle"):
                pmid_elem = article.find(".//PMID")
                title_elem = article.find(".//ArticleTitle")
                abstract_elem = article.find(".//AbstractText")
                year_elem = article.find(".//PubDate/Year")
                
                if pmid_elem is None or title_elem is None or abstract_elem is None:
                    continue
                    
                pmid = pmid_elem.text
                title = "".join(title_elem.itertext())
                
                # Abstract might have multiple parts (Background, Methods, etc)
                abstract_parts = []
                for part in article.findall(".//AbstractText"):
                    label = part.get("Label", "")
                    text = "".join(part.itertext())
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
                abstract = "\n".join(abstract_parts)
                
                year = int(year_elem.text) if year_elem is not None else None
                
                # Publication Types
                pub_types = []
                for pt in article.findall(".//PublicationType"):
                    if pt.text:
                        pub_types.append(pt.text)
                        
                classified_type = self._classify_publication_type(pub_types)
                
                abstracts.append(
                    PubMedAbstract(
                        pmid=pmid,
                        title=title,
                        abstract=abstract,
                        year=year,
                        publication_types=pub_types,
                        classified_type=classified_type
                    )
                )
        except ET.ParseError:
            pass
            
        return abstracts

    def _classify_publication_type(self, types: list[str]) -> PublicationType:
        """Map raw PubMed types to our evidence hierarchy."""
        types_lower = [t.lower() for t in types]
        
        if any("guideline" in t for t in types_lower):
            return PublicationType.GUIDELINE
        if any("meta-analysis" in t for t in types_lower):
            return PublicationType.META_ANALYSIS
        if any("systematic review" in t for t in types_lower):
            return PublicationType.SYSTEMATIC_REVIEW
        if any("randomized controlled trial" in t for t in types_lower):
            return PublicationType.RCT
        if any("cohort" in t for t in types_lower):
            return PublicationType.COHORT
        if any("case-control" in t for t in types_lower):
            return PublicationType.CASE_CONTROL
        if any("case report" in t for t in types_lower):
            return PublicationType.CASE_REPORT
            
        return PublicationType.OTHER
