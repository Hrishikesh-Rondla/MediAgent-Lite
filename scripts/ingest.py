"""Corpus ingestion script.

Downloads PubMed abstracts and HuggingFace symptom-disease dataset,
chunks them, embeds with BioBERT, and upserts to local Qdrant.
Idempotent: safe to run multiple times.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from datasets import load_dataset

from mediagent_lite.config.settings import get_settings
from mediagent_lite.rag.chunker import chunk_document
from mediagent_lite.rag.pubmed_fetcher import PubMedFetcher
from mediagent_lite.rag.retriever import ClinicalRetriever


def ingest_pubmed(fetcher: PubMedFetcher, retriever: ClinicalRetriever, settings):
    """Fetch and ingest PubMed abstracts based on config queries."""
    queries = settings.ingestion["pubmed_queries"]
    max_per_query = settings.ingestion["max_abstracts_per_query"]
    
    print(f"Fetching PubMed abstracts for {len(queries)} queries...")
    
    all_pmids = set()
    for q in queries:
        print(f"  Searching: {q}")
        pmids = fetcher.search_pmids(q, retmax=max_per_query)
        all_pmids.update(pmids)
        
    print(f"Found {len(all_pmids)} unique PMIDs. Fetching abstracts...")
    abstracts = fetcher.fetch_abstracts(list(all_pmids))
    
    docs = []
    for a in abstracts:
        text = f"Title: {a.title}\n\nAbstract: {a.abstract}"
        metadata = {
            "source": "pubmed",
            "source_id": a.pmid,
            "title": a.title,
            "publication_type": a.classified_type.value,
        }
        docs.extend(chunk_document(text, metadata))
        
    print(f"Created {len(docs)} chunks from PubMed.")
    if docs:
        retriever.add_documents(docs)
        print("Upserted PubMed chunks to Qdrant.")


def ingest_hf_dataset(retriever: ClinicalRetriever, settings):
    """Load symptom-disease dataset from HuggingFace."""
    dataset_name = settings.ingestion["hf_dataset"]
    split = settings.ingestion["hf_split"]
    max_rows = settings.ingestion["hf_max_rows"]
    
    print(f"\nLoading HuggingFace dataset: {dataset_name} ({split} split, max {max_rows} rows)...")
    try:
        dataset = load_dataset(dataset_name, split=split, cache_dir="./data/hf_cache")
    except Exception as e:
        print(f"Failed to load HuggingFace dataset: {e}")
        return
        
    # Take a subset if specified
    if max_rows and len(dataset) > max_rows:
        dataset = dataset.select(range(max_rows))
        
    docs = []
    for i, row in enumerate(dataset):
        # Format depends on specific dataset schema. 
        # Using Shaheer14326/Disease_Symptoms_Dataset
        disease = row.get("Disease", "")
        symptoms = row.get("Symptoms", "")
        overview = row.get("Overview", "")
        
        if not symptoms or not disease:
            continue
            
        text = f"Disease: {disease}\nOverview: {overview}\nSymptoms: {symptoms}"
        metadata = {
            "source": "hf_dataset",
            "source_id": f"hf_{i}",
            "disease": disease,
            "url": row.get("Link", "")
        }
        docs.extend(chunk_document(text, metadata))
        
    print(f"Created {len(docs)} chunks from HuggingFace dataset.")
    if docs:
        retriever.add_documents(docs)
        print("Upserted HF dataset chunks to Qdrant.")


def main():
    print("--- MediAgent-Lite: Corpus Ingestion ---")
    settings = get_settings()
    
    # Check NCBI_EMAIL
    if not os.getenv("NCBI_EMAIL"):
        print("WARNING: NCBI_EMAIL environment variable is not set.")
        print("PubMed E-utilities require an email address. Rate limits may be severe.")
        print("Set it in .env or via export NCBI_EMAIL=you@example.com")
        print("Proceeding anyway in 3 seconds...")
        import time
        time.sleep(3)
        
    fetcher = PubMedFetcher()
    retriever = ClinicalRetriever()
    
    # Create Qdrant collection
    retriever.ensure_collection()
    
    # 1. PubMed
    ingest_pubmed(fetcher, retriever, settings)
    
    # 2. HuggingFace Dataset
    ingest_hf_dataset(retriever, settings)
    
    print("\nIngestion complete! Vector DB is ready.")


if __name__ == "__main__":
    main()
