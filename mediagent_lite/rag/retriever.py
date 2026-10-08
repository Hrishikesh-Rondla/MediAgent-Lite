"""Qdrant local retriever with fallback ladder and term boosting."""

from typing import Any
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, ScoredPoint
from langchain_core.documents import Document

from mediagent_lite.config.settings import get_settings
from mediagent_lite.rag.embedder import BioBERTEmbeddings


class ClinicalRetriever:
    """Manages Qdrant vector DB and executes retrieval logic."""

    def __init__(self):
        self.settings = get_settings()
        self.collection_name = self.settings.collection_name
        self.client = QdrantClient(path=self.settings.qdrant_path)
        
        # We don't load the embedder until needed to save memory
        self._embedder = None
        
    @property
    def embedder(self) -> BioBERTEmbeddings:
        if self._embedder is None:
            self._embedder = BioBERTEmbeddings(self.settings.embedding_model)
        return self._embedder

    def ensure_collection(self):
        """Create collection if it doesn't exist."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.settings.rag["embedding_dim"],
                    distance=Distance.COSINE,
                ),
            )

    def add_documents(self, documents: list[Document]):
        """Embed and upsert documents into Qdrant."""
        self.ensure_collection()
        
        if not documents:
            return

        texts = [doc.page_content for doc in documents]
        embeddings = self.embedder.embed_documents(texts)
        
        points = []
        for doc, emb in zip(documents, embeddings):
            # We use a UUID based on the content and metadata so it's deterministic/idempotent
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc.metadata.get('source_id', '')}_{doc.metadata.get('chunk_index', 0)}"))
            
            points.append(
                PointStruct(
                    id=doc_id,
                    vector=emb,
                    payload={
                        "page_content": doc.page_content,
                        **doc.metadata
                    }
                )
            )
            
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[i:i + batch_size]
            )

    def _search(self, query: str, limit: int, threshold: float) -> list[tuple[Document, float]]:
        """Raw vector search."""
        query_vector = self.embedder.embed_query(query)
        
        # Qdrant returns score as cosine similarity
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=threshold,
        )
        results = response.points
        
        docs_with_scores = []
        for res in results:
            # We must handle the payload carefully as Qdrant types it as Any
            payload = res.payload or {}
            content = payload.pop("page_content", "")
            # Re-inject the Qdrant ID into metadata for traceability
            payload["qdrant_id"] = str(res.id)
            docs_with_scores.append((Document(page_content=content, metadata=payload), res.score))
            
        return docs_with_scores

    def apply_term_boosting(self, query: str, config_flags: dict[str, Any]) -> str:
        """Boost medical terms if enabled.
        
        In a full app, this would use TF-IDF over the corpus.
        For this Lite version, we just blindly repeat long words (heuristically medical).
        """
        if not config_flags.get("use_term_boosting", True):
            return query
            
        boosting_config = self.settings.rag.get("term_boosting", {})
        if not boosting_config.get("enabled", True):
            return query
            
        words = query.split()
        # Simple heuristic: words > 7 chars are likely medical terms
        long_words = [w for w in words if len(w) > 7]
        
        boost_factor = boosting_config.get("boost_factor", 2)
        top_n = boosting_config.get("top_n_terms", 3)
        
        to_boost = long_words[:top_n]
        if not to_boost:
            return query
            
        boost_string = " ".join(to_boost * (boost_factor - 1))
        return f"{query} {boost_string}"

    def retrieve(self, query: str, config_flags: dict[str, Any] = None) -> list[tuple[Document, float]]:
        """Retrieve with fallback ladder."""
        if config_flags is None:
            config_flags = {}
            
        self.ensure_collection()
        limit = self.settings.top_k
        primary_threshold = self.settings.similarity_threshold
        fallback_threshold = self.settings.rag["fallback_threshold"]
        
        boosted_query = self.apply_term_boosting(query, config_flags)
        
        # Step 1: Primary search
        results = self._search(boosted_query, limit, primary_threshold)
        if results:
            return results
            
        # Step 2: Fallback threshold
        results = self._search(boosted_query, limit, fallback_threshold)
        if results:
            return results
            
        # Step 3: Simplify query (in a real app we'd ask LLM, here we just take first N words)
        simplified_query = " ".join(query.split()[:4])
        results = self._search(simplified_query, limit, fallback_threshold)
        
        return results
