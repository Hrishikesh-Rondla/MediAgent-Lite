"""BioBERT Embeddings wrapper for LangChain.

Why this model? "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
It is fine-tuned on medical NLI and STS datasets, making it much better
at medical semantic similarity than standard sentence-transformers.
"""

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class BioBERTEmbeddings(Embeddings):
    """LangChain wrapper for SentenceTransformers BioBERT."""

    def __init__(self, model_name: str):
        # We load this lazily or once per process
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        # Convert numpy arrays to lists of floats for Qdrant
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
