# MediAgent-Lite

A scaled-down, runnable re-implementation of an adaptive multi-agent clinical decision support system based on the IEEE Access 2025 paper: *"An Adaptive Multi-Agent LLM-Based CDSS Integrating Biomedical RAG and Web Intelligence"*.

**Disclaimer:** For educational and research use only. Not a medical device.

## Features
- **LangGraph State Machine**: Orchestrates multiple LLMs in a cyclic graph.
- **RAG via Qdrant**: Local vector database using BioBERT embeddings.
- **Parallel Web Search**: Reaches out to PubMed via LangChain Tools.
- **Deterministic Confidence Math**: Uses a hardcoded formula to prevent LLM hallucination of confidence scores.
- **Adaptive Feedback Loop**: Rewrites queries and searches again if confidence is below threshold $\tau$.

## Quickstart

1. Fill out `.env` with your API keys.
2. Run ingestion to build the local Vector DB:
   ```bash
   python scripts/ingest.py
   ```
3. Run the Streamlit UI:
   ```bash
   streamlit run app.py
   ```
