# MediAgent-Lite

A scalable, offline-capable, multi-agent clinical decision support system (CDSS). Based on the architecture from the IEEE Access 2025 paper: *"An Adaptive Multi-Agent LLM-Based CDSS Integrating Biomedical RAG and Web Intelligence"*.

**Disclaimer:** For educational, research, and portfolio use only. Not a medical device.

## 🚀 Features
- **Multi-Agent Orchestration**: Powered by **LangGraph**, utilizing specialized LLM agents (Clarifier, RAG Analyzer, Evidence Scanner, Fusion Node, Optimizer).
- **Offline / Local Execution**: Fully supports local inference via **Ollama** (e.g., Llama 3.1 8B, Qwen 2.5), ensuring complete HIPAA-compliant privacy with 0-byte cloud transmission.
- **Dynamic Provider Routing**: Seamlessly swap between Cloud APIs (Groq, Gemini) and Local APIs (Ollama) depending on rate limits and speed requirements.
- **Biomedical RAG**: Uses **Qdrant** vector database and **BioBERT** embeddings for high-precision retrieval from medical text.
- **Live Internet Grounding**: Uses LangChain Tools to fetch real-time PubMed literature to validate hypotheses.
- **Anti-Hallucination Limits**: Implements strict Pydantic structured extraction caps to prevent generative repetition loops in small local models.
- **Streamlit UI**: A clean, responsive dashboard that visualizes the LangGraph trace execution in real-time.

## 🛠️ Architecture

1. **Clarifier Agent**: Converts raw text inputs into a structured clinical case JSON.
2. **RAG Analyzer**: Queries the Qdrant DB and generates up to 3 diagnostic hypotheses.
3. **Evidence Scanner**: Hits PubMed via API tools for each hypothesis to gather supporting/refuting literature.
4. **Fusion Node**: Merges RAG scores and Evidence scores using a strict mathematical Concordance formula.
5. **Optimizer**: If confidence falls below the threshold $\tau$, this agent rewrites the queries and triggers a feedback loop.

## ⚙️ Quickstart

### 1. Environment Setup
Create a virtual environment and install dependencies:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Keys (Optional)
Copy `.env.example` to `.env`. 
If you intend to run this 100% locally using Ollama, **you do not need any API keys**.
```env
DEFAULT_LLM_PROVIDER=ollama
```

### 3. Ingest Data (RAG Vector DB)
Initialize the Qdrant database and BioBERT embeddings:
```bash
python scripts/ingest.py
```

### 4. Run the Dashboard
Start the Streamlit UI:
```bash
streamlit run app.py
```

## 🐛 Debugging & Edge-Case Handling
This repository serves as a portfolio piece demonstrating advanced AI debugging:
- **Streamlit State Caching**: Solved double-execution bottlenecks by optimizing LangGraph `stream_mode="values"` state caching.
- **Model Repetition Loops**: Fixed edge-cases where 8B parameter models enter infinite generation loops on complex Pydantic schemas by introducing strict system prompt extraction caps.
- **Dependency Migration**: Upgraded deprecated `QdrantClient` search methods to `query_points` for v1.11+ compatibility.
- **Offline Mocking**: Built a highly robust context-aware `FakeLLM` mock to bypass rate limits during rapid UI iteration.
