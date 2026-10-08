# MediAgent-Lite

[![CI](https://github.com/Hrishikesh-Rondla/MediAgent-Lite/actions/workflows/ci.yml/badge.svg)](https://github.com/Hrishikesh-Rondla/MediAgent-Lite/actions/workflows/ci.yml)

A multi-agent clinical decision support system (CDSS) implementing the architecture from the IEEE Access 2025 paper: *"An Adaptive Multi-Agent LLM-Based CDSS Integrating Biomedical RAG and Web Intelligence"*.

**Disclaimer:** For educational, research, and portfolio use only. Use synthetic data only; no PHI. Not a medical device.

## 📊 Technical Achievements

- **Concurrency:** Implemented a `ThreadPoolExecutor` within the LangGraph architecture to evaluate diagnostic hypotheses in parallel, reducing PubMed API retrieval latency by **~40%** (benchmark in `results/`).
- **Local Edge Inference:** Optimized a 5-agent pipeline to run entirely locally on an 8B parameter model (Llama 3.1 8B via Ollama) using an RTX 4060 (8GB VRAM), requiring $0 in LLM API costs.
- **Hallucination Guardrails:** Designed a deterministic Python override mechanism that catches LLM JSON hallucinations (e.g., overriding LLM-generated confidence scores with the paper's mathematical formula).
- **Automated Evaluation Harness:** Built a custom evaluation script integrating the MedQA-USMLE dataset to rigorously benchmark system accuracy, computing 95% Wilson confidence intervals across various ablation configurations.
- **Biomedical RAG:** Ingested medical literature chunks into a **Qdrant** vector database, embedded via **BioBERT** (768-dimensional space) for semantic search.

## 🚀 Features
- **Multi-Agent Orchestration**: Powered by **LangGraph**, utilizing specialized LLM agents (Clarifier, RAG Analyzer, Evidence Scanner, Fusion Node, Optimizer).
- **Privacy & Data Flow**: LLM inference runs locally via Ollama. The only outbound network traffic occurs when the Evidence Scanner queries the public PubMed (NCBI E-utilities) API using non-PII symptom and diagnosis terms. 
- **Offline Mode**: A `web_evidence_enabled` config flag allows disabling PubMed entirely, making the system 100% offline-capable (relying solely on the local Qdrant database).
- **Provider Routing**: Swap between Cloud APIs (Groq, Gemini) and Local APIs (Ollama) via config.
- **One-Shot JSON Prompting**: Utilizes prompt engineering and 1-shot schema injection to prevent edge models from entering repetitive loops.

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
If you intend to run this locally using Ollama, **you do not need any API keys**.
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
This repository serves as a portfolio piece demonstrating AI debugging:
- **Pydantic Validation Resiliency**: Implemented a clamp-before-validate pattern to prevent the application from crashing when local 8B models hallucinate integers for probability fields. 
- **Programmatic Query Construction**: Bypassed weak LLM query-generation capabilities by hardcoding Boolean searches (`Diagnosis + Symptom`), guaranteeing correct PubMed retrieval for local edge models.
- **Streamlit State Caching**: Solved double-execution bottlenecks by using LangGraph `stream_mode="values"` state caching.
- **Testing Mocks**: Built a context-aware `FakeLLM` mock to enable offline unit testing.

## 🔮 Future Scope
- **Dynamic Fallback Routing (Agentic RAG):** Implementing conditional routing to bypass the local vector DB and trigger a "Deep Research Agent" if local RAG confidence falls below 0.35.
- **Multi-Modal Integration:** Upgrading the Clarifier agent to accept X-Rays and ECGs using native vision models like Gemini 1.5.
- **Human-in-the-Loop (HITL):** Implementing LangGraph breakpoints to pause execution and request physician approval on retrieved PubMed abstracts before generating the final report.
- **FHIR API Integration:** Connecting directly to EHR databases (Epic/Cerner) to automatically populate patient context, eliminating manual text entry.
