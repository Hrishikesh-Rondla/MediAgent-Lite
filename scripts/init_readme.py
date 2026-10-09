def rewrite_readme():
    readme = """# MediAgent-Lite

[![CI](https://github.com/Hrishikesh-Rondla/MediAgent-Lite/actions/workflows/ci.yml/badge.svg)](https://github.com/Hrishikesh-Rondla/MediAgent-Lite/actions/workflows/ci.yml)

A multi-agent clinical decision support system (CDSS) implementing an adaptive architecture using local LLMs, Qdrant vector retrieval, and PubMed evidence scanning.

**Disclaimer:** For educational, research, and portfolio use only. Use synthetic data only; no PHI. Not a medical device.

## 2. Architecture

```mermaid
graph TD
    A[Raw Clinical Input] --> B(Clarifier Agent)
    B --> C{Parallel Fan-Out}
    C -->|Biomedical RAG| D[RAG Analyzer]
    C -->|Web Evidence| E[Evidence Scanner]
    D --> F(Fusion Node)
    E --> F
    F --> G{Confidence > Tau?}
    G -->|No| H(Optimizer)
    H -->|Rewrite Query| C
    G -->|Yes / Max Iter| I[Final Report]
```

- **Clarifier Agent:** Extracts symptoms, demographics, and vitals into a strict Pydantic `StructuredCase`.
- **RAG Analyzer:** Queries the local Qdrant database to retrieve medical chunks and proposes initial diagnostic hypotheses.
- **Evidence Scanner:** Programmatically hits the PubMed API for each hypothesis in parallel to find supporting or refuting abstracts.
- **Fusion Node:** Synthesizes findings and calculates a deterministic confidence score.
- **Optimizer Node:** If confidence is too low, rewrites queries to cast a wider net and triggers the feedback loop.

## 3. Dashboard Traces
*(See the Streamlit application for interactive traces of the LangGraph execution, iteration timelines, and chunk-level text spans for grounded evidence).*

![Streamlit Trace](docs/assets/streamlit_trace.png)

<!-- RESULTS:START -->
<!-- RESULTS:END -->

## 6. Confidence Scoring
The system does NOT rely on an LLM's self-reported confidence. Instead, a strict deterministic formula is applied in Python:
`C = (w1 * S_rag) + (w2 * S_ev) + (w3 * S_concordance)`

- `w1=0.35`: Weight for max cosine similarity from Qdrant chunks.
- `w2=0.40`: Weight for PubMed evidence (RCTs=0.8, Case Reports=0.3).
- `w3=0.25`: Weight for concordance (RAG and PubMed agreement).
- *Weights were tuned on a separate 20-question dev split, not the MedQA test split.*
- An explicit unit test (`test_deterministic_confidence_math`) proves that hallucinated LLM scores are aggressively overwritten.

## 7. Privacy & Data Flow

| Component | Destination | Payload Fields |
|---|---|---|
| Evidence Scanner | `https://eutils.ncbi.nlm.nih.gov/` | ONLY `diagnosis` + `symptom.name`. (PII is rigorously scrubbed). |
| Cloud LLMs (optional) | Groq / Gemini API | Full clinical case JSON (if enabled). |

**Offline Mode:** Setting `web_evidence_enabled = false` in `config.yaml` physically bypasses all PubMed HTTP calls. LLM inference runs 100% locally on the host machine when `DEFAULT_LLM_PROVIDER=ollama`.

## 8. Design Decisions and Tradeoffs
- **Programmatic PubMed Queries:** Local 8B models struggle with complex Boolean search strings. We hardcoded `Diagnosis + Symptom` queries to guarantee retrieval success.
- **Clamp-before-validate:** Local LLMs hallucinate integer probabilities (e.g., `3` instead of `0.9`). We clamp values to `[0.0, 1.0]` before Pydantic validation to prevent crashes.
- **Deterministic Confidence:** Math beats LLM self-evaluations.

## 9. Limitations
- **Small Corpus:** The local RAG database contains only ~500 rows.
- **Eval Constraints:** Measured on N=10 due to local edge hardware runtime constraints; Wilson CI margins are accordingly wide.
- **Benchmark != Patients:** USMLE questions are synthetic and text-heavy, not messy real-world EHRs.
- **PubMed Abstracts:** We only retrieve abstracts, lacking full-text context.

## 10. Quickstart
```bash
# Windows
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Reproduce the Results
```bash
# 1. Ingest synthetic data
python scripts/ingest.py

# 2. Run the evaluation
python scripts/run_eval.py --n 10
python scripts/generate_plots_and_errors.py

# 3. Run the parallel benchmark
python scripts/bench_parallel.py --trials 10

# 4. Render results into README
python scripts/render_readme_results.py
```

## 11. Roadmap
1. **Dynamic Fallback Routing (Agentic RAG):** Implementing conditional routing to bypass the local vector DB if initial extraction fails.
2. **Human-in-the-Loop (HITL):** Implementing LangGraph breakpoints to pause execution and request physician approval on retrieved PubMed abstracts.
3. **Evidence Extraction Improvements:** Moving beyond abstracts to parse full-text PMC articles.
"""
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme)
    
if __name__ == "__main__":
    rewrite_readme()
