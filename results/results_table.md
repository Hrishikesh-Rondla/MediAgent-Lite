# Evaluation Results — MedQA-USMLE-4-options (N=80 test questions)

> Model: Llama 3.1 8B via Ollama (local inference)
> Seed: 42. Dev/test split: 20%/80%.
> Answer mapping: substring match between top diagnosis and option text.
> Wilson 95% CI shown in brackets.

| Config | Accuracy | 95% CI | Mean Latency | Mean Iters | Loop % |
|--------|----------|--------|-------------|------------|--------|
| E | 1.2% | 0.2%–6.8% | 2.9s | 0.00 | 0% |

**Config key:**
- A = LLM-only baseline (Ollama, no RAG, no PubMed)
- D = Full pipeline, single pass (max_iter=1)
- E = Full pipeline with adaptive feedback loop (max_iter=3)

> [!NOTE]
> Accuracy on USMLE questions is expected to be low for an 8B local model.
> The architecture's value is in grounding, traceability, and evidence scoring,
> not raw multiple-choice accuracy on questions designed for 70B+ models.