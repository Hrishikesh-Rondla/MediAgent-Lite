# 🧠 Interview Survival Guide: MediAgent-Lite

This document is your cheat sheet. If you put this project on your resume, a Senior Engineer or Hiring Manager will ask you hard questions about it. This guide gives you the precise, technical answers you need.

---

## 1. The Core Architecture

### Q: "Tell me about this project. How does it work?"
**A:** "MediAgent-Lite is an adaptive, multi-agent Clinical Decision Support System (CDSS) based on an IEEE 2025 paper. It takes unstructured clinical text and proposes a diagnosis using a 5-agent LangGraph workflow. 
1. The **Clarifier** normalizes raw text into a Pydantic schema.
2. The **RAG Analyzer** searches a local Qdrant vector database (embedded with BioBERT) to propose initial hypotheses.
3. The **Evidence Scanner** takes those hypotheses and searches PubMed in parallel to find real-world validation.
4. The **Fusion Agent** synthesizes the RAG and PubMed evidence using a deterministic math formula.
5. If the confidence score is below 0.65, the **Optimizer** rewrites the search query and the graph loops back, creating an adaptive feedback mechanism."

### Q: "Why did you use multiple agents instead of one big prompt?"
**A:** "Separation of concerns.
1. **Cost & Speed:** The Evidence Scanner needs to run 3 times in parallel (once for each hypothesis). I used Groq (Llama 3) for that because it's insanely fast (300+ tok/s). But the Fusion Agent requires complex reasoning to detect medical conflicts, so I used Gemini 1.5 Pro. If I used one massive prompt on a huge model, it would be slow and expensive.
2. **Determinism:** I wanted strict typing at the boundaries. The Clarifier forces the output into a Pydantic `StructuredCase`. The Fusion agent doesn't actually guess the confidence score—it calculates it using a hardcoded math formula ($C = w_1 S_{rag} + w_2 S_{ev} + w_3 S_{concordance}$). You can't do that reliably with a single prompt."

---

## 2. RAG (Retrieval-Augmented Generation)

### Q: "How does your RAG pipeline work?"
**A:** "It's a dense retrieval pipeline using `pritamdeka/BioBERT`. I chunked PubMed abstracts and a HuggingFace symptom-disease dataset into 512-token chunks with a 64-token overlap to preserve context. I load these into a local Qdrant Vector DB. When the Clarifier extracts symptoms, we embed that query and do a cosine-similarity search.
*Advanced detail:* I implemented **Query Term Boosting**. Symptoms over 7 characters (like 'tachycardia' or 'infarction') are artificially repeated in the search query. This forces the BioBERT embeddings to weight those rare, specific terms higher than common words like 'pain'."

### Q: "What happens if the Vector DB returns garbage?"
**A:** "I built a rule-engine filter into the `SymptomRAGAnalyzer`. The LLM is instructed to propose diagnoses *and cite the exact chunk ID*. If the LLM hallucinates a diagnosis and fails to cite a valid chunk, the Python code intercepts it and silently drops that hypothesis before it reaches the Fusion node."

---

## 3. LangGraph & Orchestration

### Q: "Why LangGraph instead of LangChain's SequentialChain?"
**A:** "Because LangGraph supports **cyclic graphs (loops)** and **conditional edges**.
In a standard chain, if the LLM gets confused, the pipeline fails. In my LangGraph, the edge after the Fusion node checks the `confidence` score. If `confidence < 0.65`, it routes to an `Optimizer` node. The Optimizer is a lightweight LLM that rewrites the search query (e.g., swapping 'chest pain' to 'angina pectoris') and loops the graph back to the RAG node. It's a true state machine."

---

## 4. Evaluation and Metrics

### Q: "How do you know it actually works? Did you just eyeball it?"
**A:** "No, I built an evaluation harness (`scripts/evaluate.py`). I run the graph against a subset of the **MedQA-USMLE** dataset. It calculates Accuracy, Mean Confidence, and Latency.
More importantly, I run **Ablation Studies**. I run the dataset through the full pipeline, then I turn off the Web Scanner (RAG only), and then I turn off RAG (baseline LLM only). This proves scientifically how much accuracy is added by the RAG and the parallel Web Scanning."

---

## 5. Potential Weaknesses (Be honest!)

### Q: "What are the limitations of your implementation?"
**A:** 
1. **ICD-10 Mapping:** I used a simple exact-match CSV lookup for ICD-10 codes. In a production system, I would need a dedicated Entity Linking model (like UMLS or MedCAT) to handle aliases.
2. **Chunking Strategy:** I used standard RecursiveCharacterTextSplitter. Medical texts often have strict sections (e.g., 'Results', 'Methods'). A semantic chunker that respects document structure would yield cleaner retrieval.
3. **Medical Device Regulation:** This is purely an educational tool. A real CDSS requires FDA/CE certification, massive clinical trial validation, and a human-in-the-loop requirement.

---
*Good luck. You built this, you understand the code, and you can defend the architecture.*
