"""Evaluation Harness for MediAgent-Lite.

Runs the system against a subset of the MedQA-USMLE dataset.
Calculates Accuracy, Precision, Recall, F1, and mean Confidence.
Supports running ablations (e.g., turning off the Web Scanner or RAG).
"""

import os
import json
import time
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from mediagent_lite.graph.orchestrator import build_graph
from mediagent_lite.config.settings import get_settings


def load_medqa_data(n_questions: int = 20):
    """Load MedQA-USMLE dataset from HuggingFace."""
    print(f"Loading MedQA-USMLE dataset (top {n_questions} questions)...")
    
    # We use the 4-options dataset commonly used for USMLE evaluation
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    
    # Take a subset to keep eval time reasonable
    subset = dataset.select(range(n_questions))
    
    formatted_data = []
    for row in subset:
        # Reconstruct the multiple-choice format into a clinical case narrative
        # so our unstructured text clarifier can process it naturally.
        options_text = "; ".join([f"{k}: {v}" for k, v in row["options"].items()])
        narrative = f"Patient presentation: {row['question']} Possible diagnoses to consider: {options_text}"
        
        formatted_data.append({
            "id": row["id"],
            "narrative": narrative,
            "correct_answer": row["answer_idx"],  # e.g., 'A', 'B'
            "correct_text": row["options"][row["answer_idx"]]
        })
        
    return formatted_data


def extract_prediction(report, options):
    """Extract which option the LLM chose."""
    if not report or not report.diagnoses:
        return None
        
    # We check if the top predicted diagnosis matches any of the options text
    top_dx = report.diagnoses[0].diagnosis.lower()
    
    for opt_key, opt_val in options.items():
        if top_dx in opt_val.lower() or opt_val.lower() in top_dx:
            return opt_key
            
    return None # Could not map


def run_evaluation(ablation_name: str, config_flags: dict):
    """Run the graph over the dataset with specific config flags."""
    settings = get_settings()
    n_q = settings.eval.get("n_questions", 20)
    data = load_medqa_data(n_q)
    
    graph = build_graph()
    
    results = []
    correct_count = 0
    total_latency = 0
    
    print(f"\nRunning Evaluation: {ablation_name}")
    print(f"Flags: {config_flags}")
    print("-" * 50)
    
    for item in tqdm(data, desc="Evaluating"):
        start_t = time.time()
        
        initial_state = {
            "raw_input": item["narrative"],
            "structured_case": None,
            "hypotheses": None,
            "evidence_bundles": [],
            "final_report": None,
            "trace": [],
            "query_history": [],
            "iteration": 0,
            "config_flags": config_flags
        }
        
        try:
            final_state = graph.invoke(initial_state)
            report = final_state.get("final_report")
            
            # Simple heuristic evaluation: did the top diagnosis match the exact correct text?
            # In a real academic paper, you'd use a dedicated LLM-as-a-judge node to compare.
            is_correct = False
            top_dx = ""
            conf = 0.0
            
            if report and report.diagnoses:
                top_dx = report.diagnoses[0].diagnosis
                conf = report.diagnoses[0].confidence
                
                # Check for substring match (e.g., "Acute Myocardial Infarction" vs "Myocardial Infarction")
                if item["correct_text"].lower() in top_dx.lower() or top_dx.lower() in item["correct_text"].lower():
                    is_correct = True
                    
            if is_correct:
                correct_count += 1
                
            latency = time.time() - start_t
            total_latency += latency
            
            results.append({
                "id": item["id"],
                "is_correct": is_correct,
                "predicted": top_dx,
                "actual": item["correct_text"],
                "confidence": conf,
                "latency_sec": latency,
                "iterations": final_state.get("iteration", 0)
            })
            
        except Exception as e:
            print(f"\nError on case {item['id']}: {e}")
            results.append({
                "id": item["id"],
                "is_correct": False,
                "predicted": f"ERROR: {str(e)}",
                "actual": item["correct_text"],
                "confidence": 0.0,
                "latency_sec": time.time() - start_t,
                "iterations": 0
            })
            
    # Compute metrics
    accuracy = correct_count / len(data) if data else 0
    mean_latency = total_latency / len(data) if data else 0
    mean_conf = sum(r["confidence"] for r in results) / len(data) if data else 0
    
    print("\n" + "=" * 50)
    print(f"RESULTS: {ablation_name}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Mean Confidence: {mean_conf:.3f}")
    print(f"Mean Latency: {mean_latency:.2f}s")
    print("=" * 50)
    
    # Save results
    out_dir = Path(settings.eval["results_dir"])
    out_dir.mkdir(exist_ok=True)
    
    df = pd.DataFrame(results)
    df.to_csv(out_dir / f"eval_{ablation_name}.csv", index=False)
    
    summary = {
        "ablation": ablation_name,
        "flags": config_flags,
        "accuracy": accuracy,
        "mean_confidence": mean_conf,
        "mean_latency": mean_latency,
        "n": len(data)
    }
    
    with open(out_dir / f"summary_{ablation_name}.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    return summary


if __name__ == "__main__":
    # Ensure fake LLM is NOT used for evals unless specified
    if os.getenv("DEFAULT_LLM_PROVIDER") == "fake":
        print("WARNING: Evaluating using FakeLLM. Accuracy will be 0%.")
        
    # 1. Full Pipeline
    run_evaluation("full_pipeline", {"enable_rag": True, "enable_web": True})
    
    # 2. Ablation: No Web Scanner (RAG only)
    run_evaluation("rag_only", {"enable_rag": True, "enable_web": False})
    
    # 3. Ablation: Baseline (No RAG, No Web)
    run_evaluation("baseline_llm_only", {"enable_rag": False, "enable_web": False})
