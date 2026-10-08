"""Generates visualizations for the evaluation results."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def generate_plots():
    results_dir = Path("results")
    if not results_dir.exists():
        print("No results directory found. Run evaluate.py first.")
        return
        
    summaries = []
    for file in results_dir.glob("summary_*.json"):
        with open(file, "r") as f:
            summaries.append(json.load(f))
            
    if not summaries:
        print("No summary JSONs found.")
        return
        
    df = pd.DataFrame(summaries)
    
    # Sort for logical progression in chart
    sort_order = {"baseline_llm_only": 0, "rag_only": 1, "full_pipeline": 2}
    df["sort_val"] = df["ablation"].map(sort_order)
    df = df.sort_values("sort_val")
    
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Accuracy comparison
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df, x="ablation", y="accuracy", palette="Blues_d")
    plt.title("Diagnostic Accuracy across Configurations (MedQA-USMLE)")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Configuration")
    plt.ylim(0, 1.0)
    
    # Add percentage labels
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{p.get_height():.1%}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
                   
    plt.tight_layout()
    plt.savefig(results_dir / "accuracy_ablation.png")
    print("Saved accuracy_ablation.png")
    
    # Plot 2: Confidence
    plt.figure(figsize=(10, 6))
    ax2 = sns.barplot(data=df, x="ablation", y="mean_confidence", palette="Greens_d")
    plt.title("System Confidence Score across Configurations")
    plt.ylabel("Mean Confidence (0-1)")
    plt.xlabel("Configuration")
    plt.ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(results_dir / "confidence_ablation.png")
    print("Saved confidence_ablation.png")


if __name__ == "__main__":
    generate_plots()
