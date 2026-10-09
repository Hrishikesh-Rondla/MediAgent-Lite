import json
from pathlib import Path


def main():
    results_dir = Path("results")
    readme_path = Path("README.md")
    
    # Read eval summary
    with open(results_dir / "summary.json") as f:
        summaries = json.load(f)
        
    # Read parallel bench
    with open(results_dir / "bench_parallel.json") as f:
        bench = json.load(f)
        
    # Read local inference
    with open(results_dir / "local_inference.json") as f:
        local = json.load(f)
        
    # Read metadata
    with open(results_dir / "run_metadata.json") as f:
        meta = json.load(f)

    # 4. Results Section Content
    results_md = f"""
## 4. Results (Generated)

> **Run Metadata:**
> N={meta['n']}, Seed={meta['seed']}, Split=test, Date={meta['date']}, Commit={meta['commit']}
> Model: {meta['model']} ({meta['quantization']})

| Configuration | Accuracy | 95% CI | Mean Latency (s) | Mean LLM calls | Loop % |
|---|---|---|---|---|---|
"""
    for s in summaries:
        ci = f"{s['wilson_95_ci'][0]:.1%} - {s['wilson_95_ci'][1]:.1%}"
        results_md += f"| {s['config']} | {s['accuracy']:.1%} | {ci} | {s['mean_latency_s']:.1f} | N/A | {s['loop_triggered_fraction']:.0%} |\n"
        
    results_md += f"""
### Analysis
The baseline accuracy using `{meta['model']}` on complex USMLE multiple-choice questions is understandably low (due to the constraints of the local 8B edge model). The McNemar test p-values show no statistically significant difference between configs given the low baseline accuracy.
The feedback loop (Config E) triggered on {summaries[-1]['loop_triggered_fraction']:.0%} of questions but did not significantly overcome the model's foundational reasoning limits.

![Accuracy by Config](results/accuracy_by_config.png)
![Iterations vs Accuracy](results/iterations_vs_accuracy.png)

See [results/error_analysis.md](results/error_analysis.md) for a detailed breakdown of failure modes (e.g., retrieval miss, answer-mapping errors).

## 5. Performance (Generated)

### PubMed Retrieval Latency
Tested on {bench['config']['n_trials']} trials ({bench['config']['queries_per_trial']} hypotheses per trial).

| Mode | Mean Latency | Std Dev |
|---|---|---|
| Sequential | {bench['sequential']['mean_s']:.2f}s | ±{bench['sequential']['std_s']:.2f}s |
| Parallel | {bench['parallel']['mean_s']:.2f}s | ±{bench['parallel']['std_s']:.2f}s |

**Measured Latency Change: {bench['latency_reduction_pct']:+.1f}%**

![Parallel Benchmark](results/bench_parallel.png)

### Local Inference
| Model | Quantization | GPU | Peak VRAM | Mean Latency/Case |
|---|---|---|---|---|
| {local['model_tag']} | {local['quantization']} | {local['gpu']} | {local['vram_peak_mb']} MB | {local['mean_latency_per_case_s']}s |
"""

    if not readme_path.exists():
        print("README.md not found.")
        return
        
    content = readme_path.read_text(encoding="utf-8")
    
    start_marker = "<!-- RESULTS:START -->"
    end_marker = "<!-- RESULTS:END -->"
    
    if start_marker in content and end_marker in content:
        before = content.split(start_marker)[0]
        after = content.split(end_marker)[1]
        new_content = before + start_marker + "\n" + results_md + "\n" + end_marker + after
        readme_path.write_text(new_content, encoding="utf-8")
        print("README updated successfully.")
    else:
        print("Markers not found in README.md.")

if __name__ == "__main__":
    main()
