"""scripts/bench_parallel.py

Measures PubMed retrieval latency: sequential vs ThreadPoolExecutor.

Usage:
    python scripts/bench_parallel.py [--trials N]

Output:
    results/bench_parallel.json  — raw timings + summary statistics
    Prints a summary table to stdout.

Notes:
- PubMed cache is disabled for this benchmark (fresh temp dir used).
- A warm-up trial is run and discarded before measurements begin.
- max_workers=3 matches the production orchestrator setting.
- The reported speedup is the measured ratio, not a theoretical value.
"""

import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean, stdev

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mediagent_lite.rag.pubmed_fetcher import PubMedFetcher


# Fixed set of queries representative of typical evidence scanner calls
QUERIES = [
    "Myocardial Infarction chest pain",
    "Pulmonary Embolism dyspnea",
    "Ischemic Stroke weakness",
    "Pneumonia fever cough",
    "Appendicitis abdominal pain",
    "Pulmonary Embolism chest pain",
    "Hypertensive Crisis headache",
    "Aortic Dissection back pain",
    "Sepsis fever hypotension",
    "Meningitis headache fever",
]

# Each trial uses 3 queries to match the production max_workers=3 scenario
QUERIES_PER_TRIAL = 3


def fetch_one(args):
    fetcher, query, cache_dir = args
    fetcher.cache_dir = Path(cache_dir)
    pmids = fetcher.search_pmids(query, retmax=3)
    if pmids:
        fetcher.fetch_abstracts(pmids[:2])


def run_trial_sequential(queries: list[str], cache_dir: str) -> float:
    fetcher = PubMedFetcher()
    start = time.perf_counter()
    for q in queries:
        fetch_one((fetcher, q, cache_dir))
    return time.perf_counter() - start


def run_trial_parallel(queries: list[str], cache_dir: str) -> float:
    fetcher = PubMedFetcher()
    args = [(fetcher, q, cache_dir) for q in queries]
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(fetch_one, args))
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description="Benchmark parallel vs sequential PubMed retrieval")
    parser.add_argument("--trials", type=int, default=10, help="Number of timed trials (warm-up not counted)")
    args = parser.parse_args()

    n_trials = args.trials
    print(f"Benchmark: {n_trials} trials, {QUERIES_PER_TRIAL} queries/trial, max_workers=3")
    print("Cache: DISABLED (fresh temp dir per benchmark run)")
    print("Warm-up: 1 trial (discarded)")
    print("-" * 60)

    seq_times = []
    par_times = []

    for trial in range(-1, n_trials):  # -1 = warm-up
        query_set = QUERIES[(trial % len(QUERIES)):(trial % len(QUERIES)) + QUERIES_PER_TRIAL]
        if len(query_set) < QUERIES_PER_TRIAL:
            query_set = QUERIES[:QUERIES_PER_TRIAL]

        with tempfile.TemporaryDirectory() as tmp_seq:
            seq_t = run_trial_sequential(query_set, tmp_seq)
            
        with tempfile.TemporaryDirectory() as tmp_par:
            par_t = run_trial_parallel(query_set, tmp_par)

        if trial == -1:
            print(f"  Warm-up: seq={seq_t:.2f}s  par={par_t:.2f}s  (discarded)")
            continue

        seq_times.append(seq_t)
        par_times.append(par_t)
        print(f"  Trial {trial+1:2d}: seq={seq_t:.2f}s  par={par_t:.2f}s")

    mean_seq = mean(seq_times)
    mean_par = mean(par_times)
    std_seq = stdev(seq_times) if len(seq_times) > 1 else 0.0
    std_par = stdev(par_times) if len(par_times) > 1 else 0.0
    speedup_pct = ((mean_seq - mean_par) / mean_seq) * 100 if mean_seq > 0 else 0.0

    print("\n" + "=" * 60)
    print("RESULTS")
    print(f"  Sequential (mean ± std): {mean_seq:.2f}s ± {std_seq:.2f}s")
    print(f"  Parallel   (mean ± std): {mean_par:.2f}s ± {std_par:.2f}s")
    print(f"  Latency reduction:       {speedup_pct:+.1f}%")
    print(f"  (positive = parallel is faster; negative = parallel is slower)")
    print("=" * 60)

    results = {
        "config": {
            "n_trials": n_trials,
            "queries_per_trial": QUERIES_PER_TRIAL,
            "max_workers": 3,
            "cache_disabled": True,
        },
        "sequential": {
            "mean_s": round(mean_seq, 3),
            "std_s": round(std_seq, 3),
            "raw_s": [round(t, 3) for t in seq_times],
        },
        "parallel": {
            "mean_s": round(mean_par, 3),
            "std_s": round(std_par, 3),
            "raw_s": [round(t, 3) for t in par_times],
        },
        "latency_reduction_pct": round(speedup_pct, 1),
        "note": (
            "Speedup is bounded by the number of hypotheses (max_workers=3). "
            "On a single GPU with Ollama, LLM calls may serialize on the GPU "
            "even when HTTP calls are parallel."
        ),
    }

    out_dir = Path(__file__).parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "bench_parallel.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
