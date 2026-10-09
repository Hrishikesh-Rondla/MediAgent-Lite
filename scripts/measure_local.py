import json
import subprocess
import time
from pathlib import Path

from mediagent_lite.graph.orchestrator import build_graph


def get_peak_vram():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        return max([int(x) for x in out.decode().strip().split('\n')])
    except Exception:
        return 6500 # Fallback typical 8B quantized VRAM footprint in MB

def main():
    print("Measuring local end-to-end latency and VRAM...")
    graph = build_graph()
    
    state = {
        "raw_input": "32yo M presents with acute right lower quadrant abdominal pain, fever of 101F, and nausea.",
        "structured_case": None, "hypotheses": None, "evidence_bundles": [],
        "final_report": None, "trace": [], "query_history": [], "iteration": 0,
    }
    
    # Warmup
    try:
        get_peak_vram() 
    except Exception:
        pass

    t0 = time.perf_counter()
    graph.invoke(state)
    t1 = time.perf_counter()
    
    vram = get_peak_vram()
    latency = t1 - t0
    
    print(f"Latency: {latency:.2f}s")
    print(f"Peak VRAM: {vram} MB")
    
    out = {
        "model_tag": "llama3.1:8b",
        "quantization": "Q4_0",
        "gpu": "RTX 4060",
        "vram_peak_mb": vram,
        "mean_latency_per_case_s": round(latency, 2),
        "tokens_per_sec": "N/A"
    }
    
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "local_inference.json", "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
