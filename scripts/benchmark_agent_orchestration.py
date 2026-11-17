"""
scripts/benchmark_agent_orchestration.py
Benchmarks Sequential vs Parallel multi-agent orchestration on target hardware.

Evaluation Protocol:
- Samples 100 representative policy claims from dataset/socioeconomic/test_socioeconomic.jsonl
- Runs 10 untracked warm-up iterations to prime PyTorch/ONNX caches and HNSW graphs
- Measures Sequential Pipeline (Agents A -> B -> C -> D -> E)
- Measures Parallel Pipeline (Agents A || B || C concurrently, followed by D -> E)
- Records P50, P90, P99, Mean, StdDev wall-clock latency
- Monitors peak CPU utilization and RSS memory delta via psutil
- Emits structured results to models/orchestration_benchmark_results.json
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

try:
    import psutil
    has_psutil = True
except ImportError:
    psutil = None
    has_psutil = False

# Add backend to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "backend"))

from orchestrator import OrchestrationWorkflow

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

TEST_JSONL = BASE_DIR / "dataset" / "socioeconomic" / "test_socioeconomic.jsonl"
OUTPUT_JSON = BASE_DIR / "models" / "orchestration_benchmark_results.json"


def load_test_claims(n: int = 100) -> List[str]:
    assert TEST_JSONL.exists(), f"Test set not found: {TEST_JSONL}"
    claims = []
    with open(TEST_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                claims.append(rec["claim"])
                if len(claims) >= n:
                    break
    return claims


def run_benchmark(mode: str, claims: List[str], orchestrator: OrchestrationWorkflow) -> Dict[str, Any]:
    latencies = []
    cpu_readings = []
    mem_before = (psutil.Process().memory_info().rss / (1024 * 1024)) if has_psutil else 0.0

    logger.info(f"Running benchmark in [{mode.upper()}] mode over {len(claims)} claims...")

    for idx, claim in enumerate(claims, start=1):
        t0 = time.perf_counter()
        if has_psutil:
            cpu_readings.append(psutil.cpu_percent(interval=None))

        if mode == "parallel":
            res = orchestrator.analyze_sync(claim, top_k_evidence=3, top_k_influence=2)
        else:
            # Sequential mode execution
            res = orchestrator.analyze_sequential(claim, top_k_evidence=3, top_k_influence=2)

        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # in milliseconds

        if idx % 25 == 0 or idx == len(claims):
            logger.info(f"[{mode.upper()}] Progress: {idx}/{len(claims)} claims processed (Current: {latencies[-1]:.1f}ms)")

    mem_after = (psutil.Process().memory_info().rss / (1024 * 1024)) if has_psutil else 0.0
    mem_delta = max(0.0, mem_after - mem_before) if has_psutil else 0.0

    arr = np.array(latencies)
    return {
        "mode": mode,
        "sample_size": len(claims),
        "mean_ms": round(float(np.mean(arr)), 2),
        "std_ms": round(float(np.std(arr)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "max_ms": round(float(np.max(arr)), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p90_ms": round(float(np.percentile(arr, 90)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "avg_cpu_percent": round(float(np.mean(cpu_readings)), 2) if cpu_readings else 0.0,
        "peak_cpu_percent": round(float(np.max(cpu_readings)), 2) if cpu_readings else 0.0,
        "rss_memory_delta_mb": round(float(mem_delta), 2),
    }


def main():
    claims = load_test_claims(100)
    logger.info(f"Loaded {len(claims)} claims from test set.")

    orchestrator = OrchestrationWorkflow(max_workers=3)

    # 1. Warm-up (10 untracked runs)
    logger.info("Executing 10 untracked warm-up iterations to prime caches...")
    for w_claim in claims[:10]:
        _ = orchestrator.analyze_sync(w_claim)
    logger.info("Warm-up complete.")

    # 2. Sequential Benchmark
    seq_metrics = run_benchmark("sequential", claims, orchestrator)

    # 3. Parallel Benchmark
    par_metrics = run_benchmark("parallel", claims, orchestrator)

    speedup = round(seq_metrics["mean_ms"] / par_metrics["mean_ms"], 2) if par_metrics["mean_ms"] > 0 else 1.0
    p90_speedup = round(seq_metrics["p90_ms"] / par_metrics["p90_ms"], 2) if par_metrics["p90_ms"] > 0 else 1.0

    selected_default = "parallel" if par_metrics["mean_ms"] <= seq_metrics["mean_ms"] else "sequential"

    benchmark_bundle = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": {
            "cpu_count_logical": psutil.cpu_count(logical=True) if has_psutil else os.cpu_count(),
            "cpu_count_physical": psutil.cpu_count(logical=False) if has_psutil else os.cpu_count(),
            "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2) if has_psutil else "N/A",
        },
        "sequential": seq_metrics,
        "parallel": par_metrics,
        "speedup_ratio_mean": speedup,
        "speedup_ratio_p90": p90_speedup,
        "recommended_default_mode": selected_default,
        "verdict": (
            f"Parallel execution is {speedup}x faster on average ({p90_speedup}x at P90) on this hardware."
            if speedup >= 1.05
            else "Sequential and Parallel have comparable performance under CPU GIL; both modes fully supported."
        ),
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(benchmark_bundle, f, indent=2)

    logger.info(f"Benchmark results saved to: {OUTPUT_JSON}")
    logger.info("\n" + json.dumps(benchmark_bundle, indent=2))


if __name__ == "__main__":
    main()
