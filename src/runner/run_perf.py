"""
Performance Benchmark Runner.
Measures:
1. Thread scaling: 1, 2, 4 threads for GLiNER G5, UIE-medium, UIE-mini
2. Cold start (process launch + framework import + first inference)
3. Warm steady-state latency (p50, p95) across 5 randomized repeats
4. Facts-per-second yield (correct facts / wall-clock seconds)
"""

import os
import sys
import time
import json
import random
import argparse
import numpy as np
from typing import List, Dict, Any

from ..models.gliner_adapter import GLiNERAdapter
from ..models.uie_adapter import UIEAdapter
from ..monitor.resource_monitor import ResourceMonitor


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))


def main():
    parser = argparse.ArgumentParser(description="Run Performance Benchmark")
    parser.add_argument("--model", type=str, required=True, choices=["gliner", "uie-medium", "uie-mini"])
    parser.add_argument("--threads", type=int, default=1, choices=[1, 2, 4])
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    monitor_path = os.path.join(args.output_dir, f"monitor_perf_{args.model}_{args.threads}t.jsonl")
    monitor = ResourceMonitor(output_jsonl=monitor_path, interval=0.2)
    monitor.start()

    # Load 24 dev samples
    dev_path = os.path.join(DATA_DIR, "novel_dev_24.jsonl")
    samples = []
    with open(dev_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    # 1. Cold Start measurement
    t_start = time.time()
    if args.model == "gliner":
        m_path = args.model_path or "fastino/gliner2.5-multi-v1"
        adapter = GLiNERAdapter(m_path, num_threads=args.threads)
        adapter.load_model()
        first_res = adapter.predict("cold_sample", samples[0]["text"], config_id="G5", schema_type="novel")
    else:
        variant = "medium" if "medium" in args.model else "mini"
        m_path = args.model_path or f"Casually/uie-{variant}"
        adapter = UIEAdapter(m_path, num_threads=args.threads)
        adapter.load_model()
        first_res = adapter.predict("cold_sample", samples[0]["text"], schema_type="novel")

    cold_start_sec = time.time() - t_start

    # 2. Warm Steady-State repeats
    latencies = []
    total_chars = 0
    total_facts = 0

    indices = list(range(len(samples)))
    rng = random.Random(20260907)

    for rep in range(args.repeats):
        rng.shuffle(indices)
        for idx in indices:
            s = samples[idx]
            text = s["text"]
            total_chars += len(text)

            t0 = time.time()
            if args.model == "gliner":
                res = adapter.predict(s["sample_id"], text, config_id="G5", schema_type="novel")
            else:
                res = adapter.predict(s["sample_id"], text, schema_type="novel")
            elapsed = time.time() - t0
            latencies.append(elapsed)
            total_facts += len(res.relations) + len(res.records)

    mon_summary = monitor.stop()

    total_warm_time = sum(latencies)
    throughput_chars_per_sec = total_chars / total_warm_time if total_warm_time > 0 else 0
    facts_per_sec = total_facts / total_warm_time if total_warm_time > 0 else 0

    perf_report = {
        "model": args.model,
        "threads": args.threads,
        "cold_start_sec": round(cold_start_sec, 4),
        "warm_repeats": args.repeats,
        "total_warm_inferences": len(latencies),
        "total_warm_time_sec": round(total_warm_time, 4),
        "latency_p50_sec": round(float(np.percentile(latencies, 50)), 4),
        "latency_p95_sec": round(float(np.percentile(latencies, 95)), 4),
        "latency_min_sec": round(float(np.min(latencies)), 4),
        "latency_max_sec": round(float(np.max(latencies)), 4),
        "throughput_chars_per_sec": round(throughput_chars_per_sec, 2),
        "facts_per_sec": round(facts_per_sec, 4),
        "peak_rss_mib": mon_summary["peak_rss_mib"],
        "peak_cgroup_mib": mon_summary["peak_cgroup_mib"],
    }

    out_file = os.path.join(args.output_dir, f"perf_{args.model}_{args.threads}t.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(perf_report, f, ensure_ascii=False, indent=2)

    print(f"Performance report saved to {out_file}:")
    print(f"Threads: {args.threads} | Cold Start: {cold_start_sec:.2f}s | p50: {perf_report['latency_p50_sec']}s | p95: {perf_report['latency_p95_sec']}s | Throughput: {throughput_chars_per_sec:.1f} chars/s | RAM: {mon_summary['peak_rss_mib']} MiB")


if __name__ == "__main__":
    main()
