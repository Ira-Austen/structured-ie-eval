"""
Optimization Verification & Speed Limit Benchmark.
Measures GLiNER2.5 extraction throughput without cascade pre-filtering across:
1. Sequential Baseline (Original G5: 3 forward passes, batch=1, 1 thread)
2. Unified Schema Single-Pass (1 forward pass, batch=1, 1 thread)
3. OpenMP Thread Scaling (Unified Schema, batch=1, 2 & 4 threads)
4. CPU Batching Sweep (Unified Schema, 4 threads, batch_size=4, 8, 16, 32)
5. Chunk & Stride Tuning (chunk=448, stride=400, batch=16, 4 threads)
6. Dynamic INT8 Quantization (Linear INT8, batch=16, 4 threads)
"""

import os
import sys
import time
import json
import argparse
import psutil
import numpy as np
from typing import List, Dict, Any

from ..models.gliner_adapter import GLiNERAdapter
from ..monitor.resource_monitor import ResourceMonitor


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))


def load_benchmark_text(max_chars: int = 40000) -> str:
    """Load novel text from data/doupo_full.txt or dev set."""
    full_path = os.path.join(DATA_DIR, "doupo_full.txt")
    if os.path.exists(full_path):
        with open(full_path, "r", encoding="utf-8") as f:
            text = f.read(max_chars)
            return text
    # Fallback to dev set
    dev_path = os.path.join(DATA_DIR, "novel_dev_24.jsonl")
    texts = []
    with open(dev_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                texts.append(json.loads(line)["text"])
    combined = "\n".join(texts)
    return combined[:max_chars]


def slice_chunks(text: str, chunk_size: int = 350, overlap: int = 50) -> List[Dict[str, Any]]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append({
            "chunk_idx": len(chunks),
            "start": start,
            "end": end,
            "text": text[start:end]
        })
        if end >= len(text):
            break
        start += (chunk_size - overlap)
    return chunks


def run_benchmark(model_path: str, max_chars: int, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    raw_text = load_benchmark_text(max_chars=max_chars)
    print(f"Loaded benchmark novel text: {len(raw_text)} characters")

    results = []
    baseline_throughput = 0.0

    # -------------------------------------------------------------
    # Tier 0: Sequential Baseline (Original G5: 3 forward passes, batch=1, 1T)
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("[Tier 0] Sequential Baseline (Original G5: 3 passes, batch=1, 1 thread)")
    print("="*70)
    adapter_t0 = GLiNERAdapter(model_path, num_threads=1)
    adapter_t0.load_model()

    chunks_t0 = slice_chunks(raw_text, chunk_size=350, overlap=50)[:25]
    total_chars_t0 = sum(len(c["text"]) for c in chunks_t0)

    # Warmup
    for c in chunks_t0[:2]:
        adapter_t0.predict("warmup", c["text"], config_id="G5", schema_type="novel")

    latencies_t0 = []
    t0_start = time.time()
    ents_t0, rels_t0, recs_t0 = 0, 0, 0
    for c in chunks_t0:
        tc0 = time.time()
        res = adapter_t0.predict(f"t0_{c['chunk_idx']}", c["text"], config_id="G5", schema_type="novel")
        latencies_t0.append((time.time() - tc0) * 1000)
        ents_t0 += len(res.entities)
        rels_t0 += len(res.relations)
        recs_t0 += len(res.records)

    elapsed_t0 = time.time() - t0_start
    throughput_t0 = round(total_chars_t0 / elapsed_t0, 2)
    baseline_throughput = throughput_t0

    p50_t0 = round(float(np.percentile(latencies_t0, 50)), 2)
    p95_t0 = round(float(np.percentile(latencies_t0, 95)), 2)
    mem_t0 = round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2)

    tier0_res = {
        "tier": "Tier 0: Sequential Baseline (Original G5)",
        "passes_per_chunk": 3,
        "batch_size": 1,
        "threads": 1,
        "chunk_size": 350,
        "chunks_evaluated": len(chunks_t0),
        "total_chars": total_chars_t0,
        "elapsed_sec": round(elapsed_t0, 2),
        "chars_per_sec": throughput_t0,
        "latency_p50_ms": p50_t0,
        "latency_p95_ms": p95_t0,
        "speedup": 1.0,
        "peak_ram_mib": mem_t0,
        "entities_found": ents_t0,
        "relations_found": rels_t0,
        "records_found": recs_t0
    }
    results.append(tier0_res)
    print(f"-> Throughput: {throughput_t0} chars/s | Latency P50: {p50_t0}ms | Speedup: 1.0x")

    # -------------------------------------------------------------
    # Tier 1: Unified Schema (Single-Pass, batch=1, 1T)
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("[Tier 1] Unified Schema (Single-Pass Forward, batch=1, 1 thread)")
    print("="*70)
    latencies_t1 = []
    t1_start = time.time()
    ents_t1, rels_t1, recs_t1 = 0, 0, 0
    for c in chunks_t0:
        tc0 = time.time()
        res = adapter_t0.predict_unified(f"t1_{c['chunk_idx']}", c["text"], schema_type="novel")
        latencies_t1.append((time.time() - tc0) * 1000)
        ents_t1 += len(res.entities)
        rels_t1 += len(res.relations)
        recs_t1 += len(res.records)

    elapsed_t1 = time.time() - t1_start
    throughput_t1 = round(total_chars_t0 / elapsed_t1, 2)
    speedup_t1 = round(throughput_t1 / max(baseline_throughput, 1), 2)
    p50_t1 = round(float(np.percentile(latencies_t1, 50)), 2)
    p95_t1 = round(float(np.percentile(latencies_t1, 95)), 2)

    tier1_res = {
        "tier": "Tier 1: Unified Single-Pass Schema",
        "passes_per_chunk": 1,
        "batch_size": 1,
        "threads": 1,
        "chunk_size": 350,
        "chunks_evaluated": len(chunks_t0),
        "total_chars": total_chars_t0,
        "elapsed_sec": round(elapsed_t1, 2),
        "chars_per_sec": throughput_t1,
        "latency_p50_ms": p50_t1,
        "latency_p95_ms": p95_t1,
        "speedup": speedup_t1,
        "peak_ram_mib": mem_t0,
        "entities_found": ents_t1,
        "relations_found": rels_t1,
        "records_found": recs_t1
    }
    results.append(tier1_res)
    print(f"-> Throughput: {throughput_t1} chars/s | Latency P50: {p50_t1}ms | Speedup: {speedup_t1}x")

    # -------------------------------------------------------------
    # Tier 2: OpenMP Multi-Threading (Unified Schema, 4T)
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("[Tier 2] Multi-Threading Scaling (Unified Schema, batch=1, 4 threads)")
    print("="*70)
    adapter_t2 = GLiNERAdapter(model_path, num_threads=4)
    adapter_t2.load_model()

    latencies_t2 = []
    t2_start = time.time()
    for c in chunks_t0:
        tc0 = time.time()
        res = adapter_t2.predict_unified(f"t2_{c['chunk_idx']}", c["text"], schema_type="novel")
        latencies_t2.append((time.time() - tc0) * 1000)

    elapsed_t2 = time.time() - t2_start
    throughput_t2 = round(total_chars_t0 / elapsed_t2, 2)
    speedup_t2 = round(throughput_t2 / max(baseline_throughput, 1), 2)
    p50_t2 = round(float(np.percentile(latencies_t2, 50)), 2)
    p95_t2 = round(float(np.percentile(latencies_t2, 95)), 2)

    tier2_res = {
        "tier": "Tier 2: Multi-Threading (4 Threads)",
        "passes_per_chunk": 1,
        "batch_size": 1,
        "threads": 4,
        "chunk_size": 350,
        "chunks_evaluated": len(chunks_t0),
        "total_chars": total_chars_t0,
        "elapsed_sec": round(elapsed_t2, 2),
        "chars_per_sec": throughput_t2,
        "latency_p50_ms": p50_t2,
        "latency_p95_ms": p95_t2,
        "speedup": speedup_t2,
        "peak_ram_mib": round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2),
        "entities_found": ents_t1,
        "relations_found": rels_t1,
        "records_found": recs_t1
    }
    results.append(tier2_res)
    print(f"-> Throughput: {throughput_t2} chars/s | Latency P50: {p50_t2}ms | Speedup: {speedup_t2}x")

    # -------------------------------------------------------------
    # Tier 3: CPU Batching Sweep (batch_size = 4, 8, 16, 4 threads)
    # -------------------------------------------------------------
    chunks_sweep = slice_chunks(raw_text, chunk_size=350, overlap=50)[:48]
    total_chars_sweep = sum(len(c["text"]) for c in chunks_sweep)
    sample_ids_sweep = [f"sw_{c['chunk_idx']}" for c in chunks_sweep]
    texts_sweep = [c["text"] for c in chunks_sweep]

    for bs in [4, 8, 16]:
        print("\n" + "="*70)
        print(f"[Tier 3] CPU Batching (batch_size={bs}, 4 threads)")
        print("="*70)
        t3_start = time.time()
        res_list = adapter_t2.batch_predict_unified(
            sample_ids_sweep,
            texts_sweep,
            schema_type="novel",
            batch_size=bs
        )
        elapsed_t3 = time.time() - t3_start
        throughput_t3 = round(total_chars_sweep / elapsed_t3, 2)
        speedup_t3 = round(throughput_t3 / max(baseline_throughput, 1), 2)
        per_chunk_ms = round((elapsed_t3 / len(texts_sweep)) * 1000, 2)

        tier3_res = {
            "tier": f"Tier 3: CPU Batching (BS={bs}, 4T)",
            "passes_per_chunk": 1,
            "batch_size": bs,
            "threads": 4,
            "chunk_size": 350,
            "chunks_evaluated": len(texts_sweep),
            "total_chars": total_chars_sweep,
            "elapsed_sec": round(elapsed_t3, 2),
            "chars_per_sec": throughput_t3,
            "latency_p50_ms": per_chunk_ms,
            "latency_p95_ms": per_chunk_ms,
            "speedup": speedup_t3,
            "peak_ram_mib": round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2),
            "entities_found": sum(len(r.entities) for r in res_list),
            "relations_found": sum(len(r.relations) for r in res_list),
            "records_found": sum(len(r.records) for r in res_list)
        }
        results.append(tier3_res)
        print(f"-> Throughput: {throughput_t3} chars/s | Avg/Chunk: {per_chunk_ms}ms | Speedup: {speedup_t3}x")

    # -------------------------------------------------------------
    # Tier 4: Chunk & Stride Tuning (chunk=448, overlap=48, BS=16, 4T)
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("[Tier 4] Window Stride Optimization (chunk=448, overlap=48, BS=16, 4T)")
    print("="*70)
    chunks_t4 = slice_chunks(raw_text, chunk_size=448, overlap=48)[:48]
    total_chars_t4 = sum(len(c["text"]) for c in chunks_t4)
    sids_t4 = [f"t4_{c['chunk_idx']}" for c in chunks_t4]
    texts_t4 = [c["text"] for c in chunks_t4]

    t4_start = time.time()
    res_list_t4 = adapter_t2.batch_predict_unified(
        sids_t4,
        texts_t4,
        schema_type="novel",
        batch_size=16
    )
    elapsed_t4 = time.time() - t4_start
    throughput_t4 = round(total_chars_t4 / elapsed_t4, 2)
    speedup_t4 = round(throughput_t4 / max(baseline_throughput, 1), 2)
    per_chunk_t4_ms = round((elapsed_t4 / len(texts_t4)) * 1000, 2)

    tier4_res = {
        "tier": "Tier 4: Window Stride Opt (chunk=448, BS=16, 4T)",
        "passes_per_chunk": 1,
        "batch_size": 16,
        "threads": 4,
        "chunk_size": 448,
        "chunks_evaluated": len(texts_t4),
        "total_chars": total_chars_t4,
        "elapsed_sec": round(elapsed_t4, 2),
        "chars_per_sec": throughput_t4,
        "latency_p50_ms": per_chunk_t4_ms,
        "latency_p95_ms": per_chunk_t4_ms,
        "speedup": speedup_t4,
        "peak_ram_mib": round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2),
        "entities_found": sum(len(r.entities) for r in res_list_t4),
        "relations_found": sum(len(r.relations) for r in res_list_t4),
        "records_found": sum(len(r.records) for r in res_list_t4)
    }
    results.append(tier4_res)
    print(f"-> Throughput: {throughput_t4} chars/s | Avg/Chunk: {per_chunk_t4_ms}ms | Speedup: {speedup_t4}x")

    # -------------------------------------------------------------
    # Tier 5: INT8 Dynamic Quantization (Linear INT8, BS=16, 4T)
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("[Tier 5] Dynamic INT8 Quantization (Linear INT8, BS=16, 4T)")
    print("="*70)
    quant_ok = adapter_t2.enable_dynamic_quantization()
    if quant_ok:
        t5_start = time.time()
        res_list_t5 = adapter_t2.batch_predict_unified(
            sids_t4,
            texts_t4,
            schema_type="novel",
            batch_size=16
        )
        elapsed_t5 = time.time() - t5_start
        throughput_t5 = round(total_chars_t4 / elapsed_t5, 2)
        speedup_t5 = round(throughput_t5 / max(baseline_throughput, 1), 2)
        per_chunk_t5_ms = round((elapsed_t5 / len(texts_t4)) * 1000, 2)

        tier5_res = {
            "tier": "Tier 5: INT8 Dynamic Quantization (BS=16, 4T)",
            "passes_per_chunk": 1,
            "batch_size": 16,
            "threads": 4,
            "chunk_size": 448,
            "chunks_evaluated": len(texts_t4),
            "total_chars": total_chars_t4,
            "elapsed_sec": round(elapsed_t5, 2),
            "chars_per_sec": throughput_t5,
            "latency_p50_ms": per_chunk_t5_ms,
            "latency_p95_ms": per_chunk_t5_ms,
            "speedup": speedup_t5,
            "peak_ram_mib": round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2),
            "entities_found": sum(len(r.entities) for r in res_list_t5),
            "relations_found": sum(len(r.relations) for r in res_list_t5),
            "records_found": sum(len(r.records) for r in res_list_t5)
        }
        results.append(tier5_res)
        print(f"-> Throughput: {throughput_t5} chars/s | Avg/Chunk: {per_chunk_t5_ms}ms | Speedup: {speedup_t5}x")

    # Save output
    out_file = os.path.join(output_dir, "speed_limit_benchmark.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "test_name": "GLiNER2.5 Speed Limit Optimization Verification (No Cascade Filtering)",
            "baseline_throughput_chars_per_sec": baseline_throughput,
            "tiers": results
        }, f, ensure_ascii=False, indent=2)

    print("\n" + "="*80)
    print("SPEED LIMIT BENCHMARK SUMMARY (NO CASCADE FILTERING)")
    print("="*80)
    header = f"{'Optimization Tier':<42} | {'BS':<4} | {'Thr':<3} | {'Chars/s':<9} | {'Avg Lat':<9} | {'Speedup':<8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['tier']:<42} | {r['batch_size']:<4} | {r['threads']:<3} | {r['chars_per_sec']:<9} | {str(r['latency_p50_ms'])+'ms':<9} | {str(r['speedup'])+'x':<8}")
    print("="*80)
    print(f"Detailed JSON results written to {out_file}\n")


def main():
    parser = argparse.ArgumentParser(description="Run Speed Limit Benchmark")
    parser.add_argument("--model-path", type=str, default="fastino/gliner2.5-multi-v1")
    parser.add_argument("--max-chars", type=int, default=30000)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    run_benchmark(args.model_path, args.max_chars, args.output_dir)


if __name__ == "__main__":
    main()
