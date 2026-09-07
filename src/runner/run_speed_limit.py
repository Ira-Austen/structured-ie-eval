"""
Optimization Verification & Speed Limit Benchmark.
Measures GLiNER2.5 extraction throughput without cascade pre-filtering across:
1. Sequential Baseline (Full G5 FP32: JointIE + Natural Records + Attributes + Validation, batch=1, 1 thread)
2. Thread & Batch Grid Sweep (Unified Schema FP32, threads: [1, 2, 4] x batch_size: [1, 4, 8, 16])
3. Isolated INT8 Dynamic Quantization Tier (Encoder-only, batch=16, 4 threads, regression gated)
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
    dev_path = os.path.join(DATA_DIR, "novel_dev_24.jsonl")
    texts = []
    with open(dev_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                texts.append(json.loads(line)["text"])
    combined = "\n".join(texts)
    return combined[:max_chars]


def slice_chunks(text: str, chunk_size: int = 448, overlap: int = 48) -> List[Dict[str, Any]]:
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


def run_benchmark(model_path: str, max_chars: int, output_dir: str, num_chunks: int = 32):
    os.makedirs(output_dir, exist_ok=True)
    raw_text = load_benchmark_text(max_chars=max_chars)
    print(f"Loaded benchmark novel text: {len(raw_text)} characters")

    # Fixed sample chunks across ALL tiers for strict controlled comparison
    chunk_size = 448
    overlap = 48
    all_chunks = slice_chunks(raw_text, chunk_size=chunk_size, overlap=overlap)[:num_chunks]
    
    unique_start = all_chunks[0]["start"]
    unique_end = all_chunks[-1]["end"]
    unique_source_chars = unique_end - unique_start
    total_chunk_chars = sum(len(c["text"]) for c in all_chunks)
    chunk_texts = [c["text"] for c in all_chunks]
    chunk_ids = [f"bench_ck_{c['chunk_idx']}" for c in all_chunks]

    print(f"Benchmark Set: {len(all_chunks)} chunks | Total Chunk Chars: {total_chunk_chars} | Unique Source Chars: {unique_source_chars}")

    results = []

    # -------------------------------------------------------------
    # Tier 0: Sequential Baseline (Full G5 FP32: JointIE + Records + Validation, 1T, BS=1)
    # -------------------------------------------------------------
    print("\n" + "="*75)
    print("[Tier 0] Sequential Baseline (Full G5 FP32: JointIE + Records, batch=1, 1 thread)")
    print("="*75)
    adapter_t0 = GLiNERAdapter(model_path, num_threads=1)
    adapter_t0.load_model()

    # Warmup
    for txt in chunk_texts[:2]:
        adapter_t0.predict("warmup", txt, config_id="G5", schema_type="novel")

    latencies_t0 = []
    ents_t0, rels_t0, recs_t0 = 0, 0, 0
    t0_start = time.time()
    psutil.cpu_percent(interval=None)

    for cid, txt in zip(chunk_ids, chunk_texts):
        tc0 = time.time()
        res = adapter_t0.predict(cid, txt, config_id="G5", schema_type="novel")
        latencies_t0.append((time.time() - tc0) * 1000)
        ents_t0 += len(res.entities)
        rels_t0 += len(res.relations)
        recs_t0 += len(res.records)
        if res.status == "FAILED_RUNTIME":
            print(f"[Error] Tier 0 chunk {cid} failed runtime: {res.error_message}")

    elapsed_t0 = time.time() - t0_start
    cpu_t0 = psutil.cpu_percent(interval=None)
    mem_t0 = round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2)
    swap_t0 = round(psutil.swap_memory().used / (1024 * 1024), 2)

    raw_t0_cps = round(total_chunk_chars / elapsed_t0, 2)
    unique_t0_cps = round(unique_source_chars / elapsed_t0, 2)
    p50_t0 = round(float(np.percentile(latencies_t0, 50)), 2)
    p90_t0 = round(float(np.percentile(latencies_t0, 90)), 2)
    p95_t0 = round(float(np.percentile(latencies_t0, 95)), 2)

    # Baseline Quality Gate
    baseline_qgate = (ents_t0 > 0 and rels_t0 > 0 and recs_t0 > 0)
    print(f"Tier 0 Baseline: {unique_t0_cps} unique c/s ({raw_t0_cps} raw c/s) | "
          f"P50: {p50_t0}ms, P95: {p95_t0}ms | Ents: {ents_t0}, Rels: {rels_t0}, Recs: {recs_t0} | "
          f"Gate: {'PASS' if baseline_qgate else 'FAIL'}")

    tier0_res = {
        "tier": "Tier 0: Full G5 FP32 Baseline (JointIE+Records)",
        "batch_size": 1,
        "threads": 1,
        "chunk_size": chunk_size,
        "chunks_evaluated": len(all_chunks),
        "total_chunk_chars": total_chunk_chars,
        "unique_source_chars": unique_source_chars,
        "elapsed_sec": round(elapsed_t0, 2),
        "raw_chars_per_sec": raw_t0_cps,
        "unique_chars_per_sec": unique_t0_cps,
        "latency_mean_ms": round((elapsed_t0 / len(all_chunks)) * 1000, 2),
        "latency_p50_ms": p50_t0,
        "latency_p90_ms": p90_t0,
        "latency_p95_ms": p95_t0,
        "speedup_vs_baseline": 1.0,
        "cpu_utilization_pct": cpu_t0,
        "peak_rss_mib": mem_t0,
        "swap_used_mib": swap_t0,
        "entities_found": ents_t0,
        "relations_found": rels_t0,
        "records_found": recs_t0,
        "quality_gate": "PASS" if baseline_qgate else "FAIL"
    }
    results.append(tier0_res)

    del adapter_t0

    # -------------------------------------------------------------
    # Grid Sweep: Threads [1, 2, 4] x Batch Sizes [1, 4, 8, 16] (Unified Schema FP32)
    # -------------------------------------------------------------
    print("\n" + "="*75)
    print("[Grid Sweep] Unified Schema FP32 (Threads: 1, 2, 4 x Batch Size: 1, 4, 8, 16)")
    print("="*75)

    thread_configs = [1, 2, 4]
    batch_sizes = [1, 4, 8, 16]

    for threads in thread_configs:
        adapter_sweep = GLiNERAdapter(model_path, num_threads=threads)
        adapter_sweep.load_model()

        # Warmup
        adapter_sweep.batch_predict_unified(["w1", "w2"], chunk_texts[:2], schema_type="novel", batch_size=2)

        for bs in batch_sizes:
            psutil.cpu_percent(interval=None)
            t_grid_start = time.time()
            batch_latencies = []

            # Process in explicit batches to measure real batch latencies
            all_preds = []
            for b_idx in range(0, len(chunk_texts), bs):
                b_ids = chunk_ids[b_idx:b_idx + bs]
                b_txts = chunk_texts[b_idx:b_idx + bs]
                tb0 = time.time()
                b_res = adapter_sweep.batch_predict_unified(b_ids, b_txts, schema_type="novel", batch_size=bs)
                b_elapsed_ms = (time.time() - tb0) * 1000
                per_sample_ms = b_elapsed_ms / len(b_txts)
                for _ in b_txts:
                    batch_latencies.append(per_sample_ms)
                all_preds.extend(b_res)

            elapsed_grid = time.time() - t_grid_start
            cpu_grid = psutil.cpu_percent(interval=None)
            mem_grid = round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2)
            swap_grid = round(psutil.swap_memory().used / (1024 * 1024), 2)

            raw_cps = round(total_chunk_chars / elapsed_grid, 2)
            unique_cps = round(unique_source_chars / elapsed_grid, 2)
            speedup = round(unique_cps / max(unique_t0_cps, 0.001), 2)

            p50 = round(float(np.percentile(batch_latencies, 50)), 2)
            p90 = round(float(np.percentile(batch_latencies, 90)), 2)
            p95 = round(float(np.percentile(batch_latencies, 95)), 2)

            ents_cnt = sum(len(r.entities) for r in all_preds)
            rels_cnt = sum(len(r.relations) for r in all_preds)
            recs_cnt = sum(len(r.records) for r in all_preds)
            qgate = (ents_cnt > 0 and rels_cnt > 0 and recs_cnt > 0)

            tier_name = f"Unified FP32 (Thr={threads}, BS={bs})"
            print(f"-> {tier_name:<30} | {unique_cps:7.1f} unique c/s ({raw_cps:7.1f} raw) | "
                  f"P50: {p50:6.1f}ms, P95: {p95:6.1f}ms | Speedup: {speedup:4.2f}x | "
                  f"Ents: {ents_cnt:4d}, Rels: {rels_cnt:3d}, Recs: {recs_cnt:3d} | "
                  f"CPU: {cpu_grid:4.1f}% | Gate: {'PASS' if qgate else 'FAIL'}")

            results.append({
                "tier": tier_name,
                "batch_size": bs,
                "threads": threads,
                "chunk_size": chunk_size,
                "chunks_evaluated": len(all_chunks),
                "total_chunk_chars": total_chunk_chars,
                "unique_source_chars": unique_source_chars,
                "elapsed_sec": round(elapsed_grid, 2),
                "raw_chars_per_sec": raw_cps,
                "unique_chars_per_sec": unique_cps,
                "latency_mean_ms": round((elapsed_grid / len(all_chunks)) * 1000, 2),
                "latency_p50_ms": p50,
                "latency_p90_ms": p90,
                "latency_p95_ms": p95,
                "speedup_vs_baseline": speedup,
                "cpu_utilization_pct": cpu_grid,
                "peak_rss_mib": mem_grid,
                "swap_used_mib": swap_grid,
                "entities_found": ents_cnt,
                "relations_found": rels_cnt,
                "records_found": recs_cnt,
                "quality_gate": "PASS" if qgate else "FAIL"
            })

        del adapter_sweep

    # -------------------------------------------------------------
    # Isolated INT8 Dynamic Quantization Tier (Encoder-only, 4T, BS=16)
    # -------------------------------------------------------------
    print("\n" + "="*75)
    print("[Experimental] INT8 Dynamic Quantization (Encoder Backbone Only, 4T, BS=16)")
    print("="*75)
    adapter_int8 = GLiNERAdapter(model_path, num_threads=4)
    adapter_int8.load_model()
    quant_ok = adapter_int8.enable_dynamic_quantization()

    if quant_ok:
        psutil.cpu_percent(interval=None)
        t_int8_start = time.time()
        int8_batch_latencies = []
        all_preds_int8 = []

        bs = 16
        for b_idx in range(0, len(chunk_texts), bs):
            b_ids = chunk_ids[b_idx:b_idx + bs]
            b_txts = chunk_texts[b_idx:b_idx + bs]
            tb0 = time.time()
            b_res = adapter_int8.batch_predict_unified(b_ids, b_txts, schema_type="novel", batch_size=bs)
            b_elapsed_ms = (time.time() - tb0) * 1000
            per_sample_ms = b_elapsed_ms / len(b_txts)
            for _ in b_txts:
                int8_batch_latencies.append(per_sample_ms)
            all_preds_int8.extend(b_res)

        elapsed_int8 = time.time() - t_int8_start
        cpu_int8 = psutil.cpu_percent(interval=None)
        mem_int8 = round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2)
        swap_int8 = round(psutil.swap_memory().used / (1024 * 1024), 2)

        raw_cps_int8 = round(total_chunk_chars / elapsed_int8, 2)
        unique_cps_int8 = round(unique_source_chars / elapsed_int8, 2)
        speedup_int8 = round(unique_cps_int8 / max(unique_t0_cps, 0.001), 2)

        p50_int8 = round(float(np.percentile(int8_batch_latencies, 50)), 2)
        p90_int8 = round(float(np.percentile(int8_batch_latencies, 90)), 2)
        p95_int8 = round(float(np.percentile(int8_batch_latencies, 95)), 2)

        ents_int8 = sum(len(r.entities) for r in all_preds_int8)
        rels_int8 = sum(len(r.relations) for r in all_preds_int8)
        recs_int8 = sum(len(r.records) for r in all_preds_int8)

        # Compare against FP32 (4T, BS=16)
        fp32_match = next((r for r in results if r["threads"] == 4 and r["batch_size"] == 16), None)
        fp32_ents = fp32_match["entities_found"] if fp32_match else 1
        retention_rate = round((ents_int8 / max(fp32_ents, 1)) * 100, 1)

        # Quality Gate: Entity retention must be >= 70% and relations/records non-zero
        int8_qgate = (ents_int8 > 0 and rels_int8 > 0 and recs_int8 > 0 and retention_rate >= 70.0)

        print(f"INT8 Experimental: {unique_cps_int8} unique c/s | Speedup: {speedup_int8}x | "
              f"Ents: {ents_int8} (retention: {retention_rate}%), Rels: {rels_int8}, Recs: {recs_int8} | "
              f"Gate: {'PASS' if int8_qgate else 'FAILED_REGRESSION (Do NOT use for full novel)'}")

        results.append({
            "tier": "Experimental: INT8 Encoder Dynamic Quantization (4T, BS=16)",
            "batch_size": 16,
            "threads": 4,
            "chunk_size": chunk_size,
            "chunks_evaluated": len(all_chunks),
            "total_chunk_chars": total_chunk_chars,
            "unique_source_chars": unique_source_chars,
            "elapsed_sec": round(elapsed_int8, 2),
            "raw_chars_per_sec": raw_cps_int8,
            "unique_chars_per_sec": unique_cps_int8,
            "latency_mean_ms": round((elapsed_int8 / len(all_chunks)) * 1000, 2),
            "latency_p50_ms": p50_int8,
            "latency_p90_ms": p90_int8,
            "latency_p95_ms": p95_int8,
            "speedup_vs_baseline": speedup_int8,
            "cpu_utilization_pct": cpu_int8,
            "peak_rss_mib": mem_int8,
            "swap_used_mib": swap_int8,
            "entities_found": ents_int8,
            "relations_found": rels_int8,
            "records_found": recs_int8,
            "retention_rate_pct": retention_rate,
            "quality_gate": "PASS" if int8_qgate else "FAILED_REGRESSION",
            "recommended_for_production": False
        })
        del adapter_int8

    # Save output
    out_file = os.path.join(output_dir, "speed_limit_benchmark.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "test_name": "GLiNER2.5 Speed Benchmark and Grid Sweep (No Cascade Filtering)",
            "baseline_unique_chars_per_sec": unique_t0_cps,
            "baseline_raw_chars_per_sec": raw_t0_cps,
            "unique_source_chars": unique_source_chars,
            "total_chunk_chars": total_chunk_chars,
            "tiers": results
        }, f, ensure_ascii=False, indent=2)

    print("\n" + "="*95)
    print("SPEED LIMIT BENCHMARK & GRID SWEEP SUMMARY (STRICT SAME-SAMPLE & UNIQUE CHARS)")
    print("="*95)
    header = f"{'Configuration':<45} | {'Thr':<3} | {'BS':<3} | {'Unique c/s':<10} | {'P50 (ms)':<9} | {'Speedup':<8} | {'Gate':<6}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['tier']:<45} | {r['threads']:<3} | {r['batch_size']:<3} | {r['unique_chars_per_sec']:<10} | {r['latency_p50_ms']:<9} | {str(r['speedup_vs_baseline'])+'x':<8} | {r['quality_gate']:<6}")
    print("="*95)
    print(f"Results successfully saved to {out_file}\n")


def main():
    parser = argparse.ArgumentParser(description="Run Speed Limit Benchmark")
    parser.add_argument("--model-path", type=str, default="fastino/gliner2.5-multi-v1")
    parser.add_argument("--max-chars", type=int, default=40000)
    parser.add_argument("--num-chunks", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    run_benchmark(
        model_path=args.model_path,
        max_chars=args.max_chars,
        output_dir=args.output_dir,
        num_chunks=args.num_chunks
    )


if __name__ == "__main__":
    main()
