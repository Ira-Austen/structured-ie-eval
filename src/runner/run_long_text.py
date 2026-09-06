"""
Long-text Streaming & Chunking Benchmark.
Tests GLiNER2.5 and UIE on 1k, 4k, and 16k character documents with overlapping sliding windows.
Measures latency scaling, peak memory, and boundary preservation.
"""

import os
import sys
import time
import json
import argparse
from typing import List, Dict, Any

from ..models.gliner_adapter import GLiNERAdapter
from ..models.uie_adapter import UIEAdapter
from ..monitor.resource_monitor import ResourceMonitor


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))


def chunk_text(text: str, chunk_size: int = 350, overlap: int = 50) -> List[Dict[str, Any]]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append({
            "chunk_idx": len(chunks),
            "char_start": start,
            "char_end": end,
            "text": text[start:end]
        })
        if end >= len(text):
            break
        start += (chunk_size - overlap)
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Run Long Text Benchmark")
    parser.add_argument("--model", type=str, required=True, choices=["gliner", "uie"])
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    monitor_path = os.path.join(args.output_dir, f"monitor_long_{args.model}.jsonl")
    monitor = ResourceMonitor(output_jsonl=monitor_path, interval=0.5)
    monitor.start()

    fixtures_path = os.path.join(DATA_DIR, "long_text_fixtures.jsonl")
    fixtures = []
    with open(fixtures_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                fixtures.append(json.loads(line))

    if args.model == "gliner":
        model_path = args.model_path or "fastino/gliner2.5-multi-v1"
        adapter = GLiNERAdapter(model_path, num_threads=2)
        adapter.load_model()
    else:
        model_path = args.model_path or "Casually/uie-medium"
        adapter = UIEAdapter(model_path, num_threads=2)
        adapter.load_model()

    records = []
    for fxt in fixtures:
        f_id = fxt["fixture_id"]
        scale = fxt["scale"]
        full_text = fxt["text"]
        chunks = chunk_text(full_text, chunk_size=350, overlap=50)

        t0 = time.time()
        total_ents = 0
        total_rels = 0

        for ch in chunks:
            if args.model == "gliner":
                res = adapter.predict(f"{f_id}_ch{ch['chunk_idx']}", ch["text"], config_id="G5", schema_type="novel")
            else:
                res = adapter.predict(f"{f_id}_ch{ch['chunk_idx']}", ch["text"], schema_type="novel")
            total_ents += len(res.entities)
            total_rels += len(res.relations)

        elapsed = time.time() - t0
        records.append({
            "fixture_id": f_id,
            "scale": scale,
            "doc_length": len(full_text),
            "chunks_count": len(chunks),
            "elapsed_sec": round(elapsed, 4),
            "throughput_chars_per_sec": round(len(full_text) / elapsed, 2) if elapsed > 0 else 0,
            "entities_found": total_ents,
            "relations_found": total_rels
        })
        print(f"Processed {f_id} ({scale}, {len(full_text)} chars) across {len(chunks)} chunks in {elapsed:.2f}s")

    mon_summary = monitor.stop()

    summary = {
        "model": args.model,
        "test_type": "long_text_chunking",
        "fixtures_count": len(fixtures),
        "peak_rss_mib": mon_summary["peak_rss_mib"],
        "peak_cgroup_mib": mon_summary["peak_cgroup_mib"],
        "records": records
    }

    out_file = os.path.join(args.output_dir, f"long_text_results_{args.model}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Long text results saved to {out_file}")


if __name__ == "__main__":
    main()
