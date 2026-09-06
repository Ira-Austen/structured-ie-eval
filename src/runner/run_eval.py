"""
Main Evaluation Runner for GLiNER2.5 and UIE Structured IE Benchmark.
Supports:
--model: gliner, uie, baseline
--config: G0, G1, G2, G3, G4a, G4b, G5, uie_medium, uie_mini, B0
--dataset: synthetic, cross_dev, cross_test, novel_dev, novel_test, robustness
--threads: 1, 2, 4
--model-path: HF hub id or local path
"""

import os
import sys
import time
import json
import argparse
from typing import List, Dict, Any

from ..models.base import ExtractionResult
from ..models.gliner_adapter import GLiNERAdapter
from ..models.uie_adapter import UIEAdapter
from ..models.baseline_rules import RuleBaselineModel
from ..validation.validator import UnifiedValidator
from ..metrics.evaluator import MetricsEvaluator
from ..monitor.resource_monitor import ResourceMonitor


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))


def load_dataset(dataset_name: str) -> List[Dict[str, Any]]:
    path_map = {
        "synthetic": "synthetic_contrast_pairs.jsonl",
        "cross_dev": "cross_domain_dev.jsonl",
        "cross_test": "cross_domain_test.jsonl",
        "novel_dev": "novel_dev_24.jsonl",
        "novel_test": "novel_test_96.jsonl",
        "robustness": "robustness_edge_cases.jsonl"
    }
    filename = path_map.get(dataset_name)
    if not filename:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    full_path = os.path.join(DATA_DIR, filename)
    items = []
    with open(full_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def main():
    parser = argparse.ArgumentParser(description="Run Structured IE Benchmark")
    parser.add_argument("--model", type=str, required=True, choices=["gliner", "uie", "baseline"])
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    monitor_path = os.path.join(args.output_dir, f"monitor_{args.model}_{args.config}_{args.dataset}_{args.threads}t.jsonl")
    monitor = ResourceMonitor(output_jsonl=monitor_path, interval=0.5)
    monitor.start()

    # Instantiate model
    print(f"Loading model: {args.model} | Config: {args.config} | Threads: {args.threads}...")
    t_load_start = time.time()
    if args.model == "gliner":
        model_path = args.model_path or "fastino/gliner2.5-multi-v1"
        adapter = GLiNERAdapter(model_path, num_threads=args.threads)
        adapter.load_model()
    elif args.model == "uie":
        default_path = "Casually/uie-medium" if "medium" in args.config.lower() else "Casually/uie-mini"
        model_path = args.model_path or default_path
        adapter = UIEAdapter(model_path, num_threads=args.threads)
        adapter.load_model()
    elif args.model == "baseline":
        adapter = RuleBaselineModel()
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    load_duration = time.time() - t_load_start

    # Load dataset
    samples = load_dataset(args.dataset)
    schema_type = "novel" if "novel" in args.dataset else "general"

    print(f"Running inference on {len(samples)} samples from {args.dataset}...")
    results: List[ExtractionResult] = []
    gold_entities_all = []
    gold_relations_all = []
    gold_records_all = []

    pred_entities_all = []
    pred_relations_all = []
    pred_records_all = []

    t_inf_total = 0.0
    total_chars = 0

    for idx, sample in enumerate(samples, 1):
        sid = sample.get("sample_id", f"sample_{idx}")
        text = sample.get("text") or sample.get("window_text", "")
        total_chars += len(text)

        t0 = time.time()
        if args.model == "gliner":
            res = adapter.predict(sid, text, config_id=args.config, schema_type=schema_type)
        elif args.model == "uie":
            res = adapter.predict(sid, text, schema_type=schema_type)
        else:
            res = adapter.predict(sid, text)
        t_elapsed = time.time() - t0
        t_inf_total += t_elapsed

        results.append(res)

        # Collect for metrics
        g_ents = sample.get("gold_entities", [])
        g_rels = sample.get("gold_relations", [])
        g_recs = sample.get("gold_records") or sample.get("gold_events", [])

        gold_entities_all.extend(g_ents)
        gold_relations_all.extend(g_rels)
        gold_records_all.extend(g_recs)

        pred_entities_all.extend([e.__dict__ for e in res.entities])
        pred_relations_all.extend([r.__dict__ for r in res.relations])
        pred_records_all.extend([rec.__dict__ for rec in res.records])

    mon_summary = monitor.stop()

    # Compute evaluation metrics
    ent_metrics = MetricsEvaluator.evaluate_mentions(pred_entities_all, gold_entities_all)
    rel_metrics = MetricsEvaluator.evaluate_directed_relations(pred_relations_all, gold_relations_all)
    rec_metrics = MetricsEvaluator.evaluate_records(pred_records_all, gold_records_all)
    mod_metrics = MetricsEvaluator.evaluate_modality_negation(pred_relations_all, gold_relations_all)

    chars_per_sec = total_chars / t_inf_total if t_inf_total > 0 else 0.0
    correct_facts = rel_metrics["tp"] + rec_metrics["complete_matches"]
    facts_per_sec = correct_facts / t_inf_total if t_inf_total > 0 else 0.0

    summary = {
        "model": args.model,
        "config": args.config,
        "dataset": args.dataset,
        "threads": args.threads,
        "sample_count": len(samples),
        "total_chars": total_chars,
        "load_time_sec": round(load_duration, 4),
        "total_inference_time_sec": round(t_inf_total, 4),
        "avg_latency_per_sample_sec": round(t_inf_total / len(samples), 4) if samples else 0.0,
        "throughput_chars_per_sec": round(chars_per_sec, 2),
        "facts_per_sec": round(facts_per_sec, 4),
        "peak_rss_mib": mon_summary["peak_rss_mib"],
        "peak_cgroup_mib": mon_summary["peak_cgroup_mib"],
        "peak_swap_mib": mon_summary["peak_swap_mib"],
        "entity_metrics": ent_metrics,
        "relation_metrics": rel_metrics,
        "record_metrics": rec_metrics,
        "modality_metrics": mod_metrics,
    }

    # Save outputs
    out_prefix = f"{args.model}_{args.config}_{args.dataset}_{args.threads}t"
    metrics_file = os.path.join(args.output_dir, f"metrics_{out_prefix}.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    pred_file = os.path.join(args.output_dir, f"preds_{out_prefix}.jsonl")
    with open(pred_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    print(f"Results saved to {metrics_file} and {pred_file}")
    print(f"Metrics Summary: Ent F1={ent_metrics['f1']} | Rel F1={rel_metrics['f1']} (RevErr={rel_metrics['reverse_error_rate']}) | Rec F1={rec_metrics['role_f1']} | Peak RAM={mon_summary['peak_rss_mib']} MiB")


if __name__ == "__main__":
    main()
