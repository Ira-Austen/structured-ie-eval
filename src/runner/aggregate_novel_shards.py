"""
Aggregation Runner for Full Novel Shards.
Aggregates shard checkpoints, validates 100% chapter and character coverage,
deduplicates facts with evidence coordinates, and builds the unified Knowledge Graph.
"""

import os
import sys
import glob
import json
import re
import argparse
from collections import Counter, defaultdict
from typing import List, Dict, Any, Set, Tuple


def load_novel_chapters(file_path: str, max_chapters: int = 0) -> List[Dict[str, Any]]:
    """Parse novel file into structured chapters (pure standard library)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Novel file not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    pattern = re.compile(r'(第[0-9一二三四五六七八九十百千万]+章[^\n\r]*)')
    splits = pattern.split(text)

    chapters = []
    # If the text starts before chapter 1 (intro/prologue)
    if splits and not pattern.match(splits[0]) and len(splits[0].strip()) > 100:
        body = splits[0].strip()
        chapters.append({
            "chapter_idx": 0,
            "title": "作品前言与背景设定",
            "text": body,
            "char_length": len(body)
        })

    chapter_counter = 1
    for i in range(1, len(splits), 2):
        title = splits[i].strip()
        body = splits[i+1].strip() if i+1 < len(splits) else ""
        chapters.append({
            "chapter_idx": chapter_counter,
            "title": title,
            "text": body,
            "char_length": len(body)
        })
        chapter_counter += 1

    if max_chapters > 0:
        chapters = chapters[:max_chapters]

    return chapters


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))


def aggregate_shards(results_dir: str, novel_file: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    chapters = load_novel_chapters(novel_file)
    total_novel_chars = sum(c["char_length"] for c in chapters)
    total_chapter_count = len(chapters)

    print("="*80)
    print("《斗破苍穹》全书分片抽取产物汇聚与覆盖率核验")
    print(f"标准章节总数: {total_chapter_count} (前言 + 第1-1646章) | 正文总字数: {total_novel_chars:,} 字符")
    print("="*80)

    # Find shard checkpoints and manifests
    checkpoint_files = sorted(glob.glob(os.path.join(results_dir, "shard_*_checkpoint.jsonl")))
    manifest_files = sorted(glob.glob(os.path.join(results_dir, "shard_*_manifest.json")))

    if not checkpoint_files:
        raise FileNotFoundError(f"No shard checkpoint files found in {results_dir}")

    print(f"发现分片检查点文件: {len(checkpoint_files)} 个")
    for f in checkpoint_files:
        print(f"  - {os.path.basename(f)}")

    # Accumulators
    covered_chapters: Set[int] = set()
    covered_windows: Set[str] = set()
    shard_perf_list = []

    entity_freq = Counter()
    entity_by_type = defaultdict(Counter)
    entity_chapter_map = defaultdict(set)

    relations_graph = Counter() # (sub, sub_t, pred, obj, obj_t) -> freq
    relations_evidence = defaultdict(list)

    battle_records = []
    seen_events = set()

    total_processed_chars = 0
    total_elapsed_cpu_sec = 0.0
    peak_rss_overall = 0.0

    # Read all manifests
    for mf in manifest_files:
        with open(mf, "r", encoding="utf-8") as f:
            m_data = json.load(f)
            shard_perf_list.append(m_data)
            total_elapsed_cpu_sec += m_data.get("elapsed_seconds", 0.0)
            if m_data.get("peak_rss_mib", 0) > peak_rss_overall:
                peak_rss_overall = m_data.get("peak_rss_mib", 0)

    # Stream through all shard checkpoint JSONLs
    for ckpt_path in checkpoint_files:
        print(f"正在读取检查点: {os.path.basename(ckpt_path)}...")
        with open(ckpt_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                win_id = item["window_id"]
                if win_id in covered_windows:
                    continue
                covered_windows.add(win_id)

                chap_idx = item["chapter_idx"]
                covered_chapters.add(chap_idx)

                # Process entities
                for ent in item.get("entities", []):
                    t = ent["text"].strip()
                    etype = ent["type"]
                    if len(t) >= 2:
                        entity_freq[t] += 1
                        entity_by_type[etype][t] += 1
                        entity_chapter_map[t].add(chap_idx)

                # Process relations
                for rel in item.get("relations", []):
                    sub = rel["subject"].strip()
                    obj = rel["object"].strip()
                    pred = rel["predicate"]
                    if sub and obj and sub != obj:
                        rel_key = (sub, rel.get("subject_type", ""), pred, obj, rel.get("object_type", ""))
                        relations_graph[rel_key] += 1
                        if rel.get("evidence_span"):
                            ev_entry = {
                                "chapter_idx": chap_idx,
                                "span": rel["evidence_span"],
                                "confidence": rel.get("confidence", 1.0)
                            }
                            if len(relations_evidence[rel_key]) < 10:
                                relations_evidence[rel_key].append(ev_entry)

                # Process battle records
                for rec in item.get("records", []):
                    anchor = rec.get("anchor", "")
                    roles = rec.get("roles", {})
                    event_type = rec.get("event_type", "战斗记录")
                    if anchor:
                        event_sig = (chap_idx, anchor, tuple(sorted(roles.items())))
                        if event_sig not in seen_events:
                            seen_events.add(event_sig)
                            battle_records.append({
                                "chapter_idx": chap_idx,
                                "chapter_title": item.get("chapter_title", ""),
                                "event_type": event_type,
                                "anchor": anchor,
                                "roles": roles,
                                "modality": rec.get("modality", "actual")
                            })

    # Validate coverage
    expected_chapter_indices = set(c["chapter_idx"] for c in chapters)
    missing_chapters = sorted(list(expected_chapter_indices - covered_chapters))

    coverage_pct = round((len(covered_chapters) / max(total_chapter_count, 1)) * 100, 2)
    print("\n" + "="*80)
    print(f"覆盖率核验结果:")
    print(f"  - 覆盖章节: {len(covered_chapters)} / {total_chapter_count} ({coverage_pct}%)")
    print(f"  - 覆盖独立窗口: {len(covered_windows)} 窗")
    if missing_chapters:
        print(f"  - ⚠️ 缺失章节清单 ({len(missing_chapters)} 章): {missing_chapters[:20]}...")
    else:
        print(f"  - ✅ 100% 章节全量覆盖完成，无遗漏！")
    print("="*80)

    # Build Top KG summaries
    top_characters = [
        {"name": name, "count": cnt, "chapter_appearances": len(entity_chapter_map[name])}
        for name, cnt in entity_by_type["人物"].most_common(50)
    ]
    top_organizations = [
        {"name": name, "count": cnt, "chapter_appearances": len(entity_chapter_map[name])}
        for name, cnt in entity_by_type["组织"].most_common(30)
    ]
    top_skills = [
        {"name": name, "count": cnt, "chapter_appearances": len(entity_chapter_map[name])}
        for name, cnt in entity_by_type["技能"].most_common(30)
    ]
    top_realms = [
        {"name": name, "count": cnt}
        for name, cnt in entity_by_type["境界"].most_common(20)
    ]
    top_items = [
        {"name": name, "count": cnt}
        for name, cnt in entity_by_type["物品"].most_common(30)
    ]

    top_relations = [
        {
            "subject": k[0],
            "subject_type": k[1],
            "predicate": k[2],
            "object": k[3],
            "object_type": k[4],
            "frequency": cnt,
            "evidence_examples": relations_evidence[k][:3]
        }
        for k, cnt in relations_graph.most_common(80)
    ]

    kg_summary = {
        "title": "斗破苍穹全本知识图谱与事件全貌 (GLiNER2.5 G5 Full Pipeline 汇聚结果)",
        "total_novel_characters": total_novel_chars,
        "chapters_covered": len(covered_chapters),
        "total_chapters_expected": total_chapter_count,
        "coverage_percentage": coverage_pct,
        "missing_chapters": missing_chapters,
        "metrics": {
            "unique_entities_total": len(entity_freq),
            "unique_characters": len(entity_by_type["人物"]),
            "unique_organizations": len(entity_by_type["组织"]),
            "unique_skills": len(entity_by_type["技能"]),
            "unique_realms": len(entity_by_type["境界"]),
            "unique_relations_triples": len(relations_graph),
            "total_battle_events": len(battle_records)
        },
        "top_characters": top_characters,
        "top_organizations": top_organizations,
        "top_skills": top_skills,
        "top_realms": top_realms,
        "top_items": top_items,
        "top_relations": top_relations,
        "battle_events_sample": battle_records[:60]
    }

    # Build performance profile
    # Calculate wall-clock and equivalent single-node throughput
    single_node_speed = round(total_novel_chars / max(total_elapsed_cpu_sec, 0.001), 2)
    perf_summary = {
        "task_name": "Doupo Full Novel Structured IE (GLiNER2.5 G5 16GB Baseline)",
        "total_novel_characters": total_novel_chars,
        "total_chapters": total_chapter_count,
        "covered_chapters": len(covered_chapters),
        "covered_windows": len(covered_windows),
        "total_shards": len(checkpoint_files),
        "total_cpu_elapsed_seconds": round(total_elapsed_cpu_sec, 2),
        "total_cpu_elapsed_hours": round(total_elapsed_cpu_sec / 3600, 2),
        "equivalent_single_node_throughput_chars_per_sec": single_node_speed,
        "peak_rss_mib": peak_rss_overall,
        "shards": shard_perf_list
    }

    # Coverage report
    coverage_report = {
        "total_chapters": total_chapter_count,
        "covered_chapters_count": len(covered_chapters),
        "coverage_percentage": coverage_pct,
        "is_complete": (len(missing_chapters) == 0),
        "missing_chapters": missing_chapters
    }

    # Write files
    kg_file = os.path.join(output_dir, "doupo_full_kg_summary.json")
    with open(kg_file, "w", encoding="utf-8") as f:
        json.dump(kg_summary, f, ensure_ascii=False, indent=2)

    perf_file = os.path.join(output_dir, "doupo_full_perf.json")
    with open(perf_file, "w", encoding="utf-8") as f:
        json.dump(perf_summary, f, ensure_ascii=False, indent=2)

    cov_file = os.path.join(output_dir, "doupo_coverage_report.json")
    with open(cov_file, "w", encoding="utf-8") as f:
        json.dump(coverage_report, f, ensure_ascii=False, indent=2)

    print(f"汇聚结果已成功保存:")
    print(f"  - 知识图谱全貌: {kg_file}")
    print(f"  - 性能汇总: {perf_file}")
    print(f"  - 覆盖率报告: {cov_file}\n")


def main():
    parser = argparse.ArgumentParser(description="Aggregate Full Novel Shards")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--novel-file", type=str, default=os.path.join(DATA_DIR, "doupo_full.txt"))
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    aggregate_shards(
        results_dir=args.results_dir,
        novel_file=args.novel_file,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
