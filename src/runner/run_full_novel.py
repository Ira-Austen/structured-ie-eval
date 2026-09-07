"""
Full Novel End-to-End Extraction Runner for 《斗破苍穹》.
Processes chapters sequentially or in batches without cascade pre-filtering.
Extracts entities, directed relations, and natural battle records across the novel.
Builds and exports the aggregated Knowledge Graph and performance profile.
"""

import os
import sys
import time
import json
import re
import argparse
import psutil
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple

from ..models.gliner_adapter import GLiNERAdapter
from ..monitor.resource_monitor import ResourceMonitor


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))


def load_novel_chapters(file_path: str, max_chapters: int = 0) -> List[Dict[str, Any]]:
    """Parse novel file into structured chapters."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Novel file not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    pattern = re.compile(r'(第[0-9一二三四五六七八九十百千万]+章[^\n\r]*)')
    splits = pattern.split(text)

    chapters = []
    # If the text starts before chapter 1 (intro/prologue)
    if splits and not pattern.match(splits[0]) and len(splits[0].strip()) > 100:
        chapters.append({
            "chapter_idx": 0,
            "title": "作品前言与背景设定",
            "text": splits[0].strip(),
            "char_length": len(splits[0].strip())
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


def chunk_chapter_text(text: str, chunk_size: int = 448, overlap: int = 48) -> List[Dict[str, Any]]:
    """Slice text into overlapping sliding chunks."""
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


def run_novel_extraction(
    model_path: str,
    novel_file: str,
    max_chapters: int,
    batch_size: int,
    chunk_size: int,
    overlap: int,
    num_threads: int,
    enable_int8: bool,
    output_dir: str
):
    os.makedirs(output_dir, exist_ok=True)
    chapters = load_novel_chapters(novel_file, max_chapters=max_chapters)
    total_novel_chars = sum(c["char_length"] for c in chapters)

    print("="*80)
    print(f"《斗破苍穹》全量端到端结构化抽取启动 (无级联初筛)")
    print(f"章节总数: {len(chapters)} 章 | 待处理文本规模: {total_novel_chars:,} 字符")
    print(f"配置: 4 vCPU, 批大小={batch_size}, 窗口={chunk_size}, 重叠={overlap}, 线程={num_threads}, INT8={enable_int8}")
    print("="*80)

    # Start monitor
    monitor_path = os.path.join(output_dir, "monitor_full_novel.jsonl")
    monitor = ResourceMonitor(output_jsonl=monitor_path, interval=1.0)
    monitor.start()

    # Load adapter
    adapter = GLiNERAdapter(model_path, num_threads=num_threads)
    adapter.load_model()
    if enable_int8:
        adapter.enable_dynamic_quantization()

    # Warmup
    warmup_text = chapters[0]["text"][:chunk_size]
    adapter.predict_unified("warmup", warmup_text, schema_type="novel")

    # Knowledge Graph accumulators
    entity_freq = Counter()
    entity_by_type = defaultdict(Counter)
    relations_graph = Counter() # (sub, rel, obj) -> count
    battle_records = []
    chapter_logs = []

    total_chunks_processed = 0
    total_chars_processed = 0
    t_start = time.time()

    for idx, chap in enumerate(chapters, 1):
        chap_title = chap["title"]
        chap_text = chap["text"]
        chap_len = len(chap_text)

        if not chap_text:
            continue

        chunks = chunk_chapter_text(chap_text, chunk_size=chunk_size, overlap=overlap)
        sids = [f"c{chap['chapter_idx']}_ck{c['chunk_idx']}" for c in chunks]
        texts = [c["text"] for c in chunks]

        t_chap_0 = time.time()
        res_list = adapter.batch_predict_unified(
            sids,
            texts,
            schema_type="novel",
            batch_size=batch_size
        )
        chap_elapsed = time.time() - t_chap_0

        chap_ents_count = 0
        chap_rels_count = 0
        chap_recs_count = 0

        # Chapter level deduplication
        seen_entities = set()
        for res, chunk_meta in zip(res_list, chunks):
            c_offset = chunk_meta["start"]
            for ent in res.entities:
                e_key = (ent.text.strip(), ent.entity_type)
                if len(ent.text.strip()) >= 2 and e_key not in seen_entities:
                    seen_entities.add(e_key)
                    entity_freq[ent.text.strip()] += 1
                    entity_by_type[ent.entity_type][ent.text.strip()] += 1
                    chap_ents_count += 1

            for rel in res.relations:
                if rel.subject_text and rel.object_text and rel.subject_text != rel.object_text:
                    r_key = (rel.subject_text.strip(), rel.relation_type, rel.object_text.strip())
                    relations_graph[r_key] += 1
                    chap_rels_count += 1

            for rec in res.records:
                if rec.anchor_text:
                    b_item = {
                        "chapter_idx": chap["chapter_idx"],
                        "chapter_title": chap_title,
                        "attacker": rec.anchor_text,
                        "roles": rec.roles
                    }
                    battle_records.append(b_item)
                    chap_recs_count += 1

        total_chunks_processed += len(chunks)
        total_chars_processed += chap_len
        elapsed_so_far = time.time() - t_start
        speed_so_far = round(total_chars_processed / max(elapsed_so_far, 0.001), 2)

        chapter_logs.append({
            "chapter_idx": chap["chapter_idx"],
            "title": chap_title,
            "char_length": chap_len,
            "chunks_count": len(chunks),
            "elapsed_sec": round(chap_elapsed, 3),
            "chars_per_sec": round(chap_len / max(chap_elapsed, 0.001), 2),
            "entities_found": chap_ents_count,
            "relations_found": chap_rels_count,
            "battle_events": chap_recs_count
        })

        # Checkpoint log every 20 chapters or first 5 chapters
        if idx <= 5 or idx % 20 == 0 or idx == len(chapters):
            progress_pct = round((idx / len(chapters)) * 100, 1)
            eta_sec = (total_novel_chars - total_chars_processed) / max(speed_so_far, 1)
            eta_min = round(eta_sec / 60, 1)
            mem_now = round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 1)

            print(f"[{idx:4d}/{len(chapters)} ({progress_pct:5.1f}%)] {chap_title[:18]:<18} | "
                  f"Chars: {total_chars_processed:,} | Speed: {speed_so_far:7.1f} c/s | "
                  f"ETA: {eta_min:4.1f}m | Ents: {len(entity_freq):4d} | Rels: {len(relations_graph):4d} | "
                  f"Battles: {len(battle_records):4d} | RAM: {mem_now}MiB")

    total_time_sec = round(time.time() - t_start, 2)
    overall_speed = round(total_chars_processed / max(total_time_sec, 0.001), 2)
    mon_summary = monitor.stop()

    # Build Knowledge Graph output
    top_characters = [{"name": name, "count": cnt} for name, cnt in entity_by_type["人物"].most_common(50)]
    top_organizations = [{"name": name, "count": cnt} for name, cnt in entity_by_type["组织"].most_common(30)]
    top_realms = [{"name": name, "count": cnt} for name, cnt in entity_by_type["境界"].most_common(20)]
    top_skills = [{"name": name, "count": cnt} for name, cnt in entity_by_type["技能"].most_common(30)]
    top_items = [{"name": name, "count": cnt} for name, cnt in entity_by_type["物品"].most_common(30)]

    top_relations = [
        {"subject": k[0], "predicate": k[1], "object": k[2], "frequency": cnt}
        for k, cnt in relations_graph.most_common(60)
    ]

    kg_summary = {
        "title": "斗破苍穹全本知识图谱与事件摘要 (GLiNER2.5 G5 Full Pipeline)",
        "chapters_processed": len(chapters),
        "total_chars_processed": total_chars_processed,
        "metrics": {
            "unique_entities_total": len(entity_freq),
            "unique_characters": len(entity_by_type["人物"]),
            "unique_organizations": len(entity_by_type["组织"]),
            "unique_skills": len(entity_by_type["技能"]),
            "unique_relations_triples": len(relations_graph),
            "total_battle_events": len(battle_records)
        },
        "top_characters": top_characters,
        "top_organizations": top_organizations,
        "top_realms": top_realms,
        "top_skills": top_skills,
        "top_items": top_items,
        "top_relations": top_relations,
        "sample_battle_records": battle_records[:40]
    }

    kg_file = os.path.join(output_dir, "doupo_full_kg_summary.json")
    with open(kg_file, "w", encoding="utf-8") as f:
        json.dump(kg_summary, f, ensure_ascii=False, indent=2)

    perf_summary = {
        "model": "GLiNER2.5-multi",
        "optimization_mode": "G5_UNIFIED_BATCH" + ("_INT8" if enable_int8 else "_FP32"),
        "total_novel_characters": total_chars_processed,
        "total_chapters": len(chapters),
        "total_chunks_processed": total_chunks_processed,
        "elapsed_seconds": total_time_sec,
        "elapsed_minutes": round(total_time_sec / 60, 2),
        "average_throughput_chars_per_sec": overall_speed,
        "peak_rss_mib": mon_summary["peak_rss_mib"],
        "peak_cgroup_mib": mon_summary["peak_cgroup_mib"],
        "chapter_logs": chapter_logs[:50] # save first 50 chapter logs
    }

    perf_file = os.path.join(output_dir, "doupo_full_perf.json")
    with open(perf_file, "w", encoding="utf-8") as f:
        json.dump(perf_summary, f, ensure_ascii=False, indent=2)

    print("\n" + "="*80)
    print("《斗破苍穹》全量抽取完成")
    print(f"总处理字符: {total_chars_processed:,} | 总耗时: {total_time_sec}s ({round(total_time_sec/60, 2)}m)")
    print(f"平均吞吐: {overall_speed} 字符/秒 | 峰值内存: {mon_summary['peak_rss_mib']} MiB")
    print(f"抽取实体: 人物 {len(entity_by_type['人物'])} 个, 门派 {len(entity_by_type['组织'])} 个, 斗技 {len(entity_by_type['技能'])} 个")
    print(f"抽取有向关系: {len(relations_graph)} 种实体三元组 | 战斗记录: {len(battle_records)} 条")
    print(f"产物落盘: {kg_file} & {perf_file}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run Full Novel Extraction Benchmark")
    parser.add_argument("--model-path", type=str, default="fastino/gliner2.5-multi-v1")
    parser.add_argument("--novel-file", type=str, default=os.path.join(DATA_DIR, "doupo_full.txt"))
    parser.add_argument("--max-chapters", type=int, default=0, help="0 means process all 1646 chapters")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=448)
    parser.add_argument("--overlap", type=int, default=48)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--enable-int8", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    run_novel_extraction(
        model_path=args.model_path,
        novel_file=args.novel_file,
        max_chapters=args.max_chapters,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        num_threads=args.threads,
        enable_int8=args.enable_int8,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
