"""
Full Novel End-to-End Extraction Runner for 《斗破苍穹》.
Features:
1. Cross-chapter independent window batching (BS=16 saturated without text concatenation).
2. Character-balanced sharding for multi-runner matrix execution.
3. Streaming flush (per-batch JSONL append) and resume from checkpoint.
4. Exact chapter-relative offset coordinates for entities, relations, and records.
5. Shard coverage manifest and progress telemetry.
"""

import os
import sys
import time
import json
import re
import argparse
import psutil
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Set

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


def partition_chapters_balanced(chapters: List[Dict[str, Any]], total_shards: int) -> List[List[Dict[str, Any]]]:
    """Partition chapters across shards balanced by character count."""
    if total_shards <= 1:
        return [chapters]

    total_chars = sum(c["char_length"] for c in chapters)
    target_per_shard = total_chars / total_shards

    shards: List[List[Dict[str, Any]]] = [[] for _ in range(total_shards)]
    curr_shard = 0
    curr_chars = 0

    for chap in chapters:
        shards[curr_shard].append(chap)
        curr_chars += chap["char_length"]
        if curr_chars >= target_per_shard and curr_shard < total_shards - 1:
            curr_shard += 1
            curr_chars = 0

    return shards


def collect_independent_windows(chapters: List[Dict[str, Any]], chunk_size: int = 448, overlap: int = 48) -> List[Dict[str, Any]]:
    """
    Collect all sliding windows across chapters as independent units.
    Does NOT concatenate text from different chapters.
    """
    windows = []
    for chap in chapters:
        c_chunks = chunk_chapter_text(chap["text"], chunk_size=chunk_size, overlap=overlap)
        for c in c_chunks:
            windows.append({
                "window_id": f"c{chap['chapter_idx']}_w{c['chunk_idx']}",
                "chapter_idx": chap["chapter_idx"],
                "chapter_title": chap["title"],
                "char_start": c["start"],
                "char_end": c["end"],
                "text": c["text"],
                "token_len": len(c["text"])
            })
    return windows


def run_novel_extraction(
    model_path: str,
    novel_file: str,
    max_chapters: int,
    batch_size: int,
    chunk_size: int,
    overlap: int,
    num_threads: int,
    enable_int8: bool,
    shard_index: int,
    total_shards: int,
    output_dir: str
):
    os.makedirs(output_dir, exist_ok=True)
    all_chapters = load_novel_chapters(novel_file, max_chapters=max_chapters)

    # Workload-balanced sharding
    sharded_groups = partition_chapters_balanced(all_chapters, total_shards)
    if shard_index >= len(sharded_groups):
        raise ValueError(f"Invalid shard_index {shard_index} for total_shards {total_shards}")
    assigned_chapters = sharded_groups[shard_index]

    shard_chars = sum(c["char_length"] for c in assigned_chapters)
    first_c = assigned_chapters[0]["chapter_idx"] if assigned_chapters else 0
    last_c = assigned_chapters[-1]["chapter_idx"] if assigned_chapters else 0

    print("="*80)
    print(f"《斗破苍穹》端到端全量抽取 [Shard {shard_index + 1}/{total_shards}] (无级联初筛)")
    print(f"分配章节: 第 {first_c} 章 至 第 {last_c} 章 (共 {len(assigned_chapters)} 章)")
    print(f"本分片字符量: {shard_chars:,} 字符 | 线程={num_threads} | 批大小={batch_size} | INT8={enable_int8}")
    print("="*80)

    # Collect independent windows
    windows = collect_independent_windows(assigned_chapters, chunk_size=chunk_size, overlap=overlap)
    total_windows = len(windows)
    print(f"总计独立窗口数: {total_windows} 窗 (平均每批满打 {batch_size} 窗)")

    # Streaming checkpoint and resume
    checkpoint_file = os.path.join(output_dir, f"shard_{shard_index}_checkpoint.jsonl")
    completed_window_ids: Set[str] = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        completed_window_ids.add(record["window_id"])
                    except Exception:
                        pass
        print(f"-> 发现断点检查点: 已完成 {len(completed_window_ids)} / {total_windows} 窗口，跳过已处理部分。")

    # Filter out completed windows
    pending_windows = [w for w in windows if w["window_id"] not in completed_window_ids]
    print(f"-> 剩余待处理窗口: {len(pending_windows)} 窗口")

    # Resource monitor
    monitor_path = os.path.join(output_dir, f"monitor_shard_{shard_index}.jsonl")
    monitor = ResourceMonitor(output_jsonl=monitor_path, interval=1.0)
    monitor.start()

    # Load adapter
    adapter = GLiNERAdapter(model_path, num_threads=num_threads)
    adapter.load_model()
    if enable_int8:
        print("[Warning] Experimental INT8 enabled for novel shard.")
        adapter.enable_dynamic_quantization()

    # Warmup
    if pending_windows:
        adapter.predict_unified("warmup", pending_windows[0]["text"], schema_type="novel")

    # Open checkpoint file in append mode
    ckpt_f = open(checkpoint_file, "a", encoding="utf-8")

    # Metrics accumulators
    entity_freq = Counter()
    entity_by_type = defaultdict(Counter)
    relations_graph = Counter()
    battle_records = []

    processed_chars = 0
    t_start = time.time()

    # Execute in saturated batches across chapter boundaries
    for b_idx in range(0, len(pending_windows), batch_size):
        batch_slice = pending_windows[b_idx:b_idx + batch_size]
        b_ids = [w["window_id"] for w in batch_slice]
        b_texts = [w["text"] for w in batch_slice]

        tb0 = time.time()
        b_results = adapter.batch_predict_unified(
            b_ids,
            b_texts,
            schema_type="novel",
            batch_size=len(batch_slice)
        )
        b_elapsed = time.time() - tb0

        # Stream flush results to disk immediately
        for win, res in zip(batch_slice, b_results):
            c_offset = win["char_start"]
            ent_list = []
            for ent in res.entities:
                abs_s = c_offset + ent.char_start
                abs_e = c_offset + ent.char_end
                ent_list.append({
                    "text": ent.text,
                    "type": ent.entity_type,
                    "start": abs_s,
                    "end": abs_e,
                    "confidence": ent.confidence
                })
                entity_freq[ent.text.strip()] += 1
                entity_by_type[ent.entity_type][ent.text.strip()] += 1

            rel_list = []
            for rel in res.relations:
                ev_span = [c_offset + rel.evidence_span[0], c_offset + rel.evidence_span[1]] if rel.evidence_span else None
                rel_list.append({
                    "subject": rel.subject_text,
                    "subject_type": rel.subject_type,
                    "predicate": rel.relation_type,
                    "object": rel.object_text,
                    "object_type": rel.object_type,
                    "confidence": rel.confidence,
                    "evidence_span": ev_span
                })
                r_key = (rel.subject_text.strip(), rel.relation_type, rel.object_text.strip())
                relations_graph[r_key] += 1

            rec_list = []
            for rec in res.records:
                rec_item = {
                    "event_type": rec.event_type,
                    "anchor": rec.anchor_text,
                    "roles": rec.roles,
                    "modality": rec.modality
                }
                rec_list.append(rec_item)
                battle_records.append({
                    "chapter_idx": win["chapter_idx"],
                    "chapter_title": win["chapter_title"],
                    **rec_item
                })

            item_record = {
                "window_id": win["window_id"],
                "chapter_idx": win["chapter_idx"],
                "chapter_title": win["chapter_title"],
                "char_start": win["char_start"],
                "char_end": win["char_end"],
                "entities": ent_list,
                "relations": rel_list,
                "records": rec_list
            }
            ckpt_f.write(json.dumps(item_record, ensure_ascii=False) + "\n")

            processed_chars += len(win["text"])

        ckpt_f.flush()

        # Telemetry progress print every 10 batches or at end
        batch_num = b_idx // batch_size + 1
        total_batches = (len(pending_windows) + batch_size - 1) // batch_size
        if batch_num <= 3 or batch_num % 10 == 0 or batch_num == total_batches:
            elapsed_so_far = time.time() - t_start
            cur_speed = round(processed_chars / max(elapsed_so_far, 0.001), 2)
            pct = round(((len(completed_window_ids) + b_idx + len(batch_slice)) / total_windows) * 100, 1)
            mem_rss = round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 1)
            print(f"[Shard {shard_index} | Batch {batch_num:3d}/{total_batches:3d} ({pct:5.1f}%)] "
                  f"Processed Chars: {processed_chars:,} | Speed: {cur_speed:6.1f} c/s | "
                  f"Ents: {len(entity_freq):4d} | Rels: {len(relations_graph):4d} | Battles: {len(battle_records):3d} | "
                  f"RAM: {mem_rss}MiB")

    ckpt_f.close()
    elapsed_total = round(time.time() - t_start, 2)
    overall_speed = round(processed_chars / max(elapsed_total, 0.001), 2)
    mon_summary = monitor.stop()

    # Save shard manifest and summary
    manifest_data = {
        "shard_index": shard_index,
        "total_shards": total_shards,
        "chapter_range": [first_c, last_c],
        "chapters_count": len(assigned_chapters),
        "shard_chars": shard_chars,
        "total_windows": total_windows,
        "completed_windows": len(completed_window_ids) + len(pending_windows),
        "elapsed_seconds": elapsed_total,
        "average_throughput_chars_per_sec": overall_speed,
        "peak_rss_mib": mon_summary.get("peak_rss_mib", 0),
        "peak_swap_mib": mon_summary.get("peak_swap_mib", 0),
        "status": "COMPLETED"
    }
    manifest_file = os.path.join(output_dir, f"shard_{shard_index}_manifest.json")
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, ensure_ascii=False, indent=2)

    print("\n" + "="*80)
    print(f"Shard {shard_index} 抽取完成！耗时: {elapsed_total}s | 吞吐: {overall_speed} c/s | 产物保存至 {checkpoint_file}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Run Full Novel Extraction (Sharded & Checkpointed)")
    parser.add_argument("--model-path", type=str, default="fastino/gliner2.5-multi-v1")
    parser.add_argument("--novel-file", type=str, default=os.path.join(DATA_DIR, "doupo_full.txt"))
    parser.add_argument("--max-chapters", type=int, default=0, help="0 = all chapters")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=448)
    parser.add_argument("--overlap", type=int, default=48)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--enable-int8", action="store_true", default=False)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--total-shards", type=int, default=1)
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
        shard_index=args.shard_index,
        total_shards=args.total_shards,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
