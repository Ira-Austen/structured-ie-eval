"""
UIE (Universal Information Extraction) Model Adapter.
Supports Casually/uie-medium and Casually/uie-mini.
- Explicit batch_size=1
- Hierarchical schemas for entities and relations
- Direction normalization mapping UIE prompt structure to canonical directed facts
- Captures raw confidence and span offsets strictly from original text
"""

import os
import sys
import time
import torch
import psutil
from typing import List, Dict, Any, Optional
from .base import EntityMention, DirectedRelation, EventRecord, ExtractionResult


class UIEAdapter:
    def __init__(self, model_path_or_name: str, num_threads: int = 1):
        self.model_name = "UIE"
        self.variant = "medium" if "medium" in model_path_or_name.lower() else "mini"
        self.model_path = model_path_or_name
        self.num_threads = num_threads

        torch.set_num_threads(num_threads)
        torch.set_grad_enabled(False)
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)

        self.tokenizer = None
        self.model = None
        self.load_time_sec = 0.0

    def load_model(self):
        from transformers import AutoTokenizer, AutoModel

        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
        self.model.eval()
        self.load_time_sec = time.time() - t0

    def predict(self, sample_id: str, text: str, schema_type: str = "general") -> ExtractionResult:
        if self.model is None:
            self.load_model()

        t0 = time.time()
        proc = psutil.Process(os.getpid())
        mem_before = proc.memory_info().rss / (1024.0 * 1024.0)

        # 1. Select hierarchical Schema
        if schema_type == "novel":
            schema_entities = ["人物", "家族", "组织", "地点", "物品", "技能", "境界", "称谓"]
            schema_relations = {
                "人物": ["父亲", "儿子", "家族", "宗门", "师父", "徒弟", "族长"]
            }
        else:
            # Business / cross-domain schema
            schema_entities = ["人物", "公司", "组织", "地点", "金额", "物品", "职务"]
            schema_relations = {
                "人物": ["任职", "雇佣", "持股", "交易对象", "父亲", "师父"],
                "公司": ["高管", "股东", "子公司", "交付目的地"]
            }

        # 2. Forward inference with batch_size=1
        try:
            # Extract entities
            res_ents = self.model.predict(
                schema=schema_entities,
                input_texts=text,
                tokenizer=self.tokenizer,
                batch_size=1
            )

            # Extract relations
            res_rels = self.model.predict(
                schema=schema_relations,
                input_texts=text,
                tokenizer=self.tokenizer,
                batch_size=1
            )
        except Exception as e:
            return ExtractionResult(
                sample_id=sample_id,
                model_name=f"UIE-{self.variant}",
                config_id=f"uie_{self.variant}",
                text=text,
                status="FAILED_RUNTIME",
                error_message=str(e)
            )

        t_elapsed = time.time() - t0
        mem_after = proc.memory_info().rss / (1024.0 * 1024.0)

        # 3. Canonicalize outputs
        entities: List[EntityMention] = []
        relations: List[DirectedRelation] = []
        records: List[EventRecord] = []

        ent_counter = 1
        # Parse entities
        if res_ents and isinstance(res_ents, list) and len(res_ents) > 0:
            doc_ents = res_ents[0]
            for ent_type, items in doc_ents.items():
                for item in items:
                    entities.append(EntityMention(
                        mention_id=f"uie_e_{ent_counter}",
                        text=item["text"],
                        entity_type=ent_type,
                        char_start=item["start"],
                        char_end=item["end"],
                        confidence=float(item.get("probability", 1.0))
                    ))
                    ent_counter += 1

        # Parse relations with careful direction mapping
        rel_counter = 1
        if res_rels and isinstance(res_rels, list) and len(res_rels) > 0:
            doc_rels = res_rels[0]
            for head_type, items in doc_rels.items():
                for item in items:
                    head_text = item["text"]
                    head_start = item["start"]
                    head_end = item["end"]
                    rel_dict = item.get("relations", {})

                    for predicate, tail_items in rel_dict.items():
                        for tail in tail_items:
                            tail_text = tail["text"]
                            tail_start = tail["start"]
                            tail_end = tail["end"]
                            conf = float(tail.get("probability", 1.0))

                            # Direction normalization
                            if predicate == "父亲":
                                # "A 的 父亲 是 B" => Canonical: B parent_of A
                                relations.append(DirectedRelation(
                                    relation_id=f"uie_r_{rel_counter}",
                                    relation_type="parent_of",
                                    subject_id="",
                                    subject_text=tail_text,
                                    subject_type="Person",
                                    object_id="",
                                    object_text=head_text,
                                    object_type="Person",
                                    confidence=conf
                                ))
                            elif predicate == "儿子":
                                # "A 的 儿子 是 B" => Canonical: A parent_of B
                                relations.append(DirectedRelation(
                                    relation_id=f"uie_r_{rel_counter}",
                                    relation_type="parent_of",
                                    subject_id="",
                                    subject_text=head_text,
                                    subject_type="Person",
                                    object_id="",
                                    object_text=tail_text,
                                    object_type="Person",
                                    confidence=conf
                                ))
                            elif predicate == "师父":
                                # "A 的 师父 是 B" => Canonical: B mentor_of A
                                relations.append(DirectedRelation(
                                    relation_id=f"uie_r_{rel_counter}",
                                    relation_type="mentor_of",
                                    subject_id="",
                                    subject_text=tail_text,
                                    subject_type="Person",
                                    object_id="",
                                    object_text=head_text,
                                    object_type="Person",
                                    confidence=conf
                                ))
                            elif predicate == "徒弟":
                                # "A 的 徒弟 是 B" => Canonical: A mentor_of B
                                relations.append(DirectedRelation(
                                    relation_id=f"uie_r_{rel_counter}",
                                    relation_type="mentor_of",
                                    subject_id="",
                                    subject_text=head_text,
                                    subject_type="Person",
                                    object_id="",
                                    object_text=tail_text,
                                    object_type="Person",
                                    confidence=conf
                                ))
                            elif predicate in {"家族", "宗门", "组织"}:
                                # "A 的 家族 是 B" => Canonical: A member_of B
                                relations.append(DirectedRelation(
                                    relation_id=f"uie_r_{rel_counter}",
                                    relation_type="member_of",
                                    subject_id="",
                                    subject_text=head_text,
                                    subject_type="Person",
                                    object_id="",
                                    object_text=tail_text,
                                    object_type="Organization",
                                    confidence=conf
                                ))
                            elif predicate in {"股东", "持股"}:
                                # "公司A 的 股东 是 B" => Canonical: B holds_equity 公司A
                                relations.append(DirectedRelation(
                                    relation_id=f"uie_r_{rel_counter}",
                                    relation_type="holds_equity",
                                    subject_id="",
                                    subject_text=tail_text,
                                    subject_type="Person",
                                    object_id="",
                                    object_text=head_text,
                                    object_type="Organization",
                                    confidence=conf
                                ))
                            elif predicate in {"任职", "高管", "族长"}:
                                relations.append(DirectedRelation(
                                    relation_id=f"uie_r_{rel_counter}",
                                    relation_type="holds_position",
                                    subject_id="",
                                    subject_text=head_text,
                                    subject_type="Person",
                                    object_id="",
                                    object_text=tail_text,
                                    object_type="Organization",
                                    role=predicate,
                                    confidence=conf
                                ))
                            rel_counter += 1

        return ExtractionResult(
            sample_id=sample_id,
            model_name=f"UIE-{self.variant}",
            config_id=f"uie_{self.variant}",
            text=text,
            entities=entities,
            relations=relations,
            records=records,
            raw_output={"res_ents": res_ents, "res_rels": res_rels},
            execution_time_sec=round(t_elapsed, 4),
            peak_memory_mib=round(mem_after, 2),
            status="COMPLETED"
        )
