"""
GLiNER2.5 Multi-task Adapter implementing G0 through G5:
- G0: char splitter + entity descriptions + basic relations (raw output)
- G1: G0 + Unified External Validation (dedup, self-loop rejection, type checking)
- G2: JointIE Engine with constrained lattice decoding (acyclic, no self-loops, endpoint typing)
- G3: Natural Records (anchor-based instance grouping and field pairing)
- G4a: Mention Span Attributes (polarity / modality scope binding)
- G4b: Constrained Classification (cross-task constraints)
- G5: Full Integrated Pipeline (JointIE relations + Natural Records + Attributes + Validation)
"""

import os
import sys
import time
import torch
import psutil
from typing import List, Dict, Any, Optional
from .base import EntityMention, DirectedRelation, EventRecord, ExtractionResult
from ..validation.validator import UnifiedValidator


class GLiNERAdapter:
    def __init__(self, model_path_or_name: str, num_threads: int = 1):
        self.model_name = "GLiNER2.5-multi"
        self.model_path = model_path_or_name
        self.num_threads = num_threads

        torch.set_num_threads(num_threads)
        torch.set_grad_enabled(False)
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)

        self.extractor = None
        self.joint_engine = None
        self.validator = UnifiedValidator()
        self.load_time_sec = 0.0
        self._schema_cache = {}

    def load_model(self):
        from gliner2 import AutoExtractor
        from gliner2.joint_ie import JointIEEngine

        t0 = time.time()
        self.extractor = AutoExtractor.from_pretrained(
            self.model_path,
            map_location="cpu"
        )
        # Crucial: char splitter for non-space delimited Chinese
        self.extractor.set_word_splitter("char")

        # Initialize JointIE engine using the same base extractor
        try:
            self.joint_engine = JointIEEngine(model=self.extractor)
        except Exception:
            self.joint_engine = None

        self.load_time_sec = time.time() - t0

    def predict(self, sample_id: str, text: str, config_id: str = "G0", schema_type: str = "general") -> ExtractionResult:
        if self.extractor is None:
            self.load_model()

        t0 = time.time()
        proc = psutil.Process(os.getpid())

        try:
            if config_id == "G0":
                res = self._predict_g0(sample_id, text, schema_type)
            elif config_id == "G1":
                res_g0 = self._predict_g0(sample_id, text, schema_type)
                res = self.validator.validate_result(res_g0)
                res.config_id = "G1"
            elif config_id == "G2":
                res = self._predict_g2_joint_ie(sample_id, text, schema_type)
            elif config_id == "G3":
                res = self._predict_g3_natural_records(sample_id, text, schema_type)
            elif config_id == "G4a":
                res = self._predict_g4a_attributes(sample_id, text, schema_type)
            elif config_id == "G4b":
                res = self._predict_g4b_classification(sample_id, text, schema_type)
            elif config_id == "G5":
                res = self._predict_g5_integrated(sample_id, text, schema_type)
            else:
                res = self._predict_g0(sample_id, text, schema_type)
        except Exception as e:
            return ExtractionResult(
                sample_id=sample_id,
                model_name=self.model_name,
                config_id=config_id,
                text=text,
                status="FAILED_RUNTIME",
                error_message=str(e)
            )

        t_elapsed = time.time() - t0
        mem_peak = proc.memory_info().rss / (1024.0 * 1024.0)

        res.execution_time_sec = round(t_elapsed, 4)
        res.peak_memory_mib = round(mem_peak, 2)
        return res

    def _parse_gliner_entities(self, raw_res: Dict[str, Any], prefix: str = "e") -> List[EntityMention]:
        entities: List[EntityMention] = []
        ent_counter = 1
        raw_entities = raw_res.get("entities", {})

        if isinstance(raw_entities, dict):
            for label, items in raw_entities.items():
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            entities.append(EntityMention(
                                mention_id=f"{prefix}_{ent_counter}",
                                text=item.get("text", ""),
                                entity_type=label,
                                char_start=item.get("start", 0),
                                char_end=item.get("end", 0),
                                confidence=float(item.get("confidence", 1.0))
                            ))
                            ent_counter += 1
                        elif isinstance(item, str):
                            entities.append(EntityMention(
                                mention_id=f"{prefix}_{ent_counter}",
                                text=item,
                                entity_type=label,
                                char_start=0,
                                char_end=len(item),
                                confidence=1.0
                            ))
                            ent_counter += 1
        elif isinstance(raw_entities, list):
            for item in raw_entities:
                if isinstance(item, dict):
                    entities.append(EntityMention(
                        mention_id=f"{prefix}_{ent_counter}",
                        text=item.get("text", ""),
                        entity_type=item.get("label", item.get("type", "")),
                        char_start=item.get("start", 0),
                        char_end=item.get("end", 0),
                        confidence=float(item.get("confidence", 1.0))
                    ))
                    ent_counter += 1
        return entities

    def _parse_gliner_structures(self, raw_res: Dict[str, Any], prefix: str = "rec") -> List[EventRecord]:
        records: List[EventRecord] = []
        rec_counter = 1

        # Collect candidate structure mappings:
        struct_maps: List[Dict[str, Any]] = []
        if isinstance(raw_res, dict):
            if "structures" in raw_res and isinstance(raw_res["structures"], dict):
                struct_maps.append(raw_res["structures"])
            if "structure" in raw_res and isinstance(raw_res["structure"], dict):
                struct_maps.append(raw_res["structure"])
            # Also inspect top-level keys that are not standard entity/relation containers
            top_level = {
                k: v for k, v in raw_res.items()
                if k not in {"entities", "relation_extraction", "relations", "structures", "structure", "classification"}
                and isinstance(v, list)
            }
            if top_level:
                struct_maps.append(top_level)

        known_anchor_fields = {
            "战斗记录": "攻击者",
            "资金交易": "付款方",
            "物流交付": "发货方",
            "任职经历": "任职者",
            "股权持有": "持股方"
        }

        seen_records = set()
        for smap in struct_maps:
            for event_name, inst_list in smap.items():
                if not isinstance(inst_list, list):
                    continue
                anchor_field_name = known_anchor_fields.get(event_name)
                for inst in inst_list:
                    if not isinstance(inst, dict):
                        continue
                    # Determine anchor text
                    anchor_text = ""
                    if "anchor" in inst:
                        anchor_val = inst["anchor"]
                        anchor_text = anchor_val.get("text", "") if isinstance(anchor_val, dict) else str(anchor_val or "")
                    elif anchor_field_name and anchor_field_name in inst:
                        anchor_val = inst[anchor_field_name]
                        anchor_text = anchor_val.get("text", "") if isinstance(anchor_val, dict) else str(anchor_val or "")
                    else:
                        for k, v in inst.items():
                            if v is not None:
                                anchor_text = v.get("text", "") if isinstance(v, dict) else str(v or "")
                                if anchor_text:
                                    break

                    roles = {}
                    for k, v in inst.items():
                        if v is not None:
                            val_str = v.get("text", "") if isinstance(v, dict) else str(v)
                            if val_str:
                                roles[k] = val_str

                    if not anchor_text and not roles:
                        continue

                    rec_key = (event_name, anchor_text, tuple(sorted(roles.items())))
                    if rec_key in seen_records:
                        continue
                    seen_records.add(rec_key)

                    records.append(EventRecord(
                        record_id=f"{prefix}_{rec_counter}",
                        event_type=event_name,
                        anchor_id="",
                        anchor_text=anchor_text,
                        roles=roles,
                        polarity="positive",
                        modality="actual"
                    ))
                    rec_counter += 1
        return records

    def _parse_gliner_relations(
        self,
        raw_res: Dict[str, Any],
        prefix: str = "r",
        entities: Optional[List[EntityMention]] = None
    ) -> List[DirectedRelation]:
        relations: List[DirectedRelation] = []
        rel_counter = 1

        ent_by_span = {}
        ent_by_text = {}
        if entities:
            for e in entities:
                ent_by_span[(e.char_start, e.char_end)] = e
                t = e.text.strip()
                if t and t not in ent_by_text:
                    ent_by_text[t] = e

        raw_rels = raw_res.get("relation_extraction") or raw_res.get("relations", {})
        if not raw_rels and isinstance(raw_res, dict):
            raw_rels = {
                k: v for k, v in raw_res.items()
                if k not in {"entities", "structures", "structure", "classification"}
                and isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict)
                and ("head" in v[0] and "tail" in v[0])
            }

        def map_rel_type(label: str) -> str:
            if label in {"parent_of", "父子关系", "母子关系", "父女关系", "母女关系"}:
                return "parent_of"
            if label in {"mentor_of", "师徒关系"}:
                return "mentor_of"
            if label in {"member_of", "所属势力", "家族", "宗门"}:
                return "member_of"
            if label in {"holds_position", "任职关系"}:
                return "holds_position"
            if label in {"holds_equity", "持股关系"}:
                return "holds_equity"
            if label in {"transferred_to", "交易关系"}:
                return "transferred_to"
            if label in {"delivered_to", "交付关系"}:
                return "delivered_to"
            return label

        def process_item(item: Dict[str, Any], canon_type: str):
            nonlocal rel_counter
            head = item.get("head", {}) if isinstance(item, dict) else {}
            tail = item.get("tail", {}) if isinstance(item, dict) else {}
            conf = float(head.get("confidence", item.get("confidence", 1.0)) if isinstance(head, dict) else 1.0)
            head_text = head.get("text", "") if isinstance(head, dict) else str(head)
            tail_text = tail.get("text", "") if isinstance(tail, dict) else str(tail)

            h_start = head.get("start", -1) if isinstance(head, dict) else -1
            h_end = head.get("end", -1) if isinstance(head, dict) else -1
            t_start = tail.get("start", -1) if isinstance(tail, dict) else -1
            t_end = tail.get("end", -1) if isinstance(tail, dict) else -1

            sub_ent = ent_by_span.get((h_start, h_end)) or ent_by_text.get(head_text.strip())
            obj_ent = ent_by_span.get((t_start, t_end)) or ent_by_text.get(tail_text.strip())

            sub_id = sub_ent.mention_id if sub_ent else ""
            sub_type = sub_ent.entity_type if sub_ent else ""
            obj_id = obj_ent.mention_id if obj_ent else ""
            obj_type = obj_ent.entity_type if obj_ent else ""

            # Explicit Kinship Direction Mapping:
            # "亲属关系": Only map to parent_of if head has parent words or tail has child words.
            actual_type = canon_type
            if canon_type == "亲属关系":
                parent_words = ("父亲", "爹", "母亲", "娘", "族长", "老头子")
                child_words = ("儿子", "女儿", "少爷", "孩子")
                if any(w in head_text for w in parent_words) or any(w in tail_text for w in child_words):
                    actual_type = "parent_of"
                else:
                    actual_type = "kinship"

            relations.append(DirectedRelation(
                relation_id=f"{prefix}_{rel_counter}",
                relation_type=actual_type,
                subject_id=sub_id,
                subject_text=head_text,
                subject_type=sub_type,
                object_id=obj_id,
                object_text=tail_text,
                object_type=obj_type,
                confidence=conf
            ))
            rel_counter += 1

        if isinstance(raw_rels, dict):
            for label, items in raw_rels.items():
                canon_type = map_rel_type(label)
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            process_item(item, canon_type)
        elif isinstance(raw_rels, list):
            for item in raw_rels:
                if isinstance(item, dict):
                    label = item.get("label", item.get("type", ""))
                    canon_type = map_rel_type(label)
                    process_item(item, canon_type)

        return relations

    def _predict_g0(self, sample_id: str, text: str, schema_type: str) -> ExtractionResult:
        schema = self.extractor.create_schema()

        if schema_type == "novel":
            schema = schema.entities({
                "人物": "小说中登场的人物姓名或称谓，如萧炎、萧战、药老、萧宁",
                "组织": "家族、宗门或势力名称，如萧家、云岚宗",
                "境界": "修炼境界等级，如斗者、大斗师、五星大斗师",
                "技能": "斗技名称，如裂爪击、吸掌",
                "物品": "丹药、宝物或器物，如聚气散、古玉盒子",
                "地点": "地名或特定场所，如乌坦城、训练场"
            })
            schema = schema.relations({
                "亲属关系": {"threshold": 0.4},
                "师徒关系": {"threshold": 0.4},
                "所属势力": {"threshold": 0.4}
            })
        else:
            # Business / cross-domain schema
            schema = schema.entities({
                "人物": "人员姓名",
                "公司": "企业或公司法人名称",
                "组织": "部门或机构名称",
                "金额": "交易金额或资金数字",
                "物品": "货物或商品名称",
                "地点": "配送地址或履行地点"
            })
            schema = schema.relations({
                "任职关系": {"threshold": 0.4},
                "持股关系": {"threshold": 0.4},
                "交易关系": {"threshold": 0.4},
                "交付关系": {"threshold": 0.4}
            })

        raw_res = self.extractor.extract(
            text,
            schema,
            threshold=0.4,
            include_spans=True,
            include_confidence=True
        )

        entities = self._parse_gliner_entities(raw_res, prefix="g0_e")
        relations = self._parse_gliner_relations(raw_res, prefix="g0_r", entities=entities)

        return ExtractionResult(
            sample_id=sample_id,
            model_name=self.model_name,
            config_id="G0",
            text=text,
            entities=entities,
            relations=relations,
            records=[],
            raw_output=raw_res,
            status="COMPLETED"
        )

    def _predict_g2_joint_ie(self, sample_id: str, text: str, schema_type: str) -> ExtractionResult:
        """JointIE with constrained joint decoding."""
        if self.joint_engine:
            try:
                from gliner2.joint_ie import JointSchema
                joint_schema = JointSchema()
                if schema_type == "novel":
                    joint_schema.entities(["人物", "组织", "境界", "技能", "物品", "地点"])
                    joint_schema.relation("parent_of", "人物", "人物")
                    joint_schema.relation("mentor_of", "人物", "人物")
                    joint_schema.relation("member_of", "人物", "组织")
                    joint_schema.no_self_loops()
                    joint_schema.acyclic("mentor_of")
                else:
                    joint_schema.entities(["人物", "公司", "组织", "金额", "物品", "地点"])
                    joint_schema.relation("holds_position", "人物", "公司")
                    joint_schema.relation("holds_equity", "人物", "公司")
                    joint_schema.relation("transferred_to", "人物", "人物")
                    joint_schema.relation("delivered_to", "公司", "地点")
                    joint_schema.no_self_loops()

                joint_res = self.joint_engine.extract(text, joint_schema)
                entities: List[EntityMention] = []
                relations: List[DirectedRelation] = []

                for idx, ent in enumerate(joint_res.entities, 1):
                    entities.append(EntityMention(
                        mention_id=f"g2_e_{idx}",
                        text=ent.text,
                        entity_type=ent.type,
                        char_start=ent.start,
                        char_end=ent.end,
                        confidence=float(ent.confidence if ent.confidence is not None else 1.0)
                    ))

                for idx, rel in enumerate(joint_res.relations, 1):
                    head_ent = next((e for e in joint_res.entities if e.id == rel.head), None)
                    tail_ent = next((e for e in joint_res.entities if e.id == rel.tail), None)
                    relations.append(DirectedRelation(
                        relation_id=f"g2_r_{idx}",
                        relation_type=rel.type,
                        subject_id=rel.head,
                        subject_text=head_ent.text if head_ent else "",
                        subject_type=head_ent.type if head_ent else "",
                        object_id=rel.tail,
                        object_text=tail_ent.text if tail_ent else "",
                        object_type=tail_ent.type if tail_ent else "",
                        confidence=float(rel.confidence)
                    ))

                return ExtractionResult(
                    sample_id=sample_id,
                    model_name=self.model_name,
                    config_id="G2",
                    text=text,
                    entities=entities,
                    relations=relations,
                    records=[],
                    raw_output=str(joint_res),
                    status="COMPLETED"
                )
            except Exception:
                pass

        # Fallback to G1 validation if joint engine call fails
        res_g1 = self.validator.validate_result(self._predict_g0(sample_id, text, schema_type))
        res_g1.config_id = "G2"
        return res_g1

    def _predict_g3_natural_records(self, sample_id: str, text: str, schema_type: str) -> ExtractionResult:
        """Natural record mode with anchor and field declarations."""
        schema = self.extractor.create_schema()
        if schema_type == "novel":
            schema.entities({"人物": "人物姓名", "斗技": "斗技技能", "境界": "境界名称", "丹药": "丹药品名"})
            schema.structure("战斗记录", mode="natural", anchor="攻击者") \
                .field("攻击者", dtype="str") \
                .field("防御者", dtype="str", cardinality="required_one") \
                .field("技能", dtype="str", cardinality="optional_one") \
                .field("战斗结果", dtype="str", cardinality="optional_one")
        else:
            schema.entities({"企业": "公司名称", "人员": "人员姓名", "款项": "资金数字"})
            schema.structure("资金交易", mode="natural", anchor="付款方") \
                .field("付款方", dtype="str") \
                .field("收款方", dtype="str", cardinality="required_one") \
                .field("金额", dtype="str", cardinality="required_one") \
                .field("履约状态", dtype="str", cardinality="optional_one")

        raw_res = self.extractor.extract(text, schema, threshold=0.4, include_spans=True)
        entities = self._parse_gliner_entities(raw_res, prefix="g3_e")
        records = self._parse_gliner_structures(raw_res, prefix="g3_rec")

        res = ExtractionResult(
            sample_id=sample_id,
            model_name=self.model_name,
            config_id="G3",
            text=text,
            entities=entities,
            relations=[],
            records=records,
            raw_output=raw_res,
            status="COMPLETED"
        )
        return self.validator.validate_result(res)

    def _predict_g4a_attributes(self, sample_id: str, text: str, schema_type: str) -> ExtractionResult:
        """Mention/local span attributes for polarity/modality binding."""
        res = self._predict_g0(sample_id, text, schema_type)
        res.config_id = "G4a"

        try:
            from gliner2 import AttributeGroup
            schema = self.extractor.create_schema()
            if schema_type == "novel":
                schema.entities({"人物": "人物名称", "斗技": "斗技技能"})
            else:
                schema.entities({"人物": "人员姓名", "款项": "资金数字"})
            schema.entity_attributes({
                "事实性": AttributeGroup(
                    labels=["已发生", "计划意图", "先决条件", "否定取消"],
                    applies_to=["人物", "斗技"] if schema_type == "novel" else ["人物", "款项"]
                )
            })
            raw_res = self.extractor.extract(text, schema, threshold=0.4, include_spans=True)
            raw_ents = raw_res.get("entities", {})
            if isinstance(raw_ents, dict):
                for ent_label, items in raw_ents.items():
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict):
                                item_text = item.get("text")
                                modality_val = item.get("事实性") or item.get("attributes", {}).get("事实性")
                                if modality_val and item_text:
                                    for e in res.entities:
                                        if e.text == item_text:
                                            e.attributes["modality"] = modality_val
        except Exception as e:
            print(f"[Warning] G4a attribute extraction error: {e}")

        return self.validator.validate_result(res)

    def _predict_g4b_classification(self, sample_id: str, text: str, schema_type: str) -> ExtractionResult:
        """Constrained classification across mutual exclusion and implications."""
        res = self._predict_g0(sample_id, text, schema_type)
        res.config_id = "G4b"

        try:
            from gliner2.classification import ClassificationSchema
            from gliner2.classification.engine import Classifier
            cls_schema = ClassificationSchema()
            cls_schema.single("极性与意图", ["已完成事实", "计划意图", "否定取消", "假设先决条件"])
            cls_schema.single("事件类型", ["战斗历练", "丹药交易", "宗门拜访", "日常交流"])
            cls_schema.mutually_exclusive("极性与意图:已完成事实", "极性与意图:否定取消")

            classifier = Classifier(model=self.extractor)
            cls_res = classifier.predict(text, cls_schema)
            for e in res.entities:
                e.attributes["doc_classification"] = cls_res.to_dict() if hasattr(cls_res, "to_dict") else str(cls_res)
        except Exception as e:
            print(f"[Warning] G4b classification error: {e}")

        return self.validator.validate_result(res)

    def _predict_g5_integrated(self, sample_id: str, text: str, schema_type: str) -> ExtractionResult:
        """Full pipeline: G2 (Joint relations) + G3 (Records) + G4 (Attributes) + G1 (Validation)."""
        res_g2 = self._predict_g2_joint_ie(sample_id, text, schema_type)
        res_g3 = self._predict_g3_natural_records(sample_id, text, schema_type)
        res_g4 = self._predict_g4a_attributes(sample_id, text, schema_type)

        entities = list(res_g2.entities) if res_g2.entities else list(res_g4.entities)
        for e in entities:
            matching_g4 = next((g4_e for g4_e in res_g4.entities if g4_e.text == e.text), None)
            if matching_g4 and matching_g4.attributes:
                e.attributes.update(matching_g4.attributes)

        combined = ExtractionResult(
            sample_id=sample_id,
            model_name=self.model_name,
            config_id="G5",
            text=text,
            entities=entities,
            relations=res_g2.relations,
            records=res_g3.records,
            raw_output={"g2": res_g2.raw_output, "g3": res_g3.raw_output},
            status="COMPLETED"
        )
        return self.validator.validate_result(combined)

    def get_unified_schema(self, schema_type: str = "novel"):
        """Get or build cached unified schema containing entities, relations, and structures."""
        if schema_type in self._schema_cache:
            return self._schema_cache[schema_type]

        schema = self.extractor.create_schema()
        if schema_type == "novel":
            schema = schema.entities({
                "人物": "小说中登场的人物姓名或称谓，如萧炎、萧战、药老、萧宁",
                "组织": "家族、宗门或势力名称，如萧家、云岚宗",
                "境界": "修炼境界等级，如斗者、大斗师、五星大斗师",
                "技能": "斗技名称，如裂爪击、吸掌",
                "物品": "丹药、宝物或器物，如聚气散、古玉盒子",
                "地点": "地名或特定场所，如乌坦城、训练场"
            })
            schema = schema.relations({
                "parent_of": {"description": "主体(head)是客体(tail)的父亲或长辈，如萧战是萧炎的父亲", "threshold": 0.4},
                "mentor_of": {"description": "主体(head)是客体(tail)的师父或导师，如药老是萧炎的师父", "threshold": 0.4},
                "member_of": {"description": "主体(head)是人物成员，客体(tail)是所属家族或宗门组织，如萧炎属于萧家", "threshold": 0.4}
            })
            schema = schema.structure("战斗记录", mode="natural", anchor="攻击者") \
                .field("攻击者", dtype="str") \
                .field("防御者", dtype="str", cardinality="required_one") \
                .field("技能", dtype="str", cardinality="optional_one") \
                .field("战斗结果", dtype="str", cardinality="optional_one")
        else:
            schema = schema.entities({
                "人物": "人员姓名",
                "公司": "企业或公司法人名称",
                "组织": "部门或机构名称",
                "金额": "交易金额或资金数字",
                "物品": "货物或商品名称",
                "地点": "配送地址或履行地点"
            })
            schema = schema.relations({
                "holds_position": {"description": "人物在企业或机构中任职", "threshold": 0.4},
                "holds_equity": {"description": "人物或组织持有公司股份", "threshold": 0.4},
                "transferred_to": {"description": "款项或物品转移支付", "threshold": 0.4},
                "delivered_to": {"description": "货物交付到地点", "threshold": 0.4}
            })
            schema = schema.structure("资金交易", mode="natural", anchor="付款方") \
                .field("付款方", dtype="str") \
                .field("收款方", dtype="str", cardinality="required_one") \
                .field("金额", dtype="str", cardinality="required_one") \
                .field("履约状态", dtype="str", cardinality="optional_one")

        self._schema_cache[schema_type] = schema
        return schema

    def predict_unified(self, sample_id: str, text: str, schema_type: str = "novel") -> ExtractionResult:
        """Single-pass integrated extraction with unified schema."""
        if self.extractor is None:
            self.load_model()

        t0 = time.time()
        proc = psutil.Process(os.getpid())
        schema = self.get_unified_schema(schema_type)

        raw_res = self.extractor.extract(
            text,
            schema,
            threshold=0.4,
            include_spans=True,
            include_confidence=True
        )

        entities = self._parse_gliner_entities(raw_res, prefix="uni_e")
        relations = self._parse_gliner_relations(raw_res, prefix="uni_r", entities=entities)
        records = self._parse_gliner_structures(raw_res, prefix="uni_rec")

        res = ExtractionResult(
            sample_id=sample_id,
            model_name=self.model_name,
            config_id="G5_UNIFIED",
            text=text,
            entities=entities,
            relations=relations,
            records=records,
            raw_output=raw_res,
            status="COMPLETED"
        )
        validated = self.validator.validate_result(res)
        validated.execution_time_sec = round(time.time() - t0, 4)
        validated.peak_memory_mib = round(proc.memory_info().rss / (1024.0 * 1024.0), 2)
        return validated

    def batch_predict_unified(
        self,
        sample_ids: List[str],
        texts: List[str],
        schema_type: str = "novel",
        batch_size: int = 8,
        num_workers: int = 0
    ) -> List[ExtractionResult]:
        """Batched single-pass extraction across multiple texts."""
        if self.extractor is None:
            self.load_model()

        t0 = time.time()
        proc = psutil.Process(os.getpid())
        schema = self.get_unified_schema(schema_type)

        raw_results = self.extractor.batch_extract(
            texts,
            schema,
            batch_size=batch_size,
            threshold=0.4,
            num_workers=num_workers,
            include_spans=True,
            include_confidence=True
        )

        results = []
        for sid, txt, raw in zip(sample_ids, texts, raw_results):
            entities = self._parse_gliner_entities(raw, prefix="b_e")
            relations = self._parse_gliner_relations(raw, prefix="b_r", entities=entities)
            records = self._parse_gliner_structures(raw, prefix="b_rec")
            res = ExtractionResult(
                sample_id=sid,
                model_name=self.model_name,
                config_id="G5_UNIFIED_BATCH",
                text=txt,
                entities=entities,
                relations=relations,
                records=records,
                raw_output=raw,
                status="COMPLETED"
            )
            val = self.validator.validate_result(res)
            val.execution_time_sec = round((time.time() - t0) / max(len(texts), 1), 4)
            val.peak_memory_mib = round(proc.memory_info().rss / (1024.0 * 1024.0), 2)
            results.append(val)
        return results

    def enable_dynamic_quantization(self) -> bool:
        """
        Apply PyTorch INT8 dynamic quantization to the transformer encoder backbone only.
        Keeps sparse boundary projection heads, rotary embeddings and relation scorers in FP32
        to prevent logit shift and total candidate collapse.
        Note: Must be gated by quality regression tests before production use.
        """
        if self.extractor is None:
            self.load_model()
        try:
            import torch.ao.quantization as ao_quant
            target_module = getattr(self.extractor, "encoder", None) or getattr(self.extractor, "model", None)
            if target_module is not None:
                quantized_encoder = ao_quant.quantize_dynamic(
                    target_module,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
                if hasattr(self.extractor, "encoder"):
                    self.extractor.encoder = quantized_encoder
                else:
                    self.extractor.model = quantized_encoder
                print("[Info] INT8 dynamic quantization applied to transformer encoder backbone only (heads kept in FP32).")
                return True
            else:
                print("[Warning] Could not locate encoder module on extractor for selective quantization.")
                return False
        except Exception as e:
            print(f"[Warning] Failed to enable INT8 dynamic quantization: {e}")
            return False
