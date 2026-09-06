"""
Rule-based Heuristic Baseline (B0).
Uses dictionary lookup and basic regex patterns for entity extraction and co-occurrence relations.
Serves as an empirical baseline to measure genuine neural value-add.
"""

import re
from typing import List, Dict, Any, Set
from .base import EntityMention, DirectedRelation, EventRecord, ExtractionResult


COMMON_PERSONS = {"萧炎", "萧战", "药老", "萧宁", "纳兰嫣然", "葛叶", "萧媚", "张伟", "李华", "王伟", "赵敏", "李雷", "韩梅梅"}
COMMON_ORGS = {"萧家", "云岚宗", "加玛帝国", "米特尔家族", "迦南学院", "华为", "腾讯", "阿里巴巴", "顺丰速运", "中通快递"}
COMMON_REALMS = {"斗之气", "斗者", "大斗师", "斗王", "斗皇", "斗宗", "斗尊", "斗圣", "斗帝", "五星大斗师", "炼药师"}
COMMON_ITEMS = {"聚气散", "筑基丹", "古玉盒子", "玄重尺"}
COMMON_SKILLS = {"裂爪击", "吸掌", "吹火掌", "八极崩"}


class RuleBaselineModel:
    def __init__(self):
        self.model_name = "RuleBaseline_B0"

    def predict(self, sample_id: str, text: str) -> ExtractionResult:
        entities: List[EntityMention] = []
        relations: List[DirectedRelation] = []
        records: List[EventRecord] = []

        # 1. Entity extraction via pattern search
        ent_idx = 1
        found_spans: Set[tuple] = set()

        for cat, word_list in [
            ("Person", COMMON_PERSONS),
            ("Organization", COMMON_ORGS),
            ("Realm", COMMON_REALMS),
            ("Item", COMMON_ITEMS),
            ("Skill", COMMON_SKILLS)
        ]:
            for w in word_list:
                start = 0
                while True:
                    pos = text.find(w, start)
                    if pos == -1:
                        break
                    span = (pos, pos + len(w), cat)
                    if span not in found_spans:
                        found_spans.add(span)
                        entities.append(EntityMention(
                            mention_id=f"rule_e_{ent_idx}",
                            text=w,
                            entity_type=cat,
                            char_start=pos,
                            char_end=pos + len(w),
                            confidence=0.9
                        ))
                        ent_idx += 1
                    start = pos + len(w)

        # 2. Directed Relation extraction via pattern triggers
        rel_idx = 1
        # Kinship: "萧炎的父亲...萧战" or "萧战...萧炎的父亲"
        if "父亲" in text:
            # Check for parent/child mentions in the text
            found_persons = [e.text for e in entities if e.entity_type == "Person"]
            if "萧炎" in found_persons and "萧战" in found_persons:
                relations.append(DirectedRelation(
                    relation_id=f"rule_r_{rel_idx}",
                    relation_type="parent_of",
                    subject_id="rule_sub",
                    subject_text="萧战",
                    subject_type="Person",
                    object_id="rule_obj",
                    object_text="萧炎",
                    object_type="Person",
                    confidence=0.8,
                    evidence_text="萧炎的父亲...萧战"
                ))
                rel_idx += 1

        # Membership: "萧家现任族长...萧战"
        clan_match = re.search(r"(\S+家族|\S+家|\S+宗)现任族长[，,、\s]*.*?([A-Z\u4e00-\u9fa5]{2,4})", text)
        if clan_match:
            org, person = clan_match.group(1), clan_match.group(2)
            clean_org = next((o for o in COMMON_ORGS if o in org), org)
            clean_person = next((p for p in COMMON_PERSONS if p in person), person)
            relations.append(DirectedRelation(
                relation_id=f"rule_r_{rel_idx}",
                relation_type="member_of",
                subject_id="rule_sub",
                subject_text=clean_person,
                subject_type="Person",
                object_id="rule_obj",
                object_text=clean_org,
                object_type="Organization",
                role="族长",
                confidence=0.8,
                evidence_text=clan_match.group(0)
            ))
            rel_idx += 1

        return ExtractionResult(
            sample_id=sample_id,
            model_name="RuleBaseline",
            config_id="B0",
            text=text,
            entities=entities,
            relations=relations,
            records=records,
            status="COMPLETED"
        )
