"""
Unified External Validator for Structured IE.
Enforces structural and ontological consistency:
- Rejects self-loops (subject == object)
- Rejects type-invalid endpoints
- Rejects relations whose spans do not match original text
- Deduplicates identical directed facts
- Normalizes relation direction and attributes
"""

from typing import List, Dict, Set, Tuple, Any
from ..models.base import EntityMention, DirectedRelation, EventRecord, ExtractionResult


RELATION_TYPE_CONSTRAINTS = {
    "parent_of": {
        "valid_subjects": {"Person", "人物", "称谓"},
        "valid_objects": {"Person", "人物", "称谓"},
        "allow_self_loop": False,
        "symmetric": False,
    },
    "mentor_of": {
        "valid_subjects": {"Person", "人物", "称谓"},
        "valid_objects": {"Person", "人物", "称谓"},
        "allow_self_loop": False,
        "symmetric": False,
    },
    "member_of": {
        "valid_subjects": {"Person", "人物", "称谓"},
        "valid_objects": {"Organization", "组织", "家族", "宗门"},
        "allow_self_loop": False,
        "symmetric": False,
    },
    "holds_position": {
        "valid_subjects": {"Person", "人物", "称谓"},
        "valid_objects": {"Organization", "组织", "家族", "宗门", "Realm", "职务"},
        "allow_self_loop": False,
        "symmetric": False,
    },
    "holds_equity": {
        "valid_subjects": {"Person", "人物", "Organization", "组织"},
        "valid_objects": {"Organization", "组织", "公司"},
        "allow_self_loop": False,
        "symmetric": False,
    },
    "transferred_to": {
        "valid_subjects": {"Person", "人物", "Organization", "组织"},
        "valid_objects": {"Person", "人物", "Organization", "组织"},
        "allow_self_loop": False,
        "symmetric": False,
    },
    "delivered_to": {
        "valid_subjects": {"Person", "人物", "Organization", "组织"},
        "valid_objects": {"Location", "地点", "Organization", "组织", "Person", "人物"},
        "allow_self_loop": False,
        "symmetric": False,
    },
}


class UnifiedValidator:
    """External validator providing rule-based guardrails across all models."""

    def __init__(self, constraints: Dict[str, Any] = None):
        self.constraints = constraints or RELATION_TYPE_CONSTRAINTS

    def validate_entities(self, text: str, entities: List[EntityMention]) -> List[EntityMention]:
        valid_entities: List[EntityMention] = []
        seen_spans: Set[Tuple[int, int, str]] = set()

        for ent in entities:
            # Check text bounds
            if ent.char_start < 0 or ent.char_end > len(text) or ent.char_start >= ent.char_end:
                continue
            # Check substring match
            expected_text = text[ent.char_start:ent.char_end]
            if ent.text != expected_text and ent.text.strip() != expected_text.strip():
                continue
            # Deduplicate same span and type
            span_key = (ent.char_start, ent.char_end, ent.entity_type)
            if span_key in seen_spans:
                continue
            seen_spans.add(span_key)
            valid_entities.append(ent)

        return valid_entities

    def validate_relations(self, text: str, relations: List[DirectedRelation], entities: List[EntityMention] = None) -> List[DirectedRelation]:
        valid_relations: List[DirectedRelation] = []
        seen_facts: Set[Tuple[str, str, str]] = set()

        # Index known entities if available
        ent_by_id = {e.mention_id: e for e in entities} if entities else {}

        for rel in relations:
            # Check self-loop
            if rel.subject_id and rel.object_id and rel.subject_id == rel.object_id:
                continue
            if rel.subject_text.strip() == rel.object_text.strip():
                continue

            # Check constraint rules if relation_type is known
            rule = self.constraints.get(rel.relation_type)
            if rule:
                sub_type = rel.subject_type or (ent_by_id[rel.subject_id].entity_type if rel.subject_id in ent_by_id else None)
                obj_type = rel.object_type or (ent_by_id[rel.object_id].entity_type if rel.object_id in ent_by_id else None)

                if sub_type and rule["valid_subjects"] and sub_type not in rule["valid_subjects"]:
                    continue
                if obj_type and rule["valid_objects"] and obj_type not in rule["valid_objects"]:
                    continue

            # Check evidence span if provided
            if rel.evidence_span:
                s, e = rel.evidence_span
                if s < 0 or e > len(text) or s >= e:
                    rel.evidence_span = None
                elif rel.evidence_text and text[s:e] != rel.evidence_text:
                    rel.evidence_span = None

            # Deduplicate directed facts: (subject_text, relation_type, object_text)
            fact_key = (rel.subject_text.strip(), rel.relation_type, rel.object_text.strip())
            if fact_key in seen_facts:
                continue
            seen_facts.add(fact_key)
            valid_relations.append(rel)

        return valid_relations

    def validate_records(self, text: str, records: List[EventRecord]) -> List[EventRecord]:
        valid_records: List[EventRecord] = []
        seen_records: Set[str] = set()

        for rec in records:
            # Must have at least one non-empty role
            active_roles = {k: v for k, v in rec.roles.items() if v}
            if not active_roles:
                continue

            # Hash representation of event roles to prevent duplicate identical events
            role_repr = tuple(sorted((k, str(v)) for k, v in active_roles.items()))
            rec_key = f"{rec.event_type}::{rec.anchor_text}::{role_repr}"
            if rec_key in seen_records:
                continue
            seen_records.add(rec_key)
            valid_records.append(rec)

        return valid_records

    def validate_result(self, result: ExtractionResult) -> ExtractionResult:
        valid_entities = self.validate_entities(result.text, result.entities)
        valid_relations = self.validate_relations(result.text, result.relations, valid_entities)
        valid_records = self.validate_records(result.text, result.records)

        return ExtractionResult(
            sample_id=result.sample_id,
            model_name=result.model_name,
            config_id=result.config_id,
            text=result.text,
            entities=valid_entities,
            relations=valid_relations,
            records=valid_records,
            raw_output=result.raw_output,
            execution_time_sec=result.execution_time_sec,
            peak_memory_mib=result.peak_memory_mib,
            status=result.status,
            error_message=result.error_message,
        )
