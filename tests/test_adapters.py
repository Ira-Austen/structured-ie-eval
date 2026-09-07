"""
Unit Tests for Model Adapters and Schemas.
Tests:
- UIE direction normalization logic
- RuleBaseline extraction
- Schema representation
"""

import unittest
from src.models.baseline_rules import RuleBaselineModel
from src.models.gliner_adapter import GLiNERAdapter
from src.models.base import EntityMention


class TestAdapters(unittest.TestCase):
    def test_rule_baseline(self):
        text = "他便是萧家现任族长，同时也是萧炎的父亲，五星大斗师，萧战！"
        model = RuleBaselineModel()
        res = model.predict("test_01", text)
        self.assertEqual(res.status, "COMPLETED")
        self.assertTrue(any(e.text == "萧炎" for e in res.entities))
        self.assertTrue(any(e.text == "萧战" for e in res.entities))
        self.assertTrue(any(r.relation_type == "parent_of" and r.subject_text == "萧战" and r.object_text == "萧炎" for r in res.relations))

    def test_gliner_parse_structures_top_level(self):
        adapter = GLiNERAdapter("dummy_model")
        raw = {
            "战斗记录": [
                {
                    "攻击者": {"text": "萧宁", "start": 0, "end": 2},
                    "防御者": {"text": "萧炎", "start": 5, "end": 7},
                    "技能": "裂爪击",
                    "战斗结果": "被击退"
                }
            ]
        }
        records = adapter._parse_gliner_structures(raw)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].event_type, "战斗记录")
        self.assertEqual(records[0].anchor_text, "萧宁")
        self.assertEqual(records[0].roles["防御者"], "萧炎")
        self.assertEqual(records[0].roles["技能"], "裂爪击")

    def test_gliner_parse_relations_with_entities(self):
        adapter = GLiNERAdapter("dummy_model")
        ents = [
            EntityMention("e1", "萧战", "人物", 0, 2),
            EntityMention("e2", "萧家", "组织", 5, 7)
        ]
        raw = {
            "relation_extraction": {
                "member_of": [
                    {
                        "head": {"text": "萧战", "start": 0, "end": 2, "confidence": 0.95},
                        "tail": {"text": "萧家", "start": 5, "end": 7, "confidence": 0.95}
                    }
                ]
            }
        }
        rels = adapter._parse_gliner_relations(raw, entities=ents)
        self.assertEqual(len(rels), 1)
        self.assertEqual(rels[0].relation_type, "member_of")
        self.assertEqual(rels[0].subject_text, "萧战")
        self.assertEqual(rels[0].subject_type, "人物")
        self.assertEqual(rels[0].object_text, "萧家")
        self.assertEqual(rels[0].object_type, "组织")


if __name__ == "__main__":
    unittest.main()
