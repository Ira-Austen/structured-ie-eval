"""
Unit Tests for Model Adapters and Schemas.
Tests:
- UIE direction normalization logic
- RuleBaseline extraction
- Schema representation
"""

import unittest
from src.models.baseline_rules import RuleBaselineModel


class TestAdapters(unittest.TestCase):
    def test_rule_baseline(self):
        text = "他便是萧家现任族长，同时也是萧炎的父亲，五星大斗师，萧战！"
        model = RuleBaselineModel()
        res = model.predict("test_01", text)
        self.assertEqual(res.status, "COMPLETED")
        self.assertTrue(any(e.text == "萧炎" for e in res.entities))
        self.assertTrue(any(e.text == "萧战" for e in res.entities))
        self.assertTrue(any(r.relation_type == "parent_of" and r.subject_text == "萧战" and r.object_text == "萧炎" for r in res.relations))


if __name__ == "__main__":
    unittest.main()
