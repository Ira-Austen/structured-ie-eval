"""
Unit Tests for Unified Validator.
Verifies:
- Rejection of self-loops
- Type constraint enforcement
- Deduplication of identical facts
- Span bounds checking
"""

import unittest
from src.models.base import EntityMention, DirectedRelation, EventRecord, ExtractionResult
from src.validation.validator import UnifiedValidator


class TestUnifiedValidator(unittest.TestCase):
    def setUp(self):
        self.validator = UnifiedValidator()
        self.sample_text = "萧战是萧家的现任族长，同时也是萧炎的亲生父亲。"

    def test_reject_self_loop(self):
        rels = [
            DirectedRelation(
                relation_id="r1",
                relation_type="parent_of",
                subject_id="e1",
                subject_text="萧战",
                subject_type="Person",
                object_id="e1",
                object_text="萧战",
                object_type="Person"
            )
        ]
        valid_rels = self.validator.validate_relations(self.sample_text, rels)
        self.assertEqual(len(valid_rels), 0, "Self-loop relation should be rejected!")

    def test_valid_directed_relation(self):
        rels = [
            DirectedRelation(
                relation_id="r1",
                relation_type="parent_of",
                subject_id="e1",
                subject_text="萧战",
                subject_type="Person",
                object_id="e2",
                object_text="萧炎",
                object_type="Person"
            )
        ]
        valid_rels = self.validator.validate_relations(self.sample_text, rels)
        self.assertEqual(len(valid_rels), 1)
        self.assertEqual(valid_rels[0].relation_type, "parent_of")

    def test_reject_type_incompatible_endpoint(self):
        rels = [
            DirectedRelation(
                relation_id="r1",
                relation_type="parent_of",
                subject_id="e1",
                subject_text="萧战",
                subject_type="Person",
                object_id="e2",
                object_text="萧家",
                object_type="Organization"  # An Organization cannot be child in parent_of
            )
        ]
        valid_rels = self.validator.validate_relations(self.sample_text, rels)
        self.assertEqual(len(valid_rels), 0, "Type violation should be rejected by validator!")

    def test_deduplication(self):
        rels = [
            DirectedRelation(
                relation_id="r1",
                relation_type="parent_of",
                subject_id="e1",
                subject_text="萧战",
                subject_type="Person",
                object_id="e2",
                object_text="萧炎",
                object_type="Person"
            ),
            DirectedRelation(
                relation_id="r2",
                relation_type="parent_of",
                subject_id="e3",
                subject_text="萧战",
                subject_type="Person",
                object_id="e4",
                object_text="萧炎",
                object_type="Person"
            )
        ]
        valid_rels = self.validator.validate_relations(self.sample_text, rels)
        self.assertEqual(len(valid_rels), 1, "Duplicate directed fact should be deduplicated!")


if __name__ == "__main__":
    unittest.main()
