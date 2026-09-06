"""
Unit Tests for Metrics Evaluator.
Verifies:
- Strict span + type P/R/F1
- Directed fact evaluation and reverse-direction error detection
- Multi-role event record complete match and cross-pairing metrics
- Modality & negation discrimination metrics
"""

import unittest
from src.metrics.evaluator import MetricsEvaluator


class TestMetricsEvaluator(unittest.TestCase):
    def test_mentions_metrics(self):
        gold = [
            {"char_start": 0, "char_end": 2, "entity_type": "Person"},
            {"char_start": 5, "char_end": 7, "entity_type": "Organization"}
        ]
        pred = [
            {"char_start": 0, "char_end": 2, "entity_type": "Person"}, # TP
            {"char_start": 10, "char_end": 12, "entity_type": "Location"} # FP
        ]
        res = MetricsEvaluator.evaluate_mentions(pred, gold)
        self.assertEqual(res["tp"], 1)
        self.assertEqual(res["fp"], 1)
        self.assertEqual(res["fn"], 1)
        self.assertAlmostEqual(res["f1"], 0.5)

    def test_reverse_direction_detection(self):
        gold = [
            {"subject_text": "萧战", "relation_type": "parent_of", "object_text": "萧炎"}
        ]
        # Prediction has the reverse direction (child -> parent)
        pred = [
            {"subject_text": "萧炎", "relation_type": "parent_of", "object_text": "萧战"}
        ]
        res = MetricsEvaluator.evaluate_directed_relations(pred, gold)
        self.assertEqual(res["tp"], 0)
        self.assertEqual(res["fp"], 1)
        self.assertEqual(res["fn"], 1)
        self.assertEqual(res["reverse_errors"], 1, "Reverse direction error must be flagged!")
        self.assertEqual(res["reverse_error_rate"], 1.0)

    def test_record_complete_match_and_cross_pairing(self):
        gold = [
            {"roles": {"payer": "公司A", "payee": "公司B", "amount": "100万"}},
            {"roles": {"payer": "公司C", "payee": "公司D", "amount": "200万"}}
        ]
        # Cross-paired prediction: A paid D 100万, C paid B 200万
        pred = [
            {"roles": {"payer": "公司A", "payee": "公司D", "amount": "100万"}},
            {"roles": {"payer": "公司C", "payee": "公司B", "amount": "200万"}}
        ]
        res = MetricsEvaluator.evaluate_records(pred, gold)
        self.assertEqual(res["complete_matches"], 0, "Cross-paired records should not complete match!")
        self.assertGreater(res["cross_pairing_errors"], 0, "Cross pairing errors must be detected!")

    def test_modality_negation_discrimination(self):
        gold = [
            {"modality": "intended", "polarity": "positive"},
            {"modality": "actual", "polarity": "negative"}
        ]
        # Model falsely predicted both as actual positive events
        pred = [
            {"modality": "actual", "polarity": "positive"},
            {"modality": "actual", "polarity": "positive"}
        ]
        res = MetricsEvaluator.evaluate_modality_negation(pred, gold)
        self.assertEqual(res["total_non_actual_gold"], 2)
        self.assertEqual(res["false_actual_predictions"], 2)
        self.assertEqual(res["false_actual_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
