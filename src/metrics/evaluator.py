"""
Comprehensive Metric Evaluator for Structured IE.
Calculates:
1. Mention strict span + type P/R/F1
2. Directed Fact P/R/F1 and Reverse Direction Error Rate
3. Event Record Role F1, Complete Match Rate, Field Cross-Pairing Rate, Merge/Split Errors
4. Modality & Negation Discrimination (Intent/Condition/Negation vs Actual)
5. Throughput and Facts-per-Second Efficiency
"""

from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict
import numpy as np


class MetricsEvaluator:
    def __init__(self):
        pass

    @staticmethod
    def evaluate_mentions(predictions: List[Dict[str, Any]], gold: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculates strict span + type P/R/F1 for entity mentions.
        Gold: list of dicts with 'char_start', 'char_end', 'entity_type' (or 'type', 'span')
        Pred: list of EntityMention dicts
        """
        def to_key(m):
            if "char_start" in m:
                return (m["char_start"], m["char_end"], m.get("entity_type") or m.get("type"))
            if "span" in m:
                return (m["span"][0], m["span"][1], m.get("entity_type") or m.get("type"))
            return None

        gold_keys = set(filter(None, [to_key(g) for g in gold]))
        pred_keys = set(filter(None, [to_key(p) for p in predictions]))

        tp = len(gold_keys & pred_keys)
        fp = len(pred_keys - gold_keys)
        fn = len(gold_keys - pred_keys)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "gold_count": len(gold_keys),
            "pred_count": len(pred_keys),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    @staticmethod
    def evaluate_directed_relations(predictions: List[Dict[str, Any]], gold: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates deduplicated directed facts: (subject_text, relation_type, object_text).
        Also explicitly measures reverse-direction errors: (object_text, relation_type, subject_text).
        """
        def to_fact_tuple(r):
            sub = r.get("subject_text") or (r.get("subject", {}).get("text") if isinstance(r.get("subject"), dict) else str(r.get("subject", "")))
            obj = r.get("object_text") or (r.get("object", {}).get("text") if isinstance(r.get("object"), dict) else str(r.get("object", "")))
            rel = r.get("relation_type") or r.get("type") or ""
            return (sub.strip(), rel.strip(), obj.strip())

        gold_facts = set(to_fact_tuple(g) for g in gold if to_fact_tuple(g)[0] and to_fact_tuple(g)[2])
        pred_facts = set(to_fact_tuple(p) for p in predictions if to_fact_tuple(p)[0] and to_fact_tuple(p)[2])

        tp = len(gold_facts & pred_facts)
        fp = len(pred_facts - gold_facts)
        fn = len(gold_facts - pred_facts)

        # Detect reverse direction errors:
        reverse_errors = 0
        gold_reversed = set((obj, rel, sub) for sub, rel, obj in gold_facts if sub != obj)
        for pred in pred_facts:
            if pred in gold_reversed and pred not in gold_facts:
                reverse_errors += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        reverse_error_rate = reverse_errors / len(pred_facts) if pred_facts else 0.0

        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "gold_count": len(gold_facts),
            "pred_count": len(pred_facts),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "reverse_errors": reverse_errors,
            "reverse_error_rate": round(reverse_error_rate, 4),
        }

    @staticmethod
    def evaluate_records(predictions: List[Dict[str, Any]], gold: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates event records:
        - Role-level P/R/F1
        - Complete match rate (all gold roles exactly matched in an event)
        - Field cross-pairing errors
        - Merge/split instance errors
        """
        gold_records = gold
        pred_records = predictions

        total_gold_records = len(gold_records)
        total_pred_records = len(pred_records)

        complete_matches = 0
        total_gold_roles = 0
        matched_roles = 0
        total_pred_roles = 0
        cross_pairing_errors = 0

        # Helper to extract roles dict
        def extract_roles(rec):
            raw_roles = rec.get("roles", {})
            cleaned = {}
            for k, v in raw_roles.items():
                if isinstance(v, dict):
                    cleaned[k] = v.get("text", "").strip()
                elif isinstance(v, str):
                    cleaned[k] = v.strip()
                elif isinstance(v, list) and v:
                    cleaned[k] = str(v[0]).strip()
            return {k: v for k, v in cleaned.items() if v}

        gold_role_list = [extract_roles(g) for g in gold_records]
        pred_role_list = [extract_roles(p) for p in pred_records]

        for g_roles in gold_role_list:
            total_gold_roles += len(g_roles)

        for p_roles in pred_role_list:
            total_pred_roles += len(p_roles)

        # Match pred records to gold records (Greedy bipartite matching by overlapping roles)
        matched_gold_indices = set()
        for p_idx, p_roles in enumerate(pred_role_list):
            best_match_idx = -1
            best_overlap = 0
            for g_idx, g_roles in enumerate(gold_role_list):
                if g_idx in matched_gold_indices:
                    continue
                overlap = sum(1 for k, v in p_roles.items() if g_roles.get(k) == v)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match_idx = g_idx

            if best_match_idx != -1 and best_overlap > 0:
                matched_gold_indices.add(best_match_idx)
                matched_roles += best_overlap
                g_roles = gold_role_list[best_match_idx]
                if best_overlap == len(g_roles) and len(p_roles) == len(g_roles):
                    complete_matches += 1

                # Check for cross pairing: pred role matches some other gold record's role
                for k, v in p_roles.items():
                    if g_roles.get(k) != v:
                        for other_idx, other_g in enumerate(gold_role_list):
                            if other_idx != best_match_idx and other_g.get(k) == v:
                                cross_pairing_errors += 1
                                break

        role_precision = matched_roles / total_pred_roles if total_pred_roles > 0 else 0.0
        role_recall = matched_roles / total_gold_roles if total_gold_roles > 0 else 0.0
        role_f1 = (2 * role_precision * role_recall) / (role_precision + role_recall) if (role_precision + role_recall) > 0 else 0.0

        complete_match_rate = complete_matches / total_gold_records if total_gold_records > 0 else 0.0
        cross_pairing_rate = cross_pairing_errors / total_pred_roles if total_pred_roles > 0 else 0.0

        # Instance merge / split
        merge_errors = max(0, total_gold_records - total_pred_records) if total_gold_records > 1 and total_pred_records == 1 else 0
        split_errors = max(0, total_pred_records - total_gold_records) if total_gold_records == 1 and total_pred_records > 1 else 0

        return {
            "total_gold_records": total_gold_records,
            "total_pred_records": total_pred_records,
            "complete_matches": complete_matches,
            "complete_match_rate": round(complete_match_rate, 4),
            "role_precision": round(role_precision, 4),
            "role_recall": round(role_recall, 4),
            "role_f1": round(role_f1, 4),
            "cross_pairing_errors": cross_pairing_errors,
            "cross_pairing_rate": round(cross_pairing_rate, 4),
            "merge_errors": merge_errors,
            "split_errors": split_errors,
        }

    @staticmethod
    def evaluate_modality_negation(predictions: List[Dict[str, Any]], gold: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates whether models correctly distinguish:
        - actual (occurred)
        - intended (future/attempted)
        - conditional (requires prerequisite)
        - negated (did not occur / cancelled / rejected)

        Critical metric: False Actual Rate (identifying an intended/negated/conditional event as actual)
        """
        total_non_actual_gold = 0
        false_actual_predictions = 0
        exact_modality_matches = 0
        total_evaluated = 0

        for g, p in zip(gold, predictions):
            g_mod = g.get("modality", "actual")
            g_pol = g.get("polarity", "positive")
            p_mod = p.get("modality", "actual")
            p_pol = p.get("polarity", "positive")

            is_non_actual = (g_mod in {"intended", "conditional", "hypothetical"} or g_pol == "negative")
            if is_non_actual:
                total_non_actual_gold += 1
                if p_mod == "actual" and p_pol == "positive":
                    false_actual_predictions += 1

            if g_mod == p_mod and g_pol == p_pol:
                exact_modality_matches += 1
            total_evaluated += 1

        false_actual_rate = false_actual_predictions / total_non_actual_gold if total_non_actual_gold > 0 else 0.0
        modality_accuracy = exact_modality_matches / total_evaluated if total_evaluated > 0 else 0.0

        return {
            "total_non_actual_gold": total_non_actual_gold,
            "false_actual_predictions": false_actual_predictions,
            "false_actual_rate": round(false_actual_rate, 4),
            "exact_modality_matches": exact_modality_matches,
            "modality_accuracy": round(modality_accuracy, 4),
        }
