"""Comprehensive metrics calculation for Intent, Escalation, and Groundedness."""

from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)
from src.intents.taxonomy import INTENTS


def compute_intent_metrics(y_true: List[str], y_pred: List[str]) -> Dict:
    """Computes standard multiclass intent classification metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=INTENTS)
    per_class_report = classification_report(y_true, y_pred, labels=INTENTS, output_dict=True, zero_division=0)

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class_report,
    }


def compute_escalation_metrics(gold_should_escalate: List[bool], system_decisions: List[str]) -> Dict:
    """Computes critical safety and operational metrics for the escalation policy.

    Args:
        gold_should_escalate: Ground-truth boolean where True = requires human escalation.
        system_decisions: List of strings ("AUTO_HANDLE" or "ESCALATE").

    Returns:
        Dict containing coverage, False Auto-Handle Rate (FAHR), escalation precision, etc.
    """
    total = len(gold_should_escalate)
    assert total == len(system_decisions), "Length mismatch"

    n_auto = 0
    n_esc = 0
    n_false_auto = 0  # Auto-handled by system, but gold requires human (CRITICAL SAFETY FAILURE)
    n_true_esc = 0    # Escalated by system, and gold requires human (CORRECT SAFETY TRIAGE)
    n_false_esc = 0   # Escalated by system, but could have been auto-handled (OPERATIONAL INEFFICIENCY)
    n_true_auto = 0   # Auto-handled by system, and gold confirmed auto-handle safe

    for gold_esc, sys_dec in zip(gold_should_escalate, system_decisions):
        is_sys_esc = (sys_dec == "ESCALATE")

        if not is_sys_esc:
            n_auto += 1
            if gold_esc:
                n_false_auto += 1
            else:
                n_true_auto += 1
        else:
            n_esc += 1
            if gold_esc:
                n_true_esc += 1
            else:
                n_false_esc += 1

    gold_esc_total = sum(1 for g in gold_should_escalate if g)
    gold_auto_total = total - gold_esc_total

    # Metrics
    coverage = n_auto / total if total > 0 else 0.0
    fahr = n_false_auto / n_auto if n_auto > 0 else 0.0
    esc_precision = n_true_esc / n_esc if n_esc > 0 else 0.0
    esc_recall = n_true_esc / gold_esc_total if gold_esc_total > 0 else 0.0
    unnecessary_esc_rate = n_false_esc / n_esc if n_esc > 0 else 0.0

    return {
        "total_evaluated": total,
        "n_auto_handled": n_auto,
        "n_escalated": n_esc,
        "coverage": round(coverage, 4),
        "false_auto_handle_rate": round(fahr, 4),  # CRITICAL SAFETY METRIC
        "escalation_precision": round(esc_precision, 4),
        "escalation_recall": round(esc_recall, 4),
        "unnecessary_escalation_rate": round(unnecessary_esc_rate, 4),
        "n_false_auto_handles": n_false_auto,
        "n_true_auto_handles": n_true_auto,
        "n_true_escalations": n_true_esc,
        "n_false_escalations": n_false_esc,
    }
