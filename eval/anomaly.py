"""Anomaly-detection eval: per-category precision/recall/F1, confusion,
and macro/micro rollups.

The evaluator is set-based: for each invoice, compare the set of predicted
categories to the set of injected categories.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from src.schemas import AnomalyCategory, AnomalyEvalItem, AnomalyReport


@dataclass
class CategoryScore:
    category: str
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f


def score_reports(
    reports: list[AnomalyReport],
    gold: list[AnomalyEvalItem],
) -> dict:
    gold_by_id = {g.invoice_id: set(g.injected_categories) for g in gold}
    pred_by_id = {
        r.invoice_id: {a.category for a in r.anomalies} for r in reports
    }

    per_cat_tp: Counter[AnomalyCategory] = Counter()
    per_cat_fp: Counter[AnomalyCategory] = Counter()
    per_cat_fn: Counter[AnomalyCategory] = Counter()

    invoice_ids = set(gold_by_id) | set(pred_by_id)
    is_flagged_tp = is_flagged_fp = is_flagged_tn = is_flagged_fn = 0

    for inv_id in invoice_ids:
        g = gold_by_id.get(inv_id, set())
        p = pred_by_id.get(inv_id, set())
        for c in p & g:
            per_cat_tp[c] += 1
        for c in p - g:
            per_cat_fp[c] += 1
        for c in g - p:
            per_cat_fn[c] += 1

        # invoice-level flag confusion (any predicted anomaly vs any gold anomaly)
        gold_flag = bool(g)
        pred_flag = bool(p)
        if gold_flag and pred_flag:
            is_flagged_tp += 1
        elif not gold_flag and pred_flag:
            is_flagged_fp += 1
        elif not gold_flag and not pred_flag:
            is_flagged_tn += 1
        else:
            is_flagged_fn += 1

    categories = [c for c in AnomalyCategory]
    per_cat: list[CategoryScore] = []
    for c in categories:
        tp = per_cat_tp[c]
        fp = per_cat_fp[c]
        fn = per_cat_fn[c]
        p, r, f = _prf(tp, fp, fn)
        per_cat.append(CategoryScore(c.value, tp, fp, fn, p, r, f))

    # Micro (pool categories)
    total_tp = sum(per_cat_tp.values())
    total_fp = sum(per_cat_fp.values())
    total_fn = sum(per_cat_fn.values())
    micro_p, micro_r, micro_f = _prf(total_tp, total_fp, total_fn)

    # Macro (mean over categories that have gold OR pred instances)
    active = [cs for cs in per_cat if (cs.tp + cs.fp + cs.fn) > 0]
    macro_p = sum(cs.precision for cs in active) / len(active) if active else 0.0
    macro_r = sum(cs.recall for cs in active) / len(active) if active else 0.0
    macro_f = sum(cs.f1 for cs in active) / len(active) if active else 0.0

    flag_p, flag_r, flag_f = _prf(is_flagged_tp, is_flagged_fp, is_flagged_fn)

    return {
        "per_category": [cs.__dict__ for cs in per_cat],
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f,
                   "tp": total_tp, "fp": total_fp, "fn": total_fn},
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f},
        "invoice_flag": {
            "precision": flag_p, "recall": flag_r, "f1": flag_f,
            "tp": is_flagged_tp, "fp": is_flagged_fp,
            "tn": is_flagged_tn, "fn": is_flagged_fn,
        },
    }
