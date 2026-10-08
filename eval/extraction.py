"""Per-field extraction metrics.

Compares a predicted Invoice against a gold Invoice, field by field, using
normalizers that reflect what a human would consider "the same value" — dates
compared as ISO strings, decimals compared with tolerance, whitespace/case
normalized for text.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from src.schemas import Invoice, LineItem

DECIMAL_TOL = Decimal("0.01")

FIELDS = (
    "invoice_id",
    "vendor_name",
    "invoice_date",
    "due_date",
    "po_reference",
    "subtotal",
    "tax",
    "total",
    "currency",
)


def _norm_text(x: Any) -> str:
    return "" if x is None else " ".join(str(x).lower().split())


def _dec_close(a: Any, b: Any, tol: Decimal = DECIMAL_TOL) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(Decimal(str(a)) - Decimal(str(b))) <= tol


def _field_match(field: str, pred: Any, gold: Any) -> bool:
    if pred is None and gold is None:
        return True
    if field in {"subtotal", "tax", "total"}:
        return _dec_close(pred, gold)
    if field in {"invoice_date", "due_date"}:
        return (pred is None and gold is None) or (pred == gold)
    return _norm_text(pred) == _norm_text(gold)


def _line_item_key(li: LineItem) -> tuple:
    """Identity key for set-based line-item F1. Uses a fixed 2-decimal
    string for unit_price so that Decimal(342.50) and Decimal(342.5) —
    which compare equal numerically — also key equal. Prior version used
    str(Decimal), which was tripping on preserved-vs-dropped trailing
    zeros in the LLM's numeric output."""
    return (_norm_text(li.description), float(li.quantity), f"{float(li.unit_price):.2f}")


def score_line_items(pred: list[LineItem], gold: list[LineItem]) -> dict:
    """Set-of-line-items F1 with (description-normalized, quantity, unit_price)
    as the identity key. Ignores order."""
    pred_keys = Counter(_line_item_key(li) for li in pred)
    gold_keys = Counter(_line_item_key(li) for li in gold)
    tp = sum((pred_keys & gold_keys).values())
    fp = sum(pred_keys.values()) - tp
    fn = sum(gold_keys.values()) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


@dataclass
class InvoiceScore:
    invoice_id: str
    per_field: dict[str, bool]
    line_item: dict
    n_correct_fields: int
    n_total_fields: int
    field_accuracy: float
    line_item_f1: float


def score_invoice(pred: Invoice, gold: Invoice) -> InvoiceScore:
    per_field = {f: _field_match(f, getattr(pred, f, None), getattr(gold, f, None)) for f in FIELDS}
    li = score_line_items(pred.line_items, gold.line_items)
    n_correct = sum(1 for v in per_field.values() if v)
    return InvoiceScore(
        invoice_id=gold.invoice_id,
        per_field=per_field,
        line_item=li,
        n_correct_fields=n_correct,
        n_total_fields=len(FIELDS),
        field_accuracy=n_correct / len(FIELDS),
        line_item_f1=li["f1"],
    )


def aggregate(scores: list[InvoiceScore]) -> dict:
    """Aggregate per-field accuracy and mean line-item F1 across invoices."""
    n = len(scores)
    if n == 0:
        return {"n": 0}
    per_field_acc = {}
    for f in FIELDS:
        per_field_acc[f] = sum(1 for s in scores if s.per_field[f]) / n
    return {
        "n": n,
        "mean_field_accuracy": sum(s.field_accuracy for s in scores) / n,
        "per_field_accuracy": per_field_acc,
        "mean_line_item_f1": sum(s.line_item_f1 for s in scores) / n,
        "perfect_invoice_rate": sum(1 for s in scores if s.n_correct_fields == s.n_total_fields)
        / n,
    }
