from datetime import date
from decimal import Decimal

from eval.extraction import aggregate, score_invoice
from src.schemas import Invoice, LineItem


def _gold_invoice() -> Invoice:
    return Invoice(
        invoice_id="INV-00042",
        vendor_name="Acme Cloud Services",
        invoice_date=date(2025, 3, 1),
        due_date=date(2025, 3, 31),
        po_reference="PO-99999",
        line_items=[
            LineItem(
                description="Widget",
                quantity=2.0,
                unit_price=Decimal("50.00"),
                line_total=Decimal("100.00"),
            ),
            LineItem(
                description="Gadget",
                quantity=1.0,
                unit_price=Decimal("25.00"),
                line_total=Decimal("25.00"),
            ),
        ],
        subtotal=Decimal("125.00"),
        tax=Decimal("10.00"),
        total=Decimal("135.00"),
    )


def test_perfect_extraction():
    gold = _gold_invoice()
    s = score_invoice(gold, gold)
    assert s.field_accuracy == 1.0
    assert s.line_item_f1 == 1.0


def test_wrong_total_hits_field_accuracy():
    gold = _gold_invoice()
    pred = gold.model_copy(update={"total": Decimal("999.99")})
    s = score_invoice(pred, gold)
    assert s.per_field["total"] is False
    assert s.field_accuracy < 1.0


def test_missing_line_item_hits_f1():
    gold = _gold_invoice()
    pred = gold.model_copy(update={"line_items": [gold.line_items[0]]})
    s = score_invoice(pred, gold)
    # Recall = 1/2, Precision = 1/1, F1 = 2/3
    assert abs(s.line_item_f1 - 2 / 3) < 1e-9


def test_case_insensitive_text_match():
    gold = _gold_invoice()
    pred = gold.model_copy(update={"vendor_name": "  ACME cloud services  "})
    s = score_invoice(pred, gold)
    assert s.per_field["vendor_name"] is True


def test_aggregate_over_multiple_invoices():
    gold = _gold_invoice()
    good = score_invoice(gold, gold)
    bad_pred = gold.model_copy(update={"total": Decimal("999.99"), "vendor_name": "Wrong"})
    bad = score_invoice(bad_pred, gold)
    agg = aggregate([good, bad])
    assert agg["n"] == 2
    assert 0.0 < agg["mean_field_accuracy"] < 1.0
    assert agg["per_field_accuracy"]["invoice_id"] == 1.0  # matched in both
