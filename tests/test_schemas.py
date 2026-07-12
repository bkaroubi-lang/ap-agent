from datetime import date
from decimal import Decimal

from src.schemas import Invoice, LineItem


def test_invoice_roundtrip_json():
    inv = Invoice(
        invoice_id="INV-00001",
        vendor_name="Acme Cloud Services",
        invoice_date=date(2025, 1, 15),
        due_date=date(2025, 2, 14),
        po_reference="PO-12345",
        line_items=[
            LineItem(
                description="Widget",
                quantity=2.0,
                unit_price=Decimal("50.00"),
                line_total=Decimal("100.00"),
            )
        ],
        subtotal=Decimal("100.00"),
        tax=Decimal("8.00"),
        total=Decimal("108.00"),
        currency="usd",
    )
    assert inv.currency == "USD"
    js = inv.model_dump_json()
    inv2 = Invoice.model_validate_json(js)
    assert inv2.total == Decimal("108.00")
    assert inv2.line_items[0].line_total == Decimal("100.00")


def test_invoice_rejects_extras():
    try:
        Invoice.model_validate(
            {
                "invoice_id": "INV-1",
                "vendor_name": "X",
                "invoice_date": "2025-01-01",
                "subtotal": "10.00",
                "tax": "0",
                "total": "10.00",
                "bogus_field": "should fail",
            }
        )
    except Exception:
        return
    raise AssertionError("Extra field should have raised a ValidationError.")
