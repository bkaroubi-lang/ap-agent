"""Deterministic synthetic invoice generator.

For each vendor, generates a stream of clean invoices at a vendor-specific
cadence and amount distribution, then injects anomalies at a configurable rate.
Emits:
  - data/invoices/*.txt   : raw invoice text (what the extraction tool sees)
  - data/ground_truth.json: gold-labeled records with structured fields and
                            injected-anomaly categories per invoice
  - data/vendor_store.json: pre-anomaly history for the vendor store

Run:
    python scripts/generate_invoices.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

# Repo-root on sys.path so `from src.schemas import ...` works when this file
# is invoked directly (not via `python -m`).
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import yaml
from pydantic import TypeAdapter

from src.schemas import (
    AnomalyCategory,
    AnomalyEvalItem,
    ExtractionEvalItem,
    Invoice,
    LineItem,
    VendorRecord,
)

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
INVOICE_DIR = DATA_DIR / "invoices"
GT_PATH = DATA_DIR / "ground_truth.json"
STORE_PATH = DATA_DIR / "vendor_store.json"
CONFIG_PATH = REPO / "configs" / "defaults.yaml"


VENDOR_CATALOG = [
    ("Acme Cloud Services", "SaaS subscription", 800, 150, 30),
    ("Blue Ridge Logistics", "Freight and delivery", 1200, 400, 14),
    ("Corner Coffee Co.", "Office refreshments", 180, 30, 7),
    ("Delta Print Shop", "Marketing materials", 350, 90, 21),
    ("Eastwind Legal LLP", "Legal services", 4500, 1200, 30),
    ("Foxtrot Facilities", "Cleaning services", 950, 100, 14),
    ("Grove Web Studio", "Website maintenance", 2200, 500, 30),
    ("Harbor Insurance Brokers", "Business insurance", 1800, 200, 30),
    ("Ivy Analytics", "Data consulting", 3200, 800, 30),
    ("Juniper Office Supplies", "Office supplies", 220, 60, 14),
    ("Kestrel HR Partners", "HR consulting", 2400, 400, 30),
    ("Larkspur Design", "Design services", 1600, 500, 21),
]


def _fmt_money(x: Decimal) -> str:
    return f"{x:,.2f}"


def _render_invoice_text(inv: Invoice) -> str:
    lines: list[str] = []
    lines.append(f"INVOICE {inv.invoice_id}")
    lines.append(f"From: {inv.vendor_name}")
    lines.append(f"Invoice date: {inv.invoice_date.isoformat()}")
    if inv.due_date:
        lines.append(f"Due date: {inv.due_date.isoformat()}")
    if inv.po_reference:
        lines.append(f"PO: {inv.po_reference}")
    lines.append("")
    lines.append("Line items:")
    for it in inv.line_items:
        lines.append(
            f"  {it.description} x{it.quantity:g} @ {_fmt_money(it.unit_price)} "
            f"= {_fmt_money(it.line_total)}"
        )
    lines.append("")
    lines.append(f"Subtotal: {_fmt_money(inv.subtotal)}")
    lines.append(f"Tax:      {_fmt_money(inv.tax)}")
    lines.append(f"TOTAL:    {_fmt_money(inv.total)} {inv.currency}")
    return "\n".join(lines)


def _make_clean_invoice(
    rng: random.Random,
    invoice_seq: int,
    vendor_name: str,
    line_desc: str,
    mean_total: float,
    std_total: float,
    invoice_date: date,
    include_po: bool,
) -> Invoice:
    total = max(50.0, rng.gauss(mean_total, std_total))
    n_items = rng.randint(1, 4)
    per_item = total / n_items
    line_items: list[LineItem] = []
    running = Decimal("0")
    for i in range(n_items):
        qty = float(rng.randint(1, 4))
        unit = Decimal(f"{per_item / qty:.2f}")
        line_total = (Decimal(qty) * unit).quantize(Decimal("0.01"))
        running += line_total
        line_items.append(
            LineItem(
                description=f"{line_desc} #{invoice_seq}-{i+1}",
                quantity=qty,
                unit_price=unit,
                line_total=line_total,
            )
        )
    subtotal = running
    tax = (subtotal * Decimal("0.08")).quantize(Decimal("0.01"))
    total_dec = (subtotal + tax).quantize(Decimal("0.01"))
    return Invoice(
        invoice_id=f"INV-{invoice_seq:05d}",
        vendor_name=vendor_name,
        invoice_date=invoice_date,
        due_date=invoice_date + timedelta(days=30),
        po_reference=f"PO-{rng.randint(10000, 99999)}" if include_po else None,
        line_items=line_items,
        subtotal=subtotal,
        tax=tax,
        total=total_dec,
    )


def generate(cfg: dict) -> None:
    gen = cfg["generation"]
    rng = random.Random(gen["seed"])
    n_vendors = gen["n_vendors"]
    n_per = gen["n_invoices_per_vendor"]
    anomaly_rate = gen["anomaly_rate"]

    vendors = VENDOR_CATALOG[:n_vendors]
    start = date(2025, 1, 1)

    all_invoices: list[Invoice] = []
    vendor_store_records: list[VendorRecord] = []
    injected: dict[str, list[AnomalyCategory]] = {}
    seq = 0

    for vendor_name, line_desc, mean_total, std_total, cadence_days in vendors:
        # 60% of vendors typically include a PO; the other 40% typically don't.
        vendor_usually_has_po = rng.random() < 0.6
        cur_date = start + timedelta(days=rng.randint(0, cadence_days))
        for i in range(n_per):
            seq += 1
            include_po = vendor_usually_has_po and rng.random() > 0.1
            inv = _make_clean_invoice(
                rng, seq, vendor_name, line_desc, mean_total, std_total, cur_date, include_po
            )
            all_invoices.append(inv)
            vendor_store_records.append(
                VendorRecord(
                    vendor_name=inv.vendor_name,
                    invoice_id=inv.invoice_id,
                    invoice_date=inv.invoice_date,
                    total=inv.total,
                    po_reference=inv.po_reference,
                )
            )
            cur_date += timedelta(days=int(rng.gauss(cadence_days, cadence_days / 5)))

    # Inject anomalies into a copy of the invoices (not the store — store
    # represents historical truth against which anomalies get flagged).
    inv_by_id = {inv.invoice_id: inv for inv in all_invoices}
    anomaly_targets = rng.sample(
        list(inv_by_id.keys()), k=int(len(inv_by_id) * anomaly_rate)
    )
    for inv_id in anomaly_targets:
        inv = inv_by_id[inv_id]
        choice = rng.choice(
            [
                AnomalyCategory.DUPLICATE,
                AnomalyCategory.AMOUNT_OUTLIER,
                AnomalyCategory.OFF_CADENCE,
                AnomalyCategory.MISSING_PO,
            ]
        )
        if choice is AnomalyCategory.DUPLICATE:
            # Copy total from an earlier invoice of the same vendor, if any.
            peers = [r for r in vendor_store_records if r.vendor_name == inv.vendor_name and r.invoice_id != inv.invoice_id]
            if peers:
                twin = rng.choice(peers)
                new_total = twin.total
                # Rebuild subtotal/tax proportionally to keep it well-formed.
                new_subtotal = (new_total / Decimal("1.08")).quantize(Decimal("0.01"))
                new_tax = (new_total - new_subtotal).quantize(Decimal("0.01"))
                inv.subtotal = new_subtotal
                inv.tax = new_tax
                inv.total = new_total
                # nudge date close to the twin
                inv.invoice_date = twin.invoice_date + timedelta(days=rng.randint(1, 10))
        elif choice is AnomalyCategory.AMOUNT_OUTLIER:
            multiplier = rng.choice([4.0, 5.0, 6.0])
            new_total = (inv.total * Decimal(str(multiplier))).quantize(Decimal("0.01"))
            new_subtotal = (new_total / Decimal("1.08")).quantize(Decimal("0.01"))
            new_tax = (new_total - new_subtotal).quantize(Decimal("0.01"))
            inv.subtotal = new_subtotal
            inv.tax = new_tax
            inv.total = new_total
        elif choice is AnomalyCategory.OFF_CADENCE:
            inv.invoice_date = inv.invoice_date + timedelta(days=rng.randint(120, 240))
            inv.due_date = inv.invoice_date + timedelta(days=30)
        elif choice is AnomalyCategory.MISSING_PO:
            inv.po_reference = None

        injected[inv_id] = [choice]

    INVOICE_DIR.mkdir(parents=True, exist_ok=True)
    for inv in all_invoices:
        (INVOICE_DIR / f"{inv.invoice_id}.txt").write_text(_render_invoice_text(inv))

    # Ground truth: extraction gold + injected anomaly labels
    extraction_gt = [
        ExtractionEvalItem(
            invoice_id=inv.invoice_id,
            invoice_text=_render_invoice_text(inv),
            gold=inv,
        )
        for inv in all_invoices
    ]
    anomaly_gt = [
        AnomalyEvalItem(
            invoice_id=inv.invoice_id,
            injected_categories=injected.get(inv.invoice_id, []),
        )
        for inv in all_invoices
    ]

    gt_payload = {
        "extraction": TypeAdapter(list[ExtractionEvalItem]).dump_python(extraction_gt, mode="json"),
        "anomaly": TypeAdapter(list[AnomalyEvalItem]).dump_python(anomaly_gt, mode="json"),
    }
    GT_PATH.write_text(json.dumps(gt_payload, indent=2, default=str))

    STORE_PATH.write_text(
        TypeAdapter(list[VendorRecord]).dump_json(vendor_store_records, indent=2).decode()
    )

    print(f"Wrote {len(all_invoices)} invoices to {INVOICE_DIR}")
    print(f"Injected anomalies: {sum(len(v) for v in injected.values())} across {len(injected)} invoices")
    print(f"Ground truth -> {GT_PATH}")
    print(f"Vendor store -> {STORE_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    generate(cfg)


if __name__ == "__main__":
    main()
