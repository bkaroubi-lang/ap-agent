"""The three agent tools, callable both from the Pydantic AI Agent and directly
from evaluation scripts (so the eval harness can bypass the agentic loop for
deterministic tests of the underlying capabilities).
"""
from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

from src.schemas import (
    Anomaly,
    AnomalyCategory,
    AnomalyReport,
    Invoice,
    VendorHistory,
)
from src.store import VendorStore

if TYPE_CHECKING:
    from pydantic_ai import Agent


# ---------- Tool 1: extract_invoice_fields ----------
# LLM-driven structured extraction. The Agent handles the actual LLM call —
# this module exposes an extraction Agent factory that returns an Invoice.

EXTRACTION_SYSTEM_PROMPT = """\
You are an accounts-payable extraction assistant. Given the raw text of a
single invoice, extract the structured fields.

Rules:
- Dates must be ISO format (YYYY-MM-DD).
- Currency amounts must be decimals, no currency symbols in the numeric fields.
- If a field is not present in the text, do not invent it.
- Extract every line item, even if the invoice has many.
- The `total` field is the final amount due; verify it equals subtotal + tax
  when both are visible.
"""


def build_extraction_agent(model: str, temperature: float = 0.0) -> "Agent[None, Invoice]":
    """Return a Pydantic AI Agent whose output_type is Invoice."""
    from pydantic_ai import Agent

    return Agent(
        model,
        output_type=Invoice,
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        model_settings={"temperature": temperature},
    )


# ---------- Tool 2: flag_anomaly ----------
# Hybrid rule + (optional) LLM. Rules cover the well-defined categories
# (duplicate, amount outlier, off-cadence, missing PO, unknown vendor).
# LLM is available for softer judgments but the rule pass is always run first.


def flag_anomaly_rules(
    invoice: Invoice,
    store: VendorStore,
    duplicate_days_window: int = 30,
    amount_zscore_threshold: float = 3.0,
    off_cadence_multiplier: float = 2.5,
) -> AnomalyReport:
    """Deterministic rule pass. Fast, testable, no LLM cost.

    Returns AnomalyReport whose `is_flagged` is True if any anomaly fires.
    """
    anomalies: list[Anomaly] = []
    hist = store.history(invoice.vendor_name)

    # Unknown vendor
    if hist is None:
        anomalies.append(
            Anomaly(
                category=AnomalyCategory.UNKNOWN_VENDOR,
                severity=0.7,
                explanation=f"No prior invoices from '{invoice.vendor_name}' in the vendor store.",
                evidence={"vendor_name": invoice.vendor_name},
            )
        )
        # Missing PO on unknown vendor is riskier — flag it too.
        if not invoice.po_reference:
            anomalies.append(
                Anomaly(
                    category=AnomalyCategory.MISSING_PO,
                    severity=0.5,
                    explanation="Invoice from unknown vendor without a PO reference.",
                )
            )
        return AnomalyReport.from_anomalies(invoice.invoice_id, anomalies)

    # Duplicate: same vendor + same total + invoice_date within window
    for r in hist.records:
        if r.invoice_id == invoice.invoice_id:
            continue
        if r.total == invoice.total and abs((r.invoice_date - invoice.invoice_date).days) <= duplicate_days_window:
            anomalies.append(
                Anomaly(
                    category=AnomalyCategory.DUPLICATE,
                    severity=0.9,
                    explanation=(
                        f"Potential duplicate of invoice {r.invoice_id} "
                        f"({r.invoice_date.isoformat()}, total {r.total})."
                    ),
                    evidence={"matched_invoice_id": r.invoice_id, "days_apart": abs((r.invoice_date - invoice.invoice_date).days)},
                )
            )
            break  # one duplicate flag is enough

    # Amount outlier: |z| > threshold vs vendor mean/std
    if hist.n_invoices >= 3 and hist.std_total > 0:
        z = float(Decimal(invoice.total) - hist.mean_total) / float(hist.std_total)
        if abs(z) > amount_zscore_threshold:
            anomalies.append(
                Anomaly(
                    category=AnomalyCategory.AMOUNT_OUTLIER,
                    severity=min(1.0, abs(z) / 6.0),
                    explanation=(
                        f"Amount {invoice.total} is {z:+.2f}σ from vendor mean "
                        f"({hist.mean_total} ± {hist.std_total})."
                    ),
                    evidence={"z": z, "mean": str(hist.mean_total), "std": str(hist.std_total)},
                )
            )

    # Off-cadence: last-gap much larger than typical
    if hist.n_invoices >= 3 and hist.mean_days_between > 0:
        gap = (invoice.invoice_date - hist.last_invoice_date).days
        if gap > hist.mean_days_between * off_cadence_multiplier:
            anomalies.append(
                Anomaly(
                    category=AnomalyCategory.OFF_CADENCE,
                    severity=min(1.0, gap / (hist.mean_days_between * 6.0)),
                    explanation=(
                        f"Gap of {gap}d since last invoice, mean is "
                        f"{hist.mean_days_between:.1f}d."
                    ),
                    evidence={"gap_days": gap, "mean_gap_days": hist.mean_days_between},
                )
            )

    # Missing PO on known vendor: only flag if the vendor's history is
    # *strongly* consistent about including one. 0.8 threshold (up from 0.5)
    # collapses false positives from the synthetic ~10% random-missing-PO
    # noise the generator produces on "usually has PO" vendors.
    if not invoice.po_reference:
        prior_with_po = sum(1 for r in hist.records if r.po_reference)
        if prior_with_po >= max(2, int(0.8 * hist.n_invoices)):
            anomalies.append(
                Anomaly(
                    category=AnomalyCategory.MISSING_PO,
                    severity=0.4,
                    explanation=(
                        f"Vendor '{invoice.vendor_name}' usually references a PO "
                        f"({prior_with_po}/{hist.n_invoices} prior invoices), but "
                        "this one does not."
                    ),
                )
            )

    return AnomalyReport.from_anomalies(invoice.invoice_id, anomalies)


# ---------- Tool 3: retrieve_vendor_history ----------

def retrieve_vendor_history(vendor_name: str, store: VendorStore) -> VendorHistory | None:
    return store.history(vendor_name)


def spend_in_window(vendor_name: str, start: date, end: date, store: VendorStore) -> Decimal:
    return store.spend_in_window(vendor_name, start, end)
