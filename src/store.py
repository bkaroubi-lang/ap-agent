"""In-memory vendor history + JSON snapshot.

The store is deliberately small and eager (all in memory) — realistic for
an SMB with ~thousands of invoices, and keeps the agent's retrieval tool a
pure function of the store snapshot for reproducibility.
"""
from __future__ import annotations

import json
import statistics
from datetime import date
from decimal import Decimal
from difflib import SequenceMatcher
from pathlib import Path

from pydantic import TypeAdapter

from src.schemas import VendorHistory, VendorRecord

_RECORD_LIST_ADAPTER = TypeAdapter(list[VendorRecord])


class VendorStore:
    def __init__(self, records: list[VendorRecord] | None = None):
        self.records: list[VendorRecord] = records or []

    # ------- persistence -------
    @classmethod
    def from_json(cls, path: Path) -> "VendorStore":
        records = _RECORD_LIST_ADAPTER.validate_json(path.read_text())
        return cls(records=records)

    def to_json(self, path: Path) -> None:
        path.write_text(_RECORD_LIST_ADAPTER.dump_json(self.records, indent=2).decode())

    def add(self, record: VendorRecord) -> None:
        self.records.append(record)

    # ------- queries -------
    def _for_vendor(self, vendor_name: str, fuzzy_cutoff: float = 0.8) -> list[VendorRecord]:
        target = vendor_name.lower().strip()
        exact = [r for r in self.records if r.vendor_name.lower().strip() == target]
        if exact:
            return sorted(exact, key=lambda r: r.invoice_date)
        # Fuzzy fallback for OCR/typo tolerance.
        scored = [
            (SequenceMatcher(None, target, r.vendor_name.lower()).ratio(), r)
            for r in self.records
        ]
        matched = [r for score, r in scored if score >= fuzzy_cutoff]
        return sorted(matched, key=lambda r: r.invoice_date)

    def history(self, vendor_name: str) -> VendorHistory | None:
        recs = self._for_vendor(vendor_name)
        if not recs:
            return None
        totals = [float(r.total) for r in recs]
        mean_total = Decimal(f"{statistics.fmean(totals):.2f}")
        std_total = Decimal(f"{statistics.pstdev(totals):.2f}") if len(totals) > 1 else Decimal("0")
        # Days between consecutive invoices; 0 for single-record vendors.
        gaps = [
            (recs[i].invoice_date - recs[i - 1].invoice_date).days
            for i in range(1, len(recs))
        ]
        mean_gap = float(statistics.fmean(gaps)) if gaps else 0.0
        return VendorHistory(
            vendor_name=recs[0].vendor_name,
            n_invoices=len(recs),
            mean_total=mean_total,
            std_total=std_total,
            mean_days_between=mean_gap,
            last_invoice_date=recs[-1].invoice_date,
            records=recs,
        )

    def all_vendors(self) -> list[str]:
        return sorted({r.vendor_name for r in self.records})

    def spend_in_window(self, vendor_name: str, start: date, end: date) -> Decimal:
        recs = self._for_vendor(vendor_name)
        return sum(
            (r.total for r in recs if start <= r.invoice_date <= end),
            start=Decimal("0"),
        )
