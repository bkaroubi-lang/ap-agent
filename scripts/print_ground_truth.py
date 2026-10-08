"""Writes data/ground_truth_reference.md — a human-readable cheat sheet
of vendor stats + anomaly labels. Use it alongside qa_gold.json to look
up the right answer for each labeling task.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

DATA = _REPO / "data"
OUT = DATA / "ground_truth_reference.md"


def main() -> None:
    store = json.loads((DATA / "vendor_store.json").read_text())
    gt = json.loads((DATA / "ground_truth.json").read_text())

    by_vendor: dict[str, list[dict]] = defaultdict(list)
    for rec in store:
        by_vendor[rec["vendor_name"]].append(rec)

    anomalies = {
        a["invoice_id"]: a["injected_categories"] for a in gt["anomaly"] if a["injected_categories"]
    }
    inv_by_id = {r["invoice_id"]: r for r in store}

    lines: list[str] = []
    lines.append("# Ground-truth reference for qa_gold.json\n")
    lines.append(
        "Use this to look up the correct answer for each item in "
        "`data/qa_gold.json`. All numbers are derived from "
        "`data/vendor_store.json` (source of truth) and "
        "`data/ground_truth.json` (anomaly labels).\n"
    )

    # ---- Vendors ----
    lines.append("## Vendor totals\n")
    lines.append("| Vendor | # inv | Total spend | Mean | First | Last |")
    lines.append("| --- | ---: | ---: | ---: | --- | --- |")
    for v in sorted(by_vendor):
        recs = sorted(by_vendor[v], key=lambda r: r["invoice_date"])
        totals = [float(r["total"]) for r in recs]
        total_sum = sum(totals)
        mean = total_sum / len(totals)
        lines.append(
            f"| {v} | {len(recs)} | ${total_sum:,.2f} | ${mean:,.2f} | "
            f"{recs[0]['invoice_date']} | {recs[-1]['invoice_date']} |"
        )
    lines.append("")

    # ---- Highest-spend ranking ----
    lines.append("## Vendors ranked by total spend\n")
    ranked = sorted(
        by_vendor.items(),
        key=lambda kv: -sum(float(r["total"]) for r in kv[1]),
    )
    for i, (v, recs) in enumerate(ranked, 1):
        total_sum = sum(float(r["total"]) for r in recs)
        lines.append(f"{i}. **{v}** — ${total_sum:,.2f}")
    lines.append("")

    # ---- Anomalies ----
    lines.append("## Injected anomalies (ground truth)\n")
    lines.append(
        "These invoices were tagged as anomalous in the synthetic data. "
        "Any invoice NOT listed here is a normal invoice — the correct "
        'answer to "was it flagged?" is **no**.\n'
    )
    lines.append("| Invoice | Vendor | Injected category |")
    lines.append("| --- | --- | --- |")
    for inv_id in sorted(anomalies):
        rec = inv_by_id[inv_id]
        cats = ", ".join(anomalies[inv_id])
        lines.append(f"| {inv_id} | {rec['vendor_name']} | {cats} |")
    lines.append("")

    # ---- Individual invoices (compact) ----
    lines.append("## All invoices (per vendor)\n")
    lines.append("Compact per-vendor list. `!` marks an injected anomaly.\n")
    for v in sorted(by_vendor):
        recs = sorted(by_vendor[v], key=lambda r: r["invoice_date"])
        lines.append(f"### {v}\n")
        lines.append("| Invoice | Date | Total | PO | Anomaly |")
        lines.append("| --- | --- | ---: | --- | --- |")
        for r in recs:
            marker = ", ".join(anomalies.get(r["invoice_id"], [])) or ""
            marker_str = f"! {marker}" if marker else ""
            lines.append(
                f"| {r['invoice_id']} | {r['invoice_date']} | "
                f"${float(r['total']):,.2f} | {r.get('po_reference') or '-'} | "
                f"{marker_str} |"
            )
        lines.append("")

    # ---- Meta ----
    earliest = min(r["invoice_date"] for r in store)
    latest = max(r["invoice_date"] for r in store)
    lines.append("## Meta\n")
    lines.append(f"- Total invoices in the vendor store: **{len(store)}**")
    lines.append(f"- Earliest invoice date: **{earliest}**")
    lines.append(f"- Latest invoice date: **{latest}**")
    lines.append(f"- Injected anomalies: **{len(anomalies)}**")
    lines.append(f"- Distinct vendors: **{len(by_vendor)}**")
    lines.append("")
    lines.append("**Anything referring to 2024 or earlier does not exist in this data.**\n")

    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
