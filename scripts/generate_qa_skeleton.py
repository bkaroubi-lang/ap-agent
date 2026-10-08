"""Deterministically build a qa_gold.json skeleton from the synthetic data
in data/vendor_store.json + data/ground_truth.json.

For each item, one candidate answer is more correct/complete than the other,
but which candidate holds the correct answer (A or B) is seeded-randomized
so the human labeler cannot form the habit of picking one side.

The human's job: read each (question, candidate_a, candidate_b) triple,
decide which is better, fill in `gold_choice` as either "A" or "B". Save.

Categories emphasize a mix of clear cases (right vs wrong number) and
subtle cases (both approximately correct, one more informative or better
formatted), so that judge accuracy vs your labels captures both trivial
correctness and finer editorial judgment.

Run:
    python scripts/generate_qa_skeleton.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

DATA = _REPO / "data"
OUT = DATA / "qa_gold.json"


def _fmt_money(x: float) -> str:
    return f"${x:,.2f}"


def _load():
    store = json.loads((DATA / "vendor_store.json").read_text())
    gt = json.loads((DATA / "ground_truth.json").read_text())

    by_vendor: dict[str, list[dict]] = defaultdict(list)
    for rec in store:
        by_vendor[rec["vendor_name"]].append(rec)

    stats: dict[str, dict] = {}
    for v, records in by_vendor.items():
        records = sorted(records, key=lambda r: r["invoice_date"])
        totals = [float(r["total"]) for r in records]
        stats[v] = {
            "records": records,
            "count": len(records),
            "sum": sum(totals),
            "mean": sum(totals) / len(totals),
            "min_total": min(totals),
            "max_total": max(totals),
            "first_date": records[0]["invoice_date"],
            "last_date": records[-1]["invoice_date"],
        }

    anomalies = {
        a["invoice_id"]: a["injected_categories"] for a in gt["anomaly"] if a["injected_categories"]
    }
    inv_by_id = {r["invoice_id"]: r for r in store}
    return stats, anomalies, inv_by_id


def _shuffle_ab(rng: random.Random, correct: str, other: str) -> tuple[str, str]:
    if rng.random() < 0.5:
        return correct, other
    return other, correct


def build_items() -> list[dict]:
    stats, anomalies, inv_by_id = _load()
    rng = random.Random(42)
    vendors = list(stats.keys())
    items: list[dict] = []

    # ---- 1. Vendor total spend: 4 items (clear right vs wrong number) ----
    for v in vendors[:4]:
        s = stats[v]
        correct = f"You spent {_fmt_money(s['sum'])} on {v} across {s['count']} invoices."
        wrong_sum = s["sum"] * (1.10 if rng.random() < 0.5 else 0.90)
        wrong = f"You spent {_fmt_money(wrong_sum)} on {v}."
        a, b = _shuffle_ab(rng, correct, wrong)
        items.append(
            {
                "question": f"How much did I spend on {v} in total?",
                "candidate_a": a,
                "candidate_b": b,
                "gold_choice": "",
            }
        )

    # ---- 2. Invoice count: 3 items (small integer, off-by-two) ----
    for v in vendors[4:7]:
        s = stats[v]
        correct = f"You have {s['count']} invoices from {v}."
        wrong = f"You have {s['count'] + 2} invoices from {v}."
        a, b = _shuffle_ab(rng, correct, wrong)
        items.append(
            {
                "question": f"How many invoices do I have from {v}?",
                "candidate_a": a,
                "candidate_b": b,
                "gold_choice": "",
            }
        )

    # ---- 3. Last invoice date: 3 items ----
    for v in vendors[7:10]:
        s = stats[v]
        correct = f"The most recent invoice from {v} was on {s['last_date']}."
        d = date.fromisoformat(s["last_date"])
        wrong_date = d - timedelta(days=45)
        wrong = f"The most recent invoice from {v} was on {wrong_date.isoformat()}."
        a, b = _shuffle_ab(rng, correct, wrong)
        items.append(
            {
                "question": f"When was the most recent invoice from {v}?",
                "candidate_a": a,
                "candidate_b": b,
                "gold_choice": "",
            }
        )

    # ---- 4. Anomaly identification (real anomalies, right category vs wrong): 5 items ----
    cat_expl = {
        "duplicate": "looks like a duplicate — the same amount was billed within a short window",
        "amount_outlier": "is an amount outlier — the total is well outside this vendor's usual range",
        "off_cadence": "is off-cadence — this vendor usually invoices more frequently than this",
        "missing_po": "is missing a PO reference despite this vendor usually including one",
        "unknown_vendor": "is from an unknown vendor with no prior history in your records",
    }
    for inv_id in list(anomalies.keys())[:5]:
        actual = anomalies[inv_id][0]
        rec = inv_by_id[inv_id]
        vendor = rec["vendor_name"]
        correct = f"Yes — invoice {inv_id} {cat_expl[actual]}."
        wrong_cat = rng.choice([c for c in cat_expl if c != actual])
        wrong = f"Yes — invoice {inv_id} {cat_expl[wrong_cat]}."
        a, b = _shuffle_ab(rng, correct, wrong)
        items.append(
            {
                "question": f"Is invoice {inv_id} from {vendor} flagged as anomalous, and if so, why?",
                "candidate_a": a,
                "candidate_b": b,
                "gold_choice": "",
            }
        )

    # ---- 5. Non-anomalous invoices (honest 'no' vs fabricated 'yes'): 3 items ----
    normal_ids = [rid for rid in inv_by_id if rid not in anomalies][:3]
    for inv_id in normal_ids:
        rec = inv_by_id[inv_id]
        vendor = rec["vendor_name"]
        correct = f"No — invoice {inv_id} was not flagged. It matches {vendor}'s usual amount and cadence."
        wrong = f"Yes — invoice {inv_id} was flagged as an amount outlier because it exceeds {vendor}'s typical range."
        a, b = _shuffle_ab(rng, correct, wrong)
        items.append(
            {
                "question": f"Was invoice {inv_id} flagged as anomalous?",
                "candidate_a": a,
                "candidate_b": b,
                "gold_choice": "",
            }
        )

    # ---- 6. Unknown vendor (honest 'no data' vs hallucinated numbers): 3 items ----
    for name in ("Zephyr Solutions", "Meridian Consulting", "Northwind Traders"):
        correct = (
            f"I don't see any invoices from {name} in the vendor store — no history to report."
        )
        wrong_amount = rng.randint(1200, 8500)
        wrong_count = rng.randint(2, 6)
        wrong = f"You spent {_fmt_money(wrong_amount)} on {name} across {wrong_count} invoices in early 2025."
        a, b = _shuffle_ab(rng, correct, wrong)
        items.append(
            {
                "question": f"How much have I spent with {name}?",
                "candidate_a": a,
                "candidate_b": b,
                "gold_choice": "",
            }
        )

    # ---- 7. Comparison (right vendor vs plausibly-wrong vendor): 1 item ----
    sorted_v = sorted(stats.items(), key=lambda kv: -kv[1]["sum"])
    top_v, top_s = sorted_v[0]
    second_v, second_s = sorted_v[1]
    correct = f"{top_v} — total spend was {_fmt_money(top_s['sum'])}, higher than any other vendor."
    wrong = (
        f"{second_v} — total spend was {_fmt_money(second_s['sum'])}, the highest of any vendor."
    )
    a, b = _shuffle_ab(rng, correct, wrong)
    items.append(
        {
            "question": "Which vendor did I spend the most on overall?",
            "candidate_a": a,
            "candidate_b": b,
            "gold_choice": "",
        }
    )

    # ---- 8. Nuanced: complete vs terse (both correct, one more informative): 2 items ----
    for v in vendors[10:12]:
        s = stats[v]
        complete = (
            f"You spent {_fmt_money(s['sum'])} on {v} across {s['count']} invoices between "
            f"{s['first_date']} and {s['last_date']} (avg {_fmt_money(s['mean'])} per invoice)."
        )
        terse = f"{_fmt_money(s['sum'])}."
        a, b = _shuffle_ab(rng, complete, terse)
        items.append(
            {
                "question": f"Summarize my spending on {v}.",
                "candidate_a": a,
                "candidate_b": b,
                "gold_choice": "",
            }
        )

    # ---- 9. Style: grammatical vs telegraphic (both correct info): 1 item ----
    v = vendors[2]
    s = stats[v]
    grammatical = f"You have {s['count']} invoices from {v}, totaling {_fmt_money(s['sum'])}."
    telegraphic = f"{s['count']} invoices {v} total {_fmt_money(s['sum'])}"
    a, b = _shuffle_ab(rng, grammatical, telegraphic)
    items.append(
        {
            "question": f"Give me a quick summary of my {v} invoices.",
            "candidate_a": a,
            "candidate_b": b,
            "gold_choice": "",
        }
    )

    # ---- 10. Out-of-range question honesty: 1 item ----
    earliest = min(s["first_date"] for s in stats.values())
    correct = (
        f"I don't have invoices from before {earliest} in the vendor store, "
        "so I can't report 2024 activity."
    )
    wrong = "Your 2024 vendor spend was approximately $28,000, spread across your usual vendors."
    a, b = _shuffle_ab(rng, correct, wrong)
    items.append(
        {
            "question": "What was my vendor spending like in 2024?",
            "candidate_a": a,
            "candidate_b": b,
            "gold_choice": "",
        }
    )

    return items


def main() -> None:
    items = build_items()
    OUT.write_text(json.dumps(items, indent=2, ensure_ascii=False))
    print(f"Wrote {len(items)} items to {OUT}")
    print()
    print("Sample entries:")
    for item in items[:3]:
        print(f"  Q: {item['question']}")
        print(f"    A: {item['candidate_a'][:110]}")
        print(f"    B: {item['candidate_b'][:110]}")
        print()
    print(f"Next step: open {OUT} in an editor and fill in gold_choice for each item.")
    print('Set gold_choice to exactly "A" or "B" (uppercase, quoted). No other values.')


if __name__ == "__main__":
    main()
