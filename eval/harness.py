"""End-to-end eval driver.

Three legs:
  1. Extraction eval — runs the extraction tool over invoice text, scores
     against gold. LLM-gated (requires --llm) since it costs money.
  2. Anomaly eval — runs the deterministic rule pass over the loaded gold
     invoices against the vendor store, scores against injected labels.
     Free (no LLM).
  3. Judge calibration — runs 1-2 judge models on a small hand-labeled set
     with A/B position swaps, computes kappa + bias + reliability.
     LLM-gated.

Run:
    python -m eval.harness --no-llm     # skip Phase 1 & 3 (free)
    python -m eval.harness              # full run (billed)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
from pydantic import TypeAdapter

from eval.anomaly import score_reports
from eval.calibration import (
    bootstrap_kappa_ci,
    cohens_kappa,
    fleiss_kappa,
    kappa_label,
    position_bias_test,
    reliability_curve,
    verbosity_bias_correlation,
)
from eval.extraction import aggregate, score_invoice
from eval.llm_cache import LLMCache
from src.schemas import (
    AnomalyEvalItem,
    ExtractionEvalItem,
    Invoice,
    VendorRecord,
)
from src.store import VendorStore
from src.tools import build_extraction_agent, flag_anomaly_rules

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
GT_PATH = DATA_DIR / "ground_truth.json"
STORE_PATH = DATA_DIR / "vendor_store.json"
CONFIG_PATH = REPO / "configs" / "defaults.yaml"
RESULTS_DIR = REPO / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _load_gt() -> tuple[list[ExtractionEvalItem], list[AnomalyEvalItem]]:
    payload = json.loads(GT_PATH.read_text())
    ext = TypeAdapter(list[ExtractionEvalItem]).validate_python(payload["extraction"])
    ano = TypeAdapter(list[AnomalyEvalItem]).validate_python(payload["anomaly"])
    return ext, ano


def _load_store() -> VendorStore:
    records = TypeAdapter(list[VendorRecord]).validate_json(STORE_PATH.read_text())
    return VendorStore(records=records)


def run_anomaly_eval() -> dict:
    print("Running anomaly rule pass (deterministic, no LLM cost) ...")
    ext, ano = _load_gt()
    store = _load_store()
    invoices = [item.gold for item in ext]
    reports = [flag_anomaly_rules(inv, store) for inv in invoices]
    scored = score_reports(reports, ano)
    (RESULTS_DIR / "anomaly_eval.json").write_text(json.dumps(scored, indent=2))
    print(
        f"  micro F1={scored['micro']['f1']:.3f}  invoice-flag F1={scored['invoice_flag']['f1']:.3f}"
    )
    print(f"  wrote {RESULTS_DIR / 'anomaly_eval.json'}")
    return scored


def run_extraction_eval(cfg: dict, cache: LLMCache) -> dict:
    import time as _time

    replaying = cache.mode == "replay"
    note = "from cache (no API)" if replaying else "LLM calls; may take a few minutes"
    print(f"Running extraction eval ({note}) ...")
    ext, _ = _load_gt()
    ext = ext[: cfg["eval"]["max_invoices"]]

    model = cfg["agent"]["model"]
    agent = build_extraction_agent(model, cfg["agent"]["temperature"])
    delay = 0.0 if replaying else float(cfg["eval"].get("rate_limit_delay_s", 0.0))

    scores = []
    for i, item in enumerate(ext):
        try:
            result = cache.run(agent, item.invoice_text, model=model, output_type=Invoice)
            pred: Invoice = result.output
        except Exception as e:
            print(f"  {item.invoice_id}: extraction failed ({type(e).__name__}: {e!s:.200})")
            if delay > 0 and i + 1 < len(ext):
                _time.sleep(delay)
            continue
        s = score_invoice(pred, item.gold)
        scores.append(s)
        print(f"  {item.invoice_id}: field_acc={s.field_accuracy:.2f} line_f1={s.line_item_f1:.2f}")
        if delay > 0 and i + 1 < len(ext):
            _time.sleep(delay)

    agg = aggregate(scores)
    payload = {"aggregate": agg, "per_invoice": [asdict(s) for s in scores]}
    (RESULTS_DIR / "extraction_eval.json").write_text(json.dumps(payload, indent=2))
    print(f"  mean field accuracy: {agg.get('mean_field_accuracy', 0):.3f}")
    print(f"  mean line-item F1:   {agg.get('mean_line_item_f1', 0):.3f}")
    print(f"  wrote {RESULTS_DIR / 'extraction_eval.json'}")
    return agg


def run_judge_calibration(cfg: dict, cache: LLMCache) -> dict:
    """Skeleton judge run: needs a hand-labeled QA gold set to be a real
    calibration exercise. On Day 1 we surface the plumbing (agreement, kappa,
    position-bias hooks) and warn if the gold set is not populated yet."""
    qa_path = DATA_DIR / "qa_gold.json"
    if not qa_path.exists():
        msg = (
            "  No data/qa_gold.json yet. Skipping judge calibration.\n"
            "  To populate: write ~20-50 items with keys "
            "question / reference_answer / candidate_answers / gold_choice."
        )
        print(msg)
        return {"skipped": True, "reason": "missing qa_gold.json"}

    from eval.judge import (
        JudgePairwiseItem,
        build_pairwise_judge,
        judge_pairwise,
    )

    qa = json.loads(qa_path.read_text())
    n_items = min(len(qa), cfg["eval"]["judge"]["max_items"])
    items_raw = qa[:n_items]

    judge_models = cfg["eval"]["judge"]["models"]
    verdicts_by_judge: dict[str, list[int]] = {m: [] for m in judge_models}
    prefer_first_flags: list[bool] = []
    confidences_by_judge: dict[str, list[float]] = {m: [] for m in judge_models}
    correct_by_judge: dict[str, list[bool]] = {m: [] for m in judge_models}
    lengths: list[int] = []
    scores_flat: list[float] = []

    import time as _time

    delay = 0.0 if cache.mode == "replay" else float(cfg["eval"].get("rate_limit_delay_s", 0.0))

    # Store per-judge results keyed by item index so unequal success rates
    # (some judges failing on some items) don't misalign the kappa arrays.
    verdicts_by_judge_by_item: dict[str, dict[int, int]] = {m: {} for m in judge_models}
    confidences_by_judge_by_item: dict[str, dict[int, float]] = {m: {} for m in judge_models}
    correct_by_judge_by_item: dict[str, dict[int, bool]] = {m: {} for m in judge_models}
    lengths_by_item: dict[int, int] = {}
    scores_flat_stream: list[float] = []
    lengths_stream: list[int] = []

    for m in judge_models:
        agent = build_pairwise_judge(m)
        for i, ex in enumerate(items_raw):
            a, b = ex["candidate_a"], ex["candidate_b"]
            gold_choice = ex["gold_choice"]  # "A" or "B"

            try:
                v = judge_pairwise(
                    agent,
                    JudgePairwiseItem(
                        item_id=f"{i}-ab", question=ex["question"], answer_a=a, answer_b=b
                    ),
                    judge_id=m,
                    cache=cache,
                )
            except Exception as e:
                print(f"  [{m}] item {i}: judge call failed ({type(e).__name__}: {e!s:.200})")
                if delay > 0:
                    _time.sleep(delay)
                continue
            verdicts_by_judge_by_item[m][i] = v.score
            confidences_by_judge_by_item[m][i] = v.confidence
            correct_by_judge_by_item[m][i] = (v.score == 5) == (gold_choice == "A")
            lengths_by_item[i] = len(a) + len(b)
            scores_flat_stream.append(float(v.score))
            lengths_stream.append(len(a) + len(b))
            print(
                f"  [{m}] item {i}: judge_score={v.score}  conf={v.confidence:.2f}  gold={gold_choice}"
            )
            if delay > 0:
                _time.sleep(delay)

            # Swapped order (only for the first configured judge — costs otherwise pile up)
            if m == judge_models[0] and cfg["eval"]["judge"]["swap_ab_for_position_test"]:
                try:
                    v_swap = judge_pairwise(
                        agent,
                        JudgePairwiseItem(
                            item_id=f"{i}-ba", question=ex["question"], answer_a=b, answer_b=a
                        ),
                        judge_id=m,
                        cache=cache,
                    )
                except Exception as e:
                    print(f"  [{m}] item {i} swap: failed ({type(e).__name__}: {e!s:.200})")
                    if delay > 0:
                        _time.sleep(delay)
                    continue
                prefer_first_flags.append((v.score == 5) and (v_swap.score == 1))
                if delay > 0:
                    _time.sleep(delay)

    # Align verdicts across judges on the intersection of items where all
    # judges succeeded — this is what allows kappa comparisons even when
    # some judges misfired on a subset of items.
    common_items = sorted(
        set.intersection(*(set(vs.keys()) for vs in verdicts_by_judge_by_item.values()))
    )
    print(f"\nJudges succeeded on all-common items: {len(common_items)}/{len(items_raw)}")
    for m in judge_models:
        per = verdicts_by_judge_by_item[m]
        print(f"  {m}: {len(per)}/{len(items_raw)} items")

    verdicts_by_judge: dict[str, list[int]] = {
        m: [verdicts_by_judge_by_item[m][i] for i in common_items] for m in judge_models
    }
    confidences_by_judge: dict[str, list[float]] = {
        m: [confidences_by_judge_by_item[m][i] for i in common_items] for m in judge_models
    }
    correct_by_judge: dict[str, list[bool]] = {
        m: [correct_by_judge_by_item[m][i] for i in common_items] for m in judge_models
    }
    lengths = lengths_stream
    scores_flat = scores_flat_stream

    # --- Inter-judge agreement ---
    kappa_out: dict = {}
    can_kappa = len(judge_models) >= 2 and len(common_items) >= 2
    if len(judge_models) >= 2 and not can_kappa:
        print(
            "  WARNING: judges share <2 items in common (a model may be "
            "unavailable); skipping inter-judge kappa."
        )
    if can_kappa:
        m1, m2 = judge_models[0], judge_models[1]
        a1 = np.asarray(verdicts_by_judge[m1])
        a2 = np.asarray(verdicts_by_judge[m2])
        ck = cohens_kappa(a1, a2)
        ck_ci = bootstrap_kappa_ci(
            cohens_kappa,
            a1,
            a2,
            n_boot=cfg["eval"]["bootstrap"]["n_boot"],
            alpha=cfg["eval"]["bootstrap"]["alpha"],
            seed=cfg["eval"]["bootstrap"]["seed"],
        )
        kappa_out["cohens_pair"] = {
            "raters": [m1, m2],
            "kappa": ck,
            "label": kappa_label(ck),
            "ci_lower": ck_ci["ci_lower"],
            "ci_upper": ck_ci["ci_upper"],
        }
    if can_kappa:
        cats = sorted({v for lst in verdicts_by_judge.values() for v in lst})
        cat_idx = {c: i for i, c in enumerate(cats)}
        n_items = len(next(iter(verdicts_by_judge.values())))
        M = np.zeros((n_items, len(cats)), dtype=float)
        for m in judge_models:
            for i, v in enumerate(verdicts_by_judge[m]):
                M[i, cat_idx[v]] += 1
        fk = fleiss_kappa(M)
        kappa_out["fleiss_all"] = {
            "raters": judge_models,
            "kappa": fk,
            "label": kappa_label(fk),
        }

    # --- Position bias (uses only judge_models[0]) ---
    pos_bias = position_bias_test(prefer_first_flags) if prefer_first_flags else None

    # --- Verbosity correlation ---
    verb = verbosity_bias_correlation(scores_flat, lengths)

    # --- Judge reliability (confidence vs correctness on gold) ---
    reliab = {}
    for m in judge_models:
        rel = reliability_curve(
            correct_by_judge[m],
            confidences_by_judge[m],
            n_bins=cfg["eval"]["calibration"]["n_bins"],
        )
        reliab[m] = rel

    payload = {
        "kappa": kappa_out,
        "position_bias": asdict(pos_bias) if pos_bias else None,
        "verbosity_bias": verb,
        "reliability": reliab,
        "raw_verdicts_per_judge": verdicts_by_judge,
    }
    (RESULTS_DIR / "judge_calibration.json").write_text(json.dumps(payload, indent=2))
    print(f"  wrote {RESULTS_DIR / 'judge_calibration.json'}")
    if kappa_out.get("cohens_pair"):
        p = kappa_out["cohens_pair"]
        print(
            f"  Cohen's kappa ({p['raters'][0]} vs {p['raters'][1]}): {p['kappa']:.3f} "
            f"[{p['ci_lower']:.3f}, {p['ci_upper']:.3f}]  ({p['label']})"
        )
    return payload


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--no-llm", action="store_true", help="skip LLM-billed legs")
    parser.add_argument("--only", choices=["anomaly", "extraction", "judge"], default=None)
    parser.add_argument(
        "--record",
        action="store_true",
        help="run the LLM legs live AND save responses to the cache for offline replay",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="reproduce the LLM legs from the cached responses — no API key, no cost",
    )
    parser.add_argument("--cache", type=Path, default=RESULTS_DIR / "llm_cache.json")
    args = parser.parse_args()

    if args.record and args.replay:
        raise SystemExit("--record and --replay are mutually exclusive.")
    cache_mode = "record" if args.record else "replay" if args.replay else "off"
    cache = LLMCache(args.cache, mode=cache_mode)

    cfg = yaml.safe_load(args.config.read_text())

    if not GT_PATH.exists():
        raise SystemExit(f"Missing {GT_PATH}. Run scripts/generate_invoices.py first.")

    if args.only in (None, "anomaly"):
        run_anomaly_eval()

    if args.no_llm:
        print("--no-llm set: skipping LLM-billed legs.")
        return

    # Replay reconstructs the LLM legs from cache, so no API key is needed.
    # Provider-agnostic key check: model strings are "provider:model_id".
    # Map provider prefixes to the env var pydantic-ai expects.
    _PROVIDER_ENV = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "groq": "GROQ_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "google-gla": "GEMINI_API_KEY",
        "google-vertex": "GOOGLE_APPLICATION_CREDENTIALS",
        "cohere": "COHERE_API_KEY",
    }
    if cache.mode != "replay":
        needed_env: set[str] = set()
        for model in [cfg["agent"]["model"], *cfg["eval"]["judge"]["models"]]:
            provider = model.split(":", 1)[0]
            env_var = _PROVIDER_ENV.get(provider)
            if env_var:
                needed_env.add(env_var)
        missing = [v for v in needed_env if not os.getenv(v)]
        if missing:
            raise SystemExit(
                f"Missing API key env var(s): {', '.join(missing)}. "
                "Populate .env or export them, run with --no-llm, or --replay from cache."
            )

    if args.only in (None, "extraction"):
        run_extraction_eval(cfg, cache)

    if args.only in (None, "judge"):
        run_judge_calibration(cfg, cache)

    cache.save()
    if cache.mode == "record":
        print(f"\nRecorded {len(cache)} LLM responses -> {args.cache}")
        print("Reproduce offline with:  python -m eval.harness --replay")


if __name__ == "__main__":
    main()
