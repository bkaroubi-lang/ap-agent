# ap-agent

[![CI](https://github.com/bkaroubi-lang/ap-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/bkaroubi-lang/ap-agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)

**A Pydantic-AI accounts-payable agent + a statistically grounded eval harness that treats LLM-as-judge agreement as an inter-rater reliability problem.**

The agent itself is standard: extract structured fields from invoices, flag
anomalies against vendor history, answer Q&A. What makes this a portfolio piece
is the evaluation layer.

## Headline: a finding that failed to replicate — and the infra that caught it

In **July 2026** two Groq judges (`llama-3.3-70b-versatile` and `gpt-oss-20b`)
produced a striking result: inter-judge κ "collapsed" from 0.92 to 0.61 when I
added subtle epistemic-honesty items, and the **larger** model was far worse on
them (llama 20% vs gpt-oss-20b 70%) — it preferred confident fabrications over
an honest "I don't have that data."

In **October 2026** I re-ran the exact same gold set to cache it for offline
reproduction. Two things happened:

1. **`llama-3.3-70b-versatile` had been retired by Groq** — every call now 404s.
   The original headline is not reproducible: the model is gone.
2. **Re-running with a current same-lab size pair (`gpt-oss-120b` vs
   `gpt-oss-20b`), the finding reversed.** No kappa collapse, and the *larger*
   model was now the *more* honest one on the subtle items.

| Metric | July: llama-70b vs gpt-oss-20b | Oct: gpt-oss-120b vs gpt-oss-20b |
|---|---|---|
| κ (trivial 26 → full 36) | 0.92 → 0.61 (**collapse**) | 0.49 → 0.46 (**no collapse**) |
| Subtle-item accuracy, larger model | **20%** | **80%** |
| Subtle-item accuracy, smaller model | 70% | 50% |
| "Smaller model is more honest" | held | **reversed** |

Same gold set, harness, protocol, `temperature=0` — only the judges changed.
**That is the result:** an LLM-eval headline was contingent on a specific model
pair that vanished within a quarter, and re-running flipped it. What survives
both runs is the durable thesis — *inter-judge agreement is not evidence of
judge quality, and judge quality on hard items is model-specific and must be
measured per model, not inferred from size or a prior run.*

This repo now ships a **record/replay cache**, so the October numbers are
reproducible by anyone with zero API keys and zero cost — the exact thing the
July run lacked when its model disappeared:

```bash
uv sync && uv run python -m eval.harness --replay   # reproduces the report offline
```

Also in the harness: deterministic anomaly detection (invoice-flag F1 = 0.842,
no model dependency), and LLM extraction at per-field / line-item F1 = 1.000 on
returned invoices — *after* I found and fixed an eval-strictness bug that had
inflated the error rate. Full method, both runs, position-bias and calibration
tests, and honest caveats:

**→ [results/report.md](results/report.md)**

**Companion project on tabular ML rigor**:
[credit-bench](https://github.com/bkaroubi-lang/credit-bench) — GBMs vs FT-Transformer on
IEEE-CIS Fraud Detection with DeLong CIs on pairwise AUC deltas. Headline: all
three tuned GBMs beat default FT-Transformer by 6.4–7.1 AUC points (p ≈ 0);
CatBoost vs XGBoost is a statistical tie.

---

## Longer form

The differentiator is not the agent — it's the evaluation layer. Any
Pydantic-AI tutorial gets you an agent-that-runs. What's rare, and what makes
this a portfolio piece rather than a demo:

- **LLM-as-judge calibration as an inter-rater reliability problem**: Cohen's
  and Fleiss' kappa on judge–human agreement, bootstrapped CIs on kappa,
  position-bias tests (A/B swap), verbosity-bias correlation, reliability
  curves for judge confidence.
- **Task-level metrics** on the concrete deliverables: per-field F1 on
  extraction, precision/recall per anomaly category, tool-use correctness on
  the Q&A leg.
- **Deterministic + LLM hybrid** where appropriate: duplicate detection is
  rules; subtler anomalies use the LLM. The eval treats the hybrid as a single
  system but reports where each branch contributed.

## Phased plan

**Phase 1 (this scaffold).** Synthetic invoices (deterministic, seeded),
schema-first agent + tools, extraction/anomaly/judge evals with calibration.
Runs end-to-end without an API key for the deterministic pieces
(extraction eval, anomaly detection rules, kappa/bias/reliability
computations); LLM calls gated behind explicit `--llm` flag.

**Phase 2.** Add DeepEval integration for agentic metrics (task completion,
tool correctness, plan adherence). Add a second judge model for cross-model
kappa. Wire RAGAS for the retrieval leg once vendor history grows large enough
to justify RAG.

**Phase 3 (optional).** Swap synthetic invoices for a public receipts corpus
(SROIE or RVL-CDIP invoice subset) and re-run the same eval harness — proves
the harness generalizes.

## Framework choice

Agent core is **Pydantic AI**. Chosen for:
- Typed I/O via Pydantic v2 (matches an econometrics-adjacent instinct to type
  everything at the boundary).
- Multi-provider support (Anthropic / OpenAI / Google) with a single Agent API.
- Small surface area — no LangGraph state-machine ceremony unless the workflow
  actually needs it.

## Setup

**Recommended — [uv](https://docs.astral.sh/uv/)** (Linux/macOS; fetches Python 3.11, no conda needed):

```bash
uv sync
cp .env.example .env      # then put GROQ_API_KEY=gsk_... in .env
uv run pytest -q                        # verify eval math + schema round-trip
uv run python scripts/generate_invoices.py   # deterministic synthetic set
uv run python -m eval.harness --no-llm        # deterministic evals, ~0 cost, no key
```

**Alternative — original Windows + Miniconda path:**

```powershell
# reuses the Miniconda installed by credit-bench
& "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda env create -f environment.yml
conda activate ap-agent
copy .env.example .env
# edit .env: put GROQ_API_KEY=gsk_... in there (default provider)
python -m pytest tests\ -q                  # verify eval math + schema round-trip
python scripts\generate_invoices.py         # deterministic synthetic set -> data/invoices/
python -m eval.harness --no-llm             # runs deterministic evals only, ~0 cost
python -m eval.harness                      # full run incl. LLM extraction + judge (billed)
```

Every command that will make paid API calls is gated behind an explicit flag
(`--llm` or the default of `harness.py`). `--no-llm` runs only the offline
eval math and requires no API key.

## Layout

```
ap-agent/
  environment.yml           conda env
  .env.example              template for ANTHROPIC_API_KEY
  configs/defaults.yaml     model IDs, cost caps, judge protocol settings
  scripts/
    generate_invoices.py    deterministic synthetic invoice + label generator
  src/
    schemas.py              Pydantic types: Invoice, LineItem, Anomaly, ...
    store.py                vendor history store (in-memory + JSON snapshot)
    tools.py                extract_invoice_fields, flag_anomaly,
                            retrieve_vendor_history
    agent.py                Pydantic AI agent wiring
    runners.py              batch runner over a task set
  eval/
    extraction.py           per-field F1 + error taxonomy
    anomaly.py              precision/recall/F1 per category + confusion matrix
    calibration.py          Cohen's/Fleiss' kappa, bias tests, reliability
    judge.py                LLM-as-judge (single + pairwise with CoT)
    harness.py              driver: runs everything and writes results/report
  data/invoices/            gitignored generated invoices + ground_truth.json
  results/                  eval outputs (csv + json)
  tests/
```

## Cost discipline

The eval harness has three cost dials in `configs/defaults.yaml`:
- `eval.max_invoices` — cap on how many invoices to process in one run
- `eval.judge.max_items` — cap on how many items the judge scores
- `eval.judge.models` — list of judge models (adding more increases cost linearly)

Full Phase-1 run at defaults (~50 invoices, 2 judges): expect **~$3-5** on
Claude Sonnet 5 pricing. The `--no-llm` path costs nothing and still exercises
the extraction eval math, anomaly rule engine, and calibration/kappa/bias
plumbing.
