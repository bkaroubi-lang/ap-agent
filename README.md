# ap-agent

[![CI](https://github.com/bkaroubi-lang/ap-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/bkaroubi-lang/ap-agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)

**A Pydantic-AI accounts-payable agent + a statistically grounded eval harness that treats LLM-as-judge agreement as an inter-rater reliability problem.**

The agent itself is standard: extract structured fields from invoices, flag
anomalies against vendor history, answer Q&A. The substance is the evaluation
layer.

## Headline: a finding that failed to replicate — and the infra that caught it

In **July 2026** two Groq judges (`llama-3.3-70b-versatile` and `gpt-oss-20b`)
produced a striking result: inter-judge κ "collapsed" from 0.92 to 0.61 when
subtle epistemic-honesty items were added, and the **larger** model was far
worse on them (llama 20% vs gpt-oss-20b 70%) — it preferred confident
fabrications over an honest "I don't have that data."

In **October 2026** the same gold set was re-run to cache it for offline
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

The repo ships a **record/replay cache**, so the October numbers are
reproducible by anyone with zero API keys and zero cost — the exact thing the
July run lacked when its model disappeared:

```bash
uv sync && uv run python -m eval.harness --replay   # reproduces the report offline
```

Also in the harness: deterministic anomaly detection (invoice-flag F1 = 0.842,
no model dependency), and LLM extraction at per-field / line-item F1 = 1.000 on
returned invoices — *after* an eval-strictness bug that had inflated the error
rate was found and fixed. Full method, both runs, position-bias and calibration
tests, and honest caveats:

**→ [results/report.md](results/report.md)**

**Related:** [credit-bench](https://github.com/bkaroubi-lang/credit-bench) —
GBMs vs FT-Transformer on IEEE-CIS Fraud Detection with DeLong CIs on pairwise
AUC deltas (all three tuned GBMs beat the default FT-Transformer by 6.4–7.1 AUC
points, p ≈ 0).

---

## What the harness covers

The agent is standard Pydantic-AI; the evaluation layer is the substance.

- **LLM-as-judge calibration as an inter-rater reliability problem**: Cohen's
  and Fleiss' kappa on judge–human agreement, bootstrapped CIs on kappa,
  position-bias tests (A/B swap), verbosity-bias correlation, reliability
  curves for judge confidence.
- **Task-level metrics** on the concrete deliverables: per-field accuracy and
  line-item F1 on extraction, precision/recall per anomaly category.
- **Deterministic + LLM hybrid** where appropriate: duplicate detection is
  rules; subtler anomalies use the LLM. The eval treats the hybrid as a single
  system but reports where each branch contributed.
- **Record/replay** of every LLM call, so the billed legs reproduce offline
  with no API key (see below).

## Scope

**Built.** Deterministic seeded synthetic invoices; a schema-first agent with
tools; extraction, anomaly, and judge-calibration evals (Cohen's/Fleiss' kappa
with bootstrapped CIs, position-bias and verbosity tests, reliability curves);
two Groq judges run pairwise; and a record/replay cache so the LLM legs
reproduce offline. The deterministic pieces and `--replay` run with no API key.

**Not done.** A cross-provider judge (e.g. a Claude judge via Anthropic) to test
the "contingent across labs" point; a larger gold set (n=36 → ≥100, balanced
trivial:subtle); a strict paired verbosity-bias test; and a real receipts corpus
(SROIE / RVL-CDIP) in place of synthetic invoices.

## Framework choice

Agent core is **Pydantic AI**. Chosen for:
- Typed I/O via Pydantic v2 — structured inputs and outputs validated at the
  boundary.
- Multi-provider support (Groq / Anthropic / OpenAI / Google) behind a single
  Agent API.
- Small surface area — no state-machine ceremony unless the workflow needs it.

## Setup

**Recommended — [uv](https://docs.astral.sh/uv/)** (Linux/macOS; fetches Python 3.11, no conda needed):

```bash
uv sync
cp .env.example .env      # then put GROQ_API_KEY=gsk_... in .env
uv run pytest -q                        # verify eval math + schema round-trip
uv run python scripts/generate_invoices.py   # deterministic synthetic set
uv run python -m eval.harness --no-llm        # deterministic evals, ~0 cost, no key
uv run python -m eval.harness --replay        # full report from the cached responses, no key
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

The LLM legs are billed; `--no-llm` runs only the offline eval math and needs no
API key, and `--replay` reproduces the full report from the committed cache
(also no key).

## Layout

```
ap-agent/
  pyproject.toml            uv project (deps + lockfile); environment.yml kept as a conda fallback
  .env.example              template for GROQ_API_KEY (default provider)
  configs/defaults.yaml     model IDs, cost caps, judge protocol settings
  scripts/
    generate_invoices.py    deterministic synthetic invoice + label generator
  src/
    schemas.py              Pydantic types: Invoice, LineItem, Anomaly, ...
    store.py                vendor history store (in-memory + JSON snapshot)
    tools.py                build_extraction_agent, flag_anomaly_rules,
                            retrieve_vendor_history, spend_in_window
    agent.py                Pydantic AI agent wiring
  eval/
    extraction.py           per-field accuracy + line-item F1
    anomaly.py              precision/recall/F1 per category
    calibration.py          Cohen's/Fleiss' kappa, bias tests, reliability
    judge.py                LLM-as-judge (single + pairwise with CoT)
    llm_cache.py            record/replay cache for offline reproduction
    harness.py              driver: runs every leg and writes results/
  data/invoices/            gitignored generated invoices + ground_truth.json
  results/                  eval outputs (csv + json) + llm_cache.json
  tests/
```

## Cost discipline

The eval harness has three cost dials in `configs/defaults.yaml`:
- `eval.max_invoices` — cap on how many invoices to process in one run
- `eval.judge.max_items` — cap on how many items the judge scores
- `eval.judge.models` — list of judge models (adding more increases cost linearly)

The default models are Groq-hosted, so a full run (~50 invoices, 2 judges) is
free on Groq's free tier (roughly $1 on paid pricing). The `--no-llm` and
`--replay` paths cost nothing and need no API key — `--replay` reproduces the
full report from the committed cache.
