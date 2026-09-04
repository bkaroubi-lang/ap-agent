# ap-agent

**A Pydantic-AI accounts-payable agent + a statistically grounded eval harness that treats LLM-as-judge agreement as an inter-rater reliability problem.**

The agent itself is standard: extract structured fields from invoices, flag
anomalies against vendor history, answer Q&A. What makes this a portfolio piece
is the evaluation layer.

## Headline finding: the kappa collapse

Two Groq-hosted LLM judges (Meta `llama-3.3-70b-versatile` and OpenAI
`gpt-oss-20b`) scored the same 26 hand-labeled Q&A pairs.

| Test set | n | Cohen's κ | 95% CI |
|---|---:|---:|---|
| Trivial (right-vs-wrong-number) | 26 | **0.923** | [0.71, 1.00] |
| Adds 10 subtle items (epistemic honesty, hedging, overreach) | 36 | **0.611** | [0.33, 0.83] |

Adding 10 items probing whether the judge prefers a confident hallucination
over an honest "I don't have that data" **dropped kappa by 31 percentage
points**, and revealed:

- **llama-3.3-70b: 20% accuracy** on the added subset. Chose the fabricated
  answer over "no data" on 5/5 unknown-vendor questions.
- **gpt-oss-20b: 70% accuracy**. The smaller model was substantially better on
  the exact axis that matters for real evaluation.

The judges themselves articulate the bias. Direct quote from llama-3.3-70b:
*"Answer B [the honest 'no data' answer] indicates a lack of information,
which does not help… therefore Answer A [with fabricated numbers] is more
informative and useful."*

**Three defensible portfolio claims from n=36**:

1. Inter-judge κ on an easy subset is not evidence of judge quality — it's
   compatible with both judges being reliable *or* both sharing errors.
2. Model size doesn't predict judgment quality on this axis.
3. LLM judges systematically prefer confident specificity over honest
   uncertainty — a portable cross-lab bias.

Also in the harness: deterministic rule-based anomaly detection (invoice-flag
F1 = 0.842), and LLM-based extraction — 50 invoices with per-field accuracy =
1.000 and line-item F1 = 1.000 (perfect on all 50), *after* I found and fixed
an eval-strictness bug that had inflated the reported error rate. Full method,
tables, direct-quote rationales, position-bias tests, calibration, and honest
caveats:

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
cp .env.example .env      # then put ANTHROPIC_API_KEY=sk-... in .env
uv run pytest -q          # verify eval math + schema round-trip
```

**Alternative — original Windows + Miniconda path:**

```powershell
# reuses the Miniconda installed by credit-bench
& "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda env create -f environment.yml
conda activate ap-agent
copy .env.example .env
# edit .env: put ANTHROPIC_API_KEY=sk-... in there
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
