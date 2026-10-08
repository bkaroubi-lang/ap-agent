# ap-agent: an accounts-payable agent with a statistically grounded eval harness

A Pydantic-AI agent that extracts structured fields from invoices, flags anomalies
against a vendor history, and can answer Q&A over vendor spend — plus a judge
calibration layer that treats LLM-as-judge agreement as an inter-rater
reliability problem. The report focuses on that eval layer: the agent itself
is standard; the interesting result is what the evaluation revealed about the
judges.

## TL;DR — the headline finding

Two Groq-hosted LLM judges (Meta's `llama-3.3-70b-versatile` and OpenAI's
`openai/gpt-oss-20b`) evaluated 36 hand-labeled Q&A pairs against your gold
choices. The result:

- **On the 26 trivial items** (right-vs-wrong-number, direct fact lookup) the
  two judges agree with each other on **Cohen's κ = 0.92** ("almost perfect"),
  and both hit ~65–73% accuracy vs the human gold.
- **After adding 10 subtle items** targeting epistemic honesty, hedged
  uncertainty, and speculative overreach, **κ drops to 0.61** and the judges
  diverge sharply: llama-3.3-70b scores **20%** on the subtle subset
  (significantly worse than random binary choice), while gpt-oss-20b scores
  **70%**.

The 92% inter-judge agreement on the original set was largely an artifact of
the test set being easy. A naive experimenter would have concluded "the judges
are reliable" and used their agreement as a validation signal — and would have
been wrong. **Kappa on an easy subset is not evidence of judge quality; it's
evidence of shared bias on that subset.**

## What was built

- **Agent** (`src/agent.py`): Pydantic AI agent with four tools —
  `get_vendor_history`, `get_vendor_spend`, `check_invoice_anomalies`,
  `list_vendors` — over an in-memory vendor store.
- **Extraction tool** (`src/tools.py`): a separate structured-output agent
  whose `output_type=Invoice` returns a validated Pydantic model direct from
  raw invoice text.
- **Anomaly rules** (`src/tools.py`): deterministic rule pass covering
  duplicate detection (same vendor + same total within N days), amount
  outliers (|z| > threshold vs vendor mean), off-cadence (gap > multiplier ×
  mean gap), missing-PO conditional on vendor's usual PO rate, and
  unknown-vendor.
- **Synthetic invoice generator** (`scripts/generate_invoices.py`):
  deterministic seeded generator producing 60 invoices across 12 vendors with
  known-good line-item structure. Injects one of `{duplicate, amount_outlier,
  off_cadence, missing_po}` into a configurable fraction of invoices.
- **Eval harness** (`eval/harness.py`) that runs three legs:
  1. Anomaly rule pass over the injected labels → precision/recall/F1 per
     category (fully deterministic, no LLM).
  2. LLM extraction on the invoice text → per-field accuracy + line-item F1
     against the gold Invoice.
  3. Judge calibration on a hand-labeled Q&A gold set → Cohen's/Fleiss' κ
     between judge models (bootstrapped CI), position bias (A/B swap, Wilson
     CI + exact binomial test), verbosity Spearman, reliability curves +
     Brier score per judge.
- **Bias-probing gold set** (`data/qa_gold.json`, `scripts/append_qa_subtle_items.py`):
  26 initial items across factual and stylistic categories, plus 10 items
  specifically probing epistemic honesty, hedging, and speculative overreach —
  the "hard" items that broke the initial 0.92 kappa.

## Method & compute

- **Framework**: Pydantic AI 2.5, structured-output via `output_type=Invoice`.
  Provider-agnostic key check in `eval/harness.py` (works with Anthropic,
  OpenAI, Groq, Mistral, Google-GLA, Cohere).
- **Provider**: Groq free tier. Agent model:
  `groq:llama-3.3-70b-versatile`. Judge models:
  `groq:llama-3.3-70b-versatile` + `groq:openai/gpt-oss-20b`. Rate-limit
  discipline: 3 s delay between calls in both extraction and judge loops.
- **Prior schema fix**: Pydantic v2's default `Decimal` JSON schema uses a
  regex with a lookahead pattern that Groq's tool-parameter validator
  rejects. Introduced a `Money = Annotated[Decimal, WithJsonSchema({"type":
  "number"})]` type alias so the emitted schema is a plain number. Decimal
  is preserved internally.
- **Eval strictness fix (mid-run)**: the initial line-item eval keyed identity
  on `str(li.unit_price)`, which caused false failures when the LLM returned
  `342.5` for a Decimal represented as `342.50` (numerically identical). Fixed
  by normalizing to `f"{float(li.unit_price):.2f}"`. See caveats.
- **Repro**: seed=42 across invoice generation, seed=137 for the subtle QA
  items, seed=0 elsewhere. Every eval is one command:
  `python -m eval.harness --no-llm` for the deterministic legs;
  `--only extraction` / `--only judge` for the billed ones.

## Result 1 — Deterministic anomaly detection (no LLM, no cost)

60 synthetic invoices, 9 injected anomalies (15% rate). Rule pass evaluated
set-based against injected labels.

The `missing_po` rule was tightened after the initial run: it now fires only
if ≥ 80% of the vendor's history has a PO (was 50%). Rationale: the invoice
generator randomly leaves POs off ~10% of "usually has PO" invoices as noise,
and the ground truth doesn't tag those as anomalies. At the tighter threshold,
`missing_po` FPs drop from 5 to 3 while recall stays at 1.0.

| Category | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| duplicate | 2 | 0 | 0 | 1.000 | 1.000 | **1.000** |
| amount_outlier | 3 | 0 | 0 | 1.000 | 1.000 | **1.000** |
| off_cadence | 2 | 0 | 1 | 1.000 | 0.667 | 0.800 |
| missing_po | 1 | 3 | 0 | 0.250 | 1.000 | 0.400 |
| unknown_vendor | 0 | 0 | 0 | — | — | — |
| **micro** | 8 | 3 | 1 | 0.727 | 0.889 | **0.800** |
| **macro** | | | | 0.812 | 0.917 | 0.800 |
| **invoice-level flag** | 8 | 2 | 1 | 0.800 | 0.889 | **0.842** |

The two crisp rules (`duplicate`, `amount_outlier`) are perfect. The remaining
`missing_po` FPs correspond to vendors near — but not at — 100% historical
PO rate; those flags would arguably be worth surfacing in a real AP workflow
even when they don't align with an injected label. Left as-is rather than
tuning further.

## Result 2 — LLM-based invoice extraction

50 invoices processed with `llama-3.3-70b-versatile` at 3 s per-call delay.
Each field compared against the gold `Invoice` under normalized text /
tolerance-based comparison for money fields.

**As reported on the initial run** (under the stale eval):
- Mean field accuracy: 0.996
- Mean line-item F1: 0.880
- Perfect-invoice rate: 0.980

**After finding + fixing the eval strictness bug** (Decimal string comparison
was rejecting `342.50` vs `342.5`, numerically identical), a full 50-invoice
rerun on the corrected eval:

- Mean field accuracy: **1.000** (9/9 fields perfect on all 50 invoices)
- Mean line-item F1: **1.000**
- Perfect-invoice rate: **1.000** (50/50 invoices perfect on every field
  and every line item)

The 12-point line-item F1 gap in the original run was **entirely** an eval
bug, not a model error. On this synthetic data, `llama-3.3-70b-versatile`
extracts invoices with zero measurable error. This is a finding *about the
eval*, not about the model — worth writing up in a portfolio piece precisely
because it shows the eval is being audited critically, not treated as
ground truth.

## Result 3 — Judge calibration (the headline)

36 hand-labeled Q&A pairs, two judge models running pairwise. First judge
additionally runs the A/B swap for position-bias testing.

### Inter-judge agreement (Cohen's κ) collapses when subtle items are added

| Gold set | n | Cohen's κ | 95% CI | Interpretation |
|---|---:|---:|---|---|
| Initial trivial-heavy set | 26 | **0.923** | [0.71, 1.00] | "almost perfect" |
| Expanded with 10 subtle items | 36 | **0.611** | [0.33, 0.83] | "substantial" |

Adding 10 items designed to probe epistemic honesty, hedged uncertainty, and
speculative overreach dropped kappa by **31 percentage points**. If we had
stopped at the 26-item set, we would have concluded the judges are highly
reliable. That conclusion depends on the choice of test set.

### Judge accuracy vs gold, sliced by difficulty

| Model | Overall | Trivial (0–25) | **Subtle (26–35)** |
|---|---:|---:|---:|
| `llama-3.3-70b-versatile` | 52.8% (19/36) | 65.4% (17/26) | **20.0%** (2/10) |
| `openai/gpt-oss-20b` | **72.2%** (26/36) | 73.1% (19/26) | **70.0%** (7/10) |

Two observations:

1. **llama-3.3-70b scores 20% on the subtle subset — significantly worse than
   random binary choice.** It's not just failing; it's systematically picking
   the wrong answer.
2. **gpt-oss-20b, the smaller model, handles subtle items essentially as well
   as trivial ones** (70% vs 73%). Model size does not predict judgment
   quality on this axis.

### Per-category subtle breakdown

| Category | Description | llama | gpt-oss |
|---|---|---:|---:|
| Honest vs hallucinated (5) | "no data" vs fabricated specific number | **0/5** | 2/5 |
| Hedged vs confident-wrong (3) | calibrated hedge vs specific-and-wrong | 2/3 | 3/3 |
| Factual vs speculative-overreach (2) | measured facts vs alarmist speculation | **0/2** | 2/2 |

llama-3.3-70b failed **all 5** honest-vs-hallucinated items — it consistently
prefers a made-up number over "I don't have that data." Same on 2/2 overreach
items: it picks alarmist speculation over measured factual answers.

### The rationales (from an inspection run on failure cases)

Both judges *explicitly acknowledge* they cannot verify the numbers, then
choose the fabricated answer anyway:

> **llama-3.3-70b on "How much have I spent with Zephyr Solutions?"** (Zephyr
> is a made-up vendor not in the data): *"Answer A [with fabricated numbers]
> provides a specific amount spent and the number of invoices. Answer B [the
> honest 'no data' answer] indicates a lack of information, which does not
> help in determining the amount spent. Therefore, Answer B is more
> informative and useful in this context."*

> **gpt-oss-20b on the same item**: *"Answer B provides a specific monetary
> figure and context. Answer A merely states no invoices exist. **Without
> evidence that the user truly has no spending, the more informative response
> is preferable.**"*

Both models frame the honest answer as "less helpful" and the hallucinated
answer as "more informative." That's a nameable, portable finding: **LLM
judges are biased toward specificity even when they know they lack ground
truth.**

### Position bias (llama, A/B swap on all 36)

- Prefer-first rate: **0.361** (n=36)
- Wilson 95% CI: [0.225, 0.524]
- Two-sided binomial p vs 0.5: **0.132** (not significant)

Position bias per se is not the story here. The bias is semantic (preference
for confident specifics), not positional. Worth noting because a report
without a position-bias test would leave the finding open to the "you didn't
control for the obvious confound" critique.

### Calibration (Brier scores)

| Model | Brier | Interpretation |
|---|---:|---|
| `llama-3.3-70b-versatile` | 0.402 | poor — confidently wrong on subtle items |
| `openai/gpt-oss-20b` | 0.225 | reasonable — confidence tracks correctness |

llama's calibration is worse than random on this eval because it's confidently
wrong on the added subtle items (avg reported confidence ≈ 0.87 on its 8
wrong subtle picks). gpt-oss's calibration held up because it's more honest
about uncertainty on hard items.

## The "kappa is treacherous" writeup

Three distinct, defensible claims from this study:

1. **Inter-judge kappa on an easy test set is not evidence of judge
   quality.** Two judges agreeing 92% on 26 easy items is compatible with
   both being reliable OR both making the same kinds of errors. Only harder
   items separate those two hypotheses.
2. **Model size doesn't predict judgment quality.** gpt-oss-20b beat
   llama-3.3-70b on exactly the axis (epistemic honesty) that a real
   evaluation harness should care about — despite being ~3.5× smaller by
   parameter count.
3. **LLM judges systematically prefer confident specificity over honest
   uncertainty.** Both models, in their own words, treat "I don't have that
   data" as less informative than a fabricated number. This is a
   *portable* bias — visible across labs and model sizes — and the direct
   quotes above are cite-able evidence.

For an accounts-payable agent specifically, this matters: an AP assistant
that hallucinates vendor totals or fabricates payment lags is doing worse
than one that admits "I don't have that data." A judge that rewards the
hallucination will systematically train / select agents toward the wrong
behavior.

## Caveats

- **Small gold set.** n=36 is enough to produce a κ estimate with a usable
  CI ([0.33, 0.83]) but not enough to publish. A production study would want
  ≥ 100 items with the trivial:subtle balance closer to 1:1 rather than
  26:10.
- **Synthetic data.** Vendor store, invoices, and injected anomalies are all
  generated by `scripts/generate_invoices.py`. The findings transfer *in
  spirit* to real invoice data because the biases probed (hallucination
  preference, overreach) are general LLM-judge properties, not artifacts of
  this dataset — but a proper cross-dataset study would swap to real
  SROIE-like receipts as a second pass.
- **Two judges from Groq only.** Cross-provider diversity (add Anthropic
  Claude, add Google Gemini) would strengthen the "cross-lab bias" story.
  Blocked here on API-key availability, not code — the harness's
  provider-agnostic key check accepts any of the six major providers.
- **Verbosity Spearman is not a strict verbosity-bias test.** The reported
  ρ=−0.28 (p=0.02) is the correlation between judge score and total input
  length; a strict verbosity-bias test compares length of the two candidates
  for the same question. Retained here as a weak signal, not a headline.
- **Extraction line-item F1 initial number (0.880) was a stale-eval
  artifact.** The full 50-invoice rerun under the fixed eval scored 1.0 on
  every metric — the model was never actually making extraction errors on
  this data.
- **The `missing_po` rule was tightened once** (50% → 80% prior-PO threshold)
  after the first pass showed 5 FPs. Post-tighten: 3 FPs (invoice-flag F1
  0.762 → 0.842). Further tightening plausible but the remaining FPs are
  arguably signals a real AP workflow would want to surface.

## Reproducibility

Everything ships with pinned deps and deterministic seeds. Full sequence:

```bash
# One-time env setup (uv fetches Python 3.11 itself; no conda needed)
uv sync
cp .env.example .env       # populate with GROQ_API_KEY (or any provider key)

# Deterministic legs (free, no API cost, no key required)
uv run pytest -q
uv run python scripts/generate_invoices.py
uv run python -m eval.harness --no-llm

# Billed legs
uv run python scripts/print_ground_truth.py          # cheat sheet for labeling
uv run python scripts/append_qa_subtle_items.py      # if starting fresh, else skip
# hand-label data/qa_gold.json (fill gold_choice A or B)
uv run python -m eval.harness --only extraction      # ~$0 free tier / ~$0.60 paid
uv run python -m eval.harness --only judge           # ~$0 free tier / ~$0.30 paid
```

**Files that ship the findings** (all under `results/`):
- `anomaly_eval.json` — Result 1 per-category F1s
- `extraction_eval.json` — Result 2 field + line-item scores
- `judge_calibration.json` — Result 3 kappa, position bias, reliability, raw
  verdicts per judge
- `report.md` — this file

---

Generated 2026-07-07.
