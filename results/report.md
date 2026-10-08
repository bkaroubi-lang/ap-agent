# ap-agent: an LLM-as-judge harness that caught its own headline failing to replicate

A Pydantic-AI accounts-payable agent (invoice extraction, anomaly flagging, Q&A)
plus a judge-calibration layer that treats LLM-as-judge agreement as an
inter-rater reliability problem. The agent is standard; the interesting part is
the evaluation — and what happened when I tried to reproduce its headline
finding three months later.

## TL;DR — a reproducibility story

In **July 2026**, two Groq-hosted judges (`llama-3.3-70b-versatile` and
`openai/gpt-oss-20b`) produced a striking result on a 36-item Q&A gold set:
inter-judge κ "collapsed" from 0.92 to 0.61 when I added subtle
epistemic-honesty items, and the **larger** model was dramatically worse on
them (llama 20% vs gpt-oss-20b 70%) — it preferred confident fabrications over
honest "I don't have that data."

In **October 2026** I re-ran the exact same gold set to cache it for offline
reproduction (`--replay`, below). Two things happened:

1. **`llama-3.3-70b-versatile` no longer exists.** Groq retired it; every call
   now 404s. The original headline is not reproducible — the model is gone.
2. **Re-running with a current same-lab size comparison (`gpt-oss-120b` vs
   `gpt-oss-20b`), the finding reversed.** There was no kappa collapse, and the
   **larger** model was now the *more* honest one on the subtle items.

That is the real result of this project: **an LLM-eval finding was contingent
on a specific model pair that was deprecated within a quarter, and re-running it
flipped the conclusion.** The lesson isn't a single number about one model — it's
that eval findings rot, and you need the infrastructure (seeded metrics + a
record/replay cache) to know when they have.

## July vs. October

| Metric | July: llama-3.3-70b vs gpt-oss-20b | October: gpt-oss-120b vs gpt-oss-20b |
|---|---|---|
| κ, trivial 26-item subset | **0.92** ("almost perfect") | **0.49** (moderate) |
| κ, full 36-item set | **0.61** | **0.46** [0.19, 0.72] |
| "Kappa collapse" on adding subtle items | yes (0.92 → 0.61) | **no** (0.49 → 0.46) |
| Subtle-item accuracy, **larger** model | **20%** (llama) | **80%** (gpt-oss-120b) |
| Subtle-item accuracy, **smaller** model | **70%** (gpt-oss-20b) | **50%** (gpt-oss-20b) |
| "Smaller model is more honest" | held | **reversed** |
| Position bias (first-listed judge) | p = 0.13 (not sig.) | **p = 0.004 (significant)** |

Same gold set, same harness, same protocol, same `temperature=0`. Only the
judges changed — and the headline changed with them.

## What's contingent vs. what's robust

The specific July claims were **contingent** on the llama/gpt-oss pair:

- There is no universal "kappa collapse." With the gpt-oss pair, the two judges
  only agree moderately (κ ≈ 0.49) even on *trivial* items, and adding subtle
  items barely moves it. The dramatic July collapse was a property of starting
  from an unusually high-agreement pair.
- "Bigger model is worse / smaller model is more honest" does **not** generalize.
  On the subtle honesty items, gpt-oss-120b scored 80% and gpt-oss-20b 50% — the
  opposite ordering to July. Model size predicts nothing here; the *specific
  model* is what matters.

What **does** survive both runs — the actual, defensible thesis:

1. **Inter-judge agreement is not evidence of judge quality.** In July, 0.92
   agreement coexisted with one judge being 20% accurate on hard items. In
   October, moderate agreement (0.46) coexists with a 30-point accuracy gap on
   subtle items (120b 80% vs 20b 50%). Agreement and quality are different axes
   in both runs.
2. **Judge quality on hard/honesty items is model-specific and must be measured
   per model, per configuration.** You cannot infer it from size, from a sister
   model, or from a previous quarter's run.
3. **Eval findings have a shelf life.** A provider can retire the model your
   headline rests on. Reproducibility infrastructure is not optional polish —
   it's what tells you your result still holds.

## October results in detail (gpt-oss-120b vs gpt-oss-20b)

### Inter-judge agreement (Cohen's κ)

| Gold set | n | Cohen's κ | 95% CI | Interpretation |
|---|---:|---:|---|---|
| Trivial subset | 26 | 0.492 | — | moderate |
| Full set | 36 | **0.464** | [0.192, 0.723] | moderate |

No collapse: the two same-family judges are only moderately consistent even on
easy items.

### Judge accuracy vs. gold, sliced by difficulty

| Model | Overall | Trivial (0–25) | **Subtle (26–35)** |
|---|---:|---:|---:|
| `gpt-oss-120b` | 69.4% (25/36) | 65.4% (17/26) | **80.0% (8/10)** |
| `gpt-oss-20b` | 63.9% (23/36) | 69.2% (18/26) | **50.0% (5/10)** |

The larger model is the stronger judge on the subtle honesty items (+30 points)
— the reverse of the July llama/gpt-oss result.

### Position bias (first-listed judge: gpt-oss-120b, A/B swap on all 36)

- Prefer-first rate: **0.25** (n = 36)
- Wilson 95% CI: [0.138, 0.411]
- Two-sided binomial p vs 0.5: **0.004 (significant)**

Unlike July (where position bias was not significant), gpt-oss-120b shows a
real, significant tendency to favor the *second*-listed answer. A position-bias
test earns its keep precisely because the answer differs by model — you cannot
assume it away.

### Confidence calibration (Brier)

| Model | Brier |
|---|---:|
| `gpt-oss-120b` | 0.232 |
| `gpt-oss-20b` | 0.240 |

Both are reasonable and similar — neither is the confidently-wrong outlier that
July's llama was (Brier 0.40).

### Verbosity

Spearman ρ between judge score and total candidate length = **−0.37** (p = 0.0015,
n = 72). A weak-to-moderate signal, not a strict paired verbosity-bias test;
retained as a caveat, not a headline.

## Deterministic anomaly detection (no LLM, no cost)

Unchanged across runs — it has no model dependency. 60 synthetic invoices, 9
injected anomalies (15%), rule pass scored set-based against injected labels:

| Category | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| duplicate | 2 | 0 | 0 | 1.000 | 1.000 | **1.000** |
| amount_outlier | 3 | 0 | 0 | 1.000 | 1.000 | **1.000** |
| off_cadence | 2 | 0 | 1 | 1.000 | 0.667 | 0.800 |
| missing_po | 1 | 3 | 0 | 0.250 | 1.000 | 0.400 |
| **micro** | 8 | 3 | 1 | 0.727 | 0.889 | **0.800** |
| **invoice-level flag** | 8 | 2 | 1 | 0.800 | 0.889 | **0.842** |

The two crisp rules are perfect; the `missing_po` false positives are vendors
near (not at) 100% historical PO rate — arguably worth surfacing in a real AP
workflow. Left as-is.

## LLM invoice extraction

50 invoices, `gpt-oss-120b`, 3 s per-call delay. Of the 50, **44 returned a
schema-valid `Invoice`** and 6 failed Pydantic-AI's output validation (the model
occasionally emits non-conforming JSON; those items are skipped, not scored).
On the 44 that returned:

- Mean field accuracy: **1.000** (all 9 fields correct on every returned invoice)
- Mean line-item F1: **1.000**

Note the **eval-strictness audit** that still stands from the original work: the
line-item eval once keyed identity on `str(unit_price)`, so `342.50` vs `342.5`
(numerically identical) were scored as mismatches, manufacturing a fake error.
Fixed by normalizing to `f"{float(x):.2f}"`. Auditing your own eval is the point.

## Reproducibility — record / replay

Every LLM call routes through a content-addressed cache
(`eval/llm_cache.py`, keyed on model + output-type + exact prompt). Downstream
metrics are seeded, so a replay reproduces the numbers above **exactly, offline,
with no API key and no cost**:

```bash
uv sync
uv run python -m eval.harness --replay        # reproduces this report from results/llm_cache.json
```

To refresh the cache against live models (billed, needs GROQ_API_KEY):

```bash
uv run python -m eval.harness --record
```

Deterministic legs need neither a key nor the cache:

```bash
uv run python scripts/generate_invoices.py
uv run python -m eval.harness --no-llm         # anomaly leg only
```

The shipped `results/llm_cache.json` (152 responses) is what makes the October
headline reproducible by anyone — the exact thing the July run lacked when its
model disappeared.

## Caveats

- **Model deprecation is the headline, not a footnote.** `llama-3.3-70b-versatile`
  was retired by Groq between the two runs; the original finding cannot be
  reproduced. This is why the cache exists.
- **Small gold set.** n = 36 (26 trivial + 10 subtle) yields usable κ CIs but is
  not publication-scale. A real study wants ≥ 100 items, ~1:1 trivial:subtle.
- **Non-determinism.** Even at `temperature=0`, hosted LLMs are not perfectly
  deterministic; the cache freezes one run so the *analysis* is exactly
  reproducible even though a fresh `--record` may differ slightly.
- **Synthetic data.** Vendors, invoices, and anomalies are generated by
  `scripts/generate_invoices.py`. The reproducibility lesson transfers; the exact
  accuracies are dataset-specific.
- **Two judges, one provider.** A cross-provider judge (e.g. a Claude judge via
  Anthropic) would strengthen the "contingent across labs" point — the harness is
  provider-agnostic and ready for it.
- **6/50 extraction failures** are schema-validation retries, not extraction
  errors on returned invoices.

---

Original run: 2026-07-07 (llama-3.3-70b-versatile, since retired).
Reproduction + reframe: 2026-10-08 (gpt-oss-120b vs gpt-oss-20b).
