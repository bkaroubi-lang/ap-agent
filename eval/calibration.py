"""Inter-rater reliability + judge-bias diagnostics.

Everything here is offline math: given rating matrices (items x raters), we
compute Cohen's/Fleiss' kappa with bootstrapped CIs, run bias tests (A/B
position swap, verbosity correlation), and build reliability curves for judge
self-reported confidence vs empirical accuracy.

The judge module (eval/judge.py) produces the ratings; this module scores them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# ---------- Cohen's kappa (2 raters, categorical) ----------


def cohens_kappa(a: Sequence, b: Sequence) -> float:
    """Cohen's kappa. Ratings can be any hashable categorical values."""
    a = np.asarray(a)
    b = np.asarray(b)
    if len(a) != len(b):
        raise ValueError("cohens_kappa: rater arrays must be equal length.")
    n = len(a)
    categories = sorted(set(a) | set(b), key=str)
    idx = {c: i for i, c in enumerate(categories)}
    K = len(categories)
    conf = np.zeros((K, K), dtype=float)
    for x, y in zip(a, b, strict=False):
        conf[idx[x], idx[y]] += 1.0
    po = np.trace(conf) / n
    pa = conf.sum(axis=1) / n
    pb = conf.sum(axis=0) / n
    pe = float((pa * pb).sum())
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return float((po - pe) / (1.0 - pe))


# ---------- Fleiss' kappa (>=2 raters, categorical) ----------


def fleiss_kappa(ratings: np.ndarray) -> float:
    """Fleiss' kappa on a matrix of shape (n_items, n_categories),
    where entry [i, k] is the number of raters who assigned category k
    to item i. The number of raters must be constant across items.
    """
    ratings = np.asarray(ratings, dtype=float)
    if ratings.ndim != 2:
        raise ValueError("fleiss_kappa: ratings must be 2-D (items x categories).")
    n_items = ratings.shape[0]
    n_raters_per_item = ratings.sum(axis=1)
    if not np.allclose(n_raters_per_item, n_raters_per_item[0]):
        raise ValueError("fleiss_kappa: number of raters per item must be constant.")
    n = int(n_raters_per_item[0])
    if n < 2:
        raise ValueError("fleiss_kappa: need >= 2 raters per item.")
    # P_i (agreement per item)
    P = ((ratings**2).sum(axis=1) - n) / (n * (n - 1))
    P_bar = float(P.mean())
    # P_e (category prevalence)
    p_j = ratings.sum(axis=0) / (n_items * n)
    P_e = float((p_j**2).sum())
    if P_e == 1.0:
        return 1.0 if P_bar == 1.0 else 0.0
    return (P_bar - P_e) / (1.0 - P_e)


# ---------- Bootstrap CIs for kappa ----------


def bootstrap_kappa_ci(
    kappa_fn,
    *args,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Nonparametric bootstrap over items. Works for both cohens_kappa
    (positional args are two 1-D arrays) and fleiss_kappa (positional
    arg is a 2-D matrix)."""
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(x) for x in args]
    n_items = arrays[0].shape[0]
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_items, size=n_items)
        resampled = tuple(x[idx] for x in arrays)
        boots[b] = kappa_fn(*resampled)
    return {
        "point": float(kappa_fn(*arrays)),
        "ci_lower": float(np.quantile(boots, alpha / 2)),
        "ci_upper": float(np.quantile(boots, 1 - alpha / 2)),
    }


# ---------- Bias tests ----------


@dataclass
class PositionBiasResult:
    prefer_first_rate: float
    n: int
    p_value: float  # two-sided binomial test vs 0.5
    ci_lower: float
    ci_upper: float


def position_bias_test(prefer_first: Sequence[bool]) -> PositionBiasResult:
    """Given a list of judge verdicts on pairwise swaps — True if the judge
    preferred the item shown in position A across both orderings — test
    whether the rate deviates from 0.5.

    Wilson score CI for the proportion; exact binomial test for p-value.
    """
    x = np.asarray(prefer_first).astype(int)
    n = len(x)
    if n == 0:
        return PositionBiasResult(0.0, 0, 1.0, 0.0, 1.0)
    k = int(x.sum())
    rate = k / n
    result = stats.binomtest(k, n, p=0.5, alternative="two-sided")
    p_value = float(result.pvalue)
    ci = result.proportion_ci(confidence_level=0.95, method="wilson")
    return PositionBiasResult(
        prefer_first_rate=rate,
        n=n,
        p_value=p_value,
        ci_lower=float(ci.low),
        ci_upper=float(ci.high),
    )


def verbosity_bias_correlation(scores: Sequence[float], lengths: Sequence[int]) -> dict:
    """Spearman correlation between judge scores and answer lengths. Positive
    correlation is the classic verbosity-bias tell."""
    scores = np.asarray(scores, dtype=float)
    lengths = np.asarray(lengths, dtype=float)
    if len(scores) < 3:
        return {"n": len(scores), "rho": 0.0, "p_value": 1.0}
    rho, p = stats.spearmanr(scores, lengths)
    return {"n": len(scores), "rho": float(rho), "p_value": float(p)}


# ---------- Judge reliability (confidence vs accuracy) ----------


def reliability_curve(
    correct: Sequence[bool], confidence: Sequence[float], n_bins: int = 10
) -> dict:
    """Given judge self-reported confidence and whether the judge's verdict
    matched gold, produce a reliability curve + Brier score. `confidence`
    should be a probability in [0,1]."""
    y = np.asarray(correct).astype(int)
    p = np.asarray(confidence, dtype=float)
    if len(y) < n_bins:
        n_bins = max(2, len(y) // 2)
    prob_true, prob_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    return {
        "brier": float(brier_score_loss(y, p)),
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
    }


# ---------- Kappa magnitude interpretation ----------


def kappa_label(k: float) -> str:
    """Landis-Koch magnitude labels, for a legend in reports (not policy)."""
    if k < 0.0:
        return "worse than chance"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"
