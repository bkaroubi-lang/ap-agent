"""Cohen's kappa and Fleiss' kappa on canonical worked examples."""

import numpy as np

from eval.calibration import (
    cohens_kappa,
    fleiss_kappa,
    position_bias_test,
    reliability_curve,
    verbosity_bias_correlation,
)


def test_cohens_kappa_binary_worked_example():
    # Two raters, binary decisions. Confusion matrix:
    # rater B:      yes  no
    # rater A yes:  20   5
    # rater A no:   10   15
    # po = (20+15)/50 = 0.70
    # p_yes = row 0 sum / 50 = 25/50 = 0.5,  col 0 sum / 50 = 30/50 = 0.6
    # p_no  = 25/50 = 0.5,                   20/50 = 0.4
    # pe = 0.5*0.6 + 0.5*0.4 = 0.5
    # kappa = (0.70 - 0.5) / (1 - 0.5) = 0.40
    a = np.array(["yes"] * 25 + ["no"] * 25)
    b = np.array(["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15)
    k = cohens_kappa(a, b)
    assert abs(k - 0.40) < 1e-9


def test_cohens_kappa_perfect_agreement():
    a = np.array([0, 1, 1, 0, 1])
    assert cohens_kappa(a, a) == 1.0


def test_cohens_kappa_random_labels_near_zero():
    rng = np.random.default_rng(0)
    n = 5000
    a = rng.integers(0, 3, size=n)
    b = rng.integers(0, 3, size=n)
    assert abs(cohens_kappa(a, b)) < 0.05


def test_fleiss_kappa_wikipedia_example():
    """Wikipedia's Fleiss' kappa worked example:
    10 items, 14 raters each, 5 categories. Expected kappa ≈ 0.210.
    """
    M = np.array(
        [
            [0, 0, 0, 0, 14],
            [0, 2, 6, 4, 2],
            [0, 0, 3, 5, 6],
            [0, 3, 9, 2, 0],
            [2, 2, 8, 1, 1],
            [7, 7, 0, 0, 0],
            [3, 2, 6, 3, 0],
            [2, 5, 3, 2, 2],
            [6, 5, 2, 1, 0],
            [0, 2, 2, 3, 7],
        ],
        dtype=float,
    )
    k = fleiss_kappa(M)
    assert abs(k - 0.2099) < 0.005


def test_position_bias_no_bias():
    rng = np.random.default_rng(42)
    flags = list(rng.random(size=200) < 0.5)
    r = position_bias_test(flags)
    assert 0.3 < r.prefer_first_rate < 0.7
    assert r.p_value > 0.01


def test_position_bias_detects_bias():
    flags = [True] * 40 + [False] * 10
    r = position_bias_test(flags)
    assert r.prefer_first_rate == 0.8
    assert r.p_value < 1e-4


def test_verbosity_correlation_direction():
    # Longer answers get higher scores -> positive rho
    scores = list(range(1, 21))
    lengths = [50 + 10 * s for s in scores]
    v = verbosity_bias_correlation(scores, lengths)
    assert v["rho"] > 0.95
    assert v["p_value"] < 1e-6


def test_reliability_curve_perfect():
    # Perfectly calibrated: high confidence -> correct, low -> incorrect
    correct = [True, True, True, True, False, False, False, False]
    conf = [0.9, 0.85, 0.8, 0.75, 0.25, 0.2, 0.15, 0.1]
    rel = reliability_curve(correct, conf, n_bins=2)
    assert rel["brier"] < 0.10
