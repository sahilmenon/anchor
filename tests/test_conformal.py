"""Tests for split conformal abstention.

The important tests here are not the unit tests -- they are the *empirical*
ones. A conformal implementation with an off-by-one in the quantile index
still passes every type check and every smoke test; the only thing that
catches it is running the whole calibrate/evaluate loop over many independent
draws and checking that the guarantee actually holds on average. That is what
`test_retention_guarantee_holds_across_seeds` does, and it is the test that
justifies the module's headline claim.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from anchor.conformal import (
    IMPOSSIBLE_THRESHOLD,
    Scored,
    always_answer_baseline,
    aurc,
    calibrate,
    evaluate,
    risk_coverage_curve,
)

ALPHA = 0.10


# --------------------------------------------------------------------------
# Synthetic generators
# --------------------------------------------------------------------------


def separable_items(rng: np.random.Generator, n: int, accuracy: float = 0.85) -> list[Scored]:
    """A ranker whose confidence genuinely tracks correctness.

    Correct items draw confidence from Beta(8, 2) (mass near 1), incorrect
    items from Beta(2, 8) (mass near 0). The distributions overlap in the
    middle, so this is a good-but-not-oracle extractor -- the realistic case.
    """
    correct = rng.random(n) < accuracy
    conf = np.where(correct, rng.beta(8.0, 2.0, size=n), rng.beta(2.0, 8.0, size=n))
    return [
        Scored(confidence=float(c), correct=bool(k))
        for c, k in zip(conf, correct, strict=True)
    ]


def calibrated_items(rng: np.random.Generator, n: int) -> list[Scored]:
    """Perfectly calibrated: P(correct | confidence = c) == c exactly."""
    conf = rng.random(n)
    correct = rng.random(n) < conf
    return [
        Scored(confidence=float(c), correct=bool(k))
        for c, k in zip(conf, correct, strict=True)
    ]


def anti_calibrated_items(rng: np.random.Generator, n: int) -> list[Scored]:
    """Pathological: the extractor is most confident exactly when it is wrong."""
    conf = rng.random(n)
    correct = rng.random(n) < (1.0 - conf)
    return [
        Scored(confidence=float(c), correct=bool(k))
        for c, k in zip(conf, correct, strict=True)
    ]


def random_confidence_items(
    rng: np.random.Generator, n: int, accuracy: float = 0.7
) -> list[Scored]:
    """Confidence is pure noise, independent of correctness."""
    correct = rng.random(n) < accuracy
    conf = rng.random(n)
    return [
        Scored(confidence=float(c), correct=bool(k))
        for c, k in zip(conf, correct, strict=True)
    ]


def perfect_ranker_items(n: int, accuracy: float = 0.7) -> list[Scored]:
    """Every correct item ranked strictly above every incorrect one."""
    n_correct = int(n * accuracy)
    items = [
        Scored(confidence=0.5 + 0.5 * (i + 1) / n_correct, correct=True) for i in range(n_correct)
    ]
    items += [
        Scored(confidence=0.5 * (i + 1) / (n - n_correct), correct=False)
        for i in range(n - n_correct)
    ]
    return items


def oracle_rerank(items: list[Scored]) -> list[Scored]:
    """Same correctness labels, confidences reassigned to rank perfectly.

    AURC is bounded below by the base error rate, so comparing two generators
    with *different* realised accuracies compares the accuracies, not the
    rankings. Holding the label multiset fixed and only permuting confidences
    isolates ranking quality, which is what AURC is supposed to measure.
    """
    n = len(items)
    n_correct = sum(1 for it in items if it.correct)
    hi = [Scored(0.5 + 0.5 * (i + 1) / max(n_correct, 1), True) for i in range(n_correct)]
    lo = [Scored(0.5 * (i + 1) / max(n - n_correct, 1), False) for i in range(n - n_correct)]
    return hi + lo


def shuffle_confidences(items: list[Scored], rng: np.random.Generator) -> list[Scored]:
    """Same labels, same confidence multiset, association destroyed."""
    confs = rng.permutation([it.confidence for it in items])
    return [Scored(float(c), it.correct) for c, it in zip(confs, items, strict=True)]


# --------------------------------------------------------------------------
# The quantile index itself
# --------------------------------------------------------------------------


def test_quantile_index_matches_closed_form():
    """Threshold must equal the ceil((n+1)(1-alpha))-th smallest score, exactly."""
    rng = np.random.default_rng(0)
    for n in (9, 10, 19, 20, 37, 100, 101):
        confs = rng.random(n)
        cal = [Scored(confidence=float(c), correct=True) for c in confs]
        k = math.ceil((n + 1) * (1.0 - ALPHA))
        expected_score = float(np.sort(1.0 - confs)[k - 1])
        assert calibrate(cal, ALPHA) == pytest.approx(1.0 - expected_score)


def test_finite_sample_correction_is_not_the_naive_quantile():
    """Guard the off-by-one: at n=10, alpha=0.1 the corrected index is 10, not 9."""
    confs = [i / 10.0 for i in range(1, 11)]  # 0.1 .. 1.0
    cal = [Scored(confidence=c, correct=True) for c in confs]
    # ceil(11 * 0.9) = 10 -> 10th smallest score = 1 - 0.1 = 0.9 -> t = 0.1.
    # The naive ceil(10 * 0.9) = 9 would give t = 0.2 and under-cover.
    assert calibrate(cal, ALPHA) == pytest.approx(0.1)


def test_incorrect_calibration_items_are_ignored():
    """Only correct items contribute nonconformity scores."""
    correct = [Scored(confidence=0.5 + 0.005 * i, correct=True) for i in range(40)]
    noise = [Scored(confidence=0.99, correct=False) for _ in range(40)]
    assert calibrate(correct, ALPHA) == pytest.approx(calibrate(correct + noise, ALPHA))


# --------------------------------------------------------------------------
# The guarantee, empirically, across many seeds
# --------------------------------------------------------------------------


def test_retention_guarantee_holds_across_seeds():
    """P(answered | correct) >= 1 - alpha on the test split.

    This is the exact distribution-free claim of split conformal. It holds in
    expectation over calibration draws, so a single seed may dip below
    1 - alpha by finite-sample noise; the mean across seeds must not.
    """
    retentions = []
    for seed in range(200):
        rng = np.random.default_rng(seed)
        items = separable_items(rng, 4000)
        cal, test = items[:2000], items[2000:]

        t = calibrate(cal, ALPHA)
        correct_test = [it for it in test if it.correct]
        answered = sum(1 for it in correct_test if it.confidence >= t)
        retentions.append(answered / len(correct_test))

    mean_retention = float(np.mean(retentions))
    assert mean_retention >= 1.0 - ALPHA, f"mean retention {mean_retention:.4f} < {1 - ALPHA}"
    # Per-seed dips are allowed but must be small and rare.
    # The guarantee is marginal, not per-calibration-draw: the threshold is an
    # order statistic, so individual seeds scatter around 1 - alpha. The mean
    # above is the guarantee; these bound the scatter.
    assert min(retentions) >= 1.0 - ALPHA - 0.05
    assert sum(r < 1.0 - ALPHA for r in retentions) / len(retentions) < 0.65


def test_error_on_answered_respects_alpha_across_seeds():
    """The headline claim, for a ranker whose confidences separate.

    With a separating confidence signal the retention guarantee translates
    into error control on the answered set: `guarantee_held` should be true on
    essentially every draw.
    """
    errors = []
    held = 0
    for seed in range(200):
        rng = np.random.default_rng(1000 + seed)
        items = separable_items(rng, 1000)
        cal, test = items[:500], items[500:]

        result = evaluate(test, calibrate(cal, ALPHA), ALPHA)
        errors.append(result.error_on_answered)
        held += int(result.guarantee_held)
        # Abstaining on half the corpus would not be a useful operating point.
        assert result.coverage > 0.5

    assert float(np.mean(errors)) <= ALPHA
    assert held / 200 >= 0.98, f"guarantee held on only {held}/200 seeds"


def test_conformal_beats_always_answer_baseline():
    """The comparison that makes the number meaningful."""
    rng = np.random.default_rng(7)
    items = separable_items(rng, 1200, accuracy=0.8)
    cal, test = items[:600], items[600:]

    conformal = evaluate(test, calibrate(cal, ALPHA), ALPHA)
    baseline = always_answer_baseline(test, ALPHA)

    assert baseline.coverage == 1.0
    assert baseline.n_answered == baseline.n_total
    assert not baseline.guarantee_held  # ~20% error at full coverage
    assert conformal.error_on_answered < baseline.error_on_answered
    assert conformal.guarantee_held


# --------------------------------------------------------------------------
# Calibration regimes: calibrated / anti-calibrated / random
# --------------------------------------------------------------------------


def test_perfectly_calibrated_confidences_retain_correct_items():
    """Under P(correct|c) = c the retention guarantee still holds exactly...

    ...even though error-on-answered does *not* fall below alpha. This is the
    honest separation between what conformal guarantees (retention) and what a
    weak confidence signal can deliver (error control).
    """
    retentions = []
    for seed in range(60):
        rng = np.random.default_rng(2000 + seed)
        items = calibrated_items(rng, 2000)
        cal, test = items[:1000], items[1000:]
        t = calibrate(cal, ALPHA)
        correct_test = [it for it in test if it.correct]
        retentions.append(sum(it.confidence >= t for it in correct_test) / len(correct_test))

    assert float(np.mean(retentions)) >= 1.0 - ALPHA


def test_anti_calibrated_confidences_fail_loudly_not_silently():
    rng = np.random.default_rng(3)
    items = anti_calibrated_items(rng, 2000)
    cal, test = items[:1000], items[1000:]
    result = evaluate(test, calibrate(cal, ALPHA), ALPHA)

    # Correct items sit at low confidence, so the threshold is low, coverage is
    # near-total and the error target is missed -- reported, not hidden.
    assert result.coverage > 0.8
    assert not result.guarantee_held
    # And the threshold-free diagnostic says the ranking is worse than noise.
    assert aurc(items) > aurc(random_confidence_items(np.random.default_rng(4), 2000, 0.5))


def test_random_confidences_give_no_selective_benefit():
    rng = np.random.default_rng(5)
    items = random_confidence_items(rng, 4000, accuracy=0.7)
    cal, test = items[:2000], items[2000:]
    result = evaluate(test, calibrate(cal, ALPHA), ALPHA)
    baseline = always_answer_baseline(test, ALPHA)

    # Noise confidences cannot reduce error: abstaining removes correct and
    # incorrect items in the same proportion.
    assert result.error_on_answered == pytest.approx(baseline.error_on_answered, abs=0.05)
    assert not result.guarantee_held


# --------------------------------------------------------------------------
# Degenerate inputs
# --------------------------------------------------------------------------


def test_empty_calibration_set_admits_nothing():
    t = calibrate([], ALPHA)
    assert t > 1.0
    assert t == IMPOSSIBLE_THRESHOLD
    result = evaluate([Scored(1.0, True), Scored(1.0, False)], t, ALPHA)
    assert result.n_answered == 0
    assert result.coverage == 0.0
    assert result.guarantee_held


def test_guarantee_impossible_for_tiny_n():
    """alpha=0.1 needs >= 9 correct calibration items; fewer admits nothing."""
    for n in range(0, 9):
        cal = [Scored(confidence=0.9, correct=True) for _ in range(n)]
        assert calibrate(cal, ALPHA) == IMPOSSIBLE_THRESHOLD, f"n={n} should be infeasible"
    cal9 = [Scored(confidence=0.9, correct=True) for _ in range(9)]
    # ceil(10 * 0.9) = 9 <= 9 -> feasible, and the threshold is finite.
    assert calibrate(cal9, ALPHA) <= 1.0


def test_n_one_is_feasible_only_for_large_alpha():
    single = [Scored(confidence=0.42, correct=True)]
    assert calibrate(single, ALPHA) == IMPOSSIBLE_THRESHOLD
    # ceil(2 * 0.5) = 1 <= 1 -> the single item's own confidence is the threshold.
    assert calibrate(single, 0.5) == pytest.approx(0.42)


def test_all_incorrect_calibration_set_admits_nothing():
    cal = [Scored(confidence=0.99, correct=False) for _ in range(500)]
    assert calibrate(cal, ALPHA) == IMPOSSIBLE_THRESHOLD


def test_all_correct_calibration_set_gives_permissive_threshold():
    confs = [0.30 + 0.001 * i for i in range(500)]
    cal = [Scored(confidence=c, correct=True) for c in confs]
    t = calibrate(cal, ALPHA)
    # Nothing was ever wrong, so the threshold sits near the bottom of the
    # observed confidence range and coverage stays high.
    assert 0.30 <= t <= 0.40
    assert evaluate(cal, t, ALPHA).coverage >= 1.0 - ALPHA


def test_empty_test_set():
    result = evaluate([], 0.5, ALPHA)
    assert result.n_total == 0
    assert result.n_answered == 0
    assert result.coverage == 0.0
    assert result.error_on_answered == 0.0


def test_empty_curve_and_aurc():
    assert risk_coverage_curve([]) == ([], [])
    assert aurc([]) == 0.0
    assert aurc([Scored(0.5, True)]) == 0.0


def test_confidence_must_be_in_unit_interval():
    with pytest.raises(ValueError):
        Scored(confidence=1.5, correct=True)
    with pytest.raises(ValueError):
        Scored(confidence=-0.01, correct=False)


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.2])
def test_invalid_alpha_rejected(bad_alpha):
    with pytest.raises(ValueError):
        calibrate([Scored(0.9, True)], bad_alpha)


# --------------------------------------------------------------------------
# Monotonicity and curve shape
# --------------------------------------------------------------------------


def test_raising_threshold_never_increases_coverage():
    rng = np.random.default_rng(11)
    items = separable_items(rng, 500)
    prev = 1.1
    for t in np.linspace(0.0, 1.0, 101):
        cov = evaluate(items, float(t), ALPHA).coverage
        assert cov <= prev + 1e-12
        prev = cov


def test_looser_alpha_never_raises_the_threshold():
    """A larger alpha asks for less retention, so it can only abstain more."""
    rng = np.random.default_rng(12)
    cal = separable_items(rng, 1000)
    thresholds = [calibrate(cal, a) for a in (0.02, 0.05, 0.10, 0.20, 0.40)]
    finite = [t for t in thresholds if t <= 1.0]
    assert finite == sorted(finite)


def test_curve_is_sorted_by_coverage_and_well_formed():
    rng = np.random.default_rng(13)
    items = separable_items(rng, 300)
    coverages, risks = risk_coverage_curve(items)

    assert len(coverages) == len(risks) > 0
    assert coverages == sorted(coverages)
    assert all(0.0 < c <= 1.0 for c in coverages)
    assert all(0.0 <= r <= 1.0 for r in risks)
    assert coverages[-1] == pytest.approx(1.0)
    # Full coverage == the always-answer baseline error.
    assert risks[-1] == pytest.approx(always_answer_baseline(items).error_on_answered)


def test_perfect_ranker_curve_starts_at_zero_risk():
    items = perfect_ranker_items(200, accuracy=0.7)
    coverages, risks = risk_coverage_curve(items)
    # Risk stays at zero while only correct items are admitted.
    assert risks[0] == 0.0
    assert max(r for c, r in zip(coverages, risks, strict=True) if c <= 0.7) == 0.0
    assert risks[-1] == pytest.approx(0.3)


# --------------------------------------------------------------------------
# AURC
# --------------------------------------------------------------------------


def test_aurc_perfect_ranker_beats_random_ranker():
    """Held at a fixed label multiset, so only the ranking differs."""
    rng = np.random.default_rng(17)
    items = separable_items(rng, 600, accuracy=0.7)
    assert aurc(oracle_rerank(items)) < aurc(shuffle_confidences(items, rng))


def test_aurc_ordering_is_stable_across_seeds():
    """perfect < real ranker < random, on identical labels, every seed."""
    for seed in range(40):
        rng = np.random.default_rng(5000 + seed)
        items = separable_items(rng, 400, accuracy=0.75)
        a_perfect = aurc(oracle_rerank(items))
        a_real = aurc(items)
        a_random = aurc(shuffle_confidences(items, rng))
        assert a_perfect <= a_real < a_random, f"seed {seed}: {a_perfect} {a_real} {a_random}"


def test_aurc_of_flawless_extractor_is_zero():
    items = [Scored(confidence=float(i) / 100.0, correct=True) for i in range(1, 101)]
    assert aurc(items) == 0.0


def test_aurc_of_random_ranker_approximates_base_error():
    """With noise confidences, risk is flat at the base error across the sweep."""
    rng = np.random.default_rng(19)
    items = random_confidence_items(rng, 5000, accuracy=0.7)
    coverages, _ = risk_coverage_curve(items)
    span = coverages[-1] - coverages[0]
    assert aurc(items) == pytest.approx(0.3 * span, abs=0.03)


def test_aurc_matches_manual_trapezoid():
    rng = np.random.default_rng(23)
    items = separable_items(rng, 150)
    coverages, risks = risk_coverage_curve(items)
    assert aurc(items) == pytest.approx(float(np.trapezoid(risks, coverages)))
