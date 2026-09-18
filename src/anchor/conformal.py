"""Split conformal abstention for extracted credit metrics.

Why this module exists
----------------------
Every confidence-gated extraction pipeline eventually has to answer one
question: *where do we put the cutoff?* The usual answer -- "0.8 looked good on
the dev set" -- is a hand-picked number with no guarantee attached, and it
silently re-breaks every time the model, the prompt, or the document mix
changes. A credit risk committee cannot sign off on that.

Split conformal prediction replaces the hand-picked cutoff with a threshold
*derived* from a held-out calibration split, carrying a finite-sample,
distribution-free guarantee. The only assumption is exchangeability of the
calibration and test items -- no assumption on the model, the confidence
scale being calibrated, or the document distribution.

Following "Mitigating LLM Hallucinations via Conformal Abstention"
(arXiv:2405.01563), the nonconformity score of a scored item is::

    s = 1 - confidence

so a *low* score means the extractor was confident. We calibrate on the
CORRECT calibration items only, and the guarantee we buy is precisely:

    P( item is answered | item is correct )  >=  1 - alpha

i.e. *at most an alpha fraction of the answers we should have given are
thrown away*. This is a retention (coverage) guarantee on correct items.

Honesty about the headline number
---------------------------------
The distribution-free part is the statement above. The quantity a credit
committee actually cares about -- ``error_on_answered <= alpha`` -- is a
*consequence* of that guarantee only when the confidence signal separates
correct from incorrect items: the threshold admits >= 1 - alpha of correct
items, and admits incorrect items only to the extent that the extractor is
confidently wrong. `evaluate` therefore measures `error_on_answered`
empirically on a test split and reports `guarantee_held` rather than
asserting it. A pipeline whose confidences do not rank at all will show a
held retention guarantee and a *failed* error target, which is the correct
and informative outcome -- it says the confidences are the problem, not the
threshold.

Vocabulary used throughout: *coverage* = fraction of items answered (not
abstained); *risk* = error rate among the answered items.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "IMPOSSIBLE_THRESHOLD",
    "ConformalResult",
    "Scored",
    "always_answer_baseline",
    "aurc",
    "calibrate",
    "evaluate",
    "risk_coverage_curve",
]


# A confidence threshold strictly above the top of the confidence scale, so
# that `confidence >= threshold` is false for every possible item. Returned
# when the calibration set is too small to support the requested alpha; see
# `calibrate`.
IMPOSSIBLE_THRESHOLD: float = 1.0 + 1e-9


@dataclass(frozen=True)
class Scored:
    """One scored extraction: a self-reported confidence and a verdict.

    Deliberately minimal and standalone. This module does not import from
    `anchor.scoring`; anything that can produce a confidence in [0, 1] and a
    boolean correctness can be adapted into a `Scored` in one line, which
    keeps the statistical core testable without the extraction stack.
    """

    confidence: float
    correct: bool

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must lie in [0, 1], got {self.confidence!r}")


@dataclass(frozen=True)
class ConformalResult:
    """Outcome of applying one threshold to a test split."""

    threshold: float
    coverage: float
    error_on_answered: float
    n_answered: int
    n_total: int
    guarantee_held: bool


def calibrate(cal: list[Scored], alpha: float = 0.10) -> float:
    """Derive a confidence threshold from a calibration split.

    Parameters
    ----------
    cal:
        Held-out calibration items. Must not overlap the test split -- split
        conformal's guarantee comes from the threshold being independent of
        the data it is evaluated on.
    alpha:
        Target miss rate in (0, 1). alpha = 0.10 targets "retain at least 90%
        of the answers we should have given".

    Returns
    -------
    float
        A *confidence* threshold `t`. Answer a field when
        ``field.confidence >= t``, abstain otherwise.

    Why the quantile index is ceil((n + 1) * (1 - alpha))
    -----------------------------------------------------
    Let the n correct calibration items have nonconformity scores
    ``s_1..s_n``, and let ``s_{n+1}`` be the score of a fresh correct test
    item. Under exchangeability all (n + 1)! orderings of
    ``s_1..s_{n+1}`` are equally likely, so the rank of ``s_{n+1}`` among the
    n + 1 values is uniform on ``{1, ..., n + 1}``::

        P( rank(s_{n+1}) <= k )  =  k / (n + 1)

    We want ``P( s_{n+1} <= s_(k) ) >= 1 - alpha`` where ``s_(k)`` is the k-th
    smallest calibration score. The event ``s_{n+1} <= s_(k)`` contains the
    event ``rank(s_{n+1}) <= k``, so it suffices that
    ``k / (n + 1) >= 1 - alpha``, i.e. ``k >= (n + 1) * (1 - alpha)``. The
    smallest integer k satisfying this is::

        k = ceil((n + 1) * (1 - alpha))

    which is exactly the ``ceil((n+1)(1-alpha)) / n`` empirical quantile of the
    calibration scores. Using ``k = ceil(n * (1 - alpha))`` instead -- the
    naive empirical quantile -- drops the finite-sample correction and the
    guarantee fails for small n; that off-by-one is the single most common way
    a conformal implementation is quietly wrong.

    Converting back to a confidence threshold: we answer when
    ``s <= s_(k)``, and ``s = 1 - confidence``, hence
    ``t = 1 - s_(k)``.

    Edge case: guarantee not supportable
    ------------------------------------
    When ``ceil((n + 1) * (1 - alpha)) > n`` -- equivalently when
    ``n < (1 - alpha) / alpha``, e.g. fewer than 9 correct items at
    alpha = 0.10 -- no order statistic of the calibration set is large enough
    to carry the guarantee. There is no valid finite threshold, and inventing
    one (``max(scores)``, or 0.0) would be a silent downgrade to an
    unguaranteed heuristic. We instead return `IMPOSSIBLE_THRESHOLD`, a value
    above 1.0 that admits nothing: the system abstains on everything and
    reports zero coverage. That is loud, honest, and trivially satisfies the
    guarantee -- it says *collect more calibration data*, or loosen alpha.
    ``n = 0`` (no calibration data at all) falls under the same rule.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")

    scores = np.array([1.0 - item.confidence for item in cal if item.correct], dtype=float)
    n = scores.size
    if n == 0:
        return IMPOSSIBLE_THRESHOLD

    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return IMPOSSIBLE_THRESHOLD

    # k is 1-indexed over the ascending order statistics.
    s_k = float(np.sort(scores)[k - 1])
    return 1.0 - s_k


def evaluate(test: list[Scored], threshold: float, alpha: float = 0.10) -> ConformalResult:
    """Apply a threshold to a test split and measure what it bought.

    `alpha` is only used to decide `guarantee_held`; the threshold itself is
    whatever `calibrate` (or a baseline) produced. Keeping it a parameter
    means this function can also grade a hand-picked cutoff against the same
    target, which is the comparison that makes the conformal number legible.

    An empty answered set is reported as ``error_on_answered = 0.0`` with
    ``coverage = 0.0``: abstaining on everything makes no errors, which is
    true but useless -- hence coverage is always reported alongside.
    """
    n_total = len(test)
    answered = [item for item in test if item.confidence >= threshold]
    n_answered = len(answered)

    coverage = (n_answered / n_total) if n_total else 0.0
    errors = sum(1 for item in answered if not item.correct)
    error_on_answered = (errors / n_answered) if n_answered else 0.0

    return ConformalResult(
        threshold=threshold,
        coverage=coverage,
        error_on_answered=error_on_answered,
        n_answered=n_answered,
        n_total=n_total,
        guarantee_held=error_on_answered <= alpha,
    )


def risk_coverage_curve(items: list[Scored]) -> tuple[list[float], list[float]]:
    """Sweep the threshold over every observed confidence.

    Returns ``(coverages, risks)`` as parallel lists sorted by coverage
    ascending, where risk is the error rate among the answered items at that
    operating point. Only thresholds that answer at least one item appear --
    risk is undefined on an empty answered set, and padding it with a zero
    would fabricate a free lunch at the left edge of the curve.

    This is the selective-prediction view of the system: it shows the whole
    accuracy/abstention frontier, so the single conformal operating point can
    be read against the alternatives rather than in isolation.
    """
    if not items:
        return [], []

    n_total = len(items)
    conf = np.array([item.confidence for item in items], dtype=float)
    wrong = np.array([not item.correct for item in items], dtype=float)

    # Sort by confidence descending: prefix i of this order is exactly the
    # answered set at threshold conf[i]. A prefix-sum then gives every
    # operating point in one pass instead of re-scanning per threshold.
    order = np.argsort(-conf, kind="stable")
    conf_desc = conf[order]
    err_cum = np.cumsum(wrong[order])

    # Ties: a threshold equal to a repeated confidence admits the whole tied
    # group, so only the LAST index of each tied run is a real operating point.
    n = conf_desc.size
    last_of_run = np.ones(n, dtype=bool)
    last_of_run[:-1] = conf_desc[:-1] != conf_desc[1:]

    idx = np.flatnonzero(last_of_run)
    n_answered = idx + 1
    coverages = (n_answered / n_total).tolist()
    risks = (err_cum[idx] / n_answered).tolist()
    # idx is ascending, so coverage is already ascending.
    return coverages, risks


def aurc(items: list[Scored]) -> float:
    """Area under the risk-coverage curve. LOWER IS BETTER.

    AURC integrates risk over coverage by the trapezoidal rule. A ranker that
    sorts every correct extraction above every incorrect one keeps risk at
    zero until the correct items are exhausted, so its area is small; a ranker
    whose confidences are noise sits at the base error rate for the whole
    sweep, so its area is roughly ``base_error * coverage_range``. It is
    therefore a single threshold-free number for "do these confidences rank
    anything at all", and is the right metric to watch when comparing
    extractors -- unlike accuracy, it cannot be gamed by moving a cutoff.

    Returns 0.0 for fewer than two curve points (no interval to integrate).
    """
    coverages, risks = risk_coverage_curve(items)
    if len(coverages) < 2:
        return 0.0
    return float(np.trapezoid(np.asarray(risks), np.asarray(coverages)))


def always_answer_baseline(items: list[Scored], alpha: float = 0.10) -> ConformalResult:
    """The no-abstention comparison: answer everything, at full coverage.

    Without this, "10% error on answered fields" is unreadable -- it could be
    an improvement or a regression. The baseline pins down what the raw
    extractor's error rate is at coverage 1.0, so the conformal result can be
    stated as the trade it actually is: *this much coverage given up, for this
    much error removed.*

    Threshold 0.0 admits every item, since confidences live in [0, 1].
    """
    return evaluate(items, threshold=0.0, alpha=alpha)
