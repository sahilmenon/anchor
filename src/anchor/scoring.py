"""Scoring and failure taxonomy for Anchor.

Turning an :class:`~anchor.schema.Extraction` into a number is the easy half.
The half that matters for a credit fund is *why* an extraction was wrong, so
this module's headline output is a taxonomy of failure modes rather than a
single accuracy figure.

The taxonomy borrows its member names from prior art in scientific-document
extraction evaluation (FAIRmat's extract-eval), which separates a wrong value
from a missing value from an invented value. We add two members that matter
specifically for evidence-anchored financial extraction:

* ``CORRECT_ABSTENTION`` -- the document genuinely does not state the item and
  the extractor said so. This is a *win*. An extractor that never abstains
  cannot be trusted on documents where the answer is absent, which is most of
  them.
* ``UNGROUNDED`` -- the citation did not check out. See the precedence note on
  :func:`score_field`.

Zero-denominator convention
---------------------------
Every rate in :class:`Report` and :class:`Stats` returns **``None``** when its
denominator is zero, never ``0.0``. "We never asked" and "we asked and it
always failed" are different facts, and collapsing them to ``0.0`` would let an
empty ambiguous-subset masquerade as perfect calibration in a report table.
Counts (``taxonomy``, ``n_*``) are plain integers and are ``0`` when empty.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from dataclasses import field as dfield
from enum import Enum

from anchor.schema import ExtractedField, Extraction, GoldenField, GoldenRecord, LineItem

__all__ = [
    "Outcome",
    "FieldScore",
    "DocumentScore",
    "Stats",
    "Report",
    "numbers_match",
    "score_field",
    "score_document",
    "aggregate",
]


class Outcome(str, Enum):  # noqa: UP042 - str-Enum matches anchor.schema and stays JSON-serialisable
    """The mutually exclusive verdicts a single field can receive.

    Exactly one is assigned per golden field, so the counts form a partition of
    the evaluated fields and the taxonomy always sums to the field total.
    """

    CORRECT = "correct"
    """Both present and the values agree within tolerance, with a citation that
    checked out (or no verification was run)."""

    MISMATCH = "mismatch"
    """Both present, values differ. A sourcing error: the extractor read the
    wrong row, the wrong period, or the wrong scale."""

    OMISSION = "omission"
    """Truth has a value, the extractor abstained. Costly but honest -- an
    analyst can go find it."""

    HALLUCINATION = "hallucination"
    """Truth says the document does not state this item (golden value is
    ``None``) and the extractor produced a number anyway. The worst failure in
    the taxonomy: nothing downstream flags it, and it looks like an answer."""

    CORRECT_ABSTENTION = "correct_abstention"
    """Truth is ``None`` and the extractor abstained. A win, not a failure.
    Reported separately so it never dilutes the accuracy denominator."""

    UNGROUNDED = "ungrounded"
    """The extractor produced a value but ``grounded is False`` -- the quote it
    cited could not be located on the page it cited."""


# ---------------------------------------------------------------------------
# Value comparison
# ---------------------------------------------------------------------------


def numbers_match(
    a: float | None,
    b: float | None,
    rel_tol: float = 0.01,
    abs_tol: float = 1.0,
) -> bool:
    """Compare two financial figures under *both* a relative and absolute tolerance.

    Either tolerance holding is enough. Both are needed because financial
    figures are restated and rounded at inconsistent magnitudes: a $412.7m
    revenue line reprinted as $413m is the same fact (relative tolerance saves
    it), while a $0 cash balance reported as $0.40 is also the same fact but has
    unbounded relative error (absolute tolerance saves it). Defaults are 1%
    and one base currency unit.

    ``None`` on either side means "no value", and no value never matches --
    including ``None`` against ``None``. Abstention agreement is a *taxonomy*
    judgement made in :func:`score_field`, not a numeric one; letting
    ``None == None`` return ``True`` here would silently score abstentions as
    CORRECT and inflate accuracy.
    """
    if a is None or b is None:
        return False
    diff = abs(a - b)
    if diff <= abs_tol:
        return True
    scale = max(abs(a), abs(b))
    return diff <= rel_tol * scale


# ---------------------------------------------------------------------------
# Per-field and per-document results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldScore:
    """The verdict on one line item, carrying enough context to explain itself."""

    name: LineItem
    outcome: Outcome
    expected: float | None
    actual: float | None
    ambiguous: bool
    grounded: bool | None

    @property
    def abstained(self) -> bool:
        """True when the extractor produced no value for this field."""
        return self.actual is None

    @property
    def answerable(self) -> bool:
        """True when ground truth has a value, i.e. the document states it."""
        return self.expected is not None

    @property
    def should_have_abstained(self) -> bool:
        """True when abstaining was the defensible call.

        Either the document does not state the item, or labelling it required a
        judgement call (``GoldenField.ambiguous``). Ambiguous fields count here
        even though they carry a value: a calibrated extractor is allowed to
        decline a coin-flip rather than guess it.
        """
        return self.expected is None or self.ambiguous


def score_field(extracted: ExtractedField | None, golden: GoldenField) -> FieldScore:
    """Assign exactly one :class:`Outcome` to one field.

    Precedence -- grounding is checked **before** value comparison, so a
    numerically correct answer with a citation that failed verification scores
    ``UNGROUNDED``, not ``CORRECT``. This is deliberate and it is the whole
    point of the harness: a credit fund cannot put an unverifiable number in an
    investment committee memo, so for our purposes it is not an answer. Ranking
    it as a success would reward an extractor for being lucky, and would let a
    model that fabricates citations post the same headline score as one that
    reads the document.

    Order of checks:

    1. No value from the extractor -> ``CORRECT_ABSTENTION`` if truth is also
       ``None``, else ``OMISSION``. Grounding is not consulted: there is no
       claim to ground.
    2. ``grounded is False`` -> ``UNGROUNDED`` (see above). ``grounded is None``
       means the verifier never ran and is treated as "not disproven".
    3. Truth is ``None`` -> ``HALLUCINATION``.
    4. Values compared with :func:`numbers_match` -> ``CORRECT`` / ``MISMATCH``.

    A field absent from the extraction entirely (``extracted is None``) is
    indistinguishable from one returned with ``value=None``: both are an
    ``OMISSION`` against a stated figure.
    """
    expected = golden.value
    actual = extracted.scaled_value if extracted is not None else None
    grounded = extracted.grounded if extracted is not None else None

    if actual is None:
        outcome = Outcome.CORRECT_ABSTENTION if expected is None else Outcome.OMISSION
    elif grounded is False:
        outcome = Outcome.UNGROUNDED
    elif expected is None:
        outcome = Outcome.HALLUCINATION
    elif numbers_match(actual, expected):
        outcome = Outcome.CORRECT
    else:
        outcome = Outcome.MISMATCH

    return FieldScore(
        name=golden.name,
        outcome=outcome,
        expected=expected,
        actual=actual,
        ambiguous=golden.ambiguous,
        grounded=grounded,
    )


@dataclass
class DocumentScore:
    """All field verdicts for one document, plus that document's cost."""

    doc_id: str
    extractor: str
    fields: list[FieldScore] = dfield(default_factory=list)
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def counts(self) -> Counter[Outcome]:
        """Per-document taxonomy counts. Sums to ``len(self.fields)``."""
        return Counter(f.outcome for f in self.fields)

    @property
    def n_correct(self) -> int:
        return self.counts[Outcome.CORRECT]

    @property
    def n_fields(self) -> int:
        return len(self.fields)


def score_document(extraction: Extraction, golden: GoldenRecord) -> DocumentScore:
    """Score one extraction against its golden record.

    The golden record drives the iteration: ground truth defines the question
    set, so an extractor cannot improve its score by returning fewer fields, and
    fields it invents outside the golden record are ignored rather than counted
    as hallucinations (the gold label for them simply does not exist).
    """
    scores = [score_field(extraction.get(g.name), g) for g in golden.fields]
    return DocumentScore(
        doc_id=golden.doc_id,
        extractor=extraction.extractor,
        fields=scores,
        latency_s=extraction.latency_s,
        input_tokens=extraction.input_tokens,
        output_tokens=extraction.output_tokens,
        cost_usd=extraction.cost_usd,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _rate(numerator: int, denominator: int) -> float | None:
    """Divide, or return ``None`` when the denominator is zero.

    See the module docstring: an undefined rate is reported as undefined rather
    than as zero, so an empty slice cannot read as a perfect or a failing score.
    """
    if denominator == 0:
        return None
    return numerator / denominator


@dataclass
class Stats:
    """Rates over an arbitrary set of :class:`FieldScore`.

    Computed once for the whole run, again for the ambiguous subset, and again
    per line item, so the three are directly comparable line by line.

    The counter fields are populated by :func:`_build_stats`; the rates are
    properties so a `Stats` stays a plain bag of integers that is trivial to
    serialise, sum, or diff between two runs.
    """

    n_fields: int = 0
    n_answerable: int = 0
    """Fields where ground truth has a value -- the accuracy denominator."""
    n_unanswerable: int = 0
    """Fields the document does not state -- the hallucination denominator."""
    n_attempted: int = 0
    """Fields where the extractor produced a value (answerable or not)."""
    n_abstained: int = 0
    n_attempted_answerable: int = 0
    """Attempts on fields that actually had an answer -- the coverage numerator."""
    n_grounded_true: int = 0
    n_justified_abstentions: int = 0
    """Abstentions on fields that were unanswerable or ambiguous."""
    taxonomy: Counter[Outcome] = dfield(default_factory=Counter)

    @property
    def accuracy(self) -> float | None:
        """CORRECT over fields the document actually states.

        Unanswerable fields are excluded from the denominator entirely so that
        correct abstentions neither help nor hurt accuracy; they are measured by
        ``abstention_precision`` instead.
        """
        return _rate(self.taxonomy[Outcome.CORRECT], self.n_answerable)

    @property
    def coverage(self) -> float | None:
        """Fraction of answerable fields the extractor attempted.

        Counts attempts, right or wrong, and pairs with ``accuracy`` to separate
        "did not try" from "tried and missed".
        """
        return _rate(self.n_attempted_answerable, self.n_answerable)

    @property
    def grounding_rate(self) -> float | None:
        """Fields with ``grounded is True`` over fields where a value was produced.

        Abstentions are excluded: there is no citation to verify. ``grounded is
        None`` (verifier never ran) counts against the rate, because an
        unverified citation is not a verified one.
        """
        return _rate(self.n_grounded_true, self.n_attempted)

    @property
    def hallucination_rate(self) -> float | None:
        """HALLUCINATION over fields the document does not state.

        Note the interaction with precedence: an invented value whose citation
        also failed verification scores UNGROUNDED, so it lands in
        ``ungrounded_rate`` rather than here. Read the two together.
        """
        return _rate(self.taxonomy[Outcome.HALLUCINATION], self.n_unanswerable)

    @property
    def omission_rate(self) -> float | None:
        """OMISSION over answerable fields -- the complement of ``coverage``."""
        return _rate(self.taxonomy[Outcome.OMISSION], self.n_answerable)

    @property
    def ungrounded_rate(self) -> float | None:
        """UNGROUNDED over fields where a value was produced."""
        return _rate(self.taxonomy[Outcome.UNGROUNDED], self.n_attempted)

    @property
    def abstention_precision(self) -> float | None:
        """Of the fields abstained on, the fraction that *should* have been.

        "Should have" means the document does not state the item, or the label
        was ambiguous. This is the key calibration signal: accuracy rewards an
        extractor for answering, and only this metric punishes it for answering
        when it should not have. An extractor that never abstains has no value
        here (``None``), which is itself the finding.
        """
        return _rate(self.n_justified_abstentions, self.n_abstained)


def _build_stats(scores: list[FieldScore]) -> Stats:
    """Fold a list of field scores into a :class:`Stats`."""
    stats = Stats(
        n_fields=len(scores),
        taxonomy=Counter(s.outcome for s in scores),
    )
    for s in scores:
        if s.answerable:
            stats.n_answerable += 1
        else:
            stats.n_unanswerable += 1
        if s.abstained:
            stats.n_abstained += 1
            if s.should_have_abstained:
                stats.n_justified_abstentions += 1
        else:
            stats.n_attempted += 1
            if s.grounded is True:
                stats.n_grounded_true += 1
            if s.answerable:
                stats.n_attempted_answerable += 1
    return stats


@dataclass
class Report:
    """Run-level results: headline rates, the taxonomy, and the cost of getting them."""

    overall: Stats = dfield(default_factory=Stats)
    on_ambiguous: Stats = dfield(default_factory=Stats)
    """The same statistics restricted to fields where ``golden.ambiguous`` is
    True -- the abstention test set. A well-calibrated extractor should show
    markedly higher ``abstention_precision`` here than overall."""
    per_field: dict[LineItem, Stats] = dfield(default_factory=dict)
    n_documents: int = 0
    total_cost_usd: float = 0.0
    total_latency_s: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    # -- headline rates, delegated to `overall` so the report reads flat -----
    @property
    def accuracy(self) -> float | None:
        return self.overall.accuracy

    @property
    def coverage(self) -> float | None:
        return self.overall.coverage

    @property
    def grounding_rate(self) -> float | None:
        return self.overall.grounding_rate

    @property
    def hallucination_rate(self) -> float | None:
        return self.overall.hallucination_rate

    @property
    def omission_rate(self) -> float | None:
        return self.overall.omission_rate

    @property
    def ungrounded_rate(self) -> float | None:
        return self.overall.ungrounded_rate

    @property
    def abstention_precision(self) -> float | None:
        return self.overall.abstention_precision

    @property
    def taxonomy(self) -> Counter[Outcome]:
        """Counter of :class:`Outcome` over every field scored -- the deliverable."""
        return self.overall.taxonomy

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def n_fields(self) -> int:
        return self.overall.n_fields


def aggregate(doc_scores: list[DocumentScore]) -> Report:
    """Fold per-document scores into a run-level :class:`Report`.

    Aggregation is micro-averaged (every field weighs the same) rather than
    macro-averaged over documents, because documents carry different numbers of
    labelled fields and a short document should not outweigh a long one per
    field. Passing an empty list yields a report whose counts are zero and whose
    rates are all ``None``.
    """
    all_scores: list[FieldScore] = [f for d in doc_scores for f in d.fields]

    by_name: dict[LineItem, list[FieldScore]] = {}
    for s in all_scores:
        by_name.setdefault(s.name, []).append(s)

    return Report(
        overall=_build_stats(all_scores),
        on_ambiguous=_build_stats([s for s in all_scores if s.ambiguous]),
        per_field={name: _build_stats(group) for name, group in by_name.items()},
        n_documents=len(doc_scores),
        total_cost_usd=sum(d.cost_usd for d in doc_scores),
        total_latency_s=sum(d.latency_s for d in doc_scores),
        total_input_tokens=sum(d.input_tokens for d in doc_scores),
        total_output_tokens=sum(d.output_tokens for d in doc_scores),
    )
