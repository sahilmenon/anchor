"""The end-to-end run: extract, verify, score, derive ratios, calibrate.

This is the module that turns the pieces into a pipeline. One `RunResult`
holds everything a reader needs to audit a run -- not just the headline rates
but the verified extraction behind every field, so a number in the report can
be traced to the quote it came from. That is the whole premise of the harness,
and it only holds if the evidence survives into the artifact.

Three things happen here that do not happen in any single scoring module:

1. **Ratios are compared, not just computed.** `anchor.ratios` derives leverage
   and DSCR from an extraction; it will equally derive them from ground truth.
   Running both and comparing answers the question a credit team actually asks
   -- *would this pipeline have reached the analyst's conclusion?* -- which is
   not the same question as per-field accuracy. A run can score 100% on cash
   and still produce a DSCR nobody should act on.

2. **Abstention is scored as a selective-prediction problem.** Each attempted
   field becomes a `conformal.Scored` (its self-reported confidence, and
   whether it was CORRECT), so the run reports AURC and a conformal operating
   point alongside the flat rates.

3. **The calibration split is taken by document, deterministically.** Fields
   from one document are not independent -- they share a units declaration, a
   text layer and a layout -- so splitting mid-document would leak. Documents
   are assigned to calibration and test by their position in the sorted corpus,
   which also means two runs of the same corpus produce the same split and the
   gate stays reproducible.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from dataclasses import field as dfield
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from anchor import conformal
from anchor.corpus import Corpus, CorpusItem
from anchor.extractors.base import Document, Extractor
from anchor.ratios import RatioResult, compute_all
from anchor.schema import Extraction, GoldenRecord, LineItem
from anchor.scoring import (
    DocumentScore,
    Outcome,
    Report,
    Stats,
    aggregate,
    numbers_match,
    score_document,
)
from anchor.verify import verify_extraction

__all__ = [
    "DEFAULT_ALPHA",
    "DocumentResult",
    "METRIC_NAMES",
    "RatioComparison",
    "RatioOutcome",
    "RunResult",
    "SelectiveSummary",
    "inspect_document",
    "load_run",
    "metric_value",
    "run_extractor",
]

#: Default conformal miss rate. 0.10 targets "keep at least 90% of the answers
#: we should have given".
DEFAULT_ALPHA = 0.10


# ---------------------------------------------------------------------------
# Ratio comparison
# ---------------------------------------------------------------------------


class RatioOutcome(str, Enum):  # noqa: UP042 - matches the str-Enum convention in anchor.schema
    """How a ratio derived from an extraction compares to one derived from truth."""

    AGREE = "agree"
    """Both sides computed a value and the values match."""

    DISAGREE = "disagree"
    """Both sides computed a value and they differ. The extraction would have
    put a wrong multiple in front of a credit committee."""

    BOTH_ABSTAINED = "both_abstained"
    """Neither side could compute it. A win: the document does not support the
    ratio and the pipeline said so."""

    EXTRACTION_ABSTAINED = "extraction_abstained"
    """Truth supports the ratio, the extraction could not reach it. Costly but
    safe -- an analyst can go and find the missing input."""

    GOLDEN_ABSTAINED = "golden_abstained"
    """The extraction produced a ratio the document does not support. The
    dangerous one: it looks like an answer and nothing downstream flags it."""


@dataclass(frozen=True)
class RatioComparison:
    """One ratio, derived twice, with the verdict and both refusal reasons."""

    name: str
    extracted: float | None
    expected: float | None
    outcome: RatioOutcome
    extracted_reason: str = ""
    expected_reason: str = ""
    inputs_used: list[LineItem] = dfield(default_factory=list)

    @property
    def safe(self) -> bool:
        """True when the pipeline did not assert something truth cannot support.

        Agreement and joint abstention are both safe outcomes; so is abstaining
        where truth has an answer, because a missing ratio is visibly missing.
        The two unsafe outcomes are the ones that produce a confident number.
        """
        return self.outcome in (
            RatioOutcome.AGREE,
            RatioOutcome.BOTH_ABSTAINED,
            RatioOutcome.EXTRACTION_ABSTAINED,
        )


def _golden_mapping(golden: GoldenRecord) -> dict[LineItem, float | None]:
    """Ground truth as a plain mapping the ratio layer accepts."""
    return {f.name: f.value for f in golden.fields}


def compare_ratios(extraction: Extraction, golden: GoldenRecord) -> list[RatioComparison]:
    """Derive every ratio from both sides and classify the disagreement."""
    got = compute_all(extraction)
    want = compute_all(_golden_mapping(golden))

    out: list[RatioComparison] = []
    for name in got:
        g: RatioResult = got[name]
        w: RatioResult = want[name]
        if g.abstained and w.abstained:
            outcome = RatioOutcome.BOTH_ABSTAINED
        elif g.abstained:
            outcome = RatioOutcome.EXTRACTION_ABSTAINED
        elif w.abstained:
            outcome = RatioOutcome.GOLDEN_ABSTAINED
        elif numbers_match(g.value, w.value, rel_tol=0.01, abs_tol=0.0):
            outcome = RatioOutcome.AGREE
        else:
            outcome = RatioOutcome.DISAGREE
        out.append(
            RatioComparison(
                name=name,
                extracted=g.value,
                expected=w.value,
                outcome=outcome,
                extracted_reason=g.reason,
                expected_reason=w.reason,
                inputs_used=list(g.inputs_used),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Selective prediction summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectiveSummary:
    """What the extractor's confidences are worth, as a ranking and as a gate.

    ``aurc`` is the threshold-free number: it asks whether the confidences rank
    correct extractions above incorrect ones at all, and cannot be improved by
    moving a cutoff. The conformal fields are one operating point on that same
    curve, derived rather than hand-picked.
    """

    n_items: int
    n_calibration: int
    n_calibration_correct: int
    n_test: int
    alpha: float
    aurc: float
    threshold: float
    threshold_supported: bool
    """False when the calibration split was too small to carry the guarantee at
    this alpha, in which case the threshold admits nothing. See
    `conformal.calibrate`."""
    coverage: float
    error_on_answered: float
    guarantee_held: bool
    baseline_error: float
    """Error rate over every scored item with no abstention at all -- what the
    conformal operating point is trading against, on the same population AURC
    is measured over."""

    n_distinct_confidences: int = 0
    """How many distinct confidence values the extractor actually produced."""

    curve_coverage: list[float] = dfield(default_factory=list)
    curve_risk: list[float] = dfield(default_factory=list)
    """The full risk-coverage frontier, carried so a reader can see the single
    conformal operating point against every alternative rather than in
    isolation. Parallel lists, sorted by coverage ascending."""

    @property
    def aurc_is_informative(self) -> bool:
        """False when the confidence signal is too flat for AURC to mean much.

        AURC integrates risk over the observed coverage range, so an extractor
        that reports one confidence for nearly every field produces a curve one
        or two points wide and an area that mostly measures that width. The
        number is still correct; it just stops being a statement about ranking.
        Worth knowing before putting AURC in a thresholds file, where a flat
        ranker would sail under a ceiling it never really cleared.
        """
        return self.n_distinct_confidences >= 4


def _split_by_document(
    per_doc: list[tuple[str, list[conformal.Scored]]],
) -> tuple[list[conformal.Scored], list[conformal.Scored]]:
    """Alternate whole documents into calibration and test.

    Fields within a document share a units declaration, a layout and a text
    layer, so they are not exchangeable with each other in the way conformal
    needs. Splitting at the document boundary keeps the exchangeability
    assumption on the unit it actually holds for, and alternating by position
    over a sorted corpus makes the split reproducible.
    """
    cal: list[conformal.Scored] = []
    test: list[conformal.Scored] = []
    for i, (_doc_id, items) in enumerate(per_doc):
        (cal if i % 2 == 0 else test).extend(items)
    return cal, test


def _scored_items(extraction: Extraction, doc_score: DocumentScore) -> list[conformal.Scored]:
    """Attempted fields as (confidence, correct) pairs.

    Abstentions are excluded: they carry no confidence to calibrate on, and
    selective prediction is a question about the answers a system gives.
    """
    items: list[conformal.Scored] = []
    for fs in doc_score.fields:
        if fs.abstained:
            continue
        fld = extraction.get(fs.name)
        if fld is None:  # pragma: no cover - an attempted field exists by construction
            continue
        items.append(
            conformal.Scored(
                confidence=float(fld.confidence),
                correct=fs.outcome is Outcome.CORRECT,
            )
        )
    return items


def _summarise_selective(
    per_doc: list[tuple[str, list[conformal.Scored]]], alpha: float
) -> SelectiveSummary:
    all_items = [s for _, items in per_doc for s in items]
    cal, test = _split_by_document(per_doc)

    curve_coverage, curve_risk = conformal.risk_coverage_curve(all_items)

    threshold = conformal.calibrate(cal, alpha=alpha)
    supported = threshold < conformal.IMPOSSIBLE_THRESHOLD

    # The operating point is measured on the test split only -- that is the
    # whole point of split conformal, and evaluating a threshold on the data it
    # was derived from would report a number the guarantee does not cover.
    result = conformal.evaluate(test, threshold, alpha=alpha)

    # The baseline is measured on *all* items, matching AURC's denominator, so
    # the two threshold-free figures printed next to each other describe the
    # same population. Comparing an all-items AURC against a test-split base
    # error would make the trade look better or worse than it is by an accident
    # of the split.
    baseline = conformal.always_answer_baseline(all_items, alpha=alpha)

    return SelectiveSummary(
        n_items=len(all_items),
        n_calibration=len(cal),
        n_calibration_correct=sum(1 for s in cal if s.correct),
        n_test=len(test),
        alpha=alpha,
        aurc=conformal.aurc(all_items),
        threshold=threshold,
        threshold_supported=supported,
        coverage=result.coverage,
        error_on_answered=result.error_on_answered,
        guarantee_held=result.guarantee_held,
        baseline_error=baseline.error_on_answered,
        n_distinct_confidences=len({s.confidence for s in all_items}),
        curve_coverage=curve_coverage,
        curve_risk=curve_risk,
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


@dataclass
class DocumentResult:
    """Everything produced for one document, evidence included."""

    doc_id: str
    score: DocumentScore
    extraction: Extraction
    ratios: list[RatioComparison]
    golden: GoldenRecord
    n_pages: int
    error: str = ""
    """Non-empty when the document could not be read at all."""


@dataclass
class RunResult:
    """One scored run of one extractor over one corpus."""

    run_id: str
    extractor: str
    corpus_root: str
    created_at: str
    alpha: float
    verified: bool
    documents: list[DocumentResult] = dfield(default_factory=list)
    report: Report = dfield(default_factory=Report)
    selective: SelectiveSummary | None = None
    skipped: list[str] = dfield(default_factory=list)
    """Labelled documents with no source file. Reported, never silently dropped."""

    # -- derived ----------------------------------------------------------
    @property
    def ratio_counts(self) -> Counter[RatioOutcome]:
        return Counter(r.outcome for d in self.documents for r in d.ratios)

    @property
    def n_ratios(self) -> int:
        return sum(len(d.ratios) for d in self.documents)

    @property
    def latency_p50(self) -> float | None:
        """Median per-document latency.

        The median rather than the mean because one pathological document --
        a scan that takes four retries -- should not set the number a reader
        uses to estimate a batch.
        """
        times = sorted(d.score.latency_s for d in self.documents)
        if not times:
            return None
        mid = len(times) // 2
        if len(times) % 2:
            return times[mid]
        return (times[mid - 1] + times[mid]) / 2

    @property
    def cost_per_document(self) -> float | None:
        n = len(self.documents)
        return None if n == 0 else self.report.total_cost_usd / n

    @property
    def ratio_agreement(self) -> float | None:
        """Ratios where both sides computed the same value, or both refused."""
        total = self.n_ratios
        if total == 0:
            return None
        counts = self.ratio_counts
        agreed = counts[RatioOutcome.AGREE] + counts[RatioOutcome.BOTH_ABSTAINED]
        return agreed / total

    @property
    def ratio_safety(self) -> float | None:
        """Ratios where the pipeline did not assert an unsupported number."""
        total = self.n_ratios
        if total == 0:
            return None
        safe = sum(1 for d in self.documents for r in d.ratios if r.safe)
        return safe / total


def run_extractor(
    corpus: Corpus,
    extractor: Extractor,
    *,
    verify: bool = True,
    alpha: float = DEFAULT_ALPHA,
    run_id: str | None = None,
) -> RunResult:
    """Score `extractor` over every resolvable document in `corpus`.

    `verify=False` skips quote checking, which leaves every `grounded` as None.
    That is a diagnostic mode, not a faster one: scoring treats an unverified
    citation as "not disproven", so accuracy goes up and the UNGROUNDED bucket
    empties. It exists so a reader can see exactly how much of the headline
    number the verifier is holding back, and the flag is recorded in the run so
    the two are never confused.

    Documents whose labels have no source file are listed in `skipped` and
    excluded from every rate. Reading a rate without reading `skipped` is how a
    partial corpus gets published as a whole one.
    """
    created = datetime.now(UTC)
    stamp = created.strftime("%Y%m%dT%H%M%S")
    rid = run_id or f"{stamp}Z-{_name(extractor)}"

    results: list[DocumentResult] = []
    skipped: list[str] = []
    per_doc_scored: list[tuple[str, list[conformal.Scored]]] = []

    for item in corpus.items:
        if not item.resolvable:
            skipped.append(item.doc_id)
            continue
        results.append(_run_one(item, extractor, verify=verify))

    for r in results:
        per_doc_scored.append((r.doc_id, _scored_items(r.extraction, r.score)))

    return RunResult(
        run_id=rid,
        extractor=getattr(extractor, "name", "extractor"),
        corpus_root=str(corpus.root),
        created_at=created.isoformat(timespec="seconds"),
        alpha=alpha,
        verified=verify,
        documents=results,
        report=aggregate([r.score for r in results]),
        selective=_summarise_selective(per_doc_scored, alpha),
        skipped=skipped,
    )


def _name(extractor: Extractor) -> str:
    return getattr(extractor, "name", "extractor")


def _extract(item: CorpusItem, doc: Document, extractor: Extractor) -> Extraction:
    """Run an extractor over one item by the most faithful route available.

    `extract_document` is preferred wherever an extractor offers it, because it
    takes the same parsed `Document` the verifier will check quotes against --
    handing the extractor a path and the verifier a separate parse would let a
    quote fail verification for no reason but two different text layers.

    An extractor that only accepts a path can still run, but only on an item
    backed by a real PDF. Asking it to open a page-text JSON file would have it
    fail silently into a wall of abstentions that looks like a bad extractor
    rather than a corpus it cannot read, so that case is an explicit error.
    """
    if hasattr(extractor, "extract_document"):
        return extractor.extract_document(doc, item.doc_id)  # type: ignore[attr-defined]
    if item.kind != "pdf" or item.source is None:
        raise TypeError(
            f"{_name(extractor)} only accepts a PDF path, but {item.doc_id} is "
            f"backed by {item.kind}. Give the extractor an extract_document method "
            f"or add corpus/pdfs/{item.doc_id}.pdf."
        )
    return extractor.extract(item.source, item.doc_id)


def _run_one(item: CorpusItem, extractor: Extractor, *, verify: bool) -> DocumentResult:
    """Extract, verify and score a single document.

    A document that cannot be parsed yields a fully abstained extraction with
    the error recorded, rather than raising. One unreadable file in a corpus of
    fifteen should cost that document's fields, not the whole run.
    """
    error = ""
    try:
        doc = item.document()
    except Exception as exc:
        doc = Document.from_pages(item.doc_id, [])
        error = str(exc)

    try:
        extraction = _extract(item, doc, extractor)
    except Exception as exc:
        extraction = Extraction(doc_id=item.doc_id, fields=[], extractor=_name(extractor))
        error = error or str(exc)

    if verify:
        extraction = verify_extraction(extraction, doc)

    return DocumentResult(
        doc_id=item.doc_id,
        score=score_document(extraction, item.golden),
        extraction=extraction,
        ratios=compare_ratios(extraction, item.golden),
        golden=item.golden,
        n_pages=doc.n_pages,
        error=error,
    )


# ---------------------------------------------------------------------------
# Inspection: one document, no answer key
# ---------------------------------------------------------------------------


def inspect_document(
    document: Document,
    extractor: Extractor,
    *,
    verify: bool = True,
    golden: GoldenRecord | None = None,
) -> dict[str, Any]:
    """Extract and verify one document without scoring it.

    Scoring needs a label. Verification does not, and that asymmetry is what
    makes this useful: for a document nobody has labelled yet, Anchor can still
    say what it read, which page and quote it read it from, whether that
    citation checks out, and which ratios fall out. What it cannot say is
    whether any of it is right.

    So this returns no accuracy, no taxonomy and no outcome per field, and the
    interface shows none. Presenting an unlabelled document beside a
    percentage would invent an answer key out of the extractor's own output,
    which is the exact failure the harness exists to catch.

    `golden` is optional and, when passed, carries the existing label alongside
    each field so an editor can show what was recorded last time next to what
    the extractor reads now.
    """
    extraction = extractor.extract_document(document, document.doc_id)  # type: ignore[attr-defined]
    if verify:
        extraction = verify_extraction(extraction, document)

    ratios = compute_all(extraction)

    fields: list[dict[str, Any]] = []
    for item in LineItem:
        fld = extraction.get(item)
        g = None if golden is None else golden.get(item)
        fields.append(
            {
                "name": item.value,
                "value": None if fld is None else fld.scaled_value,
                "raw_value": None if fld is None else fld.value,
                "unit_scale": 1.0 if fld is None else fld.unit_scale,
                "currency": None if fld is None else fld.currency,
                "confidence": None if fld is None else fld.confidence,
                "grounded": None if fld is None else fld.grounded,
                "page": None if fld is None or fld.evidence is None else fld.evidence.page,
                "quote": None if fld is None or fld.evidence is None else fld.evidence.quote,
                "golden_value": None if g is None else g.value,
                "golden_stated": None if g is None else g.value is not None,
                "golden_ambiguous": False if g is None else g.ambiguous,
                "golden_note": "" if g is None else g.note,
            }
        )

    return {
        "doc_id": document.doc_id,
        "extractor": _name(extractor),
        "n_pages": document.n_pages,
        "verified": verify,
        "labelled": golden is not None,
        "source": "" if golden is None else golden.source,
        "scanned": False if golden is None else golden.scanned,
        "fields": fields,
        "ratios": [
            {
                "name": r.name,
                "value": r.value,
                "abstained": r.abstained,
                "reason": r.reason,
                "inputs_used": [i.value for i in r.inputs_used],
            }
            for r in ratios.values()
        ],
    }


# ---------------------------------------------------------------------------
# Metric lookup -- the namespace the gate and the CLI address metrics by
# ---------------------------------------------------------------------------

#: Rate names available on a `Stats`.
_STATS_METRICS = (
    "accuracy",
    "coverage",
    "grounding_rate",
    "hallucination_rate",
    "omission_rate",
    "ungrounded_rate",
    "abstention_precision",
)

#: Every addressable metric name, for `--help` and for validating a thresholds
#: file before it is used. Field-scoped names (``field.<line_item>.<metric>``)
#: are accepted too but not enumerated, since they depend on the corpus.
METRIC_NAMES = (
    *_STATS_METRICS,
    *(f"ambiguous.{m}" for m in _STATS_METRICS),
    "ratio.agreement",
    "ratio.safety",
    "selective.aurc",
    "selective.coverage",
    "selective.error_on_answered",
    "selective.baseline_error",
)


class UnknownMetric(KeyError):
    """Raised when a thresholds file names a metric that does not exist."""


def _stats_metric(stats: Stats, name: str) -> float | None:
    if name not in _STATS_METRICS:
        raise UnknownMetric(name)
    return getattr(stats, name)


def metric_value(run: RunResult, name: str) -> float | None:
    """Look one metric up on a run by its dotted name.

    Returning ``None`` is meaningful and is not an error: it means the metric's
    denominator was zero (see the zero-denominator convention in
    `anchor.scoring`). An unknown *name*, by contrast, raises -- a thresholds
    file with a typo in it must fail loudly rather than gate on nothing.
    """
    if name.startswith("ambiguous."):
        return _stats_metric(run.report.on_ambiguous, name.split(".", 1)[1])

    if name.startswith("field."):
        _, item_name, metric = name.split(".", 2)
        try:
            line_item = LineItem(item_name)
        except ValueError as exc:
            raise UnknownMetric(name) from exc
        stats = run.report.per_field.get(line_item)
        return None if stats is None else _stats_metric(stats, metric)

    if name == "ratio.agreement":
        return run.ratio_agreement
    if name == "ratio.safety":
        return run.ratio_safety

    if name.startswith("selective."):
        if run.selective is None:
            return None
        attr = name.split(".", 1)[1]
        if attr not in ("aurc", "coverage", "error_on_answered", "baseline_error"):
            raise UnknownMetric(name)
        return float(getattr(run.selective, attr))

    return _stats_metric(run.report.overall, name)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _stats_dict(stats: Stats) -> dict[str, Any]:
    return {
        "n_fields": stats.n_fields,
        "n_answerable": stats.n_answerable,
        "n_unanswerable": stats.n_unanswerable,
        "n_attempted": stats.n_attempted,
        "n_abstained": stats.n_abstained,
        "taxonomy": {o.value: stats.taxonomy[o] for o in Outcome},
        **{m: _stats_metric(stats, m) for m in _STATS_METRICS},
    }


def _run_dict(run: RunResult) -> dict[str, Any]:
    sel = run.selective
    return {
        "run_id": run.run_id,
        "extractor": run.extractor,
        "corpus_root": run.corpus_root,
        "created_at": run.created_at,
        "alpha": run.alpha,
        "verified": run.verified,
        "skipped": list(run.skipped),
        "report": {
            "n_documents": run.report.n_documents,
            "n_fields": run.report.n_fields,
            "total_cost_usd": run.report.total_cost_usd,
            "total_latency_s": run.report.total_latency_s,
            "total_input_tokens": run.report.total_input_tokens,
            "total_output_tokens": run.report.total_output_tokens,
            "overall": _stats_dict(run.report.overall),
            "on_ambiguous": _stats_dict(run.report.on_ambiguous),
            "per_field": {
                name.value: _stats_dict(s) for name, s in run.report.per_field.items()
            },
        },
        "latency_p50": run.latency_p50,
        "cost_per_document": run.cost_per_document,
        "ratios": {
            "agreement": run.ratio_agreement,
            "safety": run.ratio_safety,
            "counts": {o.value: run.ratio_counts[o] for o in RatioOutcome},
        },
        "selective": None
        if sel is None
        else {
            "n_items": sel.n_items,
            "n_calibration": sel.n_calibration,
            "n_calibration_correct": sel.n_calibration_correct,
            "n_test": sel.n_test,
            "alpha": sel.alpha,
            "aurc": sel.aurc,
            "threshold": sel.threshold,
            "threshold_supported": sel.threshold_supported,
            "coverage": sel.coverage,
            "error_on_answered": sel.error_on_answered,
            "guarantee_held": sel.guarantee_held,
            "baseline_error": sel.baseline_error,
            "n_distinct_confidences": sel.n_distinct_confidences,
            "aurc_is_informative": sel.aurc_is_informative,
            "curve": {"coverage": sel.curve_coverage, "risk": sel.curve_risk},
        },
        "documents": [
            {
                "doc_id": d.doc_id,
                "n_pages": d.n_pages,
                "scanned": d.golden.scanned,
                "source": d.golden.source,
                "error": d.error,
                "latency_s": d.score.latency_s,
                "cost_usd": d.score.cost_usd,
                "fields": [
                    {
                        "name": fs.name.value,
                        "outcome": fs.outcome.value,
                        "expected": fs.expected,
                        "actual": fs.actual,
                        "ambiguous": fs.ambiguous,
                        "grounded": fs.grounded,
                        "note": _golden_note(d.golden, fs.name),
                        **_evidence_dict(d.extraction, fs.name),
                    }
                    for fs in d.score.fields
                ],
                "ratios": [
                    {
                        "name": r.name,
                        "outcome": r.outcome.value,
                        "extracted": r.extracted,
                        "expected": r.expected,
                        "extracted_reason": r.extracted_reason,
                        "expected_reason": r.expected_reason,
                        "inputs_used": [i.value for i in r.inputs_used],
                    }
                    for r in d.ratios
                ],
            }
            for d in run.documents
        ],
    }


def _golden_note(golden: GoldenRecord, name: LineItem) -> str:
    g = golden.get(name)
    return "" if g is None else g.note


def _evidence_dict(extraction: Extraction, name: LineItem) -> dict[str, Any]:
    """The extractor's own claim for one field, quote included.

    Carried into the artifact deliberately: a report that states a field was
    wrong, without the quote the extractor read, cannot be audited -- and an
    unauditable report is the thing this project exists to replace.
    """
    fld = extraction.get(name)
    if fld is None:
        return {"confidence": None, "unit_scale": None, "page": None, "quote": None}
    return {
        "confidence": fld.confidence,
        "unit_scale": fld.unit_scale,
        "currency": fld.currency,
        "raw_value": fld.value,
        "page": None if fld.evidence is None else fld.evidence.page,
        "quote": None if fld.evidence is None else fld.evidence.quote,
    }


def to_dict(run: RunResult) -> dict[str, Any]:
    """A run as plain JSON-serialisable data. The web API serves this verbatim."""
    return _run_dict(run)


def save_run(run: RunResult, directory: Path | str) -> Path:
    """Write ``<directory>/<run_id>.json`` and return the path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run.run_id}.json"
    path.write_text(json.dumps(to_dict(run), indent=2) + "\n", encoding="utf-8")
    return path


def load_run(path: Path | str) -> dict[str, Any]:
    """Read a saved run back as plain data.

    Deliberately returns the dict rather than rebuilding a `RunResult`: saved
    runs are an archive format read by the web interface and by diffing tools,
    and a loader that reconstructs dataclasses would have to be kept in step
    with every future field or start rejecting last month's runs.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))
