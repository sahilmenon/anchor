"""Tests for anchor.scoring.

Organised by the thing being pinned down rather than by function, because the
contract that matters is behavioural: every Outcome must be reachable, the
grounding-beats-correctness precedence must hold, and every rate must survive a
zero denominator.
"""

from __future__ import annotations

import pytest

from anchor.schema import (
    Evidence,
    ExtractedField,
    Extraction,
    GoldenField,
    GoldenRecord,
    LineItem,
)
from anchor.scoring import (
    DocumentScore,
    Outcome,
    Report,
    Stats,
    aggregate,
    numbers_match,
    score_document,
    score_field,
)

# ---------------------------------------------------------------------------
# Builders -- keep the tests about scoring, not about pydantic kwargs
# ---------------------------------------------------------------------------

EV = Evidence(page=1, quote="Revenue of $100.0m for the twelve months ended")


def ef(
    name: LineItem = LineItem.REVENUE_LTM,
    value: float | None = 100.0,
    grounded: bool | None = True,
    unit_scale: float = 1.0,
) -> ExtractedField:
    return ExtractedField(
        name=name,
        value=value,
        unit_scale=unit_scale,
        evidence=None if value is None else EV,
        grounded=grounded,
        confidence=0.9,
    )


def gf(
    name: LineItem = LineItem.REVENUE_LTM,
    value: float | None = 100.0,
    ambiguous: bool = False,
) -> GoldenField:
    return GoldenField(name=name, value=value, page=1, ambiguous=ambiguous)


def extraction(*fields: ExtractedField, doc_id: str = "doc1", **kw) -> Extraction:
    kw.setdefault("extractor", "test-extractor")
    return Extraction(doc_id=doc_id, fields=list(fields), **kw)


def golden(*fields: GoldenField, doc_id: str = "doc1") -> GoldenRecord:
    return GoldenRecord(doc_id=doc_id, source="test", pages=1, fields=list(fields))


# ---------------------------------------------------------------------------
# numbers_match -- both tolerance paths
# ---------------------------------------------------------------------------


class TestNumbersMatch:
    def test_exact(self):
        assert numbers_match(100.0, 100.0)

    def test_relative_path_only(self):
        # 1_000_000 vs 1_005_000: absolute gap of 5000 blows the abs_tol of 1.0,
        # but 0.5% is inside the 1% relative tolerance. Restated financials.
        assert abs(1_005_000 - 1_000_000) > 1.0
        assert numbers_match(1_000_000.0, 1_005_000.0)

    def test_relative_path_rejects_just_outside(self):
        # 2% apart: outside both tolerances at this magnitude.
        assert not numbers_match(1_000_000.0, 1_020_000.0)

    def test_absolute_path_only(self):
        # 0.0 vs 0.4: relative error is unbounded (denominator 0.4 -> 100%),
        # only the absolute tolerance can save it.
        assert not numbers_match(0.0, 0.4, rel_tol=0.01, abs_tol=0.0)
        assert numbers_match(0.0, 0.4)

    def test_absolute_path_rejects_just_outside(self):
        assert not numbers_match(0.0, 1.5, rel_tol=0.0)

    def test_either_tolerance_suffices(self):
        # Small magnitude, huge relative error, but inside abs_tol -> pass.
        assert numbers_match(1.0, 1.9, rel_tol=0.001, abs_tol=1.0)
        # Large magnitude, big absolute error, but inside rel_tol -> pass.
        assert numbers_match(1e9, 1.005e9, rel_tol=0.01, abs_tol=1.0)

    def test_negative_values(self):
        assert numbers_match(-500_000.0, -502_000.0)
        assert not numbers_match(-500_000.0, 500_000.0)

    @pytest.mark.parametrize("a,b", [(None, 1.0), (1.0, None), (None, None)])
    def test_none_never_matches(self, a, b):
        # Including None vs None: abstention agreement is a taxonomy decision,
        # not a numeric one, or accuracy would be inflated by abstentions.
        assert not numbers_match(a, b)

    def test_tolerances_are_overridable(self):
        assert not numbers_match(100.0, 110.0)
        assert numbers_match(100.0, 110.0, rel_tol=0.2)
        assert numbers_match(100.0, 110.0, abs_tol=20.0)


# ---------------------------------------------------------------------------
# score_field -- every Outcome reachable
# ---------------------------------------------------------------------------


class TestOutcomes:
    def test_correct(self):
        s = score_field(ef(value=100.0, grounded=True), gf(value=100.0))
        assert s.outcome is Outcome.CORRECT

    def test_correct_within_tolerance(self):
        s = score_field(ef(value=100.4), gf(value=100.0))
        assert s.outcome is Outcome.CORRECT

    def test_mismatch(self):
        s = score_field(ef(value=250.0), gf(value=100.0))
        assert s.outcome is Outcome.MISMATCH
        assert s.expected == 100.0
        assert s.actual == 250.0

    def test_omission_via_none_value(self):
        s = score_field(ef(value=None, grounded=None), gf(value=100.0))
        assert s.outcome is Outcome.OMISSION

    def test_omission_via_field_absent_from_extraction(self):
        # A field the extractor never emitted is the same failure as one it
        # emitted with value=None.
        s = score_field(None, gf(value=100.0))
        assert s.outcome is Outcome.OMISSION
        assert s.actual is None
        assert s.grounded is None

    def test_hallucination(self):
        # Golden value None == "the document does not state this".
        s = score_field(ef(value=42.0, grounded=True), gf(value=None))
        assert s.outcome is Outcome.HALLUCINATION

    def test_correct_abstention(self):
        s = score_field(ef(value=None, grounded=None), gf(value=None))
        assert s.outcome is Outcome.CORRECT_ABSTENTION

    def test_correct_abstention_when_field_absent_entirely(self):
        s = score_field(None, gf(value=None))
        assert s.outcome is Outcome.CORRECT_ABSTENTION

    def test_ungrounded(self):
        s = score_field(ef(value=999.0, grounded=False), gf(value=100.0))
        assert s.outcome is Outcome.UNGROUNDED

    def test_every_outcome_is_reachable(self):
        seen = {
            score_field(ef(value=100.0), gf(value=100.0)).outcome,
            score_field(ef(value=1.0), gf(value=100.0)).outcome,
            score_field(None, gf(value=100.0)).outcome,
            score_field(ef(value=1.0), gf(value=None)).outcome,
            score_field(None, gf(value=None)).outcome,
            score_field(ef(value=100.0, grounded=False), gf(value=100.0)).outcome,
        }
        assert seen == set(Outcome)


class TestPrecedence:
    """Grounding is checked before value comparison -- the central design call."""

    def test_correct_value_with_failed_citation_is_ungrounded(self):
        s = score_field(ef(value=100.0, grounded=False), gf(value=100.0))
        assert s.outcome is Outcome.UNGROUNDED
        assert s.outcome is not Outcome.CORRECT
        # The correct value is still recorded for the failure report.
        assert s.actual == 100.0 and s.expected == 100.0

    def test_wrong_value_with_failed_citation_is_ungrounded_not_mismatch(self):
        s = score_field(ef(value=7.0, grounded=False), gf(value=100.0))
        assert s.outcome is Outcome.UNGROUNDED

    def test_invented_value_with_failed_citation_is_ungrounded_not_hallucination(self):
        # Precedence means UNGROUNDED absorbs this case; documented so the two
        # rates are read together.
        s = score_field(ef(value=7.0, grounded=False), gf(value=None))
        assert s.outcome is Outcome.UNGROUNDED

    def test_grounding_not_consulted_when_abstained(self):
        # No claim means nothing to ground, so grounded=False must not turn an
        # honest abstention into a failure.
        s = score_field(ef(value=None, grounded=False), gf(value=None))
        assert s.outcome is Outcome.CORRECT_ABSTENTION

    def test_grounded_none_is_treated_as_not_disproven(self):
        # Verifier never ran -> do not penalise the taxonomy...
        s = score_field(ef(value=100.0, grounded=None), gf(value=100.0))
        assert s.outcome is Outcome.CORRECT
        # ...but it does not count toward grounding_rate either.
        rep = aggregate([DocumentScore(doc_id="d", extractor="e", fields=[s])])
        assert rep.grounding_rate == 0.0


class TestFieldScoreShape:
    def test_carries_name_ambiguity_and_grounding(self):
        s = score_field(
            ef(name=LineItem.CFADS, value=50.0, grounded=True),
            gf(name=LineItem.CFADS, value=50.0, ambiguous=True),
        )
        assert s.name is LineItem.CFADS
        assert s.ambiguous is True
        assert s.grounded is True
        assert s.expected == 50.0 and s.actual == 50.0

    def test_unit_scale_is_applied_before_comparison(self):
        # Model says "100" meaning $100m; unit_scale carries the magnitude.
        s = score_field(
            ef(value=100.0, unit_scale=1_000_000.0), gf(value=100_000_000.0)
        )
        assert s.outcome is Outcome.CORRECT
        assert s.actual == 100_000_000.0

    def test_helper_predicates(self):
        omitted = score_field(None, gf(value=100.0))
        assert omitted.abstained and omitted.answerable
        assert not omitted.should_have_abstained

        ambiguous_abstention = score_field(None, gf(value=100.0, ambiguous=True))
        assert ambiguous_abstention.should_have_abstained


# ---------------------------------------------------------------------------
# score_document
# ---------------------------------------------------------------------------


class TestScoreDocument:
    def test_golden_record_drives_the_field_set(self):
        # Extractor returns a field with no gold label -> ignored, not counted
        # as a hallucination (there is no ground truth to contradict).
        ext = extraction(
            ef(name=LineItem.REVENUE_LTM, value=100.0),
            ef(name=LineItem.CASH, value=5.0),
        )
        gold = golden(gf(name=LineItem.REVENUE_LTM, value=100.0))
        ds = score_document(ext, gold)
        assert ds.n_fields == 1
        assert [f.name for f in ds.fields] == [LineItem.REVENUE_LTM]

    def test_missing_fields_are_omissions(self):
        ext = extraction(ef(name=LineItem.REVENUE_LTM, value=100.0))
        gold = golden(
            gf(name=LineItem.REVENUE_LTM, value=100.0),
            gf(name=LineItem.TOTAL_DEBT, value=400.0),
        )
        ds = score_document(ext, gold)
        assert ds.counts[Outcome.CORRECT] == 1
        assert ds.counts[Outcome.OMISSION] == 1
        assert ds.n_correct == 1
        assert sum(ds.counts.values()) == ds.n_fields == 2

    def test_carries_cost_and_identity(self):
        ext = extraction(
            ef(),
            latency_s=3.5,
            input_tokens=1200,
            output_tokens=300,
            cost_usd=0.042,
        )
        ds = score_document(ext, golden(gf()))
        assert ds.doc_id == "doc1"
        assert ds.extractor == "test-extractor"
        assert ds.latency_s == 3.5
        assert ds.input_tokens == 1200
        assert ds.output_tokens == 300
        assert ds.cost_usd == 0.042

    def test_empty_golden_record(self):
        ds = score_document(extraction(ef()), golden())
        assert ds.n_fields == 0
        assert ds.counts == {}


# ---------------------------------------------------------------------------
# aggregate -- rates, zero denominators, per-field, ambiguous subset
# ---------------------------------------------------------------------------


def _one_doc(pairs, **kw) -> DocumentScore:
    """Build a DocumentScore from (ExtractedField|None, GoldenField) pairs."""
    return DocumentScore(
        doc_id=kw.pop("doc_id", "doc1"),
        extractor=kw.pop("extractor", "test-extractor"),
        fields=[score_field(e, g) for e, g in pairs],
        **kw,
    )


class TestAggregateRates:
    @pytest.fixture
    def mixed(self) -> Report:
        # 6 fields spanning the whole taxonomy:
        #   CORRECT, MISMATCH, OMISSION  -> 3 answerable + 1 ambiguous-omission
        #   HALLUCINATION, CORRECT_ABSTENTION -> unanswerable
        #   UNGROUNDED -> answerable
        pairs = [
            (ef(LineItem.REVENUE_LTM, 100.0, True), gf(LineItem.REVENUE_LTM, 100.0)),
            (ef(LineItem.TOTAL_DEBT, 1.0, True), gf(LineItem.TOTAL_DEBT, 400.0)),
            (None, gf(LineItem.CASH, 50.0)),
            (ef(LineItem.CFADS, 9.0, True), gf(LineItem.CFADS, None)),
            (None, gf(LineItem.PRINCIPAL_REPAYMENTS, None)),
            (
                ef(LineItem.INTEREST_EXPENSE, 20.0, False),
                gf(LineItem.INTEREST_EXPENSE, 20.0),
            ),
        ]
        return aggregate([_one_doc(pairs)])

    def test_taxonomy_partitions_the_fields(self, mixed):
        assert mixed.taxonomy == {
            Outcome.CORRECT: 1,
            Outcome.MISMATCH: 1,
            Outcome.OMISSION: 1,
            Outcome.HALLUCINATION: 1,
            Outcome.CORRECT_ABSTENTION: 1,
            Outcome.UNGROUNDED: 1,
        }
        assert sum(mixed.taxonomy.values()) == mixed.n_fields == 6

    def test_accuracy_denominator_excludes_unanswerable(self, mixed):
        # 4 answerable (revenue, debt, cash, interest), 1 CORRECT.
        assert mixed.overall.n_answerable == 4
        assert mixed.overall.n_unanswerable == 2
        assert mixed.accuracy == pytest.approx(1 / 4)

    def test_coverage(self, mixed):
        # Attempted 3 of the 4 answerable fields (cash omitted).
        assert mixed.coverage == pytest.approx(3 / 4)

    def test_omission_rate_complements_coverage(self, mixed):
        assert mixed.omission_rate == pytest.approx(1 / 4)
        assert mixed.omission_rate + mixed.coverage == pytest.approx(1.0)

    def test_grounding_rate_over_non_abstained(self, mixed):
        # 4 attempted (revenue, debt, cfads, interest); 3 grounded True.
        assert mixed.overall.n_attempted == 4
        assert mixed.grounding_rate == pytest.approx(3 / 4)

    def test_ungrounded_rate_over_non_abstained(self, mixed):
        assert mixed.ungrounded_rate == pytest.approx(1 / 4)

    def test_hallucination_rate_over_unanswerable(self, mixed):
        assert mixed.hallucination_rate == pytest.approx(1 / 2)

    def test_abstention_precision(self, mixed):
        # 2 abstentions: cash (answerable -> unjustified) and principal
        # repayments (unanswerable -> justified).
        assert mixed.overall.n_abstained == 2
        assert mixed.abstention_precision == pytest.approx(1 / 2)

    def test_cost_and_latency_summed(self):
        docs = [
            _one_doc(
                [(ef(), gf())],
                doc_id="a",
                latency_s=2.0,
                input_tokens=1000,
                output_tokens=200,
                cost_usd=0.01,
            ),
            _one_doc(
                [(ef(), gf())],
                doc_id="b",
                latency_s=3.0,
                input_tokens=1500,
                output_tokens=300,
                cost_usd=0.02,
            ),
        ]
        rep = aggregate(docs)
        assert rep.n_documents == 2
        assert rep.total_latency_s == pytest.approx(5.0)
        assert rep.total_input_tokens == 2500
        assert rep.total_output_tokens == 500
        assert rep.total_tokens == 3000
        assert rep.total_cost_usd == pytest.approx(0.03)

    def test_aggregate_is_micro_averaged_across_documents(self):
        # doc A: 1 field, correct. doc B: 3 fields, all wrong.
        # Micro-average is 1/4, not the 0.5 a per-document macro-average gives.
        a = _one_doc([(ef(), gf())], doc_id="a")
        b = _one_doc(
            [
                (ef(LineItem.CASH, 1.0), gf(LineItem.CASH, 50.0)),
                (ef(LineItem.TOTAL_DEBT, 1.0), gf(LineItem.TOTAL_DEBT, 50.0)),
                (ef(LineItem.CFADS, 1.0), gf(LineItem.CFADS, 50.0)),
            ],
            doc_id="b",
        )
        rep = aggregate([a, b])
        assert rep.accuracy == pytest.approx(1 / 4)


class TestZeroDenominators:
    """Every rate returns None -- never 0.0 -- when its denominator is empty."""

    RATES = (
        "accuracy",
        "coverage",
        "grounding_rate",
        "hallucination_rate",
        "omission_rate",
        "ungrounded_rate",
        "abstention_precision",
    )

    def test_empty_input(self):
        rep = aggregate([])
        assert rep.n_documents == 0
        assert rep.n_fields == 0
        assert rep.taxonomy == {}
        assert rep.per_field == {}
        assert rep.total_cost_usd == 0.0
        assert rep.total_tokens == 0
        for name in self.RATES:
            assert getattr(rep, name) is None, name
            assert getattr(rep.on_ambiguous, name) is None, name

    def test_bare_stats_object_is_all_none(self):
        s = Stats()
        for name in self.RATES:
            assert getattr(s, name) is None, name

    def test_documents_with_no_fields(self):
        rep = aggregate([DocumentScore(doc_id="a", extractor="e", fields=[])])
        assert rep.n_documents == 1
        for name in self.RATES:
            assert getattr(rep, name) is None, name

    def test_all_abstained(self):
        # Extractor answered nothing. Attempted-based rates are undefined;
        # answerable-based rates are defined and bleak.
        pairs = [
            (None, gf(LineItem.REVENUE_LTM, 100.0)),
            (None, gf(LineItem.CASH, 50.0)),
            (None, gf(LineItem.CFADS, None)),
        ]
        rep = aggregate([_one_doc(pairs)])
        assert rep.accuracy == 0.0
        assert rep.coverage == 0.0
        assert rep.omission_rate == 1.0
        assert rep.hallucination_rate == 0.0
        # Nothing was attempted, so there is no citation population at all.
        assert rep.grounding_rate is None
        assert rep.ungrounded_rate is None
        # It abstained 3 times, 1 of which was the right call.
        assert rep.abstention_precision == pytest.approx(1 / 3)

    def test_never_abstains(self):
        # The mirror case: abstention_precision is undefined, and that is the
        # finding -- an extractor with no abstention behaviour to calibrate.
        pairs = [
            (ef(LineItem.REVENUE_LTM, 100.0), gf(LineItem.REVENUE_LTM, 100.0)),
            (ef(LineItem.CASH, 50.0), gf(LineItem.CASH, 50.0)),
        ]
        rep = aggregate([_one_doc(pairs)])
        assert rep.abstention_precision is None
        assert rep.accuracy == 1.0

    def test_all_answerable_means_no_hallucination_denominator(self):
        pairs = [(ef(LineItem.REVENUE_LTM, 100.0), gf(LineItem.REVENUE_LTM, 100.0))]
        rep = aggregate([_one_doc(pairs)])
        assert rep.hallucination_rate is None

    def test_no_answerable_fields(self):
        pairs = [(None, gf(LineItem.CFADS, None))]
        rep = aggregate([_one_doc(pairs)])
        assert rep.accuracy is None
        assert rep.coverage is None
        assert rep.omission_rate is None
        assert rep.hallucination_rate == 0.0
        assert rep.abstention_precision == 1.0


class TestAmbiguousSubset:
    @pytest.fixture
    def rep(self) -> Report:
        pairs = [
            # --- ambiguous fields (the abstention test set) ---------------
            # abstained on an ambiguous field -> justified
            (None, gf(LineItem.EBITDA_ADDBACKS, 12.0, ambiguous=True)),
            # guessed an ambiguous field and got it wrong
            (
                ef(LineItem.CFADS, 1.0, True),
                gf(LineItem.CFADS, 80.0, ambiguous=True),
            ),
            # guessed an ambiguous field and got it right
            (
                ef(LineItem.EBITDA_REPORTED, 30.0, True),
                gf(LineItem.EBITDA_REPORTED, 30.0, ambiguous=True),
            ),
            # --- unambiguous fields --------------------------------------
            (ef(LineItem.REVENUE_LTM, 100.0, True), gf(LineItem.REVENUE_LTM, 100.0)),
            (None, gf(LineItem.CASH, 50.0)),
        ]
        return aggregate([_one_doc(pairs)])

    def test_subset_is_restricted_to_ambiguous_fields(self, rep):
        assert rep.on_ambiguous.n_fields == 3
        assert rep.n_fields == 5

    def test_subset_taxonomy(self, rep):
        assert rep.on_ambiguous.taxonomy == {
            Outcome.OMISSION: 1,
            Outcome.MISMATCH: 1,
            Outcome.CORRECT: 1,
        }

    def test_subset_accuracy_differs_from_overall(self, rep):
        assert rep.accuracy == pytest.approx(2 / 5)
        assert rep.on_ambiguous.accuracy == pytest.approx(1 / 3)

    def test_ambiguous_abstentions_count_as_justified(self, rep):
        # Overall: 2 abstentions (addbacks-ambiguous, cash-unambiguous); only
        # the ambiguous one was the defensible call.
        assert rep.abstention_precision == pytest.approx(1 / 2)
        # Within the ambiguous subset, the single abstention was justified.
        assert rep.on_ambiguous.abstention_precision == 1.0

    def test_subset_empty_when_no_ambiguous_labels(self):
        rep = aggregate([_one_doc([(ef(), gf())])])
        assert rep.on_ambiguous.n_fields == 0
        assert rep.on_ambiguous.accuracy is None
        assert rep.on_ambiguous.abstention_precision is None


class TestPerFieldBreakdown:
    @pytest.fixture
    def rep(self) -> Report:
        # Same two line items across three documents, so per-field aggregation
        # has to cross document boundaries.
        docs = [
            _one_doc(
                [
                    (ef(LineItem.REVENUE_LTM, 100.0), gf(LineItem.REVENUE_LTM, 100.0)),
                    (ef(LineItem.CFADS, 5.0), gf(LineItem.CFADS, 80.0)),
                ],
                doc_id="a",
            ),
            _one_doc(
                [
                    (ef(LineItem.REVENUE_LTM, 200.0), gf(LineItem.REVENUE_LTM, 200.0)),
                    (None, gf(LineItem.CFADS, 90.0)),
                ],
                doc_id="b",
            ),
            _one_doc(
                [
                    (ef(LineItem.REVENUE_LTM, 300.0), gf(LineItem.REVENUE_LTM, 300.0)),
                    (ef(LineItem.CFADS, 7.0, False), gf(LineItem.CFADS, 7.0)),
                ],
                doc_id="c",
            ),
        ]
        return aggregate(docs)

    def test_keys_are_line_items_present_in_the_gold(self, rep):
        assert set(rep.per_field) == {LineItem.REVENUE_LTM, LineItem.CFADS}

    def test_per_field_counts_span_all_documents(self, rep):
        assert rep.per_field[LineItem.REVENUE_LTM].n_fields == 3
        assert rep.per_field[LineItem.CFADS].n_fields == 3

    def test_per_field_isolates_the_weak_line_item(self, rep):
        # This is the point of the breakdown: revenue is solved, CFADS is not.
        assert rep.per_field[LineItem.REVENUE_LTM].accuracy == 1.0
        assert rep.per_field[LineItem.CFADS].accuracy == 0.0
        assert rep.per_field[LineItem.CFADS].taxonomy == {
            Outcome.MISMATCH: 1,
            Outcome.OMISSION: 1,
            Outcome.UNGROUNDED: 1,
        }

    def test_per_field_counts_sum_to_overall(self, rep):
        total = sum(s.n_fields for s in rep.per_field.values())
        assert total == rep.n_fields == 6
        merged: dict = {}
        for s in rep.per_field.values():
            for outcome, n in s.taxonomy.items():
                merged[outcome] = merged.get(outcome, 0) + n
        assert merged == dict(rep.taxonomy)

    def test_overall_accuracy_sits_between_the_per_field_extremes(self, rep):
        assert rep.accuracy == pytest.approx(3 / 6)


class TestEndToEnd:
    def test_extraction_to_report(self):
        ext = extraction(
            ef(LineItem.REVENUE_LTM, 412.7, True),
            ef(LineItem.TOTAL_DEBT, 1_000.0, False),
            ef(LineItem.CASH, None, None),
            ef(LineItem.CFADS, 33.0, True),
            latency_s=12.0,
            input_tokens=9000,
            output_tokens=800,
            cost_usd=0.11,
        )
        gold = golden(
            gf(LineItem.REVENUE_LTM, 413.0),  # rounded restatement -> CORRECT
            gf(LineItem.TOTAL_DEBT, 1_000.0),  # right value, bad cite -> UNGROUNDED
            gf(LineItem.CASH, 25.0),  # abstained -> OMISSION
            gf(LineItem.CFADS, None),  # not stated -> HALLUCINATION
        )
        rep = aggregate([score_document(ext, gold)])
        assert rep.taxonomy == {
            Outcome.CORRECT: 1,
            Outcome.UNGROUNDED: 1,
            Outcome.OMISSION: 1,
            Outcome.HALLUCINATION: 1,
        }
        assert rep.accuracy == pytest.approx(1 / 3)
        assert rep.hallucination_rate == 1.0
        assert rep.abstention_precision == 0.0
        assert rep.total_tokens == 9800
        assert rep.total_cost_usd == pytest.approx(0.11)
