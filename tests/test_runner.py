"""Tests for the end-to-end run pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anchor import conformal
from anchor.corpus import load_corpus
from anchor.extractors.heuristic import HeuristicExtractor
from anchor.ratios import LineItem
from anchor.runner import (
    RatioOutcome,
    UnknownMetric,
    _split_by_document,
    compare_ratios,
    load_run,
    metric_value,
    run_extractor,
    save_run,
    to_dict,
)
from anchor.schema import Evidence, ExtractedField, Extraction, GoldenField, GoldenRecord


@pytest.fixture
def run(corpus_dir: Path):
    return run_extractor(load_corpus(corpus_dir), HeuristicExtractor())


class TestRunExtractor:
    def test_scores_every_resolvable_document(self, run) -> None:
        assert run.report.n_documents == 2
        assert [d.doc_id for d in run.documents] == ["doc-one", "doc-two"]

    def test_is_deterministic(self, corpus_dir: Path) -> None:
        a = run_extractor(load_corpus(corpus_dir), HeuristicExtractor(), run_id="fixed")
        b = run_extractor(load_corpus(corpus_dir), HeuristicExtractor(), run_id="fixed")

        # Wall-clock latency is measured, not derived, so it differs between two
        # runs of identical work. Everything that is a claim about the documents
        # must not.
        def claims(run):
            data = to_dict(run)
            report = {k: v for k, v in data["report"].items() if k != "total_latency_s"}
            return report, data["selective"], data["ratios"]

        assert claims(a) == claims(b)

    def test_unresolvable_documents_are_listed_not_silently_dropped(
        self, corpus_dir: Path
    ) -> None:
        (corpus_dir / "text" / "doc-two.json").unlink()
        result = run_extractor(load_corpus(corpus_dir), HeuristicExtractor())
        assert result.skipped == ["doc-two"]
        assert result.report.n_documents == 1

    def test_verification_runs_by_default(self, run) -> None:
        assert run.verified
        grounded = [f.grounded for d in run.documents for f in d.score.fields]
        assert any(g is True for g in grounded)

    def test_no_verify_leaves_grounding_undecided_and_records_that(
        self, corpus_dir: Path
    ) -> None:
        result = run_extractor(
            load_corpus(corpus_dir), HeuristicExtractor(), verify=False
        )
        assert result.verified is False
        assert all(f.grounded is None for d in result.documents for f in d.score.fields)

    def test_unreadable_document_costs_its_own_fields_not_the_run(
        self, corpus_dir: Path
    ) -> None:
        (corpus_dir / "text" / "doc-two.json").write_text("{ not json", encoding="utf-8")
        result = run_extractor(load_corpus(corpus_dir), HeuristicExtractor())

        assert result.report.n_documents == 2
        broken = next(d for d in result.documents if d.doc_id == "doc-two")
        assert broken.error
        assert all(f.actual is None for f in broken.score.fields)

        healthy = next(d for d in result.documents if d.doc_id == "doc-one")
        assert healthy.score.n_correct > 0

    def test_path_only_extractor_is_refused_on_a_text_corpus(self, corpus_dir: Path) -> None:
        class PathOnly:
            name = "path-only"

            def extract(self, pdf_path, doc_id):  # pragma: no cover - never reached
                raise AssertionError("should not be called for a text-backed item")

        result = run_extractor(load_corpus(corpus_dir), PathOnly())
        assert all("only accepts a PDF path" in d.error for d in result.documents)


class TestSelectiveSummary:
    def test_splits_whole_documents_between_calibration_and_test(self) -> None:
        a = [conformal.Scored(0.9, True)]
        b = [conformal.Scored(0.4, False), conformal.Scored(0.5, True)]
        c = [conformal.Scored(0.7, True)]
        cal, test = _split_by_document([("a", a), ("b", b), ("c", c)])
        assert cal == a + c
        assert test == b

    def test_counts_only_attempted_fields(self, run) -> None:
        attempted = sum(
            1 for d in run.documents for f in d.score.fields if f.actual is not None
        )
        assert run.selective is not None
        assert run.selective.n_items == attempted

    def test_flags_a_flat_confidence_signal_as_uninformative_for_aurc(self, run) -> None:
        # The heuristic reports at most three confidence tiers, which is exactly
        # the case where AURC stops measuring ranking quality.
        assert run.selective is not None
        assert run.selective.n_distinct_confidences <= 3
        assert run.selective.aurc_is_informative is False


class TestCompareRatios:
    def _extraction(self, **values: float) -> Extraction:
        return Extraction(
            doc_id="d",
            extractor="test",
            fields=[
                ExtractedField(
                    name=LineItem(name),
                    value=value,
                    confidence=0.9,
                    evidence=Evidence(page=1, quote=str(value)),
                    grounded=True,
                )
                for name, value in values.items()
            ],
        )

    def _golden(self, **values: float | None) -> GoldenRecord:
        return GoldenRecord(
            doc_id="d",
            fields=[GoldenField(name=LineItem(n), value=v) for n, v in values.items()],
        )

    def test_agreement_when_both_sides_compute_the_same_value(self) -> None:
        got = compare_ratios(
            self._extraction(total_debt=100.0, cash=20.0, ebitda_reported=40.0),
            self._golden(total_debt=100.0, cash=20.0, ebitda_reported=40.0),
        )
        leverage = next(r for r in got if r.name == "leverage")
        assert leverage.outcome is RatioOutcome.AGREE
        assert leverage.extracted == pytest.approx(2.0)

    def test_disagreement_when_an_input_was_misread(self) -> None:
        got = compare_ratios(
            self._extraction(total_debt=50.0, cash=20.0, ebitda_reported=40.0),
            self._golden(total_debt=100.0, cash=20.0, ebitda_reported=40.0),
        )
        leverage = next(r for r in got if r.name == "leverage")
        assert leverage.outcome is RatioOutcome.DISAGREE
        assert not leverage.safe

    def test_joint_abstention_is_a_safe_outcome(self) -> None:
        got = compare_ratios(self._extraction(), self._golden())
        dscr = next(r for r in got if r.name == "dscr")
        assert dscr.outcome is RatioOutcome.BOTH_ABSTAINED
        assert dscr.safe

    def test_asserting_a_ratio_truth_cannot_support_is_unsafe(self) -> None:
        got = compare_ratios(
            self._extraction(total_debt=100.0, cash=20.0, ebitda_reported=40.0),
            self._golden(total_debt=None, cash=20.0, ebitda_reported=40.0),
        )
        leverage = next(r for r in got if r.name == "leverage")
        assert leverage.outcome is RatioOutcome.GOLDEN_ABSTAINED
        assert not leverage.safe

    def test_abstaining_where_truth_has_an_answer_is_safe_but_not_agreement(self) -> None:
        got = compare_ratios(
            self._extraction(cash=20.0, ebitda_reported=40.0),
            self._golden(total_debt=100.0, cash=20.0, ebitda_reported=40.0),
        )
        leverage = next(r for r in got if r.name == "leverage")
        assert leverage.outcome is RatioOutcome.EXTRACTION_ABSTAINED
        assert leverage.safe

    def test_refusal_reason_survives_into_the_comparison(self) -> None:
        got = compare_ratios(
            self._extraction(cfads=100.0, interest_expense=10.0),
            self._golden(cfads=100.0, interest_expense=10.0),
        )
        dscr = next(r for r in got if r.name == "dscr")
        assert "PRINCIPAL_REPAYMENTS" in dscr.extracted_reason


class TestMetricValue:
    def test_bare_names_read_the_overall_stats(self, run) -> None:
        assert metric_value(run, "accuracy") == run.report.overall.accuracy

    def test_ambiguous_prefix_reads_the_abstention_test_set(self, run) -> None:
        assert metric_value(run, "ambiguous.accuracy") == run.report.on_ambiguous.accuracy

    def test_field_scoped_names(self, run) -> None:
        stats = run.report.per_field[LineItem.CASH]
        assert metric_value(run, "field.cash.accuracy") == stats.accuracy

    def test_ratio_and_selective_names(self, run) -> None:
        assert metric_value(run, "ratio.safety") == run.ratio_safety
        assert run.selective is not None
        assert metric_value(run, "selective.aurc") == run.selective.aurc

    def test_undefined_metric_returns_none_rather_than_zero(self, run) -> None:
        # No field named principal_repayments was ever attempted here, so its
        # grounding rate has a zero denominator.
        stats = run.report.per_field[LineItem.PRINCIPAL_REPAYMENTS]
        assert stats.n_attempted == 0
        assert metric_value(run, "field.principal_repayments.grounding_rate") is None

    @pytest.mark.parametrize(
        "name",
        ["nonsense", "ambiguous.nonsense", "field.not_a_line_item.accuracy",
         "field.cash.nonsense", "selective.nonsense"],
    )
    def test_unknown_names_raise_so_a_typo_cannot_gate_on_nothing(self, run, name) -> None:
        with pytest.raises(UnknownMetric):
            metric_value(run, name)


class TestSerialisation:
    def test_round_trips_through_disk(self, run, tmp_path: Path) -> None:
        path = save_run(run, tmp_path / "runs")
        assert path.name == f"{run.run_id}.json"
        assert load_run(path) == to_dict(run)

    def test_carries_evidence_so_a_number_can_be_audited(self, run) -> None:
        data = to_dict(run)
        revenue = next(
            f for f in data["documents"][0]["fields"] if f["name"] == "revenue_ltm"
        )
        assert "24,180" in revenue["quote"]
        assert revenue["page"] == 1
        assert revenue["unit_scale"] == 1000.0

    def test_carries_the_labellers_note(self, run) -> None:
        data = to_dict(run)
        interest = next(
            f for f in data["documents"][0]["fields"] if f["name"] == "interest_expense"
        )
        assert interest["ambiguous"] is True
        assert interest["note"] == "printed in parentheses"

    def test_is_json_serialisable(self, run) -> None:
        assert json.loads(json.dumps(to_dict(run)))["extractor"] == "heuristic"

    def test_taxonomy_counts_partition_the_fields(self, run) -> None:
        overall = to_dict(run)["report"]["overall"]
        assert sum(overall["taxonomy"].values()) == overall["n_fields"]
