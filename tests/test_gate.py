"""Tests for the regression gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anchor.corpus import load_corpus
from anchor.extractors.heuristic import HeuristicExtractor
from anchor.gate import (
    GateError,
    Status,
    Threshold,
    Thresholds,
    check_run,
    load_thresholds,
)
from anchor.runner import run_extractor


@pytest.fixture
def run(corpus_dir: Path):
    return run_extractor(load_corpus(corpus_dir), HeuristicExtractor())


def write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestLoadThresholds:
    def test_reads_min_and_max_entries(self, tmp_path: Path) -> None:
        path = write(
            tmp_path / "t.json",
            {"metrics": {"accuracy": {"min": 0.4}, "hallucination_rate": {"max": 0.1}}},
        )
        loaded = load_thresholds(path)
        assert len(loaded) == 2
        by_name = {e.metric: e for e in loaded.entries}
        assert by_name["accuracy"].direction == "min"
        assert by_name["hallucination_rate"].direction == "max"

    def test_a_bare_number_is_read_as_a_minimum(self, tmp_path: Path) -> None:
        path = write(tmp_path / "t.json", {"metrics": {"accuracy": 0.4}})
        entry = load_thresholds(path).entries[0]
        assert (entry.direction, entry.bound) == ("min", 0.4)

    def test_both_bounds_at_once_is_rejected_as_ambiguous(self, tmp_path: Path) -> None:
        path = write(tmp_path / "t.json", {"metrics": {"accuracy": {"min": 0.1, "max": 0.9}}})
        with pytest.raises(GateError, match="exactly one of"):
            load_thresholds(path)

    def test_neither_bound_is_rejected(self, tmp_path: Path) -> None:
        path = write(tmp_path / "t.json", {"metrics": {"accuracy": {"note": "hi"}}})
        with pytest.raises(GateError, match="exactly one of"):
            load_thresholds(path)

    def test_a_typo_in_a_metric_name_fails_loudly(self, tmp_path: Path) -> None:
        path = write(tmp_path / "t.json", {"metrics": {"accuarcy": {"min": 0.4}}})
        with pytest.raises(GateError, match="unknown metric"):
            load_thresholds(path)

    def test_field_scoped_names_are_accepted(self, tmp_path: Path) -> None:
        path = write(tmp_path / "t.json", {"metrics": {"field.cash.accuracy": {"min": 0.5}}})
        assert load_thresholds(path).entries[0].metric == "field.cash.accuracy"

    def test_an_unknown_line_item_is_rejected(self, tmp_path: Path) -> None:
        path = write(tmp_path / "t.json", {"metrics": {"field.goodwill.accuracy": {"min": 0.5}}})
        with pytest.raises(GateError, match="unknown metric"):
            load_thresholds(path)

    def test_empty_metrics_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(GateError, match="non-empty"):
            load_thresholds(write(tmp_path / "t.json", {"metrics": {}}))

    def test_a_boolean_is_not_a_bound(self, tmp_path: Path) -> None:
        path = write(tmp_path / "t.json", {"metrics": {"accuracy": {"min": True}}})
        with pytest.raises(GateError, match="must be a number"):
            load_thresholds(path)

    def test_unreadable_file_names_itself(self, tmp_path: Path) -> None:
        with pytest.raises(GateError, match="not readable"):
            load_thresholds(tmp_path / "absent.json")

    def test_notes_survive_loading(self, tmp_path: Path) -> None:
        path = write(
            tmp_path / "t.json",
            {"metrics": {"accuracy": {"min": 0.4, "note": "measured 0.50 on 6 docs"}}},
        )
        assert load_thresholds(path).entries[0].note == "measured 0.50 on 6 docs"


class TestCheckRun:
    def test_a_met_floor_passes(self, run) -> None:
        result = check_run(run, Thresholds([Threshold("accuracy", "min", 0.0)]))
        assert result.passed
        assert result.checks[0].status is Status.PASS

    def test_a_breached_floor_fails_and_is_listed(self, run) -> None:
        result = check_run(run, Thresholds([Threshold("accuracy", "min", 1.01)]))
        assert not result.passed
        assert [c.threshold.metric for c in result.failures] == ["accuracy"]
        assert "BREACH" in result.checks[0].describe()

    def test_a_breached_ceiling_fails(self, run) -> None:
        result = check_run(run, Thresholds([Threshold("hallucination_rate", "max", -0.01)]))
        assert not result.passed

    def test_an_undefined_metric_is_a_breach_by_default(self, run) -> None:
        # Zero denominator: nothing was attempted for this line item.
        threshold = Threshold("field.principal_repayments.grounding_rate", "min", 0.5)
        result = check_run(run, Thresholds([threshold]))
        assert result.checks[0].status is Status.UNDEFINED
        assert not result.passed
        assert "undefined" in result.checks[0].describe()

    def test_allow_undefined_downgrades_it_to_a_pass(self, run) -> None:
        threshold = Threshold("field.principal_repayments.grounding_rate", "min", 0.5)
        result = check_run(run, Thresholds([threshold], allow_undefined=True))
        assert result.passed

    def test_every_threshold_is_checked_not_just_the_first_failure(self, run) -> None:
        result = check_run(
            run,
            Thresholds(
                [
                    Threshold("accuracy", "min", 1.01),
                    Threshold("coverage", "min", 1.01),
                    Threshold("grounding_rate", "min", 0.0),
                ]
            ),
        )
        assert len(result.checks) == 3
        assert len(result.failures) == 2


class TestCommittedThresholds:
    """The floors checked into this repository must stay loadable and honest."""

    def test_synthetic_thresholds_load(self) -> None:
        loaded = load_thresholds(Path("corpus/synthetic/thresholds.json"))
        assert len(loaded) >= 5
        assert all(e.note for e in loaded.entries), "every floor records its provenance"

    def test_real_corpus_thresholds_load(self) -> None:
        loaded = load_thresholds(Path("corpus/thresholds.json"))
        assert all(e.note for e in loaded.entries)

    def test_the_synthetic_gate_passes_on_the_committed_baseline(self) -> None:
        run = run_extractor(load_corpus("corpus/synthetic"), HeuristicExtractor())
        result = check_run(run, load_thresholds(Path("corpus/synthetic/thresholds.json")))
        assert result.passed, [c.describe() for c in result.failures]

    def test_the_floors_are_close_enough_to_the_baseline_to_catch_a_regression(self) -> None:
        """A floor far below the measured value gates on nothing.

        This is the test that keeps the gate honest over time: if someone
        lowers a floor to make a red build green, the slack check fails and
        says so.
        """
        run = run_extractor(load_corpus("corpus/synthetic"), HeuristicExtractor())
        loaded = load_thresholds(Path("corpus/synthetic/thresholds.json"))
        from anchor.runner import metric_value

        for entry in loaded.entries:
            observed = metric_value(run, entry.metric)
            assert observed is not None
            slack = (
                observed - entry.bound if entry.direction == "min" else entry.bound - observed
            )
            assert 0 <= slack <= 0.25, (
                f"{entry.metric}: bound {entry.bound} is {slack:.3f} away from the "
                f"measured {observed:.3f}. Too much slack and the floor gates on nothing."
            )
