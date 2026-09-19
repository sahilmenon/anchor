"""Tests for the command line.

The exit codes are the contract with CI, so most of these assert on a code
rather than on wording: a gate breach must be distinguishable from a corpus
that could not be read, or a broken path reads as a quality regression.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anchor.cli import EXIT_DATA_ERROR, EXIT_GATE_FAILED, EXIT_OK, main


def run_cli(args: list[str]) -> int:
    return main(args)


class TestCorpusCommand:
    def test_lists_the_corpus(self, corpus_dir: Path, capsys) -> None:
        assert run_cli(["corpus", "--corpus", str(corpus_dir)]) == EXIT_OK
        out = capsys.readouterr().out
        assert "doc-one" in out and "doc-two" in out

    def test_json_output_is_machine_readable(self, corpus_dir: Path, capsys) -> None:
        run_cli(["corpus", "--corpus", str(corpus_dir), "--json"])
        data = json.loads(capsys.readouterr().out)
        assert data["n_documents"] == 2
        assert {d["doc_id"] for d in data["documents"]} == {"doc-one", "doc-two"}

    def test_a_label_with_no_document_is_a_data_error(self, corpus_dir: Path, capsys) -> None:
        (corpus_dir / "text" / "doc-two.json").unlink()
        assert run_cli(["corpus", "--corpus", str(corpus_dir)]) == EXIT_DATA_ERROR
        assert "MISSING SOURCE DOCUMENTS" in capsys.readouterr().out

    def test_a_missing_corpus_directory_is_a_data_error(self, tmp_path: Path, capsys) -> None:
        assert run_cli(["corpus", "--corpus", str(tmp_path / "nope")]) == EXIT_DATA_ERROR
        assert "no golden/ directory" in capsys.readouterr().err


class TestRunCommand:
    def test_prints_a_report_and_writes_an_artifact(
        self, corpus_dir: Path, tmp_path: Path, capsys
    ) -> None:
        out_dir = tmp_path / "runs"
        code = run_cli(
            ["run", "--corpus", str(corpus_dir), "--out", str(out_dir)]
        )
        assert code == EXIT_OK
        out = capsys.readouterr().out
        assert "FAILURE TAXONOMY" in out
        assert "DERIVED RATIOS" in out

        artifacts = list(out_dir.glob("*.json"))
        assert len(artifacts) == 1
        assert json.loads(artifacts[0].read_text(encoding="utf-8"))["extractor"] == "heuristic"

    def test_no_save_writes_nothing(self, corpus_dir: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "runs"
        run_cli(["run", "--corpus", str(corpus_dir), "--out", str(out_dir), "--no-save"])
        assert not out_dir.exists()

    def test_json_output_parses(self, corpus_dir: Path, capsys) -> None:
        run_cli(["run", "--corpus", str(corpus_dir), "--no-save", "--json"])
        data = json.loads(capsys.readouterr().out)
        assert data["report"]["n_documents"] == 2

    def test_verbose_lists_non_correct_fields(self, corpus_dir: Path, capsys) -> None:
        run_cli(["run", "--corpus", str(corpus_dir), "--no-save", "-v"])
        assert "PER DOCUMENT" in capsys.readouterr().out

    def test_no_verify_is_flagged_in_the_report(self, corpus_dir: Path, capsys) -> None:
        run_cli(["run", "--corpus", str(corpus_dir), "--no-save", "--no-verify"])
        out = capsys.readouterr().out
        assert "--no-verify was used" in out
        assert "upper bound" in out

    def test_a_corpus_with_no_resolvable_documents_is_a_data_error(
        self, corpus_dir: Path, capsys
    ) -> None:
        for p in (corpus_dir / "text").glob("*.json"):
            p.unlink()
        assert run_cli(["run", "--corpus", str(corpus_dir), "--no-save"]) == EXIT_DATA_ERROR
        assert "Nothing to score" in capsys.readouterr().err

    def test_an_unknown_extractor_is_refused(self, corpus_dir: Path) -> None:
        with pytest.raises(SystemExit, match="unknown extractor"):
            run_cli(["run", "--corpus", str(corpus_dir), "--extractor", "gpt-9"])


class TestGateCommand:
    def test_passes_when_floors_are_met(
        self, corpus_dir: Path, thresholds_file: Path, capsys
    ) -> None:
        code = run_cli(
            ["gate", "--corpus", str(corpus_dir), "--thresholds", str(thresholds_file)]
        )
        assert code == EXIT_OK
        assert "PASS" in capsys.readouterr().out

    def test_breach_exits_one_and_explains_the_trade(
        self, corpus_dir: Path, tmp_path: Path, capsys
    ) -> None:
        path = tmp_path / "strict.json"
        path.write_text(json.dumps({"metrics": {"accuracy": {"min": 1.01}}}), encoding="utf-8")
        assert (
            run_cli(["gate", "--corpus", str(corpus_dir), "--thresholds", str(path)])
            == EXIT_GATE_FAILED
        )
        out = capsys.readouterr().out
        assert "FAIL" in out
        assert "Do not widen it silently" in out

    def test_a_breach_and_a_data_error_use_different_exit_codes(
        self, corpus_dir: Path, tmp_path: Path
    ) -> None:
        strict = tmp_path / "strict.json"
        strict.write_text(json.dumps({"metrics": {"accuracy": {"min": 1.01}}}), encoding="utf-8")
        breach = run_cli(["gate", "--corpus", str(corpus_dir), "--thresholds", str(strict)])
        broken = run_cli(["gate", "--corpus", str(corpus_dir), "--thresholds", str(tmp_path / "x")])
        assert breach == EXIT_GATE_FAILED
        assert broken == EXIT_DATA_ERROR
        assert breach != broken

    def test_thresholds_default_to_the_corpus_directory(
        self, corpus_dir: Path, capsys
    ) -> None:
        (corpus_dir / "thresholds.json").write_text(
            json.dumps({"metrics": {"accuracy": {"min": 0.0}}}), encoding="utf-8"
        )
        assert run_cli(["gate", "--corpus", str(corpus_dir)]) == EXIT_OK

    def test_a_malformed_thresholds_file_fails_before_the_run(
        self, corpus_dir: Path, tmp_path: Path, capsys
    ) -> None:
        path = tmp_path / "bad.json"
        path.write_text('{"metrics": {"accuarcy": 0.4}}', encoding="utf-8")
        assert (
            run_cli(["gate", "--corpus", str(corpus_dir), "--thresholds", str(path)])
            == EXIT_DATA_ERROR
        )
        err = capsys.readouterr().err
        assert "unknown metric" in err

    def test_json_verdict_parses(
        self, corpus_dir: Path, thresholds_file: Path, capsys
    ) -> None:
        run_cli(
            ["gate", "--corpus", str(corpus_dir), "--thresholds", str(thresholds_file), "--json"]
        )
        data = json.loads(capsys.readouterr().out)
        assert data["passed"] is True
        assert {c["metric"] for c in data["checks"]} == {"accuracy", "hallucination_rate"}


class TestExtractCommand:
    def test_prints_one_documents_claim(self, corpus_dir: Path, capsys) -> None:
        path = corpus_dir / "text" / "doc-one.json"
        assert run_cli(["extract", str(path)]) == EXIT_OK
        data = json.loads(capsys.readouterr().out)
        revenue = next(f for f in data["fields"] if f["name"] == "revenue_ltm")
        assert revenue["value"] == 24180.0
        assert revenue["unit_scale"] == 1000.0
        assert revenue["grounded"] is True

    def test_no_verify_leaves_grounding_undecided(self, corpus_dir: Path, capsys) -> None:
        path = corpus_dir / "text" / "doc-one.json"
        run_cli(["extract", str(path), "--no-verify"])
        data = json.loads(capsys.readouterr().out)
        assert all(f["grounded"] is None for f in data["fields"])

    def test_a_missing_file_is_a_data_error(self, tmp_path: Path, capsys) -> None:
        assert run_cli(["extract", str(tmp_path / "nope.json")]) == EXIT_DATA_ERROR
        assert "no such file" in capsys.readouterr().err


class TestParser:
    def test_a_command_is_required(self) -> None:
        with pytest.raises(SystemExit):
            run_cli([])

    def test_version_is_available(self, capsys) -> None:
        with pytest.raises(SystemExit):
            run_cli(["--version"])
        assert "anchor" in capsys.readouterr().out


class TestSweepCommand:
    def test_baseline_only_sweep_renders_a_frontier(self, corpus_dir: Path, capsys) -> None:
        code = run_cli(
            ["sweep", "--corpus", str(corpus_dir), "--models", "", "--no-save"]
        )
        assert code == EXIT_OK
        out = capsys.readouterr().out
        assert "heuristic" in out
        assert "frontier" in out
        assert "Read every figure with its N" in out

    def test_a_configuration_that_fails_does_not_abort_the_sweep(
        self, corpus_dir: Path, capsys
    ) -> None:
        """One unreachable model must not cost the rows that did work."""
        code = run_cli(
            [
                "sweep",
                "--corpus",
                str(corpus_dir),
                "--models",
                "definitely-not-a-model",
                "--no-save",
                "--json",
            ]
        )
        rows = json.loads(capsys.readouterr().out)["rows"]
        labels = {r["label"] for r in rows}
        assert "heuristic" in labels
        assert "definitely-not-a-model" in labels
        failed = next(r for r in rows if r["label"] == "definitely-not-a-model")
        assert failed["error"]
        # The baseline still scored, so the sweep is still useful.
        assert next(r for r in rows if r["label"] == "heuristic")["accuracy"] is not None
        assert code == EXIT_OK

    def test_json_rows_carry_both_halves_of_the_trade(
        self, corpus_dir: Path, capsys
    ) -> None:
        run_cli(
            ["sweep", "--corpus", str(corpus_dir), "--models", "", "--no-save", "--json"]
        )
        row = json.loads(capsys.readouterr().out)["rows"][0]
        for key in ("accuracy", "grounding_rate", "cost_per_document", "latency_p50"):
            assert key in row

    def test_the_heuristic_refuses_a_model_flag(self, corpus_dir: Path) -> None:
        with pytest.raises(SystemExit, match="takes no --model"):
            run_cli(
                [
                    "run",
                    "--corpus",
                    str(corpus_dir),
                    "--extractor",
                    "heuristic",
                    "--model",
                    "claude-opus-5",
                    "--no-save",
                ]
            )
