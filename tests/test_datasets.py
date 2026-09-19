"""Tests for external-dataset import.

Built against a fixture in Kleister's exact on-disk format rather than the real
12GB checkout: the format is what the adapter depends on, and a test that needs
a download is a test that stops running.

The assertions that matter are not about parsing. They are about what an import
is allowed to claim: a field the dataset never annotated must not become a
golden `null`, because that would score a correct extraction as a hallucination.
"""

from __future__ import annotations

import json
import lzma
from pathlib import Path

import pytest

from anchor.corpus import CorpusError, load_corpus
from anchor.datasets import DATASETS, KLEISTER_CHARITY, import_kleister
from anchor.schema import LineItem

ESCAPED_NEWLINE = chr(92) + "n"

ALL_KEYS = (
    "address__post_town address__postcode charity_name charity_number "
    "income_annually_in_british_pounds report_date spending_annually_in_british_pounds"
)


def _doc_text(*lines: str) -> str:
    """Kleister ships text with newlines escaped as a literal backslash-n."""
    return ESCAPED_NEWLINE.join(lines)


@pytest.fixture
def kleister(tmp_path: Path) -> Path:
    """A three-document checkout: a value, a decoy, and an unrequested key."""
    split = tmp_path / "dev-0"
    split.mkdir(parents=True)

    rows = [
        # Annotated with a value.
        [
            "aaa111.pdf",
            ALL_KEYS,
            _doc_text("Havens Hospice", "Total income 10,348,000"),
            "tesseract text",
            "textract text",
            _doc_text("Havens Hospice", "Total income 10,348,000"),
        ],
        # income requested but deliberately unanswered: a decoy key.
        [
            "bbb222.pdf",
            ALL_KEYS,
            _doc_text("Wormington Village Society", "No income figure stated"),
            "tesseract text",
            "textract text",
            _doc_text("Wormington Village Society", "No income figure stated"),
        ],
        # income not among the requested keys at all.
        [
            "ccc333.pdf",
            "charity_name charity_number report_date",
            _doc_text("Some Trust", "Total income 55,000"),
            "tesseract text",
            "textract text",
            _doc_text("Some Trust", "Total income 55,000"),
        ],
    ]
    with lzma.open(split / "in.tsv.xz", "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write("\t".join(row) + "\n")

    (split / "expected.tsv").write_text(
        "\n".join(
            [
                "charity_name=Havens_Hospice income_annually_in_british_pounds=10348000.00",
                "charity_name=Wormington_Village_Society",
                "charity_name=Some_Trust",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return tmp_path


class TestImport:
    def test_imports_a_scoreable_corpus(self, kleister: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        result = import_kleister(kleister, out)
        assert result.n_documents == 2  # the third annotates nothing
        corpus = load_corpus(out)
        assert len(corpus) == 2
        assert all(item.resolvable for item in corpus.items)

    def test_a_stated_value_becomes_a_golden_value(self, kleister: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        import_kleister(kleister, out)
        item = next(i for i in load_corpus(out).items if "aaa111" in i.doc_id)
        field = item.golden.get(LineItem.REVENUE_LTM)
        assert field.value == 10_348_000.0

    def test_a_decoy_key_becomes_a_true_absence(self, kleister: Path, tmp_path: Path) -> None:
        """The one thing no other surveyed dataset provides: labelled absence."""
        out = tmp_path / "out"
        result = import_kleister(kleister, out)
        item = next(i for i in load_corpus(out).items if "bbb222" in i.doc_id)
        field = item.golden.get(LineItem.REVENUE_LTM)
        assert field is not None
        assert field.value is None
        assert "decoy" in field.note
        assert result.n_absent == 1

    def test_an_unrequested_key_is_omitted_not_labelled_absent(
        self, kleister: Path, tmp_path: Path
    ) -> None:
        """The correctness rule the whole adapter turns on.

        `ccc333` states income in its text but the annotators never asked for
        it. Writing `null` would score an extractor that reads 55,000 correctly
        as a HALLUCINATION.
        """
        out = tmp_path / "out"
        import_kleister(kleister, out)
        assert not any("ccc333" in i.doc_id for i in load_corpus(out).items)

    def test_unannotated_line_items_are_never_scored(
        self, kleister: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        import_kleister(kleister, out)
        for item in load_corpus(out).items:
            names = {f.name for f in item.golden.fields}
            assert names == {LineItem.REVENUE_LTM}
            assert LineItem.TOTAL_DEBT not in names
            assert LineItem.CFADS not in names

    def test_values_are_flagged_ambiguous_by_default(
        self, kleister: Path, tmp_path: Path
    ) -> None:
        """The mapping embeds a definitional choice, not a reading."""
        out = tmp_path / "out"
        import_kleister(kleister, out)
        item = next(i for i in load_corpus(out).items if "aaa111" in i.doc_id)
        assert item.golden.get(LineItem.REVENUE_LTM).ambiguous is True

    def test_ambiguity_flag_can_be_turned_off(self, kleister: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        import_kleister(kleister, out, mark_ambiguous=False)
        item = next(i for i in load_corpus(out).items if "aaa111" in i.doc_id)
        assert item.golden.get(LineItem.REVENUE_LTM).ambiguous is False

    def test_text_is_unescaped_so_quotes_can_verify(
        self, kleister: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        import_kleister(kleister, out)
        item = next(i for i in load_corpus(out).items if "aaa111" in i.doc_id)
        text = item.document().page_text(1)
        assert ESCAPED_NEWLINE not in text
        assert "Total income 10,348,000" in text

    def test_provenance_survives_into_the_record(self, kleister: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        import_kleister(kleister, out)
        item = next(i for i in load_corpus(out).items if "aaa111" in i.doc_id)
        assert "kleister-charity" in item.golden.source
        assert "aaa111.pdf" in item.golden.source

    def test_limit_truncates(self, kleister: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        assert import_kleister(kleister, out, limit=1).n_documents == 1

    def test_writes_a_readme_stating_what_is_not_measured(
        self, kleister: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        import_kleister(kleister, out)
        readme = (out / "README.md").read_text(encoding="utf-8")
        assert "never scored" in readme
        assert "Do not commit" in readme  # Kleister declares no licence


class TestGuards:
    def test_a_missing_checkout_says_how_to_get_one(self, tmp_path: Path) -> None:
        with pytest.raises(CorpusError, match="git clone"):
            import_kleister(tmp_path, tmp_path / "out")

    def test_a_split_without_labels_is_refused(self, kleister: Path, tmp_path: Path) -> None:
        (kleister / "dev-0" / "expected.tsv").unlink()
        with pytest.raises(CorpusError, match="no labels"):
            import_kleister(kleister, tmp_path / "out")

    def test_mismatched_row_counts_are_refused(self, kleister: Path, tmp_path: Path) -> None:
        """Silently zipping to the shorter list would mislabel every document."""
        path = kleister / "dev-0" / "expected.tsv"
        path.write_text("charity_name=Only_One\n", encoding="utf-8")
        with pytest.raises(CorpusError, match="inconsistent"):
            import_kleister(kleister, tmp_path / "out")

    def test_an_unknown_text_column_is_refused(self, kleister: Path, tmp_path: Path) -> None:
        with pytest.raises(CorpusError, match="unknown text column"):
            import_kleister(kleister, tmp_path / "out", text_column="text_ocr")


class TestSpec:
    def test_kleister_is_registered_and_marked_uncommittable(self) -> None:
        assert DATASETS["kleister-charity"] is KLEISTER_CHARITY
        assert KLEISTER_CHARITY.committable is False
        assert KLEISTER_CHARITY.licence == "none declared"

    def test_only_revenue_is_mapped(self) -> None:
        """Seven of eight line items have no counterpart, and that is the point."""
        assert set(KLEISTER_CHARITY.field_map.values()) == {LineItem.REVENUE_LTM}


class TestEndToEnd:
    def test_an_imported_corpus_scores_without_a_hallucination_denominator(
        self, kleister: Path, tmp_path: Path
    ) -> None:
        """An import has almost no unanswerable fields, so the rate is undefined.

        Reported as None rather than 0.0, which would read as "this extractor
        never hallucinates" when it means "nobody asked it the question".
        """
        from anchor.extractors.heuristic import HeuristicExtractor
        from anchor.runner import run_extractor

        out = tmp_path / "out"
        import_kleister(kleister, out)
        run = run_extractor(load_corpus(out), HeuristicExtractor())

        assert run.report.n_fields == 2
        assert run.report.overall.n_unanswerable == 1  # the decoy only
        assert json.loads(json.dumps(run.report.overall.taxonomy or {}, default=str)) is not None
