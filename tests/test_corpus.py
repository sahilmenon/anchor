"""Tests for corpus loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anchor.corpus import Corpus, CorpusError, load_corpus, load_text_document


class TestLoadCorpus:
    def test_loads_every_golden_record(self, corpus_dir: Path) -> None:
        corpus = load_corpus(corpus_dir)
        assert len(corpus) == 2
        assert [i.doc_id for i in corpus.items] == ["doc-one", "doc-two"]

    def test_items_are_sorted_so_the_calibration_split_is_reproducible(
        self, corpus_dir: Path
    ) -> None:
        first = [i.doc_id for i in load_corpus(corpus_dir).items]
        second = [i.doc_id for i in load_corpus(corpus_dir).items]
        assert first == second == sorted(first)

    def test_resolves_text_documents(self, corpus_dir: Path) -> None:
        item = load_corpus(corpus_dir).get("doc-one")
        assert item is not None
        assert item.kind == "text"
        assert item.resolvable
        assert item.document().n_pages == 2

    def test_pdf_wins_over_committed_text(self, corpus_dir: Path) -> None:
        pdfs = corpus_dir / "pdfs"
        pdfs.mkdir()
        (pdfs / "doc-one.pdf").write_bytes(b"%PDF-1.4 not really a pdf")
        item = load_corpus(corpus_dir).get("doc-one")
        assert item is not None
        assert item.kind == "pdf"
        assert item.source == pdfs / "doc-one.pdf"

    def test_label_with_no_document_is_reported_not_dropped(self, corpus_dir: Path) -> None:
        (corpus_dir / "text" / "doc-two.json").unlink()
        corpus = load_corpus(corpus_dir)

        # Still in the corpus -- silently dropping it would let a partial run
        # publish as a whole one.
        assert len(corpus) == 2
        assert [i.doc_id for i in corpus.missing] == ["doc-two"]
        assert [i.doc_id for i in corpus.resolvable] == ["doc-one"]

    def test_unresolvable_item_raises_rather_than_yielding_an_empty_document(
        self, corpus_dir: Path
    ) -> None:
        (corpus_dir / "text" / "doc-two.json").unlink()
        item = load_corpus(corpus_dir).get("doc-two")
        assert item is not None
        with pytest.raises(CorpusError, match="no source document"):
            item.document()

    def test_missing_golden_directory_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(CorpusError, match="no golden/ directory"):
            load_corpus(tmp_path)

    def test_malformed_golden_record_names_the_file(self, corpus_dir: Path) -> None:
        bad = corpus_dir / "golden" / "broken.json"
        bad.write_text('{"doc_id": 5, "fields": "not a list"}', encoding="utf-8")
        with pytest.raises(CorpusError, match="broken.json"):
            load_corpus(corpus_dir)

    def test_thresholds_path_is_conventional(self, corpus_dir: Path) -> None:
        assert load_corpus(corpus_dir).thresholds_path == corpus_dir / "thresholds.json"

    def test_get_returns_none_for_an_unknown_doc_id(self, corpus_dir: Path) -> None:
        assert load_corpus(corpus_dir).get("nope") is None

    def test_empty_golden_directory_is_an_empty_corpus_not_an_error(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "golden").mkdir()
        corpus = load_corpus(tmp_path)
        assert isinstance(corpus, Corpus)
        assert len(corpus) == 0


class TestLoadTextDocument:
    def test_reads_pages(self, tmp_path: Path) -> None:
        path = tmp_path / "d.json"
        path.write_text(json.dumps({"doc_id": "d", "pages": ["a", "b"]}), encoding="utf-8")
        doc = load_text_document(path)
        assert doc.n_pages == 2
        assert doc.page_text(1) == "a"

    def test_doc_id_falls_back_to_the_filename(self, tmp_path: Path) -> None:
        path = tmp_path / "from-stem.json"
        path.write_text(json.dumps({"pages": ["x"]}), encoding="utf-8")
        assert load_text_document(path).doc_id == "from-stem"

    @pytest.mark.parametrize(
        "payload",
        ['{"pages": "not a list"}', '{"pages": [1, 2]}', '["no", "object"]', "{}"],
    )
    def test_rejects_malformed_payloads(self, tmp_path: Path, payload: str) -> None:
        path = tmp_path / "bad.json"
        path.write_text(payload, encoding="utf-8")
        with pytest.raises(CorpusError):
            load_text_document(path)

    def test_missing_file_raises_corpus_error(self, tmp_path: Path) -> None:
        with pytest.raises(CorpusError, match="not readable"):
            load_text_document(tmp_path / "absent.json")
