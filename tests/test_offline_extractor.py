"""Tests for the offline extractor.

The behaviour worth pinning is that a replayed claim gets no easier a ride than
a live one. Most of these assert that a transcript is rejected in the same
circumstances a model's answer would be.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anchor.corpus import CorpusError
from anchor.extractors.base import Document
from anchor.extractors.offline import OfflineExtractor, write_claims
from anchor.schema import LineItem

PAGES = [
    "ACME PTY LTD\nAll amounts in A$'000\n\n"
    "Total revenue                               24,180      21,405\n"
    "Finance costs                                 (842)       (774)\n"
]

NAMES = [item.value for item in LineItem]


def claim(name, value, page=None, quote=None, scale=1.0, confidence=0.8):
    return {
        "name": name,
        "value": value,
        "unit_scale": scale,
        "currency": None,
        "page": page,
        "quote": quote,
        "confidence": confidence,
    }


def transcript(*claims):
    named = {c["name"] for c in claims}
    return list(claims) + [claim(n, None) for n in NAMES if n not in named]


@pytest.fixture
def doc() -> Document:
    return Document.from_pages("acme", PAGES)


REVENUE_ROW = "Total revenue                               24,180      21,405"


class TestReplay:
    def test_a_grounded_claim_survives(self, doc, tmp_path: Path) -> None:
        write_claims(tmp_path, "acme", transcript(
            claim("revenue_ltm", 24180, 1, REVENUE_ROW, 1000.0)))
        result = OfflineExtractor(root=tmp_path).extract_document(doc, "acme")
        field = result.get(LineItem.REVENUE_LTM)
        assert field.value == 24180
        assert field.scaled_value == 24_180_000

    def test_claims_no_cost_and_names_itself_offline(self, doc, tmp_path: Path) -> None:
        """A transcript has no provenance, so it must not look like a model row."""
        write_claims(tmp_path, "acme", transcript(
            claim("revenue_ltm", 24180, 1, REVENUE_ROW, 1000.0)))
        result = OfflineExtractor(root=tmp_path, label="manual").extract_document(doc, "acme")
        assert result.extractor == "offline:manual"
        assert result.cost_usd == 0.0
        assert result.model == "manual"

    def test_a_document_with_no_transcript_abstains(self, doc, tmp_path: Path) -> None:
        result = OfflineExtractor(root=tmp_path).extract_document(doc, "acme")
        assert all(f.abstained for f in result.fields)

    def test_latency_from_the_envelope_survives(self, doc, tmp_path: Path) -> None:
        write_claims(tmp_path, "acme", transcript(claim("cash", None)), latency_s=12.5)
        assert OfflineExtractor(root=tmp_path).extract_document(doc, "acme").latency_s == 12.5


class TestVerificationIsNotRelaxed:
    """The point of the whole extractor: replay is held to the same standard."""

    def test_a_quote_that_names_the_row_but_omits_the_figure_is_rejected(
        self, doc, tmp_path: Path
    ) -> None:
        """The mistake a careless citer makes, and it must not pass."""
        write_claims(tmp_path, "acme", transcript(
            claim("revenue_ltm", 24180, 1, "Total revenue", 1000.0)))
        result = OfflineExtractor(root=tmp_path).extract_document(doc, "acme")
        assert result.get(LineItem.REVENUE_LTM).abstained

    def test_a_retyped_quote_that_differs_from_the_page_is_rejected(
        self, doc, tmp_path: Path
    ) -> None:
        """Paraphrasing a citation fails, whoever does it."""
        write_claims(tmp_path, "acme", transcript(
            claim("revenue_ltm", 24180, 1, "Total revenue 24180 21405", 1000.0)))
        result = OfflineExtractor(root=tmp_path).extract_document(doc, "acme")
        assert result.get(LineItem.REVENUE_LTM).abstained

    def test_a_value_with_no_quote_is_rejected(self, doc, tmp_path: Path) -> None:
        """Covers a figure computed by summing rows: real, and uncitable."""
        write_claims(tmp_path, "acme", transcript(claim("revenue_ltm", 24180, 1, None)))
        result = OfflineExtractor(root=tmp_path).extract_document(doc, "acme")
        assert result.get(LineItem.REVENUE_LTM).abstained

    def test_a_page_out_of_range_is_rejected(self, doc, tmp_path: Path) -> None:
        write_claims(tmp_path, "acme", transcript(
            claim("revenue_ltm", 24180, 99, REVENUE_ROW, 1000.0)))
        assert OfflineExtractor(root=tmp_path).extract_document(doc, "acme").get(
            LineItem.REVENUE_LTM).abstained

    def test_verification_can_be_turned_off_for_diagnosis(self, doc, tmp_path: Path) -> None:
        write_claims(tmp_path, "acme", transcript(
            claim("revenue_ltm", 24180, 1, "Total revenue", 1000.0)))
        result = OfflineExtractor(root=tmp_path, verify_quotes=False).extract_document(
            doc, "acme")
        assert result.get(LineItem.REVENUE_LTM).value == 24180

    def test_the_magnitude_convention_applies_to_replay(self, doc, tmp_path: Path) -> None:
        write_claims(tmp_path, "acme", transcript(claim(
            "interest_expense", -842, 1,
            "Finance costs                                 (842)       (774)", 1000.0)))
        result = OfflineExtractor(root=tmp_path).extract_document(doc, "acme")
        assert result.get(LineItem.INTEREST_EXPENSE).value == 842


class TestMalformedTranscripts:
    def test_invalid_json_names_the_file(self, doc, tmp_path: Path) -> None:
        (tmp_path / "acme.json").write_text("{ not json", encoding="utf-8")
        with pytest.raises(CorpusError, match="acme.json"):
            OfflineExtractor(root=tmp_path).extract_document(doc, "acme")

    def test_a_missing_fields_key_is_refused(self, doc, tmp_path: Path) -> None:
        (tmp_path / "acme.json").write_text(json.dumps({"latency_s": 1.0}), encoding="utf-8")
        with pytest.raises(CorpusError, match="fields"):
            OfflineExtractor(root=tmp_path).extract_document(doc, "acme")

    def test_an_unknown_line_item_is_refused(self, doc, tmp_path: Path) -> None:
        (tmp_path / "acme.json").write_text(
            json.dumps({"fields": [claim("not_a_line_item", 1)]}), encoding="utf-8")
        with pytest.raises(CorpusError):
            OfflineExtractor(root=tmp_path).extract_document(doc, "acme")

    def test_write_claims_validates_before_writing(self, tmp_path: Path) -> None:
        """A malformed transcript fails where it is written, not mid-run."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            write_claims(tmp_path, "acme", [claim("not_a_line_item", 1)])
