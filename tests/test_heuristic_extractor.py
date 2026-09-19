"""Tests for the deterministic baseline extractor.

These run on synthetic page text via `Document.from_pages`, never on a real
PDF: the extractor's job is text -> line items, and binding the tests to PDF
fixtures would only test PyMuPDF. The cases below are written to look like the
statement text PyMuPDF actually emits -- ragged spacing, wrapped rows,
accounting parentheses and a units note above the table.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from anchor.extractors.base import Document, Extractor
from anchor.extractors.heuristic import (
    HeuristicExtractor,
    detect_currency,
    detect_unit_scale,
    parse_number,
)
from anchor.schema import LineItem

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

INCOME_STATEMENT = """Alpine Infrastructure Holdings Pty Ltd
Consolidated Statement of Profit or Loss
For the year ended 30 June 2024
All figures in thousands of Australian dollars (A$'000)

Total revenue                                  412,905
Cost of sales                                 (238,110)
Gross profit                                   174,795
Employee benefits expense                      (61,220)
EBITDA                                          98,430
Depreciation and amortisation                  (31,004)
Finance costs                                  (18,762)
Profit before tax                               48,664
"""

BALANCE_SHEET = """Consolidated Statement of Financial Position
As at 30 June 2024 (A$'000)

Cash and cash equivalents                       27,430
Trade and other receivables                     55,118
Total current assets                           142,660
Total debt                                     286,500
Trade and other payables                        44,012
"""

CASH_FLOW = """Consolidated Statement of Cash Flows (A$'000)

Net cash from operating activities             112,004
Repayment of borrowings                        (24,000)
Cash flow available for debt service            88,004
EBITDA add-backs
                                                 6,150
"""

FULL_DOC = [INCOME_STATEMENT, BALANCE_SHEET, CASH_FLOW]


def _extract(pages: list[str], doc_id: str = "alpine-2024"):
    return HeuristicExtractor().extract_document(
        Document.from_pages(doc_id, pages), doc_id
    )


# --------------------------------------------------------------------------
# Protocol conformance
# --------------------------------------------------------------------------


def test_conforms_to_extractor_protocol() -> None:
    assert isinstance(HeuristicExtractor(), Extractor)


def test_reports_its_identity_and_zero_cost() -> None:
    result = _extract(FULL_DOC)
    assert result.extractor == "heuristic"
    assert result.model is None
    assert result.input_tokens == 0
    assert result.output_tokens == 0
    assert result.cost_usd == 0.0
    assert result.latency_s >= 0.0


def test_always_emits_every_line_item() -> None:
    result = _extract(FULL_DOC)
    assert [f.name for f in result.fields] == list(LineItem)


# --------------------------------------------------------------------------
# Each line item is found
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        (LineItem.REVENUE_LTM, 412_905),
        (LineItem.EBITDA_REPORTED, 98_430),
        (LineItem.EBITDA_ADDBACKS, 6_150),
        (LineItem.TOTAL_DEBT, 286_500),
        (LineItem.CASH, 27_430),
        # Stored positive: both are MAGNITUDE_ITEMS, and the document prints
        # them parenthesised because they are outflows.
        (LineItem.INTEREST_EXPENSE, 18_762),
        (LineItem.PRINCIPAL_REPAYMENTS, 24_000),
        (LineItem.CFADS, 88_004),
    ],
)
def test_finds_each_line_item(item: LineItem, expected: float) -> None:
    field = _extract(FULL_DOC).get(item)
    assert field is not None
    assert field.value == expected


def test_cites_the_page_the_value_came_from() -> None:
    result = _extract(FULL_DOC)
    assert result.get(LineItem.REVENUE_LTM).evidence.page == 1
    assert result.get(LineItem.TOTAL_DEBT).evidence.page == 2
    assert result.get(LineItem.CFADS).evidence.page == 3


def test_specific_labels_beat_overlapping_broad_ones() -> None:
    """CASH must not swallow the "Cash flow available for debt service" row."""
    doc = ["Cash flow available for debt service   88,004\n"]
    result = _extract(doc)
    assert result.get(LineItem.CFADS).value == 88_004
    assert result.get(LineItem.CASH).abstained


# --------------------------------------------------------------------------
# Units -- the 1000x error class
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("declaration", "scale"),
    [
        ("All figures in thousands", 1e3),
        ("A$'000", 1e3),
        ("$'000", 1e3),
        ("Amounts in millions", 1e6),
        ("A$m", 1e6),
        ("Amounts in billions", 1e9),
        ("no declaration at all", 1.0),
    ],
)
def test_detect_unit_scale(declaration: str, scale: float) -> None:
    assert detect_unit_scale(f"Statement of cash flows\n{declaration}\n") == scale


def test_units_declaration_is_applied_to_every_field_on_the_page() -> None:
    result = _extract(FULL_DOC)
    revenue = result.get(LineItem.REVENUE_LTM)
    assert revenue.unit_scale == 1000.0
    assert revenue.scaled_value == 412_905_000.0


def test_page_without_units_declaration_scales_by_one() -> None:
    page = "Summary\nTotal debt   286500\n"
    field = _extract([page]).get(LineItem.TOTAL_DEBT)
    assert field.unit_scale == 1.0
    assert field.scaled_value == 286_500


def test_units_are_detected_per_page_not_globally() -> None:
    pages = [
        "Income statement (in thousands)\nTotal revenue   412,905\n",
        "Debt summary (in millions)\nTotal debt   286.5\n",
    ]
    result = _extract(pages)
    assert result.get(LineItem.REVENUE_LTM).unit_scale == 1e3
    assert result.get(LineItem.TOTAL_DEBT).unit_scale == 1e6


def test_inline_magnitude_suffix_wins_over_page_declaration() -> None:
    """"A$1.2m" already carries its scale; applying "in thousands" would double it."""
    page = "Facility summary (in thousands)\nTotal debt   A$1.2m\n"
    field = _extract([page]).get(LineItem.TOTAL_DEBT)
    assert field.unit_scale == 1e6
    assert field.scaled_value == pytest.approx(1_200_000)


def test_detects_currency_but_not_from_a_bare_dollar_sign() -> None:
    assert detect_currency("Amounts in A$'000") == "AUD"
    assert detect_currency("Amounts in USD millions") == "USD"
    assert detect_currency("Total debt $286,500") is None


# --------------------------------------------------------------------------
# Number parsing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1,234", 1234.0),
        ("$1,234.50", 1234.50),
        ("(18,762)", -18762.0),
        ("-4,500", -4500.0),
        ("A$1.2m", 1_200_000.0),
        ("3.4bn", 3_400_000_000.0),
        ("850k", 850_000.0),
        ("12 million", 12_000_000.0),
        ("0", 0.0),
    ],
)
def test_parse_number(text: str, expected: float) -> None:
    parsed = parse_number(text)
    assert parsed is not None
    assert parsed.value * parsed.inline_scale == pytest.approx(expected)


@pytest.mark.parametrize("text", ["", "no digits here", "2024", "Note 12", "12.5%"])
def test_parse_number_rejects_implausible_candidates(text: str) -> None:
    assert parse_number(text) is None


def test_parenthesised_debt_service_is_stored_as_a_magnitude() -> None:
    """Parentheses on a debt-service row mean "outflow", not "negative".

    Passing the printed sign through nets one leg of debt service against the
    other, understating it and overstating DSCR -- the direction that flatters
    a borrower, which is the direction this harness must not fail in quietly.
    """
    page = "Statement of cash flows\nRepayment of borrowings   (24,000)\n"
    field = _extract([page]).get(LineItem.PRINCIPAL_REPAYMENTS)
    assert field.value == 24_000
    assert not math.isnan(field.value)


def test_parenthesised_negatives_survive_on_items_that_are_not_magnitudes() -> None:
    """The convention is scoped to debt service, not applied everywhere.

    An add-back can genuinely be a deduction, so its sign is part of the value.
    """
    page = "Reconciliation\nNormalisation adjustments   (1,200)\n"
    field = _extract([page]).get(LineItem.EBITDA_ADDBACKS)
    assert field.value == -1_200


def test_a_units_declaration_is_not_read_as_a_value() -> None:
    """A label that wraps onto its own line must not pick up the scale marker.

    "$'000" contains the digits 000. Reading them as a figure attaches a zero to
    the label above it, with a real page and a real quote behind it, and a zero
    survives every downstream sanity check a reader might apply.
    """
    page = "Total interest bearing liabilities\n$'000\n\nBank overdraft   420\n"
    field = _extract([page]).get(LineItem.TOTAL_DEBT)
    assert field is not None
    assert field.value != 0


@pytest.mark.parametrize("text", ["A$'000", "$ '000", "'000", "in 000s"])
def test_parse_number_rejects_units_markers(text: str) -> None:
    assert parse_number(text) is None


def test_year_in_a_column_heading_is_not_mistaken_for_a_value() -> None:
    pages = ["Total revenue      2024      2023\n              412,905   380,110\n"]
    field = _extract(pages).get(LineItem.REVENUE_LTM)
    assert field.value == 412_905


# --------------------------------------------------------------------------
# Wrapped rows
# --------------------------------------------------------------------------


def test_reads_a_number_from_the_following_line_when_the_row_wraps() -> None:
    field = _extract(FULL_DOC).get(LineItem.EBITDA_ADDBACKS)
    assert field.value == 6_150
    assert "add-backs" in field.evidence.quote.lower()


def test_wrapped_rows_are_reported_with_lower_confidence() -> None:
    result = _extract(FULL_DOC)
    wrapped = result.get(LineItem.EBITDA_ADDBACKS)
    same_line = result.get(LineItem.REVENUE_LTM)
    assert wrapped.confidence < same_line.confidence


def test_does_not_steal_the_next_labelled_row_s_number() -> None:
    page = "Total debt\nCash and cash equivalents   27,430\n"
    result = _extract([page])
    assert result.get(LineItem.TOTAL_DEBT).abstained
    assert result.get(LineItem.CASH).value == 27_430


# --------------------------------------------------------------------------
# Confidence
# --------------------------------------------------------------------------


def test_confidence_is_never_high() -> None:
    """A regex baseline has no business claiming certainty."""
    for field in _extract(FULL_DOC).fields:
        assert field.confidence <= 0.6


def test_exact_labels_outrank_fuzzy_ones() -> None:
    exact = _extract(["Total revenue   412,905\n"]).get(LineItem.REVENUE_LTM)
    fuzzy = _extract(["Net sales   412,905\n"]).get(LineItem.REVENUE_LTM)
    assert exact.value == fuzzy.value == 412_905
    assert exact.confidence > fuzzy.confidence


def test_abstained_fields_carry_zero_confidence() -> None:
    for field in _extract(["Nothing of interest here.\n"]).fields:
        assert field.abstained
        assert field.confidence == 0.0


# --------------------------------------------------------------------------
# Abstention
# --------------------------------------------------------------------------


def test_label_present_but_no_number_abstains() -> None:
    page = "Covenant package\nTotal debt is subject to the leverage covenant.\n"
    field = _extract([page]).get(LineItem.TOTAL_DEBT)
    assert field.abstained
    assert field.evidence is None


def test_a_later_occurrence_can_still_supply_the_number() -> None:
    page = (
        "Total debt is subject to the leverage covenant.\n"
        "\n"
        "Total debt                286,500\n"
    )
    field = _extract([page]).get(LineItem.TOTAL_DEBT)
    assert field.value == 286_500


def test_empty_document_abstains_on_everything() -> None:
    result = _extract([], doc_id="empty")
    assert result.doc_id == "empty"
    assert len(result.fields) == len(LineItem)
    assert all(f.abstained and f.evidence is None for f in result.fields)


def test_document_of_empty_pages_abstains() -> None:
    result = _extract(["", "   ", "\n\n"])
    assert all(f.abstained for f in result.fields)


def test_document_with_no_financial_content_abstains() -> None:
    page = (
        "Chapter 4: The Migration of the Arctic Tern\n"
        "The tern travels roughly 70,000 kilometres each year, tracing a\n"
        "figure of eight across the Atlantic between breeding seasons.\n"
    )
    result = _extract([page])
    assert all(f.abstained for f in result.fields)


def test_never_raises_on_hostile_input() -> None:
    hostile = ["\x00\x00", "(((", "$", "--,--", "cash", "1" * 500]
    result = _extract(hostile)
    assert result.extractor == "heuristic"


def test_extract_on_a_missing_pdf_abstains_instead_of_raising() -> None:
    result = HeuristicExtractor().extract(Path("no-such-file.pdf"), "missing")
    assert result.doc_id == "missing"
    assert all(f.abstained for f in result.fields)


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------


def test_every_quote_is_verbatim_on_the_page_it_cites() -> None:
    doc = Document.from_pages("alpine-2024", FULL_DOC)
    result = HeuristicExtractor().extract_document(doc, "alpine-2024")
    cited = [f for f in result.fields if f.evidence is not None]
    assert len(cited) == len(LineItem)
    for field in cited:
        assert field.evidence.quote in doc.page_text(field.evidence.page)


def test_quote_is_not_cleaned_up() -> None:
    """The verifier matches quotes literally, so whitespace must survive."""
    page = "Total revenue \t   412,905   \n"
    field = _extract([page]).get(LineItem.REVENUE_LTM)
    assert field.evidence.quote == "Total revenue \t   412,905   "


def test_extraction_is_deterministic() -> None:
    a = _extract(FULL_DOC)
    b = _extract(FULL_DOC)
    assert [(f.name, f.value, f.unit_scale) for f in a.fields] == [
        (f.name, f.value, f.unit_scale) for f in b.fields
    ]
