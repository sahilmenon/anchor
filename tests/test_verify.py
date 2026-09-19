"""Adversarial tests for `anchor.verify`.

The module's job is to reject figures that are not supported by the document,
so most of these tests are written from the attacker's side: a citation that
looks right and is not. The happy path gets one test; the ways of faking it
get the rest.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from anchor.extractors.base import Document
from anchor.schema import Evidence, ExtractedField, Extraction, LineItem
from anchor.verify import (
    grounding_rate,
    normalise,
    parse_number,
    quote_on_page,
    value_in_quote,
    verify_extraction,
    verify_field,
)

# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------

PAGE_1 = "Consolidated Statement of Operations\nRevenue for the period was $1,234 in the aggregate."
PAGE_2 = "Total debt of 5,678 and cash of 910 as at 31 December 2024."
PAGE_3 = "Interest expense was (1,234) for the year."


def make_doc(pages: list[str] | None = None) -> Document:
    return Document.from_pages("doc-1", pages if pages is not None else [PAGE_1, PAGE_2, PAGE_3])


def field(
    name: LineItem = LineItem.REVENUE_LTM,
    value: float | None = 1234.0,
    page: int | None = 1,
    quote: str | None = "Revenue for the period was $1,234 in the aggregate.",
    unit_scale: float = 1.0,
) -> ExtractedField:
    ev = None
    if page is not None and quote is not None:
        ev = Evidence(page=page, quote=quote)
    return ExtractedField(
        name=name, value=value, unit_scale=unit_scale, evidence=ev, confidence=0.9
    )


def extraction(fields: list[ExtractedField]) -> Extraction:
    return Extraction(doc_id="doc-1", fields=fields, extractor="test")


# --------------------------------------------------------------------------
# normalise
# --------------------------------------------------------------------------


def test_normalise_collapses_whitespace_and_casefolds():
    assert normalise("  Total   Debt\n\tOutstanding ") == "total debt outstanding"


def test_normalise_maps_nbsp_and_dashes():
    assert normalise("1,234 – 5,678") == "1,234 - 5,678"
    assert normalise("−1234") == "-1234"


def test_normalise_maps_curly_quotes_and_ligatures():
    assert normalise("the Company’s “EBITDA”") == 'the company\'s "ebitda"'
    assert normalise("deﬁcit") == "deficit"


def test_normalise_drops_zero_width_and_soft_hyphens():
    assert normalise("EBIT­DA​margin") == "ebitdamargin"


def test_normalise_empty():
    assert normalise("") == ""


# --------------------------------------------------------------------------
# parse_number
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1234", 1234.0),
        ("1,234,567", 1234567.0),
        ("1,234.56", 1234.56),
        ("-1,234", -1234.0),
        ("1,234-", -1234.0),
        ("+99", 99.0),
        ("0", 0.0),
        ("0.5", 0.5),
    ],
)
def test_parse_number_plain(raw, expected):
    assert parse_number(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("(1,234)", -1234.0),
        ("(1,234.5)", -1234.5),
        ("($1,234)", -1234.0),
        ("(0.5)", -0.5),
    ],
)
def test_parse_number_parenthesised_negative(raw, expected):
    """Accounting parentheses mean negative -- the single most common way to
    get a sign wrong when reading a filing."""
    assert parse_number(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("$1,234", 1234.0),
        ("£1,234", 1234.0),
        ("€1,234", 1234.0),
        ("A$1,234", 1234.0),
        ("US$ 1,234", 1234.0),
        ("$ 1,234.50", 1234.5),
    ],
)
def test_parse_number_currency_symbols(raw, expected):
    assert parse_number(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1.2m", 1_200_000.0),
        ("1.2 million", 1_200_000.0),
        ("3.4bn", 3_400_000_000.0),
        ("500k", 500_000.0),
        ("2mm", 2_000_000.0),
        ("$1.5bn", 1_500_000_000.0),
        ("(2.5m)", -2_500_000.0),
    ],
)
def test_parse_number_magnitude_suffixes(raw, expected):
    assert parse_number(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["", "   ", "n/a", "nil", "--", "EBITDA", "1,234 (unaudited)", "note3", "1234abc", "$"],
)
def test_parse_number_unparseable(raw):
    """Whole-string parse: prose around a figure is a refusal, not a guess."""
    assert parse_number(raw) is None


def test_parse_number_does_not_eat_following_word():
    assert parse_number("1,234 members") is None


# --------------------------------------------------------------------------
# quote_on_page
# --------------------------------------------------------------------------


def test_quote_on_page_exact():
    assert quote_on_page("Revenue for the period was $1,234 in the aggregate.", PAGE_1)


def test_quote_on_page_case_and_spacing_insensitive():
    assert quote_on_page("REVENUE   for the   period was $1,234 in the aggregate.", PAGE_1)


def test_quote_on_page_unicode_mangled_pdf_text():
    """The page text is what a PDF layer produced: NBSPs and an en dash."""
    page = "Total debt of 5,678 – net of cash"
    assert quote_on_page("Total debt of 5,678 - net of cash", page)


def test_quote_on_page_whitespace_fallback_for_broken_spans():
    """PDFs split words across text spans; the fallback exists for exactly this."""
    page = "Total de bt of 5, 678 as at year end"
    assert quote_on_page("Total debt of 5,678", page)


def test_quote_on_page_rejects_absent_quote():
    assert not quote_on_page("Revenue was $9,999", PAGE_1)


def test_quote_on_page_rejects_empty_inputs():
    assert not quote_on_page("", PAGE_1)
    assert not quote_on_page("anything", "")


# --------------------------------------------------------------------------
# value_in_quote
# --------------------------------------------------------------------------


def test_value_in_quote_exact():
    assert value_in_quote(1234.0, "Revenue was $1,234 for the period")


def test_value_in_quote_rejects_absent_number():
    assert not value_in_quote(9999.0, "Revenue was $1,234 for the period")


def test_value_in_quote_no_numbers_at_all():
    assert not value_in_quote(1234.0, "Revenue increased materially year on year")


def test_value_in_quote_respects_tolerance():
    assert value_in_quote(1235.0, "Revenue was 1,234.6")
    assert not value_in_quote(1300.0, "Revenue was 1,234.6")


def test_value_in_quote_tolerance_is_configurable():
    assert not value_in_quote(1300.0, "Revenue was 1,234")
    assert value_in_quote(1300.0, "Revenue was 1,234", rel_tol=0.10)


def test_value_in_quote_parenthesised_negative():
    assert value_in_quote(-1234.0, "Interest expense was (1,234) for the year")
    assert not value_in_quote(1234.0, "Interest expense was (1,234) for the year")


def test_value_in_quote_unit_scale_table_units():
    """Table prints thousands; extractor reports the printed figure."""
    assert value_in_quote(1234.0, "Revenue was 1,234", unit_scale=1000.0)


def test_value_in_quote_unit_scale_full_units():
    """Same scale, but the quote spells the figure out in full."""
    assert value_in_quote(1234.0, "Revenue was 1,234,000", unit_scale=1000.0)


def test_value_in_quote_unit_scale_does_not_excuse_a_wrong_number():
    assert not value_in_quote(999.0, "Revenue was 1,234", unit_scale=1000.0)


def test_value_in_quote_picks_any_number_in_the_quote():
    assert value_in_quote(910.0, "Total debt of 5,678 and cash of 910")


def test_value_in_quote_zero():
    assert value_in_quote(0.0, "Principal repayments of 0 in the period")
    assert not value_in_quote(0.0, "Principal repayments of 5,678")


# --------------------------------------------------------------------------
# verify_field
# --------------------------------------------------------------------------


def test_verify_field_correct_citation():
    out = verify_field(field(), make_doc())
    assert out.grounded is True


def test_verify_field_does_not_mutate_input():
    f = field()
    verify_field(f, make_doc())
    assert f.grounded is None


def test_verify_field_hallucinated_citation_right_page_wrong_number():
    """The quote is verbatim and on the cited page -- but it does not contain
    the number reported. This is the failure the module exists to catch."""
    f = field(value=9_999.0)
    out = verify_field(f, make_doc())
    assert quote_on_page(f.evidence.quote, make_doc().page_text(1))
    assert out.grounded is False


def test_verify_field_quote_on_a_different_page():
    f = field(value=5678.0, page=1, quote="Total debt of 5,678")
    assert verify_field(f, make_doc()).grounded is False
    # ... and the same field cited correctly is grounded.
    assert (
        verify_field(field(value=5678.0, page=2, quote="Total debt of 5,678"), make_doc()).grounded
        is True
    )


def test_verify_field_page_beyond_document():
    f = field(page=99)
    assert verify_field(f, make_doc()).grounded is False


def test_verify_field_page_zero_is_rejected_by_the_schema():
    """Evidence pages are 1-indexed; page 0 cannot even be constructed."""
    with pytest.raises(ValidationError):
        Evidence(page=0, quote="Revenue")


def test_verify_field_page_zero_if_schema_is_bypassed():
    """Defence in depth: a field built with model_construct skips validation,
    so verify must still range-check rather than trust the page number."""
    ev = Evidence.model_construct(
        page=0, quote="Revenue for the period was $1,234 in the aggregate."
    )
    f = ExtractedField(name=LineItem.REVENUE_LTM, value=1234.0, confidence=0.9)
    f.evidence = ev
    assert verify_field(f, make_doc()).grounded is False


def test_verify_field_negative_page():
    ev = Evidence.model_construct(page=-1, quote="Revenue")
    f = ExtractedField(name=LineItem.REVENUE_LTM, value=1234.0, confidence=0.9)
    f.evidence = ev
    assert verify_field(f, make_doc()).grounded is False


def test_verify_field_no_evidence_is_not_grounded():
    f = field(page=None, quote=None)
    assert f.evidence is None
    assert verify_field(f, make_doc()).grounded is False


def test_verify_field_abstained_is_none_not_false():
    """Abstention is a legitimate outcome, so it must not be scored as a
    grounding failure."""
    f = field(value=None, page=None, quote=None)
    assert verify_field(f, make_doc()).grounded is None


def test_verify_field_abstained_with_evidence_still_none():
    f = field(value=None)
    assert verify_field(f, make_doc()).grounded is None


def test_verify_field_paraphrase_is_rejected():
    f = field(quote="Revenue for the period amounted to 1,234 dollars in total")
    assert verify_field(f, make_doc()).grounded is False


def test_verify_field_survives_pdf_mangled_page_text():
    doc = Document.from_pages("d", ["Revenue for the per iod was $1,​234 in the agg regate."])
    f = field(quote="Revenue for the period was $1,234 in the aggregate.")
    assert verify_field(f, doc).grounded is True


def test_verify_field_with_unit_scale():
    doc = Document.from_pages("d", ["$ in thousands. Total debt 5,678"])
    f = field(
        name=LineItem.TOTAL_DEBT, value=5678.0, page=1, quote="Total debt 5,678", unit_scale=1000.0
    )
    out = verify_field(f, doc)
    assert out.grounded is True
    assert out.scaled_value == 5_678_000.0


def test_verify_field_negative_value_from_parenthesised_quote():
    f = field(
        name=LineItem.INTEREST_EXPENSE,
        value=-1234.0,
        page=3,
        quote="Interest expense was (1,234) for the year.",
    )
    assert verify_field(f, make_doc()).grounded is True


def test_verify_field_sign_flip_is_caught_on_an_ordinary_item():
    """Outside MAGNITUDE_ITEMS a sign flip is a real disagreement with evidence."""
    f = field(
        name=LineItem.EBITDA_ADDBACKS,
        value=1234.0,
        page=3,
        quote="Interest expense was (1,234) for the year.",
    )
    assert verify_field(f, make_doc()).grounded is False


def test_verify_field_accepts_a_magnitude_against_a_parenthesised_quote():
    """Debt-service items are stored positive by convention.

    Reading the convention correctly must not cost the field its grounding, or
    the harness punishes an extractor for doing the right thing.
    """
    f = field(
        name=LineItem.INTEREST_EXPENSE,
        value=1234.0,
        page=3,
        quote="Interest expense was (1,234) for the year.",
    )
    assert verify_field(f, make_doc()).grounded is True


def test_magnitude_matching_does_not_relax_the_hallucination_check():
    """Ignoring the sign must not start accepting figures that are not there."""
    f = field(
        name=LineItem.INTEREST_EXPENSE,
        value=9999.0,
        page=3,
        quote="Interest expense was (1,234) for the year.",
    )
    assert verify_field(f, make_doc()).grounded is False


def test_a_units_header_cannot_ground_a_reported_zero():
    """A scale marker contains three zeroes and is not a figure.

    Without this, any page carrying a units declaration could ground a
    fabricated zero -- and a zero reads as a real answer all the way downstream.
    """
    assert not value_in_quote(0.0, "All amounts in A$'000")
    assert not value_in_quote(0.0, "Amounts in $ '000 unless otherwise stated")


def test_verify_field_empty_document():
    f = field()
    assert verify_field(f, Document.from_pages("empty", [])).grounded is False


# --------------------------------------------------------------------------
# verify_extraction
# --------------------------------------------------------------------------


def test_verify_extraction_verifies_every_field_and_copies():
    ex = extraction(
        [
            field(),
            field(name=LineItem.TOTAL_DEBT, value=5678.0, page=2, quote="Total debt of 5,678"),
            field(name=LineItem.CASH, value=9_999.0, page=2, quote="cash of 910"),
            field(name=LineItem.CFADS, value=None, page=None, quote=None),
        ]
    )
    out = verify_extraction(ex, make_doc())

    assert [f.grounded for f in out.fields] == [True, True, False, None]
    assert [f.grounded for f in ex.fields] == [None, None, None, None]
    assert out.doc_id == ex.doc_id and out.extractor == ex.extractor
    assert out.get(LineItem.CASH).grounded is False


def test_verify_extraction_empty_fields():
    out = verify_extraction(extraction([]), make_doc())
    assert out.fields == []


# --------------------------------------------------------------------------
# grounding_rate
# --------------------------------------------------------------------------


def test_grounding_rate_all_grounded():
    ex = verify_extraction(
        extraction(
            [
                field(),
                field(name=LineItem.TOTAL_DEBT, value=5678.0, page=2, quote="Total debt of 5,678"),
            ]
        ),
        make_doc(),
    )
    assert grounding_rate(ex) == 1.0


def test_grounding_rate_half():
    ex = verify_extraction(
        extraction(
            [field(), field(name=LineItem.CASH, value=9_999.0, page=2, quote="cash of 910")]
        ),
        make_doc(),
    )
    assert grounding_rate(ex) == 0.5


def test_grounding_rate_excludes_abstentions_from_denominator():
    ex = verify_extraction(
        extraction([field(), field(name=LineItem.CFADS, value=None, page=None, quote=None)]),
        make_doc(),
    )
    assert grounding_rate(ex) == 1.0


def test_grounding_rate_empty_extraction_is_zero():
    """Documented choice: nothing claimed means nothing grounded, so 0.0 --
    otherwise a blank answer would score a perfect grounding rate."""
    assert grounding_rate(extraction([])) == 0.0


def test_grounding_rate_all_abstained_is_zero():
    ex = verify_extraction(
        extraction(
            [
                field(value=None, page=None, quote=None),
                field(name=LineItem.CASH, value=None, page=None, quote=None),
            ]
        ),
        make_doc(),
    )
    assert grounding_rate(ex) == 0.0


def test_grounding_rate_treats_unverified_none_as_not_grounded():
    """A field that was never run through verify_field has grounded=None; it
    must not be counted as grounded just because it is not False."""
    ex = extraction([field()])
    assert ex.fields[0].grounded is None
    assert grounding_rate(ex) == 0.0


def test_word_magnitude_in_quote_is_honoured():
    """A quote reading "$1,234 thousand" states 1,234,000. An extractor
    reporting 1234 is only grounded if it also declares unit_scale=1000."""
    quote = "Revenue for the period was $1,234 thousand."
    assert value_in_quote(1234.0, quote, unit_scale=1000.0)
    assert not value_in_quote(1234.0, quote, unit_scale=1.0)
    assert value_in_quote(1_234_000.0, quote)
