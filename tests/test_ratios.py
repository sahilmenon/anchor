"""Tests for `anchor.ratios`.

The arithmetic here is meant to be correct by construction, so these tests lean
hard on the adversarial cases -- ungrounded fields, unit scales, non-positive
denominators, half-disclosed debt service -- rather than on the happy path.
"""

from __future__ import annotations

import math

import pytest

from anchor.ratios import (
    RatioResult,
    adjusted_ebitda,
    compute_all,
    dscr,
    leverage,
    ltm_revenue,
)
from anchor.schema import Evidence, ExtractedField, Extraction, LineItem


def fld(
    name: LineItem,
    value: float | None,
    *,
    unit_scale: float = 1.0,
    grounded: bool | None = None,
) -> ExtractedField:
    return ExtractedField(
        name=name,
        value=value,
        unit_scale=unit_scale,
        grounded=grounded,
        evidence=Evidence(page=1, quote="x"),
        confidence=0.9,
    )


def ext(*fields: ExtractedField) -> Extraction:
    return Extraction(doc_id="d1", fields=list(fields), extractor="test")


FULL = ext(
    fld(LineItem.REVENUE_LTM, 1000.0),
    fld(LineItem.EBITDA_REPORTED, 200.0),
    fld(LineItem.EBITDA_ADDBACKS, 50.0),
    fld(LineItem.TOTAL_DEBT, 750.0),
    fld(LineItem.CASH, 100.0),
    fld(LineItem.INTEREST_EXPENSE, 40.0),
    fld(LineItem.PRINCIPAL_REPAYMENTS, 60.0),
    fld(LineItem.CFADS, 250.0),
)


# --- happy paths -----------------------------------------------------------


def test_ltm_revenue_passthrough():
    r = ltm_revenue(FULL)
    assert (r.value, r.abstained, r.reason) == (1000.0, False, "")
    assert r.inputs_used == [LineItem.REVENUE_LTM]
    assert r.name == "ltm_revenue"


def test_adjusted_ebitda_happy():
    r = adjusted_ebitda(FULL)
    assert r.value == pytest.approx(250.0)
    assert not r.abstained and r.reason == ""
    assert set(r.inputs_used) == {LineItem.EBITDA_REPORTED, LineItem.EBITDA_ADDBACKS}


def test_leverage_happy():
    r = leverage(FULL)
    # (750 - 100) / 250
    assert r.value == pytest.approx(2.6)
    assert not r.abstained
    assert LineItem.CASH in r.inputs_used and LineItem.TOTAL_DEBT in r.inputs_used


def test_dscr_happy_uses_cfads_not_ebitda():
    r = dscr(FULL)
    assert r.value == pytest.approx(2.5)  # 250 / (40 + 60)
    assert not r.abstained and r.reason == ""
    assert LineItem.CFADS in r.inputs_used
    assert LineItem.EBITDA_REPORTED not in r.inputs_used


def test_compute_all_returns_four_named_results():
    out = compute_all(FULL)
    assert set(out) == {"ltm_revenue", "adjusted_ebitda", "leverage", "dscr"}
    assert all(isinstance(v, RatioResult) for v in out.values())
    assert all(k == v.name for k, v in out.items())


def test_mapping_input_supported():
    data = {
        LineItem.EBITDA_REPORTED: 100.0,
        LineItem.EBITDA_ADDBACKS: 20.0,
        LineItem.TOTAL_DEBT: 500.0,
        LineItem.CASH: 20.0,
    }
    assert adjusted_ebitda(data).value == pytest.approx(120.0)
    assert leverage(data).value == pytest.approx(4.0)


# --- unit scaling ----------------------------------------------------------


def test_unit_scale_is_applied_everywhere():
    e = ext(
        fld(LineItem.REVENUE_LTM, 1.5, unit_scale=1_000_000.0),
        fld(LineItem.EBITDA_REPORTED, 200.0, unit_scale=1_000.0),
        fld(LineItem.EBITDA_ADDBACKS, 50.0, unit_scale=1_000.0),
        fld(LineItem.TOTAL_DEBT, 1_000.0, unit_scale=1_000.0),
        fld(LineItem.CASH, 250.0, unit_scale=1_000.0),
    )
    assert ltm_revenue(e).value == pytest.approx(1_500_000.0)
    assert adjusted_ebitda(e).value == pytest.approx(250_000.0)
    assert leverage(e).value == pytest.approx(3.0)  # (1_000k - 250k) / 250k


def test_mixed_unit_scales_do_not_cancel_out():
    """Debt in thousands against EBITDA in units must not be compared raw."""
    e = ext(
        fld(LineItem.EBITDA_REPORTED, 250_000.0, unit_scale=1.0),
        fld(LineItem.TOTAL_DEBT, 1_000.0, unit_scale=1_000.0),
        fld(LineItem.CASH, 0.0),
    )
    assert leverage(e).value == pytest.approx(4.0)


def test_zero_unit_scale_yields_zero_not_none():
    e = ext(fld(LineItem.REVENUE_LTM, 1000.0, unit_scale=0.0))
    r = ltm_revenue(e)
    assert r.value == 0.0 and not r.abstained


# --- abstention: missing fields -------------------------------------------


def test_ltm_revenue_missing_abstains():
    r = ltm_revenue(ext())
    assert r.abstained and r.value is None and "REVENUE_LTM" in r.reason


def test_ltm_revenue_value_none_abstains_not_zero():
    r = ltm_revenue(ext(fld(LineItem.REVENUE_LTM, None)))
    assert r.abstained and r.value is None


def test_adjusted_ebitda_missing_reported_abstains():
    r = adjusted_ebitda(ext(fld(LineItem.EBITDA_ADDBACKS, 50.0)))
    assert r.abstained and r.value is None and "EBITDA_REPORTED" in r.reason


def test_adjusted_ebitda_missing_addbacks_is_zero_but_noted():
    r = adjusted_ebitda(ext(fld(LineItem.EBITDA_REPORTED, 200.0)))
    assert r.value == pytest.approx(200.0)
    assert not r.abstained
    assert "ADDBACKS" in r.reason and "0" in r.reason
    assert r.inputs_used == [LineItem.EBITDA_REPORTED]


def test_abstained_addbacks_field_is_same_as_missing():
    r = adjusted_ebitda(
        ext(fld(LineItem.EBITDA_REPORTED, 200.0), fld(LineItem.EBITDA_ADDBACKS, None))
    )
    assert r.value == pytest.approx(200.0) and r.reason


def test_leverage_missing_debt_abstains():
    e = ext(fld(LineItem.EBITDA_REPORTED, 100.0), fld(LineItem.CASH, 10.0))
    r = leverage(e)
    assert r.abstained and "TOTAL_DEBT" in r.reason


def test_leverage_missing_cash_treated_as_zero_and_noted():
    e = ext(fld(LineItem.EBITDA_REPORTED, 100.0), fld(LineItem.TOTAL_DEBT, 300.0))
    r = leverage(e)
    assert r.value == pytest.approx(3.0)
    assert "CASH" in r.reason
    assert LineItem.CASH not in r.inputs_used


def test_leverage_propagates_ebitda_abstention():
    e = ext(fld(LineItem.TOTAL_DEBT, 300.0), fld(LineItem.CASH, 10.0))
    r = leverage(e)
    assert r.abstained and r.value is None
    assert "EBITDA_REPORTED" in r.reason


def test_leverage_note_carries_addback_assumption():
    e = ext(fld(LineItem.TOTAL_DEBT, 300.0), fld(LineItem.EBITDA_REPORTED, 100.0))
    r = leverage(e)
    assert "CASH" in r.reason and "ADDBACKS" in r.reason


# --- abstention: non-positive / degenerate denominators --------------------


@pytest.mark.parametrize("ebitda_value", [0.0, -1.0, -500.0, -0.0])
def test_leverage_abstains_on_non_positive_ebitda(ebitda_value):
    e = ext(
        fld(LineItem.TOTAL_DEBT, 300.0),
        fld(LineItem.CASH, 10.0),
        fld(LineItem.EBITDA_REPORTED, ebitda_value),
    )
    r = leverage(e)
    assert r.abstained and r.value is None
    assert "non-positive" in r.reason


def test_leverage_abstains_when_addbacks_drag_ebitda_negative():
    """Positive reported EBITDA plus a negative add-back can cross zero."""
    e = ext(
        fld(LineItem.TOTAL_DEBT, 300.0),
        fld(LineItem.EBITDA_REPORTED, 100.0),
        fld(LineItem.EBITDA_ADDBACKS, -150.0),
    )
    assert adjusted_ebitda(e).value == pytest.approx(-50.0)
    assert leverage(e).abstained


def test_leverage_negative_net_debt_is_a_real_answer():
    """Cash above debt is net cash -- a valid negative multiple, not an abstention."""
    e = ext(
        fld(LineItem.TOTAL_DEBT, 100.0),
        fld(LineItem.CASH, 300.0),
        fld(LineItem.EBITDA_REPORTED, 50.0),
    )
    r = leverage(e)
    assert r.value == pytest.approx(-4.0) and not r.abstained


def test_dscr_abstains_on_zero_debt_service():
    e = ext(
        fld(LineItem.CFADS, 100.0),
        fld(LineItem.INTEREST_EXPENSE, 0.0),
        fld(LineItem.PRINCIPAL_REPAYMENTS, 0.0),
    )
    r = dscr(e)
    assert r.abstained and r.value is None
    assert "non-positive" in r.reason


def test_dscr_abstains_on_offsetting_debt_service():
    """Interest and principal that net to zero must not divide by zero."""
    e = ext(
        fld(LineItem.CFADS, 100.0),
        fld(LineItem.INTEREST_EXPENSE, 40.0),
        fld(LineItem.PRINCIPAL_REPAYMENTS, -40.0),
    )
    r = dscr(e)
    assert r.abstained and r.value is None


# --- abstention: undisclosed debt service ---------------------------------


def test_dscr_abstains_when_principal_not_disclosed():
    """The common real case: interest is stated, scheduled principal is not."""
    e = ext(fld(LineItem.CFADS, 100.0), fld(LineItem.INTEREST_EXPENSE, 40.0))
    r = dscr(e)
    assert r.abstained and r.value is None
    assert "PRINCIPAL_REPAYMENTS" in r.reason
    assert "INTEREST_EXPENSE" not in r.reason


def test_dscr_abstains_when_interest_not_disclosed():
    e = ext(fld(LineItem.CFADS, 100.0), fld(LineItem.PRINCIPAL_REPAYMENTS, 40.0))
    r = dscr(e)
    assert r.abstained and "INTEREST_EXPENSE" in r.reason


def test_dscr_abstains_when_no_debt_service_at_all():
    r = dscr(ext(fld(LineItem.CFADS, 100.0)))
    assert r.abstained
    assert "INTEREST_EXPENSE" in r.reason and "PRINCIPAL_REPAYMENTS" in r.reason


def test_dscr_falls_back_to_ebitda_proxy():
    e = ext(
        fld(LineItem.EBITDA_REPORTED, 200.0),
        fld(LineItem.EBITDA_ADDBACKS, 50.0),
        fld(LineItem.INTEREST_EXPENSE, 50.0),
        fld(LineItem.PRINCIPAL_REPAYMENTS, 50.0),
    )
    r = dscr(e)
    assert r.value == pytest.approx(2.5)
    assert "proxy" in r.reason
    assert LineItem.EBITDA_REPORTED in r.inputs_used
    assert LineItem.CFADS not in r.inputs_used


def test_dscr_abstains_when_proxy_also_unavailable():
    e = ext(
        fld(LineItem.INTEREST_EXPENSE, 50.0),
        fld(LineItem.PRINCIPAL_REPAYMENTS, 50.0),
    )
    r = dscr(e)
    assert r.abstained and "CFADS" in r.reason and "proxy" in r.reason


def test_dscr_proxy_does_not_hide_negative_cfads():
    """Negative CFADS is a real, reportable DSCR -- no proxy substitution."""
    e = ext(
        fld(LineItem.CFADS, -100.0),
        fld(LineItem.EBITDA_REPORTED, 500.0),
        fld(LineItem.INTEREST_EXPENSE, 50.0),
        fld(LineItem.PRINCIPAL_REPAYMENTS, 50.0),
    )
    r = dscr(e)
    assert r.value == pytest.approx(-1.0) and not r.abstained


def test_dscr_zero_cfads_is_computed_not_abstained():
    e = ext(
        fld(LineItem.CFADS, 0.0),
        fld(LineItem.EBITDA_REPORTED, 900.0),
        fld(LineItem.INTEREST_EXPENSE, 50.0),
        fld(LineItem.PRINCIPAL_REPAYMENTS, 50.0),
    )
    r = dscr(e)
    assert r.value == 0.0 and not r.abstained and r.reason == ""


# --- grounded=False rejection ---------------------------------------------


@pytest.mark.parametrize(
    "target",
    [
        LineItem.REVENUE_LTM,
        LineItem.EBITDA_REPORTED,
        LineItem.EBITDA_ADDBACKS,
        LineItem.TOTAL_DEBT,
        LineItem.CASH,
        LineItem.INTEREST_EXPENSE,
        LineItem.PRINCIPAL_REPAYMENTS,
        LineItem.CFADS,
    ],
)
def test_any_ungrounded_field_forces_an_abstention(target):
    """Poisoning one field must abstain at least one ratio that consumes it."""
    fields = [
        fld(f.name, f.value, grounded=(False if f.name == target else True))
        for f in FULL.fields
    ]
    results = compute_all(ext(*fields))
    poisoned = [r for r in results.values() if r.abstained]
    assert poisoned, f"{target} rejection was swallowed"
    assert all("grounded=False" in r.reason or "unavailable" in r.reason for r in poisoned)


def test_ungrounded_addbacks_do_not_default_to_zero():
    """A rejected add-back is unusable, not absent -- computing 200 would launder it."""
    e = ext(
        fld(LineItem.EBITDA_REPORTED, 200.0, grounded=True),
        fld(LineItem.EBITDA_ADDBACKS, 50.0, grounded=False),
    )
    r = adjusted_ebitda(e)
    assert r.abstained and r.value is None
    assert "EBITDA_ADDBACKS" in r.reason and "grounded=False" in r.reason


def test_ungrounded_cash_does_not_default_to_zero():
    e = ext(
        fld(LineItem.TOTAL_DEBT, 300.0, grounded=True),
        fld(LineItem.CASH, 100.0, grounded=False),
        fld(LineItem.EBITDA_REPORTED, 100.0, grounded=True),
    )
    r = leverage(e)
    assert r.abstained and "CASH" in r.reason


def test_grounded_none_is_allowed():
    """None means 'not verified yet', which must still compute."""
    e = ext(
        fld(LineItem.TOTAL_DEBT, 300.0, grounded=None),
        fld(LineItem.CASH, 100.0, grounded=None),
        fld(LineItem.EBITDA_REPORTED, 100.0, grounded=None),
    )
    assert leverage(e).value == pytest.approx(2.0)


def test_grounded_true_computes_normally():
    fields = [fld(f.name, f.value, grounded=True) for f in FULL.fields]
    out = compute_all(ext(*fields))
    assert not any(r.abstained for r in out.values())


def test_ungrounded_cfads_does_not_silently_fall_back_to_proxy():
    """Falling back here would substitute EBITDA for a value we know is bad."""
    e = ext(
        fld(LineItem.CFADS, 100.0, grounded=False),
        fld(LineItem.EBITDA_REPORTED, 500.0, grounded=True),
        fld(LineItem.INTEREST_EXPENSE, 50.0, grounded=True),
        fld(LineItem.PRINCIPAL_REPAYMENTS, 50.0, grounded=True),
    )
    r = dscr(e)
    assert r.abstained and "CFADS" in r.reason and "grounded=False" in r.reason


# --- robustness ------------------------------------------------------------


def test_nothing_raises_on_an_empty_extraction():
    out = compute_all(ext())
    assert all(r.abstained and r.value is None and r.reason for r in out.values())


def test_nothing_raises_on_an_empty_mapping():
    out = compute_all({})
    assert all(r.abstained for r in out.values())


def test_all_fields_abstained_abstains_everything():
    e = ext(*[fld(n, None) for n in LineItem])
    out = compute_all(e)
    assert all(r.abstained for r in out.values())


def test_abstained_results_carry_no_inputs():
    out = compute_all(ext())
    assert all(r.inputs_used == [] for r in out.values())


def test_results_are_independent_objects():
    a, b = compute_all(FULL), compute_all(FULL)
    a["leverage"].inputs_used.append(LineItem.CFADS)
    assert LineItem.CFADS not in b["leverage"].inputs_used


def test_duplicate_fields_use_the_first_occurrence():
    """`Extraction.get` returns the first match; ratios must not double-count."""
    e = ext(fld(LineItem.REVENUE_LTM, 10.0), fld(LineItem.REVENUE_LTM, 99.0))
    assert ltm_revenue(e).value == pytest.approx(10.0)


# --- float precision -------------------------------------------------------


def test_float_precision_addbacks_sum():
    e = ext(fld(LineItem.EBITDA_REPORTED, 0.1), fld(LineItem.EBITDA_ADDBACKS, 0.2))
    r = adjusted_ebitda(e)
    assert r.value == pytest.approx(0.3, rel=1e-12)
    assert r.value != 0.3  # binary floating point, not decimal -- documented, not hidden


def test_float_precision_leverage_is_exact_division():
    e = ext(
        fld(LineItem.TOTAL_DEBT, 1.0),
        fld(LineItem.CASH, 0.0),
        fld(LineItem.EBITDA_REPORTED, 3.0),
    )
    assert leverage(e).value == pytest.approx(1.0 / 3.0, rel=1e-15)


def test_float_precision_survives_large_unit_scales():
    e = ext(
        fld(LineItem.TOTAL_DEBT, 1_234.567, unit_scale=1_000_000.0),
        fld(LineItem.CASH, 234.567, unit_scale=1_000_000.0),
        fld(LineItem.EBITDA_REPORTED, 500.0, unit_scale=1_000_000.0),
    )
    assert leverage(e).value == pytest.approx(2.0, rel=1e-12)


def test_no_nan_or_inf_escapes_a_computed_ratio():
    for r in compute_all(FULL).values():
        assert r.value is not None
        assert math.isfinite(r.value)
