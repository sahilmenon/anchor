"""Credit ratios, derived in plain Python.

The extractor is only ever asked for line items; every ratio in this module is
computed here so that the arithmetic is correct by construction and any
remaining error is a *sourcing* error, attributable to a specific line item.

Two rules follow from that, and they drive the whole file:

1. A ratio is never guessed. If an input is absent, abstained, or rejected by
   the verifier, the ratio abstains and says which input killed it. Abstention
   is a first-class result here, not a failure -- a credit analyst who says
   "the statements don't disclose principal repayments" is *right*, and the
   harness scores them as right.
2. A missing input is never silently coerced to zero. The two exceptions
   (EBITDA add-backs, cash) are genuine accounting defaults -- no add-backs
   means no add-backs -- and even those are recorded in `reason` so a reader
   can tell a real zero from an assumed one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from anchor.schema import Extraction, LineItem

__all__ = [
    "RatioResult",
    "RatioInput",
    "ltm_revenue",
    "adjusted_ebitda",
    "leverage",
    "dscr",
    "compute_all",
]

#: Either a full `Extraction` or a bare mapping of line item -> already-scaled
#: value. The mapping form exists for tests and for golden records, which carry
#: no unit scale and no grounding.
RatioInput = Extraction | Mapping[LineItem, float | None]


@dataclass
class RatioResult:
    """One computed ratio, or one reasoned refusal to compute it."""

    name: str
    value: float | None = None
    inputs_used: list[LineItem] = field(default_factory=list)
    abstained: bool = False
    reason: str = ""


# --- input resolution ------------------------------------------------------

# Resolution outcomes. `REJECTED` is distinct from `MISSING` because the
# difference matters downstream: a rejected field was present in the document
# but failed quote verification, so defaulting it to zero would launder a
# hallucination into a number.
_OK = "ok"
_MISSING = "missing"
_REJECTED = "rejected"


def _resolve(data: RatioInput, name: LineItem) -> tuple[float | None, str]:
    """Return the usable scaled value for `name` plus why it is unusable.

    Uses `scaled_value`, never the raw `value` -- an extractor reporting
    "1,250" in thousands is only correct once multiplied by its unit scale.
    """
    if isinstance(data, Extraction):
        fld = data.get(name)
        if fld is None:
            return None, _MISSING
        # `grounded is False` only; None means "not yet verified", which is
        # allowed to flow through.
        if fld.grounded is False:
            return None, _REJECTED
        scaled = fld.scaled_value
        return (None, _MISSING) if scaled is None else (scaled, _OK)

    raw = data.get(name)
    return (None, _MISSING) if raw is None else (float(raw), _OK)


def _label(name: LineItem) -> str:
    return name.name


def _rejected_reason(name: LineItem) -> str:
    return f"{_label(name)} failed evidence verification (grounded=False) and is unusable"


def _missing_reason(name: LineItem) -> str:
    return f"{_label(name)} is missing or abstained"


# --- ratios ----------------------------------------------------------------


def ltm_revenue(data: RatioInput) -> RatioResult:
    """LTM revenue. A passthrough, but one that still obeys the abstention rules."""
    value, status = _resolve(data, LineItem.REVENUE_LTM)
    if status == _REJECTED:
        return RatioResult(
            "ltm_revenue", None, [], True, _rejected_reason(LineItem.REVENUE_LTM)
        )
    if status == _MISSING:
        return RatioResult(
            "ltm_revenue", None, [], True, _missing_reason(LineItem.REVENUE_LTM)
        )
    return RatioResult("ltm_revenue", value, [LineItem.REVENUE_LTM], False, "")


def adjusted_ebitda(data: RatioInput) -> RatioResult:
    """Reported EBITDA plus add-backs.

    Absent add-backs mean zero add-backs -- that is the accounting default, not
    a guess -- but it is noted so nobody mistakes an unadjusted figure for an
    adjusted one.
    """
    reported, r_status = _resolve(data, LineItem.EBITDA_REPORTED)
    if r_status == _REJECTED:
        return RatioResult(
            "adjusted_ebitda", None, [], True, _rejected_reason(LineItem.EBITDA_REPORTED)
        )
    if r_status == _MISSING:
        return RatioResult(
            "adjusted_ebitda", None, [], True, _missing_reason(LineItem.EBITDA_REPORTED)
        )

    addbacks, a_status = _resolve(data, LineItem.EBITDA_ADDBACKS)
    if a_status == _REJECTED:
        return RatioResult(
            "adjusted_ebitda", None, [], True, _rejected_reason(LineItem.EBITDA_ADDBACKS)
        )

    inputs = [LineItem.EBITDA_REPORTED]
    if a_status == _OK:
        assert addbacks is not None
        return RatioResult(
            "adjusted_ebitda",
            reported + addbacks,
            inputs + [LineItem.EBITDA_ADDBACKS],
            False,
            "",
        )
    return RatioResult(
        "adjusted_ebitda",
        reported,
        inputs,
        False,
        "EBITDA_ADDBACKS not disclosed; treated as 0",
    )


def leverage(data: RatioInput) -> RatioResult:
    """Net debt / adjusted EBITDA.

    Abstains on non-positive EBITDA: a negative or zero denominator turns
    leverage into a number that looks like a multiple and means nothing. "Not
    meaningful" is the correct credit answer, so say that instead.
    """
    debt, d_status = _resolve(data, LineItem.TOTAL_DEBT)
    if d_status == _REJECTED:
        return RatioResult("leverage", None, [], True, _rejected_reason(LineItem.TOTAL_DEBT))
    if d_status == _MISSING:
        return RatioResult("leverage", None, [], True, _missing_reason(LineItem.TOTAL_DEBT))

    cash, c_status = _resolve(data, LineItem.CASH)
    if c_status == _REJECTED:
        return RatioResult("leverage", None, [], True, _rejected_reason(LineItem.CASH))

    ebitda = adjusted_ebitda(data)
    if ebitda.abstained:
        return RatioResult(
            "leverage",
            None,
            [],
            True,
            f"adjusted EBITDA unavailable: {ebitda.reason}",
        )
    assert ebitda.value is not None
    if ebitda.value <= 0:
        return RatioResult(
            "leverage",
            None,
            [],
            True,
            (
                f"adjusted EBITDA is non-positive ({ebitda.value:g}); "
                "leverage is not meaningful"
            ),
        )

    notes: list[str] = []
    inputs = [LineItem.TOTAL_DEBT]
    if c_status == _OK:
        assert cash is not None
        net_debt = debt - cash
        inputs.append(LineItem.CASH)
    else:
        net_debt = debt
        notes.append("CASH not disclosed; treated as 0")
    inputs.extend(ebitda.inputs_used)
    if ebitda.reason:
        notes.append(ebitda.reason)

    return RatioResult("leverage", net_debt / ebitda.value, inputs, False, "; ".join(notes))


def dscr(data: RatioInput) -> RatioResult:
    """CFADS / (interest + principal repayments).

    Debt service is the hard part. Statements routinely disclose interest but
    not scheduled principal, and defaulting the missing leg to zero inflates
    DSCR in exactly the direction that flatters a borrower. So a partially
    disclosed debt service abstains, not approximates.

    CFADS itself is often absent; adjusted EBITDA is the accepted proxy, used
    here only with an explicit note that it is a proxy.
    """
    cfads, cf_status = _resolve(data, LineItem.CFADS)
    if cf_status == _REJECTED:
        return RatioResult("dscr", None, [], True, _rejected_reason(LineItem.CFADS))

    notes: list[str] = []
    inputs: list[LineItem] = []

    if cf_status == _OK:
        numerator = cfads
        inputs.append(LineItem.CFADS)
    else:
        proxy = adjusted_ebitda(data)
        if proxy.abstained:
            return RatioResult(
                "dscr",
                None,
                [],
                True,
                f"CFADS not disclosed and adjusted EBITDA proxy unavailable: {proxy.reason}",
            )
        numerator = proxy.value
        inputs.extend(proxy.inputs_used)
        notes.append("CFADS not disclosed; adjusted EBITDA used as proxy")
        if proxy.reason:
            notes.append(proxy.reason)

    interest, i_status = _resolve(data, LineItem.INTEREST_EXPENSE)
    if i_status == _REJECTED:
        return RatioResult("dscr", None, [], True, _rejected_reason(LineItem.INTEREST_EXPENSE))
    principal, p_status = _resolve(data, LineItem.PRINCIPAL_REPAYMENTS)
    if p_status == _REJECTED:
        return RatioResult(
            "dscr", None, [], True, _rejected_reason(LineItem.PRINCIPAL_REPAYMENTS)
        )

    absent = [
        _label(n)
        for n, s in (
            (LineItem.INTEREST_EXPENSE, i_status),
            (LineItem.PRINCIPAL_REPAYMENTS, p_status),
        )
        if s != _OK
    ]
    if absent:
        return RatioResult(
            "dscr",
            None,
            [],
            True,
            (
                f"debt service not disclosed ({', '.join(absent)} missing); "
                "assuming zero would overstate DSCR"
            ),
        )

    assert interest is not None and principal is not None
    debt_service = interest + principal
    if debt_service <= 0:
        return RatioResult(
            "dscr",
            None,
            [],
            True,
            f"debt service is non-positive ({debt_service:g}); DSCR is undefined",
        )

    assert numerator is not None
    inputs.extend([LineItem.INTEREST_EXPENSE, LineItem.PRINCIPAL_REPAYMENTS])
    return RatioResult("dscr", numerator / debt_service, inputs, False, "; ".join(notes))


def compute_all(data: RatioInput) -> dict[str, RatioResult]:
    """Every ratio Anchor scores, keyed by name. Never raises."""
    return {
        r.name: r
        for r in (ltm_revenue(data), adjusted_ebitda(data), leverage(data), dscr(data))
    }
