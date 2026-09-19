"""Core data contracts for Anchor.

Every other module in this package depends on these types. They are the
stable interface: extractors produce `Extraction`, the verifier annotates
`ExtractedField`, and scoring compares an `Extraction` against a `GoldenRecord`.

Design note
-----------
The model is only ever asked for *line items* -- never for a computed ratio.
Ratios are derived in `anchor.ratios` by plain Python so that arithmetic is
correct by construction and every remaining error is a sourcing error.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel
from pydantic import Field as PField


class LineItem(str, Enum):
    """The line items an extractor is asked to find.

    Deliberately *not* ratios. See module docstring.
    """

    REVENUE_LTM = "revenue_ltm"
    EBITDA_REPORTED = "ebitda_reported"
    EBITDA_ADDBACKS = "ebitda_addbacks"
    TOTAL_DEBT = "total_debt"
    CASH = "cash"
    INTEREST_EXPENSE = "interest_expense"
    PRINCIPAL_REPAYMENTS = "principal_repayments"
    CFADS = "cfads"


#: Line items whose printed sign is a presentation convention, not part of the
#: value.
#:
#: A statement prints interest and principal as outflows -- ``(842)``, ``-1,043``
#: -- because they reduce cash, not because the quantity is negative. Debt
#: service of 842 is what the document discloses, and that magnitude is what
#: every downstream consumer wants: DSCR divides by interest + principal, and a
#: negative leg turns the ratio into something that looks like a coverage
#: multiple and is not one.
#:
#: So the harness fixes one canonical representation and states it here:
#: **magnitude items are stored positive.** Extractors normalise to it, golden
#: labels follow it, `anchor.verify` reads a sign difference against the quote
#: as a convention rather than a mismatch, and `anchor.ratios` checks the
#: convention instead of assuming it.
#:
#: Deliberately not in this set:
#:
#: * ``CFADS`` -- a business can genuinely burn cash, and a negative CFADS is a
#:   reportable DSCR, not a misread sign.
#: * ``EBITDA_ADDBACKS`` -- an adjustment can genuinely be a deduction.
#: * ``REVENUE_LTM``, ``TOTAL_DEBT``, ``CASH``, ``EBITDA_REPORTED`` -- a negative
#:   here is either a real (if unusual) figure or an extraction error, and
#:   neither is a convention to normalise away.
MAGNITUDE_ITEMS: frozenset[LineItem] = frozenset(
    {LineItem.INTEREST_EXPENSE, LineItem.PRINCIPAL_REPAYMENTS}
)


class Evidence(BaseModel):
    """Where a value came from in the source document.

    `quote` must appear verbatim on `page`. The verifier enforces this; any
    field whose quote cannot be located is rejected rather than trusted.
    """

    page: int = PField(..., ge=1, description="1-indexed page number")
    quote: str = PField(..., min_length=1, description="Verbatim source text")


class ExtractedField(BaseModel):
    """One extracted line item, with its evidence and the model's confidence."""

    name: LineItem
    value: float | None = PField(
        None, description="None means the extractor abstained on this field"
    )
    unit_scale: float = PField(
        1.0, description="Multiplier applied to reach base currency units"
    )
    currency: str | None = None
    evidence: Evidence | None = None
    confidence: float = PField(
        0.0, ge=0.0, le=1.0, description="Extractor self-reported confidence"
    )

    # Set by anchor.verify, not by the extractor.
    grounded: bool | None = PField(
        None, description="True if quote was located on the cited page"
    )

    @property
    def abstained(self) -> bool:
        return self.value is None

    @property
    def scaled_value(self) -> float | None:
        return None if self.value is None else self.value * self.unit_scale


class Extraction(BaseModel):
    """An extractor's full output for one document."""

    doc_id: str
    fields: list[ExtractedField] = PField(default_factory=list)
    extractor: str = PField(..., description="Name of the extractor that ran")
    model: str | None = None
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    retries: int = 0

    def get(self, name: LineItem) -> ExtractedField | None:
        for f in self.fields:
            if f.name == name:
                return f
        return None


class GoldenField(BaseModel):
    """A hand-labelled ground-truth value for one line item."""

    name: LineItem
    value: float | None = PField(
        None, description="None means the document does not state this item"
    )
    page: int | None = None
    ambiguous: bool = PField(
        False,
        description=(
            "True when labelling required a judgement call -- e.g. a negotiated "
            "EBITDA add-back, or a DSCR whose debt service is not disclosed. "
            "These fields form the abstention test set."
        ),
    )
    note: str = ""


class GoldenRecord(BaseModel):
    """Ground truth for one source document."""

    doc_id: str
    source: str = PField("", description="Where the document came from")
    pages: int = 0
    scanned: bool = False
    fields: list[GoldenField] = PField(default_factory=list)

    def get(self, name: LineItem) -> GoldenField | None:
        for f in self.fields:
            if f.name == name:
                return f
        return None
