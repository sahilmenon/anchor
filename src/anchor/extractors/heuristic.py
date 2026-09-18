"""A deterministic, keyword-anchored baseline extractor.

Why this exists
---------------
Two reasons, both about trust rather than accuracy:

1. **The harness must be runnable by anyone.** No API key, no network, no
   non-determinism. Every other part of Anchor -- the verifier, the ratio
   layer, scoring, the CLI -- can be exercised end to end against this
   extractor, so a contributor can prove the plumbing works before spending a
   cent on tokens.
2. **The LLM needs a floor to clear.** "The model extracted 6 of 8 fields" is
   meaningless without a baseline. A regex that anchors on labels and grabs the
   nearest number is roughly what a careful person would write in an afternoon;
   if a frontier model cannot beat it on a real credit document, that is the
   finding.

Design stance: *honest, not clever.* This extractor has no notion of financial
statements -- it cannot tell a prior-year comparative column from the current
one, cannot add a "current" and "non-current" borrowings row together, and will
happily anchor on the first "revenue" it sees even if it sits in a segment note.
Those are real limits, so the confidence it reports is capped low (0.6) and it
abstains rather than guesses whenever a label is missing or no plausible number
sits near it. Abstention is a scored outcome in Anchor, not a failure.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import NamedTuple

from anchor.extractors.base import Document
from anchor.schema import Evidence, ExtractedField, Extraction, LineItem

# --------------------------------------------------------------------------
# Confidence tiers
#
# Deliberately low. A regex that matched the exact canonical phrase for a line
# item ("total debt") is more trustworthy than one that matched a broad
# fallback ("debt"), but neither deserves to outrank a careful reader, and both
# are blind to the column/period ambiguity that dominates real errors.
# --------------------------------------------------------------------------
CONF_EXACT = 0.6
CONF_FUZZY = 0.45
#: Subtracted when the number was found on the line *after* the label, which
#: happens with wrapped statement rows but is also how mis-anchoring happens.
CONF_WRAP_PENALTY = 0.1


class _Label(NamedTuple):
    """One label pattern for one line item."""

    pattern: re.Pattern[str]
    confidence: float


def _compile(phrase: str, confidence: float) -> _Label:
    """Compile a human-written label phrase into a tolerant regex.

    Runs of whitespace in the phrase match runs of whitespace in the document,
    because PDF text extraction routinely turns a single space into several
    when it reconstructs a table row.
    """
    escaped = r"\s+".join(re.escape(word) for word in phrase.split())
    return _Label(re.compile(r"\b" + escaped + r"\b", re.IGNORECASE), confidence)


def _labels(exact: list[str], fuzzy: list[str]) -> list[_Label]:
    return [_compile(p, CONF_EXACT) for p in exact] + [
        _compile(p, CONF_FUZZY) for p in fuzzy
    ]


#: Label patterns per line item.
#:
#: The *exact* list holds phrases that are unambiguous in a credit context; the
#: *fuzzy* list holds broad fallbacks that catch more documents at the cost of
#: precision (and so report lower confidence). Overlap between items -- "cash"
#: vs "cash flow available for debt service", "debt" vs "total debt" -- is
#: resolved at match time by preferring the *longest* matched span, so the more
#: specific label always wins the line.
LABEL_MAP: dict[LineItem, list[_Label]] = {
    LineItem.REVENUE_LTM: _labels(
        exact=[
            "total revenue",
            "revenue from contracts with customers",
            "sales revenue",
            "turnover",
            "total operating revenue",
        ],
        fuzzy=["revenue", "net sales", "total sales"],
    ),
    LineItem.EBITDA_REPORTED: _labels(
        exact=[
            "reported ebitda",
            "statutory ebitda",
            "ebitda",
            "earnings before interest tax depreciation and amortisation",
        ],
        fuzzy=["adjusted ebitda", "underlying ebitda", "normalised ebitda"],
    ),
    LineItem.EBITDA_ADDBACKS: _labels(
        exact=[
            "ebitda add-backs",
            "ebitda addbacks",
            "add-backs",
            "addbacks",
            "add backs",
            "normalisation adjustments",
            "normalization adjustments",
        ],
        fuzzy=["adjustments to ebitda", "one-off items", "non-recurring items"],
    ),
    LineItem.TOTAL_DEBT: _labels(
        exact=[
            "total debt",
            "total borrowings",
            "total interest bearing debt",
            "total interest-bearing debt",
            "total interest bearing liabilities",
            "interest-bearing liabilities",
            "interest bearing liabilities",
            "total loans and borrowings",
        ],
        fuzzy=["borrowings", "loans and borrowings", "debt"],
    ),
    LineItem.CASH: _labels(
        exact=[
            "cash and cash equivalents",
            "cash and bank balances",
            "cash at bank",
        ],
        fuzzy=["cash"],
    ),
    LineItem.INTEREST_EXPENSE: _labels(
        exact=[
            "interest expense",
            "net interest expense",
            "finance costs",
            "net finance costs",
            "finance expense",
            "finance costs - net",
        ],
        fuzzy=["interest paid", "finance charges", "borrowing costs"],
    ),
    LineItem.PRINCIPAL_REPAYMENTS: _labels(
        exact=[
            "repayment of borrowings",
            "repayments of borrowings",
            "repayment of lease liabilities",
            "repayments of lease liabilities",
            "principal repayment",
            "principal repayments",
            "repayment of debt",
            "scheduled principal amortisation",
        ],
        fuzzy=["repayments", "debt amortisation"],
    ),
    LineItem.CFADS: _labels(
        exact=[
            "cash flow available for debt service",
            "cash flows available for debt service",
            "cfads",
        ],
        fuzzy=["cash available for debt service", "net cash available for debt service"],
    ),
}


# --------------------------------------------------------------------------
# Units
#
# Statements are almost never printed in base currency units. A document that
# says "$'000" at the top of the page and "1,250" in the debt row means
# $1,250,000. Missing that is a silent 1000x error that survives every sanity
# check a reader might apply to the ratio, so it is handled explicitly and
# tested.
# --------------------------------------------------------------------------
_UNIT_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"in\s+billions|\$\s*'?\s*bn\b|\bA\$bn\b|\bUS\$bn\b|\$bn\b", re.I), 1e9),
    (
        re.compile(
            r"in\s+millions|\$\s*'?\s*000\s*'?\s*000\b|\bA\$m\b|\bUS\$m\b|\$m\b"
            r"|\$\s*millions?\b|millions\s+of\s+dollars",
            re.I,
        ),
        1e6,
    ),
    (
        re.compile(
            r"in\s+thousands|\$\s*'\s*000\b|\bA\$'000\b|\bUS\$'000\b|'000\b"
            r"|\$\s*thousands?\b|thousands\s+of\s+dollars",
            re.I,
        ),
        1e3,
    ),
]

_CURRENCY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bAUD\b|A\$", re.I), "AUD"),
    (re.compile(r"\bNZD\b|NZ\$", re.I), "NZD"),
    (re.compile(r"\bSGD\b|S\$", re.I), "SGD"),
    (re.compile(r"\bUSD\b|US\$", re.I), "USD"),
    (re.compile(r"\bEUR\b|€"), "EUR"),
    (re.compile(r"\bGBP\b|£"), "GBP"),
]


def detect_unit_scale(page_text: str) -> float:
    """Return the multiplier declared by this page's units statement.

    Whichever declaration appears *earliest* in the page wins, because the
    units note is printed above the table it governs. Returns 1.0 when the page
    declares nothing -- assuming a scale that was never stated would be exactly
    the kind of invented precision this baseline is meant to avoid.
    """
    best_pos: int | None = None
    best_scale = 1.0
    for pattern, scale in _UNIT_PATTERNS:
        match = pattern.search(page_text)
        if match is not None and (best_pos is None or match.start() < best_pos):
            best_pos = match.start()
            best_scale = scale
    return best_scale


def detect_currency(page_text: str) -> str | None:
    """Return an ISO currency code if the page names one unambiguously.

    A bare ``$`` is left as ``None``: it could be AUD, USD or SGD, and this
    extractor has no business guessing which.
    """
    best_pos: int | None = None
    best_code: str | None = None
    for pattern, code in _CURRENCY_PATTERNS:
        match = pattern.search(page_text)
        if match is not None and (best_pos is None or match.start() < best_pos):
            best_pos = match.start()
            best_code = code
    return best_code


# --------------------------------------------------------------------------
# Number parsing
# --------------------------------------------------------------------------
_MAGNITUDES: dict[str, float] = {
    "k": 1e3,
    "thousand": 1e3,
    "thousands": 1e3,
    "m": 1e6,
    "mm": 1e6,
    "million": 1e6,
    "millions": 1e6,
    "bn": 1e9,
    "b": 1e9,
    "bln": 1e9,
    "billion": 1e9,
    "billions": 1e9,
}

_NUMBER_RE = re.compile(
    r"""
    (?P<open>\()?\s*
    (?P<cur>[A-Z]{0,3}\$|[€£])?\s*
    (?P<sign>[-‐-―])?\s*
    (?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)
    (?:\s?(?P<suffix>billions?|millions?|thousands?|bln|bn|mm|tn|k|m|b)\b)?
    \s*(?P<close>\))?
    """,
    re.IGNORECASE | re.VERBOSE,
)


class ParsedNumber(NamedTuple):
    """A number lifted out of a line, plus how it was written."""

    value: float
    #: Multiplier the number carried itself (a "1.2m" suffix). When non-1.0 the
    #: page-level units declaration must be ignored, or the scale is applied
    #: twice.
    inline_scale: float
    start: int
    end: int


def _looks_like_a_year(token: str, has_currency: bool, has_suffix: bool) -> bool:
    """Four bare digits in 1900-2099 are far more often a period heading."""
    if has_currency or has_suffix or "," in token or "." in token:
        return False
    return bool(re.fullmatch(r"(19|20)\d{2}", token))


def parse_number(text: str, start: int = 0) -> ParsedNumber | None:
    """Find the first *plausible* number in ``text`` at or after ``start``.

    Handles thousands separators, currency symbols, unicode minus signs,
    accounting parentheses-as-negative and magnitude suffixes. Rejects
    candidates that are almost certainly not values: bare years, note
    references and percentages. Returns None when nothing plausible is present,
    which is the signal to abstain.
    """
    for match in _NUMBER_RE.finditer(text, start):
        token = match.group("num")
        suffix = (match.group("suffix") or "").lower()
        has_currency = match.group("cur") is not None

        if _looks_like_a_year(token, has_currency, bool(suffix)):
            continue

        # "Note 12" / "note 12(b)" -- a cross-reference, not a value.
        preceding = text[max(0, match.start() - 8) : match.start()].lower()
        if re.search(r"note[s]?\s*$", preceding):
            continue

        # Percentages are ratios, never line-item values.
        trailing = text[match.end() : match.end() + 1]
        if trailing == "%":
            continue

        try:
            value = float(token.replace(",", ""))
        except ValueError:  # pragma: no cover - regex guarantees parseability
            continue

        inline_scale = _MAGNITUDES.get(suffix, 1.0)
        # Parentheses only mean "negative" when they actually close.
        negative = match.group("sign") is not None or (
            match.group("open") is not None and match.group("close") is not None
        )
        if negative:
            value = -value
        return ParsedNumber(value, inline_scale, match.start(), match.end())
    return None


# --------------------------------------------------------------------------
# Line matching
# --------------------------------------------------------------------------
class _LabelHit(NamedTuple):
    item: LineItem
    confidence: float
    end: int
    span: int


def _best_label_hit(line: str) -> _LabelHit | None:
    """Return the best line-item label match on ``line``, if any.

    "Best" means the *longest* matched span. This is what stops CASH stealing a
    "Cash flow available for debt service" row from CFADS, and stops the broad
    "debt" fallback stealing a "Total debt" row from the exact pattern.
    """
    best: _LabelHit | None = None
    for item, labels in LABEL_MAP.items():
        for label in labels:
            match = label.pattern.search(line)
            if match is None:
                continue
            span = match.end() - match.start()
            if best is None or span > best.span:
                best = _LabelHit(item, label.confidence, match.end(), span)
    return best


class HeuristicExtractor:
    """Keyword-anchored regex extractor. Satisfies the `Extractor` protocol.

    Stateless and deterministic: the same document always yields the same
    `Extraction`, which is what makes it usable as a regression fixture for the
    rest of the harness.
    """

    name = "heuristic"

    def extract(self, pdf_path: Path, doc_id: str) -> Extraction:
        """Parse a PDF and extract line items.

        Parsing is the only step that can plausibly blow up (encrypted file,
        truncated stream), so a failure there degrades to a fully abstained
        `Extraction` rather than propagating -- the harness must be able to
        score a document it could not read.
        """
        started = time.perf_counter()
        try:
            doc = Document.from_pdf(Path(pdf_path), doc_id)
        except Exception:
            return self._abstained(doc_id, time.perf_counter() - started)
        return self.extract_document(doc, doc_id)

    def extract_document(self, doc: Document, doc_id: str) -> Extraction:
        """Extract line items from an already-parsed `Document`.

        Split out from `extract` so the whole extractor is testable from plain
        strings -- no PDF fixtures, no PyMuPDF, no I/O.
        """
        started = time.perf_counter()
        try:
            found = self._scan(doc)
        except Exception:
            found = {}

        fields = [
            found.get(item, ExtractedField(name=item, value=None, confidence=0.0))
            for item in LineItem
        ]
        return Extraction(
            doc_id=doc_id,
            fields=fields,
            extractor=self.name,
            model=None,
            latency_s=time.perf_counter() - started,
        )

    # -- internals ---------------------------------------------------------

    def _abstained(self, doc_id: str, latency_s: float) -> Extraction:
        """A full abstention set. Used when the document could not be read."""
        return Extraction(
            doc_id=doc_id,
            fields=[
                ExtractedField(name=item, value=None, confidence=0.0)
                for item in LineItem
            ],
            extractor=self.name,
            model=None,
            latency_s=latency_s,
        )

    def _scan(self, doc: Document) -> dict[LineItem, ExtractedField]:
        """Walk the document in reading order, first hit per line item wins.

        First-hit-wins is a simplification with a real cost -- a summary table
        on page 1 beats the audited statement on page 40 -- but it is
        predictable, which matters more in a baseline than being right on the
        hard cases.
        """
        found: dict[LineItem, ExtractedField] = {}

        for page_no in range(1, doc.n_pages + 1):
            page_text = doc.page_text(page_no)
            if not page_text:
                continue
            unit_scale = detect_unit_scale(page_text)
            currency = detect_currency(page_text)
            lines = page_text.split("\n")

            for idx, line in enumerate(lines):
                hit = _best_label_hit(line)
                if hit is None or hit.item in found:
                    continue

                confidence = hit.confidence
                parsed = parse_number(line, hit.end)
                quote = line

                if parsed is None:
                    # Statement rows wrap: the label sits on one line and its
                    # figures on the next. Only follow the wrap if the next
                    # line is not itself a labelled row, otherwise we would
                    # attach one item's value to another item's label.
                    nxt = lines[idx + 1] if idx + 1 < len(lines) else None
                    if nxt is not None and _best_label_hit(nxt) is None:
                        parsed = parse_number(nxt)
                        if parsed is not None:
                            quote = line + "\n" + nxt
                            confidence = max(0.0, confidence - CONF_WRAP_PENALTY)

                if parsed is None:
                    # Label present, no number near it -> abstain and keep
                    # looking; a later occurrence may carry the figure.
                    continue
                if not quote.strip():  # pragma: no cover - a label implies text
                    continue

                # A number that carries its own magnitude ("A$1.2m") has
                # already absorbed the scale; applying the page declaration on
                # top would multiply it twice.
                scale = parsed.inline_scale if parsed.inline_scale != 1.0 else unit_scale

                found[hit.item] = ExtractedField(
                    name=hit.item,
                    value=parsed.value,
                    unit_scale=scale,
                    currency=currency,
                    evidence=Evidence(page=page_no, quote=quote),
                    confidence=confidence,
                )

            if len(found) == len(LineItem):
                break

        return found
