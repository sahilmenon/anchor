"""Evidence verification -- the module that makes Anchor's claim testable.

An extractor can always produce a plausible-looking number. What it cannot
fake is a *verbatim* quote that (a) really occurs on the page it cited and
(b) really contains the number it reported. This module checks both, and
stamps `ExtractedField.grounded` with the verdict.

Two failure modes matter and they are different:

* **Bad citation** -- the quote is not on the cited page at all. Usually the
  model paraphrased, or guessed a page number.
* **Hallucinated citation** -- the quote *is* on the page, verbatim, but the
  number the model reported appears nowhere in it. This is the dangerous one,
  because every cheap check (page exists, quote matches) passes. It is caught
  here by `value_in_quote`.

Abstention is not a failure. A field with `value is None` gets
`grounded = None`: there is no claim to ground, and scoring treats abstention
as a first-class outcome rather than a miss.
"""

from __future__ import annotations

import re
import unicodedata

from anchor.extractors.base import Document
from anchor.schema import ExtractedField, Extraction

__all__ = [
    "normalise",
    "parse_number",
    "quote_on_page",
    "value_in_quote",
    "verify_field",
    "verify_extraction",
    "grounding_rate",
]


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------

# Characters PDF text layers emit that are semantically identical to an ASCII
# character but break naive `in` tests. Mapped before casefolding so that the
# comparison operates on one canonical alphabet.
_CHAR_MAP = {
    " ": " ",  # NBSP -- extremely common in financial tables
    " ": " ",  # figure space
    " ": " ",  # narrow NBSP
    " ": " ",  # thin space
    " ": " ",
    " ": " ",
    "‐": "-",  # hyphen
    "‑": "-",  # non-breaking hyphen
    "‒": "-",  # figure dash
    "–": "-",  # en dash
    "—": "-",  # em dash
    "―": "-",  # horizontal bar
    "−": "-",  # MINUS SIGN -- fonts use this for negatives
    "‘": "'",
    "’": "'",  # curly apostrophe: "company's" vs "company's"
    "‚": "'",
    "‛": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "′": "'",  # prime
    "″": '"',  # double prime
}

# Invisible characters that carry no meaning but defeat substring matching.
# Soft hyphens in particular are inserted at line-break points by typesetters.
_ZERO_WIDTH = dict.fromkeys(map(ord, "­​‌‍⁠﻿"), None)

_TRANSLATION = {ord(k): v for k, v in _CHAR_MAP.items()}
_TRANSLATION.update(_ZERO_WIDTH)

_WS_RE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Canonicalise text so that two renderings of the same string compare equal.

    Order matters. NFKC first, because it folds ligatures (``ﬁ`` -> ``fi``) and
    compatibility forms that the explicit table cannot enumerate. Then the
    explicit table, because NFKC deliberately leaves dashes and curly quotes
    alone -- and those are exactly what a PDF quote and a model's echo of it
    disagree on. Whitespace is collapsed last so that the substituted spaces
    (from NBSP etc.) get folded in too, and casefold runs at the end so it
    applies to the final, fully-substituted string.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_TRANSLATION)
    text = _WS_RE.sub(" ", text)
    return text.strip().casefold()


def _strip_ws(text: str) -> str:
    """Remove every whitespace character. Used only for the fallback match."""
    return _WS_RE.sub("", text)


# --------------------------------------------------------------------------
# Number parsing
# --------------------------------------------------------------------------

# Magnitude suffixes as written in filings. ``mm`` and ``mn`` both mean
# million; ``m`` alone does too (nobody writes "1.2 metres" in a debt
# schedule). Longest alternatives must come first so the alternation does not
# stop at "m" inside "million".
_SUFFIXES: dict[str, float] = {
    "trillion": 1e12,
    "billion": 1e9,
    "million": 1e6,
    "thousand": 1e3,
    "trn": 1e12,
    "tn": 1e12,
    "bn": 1e9,
    "mm": 1e6,
    "mn": 1e6,
    "k": 1e3,
    "m": 1e6,
    "b": 1e9,
    "t": 1e12,
}

_SUFFIX_ALT = "|".join(sorted(_SUFFIXES, key=len, reverse=True))

# One number as it appears in a financial statement. Every adornment is
# optional except the digits themselves.
#
#   (?<![\w.])  -- do not start mid-token ("note3", "v1.2" are not figures)
#   open/close  -- accounting parentheses, which mean NEGATIVE, not grouping
#   cur         -- $, £, €, optionally prefixed (A$, US$, S$, HK$)
#   trailsign   -- some systems print the minus after the digits ("1,234-")
_NUMBER_PATTERN = rf"""
    (?<![\w.])
    (?P<open>\()?[ ]?
    (?P<cur>(?:a|us|nz|s|hk|c|r)?[ ]?[$£€¥])?[ ]?
    (?P<sign>[-+])?[ ]?
    (?P<int>\d{{1,3}}(?:,\d{{3}})+|\d+)
    (?P<dec>\.\d+)?
    [ ]?(?P<suf>{_SUFFIX_ALT})?(?![a-z0-9])
    [ ]?(?P<close>\))?
    (?P<trailsign>-)?
"""

_NUMBER_RE = re.compile(_NUMBER_PATTERN, re.VERBOSE)
_NUMBER_FULL_RE = re.compile(rf"{_NUMBER_PATTERN}\Z", re.VERBOSE)


def _from_match(m: re.Match[str]) -> float | None:
    """Turn a `_NUMBER_RE` match into a signed float, or None if degenerate."""
    digits = m.group("int").replace(",", "")
    if not digits:
        return None
    try:
        value = float(digits + (m.group("dec") or ""))
    except ValueError:  # pragma: no cover -- regex guarantees the shape
        return None

    suffix = m.group("suf")
    if suffix:
        value *= _SUFFIXES[suffix]

    # Accounting parentheses are the dominant negative notation in filings, and
    # they only count when balanced -- "(1,234" is a broken span, not a figure.
    negative = bool(m.group("open") and m.group("close"))
    if m.group("sign") == "-" or m.group("trailsign"):
        negative = True
    return -value if negative else value


def parse_number(s: str) -> float | None:
    """Parse one financial-statement number, or return None if it is not one.

    Handles thousands separators, currency symbols, decimals, leading or
    trailing minus, magnitude suffixes, and parenthesised negatives. This is a
    whole-string parse: trailing prose ("1,234 (unaudited)") yields None,
    because a caller asking to parse a *number* should not silently get the
    first number out of a sentence. Use `value_in_quote` for scanning prose.
    """
    if not s:
        return None
    text = unicodedata.normalize("NFKC", s).translate(_TRANSLATION)
    text = _WS_RE.sub(" ", text).strip().casefold()
    if not text:
        return None
    m = _NUMBER_FULL_RE.match(text)
    if m is None or m.start() != 0:
        return None
    return _from_match(m)


def _iter_numbers(text: str) -> list[float]:
    """Every number occurring in already-normalised text, in order."""
    out: list[float] = []
    for m in _NUMBER_RE.finditer(text):
        v = _from_match(m)
        if v is not None:
            out.append(v)
    return out


# --------------------------------------------------------------------------
# Quote and value checks
# --------------------------------------------------------------------------


def quote_on_page(quote: str, page_text: str) -> bool:
    """Is `quote` present in `page_text` after normalisation?

    Two passes. The normalised substring test is the honest one. The
    whitespace-stripped fallback exists because PDF text extraction routinely
    splits a word across spans ("EBIT DA", "1, 234") or drops the space
    between a label and its figure -- differences that are artefacts of the
    text layer, not of the quote. Stripping all whitespace from both sides
    loses word boundaries, so it is only ever a fallback, never the first test.
    """
    if not quote:
        return False
    n_quote = normalise(quote)
    if not n_quote:
        return False
    n_page = normalise(page_text)
    if not n_page:
        return False
    if n_quote in n_page:
        return True
    return _strip_ws(n_quote) in _strip_ws(n_page)


def _close(a: float, b: float, rel_tol: float) -> bool:
    """Relative comparison with a sane zero case."""
    if a == b:
        return True
    if b == 0.0:
        return abs(a) <= rel_tol
    return abs(a - b) <= rel_tol * abs(b)


def value_in_quote(
    value: float,
    quote: str,
    unit_scale: float = 1.0,
    rel_tol: float = 0.01,
) -> bool:
    """Does a number matching `value` actually appear inside `quote`?

    This is the hallucinated-citation check. A quote can be perfectly real and
    still not support the figure attributed to it.

    Two targets are accepted. `value` itself, because the extractor normally
    reports the number as printed in a table whose header says "$ in
    thousands" and carries the 1000 in `unit_scale`. And `value * unit_scale`,
    because a quote sometimes spells the figure out in full units instead.
    Either reading grounds the field.

    Tolerance is relative, not exact. Statements round, and extractors
    re-derive: a quote printing "1,234.6" should still ground a reported
    1,235. Exact equality would reject correct extractions as hallucinations,
    which is the more expensive error for an auditability harness to make.
    """
    numbers = _iter_numbers(normalise(quote))
    if not numbers:
        return False

    targets = [value]
    if unit_scale not in (1.0, 0.0):
        targets.append(value * unit_scale)

    return any(_close(n, t, rel_tol) for n in numbers for t in targets)


# --------------------------------------------------------------------------
# Field / extraction verification
# --------------------------------------------------------------------------


def verify_field(field: ExtractedField, doc: Document) -> ExtractedField:
    """Return a copy of `field` with `grounded` decided against `doc`.

    Never mutates the input: the harness compares pre- and post-verification
    extractions, and an in-place stamp would destroy that record.

    Verdicts:
      * abstained (value is None) -> None   (no claim, so nothing to ground)
      * no evidence               -> False
      * page out of range         -> False
      * quote not on cited page   -> False
      * quote present, value absent -> False  (hallucinated citation)
      * otherwise                 -> True
    """
    out = field.model_copy(deep=True)

    if out.value is None:
        out.grounded = None
        return out

    ev = out.evidence
    if ev is None:
        out.grounded = False
        return out

    if ev.page < 1 or ev.page > doc.n_pages:
        out.grounded = False
        return out

    page_text = doc.page_text(ev.page)
    if not quote_on_page(ev.quote, page_text):
        out.grounded = False
        return out

    out.grounded = value_in_quote(out.value, ev.quote, out.unit_scale)
    return out


def verify_extraction(extraction: Extraction, doc: Document) -> Extraction:
    """Return a copy of `extraction` with every field verified."""
    out = extraction.model_copy(deep=True)
    out.fields = [verify_field(f, doc) for f in extraction.fields]
    return out


def grounding_rate(extraction: Extraction) -> float:
    """Fraction of non-abstained fields whose `grounded` is exactly True.

    Abstained fields are excluded from both numerator and denominator: an
    extractor that declines to answer should be neither rewarded nor punished
    by this metric (abstention quality is scored separately).

    Returns 0.0 when there are no non-abstained fields. That is a deliberate
    choice over 1.0 or NaN -- an extraction that claims nothing has grounded
    nothing, and a metric that reads as perfect for a blank answer would make
    the headline number gameable. Callers that need to distinguish "nothing
    claimed" from "everything wrong" should check the field list themselves.
    """
    claimed = [f for f in extraction.fields if f.value is not None]
    if not claimed:
        return 0.0
    return sum(1 for f in claimed if f.grounded is True) / len(claimed)
