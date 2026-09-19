"""An LLM extractor: Claude, constrained to the evidence schema.

What this is for
----------------
The heuristic extractor establishes a floor. This one is the thing the harness
was built to measure: a model that reads a document the way an analyst does,
and that can therefore fail in ways a regex cannot — inventing a figure,
citing a page that does not support it, or confidently answering a question the
document never addresses.

Four decisions shape the implementation.

**The model is asked for line items and evidence, never a ratio.** Same
contract as every other extractor (see `anchor.schema`). Leverage and DSCR are
derived in plain Python, so a wrong multiple is always traceable to a wrong
input rather than to arithmetic.

**It is shown page text, not the PDF.** The API accepts PDFs natively and that
would likely read scanned documents better. It would also break the thing that
makes this harness worth anything: `anchor.verify` checks each quote against
the page text the extractor saw, and if the model quotes from Anthropic's own
rendering of the PDF while the verifier checks PyMuPDF's, honest quotes fail
verification and the grounding rate measures a text-layer disagreement instead
of the model. Feeding both sides the same text keeps the grounding number
meaning what it claims to mean.

**Output is constrained by a Pydantic schema**, not requested in prose. The
schema goes to the API as a JSON schema (`output_config.format`), so a
structurally invalid response is a rejected generation rather than a parsing
problem downstream. Validation still runs locally, because a schema constrains
shape and not sense: a `page` of 900 in a 12-page document satisfies the schema.

**Failure is retried once with the failure fed back, then abstained.** See
`_attempt` and the module's retry section below.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from anchor.extractors.base import Document
from anchor.schema import (
    MAGNITUDE_ITEMS,
    Evidence,
    ExtractedField,
    Extraction,
    LineItem,
)
from anchor.verify import quote_on_page, value_in_quote

__all__ = [
    "DEFAULT_MODEL",
    "PRICING",
    "ClaudeExtractor",
    "estimate_cost",
]

#: Default model. Overridden per run by `anchor run --model`, and swept by
#: `anchor sweep`, which is where the cost/accuracy question actually gets
#: answered rather than assumed.
DEFAULT_MODEL = "claude-opus-5"


@dataclass(frozen=True)
class Price:
    """USD per million tokens, by token class.

    Cache reads bill at roughly a tenth of the input rate and cache writes at
    1.25x, so a run that caches its document prefix across a retry is priced
    quite differently from one that does not. Folding those in here keeps the
    reported ``cost_usd`` honest instead of counting every input token at the
    full rate.
    """

    input: float
    output: float

    @property
    def cache_read(self) -> float:
        return self.input * 0.1

    @property
    def cache_write(self) -> float:
        return self.input * 1.25


#: Published list prices. A model missing from this table still runs; its cost
#: is reported as 0.0 and `ClaudeExtractor.priced` is False, so a sweep can say
#: "unpriced" rather than quietly charging nothing.
PRICING: dict[str, Price] = {
    "claude-opus-5": Price(5.00, 25.00),
    "claude-opus-4-8": Price(5.00, 25.00),
    "claude-sonnet-5": Price(3.00, 15.00),
    "claude-haiku-4-5": Price(1.00, 5.00),
    "claude-fable-5": Price(10.00, 50.00),
}


def estimate_cost(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Price one call. Returns 0.0 for a model with no published rate."""
    price = PRICING.get(model)
    if price is None:
        return 0.0
    return (
        input_tokens * price.input
        + output_tokens * price.output
        + cache_read_tokens * price.cache_read
        + cache_write_tokens * price.cache_write
    ) / 1_000_000


# ---------------------------------------------------------------------------
# The constrained output schema
# ---------------------------------------------------------------------------


class ClaimedField(BaseModel):
    """One line item as the model reports it.

    ``extra="forbid"`` emits ``additionalProperties: false``, which structured
    outputs require on every object. Bounds like "confidence is in [0, 1]" are
    deliberately absent: the structured-output schema dialect does not support
    numeric constraints, so stating them here would be a promise enforced
    nowhere. `_coerce` enforces them locally instead.
    """

    model_config = ConfigDict(extra="forbid")

    name: LineItem
    value: float | None
    unit_scale: float
    currency: str | None
    page: int | None
    quote: str | None
    confidence: float


class ClaimedExtraction(BaseModel):
    """The model's full answer for one document."""

    model_config = ConfigDict(extra="forbid")

    fields: list[ClaimedField]


SYSTEM_PROMPT = """\
You read financial statements and credit documents and extract specific line \
items, with the evidence for each.

Return one entry for every line item in the schema, in the order given.

For each line item:

- `value`: the figure as printed in the document, before any unit scaling. Use \
null when the document does not state the item.
- `unit_scale`: the multiplier that turns `value` into base currency units, \
taken from the document's own units declaration. A page headed "$'000" gives \
1000; a page in millions gives 1000000; a figure printed in full units gives 1.
- `currency`: the ISO code if the document names one, else null. A bare "$" is \
not enough to name one.
- `page`: the 1-indexed page the figure appears on, or null when you abstain.
- `quote`: text copied character for character from that page, containing the \
figure. Not a paraphrase, not reformatted, not stitched together from \
different rows. Null when you abstain.
- `confidence`: 0 to 1, how likely your value is the figure a credit analyst \
would record.

Abstaining is a correct answer, not a failure. A set of statutory accounts \
often does not disclose scheduled principal repayments, and reporting null \
for one is right. Reporting a plausible number for a figure the document does \
not state is the worst outcome available to you: it looks like an answer and \
nothing downstream catches it.

Two conventions:

- `interest_expense` and `principal_repayments` are magnitudes. A statement \
prints them parenthesised because the cash went out, not because the quantity \
is negative. Report 842, not -842, for a row reading "(842)".
- Report the consolidated figure for the current period. Comparative columns \
and segment notes are not the answer, even when they appear first on the page.

Where a figure is split across rows the document does state — current and \
non-current borrowings, or principal split between a loan repayment and a \
lease repayment — report the total and quote the rows it came from.\
"""


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


@dataclass
class _Attempt:
    """One API round trip and what it produced."""

    fields: dict[LineItem, ExtractedField]
    problems: list[str]
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    error: str = ""


class ClaudeExtractor:
    """Claude behind the `Extractor` protocol.

    Retry and abstain
    -----------------
    Two things can go wrong that a retry can actually fix, and both are checked
    locally before the answer is accepted:

    * the response does not validate against the schema, or the API call fails;
    * a field's quote is not on the page it cites, or does not contain the
      figure it reports.

    The second is the interesting one. It is exactly what `anchor.verify`
    would reject, so catching it here and handing the specific failure back
    ("your quote for total_debt is not on page 4") gives the model the one
    piece of information it needs to correct itself. Anything still failing
    after the retry is **abstained rather than reported** -- an ungrounded
    figure that survives a correction round is not an answer, and a harness
    that passes it through is measuring the wrong thing.

    Retries are counted and surfaced on the `Extraction`, so a model that only
    reaches its score on the second attempt cannot look identical to one that
    gets it right first time.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        effort: str = "high",
        max_tokens: int = 8000,
        max_retries: int = 1,
        client: Any | None = None,
        max_pages: int = 60,
    ) -> None:
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.max_pages = max_pages
        self._client = client

    @property
    def name(self) -> str:
        """Scored under the model name, so a sweep's rows are distinguishable."""
        return f"claude:{self.model}"

    @property
    def priced(self) -> bool:
        return self.model in PRICING

    # -- client ------------------------------------------------------------

    def client(self) -> Any:
        """The Anthropic client, constructed on first use.

        Deferred so that importing this module, listing extractors, or running
        `anchor corpus` costs nothing and needs no credentials. The optional
        dependency and the API key are only required at the moment a document
        is actually sent.
        """
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on install
                raise RuntimeError(
                    "the claude extractor needs the anthropic SDK: "
                    'pip install -e ".[llm]"'
                ) from exc
            self._client = anthropic.Anthropic()
        return self._client

    # -- protocol ----------------------------------------------------------

    def extract(self, pdf_path: Path, doc_id: str) -> Extraction:
        started = time.perf_counter()
        try:
            doc = Document.from_pdf(Path(pdf_path), doc_id)
        except Exception as exc:
            return self._abstained(doc_id, time.perf_counter() - started, str(exc))
        return self.extract_document(doc, doc_id)

    def extract_document(self, doc: Document, doc_id: str) -> Extraction:
        """Extract one document, retrying once on a correctable failure."""
        started = time.perf_counter()
        if doc.n_pages == 0:
            return self._abstained(doc_id, time.perf_counter() - started, "no pages")

        accepted: dict[LineItem, ExtractedField] = {}
        totals = _Attempt(fields={}, problems=[])
        feedback: list[str] = []
        retries = 0

        for attempt in range(self.max_retries + 1):
            result = self._attempt(doc, feedback)
            _accumulate(totals, result)
            if result.error:
                # An API failure or a refusal ends the attempt loop. Whatever
                # verified on an earlier attempt is kept; the rest abstains.
                break

            # Fields that verified are kept even if others failed; a retry
            # should not put a good answer at risk.
            for name, field in result.fields.items():
                accepted.setdefault(name, field)

            if not result.problems:
                break
            if attempt == self.max_retries:
                # Out of retries. Everything still unverified is abstained,
                # which is what `accepted` not containing it already means.
                break
            retries += 1
            feedback = result.problems

        fields = [
            accepted.get(item, ExtractedField(name=item, value=None, confidence=0.0))
            for item in LineItem
        ]
        return Extraction(
            doc_id=doc_id,
            fields=fields,
            extractor=self.name,
            model=self.model,
            latency_s=time.perf_counter() - started,
            input_tokens=totals.input_tokens + totals.cache_read_tokens
            + totals.cache_write_tokens,
            output_tokens=totals.output_tokens,
            cost_usd=totals.cost_usd,
            retries=retries,
        )

    # -- internals ---------------------------------------------------------

    def _attempt(self, doc: Document, feedback: list[str]) -> _Attempt:
        """One API call, validated and quote-checked locally."""
        try:
            response = self.client().messages.create(**self._request(doc, feedback))
        except Exception as exc:
            return _Attempt(fields={}, problems=[], error=f"{type(exc).__name__}: {exc}")

        usage = getattr(response, "usage", None)
        result = _Attempt(fields={}, problems=[])
        result.input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        result.output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        result.cache_read_tokens = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        result.cache_write_tokens = int(
            getattr(usage, "cache_creation_input_tokens", 0) or 0
        )
        result.cost_usd = estimate_cost(
            self.model,
            result.input_tokens,
            result.output_tokens,
            result.cache_read_tokens,
            result.cache_write_tokens,
        )

        # A safety decline is a real outcome, not an exception. Reading
        # `content[0]` without checking would raise on an empty content list.
        if getattr(response, "stop_reason", None) == "refusal":
            result.error = "the model declined the request (stop_reason=refusal)"
            return result

        text = _first_text(response)
        if text is None:
            result.error = "no text block in the response"
            return result

        try:
            claimed = ClaimedExtraction.model_validate_json(text)
        except ValidationError as exc:
            # Recoverable: hand the validation error back and let it try again.
            result.problems = [f"your previous reply did not match the schema: {exc}"]
            return result

        for claim in claimed.fields:
            field, problem = self._coerce(claim, doc)
            if problem:
                result.problems.append(problem)
            if field is not None:
                result.fields[claim.name] = field
        return result

    def _coerce(
        self, claim: ClaimedField, doc: Document
    ) -> tuple[ExtractedField | None, str]:
        """Turn one claim into a field, or say what is wrong with it.

        Abstentions pass straight through: there is no quote to check and no
        figure to ground, and a model that declines is giving the answer the
        corpus often wants.
        """
        label = claim.name.value

        if claim.value is None:
            return ExtractedField(
                name=claim.name,
                value=None,
                confidence=_clamp(claim.confidence),
            ), ""

        if claim.page is None or not claim.quote:
            return None, (
                f"{label}: you reported {claim.value} with no page or quote. "
                "Either cite the page and quote it verbatim, or set value to null."
            )
        if claim.page < 1 or claim.page > doc.n_pages:
            return None, (
                f"{label}: you cited page {claim.page}, but the document has "
                f"{doc.n_pages} page(s)."
            )

        page_text = doc.page_text(claim.page)
        if not quote_on_page(claim.quote, page_text):
            return None, (
                f"{label}: your quote is not on page {claim.page}. Copy the row "
                "exactly as it appears, or set value to null if it is not there."
            )

        scale = claim.unit_scale if claim.unit_scale and claim.unit_scale > 0 else 1.0
        value = abs(claim.value) if claim.name in MAGNITUDE_ITEMS else claim.value

        if not value_in_quote(
            value, claim.quote, scale, match_magnitude=claim.name in MAGNITUDE_ITEMS
        ):
            return None, (
                f"{label}: you reported {claim.value} but that figure does not "
                f"appear in the quote you gave. Quote the row containing it, or "
                "set value to null."
            )

        return ExtractedField(
            name=claim.name,
            value=value,
            unit_scale=scale,
            currency=claim.currency,
            evidence=Evidence(page=claim.page, quote=claim.quote),
            confidence=_clamp(claim.confidence),
        ), ""

    def _request(self, doc: Document, feedback: list[str]) -> dict[str, Any]:
        """Build the API request.

        The document sits in its own content block with a cache breakpoint on
        it. Within one document the prefix is byte-identical across the retry,
        so the correction round reads the pages from cache instead of paying
        for them twice -- which is most of what a retry would otherwise cost.
        """
        pages = "\n\n".join(
            f"<page number=\"{i}\">\n{doc.page_text(i)}\n</page>"
            for i in range(1, min(doc.n_pages, self.max_pages) + 1)
        )

        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": f"<document id=\"{doc.doc_id}\">\n{pages}\n</document>",
                "cache_control": {"type": "ephemeral"},
            }
        ]
        if feedback:
            problems = "\n".join(f"- {p}" for p in feedback)
            content.append(
                {
                    "type": "text",
                    "text": (
                        "Your previous answer had problems with these fields:\n"
                        f"{problems}\n\n"
                        "Answer again for every line item. Keep the fields that "
                        "were fine. For each problem above, either cite evidence "
                        "that holds up or set that field's value to null."
                    ),
                }
            )
        else:
            content.append(
                {"type": "text", "text": "Extract the line items from this document."}
            )

        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": [{"type": "text", "text": SYSTEM_PROMPT}],
            "messages": [{"role": "user", "content": content}],
            "output_config": {
                "effort": self.effort,
                "format": {
                    "type": "json_schema",
                    "schema": ClaimedExtraction.model_json_schema(),
                },
            },
        }

    def _abstained(self, doc_id: str, latency_s: float, error: str = "") -> Extraction:
        return Extraction(
            doc_id=doc_id,
            fields=[
                ExtractedField(name=item, value=None, confidence=0.0)
                for item in LineItem
            ],
            extractor=self.name,
            model=self.model,
            latency_s=latency_s,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(value: float) -> float:
    """Confidence into [0, 1]. The schema cannot express the bound."""
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _first_text(response: Any) -> str | None:
    """The first text block's content, or None.

    A response carries thinking blocks as well as text on a thinking-enabled
    model, so indexing `content[0]` is wrong; the text block is not reliably
    first.
    """
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) == "text":
            return getattr(block, "text", None)
    return None


def _accumulate(totals: _Attempt, attempt: _Attempt) -> None:
    totals.input_tokens += attempt.input_tokens
    totals.output_tokens += attempt.output_tokens
    totals.cache_read_tokens += attempt.cache_read_tokens
    totals.cache_write_tokens += attempt.cache_write_tokens
    totals.cost_usd += attempt.cost_usd


def api_key_present() -> bool:
    """Is there a credential for the CLI to find?

    Used to fail `anchor run --extractor claude` with a sentence rather than a
    traceback from inside the SDK.
    """
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def schema_json() -> str:
    """The constrained output schema, for documentation and debugging."""
    return json.dumps(ClaimedExtraction.model_json_schema(), indent=2)
