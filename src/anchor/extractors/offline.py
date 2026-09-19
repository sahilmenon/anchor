"""An extractor that reads claims from disk instead of producing them.

Why this exists
---------------
`Extractor` is a protocol, and the point of the project is that Anchor scores
anything satisfying it. Some of the things worth scoring do not run in a loop:
a person reading documents, a model reached through a chat window, a
one-off batch from a tool that has no Python bindings, or an API run captured
months ago and replayed after a scoring change.

This extractor takes the claims such a process produced and puts them through
the identical pipeline: the same coercion, the same magnitude convention, the
same quote verification, the same scoring. A figure that cannot be found on the
page it cites is rejected here exactly as it would be from a live model.

What it is not
--------------
**It is not a model benchmark, and a run of it must not be reported as one.**
The file it reads says nothing about what produced the claims, how long they
took, what they cost, or whether the producer had seen the answers. A row in a
cost/accuracy table that came from here would be a number with no provenance,
which is the thing this repository exists to argue against.

`cost_usd` is therefore 0.0 and `latency_s` is whatever the file records, and
`Extraction.model` carries the label the file gives itself rather than a model
id. `anchor sweep` refuses to include it for the same reason.

The honest use is a labelled sidebar: "here is what a careful reader found on
these documents, scored by the same rules", which answers whether a corpus is
extractable at all. That is worth knowing, and it is not a leaderboard entry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from anchor.corpus import CorpusError
from anchor.extractors.base import Document
from anchor.extractors.claude import ClaimedExtraction, ClaimedField
from anchor.schema import MAGNITUDE_ITEMS, Evidence, ExtractedField, Extraction, LineItem
from anchor.verify import quote_on_page, value_in_quote

__all__ = ["OfflineExtractor", "write_claims"]


@dataclass
class OfflineExtractor:
    """Replays claims recorded in ``<root>/<doc_id>.json``.

    The file uses the same shape a live model is constrained to produce, so a
    transcript captured from one source can be scored beside another without a
    second format to keep in step.

    A document with no file abstains on everything. That is deliberate: a
    partial transcript should score as "did not answer" for what it omits,
    never as an error that stops the run.
    """

    root: Path
    label: str = "offline"
    verify_quotes: bool = True
    """Reject a claim whose quote is not on the cited page, or does not contain
    the figure. On by default. Turning it off is a diagnostic, and the run
    records that it happened."""

    @property
    def name(self) -> str:
        return f"offline:{self.label}"

    def extract(self, pdf_path: Path, doc_id: str) -> Extraction:  # pragma: no cover
        return self.extract_document(Document.from_pdf(Path(pdf_path), doc_id), doc_id)

    def extract_document(self, doc: Document, doc_id: str) -> Extraction:
        path = Path(self.root) / f"{doc_id}.json"
        if not path.is_file():
            return self._abstained(doc_id)

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or "fields" not in raw:
                raise CorpusError(f"{path}: expected an object with a 'fields' key")
            # Validate the claims alone. The envelope carries transcript
            # metadata such as latency that the claim schema forbids, and the
            # schema forbidding it is correct: a live model must not invent
            # envelope keys.
            claimed = ClaimedExtraction.model_validate({"fields": raw["fields"]})
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise CorpusError(f"{path}: not a valid claim file ({exc})") from exc

        found: dict[LineItem, ExtractedField] = {}
        for claim in claimed.fields:
            field = self._coerce(claim, doc)
            if field is not None:
                found[claim.name] = field

        return Extraction(
            doc_id=doc_id,
            fields=[
                found.get(item, ExtractedField(name=item, value=None, confidence=0.0))
                for item in LineItem
            ],
            extractor=self.name,
            model=self.label,
            latency_s=float(raw.get("latency_s", 0.0)),
        )

    def _coerce(self, claim: ClaimedField, doc: Document) -> ExtractedField | None:
        """Apply the same rules a live extractor's claims go through."""
        if claim.value is None:
            return ExtractedField(name=claim.name, value=None, confidence=claim.confidence)

        scale = claim.unit_scale if claim.unit_scale and claim.unit_scale > 0 else 1.0
        value = abs(claim.value) if claim.name in MAGNITUDE_ITEMS else claim.value

        if self.verify_quotes:
            if claim.page is None or not claim.quote:
                return None
            if claim.page < 1 or claim.page > doc.n_pages:
                return None
            if not quote_on_page(claim.quote, doc.page_text(claim.page)):
                return None
            if not value_in_quote(
                value, claim.quote, scale, match_magnitude=claim.name in MAGNITUDE_ITEMS
            ):
                return None

        evidence = (
            Evidence(page=claim.page, quote=claim.quote)
            if claim.page and claim.quote
            else None
        )
        return ExtractedField(
            name=claim.name,
            value=value,
            unit_scale=scale,
            currency=claim.currency,
            evidence=evidence,
            confidence=min(1.0, max(0.0, claim.confidence)),
        )

    def _abstained(self, doc_id: str) -> Extraction:
        return Extraction(
            doc_id=doc_id,
            fields=[
                ExtractedField(name=item, value=None, confidence=0.0) for item in LineItem
            ],
            extractor=self.name,
            model=self.label,
        )


def write_claims(
    root: Path | str,
    doc_id: str,
    fields: list[dict[str, Any]],
    latency_s: float = 0.0,
) -> Path:
    """Record one document's claims in the format `OfflineExtractor` reads.

    Validated on the way out, so a malformed transcript fails where it is
    written rather than halfway through a run.
    """
    payload = {"fields": fields}
    ClaimedExtraction.model_validate(payload)
    payload["latency_s"] = latency_s

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{doc_id}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
