"""The extractor interface.

This is the point of the project. Anchor scores *any* extractor that
satisfies this protocol -- the two implementations shipped here exist to
prove the interface works and to give the LLM a baseline worth beating.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from anchor.schema import Extraction


@runtime_checkable
class Extractor(Protocol):
    """Anything that turns a document into an `Extraction`.

    Implementations must not raise on a malformed document. Return an
    `Extraction` with abstained fields instead -- the harness scores
    abstention as a first-class outcome, not as a failure.
    """

    name: str

    def extract(self, pdf_path: Path, doc_id: str) -> Extraction:
        """Extract line items from one document."""
        ...


class Document:
    """Page text for one source document.

    Extractors receive this rather than a raw path so that parsing happens
    once and the verifier can check quotes against exactly the text the
    extractor saw.
    """

    def __init__(self, doc_id: str, pages: list[str]) -> None:
        self.doc_id = doc_id
        self.pages = pages

    @property
    def n_pages(self) -> int:
        return len(self.pages)

    def page_text(self, page: int) -> str:
        """1-indexed page text. Returns '' for out-of-range pages."""
        if page < 1 or page > len(self.pages):
            return ""
        return self.pages[page - 1]

    @classmethod
    def from_pdf(cls, pdf_path: Path, doc_id: str | None = None) -> Document:
        """Parse a PDF into page text using PyMuPDF."""
        import pymupdf

        doc = pymupdf.open(pdf_path)
        pages = [p.get_text() for p in doc]
        doc.close()
        return cls(doc_id or Path(pdf_path).stem, pages)

    @classmethod
    def from_pages(cls, doc_id: str, pages: list[str]) -> Document:
        """Build directly from text. Used by tests and fixtures."""
        return cls(doc_id, pages)
