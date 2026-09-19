"""Loading a labelled corpus off disk.

A corpus is a directory laid out like this::

    <root>/golden/<doc_id>.json     one GoldenRecord per document  (committed)
    <root>/pdfs/<doc_id>.pdf        the source document            (git-ignored)
    <root>/text/<doc_id>.json       extracted page text            (optional)
    <root>/thresholds.json          the regression gate floors     (committed)

The golden set defines the corpus. A document with no label is not part of the
corpus, because there is nothing to score it against; a label whose document
cannot be found is part of the corpus and *unresolvable*, which is a different
thing and is reported as such rather than skipped silently. A run that quietly
drops half its documents and reports the accuracy of the remainder is the
single easiest way to publish a flattering number by accident.

Which half of a corpus is committed, and why
--------------------------------------------
``golden/`` is committed and ``pdfs/`` is git-ignored, and that split is the
point rather than an oversight. Labels are the work: they are small, they are
the thing two people need to agree on, and sharing them through git is how a
disagreement becomes a reviewable diff. The documents behind them are borrower
financials and third-party filings, which this repository has no business
redistributing. Anything written here by `save_pdf` therefore stays on the
machine that uploaded it, while `save_golden` writes something you are expected
to commit.

Why two source forms
--------------------
``pdfs/`` is the real thing and wins whenever it is present. ``text/`` holds
page text as a JSON array of strings, which exists for two cases: the synthetic
fixture corpus, which has no PDFs at all and must run in CI; and any document
whose PDF cannot be redistributed but whose text layer can be checked in. The
verifier compares quotes against exactly the page text the extractor saw, so
either form is verifiable -- what would not be verifiable is a corpus with no
document behind the label.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from anchor.extractors.base import Document
from anchor.schema import GoldenRecord

__all__ = [
    "DEFAULT_CORPUS",
    "MAX_PDF_BYTES",
    "PDF_MAGIC",
    "Corpus",
    "CorpusError",
    "CorpusItem",
    "load_corpus",
    "load_golden",
    "load_text_document",
    "safe_doc_id",
    "save_golden",
    "save_pdf",
    "unlabelled_sources",
]

#: Leading bytes of a PDF. Checked on upload so a mislabelled file fails at the
#: door rather than three layers down inside PyMuPDF.
PDF_MAGIC = b"%PDF-"

#: Upload ceiling. A scanned 200-page information memorandum runs to a few tens
#: of megabytes; anything past this is a mistake, and refusing it early beats
#: discovering it after the write.
MAX_PDF_BYTES = 64 * 1024 * 1024

#: Where `anchor` looks when no --corpus is given.
DEFAULT_CORPUS = Path("corpus")


class CorpusError(Exception):
    """Raised when a corpus directory cannot be read as one."""


@dataclass(frozen=True)
class CorpusItem:
    """One labelled document, and the file the document itself lives in."""

    golden: GoldenRecord
    source: Path | None
    kind: str
    """``"pdf"``, ``"text"``, or ``"missing"``."""

    @property
    def doc_id(self) -> str:
        return self.golden.doc_id

    @property
    def resolvable(self) -> bool:
        """True when there is a document to run an extractor over."""
        return self.kind != "missing"

    def document(self) -> Document:
        """Parse this item's source into page text.

        Raises `CorpusError` for an unresolvable item rather than returning an
        empty document: an extractor handed zero pages abstains on everything
        and scores as a wall of omissions, which reads like an extractor
        failure when it is really a missing file.
        """
        if self.source is None:
            raise CorpusError(
                f"{self.doc_id}: no source document. Expected "
                f"pdfs/{self.doc_id}.pdf or text/{self.doc_id}.json"
            )
        if self.kind == "pdf":
            return Document.from_pdf(self.source, self.doc_id)
        return load_text_document(self.source)


@dataclass(frozen=True)
class Corpus:
    """A labelled corpus: its root directory and every item in it."""

    root: Path
    items: list[CorpusItem]

    def __len__(self) -> int:
        return len(self.items)

    @property
    def resolvable(self) -> list[CorpusItem]:
        return [i for i in self.items if i.resolvable]

    @property
    def missing(self) -> list[CorpusItem]:
        return [i for i in self.items if not i.resolvable]

    @property
    def thresholds_path(self) -> Path:
        """Conventional location of this corpus's gate floors."""
        return self.root / "thresholds.json"

    def get(self, doc_id: str) -> CorpusItem | None:
        for item in self.items:
            if item.doc_id == doc_id:
                return item
        return None


def load_text_document(path: Path) -> Document:
    """Read a ``text/<doc_id>.json`` page-text file.

    Format: ``{"doc_id": str, "pages": [str, ...]}``. ``doc_id`` is optional and
    falls back to the filename stem, so a file can be renamed without editing
    it.
    """
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusError(f"{path}: not readable as page text ({exc})") from exc

    if not isinstance(raw, dict) or "pages" not in raw:
        raise CorpusError(f"{path}: expected an object with a 'pages' key")
    pages = raw["pages"]
    if not isinstance(pages, list) or not all(isinstance(p, str) for p in pages):
        raise CorpusError(f"{path}: 'pages' must be a list of strings")

    return Document.from_pages(str(raw.get("doc_id") or path.stem), pages)


def _resolve_source(root: Path, doc_id: str) -> tuple[Path | None, str]:
    """Find the document behind a label. PDF wins over committed text."""
    pdf = root / "pdfs" / f"{doc_id}.pdf"
    if pdf.is_file():
        return pdf, "pdf"
    text = root / "text" / f"{doc_id}.json"
    if text.is_file():
        return text, "text"
    return None, "missing"


# ---------------------------------------------------------------------------
# Writing: uploads and labels
# ---------------------------------------------------------------------------

_UNSAFE = re.compile(r"[^a-z0-9._-]+")
_RESERVED = frozenset({"", ".", ".."})


def safe_doc_id(filename: str) -> str:
    """Turn an uploaded filename into a corpus ``doc_id``.

    A ``doc_id`` becomes a path segment under ``golden/`` and ``pdfs/``, so it
    has to be built rather than trusted. ``Path(...).name`` strips any directory
    the client sent -- including the ``..\\`` and ``/`` forms Windows and POSIX
    disagree about -- and the remaining text is reduced to lowercase ASCII,
    digits, dot, dash and underscore.

    Leading dots go too. A name like ``.gitignore`` is a legal filename and a
    terrible doc_id, and on a corpus directory it would be a file the owner
    never sees in a listing.

    Accented characters are folded to their base letter rather than dropped, so
    a French or Spanish borrower name survives as something its owner can still
    recognise in a directory listing. Refusing those outright would rule out a
    good share of real filings over a naming rule.
    """
    stem = Path(Path(filename).name).stem
    folded = (
        unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode("ascii")
    )
    slug = _UNSAFE.sub("-", folded.lower()).strip("-.")
    if slug in _RESERVED or not slug:
        raise CorpusError(
            f"{filename!r}: cannot be turned into a document id. Give the file a "
            "name containing letters or digits."
        )
    return slug[:100]


def _atomic_write(path: Path, data: bytes) -> None:
    """Write via a temporary file in the same directory, then rename.

    A half-written label is worse than no label: it fails to parse, and
    `load_corpus` then refuses the whole corpus rather than the one file. The
    temporary file shares a directory with the target so the rename stays on one
    filesystem and stays atomic.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def save_pdf(root: Path | str, doc_id: str, data: bytes) -> Path:
    """Store an uploaded PDF as this corpus's source for ``doc_id``.

    Validated before it is written, not after: an empty body, an oversized one,
    or a file that is not a PDF at all are all caught here. PyMuPDF would reject
    most of them too, but it would do so later, from inside an extractor, where
    the failure reads as a bad extractor rather than a bad upload.
    """
    if not data:
        raise CorpusError("empty upload")
    if len(data) > MAX_PDF_BYTES:
        raise CorpusError(
            f"upload is {len(data) / 1e6:.1f} MB, over the "
            f"{MAX_PDF_BYTES / 1e6:.0f} MB limit"
        )
    if not data.startswith(PDF_MAGIC):
        raise CorpusError("not a PDF: the file does not begin with %PDF-")

    path = Path(root) / "pdfs" / f"{doc_id}.pdf"
    _atomic_write(path, data)
    return path


def save_golden(root: Path | str, record: GoldenRecord) -> Path:
    """Write a hand label, replacing any existing one for the same document."""
    if not record.doc_id:
        raise CorpusError("a golden record needs a doc_id")
    path = Path(root) / "golden" / f"{record.doc_id}.json"
    payload = record.model_dump_json(indent=2) + "\n"
    _atomic_write(path, payload.encode("utf-8"))
    return path


def load_golden(root: Path | str, doc_id: str) -> GoldenRecord | None:
    """Read one label back, or None when the document is not labelled yet."""
    path = Path(root) / "golden" / f"{doc_id}.json"
    if not path.is_file():
        return None
    try:
        return GoldenRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CorpusError(f"{path}: not a valid GoldenRecord ({exc})") from exc


def unlabelled_sources(root: Path | str) -> list[str]:
    """Documents present under ``pdfs/`` or ``text/`` with no label yet.

    These are invisible to `load_corpus` by design -- the golden set defines the
    corpus -- but the interface has to surface them, or a freshly uploaded file
    vanishes with no explanation of what to do next.
    """
    root = Path(root)
    labelled = {p.stem for p in (root / "golden").glob("*.json")}
    found: set[str] = set()
    for sub, suffix in (("pdfs", "*.pdf"), ("text", "*.json")):
        directory = root / sub
        if directory.is_dir():
            found.update(p.stem for p in directory.glob(suffix))
    return sorted(found - labelled)


def load_corpus(root: Path | str = DEFAULT_CORPUS) -> Corpus:
    """Load every golden record under ``root`` and locate its document.

    Items come back sorted by ``doc_id`` so that two runs of the same corpus
    produce the same ordering -- the conformal calibration split is taken by
    document position, and a split that reshuffles between runs would make the
    gate non-deterministic.
    """
    root = Path(root)
    golden_dir = root / "golden"
    if not golden_dir.is_dir():
        raise CorpusError(
            f"{root}: no golden/ directory. A corpus is defined by its labels; "
            f"expected {golden_dir}"
        )

    items: list[CorpusItem] = []
    for path in sorted(golden_dir.glob("*.json")):
        try:
            golden = GoldenRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise CorpusError(f"{path}: not a valid GoldenRecord ({exc})") from exc
        source, kind = _resolve_source(root, golden.doc_id)
        items.append(CorpusItem(golden=golden, source=source, kind=kind))

    return Corpus(root=root, items=items)
