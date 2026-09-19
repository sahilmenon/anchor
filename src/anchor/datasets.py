"""Importing external key-information-extraction datasets as Anchor corpora.

Why bother
----------
Hand-labelling a corpus is slow and produces a number nobody outside this
repository can check. An external dataset gives you documents that are real,
messy, and already annotated by someone else, plus a published leaderboard to
sit your own figure next to. That is worth more than a larger bespoke corpus.

What an imported corpus cannot do
---------------------------------
It cannot replace the hand-labelled one, and the reason is specific rather than
territorial. Every dataset surveyed labels **values that are present**. Anchor's
two most interesting measurements need labels that no external set carries:

* **Absence** -- "this document does not state EBITDA", as the *correct*
  answer. That is the hallucination test set.
* **Ambiguity** -- "two competent analysts would disagree here". That is the
  abstention test set.

Kleister is the partial exception, and it is why it is the first adapter: its
task definition includes decoy keys, "for which no value should be given". A
key listed for a document with no corresponding value is a real absence label,
and it maps exactly onto a golden `value` of None.

The rule that keeps an import honest
------------------------------------
**Only emit fields the source dataset actually annotates.** `score_document`
iterates the golden record's fields, so a field left out of the record is never
scored. That matters more than it sounds: writing ``total_debt: null`` into a
record for a dataset that never looked at total debt would score an extractor
that correctly read the balance sheet as a HALLUCINATION. Omitting the field
scores nothing and claims nothing, which is the truthful outcome.

A consequence worth expecting: an imported corpus has few or no ambiguous
fields, so `abstention_precision` and the `on_ambiguous` block come back as
``None`` rather than as a flattering number. That is the zero-denominator
convention doing its job.
"""

from __future__ import annotations

import json
import lzma
from dataclasses import dataclass
from dataclasses import field as dfield
from pathlib import Path

from anchor.corpus import CorpusError, safe_doc_id, save_golden
from anchor.schema import GoldenField, GoldenRecord, LineItem

__all__ = [
    "DATASETS",
    "DatasetSpec",
    "ImportResult",
    "import_kleister",
]


@dataclass(frozen=True)
class DatasetSpec:
    """What a dataset is, and what you are allowed to do with it."""

    name: str
    description: str
    url: str
    licence: str
    committable: bool
    """May labels derived from it be committed to this repository?

    False where the dataset declares no licence. Anchor then writes the
    imported corpus somewhere git-ignored, so a provenance question never
    becomes a distribution question.
    """
    field_map: dict[str, LineItem] = dfield(default_factory=dict)
    notes: str = ""


KLEISTER_CHARITY = DatasetSpec(
    name="kleister-charity",
    description=(
        "2,788 annual financial reports filed with the Charity Commission for "
        "England and Wales. Mixed scanned and born-digital, OCR'd four ways, "
        "long and inconsistently laid out."
    ),
    url="https://github.com/applicaai/kleister-charity",
    licence="none declared",
    committable=False,
    field_map={"income_annually_in_british_pounds": LineItem.REVENUE_LTM},
    notes=(
        "Only one of Anchor's eight line items has a counterpart. "
        "`spending_annually_in_british_pounds` has none: it is neither a credit "
        "line item nor derivable into one. The remaining six are not annotated "
        "and are therefore omitted rather than labelled absent."
    ),
)

#: Datasets with a working adapter, by name.
DATASETS: dict[str, DatasetSpec] = {KLEISTER_CHARITY.name: KLEISTER_CHARITY}


@dataclass
class ImportResult:
    """What an import produced, in enough detail to audit it."""

    dataset: str
    out: Path
    n_documents: int = 0
    n_values: int = 0
    """Fields imported with a stated value."""
    n_absent: int = 0
    """Fields imported as a genuine absence (a decoy key)."""
    n_fields_omitted_per_doc: int = 0
    """Line items the dataset does not annotate, left unscored."""
    doc_ids: list[str] = dfield(default_factory=list)
    committable: bool = False

    def describe(self) -> str:
        spec = DATASETS.get(self.dataset)
        lines = [
            f"imported {self.n_documents} document(s) from {self.dataset} into {self.out}",
            f"  {self.n_values} labelled value(s), {self.n_absent} labelled absence(s)",
            f"  {self.n_fields_omitted_per_doc} of {len(LineItem)} line items are not "
            "annotated by this dataset and are omitted, so they are never scored",
        ]
        if spec and not spec.committable:
            lines.append(
                f"  licence: {spec.licence}. The imported corpus is git-ignored; "
                "do not commit labels derived from it."
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Kleister
# ---------------------------------------------------------------------------

#: How Kleister escapes newlines inside its tab-separated text columns. Written
#: with chr() rather than a literal so the intent survives every layer of
#: quoting between here and a shell.
_ESCAPED_NEWLINE = chr(92) + "n"

#: Columns of `in.tsv`, in order. The text is shipped four ways; `text_best`
#: combines the pdf2djvu and tesseract passes and is the sensible default.
_IN_COLUMNS = (
    "filename",
    "keys",
    "text_djvu",
    "text_tesseract",
    "text_textract",
    "text_best",
)


def _parse_expected(line: str) -> dict[str, str]:
    """One line of `expected.tsv` into a mapping.

    The format is space-separated ``key=value`` pairs with spaces inside values
    replaced by underscores. A key may legitimately repeat when the annotators
    allowed more than one acceptable answer; the first is taken, because Anchor
    scores against a single ground truth and picking arbitrarily would be worse
    than picking first consistently.
    """
    out: dict[str, str] = {}
    for token in line.split():
        if "=" in token:
            key, value = token.split("=", 1)
            out.setdefault(key, value)
    return out


def _to_float(raw: str) -> float | None:
    try:
        return float(raw.replace(",", "").replace("_", ""))
    except ValueError:
        return None


def _read_split(source: Path, split: str) -> list[tuple[list[str], dict[str, str]]]:
    """Pair each `in.tsv` row with its `expected.tsv` labels."""
    in_path = source / split / "in.tsv.xz"
    expected_path = source / split / "expected.tsv"
    if not in_path.is_file():
        raise CorpusError(
            f"{in_path} not found. Clone the dataset first:\n"
            f"  git clone {KLEISTER_CHARITY.url}"
        )
    if not expected_path.is_file():
        raise CorpusError(
            f"{expected_path} not found. The {split!r} split has no labels; "
            "test splits are withheld by the dataset authors."
        )

    with lzma.open(in_path, "rt", encoding="utf-8", errors="replace") as fh:
        rows = [line.rstrip("\n").split("\t") for line in fh if line.strip()]
    expected = [
        _parse_expected(line)
        for line in expected_path.read_text(encoding="utf-8", errors="replace").splitlines()
    ]

    if len(rows) != len(expected):
        raise CorpusError(
            f"{split}: {len(rows)} document(s) but {len(expected)} label line(s). "
            "The split is inconsistent; re-clone it."
        )
    return list(zip(rows, expected, strict=True))


def import_kleister(
    source: Path | str,
    out: Path | str,
    *,
    split: str = "dev-0",
    limit: int | None = None,
    text_column: str = "text_best",
    mark_ambiguous: bool = True,
) -> ImportResult:
    """Build an Anchor corpus from a local Kleister-Charity checkout.

    `source` is the cloned repository. The PDFs live in git-annex and are not
    needed: the repository ships the OCR'd text, which is what Anchor's verifier
    checks quotes against anyway.

    Page structure
    --------------
    Kleister's text is one blob per document with no page separators, so each
    document is imported as a **single page**. Evidence checking still works --
    a quote must appear in the text -- but page-level citation degrades to
    document-level, and every imported `page` is 1. An extractor cannot be
    wrong about the page here, so grounding numbers from an imported corpus are
    not comparable with grounding numbers from a paginated one.

    `mark_ambiguous`
    ----------------
    On by default, and the reason is the mapping rather than the dataset.
    Kleister's `income_annually_in_british_pounds` is a charity's total
    incoming resources, grants included. Whether that is "revenue" for a credit
    assessment is a lender's call -- the same judgement Anchor's own corpus
    flags on any NFP. Importing it as unambiguous ground truth would launder a
    definitional choice into a fact.
    """
    source, out = Path(source), Path(out)
    spec = KLEISTER_CHARITY
    if text_column not in _IN_COLUMNS:
        raise CorpusError(
            f"unknown text column {text_column!r}. One of: {', '.join(_IN_COLUMNS[2:])}"
        )
    column = _IN_COLUMNS.index(text_column)

    pairs = _read_split(source, split)
    if limit is not None:
        pairs = pairs[:limit]

    text_dir = out / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    (out / "golden").mkdir(parents=True, exist_ok=True)

    result = ImportResult(
        dataset=spec.name,
        out=out,
        n_fields_omitted_per_doc=len(LineItem) - len(spec.field_map),
        committable=spec.committable,
    )

    for row, labels in pairs:
        if len(row) <= column:
            continue
        doc_id = safe_doc_id(f"kleister-{row[0]}")
        text = row[column].replace(_ESCAPED_NEWLINE, "\n")
        if not text.strip():
            continue

        (text_dir / f"{doc_id}.json").write_text(
            json.dumps({"doc_id": doc_id, "pages": [text]}, indent=2) + "\n",
            encoding="utf-8",
        )

        requested = set(row[1].split())
        fields: list[GoldenField] = []
        for source_key, line_item in spec.field_map.items():
            # Not asked for on this document: the annotators never looked, so
            # Anchor must not score it either way.
            if source_key not in requested:
                continue

            raw = labels.get(source_key)
            if raw is None:
                # Asked for and deliberately unanswered -- a decoy key. This is
                # a genuine absence label and the only kind any surveyed
                # dataset provides.
                fields.append(
                    GoldenField(
                        name=line_item,
                        value=None,
                        ambiguous=False,
                        note=(
                            f"{spec.name}: '{source_key}' was among the keys requested "
                            "for this document and no value is expected. A decoy key, "
                            "imported as a true absence."
                        ),
                    )
                )
                result.n_absent += 1
                continue

            value = _to_float(raw)
            if value is None:
                continue
            fields.append(
                GoldenField(
                    name=line_item,
                    value=value,
                    page=1,
                    ambiguous=mark_ambiguous,
                    note=(
                        f"{spec.name}: imported from '{source_key}' = {raw}. Total "
                        "incoming resources including grants; whether that is revenue "
                        "for a credit assessment is a lender's call. The label is in GBP "
                        "and the annotators converted where a charity reports in "
                        "another currency, so an extractor reading the printed "
                        "figure disagrees on those documents without misreading "
                        "the page. Page is 1 because "
                        "the source text carries no page separators."
                    ),
                )
            )
            result.n_values += 1

        if not fields:
            (text_dir / f"{doc_id}.json").unlink(missing_ok=True)
            continue

        save_golden(
            out,
            GoldenRecord(
                doc_id=doc_id,
                source=f"{spec.name} {split} ({spec.url}); original file {row[0]}",
                pages=1,
                scanned=True,
                fields=fields,
            ),
        )
        result.n_documents += 1
        result.doc_ids.append(doc_id)

    _write_readme(out, spec, split, result)
    return result


def _licence_warning(spec: DatasetSpec) -> str:
    if spec.committable:
        return ""
    return (
        "**Do not commit this directory.** The source dataset declares no "
        "licence, so labels derived from it stay local."
    )


def _write_readme(out: Path, spec: DatasetSpec, split: str, result: ImportResult) -> None:
    """Leave a note in the imported corpus saying what it is and is not."""
    (out / "README.md").write_text(
        f"""# Imported corpus: {spec.name}

Generated by `anchor import`. Do not hand-edit; re-import instead.

- **Source**: {spec.url} ({split} split)
- **Licence**: {spec.licence}
- **Documents**: {result.n_documents}
- **Labelled values**: {result.n_values}
- **Labelled absences**: {result.n_absent} (decoy keys)

## What this corpus does not measure

{result.n_fields_omitted_per_doc} of Anchor's {len(LineItem)} line items are not
annotated by this dataset. They are omitted from every golden record rather than
labelled absent, so they are never scored. An extractor that reads them
correctly is neither rewarded nor punished here.

Page-level citation is degraded: the source ships one text blob per document
with no page separators, so every document is a single page and every `page` is
1. Quote verification still works. Grounding rates from this corpus are not
comparable with grounding rates from a paginated one.

Almost every imported value is flagged `ambiguous`, because the mapping from a
charity's total incoming resources to "revenue for a credit assessment" is a
judgement rather than a reading. Expect `abstention_precision` to be dominated
by that single decision.

{_licence_warning(spec)}
""",
        encoding="utf-8",
    )
