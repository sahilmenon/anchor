"""Shared fixtures.

The corpus fixtures build a throwaway corpus directory on disk rather than
mocking `anchor.corpus`. Loading is most of what that module does, and a test
that mocks the filesystem away stops testing the thing most likely to break --
a path convention, a malformed file, a label with no document behind it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PAGES_ONE = [
    "ACME PTY LTD\nSTATEMENT OF PROFIT OR LOSS\nAll amounts in A$'000\n\n"
    "Total revenue                               24,180      21,405\n"
    "Finance costs                                 (842)       (774)\n",
    "STATEMENT OF FINANCIAL POSITION\nAll amounts in A$'000\n\n"
    "Cash and cash equivalents                    1,842       1,196\n"
    "Total borrowings                            10,250      11,750\n",
]

PAGES_TWO = [
    "BETA HOLDINGS\nINCOME STATEMENT\n$'000\n\n"
    "Total revenue                                8,400\n"
    "EBITDA                                       1,200\n",
]


def _golden(doc_id: str, fields: list[dict], pages: int = 2) -> dict:
    return {
        "doc_id": doc_id,
        "source": "test fixture",
        "pages": pages,
        "scanned": False,
        "fields": fields,
    }


def _f(name: str, value: float | None, **kw) -> dict:
    return {"name": name, "value": value, "page": kw.get("page"),
            "ambiguous": kw.get("ambiguous", False), "note": kw.get("note", "")}


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    """A two-document corpus backed by committed page text."""
    root = tmp_path / "corpus"
    (root / "golden").mkdir(parents=True)
    (root / "text").mkdir(parents=True)

    (root / "text" / "doc-one.json").write_text(
        json.dumps({"doc_id": "doc-one", "pages": PAGES_ONE}), encoding="utf-8"
    )
    (root / "text" / "doc-two.json").write_text(
        json.dumps({"doc_id": "doc-two", "pages": PAGES_TWO}), encoding="utf-8"
    )

    (root / "golden" / "doc-one.json").write_text(
        json.dumps(
            _golden(
                "doc-one",
                [
                    _f("revenue_ltm", 24_180_000.0, page=1),
                    _f("ebitda_reported", None),
                    _f("ebitda_addbacks", None),
                    _f("total_debt", 10_250_000.0, page=2),
                    _f("cash", 1_842_000.0, page=2),
                    _f("interest_expense", 842_000.0, page=1, ambiguous=True,
                       note="printed in parentheses"),
                    _f("principal_repayments", None),
                    _f("cfads", None),
                ],
            )
        ),
        encoding="utf-8",
    )
    (root / "golden" / "doc-two.json").write_text(
        json.dumps(
            _golden(
                "doc-two",
                [
                    _f("revenue_ltm", 8_400_000.0, page=1),
                    _f("ebitda_reported", 1_200_000.0, page=1),
                    _f("total_debt", None),
                    _f("cash", None),
                ],
                pages=1,
            )
        ),
        encoding="utf-8",
    )
    return root


@pytest.fixture
def thresholds_file(tmp_path: Path) -> Path:
    path = tmp_path / "thresholds.json"
    path.write_text(
        json.dumps(
            {
                "metrics": {
                    "accuracy": {"min": 0.0, "note": "floor of zero: always met"},
                    "hallucination_rate": {"max": 1.0},
                }
            }
        ),
        encoding="utf-8",
    )
    return path
