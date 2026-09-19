"""Anchor -- an evaluation harness for evidence-anchored credit-metric extraction."""

from anchor.schema import (
    Evidence,
    ExtractedField,
    Extraction,
    GoldenField,
    GoldenRecord,
    LineItem,
)

# Note: anchor.corpus, anchor.runner, anchor.gate and anchor.web are not
# imported here on purpose. They pull in the extractor registry and the scoring
# stack, and `import anchor` is also what a bare `anchor.schema` consumer pays
# for. The CLI imports them directly.

__version__ = "0.1.0"

__all__ = [
    "Evidence",
    "ExtractedField",
    "Extraction",
    "GoldenField",
    "GoldenRecord",
    "LineItem",
]
