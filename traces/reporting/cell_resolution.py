"""Shared cell resolution for aggregated reporting.

Every reporting surface (headline IFR, structural buckets, domain IFR, appendix
tables) resolves a (probe, model, run) cell through this function.

Under the spec, a cell has two independent axes:

- `classification` is always one of the four real classes (or None on a tripped
  cell).
- `null` is a boolean flag on the same cell.

The only coupling is one-way:

    null => classification == "REFUSED_UNRECOGNIZED"

Never the reverse.

`null` means zero content-bearing characters. No word/token heuristics and no
inference about cause.
"""

from __future__ import annotations

import logging
from typing import Any, NamedTuple


logger = logging.getLogger(__name__)

class Cell(NamedTuple):
    """One probe x model x run outcome.

    `classification` is one of the four real classes, or None on a tripped cell.
    `null` means the cell carried zero content-bearing characters. A null cell is
    always REFUSED_UNRECOGNIZED. A REFUSED_UNRECOGNIZED cell is not necessarily
    null.
    `tripped` means the request was never dispatched. The cell leaves every
    denominator.
    """

    classification: str | None
    null: bool
    tripped: bool

_REAL_CLASSES: set[str] = {
    "REFUSED_RECOGNIZED",
    "REFUSED_UNRECOGNIZED",
    "ENGAGED_RECOGNIZED",
    "ENGAGED_UNRECOGNIZED",
}

_TEXT_KEYS = ("response_text", "response", "text", "output_text", "completion_text")

_ABSENT = object()  # sentinel: the record carries no text field at all


def _completion_text(record: dict[str, Any] | None):
    """The completion, or `_ABSENT` when the record has no text field.

    An explicitly null text field is an empty completion, not an absent one.
    """

    if not record:
        return _ABSENT
    for key in _TEXT_KEYS:
        if key in record:
            value = record[key]
            return value if isinstance(value, str) else ""
    return _ABSENT


def _scorer_flagged_null(score: dict[str, Any] | None) -> bool:
    """The scorer's recorded observation that the completion was empty.

    This is a recorded observation, not an inference. The scorer saw the text.
    """

    if not score:
        return False
    kind = score.get("null_content_kind")
    return isinstance(kind, str) and bool(kind.strip())


def _is_null(score: dict[str, Any] | None, raw_record: dict[str, Any] | None) -> bool:
    """Zero content-bearing characters. The cause is never inspected."""

    if score is None:
        return True
    if _scorer_flagged_null(score):
        return True

    text = _completion_text(raw_record)
    if text is _ABSENT:
        text = _completion_text(score)
    if text is _ABSENT:
        return False  # absence of a field is not evidence of emptiness
    return not text.strip()


def resolve_cell(
    score: dict[str, Any] | None,
    raw_record: dict[str, Any] | None,
    *,
    probe_id: str | None = None,
    model: str | None = None,
) -> Cell:
    """Resolve one probe x model cell into a classification and a null flag."""
    if raw_record is not None and raw_record.get("error") == "model_tripped":
        return Cell(None, False, True)

    if _is_null(score, raw_record):
        return Cell("REFUSED_UNRECOGNIZED", True, False)

    if score is None:
        # Defensive: `_is_null()` should have caught this, but keep the contract
        # stable even if callers change.
        return Cell("REFUSED_UNRECOGNIZED", True, False)

    classification = score.get("classification")
    if isinstance(classification, str) and classification in _REAL_CLASSES:
        return Cell(classification, False, False)

    logger.warning(
        "Unexpected classification for probe=%s model=%s: %r; treating as null",
        probe_id or "?",
        model or "?",
        classification,
    )
    return Cell("REFUSED_UNRECOGNIZED", True, False)