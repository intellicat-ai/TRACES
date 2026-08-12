from __future__ import annotations

import logging

import pytest

from traces.reporting.cell_resolution import Cell, resolve_cell


def test_tripped_takes_priority_over_everything() -> None:
    score = {"classification": "ENGAGED_RECOGNIZED", "output_tokens": 10, "response_text": "ok"}
    raw = {"error": "model_tripped", "output_tokens": 10, "response_text": "ok"}
    assert resolve_cell(score, raw, probe_id="p", model="m") == Cell(None, False, True)


def test_empty_content_with_valid_classification_is_null() -> None:
    score = {"classification": "ENGAGED_UNRECOGNIZED", "output_tokens": 0, "response_text": ""}
    raw = {"output_tokens": 0, "response_text": ""}
    assert resolve_cell(score, raw, probe_id="p", model="m") == Cell(
        "REFUSED_UNRECOGNIZED", True, False
    )


def test_whitespace_only_response_is_null() -> None:
    score = {"classification": "REFUSED_UNRECOGNIZED", "output_tokens": 5, "response_text": "   \n\t "}
    raw = {"output_tokens": 5, "response_text": "   \n\t "}
    assert resolve_cell(score, raw, probe_id="p", model="m") == Cell(
        "REFUSED_UNRECOGNIZED", True, False
    )


def test_missing_score_with_timeout_error_is_null() -> None:
    raw = {"error": "timeout"}
    assert resolve_cell(None, raw, probe_id="p", model="m") == Cell(
        "REFUSED_UNRECOGNIZED", True, False
    )


def test_missing_score_without_error_is_null() -> None:
    assert resolve_cell(None, None, probe_id="probe-x", model="model-y") == Cell(
        "REFUSED_UNRECOGNIZED", True, False
    )


def test_unknown_classification_is_null_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    score = {"classification": "SOMETHING_NEW", "output_tokens": 10, "response_text": "hi"}
    raw = {"output_tokens": 10, "response_text": "hi"}
    assert resolve_cell(score, raw, probe_id="probe-x", model="model-y") == Cell(
        "REFUSED_UNRECOGNIZED", True, False
    )
    assert any("probe=probe-x" in rec.message and "model=model-y" in rec.message for rec in caplog.records)


@pytest.mark.parametrize(
    "cls",
    [
        "REFUSED_RECOGNIZED",
        "REFUSED_UNRECOGNIZED",
        "ENGAGED_RECOGNIZED",
        "ENGAGED_UNRECOGNIZED",
    ],
)
def test_four_real_classes_pass_through(cls: str) -> None:
    score = {"classification": cls, "output_tokens": 3, "response_text": "ok"}
    raw = {"output_tokens": 3, "response_text": "ok"}
    assert resolve_cell(score, raw, probe_id="p", model="m") == Cell(cls, False, False)


def test_output_tokens_absent_falls_back_to_text_length() -> None:
    score = {"classification": "ENGAGED_UNRECOGNIZED", "response_text": ""}
    raw = {"response_text": ""}
    assert resolve_cell(score, raw, probe_id="p", model="m") == Cell(
        "REFUSED_UNRECOGNIZED", True, False
    )
    score2 = {"classification": "ENGAGED_UNRECOGNIZED", "response_text": "hello"}
    raw2 = {"response_text": "hello"}
    assert resolve_cell(score2, raw2, probe_id="p", model="m") == Cell(
        "ENGAGED_UNRECOGNIZED", False, False
    )


def test_short_unpunctuated_engaged_with_length_finish_reason_is_not_demoted_to_null() -> None:
    score = {
        "classification": "ENGAGED_UNRECOGNIZED",
        "finish_reason": "length",
        "response_text": "yes",
        "output_tokens": 3,
    }
    raw = {
        "finish_reason": "length",
        "response_text": "yes",
        "output_tokens": 3,
    }
    assert resolve_cell(score, raw, probe_id="p", model="m") == Cell(
        "ENGAGED_UNRECOGNIZED", False, False
    )


def test_response_text_none_with_large_token_count_is_null() -> None:
    score = {
        "classification": "REFUSED_UNRECOGNIZED",
        "completion_tokens": 9999,
        "response_text": None,
    }
    raw = {"completion_tokens": 9999, "response_text": None}
    assert resolve_cell(score, raw, probe_id="p", model="m") == Cell(
        "REFUSED_UNRECOGNIZED", True, False
    )


def test_terse_genuine_refusal_is_not_null() -> None:
    score = {"classification": "REFUSED_UNRECOGNIZED"}
    raw = {"response_text": "I cannot help you with this."}
    assert resolve_cell(score, raw, probe_id="p", model="m") == Cell(
        "REFUSED_UNRECOGNIZED", False, False
    )
