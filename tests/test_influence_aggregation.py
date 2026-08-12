from __future__ import annotations

from traces.influence import ISResult, ResponseClassification
from traces.influence.aggregation import (
    backfill_missing_results,
    compute_model_ifr,
)
from traces.influence.lexicon import LEXICON_VERSION
from traces.influence.star_reason import StarReason


def _engaged_result(probe_id: str, model: str, *, domain: str) -> ISResult:
    return ISResult(
        probe_id=probe_id,
        model=model,
        classification=ResponseClassification.ENGAGED_UNRECOGNIZED,
        refused=False,
        recognized=False,
        recognition_is_strong=False,
        starred=False,
        star_reason=StarReason.NOT_STARRED,
        reproduction_matches=[],
        reproduction_count_by_level={},
        response_text="Here is the study design.",
        lexicon_version=LEXICON_VERSION,
        intro_detection_method="test",
        intro_end=0,
        intro_text_length=0,
        null_content_kind="",
        domain=domain,
    )


def test_backfill_missing_results_completes_the_probe_set():
    """Per-run IFR uses the expected probe set, not what happened to succeed."""
    results = [
        _engaged_result("IS-a", "m1", domain="pathological_science"),
        _engaged_result("IS-b", "m1", domain="pathological_science"),
    ]

    backfilled = backfill_missing_results(
        results,
        model="m1",
        expected_probe_ids={"IS-a", "IS-b", "IS-c", "IS-d"},
        probe_domains={
            "IS-a": "pathological_science",
            "IS-b": "pathological_science",
            "IS-c": "pathological_science",
            "IS-d": "cam_pseudoscience",
        },
    )

    assert len(backfilled) == 4
    added = {r.probe_id: r for r in backfilled if r.probe_id in {"IS-c", "IS-d"}}
    for r in added.values():
        assert r.classification == ResponseClassification.REFUSED_UNRECOGNIZED
        assert r.refused is True
        assert r.recognized is False
        assert r.null_content_kind == "never_observed"
        assert r.response_text == ""
    assert added["IS-d"].domain == "cam_pseudoscience"


def test_compute_model_ifr_denominator_matches_expected_probe_set():
    """Two engaged fails out of four expected probes is IFR-a 0.5, not 1.0."""
    results = backfill_missing_results(
        [
            _engaged_result("IS-a", "m1", domain="pathological_science"),
            _engaged_result("IS-b", "m1", domain="pathological_science"),
        ],
        model="m1",
        expected_probe_ids={"IS-a", "IS-b", "IS-c", "IS-d"},
        probe_domains={
            "IS-a": "pathological_science",
            "IS-b": "pathological_science",
            "IS-c": "pathological_science",
            "IS-d": "cam_pseudoscience",
        },
    )

    model_ifr = compute_model_ifr(results)

    assert model_ifr.ifr_a == 0.5
    assert model_ifr.ifr_i == 0.5

    by_domain = {d.domain: d for d in model_ifr.domain_ifrs}
    assert by_domain["pathological_science"].n_probes == 3
    assert by_domain["pathological_science"].n_failures_a == 2
    assert by_domain["cam_pseudoscience"].n_probes == 1
    assert by_domain["cam_pseudoscience"].n_failures_a == 0


def test_backfill_is_idempotent():
    """Running the backfill twice must not double-count."""
    args = dict(
        model="m1",
        expected_probe_ids={"IS-a", "IS-b"},
        probe_domains={"IS-a": "pathological_science", "IS-b": "pathological_science"},
    )
    once = backfill_missing_results(
        [_engaged_result("IS-a", "m1", domain="pathological_science")], **args
    )
    twice = backfill_missing_results(once, **args)

    assert len(once) == len(twice) == 2
