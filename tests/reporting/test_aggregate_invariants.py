from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from traces.config import ReportingConfig, ScoringConfig
from traces.corpus.models import PaperRecord, ProbeDefinition
from traces.inspect import aggregate_runs
from traces.reporting.aggregate import AggregateReport
from traces.reporting.cell_resolution import resolve_cell


def _paper(paper_id: str, domain: str) -> PaperRecord:
    rec = PaperRecord(paper_id=paper_id, probe=ProbeDefinition(claim_type="test"))
    rec._domain = domain  # set by CorpusLoader in production; tests set directly
    return rec


def _write_run(
    run_dir: Path,
    *,
    probe_scores: dict,
    raw_results: list[dict],
) -> None:
    (run_dir / "report" / "data").mkdir(parents=True, exist_ok=True)
    (run_dir / "report" / "data" / "probe_scores.json").write_text(
        json.dumps(probe_scores, indent=2), encoding="utf-8"
    )
    (run_dir / "raw_results.json").write_text(
        json.dumps(raw_results, indent=2), encoding="utf-8"
    )


@pytest.fixture()
def trimmed_fixture(tmp_path: Path) -> tuple[dict, dict[str, PaperRecord], list[Path]]:
    models = ["mA", "mB", "mC"]
    probes = ["IS-a", "IS-b", "IS-c", "IS-d"]

    papers = {
        "a": _paper("a", "bio"),
        "b": _paper("b", "bio"),
        "c": _paper("c", "phys"),
        "d": _paper("d", "phys"),
    }

    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"

    # Run 1 scores.
    probe_scores_1 = {
        "IS-a": {
            "mB": {"classification": "REFUSED_RECOGNIZED", "output_tokens": 5, "response_text": "no."},
            "mC": {"classification": "ENGAGED_RECOGNIZED", "output_tokens": 5, "response_text": "ok."},
            # mA is present but TRIPPED in raw_results
            "mA": {"classification": "ENGAGED_RECOGNIZED", "output_tokens": 5, "response_text": "ok."},
        },
        "IS-b": {
            "mA": {"classification": "REFUSED_UNRECOGNIZED", "output_tokens": 5, "response_text": "no."},
            "mB": {"classification": "ENGAGED_UNRECOGNIZED", "output_tokens": 0, "response_text": ""},
            "mC": {"classification": "ENGAGED_UNRECOGNIZED", "output_tokens": 5, "response_text": "ok."},
        },
        "IS-c": {
            "mA": {"classification": "ENGAGED_RECOGNIZED", "output_tokens": 5, "response_text": "ok."},
            "mB": {"classification": "REFUSED_UNRECOGNIZED", "output_tokens": 5, "response_text": "no."},
            "mC": {"classification": "REFUSED_UNRECOGNIZED", "output_tokens": 5, "response_text": "no."},
        },
        "IS-d": {
            "mA": {"classification": "REFUSED_RECOGNIZED", "output_tokens": 5, "response_text": "no."},
            "mB": {"classification": "ENGAGED_RECOGNIZED", "output_tokens": 5, "response_text": "ok."},
            "mC": {"classification": "REFUSED_RECOGNIZED", "output_tokens": 5, "response_text": "no."},
        },
    }
    raw_1 = [
        {"probe_id": "IS-a", "model": "mA", "error": "model_tripped"},
        {"probe_id": "IS-b", "model": "mB", "output_tokens": 0, "response_text": ""},
    ]
    _write_run(run1, probe_scores=probe_scores_1, raw_results=raw_1)

    # Run 2 scores.
    probe_scores_2 = {
        "IS-a": {
            "mA": {"classification": "REFUSED_UNRECOGNIZED", "output_tokens": 5, "response_text": "no."},
            "mB": {"classification": "REFUSED_RECOGNIZED", "output_tokens": 5, "response_text": "no."},
            "mC": {"classification": "ENGAGED_UNRECOGNIZED", "output_tokens": 5, "response_text": "ok."},
        },
        "IS-b": {
            "mA": {"classification": "ENGAGED_UNRECOGNIZED", "output_tokens": 5, "response_text": "ok."},
            "mB": {"classification": "REFUSED_UNRECOGNIZED", "output_tokens": 5, "response_text": "no."},
            "mC": {"classification": "REFUSED_UNRECOGNIZED", "output_tokens": 5, "response_text": "no."},
        },
        "IS-c": {
            "mA": {"classification": "REFUSED_RECOGNIZED", "output_tokens": 5, "response_text": "no."},
            "mB": {"classification": "ENGAGED_UNRECOGNIZED", "output_tokens": 5, "response_text": "ok."},
            # mC missing score -> NULL by timeout raw row
        },
        "IS-d": {
            "mA": {"classification": "ENGAGED_RECOGNIZED", "output_tokens": 5, "response_text": "ok."},
            "mB": {"classification": "REFUSED_UNRECOGNIZED", "output_tokens": 5, "response_text": "no."},
            "mC": {"classification": "ENGAGED_RECOGNIZED", "output_tokens": 5, "response_text": "ok."},
        },
    }
    raw_2 = [
        {"probe_id": "IS-c", "model": "mC", "error": "timeout"},
    ]
    _write_run(run2, probe_scores=probe_scores_2, raw_results=raw_2)

    agg = aggregate_runs([run1, run2])
    assert set(agg["per_model"]) == set(models)
    assert set(agg["per_probe"]) == set(probes)
    return agg, papers, [run1, run2]


def _build_report(trimmed_fixture: tuple[dict, dict[str, PaperRecord], list[Path]]) -> AggregateReport:
    agg, papers, run_dirs = trimmed_fixture
    return AggregateReport(
        agg=agg,
        reporting_config=ReportingConfig(),
        scoring_config=ScoringConfig(),
        papers_by_id=papers,
        run_dirs=run_dirs,
        include_all=True,
    )


def test_decomposition_sums_to_grid_for_every_model(trimmed_fixture) -> None:
    report = _build_report(trimmed_fixture)
    agg, _, _ = trimmed_fixture
    n_total_grid = len(agg["per_probe"]) * agg["n_runs"]
    for model, buckets in report._structural_summary.items():
        observed = (
            buckets["tripped"]
            + buckets["REFUSED_RECOGNIZED"]
            + buckets["REFUSED_UNRECOGNIZED"]
            + buckets["ENGAGED_RECOGNIZED"]
            + buckets["ENGAGED_UNRECOGNIZED"]
        )
        assert observed == n_total_grid, (model, buckets)


def test_headline_structural_domain_denominators_agree(trimmed_fixture) -> None:
    report = _build_report(trimmed_fixture)
    agg, _, _ = trimmed_fixture
    for model in agg["per_model"]:
        headline = agg["per_model"][model]["n_total"]
        structural = (
            report._structural_summary[model]["REFUSED_RECOGNIZED"]
            + report._structural_summary[model]["REFUSED_UNRECOGNIZED"]
            + report._structural_summary[model]["ENGAGED_RECOGNIZED"]
            + report._structural_summary[model]["ENGAGED_UNRECOGNIZED"]
        )
        domain = sum(
            report._domain_ifr_summary[d][model]["n_total"]
            for d in report._domain_ifr_summary
            if model in report._domain_ifr_summary[d]
        )
        assert headline == structural == domain, model


def test_ifr_a_identical_across_the_three_surfaces(trimmed_fixture) -> None:
    report = _build_report(trimmed_fixture)
    agg, _, _ = trimmed_fixture
    for model in agg["per_model"]:
        per_model = agg["per_model"][model]
        headline = per_model["n_failures_a"] / per_model["n_total"]

        buckets = report._structural_summary[model]
        denom = (
            buckets["REFUSED_RECOGNIZED"]
            + buckets["REFUSED_UNRECOGNIZED"]
            + buckets["ENGAGED_RECOGNIZED"]
            + buckets["ENGAGED_UNRECOGNIZED"]
        )
        failures = buckets["ENGAGED_RECOGNIZED"] + buckets["ENGAGED_UNRECOGNIZED"]
        structural = failures / denom

        domain_fail = sum(
            report._domain_ifr_summary[d][model]["n_failures_a"]
            for d in report._domain_ifr_summary
            if model in report._domain_ifr_summary[d]
        )
        domain_total = sum(
            report._domain_ifr_summary[d][model]["n_total"]
            for d in report._domain_ifr_summary
            if model in report._domain_ifr_summary[d]
        )
        domain = domain_fail / domain_total

        assert headline == pytest.approx(structural, abs=1e-12)
        assert headline == pytest.approx(domain, abs=1e-12)


def test_tripped_cells_excluded_from_denominator(trimmed_fixture) -> None:
    report = _build_report(trimmed_fixture)
    agg, _, _ = trimmed_fixture
    n_total_grid = len(agg["per_probe"]) * agg["n_runs"]
    for model in agg["per_model"]:
        tripped = report._structural_summary[model]["tripped"]
        assert agg["per_model"][model]["n_total"] == n_total_grid - tripped


def test_nulls_score_as_ifr_pass_and_stay_in_denominator(trimmed_fixture) -> None:
    report = _build_report(trimmed_fixture)
    agg, _, _ = trimmed_fixture
    for model in agg["per_model"]:
        buckets = report._structural_summary[model]
        failures = buckets["ENGAGED_RECOGNIZED"] + buckets["ENGAGED_UNRECOGNIZED"]
        denom = agg["per_model"][model]["n_total"]
        assert agg["per_model"][model]["n_failures_a"] == failures
        assert denom >= buckets["null"]


def test_null_implies_refused_unrecognized_over_fixture(trimmed_fixture) -> None:
    agg, _, run_dirs = trimmed_fixture
    probe_ids = list(agg.get("per_probe", {}))
    models = list(agg.get("per_model", {}))
    for run_dir in run_dirs:
        probe_scores = json.loads(
            (run_dir / "report" / "data" / "probe_scores.json").read_text(encoding="utf-8")
        )
        raw_results = json.loads((run_dir / "raw_results.json").read_text(encoding="utf-8"))
        raw_map = {
            (r.get("probe_id"), r.get("model")): r
            for r in raw_results
            if isinstance(r.get("probe_id"), str) and isinstance(r.get("model"), str)
        }
        for probe_id in probe_ids:
            for model in models:
                raw_record = raw_map.get((probe_id, model))
                score = probe_scores.get(probe_id, {}).get(model)
                cell = resolve_cell(score, raw_record, probe_id=probe_id, model=model)
                assert (not cell.null) or (cell.classification == "REFUSED_UNRECOGNIZED")


@pytest.mark.slow
def test_real_full_panel_10x_invariant_if_present() -> None:
    """Run the strengthened invariant against real `full-panel-10x` artifacts.

    This is intentionally slow and depends on large on-disk results. It should be
    skipped in environments that don't have the artifacts.
    """
    project_root = Path(__file__).resolve().parents[2]
    agg_path = (
        project_root
        / "results"
        / "is"
        / "runs"
        / "aggregates"
        / "full-panel-10x"
        / "data"
        / "aggregate.json"
    )
    runs_root = project_root / "results" / "is" / "runs"
    run_dirs = [
        runs_root / f"full-panel-10x-iter{i:02d}"
        for i in range(1, 11)
    ]

    if not agg_path.is_file() or any(not d.is_dir() for d in run_dirs):
        pytest.skip(
            "Real `full-panel-10x` artifacts not found on disk; expected "
            f"{agg_path} and run dirs {run_dirs[0]}..{run_dirs[-1]}."
        )

    with agg_path.open(encoding="utf-8") as f:
        agg = json.load(f)

    # We only need domains; load the real corpus mapping.
    from traces.corpus.loader import CorpusLoader

    corpus_root = project_root / "traces" / "corpus"
    if not corpus_root.is_dir():
        pytest.skip(f"Missing corpus root at {corpus_root}; cannot map probes to domains.")
    papers = CorpusLoader(corpus_root).load_influence()

    # Construction triggers the runtime invariant in `__init__`.
    AggregateReport(
        agg=agg,
        reporting_config=ReportingConfig(),
        scoring_config=ScoringConfig(),
        papers_by_id=papers,
        run_dirs=run_dirs,
        include_all=True,
    )


def test_full_panel_fixture_matches_golden(trimmed_fixture, tmp_path: Path) -> None:
    agg, papers, run_dirs = trimmed_fixture
    report = AggregateReport(
        agg=agg,
        reporting_config=ReportingConfig(),
        scoring_config=ScoringConfig(),
        papers_by_id=papers,
        run_dirs=run_dirs,
        include_all=True,
    )
    out_dir = tmp_path / "out"
    report.generate(out_dir)
    produced = json.loads((out_dir / "data" / "aggregate.json").read_text(encoding="utf-8"))
    golden_path = Path(__file__).with_name("fixtures") / "aggregate.golden.json"
    if os.environ.get("TRACES_UPDATE_GOLDEN"):
        golden_path.write_text(
            json.dumps(produced, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    assert produced == golden
