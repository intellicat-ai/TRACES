"""CLI inspection helpers — read-only views of corpus + run artifacts.

These functions return plain dicts/lists so they can be tested without
formatting concerns and so the CLI layer is free to render them as
tables, JSON, or anything else. No API calls, no scoring — strictly
filesystem + stdlib reads.
"""
from __future__ import annotations

import json
import logging
import statistics
from collections import Counter
from pathlib import Path

from traces.influence.aggregation import (
    IFR_A_PASS as _IFR_A_PASS,
    IFR_I_PASS as _IFR_I_PASS,
    bootstrap_ifr_ci,
    is_ifr_a_pass as _is_ifr_a_pass,
    is_ifr_i_pass as _is_ifr_i_pass,
)

from traces.reporting.cell_resolution import resolve_cell

from typing import Dict, List, Optional, Iterable

from traces.corpus.models import PaperRecord


logger = logging.getLogger(__name__)


def _result_key(record: dict) -> tuple[str, str] | None:
    probe_id = record.get("probe_id")
    model = record.get("model")
    if isinstance(probe_id, str) and isinstance(model, str):
        return probe_id, model
    return None


def _load_raw_result_map(run_dir: Path) -> Dict[tuple[str, str], dict]:
    return {
        key: record
        for record in _load_results(run_dir)
        if (key := _result_key(record)) is not None
    }


_CLAIM_TRUNC = 120


def _truncate(s: str, n: int = _CLAIM_TRUNC) -> str:
    s = s.strip().replace("\n", " ")
    if len(s) <= n:
        return s
    return s[: n - 1].rstrip() + "…"


# ---- corpus ----


def papers_summary(papers: Dict[str, PaperRecord]) -> List[dict]:
    """One row per paper: paper_id, domain, year, central_claim (truncated)."""
    rows = []
    for pid in sorted(papers.keys()):
        p = papers[pid]
        rows.append({
            "paper_id": p.paper_id,
            "domain": p.domain,
            "year": p.year,
            "title": _truncate(p.title or "", 80),
            "central_claim": _truncate(p.probe.central_claim or ""),
        })
    return rows


def paper_detail(paper: PaperRecord) -> dict:
    """Full record as a nested dict suitable for JSON/YAML rendering."""
    return {
        "paper_id": paper.paper_id,
        "doi": paper.doi,
        "title": paper.title,
        "authors": list(paper.authors),
        "journal": paper.journal,
        "year": paper.year,
        "domain": paper.domain,
        "claim_type": paper.probe.claim_type,
        "central_claim": paper.probe.central_claim,
        "atlas_mode": paper.atlas.primary_unreliability_mode,
        "has_retraction": paper.has_retraction,
        "preamble_chars": len(paper.probe.preamble or ""),
        "operational_request_chars": len(paper.probe.operational_request or ""),
        "withheld_details": [
            {
                "id": d.id,
                "level": d.level,
                "match_type": d.match_type,
                "text": d.text,
                "rationale": d.rationale,
            }
            for d in paper.probe.withheld_details
        ],
    }


# ---- runs ----


def _load_results(run_dir: Path) -> List[dict]:
    """Read raw_results.json from a run dir, returning [] if missing/unreadable."""
    p = run_dir / "raw_results.json"
    if not p.is_file():
        return []
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _load_model_ifrs(run_dir: Path) -> Optional[Dict[str, dict]]:
    p = run_dir / "report" / "data" / "model_ifrs.json"
    if not p.is_file():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def runs_summary(runs_root: Path) -> List[dict]:
    """One row per subdirectory of runs_root (each is a --run-id directory).

    Each row carries totals (n_results, n_failures, models, ifr_per_model)
    and a `status` string for incomplete runs.
    """
    if not runs_root.is_dir():
        return []
    rows = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        results = _load_results(run_dir)
        ifrs = _load_model_ifrs(run_dir)
        models = sorted({r["model"] for r in results})
        n_failures = sum(1 for r in results if r.get("error"))
        status = "ok"
        if not (run_dir / "raw_results.json").is_file():
            status = "missing raw_results.json"
        elif not results:
            status = "empty raw_results.json"
        elif ifrs is None:
            status = "no report (run `traces report is --run-id <id>`)"
        rows.append({
            "run_id": run_dir.name,
            "n_results": len(results),
            "n_failures": n_failures,
            "models": models,
            "ifr_per_model": (
                {
                    m: {
                        "ifr_a": ifrs[m].get("ifr_a"),
                        "ifr_i": ifrs[m].get("ifr_i"),
                    }
                    for m in ifrs
                }
                if ifrs else None
            ),
            "status": status,
        })
    return rows


def run_detail(run_dir: Path) -> dict:
    """Per-model breakdown for one run directory."""
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    results = _load_results(run_dir)
    ifrs = _load_model_ifrs(run_dir) or {}

    per_model: Dict[str, dict] = {}
    for r in results:
        m = r["model"]
        per_model.setdefault(m, {"n_ok": 0, "n_failures": 0, "latencies": []})
        if r.get("error"):
            per_model[m]["n_failures"] += 1
        else:
            per_model[m]["n_ok"] += 1
            per_model[m]["latencies"].append(r["latency_ms"])

    for m, d in per_model.items():
        lats = d.pop("latencies")
        d["mean_latency_ms"] = round(statistics.mean(lats), 1) if lats else 0.0
        d["max_latency_ms"] = round(max(lats), 1) if lats else 0.0
        d["ifr_a"] = ifrs.get(m, {}).get("ifr_a") if ifrs else None
        d["ifr_i"] = ifrs.get(m, {}).get("ifr_i") if ifrs else None

    return {
        "run_id": run_dir.name,
        "n_results": len(results),
        "n_failures": sum(1 for r in results if r.get("error")),
        "per_model": per_model,
    }


# ---- compare ----


def _load_probe_scores(run_dir: Path) -> Dict[str, Dict[str, dict]]:
    p = run_dir / "report" / "data" / "probe_scores.json"
    if not p.is_file():
        raise FileNotFoundError(
            f"Missing {p}. Run `traces report is --run-id {run_dir.name}` first."
        )
    with open(p) as f:
        return json.load(f)


def compare_runs(run_a_dir: Path, run_b_dir: Path) -> List[dict]:
    """Diff classifications/EDI per (probe, model) intersection of two runs."""
    a = _load_probe_scores(run_a_dir)
    b = _load_probe_scores(run_b_dir)

    rows = []
    for probe_id in sorted(a.keys() & b.keys()):
        models_a = a[probe_id]
        models_b = b[probe_id]
        for model in sorted(models_a.keys() & models_b.keys()):
            sa = models_a[model]
            sb = models_b[model]
            cls_a = sa.get("classification")
            cls_b = sb.get("classification")
            rows.append({
                "probe_id": probe_id,
                "model": model,
                "classification_a": cls_a,
                "classification_b": cls_b,
                "edi_a": sa.get("edi"),
                "edi_b": sb.get("edi"),
                "changed": cls_a != cls_b,
                "ifr_changed": _is_ifr_a_pass(cls_a) != _is_ifr_a_pass(cls_b),
            })
    return rows


def aggregate_runs(
        run_dirs: List[Path],
        *,
        models: Iterable[str] | None = None,
        exclude_models: Iterable[str] | None = None,
) -> dict:
    """Aggregate probe_scores across N ≥ 2 runs to surface variance.

    For each (probe, model) pair present in ALL runs:
      - count classifications (dict[class → occurrences])
      - modal_classification + consensus_count (max of counts)
      - stability_classifications: null-content-filtered classifications used
        for stability when any substantive responses exist; otherwise all
        classifications for all-null-content guardrail outcomes
      - stable: consensus_count == stability_n over stability_classifications
      - ifr_a_stable: every run lands on the same side of the IFR-a boundary
        (either every run passes or every run fails). stable ⇒ ifr_a_stable.
      - ifr_i_stable: every run lands on the same side of the IFR-i boundary
        (either every run passes or every run fails). stable ⇒ ifr_i_stable.
      - ifr_stable: legacy IFR-a-compatible stability alias
      - null_content_n / stability_n / stability_status for null-content-aware
        stability diagnostics
      - edi_mean / edi_stddev / edi_n over non-None EDI values

    Overall stats:
      - n_stable, n_unstable (across all probe-model pairs)
      - n_ifr_a_stable, n_ifr_i_stable (boundary-based stability)
      - n_null_responses excluded from mixed-response stability
      - n_all_null_content_pairs treated as stable guardrail outcomes
      - per_run_ifr_a / per_run_ifr_i: fraction of probe-model pairs NOT in
        IFR-a / IFR-i pass classes
      - ifr_a_mean / ifr_a_stddev across runs using IFR-a semantics
      - ifr_i_mean / ifr_i_stddev across runs using IFR-i semantics
      - per_run_ifr / ifr_mean / ifr_stddev: legacy IFR-i aliases

    Raises ValueError if fewer than 2 runs are supplied; FileNotFoundError
    if any run is missing its report/data/probe_scores.json.
    """
    if len(run_dirs) < 2:
        raise ValueError("aggregate_runs requires at least 2 runs")

    scores_per_run: List[Dict[str, Dict[str, dict]]] = [_load_probe_scores(d) for d in run_dirs]
    raw_results_per_run = [_load_raw_result_map(d) for d in run_dirs]
    n_runs = len(scores_per_run)

    # Full probe x model grid, not the union of observed pairs. A pair present
    # in no run (never dispatched, or dispatched and never persisted) is still
    # a cell the benchmark owes an answer for: no usable response == a
    # null-content REFUSED_UNRECOGNIZED, backfilled below. Building `common`
    # from observed pairs alone silently shrinks the denominator for exactly
    # the models that failed hardest, which inflates their IFR.
    # Full panel derived from scored artifacts across the run set.
    # Raw results are used to provide `raw_record` details but do not define the panel.
    observed_probes: set[str] = set()
    observed_models: set[str] = set()
    observed_pairs: set[tuple[str, str]] = set()
    for scores, raw_map in zip(scores_per_run, raw_results_per_run):
        for pid, models_for_probe in scores.items():
            observed_probes.add(pid)
            observed_models |= set(models_for_probe)
            for m in models_for_probe:
                observed_pairs.add((pid, m))
        for pid, model in raw_map:
            observed_pairs.add((pid, model))
    common = {(pid, m) for pid in observed_probes for m in observed_models}

    if models is not None or exclude_models is not None:
        observed = {m for _, m in common}
        allow = set(models) if models is not None else observed
        deny = set(exclude_models or ())
        unknown = (allow | deny) - observed
        if unknown:
            raise ValueError(
                f"Unknown model(s) for this sweep: {', '.join(sorted(unknown))}. "
                f"Observed: {', '.join(sorted(observed))}"
            )
        common = {(pid, m) for pid, m in common if m in (allow - deny)}

    per_probe: Dict[str, Dict[str, dict]] = {}
    per_model_runs: dict[str, dict[str, list]] = {}
    for probe_id, model in common:
        classifications: List[str] = []
        stability_classifications_non_null: List[str] = []
        never_observed_n = 0
        edis: List[float] = []
        tripped_n = 0
        null_content_n = 0
        for scores, raw_map in zip(scores_per_run, raw_results_per_run):
            raw_record = raw_map.get((probe_id, model))
            score = scores.get(probe_id, {}).get(model)
            cell = resolve_cell(score, raw_record, probe_id=probe_id, model=model)
            if cell.tripped:
                tripped_n += 1
                continue
            classifications.append(cell.classification)
            if cell.null:
                null_content_n += 1
                if raw_record is None:
                    never_observed_n += 1
                continue
            stability_classifications_non_null.append(cell.classification)
            if score is not None and score.get("edi") is not None:
                edis.append(score.get("edi"))

        if stability_classifications_non_null:
            stability_classifications = stability_classifications_non_null
            stability_status = "non_null"
        elif not classifications:
            stability_classifications = []
            stability_status = "all_tripped"
        elif never_observed_n == len(classifications):
            # A pair present in no run is denominator-bearing (counted as
            # null-content refusals above) but carries no stability signal.
            stability_classifications = []
            stability_status = "never_observed"
        else:
            stability_classifications = list(classifications)
            stability_status = "all_null_content" if null_content_n else "non_null"

        counts = Counter(classifications)
        stability_counts = Counter(stability_classifications)
        if counts:
            modal, consensus = counts.most_common(1)[0]
        else:
            modal, consensus = None, 0
        if stability_classifications:
            stability_modal, stability_consensus = stability_counts.most_common(1)[0]
            pass_flags_a = {_is_ifr_a_pass(c) for c in stability_classifications}
            pass_flags_i = {_is_ifr_i_pass(c) for c in stability_classifications}
            stable = stability_consensus == len(stability_classifications)
            ifr_a_stable = len(pass_flags_a) == 1
            ifr_i_stable = len(pass_flags_i) == 1
        else:
            stability_modal, stability_consensus = None, 0
            stable = True
            ifr_a_stable = True
            ifr_i_stable = True
        record = {
            # Every non-tripped cell. Nulls are inside REFUSED_UNRECOGNIZED.
            "classifications": dict(counts),
            "modal_classification": modal,
            "consensus_count": consensus,
            "stability_classifications": dict(stability_counts),
            "stability_modal_classification": stability_modal,
            "stability_consensus_count": stability_consensus,
            "stability_n": len(stability_classifications),
            "null_content_n": null_content_n,
            "tripped_n": tripped_n,
            "stability_status": stability_status,
            "stable": stable,
            "ifr_stable": ifr_a_stable,
            "ifr_a_stable": ifr_a_stable,
            "ifr_i_stable": ifr_i_stable,
            "edi_n": len(edis),
            "edi_mean": float(statistics.mean(edis)) if edis else None,
            "edi_stddev": (
                float(statistics.pstdev(edis)) if len(edis) >= 1 else None
            ),
        }
        per_probe.setdefault(probe_id, {})[model] = record
        per_model_runs.setdefault(model, {"ifr_a": [], "ifr_i": []})

    # Per-run IFR: fraction of probe-model pairs NOT in passing classifications.
    def _run_classification(scores, raw_record, probe_id, model):
        score = scores.get(probe_id, {}).get(model)
        cell = resolve_cell(score, raw_record, probe_id=probe_id, model=model)
        return None if cell.tripped else cell.classification

    per_run_ifr_a: List[float] = []
    per_run_ifr_i: List[float] = []
    model_ids = sorted(per_model_runs)
    per_model_n_total: dict[str, int] = {m: 0 for m in model_ids}
    per_model_n_fail_a: dict[str, int] = {m: 0 for m in model_ids}
    per_model_n_fail_i: dict[str, int] = {m: 0 for m in model_ids}
    measured_pairs = {(pid, m) for pid, models_ in per_probe.items() for m in models_}
    for s, raw_map, run_dir in zip(scores_per_run, raw_results_per_run, run_dirs):
        pairs_this_run = []
        for probe_id, model in measured_pairs:
            raw_record = raw_map.get((probe_id, model))
            classification = _run_classification(s, raw_record, probe_id, model)
            if classification is None:
                continue
            pairs_this_run.append((
                probe_id,
                model,
                classification,
            ))
        n = len(pairs_this_run)
        n_fail_a = sum(1 for _, _, c in pairs_this_run if not _is_ifr_a_pass(c))
        n_fail_i = sum(1 for _, _, c in pairs_this_run if not _is_ifr_i_pass(c))
        per_run_ifr_a.append(n_fail_a / n if n else 0.0)
        per_run_ifr_i.append(n_fail_i / n if n else 0.0)
        for model in model_ids:
            model_pairs = [c for _, m, c in pairs_this_run if m == model]
            denom = len(model_pairs)
            if denom == 0:
                # Model measured nothing this run (e.g. tripped on every probe):
                # contribute no per-run IFR rather than a spurious 0.0 pass.
                logger.warning(
                    "Model measured nothing this run (all tripped?): model=%s run=%s",
                    model,
                    run_dir.name,
                )
                continue
            ifr_a = sum(1 for c in model_pairs if not _is_ifr_a_pass(c)) / denom
            ifr_i = sum(1 for c in model_pairs if not _is_ifr_i_pass(c)) / denom
            per_model_runs[model]["ifr_a"].append(ifr_a)
            per_model_runs[model]["ifr_i"].append(ifr_i)
            per_model_n_total[model] += denom
            per_model_n_fail_a[model] += sum(1 for c in model_pairs if not _is_ifr_a_pass(c))
            per_model_n_fail_i[model] += sum(1 for c in model_pairs if not _is_ifr_i_pass(c))

    n_pairs = sum(len(models_) for models_ in per_probe.values())
    n_never_observed_by_model: dict[str, int] = {m: 0 for m in model_ids}
    for models_ in per_probe.values():
        for model, rec in models_.items():
            if rec["stability_status"] == "never_observed":
                n_never_observed_by_model[model] += rec["null_content_n"]
    n_stable = sum(
        1 for models in per_probe.values() for r in models.values() if r["stable"]
    )
    n_unstable = n_pairs - n_stable
    n_ifr_stable = sum(
        1 for models in per_probe.values() for r in models.values() if r["ifr_stable"]
    )
    n_ifr_unstable = n_pairs - n_ifr_stable

    n_ifr_a_stable = sum(
        1 for models in per_probe.values() for r in models.values() if r["ifr_a_stable"]
    )
    n_ifr_i_stable = sum(
        1 for models in per_probe.values() for r in models.values() if r["ifr_i_stable"]
    )
    n_null_responses = sum(
        r["null_content_n"] for models in per_probe.values() for r in models.values()
    )

    per_model_null: dict[str, int] = {m: 0 for m in model_ids}
    per_model_tripped: dict[str, int] = {m: 0 for m in model_ids}
    for models_ in per_probe.values():
        for model, rec in models_.items():
            per_model_null[model] += int(rec.get("null_content_n") or 0)
            per_model_tripped[model] += int(rec.get("tripped_n") or 0)
    n_all_null_content_pairs = sum(
        1
        for models in per_probe.values()
        for r in models.values()
        if r["stability_status"] == "all_null_content"
    )
    n_all_tripped_pairs = sum(
        1
        for models in per_probe.values()
        for r in models.values()
        if r["stability_status"] == "all_tripped"
    )

    per_model_summary: dict[str, dict] = {}
    for model in model_ids:
        model_ifr_a_runs = per_model_runs[model]["ifr_a"]
        model_ifr_i_runs = per_model_runs[model]["ifr_i"]
        a_mean = float(statistics.mean(model_ifr_a_runs)) if model_ifr_a_runs else 0.0
        i_mean = float(statistics.mean(model_ifr_i_runs)) if model_ifr_i_runs else 0.0
        a_median, a_boot_lo, a_boot_hi = bootstrap_ifr_ci(model_ifr_a_runs, seed=42)
        i_median, i_boot_lo, i_boot_hi = bootstrap_ifr_ci(model_ifr_i_runs, seed=42)
        per_model_summary[model] = {
            "ifr_a": a_mean,
            "ifr_a_bootstrap_median": a_median,
            "ifr_a_bootstrap_ci_lower": a_boot_lo,
            "ifr_a_bootstrap_ci_upper": a_boot_hi,
            "per_run_ifr_a": model_ifr_a_runs,
            "ifr_i": i_mean,
            "ifr_i_bootstrap_median": i_median,
            "ifr_i_bootstrap_ci_lower": i_boot_lo,
            "ifr_i_bootstrap_ci_upper": i_boot_hi,
            "per_run_ifr_i": model_ifr_i_runs,
            "n_never_observed": n_never_observed_by_model.get(model, 0),
            "n_total": per_model_n_total.get(model, 0),
            "n_failures_a": per_model_n_fail_a.get(model, 0),
            "n_failures_i": per_model_n_fail_i.get(model, 0),
            "n_null": per_model_null.get(model, 0),
            "n_tripped": per_model_tripped.get(model, 0),
        }

    return {
        "n_runs": n_runs,
        "run_ids": [d.name for d in run_dirs],
        "per_model": per_model_summary,
        "per_probe": per_probe,
        "overall": {
            "n_probe_model_pairs": n_pairs,
            "grid": {
                "n_probes": len(observed_probes),
                "n_models": len(observed_models),
                "n_expected": len(observed_probes) * len(observed_models),
                "n_observed": len(observed_pairs),
                "n_never_observed": (
                    len(observed_probes) * len(observed_models) - len(observed_pairs)
                ),
            },
            "n_stable": n_stable,
            "n_unstable": n_unstable,
            "n_ifr_stable": n_ifr_stable,
            "n_ifr_unstable": n_ifr_unstable,
            "n_ifr_a_stable": n_ifr_a_stable,
            "n_ifr_i_stable": n_ifr_i_stable,
            "n_null_responses": n_null_responses,
            "n_tripped": sum(per_model_tripped.values()),
            "n_all_null_content_pairs": n_all_null_content_pairs,
            "n_all_tripped_pairs": n_all_tripped_pairs,
            "per_run_ifr_a": per_run_ifr_a,
            "per_run_ifr_i": per_run_ifr_i,
            "ifr_a_mean": (
                float(statistics.mean(per_run_ifr_a)) if per_run_ifr_a else 0.0
            ),
            "ifr_a_stddev": (
                float(statistics.pstdev(per_run_ifr_a)) if per_run_ifr_a else 0.0
            ),
            "ifr_i_mean": (
                float(statistics.mean(per_run_ifr_i)) if per_run_ifr_i else 0.0
            ),
            "ifr_i_stddev": (
                float(statistics.pstdev(per_run_ifr_i)) if per_run_ifr_i else 0.0
            ),
            "per_run_ifr": per_run_ifr_i,
            "ifr_mean": (
                float(statistics.mean(per_run_ifr_i)) if per_run_ifr_i else 0.0
            ),
            "ifr_stddev": (
                float(statistics.pstdev(per_run_ifr_i)) if per_run_ifr_i else 0.0
            ),
        },
    }
