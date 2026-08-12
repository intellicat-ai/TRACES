"""Aggregate reporting for multi-run IFR / EDI summaries."""
from __future__ import annotations

import json
import logging
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from traces.reporting.cell_resolution import resolve_cell

from traces.config import ReportingConfig, ScoringConfig
from traces.corpus import PaperRecord
from traces.influence.edi import structural_edi_max
from traces.inspect import _is_ifr_a_pass, _is_ifr_i_pass, _load_results
from traces.reporting.common import METHODOLOGY_SECTION, render_corpus_paper_catalog

logger = logging.getLogger(__name__)

_ENGAGED_CLASSES = {"ENGAGED_RECOGNIZED", "ENGAGED_UNRECOGNIZED"}
_COLORBLIND_PALETTE = [
    "#66C2A5",
    "#FC8D62",
    "#8DA0CB",
    "#E78AC3",
    "#A6D854",
    "#FFD92F",
    "#E5C494",
    "#B3B3B3",
]


def generate_aggregate_report(
    *,
    agg: dict[str, Any],
    output_dir: Path,
    reporting_config: ReportingConfig,
    scoring_config: ScoringConfig,
    papers_by_id: dict[str, PaperRecord],
    run_dirs: list[Path],
    include_all: bool = False,
) -> str:
    report = AggregateReport(
        agg=agg,
        reporting_config=reporting_config,
        scoring_config=scoring_config,
        papers_by_id=papers_by_id,
        run_dirs=run_dirs,
        include_all=include_all,
    )
    return report.generate(output_dir)


class AggregateReport:
    def __init__(
        self,
        *,
        agg: dict[str, Any],
        reporting_config: ReportingConfig,
        scoring_config: ScoringConfig,
        papers_by_id: dict[str, PaperRecord],
        run_dirs: list[Path],
        include_all: bool,
    ) -> None:
        self.agg = agg
        self.reporting_config = reporting_config
        self.scoring_config = scoring_config
        self.papers_by_id = papers_by_id
        self.run_dirs = run_dirs
        self.include_all = include_all
        self._generated_plot_stems: set[str] = set()
        self._edi_summary = self._compute_aggregate_edi_summary()
        self._structural_summary = self._compute_structural_classification_summary()
        self._domain_ifr_summary = self._compute_domain_ifr_summary()

        self._assert_cell_decomposition_invariant()

    def _assert_cell_decomposition_invariant(self) -> None:
        models = list(self.agg.get("per_model", {}))
        n_probes = len(self.agg.get("per_probe", {}))
        n_runs = len(self.run_dirs)
        expected = n_probes * n_runs

        probe_ids = list(self.agg.get("per_probe", {}))

        for model in models:
            structural = self._structural_summary.get(model)
            if not structural:
                continue

            observed = (
                structural.get("tripped", 0)
                + structural.get("REFUSED_RECOGNIZED", 0)
                + structural.get("REFUSED_UNRECOGNIZED", 0)
                + structural.get("ENGAGED_RECOGNIZED", 0)
                + structural.get("ENGAGED_UNRECOGNIZED", 0)
            )
            if observed != expected:
                raise ValueError(
                    "Aggregate invariant violation for model="
                    f"{model}: observed={observed} decomposition={structural} expected={expected} "
                    f"(n_probes={n_probes} n_runs={n_runs})"
                )

            if structural["null"] > structural["REFUSED_UNRECOGNIZED"]:
                raise ValueError(
                    f"Null exceeds REFUSED_UNRECOGNIZED for model={model}: "
                    f"null={structural['null']} ru={structural['REFUSED_UNRECOGNIZED']}. "
                    "Every null cell must be REFUSED_UNRECOGNIZED."
                )

            per_model = self.agg.get("per_model", {}).get(model, {})
            headline_ifr_a = per_model.get("ifr_a")
            headline_ifr_i = per_model.get("ifr_i")
            if headline_ifr_a is None or headline_ifr_i is None:
                continue

            # Recompute per-run IFR from the shared resolver to match headline semantics
            # (`per_model[model]["ifr_*" ]` is the mean of per-run IFRs).
            per_run_ifr_a: list[float] = []
            per_run_ifr_i: list[float] = []
            denom_total = 0
            failures_total_a = 0
            failures_total_i = 0
            for run_dir in self.run_dirs:
                probe_scores = self._load_probe_scores(run_dir)
                raw_map = self._load_raw_result_map(run_dir)
                denom = 0
                tripped = 0
                failures_a = 0
                failures_i = 0
                for probe_id in probe_ids:
                    raw_record = raw_map.get((probe_id, model))
                    score = probe_scores.get(probe_id, {}).get(model)
                    cell = resolve_cell(score, raw_record, probe_id=probe_id, model=model)
                    if cell.tripped:
                        tripped += 1
                        continue
                    denom += 1
                    if not _is_ifr_a_pass(cell.classification):
                        failures_a += 1
                    if not _is_ifr_i_pass(cell.classification):
                        failures_i += 1
                if denom + tripped != n_probes:
                    raise ValueError(
                        f"Denominator drift for model={model} run={run_dir.name}: "
                        f"denom={denom} tripped={tripped} n_probes={n_probes}"
                    )
                if denom:
                    per_run_ifr_a.append(failures_a / denom)
                    per_run_ifr_i.append(failures_i / denom)
                denom_total += denom
                failures_total_a += failures_a
                failures_total_i += failures_i

            derived_ifr_a = float(statistics.mean(per_run_ifr_a)) if per_run_ifr_a else 0.0
            derived_ifr_i = float(statistics.mean(per_run_ifr_i)) if per_run_ifr_i else 0.0

            if not math.isclose(
                headline_ifr_a,
                derived_ifr_a,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"Aggregate IFR-a mismatch for model={model}: headline={headline_ifr_a} structural_derived={derived_ifr_a} "
                    f"(denom_total={denom_total} failures_total={failures_total_a})"
                )
            if not math.isclose(
                headline_ifr_i,
                derived_ifr_i,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"Aggregate IFR-i mismatch for model={model}: headline={headline_ifr_i} structural_derived={derived_ifr_i} "
                    f"(denom_total={denom_total} failures_total={failures_total_i})"
                )

            # Domain decomposition must exactly match non-tripped structural totals.
            domain_total = 0
            domain_fail_a = 0
            domain_fail_i = 0
            for domain in self._domain_ifr_summary.values():
                rec = domain.get(model)
                if not rec:
                    continue
                domain_total += int(rec.get("n_total", 0) or 0)
                domain_fail_a += int(rec.get("n_failures_a", 0) or 0)
                domain_fail_i += int(rec.get("n_failures_i", 0) or 0)

            if domain_total != denom_total:
                raise ValueError(
                    f"Aggregate domain n_total mismatch for model={model}: domain_sum={domain_total} non_tripped={denom_total}"
                )
            if domain_fail_a != failures_total_a:
                raise ValueError(
                    f"Aggregate domain failures_a mismatch for model={model}: domain_sum={domain_fail_a} structural_failures={failures_total_a}"
                )
            if domain_fail_i != failures_total_i:
                raise ValueError(
                    f"Aggregate domain failures_i mismatch for model={model}: domain_sum={domain_fail_i} structural_failures={failures_total_i}"
                )

    def generate(self, output_dir: Path) -> str:
        output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        data_dir = output_dir / "data"
        data_dir.mkdir(exist_ok=True)

        self._generate_plots(plots_dir)

        sections = [
            self._title_section(),
            self._paper_catalog_section(),
            self._headline_ifr_section(),
            self._domain_stratified_ifr_section(),
            self._response_structural_classification_section(),
            self._per_probe_refusal_section(),
            self._per_model_null_section(),
            self._stability_summary_section(),
            self._per_run_ifr_section(),
            self._aggregate_edi_section(),
            self._per_model_ifr_section(),
            self._probe_model_stability_section(),
            self._methodology_section(),
        ]
        markdown = "\n\n---\n\n".join(sections)
        report_path = output_dir / "report.md"
        report_path.write_text(markdown, encoding="utf-8")

        self._save_data(data_dir)
        logger.info("Aggregate report written to %s", report_path)
        return str(report_path)

    def _title_section(self) -> str:
        n_models = len(self.agg.get("per_model", {}))
        n_probes = len(self.agg.get("per_probe", {}))
        lines = [
            "# TRACES Aggregate Influence Score Report",
            "",
            f"Models tested: {n_models}  ",
            f"Probes per model: {n_probes}",
        ]
        return "\n".join(lines)

    def _paper_catalog_section(self) -> str:
        return render_corpus_paper_catalog(self.papers_by_id, self.scoring_config)

    def _headline_ifr_section(self) -> str:
        overall = self.agg["overall"]
        return "\n".join([
            *[
                "## Headline: Aggregate Influence Failure Rates",
                "",
                "| Metric | Mean | Stddev | Per-run values |",
                "|---|---:|---:|---|",
                (
                    f"| IFR-a | {overall['ifr_a_mean']:.3f} | {overall['ifr_a_stddev']:.3f} | "
                    f"{', '.join(f'{value:.3f}' for value in overall['per_run_ifr_a'])} |"
                ),
                (
                    f"| IFR-i | {overall['ifr_i_mean']:.3f} | {overall['ifr_i_stddev']:.3f} | "
                    f"{', '.join(f'{value:.3f}' for value in overall['per_run_ifr_i'])} |"
                ),
                "",
            ],
            *self._null_adjusted_lines(),
            *self._plot_lines("aggregate_ifr_a_by_model", "aggregate_ifr_i_by_model"),
        ])

    def _null_adjusted_totals(self) -> dict[str, float]:
        """Engagement pooled over cells that actually returned content.

        Headline IFR-a scores a null completion as a refusal, so a model that
        never answers reads the same as a model that always declines. This view
        drops null cells from the denominator and reports the share of
        content-bearing responses that engaged. Engaged cells are never null, so
        the numerator is unchanged. Tripped cells sit outside every structural
        bucket and never enter here.
        """
        dispatched = 0
        engaged = 0
        nulls = 0
        for summary in self._structural_summary.values():
            dispatched += (
                    summary["REFUSED_RECOGNIZED"]
                    + summary["REFUSED_UNRECOGNIZED"]
                    + summary["ENGAGED_RECOGNIZED"]
                    + summary["ENGAGED_UNRECOGNIZED"]
            )
            engaged += summary["ENGAGED_RECOGNIZED"] + summary["ENGAGED_UNRECOGNIZED"]
            nulls += summary["null"]
        non_null = dispatched - nulls
        return {
            "n_dispatched": dispatched,
            "n_null": nulls,
            "n_non_null": non_null,
            "n_engaged": engaged,
            "ifr_a_pooled": (engaged / dispatched) if dispatched else 0.0,
            "ifr_a_null_adjusted": (engaged / non_null) if non_null else 0.0,
            "null_fraction": (nulls / dispatched) if dispatched else 0.0,
        }

    def _null_adjusted_lines(self) -> list[str]:
        totals = self._null_adjusted_totals()
        return [
            "### Null-Adjusted Engagement",
            "",
            "| Metric | Value | Basis |",
            "|---|---:|---|",
            (
                f"| IFR-a, nulls scored as refusals | {totals['ifr_a_pooled']:.3f} | "
                f"{int(totals['n_engaged'])}/{int(totals['n_dispatched'])} |"
            ),
            (
                f"| IFR-a, nulls excluded from denominator | {totals['ifr_a_null_adjusted']:.3f} | "
                f"{int(totals['n_engaged'])}/{int(totals['n_non_null'])} |"
            ),
            (
                f"| Null share of dispatched cells | {totals['null_fraction']:.3f} | "
                f"{int(totals['n_null'])}/{int(totals['n_dispatched'])} |"
            ),
            "",
            "Null-adjusted IFR-a is the fraction of content-bearing responses that "
            "engaged with the corrupted claim. Both rows here are pooled over every "
            "model x probe x run cell, so they differ slightly from the headline "
            "table above, which is the mean of per-run rates.",
            "",
        ]

    def _domain_stratified_ifr_section(self) -> str:
        lines = ["## Domain-Stratified IFR", "", "### IFR-a", ""]
        domains = sorted(self._domain_ifr_summary)
        if not domains:
            lines.append("No domain-stratified IFR data available.")
            return "\n".join(lines)

        header = "| Model | " + " | ".join(domains) + " |"
        sep = "|---|" + "|".join(["---:"] * len(domains)) + "|"
        lines.extend([header, sep])
        for model in sorted(self.agg.get("per_model", {})):
            cells = [self._format_domain_ifr_cell(model, domain, "ifr_a") for domain in domains]
            lines.append(f"| {model} | " + " | ".join(cells) + " |")

        lines.extend(["", "### IFR-i", "", header, sep])
        for model in sorted(self.agg.get("per_model", {})):
            cells = [self._format_domain_ifr_cell(model, domain, "ifr_i") for domain in domains]
            lines.append(f"| {model} | " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def _response_structural_classification_section(self) -> str:
        lines = [
            "## Response Structural Classification",
            "",
            "| Model | RR | RU | of which null | ER | EU | Tripped |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for model in sorted(self._structural_summary):
            summary = self._structural_summary[model]
            lines.append(
                f"| {model} | {summary['REFUSED_RECOGNIZED']} | {summary['REFUSED_UNRECOGNIZED']} "
                f"| {summary['null']} | {summary['ENGAGED_RECOGNIZED']} | {summary['ENGAGED_UNRECOGNIZED']} "
                f"| {summary['tripped']} |"
            )
        lines.extend([
            "",
            "Note: `of which null` is a subset of `RU` and is not additive.",
        ])
        return "\n".join(lines)


    def _per_probe_refusal_section(self) -> str:
        """Per-probe refusal totals across the full model×run panel.

        Nulls are REFUSED_UNRECOGNIZED, so they sit in both the numerator and the
        denominator. Trips were never dispatched and sit in neither.
        """

        n_runs = int(self.agg.get("n_runs") or 0)
        per_probe = self.agg.get("per_probe") or {}
        domains = self._probe_domain_map()

        lines = [
            "## Per-probe Refusal Rates",
            "",
            "| Probe | Refused/Total | Rate | Nulls | Tripped | Domain |",
            "|---|---:|---:|---:|---:|---|",
        ]

        rows: list[tuple[float, str, str]] = []
        for probe_id, by_model in per_probe.items():
            probe_label = probe_id.removeprefix("IS-")
            domain = domains.get(probe_id) or domains.get(probe_label) or ""
            total_content = 0
            refused = 0
            nulls = 0
            tripped = 0
            for cell in (by_model or {}).values():
                counts = cell.get("classifications") or {}
                total_content += sum(int(v) for v in counts.values())
                refused += int(counts.get("REFUSED_RECOGNIZED", 0)) + int(
                    counts.get("REFUSED_UNRECOGNIZED", 0)
                )
                nulls += int(cell.get("null_content_n") or 0)
                tripped += int(cell.get("tripped_n") or 0)
            attempts = total_content

            if n_runs and (attempts + tripped) != (len(by_model) * n_runs):
                logger.warning(
                    "Per-probe refusal parts do not reconcile for probe=%s: content=%d nulls=%d tripped=%d n_models=%d n_runs=%d",
                    probe_id,
                    total_content,
                    nulls,
                    tripped,
                    len(by_model),
                    n_runs,
                )

            rate = (refused / attempts) if attempts else 0.0
            rows.append(
                (
                    rate,
                    probe_label,
                    f"| {probe_label} | {refused}/{attempts} | {rate:.3f} | {nulls} | {tripped} | {domain} |",
                )
            )

        for _, _, row in sorted(rows, key=lambda item: (-item[0], item[1])):
            lines.append(row)

        return "\n".join(lines)

    def _format_domain_ifr_cell(self, model: str, domain: str, metric: str) -> str:
        summary = self._domain_ifr_summary.get(domain, {}).get(model)
        if not summary or summary["n_total"] <= 0:
            return "—"
        if metric == "ifr_a":
            failures = summary["n_failures_a"]
            value = failures / summary["n_total"]
        else:
            failures = summary["n_failures_i"]
            value = failures / summary["n_total"]
        return f"{value:.3f} ({failures}/{summary['n_total']})"

    def _stability_summary_section(self) -> str:
        overall = self.agg["overall"]
        total = overall["n_probe_model_pairs"]
        stable = overall["n_stable"]
        unstable = overall["n_unstable"]
        ifr_a_stable = overall["n_ifr_a_stable"]
        ifr_i_stable = overall["n_ifr_i_stable"]
        all_null = overall.get("n_all_null_content_pairs", 0)
        null_responses = overall.get("n_null_responses", 0)
        null_total = sum(s.get("null", 0) for s in self._structural_summary.values())
        tripped_total = sum(s.get("tripped", 0) for s in self._structural_summary.values())
        return "\n".join([
            *[
                "## Stability Summary",
                "",
                "| Metric | Count | Percent |",
                "|---|---:|---:|",
                f"| Stable, null-content-aware classification | {stable} / {total} | {self._pct(stable, total):.1f}% |",
                f"| Unstable, substantive disagreement | {unstable} / {total} | {self._pct(unstable, total):.1f}% |",
                f"| IFR-a stable | {ifr_a_stable} / {total} | {self._pct(ifr_a_stable, total):.1f}% |",
                f"| IFR-i stable | {ifr_i_stable} / {total} | {self._pct(ifr_i_stable, total):.1f}% |",
                f"| Null responses | {null_total} | — |",
                f"| Tripped (excluded from IFR denominators) | {tripped_total} | — |",
                f"| All-null-content probe×model pairs | {all_null} / {total} | {self._pct(all_null, total):.1f}% |",
                "",
                "Null cells are scored as REFUSED_UNRECOGNIZED and stay in every IFR denominator. They are ignored only for mixed-response stability. TRIPPED cells were never dispatched and are excluded from every IFR denominator.",
                "",
            ],
            *self._plot_lines("stability_summary"),
        ])

    def _per_run_ifr_section(self) -> str:
        overall = self.agg["overall"]
        run_ids = self.agg.get("run_ids", [])
        rows = [
            "## Per-Run IFR",
            "",
            "| Run ID | IFR-a | IFR-i |",
            "|---|---:|---:|",
        ]
        for idx, run_id in enumerate(run_ids):
            rows.append(
                f"| {run_id} | {overall['per_run_ifr_a'][idx]:.3f} | {overall['per_run_ifr_i'][idx]:.3f} |"
            )
        return "\n".join([
            *rows,
            "",
            *self._plot_lines("per_run_ifr_distribution"),
        ])

    def _aggregate_edi_section(self) -> str:
        lines = [
            "## Aggregate Engagement Depth",
            "",
            "| Model | Engaged/run mean | Mean EDI ± SD | Median EDI ± SD | Achievement ± SD |",
            "|---|---:|---:|---:|---:|",
        ]
        overall = self._edi_summary.get("overall")
        if overall is not None:
            lines.append(
                f"| overall | {overall['engaged_run_mean']:.1f} | {overall['mean_edi_mean']:.3f} ± {overall['mean_edi_stddev']:.3f} | {overall['median_edi_mean']:.3f} ± {overall['median_edi_stddev']:.3f} | {overall['achievement_mean']:.3f} ± {overall['achievement_stddev']:.3f} |"
            )
        for model in sorted(self._edi_summary.get("per_model", {})):
            summary = self._edi_summary["per_model"][model]
            lines.append(
                f"| {model} | {summary['engaged_run_mean']:.1f} | {summary['mean_edi_mean']:.3f} ± {summary['mean_edi_stddev']:.3f} | {summary['median_edi_mean']:.3f} ± {summary['median_edi_stddev']:.3f} | {summary['achievement_mean']:.3f} ± {summary['achievement_stddev']:.3f} |"
            )
        return "\n".join([
            *lines,
            "",
            *self._plot_lines("aggregate_edi_by_model"),
        ])

    def _per_model_ifr_section(self) -> str:
        lines = [
            "## Per-Model IFR",
            "",
            "| Model | IFR-a | IFR-a bootstrap median | IFR-a CI | IFR-i | IFR-i bootstrap median | IFR-i CI |",
            "|---|---:|---:|---|---:|---:|---|",
        ]
        for model in sorted(self.agg.get("per_model", {})):
            summary = self.agg["per_model"][model]
            lines.append(
                f"| {model} | {summary['ifr_a']:.3f} | {summary['ifr_a_bootstrap_median']:.3f} | [{summary['ifr_a_bootstrap_ci_lower']:.3f}, {summary['ifr_a_bootstrap_ci_upper']:.3f}] | {summary['ifr_i']:.3f} | {summary['ifr_i_bootstrap_median']:.3f} | [{summary['ifr_i_bootstrap_ci_lower']:.3f}, {summary['ifr_i_bootstrap_ci_upper']:.3f}] |"
            )
        return "\n".join(lines)

    def _probe_model_stability_section(self) -> str:
        rows = self._stability_rows()
        lines = [
            "## Probe/Model Stability Details",
            "",
            "| Probe | Model | Modal | Consensus | Stability N | Null-content | Tripped | IFR-a stable | IFR-i stable | EDI mean±SD | Top distribution |",
            "|---|---|---|---:|---:|---:|---:|---|---|---|---|",
        ]
        if not rows:
            lines.append("| — | — | — | — | — | — | — | — | — | — |")
            return "\n".join(lines)
        for row in rows:
            lines.append(
                f"| {row['probe_id']} | {row['model']} | {row['modal']} | {row['consensus']} | {row['stability_n']} | {row['null_content']} | {row['tripped']} | {row['ifr_a_stable']} | {row['ifr_i_stable']} | {row['edi']} | {row['distribution']} |"
            )
        return "\n".join(lines)

    def _per_model_null_section(self) -> str:
        """Null/empty completions per model, the per-model mirror of the
        per-probe refusal table.

        Nulls are a subset of REFUSED_UNRECOGNIZED, so they sit inside both the
        refusal count and the IFR denominators. Tripped cells were never
        dispatched and are outside both.
        """
        lines = [
            "## Per-model Null Responses",
            "",
            "| Model | Nulls | Dispatched | Null rate | Refusals | Null share of refusals | Tripped |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        rows: list[tuple[float, str, str]] = []
        for model, summary in self._structural_summary.items():
            dispatched = (
                    summary["REFUSED_RECOGNIZED"]
                    + summary["REFUSED_UNRECOGNIZED"]
                    + summary["ENGAGED_RECOGNIZED"]
                    + summary["ENGAGED_UNRECOGNIZED"]
            )
            refusals = summary["REFUSED_RECOGNIZED"] + summary["REFUSED_UNRECOGNIZED"]
            nulls = summary["null"]
            null_rate = (nulls / dispatched) if dispatched else 0.0
            null_share = (nulls / refusals) if refusals else 0.0
            rows.append(
                (
                    null_rate,
                    model,
                    f"| {model} | {nulls} | {dispatched} | {null_rate:.3f} | {refusals} "
                    f"| {null_share:.3f} | {summary['tripped']} |",
                )
            )

        for _, _, row in sorted(rows, key=lambda item: (-item[0], item[1])):
            lines.append(row)

        lines.extend([
            "",
            "Nulls are a subset of the refusal count and stay in every IFR denominator. "
            "`Dispatched` excludes circuit-breaker trips.",
        ])
        return "\n".join(lines)

    def _methodology_section(self) -> str:
        tripped_by_model = {
            model: summary.get("tripped", 0)
            for model, summary in self._structural_summary.items()
            if summary.get("tripped", 0) > 0
        }
        if not tripped_by_model:
            return METHODOLOGY_SECTION
        lines = [
            METHODOLOGY_SECTION.rstrip(),
            "",
            "### Circuit-breaker trips (excluded from IFR denominators)",
            "",
            "The runner may trip a circuit-breaker (`model_tripped`) before dispatch; these cells were never dispatched and are excluded from all IFR denominators, so the sweep is undercounted.",
            "",
            *[f"- `{model}`: {count}" for model, count in sorted(tripped_by_model.items())],
        ]
        return "\n".join(lines)

    def _save_data(self, data_dir: Path) -> None:
        payload = {
            "n_runs": self.agg["n_runs"],
            "run_ids": self.agg.get("run_ids", []),
            "overall": self.agg["overall"],
            "per_model": self.agg.get("per_model", {}),
            "per_probe": self.agg.get("per_probe", {}),
            "edi": self._edi_summary,
            "domain_ifr": self._domain_ifr_summary,
            "structural_classification": self._structural_summary,
            "null_adjusted": self._null_adjusted_totals(),
        }
        with open(data_dir / "aggregate.json", "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _generate_plots(self, plots_dir: Path) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available; skipping aggregate plots")
            return

        per_model = self.agg.get("per_model", {})
        models = sorted(per_model)
        if models:
            for metric, color, filename, title, metric_label in [
                ("ifr_a", "#c0392b", "aggregate_ifr_a_by_model", "Aggregated IFR-a by Model", "IFR-a"),
                ("ifr_i", "#2980b9", "aggregate_ifr_i_by_model", "Aggregated IFR-i by Model", "IFR-i"),
            ]:
                plot_models = sorted(models, key=lambda model: per_model[model][metric], reverse=True)
                values = [per_model[model][metric] for model in plot_models]
                fig_height = max(6.0, 0.35 * len(plot_models))
                fig, ax = plt.subplots(figsize=(10, fig_height))
                bars = ax.barh(plot_models, values, color=color)
                ax.invert_yaxis()
                ax.set_xlim(0, 1)
                ax.set_xlabel(f"Influence Failure Rate ({metric_label})")
                ax.grid(axis="x", alpha=0.25)
                ax.set_title(title)
                for bar, value in zip(bars, values):
                    ax.text(
                        min(value + 0.01, 0.98),
                        bar.get_y() + (bar.get_height() / 2),
                        f"{value:.3f}",
                        va="center",
                        fontsize=8,
                    )
                fig.tight_layout()
                fig.savefig(
                    plots_dir / f"{filename}.{self.reporting_config.plot_format}",
                    dpi=self.reporting_config.plot_dpi,
                )
                self._generated_plot_stems.add(filename)
                plt.close(fig)

        overall = self.agg["overall"]
        fig, ax = plt.subplots(figsize=(6, 5))
        boxplot_kwargs: dict[str, Any] = {
            "tick_labels": ["IFR-a", "IFR-i"],
            "patch_artist": True,
        }
        try:
            boxplot = ax.boxplot(
                [overall["per_run_ifr_a"], overall["per_run_ifr_i"]],
                **boxplot_kwargs,
            )
        except TypeError:
            boxplot_kwargs = {
                "labels": ["IFR-a", "IFR-i"],
                "patch_artist": True,
            }
            boxplot = ax.boxplot(
                [overall["per_run_ifr_a"], overall["per_run_ifr_i"]],
                **boxplot_kwargs,
            )
        for patch, color in zip(boxplot["boxes"], [_COLORBLIND_PALETTE[0], _COLORBLIND_PALETTE[1]]):
            patch.set_facecolor(color)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Influence Failure Rate")
        ax.set_title("Per-Run IFR Distribution")
        fig.tight_layout()
        fig.savefig(
            plots_dir / f"per_run_ifr_distribution.{self.reporting_config.plot_format}",
            dpi=self.reporting_config.plot_dpi,
        )
        self._generated_plot_stems.add("per_run_ifr_distribution")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        labels = [
            "Stable",
            "Unstable",
            "IFR-a stable",
            "IFR-i stable",
            "All-null-content",
        ]
        values = [
            overall["n_stable"],
            overall["n_unstable"],
            overall["n_ifr_a_stable"],
            overall["n_ifr_i_stable"],
            overall.get("n_all_null_content_pairs", 0),
        ]
        ax.bar(
            labels,
            values,
            color=[
                _COLORBLIND_PALETTE[0],
                _COLORBLIND_PALETTE[1],
                _COLORBLIND_PALETTE[2],
                _COLORBLIND_PALETTE[3],
                _COLORBLIND_PALETTE[7],
            ],
        )
        ax.set_ylabel("Probe×model pairs")
        ax.set_title("Aggregate Stability Summary")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        fig.tight_layout()
        fig.savefig(
            plots_dir / f"stability_summary.{self.reporting_config.plot_format}",
            dpi=self.reporting_config.plot_dpi,
        )
        self._generated_plot_stems.add("stability_summary")
        plt.close(fig)

        edi_models = sorted(self._edi_summary.get("per_model", {}))
        if edi_models:
            fig, ax = plt.subplots(figsize=(10, 6))
            means = [self._edi_summary["per_model"][model]["mean_edi_mean"] for model in edi_models]
            stddevs = [self._edi_summary["per_model"][model]["mean_edi_stddev"] for model in edi_models]
            ax.barh(edi_models, means, color=_COLORBLIND_PALETTE[2])
            for idx, (mean, stddev) in enumerate(zip(means, stddevs)):
                ax.hlines(idx, max(0.0, mean - stddev), min(1.0, mean + stddev), color=_COLORBLIND_PALETTE[7], linewidth=2)
                ax.plot(mean, idx, marker="o", color=_COLORBLIND_PALETTE[7])
            ax.set_xlim(0, 1)
            ax.set_xlabel("Mean EDI across runs")
            ax.set_title("Aggregate EDI by Model")
            fig.tight_layout()
            fig.savefig(
                plots_dir / f"aggregate_edi_by_model.{self.reporting_config.plot_format}",
                dpi=self.reporting_config.plot_dpi,
            )
            self._generated_plot_stems.add("aggregate_edi_by_model")
            plt.close(fig)

    def _stability_rows(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        n_runs = self.agg["n_runs"]
        for probe_id in sorted(self.agg.get("per_probe", {})):
            for model in sorted(self.agg["per_probe"][probe_id]):
                rec = self.agg["per_probe"][probe_id][model]
                interesting = (
                    not rec.get("stable", False)
                    or rec.get("null_content_n", 0) > 0
                )
                if not self.include_all and not interesting:
                    continue
                edi_mean = rec.get("edi_mean")
                edi_stddev = rec.get("edi_stddev")
                edi_str = "—" if edi_mean is None else f"{edi_mean:.2f}±{(edi_stddev or 0.0):.2f}"
                dist_pairs = sorted(
                    rec.get("classifications", {}).items(),
                    key=lambda item: (-item[1], item[0]),
                )
                dist = " ".join(f"{name}={count}" for name, count in dist_pairs[:2])
                rows.append({
                    "probe_id": probe_id,
                    "model": model,
                    "modal": rec["modal_classification"],
                    "consensus": f"{rec['consensus_count']}/{n_runs}",
                    "stability_n": f"{rec['stability_n']}/{n_runs}",
                    "null_content": str(rec["null_content_n"]),
                    "tripped": str(rec.get("tripped_n", 0)),
                    "ifr_a_stable": "yes" if rec["ifr_a_stable"] else "no",
                    "ifr_i_stable": "yes" if rec["ifr_i_stable"] else "no",
                    "edi": edi_str,
                    "distribution": dist,
                })
        return rows

    def _compute_structural_classification_summary(self) -> dict[str, dict[str, int]]:
        # Every model x probe x run cell lands in exactly one bucket, so the
        # per-model sum is always n_probes * n_runs. `tripped` is the only
        # bucket excluded from IFR denominators.
        models = list(self.agg.get("per_model", {}))
        probe_ids = list(self.agg.get("per_probe", {}))
        summary: dict[str, dict[str, int]] = {
            model: {
                "REFUSED_RECOGNIZED": 0,
                "REFUSED_UNRECOGNIZED": 0,
                "ENGAGED_RECOGNIZED": 0,
                "ENGAGED_UNRECOGNIZED": 0,
                "tripped": 0,
                # Subset of REFUSED_UNRECOGNIZED. Excluded from the bucket sum.
                "null": 0,
            }
            for model in models
        }
        for run_dir in self.run_dirs:
            probe_scores = self._load_probe_scores(run_dir)
            raw_map = self._load_raw_result_map(run_dir)
            for model in models:
                model_summary = summary[model]
                for probe_id in probe_ids:
                    raw_record = raw_map.get((probe_id, model))
                    score = probe_scores.get(probe_id, {}).get(model)
                    cell = resolve_cell(score, raw_record, probe_id=probe_id, model=model)
                    if cell.tripped:
                        model_summary["tripped"] += 1
                        continue
                    model_summary[cell.classification] += 1
                    if cell.null:
                        model_summary["null"] += 1
        return summary

    def _compute_domain_ifr_summary(self) -> dict[str, dict[str, dict[str, int]]]:
        # Full model x probe grid, matching aggregate_runs. Nulls are inside
        # REFUSED_UNRECOGNIZED and stay in the denominator; tripped leaves the
        # denominator.
        summary: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
        probe_domains = self._probe_domain_map()
        models = list(self.agg.get("per_model", {}))
        probe_ids = list(self.agg.get("per_probe", {}))
        for run_dir in self.run_dirs:
            probe_scores = self._load_probe_scores(run_dir)
            raw_map = self._load_raw_result_map(run_dir)
            for probe_id in probe_ids:
                domain = probe_domains.get(probe_id)
                if domain is None:
                    continue
                model_scores = probe_scores.get(probe_id, {})
                for model in models:
                    raw_record = raw_map.get((probe_id, model))
                    score = model_scores.get(model)
                    cell = resolve_cell(score, raw_record, probe_id=probe_id, model=model)
                    if cell.tripped:
                        continue
                    domain_model = summary[domain].setdefault(model, {
                        "n_total": 0,
                        "n_failures_a": 0,
                        "n_failures_i": 0,
                    })
                    domain_model["n_total"] += 1
                    if not _is_ifr_a_pass(cell.classification):
                        domain_model["n_failures_a"] += 1
                    if not _is_ifr_i_pass(cell.classification):
                        domain_model["n_failures_i"] += 1
        return {domain: models for domain, models in sorted(summary.items())}

    def _probe_domain_map(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for paper in self.papers_by_id.values():
            mapping[paper.paper_id] = paper.domain
            mapping[f"IS-{paper.paper_id}"] = paper.domain
        return mapping

    def _load_probe_scores(self, run_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
        score_path = run_dir / "report" / "data" / "probe_scores.json"
        if not score_path.exists():
            raise FileNotFoundError(f"Missing {score_path}")
        with open(score_path, encoding="utf-8") as handle:
            return json.load(handle)

    def _load_raw_result_map(self, run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
        mapping: dict[tuple[str, str], dict[str, Any]] = {}
        for record in _load_results(run_dir):
            probe_id = record.get("probe_id")
            model = record.get("model")
            if isinstance(probe_id, str) and isinstance(model, str):
                mapping[(probe_id, model)] = record
        return mapping


    def _compute_aggregate_edi_summary(self) -> dict[str, Any]:
        models = sorted(self.agg.get("per_model", {}))
        per_model_runs: dict[str, dict[str, list[float]]] = {
            model: {
                "engaged_counts": [],
                "mean_edi": [],
                "median_edi": [],
                "achievement": [],
            }
            for model in models
        }
        overall_per_run: list[dict[str, float]] = []
        ceilings = self._probe_ceiling_map()
        for run_dir in self.run_dirs:
            raw_map = self._load_raw_result_map(run_dir)
            score_path = run_dir / "report" / "data" / "probe_scores.json"
            if not score_path.exists():
                raise FileNotFoundError(f"Missing {score_path}")
            with open(score_path, encoding="utf-8") as handle:
                probe_scores = json.load(handle)

            allowed = set(self.agg.get("per_model", {}))

            run_overall_edis: list[float] = []
            run_overall_achievement_edis: list[float] = []
            run_overall_ceilings: list[float] = []
            run_overall_engaged = 0
            per_model_edis: dict[str, list[float]] = {}
            per_model_achievement_edis: dict[str, list[float]] = {}
            per_model_ceilings: dict[str, list[float]] = {}
            per_model_engaged: dict[str, int] = {}
            for probe_id, model_scores in probe_scores.items():
                ceiling = ceilings.get(probe_id)
                for model, score in model_scores.items():
                    if model not in allowed:
                        continue
                    cell = resolve_cell(
                        score,
                        raw_map.get((probe_id, model)),
                        probe_id=probe_id,
                        model=model,
                    )
                    if cell.tripped:
                        continue
                    if cell.classification in _ENGAGED_CLASSES:
                        per_model_engaged[model] = per_model_engaged.get(model, 0) + 1
                        run_overall_engaged += 1
                        edi_for_stats = float(score.get("edi") or 0.0)
                        per_model_edis.setdefault(model, []).append(edi_for_stats)
                        run_overall_edis.append(edi_for_stats)

                        edi = score.get("edi")
                        if edi is None or ceiling is None or ceiling <= 0:
                            continue
                        per_model_achievement_edis.setdefault(model, []).append(float(edi))
                        per_model_ceilings.setdefault(model, []).append(ceiling)
                        run_overall_achievement_edis.append(float(edi))
                        run_overall_ceilings.append(ceiling)

            for model in models:
                model_stats = per_model_runs[model]
                model_stats["engaged_counts"].append(float(per_model_engaged.get(model, 0)))
                edis = per_model_edis.get(model, [])
                achievement_edis = per_model_achievement_edis.get(model, [])
                ceiling_values = per_model_ceilings.get(model, [])
                model_stats["mean_edi"].append(statistics.mean(edis) if edis else 0.0)
                model_stats["median_edi"].append(statistics.median(edis) if edis else 0.0)
                model_stats["achievement"].append(
                    (sum(achievement_edis) / sum(ceiling_values)) if ceiling_values else 0.0
                )

            overall_per_run.append({
                "engaged_count": float(run_overall_engaged),
                "mean_edi": statistics.mean(run_overall_edis) if run_overall_edis else 0.0,
                "median_edi": statistics.median(run_overall_edis) if run_overall_edis else 0.0,
                "achievement": (
                    sum(run_overall_achievement_edis) / sum(run_overall_ceilings)
                    if run_overall_ceilings else 0.0
                ),
            })

        return {
            "overall": self._summarize_run_metric_dicts(overall_per_run),
            "per_model": {
                model: self._summarize_run_metric_lists(metrics)
                for model, metrics in per_model_runs.items()
            },
        }

    def _probe_ceiling_map(self) -> dict[str, float | None]:
        ceilings: dict[str, float | None] = {}
        for paper in self.papers_by_id.values():
            ceiling = structural_edi_max(
                list(paper.probe.withheld_details),
                self.scoring_config.edi,
            )
            ceilings[paper.paper_id] = ceiling
            ceilings[f"IS-{paper.paper_id}"] = ceiling
        return ceilings

    @staticmethod
    def _summarize_run_metric_lists(metrics: dict[str, list[float]]) -> dict[str, float]:
        return {
            "engaged_run_mean": statistics.mean(metrics["engaged_counts"]) if metrics["engaged_counts"] else 0.0,
            "mean_edi_mean": statistics.mean(metrics["mean_edi"]) if metrics["mean_edi"] else 0.0,
            "mean_edi_stddev": statistics.pstdev(metrics["mean_edi"]) if metrics["mean_edi"] else 0.0,
            "median_edi_mean": statistics.mean(metrics["median_edi"]) if metrics["median_edi"] else 0.0,
            "median_edi_stddev": statistics.pstdev(metrics["median_edi"]) if metrics["median_edi"] else 0.0,
            "achievement_mean": statistics.mean(metrics["achievement"]) if metrics["achievement"] else 0.0,
            "achievement_stddev": statistics.pstdev(metrics["achievement"]) if metrics["achievement"] else 0.0,
        }

    def _summarize_run_metric_dicts(self, rows: list[dict[str, float]]) -> dict[str, float] | None:
        if not rows:
            return None
        return {
            "engaged_run_mean": statistics.mean(row["engaged_count"] for row in rows),
            "mean_edi_mean": statistics.mean(row["mean_edi"] for row in rows),
            "mean_edi_stddev": statistics.pstdev(row["mean_edi"] for row in rows),
            "median_edi_mean": statistics.mean(row["median_edi"] for row in rows),
            "median_edi_stddev": statistics.pstdev(row["median_edi"] for row in rows),
            "achievement_mean": statistics.mean(row["achievement"] for row in rows),
            "achievement_stddev": statistics.pstdev(row["achievement"] for row in rows),
        }

    def _plot_markdown(self, stem: str) -> str:
        if stem not in self._generated_plot_stems:
            return ""
        return f"![{stem}](plots/{stem}.{self.reporting_config.plot_format})"

    def _plot_lines(self, *stems: str) -> list[str]:
        return [line for stem in stems if (line := self._plot_markdown(stem))]

    @staticmethod
    def _pct(part: int, whole: int) -> float:
        return (part / whole * 100.0) if whole else 0.0