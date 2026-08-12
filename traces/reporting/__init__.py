"""Reporting modules for TRACES benchmark.

Keep this package initializer import-light: some low-level modules (e.g.
`traces.inspect`) need to import tiny reporting utilities without pulling in the
full report generator, which itself imports `traces.inspect`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from traces.reporting.influence import InfluenceReport, ReportModule

if TYPE_CHECKING:
    from traces.reporting.aggregate import generate_aggregate_report as generate_aggregate_report


def __getattr__(name: str):
    if name == "generate_aggregate_report":
        from traces.reporting.aggregate import generate_aggregate_report as fn

        return fn
    raise AttributeError(name)


__all__ = ["InfluenceReport", "ReportModule", "generate_aggregate_report"]
