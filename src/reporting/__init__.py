"""Reporting module — episode report generation and HTML dashboard rendering."""

from .dashboard import DashboardRenderer
from .finding_descriptions import (
    enrich_report_data,
    get_finding_description,
    get_severity_info,
)
from .report import (
    EpisodeMetrics,
    EpisodeReport,
    FindingReport,
    ReportGenerator,
    SessionReport,
)

__all__ = [
    "DashboardRenderer",
    "EpisodeMetrics",
    "EpisodeReport",
    "FindingReport",
    "ReportGenerator",
    "SessionReport",
    "enrich_report_data",
    "get_finding_description",
    "get_severity_info",
]
