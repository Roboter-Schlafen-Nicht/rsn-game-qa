"""HTML dashboard renderer — converts session reports to visual dashboards.

Uses Jinja2 templating to produce a self-contained HTML file with
executive summary, enriched findings with plain-English descriptions,
severity explanations, visual evidence, actionable recommendations,
and optional trained-vs-random comparison.

Falls back to a built-in minimal template when the configured template
file does not exist on disk.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .finding_descriptions import enrich_report_data

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    _JINJA2_AVAILABLE = True
except ImportError:
    _JINJA2_AVAILABLE = False

if TYPE_CHECKING:
    import jinja2

logger = logging.getLogger(__name__)


# Allowed image extensions and their MIME types for screenshot embedding.
_ALLOWED_IMAGE_EXTENSIONS: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

# Maximum screenshot file size (10 MB).  Files larger than this are
# skipped to prevent memory exhaustion when base64-encoding.
_MAX_SCREENSHOT_BYTES: int = 10 * 1024 * 1024


def _embed_screenshots(report_data: dict[str, Any]) -> dict[str, Any]:
    """Convert screenshot file paths to base64 data URIs for inline embedding.

    Modifies the report data in-place. Screenshots that cannot be read,
    have disallowed extensions, or exceed the 10 MB size limit are
    silently skipped (data URI set to ``None``).

    Parameters
    ----------
    report_data : dict[str, Any]
        The session report dict. Each finding may have a
        ``screenshot_path`` field.

    Returns
    -------
    dict[str, Any]
        The report data with ``screenshot_data_uri`` added to findings
        that have valid screenshot paths.
    """
    for episode in report_data.get("episodes", []):
        for finding in episode.get("findings", []):
            path_str = finding.get("screenshot_path")
            if not path_str:
                finding["screenshot_data_uri"] = None
                continue

            path = Path(path_str)

            # Validate extension against allowlist.
            suffix = path.suffix.lower()
            if suffix not in _ALLOWED_IMAGE_EXTENSIONS:
                logger.warning(
                    "Screenshot skipped (disallowed extension %s): %s",
                    suffix,
                    path,
                )
                finding["screenshot_data_uri"] = None
                continue

            if path.is_file():
                try:
                    file_size = path.stat().st_size
                    if file_size > _MAX_SCREENSHOT_BYTES:
                        logger.warning(
                            "Screenshot skipped (%.1f MB exceeds %d MB limit): %s",
                            file_size / (1024 * 1024),
                            _MAX_SCREENSHOT_BYTES // (1024 * 1024),
                            path,
                        )
                        finding["screenshot_data_uri"] = None
                        continue

                    img_bytes = path.read_bytes()
                    b64 = base64.b64encode(img_bytes).decode("ascii")
                    mime = _ALLOWED_IMAGE_EXTENSIONS[suffix]
                    finding["screenshot_data_uri"] = f"data:{mime};base64,{b64}"
                except OSError:
                    logger.warning("Could not read screenshot: %s", path)
                    finding["screenshot_data_uri"] = None
            else:
                finding["screenshot_data_uri"] = None

    return report_data


def _prepare_comparison(
    report_data: dict[str, Any],
    baseline_data: dict[str, Any] | None,
) -> dict[str, Any]:
    """Generate comparison narrative between trained and baseline reports.

    Parameters
    ----------
    report_data : dict[str, Any]
        The primary (trained agent) report.
    baseline_data : dict[str, Any] | None
        The baseline (random agent) report, or ``None`` if no comparison.

    Returns
    -------
    dict[str, Any]
        The report data with ``comparison`` key added.
    """
    if baseline_data is None:
        report_data["comparison"] = None
        return report_data

    trained_summary = report_data.get("summary", {})
    baseline_summary = baseline_data.get("summary", {})

    trained_length = trained_summary.get("mean_episode_length", 0)
    baseline_length = baseline_summary.get("mean_episode_length", 0)

    trained_critical = trained_summary.get("critical_findings", 0)
    baseline_critical = baseline_summary.get("critical_findings", 0)

    trained_total = trained_summary.get("total_findings", 0)
    baseline_total = baseline_summary.get("total_findings", 0)

    trained_reward = trained_summary.get("mean_episode_reward", 0)
    baseline_reward = baseline_summary.get("mean_episode_reward", 0)

    # Calculate ratios safely
    length_ratio = trained_length / baseline_length if baseline_length > 0 else 0
    finding_ratio = trained_total / baseline_total if baseline_total > 0 else 0

    # Build narrative
    parts = []
    if length_ratio > 1:
        parts.append(
            f"The trained agent survived **{length_ratio:.1f}x longer** "
            f"than the random baseline ({trained_length:.0f} vs "
            f"{baseline_length:.0f} steps on average)."
        )
    elif length_ratio > 0:
        parts.append(
            f"The random baseline survived longer than the trained agent "
            f"({baseline_length:.0f} vs {trained_length:.0f} steps on "
            f"average, {1 / length_ratio:.1f}x ratio)."
        )
    else:
        parts.append(
            f"Average episode length: trained {trained_length:.0f} steps, "
            f"baseline {baseline_length:.0f} steps."
        )

    unique_trained = trained_critical - baseline_critical
    if unique_trained > 0:
        parts.append(
            f"The trained agent found **{unique_trained} critical "
            f"issue{'s' if unique_trained != 1 else ''}** that the "
            f"random baseline missed."
        )
    elif trained_critical > 0 and baseline_critical > 0:
        parts.append(
            f"Both agents found critical issues (trained: "
            f"{trained_critical}, baseline: {baseline_critical})."
        )
    elif trained_critical == 0 and baseline_critical == 0:
        parts.append("Neither agent found critical issues.")

    if trained_total > baseline_total:
        parts.append(
            f"Overall, the trained agent detected **{finding_ratio:.1f}x "
            f"more findings** ({trained_total} vs {baseline_total})."
        )

    report_data["comparison"] = {
        "baseline_summary": baseline_summary,
        "trained_summary": trained_summary,
        "length_ratio": length_ratio,
        "finding_ratio": finding_ratio,
        "narrative": " ".join(parts),
        "trained_length": trained_length,
        "baseline_length": baseline_length,
        "trained_critical": trained_critical,
        "baseline_critical": baseline_critical,
        "trained_total": trained_total,
        "baseline_total": baseline_total,
        "trained_reward": trained_reward,
        "baseline_reward": baseline_reward,
    }

    return report_data


def _prepare_grouped_findings(
    report_data: dict[str, Any],
) -> dict[str, Any]:
    """Group and deduplicate findings for compact template rendering.

    Creates ``grouped_findings`` on the report data: a list of finding
    groups sorted by severity (critical first), each containing a
    representative finding, occurrence count, and list of
    (episode_id, step) locations.

    Findings are grouped by ``(severity, oracle_key)`` where
    ``oracle_key`` is ``oracle_name:type`` or just ``oracle_name``.

    Parameters
    ----------
    report_data : dict[str, Any]
        The enriched report data dict (after ``enrich_report_data``
        and ``_embed_screenshots``).  Modified in-place.

    Returns
    -------
    dict[str, Any]
        The report data with ``grouped_findings`` added.
    """
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    groups: dict[str, dict[str, Any]] = {}

    for episode in report_data.get("episodes", []):
        ep_id = episode.get("episode_id", 0)
        for finding in episode.get("findings", []):
            oracle_name = finding.get("oracle_name", "unknown")
            data = finding.get("data", {})
            finding_type = data.get("type") if isinstance(data, dict) else None
            key = f"{oracle_name}:{finding_type}" if finding_type else oracle_name
            severity = finding.get("severity", "info")
            group_key = f"{severity}:{key}"

            if group_key not in groups:
                groups[group_key] = {
                    "key": key,
                    "severity": severity,
                    "finding": finding,  # representative finding
                    "count": 0,
                    "locations": [],
                    "screenshots": [],
                }

            groups[group_key]["count"] += 1
            groups[group_key]["locations"].append(
                {"episode_id": ep_id, "step": finding.get("step", 0)}
            )

            # Collect screenshots (only from findings that have them)
            screenshot = finding.get("screenshot_data_uri")
            if screenshot:
                groups[group_key]["screenshots"].append(
                    {
                        "data_uri": screenshot,
                        "episode_id": ep_id,
                        "step": finding.get("step", 0),
                    }
                )

    # Sort: critical first, then warning, then info; within same
    # severity, sort by count descending
    grouped = list(groups.values())
    grouped.sort(key=lambda g: (severity_order.get(g["severity"], 2), -g["count"]))

    report_data["grouped_findings"] = grouped
    return report_data


# ── RSN logo (inline SVG) ───────────────────────────────────────────

_RSN_LOGO_SVG = """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 280 40" width="280" height="40">
  <!-- RSN Robot icon (white, 36px) -->
  <g transform="translate(2,2) scale(0.3)">
    <rect x="34" y="18" width="52" height="36" rx="8" fill="#fff"/>
    <ellipse cx="50" cy="36" rx="6" ry="7" fill="#1f3c88"/>
    <ellipse cx="70" cy="36" rx="6" ry="7" fill="#1f3c88"/>
    <rect x="40" y="6" width="12" height="16" rx="6" fill="#fff"/>
    <rect x="68" y="6" width="12" height="16" rx="6" fill="#fff"/>
    <rect x="46" y="10" width="28" height="8" rx="4" fill="#fff"/>
    <rect x="50" y="54" width="20" height="8" rx="3" fill="#fff"/>
    <rect x="28" y="60" width="64" height="50" rx="8" fill="#fff"/>
    <rect x="8" y="64" width="24" height="16" rx="8" fill="#fff"/>
    <rect x="8" y="76" width="12" height="22" rx="6" fill="#fff"/>
    <rect x="88" y="64" width="24" height="16" rx="8" fill="#fff"/>
    <rect x="100" y="76" width="12" height="22" rx="6" fill="#fff"/>
    <circle cx="72" cy="76" r="5" fill="#1f3c88"/>
    <circle cx="48" cy="96" r="5" fill="#1f3c88"/>
    <circle cx="72" cy="96" r="4" fill="#1f3c88"/>
    <line x1="72" y1="76" x2="48" y2="96" stroke="#1f3c88" stroke-width="3" stroke-linecap="round"/>
    <line x1="48" y1="96" x2="72" y2="96" stroke="#1f3c88" stroke-width="3" stroke-linecap="round"/>
    <circle cx="60" cy="86" r="2.5" fill="#1f3c88"/>
  </g>
  <!-- Wordmark -->
  <text x="46" y="17" font-family="Inter, -apple-system, system-ui, sans-serif"
        font-size="13" font-weight="700" fill="#ffffff" letter-spacing="0.5">RSN</text>
  <text x="46" y="32" font-family="'IBM Plex Mono', Consolas, monospace"
        font-size="11" fill="#a3c8eb">Game QA</text>
</svg>"""


# ── Built-in professional template ─────────────────────────────────

_BUILTIN_TEMPLATE = (
    """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ report.game | default('Game') }} — QA Report | RSN Game QA</title>
<style>
  /* ── Reset & base (dark theme) ── */
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'IBM Plex Mono', 'Fira Code', 'JetBrains Mono', Consolas, Monaco, monospace;
    font-size: 15px;
    line-height: 1.8;
    color: #b8b8bf;
    background: #0c0c0e;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
  a { color: #5893d4; text-decoration: none; }
  a:hover { text-decoration: underline; color: #7baee0; }

  /* ── Layout ── */
  .page { max-width: 960px; margin: 0 auto; padding: 24px; }
  .header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 20px 24px; background: #1f3c88; color: #fff; border-radius: 8px 8px 0 0;
  }
  .header-logo { display: flex; align-items: center; gap: 16px; }
  .header-meta { text-align: right; font-size: 13px; color: #a3c8eb; }
  .header-meta strong { color: #fff; }
  .content { background: #131316; padding: 32px; border-radius: 0 0 8px 8px;
             border: 1px solid #2a2a31; border-top: none; }
  .footer {
    text-align: center; padding: 20px; margin-top: 24px;
    font-size: 12px; color: #62626d;
  }

  /* ── QW3: Sticky nav bar ── */
  .nav-bar {
    position: sticky; top: 0; z-index: 100;
    display: flex; gap: 0; background: #0f1f4a;
    border-bottom: none;
    padding: 0; margin: 0;
    overflow-x: auto;
  }
  .nav-bar a {
    padding: 10px 16px; font-size: 13px; font-weight: 600;
    color: #8e8e96; white-space: nowrap; border-bottom: 3px solid transparent;
    transition: color 0.15s, border-color 0.15s;
    font-family: Inter, -apple-system, system-ui, sans-serif;
  }
  .nav-bar a:hover { color: #eff0f2; text-decoration: none; }
  .nav-bar a.active { color: #fff; border-bottom-color: #5893d4; }

  /* ── Typography ── */
  h1 { font-family: Inter, -apple-system, system-ui, sans-serif;
       font-size: 24px; font-weight: 700; margin-bottom: 4px; color: #eff0f2; }
  h2 { font-family: Inter, -apple-system, system-ui, sans-serif;
       font-size: 20px; font-weight: 700; margin: 32px 0 16px; padding-bottom: 8px;
       border-bottom: 1px solid #2a2a31; color: #eff0f2; }
  h3 { font-family: Inter, -apple-system, system-ui, sans-serif;
       font-size: 16px; font-weight: 600; margin: 20px 0 8px; color: #b8b8bf; }
  p { margin-bottom: 12px; }

  /* ── Verdict banner ── */
  .verdict-banner {
    display: flex; align-items: center; gap: 16px;
    padding: 20px 24px; border-radius: 8px; margin-bottom: 24px;
  }
  .verdict-fail { background: #1c0a0a; border-left: 5px solid #dc2626; }
  .verdict-warn { background: #1c1507; border-left: 5px solid #d97706; }
  .verdict-pass { background: #052e16; border-left: 5px solid #16a34a; }
  .verdict-label {
    font-family: Inter, -apple-system, system-ui, sans-serif;
    font-size: 28px; font-weight: 800; letter-spacing: 1px;
  }
  .verdict-fail .verdict-label { color: #ef4444; }
  .verdict-warn .verdict-label { color: #f59e0b; }
  .verdict-pass .verdict-label { color: #22c55e; }
  .verdict-narrative { font-size: 15px; color: #b8b8bf; }
  .verdict-narrative strong { color: #eff0f2; }

  /* ── Stats cards ── */
  .stats-row {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px; margin-bottom: 24px;
  }
  .stat-card {
    text-align: center; padding: 16px 12px; border-radius: 6px;
    background: #1c1c21; border: 1px solid #2a2a31;
  }
  .stat-value { font-size: 28px; font-weight: 700;
                font-family: Inter, -apple-system, system-ui, sans-serif; }
  .stat-label { font-size: 12px; color: #62626d; text-transform: uppercase;
                letter-spacing: 0.5px; margin-top: 4px;
                font-family: Inter, -apple-system, system-ui, sans-serif; }
  .stat-critical .stat-value { color: #ef4444; }
  .stat-warning .stat-value { color: #f59e0b; }
  .stat-info .stat-value { color: #60a5fa; }
  .stat-neutral .stat-value { color: #5893d4; }
  .stat-good .stat-value { color: #22c55e; }

  /* ── Severity badges ── */
  .badge {
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    font-size: 12px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.5px;
    font-family: Inter, -apple-system, system-ui, sans-serif;
  }
  .badge-critical { background: #dc2626; color: #fff; }
  .badge-warning { background: #d97706; color: #fff; }
  .badge-info { background: #2563eb; color: #fff; }
  .badge-pass { background: #16a34a; color: #fff; }
  .badge-fail { background: #dc2626; color: #fff; }

  /* ── Tables ── */
  table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
  th { text-align: left; padding: 10px 12px; background: #1c1c21;
       border-bottom: 2px solid #2a2a31; font-size: 13px;
       text-transform: uppercase; color: #8e8e96; letter-spacing: 0.5px;
       font-family: Inter, -apple-system, system-ui, sans-serif; }
  td { padding: 10px 12px; border-bottom: 1px solid #2a2a31;
       font-size: 14px; vertical-align: top; color: #b8b8bf; }
  tr:hover { background: #1c1c21; }

  /* ── Finding cards ── */
  .finding-card {
    border: 1px solid #2a2a31; border-radius: 6px; padding: 16px;
    margin-bottom: 12px; background: #1c1c21;
  }
  .finding-card-critical {
    border-left: 6px solid #dc2626; background: #1c0a0a;
  }
  .finding-card-warning {
    border-left: 4px solid #d97706; background: #1c1507;
  }
  .finding-card-info {
    border-left: 4px solid #2563eb; background: #0a1628;
    padding: 12px 16px;
  }
  .finding-header {
    display: flex; align-items: center; gap: 10px; margin-bottom: 8px;
    flex-wrap: wrap;
  }
  .finding-title { font-weight: 600; font-size: 15px; color: #eff0f2;
                   font-family: Inter, -apple-system, system-ui, sans-serif; }
  .finding-meta { font-size: 12px; color: #62626d; }
  .finding-body { font-size: 14px; color: #b8b8bf; margin-bottom: 8px; }
  .finding-recommendation {
    font-size: 13px; color: #b8b8bf; background: #0c0c0e;
    padding: 10px 14px; border-radius: 6px; margin-top: 8px;
    border: 1px solid #2a2a31;
  }
  .finding-recommendation strong { color: #eff0f2; }
  .finding-screenshot {
    margin-top: 10px; border-radius: 6px; max-width: 100%;
    border: 1px solid #2a2a31; cursor: pointer;
  }
  .finding-screenshot:hover { box-shadow: 0 2px 12px rgba(0,0,0,0.4); }
  .finding-count-badge {
    display: inline-block; padding: 2px 8px; border-radius: 10px;
    font-size: 11px; font-weight: 700; background: #172d66; color: #5893d4;
    margin-left: 4px;
  }
  .finding-locations {
    font-size: 12px; color: #62626d; margin-top: 6px;
  }

  /* ── QW5: Severity filter buttons ── */
  .filter-bar {
    display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap;
    align-items: center;
  }
  .filter-bar span { font-size: 13px; color: #62626d; margin-right: 4px; }
  .filter-btn {
    padding: 6px 14px; border-radius: 20px; border: 2px solid;
    font-size: 13px; font-weight: 600; cursor: pointer;
    transition: opacity 0.15s, background 0.15s;
    background: transparent;
    font-family: Inter, -apple-system, system-ui, sans-serif;
  }
  .filter-btn.active { color: #fff; }
  .filter-btn:not(.active) { opacity: 0.45; }
  .filter-btn-critical { border-color: #dc2626; color: #ef4444; }
  .filter-btn-critical.active { background: #dc2626; color: #fff; }
  .filter-btn-warning { border-color: #d97706; color: #f59e0b; }
  .filter-btn-warning.active { background: #d97706; color: #fff; }
  .filter-btn-info { border-color: #2563eb; color: #60a5fa; }
  .filter-btn-info.active { background: #2563eb; color: #fff; }

  /* ── QW1: Info findings collapsible ── */
  .info-group-summary {
    cursor: pointer; padding: 14px 16px; border-radius: 6px;
    background: #0a1628; border: 1px solid #1e40af; border-left: 4px solid #2563eb;
    margin-bottom: 12px; font-size: 14px; color: #b8b8bf;
    list-style: none;
  }
  .info-group-summary::-webkit-details-marker { display: none; }
  .info-group-summary::before {
    content: '\\25B6'; display: inline-block; margin-right: 8px;
    font-size: 11px; transition: transform 0.2s;
  }
  details[open] > .info-group-summary::before { transform: rotate(90deg); }
  .info-group-summary strong { color: #eff0f2; }
  .info-group-summary .count { font-weight: 700; color: #60a5fa; }

  /* ── Comparison section ── */
  .comparison-table td:first-child { font-weight: 600; width: 40%; color: #eff0f2; }
  .comparison-narrative { font-size: 15px; color: #b8b8bf;
                          margin-bottom: 16px; line-height: 1.7; }
  .comparison-narrative strong { color: #eff0f2; }

  /* ── Recommendation priority ── */
  .rec-list { list-style: none; counter-reset: rec-counter; }
  .rec-item {
    counter-increment: rec-counter; padding: 14px 16px 14px 52px;
    margin-bottom: 8px; border-radius: 6px; background: #1c1c21;
    border: 1px solid #2a2a31; position: relative; font-size: 14px;
  }
  .rec-item::before {
    content: counter(rec-counter); position: absolute; left: 16px;
    top: 14px; width: 24px; height: 24px; border-radius: 50%;
    background: #5893d4; color: #fff; text-align: center;
    line-height: 24px; font-size: 12px; font-weight: 700;
    font-family: Inter, -apple-system, system-ui, sans-serif;
  }
  .rec-title { font-weight: 600; margin-bottom: 4px; color: #eff0f2;
               font-family: Inter, -apple-system, system-ui, sans-serif; }
  .rec-body { color: #b8b8bf; font-size: 13px; }
  .rec-count { font-size: 12px; color: #62626d; }

  /* ── Severity explanation ── */
  .severity-legend {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;
    margin-bottom: 24px;
  }
  .severity-legend-item {
    padding: 12px 16px; border-radius: 8px; font-size: 13px;
  }

  /* ── Back to top ── */
  .back-to-top {
    position: fixed; bottom: 24px; right: 24px; z-index: 99;
    width: 40px; height: 40px; border-radius: 50%;
    background: #1f3c88; color: #fff; border: none; cursor: pointer;
    font-size: 18px; display: none; align-items: center; justify-content: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    transition: opacity 0.2s, background 0.15s;
  }
  .back-to-top:hover { background: #5893d4; }
  .back-to-top.visible { display: flex; }

  /* ── QW6 + Print styles ── */
  @media print {
    body { background: #fff; color: #1c1c21; }
    .page { max-width: 100%; padding: 0; }
    .header { border-radius: 0; }
    .content { background: #fff; border: none; border-radius: 0; color: #1c1c21; }
    h1, h2, h3 { color: #0c0c0e; }
    .finding-card { break-inside: avoid; background: #fff; border-color: #dcdce0; }
    .finding-title { color: #0c0c0e; }
    .finding-body, .finding-recommendation, .rec-body,
    .verdict-narrative, .comparison-narrative { color: #3d3d46; }
    td { color: #1c1c21; border-color: #dcdce0; }
    th { background: #f8f8fa; color: #3d3d46; border-color: #dcdce0; }
    .stat-card { background: #f8f8fa; border-color: #dcdce0; }
    .rec-item { background: #f8f8fa; border-color: #dcdce0; }
    .verdict-banner { break-inside: avoid; }
    h2 { break-after: avoid; border-color: #dcdce0; }
    .nav-bar { display: none; }
    .back-to-top { display: none !important; }
    .filter-bar { display: none; }
    /* QW6: Hide info findings in print */
    .finding-group-info { display: none !important; }
    .info-print-note { display: block !important; }
  }
  .info-print-note { display: none; font-size: 13px; color: #62626d;
                     font-style: italic; margin: 8px 0 16px; }

  /* ── Responsive ── */
  @media (max-width: 768px) {
    .header { flex-direction: column; gap: 12px; text-align: center; }
    .header-meta { text-align: center; }
    .severity-legend { grid-template-columns: 1fr; }
    .page { padding: 12px; }
    .content { padding: 20px; }
    .nav-bar a { padding: 8px 12px; font-size: 12px; }
  }

  /* ── Screenshot modal ── */
  .modal-overlay {
    display: none; position: fixed; top: 0; left: 0; width: 100%;
    height: 100%; background: rgba(0,0,0,0.8); z-index: 1000;
    justify-content: center; align-items: center; cursor: pointer;
  }
  .modal-overlay.active { display: flex; }
  .modal-overlay img {
    max-width: 90%; max-height: 90%; border-radius: 8px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
  }
</style>
</head>
<body>
<div class="page">

  <!-- ════════════════════════════════════════════════════════════════ -->
  <!-- Header                                                         -->
  <!-- ════════════════════════════════════════════════════════════════ -->
  <div class="header">
    <div class="header-logo">
      """
    + _RSN_LOGO_SVG
    + """
    </div>
    <div class="header-meta">
      <strong>QA Report</strong><br>
      {{ report.game | default('Unknown Game') }}<br>
      {% if report.timestamp %}{% set ts = report.timestamp[:19] | replace('T', ' ') %}{{ ts }} UTC{% endif %}<br>
      <span style="font-size:11px">Report ID: {{ (report.session_id | default(''))[:8] }}</span>
    </div>
  </div>

  <!-- ════════════════════════════════════════════════════════════════ -->
  <!-- QW3: Sticky navigation bar                                     -->
  <!-- ════════════════════════════════════════════════════════════════ -->
  <nav class="nav-bar" id="navBar">
    <a href="#executive-summary">Summary</a>
    <a href="#recommendations">Recommendations</a>
    <a href="#episode-summary">Episodes</a>
    {% if report.comparison %}<a href="#comparison">Comparison</a>{% endif %}
    <a href="#findings">Findings</a>
    <a href="#severity-guide">Severity Guide</a>
  </nav>

  <div class="content">

  <!-- ════════════════════════════════════════════════════════════════ -->
  <!-- Executive Summary                                              -->
  <!-- ════════════════════════════════════════════════════════════════ -->
  {% if report.executive_summary %}
  <h2 id="executive-summary">Executive Summary</h2>

  <div class="verdict-banner {{ report.executive_summary.verdict_class }}">
    <div class="verdict-label">{{ report.executive_summary.verdict }}</div>
    <div class="verdict-narrative">{{ report.executive_summary.narrative | replace('**', '') }}</div>
  </div>

  <!-- Key metrics -->
  {% set s = report.summary %}
  <div class="stats-row">
    <div class="stat-card stat-neutral">
      <div class="stat-value">{{ s.total_episodes | default(0) }}</div>
      <div class="stat-label">Episodes</div>
    </div>
    <div class="stat-card stat-critical">
      <div class="stat-value">{{ s.critical_findings | default(0) }}</div>
      <div class="stat-label">Critical</div>
    </div>
    <div class="stat-card stat-warning">
      <div class="stat-value">{{ s.warning_findings | default(0) }}</div>
      <div class="stat-label">Warnings</div>
    </div>
    <div class="stat-card stat-info">
      <div class="stat-value">{{ s.info_findings | default(0) }}</div>
      <div class="stat-label">Info</div>
    </div>
    <div class="stat-card stat-neutral">
      <div class="stat-value">{{ "%.0f" | format(s.mean_episode_length | default(0)) }}</div>
      <div class="stat-label">Avg Steps</div>
    </div>
    <div class="stat-card stat-neutral">
      <div class="stat-value">{{ "%.1f" | format(s.mean_episode_reward | default(0)) }}</div>
      <div class="stat-label">Avg Reward</div>
    </div>
  </div>
  {% endif %}

  <!-- ════════════════════════════════════════════════════════════════ -->
  <!-- QW2: Recommendations (moved above Findings)                    -->
  <!-- ════════════════════════════════════════════════════════════════ -->
  {% if report.recommendations %}
  <h2 id="recommendations">Recommendations</h2>
  <p style="color: #8e8e96; font-size: 14px; margin-bottom: 16px;">
    Prioritized list of issues to address, grouped by type and sorted by severity.
  </p>
  <ol class="rec-list">
    {% for rec in report.recommendations %}
    <li class="rec-item">
      <div class="rec-title">
        <span class="badge badge-{{ rec.severity }}">{{ rec.severity }}</span>
        {{ rec.title }}
      </div>
      <div class="rec-body">{{ rec.recommendation }}</div>
      <div class="rec-count">Occurred {{ rec.count }} time{{ 's' if rec.count != 1 }} across episodes</div>
    </li>
    {% endfor %}
  </ol>
  {% endif %}

  <!-- ════════════════════════════════════════════════════════════════ -->
  <!-- QW2: Episode Summary Table (moved above Findings)              -->
  <!-- ════════════════════════════════════════════════════════════════ -->
  <h2 id="episode-summary">Episode Summary</h2>
  <table>
    <thead>
      <tr>
        <th>#</th><th>Status</th><th>Steps</th><th>Reward</th>
        <th>Findings</th><th>Avg FPS</th>
      </tr>
    </thead>
    <tbody>
    {% for ep in report.episodes %}
      {% set has_critical = ep.findings | selectattr("severity", "equalto", "critical") | list | length > 0 %}
      <tr>
        <td>{{ ep.episode_id }}</td>
        <td><span class="badge {{ 'badge-fail' if has_critical else 'badge-pass' }}">
          {{ "FAIL" if has_critical else "PASS" }}
        </span></td>
        <td>{{ ep.steps }}</td>
        <td>{{ "%.1f" | format(ep.total_reward) }}</td>
        <td>
          {% set crit = ep.findings | selectattr("severity", "equalto", "critical") | list | length %}
          {% set warn = ep.findings | selectattr("severity", "equalto", "warning") | list | length %}
          {% set inf = ep.findings | selectattr("severity", "equalto", "info") | list | length %}
          {% if crit %}<span class="badge badge-critical">{{ crit }} critical</span> {% endif %}
          {% if warn %}<span class="badge badge-warning">{{ warn }} warning{{ 's' if warn != 1 }}</span> {% endif %}
          {% if inf %}<span class="badge badge-info">{{ inf }} info</span> {% endif %}
          {% if not ep.findings %}&mdash;{% endif %}
        </td>
        <td>{{ "%.1f" | format(ep.metrics.mean_fps) if ep.metrics.mean_fps is not none else "&mdash;" }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>

  <!-- ════════════════════════════════════════════════════════════════ -->
  <!-- Trained vs Random Comparison                                   -->
  <!-- ════════════════════════════════════════════════════════════════ -->
  {% if report.comparison %}
  <h2 id="comparison">Trained Agent vs Random Baseline</h2>

  <p class="comparison-narrative">{{ report.comparison.narrative | replace('**', '') }}</p>

  <table class="comparison-table">
    <thead>
      <tr><th>Metric</th><th>Trained Agent</th><th>Random Baseline</th></tr>
    </thead>
    <tbody>
      <tr>
        <td>Average Episode Length</td>
        <td>{{ "%.0f" | format(report.comparison.trained_length) }} steps</td>
        <td>{{ "%.0f" | format(report.comparison.baseline_length) }} steps</td>
      </tr>
      <tr>
        <td>Average Reward</td>
        <td>{{ "%.1f" | format(report.comparison.trained_reward) }}</td>
        <td>{{ "%.1f" | format(report.comparison.baseline_reward) }}</td>
      </tr>
      <tr>
        <td>Critical Issues Found</td>
        <td>{{ report.comparison.trained_critical }}</td>
        <td>{{ report.comparison.baseline_critical }}</td>
      </tr>
      <tr>
        <td>Total Findings</td>
        <td>{{ report.comparison.trained_total }}</td>
        <td>{{ report.comparison.baseline_total }}</td>
      </tr>
    </tbody>
  </table>
  {% endif %}

  <!-- ════════════════════════════════════════════════════════════════ -->
  <!-- QW4+QW1+QW5: Findings (grouped, deduplicated, filterable)      -->
  <!-- ════════════════════════════════════════════════════════════════ -->
  <h2 id="findings">Findings</h2>

  {% if report.grouped_findings %}

  <!-- QW5: Filter buttons -->
  {% set crit_count = report.grouped_findings | selectattr("severity", "equalto", "critical") | list | length %}
  {% set warn_count = report.grouped_findings | selectattr("severity", "equalto", "warning") | list | length %}
  {% set info_count = report.grouped_findings | selectattr("severity", "equalto", "info") | list | length %}
  <div class="filter-bar">
    <span>Show:</span>
    <button class="filter-btn filter-btn-critical active" onclick="toggleFilter('critical', this)"
            type="button">Critical ({{ crit_count }})</button>
    <button class="filter-btn filter-btn-warning active" onclick="toggleFilter('warning', this)"
            type="button">Warning ({{ warn_count }})</button>
    <button class="filter-btn filter-btn-info" onclick="toggleFilter('info', this)"
            type="button">Info ({{ info_count }})</button>
  </div>

  <!-- QW6: Print note for hidden info findings -->
  {% set total_info_findings = report.grouped_findings | selectattr("severity", "equalto", "info") | sum(attribute="count") %}
  <p class="info-print-note">
    Info-level findings ({{ total_info_findings }}) omitted from print.
    See the HTML report for full details.
  </p>

  <!-- QW4: Grouped/deduplicated finding cards, sorted by severity -->
  {% for group in report.grouped_findings %}
    {% if group.severity == 'info' %}
    <!-- QW1: Info findings wrapped in collapsible group -->
    <div class="finding-group-info" data-severity="info"
         {% if loop.first or report.grouped_findings[loop.index0 - 1].severity != 'info' %}
         id="info-findings-start"
         {% endif %}>
      <details>
        <summary class="info-group-summary">
          <strong>{{ group.finding.human_title | default(group.finding.oracle_name) }}</strong>
          &mdash; <span class="count">{{ group.count }} occurrence{{ 's' if group.count != 1 }}</span>
          across episodes.
          <em>{{ group.finding.human_description | default(group.finding.description) | truncate(120) }}</em>
        </summary>
        <div class="finding-card finding-card-info" style="margin-top: 8px;">
          <div class="finding-header">
            <span class="badge badge-info">INFO</span>
            <span class="finding-title">{{ group.finding.human_title | default(group.finding.oracle_name) }}</span>
            <span class="finding-count-badge">{{ group.count }}&times;</span>
          </div>
          <div class="finding-body">
            {{ group.finding.human_description | default(group.finding.description) }}
          </div>
          {% if group.finding.human_recommendation %}
          <div class="finding-recommendation">
            <strong>Recommendation:</strong> {{ group.finding.human_recommendation }}
          </div>
          {% endif %}
          <div class="finding-locations">
            Detected in episodes:
            {% set ep_ids = group.locations | map(attribute='episode_id') | unique | list %}
            {{ ep_ids | join(', ') }}
          </div>
        </div>
      </details>
    </div>
    {% else %}
    <!-- Critical/Warning finding card (always visible) -->
    <div class="finding-group-{{ group.severity }}" data-severity="{{ group.severity }}">
      <div class="finding-card finding-card-{{ group.severity }}">
        <div class="finding-header">
          <span class="badge badge-{{ group.severity }}">{{ group.finding.severity_info.label | default(group.severity | upper) }}</span>
          <span class="finding-title">{{ group.finding.human_title | default(group.finding.oracle_name) }}</span>
          {% if group.count > 1 %}
          <span class="finding-count-badge">{{ group.count }}&times;</span>
          {% endif %}
        </div>
        <div class="finding-body">
          {{ group.finding.human_description | default(group.finding.description) }}
        </div>
        {% for ss in group.screenshots %}
        <div style="margin-top: 8px;">
          <img class="finding-screenshot" src="{{ ss.data_uri }}"
               alt="Screenshot at Episode {{ ss.episode_id }}, Step {{ ss.step }}"
               onclick="openModal(this.src)"
               title="Episode {{ ss.episode_id }}, Step {{ ss.step }} — click to enlarge">
          <div class="finding-meta">Episode {{ ss.episode_id }} &middot; Step {{ ss.step }}</div>
        </div>
        {% endfor %}
        {% if group.finding.human_recommendation %}
        <div class="finding-recommendation">
          <strong>Recommendation:</strong> {{ group.finding.human_recommendation }}
        </div>
        {% endif %}
        {% if group.count > 1 %}
        <div class="finding-locations">
          Occurred {{ group.count }} time{{ 's' if group.count != 1 }}:
          {% for loc in group.locations[:20] %}
            Ep {{ loc.episode_id }} step {{ loc.step }}{{ ', ' if not loop.last }}
          {% endfor %}
          {% if group.locations | length > 20 %}
            and {{ group.locations | length - 20 }} more&hellip;
          {% endif %}
        </div>
        {% else %}
        <div class="finding-meta" style="margin-top: 6px;">
          Episode {{ group.locations[0].episode_id }} &middot; Step {{ group.locations[0].step }}
        </div>
        {% endif %}
      </div>
    </div>
    {% endif %}
  {% endfor %}

  {% else %}
  <p style="color: #22c55e; font-weight: 600;">No issues were detected during testing.</p>
  {% endif %}

  <!-- ════════════════════════════════════════════════════════════════ -->
  <!-- Severity Guide (moved to appendix position)                    -->
  <!-- ════════════════════════════════════════════════════════════════ -->
  {% if report.severity_definitions %}
  <h2 id="severity-guide">Severity Guide</h2>
  <div class="severity-legend">
    {% for level, info in report.severity_definitions.items() %}
    <div class="severity-legend-item" style="background: {{ info.bg_color }}; border-left: 4px solid {{ info.color }};">
      <strong style="color: {{ info.color }}">{{ info.label }}</strong><br>
      <span style="font-size: 12px; color: #8e8e96;">{{ info.impact }}</span>
    </div>
    {% endfor %}
  </div>
  {% endif %}

  </div><!-- .content -->

  <!-- ════════════════════════════════════════════════════════════════ -->
  <!-- Footer                                                         -->
  <!-- ════════════════════════════════════════════════════════════════ -->
  <div class="footer">
    <strong>Roboter Schlafen Nicht</strong> &mdash; Autonomous Game QA<br>
    This report was generated automatically by the RSN Game QA platform.<br>
    &copy; {{ report.timestamp[:4] | default('2026') }} Roboter Schlafen Nicht.
    All rights reserved.
  </div>

</div><!-- .page -->

<!-- Back to top button -->
<button class="back-to-top" id="backToTop" onclick="window.scrollTo({top:0,behavior:'smooth'})"
        title="Back to top" aria-label="Back to top">&uarr;</button>

<!-- Screenshot modal -->
<div class="modal-overlay" id="screenshotModal" onclick="closeModal()"
     role="dialog" aria-modal="true">
  <img id="modalImg" src="" alt="Screenshot">
</div>
<script>
/* Screenshot modal */
function openModal(src) {
  document.getElementById('modalImg').src = src;
  document.getElementById('screenshotModal').classList.add('active');
}
function closeModal() {
  document.getElementById('screenshotModal').classList.remove('active');
}
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') closeModal();
});

/* QW5: Severity filter toggle */
var filterState = {critical: true, warning: true, info: false};
function toggleFilter(severity, btn) {
  filterState[severity] = !filterState[severity];
  btn.classList.toggle('active');
  var groups = document.querySelectorAll('[data-severity="' + severity + '"]');
  for (var i = 0; i < groups.length; i++) {
    groups[i].style.display = filterState[severity] ? '' : 'none';
  }
}
/* Apply initial filter state (info hidden by default) */
document.addEventListener('DOMContentLoaded', function() {
  var infoGroups = document.querySelectorAll('[data-severity="info"]');
  for (var i = 0; i < infoGroups.length; i++) {
    infoGroups[i].style.display = 'none';
  }
});

/* QW3: Nav bar scroll highlighting */
(function() {
  var sections = document.querySelectorAll('h2[id]');
  var navLinks = document.querySelectorAll('.nav-bar a');
  var observer = new IntersectionObserver(function(entries) {
    entries.forEach(function(entry) {
      if (entry.isIntersecting) {
        navLinks.forEach(function(a) { a.classList.remove('active'); });
        var target = document.querySelector('.nav-bar a[href="#' + entry.target.id + '"]');
        if (target) target.classList.add('active');
      }
    });
  }, {rootMargin: '-20% 0px -80% 0px'});
  sections.forEach(function(s) { observer.observe(s); });
})();

/* Back to top visibility */
window.addEventListener('scroll', function() {
  var btn = document.getElementById('backToTop');
  if (window.scrollY > 400) {
    btn.classList.add('visible');
  } else {
    btn.classList.remove('visible');
  }
});
</script>
</body>
</html>
"""
)


class DashboardRenderer:
    """Renders QA session reports as professional HTML dashboards.

    The renderer first looks for a Jinja2 template file at
    ``template_dir/template_name``.  If the file does not exist it
    falls back to a built-in professional template so that dashboards
    can always be generated without an external template directory.

    Parameters
    ----------
    template_dir : str or Path
        Directory containing Jinja2 template files.
        Default is ``"templates/"``.
    template_name : str
        Name of the main dashboard template file.
        Default is ``"dashboard.html.j2"``.

    Raises
    ------
    RuntimeError
        If Jinja2 is not installed.
    """

    def __init__(
        self,
        template_dir: str | Path = "templates",
        template_name: str = "dashboard.html.j2",
    ) -> None:
        if not _JINJA2_AVAILABLE:
            raise RuntimeError(
                "Jinja2 is required for DashboardRenderer. Install it with: pip install Jinja2"
            )

        self.template_dir = Path(template_dir)
        self.template_name = template_name

    def _get_template(self) -> jinja2.Template:
        """Load the Jinja2 template, falling back to the built-in one.

        Returns
        -------
        jinja2.Template
            A compiled Jinja2 template ready for rendering.
        """
        template_path = self.template_dir / self.template_name
        if template_path.is_file():
            env = Environment(
                loader=FileSystemLoader(str(self.template_dir)),
                autoescape=select_autoescape(["html"]),
            )
            return env.get_template(self.template_name)

        # Fall back to built-in template
        env = Environment(autoescape=select_autoescape(["html"]))
        return env.from_string(_BUILTIN_TEMPLATE)

    def render(
        self,
        report_data: dict[str, Any],
        baseline_data: dict[str, Any] | None = None,
    ) -> str:
        """Render a session report dict to a professional HTML string.

        Enriches findings with human-readable descriptions, embeds
        screenshots as base64 data URIs, and optionally includes a
        trained-vs-random comparison section.

        Parameters
        ----------
        report_data : dict[str, Any]
            The session report as returned by
            ``ReportGenerator.to_dict()``.
        baseline_data : dict[str, Any] | None
            Optional baseline (random agent) report for comparison.

        Returns
        -------
        str
            Complete HTML document as a string.
        """
        # Enrich with human-readable descriptions
        report_data = enrich_report_data(report_data)

        # Embed screenshots as base64
        report_data = _embed_screenshots(report_data)

        # Add comparison data if baseline provided
        report_data = _prepare_comparison(report_data, baseline_data)

        # Prepare grouped/deduplicated findings for the template
        report_data = _prepare_grouped_findings(report_data)

        template = self._get_template()
        return template.render(report=report_data)

    def render_to_file(
        self,
        report_data: dict[str, Any],
        output_path: str | Path,
        baseline_data: dict[str, Any] | None = None,
    ) -> Path:
        """Render a session report and write it to an HTML file.

        Creates parent directories if they do not exist.

        Parameters
        ----------
        report_data : dict[str, Any]
            The session report dict.
        output_path : str or Path
            Where to write the HTML file.
        baseline_data : dict[str, Any] | None
            Optional baseline report for comparison.

        Returns
        -------
        Path
            Path to the written HTML file.
        """
        html = self.render(report_data, baseline_data=baseline_data)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        return out

    def render_to_pdf(
        self,
        report_data: dict[str, Any],
        output_path: str | Path,
        baseline_data: dict[str, Any] | None = None,
    ) -> Path:
        """Render a session report to a PDF file.

        Uses weasyprint to convert the HTML report to PDF. Falls back
        to raising ``RuntimeError`` if weasyprint is not installed.

        Parameters
        ----------
        report_data : dict[str, Any]
            The session report dict.
        output_path : str or Path
            Where to write the PDF file.
        baseline_data : dict[str, Any] | None
            Optional baseline report for comparison.

        Returns
        -------
        Path
            Path to the written PDF file.

        Raises
        ------
        RuntimeError
            If weasyprint is not installed.
        """
        try:
            import weasyprint
        except ImportError:
            raise RuntimeError(
                "weasyprint is required for PDF export. Install it with: pip install weasyprint"
            ) from None

        html = self.render(report_data, baseline_data=baseline_data)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        weasyprint.HTML(string=html).write_pdf(str(out))
        return out

    def generate_dashboard(
        self,
        report_json_path: str | Path,
        output_path: str | Path | None = None,
        baseline_json_path: str | Path | None = None,
    ) -> Path:
        """Load a JSON report file and produce an HTML dashboard.

        This is the main entry point for CLI usage::

            renderer = DashboardRenderer()
            renderer.generate_dashboard("reports/session_abc.json")

        Parameters
        ----------
        report_json_path : str or Path
            Path to the session report JSON file.
        output_path : str or Path, optional
            Where to write the HTML file.  Defaults to the same
            directory as the JSON file with a ``.html`` extension.
        baseline_json_path : str or Path, optional
            Path to a baseline (random agent) report JSON for comparison.

        Returns
        -------
        Path
            Path to the written HTML file.

        Raises
        ------
        FileNotFoundError
            If the JSON report file does not exist.
        """
        json_path = Path(report_json_path)
        if not json_path.is_file():
            raise FileNotFoundError(f"Report file not found: {json_path}")

        with open(json_path, encoding="utf-8") as fh:
            report_data = json.load(fh)

        baseline_data = None
        if baseline_json_path is not None:
            bl_path = Path(baseline_json_path)
            if not bl_path.is_file():
                raise FileNotFoundError(f"Baseline file not found: {bl_path}")
            with open(bl_path, encoding="utf-8") as fh:
                baseline_data = json.load(fh)

        if output_path is None:
            output_path = json_path.with_suffix(".html")

        return self.render_to_file(report_data, output_path, baseline_data=baseline_data)
