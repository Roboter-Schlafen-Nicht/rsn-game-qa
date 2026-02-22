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


def _embed_screenshots(report_data: dict[str, Any]) -> dict[str, Any]:
    """Convert screenshot file paths to base64 data URIs for inline embedding.

    Modifies the report data in-place. Screenshots that cannot be read
    are silently skipped (path set to ``None``).

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
            if path.is_file():
                try:
                    img_bytes = path.read_bytes()
                    b64 = base64.b64encode(img_bytes).decode("ascii")
                    # Determine MIME type from extension
                    suffix = path.suffix.lower()
                    mime = {
                        ".png": "image/png",
                        ".jpg": "image/jpeg",
                        ".jpeg": "image/jpeg",
                        ".gif": "image/gif",
                        ".webp": "image/webp",
                    }.get(suffix, "image/png")
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


# ── RSN logo (inline SVG) ───────────────────────────────────────────

_RSN_LOGO_SVG = """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 40" width="200" height="40">
  <rect width="200" height="40" rx="6" fill="#1a1a2e"/>
  <text x="12" y="28" font-family="system-ui, -apple-system, sans-serif"
        font-size="20" font-weight="700" fill="#e94560">RSN</text>
  <text x="62" y="28" font-family="system-ui, -apple-system, sans-serif"
        font-size="14" fill="#e8e8e8">Game QA</text>
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
  /* ── Reset & base ── */
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    line-height: 1.6;
    color: #1a1a2e;
    background: #f8f9fa;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
  a { color: #e94560; text-decoration: none; }
  a:hover { text-decoration: underline; }

  /* ── Layout ── */
  .page { max-width: 960px; margin: 0 auto; padding: 24px; }
  .header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 20px 24px; background: #1a1a2e; color: #fff; border-radius: 8px 8px 0 0;
  }
  .header-logo { display: flex; align-items: center; gap: 16px; }
  .header-meta { text-align: right; font-size: 13px; color: #adb5bd; }
  .header-meta strong { color: #fff; }
  .content { background: #fff; padding: 32px; border-radius: 0 0 8px 8px;
             box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  .footer {
    text-align: center; padding: 20px; margin-top: 24px;
    font-size: 12px; color: #868e96;
  }

  /* ── Typography ── */
  h1 { font-size: 24px; margin-bottom: 4px; }
  h2 { font-size: 20px; margin: 32px 0 16px; padding-bottom: 8px;
       border-bottom: 2px solid #e9ecef; color: #1a1a2e; }
  h3 { font-size: 16px; margin: 20px 0 8px; color: #495057; }
  p { margin-bottom: 12px; }

  /* ── Verdict banner ── */
  .verdict-banner {
    display: flex; align-items: center; gap: 16px;
    padding: 20px 24px; border-radius: 8px; margin-bottom: 24px;
  }
  .verdict-fail { background: #f8d7da; border-left: 5px solid #dc3545; }
  .verdict-warn { background: #fff3cd; border-left: 5px solid #fd7e14; }
  .verdict-pass { background: #d1e7dd; border-left: 5px solid #198754; }
  .verdict-label {
    font-size: 28px; font-weight: 800; letter-spacing: 1px;
  }
  .verdict-fail .verdict-label { color: #dc3545; }
  .verdict-warn .verdict-label { color: #fd7e14; }
  .verdict-pass .verdict-label { color: #198754; }
  .verdict-narrative { font-size: 15px; color: #495057; }
  .verdict-narrative strong { color: #1a1a2e; }

  /* ── Stats cards ── */
  .stats-row {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px; margin-bottom: 24px;
  }
  .stat-card {
    text-align: center; padding: 16px 12px; border-radius: 8px;
    background: #f8f9fa; border: 1px solid #e9ecef;
  }
  .stat-value { font-size: 28px; font-weight: 700; }
  .stat-label { font-size: 12px; color: #868e96; text-transform: uppercase;
                letter-spacing: 0.5px; margin-top: 4px; }
  .stat-critical .stat-value { color: #dc3545; }
  .stat-warning .stat-value { color: #fd7e14; }
  .stat-info .stat-value { color: #0d6efd; }
  .stat-neutral .stat-value { color: #1a1a2e; }
  .stat-good .stat-value { color: #198754; }

  /* ── Severity badges ── */
  .badge {
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    font-size: 12px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .badge-critical { background: #dc3545; color: #fff; }
  .badge-warning { background: #fd7e14; color: #fff; }
  .badge-info { background: #0d6efd; color: #fff; }
  .badge-pass { background: #198754; color: #fff; }
  .badge-fail { background: #dc3545; color: #fff; }

  /* ── Tables ── */
  table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
  th { text-align: left; padding: 10px 12px; background: #f8f9fa;
       border-bottom: 2px solid #dee2e6; font-size: 13px;
       text-transform: uppercase; color: #495057; letter-spacing: 0.5px; }
  td { padding: 10px 12px; border-bottom: 1px solid #e9ecef;
       font-size: 14px; vertical-align: top; }
  tr:hover { background: #f8f9fa; }

  /* ── Finding cards ── */
  .finding-card {
    border: 1px solid #e9ecef; border-radius: 8px; padding: 16px;
    margin-bottom: 12px; background: #fff;
  }
  .finding-card-critical { border-left: 4px solid #dc3545; }
  .finding-card-warning { border-left: 4px solid #fd7e14; }
  .finding-card-info { border-left: 4px solid #0d6efd; }
  .finding-header {
    display: flex; align-items: center; gap: 10px; margin-bottom: 8px;
  }
  .finding-title { font-weight: 600; font-size: 15px; }
  .finding-meta { font-size: 12px; color: #868e96; }
  .finding-body { font-size: 14px; color: #495057; margin-bottom: 8px; }
  .finding-recommendation {
    font-size: 13px; color: #495057; background: #f8f9fa;
    padding: 10px 14px; border-radius: 6px; margin-top: 8px;
  }
  .finding-recommendation strong { color: #1a1a2e; }
  .finding-screenshot {
    margin-top: 10px; border-radius: 6px; max-width: 100%;
    border: 1px solid #e9ecef; cursor: pointer;
  }
  .finding-screenshot:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.15); }

  /* ── Comparison section ── */
  .comparison-table td:first-child { font-weight: 600; width: 40%; }
  .comparison-narrative { font-size: 15px; color: #495057;
                          margin-bottom: 16px; line-height: 1.7; }
  .comparison-narrative strong { color: #1a1a2e; }

  /* ── Recommendation priority ── */
  .rec-list { list-style: none; counter-reset: rec-counter; }
  .rec-item {
    counter-increment: rec-counter; padding: 14px 16px 14px 52px;
    margin-bottom: 8px; border-radius: 8px; background: #f8f9fa;
    border: 1px solid #e9ecef; position: relative; font-size: 14px;
  }
  .rec-item::before {
    content: counter(rec-counter); position: absolute; left: 16px;
    top: 14px; width: 24px; height: 24px; border-radius: 50%;
    background: #1a1a2e; color: #fff; text-align: center;
    line-height: 24px; font-size: 12px; font-weight: 700;
  }
  .rec-title { font-weight: 600; margin-bottom: 4px; }
  .rec-body { color: #495057; font-size: 13px; }
  .rec-count { font-size: 12px; color: #868e96; }

  /* ── Severity explanation ── */
  .severity-legend {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;
    margin-bottom: 24px;
  }
  .severity-legend-item {
    padding: 12px 16px; border-radius: 8px; font-size: 13px;
  }

  /* ── Print styles ── */
  @media print {
    body { background: #fff; }
    .page { max-width: 100%; padding: 0; }
    .header { border-radius: 0; }
    .content { box-shadow: none; border-radius: 0; }
    .finding-card { break-inside: avoid; }
    .verdict-banner { break-inside: avoid; }
    h2 { break-after: avoid; }
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
      {{ report.timestamp | default('') }}<br>
      <span style="font-size:11px">Session {{ (report.session_id | default(''))[:8] }}</span>
    </div>
  </div>

  <div class="content">

  <!-- ════════════════════════════════════════════════════════════════ -->
  <!-- Executive Summary                                              -->
  <!-- ════════════════════════════════════════════════════════════════ -->
  {% if report.executive_summary %}
  <h2>Executive Summary</h2>

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
  <!-- Severity Guide                                                 -->
  <!-- ════════════════════════════════════════════════════════════════ -->
  {% if report.severity_definitions %}
  <h2>Severity Guide</h2>
  <div class="severity-legend">
    {% for level, info in report.severity_definitions.items() %}
    <div class="severity-legend-item" style="background: {{ info.bg_color }}; border-left: 4px solid {{ info.color }};">
      <strong style="color: {{ info.color }}">{{ info.label }}</strong><br>
      <span style="font-size: 12px; color: #495057;">{{ info.impact }}</span>
    </div>
    {% endfor %}
  </div>
  {% endif %}

  <!-- ════════════════════════════════════════════════════════════════ -->
  <!-- Trained vs Random Comparison                                   -->
  <!-- ════════════════════════════════════════════════════════════════ -->
  {% if report.comparison %}
  <h2>Trained Agent vs Random Baseline</h2>

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
  <!-- Findings Detail                                                -->
  <!-- ════════════════════════════════════════════════════════════════ -->
  <h2>Findings</h2>

  {% set all_findings = [] %}
  {% for ep in report.episodes %}
    {% for f in ep.findings %}
      {% set _ = all_findings.append({"finding": f, "episode_id": ep.episode_id, "episode_steps": ep.steps}) %}
    {% endfor %}
  {% endfor %}

  {% if all_findings %}
  {% for item in all_findings %}
    {% set f = item.finding %}
    <div class="finding-card finding-card-{{ f.severity }}">
      <div class="finding-header">
        <span class="badge badge-{{ f.severity }}">{{ f.severity_info.label | default(f.severity) if f.severity_info is defined else f.severity | upper }}</span>
        <span class="finding-title">{{ f.human_title | default(f.oracle_name) if f.human_title is defined else f.oracle_name }}</span>
        <span class="finding-meta">Episode {{ item.episode_id }} &middot; Step {{ f.step }}</span>
      </div>
      <div class="finding-body">
        {{ f.human_description | default(f.description) if f.human_description is defined else f.description }}
      </div>
      {% if f.screenshot_data_uri is defined and f.screenshot_data_uri %}
      <img class="finding-screenshot" src="{{ f.screenshot_data_uri }}"
           alt="Screenshot at step {{ f.step }}" onclick="openModal(this.src)"
           title="Click to enlarge">
      {% endif %}
      {% if f.human_recommendation is defined and f.human_recommendation %}
      <div class="finding-recommendation">
        <strong>Recommendation:</strong> {{ f.human_recommendation }}
      </div>
      {% endif %}
    </div>
  {% endfor %}
  {% else %}
  <p style="color: #198754; font-weight: 600;">No issues were detected during testing.</p>
  {% endif %}

  <!-- ════════════════════════════════════════════════════════════════ -->
  <!-- Episode Summary Table                                          -->
  <!-- ════════════════════════════════════════════════════════════════ -->
  <h2>Episode Summary</h2>
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
  <!-- Recommendations                                                -->
  <!-- ════════════════════════════════════════════════════════════════ -->
  {% if report.recommendations %}
  <h2>Recommendations</h2>
  <p style="color: #495057; font-size: 14px; margin-bottom: 16px;">
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

<!-- Screenshot modal -->
<div class="modal-overlay" id="screenshotModal" onclick="closeModal()">
  <img id="modalImg" src="" alt="Screenshot">
</div>
<script>
function openModal(src) {
  document.getElementById('modalImg').src = src;
  document.getElementById('screenshotModal').classList.add('active');
}
function closeModal() {
  document.getElementById('screenshotModal').classList.remove('active');
}
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
