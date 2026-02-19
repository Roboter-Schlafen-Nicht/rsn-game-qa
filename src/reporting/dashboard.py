"""HTML dashboard renderer — converts session reports to visual dashboards.

Uses Jinja2 templating to produce a self-contained HTML file with
episode summaries, finding details, and basic metrics.  Falls back to
a built-in minimal template when the configured template file does not
exist on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    _JINJA2_AVAILABLE = True
except ImportError:
    _JINJA2_AVAILABLE = False

if TYPE_CHECKING:
    import jinja2

#: Minimal self-contained Bootstrap 5 dashboard template.
#: Used as a fallback when no external template file is provided.
_BUILTIN_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RSN Game QA Dashboard</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
      rel="stylesheet">
<style>
  .severity-critical { color: #dc3545; font-weight: bold; }
  .severity-warning  { color: #fd7e14; }
  .severity-info     { color: #198754; }
  .badge-pass { background-color: #198754; }
  .badge-fail { background-color: #dc3545; }
</style>
</head>
<body>
<div class="container my-4">
  <h1>RSN Game QA &mdash; Dashboard</h1>

  {% for run in runs %}
  <div class="card my-3">
    <div class="card-header">
      <strong>{{ run.game }}</strong>
      &mdash; Session {{ run.session_id[:8] }}
      &mdash; Build <code>{{ run.build_id }}</code>
      <span class="text-muted float-end">{{ run.timestamp }}</span>
    </div>
    <div class="card-body">

      {# ── Summary row ── #}
      {% if run.summary %}
      <div class="row mb-3">
        <div class="col">Episodes: <strong>{{ run.summary.total_episodes }}</strong></div>
        <div class="col">Failed: <strong>{{ run.summary.episodes_failed }}</strong></div>
        <div class="col">Critical: <strong>{{ run.summary.critical_findings }}</strong></div>
        <div class="col">Warnings: <strong>{{ run.summary.warning_findings }}</strong></div>
        <div class="col">Info: <strong>{{ run.summary.info_findings }}</strong></div>
      </div>
      {% endif %}

      {# ── Episodes table ── #}
      <table class="table table-sm table-striped">
        <thead>
          <tr>
            <th>#</th><th>Status</th><th>Steps</th><th>Reward</th>
            <th>Findings</th><th>FPS (avg)</th>
          </tr>
        </thead>
        <tbody>
        {% for ep in run.episodes %}
          {% set failed = ep.findings | selectattr("severity", "equalto", "critical") | list | length > 0 %}
          <tr>
            <td>{{ ep.episode_id }}</td>
            <td><span class="badge {{ 'badge-fail' if failed else 'badge-pass' }}">
              {{ "FAIL" if failed else "PASS" }}
            </span></td>
            <td>{{ ep.steps }}</td>
            <td>{{ "%.1f" | format(ep.total_reward) }}</td>
            <td>
              {% for f in ep.findings %}
              <span class="severity-{{ f.severity }}">{{ f.oracle_name }}: {{ f.description }}</span><br>
              {% endfor %}
              {% if not ep.findings %}&mdash;{% endif %}
            </td>
            <td>{{ "%.1f" | format(ep.metrics.mean_fps) if ep.metrics.mean_fps is not none else "&mdash;" }}</td>
          </tr>
        {% endfor %}
        </tbody>
      </table>

    </div>
  </div>
  {% endfor %}

  {% if not runs %}
  <div class="alert alert-info">No reports found.</div>
  {% endif %}

</div>
</body>
</html>
"""


class DashboardRenderer:
    """Renders QA session reports as HTML dashboards.

    The renderer first looks for a Jinja2 template file at
    ``template_dir/template_name``.  If the file does not exist it
    falls back to a built-in Bootstrap 5 template so that dashboards
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

    def render(self, report_data: dict[str, Any]) -> str:
        """Render a session report dict to an HTML string.

        The report data is wrapped in a single-element ``runs`` list
        so the template can iterate uniformly over one or many runs.

        Parameters
        ----------
        report_data : dict[str, Any]
            The session report as returned by
            ``ReportGenerator.to_dict()``.

        Returns
        -------
        str
            Complete HTML document as a string.
        """
        template = self._get_template()
        return template.render(runs=[report_data])

    def render_to_file(
        self,
        report_data: dict[str, Any],
        output_path: str | Path,
    ) -> Path:
        """Render a session report and write it to an HTML file.

        Creates parent directories if they do not exist.

        Parameters
        ----------
        report_data : dict[str, Any]
            The session report dict.
        output_path : str or Path
            Where to write the HTML file.

        Returns
        -------
        Path
            Path to the written HTML file.
        """
        html = self.render(report_data)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        return out

    def generate_dashboard(
        self,
        report_json_path: str | Path,
        output_path: str | Path | None = None,
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

        if output_path is None:
            output_path = json_path.with_suffix(".html")

        return self.render_to_file(report_data, output_path)
