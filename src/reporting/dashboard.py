"""HTML dashboard renderer — converts session reports to visual dashboards.

Uses Jinja2 templating to produce a self-contained HTML file with
episode summaries, finding details, and basic metrics charts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape  # noqa: F401

    _JINJA2_AVAILABLE = True
except ImportError:
    _JINJA2_AVAILABLE = False


class DashboardRenderer:
    """Renders QA session reports as HTML dashboards.

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
                "Jinja2 is required for DashboardRenderer. "
                "Install it with: pip install Jinja2"
            )

        self.template_dir = Path(template_dir)
        self.template_name = template_name

    def render(self, report_data: dict[str, Any]) -> str:
        """Render a session report dict to an HTML string.

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
        raise NotImplementedError("Dashboard rendering not yet implemented")

    def render_to_file(
        self,
        report_data: dict[str, Any],
        output_path: str | Path,
    ) -> Path:
        """Render a session report and write it to an HTML file.

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
        raise NotImplementedError("Dashboard file writing not yet implemented")

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
        """
        raise NotImplementedError("Dashboard generation not yet implemented")
