"""Tests for the reporting module (ReportGenerator + DashboardRenderer)."""

from src.reporting.report import (
    EpisodeReport,
    FindingReport,
    ReportGenerator,
)


class TestFindingReport:
    """Tests for the FindingReport dataclass."""

    def test_finding_report_creation(self):
        """FindingReport can be constructed with required fields."""
        fr = FindingReport(
            oracle_name="crash",
            severity="critical",
            step=100,
            description="Game window disappeared",
        )
        assert fr.oracle_name == "crash"
        assert fr.data == {}
        assert fr.screenshot_path is None


class TestEpisodeReport:
    """Tests for the EpisodeReport dataclass."""

    def test_episode_report_defaults(self):
        """EpisodeReport should have sensible defaults."""
        ep = EpisodeReport(episode_id=1)
        assert ep.steps == 0
        assert ep.total_reward == 0.0
        assert ep.findings == []
        assert ep.metrics.mean_fps is None


class TestReportGenerator:
    """Placeholder tests for ReportGenerator."""

    def test_report_generator_init(self):
        """ReportGenerator should initialise with default args."""
        gen = ReportGenerator()
        assert gen.game_name == "breakout-71"

    def test_add_episode(self):
        """add_episode should append to the session."""
        gen = ReportGenerator()
        ep = EpisodeReport(episode_id=1, steps=500, total_reward=42.0)
        gen.add_episode(ep)
        assert len(gen._session.episodes) == 1

    def test_to_dict_structure(self):
        """to_dict should return a dict with required keys."""
        # TODO: implement once compute_summary is implemented
        pass


class TestDashboardRenderer:
    """Placeholder tests for DashboardRenderer."""

    def test_dashboard_renderer_requires_jinja2(self):
        """DashboardRenderer should raise if Jinja2 is missing."""
        # TODO: mock Jinja2 unavailability
        pass
