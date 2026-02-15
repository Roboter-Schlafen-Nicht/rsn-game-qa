"""Tests for the reporting module (ReportGenerator + DashboardRenderer).

Covers:
- FindingReport / EpisodeReport / SessionReport dataclass construction
- ReportGenerator: compute_summary, save (JSON I/O), to_dict
- DashboardRenderer: render, render_to_file, generate_dashboard
"""

from __future__ import annotations

import json
from pathlib import Path

from src.reporting.report import (
    EpisodeMetrics,
    EpisodeReport,
    FindingReport,
    ReportGenerator,
    SessionReport,
)
from src.reporting.dashboard import DashboardRenderer


# ── Helpers ──────────────────────────────────────────────────────────


def _make_finding(
    oracle: str = "crash",
    severity: str = "critical",
    step: int = 100,
    desc: str = "Game crashed",
) -> FindingReport:
    """Create a ``FindingReport`` with sensible defaults."""
    return FindingReport(
        oracle_name=oracle,
        severity=severity,
        step=step,
        description=desc,
    )


def _make_episode(
    episode_id: int = 1,
    steps: int = 500,
    reward: float = 42.0,
    findings: list[FindingReport] | None = None,
    **kwargs,
) -> EpisodeReport:
    """Create an ``EpisodeReport`` with sensible defaults."""
    return EpisodeReport(
        episode_id=episode_id,
        steps=steps,
        total_reward=reward,
        findings=findings or [],
        **kwargs,
    )


# ── FindingReport ────────────────────────────────────────────────────


class TestFindingReport:
    """Tests for the FindingReport dataclass."""

    def test_creation_with_required_fields(self):
        """FindingReport can be constructed with required fields."""
        fr = FindingReport(
            oracle_name="crash",
            severity="critical",
            step=100,
            description="Game window disappeared",
        )
        assert fr.oracle_name == "crash"
        assert fr.severity == "critical"
        assert fr.step == 100
        assert fr.data == {}
        assert fr.screenshot_path is None

    def test_creation_with_optional_fields(self):
        """FindingReport accepts optional data and screenshot_path."""
        fr = FindingReport(
            oracle_name="perf",
            severity="warning",
            step=50,
            description="Low FPS",
            data={"fps": 12.5},
            screenshot_path="screens/step50.png",
        )
        assert fr.data == {"fps": 12.5}
        assert fr.screenshot_path == "screens/step50.png"


# ── EpisodeReport ────────────────────────────────────────────────────


class TestEpisodeReport:
    """Tests for the EpisodeReport dataclass."""

    def test_defaults(self):
        """EpisodeReport should have sensible defaults."""
        ep = EpisodeReport(episode_id=1)
        assert ep.steps == 0
        assert ep.total_reward == 0.0
        assert ep.terminated is False
        assert ep.truncated is False
        assert ep.findings == []
        assert ep.metrics.mean_fps is None
        assert ep.seed is None

    def test_with_seed(self):
        """EpisodeReport accepts an optional seed."""
        ep = EpisodeReport(episode_id=1, seed=42)
        assert ep.seed == 42

    def test_with_metrics(self):
        """EpisodeReport accepts an EpisodeMetrics instance."""
        m = EpisodeMetrics(mean_fps=30.0, min_fps=18.5)
        ep = EpisodeReport(episode_id=1, metrics=m)
        assert ep.metrics.mean_fps == 30.0
        assert ep.metrics.min_fps == 18.5


# ── SessionReport ────────────────────────────────────────────────────


class TestSessionReport:
    """Tests for the SessionReport dataclass."""

    def test_defaults(self):
        """SessionReport generates UUID, timestamp, and build_id."""
        sr = SessionReport()
        assert len(sr.session_id) == 36  # UUID format
        assert sr.game == "breakout-71"
        assert sr.build_id == "local"  # no CI env var set
        assert "T" in sr.timestamp  # ISO-8601
        assert sr.episodes == []
        assert sr.summary == {}

    def test_build_id_from_env(self, monkeypatch):
        """SessionReport reads build_id from CI_COMMIT_SHORT_SHA."""
        monkeypatch.setenv("CI_COMMIT_SHORT_SHA", "abc1234")
        sr = SessionReport()
        assert sr.build_id == "abc1234"


# ── ReportGenerator ─────────────────────────────────────────────────


class TestReportGenerator:
    """Tests for ReportGenerator: init, add_episode, compute_summary, save."""

    def test_init_defaults(self):
        """ReportGenerator should initialise with default args."""
        gen = ReportGenerator()
        assert gen.game_name == "breakout-71"
        assert gen.output_dir == Path("reports")
        assert gen.session.game == "breakout-71"

    def test_add_episode(self):
        """add_episode should append to the session."""
        gen = ReportGenerator()
        ep = _make_episode()
        gen.add_episode(ep)
        assert len(gen.session.episodes) == 1
        assert gen.session.episodes[0].episode_id == 1

    # ── compute_summary ──────────────────────────────────────────

    def test_compute_summary_empty(self):
        """compute_summary with no episodes returns zeros."""
        gen = ReportGenerator()
        s = gen.compute_summary()
        assert s["total_episodes"] == 0
        assert s["total_findings"] == 0
        assert s["episodes_failed"] == 0
        assert s["mean_episode_reward"] == 0.0

    def test_compute_summary_no_findings(self):
        """compute_summary with clean episodes reports zero findings."""
        gen = ReportGenerator()
        gen.add_episode(_make_episode(episode_id=1, steps=100, reward=10.0))
        gen.add_episode(_make_episode(episode_id=2, steps=200, reward=20.0))
        s = gen.compute_summary()
        assert s["total_episodes"] == 2
        assert s["total_findings"] == 0
        assert s["critical_findings"] == 0
        assert s["warning_findings"] == 0
        assert s["info_findings"] == 0
        assert s["episodes_failed"] == 0
        assert s["mean_episode_reward"] == 15.0
        assert s["mean_episode_length"] == 150.0

    def test_compute_summary_with_findings(self):
        """compute_summary counts severity levels and failed episodes."""
        gen = ReportGenerator()
        gen.add_episode(
            _make_episode(
                episode_id=1,
                steps=100,
                reward=10.0,
                findings=[
                    _make_finding(severity="critical"),
                    _make_finding(severity="warning", oracle="perf"),
                ],
            )
        )
        gen.add_episode(
            _make_episode(
                episode_id=2,
                steps=200,
                reward=30.0,
                findings=[
                    _make_finding(severity="info", oracle="stuck"),
                ],
            )
        )
        s = gen.compute_summary()
        assert s["total_episodes"] == 2
        assert s["total_findings"] == 3
        assert s["critical_findings"] == 1
        assert s["warning_findings"] == 1
        assert s["info_findings"] == 1
        assert s["episodes_failed"] == 1  # only ep1 has critical
        assert s["mean_episode_reward"] == 20.0
        assert s["mean_episode_length"] == 150.0

    def test_compute_summary_all_failed(self):
        """episodes_failed counts every episode with a critical finding."""
        gen = ReportGenerator()
        for i in range(3):
            gen.add_episode(
                _make_episode(
                    episode_id=i,
                    findings=[_make_finding(severity="critical")],
                )
            )
        s = gen.compute_summary()
        assert s["episodes_failed"] == 3

    # ── save ─────────────────────────────────────────────────────

    def test_save_creates_json(self, tmp_path):
        """save() writes valid JSON with the expected structure."""
        gen = ReportGenerator(output_dir=tmp_path, game_name="test-game")
        gen.add_episode(
            _make_episode(
                episode_id=1,
                steps=100,
                reward=10.0,
                findings=[_make_finding(severity="warning", oracle="perf")],
            )
        )
        out = gen.save("test_report.json")

        assert out.exists()
        assert out.name == "test_report.json"

        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["game"] == "test-game"
        assert data["build_id"] == "local"
        assert len(data["episodes"]) == 1
        assert data["summary"]["total_episodes"] == 1
        assert data["summary"]["warning_findings"] == 1

    def test_save_auto_filename(self, tmp_path):
        """save() with no filename uses game_sessionid pattern."""
        gen = ReportGenerator(output_dir=tmp_path)
        out = gen.save()
        assert out.exists()
        assert out.name.startswith("breakout-71_")
        assert out.suffix == ".json"

    def test_save_creates_directory(self, tmp_path):
        """save() creates the output directory if it doesn't exist."""
        nested = tmp_path / "deep" / "nested"
        gen = ReportGenerator(output_dir=nested)
        out = gen.save("report.json")
        assert out.exists()
        assert nested.is_dir()

    # ── to_dict ──────────────────────────────────────────────────

    def test_to_dict_structure(self):
        """to_dict returns a dict with all required top-level keys."""
        gen = ReportGenerator()
        gen.add_episode(_make_episode())
        d = gen.to_dict()
        assert "session_id" in d
        assert "game" in d
        assert "build_id" in d
        assert "timestamp" in d
        assert "episodes" in d
        assert "summary" in d
        assert d["summary"]["total_episodes"] == 1

    def test_to_dict_findings_serialised(self):
        """to_dict serialises findings as plain dicts."""
        gen = ReportGenerator()
        gen.add_episode(
            _make_episode(
                findings=[_make_finding()],
            )
        )
        d = gen.to_dict()
        finding = d["episodes"][0]["findings"][0]
        assert isinstance(finding, dict)
        assert finding["oracle_name"] == "crash"
        assert finding["severity"] == "critical"


# ── DashboardRenderer ───────────────────────────────────────────────


class TestDashboardRenderer:
    """Tests for DashboardRenderer: render, render_to_file, generate_dashboard."""

    def _sample_report(self) -> dict:
        """Create a sample report dict for rendering tests."""
        gen = ReportGenerator(game_name="test-game")
        gen.add_episode(
            _make_episode(
                episode_id=1,
                steps=100,
                reward=25.0,
                findings=[
                    _make_finding(severity="critical", desc="Game crashed"),
                    _make_finding(severity="info", oracle="perf", desc="Low FPS"),
                ],
            )
        )
        gen.add_episode(
            _make_episode(
                episode_id=2,
                steps=200,
                reward=50.0,
            )
        )
        return gen.to_dict()

    def test_render_returns_html(self):
        """render() returns a non-empty HTML string."""
        renderer = DashboardRenderer()
        html = renderer.render(self._sample_report())
        assert "<!DOCTYPE html>" in html
        assert "RSN Game QA" in html

    def test_render_contains_episode_data(self):
        """render() includes episode-level information."""
        renderer = DashboardRenderer()
        html = renderer.render(self._sample_report())
        assert "FAIL" in html  # episode 1 has critical finding
        assert "PASS" in html  # episode 2 is clean
        assert "Game crashed" in html
        assert "test-game" in html

    def test_render_to_file(self, tmp_path):
        """render_to_file() writes HTML to disk."""
        renderer = DashboardRenderer()
        report = self._sample_report()
        out = renderer.render_to_file(report, tmp_path / "dashboard.html")
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content

    def test_render_to_file_creates_dirs(self, tmp_path):
        """render_to_file() creates parent directories."""
        renderer = DashboardRenderer()
        report = self._sample_report()
        out = renderer.render_to_file(report, tmp_path / "a" / "b" / "dash.html")
        assert out.exists()

    def test_generate_dashboard_from_json(self, tmp_path):
        """generate_dashboard() reads JSON and produces HTML."""
        # Write a JSON report first
        gen = ReportGenerator(output_dir=tmp_path)
        gen.add_episode(_make_episode(steps=300, reward=75.0))
        json_path = gen.save("session.json")

        # Generate dashboard from it
        renderer = DashboardRenderer()
        html_path = renderer.generate_dashboard(json_path)

        assert html_path.exists()
        assert html_path.suffix == ".html"
        assert html_path.stem == "session"
        content = html_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content

    def test_generate_dashboard_custom_output(self, tmp_path):
        """generate_dashboard() respects a custom output_path."""
        gen = ReportGenerator(output_dir=tmp_path)
        gen.add_episode(_make_episode())
        json_path = gen.save("data.json")

        renderer = DashboardRenderer()
        custom = tmp_path / "custom" / "index.html"
        html_path = renderer.generate_dashboard(json_path, output_path=custom)

        assert html_path == custom
        assert html_path.exists()

    def test_generate_dashboard_missing_json(self, tmp_path):
        """generate_dashboard() raises FileNotFoundError for missing files."""
        renderer = DashboardRenderer()
        import pytest

        with pytest.raises(FileNotFoundError, match="not found"):
            renderer.generate_dashboard(tmp_path / "nonexistent.json")

    def test_render_empty_report(self):
        """render() handles a report with no episodes gracefully."""
        renderer = DashboardRenderer()
        gen = ReportGenerator()
        html = renderer.render(gen.to_dict())
        assert "<!DOCTYPE html>" in html
