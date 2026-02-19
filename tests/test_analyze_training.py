"""Tests for scripts/analyze_training.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_training import (
    analyze_paddle_movement,
    analyze_per_episode_rnd,
    build_analysis,
    compute_coverage_stats,
    compute_episode_stats,
    compute_step_stats,
    extract_events_by_type,
    format_report,
    identify_degenerate_episodes,
    main,
    parse_jsonl,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config_event(**overrides: object) -> dict:
    """Create a config event with defaults."""
    args = {
        "policy": "cnn",
        "reward_mode": "rnd",
        "timesteps": 200000,
        "max_steps": 2000,
        "survival_bonus": 0.0,
        "epsilon_greedy": 0.1,
    }
    args.update(overrides)
    return {"event": "config", "game": "breakout71", "args": args}


def _make_episode_event(
    episode: int = 0,
    steps: int = 100,
    total_reward: float = -5.0,
    termination: str = "game_over",
    rnd_mean: float = 0.005,
    rnd_total: float = 0.5,
    brick_count: int = 0,
) -> dict:
    return {
        "event": "episode_end",
        "episode": episode,
        "steps": steps,
        "total_reward": total_reward,
        "mean_step_reward": total_reward / max(steps, 1),
        "termination": termination,
        "brick_count": brick_count,
        "duration_seconds": steps * 0.07,
        "mean_fps": 14.3,
        "rnd_intrinsic_total": rnd_total,
        "rnd_intrinsic_mean": rnd_mean,
    }


def _make_step_event(
    step: int = 100,
    episode: int = 0,
    reward: float = 0.001,
    paddle_x: float = 0.5,
    rnd_raw: float = 0.001,
    rnd_norm: float = 0.0001,
    fps: float = 15.0,
    brick_count: int = 0,
) -> dict:
    return {
        "event": "step_summary",
        "step": step,
        "episode": episode,
        "reward": reward,
        "cumulative_reward": reward * step,
        "ball_detected": True,
        "brick_count": brick_count,
        "paddle_x": paddle_x,
        "action": 0.5,
        "fps": fps,
        "no_ball_count": 0,
        "rnd_intrinsic_raw": rnd_raw,
        "rnd_intrinsic_norm": rnd_norm,
    }


def _make_coverage_event(step: int = 10000, unique_states: int = 5000) -> dict:
    return {
        "event": "coverage_summary",
        "step": step,
        "unique_states": unique_states,
    }


@pytest.fixture
def sample_jsonl(tmp_path: Path) -> Path:
    """Create a sample JSONL file with realistic training data."""
    events = [
        _make_config_event(),
        _make_episode_event(0, 3, 234.2, "game_over", 0.0196, 0.042),
        _make_step_event(100, 1, 0.0006, 0.618, 7.2e-6, 0.0006, 15.4),
        _make_step_event(200, 1, 0.0002, 0.618, 3.1e-6, 0.0002, 15.0),
        _make_episode_event(1, 250, -3.02, "game_over", 0.005, 0.1),
        _make_step_event(300, 2, 0.001, 0.4, 0.01, 0.001, 14.5),
        _make_step_event(400, 2, 0.0008, 0.45, 0.008, 0.0008, 14.8),
        _make_step_event(500, 2, 0.0005, 0.5, 0.005, 0.0005, 15.0),
        _make_episode_event(2, 2000, 3.67, "truncated", 0.00003, 0.06),
        _make_coverage_event(10000, 5000),
        _make_episode_event(3, 42, -8.49, "game_over", 0.002, 0.092),
        _make_episode_event(4, 10000, 3.76, "truncated", 0.000027, 0.27),
        _make_coverage_event(20000, 9500),
    ]

    path = tmp_path / "training.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")
    return path


@pytest.fixture
def sample_events(sample_jsonl: Path) -> list[dict]:
    """Parse the sample JSONL file."""
    return parse_jsonl(sample_jsonl)


# ---------------------------------------------------------------------------
# parse_jsonl
# ---------------------------------------------------------------------------


class TestParseJsonl:
    """Tests for parse_jsonl."""

    def test_parse_valid_file(self, sample_jsonl: Path) -> None:
        events = parse_jsonl(sample_jsonl)
        assert len(events) == 13
        assert events[0]["event"] == "config"

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            parse_jsonl(tmp_path / "nonexistent.jsonl")

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        assert parse_jsonl(path) == []

    def test_blank_lines_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "blanks.jsonl"
        path.write_text('{"event": "config"}\n\n\n{"event": "episode_end"}\n')
        events = parse_jsonl(path)
        assert len(events) == 2

    def test_malformed_line_skipped(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text('{"event": "config"}\nNOT_JSON\n{"event": "episode_end"}\n')
        events = parse_jsonl(path)
        assert len(events) == 2
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "line 2" in captured.err


# ---------------------------------------------------------------------------
# extract_events_by_type
# ---------------------------------------------------------------------------


class TestExtractEventsByType:
    """Tests for extract_events_by_type."""

    def test_groups_correctly(self, sample_events: list[dict]) -> None:
        by_type = extract_events_by_type(sample_events)
        assert "config" in by_type
        assert "episode_end" in by_type
        assert "step_summary" in by_type
        assert "coverage_summary" in by_type

    def test_event_counts(self, sample_events: list[dict]) -> None:
        by_type = extract_events_by_type(sample_events)
        assert len(by_type["config"]) == 1
        assert len(by_type["episode_end"]) == 5
        assert len(by_type["step_summary"]) == 5
        assert len(by_type["coverage_summary"]) == 2

    def test_empty_events(self) -> None:
        by_type = extract_events_by_type([])
        assert by_type == {}

    def test_unknown_event_type(self) -> None:
        events = [{"data": "no_event_field"}, {"event": "config"}]
        by_type = extract_events_by_type(events)
        assert len(by_type["unknown"]) == 1
        assert len(by_type["config"]) == 1


# ---------------------------------------------------------------------------
# compute_episode_stats
# ---------------------------------------------------------------------------


class TestComputeEpisodeStats:
    """Tests for compute_episode_stats."""

    def test_empty_episodes(self) -> None:
        stats = compute_episode_stats([])
        assert stats["count"] == 0

    def test_basic_stats(self) -> None:
        episodes = [
            _make_episode_event(0, 100, -5.0, "game_over", 0.005),
            _make_episode_event(1, 200, -3.0, "game_over", 0.003),
            _make_episode_event(2, 2000, 3.0, "truncated", 0.00003),
        ]
        stats = compute_episode_stats(episodes)
        assert stats["count"] == 3
        assert stats["steps"]["min"] == 100
        assert stats["steps"]["max"] == 2000
        assert stats["termination"]["game_over"] == 2
        assert stats["termination"]["truncated"] == 1

    def test_rnd_stats_present(self) -> None:
        episodes = [
            _make_episode_event(0, 100, -5.0, rnd_mean=0.005),
            _make_episode_event(1, 200, -3.0, rnd_mean=0.003),
        ]
        stats = compute_episode_stats(episodes)
        assert stats["rnd"]["count"] == 2
        assert stats["rnd"]["mean"] == pytest.approx(0.004, abs=0.001)


# ---------------------------------------------------------------------------
# compute_step_stats
# ---------------------------------------------------------------------------


class TestComputeStepStats:
    """Tests for compute_step_stats."""

    def test_empty_steps(self) -> None:
        stats = compute_step_stats([])
        assert stats["count"] == 0

    def test_basic_step_stats(self) -> None:
        steps = [
            _make_step_event(100, fps=14.0, paddle_x=0.3),
            _make_step_event(200, fps=15.0, paddle_x=0.5),
            _make_step_event(300, fps=16.0, paddle_x=0.7),
        ]
        stats = compute_step_stats(steps)
        assert stats["count"] == 3
        assert stats["total_steps"] == 300
        assert stats["fps"]["mean"] == 15.0
        assert stats["paddle"]["range"] == pytest.approx(0.4, abs=0.01)


# ---------------------------------------------------------------------------
# compute_coverage_stats
# ---------------------------------------------------------------------------


class TestComputeCoverageStats:
    """Tests for compute_coverage_stats."""

    def test_empty_coverage(self) -> None:
        stats = compute_coverage_stats([])
        assert stats["count"] == 0

    def test_single_checkpoint(self) -> None:
        events = [_make_coverage_event(10000, 5000)]
        stats = compute_coverage_stats(events)
        assert stats["count"] == 1
        assert stats["latest_unique_states"] == 5000
        assert stats["growth_per_1k_steps"] == 500.0

    def test_multiple_checkpoints(self) -> None:
        events = [
            _make_coverage_event(10000, 5000),
            _make_coverage_event(20000, 9000),
            _make_coverage_event(30000, 12000),
        ]
        stats = compute_coverage_stats(events)
        assert stats["count"] == 3
        assert stats["latest_unique_states"] == 12000
        assert stats["latest_step"] == 30000
        # Growth: (12000 - 5000) / (30000 - 10000) * 1000 = 350
        assert stats["growth_per_1k_steps"] == 350.0


# ---------------------------------------------------------------------------
# analyze_paddle_movement
# ---------------------------------------------------------------------------


class TestAnalyzePaddleMovement:
    """Tests for analyze_paddle_movement."""

    def test_empty_steps(self) -> None:
        result = analyze_paddle_movement([])
        assert result["unique_positions"] == 0
        assert result["degenerate"] is False

    def test_healthy_movement(self) -> None:
        steps = [
            _make_step_event(paddle_x=0.2),
            _make_step_event(paddle_x=0.4),
            _make_step_event(paddle_x=0.6),
            _make_step_event(paddle_x=0.8),
            _make_step_event(paddle_x=0.3),
        ]
        result = analyze_paddle_movement(steps)
        assert result["unique_positions"] == 5
        assert result["degenerate"] is False

    def test_degenerate_movement(self) -> None:
        # 9/10 at same position = 90% > 80% threshold
        steps = [_make_step_event(paddle_x=0.618)] * 9 + [_make_step_event(paddle_x=0.3)]
        result = analyze_paddle_movement(steps)
        assert result["degenerate"] is True
        assert result["most_common_position"] == 0.618
        assert result["most_common_pct"] == 90.0

    def test_no_paddle_data(self) -> None:
        steps = [{"event": "step_summary", "step": 100}]
        result = analyze_paddle_movement(steps)
        assert result["unique_positions"] == 0

    def test_position_frequency_top_10(self) -> None:
        # Create 15 unique positions
        steps = [_make_step_event(paddle_x=i * 0.05) for i in range(15)]
        result = analyze_paddle_movement(steps)
        assert len(result["position_frequency"]) <= 10


# ---------------------------------------------------------------------------
# analyze_per_episode_rnd
# ---------------------------------------------------------------------------


class TestAnalyzePerEpisodeRnd:
    """Tests for analyze_per_episode_rnd."""

    def test_empty_episodes(self) -> None:
        assert analyze_per_episode_rnd([]) == []

    def test_collapse_detection(self) -> None:
        episodes = [
            _make_episode_event(0, rnd_mean=0.005),  # not collapsed
            _make_episode_event(1, rnd_mean=0.00003),  # collapsed
            _make_episode_event(2, rnd_mean=0.00009),  # collapsed (just below 1e-4)
        ]
        results = analyze_per_episode_rnd(episodes)
        assert len(results) == 3
        assert results[0]["rnd_collapsed"] is False
        assert results[1]["rnd_collapsed"] is True
        assert results[2]["rnd_collapsed"] is True

    def test_boundary_value(self) -> None:
        # Exactly 1e-4 is NOT collapsed (< 1e-4)
        episodes = [_make_episode_event(0, rnd_mean=0.0001)]
        results = analyze_per_episode_rnd(episodes)
        # 0.0001 == 1e-4, so < 1e-4 is False
        assert results[0]["rnd_collapsed"] is False


# ---------------------------------------------------------------------------
# identify_degenerate_episodes
# ---------------------------------------------------------------------------


class TestIdentifyDegenerateEpisodes:
    """Tests for identify_degenerate_episodes."""

    def test_empty_episodes(self) -> None:
        assert identify_degenerate_episodes([]) == []

    def test_identifies_degenerate(self) -> None:
        episodes = [
            _make_episode_event(0, 50, termination="game_over"),
            _make_episode_event(1, 10000, termination="truncated"),
            _make_episode_event(2, 2000, termination="truncated"),
            _make_episode_event(3, 100, termination="game_over"),
        ]
        degenerate = identify_degenerate_episodes(episodes, threshold=500)
        assert len(degenerate) == 2
        assert degenerate[0]["episode"] == 1
        assert degenerate[1]["episode"] == 2

    def test_custom_threshold(self) -> None:
        episodes = [
            _make_episode_event(0, 600, termination="truncated"),
        ]
        # Threshold 1000 — 600 < 1000, so not degenerate
        assert identify_degenerate_episodes(episodes, threshold=1000) == []
        # Threshold 500 — 600 >= 500, degenerate
        assert len(identify_degenerate_episodes(episodes, threshold=500)) == 1

    def test_game_over_not_degenerate(self) -> None:
        # Long episode but ends by game_over = legitimate play
        episodes = [_make_episode_event(0, 5000, termination="game_over")]
        assert identify_degenerate_episodes(episodes) == []


# ---------------------------------------------------------------------------
# build_analysis
# ---------------------------------------------------------------------------


class TestBuildAnalysis:
    """Tests for build_analysis."""

    def test_full_analysis(self, sample_events: list[dict]) -> None:
        analysis = build_analysis(sample_events)
        assert "config" in analysis
        assert analysis["config"]["game"] == "breakout71"
        assert analysis["config"]["policy"] == "cnn"
        assert analysis["episode_stats"]["count"] == 5
        assert analysis["step_stats"]["count"] == 5
        assert analysis["coverage"]["count"] == 2
        assert "paddle_analysis" in analysis
        assert "per_episode_rnd" in analysis
        assert "degenerate_episodes" in analysis

    def test_empty_events(self) -> None:
        analysis = build_analysis([])
        assert analysis["config"]["game"] == "unknown"
        assert analysis["episode_stats"]["count"] == 0
        assert analysis["step_stats"]["count"] == 0
        assert analysis["coverage"]["count"] == 0

    def test_config_extraction(self) -> None:
        events = [_make_config_event(policy="mlp", reward_mode="survival")]
        analysis = build_analysis(events)
        assert analysis["config"]["policy"] == "mlp"
        assert analysis["config"]["reward_mode"] == "survival"

    def test_degenerate_episodes_detected(self, sample_events: list[dict]) -> None:
        analysis = build_analysis(sample_events)
        # Episodes 2 (2000 steps, truncated) and 4 (10000, truncated) are degenerate
        assert len(analysis["degenerate_episodes"]) == 2


# ---------------------------------------------------------------------------
# format_report
# ---------------------------------------------------------------------------


class TestFormatReport:
    """Tests for format_report."""

    def test_contains_sections(self, sample_events: list[dict]) -> None:
        analysis = build_analysis(sample_events)
        report = format_report(analysis)
        assert "TRAINING ANALYSIS REPORT" in report
        assert "Configuration:" in report
        assert "Episode Statistics:" in report
        assert "Step Statistics:" in report
        assert "State Coverage:" in report
        assert "Paddle Movement Analysis:" in report

    def test_empty_analysis_report(self) -> None:
        analysis = build_analysis([])
        report = format_report(analysis)
        assert "TRAINING ANALYSIS REPORT" in report
        assert "Total episodes:  0" in report

    def test_degenerate_section(self, sample_events: list[dict]) -> None:
        analysis = build_analysis(sample_events)
        report = format_report(analysis)
        assert "Degenerate Episodes" in report

    def test_rnd_collapse_section(self, sample_events: list[dict]) -> None:
        analysis = build_analysis(sample_events)
        report = format_report(analysis)
        assert "RND Collapse Analysis:" in report


# ---------------------------------------------------------------------------
# _describe helper
# ---------------------------------------------------------------------------


class TestDescribe:
    """Tests for _describe helper."""

    def test_single_value(self) -> None:
        from scripts.analyze_training import _describe

        result = _describe([5.0])
        assert result["count"] == 1
        assert result["mean"] == 5.0
        assert result["median"] == 5.0
        assert result["min"] == 5.0
        assert result["max"] == 5.0
        assert result["std"] == 0.0

    def test_even_count_median(self) -> None:
        from scripts.analyze_training import _describe

        result = _describe([1.0, 2.0, 3.0, 4.0])
        assert result["median"] == 2.5  # (2 + 3) / 2

    def test_empty_list(self) -> None:
        from scripts.analyze_training import _describe

        assert _describe([]) == {}


# ---------------------------------------------------------------------------
# CLI (main)
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for main CLI entry point."""

    def test_console_report(self, sample_jsonl: Path, capsys: pytest.CaptureFixture) -> None:
        main([str(sample_jsonl)])
        captured = capsys.readouterr()
        assert "TRAINING ANALYSIS REPORT" in captured.out
        assert "breakout71" in captured.out

    def test_json_output(self, sample_jsonl: Path, capsys: pytest.CaptureFixture) -> None:
        main([str(sample_jsonl), "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["config"]["game"] == "breakout71"
        assert data["episode_stats"]["count"] == 5

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            main([str(tmp_path / "missing.jsonl")])

    def test_empty_file_exits(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        with pytest.raises(SystemExit):
            main([str(path)])

    def test_top_episodes_arg(self, sample_jsonl: Path, capsys: pytest.CaptureFixture) -> None:
        # Just verify the arg is accepted without error
        main([str(sample_jsonl), "--top-episodes", "3"])
        captured = capsys.readouterr()
        assert "TRAINING ANALYSIS REPORT" in captured.out
