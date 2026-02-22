"""Tests for analyze_demo -- knowledge extraction from human demo recordings.

Covers:
- JSONL + manifest parsing
- Macro-action clustering from raw events
- Action frequency analysis
- Spatial heatmap generation
- Game state correlation (actions at states)
- Build order extraction (temporal macro-action sequences)
- Reward candidate identification (game state metric changes)
- Console report formatting
- CLI integration
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# -- Test data helpers --------------------------------------------------------


def _make_step(
    step: int = 0,
    episode_id: int = 0,
    action: int | float | list = 0,
    reward: float = 0.01,
    terminated: bool = False,
    truncated: bool = False,
    human_events: list | None = None,
    game_state: dict | None = None,
    obs_hash: str | None = None,
    frame_file: str | None = None,
    oracle_findings: list | None = None,
    timestamp: str = "2026-02-22T12:00:00+00:00",
) -> dict:
    """Create a single JSONL step record matching DemoRecorder output."""
    return {
        "step": step,
        "episode_id": episode_id,
        "timestamp": timestamp,
        "action": action,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "human_events": human_events if human_events is not None else [],
        "game_state": game_state if game_state is not None else {},
        "obs_hash": obs_hash,
        "frame_file": frame_file,
        "oracle_findings": oracle_findings if oracle_findings is not None else [],
    }


def _make_manifest(
    game_name: str = "test_game",
    total_steps: int = 10,
    total_episodes: int = 1,
    episodes: list | None = None,
) -> dict:
    """Create a manifest matching DemoRecorder output."""
    return {
        "game_name": game_name,
        "total_steps": total_steps,
        "total_episodes": total_episodes,
        "frame_capture_interval": 1,
        "episodes": episodes
        if episodes is not None
        else [
            {
                "episode_id": 0,
                "steps": total_steps,
                "total_reward": 0.1,
                "terminated": True,
                "truncated": False,
            }
        ],
    }


def _write_demo(tmp_path: Path, steps: list[dict], manifest: dict) -> Path:
    """Write a demo recording to disk (JSONL + manifest)."""
    demo_dir = tmp_path / "demo_test"
    demo_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = demo_dir / "demo.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for step_record in steps:
            f.write(json.dumps(step_record) + "\n")

    manifest_path = demo_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    return demo_dir


# ===========================================================================
# Parsing Tests
# ===========================================================================


class TestDemoParsing:
    """Test JSONL + manifest parsing."""

    def test_load_demo_reads_jsonl(self, tmp_path):
        """load_demo() parses JSONL records into list of dicts."""
        from scripts.analyze_demo import load_demo

        steps = [_make_step(step=i) for i in range(5)]
        demo_dir = _write_demo(tmp_path, steps, _make_manifest(total_steps=5))
        result = load_demo(demo_dir)
        assert len(result["steps"]) == 5

    def test_load_demo_reads_manifest(self, tmp_path):
        """load_demo() reads the manifest.json file."""
        from scripts.analyze_demo import load_demo

        manifest = _make_manifest(game_name="hextris", total_steps=3)
        steps = [_make_step(step=i) for i in range(3)]
        demo_dir = _write_demo(tmp_path, steps, manifest)
        result = load_demo(demo_dir)
        assert result["manifest"]["game_name"] == "hextris"

    def test_load_demo_missing_jsonl_raises(self, tmp_path):
        """load_demo() raises FileNotFoundError for missing demo.jsonl."""
        from scripts.analyze_demo import load_demo

        demo_dir = tmp_path / "demo_empty"
        demo_dir.mkdir()
        manifest_path = demo_dir / "manifest.json"
        manifest_path.write_text("{}")
        with pytest.raises(FileNotFoundError, match="demo.jsonl"):
            load_demo(demo_dir)

    def test_load_demo_missing_manifest_raises(self, tmp_path):
        """load_demo() raises FileNotFoundError for missing manifest.json."""
        from scripts.analyze_demo import load_demo

        demo_dir = tmp_path / "demo_empty"
        demo_dir.mkdir()
        jsonl_path = demo_dir / "demo.jsonl"
        jsonl_path.write_text("")
        with pytest.raises(FileNotFoundError, match="manifest.json"):
            load_demo(demo_dir)

    def test_load_demo_skips_malformed_lines(self, tmp_path, capsys):
        """load_demo() skips malformed JSONL lines with a warning to stderr."""
        from scripts.analyze_demo import load_demo

        demo_dir = tmp_path / "demo_bad"
        demo_dir.mkdir()
        jsonl_path = demo_dir / "demo.jsonl"
        jsonl_path.write_text(
            json.dumps(_make_step(step=0))
            + "\n"
            + "not json\n"
            + json.dumps(_make_step(step=1))
            + "\n"
        )
        manifest_path = demo_dir / "manifest.json"
        manifest_path.write_text(json.dumps(_make_manifest(total_steps=2)))
        result = load_demo(demo_dir)
        assert len(result["steps"]) == 2
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "malformed" in captured.err

    def test_load_demo_groups_by_episode(self, tmp_path):
        """load_demo() groups steps by episode_id."""
        from scripts.analyze_demo import load_demo

        steps = [
            _make_step(step=0, episode_id=0),
            _make_step(step=1, episode_id=0),
            _make_step(step=0, episode_id=1),
        ]
        manifest = _make_manifest(
            total_steps=3,
            total_episodes=2,
            episodes=[
                {
                    "episode_id": 0,
                    "steps": 2,
                    "total_reward": 0.02,
                    "terminated": True,
                    "truncated": False,
                },
                {
                    "episode_id": 1,
                    "steps": 1,
                    "total_reward": 0.01,
                    "terminated": True,
                    "truncated": False,
                },
            ],
        )
        demo_dir = _write_demo(tmp_path, steps, manifest)
        result = load_demo(demo_dir)
        assert len(result["episodes"][0]) == 2
        assert len(result["episodes"][1]) == 1


# ===========================================================================
# Macro-Action Clustering Tests
# ===========================================================================


class TestMacroActionClustering:
    """Test grouping raw input events into semantic macro-actions."""

    def test_cluster_click_events(self):
        """Mousedown+mouseup at same position clusters into a 'click'."""
        from scripts.analyze_demo import cluster_macro_actions

        events = [
            {"type": "mousedown", "x": 400, "y": 300, "t": 1000, "button": 0},
            {"type": "mouseup", "x": 400, "y": 300, "t": 1050, "button": 0},
        ]
        macros = cluster_macro_actions(events)
        assert len(macros) == 1
        assert macros[0]["action"] == "click"
        assert macros[0]["x"] == 400
        assert macros[0]["y"] == 300

    def test_cluster_drag_events(self):
        """Mousedown + mousemove + mouseup clusters into a 'drag'."""
        from scripts.analyze_demo import cluster_macro_actions

        events = [
            {"type": "mousedown", "x": 100, "y": 100, "t": 1000, "button": 0},
            {"type": "mousemove", "x": 200, "y": 200, "t": 1050},
            {"type": "mousemove", "x": 300, "y": 300, "t": 1100},
            {"type": "mouseup", "x": 300, "y": 300, "t": 1150, "button": 0},
        ]
        macros = cluster_macro_actions(events)
        assert len(macros) == 1
        assert macros[0]["action"] == "drag"
        assert macros[0]["start_x"] == 100
        assert macros[0]["end_x"] == 300

    def test_cluster_keypress(self):
        """Keydown + keyup for same key clusters into a 'keypress'."""
        from scripts.analyze_demo import cluster_macro_actions

        events = [
            {"type": "keydown", "key": "1", "t": 1000},
            {"type": "keyup", "key": "1", "t": 1100},
        ]
        macros = cluster_macro_actions(events)
        assert len(macros) == 1
        assert macros[0]["action"] == "keypress"
        assert macros[0]["key"] == "1"

    def test_cluster_key_hold(self):
        """Keydown without keyup within threshold is a 'key_hold'."""
        from scripts.analyze_demo import cluster_macro_actions

        events = [
            {"type": "keydown", "key": "ArrowLeft", "t": 1000},
            {"type": "keyup", "key": "ArrowLeft", "t": 2500},
        ]
        macros = cluster_macro_actions(events, hold_threshold_ms=500)
        assert len(macros) == 1
        assert macros[0]["action"] == "key_hold"
        assert macros[0]["duration_ms"] == 1500

    def test_cluster_mousemove_only(self):
        """Standalone mousemove events cluster into 'hover' actions."""
        from scripts.analyze_demo import cluster_macro_actions

        events = [
            {"type": "mousemove", "x": 100, "y": 100, "t": 1000},
            {"type": "mousemove", "x": 200, "y": 200, "t": 1100},
        ]
        macros = cluster_macro_actions(events)
        # Standalone mousemoves should be grouped into hover/move
        assert len(macros) >= 1
        assert all(m["action"] == "hover" for m in macros)

    def test_cluster_empty_events(self):
        """Empty event list returns empty macro list."""
        from scripts.analyze_demo import cluster_macro_actions

        macros = cluster_macro_actions([])
        assert macros == []

    def test_cluster_wheel_event(self):
        """Wheel events cluster into 'scroll' macro-actions."""
        from scripts.analyze_demo import cluster_macro_actions

        events = [
            {"type": "wheel", "x": 400, "y": 300, "deltaY": -120, "t": 1000},
        ]
        macros = cluster_macro_actions(events)
        assert len(macros) == 1
        assert macros[0]["action"] == "scroll"

    def test_cluster_pending_key_has_duration_ms(self):
        """Pending keydown (no keyup) is flushed with duration_ms using stream end time."""
        from scripts.analyze_demo import cluster_macro_actions

        events = [
            {"type": "keydown", "key": "a", "t": 1000},
            {"type": "keydown", "key": "b", "t": 2000},
            # No keyup for either key; last event timestamp is 2000
        ]
        macros = cluster_macro_actions(events)
        assert len(macros) == 2
        for m in macros:
            assert "duration_ms" in m
            assert m["action"] == "keypress"
        # Key "a" pressed at 1000, stream ends at 2000 → duration_ms=1000
        key_a = next(m for m in macros if m["key"] == "a")
        assert key_a["duration_ms"] == 1000
        # Key "b" pressed at 2000, stream ends at 2000 → duration_ms=0
        key_b = next(m for m in macros if m["key"] == "b")
        assert key_b["duration_ms"] == 0

    def test_cluster_pending_mousedown_flushed(self):
        """Pending mousedown (no mouseup) is flushed as a click action."""
        from scripts.analyze_demo import cluster_macro_actions

        events = [
            {"type": "mousedown", "x": 300, "y": 400, "t": 1000, "button": 0},
            # No mouseup event
        ]
        macros = cluster_macro_actions(events)
        assert len(macros) == 1
        assert macros[0]["action"] == "click"
        assert macros[0]["x"] == 300
        assert macros[0]["y"] == 400

    def test_cluster_across_steps(self):
        """cluster_macro_actions_from_demo groups events across all steps."""
        from scripts.analyze_demo import cluster_macro_actions_from_demo

        steps = [
            _make_step(
                step=0,
                human_events=[
                    {"type": "keydown", "key": "1", "t": 1000},
                    {"type": "keyup", "key": "1", "t": 1100},
                ],
            ),
            _make_step(
                step=1,
                human_events=[
                    {"type": "mousedown", "x": 400, "y": 300, "t": 2000, "button": 0},
                    {"type": "mouseup", "x": 400, "y": 300, "t": 2050, "button": 0},
                ],
            ),
        ]
        macros = cluster_macro_actions_from_demo(steps)
        assert len(macros) == 2


# ===========================================================================
# Action Frequency Analysis Tests
# ===========================================================================


class TestActionFrequency:
    """Test action frequency analysis."""

    def test_frequency_counts_macro_types(self):
        """analyze_action_frequency() counts macro-action types."""
        from scripts.analyze_demo import analyze_action_frequency

        macros = [
            {"action": "click", "x": 100, "y": 100, "t": 1000},
            {"action": "click", "x": 200, "y": 200, "t": 2000},
            {"action": "keypress", "key": "1", "t": 3000},
        ]
        freq = analyze_action_frequency(macros)
        assert freq["click"] == 2
        assert freq["keypress"] == 1

    def test_frequency_empty_macros(self):
        """analyze_action_frequency() returns empty dict for no macros."""
        from scripts.analyze_demo import analyze_action_frequency

        freq = analyze_action_frequency([])
        assert freq == {}

    def test_frequency_key_breakdown(self):
        """analyze_key_frequency() provides per-key counts."""
        from scripts.analyze_demo import analyze_key_frequency

        macros = [
            {"action": "keypress", "key": "1", "t": 1000},
            {"action": "keypress", "key": "1", "t": 2000},
            {"action": "keypress", "key": "2", "t": 3000},
            {"action": "click", "x": 100, "y": 100, "t": 4000},
        ]
        key_freq = analyze_key_frequency(macros)
        assert key_freq["1"] == 2
        assert key_freq["2"] == 1
        assert "click" not in key_freq


# ===========================================================================
# Spatial Heatmap Tests
# ===========================================================================


class TestSpatialHeatmap:
    """Test spatial heatmap generation from click/drag positions."""

    def test_heatmap_from_clicks(self):
        """generate_spatial_heatmap() creates grid from click positions."""
        from scripts.analyze_demo import generate_spatial_heatmap

        macros = [
            {"action": "click", "x": 100, "y": 100, "t": 1000},
            {"action": "click", "x": 100, "y": 100, "t": 2000},
            {"action": "click", "x": 500, "y": 500, "t": 3000},
        ]
        heatmap = generate_spatial_heatmap(macros, grid_size=10, width=1000, height=1000)
        # 100,100 is in cell (1,1); 500,500 is in cell (5,5)
        assert heatmap[1][1] == 2
        assert heatmap[5][5] == 1

    def test_heatmap_empty(self):
        """generate_spatial_heatmap() returns zero grid for no spatial events."""
        from scripts.analyze_demo import generate_spatial_heatmap

        heatmap = generate_spatial_heatmap([], grid_size=5, width=500, height=500)
        total = sum(sum(row) for row in heatmap)
        assert total == 0

    def test_heatmap_includes_drag_endpoints(self):
        """generate_spatial_heatmap() includes drag start and end points."""
        from scripts.analyze_demo import generate_spatial_heatmap

        macros = [
            {"action": "drag", "start_x": 0, "start_y": 0, "end_x": 999, "end_y": 999, "t": 1000},
        ]
        heatmap = generate_spatial_heatmap(macros, grid_size=10, width=1000, height=1000)
        assert heatmap[0][0] >= 1  # start
        assert heatmap[9][9] >= 1  # end

    def test_heatmap_clamps_out_of_bounds(self):
        """generate_spatial_heatmap() clamps coordinates to grid bounds."""
        from scripts.analyze_demo import generate_spatial_heatmap

        macros = [
            {"action": "click", "x": -50, "y": -50, "t": 1000},
            {"action": "click", "x": 9999, "y": 9999, "t": 2000},
        ]
        heatmap = generate_spatial_heatmap(macros, grid_size=10, width=1000, height=1000)
        assert heatmap[0][0] >= 1  # clamped to (0,0)
        assert heatmap[9][9] >= 1  # clamped to (9,9)


# ===========================================================================
# Game State Correlation Tests
# ===========================================================================


class TestGameStateCorrelation:
    """Test correlating actions with game states."""

    def test_correlate_actions_to_states(self):
        """correlate_actions_to_states() maps state keys to action distributions."""
        from scripts.analyze_demo import correlate_actions_to_states

        steps = [
            _make_step(
                step=0,
                game_state={"level": 1, "score": 0},
                human_events=[
                    {"type": "keydown", "key": "1", "t": 1000},
                    {"type": "keyup", "key": "1", "t": 1100},
                ],
            ),
            _make_step(
                step=1,
                game_state={"level": 1, "score": 100},
                human_events=[
                    {"type": "mousedown", "x": 400, "y": 300, "t": 2000, "button": 0},
                    {"type": "mouseup", "x": 400, "y": 300, "t": 2050, "button": 0},
                ],
            ),
            _make_step(
                step=2,
                game_state={"level": 2, "score": 200},
                human_events=[
                    {"type": "keydown", "key": "1", "t": 3000},
                    {"type": "keyup", "key": "1", "t": 3100},
                ],
            ),
        ]
        corr = correlate_actions_to_states(steps, state_key="level")
        # Level 1 should have keypress + click, level 2 should have keypress
        assert "keypress" in corr[1]
        assert "click" in corr[1]
        assert "keypress" in corr[2]

    def test_correlate_empty_steps(self):
        """correlate_actions_to_states() returns empty dict for no steps."""
        from scripts.analyze_demo import correlate_actions_to_states

        corr = correlate_actions_to_states([], state_key="level")
        assert corr == {}

    def test_correlate_missing_state_key(self):
        """correlate_actions_to_states() skips steps without the state key."""
        from scripts.analyze_demo import correlate_actions_to_states

        steps = [
            _make_step(
                step=0,
                game_state={"score": 100},
                human_events=[
                    {"type": "keydown", "key": "1", "t": 1000},
                    {"type": "keyup", "key": "1", "t": 1100},
                ],
            ),
        ]
        corr = correlate_actions_to_states(steps, state_key="level")
        assert corr == {}


# ===========================================================================
# Build Order Extraction Tests
# ===========================================================================


class TestBuildOrderExtraction:
    """Test temporal macro-action sequence extraction."""

    def test_extract_build_order(self):
        """extract_build_order() returns ordered list of macro-actions with timestamps."""
        from scripts.analyze_demo import extract_build_order

        macros = [
            {"action": "keypress", "key": "1", "t": 1000},
            {"action": "click", "x": 400, "y": 300, "t": 2000},
            {"action": "keypress", "key": "2", "t": 3000},
            {"action": "click", "x": 500, "y": 400, "t": 4000},
        ]
        order = extract_build_order(macros)
        assert len(order) == 4
        assert order[0]["action"] == "keypress"
        assert order[0]["t"] == 1000
        assert order[3]["t"] == 4000

    def test_extract_build_order_empty(self):
        """extract_build_order() returns empty list for no macros."""
        from scripts.analyze_demo import extract_build_order

        order = extract_build_order([])
        assert order == []

    def test_extract_build_order_sorted(self):
        """extract_build_order() sorts by timestamp."""
        from scripts.analyze_demo import extract_build_order

        macros = [
            {"action": "click", "x": 100, "y": 100, "t": 3000},
            {"action": "keypress", "key": "1", "t": 1000},
        ]
        order = extract_build_order(macros)
        assert order[0]["t"] == 1000
        assert order[1]["t"] == 3000

    def test_extract_build_order_phases(self):
        """extract_build_order_phases() groups actions into temporal phases."""
        from scripts.analyze_demo import extract_build_order_phases

        macros = [
            {"action": "keypress", "key": "1", "t": 1000},
            {"action": "click", "x": 400, "y": 300, "t": 1500},
            # 5-second gap
            {"action": "keypress", "key": "2", "t": 7000},
            {"action": "click", "x": 500, "y": 400, "t": 7500},
        ]
        phases = extract_build_order_phases(macros, gap_threshold_ms=3000)
        assert len(phases) == 2
        assert len(phases[0]) == 2
        assert len(phases[1]) == 2


# ===========================================================================
# Reward Candidate Identification Tests
# ===========================================================================


class TestRewardCandidates:
    """Test identification of game state metrics that change during play."""

    def test_identify_changing_metrics(self):
        """identify_reward_candidates() finds metrics that change over time."""
        from scripts.analyze_demo import identify_reward_candidates

        steps = [
            _make_step(step=0, game_state={"score": 0, "level": 1, "buildings": 0}),
            _make_step(step=1, game_state={"score": 10, "level": 1, "buildings": 1}),
            _make_step(step=2, game_state={"score": 25, "level": 1, "buildings": 2}),
            _make_step(step=3, game_state={"score": 50, "level": 2, "buildings": 3}),
        ]
        candidates = identify_reward_candidates(steps)
        # score, level, buildings all change
        assert "score" in candidates
        assert "level" in candidates
        assert "buildings" in candidates

    def test_identify_static_metrics_excluded(self):
        """identify_reward_candidates() excludes metrics that never change."""
        from scripts.analyze_demo import identify_reward_candidates

        steps = [
            _make_step(step=0, game_state={"score": 0, "constant": 42}),
            _make_step(step=1, game_state={"score": 10, "constant": 42}),
            _make_step(step=2, game_state={"score": 20, "constant": 42}),
        ]
        candidates = identify_reward_candidates(steps)
        assert "score" in candidates
        assert "constant" not in candidates

    def test_reward_candidate_stats(self):
        """identify_reward_candidates() returns delta stats per metric."""
        from scripts.analyze_demo import identify_reward_candidates

        steps = [
            _make_step(step=0, game_state={"score": 0}),
            _make_step(step=1, game_state={"score": 10}),
            _make_step(step=2, game_state={"score": 25}),
        ]
        candidates = identify_reward_candidates(steps)
        # score deltas: 10, 15
        assert candidates["score"]["total_delta"] == 25
        assert candidates["score"]["num_changes"] == 2
        assert candidates["score"]["mean_delta"] == 12.5

    def test_reward_candidates_empty(self):
        """identify_reward_candidates() returns empty for no steps."""
        from scripts.analyze_demo import identify_reward_candidates

        candidates = identify_reward_candidates([])
        assert candidates == {}

    def test_reward_candidates_non_numeric_excluded(self):
        """identify_reward_candidates() excludes non-numeric metrics."""
        from scripts.analyze_demo import identify_reward_candidates

        steps = [
            _make_step(step=0, game_state={"score": 0, "state": "playing"}),
            _make_step(step=1, game_state={"score": 10, "state": "playing"}),
            _make_step(step=2, game_state={"score": 20, "state": "paused"}),
        ]
        candidates = identify_reward_candidates(steps)
        assert "score" in candidates
        assert "state" not in candidates


# ===========================================================================
# Report Formatting Tests
# ===========================================================================


class TestReportFormatting:
    """Test console report generation."""

    def test_format_report_returns_string(self, tmp_path):
        """format_report() returns a non-empty string."""
        from scripts.analyze_demo import format_report, load_demo

        steps = [
            _make_step(
                step=0,
                human_events=[
                    {"type": "mousedown", "x": 100, "y": 100, "t": 1000, "button": 0},
                    {"type": "mouseup", "x": 100, "y": 100, "t": 1050, "button": 0},
                ],
                game_state={"score": 0},
            ),
            _make_step(
                step=1,
                human_events=[],
                game_state={"score": 10},
            ),
        ]
        demo_dir = _write_demo(tmp_path, steps, _make_manifest(total_steps=2))
        demo = load_demo(demo_dir)
        report = format_report(demo)
        assert isinstance(report, str)
        assert len(report) > 0

    def test_format_report_includes_sections(self, tmp_path):
        """format_report() includes key analysis sections."""
        from scripts.analyze_demo import format_report, load_demo

        steps = [
            _make_step(
                step=0,
                human_events=[
                    {"type": "keydown", "key": "1", "t": 1000},
                    {"type": "keyup", "key": "1", "t": 1100},
                ],
                game_state={"score": 0},
            ),
            _make_step(
                step=1,
                human_events=[],
                game_state={"score": 10},
            ),
        ]
        demo_dir = _write_demo(tmp_path, steps, _make_manifest(total_steps=2))
        demo = load_demo(demo_dir)
        report = format_report(demo)
        assert "Macro-Action" in report or "macro" in report.lower()
        assert "Frequency" in report or "frequency" in report.lower()
        assert "Reward" in report or "reward" in report.lower()

    def test_format_report_uses_custom_grid_params(self, tmp_path):
        """format_report() forwards custom grid/screen params to analysis."""
        from scripts.analyze_demo import format_report, load_demo

        steps = [
            _make_step(
                step=0,
                human_events=[
                    {"type": "mousedown", "x": 100, "y": 100, "t": 1000, "button": 0},
                    {"type": "mouseup", "x": 100, "y": 100, "t": 1050, "button": 0},
                ],
            ),
        ]
        demo_dir = _write_demo(tmp_path, steps, _make_manifest(total_steps=1))
        demo = load_demo(demo_dir)
        # With grid_size=5 and width=500, click at (100,100) is cell (1,1)
        # The report should use the custom grid size
        report = format_report(demo, grid_size=5, screen_width=500, screen_height=500)
        assert "Spatial Heatmap" in report


# ===========================================================================
# JSON Output Tests
# ===========================================================================


class TestJsonOutput:
    """Test JSON output mode."""

    def test_analyze_returns_dict(self, tmp_path):
        """analyze_demo() returns a structured dict with all analysis results."""
        from scripts.analyze_demo import analyze_demo, load_demo

        steps = [
            _make_step(
                step=0,
                human_events=[
                    {"type": "mousedown", "x": 100, "y": 100, "t": 1000, "button": 0},
                    {"type": "mouseup", "x": 100, "y": 100, "t": 1050, "button": 0},
                ],
                game_state={"score": 0},
            ),
            _make_step(
                step=1,
                human_events=[],
                game_state={"score": 10},
            ),
        ]
        demo_dir = _write_demo(tmp_path, steps, _make_manifest(total_steps=2))
        demo = load_demo(demo_dir)
        result = analyze_demo(demo)
        assert "macro_actions" in result
        assert "action_frequency" in result
        assert "reward_candidates" in result
        assert "build_order" in result


# ===========================================================================
# CLI Integration Tests
# ===========================================================================


class TestCLI:
    """Test CLI argument parsing and entry point."""

    def test_parse_args_demo_dir(self):
        """parse_args() accepts a demo directory path."""
        from scripts.analyze_demo import parse_args

        args = parse_args(["some/path/demo_dir"])
        assert args.demo_dir == "some/path/demo_dir"

    def test_parse_args_json_flag(self):
        """parse_args() accepts --json flag."""
        from scripts.analyze_demo import parse_args

        args = parse_args(["some/path", "--json"])
        assert args.json is True

    def test_parse_args_default_no_json(self):
        """parse_args() defaults to json=False."""
        from scripts.analyze_demo import parse_args

        args = parse_args(["some/path"])
        assert args.json is False

    def test_parse_args_grid_size(self):
        """parse_args() accepts --grid-size option."""
        from scripts.analyze_demo import parse_args

        args = parse_args(["some/path", "--grid-size", "20"])
        assert args.grid_size == 20

    def test_parse_args_screen_size(self):
        """parse_args() accepts --screen-width and --screen-height."""
        from scripts.analyze_demo import parse_args

        args = parse_args(["some/path", "--screen-width", "1920", "--screen-height", "1080"])
        assert args.screen_width == 1920
        assert args.screen_height == 1080
