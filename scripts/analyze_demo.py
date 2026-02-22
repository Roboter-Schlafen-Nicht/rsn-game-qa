#!/usr/bin/env python
"""Analyze human demo recordings and extract structured knowledge.

Post-processes demo recordings produced by ``DemoRecorder`` (JSONL +
manifest format) to extract actionable knowledge for RL training:

- **Macro-action clustering**: groups raw JS input events into semantic
  actions (click, drag, keypress, key_hold, hover, scroll).
- **Action frequency analysis**: counts macro-action types and per-key
  usage.
- **Spatial heatmap**: 2D grid of click/drag positions showing where
  the human acts on screen.
- **Game state correlation**: maps game state values to action
  distributions (e.g., what actions at each level).
- **Build order extraction**: temporal sequence of macro-actions with
  optional phase grouping by time gaps.
- **Reward candidate identification**: finds numeric game state metrics
  that change during play, with delta statistics.

Usage::

    python scripts/analyze_demo.py /path/to/demo_dir
    python scripts/analyze_demo.py /path/to/demo_dir --json
    python scripts/analyze_demo.py /path/to/demo_dir --grid-size 20
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def load_demo(demo_dir: str | Path) -> dict:
    """Load a demo recording from disk.

    Reads the JSONL step records and ``manifest.json`` from *demo_dir*,
    groups steps by ``episode_id``, and returns a structured dict.

    Parameters
    ----------
    demo_dir : str | Path
        Path to the demo recording directory containing ``demo.jsonl``
        and ``manifest.json``.

    Returns
    -------
    dict
        Keys: ``steps`` (list of all step dicts), ``manifest`` (parsed
        manifest dict), ``episodes`` (dict mapping episode_id to list
        of steps for that episode).

    Raises
    ------
    FileNotFoundError
        If ``demo.jsonl`` or ``manifest.json`` is missing.
    """
    demo_dir = Path(demo_dir)

    jsonl_path = demo_dir / "demo.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"demo.jsonl not found in {demo_dir}")

    manifest_path = demo_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {demo_dir}")

    # Parse JSONL
    steps: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                steps.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed lines gracefully
                pass

    # Parse manifest
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    # Group by episode_id
    episodes: dict[int, list[dict]] = defaultdict(list)
    for step in steps:
        ep_id = step.get("episode_id", 0)
        episodes[ep_id].append(step)

    return {
        "steps": steps,
        "manifest": manifest,
        "episodes": dict(episodes),
    }


# ---------------------------------------------------------------------------
# Macro-Action Clustering
# ---------------------------------------------------------------------------


def cluster_macro_actions(
    events: list[dict],
    hold_threshold_ms: float = 500,
) -> list[dict]:
    """Cluster raw JS input events into semantic macro-actions.

    Groups raw events (mousemove, mousedown, mouseup, click, keydown,
    keyup, wheel) into higher-level actions:

    - **click**: mousedown + mouseup at same position (no intervening
      mousemove with mouse held).
    - **drag**: mousedown + mousemove(s) + mouseup at different position.
    - **keypress**: keydown + keyup for same key within hold threshold.
    - **key_hold**: keydown + keyup for same key exceeding hold threshold.
    - **hover**: standalone mousemove events (mouse not held down).
    - **scroll**: wheel events.

    Parameters
    ----------
    events : list[dict]
        Raw JS events with ``type``, ``t`` (timestamp), and
        type-specific fields (``x``, ``y``, ``key``, ``button``,
        ``deltaY``).
    hold_threshold_ms : float
        Duration in ms above which a keydown→keyup pair is classified
        as ``key_hold`` instead of ``keypress``.

    Returns
    -------
    list[dict]
        List of macro-action dicts, each with an ``action`` field and
        action-specific metadata.
    """
    if not events:
        return []

    macros: list[dict] = []

    # State tracking for mouse
    mouse_down: dict | None = None
    mouse_moved_during_drag = False

    # State tracking for keys: key -> keydown event
    pending_keys: dict[str, dict] = {}

    # Standalone mousemoves (not during drag)
    standalone_moves: list[dict] = []

    for event in events:
        etype = event.get("type", "")

        if etype == "mousedown":
            # Flush standalone moves before new mouse interaction
            macros.extend(_flush_hovers(standalone_moves))
            standalone_moves = []
            mouse_down = event
            mouse_moved_during_drag = False

        elif etype == "mousemove":
            if mouse_down is not None:
                mouse_moved_during_drag = True
            else:
                standalone_moves.append(event)

        elif etype == "mouseup":
            if mouse_down is not None:
                if mouse_moved_during_drag:
                    macros.append(
                        {
                            "action": "drag",
                            "start_x": mouse_down.get("x", 0),
                            "start_y": mouse_down.get("y", 0),
                            "end_x": event.get("x", 0),
                            "end_y": event.get("y", 0),
                            "t": mouse_down.get("t", 0),
                            "duration_ms": event.get("t", 0) - mouse_down.get("t", 0),
                        }
                    )
                else:
                    macros.append(
                        {
                            "action": "click",
                            "x": mouse_down.get("x", 0),
                            "y": mouse_down.get("y", 0),
                            "t": mouse_down.get("t", 0),
                            "button": mouse_down.get("button", 0),
                        }
                    )
                mouse_down = None
                mouse_moved_during_drag = False

        elif etype == "click":
            # Standalone click event (if not preceded by mousedown/mouseup)
            # Only add if we didn't just process a mousedown→mouseup pair
            if mouse_down is None:
                macros.extend(_flush_hovers(standalone_moves))
                standalone_moves = []
                macros.append(
                    {
                        "action": "click",
                        "x": event.get("x", 0),
                        "y": event.get("y", 0),
                        "t": event.get("t", 0),
                        "button": event.get("button", 0),
                    }
                )

        elif etype == "keydown":
            key = event.get("key", "")
            if key not in pending_keys:
                pending_keys[key] = event

        elif etype == "keyup":
            key = event.get("key", "")
            if key in pending_keys:
                down_event = pending_keys.pop(key)
                duration = event.get("t", 0) - down_event.get("t", 0)
                if duration > hold_threshold_ms:
                    macros.append(
                        {
                            "action": "key_hold",
                            "key": key,
                            "t": down_event.get("t", 0),
                            "duration_ms": duration,
                        }
                    )
                else:
                    macros.append(
                        {
                            "action": "keypress",
                            "key": key,
                            "t": down_event.get("t", 0),
                            "duration_ms": duration,
                        }
                    )

        elif etype == "wheel":
            macros.extend(_flush_hovers(standalone_moves))
            standalone_moves = []
            macros.append(
                {
                    "action": "scroll",
                    "x": event.get("x", 0),
                    "y": event.get("y", 0),
                    "deltaY": event.get("deltaY", 0),
                    "t": event.get("t", 0),
                }
            )

    # Flush remaining standalone moves
    macros.extend(_flush_hovers(standalone_moves))

    # Flush any pending keys that never got a keyup
    for key, down_event in pending_keys.items():
        macros.append(
            {
                "action": "keypress",
                "key": key,
                "t": down_event.get("t", 0),
            }
        )

    return macros


def _flush_hovers(moves: list[dict]) -> list[dict]:
    """Convert standalone mousemove events into hover macro-actions.

    Parameters
    ----------
    moves : list[dict]
        List of mousemove events not part of a drag.

    Returns
    -------
    list[dict]
        One hover macro per mousemove event.
    """
    result = []
    for m in moves:
        result.append(
            {
                "action": "hover",
                "x": m.get("x", 0),
                "y": m.get("y", 0),
                "t": m.get("t", 0),
            }
        )
    return result


def cluster_macro_actions_from_demo(
    steps: list[dict],
    hold_threshold_ms: float = 500,
) -> list[dict]:
    """Cluster macro-actions from all human events across demo steps.

    Concatenates ``human_events`` from all steps and runs
    ``cluster_macro_actions()`` on the combined event stream.

    Parameters
    ----------
    steps : list[dict]
        Demo step records (each may contain ``human_events``).
    hold_threshold_ms : float
        Hold threshold passed to ``cluster_macro_actions()``.

    Returns
    -------
    list[dict]
        All macro-actions extracted from the demo.
    """
    all_events: list[dict] = []
    for step in steps:
        events = step.get("human_events", [])
        all_events.extend(events)
    return cluster_macro_actions(all_events, hold_threshold_ms=hold_threshold_ms)


# ---------------------------------------------------------------------------
# Action Frequency Analysis
# ---------------------------------------------------------------------------


def analyze_action_frequency(macros: list[dict]) -> dict[str, int]:
    """Count macro-action types.

    Parameters
    ----------
    macros : list[dict]
        List of macro-action dicts (each with ``action`` field).

    Returns
    -------
    dict[str, int]
        Mapping from action type to count.
    """
    freq: dict[str, int] = {}
    for m in macros:
        action = m.get("action", "unknown")
        freq[action] = freq.get(action, 0) + 1
    return freq


def analyze_key_frequency(macros: list[dict]) -> dict[str, int]:
    """Count per-key usage for keyboard macro-actions.

    Only considers ``keypress`` and ``key_hold`` macro-actions.

    Parameters
    ----------
    macros : list[dict]
        List of macro-action dicts.

    Returns
    -------
    dict[str, int]
        Mapping from key name to count.
    """
    freq: dict[str, int] = {}
    for m in macros:
        if m.get("action") in ("keypress", "key_hold"):
            key = m.get("key", "unknown")
            freq[key] = freq.get(key, 0) + 1
    return freq


# ---------------------------------------------------------------------------
# Spatial Heatmap
# ---------------------------------------------------------------------------


def generate_spatial_heatmap(
    macros: list[dict],
    grid_size: int = 10,
    width: int = 1280,
    height: int = 1024,
) -> list[list[int]]:
    """Generate a 2D spatial heatmap from click/drag positions.

    Creates a *grid_size* x *grid_size* grid and counts the number of
    spatial events (clicks, drag start/end points) in each cell.

    Parameters
    ----------
    macros : list[dict]
        List of macro-action dicts.
    grid_size : int
        Number of grid cells per axis.
    width : int
        Screen width in pixels.
    height : int
        Screen height in pixels.

    Returns
    -------
    list[list[int]]
        2D grid indexed as ``heatmap[row][col]``.
    """
    heatmap = [[0] * grid_size for _ in range(grid_size)]

    def _add_point(x: int | float, y: int | float) -> None:
        col = int(x / width * grid_size)
        row = int(y / height * grid_size)
        # Clamp to grid bounds
        col = max(0, min(col, grid_size - 1))
        row = max(0, min(row, grid_size - 1))
        heatmap[row][col] += 1

    for m in macros:
        action = m.get("action", "")
        if action == "click":
            _add_point(m.get("x", 0), m.get("y", 0))
        elif action == "drag":
            _add_point(m.get("start_x", 0), m.get("start_y", 0))
            _add_point(m.get("end_x", 0), m.get("end_y", 0))

    return heatmap


# ---------------------------------------------------------------------------
# Game State Correlation
# ---------------------------------------------------------------------------


def correlate_actions_to_states(
    steps: list[dict],
    state_key: str,
    hold_threshold_ms: float = 500,
) -> dict:
    """Map game state values to action distributions.

    For each unique value of *state_key* in the game state, counts which
    macro-action types the human performed at that state.

    Parameters
    ----------
    steps : list[dict]
        Demo step records.
    state_key : str
        The game state field to correlate on (e.g., ``"level"``).
    hold_threshold_ms : float
        Hold threshold for macro-action clustering.

    Returns
    -------
    dict
        Mapping from state value to dict of action type counts.
    """
    if not steps:
        return {}

    result: dict = {}
    for step in steps:
        game_state = step.get("game_state", {})
        if state_key not in game_state:
            continue
        state_val = game_state[state_key]
        events = step.get("human_events", [])
        if not events:
            continue
        macros = cluster_macro_actions(events, hold_threshold_ms=hold_threshold_ms)
        if state_val not in result:
            result[state_val] = {}
        for m in macros:
            action = m.get("action", "unknown")
            result[state_val][action] = result[state_val].get(action, 0) + 1

    return result


# ---------------------------------------------------------------------------
# Build Order Extraction
# ---------------------------------------------------------------------------


def extract_build_order(macros: list[dict]) -> list[dict]:
    """Extract a temporally sorted sequence of macro-actions.

    Parameters
    ----------
    macros : list[dict]
        List of macro-action dicts (each with ``t`` field).

    Returns
    -------
    list[dict]
        Macro-actions sorted by timestamp.
    """
    if not macros:
        return []
    return sorted(macros, key=lambda m: m.get("t", 0))


def extract_build_order_phases(
    macros: list[dict],
    gap_threshold_ms: float = 5000,
) -> list[list[dict]]:
    """Group macro-actions into temporal phases separated by time gaps.

    Parameters
    ----------
    macros : list[dict]
        List of macro-action dicts (each with ``t`` field).
    gap_threshold_ms : float
        Minimum gap in ms between consecutive macro-actions to start
        a new phase.

    Returns
    -------
    list[list[dict]]
        List of phases, each a list of macro-action dicts.
    """
    if not macros:
        return []

    ordered = extract_build_order(macros)
    phases: list[list[dict]] = [[ordered[0]]]

    for i in range(1, len(ordered)):
        gap = ordered[i].get("t", 0) - ordered[i - 1].get("t", 0)
        if gap >= gap_threshold_ms:
            phases.append([ordered[i]])
        else:
            phases[-1].append(ordered[i])

    return phases


# ---------------------------------------------------------------------------
# Reward Candidate Identification
# ---------------------------------------------------------------------------


def identify_reward_candidates(steps: list[dict]) -> dict:
    """Identify numeric game state metrics that change during play.

    Scans all steps for numeric fields in ``game_state`` and computes
    delta statistics for fields that change at least once.

    Parameters
    ----------
    steps : list[dict]
        Demo step records.

    Returns
    -------
    dict
        Mapping from metric name to dict with ``total_delta``,
        ``num_changes``, ``mean_delta``, ``min_delta``, ``max_delta``.
        Only includes metrics that changed at least once.
    """
    if not steps:
        return {}

    # Collect time series for each numeric metric
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for step in steps:
        game_state = step.get("game_state", {})
        step_idx = step.get("step", 0)
        for key, value in game_state.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                series[key].append((step_idx, value))

    # Compute delta stats for changing metrics
    candidates: dict[str, dict] = {}
    for key, values in series.items():
        if len(values) < 2:
            continue

        deltas: list[float] = []
        for i in range(1, len(values)):
            delta = values[i][1] - values[i - 1][1]
            if delta != 0:
                deltas.append(delta)

        if not deltas:
            continue

        total_delta = values[-1][1] - values[0][1]
        candidates[key] = {
            "total_delta": total_delta,
            "num_changes": len(deltas),
            "mean_delta": sum(abs(d) for d in deltas) / len(deltas),
            "min_delta": min(deltas),
            "max_delta": max(deltas),
        }

    return candidates


# ---------------------------------------------------------------------------
# Top-Level Analysis
# ---------------------------------------------------------------------------


def analyze_demo(demo: dict) -> dict:
    """Run all analyses on a loaded demo and return structured results.

    Parameters
    ----------
    demo : dict
        Output from ``load_demo()``.

    Returns
    -------
    dict
        Keys: ``macro_actions``, ``action_frequency``,
        ``key_frequency``, ``spatial_heatmap``, ``reward_candidates``,
        ``build_order``, ``manifest``.
    """
    steps = demo.get("steps", [])
    manifest = demo.get("manifest", {})

    macros = cluster_macro_actions_from_demo(steps)
    action_freq = analyze_action_frequency(macros)
    key_freq = analyze_key_frequency(macros)
    heatmap = generate_spatial_heatmap(macros)
    reward_cands = identify_reward_candidates(steps)
    build_order = extract_build_order(macros)

    return {
        "macro_actions": macros,
        "action_frequency": action_freq,
        "key_frequency": key_freq,
        "spatial_heatmap": heatmap,
        "reward_candidates": reward_cands,
        "build_order": build_order,
        "manifest": manifest,
    }


# ---------------------------------------------------------------------------
# Report Formatting
# ---------------------------------------------------------------------------


def format_report(demo: dict) -> str:
    """Format analysis results as a human-readable console report.

    Parameters
    ----------
    demo : dict
        Output from ``load_demo()``.

    Returns
    -------
    str
        Formatted text report.
    """
    analysis = analyze_demo(demo)
    manifest = analysis["manifest"]

    lines: list[str] = []
    sep = "=" * 60

    # Header
    lines.append(sep)
    lines.append("  DEMO ANALYSIS REPORT")
    lines.append(sep)

    # Manifest info
    lines.append("")
    lines.append("Recording Info:")
    lines.append(f"  Game:            {manifest.get('game_name', 'unknown')}")
    lines.append(f"  Total steps:     {manifest.get('total_steps', 0)}")
    lines.append(f"  Total episodes:  {manifest.get('total_episodes', 0)}")

    # Macro-action summary
    macros = analysis["macro_actions"]
    lines.append("")
    lines.append("-" * 60)
    lines.append(f"Macro-Action Summary ({len(macros)} total):")
    action_freq = analysis["action_frequency"]
    for action, count in sorted(action_freq.items(), key=lambda x: -x[1]):
        lines.append(f"  {action}: {count}")

    # Key frequency
    key_freq = analysis["key_frequency"]
    if key_freq:
        lines.append("")
        lines.append("Key Frequency:")
        for key, count in sorted(key_freq.items(), key=lambda x: -x[1]):
            lines.append(f"  {key}: {count}")

    # Spatial heatmap
    heatmap = analysis["spatial_heatmap"]
    total_spatial = sum(sum(row) for row in heatmap)
    if total_spatial > 0:
        lines.append("")
        lines.append("-" * 60)
        lines.append(f"Spatial Heatmap ({total_spatial} spatial events):")
        for row in heatmap:
            lines.append("  " + " ".join(f"{c:3d}" for c in row))

    # Reward candidates
    reward_cands = analysis["reward_candidates"]
    if reward_cands:
        lines.append("")
        lines.append("-" * 60)
        lines.append(f"Reward Candidates ({len(reward_cands)} changing metrics):")
        for metric, stats in sorted(reward_cands.items(), key=lambda x: -abs(x[1]["total_delta"])):
            lines.append(
                f"  {metric}: total_delta={stats['total_delta']:.2f}, "
                f"changes={stats['num_changes']}, "
                f"mean_delta={stats['mean_delta']:.2f}"
            )

    # Build order summary
    build_order = analysis["build_order"]
    if build_order:
        lines.append("")
        lines.append("-" * 60)
        lines.append(f"Build Order ({len(build_order)} actions):")
        # Show first 20 actions
        for m in build_order[:20]:
            extra = ""
            if m.get("key"):
                extra = f" key={m['key']}"
            elif m.get("x") is not None:
                extra = f" at ({m.get('x', 0)}, {m.get('y', 0)})"
            lines.append(f"  t={m.get('t', 0):>8}: {m['action']}{extra}")
        if len(build_order) > 20:
            lines.append(f"  ... and {len(build_order) - 20} more actions")

    lines.append("")
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments. Uses ``sys.argv[1:]`` if *None*.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Analyze human demo recordings for knowledge extraction",
    )
    parser.add_argument(
        "demo_dir",
        help="Path to the demo recording directory",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output raw JSON instead of formatted report",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=10,
        help="Grid size for spatial heatmap (default: 10)",
    )
    parser.add_argument(
        "--screen-width",
        type=int,
        default=1280,
        help="Screen width in pixels for heatmap (default: 1280)",
    )
    parser.add_argument(
        "--screen-height",
        type=int,
        default=1024,
        help="Screen height in pixels for heatmap (default: 1024)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the demo analysis CLI.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments. Uses ``sys.argv[1:]`` if *None*.
    """
    args = parse_args(argv)

    demo = load_demo(args.demo_dir)

    if args.json:
        analysis = analyze_demo(demo)
        # Override heatmap with custom grid/screen size
        macros = analysis["macro_actions"]
        analysis["spatial_heatmap"] = generate_spatial_heatmap(
            macros,
            grid_size=args.grid_size,
            width=args.screen_width,
            height=args.screen_height,
        )
        print(json.dumps(analysis, indent=2, default=str))
    else:
        print(format_report(demo))


if __name__ == "__main__":
    main()
