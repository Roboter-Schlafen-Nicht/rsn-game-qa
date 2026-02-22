#!/usr/bin/env python
"""Analyze JSONL training logs and produce a console report.

Parses training log files produced by ``train_rl.py`` and generates
summary statistics, episode metrics, RND intrinsic reward analysis,
state coverage growth, and paddle movement analysis.

Supports multiple log files (e.g. from resumed training runs) which
are concatenated in the order given.

Usage::

    python scripts/analyze_training.py /path/to/training.jsonl
    python scripts/analyze_training.py log1.jsonl log2.jsonl
    python scripts/analyze_training.py /path/to/training.jsonl --top-episodes 10
    python scripts/analyze_training.py /path/to/training.jsonl --json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def parse_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL file into a list of event dicts.

    Parameters
    ----------
    path : Path
        Path to the JSONL file.

    Returns
    -------
    list[dict]
        List of parsed event dictionaries.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Training log not found: {path}")

    events: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(
                    f"WARNING: Skipping malformed line {line_num}: {exc}",
                    file=sys.stderr,
                )
    return events


def extract_events_by_type(events: list[dict]) -> dict[str, list[dict]]:
    """Group events by their ``event`` field.

    Parameters
    ----------
    events : list[dict]
        Raw parsed events from JSONL.

    Returns
    -------
    dict[str, list[dict]]
        Mapping from event type to list of events of that type.
    """
    by_type: dict[str, list[dict]] = {}
    for event in events:
        event_type = event.get("event", "unknown")
        by_type.setdefault(event_type, []).append(event)
    return by_type


def compute_episode_stats(episodes: list[dict]) -> dict:
    """Compute aggregate statistics from episode_end events.

    Parameters
    ----------
    episodes : list[dict]
        List of ``episode_end`` event dicts.

    Returns
    -------
    dict
        Summary statistics including mean/median/min/max for steps,
        reward, and termination breakdown.
    """
    if not episodes:
        return {
            "count": 0,
            "steps": {},
            "reward": {},
            "termination": {},
            "rnd": {},
        }

    steps = [ep["steps"] for ep in episodes]
    rewards = [ep["total_reward"] for ep in episodes]

    termination_counts: dict[str, int] = {}
    for ep in episodes:
        term = ep.get("termination", "unknown")
        termination_counts[term] = termination_counts.get(term, 0) + 1

    rnd_means = [ep["rnd_intrinsic_mean"] for ep in episodes if "rnd_intrinsic_mean" in ep]

    return {
        "count": len(episodes),
        "steps": _describe(steps),
        "reward": _describe(rewards),
        "termination": termination_counts,
        "rnd": _describe(rnd_means) if rnd_means else {},
    }


def compute_step_stats(steps: list[dict]) -> dict:
    """Compute aggregate statistics from step_summary events.

    Parameters
    ----------
    steps : list[dict]
        List of ``step_summary`` event dicts.

    Returns
    -------
    dict
        Summary statistics for rewards, RND, FPS, and paddle position.
    """
    if not steps:
        return {"count": 0, "reward": {}, "rnd": {}, "fps": {}, "paddle": {}}

    rewards = [s["reward"] for s in steps]
    fps_vals = [s["fps"] for s in steps if "fps" in s]

    rnd_raw = [s["rnd_intrinsic_raw"] for s in steps if "rnd_intrinsic_raw" in s]
    rnd_norm = [s["rnd_intrinsic_norm"] for s in steps if "rnd_intrinsic_norm" in s]

    paddle_positions = [
        s["paddle_x"] for s in steps if "paddle_x" in s and s["paddle_x"] is not None
    ]

    return {
        "count": len(steps),
        "total_steps": steps[-1]["step"] if steps else 0,
        "reward": _describe(rewards),
        "rnd_raw": _describe(rnd_raw) if rnd_raw else {},
        "rnd_norm": _describe(rnd_norm) if rnd_norm else {},
        "fps": _describe(fps_vals) if fps_vals else {},
        "paddle": _describe_paddle(paddle_positions) if paddle_positions else {},
    }


def compute_coverage_stats(coverage_events: list[dict]) -> dict:
    """Compute state coverage growth statistics.

    Parameters
    ----------
    coverage_events : list[dict]
        List of ``coverage_summary`` event dicts.

    Returns
    -------
    dict
        Coverage statistics including growth rate.
    """
    if not coverage_events:
        return {"count": 0}

    entries = [{"step": c["step"], "unique_states": c["unique_states"]} for c in coverage_events]

    # Growth rate: unique states per 1K steps
    if len(entries) >= 2:
        first = entries[0]
        last = entries[-1]
        step_delta = last["step"] - first["step"]
        state_delta = last["unique_states"] - first["unique_states"]
        growth_per_1k = (state_delta / step_delta * 1000) if step_delta > 0 else 0
    elif len(entries) == 1:
        growth_per_1k = (
            entries[0]["unique_states"] / entries[0]["step"] * 1000
            if entries[0]["step"] > 0
            else 0
        )
    else:
        growth_per_1k = 0

    return {
        "count": len(entries),
        "entries": entries,
        "latest_unique_states": entries[-1]["unique_states"],
        "latest_step": entries[-1]["step"],
        "growth_per_1k_steps": round(growth_per_1k, 1),
    }


def analyze_paddle_movement(steps: list[dict]) -> dict:
    """Analyze paddle movement patterns from step_summary events.

    Detects degenerate behavior where the paddle stays at a fixed
    position for extended periods.

    Parameters
    ----------
    steps : list[dict]
        List of ``step_summary`` event dicts.

    Returns
    -------
    dict
        Paddle movement analysis including unique positions,
        most common position, and degenerate episode detection.
    """
    if not steps:
        return {"unique_positions": 0, "degenerate": False}

    positions = [s["paddle_x"] for s in steps if "paddle_x" in s and s["paddle_x"] is not None]
    if not positions:
        return {"unique_positions": 0, "degenerate": False}

    unique = set(positions)

    # Count position frequency
    freq: dict[float, int] = {}
    for p in positions:
        freq[p] = freq.get(p, 0) + 1

    most_common_pos = max(freq, key=lambda k: freq[k])
    most_common_pct = freq[most_common_pos] / len(positions) * 100

    # Degenerate = one position accounts for >80% of all samples
    degenerate = most_common_pct > 80

    return {
        "unique_positions": len(unique),
        "total_samples": len(positions),
        "most_common_position": round(most_common_pos, 4),
        "most_common_pct": round(most_common_pct, 1),
        "degenerate": degenerate,
        "position_frequency": {
            round(k, 4): v for k, v in sorted(freq.items(), key=lambda x: -x[1])[:10]
        },
    }


def analyze_per_episode_rnd(episodes: list[dict]) -> list[dict]:
    """Analyze RND intrinsic reward per episode.

    Parameters
    ----------
    episodes : list[dict]
        List of ``episode_end`` event dicts.

    Returns
    -------
    list[dict]
        Per-episode RND analysis with collapse detection.
    """
    results = []
    for ep in episodes:
        rnd_mean = ep.get("rnd_intrinsic_mean", 0)
        rnd_total = ep.get("rnd_intrinsic_total", 0)
        steps = ep.get("steps", 0)

        # RND collapsed = mean intrinsic reward < 1e-4
        collapsed = rnd_mean < 1e-4

        results.append(
            {
                "episode": ep["episode"],
                "steps": steps,
                "rnd_mean": rnd_mean,
                "rnd_total": rnd_total,
                "termination": ep.get("termination", "unknown"),
                "rnd_collapsed": collapsed,
            }
        )
    return results


def identify_degenerate_episodes(episodes: list[dict], threshold: int = 500) -> list[dict]:
    """Identify episodes with degenerate survival behavior.

    Degenerate episodes are long (>threshold steps) and end by
    truncation rather than game_over. These indicate the agent
    has found a survival exploit rather than playing the game.

    Parameters
    ----------
    episodes : list[dict]
        List of ``episode_end`` event dicts.
    threshold : int
        Minimum steps to consider an episode potentially degenerate.

    Returns
    -------
    list[dict]
        List of degenerate episode summaries.
    """
    degenerate = []
    for ep in episodes:
        steps = ep.get("steps", 0)
        term = ep.get("termination", "unknown")
        rnd_mean = ep.get("rnd_intrinsic_mean", 0)

        # Degenerate: long episode + truncated + RND collapsed
        if steps >= threshold and term == "truncated":
            degenerate.append(
                {
                    "episode": ep["episode"],
                    "steps": steps,
                    "reward": ep.get("total_reward", 0),
                    "rnd_mean": rnd_mean,
                    "termination": term,
                }
            )
    return degenerate


def build_analysis(events: list[dict]) -> dict:
    """Build the complete analysis from parsed events.

    Parameters
    ----------
    events : list[dict]
        All parsed events from the JSONL file(s).

    Returns
    -------
    dict
        Complete analysis results.
    """
    by_type = extract_events_by_type(events)

    config = by_type.get("config", [{}])[0] if by_type.get("config") else {}
    episodes = by_type.get("episode_end", [])
    steps = by_type.get("step_summary", [])
    coverage = by_type.get("coverage_summary", [])

    episode_stats = compute_episode_stats(episodes)
    step_stats = compute_step_stats(steps)
    coverage_stats = compute_coverage_stats(coverage)
    paddle_analysis = analyze_paddle_movement(steps)
    per_episode_rnd = analyze_per_episode_rnd(episodes)
    degenerate = identify_degenerate_episodes(episodes)

    return {
        "config": {
            "game": config.get("game", "unknown"),
            "policy": config.get("args", {}).get("policy", "unknown"),
            "reward_mode": config.get("args", {}).get("reward_mode", "unknown"),
            "timesteps": config.get("args", {}).get("timesteps", 0),
            "max_steps": config.get("args", {}).get("max_steps", 0),
            "survival_bonus": config.get("args", {}).get("survival_bonus"),
            "epsilon_greedy": config.get("args", {}).get("epsilon_greedy"),
        },
        "episode_stats": episode_stats,
        "step_stats": step_stats,
        "coverage": coverage_stats,
        "paddle_analysis": paddle_analysis,
        "per_episode_rnd": per_episode_rnd,
        "degenerate_episodes": degenerate,
    }


def format_report(analysis: dict, *, top_episodes: int = 5) -> str:
    """Format analysis results as a human-readable console report.

    Parameters
    ----------
    analysis : dict
        Complete analysis from ``build_analysis()``.
    top_episodes : int
        Number of longest and shortest episodes to highlight.

    Returns
    -------
    str
        Formatted text report.
    """
    lines: list[str] = []
    sep = "=" * 60

    # Header
    lines.append(sep)
    lines.append("  TRAINING ANALYSIS REPORT")
    lines.append(sep)

    # Config
    cfg = analysis["config"]
    lines.append("")
    lines.append("Configuration:")
    lines.append(f"  Game:            {cfg['game']}")
    lines.append(f"  Policy:          {cfg['policy']}")
    lines.append(f"  Reward mode:     {cfg['reward_mode']}")
    lines.append(f"  Target steps:    {cfg['timesteps']:,}")
    lines.append(f"  Max steps/ep:    {cfg['max_steps']:,}")
    if cfg.get("survival_bonus") is not None:
        lines.append(f"  Survival bonus:  {cfg['survival_bonus']}")
    if cfg.get("epsilon_greedy") is not None:
        lines.append(f"  Epsilon greedy:  {cfg['epsilon_greedy']}")

    # Episode stats
    ep = analysis["episode_stats"]
    lines.append("")
    lines.append("-" * 60)
    lines.append("Episode Statistics:")
    lines.append(f"  Total episodes:  {ep['count']}")
    if ep["count"] > 0:
        lines.append(_format_stat_block("  Steps", ep["steps"]))
        lines.append(_format_stat_block("  Reward", ep["reward"]))
        lines.append("  Termination breakdown:")
        for term, count in sorted(ep["termination"].items()):
            pct = count / ep["count"] * 100
            lines.append(f"    {term}: {count} ({pct:.1f}%)")
        if ep.get("rnd"):
            lines.append(_format_stat_block("  RND mean", ep["rnd"]))

    # Top episodes (longest and shortest)
    per_ep_rnd = analysis.get("per_episode_rnd", [])
    if per_ep_rnd and top_episodes > 0:
        lines.append("")
        lines.append("-" * 60)
        by_steps = sorted(per_ep_rnd, key=lambda e: e["steps"], reverse=True)
        n = min(top_episodes, len(by_steps))
        lines.append(f"Top {n} Longest Episodes:")
        for e in by_steps[:n]:
            lines.append(
                f"  Episode {e['episode']:>3}: {e['steps']:>6,} steps, "
                f"RND={e['rnd_mean']:.6f}, {e['termination']}"
            )
        lines.append(f"Top {n} Shortest Episodes:")
        for e in reversed(by_steps[-n:]):
            lines.append(
                f"  Episode {e['episode']:>3}: {e['steps']:>6,} steps, "
                f"RND={e['rnd_mean']:.6f}, {e['termination']}"
            )

    # Step stats
    ss = analysis["step_stats"]
    lines.append("")
    lines.append("-" * 60)
    lines.append("Step Statistics:")
    lines.append(f"  Logged steps:    {ss['count']} (every 100th)")
    lines.append(f"  Total steps:     {ss.get('total_steps', 0):,}")
    if ss.get("fps"):
        lines.append(_format_stat_block("  FPS", ss["fps"]))
    if ss.get("rnd_raw"):
        lines.append(_format_stat_block("  RND raw", ss["rnd_raw"]))
    if ss.get("rnd_norm"):
        lines.append(_format_stat_block("  RND norm", ss["rnd_norm"]))

    # Coverage
    cov = analysis["coverage"]
    if cov.get("count", 0) > 0:
        lines.append("")
        lines.append("-" * 60)
        lines.append("State Coverage:")
        lines.append(f"  Checkpoints:     {cov['count']}")
        lines.append(f"  Latest states:   {cov['latest_unique_states']:,}")
        lines.append(f"  At step:         {cov['latest_step']:,}")
        lines.append(f"  Growth rate:     {cov['growth_per_1k_steps']}/1K steps")
        if cov.get("entries"):
            lines.append("  Timeline:")
            for entry in cov["entries"]:
                lines.append(
                    f"    Step {entry['step']:>7,}: {entry['unique_states']:,} unique states"
                )

    # Paddle analysis
    pa = analysis["paddle_analysis"]
    lines.append("")
    lines.append("-" * 60)
    lines.append("Paddle Movement Analysis:")
    if pa.get("unique_positions", 0) > 0:
        lines.append(f"  Unique positions: {pa['unique_positions']}")
        lines.append(f"  Total samples:    {pa.get('total_samples', 0)}")
        lines.append(
            f"  Most common:      {pa.get('most_common_position', 0):.4f} "
            f"({pa.get('most_common_pct', 0):.1f}%)"
        )
        if pa.get("degenerate"):
            lines.append("  ** DEGENERATE: Paddle stuck at one position >80% **")
        else:
            lines.append("  Paddle movement: HEALTHY")
    else:
        lines.append("  No paddle data available")

    # Degenerate episodes
    degen = analysis["degenerate_episodes"]
    if degen:
        lines.append("")
        lines.append("-" * 60)
        lines.append(f"Degenerate Episodes ({len(degen)}):")
        for d in degen:
            lines.append(
                f"  Episode {d['episode']:>3}: {d['steps']:,} steps, "
                f"reward={d['reward']:.2f}, RND mean={d['rnd_mean']:.6f}"
            )

    # Per-episode RND summary
    rnd_data = analysis["per_episode_rnd"]
    if rnd_data:
        collapsed = [e for e in rnd_data if e["rnd_collapsed"]]
        lines.append("")
        lines.append("-" * 60)
        lines.append("RND Collapse Analysis:")
        lines.append(f"  Total episodes:    {len(rnd_data)}")
        lines.append(
            f"  RND collapsed:     {len(collapsed)} ({len(collapsed) / len(rnd_data) * 100:.1f}%)"
        )
        if collapsed:
            lines.append("  Collapsed episodes (RND mean < 1e-4):")
            for e in collapsed[:10]:
                lines.append(
                    f"    Episode {e['episode']:>3}: "
                    f"{e['steps']:,} steps, "
                    f"RND mean={e['rnd_mean']:.6f}, "
                    f"{e['termination']}"
                )
            if len(collapsed) > 10:
                lines.append(f"    ... and {len(collapsed) - 10} more")

    lines.append("")
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _describe(values: list[float | int]) -> dict:
    """Compute descriptive statistics for a list of numbers.

    Parameters
    ----------
    values : list[float | int]
        Numeric values.

    Returns
    -------
    dict
        Keys: mean, median, min, max, std, count.
    """
    if not values:
        return {}

    n = len(values)
    mean = sum(values) / n
    sorted_vals = sorted(values)
    median = (
        sorted_vals[n // 2] if n % 2 == 1 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    )
    variance = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0
    std = math.sqrt(variance)

    return {
        "count": n,
        "mean": round(mean, 4),
        "median": round(median, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "std": round(std, 4),
    }


def _describe_paddle(positions: list[float]) -> dict:
    """Compute paddle position statistics.

    Parameters
    ----------
    positions : list[float]
        Paddle x-positions (normalized 0-1).

    Returns
    -------
    dict
        Descriptive stats plus range.
    """
    stats = _describe(positions)
    if stats:
        stats["range"] = round(stats["max"] - stats["min"], 4)
    return stats


def _format_stat_block(label: str, stats: dict) -> str:
    """Format a statistics dict into a compact display line.

    Parameters
    ----------
    label : str
        Label prefix.
    stats : dict
        Output from ``_describe()``.

    Returns
    -------
    str
        Formatted string.
    """
    if not stats:
        return f"{label}: no data"
    return (
        f"{label}: "
        f"mean={stats.get('mean', 0):.4f}, "
        f"median={stats.get('median', 0):.4f}, "
        f"min={stats.get('min', 0):.4f}, "
        f"max={stats.get('max', 0):.4f}, "
        f"std={stats.get('std', 0):.4f}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Run the training analysis CLI.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments. Uses ``sys.argv[1:]`` if *None*.
    """
    parser = argparse.ArgumentParser(
        description="Analyze JSONL training logs from train_rl.py",
    )
    parser.add_argument(
        "log_files",
        type=Path,
        nargs="+",
        help="Path(s) to JSONL training log file(s). Multiple files are concatenated in order.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted report",
    )
    parser.add_argument(
        "--top-episodes",
        type=int,
        default=5,
        help="Number of top/bottom episodes to highlight (default: 5)",
    )

    args = parser.parse_args(argv)

    all_events: list[dict] = []
    for log_file in args.log_files:
        all_events.extend(parse_jsonl(log_file))

    if not all_events:
        print("No events found in log file(s).", file=sys.stderr)
        sys.exit(1)

    analysis = build_analysis(all_events)

    if args.json:
        print(json.dumps(analysis, indent=2, default=str))
    else:
        print(format_report(analysis, top_episodes=args.top_episodes))


if __name__ == "__main__":
    main()
