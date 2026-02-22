"""Human-readable finding descriptions for QA reports.

Maps oracle finding types to plain-English descriptions, business impact
explanations, and actionable recommendations suitable for non-technical
readers (e.g. a project manager or game producer).

The mapping handles two patterns:
1. Findings with a ``data["type"]`` key (most oracles).
2. Findings without a ``type`` key (CrashOracle process-dead,
   StuckOracle, ScoreAnomalyOracle) -- uses oracle_name + description
   pattern matching as fallback.
"""

from __future__ import annotations

from typing import Any

# ── Severity to business impact ─────────────────────────────────────

SEVERITY_INFO: dict[str, dict[str, str]] = {
    "critical": {
        "label": "Critical",
        "impact": (
            "This issue would cause a player to lose progress, be unable "
            "to continue, or have a severely degraded experience. It "
            "should be fixed before release."
        ),
        "color": "#ef4444",
        "bg_color": "#1c0a0a",
        "badge_class": "severity-critical",
    },
    "warning": {
        "label": "Warning",
        "impact": (
            "This issue degrades the player experience but does not "
            "prevent gameplay. It should be addressed in the next "
            "update cycle."
        ),
        "color": "#f59e0b",
        "bg_color": "#1c1507",
        "badge_class": "severity-warning",
    },
    "info": {
        "label": "Info",
        "impact": (
            "A minor observation that may indicate a deeper issue. "
            "Worth investigating but not blocking."
        ),
        "color": "#60a5fa",
        "bg_color": "#0a1628",
        "badge_class": "severity-info",
    },
}

# ── Finding type descriptions ───────────────────────────────────────
#
# Key format: "oracle_name:type" or "oracle_name" (for typeless findings).
# Each entry has:
#   - title: short human-readable title
#   - description: what happened (template with {placeholders})
#   - recommendation: what the developer should do

_FINDING_DESCRIPTIONS: dict[str, dict[str, str]] = {
    # ── CrashOracle ──
    "crash:black_frame": {
        "title": "Black Screen",
        "description": (
            "The game screen went completely black, indicating a "
            "possible crash or rendering failure."
        ),
        "recommendation": (
            "Check the rendering pipeline for errors that could cause "
            "the canvas to stop drawing. Review GPU resource management "
            "and shader compilation error handling."
        ),
    },
    "crash:frozen_frame": {
        "title": "Game Freeze",
        "description": (
            "The game screen stopped updating for an extended period. "
            "The display was frozen while the process remained running."
        ),
        "recommendation": (
            "Look for infinite loops, deadlocks, or blocking operations "
            "on the main thread. Check for unhandled exceptions in the "
            "game loop that silently stop rendering."
        ),
    },
    "crash": {
        "title": "Process Crash",
        "description": ("The game process appears to have died or become unresponsive."),
        "recommendation": (
            "Review crash logs and error handlers. Ensure the game "
            "handles out-of-memory conditions and GPU device loss "
            "gracefully."
        ),
    },
    # ── StuckOracle ──
    "stuck": {
        "title": "Agent Stuck",
        "description": (
            "The game entered a state where no meaningful progress "
            "was being made. Reward and observations remained stagnant."
        ),
        "recommendation": (
            "Check for soft-lock conditions where the player cannot "
            "progress. Ensure all game states have valid exit paths."
        ),
    },
    # ── ScoreAnomalyOracle ──
    "score_anomaly": {
        "title": "Score Anomaly",
        "description": (
            "An unexpected change in the game score was detected. "
            "This could indicate a scoring bug or exploit."
        ),
        "recommendation": (
            "Review the scoring logic for edge cases: negative scores, "
            "integer overflow, race conditions in score updates, and "
            "unintended multiplier interactions."
        ),
    },
    # ── VisualGlitchOracle ──
    "visual_glitch:phash_anomaly": {
        "title": "Visual Anomaly (Perceptual Hash)",
        "description": (
            "A sudden, unusual visual change was detected between "
            "consecutive frames that does not match normal gameplay "
            "patterns."
        ),
        "recommendation": (
            "Check for rendering glitches, z-order issues, texture "
            "corruption, or visual artifacts caused by race conditions "
            "in the rendering pipeline."
        ),
    },
    "visual_glitch:ssim_anomaly": {
        "title": "Visual Anomaly (Structural Similarity)",
        "description": (
            "A frame showed an unexpected structural difference from "
            "recent frames, suggesting a visual glitch or rendering "
            "error."
        ),
        "recommendation": (
            "Inspect the rendering output around this step for "
            "flickering, tearing, or misaligned sprites. Check "
            "frame buffer management."
        ),
    },
    # ── PerformanceOracle ──
    "performance:low_fps": {
        "title": "Low Frame Rate",
        "description": (
            "The game's frame rate dropped below acceptable levels, "
            "causing visible stuttering or lag."
        ),
        "recommendation": (
            "Profile the game loop for expensive operations. Check for "
            "unnecessary object allocation, unoptimized draw calls, or "
            "excessive DOM manipulation per frame."
        ),
    },
    "performance:high_cpu": {
        "title": "High CPU Usage",
        "description": (
            "CPU utilization exceeded normal thresholds during "
            "gameplay, which may cause slowdowns on lower-end "
            "hardware."
        ),
        "recommendation": (
            "Profile CPU-intensive code paths. Look for tight loops, "
            "unthrottled timers, or heavy computation that could be "
            "deferred or cached."
        ),
    },
    "performance:high_ram": {
        "title": "High Memory Usage",
        "description": (
            "Memory consumption exceeded expected levels. This may "
            "indicate a memory leak or excessive asset loading."
        ),
        "recommendation": (
            "Check for memory leaks: unreleased event listeners, "
            "growing arrays, cached objects that are never freed, "
            "or textures loaded but never unloaded."
        ),
    },
    # ── PhysicsViolationOracle ──
    "physics_violation:tunneling": {
        "title": "Object Tunneling",
        "description": (
            "A game object passed through a solid boundary, "
            "suggesting a physics collision detection failure."
        ),
        "recommendation": (
            "Implement continuous collision detection (CCD) or "
            "reduce the physics timestep. Check that collision masks "
            "and layers are configured correctly."
        ),
    },
    "physics_violation:paddle_pass_through": {
        "title": "Paddle Pass-Through",
        "description": (
            "The ball passed through the paddle without bouncing, "
            "indicating a collision detection failure."
        ),
        "recommendation": (
            "Review the paddle-ball collision logic. Ensure collision "
            "checks account for high ball speeds and paddle movement "
            "between frames."
        ),
    },
    "physics_violation:ghost_collision": {
        "title": "Ghost Collision",
        "description": (
            "An unexpected collision was detected where none should "
            "have occurred, causing erratic object behavior."
        ),
        "recommendation": (
            "Audit collision geometry and hit-box definitions. Check "
            "for stale collision data from destroyed objects."
        ),
    },
    # ── BoundaryOracle ──
    "boundary:hard_oob": {
        "title": "Object Out of Bounds (Hard)",
        "description": (
            "A game object left the valid play area entirely, which "
            "should not be possible under normal game rules."
        ),
        "recommendation": (
            "Add boundary clamping to object positions. Review physics "
            "impulse calculations that could launch objects beyond "
            "the game boundaries."
        ),
    },
    "boundary:soft_oob": {
        "title": "Object Near Boundary",
        "description": (
            "A game object was detected very close to the edge of "
            "the play area, which may indicate a boundary issue."
        ),
        "recommendation": (
            "Monitor boundary behavior under sustained testing. "
            "Ensure edge-case physics interactions near walls are "
            "handled correctly."
        ),
    },
    # ── StateTransitionOracle ──
    "state_transition:excessive_lives_loss": {
        "title": "Excessive Lives Lost",
        "description": (
            "Multiple lives were lost in rapid succession, which "
            "may indicate a bug in the life-loss handling logic."
        ),
        "recommendation": (
            "Add invulnerability frames after losing a life. Check "
            "for re-entrant life-loss triggers and ensure the life "
            "counter is properly debounced."
        ),
    },
    "state_transition:lives_increase": {
        "title": "Unexpected Life Gain",
        "description": (
            "The player's life count increased unexpectedly, "
            "which may indicate a scoring or power-up bug."
        ),
        "recommendation": (
            "Audit all code paths that modify the life counter. "
            "Ensure extra-life power-ups cannot stack or trigger "
            "multiple times."
        ),
    },
    "state_transition:level_skip": {
        "title": "Level Skip",
        "description": (
            "The game advanced by more than one level in a single "
            "transition, skipping intermediate content."
        ),
        "recommendation": (
            "Review the level progression logic. Ensure level "
            "completion events cannot fire multiple times per "
            "transition."
        ),
    },
    "state_transition:level_decrease": {
        "title": "Level Decrease",
        "description": (
            "The level counter decreased during gameplay, which "
            "should not occur under normal game rules."
        ),
        "recommendation": (
            "Check for integer underflow or incorrect level assignment in the progression system."
        ),
    },
    "state_transition:invalid_transition": {
        "title": "Invalid State Transition",
        "description": (
            "The game transitioned to an unexpected state that "
            "violates the expected state machine rules."
        ),
        "recommendation": (
            "Review the game state machine. Add assertions or guards "
            "to prevent invalid transitions in production."
        ),
    },
    # ── EpisodeLengthOracle ──
    "episode_length:long_episode": {
        "title": "Unusually Long Episode",
        "description": (
            "This episode lasted significantly longer than expected, "
            "which may indicate a soft-lock or degenerate game state."
        ),
        "recommendation": (
            "Check for game states where the player cannot lose or "
            "progress. Ensure timeout or difficulty scaling mechanics "
            "prevent infinite sessions."
        ),
    },
    "episode_length:short_episode": {
        "title": "Unusually Short Episode",
        "description": (
            "This episode ended much faster than expected, which "
            "may indicate an instant-death bug or broken start "
            "sequence."
        ),
        "recommendation": (
            "Review the game initialization sequence. Check for "
            "race conditions during level loading that could cause "
            "immediate failure."
        ),
    },
    "episode_length:statistical_outlier": {
        "title": "Episode Length Outlier",
        "description": (
            "This episode's length was a statistical outlier compared "
            "to other episodes in the session."
        ),
        "recommendation": (
            "Investigate what made this episode different. The outlier "
            "may reveal an intermittent bug or an unusual game state "
            "path."
        ),
    },
    # ── TemporalAnomalyOracle ──
    "temporal_anomaly:teleportation": {
        "title": "Object Teleportation",
        "description": (
            "A game object moved an implausibly large distance "
            "between consecutive frames, appearing to teleport."
        ),
        "recommendation": (
            "Cap maximum per-frame displacement. Check for position "
            "resets, uninitialized coordinates, or NaN propagation "
            "in physics calculations."
        ),
    },
    "temporal_anomaly:flickering": {
        "title": "Object Flickering",
        "description": (
            "A game object rapidly appeared and disappeared across consecutive frames."
        ),
        "recommendation": (
            "Review object lifecycle management. Check for visibility "
            "toggling bugs, z-fighting, or detection instability in "
            "the rendering layer."
        ),
    },
    # ── RewardConsistencyOracle ──
    "reward_consistency:score_reward_mismatch": {
        "title": "Score/Reward Mismatch",
        "description": (
            "The internal reward signal did not match the visible "
            "score change, suggesting a reward calculation bug."
        ),
        "recommendation": (
            "Audit the reward function. Ensure score deltas are "
            "correctly captured and reward scaling matches the "
            "intended design."
        ),
    },
    "reward_consistency:phantom_reward": {
        "title": "Phantom Reward",
        "description": (
            "A reward was given without any corresponding game "
            "event, suggesting a reward signal error."
        ),
        "recommendation": (
            "Trace the reward source. Check for delayed reward "
            "signals from previous events or unintended reward "
            "triggers."
        ),
    },
    "reward_consistency:lives_reward_mismatch": {
        "title": "Lives/Reward Mismatch",
        "description": ("A life change occurred but the reward did not reflect it appropriately."),
        "recommendation": (
            "Verify that life-loss penalties and life-gain bonuses "
            "are correctly applied in the reward function."
        ),
    },
    "reward_consistency:brick_reward_mismatch": {
        "title": "Brick/Reward Mismatch",
        "description": (
            "Bricks were destroyed but the reward did not reflect the expected value."
        ),
        "recommendation": (
            "Check the brick destruction detection and reward "
            "calculation. Ensure simultaneous brick destructions "
            "are counted correctly."
        ),
    },
    # ── SoakOracle ──
    "soak:memory_leak": {
        "title": "Memory Leak Detected",
        "description": (
            "Memory usage showed a sustained upward trend over the "
            "session, indicating a probable memory leak."
        ),
        "recommendation": (
            "Profile memory allocation over time. Use heap snapshots "
            "to identify objects that accumulate without being "
            "garbage collected. Common culprits: event listeners, "
            "closures, DOM node references."
        ),
    },
    "soak:fps_degradation": {
        "title": "FPS Degradation Over Time",
        "description": (
            "Frame rate decreased steadily over the session, "
            "suggesting a performance regression over time."
        ),
        "recommendation": (
            "Profile the game loop at different session lengths. "
            "Check for growing data structures, accumulating draw "
            "calls, or particle systems that are not properly culled."
        ),
    },
}


def get_finding_description(finding: dict[str, Any]) -> dict[str, str]:
    """Return a human-readable description for a finding.

    Looks up the finding by ``oracle_name:data.type`` first, then
    falls back to ``oracle_name`` alone.

    Parameters
    ----------
    finding : dict[str, Any]
        A finding dict as produced by ``FindingReport`` serialisation.
        Must contain at least ``oracle_name`` and ``description``.

    Returns
    -------
    dict[str, str]
        Dict with ``title``, ``description``, and ``recommendation``.
        Falls back to the raw finding data if no mapping exists.
    """
    oracle_name = finding.get("oracle_name", "unknown")
    data = finding.get("data", {})
    finding_type = data.get("type") if isinstance(data, dict) else None

    # Try specific key first: "oracle_name:type"
    if finding_type:
        key = f"{oracle_name}:{finding_type}"
        if key in _FINDING_DESCRIPTIONS:
            return _FINDING_DESCRIPTIONS[key]

    # Fall back to oracle_name only
    if oracle_name in _FINDING_DESCRIPTIONS:
        return _FINDING_DESCRIPTIONS[oracle_name]

    # Final fallback: use raw data
    return {
        "title": oracle_name.replace("_", " ").title(),
        "description": finding.get("description", "No description available."),
        "recommendation": "Investigate this finding manually.",
    }


def get_severity_info(severity: str) -> dict[str, str]:
    """Return display information for a severity level.

    Parameters
    ----------
    severity : str
        One of ``"critical"``, ``"warning"``, or ``"info"``.

    Returns
    -------
    dict[str, str]
        Dict with ``label``, ``impact``, ``color``, ``bg_color``,
        ``badge_class``.
    """
    return SEVERITY_INFO.get(
        severity,
        {
            "label": severity.title(),
            "impact": "Unknown severity level.",
            "color": "#8e8e96",
            "bg_color": "#1c1c21",
            "badge_class": "severity-unknown",
        },
    )


def enrich_report_data(report_data: dict[str, Any]) -> dict[str, Any]:
    """Add human-readable enrichments to a report data dict.

    Adds to each finding:
    - ``human_title``: short descriptive title
    - ``human_description``: plain-English explanation
    - ``human_recommendation``: actionable fix suggestion
    - ``severity_info``: display metadata for the severity level

    Adds to the top-level report:
    - ``severity_definitions``: all severity levels with impact descriptions
    - ``executive_summary``: generated narrative summary

    Parameters
    ----------
    report_data : dict[str, Any]
        The session report dict from ``ReportGenerator.to_dict()``.
        Modified in-place and returned.

    Returns
    -------
    dict[str, Any]
        The enriched report data dict.
    """
    # Enrich each finding
    for episode in report_data.get("episodes", []):
        for finding in episode.get("findings", []):
            desc = get_finding_description(finding)
            finding["human_title"] = desc["title"]
            finding["human_description"] = desc["description"]
            finding["human_recommendation"] = desc["recommendation"]
            finding["severity_info"] = get_severity_info(finding.get("severity", "info"))

    # Add severity definitions
    report_data["severity_definitions"] = SEVERITY_INFO

    # Generate executive summary
    report_data["executive_summary"] = _generate_executive_summary(report_data)

    # Generate grouped recommendations
    report_data["recommendations"] = _generate_recommendations(report_data)

    return report_data


def _generate_executive_summary(report_data: dict[str, Any]) -> dict[str, Any]:
    """Generate an executive summary from the report data.

    Parameters
    ----------
    report_data : dict[str, Any]
        The session report dict.

    Returns
    -------
    dict[str, Any]
        Summary with ``verdict``, ``narrative``, and ``highlights``.
    """
    summary = report_data.get("summary", {})
    game = report_data.get("game", "Unknown Game")
    total_episodes = summary.get("total_episodes", 0)
    critical = summary.get("critical_findings", 0)
    warnings = summary.get("warning_findings", 0)
    info_count = summary.get("info_findings", 0)
    total_findings = summary.get("total_findings", 0)
    episodes_failed = summary.get("episodes_failed", 0)
    mean_length = summary.get("mean_episode_length", 0)

    # Determine verdict
    if critical > 0:
        verdict = "ISSUES FOUND"
        verdict_class = "verdict-fail"
    elif warnings > 0:
        verdict = "WARNINGS"
        verdict_class = "verdict-warn"
    else:
        verdict = "PASS"
        verdict_class = "verdict-pass"

    # Build narrative
    parts = []
    parts.append(
        f"We tested **{game}** over **{total_episodes} episode"
        f"{'s' if total_episodes != 1 else ''}** "
        f"(average {mean_length:.0f} steps each)."
    )

    if total_findings == 0:
        parts.append("No issues were detected during testing.")
    else:
        parts.append(
            f"A total of **{total_findings} finding"
            f"{'s' if total_findings != 1 else ''}** "
            f"{'were' if total_findings != 1 else 'was'} detected."
        )

    if critical > 0:
        parts.append(
            f"**{critical} critical issue{'s' if critical != 1 else ''}** "
            f"{'were' if critical != 1 else 'was'} found that would "
            f"impact players. {episodes_failed} of {total_episodes} "
            f"episodes experienced critical failures."
        )

    if warnings > 0:
        parts.append(
            f"{warnings} warning{'s' if warnings != 1 else ''} "
            f"{'were' if warnings != 1 else 'was'} raised for issues "
            f"that degrade the experience but don't prevent gameplay."
        )

    if info_count > 0:
        parts.append(
            f"{info_count} informational observation"
            f"{'s' if info_count != 1 else ''} "
            f"{'were' if info_count != 1 else 'was'} recorded."
        )

    # Highlights
    highlights = []
    if critical > 0:
        highlights.append(f"{critical} critical issue{'s' if critical != 1 else ''}")
    if warnings > 0:
        highlights.append(f"{warnings} warning{'s' if warnings != 1 else ''}")
    if episodes_failed > 0:
        fail_pct = (episodes_failed / total_episodes * 100) if total_episodes > 0 else 0
        highlights.append(f"{fail_pct:.0f}% episode failure rate")

    return {
        "verdict": verdict,
        "verdict_class": verdict_class,
        "narrative": " ".join(parts),
        "highlights": highlights,
    }


def _generate_recommendations(report_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate grouped, deduplicated recommendations from findings.

    Groups findings by type, deduplicates recommendations, and
    sorts by severity (critical first).

    Parameters
    ----------
    report_data : dict[str, Any]
        The session report dict.

    Returns
    -------
    list[dict[str, Any]]
        List of recommendation dicts with ``title``, ``severity``,
        ``count``, ``recommendation``, ``description``.
    """
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    seen: dict[str, dict[str, Any]] = {}

    for episode in report_data.get("episodes", []):
        for finding in episode.get("findings", []):
            oracle_name = finding.get("oracle_name", "unknown")
            data = finding.get("data", {})
            finding_type = data.get("type") if isinstance(data, dict) else None
            key = f"{oracle_name}:{finding_type}" if finding_type else oracle_name

            if key not in seen:
                desc = get_finding_description(finding)
                seen[key] = {
                    "title": desc["title"],
                    "severity": finding.get("severity", "info"),
                    "count": 0,
                    "recommendation": desc["recommendation"],
                    "description": desc["description"],
                }
            seen[key]["count"] += 1

            # Escalate severity if a more severe instance is found
            existing_order = severity_order.get(seen[key]["severity"], 2)
            new_order = severity_order.get(finding.get("severity", "info"), 2)
            if new_order < existing_order:
                seen[key]["severity"] = finding["severity"]

    recommendations = list(seen.values())
    recommendations.sort(key=lambda r: severity_order.get(r["severity"], 2))
    return recommendations
