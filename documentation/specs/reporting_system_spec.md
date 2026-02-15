# Reporting System Spec

> Design spec for the reporting subsystem. Updated to reflect the
> implemented dataclass-based architecture in `src/reporting/`.

## Overview

The reporting system converts QA run data (episodes, oracle findings,
performance metrics) into:

1. **JSON reports** -- structured, machine-readable session reports
2. **HTML dashboard** -- human-readable Bootstrap 5 summary page
3. **CI artifacts** -- uploaded to GitHub Actions for review

---

## Data Model

The reporting module is built around four dataclasses and two service
classes:

| Class              | Role                                            |
|--------------------|-------------------------------------------------|
| `FindingReport`    | Single oracle finding (severity, step, metadata) |
| `EpisodeMetrics`   | Performance stats for one episode (FPS, duration) |
| `EpisodeReport`    | All data for one episode (steps, reward, findings, metrics) |
| `SessionReport`    | Aggregated session (list of episodes + summary)  |
| `ReportGenerator`  | Builds a `SessionReport`, computes summary, writes JSON |
| `DashboardRenderer`| Renders session JSON to an HTML dashboard via Jinja2 |

---

## Severity Levels

The implementation uses the Oracle subsystem naming convention rather
than the original high/medium/low labels:

| Implementation | Spec equivalent | Meaning                          |
|----------------|-----------------|----------------------------------|
| `"critical"`   | high            | Crashes, blocking bugs           |
| `"warning"`    | medium          | Performance issues, stuck states |
| `"info"`       | low             | Minor observations               |

An episode is marked **FAIL** if it contains at least one `"critical"`
severity finding.

---

## JSON Report Schema

```json
{
  "session_id": "uuid-string",
  "game": "breakout-71",
  "build_id": "local",
  "timestamp": "2026-02-15T08:00:00+00:00",
  "episodes": [
    {
      "episode_id": 0,
      "steps": 350,
      "total_reward": 42.5,
      "terminated": true,
      "truncated": false,
      "seed": 123,
      "findings": [
        {
          "oracle_name": "CrashOracle",
          "severity": "critical",
          "step": 312,
          "description": "Game process died during episode.",
          "data": {},
          "screenshot_path": null
        },
        {
          "oracle_name": "PerformanceOracle",
          "severity": "warning",
          "step": 180,
          "description": "Step times high: avg=38.2 ms, p99=85.4 ms",
          "data": {"avg_ms": 38.2, "p99_ms": 85.4},
          "screenshot_path": null
        }
      ],
      "metrics": {
        "mean_fps": 55.0,
        "min_fps": 30.0,
        "max_reward_per_step": 3.0,
        "total_duration_seconds": 41.1
      }
    }
  ],
  "summary": {
    "total_episodes": 5,
    "total_findings": 6,
    "critical_findings": 1,
    "warning_findings": 2,
    "info_findings": 3,
    "episodes_failed": 1,
    "mean_episode_reward": 38.5,
    "mean_episode_length": 1100
  }
}
```

### Field Descriptions

| Field                             | Type         | Description                                          |
|-----------------------------------|--------------|------------------------------------------------------|
| `session_id`                      | string       | UUID generated at session start                      |
| `game`                            | string       | Name of game under test                              |
| `build_id`                        | string       | Git commit SHA or build number (from `CI_COMMIT_SHORT_SHA`, default `"local"`) |
| `timestamp`                       | string       | ISO-8601 timestamp of session start                  |
| `episodes`                        | array        | Per-episode data                                     |
| `episodes[].episode_id`           | int          | Sequential episode number                            |
| `episodes[].steps`                | int          | Total steps taken                                    |
| `episodes[].total_reward`         | float        | Cumulative reward                                    |
| `episodes[].terminated`           | bool         | Whether the episode ended naturally                  |
| `episodes[].truncated`            | bool         | Whether the episode was truncated (time/step limit)  |
| `episodes[].seed`                 | int \| null  | Random seed used, if any                             |
| `episodes[].findings`             | array        | Oracle findings for this episode                     |
| `episodes[].findings[].oracle_name` | string     | Name of the oracle that detected the issue           |
| `episodes[].findings[].severity`  | string       | `"critical"`, `"warning"`, or `"info"`               |
| `episodes[].findings[].step`      | int          | Environment step at which finding occurred           |
| `episodes[].findings[].description` | string     | Human-readable description                           |
| `episodes[].findings[].data`      | object       | Arbitrary metadata (oracle-specific)                 |
| `episodes[].findings[].screenshot_path` | string \| null | Path to saved screenshot, if available         |
| `episodes[].metrics`              | object       | Performance and gameplay metrics                     |
| `episodes[].metrics.mean_fps`     | float \| null | Average FPS during the episode                      |
| `episodes[].metrics.min_fps`      | float \| null | Minimum FPS observed                                |
| `episodes[].metrics.max_reward_per_step` | float \| null | Highest single-step reward                   |
| `episodes[].metrics.total_duration_seconds` | float \| null | Wall-clock duration in seconds             |
| `summary`                         | object       | Aggregated counts across all episodes                |
| `summary.total_episodes`          | int          | Count of episodes                                    |
| `summary.total_findings`          | int          | Total findings across all episodes                   |
| `summary.critical_findings`       | int          | Total critical-severity findings                     |
| `summary.warning_findings`        | int          | Total warning-severity findings                      |
| `summary.info_findings`           | int          | Total info-severity findings                         |
| `summary.episodes_failed`         | int          | Episodes with at least one critical finding          |
| `summary.mean_episode_reward`     | float        | Average cumulative reward per episode                |
| `summary.mean_episode_length`     | float        | Average steps per episode                            |

---

## ReportGenerator

The `ReportGenerator` class builds up a session report incrementally
and writes it to JSON.

```python
from src.reporting import ReportGenerator, EpisodeReport, FindingReport, EpisodeMetrics

gen = ReportGenerator(output_dir="reports", game_name="breakout-71")

episode = EpisodeReport(
    episode_id=0,
    steps=350,
    total_reward=42.5,
    terminated=True,
    truncated=False,
    seed=123,
    findings=[
        FindingReport(
            oracle_name="CrashOracle",
            severity="critical",
            step=312,
            description="Game process died during episode.",
        ),
    ],
    metrics=EpisodeMetrics(mean_fps=55.0, min_fps=30.0),
)

gen.add_episode(episode)
report_path = gen.save()  # writes reports/breakout-71_<uuid>.json
```

Key methods:

| Method            | Description                                       |
|-------------------|---------------------------------------------------|
| `add_episode(ep)` | Append an `EpisodeReport` to the session           |
| `compute_summary()` | Calculate aggregated stats (called automatically by `save()`) |
| `save(filename)`  | Write the session report to JSON                   |
| `to_dict()`       | Return the session report as a plain dict          |

---

## HTML Dashboard

### DashboardRenderer

The `DashboardRenderer` class uses Jinja2 to produce a self-contained
HTML file. It first looks for an external template at
`template_dir/template_name`; if the file does not exist, it falls back
to a built-in Bootstrap 5 template.

```python
from src.reporting import DashboardRenderer, ReportGenerator

gen = ReportGenerator()
# ... add episodes ...

renderer = DashboardRenderer()
renderer.render_to_file(gen.to_dict(), "reports/dashboard.html")

# Or from a saved JSON file:
renderer.generate_dashboard("reports/breakout-71_abc12345.json")
```

### Dashboard Features

**Run-level view:**
- Session ID, game name, build ID, timestamp
- Summary: episodes, failures, critical/warning/info counts

**Episode-level view:**
- Status: PASS/FAIL (based on any critical finding)
- Steps, total reward, average FPS
- Findings list with severity coloring
- CSS classes: `.severity-critical` (red), `.severity-warning` (orange),
  `.severity-info` (green)

### Template Customisation

Drop a Jinja2 template at `templates/dashboard.html.j2` to override the
built-in template. The template receives a `runs` list where each item
is a session report dict.

---

## CI Artifact Layout

```
reports/
  <game>_<session_id>.json   # structured report
  dashboard.html             # rendered dashboard
```

---

## Future Work

These features are documented for future implementation:

- **Coverage tracking** -- `novel_states`, `coverage_ratio` per episode
- **Artifact paths** -- `video_path`, `screenshots` per episode
- **JUnit XML** -- optional CI pass/fail integration where each episode
  maps to a test case

---

## Source Files

- `src/reporting/__init__.py` -- public API exports
- `src/reporting/report.py` -- `FindingReport`, `EpisodeMetrics`, `EpisodeReport`, `SessionReport`, `ReportGenerator`
- `src/reporting/dashboard.py` -- `DashboardRenderer` (Jinja2 + built-in Bootstrap 5 template)
- `tests/test_reporting.py` -- comprehensive test suite
- `docs/api/reporting.rst` -- Sphinx API reference
