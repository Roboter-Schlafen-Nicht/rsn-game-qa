# Reporting System Spec

> Extracted from `documentation/sessions/session1.md`. This is the reference
> design for JSON episode reports, the HTML dashboard, and CI artifact
> integration.

## Overview

The reporting system converts QA run data (episodes, oracle findings,
performance metrics) into:

1. **JSON reports** — structured, machine-readable session reports
2. **HTML dashboard** — human-readable Bootstrap-based summary page
3. **CI artifacts** — uploaded to GitHub Actions for review

---

## JSON Report Schema

```json
{
  "run_id": "2026-02-10T00:20Z_breakout71_smoke",
  "game": "Breakout71",
  "build_id": "commit_sha_or_build_number",
  "episodes": [
    {
      "episode_id": 0,
      "seed": 123,
      "steps": 350,
      "total_reward": 42.5,
      "findings": [
        {
          "type": "crash",
          "severity": "high",
          "message": "Game process/window died during episode."
        },
        {
          "type": "perf_frame_time",
          "severity": "medium",
          "message": "Step times high: avg=38.2 ms, p99=85.4 ms"
        }
      ],
      "coverage": {
        "novel_states": 120,
        "coverage_ratio": 0.87
      },
      "metrics": {
        "avg_fps": 55.0,
        "min_fps": 30.0,
        "max_score": 380
      },
      "artifacts": {
        "video_path": "videos/episode_0.mp4",
        "screenshots": ["screens/ep0_step120.png"]
      }
    }
  ],
  "summary": {
    "episodes_total": 5,
    "episodes_failed": 1,
    "high_severity_findings": 1,
    "medium_severity_findings": 2,
    "low_severity_findings": 3
  }
}
```

### Field Descriptions

| Field                    | Type       | Description                                          |
|--------------------------|------------|------------------------------------------------------|
| `run_id`                 | string     | Timestamp-based unique run identifier                |
| `game`                   | string     | Name of game under test                              |
| `build_id`               | string     | Git commit SHA or build number (from `CI_COMMIT_SHORT_SHA`) |
| `episodes`               | array      | Per-episode data                                     |
| `episodes[].episode_id`  | int        | Sequential episode number                            |
| `episodes[].seed`        | int        | Random seed used                                     |
| `episodes[].steps`       | int        | Total steps taken                                    |
| `episodes[].total_reward`| float      | Cumulative reward                                    |
| `episodes[].findings`    | array      | Oracle findings (type, severity, message + extras)   |
| `episodes[].coverage`    | object     | Novel states visited, coverage ratio                 |
| `episodes[].metrics`     | object     | avg_fps, min_fps, max_score                          |
| `episodes[].artifacts`   | object     | Paths to video/screenshots                           |
| `summary`                | object     | Aggregated counts across all episodes                |

---

## Report Writer (`write_run_report`)

```python
import json
import os
from pathlib import Path

def write_run_report(run_id, episodes, output_dir="reports"):
    summary = {
        "episodes_total": len(episodes),
        "episodes_failed": sum(
            any(f["severity"] == "high" for f in ep["findings"]) for ep in episodes
        ),
        "high_severity_findings": sum(
            sum(1 for f in ep["findings"] if f["severity"] == "high") for ep in episodes
        ),
        "medium_severity_findings": sum(
            sum(1 for f in ep["findings"] if f["severity"] == "medium") for ep in episodes
        ),
        "low_severity_findings": sum(
            sum(1 for f in ep["findings"] if f["severity"] == "low") for ep in episodes
        ),
    }

    report = {
        "run_id": run_id,
        "game": "Breakout71",
        "build_id": os.getenv("CI_COMMIT_SHORT_SHA", "local"),
        "episodes": episodes,
        "summary": summary,
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{run_id}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
```

---

## HTML Dashboard

### Features

**Run-level view:**
- List of all runs (each JSON file)
- Summary: episodes, failures, high/medium/low findings, average FPS, coverage

**Episode-level view:**
- Status: PASS/FAIL (based on any high-severity finding)
- Steps, total reward, key metrics
- Findings list with severity coloring
- Links to video/screenshot artifacts

### Dashboard Generator (`generate_dashboard.py`)

```python
import json
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

def load_run_reports(reports_dir="reports"):
    reports_path = Path(reports_dir)
    runs = []
    for path in sorted(reports_path.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            runs.append(json.load(f))
    return runs

def generate_dashboard(
    reports_dir="reports",
    templates_dir="templates",
    output_html="reports/index.html",
):
    runs = load_run_reports(reports_dir)
    if not runs:
        print("No JSON reports found, nothing to render.")
        return

    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template("index.html")
    html = template.render(runs=runs)

    out_path = Path(output_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Dashboard written to {out_path}")

if __name__ == "__main__":
    generate_dashboard()
```

### Jinja2 Template (`templates/index.html`)

Bootstrap 5 based, self-contained HTML. Key elements:

- **Runs table:** Run ID, Game, Build, Episodes, Failures, High/Medium/Low counts
- **Per-run episode table:** Episode #, Status (PASS/FAIL), Steps, Total Reward,
  Findings list (color-coded by severity), Key Metrics (avg FPS, min FPS, max score)
- **CSS classes:** `.severity-high` (red), `.severity-medium` (orange),
  `.severity-low` (green)
- **Episode status:** FAIL if any finding has `severity == "high"`, else PASS

See the full template in `documentation/sessions/session1.md` lines 3721-3836.

---

## JUnit XML (Optional)

For CI pass/fail integration, generate a JUnit XML test suite where:

- Each **episode** maps to a test case
- Test name: `game=Breakout71 episode=0`
- High-severity findings mark the test as failed
- Medium/low go into `<system-out>`
- Use the `junit-xml` Python package or write minimal XML directly

---

## CI Artifact Layout

```
reports/
  <run_id>.json          # structured report
  index.html             # rendered dashboard
videos/
  episode_0.mp4          # optional video recordings
screens/
  ep0_step120.png        # screenshots at finding steps
```

---

## Metrics Tracked

### Per-Episode

| Metric          | Source                                         |
|-----------------|------------------------------------------------|
| `avg_fps`       | Mean of `1.0 / step_duration_s`                |
| `min_fps`       | Minimum instantaneous FPS                      |
| `max_score`     | Highest score reached                          |
| `steps`         | Total environment steps                        |
| `total_reward`  | Cumulative RL reward                           |
| `novel_states`  | Count of unique states visited (coverage)      |
| `coverage_ratio`| novel_states / total_possible (estimated)      |

### Per-Run (Summary)

| Metric                    | Computation                              |
|---------------------------|------------------------------------------|
| `episodes_total`          | Count of episodes                        |
| `episodes_failed`         | Episodes with any high-severity finding  |
| `high_severity_findings`  | Total high-severity findings             |
| `medium_severity_findings`| Total medium-severity findings           |
| `low_severity_findings`   | Total low-severity findings              |

## Source Files

- `src/reporting/report.py` — `EpisodeReport`, `SessionReport`, `ReportGenerator`
- `src/reporting/dashboard.py` — `DashboardRenderer` (Jinja2)
