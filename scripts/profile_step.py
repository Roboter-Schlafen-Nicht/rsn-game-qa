#!/usr/bin/env python3
"""Profile individual components of the step() loop.

Instruments apply_action, _capture_frame, _detect_objects,
build_observation, _check_late_game_over, check_termination,
compute_reward, _make_info, and _run_oracles individually.
"""

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

# Silence noisy loggers
for name in (
    "selenium",
    "urllib3",
    "ultralytics",
    "httpcore",
    "httpx",
    "PIL",
    "onnxruntime",
    "openvino",
):
    logging.getLogger(name).setLevel(logging.WARNING)

logging.basicConfig(level=logging.WARNING)

_project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, os.environ.get("PYTHONPATH", _project_root))

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

# -- Setup browser --
game_dir = os.environ.get("BREAKOUT71_DIR")
if not game_dir:
    raise SystemExit(
        "BREAKOUT71_DIR environment variable is required. "
        "Set it to the path of the Breakout 71 testbed directory."
    )
dist_index = os.path.join(game_dir, "dist", "index.html")

options = Options()
options.add_argument("--headless=new")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")
options.add_argument("--window-size=500,900")
options.add_argument("--mute-audio")

service = Service()
driver = webdriver.Chrome(service=service, options=options)
driver.get(f"file://{dist_index}")
time.sleep(2)

# -- Create env --
from games.breakout71.env import Breakout71Env

env = Breakout71Env(
    window_title="Breakout",
    yolo_weights="weights/breakout71/best.pt",
    max_steps=10_000,
    driver=driver,
    device="auto",
    headless=True,
)

print("Resetting env...")
obs, info = env.reset()
print(f"Reset done. obs shape: {obs.shape}")

# -- Profile individual components --
N = 50
timings = {
    "apply_action": [],
    "should_check_modals": [],
    "capture_frame": [],
    "detect_objects": [],
    "build_observation": [],
    "check_late_game_over": [],
    "check_termination": [],
    "compute_reward": [],
    "make_info": [],
    "run_oracles": [],
    "total_step": [],
}

for i in range(N):
    action = env.action_space.sample()
    step_start = time.perf_counter()

    # 1. apply_action
    t0 = time.perf_counter()
    env.apply_action(action)
    timings["apply_action"].append(time.perf_counter() - t0)

    # 2. _should_check_modals
    t0 = time.perf_counter()
    should_check = env._should_check_modals()
    timings["should_check_modals"].append(time.perf_counter() - t0)

    # (skip actual modal check for profiling — it would change game state)
    # In normal step, this only fires when _no_ball_count > 0

    # 3. _capture_frame
    t0 = time.perf_counter()
    frame = env._capture_frame()
    timings["capture_frame"].append(time.perf_counter() - t0)

    # 4. _detect_objects
    t0 = time.perf_counter()
    detections = env._detect_objects(frame)
    timings["detect_objects"].append(time.perf_counter() - t0)

    # 5. build_observation
    t0 = time.perf_counter()
    obs = env.build_observation(detections)
    timings["build_observation"].append(time.perf_counter() - t0)

    # 6. _check_late_game_over
    t0 = time.perf_counter()
    late_go = env._check_late_game_over(detections)
    timings["check_late_game_over"].append(time.perf_counter() - t0)

    if late_go:
        print(f"  Step {i}: late game over detected, stopping")
        break

    # 7. check_termination
    t0 = time.perf_counter()
    terminated, level_cleared = env.check_termination(detections)
    timings["check_termination"].append(time.perf_counter() - t0)

    # 8. compute_reward
    t0 = time.perf_counter()
    reward = env.compute_reward(detections, terminated, level_cleared)
    timings["compute_reward"].append(time.perf_counter() - t0)

    # 9. _make_info
    t0 = time.perf_counter()
    info_dict = env._make_info(detections)
    timings["make_info"].append(time.perf_counter() - t0)

    # 10. _run_oracles
    t0 = time.perf_counter()
    findings = env._run_oracles(obs, reward, terminated, False, info_dict)
    timings["run_oracles"].append(time.perf_counter() - t0)

    timings["total_step"].append(time.perf_counter() - step_start)

    env._step_count += 1

    if terminated:
        print(f"  Step {i}: terminated (level_cleared={level_cleared})")
        break

# -- Print results --
print(f"\n{'=' * 60}")
print(f"Step profiling results ({len(timings['total_step'])} steps)")
print(f"{'=' * 60}")

total_mean = np.mean(timings["total_step"]) * 1000

for key in [
    "apply_action",
    "should_check_modals",
    "capture_frame",
    "detect_objects",
    "build_observation",
    "check_late_game_over",
    "check_termination",
    "compute_reward",
    "make_info",
    "run_oracles",
    "total_step",
]:
    vals = np.array(timings[key]) * 1000  # to ms
    mean = np.mean(vals)
    std = np.std(vals)
    pct = (mean / total_mean * 100) if key != "total_step" else 100.0
    print(f"  {key:25s}: {mean:7.1f}ms ± {std:5.1f}ms  ({pct:5.1f}%)")

accounted = sum(np.mean(timings[k]) for k in timings if k != "total_step") * 1000
unaccounted = total_mean - accounted
print(f"\n  {'accounted':25s}: {accounted:7.1f}ms")
print(f"  {'unaccounted (overhead)':25s}: {unaccounted:7.1f}ms")
print(f"  {'FPS':25s}: {1000 / total_mean:7.1f}")

# Cleanup
env.close()
driver.quit()
print("\nDone.")
