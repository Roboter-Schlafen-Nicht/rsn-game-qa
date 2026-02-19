#!/usr/bin/env python3
"""Compare headless capture methods: Selenium PNG vs CDP JPEG vs canvas toDataURL."""

import base64
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

for name in ("selenium", "urllib3", "ultralytics", "openvino"):
    logging.getLogger(name).setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)

_project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, os.environ.get("PYTHONPATH", _project_root))

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

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

import cv2

N = 30

# --- Method 1: Selenium get_screenshot_as_png ---
times_png = []
for _ in range(N):
    t0 = time.perf_counter()
    png_bytes = driver.get_screenshot_as_png()
    nparr = np.frombuffer(png_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    times_png.append(time.perf_counter() - t0)

# --- Method 2: CDP Page.captureScreenshot (JPEG) ---
times_cdp_jpeg = []
for _ in range(N):
    t0 = time.perf_counter()
    result = driver.execute_cdp_cmd(
        "Page.captureScreenshot",
        {
            "format": "jpeg",
            "quality": 80,
        },
    )
    img_bytes = base64.b64decode(result["data"])
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    times_cdp_jpeg.append(time.perf_counter() - t0)

# --- Method 3: CDP Page.captureScreenshot (PNG) ---
times_cdp_png = []
for _ in range(N):
    t0 = time.perf_counter()
    result = driver.execute_cdp_cmd(
        "Page.captureScreenshot",
        {
            "format": "png",
        },
    )
    img_bytes = base64.b64decode(result["data"])
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    times_cdp_png.append(time.perf_counter() - t0)

# --- Method 4: canvas.toDataURL (JPEG) ---
times_canvas = []
for _ in range(N):
    t0 = time.perf_counter()
    data_url = driver.execute_script(
        "return document.getElementById('game').toDataURL('image/jpeg', 0.8);"
    )
    # data:image/jpeg;base64,...
    b64_data = data_url.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    times_canvas.append(time.perf_counter() - t0)

# --- Method 5: canvas.toDataURL (PNG) ---
times_canvas_png = []
for _ in range(N):
    t0 = time.perf_counter()
    data_url = driver.execute_script(
        "return document.getElementById('game').toDataURL('image/png');"
    )
    b64_data = data_url.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    times_canvas_png.append(time.perf_counter() - t0)

driver.quit()

print(f"\n{'=' * 60}")
print(f"Capture method comparison ({N} frames each)")
print(f"{'=' * 60}")
for name, times in [
    ("Selenium PNG", times_png),
    ("CDP JPEG (q=80)", times_cdp_jpeg),
    ("CDP PNG", times_cdp_png),
    ("Canvas toDataURL JPEG", times_canvas),
    ("Canvas toDataURL PNG", times_canvas_png),
]:
    arr = np.array(times) * 1000
    print(
        f"  {name:25s}: {np.mean(arr):6.1f}ms ± {np.std(arr):4.1f}ms  (FPS: {1000 / np.mean(arr):5.1f})"
    )
