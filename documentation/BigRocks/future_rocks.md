# Next Big Rock – Breakout-71 Perception Stub

## Goal

Have a `perception/breakout_capture.py` that can grab frames from the Android device and return them as images, plus a stub `detect_objects`.

## Tasks

- [ ] Create `perception/breakout_capture.py`.
- [ ] Implement `grab_frame()`:
  - [ ] Use `EmulatorController.screencap()` to get PNG bytes.
  - [ ] Decode PNG into an image/NumPy array (e.g. via Pillow or OpenCV).
- [ ] Implement `detect_objects(frame)` stub:
  - [ ] Currently returns an empty list or simple mock.
  - [ ] Add clear TODOs for YOLO integration (paddle, ball, bricks, coins, combo).
- [ ] Wire into controller:
  - [ ] In `breakout_71_controller.py`, call `grab_frame()` once per control loop iteration.
  - [ ] (Optional) Every N steps, save a frame to `data/raw_frames/` for future labeling.
- [ ] Commit: `feat: add breakout perception stub and frame capture`

## 3. Perception placeholder for YOLO

- [ ] Create `perception/breakout_capture.py`.
- [ ] Implement `grab_frame()`:
  - [ ] Capture a screenshot of the game window/region.
  - [ ] Return it as an image/array object.

- [ ] Implement `detect_objects(frame)` stub:
  - [ ] Current implementation returns empty or mock detections.
  - [ ] Add TODOs for future YOLO integration (paddle, ball, bricks, coins, combo).

- [ ] Wire controller to perception:
  - [ ] In the controller loop, call `grab_frame()` each iteration.
  - [ ] (Optional today) Log one frame every N steps to `data/raw_frames/` for future labeling.

---

## 4. Wrap up

- [ ] Run the full loop (controller + perception stub) for a few minutes.
- [ ] Fix any obvious crashes or path issues.
- [ ] Commit with message: `feat: initial breakout controller and perception stubs`.
- [ ] Jot down 2–3 notes for next session (e.g. “decide on final capture method”, “prepare 10 labeled images”).
