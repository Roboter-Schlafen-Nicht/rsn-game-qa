"""Visual Glitch Oracle — detects rendering artifacts and visual anomalies.

Compares consecutive frames using perceptual hashing and structural
similarity to catch texture corruption, z-fighting, flickering, and
other visual glitches.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Oracle


class VisualGlitchOracle(Oracle):
    """Detects visual glitches by comparing consecutive frames.

    Detection strategy
    ------------------
    1. Compute perceptual hash (pHash) of each frame and compare to
       the previous frame's hash.  A large hamming distance in a short
       time indicates flickering or corruption.
    2. Compute SSIM (Structural Similarity Index) between consecutive
       frames.  A sudden drop below ``ssim_threshold`` flags a glitch.
    3. Optionally run a lightweight CNN classifier trained to
       distinguish "normal" vs "glitched" frames (future enhancement).

    Parameters
    ----------
    ssim_threshold : float
        Minimum SSIM between consecutive frames.  Below this, a
        warning is raised.  Default is 0.5.
    hash_distance_threshold : int
        Maximum allowable hamming distance between consecutive frame
        pHashes.  Above this, a warning is raised.  Default is 20.
    min_interval : int
        Minimum steps between consecutive findings to avoid spam.
        Default is 10.
    """

    def __init__(
        self,
        ssim_threshold: float = 0.5,
        hash_distance_threshold: int = 20,
        min_interval: int = 10,
    ) -> None:
        super().__init__(name="visual_glitch")
        self.ssim_threshold = ssim_threshold
        self.hash_distance_threshold = hash_distance_threshold
        self.min_interval = min_interval

        self._prev_frame: np.ndarray | None = None
        self._prev_hash: Any = None
        self._step_count: int = 0
        self._last_finding_step: int = -self.min_interval

    def on_reset(self, obs: np.ndarray, info: dict[str, Any]) -> None:
        """Reset visual tracking at episode start.

        Parameters
        ----------
        obs : np.ndarray
            Initial observation.
        info : dict[str, Any]
            Reset info dict.
        """
        self._prev_frame = None
        self._prev_hash = None
        self._step_count = 0
        self._last_finding_step = -self.min_interval

    def on_step(
        self,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        """Compare current frame to previous and check for visual glitches.

        Parameters
        ----------
        obs : np.ndarray
            Current observation.
        reward : float
            Step reward.
        terminated : bool
            Episode terminated flag.
        truncated : bool
            Episode truncated flag.
        info : dict[str, Any]
            Step info dict.  Expected to contain ``"frame"`` (raw BGR
            image) for visual analysis.
        """
        self._step_count += 1

        frame = info.get("frame")
        if frame is None:
            return

        # Always compute current hash for tracking
        current_hash = self._compute_phash(frame)

        # Rate-limit findings
        can_report = (self._step_count - self._last_finding_step) >= self.min_interval

        if self._prev_frame is not None and can_report:
            # Check perceptual hash distance
            if self._prev_hash is not None and current_hash is not None:
                distance = self._prev_hash - current_hash
                if distance > self.hash_distance_threshold:
                    self._add_finding(
                        severity="warning",
                        step=self._step_count,
                        description=(
                            f"Perceptual hash distance {distance} exceeds "
                            f"threshold {self.hash_distance_threshold} — "
                            f"possible visual glitch"
                        ),
                        data={
                            "type": "phash_anomaly",
                            "hash_distance": distance,
                            "threshold": self.hash_distance_threshold,
                        },
                        frame=frame,
                    )
                    self._last_finding_step = self._step_count

            # Check SSIM (re-check rate limit after potential phash finding)
            still_can_report = (
                self._step_count - self._last_finding_step
            ) >= self.min_interval

            if still_can_report:
                ssim_val = self._compute_ssim(self._prev_frame, frame)
                if ssim_val < self.ssim_threshold:
                    self._add_finding(
                        severity="warning",
                        step=self._step_count,
                        description=(
                            f"SSIM {ssim_val:.3f} below threshold "
                            f"{self.ssim_threshold} — possible visual glitch"
                        ),
                        data={
                            "type": "ssim_anomaly",
                            "ssim": ssim_val,
                            "threshold": self.ssim_threshold,
                        },
                        frame=frame,
                    )
                    self._last_finding_step = self._step_count

        self._prev_frame = frame.copy()
        self._prev_hash = current_hash

    def _compute_phash(self, frame: np.ndarray) -> Any:
        """Compute the perceptual hash of a frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image.

        Returns
        -------
        imagehash.ImageHash or None
            The perceptual hash of the frame, or None if imagehash
            is not available.
        """
        try:
            import imagehash
            from PIL import Image
        except ImportError:
            return None

        # Convert BGR numpy array to PIL Image
        try:
            import cv2
        except ImportError:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        return imagehash.phash(pil_image)

    def _compute_ssim(self, frame_a: np.ndarray, frame_b: np.ndarray) -> float:
        """Compute SSIM between two frames.

        Uses a manual SSIM implementation based on the Wang et al. formula
        to avoid dependency on scikit-image.

        Parameters
        ----------
        frame_a : np.ndarray
            First BGR frame.
        frame_b : np.ndarray
            Second BGR frame.

        Returns
        -------
        float
            SSIM value in [0.0, 1.0].
        """
        # Convert to grayscale float
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV (cv2) is required to compute SSIM in "
                "VisualGlitchOracle. Install 'opencv-python' or "
                "'opencv-python-headless'."
            ) from exc

        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY).astype(np.float64)

        # Constants for stability (standard values for 8-bit images)
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        mu_a = cv2.GaussianBlur(gray_a, (11, 11), 1.5)
        mu_b = cv2.GaussianBlur(gray_b, (11, 11), 1.5)

        mu_a_sq = mu_a**2
        mu_b_sq = mu_b**2
        mu_ab = mu_a * mu_b

        sigma_a_sq = cv2.GaussianBlur(gray_a**2, (11, 11), 1.5) - mu_a_sq
        sigma_b_sq = cv2.GaussianBlur(gray_b**2, (11, 11), 1.5) - mu_b_sq
        sigma_ab = cv2.GaussianBlur(gray_a * gray_b, (11, 11), 1.5) - mu_ab

        numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
        denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)

        ssim_map = numerator / denominator
        return float(ssim_map.mean())
