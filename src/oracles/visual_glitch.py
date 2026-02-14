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
        raise NotImplementedError(
            "VisualGlitchOracle.on_step: visual glitch detection not yet implemented"
        )

    def _compute_phash(self, frame: np.ndarray) -> Any:
        """Compute the perceptual hash of a frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image.

        Returns
        -------
        imagehash.ImageHash
            The perceptual hash of the frame.
        """
        raise NotImplementedError("Perceptual hash computation not yet implemented")

    def _compute_ssim(self, frame_a: np.ndarray, frame_b: np.ndarray) -> float:
        """Compute SSIM between two frames.

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
        raise NotImplementedError("SSIM computation not yet implemented")
