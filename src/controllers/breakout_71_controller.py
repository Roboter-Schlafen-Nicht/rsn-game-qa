from typing import Optional
import random
from controllers.adb_controller import EmulatorController


class BreakOut71Controller(EmulatorController):
    """
    Minimal game-specific helper for Breakout 71 built on top of
    EmulatorController.

    V1 goal:
    - Press and hold near the paddle and move left/right while holding.
    """

    SCREEN_WIDTH = 1080
    SCREEN_HEIGHT = 1920
    PADDLE_Y = 1800

    def __init__(
        self,
        adb_path: str = "adb",
        serial: Optional[str] = None,
        default_timeout: float = 10.0,
    ) -> None:
        super().__init__(
            adb_path=adb_path,
            serial=serial,
            default_timeout=default_timeout
        )

    # ---------- low-level "hold" helpers ----------

    def random_continuous_slide(
        self,
        min_duration_s: float = 0.8,
        max_duration_s: float = 2.5,
        edge_margin_frac: float = 0.1,
    ) -> None:
        """
        Single random continuous slide:

        - Random direction (left->right or right->left).
        - Random end x within a margin from the edge.
        - Finger stays down for the whole swipe.
        """
        duration_s = random.uniform(min_duration_s, max_duration_s)
        duration_ms = int(duration_s * 1000)

        left_min = int(self.SCREEN_WIDTH * edge_margin_frac)
        right_max = int(self.SCREEN_WIDTH * (1.0 - edge_margin_frac))
        y = self.PADDLE_Y

        direction = random.choice(["left_to_right", "right_to_left"])

        if direction == "left_to_right":
            start_x = random.randint(left_min, (left_min + right_max) // 2)
            end_x = random.randint(start_x, right_max)
        else:
            start_x = random.randint((left_min + right_max) // 2, right_max)
            end_x = random.randint(left_min, start_x)

        self.swipe(start_x, y, end_x, y, duration_ms=duration_ms)
