"""RND (Random Network Distillation) reward wrapper for exploration-driven RL.

Implements the RND intrinsic motivation method from Burda et al. (2018)
as a Stable Baselines 3 ``VecEnvWrapper``.  The wrapper adds an intrinsic
novelty bonus to the extrinsic (survival) reward, encouraging the agent
to visit diverse game states rather than optimising a single strategy.

Architecture
------------
- **Target network**: Fixed random CNN (3 conv layers -> 512-dim embedding).
  Never trained — provides a stable, unpredictable reference mapping.
- **Predictor network**: Trainable CNN with same backbone + deeper FC head.
  Trained via MSE loss to match the target's output.
- **Intrinsic reward**: MSE between target and predictor embeddings.
  High for novel states (predictor hasn't learned them), low for familiar.
- **Reward normalisation**: Running variance scaling (no mean subtraction)
  to keep intrinsic rewards at unit scale.
- **Observation normalisation**: Running mean/std with clipping to [-5, 5].

Integration
-----------
Inserted as the outermost VecEnv wrapper in the training pipeline::

    Env -> CnnObservationWrapper -> DummyVecEnv -> VecFrameStack
        -> VecTransposeImage -> RNDRewardWrapper -> PPO

The base env uses ``reward_mode="survival"`` (extrinsic signal); the RND
wrapper adds the intrinsic bonus externally.

Reference
---------
Burda, Y., Edwards, H., Storkey, A., & Klimov, O. (2018).
Exploration by Random Network Distillation. arXiv:1810.12894.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    # Allow ``import src.platform`` without torch installed (CI/docs).
    # Actual RND usage raises a clear error at construction time.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper


# ---------------------------------------------------------------------------
# Running Mean / Std tracker
# ---------------------------------------------------------------------------
class RunningMeanStd:
    """Track running mean and variance using Welford's online algorithm.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the statistic to track.  Use ``()`` for scalar (reward),
        or ``(H, W)`` for per-pixel observation normalisation.
    """

    def __init__(self, shape: tuple[int, ...] = ()) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count: float = 0.0

    def update(self, batch: np.ndarray) -> None:
        """Update statistics with a new batch of data.

        Parameters
        ----------
        batch : np.ndarray
            Array where the first axis is the batch dimension.
            For scalar shape, ``batch`` has shape ``(N,)``.
            For image shape, ``batch`` has shape ``(N, H, W)``.
        """
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == 0:
            batch = batch.reshape(1)
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: float,
    ) -> None:
        """Parallel Welford update from batch moments."""
        if self.count == 0:
            self.mean = batch_mean
            self.var = batch_var
            self.count = batch_count
            return

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count


# ---------------------------------------------------------------------------
# RND Networks
# ---------------------------------------------------------------------------
def _compute_conv_output_size(
    obs_shape: tuple[int, ...],
) -> int:
    """Compute the flattened output size of the 3-conv backbone.

    Parameters
    ----------
    obs_shape : tuple[int, ...]
        Observation shape in CHW format, e.g. ``(1, 84, 84)``.

    Returns
    -------
    int
        Number of features after the convolutional layers.
    """
    c, h, w = obs_shape
    # Conv1: 8x8, stride 4
    h = (h - 8) // 4 + 1
    w = (w - 8) // 4 + 1
    # Conv2: 4x4, stride 2
    h = (h - 4) // 2 + 1
    w = (w - 4) // 2 + 1
    # Conv3: 3x3, stride 1
    h = (h - 3) // 1 + 1
    w = (w - 3) // 1 + 1
    return 64 * h * w


class RNDTargetNetwork(nn.Module):
    """Fixed random CNN target network for RND.

    Architecture: 3 conv layers (32/64/64 filters) -> flatten -> linear
    to ``embedding_dim``.  All parameters are frozen (``requires_grad=False``).

    Parameters
    ----------
    obs_shape : tuple[int, ...]
        Observation shape in CHW format, e.g. ``(1, 84, 84)``.
    embedding_dim : int
        Output embedding dimensionality.
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...] = (1, 84, 84),
        embedding_dim: int = 512,
    ) -> None:
        super().__init__()
        c = obs_shape[0]
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        conv_out_size = _compute_conv_output_size(obs_shape)
        self.fc = nn.Linear(conv_out_size, embedding_dim)

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute fixed embedding for observations.

        Parameters
        ----------
        x : torch.Tensor
            Observations of shape ``(batch, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Embeddings of shape ``(batch, embedding_dim)``.
        """
        return self.fc(self.conv(x))


class RNDPredictorNetwork(nn.Module):
    """Trainable predictor network for RND.

    Same convolutional backbone as the target, plus a deeper FC head
    (3 additional linear layers with ReLU).

    Parameters
    ----------
    obs_shape : tuple[int, ...]
        Observation shape in CHW format, e.g. ``(1, 84, 84)``.
    embedding_dim : int
        Output embedding dimensionality (must match target).
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...] = (1, 84, 84),
        embedding_dim: int = 512,
    ) -> None:
        super().__init__()
        c = obs_shape[0]
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        conv_out_size = _compute_conv_output_size(obs_shape)
        self.head = nn.Sequential(
            nn.Linear(conv_out_size, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute predicted embedding for observations.

        Parameters
        ----------
        x : torch.Tensor
            Observations of shape ``(batch, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Predicted embeddings of shape ``(batch, embedding_dim)``.
        """
        return self.head(self.conv(x))


# ---------------------------------------------------------------------------
# VecEnv Wrapper
# ---------------------------------------------------------------------------
class RNDRewardWrapper(VecEnvWrapper):
    """Add RND intrinsic reward bonus to environment rewards.

    Wraps a ``VecEnv`` and modifies the reward in ``step_wait()`` by adding
    a normalised intrinsic novelty bonus computed via Random Network
    Distillation.

    Parameters
    ----------
    venv : VecEnv
        The vectorised environment to wrap.  Observations must be in CHW
        format (e.g. after ``VecTransposeImage``).
    int_coeff : float
        Weight for intrinsic reward.  Default 1.0.
    ext_coeff : float
        Weight for extrinsic reward.  Default 2.0.
    embedding_dim : int
        RND embedding dimensionality.  Default 512.
    predictor_lr : float
        Learning rate for the predictor network.  Default 1e-3.
    update_proportion : float
        Fraction of each batch used for predictor updates.  Default 0.25.
    device : str
        PyTorch device.  Default ``"cpu"``.
    """

    def __init__(
        self,
        venv: VecEnv,
        int_coeff: float = 1.0,
        ext_coeff: float = 2.0,
        embedding_dim: int = 512,
        predictor_lr: float = 1e-3,
        update_proportion: float = 0.25,
        device: str = "cpu",
    ) -> None:
        super().__init__(venv)

        self.int_coeff = int_coeff
        self.ext_coeff = ext_coeff
        self.embedding_dim = embedding_dim
        if not (0.0 < update_proportion <= 1.0):
            raise ValueError(
                f"update_proportion must be in (0.0, 1.0], got {update_proportion}"
            )
        self.update_proportion = update_proportion
        self.device = torch.device(device)

        # Infer observation shape from the VecEnv
        obs_shape = venv.observation_space.shape
        if obs_shape is None or len(obs_shape) != 3:
            raise ValueError(
                f"RNDRewardWrapper requires 3D (CHW) observations, got {obs_shape}"
            )

        # Build networks
        self.target_network = RNDTargetNetwork(
            obs_shape=obs_shape, embedding_dim=embedding_dim
        ).to(self.device)
        self.predictor_network = RNDPredictorNetwork(
            obs_shape=obs_shape, embedding_dim=embedding_dim
        ).to(self.device)

        self.target_network.eval()

        self.optimizer = torch.optim.Adam(
            self.predictor_network.parameters(), lr=predictor_lr
        )

        # Normalisation trackers (non-episodic)
        # Obs normalisation: per-pixel running mean/std
        # Use the spatial dims (H, W) for normalisation, broadcast over channels
        self.obs_rms = RunningMeanStd(shape=obs_shape[1:])  # (H, W)
        # Reward normalisation: scalar running variance
        self.reward_rms = RunningMeanStd(shape=())

        # Discounted return tracker for reward normalisation
        self._int_return = np.zeros(venv.num_envs, dtype=np.float64)
        self._gamma_int = 0.99

    def reset(self) -> np.ndarray:
        """Reset the wrapped environment.

        Returns
        -------
        np.ndarray
            Initial observations.
        """
        obs = self.venv.reset()
        return obs

    def step_wait(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Wait for step to complete, adding RND intrinsic reward.

        Returns
        -------
        tuple
            ``(obs, combined_reward, dones, infos)`` where
            ``combined_reward = ext_coeff * extrinsic + int_coeff * normalised_intrinsic``.
        """
        obs, extrinsic_reward, dones, infos = self.venv.step_wait()

        # Update obs normalisation statistics
        # obs shape: (n_envs, C, H, W) — normalise per pixel across batch
        obs_for_rms = obs.mean(axis=1)  # (n_envs, H, W) — average over channels
        self.obs_rms.update(obs_for_rms)

        # Normalise observations once for both reward computation and training
        obs_tensor = self._normalise_obs(obs)

        # Compute target embeddings (shared across reward and update)
        with torch.no_grad():
            target_embed = self.target_network(obs_tensor)

        # Compute intrinsic reward using detached predictor output
        with torch.no_grad():
            predicted_embed_reward = self.predictor_network(obs_tensor)
        intrinsic_reward = (
            ((target_embed - predicted_embed_reward) ** 2).mean(dim=1).cpu().numpy()
        )

        # Normalise intrinsic reward using non-episodic discounted return
        self._int_return = self._int_return * self._gamma_int + intrinsic_reward
        self.reward_rms.update(self._int_return)
        reward_std = np.sqrt(self.reward_rms.var + 1e-8)
        normalised_intrinsic = intrinsic_reward / reward_std

        # Combine rewards
        combined_reward = (
            self.ext_coeff * extrinsic_reward + self.int_coeff * normalised_intrinsic
        )

        # Update predictor network (subsample, single forward pass)
        self._update_predictor_with_targets(obs_tensor, target_embed)

        # Store raw intrinsic reward in info for logging
        for i, info in enumerate(infos):
            info["rnd_intrinsic_reward"] = float(intrinsic_reward[i])
            info["rnd_normalised_intrinsic"] = float(normalised_intrinsic[i])

        return obs, combined_reward.astype(np.float32), dones, infos

    def _normalise_obs(self, obs: np.ndarray) -> torch.Tensor:
        """Normalise observations using running mean/std and clip to [-5, 5].

        Parameters
        ----------
        obs : np.ndarray
            Raw observations of shape ``(batch, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Normalised observations on ``self.device``.
        """
        obs = obs.astype(np.float32, copy=False)
        # Normalise per pixel: subtract mean, divide by std
        # obs_rms tracks (H, W); broadcast over (batch, C, H, W)
        mean = self.obs_rms.mean.astype(np.float32)
        std = np.sqrt(self.obs_rms.var + 1e-8).astype(np.float32)
        normalised = (obs - mean) / std
        normalised = np.clip(normalised, -5.0, 5.0)
        return torch.from_numpy(normalised).to(self.device)

    @torch.no_grad()
    def _compute_intrinsic_reward(self, obs: np.ndarray) -> np.ndarray:
        """Compute RND intrinsic reward (MSE prediction error).

        Parameters
        ----------
        obs : np.ndarray
            Observations of shape ``(n_envs, C, H, W)``.

        Returns
        -------
        np.ndarray
            Intrinsic rewards of shape ``(n_envs,)``.
        """
        obs_tensor = self._normalise_obs(obs)
        target_embed = self.target_network(obs_tensor)
        predicted_embed = self.predictor_network(obs_tensor)
        # Per-sample MSE
        intrinsic = ((target_embed - predicted_embed) ** 2).mean(dim=1)
        return intrinsic.cpu().numpy()

    def _update_predictor(self, obs: np.ndarray) -> None:
        """Train the predictor network on a batch of observations.

        Uses ``update_proportion`` to subsample the batch.

        Parameters
        ----------
        obs : np.ndarray
            Observations of shape ``(n_envs, C, H, W)``.
        """
        n = obs.shape[0]
        n_update = max(1, int(n * self.update_proportion))

        # Random subsample
        indices = np.random.choice(n, size=n_update, replace=False)
        obs_batch = obs[indices]

        obs_tensor = self._normalise_obs(obs_batch)

        with torch.no_grad():
            target_embed = self.target_network(obs_tensor)

        predicted_embed = self.predictor_network(obs_tensor)
        loss = ((target_embed - predicted_embed) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _update_predictor_with_targets(
        self,
        obs_tensor: torch.Tensor,
        target_embed: torch.Tensor,
    ) -> None:
        """Train the predictor using pre-computed normalised obs and targets.

        Avoids redundant forward passes when called from ``step_wait()``,
        which already computed target embeddings for the intrinsic reward.

        Parameters
        ----------
        obs_tensor : torch.Tensor
            Already-normalised observations on ``self.device``.
        target_embed : torch.Tensor
            Pre-computed target embeddings (detached).
        """
        n = obs_tensor.shape[0]
        n_update = max(1, int(n * self.update_proportion))

        # Random subsample
        indices = np.random.choice(n, size=n_update, replace=False)
        obs_batch = obs_tensor[indices]
        target_batch = target_embed[indices]

        predicted_embed = self.predictor_network(obs_batch)
        loss = ((target_batch - predicted_embed) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_rnd_state(self) -> dict[str, Any]:
        """Get RND state for checkpointing.

        Returns
        -------
        dict
            Contains predictor weights, normalisation statistics.
        """
        return {
            "predictor_state_dict": self.predictor_network.state_dict(),
            "obs_rms": {
                "mean": self.obs_rms.mean.copy(),
                "var": self.obs_rms.var.copy(),
                "count": self.obs_rms.count,
            },
            "reward_rms": {
                "mean": self.reward_rms.mean.copy(),
                "var": self.reward_rms.var.copy(),
                "count": self.reward_rms.count,
            },
            "int_return": self._int_return.copy(),
        }

    def load_rnd_state(self, state: dict[str, Any]) -> None:
        """Load RND state from a checkpoint.

        Parameters
        ----------
        state : dict
            State dict from ``get_rnd_state()``.
        """
        self.predictor_network.load_state_dict(state["predictor_state_dict"])

        obs_rms_data = state["obs_rms"]
        self.obs_rms.mean = obs_rms_data["mean"].copy()
        self.obs_rms.var = obs_rms_data["var"].copy()
        self.obs_rms.count = obs_rms_data["count"]

        reward_rms_data = state["reward_rms"]
        self.reward_rms.mean = reward_rms_data["mean"].copy()
        self.reward_rms.var = reward_rms_data["var"].copy()
        self.reward_rms.count = reward_rms_data["count"]

        if "int_return" in state:
            self._int_return = state["int_return"].copy()
