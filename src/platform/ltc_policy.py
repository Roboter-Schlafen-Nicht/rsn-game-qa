"""CfC-based recurrent policy for CNN observations (LTC module).

Provides ``CnnCfCPolicy``, a drop-in replacement for sb3-contrib's
``RecurrentActorCriticCnnPolicy`` that swaps the LSTM cells with
CfC (Closed-form Continuous-depth) cells from the ``ncps`` library.

The CfC cell is a Liquid Time-Constant (LTC) network variant that
provides richer temporal representations than frame stacking (4-frame
``VecFrameStack``) while being computationally efficient.

Architecture::

    NatureCNN(84x84x1) → CfCAdapter(512 → hidden) → MLP → action/value

Key design: ``CfCAdapter`` wraps a ``CfC`` cell and exposes LSTM-
compatible attributes (``num_layers``, ``hidden_size``, ``input_size``)
so that ``RecurrentPPO._setup_model()`` can initialize buffers without
modification.

Usage::

    from sb3_contrib import RecurrentPPO
    from src.platform.ltc_policy import CnnCfCPolicy

    model = RecurrentPPO(
        CnnCfCPolicy,
        vec_env,
        policy_kwargs={"lstm_hidden_size": 64},
        device="xpu:1",
    )
    model.learn(total_timesteps=100_000)
"""

from __future__ import annotations

import logging
from typing import Any

import torch as th
from gymnasium import spaces
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import zip_strict
from torch import nn

logger = logging.getLogger(__name__)


class CfCAdapter(nn.Module):
    """Thin adapter wrapping a ``CfC`` cell with LSTM-compatible interface.

    ``RecurrentPPO._setup_model()`` accesses ``lstm_actor.num_layers``
    and ``lstm_actor.hidden_size`` to initialize state buffers.  This
    adapter exposes those attributes while delegating computation to
    the underlying CfC cell.

    **State convention:**  CfC uses a single hidden state tensor of
    shape ``(batch, state_size)``.  To satisfy the ``(h, c)`` tuple
    interface expected by ``RecurrentPPO``, this adapter:

    - On input: uses only ``h`` from the ``(h, c)`` tuple (``c`` is
      ignored).
    - On output: returns ``(h_new, zeros)`` as the state tuple.

    Parameters
    ----------
    input_size : int
        Dimensionality of input features (e.g. 512 from NatureCNN).
    hidden_size : int
        Number of hidden units in the CfC cell.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        from ncps.torch import CfC

        self._cfc = CfC(
            input_size=input_size,
            units=hidden_size,
            batch_first=True,
        )
        self._input_size = input_size
        self._hidden_size = hidden_size

    # -- LSTM-compatible attributes for RecurrentPPO._setup_model() ----

    @property
    def num_layers(self) -> int:
        """Number of recurrent layers (always 1 for CfC)."""
        return 1

    @property
    def hidden_size(self) -> int:
        """Hidden state dimensionality."""
        return self._hidden_size

    @property
    def input_size(self) -> int:
        """Input feature dimensionality."""
        return self._input_size

    def forward(
        self,
        x: th.Tensor,
        hx: tuple[th.Tensor, th.Tensor],
    ) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        """Forward pass through the CfC cell.

        Parameters
        ----------
        x : th.Tensor
            Input tensor of shape ``(seq_len, batch, input_size)``.
            Note: LSTM convention is seq-first when ``batch_first=False``.
        hx : tuple[th.Tensor, th.Tensor]
            Previous ``(h, c)`` states.  ``h`` has shape
            ``(1, batch, hidden_size)``; ``c`` is ignored.

        Returns
        -------
        output : th.Tensor
            CfC output of shape ``(seq_len, batch, hidden_size)``.
        (h_new, c_new) : tuple[th.Tensor, th.Tensor]
            Updated states.  ``c_new`` is zeros.
        """
        h = hx[0]  # (1, batch, hidden)

        # CfC expects batch_first: (batch, seq, features)
        # Input is seq-first: (seq, batch, features)
        x_bf = x.transpose(0, 1)  # (batch, seq, features)

        # Squeeze h from (1, batch, hidden) -> (batch, hidden)
        h_squeezed = h.squeeze(0)

        # CfC forward
        output, h_new = self._cfc(x_bf, h_squeezed)

        # Back to seq-first: (batch, seq, hidden) -> (seq, batch, hidden)
        output_sf = output.transpose(0, 1)

        # Unsqueeze h_new: (batch, hidden) -> (1, batch, hidden)
        h_new = h_new.unsqueeze(0)

        # Return LSTM-compatible (h, c) tuple with c = zeros
        c_new = th.zeros_like(h_new)

        return output_sf, (h_new, c_new)


class CnnCfCPolicy(RecurrentActorCriticPolicy):
    """CNN + CfC recurrent policy for ``RecurrentPPO``.

    Subclasses ``RecurrentActorCriticPolicy`` and replaces the LSTM
    cells with ``CfCAdapter`` instances.  Uses ``NatureCNN`` as the
    feature extractor (standard for 84x84 image observations).

    Parameters
    ----------
    observation_space : spaces.Space
        Must be a Box with shape ``(C, H, W)`` (e.g. ``(1, 84, 84)``).
    action_space : spaces.Space
        Action space of the environment.
    lr_schedule : Schedule
        Learning rate schedule.
    lstm_hidden_size : int
        Number of hidden units for the CfC cell.  Default is 256.
    enable_critic_lstm : bool
        Whether to use a separate CfC cell for the critic.
        Default is ``True``.
    shared_lstm : bool
        Whether to share the CfC cell between actor and critic.
        Default is ``False``.
    **kwargs
        Additional keyword arguments passed to
        ``RecurrentActorCriticPolicy``.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type = NatureCNN,
        features_extractor_kwargs: dict[str, Any] | None = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        lstm_kwargs: dict[str, Any] | None = None,
    ):
        # Parent __init__ creates nn.LSTM instances -- we'll replace them
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            lstm_hidden_size,
            n_lstm_layers,
            shared_lstm,
            enable_critic_lstm,
            lstm_kwargs,
        )

        # Replace LSTM actor with CfC adapter
        self.lstm_actor = CfCAdapter(self.features_dim, lstm_hidden_size)

        # Replace LSTM critic (if enabled) with CfC adapter
        if self.enable_critic_lstm and self.lstm_critic is not None:
            self.lstm_critic = CfCAdapter(self.features_dim, lstm_hidden_size)

        # Re-create optimizer to include CfC parameters
        # (parent __init__ created optimizer with LSTM params)
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

        logger.info(
            "CnnCfCPolicy: features_dim=%d, cfc_hidden=%d, shared=%s, critic_cfc=%s",
            self.features_dim,
            lstm_hidden_size,
            shared_lstm,
            enable_critic_lstm,
        )

    @staticmethod
    def _process_sequence(
        features: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        lstm: nn.Module,
    ) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        """Forward pass through the CfC cell with episode reset handling.

        This replaces the parent's LSTM-specific ``_process_sequence``.
        The logic mirrors the parent but adapts to CfC's single-state
        semantics (the ``(h, c)`` tuple is maintained for interface
        compatibility, but ``c`` is always zeros).

        Parameters
        ----------
        features : th.Tensor
            Input features from the CNN extractor.
        lstm_states : tuple[th.Tensor, th.Tensor]
            Previous ``(h, c)`` states.
        episode_starts : th.Tensor
            Binary mask indicating new episode starts.
        lstm : nn.Module
            The ``CfCAdapter`` (or ``nn.LSTM`` for fallback).

        Returns
        -------
        output : th.Tensor
            Sequence output, flattened to ``(batch_size, hidden_size)``.
        new_states : tuple[th.Tensor, th.Tensor]
            Updated ``(h, c)`` states.
        """
        n_seq = lstm_states[0].shape[1]

        # Reshape: (padded_batch, features) -> (max_len, n_seq, features)
        features_sequence = features.reshape((n_seq, -1, lstm.input_size)).swapaxes(0, 1)
        episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)

        # Fast path: no episode resets in the sequence
        if th.all(episode_starts == 0.0):
            output, new_states = lstm(features_sequence, lstm_states)
            output = th.flatten(output.transpose(0, 1), start_dim=0, end_dim=1)
            return output, new_states

        # Slow path: step through sequence handling episode resets
        rnn_output = []
        for features_t, episode_start in zip_strict(features_sequence, episode_starts):
            # Reset states at episode boundaries
            hidden, lstm_states = lstm(
                features_t.unsqueeze(dim=0),
                (
                    (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[0],
                    (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[1],
                ),
            )
            rnn_output.append(hidden)

        # (seq_len, n_seq, hidden) -> (batch_size, hidden)
        output = th.flatten(th.cat(rnn_output).transpose(0, 1), start_dim=0, end_dim=1)
        return output, lstm_states
