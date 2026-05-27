# Copyright (c) 2025, Unitree Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""AMP-specific observation helpers."""

from __future__ import annotations

import torch


def amp_obs(env) -> torch.Tensor:
    """Expose the stacked AMP observation maintained by :class:`UnitreeAmpEnv`.

    The env owns a ``(num_envs, stack_k, frame_dim)`` circular buffer and a
    ``_flat_amp_obs()`` helper that flattens it to ``(num_envs, amp_obs_dim)``.
    During ObservationManager setup (inside ``super().__init__()``) the buffer
    does not exist yet, so fall back to a zeros tensor sized off the spec —
    the real value is produced once ``UnitreeAmpEnv.__init__`` finishes.

    ``env.cfg.amp_spec`` must already be set by the concrete env cfg's
    ``__post_init__``; leaving it ``None`` is a hard error.
    """
    buf = getattr(env, "_amp_history_buf", None)
    if buf is None:
        spec = env.cfg.amp_spec
        if spec is None:
            raise ValueError(
                "env.cfg.amp_spec is None during ObservationManager setup — "
                "the env cfg's __post_init__ must construct an AmpObsSpec."
            )
        return torch.zeros(env.num_envs, spec.amp_obs_dim, device=env.device)
    return env._flat_amp_obs()
