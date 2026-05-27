# Copyright (c) 2025, Unitree Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""AMP (Adversarial Motion Priors) training stack for Unitree RL Lab.

Drop-in replacement for the standard ``rsl_rl`` on-policy training pipeline
that adds a discriminator-based style reward on top of PPO.

Top-level exports:

- :class:`AmpPPO`: PPO variant with AMP.
- :class:`OnPolicyAmpRunner`: runner mirroring ``rsl_rl.runners.OnPolicyRunner``.
- :class:`AmpDiscriminator`: MLP discriminator.
- :class:`AmpRolloutStorage`: rollout storage with AMP extras.
- :class:`MotionDataset`: expert motion loader / sampler.
- :class:`AmpObsSpec` / :class:`AmpObsState`: unified feature layout.
- :func:`build_amp_frame_from_state` / :func:`build_amp_window`: single-source
  feature constructors shared between env and dataset.
"""

from __future__ import annotations

from .algorithms.amp_ppo import AmpPPO, AMP_OBS_SET_NAME
from .features import (
    AmpObsSpec,
    AmpObsState,
    build_amp_frame_from_state,
    build_amp_window,
    concat_frame_history,
)
from .modules.amp_discriminator import AmpDiscriminator
from .runners.on_policy_amp_runner import OnPolicyAmpRunner
from .storage.amp_rollout_storage import AmpRolloutStorage
from .storage.motion_dataset import MotionDataset, MotionResetPayload

__all__ = [
    "AMP_OBS_SET_NAME",
    "AmpDiscriminator",
    "AmpObsSpec",
    "AmpObsState",
    "AmpPPO",
    "AmpRolloutStorage",
    "MotionDataset",
    "MotionResetPayload",
    "OnPolicyAmpRunner",
    "build_amp_frame_from_state",
    "build_amp_window",
    "concat_frame_history",
]
