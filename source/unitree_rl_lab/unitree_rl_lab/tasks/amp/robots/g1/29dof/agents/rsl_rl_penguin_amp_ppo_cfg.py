# Copyright (c) 2025, Unitree Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""RSL-RL agent cfg for the Penguin AMP PPO task on G1 23-DoF.

Inherits the general :class:`AmpPPORunnerCfg` and only overrides the
``experiment_name`` so logs/checkpoints land under a dedicated directory
(``logs/rsl_rl/g1_23dof_penguin_amp/...``) instead of the generic AMP
velocity run. All AMP-specific hyperparameters, discriminator architecture,
and the only-up AMP reward curriculum (α_amp: 0.2 → 0.8) are kept as-is —
they are the Penguin V1 baseline.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .rsl_rl_amp_ppo_cfg import AmpPPORunnerCfg


@configclass
class PenguinAmpPPORunnerCfg(AmpPPORunnerCfg):
    """Runner cfg for the ``Unitree-G1-23dof-Penguin-AMP`` task."""

    experiment_name: str = "g1_23dof_penguin_amp"
