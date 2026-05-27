# Copyright (c) 2025, Unitree Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""AMP-specific environment subclasses.

The :class:`UnitreeAmpEnv` subclass of :class:`ManagerBasedRLEnv` maintains the
AMP K-frame circular buffer, injects ``obs["amp"]`` into the observation dict,
captures the pre-reset AMP observation for done envs (the "terminal next
``amp_obs_{t+1}`` fix"), and implements the Phase-1 motion reset strategy.
"""

from __future__ import annotations

from .unitree_amp_env import (
    UnitreeAmpEnv,
    UnitreeAmpEnvCfg,
    UnitreeAmpResetCfg,
)

__all__ = [
    "UnitreeAmpEnv",
    "UnitreeAmpEnvCfg",
    "UnitreeAmpResetCfg",
]
