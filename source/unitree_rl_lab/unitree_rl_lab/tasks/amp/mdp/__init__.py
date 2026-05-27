# Copyright (c) 2025, Unitree Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""MDP terms for AMP tasks.

The Basic AMP baseline reuses locomotion MDP terms verbatim (observations,
rewards, commands, curriculums). This module re-exports them so env cfgs can
write ``from unitree_rl_lab.tasks.amp import mdp`` symmetrically with the
locomotion package, and leaves room for AMP-specific overrides later without
touching call sites.
"""

from unitree_rl_lab.tasks.locomotion.mdp import *  # noqa: F401,F403

from .observations import amp_obs  # noqa: F401
from .rewards import (  # noqa: F401
    feet_flat_orientation_l2,
    flat_pitch_l2,
    stand_still_on_ground,
)
