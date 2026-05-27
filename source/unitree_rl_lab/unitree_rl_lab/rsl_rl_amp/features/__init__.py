# Copyright (c) 2025, Unitree Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""Unified AMP feature construction.

Single-source-of-truth for the AMP observation layout. Both the simulation
side (``unitree_rl_lab.tasks.amp.env``) and the dataset side
(``unitree_rl_lab.rsl_rl_amp.storage.motion_dataset.MotionDataset``) call
into this module to build per-frame style features and the stacked
``amp_obs`` vector consumed by the discriminator.

Per-task AMP specs (V1 / V2 / V3) are constructed inline in each task's
env cfg; this module only provides the shared types and helpers.
"""

from __future__ import annotations

from .amp_features import (
    AmpObsSpec,
    AmpObsState,
    build_amp_frame_from_state,
    build_amp_window,
    concat_frame_history,
    resolve_indices,
    resolve_spec_indices,
)

__all__ = [
    "AmpObsSpec",
    "AmpObsState",
    "build_amp_frame_from_state",
    "build_amp_window",
    "concat_frame_history",
    "resolve_indices",
    "resolve_spec_indices",
]
