# Copyright (c) 2025, Unitree Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""AMP (Adversarial Motion Priors) task package.

Environments here expose an additional ``amp`` observation group used by the
AMP discriminator, together with ``amp_spec`` / ``amp_dataset_cfg`` metadata
attached to the env cfg. Training is driven by :class:`OnPolicyAmpRunner`
from :mod:`unitree_rl_lab.rsl_rl_amp`.
"""
