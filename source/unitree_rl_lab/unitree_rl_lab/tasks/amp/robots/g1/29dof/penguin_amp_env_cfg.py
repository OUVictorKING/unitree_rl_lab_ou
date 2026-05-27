# Copyright (c) 2025, Unitree Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""Penguin-style AMP env cfg for G1 23-DoF.

Inherits the general "AMP velocity baseline" (:mod:`velocity_amp_env_cfg`) and
overrides the bits that are specific to the *penguin* style:

1. Reward scales: weaken the three terms that smother waddling
   (``flat_orientation_l2``, ``base_height``, ``joint_deviation_waists``) to
   "medium" levels so the discriminator has room to reward the style while
   stability is still enforced.
2. Command range: expert clip is pure forward locomotion, so the initial
   ``lin_vel_x`` range is positive-only, ``lin_vel_y`` is pinned to zero, and
   the ``ang_vel_z`` range is tightened/expanded for low-speed turning.
3. AMP dataset / spec / reward mixing: unchanged — inherited from the parent
   (Penguin V1 spec, :file:`g1_qie_motion.npz`, only-up AMP curriculum
   α_amp: 0.2 → 0.8). No gait-clock / contact-schedule / feet-air-time phase
   rewards exist anywhere along this inheritance chain.

Task registration:
    gym id               : ``Unitree-G1-23dof-Penguin-AMP``
    env cfg entry point  : ``...penguin_amp_env_cfg:RobotEnvCfg``
    play cfg entry point : ``...penguin_amp_env_cfg:RobotPlayEnvCfg``
    agent cfg entry point: ``...agents.rsl_rl_penguin_amp_ppo_cfg:PenguinAmpPPORunnerCfg``
"""

from __future__ import annotations

from isaaclab.utils import configclass

from unitree_rl_lab.tasks.amp import mdp

from .velocity_amp_env_cfg import RobotEnvCfg as _VelocityAmpEnvCfg


@configclass
class RobotEnvCfg(_VelocityAmpEnvCfg):
    """Penguin-style AMP env — overrides waddling-suppressing reward scales
    and the command range to match the expert clip (pure forward, low speed).
    """

    def __post_init__(self):
        super().__post_init__()

        # ================================================================
        # 1) Reward scales — "medium" waddling-friendly setting
        # ================================================================
        # Rationale (see audit / AskUserQuestion decision):
        # - flat_orientation_l2 -5.0 was the single largest blocker of
        #   pelvis roll. Drop to -0.5 to still discourage face-plants but
        #   let the discriminator reward the natural side sway.
        # - base_height -10.0 @ target 0.78 heavily penalized the small
        #   vertical bob that accompanies waddling. -2.0 keeps the robot
        #   near standing height without flattening the gait.
        # - joint_deviation_waists -0.5 froze the waist; weakening to
        #   -0.05 lets the torso contribute to the side sway.
        # Everything else (tracking, stabilization, safety, smoothness)
        # is left at the parent values — only the three suppressors move.
        self.rewards.flat_orientation_l2.weight = -0.5
        self.rewards.base_height.weight = -2.0
        self.rewards.joint_deviation_waists.weight = -0.05

        # ================================================================
        # 2) Commands — positive-only low-speed forward + modest yaw
        # ================================================================
        # Expert clip (g1_qie_motion.npz) is pure forward walking, so the
        # initial range must not contain backward or lateral commands (AMP
        # style reward and task reward would fight otherwise). The
        # UniformLevelVelocityCommandCfg curriculum keeps the initial
        # ``ranges`` and expands toward ``limit_ranges`` as training
        # progresses.
        self.commands.base_velocity.ranges = (
            mdp.UniformLevelVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 0.30),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-0.3, 0.3),
            )
        )
        self.commands.base_velocity.limit_ranges = (
            mdp.UniformLevelVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 0.50),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-0.6, 0.6),
            )
        )


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    """Play / export variant.

    Inherits from the penguin :class:`RobotEnvCfg` (so penguin reward +
    command overrides are already applied) and then pins the scene to a
    single env, disables corruption / push / curriculum, and locks the
    command to a modest forward speed near the expert clip's mean.
    """

    def __post_init__(self):
        super().__post_init__()

        # Single-env rollout + clean init pose.
        self.scene.num_envs = 1
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.8)

        # Turn off the things that should not run at play/export time.
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.curriculum = None

        # Pinned command: modest forward speed + no yaw — close to the
        # penguin clip's natural motion so the exported rollout looks like
        # the trained behavior.
        self.commands.base_velocity.ranges.lin_vel_x = (0.25, 0.25)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
