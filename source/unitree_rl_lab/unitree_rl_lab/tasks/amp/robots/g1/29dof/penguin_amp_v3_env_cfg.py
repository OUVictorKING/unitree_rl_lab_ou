# Copyright (c) 2025, Unitree Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""Penguin AMP V3 env cfg for G1 23-DoF — self-contained, no chain through V1/V2.

The V3 env cfg is intentionally **flat** — every knob that V1 penguin and
V2 smoothness-softening layered on top of the AMP velocity baseline is
re-declared here with its effective value, so a reader can see the full
V3 training recipe in one file (mirroring the flattened V3 agent cfg).

Values are identical to what the old chain ``V3 → V2 → V1 → velocity``
produced; only the organisation changed. Fields that are unchanged across
V1/V2/V3 (scene / events / actions / observations / terminations /
curriculum) are still imported from :mod:`velocity_amp_env_cfg` — they are
plain environment plumbing, not Penguin-specific knobs.

Effective overrides vs the velocity AMP baseline (all made visible below):
    - commands.base_velocity.ranges / limit_ranges — V1 penguin
    - rewards.flat_orientation_l2.weight        -5.0  → -0.5   (V1)
    - rewards.base_height.weight               -10.0  → -2.0   (V1)
    - rewards.joint_deviation_waists.weight     -0.5  → -0.05  (V1)
    - rewards.action_rate.weight               -0.04  → -0.01  (V2)
    - rewards.base_linear_velocity.weight       -1.5  → -1.0   (V2)
    - rewards.alive.weight                      0.15  → 0.5    (Step 3: survival)
    - rewards.base_angular_velocity.weight     -0.06  → -0.15  (Step 3: tip guard)
    - rewards.termination_penalty               (new) -200.0   (Step 3: hard fall cost)
    - rewards.joint_deviation_legs              (new) -0.2     (Step 3: hip yaw/roll anchor)
    - rewards.feet_air_time                     (new)  0.25    (Step 3: stable stepping)
    - amp_spec                                  V1 80-dim → V3 58-dim (pose-only)
    - amp_motion_files                          original clip only (no augment)
    - amp_reset.use_motion_reset                True  → False  (Step 1: no RSI)

Task registration:
    gym id               : ``Unitree-G1-23dof-Penguin-AMP-V3``
    env cfg entry point  : ``...penguin_amp_v3_env_cfg:RobotEnvCfg``
    play cfg entry point : ``...penguin_amp_v3_env_cfg:RobotPlayEnvCfg``
    agent cfg entry point: ``...agents.rsl_rl_penguin_amp_v3_ppo_cfg:PenguinAmpV3PPORunnerCfg``
"""

from __future__ import annotations

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.rsl_rl_amp.features import AmpObsSpec
from unitree_rl_lab.tasks.amp import mdp

from .velocity_amp_env_cfg import (
    DEFAULT_PENGUIN_MOTION_FILES,
    ActionsCfg,
    CurriculumCfg,
    EventCfg,
    ObservationsCfg,
    RobotSceneCfg,
    TerminationsCfg,
)
from .velocity_amp_env_cfg import RobotEnvCfg as _VelocityAmpEnvCfg


# G1 23-DoF articulation joint order — copied per env cfg so each task's
# AMP spec is self-contained. Keep in sync with the robot articulation.
G1_23DOF_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
)


# =============================================================================
# Commands — V1 penguin ranges (positive-only forward + modest yaw)
# =============================================================================
# Expert clip (g1_qie_motion.npz) is pure forward walking, so the initial
# ``ranges`` must not contain backward or lateral commands (AMP style
# reward and task reward would fight otherwise). The UniformLevelVelocity
# curriculum starts narrow and expands toward ``limit_ranges`` as
# training progresses.
@configclass
class CommandsCfg:
    """V3 commands — identical to V1 penguin."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.30),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.3, 0.3),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.50),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.6, 0.6),
        ),
    )


# =============================================================================
# Rewards — baseline + V1 penguin overrides + V2 smoothness softening + Step 3 robustness
# =============================================================================
# Every weight is the *final effective* value used during training.
# Tags in the trailing comments note where each override originated:
#   (V1) = penguin_amp_env_cfg softening (waddle room: orientation / height / waists)
#   (V2) = penguin_amp_v2_env_cfg softening (smoothness: action_rate / lin_vel_z)
#   (S3) = Step 3 robustness shaping, added after the 16-28-43 run showed
#          D-saturation + ~10% bad_orientation termination rate. Only terms
#          that are orthogonal to the waddle style are touched — no new
#          pitch / height / waist constraints are tightened.
# Everything else is carried straight from the velocity AMP baseline.
@configclass
class RewardsCfg:
    """V3 rewards — every term's final weight explicit, no inheritance chain."""

    # -- task tracking
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    alive = RewTerm(func=mdp.is_alive, weight=0.5)                           # S3: 0.15 → 0.5

    # -- fall penalty: fires only on non-timeout termination (bad_orientation
    #    or base_height). Large negative weight to make "don't fall" the
    #    dominant per-episode signal, independent of per-step posture terms.
    termination_penalty = RewTerm(                                           # S3: new
        func=mdp.is_terminated_term,
        weight=-200.0,
        params={"term_keys": ["bad_orientation", "base_height"]},
    )

    # -- base stabilization
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)       # V2: -1.5 → -1.0
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.15)    # S3: -0.06 → -0.15
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)             # V2: -0.04 → -0.01
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    # -- posture (light deviation only; bulk style comes from AMP)
    joint_deviation_waists = RewTerm(                                        # V1: -0.5 → -0.05
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"])},
    )
    # Anchor hip yaw / hip roll to default — expert waddle is pitch-driven,
    # so yaw/roll drift is almost always a fall precursor rather than style.
    joint_deviation_legs = RewTerm(                                          # S3: new
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg(
            "robot", joint_names=[".*hip_yaw_joint", ".*hip_roll_joint"])},
    )

    # -- body pose
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5) # V1: -5.0 → -0.5
    base_height = RewTerm(                                                   # V1: -10.0 → -2.0
        func=mdp.base_height_l2, weight=-2.0, params={"target_height": 0.78}
    )

    # -- feet
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    # Reward sustained single-foot contact above ``threshold`` seconds while
    # a non-zero command is active. Promotes a stable gait without dictating
    # the specific waddle shape.
    feet_air_time = RewTerm(                                                 # S3: new
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "threshold": 0.4,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    # -- safety
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["(?!.*ankle.*).*"]
            ),
        },
    )


# =============================================================================
# Training env cfg
# =============================================================================
@configclass
class RobotEnvCfg(_VelocityAmpEnvCfg):
    """V3 env cfg — flat inheritance: straight from the AMP velocity baseline.

    Binds the V3 CommandsCfg + RewardsCfg defined above, reuses unchanged
    plumbing from :mod:`velocity_amp_env_cfg`, and pins the AMP spec to
    the pose-only 58-dim variant.
    """

    # -- unchanged plumbing (imported verbatim from velocity AMP baseline)
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=256, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # -- V3-specific (defined inline above with every value visible)
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        # Pin AMP hooks BEFORE calling super().__post_init__, so the
        # baseline's ``if self.amp_spec is None: ...`` defaulting branch
        # does nothing.
        #
        # V3 spec — pose-only 58 dim (joint_pos_rel 23 + joint_vel 23
        # + foot_pos_rel_pelvis_b 6 + hand_pos_rel_pelvis_b 6). Drops
        # every root/orientation row and both end-effector velocities;
        # with stack_k=1 the discriminator input is 2 * 58 = 116.
        self.amp_spec = AmpObsSpec(
            joint_names=G1_23DOF_JOINT_NAMES,
            pelvis_body_name="pelvis",
            foot_body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
            hand_body_names=(
                "left_wrist_roll_rubber_hand",
                "right_wrist_roll_rubber_hand",
            ),
            stack_k=1,
            # -- root / joints: pose-only — drop every root row (height,
            #    gravity, heading-frame lin vel, body-frame ang vel), keep
            #    only joint channels (23 each).
            include_root_height=False,
            include_projected_gravity=False,
            include_root_lin_vel_heading=False,
            include_root_ang_vel_body=False,
            include_joint_pos_rel=True,
            include_joint_vel=True,
            # -- feet (4-way split: position / orientation / linear vel / angular vel)
            #    Implemented rows: position 6-dim (3 xyz × L,R), linear_velocity 6-dim.
            #    Orientation + angular-velocity flags are placeholders in AmpObsSpec —
            #    __post_init__ raises NotImplementedError if set True. Listed here as
            #    =False so every switch is visible at a glance.
            include_feet_position=True,
            include_feet_orientation=False,         # [UNIMPLEMENTED placeholder]
            include_feet_linear_velocity=False,
            include_feet_angular_velocity=False,    # [UNIMPLEMENTED placeholder]
            # -- hands (same 4-way split as feet): position only (6-dim total).
            include_hand_position=True,
            include_hand_orientation=False,         # [UNIMPLEMENTED placeholder]
            include_hand_linear_velocity=False,
            include_hand_angular_velocity=False,    # [UNIMPLEMENTED placeholder]
        )
        self.amp_motion_files = list(DEFAULT_PENGUIN_MOTION_FILES)

        # Baseline sim plumbing (decimation / dt / sensors / terrain
        # curriculum gate). Single super() call — V1 / V2 are not in this
        # MRO, so no hidden overrides kick in.
        super().__post_init__()

        # Step 1 — disable RSI. Envs always reset to the robot's default
        # pose at t=0 rather than sampling a random expert frame, so the
        # policy never inherits ground-truth expert joint state and has to
        # learn the stance from its own rollouts (TienKung-Lab recipe).
        self.amp_reset.use_motion_reset = False


# =============================================================================
# Play / export env cfg
# =============================================================================
@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    """Play / export variant — 1 env, no corruption / push / curriculum, pinned cmd."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.8)

        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.curriculum = None

        self.commands.base_velocity.ranges.lin_vel_x = (0.25, 0.25)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
