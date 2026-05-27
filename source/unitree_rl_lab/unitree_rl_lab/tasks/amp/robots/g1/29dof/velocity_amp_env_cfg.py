# Copyright (c) 2025, Unitree Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""Velocity-tracking env cfg for Penguin AMP V1 training.

Derived from :mod:`unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg`
but trimmed for the Penguin AMP V1 baseline:

* flat terrain (no cobblestone generator / terrain curriculum)
* command restricted to ``lin_vel_x`` and ``ang_vel_z`` (no lateral), range
  clamped around the penguin clip's mean forward speed
* policy obs stripped of ``gait_phase``
* rewards drop all gait-style shaping (gait, feet_clearance, stand_still,
  feet_contact_still); task + core stabilization rewards kept
* AMP observation is produced through the manager stack as an ``"amp"``
  obs group whose term reads the K-frame circular buffer maintained by
  :class:`UnitreeAmpEnv`.

The env cfg exposes ``amp_spec`` / ``amp_motion_files`` / ``amp_reset`` hooks
consumed by :class:`UnitreeAmpEnv` and :meth:`AmpPPO.construct_algorithm`.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_23DOF_CFG as ROBOT_CFG
from unitree_rl_lab.rsl_rl_amp.features import AmpObsSpec
from unitree_rl_lab.tasks.amp import mdp
from unitree_rl_lab.tasks.amp.envs import UnitreeAmpEnvCfg, UnitreeAmpResetCfg


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


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Scene for the AMP velocity environment: flat ground, no terrain gen."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Mild domain randomization for AMP training.

    Note: ``reset_base`` / ``reset_robot_joints`` still run but are immediately
    overwritten by :meth:`UnitreeAmpEnv._apply_motion_reset` when motion
    reset is enabled (:class:`UnitreeAmpResetCfg.use_motion_reset`). They
    remain as a safe fallback when motion reset is disabled and as the
    first-frame init before the env fills the AMP K-buffer.
    """

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.0),
            "dynamic_friction_range": (0.5, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-0.5, 0.5),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 10.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )


@configclass
class CommandsCfg:
    """Penguin AMP V1: forward / yaw commands only.

    The ``limit_ranges`` cap the forward speed near the clip's mean speed
    (~0.45 m/s for the penguin clip). The Phase-1 curriculum starts at a
    small neighborhood around 0.0 and expands toward the limit as training
    progresses (see :class:`CurriculumCfg`).
    """

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.1, 0.1)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.45, 0.55), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.4, 0.4)
        ),
    )


@configclass
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation groups: policy + critic. The AMP obs is injected by the env."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)
        height_scanner = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 5.0),
        )

        def __post_init__(self):
            self.history_length = 5

    critic: CriticCfg = CriticCfg()

    @configclass
    class AmpCfg(ObsGroup):
        """AMP observation group — thin window onto the env's stacked K-buffer."""

        amp = ObsTerm(func=mdp.amp_obs)

        def __post_init__(self):
            self.concatenate_terms = True
            self.history_length = 0
            self.enable_corruption = False

    amp: AmpCfg = AmpCfg()


@configclass
class RewardsCfg:
    """Task + core stabilization rewards. Gait-style shaping removed."""

    # -- task
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
    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # -- base stabilization
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.5)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.06)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.04)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    # -- posture (light deviation only; bulk style comes from AMP)
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"])},
    )

    # -- body pose
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    base_height = RewTerm(
        func=mdp.base_height_l2, weight=-10.0, params={"target_height": 0.78}
    )

    # -- feet (keep slide penalty; drop gait/clearance/stand-still shaping)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
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


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.2}
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    """Flat-terrain AMP baseline: only command-range curriculum, no terrain levels."""

    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)
    ang_vel_cmd_levels = CurrTerm(func=mdp.ang_vel_cmd_levels)


# ----------------------------- AMP metadata ----------------------------- #

# Penguin V1 AMP: single unified spec (80-dim phi_t, stack_k=4). Built inside
# __post_init__ so callers can customize via cfg.amp_spec.include_* flags.
# Motion files: override per-run via the agent cfg and/or this placeholder.
DEFAULT_PENGUIN_MOTION_FILES: list[str] = [
    "/home/woan/HumanoidProject/unitree_rl_lab/motion_datasets/penguin/g1_qie_motion.npz",
]


@configclass
class RobotEnvCfg(UnitreeAmpEnvCfg):
    """Configuration for the AMP velocity-tracking environment (flat terrain)."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=256, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # Flat ground: explicitly disable terrain-gen curriculum if it was
        # inherited via subclassing.
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False

        # Populate AMP hooks consumed by UnitreeAmpEnv + AmpPPO.
        #
        # V1 spec — every currently-implemented include_* flag on:
        #   root_height 1 + proj_gravity 3 + root_lin_vel_h 3 + root_ang_vel_b 3
        #   + joint_pos_rel 23 + joint_vel 23
        #   + foot_position 6 + foot_linear_velocity 6
        #   + hand_position 6 + hand_linear_velocity 6
        #   = frame_dim 80, amp_obs_dim 80*4 = 320, disc_input_dim 640.
        # Orientation / angular-velocity flags are placeholders in AmpObsSpec
        # (rejected by __post_init__ if set True) — left at their False
        # default and not mentioned further here.
        if self.amp_spec is None:
            self.amp_spec = AmpObsSpec(
                joint_names=G1_23DOF_JOINT_NAMES,
                pelvis_body_name="pelvis",
                foot_body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
                hand_body_names=(
                    "left_wrist_roll_rubber_hand",
                    "right_wrist_roll_rubber_hand",
                ),
                stack_k=4,
                # -- root / joints
                include_root_height=True,
                include_projected_gravity=True,
                include_root_lin_vel_heading=True,
                include_root_ang_vel_body=True,
                include_joint_pos_rel=True,
                include_joint_vel=True,
                # -- feet (4-way split: position / orientation / linear vel / angular vel)
                #    Implemented: position 6-dim (3 xyz × L,R) + linear_velocity 6-dim.
                #    Orientation + angular-velocity are placeholders in AmpObsSpec —
                #    __post_init__ raises NotImplementedError if set True. Spelled out
                #    as =False so every switch is visible at a glance.
                include_feet_position=True,
                include_feet_orientation=False,         # [UNIMPLEMENTED placeholder]
                include_feet_linear_velocity=True,
                include_feet_angular_velocity=False,    # [UNIMPLEMENTED placeholder]
                # -- hands (same 4-way split as feet): position 6 + linear_velocity 6.
                include_hand_position=True,
                include_hand_orientation=False,         # [UNIMPLEMENTED placeholder]
                include_hand_linear_velocity=True,
                include_hand_angular_velocity=False,    # [UNIMPLEMENTED placeholder]
            )
        if not self.amp_motion_files:
            self.amp_motion_files = list(DEFAULT_PENGUIN_MOTION_FILES)


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.8)
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        # self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.3)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.1, 0.1)
        self.curriculum = None
