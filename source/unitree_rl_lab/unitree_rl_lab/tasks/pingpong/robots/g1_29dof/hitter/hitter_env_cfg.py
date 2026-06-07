"""29-DoF mirror of the 23-DoF pingpong HITTER env.

Same reward weights and curriculum verbatim — used as a controlled experiment
to disentangle "RL design plateau" from "23-dof mechanical reach limit"
(only 5-DoF wrist chain + 1-DoF waist on 23-dof). 29-dof has a full 7-DoF
arm + 3-DoF waist, so if this env breaks through the same plateau, the
plateau was mechanical; if it also stalls, it's RL design.

Differences vs the 23-dof env:
  * Spawn from `UNITREE_G1_29DOF_PADDLE_MIMIC_CFG` (29-dof paddle URDF).
  * `imitation_joint_names`: 12 joints — full 3-DoF waist + full 7-DoF LEFT arm
    + right_shoulder_{pitch,roll} only. The right-arm distal 5 joints
    (shoulder_yaw / elbow / wrist_roll/pitch/yaw) are EXCLUDED, mirroring the
    23-dof paddle-orientation-freedom pattern (see IMITATION_JOINT_NAMES_29DOF
    below). NOTE: an earlier version of this docstring claimed "full 17-joint,
    no exclusions" — that was wrong; the right hitting arm is freed here too.
  * `tracked_body_names`: 10 upper-body links (torso + full left arm + right
    shoulder pitch/roll), right-arm distal links excluded — synced with the
    joint exclusions above.
  * Motion NPZ paths come from module constants below (`MOTION_FORWARD_NPZ_29DOF`,
    `MOTION_BACKWARD_NPZ_29DOF`); a startup assertion fails fast if either
    still holds the TODO sentinel.
  * `paddle_table_contact` / `body_table_contact` / `undesired_contacts` /
    `non_paddle_table_stuck` body-name regexes use the 29-dof link names
    (`right_rubber_hand` / `left_rubber_hand` instead of
    `*_wrist_roll_rubber_hand`).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_PADDLE_MIMIC_ACTION_SCALE
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_PADDLE_MIMIC_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.pingpong import mdp
from unitree_rl_lab.tasks.pingpong.mdp.motion_loader import DEFAULT_EXPERT_ROOT


# === Motion clip paths ===
# 29-dof retargeted forehand / backhand reference clips. Pre-rotated by -90deg
# about world Z so impact-frame pelvis_yaw matches the 23-dof rotated convention
# (forward ~ -5deg, backward ~ +11deg, both facing the table) — keeps RSI
# initial pose geometry identical between the two robots so the controlled
# experiment (mechanical-vs-RL plateau) isn't confounded by clip orientation.
# body_names list was de-duplicated (removed stale `right_paddle_blade` column);
# originals stashed alongside as `*.bak_with_dup`.
MOTION_FORWARD_NPZ_29DOF: str = str(
    DEFAULT_EXPERT_ROOT / "new" / "forward" / "npz_29dof" / "forward_003_rotated.npz"
)
MOTION_BACKWARD_NPZ_29DOF: str = str(
    DEFAULT_EXPERT_ROOT / "new" / "backward" / "npz_29dof" / "backward_001_rotated.npz"
)


VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


# Mirror V1 23dof's paddle-freedom pattern: right arm distal to shoulder_roll
# is removed from imitation so the policy can find its own paddle pose. Only
# right_shoulder_pitch/roll retained to anchor the gross arm root. 23dof V1
# excludes 3 right-arm joints (shoulder_yaw + elbow + wrist_roll); 29dof has
# 2 extra wrist DOFs in the same chain so we extend the exclusion to all 5
# distal joints for parity. waist_roll/pitch KEPT — they are 29dof's only
# positive shaping signal for pelvis stance (23dof structurally lacks them
# and uses pelvis_orientation_l2 alone). 17 -> 12 imitation joints.
IMITATION_JOINT_NAMES_29DOF = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    # right_shoulder_yaw / right_elbow / right_wrist_{roll,pitch,yaw} excluded
    # for paddle-orientation control freedom (mirrors V1 23dof commands.py:688-690).
]

# Body-level mirror of the joint exclusions: right arm distal to shoulder_roll
# is removed from tracked_body_names too. Without this, body-pos imitation
# (weight 0.25*w_i, body_dominant split running 0.60*w_i) would still drag
# the right arm toward clip pose even after the joint set is freed — defeating
# the whole point. 15 -> 10 tracked bodies.
TRACKED_BODY_NAMES_29DOF = [
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    # right_shoulder_yaw_link / right_elbow_link / right_wrist_{roll,pitch,yaw}_link
    # excluded — synced with IMITATION_JOINT_NAMES_29DOF above.
]


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
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
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(2.74, 1.525, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.9,
                dynamic_friction=0.8,
                restitution=0.2,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.35, 0.55)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.77, 0.0, -10.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=10.0,
        debug_vis=False,
    )
    robot_table_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"],
        history_length=3,
        debug_vis=False,
    )


@configclass
class CommandsCfg:
    pingpong = mdp.PingpongCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        # Initial value only — the window curriculum (curriculums.py
        # _WINDOW_CURRICULUM_TIERS) shrinks this monotonically as
        # hit_success_rate climbs: 0.10 -> 0.06 -> 0.04 -> 0.02 -> 0.01.
        # Pair with goal_position / goal_velocity / goal_orientation weights
        # below, which the same curriculum ramps up in lockstep.
        strike_window=0.10,
        # 29-dof clip retargeting — module constants above must be filled in
        # before training (see startup assertion in RobotEnvCfg.__post_init__).
        forward_motion_file=MOTION_FORWARD_NPZ_29DOF,
        backward_motion_file=MOTION_BACKWARD_NPZ_29DOF,
        # Full upper-body tracking (vs 23-dof's 8-joint / 8-body subset).
        imitation_joint_names=IMITATION_JOINT_NAMES_29DOF,
        tracked_body_names=TRACKED_BODY_NAMES_29DOF,
        # RSI on. Rotated NPZ (Rz(-90°)) is already loaded above, so clip frame 0
        # spawns the robot facing the table correctly. Without RSI the robot starts
        # at default_joint_pos while the imit reference is the clip's stance pose,
        # creating ~33° per-joint gap → exp(-2*5.6)=1.2e-5 reward dead zone.
        disable_rsi=False,
    )


@configclass
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=UNITREE_G1_29DOF_PADDLE_MIMIC_ACTION_SCALE,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Sensor channels routed through `randomize_imu_offset` + `comm_delay` (paper §V-B3).
        base_ang_vel = ObsTerm(
            func=mdp.DelayedObservation,
            params={"inner_func": mdp.base_ang_vel_imu, "inner_params": {}},
        )
        projected_gravity = ObsTerm(
            func=mdp.DelayedObservation,
            params={"inner_func": mdp.projected_gravity_imu, "inner_params": {}},
        )
        base_yaw = ObsTerm(
            func=mdp.DelayedObservation,
            params={"inner_func": mdp.base_yaw_encoding_imu, "inner_params": {}},
        )
        base_err = ObsTerm(func=mdp.pingpong_base_position_error, params={"command_name": "pingpong", "noisy": True})
        hit_pos = ObsTerm(func=mdp.pingpong_hit_position_b, params={"command_name": "pingpong", "noisy": True})
        racket_vel = ObsTerm(func=mdp.pingpong_racket_velocity_w, params={"command_name": "pingpong", "noisy": True})
        t_to_hit = ObsTerm(func=mdp.pingpong_t_to_hit, params={"command_name": "pingpong", "noisy": True})
        # swing_type = ObsTerm(func=mdp.pingpong_swing_type, params={"command_name": "pingpong"})
        joint_pos = ObsTerm(
            func=mdp.DelayedObservation,
            params={"inner_func": mdp.joint_pos_rel, "inner_params": {}},
        )
        joint_vel = ObsTerm(
            func=mdp.DelayedObservation,
            params={"inner_func": mdp.joint_vel_rel, "inner_params": {}},
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_yaw = ObsTerm(func=mdp.base_yaw_encoding)
        base_err = ObsTerm(func=mdp.pingpong_base_position_error, params={"command_name": "pingpong", "noisy": False})
        hit_pos = ObsTerm(func=mdp.pingpong_hit_position_b, params={"command_name": "pingpong", "noisy": False})
        racket_vel = ObsTerm(func=mdp.pingpong_racket_velocity_w, params={"command_name": "pingpong", "noisy": False})
        t_to_hit = ObsTerm(func=mdp.pingpong_t_to_hit, params={"command_name": "pingpong", "noisy": False})
        # swing_type = ObsTerm(func=mdp.pingpong_swing_type, params={"command_name": "pingpong"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        last_action = ObsTerm(func=mdp.last_action)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        ref_body_state = ObsTerm(func=mdp.pingpong_ref_body_state, params={"command_name": "pingpong"})
        time_left = ObsTerm(func=mdp.episode_time_left)
        ref_joint_state = ObsTerm(func=mdp.pingpong_ref_joint_state, params={"command_name": "pingpong"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )
    add_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    randomize_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "friction_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    randomize_joint_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "damping_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    randomize_imu_offset = EventTerm(
        func=mdp.randomize_imu_offset,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sigma_deg": 2.0,
            "distribution": "gaussian",
        },
    )
    randomize_comm_delay = EventTerm(
        func=mdp.randomize_comm_delay,
        mode="startup",
        params={"max_delay_steps": 1},
    )
    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
    )
    # R8 table-guard: every reset, place table at HIDDEN (z=-10) until the
    # `table_guard` curriculum sets `env._pingpong_table_active=True`, after
    # which subsequent resets place it at the paper position (z=0.735).
    reset_table = EventTerm(
        func=mdp.reset_table_position_by_stage,
        mode="reset",
        params={"asset_name": "table"},
    )

# Initial w_i; the imit_anneal curriculum overwrites these weights at runtime
# according to its iter schedule (phase 0 = w_i_values[0]).
w_i = 0.5
@configclass
class RewardsCfg:
    # imitation: top-level w_i folded into sub-term weights (split 0.65/0.10/0.25)
    # Hybrid gate (run 2026-05-29 ported from 23dof Plan B):
    #   joint_pos / joint_vel: gate_pre_strike=False — track demo throughout.
    #     29dof failure mode in run 2026-05-29_11-13-14 was EL=33 stuck for 8000
    #     iter (bad_orientation=99.9% terminations, never standing). With imit
    #     gate=True AND short episodes, t_to_hit > 0 frames per episode were
    #     ~10, so imit reward was starved (~0.09 total vs 23dof V1 same iter
    #     ~0.18). Opening joint-level gates lets imit fire on every frame
    #     including early-fall frames, breaking the starvation loop.
    #   body_pos: gate_pre_strike=False — v63 sync with 23dof. Post-strike body
    #     imitation ON (return-to-ready follow-through); strike-frame paddle-pose
    #     pollution is handled by goal_orientation dominating the 1-2 strike frames.
    imitation_joint_pos = RewTerm(
        func=mdp.imitation_joint_pos,
        weight=0.65 * w_i,
        params={"command_name": "pingpong", "gate_pre_strike": False},
    )
    imitation_joint_vel = RewTerm(
        func=mdp.imitation_joint_vel,
        weight=0.1 * w_i,
        params={"command_name": "pingpong", "gate_pre_strike": False},
    )
    imitation_body_pos = RewTerm(
        func=mdp.imitation_body_pos_anchor_relative,
        weight=0.25 * w_i,
        params={"command_name": "pingpong", "gate_pre_strike": False},
    )

    # task goal — initial weights are tier-0 (window=0.10) baseline. The
    # window curriculum in curriculums.py ramps these UP in lockstep with the
    # window-shrink: tier-0 (2/2/0.5) -> tier-1 (3/3/1.0) -> ... ->
    # tier-final (12/12/4.0) at hit_success_rate >= 0.80. Pre-strike weights
    # below are independent — they don't depend on _strike_gate.
    goal_position = RewTerm(func=mdp.goal_position, weight=2.0, params={"command_name": "pingpong"})
    # v63 sync: pre_strike weights restored to 23dof baselines (1.0/1.0/0.5).
    # The B7 zeroing is superseded by the task_phase 3-phase curriculum, which
    # zeros ALL goal_* (incl. pre_strike) in Phase 0/1 and resets them to these
    # baselines on Phase 2 entry — so these env_cfg values are the Phase-2 target.
    goal_position_pre_strike = RewTerm(
        func=mdp.goal_position_pre_strike,
        weight=1.0,
        params={"command_name": "pingpong", "std": 0.2, "ramp_time": 0.2},
    )
    # v63 sync: std 0.45 → 1.50 — CRITICAL. The shared goal_velocity is now the
    # v62 Gaussian exp(-||Δv||²/σ²); with σ=0.45 it gives ~0 everywhere
    # (exp(-4/0.2)≈2e-9). (The pingpong curriculum's std_g_vel tier table also
    # overwrites this each tick on the same Gaussian scale 1.5→0.5.)
    goal_velocity = RewTerm(func=mdp.goal_velocity, weight=2.0, params={"command_name": "pingpong", "std": 1.50})
    goal_velocity_pre_strike = RewTerm(
        func=mdp.goal_velocity_pre_strike,
        weight=1.0,
        params={"command_name": "pingpong", "std": 0.6, "ramp_time": 0.1},
    )
    goal_orientation = RewTerm(func=mdp.goal_orientation, weight=0.5, params={"command_name": "pingpong", "std": 0.4})
    goal_orientation_pre_strike = RewTerm(
        func=mdp.goal_orientation_pre_strike,
        weight=0.5,
        params={"command_name": "pingpong", "std": 0.4, "ramp_time": 0.2},
    )
    goal_base = RewTerm(func=mdp.goal_base_position, weight=0.8, params={"command_name": "pingpong", "std": 0.3})

    # regularization
    alive = RewTerm(func=mdp.is_alive, weight=0.04)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.0005)
    joint_torque = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-3.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1e-7)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    pelvis_orientation = RewTerm(func=mdp.pelvis_orientation_l2, weight=-1.0)
    # Penalize pelvis roll/pitch RATE only (not yaw — swing needs yaw rotation).
    # Targets the bad_orientation 99.9% termination mode in run 2026-05-29_11-13-14.
    pelvis_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # Penalize vertical bounce. v63 sync: -0.8 → -1.5 (23dof locomotion default).
    pelvis_lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.5)
    pelvis_height = RewTerm(func=mdp.base_height_l2, weight=-5.0, params={"target_height": 0.74})
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.30,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    # v63 sync: lower-body (leg) regularizers — same as 23dof (leg joint names
    # hip_roll/yaw + ankle_roll are identical across 23/29-dof). Fix the
    # post-strike "lift a leg / single-leg stand / sway" idle posture. Weights
    # below are dead initial values — task_phase (leg_reg_phase_weights)
    # overwrites all three every tick with phase-scaled values (Phase 2 weaker).
    leg_joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )
    feet_contact_no_strike = RewTerm(
        func=mdp.feet_contact_no_strike,
        weight=0.20,
        params={
            "command_name": "pingpong",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "t_thresh": 0.0,
        },
    )
    feet_distance_no_strike = RewTerm(
        func=mdp.feet_distance_no_strike,
        weight=-0.5,
        params={
            "command_name": "pingpong",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "nominal": 0.20,
            "wide_scale": 0.3,
            "t_thresh": 0.0,
        },
    )
    # 29-dof URDF link names: hands are `left_rubber_hand` / `right_rubber_hand`
    # (vs 23-dof's `*_wrist_roll_rubber_hand`). Paddle attaches to
    # `right_rubber_hand` in the 29-dof paddle URDF (same convention — fixed
    # joint adds `right_paddle_blade` body).
    #
    # 29-dof has 8 kinematic intermediate links (2 waist + 6 wrist) that 23-dof
    # does not. These internal links self-collide with neighbors at default
    # standing pose, producing per-step undesired_contacts ~6.7× larger than
    # 23-dof. Policy then learns to die immediately to escape the penalty
    # (EpLen→1, hard_contact rate→1.0). Excluded below.
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
                    r"(?!left_rubber_hand$)(?!right_rubber_hand$)(?!right_paddle_blade$)"
                    r"(?!waist_yaw_link$)(?!waist_roll_link$)"
                    r"(?!left_wrist_roll_link$)(?!left_wrist_pitch_link$)(?!left_wrist_yaw_link$)"
                    r"(?!right_wrist_roll_link$)(?!right_wrist_pitch_link$)(?!right_wrist_yaw_link$)"
                    r".+$"
                ],
            ),
        },
    )
    # Splits the original `table_contact` penalty into paddle-side (catastrophic:
    # blocks ball strike, allows balance cheat where paddle rests on table) and
    # body-side (incidental: leg/torso brushing edge). Threshold lowered for
    # paddle to catch light static contact; weight 10x higher.
    #
    # NOTE (R8 table-guard curriculum): initial weights are 0. The
    # `table_guard` term in CurriculumCfg ramps these to (-10.0, -1.0) once the
    # policy can stand + hit + align paddle. While the table is hidden (Stage
    # 0/1), the contact sensor produces no readings anyway, so weights=0 is
    # also a defensive belt-and-braces measure.
    paddle_table_contact = RewTerm(
        func=mdp.robot_table_contact_penalty,
        weight=0.0,
        params={
            "threshold": 0.1,
            "sensor_cfg": SceneEntityCfg(
                "robot_table_contact",
                body_names=["right_rubber_hand", "right_paddle_blade"],
            ),
        },
    )
    body_table_contact = RewTerm(
        func=mdp.robot_table_contact_penalty,
        weight=0.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg(
                "robot_table_contact",
                body_names=[r"^(?!right_rubber_hand$)(?!right_paddle_blade$).+$"],
            ),
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.30})
    # Same as 23dof. Earlier run 2026-05-28_16-04-49 hit bad_orientation 99.99%
    # at this limit because UNITREE_G1_29DOF_PADDLE_MIMIC_CFG default_joint_pos
    # was squat-pose (knee=0.669) while RSI starts robot in clip's standing pose
    # (knee~0.4) — fixed by overriding default to clip frame 0 (unitree.py:973+).
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
    hard_contact = DoneTerm(
        func=mdp.hard_undesired_contact,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["pelvis", "torso_link", "head_link", ".*_hip_pitch_link"],
            ),
        },
    )
    # Catches alternative cheats: elbow propping / body leaning on table. Sustained
    # >= duration_s threshold avoids spurious terminations from incidental brushing
    # during fast moves. Per-step gradient is unaffected (kept by body_table_contact
    # reward at -1.0); termination just truncates trajectories that fall into the
    # corner so they stop dominating the policy gradient batch.
    non_paddle_table_stuck = DoneTerm(
        func=mdp.body_table_contact_sustained,
        params={
            "sensor_cfg": SceneEntityCfg(
                "robot_table_contact",
                body_names=[r"^(?!right_rubber_hand$)(?!right_paddle_blade$).+$"],
            ),
            "force_threshold": 3.0,
            "duration_s": 0.3,
        },
    )


@configclass
class CurriculumCfg:
    # Order matters: imit_anneal updates _EP_LENGTH_EMA, which pingpong's
    # window-advance gate reads. Putting imit_anneal first eliminates the
    # 1-tick EMA lag.
    imit_anneal = CurrTerm(
        func=mdp.update_imitation_weight,
        params={
            # Metric mode: anneal w_i based on actual hit-skill EMAs rather
            # than wall-clock iter. Phase advances when ALL listed metric
            # thresholds clear simultaneously. Latch is monotone — once
            # advanced, never roll back. From-scratch run 2026-05-24_07-52-04
            # showed why pure-iter anneal is wrong: 33k iter at EL≈30 still
            # cut w_i to 0.15 at iter 8000, killing the only positive shaping
            # signal before the policy could stand. Metric mode keeps
            # imitation strong until the policy demonstrates competence.
            "schedule": "metric",
            "command_name": "pingpong",
            "num_steps_per_env": 24,
            # v63: B7 ×4 boost REMOVED. The task_phase 3-phase curriculum now
            # governs imit weights (Phase 0/1/2 = 0.10/1.00/0.30 via
            # _TASK_PHASE_IMIT_SPLIT, set every tick AFTER imit_anneal), so these
            # w_i_values now only feed the EMA trackers. Aligned to 23dof.
            # Stand-up relies on the new leg regularizers + alive + pelvis terms
            # (mirrors 23dof). If 29dof stalls at low EL, the knob to turn is
            # task_phase.imit_w_phase0/1 below (restore some stand-up boost).
            "w_i_values": (0.5, 0.3, 0.15),
            # Imitation split preset (toggle this single line):
            #   "default" / "joint_dominant" : joint 0.65 / vel 0.10 / body 0.25  (plan A/B, original)
            #   "body_dominant"              : joint 0.30 / vel 0.10 / body 0.60  (plan C)
            #   None                         : use module default in curriculums.py
            #   {dict}                       : ad-hoc, e.g. {"imitation_joint_pos": 0.4, "imitation_joint_vel": 0.1, "imitation_body_pos": 0.5}
            "split": "body_dominant",
            # Phase transition thresholds (one dict per transition).
            # phase 0 → 1: "已经学会一点" — basic competence.
            #   policy stands (~5s), hits ball with reasonable rate, paddle
            #   face roughly aligned. Loosen w_i to give task rewards more
            #   room.
            # phase 1 → 2: "已经掌握" — solid competence.
            #   sustained hit_success, tight pos/vel/ori, near-full episode.
            #   Drop w_i further so PPO chases task precision.
            # v63 sync: relaxed to 23dof v62 values (vel/ori gates lowered) so
            # the imit-feedback trap (high w_i → copy demo joint vel → demo
            # vel_target ≠ ball-physics vel_target → vel_fail stuck → phase can't
            # advance) can't re-form. (Mostly feeds EMAs now; task_phase governs
            # the actual imit weight.)
            "phase_thresholds": (
                {
                    "hit_success_rate": 0.30,
                    "pos_success_rate": 0.40,
                    "vel_success_rate": 0.40,
                    "ori_success_rate": 0.55,
                    "min_ep_length": 250,
                },
                {
                    "hit_success_rate": 0.50,
                    "pos_success_rate": 0.70,
                    "vel_success_rate": 0.65,
                    "ori_success_rate": 0.75,
                    "min_ep_length": 400,
                },
            ),
            "metric_ema_alpha": 0.05,
        },
    )
    pingpong = CurrTerm(
        func=mdp.update_pingpong_curriculum,
        params={
            "command_name": "pingpong",
            "enable_noise": False,
            "enable_range": True,
            "enable_y_curriculum": True,
            "enable_v_curriculum": True,
            "enable_window_curriculum": True,
            # Stand-up gate on the window/weight ratchet (matches the imit_anneal
            # gate). Without it, signed-ori reward generated ~20% hit_success at
            # iter ~500 in run 2026-05-25_10-08-03, tripping the ratchet to
            # tier-1 (window=0.06, weights 3/3/1) before the policy could
            # stand. The strong upper-body strike gradient then prevented the
            # EL=40→234 breakthrough that baseline 23-07-21 saw at iter ~1800.
            "min_ep_length_for_window_advance": 250,
            # Stand-up gate on the signed goal_orientation reward (monotone
            # latch). Run 2026-05-25_11-01-44 stalled at EL=40 for 2000 iter
            # with the window gate alone — signed-ori still gave clean
            # paddle-direction gradient that, combined with lenient
            # goal_velocity (std=0.5), let policy commit to swing-while-falling
            # basin (vel_fail=0.001 / hit_success=0.21 yet hard_contact=0.999).
            # Force goal_orientation.weight=0 until EMA crosses threshold;
            # once opened, the latch never re-closes.
            "min_ep_length_for_ori_advance": 250,
            # Stand-up gate on goal_position / goal_velocity (+ pre_strike).
            # After the M1 RSI base-yaw fix corrected blade orientation,
            # run 2026-05-25_14-51-08 stalled at EL=41 for 1680+ iter —
            # policy farmed goal_velocity reward (14× baseline) by swinging
            # during falls. Without this latch the four pos/vel rewards
            # (weights 2.0/2.0/0.3/1.0) were gated only by |t_to_hit| ≤
            # strike_window — same window-only gate the unfixed run had,
            # but now reachable because RSI made the blade actually point
            # at v_hat. Force the four weights to 0 until EMA(EL)≥250;
            # once opened, the window curriculum's monotone max() raises
            # them back to whatever tier applies.
            "min_ep_length_for_pos_vel_advance": 250,
            # Sequenced curriculum (Stage 1 → 2 → 3): forensic on run
            # 2026-05-26_20-52-38 — at iter 14k, shape_tier was still 1.3
            # but v_in_mag had been pushed to 2.71 in parallel. Policy hit a
            # vel_fail=0.75 / hsr=0.17 reward-hacking corner (only 追 base_pos,
            # abandoned vel/ori/imit) and at iter 28k the actor std went
            # negative → RuntimeError crash. Sequencing forces the
            # window/weight ratchet to graduate first (Stage 1), then
            # v_in_mag (Stage 2), then hit_y (Stage 3). Each stage's
            # unlock condition references the prior stage's success metric.
            "sequenced_curriculum": True,
            # Stage 2 unlock: window curriculum has driven shape_tier ≥ 6
            # (paper-strict reward shaping, top of 7-tier ladder) AND hsr_ema
            # ≥ 0.85 AND cos_sim_ema ≥ 0.55 (paddle face is consistently on-target).
            "v_unlock_shape_tier": 6,
            "v_unlock_hsr": 0.85,
            "v_unlock_cos_sim": 0.55,
            # Stage 3 unlock: Stage 2 has driven v_in_high to 3.5 m/s AND
            # hsr_ema ≥ 0.80 still holds at the higher ball speed.
            "y_unlock_v_in_high": 3.5,
            "y_unlock_hsr": 0.80,
            # cos_sim collapse retreat: when cos_sim_ema dips below 0.35
            # (deeper than the freeze threshold 0.50), actively roll v_in
            # back to 2.5 and hit_y back to a 0.10-half-width band. This is
            # the reverse-ratchet missing from the prior (monotone-only)
            # curriculum; without it, run 20-52-38 stayed at v_in=2.50
            # floor while cos_sim eroded to 0.38, with no recovery path.
            "cos_sim_collapse_threshold": 0.35,
            "cos_sim_collapse_retreat_v_in_high": 2.5,
            "cos_sim_collapse_retreat_hit_y_half_w": 0.10,
        },
    )
    # v63 sync: 3-phase task curriculum (stand → imit → strike). MUST run AFTER
    # `pingpong` and `imit_anneal` so it overrides their imit / goal_* weight
    # decisions, and BEFORE `table_guard`. Phase 0: stand (low imit, goal_*=0).
    # Phase 1: heavy imit, goal_*=0. Phase 2: paper task + leg regs ease off.
    # Monotone latches. imit_w_phase mirrors 23dof exactly (per user decision);
    # 29dof stand-up now leans on the new leg regularizers instead of B7's imit
    # boost — if EL stalls, bump imit_w_phase0/1 here.
    task_phase = CurrTerm(
        func=mdp.update_task_phase,
        params={
            "el_phase_0_to_1": 350.0,
            "el_phase_1_to_2": 450.0,
            "phase_1_min_iters": 2000,
            "imit_w_phase0": 0.10,
            "imit_w_phase1": 1.00,
            "imit_w_phase2": 0.30,
            "leg_reg_phase_weights": {
                "leg_joint_deviation": (-0.5, -0.5, -0.3),
                "feet_contact_no_strike": (0.20, 0.20, 0.10),
                "feet_distance_no_strike": (-0.5, -0.5, -0.3),
            },
        },
    )
    # R8 table-guard: hide the table during stand-up + swing-learning. Once the
    # policy reliably stands (ep_length_ema ≥ 400), hits (hsr_ema ≥ 0.65) and
    # has aligned paddle (cos_sim_ema ≥ 0.50), teleport the table back and
    # ramp paddle/body table_contact penalties from 0 to (-10, -1) over
    # `ramp_iters`. Stage transitions:
    #     0 hidden  -> 1 unlocked -> 2 ramping -> 3 active
    # Place AFTER `pingpong` so _COS_SIM_EMA / _IMIT_METRIC_EMA / _EP_LENGTH_EMA
    # are already refreshed this tick.
    table_guard = CurrTerm(
        func=mdp.update_table_guard_stage,
        params={
            "num_steps_per_env": 24,
            "min_hsr_ema": 0.65,
            "min_cos_sim_ema": 0.45,
            "min_ep_length_ema": 400.0,
            "min_iter": 1500,
            "ramp_iters": 500,
            "target_paddle_weight": -10.0,
            "target_body_weight": -1.0,
            "paddle_term_name": "paddle_table_contact",
            "body_term_name": "body_table_contact",
        },
    )


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # Fail fast if NPZs are missing (typo / forgot to retarget).
        from pathlib import Path

        for name, path in (
            ("MOTION_FORWARD_NPZ_29DOF", MOTION_FORWARD_NPZ_29DOF),
            ("MOTION_BACKWARD_NPZ_29DOF", MOTION_BACKWARD_NPZ_29DOF),
        ):
            if not Path(path).is_file():
                raise FileNotFoundError(
                    f"{name} points to missing file: {path!r}. "
                    f"Edit hitter_env_cfg.py to fix the path."
                )

        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15


class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 4.0
