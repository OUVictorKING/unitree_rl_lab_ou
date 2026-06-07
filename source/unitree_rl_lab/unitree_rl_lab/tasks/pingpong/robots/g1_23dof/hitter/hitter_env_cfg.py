from __future__ import annotations

import os

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

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_23DOF_PADDLE_MIMIC_ACTION_SCALE
# from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_23DOF_PADDLE_MIMIC_CFG as ROBOT_CFG
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_23DOF_PADDLE_MIMIC_CFG_low_PD as ROBOT_CFG
from unitree_rl_lab.tasks.pingpong import mdp

# A/B ablation toggle (env var, read at import; default = control arm = original
# pre_strike window). PINGPONG_PRESTRIKE_RAMP sets ramp_time for goal_position_pre_strike
# + goal_orientation_pre_strike (the wind-up shaping window). 0.2 = original (default);
# lower (e.g. 0.1 / 0.05) shrinks the pre_strike window so shaping focuses near contact.
# The actual value is recorded in the run's params/env.yaml via the RewTerm params below.
_PRESTRIKE_RAMP = float(os.environ.get("PINGPONG_PRESTRIKE_RAMP", "0.2"))


VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


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
        # v64: tighter ori success bar for backhand (forehand keeps 0.25 → cos>0.75).
        # 0.20 → requires signed-cos > 0.80 for a backhand to count as a hit, so the
        # policy can't keep succeeding with the mediocre face it drifts to once imit_w
        # drops in Phase 2. Modest on purpose — only bites the 0.75–0.80 cos band, and
        # hsr_backhand feeds hsr_ema which the curricula gate on. Watch cos_sim_at_strike_backhand.
        success_ori_cos_dist_thresh_backhand=0.20,
        # ── Hit plane at the table NEAR EDGE (pre_strike table-collision fix) ──
        # Table near edge is at world x = 1.77 - 1.37 = 0.40. Put the virtual hit
        # plane there (hit_x = env-frame x = 0.40), and stand the robot back so its
        # demo reach (~0.538 pelvis→blade) lands exactly on the edge with ZERO forced
        # fore-aft motion: reset_root_pos.x = 0.40 - 0.538 = -0.138 (robot ~0.54 m from
        # the edge = base_target_x). Then the pre_strike back-projection (p_hit - v_hat·t,
        # toward the robot) sits at x < 0.40 = OFF the table → no table penetration for
        # low hits even with the long 0.2 s ramp. NOTE: the MuJoCo/deploy side must match
        # this (robot ~0.54 m from edge, hit at edge) or the reach depth is OOD.
        hit_x=0.40,
        reset_root_pos=(-0.138, 0.0, 0.74),
        # body_pos POSTURE anchor: re-add the right upper-arm bodies (shoulder_yaw +
        # elbow) to tracked_body_names so imitation_body_pos POSITION-anchors the
        # right-arm posture → blocks the "forehand posture + wrist-flip for backhand"
        # degenerate (run 18-26-56) WHILE leaving the paddle FACE free (position-
        # tracking ≠ orientation-lock; the face stays on wrist_roll). Right wrist/paddle
        # deliberately NOT tracked so the wrist keeps its face-tuning freedom.
        tracked_body_names=[
            "torso_link",
            "left_shoulder_pitch_link",
            "left_shoulder_roll_link",
            "left_shoulder_yaw_link",
            "left_elbow_link",
            "left_wrist_roll_rubber_hand",
            "right_shoulder_pitch_link",
            "right_shoulder_roll_link",
            "right_shoulder_yaw_link",
            "right_elbow_link",
        ],
        # ───────────────────────────────────────────────────────────────────
        # EXPERIMENT (2026-05-31): FULL upper-body JOINT imitation.
        # Baseline (deviation run) frees the right-arm distal joints
        # (shoulder_yaw + elbow + wrist_roll) for paddle-orientation control —
        # which lets the policy reach forehand points with a contorted
        # backhand-style posture + wrist flip (the "正手用反手姿势" observed).
        # This override adds ALL upper-body joints back to imitation so the
        # demo's forehand arm posture is imitated. Imitation WEIGHTS unchanged.
        #
        # Deliberately leave tracked_body_names at the default (8) — the
        # imitation_body_pos term has the largest imit share (body_dominant
        # 0.50), so NOT expanding body-tracking keeps the imit gradient from
        # over-grabbing (user's concern) and avoids the strongest M3 face-lock
        # lever. Joints alone shape posture, and obs dim is unchanged so this
        # CAN resume from a freed-arm checkpoint (model_7000).
        #
        # WATCH: cos_sim_forehand/backhand (if they drop vs deviation → the
        # right-arm joint imitation is over-constraining the paddle face, M3)
        # and hit_success_rate (if it drops → imit is hurting task learning).
        # ROLL BACK: delete this override (falls back to commands.py freed-arm
        # default).
        imitation_joint_names=[
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
        ],
    )


@configclass
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=UNITREE_G1_23DOF_PADDLE_MIMIC_ACTION_SCALE,
        use_default_offset=True,
        # v64 robustness: clip the raw (pre-scale) action. action_l2/action_rate_l2
        # are unbounded sum(action²) — when the actor briefly diverges (e.g. the
        # resume curriculum-jolt at iter 60002, or the v61 crash at iter ~35k) the
        # penalty hits −1e22 and detonates the value function → runaway. Normal
        # actions are ~±3; ±10 (≈±1.5 rad offset on the arm) never bites a real
        # swing but caps the penalty to a finite, recoverable value.
        clip={".*": (-10.0, 10.0)},
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
        # v64: explicit paddle-face alignment signal. The actor otherwise can't see
        # its own face (only racket_vel) and must infer it from raw joints via FK —
        # the likely reason face-alignment was so hard. active_face is SIGNED by
        # swing_type (the exact vector goal_orientation scores), so the task is
        # uniformly "point active_face at target_normal" for fh AND bh. Both in base
        # frame. NOT swing_type ±1 (that mode-collapsed) — this is face feedback.
        active_face = ObsTerm(func=mdp.pingpong_active_face_b, params={"command_name": "pingpong", "noisy": True})
        target_normal = ObsTerm(func=mdp.pingpong_target_normal_b, params={"command_name": "pingpong", "noisy": True})
        # swing_type removed from actor obs — paper Table I doesn't include it,
        # and run 2026-05-28_11-10-06 showed adding it as ±1 scalar caused mode
        # collapse to forehand=0.97 by iter 2000, EpLen plateau at 125,
        # cos_sim=-0.25 (paddle facing wrong way). Critic still sees it below.
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
        # v64: paddle-face alignment signal (see PolicyCfg). Critic sees noiseless.
        active_face = ObsTerm(func=mdp.pingpong_active_face_b, params={"command_name": "pingpong", "noisy": False})
        target_normal = ObsTerm(func=mdp.pingpong_target_normal_b, params={"command_name": "pingpong", "noisy": False})
        # V1 baseline: no swing_type in critic obs. V2 (15-28-09) ablation
        # showed adding it slowed cos_sim learning ~1500 iter vs V1.
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
    # Hybrid gate (run 2026-05-29_12-07-26 forensic):
    #   joint_pos / joint_vel: gate_pre_strike=False — track demo throughout
    #     including follow-through (joint angles return-to-ready post-strike).
    #   body_pos:            gate_pre_strike=True  — body imitation OFF post-strike.
    #     Reason: with body_dominant split=0.60, body_pos was the dominant reward
    #     (0.281/step at iter 2000) and pulled paddle pose toward the avg of demo
    #     trajectory rather than the strike-frame target, driving cos_sim to -0.76
    #     by iter 2000 while goal_orientation_pre_strike was 0.0002 (no signal).
    #     Gating body_pos pre-strike-only restores V1 21-04-08's cos_sim landscape.
    imitation_joint_pos = RewTerm(
        func=mdp.imitation_joint_pos,
        weight=0.65 * w_i,
        params={"command_name": "pingpong", "gate_pre_strike": False, "post_strike_scale": 1.0, "post_strike_delay": 0.04},
    )
    imitation_joint_vel = RewTerm(
        func=mdp.imitation_joint_vel,
        weight=0.1 * w_i,
        params={"command_name": "pingpong", "gate_pre_strike": False, "post_strike_scale": 1.0, "post_strike_delay": 0.04},
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
    goal_position_pre_strike = RewTerm(
        func=mdp.goal_position_pre_strike,
        weight=1.0,
        params={"command_name": "pingpong", "std": 0.2, "ramp_time": 0.2},
    )
    goal_velocity = RewTerm(func=mdp.goal_velocity, weight=2.0, params={"command_name": "pingpong", "std": 1.50})
    goal_velocity_pre_strike = RewTerm(
        func=mdp.goal_velocity_pre_strike,
        weight=1.0,
        params={"command_name": "pingpong", "std": 0.6, "ramp_time": 0.2},
    )
    goal_orientation = RewTerm(func=mdp.goal_orientation, weight=0.5, params={"command_name": "pingpong", "std": 0.4})
    goal_orientation_pre_strike = RewTerm(
        func=mdp.goal_orientation_pre_strike,
        weight=0.5,
        params={"command_name": "pingpong", "std": 0.4, "ramp_time": 0.2},
    )
    goal_base = RewTerm(func=mdp.goal_base_position, weight=1.5, params={"command_name": "pingpong", "std": 0.3})
    # V58: anchor base yaw to +X (face table). Pairs with swing-first sampling
    # to break the backhand mode-cheat (fh_share=0.003 in run 14-54-15) — forces
    # policy to translate laterally instead of rotating body to cover left/right
    # hit points. weight 0.3 < goal_base_position 0.8 so position pull dominates.
    goal_base_orientation = RewTerm(
        func=mdp.goal_base_orientation,
        weight=0.3,
        params={"command_name": "pingpong", "std": 0.3},
    )

    # regularization
    alive = RewTerm(func=mdp.is_alive, weight=0.04)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_bounded, weight=-0.001)
    action_l2 = RewTerm(func=mdp.action_l2_bounded, weight=-0.0005)
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
    # Targets the post-strike topple mode where reaction torque from arm extension
    # rocks the base before next cmd resamples.
    pelvis_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # Penalize vertical bounce — model_3000 from 12-07-26 showed robot hopping
    # during repositioning. V58 bumped to -1.5 (locomotion default) per user
    # request to harden anti-bounce after fh_share cheat investigation.
    pelvis_lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.5)
    pelvis_height = RewTerm(func=mdp.base_height_l2, weight=-5.0, params={"target_height": 0.74})
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    # Lower-body (leg) regularizers — fix the post-strike "lift a leg / single-leg
    # stand / sway" idle posture (legs were fully unconstrained: imitation is
    # upper-body only, no leg default-pose anchor). NOTE: the weights below are
    # dead initial values — the `task_phase` curriculum (leg_reg_phase_weights)
    # overwrites all three every tick with phase-scaled values (gentle, Phase 2
    # weaker so they don't fight lateral repositioning / low-ball squats).
    #
    # leg_joint_deviation: hip_roll/yaw deviation from default, ALWAYS ON
    #   (matches locomotion joint_deviation_legs). Only hip_roll/yaw — hip_pitch/
    #   knee/ankle left free so stepping + squatting are unaffected; kills the
    #   sideways leg splay.
    leg_joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )
    # feet_contact_no_strike: reward both feet grounded while NOT approaching a
    #   hit (t_to_hit<=0). Gated (not always-on) so the approach phase can still
    #   step laterally — mirrors locomotion gating feet_contact to "no command".
    feet_contact_no_strike = RewTerm(
        func=mdp.feet_contact_no_strike,
        weight=0.20,
        params={
            "command_name": "pingpong",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "t_thresh": 0.0,
        },
    )
    # feet_distance_no_strike: penalize stance-width deviation while waiting.
    #   Asymmetric: crossing legs (too narrow) penalized fully, spreading
    #   (too wide) scaled by wide_scale=0.3 since a wide stance aids low-ball
    #   squats. Gated to t_to_hit<=0.
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
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_roll_rubber_hand$)(?!right_wrist_roll_rubber_hand$)(?!right_paddle_blade$).+$"
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
                body_names=["right_wrist_roll_rubber_hand", "right_paddle_blade"],
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
                body_names=[r"^(?!right_wrist_roll_rubber_hand$)(?!right_paddle_blade$).+$"],
            ),
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.30})
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
                body_names=[r"^(?!right_wrist_roll_rubber_hand$)(?!right_paddle_blade$).+$"],
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
            # v60 update: thresholds loosened after run 2026-05-30_00-43-25 showed
            # phase stuck at 0 because vel_success≥0.70 was unreachable (vel_ema
            # plateau at 0.52-0.57). Imit feedback trap: high w_i → policy copies
            # demo joint angles → demo vel_target ≠ ball-physics vel_target →
            # vel_fail stays high → phase can't advance → w_i stays high → loop.
            # Lowering vel/ori gates lets phase advance, w_i drops, goal_velocity
            # gets relative boost, policy escapes the trap.
            "phase_thresholds": (
                {
                    "hit_success_rate": 0.30,
                    "pos_success_rate": 0.40,
                    "vel_success_rate": 0.40,  # v60: 0.70 → 0.40
                    "ori_success_rate": 0.55,  # v60: 0.70 → 0.55
                    "min_ep_length": 250,
                },
                {
                    "hit_success_rate": 0.50,  # v60: 0.60 → 0.50
                    "pos_success_rate": 0.70,
                    "vel_success_rate": 0.65,  # v60: 0.85 → 0.65
                    "ori_success_rate": 0.75,  # v60: 0.85 → 0.75
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
    # v61: 3-phase task curriculum (stand → imit → strike). MUST run AFTER
    # `pingpong` and `imit_anneal` so it overrides their imit / goal_* weight
    # decisions based on task phase. User-driven design from sim observation
    # of forehand-cheat-as-backhand. Phase 0: stand (low imit, no goal). Phase 1:
    # heavy imit, no goal. Phase 2: paper task. Monotone latches (one-way valves).
    # phase_1_min_iters是模仿的iter总步数，确保模仿阶段至少持续这么多步，防止过快进入阶段2（击球阶段），错过模仿学习的机会。
    task_phase = CurrTerm(
        func=mdp.update_task_phase,
        params={
            "el_phase_0_to_1": 350.0,  # Phase 0→1: EL EMA ≥ 350 (stand learned)
            "el_phase_1_to_2": 450.0,  # Phase 1→2: EL EMA ≥ 450 (imit + stand both stable)
            "phase_1_min_iters": 2000, # FROM-SCRATCH: Phase 1 ≥2000 iter — gives the intra-Phase-1 face ramp (posture-first) room to learn distinct fh/bh postures THEN raise the face reward. Orig rationale:
                                       # before advancing to Phase 2. Without this,
                                       # rapid EL surge (run 15-41-23: EL 339→448 in
                                       # 100 iter) lets policy skip Phase 1 entirely,
                                       # bypassing heavy-imit forehand/backhand learning.
            "imit_w_phase0": 0.10,     # weak imit prior in stand phase
            "imit_w_phase1": 1.00,     # heavy imit in imit phase
            "imit_w_phase2": 0.30,     # paper-balanced in strike phase
            # Phase-scaled leg regularizers (term -> (phase0_w, phase1_w, phase2_w)).
            # Gentle profile, Phase 2 slightly weaker: legs need stepping room in
            # every phase (goal_base lateral move-to-position is active from Phase
            # 0), and Phase 2 also squats for low balls. These overwrite the dead
            # initial weights on the three RewTerms above every tick.
            "leg_reg_phase_weights": {
                "leg_joint_deviation": (-0.5, -0.5, -0.3),
                "feet_contact_no_strike": (0.20, 0.20, 0.10),
                "feet_distance_no_strike": (-0.5, -0.5, -0.3),
            },
            # Early face learning: turn on the DENSE face signal (goal_orientation_
            # pre_strike) already in Phase 1 — pos/vel stay off till Phase 2, so the
            # face is learned uncontested (no gradient fight with pos/vel). (p0,p1,p2).
            "command_name": "pingpong",
            "face_prestrike_phase_weights": (0.0, 1.8, 2.5),
            # Per-phase face-joint imitation weight so the dense face reward can
            # recruit waist/wrist (pure-face → low 0.3) while keeping shoulder_yaw/
            # elbow (dual swing+face → 0.6) and the rest fully imitated. Phase 2 a bit
            # lower; imit_orient_anneal decays further on stall. (p0, p1, p2).
            "face_imit_phase_weights": {
                "waist_yaw_joint": (1.0, 0.3, 0.2),
                "right_wrist_roll_joint": (1.0, 0.3, 0.2),
                "right_shoulder_yaw_joint": (1.0, 0.6, 0.4),
                "right_elbow_joint": (1.0, 0.6, 0.4),
            },
            # Intra-Phase-1 ramp (POSTURE-FIRST): EARLY Phase 1 = high face-joint imit
            # (face_imit_p1_early=1.0) + LOW face reward (face_prestrike_p1_early=0.2)
            # → learn distinct fh/bh postures; then ramp to the p1 tuple values over
            # [0.4, 0.8]·phase_1_min_iters so the face refines inside the learned basin.
            "face_p1_ramp_frac": (0.4, 0.8),
            "face_prestrike_p1_early": 0.2,
            "face_imit_p1_early": 1.0,
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

    # v65: down-weight waist_yaw + right-arm distal imitation as the face plateaus,
    # so goal_orientation can recruit them to push the paddle face past the demo's
    # ~0.80 cap (the joints stay imitated, just down-weighted; stall-driven). MUST run
    # after `pingpong` (sets cos_sim_ema) and `task_phase` (sets the phase).
    imit_orient_anneal = CurrTerm(
        func=mdp.update_imit_orient_weight,
        params={
            "command_name": "pingpong",
            "orient_joint_names": (
                "waist_yaw_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
            ),
            "active_phase": 2,
            "stall_iters": 600,
            "stall_decay": 0.6,
            "floor": 0.05,
        },
    )

    # Stall-driven pre_strike shaping anneal: the long pre_strike window (ramp_time
    # 0.2) gives dense pos/vel/ori guidance early so the policy learns the waist+base
    # coordination for the paddle face. Once cos_sim_ema plateaus, shrink ramp_time
    # x0.6 per stall and finally DISABLE the pre_strike signals (ramp_time→0 +
    # weight→0) so the policy relies on the true strike-instant reward (removes the
    # shaping crutch). MUST run after `pingpong` (cos_sim_ema) and `task_phase` (phase).
    prestrike_ramp_anneal = CurrTerm(
        func=mdp.update_prestrike_ramp_anneal,
        params={
            "command_name": "pingpong",
            "prestrike_terms": (
                "goal_position_pre_strike",
                "goal_velocity_pre_strike",
                "goal_orientation_pre_strike",
            ),
            "active_phase": 2,
            "initial_ramp": 0.2,
            "stall_iters": 600,
            "ramp_decay": 0.6,
            "off_threshold": 0.05,
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
