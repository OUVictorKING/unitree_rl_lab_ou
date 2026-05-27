from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.pingpong import mdp
from unitree_rl_lab.tasks.pingpong.robots.g1_23dof.hitter.hitter_env_cfg import (
    ActionsCfg,
    EventCfg,
    RobotEnvCfg as HitterRobotEnvCfg,
    RobotPlayEnvCfg as HitterRobotPlayEnvCfg,
    RobotSceneCfg as HitterRobotSceneCfg,
    TerminationsCfg as HitterTerminationsCfg,
)
from unitree_rl_lab.tasks.pingpong.robots.g1_23dof.hitter.hitter_env_cfg import RewardsCfg as HitterRewardsCfg


@configclass
class RealRobotSceneCfg(HitterRobotSceneCfg):
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(2.74, 1.525, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=0.9,
                dynamic_friction=0.8,
                restitution=1.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.35, 0.55)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.77, 0.0, 0.735), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.02,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0027),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_linear_velocity=20.0,
                max_angular_velocity=500.0,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=0.25,
                dynamic_friction=0.20,
                restitution=0.876,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.55, 0.05)),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.75, 0.0, 1.05), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    net: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Net",
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 1.83, 0.1525),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.4,
                dynamic_friction=0.3,
                restitution=0.05,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.02, 0.02, 0.02)),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.77, 0.0, 0.83625), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    ball_table_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"],
        history_length=3,
        force_threshold=0.05,
        debug_vis=False,
    )
    ball_racket_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/.*right_paddle_blade"],
        history_length=3,
        force_threshold=0.05,
        debug_vis=False,
    )
    ball_net_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Net"],
        history_length=3,
        force_threshold=0.05,
        debug_vis=False,
    )
    ball_robot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Ball"],
        history_length=3,
        force_threshold=0.05,
        debug_vis=False,
    )


@configclass
class CommandsCfg:
    pingpong = mdp.RealPingpongCommandCfg(
        asset_name="robot",
        ball_name="ball",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        t_post_swing_fixed=0.60,
        target_land=(2.45, 0.0, 0.78),
        serve_pos_z_range=(1.05, 1.20),
        serve_hit_z_range=(0.95, 1.15),
        serve_t_to_hit_range=(0.55, 0.75),
        planner_hit_z_range=(0.85, 1.25),
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
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
class RewardsCfg(HitterRewardsCfg):
    ball_contact = RewTerm(func=mdp.real_ball_contact, weight=2.0, params={"command_name": "pingpong"})
    return_direction = RewTerm(func=mdp.real_return_direction, weight=0.5, params={"command_name": "pingpong"})
    clear_net = RewTerm(func=mdp.real_clear_net, weight=1.0, params={"command_name": "pingpong"})
    opponent_land = RewTerm(func=mdp.real_opponent_land, weight=3.0, params={"command_name": "pingpong"})
    target_land = RewTerm(func=mdp.real_target_land, weight=2.0, params={"command_name": "pingpong"})
    illegal = RewTerm(func=mdp.real_illegal, weight=-2.0, params={"command_name": "pingpong"})


@configclass
class TerminationsCfg(HitterTerminationsCfg):
    ball_dead = DoneTerm(func=mdp.real_ball_dead, params={"command_name": "pingpong"})


@configclass
class CurriculumCfg:
    pingpong = CurrTerm(
        func=mdp.update_real_pingpong_curriculum,
        params={"command_name": "pingpong", "enable_stage_updates": True},
    )


@configclass
class RobotEnvCfg(HitterRobotEnvCfg):
    scene: RealRobotSceneCfg = RealRobotSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.physx.enable_ccd = True
        self.sim.physx.gpu_max_rigid_patch_count = 12 * 2**15


class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 4.0
        self.commands.pingpong.debug_vis = True
        self.commands.pingpong.post_outcome_hold_time = 1.5
        self.commands.pingpong.ball_dead_z = -5.0
        self.commands.pingpong.ball_dead_y_abs = 8.0
        self.commands.pingpong.ball_dead_x_abs = 8.0
        self.commands.pingpong.debug_ball_traj_len = 160
        self.commands.pingpong.debug_show_aux_targets = True
        self.commands.pingpong.debug_show_vectors = False
        self.commands.pingpong.debug_show_current_ball_marker = False
        self.commands.pingpong.debug_show_net_points = False
        self.commands.pingpong.debug_show_blade_traj = False
        self.commands.pingpong.debug_show_planner_traj = False
        self.commands.pingpong.debug_show_ball_traj = True
        self.commands.pingpong.debug_show_landing_points = True
        self.commands.pingpong.debug_show_direction_arrows = True
        self.curriculum.pingpong = None
