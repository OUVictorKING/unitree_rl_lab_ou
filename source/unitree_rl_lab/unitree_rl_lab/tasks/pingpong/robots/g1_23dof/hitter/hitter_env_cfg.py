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

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_23DOF_PADDLE_MIMIC_ACTION_SCALE
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_23DOF_PADDLE_MIMIC_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.pingpong import mdp


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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.77, 0.0, 0.735), rot=(1.0, 0.0, 0.0, 0.0)),
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
    )


@configclass
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=UNITREE_G1_23DOF_PADDLE_MIMIC_ACTION_SCALE,
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


@configclass
class RewardsCfg:
    # imitation: top-level w_i=0.5 folded into sub-term weights
    imitation_joint_pos = RewTerm(func=mdp.imitation_joint_pos, weight=0.325, params={"command_name": "pingpong"})
    imitation_joint_vel = RewTerm(func=mdp.imitation_joint_vel, weight=0.05, params={"command_name": "pingpong"})
    imitation_body_pos = RewTerm(
        func=mdp.imitation_body_pos_anchor_relative, weight=0.125, params={"command_name": "pingpong"}
    )

    # task goal
    goal_position = RewTerm(func=mdp.goal_position, weight=2.0, params={"command_name": "pingpong"})
    goal_velocity = RewTerm(func=mdp.goal_velocity, weight=1.0, params={"command_name": "pingpong", "std": 0.5})
    goal_orientation = RewTerm(func=mdp.goal_orientation, weight=0.5, params={"command_name": "pingpong", "std": 0.2})
    goal_base = RewTerm(func=mdp.goal_base_position, weight=0.3, params={"command_name": "pingpong", "std": 0.3})

    # regularization
    alive = RewTerm(func=mdp.is_alive, weight=0.1)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.0005)
    joint_torque = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    pelvis_orientation = RewTerm(func=mdp.pelvis_orientation_l2, weight=-1.0)
    pelvis_height = RewTerm(func=mdp.base_height_l2, weight=-10.0, params={"target_height": 0.74})
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_no_command,
        weight=0.5,
        params={"threshold": 0.4, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")},
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
    table_contact = RewTerm(
        func=mdp.robot_table_contact_penalty,
        weight=-1.0,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("robot_table_contact", body_names=".*")},
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


@configclass
class CurriculumCfg:
    pingpong = CurrTerm(
        func=mdp.update_pingpong_curriculum,
        params={"command_name": "pingpong", "enable_noise": False, "enable_range": False},
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
