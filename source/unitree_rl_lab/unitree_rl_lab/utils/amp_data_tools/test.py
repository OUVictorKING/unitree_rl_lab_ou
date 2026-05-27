# from __future__ import annotations
# import argparse
# from pathlib import Path

# import numpy as np
# import torch

# from isaaclab.app import AppLauncher

# parser = argparse.ArgumentParser(description="Replay G1 motion from npz.")
# parser.add_argument(
#     "--input_npz",
#     type=str,
#     default="/home/woan/HumanoidProject/unitree_rl_lab/motion_datasets/g1/ember_lafan_g1/LAFAN_walk1_subject1_0_-1.npz",
# )
# parser.add_argument("--fps", type=int, default=None, help="Override fps if needed.")
# AppLauncher.add_app_launcher_args(parser)
# args_cli = parser.parse_args()

# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app

# import isaaclab.sim as sim_utils
# from isaaclab.assets import ArticulationCfg, AssetBaseCfg
# from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
# from isaaclab.sim import SimulationContext
# from isaaclab.utils import configclass
# from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_23DOF_CFG as ROBOT_CFG


# TARGET_G1_23_DOF_NAMES = [
#     "left_hip_pitch_joint",
#     "left_hip_roll_joint",
#     "left_hip_yaw_joint",
#     "left_knee_joint",
#     "left_ankle_pitch_joint",
#     "left_ankle_roll_joint",
#     "right_hip_pitch_joint",
#     "right_hip_roll_joint",
#     "right_hip_yaw_joint",
#     "right_knee_joint",
#     "right_ankle_pitch_joint",
#     "right_ankle_roll_joint",
#     "waist_yaw_joint",
#     "left_shoulder_pitch_joint",
#     "left_shoulder_roll_joint",
#     "left_shoulder_yaw_joint",
#     "left_elbow_joint",
#     "left_wrist_roll_joint",
#     "right_shoulder_pitch_joint",
#     "right_shoulder_roll_joint",
#     "right_shoulder_yaw_joint",
#     "right_elbow_joint",
#     "right_wrist_roll_joint",
# ]


# @configclass
# class ReplaySceneCfg(InteractiveSceneCfg):
#     ground = AssetBaseCfg(
#         prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
#     )
#     sky_light = AssetBaseCfg(
#         prim_path="/World/skyLight",
#         spawn=sim_utils.DomeLightCfg(
#             intensity=750.0,
#             texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
#         ),
#     )
#     robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# def _load_npz_motion(path: str | Path):
#     # 下载下来的数据是29dof的，自动裁成23dof
#     data = np.load(path, allow_pickle=True)

#     # mimic 风格
#     if "joint_pos" in data and "body_pos_w" in data and "body_quat_w" in data:
#         joint_pos = data["joint_pos"].astype(np.float32)
#         joint_vel = data["joint_vel"].astype(np.float32)
#         root_pos = data["body_pos_w"][:, 0].astype(np.float32)
#         root_quat = data["body_quat_w"][:, 0].astype(np.float32)
#         fps = int(data["fps"][0] if np.ndim(data["fps"]) > 0 else data["fps"])
#         joint_names = (
#             data["joint_names"].tolist()
#             if "joint_names" in data
#             else TARGET_G1_23_DOF_NAMES
#         )

#     # AMP 风格
#     elif (
#         "dof_positions" in data
#         and "body_positions" in data
#         and "body_rotations" in data
#     ):
#         joint_pos = data["dof_positions"].astype(np.float32)
#         joint_vel = data["dof_velocities"].astype(np.float32)
#         root_pos = data["body_positions"][:, 0].astype(np.float32)
#         root_quat = data["body_rotations"][:, 0].astype(np.float32)
#         fps = int(data["fps"][0] if np.ndim(data["fps"]) > 0 else data["fps"])
#         joint_names = data["dof_names"].tolist() if "dof_names" in data else None
#     else:
#         raise ValueError(f"Unsupported npz format: {path}")

#     # 如果是 29DOF，自动裁成 23DOF
#     if joint_names is not None and len(joint_names) == 29:
#         keep_dof_indices = [
#             0,
#             1,
#             2,
#             3,
#             4,
#             5,
#             6,
#             7,
#             8,
#             9,
#             10,
#             11,
#             12,
#             15,
#             16,
#             17,
#             18,
#             19,
#             22,
#             23,
#             24,
#             25,
#             26,
#         ]
#         joint_pos = joint_pos[:, keep_dof_indices]
#         joint_vel = joint_vel[:, keep_dof_indices]
#         joint_names = [joint_names[i] for i in keep_dof_indices]
#         print(
#             "[INFO] detected 29DOF motion, remapped to 23DOF for UNITREE_G1_23DOF_CFG"
#         )

#     return fps, root_pos, root_quat, joint_pos, joint_vel, joint_names


# def run():
#     fps, root_pos, root_quat, joint_pos_np, joint_vel_np, joint_names, fmt = (
#         _load_npz_motion(args_cli.input_npz)
#     )
#     if args_cli.fps is not None:
#         fps = int(args_cli.fps)

#     print(f"[INFO] format={fmt}, frames={joint_pos_np.shape[0]}, fps={fps}")
#     print(f"[INFO] joint_names={joint_names}")

#     sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
#     sim_cfg.dt = 1.0 / fps
#     sim = SimulationContext(sim_cfg)

#     scene_cfg = ReplaySceneCfg(num_envs=1, env_spacing=2.0)
#     scene = InteractiveScene(scene_cfg)
#     sim.reset()

#     robot = scene["robot"]
#     joint_ids = robot.find_joints(joint_names, preserve_order=True)[0]

#     root_pos_t = torch.from_numpy(root_pos).to(sim.device)
#     root_quat_t = torch.from_numpy(root_quat).to(sim.device)
#     joint_pos_t = torch.from_numpy(joint_pos_np).to(sim.device)
#     joint_vel_t = torch.from_numpy(joint_vel_np).to(sim.device)

#     frame = 0
#     while simulation_app.is_running():
#         root_state = robot.data.default_root_state.clone()
#         root_state[:, :3] = root_pos_t[frame : frame + 1]
#         root_state[:, :2] += scene.env_origins[:, :2]
#         root_state[:, 3:7] = root_quat_t[frame : frame + 1]

#         joint_pos = robot.data.default_joint_pos.clone()
#         joint_vel = robot.data.default_joint_vel.clone()
#         joint_pos[:, joint_ids] = joint_pos_t[frame : frame + 1]
#         joint_vel[:, joint_ids] = joint_vel_t[frame : frame + 1]

#         robot.write_root_state_to_sim(root_state)
#         robot.write_joint_state_to_sim(joint_pos, joint_vel)

#         sim.render()
#         scene.update(sim.get_physics_dt())

#         lookat = root_state[0, :3].cpu().numpy()
#         sim.set_camera_view(lookat + np.array([2.0, 2.0, 0.8]), lookat)

#         frame += 1
#         if frame >= joint_pos_np.shape[0]:
#             frame = 0


# if __name__ == "__main__":
#     try:
#         run()
#     finally:
#         simulation_app.close()

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay G1 motion from npz.")
parser.add_argument(
    "--input_npz",
    type=str,
    default="/home/woan/HumanoidProject/unitree_rl_lab/motion_datasets/g1/ember_lafan_g1/LAFAN_walk1_subject1_0_-1.npz",
)
parser.add_argument("--fps", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_23DOF_CFG as ROBOT_CFG


TARGET_G1_23_DOF_NAMES = [
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
]


KEEP_DOF_29_TO_23 = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    15,
    16,
    17,
    18,
    19,
    22,
    23,
    24,
    25,
    26,
]


@configclass
class ReplaySceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def load_motion_npz(path: str | Path):
    data = np.load(path, allow_pickle=True)

    if "joint_pos" in data and "body_pos_w" in data and "body_quat_w" in data:
        fmt = "mimic"
        joint_pos = data["joint_pos"].astype(np.float32)
        joint_vel = data["joint_vel"].astype(np.float32)
        root_pos = data["body_pos_w"][:, 0].astype(np.float32)
        root_quat = data["body_quat_w"][:, 0].astype(np.float32)
        fps = int(data["fps"][0] if np.ndim(data["fps"]) > 0 else data["fps"])
        joint_names = data["joint_names"].tolist() if "joint_names" in data else None

    elif (
        "dof_positions" in data
        and "body_positions" in data
        and "body_rotations" in data
    ):
        fmt = "amp"
        joint_pos = data["dof_positions"].astype(np.float32)
        joint_vel = data["dof_velocities"].astype(np.float32)
        root_pos = data["body_positions"][:, 0].astype(np.float32)
        root_quat = data["body_rotations"][:, 0].astype(np.float32)
        fps = int(data["fps"][0] if np.ndim(data["fps"]) > 0 else data["fps"])
        joint_names = data["dof_names"].tolist() if "dof_names" in data else None

    else:
        raise ValueError(f"Unsupported npz format: {path}")

    if joint_names is None:
        raise ValueError("joint_names / dof_names not found in npz")

    print(f"[INFO] format={fmt}, frames={joint_pos.shape[0]}, fps={fps}")
    print(f"[INFO] original dof count={len(joint_names)}")
    print(f"[INFO] joint_names={joint_names}")

    if len(joint_names) == 29:
        joint_pos = joint_pos[:, KEEP_DOF_29_TO_23]
        joint_vel = joint_vel[:, KEEP_DOF_29_TO_23]
        joint_names = [joint_names[i] for i in KEEP_DOF_29_TO_23]
        print(
            "[INFO] detected 29DOF motion, remapped to 23DOF for UNITREE_G1_23DOF_CFG"
        )

    print(f"[INFO] replay dof count={len(joint_names)}")
    print(f"[INFO] replay joint_names={joint_names}")

    return fps, root_pos, root_quat, joint_pos, joint_vel, joint_names


def run():
    fps, root_pos, root_quat, joint_pos_np, joint_vel_np, joint_names = load_motion_npz(
        args_cli.input_npz
    )

    if args_cli.fps is not None:
        fps = int(args_cli.fps)

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / fps
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplaySceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO] scene reset complete")

    robot = scene["robot"]
    joint_ids = robot.find_joints(joint_names, preserve_order=True)[0]
    print(f"[INFO] found {len(joint_ids)} robot joints")

    root_pos_t = torch.from_numpy(root_pos).to(sim.device)
    root_quat_t = torch.from_numpy(root_quat).to(sim.device)
    joint_pos_t = torch.from_numpy(joint_pos_np).to(sim.device)
    joint_vel_t = torch.from_numpy(joint_vel_np).to(sim.device)

    frame = 0
    num_frames = joint_pos_np.shape[0]

    print("[INFO] starting replay loop")
    while simulation_app.is_running():
        root_state = robot.data.default_root_state.clone()
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()

        root_state[:, :3] = root_pos_t[frame : frame + 1]
        root_state[:, :2] += scene.env_origins[:, :2]
        root_state[:, 3:7] = root_quat_t[frame : frame + 1]
        root_state[:, 7:13] = 0.0

        joint_pos[:, joint_ids] = joint_pos_t[frame : frame + 1]
        joint_vel[:, joint_ids] = joint_vel_t[frame : frame + 1]

        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.render()
        scene.update(sim.get_physics_dt())

        lookat = root_state[0, :3].cpu().numpy()
        sim.set_camera_view(lookat + np.array([2.0, 2.0, 0.8]), lookat)

        frame += 1
        if frame >= num_frames:
            frame = 0


if __name__ == "__main__":
    try:
        run()
    finally:
        simulation_app.close()
