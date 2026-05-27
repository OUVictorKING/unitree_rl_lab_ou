#!/usr/bin/env python3
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Replay 23DOF G1 motion from single or merged NPZ."
)
parser.add_argument("--input_npz", type=str, required=True)
parser.add_argument(
    "--clip-index", type=int, default=None, help="Only for merged bank npz."
)
parser.add_argument(
    "--clip-name", type=str, default=None, help="Only for merged bank npz."
)
parser.add_argument("--fps", type=int, default=None, help="Override fps.")
parser.add_argument(
    "--real-time-sleep",
    action="store_true",
    help="Sleep(dt) for slower realtime playback.",
)
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


@configclass
class ReplaySceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _scalar(x):
    arr = np.asarray(x)
    if arr.ndim == 0:
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return x


def _extract_clip_from_data(data):
    if "clip_names" in data:
        clip_names = data["clip_names"].tolist()
        clip_lengths = data["clip_lengths"].astype(int).tolist()
        clip_starts = data["clip_start_indices"].astype(int).tolist()
        if args_cli.clip_name is not None:
            if args_cli.clip_name not in clip_names:
                raise ValueError(f"clip-name not found: {args_cli.clip_name}")
            idx = clip_names.index(args_cli.clip_name)
        else:
            idx = 0 if args_cli.clip_index is None else int(args_cli.clip_index)
            if idx < 0 or idx >= len(clip_names):
                raise ValueError(f"clip-index out of range: {idx}")
        start = clip_starts[idx]
        length = clip_lengths[idx]
        end = start + length
        selected_name = clip_names[idx]
        print(
            f"[INFO] merged bank detected, replaying clip [{idx}] {selected_name}, frames={length}, range=[{start}, {end})"
        )
        return slice(start, end), selected_name
    print("[INFO] single clip detected")
    return slice(None), Path(args_cli.input_npz).stem


def load_motion_npz(path: str | Path):
    data = np.load(path, allow_pickle=True)
    clip_slice, clip_name = _extract_clip_from_data(data)

    if "joint_pos" in data and "body_pos_w" in data and "body_quat_w" in data:
        joint_pos = data["joint_pos"][clip_slice].astype(np.float32)
        joint_vel = data["joint_vel"][clip_slice].astype(np.float32)
        root_pos = data["body_pos_w"][clip_slice, 0].astype(np.float32)
        root_quat = data["body_quat_w"][clip_slice, 0].astype(np.float32)
        fps = int(_scalar(data["fps"]))
        joint_names = data["joint_names"].tolist()
    elif (
        "dof_positions" in data
        and "body_positions" in data
        and "body_rotations" in data
    ):
        joint_pos = data["dof_positions"][clip_slice].astype(np.float32)
        joint_vel = data["dof_velocities"][clip_slice].astype(np.float32)
        root_pos = data["body_positions"][clip_slice, 0].astype(np.float32)
        root_quat = data["body_rotations"][clip_slice, 0].astype(np.float32)
        fps = int(_scalar(data["fps"]))
        joint_names = data["dof_names"].tolist()
    else:
        raise ValueError(f"Unsupported npz format: {path}")

    if len(joint_names) != 23:
        raise ValueError(
            f"Replay script expects 23DOF npz, got {len(joint_names)} joints.\njoint_names={joint_names}"
        )
    if joint_names != TARGET_G1_23_DOF_NAMES:
        print("[WARN] joint_names differ from TARGET_G1_23_DOF_NAMES.")
        print("[WARN] file joint_names =", joint_names)
        print("[WARN] target joint_names =", TARGET_G1_23_DOF_NAMES)

    return fps, root_pos, root_quat, joint_pos, joint_vel, joint_names, clip_name


def run():
    fps, root_pos, root_quat, joint_pos_np, joint_vel_np, joint_names, clip_name = (
        load_motion_npz(args_cli.input_npz)
    )
    if args_cli.fps is not None:
        fps = int(args_cli.fps)

    print(f"[INFO] clip={clip_name}, frames={joint_pos_np.shape[0]}, fps={fps}")
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

        if args_cli.real_time_sleep:
            time.sleep(1.0 / fps)


if __name__ == "__main__":
    try:
        run()
    finally:
        simulation_app.close()
