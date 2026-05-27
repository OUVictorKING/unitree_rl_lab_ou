#!/usr/bin/env python3
from __future__ import annotations
"""
Recompute body pose/velocity fields using Isaac Lab FK for all processed 23DOF ember files.

Input layout (expected):
    <input_root>/
      single/<category>/*.npz
      merged/*_bank_23dof.npz

Output layout:
    <output_root>/
      single/<category>/*.npz
      merged/*_bank_23dof.npz

This script:
- loads 23DOF single clips and merged banks
- extracts root pose/velocity from the root body ("pelvis" if present, else index 0)
- writes root state + joint state into Isaac Lab
- reads back body_link_pos_w/body_link_quat_w/body_link_lin_vel_w/body_link_ang_vel_w
- overwrites body_* fields in both AMP-style and mimic-style keys
- preserves clip metadata for merged banks

Recommended run:
    python recompute_fk_ember_23dof.py \
      --input-root /home/woan/HumanoidProject/unitree_rl_lab/motion_datasets/g1/ember_lafan_g1_processed \
      --output-root /home/woan/HumanoidProject/unitree_rl_lab/motion_datasets/g1/ember_lafan_g1_processed/FK_finnal \
      --headless

Notes:
- The AppLauncher supports `--headless`, which launches Isaac Sim without GUI.
- Uses articulation body link states in simulation world frame.
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Recompute body arrays with Isaac Lab FK for 23DOF ember files.")
parser.add_argument(
    "--input-root",
    type=str,
    default="/home/woan/HumanoidProject/unitree_rl_lab/motion_datasets/g1/ember_lafan_g1_processed",
)
parser.add_argument(
    "--output-root",
    type=str,
    default="/home/woan/HumanoidProject/unitree_rl_lab/motion_datasets/g1/ember_lafan_g1_processed/FK_finnal",
)
parser.add_argument(
    "--subset",
    type=str,
    choices=["single", "merged", "all"],
    default="all",
)
parser.add_argument(
    "--category",
    type=str,
    default=None,
    help="Optional category filter, e.g. walk/run/dance",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite existing outputs",
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
class FKSceneCfg(InteractiveSceneCfg):
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


def _scalar(x):
    arr = np.asarray(x)
    if arr.ndim == 0:
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return x


def _find_root_body_index(body_names: list[str]) -> int:
    if "pelvis" in body_names:
        return body_names.index("pelvis")
    return 0


def _reorder_pelvis_first(body_names: list[str]) -> Tuple[list[str], list[int]]:
    if "pelvis" not in body_names:
        return body_names, list(range(len(body_names)))
    pelvis_idx = body_names.index("pelvis")
    order = [pelvis_idx] + [i for i in range(len(body_names)) if i != pelvis_idx]
    reordered = [body_names[i] for i in order]
    return reordered, order


def _load_npz_motion(path: Path):
    data = np.load(path, allow_pickle=True)

    if "joint_pos" in data and "body_pos_w" in data and "body_quat_w" in data:
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
        joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
        body_pos = np.asarray(data["body_pos_w"], dtype=np.float32)
        body_quat = np.asarray(data["body_quat_w"], dtype=np.float32)
        body_lin = np.asarray(data["body_lin_vel_w"], dtype=np.float32)
        body_ang = np.asarray(data["body_ang_vel_w"], dtype=np.float32)
        joint_names = [str(x) for x in data["joint_names"].tolist()]
        body_names = [str(x) for x in data["body_names"].tolist()]
    elif "dof_positions" in data and "body_positions" in data and "body_rotations" in data:
        joint_pos = np.asarray(data["dof_positions"], dtype=np.float32)
        joint_vel = np.asarray(data["dof_velocities"], dtype=np.float32)
        body_pos = np.asarray(data["body_positions"], dtype=np.float32)
        body_quat = np.asarray(data["body_rotations"], dtype=np.float32)
        body_lin = np.asarray(data["body_linear_velocities"], dtype=np.float32)
        body_ang = np.asarray(data["body_angular_velocities"], dtype=np.float32)
        joint_names = [str(x) for x in data["dof_names"].tolist()]
        body_names = [str(x) for x in data["body_names"].tolist()]
    else:
        raise ValueError(f"Unsupported npz format: {path}")

    if len(joint_names) != 23:
        raise ValueError(f"{path.name}: expected 23DOF input, got {len(joint_names)} joints")

    root_idx = _find_root_body_index(body_names)
    root_pos = body_pos[:, root_idx].astype(np.float32)
    root_quat = body_quat[:, root_idx].astype(np.float32)
    root_lin = body_lin[:, root_idx].astype(np.float32)
    root_ang = body_ang[:, root_idx].astype(np.float32)

    return data, joint_names, body_names, joint_pos, joint_vel, root_pos, root_quat, root_lin, root_ang


def _recompute_body_fields(
    scene: InteractiveScene,
    sim: SimulationContext,
    joint_names: list[str],
    joint_pos_np: np.ndarray,
    joint_vel_np: np.ndarray,
    root_pos_np: np.ndarray,
    root_quat_np: np.ndarray,
    root_lin_np: np.ndarray,
    root_ang_np: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    robot = scene["robot"]
    joint_ids = robot.find_joints(joint_names, preserve_order=True)[0]

    body_names_src = list(robot.body_names)
    body_names, body_order = _reorder_pelvis_first(body_names_src)

    n_frames = joint_pos_np.shape[0]
    n_bodies = len(body_names_src)

    body_pos_out = np.zeros((n_frames, n_bodies, 3), dtype=np.float32)
    body_quat_out = np.zeros((n_frames, n_bodies, 4), dtype=np.float32)
    body_lin_out = np.zeros((n_frames, n_bodies, 3), dtype=np.float32)
    body_ang_out = np.zeros((n_frames, n_bodies, 3), dtype=np.float32)

    root_state = robot.data.default_root_state.clone()
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()

    device = sim.device
    env_xy = scene.env_origins[:, :2]

    for t in range(n_frames):
        root_state[:, :3] = torch.as_tensor(root_pos_np[t : t + 1], dtype=torch.float32, device=device)
        root_state[:, :2] += env_xy
        root_state[:, 3:7] = torch.as_tensor(root_quat_np[t : t + 1], dtype=torch.float32, device=device)
        root_state[:, 7:10] = torch.as_tensor(root_lin_np[t : t + 1], dtype=torch.float32, device=device)
        root_state[:, 10:13] = torch.as_tensor(root_ang_np[t : t + 1], dtype=torch.float32, device=device)

        joint_pos[:] = robot.data.default_joint_pos
        joint_vel[:] = robot.data.default_joint_vel
        joint_pos[:, joint_ids] = torch.as_tensor(joint_pos_np[t : t + 1], dtype=torch.float32, device=device)
        joint_vel[:, joint_ids] = torch.as_tensor(joint_vel_np[t : t + 1], dtype=torch.float32, device=device)

        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.render()
        scene.update(sim.get_physics_dt())

        body_pos_frame = robot.data.body_link_pos_w[0].detach().cpu().numpy().astype(np.float32)
        body_quat_frame = robot.data.body_link_quat_w[0].detach().cpu().numpy().astype(np.float32)
        body_lin_frame = robot.data.body_link_lin_vel_w[0].detach().cpu().numpy().astype(np.float32)
        body_ang_frame = robot.data.body_link_ang_vel_w[0].detach().cpu().numpy().astype(np.float32)

        # remove env origin offset so saved data stays in the same world coordinates as input
        body_pos_frame[:, :2] -= scene.env_origins[0, :2].cpu().numpy()

        body_pos_out[t] = body_pos_frame[body_order]
        body_quat_out[t] = body_quat_frame[body_order]
        body_lin_out[t] = body_lin_frame[body_order]
        body_ang_out[t] = body_ang_frame[body_order]

        if t % 1000 == 0:
            print(f"    frame {t}/{n_frames}")

    return body_names, body_pos_out, body_quat_out, body_lin_out, body_ang_out


def _save_with_recomputed_body_fields(
    src_data,
    out_path: Path,
    body_names: list[str],
    body_pos: np.ndarray,
    body_quat: np.ndarray,
    body_lin: np.ndarray,
    body_ang: np.ndarray,
):
    out = {}
    for key in src_data.files:
        out[key] = src_data[key]

    # overwrite body-related keys in both styles
    out["body_names"] = np.asarray(body_names, dtype=np.str_)

    if "body_positions" in out:
        out["body_positions"] = body_pos.astype(np.float32)
    if "body_rotations" in out:
        out["body_rotations"] = body_quat.astype(np.float32)
    if "body_linear_velocities" in out:
        out["body_linear_velocities"] = body_lin.astype(np.float32)
    if "body_angular_velocities" in out:
        out["body_angular_velocities"] = body_ang.astype(np.float32)

    if "body_pos_w" in out:
        out["body_pos_w"] = body_pos.astype(np.float32)
    if "body_quat_w" in out:
        out["body_quat_w"] = body_quat.astype(np.float32)
    if "body_lin_vel_w" in out:
        out["body_lin_vel_w"] = body_lin.astype(np.float32)
    if "body_ang_vel_w" in out:
        out["body_ang_vel_w"] = body_ang.astype(np.float32)

    out["fk_recomputed_with"] = np.array("isaaclab_g1_23dof", dtype=np.str_)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)


def _iter_input_files(input_root: Path, subset: str, category: str | None):
    if subset in ("single", "all"):
        for path in sorted((input_root / "single").rglob("*.npz")):
            if category is not None and path.parent.name != category:
                continue
            rel = path.relative_to(input_root)
            yield path, rel

    if subset in ("merged", "all"):
        for path in sorted((input_root / "merged").glob("*.npz")):
            if category is not None and not path.name.startswith(f"{category}_"):
                continue
            rel = path.relative_to(input_root)
            yield path, rel


def main():
    input_root = Path(args_cli.input_root).expanduser().resolve()
    output_root = Path(args_cli.output_root).expanduser().resolve()

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / 60.0
    sim = SimulationContext(sim_cfg)

    scene_cfg = FKSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO] FK scene ready")

    files = list(_iter_input_files(input_root, args_cli.subset, args_cli.category))
    if not files:
        raise FileNotFoundError(f"No input files found under {input_root} for subset={args_cli.subset}, category={args_cli.category}")

    print(f"[INFO] Found {len(files)} files to process")

    for src_path, rel_path in files:
        out_path = output_root / rel_path
        if out_path.exists() and not args_cli.overwrite:
            print(f"[SKIP] {out_path} already exists")
            continue

        print(f"[PROCESS] {src_path}")
        (
            src_data,
            joint_names,
            body_names_in,
            joint_pos,
            joint_vel,
            root_pos,
            root_quat,
            root_lin,
            root_ang,
        ) = _load_npz_motion(src_path)

        if joint_names != TARGET_G1_23_DOF_NAMES:
            print("[WARN] input joint names differ from TARGET_G1_23_DOF_NAMES")
            print("       input =", joint_names)
            print("       target =", TARGET_G1_23_DOF_NAMES)

        body_names, body_pos, body_quat, body_lin, body_ang = _recompute_body_fields(
            scene=scene,
            sim=sim,
            joint_names=joint_names,
            joint_pos_np=joint_pos,
            joint_vel_np=joint_vel,
            root_pos_np=root_pos,
            root_quat_np=root_quat,
            root_lin_np=root_lin,
            root_ang_np=root_ang,
        )

        _save_with_recomputed_body_fields(
            src_data=src_data,
            out_path=out_path,
            body_names=body_names,
            body_pos=body_pos,
            body_quat=body_quat,
            body_lin=body_lin,
            body_ang=body_ang,
        )
        print(f"[OK] Saved {out_path}")

    print("[DONE]")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
