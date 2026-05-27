#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Flexible Isaac Lab FK recomputation for 23DOF G1 NPZ files.")
parser.add_argument("--input-root", type=str, default=None, help="Dataset root or flat folder containing npz files.")
parser.add_argument("--input-file", type=str, default=None, help="Optional single external npz file.")
parser.add_argument("--output-root", type=str, required=True)
parser.add_argument("--subset", type=str, choices=["single", "merged", "all"], default="all")
parser.add_argument("--category", type=str, default=None, help="Optional category filter for standard layout.")
parser.add_argument("--overwrite", action="store_true")
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
    "left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint","left_knee_joint",
    "left_ankle_pitch_joint","left_ankle_roll_joint","right_hip_pitch_joint","right_hip_roll_joint",
    "right_hip_yaw_joint","right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint",
    "waist_yaw_joint","left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint",
    "left_elbow_joint","left_wrist_roll_joint","right_shoulder_pitch_joint","right_shoulder_roll_joint",
    "right_shoulder_yaw_joint","right_elbow_joint","right_wrist_roll_joint",
]

@configclass
class FKSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

def _find_root_body_index(body_names: list[str]) -> int:
    if "pelvis" in body_names:
        return body_names.index("pelvis")
    return 0

def _reorder_pelvis_first(body_names: list[str]) -> Tuple[list[str], list[int]]:
    if "pelvis" not in body_names:
        return body_names, list(range(len(body_names)))
    pelvis_idx = body_names.index("pelvis")
    order = [pelvis_idx] + [i for i in range(len(body_names)) if i != pelvis_idx]
    return [body_names[i] for i in order], order

def _is_merged_bank_npz(path: Path) -> bool:
    try:
        data = np.load(path, allow_pickle=True)
        return "clip_names" in data.files and "clip_lengths" in data.files and "clip_start_indices" in data.files
    except Exception:
        return False

def _infer_output_relpath(path: Path, input_root: Path | None) -> Path:
    if input_root is not None:
        try:
            rel = path.relative_to(input_root)
            if "single" in path.parts or "merged" in path.parts:
                return rel
        except Exception:
            pass
    if _is_merged_bank_npz(path):
        return Path("merged") / path.name
    return Path("single") / "external" / path.name

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
    return data, joint_names, joint_pos, joint_vel, root_pos, root_quat, root_lin, root_ang

def _recompute_body_fields(scene, sim, joint_names, joint_pos_np, joint_vel_np, root_pos_np, root_quat_np, root_lin_np, root_ang_np):
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
        root_state[:, :3] = torch.as_tensor(root_pos_np[t:t+1], dtype=torch.float32, device=device)
        root_state[:, :2] += env_xy
        root_state[:, 3:7] = torch.as_tensor(root_quat_np[t:t+1], dtype=torch.float32, device=device)
        root_state[:, 7:10] = torch.as_tensor(root_lin_np[t:t+1], dtype=torch.float32, device=device)
        root_state[:, 10:13] = torch.as_tensor(root_ang_np[t:t+1], dtype=torch.float32, device=device)

        joint_pos[:] = robot.data.default_joint_pos
        joint_vel[:] = robot.data.default_joint_vel
        joint_pos[:, joint_ids] = torch.as_tensor(joint_pos_np[t:t+1], dtype=torch.float32, device=device)
        joint_vel[:, joint_ids] = torch.as_tensor(joint_vel_np[t:t+1], dtype=torch.float32, device=device)

        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.render()
        scene.update(sim.get_physics_dt())

        body_pos_frame = robot.data.body_link_pos_w[0].detach().cpu().numpy().astype(np.float32)
        body_quat_frame = robot.data.body_link_quat_w[0].detach().cpu().numpy().astype(np.float32)
        body_lin_frame = robot.data.body_link_lin_vel_w[0].detach().cpu().numpy().astype(np.float32)
        body_ang_frame = robot.data.body_link_ang_vel_w[0].detach().cpu().numpy().astype(np.float32)

        body_pos_frame[:, :2] -= scene.env_origins[0, :2].cpu().numpy()

        body_pos_out[t] = body_pos_frame[body_order]
        body_quat_out[t] = body_quat_frame[body_order]
        body_lin_out[t] = body_lin_frame[body_order]
        body_ang_out[t] = body_ang_frame[body_order]

        if t % 1000 == 0:
            print(f"    frame {t}/{n_frames}")

    return body_names, body_pos_out, body_quat_out, body_lin_out, body_ang_out

def _save_with_recomputed_body_fields(src_data, out_path: Path, body_names, body_pos, body_quat, body_lin, body_ang, source_path: str):
    out = {key: src_data[key] for key in src_data.files}
    out["body_names"] = np.asarray(body_names, dtype=np.str_)
    for k in ("body_positions", "body_pos_w"):
        if k in out:
            out[k] = body_pos.astype(np.float32)
    for k in ("body_rotations", "body_quat_w"):
        if k in out:
            out[k] = body_quat.astype(np.float32)
    for k in ("body_linear_velocities", "body_lin_vel_w"):
        if k in out:
            out[k] = body_lin.astype(np.float32)
    for k in ("body_angular_velocities", "body_ang_vel_w"):
        if k in out:
            out[k] = body_ang.astype(np.float32)
    out["fk_recomputed_with"] = np.array("isaaclab_g1_23dof_flexible", dtype=np.str_)
    out["fk_source_path"] = np.array(source_path, dtype=np.str_)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)

def _iter_input_files(input_root: Path | None, subset: str, category: str | None, input_file: Path | None):
    if input_file is not None:
        yield input_file, _infer_output_relpath(input_file, None)
        return
    if input_root is None:
        raise ValueError("Either --input-root or --input-file must be provided")

    seen = set()
    if subset in ("single", "all"):
        single_dir = input_root / "single"
        if single_dir.exists():
            for path in sorted(single_dir.rglob("*.npz")):
                if category is not None and path.parent.name != category:
                    continue
                seen.add(path.resolve())
                yield path, path.relative_to(input_root)
    if subset in ("merged", "all"):
        merged_dir = input_root / "merged"
        if merged_dir.exists():
            for path in sorted(merged_dir.glob("*.npz")):
                if category is not None and not path.name.startswith(f"{category}_"):
                    continue
                seen.add(path.resolve())
                yield path, path.relative_to(input_root)

    for path in sorted(input_root.glob("*.npz")):
        if path.resolve() in seen:
            continue
        if subset == "merged" and not _is_merged_bank_npz(path):
            continue
        if subset == "single" and _is_merged_bank_npz(path):
            continue
        yield path, _infer_output_relpath(path, input_root)

def main():
    input_root = Path(args_cli.input_root).expanduser().resolve() if args_cli.input_root else None
    input_file = Path(args_cli.input_file).expanduser().resolve() if args_cli.input_file else None
    output_root = Path(args_cli.output_root).expanduser().resolve()

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / 60.0
    sim = SimulationContext(sim_cfg)

    scene_cfg = FKSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO] FK scene ready")

    files = list(_iter_input_files(input_root, args_cli.subset, args_cli.category, input_file))
    if not files:
        raise FileNotFoundError(f"No input files found. input_root={input_root}, input_file={input_file}, subset={args_cli.subset}, category={args_cli.category}")

    print(f"[INFO] Found {len(files)} files to process")
    for src_path, rel_path in files:
        out_path = output_root / rel_path
        if out_path.exists() and not args_cli.overwrite:
            print(f"[SKIP] {out_path} already exists")
            continue

        print(f"[PROCESS] {src_path}")
        src_data, joint_names, joint_pos, joint_vel, root_pos, root_quat, root_lin, root_ang = _load_npz_motion(src_path)

        if joint_names != TARGET_G1_23_DOF_NAMES:
            print("[WARN] input joint names differ from TARGET_G1_23_DOF_NAMES")
            print("       input =", joint_names)
            print("       target =", TARGET_G1_23_DOF_NAMES)

        body_names, body_pos, body_quat, body_lin, body_ang = _recompute_body_fields(
            scene, sim, joint_names, joint_pos, joint_vel, root_pos, root_quat, root_lin, root_ang
        )

        _save_with_recomputed_body_fields(
            src_data=src_data, out_path=out_path, body_names=body_names,
            body_pos=body_pos, body_quat=body_quat, body_lin=body_lin, body_ang=body_ang,
            source_path=str(src_path),
        )
        print(f"[OK] Saved {out_path}")

    print("[DONE]")

if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
