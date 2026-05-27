#!/usr/bin/env python3
from __future__ import annotations
"""
Download the full ember-lab-berkeley/LAFAN-G1 dataset and build
same-category merged motion-bank NPZ files.

What it does:
1) snapshot_download the whole dataset from Hugging Face
2) scan all LAFAN_*.npz files
3) group them by category:
   - walk, run, sprint, dance, jumps, fight, fightAndSports, fallAndGetUp, ...
4) optionally remap 29DOF -> 23DOF for Unitree G1-23DOF
5) save:
   a) normalized single-clip NPZ files
   b) merged category-bank NPZ files (concatenated clips + clip metadata)

Example:
    python download_and_merge_ember_lafan_g1.py \
      --download-dir /home/woan/HumanoidProject/unitree_rl_lab/motion_datasets/g1/ember_lafan_g1_raw \
      --output-dir /home/woan/HumanoidProject/unitree_rl_lab/motion_datasets/g1/ember_lafan_g1_processed \
      --target-dof 23 \
      --copy-original-single-clips

Notes:
- The source dataset is AMP-style NPZ.
- The merged bank NPZ is NOT a single motion clip. It is a category motion bank:
    clip_names
    clip_start_indices
    clip_lengths
  plus concatenated arrays in both AMP-style keys and mimic-style aliases.
- Use allow_pickle=True when loading string arrays from NPZ.
"""

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from huggingface_hub import snapshot_download

KEEP_DOF_29_TO_23 = [
    0, 1, 2, 3, 4, 5,
    6, 7, 8, 9, 10, 11,
    12,
    15, 16, 17, 18, 19,
    22, 23, 24, 25, 26,
]

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


@dataclass
class MotionClip:
    name: str
    category: str
    fps: int
    joint_names: list[str]
    body_names: list[str]
    dof_positions: np.ndarray
    dof_velocities: np.ndarray
    body_positions: np.ndarray
    body_rotations: np.ndarray
    body_linear_velocities: np.ndarray
    body_angular_velocities: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--repo-id", type=str, default="ember-lab-berkeley/LAFAN-G1")
    parser.add_argument("--target-dof", type=int, choices=[23, 29], default=23)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--copy-original-single-clips", action="store_true")
    return parser.parse_args()


def extract_category_from_filename(path: Path) -> str:
    stem = path.stem
    if not stem.startswith("LAFAN_"):
        raise ValueError(f"Unexpected file name: {path.name}")
    middle = stem[len("LAFAN_"):]
    motion_token = middle.split("_subject")[0]
    category = re.sub(r"\d+$", "", motion_token)
    if not category:
        raise ValueError(f"Could not parse category from: {path.name}")
    return category


def discover_npz_files(download_dir: Path) -> list[Path]:
    files = sorted(download_dir.glob("LAFAN_*.npz"))
    if not files:
        raise FileNotFoundError(f"No LAFAN_*.npz files found under {download_dir}")
    return files


def load_amp_npz(path: Path) -> MotionClip:
    data = np.load(path, allow_pickle=True)
    required = [
        "fps",
        "dof_names",
        "body_names",
        "dof_positions",
        "dof_velocities",
        "body_positions",
        "body_rotations",
        "body_linear_velocities",
        "body_angular_velocities",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"{path.name} missing keys: {missing}")

    fps = int(data["fps"][0] if np.ndim(data["fps"]) > 0 else data["fps"])
    joint_names = [str(x) for x in data["dof_names"].tolist()]
    body_names = [str(x) for x in data["body_names"].tolist()]

    return MotionClip(
        name=path.stem,
        category=extract_category_from_filename(path),
        fps=fps,
        joint_names=joint_names,
        body_names=body_names,
        dof_positions=np.asarray(data["dof_positions"], dtype=np.float32),
        dof_velocities=np.asarray(data["dof_velocities"], dtype=np.float32),
        body_positions=np.asarray(data["body_positions"], dtype=np.float32),
        body_rotations=np.asarray(data["body_rotations"], dtype=np.float32),
        body_linear_velocities=np.asarray(data["body_linear_velocities"], dtype=np.float32),
        body_angular_velocities=np.asarray(data["body_angular_velocities"], dtype=np.float32),
    )


def maybe_remap_29_to_23(clip: MotionClip, target_dof: int) -> MotionClip:
    if target_dof == 29:
        return clip
    if len(clip.joint_names) == 23:
        return clip
    if len(clip.joint_names) != 29:
        raise ValueError(f"{clip.name}: expected 29 or 23 dof names, got {len(clip.joint_names)}")

    joint_names_23 = [clip.joint_names[i] for i in KEEP_DOF_29_TO_23]
    if joint_names_23 != TARGET_G1_23_DOF_NAMES:
        raise ValueError(
            f"{clip.name}: remapped joint names do not match G1-23 target order.\n"
            f"Got: {joint_names_23}\nExpected: {TARGET_G1_23_DOF_NAMES}"
        )

    return MotionClip(
        name=clip.name,
        category=clip.category,
        fps=clip.fps,
        joint_names=joint_names_23,
        body_names=clip.body_names,
        dof_positions=clip.dof_positions[:, KEEP_DOF_29_TO_23],
        dof_velocities=clip.dof_velocities[:, KEEP_DOF_29_TO_23],
        body_positions=clip.body_positions,
        body_rotations=clip.body_rotations,
        body_linear_velocities=clip.body_linear_velocities,
        body_angular_velocities=clip.body_angular_velocities,
    )


def save_single_clip_npz(out_path: Path, clip: MotionClip) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        fps=np.array(clip.fps, dtype=np.int32),
        dof_names=np.asarray(clip.joint_names, dtype=np.str_),
        body_names=np.asarray(clip.body_names, dtype=np.str_),
        dof_positions=clip.dof_positions.astype(np.float32),
        dof_velocities=clip.dof_velocities.astype(np.float32),
        body_positions=clip.body_positions.astype(np.float32),
        body_rotations=clip.body_rotations.astype(np.float32),
        body_linear_velocities=clip.body_linear_velocities.astype(np.float32),
        body_angular_velocities=clip.body_angular_velocities.astype(np.float32),
        joint_names=np.asarray(clip.joint_names, dtype=np.str_),
        joint_pos=clip.dof_positions.astype(np.float32),
        joint_vel=clip.dof_velocities.astype(np.float32),
        body_pos_w=clip.body_positions.astype(np.float32),
        body_quat_w=clip.body_rotations.astype(np.float32),
        body_lin_vel_w=clip.body_linear_velocities.astype(np.float32),
        body_ang_vel_w=clip.body_angular_velocities.astype(np.float32),
    )


def build_category_bank(clips: list[MotionClip], category: str, target_dof: int) -> dict:
    if not clips:
        raise ValueError(f"No clips for category={category}")

    joint_names = clips[0].joint_names
    body_names = clips[0].body_names
    fps = clips[0].fps

    for clip in clips[1:]:
        if clip.joint_names != joint_names:
            raise ValueError(f"{category}: joint_names mismatch in {clip.name}")
        if clip.body_names != body_names:
            raise ValueError(f"{category}: body_names mismatch in {clip.name}")
        if clip.fps != fps:
            raise ValueError(f"{category}: fps mismatch in {clip.name}")

    clip_names = []
    clip_lengths = []
    clip_start_indices = []

    pos_list = []
    vel_list = []
    body_pos_list = []
    body_rot_list = []
    body_lin_list = []
    body_ang_list = []

    start = 0
    for clip in clips:
        length = clip.dof_positions.shape[0]
        clip_names.append(clip.name)
        clip_lengths.append(length)
        clip_start_indices.append(start)

        pos_list.append(clip.dof_positions)
        vel_list.append(clip.dof_velocities)
        body_pos_list.append(clip.body_positions)
        body_rot_list.append(clip.body_rotations)
        body_lin_list.append(clip.body_linear_velocities)
        body_ang_list.append(clip.body_angular_velocities)
        start += length

    dof_positions = np.concatenate(pos_list, axis=0).astype(np.float32)
    dof_velocities = np.concatenate(vel_list, axis=0).astype(np.float32)
    body_positions = np.concatenate(body_pos_list, axis=0).astype(np.float32)
    body_rotations = np.concatenate(body_rot_list, axis=0).astype(np.float32)
    body_linear_velocities = np.concatenate(body_lin_list, axis=0).astype(np.float32)
    body_angular_velocities = np.concatenate(body_ang_list, axis=0).astype(np.float32)

    return {
        "format_version": np.array("motion_bank_v1", dtype=np.str_),
        "source_repo": np.array("ember-lab-berkeley/LAFAN-G1", dtype=np.str_),
        "category": np.array(category, dtype=np.str_),
        "target_dof": np.array(target_dof, dtype=np.int32),
        "fps": np.array(fps, dtype=np.int32),
        "num_clips": np.array(len(clips), dtype=np.int32),
        "clip_names": np.asarray(clip_names, dtype=np.str_),
        "clip_lengths": np.asarray(clip_lengths, dtype=np.int32),
        "clip_start_indices": np.asarray(clip_start_indices, dtype=np.int32),
        "dof_names": np.asarray(joint_names, dtype=np.str_),
        "body_names": np.asarray(body_names, dtype=np.str_),
        "dof_positions": dof_positions,
        "dof_velocities": dof_velocities,
        "body_positions": body_positions,
        "body_rotations": body_rotations,
        "body_linear_velocities": body_linear_velocities,
        "body_angular_velocities": body_angular_velocities,
        "joint_names": np.asarray(joint_names, dtype=np.str_),
        "joint_pos": dof_positions,
        "joint_vel": dof_velocities,
        "body_pos_w": body_positions,
        "body_quat_w": body_rotations,
        "body_lin_vel_w": body_linear_velocities,
        "body_ang_vel_w": body_angular_velocities,
    }


def save_category_bank(out_path: Path, bank: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **bank)


def main() -> None:
    args = parse_args()
    download_dir = Path(args.download_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not args.no_download:
        print(f"[INFO] Downloading full dataset: {args.repo_id}")
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=str(download_dir),
            local_dir_use_symlinks=False,
        )
        print(f"[INFO] Download complete: {download_dir}")
    else:
        print(f"[INFO] Skipping download, using local directory: {download_dir}")

    files = discover_npz_files(download_dir)
    print(f"[INFO] Found {len(files)} NPZ files")

    grouped: Dict[str, List[MotionClip]] = defaultdict(list)

    for path in files:
        clip = load_amp_npz(path)
        if args.category is not None and clip.category != args.category:
            continue
        clip = maybe_remap_29_to_23(clip, args.target_dof)
        grouped[clip.category].append(clip)

        if args.copy_original_single_clips:
            single_dir = output_dir / "single" / clip.category
            suffix = f"_{args.target_dof}dof.npz"
            save_single_clip_npz(single_dir / f"{clip.name}{suffix}", clip)

    if not grouped:
        raise ValueError("No clips matched the requested filters.")

    print("[INFO] Categories discovered:")
    for category in sorted(grouped.keys()):
        print(f"  - {category}: {len(grouped[category])} clips")

    for category in sorted(grouped.keys()):
        clips = sorted(grouped[category], key=lambda c: c.name)
        bank = build_category_bank(clips, category=category, target_dof=args.target_dof)
        out_path = output_dir / "merged" / f"{category}_bank_{args.target_dof}dof.npz"
        save_category_bank(out_path, bank)
        print(f"[OK] Saved category bank: {out_path}")

    print("[DONE]")


if __name__ == "__main__":
    main()
