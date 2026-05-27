#!/usr/bin/env python3
# Copyright (c) 2025, Unitree Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""Standalone sanity check for Penguin AMP V1 features + curriculum.

Run without Isaac Sim — this script only touches pure-Python / torch code
paths. It verifies:

1. The V1 ``AmpObsSpec`` (every ``include_*`` on, ``stack_k=4``) produces
   the expected 80-dim frame and the matching stacked AMP obs dim
   (``frame_dim * stack_k``).
2. ``build_amp_frame_from_state`` on a synthetic zero-state produces a
   vector of exactly ``frame_dim``, with the documented layout offsets.
3. ``build_amp_window`` / ``concat_frame_history`` preserve the oldest-first
   ordering across a K-step shift.
4. ``AmpRewardCurriculum`` advances exactly once per
   ``required_consecutive_passes`` window, never retreats, and saturates
   at ``alpha_max``.

Usage
-----
    python scripts/AMP/debug_amp_features.py

No CLI flags — this is a smoke test, not a tunable utility.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch


# ---- Stub isaaclab.utils.math so we can load amp_features without Isaac Lab.
# We only need quat_rotate_inverse + yaw_quat (wxyz convention).
def _quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # q: (..., 4) wxyz, v: (..., 3). Returns R(q)^T @ v.
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    # Using the identity: R(q)^T v = 2*(q_v . v)*q_v + (w^2 - q_v . q_v)*v - 2*w*(q_v x v)
    qv_dot_v = x * vx + y * vy + z * vz
    qv_dot_qv = x * x + y * y + z * z
    w2 = w * w
    cx = y * vz - z * vy
    cy = z * vx - x * vz
    cz = x * vy - y * vx
    rx = 2.0 * qv_dot_v * x + (w2 - qv_dot_qv) * vx - 2.0 * w * cx
    ry = 2.0 * qv_dot_v * y + (w2 - qv_dot_qv) * vy - 2.0 * w * cy
    rz = 2.0 * qv_dot_v * z + (w2 - qv_dot_qv) * vz - 2.0 * w * cz
    return torch.stack([rx, ry, rz], dim=-1)


def _yaw_quat(q: torch.Tensor) -> torch.Tensor:
    # Extract yaw-only quaternion from a wxyz quaternion.
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = 0.5 * yaw
    out = torch.zeros_like(q)
    out[..., 0] = torch.cos(half)
    out[..., 3] = torch.sin(half)
    return out


_isaaclab_pkg = types.ModuleType("isaaclab")
_isaaclab_utils = types.ModuleType("isaaclab.utils")
_isaaclab_math = types.ModuleType("isaaclab.utils.math")
_isaaclab_math.quat_rotate_inverse = _quat_rotate_inverse
_isaaclab_math.quat_apply_inverse = _quat_rotate_inverse
_isaaclab_math.yaw_quat = _yaw_quat
sys.modules.setdefault("isaaclab", _isaaclab_pkg)
sys.modules.setdefault("isaaclab.utils", _isaaclab_utils)
sys.modules.setdefault("isaaclab.utils.math", _isaaclab_math)


# ---- Load the two modules we need directly, bypassing the package __init__
# (which pulls in tensordict / rsl_rl via amp_ppo).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_FEATURES_PATH = (
    _REPO_ROOT
    / "source/unitree_rl_lab/unitree_rl_lab/rsl_rl_amp/features/amp_features.py"
)
_CURRICULUM_PATH = (
    _REPO_ROOT
    / "source/unitree_rl_lab/unitree_rl_lab/rsl_rl_amp/algorithms/amp_curriculum.py"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_features = _load_module("_amp_features_std", _FEATURES_PATH)
_curriculum = _load_module("_amp_curriculum_std", _CURRICULUM_PATH)

AmpObsSpec = _features.AmpObsSpec
AmpObsState = _features.AmpObsState
build_amp_frame_from_state = _features.build_amp_frame_from_state
build_amp_window = _features.build_amp_window
concat_frame_history = _features.concat_frame_history

AmpRewardCurriculum = _curriculum.AmpRewardCurriculum
AmpRewardCurriculumCfg = _curriculum.AmpRewardCurriculumCfg


# G1 23-DoF articulation joint order — mirrors the per-env-cfg copies in
# tasks/amp/robots/g1/29dof/*.py. Kept local to this script so the smoke
# test stays self-contained.
_G1_23DOF_JOINT_NAMES: tuple[str, ...] = (
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


def _penguin_v1_spec(stack_k: int = 4) -> AmpObsSpec:
    """Local V1-equivalent spec (every include_* flag on)."""
    return AmpObsSpec(
        joint_names=_G1_23DOF_JOINT_NAMES,
        pelvis_body_name="pelvis",
        foot_body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
        hand_body_names=(
            "left_wrist_roll_rubber_hand",
            "right_wrist_roll_rubber_hand",
        ),
        stack_k=int(stack_k),
    )


# =========================================================================
# 1. Spec layout
# =========================================================================
def check_spec_layout() -> AmpObsSpec:
    spec = _penguin_v1_spec(stack_k=4)
    print("=== AmpObsSpec (V1-equivalent, stack_k=4) ===")
    print(f"  num_joints        : {spec.num_joints}")
    print(f"  frame_dim         : {spec.frame_dim}")
    print(f"  stack_k           : {spec.stack_k}")
    print(f"  amp_obs_dim       : {spec.amp_obs_dim}")
    print(f"  disc_input_dim    : {spec.discriminator_input_dim}")
    print("  frame_layout (name, dim):")
    offset = 0
    for name, dim in spec.frame_layout():
        print(f"    [{offset:3d}:{offset+dim:3d}] {name}  (+{dim})")
        offset += dim
    assert offset == spec.frame_dim, (offset, spec.frame_dim)
    expected = (
        1  # root_height
        + 3  # projected_gravity
        + 3  # root_lin_vel_h
        + 3  # root_ang_vel_b
        + spec.num_joints  # joint_pos_rel
        + spec.num_joints  # joint_vel
        + 6 + 6  # feet pos/vel (L+R)
        + 6 + 6  # hands pos/vel (L+R)
    )
    assert spec.frame_dim == expected, f"frame_dim {spec.frame_dim} != expected {expected}"
    assert spec.frame_dim == 80, f"Penguin V1 expects frame_dim=80, got {spec.frame_dim}"
    assert spec.amp_obs_dim == 320, f"expected amp_obs_dim=320, got {spec.amp_obs_dim}"
    print("  [OK] Penguin V1 expected dims confirmed (80 / 320).")
    return spec


# =========================================================================
# 2. Frame construction on synthetic state
# =========================================================================
def check_frame_construction(spec: AmpObsSpec) -> None:
    print("\n=== build_amp_frame_from_state (zero-state sanity) ===")
    B = 5
    J = spec.num_joints

    # Identity quat (wxyz), zero velocities, known root height.
    root_pos = torch.zeros(B, 3)
    root_pos[:, 2] = 0.78  # standing height
    root_quat = torch.zeros(B, 4)
    root_quat[:, 0] = 1.0  # wxyz identity

    state = AmpObsState(
        root_pos_w=root_pos,
        root_quat_w=root_quat,
        root_lin_vel_w=torch.zeros(B, 3),
        root_ang_vel_w=torch.zeros(B, 3),
        joint_pos=torch.zeros(B, J),
        joint_vel=torch.zeros(B, J),
        default_joint_pos=torch.zeros(J),
        foot_pos_w=torch.zeros(B, 2, 3),
        foot_lin_vel_w=torch.zeros(B, 2, 3),
        hand_pos_w=torch.zeros(B, 2, 3),
        hand_lin_vel_w=torch.zeros(B, 2, 3),
    )
    phi = build_amp_frame_from_state(state, spec)
    print(f"  phi shape: {tuple(phi.shape)}  (expected ({B}, {spec.frame_dim}))")
    assert phi.shape == (B, spec.frame_dim)

    # Index sanity: root_height column should be 0.78.
    assert torch.allclose(phi[:, 0], torch.full((B,), 0.78)), phi[:, 0]
    # projected_gravity in body frame for identity quat -> (0, 0, -1).
    assert torch.allclose(phi[:, 1:4], torch.tensor([0.0, 0.0, -1.0]).expand(B, 3))
    print("  [OK] root_height column == 0.78, proj_gravity == (0,0,-1).")


# =========================================================================
# 3. K-frame window ordering
# =========================================================================
def check_window_semantics(spec: AmpObsSpec) -> None:
    print("\n=== build_amp_window / concat_frame_history ordering ===")
    B = 3
    D = spec.frame_dim
    K = spec.stack_k

    # Make each frame's first channel encode its age-ID.
    history = torch.zeros(B, K, D)
    for k in range(K):
        history[:, k, 0] = float(k)  # 0 = oldest, K-1 = newest

    flat = build_amp_window(history, K)
    assert flat.shape == (B, K * D)
    # After flatten, the newest frame is the last slot.
    newest_slice = flat[:, -D:]
    assert torch.allclose(newest_slice[:, 0], torch.full((B,), float(K - 1)))
    oldest_slice = flat[:, :D]
    assert torch.allclose(oldest_slice[:, 0], torch.zeros(B))
    print("  window[...,0:D][:, 0] (oldest)  == 0.0  OK")
    print(f"  window[...,-D:][:, 0] (newest)  == {K - 1}.0  OK")

    # Advance the buffer with a new frame tagged K; we should drop the 0.
    new_frame = torch.zeros(B, D)
    new_frame[:, 0] = float(K)
    advanced = concat_frame_history(new_frame, history, K)
    assert advanced.shape == history.shape
    assert torch.allclose(advanced[:, 0, 0], torch.full((B,), 1.0))  # shifted
    assert torch.allclose(advanced[:, -1, 0], torch.full((B,), float(K)))
    print("  concat_frame_history: oldest dropped, newest appended.  OK")


# =========================================================================
# 4. Curriculum: monotonic advance + saturation
# =========================================================================
def check_curriculum() -> None:
    print("\n=== AmpRewardCurriculum (only-up gating) ===")

    cfg = AmpRewardCurriculumCfg(
        enabled=True,
        alpha_init=0.2,
        alpha_max=0.8,
        alpha_step=0.2,  # fewer steps for the smoke test
        warmup_updates=3,
        required_consecutive_passes=5,
        ema_alpha=1.0,  # no smoothing — use raw values
        episode_length_threshold=0.7,
        task_reward_threshold=0.6,
        termination_ratio_max=0.05,
        tracking_score_threshold=0.7,
    )
    curr = AmpRewardCurriculum(cfg)

    # Warmup: pass everything, expect no advance.
    for _ in range(cfg.warmup_updates):
        curr.update(
            episode_length_norm=0.95,
            task_reward_mean=0.9,
            termination_ratio=0.0,
            tracking_score=0.95,
        )
    assert curr.alpha_amp == cfg.alpha_init, (curr.alpha_amp, cfg.alpha_init)
    print(f"  after warmup ({cfg.warmup_updates}): alpha_amp = {curr.alpha_amp:.3f} (unchanged)")

    # One failing tick breaks the consecutive count.
    for _ in range(cfg.required_consecutive_passes - 1):
        curr.update(
            episode_length_norm=0.95,
            task_reward_mean=0.9,
            termination_ratio=0.0,
            tracking_score=0.95,
        )
    curr.update(
        episode_length_norm=0.2,
        task_reward_mean=0.1,
        termination_ratio=0.3,
        tracking_score=0.1,
    )
    assert curr.alpha_amp == cfg.alpha_init, "single failure must reset the streak"
    print("  one failure resets streak:  OK")

    # Run passes until saturated; count stages.
    last_alpha = curr.alpha_amp
    stages_seen = 0
    iterations = 0
    max_iter = cfg.required_consecutive_passes * 100
    while not curr.saturated and iterations < max_iter:
        curr.update(
            episode_length_norm=0.95,
            task_reward_mean=0.9,
            termination_ratio=0.0,
            tracking_score=0.95,
        )
        if curr.alpha_amp > last_alpha + 1e-9:
            stages_seen += 1
            assert curr.alpha_amp >= last_alpha, "curriculum must never retreat"
            last_alpha = curr.alpha_amp
        iterations += 1

    print(
        f"  stages advanced: {stages_seen}  final alpha = {curr.alpha_amp:.3f}  "
        f"saturated = {curr.saturated}"
    )
    expected_stages = int(round((cfg.alpha_max - cfg.alpha_init) / cfg.alpha_step))
    assert stages_seen == expected_stages, (stages_seen, expected_stages)
    assert curr.saturated
    assert curr.alpha_amp == cfg.alpha_max

    # After saturation, a pass cycle should not advance further.
    for _ in range(cfg.required_consecutive_passes * 2):
        curr.update(
            episode_length_norm=0.95,
            task_reward_mean=0.9,
            termination_ratio=0.0,
            tracking_score=0.95,
        )
    assert curr.alpha_amp == cfg.alpha_max
    print("  post-saturation: alpha_amp stays pinned at alpha_max.  OK")

    # Round-trip the state dict.
    s = curr.save_state()
    curr2 = AmpRewardCurriculum(cfg)
    curr2.load_state(s)
    assert curr2.alpha_amp == curr.alpha_amp
    assert curr2.stage == curr.stage
    print("  save_state / load_state round-trip:  OK")


def main() -> None:
    spec = check_spec_layout()
    check_frame_construction(spec)
    check_window_semantics(spec)
    check_curriculum()
    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    main()
