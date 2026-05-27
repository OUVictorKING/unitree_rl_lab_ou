#!/usr/bin/env python3
# Copyright (c) 2025, Unitree Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""Standalone sanity check for the time-aligned MotionDataset sampler.

Runs without Isaac Sim — stubs ``isaaclab.utils.math`` with local helpers
(matching the pattern in ``debug_amp_features.py``) and loads
``amp_features.py`` + ``motion_dataset.py`` directly.

Verifies:

1. Resampling grid length ``T_rs`` is auto-derived from ``npz_fps`` and
   ``env_step_dt`` (no hard-coded integer).
2. ``sample(B)`` returns ``(B, K*D)`` tensors where every slot is a pure
   gather off the pre-resampled buffer — no interp in the hot path.
3. Adjacent-slot alignment: ``next[:, k*D:(k+1)*D] == now[:, (k+1)*D:(k+2)*D]``
   for ``k ∈ [0, K-2]``.
4. Non-wrap valid-index range: ``idx_min = K-1``, ``idx_max = T_rs-2``.
5. ``motion_weights`` drives the empirical clip distribution.
6. Clips too short for the requested ``stack_k`` raise ``ValueError`` at
   init.
7. Resampling correctness on a non-linear (sine) feature — the max
   per-sample error vs the analytic sine on the dst grid is bounded by
   the linear-interp theoretical bound.

Usage
-----
    /home/woan/HumanoidProject/IsaacLab/_isaac_sim/python.sh \
        scripts/AMP/debug_motion_dataset.py
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
import torch


# ---- Stub isaaclab.utils.math (same math as debug_amp_features.py).
def _quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
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


# ---- Load amp_features + motion_dataset directly (without the package
# __init__, which pulls in tensordict / rsl_rl). We set up a fake package
# tree so the ``from ..features.amp_features`` relative import in
# motion_dataset.py resolves.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_FEATURES_PATH = (
    _REPO_ROOT
    / "source/unitree_rl_lab/unitree_rl_lab/rsl_rl_amp/features/amp_features.py"
)
_MOTION_DATASET_PATH = (
    _REPO_ROOT
    / "source/unitree_rl_lab/unitree_rl_lab/rsl_rl_amp/storage/motion_dataset.py"
)


def _load_module_as(name: str, path: Path, package: str | None = None):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    if package is not None:
        mod.__package__ = package
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Build a fake package tree rooted at "_ut_amp" so relative imports work.
_pkg = types.ModuleType("_ut_amp")
_pkg.__path__ = []  # mark as a package
_pkg_features = types.ModuleType("_ut_amp.features")
_pkg_features.__path__ = []
_pkg_storage = types.ModuleType("_ut_amp.storage")
_pkg_storage.__path__ = []
sys.modules["_ut_amp"] = _pkg
sys.modules["_ut_amp.features"] = _pkg_features
sys.modules["_ut_amp.storage"] = _pkg_storage

_features = _load_module_as("_ut_amp.features.amp_features", _FEATURES_PATH,
                            package="_ut_amp.features")
_pkg_features.amp_features = _features

_motion_dataset = _load_module_as("_ut_amp.storage.motion_dataset", _MOTION_DATASET_PATH,
                                  package="_ut_amp.storage")
_pkg_storage.motion_dataset = _motion_dataset

AmpObsSpec = _features.AmpObsSpec
MotionDataset = _motion_dataset.MotionDataset
_resample_phi_linear = _motion_dataset._resample_phi_linear


# ---- Test fixtures ---------------------------------------------------------

# Minimal spec: 1 joint, joint_pos_rel only. frame_dim = 1. stack_k variable.
_JOINT_NAMES: tuple[str, ...] = ("j0",)
_PELVIS = "pelvis"
_FEET = ("foot_L", "foot_R")
_HANDS = ("hand_L", "hand_R")
# amp_body_names must include pelvis + feet + hands at resolvable indices.
_BODY_NAMES: tuple[str, ...] = (_PELVIS, _FEET[0], _FEET[1], _HANDS[0], _HANDS[1])


def _minimal_spec(stack_k: int = 4) -> AmpObsSpec:
    """Single-joint spec with joint_pos_rel only (frame_dim = 1)."""
    return AmpObsSpec(
        joint_names=_JOINT_NAMES,
        pelvis_body_name=_PELVIS,
        foot_body_names=_FEET,
        hand_body_names=_HANDS,
        stack_k=int(stack_k),
        include_root_height=False,
        include_projected_gravity=False,
        include_root_lin_vel_heading=False,
        include_root_ang_vel_body=False,
        include_joint_pos_rel=True,
        include_joint_vel=False,
        include_feet_position=False,
        include_feet_orientation=False,
        include_feet_linear_velocity=False,
        include_feet_angular_velocity=False,
        include_hand_position=False,
        include_hand_orientation=False,
        include_hand_linear_velocity=False,
        include_hand_angular_velocity=False,
    )


def _synthetic_raw(
    T: int,
    fps: float,
    joint_pos_fn,
) -> dict[str, np.ndarray]:
    """Build an npz-dict-like payload.

    ``joint_pos_fn(t)`` supplies the joint_pos scalar for frame t (single
    joint). Bodies are filled with zero positions / identity quats / zero
    velocities so root-frame transforms are well-defined but irrelevant.
    """
    joint_pos = np.array([[joint_pos_fn(t)] for t in range(T)], dtype=np.float32)
    joint_vel = np.zeros_like(joint_pos)
    body_pos = np.zeros((T, len(_BODY_NAMES), 3), dtype=np.float32)
    body_quat = np.zeros((T, len(_BODY_NAMES), 4), dtype=np.float32)
    body_quat[..., 0] = 1.0  # wxyz identity
    body_lin_vel = np.zeros_like(body_pos)
    body_ang_vel = np.zeros_like(body_pos)
    return {
        "fps": float(fps),
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": body_lin_vel,
        "body_ang_vel_w": body_ang_vel,
        # Self-describing name lists — mirrors what csv_to_npz_final.py
        # writes. MotionDataset should prefer these over the caller's
        # amp_joint_names / amp_body_names (and cross-check them when
        # both are supplied).
        "joint_names": np.asarray(list(_JOINT_NAMES)),
        "body_names": np.asarray(list(_BODY_NAMES)),
    }


def _build_dataset(
    raws: list[dict[str, np.ndarray]],
    *,
    stack_k: int,
    env_step_dt: float,
    wrap_around: bool,
    motion_weights: list[float] | None = None,
) -> MotionDataset:
    """Build a MotionDataset while monkey-patching ``_load_npz`` so the
    synthetic raw dicts are used in place of on-disk files."""
    original_load = MotionDataset._load_npz

    def _fake_load(path: str) -> dict[str, np.ndarray]:
        return raws[int(path.rsplit("/", 1)[-1])]

    MotionDataset._load_npz = staticmethod(_fake_load)  # type: ignore[method-assign]
    try:
        fake_paths = [f"synthetic/clip/{i}" for i in range(len(raws))]
        ds = MotionDataset(
            motion_files=fake_paths,
            spec=_minimal_spec(stack_k),
            amp_joint_names=_JOINT_NAMES,
            amp_body_names=_BODY_NAMES,
            default_joint_pos=np.zeros(1, dtype=np.float32),
            env_step_dt=env_step_dt,
            motion_weights=motion_weights,
            wrap_around=wrap_around,
            device="cpu",
        )
    finally:
        MotionDataset._load_npz = staticmethod(original_load)  # type: ignore[method-assign]
    return ds


# ---- Checks ----------------------------------------------------------------

def check_T_rs_auto_derived() -> None:
    print("=== 1. T_rs auto-derived from fps_src + env_step_dt ===")
    raw = _synthetic_raw(T=110, fps=30.0, joint_pos_fn=lambda t: t * (1.0 / 30.0))
    ds = _build_dataset([raw], stack_k=4, env_step_dt=0.02, wrap_around=True)
    s = ds.summary()
    expected = int(round(110 * (1.0 / 30.0) / 0.02))
    print(f"  env_step_dt={s['env_step_dt']}, clip_fps_src={s['clip_fps_src']}, "
          f"T_src={s['clip_T_src']}, T_rs={s['clip_T_rs']}")
    assert s["clip_T_rs"] == [expected], (s["clip_T_rs"], expected)
    print(f"  [OK] T_rs={expected} matches round(T_src * frame_dt_src / env_step_dt).")


def check_sample_layout_wrap() -> None:
    print("\n=== 2. sample() shapes + membership + adjacency (wrap_around=True) ===")
    # Use a sine signal so wrap-around has no discontinuity artifacts —
    # the ramp test in check 3 uses wrap_around=False for clean value
    # reconstruction.
    fps = 30.0
    T_src = 120
    # One full period over the clip so sin(2π*T/T) == sin(0) — wrap is clean.
    raw = _synthetic_raw(
        T=T_src, fps=fps,
        joint_pos_fn=lambda t: math.sin(2.0 * math.pi * t / float(T_src)),
    )
    K = 4
    env_step_dt = 0.02
    ds = _build_dataset([raw], stack_k=K, env_step_dt=env_step_dt, wrap_around=True)
    T_rs = ds._clip_lengths_rs_list[0]
    phi_rs = ds._phi_rs_clips[0]
    D = ds.frame_dim
    assert D == 1, D

    torch.manual_seed(0)
    B = 256
    now, nxt = ds.sample(B)
    assert now.shape == (B, K * D), now.shape
    assert nxt.shape == (B, K * D), nxt.shape

    # Pure-gather invariant: every value in the output must be *exactly*
    # present in phi_rs (sample() must not do any on-the-fly interp).
    grid = set(float(v) for v in phi_rs[:, 0].tolist())
    for i in range(B):
        for k in range(K):
            v_now = float(now[i, k * D].item())
            v_nxt = float(nxt[i, k * D].item())
            assert v_now in grid, (i, k, "now", v_now)
            assert v_nxt in grid, (i, k, "nxt", v_nxt)
    print("  [OK] every sampled value is a verbatim phi_rs grid point (no hot-path interp).")

    # Adjacency invariant (signal-agnostic): the "next" window is the "now"
    # window shifted by one env_step_dt index, so next[:, k] == now[:, k+1]
    # for k ∈ [0, K-2].
    for k in range(K - 1):
        lhs = nxt[:, k * D:(k + 1) * D]
        rhs = now[:, (k + 1) * D:(k + 2) * D]
        assert torch.allclose(lhs, rhs, atol=1e-6), k
    print("  [OK] adjacent-slot alignment: next[k] == now[k+1] for k ∈ [0, K-2].")

    # Derive idx_now from newest slot via tensor matching (works for any
    # signal) and verify full-window alignment against phi_rs gather.
    newest = now[:, -1]  # (B,)
    # For each sample find j such that phi_rs[j, 0] == newest[i]. Values
    # in phi_rs may repeat (sine is non-injective) — pick the first match
    # and verify the whole window is consistent with one valid j.
    for i in range(B):
        matches = (phi_rs[:, 0] == newest[i]).nonzero(as_tuple=True)[0].tolist()
        assert matches, (i, float(newest[i]))
        # One of the matches must reproduce the full now/nxt window.
        found = False
        for j in matches:
            ok = True
            for k in range(K):
                want_now = float(phi_rs[(j - (K - 1 - k)) % T_rs, 0])
                want_nxt = float(phi_rs[(j - (K - 1 - k) + 1) % T_rs, 0])
                if not math.isclose(float(now[i, k * D]), want_now, abs_tol=1e-6):
                    ok = False
                    break
                if not math.isclose(float(nxt[i, k * D]), want_nxt, abs_tol=1e-6):
                    ok = False
                    break
            if ok:
                found = True
                break
        assert found, i
    print("  [OK] every sample's full now/nxt window matches a phi_rs gather at some valid idx.")


def check_non_wrap_valid_range() -> None:
    print("\n=== 3. Non-wrap valid-index range + no out-of-range samples ===")
    fps = 30.0
    raw = _synthetic_raw(T=60, fps=fps, joint_pos_fn=lambda t: float(t))
    K = 4
    env_step_dt = 0.02
    ds = _build_dataset([raw], stack_k=K, env_step_dt=env_step_dt, wrap_around=False)
    T_rs = ds._clip_lengths_rs_list[0]
    assert ds._clip_idx_min_list == [K - 1], ds._clip_idx_min_list
    assert ds._clip_idx_max_list == [T_rs - 2], ds._clip_idx_max_list
    print(f"  T_rs={T_rs}, idx_min={K - 1}, idx_max={T_rs - 2}")

    torch.manual_seed(1)
    B = 4096
    now, nxt = ds.sample(B)
    # Reconstruct the idx_now from the newest slot via the ramp.
    # With default_joint_pos=0 and joint_pos=t, phi_raw[t, 0] = t.
    # After resampling onto env_step_dt grid: phi_rs[j, 0] equals the linear
    # interp of t at τ = j * env_step_dt, expressed in *frame units*. That
    # is, phi_rs[j] = τ_j / frame_dt_src = j * env_step_dt * fps.
    vals = now[:, -1] / (env_step_dt * fps)
    idx_now = torch.round(vals).to(torch.long)
    assert int(idx_now.min()) >= K - 1, int(idx_now.min())
    assert int(idx_now.max()) <= T_rs - 2, int(idx_now.max())
    # Verify no sample straddles out-of-range in oldest / next.
    oldest_idx = idx_now - (K - 1)
    next_idx = idx_now + 1
    assert int(oldest_idx.min()) >= 0
    assert int(next_idx.max()) <= T_rs - 1
    print("  [OK] all sampled indices stay in [K-1, T_rs-2].")


def check_motion_weights_distribution() -> None:
    print("\n=== 4. motion_weights drives clip-id distribution ===")
    raw_a = _synthetic_raw(T=60, fps=30.0, joint_pos_fn=lambda t: 0.1)
    raw_b = _synthetic_raw(T=400, fps=60.0, joint_pos_fn=lambda t: 0.5)
    ds = _build_dataset(
        [raw_a, raw_b], stack_k=1, env_step_dt=0.02,
        wrap_around=True, motion_weights=[0.9, 0.1],
    )
    torch.manual_seed(2)
    B = 100_000
    now, _ = ds.sample(B)
    # Channel 0: clip A = 0.1, clip B = 0.5 (constants). Count samples.
    counts = torch.tensor(
        [int((now[:, 0].round(decimals=4) == 0.1).sum().item()),
         int((now[:, 0].round(decimals=4) == 0.5).sum().item())]
    )
    frac = counts.float() / float(B)
    print(f"  empirical fractions = {frac.tolist()}  (target [0.9, 0.1])")
    assert abs(frac[0].item() - 0.9) < 0.02, frac
    assert abs(frac[1].item() - 0.1) < 0.02, frac
    print("  [OK] empirical frac within 2% of configured weights.")


def check_too_short_clip_raises() -> None:
    print("\n=== 5. Too-short clip raises ValueError ===")
    # T_src=3, fps_src=30 => duration = 2/30 ≈ 0.0667s.
    # env_step_dt=0.05 => T_rs = floor(0.0667/0.05) + 1 = 2.
    # stack_k=4 => required = K+1 = 5. Must raise.
    raw = _synthetic_raw(T=3, fps=30.0, joint_pos_fn=lambda t: float(t))
    try:
        _build_dataset([raw], stack_k=4, env_step_dt=0.05, wrap_around=False)
    except ValueError as e:
        msg = str(e)
        assert "T_rs" in msg or "too short" in msg.lower() or "stack_k" in msg, msg
        print(f"  [OK] raised ValueError: {msg[:120]}...")
        return
    raise AssertionError("Expected ValueError on too-short clip.")


def check_resampling_sine_error() -> None:
    print("\n=== 6. Resampling correctness on sine (linear-interp bound) ===")
    T_src = 240
    fps_src = 60.0
    frame_dt_src = 1.0 / fps_src
    env_step_dt = 0.02
    # phi_raw(t) = sin(2π t / T_src) -- one period over the clip.
    # Build tensor directly to keep the test independent of the feature
    # pipeline.
    t = torch.arange(T_src, dtype=torch.float32)
    phi_raw = torch.sin(2.0 * math.pi * t / float(T_src)).unsqueeze(-1)  # (T_src, 1)

    T_rs = max(2, int(round(T_src * frame_dt_src / env_step_dt)))
    phi_rs = _resample_phi_linear(
        phi_raw=phi_raw,
        frame_dt_src=frame_dt_src,
        env_step_dt=env_step_dt,
        T_rs=T_rs,
        wrap_around=True,
    )
    # Analytic reference on the dst grid.
    tau_j = torch.arange(T_rs, dtype=torch.float32) * env_step_dt
    # Corresponding continuous time in clip units: τ_j / frame_dt_src.
    x_src_time = tau_j / frame_dt_src  # in "frame" units
    ref = torch.sin(2.0 * math.pi * x_src_time / float(T_src)).unsqueeze(-1)
    err = (phi_rs - ref).abs().max().item()
    # Linear-interp error bound for sin(ωt) over step h: ≤ (ω·h)²/8.
    omega = 2.0 * math.pi / float(T_src) / frame_dt_src  # rad/sec
    bound = (omega * env_step_dt) ** 2 / 8.0 + 1e-5
    print(f"  max abs error = {err:.3e}, theoretical bound ≈ {bound:.3e}")
    assert err < bound, (err, bound)
    print("  [OK] sine resample stays within linear-interp bound.")


def main() -> None:
    check_T_rs_auto_derived()
    check_sample_layout_wrap()
    check_non_wrap_valid_range()
    check_motion_weights_distribution()
    check_too_short_clip_raises()
    check_resampling_sine_error()
    print("\nAll MotionDataset sanity checks passed.")


if __name__ == "__main__":
    main()
