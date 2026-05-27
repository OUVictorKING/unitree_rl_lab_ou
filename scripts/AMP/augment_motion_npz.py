#!/usr/bin/env python3
# Copyright (c) 2025, Unitree Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""Offline data augmentation for AMP motion NPZ files.

Pipeline (order matters — interpolation first, then mirror, then concat):

    original_npz
        │
        ├─ temporal upsampling (SLERP for quaternions, linear otherwise)
        │     → interp_npz (k · T frames @ k · fps)
        │
        ├─ left↔right mirror of interp_npz (swap L/R joints + bodies,
        │     sign-flip roll/yaw joint values, negate world-y position /
        │     linear-velocity components, pseudo-vector flip for
        │     angular-velocity, wxyz quaternion mirror (w, -x, y, -z))
        │     → mirrored_npz (k · T frames @ k · fps)
        │
        └─ concat(interp_npz, mirrored_npz)
              → augmented_npz (2 · k · T frames @ k · fps)

Doing interpolation before the mirror avoids interpolating across the
seam between the original and its mirror — which would produce a
nonphysical "fold-over" pose halfway through.

NPZ schema (matches :mod:`unitree_rl_lab.rsl_rl_amp.storage.motion_dataset`):

    fps             shape (1,), int or float
    joint_pos       (T, J)           float32
    joint_vel       (T, J)           float32 (rad/s)
    body_pos_w      (T, N, 3)        float32
    body_quat_w     (T, N, 4)        float32, wxyz
    body_lin_vel_w  (T, N, 3)        float32
    body_ang_vel_w  (T, N, 3)        float32
    joint_names     (J,) optional    column-order joint names (self-describing)
    body_names      (N,) optional    row-order body names (self-describing)

``joint_names`` / ``body_names`` describe the articulation slot each
``joint_pos`` / ``body_pos_w`` column (respectively row) maps to. The
mirror operation swaps left/right **values** between columns; the name at
a given index is unchanged because it still describes which articulation
slot that column feeds. Both optional keys are preserved through
upsample → mirror → concat, unmodified.

The script does NOT need Isaac Sim. It runs on pure numpy.

Usage
-----
    python scripts/AMP/augment_motion_npz.py \
        --input  motion_datasets/penguin/g1_qie_motion.npz \
        --output motion_datasets/penguin/g1_qie_motion_aug.npz \
        --upsample 3 \
        --mirror

    # Interp-only (no mirror), upsample 2x:
    python scripts/AMP/augment_motion_npz.py -i IN.npz -o OUT.npz -k 2 --no-mirror

    # Mirror-only (upsample=1), keeps original frames intact:
    python scripts/AMP/augment_motion_npz.py -i IN.npz -o OUT.npz -k 1 --mirror

    # Use a custom body name list (one name per line, in NPZ body order):
    python scripts/AMP/augment_motion_npz.py -i IN.npz -o OUT.npz \
        --body-names-file my_body_order.txt

    # Print the baked-in G1 23DoF body/joint mirror plan and exit:
    python scripts/AMP/augment_motion_npz.py --print-plan

Defaults are tailored to the G1 23-DoF / 24-body layout used by this
repo (the joint order is baked into each env cfg under
``tasks/amp/robots/g1/29dof/*.py`` as ``G1_23DOF_JOINT_NAMES``).
If your NPZ uses a different body order, pass ``--body-names-file``.
"""

from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path


# =========================================================================
# G1 23-DoF mirror plan
# =========================================================================
#
# Joint order MUST match the NPZ. Kept in sync with the per-env-cfg
# ``G1_23DOF_JOINT_NAMES`` copies under tasks/amp/robots/g1/29dof/.
G1_23DOF_JOINT_NAMES: tuple[str, ...] = (
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

# Per-joint sign flip under left↔right mirror. Rule of thumb for this
# URDF (verified from `g1_23dof_rev_1_0.urdf`): both left and right
# joints share the SAME axis vector in their local frames, so pitch
# joints (axis=Y) are sign-preserved on mirror, but roll (axis=X) and
# yaw (axis=Z) joints must be negated (physically they describe a
# lateral / twisting motion that reverses sign under a y-mirror).
# ``waist_yaw`` is the only unpaired joint — it flips sign in place.
_G1_23DOF_SIGN_FLIP: dict[str, bool] = {
    "left_hip_pitch_joint": False,
    "left_hip_roll_joint": True,
    "left_hip_yaw_joint": True,
    "left_knee_joint": False,
    "left_ankle_pitch_joint": False,
    "left_ankle_roll_joint": True,
    "right_hip_pitch_joint": False,
    "right_hip_roll_joint": True,
    "right_hip_yaw_joint": True,
    "right_knee_joint": False,
    "right_ankle_pitch_joint": False,
    "right_ankle_roll_joint": True,
    "waist_yaw_joint": True,
    "left_shoulder_pitch_joint": False,
    "left_shoulder_roll_joint": True,
    "left_shoulder_yaw_joint": True,
    "left_elbow_joint": False,
    "left_wrist_roll_joint": True,
    "right_shoulder_pitch_joint": False,
    "right_shoulder_roll_joint": True,
    "right_shoulder_yaw_joint": True,
    "right_elbow_joint": False,
    "right_wrist_roll_joint": True,
}

# Default body name list for G1 23-DoF (24 articulated bodies, Isaac Lab
# breadth-first USD traversal from the ``pelvis`` root). Override via
# ``--body-names-file`` if the NPZ was generated with a different order.
G1_23DOF_BODY_NAMES_DEFAULT: tuple[str, ...] = (
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "torso_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_roll_rubber_hand",
    "right_wrist_roll_rubber_hand",
)


# =========================================================================
# Quaternion helpers (wxyz convention)
# =========================================================================


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize (..., 4) quaternions, safe for zero-norm inputs."""
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return q / n


def _slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Batched SLERP of wxyz quaternions.

    Parameters
    ----------
    q0, q1 : (B, 4)  start / end quaternions (wxyz).
    t      : (B,)    interpolation parameter in [0, 1]; 0 → q0, 1 → q1.

    Returns
    -------
    (B, 4) interpolated quaternions, normalized.
    """
    q0 = _quat_normalize(q0)
    q1 = _quat_normalize(q1)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)

    # Take the short path — if dot < 0, flip q1.
    flip = dot < 0.0
    q1 = np.where(flip, -q1, q1)
    dot = np.where(flip, -dot, dot)

    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)                      # (B, 1)
    sin_theta = np.sin(theta)

    # Near-parallel quats: fall back to linear interp (SLERP is ill-posed
    # when sin_theta → 0). Threshold matches common robotics code.
    close = sin_theta.squeeze(-1) < 1.0e-6
    t_col = t.reshape(-1, 1)
    s0 = np.where(close[:, None], 1.0 - t_col, np.sin((1.0 - t_col) * theta) / np.where(sin_theta < 1e-12, 1.0, sin_theta))
    s1 = np.where(close[:, None], t_col,       np.sin(t_col * theta)         / np.where(sin_theta < 1e-12, 1.0, sin_theta))
    return _quat_normalize(s0 * q0 + s1 * q1)


# =========================================================================
# Step 1 — temporal upsampling
# =========================================================================


def upsample_motion(raw: dict[str, np.ndarray], k: int) -> dict[str, np.ndarray]:
    """Return a new raw dict with ``(k * T - (k - 1))`` frames @ ``k * fps``.

    Uses linear interpolation for positions / velocities / joint values and
    SLERP for body quaternions. The interp keeps the first + last frame of
    the original intact (so the first mirrored frame lines up cleanly with
    the last original frame under the wrap-around transition convention).

    If ``k == 1`` this is the identity (aside from a defensive copy).
    """
    if k == 1:
        return {key: np.array(val, copy=True) for key, val in raw.items()}
    if k < 1:
        raise ValueError(f"--upsample must be >= 1, got {k}")

    fps_src = float(np.asarray(raw["fps"]).reshape(-1)[0])
    fps_dst = fps_src * k

    joint_pos_src = raw["joint_pos"]
    T = int(joint_pos_src.shape[0])
    T_new = (T - 1) * k + 1  # first + last frame preserved

    # Build the interp index grid in "source frame units" over T_new samples.
    # new_idx / k maps 0 → 0, T_new-1 → T-1.
    idx_new = np.arange(T_new, dtype=np.float64)
    t_src = idx_new / k                           # (T_new,) in [0, T-1]
    i0 = np.floor(t_src).astype(np.int64)
    i1 = np.clip(i0 + 1, 0, T - 1)
    alpha = (t_src - i0.astype(np.float64))[:, None]  # (T_new, 1)

    def _linear_nd(arr: np.ndarray) -> np.ndarray:
        # arr: (T, ...) → (T_new, ...)
        a0 = arr[i0]
        a1 = arr[i1]
        # Broadcast alpha to match arr's trailing dims.
        shape_alpha = [T_new] + [1] * (arr.ndim - 1)
        a = alpha.reshape(shape_alpha)
        return ((1.0 - a) * a0 + a * a1).astype(arr.dtype)

    def _slerp_nd(arr: np.ndarray) -> np.ndarray:
        # arr: (T, N, 4); do SLERP per-body, flattened over the batch axis.
        N = arr.shape[1]
        q0 = arr[i0].reshape(-1, 4).astype(np.float64)
        q1 = arr[i1].reshape(-1, 4).astype(np.float64)
        t = np.repeat(alpha.reshape(-1), N)
        out = _slerp(q0, q1, t).reshape(T_new, N, 4)
        return out.astype(arr.dtype)

    out: dict[str, np.ndarray] = {}
    out["fps"] = np.array([fps_dst], dtype=np.asarray(raw["fps"]).dtype)
    out["joint_pos"] = _linear_nd(raw["joint_pos"])
    out["joint_vel"] = _linear_nd(raw["joint_vel"])
    out["body_pos_w"] = _linear_nd(raw["body_pos_w"])
    out["body_quat_w"] = _slerp_nd(raw["body_quat_w"])
    out["body_lin_vel_w"] = _linear_nd(raw["body_lin_vel_w"])
    out["body_ang_vel_w"] = _linear_nd(raw["body_ang_vel_w"])
    # Pass through any extra keys untouched (won't be consumed by the loader).
    for k_extra in raw.keys():
        if k_extra not in out:
            out[k_extra] = np.array(raw[k_extra], copy=True)
    return out


# =========================================================================
# Step 2 — left / right mirror
# =========================================================================


def _auto_pair_lr(names: list[str]) -> tuple[list[tuple[int, int]], list[int]]:
    """Return ``(pairs, unpaired)`` given a mixed left_/right_ name list.

    Pair indices are returned as ``(left_idx, right_idx)``; unpaired names
    (e.g. ``pelvis``, ``torso_link``, ``waist_yaw_joint``) are returned as
    their own indices and get a self-map (no swap, value-level sign flip
    handled elsewhere if applicable).
    """
    name_to_idx = {n: i for i, n in enumerate(names)}
    pairs: list[tuple[int, int]] = []
    seen: set[int] = set()
    unpaired: list[int] = []
    for i, n in enumerate(names):
        if i in seen:
            continue
        if n.startswith("left_"):
            mate = "right_" + n[len("left_"):]
            if mate in name_to_idx:
                j = name_to_idx[mate]
                pairs.append((i, j))
                seen.add(i)
                seen.add(j)
                continue
        elif n.startswith("right_"):
            mate = "left_" + n[len("right_"):]
            if mate in name_to_idx:
                # Already handled when the left side was visited.
                continue
        unpaired.append(i)
        seen.add(i)
    return pairs, unpaired


def mirror_motion(
    raw: dict[str, np.ndarray],
    joint_names: list[str],
    body_names: list[str],
    joint_sign_flip: dict[str, bool],
) -> dict[str, np.ndarray]:
    """Return a left↔right mirror of ``raw``.

    World-frame transformations (mirror plane = xz, normal = y):

    - positions:      (x,  y,  z)        → (x, -y,  z)
    - linear vels:    (vx, vy, vz)       → (vx, -vy, vz)
    - angular vels:   (ωx, ωy, ωz)       → (-ωx, ωy, -ωz)   (pseudovector)
    - quaternions:    (w,  x,  y,  z)    → (w, -x,  y, -z)

    Joint-space transformations:

    - swap left_* ↔ right_* joint columns,
    - negate the columns flagged in ``joint_sign_flip``,
    - also negate unpaired joints flagged in ``joint_sign_flip``
      (e.g. ``waist_yaw_joint``).

    Body-space transformations:

    - swap left_* ↔ right_* body rows in every (..., N, ...) array.
    """
    if raw["joint_pos"].shape[1] != len(joint_names):
        raise ValueError(
            f"joint_names has length {len(joint_names)} but joint_pos has "
            f"{raw['joint_pos'].shape[1]} columns."
        )
    if raw["body_pos_w"].shape[1] != len(body_names):
        raise ValueError(
            f"body_names has length {len(body_names)} but body_pos_w has "
            f"{raw['body_pos_w'].shape[1]} rows."
        )

    out = {k: np.array(v, copy=True) for k, v in raw.items()}

    # -- Body-level mirror (y-flip + pseudo-vector flip + quat wxyz mirror).
    out["body_pos_w"][:, :, 1] *= -1.0
    out["body_lin_vel_w"][:, :, 1] *= -1.0
    out["body_ang_vel_w"][:, :, 0] *= -1.0
    out["body_ang_vel_w"][:, :, 2] *= -1.0
    out["body_quat_w"][:, :, 1] *= -1.0   # x component
    out["body_quat_w"][:, :, 3] *= -1.0   # z component

    # -- Swap left/right body rows.
    body_pairs, body_unpaired = _auto_pair_lr(body_names)
    for li, ri in body_pairs:
        out["body_pos_w"][:, [li, ri]] = out["body_pos_w"][:, [ri, li]]
        out["body_quat_w"][:, [li, ri]] = out["body_quat_w"][:, [ri, li]]
        out["body_lin_vel_w"][:, [li, ri]] = out["body_lin_vel_w"][:, [ri, li]]
        out["body_ang_vel_w"][:, [li, ri]] = out["body_ang_vel_w"][:, [ri, li]]
    del body_unpaired  # unpaired bodies (pelvis, torso_link) need no swap — already y-mirrored above.

    # -- Joint-level swap + per-joint sign flip.
    joint_pairs, joint_unpaired = _auto_pair_lr(joint_names)
    # Sign flips are applied *after* the swap so the flip acts on the
    # already-swapped value (swap then flip == flip then swap when both
    # sides share the flag, which they always do for L/R symmetric joints).
    for li, ri in joint_pairs:
        out["joint_pos"][:, [li, ri]] = out["joint_pos"][:, [ri, li]]
        out["joint_vel"][:, [li, ri]] = out["joint_vel"][:, [ri, li]]
    for i, name in enumerate(joint_names):
        if joint_sign_flip.get(name, False):
            out["joint_pos"][:, i] *= -1.0
            out["joint_vel"][:, i] *= -1.0
    del joint_unpaired  # flips on unpaired joints (e.g. waist_yaw) handled by the loop above.

    return out


# =========================================================================
# Step 3 — concat and save
# =========================================================================


def concat_motions(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Concatenate two NPZ dicts along the time axis. ``fps`` must match."""
    fps_a = float(np.asarray(a["fps"]).reshape(-1)[0])
    fps_b = float(np.asarray(b["fps"]).reshape(-1)[0])
    if abs(fps_a - fps_b) > 1e-6:
        raise ValueError(f"fps mismatch: a={fps_a}, b={fps_b}")
    out: dict[str, np.ndarray] = {"fps": a["fps"]}
    for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                "body_lin_vel_w", "body_ang_vel_w"):
        out[key] = np.concatenate([a[key], b[key]], axis=0)
    # Preserve any extra keys from ``a`` unchanged (assumed scalar / metadata).
    for key, val in a.items():
        if key not in out:
            out[key] = val
    return out


# =========================================================================
# CLI
# =========================================================================


def _print_plan() -> None:
    print("G1 23-DoF mirror plan")
    print("=" * 70)
    print(f"Joints ({len(G1_23DOF_JOINT_NAMES)}):")
    pairs, unpaired = _auto_pair_lr(list(G1_23DOF_JOINT_NAMES))
    for li, ri in pairs:
        lname = G1_23DOF_JOINT_NAMES[li]
        rname = G1_23DOF_JOINT_NAMES[ri]
        flip_l = _G1_23DOF_SIGN_FLIP[lname]
        flip_r = _G1_23DOF_SIGN_FLIP[rname]
        tag = "swap + sign-flip" if flip_l and flip_r else ("swap only" if not (flip_l or flip_r) else "swap + asymmetric flip (!)")
        print(f"  [{li:2d}]↔[{ri:2d}]  {lname:28s} ↔ {rname:28s}  ({tag})")
    for i in unpaired:
        name = G1_23DOF_JOINT_NAMES[i]
        flip = _G1_23DOF_SIGN_FLIP.get(name, False)
        tag = "sign-flip in place" if flip else "no change"
        print(f"  [{i:2d}]    {name:28s}  ({tag})")
    print()
    print(f"Bodies (default, {len(G1_23DOF_BODY_NAMES_DEFAULT)}):")
    body_pairs, body_unpaired = _auto_pair_lr(list(G1_23DOF_BODY_NAMES_DEFAULT))
    for li, ri in body_pairs:
        print(f"  [{li:2d}]↔[{ri:2d}]  {G1_23DOF_BODY_NAMES_DEFAULT[li]:32s} ↔ {G1_23DOF_BODY_NAMES_DEFAULT[ri]}")
    for i in body_unpaired:
        print(f"  [{i:2d}]    {G1_23DOF_BODY_NAMES_DEFAULT[i]}  (unpaired, y-mirror only)")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    required = ("fps", "joint_pos", "joint_vel", "body_pos_w",
                "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
    for k in required:
        if k not in data.files:
            raise KeyError(f"{path}: missing required key {k!r}")
    return {k: np.asarray(data[k]) for k in data.files}


def _save_npz(path: Path, raw: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **raw)


def _load_body_names(path: Path | None) -> list[str]:
    if path is None:
        return list(G1_23DOF_BODY_NAMES_DEFAULT)
    lines = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip() and not ln.strip().startswith("#")]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-i", "--input", type=Path, help="Input NPZ path.")
    parser.add_argument("-o", "--output", type=Path, help="Output NPZ path.")
    parser.add_argument("-k", "--upsample", type=int, default=3,
                        help="Temporal upsample factor (>=1). Default: 3.")
    mirror_grp = parser.add_mutually_exclusive_group()
    mirror_grp.add_argument("--mirror", dest="mirror", action="store_true",
                            help="Apply left-right mirror (default).")
    mirror_grp.add_argument("--no-mirror", dest="mirror", action="store_false",
                            help="Skip the mirror step; just upsample.")
    parser.set_defaults(mirror=True)
    parser.add_argument("--body-names-file", type=Path, default=None,
                        help="Text file with one body name per line, in NPZ body order. "
                             "Defaults to G1_23DOF_BODY_NAMES_DEFAULT.")
    parser.add_argument("--keep-original", dest="keep_original", action="store_true",
                        help="Concat the interp result before mirroring. Default: on.")
    parser.add_argument("--only-mirror", dest="keep_original", action="store_false",
                        help="Emit only the mirrored half (no concat). Halves the output.")
    parser.set_defaults(keep_original=True)
    parser.add_argument("--print-plan", action="store_true",
                        help="Print the baked-in G1 23-DoF mirror plan and exit.")
    args = parser.parse_args()

    if args.print_plan:
        _print_plan()
        return

    if args.input is None or args.output is None:
        parser.error("--input and --output are required (unless --print-plan is set).")

    src = _load_npz(args.input)
    T_src = int(src["joint_pos"].shape[0])
    fps_src = float(np.asarray(src["fps"]).reshape(-1)[0])
    has_joint_names = "joint_names" in src
    has_body_names = "body_names" in src
    print(
        f"[load]  {args.input}  T={T_src}  fps={fps_src}  "
        f"J={src['joint_pos'].shape[1]}  N={src['body_pos_w'].shape[1]}  "
        f"joint_names={'yes' if has_joint_names else 'no'}  "
        f"body_names={'yes' if has_body_names else 'no'}"
    )

    # --- Step 1: interpolation (always run; k=1 is identity copy).
    interp = upsample_motion(src, int(args.upsample))
    T_interp = int(interp["joint_pos"].shape[0])
    fps_interp = float(np.asarray(interp["fps"]).reshape(-1)[0])
    print(f"[interp] k={args.upsample}  T={T_interp}  fps={fps_interp}")

    # --- Step 2: mirror (on the interpolated clip, not the original).
    if args.mirror:
        # Prefer the npz's own name lists when present — they are
        # authoritative for the clip's column/row ordering. Fall back to
        # the baked G1-23DoF defaults otherwise (maintains old behavior
        # for legacy npz files). ``--body-names-file`` always wins.
        if args.body_names_file is not None:
            body_names = _load_body_names(args.body_names_file)
        elif "body_names" in interp:
            body_names = [str(x) for x in np.asarray(interp["body_names"]).reshape(-1).tolist()]
        else:
            body_names = list(G1_23DOF_BODY_NAMES_DEFAULT)

        if "joint_names" in interp:
            joint_names = [str(x) for x in np.asarray(interp["joint_names"]).reshape(-1).tolist()]
            # Cross-check that every mirror-plan joint is present — the
            # sign-flip table is keyed by name, so unknown names would
            # silently default to ``False`` (no flip).
            unknown = [n for n in joint_names if n not in _G1_23DOF_SIGN_FLIP]
            if unknown:
                raise ValueError(
                    f"npz joint_names include entries not in the baked G1-23DoF "
                    f"mirror plan: {unknown}. Extend _G1_23DOF_SIGN_FLIP or "
                    "run without --mirror."
                )
        else:
            joint_names = list(G1_23DOF_JOINT_NAMES)
        if interp["body_pos_w"].shape[1] != len(body_names):
            raise ValueError(
                f"Body-name list length {len(body_names)} != NPZ body count "
                f"{interp['body_pos_w'].shape[1]}. Pass --body-names-file with the "
                f"correct order (one name per line)."
            )
        if interp["joint_pos"].shape[1] != len(joint_names):
            raise ValueError(
                f"Joint-name list length {len(joint_names)} != NPZ joint count "
                f"{interp['joint_pos'].shape[1]}. This script is baked to G1 23DoF; "
                f"edit G1_23DOF_JOINT_NAMES if using a different robot."
            )
        mirrored = mirror_motion(
            interp,
            joint_names=joint_names,
            body_names=body_names,
            joint_sign_flip=_G1_23DOF_SIGN_FLIP,
        )
        T_mir = int(mirrored["joint_pos"].shape[0])
        print(f"[mirror] T={T_mir}  (L/R body pairs="
              f"{len(_auto_pair_lr(body_names)[0])}, joint pairs="
              f"{len(_auto_pair_lr(joint_names)[0])})")
    else:
        mirrored = None

    # --- Step 3: concat.
    if mirrored is None:
        final = interp
    elif args.keep_original:
        final = concat_motions(interp, mirrored)
    else:
        final = mirrored

    T_out = int(final["joint_pos"].shape[0])
    fps_out = float(np.asarray(final["fps"]).reshape(-1)[0])
    print(f"[save]   {args.output}  T={T_out}  fps={fps_out}  "
          f"(transitions @ wrap_around={T_out})")
    _save_npz(args.output, final)


if __name__ == "__main__":
    main()
