# Copyright (c) 2025, Unitree Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""AMP motion dataset loader with time-aligned transition sampling.

Expert features are built once from the native-fps ``.npz`` clips using the
single unified feature constructor from
:mod:`unitree_rl_lab.rsl_rl_amp.features.amp_features`, then **resampled at
class-init onto the ``env_step_dt`` grid** using linear interpolation.
``sample()`` thereafter does pure index lookup on the pre-resampled buffer —
no interpolation or floating-point blend work in the hot path.

This is the invariant that makes the AMP discriminator meaningful: it
compares expert transitions ``(amp_obs_E(t), amp_obs_E(t + env_step_dt))``
against policy transitions ``(amp_obs_pi(t), amp_obs_pi(t + env_step_dt))``.
Both sides use the same time delta — the control-step dt
(``sim.dt * decimation``) — rather than the expert clip's raw frame cadence.

Expected npz schema (as produced by ``scripts/mimic/csv_to_npz.py`` /
``scripts/AMP/csv_to_npz_final.py``):

- ``fps``: shape ``(1,)`` int.
- ``joint_pos``: ``(T, J)`` radians.
- ``joint_vel``: ``(T, J)`` rad/s.
- ``body_pos_w``:     ``(T, N, 3)`` world-frame body positions.
- ``body_quat_w``:    ``(T, N, 4)`` world-frame body quaternions, wxyz.
- ``body_lin_vel_w``: ``(T, N, 3)``.
- ``body_ang_vel_w``: ``(T, N, 3)``.
- ``joint_names`` (optional, length ``J``): name list matching
  ``joint_pos`` column order. Written by ``csv_to_npz_final.py`` from
  ``robot.data.joint_names`` so the npz is self-describing.
- ``body_names`` (optional, length ``N``): name list matching
  ``body_pos_w`` row order (``robot.data.body_names``).

If the npz carries ``joint_names`` / ``body_names``, the dataset uses those
as the authoritative column ordering and the caller-supplied
``amp_joint_names`` / ``amp_body_names`` are optional. When the caller
provides them as well, they are cross-checked against the npz and a
mismatch raises at init — so a stale caller-side name list can never
silently drift out of sync with the serialized clip.

Resampling cadence is auto-derived from ``npz['fps']`` and the caller-
supplied ``env_step_dt``; nothing is hard-coded. If training is restarted
with a different ``env_step_dt``, a fresh :class:`MotionDataset` is
constructed, so the resample is recomputed (once per training run).
"""

from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from typing import Iterable

from ..features.amp_features import (
    AmpObsSpec,
    AmpObsState,
    build_amp_frame_from_state,
    resolve_indices,
    resolve_spec_indices,
)


# ---------------------------------------------------------------------------
# Small utility: build the AmpObsState from an npz dict.
# ---------------------------------------------------------------------------

def _npz_frames_to_state(
    raw: dict[str, np.ndarray],
    *,
    pelvis_idx: int,
    foot_idx: Sequence[int],
    hand_idx: Sequence[int],
    joint_idx: Sequence[int],
    default_joint_pos_np: np.ndarray,
    device: str | torch.device,
) -> AmpObsState:
    """Return an :class:`AmpObsState` covering all ``T`` frames of one npz.

    ``joint_idx`` permutes the npz joint columns (order =
    ``amp_joint_names``, typically the Isaac articulation's native joint
    order) into the spec's joint order before feature construction. Body
    arrays are indexed by name-resolved ``pelvis_idx`` / ``foot_idx`` /
    ``hand_idx`` into ``amp_body_names``.
    """
    dtype = torch.float32
    body_pos_w = torch.from_numpy(raw["body_pos_w"]).to(device, dtype=dtype)
    body_quat_w = torch.from_numpy(raw["body_quat_w"]).to(device, dtype=dtype)
    body_lin_vel_w = torch.from_numpy(raw["body_lin_vel_w"]).to(device, dtype=dtype)
    body_ang_vel_w = torch.from_numpy(raw["body_ang_vel_w"]).to(device, dtype=dtype)

    joint_pos_raw = torch.from_numpy(np.asarray(raw["joint_pos"], dtype=np.float32)).to(device)
    joint_vel_raw = torch.from_numpy(np.asarray(raw["joint_vel"], dtype=np.float32)).to(device)
    j_idx = torch.as_tensor(list(joint_idx), dtype=torch.long, device=device)
    joint_pos = joint_pos_raw.index_select(1, j_idx)
    joint_vel = joint_vel_raw.index_select(1, j_idx)

    root_pos_w = body_pos_w[:, pelvis_idx, :]
    root_quat_w = body_quat_w[:, pelvis_idx, :]
    root_lin_vel_w = body_lin_vel_w[:, pelvis_idx, :]
    root_ang_vel_w = body_ang_vel_w[:, pelvis_idx, :]
    foot_pos_w = body_pos_w[:, list(foot_idx), :]
    foot_lin_vel_w = body_lin_vel_w[:, list(foot_idx), :]
    hand_pos_w = body_pos_w[:, list(hand_idx), :]
    hand_lin_vel_w = body_lin_vel_w[:, list(hand_idx), :]

    default_jp = torch.from_numpy(default_joint_pos_np.astype(np.float32)).to(device)

    return AmpObsState(
        root_pos_w=root_pos_w,
        root_quat_w=root_quat_w,
        root_lin_vel_w=root_lin_vel_w,
        root_ang_vel_w=root_ang_vel_w,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        default_joint_pos=default_jp,
        foot_pos_w=foot_pos_w,
        foot_lin_vel_w=foot_lin_vel_w,
        hand_pos_w=hand_pos_w,
        hand_lin_vel_w=hand_lin_vel_w,
    )


# ---------------------------------------------------------------------------
# Motion reset sampling payload — raw per-frame state (unchanged by the
# transition-alignment rework).
# ---------------------------------------------------------------------------

class MotionResetPayload:
    """Per-motion frame samples used to initialize the env at reset.

    Exposes the raw per-frame states needed to drive
    ``articulation.write_root_state_to_sim`` /
    ``articulation.write_joint_state_to_sim``. Joint order is the user-
    provided ``amp_joint_names`` order, body order matches ``amp_body_names``.
    """

    __slots__ = (
        "joint_pos",
        "joint_vel",
        "root_pos",
        "root_quat",
        "root_lin_vel",
        "root_ang_vel",
        "num_frames",
        "fps",
    )

    def __init__(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
        fps: float,
    ) -> None:
        self.joint_pos = joint_pos
        self.joint_vel = joint_vel
        self.root_pos = root_pos
        self.root_quat = root_quat
        self.root_lin_vel = root_lin_vel
        self.root_ang_vel = root_ang_vel
        self.num_frames = int(joint_pos.shape[0])
        self.fps = float(fps)


# ---------------------------------------------------------------------------
# Name-list resolution (npz joint_names / body_names vs caller-supplied)
# ---------------------------------------------------------------------------

def _resolve_name_list(
    *,
    caller_names: list[str] | None,
    raws: list[dict],
    paths: list[str],
    key: str,
    kind: str,
) -> list[str]:
    """Resolve the authoritative column-order name list for an npz field.

    Precedence:

    1. If any loaded npz carries ``key`` (e.g. ``joint_names``), it is the
       authoritative list. Every other npz that carries ``key`` must agree.
    2. If no npz carries ``key``, fall back to ``caller_names``.
    3. If both a npz and ``caller_names`` are provided, they must agree —
       a mismatch raises so a stale caller-side list never silently drifts
       out of sync with the serialized clip.
    4. If neither provides names, raise a clear error pointing the user to
       regenerate via ``csv_to_npz_final.py``.
    """
    npz_names: list[str] | None = None
    npz_source: str | None = None
    for p, raw in zip(paths, raws):
        if key in raw:
            nm = [str(x) for x in list(raw[key])]
            if npz_names is None:
                npz_names = nm
                npz_source = p
            elif nm != npz_names:
                raise ValueError(
                    f"Inconsistent {key!r} across motion files. "
                    f"{npz_source} → {npz_names}, {p} → {nm}."
                )

    if npz_names is not None and caller_names is not None:
        caller_norm = [str(x) for x in caller_names]
        if caller_norm != npz_names:
            raise ValueError(
                f"Caller-supplied amp_{kind}_names disagree with {key!r} in "
                f"{npz_source}. "
                f"caller={caller_norm}, npz={npz_names}. "
                "Remove the caller-side list (the npz is self-describing) or "
                "regenerate the npz with csv_to_npz_final.py."
            )
        return list(npz_names)
    if npz_names is not None:
        return list(npz_names)
    if caller_names is not None:
        return [str(x) for x in caller_names]
    raise ValueError(
        f"No {kind} name list available: npz files carry no {key!r} "
        "(regenerate with scripts/AMP/csv_to_npz_final.py so the npz is "
        f"self-describing) and the caller did not pass amp_{kind}_names."
    )


# ---------------------------------------------------------------------------
# Resampling helper: linear interp phi_raw (native fps) -> phi_rs (env_step_dt)
# ---------------------------------------------------------------------------

def _resample_phi_linear(
    phi_raw: torch.Tensor,
    frame_dt_src: float,
    env_step_dt: float,
    T_rs: int,
    wrap_around: bool,
) -> torch.Tensor:
    """Linear-interpolate a ``(T_src, D)`` phi buffer onto a regular
    ``env_step_dt`` grid of length ``T_rs``.

    Done once per clip at dataset init. Vectorized — one pass with two
    gathers + a blend. Grid points for ``j ∈ [0, T_rs-1]`` are
    ``τ_j = j * env_step_dt``. For ``wrap_around=True`` the output is a
    phase-aligned period (``phi_rs[0] == phi_raw[0]``); for
    ``wrap_around=False`` the output's last point is the last grid point
    strictly ``≤ (T_src - 1) * frame_dt_src``.
    """
    T_src, D = int(phi_raw.shape[0]), int(phi_raw.shape[1])
    device = phi_raw.device

    j = torch.arange(T_rs, device=device, dtype=torch.float64)
    tau = j * float(env_step_dt)
    x = tau / float(frame_dt_src)  # float64 for accuracy, cast blend down later

    idx0 = torch.floor(x).to(torch.long)
    blend = (x - torch.floor(x)).to(phi_raw.dtype).unsqueeze(-1)  # (T_rs, 1)
    if wrap_around:
        idx0 = idx0 % T_src
        idx1 = (idx0 + 1) % T_src
    else:
        # Last dst sample is guaranteed in [0, T_src-1] by the T_rs formula,
        # but clamp idx1 defensively to avoid an out-of-range when τ exactly
        # hits the last frame.
        idx1 = torch.clamp(idx0 + 1, max=T_src - 1)
        idx0 = torch.clamp(idx0, max=T_src - 1)

    f0 = phi_raw.index_select(0, idx0)  # (T_rs, D)
    f1 = phi_raw.index_select(0, idx1)
    return (1.0 - blend) * f0 + blend * f1


# ---------------------------------------------------------------------------
# MotionDataset
# ---------------------------------------------------------------------------

class MotionDataset:
    """Expert motion dataset for AMP discriminator training.

    Time-aligned transition sampler. Each clip is loaded at its native fps,
    passed through :func:`build_amp_frame_from_state` to get a per-frame
    phi buffer, and then **linearly resampled at init** onto the policy
    control-step grid (``env_step_dt``). At sample time the dataset picks
    a clip by ``motion_weights``, picks a uniform now-index on the
    resampled grid, and gathers ``(amp_obs_t, amp_obs_{t+env_step_dt})``
    as the ``K`` most recent slots of two adjacent windows on that grid.

    The discriminator thereby sees expert transitions whose time delta
    equals the policy control-step dt (``sim.dt * decimation``), matching
    the policy rollout's ``(amp_obs_before_step, amp_obs_after_step)``
    pairs.

    Parameters
    ----------
    motion_files:
        Path or list of paths to ``.npz`` files.
    spec:
        Shared AMP feature layout. The dataset validates array shapes
        against it and only uses the spec's ``include_*`` flags / names.
    amp_joint_names:
        Optional. Full ordered joint name list that matches the npz joint
        arrays. If ``None``, the names are read from the npz
        (``joint_names`` field, written by ``csv_to_npz_final.py``). If
        both the caller and the npz provide names, they are cross-checked
        and any mismatch raises at init.
    amp_body_names:
        Optional. Full ordered body name list that matches the npz body
        arrays. Same precedence rules as ``amp_joint_names``.
    default_joint_pos:
        ``(J,)`` reference pose used to compute ``joint_pos_rel``. Must be
        provided by the caller (typically pulled from the env's articulation
        init state) so the dataset features match the runtime features.
    env_step_dt:
        Policy control-step dt in seconds (``sim.dt * decimation``). The
        resampling grid cadence. Required. ``env.step_dt`` in IsaacLab.
    motion_weights:
        Per-clip sampling weights used by :meth:`sample`. ``None`` =>
        uniform over clips. Length must equal ``len(motion_files)``.
    wrap_around:
        If True, the resampled buffer is treated as periodic (modular
        gather), which is the right default for cyclic gait clips. If
        False, the valid now-index range is clipped so both the oldest
        slot of the ``now`` window and the newest slot of the ``next``
        window stay strictly in-range.
    device:
        Target torch device for the stored buffers.

    Extension hooks
    ---------------
    ``mirror_augmentation`` / ``time_scaling_range`` are currently no-ops;
    they exist to freeze the call signature and will be implemented in
    Phase-2.

    Notes
    -----
    For ``wrap_around=True`` with a clip whose total duration
    ``T_src * (1/fps)`` is not an exact multiple of ``env_step_dt``, the
    resampled grid's period is ``round(...)`` so there is a sub-
    ``env_step_dt`` phase offset per cycle. This is negligible for
    periodic gait clips and is not treated as an error.
    """

    def __init__(
        self,
        motion_files: str | Sequence[str],
        *,
        spec: AmpObsSpec,
        amp_joint_names: Sequence[str] | None = None,
        amp_body_names: Sequence[str] | None = None,
        default_joint_pos: Iterable[float] | np.ndarray,
        env_step_dt: float,
        motion_weights: Sequence[float] | None = None,
        wrap_around: bool = True,
        mirror_augmentation: bool = False,  # reserved, Phase-2
        time_scaling_range: tuple[float, float] | None = None,  # reserved, Phase-2
        device: str | torch.device = "cpu",
    ) -> None:
        del mirror_augmentation, time_scaling_range  # reserved hooks

        if isinstance(motion_files, str):
            motion_files = [motion_files]
        if len(motion_files) == 0:
            raise ValueError("MotionDataset requires at least one motion file.")

        if not math.isfinite(float(env_step_dt)) or float(env_step_dt) <= 0.0:
            raise ValueError(
                f"env_step_dt must be a positive finite float, got {env_step_dt!r}."
            )
        env_step_dt = float(env_step_dt)

        self.device = device
        self.spec = spec
        self.wrap_around = bool(wrap_around)
        self.env_step_dt = env_step_dt

        caller_joint_names = (
            list(amp_joint_names) if amp_joint_names is not None else None
        )
        caller_body_names = (
            list(amp_body_names) if amp_body_names is not None else None
        )

        # -- Eagerly load all clips once so we can harvest name lists and
        # validate ordering consistency across clips. Returned dicts are
        # passed into the per-clip build loop below (no double-load).
        raw_clips: list[dict[str, np.ndarray]] = []
        for path in motion_files:
            raw_clips.append(self._load_npz(path))

        # -- Authoritative name lists. Preference order:
        #    1. first npz that carries joint_names / body_names
        #    2. caller-supplied amp_joint_names / amp_body_names
        # All other clips must agree; when the caller also passes names, they
        # must match the npz's. This catches stale spec-vs-clip drift.
        amp_joint_names_list = _resolve_name_list(
            caller_names=caller_joint_names,
            raws=raw_clips,
            paths=list(motion_files),
            key="joint_names",
            kind="joint",
        )
        amp_body_names_list = _resolve_name_list(
            caller_names=caller_body_names,
            raws=raw_clips,
            paths=list(motion_files),
            key="body_names",
            kind="body",
        )

        # Re-expose the resolved lists as attributes so callers / loggers can
        # introspect the *actual* ordering used by the discriminator features.
        self.amp_joint_names: list[str] = amp_joint_names_list
        self.amp_body_names: list[str] = amp_body_names_list

        # -- Resolve the spec's joint order into the npz column order.
        # ``amp_joint_names_list`` describes the npz joint column order —
        # typically the Isaac articulation's native joint order, since the
        # npz producer (``scripts/AMP/csv_to_npz_final.py``) saves
        # ``robot.data.joint_pos`` unpermuted. We resolve per-spec-joint
        # indices into that order and permute the raw arrays to spec order
        # during :func:`_npz_frames_to_state`.
        self._joint_idx: list[int] = resolve_indices(
            spec.joint_names, amp_joint_names_list, "motion.joints.spec_in_amp"
        )

        # -- Resolve pelvis / feet / hands indices against the body name list.
        # (Body arrays are name-resolved, so body ordering can differ between
        # npz and spec — no permutation of the raw body tensors is needed.)
        idx_map = resolve_spec_indices(
            spec, amp_joint_names_list, amp_body_names_list
        )
        pelvis_idx = idx_map["pelvis"][0]
        foot_idx = idx_map["feet"]
        hand_idx = idx_map["hands"]

        default_jp_np = np.asarray(list(default_joint_pos), dtype=np.float32)
        if default_jp_np.shape != (len(spec.joint_names),):
            raise ValueError(
                f"default_joint_pos shape {default_jp_np.shape} must match "
                f"len(spec.joint_names)={len(spec.joint_names)} (spec order)."
            )

        # -- Validate motion_weights up-front.
        if motion_weights is None:
            weights_np = np.ones(len(motion_files), dtype=np.float64)
        else:
            weights_np = np.asarray(list(motion_weights), dtype=np.float64)
            if weights_np.shape != (len(motion_files),):
                raise ValueError(
                    f"motion_weights length {weights_np.shape[0]} != "
                    f"len(motion_files)={len(motion_files)}."
                )
            if np.any(weights_np < 0.0) or not np.isfinite(weights_np).all():
                raise ValueError(
                    "motion_weights must be non-negative and finite; "
                    f"got {weights_np.tolist()}."
                )
            if weights_np.sum() <= 0.0:
                raise ValueError(
                    f"motion_weights sum must be positive; got {weights_np.tolist()}."
                )
        weights_np = weights_np / weights_np.sum()

        # ---- Per-clip state (populated in the load loop below).
        self.motion_files: list[str] = list(motion_files)
        self.motion_fps: list[float] = []               # native fps per clip
        self.motion_lengths: list[int] = []             # native T_src per clip
        self.motion_reset_payloads: list[MotionResetPayload] = []
        self._phi_rs_clips: list[torch.Tensor] = []     # (T_rs, D) per clip, resampled
        self._clip_lengths_rs_list: list[int] = []      # T_rs per clip
        self._clip_idx_min_list: list[int] = []
        self._clip_idx_max_list: list[int] = []
        self._clip_duration_src: list[float] = []       # (T_src - 1) * frame_dt_src

        K = int(spec.stack_k)

        for path, raw, clip_w in zip(motion_files, raw_clips, weights_np.tolist()):
            del clip_w  # weights stacked later after normalization
            T_src = int(raw["joint_pos"].shape[0])
            fps_src = float(raw["fps"])
            if not math.isfinite(fps_src) or fps_src <= 0.0:
                raise ValueError(
                    f"{path}: invalid fps {fps_src}. Must be a positive finite number."
                )
            frame_dt_src = 1.0 / fps_src
            duration_src = (T_src - 1) * frame_dt_src

            # Shape checks against npz.
            if raw["joint_pos"].shape != (T_src, len(amp_joint_names_list)):
                raise ValueError(
                    f"{path}: joint_pos shape {raw['joint_pos'].shape} != "
                    f"(T, {len(amp_joint_names_list)})."
                )
            if raw["body_pos_w"].shape != (T_src, len(amp_body_names_list), 3):
                raise ValueError(
                    f"{path}: body_pos_w shape {raw['body_pos_w'].shape} != "
                    f"(T, {len(amp_body_names_list)}, 3). "
                    "Make sure amp_body_names matches the npz body order."
                )

            # Build state + phi_raw on native fps.
            state = _npz_frames_to_state(
                raw,
                pelvis_idx=pelvis_idx,
                foot_idx=foot_idx,
                hand_idx=hand_idx,
                joint_idx=self._joint_idx,
                default_joint_pos_np=default_jp_np,
                device=device,
            )
            phi_raw = build_amp_frame_from_state(state, spec)  # (T_src, D)

            # Auto-derive T_rs on the env_step_dt grid.
            if self.wrap_around:
                T_rs = max(2, int(round((T_src * frame_dt_src) / env_step_dt)))
            else:
                if duration_src < env_step_dt:
                    raise ValueError(
                        f"{path}: duration {duration_src:.6f}s < env_step_dt "
                        f"{env_step_dt:.6f}s (fps_src={fps_src}, T_src={T_src}). "
                        "Clip too short to build a single env_step_dt transition."
                    )
                T_rs = int(math.floor(duration_src / env_step_dt)) + 1

            required = max(2, K + 1)
            if T_rs < required:
                raise ValueError(
                    f"{path}: resampled length T_rs={T_rs} < {required} "
                    f"(fps_src={fps_src}, T_src={T_src}, env_step_dt={env_step_dt}, "
                    f"stack_k={K}, wrap_around={self.wrap_around}). "
                    "Clip is too short for the requested stack_k at this control rate."
                )

            phi_rs = _resample_phi_linear(
                phi_raw=phi_raw,
                frame_dt_src=frame_dt_src,
                env_step_dt=env_step_dt,
                T_rs=T_rs,
                wrap_around=self.wrap_around,
            ).contiguous()

            if self.wrap_around:
                idx_min = 0
                idx_max = T_rs - 1
            else:
                idx_min = K - 1
                idx_max = T_rs - 2
            if idx_min > idx_max:
                # Should be impossible given the T_rs >= K+1 guard above;
                # defensive check.
                raise RuntimeError(
                    f"{path}: invalid now-index range [{idx_min}, {idx_max}] "
                    f"after T_rs={T_rs}, K={K}."
                )

            self._phi_rs_clips.append(phi_rs)
            self._clip_lengths_rs_list.append(T_rs)
            self._clip_idx_min_list.append(idx_min)
            self._clip_idx_max_list.append(idx_max)
            self._clip_duration_src.append(duration_src)

            # Motion reset payload — raw per-frame state for init.
            self.motion_reset_payloads.append(
                MotionResetPayload(
                    joint_pos=state.joint_pos.detach().clone(),
                    joint_vel=state.joint_vel.detach().clone(),
                    root_pos=state.root_pos_w.detach().clone(),
                    root_quat=state.root_quat_w.detach().clone(),
                    root_lin_vel=state.root_lin_vel_w.detach().clone(),
                    root_ang_vel=state.root_ang_vel_w.detach().clone(),
                    fps=fps_src,
                )
            )
            self.motion_lengths.append(T_src)
            self.motion_fps.append(fps_src)

        # Pack per-clip metadata into device tensors for vectorized sampling.
        self._clip_weights = torch.tensor(
            weights_np, dtype=torch.float32, device=self.device
        )
        self._clip_lengths_rs = torch.tensor(
            self._clip_lengths_rs_list, dtype=torch.long, device=self.device
        )
        self._clip_idx_min = torch.tensor(
            self._clip_idx_min_list, dtype=torch.long, device=self.device
        )
        self._clip_idx_max = torch.tensor(
            self._clip_idx_max_list, dtype=torch.long, device=self.device
        )

        self.frame_dim = int(spec.frame_dim)
        self.amp_obs_dim = int(spec.amp_obs_dim)

    # ------------------------------------------------------------------
    # Loading helper
    # ------------------------------------------------------------------
    @staticmethod
    def _load_npz(path: str) -> dict[str, np.ndarray]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Motion file not found: {path}")
        data = np.load(path, allow_pickle=False)
        required = ("fps", "joint_pos", "joint_vel", "body_pos_w",
                    "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
        for k in required:
            if k not in data.files:
                raise KeyError(f"Motion file '{path}' missing required key '{k}'.")
        out: dict[str, np.ndarray] = {k: data[k] for k in data.files}
        out["fps"] = float(np.asarray(out["fps"]).reshape(-1)[0])
        # Optional self-describing name lists. When present we materialize
        # them as plain Python ``list[str]`` so the caller doesn't have to
        # juggle numpy ``<U..`` dtypes.
        for name_key in ("joint_names", "body_names"):
            if name_key in out:
                out[name_key] = [str(x) for x in np.asarray(out[name_key]).reshape(-1).tolist()]
        return out

    # ------------------------------------------------------------------
    # Sampling — pure index lookup on the pre-resampled buffers.
    # ------------------------------------------------------------------
    def sample(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ``num_samples`` time-aligned expert transitions.

        Returns ``(amp_obs_t, amp_obs_{t+env_step_dt})`` each of shape
        ``(num_samples, amp_obs_dim)``. Both windows are assembled on the
        ``env_step_dt`` grid; the "next" window is the "now" window
        shifted by one grid index, so adjacent slots align
        (``next[:, k*D:(k+1)*D] == now[:, (k+1)*D:(k+2)*D]`` for
        ``k ∈ [0, K-2]``).
        """
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}.")

        B = int(num_samples)
        K = int(self.spec.stack_k)
        D = self.frame_dim

        clip_ids = torch.multinomial(self._clip_weights, B, replacement=True)
        lo = self._clip_idx_min[clip_ids]
        hi = self._clip_idx_max[clip_ids]
        # Uniform integer in [lo, hi] (inclusive). Using rand + long cast
        # avoids a per-clip loop over randint.
        u = torch.rand(B, device=self.device)
        span = (hi - lo + 1).to(torch.float32)
        idx_now = lo + (u * span).to(torch.long)
        idx_now = torch.minimum(idx_now, hi)  # guard the u==1.0 corner

        T_rs = self._clip_lengths_rs[clip_ids]  # (B,)

        now = torch.empty(B, K * D, device=self.device)
        nxt = torch.empty(B, K * D, device=self.device)

        unique_clips = torch.unique(clip_ids).tolist()
        for k in range(K):
            lag = K - 1 - k  # in units of env_step_dt; newest slot => lag=0
            off_now = idx_now - lag
            off_next = off_now + 1
            if self.wrap_around:
                off_now = off_now % T_rs
                off_next = off_next % T_rs
            # off_now / off_next are already in [0, T_rs-1] by construction
            # when wrap_around=False (valid idx range enforced by idx_min/max).
            for c in unique_clips:
                m = clip_ids == c
                phi_c = self._phi_rs_clips[c]  # (T_rs_c, D)
                now[m, k * D:(k + 1) * D] = phi_c.index_select(0, off_now[m])
                nxt[m, k * D:(k + 1) * D] = phi_c.index_select(0, off_next[m])
        return now, nxt

    def sample_reset_frames(
        self, num_samples: int, rng: torch.Generator | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Uniformly sample ``(motion_idx, frame_idx)`` + the state payload.

        Unchanged by the transition-alignment rework — still samples raw
        per-frame payloads for the env-side motion-reset strategy. Used by
        the env wrapper to seed root + joint state from a randomly sampled
        expert frame.
        """
        if len(self.motion_reset_payloads) == 1:
            payload = self.motion_reset_payloads[0]
            motion_idx = torch.zeros(num_samples, dtype=torch.long, device=self.device)
            frame_idx = torch.randint(
                low=0, high=payload.num_frames, size=(num_samples,),
                device=self.device, generator=rng,
            )
        else:
            motion_idx = torch.randint(
                low=0, high=len(self.motion_reset_payloads), size=(num_samples,),
                device=self.device, generator=rng,
            )
            frame_idx = torch.empty(num_samples, dtype=torch.long, device=self.device)
            for i in range(num_samples):
                m = int(motion_idx[i].item())
                frame_idx[i] = torch.randint(
                    low=0, high=self.motion_reset_payloads[m].num_frames, size=(),
                    device=self.device, generator=rng,
                )

        joint_pos = torch.empty(num_samples, self.spec.num_joints, device=self.device)
        joint_vel = torch.empty_like(joint_pos)
        root_pos = torch.empty(num_samples, 3, device=self.device)
        root_quat = torch.empty(num_samples, 4, device=self.device)
        root_lin_vel = torch.empty_like(root_pos)
        root_ang_vel = torch.empty_like(root_pos)

        for i in range(num_samples):
            m = int(motion_idx[i].item())
            f = int(frame_idx[i].item())
            p = self.motion_reset_payloads[m]
            joint_pos[i] = p.joint_pos[f]
            joint_vel[i] = p.joint_vel[f]
            root_pos[i] = p.root_pos[f]
            root_quat[i] = p.root_quat[f]
            root_lin_vel[i] = p.root_lin_vel[f]
            root_ang_vel[i] = p.root_ang_vel[f]

        out = {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "root_pos": root_pos,
            "root_quat": root_quat,
            "root_lin_vel": root_lin_vel,
            "root_ang_vel": root_ang_vel,
        }
        return motion_idx, out

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """Number of clips (not transitions — transitions are drawn from a
        continuous resampled grid)."""
        return len(self._phi_rs_clips)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"MotionDataset(files={len(self.motion_files)}, "
            f"env_step_dt={self.env_step_dt:.6f}, "
            f"clip_fps={self.motion_fps}, "
            f"T_src_per_clip={self.motion_lengths}, "
            f"T_rs_per_clip={self._clip_lengths_rs_list}, "
            f"frame_dim={self.frame_dim}, stack_k={self.spec.stack_k}, "
            f"amp_obs_dim={self.amp_obs_dim}, wrap_around={self.wrap_around})"
        )

    def summary(self) -> dict[str, object]:
        """Return a metadata dict convenient for logging / W&B config.

        Exposes the auto-derived resampling cadence (``env_step_dt``,
        per-clip ``T_rs``, valid now-index range) so a reader can verify
        the discriminator is being fed transitions at the control rate.
        """
        return {
            "motion_files": list(self.motion_files),
            "env_step_dt": float(self.env_step_dt),
            "clip_fps_src": list(self.motion_fps),
            "clip_T_src": list(self.motion_lengths),
            "clip_T_rs": list(self._clip_lengths_rs_list),
            "clip_duration_src_s": list(self._clip_duration_src),
            "clip_idx_min": list(self._clip_idx_min_list),
            "clip_idx_max": list(self._clip_idx_max_list),
            "clip_weights": self._clip_weights.detach().cpu().tolist(),
            "frame_dim": self.frame_dim,
            "stack_k": int(self.spec.stack_k),
            "amp_obs_dim": self.amp_obs_dim,
            "wrap_around": self.wrap_around,
            "amp_joint_names": list(self.amp_joint_names),
            "amp_body_names": list(self.amp_body_names),
        }
