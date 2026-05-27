from __future__ import annotations

import math
import numpy as np
import os
import torch
import torch.nn.functional as F
from collections.abc import Sequence
from dataclasses import MISSING
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_REQUIRED_KEYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


def _resolve_motion_files(
    motion_files: list[str] | None,
    motion_dir: str | None,
    motion_file_pattern: str,
) -> list[str]:
    if motion_files:
        files = [str(Path(p).resolve()) for p in motion_files]
        for p in files:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"motion_files entry not found: {p}")
        return files
    if motion_dir:
        root = Path(motion_dir)
        if not root.is_dir():
            raise NotADirectoryError(f"motion_dir is not a directory: {motion_dir}")
        files = sorted(str(p.resolve()) for p in root.rglob(motion_file_pattern))
        if not files:
            raise FileNotFoundError(
                f"No NPZ matching {motion_file_pattern!r} found under {motion_dir}"
            )
        return files
    raise ValueError(
        "MultiMotionCommandCfg: must specify either motion_files or motion_dir"
    )


class MultiMotionLoader:
    """Loads N independent motion clips and packs them into padded tensors.

    Layout (after init):
        joint_pos:      [M, T_max, J]
        joint_vel:      [M, T_max, J]
        body_pos_w:     [M, T_max, B_sel, 3]   (already body-subselected)
        body_quat_w:    [M, T_max, B_sel, 4]
        body_lin_vel_w: [M, T_max, B_sel, 3]
        body_ang_vel_w: [M, T_max, B_sel, 3]
        lengths:        [M]                    (per-motion valid frame count)
    """

    def __init__(
        self,
        motion_files: list[str],
        body_indexes: Sequence[int],
        device: str = "cpu",
    ):
        if len(motion_files) == 0:
            raise ValueError("MultiMotionLoader: motion_files is empty")

        per_motion: list[dict] = []
        kept_paths: list[str] = []
        skipped: list[tuple[str, str]] = []

        for path in motion_files:
            try:
                data = np.load(path)
            except Exception as e:  # noqa: BLE001
                skipped.append((path, f"failed to open: {e}"))
                continue
            missing = [k for k in _REQUIRED_KEYS if k not in data.files]
            if missing:
                skipped.append((path, f"missing keys {missing}"))
                continue
            per_motion.append(
                {
                    "path": path,
                    "fps": int(np.asarray(data["fps"]).reshape(-1)[0]),
                    "joint_pos": np.asarray(data["joint_pos"], dtype=np.float32),
                    "joint_vel": np.asarray(data["joint_vel"], dtype=np.float32),
                    "body_pos_w": np.asarray(data["body_pos_w"], dtype=np.float32),
                    "body_quat_w": np.asarray(data["body_quat_w"], dtype=np.float32),
                    "body_lin_vel_w": np.asarray(
                        data["body_lin_vel_w"], dtype=np.float32
                    ),
                    "body_ang_vel_w": np.asarray(
                        data["body_ang_vel_w"], dtype=np.float32
                    ),
                }
            )
            kept_paths.append(path)

        if skipped:
            print(
                f"[MultiMotionLoader] skipped {len(skipped)} file(s):"
            )
            for p, why in skipped:
                print(f"  [SKIP] {p} ({why})")

        if not per_motion:
            raise RuntimeError("MultiMotionLoader: no usable motion files after filtering")

        # Consistency checks: joint dim must match; body dim must match; fps recommend match.
        joint_dim = per_motion[0]["joint_pos"].shape[1]
        body_dim_full = per_motion[0]["body_pos_w"].shape[1]
        ref_fps = per_motion[0]["fps"]
        for m in per_motion:
            if m["joint_pos"].shape[1] != joint_dim:
                raise ValueError(
                    f"joint dim mismatch in {m['path']}: "
                    f"{m['joint_pos'].shape[1]} vs expected {joint_dim}"
                )
            if m["body_pos_w"].shape[1] != body_dim_full:
                raise ValueError(
                    f"body dim mismatch in {m['path']}: "
                    f"{m['body_pos_w'].shape[1]} vs expected {body_dim_full}"
                )
            if m["fps"] != ref_fps:
                print(
                    f"[MultiMotionLoader] WARNING: fps mismatch ({m['fps']} vs {ref_fps}) "
                    f"in {m['path']}"
                )

        # Validate body_indexes against full-body dim of motions
        body_indexes_t = torch.as_tensor(body_indexes, dtype=torch.long)
        if body_indexes_t.numel() == 0:
            raise ValueError("MultiMotionLoader: body_indexes is empty")
        if int(body_indexes_t.max().item()) >= body_dim_full:
            raise ValueError(
                f"body_indexes references body {int(body_indexes_t.max().item())}, "
                f"but motion contains only {body_dim_full} bodies"
            )

        num_motions = len(per_motion)
        lengths = np.array([m["joint_pos"].shape[0] for m in per_motion], dtype=np.int64)
        max_len = int(lengths.max())
        b_sel = body_indexes_t.numel()

        joint_pos = np.zeros((num_motions, max_len, joint_dim), dtype=np.float32)
        joint_vel = np.zeros_like(joint_pos)
        body_pos_w = np.zeros((num_motions, max_len, b_sel, 3), dtype=np.float32)
        body_quat_w = np.zeros((num_motions, max_len, b_sel, 4), dtype=np.float32)
        body_quat_w[..., 0] = 1.0
        body_lin_vel_w = np.zeros_like(body_pos_w)
        body_ang_vel_w = np.zeros_like(body_pos_w)

        sel_idx = body_indexes_t.cpu().numpy()
        for i, m in enumerate(per_motion):
            t = m["joint_pos"].shape[0]
            joint_pos[i, :t] = m["joint_pos"]
            joint_vel[i, :t] = m["joint_vel"]
            body_pos_w[i, :t] = m["body_pos_w"][:, sel_idx]
            body_quat_w[i, :t] = m["body_quat_w"][:, sel_idx]
            body_lin_vel_w[i, :t] = m["body_lin_vel_w"][:, sel_idx]
            body_ang_vel_w[i, :t] = m["body_ang_vel_w"][:, sel_idx]
            # Replicate last valid frame into padding region so accidental reads
            # remain physically plausible.
            if t < max_len:
                joint_pos[i, t:] = m["joint_pos"][-1]
                joint_vel[i, t:] = 0.0
                body_pos_w[i, t:] = m["body_pos_w"][-1, sel_idx]
                body_quat_w[i, t:] = m["body_quat_w"][-1, sel_idx]
                body_lin_vel_w[i, t:] = 0.0
                body_ang_vel_w[i, t:] = 0.0

        self.fps = ref_fps
        self.num_motions = num_motions
        self.max_time_steps = max_len
        self.lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        self.motion_paths = kept_paths
        self.joint_pos = torch.from_numpy(joint_pos).to(device)
        self.joint_vel = torch.from_numpy(joint_vel).to(device)
        self.body_pos_w = torch.from_numpy(body_pos_w).to(device)
        self.body_quat_w = torch.from_numpy(body_quat_w).to(device)
        self.body_lin_vel_w = torch.from_numpy(body_lin_vel_w).to(device)
        self.body_ang_vel_w = torch.from_numpy(body_ang_vel_w).to(device)

        print(
            f"[MultiMotionLoader] loaded {num_motions} motion(s), "
            f"joint_dim={joint_dim}, body_dim_full={body_dim_full}, "
            f"body_sel={b_sel}, max_len={max_len}, fps={ref_fps}"
        )


class MultiMotionCommand(CommandTerm):
    """Multi-motion command term with dual (motion-level + intra-motion) adaptive resampling."""

    cfg: MultiMotionCommandCfg

    def __init__(self, cfg: MultiMotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(
            self.cfg.anchor_body_name
        )
        self.motion_anchor_body_index = self.cfg.body_names.index(
            self.cfg.anchor_body_name
        )
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )

        files = _resolve_motion_files(
            cfg.motion_files, cfg.motion_dir, cfg.motion_file_pattern
        )
        self.motion = MultiMotionLoader(files, self.body_indexes, device=self.device)

        self.num_motions: int = self.motion.num_motions
        self.num_bins: int = int(cfg.adaptive_num_bins)
        if self.num_bins <= 0:
            raise ValueError("adaptive_num_bins must be > 0")

        # Per-env state: which motion + which frame each env is currently tracking.
        self.motion_ids = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.time_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        # Dual-level failure stats: [num_motions, num_bins]
        self.bin_failed_count = torch.zeros(
            self.num_motions, self.num_bins, dtype=torch.float, device=self.device
        )
        self._current_bin_failed = torch.zeros_like(self.bin_failed_count)

        kernel_size = max(int(cfg.adaptive_kernel_size), 1)
        self.kernel = torch.tensor(
            [cfg.adaptive_lambda**i for i in range(kernel_size)],
            device=self.device,
        )
        self.kernel = self.kernel / self.kernel.sum()
        self._kernel_size = kernel_size

        # Anchor-relative buffers reused by rewards/observations.
        self.body_pos_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 3, device=self.device
        )
        self.body_quat_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 4, device=self.device
        )
        self.body_quat_relative_w[:, :, 0] = 1.0

        # ---- metrics (mirroring single-motion command, plus multi-motion stats) ----
        z = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_pos"] = z.clone()
        self.metrics["error_anchor_rot"] = z.clone()
        self.metrics["error_anchor_lin_vel"] = z.clone()
        self.metrics["error_anchor_ang_vel"] = z.clone()
        self.metrics["error_body_pos"] = z.clone()
        self.metrics["error_body_rot"] = z.clone()
        self.metrics["error_body_lin_vel"] = z.clone()
        self.metrics["error_body_ang_vel"] = z.clone()
        self.metrics["error_joint_pos"] = z.clone()
        self.metrics["error_joint_vel"] = z.clone()
        # sampling diagnostics
        self.metrics["sampling_motion_entropy"] = z.clone()
        self.metrics["sampling_motion_top1_prob"] = z.clone()
        self.metrics["sampling_motion_top1_id"] = z.clone()
        self.metrics["sampling_bin_entropy_mean"] = z.clone()
        self.metrics["sampling_bin_top1_prob_mean"] = z.clone()
        self.metrics["sampling_motion_id_mean"] = z.clone()
        self.metrics["sampling_motion_id_min"] = z.clone()
        self.metrics["sampling_motion_id_max"] = z.clone()

    # ------------------------------------------------------------------ #
    # Reference-motion accessors (per-env via [motion_ids, time_steps])  #
    # ------------------------------------------------------------------ #

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.motion_ids, self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.motion_ids, self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return (
            self.motion.body_pos_w[self.motion_ids, self.time_steps]
            + self._env.scene.env_origins[:, None, :]
        )

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.motion_ids, self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.motion_ids, self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.motion_ids, self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return (
            self.motion.body_pos_w[
                self.motion_ids, self.time_steps, self.motion_anchor_body_index
            ]
            + self._env.scene.env_origins
        )

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[
            self.motion_ids, self.time_steps, self.motion_anchor_body_index
        ]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[
            self.motion_ids, self.time_steps, self.motion_anchor_body_index
        ]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[
            self.motion_ids, self.time_steps, self.motion_anchor_body_index
        ]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    # ------------------------------------------------------------------ #
    # Metrics                                                             #
    # ------------------------------------------------------------------ #

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(
            self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
        )
        self.metrics["error_anchor_rot"] = quat_error_magnitude(
            self.anchor_quat_w, self.robot_anchor_quat_w
        )
        self.metrics["error_anchor_lin_vel"] = torch.norm(
            self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
        )
        self.metrics["error_anchor_ang_vel"] = torch.norm(
            self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
        )

        self.metrics["error_body_pos"] = torch.norm(
            self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(
            self.body_quat_relative_w, self.robot_body_quat_w
        ).mean(dim=-1)

        self.metrics["error_body_lin_vel"] = torch.norm(
            self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_ang_vel"] = torch.norm(
            self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
        ).mean(dim=-1)

        self.metrics["error_joint_pos"] = torch.norm(
            self.joint_pos - self.robot_joint_pos, dim=-1
        )
        self.metrics["error_joint_vel"] = torch.norm(
            self.joint_vel - self.robot_joint_vel, dim=-1
        )

        # Multi-motion id stats (cheap; helps confirm we're not stuck on one clip)
        mid_f = self.motion_ids.float()
        self.metrics["sampling_motion_id_mean"][:] = mid_f.mean()
        self.metrics["sampling_motion_id_min"][:] = mid_f.min()
        self.metrics["sampling_motion_id_max"][:] = mid_f.max()

    # ------------------------------------------------------------------ #
    # Adaptive sampling                                                   #
    # ------------------------------------------------------------------ #

    def _record_failures(self, env_ids: torch.Tensor):
        """Accumulate per-(motion, bin) failure counts for envs being reset."""
        if env_ids.numel() == 0:
            return
        terminated = self._env.termination_manager.terminated[env_ids]
        if not torch.any(terminated):
            return
        failed_idx = env_ids[terminated]
        f_motion = self.motion_ids[failed_idx]
        f_step = self.time_steps[failed_idx]
        f_len = self.motion.lengths[f_motion].clamp(min=1)
        f_bin = torch.clamp(
            (f_step * self.num_bins) // f_len, max=self.num_bins - 1
        )
        flat = (f_motion * self.num_bins + f_bin).long()
        ones = torch.ones_like(flat, dtype=torch.float)
        self._current_bin_failed.view(-1).index_add_(0, flat, ones)

    def _sample_motion_ids(self, n: int) -> torch.Tensor:
        if self.cfg.motion_sampling_strategy == "uniform" or self.num_motions == 1:
            return torch.randint(
                0, self.num_motions, (n,), dtype=torch.long, device=self.device
            )
        # adaptive
        score = self.bin_failed_count.sum(dim=1) + float(self.cfg.motion_uniform_ratio)
        score = score.clamp(min=1e-12)
        prob = score / score.sum()
        sampled = torch.multinomial(prob, n, replacement=True)

        # metrics
        H = -(prob * (prob + 1e-12).log()).sum()
        H_norm = H / max(math.log(max(self.num_motions, 1)), 1e-12)
        pmax, imax = prob.max(dim=0)
        self.metrics["sampling_motion_entropy"][:] = H_norm
        self.metrics["sampling_motion_top1_prob"][:] = pmax
        self.metrics["sampling_motion_top1_id"][:] = imax.float()
        return sampled

    def _sample_time_steps(self, motion_ids: torch.Tensor) -> torch.Tensor:
        n = motion_ids.numel()
        if n == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)

        # Per-env bin distribution: take that motion's row + base uniform.
        per_env_score = self.bin_failed_count[motion_ids] + (
            float(self.cfg.adaptive_uniform_ratio) / float(self.num_bins)
        )

        # Optional kernel smoothing across the bin axis (per env, replicate-pad on the right).
        if self._kernel_size > 1:
            sm = F.pad(
                per_env_score.unsqueeze(1),
                (0, self._kernel_size - 1),
                mode="replicate",
            )
            sm = F.conv1d(sm, self.kernel.view(1, 1, -1)).squeeze(1)
        else:
            sm = per_env_score
        sm = sm.clamp(min=1e-12)
        sm = sm / sm.sum(dim=1, keepdim=True)

        sampled_bins = torch.multinomial(sm, 1, replacement=True).squeeze(1)

        # Map bin -> frame within that motion's valid length.
        future = max(int(self.cfg.future_horizon), 0)
        lengths = self.motion.lengths[motion_ids]
        max_starts = (lengths - future - 1).clamp(min=1).float()
        rand = torch.rand_like(max_starts)
        bin_lo = (sampled_bins.float() / float(self.num_bins)) * max_starts
        bin_hi = ((sampled_bins.float() + 1.0) / float(self.num_bins)) * max_starts
        frames = (bin_lo + rand * (bin_hi - bin_lo)).long().clamp(min=0)
        # Final guard against rounding off-by-one past the valid region.
        frames = torch.minimum(frames, (lengths - 1).clamp(min=0))

        # bin distribution metrics (averaged across envs sampled this call)
        H = -(sm * (sm + 1e-12).log()).sum(dim=1)
        H_norm = (H / math.log(max(self.num_bins, 2))).mean()
        pmax = sm.max(dim=1).values.mean()
        self.metrics["sampling_bin_entropy_mean"][:] = H_norm
        self.metrics["sampling_bin_top1_prob_mean"][:] = pmax

        return frames

    def _resample_command(self, env_ids: Sequence[int]):
        if isinstance(env_ids, torch.Tensor):
            env_ids_t = env_ids
        else:
            env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids_t.numel() == 0:
            return

        # 1. Record failures of envs we are about to reset.
        self._record_failures(env_ids_t)

        # 2. Sample new (motion_id, time_step) for each env being reset.
        new_motion_ids = self._sample_motion_ids(env_ids_t.numel())
        new_time_steps = self._sample_time_steps(new_motion_ids)
        self.motion_ids[env_ids_t] = new_motion_ids
        self.time_steps[env_ids_t] = new_time_steps

        # 3. Reset robot state to match reference (with randomization), reusing the
        #    same logic as the single-motion command.
        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        n_envs = env_ids_t.numel()

        range_list = [
            self.cfg.pose_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(
            ranges[:, 0], ranges[:, 1], (n_envs, 6), device=self.device
        )
        root_pos[env_ids_t] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(
            rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
        )
        root_ori[env_ids_t] = quat_mul(orientations_delta, root_ori[env_ids_t])
        range_list = [
            self.cfg.velocity_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(
            ranges[:, 0], ranges[:, 1], (n_envs, 6), device=self.device
        )
        root_lin_vel[env_ids_t] += rand_samples[:, :3]
        root_ang_vel[env_ids_t] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(
            *self.cfg.joint_position_range, joint_pos.shape, joint_pos.device
        )
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids_t]
        joint_pos[env_ids_t] = torch.clip(
            joint_pos[env_ids_t],
            soft_joint_pos_limits[:, :, 0],
            soft_joint_pos_limits[:, :, 1],
        )
        self.robot.write_joint_state_to_sim(
            joint_pos[env_ids_t], joint_vel[env_ids_t], env_ids=env_ids_t
        )
        self.robot.write_root_state_to_sim(
            torch.cat(
                [
                    root_pos[env_ids_t],
                    root_ori[env_ids_t],
                    root_lin_vel[env_ids_t],
                    root_ang_vel[env_ids_t],
                ],
                dim=-1,
            ),
            env_ids=env_ids_t,
        )

    def _update_command(self):
        # Advance time, then resample any envs that crossed THEIR motion's end frame.
        self.time_steps += 1
        env_lengths = self.motion.lengths[self.motion_ids]
        env_ids = torch.where(self.time_steps >= env_lengths)[0]
        self._resample_command(env_ids)

        # Anchor-relative reference targets (xy follows robot, z follows reference,
        # yaw aligned to robot).
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1
        )
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1
        )
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1
        )
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1
        )

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(
            quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
        )

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(
            delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
        )

        # EMA update of (motion, bin) failure counts.
        self.bin_failed_count = (
            float(self.cfg.adaptive_alpha) * self._current_bin_failed
            + (1.0 - float(self.cfg.adaptive_alpha)) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    # ------------------------------------------------------------------ #
    # Visualization (mirrors single-motion command)                       #
    # ------------------------------------------------------------------ #

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(
                        prim_path="/Visuals/Command/current/anchor"
                    )
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(
                        prim_path="/Visuals/Command/goal/anchor"
                    )
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(
                                prim_path="/Visuals/Command/current/" + name
                            )
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(
                                prim_path="/Visuals/Command/goal/" + name
                            )
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)
        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        self.current_anchor_visualizer.visualize(
            self.robot_anchor_pos_w, self.robot_anchor_quat_w
        )
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)
        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(
                self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i]
            )
            self.goal_body_visualizers[i].visualize(
                self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i]
            )


@configclass
class MultiMotionCommandCfg(CommandTermCfg):
    """Configuration for the multi-motion command (BeyondMimic)."""

    class_type: type = MultiMotionCommand

    asset_name: str = MISSING

    # Motion source: at least one of motion_files / motion_dir must be set.
    motion_files: list[str] = []
    motion_dir: str | None = None
    motion_file_pattern: str = "*.npz"

    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    # Adaptive sampling (dual-level).
    adaptive_num_bins: int = 50
    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    motion_sampling_strategy: str = "adaptive"  # "uniform" or "adaptive"
    motion_uniform_ratio: float = 1.0

    # How many frames into the future a downstream consumer might peek; the
    # sampler keeps that many frames as headroom when picking a start frame.
    future_horizon: int = 1

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose"
    )
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose"
    )
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
