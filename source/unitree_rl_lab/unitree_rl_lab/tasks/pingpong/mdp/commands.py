from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import SPHERE_MARKER_CFG
from isaaclab.utils import configclass
try:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz, sample_uniform
except ImportError:
    from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_rotate_inverse as quat_apply_inverse, sample_uniform

from .motion_loader import DEFAULT_EXPERT_ROOT, PingpongMotionLoader, PingpongRefState, yaw_from_wxyz

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


SWING_FOREHAND = 0
SWING_BACKHAND = 1
BLADE_NORMAL_LOCAL = (0.0, 1.0, 0.0)


def _as_env_ids(env_ids: Sequence[int] | torch.Tensor | slice, num_envs: int, device: torch.device) -> torch.Tensor:
    if isinstance(env_ids, slice):
        return torch.arange(num_envs, device=device)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=device, dtype=torch.long)
    return torch.tensor(env_ids, dtype=torch.long, device=device)


def _rotate_yaw_2d(vec_xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    return torch.stack((c * vec_xy[:, 0] - s * vec_xy[:, 1], s * vec_xy[:, 0] + c * vec_xy[:, 1]), dim=-1)


def _sample_peak_uniform(
    low: float,
    high: float,
    peak_low: float,
    peak_high: float,
    shape: tuple[int, ...],
    device: torch.device,
    peak_prob: float = 0.7,
) -> torch.Tensor:
    full = sample_uniform(low, high, shape, device=device)
    peak = sample_uniform(peak_low, peak_high, shape, device=device)
    mask = torch.rand(shape, device=device) < peak_prob
    return torch.where(mask, peak, full)


class PingpongCommand(CommandTerm):
    cfg: "PingpongCommandCfg"

    def __init__(self, cfg: "PingpongCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.dt = float(getattr(env, "step_dt", env.cfg.sim.dt * env.cfg.decimation))

        self.tracked_body_ids = torch.tensor(
            self.robot.find_bodies(cfg.tracked_body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )
        self.pelvis_body_id = self.robot.find_bodies(cfg.anchor_body_name, preserve_order=True)[0][0]
        self.blade_body_id = self.robot.find_bodies(cfg.blade_body_name, preserve_order=True)[0][0]
        self.upper_joint_ids = torch.tensor(
            self.robot.find_joints(cfg.imitation_joint_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )

        self.motion = PingpongMotionLoader(
            cfg.forward_motion_file,
            cfg.backward_motion_file,
            cfg.tracked_body_names,
            device=self.device,
        )
        self.expert_offset_base = self.motion.expert_offset_base.to(self.device)

        n = self.num_envs
        self.swing_type = torch.zeros(n, dtype=torch.long, device=self.device)
        self.swing_change_remaining = torch.zeros(n, dtype=torch.long, device=self.device)
        self.p_hit_world = torch.zeros(n, 3, device=self.device)
        self.v_ball_in_world = torch.zeros(n, 3, device=self.device)
        self.target_land_world = torch.zeros(n, 3, device=self.device)
        self.flight_time = torch.zeros(n, device=self.device)
        self.paddle_cor = torch.full((n,), cfg.paddle_cor, device=self.device)
        self.v_racket_hat_world = torch.zeros(n, 3, device=self.device)
        self.n_target_world = torch.zeros(n, 3, device=self.device)
        self.v_ball_out_world = torch.zeros(n, 3, device=self.device)
        self.p_base_xy_world = torch.zeros(n, 2, device=self.device)
        self.t_pre_initial = torch.zeros(n, device=self.device)
        self.t_post_swing = torch.zeros(n, device=self.device)
        self.t_to_hit = torch.zeros(n, device=self.device)
        self.cur_step = torch.zeros(n, dtype=torch.long, device=self.device)
        self.hit_y_base = torch.zeros(n, device=self.device)

        self.noise_p = torch.zeros(n, 3, device=self.device)
        self.noise_v = torch.zeros(n, 3, device=self.device)
        self.noise_base = torch.zeros(n, 2, device=self.device)
        self.noise_t = torch.zeros(n, 1, device=self.device)
        self.last_resample_was_degenerate = torch.zeros(n, dtype=torch.bool, device=self.device)

        self.ref_state = self._empty_ref_state()

        self._strike_seen = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._pos_ok_window = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._vel_ok_window = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._ori_ok_window = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._success_window = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._swing_count = torch.zeros(n, device=self.device)
        self._success_count = torch.zeros(n, device=self.device)
        self._pos_fail_count = torch.zeros(n, device=self.device)
        self._vel_fail_count = torch.zeros(n, device=self.device)
        self._ori_fail_count = torch.zeros(n, device=self.device)
        self._swing_change_used_count = torch.zeros(n, device=self.device)
        self._dead_zone_count = torch.zeros(n, device=self.device)
        self._strike_dist_min = torch.full((n,), float("inf"), device=self.device)
        self._last_strike_dist_min = torch.zeros(n, device=self.device)
        self._debug_traj_points = torch.zeros(n, cfg.debug_traj_len, 3, device=self.device)
        self._debug_traj_valid = torch.zeros(n, cfg.debug_traj_len, dtype=torch.bool, device=self.device)
        self._debug_traj_cursor = torch.zeros(n, dtype=torch.long, device=self.device)

        metric_names = [
            "hit_success_rate",
            "hit_success_pos_fail_rate",
            "hit_success_vel_fail_rate",
            "hit_success_ori_fail_rate",
            "swing_ratio_forehand",
            "dead_zone_trigger_rate",
            "swing_flip_rate_per_episode",
            "base_y_drift_meanabs",
            "v_racket_hat_world_mag_mean",
            "v_racket_hat_world_mag_std",
            "solve_paddle_degenerate_rate",
            "cos_sim_n_blade_n_target_at_impact",
            "swing_change_remaining_used_rate",
            "strike_blade_hit_dist_min",
        ]
        for name in metric_names:
            self.metrics[name] = torch.zeros(n, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat((self.p_base_xy_world, self.p_hit_world, self.v_racket_hat_world, self.t_to_hit.unsqueeze(-1)), dim=-1)

    @property
    def robot_pelvis_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.pelvis_body_id]

    @property
    def robot_pelvis_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.pelvis_body_id]

    @property
    def robot_blade_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.blade_body_id]

    @property
    def robot_blade_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.blade_body_id]

    @property
    def robot_blade_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.blade_body_id]

    @property
    def robot_tracked_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.tracked_body_ids]

    @property
    def robot_tracked_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.tracked_body_ids]

    def _empty_ref_state(self) -> PingpongRefState:
        n = self.num_envs
        b = len(self.cfg.tracked_body_names)
        return PingpongRefState(
            joint_pos=torch.zeros(n, self.robot.num_joints, device=self.device),
            joint_vel=torch.zeros(n, self.robot.num_joints, device=self.device),
            body_pos_w=torch.zeros(n, b, 3, device=self.device),
            body_quat_w=torch.nn.functional.pad(torch.zeros(n, b, 3, device=self.device), (1, 0), value=1.0),
            body_lin_vel_w=torch.zeros(n, b, 3, device=self.device),
            body_ang_vel_w=torch.zeros(n, b, 3, device=self.device),
            pelvis_pos_w=torch.zeros(n, 3, device=self.device),
            pelvis_quat_w=torch.nn.functional.pad(torch.zeros(n, 3, device=self.device), (1, 0), value=1.0),
            pelvis_lin_vel_w=torch.zeros(n, 3, device=self.device),
            pelvis_ang_vel_w=torch.zeros(n, 3, device=self.device),
            ref_frame_f=torch.zeros(n, device=self.device),
        )

    def _update_metrics(self):
        self._update_success_window()
        self._refresh_metrics_from_counts()

    def _refresh_metrics_from_counts(self):
        denom = torch.clamp(self._swing_count, min=1.0)
        self.metrics["hit_success_rate"] = self._success_count / denom
        self.metrics["hit_success_pos_fail_rate"] = self._pos_fail_count / denom
        self.metrics["hit_success_vel_fail_rate"] = self._vel_fail_count / denom
        self.metrics["hit_success_ori_fail_rate"] = self._ori_fail_count / denom
        self.metrics["swing_ratio_forehand"] = (self.swing_type == SWING_FOREHAND).float()
        self.metrics["dead_zone_trigger_rate"] = self._dead_zone_count / torch.clamp(self.command_counter.float(), min=1.0)
        self.metrics["swing_flip_rate_per_episode"] = self._swing_change_used_count
        self.metrics["base_y_drift_meanabs"] = torch.abs(self.robot.data.root_pos_w[:, 1] - self._env.scene.env_origins[:, 1])
        self.metrics["v_racket_hat_world_mag_mean"] = torch.linalg.norm(self.v_racket_hat_world, dim=-1)
        self.metrics["v_racket_hat_world_mag_std"] = torch.zeros_like(self.metrics["v_racket_hat_world_mag_mean"])
        self.metrics["solve_paddle_degenerate_rate"] = self.last_resample_was_degenerate.float()
        self.metrics["cos_sim_n_blade_n_target_at_impact"] = self._blade_target_cosine()
        self.metrics["swing_change_remaining_used_rate"] = self._swing_change_used_count / torch.clamp(
            self.command_counter.float(), min=1.0
        )
        self.metrics["strike_blade_hit_dist_min"] = torch.where(
            torch.isfinite(self._strike_dist_min), self._strike_dist_min, self._last_strike_dist_min
        )

    def finalize_partial_swings(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        ids = _as_env_ids(slice(None) if env_ids is None else env_ids, self.num_envs, self.device)
        self._update_success_window()
        completed_on_timeout = ids[self._strike_seen[ids]]
        if len(completed_on_timeout) > 0:
            self._complete_swing(completed_on_timeout)
        self._refresh_metrics_from_counts()

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        self.finalize_partial_swings(env_ids)
        return super().reset(env_ids)

    def _resample_command(self, env_ids: Sequence[int]):
        ids = _as_env_ids(env_ids, self.num_envs, self.device)
        if ids.numel() == 0:
            return
        self._reset_counters(ids)
        root_pos, root_quat = self._write_nominal_root(ids)
        self._sample_new_swing(ids, reset_robot=True, root_pos_override=root_pos, root_quat_override=root_quat)

    def _update_command(self):
        self.t_to_hit -= self.dt
        self.cur_step += 1

        pre_ids = torch.nonzero((self.t_to_hit > 0.0) & (self.swing_change_remaining > 0), as_tuple=False).flatten()
        if len(pre_ids) > 0:
            new_swing, hit_y_base = self._compute_swing_type(pre_ids, self.robot.data.root_pos_w[pre_ids, :2], self.robot.data.root_quat_w[pre_ids])
            changed = new_swing != self.swing_type[pre_ids]
            if torch.any(changed):
                ids = pre_ids[changed]
                self.swing_type[ids] = new_swing[changed]
                self.swing_change_remaining[ids] = 0
                self.hit_y_base[ids] = hit_y_base[changed]
                self.p_base_xy_world[ids] = self._compute_base_target(ids, self.robot.data.root_quat_w[ids])
                self._swing_change_used_count[ids] += 1.0

        done_ids = torch.nonzero(self.t_to_hit <= -self.t_post_swing, as_tuple=False).flatten()
        if len(done_ids) > 0:
            self._complete_swing(done_ids)
            self._sample_new_swing(done_ids, reset_robot=False)
            self.command_counter[done_ids] += 1
            self.time_left[done_ids] = self.cfg.resampling_time_range[1]

        self._update_ref_state()

    def _sample_new_swing(
        self,
        ids: torch.Tensor,
        reset_robot: bool,
        root_pos_override: torch.Tensor | None = None,
        root_quat_override: torch.Tensor | None = None,
    ) -> None:
        root_pos = self.robot.data.root_pos_w[ids] if root_pos_override is None else root_pos_override
        root_quat = self.robot.data.root_quat_w[ids] if root_quat_override is None else root_quat_override
        env_origins = self._env.scene.env_origins[ids]

        hit_y = sample_uniform(self.cfg.hit_y_range[0], self.cfg.hit_y_range[1], (len(ids),), device=self.device)
        hit_z = sample_uniform(self.cfg.hit_z_range[0], self.cfg.hit_z_range[1], (len(ids),), device=self.device)
        local_hit = torch.stack((torch.full_like(hit_y, self.cfg.hit_x), hit_y, hit_z), dim=-1)
        self.p_hit_world[ids] = env_origins + local_hit

        v_mag = sample_uniform(self.cfg.v_in_mag_range[0], self.cfg.v_in_mag_range[1], (len(ids),), device=self.device)
        v_yaw = math.pi + sample_uniform(-math.radians(40.0), math.radians(40.0), (len(ids),), device=self.device)
        v_pitch = sample_uniform(-math.radians(75.0), math.radians(75.0), (len(ids),), device=self.device)
        self.v_ball_in_world[ids] = v_mag.unsqueeze(-1) * torch.stack(
            (torch.cos(v_yaw) * torch.cos(v_pitch), torch.sin(v_yaw) * torch.cos(v_pitch), torch.sin(v_pitch)), dim=-1
        )

        local_target = torch.tensor(self.cfg.target_land, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.target_land_world[ids] = env_origins + local_target
        self.flight_time[ids] = sample_uniform(
            self.cfg.flight_time_range[0], self.cfg.flight_time_range[1], (len(ids),), device=self.device
        )
        self.paddle_cor[ids] = sample_uniform(
            self.cfg.paddle_cor_range[0], self.cfg.paddle_cor_range[1], (len(ids),), device=self.device
        )
        self._solve_paddle_target(ids)

        swing, hit_y_base = self._compute_swing_type(ids, root_pos[:, :2], root_quat)
        self.swing_type[ids] = swing
        self.hit_y_base[ids] = hit_y_base
        self.swing_change_remaining[ids] = 1
        self._dead_zone_count[ids] += (torch.abs(hit_y_base - self.cfg.y_mid_base) < self.cfg.swing_dead_zone).float()

        self.p_base_xy_world[ids] = self._compute_base_target(ids, root_quat)
        self.t_pre_initial[ids] = _sample_peak_uniform(0.20, 0.90, 0.30, 0.65, (len(ids),), self.device)
        self.t_post_swing[ids] = float(self.cfg.t_post_swing_fixed)
        self.t_to_hit[ids] = self.t_pre_initial[ids]
        self.cur_step[ids] = 0

        if reset_robot:
            self._write_rsi_joint_state(ids)

        self._reset_window_flags(ids)
        self._freeze_noise(ids)
        self._update_ref_state(ids)

    def _solve_paddle_target(self, ids: torch.Tensor) -> None:
        g = 9.81
        t = self.flight_time[ids].unsqueeze(-1)
        gravity_term = torch.tensor((0.0, 0.0, 0.5 * g), device=self.device).unsqueeze(0) * t
        v_out = (self.target_land_world[ids] - self.p_hit_world[ids]) / t + gravity_term
        delta_v = v_out - self.v_ball_in_world[ids]
        norm = torch.linalg.norm(delta_v, dim=-1, keepdim=True)
        degenerate = norm.squeeze(-1) < 1e-9
        n_target = delta_v / norm.clamp_min(1e-9)
        fallback_n = torch.tensor((-1.0, 0.0, 0.0), device=self.device).expand_as(n_target)
        n_target = torch.where(degenerate.unsqueeze(-1), fallback_n, n_target)
        v_in_n = torch.sum(self.v_ball_in_world[ids] * n_target, dim=-1)
        v_out_n = torch.sum(v_out * n_target, dim=-1)
        cor = self.paddle_cor[ids]
        v_pad_n = (v_out_n + cor * v_in_n) / (1.0 + cor)
        v_racket = v_pad_n.unsqueeze(-1) * n_target
        v_racket = torch.where(degenerate.unsqueeze(-1), 2.0 * fallback_n, v_racket)

        self.v_ball_out_world[ids] = v_out
        self.n_target_world[ids] = n_target
        self.v_racket_hat_world[ids] = v_racket
        self.last_resample_was_degenerate[ids] = degenerate

    def _compute_swing_type(self, ids: torch.Tensor, root_xy: torch.Tensor, root_quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        yaw = yaw_from_wxyz(root_quat)
        diff = self.p_hit_world[ids, :2] - root_xy
        hit_base = _rotate_yaw_2d(diff, -yaw)
        hit_y_base = hit_base[:, 1]
        swing = torch.where(hit_y_base > self.cfg.y_mid_base, SWING_FOREHAND, SWING_BACKHAND).long()
        return swing, hit_y_base

    def _compute_base_target(self, ids: torch.Tensor, root_quat: torch.Tensor) -> torch.Tensor:
        yaw = yaw_from_wxyz(root_quat)
        offsets = self.expert_offset_base[self.swing_type[ids]]
        offsets_world = _rotate_yaw_2d(offsets, yaw)
        return self.p_hit_world[ids, :2] - offsets_world

    def _write_nominal_root(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        env_origins = self._env.scene.env_origins[ids]
        root_pos = env_origins + torch.tensor(self.cfg.reset_root_pos, dtype=torch.float32, device=self.device).unsqueeze(0)
        yaw_noise = sample_uniform(self.cfg.reset_yaw_noise[0], self.cfg.reset_yaw_noise[1], (len(ids),), device=self.device)
        root_quat = quat_from_euler_xyz(torch.zeros_like(yaw_noise), torch.zeros_like(yaw_noise), yaw_noise)
        root_lin = torch.zeros(len(ids), 3, device=self.device)
        root_ang = torch.zeros(len(ids), 3, device=self.device)
        self.robot.write_root_state_to_sim(torch.cat((root_pos, root_quat, root_lin, root_ang), dim=-1), env_ids=ids)
        return root_pos, root_quat

    def _write_rsi_joint_state(self, ids: torch.Tensor) -> None:
        frames = torch.empty(len(ids), dtype=torch.long, device=self.device)
        for swing_idx, name in ((SWING_FOREHAND, "forehand"), (SWING_BACKHAND, "backhand")):
            sub_ids = torch.nonzero(self.swing_type[ids] == swing_idx, as_tuple=False).flatten()
            if len(sub_ids) == 0:
                continue
            clip = self.motion.clips[name]
            frames[sub_ids] = torch.randint(0, clip.length, (len(sub_ids),), device=self.device)
        joint_pos = self.robot.data.default_joint_pos[ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        for swing_idx, name in ((SWING_FOREHAND, "forehand"), (SWING_BACKHAND, "backhand")):
            local = torch.nonzero(self.swing_type[ids] == swing_idx, as_tuple=False).flatten()
            if len(local) == 0:
                continue
            clip = self.motion.clips[name]
            joint_pos[local] = clip.joint_pos[frames[local]]
            joint_vel[local] = clip.joint_vel[frames[local]]
        limits = self.robot.data.soft_joint_pos_limits[ids]
        joint_pos = torch.clamp(joint_pos, limits[..., 0], limits[..., 1])
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=ids)

    def _freeze_noise(self, ids: torch.Tensor) -> None:
        sig_p = self.cfg.noise_p_sigma
        sig_v = self.cfg.noise_v_sigma
        sig_base = self.cfg.noise_base_sigma
        sig_t = self.cfg.noise_t_sigma
        self.noise_p[ids] = torch.clamp(torch.randn(len(ids), 3, device=self.device) * sig_p, -3.0 * sig_p, 3.0 * sig_p)
        self.noise_v[ids] = torch.clamp(torch.randn(len(ids), 3, device=self.device) * sig_v, -3.0 * sig_v, 3.0 * sig_v)
        self.noise_base[ids] = torch.clamp(
            torch.randn(len(ids), 2, device=self.device) * sig_base, -3.0 * sig_base, 3.0 * sig_base
        )
        self.noise_t[ids] = torch.clamp(torch.randn(len(ids), 1, device=self.device) * sig_t, -3.0 * sig_t, 3.0 * sig_t)

    def _update_ref_state(self, ids: torch.Tensor | None = None) -> None:
        if ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        sub = self.motion.sample(
            self.swing_type[ids],
            self.cur_step[ids],
            self.t_pre_initial[ids],
            self.t_post_swing[ids],
            self.dt,
            self._env.scene.env_origins[ids],
        )
        for name in self.ref_state.__dataclass_fields__:
            getattr(self.ref_state, name)[ids] = getattr(sub, name)

    def _blade_target_cosine(self) -> torch.Tensor:
        normal_local = torch.tensor(BLADE_NORMAL_LOCAL, dtype=torch.float32, device=self.device).expand(self.num_envs, 3)
        n_blade = quat_apply(self.robot_blade_quat_w, normal_local)
        return torch.sum(n_blade * self.n_target_world, dim=-1).clamp(-1.0, 1.0)

    def _task_errors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w
        p_blade_base = quat_apply_inverse(root_quat, self.robot_blade_pos_w - root_pos)
        p_hit_base = quat_apply_inverse(root_quat, self.p_hit_world - root_pos)
        v_blade_base = quat_apply_inverse(root_quat, self.robot_blade_lin_vel_w)
        v_hat_base = quat_apply_inverse(root_quat, self.v_racket_hat_world)
        pos_err = torch.linalg.norm(p_blade_base - p_hit_base, dim=-1)
        vel_err = torch.linalg.norm(v_blade_base - v_hat_base, dim=-1)
        ori_dist = 1.0 - self._blade_target_cosine()
        return pos_err, vel_err, ori_dist

    def _update_success_window(self) -> None:
        in_window = torch.abs(self.t_to_hit) <= self.cfg.strike_window
        if not torch.any(in_window):
            return
        pos_err, vel_err, ori_dist = self._task_errors()
        pos_thresh = torch.clamp(2.0 * torch.full_like(pos_err, self.cfg.sigma_g_pos), min=self.cfg.success_pos_floor)
        pos_ok = pos_err < pos_thresh
        vel_ok = vel_err < self.cfg.success_vel_thresh
        ori_ok = ori_dist < self.cfg.success_ori_cos_dist_thresh
        blade_hit_dist = torch.linalg.norm(self.robot_blade_pos_w - self.p_hit_world, dim=-1)
        self._strike_dist_min = torch.where(in_window, torch.minimum(self._strike_dist_min, blade_hit_dist), self._strike_dist_min)
        traj_ids = torch.nonzero(in_window, as_tuple=False).flatten()
        if len(traj_ids) > 0:
            slots = self._debug_traj_cursor[traj_ids] % self.cfg.debug_traj_len
            self._debug_traj_points[traj_ids, slots] = self.robot_blade_pos_w[traj_ids]
            self._debug_traj_valid[traj_ids, slots] = True
            self._debug_traj_cursor[traj_ids] += 1
        self._strike_seen |= in_window
        self._pos_ok_window |= in_window & pos_ok
        self._vel_ok_window |= in_window & vel_ok
        self._ori_ok_window |= in_window & ori_ok
        self._success_window |= in_window & pos_ok & vel_ok & ori_ok

    def _complete_swing(self, ids: torch.Tensor) -> None:
        valid = self._strike_seen[ids]
        if not torch.any(valid):
            self._reset_window_flags(ids)
            return
        done = ids[valid]
        self._swing_count[done] += 1.0
        self._success_count[done] += self._success_window[done].float()
        self._pos_fail_count[done] += (~self._pos_ok_window[done]).float()
        self._vel_fail_count[done] += (~self._vel_ok_window[done]).float()
        self._ori_fail_count[done] += (~self._ori_ok_window[done]).float()
        self._last_strike_dist_min[done] = torch.where(
            torch.isfinite(self._strike_dist_min[done]), self._strike_dist_min[done], self._last_strike_dist_min[done]
        )
        self._reset_window_flags(ids)

    def _reset_counters(self, ids: torch.Tensor) -> None:
        self._swing_count[ids] = 0.0
        self._success_count[ids] = 0.0
        self._pos_fail_count[ids] = 0.0
        self._vel_fail_count[ids] = 0.0
        self._ori_fail_count[ids] = 0.0
        self._swing_change_used_count[ids] = 0.0
        self._dead_zone_count[ids] = 0.0
        self._last_strike_dist_min[ids] = 0.0
        self._reset_window_flags(ids)

    def _reset_window_flags(self, ids: torch.Tensor) -> None:
        self._strike_seen[ids] = False
        self._pos_ok_window[ids] = False
        self._vel_ok_window[ids] = False
        self._ori_ok_window[ids] = False
        self._success_window[ids] = False
        self._strike_dist_min[ids] = float("inf")
        self._debug_traj_valid[ids] = False
        self._debug_traj_cursor[ids] = 0

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "target_visualizer"):
                self.target_visualizer = VisualizationMarkers(
                    SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Pingpong/targets")
                )
            self.target_visualizer.set_visibility(True)
        elif hasattr(self, "target_visualizer"):
            self.target_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized or not hasattr(self, "target_visualizer"):
            return
        normal_local = torch.tensor(BLADE_NORMAL_LOCAL, dtype=torch.float32, device=self.device).expand(self.num_envs, 3)
        blade_normal_end = self.robot_blade_pos_w + 0.25 * quat_apply(self.robot_blade_quat_w, normal_local)
        target_normal_end = self.p_hit_world + 0.25 * self.n_target_world
        racket_vel_end = self.p_hit_world + 0.10 * self.v_racket_hat_world
        base_target = torch.cat(
            (
                self.p_base_xy_world,
                torch.full((self.num_envs, 1), self.cfg.reset_root_pos[2], device=self.device),
            ),
            dim=-1,
        )
        traj_points = self._debug_traj_points[self._debug_traj_valid]
        if traj_points.numel() > 0:
            points = torch.cat(
                (self.p_hit_world, base_target, target_normal_end, racket_vel_end, blade_normal_end, traj_points),
                dim=0,
            )
        else:
            points = torch.cat((self.p_hit_world, base_target, target_normal_end, racket_vel_end, blade_normal_end), dim=0)
        self.target_visualizer.visualize(translations=points)


@configclass
class PingpongCommandCfg(CommandTermCfg):
    class_type: type = PingpongCommand
    asset_name: str = MISSING

    expert_root: str = str(DEFAULT_EXPERT_ROOT)
    forward_motion_file: str = str(DEFAULT_EXPERT_ROOT / "forward" / "forward_001.npz")
    backward_motion_file: str = str(DEFAULT_EXPERT_ROOT / "backward" / "backward_004.npz")

    anchor_body_name: str = "pelvis"
    blade_body_name: str = "right_paddle_blade"
    tracked_body_names: list[str] = [
        "torso_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "left_wrist_roll_rubber_hand",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "right_wrist_roll_rubber_hand",
    ]
    imitation_joint_names: list[str] = [
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
    ]

    hit_x: float = 0.4
    hit_y_range: tuple[float, float] = (0.05, 0.25)
    hit_z_range: tuple[float, float] = (0.95, 1.15)
    v_in_mag_range: tuple[float, float] = (2.0, 4.0)
    target_land: tuple[float, float, float] = (2.45, 0.0, 0.78)
    flight_time_range: tuple[float, float] = (0.30, 0.65)
    paddle_cor: float = 0.85
    paddle_cor_range: tuple[float, float] = (0.80, 0.90)
    y_mid_base: float = 0.157
    swing_dead_zone: float = 0.01
    strike_window: float = 0.06
    t_post_swing_fixed: float = 0.60
    debug_traj_len: int = 8

    reset_root_pos: tuple[float, float, float] = (0.0, 0.0, 0.74)
    reset_yaw_noise: tuple[float, float] = (-math.radians(10.0), math.radians(10.0))

    sigma_g_pos: float = 0.10
    success_pos_floor: float = 0.06
    success_vel_thresh: float = 1.0
    success_ori_cos_dist_thresh: float = 0.25

    noise_p_sigma: float = 0.0
    noise_v_sigma: float = 0.0
    noise_base_sigma: float = 0.0
    noise_t_sigma: float = 0.0
