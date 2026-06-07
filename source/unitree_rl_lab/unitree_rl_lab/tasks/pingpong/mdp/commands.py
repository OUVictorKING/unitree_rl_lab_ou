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
# Forehand face direction in paddle local frame.
# URDF g1_*_rev_1_0_paddle.urdf rotates the paddle by -135° around X at the
# wrist fixed joint, so the URDF's local -Y axis is the forehand-hitting face
# (-Y points forward toward the table at impact for a forehand swing — verified
# from forward_003_rotated.npz at impact_frame=50: -Y in world = +0.652+X).
# Backhand face is local +Y. Forehand reward sign=+1 rewards alignment of
# this BLADE_NORMAL with n_target; backhand sign=-1 rewards -BLADE_NORMAL
# alignment (i.e. +Y face = backhand face). Wrong sign here was the systemic
# bug behind V1 21-04-08's "rigid arm twisted-wrist" pose at iter 8000.
BLADE_NORMAL_LOCAL = (0.0, -1.0, 0.0)


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

        # Per-imitated-joint weight on the imitation reward (joint_pos + joint_vel),
        # indexed in the same order as upper_joint_ids / cfg.imitation_joint_names.
        # Default 1.0 (imitate all equally). The imit_orient_anneal curriculum lowers
        # the entries for waist_yaw + right-arm distal as the policy plateaus, so
        # goal_orientation can recruit those joints to push the face past the demo's
        # ~0.80 cap (the joints are NOT removed from imitation, just down-weighted).
        self.imit_joint_weights = torch.ones(self.upper_joint_ids.numel(), device=self.device)

        # Right-arm joints for the torque-saturation diagnostic. Regex covers
        # both 23dof (shoulder pitch/roll/yaw, elbow, wrist_roll) and 29dof
        # (+ wrist pitch/yaw). _sh_pitch_local = index of shoulder_pitch within.
        _rarm = self.robot.find_joints(
            ["right_shoulder.*_joint", "right_elbow.*_joint", "right_wrist.*_joint"], preserve_order=True
        )
        self.rarm_joint_ids = torch.tensor(_rarm[0], dtype=torch.long, device=self.device)
        self._sh_pitch_local = next((i for i, nm in enumerate(_rarm[1]) if "shoulder_pitch" in nm), 0)

        self.motion = PingpongMotionLoader(
            cfg.forward_motion_file,
            cfg.backward_motion_file,
            cfg.tracked_body_names,
            device=self.device,
        )
        self.expert_offset_base = self.motion.expert_offset_base.to(self.device)

        # Pelvis-frame (base-frame) blade y at impact, taken from
        # clip.expert_offset_base[1] which motion_loader has already yaw-rotated
        # by -pelvis_yaw at impact_frame. Forehand vs backhand are NOT a
        # property of the y sign — they're identified by which clip is loaded
        # under each name. The sign of (forehand_y - backhand_y) is persisted
        # below so the swing classifier and curriculum work for either ordering.
        forehand_y = float(self.motion.clips["forehand"].expert_offset_base[1])
        backhand_y = float(self.motion.clips["backhand"].expert_offset_base[1])
        forehand_x = float(self.motion.clips["forehand"].expert_offset_base[0])
        backhand_x = float(self.motion.clips["backhand"].expert_offset_base[0])

        # Right-arm singularity-avoidance clamp on forehand reach (in the
        # forehand-extension direction). If forehand_y < 0 (clip swings to -y),
        # clamp pulls towards 0; if forehand_y > 0, clamp pulls towards 0.
        # Magnitude of clamp is |cfg.forehand_y_safety_clamp|; set None to keep
        # clip's natural reach.
        if cfg.forehand_y_safety_clamp is not None:
            cap = abs(float(cfg.forehand_y_safety_clamp))
            if forehand_y < 0:
                forehand_y_eff = max(forehand_y, -cap)
            else:
                forehand_y_eff = min(forehand_y, cap)
        else:
            forehand_y_eff = forehand_y

        self._forehand_y_eff = forehand_y_eff
        self._backhand_y_clip = backhand_y

        # Pelvis-frame y bounds for hit-point sampling (curriculum reads these
        # via cfg). Order them low/high so the rest of the code is sign-agnostic.
        if cfg.hit_y_cap_low is None:
            cfg.hit_y_cap_low = min(forehand_y_eff, backhand_y)
        if cfg.hit_y_cap_high is None:
            cfg.hit_y_cap_high = max(forehand_y_eff, backhand_y)

        # Auto-derive y_mid_base from clipped forehand and clip backhand y.
        # Pelvis-frame, not world. Curriculum centers expanding tiers on this.
        if cfg.y_mid_base is None:
            cfg.y_mid_base = 0.5 * (forehand_y_eff + backhand_y)

        # v59 base-frame range auto-derive (replaces v58 hit_y_range).
        # Centered at y_mid_base, NOT clamped to data caps — curriculum's
        # _tier_range can extend up to hit_y_base_max_half_width (0.50).
        # Floor: ensure tier 0 range crosses 0 in base frame so that when
        # base drifts to world cap edge the cap intersect doesn't double-empty.
        # For y_mid_base=-0.188, this bumps initial half_w from 0.10 to 0.238.
        if cfg.hit_y_base_range is None:
            h_user = float(cfg.hit_y_base_initial_half_width)
            h_floor = abs(float(cfg.y_mid_base)) + 0.05
            h = max(h_user, h_floor)
            cfg.hit_y_base_range = (cfg.y_mid_base - h, cfg.y_mid_base + h)

        # Auto-derive command hit_x from clip mean x offset (pelvis-frame).
        if cfg.hit_x is None:
            cfg.hit_x = 0.5 * (forehand_x + backhand_x)

        # Auto-derive t_post_swing_fixed from clip post durations. "max" plays
        # the longer clip's full follow-through, "min" cuts to the shorter to
        # save sim steps; default is max for full-clip imitation coverage.
        if cfg.t_post_swing_fixed is None:
            fp = float(self.motion.clips["forehand"].post_duration)
            bp = float(self.motion.clips["backhand"].post_duration)
            mode = str(cfg.t_post_swing_mode).lower()
            if mode == "min":
                cfg.t_post_swing_fixed = min(fp, bp)
            elif mode == "mean":
                cfg.t_post_swing_fixed = 0.5 * (fp + bp)
            else:
                cfg.t_post_swing_fixed = max(fp, bp)

        self._swing_y_sign = 1.0 if forehand_y > backhand_y else -1.0

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

        # v61: per-swing diagnostic captures at strike instant. Updated each
        # step inside _update_success_window when in_window=True. Used to
        # detect "forehand command but actually backhand stroke" cheat.
        self._paddle_y_base_at_strike = torch.zeros(n, device=self.device)
        self._cos_sim_at_strike = torch.zeros(n, device=self.device)

        # Torque-saturation diagnostic (right-arm joints). Per-env running max of
        # |applied_torque|/effort_limit over the *current* swing, plus the captured
        # peak of the *last completed* swing (the value reported, split fh/bh).
        # Captures the swing peak (acceleration phase), not just the near-contact
        # frame. sat ~1.0 => joint hit its torque limit (torque-limited);
        # sat far below 1 with a weak strike => kinematic / action_scale / policy.
        self._rarm_sat_max = torch.zeros(n, device=self.device)
        self._shp_sat_max = torch.zeros(n, device=self.device)
        self._rarm_sat_last = torch.zeros(n, device=self.device)
        self._shp_sat_last = torch.zeros(n, device=self.device)

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
            # v61: per-swing diagnostics (forehand vs backhand split)
            "hsr_forehand_only",
            "hsr_backhand_only",
            "cos_sim_forehand_only",
            "cos_sim_backhand_only",
            "paddle_y_base_at_strike_forehand",
            "paddle_y_base_at_strike_backhand",
            # v64: strike-instant face cos (split) — true face at contact, not the
            # deflated current-frame cos_sim_*_only used for the curriculum gates.
            "cos_sim_at_strike_forehand",
            "cos_sim_at_strike_backhand",
            "cos_sim_at_strike",  # per-env combined; curriculum gates on THIS (true strike face), not deflated current-frame
            # v63: torque-saturation diagnostic (peak |tau|/effort over last swing)
            "rarm_torque_sat_forehand",
            "rarm_torque_sat_backhand",
            "shoulder_pitch_sat_forehand",
            "shoulder_pitch_sat_backhand",
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
        self._update_torque_sat()
        self._update_success_window()
        self._refresh_metrics_from_counts()

    def _update_torque_sat(self) -> None:
        """Running per-env max of |applied_torque|/effort_limit over the right-arm
        joints, for the current swing. applied_torque is the torque actually applied
        by the sim (clamped to the effort limit), so sat in [0, 1]; sat -> 1 means
        the joint saturated. Tracking the max over the whole swing captures the
        acceleration peak (which precedes contact), not just the near-contact frame."""
        tau = self.robot.data.applied_torque[:, self.rarm_joint_ids].abs()
        lim = self.robot.data.joint_effort_limits
        lim = lim[:, self.rarm_joint_ids] if lim.dim() == 2 else lim[self.rarm_joint_ids]
        sat = tau / lim.clamp_min(1e-6)  # (n, n_rarm)
        self._rarm_sat_max = torch.maximum(self._rarm_sat_max, sat.amax(dim=-1))
        self._shp_sat_max = torch.maximum(self._shp_sat_max, sat[:, self._sh_pitch_local])

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

        # v61: per-swing diagnostics — split metrics by commanded swing_type.
        # Mean computed only over envs of that swing type, then broadcast so
        # IsaacLab's mean-over-envs aggregation gives back the correct per-swing
        # mean. If no envs have that swing type this iter, defaults to 0.
        hsr_per_env = self._success_count / denom
        cos_per_env = self._blade_target_cosine()  # signed by swing_type
        paddle_y_per_env = self._paddle_y_base_at_strike

        fh_mask = (self.swing_type == SWING_FOREHAND).float()
        bh_mask = (self.swing_type == SWING_BACKHAND).float()
        fh_count = fh_mask.sum().clamp_min(1.0)
        bh_count = bh_mask.sum().clamp_min(1.0)

        hsr_fh = (hsr_per_env * fh_mask).sum() / fh_count
        hsr_bh = (hsr_per_env * bh_mask).sum() / bh_count
        cos_fh = (cos_per_env * fh_mask).sum() / fh_count
        cos_bh = (cos_per_env * bh_mask).sum() / bh_count
        py_fh = (paddle_y_per_env * fh_mask).sum() / fh_count
        py_bh = (paddle_y_per_env * bh_mask).sum() / bh_count

        # Broadcast scalar means to per-env tensors so IsaacLab logging mean
        # over envs returns the correct value.
        self.metrics["hsr_forehand_only"] = hsr_fh.expand(self.num_envs).clone()
        self.metrics["hsr_backhand_only"] = hsr_bh.expand(self.num_envs).clone()
        self.metrics["cos_sim_forehand_only"] = cos_fh.expand(self.num_envs).clone()
        self.metrics["cos_sim_backhand_only"] = cos_bh.expand(self.num_envs).clone()
        self.metrics["paddle_y_base_at_strike_forehand"] = py_fh.expand(self.num_envs).clone()
        self.metrics["paddle_y_base_at_strike_backhand"] = py_bh.expand(self.num_envs).clone()

        # v64: strike-instant signed-cos, split. _cos_sim_at_strike holds the cos
        # at the most recent strike (persists across the swing), so this is the
        # true contact-face quality — distinct from cos_sim_*_only (current-frame).
        cos_strike_fh = (self._cos_sim_at_strike * fh_mask).sum() / fh_count
        cos_strike_bh = (self._cos_sim_at_strike * bh_mask).sum() / bh_count
        self.metrics["cos_sim_at_strike_forehand"] = cos_strike_fh.expand(self.num_envs).clone()
        self.metrics["cos_sim_at_strike_backhand"] = cos_strike_bh.expand(self.num_envs).clone()
        # per-env combined strike-instant signed cos — the curriculum's cos_sim_ema
        # gate reads THIS (true contact face ~0.8) instead of the deflated current-
        # frame cos (~0.46) that was stalling Stage-2 unlock / flickering the freeze.
        self.metrics["cos_sim_at_strike"] = self._cos_sim_at_strike.clone()

        # v63: torque-saturation — peak |tau|/effort over the last completed swing.
        sat_fh = (self._rarm_sat_last * fh_mask).sum() / fh_count
        sat_bh = (self._rarm_sat_last * bh_mask).sum() / bh_count
        shp_fh = (self._shp_sat_last * fh_mask).sum() / fh_count
        shp_bh = (self._shp_sat_last * bh_mask).sum() / bh_count
        self.metrics["rarm_torque_sat_forehand"] = sat_fh.expand(self.num_envs).clone()
        self.metrics["rarm_torque_sat_backhand"] = sat_bh.expand(self.num_envs).clone()
        self.metrics["shoulder_pitch_sat_forehand"] = shp_fh.expand(self.num_envs).clone()
        self.metrics["shoulder_pitch_sat_backhand"] = shp_bh.expand(self.num_envs).clone()

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

        # v59: swing_change_remaining flip block removed. swing_type is now a
        # task-input determined at sample time (Bernoulli + boundary override),
        # not a fact-classification that changes mid-swing as base drifts.
        # Field self.swing_change_remaining kept (always 0) for backward-compat
        # metric "swing_change_remaining_used_rate".

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
        n = len(ids)

        # ═══ Step 1 (v59): yaw-independent samples (swing_target, hit_z, ball/target) ═══
        # swing_target is paper Table I "task input" — Bernoulli(0.50) fixed
        # (v61: removed v60 swing_p_forehand_warmup curriculum; 3-phase task
        # curriculum handles single-task → dual-task progression instead).
        p_fh = float(self.cfg.swing_p_forehand)
        swing_target = (torch.rand(n, device=self.device) >= p_fh).long()
        # CRITICAL: set self.swing_type EARLY (before RSI) so that
        # _sample_rsi_frames / _write_rsi_joint_state read consistent clip
        # selection. Boundary override (Step 5) may further mutate swing_target
        # locally; we re-write self.swing_type at Step 5b to reflect overrides.
        # If reset_robot=True, base is at env_origin so boundary override
        # cannot trigger here, keeping rsi_frames consistent with final swing_type.
        self.swing_type[ids] = swing_target
        hit_z = sample_uniform(self.cfg.hit_z_range[0], self.cfg.hit_z_range[1], (n,), device=self.device)

        v_mag = sample_uniform(self.cfg.v_in_mag_range[0], self.cfg.v_in_mag_range[1], (n,), device=self.device)
        v_yaw = math.pi + sample_uniform(-math.radians(40.0), math.radians(40.0), (n,), device=self.device)
        v_pitch = sample_uniform(-math.radians(75.0), math.radians(75.0), (n,), device=self.device)
        self.v_ball_in_world[ids] = v_mag.unsqueeze(-1) * torch.stack(
            (torch.cos(v_yaw) * torch.cos(v_pitch), torch.sin(v_yaw) * torch.cos(v_pitch), torch.sin(v_pitch)), dim=-1
        )

        local_target = torch.tensor(self.cfg.target_land, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.target_land_world[ids] = env_origins + local_target
        self.flight_time[ids] = sample_uniform(
            self.cfg.flight_time_range[0], self.cfg.flight_time_range[1], (n,), device=self.device
        )
        self.paddle_cor[ids] = sample_uniform(
            self.cfg.paddle_cor_range[0], self.cfg.paddle_cor_range[1], (n,), device=self.device
        )

        # ═══ Step 2: RSI override root_quat (if reset) ═════════════════════
        # Done BEFORE hit_y_world computation so the conversion uses the final
        # yaw the policy will see at episode start (RSI matches clip pelvis_yaw).
        rsi_frames: torch.Tensor | None = None
        if reset_robot and not self.cfg.disable_rsi:
            rsi_frames, pelvis_yaws = self._sample_rsi_frames(ids)
            yaw_noise = sample_uniform(
                self.cfg.reset_yaw_noise[0], self.cfg.reset_yaw_noise[1], (n,), device=self.device
            )
            final_yaw = pelvis_yaws + yaw_noise
            zeros = torch.zeros_like(final_yaw)
            new_root_quat = quat_from_euler_xyz(zeros, zeros, final_yaw)
            root_lin = torch.zeros(n, 3, device=self.device)
            root_ang = torch.zeros(n, 3, device=self.device)
            self.robot.write_root_state_to_sim(
                torch.cat((root_pos, new_root_quat, root_lin, root_ang), dim=-1), env_ids=ids
            )
            root_quat = new_root_quat

        # ═══ Step 3-7 (v60): WORLD-FRAME sampling with base-position divider ══
        # Geometry:
        #   - World y range (env-local cap): hit_y_world ∈ [env.y - cap, env.y + cap]
        #   - Divider in world (per-env, depends on root pose):
        #       divider_world = root.y + (y_mid_base + sin*diff_x_world)/cos
        #     This is the world y where the BASE-frame line "y_base = y_mid_base"
        #     crosses the fixed world hit_x = env.x + cfg.hit_x.
        #   - Forehand world range: side of divider matching the forehand demo's
        #     y_mid_base relationship (controlled by _swing_y_sign).
        #   - Backhand world range: the other side.
        #
        # Why this is structurally clean:
        #   * X formula:  hit_x_world = env_origins[:, 0] + cfg.hit_x      ← per-env
        #   * Y formula:  hit_y_world = env_origins[:, 1] + sample_in_cap  ← per-env (NEW)
        #   Both use env_origins explicitly. No path that uses root_pos[:, 1] as
        #   a baseline (which broke v59 — cap was treated as absolute world).
        yaw = yaw_from_wxyz(root_quat)
        cos_y = torch.cos(yaw)
        sin_y = torch.sin(yaw)
        cos_y_abs_small = cos_y.abs() < 0.05
        cos_y_safe = torch.where(
            cos_y_abs_small,
            torch.where(cos_y >= 0, torch.full_like(cos_y, 0.05), torch.full_like(cos_y, -0.05)),
            cos_y,
        )

        hit_x_world = env_origins[:, 0] + self.cfg.hit_x  # fixed per env
        diff_x_world = hit_x_world - root_pos[:, 0]

        # World cap range (env-local): hit_y_world bounded ENV-LOCAL by ±cap
        cap = float(self.cfg.hit_y_world_cap)
        world_y_lo = env_origins[:, 1] - cap
        world_y_hi = env_origins[:, 1] + cap

        # Divider in world: ANCHORED at env_origin (v61 fix). The previous v60
        # design `divider = root.y + ...` let policy MOVE BASE to redefine
        # forehand vs backhand region — confirmed by user sim observation:
        # forehand-commanded swings became cross-body backhand-style strokes
        # because base drift made forehand region span world-y left of body.
        # Anchoring divider at env_origin closes this geometric cheat: forehand
        # is always world.y < env.y + y_mid_base regardless of base drift.
        # Robot must stay near env_origin to satisfy forehand commands; goal_base
        # naturally pulls it back. y_mid_base added directly (not yaw-rotated)
        # because hit point is at fixed world coordinates, not robot-relative.
        y_mid = float(self.cfg.y_mid_base)
        divider_world = env_origins[:, 1] + y_mid

        # Per-swing-target world range:
        #   _swing_y_sign = +1 → forehand on +y in base frame (forehand_y > backhand_y)
        #   _swing_y_sign = -1 → forehand on -y in base frame (forehand_y < backhand_y)
        # In world: forehand on +y / -y of divider correspondingly.
        if self._swing_y_sign > 0:
            # Forehand: world y > divider
            fh_lo_eff = torch.maximum(divider_world, world_y_lo)
            fh_hi_eff = world_y_hi.clone()
            bh_lo_eff = world_y_lo.clone()
            bh_hi_eff = torch.minimum(divider_world, world_y_hi)
        else:
            # Forehand: world y < divider (current 23dof case: forehand_y=-0.40)
            fh_lo_eff = world_y_lo.clone()
            fh_hi_eff = torch.minimum(divider_world, world_y_hi)
            bh_lo_eff = torch.maximum(divider_world, world_y_lo)
            bh_hi_eff = world_y_hi.clone()

        # Validate ranges (boundary effect: divider outside cap → one half empty)
        fh_valid = fh_hi_eff > fh_lo_eff + 1e-4
        bh_valid = bh_hi_eff > bh_lo_eff + 1e-4

        # ═══ Step 5: boundary OVERRIDE swing_target ═══════════════════════
        # If desired half has empty intersection, force the other half (which
        # then drives goal_base_position to pull robot back into bounds).
        is_forehand_target = swing_target == SWING_FOREHAND
        force_to_bh = is_forehand_target & (~fh_valid) & bh_valid
        force_to_fh = (~is_forehand_target) & (~bh_valid) & fh_valid
        swing_target = torch.where(
            force_to_bh, torch.full_like(swing_target, SWING_BACKHAND), swing_target
        )
        swing_target = torch.where(
            force_to_fh, torch.full_like(swing_target, SWING_FOREHAND), swing_target
        )
        is_forehand_target = swing_target == SWING_FOREHAND
        force_overridden = force_to_bh | force_to_fh
        self._dead_zone_count[ids] += force_overridden.float()

        # Step 5b: Re-write self.swing_type to reflect any boundary overrides.
        # At reset (reset_robot=True), boundary cannot trigger because base is at
        # env_origin (root.y = env.y → divider = env.y + y_mid_base, well within
        # world cap), so this is a no-op for reset envs AND keeps RSI consistent.
        self.swing_type[ids] = swing_target

        # Extreme fallback: both halves invalid (rare, would require base drift
        # > cap from env_origin AND yaw skew). Defensively clamp to env_origin.
        both_invalid = (~fh_valid) & (~bh_valid)
        if torch.any(both_invalid):
            fallback_lo = env_origins[:, 1].clone()
            fallback_hi = env_origins[:, 1] + 1e-3
            fh_lo_eff = torch.where(both_invalid, fallback_lo, fh_lo_eff)
            fh_hi_eff = torch.where(both_invalid, fallback_hi, fh_hi_eff)
            bh_lo_eff = torch.where(both_invalid, fallback_lo, bh_lo_eff)
            bh_hi_eff = torch.where(both_invalid, fallback_hi, bh_hi_eff)

        # ═══ Step 6: sample hit_y_world directly from valid swing_target range ═
        rand_fh = torch.rand(n, device=self.device)
        rand_bh = torch.rand(n, device=self.device)
        fh_y_world = rand_fh * (fh_hi_eff - fh_lo_eff) + fh_lo_eff
        bh_y_world = rand_bh * (bh_hi_eff - bh_lo_eff) + bh_lo_eff
        hit_y_world = torch.where(is_forehand_target, fh_y_world, bh_y_world)

        # ═══ Step 7 (v60): hit_y_base for diagnostic only — derived FROM world ═
        # base-frame y of the sampled world hit point (for metric/debug use).
        # Forward transform (same as _compute_swing_type):
        #   hit_y_base = -sin*diff_x_world + cos*(hit_y_world - root.y)
        hit_y_base = -sin_y * diff_x_world + cos_y * (hit_y_world - root_pos[:, 1])

        # ═══ Step 8: write p_hit_world (absolute world frame) ══════════════
        p_hit_world_new = torch.stack(
            (hit_x_world, hit_y_world, env_origins[:, 2] + hit_z), dim=-1
        )
        self.p_hit_world[ids] = p_hit_world_new

        # ═══ Step 9: solve Eq.5/Eq.6 (depends on p_hit_world) ══════════════
        self._solve_paddle_target(ids)

        # ═══ Step 10 (v59): swing_type already set at Step 1 + Step 5b ═════
        # Construct guarantee: hit_y_base ∈ swing_target's half by sampling.
        # No post-hoc _compute_swing_type — label is by design.
        self.hit_y_base[ids] = hit_y_base
        self.swing_change_remaining[ids] = 0
        # _dead_zone_count repurposed above to track boundary overrides.

        # ═══ Step 11: base target + time fields (unchanged from v58) ════════
        self.p_base_xy_world[ids] = self._compute_base_target(ids, root_quat)
        self.t_pre_initial[ids] = _sample_peak_uniform(0.20, 0.90, 0.30, 0.65, (n,), self.device)
        self.t_post_swing[ids] = float(self.cfg.t_post_swing_fixed)
        self.t_to_hit[ids] = self.t_pre_initial[ids]
        self.cur_step[ids] = 0

        if reset_robot and not self.cfg.disable_rsi:
            self._write_rsi_joint_state(ids, frames=rsi_frames)

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
        is_forehand = (hit_y_base - self.cfg.y_mid_base) * self._swing_y_sign > 0
        swing = torch.where(is_forehand, SWING_FOREHAND, SWING_BACKHAND).long()
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

    def _sample_rsi_frames(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Pre-sample one clip frame per env (forehand or backhand based on
        # current swing_type) and look up its pelvis yaw. Used by the RSI
        # base-yaw override and also passed back into _write_rsi_joint_state
        # to guarantee root_quat and joint_pos come from the same frame.
        frames = torch.zeros(len(ids), dtype=torch.long, device=self.device)
        pelvis_yaws = torch.zeros(len(ids), device=self.device)
        for swing_idx, name in ((SWING_FOREHAND, "forehand"), (SWING_BACKHAND, "backhand")):
            sub_ids = torch.nonzero(self.swing_type[ids] == swing_idx, as_tuple=False).flatten()
            if len(sub_ids) == 0:
                continue
            clip = self.motion.clips[name]
            sub_frames = torch.randint(0, clip.length, (len(sub_ids),), device=self.device)
            frames[sub_ids] = sub_frames
            pelvis_yaws[sub_ids] = clip.pelvis_yaw_at_frame(sub_frames)
        return frames, pelvis_yaws

    def _write_rsi_joint_state(self, ids: torch.Tensor, frames: torch.Tensor | None = None) -> None:
        if frames is None:
            frames, _ = self._sample_rsi_frames(ids)
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
        """Signed paddle-alignment: sign(swing) * (n_blade . n_target).

        Forehand swing (sign=+1) requires the front face (+n_blade) to point
        toward n_target; backhand swing (sign=-1) requires the back face
        (-n_blade). This is the physically correct definition — the active
        face is fixed by swing_type at sample time. The earlier symmetric
        |dot| version killed the swing-discovery gradient: ori_ok was
        trivially satisfiable at any orientation, so the policy parked the
        paddle and never swung through the strike volume, leaving pos_fail
        at ~0.99 (run 2026-05-24_22-46-22). Returns in [-1, 1]; drives the
        cos_sim_n_blade_n_target_at_impact metric and the ori_ok success
        check (via _task_errors), so reward / metric / success agree.
        """
        normal_local = torch.tensor(BLADE_NORMAL_LOCAL, dtype=torch.float32, device=self.device).expand(self.num_envs, 3)
        n_blade = quat_apply(self.robot_blade_quat_w, normal_local)
        sign = 1.0 - 2.0 * self.swing_type.float()
        return (sign * torch.sum(n_blade * self.n_target_world, dim=-1)).clamp(-1.0, 1.0)

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
        pos_thresh = torch.full_like(pos_err, self.cfg.success_pos_thresh)
        pos_ok = pos_err < pos_thresh
        vel_ok = vel_err < self.cfg.success_vel_thresh
        # v64: swing-split ori bar. Backhand succeeds with a mediocre face under the
        # shared 0.25 bar, so it has no pressure to align (and regresses once imit_w
        # drops). A tighter backhand bar makes a good face a precondition for success.
        ori_thresh = torch.where(
            self.swing_type == SWING_BACKHAND,
            self.swing_type.new_full((), self.cfg.success_ori_cos_dist_thresh_backhand, dtype=torch.float),
            self.swing_type.new_full((), self.cfg.success_ori_cos_dist_thresh, dtype=torch.float),
        )
        ori_ok = ori_dist < ori_thresh
        blade_hit_dist = torch.linalg.norm(self.robot_blade_pos_w - self.p_hit_world, dim=-1)
        self._strike_dist_min = torch.where(in_window, torch.minimum(self._strike_dist_min, blade_hit_dist), self._strike_dist_min)

        # v61: capture per-swing diagnostics at strike instant.
        # paddle_y_base = quat_apply_inverse(root_quat, paddle - pelvis)[1]
        # signed cos_sim = sign(swing_type) * (n_blade · n_target)
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        paddle_rel_world = self.robot_blade_pos_w - self.robot_pelvis_pos_w
        paddle_in_base = quat_apply_inverse(root_quat_w, paddle_rel_world)
        paddle_y_base = paddle_in_base[:, 1]
        cos_now = self._blade_target_cosine()
        self._paddle_y_base_at_strike = torch.where(in_window, paddle_y_base, self._paddle_y_base_at_strike)
        self._cos_sim_at_strike = torch.where(in_window, cos_now, self._cos_sim_at_strike)

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
        # capture the swing's torque-saturation peak before _reset_window_flags clears it
        self._rarm_sat_last[done] = self._rarm_sat_max[done]
        self._shp_sat_last[done] = self._shp_sat_max[done]
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
        self._rarm_sat_max[ids] = 0.0
        self._shp_sat_max[ids] = 0.0

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
    forward_motion_file: str = str(DEFAULT_EXPERT_ROOT / "new_3" / "forward" / "npz" / "forward_001_wristfix_rotated.npz")
    backward_motion_file: str = str(DEFAULT_EXPERT_ROOT / "new_3" / "backward" / "npz" / "backward_001_rotated.npz")

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
        # right_shoulder_yaw_link / right_elbow_link / right_wrist_roll_rubber_hand
        # removed in sync with imitation_joint_names: full right-arm freedom for
        # paddle-orientation control. Only right_shoulder_{pitch,roll} retained
        # to anchor the gross arm root and prevent shoulder drift.
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
        # right_shoulder_yaw + right_elbow freed for paddle-orientation control;
        # right_wrist_roll already excluded. Body-level tracking of the right arm
        # is also disabled in tracked_body_names above (sync'd 2026-05-23 imsmall_window).
    ]

    # === Hit-point geometry (auto-derived from expert clips at __init__) ===
    # Set any of these to None to auto-derive from the loaded clips' pelvis-
    # frame impact-pose offsets; explicit values override. Switching expert
    # npz files therefore only requires changing forward_motion_file /
    # backward_motion_file — y_mid_base, hit_y_base_range, hit_x, t_post_swing_fixed,
    # hit_y_cap_low/high all reshape automatically.
    #
    # hit_x: pelvis-frame x of the commanded ball point. Default = mean of
    # clip x offsets at impact.
    hit_x: float | None = None

    # === v59 base-frame swing-first sampling (replaces hit_y_range) ===
    # hit_y_base_range: BASE-frame y range, centered at y_mid_base. Auto-derived
    # in __init__ from y_mid_base ± hit_y_base_initial_half_width if None.
    # Curriculum drives this through _tier_range(half_w) — half_w grows over
    # training, NOT clamped to demo data caps.
    hit_y_base_range: tuple[float, float] | None = None

    # Initial half-width (curriculum tier 0). Demo's intrinsic half-range is
    # ~0.05 (forehand_y +0.18 vs backhand_y +0.10, half-spread = 0.05), but
    # tier 0 lets policy learn the full demo span first.
    hit_y_base_initial_half_width: float = 0.10

    # Max half-width that the curriculum can extend to (top tier).
    # Practical span: y_mid ± 0.50 ≈ [-0.343, +0.657] base-frame, total 1.0m.
    hit_y_base_max_half_width: float = 0.50

    # World-frame absolute cap on |hit_y_world - env_origin.y| (env-local).
    # v60: cap is now ENV-LOCAL (relative to env_origin) — fixes the v59 bug
    # where cap was treated as absolute world coordinate, breaking envs at
    # non-zero env_origin (the env grid). Sampling restricts world-y to:
    #   hit_y_world ∈ [env_origin.y - cap, env_origin.y + cap]
    # Curriculum drives this value: starts at hit_y_world_cap_initial (narrow,
    # but wide enough to cover BOTH demo hit positions: forehand_y_base=-0.40,
    # backhand_y_base=+0.024), grows to hit_y_world_cap_max as success rises.
    # IMPORTANT: cap_initial must be > max(|forehand_y_base|, |backhand_y_base|)
    # = 0.40 + safety margin, else sampled hit can't reach demo positions and
    # imit ref pose desyncs from sampled hit point (policy can't learn).
    hit_y_world_cap: float = 0.45  # current value, mutated by curriculum
    hit_y_world_cap_initial: float = 0.45  # tier 0 (covers ±0.40 demo + 5cm margin)
    hit_y_world_cap_max: float = 1.00  # tier 4 (paper-aligned widest)

    # Bernoulli p(forehand) for swing_target sampling. Fixed at 0.50 (paper
    # design). The v60 swing_p_forehand_warmup curriculum (90:10 → 50:50) was
    # removed in v61 because the 3-phase task curriculum (stand/imit/strike)
    # already handles single-task → dual-task progression more cleanly.
    swing_p_forehand: float = 0.50

    # Right-arm singularity-avoidance cap on forehand reach. Magnitude only;
    # sign is taken from forehand_y direction. Set None to use clip's natural
    # forehand_y. Acts in pelvis frame.
    forehand_y_safety_clamp: float | None = 0.40

    # Pelvis-frame caps on hit-point y (low/high in numerical order, sign-
    # agnostic). Auto-set in __init__ from forehand_y_eff and backhand_y.
    # v59 NOTE: kept for legacy / sanity check (validates demo data range)
    # but NO LONGER used to clamp curriculum range. Curriculum's _tier_range
    # extends past these caps freely up to hit_y_base_max_half_width.
    hit_y_cap_low: float | None = None
    hit_y_cap_high: float | None = None

    # Upper bound raised 1.15 → 1.25 to match the new_new clips' high contact
    # point (forehand impact z≈1.16, backhand z≈1.26). With 1.15 the demo hit
    # height sat above the commandable range, so every command forced the policy
    # below the demo. 1.25 also matches the z-curriculum's max tier (consistent).
    hit_z_range: tuple[float, float] = (0.95, 1.25)
    v_in_mag_range: tuple[float, float] = (1.5, 2.0)
    target_land: tuple[float, float, float] = (2.45, 0.0, 0.78)
    flight_time_range: tuple[float, float] = (0.30, 0.65)
    paddle_cor: float = 0.85
    paddle_cor_range: tuple[float, float] = (0.80, 0.90)

    # Swing-classification midpoint in pelvis frame. Default auto-derived as
    # 0.5 * (forehand_y_eff + backhand_y). Sign of (hit_y_base - y_mid_base)
    # is interpreted via _swing_y_sign (set in __init__) so the classifier
    # works for either forehand_y > backhand_y or forehand_y < backhand_y.
    y_mid_base: float | None = None
    swing_dead_zone: float = 0.01
    strike_window: float = 0.1

    # Episode follow-through duration after impact. None auto-derives from
    # clip post_durations using t_post_swing_mode: "max" plays the longer
    # clip's full follow-through, "min" cuts to the shorter, "mean" averages.
    t_post_swing_fixed: float | None = None
    t_post_swing_mode: str = "max"

    debug_traj_len: int = 8

    reset_root_pos: tuple[float, float, float] = (0.0, 0.0, 0.74)
    reset_yaw_noise: tuple[float, float] = (-math.radians(10.0), math.radians(10.0))

    # Plan C: skip Reference State Initialization (do not overwrite root_quat
    # or joint_pos from clip at episode reset). Robot starts from default_joint_pos
    # and identity yaw + reset_yaw_noise. Used to isolate whether 29dof EpLen=1
    # collapse is caused by clip-induced initial pose vs env-side issues
    # (PD gains, contact regex, termination thresholds).
    disable_rsi: bool = False

    sigma_g_pos: float = 0.30
    success_pos_floor: float = 0.06
    # Decoupled success criterion: pos_ok now uses a fixed threshold rather
    # than tracking 2*sigma_g_pos. Coupled-mode goalposts moved as the shape
    # curriculum tightened sigma (run 2026-05-25_22-50-41 iter 1707-3273:
    # pos_thresh shrank 0.60→0.40→0.30m, pos_fail "jumped" 0.30→0.44 even
    # though strike_dist_min was monotonically improving 0.44→0.21m). Fixed
    # threshold lets sigma anneal for reward-shape sharpness without
    # corrupting the success metric the curriculum gates on.
    success_pos_thresh: float = 0.15
    success_vel_thresh: float = 1.0
    success_ori_cos_dist_thresh: float = 0.25
    # v64: backhand-specific ori bar (ori_dist = 1 - signed_cos). Defaults to the
    # shared value (no behaviour change); set tighter in env_cfg so backhand
    # success requires a crisper face. Tightening lowers hsr_backhand → hsr_ema,
    # which the shape/window/phase curricula gate on — keep the change modest.
    success_ori_cos_dist_thresh_backhand: float = 0.25

    noise_p_sigma: float = 0.0
    noise_v_sigma: float = 0.0
    noise_base_sigma: float = 0.0
    noise_t_sigma: float = 0.0
