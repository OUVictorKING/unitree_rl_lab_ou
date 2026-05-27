from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
try:
    from isaaclab.utils.math import quat_apply, sample_uniform
except ImportError:
    from isaaclab.utils.math import quat_apply, sample_uniform

from .commands import BLADE_NORMAL_LOCAL, PingpongCommand, PingpongCommandCfg, _as_env_ids
from .planner_for_training import PLAN_FROZEN, PLAN_FRESH, plan_pingpong_hits

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _quat_from_x_axis(direction: torch.Tensor) -> torch.Tensor:
    """Return wxyz quaternions that rotate the marker +X axis onto each world direction."""
    norm = torch.linalg.norm(direction, dim=-1, keepdim=True)
    unit = direction / norm.clamp_min(1.0e-6)
    default_dir = torch.zeros_like(unit)
    default_dir[:, 0] = 1.0
    unit = torch.where(norm > 1.0e-6, unit, default_dir)

    dot = unit[:, 0].clamp(-1.0, 1.0)
    quat = torch.zeros(direction.shape[0], 4, dtype=direction.dtype, device=direction.device)
    quat[:, 0] = 1.0 + dot
    quat[:, 2] = -unit[:, 2]
    quat[:, 3] = unit[:, 1]

    opposite = dot < -0.9999
    if torch.any(opposite):
        quat[opposite] = torch.tensor((0.0, 0.0, 0.0, 1.0), dtype=direction.dtype, device=direction.device)
    small = norm.squeeze(-1) <= 1.0e-6
    if torch.any(small):
        quat[small] = torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=direction.dtype, device=direction.device)
    return quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(1.0e-6)


def _arrow_scales(
    magnitude: torch.Tensor,
    base_scale: tuple[float, float, float],
    gain: float,
    min_x: float,
    max_x: float,
) -> torch.Tensor:
    scale = torch.tensor(base_scale, dtype=magnitude.dtype, device=magnitude.device).repeat(magnitude.shape[0], 1)
    scale[:, 0] *= torch.clamp(magnitude * gain, min=min_x, max=max_x)
    return scale


class RealPingpongCommand(PingpongCommand):
    """Pingpong command driven by a simulated ball and a batched training planner."""

    cfg: "RealPingpongCommandCfg"

    def __init__(self, cfg: "RealPingpongCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.ball: RigidObject = env.scene[cfg.ball_name]

        n = self.num_envs
        self.planner_valid = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.plan_mode = torch.zeros(n, dtype=torch.long, device=self.device)
        self.active_swing = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.target_frozen = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.legal_contact = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.illegal_contact = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.return_direction = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.clear_net = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.opponent_land = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.target_land_success = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.planner_success = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.first_post_hit_bounce_done = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.net_cross_checked = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.missed_swing = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.outcome_hold_active = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.outcome_hold_time_left = torch.zeros(n, device=self.device)

        self.actual_contact_pos_world = torch.zeros(n, 3, device=self.device)
        self.actual_contact_time_err = torch.zeros(n, device=self.device)
        self.actual_contact_pos_err = torch.zeros(n, device=self.device)
        self.net_clearance = torch.zeros(n, device=self.device)
        self.landing_error = torch.zeros(n, device=self.device)
        self.landing_pos_world = torch.zeros(n, 3, device=self.device)
        self.landing_pos_valid = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.prev_ball_pos_world = torch.zeros(n, 3, device=self.device)
        self.prev_ball_vel_world = torch.zeros(n, 3, device=self.device)
        self._planner_traj_points = torch.zeros(n, cfg.debug_planner_traj_len, 3, device=self.device)
        self._planner_traj_valid = torch.zeros(n, cfg.debug_planner_traj_len, dtype=torch.bool, device=self.device)
        self._ball_traj_points = torch.zeros(n, cfg.debug_ball_traj_len, 3, device=self.device)
        self._ball_traj_valid = torch.zeros(n, cfg.debug_ball_traj_len, dtype=torch.bool, device=self.device)
        self._ball_traj_cursor = torch.zeros(n, dtype=torch.long, device=self.device)

        self._real_swing_count = torch.zeros(n, device=self.device)
        self._real_hit_count = torch.zeros(n, device=self.device)
        self._real_return_count = torch.zeros(n, device=self.device)
        self._real_target_count = torch.zeros(n, device=self.device)
        self._real_planner_success_count = torch.zeros(n, device=self.device)
        self._real_illegal_count = torch.zeros(n, device=self.device)
        self._real_net_contact_count = torch.zeros(n, device=self.device)
        self._post_recovery_interrupt_count = torch.zeros(n, device=self.device)

        self._reward_ball_contact = torch.zeros(n, device=self.device)
        self._reward_return_direction = torch.zeros(n, device=self.device)
        self._reward_clear_net = torch.zeros(n, device=self.device)
        self._reward_opponent_land = torch.zeros(n, device=self.device)
        self._reward_target_land = torch.zeros(n, device=self.device)
        self._reward_illegal = torch.zeros(n, device=self.device)

        for name in (
            "real_hit_success_rate",
            "real_return_success_rate",
            "real_target_success_rate",
            "real_planner_success_rate",
            "real_planner_valid_rate",
            "real_planner_time_error_mean",
            "real_planner_pos_error_mean",
            "real_net_clearance_mean",
            "real_landing_error_mean",
            "real_illegal_contact_rate",
            "real_ball_net_contact_rate",
            "real_post_recovery_interrupt_rate",
        ):
            self.metrics[name] = torch.zeros(n, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        ids = _as_env_ids(env_ids, self.num_envs, self.device)
        if ids.numel() == 0:
            return
        self._reset_counters(ids)
        self._reset_real_counters(ids)
        root_pos, root_quat = self._write_nominal_root(ids)
        self._serve_ball(ids)
        self._reset_real_swing_state(ids)
        self._start_or_update_plan(ids, update_timing=True, force=True, root_pos_override=root_pos, root_quat_override=root_quat)
        if torch.any(~self.planner_valid[ids]):
            invalid = ids[~self.planner_valid[ids]]
            self._set_fallback_command(invalid, root_pos_override=root_pos[~self.planner_valid[ids]], root_quat_override=root_quat[~self.planner_valid[ids]])
        self._write_rsi_joint_state(ids)
        self._freeze_noise(ids)
        self._update_ref_state(ids)

    def finalize_partial_swings(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        ids = _as_env_ids(slice(None) if env_ids is None else env_ids, self.num_envs, self.device)
        self._update_success_window()
        active = ids[self.active_swing[ids]]
        if len(active) > 0:
            self._complete_real_swing(active)
        self._refresh_metrics_from_counts()

    def _update_command(self):
        self._clear_reward_pulses()

        active = torch.nonzero(self.active_swing, as_tuple=False).flatten()
        if len(active) > 0:
            self.t_to_hit[active] -= self.dt
            self.cur_step[active] += 1
        holding = torch.nonzero(self.active_swing & self.outcome_hold_active, as_tuple=False).flatten()
        if len(holding) > 0:
            self.outcome_hold_time_left[holding] = torch.clamp(
                self.outcome_hold_time_left[holding] - self.dt,
                min=0.0,
            )

        self._record_ball_debug_traj()

        no_active = torch.nonzero(~self.active_swing, as_tuple=False).flatten()
        if len(no_active) > 0:
            self._start_or_update_plan(no_active, update_timing=True, force=False)

        pre_update = torch.nonzero(self.active_swing & (self.t_to_hit > self.cfg.freeze_time_before_hit), as_tuple=False).flatten()
        if len(pre_update) > 0:
            self._start_or_update_plan(pre_update, update_timing=False, force=False)

        freeze_ids = torch.nonzero(self.active_swing & (~self.target_frozen) & (self.t_to_hit <= self.cfg.freeze_time_before_hit), as_tuple=False).flatten()
        if len(freeze_ids) > 0:
            self.target_frozen[freeze_ids] = True
            self.plan_mode[freeze_ids] = PLAN_FROZEN

        pre_ids = torch.nonzero(self.active_swing & (self.t_to_hit > 0.0) & (self.swing_change_remaining > 0), as_tuple=False).flatten()
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

        self._update_real_events()

        missed = torch.nonzero(self.active_swing & (~self.legal_contact) & (self.t_to_hit < -self.cfg.miss_grace_time), as_tuple=False).flatten()
        if len(missed) > 0:
            self.missed_swing[missed] = True

        done_mask = (
            self.active_swing
            & (
                (self.t_to_hit <= -self.t_post_swing)
                | self.target_land_success
                | self.missed_swing
                | self.illegal_contact
            )
        )
        if self.cfg.post_outcome_hold_time > 0.0:
            start_hold = done_mask & (~self.outcome_hold_active)
            if torch.any(start_hold):
                self.outcome_hold_active[start_hold] = True
                self.outcome_hold_time_left[start_hold] = float(self.cfg.post_outcome_hold_time)
            done_mask = done_mask & self.outcome_hold_active & (self.outcome_hold_time_left <= 0.0)

        done = torch.nonzero(done_mask, as_tuple=False).flatten()
        if len(done) > 0:
            self._complete_real_swing(done)
            self._serve_ball(done)
            self._reset_real_swing_state(done)
            self._start_or_update_plan(done, update_timing=True, force=True)
            if torch.any(~self.planner_valid[done]):
                invalid = done[~self.planner_valid[done]]
                self._set_fallback_command(invalid)
            self.command_counter[done] += 1
            self.time_left[done] = self.cfg.resampling_time_range[1]

        self._update_ref_state()
        self.prev_ball_pos_world[:] = self.ball.data.root_pos_w
        self.prev_ball_vel_world[:] = self.ball.data.root_vel_w[:, :3]

    def _serve_ball(self, ids: torch.Tensor) -> None:
        env_origins = self._env.scene.env_origins[ids]
        n = len(ids)
        serve_x = sample_uniform(self.cfg.serve_pos_x_range[0], self.cfg.serve_pos_x_range[1], (n,), device=self.device)
        serve_y = sample_uniform(self.cfg.serve_pos_y_range[0], self.cfg.serve_pos_y_range[1], (n,), device=self.device)
        serve_z = sample_uniform(self.cfg.serve_pos_z_range[0], self.cfg.serve_pos_z_range[1], (n,), device=self.device)
        hit_y = sample_uniform(self.cfg.serve_hit_y_range[0], self.cfg.serve_hit_y_range[1], (n,), device=self.device)
        hit_z = sample_uniform(self.cfg.serve_hit_z_range[0], self.cfg.serve_hit_z_range[1], (n,), device=self.device)
        t_hit = sample_uniform(self.cfg.serve_t_to_hit_range[0], self.cfg.serve_t_to_hit_range[1], (n,), device=self.device)

        local_pos = torch.stack((serve_x, serve_y, serve_z), dim=-1)
        local_hit = torch.stack((torch.full_like(hit_y, self.cfg.hit_x), hit_y, hit_z), dim=-1)
        pos = env_origins + local_pos
        hit = env_origins + local_hit
        gravity = torch.tensor((0.0, 0.0, -9.81), device=self.device).view(1, 3)
        vel = (hit - pos - 0.5 * gravity * t_hit.unsqueeze(-1) ** 2) / t_hit.unsqueeze(-1)
        speed = torch.linalg.norm(vel, dim=-1, keepdim=True)
        max_speed = float(self.cfg.serve_max_speed)
        vel = torch.where(speed > max_speed, vel / speed.clamp_min(1.0e-6) * max_speed, vel)

        quat = torch.zeros(n, 4, device=self.device)
        quat[:, 0] = 1.0
        ang = torch.zeros(n, 3, device=self.device)
        root_state = torch.cat((pos, quat, vel, ang), dim=-1)
        self.ball.write_root_state_to_sim(root_state, env_ids=ids)
        self.prev_ball_pos_world[ids] = pos
        self.prev_ball_vel_world[ids] = vel

    def _start_or_update_plan(
        self,
        ids: torch.Tensor,
        update_timing: bool,
        force: bool,
        root_pos_override: torch.Tensor | None = None,
        root_quat_override: torch.Tensor | None = None,
    ) -> None:
        if len(ids) == 0:
            return
        root_pos = self.robot.data.root_pos_w[ids] if root_pos_override is None else root_pos_override
        root_quat = self.robot.data.root_quat_w[ids] if root_quat_override is None else root_quat_override
        env_origins = self._env.scene.env_origins[ids]
        target_land = env_origins + torch.tensor(self.cfg.target_land, dtype=torch.float32, device=self.device).view(1, 3)
        x_hit = env_origins[:, 0] + self.cfg.hit_x
        table_center_x = env_origins[:, 0] + self.cfg.table_center_x

        plan = plan_pingpong_hits(
            self.ball.data.root_pos_w[ids],
            self.ball.data.root_vel_w[ids, :3],
            root_pos,
            root_quat,
            target_land,
            table_top_z=env_origins[:, 2] + self.cfg.table_top_z,
            ball_radius=self.cfg.ball_radius,
            valid_mask=torch.ones(len(ids), dtype=torch.bool, device=self.device),
            x_hit_world=x_hit,
            table_center_x_world=table_center_x,
            table_center_y_world=env_origins[:, 1],
            table_half_x=self.cfg.table_half_x,
            table_half_y=self.cfg.table_half_y,
            expert_offset_base=self.expert_offset_base,
            y_mid_base=self.cfg.y_mid_base,
            flight_time=self.cfg.return_flight_time,
            paddle_cor=self.cfg.paddle_cor,
            dt=self.cfg.planner_dt,
            max_time=self.cfg.planner_max_time,
            drag_k=self.cfg.planner_drag_k,
            bounce_ch=self.cfg.planner_bounce_ch,
            bounce_cv=self.cfg.planner_bounce_cv,
            min_t_to_hit=self.cfg.planner_min_t_to_hit,
            max_t_to_hit=self.cfg.planner_max_t_to_hit,
            hit_z_range=self.cfg.planner_hit_z_range,
        )
        self._store_planner_debug_traj(ids, plan.traj_p, plan.traj_valid & plan.planner_valid.unsqueeze(-1))
        valid = plan.planner_valid | force
        if not torch.any(valid):
            self.planner_valid[ids] = False
            return

        use_ids = ids[valid]
        local_valid = valid
        self.planner_valid[use_ids] = plan.planner_valid[local_valid]
        self.plan_mode[use_ids] = torch.where(plan.planner_valid[local_valid], plan.plan_mode[local_valid], torch.zeros_like(plan.plan_mode[local_valid]))
        self.p_hit_world[use_ids] = plan.p_hit_world[local_valid]
        self.v_ball_in_world[use_ids] = plan.v_ball_in_world[local_valid]
        self.target_land_world[use_ids] = plan.target_land_world[local_valid]
        self.flight_time[use_ids] = float(self.cfg.return_flight_time)
        self.paddle_cor[use_ids] = float(self.cfg.paddle_cor)
        self.v_ball_out_world[use_ids] = plan.v_ball_out_world[local_valid]
        self.n_target_world[use_ids] = plan.n_target_world[local_valid]
        self.v_racket_hat_world[use_ids] = plan.v_racket_hat_world[local_valid]
        self.swing_type[use_ids] = plan.swing_type[local_valid]
        _, hit_y_base = self._compute_swing_type(use_ids, root_pos[local_valid, :2], root_quat[local_valid])
        self.hit_y_base[use_ids] = hit_y_base
        self.p_base_xy_world[use_ids] = plan.p_base_xy_world[local_valid]

        if update_timing:
            t = torch.where(plan.planner_valid[local_valid], plan.t_to_hit[local_valid], torch.full_like(plan.t_to_hit[local_valid], self.cfg.default_t_to_hit))
            self.t_pre_initial[use_ids] = t.clamp_min(self.dt)
            self.t_post_swing[use_ids] = float(self.cfg.t_post_swing_fixed)
            self.t_to_hit[use_ids] = self.t_pre_initial[use_ids]
            self.cur_step[use_ids] = 0
            self.active_swing[use_ids] = True
            self.target_frozen[use_ids] = False
            self.swing_change_remaining[use_ids] = 1
            self._reset_window_flags(use_ids)
            self._reset_real_swing_flags(use_ids)

    def _set_fallback_command(
        self,
        ids: torch.Tensor,
        root_pos_override: torch.Tensor | None = None,
        root_quat_override: torch.Tensor | None = None,
    ) -> None:
        if len(ids) == 0:
            return
        root_pos = self.robot.data.root_pos_w[ids] if root_pos_override is None else root_pos_override
        root_quat = self.robot.data.root_quat_w[ids] if root_quat_override is None else root_quat_override
        env_origins = self._env.scene.env_origins[ids]
        local_hit = torch.tensor((self.cfg.hit_x, 0.0, self.cfg.fallback_hit_z), dtype=torch.float32, device=self.device).view(1, 3)
        self.p_hit_world[ids] = env_origins + local_hit
        self.v_ball_in_world[ids] = torch.tensor((-3.0, 0.0, -0.5), dtype=torch.float32, device=self.device).view(1, 3)
        self.target_land_world[ids] = env_origins + torch.tensor(self.cfg.target_land, dtype=torch.float32, device=self.device).view(1, 3)
        self._solve_paddle_target(ids)
        swing, hit_y_base = self._compute_swing_type(ids, root_pos[:, :2], root_quat)
        self.swing_type[ids] = swing
        self.hit_y_base[ids] = hit_y_base
        self.p_base_xy_world[ids] = self._compute_base_target(ids, root_quat)
        self.t_pre_initial[ids] = self.cfg.default_t_to_hit
        self.t_post_swing[ids] = float(self.cfg.t_post_swing_fixed)
        self.t_to_hit[ids] = self.t_pre_initial[ids]
        self.cur_step[ids] = 0
        self.active_swing[ids] = True
        self.planner_valid[ids] = False
        self.plan_mode[ids] = 0

    def _update_real_events(self) -> None:
        in_window = torch.abs(self.t_to_hit) <= self.cfg.real_contact_window
        ball_pos = self.ball.data.root_pos_w
        ball_vel = self.ball.data.root_vel_w[:, :3]

        racket_contact = self._contact_sensor_active(self.cfg.ball_racket_sensor_name, self.cfg.contact_force_threshold)
        robot_contact = self._contact_sensor_active(self.cfg.ball_robot_sensor_name, self.cfg.contact_force_threshold)
        table_contact = self._contact_sensor_active(self.cfg.ball_table_sensor_name, self.cfg.contact_force_threshold)
        net_contact = self._contact_sensor_active(self.cfg.ball_net_sensor_name, self.cfg.contact_force_threshold)

        # Distance fallback keeps smoke tests usable even if filtered contact reports are unavailable.
        dist_to_blade = torch.linalg.norm(ball_pos - self.robot_blade_pos_w, dim=-1)
        racket_contact = racket_contact | ((dist_to_blade < self.cfg.racket_contact_distance) & in_window)

        new_legal = self.active_swing & (~self.legal_contact) & racket_contact & in_window
        if torch.any(new_legal):
            self.legal_contact[new_legal] = True
            self.actual_contact_pos_world[new_legal] = ball_pos[new_legal]
            self.actual_contact_time_err[new_legal] = torch.abs(self.t_to_hit[new_legal])
            self.actual_contact_pos_err[new_legal] = torch.linalg.norm(ball_pos[new_legal] - self.p_hit_world[new_legal], dim=-1)
            self.planner_success[new_legal] = (
                self.planner_valid[new_legal]
                & (self.actual_contact_time_err[new_legal] < self.cfg.planner_success_time_thresh)
                & (self.actual_contact_pos_err[new_legal] < self.cfg.planner_success_pos_thresh)
            )
            self._reward_ball_contact[new_legal] = 1.0
            direction_ok = ball_vel[new_legal, 0] > self.cfg.return_direction_min_vx
            self.return_direction[new_legal] = direction_ok
            self._reward_return_direction[new_legal] = direction_ok.float()

        new_illegal = self.active_swing & (~self.legal_contact) & (robot_contact | net_contact) & (~racket_contact)
        out_of_bounds = self.active_swing & (
            (ball_pos[:, 2] < self.cfg.ball_dead_z)
            | (torch.abs(ball_pos[:, 1] - self._env.scene.env_origins[:, 1]) > self.cfg.ball_dead_y_abs)
            | (torch.abs(ball_pos[:, 0] - self._env.scene.env_origins[:, 0] - self.cfg.table_center_x) > self.cfg.ball_dead_x_abs)
        )
        new_illegal = new_illegal | out_of_bounds
        if torch.any(new_illegal & (~self.illegal_contact)):
            ids = new_illegal & (~self.illegal_contact)
            self.illegal_contact[ids] = True
            self._reward_illegal[ids] = 1.0

        if torch.any(net_contact):
            self._real_net_contact_count += (net_contact & self.active_swing).float()

        net_x = self._env.scene.env_origins[:, 0] + self.cfg.net_x
        crossed_net = self.legal_contact & (~self.net_cross_checked) & (self.prev_ball_pos_world[:, 0] < net_x) & (ball_pos[:, 0] >= net_x)
        if torch.any(crossed_net):
            denom = (ball_pos[:, 0] - self.prev_ball_pos_world[:, 0]).clamp_min(1.0e-6)
            alpha = ((net_x - self.prev_ball_pos_world[:, 0]) / denom).clamp(0.0, 1.0)
            z_cross = self.prev_ball_pos_world[:, 2] + alpha * (ball_pos[:, 2] - self.prev_ball_pos_world[:, 2])
            clearance = z_cross - (self.cfg.net_top_z + self.cfg.ball_radius)
            self.net_clearance[crossed_net] = clearance[crossed_net]
            clear = crossed_net & (clearance > 0.0)
            self.clear_net[clear] = True
            self._reward_clear_net[clear] = 1.0
            self.net_cross_checked[crossed_net] = True

        post_hit_table = self.legal_contact & (~self.first_post_hit_bounce_done) & table_contact
        if torch.any(post_hit_table):
            env_origins = self._env.scene.env_origins
            in_opp = (
                (ball_pos[:, 0] > env_origins[:, 0] + self.cfg.net_x)
                & (ball_pos[:, 0] < env_origins[:, 0] + self.cfg.far_edge_x)
                & (torch.abs(ball_pos[:, 1] - env_origins[:, 1]) < self.cfg.table_half_y)
            )
            success = post_hit_table & self.clear_net & in_opp
            self.opponent_land[success] = True
            self._reward_opponent_land[success] = 1.0
            land_err = torch.linalg.norm(ball_pos[:, :2] - self.target_land_world[:, :2], dim=-1)
            self.landing_error[post_hit_table] = land_err[post_hit_table]
            self.landing_pos_world[post_hit_table] = ball_pos[post_hit_table]
            self.landing_pos_valid[post_hit_table] = True
            target = success & (land_err < self.cfg.target_land_radius)
            self.target_land_success[target] = True
            self._reward_target_land[target] = 1.0
            self.first_post_hit_bounce_done[post_hit_table] = True

    def _contact_sensor_active(self, sensor_name: str, threshold: float) -> torch.Tensor:
        if not hasattr(self._env.scene, "sensors"):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        try:
            sensor = self._env.scene.sensors[sensor_name]
        except (KeyError, TypeError):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        forces = getattr(sensor.data, "force_matrix_w_history", None)
        if forces is None:
            forces = getattr(sensor.data, "net_forces_w_history", None)
        if forces is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        norm = torch.linalg.norm(forces, dim=-1)
        reduce_dims = tuple(range(1, norm.ndim))
        return torch.amax(norm, dim=reduce_dims) > threshold

    def _complete_real_swing(self, ids: torch.Tensor) -> None:
        self._complete_swing(ids)
        self._real_swing_count[ids] += 1.0
        self._real_hit_count[ids] += self.legal_contact[ids].float()
        self._real_return_count[ids] += (self.legal_contact[ids] & self.clear_net[ids] & self.opponent_land[ids]).float()
        self._real_target_count[ids] += self.target_land_success[ids].float()
        self._real_planner_success_count[ids] += self.planner_success[ids].float()
        self._real_illegal_count[ids] += self.illegal_contact[ids].float()
        self._refresh_metrics_from_counts()

    def _refresh_metrics_from_counts(self):
        super()._refresh_metrics_from_counts()
        denom = torch.clamp(self._real_swing_count, min=1.0)
        self.metrics["real_hit_success_rate"] = self._real_hit_count / denom
        self.metrics["real_return_success_rate"] = self._real_return_count / denom
        self.metrics["real_target_success_rate"] = self._real_target_count / denom
        self.metrics["real_planner_success_rate"] = self._real_planner_success_count / denom
        self.metrics["real_planner_valid_rate"] = self.planner_valid.float()
        self.metrics["real_planner_time_error_mean"] = self.actual_contact_time_err
        self.metrics["real_planner_pos_error_mean"] = self.actual_contact_pos_err
        self.metrics["real_net_clearance_mean"] = self.net_clearance
        self.metrics["real_landing_error_mean"] = self.landing_error
        self.metrics["real_illegal_contact_rate"] = self._real_illegal_count / denom
        self.metrics["real_ball_net_contact_rate"] = self._real_net_contact_count / denom
        self.metrics["real_post_recovery_interrupt_rate"] = self._post_recovery_interrupt_count / torch.clamp(self.command_counter.float(), min=1.0)

    def _reset_real_counters(self, ids: torch.Tensor) -> None:
        self._real_swing_count[ids] = 0.0
        self._real_hit_count[ids] = 0.0
        self._real_return_count[ids] = 0.0
        self._real_target_count[ids] = 0.0
        self._real_planner_success_count[ids] = 0.0
        self._real_illegal_count[ids] = 0.0
        self._real_net_contact_count[ids] = 0.0
        self._post_recovery_interrupt_count[ids] = 0.0
        self._reset_real_swing_flags(ids)

    def _reset_real_swing_state(self, ids: torch.Tensor) -> None:
        self.active_swing[ids] = False
        self.target_frozen[ids] = False
        self.planner_valid[ids] = False
        self.plan_mode[ids] = 0
        self._reset_real_swing_flags(ids)

    def _reset_real_swing_flags(self, ids: torch.Tensor) -> None:
        self.legal_contact[ids] = False
        self.illegal_contact[ids] = False
        self.return_direction[ids] = False
        self.clear_net[ids] = False
        self.opponent_land[ids] = False
        self.target_land_success[ids] = False
        self.planner_success[ids] = False
        self.first_post_hit_bounce_done[ids] = False
        self.net_cross_checked[ids] = False
        self.missed_swing[ids] = False
        self.outcome_hold_active[ids] = False
        self.outcome_hold_time_left[ids] = 0.0
        self.actual_contact_pos_world[ids] = 0.0
        self.actual_contact_time_err[ids] = 0.0
        self.actual_contact_pos_err[ids] = 0.0
        self.net_clearance[ids] = 0.0
        self.landing_error[ids] = 0.0
        self.landing_pos_world[ids] = 0.0
        self.landing_pos_valid[ids] = False
        self._planner_traj_valid[ids] = False
        self._ball_traj_valid[ids] = False
        self._ball_traj_cursor[ids] = 0
        self._clear_reward_pulses(ids)

    def _clear_reward_pulses(self, ids: torch.Tensor | None = None) -> None:
        if ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        self._reward_ball_contact[ids] = 0.0
        self._reward_return_direction[ids] = 0.0
        self._reward_clear_net[ids] = 0.0
        self._reward_opponent_land[ids] = 0.0
        self._reward_target_land[ids] = 0.0
        self._reward_illegal[ids] = 0.0

    def _record_ball_debug_traj(self) -> None:
        slots = self._ball_traj_cursor % self.cfg.debug_ball_traj_len
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._ball_traj_points[env_ids, slots] = self.ball.data.root_pos_w
        self._ball_traj_valid[env_ids, slots] = True
        self._ball_traj_cursor += 1

    def _store_planner_debug_traj(self, ids: torch.Tensor, traj_p: torch.Tensor, traj_valid: torch.Tensor) -> None:
        n = min(self.cfg.debug_planner_traj_len, traj_p.shape[1])
        self._planner_traj_points[ids] = 0.0
        self._planner_traj_valid[ids] = False
        self._planner_traj_points[ids, :n] = traj_p[:, :n]
        self._planner_traj_valid[ids, :n] = traj_valid[:, :n]

    def _set_debug_vis_impl(self, debug_vis: bool):
        super()._set_debug_vis_impl(debug_vis)
        if debug_vis:
            if self.cfg.debug_show_direction_arrows and not hasattr(self, "target_normal_arrow_visualizer"):
                normal_cfg = RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Pingpong/target_normal")
                desired_vel_cfg = GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Pingpong/desired_racket_velocity")
                current_vel_cfg = BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Pingpong/current_blade_velocity")
                normal_cfg.markers["arrow"].scale = self.cfg.debug_normal_arrow_base_scale
                desired_vel_cfg.markers["arrow"].scale = self.cfg.debug_desired_velocity_arrow_base_scale
                current_vel_cfg.markers["arrow"].scale = self.cfg.debug_current_velocity_arrow_base_scale
                self.target_normal_arrow_visualizer = VisualizationMarkers(normal_cfg)
                self.desired_velocity_arrow_visualizer = VisualizationMarkers(desired_vel_cfg)
                self.current_velocity_arrow_visualizer = VisualizationMarkers(current_vel_cfg)
            if hasattr(self, "target_normal_arrow_visualizer"):
                visible = bool(self.cfg.debug_show_direction_arrows)
                self.target_normal_arrow_visualizer.set_visibility(visible)
                self.desired_velocity_arrow_visualizer.set_visibility(visible)
                self.current_velocity_arrow_visualizer.set_visibility(visible)
        elif hasattr(self, "target_normal_arrow_visualizer"):
            self.target_normal_arrow_visualizer.set_visibility(False)
            self.desired_velocity_arrow_visualizer.set_visibility(False)
            self.current_velocity_arrow_visualizer.set_visibility(False)

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
        env_origins = self._env.scene.env_origins
        net_x = env_origins[:, 0] + self.cfg.net_x
        net_y = env_origins[:, 1]
        net_top = torch.full((self.num_envs,), self.cfg.net_top_z, dtype=torch.float32, device=self.device) + env_origins[:, 2]
        net_points = torch.stack(
            (
                torch.stack((net_x, net_y, net_top), dim=-1),
                torch.stack((net_x, net_y - self.cfg.table_half_y, net_top), dim=-1),
                torch.stack((net_x, net_y + self.cfg.table_half_y, net_top), dim=-1),
            ),
            dim=1,
        ).reshape(-1, 3)

        point_groups = [self.p_hit_world]
        if self.cfg.debug_show_aux_targets:
            point_groups.append(base_target)
        if self.cfg.debug_show_vectors:
            point_groups.extend((target_normal_end, racket_vel_end, blade_normal_end))
        if self.cfg.debug_show_current_ball_marker:
            point_groups.append(self.ball.data.root_pos_w)
        if self.cfg.debug_show_net_points:
            point_groups.append(net_points)
        blade_traj_points = self._debug_traj_points[self._debug_traj_valid]
        planner_traj_points = self._planner_traj_points[self._planner_traj_valid]
        ball_traj_points = self._ball_traj_points[self._ball_traj_valid]
        landing_points = self.landing_pos_world[self.landing_pos_valid]
        optional_groups = (
            (self.cfg.debug_show_blade_traj, blade_traj_points),
            (self.cfg.debug_show_planner_traj, planner_traj_points),
            (self.cfg.debug_show_ball_traj, ball_traj_points),
            (self.cfg.debug_show_landing_points, landing_points),
        )
        for enabled, points in optional_groups:
            if not enabled:
                continue
            if points.numel() > 0:
                point_groups.append(points)
        debug_points = torch.cat(point_groups, dim=0)
        debug_scales = torch.tensor(self.cfg.debug_point_scale, dtype=torch.float32, device=self.device).repeat(
            debug_points.shape[0], 1
        )
        self.target_visualizer.visualize(translations=debug_points, scales=debug_scales)
        self._debug_visualize_direction_arrows()

    def _debug_visualize_direction_arrows(self) -> None:
        if not self.cfg.debug_show_direction_arrows or not hasattr(self, "target_normal_arrow_visualizer"):
            return
        z_offset = torch.tensor((0.0, 0.0, 0.035), dtype=torch.float32, device=self.device).view(1, 3)

        normal_mag = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        normal_scale = _arrow_scales(
            normal_mag,
            self.cfg.debug_normal_arrow_base_scale,
            gain=self.cfg.debug_normal_arrow_gain,
            min_x=self.cfg.debug_normal_arrow_min_x,
            max_x=self.cfg.debug_normal_arrow_max_x,
        )
        self.target_normal_arrow_visualizer.visualize(
            translations=self.p_hit_world + z_offset,
            orientations=_quat_from_x_axis(self.n_target_world),
            scales=normal_scale,
        )

        desired_mag = torch.linalg.norm(self.v_racket_hat_world, dim=-1)
        desired_scale = _arrow_scales(
            desired_mag,
            self.cfg.debug_desired_velocity_arrow_base_scale,
            gain=self.cfg.debug_desired_velocity_arrow_gain,
            min_x=self.cfg.debug_desired_velocity_arrow_min_x,
            max_x=self.cfg.debug_desired_velocity_arrow_max_x,
        )
        self.desired_velocity_arrow_visualizer.visualize(
            translations=self.p_hit_world
            + torch.tensor(self.cfg.debug_desired_velocity_arrow_offset, dtype=torch.float32, device=self.device).view(1, 3),
            orientations=_quat_from_x_axis(self.v_racket_hat_world),
            scales=desired_scale,
        )

        current_vel = self.robot_blade_lin_vel_w
        current_mag = torch.linalg.norm(current_vel, dim=-1)
        current_scale = _arrow_scales(
            current_mag,
            self.cfg.debug_current_velocity_arrow_base_scale,
            gain=self.cfg.debug_current_velocity_arrow_gain,
            min_x=self.cfg.debug_current_velocity_arrow_min_x,
            max_x=self.cfg.debug_current_velocity_arrow_max_x,
        )
        self.current_velocity_arrow_visualizer.visualize(
            translations=self.robot_blade_pos_w + z_offset,
            orientations=_quat_from_x_axis(current_vel),
            scales=current_scale,
        )


@configclass
class RealPingpongCommandCfg(PingpongCommandCfg):
    class_type: type = RealPingpongCommand
    ball_name: str = MISSING

    table_center_x: float = 1.77
    table_top_z: float = 0.76
    table_half_x: float = 1.37
    table_half_y: float = 0.7625
    net_x: float = 1.77
    far_edge_x: float = 3.14
    net_top_z: float = 0.9125
    ball_radius: float = 0.02

    t_post_swing_fixed: float = 0.60
    post_outcome_hold_time: float = 0.0
    freeze_time_before_hit: float = 0.20
    miss_grace_time: float = 0.10
    real_contact_window: float = 0.08
    default_t_to_hit: float = 0.60
    return_flight_time: float = 0.45
    target_land_radius: float = 0.45

    serve_pos_x_range: tuple[float, float] = (2.55, 3.05)
    serve_pos_y_range: tuple[float, float] = (-0.20, 0.20)
    serve_pos_z_range: tuple[float, float] = (0.95, 1.25)
    serve_hit_y_range: tuple[float, float] = (-0.35, 0.35)
    serve_hit_z_range: tuple[float, float] = (0.95, 1.15)
    serve_t_to_hit_range: tuple[float, float] = (0.55, 0.90)
    serve_max_speed: float = 6.0
    fallback_hit_z: float = 1.05

    planner_dt: float = 0.01
    planner_max_time: float = 1.50
    planner_drag_k: float = 0.10257265376884504
    planner_bounce_ch: float = 0.727005044772834
    planner_bounce_cv: float = 0.9018357357260598
    planner_min_t_to_hit: float = 0.05
    planner_max_t_to_hit: float = 1.20
    planner_hit_z_range: tuple[float, float] = (0.85, 1.25)

    ball_racket_sensor_name: str = "ball_racket_contact"
    ball_robot_sensor_name: str = "ball_robot_contact"
    ball_table_sensor_name: str = "ball_table_contact"
    ball_net_sensor_name: str = "ball_net_contact"
    contact_force_threshold: float = 0.05
    racket_contact_distance: float = 0.045
    return_direction_min_vx: float = 0.05
    planner_success_time_thresh: float = 0.08
    planner_success_pos_thresh: float = 0.10
    ball_dead_z: float = 0.05
    ball_dead_y_abs: float = 3.0
    ball_dead_x_abs: float = 3.5
    debug_planner_traj_len: int = 64
    debug_ball_traj_len: int = 64
    debug_show_aux_targets: bool = True
    debug_show_vectors: bool = True
    debug_show_current_ball_marker: bool = False
    debug_show_net_points: bool = False
    debug_show_blade_traj: bool = False
    debug_show_planner_traj: bool = False
    debug_show_ball_traj: bool = True
    debug_show_landing_points: bool = True
    debug_show_direction_arrows: bool = True
    debug_point_scale: tuple[float, float, float] = (0.35, 0.35, 0.35)
    debug_normal_arrow_base_scale: tuple[float, float, float] = (0.35, 0.08, 0.08)
    debug_desired_velocity_arrow_base_scale: tuple[float, float, float] = (0.55, 0.12, 0.12)
    debug_current_velocity_arrow_base_scale: tuple[float, float, float] = (0.30, 0.08, 0.08)
    debug_desired_velocity_arrow_offset: tuple[float, float, float] = (0.0, -0.12, 0.10)
    debug_normal_arrow_gain: float = 1.0
    debug_normal_arrow_min_x: float = 1.0
    debug_normal_arrow_max_x: float = 1.0
    debug_desired_velocity_arrow_gain: float = 0.65
    debug_desired_velocity_arrow_min_x: float = 0.60
    debug_desired_velocity_arrow_max_x: float = 2.40
    debug_current_velocity_arrow_gain: float = 0.35
    debug_current_velocity_arrow_min_x: float = 0.25
    debug_current_velocity_arrow_max_x: float = 1.40
