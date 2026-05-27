from __future__ import annotations

from dataclasses import dataclass

import torch

from .motion_loader import yaw_from_wxyz


PLAN_INVALID = 0
PLAN_FRESH = 1
PLAN_HELD = 2
PLAN_FROZEN = 3


@dataclass
class TrainingPlannerOutput:
    p_hit_world: torch.Tensor
    v_ball_in_world: torch.Tensor
    v_ball_out_world: torch.Tensor
    v_racket_hat_world: torch.Tensor
    n_target_world: torch.Tensor
    target_land_world: torch.Tensor
    p_base_xy_world: torch.Tensor
    t_to_hit: torch.Tensor
    swing_type: torch.Tensor
    planner_valid: torch.Tensor
    plan_mode: torch.Tensor
    bounce_count_pred: torch.Tensor
    x_hit_used: torch.Tensor
    fallback_reason: torch.Tensor
    traj_p: torch.Tensor
    traj_valid: torch.Tensor


def _rotate_yaw_2d(vec_xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    return torch.stack((c * vec_xy[:, 0] - s * vec_xy[:, 1], s * vec_xy[:, 0] + c * vec_xy[:, 1]), dim=-1)


def _empty_output(
    ball_pos_world: torch.Tensor,
    target_land_world: torch.Tensor,
    x_hit_world: torch.Tensor,
    max_steps: int,
) -> TrainingPlannerOutput:
    n = ball_pos_world.shape[0]
    device = ball_pos_world.device
    dtype = ball_pos_world.dtype
    zeros3 = torch.zeros(n, 3, device=device, dtype=dtype)
    zeros2 = torch.zeros(n, 2, device=device, dtype=dtype)
    return TrainingPlannerOutput(
        p_hit_world=ball_pos_world.clone(),
        v_ball_in_world=zeros3.clone(),
        v_ball_out_world=zeros3.clone(),
        v_racket_hat_world=zeros3.clone(),
        n_target_world=torch.tensor((-1.0, 0.0, 0.0), device=device, dtype=dtype).expand(n, 3).clone(),
        target_land_world=target_land_world.clone(),
        p_base_xy_world=zeros2,
        t_to_hit=torch.zeros(n, device=device, dtype=dtype),
        swing_type=torch.zeros(n, device=device, dtype=torch.long),
        planner_valid=torch.zeros(n, device=device, dtype=torch.bool),
        plan_mode=torch.zeros(n, device=device, dtype=torch.long),
        bounce_count_pred=torch.zeros(n, device=device, dtype=torch.long),
        x_hit_used=x_hit_world.clone(),
        fallback_reason=torch.ones(n, device=device, dtype=torch.long),
        traj_p=torch.zeros(n, max_steps + 1, 3, device=device, dtype=dtype),
        traj_valid=torch.zeros(n, max_steps + 1, device=device, dtype=torch.bool),
    )


def solve_paddle_targets_batched(
    p_hit_world: torch.Tensor,
    v_ball_in_world: torch.Tensor,
    target_land_world: torch.Tensor,
    flight_time: torch.Tensor | float = 0.45,
    paddle_cor: torch.Tensor | float = 0.85,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched Eq.5/Eq.6 target solve used by the real training planner."""
    if not torch.is_tensor(flight_time):
        flight_time = torch.full((p_hit_world.shape[0],), float(flight_time), device=p_hit_world.device, dtype=p_hit_world.dtype)
    if not torch.is_tensor(paddle_cor):
        paddle_cor = torch.full((p_hit_world.shape[0],), float(paddle_cor), device=p_hit_world.device, dtype=p_hit_world.dtype)

    t = flight_time.to(device=p_hit_world.device, dtype=p_hit_world.dtype).clamp_min(1.0e-3).unsqueeze(-1)
    cor = paddle_cor.to(device=p_hit_world.device, dtype=p_hit_world.dtype)
    gravity_term = torch.tensor((0.0, 0.0, 0.5 * 9.81), device=p_hit_world.device, dtype=p_hit_world.dtype).view(1, 3) * t
    v_ball_out_world = (target_land_world - p_hit_world) / t + gravity_term

    delta_v = v_ball_out_world - v_ball_in_world
    norm = torch.linalg.norm(delta_v, dim=-1, keepdim=True)
    degenerate = norm.squeeze(-1) < 1.0e-9
    n_target_world = delta_v / norm.clamp_min(1.0e-9)
    fallback_n = torch.tensor((-1.0, 0.0, 0.0), device=p_hit_world.device, dtype=p_hit_world.dtype).expand_as(n_target_world)
    n_target_world = torch.where(degenerate.unsqueeze(-1), fallback_n, n_target_world)

    v_in_n = torch.sum(v_ball_in_world * n_target_world, dim=-1)
    v_out_n = torch.sum(v_ball_out_world * n_target_world, dim=-1)
    v_pad_n = (v_out_n + cor * v_in_n) / (1.0 + cor)
    v_racket_hat_world = v_pad_n.unsqueeze(-1) * n_target_world
    v_racket_hat_world = torch.where(degenerate.unsqueeze(-1), 2.0 * fallback_n, v_racket_hat_world)
    return v_ball_out_world, n_target_world, v_racket_hat_world


def plan_pingpong_hits(
    ball_pos_world: torch.Tensor,
    ball_vel_world: torch.Tensor,
    robot_root_pos_world: torch.Tensor,
    robot_root_quat_world: torch.Tensor,
    target_land_world: torch.Tensor,
    table_top_z: torch.Tensor | float = 0.76,
    ball_radius: torch.Tensor | float = 0.02,
    valid_mask: torch.Tensor | None = None,
    *,
    x_hit_world: torch.Tensor | float = 0.4,
    table_center_x_world: torch.Tensor | float | None = None,
    table_center_y_world: torch.Tensor | float | None = None,
    table_half_x: float = 1.37,
    table_half_y: float = 0.7625,
    expert_offset_base: torch.Tensor | None = None,
    y_mid_base: float = 0.157,
    flight_time: torch.Tensor | float = 0.45,
    paddle_cor: torch.Tensor | float = 0.85,
    dt: float = 0.01,
    max_time: float = 1.50,
    drag_k: float = 0.10257265376884504,
    bounce_ch: float = 0.727005044772834,
    bounce_cv: float = 0.9018357357260598,
    min_t_to_hit: float = 0.05,
    max_t_to_hit: float = 1.20,
    hit_z_range: tuple[float, float] = (0.85, 1.25),
) -> TrainingPlannerOutput:
    """Predict hit commands from batched ball states.

    The planner rolls the ball forward using the same simple form as the runtime
    planner: quadratic drag in flight and anisotropic table bounce. It searches
    for the first crossing of ``x_hit_world`` by a ball moving toward the robot.
    """
    n = ball_pos_world.shape[0]
    device = ball_pos_world.device
    dtype = ball_pos_world.dtype
    max_steps = max(1, int(max_time / dt))

    if valid_mask is None:
        valid_mask = torch.ones(n, device=device, dtype=torch.bool)
    else:
        valid_mask = valid_mask.to(device=device, dtype=torch.bool)

    if not torch.is_tensor(x_hit_world):
        x_hit_world = torch.full((n,), float(x_hit_world), device=device, dtype=dtype)
    else:
        x_hit_world = x_hit_world.to(device=device, dtype=dtype)
    if not torch.is_tensor(table_top_z):
        table_top_z = torch.full((n,), float(table_top_z), device=device, dtype=dtype)
    else:
        table_top_z = table_top_z.to(device=device, dtype=dtype)
    if not torch.is_tensor(ball_radius):
        ball_radius = torch.full((n,), float(ball_radius), device=device, dtype=dtype)
    else:
        ball_radius = ball_radius.to(device=device, dtype=dtype)
    if table_center_x_world is None:
        table_center_x_world = x_hit_world + table_half_x
    elif not torch.is_tensor(table_center_x_world):
        table_center_x_world = torch.full((n,), float(table_center_x_world), device=device, dtype=dtype)
    else:
        table_center_x_world = table_center_x_world.to(device=device, dtype=dtype)
    if table_center_y_world is None:
        table_center_y_world = torch.zeros(n, device=device, dtype=dtype)
    elif not torch.is_tensor(table_center_y_world):
        table_center_y_world = torch.full((n,), float(table_center_y_world), device=device, dtype=dtype)
    else:
        table_center_y_world = table_center_y_world.to(device=device, dtype=dtype)

    out = _empty_output(ball_pos_world, target_land_world, x_hit_world, max_steps)
    out.traj_p[:, 0] = ball_pos_world
    out.traj_valid[:, 0] = valid_mask

    p = ball_pos_world.clone()
    v = ball_vel_world.clone()
    prev_p = p.clone()
    prev_v = v.clone()
    found = torch.zeros(n, device=device, dtype=torch.bool)
    bounce_count = torch.zeros(n, device=device, dtype=torch.long)
    g = torch.tensor((0.0, 0.0, -9.81), device=device, dtype=dtype).view(1, 3)
    center_z = table_top_z + ball_radius

    for step in range(1, max_steps + 1):
        speed = torch.linalg.norm(v, dim=-1, keepdim=True)
        acc = g - float(drag_k) * speed * v
        v_next = v + acc * float(dt)
        p_next = p + v_next * float(dt)

        on_table_xy = (torch.abs(p_next[:, 0] - table_center_x_world) <= table_half_x) & (
            torch.abs(p_next[:, 1] - table_center_y_world) <= table_half_y
        )
        bounced = (p[:, 2] > center_z) & (p_next[:, 2] <= center_z) & (v_next[:, 2] < 0.0) & on_table_xy
        if torch.any(bounced):
            p_next[bounced, 2] = center_z[bounced]
            v_next[bounced, :2] = v_next[bounced, :2] * float(bounce_ch)
            v_next[bounced, 2] = -v_next[bounced, 2] * float(bounce_cv)
            bounce_count[bounced] += 1

        out.traj_p[:, step] = p_next
        out.traj_valid[:, step] = valid_mask

        moving_to_robot = prev_v[:, 0] < -0.05
        crosses = (prev_p[:, 0] >= x_hit_world) & (p_next[:, 0] <= x_hit_world)
        eligible = valid_mask & (~found) & moving_to_robot & crosses
        if torch.any(eligible):
            denom = (p_next[:, 0] - prev_p[:, 0]).clamp(max=-1.0e-6)
            alpha = ((x_hit_world - prev_p[:, 0]) / denom).clamp(0.0, 1.0)
            p_hit = prev_p + alpha.unsqueeze(-1) * (p_next - prev_p)
            v_hit = prev_v + alpha.unsqueeze(-1) * (v_next - prev_v)
            t_hit = (float(step - 1) + alpha) * float(dt)
            z_ok = (p_hit[:, 2] >= hit_z_range[0]) & (p_hit[:, 2] <= hit_z_range[1])
            t_ok = (t_hit >= min_t_to_hit) & (t_hit <= max_t_to_hit)
            use = eligible & z_ok & t_ok
            out.p_hit_world[use] = p_hit[use]
            out.v_ball_in_world[use] = v_hit[use]
            out.t_to_hit[use] = t_hit[use]
            out.planner_valid[use] = True
            out.plan_mode[use] = PLAN_FRESH
            out.bounce_count_pred[use] = bounce_count[use]
            out.fallback_reason[use] = 0
            found |= use
            out.fallback_reason[eligible & ~z_ok] = 2
            out.fallback_reason[eligible & z_ok & ~t_ok] = 3

        prev_p = p
        prev_v = v
        p = p_next
        v = v_next

    out.fallback_reason[valid_mask & ~found & (ball_vel_world[:, 0] >= -0.05)] = 4

    if torch.any(out.planner_valid):
        v_out, n_target, v_racket = solve_paddle_targets_batched(
            out.p_hit_world,
            out.v_ball_in_world,
            target_land_world,
            flight_time=flight_time,
            paddle_cor=paddle_cor,
        )
        out.v_ball_out_world = v_out
        out.n_target_world = n_target
        out.v_racket_hat_world = v_racket

    if expert_offset_base is None:
        expert_offset_base = torch.tensor(
            ((0.496, 0.208), (0.428, 0.106)),
            device=device,
            dtype=dtype,
        )
    else:
        expert_offset_base = expert_offset_base.to(device=device, dtype=dtype)

    yaw = yaw_from_wxyz(robot_root_quat_world)
    diff_xy = out.p_hit_world[:, :2] - robot_root_pos_world[:, :2]
    hit_base_xy = _rotate_yaw_2d(diff_xy, -yaw)
    forehand = hit_base_xy[:, 1] > float(y_mid_base)
    out.swing_type = torch.where(forehand, torch.zeros_like(out.swing_type), torch.ones_like(out.swing_type))
    offsets = expert_offset_base[out.swing_type]
    offsets_world = _rotate_yaw_2d(offsets, yaw)
    out.p_base_xy_world = out.p_hit_world[:, :2] - offsets_world

    return out
