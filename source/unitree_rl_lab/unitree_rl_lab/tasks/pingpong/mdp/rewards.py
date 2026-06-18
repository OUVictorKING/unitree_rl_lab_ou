from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

try:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_apply, quat_rotate_inverse as quat_apply_inverse

from .commands import BLADE_NORMAL_LOCAL, PingpongCommand
from .motion_loader import yaw_from_wxyz

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _command(env: "ManagerBasedRLEnv", command_name: str) -> PingpongCommand:
    return env.command_manager.get_term(command_name)


def _strike_gate(cmd: PingpongCommand) -> torch.Tensor:
    return (torch.abs(cmd.t_to_hit) <= cmd.cfg.strike_window).float()


def action_l2_bounded(env: "ManagerBasedRLEnv", max_sq: float = 25.0) -> torch.Tensor:
    """action_l2 with each joint's squared term clamped. action_manager.action is the
    RAW (pre-clip) action, so a divergent actor (raw action -> 1e10) makes the stock
    sum(action²) hit ~1e20 -> penalty -1e22 -> value function detonates -> runaway
    (run 2026-06-01_23-00-45 at iter 60002; same class as the v61 iter-35k crash).
    max_sq=25 caps |action|>5; normal actions are ~±3 so training is unchanged."""
    sq = torch.square(env.action_manager.action)
    return torch.sum(torch.clamp(sq, max=max_sq), dim=1)


def action_rate_l2_bounded(env: "ManagerBasedRLEnv", max_sq: float = 4.0) -> torch.Tensor:
    """action_rate_l2 with each joint's squared delta clamped. max_sq=4 caps a per-step
    |Δaction|>2; normal deltas are <<2 so this only bounds divergence, not real swings."""
    sq = torch.square(env.action_manager.action - env.action_manager.prev_action)
    return torch.sum(torch.clamp(sq, max=max_sq), dim=1)


def imitation_joint_pos(
    env: "ManagerBasedRLEnv", command_name: str, k: float = 2.0, gate_pre_strike: bool = False,
    post_strike_scale: float = 1.0, post_strike_delay: float = -1.0,
) -> torch.Tensor:
    cmd = _command(env, command_name)
    ids = cmd.upper_joint_ids
    # Follow-through: `post_strike_delay` s AFTER the strike (t_to_hit <= -delay) RESTORE
    # full per-joint imitation weight (1.0) so the arm tracks the demo's post-strike
    # return — overriding the lowered face-joint weights that free the paddle face during
    # the approach/strike. The delay leaves a ~1-2 frame CONTACT BUFFER (ball still near
    # the paddle) before the follow-through kicks in. delay < 0 disables (legacy).
    if post_strike_delay >= 0.0:
        ft = (cmd.t_to_hit <= -post_strike_delay).float().unsqueeze(-1)
        w = cmd.imit_joint_weights.unsqueeze(0) * (1.0 - ft) + ft
    else:
        w = cmd.imit_joint_weights.unsqueeze(0)
    err = torch.sum(w * torch.square(cmd.robot.data.joint_pos[:, ids] - cmd.ref_state.joint_pos[:, ids]), dim=-1)
    reward = torch.exp(-k * err)
    if gate_pre_strike:
        reward = reward * (cmd.t_to_hit > 0.0).float()
    elif post_strike_scale != 1.0:
        # weaken imitation only in the CONTACT BUFFER (−delay < t_to_hit ≤ 0); the
        # follow-through (t_to_hit ≤ −delay) stays full. delay<0 → legacy: all post-strike.
        if post_strike_delay >= 0.0:
            post = ((cmd.t_to_hit <= 0.0) & (cmd.t_to_hit > -post_strike_delay)).float()
        else:
            post = (cmd.t_to_hit <= 0.0).float()
        reward = reward * (1.0 - (1.0 - post_strike_scale) * post)
    return reward


def imitation_joint_vel(
    env: "ManagerBasedRLEnv", command_name: str, k: float = 0.1, gate_pre_strike: bool = False,
    post_strike_scale: float = 1.0, post_strike_delay: float = -1.0,
) -> torch.Tensor:
    cmd = _command(env, command_name)
    ids = cmd.upper_joint_ids
    # Same delayed follow-through restore as imitation_joint_pos (see its docstring).
    if post_strike_delay >= 0.0:
        ft = (cmd.t_to_hit <= -post_strike_delay).float().unsqueeze(-1)
        w = cmd.imit_joint_weights.unsqueeze(0) * (1.0 - ft) + ft
    else:
        w = cmd.imit_joint_weights.unsqueeze(0)
    err = torch.sum(w * torch.square(cmd.robot.data.joint_vel[:, ids] - cmd.ref_state.joint_vel[:, ids]), dim=-1)
    reward = torch.exp(-k * err)
    if gate_pre_strike:
        reward = reward * (cmd.t_to_hit > 0.0).float()
    elif post_strike_scale != 1.0:
        if post_strike_delay >= 0.0:
            post = ((cmd.t_to_hit <= 0.0) & (cmd.t_to_hit > -post_strike_delay)).float()
        else:
            post = (cmd.t_to_hit <= 0.0).float()
        reward = reward * (1.0 - (1.0 - post_strike_scale) * post)
    return reward


def _anchor_relative_pos(body_pos: torch.Tensor, pelvis_pos: torch.Tensor) -> torch.Tensor:
    rel_xy = body_pos[..., :2] - pelvis_pos[:, None, :2]
    z = body_pos[..., 2:3]
    return torch.cat((rel_xy, z), dim=-1)


def imitation_body_pos_anchor_relative(
    env: "ManagerBasedRLEnv", command_name: str, k: float = 10.0, gate_pre_strike: bool = False
) -> torch.Tensor:
    cmd = _command(env, command_name)
    sim_rel = _anchor_relative_pos(cmd.robot_tracked_body_pos_w, cmd.robot_pelvis_pos_w)
    ref_rel = _anchor_relative_pos(cmd.ref_state.body_pos_w, cmd.ref_state.pelvis_pos_w)
    err = torch.sum(torch.square(sim_rel - ref_rel), dim=(1, 2))
    reward = torch.exp(-k * err)
    if gate_pre_strike:
        reward = reward * (cmd.t_to_hit > 0.0).float()
    return reward


def goal_position(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    cmd = _command(env, command_name)
    root_pos = cmd.robot.data.root_pos_w
    root_quat = cmd.robot.data.root_quat_w
    p_blade_b = quat_apply_inverse(root_quat, cmd.robot_blade_pos_w - root_pos)
    p_hit_b = quat_apply_inverse(root_quat, cmd.p_hit_world - root_pos)
    err = torch.sum(torch.square(p_blade_b - p_hit_b), dim=-1)
    return torch.exp(-err / (cmd.cfg.sigma_g_pos**2)) * _strike_gate(cmd)


def goal_position_pre_strike(env: "ManagerBasedRLEnv", command_name: str, std: float = 0.5, ramp_time: float = 0.20) -> torch.Tensor:
    # Linear back-projection: target paddle position now is where it must be to
    # reach p_hit with velocity v_hat at strike instant. Forces wind-up trajectory
    # along v_hat instead of static parking at p_hit.
    cmd = _command(env, command_name)
    root_pos = cmd.robot.data.root_pos_w
    root_quat = cmd.robot.data.root_quat_w
    p_target_w = cmd.p_hit_world - cmd.v_racket_hat_world * cmd.t_to_hit.unsqueeze(-1)
    p_blade_b = quat_apply_inverse(root_quat, cmd.robot_blade_pos_w - root_pos)
    p_target_b = quat_apply_inverse(root_quat, p_target_w - root_pos)
    err = torch.sum(torch.square(p_blade_b - p_target_b), dim=-1)
    gate = ((cmd.t_to_hit > 0.0) & (cmd.t_to_hit < ramp_time)).float()
    return torch.exp(-err / (std**2)) * gate


def goal_velocity(env: "ManagerBasedRLEnv", command_name: str, std: float = 1.5, half_width: float | None = None) -> torch.Tensor:
    """v62: Gaussian formula (squared norm) — was Laplacian (linear norm) in
    v61 and earlier. Laplacian gave near-zero gradient at moderate ||Δv|| (e.g.,
    σ=0.45, ||Δv||=2 m/s → reward = 0.001). Gaussian with wider σ provides
    perceptible gradient across the realistic error range during training:
    σ=1.5, ||Δv||=2 m/s → reward = 0.17 (170× larger). Curriculum tightens
    σ as policy improves (1.5 → 0.5 paper-strict), staggered with v_in_mag
    curriculum to avoid simultaneous tightening + ball-speed increase that
    would shock the policy out of reward range (run 2026-05-30 plateau evidence).
    """
    cmd = _command(env, command_name)
    root_quat = cmd.robot.data.root_quat_w
    v_blade_b = quat_apply_inverse(root_quat, cmd.robot_blade_lin_vel_w)
    v_hat_b = quat_apply_inverse(root_quat, cmd.v_racket_hat_world)
    err = torch.sum(torch.square(v_blade_b - v_hat_b), dim=-1)
    if half_width is None:
        gate = _strike_gate(cmd)
    else:
        gate = (torch.abs(cmd.t_to_hit) <= half_width).float()
    return torch.exp(-err / (std**2)) * gate


def goal_velocity_pre_strike(
    env: "ManagerBasedRLEnv", command_name: str, std: float = 0.6, ramp_time: float = 0.1
) -> torch.Tensor:
    cmd = _command(env, command_name)
    root_quat = cmd.robot.data.root_quat_w
    v_blade_b = quat_apply_inverse(root_quat, cmd.robot_blade_lin_vel_w)
    v_hat_b = quat_apply_inverse(root_quat, cmd.v_racket_hat_world)
    ramp = torch.clamp(1.0 - cmd.t_to_hit / ramp_time, 0.0, 1.0)
    v_target_b = ramp.unsqueeze(-1) * v_hat_b
    err = torch.sum(torch.square(v_blade_b - v_target_b), dim=-1)
    gate = ((cmd.t_to_hit > 0.0) & (cmd.t_to_hit < ramp_time)).float()
    return torch.exp(-err / (std**2)) * gate


def blade_normal_world(cmd: PingpongCommand) -> torch.Tensor:
    normal_local = torch.tensor(BLADE_NORMAL_LOCAL, dtype=torch.float32, device=cmd.device).expand(cmd.num_envs, 3)
    return quat_apply(cmd.robot_blade_quat_w, normal_local)


def goal_orientation(env: "ManagerBasedRLEnv", command_name: str, std: float = 0.2) -> torch.Tensor:
    # Signed by swing_type: forehand rewards +n_blade alignment, backhand
    # rewards -n_blade alignment. Matches _blade_target_cosine() in commands.py
    # so reward, metric, and ori_ok success check share one definition.
    # Symmetric |dot| killed the swing-discovery gradient — see commands.py
    # _blade_target_cosine docstring.
    cmd = _command(env, command_name)
    sign = 1.0 - 2.0 * cmd.swing_type.float()
    dot = sign * torch.sum(blade_normal_world(cmd) * cmd.n_target_world, dim=-1)
    cos_dist = (1.0 - dot).clamp(min=0.0)
    return torch.exp(-torch.square(cos_dist) / (std**2)) * _strike_gate(cmd)


def goal_orientation_pre_strike(
    env: "ManagerBasedRLEnv", command_name: str, std: float = 0.3, ramp_time: float = 0.2
) -> torch.Tensor:
    # Continuous wind-up guidance: encourage the active paddle face (signed by
    # swing_type) to align with n_target in the ramp_time window before strike.
    # Pairs with goal_position_pre_strike / goal_velocity_pre_strike to build
    # a pre-strike trajectory shaping signal that survives the strike window
    # being only 1-2 frames. Currently disabled in env_cfg (rolled back as
    # part of C1) but kept signed so it stays consistent if re-enabled.
    cmd = _command(env, command_name)
    sign = 1.0 - 2.0 * cmd.swing_type.float()
    dot = sign * torch.sum(blade_normal_world(cmd) * cmd.n_target_world, dim=-1)
    cos_dist = (1.0 - dot).clamp(min=0.0)
    gate = ((cmd.t_to_hit > 0.0) & (cmd.t_to_hit < ramp_time)).float()
    return torch.exp(-torch.square(cos_dist) / (std**2)) * gate


def goal_base_position(env: "ManagerBasedRLEnv", command_name: str, std: float = 0.3) -> torch.Tensor:
    cmd = _command(env, command_name)
    err = torch.sum(torch.square(cmd.robot.data.root_pos_w[:, :2] - cmd.p_base_xy_world), dim=-1)
    # DEBUG: stash err so curriculum can log it
    cmd._goal_base_err_diag = err.detach()
    cmd._goal_base_gate_diag = (cmd.t_to_hit > 0.0).float().detach()
    cmd._goal_base_root_x_diag = cmd.robot.data.root_pos_w[:, 0].detach()
    cmd._goal_base_root_y_diag = cmd.robot.data.root_pos_w[:, 1].detach()
    cmd._goal_base_target_x_diag = cmd.p_base_xy_world[:, 0].detach()
    cmd._goal_base_target_y_diag = cmd.p_base_xy_world[:, 1].detach()
    cmd._goal_base_phit_x_diag = cmd.p_hit_world[:, 0].detach()
    cmd._goal_base_phit_y_diag = cmd.p_hit_world[:, 1].detach()
    # KEY: per-env delta — if root-target are coordinate-aligned, delta should be SMALL
    # (~0.5m offset). If misaligned (one is world, other is env-local), delta absmax
    # tracks env grid scale (~100m).
    delta_xy = cmd.robot.data.root_pos_w[:, :2] - cmd.p_base_xy_world
    cmd._goal_base_delta_x_absmax_diag = delta_xy[:, 0].abs().max().detach().unsqueeze(0)
    cmd._goal_base_delta_y_absmax_diag = delta_xy[:, 1].abs().max().detach().unsqueeze(0)
    cmd._goal_base_delta_x_mean_diag = delta_xy[:, 0].mean().detach().unsqueeze(0)
    cmd._goal_base_delta_y_mean_diag = delta_xy[:, 1].mean().detach().unsqueeze(0)
    # CRITICAL: per-axis absmax to localize the bug
    # If root_y_absmax ≈ 126: data.root_pos_w[:, 1] is WORLD frame (per-env)
    # If root_y_absmax ≈ 0.5: data.root_pos_w[:, 1] is ENV-LOCAL (constant across envs)
    cmd._goal_base_root_x_absmax_diag = cmd.robot.data.root_pos_w[:, 0].abs().max().detach().unsqueeze(0)
    cmd._goal_base_root_y_absmax_diag = cmd.robot.data.root_pos_w[:, 1].abs().max().detach().unsqueeze(0)
    cmd._goal_base_target_y_absmax_diag = cmd.p_base_xy_world[:, 1].abs().max().detach().unsqueeze(0)
    cmd._goal_base_phit_y_absmax_diag = cmd.p_hit_world[:, 1].abs().max().detach().unsqueeze(0)
    # Per-env env_origin diagnostic — to confirm if envs ARE on a grid
    env_origins = env.scene.env_origins
    cmd._goal_base_env_origin_x_absmax_diag = env_origins[:, 0].abs().max().detach().unsqueeze(0)
    cmd._goal_base_env_origin_y_absmax_diag = env_origins[:, 1].abs().max().detach().unsqueeze(0)
    return torch.exp(-err / (std**2)) * (cmd.t_to_hit > 0.0).float()


def goal_base_orientation(
    env: "ManagerBasedRLEnv", command_name: str, std: float = 0.3
) -> torch.Tensor:
    """Reward base yaw facing +X (table center direction).

    Pre-strike gated. Std in radians. Anchors base orientation to face the
    table so the policy must move LATERALLY (xy translation) instead of
    rotating its body to cover left/right hit points — pairs with swing-first
    cmd sampling to break the backhand mode-cheat (run 14-54-15 fh_share=0.003).
    """
    cmd = _command(env, command_name)
    yaw = yaw_from_wxyz(cmd.robot.data.root_quat_w)
    err = torch.square(yaw)
    return torch.exp(-err / (std**2)) * (cmd.t_to_hit > 0.0).float()


def pelvis_orientation_l2(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=-1)


def feet_air_time_no_command(env: "ManagerBasedRLEnv", threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    return torch.clamp(torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0], max=threshold)


def robot_table_contact_penalty(env: "ManagerBasedRLEnv", threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.force_matrix_w_history
    if forces is not None:
        if not isinstance(sensor_cfg.body_ids, slice):
            forces = forces[:, :, sensor_cfg.body_ids]
        force_norm = torch.linalg.norm(forces, dim=-1).amax(dim=1).amax(dim=-1)
        return torch.sum(force_norm > threshold, dim=-1).float()

    net_forces = contact_sensor.data.net_forces_w_history
    if not isinstance(sensor_cfg.body_ids, slice):
        net_forces = net_forces[:, :, sensor_cfg.body_ids]
    force_norm = torch.linalg.norm(net_forces, dim=-1).amax(dim=1)
    return torch.sum(force_norm > threshold, dim=-1).float()


def pre_strike_feet_gait(
    env: "ManagerBasedRLEnv",
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    offset: list[float] | tuple[float, float] = (0.0, 0.5),
    threshold: float = 0.55,
    move_threshold: float = 0.06,
    settle_time: float = 0.15,
) -> torch.Tensor:
    """Reward a single gait cycle while the robot is approaching the strike.

    The clock is one-shot: it advances from 0 to 1 over ``t_pre_initial`` for the
    current planner command and becomes inactive when the command is no longer in
    the pre-strike phase. Post-strike stability is still handled by the existing
    no-strike foot regularizers.
    """
    cmd = _command(env, command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0

    denom = torch.clamp(cmd.t_pre_initial, min=1.0e-6)
    global_phase = torch.clamp((cmd.t_pre_initial - cmd.t_to_hit) / denom, 0.0, 1.0).unsqueeze(-1)
    phases = [torch.remainder(global_phase + float(offset_), 1.0) for offset_ in offset]
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    base_err = cmd.p_base_xy_world - cmd.robot.data.root_pos_w[:, :2]
    move_needed = torch.linalg.norm(base_err, dim=-1) > move_threshold
    gate = (cmd.t_to_hit > settle_time) & move_needed
    return reward * gate.float()


# ---------------------------------------------------------------------------
# Lower-body (leg) recovery regularizers (ported from locomotion).
#
# Rationale: pingpong's imitation set is upper-body only (waist + arms), and no
# reward anchors the legs. With only pelvis height/orientation constraints, the
# policy adopts a free "idle" posture between hits — lifting a leg out to the
# side as a swing counterweight, single-leg standing, fore-aft sway. The legs
# are unconstrained and an airborne foot at rest costs nothing.
#
# These two functions re-gate locomotion's leg terms onto the pingpong
# `t_to_hit` signal: in locomotion they fire when there is "no base_velocity
# command" (cmd_norm < 0.1, i.e. not walking); the pingpong analogue of "not
# actively moving to a target" is "not approaching a hit" = t_to_hit <= 0
# (post-strike / waiting for the next swing). Gating — rather than always-on —
# is deliberate and matches locomotion: when the robot IS approaching a hit
# (t_to_hit > 0) it must be free to lift a foot and step laterally to reach the
# commanded base position, so these constraints switch OFF then.
#
# The hip_roll/hip_yaw deviation term is left always-on (registered in env_cfg
# directly via the shared `joint_deviation_l1`, matching locomotion's
# always-on joint_deviation_legs); only the two functions below need the
# pingpong-specific t_to_hit gate.
# ---------------------------------------------------------------------------


def feet_contact_no_strike(
    env: "ManagerBasedRLEnv", command_name: str, sensor_cfg: SceneEntityCfg, t_thresh: float = 0.0
) -> torch.Tensor:
    """Reward both feet being in contact while the robot is NOT approaching a hit.

    Pingpong port of locomotion `feet_contact_without_cmd`: the no-command gate
    (cmd_norm < 0.1) is replaced by the post-strike / waiting gate t_to_hit <=
    t_thresh. Kills the single-leg idle stance observed post-strike. Gated (not
    always-on) so that during the approach (t_to_hit > 0) the policy can still
    lift a foot to step laterally toward the commanded hit position.
    """
    cmd = _command(env, command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    gate = (cmd.t_to_hit <= t_thresh).float()
    return torch.sum(is_contact, dim=-1).float() * gate


def feet_distance_no_strike(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg,
    nominal: float = 0.20,
    wide_scale: float = 0.3,
    t_thresh: float = 0.0,
) -> torch.Tensor:
    """Penalize stance-width deviation from `nominal` while NOT approaching a hit.

    Asymmetric (user request): crossing the legs (feet too close, `deficit`) is
    penalized at full weight; spreading the legs (feet too wide, `excess`) is
    scaled down by `wide_scale` — a wide stance helps balance / squatting to
    reach low balls, so it should be only mildly discouraged. Returns a positive
    penalty magnitude; the RewTerm weight (negative) sets the overall scale.
    Gated to the post-strike / waiting window (t_to_hit <= t_thresh) like
    feet_contact_no_strike.
    """
    cmd = _command(env, command_name)
    asset = env.scene[asset_cfg.name]
    feet = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    dist = torch.linalg.norm(feet[:, 0] - feet[:, 1], dim=-1)
    deficit = (nominal - dist).clamp(min=0.0)
    excess = (dist - nominal).clamp(min=0.0)
    gate = (cmd.t_to_hit <= t_thresh).float()
    return (deficit + wide_scale * excess) * gate
