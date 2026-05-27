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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _command(env: "ManagerBasedRLEnv", command_name: str) -> PingpongCommand:
    return env.command_manager.get_term(command_name)


def _strike_gate(cmd: PingpongCommand) -> torch.Tensor:
    return (torch.abs(cmd.t_to_hit) <= cmd.cfg.strike_window).float()


def imitation_joint_pos(env: "ManagerBasedRLEnv", command_name: str, k: float = 2.0) -> torch.Tensor:
    cmd = _command(env, command_name)
    ids = cmd.upper_joint_ids
    err = torch.sum(torch.square(cmd.robot.data.joint_pos[:, ids] - cmd.ref_state.joint_pos[:, ids]), dim=-1)
    return torch.exp(-k * err)


def imitation_joint_vel(env: "ManagerBasedRLEnv", command_name: str, k: float = 0.1) -> torch.Tensor:
    cmd = _command(env, command_name)
    ids = cmd.upper_joint_ids
    err = torch.sum(torch.square(cmd.robot.data.joint_vel[:, ids] - cmd.ref_state.joint_vel[:, ids]), dim=-1)
    return torch.exp(-k * err)


def _anchor_relative_pos(body_pos: torch.Tensor, pelvis_pos: torch.Tensor) -> torch.Tensor:
    rel_xy = body_pos[..., :2] - pelvis_pos[:, None, :2]
    z = body_pos[..., 2:3]
    return torch.cat((rel_xy, z), dim=-1)


def imitation_body_pos_anchor_relative(env: "ManagerBasedRLEnv", command_name: str, k: float = 40.0) -> torch.Tensor:
    cmd = _command(env, command_name)
    sim_rel = _anchor_relative_pos(cmd.robot_tracked_body_pos_w, cmd.robot_pelvis_pos_w)
    ref_rel = _anchor_relative_pos(cmd.ref_state.body_pos_w, cmd.ref_state.pelvis_pos_w)
    err = torch.sum(torch.square(sim_rel - ref_rel), dim=(1, 2))
    return torch.exp(-k * err)


def goal_position(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    cmd = _command(env, command_name)
    root_pos = cmd.robot.data.root_pos_w
    root_quat = cmd.robot.data.root_quat_w
    p_blade_b = quat_apply_inverse(root_quat, cmd.robot_blade_pos_w - root_pos)
    p_hit_b = quat_apply_inverse(root_quat, cmd.p_hit_world - root_pos)
    err = torch.sum(torch.square(p_blade_b - p_hit_b), dim=-1)
    return torch.exp(-err / (cmd.cfg.sigma_g_pos**2)) * _strike_gate(cmd)


def goal_velocity(env: "ManagerBasedRLEnv", command_name: str, std: float = 0.5) -> torch.Tensor:
    cmd = _command(env, command_name)
    root_quat = cmd.robot.data.root_quat_w
    v_blade_b = quat_apply_inverse(root_quat, cmd.robot_blade_lin_vel_w)
    v_hat_b = quat_apply_inverse(root_quat, cmd.v_racket_hat_world)
    err = torch.sum(torch.square(v_blade_b - v_hat_b), dim=-1)
    return torch.exp(-err / (std**2)) * _strike_gate(cmd)


def blade_normal_world(cmd: PingpongCommand) -> torch.Tensor:
    normal_local = torch.tensor(BLADE_NORMAL_LOCAL, dtype=torch.float32, device=cmd.device).expand(cmd.num_envs, 3)
    return quat_apply(cmd.robot_blade_quat_w, normal_local)


def goal_orientation(env: "ManagerBasedRLEnv", command_name: str, std: float = 0.2) -> torch.Tensor:
    cmd = _command(env, command_name)
    dot = torch.sum(blade_normal_world(cmd) * cmd.n_target_world, dim=-1).clamp(-1.0, 1.0)
    cos_dist = 1.0 - dot
    return torch.exp(-torch.square(cos_dist) / (std**2)) * _strike_gate(cmd)


def goal_base_position(env: "ManagerBasedRLEnv", command_name: str, std: float = 0.3) -> torch.Tensor:
    cmd = _command(env, command_name)
    err = torch.sum(torch.square(cmd.robot.data.root_pos_w[:, :2] - cmd.p_base_xy_world), dim=-1)
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
