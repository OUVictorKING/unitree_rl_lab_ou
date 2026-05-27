from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from .commands import PingpongCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def update_pingpong_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    command_name: str = "pingpong",
    enable_noise: bool = False,
    enable_range: bool = False,
) -> dict[str, float]:
    """Update v5.7 pingpong curricula from the command success metric."""
    command: PingpongCommand = env.command_manager.get_term(command_name)
    command.finalize_partial_swings(env_ids)
    if isinstance(env_ids, slice):
        metric_ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, torch.Tensor):
        metric_ids = env_ids.to(device=env.device, dtype=torch.long)
    else:
        metric_ids = torch.tensor(env_ids, dtype=torch.long, device=env.device)
    success_rate = float(torch.mean(command.metrics["hit_success_rate"][metric_ids]).item())

    if success_rate >= 0.80:
        sigma_target = 0.02
    elif success_rate >= 0.65:
        sigma_target = 0.03
    elif success_rate >= 0.50:
        sigma_target = 0.04
    elif success_rate >= 0.30:
        sigma_target = 0.06
    else:
        sigma_target = 0.10
    command.cfg.sigma_g_pos = min(command.cfg.sigma_g_pos, sigma_target)

    if enable_noise:
        if success_rate >= 0.50:
            command.cfg.noise_t_sigma = max(command.cfg.noise_t_sigma, 0.005)
        if success_rate >= 0.75:
            command.cfg.noise_p_sigma = max(command.cfg.noise_p_sigma, 0.005)
            command.cfg.noise_v_sigma = max(command.cfg.noise_v_sigma, 0.05)
            command.cfg.noise_base_sigma = max(command.cfg.noise_base_sigma, 0.015)

    if enable_range:
        if success_rate >= 0.75:
            command.cfg.hit_y_range = (-0.65, 0.65)
            command.cfg.hit_z_range = (0.85, 1.25)
            command.cfg.v_in_mag_range = (2.0, 5.5)
        elif success_rate >= 0.50:
            command.cfg.hit_y_range = (-0.35, 0.55)
            command.cfg.hit_z_range = (0.88, 1.22)
            command.cfg.v_in_mag_range = (2.0, 5.0)
        elif success_rate >= 0.30:
            command.cfg.hit_y_range = (-0.15, 0.35)
            command.cfg.hit_z_range = (0.92, 1.18)
            command.cfg.v_in_mag_range = (2.0, 4.5)

    return {
        "hit_success_rate": success_rate,
        "sigma_g_pos": float(command.cfg.sigma_g_pos),
        "noise_p_sigma": float(command.cfg.noise_p_sigma),
        "noise_v_sigma": float(command.cfg.noise_v_sigma),
        "noise_base_sigma": float(command.cfg.noise_base_sigma),
        "noise_t_sigma": float(command.cfg.noise_t_sigma),
        "hit_y_max": float(command.cfg.hit_y_range[1]),
        "hit_z_low": float(command.cfg.hit_z_range[0]),
        "v_in_mag_high": float(command.cfg.v_in_mag_range[1]),
    }
