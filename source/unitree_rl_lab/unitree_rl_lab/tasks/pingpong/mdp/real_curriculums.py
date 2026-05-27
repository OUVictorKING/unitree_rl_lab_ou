from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from .real_commands import RealPingpongCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def update_real_pingpong_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    command_name: str = "pingpong",
    enable_stage_updates: bool = True,
) -> dict[str, float]:
    """Stage curriculum for the real-rally task.

    Stage changes are intentionally conservative and monotonic. The command term
    owns the actual sampling ranges so this function can be used by IsaacLab's
    curriculum manager without requiring a custom environment subclass.
    """
    command: RealPingpongCommand = env.command_manager.get_term(command_name)
    command.finalize_partial_swings(env_ids)
    if isinstance(env_ids, slice):
        metric_ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, torch.Tensor):
        metric_ids = env_ids.to(device=env.device, dtype=torch.long)
    else:
        metric_ids = torch.tensor(env_ids, dtype=torch.long, device=env.device)

    hit_rate = float(torch.mean(command.metrics["real_hit_success_rate"][metric_ids]).item())
    return_rate = float(torch.mean(command.metrics["real_return_success_rate"][metric_ids]).item())
    target_rate = float(torch.mean(command.metrics["real_target_success_rate"][metric_ids]).item())

    if enable_stage_updates:
        if target_rate > 0.35 and return_rate > 0.55:
            command.cfg.target_land_radius = min(command.cfg.target_land_radius, 0.30)
            command.cfg.serve_pos_y_range = (-0.75, 0.75)
            command.cfg.serve_hit_y_range = (-0.75, 0.75)
            command.cfg.serve_t_to_hit_range = (0.35, 0.90)
        elif return_rate > 0.45:
            command.cfg.target_land_radius = min(command.cfg.target_land_radius, 0.35)
            command.cfg.serve_pos_y_range = (-0.65, 0.65)
            command.cfg.serve_hit_y_range = (-0.75, 0.75)
            command.cfg.serve_t_to_hit_range = (0.40, 0.90)
        elif hit_rate > 0.60:
            command.cfg.target_land_radius = min(command.cfg.target_land_radius, 0.45)
            command.cfg.serve_pos_y_range = (-0.35, 0.35)
            command.cfg.serve_hit_y_range = (-0.50, 0.50)

    return {
        "real_hit_success_rate": hit_rate,
        "real_return_success_rate": return_rate,
        "real_target_success_rate": target_rate,
        "target_land_radius": float(command.cfg.target_land_radius),
        "serve_y_max": float(command.cfg.serve_pos_y_range[1]),
        "serve_hit_y_max": float(command.cfg.serve_hit_y_range[1]),
        "serve_t_low": float(command.cfg.serve_t_to_hit_range[0]),
    }
