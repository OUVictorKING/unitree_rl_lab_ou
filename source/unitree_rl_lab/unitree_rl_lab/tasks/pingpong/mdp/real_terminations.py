from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .real_commands import RealPingpongCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def real_ball_dead(env: "ManagerBasedRLEnv", command_name: str = "pingpong") -> torch.Tensor:
    cmd: RealPingpongCommand = env.command_manager.get_term(command_name)
    ball_pos = cmd.ball.data.root_pos_w
    env_origins = env.scene.env_origins
    return (
        (ball_pos[:, 2] < cmd.cfg.ball_dead_z)
        | (torch.abs(ball_pos[:, 1] - env_origins[:, 1]) > cmd.cfg.ball_dead_y_abs)
        | (torch.abs(ball_pos[:, 0] - env_origins[:, 0] - cmd.cfg.table_center_x) > cmd.cfg.ball_dead_x_abs)
    )
