from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .real_commands import RealPingpongCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _command(env: "ManagerBasedRLEnv", command_name: str) -> RealPingpongCommand:
    return env.command_manager.get_term(command_name)


def real_ball_contact(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    return _command(env, command_name)._reward_ball_contact


def real_return_direction(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    return _command(env, command_name)._reward_return_direction


def real_clear_net(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    return _command(env, command_name)._reward_clear_net


def real_opponent_land(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    return _command(env, command_name)._reward_opponent_land


def real_target_land(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    return _command(env, command_name)._reward_target_land


def real_illegal(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    return _command(env, command_name)._reward_illegal
