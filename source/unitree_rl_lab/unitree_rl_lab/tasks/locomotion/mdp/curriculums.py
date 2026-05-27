from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# def lin_vel_cmd_levels(
#     env: ManagerBasedRLEnv,
#     env_ids: Sequence[int],
# ) -> torch.Tensor:
#     """Curriculum for forward/lateral linear velocity.

#     Expand x first. Expand y only after forward tracking is already stable.
#     """

#     command_term = env.command_manager.get_term("base_velocity")
#     ranges = command_term.cfg.ranges
#     limit_ranges = command_term.cfg.limit_ranges

#     if env.common_step_counter % env.max_episode_length != 0:
#         return torch.tensor(ranges.lin_vel_x[1], device=env.device)

#     reward_manager = env.reward_manager

#     track_xy = (
#         torch.mean(reward_manager._episode_sums["track_lin_vel_xy"][env_ids])
#         / env.max_episode_length_s
#     )
#     track_yaw = (
#         torch.mean(reward_manager._episode_sums["track_ang_vel_z"][env_ids])
#         / env.max_episode_length_s
#     )
#     alive = (
#         torch.mean(reward_manager._episode_sums["alive"][env_ids])
#         / env.max_episode_length_s
#     )
#     alive_weight = env.reward_manager.get_term_cfg("alive").weight
#     alive_ratio = alive / max(alive_weight, 1e-6)

#     cur_x = torch.tensor(ranges.lin_vel_x, device=env.device, dtype=torch.float32)
#     cur_y = torch.tensor(ranges.lin_vel_y, device=env.device, dtype=torch.float32)

#     lim_x = torch.tensor(limit_ranges.lin_vel_x, device=env.device, dtype=torch.float32)
#     lim_y = torch.tensor(limit_ranges.lin_vel_y, device=env.device, dtype=torch.float32)

#     x_width = cur_x[1] - cur_x[0]

#     good_xy = track_xy > 1.25
#     good_yaw = track_yaw > 0.55
#     good_survival = alive_ratio > 0.96

#     bad_xy = track_xy < 0.95
#     bad_survival = alive_ratio < 0.90

#     if x_width < 0.5:
#         up_x = torch.tensor([-0.02, 0.04], device=env.device)
#     elif x_width < 0.9:
#         up_x = torch.tensor([-0.01, 0.03], device=env.device)
#     else:
#         up_x = torch.tensor([-0.005, 0.01], device=env.device)

#     # y 只在 x 比较稳之后再扩
#     if cur_x[1] < 0.6:
#         up_y = torch.tensor([0.0, 0.0], device=env.device)
#     else:
#         up_y = torch.tensor([-0.005, 0.005], device=env.device)

#     down_x = torch.tensor([0.01, -0.01], device=env.device)
#     down_y = torch.tensor([0.003, -0.003], device=env.device)

#     if good_xy and good_yaw and good_survival:
#         new_x = torch.clamp(cur_x + up_x, min=lim_x[0], max=lim_x[1])
#         new_y = torch.clamp(cur_y + up_y, min=lim_y[0], max=lim_y[1])

#         ranges.lin_vel_x = (float(new_x[0]), float(new_x[1]))
#         ranges.lin_vel_y = (float(new_y[0]), float(new_y[1]))

#         print(
#             f"[Curriculum][LIN][UP] "
#             f"x={ranges.lin_vel_x}, y={ranges.lin_vel_y}, "
#             f"track_xy={track_xy.item():.3f}, "
#             f"track_yaw={track_yaw.item():.3f}, "
#             f"alive_ratio={alive_ratio.item():.3f}"
#         )

#     elif bad_xy or bad_survival:
#         new_x = torch.clamp(cur_x + down_x, min=lim_x[0], max=lim_x[1])
#         new_y = torch.clamp(cur_y + down_y, min=lim_y[0], max=lim_y[1])

#         ranges.lin_vel_x = (float(new_x[0]), float(new_x[1]))
#         ranges.lin_vel_y = (float(new_y[0]), float(new_y[1]))

#         print(
#             f"[Curriculum][LIN][DOWN] "
#             f"x={ranges.lin_vel_x}, y={ranges.lin_vel_y}, "
#             f"track_xy={track_xy.item():.3f}, "
#             f"track_yaw={track_yaw.item():.3f}, "
#             f"alive_ratio={alive_ratio.item():.3f}"
#         )

#     else:
#         print(
#             f"[Curriculum][LIN][HOLD] "
#             f"x={ranges.lin_vel_x}, y={ranges.lin_vel_y}, "
#             f"track_xy={track_xy.item():.3f}, "
#             f"track_yaw={track_yaw.item():.3f}, "
#             f"alive_ratio={alive_ratio.item():.3f}"
#         )

#     return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = (
        torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids])
        / env.max_episode_length_s
    )

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            # delta_command = torch.tensor([-0.02, 0.02], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = (
        torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids])
        / env.max_episode_length_s
    )

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            # delta_command = torch.tensor([-0.02, 0.02], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


# def ang_vel_cmd_levels(
#     env: ManagerBasedRLEnv,
#     env_ids: Sequence[int],
# ) -> torch.Tensor:
#     """Curriculum for yaw angular velocity.

#     Enable only after forward velocity tracking is already reasonably good.
#     """

#     command_term = env.command_manager.get_term("base_velocity")
#     ranges = command_term.cfg.ranges
#     limit_ranges = command_term.cfg.limit_ranges

#     if env.common_step_counter % env.max_episode_length != 0:
#         return torch.tensor(ranges.ang_vel_z[1], device=env.device)

#     reward_manager = env.reward_manager

#     track_xy = (
#         torch.mean(reward_manager._episode_sums["track_lin_vel_xy"][env_ids])
#         / env.max_episode_length_s
#     )
#     track_yaw = (
#         torch.mean(reward_manager._episode_sums["track_ang_vel_z"][env_ids])
#         / env.max_episode_length_s
#     )
#     alive = (
#         torch.mean(reward_manager._episode_sums["alive"][env_ids])
#         / env.max_episode_length_s
#     )
#     alive_weight = env.reward_manager.get_term_cfg("alive").weight
#     alive_ratio = alive / max(alive_weight, 1e-6)

#     cur_yaw = torch.tensor(ranges.ang_vel_z, device=env.device, dtype=torch.float32)
#     lim_yaw = torch.tensor(
#         limit_ranges.ang_vel_z, device=env.device, dtype=torch.float32
#     )

#     # 只有线速度稳定后，才开始放 yaw
#     if ranges.lin_vel_x[1] < 0.5:
#         print(
#             f"[Curriculum][YAW][HOLD] wait for lin_vel_x, "
#             f"x_max={ranges.lin_vel_x[1]:.3f}"
#         )
#         return torch.tensor(ranges.ang_vel_z[1], device=env.device)

#     good_xy = track_xy > 1.20
#     good_yaw = track_yaw > 0.50
#     good_survival = alive_ratio > 0.96

#     bad_yaw = track_yaw < 0.35
#     bad_survival = alive_ratio < 0.90

#     yaw_width = cur_yaw[1] - cur_yaw[0]
#     if yaw_width < 0.20:
#         up_yaw = torch.tensor([-0.02, 0.02], device=env.device)
#     else:
#         up_yaw = torch.tensor([-0.01, 0.01], device=env.device)

#     down_yaw = torch.tensor([0.01, -0.01], device=env.device)

#     if good_xy and good_yaw and good_survival:
#         new_yaw = torch.clamp(cur_yaw + up_yaw, min=lim_yaw[0], max=lim_yaw[1])
#         ranges.ang_vel_z = (float(new_yaw[0]), float(new_yaw[1]))

#         print(
#             f"[Curriculum][YAW][UP] "
#             f"yaw={ranges.ang_vel_z}, "
#             f"track_xy={track_xy.item():.3f}, "
#             f"track_yaw={track_yaw.item():.3f}, "
#             f"alive_ratio={alive_ratio.item():.3f}"
#         )

#     elif bad_yaw or bad_survival:
#         new_yaw = torch.clamp(cur_yaw + down_yaw, min=lim_yaw[0], max=lim_yaw[1])
#         ranges.ang_vel_z = (float(new_yaw[0]), float(new_yaw[1]))

#         print(
#             f"[Curriculum][YAW][DOWN] "
#             f"yaw={ranges.ang_vel_z}, "
#             f"track_xy={track_xy.item():.3f}, "
#             f"track_yaw={track_yaw.item():.3f}, "
#             f"alive_ratio={alive_ratio.item():.3f}"
#         )

#     else:
#         print(
#             f"[Curriculum][YAW][HOLD] "
#             f"yaw={ranges.ang_vel_z}, "
#             f"track_xy={track_xy.item():.3f}, "
#             f"track_yaw={track_yaw.item():.3f}, "
#             f"alive_ratio={alive_ratio.item():.3f}"
#         )

#     return torch.tensor(ranges.ang_vel_z[1], device=env.device)
