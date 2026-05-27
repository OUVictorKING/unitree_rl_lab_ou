from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from unitree_rl_lab.tasks.mimic.mdp.events import randomize_joint_default_pos, randomize_rigid_body_com

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = [
    "randomize_joint_default_pos",
    "randomize_rigid_body_com",
    "randomize_imu_offset",
    "randomize_comm_delay",
    "get_imu_offset_quat",
    "get_obs_delay_steps",
]


_IMU_OFFSET_ATTR = "_pingpong_imu_offset_quat"
_OBS_DELAY_ATTR = "_pingpong_obs_delay_steps"


def get_imu_offset_quat(env: "ManagerBasedEnv", asset_name: str = "robot") -> torch.Tensor:
    """Return per-env IMU offset quaternion (w, x, y, z); identity if event not configured."""
    q = getattr(env, _IMU_OFFSET_ATTR, None)
    if q is None:
        asset = env.scene[asset_name]
        n = env.scene.num_envs
        q = torch.zeros(n, 4, device=asset.device)
        q[:, 0] = 1.0
        setattr(env, _IMU_OFFSET_ATTR, q)
    return q


def get_obs_delay_steps(env: "ManagerBasedEnv") -> torch.Tensor:
    """Return per-env obs delay step count (>=0); zeros if event not configured."""
    d = getattr(env, _OBS_DELAY_ATTR, None)
    if d is None:
        n = env.scene.num_envs
        d = torch.zeros(n, dtype=torch.long, device=env.device)
        setattr(env, _OBS_DELAY_ATTR, d)
    return d


def randomize_imu_offset(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    sigma_deg: float = 2.0,
    distribution: str = "gaussian",
) -> None:
    """Sample per-env quaternion offset simulating base-IMU calibration error.

    Stored on ``env._pingpong_imu_offset_quat`` (num_envs, 4) in (w, x, y, z) layout.
    Wrapper observation functions in ``observations.py`` rotate the perceived
    base quat through this offset; sensor truth is unchanged.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.device
    n = env.scene.num_envs

    q_buf = get_imu_offset_quat(env, asset_cfg.name)
    if env_ids is None:
        env_ids = torch.arange(n, device=device)

    n_e = int(len(env_ids))
    sigma_rad = math.radians(float(sigma_deg))
    if distribution == "uniform":
        rpy = (torch.rand(n_e, 3, device=device) * 2.0 - 1.0) * sigma_rad
    else:
        rpy = torch.randn(n_e, 3, device=device) * sigma_rad
    q_buf[env_ids] = math_utils.quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])


def randomize_comm_delay(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    max_delay_steps: int = 1,
) -> None:
    """Sample per-env integer obs delay in {0, ..., max_delay_steps}.

    Stored on ``env._pingpong_obs_delay_steps`` (num_envs,). Wrapper obs class
    ``DelayedObservation`` reads this to gate between current and previous step
    output. With control dt=20 ms and ``max_delay_steps=1`` this realises the
    paper's 0–20 ms communication delay range as a per-env binary draw.
    """
    n = env.scene.num_envs
    device = env.device

    d_buf = get_obs_delay_steps(env)
    if env_ids is None:
        env_ids = torch.arange(n, device=device)

    high = max(0, int(max_delay_steps)) + 1
    d_buf[env_ids] = torch.randint(0, high, (int(len(env_ids)),), device=device, dtype=torch.long)
