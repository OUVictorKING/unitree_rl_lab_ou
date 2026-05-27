from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def flat_pitch_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize pitch only (forward/backward lean) of the base.

    Uses the x-component of `projected_gravity_b` (body frame): for a body-y
    rotation (pitch) by angle theta, gravity rotates into the body-x axis as
    `gx = sin(theta)`. Squaring gives a standard L2 pitch penalty that leaves
    roll (gy) untouched — penguin waddle's left/right roll is style, not
    compensation, so it must not be penalized here.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.projected_gravity_b[:, 0])


def stand_still_on_ground(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
    joint_vel_sigma: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Positive reward for standing still when command ≈ 0.

    Reward fires only when **all three** conditions hold per env:
      1. `||cmd||_2 < command_threshold` — user intent is "don't move"
      2. every body in `sensor_cfg.body_ids` is currently in contact — double
         stance, so the robot is mechanically stable
      3. joint velocities are small — measured by an RBF kernel
         `exp(-|qvel|² / σ²)` over all joints; σ=1 rad/s means ~0.8 at
         0.1 rad/s/joint and ~0 at 0.5 rad/s/joint

    When any of (1)-(2) fails the reward is 0; the RBF term from (3) provides
    the shaping gradient within the active window. Used with a positive
    weight on the RewTerm, this trains the policy to hold still on both feet
    under zero command without fighting the AMP style reward (which is
    silenced indirectly by the policy simply not moving).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    feet_mask = is_contact.all(dim=-1).float()

    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    cmd_mask = (cmd_norm < command_threshold).float()

    qvel_sq = torch.sum(torch.square(asset.data.joint_vel), dim=1)
    stillness = torch.exp(-qvel_sq / (joint_vel_sigma**2))

    return stillness * feet_mask * cmd_mask


def feet_flat_orientation_l2(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
) -> torch.Tensor:
    """Penalize feet not being flat, gated on ground contact.

    Project world-down gravity into each foot's body frame; xy components
    encode foot tilt. Only penalize while the foot is in contact — during
    swing phase we don't care about foot attitude.

    Orthogonal to AMP: foot orientation is NOT in the V2/V3 AmpObsSpec
    (include_feet_orientation=False), so this can't fight the expert.
    """
    from isaaclab.utils.math import quat_rotate_inverse

    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    foot_quat = asset.data.body_quat_w[:, asset_cfg.body_ids]  # (N, F, 4)
    N, F, _ = foot_quat.shape
    gravity_w = torch.tensor([0.0, 0.0, -1.0], device=asset.device).expand(N, F, 3)
    grav_f = quat_rotate_inverse(
        foot_quat.reshape(-1, 4), gravity_w.reshape(-1, 3)
    ).view(N, F, 3)
    tilt_sq = grav_f[..., 0] ** 2 + grav_f[..., 1] ** 2  # (N, F)

    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    return torch.sum(tilt_sq * in_contact.float(), dim=-1)
