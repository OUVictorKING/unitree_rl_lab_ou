from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def hard_undesired_contact(env: "ManagerBasedRLEnv", threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history
    if not isinstance(sensor_cfg.body_ids, slice):
        forces = forces[:, :, sensor_cfg.body_ids]
    force_norm = torch.linalg.norm(forces, dim=-1).amax(dim=1)
    return torch.any(force_norm > threshold, dim=-1)
