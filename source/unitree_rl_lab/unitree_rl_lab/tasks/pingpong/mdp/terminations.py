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


def body_table_contact_sustained(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 3.0,
    duration_s: float = 0.3,
) -> torch.Tensor:
    """Terminate when any filtered body has sustained table contact for >= duration_s.

    Maintains a per-env counter on the env object, keyed by sensor name. The counter
    increments each step a filtered body is in contact above force_threshold and
    resets to zero otherwise. Stale counters from previous episodes are cleared when
    the counter exceeds the current episode length (i.e. an env reset happened).

    Use with a sensor whose `filter_prim_paths_expr` is set to the table prim, and
    `body_names` restricted to the bodies you want to forbid (typically all bodies
    EXCEPT the paddle blade and rubber hand, since the paddle is allowed near the
    table during a strike).

    Stage-aware: when the table-guard curriculum keeps the table hidden (Stage 0/1),
    ``env._pingpong_table_active`` is False and this termination short-circuits
    to all-zeros so the policy isn't terminated for impossible contacts during
    the swing-learning phase.
    """
    if not bool(getattr(env, "_pingpong_table_active", False)):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Prefer filter_matrix (only counts forces against the filter prim, e.g. Table).
    forces = contact_sensor.data.force_matrix_w_history
    if forces is not None:
        if not isinstance(sensor_cfg.body_ids, slice):
            forces = forces[:, :, sensor_cfg.body_ids]
        # forces: (N, T, B, F, 3) -> norm -> (N, T, B, F) -> amax over T then F -> (N, B)
        force_norm = torch.linalg.norm(forces, dim=-1).amax(dim=1).amax(dim=-1)
    else:
        net = contact_sensor.data.net_forces_w_history
        if not isinstance(sensor_cfg.body_ids, slice):
            net = net[:, :, sensor_cfg.body_ids]
        force_norm = torch.linalg.norm(net, dim=-1).amax(dim=1)  # (N, B)

    in_contact = (force_norm > force_threshold).any(dim=-1)  # (N,)

    buf_name = f"_pingpong_table_stuck_{sensor_cfg.name}"
    buf = getattr(env, buf_name, None)
    if buf is None or buf.shape[0] != env.num_envs:
        buf = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)

    # Clear stale counters from previous episodes: if buf > episode_length_buf, the
    # env reset since we last incremented it.
    if hasattr(env, "episode_length_buf"):
        ep_len = env.episode_length_buf.to(buf.dtype)
        buf = torch.where(buf > ep_len, torch.zeros_like(buf), buf)

    new_buf = torch.where(in_contact, buf + 1, torch.zeros_like(buf))
    setattr(env, buf_name, new_buf)

    step_dt = float(getattr(env, "step_dt", env.cfg.sim.dt * env.cfg.decimation))
    duration_steps = max(1, int(round(duration_s / step_dt)))
    return new_buf >= duration_steps
