from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg

try:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul
except ImportError:  # pragma: no cover - older IsaacLab fallback
    from isaaclab.utils.math import (
        quat_apply,
        quat_mul,
        quat_rotate_inverse as quat_apply_inverse,
    )

from .commands import PingpongCommand
from .events import get_imu_offset_quat, get_obs_delay_steps
from .motion_loader import yaw_from_wxyz

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def _command(env: "ManagerBasedEnv", command_name: str) -> PingpongCommand:
    return env.command_manager.get_term(command_name)


def _perceived_root_quat(
    env: "ManagerBasedEnv", asset_name: str = "robot"
) -> torch.Tensor:
    """Return root_quat_w pre-multiplied by per-env IMU offset (identity if event off)."""
    asset: RigidObject = env.scene[asset_name]
    q_true = asset.data.root_quat_w
    q_offset = get_imu_offset_quat(env, asset_name)
    return quat_mul(q_true, q_offset)


def base_yaw_encoding(
    env: "ManagerBasedEnv", asset_cfg_name: str = "robot"
) -> torch.Tensor:
    """Clean base yaw cos/sin encoding (used by Critic / clean reward computation)."""
    asset: RigidObject = env.scene[asset_cfg_name]
    yaw = yaw_from_wxyz(asset.data.root_quat_w)
    return torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)


def base_yaw_encoding_imu(
    env: "ManagerBasedEnv", asset_cfg_name: str = "robot"
) -> torch.Tensor:
    """Base yaw cos/sin encoding through perceived (IMU-offset) root quat for Actor."""
    yaw = yaw_from_wxyz(_perceived_root_quat(env, asset_cfg_name))
    return torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)


def base_ang_vel_imu(
    env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Body-frame ang vel rotated through IMU calibration offset."""
    asset: RigidObject = env.scene[asset_cfg.name]
    q_offset = get_imu_offset_quat(env, asset_cfg.name)
    return quat_apply_inverse(q_offset, asset.data.root_ang_vel_b)


def projected_gravity_imu(
    env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Gravity projected into the perceived (offset-rotated) body frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    q_perc = _perceived_root_quat(env, asset_cfg.name)
    return quat_apply_inverse(q_perc, asset.data.GRAVITY_VEC_W)


def pingpong_base_position_error(
    env: "ManagerBasedEnv", command_name: str, noisy: bool = False
) -> torch.Tensor:
    cmd = _command(env, command_name)
    target = cmd.p_base_xy_world
    if noisy:
        target = target + cmd.noise_base
    return target - cmd.robot.data.root_pos_w[:, :2]


def pingpong_hit_position_b(
    env: "ManagerBasedEnv", command_name: str, noisy: bool = False
) -> torch.Tensor:
    cmd = _command(env, command_name)
    target = cmd.p_hit_world
    if noisy:
        target = target + cmd.noise_p
    return quat_apply_inverse(
        cmd.robot.data.root_quat_w, target - cmd.robot.data.root_pos_w
    )


def pingpong_racket_velocity_w(
    env: "ManagerBasedEnv", command_name: str, noisy: bool = False
) -> torch.Tensor:
    cmd = _command(env, command_name)
    vel = cmd.v_racket_hat_world
    if noisy:
        vel = vel + cmd.noise_v
    return vel


def pingpong_t_to_hit(
    env: "ManagerBasedEnv", command_name: str, noisy: bool = False
) -> torch.Tensor:
    cmd = _command(env, command_name)
    t = cmd.t_to_hit.unsqueeze(-1)
    if noisy:
        t = t + cmd.noise_t
    return t


def pingpong_ref_body_state(env: "ManagerBasedEnv", command_name: str) -> torch.Tensor:
    cmd = _command(env, command_name)
    return torch.cat(
        (cmd.ref_state.body_pos_w, cmd.ref_state.body_quat_w), dim=-1
    ).reshape(env.num_envs, -1)


def pingpong_ref_joint_state(env: "ManagerBasedEnv", command_name: str) -> torch.Tensor:
    cmd = _command(env, command_name)
    return torch.cat((cmd.ref_state.joint_pos, cmd.ref_state.joint_vel), dim=-1)


def episode_time_left(env: "ManagerBasedRLEnv") -> torch.Tensor:
    remaining = (env.max_episode_length - env.episode_length_buf).to(
        torch.float32
    ) * env.step_dt
    return remaining.unsqueeze(-1)


_INNER_FUNC_REGISTRY: dict[str, Callable[..., torch.Tensor]] = {}


def register_delayable_func(
    func: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """Decorator-friendly registration of inner observation callables for ``DelayedObservation``.

    The wrapper resolves ``inner_func`` by name to keep ObservationTermCfg.params
    JSON/yaml-serializable.
    """
    _INNER_FUNC_REGISTRY[func.__name__] = func
    return func


def _resolve_inner_func(inner_func: Any) -> Callable[..., torch.Tensor]:
    if callable(inner_func):
        return inner_func
    if isinstance(inner_func, str) and inner_func in _INNER_FUNC_REGISTRY:
        return _INNER_FUNC_REGISTRY[inner_func]
    raise ValueError(
        f"DelayedObservation.inner_func must be callable or a registered name, got {inner_func!r}"
    )


class DelayedObservation(ManagerTermBase):
    """Observation wrapper that returns the previous-step value for envs whose
    ``env._pingpong_obs_delay_steps`` is > 0, simulating 0–20 ms communication delay.

    Configuration::

        ObsTerm(
            func=DelayedObservation,
            params={
                "inner_func": joint_pos_rel,                # callable returning (num_envs, D)
                "inner_params": {"asset_cfg": SceneEntityCfg("robot")},
            },
        )
    """

    def __init__(self, cfg: ObservationTermCfg, env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        params = cfg.params or {}
        inner_func = params.get("inner_func")
        self._inner_func = _resolve_inner_func(inner_func)
        self._inner_params: dict[str, Any] = dict(params.get("inner_params") or {})
        self._buffer: torch.Tensor | None = None

    def reset(self, env_ids: torch.Tensor | None = None) -> None:  # type: ignore[override]
        if self._buffer is None or env_ids is None:
            self._buffer = None
            return
        self._buffer[env_ids] = 0.0

    # def __call__(self, env: "ManagerBasedEnv", **kwargs) -> torch.Tensor:  # type: ignore[override]
    #     merged = {**self._inner_params, **{k: v for k, v in kwargs.items() if k not in ("inner_func", "inner_params")}}
    #     cur = self._inner_func(env, **merged)
    def __call__(
        self,
        env: "ManagerBasedEnv",
        inner_func: Any = None,
        inner_params: dict | None = None,
    ) -> torch.Tensor:  # type: ignore[override]
        cur = self._inner_func(env, **self._inner_params)
        delay = get_obs_delay_steps(env)
        if self._buffer is None or self._buffer.shape != cur.shape:
            self._buffer = cur.detach().clone()
        delay_mask = (delay > 0).view(-1, *([1] * (cur.dim() - 1))).expand_as(cur)
        out = torch.where(delay_mask, self._buffer, cur)
        self._buffer = cur.detach().clone()
        return out


for _f in (
    base_ang_vel_imu,
    projected_gravity_imu,
    base_yaw_encoding,
    base_yaw_encoding_imu,
    pingpong_base_position_error,
    pingpong_hit_position_b,
    pingpong_racket_velocity_w,
    pingpong_t_to_hit,
):
    register_delayable_func(_f)
