# Copyright (c) 2025, Unitree Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""Unitree AMP environment (Penguin V1).

This module provides :class:`UnitreeAmpEnv`, a subclass of
:class:`~isaaclab.envs.ManagerBasedRLEnv` that integrates the unified AMP
feature pipeline:

- Maintains a per-env AMP K-frame circular buffer shaped ``(N, K, D)``.
- Injects a stacked AMP observation ``obs["amp"]`` into the observation dict.
- Captures the **pre-reset** stacked AMP obs for terminated envs and exposes
  it through ``extras["amp"]["terminal_next_amp_obs"]`` so the AMP algorithm
  can build clean ``(amp_obs_t, amp_obs_{t+1})`` transitions even across
  episode boundaries.
- Implements the Phase-1 **motion-reset** strategy: resampled root + joint
  state from an expert clip plus small noise, XY zeroing, and yaw
  randomization. Warmup fill policy for the K-buffer is configurable;
  Phase-1 default is *repeat the reset frame K times*.

The parent :meth:`ManagerBasedRLEnv.step` auto-resets done envs before
returning the next observation. We preserve that behavior for the policy/
critic obs groups (they keep the post-reset semantics) but maintain the AMP
buffer manually so the stored ``next_amp_obs`` for done envs is the **real**
pre-reset feature, not the post-reset one.

The env's :class:`AmpObsSpec`, joint/body name lists, and default joint pose
are published on ``env.unwrapped`` for :meth:`AmpPPO.construct_algorithm`.
"""

from __future__ import annotations

import dataclasses
import math
import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from unitree_rl_lab.rsl_rl_amp.features import (
    AmpObsSpec,
    AmpObsState,
    build_amp_frame_from_state,
    build_amp_window,
    concat_frame_history,
    resolve_indices,
    resolve_spec_indices,
)
from unitree_rl_lab.rsl_rl_amp.storage.motion_dataset import MotionDataset

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


# ---------------------------------------------------------------------------
# Reset-strategy cfg
# ---------------------------------------------------------------------------

@configclass
class UnitreeAmpResetCfg:
    """Phase-1 motion-reset strategy parameters.

    Attributes
    ----------
    use_motion_reset:
        If True (Phase-1 default), on env reset we sample one expert frame
        per env from the motion dataset and write its root + joint state to
        the simulation.  If False, the env falls back to the standard event-
        manager reset pipeline.
    joint_pos_noise:
        Gaussian std (rad) added to the sampled joint positions at reset.
    joint_vel_noise:
        Gaussian std (rad/s) added to the sampled joint velocities at reset.
    xy_zero:
        If True, overwrite the sampled world (x,y) with zero so the robot
        spawns at the env origin (plus yaw randomization).  Recommended for
        flat-ground locomotion training.
    yaw_randomize:
        If True, sample a uniform random yaw and rotate the sampled root
        quaternion / linear velocity into it.
    root_lin_vel_noise:
        Gaussian std (m/s) added to the sampled linear velocity.
    root_ang_vel_noise:
        Gaussian std (rad/s) added to the sampled angular velocity.
    kbuf_fill_mode:
        How the AMP K-buffer is seeded at reset:

        - ``"repeat"`` (Phase-1 default): all K slots filled with the reset
          frame's ``phi``.
        - ``"zero"``: zero-fill (useful for ablations; K-1 warmup frames
          disadvantage the discriminator on episode starts).
    """

    use_motion_reset: bool = True
    joint_pos_noise: float = 0.02
    joint_vel_noise: float = 0.10
    xy_zero: bool = True
    yaw_randomize: bool = True
    root_lin_vel_noise: float = 0.10
    root_ang_vel_noise: float = 0.20
    kbuf_fill_mode: str = "repeat"


# ---------------------------------------------------------------------------
# Env cfg base
# ---------------------------------------------------------------------------

@configclass
class UnitreeAmpEnvCfg(ManagerBasedRLEnvCfg):
    """Mixin-style base cfg for AMP-enabled envs.

    Concrete task cfgs (e.g. ``velocity_amp_env_cfg.RobotEnvCfg``) inherit
    this and populate the ``amp_*`` fields in their ``__post_init__``.

    AMP-specific fields
    -------------------
    - ``amp_spec``: :class:`AmpObsSpec` describing the feature layout. Must
      be set by the concrete env cfg's ``__post_init__``; leaving it
      ``None`` at env construction time is a hard error.
    - ``amp_motion_files``: list of ``.npz`` expert clips used for the motion
      reset pool and (downstream) discriminator training.
    - ``amp_wrap_around``: cyclic transition pairs at clip ends.  Phase-1
      default True (penguin is a single periodic gait).
    - ``amp_reset``: :class:`UnitreeAmpResetCfg`.
    """

    amp_spec: AmpObsSpec | None = None
    amp_motion_files: list[str] = dataclasses.field(default_factory=list)
    amp_wrap_around: bool = True
    amp_reset: UnitreeAmpResetCfg = dataclasses.field(default_factory=UnitreeAmpResetCfg)


# ---------------------------------------------------------------------------
# The env subclass
# 继承ManagerBasedRLEnv，重写step()方法，在其中维护AMP K-buffer和motion-reset逻辑
# ---------------------------------------------------------------------------

class UnitreeAmpEnv(ManagerBasedRLEnv):
    """AMP-aware env: K-buffer, terminal fix, motion reset."""

    cfg: UnitreeAmpEnvCfg

    AMP_OBS_KEY: ClassVar[str] = "amp"

    def __init__(self, cfg: UnitreeAmpEnvCfg, render_mode: str | None = None, **kwargs: Any):
        # ``ManagerBasedRLEnv.__init__`` drives scene construction and manager
        # setup; AMP state is initialized after that so we can read the
        # articulation's joint / body name lists.
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        # --- Resolve spec + naming against the live articulation.
        self._robot: "Articulation" = self.scene["robot"]
        art_joint_names = list(self._robot.data.joint_names)
        art_body_names = list(self._robot.data.body_names)

        spec = cfg.amp_spec
        if spec is None:
            raise ValueError(
                "cfg.amp_spec is None — the concrete env cfg's __post_init__ "
                "must construct an AmpObsSpec and assign it before the env is built."
            )
        if not isinstance(spec, AmpObsSpec):
            raise TypeError(f"cfg.amp_spec must be an AmpObsSpec, got {type(spec).__name__}.")

        # Validate + resolve indices once up front — clear errors beat silent drift.
        idx_map = resolve_spec_indices(spec, art_joint_names, art_body_names)
        self.amp_spec: AmpObsSpec = spec
        # Both name lists describe the *articulation* column order — this is
        # what MotionDataset expects as ``amp_joint_names`` / ``amp_body_names``
        # (i.e. the npz column order, which csv_to_npz_final.py also writes
        # from ``robot.data.joint_names`` / ``robot.data.body_names``). The
        # spec's own ``joint_names`` is a separate axis; MotionDataset
        # resolves it via ``resolve_indices(spec.joint_names, amp_joint_names, ...)``.
        self.amp_joint_names: list[str] = art_joint_names
        self.amp_body_names: list[str] = art_body_names
        self._amp_joint_ids: list[int] = idx_map["joints"]
        self._amp_pelvis_id: int = idx_map["pelvis"][0]
        self._amp_foot_ids: list[int] = idx_map["feet"]
        self._amp_hand_ids: list[int] = idx_map["hands"]

        # Default joint pose in the spec order — read from the articulation
        # (this is the same pose used by the init_state / reset events).
        default_jp_full = self._robot.data.default_joint_pos[0].detach()
        self._amp_default_joint_pos = default_jp_full[self._amp_joint_ids].to(self.device)
        # Expose for AmpPPO.construct_algorithm fallback.
        self.amp_default_joint_pos = self._amp_default_joint_pos.clone().cpu().numpy()

        # --- K-buffer allocation.
        self._amp_stack_k = int(spec.stack_k)
        self._amp_frame_dim = int(spec.frame_dim)
        self._amp_obs_dim = int(spec.amp_obs_dim)
        self._amp_history_buf = torch.zeros(
            self.num_envs, self._amp_stack_k, self._amp_frame_dim,
            device=self.device, dtype=torch.float32,
        )
        # Scratch buffer for the pre-reset terminal AMP obs (published via extras).
        self._amp_terminal_next_obs = torch.zeros(
            self.num_envs, self._amp_obs_dim, device=self.device, dtype=torch.float32,
        )

        # --- Optional motion dataset for the motion-reset strategy.
        self._amp_motion_dataset: MotionDataset | None = None
        if cfg.amp_reset.use_motion_reset:
            if not cfg.amp_motion_files:
                raise ValueError(
                    "amp_reset.use_motion_reset=True but amp_motion_files is empty. "
                    "Populate cfg.amp_motion_files with the expert .npz path(s)."
                )
            self._amp_motion_dataset = MotionDataset(
                motion_files=list(cfg.amp_motion_files),
                spec=spec,
                # Pass the live articulation's joint / body names as a
                # cross-check against the npz. If the npz was written by
                # ``scripts/AMP/csv_to_npz_final.py`` it carries its own
                # ``joint_names`` / ``body_names`` — MotionDataset treats
                # those as authoritative and raises if the args disagree.
                # For legacy npz files without names, these args are the
                # fallback ordering. Either way the spec-joint order is
                # resolved into this articulation order via
                # ``resolve_indices(spec.joint_names, amp_joint_names, ...)``
                # so discriminator expert features and policy features
                # share one joint axis.
                amp_joint_names=art_joint_names,
                amp_body_names=art_body_names,
                default_joint_pos=self._amp_default_joint_pos.detach().cpu().numpy(),
                env_step_dt=float(self.step_dt),
                wrap_around=bool(cfg.amp_wrap_around),
                device=self.device,
            )

        # Seed the K-buffer with the initial articulation state so the very
        # first ``obs["amp"]`` is well-defined.
        self._seed_kbuffer_from_current_state(torch.arange(self.num_envs, device=self.device))

        # Inject the initial ``obs["amp"]`` into the obs_buf produced by the
        # parent's __init__ reset so the first ``env.get_observations()`` call
        # already contains a valid AMP obs.
        self._inject_amp_obs()

        # extras["amp"] slot for terminal next_amp_obs + done mask.
        if "amp" not in self.extras:
            self.extras["amp"] = {}

    # ------------------------------------------------------------------
    # AMP observation construction
    # ------------------------------------------------------------------
    def _current_amp_state(self, env_ids: torch.Tensor | slice | None = None) -> AmpObsState:
        """Build an :class:`AmpObsState` from the current articulation state."""
        if env_ids is None:
            env_ids = slice(None)

        data = self._robot.data
        root_pos_w = data.root_pos_w[env_ids]
        root_quat_w = data.root_quat_w[env_ids]
        root_lin_vel_w = data.root_lin_vel_w[env_ids]
        root_ang_vel_w = data.root_ang_vel_w[env_ids]

        joint_pos = data.joint_pos[env_ids][:, self._amp_joint_ids]
        joint_vel = data.joint_vel[env_ids][:, self._amp_joint_ids]

        body_pos_w = data.body_pos_w[env_ids]
        body_lin_vel_w = data.body_lin_vel_w[env_ids]
        foot_pos_w = body_pos_w[:, self._amp_foot_ids, :]
        foot_lin_vel_w = body_lin_vel_w[:, self._amp_foot_ids, :]
        hand_pos_w = body_pos_w[:, self._amp_hand_ids, :]
        hand_lin_vel_w = body_lin_vel_w[:, self._amp_hand_ids, :]

        return AmpObsState(
            root_pos_w=root_pos_w,
            root_quat_w=root_quat_w,
            root_lin_vel_w=root_lin_vel_w,
            root_ang_vel_w=root_ang_vel_w,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            default_joint_pos=self._amp_default_joint_pos,
            foot_pos_w=foot_pos_w,
            foot_lin_vel_w=foot_lin_vel_w,
            hand_pos_w=hand_pos_w,
            hand_lin_vel_w=hand_lin_vel_w,
        )

    def _compute_phi_all(self) -> torch.Tensor:
        """``(num_envs, frame_dim)`` per-frame AMP feature for every env."""
        state = self._current_amp_state(None)
        return build_amp_frame_from_state(state, self.amp_spec)

    def _flat_amp_obs(self) -> torch.Tensor:
        """Return the stacked AMP obs from the current history buffer."""
        return build_amp_window(self._amp_history_buf, self._amp_stack_k)

    def _inject_amp_obs(self) -> None:
        """Place the current stacked AMP obs under ``self.obs_buf['amp']``."""
        if not hasattr(self, "obs_buf") or self.obs_buf is None:
            return
        self.obs_buf[self.AMP_OBS_KEY] = self._flat_amp_obs().detach().clone()

    def _seed_kbuffer_from_current_state(self, env_ids: torch.Tensor) -> None:
        """Fill the K-buffer at ``env_ids`` with the current frame repeated."""
        if env_ids.numel() == 0:
            return
        state = self._current_amp_state(env_ids)
        phi = build_amp_frame_from_state(state, self.amp_spec)  # (n, D)
        if self.cfg.amp_reset.kbuf_fill_mode == "zero":
            self._amp_history_buf[env_ids] = 0.0
        else:  # "repeat"
            self._amp_history_buf[env_ids] = phi.unsqueeze(1).expand(
                -1, self._amp_stack_k, -1
            ).contiguous()

    # ------------------------------------------------------------------
    # step() override — minimally re-implemented to inject AMP logic.
    # ------------------------------------------------------------------
    def step(self, action: torch.Tensor):  # type: ignore[override]
        """Drive one env step, maintaining the AMP K-buffer around resets."""
        # -- 1. Actions & physics (mirror ManagerBasedRLEnv.step exactly).
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        # -- 2. Post-step bookkeeping.
        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- 3. AMP: advance the K-buffer with the pre-reset phi for every env.
        #        This is the "true" next-frame feature; for done envs it will
        #        be captured as the terminal next_amp_obs below.
        with torch.no_grad():
            phi_pre = self._compute_phi_all()  # (N, D)
            self._amp_history_buf = concat_frame_history(
                phi_pre, self._amp_history_buf, self._amp_stack_k
            )
            pre_reset_amp_obs = self._flat_amp_obs().detach().clone()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        # -- 4. Publish the pre-reset stacked AMP obs — only for done envs the
        #       algorithm will read this back as the true ``amp_obs_{t+1}``.
        self._amp_terminal_next_obs.zero_()
        if reset_env_ids.numel() > 0:
            self._amp_terminal_next_obs[reset_env_ids] = pre_reset_amp_obs[reset_env_ids]

        # -- 5. Reset terminated envs (standard IsaacLab path).
        if reset_env_ids.numel() > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

            # After the standard reset, overlay the motion-reset state (if any)
            # and reseed the K-buffer for the reset envs.
            self._apply_motion_reset(reset_env_ids)
            self._seed_kbuffer_from_current_state(reset_env_ids)

        # -- 6. Post-reset managers.
        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        # -- 7. Final obs (post-reset) — matches the parent's contract for
        #       policy/critic groups; AMP obs is injected separately.
        self.obs_buf = self.observation_manager.compute(update_history=True)
        self._inject_amp_obs()

        # -- 8. Export extras for the AMP algorithm.
        if "amp" not in self.extras:
            self.extras["amp"] = {}
        self.extras["amp"]["terminal_next_amp_obs"] = self._amp_terminal_next_obs.clone()
        self.extras["amp"]["reset_buf"] = self.reset_buf.detach().clone()

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    # ------------------------------------------------------------------
    # Motion-reset strategy
    # ------------------------------------------------------------------
    def _apply_motion_reset(self, env_ids: torch.Tensor) -> None:
        """Overlay the motion-reset state onto freshly-reset envs."""
        cfg = self.cfg.amp_reset
        if not cfg.use_motion_reset or self._amp_motion_dataset is None:
            return
        if env_ids.numel() == 0:
            return

        n = int(env_ids.numel())
        _, state = self._amp_motion_dataset.sample_reset_frames(n)

        joint_pos = state["joint_pos"]
        joint_vel = state["joint_vel"]
        root_pos = state["root_pos"].clone()
        root_quat = state["root_quat"].clone()
        root_lin_vel = state["root_lin_vel"].clone()
        root_ang_vel = state["root_ang_vel"].clone()

        # XY zeroing + env-origin offset so envs spawn at their cell origins.
        if cfg.xy_zero:
            root_pos[:, 0:2] = 0.0
        root_pos = root_pos + self.scene.env_origins[env_ids]

        # Yaw randomization — rotate root_quat + root_lin_vel about world Z.
        if cfg.yaw_randomize:
            yaw = (torch.rand(n, device=self.device) * 2.0 - 1.0) * math.pi
            cos_h = torch.cos(0.5 * yaw)
            sin_h = torch.sin(0.5 * yaw)
            yaw_q = torch.stack(
                [cos_h, torch.zeros_like(cos_h), torch.zeros_like(cos_h), sin_h], dim=-1
            )  # (n, 4) wxyz
            root_quat = _quat_mul_wxyz(yaw_q, root_quat)
            root_lin_vel = _rotate_vec_by_yaw(root_lin_vel, yaw)

        # Additive noise.
        if cfg.joint_pos_noise > 0.0:
            joint_pos = joint_pos + cfg.joint_pos_noise * torch.randn_like(joint_pos)
        if cfg.joint_vel_noise > 0.0:
            joint_vel = joint_vel + cfg.joint_vel_noise * torch.randn_like(joint_vel)
        if cfg.root_lin_vel_noise > 0.0:
            root_lin_vel = root_lin_vel + cfg.root_lin_vel_noise * torch.randn_like(root_lin_vel)
        if cfg.root_ang_vel_noise > 0.0:
            root_ang_vel = root_ang_vel + cfg.root_ang_vel_noise * torch.randn_like(root_ang_vel)

        # Map the spec-joint columns back to the articulation's full joint order.
        full_jp = self._robot.data.default_joint_pos[env_ids].clone()
        full_jv = torch.zeros_like(full_jp)
        full_jp[:, self._amp_joint_ids] = joint_pos
        full_jv[:, self._amp_joint_ids] = joint_vel

        root_state = torch.cat(
            [root_pos, root_quat, root_lin_vel, root_ang_vel], dim=-1
        )  # (n, 13)

        self._robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(full_jp, full_jv, env_ids=env_ids)

        # Refresh data buffers so the K-buffer seed below reads the new state.
        self.scene.update(dt=self.physics_dt)


# ---------------------------------------------------------------------------
# Small quaternion / yaw helpers (world-frame Z rotation).
# ---------------------------------------------------------------------------

def _quat_mul_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two batches of wxyz quaternions: out = q1 * q2."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def _rotate_vec_by_yaw(vec: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Rotate a world-frame 3-vector about the world Z axis by ``yaw`` (rad)."""
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    x = c * vec[..., 0] - s * vec[..., 1]
    y = s * vec[..., 0] + c * vec[..., 1]
    z = vec[..., 2]
    return torch.stack([x, y, z], dim=-1)
