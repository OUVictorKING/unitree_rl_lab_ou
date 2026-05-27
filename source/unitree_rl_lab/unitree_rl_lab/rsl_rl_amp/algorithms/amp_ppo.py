# Copyright (c) 2025, Unitree Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""AMP variant of PPO.

Extends ``rsl_rl.algorithms.PPO`` with:

- An AMP discriminator that distinguishes expert transitions (from a motion
  dataset) from transitions collected by the policy during rollouts.
- A style reward derived from the discriminator which is added to the task
  reward with a configurable coefficient before advantage computation.
- A discriminator update loop (BCE + optional gradient penalty + optional
  weight decay) interleaved with PPO updates.

The algorithm is agnostic to the AMP feature semantics. It consumes the
environment's ``obs["amp"]`` tensor and an expert ``MotionDataset`` whose
features must match in dimension.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict
from typing import Any

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.utils import resolve_callable, resolve_obs_groups

from ..features.amp_features import AmpObsSpec
from ..modules.amp_discriminator import AmpDiscriminator
from ..storage.amp_rollout_storage import AmpRolloutStorage
from ..storage.motion_dataset import MotionDataset
from .amp_curriculum import AmpRewardCurriculum, AmpRewardCurriculumCfg


# The obs-group name expected by this algorithm. The environment must expose
# an observation group named exactly "amp".
AMP_OBS_SET_NAME = "amp"


class AmpPPO(PPO):
    """PPO + AMP discriminator.

    Args:
        actor: Policy network (MLPModel).
        critic: Value network (MLPModel).
        storage: An :class:`AmpRolloutStorage` instance.
        discriminator: An :class:`AmpDiscriminator` instance.
        motion_dataset: The expert :class:`MotionDataset`.
        amp_reward_coef: Scalar multiplier applied to the discriminator-based
            style reward before mixing with the task reward.
        amp_discriminator_learning_rate: Learning rate for the discriminator
            optimizer.
        amp_num_discriminator_updates: Number of discriminator mini-batch
            updates per PPO update call.
        amp_discriminator_mini_batch_size: Batch size used for each
            discriminator update.
        amp_gradient_penalty_coef: Coefficient of the R1-style gradient
            penalty on real samples. Set to ``0`` to disable.
        amp_discriminator_weight_decay: L2 regularization for the
            discriminator optimizer.
        normalize_amp_obs: If ``True``, maintain an EmpiricalNormalization
            over AMP observations and apply it to both real and fake samples.
        **kwargs: Forwarded to :class:`rsl_rl.algorithms.PPO`.
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: AmpRolloutStorage,
        discriminator: AmpDiscriminator,
        motion_dataset: MotionDataset,
        *,
        amp_reward_coef: float = 0.2,
        amp_discriminator_learning_rate: float = 1.0e-4,
        amp_num_discriminator_updates: int = 1,
        amp_discriminator_mini_batch_size: int = 4096,
        amp_gradient_penalty_coef: float = 5.0,
        amp_discriminator_weight_decay: float = 1.0e-4,
        amp_discriminator_update_every_n_iters: int = 1,
        normalize_amp_obs: bool = True,
        amp_curriculum_cfg: AmpRewardCurriculumCfg | dict | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(actor=actor, critic=critic, storage=storage, **kwargs)

        if not isinstance(storage, AmpRolloutStorage):
            raise TypeError(
                "AmpPPO requires an AmpRolloutStorage instance, got "
                f"{type(storage).__name__}."
            )

        self.discriminator = discriminator.to(self.device)
        self.motion_dataset = motion_dataset

        # Replace the parent's transition container with the AMP-aware one.
        self.transition = AmpRolloutStorage.Transition()

        # Only-up AMP reward curriculum (alpha_init → alpha_max via EMA gating).
        if amp_curriculum_cfg is None:
            curr_cfg = AmpRewardCurriculumCfg(alpha_init=float(amp_reward_coef))
        elif isinstance(amp_curriculum_cfg, AmpRewardCurriculumCfg):
            curr_cfg = amp_curriculum_cfg
        else:
            curr_cfg = AmpRewardCurriculumCfg(**dict(amp_curriculum_cfg))
        self.curriculum = AmpRewardCurriculum(curr_cfg)
        self.amp_reward_coef = float(self.curriculum.alpha_amp)

        self.amp_discriminator_learning_rate = float(amp_discriminator_learning_rate)
        self.amp_num_discriminator_updates = int(amp_num_discriminator_updates)
        self.amp_discriminator_mini_batch_size = int(amp_discriminator_mini_batch_size)
        self.amp_gradient_penalty_coef = float(amp_gradient_penalty_coef)
        self.amp_discriminator_weight_decay = float(amp_discriminator_weight_decay)
        self.amp_discriminator_update_every_n_iters = max(
            1, int(amp_discriminator_update_every_n_iters)
        )

        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=self.amp_discriminator_learning_rate,
            weight_decay=self.amp_discriminator_weight_decay,
        )

        # Optional running-stats normalization for AMP observations.
        self.normalize_amp_obs = bool(normalize_amp_obs)
        if self.normalize_amp_obs:
            self.amp_obs_normalizer = EmpiricalNormalization(
                shape=self.discriminator.amp_obs_dim
            ).to(self.device)
        else:
            self.amp_obs_normalizer = nn.Identity().to(self.device)

        # Last-iteration bookkeeping, surfaced by the runner for logging.
        self._last_task_reward_mean: float = 0.0
        self._last_amp_reward_mean: float = 0.0

        # Per-rollout accumulators used by the AMP reward curriculum.
        self._rollout_num_dones: int = 0
        self._rollout_num_nontimeout_dones: int = 0
        self._rollout_steps: int = 0
        self._last_curriculum_log: dict[str, float] = {}
        # Set by the runner before each update() call so the curriculum can
        # read env-side metrics that aren't in the rollout buffer.
        self._curriculum_episode_length_norm: float = 0.0
        self._curriculum_tracking_score: float = 0.0

        # Disc-update scheduling state. ``_update_iter_count`` ticks every
        # full ``update()`` call; disc is skipped when the curriculum still
        # has it disabled or the modulo gate hasn't hit. ``_last_disc_log``
        # caches the previous real disc update dict so tensorboard plots
        # stay continuous across skipped iterations.
        self._update_iter_count: int = 0
        self._last_disc_log: dict[str, float] = {
            "amp_discriminator_bce": 0.0,
            "amp_discriminator_gp": 0.0,
            "amp_logit_real": 0.0,
            "amp_logit_fake": 0.0,
            "amp_acc_real": 0.0,
            "amp_acc_fake": 0.0,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _normalize_amp(
        self, amp_obs: torch.Tensor, update: bool = False
    ) -> torch.Tensor:
        """Apply (and optionally update) the AMP observation normalizer."""
        if not self.normalize_amp_obs:
            return amp_obs
        if update and self.training:
            # EmpiricalNormalization.update mutates running stats.
            self.amp_obs_normalizer.update(amp_obs)  # type: ignore[attr-defined]
        return self.amp_obs_normalizer(amp_obs)

    def _extract_amp_obs(self, obs: TensorDict) -> torch.Tensor:
        """Extract ``obs["amp"]`` and fail clearly if missing."""
        if AMP_OBS_SET_NAME not in obs.keys():
            raise KeyError(
                f"AMP algorithm requires obs['{AMP_OBS_SET_NAME}'] but the environment "
                f"observation dict only has keys: {list(obs.keys())}."
            )
        return obs[AMP_OBS_SET_NAME].to(self.device)

    # ------------------------------------------------------------------
    # Rollout hooks
    # ------------------------------------------------------------------
    def act(self, obs: TensorDict) -> torch.Tensor:  # type: ignore[override]
        """Run parent ``act`` and stash the current AMP observation."""
        amp_obs = self._extract_amp_obs(obs).detach().clone()
        # Note: assigning on the AMP Transition container. The parent's act()
        # will populate the standard PPO fields (actions, values, ...).
        self.transition.amp_observations = amp_obs
        return super().act(obs)

    def process_env_step(  # type: ignore[override]
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        """Mix the style reward into the environment reward before storing.

        For done envs, prefer ``extras['amp']['terminal_next_amp_obs']`` as
        the true pre-reset ``amp_obs_{t+1}`` — otherwise the discriminator
        would train on a (pre-reset, post-reset) transition and drift.
        父类 PPO.process_env_step 把 transition（含 mixed_reward）push 进 storage，然后清空 transition
        """
        next_amp_obs = self._extract_amp_obs(obs).detach().clone()

        # Terminal-transition fix: patch done rows with the env's pre-reset
        # snapshot. If the env hasn't provided one, we fall back to the
        # (post-reset) obs["amp"] which matches legacy behavior.
        amp_extras = extras.get("amp") if isinstance(extras, dict) else None
        if isinstance(amp_extras, dict) and "terminal_next_amp_obs" in amp_extras:
            terminal = amp_extras["terminal_next_amp_obs"]
            if terminal.device != next_amp_obs.device:
                terminal = terminal.to(next_amp_obs.device)
            done_mask = dones.to(torch.bool).to(next_amp_obs.device).view(-1)
            if done_mask.any():
                next_amp_obs[done_mask] = terminal[done_mask].detach()

        self.transition.next_amp_observations = next_amp_obs

        # Compute the style reward for this step. We normalize using the
        # *current* stats (not-in-place); the normalizer itself is updated
        # during update() on the full rollout.
        with torch.no_grad():
            amp_t = self.transition.amp_observations
            amp_tp1 = next_amp_obs
            if self.normalize_amp_obs:
                amp_t_n = self.amp_obs_normalizer(amp_t)
                amp_tp1_n = self.amp_obs_normalizer(amp_tp1)
            else:
                amp_t_n = amp_t
                amp_tp1_n = amp_tp1
            style_reward = self.discriminator.predict_reward(
                amp_t_n, amp_tp1_n
            ).squeeze(-1)

        scaled_amp_reward = self.amp_reward_coef * style_reward.to(rewards.dtype).to(
            rewards.device
        )
        mixed_reward = rewards + scaled_amp_reward

        # Keep per-component rewards for logging.
        self.transition.task_rewards = rewards.detach().clone()
        self.transition.amp_rewards = scaled_amp_reward.detach().clone()

        self._last_task_reward_mean = float(rewards.detach().mean().item())
        self._last_amp_reward_mean = float(scaled_amp_reward.detach().mean().item())

        # Curriculum accumulators (plain Python scalars avoid per-step
        # host-device sync — the tensor ops already returned cheap floats).
        with torch.no_grad():
            self._rollout_steps += int(dones.numel())
            self._rollout_num_dones += int(dones.to(torch.bool).sum().item())
            # Non-timeout termination ≡ done AND not time_out. When the env
            # exposes ``extras['time_outs']`` (standard IsaacLab contract) we
            # use it; otherwise treat all dones as non-timeout (conservative).
            time_outs = None
            if isinstance(extras, dict) and "time_outs" in extras:
                time_outs = extras["time_outs"]
            if time_outs is not None:
                nontimeout = dones.to(torch.bool) & ~time_outs.to(torch.bool).to(
                    dones.device
                )
                self._rollout_num_nontimeout_dones += int(nontimeout.sum().item())
            else:
                self._rollout_num_nontimeout_dones += int(
                    dones.to(torch.bool).sum().item()
                )
        # 这里只是记录总reward，并没有参与更新
        super().process_env_step(obs, mixed_reward, dones, extras)

    # ------------------------------------------------------------------
    # Discriminator update
    # ------------------------------------------------------------------
    def _update_amp_normalizer(self) -> None:
        """Update the AMP observation normalizer on a concatenation of real and fake samples."""
        if not self.normalize_amp_obs:
            return
        storage: AmpRolloutStorage = self.storage  # type: ignore[assignment]
        amp_flat = storage.amp_observations.reshape(-1, storage.amp_obs_dim)
        next_amp_flat = storage.next_amp_observations.reshape(-1, storage.amp_obs_dim)
        # Sample a fresh real batch to include its statistics as well.
        real_now, real_next = self.motion_dataset.sample(
            num_samples=min(amp_flat.shape[0], self.amp_discriminator_mini_batch_size)
        )
        all_obs = torch.cat([amp_flat, next_amp_flat, real_now, real_next], dim=0)
        self.amp_obs_normalizer.update(all_obs)  # type: ignore[attr-defined]

    def _update_discriminator(self) -> dict[str, float]:
        """Run ``amp_num_discriminator_updates`` BCE updates on the discriminator."""
        storage: AmpRolloutStorage = self.storage  # type: ignore[assignment]
        bce_losses: list[float] = []
        gp_losses: list[float] = []
        real_logit_means: list[float] = []
        fake_logit_means: list[float] = []
        real_acc: list[float] = []
        fake_acc: list[float] = []

        for _ in range(self.amp_num_discriminator_updates):
            real_now, real_next = self.motion_dataset.sample(
                self.amp_discriminator_mini_batch_size
            )
            fake_now, fake_next = storage.sample_amp_transitions(
                self.amp_discriminator_mini_batch_size
            )

            real_now = self._normalize_amp(real_now.to(self.device))
            real_next = self._normalize_amp(real_next.to(self.device))
            fake_now = self._normalize_amp(fake_now.to(self.device))
            fake_next = self._normalize_amp(fake_next.to(self.device))

            real_logit = self.discriminator(real_now, real_next)
            fake_logit = self.discriminator(fake_now, fake_next)

            bce = self.discriminator.bce_loss(real_logit, fake_logit)

            gp = torch.zeros((), device=self.device)
            if self.amp_gradient_penalty_coef > 0.0:
                gp = self.discriminator.gradient_penalty(real_now, real_next)

            loss = bce + self.amp_gradient_penalty_coef * gp

            self.discriminator_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.discriminator.parameters(), self.max_grad_norm
            )
            self.discriminator_optimizer.step()

            bce_losses.append(float(bce.item()))
            gp_losses.append(float(gp.item()))
            real_logit_means.append(float(real_logit.mean().item()))
            fake_logit_means.append(float(fake_logit.mean().item()))
            real_acc.append(float((real_logit > 0).float().mean().item()))
            fake_acc.append(float((fake_logit < 0).float().mean().item()))

        def _mean(xs: list[float]) -> float:
            return sum(xs) / max(len(xs), 1)

        return {
            "amp_discriminator_bce": _mean(bce_losses),
            "amp_discriminator_gp": _mean(gp_losses),
            "amp_logit_real": _mean(real_logit_means),
            "amp_logit_fake": _mean(fake_logit_means),
            "amp_acc_real": _mean(real_acc),
            "amp_acc_fake": _mean(fake_acc),
        }

    # ------------------------------------------------------------------
    # Full update (PPO + discriminator)
    # ------------------------------------------------------------------
    def update(self) -> dict[str, float]:  # type: ignore[override]
        """Run PPO update then discriminator update; return combined loss dict."""
        self._update_iter_count += 1

        # Update AMP normalizer with the full rollout *before* PPO/disc updates
        # so all consumers see consistent stats.
        # 计算得到新的rollout里面的mean和std，供后面更新使用
        self._update_amp_normalizer()

        # Discriminator update gated by the curriculum (V3 two-phase mode
        # skips disc training during stage-0 task-only warmup) and by the
        # every-N-iters cadence. When skipped we re-emit the last real disc
        # log so tensorboard time series stay continuous.
        disc_gate_open = (
            self.curriculum.disc_training_enabled
            and (self._update_iter_count % self.amp_discriminator_update_every_n_iters) == 0
        )
        if disc_gate_open:
            amp_loss_dict = self._update_discriminator()
            self._last_disc_log = dict(amp_loss_dict)
        else:
            amp_loss_dict = dict(self._last_disc_log)
        amp_loss_dict["amp_disc_update_fired"] = 1.0 if disc_gate_open else 0.0

        # PPO update (also clears self.storage at the end).
        ppo_loss_dict = super().update()

        # Per-rollout reward component logging.
        storage: AmpRolloutStorage = self.storage  # type: ignore[assignment]
        with torch.no_grad():
            task_reward_mean = float(storage.task_rewards.mean().item())
            amp_reward_mean = float(storage.amp_rewards.mean().item())

        # Tick the only-up AMP reward curriculum and sync alpha_amp.
        curr_log = self._step_curriculum(task_reward_mean)

        out = dict(ppo_loss_dict)
        out.update(amp_loss_dict)
        out.update(
            {
                "task_reward_mean": task_reward_mean,
                "amp_reward_mean": amp_reward_mean,
                "amp_reward_coef": self.amp_reward_coef,
            }
        )
        out.update(curr_log)
        return out

    # ------------------------------------------------------------------
    # Curriculum tick
    # ------------------------------------------------------------------
    def set_curriculum_inputs(
        self,
        *,
        episode_length_norm: float,
        tracking_score: float,
    ) -> None:
        """Runner-side hook: push env metrics the algorithm can't compute itself."""
        self._curriculum_episode_length_norm = float(episode_length_norm)
        self._curriculum_tracking_score = float(tracking_score)

    def _step_curriculum(self, task_reward_mean: float) -> dict[str, float]:
        """Advance the curriculum using this iteration's accumulators."""
        termination_ratio = 0.0
        if self._rollout_steps > 0:
            termination_ratio = self._rollout_num_nontimeout_dones / float(
                self._rollout_steps
            )

        log = self.curriculum.update(
            episode_length_norm=self._curriculum_episode_length_norm,
            task_reward_mean=task_reward_mean,
            termination_ratio=termination_ratio,
            tracking_score=self._curriculum_tracking_score,
        )
        self.amp_reward_coef = float(self.curriculum.alpha_amp)
        self._last_curriculum_log = dict(log)

        # Reset per-rollout accumulators for the next iteration.
        self._rollout_num_dones = 0
        self._rollout_num_nontimeout_dones = 0
        self._rollout_steps = 0

        return log

    # ------------------------------------------------------------------
    # Modes
    # ------------------------------------------------------------------
    def train_mode(self) -> None:  # type: ignore[override]
        super().train_mode()
        self.discriminator.train()
        if self.normalize_amp_obs:
            self.amp_obs_normalizer.train()

    def eval_mode(self) -> None:  # type: ignore[override]
        super().eval_mode()
        self.discriminator.eval()
        if self.normalize_amp_obs:
            self.amp_obs_normalizer.eval()

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def save(self) -> dict:  # type: ignore[override]
        """Return a dict with PPO + AMP component states."""
        saved = super().save()
        saved["amp_discriminator_state_dict"] = self.discriminator.state_dict()
        saved["amp_discriminator_optimizer_state_dict"] = (
            self.discriminator_optimizer.state_dict()
        )
        if self.normalize_amp_obs:
            saved["amp_obs_normalizer_state_dict"] = (
                self.amp_obs_normalizer.state_dict()
            )
        saved["amp_cfg"] = {
            "amp_reward_coef": self.amp_reward_coef,
            "amp_discriminator_learning_rate": self.amp_discriminator_learning_rate,
            "amp_num_discriminator_updates": self.amp_num_discriminator_updates,
            "amp_discriminator_mini_batch_size": self.amp_discriminator_mini_batch_size,
            "amp_gradient_penalty_coef": self.amp_gradient_penalty_coef,
            "amp_discriminator_weight_decay": self.amp_discriminator_weight_decay,
            "normalize_amp_obs": self.normalize_amp_obs,
        }
        saved["amp_curriculum_state"] = self.curriculum.save_state()
        return saved

    def load(  # type: ignore[override]
        self,
        loaded_dict: dict,
        load_cfg: dict | None,
        strict: bool,
    ) -> bool:
        """Load PPO + AMP components from a same-schema checkpoint.

        ``load_cfg`` forwards standard PPO keys (``actor``, ``critic``,
        ``optimizer``, ``iteration``, ``rnd``) to the parent and recognizes
        three AMP-specific keys:

        - ``amp``: load AMP discriminator weights if present (default True).
        - ``amp_optimizer``: load discriminator optimizer state (default True).
        - ``amp_normalizer``: load AMP observation normalizer (default True).

        The checkpoint must have been produced by an AMP config with the same
        feature layout — there is no compatibility fallback. Change the AMP
        feature set → train from scratch.
        """
        lc = dict(load_cfg) if load_cfg is not None else {}

        load_amp = bool(lc.get("amp", True))
        load_amp_opt = bool(lc.get("amp_optimizer", lc.get("amp", True)))
        load_amp_norm = bool(lc.get("amp_normalizer", lc.get("amp", True)))

        # Strip AMP-only keys so the parent PPO.load doesn't see them.
        parent_cfg = (
            None
            if load_cfg is None
            else {
                k: v
                for k, v in lc.items()
                if k in ("actor", "critic", "optimizer", "iteration", "rnd")
            }
        )
        load_it = super().load(loaded_dict, parent_cfg, strict)

        if not load_amp:
            return load_it

        if "amp_discriminator_state_dict" in loaded_dict:
            self.discriminator.load_state_dict(
                loaded_dict["amp_discriminator_state_dict"], strict=strict
            )
        if load_amp_opt and "amp_discriminator_optimizer_state_dict" in loaded_dict:
            self.discriminator_optimizer.load_state_dict(
                loaded_dict["amp_discriminator_optimizer_state_dict"]
            )
        if (
            load_amp_norm
            and self.normalize_amp_obs
            and "amp_obs_normalizer_state_dict" in loaded_dict
        ):
            self.amp_obs_normalizer.load_state_dict(
                loaded_dict["amp_obs_normalizer_state_dict"], strict=strict
            )

        # Restore the curriculum (alpha_amp, EMAs, stage counters). When the
        # checkpoint predates the curriculum, we keep the freshly-constructed
        # state — equivalent to resuming at alpha_init.
        if "amp_curriculum_state" in loaded_dict:
            self.curriculum.load_state(loaded_dict["amp_curriculum_state"])
            self.amp_reward_coef = float(self.curriculum.alpha_amp)

        return load_it

    # ------------------------------------------------------------------
    # Multi-GPU parameter sync
    # ------------------------------------------------------------------
    def broadcast_parameters(self) -> None:  # type: ignore[override]
        super().broadcast_parameters()
        if self.is_multi_gpu:
            torch.distributed.broadcast_object_list(
                [self.discriminator.state_dict()], src=0
            )

    def reduce_parameters(self) -> None:  # type: ignore[override]
        super().reduce_parameters()
        if not self.is_multi_gpu:
            return
        params = list(self.discriminator.parameters())
        grads = [p.grad.view(-1) for p in params if p.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for p in params:
            if p.grad is not None:
                numel = p.numel()
                p.grad.data.copy_(
                    all_grads[offset : offset + numel].view_as(p.grad.data)
                )
                offset += numel

    # ------------------------------------------------------------------
    # Construction entry point
    # ------------------------------------------------------------------
    @staticmethod
    def construct_algorithm(  # type: ignore[override]
        obs: TensorDict, env: VecEnv, cfg: dict, device: str
    ) -> "AmpPPO":
        """Construct an :class:`AmpPPO` from a configuration dict.

        Config layout expected under ``cfg``:

        - ``algorithm``: PPO + AMP kwargs. ``class_name`` is the AMP class.
        - ``actor`` / ``critic``: forwarded to their respective MLP models.
        - ``obs_groups``: dict mapping obs-set name to a list of obs groups.
            Must contain ``"amp"``. If absent, defaults to ``[AMP_OBS_SET_NAME]``
            when a group of that name exists in ``obs``.
        - ``amp`` (required): discriminator + dataset kwargs. Keys:

            * ``motion_files``: list[str] | str.
            * ``motion_weights``: optional list[float], per-clip sampling weights
              for the discriminator data loader. None => uniform over clips.
            * ``default_joint_pos``: optional list/array. If absent, pulled
              from ``env.unwrapped.amp_default_joint_pos``.
            * ``wrap_around``: bool, default True. Cyclic transition pairs.
            * ``discriminator``: kwargs forwarded to :class:`AmpDiscriminator`.

        The MotionDataset's transition dt is auto-derived from the env's
        ``step_dt`` (= ``sim.dt * decimation``). Expert clips are resampled
        once at dataset init onto that grid so the discriminator sees
        real / fake transitions with the same time delta.

        The environment is required to expose on its ``.unwrapped``:

        - ``amp_spec``: :class:`AmpObsSpec` — the unified feature layout.
        - ``amp_joint_names``: ordered joint name list in the npz column
          order (= live articulation order). MotionDataset resolves the
          spec-joint axis via ``resolve_indices(spec.joint_names, ...)``.
        - ``amp_body_names``: ordered body name list in the npz column
          order (= live articulation order).
        - ``amp_default_joint_pos``: optional fallback ``(J,)`` tensor/array
          used when ``cfg['amp']['default_joint_pos']`` is not provided.

        This mirrors the parent ``PPO.construct_algorithm`` signature so the
        AMP algorithm can be plugged into ``OnPolicyAmpRunner`` with minimal
        friction.
        """
        # --------------------------------------------------------------
        # Resolve classes 使用resolve_callable将这些类的名字字符串解析成类对象
        # type[MLPModel]是rslrl里面的类，能够自动创建MLPModel实例
        # pop就是将这个class name从字典里面摘除，后面字典里面就没有这个信息了
        # --------------------------------------------------------------
        alg_class: type[AmpPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        # --------------------------------------------------------------
        # Resolve obs_groups (must include "amp")
        # --------------------------------------------------------------
        # Fill the AMP set explicitly from the environment obs group named
        # "amp" rather than falling back to "policy" like resolve_obs_groups
        # would do -- AMP has no sensible policy-obs fallback.
        if AMP_OBS_SET_NAME not in cfg["obs_groups"]:
            if AMP_OBS_SET_NAME in obs.keys():
                cfg["obs_groups"][AMP_OBS_SET_NAME] = [AMP_OBS_SET_NAME]
            else:
                raise KeyError(
                    f"obs_groups is missing '{AMP_OBS_SET_NAME}' and no observation group "
                    f"named '{AMP_OBS_SET_NAME}' is exposed by the environment. AMP requires "
                    f"a dedicated AMP observation group."
                )
        default_sets = ["actor", "critic"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # --------------------------------------------------------------
        # Determine AMP observation dimension (from the unified spec).
        # --------------------------------------------------------------
        spec = _require_amp_spec(env)
        amp_obs_dim = _infer_amp_obs_dim(obs, cfg["obs_groups"][AMP_OBS_SET_NAME], env)
        if amp_obs_dim != spec.amp_obs_dim:
            raise RuntimeError(
                f"AMP obs-group dim ({amp_obs_dim}) does not match spec.amp_obs_dim "
                f"({spec.amp_obs_dim}). The env's AMP observation group must expose "
                "exactly the stacked AMP observation."
            )

        # --------------------------------------------------------------
        # Pop AMP-specific config (required)
        # --------------------------------------------------------------
        if "amp" not in cfg:
            raise KeyError(
                "Config is missing the 'amp' section. Provide at least "
                "cfg['amp']['motion_files']."
            )
        amp_cfg = cfg["amp"]
        motion_files = amp_cfg.get("motion_files")
        if motion_files is None:
            raise KeyError("cfg['amp']['motion_files'] is required.")

        # ``feature_mode`` is no longer supported — single unified spec only.
        if "feature_mode" in amp_cfg and amp_cfg["feature_mode"] not in (None, ""):
            print(
                f"[AmpPPO] Ignoring cfg['amp']['feature_mode']={amp_cfg['feature_mode']!r}: "
                "Penguin V1 uses a single unified AMP spec."
            )

        default_joint_pos = amp_cfg.get("default_joint_pos")
        if default_joint_pos is None:
            default_joint_pos = getattr(env.unwrapped, "amp_default_joint_pos", None)
        if default_joint_pos is None:
            raise KeyError(
                "default_joint_pos must be provided either via cfg['amp']['default_joint_pos'] "
                "or env.unwrapped.amp_default_joint_pos."
            )
        amp_joint_names = _require_attr(env, "amp_joint_names")
        amp_body_names = _require_attr(env, "amp_body_names")
        wrap_around = bool(amp_cfg.get("wrap_around", True))
        disc_kwargs = dict(amp_cfg.get("discriminator", {}))

        # Time-aligned expert sampling: the MotionDataset resamples each
        # clip at init onto the policy control-step grid so
        # (real_now, real_next) and (fake_now, fake_next) share the same
        # transition dt. env.step_dt = sim.dt * decimation.
        env_step_dt = float(getattr(env.unwrapped, "step_dt"))
        motion_weights = amp_cfg.get("motion_weights")

        # If the user supplied cfg['amp']['curriculum'], lift it into the
        # algorithm kwargs. cfg['algorithm']['amp_curriculum_cfg'] takes
        # precedence (explicit wins over implicit).
        curriculum_cfg = amp_cfg.get("curriculum", None)
        if curriculum_cfg is not None and "amp_curriculum_cfg" not in cfg["algorithm"]:
            cfg["algorithm"]["amp_curriculum_cfg"] = dict(curriculum_cfg)

        # --------------------------------------------------------------
        # Build components
        # --------------------------------------------------------------
        actor: MLPModel = actor_class(
            obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]
        ).to(device)
        print(f"Actor Model: {actor}")

        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore[attr-defined]
        critic: MLPModel = critic_class(
            obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]
        ).to(device)
        print(f"Critic Model: {critic}")

        storage = AmpRolloutStorage(
            training_type="rl",
            num_envs=env.num_envs,
            num_transitions_per_env=cfg["num_steps_per_env"],
            obs=obs,
            actions_shape=[env.num_actions],
            amp_obs_dim=amp_obs_dim,
            device=device,
        )

        discriminator = AmpDiscriminator(amp_obs_dim=amp_obs_dim, **disc_kwargs).to(
            device
        )

        motion_dataset = MotionDataset(
            motion_files=motion_files,
            spec=spec,
            amp_joint_names=amp_joint_names,
            amp_body_names=amp_body_names,
            default_joint_pos=default_joint_pos,
            env_step_dt=env_step_dt,
            motion_weights=motion_weights,
            wrap_around=wrap_around,
            device=device,
        )

        alg: AmpPPO = alg_class(
            actor=actor,
            critic=critic,
            storage=storage,
            discriminator=discriminator,
            motion_dataset=motion_dataset,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )
        return alg


def _require_amp_spec(env: VecEnv) -> AmpObsSpec:
    """Return the env's :class:`AmpObsSpec`, failing clearly if missing."""
    spec = getattr(getattr(env, "unwrapped", env), "amp_spec", None)
    if not isinstance(spec, AmpObsSpec):
        raise RuntimeError(
            "AmpPPO requires env.unwrapped.amp_spec to be an AmpObsSpec instance. "
            f"Got: {type(spec).__name__}. Make sure the env constructs the unified "
            "AMP spec at init time."
        )
    return spec


def _require_attr(env: VecEnv, name: str):
    """Fetch an attribute from ``env.unwrapped`` or raise a clear error."""
    unwrapped = getattr(env, "unwrapped", env)
    if not hasattr(unwrapped, name):
        raise RuntimeError(
            f"env.unwrapped is missing required attribute '{name}'. The AMP env "
            "subclass must populate the AMP naming info at construction time."
        )
    return getattr(unwrapped, name)


def _infer_amp_obs_dim(obs: TensorDict, amp_obs_groups: list[str], env: VecEnv) -> int:
    """Infer the AMP observation dimension.

    Priority order:

    1. ``env.unwrapped.amp_spec.amp_obs_dim`` if set by the environment.
    2. Sum of ``obs[group].shape[-1]`` for every group in the ``amp`` obs-set.
    """
    spec = getattr(getattr(env, "unwrapped", env), "amp_spec", None)
    if isinstance(spec, AmpObsSpec):
        return int(spec.amp_obs_dim)
    total = 0
    for g in amp_obs_groups:
        if g not in obs:
            raise KeyError(
                f"AMP obs-group '{g}' not present in observations (available keys: {list(obs.keys())})."
            )
        total += int(obs[g].shape[-1])
    if total == 0:
        raise RuntimeError(
            "Could not infer AMP observation dimension from the environment."
        )
    return total
