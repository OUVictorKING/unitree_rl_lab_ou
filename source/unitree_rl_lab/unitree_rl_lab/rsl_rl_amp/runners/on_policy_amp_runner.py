# Copyright (c) 2025, Unitree Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""On-policy runner for AMP training.

Mirrors :class:`rsl_rl.runners.OnPolicyRunner` but constructs an
:class:`AmpPPO` algorithm (which expects an ``"amp"`` observation group in the
environment's obs TensorDict).

The runner itself is thin: AMP-specific bookkeeping (style reward computation,
discriminator update, normalization) lives in :class:`AmpPPO`. This keeps the
runner very close to the stock rsl-rl OnPolicyRunner so upstream changes are
easy to follow.

Curriculum inputs
-----------------
Per iteration, the runner computes two scalars that the algorithm-side
curriculum cannot observe directly and pushes them via
:meth:`AmpPPO.set_curriculum_inputs`:

- ``episode_length_norm`` — ``mean(env.episode_length_buf) / max_episode_length``
- ``tracking_score`` — rollout-average of (``track_lin_vel_xy`` +
  ``track_ang_vel_z``) raw step-rewards, normalized by the sum of their
  maxes (both are ``exp(...)`` in [0, 1], so max sum = 2.0 × weight).
"""

from __future__ import annotations

import os
import time
import torch

from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.utils import check_nan, resolve_callable
from rsl_rl.utils.logger import Logger

from ..algorithms.amp_ppo import AmpPPO


class OnPolicyAmpRunner:
    """On-policy runner wrapping :class:`AmpPPO`.

    Mirrors the stock rsl-rl runner so all other pieces of the Unitree RL Lab
    training pipeline (logger, git-tagging, checkpoint paths, export helpers)
    continue to work.
    """

    alg: AmpPPO

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.env = env
        self.cfg = train_cfg
        self.device = device

        # Multi-GPU setup (identical to OnPolicyRunner).
        self._configure_multi_gpu()

        # Query observations so we can shape the rollout buffers.
        obs = self.env.get_observations()

        # Resolve the algorithm class. Default to AmpPPO if not specified.
        alg_class_name = self.cfg["algorithm"].get("class_name", None)
        if alg_class_name is None:
            raise KeyError(
                "train_cfg['algorithm']['class_name'] is required. "
                "Point it at AmpPPO (or a subclass)."
            )
        alg_class: type[AmpPPO] = resolve_callable(alg_class_name)  # type: ignore[assignment]
        # 初始化，创建网络，actor、critic、discriminator
        self.alg = alg_class.construct_algorithm(obs, self.env, self.cfg, self.device)

        # Logger.
        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )

        self.current_learning_iteration = 0

        # Resolve tracking-reward term indices once. These feed
        # `tracking_score` into the AMP reward curriculum.
        self._tracking_term_indices, self._tracking_max_sum = (
            self._resolve_tracking_terms()
        )

    # ------------------------------------------------------------------
    # Curriculum-input helpers
    # ------------------------------------------------------------------
    def _resolve_tracking_terms(
        self,
        names: tuple[str, ...] = ("track_lin_vel_xy", "track_ang_vel_z"),
    ) -> tuple[list[int], float]:
        """Find the indices of tracking reward terms in the reward manager.

        Returns a pair ``(indices, max_sum)`` where ``max_sum`` is the sum of
        each term's weight. For exp-style tracking rewards in ``[0, 1]``, the
        per-step max sum equals sum-of-weights.
        """
        env_u = getattr(self.env, "unwrapped", self.env)
        rm = getattr(env_u, "reward_manager", None)
        if rm is None or not hasattr(rm, "active_terms"):
            return [], 0.0
        active = list(rm.active_terms)
        indices: list[int] = []
        max_sum = 0.0
        for name in names:
            if name not in active:
                continue
            idx = active.index(name)
            indices.append(idx)
            try:
                max_sum += abs(float(rm.get_term_cfg(name).weight))
            except Exception:
                max_sum += 1.0
        return indices, max_sum

    def _tracking_step_mean(self) -> float:
        """Mean of (track_lin + track_ang) raw step reward over envs (this step)."""
        env_u = getattr(self.env, "unwrapped", self.env)
        rm = getattr(env_u, "reward_manager", None)
        if rm is None or not self._tracking_term_indices:
            return 0.0
        step_reward = getattr(rm, "_step_reward", None)
        if step_reward is None:
            return 0.0
        tr = step_reward[:, self._tracking_term_indices].sum(dim=-1)
        return float(tr.mean().item())

    def _episode_length_norm(self) -> float:
        """Mean episode length in ``[0, 1]`` relative to ``max_episode_length``."""
        env_u = getattr(self.env, "unwrapped", self.env)
        length_buf = getattr(env_u, "episode_length_buf", None)
        max_len = float(getattr(env_u, "max_episode_length", 0.0) or 0.0)
        if length_buf is None or max_len <= 0.0:
            return 0.0
        return float(length_buf.float().mean().item() / max_len)

    # ------------------------------------------------------------------
    # Learning loop
    # ------------------------------------------------------------------
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Main learning loop, mirrors OnPolicyRunner.learn()."""
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            tracking_step_sum = 0.0
            tracking_steps = 0
            # Rollout phase.
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    # 计算总reward，AMP的reward只计算，不更新，amp reward也会使用后面的那个课程的指标缓慢增大AMP的系数
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # AMP has no intrinsic reward channel comparable to RND; pass None.
                    self.logger.process_env_step(rewards, dones, extras, None)

                    # Accumulate tracking-reward step means for the curriculum.课程指标累加
                    tracking_step_sum += self._tracking_step_mean()
                    tracking_steps += 1

                stop = time.time()
                collect_time = stop - start
                start = stop

                # 这里计算整个episode计算return还有GAE，供给后面的网络更新使用
                self.alg.compute_returns(obs)

            # Push curriculum inputs before alg.update() ticks the curriculum.
            track_max = max(self._tracking_max_sum, 1e-9)
            tracking_score = (
                (tracking_step_sum / max(tracking_steps, 1)) / track_max
                if self._tracking_term_indices
                else 0.0
            )
            # 单纯存储两个env的指标，提供给前面的 _step_curriculum 使用
            self.alg.set_curriculum_inputs(
                episode_length_norm=self._episode_length_norm(),
                tracking_score=float(tracking_score),
            )

            # Update (PPO + discriminator).
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Logging.
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=None,
            )

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore[arg-type]

        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore[arg-type]
            self.logger.stop_logging_writer()

    # ------------------------------------------------------------------
    # Save / load / export
    # ------------------------------------------------------------------
    def save(self, path: str, infos: dict | None = None) -> None:
        """Persist algorithm state (PPO + AMP)."""
        saved = self.alg.save()
        saved["iter"] = self.current_learning_iteration
        saved["infos"] = infos
        torch.save(saved, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        """Load algorithm state (PPO + AMP)."""
        loaded = torch.load(path, weights_only=False, map_location=map_location)
        if self.alg.load(loaded, load_cfg, strict):
            self.current_learning_iteration = loaded["iter"]
        return loaded.get("infos")

    def get_inference_policy(self, device: str | None = None) -> MLPModel:
        self.alg.eval_mode()
        return self.alg.get_policy().to(device)  # type: ignore[return-value]

    def export_policy_to_jit(self, path: str, filename: str = "policy.pt") -> None:
        """Export the policy to a TorchScript file (mirrors OnPolicyRunner)."""
        jit_model = self.alg.get_policy().as_jit()
        jit_model.to("cpu")
        os.makedirs(path, exist_ok=True)
        traced = torch.jit.script(jit_model)
        traced.save(os.path.join(path, filename))

    def export_policy_to_onnx(
        self, path: str, filename: str = "policy.onnx", verbose: bool = False
    ) -> None:
        """Export the policy to ONNX (mirrors OnPolicyRunner)."""
        onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
        onnx_model.to("cpu")
        onnx_model.eval()
        os.makedirs(path, exist_ok=True)
        torch.onnx.export(
            onnx_model,
            onnx_model.get_dummy_inputs(),  # type: ignore[arg-type]
            os.path.join(path, filename),
            export_params=True,
            opset_version=18,
            verbose=verbose,
            input_names=onnx_model.input_names,  # type: ignore[attr-defined]
            output_names=onnx_model.output_names,  # type: ignore[attr-defined]
        )

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        self.logger.git_status_repos.append(repo_file_path)

    # ------------------------------------------------------------------
    # Multi-GPU
    # ------------------------------------------------------------------
    def _configure_multi_gpu(self) -> None:
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.cfg["multi_gpu"] = None
            return

        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        self.cfg["multi_gpu"] = {
            "global_rank": self.gpu_global_rank,
            "local_rank": self.gpu_local_rank,
            "world_size": self.gpu_world_size,
        }

        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank "
                f"'{self.gpu_local_rank}'."
            )
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' >= world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' >= world size '{self.gpu_world_size}'."
            )

        torch.distributed.init_process_group(
            backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size
        )
        torch.cuda.set_device(self.gpu_local_rank)
