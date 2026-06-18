# Copyright (c) 2025, Unitree Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""On-policy runner for AMP training on the installed RSL-RL 3.x API."""

from __future__ import annotations

import os
import statistics
import time
import torch
from collections import deque
from typing import Any

import rsl_rl
from rsl_rl.env import VecEnv
from rsl_rl.modules import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import store_code_state

from ..algorithms.amp_ppo import AmpPPO


class OnPolicyAmpRunner(OnPolicyRunner):
    """RSL-RL 3.x runner that constructs :class:`AmpPPO` and feeds its curriculum."""

    alg: AmpPPO

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(env=env, train_cfg=train_cfg, log_dir=log_dir, device=device)
        self._tracking_term_indices, self._tracking_max_sum = self._resolve_tracking_terms()

    def _construct_algorithm(self, obs) -> AmpPPO:  # type: ignore[override]
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)
        return AmpPPO.construct_algorithm(
            obs=obs,
            env=self.env,
            cfg=self.cfg,
            device=self.device,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:  # type: ignore[override]
        self._prepare_logging_writer()

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.train_mode()

        ep_infos: list[dict] = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            tracking_step_sum = 0.0
            tracking_steps = 0

            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    self.alg.process_env_step(obs, rewards, dones, extras)

                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None
                    tracking_step_sum += self._tracking_step_mean()
                    tracking_steps += 1

                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop
                self.alg.compute_returns(obs)

            tracking_score = 0.0
            if self._tracking_term_indices:
                tracking_score = (tracking_step_sum / max(tracking_steps, 1)) / max(self._tracking_max_sum, 1e-9)
            self.alg.set_curriculum_inputs(
                episode_length_norm=self._episode_length_norm(),
                tracking_score=float(tracking_score),
            )

            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()
            if it == start_iter and not self.disable_logs:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def save(self, path: str, infos: dict | None = None) -> None:  # type: ignore[override]
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "actor_state_dict": self.alg.actor.state_dict(),
            "critic_state_dict": self.alg.critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        saved_dict.update(self.alg.extra_state_dict())
        torch.save(saved_dict, path)

        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
        load_optimizer: bool | None = None,
    ) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        lc = dict(load_cfg) if load_cfg is not None else {}
        if load_optimizer is not None:
            lc["optimizer"] = bool(load_optimizer)

        load_actor = bool(lc.get("actor", True))
        load_critic = bool(lc.get("critic", True))
        load_opt = bool(lc.get("optimizer", True))
        load_iter = bool(lc.get("iteration", True))
        load_rnd = bool(lc.get("rnd", True))

        if "model_state_dict" in loaded_dict and load_actor and load_critic:
            resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"], strict=strict)
        elif "model_state_dict" in loaded_dict:
            _load_policy_subset(self.alg.policy, loaded_dict["model_state_dict"], load_actor, load_critic, strict)
            resumed_training = True
        else:
            resumed_training = True
            if load_actor and "actor_state_dict" in loaded_dict:
                self.alg.actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
            if load_critic and "critic_state_dict" in loaded_dict:
                self.alg.critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)

        if load_rnd and self.alg.rnd and "rnd_state_dict" in loaded_dict:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=strict)
        if load_opt and resumed_training:
            if "optimizer_state_dict" in loaded_dict:
                self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            if load_rnd and self.alg.rnd and "rnd_optimizer_state_dict" in loaded_dict:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        if load_iter and resumed_training and "iter" in loaded_dict:
            self.current_learning_iteration = loaded_dict["iter"]

        self.alg.load_amp_state(loaded_dict, load_cfg=lc, strict=strict)
        return loaded_dict.get("infos")

    def get_inference_policy(self, device: str | None = None) -> callable:  # type: ignore[override]
        self.eval_mode()
        if device is not None:
            self.alg.policy.to(device)
        return self.alg.policy.act_inference

    def train_mode(self) -> None:  # type: ignore[override]
        self.alg.train_mode()

    def eval_mode(self) -> None:  # type: ignore[override]
        self.alg.eval_mode()

    def _resolve_tracking_terms(
        self,
        names: tuple[str, ...] = ("track_lin_vel_xy", "track_ang_vel_z"),
    ) -> tuple[list[int], float]:
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
        env_u = getattr(self.env, "unwrapped", self.env)
        length_buf = getattr(env_u, "episode_length_buf", None)
        max_len = float(getattr(env_u, "max_episode_length", 0.0) or 0.0)
        if length_buf is None or max_len <= 0.0:
            return 0.0
        return float(length_buf.float().mean().item() / max_len)


def _load_policy_subset(policy, model_state_dict: dict, load_actor: bool, load_critic: bool, strict: bool) -> None:
    current = policy.state_dict()
    for key, value in model_state_dict.items():
        if key not in current:
            continue
        if load_actor and _is_actor_policy_key(key):
            current[key] = value
        elif load_critic and _is_critic_policy_key(key):
            current[key] = value
    policy.load_state_dict(current, strict=strict)


def _is_actor_policy_key(key: str) -> bool:
    return (
        key.startswith("actor.")
        or key.startswith("actor_obs_normalizer.")
        or key in {"std", "log_std"}
    )


def _is_critic_policy_key(key: str) -> bool:
    return key.startswith("critic.") or key.startswith("critic_obs_normalizer.")


__all__ = ["OnPolicyAmpRunner"]
