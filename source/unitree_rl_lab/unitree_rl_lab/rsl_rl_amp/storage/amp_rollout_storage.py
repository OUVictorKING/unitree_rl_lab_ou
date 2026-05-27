# Copyright (c) 2025, Unitree Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""Rollout storage for AMP training.

Extends ``rsl_rl.storage.RolloutStorage`` with per-step AMP observations so
that the discriminator can be trained on *transition* pairs
``(amp_obs_t, amp_obs_{t+1})`` collected from the policy's own rollouts.

Also tracks per-step task and style rewards separately for logging, even
though only the mixed reward is what the value function sees (see
``algorithms.amp_ppo.AmpPPO``).
"""

from __future__ import annotations

import torch
from collections.abc import Generator

from rsl_rl.storage import RolloutStorage


class AmpRolloutStorage(RolloutStorage):
    """Rollout storage with AMP-specific extras.

    New fields:

    - ``amp_observations``: ``(T, N, amp_dim)``, AMP obs at step ``t``.
    - ``next_amp_observations``: ``(T, N, amp_dim)``, AMP obs at step ``t+1``.
    - ``task_rewards`` / ``amp_rewards``: ``(T, N, 1)`` reward components,
      filled by :class:`AmpPPO` for bookkeeping.
    """

    class Transition(RolloutStorage.Transition):
        """Transition container with AMP extras."""

        def __init__(self) -> None:
            super().__init__()
            self.amp_observations: torch.Tensor | None = None
            """AMP obs at time ``t``."""

            self.next_amp_observations: torch.Tensor | None = None
            """AMP obs at time ``t+1``."""

            self.task_rewards: torch.Tensor | None = None
            """Raw task reward before mixing with the style reward."""

            self.amp_rewards: torch.Tensor | None = None
            """Style reward from the discriminator, after mixing coefficient."""

        def clear(self) -> None:
            self.__init__()

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs,
        actions_shape,
        amp_obs_dim: int,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            training_type=training_type,
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            obs=obs,
            actions_shape=actions_shape,
            device=device,
        )

        self.amp_obs_dim = int(amp_obs_dim)

        self.amp_observations = torch.zeros(
            num_transitions_per_env, num_envs, self.amp_obs_dim, device=device
        )
        self.next_amp_observations = torch.zeros(
            num_transitions_per_env, num_envs, self.amp_obs_dim, device=device
        )
        self.task_rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.amp_rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------
    def add_transition(self, transition: "AmpRolloutStorage.Transition") -> None:  # type: ignore[override]
        """Record one transition including AMP extras."""
        if transition.amp_observations is None or transition.next_amp_observations is None:
            raise ValueError(
                "AmpRolloutStorage.Transition requires amp_observations and next_amp_observations."
            )
        if self.step >= self.num_transitions_per_env:
            raise OverflowError(
                "Rollout buffer overflow! You should call clear() before adding new transitions."
            )

        # Copy AMP extras at the current step index *before* super() advances the cursor.
        step = self.step
        self.amp_observations[step].copy_(transition.amp_observations.view(self.num_envs, -1))
        self.next_amp_observations[step].copy_(
            transition.next_amp_observations.view(self.num_envs, -1)
        )
        if transition.task_rewards is not None:
            self.task_rewards[step].copy_(transition.task_rewards.view(-1, 1))
        if transition.amp_rewards is not None:
            self.amp_rewards[step].copy_(transition.amp_rewards.view(-1, 1))

        # Delegate core RolloutStorage bookkeeping (also increments self.step).
        super().add_transition(transition)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample_amp_transitions(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ``num_samples`` fake AMP transitions from the stored rollout.

        Returns flat tensors of shape ``(num_samples, amp_obs_dim)``.
        """
        total = self.num_envs * self.num_transitions_per_env
        if total <= 0:
            raise RuntimeError("AmpRolloutStorage is empty; call add_transition first.")
        amp_flat = self.amp_observations.reshape(total, -1)
        next_amp_flat = self.next_amp_observations.reshape(total, -1)
        indices = torch.randint(low=0, high=total, size=(num_samples,), device=self.device)
        return amp_flat[indices], next_amp_flat[indices]

    def amp_mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int
    ) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        """Yield shuffled fake AMP transitions in mini-batches.

        Useful when the user wants the discriminator to sweep over the rollout
        data once per PPO epoch instead of sampling uniformly.
        """
        total = self.num_envs * self.num_transitions_per_env
        mini_batch_size = total // num_mini_batches
        amp_flat = self.amp_observations.reshape(total, -1)
        next_amp_flat = self.next_amp_observations.reshape(total, -1)

        for _ in range(num_epochs):
            perm = torch.randperm(num_mini_batches * mini_batch_size, device=self.device)
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                idx = perm[start:stop]
                yield amp_flat[idx], next_amp_flat[idx]
