# Copyright (c) 2025, Unitree Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""Only-up AMP reward-coefficient curriculum.

The :class:`AmpRewardCurriculum` monotonically *raises* ``alpha_amp`` (the
style-reward mixing coefficient) whenever a conjunction of four EMA gating
metrics is simultaneously satisfied for a configurable number of
consecutive iterations. It never decreases the coefficient — we assume that
making AMP more dominant should only happen once policy competence is
established, and stage backtracking would re-expose the policy to a
regime its value function no longer represents.

Gating metrics (AND semantics)
------------------------------
1. ``episode_length_ema`` ≥ ``episode_length_threshold``
2. ``task_reward_ema`` ≥ ``task_reward_threshold``
3. ``termination_ratio_ema`` ≤ ``termination_ratio_max`` (fraction of
   episodes that ended via non-timeout termination — bad_orientation /
   fell-below-min-height / etc.)
4. ``tracking_score_ema`` ≥ ``tracking_score_threshold`` — a scalar in
   ``[0, 1]`` computed by the runner from the tracking reward terms.

Stage advance
-------------
- Up to ``warmup_updates`` iterations, all checks are skipped. The
  coefficient stays at ``alpha_init``.
- After warmup, the four gates are evaluated each iteration. When all four
  are satisfied for ``required_consecutive_passes`` calls in a row, we
  advance one stage: ``alpha_amp ← min(alpha_amp + alpha_step, alpha_max)``.
  The consecutive counter is reset and the next stage starts.
- When ``alpha_amp`` reaches ``alpha_max``, the curriculum is marked
  ``saturated`` and no further advances are attempted.

Checkpointing
-------------
The curriculum's state dict (EMA values, alpha, stage counters) is
serializable and round-trips via :meth:`save_state` / :meth:`load_state` so
training can resume at the exact curriculum point.
"""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass
class AmpRewardCurriculumCfg:
    """Hyperparameters for the only-up AMP reward curriculum (Phase-1 defaults)."""

    enabled: bool = True

    # Stage schedule (all scalar).
    alpha_init: float = 0.2
    alpha_max: float = 0.8
    alpha_step: float = 0.05

    # Warmup + advance cadence.
    warmup_updates: int = 500
    required_consecutive_passes: int = 20
    ema_alpha: float = 0.05

    # Gating thresholds.
    episode_length_threshold: float = 0.7  # fraction of max_episode_length
    task_reward_threshold: float = 0.6     # mean task reward per step
    termination_ratio_max: float = 0.05    # ≤ 5% non-timeout terminations per step
    tracking_score_threshold: float = 0.7  # (tracking reward sum) / (max tracking sum)

    # Two-phase mode (V3 Plan B):
    # When True, stage 0 is a pure task-learning phase — ``alpha_amp`` stays
    # at 0 and the caller (AmpPPO) should also skip discriminator updates
    # (via :attr:`AmpRewardCurriculum.disc_training_enabled`). On the first
    # advance (stage 0 → 1) ``alpha_amp`` jumps to ``alpha_init`` and
    # subsequent advances add ``alpha_step`` as usual. Default ``False`` so
    # existing configs behave exactly as before.
    warmup_disables_amp: bool = False


class AmpRewardCurriculum:
    """EMA-gated monotonic advance of ``alpha_amp``.

    The curriculum owns the current ``alpha_amp`` and a few running EMAs.
    The AMP algorithm calls :meth:`update` once per iteration with the
    metrics collected during the rollout; the method returns the (possibly
    advanced) ``alpha_amp`` and a dict of log scalars.
    """

    def __init__(self, cfg: AmpRewardCurriculumCfg) -> None:
        self.cfg = cfg
        # Two-phase mode: stage-0 runs at alpha=0 (AMP reward off) until the
        # first advance, at which point alpha_amp jumps to alpha_init.
        if cfg.warmup_disables_amp:
            self.alpha_amp = 0.0
        else:
            self.alpha_amp = float(cfg.alpha_init)

        self._update_count = 0
        self._consecutive_pass_count = 0
        self._stage = 0

        # EMAs — None until first update so the first tick is the exact
        # observed value (rather than a slow rise from 0).
        self._ema_episode_length: float | None = None
        self._ema_task_reward: float | None = None
        self._ema_termination_ratio: float | None = None
        self._ema_tracking_score: float | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def saturated(self) -> bool:
        return self.alpha_amp >= self.cfg.alpha_max - 1e-9

    @property
    def stage(self) -> int:
        return self._stage

    @property
    def disc_training_enabled(self) -> bool:
        """Whether the discriminator should be trained this iteration.

        In the default (one-phase) mode this is always ``True``. With
        ``warmup_disables_amp=True`` the discriminator is frozen while the
        curriculum is still at stage 0 so the policy can learn the task
        against a fixed (random-init) disc; training switches on as soon as
        the first advance fires.
        """
        if not self.cfg.warmup_disables_amp:
            return True
        return self._stage >= 1

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def _ema(self, prev: float | None, val: float) -> float:
        if prev is None:
            return float(val)
        a = float(self.cfg.ema_alpha)
        return (1.0 - a) * prev + a * float(val)

    def update(
        self,
        *,
        episode_length_norm: float,
        task_reward_mean: float,
        termination_ratio: float,
        tracking_score: float,
    ) -> dict[str, Any]:
        """Tick the curriculum with this iteration's metrics.

        Parameters
        ----------
        episode_length_norm:
            Mean episode length divided by the env's ``max_episode_length``
            (a fraction in ``[0, 1]``). Use the full denominator so the
            threshold is semantically "envs survived X% of episode".
        task_reward_mean:
            Mean of the raw (pre-mixing) task reward across the rollout.
        termination_ratio:
            Fraction of env steps that ended via a non-timeout termination.
        tracking_score:
            Single scalar in ``[0, 1]`` summarizing tracking competence.
            Typically ``(track_lin_vel + track_ang_vel) / max_sum``.
        """
        cfg = self.cfg
        self._update_count += 1

        # Update EMAs (always, even during warmup, so logs are meaningful).
        self._ema_episode_length = self._ema(self._ema_episode_length, episode_length_norm)
        self._ema_task_reward = self._ema(self._ema_task_reward, task_reward_mean)
        self._ema_termination_ratio = self._ema(self._ema_termination_ratio, termination_ratio)
        self._ema_tracking_score = self._ema(self._ema_tracking_score, tracking_score)

        advanced = False
        if not cfg.enabled or self.saturated or self._update_count <= cfg.warmup_updates:
            self._consecutive_pass_count = 0
        else:
            passes = (
                self._ema_episode_length >= cfg.episode_length_threshold
                and self._ema_task_reward >= cfg.task_reward_threshold
                and self._ema_termination_ratio <= cfg.termination_ratio_max
                and self._ema_tracking_score >= cfg.tracking_score_threshold
            )
            if passes:
                self._consecutive_pass_count += 1
                if self._consecutive_pass_count >= cfg.required_consecutive_passes:
                    # First advance in warmup-disables-amp mode: jump from 0
                    # to alpha_init (not alpha_step). Subsequent advances
                    # add alpha_step like the default schedule.
                    if cfg.warmup_disables_amp and self._stage == 0:
                        new_alpha = min(cfg.alpha_init, cfg.alpha_max)
                    else:
                        new_alpha = min(self.alpha_amp + cfg.alpha_step, cfg.alpha_max)
                    if new_alpha > self.alpha_amp + 1e-12:
                        advanced = True
                        self.alpha_amp = float(new_alpha)
                        self._stage += 1
                    self._consecutive_pass_count = 0
            else:
                self._consecutive_pass_count = 0

        return {
            "amp_curr/alpha_amp": float(self.alpha_amp),
            "amp_curr/stage": int(self._stage),
            "amp_curr/update_count": int(self._update_count),
            "amp_curr/consecutive_pass": int(self._consecutive_pass_count),
            "amp_curr/ema_episode_length": float(self._ema_episode_length),
            "amp_curr/ema_task_reward": float(self._ema_task_reward),
            "amp_curr/ema_termination_ratio": float(self._ema_termination_ratio),
            "amp_curr/ema_tracking_score": float(self._ema_tracking_score),
            "amp_curr/advanced": 1.0 if advanced else 0.0,
            "amp_curr/saturated": 1.0 if self.saturated else 0.0,
            "amp_curr/disc_training_enabled": 1.0 if self.disc_training_enabled else 0.0,
        }

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def save_state(self) -> dict[str, Any]:
        return {
            "alpha_amp": float(self.alpha_amp),
            "update_count": int(self._update_count),
            "consecutive_pass_count": int(self._consecutive_pass_count),
            "stage": int(self._stage),
            "ema_episode_length": self._ema_episode_length,
            "ema_task_reward": self._ema_task_reward,
            "ema_termination_ratio": self._ema_termination_ratio,
            "ema_tracking_score": self._ema_tracking_score,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self.alpha_amp = float(state["alpha_amp"])
        self._update_count = int(state["update_count"])
        self._consecutive_pass_count = int(state["consecutive_pass_count"])
        self._stage = int(state["stage"])
        self._ema_episode_length = state.get("ema_episode_length")
        self._ema_task_reward = state.get("ema_task_reward")
        self._ema_termination_ratio = state.get("ema_termination_ratio")
        self._ema_tracking_score = state.get("ema_tracking_score")
