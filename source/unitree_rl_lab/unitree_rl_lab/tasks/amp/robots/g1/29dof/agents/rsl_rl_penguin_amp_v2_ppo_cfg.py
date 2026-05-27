# Copyright (c) 2025, Unitree Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""RSL-RL agent cfg for Penguin AMP V2 (V3 recipe + Step 4 hyperparams).

This agent cfg used to be the "Plan A" V2 branch. It has been
**re-purposed as a clean V3-derived recipe with Step 4 hyperparameter
changes layered on top**, preserving only the gym task id
``Unitree-G1-23dof-Penguin-AMP-V2`` so training scripts and launch
configs keep working.

The cfg is intentionally **flat** — every field on the three V2 sub-cfgs
(curriculum / algorithm / dataset) is re-declared here with its effective
value, so a reader can see the entire training recipe without chasing
parent classes.

Deltas vs V3 agent cfg (see :mod:`rsl_rl_penguin_amp_v3_ppo_cfg`):

1. ``amp_reward_coef``                    ``0.3 → 0.5``
   Step 4 — amp signal was too weak on V3 (final amp_r ≈ 0.09). Raising
   to 0.5 increases the share of style vs task. Paired with Step-4
   ``flat_pitch_l2`` in the env cfg so the stronger style push does not
   amplify the observed backward-lean reward hack.
2. ``amp_discriminator_learning_rate``    ``3.0e-5 → 1.0e-5``
   Step 4 — V3's D drifted into deep saturation (logit_fake → -10 by
   iter 12k). Halving + a bit more slows further D learning; combined
   with the ×3 GP bump below this is the main anti-saturation lever.
3. ``amp_gradient_penalty_coef``          ``5.0 → 15.0``
   Step 4 — main Lipschitz lever against D saturation. ×3 from V3.
4. Curriculum ``alpha_init / alpha_max``  ``0.3 → 0.5``
   Step 4 — kept consistent with ``amp_reward_coef``. Curriculum is
   still ``enabled=False`` so update() is a no-op; alpha is pinned.

Everything else (PPO core, weight decay, disc arch, motion files) is
identical to V3.

Task registration:
    gym id               : ``Unitree-G1-23dof-Penguin-AMP-V2``
    env cfg entry point  : ``...penguin_amp_v2_env_cfg:RobotEnvCfg``
    play cfg entry point : ``...penguin_amp_v2_env_cfg:RobotPlayEnvCfg``
    agent cfg entry point: ``...agents.rsl_rl_penguin_amp_v2_ppo_cfg:PenguinAmpV2PPORunnerCfg``
"""

from __future__ import annotations

from dataclasses import field

from isaaclab.utils import configclass

from .rsl_rl_amp_ppo_cfg import (
    AmpCurriculumCfg,
    AmpDatasetCfg,
    AmpPpoAlgorithmCfg,
)
from .rsl_rl_penguin_amp_ppo_cfg import PenguinAmpPPORunnerCfg


# ==============================================================================
# Curriculum — disabled, pinned to alpha=0.5 (Step 4)
# ==============================================================================
@configclass
class PenguinAmpV2CurriculumCfg(AmpCurriculumCfg):
    """V2 curriculum cfg — disabled, style coef pinned to 0.5 (Step 4)."""

    # -- master switch (OFF so update() is a no-op)
    enabled: bool = False

    # -- alpha schedule (pinned, init == max)
    alpha_init: float = 0.5
    alpha_max: float = 0.5
    alpha_step: float = 0.05

    # -- advance gating (unused while enabled=False)
    warmup_updates: int = 500
    required_consecutive_passes: int = 20
    ema_alpha: float = 0.05

    # -- EMA thresholds (unused while enabled=False)
    episode_length_threshold: float = 0.7
    task_reward_threshold: float = 0.6
    termination_ratio_max: float = 0.05
    tracking_score_threshold: float = 0.7


# ==============================================================================
# Algorithm cfg — PPO core + AMP kwargs, every field explicit
# ==============================================================================
@configclass
class PenguinAmpV2AlgorithmCfg(AmpPpoAlgorithmCfg):
    """V2 algorithm cfg — V3 PPO defaults + Step-4 AMP hyperparams."""

    # -- routing: resolved by rsl_rl.utils.resolve_callable
    class_name: str = "unitree_rl_lab.rsl_rl_amp.algorithms.amp_ppo:AmpPPO"

    # ---------------- PPO core: optimization loop ----------------
    num_learning_epochs: int = 5          # epochs per rollout
    num_mini_batches: int = 4             # PPO mini-batch split
    learning_rate: float = 5.0e-4         # initial policy LR
    schedule: str = "adaptive"            # "fixed" | "adaptive" (KL-based LR)
    optimizer: str = "adam"               # "adam" | "adamw" | "sgd" | "rmsprop"

    # ---------------- PPO core: advantage / return ----------------
    gamma: float = 0.99                   # discount factor
    lam: float = 0.95                     # GAE λ
    normalize_advantage_per_mini_batch: bool = False

    # ---------------- PPO core: loss weighting ----------------
    entropy_coef: float = 0.005           # exploration bonus
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True

    # ---------------- PPO core: clipping / trust region ----------------
    clip_param: float = 0.2               # policy ratio clip
    desired_kl: float = 0.01              # target KL for "adaptive" schedule
    max_grad_norm: float = 1.0            # global-norm grad clip

    # ---------------- AMP kwargs forwarded to AmpPPO.__init__ ----------------
    # Step 4 overrides vs V3 (0.3 / 3e-5 / 5.0):
    amp_reward_coef: float = 0.5                     # S4: 0.3 → 0.5  (stronger style)
    amp_discriminator_learning_rate: float = 1.0e-5  # S4: 3e-5 → 1e-5 (slower D)
    amp_gradient_penalty_coef: float = 15.0          # S4: 5.0 → 15.0  (×3 GP, anti-saturation)
    # Identical to V3:
    amp_num_discriminator_updates: int = 1
    amp_discriminator_mini_batch_size: int = 4096
    amp_discriminator_weight_decay: float = 1.0e-3
    normalize_amp_obs: bool = True


# ==============================================================================
# Dataset cfg — motion files + discriminator arch + curriculum
# ==============================================================================
@configclass
class PenguinAmpV2DatasetCfg(AmpDatasetCfg):
    """V2 dataset cfg — identical arch to V3, motion list set on the runner."""

    # -- motion clip list (runner sets the concrete path; factory stays empty)
    motion_files: list[str] = field(default_factory=list)

    # -- default pose for relative joint encodings
    #    None ⇒ MotionDataset pulls ``env.unwrapped.amp_default_joint_pos``
    default_joint_pos: list[float] | None = None

    # -- wrap-around transitions for cyclic clips
    wrap_around: bool = True

    # -- discriminator architecture + loss kwargs (same as V3)
    discriminator: dict = field(
        default_factory=lambda: {
            "hidden_dims": (1024, 512, 256),
            "activation": "elu",
            "dropout": 0.1,
            "reward_style": "log",
            "reward_eps": 1.0e-4,
            "reward_clip": None,
            "use_transition_input": True,
        }
    )

    # -- AMP reward curriculum (lifted into AmpPPO kwargs by construct_algorithm)
    curriculum: AmpCurriculumCfg = PenguinAmpV2CurriculumCfg()


# ==============================================================================
# Runner cfg — binds algorithm + dataset + motion clip path
# ==============================================================================
@configclass
class PenguinAmpV2PPORunnerCfg(PenguinAmpPPORunnerCfg):
    """Runner cfg for the ``Unitree-G1-23dof-Penguin-AMP-V2`` task."""

    experiment_name: str = "g1_23dof_penguin_amp_v2"

    algorithm: AmpPpoAlgorithmCfg = PenguinAmpV2AlgorithmCfg()
    amp: AmpDatasetCfg = PenguinAmpV2DatasetCfg(
        motion_files=[
            "/home/woan/HumanoidProject/unitree_rl_lab/motion_datasets/penguin/g1_qie_motion.npz",
        ],
    )
