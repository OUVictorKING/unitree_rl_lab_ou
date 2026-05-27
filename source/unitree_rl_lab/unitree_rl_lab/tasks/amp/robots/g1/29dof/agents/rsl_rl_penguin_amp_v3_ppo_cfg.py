# Copyright (c) 2025, Unitree Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""RSL-RL agent cfg for Penguin AMP V3 (pose-only 58-dim spec).

This cfg is intentionally **flat** — every inherited field on the three
V3 sub-cfgs (curriculum / algorithm / dataset) is re-declared here with
its effective value, so a reader can see the entire training recipe
without chasing parent classes. Values are identical to what inheritance
from :class:`AmpCurriculumCfg` / :class:`AmpPpoAlgorithmCfg` /
:class:`AmpDatasetCfg` would produce.

Key V3 decisions (everything else is just the base default made visible):

1. Env cfg entry point points at :mod:`penguin_amp_v3_env_cfg`, which
   constructs the AMP spec inline — 58-dim per frame (``joint_pos_rel``
   + ``joint_vel`` + foot/hand positions in the pelvis body frame;
   every root / orientation / end-effector-velocity row is dropped).
2. ``amp_reward_coef`` pinned to ``0.3`` (Step 2 — AMP weight lightened
   so the task reward dominates), with the curriculum fully disabled.
   Mixed reward is ``task_reward + 0.3 * style_reward``.
3. Discriminator: TienKung-style ``(1024, 512, 256)`` MLP + dropout 0.1,
   low LR ``3e-5``. R1 gradient-penalty coef lowered ``10 → 5`` and
   trunk weight-decay raised ``1e-4 → 1e-3`` (Step 2 — shift regularisation
   from GP toward direct parameter shrinkage to delay D-saturation).
4. Motion dataset: only the original (non-augmented) clip.

Task registration:
    gym id               : ``Unitree-G1-23dof-Penguin-AMP-V3``
    env cfg entry point  : ``...penguin_amp_v3_env_cfg:RobotEnvCfg``
    play cfg entry point : ``...penguin_amp_v3_env_cfg:RobotPlayEnvCfg``
    agent cfg entry point: ``...agents.rsl_rl_penguin_amp_v3_ppo_cfg:PenguinAmpV3PPORunnerCfg``
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
# Curriculum — disabled, pinned to alpha=0.3
# ==============================================================================
# ``AmpPPO`` seeds its ``alpha_amp`` from ``curriculum.alpha_init`` when
# a curriculum cfg is supplied (see ``rsl_rl_amp.algorithms.amp_ppo``
# lines ~108-115). With ``enabled=False`` the update loop is a no-op, so
# pinning ``alpha_init = alpha_max = 0.3`` keeps the effective coef at
# 0.3 for the whole run. The remaining fields are unused (EMA / warmup /
# thresholds) but are written out for readability and to document the
# parent defaults.
@configclass
class PenguinAmpV3CurriculumCfg(AmpCurriculumCfg):
    """V3 curriculum cfg — disabled, style coef pinned to 0.3."""

    # -- master switch (V3: OFF so update() is a no-op)
    enabled: bool = False

    # -- alpha schedule (V3: pinned, so init == max)
    alpha_init: float = 0.3
    alpha_max: float = 0.3
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
class PenguinAmpV3AlgorithmCfg(AmpPpoAlgorithmCfg):
    """V3 algorithm cfg — PPO defaults + AMP kwargs, all written out."""

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
    # V3 overrides vs parent defaults (0.2 / 1e-4 / 5.0):
    amp_reward_coef: float = 0.3                     # pinned by curriculum (Step 2: AMP lightened)
    amp_discriminator_learning_rate: float = 3.0e-5  # low LR → slow disc
    amp_gradient_penalty_coef: float = 5.0           # R1 GP (Step 2: 10 → 5, paired with wd ↑)
    # Parent defaults, kept explicit for readability:
    amp_num_discriminator_updates: int = 1
    amp_discriminator_mini_batch_size: int = 4096
    amp_discriminator_weight_decay: float = 1.0e-3   # trunk wd (Step 2: 1e-4 → 1e-3)
    normalize_amp_obs: bool = True


# ==============================================================================
# Dataset cfg — motion files + discriminator arch + curriculum
# ==============================================================================
@configclass
class PenguinAmpV3DatasetCfg(AmpDatasetCfg):
    """V3 dataset cfg — every field explicit (motion list set on the runner)."""

    # -- motion clip list (runner sets the concrete path; factory stays empty)
    motion_files: list[str] = field(default_factory=list)

    # -- default pose for relative joint encodings
    #    None ⇒ MotionDataset pulls ``env.unwrapped.amp_default_joint_pos``
    default_joint_pos: list[float] | None = None

    # -- wrap-around transitions for cyclic clips
    wrap_around: bool = True

    # -- discriminator architecture + loss kwargs (forwarded to AmpDiscriminator)
    discriminator: dict = field(
        default_factory=lambda: {
            # 后面可以对齐天工机器人的 [1024,512,256] 风格判别器网络结构
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
    curriculum: AmpCurriculumCfg = PenguinAmpV3CurriculumCfg()


# ==============================================================================
# Runner cfg — binds algorithm + dataset + motion clip path
# ==============================================================================
@configclass
class PenguinAmpV3PPORunnerCfg(PenguinAmpPPORunnerCfg):
    """Runner cfg for the ``Unitree-G1-23dof-Penguin-AMP-V3`` task."""

    experiment_name: str = "g1_23dof_penguin_amp_v3"

    algorithm: AmpPpoAlgorithmCfg = PenguinAmpV3AlgorithmCfg()
    amp: AmpDatasetCfg = PenguinAmpV3DatasetCfg(
        motion_files=[
            "/home/woan/HumanoidProject/unitree_rl_lab/motion_datasets/penguin/g1_qie_motion.npz",
        ],
    )
