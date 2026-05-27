# Copyright (c) 2025, Unitree Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""RSL-RL agent config for AMP PPO training on the G1 velocity task.

The runner ``class_name`` is set to ``OnPolicyAmpRunner`` and the algorithm
``class_name`` to ``AmpPPO``; :file:`scripts/rsl_rl/train.py` /
:file:`play.py` inspect those fields to route the env through the AMP runner
defined in :mod:`unitree_rl_lab.rsl_rl_amp`.

Penguin AMP V1 defaults
-----------------------
- Unified AMP spec (no ``feature_mode`` — spec is built by the env cfg).
- Discriminator: MLP ``(1024, 512, 256)`` with ELU + R1 gradient penalty.
- ``amp_reward_coef`` starts at ``0.2`` and is managed by the only-up
  :class:`AmpRewardCurriculum` (alpha_init=0.2 → alpha_max=0.8 in 0.05 steps).
"""

from __future__ import annotations

from dataclasses import field

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class AmpCurriculumCfg:
    """Mirror of :class:`AmpRewardCurriculumCfg` for the agent cfg.

    Emitted under ``cfg['amp']['curriculum']`` and lifted into the algorithm
    kwargs by :meth:`AmpPPO.construct_algorithm`.
    """

    enabled: bool = True

    alpha_init: float = 0.2
    alpha_max: float = 0.8
    alpha_step: float = 0.05

    warmup_updates: int = 500
    required_consecutive_passes: int = 20
    ema_alpha: float = 0.05

    episode_length_threshold: float = 0.7
    task_reward_threshold: float = 0.6
    termination_ratio_max: float = 0.05
    tracking_score_threshold: float = 0.7


@configclass
class AmpPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Algorithm section for AMP PPO.

    Inherits every field from :class:`RslRlPpoAlgorithmCfg` (including
    ``rnd_cfg`` / ``symmetry_cfg``, which rsl_rl's Logger reads at runtime)
    and adds the AMP-specific kwargs forwarded to :class:`AmpPPO`.
    """

    # Dotted path resolved by rsl_rl.utils.resolve_callable.
    class_name: str = "unitree_rl_lab.rsl_rl_amp.algorithms.amp_ppo:AmpPPO"

    # ============================================================
    # PPO core hyperparameters (forwarded to rsl_rl.algorithms.PPO)
    # ============================================================
    # -- optimization loop
    num_learning_epochs: int = 5  # epochs per rollout
    num_mini_batches: int = 4  # PPO mini-batch split
    learning_rate: float = 5.0e-4  # initial policy LR
    schedule: str = "adaptive"  # "fixed" | "adaptive" (KL-based LR)
    optimizer: str = "adam"  # "adam" | "adamw" | "sgd" | "rmsprop"

    # -- advantage / return estimation
    gamma: float = 0.99  # discount factor
    lam: float = 0.95  # GAE λ
    normalize_advantage_per_mini_batch: bool = False

    # -- loss weighting
    entropy_coef: float = 0.005  # exploration bonus
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True

    # -- PPO clipping + trust region
    clip_param: float = 0.2  # policy ratio clip
    desired_kl: float = 0.01  # target KL for "adaptive" schedule
    max_grad_norm: float = 1.0  # global-norm grad clip

    # ============================================================
    # AMP-specific kwargs forwarded to AmpPPO.__init__
    # ============================================================
    # Penguin V1: start small, let the curriculum grow it to 0.8.
    amp_reward_coef: float = 0.2
    amp_discriminator_learning_rate: float = 1.0e-4
    amp_num_discriminator_updates: int = 1
    amp_discriminator_mini_batch_size: int = 4096
    amp_gradient_penalty_coef: float = 5.0
    amp_discriminator_weight_decay: float = 1.0e-4
    normalize_amp_obs: bool = True


@configclass
class AmpDatasetCfg:
    """AMP dataset + discriminator construction options.

    Emitted at the top level of the runner cfg under ``amp`` so
    :meth:`AmpPPO.construct_algorithm` can consume it directly.
    """

    # List of .npz motion clip files. Placeholder; override per-task.
    motion_files: list[str] = field(default_factory=list)

    # Per-clip sampling weights used by the discriminator data loader.
    # None => uniform over clips. Length must equal len(motion_files).
    motion_weights: list[float] | None = None

    # Optional default joint positions (e.g. standing pose) used for relative
    # encodings. ``None`` => MotionDataset pulls from env.unwrapped.amp_default_joint_pos.
    default_joint_pos: list[float] | None = None

    # Wrap-around transitions for cyclic gait clips.
    wrap_around: bool = True

    # Discriminator architecture / loss kwargs (forwarded to AmpDiscriminator).
    discriminator: dict = field(
        default_factory=lambda: {
            "hidden_dims": (1024, 512, 256),
            "activation": "elu",
            "dropout": 0.0,
            "reward_style": "log",
            "reward_eps": 1.0e-4,
            "reward_clip": None,
            "use_transition_input": True,
        }
    )

    # Only-up AMP reward curriculum. Lifted into alg kwargs by construct_algorithm.
    curriculum: AmpCurriculumCfg = AmpCurriculumCfg()


@configclass
class AmpPPORunnerCfg(BasePPORunnerCfg):
    """AMP PPO runner config.

    Inherits generic runner plumbing (``seed``, ``device``, ``resume``,
    ``load_run``, ``load_checkpoint``, ``logger``, ``check_for_nan``, …) from
    :class:`BasePPORunnerCfg`. Everything the user is likely to tune —
    rollout length, network size, PPO hyperparams, AMP params — is overridden
    here explicitly so the whole training recipe is visible in one file.
    """

    # ============================================================
    # Runner plumbing
    # ============================================================
    class_name: str = "OnPolicyAmpRunner"

    num_steps_per_env: int = 24  # per-env rollout length (batch = num_envs * this)
    max_iterations: int = 500000  # total training iterations
    save_interval: int = 1000  # checkpoint every N iterations
    experiment_name: str = ""  # empty = uses task name
    empirical_normalization: bool = False

    # ============================================================
    # Actor / Critic network (policy = deprecated rsl_rl ≤ 4.x API;
    # handle_deprecated_rsl_rl_cfg in train.py auto-converts to actor/critic
    # RslRlMLPModelCfg for rsl_rl ≥ 4.0.0, so this is the single tuning knob).
    # 通过train里面的 handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version) 来解析下面的policy字段
    # ============================================================
    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # ============================================================
    # PPO + AMP algorithm kwargs
    # ============================================================
    algorithm: AmpPpoAlgorithmCfg = AmpPpoAlgorithmCfg()

    # ============================================================
    # AMP dataset / discriminator / curriculum
    # Top-level field read by AmpPPO.construct_algorithm via cfg["amp"].
    # 通过这个实例化展开amp的参数
    # ============================================================
    amp: AmpDatasetCfg = AmpDatasetCfg(
        motion_files=[
            "/home/woan/HumanoidProject/unitree_rl_lab/motion_datasets/penguin/g1_qie_motion.npz",
        ],
    )
