import gymnasium as gym

gym.register(
    id="Unitree-G1-23dof-Velocity-AMP",
    entry_point="unitree_rl_lab.tasks.amp.envs:UnitreeAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_amp_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_amp_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.agents.rsl_rl_amp_ppo_cfg:AmpPPORunnerCfg"
        ),
    },
)

# Penguin-style AMP task — inherits the velocity-AMP baseline and overrides
# reward scales / command range so the discriminator can actually express
# the waddling style. See penguin_amp_env_cfg.py for the full rationale.
gym.register(
    id="Unitree-G1-23dof-Penguin-AMP",
    entry_point="unitree_rl_lab.tasks.amp.envs:UnitreeAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.penguin_amp_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.penguin_amp_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.agents.rsl_rl_penguin_amp_ppo_cfg:PenguinAmpPPORunnerCfg"
        ),
    },
)

# Penguin AMP V2 (Plan A) — trimmed 61-dim AMP spec (stack_k=1), fixed
# amp_reward_coef=0.3 (curriculum disabled), softened action_rate /
# base_linear_velocity. V1 remains registered and unchanged.
gym.register(
    id="Unitree-G1-23dof-Penguin-AMP-V2",
    entry_point="unitree_rl_lab.tasks.amp.envs:UnitreeAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.penguin_amp_v2_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.penguin_amp_v2_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.agents.rsl_rl_penguin_amp_v2_ppo_cfg:PenguinAmpV2PPORunnerCfg"
        ),
    },
)

# Penguin AMP V3 — pose-only 58-dim AMP spec (joint pos/vel + foot/hand
# positions in pelvis body frame). Style coef pinned to 0.5, curriculum
# disabled, non-augmented motion clip only. Uses its own env cfg so V2
# stays unchanged.
gym.register(
    id="Unitree-G1-23dof-Penguin-AMP-V3",
    entry_point="unitree_rl_lab.tasks.amp.envs:UnitreeAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.penguin_amp_v3_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.penguin_amp_v3_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.agents.rsl_rl_penguin_amp_v3_ppo_cfg:PenguinAmpV3PPORunnerCfg"
        ),
    },
)
