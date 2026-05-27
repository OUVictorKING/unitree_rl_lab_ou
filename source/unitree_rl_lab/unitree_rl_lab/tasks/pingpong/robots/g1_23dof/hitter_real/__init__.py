import gymnasium as gym

gym.register(
    id="Unitree-G1-23dof-Pingpong-HITTER-REAL",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hitter_real_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.hitter_real_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.pingpong.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
