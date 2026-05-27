# # Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# """Script to train RL agent with RSL-RL."""

# """Launch Isaac Sim Simulator first."""


# import gymnasium as gym
# import pathlib
# import sys

# sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
# from list_envs import import_packages  # noqa: F401

# sys.path.pop(0)

# tasks = []
# for task_spec in gym.registry.values():
#     if "Unitree" in task_spec.id and "Isaac" not in task_spec.id:
#         tasks.append(task_spec.id)

# import argparse

# import argcomplete

# from isaaclab.app import AppLauncher

# # local imports
# import cli_args  # isort: skip

# # add argparse arguments
# parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
# parser.add_argument(
#     "--video", action="store_true", default=False, help="Record videos during training."
# )
# parser.add_argument(
#     "--video_length",
#     type=int,
#     default=200,
#     help="Length of the recorded video (in steps).",
# )
# parser.add_argument(
#     "--video_interval",
#     type=int,
#     default=2000,
#     help="Interval between video recordings (in steps).",
# )
# parser.add_argument(
#     "--num_envs", type=int, default=None, help="Number of environments to simulate."
# )
# parser.add_argument(
#     "--task", type=str, default=None, choices=tasks, help="Name of the task."
# )
# parser.add_argument(
#     "--seed", type=int, default=None, help="Seed used for the environment"
# )
# parser.add_argument(
#     "--max_iterations", type=int, default=None, help="RL Policy training iterations."
# )
# parser.add_argument(
#     "--distributed",
#     action="store_true",
#     default=False,
#     help="Run training with multiple GPUs or nodes.",
# )
# # append RSL-RL cli arguments
# cli_args.add_rsl_rl_args(parser)
# # append AppLauncher cli args
# AppLauncher.add_app_launcher_args(parser)
# argcomplete.autocomplete(parser)
# args_cli, hydra_args = parser.parse_known_args()

# # always enable cameras to record video
# if args_cli.video:
#     args_cli.enable_cameras = True

# # clear out sys.argv for Hydra
# sys.argv = [sys.argv[0]] + hydra_args

# # launch omniverse app
# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app

# """Check for minimum supported RSL-RL version."""

# import importlib.metadata as metadata
# import platform

# from packaging import version

# # for distributed training, check minimum supported rsl-rl version
# RSL_RL_VERSION = "2.3.1"
# installed_version = metadata.version("rsl-rl-lib")
# if args_cli.distributed and version.parse(installed_version) < version.parse(
#     RSL_RL_VERSION
# ):
#     if platform.system() == "Windows":
#         cmd = [
#             r".\isaaclab.bat",
#             "-p",
#             "-m",
#             "pip",
#             "install",
#             f"rsl-rl-lib=={RSL_RL_VERSION}",
#         ]
#     else:
#         cmd = [
#             "./isaaclab.sh",
#             "-p",
#             "-m",
#             "pip",
#             "install",
#             f"rsl-rl-lib=={RSL_RL_VERSION}",
#         ]
#     print(
#         f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
#         f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
#         f"\n\n\t{' '.join(cmd)}\n"
#     )
#     exit(1)

# """Rest everything follows."""

# import gymnasium as gym
# import inspect
# import os
# import shutil
# import torch
# from datetime import datetime

# from rsl_rl.runners import (
#     OnPolicyRunner,
# )  # TODO: Consider printing the experiment name in the terminal.

# import isaaclab_tasks  # noqa: F401
# from isaaclab.envs import (
#     DirectMARLEnv,
#     DirectMARLEnvCfg,
#     DirectRLEnvCfg,
#     ManagerBasedRLEnvCfg,
#     multi_agent_to_single_agent,
# )
# from isaaclab.utils.dict import print_dict
# from isaaclab.utils.io import dump_yaml
# from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

# from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

# from isaaclab_tasks.utils import get_checkpoint_path
# from isaaclab_tasks.utils.hydra import hydra_task_config

# import unitree_rl_lab.tasks  # noqa: F401
# from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
# torch.backends.cudnn.deterministic = False
# torch.backends.cudnn.benchmark = False


# @hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
# def main(
#     env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
#     agent_cfg: RslRlOnPolicyRunnerCfg,
# ):
#     """Train with RSL-RL agent."""
#     # override configurations with non-hydra CLI arguments
#     agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
#     env_cfg.scene.num_envs = (
#         args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
#     )
#     agent_cfg.max_iterations = (
#         args_cli.max_iterations
#         if args_cli.max_iterations is not None
#         else agent_cfg.max_iterations
#     )

#     # set the environment seed
#     # note: certain randomizations occur in the environment initialization so we set the seed here
#     env_cfg.seed = agent_cfg.seed
#     env_cfg.sim.device = (
#         args_cli.device if args_cli.device is not None else env_cfg.sim.device
#     )

#     # multi-gpu training configuration
#     if args_cli.distributed:
#         env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
#         agent_cfg.device = f"cuda:{app_launcher.local_rank}"

#         # set seed to have diversity in different threads
#         seed = agent_cfg.seed + app_launcher.local_rank
#         env_cfg.seed = seed
#         agent_cfg.seed = seed

#     # specify directory for logging experiments
#     log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
#     log_root_path = os.path.abspath(log_root_path)
#     print(f"[INFO] Logging experiment in directory: {log_root_path}")
#     # specify directory for logging runs: {time-stamp}_{run_name}
#     log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     # This way, the Ray Tune workflow can extract experiment name.
#     print(f"Exact experiment name requested from command line: {log_dir}")
#     if agent_cfg.run_name:
#         log_dir += f"_{agent_cfg.run_name}"
#     log_dir = os.path.join(log_root_path, log_dir)

#     # create isaac environment
#     env = gym.make(
#         args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
#     )

#     # convert to single-agent instance if required by the RL algorithm
#     if isinstance(env.unwrapped, DirectMARLEnv):
#         env = multi_agent_to_single_agent(env)

#     # save resume path before creating a new log_dir
#     if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
#         resume_path = get_checkpoint_path(
#             log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
#         )

#     # wrap for video recording
#     if args_cli.video:
#         video_kwargs = {
#             "video_folder": os.path.join(log_dir, "videos", "train"),
#             "step_trigger": lambda step: step % args_cli.video_interval == 0,
#             "video_length": args_cli.video_length,
#             "disable_logger": True,
#         }
#         print("[INFO] Recording videos during training.")
#         print_dict(video_kwargs, nesting=4)
#         env = gym.wrappers.RecordVideo(env, **video_kwargs)

#     # # wrap around environment for rsl-rl
#     # env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
#     # # convert deprecated RSL-RL config fields (e.g. policy -> actor/critic)
#     # agent_cfg_dict = agent_cfg.to_dict()
#     # runner = OnPolicyRunner(
#     #     env, agent_cfg_dict, log_dir=log_dir, device=agent_cfg.device
#     # )
#     base_env = env.unwrapped

#     print("\n================ DEBUG: robot / obs info ================\n")

#     reset_out = env.reset()
#     if isinstance(reset_out, tuple):
#         obs_dict = reset_out[0]
#     else:
#         obs_dict = reset_out

#     robot = base_env.scene["robot"]

#     print("==== robot joint names ====")
#     for i, name in enumerate(robot.data.joint_names):
#         print(i, name)

#     print("\n==== default joint pos (all joints) ====")
#     print(robot.data.default_joint_pos[0].detach().cpu().numpy())

#     print("\n==== current joint pos (all joints) ====")
#     print(robot.data.joint_pos[0].detach().cpu().numpy())

#     print("\n==== current joint vel (all joints) ====")
#     print(robot.data.joint_vel[0].detach().cpu().numpy())

#     print("\n==== root_ang_vel_b raw ====")
#     print(robot.data.root_ang_vel_b[0].detach().cpu().numpy())

#     print("\n==== projected_gravity_b raw ====")
#     print(robot.data.projected_gravity_b[0].detach().cpu().numpy())

#     print("\n==== command raw ====")
#     cmd = base_env.command_manager.get_command("base_velocity")
#     print(cmd[0].detach().cpu().numpy())

#     print("\n==== last_action raw ====")
#     print(base_env.action_manager.action[0].detach().cpu().numpy())

#     print("\n==== observation keys ====")
#     print(list(obs_dict.keys()))

#     policy_obs = obs_dict["policy"][0].detach().cpu()

#     print("\n==== policy obs shape ====")
#     print(policy_obs.shape)

#     print("\n==== policy obs segmented ====")
#     print("base_ang_vel history   [0:15]   =")
#     print(policy_obs[0:15].numpy())

#     print("\nprojected_gravity hist [15:30]  =")
#     print(policy_obs[15:30].numpy())

#     print("\nvelocity_commands hist [30:45]  =")
#     print(policy_obs[30:45].numpy())

#     print("\njoint_pos_rel hist     [45:160] =")
#     print(policy_obs[45:160].numpy())

#     print("\njoint_vel_rel hist     [160:275] =")
#     print(policy_obs[160:275].numpy())

#     print("\nlast_action hist       [275:390] =")
#     print(policy_obs[275:390].numpy())

#     print("\n================ END DEBUG INFO ================\n")

#     # wrap around environment for rsl-rl
#     env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

#     installed_version = metadata.version("rsl-rl-lib")
#     agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
#     agent_cfg_dict = agent_cfg.to_dict()
#     print(agent_cfg_dict["actor"])
#     print(agent_cfg_dict["critic"])

#     runner = OnPolicyRunner(
#         env, agent_cfg_dict, log_dir=log_dir, device=agent_cfg.device
#     )

#     # # =========================
#     # # Debug print for sim2sim
#     # # =========================
#     # print("\n================ DEBUG: robot / obs info ================\n")

#     # # 注意：这里要取 wrapper 里面真正的 IsaacLab env
#     # base_env = env.unwrapped
#     # robot = base_env.scene["robot"]

#     # print("==== robot joint names ====")
#     # for i, name in enumerate(robot.joint_names):
#     #     print(i, name)

#     # print("\n==== default joint pos (all joints) ====")
#     # print(robot.data.default_joint_pos[0].cpu().numpy())

#     # # 打印动作项绑定的 joint
#     # try:
#     #     action_term = base_env.action_manager._terms["JointPositionAction"]
#     #     print("\n==== action joint ids ====")
#     #     print(action_term._joint_ids)

#     #     print("\n==== action joint names ====")
#     #     print([robot.joint_names[i] for i in action_term._joint_ids])

#     #     print("\n==== default joint pos (action joints only) ====")
#     #     action_joint_ids = action_term._joint_ids
#     #     print(robot.data.default_joint_pos[0, action_joint_ids].cpu().numpy())
#     # except Exception as e:
#     #     print(f"[WARN] failed to print action joint info: {e}")

#     # # 打印观测
#     # try:
#     #     obs = env.get_observations()
#     #     if isinstance(obs, tuple):
#     #         obs = obs[0]

#     #     print("\n==== observation keys ====")
#     #     if hasattr(obs, "keys"):
#     #         print(list(obs.keys()))
#     #     else:
#     #         print(type(obs))

#     #     if hasattr(obs, "keys"):
#     #         if "policy" in obs:
#     #             print("\n==== policy obs shape ====")
#     #             print(obs["policy"].shape)
#     #             print("policy obs first env:")
#     #             print(obs["policy"][0].detach().cpu().numpy())

#     #         if "critic" in obs:
#     #             print("\n==== critic obs shape ====")
#     #             print(obs["critic"].shape)
#     # except Exception as e:
#     #     print(f"[WARN] failed to print observations: {e}")

#     # print("\n================ END DEBUG INFO ================\n")

#     # # wrap around environment for rsl-rl
#     # env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

#     # # 原来的
#     # runner = OnPolicyRunner(
#     #     env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
#     # )

#     # write git state to logs
#     runner.add_git_repo_to_log(__file__)
#     # load the checkpoint
#     if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
#         print(f"[INFO]: Loading model checkpoint from: {resume_path}")
#         # load previously trained model
#         runner.load(resume_path)

#     # dump the configuration into log-directory
#     dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
#     dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
#     export_deploy_cfg(env.unwrapped, log_dir)
#     # copy the environment configuration file to the log directory
#     shutil.copy(
#         inspect.getfile(env_cfg.__class__),
#         os.path.join(
#             log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))
#         ),
#     )

#     # run training
#     runner.learn(
#         num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False
#     )

#     # close the simulator
#     env.close()


# if __name__ == "__main__":
#     # run the main function
#     main()
#     # close sim app
#     simulation_app.close()


# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""


import gymnasium as gym
import pathlib
import sys

sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401

sys.path.pop(0)

tasks = []
for task_spec in gym.registry.values():
    if "Unitree" in task_spec.id and "Isaac" not in task_spec.id:
        tasks.append(task_spec.id)

import argparse

import argcomplete

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=2000,
    help="Interval between video recordings (in steps).",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--max_iterations", type=int, default=None, help="RL Policy training iterations."
)
parser.add_argument(
    "--distributed",
    action="store_true",
    default=False,
    help="Run training with multiple GPUs or nodes.",
)
# ---- W&B dual logging (optional) ---------------------------------------
# When ``--wandb_project`` is set, wandb.init(sync_tensorboard=True) is called
# before the rsl_rl runner opens its TB writer. All TB scalars (PPO loss,
# amp_discriminator_*, amp_curr/*, task_reward_mean, ...) are mirrored to W&B
# with no extra code in the runner or algorithm.
parser.add_argument(
    "--wandb_project",
    type=str,
    default=None,
    help="W&B project name. When set, enables W&B + TB dual logging via sync_tensorboard.",
)
parser.add_argument(
    "--wandb_entity", type=str, default=None, help="Optional W&B entity (team)."
)
parser.add_argument(
    "--wandb_run_name", type=str, default=None, help="Optional explicit W&B run name."
)
parser.add_argument(
    "--wandb_mode",
    type=str,
    default=None,
    choices=[None, "online", "offline", "disabled"],
    help="W&B mode. Defaults to online; 'offline' is useful for headless nodes.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
argcomplete.autocomplete(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(
    RSL_RL_VERSION
):
    if platform.system() == "Windows":
        cmd = [
            r".\isaaclab.bat",
            "-p",
            "-m",
            "pip",
            "install",
            f"rsl-rl-lib=={RSL_RL_VERSION}",
        ]
    else:
        cmd = [
            "./isaaclab.sh",
            "-p",
            "-m",
            "pip",
            "install",
            f"rsl-rl-lib=={RSL_RL_VERSION}",
        ]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import inspect
import os
import shutil
import torch
from datetime import datetime

from rsl_rl.runners import (
    OnPolicyRunner,
)  # TODO: Consider printing the experiment name in the terminal.


def _resolve_runner_class(train_cfg_dict: dict):
    """Pick the rsl_rl runner class from the agent cfg.

    Priority:
        1. ``train_cfg["runner"]["runner_class_name"]`` (explicit top-level switch).
        2. Runner-level ``class_name`` field (how isaaclab_rl already marks
           Distillation etc.).
        3. Fallback: ``train_cfg["algorithm"]["class_name"]`` — if it points at
           an AMP-flavored PPO class (``AmpPPO`` / ``AMPPPO``, including dotted
           paths like ``unitree_rl_lab...:AmpPPO``), pick the AMP runner.
        4. Default: :class:`OnPolicyRunner`.
    """
    runner_name = None

    runner_section = train_cfg_dict.get("runner")
    if isinstance(runner_section, dict):
        runner_name = runner_section.get("runner_class_name") or runner_section.get(
            "class_name"
        )
    if runner_name is None:
        runner_name = train_cfg_dict.get("class_name")

    if runner_name is None:
        alg_name = (train_cfg_dict.get("algorithm") or {}).get("class_name", "")
        leaf = alg_name.split(":")[-1].split(".")[-1]
        if leaf in ("AmpPPO", "AMPPPO"):
            runner_name = "OnPolicyAmpRunner"

    if runner_name in (None, "OnPolicyRunner"):
        return OnPolicyRunner
    if runner_name == "OnPolicyAmpRunner":
        from unitree_rl_lab.rsl_rl_amp import OnPolicyAmpRunner

        return OnPolicyAmpRunner
    if runner_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        return DistillationRunner
    raise ValueError(f"Unsupported runner class: {runner_name}")


import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

from checkpoint_compat import (
    build_checkpoint_metadata,
    get_runner_policy_state_dicts,
    load_checkpoint_summary,
    print_checkpoint_compat_report,
    task_names_match,
    wrap_runner_save_with_metadata,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlOnPolicyRunnerCfg,
):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    agent_cfg.max_iterations = (
        args_cli.max_iterations
        if args_cli.max_iterations is not None
        else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Resolve the checkpoint before creating a new log_dir.  A bare
    # ``--checkpoint /path/model.pt`` is treated as a warm-start; ``--resume``
    # keeps same-task full resume semantics.
    checkpoint_load_requested = bool(args_cli.checkpoint) or agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation"
    resume_path = None
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    elif agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 兼容性转化，rslrl需要的！
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # create runner from rsl-rl,如果希望使用其他的库，如skrl，就需要修改这个
    agent_cfg_dict = agent_cfg.to_dict()
    runner_cls = _resolve_runner_class(agent_cfg_dict)
    print(f"[INFO] Using rsl_rl runner: {runner_cls.__name__}")

    # -------- W&B + TB dual logging (must run BEFORE the runner opens TB) --
    wandb_active = False
    if args_cli.wandb_project:
        try:
            import wandb  # type: ignore

            run_name = args_cli.wandb_run_name or os.path.basename(log_dir)
            wandb.init(
                project=args_cli.wandb_project,
                entity=args_cli.wandb_entity,
                name=run_name,
                dir=log_dir,
                config={
                    "task": args_cli.task,
                    "agent_cfg": agent_cfg_dict,
                },
                sync_tensorboard=True,
                mode=args_cli.wandb_mode or "online",
            )
            wandb_active = True
            print(f"[INFO] W&B dual logging enabled: project={args_cli.wandb_project}")
        except ImportError:
            print(
                "[WARN] --wandb_project set but wandb is not installed; "
                "install with `pip install wandb` to enable dual logging."
            )
        except Exception as e:
            print(f"[WARN] wandb.init failed: {e!r}; continuing without W&B.")

    runner = runner_cls(env, agent_cfg_dict, log_dir=log_dir, device=agent_cfg.device)
    wrap_runner_save_with_metadata(
        runner,
        build_checkpoint_metadata(args_cli.task, agent_cfg.experiment_name, runner),
    )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if checkpoint_load_requested:
        if resume_path is None:
            raise RuntimeError("Checkpoint loading was requested, but no checkpoint path was resolved.")
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        checkpoint_summary = load_checkpoint_summary(resume_path)
        target_actor_sd, target_critic_sd = get_runner_policy_state_dicts(runner)
        compatible, _, _ = print_checkpoint_compat_report(
            checkpoint_summary,
            args_cli.task,
            target_actor_sd,
            target_critic_sd,
        )
        if not compatible:
            raise RuntimeError(
                "Checkpoint is not compatible with the current training task. "
                "Actor and critic state_dict keys/shapes must match exactly."
            )

        same_task = task_names_match(checkpoint_summary.task_name, args_cli.task)
        if agent_cfg.resume and same_task:
            print("[INFO] Checkpoint load mode: full resume, optimizer=True, iteration=True")
            runner.load(resume_path)
        else:
            warm_start_load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
                "amp": False,
                "amp_optimizer": False,
                "amp_normalizer": False,
            }
            if same_task:
                reason = "--checkpoint warm-start"
            else:
                reason = "cross-task warm-start"
            print(
                f"[INFO] Checkpoint load mode: {reason} actor+critic, "
                "optimizer=False, iteration=False"
            )
            runner.load(resume_path, load_cfg=warm_start_load_cfg)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    export_deploy_cfg(env.unwrapped, log_dir)
    # copy the environment configuration file to the log directory
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(
            log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))
        ),
    )

    # run training
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
    )

    # close the simulator
    env.close()

    # Close out W&B cleanly so the last TB scalars get flushed.
    if wandb_active:
        try:
            import wandb  # type: ignore

            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
