#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Play IsaacLab policy using exported JIT.")
parser.add_argument("--task", type=str, required=True, help="Task name.")
parser.add_argument(
    "--jit", type=str, required=True, help="Absolute path to policy_jit.pt"
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=None, help="Random seed.")

# AppLauncher 会加 --device / --headless
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Delayed imports
# -----------------------------------------------------------------------------
import gymnasium as gym
import torch
import numpy as np

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def extract_actor_obs(obs):
    """从 env observation 中提取 actor 用的 policy obs tensor."""
    if isinstance(obs, tuple):
        obs = obs[0]

    if isinstance(obs, dict):
        if "policy" in obs:
            out = obs["policy"]
        elif "actor" in obs:
            out = obs["actor"]
        else:
            raise KeyError(
                f"Cannot find actor obs key in dict. Keys: {list(obs.keys())}"
            )
    elif hasattr(obs, "keys"):
        # TensorDict-like
        if "policy" in obs.keys():
            out = obs["policy"]
        elif "actor" in obs.keys():
            out = obs["actor"]
        else:
            raise KeyError(f"Cannot find actor obs key. Keys: {list(obs.keys())}")
    elif isinstance(obs, torch.Tensor):
        out = obs
    else:
        raise TypeError(f"Unsupported obs type: {type(obs)}")

    if not isinstance(out, torch.Tensor):
        out = torch.as_tensor(out, device=args_cli.device)

    return out


def main():
    jit_path = os.path.abspath(os.path.expanduser(args_cli.jit))
    if not os.path.isfile(jit_path):
        raise FileNotFoundError(f"JIT file not found: {jit_path}")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )

    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    print(f"[INFO] Loading JIT policy from: {jit_path}")
    policy = torch.jit.load(jit_path, map_location=args_cli.device)
    policy.eval()

    robot = env.unwrapped.scene["robot"]
    print("\n================ ISAAC ACTUATOR CFG DEBUG ================")
    print("robot cfg actuator groups:", list(robot.cfg.actuators.keys()))
    for group_name, act_cfg in robot.cfg.actuators.items():
        print(f"\n[CFG] group = {group_name}")
        print("type =", type(act_cfg))
        for attr in [
            "joint_names_expr",
            "stiffness",
            "damping",
            "armature",
            "effort_limit",
            "velocity_limit",
            "effort_limit_sim",
            "velocity_limit_sim",
            "friction",
            "dynamic_friction",
            "viscous_friction",
        ]:
            if hasattr(act_cfg, attr):
                print(f"{attr} = {getattr(act_cfg, attr)}")
    print("=============== END ISAAC ACTUATOR CFG DEBUG ===============\n")

    obs, _ = env.reset()

    # # ====== DEBUG: 打印 reset 后第一帧 obs / action ======
    # actor_obs0 = extract_actor_obs(obs).to(args_cli.device)

    # with torch.inference_mode():
    #     actions0 = policy(actor_obs0)

    # print("\n================ ISAAC FIRST FRAME DEBUG ================")
    # print("[ISAAC] actor_obs shape:", tuple(actor_obs0.shape))
    # print("[ISAAC] action shape   :", tuple(actions0.shape))

    # # 只看第 0 个环境
    # obs0_np = actor_obs0[0].detach().cpu().numpy()
    # act0_np = actions0[0].detach().cpu().numpy()

    # np.set_printoptions(precision=8, suppress=False, linewidth=220)

    # print("\n[ISAAC] actor_obs[0] full =")
    # print(obs0_np)

    # print("\n[ISAAC] action[0] full =")
    # print(act0_np)

    # print(
    #     "\n[ISAAC] action min/max/mean =", act0_np.min(), act0_np.max(), act0_np.mean()
    # )

    # # 按 term 拆开打印，方便和 MuJoCo 对拍
    # HISTORY_LEN = 5
    # NUM_ACTIONS = 23

    # idx = 0
    # ang_vel = obs0_np[idx : idx + 3 * HISTORY_LEN]
    # idx += 3 * HISTORY_LEN
    # gravity = obs0_np[idx : idx + 3 * HISTORY_LEN]
    # idx += 3 * HISTORY_LEN
    # cmd = obs0_np[idx : idx + 3 * HISTORY_LEN]
    # idx += 3 * HISTORY_LEN
    # qpos_rel = obs0_np[idx : idx + NUM_ACTIONS * HISTORY_LEN]
    # idx += NUM_ACTIONS * HISTORY_LEN
    # qvel_rel = obs0_np[idx : idx + NUM_ACTIONS * HISTORY_LEN]
    # idx += NUM_ACTIONS * HISTORY_LEN
    # last_action = obs0_np[idx : idx + NUM_ACTIONS * HISTORY_LEN]
    # idx += NUM_ACTIONS * HISTORY_LEN

    # print("\n[ISAAC] ang_vel history =")
    # print(ang_vel.reshape(HISTORY_LEN, 3))

    # print("\n[ISAAC] gravity history =")
    # print(gravity.reshape(HISTORY_LEN, 3))

    # print("\n[ISAAC] cmd history =")
    # print(cmd.reshape(HISTORY_LEN, 3))

    # print("\n[ISAAC] qpos_rel history =")
    # print(qpos_rel.reshape(HISTORY_LEN, NUM_ACTIONS))

    # print("\n[ISAAC] qvel_rel history =")
    # print(qvel_rel.reshape(HISTORY_LEN, NUM_ACTIONS))

    # print("\n[ISAAC] last_action history =")
    # print(last_action.reshape(HISTORY_LEN, NUM_ACTIONS))

    # print("=============== END ISAAC FIRST FRAME DEBUG ===============\n")
    # # ====== DEBUG END ======
    robot = env.unwrapped.scene["robot"]

    print("\n================ ISAAC ACTUATOR RUNTIME DEBUG ================")
    print("\n================ ISAAC JOINT DEBUG ================")
    if hasattr(robot.data, "joint_pos"):
        print("joint_pos shape =", robot.data.joint_pos.shape)
        print("joint_vel shape =", robot.data.joint_vel.shape)

    joint_names = getattr(robot, "joint_names", None)
    if joint_names is not None:
        print("joint_names =", joint_names)

    try:
        print("joint_pos[0] =", robot.data.joint_pos[0].detach().cpu().numpy())
        print("joint_vel[0] =", robot.data.joint_vel[0].detach().cpu().numpy())
    except Exception as e:
        print("print joint state failed:", e)

    print("=============== END ISAAC JOINT DEBUG ===============\n")
    print("runtime actuator groups:", list(robot.actuators.keys()))

    for group_name, act in robot.actuators.items():
        print(f"\n[RUNTIME] group = {group_name}")
        print("type =", type(act))

        for attr in [
            "joint_names",
            "joint_indices",
            "stiffness",
            "damping",
            "armature",
            "effort_limit",
            "velocity_limit",
            "computed_effort_limit",
            "computed_velocity_limit",
        ]:
            if hasattr(act, attr):
                val = getattr(act, attr)
                try:
                    if hasattr(val, "detach"):
                        val = val.detach().cpu()
                except Exception:
                    pass
                print(f"{attr} = {val}")

    print("=============== END ISAAC ACTUATOR RUNTIME DEBUG ===============\n")

    step_count = 0

    with torch.inference_mode():
        while simulation_app.is_running():
            actor_obs = extract_actor_obs(obs).to(args_cli.device)

            # 直接用 JIT actor 推理
            actions = policy(actor_obs)

            # 保险检查
            if not torch.isfinite(actions).all():
                print("[ERROR] Non-finite action detected.")
                break

            obs, rew, terminated, truncated, info = env.step(actions)

            step_count += 1
            if step_count % 200 == 0:
                print(
                    f"[STEP {step_count}] "
                    f"action_mean={actions.mean().item():.4f}, "
                    f"action_std={actions.std().item():.4f}"
                )

            # 如果是单环境，可自动 reset
            done = terminated | truncated
            if isinstance(done, torch.Tensor):
                if done.any():
                    obs, _ = env.reset()
            else:
                if done:
                    obs, _ = env.reset()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
