import argparse
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
        entry_point_key="play_env_cfg_entry_point",
    )

    env = gym.make(args_cli.task, cfg=env_cfg)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    obs, _ = env.reset()

    robot = env.unwrapped.scene["robot"]

    # 找到 torso_link 的 body index
    body_names = robot.data.body_names
    print("Body names:", body_names)

    if "torso_link" not in body_names:
        raise ValueError(f"torso_link 不在 body_names 里，当前可用名称: {body_names}")

    body_id = body_names.index("torso_link")
    print(f"[INFO] torso_link body_id = {body_id}")

    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs

    # force / torque 张量形状: [num_envs, num_bodies, 3]
    forces = torch.zeros((num_envs, len(body_names), 3), device=device)
    torques = torch.zeros((num_envs, len(body_names), 3), device=device)

    step_count = 0
    push_interval = 200  # 每 200 个控制步推一次
    push_duration = 20  # 连续施加 20 步
    push_force = torch.tensor([200.0, 0.0, 0.0], device=device)  # 沿 x 方向推
    # 你也可以改成侧向:
    # push_force = torch.tensor([0.0, 200.0, 0.0], device=device)

    while simulation_app.is_running():
        step_count += 1

        # 默认清零
        forces.zero_()
        torques.zero_()

        # 周期性推一下 torso_link
        phase = step_count % push_interval
        if phase < push_duration:
            forces[:, body_id, :] = push_force
            print(
                f"[INFO] step={step_count}, apply force={push_force.tolist()} to torso_link"
            )

        # 关键：把外力写进机器人
        robot.set_external_force_and_torque(
            forces=forces,
            torques=torques,
            body_ids=[body_id],
        )

        # 给个零动作，单纯看推力效果
        actions = torch.zeros(
            (num_envs, env.unwrapped.action_manager.total_action_dim), device=device
        )

        with torch.inference_mode():
            obs, _, terminated, truncated, _ = env.step(actions)

        if torch.any(terminated) or torch.any(truncated):
            obs, _ = env.reset()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
