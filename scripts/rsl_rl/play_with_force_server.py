# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a checkpoint and accept external force commands over UDP."""

import argparse
import json
import socket
import threading
import time
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Play an RL agent with external force server."
)
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos."
)
parser.add_argument(
    "--video_length", type=int, default=200, help="Recorded video length in steps."
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments."
)
parser.add_argument("--task", type=str, default=None, help="Task name.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use pretrained checkpoint from Nucleus.",
)
parser.add_argument(
    "--real-time",
    action="store_true",
    default=False,
    help="Run in real-time if possible.",
)

# force server args
parser.add_argument("--force_host", type=str, default="127.0.0.1", help="UDP host.")
parser.add_argument("--force_port", type=int, default=5005, help="UDP port.")

# append RSL-RL cli args
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# imports after app launch
# -----------------------------------------------------------------------------
import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


# -----------------------------------------------------------------------------
# global force state
# -----------------------------------------------------------------------------
FORCE_LOCK = threading.Lock()
FORCE_STATE = {
    "enabled": False,
    "body_name": "torso_link",
    "force": [0.0, 0.0, 0.0],
    "torque": [0.0, 0.0, 0.0],
    "remaining_steps": 0,
}


# -----------------------------------------------------------------------------
# UDP server
# -----------------------------------------------------------------------------
def force_server(host: str, port: int):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    print(f"[ForceServer] Listening on udp://{host}:{port}")

    while True:
        try:
            data, addr = sock.recvfrom(4096)
            msg = data.decode("utf-8").strip()
            payload = json.loads(msg)

            body_name = payload.get("body_name", "torso_link")
            force = payload.get("force", [0.0, 0.0, 0.0])
            torque = payload.get("torque", [0.0, 0.0, 0.0])
            steps = int(payload.get("steps", 1))

            with FORCE_LOCK:
                FORCE_STATE["enabled"] = True
                FORCE_STATE["body_name"] = body_name
                FORCE_STATE["force"] = [
                    float(force[0]),
                    float(force[1]),
                    float(force[2]),
                ]
                FORCE_STATE["torque"] = [
                    float(torque[0]),
                    float(torque[1]),
                    float(torque[2]),
                ]
                FORCE_STATE["remaining_steps"] = max(steps, 1)

            print(
                f"[ForceServer] New command from {addr}: "
                f"body={body_name}, force={FORCE_STATE['force']}, "
                f"torque={FORCE_STATE['torque']}, steps={FORCE_STATE['remaining_steps']}"
            )
        except Exception as e:
            print(f"[ForceServer] Bad packet: {e}")


# -----------------------------------------------------------------------------
# articulation force helpers
# -----------------------------------------------------------------------------
def find_body_index(robot, body_name: str) -> int:
    names = robot.body_names
    if body_name not in names:
        raise ValueError(f"Body '{body_name}' not found. Available bodies: {names}")
    return names.index(body_name)


def clear_force_for_body(env, body_name: str = "torso_link"):
    robot = env.unwrapped.scene["robot"]
    device = env.unwrapped.device
    body_idx = find_body_index(robot, body_name)

    forces = torch.zeros((env.num_envs, 1, 3), device=device, dtype=torch.float32)
    torques = torch.zeros((env.num_envs, 1, 3), device=device, dtype=torch.float32)

    robot.set_external_force_and_torque(
        forces=forces,
        torques=torques,
        body_ids=[body_idx],
    )
    robot.write_data_to_sim()


def apply_force_if_needed(env, debug_every=10):
    robot = env.unwrapped.scene["robot"]
    device = env.unwrapped.device

    with FORCE_LOCK:
        enabled = FORCE_STATE["enabled"]
        body_name = FORCE_STATE["body_name"]
        force = list(FORCE_STATE["force"])
        torque = list(FORCE_STATE["torque"])
        remaining_steps = int(FORCE_STATE["remaining_steps"])

    if not enabled or remaining_steps <= 0:
        return

    body_idx = find_body_index(robot, body_name)

    # IMPORTANT:
    # use shape (num_envs, 1, 3) when passing body_ids=[body_idx]
    forces = torch.zeros((env.num_envs, 1, 3), device=device, dtype=torch.float32)
    torques = torch.zeros((env.num_envs, 1, 3), device=device, dtype=torch.float32)

    forces[:, 0, :] = torch.tensor(force, device=device, dtype=torch.float32)
    torques[:, 0, :] = torch.tensor(torque, device=device, dtype=torch.float32)

    robot.set_external_force_and_torque(
        forces=forces,
        torques=torques,
        body_ids=[body_idx],
    )
    robot.write_data_to_sim()

    # debug print every few steps
    if remaining_steps % debug_every == 0 or remaining_steps <= 3:
        try:
            root_lin_vel = robot.data.root_lin_vel_w[0].detach().cpu().numpy()
            root_ang_vel = robot.data.root_ang_vel_w[0].detach().cpu().numpy()
            print(
                f"[ForceDebug] body={body_name}, idx={body_idx}, "
                f"force={force}, torque={torque}, remaining={remaining_steps}, "
                f"root_lin_vel={root_lin_vel}, root_ang_vel={root_ang_vel}"
            )
        except Exception as e:
            print(f"[ForceDebug] print failed: {e}")

    with FORCE_LOCK:
        FORCE_STATE["remaining_steps"] -= 1
        if FORCE_STATE["remaining_steps"] <= 0:
            FORCE_STATE["enabled"] = False


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(
        args_cli.task, args_cli
    )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] No pre-trained checkpoint available for this task.")
            return
    elif args_cli.checkpoint:
        # IMPORTANT: expand '~'
        ckpt = os.path.expanduser(args_cli.checkpoint)
        resume_path = retrieve_file_path(ckpt)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )

    log_dir = os.path.dirname(resume_path)

    # create environment
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    installed_version = version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    agent_cfg_dict = agent_cfg.to_dict()

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(
            env, agent_cfg_dict, log_dir=None, device=agent_cfg.device
        )
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(
            env, agent_cfg_dict, log_dir=None, device=agent_cfg.device
        )
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    runner.load(resume_path)

    # inference policy
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # start UDP server thread
    server_thread = threading.Thread(
        target=force_server,
        args=(args_cli.force_host, args_cli.force_port),
        daemon=True,
    )
    server_thread.start()

    print("[INFO] Force server ready.")
    print("[INFO] Example sender: python send_force.py --fx 300 --steps 20")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()

    timestep = 0

    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            with FORCE_LOCK:
                active = FORCE_STATE["enabled"]
                body_name = FORCE_STATE["body_name"]

            if active:
                apply_force_if_needed(env)
            else:
                clear_force_for_body(env, body_name)

            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
