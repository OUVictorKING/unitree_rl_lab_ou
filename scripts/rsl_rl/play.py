# # Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# """Script to play a checkpoint if an RL agent from RSL-RL."""

# """Launch Isaac Sim Simulator first."""

# import argparse
# from importlib.metadata import version

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
#     "--disable_fabric",
#     action="store_true",
#     default=False,
#     help="Disable fabric and use USD I/O operations.",
# )
# parser.add_argument(
#     "--num_envs", type=int, default=None, help="Number of environments to simulate."
# )
# parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# parser.add_argument(
#     "--use_pretrained_checkpoint",
#     action="store_true",
#     help="Use the pre-trained checkpoint from Nucleus.",
# )
# parser.add_argument(
#     "--real-time",
#     action="store_true",
#     default=False,
#     help="Run in real-time, if possible.",
# )
# # append RSL-RL cli arguments
# cli_args.add_rsl_rl_args(parser)
# # append AppLauncher cli args
# AppLauncher.add_app_launcher_args(parser)
# args_cli = parser.parse_args()
# # always enable cameras to record video
# if args_cli.video:
#     args_cli.enable_cameras = True

# # launch omniverse app
# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app

# """Rest everything follows."""

# import gymnasium as gym
# import os
# import time
# import torch

# from rsl_rl.runners import OnPolicyRunner

# import isaaclab_tasks  # noqa: F401
# from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
# from isaaclab.utils.assets import retrieve_file_path
# from isaaclab.utils.dict import print_dict

# # from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
# from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
# from isaaclab_rl.rsl_rl import (
#     RslRlOnPolicyRunnerCfg,
#     RslRlVecEnvWrapper,
#     export_policy_as_jit,
#     export_policy_as_onnx,
#     handle_deprecated_rsl_rl_cfg,
# )
# from isaaclab_tasks.utils import get_checkpoint_path

# import unitree_rl_lab.tasks  # noqa: F401
# from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


# def main():
#     """Play with RSL-RL agent."""
#     # parse configuration
#     env_cfg = parse_env_cfg(
#         args_cli.task,
#         device=args_cli.device,
#         num_envs=args_cli.num_envs,
#         use_fabric=not args_cli.disable_fabric,
#         entry_point_key="play_env_cfg_entry_point",
#     )
#     agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(
#         args_cli.task, args_cli
#     )

#     # specify directory for logging experiments
#     log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
#     log_root_path = os.path.abspath(log_root_path)
#     print(f"[INFO] Loading experiment from directory: {log_root_path}")
#     if args_cli.use_pretrained_checkpoint:
#         resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
#         if not resume_path:
#             print(
#                 "[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task."
#             )
#             return
#     elif args_cli.checkpoint:
#         resume_path = retrieve_file_path(args_cli.checkpoint)
#     else:
#         resume_path = get_checkpoint_path(
#             log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
#         )

#     log_dir = os.path.dirname(resume_path)

#     # create isaac environment
#     env = gym.make(
#         args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
#     )

#     # convert to single-agent instance if required by the RL algorithm
#     if isinstance(env.unwrapped, DirectMARLEnv):
#         env = multi_agent_to_single_agent(env)

#     # wrap for video recording
#     if args_cli.video:
#         video_kwargs = {
#             "video_folder": os.path.join(log_dir, "videos", "play"),
#             "step_trigger": lambda step: step == 0,
#             "video_length": args_cli.video_length,
#             "disable_logger": True,
#         }
#         print("[INFO] Recording videos during training.")
#         print_dict(video_kwargs, nesting=4)
#         env = gym.wrappers.RecordVideo(env, **video_kwargs)

#     # # wrap around environment for rsl-rl
#     # env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
#     # print(f"[INFO]: Loading model checkpoint from: {resume_path}")
#     # # load previously trained model
#     # if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
#     #     runner = OnPolicyRunner(
#     #         env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
#     #     )
#     # elif agent_cfg.class_name == "DistillationRunner":
#     #     from rsl_rl.runners import DistillationRunner

#     #     runner = DistillationRunner(
#     #         env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
#     #     )
#     # else:
#     #     raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

#     # wrap around environment for rsl-rl
#     env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

#     installed_version = version("rsl-rl-lib")
#     agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
#     agent_cfg_dict = agent_cfg.to_dict()

#     print(f"[INFO]: Loading model checkpoint from: {resume_path}")
#     # load previously trained model
#     if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
#         runner = OnPolicyRunner(
#             env, agent_cfg_dict, log_dir=None, device=agent_cfg.device
#         )
#     elif agent_cfg.class_name == "DistillationRunner":
#         from rsl_rl.runners import DistillationRunner

#         runner = DistillationRunner(
#             env, agent_cfg_dict, log_dir=None, device=agent_cfg.device
#         )
#     else:
#         raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

#     runner.load(resume_path)

#     # # obtain the trained policy for inference
#     # policy = runner.get_inference_policy(device=env.unwrapped.device)

#     # # extract the neural network module
#     # # we do this in a try-except to maintain backwards compatibility.
#     # try:
#     #     # version 2.3 onwards
#     #     policy_nn = runner.alg.policy
#     # except AttributeError:
#     #     # version 2.2 and below
#     #     policy_nn = runner.alg.actor_critic

#     # # extract the normalizer
#     # if hasattr(policy_nn, "actor_obs_normalizer"):
#     #     normalizer = policy_nn.actor_obs_normalizer
#     # elif hasattr(policy_nn, "student_obs_normalizer"):
#     #     normalizer = policy_nn.student_obs_normalizer
#     # else:
#     #     normalizer = None

#     # obtain the trained policy for inference
#     policy = runner.get_inference_policy(device=env.unwrapped.device)

#     # # for rsl-rl 5.x, use actor model directly for export
#     # if hasattr(runner.alg, "actor"):
#     #     policy_nn = runner.alg.actor
#     # elif hasattr(runner.alg, "policy"):
#     #     policy_nn = runner.alg.policy
#     # elif hasattr(runner.alg, "actor_critic"):
#     #     policy_nn = runner.alg.actor_critic
#     # else:
#     #     raise AttributeError(
#     #         f"Cannot find exportable policy network on runner.alg. "
#     #         f"Available attrs: {dir(runner.alg)}"
#     #     )

#     # # extract the normalizer
#     # if hasattr(policy_nn, "obs_normalizer"):
#     #     normalizer = policy_nn.obs_normalizer
#     # elif hasattr(policy_nn, "actor_obs_normalizer"):
#     #     normalizer = policy_nn.actor_obs_normalizer
#     # elif hasattr(policy_nn, "student_obs_normalizer"):
#     #     normalizer = policy_nn.student_obs_normalizer
#     # else:
#     #     normalizer = None

#     # export policy to onnx/jit
#     # export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
#     # export_policy_as_jit(
#     #     policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt"
#     # )
#     # export_policy_as_onnx(
#     #     policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx"
#     # )

#     dt = env.unwrapped.step_dt

#     # reset environment
#     obs = env.get_observations()
#     if version("rsl-rl-lib").startswith("2.3."):
#         obs, _ = env.get_observations()
#     timestep = 0
#     # simulate environment
#     while simulation_app.is_running():
#         start_time = time.time()
#         # run everything in inference mode
#         with torch.inference_mode():
#             # agent stepping
#             actions = policy(obs)
#             # env stepping
#             obs, _, _, _ = env.step(actions)
#         if args_cli.video:
#             timestep += 1
#             # Exit the play loop after recording one video
#             if timestep == args_cli.video_length:
#                 break

#         # time delay for real-time evaluation
#         sleep_time = dt - (time.time() - start_time)
#         if args_cli.real_time and sleep_time > 0:
#             time.sleep(sleep_time)

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

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib.metadata import version

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
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--real-time",
    action="store_true",
    default=False,
    help="Run in real-time, if possible.",
)
parser.add_argument(
    "--no_pingpong_error_plot",
    action="store_true",
    default=False,
    help="Disable the Pingpong HITTER-REAL live error plot.",
)
parser.add_argument(
    "--pingpong_error_window_s",
    type=float,
    default=6.0,
    help="Time window, in seconds, shown by the Pingpong HITTER-REAL live error plot.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
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
    export_policy_as_jit,
    export_policy_as_onnx,
)
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

from checkpoint_compat import (
    get_runner_actor_state_dict,
    load_checkpoint_summary,
    print_actor_checkpoint_compat_report,
)
from rsl_rl_compat import handle_deprecated_rsl_rl_cfg, load_runner_checkpoint


class PingpongPlayErrorMonitor:
    """Live play-only diagnostics for the HITTER-REAL command tracking errors."""

    def __init__(
        self,
        base_env,
        task_name: str | None,
        dt: float,
        *,
        enabled: bool,
        plot_enabled: bool,
        window_s: float,
    ):
        self.enabled = bool(enabled and task_name and "Pingpong-HITTER-REAL" in task_name)
        self.dt = float(dt)
        self.window_s = max(float(window_s), 1.0)
        self.elapsed_s = 0.0
        self.times: list[float] = []
        self.pos_errors: list[float] = []
        self.vel_errors: list[float] = []
        self.ori_errors: list[float] = []
        self.strike_flags: list[bool] = []
        self.hit_samples: list[tuple[float, float, float, float, float]] = []
        self._prev_t_to_hit: float | None = None
        self._active_prev = False
        self._hit_candidate: tuple[float, float, float, float, float] | None = None
        self._hit_reported = False
        self._swing_index = 0
        self._plot_tick = 0
        self._plot_enabled = False
        self._plt = None

        if not self.enabled:
            return

        try:
            self.command = base_env.command_manager.get_term("pingpong")
        except Exception as exc:  # noqa: BLE001
            print(f"[PingpongErrorMonitor] disabled: cannot get pingpong command term ({exc}).")
            self.enabled = False
            return

        if not hasattr(self.command, "_task_errors") or not hasattr(self.command, "t_to_hit"):
            print("[PingpongErrorMonitor] disabled: command term does not expose _task_errors/t_to_hit.")
            self.enabled = False
            return

        self.strike_window = float(getattr(self.command.cfg, "strike_window", 0.1))
        if plot_enabled:
            self._init_plot()

        print(
            "[PingpongErrorMonitor] enabled: logging pos/velocity/orientation errors; "
            f"strike_window=±{self.strike_window:.3f}s."
        )

    @staticmethod
    def _first_float(value, default: float = 0.0) -> float:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return default
            return float(value.detach().flatten()[0].cpu())
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _init_plot(self) -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # noqa: BLE001
            print(f"[PingpongErrorMonitor] live plot disabled: matplotlib unavailable ({exc}).")
            return

        self._plt = plt
        plt.ion()
        self.fig, self.axes = plt.subplots(3, 1, sharex=True, num="Pingpong HITTER-REAL Errors")
        labels = ("pos error (m)", "velocity error (m/s)", "ori error (1-cos)")
        colors = ("tab:blue", "tab:green", "tab:purple")
        self.error_lines = []
        self.strike_lines = []
        self.hit_vlines = []
        for ax, label, color in zip(self.axes, labels, colors, strict=True):
            line, = ax.plot([], [], color=color, linewidth=1.5)
            strike_line, = ax.plot([], [], color="red", linewidth=2.4)
            hit_vline = ax.axvline(0.0, color="red", linestyle="--", linewidth=1.0, alpha=0.75, visible=False)
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.25)
            self.error_lines.append(line)
            self.strike_lines.append(strike_line)
            self.hit_vlines.append(hit_vline)
        self.axes[-1].set_xlabel("play time (s)")
        self.fig.tight_layout()
        self.fig.show()
        self._plot_enabled = True

    def update(self) -> None:
        if not self.enabled:
            return

        self.elapsed_s += self.dt
        try:
            pos_err_t, vel_err_t, ori_err_t = self.command._task_errors()
            t_to_hit = self._first_float(self.command.t_to_hit)
            pos_err = self._first_float(pos_err_t)
            vel_err = self._first_float(vel_err_t)
            ori_err = self._first_float(ori_err_t)
            active_value = getattr(self.command, "active_swing", None)
            active = True if active_value is None else self._first_float(active_value) > 0.5
        except Exception as exc:  # noqa: BLE001
            print(f"[PingpongErrorMonitor] disabled after read failure: {exc}")
            self.enabled = False
            return

        if not active:
            self._reset_swing_tracking()
            return

        new_swing = (not self._active_prev) or (
            self._prev_t_to_hit is not None and t_to_hit > self._prev_t_to_hit + 0.20
        )
        if new_swing:
            self._swing_index += 1
            self._hit_candidate = None
            self._hit_reported = False

        in_strike = abs(t_to_hit) <= self.strike_window
        self.times.append(self.elapsed_s)
        self.pos_errors.append(pos_err)
        self.vel_errors.append(vel_err)
        self.ori_errors.append(ori_err)
        self.strike_flags.append(in_strike)
        self._trim_history()

        sample = (self.elapsed_s, t_to_hit, pos_err, vel_err, ori_err)
        if self._hit_candidate is None or abs(t_to_hit) < abs(self._hit_candidate[1]):
            self._hit_candidate = sample

        crossed_hit = self._prev_t_to_hit is not None and self._prev_t_to_hit > 0.0 >= t_to_hit
        expired_hit = t_to_hit < -self.strike_window
        if (crossed_hit or expired_hit) and not self._hit_reported:
            self._record_hit_sample()

        self._active_prev = True
        self._prev_t_to_hit = t_to_hit
        self._refresh_plot()

    def _reset_swing_tracking(self) -> None:
        self._active_prev = False
        self._prev_t_to_hit = None
        self._hit_candidate = None
        self._hit_reported = False

    def _trim_history(self) -> None:
        cutoff = self.elapsed_s - self.window_s
        while self.times and self.times[0] < cutoff:
            self.times.pop(0)
            self.pos_errors.pop(0)
            self.vel_errors.pop(0)
            self.ori_errors.pop(0)
            self.strike_flags.pop(0)
        while self.hit_samples and self.hit_samples[0][0] < cutoff:
            self.hit_samples.pop(0)

    def _record_hit_sample(self) -> None:
        if self._hit_candidate is None:
            return
        self._hit_reported = True
        self.hit_samples.append(self._hit_candidate)
        t_abs, t_to_hit, pos_err, vel_err, ori_err = self._hit_candidate
        print(
            "[PingpongHitError] "
            f"swing={self._swing_index} time={t_abs:.3f}s t_to_hit={t_to_hit:+.4f}s "
            f"pos={pos_err:.4f}m vel={vel_err:.4f}m/s ori={ori_err:.4f}"
        )

    def _refresh_plot(self) -> None:
        if not self._plot_enabled:
            return
        self._plot_tick += 1
        if self._plot_tick % 5 != 0:
            return
        if not self.times:
            return
        if not self._plt.fignum_exists(self.fig.number):
            self._plot_enabled = False
            return

        series = (self.pos_errors, self.vel_errors, self.ori_errors)
        latest_hit = self.hit_samples[-1] if self.hit_samples else None
        for index, (line, strike_line, values) in enumerate(zip(self.error_lines, self.strike_lines, series, strict=True)):
            strike_values = [value if flag else float("nan") for value, flag in zip(values, self.strike_flags, strict=True)]
            line.set_data(self.times, values)
            strike_line.set_data(self.times, strike_values)
            ax = self.axes[index]
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)

        x_min = max(0.0, self.elapsed_s - self.window_s)
        x_max = max(self.window_s, self.elapsed_s + 0.10)
        for ax, hit_line in zip(self.axes, self.hit_vlines, strict=True):
            ax.set_xlim(x_min, x_max)
            if latest_hit is not None and latest_hit[0] >= x_min:
                hit_line.set_xdata([latest_hit[0], latest_hit[0]])
                hit_line.set_visible(True)
            else:
                hit_line.set_visible(False)

        if latest_hit is not None:
            _, t_to_hit, pos_err, vel_err, ori_err = latest_hit
            self.axes[0].set_title(f"latest hit-frame pos error: {pos_err:.4f} m (t_to_hit={t_to_hit:+.4f}s)")
            self.axes[1].set_title(f"latest hit-frame velocity error: {vel_err:.4f} m/s")
            self.axes[2].set_title(f"latest hit-frame ori error: {ori_err:.4f}")

        self.fig.canvas.draw_idle()
        self._plt.pause(0.001)


def main():
    """Play with RSL-RL agent."""
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
            print(
                "[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task."
            )
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )
    if args_cli.task and "Pingpong" in args_cli.task:
        try:
            env.unwrapped.sim.set_camera_view([4.0, -2.8, 1.7], [1.35, 0.0, 0.85])
        except AttributeError:
            pass

    # convert to single-agent instance if required by the RL algorithm
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
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 兼容性问题，和train一样，需要在构造OnPolicyRunner前进行转换！
    installed_version = version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    agent_cfg_dict = agent_cfg.to_dict()
    runner_name = None
    if isinstance(agent_cfg_dict.get("runner"), dict):
        runner_name = agent_cfg_dict["runner"].get("runner_class_name") or agent_cfg_dict[
            "runner"
        ].get("class_name")
    if runner_name is None:
        runner_name = agent_cfg_dict.get("class_name")
    if runner_name is None:
        alg_name = (agent_cfg_dict.get("algorithm") or {}).get("class_name", "")
        leaf = alg_name.split(":")[-1].split(".")[-1]
        if leaf in ("AmpPPO", "AMPPPO"):
            runner_name = "OnPolicyAmpRunner"

    if runner_name in (None, "OnPolicyRunner"):
        runner = OnPolicyRunner(
            env, agent_cfg_dict, log_dir=None, device=agent_cfg.device
        )
    elif runner_name == "OnPolicyAmpRunner":
        from unitree_rl_lab.rsl_rl_amp import OnPolicyAmpRunner

        runner = OnPolicyAmpRunner(
            env, agent_cfg_dict, log_dir=None, device=agent_cfg.device
        )
    elif runner_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(
            env, agent_cfg_dict, log_dir=None, device=agent_cfg.device
        )
    else:
        raise ValueError(f"Unsupported runner class: {runner_name}")
    print(f"[INFO] Using rsl_rl runner: {type(runner).__name__}")

    checkpoint_summary = load_checkpoint_summary(resume_path)
    target_actor_sd = get_runner_actor_state_dict(runner)
    compatible, _ = print_actor_checkpoint_compat_report(
        checkpoint_summary,
        args_cli.task,
        target_actor_sd,
    )
    if not compatible:
        raise RuntimeError(
            "Checkpoint is not compatible with the current play task. "
            "Actor state_dict keys/shapes must match exactly."
        )

    play_load_cfg = {
        "actor": True,
        "critic": False,
        "optimizer": False,
        "iteration": True,
        "rnd": False,
        "amp": False,
        "amp_optimizer": False,
        "amp_normalizer": False,
    }
    print(
        "[INFO] Checkpoint load mode: play actor-only, "
        "critic=False, optimizer=False, iteration=True"
    )
    load_runner_checkpoint(runner, resume_path, load_cfg=play_load_cfg)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    # try:
    #     # version 2.3 onwards
    #     policy_nn = runner.alg.policy
    # except AttributeError:
    #     # version 2.2 and below
    #     policy_nn = runner.alg.actor_critic

    # # extract the normalizer
    # if hasattr(policy_nn, "actor_obs_normalizer"):
    #     normalizer = policy_nn.actor_obs_normalizer
    # elif hasattr(policy_nn, "student_obs_normalizer"):
    #     normalizer = policy_nn.student_obs_normalizer
    # else:
    #     normalizer = None

    # # export policy to onnx/jit
    # export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # export_policy_as_jit(
    #     policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt"
    # )
    # export_policy_as_onnx(
    #     policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx"
    # )

    dt = env.unwrapped.step_dt
    pingpong_error_monitor = PingpongPlayErrorMonitor(
        env.unwrapped,
        args_cli.task,
        dt,
        enabled=not args_cli.no_pingpong_error_plot,
        plot_enabled=not getattr(args_cli, "headless", False),
        window_s=args_cli.pingpong_error_window_s,
    )

    # reset environment
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
            pingpong_error_monitor.update()
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
