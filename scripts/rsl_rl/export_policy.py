# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export RSL-RL policy checkpoint to JIT / ONNX.

Usage:
    python export_policy.py \
      --task Unitree-G1-23dof-Velocity \
      --num_envs 1 \
      --device cuda:0 \
      --checkpoint /abs/path/to/model_40000.pt \
      --headless
"""

import argparse
import os
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Export RSL-RL policy to JIT / ONNX.")
parser.add_argument("--task", type=str, required=True, help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--outdir",
    type=str,
    default=None,
    help="Output directory. Default: <checkpoint_dir>/exported",
)
parser.add_argument(
    "--onnx_opset",
    type=int,
    default=17,
    help="ONNX opset version.",
)

# NOTE:
# cli_args.add_rsl_rl_args() already adds --checkpoint / --resume / --load_run ...
cli_args.add_rsl_rl_args(parser)

# NOTE:
# AppLauncher.add_app_launcher_args() already adds --device / --headless ...
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
import torch.nn as nn

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def resolve_checkpoint(agent_cfg, args_cli) -> str:
    """Resolve checkpoint path from CLI / run config."""
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if getattr(args_cli, "use_pretrained_checkpoint", False):
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            raise FileNotFoundError(
                "No published pretrained checkpoint is available for this task."
            )
        return resume_path

    if getattr(args_cli, "checkpoint", None):
        return retrieve_file_path(args_cli.checkpoint)

    return get_checkpoint_path(
        log_root_path,
        getattr(agent_cfg, "load_run", None),
        getattr(agent_cfg, "load_checkpoint", None),
    )


def extract_obs_dict(obs):
    """Normalize env observations to a plain python dict[str, Tensor]."""
    # env.get_observations() may return (obs, extras)
    if isinstance(obs, tuple):
        obs = obs[0]

    # TensorDict / dict-like
    if hasattr(obs, "keys"):
        out = {}
        for k in obs.keys():
            v = obs[k]
            if not isinstance(v, torch.Tensor):
                v = torch.as_tensor(v)
            if v.ndim == 1:
                v = v.unsqueeze(0)
            out[str(k)] = v
        return out

    # plain tensor
    if isinstance(obs, torch.Tensor):
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        return {"policy": obs}

    raise TypeError(f"Unsupported observation type: {type(obs)}")


def infer_actor_obs_key(obs_dict, runner) -> str:
    """Infer which observation key should feed the actor."""
    if "policy" in obs_dict:
        return "policy"
    if "actor" in obs_dict:
        return "actor"

    # try actor model metadata
    try:
        if hasattr(runner.alg, "actor") and hasattr(runner.alg.actor, "obs_groups"):
            obs_groups = runner.alg.actor.obs_groups
            if isinstance(obs_groups, (list, tuple)) and len(obs_groups) > 0:
                key = str(obs_groups[0])
                if key in obs_dict:
                    return key
    except Exception:
        pass

    raise KeyError(
        f"Cannot infer actor observation key. Available keys: {list(obs_dict.keys())}"
    )


class TensorInputPolicyWrapper(nn.Module):
    """Wrap policy so exported model input is a plain tensor.

    Input:
        obs_tensor: [B, obs_dim]
    Output:
        action_tensor: [B, act_dim]
    """

    def __init__(self, policy_callable, obs_key: str):
        super().__init__()
        self.policy_callable = policy_callable
        self.obs_key = obs_key

    def forward(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        obs_dict = {self.obs_key: obs_tensor}
        out = self.policy_callable(obs_dict)

        if not isinstance(out, torch.Tensor):
            raise TypeError(f"Policy output must be torch.Tensor, but got {type(out)}")
        return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # -------------------------------------------------------------------------
    # Parse configs
    # -------------------------------------------------------------------------
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

    installed_version = version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    agent_cfg_dict = agent_cfg.to_dict()

    resume_path = resolve_checkpoint(agent_cfg, args_cli)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    # -------------------------------------------------------------------------
    # Create env
    # -------------------------------------------------------------------------
    env = gym.make(args_cli.task, cfg=env_cfg)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # -------------------------------------------------------------------------
    # Build runner
    #
    # Resolution order mirrors scripts/rsl_rl/{train,play}.py:
    #   1. ``train_cfg["runner"]["runner_class_name"]`` / ``class_name``.
    #   2. Top-level ``class_name`` (set by BasePPORunnerCfg et al.).
    #   3. Fallback: inspect ``algorithm.class_name`` — if it leaf-resolves to
    #      ``AmpPPO``/``AMPPPO`` (possibly with a dotted ``module:Class`` path
    #      like ``unitree_rl_lab.rsl_rl_amp.algorithms.amp_ppo:AmpPPO``),
    #      pick :class:`OnPolicyAmpRunner`.
    # -------------------------------------------------------------------------
    runner_name = None
    if isinstance(agent_cfg_dict.get("runner"), dict):
        runner_name = agent_cfg_dict["runner"].get(
            "runner_class_name"
        ) or agent_cfg_dict["runner"].get("class_name")
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

    # AMP runner: only the actor (+ optional critic) is needed for export —
    # skip discriminator / AMP optimizer / AMP normalizer so the load path
    # does not fail on checkpoints trained with different curriculum state.
    if type(runner).__name__ == "OnPolicyAmpRunner":
        export_load_cfg = {
            "actor": True,
            "critic": True,
            "optimizer": False,
            "iteration": True,
            "rnd": False,
            "amp": False,
            "amp_optimizer": False,
            "amp_normalizer": False,
        }
        runner.load(resume_path, load_cfg=export_load_cfg)
    else:
        runner.load(resume_path)

    # -------------------------------------------------------------------------
    # Get observations and infer actor input key
    # -------------------------------------------------------------------------
    raw_obs = env.get_observations()
    obs_dict = extract_obs_dict(raw_obs)
    obs_key = infer_actor_obs_key(obs_dict, runner)
    actor_obs = obs_dict[obs_key].to(env.unwrapped.device)

    print(f"[INFO] Actor obs key: {obs_key}")
    print(f"[INFO] Actor obs shape: {tuple(actor_obs.shape)}")

    # -------------------------------------------------------------------------
    # Use inference policy, but wrap it to accept a plain Tensor input
    # -------------------------------------------------------------------------
    policy_callable = runner.get_inference_policy(device=env.unwrapped.device)
    export_module = TensorInputPolicyWrapper(policy_callable, obs_key).to(
        env.unwrapped.device
    )
    export_module.eval()

    # sanity check
    with torch.inference_mode():
        actions = export_module(actor_obs)

    print(f"[INFO] Action shape: {tuple(actions.shape)}")

    # -------------------------------------------------------------------------
    # Output directory
    # -------------------------------------------------------------------------
    if args_cli.outdir is None:
        export_dir = os.path.join(os.path.dirname(resume_path), "exported")
    else:
        export_dir = os.path.abspath(args_cli.outdir)

    os.makedirs(export_dir, exist_ok=True)

    ckpt_base = os.path.splitext(os.path.basename(resume_path))[0]
    # ckpt_base = "model_60000"
    jit_path = os.path.join(export_dir, f"{ckpt_base}_jit.pt")
    onnx_path = os.path.join(export_dir, f"{ckpt_base}.onnx")

    # jit_path = os.path.join(export_dir, "policy_jit.pt")
    # onnx_path = os.path.join(export_dir, "policy.onnx")

    # -------------------------------------------------------------------------
    # Export JIT
    # -------------------------------------------------------------------------
    traced = torch.jit.trace(export_module, actor_obs)
    traced.save(jit_path)
    print(f"[INFO] Saved JIT to: {jit_path}")

    # -------------------------------------------------------------------------
    # Export ONNX
    # -------------------------------------------------------------------------
    policy_callable_cpu = runner.get_inference_policy(device="cpu")
    export_module_cpu = TensorInputPolicyWrapper(policy_callable_cpu, obs_key).cpu()
    export_module_cpu.eval()

    actor_obs_cpu = actor_obs.detach().cpu()

    torch.onnx.export(
        export_module_cpu,
        actor_obs_cpu,
        onnx_path,
        export_params=True,
        opset_version=args_cli.onnx_opset,
        do_constant_folding=True,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={
            "obs": {0: "batch"},
            "actions": {0: "batch"},
        },
    )
    print(f"[INFO] Saved ONNX to: {onnx_path}")

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    env.close()
    simulation_app.close()
    print("[INFO] Export finished successfully.")


if __name__ == "__main__":
    main()
