from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is normally present with IsaacLab.
    yaml = None


@dataclass(frozen=True)
class ShapeComparison:
    ok: bool
    mismatches: list[str]


@dataclass
class CheckpointSummary:
    path: str
    checkpoint: dict[str, Any]
    actor_state_dict: Mapping[str, torch.Tensor]
    critic_state_dict: Mapping[str, torch.Tensor] | None
    iteration: int | None
    infos: Any
    task_name: str
    task_source: str
    experiment_name: str | None
    actor_obs_dim: int | None
    critic_obs_dim: int | None
    action_dim: int | None


def load_checkpoint_summary(path: str, map_location: str = "cpu") -> CheckpointSummary:
    """Load a checkpoint and extract the metadata needed for compatibility checks."""
    abs_path = os.path.abspath(os.path.expanduser(path))
    checkpoint = torch.load(abs_path, weights_only=False, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(checkpoint).__name__}: {abs_path}")
    if "actor_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint has no actor_state_dict: {abs_path}")

    actor_sd = checkpoint["actor_state_dict"]
    critic_sd = checkpoint.get("critic_state_dict")
    if not isinstance(actor_sd, Mapping):
        raise TypeError(f"actor_state_dict must be a mapping, got {type(actor_sd).__name__}")
    if critic_sd is not None and not isinstance(critic_sd, Mapping):
        raise TypeError(f"critic_state_dict must be a mapping, got {type(critic_sd).__name__}")

    task_name, task_source, experiment_name = infer_checkpoint_task(abs_path, checkpoint)
    return CheckpointSummary(
        path=abs_path,
        checkpoint=checkpoint,
        actor_state_dict=actor_sd,
        critic_state_dict=critic_sd,
        iteration=checkpoint.get("iter"),
        infos=checkpoint.get("infos"),
        task_name=task_name,
        task_source=task_source,
        experiment_name=experiment_name,
        actor_obs_dim=infer_actor_obs_dim(actor_sd),
        critic_obs_dim=infer_critic_obs_dim(critic_sd),
        action_dim=infer_actor_action_dim(actor_sd),
    )


def infer_checkpoint_task(path: str, checkpoint: Mapping[str, Any]) -> tuple[str, str, str | None]:
    """Infer the task that produced a checkpoint.

    New checkpoints store this in ``infos["task"]``. Older checkpoints in this
    repo usually have ``infos=None``, so we fall back to the sibling
    ``params/agent.yaml`` and then the ``logs/rsl_rl/<experiment_name>`` path.
    """
    infos = checkpoint.get("infos")
    if isinstance(infos, Mapping):
        task = infos.get("task")
        exp = infos.get("experiment_name")
        if task:
            return str(task), "infos.task", str(exp) if exp else _experiment_from_path(path)
        meta = infos.get("checkpoint_meta")
        if isinstance(meta, Mapping) and meta.get("task"):
            exp = meta.get("experiment_name") or exp
            return str(meta["task"]), "infos.checkpoint_meta.task", str(exp) if exp else _experiment_from_path(path)

    agent_yaml = os.path.join(os.path.dirname(path), "params", "agent.yaml")
    exp = _experiment_from_agent_yaml(agent_yaml)
    if exp:
        return _experiment_to_task_name(exp), "params/agent.yaml:experiment_name", exp

    exp = _experiment_from_path(path)
    if exp:
        return _experiment_to_task_name(exp), "path:experiment_name", exp

    return "unknown", "unknown", None


def compare_state_dict_shapes(
    checkpoint_sd: Mapping[str, Any],
    target_sd: Mapping[str, Any],
    prefix: str,
) -> ShapeComparison:
    """Compare state_dict key sets and tensor shapes."""
    mismatches: list[str] = []
    checkpoint_keys = set(checkpoint_sd.keys())
    target_keys = set(target_sd.keys())

    for key in sorted(target_keys - checkpoint_keys):
        mismatches.append(f"{prefix}.{key}: missing in checkpoint, target_shape={_shape(target_sd[key])}")
    for key in sorted(checkpoint_keys - target_keys):
        mismatches.append(f"{prefix}.{key}: unexpected in checkpoint, ckpt_shape={_shape(checkpoint_sd[key])}")
    for key in sorted(checkpoint_keys & target_keys):
        ckpt_shape = _shape(checkpoint_sd[key])
        target_shape = _shape(target_sd[key])
        if ckpt_shape != target_shape:
            mismatches.append(f"{prefix}.{key}: ckpt={ckpt_shape}, target={target_shape}")
    return ShapeComparison(ok=not mismatches, mismatches=mismatches)


def get_runner_policy_state_dicts(
    runner: Any,
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]:
    """Return actor and critic state_dicts from an RSL-RL runner."""
    alg = runner.alg
    actor = getattr(alg, "actor", None)
    critic = getattr(alg, "critic", None)
    if actor is None or critic is None:
        raise AttributeError(
            "Expected runner.alg.actor and runner.alg.critic for compatibility checks. "
            f"Available alg attrs: {sorted(a for a in dir(alg) if not a.startswith('_'))}"
        )
    return actor.state_dict(), critic.state_dict()


def get_runner_actor_state_dict(runner: Any) -> Mapping[str, torch.Tensor]:
    """Return only the actor state_dict from an RSL-RL runner."""
    alg = runner.alg
    actor = getattr(alg, "actor", None)
    if actor is None:
        raise AttributeError(
            "Expected runner.alg.actor for actor-only compatibility checks. "
            f"Available alg attrs: {sorted(a for a in dir(alg) if not a.startswith('_'))}"
        )
    return actor.state_dict()


def check_checkpoint_compatibility(
    summary: CheckpointSummary,
    target_actor_sd: Mapping[str, Any],
    target_critic_sd: Mapping[str, Any],
) -> tuple[ShapeComparison, ShapeComparison]:
    actor_cmp = compare_state_dict_shapes(summary.actor_state_dict, target_actor_sd, "actor")
    if summary.critic_state_dict is None:
        critic_cmp = ShapeComparison(False, ["critic_state_dict: missing in checkpoint"])
    else:
        critic_cmp = compare_state_dict_shapes(summary.critic_state_dict, target_critic_sd, "critic")
    return actor_cmp, critic_cmp


def print_actor_checkpoint_compat_report(
    summary: CheckpointSummary,
    current_task: str,
    target_actor_sd: Mapping[str, Any],
    *,
    max_mismatches: int = 20,
) -> tuple[bool, ShapeComparison]:
    """Print and return an actor-only compatibility report for play/deployment."""
    target_actor_obs_dim = infer_actor_obs_dim(target_actor_sd)
    target_action_dim = infer_actor_action_dim(target_actor_sd)
    actor_cmp = compare_state_dict_shapes(summary.actor_state_dict, target_actor_sd, "actor")
    ok = actor_cmp.ok

    print("\n============ Checkpoint Compatibility (Actor Only) ============")
    print(f"[INFO] Checkpoint path : {summary.path}")
    print(f"[INFO] Checkpoint task : {summary.task_name} (source={summary.task_source})")
    print(f"[INFO] Current task    : {current_task}")
    print(f"[INFO] Checkpoint iter : {summary.iteration}")
    print(
        "[INFO] Actor dims      : "
        f"ckpt_obs={summary.actor_obs_dim}, target_obs={target_actor_obs_dim}, "
        f"ckpt_action={summary.action_dim}, target_action={target_action_dim}"
    )
    print(
        "[INFO] Critic dims     : "
        f"ckpt_obs={summary.critic_obs_dim}, target_obs=ignored_for_play"
    )
    print(f"[INFO] Shape check     : {'PASS' if ok else 'FAIL'} (actor only)")
    if not ok:
        for msg in actor_cmp.mismatches[:max_mismatches]:
            print(f"[ERROR] {msg}")
        if len(actor_cmp.mismatches) > max_mismatches:
            print(f"[ERROR] ... {len(actor_cmp.mismatches) - max_mismatches} more actor mismatches")
    print("===============================================================\n")
    return ok, actor_cmp


def print_checkpoint_compat_report(
    summary: CheckpointSummary,
    current_task: str,
    target_actor_sd: Mapping[str, Any],
    target_critic_sd: Mapping[str, Any],
    *,
    max_mismatches: int = 20,
) -> tuple[bool, ShapeComparison, ShapeComparison]:
    target_actor_obs_dim = infer_actor_obs_dim(target_actor_sd)
    target_critic_obs_dim = infer_critic_obs_dim(target_critic_sd)
    target_action_dim = infer_actor_action_dim(target_actor_sd)
    actor_cmp, critic_cmp = check_checkpoint_compatibility(summary, target_actor_sd, target_critic_sd)
    ok = actor_cmp.ok and critic_cmp.ok

    print("\n================ Checkpoint Compatibility ================")
    print(f"[INFO] Checkpoint path : {summary.path}")
    print(f"[INFO] Checkpoint task : {summary.task_name} (source={summary.task_source})")
    print(f"[INFO] Current task    : {current_task}")
    print(f"[INFO] Checkpoint iter : {summary.iteration}")
    print(
        "[INFO] Actor dims      : "
        f"ckpt_obs={summary.actor_obs_dim}, target_obs={target_actor_obs_dim}, "
        f"ckpt_action={summary.action_dim}, target_action={target_action_dim}"
    )
    print(
        "[INFO] Critic dims     : "
        f"ckpt_obs={summary.critic_obs_dim}, target_obs={target_critic_obs_dim}"
    )
    print(f"[INFO] Shape check     : {'PASS' if ok else 'FAIL'}")
    if not ok:
        mismatches = actor_cmp.mismatches + critic_cmp.mismatches
        for msg in mismatches[:max_mismatches]:
            print(f"[ERROR] {msg}")
        if len(mismatches) > max_mismatches:
            print(f"[ERROR] ... {len(mismatches) - max_mismatches} more mismatches")
    print("==========================================================\n")
    return ok, actor_cmp, critic_cmp


def task_names_match(checkpoint_task: str, current_task: str) -> bool:
    if checkpoint_task == "unknown":
        return False
    return _normalize_task_name(checkpoint_task) == _normalize_task_name(current_task)


def build_checkpoint_metadata(task: str, experiment_name: str, runner: Any) -> dict[str, Any]:
    actor_sd, critic_sd = get_runner_policy_state_dicts(runner)
    return {
        "task": task,
        "experiment_name": experiment_name,
        "actor_obs_dim": infer_actor_obs_dim(actor_sd),
        "critic_obs_dim": infer_critic_obs_dim(critic_sd),
        "action_dim": infer_actor_action_dim(actor_sd),
        "runner_class": type(runner).__name__,
    }


def wrap_runner_save_with_metadata(runner: Any, metadata: Mapping[str, Any]) -> None:
    """Patch runner.save so future checkpoints carry task/shape metadata."""
    original_save = runner.save

    def save_with_metadata(path: str, infos: dict | None = None) -> None:
        merged: dict[str, Any] = {}
        if isinstance(infos, Mapping):
            merged.update(dict(infos))
        elif infos is not None:
            merged["source_infos"] = infos
        merged.update(dict(metadata))
        original_save(path, infos=merged)

    runner.save = save_with_metadata


def infer_actor_obs_dim(state_dict: Mapping[str, Any] | None) -> int | None:
    if state_dict is None:
        return None
    first_linear = _first_linear_weight(state_dict)
    return int(first_linear.shape[1]) if first_linear is not None else None


def infer_critic_obs_dim(state_dict: Mapping[str, Any] | None) -> int | None:
    return infer_actor_obs_dim(state_dict)


def infer_actor_action_dim(state_dict: Mapping[str, Any] | None) -> int | None:
    if state_dict is None:
        return None
    std = state_dict.get("distribution.std_param")
    if hasattr(std, "shape") and len(std.shape) == 1:
        return int(std.shape[0])
    last_linear = _last_linear_weight(state_dict)
    return int(last_linear.shape[0]) if last_linear is not None else None


def _first_linear_weight(state_dict: Mapping[str, Any]) -> torch.Tensor | None:
    for key, value in state_dict.items():
        if key.endswith(".weight") and hasattr(value, "shape") and len(value.shape) == 2:
            return value
    return None


def _last_linear_weight(state_dict: Mapping[str, Any]) -> torch.Tensor | None:
    last = None
    for key, value in state_dict.items():
        if key.endswith(".weight") and hasattr(value, "shape") and len(value.shape) == 2:
            last = value
    return last


def _shape(value: Any) -> tuple[int, ...] | str:
    if hasattr(value, "shape"):
        return tuple(int(x) for x in value.shape)
    return type(value).__name__


def _experiment_from_agent_yaml(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    if yaml is not None:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        exp = data.get("experiment_name") if isinstance(data, Mapping) else None
        return str(exp) if exp else None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("experiment_name:"):
                return line.split(":", 1)[1].strip().strip("'\"") or None
    return None


def _experiment_from_path(path: str) -> str | None:
    parts = os.path.abspath(path).split(os.sep)
    for index, part in enumerate(parts):
        if part == "rsl_rl" and index + 1 < len(parts):
            return parts[index + 1]
    return None


def _experiment_to_task_name(experiment_name: str) -> str:
    known = {
        "unitree_g1_23dof_pingpong_hitter": "Unitree-G1-23dof-Pingpong-HITTER",
        "unitree_g1_23dof_pingpong_hitter_real": "Unitree-G1-23dof-Pingpong-HITTER-REAL",
    }
    return known.get(experiment_name, experiment_name)


def _normalize_task_name(name: str) -> str:
    return name.lower().replace("-", "_").removesuffix("_play")
