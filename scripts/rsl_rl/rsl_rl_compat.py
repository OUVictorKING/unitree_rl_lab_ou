"""Compatibility helpers for Isaac Lab / RSL-RL API drift."""

from __future__ import annotations

from collections.abc import Mapping

import torch

try:
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg as _handle_deprecated_rsl_rl_cfg
except ImportError:
    _handle_deprecated_rsl_rl_cfg = None


def handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version: str):
    """Return an RSL-RL config compatible with the installed Isaac Lab stack.

    Isaac Lab 0.49 / isaaclab_rl 0.4.5 does not export
    ``handle_deprecated_rsl_rl_cfg`` and already uses the config layout expected
    by rsl-rl-lib 3.1.x.  In that stack, compatibility conversion is a no-op.
    """
    if _handle_deprecated_rsl_rl_cfg is None:
        return agent_cfg
    return _handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)


def load_runner_checkpoint(
    runner,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
):
    """Load checkpoints with optional actor/critic/optimizer selection.

    RSL-RL 3.1.x stock ``OnPolicyRunner.load`` only supports full checkpoint
    loading. This helper preserves that fast path and adds the selective
    ``load_cfg`` behavior used by this repo's train/play/export scripts.
    """
    if load_cfg is None:
        return runner.load(path, map_location=map_location)

    # AMP runner implements the same selective contract natively.
    if type(runner).__name__ == "OnPolicyAmpRunner":
        return runner.load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)

    loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
    lc = dict(load_cfg)

    load_actor = bool(lc.get("actor", True))
    load_critic = bool(lc.get("critic", True))
    load_opt = bool(lc.get("optimizer", True))
    load_iter = bool(lc.get("iteration", True))
    load_rnd = bool(lc.get("rnd", True))

    policy = runner.alg.policy
    if "model_state_dict" in loaded_dict and load_actor and load_critic:
        resumed_training = policy.load_state_dict(loaded_dict["model_state_dict"], strict=strict)
    elif "model_state_dict" in loaded_dict:
        _load_policy_subset(policy, loaded_dict["model_state_dict"], load_actor, load_critic, strict)
        resumed_training = True
    else:
        resumed_training = True
        actor_sd, critic_sd = _extract_actor_critic_state_dicts(loaded_dict)
        if load_actor and actor_sd is not None:
            policy.actor.load_state_dict(actor_sd, strict=strict)
        if load_critic and critic_sd is not None:
            policy.critic.load_state_dict(critic_sd, strict=strict)

    if load_rnd and getattr(runner.alg, "rnd", None) and "rnd_state_dict" in loaded_dict:
        runner.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=strict)

    if load_opt and resumed_training:
        if "optimizer_state_dict" in loaded_dict:
            runner.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if (
            load_rnd
            and getattr(runner.alg, "rnd", None)
            and "rnd_optimizer_state_dict" in loaded_dict
        ):
            runner.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])

    if load_iter and resumed_training and "iter" in loaded_dict:
        runner.current_learning_iteration = loaded_dict["iter"]

    return loaded_dict.get("infos")


def _extract_actor_critic_state_dicts(
    checkpoint: Mapping,
) -> tuple[Mapping | None, Mapping | None]:
    actor_sd = checkpoint.get("actor_state_dict")
    critic_sd = checkpoint.get("critic_state_dict")
    if actor_sd is not None or critic_sd is not None:
        return actor_sd, critic_sd

    model_sd = checkpoint.get("model_state_dict")
    if not isinstance(model_sd, Mapping):
        raise KeyError("Checkpoint has neither split actor/critic weights nor model_state_dict.")
    return _strip_state_prefix(model_sd, "actor."), _strip_state_prefix(model_sd, "critic.")


def _strip_state_prefix(state_dict: Mapping, prefix: str) -> dict:
    return {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def _load_policy_subset(
    policy,
    model_state_dict: Mapping,
    load_actor: bool,
    load_critic: bool,
    strict: bool,
) -> None:
    current = policy.state_dict()
    for key, value in model_state_dict.items():
        if key not in current:
            continue
        if load_actor and _is_actor_policy_key(key):
            current[key] = value
        elif load_critic and _is_critic_policy_key(key):
            current[key] = value
    policy.load_state_dict(current, strict=strict)


def _is_actor_policy_key(key: str) -> bool:
    return (
        key.startswith("actor.")
        or key.startswith("actor_obs_normalizer.")
        or key in {"std", "log_std"}
    )


def _is_critic_policy_key(key: str) -> bool:
    return key.startswith("critic.") or key.startswith("critic_obs_normalizer.")
