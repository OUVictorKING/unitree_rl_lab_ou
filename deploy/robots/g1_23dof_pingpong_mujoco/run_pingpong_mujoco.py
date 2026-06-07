#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
import types
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
SOURCE_ROOT = REPO_ROOT / "source" / "unitree_rl_lab"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


SDK_MOTOR_NAMES_27 = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "",
    "",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "",
    "",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
]
BLADE_NORMAL_LOCAL = np.array([0.0, -1.0, 0.0], dtype=np.float32)
GRAVITY = np.array([0.0, 0.0, -9.81], dtype=np.float32)


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def quat_normalize_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    n = np.linalg.norm(q)
    return q / n if n > 1.0e-9 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float32,
    )


def quat_rotate_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = quat_normalize_wxyz(q)
    vv = np.array([0.0, v[0], v[1], v[2]], dtype=np.float32)
    return quat_mul_wxyz(quat_mul_wxyz(q, vv), quat_conj_wxyz(q))[1:]


def quat_rotate_inverse_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_rotate_wxyz(quat_conj_wxyz(q), v)


def yaw_from_wxyz(q: np.ndarray) -> float:
    q = quat_normalize_wxyz(q)
    w, x, y, z = q
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def rotate_yaw_2d(vec_xy: np.ndarray, yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array(
        [c * vec_xy[0] - s * vec_xy[1], s * vec_xy[0] + c * vec_xy[1]], dtype=np.float32
    )


def solve_paddle_target_np(
    p_hit_world: np.ndarray,
    v_ball_in_world: np.ndarray,
    target_land_world: np.ndarray,
    flight_time: float,
    paddle_cor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = max(float(flight_time), 1.0e-3)
    v_out = (target_land_world - p_hit_world) / t + np.array(
        [0.0, 0.0, 0.5 * 9.81 * t], dtype=np.float32
    )
    delta_v = v_out - v_ball_in_world
    norm = float(np.linalg.norm(delta_v))
    if norm < 1.0e-9:
        n_target = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        v_racket = 2.0 * n_target
        return v_out.astype(np.float32), n_target, v_racket.astype(np.float32)
    n_target = (delta_v / norm).astype(np.float32)
    v_in_n = float(np.dot(v_ball_in_world, n_target))
    v_out_n = float(np.dot(v_out, n_target))
    v_pad_n = (v_out_n + float(paddle_cor) * v_in_n) / (1.0 + float(paddle_cor))
    return v_out.astype(np.float32), n_target, (v_pad_n * n_target).astype(np.float32)


class TrainingHitGeometry:
    """Deploy copy of the HITTER training task's npz-derived swing/base geometry."""

    def __init__(self, cfg: dict[str, Any]):
        forward_npz = self._required_npz_path(cfg, "forward_motion_file")
        backward_npz = self._required_npz_path(cfg, "backward_motion_file")
        self.forehand_offset_base = self._load_impact_offset(forward_npz)
        self.backhand_offset_base = self._load_impact_offset(backward_npz)
        self.expert_offset_base = np.stack(
            (self.forehand_offset_base, self.backhand_offset_base), axis=0
        ).astype(np.float32)

        forehand_y = float(self.forehand_offset_base[1])
        backhand_y = float(self.backhand_offset_base[1])
        clamp = cfg.get("forehand_y_safety_clamp", 0.40)
        if clamp is not None:
            cap = abs(float(clamp))
            forehand_y_eff = (
                max(forehand_y, -cap) if forehand_y < 0.0 else min(forehand_y, cap)
            )
        else:
            forehand_y_eff = forehand_y
        self.y_mid_base = 0.5 * (forehand_y_eff + backhand_y)
        self.swing_y_sign = 1.0 if forehand_y > backhand_y else -1.0
        self.hit_x_base = 0.5 * (
            float(self.forehand_offset_base[0]) + float(self.backhand_offset_base[0])
        )
        print(
            "[TrainingGeometry] "
            f"forehand_offset={self.forehand_offset_base.round(4).tolist()} "
            f"backhand_offset={self.backhand_offset_base.round(4).tolist()} "
            f"y_mid_base={self.y_mid_base:.4f} swing_y_sign={self.swing_y_sign:+.1f}"
        )

    @staticmethod
    def _required_npz_path(cfg: dict[str, Any], key: str) -> Path:
        value = cfg.get(key)
        if (
            value is None
            or str(value).strip() == ""
            or str(value).startswith("REPLACE_WITH_")
        ):
            raise ValueError(
                f"planner.{key} must be set to an expert .npz path in config.yaml "
                f"or passed via --{'forward' if 'forward' in key else 'backward'}-npz"
            )
        path = resolve_path(value)
        if not path.is_file():
            raise FileNotFoundError(f"planner.{key} points to missing npz: {path}")
        return path

    @staticmethod
    def _load_impact_offset(path: Path) -> np.ndarray:
        data = np.load(path, allow_pickle=True)
        body_names = [str(x) for x in data["body_names"].tolist()]
        pelvis_id = body_names.index("pelvis")
        blade_id = body_names.index("right_paddle_blade")
        impact_frame = int(data["impact_frame"][0])
        body_pos = np.asarray(data["body_pos_w"], dtype=np.float32)
        body_quat = np.asarray(data["body_quat_w"], dtype=np.float32)
        pelvis_pos = body_pos[impact_frame, pelvis_id]
        blade_pos = body_pos[impact_frame, blade_id]
        pelvis_quat = body_quat[impact_frame, pelvis_id]
        yaw = yaw_from_wxyz(pelvis_quat)
        diff = blade_pos[:2] - pelvis_pos[:2]
        return rotate_yaw_2d(diff, -yaw)

    def classify_and_base_target(
        self,
        p_hit_world: np.ndarray,
        root_pos_world: np.ndarray,
        root_quat_wxyz: np.ndarray,
    ) -> tuple[int, np.ndarray, float]:
        yaw = yaw_from_wxyz(root_quat_wxyz)
        diff = p_hit_world[:2] - root_pos_world[:2]
        hit_base = rotate_yaw_2d(diff, -yaw)
        hit_y_base = float(hit_base[1])
        is_forehand = (hit_y_base - self.y_mid_base) * self.swing_y_sign > 0.0
        swing_type = 0 if is_forehand else 1
        offset_world = rotate_yaw_2d(self.expert_offset_base[swing_type], yaw)
        p_base_xy_world = p_hit_world[:2] - offset_world
        return swing_type, p_base_xy_world.astype(np.float32), hit_y_base


def remap_full_or_policy_order(
    values: list[float], joint_ids_map: list[int]
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if len(arr) == len(joint_ids_map):
        return arr.copy()
    if len(arr) > max(joint_ids_map):
        return arr[np.asarray(joint_ids_map, dtype=np.int64)].copy()
    raise ValueError(
        f"Cannot remap value list of length {len(arr)} with joint_ids_map={joint_ids_map}"
    )


def load_deploy_yaml(path: str | Path) -> dict[str, Any]:
    with resolve_path(path).open("r") as f:
        return yaml.safe_load(f)


def sanitize_torch_import_path() -> None:
    """Avoid importing IsaacSim's pip_prebundle torch inside the MuJoCo deploy env."""
    bad_tokens = (
        "omni.isaac.ml_archive",
        "_isaac_sim/exts",
        "_isaac_sim/kit/python",
        "pip_prebundle/torch",
    )
    sys.path[:] = [
        p
        for p in sys.path
        if not any(token in p.replace("\\", "/") for token in bad_tokens)
    ]
    loaded = sys.modules.get("torch")
    loaded_file = str(getattr(loaded, "__file__", "")) if loaded is not None else ""
    if loaded is not None and any(
        token in loaded_file.replace("\\", "/") for token in bad_tokens
    ):
        for name in list(sys.modules):
            if name == "torch" or name.startswith("torch."):
                sys.modules.pop(name, None)


def load_training_planner_module():
    """Load planner_for_training.py without importing IsaacLab-only packages.

    Importing ``unitree_rl_lab.tasks.pingpong.mdp`` normally executes its
    ``__init__`` and pulls IsaacLab into the MuJoCo deploy environment. The
    training planner itself only needs torch + motion_loader, so we load it as
    a tiny private package to preserve its relative import while avoiding the
    IsaacLab dependency chain.
    """
    sanitize_torch_import_path()
    mdp_dir = SOURCE_ROOT / "unitree_rl_lab" / "tasks" / "pingpong" / "mdp"
    package_name = "pingpong_deploy_mdp"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [mdp_dir.as_posix()]
        sys.modules[package_name] = package

    for short_name in ("motion_loader", "planner_for_training"):
        module_name = f"{package_name}.{short_name}"
        if module_name in sys.modules:
            continue
        path = mdp_dir / f"{short_name}.py"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {short_name}.py from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[f"{package_name}.planner_for_training"]


@dataclass
class RobotState:
    base_pos: np.ndarray
    base_quat_wxyz: np.ndarray
    base_ang_vel_b: np.ndarray
    dof_pos: np.ndarray
    dof_vel: np.ndarray
    paddle_pos_world: np.ndarray
    paddle_quat_wxyz: np.ndarray
    paddle_lin_vel_world: np.ndarray   # paddle body COM linear velocity, world frame
    ball_pos_world: np.ndarray
    ball_vel_world: np.ndarray


@dataclass
class CommandState:
    p_hit_world: np.ndarray
    v_ball_in_world: np.ndarray
    v_ball_out_world: np.ndarray
    v_racket_hat_world: np.ndarray
    n_target_world: np.ndarray
    target_land_world: np.ndarray
    p_base_xy_world: np.ndarray
    t_to_hit: float
    swing_type: int
    planner_valid: bool
    plan_mode: str
    active: bool
    # Planner-predicted ball trajectory leading up to the hit (world frame).
    # Each row is one substep position; planner_traj_valid masks out post-hit
    # / invalid steps. Used for the debug-viz cyan-dot trail and to sanity
    # check whether the planner thinks the ball will reach p_hit_world.
    planner_traj_world: np.ndarray | None = None    # shape (N, 3) or None
    planner_traj_valid: np.ndarray | None = None    # shape (N,) bool or None


class RslRlActorPolicy:
    def __init__(self, cfg: dict[str, Any]):
        self.name = str(cfg["name"])
        self.deploy = load_deploy_yaml(cfg["deploy_yaml"])
        self.joint_ids_map = [int(x) for x in self.deploy["joint_ids_map"]]
        self.joint_names = [SDK_MOTOR_NAMES_27[i] for i in self.joint_ids_map]
        if any(not name for name in self.joint_names):
            raise ValueError(
                f"{self.name}: joint_ids_map contains unsupported empty SDK slots"
            )

        action_cfg = self.deploy["actions"]["JointPositionAction"]
        self.default_pos = np.asarray(action_cfg["offset"], dtype=np.float32)
        self.action_scales = np.asarray(action_cfg["scale"], dtype=np.float32)
        action_clip = action_cfg.get("clip")
        self.processed_action_clip = (
            np.asarray(action_clip, dtype=np.float32)
            if action_clip is not None
            else None
        )
        self.stiffness = remap_full_or_policy_order(
            self.deploy["stiffness"], self.joint_ids_map
        )
        self.damping = remap_full_or_policy_order(
            self.deploy["damping"], self.joint_ids_map
        )
        self.num_actions = len(self.default_pos)
        # Optional extra safety clamp on the actor output. The deploy.yaml clip is
        # still applied later to the processed joint-position target, matching
        # IsaacLab/C++ deploy semantics.
        self.raw_action_clip = cfg.get("action_clip", None)
        self.action_beta = float(cfg.get("action_beta", 1.0))
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)

        self.obs_terms = self.deploy["observations"]
        self.obs_history_length = max(
            int(term.get("history_length", 1) or 1) for term in self.obs_terms.values()
        )
        self.history_buf: deque[dict[str, np.ndarray]] = deque(
            maxlen=self.obs_history_length
        )
        self._warned_unknown_terms: set[str] = set()

        self.kind = str(cfg.get("type", "rsl_rl_actor"))
        self.actor = self._load_actor(
            resolve_path(cfg["checkpoint"]), str(cfg.get("activation", "elu"))
        )
        print(
            f"[Policy] {self.name}: loaded {self.kind}, obs_dim={self.obs_dim}, actions={self.num_actions}"
        )

    @property
    def obs_dim(self) -> int:
        return int(
            sum(
                len(scale) * int(term.get("history_length", 1) or 1)
                for term in self.obs_terms.values()
                for scale in [term.get("scale") or []]
            )
        )

    def print_audit(self) -> None:
        print(
            f"[PolicyAudit] policy={self.name} obs_dim={self.obs_dim} actions={self.num_actions}"
        )
        start = 0
        for name, term in self.obs_terms.items():
            scale = term.get("scale") or []
            clip = term.get("clip")
            h = int(term.get("history_length", 1) or 1)
            dim = len(scale) * h
            scale_desc = (
                "none"
                if not scale
                else (
                    "1.0"
                    if all(float(x) == 1.0 for x in scale)
                    else np.asarray(scale).round(4).tolist()
                )
            )
            print(
                f"  obs[{start:03d}:{start + dim:03d}] {name:<18s} "
                f"dim={dim:2d} hist={h} scale={scale_desc} clip={clip}"
            )
            start += dim
        print(
            "  action processed = offset + raw * scale, then deploy.yaml processed clip"
        )
        for i, (name, sdk_id) in enumerate(zip(self.joint_names, self.joint_ids_map)):
            clip = (
                None
                if self.processed_action_clip is None
                else self.processed_action_clip[i].round(4).tolist()
            )
            print(
                f"  act[{i:02d}] sdk[{sdk_id:02d}] {name:<28s} "
                f"offset={self.default_pos[i]:+.4f} scale={self.action_scales[i]:.4f} "
                f"kp={self.stiffness[i]:.3f} kd={self.damping[i]:.3f} clip={clip}"
            )

    def reset(self) -> None:
        self.last_action[:] = 0.0
        self.history_buf.clear()

    def _load_actor(self, path: Path, activation: str):
        sanitize_torch_import_path()
        if path.suffix == ".onnx":
            import onnxruntime as ort

            session = ort.InferenceSession(
                path.as_posix(), providers=["CPUExecutionProvider"]
            )
            self._actor_mode = "onnx"
            self._onnx_input_name = session.get_inputs()[0].name
            self._onnx_output_name = session.get_outputs()[0].name
            return session

        import torch
        import torch.nn as nn

        if path.name.endswith("jit.pt") or path.name.endswith("_jit.pt"):
            self._actor_mode = "jit"
            return torch.jit.load(path.as_posix(), map_location="cpu")

        checkpoint = torch.load(path.as_posix(), map_location="cpu")
        state = (
            checkpoint.get("actor_state_dict", checkpoint)
            if isinstance(checkpoint, dict)
            else checkpoint
        )
        state = {k: v for k, v in state.items() if k.startswith("mlp.")}
        in_dim = int(state["mlp.0.weight"].shape[1])
        out_dim = int(state["mlp.6.weight"].shape[0])
        widths = [
            int(state["mlp.0.weight"].shape[0]),
            int(state["mlp.2.weight"].shape[0]),
            int(state["mlp.4.weight"].shape[0]),
        ]
        if in_dim != self.obs_dim:
            raise ValueError(
                f"{self.name}: actor obs dim {in_dim} != deploy obs dim {self.obs_dim}"
            )
        if out_dim != self.num_actions:
            raise ValueError(
                f"{self.name}: actor action dim {out_dim} != deploy action dim {self.num_actions}"
            )
        act_cls = nn.ELU if activation.lower() == "elu" else nn.ReLU

        class Actor(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = nn.Sequential(
                    nn.Linear(in_dim, widths[0]),
                    act_cls(),
                    nn.Linear(widths[0], widths[1]),
                    act_cls(),
                    nn.Linear(widths[1], widths[2]),
                    act_cls(),
                    nn.Linear(widths[2], out_dim),
                )

            def forward(self, obs):
                return self.mlp(obs)

        model = Actor()
        model.load_state_dict(state, strict=True)
        model.eval()
        self._actor_mode = "torch"
        return model

    def _scale_term(self, name: str, value: np.ndarray) -> np.ndarray:
        scale = self.obs_terms[name].get("scale")
        clip = self.obs_terms[name].get("clip")
        out = np.asarray(value, dtype=np.float32)
        if clip is not None:
            clip_arr = np.asarray(clip, dtype=np.float32)
            if clip_arr.ndim == 1 and clip_arr.shape[0] == 2:
                out = np.clip(out, clip_arr[0], clip_arr[1])
            elif clip_arr.shape == (out.shape[0], 2):
                out = np.clip(out, clip_arr[:, 0], clip_arr[:, 1])
            else:
                raise ValueError(
                    f"Unsupported clip shape for obs term '{name}': {clip_arr.shape}"
                )
        if scale is not None:
            out = out * np.asarray(scale, dtype=np.float32)
        return out.astype(np.float32)

    def _empty_term(self, name: str) -> np.ndarray:
        scale = self.obs_terms[name].get("scale")
        return np.zeros(len(scale) if scale is not None else 0, dtype=np.float32)

    def _push_history(self, terms_current: dict[str, np.ndarray]) -> None:
        if len(self.history_buf) == 0:
            for _ in range(self.obs_history_length):
                self.history_buf.append(terms_current)
        else:
            self.history_buf.append(terms_current)

    def _assemble_obs(self) -> np.ndarray:
        chunks: list[np.ndarray] = []
        h_max = self.obs_history_length
        for name, term in self.obs_terms.items():
            h = int(term.get("history_length", 1) or 1)
            for k in range(h):
                snap = self.history_buf[h_max - h + k]
                chunks.append(snap.get(name, self._empty_term(name)))
        return np.concatenate(chunks, axis=0).astype(np.float32)

    def _build_terms(
        self, state: RobotState, cmd: CommandState
    ) -> dict[str, np.ndarray]:
        root_q = state.base_quat_wxyz
        yaw = yaw_from_wxyz(root_q)
        gravity_b = quat_rotate_inverse_wxyz(
            root_q, np.array([0.0, 0.0, -1.0], dtype=np.float32)
        )
        hit_pos_b = quat_rotate_inverse_wxyz(root_q, cmd.p_hit_world - state.base_pos)
        base_err = cmd.p_base_xy_world - state.base_pos[:2]
        n_blade_w = quat_rotate_wxyz(state.paddle_quat_wxyz, BLADE_NORMAL_LOCAL)
        sign = 1.0 - 2.0 * float(cmd.swing_type)
        active_face_b = quat_rotate_inverse_wxyz(root_q, sign * n_blade_w)
        target_normal_b = quat_rotate_inverse_wxyz(root_q, cmd.n_target_world)

        values = {
            "base_ang_vel": state.base_ang_vel_b,
            "projected_gravity": gravity_b,
            "base_yaw": np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32),
            "base_err": base_err,
            "hit_pos": hit_pos_b,
            "racket_vel": cmd.v_racket_hat_world,
            "t_to_hit": np.array([cmd.t_to_hit], dtype=np.float32),
            "active_face": active_face_b,
            "target_normal": target_normal_b,
            "joint_pos": state.dof_pos - self.default_pos,
            "joint_vel": state.dof_vel,
            "last_action": self.last_action,
        }

        terms: dict[str, np.ndarray] = {}
        for name in self.obs_terms:
            if name in values:
                terms[name] = self._scale_term(name, values[name])
            elif name not in self._warned_unknown_terms:
                self._warned_unknown_terms.add(name)
                print(
                    f"[Policy][WARN] obs term '{name}' is not implemented; filling zeros"
                )
                terms[name] = self._empty_term(name)
        return terms

    def _infer(self, obs: np.ndarray) -> np.ndarray:
        if self._actor_mode == "onnx":
            out = self.actor.run(
                [self._onnx_output_name],
                {self._onnx_input_name: obs[None].astype(np.float32)},
            )[0]
            return np.asarray(out).squeeze().astype(np.float32)
        import torch

        with torch.no_grad():
            out = (
                self.actor(torch.from_numpy(obs).float().unsqueeze(0))
                .cpu()
                .numpy()
                .squeeze()
            )
        return np.asarray(out, dtype=np.float32)

    def act(self, state: RobotState, cmd: CommandState) -> np.ndarray:
        terms = self._build_terms(state, cmd)
        self._push_history(terms)
        obs = self._assemble_obs()
        if obs.shape[0] != self.obs_dim:
            raise RuntimeError(
                f"{self.name}: obs shape {obs.shape[0]} != expected {self.obs_dim}"
            )
        raw = self._infer(obs)
        raw = (1.0 - self.action_beta) * self.last_action + self.action_beta * raw
        self.last_action = raw.copy()
        if self.raw_action_clip is not None:
            raw = np.clip(
                raw, -float(self.raw_action_clip), float(self.raw_action_clip)
            )
        processed = self.default_pos + raw * self.action_scales
        if self.processed_action_clip is not None:
            processed = np.clip(
                processed,
                self.processed_action_clip[:, 0],
                self.processed_action_clip[:, 1],
            )
        return processed.astype(np.float32)


class FixedPosePolicy:
    def __init__(self, cfg: dict[str, Any]):
        self.name = str(cfg["name"])
        self.deploy = load_deploy_yaml(cfg["deploy_yaml"])
        self.joint_ids_map = [int(x) for x in self.deploy["joint_ids_map"]]
        self.joint_names = [SDK_MOTOR_NAMES_27[i] for i in self.joint_ids_map]
        action_cfg = self.deploy["actions"]["JointPositionAction"]
        self.default_pos = np.asarray(action_cfg["offset"], dtype=np.float32)
        self.stiffness = remap_full_or_policy_order(
            self.deploy["stiffness"], self.joint_ids_map
        )
        self.damping = remap_full_or_policy_order(
            self.deploy["damping"], self.joint_ids_map
        )
        self.num_actions = len(self.default_pos)
        print(f"[Policy] {self.name}: fixed pose, actions={self.num_actions}")

    def reset(self) -> None:
        pass

    def act(self, state: RobotState, cmd: CommandState) -> np.ndarray:
        return self.default_pos.copy()


class PolicyManager:
    def __init__(self, cfg: dict[str, Any]):
        self.policies = []
        for item in cfg["policies"]:
            if item.get("type") == "fixed_pose":
                self.policies.append(FixedPosePolicy(item))
            else:
                self.policies.append(RslRlActorPolicy(item))
        names = [p.name for p in self.policies]
        default_name = cfg.get("default_policy", names[0])
        self.current = names.index(default_name) if default_name in names else 0
        self._validate_compatible()
        print(f"[PolicyManager] active policy: {self.active.name}")

    @property
    def active(self):
        return self.policies[self.current]

    def _validate_compatible(self) -> None:
        ref_names = self.policies[0].joint_names
        for policy in self.policies[1:]:
            if policy.joint_names != ref_names:
                raise ValueError(
                    f"Policy '{policy.name}' joint order does not match '{self.policies[0].name}'"
                )

    def print_audit(self) -> None:
        for policy in self.policies:
            if hasattr(policy, "print_audit"):
                policy.print_audit()

    def reset(self) -> None:
        for policy in self.policies:
            policy.reset()

    def switch_to(self, name_or_idx: str | int) -> None:
        if isinstance(name_or_idx, int):
            self.current = name_or_idx % len(self.policies)
        else:
            names = [p.name for p in self.policies]
            if name_or_idx not in names:
                raise KeyError(f"Unknown policy '{name_or_idx}', choices={names}")
            self.current = names.index(name_or_idx)
        self.active.reset()
        print(f"[PolicyManager] switched to: {self.active.name}")

    def next(self) -> None:
        self.switch_to(self.current + 1)

    def prev(self) -> None:
        self.switch_to(self.current - 1)

    def act(self, state: RobotState, cmd: CommandState) -> np.ndarray:
        return self.active.act(state, cmd)


class PingpongCommandManager:
    def __init__(self, cfg: dict[str, Any]):
        pcfg = cfg["planner"]
        self.cfg = cfg
        self.geometry = TrainingHitGeometry(pcfg)
        planner_module = load_training_planner_module()
        self._plan_pingpong_hits = planner_module.plan_pingpong_hits
        self._torch = sys.modules["torch"]
        self._plan_mode_names = {
            int(planner_module.PLAN_INVALID): "invalid",
            int(planner_module.PLAN_FRESH): "fresh",
            int(planner_module.PLAN_HELD): "held",
            int(planner_module.PLAN_FROZEN): "frozen",
        }
        self._expert_offset_base_t = self._torch.as_tensor(
            self.geometry.expert_offset_base, dtype=self._torch.float32
        )
        x_hit_default = pcfg.get("x_hit_default")
        if x_hit_default is None:
            x_hit_default = float(self.geometry.hit_x_base)
            pcfg["x_hit_default"] = x_hit_default
            print(
                f"[TrainingGeometry] x_hit_default derived from npz: {x_hit_default:.4f}"
            )
        if cfg.get("serve", {}).get("hit_x") is None:
            cfg["serve"]["hit_x"] = float(x_hit_default)
        self.freeze_time = float(pcfg["freeze_time_before_hit"])
        self.post_swing_time = float(pcfg["post_swing_time"])
        self.post_hit_imitation = bool(pcfg.get("post_hit_imitation", True))
        self.fresh_min_t = float(pcfg["fresh_min_t_to_hit"])
        self.t_hit_abs: float | None = None
        self.frozen = False
        self.active = False
        self.last_cmd = self._fallback_cmd()
        print(
            f"[CommandTiming] post_hit_imitation={self.post_hit_imitation} "
            f"post_swing_time={self.post_swing_time:.2f}s"
        )

    def reset(self) -> None:
        self.t_hit_abs = None
        self.frozen = False
        self.active = False
        self.last_cmd = self._fallback_cmd()

    def _fallback_cmd(self) -> CommandState:
        pcfg = self.cfg["planner"]
        p_hit = np.array([float(pcfg["x_hit_default"]), 0.0, 1.05], dtype=np.float32)
        v_in = np.array([-3.0, 0.0, -0.5], dtype=np.float32)
        target = np.asarray(pcfg["target_land_world"], dtype=np.float32)
        v_out, n_target, v_racket = solve_paddle_target_np(
            p_hit,
            v_in,
            target,
            float(pcfg["flight_time"]),
            float(pcfg["paddle_cor"]),
        )
        swing_type, p_base_xy_world, _ = self.geometry.classify_and_base_target(
            p_hit,
            np.asarray(self.cfg["world"]["reset_root_pos"], dtype=np.float32),
            np.asarray(self.cfg["world"]["reset_root_quat_wxyz"], dtype=np.float32),
        )
        return CommandState(
            p_hit_world=p_hit,
            v_ball_in_world=v_in,
            v_ball_out_world=v_out,
            v_racket_hat_world=v_racket,
            n_target_world=n_target,
            target_land_world=target,
            p_base_xy_world=p_base_xy_world,
            # Idle t_to_hit: training keeps this strongly negative between swings
            # (RealPingpongCommand decrements every step, never clamps). Use a
            # large negative sentinel so the actor sees "no incoming ball" — the
            # same out-of-window state it learned to ignore. Anything in (0, 1.0s]
            # would falsely tell the policy "swing is imminent" → premature flail.
            t_to_hit=-2.0,
            swing_type=swing_type,
            planner_valid=False,
            plan_mode="fallback",
            active=False,
        )

    @staticmethod
    def _torch_row_to_np(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value[0].detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32).reshape(-1)

    def _plan_once(self, state: RobotState):
        torch = self._torch
        pcfg = self.cfg["planner"]
        wcfg = self.cfg["world"]
        table_size = np.asarray(wcfg["table_size"], dtype=np.float32)
        table_center = np.asarray(wcfg["table_center"], dtype=np.float32)
        ball_pos = torch.as_tensor(state.ball_pos_world, dtype=torch.float32).view(1, 3)
        ball_vel = torch.as_tensor(state.ball_vel_world, dtype=torch.float32).view(1, 3)
        root_pos = torch.as_tensor(state.base_pos, dtype=torch.float32).view(1, 3)
        root_quat = torch.as_tensor(state.base_quat_wxyz, dtype=torch.float32).view(
            1, 4
        )
        target_land = torch.as_tensor(
            pcfg["target_land_world"], dtype=torch.float32
        ).view(1, 3)
        return self._plan_pingpong_hits(
            ball_pos,
            ball_vel,
            root_pos,
            root_quat,
            target_land,
            table_top_z=float(wcfg["table_top_z"]),
            ball_radius=float(wcfg["ball_radius"]),
            valid_mask=torch.ones(1, dtype=torch.bool),
            x_hit_world=float(pcfg["x_hit_default"]),
            table_center_x_world=float(table_center[0]),
            table_center_y_world=float(table_center[1]),
            table_half_x=float(0.5 * table_size[0]),
            table_half_y=float(0.5 * table_size[1]),
            expert_offset_base=self._expert_offset_base_t,
            y_mid_base=float(self.geometry.y_mid_base),
            flight_time=float(pcfg["flight_time"]),
            paddle_cor=float(pcfg["paddle_cor"]),
            dt=float(pcfg.get("planner_dt", 0.01)),
            max_time=float(pcfg.get("planner_max_time", 1.50)),
            drag_k=float(pcfg.get("planner_drag_k", 0.10257265376884504)),
            bounce_ch=float(pcfg.get("planner_bounce_ch", 0.727005044772834)),
            bounce_cv=float(pcfg.get("planner_bounce_cv", 0.9018357357260598)),
            min_t_to_hit=float(pcfg.get("planner_min_t_to_hit", self.fresh_min_t)),
            max_t_to_hit=float(pcfg.get("planner_max_t_to_hit", 1.20)),
            hit_z_range=(float(pcfg["z_min_world"]), float(pcfg["z_max_world"])),
        )

    def _plan_to_cmd(self, plan: Any, t_to_hit: float, active: bool) -> CommandState:
        plan_mode_code = int(plan.plan_mode[0].detach().cpu().item())
        plan_mode = self._plan_mode_names.get(plan_mode_code, f"mode_{plan_mode_code}")
        # Planner-predicted ball trajectory (world frame). traj_p shape is
        # (n, max_steps+1, 3) with traj_valid (n, max_steps+1) bool. We grab
        # row 0 (single-env deploy) and copy to numpy for the visualizer.
        traj_world = None
        traj_valid = None
        try:
            tp = plan.traj_p[0].detach().cpu().numpy().astype(np.float32)
            tv = plan.traj_valid[0].detach().cpu().numpy().astype(bool)
            traj_world = tp.reshape(-1, 3)
            traj_valid = tv.reshape(-1)
        except Exception:  # noqa: BLE001
            pass
        return CommandState(
            p_hit_world=self._torch_row_to_np(plan.p_hit_world)[:3],
            v_ball_in_world=self._torch_row_to_np(plan.v_ball_in_world)[:3],
            v_ball_out_world=self._torch_row_to_np(plan.v_ball_out_world)[:3],
            v_racket_hat_world=self._torch_row_to_np(plan.v_racket_hat_world)[:3],
            n_target_world=self._torch_row_to_np(plan.n_target_world)[:3],
            target_land_world=self._torch_row_to_np(plan.target_land_world)[:3],
            p_base_xy_world=self._torch_row_to_np(plan.p_base_xy_world)[:2],
            t_to_hit=float(t_to_hit),
            swing_type=int(plan.swing_type[0].detach().cpu().item()),
            planner_valid=bool(plan.planner_valid[0].detach().cpu().item()),
            plan_mode=f"{plan_mode}+training_geom",
            active=active,
            planner_traj_world=traj_world,
            planner_traj_valid=traj_valid,
        )

    def _policy_t_to_hit(self, raw_t_to_hit: float) -> float:
        if self.post_hit_imitation:
            return max(float(raw_t_to_hit), -self.post_swing_time)
        if raw_t_to_hit <= 0.0:
            return -self.post_swing_time
        return float(raw_t_to_hit)

    def update(self, t_now: float, state: RobotState) -> tuple[CommandState, bool]:
        # HITTER-REAL-style timing with no planner spatial freeze:
        #   1) first valid planner sample starts an active swing and latches
        #      t_hit_abs = now + planner_t_to_hit;
        #   2) t_to_hit then decreases from that local clock every control step;
        #   3) planner-valid frames still refresh the spatial target every step;
        #   4) post-hit timing is configurable:
        #      post_hit_imitation=true  -> decrease through follow-through to -post_swing_time;
        #      post_hit_imitation=false -> jump to -post_swing_time at impact.
        plan = self._plan_once(state)
        valid = (
            plan is not None
            and bool(plan.planner_valid[0].detach().cpu().item())
        )
        fresh_t = (
            float(plan.t_to_hit[0].detach().cpu().item()) if plan is not None else 0.0
        )
        if valid and fresh_t > self.fresh_min_t:
            if not self.active or self.t_hit_abs is None:
                self.t_hit_abs = float(t_now) + fresh_t
            self.active = True
            local_t = self._policy_t_to_hit(float(self.t_hit_abs) - float(t_now))
            self.last_cmd = self._plan_to_cmd(plan, local_t, active=True)
        # If planner became invalid mid-swing (e.g. ball already past hit
        # plane, or hit-plane prediction failed), hold the last spatial cmd but
        # keep the HITTER-REAL timing clock moving through follow-through.
        elif self.active and self.t_hit_abs is not None:
            self.last_cmd.t_to_hit = self._policy_t_to_hit(float(self.t_hit_abs) - float(t_now))
            self.last_cmd.active = True
        return self.last_cmd, False


class ServeMachine:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg["serve"]

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        c = self.cfg
        pos = np.array(
            [
                np.random.uniform(*c["pos_x_range"]),
                np.random.uniform(*c["pos_y_range"]),
                np.random.uniform(*c["pos_z_range"]),
            ],
            dtype=np.float32,
        )
        hit = np.array(
            [
                float(c["hit_x"]),
                np.random.uniform(*c["hit_y_range"]),
                np.random.uniform(*c["hit_z_range"]),
            ],
            dtype=np.float32,
        )
        t_hit = float(np.random.uniform(*c["t_to_hit_range"]))
        vel = (hit - pos - 0.5 * GRAVITY * t_hit * t_hit) / t_hit
        speed = float(np.linalg.norm(vel))
        if speed > float(c["max_speed"]):
            vel = vel / max(speed, 1.0e-6) * float(c["max_speed"])
        return pos, vel.astype(np.float32)


class VirtualBall:
    def __init__(self):
        self.pos = np.array([2.75, 0.0, 1.1], dtype=np.float32)
        self.vel = np.zeros(3, dtype=np.float32)

    def reset(self, pos: np.ndarray, vel: np.ndarray) -> None:
        self.pos = np.asarray(pos, dtype=np.float32).copy()
        self.vel = np.asarray(vel, dtype=np.float32).copy()

    def step(self, dt: float) -> None:
        self.pos = self.pos + self.vel * dt + 0.5 * GRAVITY * dt * dt
        self.vel = self.vel + GRAVITY * dt


class MujocoPingpongEnv:
    def __init__(
        self, cfg: dict[str, Any], joint_names: list[str], render: bool = True
    ):
        import mujoco

        self.mujoco = mujoco
        self.cfg = cfg
        rcfg = cfg["robot"]
        self.model = mujoco.MjModel.from_xml_path(resolve_path(rcfg["xml"]).as_posix())
        self.model.opt.timestep = float(rcfg["sim_dt"])
        self.data = mujoco.MjData(self.model)
        self.sim_decimation = int(rcfg["sim_decimation"])
        self.control_dt = float(rcfg["control_dt"])
        self.joint_names = joint_names
        self.joint_qposadr = []
        self.joint_dofadr = []
        self.actuator_ids = []
        self.torque_limits = []
        for name in joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise KeyError(f"MuJoCo joint not found: {name}")
            self.joint_qposadr.append(int(self.model.jnt_qposadr[jid]))
            self.joint_dofadr.append(int(self.model.jnt_dofadr[jid]))
            aid = self._actuator_for_joint(jid, name)
            self.actuator_ids.append(aid)
            limit = np.asarray(self.model.jnt_actfrcrange[jid], dtype=np.float32)
            if limit.shape[0] == 2 and limit[1] > limit[0]:
                self.torque_limits.append(
                    max(abs(float(limit[0])), abs(float(limit[1])))
                )
            else:
                self.torque_limits.append(200.0)
        self.joint_qposadr = np.asarray(self.joint_qposadr, dtype=np.int32)
        self.joint_dofadr = np.asarray(self.joint_dofadr, dtype=np.int32)
        self.actuator_ids = np.asarray(self.actuator_ids, dtype=np.int32)
        self.torque_limits = np.asarray(self.torque_limits, dtype=np.float32)
        self.root_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint"
        )
        self.root_qadr = int(self.model.jnt_qposadr[self.root_joint])
        self.root_dadr = int(self.model.jnt_dofadr[self.root_joint])
        self.paddle_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, rcfg["paddle_body"]
        )
        self.ball_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, rcfg["ball_free_joint"]
        )
        self.ball_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, rcfg["ball_body"]
        )
        self.ball_qadr = int(self.model.jnt_qposadr[self.ball_joint])
        self.ball_dadr = int(self.model.jnt_dofadr[self.ball_joint])
        self.pelvis_gyro_sensor = self._sensor_slice("imu-pelvis-angular-velocity")
        self.viewer = None
        if render:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data, key_callback=self._key_callback
            )
            root_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, rcfg["root_body"]
            )
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewer.cam.trackbodyid = root_body_id
            self.viewer.cam.distance = 3.0
            self.viewer.cam.azimuth = 90.0
            self.viewer.cam.elevation = -20.0
        self.pending_keys: list[str] = []

    def _actuator_for_joint(self, joint_id: int, joint_name: str) -> int:
        for aid in range(self.model.nu):
            if int(self.model.actuator_trnid[aid, 0]) == joint_id:
                return aid
        aid = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name
        )
        if aid < 0:
            raise KeyError(f"MuJoCo actuator not found for joint: {joint_name}")
        return aid

    def _sensor_slice(self, sensor_name: str) -> slice | None:
        sid = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_SENSOR, sensor_name
        )
        if sid < 0:
            return None
        start = int(self.model.sensor_adr[sid])
        dim = int(self.model.sensor_dim[sid])
        return slice(start, start + dim)

    def _key_callback(self, key: int) -> None:
        glfw = self.mujoco.glfw.glfw
        mapping = {
            glfw.KEY_R: "reset",
            glfw.KEY_N: "next_policy",
            glfw.KEY_P: "prev_policy",
            glfw.KEY_H: "policy:hitter",
            glfw.KEY_F: "policy:stand",
            glfw.KEY_SPACE: "serve",
        }
        if key in mapping:
            self.pending_keys.append(mapping[key])

    def pop_keys(self) -> list[str]:
        out = self.pending_keys
        self.pending_keys = []
        return out

    def sim_time(self) -> float:
        return float(self.data.time)

    def reset(self, default_pos: np.ndarray) -> None:
        self.mujoco.mj_resetData(self.model, self.data)
        wcfg = self.cfg["world"]
        self.data.qpos[self.root_qadr : self.root_qadr + 3] = np.asarray(
            wcfg["reset_root_pos"], dtype=np.float32
        )
        self.data.qpos[self.root_qadr + 3 : self.root_qadr + 7] = np.asarray(
            wcfg["reset_root_quat_wxyz"], dtype=np.float32
        )
        self.data.qpos[self.joint_qposadr] = default_pos
        self.data.qvel[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)

    def set_ball_state(self, pos: np.ndarray, vel: np.ndarray) -> None:
        self.data.qpos[self.ball_qadr : self.ball_qadr + 3] = pos
        self.data.qpos[self.ball_qadr + 3 : self.ball_qadr + 7] = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float32
        )
        self.data.qvel[self.ball_dadr : self.ball_dadr + 3] = vel
        self.data.qvel[self.ball_dadr + 3 : self.ball_dadr + 6] = 0.0
        self.mujoco.mj_forward(self.model, self.data)

    def get_state(self) -> RobotState:
        root_pos = (
            self.data.qpos[self.root_qadr : self.root_qadr + 3]
            .astype(np.float32)
            .copy()
        )
        root_quat = quat_normalize_wxyz(
            self.data.qpos[self.root_qadr + 3 : self.root_qadr + 7]
            .astype(np.float32)
            .copy()
        )
        if self.pelvis_gyro_sensor is not None:
            # Training actor uses base_ang_vel_imu rooted at pelvis/base. The
            # MuJoCo gyro sensor reports angular velocity in the pelvis site
            # frame, avoiding freejoint-qvel convention ambiguity.
            ang_vel_b = (
                self.data.sensordata[self.pelvis_gyro_sensor].astype(np.float32).copy()
            )
        else:
            ang_vel_b = (
                self.data.qvel[self.root_dadr + 3 : self.root_dadr + 6]
                .astype(np.float32)
                .copy()
            )
        # Paddle COM linear velocity in world frame. mj_objectVelocity returns
        # 6D [angvel; linvel] for FLG_LINEAR=0 in WORLD frame (flg_local=0).
        paddle_vel6 = np.zeros(6, dtype=np.float64)
        self.mujoco.mj_objectVelocity(
            self.model, self.data,
            int(self.mujoco.mjtObj.mjOBJ_BODY), int(self.paddle_body_id),
            paddle_vel6, 0,   # flg_local=0 → world frame
        )
        paddle_lin_w = paddle_vel6[3:6].astype(np.float32).copy()
        return RobotState(
            base_pos=root_pos,
            base_quat_wxyz=root_quat,
            base_ang_vel_b=ang_vel_b,
            dof_pos=self.data.qpos[self.joint_qposadr].astype(np.float32).copy(),
            dof_vel=self.data.qvel[self.joint_dofadr].astype(np.float32).copy(),
            paddle_pos_world=self.data.xpos[self.paddle_body_id]
            .astype(np.float32)
            .copy(),
            paddle_quat_wxyz=quat_normalize_wxyz(
                self.data.xquat[self.paddle_body_id].astype(np.float32).copy()
            ),
            paddle_lin_vel_world=paddle_lin_w,
            ball_pos_world=self.data.xpos[self.ball_body_id].astype(np.float32).copy(),
            ball_vel_world=self.data.qvel[self.ball_dadr : self.ball_dadr + 3]
            .astype(np.float32)
            .copy(),
        )

    def step(
        self, pd_target: np.ndarray, stiffness: np.ndarray, damping: np.ndarray,
        before_sync_cb=None,
    ) -> None:
        for _ in range(self.sim_decimation):
            q = self.data.qpos[self.joint_qposadr]
            dq = self.data.qvel[self.joint_dofadr]
            tau = (pd_target - q) * stiffness - dq * damping
            tau = np.clip(tau, -self.torque_limits, self.torque_limits)
            self.data.ctrl[self.actuator_ids] = tau
            self.mujoco.mj_step(self.model, self.data)
        if before_sync_cb is not None:
            before_sync_cb()
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

    def step_damping(self, damping: np.ndarray) -> None:
        """Damping-only step for fall recovery / safe-mode (kp=0, target=current_q).

        Mirrors the C++ Passive FSM (deploy/include/FSM/State_Passive.h): zero kp,
        non-zero kd, no commanded torque. Joints decay velocity to zero passively
        without trying to hit a position target — same recipe used on the real G1
        when bad_orientation triggers, so the robot lays down softly instead of
        thrashing under the policy when it has already fallen."""
        for _ in range(self.sim_decimation):
            dq = self.data.qvel[self.joint_dofadr]
            tau = -dq * damping
            tau = np.clip(tau, -self.torque_limits, self.torque_limits)
            self.data.ctrl[self.actuator_ids] = tau
            self.mujoco.mj_step(self.model, self.data)
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

    def shutdown(self) -> None:
        if self.viewer is not None:
            self.viewer.close()

    def ball_paddle_contact(self) -> bool:
        """True iff at least one current contact pair is between the ball geom
        and any geom on the right_paddle_blade body. Used by ErrorRecorder to
        nail down the precise impact frame."""
        ball_gid = self.ball_body_id  # NOTE: contact stores geom ids, not body ids
        # cache geom ids on first call
        if not hasattr(self, "_ball_geom_id"):
            self._ball_geom_id = self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_GEOM, "pingpong_ball_geom"
            )
            self._paddle_geom_ids = set()
            for gi in range(self.model.ngeom):
                if int(self.model.geom_bodyid[gi]) == int(self.paddle_body_id):
                    self._paddle_geom_ids.add(gi)
        if self._ball_geom_id < 0 or not self._paddle_geom_ids:
            return False
        for ci in range(self.data.ncon):
            c = self.data.contact[ci]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 == self._ball_geom_id and g2 in self._paddle_geom_ids) or \
               (g2 == self._ball_geom_id and g1 in self._paddle_geom_ids):
                return True
        return False


class UnitreeSdkBackend:
    def __init__(self, cfg: dict[str, Any], joint_ids_map: list[int], network: str):
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelPublisher,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.default import (
            unitree_hg_msg_dds__LowCmd_,
            unitree_hg_msg_dds__LowState_,
        )
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
        from unitree_sdk2py.utils.crc import CRC

        self.cfg = cfg
        self.joint_ids_map = joint_ids_map
        self.control_dt = float(cfg["unitree"]["control_dt"])
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self._received = False
        self.crc = CRC()
        ChannelFactoryInitialize(0, network)
        self.pub = ChannelPublisher(cfg["unitree"]["lowcmd_topic"], LowCmdHG)
        self.pub.Init()
        self.sub = ChannelSubscriber(cfg["unitree"]["lowstate_topic"], LowStateHG)
        self.sub.Init(self._low_state_handler, 10)
        self._init_low_cmd()
        print(f"[Unitree] waiting for low_state on {network} ...")
        while not self._received:
            time.sleep(self.control_dt)
        print("[Unitree] connected")

    def _init_low_cmd(self) -> None:
        self.low_cmd.mode_pr = 0
        self.low_cmd.mode_machine = int(self.cfg["unitree"].get("mode_machine", 0))
        for mc in self.low_cmd.motor_cmd:
            mc.mode = 1
            mc.q = 0.0
            if hasattr(mc, "qd"):
                mc.qd = 0.0
            if hasattr(mc, "dq"):
                mc.dq = 0.0
            mc.tau = 0.0
            mc.kp = 0.0
            mc.kd = 0.0

    def _low_state_handler(self, msg) -> None:
        self.low_state = msg
        self._received = True

    def reset(self, default_pos: np.ndarray) -> None:
        del default_pos

    def pop_keys(self) -> list[str]:
        return []

    def get_state_with_ball(
        self,
        ball: VirtualBall,
        paddle_quat_wxyz: np.ndarray | None = None,
        paddle_pos: np.ndarray | None = None,
    ) -> RobotState:
        dof_pos = np.asarray(
            [self.low_state.motor_state[i].q for i in self.joint_ids_map],
            dtype=np.float32,
        )
        dof_vel = np.asarray(
            [self.low_state.motor_state[i].dq for i in self.joint_ids_map],
            dtype=np.float32,
        )
        base_quat = quat_normalize_wxyz(
            np.asarray(self.low_state.imu_state.quaternion, dtype=np.float32)
        )
        base_pos = np.asarray(self.cfg["world"]["reset_root_pos"], dtype=np.float32)
        return RobotState(
            base_pos=base_pos,
            base_quat_wxyz=base_quat,
            base_ang_vel_b=np.asarray(
                self.low_state.imu_state.gyroscope, dtype=np.float32
            ),
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            paddle_pos_world=(
                np.zeros(3, dtype=np.float32) if paddle_pos is None else paddle_pos
            ),
            paddle_quat_wxyz=(
                base_quat if paddle_quat_wxyz is None else paddle_quat_wxyz
            ),
            paddle_lin_vel_world=np.zeros(3, dtype=np.float32),
            ball_pos_world=ball.pos.copy(),
            ball_vel_world=ball.vel.copy(),
        )

    def step(
        self, pd_target: np.ndarray, stiffness: np.ndarray, damping: np.ndarray
    ) -> None:
        for local_i, motor_i in enumerate(self.joint_ids_map):
            mc = self.low_cmd.motor_cmd[motor_i]
            mc.mode = 1
            mc.q = float(pd_target[local_i])
            if hasattr(mc, "qd"):
                mc.qd = 0.0
            if hasattr(mc, "dq"):
                mc.dq = 0.0
            mc.tau = 0.0
            mc.kp = float(stiffness[local_i])
            mc.kd = float(damping[local_i])
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

    def step_damping(self, damping: np.ndarray) -> None:
        """Real-robot Passive recipe: kp=0, kd>0, q tracks current motor q, no tau.
        Matches deploy/include/FSM/State_Passive.h on the real G1."""
        for local_i, motor_i in enumerate(self.joint_ids_map):
            mc = self.low_cmd.motor_cmd[motor_i]
            mc.mode = 1
            mc.q = float(self.low_state.motor_state[motor_i].q)
            if hasattr(mc, "qd"):
                mc.qd = 0.0
            if hasattr(mc, "dq"):
                mc.dq = 0.0
            mc.tau = 0.0
            mc.kp = 0.0
            mc.kd = float(damping[local_i])
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

    def shutdown(self) -> None:
        for mc in self.low_cmd.motor_cmd:
            mc.kp = 0.0
            mc.kd = 2.0
            if hasattr(mc, "qd"):
                mc.qd = 0.0
            if hasattr(mc, "dq"):
                mc.dq = 0.0
            mc.tau = 0.0
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)


def ball_dead(cfg: dict[str, Any], pos: np.ndarray) -> bool:
    return bool(pos[2] < 0.05 or pos[0] < -1.0 or pos[0] > 4.2 or abs(pos[1]) > 3.0)


class DebugVisualizer:
    """Render in the MuJoCo viewer the desired (RED) and actual (GREEN) hit-state.

    Mirrors the IsaacLab hitter_real `_debug_vis_callback` (real_commands.py:626):
      RED   sphere @ p_hit_world           — desired contact point
      RED   arrow  @ p_hit + n_target dir  — desired paddle face normal
      RED   arrow  @ p_hit + v_racket dir  — desired racket velocity
      GREEN arrow  @ paddle pos + n_blade  — actual paddle face normal (from FK)
      GREEN arrow  @ paddle pos + blade_vel— actual racket velocity (world frame)

    All drawn into ``viewer.user_scn`` per frame, no persistent state. Called
    from the main loop after the policy step and before viewer.sync() (which
    happens inside env.step()). When the cmd is inactive (idle, no swing) the
    desired-side markers disappear; the green ones still show because the
    paddle has a velocity even between rallies (drift, footwork, etc).
    """

    RED   = np.array([1.0, 0.10, 0.05, 1.0], dtype=np.float32)
    GREEN = np.array([0.05, 0.85, 0.20, 1.0], dtype=np.float32)
    CYAN  = np.array([0.10, 0.85, 0.95, 0.85], dtype=np.float32)  # planner-predicted ball trajectory
    SPHERE_R = 0.025                # marker sphere radius (m)
    TRAJ_DOT_R = 0.012              # planner trajectory dot radius (smaller, ~ ball radius)
    ARROW_W  = 0.0125               # arrow shaft width   (m)
    NORMAL_LEN_RED   = 0.20         # fixed length for desired-normal arrow
    NORMAL_LEN_GREEN = 0.15         # fixed length for actual-normal arrow
    VEL_SCALE  = 0.05               # arrow_length = vel_scale * |v| (clamped)
    VEL_LEN_MIN = 0.10              # min arrow length so slow vel is still visible
    VEL_LEN_MAX = 0.80              # max arrow length so fast vel doesn't fill scene

    def __init__(self, mujoco_module, viewer):
        self._mj = mujoco_module
        self._viewer = viewer

    def update(self, paddle_pos_w: np.ndarray, paddle_quat_wxyz: np.ndarray,
               paddle_lin_vel_w: np.ndarray, cmd: "CommandState") -> None:
        """Refresh the user_scn markers. Safe to call when viewer is None."""
        if self._viewer is None or not self._viewer.is_running():
            return
        scn = self._viewer.user_scn
        scn.ngeom = 0

        # -- ACTUAL (green) — derived from current robot/paddle state --
        n_blade_w = quat_rotate_wxyz(paddle_quat_wxyz, BLADE_NORMAL_LOCAL)
        # signed by swing_type so the visualized "active face" matches what the
        # policy/reward sees (forehand=+normal, backhand=-normal); see
        # observations.py:120 pingpong_active_face_b
        sign = 1.0 - 2.0 * float(cmd.swing_type) if cmd is not None else 1.0
        n_blade_signed = sign * n_blade_w
        self._draw_arrow(paddle_pos_w, n_blade_signed, self.NORMAL_LEN_GREEN, self.GREEN)
        speed = float(np.linalg.norm(paddle_lin_vel_w))
        if speed > 0.05:
            length = float(np.clip(self.VEL_SCALE * speed, self.VEL_LEN_MIN, self.VEL_LEN_MAX))
            self._draw_arrow(paddle_pos_w, paddle_lin_vel_w, length, self.GREEN)

        # -- DESIRED (red) — only when the cmd has an active swing target --
        if cmd is None or not cmd.active or not cmd.planner_valid:
            return
        # Hit point sphere
        self._draw_sphere(cmd.p_hit_world, self.SPHERE_R, self.RED)
        # Desired paddle normal at the hit point
        self._draw_arrow(cmd.p_hit_world, cmd.n_target_world, self.NORMAL_LEN_RED, self.RED)
        # Desired racket velocity at the hit point — slight stagger so it doesn't
        # overlap the normal arrow visually.
        v_des = np.asarray(cmd.v_racket_hat_world, dtype=np.float32)
        speed_des = float(np.linalg.norm(v_des))
        if speed_des > 0.05:
            length = float(np.clip(self.VEL_SCALE * speed_des, self.VEL_LEN_MIN, self.VEL_LEN_MAX))
            offset = 0.04 * cmd.n_target_world / max(np.linalg.norm(cmd.n_target_world), 1e-6)
            self._draw_arrow(cmd.p_hit_world + offset, v_des, length, self.RED)
        # -- PLANNER PREDICTION (cyan) — predicted ball trajectory leading up
        # to the hit. Each valid substep is a small cyan dot so you can eyeball
        # whether the planner thinks the ball reaches p_hit_world. If the dots
        # don't pass through the red sphere, the planner is wrong about where
        # the ball will be at hit time.
        if cmd.planner_traj_world is not None and cmd.planner_traj_valid is not None:
            traj = cmd.planner_traj_world
            valid = cmd.planner_traj_valid
            # Subsample to keep marker count reasonable (max_geom in user_scn).
            stride = max(1, len(traj) // 32)
            for i in range(0, len(traj), stride):
                if i < len(valid) and bool(valid[i]):
                    self._draw_sphere(traj[i], self.TRAJ_DOT_R, self.CYAN)

    def _draw_sphere(self, center: np.ndarray, radius: float, rgba: np.ndarray) -> None:
        scn = self._viewer.user_scn
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        self._mj.mjv_initGeom(
            g,
            type=self._mj.mjtGeom.mjGEOM_SPHERE,
            size=np.array([radius, radius, radius], dtype=np.float64),
            pos=np.asarray(center, dtype=np.float64),
            mat=np.eye(3, dtype=np.float64).flatten(),
            rgba=rgba.astype(np.float32),
        )
        scn.ngeom += 1

    def _draw_arrow(self, base: np.ndarray, direction: np.ndarray, length: float, rgba: np.ndarray) -> None:
        scn = self._viewer.user_scn
        if scn.ngeom >= scn.maxgeom:
            return
        d = np.asarray(direction, dtype=np.float64).reshape(3)
        n = float(np.linalg.norm(d))
        if n < 1.0e-9:
            return
        d = d / n
        start = np.asarray(base, dtype=np.float64).reshape(3)
        end = start + length * d
        g = scn.geoms[scn.ngeom]
        # mjv_initGeom with size=[0,0,0] leaves geometry to be set by mjv_connector
        self._mj.mjv_initGeom(
            g,
            type=self._mj.mjtGeom.mjGEOM_ARROW,
            size=np.zeros(3, dtype=np.float64),
            pos=np.zeros(3, dtype=np.float64),
            mat=np.eye(3, dtype=np.float64).flatten(),
            rgba=rgba.astype(np.float32),
        )
        # mjv_connector (MuJoCo ≥ 3.x): set arrow geometry from-to with given width
        self._mj.mjv_connector(
            g,
            type=self._mj.mjtGeom.mjGEOM_ARROW,
            width=self.ARROW_W,
            from_=start,
            to=end,
        )
        scn.ngeom += 1


class ErrorRecorder:
    """Live matplotlib viewer of the desired-vs-actual racket-velocity / paddle-
    normal error. Two scalar errors logged each control step:
      v_err = ‖v_paddle_actual − v_racket_desired‖           (m/s)
      n_err = 1 − cos(n_blade_actual_signed, n_target_desired) ∈ [0, 2]

    The plot is opened once with matplotlib's interactive mode (TkAgg / Qt)
    and updated in-place from the main loop — no PNG files written. The most
    recent impact is marked as a red vertical line with an annotation showing
    that frame's v_err / n_err values; on every new impact the marker moves.

    Impact-frame detection (in priority order):
      1) Ball-paddle contact (env.ball_paddle_contact() this step)
      2) cmd.t_to_hit zero-crossing while active

    Refresh rate is throttled to ~PLOT_REFRESH_HZ so the matplotlib redraw
    doesn't bottleneck the 50 Hz control loop."""

    BUFFER_S = 10.0               # rolling window (covers ~3 swings @ 3 s cadence)
    PLOT_REFRESH_HZ = 10.0        # plot redraw cadence (control loop is 50 Hz)

    def __init__(self, control_dt: float, hit_plane_x: float = 0.5373):
        self.dt = float(control_dt)
        self.cap = max(int(self.BUFFER_S / self.dt), 32)
        self.hit_plane_x = float(hit_plane_x)
        self.t = deque(maxlen=self.cap)
        self.v_err = deque(maxlen=self.cap)
        self.n_err = deque(maxlen=self.cap)
        self.p_err = deque(maxlen=self.cap)
        self.t_to_hit = deque(maxlen=self.cap)
        self.v_actual = deque(maxlen=self.cap)
        self.v_desired = deque(maxlen=self.cap)
        self.cos_actual_desired = deque(maxlen=self.cap)
        self._last_t_to_hit: float | None = None
        self._impact_t: float | None = None
        self._impact_v_err: float | None = None
        self._impact_n_err: float | None = None
        self._impact_p_err: float | None = None
        self._impact_t_to_hit: float | None = None
        # Per-serve planner-vs-actual timing tracking. We compare the planner's
        # initial estimate (= cmd.t_to_hit at the FIRST step where cmd.active
        # becomes True after a serve) against the moment the ball PHYSICALLY
        # crosses the hit plane (x = hit_plane_x). The error is reported as
        # planner_initial_estimate − actual_serve_to_plane.
        self._serve_t: float | None = None
        self._serve_planner_initial: float | None = None
        self._serve_arrived: bool = False
        self._last_ball_x: float | None = None
        self._serve_events: deque = deque(maxlen=8)  # (serve_t, planner_initial, actual_arr_t, error)
        self._last_redraw_t: float = -1.0
        self._mpl_ok = False
        self._fig = None
        self._axes = None
        self._lines: dict[str, Any] = {}
        self._impact_lines: list[Any] = []
        self._impact_text: list[Any] = []
        self._serve_marker_lines: list[Any] = []
        self._init_plot()

    def _init_plot(self) -> None:
        try:
            import matplotlib
            # If pyplot has already been imported by an earlier dependency, the
            # 'use' call must come BEFORE another import would lock the backend.
            # We try interactive backends in order and stop at the first that
            # actually accepts a switch.
            chosen = None
            for backend in ("TkAgg", "QtAgg", "Qt5Agg", "GTK3Agg"):
                try:
                    matplotlib.use(backend, force=True)
                    chosen = backend
                    break
                except Exception:  # noqa: BLE001
                    continue
            import matplotlib.pyplot as plt
            actual_backend = plt.get_backend()
            print(f"[ErrorRecorder] matplotlib backend: requested={chosen} actual={actual_backend}")
            plt.ion()
            self._plt = plt
            self._fig, self._axes = plt.subplots(4, 1, figsize=(9.0, 9.5), sharex=True)
            try:
                self._fig.canvas.manager.set_window_title("Pingpong swing — desired vs actual")
            except Exception:  # noqa: BLE001
                pass
            ax_v, ax_n, ax_p, ax_t = self._axes
            self._lines["v_err"],     = ax_v.plot([], [], color="tab:blue",   linewidth=1.6, label="‖v_actual − v_desired‖")
            self._lines["v_actual"],  = ax_v.plot([], [], color="tab:green",  linewidth=1.0, linestyle="--", alpha=0.7, label="‖v_paddle‖ (actual)")
            self._lines["v_desired"], = ax_v.plot([], [], color="tab:red",    linewidth=1.0, linestyle="--", alpha=0.7, label="‖v_racket_hat‖ (desired)")
            ax_v.set_ylabel("racket vel (m/s)")
            ax_v.grid(True, alpha=0.3); ax_v.legend(loc="upper left", fontsize=8)
            self._lines["n_err"],     = ax_n.plot([], [], color="tab:purple", linewidth=1.6, label="1 − cos(n_blade, n_target)")
            ax_n.axhline(0, color="k", linewidth=0.4)
            ax_n.axhline(2, color="k", linewidth=0.4, linestyle=":")
            ax_n.set_ylabel("paddle face err (1−cos)")
            ax_n.grid(True, alpha=0.3); ax_n.legend(loc="upper left", fontsize=8)
            self._lines["p_err"],     = ax_p.plot([], [], color="tab:orange", linewidth=1.6, label="‖p_paddle − p_hit‖")
            ax_p.axhline(0, color="k", linewidth=0.4)
            ax_p.set_ylabel("hit pos err (m)")
            ax_p.grid(True, alpha=0.3); ax_p.legend(loc="upper left", fontsize=8)
            # 4th panel: planner-predicted t_to_hit. The impact frame's value
            # IS the timing error — perfect planner = 0 at impact, positive
            # means ball arrived earlier than planner expected, negative means
            # later. Black line at 0 is the ideal target.
            self._lines["t_to_hit"], = ax_t.plot([], [], color="tab:cyan", linewidth=1.6, label="planner cmd.t_to_hit")
            ax_t.axhline(0, color="r", linewidth=0.6, linestyle="--", alpha=0.6, label="ideal (impact at t_to_hit=0)")
            ax_t.set_ylabel("t_to_hit (s)")
            ax_t.set_xlabel("time (s)")
            ax_t.grid(True, alpha=0.3); ax_t.legend(loc="upper left", fontsize=8)
            self._fig.suptitle("Pingpong swing — live error trace (impact = red dotted line)")
            self._fig.tight_layout()
            # plt.show(block=False) actually pops the window with TkAgg/Qt
            # backends. fig.show() alone can be a silent no-op on some setups
            # if the toplevel's mainloop never gets a chance to spin.
            plt.show(block=False)
            # Pump the GUI event loop so the window paints before the policy
            # starts blocking the main thread. ~0.2s is enough for Tk to lay
            # the geometry; without it the window may stay un-mapped.
            plt.pause(0.2)
            # Try to raise the window above MuJoCo's viewer (which gets focus
            # from launch_passive). Best-effort — different backends have
            # different APIs; failures here are silent and don't affect the
            # plot's correctness.
            try:
                mgr = self._fig.canvas.manager
                if hasattr(mgr, "window"):
                    win = mgr.window
                    # Tk: lift to top, then drop the topmost flag so the user
                    # can move it behind other windows if they want.
                    if hasattr(win, "lift"):
                        win.lift()
                    if hasattr(win, "attributes"):
                        try:
                            win.attributes("-topmost", True)
                            win.after(200, lambda: win.attributes("-topmost", False))
                        except Exception:  # noqa: BLE001
                            pass
                    # Qt: activateWindow + raise_
                    if hasattr(win, "activateWindow"):
                        win.activateWindow()
                    if hasattr(win, "raise_"):
                        win.raise_()
            except Exception:  # noqa: BLE001
                pass
            self._mpl_ok = True
            print(f"[ErrorRecorder] live error window opened (look for "
                  f"'Pingpong swing — desired vs actual' window; if hidden, "
                  f"check your taskbar)")
        except Exception as exc:  # noqa: BLE001
            import traceback
            print(f"[ErrorRecorder][warn] matplotlib live plot disabled: {exc}")
            traceback.print_exc()
            self._mpl_ok = False

    def notify_serve(self, t_now: float) -> None:
        """Called from run() the instant a new serve fires. Resets per-serve
        tracking so we measure the ball's flight from this exact moment until
        it crosses the hit plane."""
        self._serve_t = float(t_now)
        self._serve_planner_initial = None
        self._serve_arrived = False
        self._last_ball_x = None
        print(f"[ErrorRecorder] new serve @ t={t_now:.3f}s — tracking ball arrival at hit plane x={self.hit_plane_x:.3f}")

    def step(self, t_now: float, paddle_pos_w: np.ndarray, paddle_lin_vel_w: np.ndarray,
             paddle_quat_wxyz: np.ndarray, cmd: "CommandState", contact_now: bool,
             ball_pos_w: np.ndarray | None = None) -> None:
        # Compute the three errors for this control step.
        sign = 1.0 - 2.0 * float(cmd.swing_type) if cmd is not None else 1.0
        n_blade_w = sign * quat_rotate_wxyz(paddle_quat_wxyz, BLADE_NORMAL_LOCAL)
        n_blade_w = n_blade_w / max(float(np.linalg.norm(n_blade_w)), 1e-9)
        if cmd is not None and cmd.planner_valid:
            n_target = np.asarray(cmd.n_target_world, dtype=np.float32)
            nt_norm = float(np.linalg.norm(n_target))
            if nt_norm > 1e-6:
                n_target = n_target / nt_norm
                cos_sim = float(np.dot(n_blade_w, n_target))
            else:
                cos_sim = 1.0
            v_des = np.asarray(cmd.v_racket_hat_world, dtype=np.float32)
            v_err = float(np.linalg.norm(paddle_lin_vel_w - v_des))
            # Hit-point position error (matches r_g_pos training reward
            # surface: distance from paddle blade to desired contact point).
            p_err = float(np.linalg.norm(np.asarray(paddle_pos_w, dtype=np.float32)
                                         - np.asarray(cmd.p_hit_world, dtype=np.float32)))
            t_to_hit = float(cmd.t_to_hit)
            active = bool(cmd.active)
        else:
            v_des = np.zeros(3, dtype=np.float32)
            cos_sim = 1.0
            v_err = float(np.linalg.norm(paddle_lin_vel_w))
            p_err = float("nan")
            t_to_hit = float("nan")
            active = False

        # Snapshot the planner's INITIAL estimate at the first active step
        # after a serve. cmd.t_to_hit is what the planner predicted as the
        # remaining time-to-impact at this moment; subtract from current
        # (t_now − serve_t) to get planner-believed total flight time, but
        # the cleanest thing is just save (t_now, t_to_hit) and, when ball
        # actually arrives at the plane, compute error.
        if (
            self._serve_t is not None
            and self._serve_planner_initial is None
            and active
            and cmd is not None
            and cmd.planner_valid
        ):
            # planner_initial = how long FROM SERVE TIME the planner expects the
            # ball to take to reach hit-point. If we sample at t_now, planner
            # says "t_to_hit more seconds from now", so total estimate from
            # serve = (t_now − serve_t) + t_to_hit.
            self._serve_planner_initial = (t_now - self._serve_t) + float(t_to_hit)
            print(f"[ErrorRecorder] planner first estimate after serve: "
                  f"serve_to_hit_predicted={self._serve_planner_initial:.3f}s "
                  f"(at t={t_now:.3f}s, planner cmd.t_to_hit={t_to_hit:.3f}s)")

        # Detect ball crossing the hit plane (ball.x decreases past plane_x).
        # Only counted once per serve and only while ball is still travelling
        # toward the robot (decreasing x).
        if (
            ball_pos_w is not None
            and self._serve_t is not None
            and not self._serve_arrived
        ):
            bx = float(ball_pos_w[0])
            if self._last_ball_x is not None:
                if self._last_ball_x > self.hit_plane_x and bx <= self.hit_plane_x:
                    actual_arr = t_now - self._serve_t
                    err = (self._serve_planner_initial - actual_arr) if self._serve_planner_initial is not None else float("nan")
                    self._serve_events.append((self._serve_t, self._serve_planner_initial, actual_arr, err))
                    self._serve_arrived = True
                    if self._serve_planner_initial is not None:
                        print(f"[ErrorRecorder] BALL CROSSED HIT PLANE @ t={t_now:.3f}s  "
                              f"actual_serve_to_plane={actual_arr:.3f}s  "
                              f"planner_initial={self._serve_planner_initial:.3f}s  "
                              f"error={err*1000:+.0f} ms (planner − actual)")
                    else:
                        print(f"[ErrorRecorder] BALL CROSSED HIT PLANE @ t={t_now:.3f}s  "
                              f"actual_serve_to_plane={actual_arr:.3f}s  (no planner estimate captured)")
            self._last_ball_x = bx

        self.t.append(t_now)
        self.v_err.append(v_err)
        self.n_err.append(1.0 - cos_sim)
        self.p_err.append(p_err)
        self.t_to_hit.append(t_to_hit)
        self.v_actual.append(float(np.linalg.norm(paddle_lin_vel_w)))
        self.v_desired.append(float(np.linalg.norm(v_des)))
        self.cos_actual_desired.append(cos_sim)

        # Detect impact: contact OR t_to_hit zero-cross while active
        is_impact = False
        if active and self._last_t_to_hit is not None and self._last_t_to_hit > 0 and t_to_hit <= 0:
            is_impact = True
        if active and contact_now:
            is_impact = True
        if is_impact:
            self._impact_t = t_now
            self._impact_v_err = v_err
            self._impact_n_err = 1.0 - cos_sim
            self._impact_p_err = p_err if p_err == p_err else None  # None if NaN
            self._impact_t_to_hit = t_to_hit if t_to_hit == t_to_hit else None  # NaN-safe
            t_to_hit_str = f"{t_to_hit:+.3f}s" if self._impact_t_to_hit is not None else "n/a"
            print(f"[ErrorRecorder] impact @ t={t_now:.3f}s  v_err={v_err:.3f} m/s  "
                  f"n_err={1-cos_sim:.3f}  p_err={p_err:.3f} m  "
                  f"t_to_hit_at_impact={t_to_hit_str}  cos={cos_sim:.3f}")

        self._last_t_to_hit = t_to_hit

        # Throttled redraw — keep the live figure responsive without blocking
        # the 50 Hz control loop.
        if self._mpl_ok and (t_now - self._last_redraw_t) >= 1.0 / self.PLOT_REFRESH_HZ:
            self._last_redraw_t = t_now
            self._refresh_plot()

    def _refresh_plot(self) -> None:
        if not self._mpl_ok or len(self.t) < 2:
            return
        ts = np.asarray(self.t, dtype=np.float64)
        self._lines["v_err"].set_data(ts, np.asarray(self.v_err))
        self._lines["v_actual"].set_data(ts, np.asarray(self.v_actual))
        self._lines["v_desired"].set_data(ts, np.asarray(self.v_desired))
        self._lines["n_err"].set_data(ts, np.asarray(self.n_err))
        # p_err / t_to_hit may contain NaN during idle — matplotlib handles
        # NaN gaps natively, so we just hand it the raw arrays.
        self._lines["p_err"].set_data(ts, np.asarray(self.p_err))
        self._lines["t_to_hit"].set_data(ts, np.asarray(self.t_to_hit))
        for ax in self._axes:
            ax.relim()
            ax.autoscale_view(scalex=True, scaley=True)
        # Refresh the impact marker (vertical line + annotation) so it always
        # reflects the latest detected impact.
        for line in self._impact_lines:
            try:
                line.remove()
            except Exception:  # noqa: BLE001
                pass
        for line in self._serve_marker_lines:
            try:
                line.remove()
            except Exception:  # noqa: BLE001
                pass
        for txt in self._impact_text:
            try:
                txt.remove()
            except Exception:  # noqa: BLE001
                pass
        self._impact_lines.clear()
        self._serve_marker_lines.clear()
        self._impact_text.clear()
        # Draw a green dotted line for each recent serve. Aids reading the
        # cmd.t_to_hit panel: each serve should re-anchor the curve to a
        # positive value.
        if len(self._serve_events) > 0:
            ts_arr = np.asarray(self.t, dtype=np.float64) if len(self.t) else np.array([0.0])
            t_min, t_max = float(ts_arr.min()), float(ts_arr.max())
            for evt in self._serve_events:
                serve_t = evt[0]
                if serve_t < t_min - 0.5:
                    continue  # off-plot
                for ax in self._axes:
                    self._serve_marker_lines.append(
                        ax.axvline(serve_t, color="tab:green", linewidth=0.8,
                                   linestyle="--", alpha=0.5)
                    )
        if self._impact_t is not None:
            for ax in self._axes:
                self._impact_lines.append(
                    ax.axvline(self._impact_t, color="r", linewidth=1.2, linestyle=":", alpha=0.85)
                )
            ax_v, ax_n, ax_p, ax_t = self._axes
            self._impact_text.append(ax_v.annotate(
                f"impact\nv_err={self._impact_v_err:.2f} m/s",
                xy=(self._impact_t, self._impact_v_err),
                xytext=(8, 12), textcoords="offset points",
                fontsize=8, color="r",
                arrowprops=dict(arrowstyle="->", color="r", lw=0.8),
            ))
            self._impact_text.append(ax_n.annotate(
                f"impact\nn_err={self._impact_n_err:.3f}",
                xy=(self._impact_t, self._impact_n_err),
                xytext=(8, 12), textcoords="offset points",
                fontsize=8, color="r",
                arrowprops=dict(arrowstyle="->", color="r", lw=0.8),
            ))
            if self._impact_p_err is not None:
                self._impact_text.append(ax_p.annotate(
                    f"impact\np_err={self._impact_p_err*100:.1f} cm",
                    xy=(self._impact_t, self._impact_p_err),
                    xytext=(8, 12), textcoords="offset points",
                    fontsize=8, color="r",
                    arrowprops=dict(arrowstyle="->", color="r", lw=0.8),
                ))
            if self._impact_t_to_hit is not None:
                # Annotate the timing error: how far cmd.t_to_hit is from 0
                # at the actual moment of contact. >0 = ball arrived earlier
                # than planner expected; <0 = planner under-estimated, ball
                # arrived late.
                err_ms = self._impact_t_to_hit * 1000.0
                self._impact_text.append(ax_t.annotate(
                    f"timing err\n={err_ms:+.0f} ms",
                    xy=(self._impact_t, self._impact_t_to_hit),
                    xytext=(8, 12), textcoords="offset points",
                    fontsize=8, color="r",
                    arrowprops=dict(arrowstyle="->", color="r", lw=0.8),
                ))
        try:
            self._fig.canvas.draw_idle()
            # plt.pause(tiny) is the only reliable way to pump the Tk event
            # loop here — fig.canvas.flush_events() alone leaves the window
            # unresponsive while the main loop is busy in mj_step. The pause
            # is short enough (<2 ms typical) that the 50 Hz control loop is
            # not noticeably impacted.
            self._plt.pause(0.001)
        except Exception as exc:  # noqa: BLE001
            print(f"[ErrorRecorder][warn] draw error, disabling live plot: {exc}")
            self._mpl_ok = False

    def reset(self) -> None:
        """Drop the buffer (e.g. on R / serve)."""
        self.t.clear(); self.v_err.clear(); self.n_err.clear(); self.p_err.clear()
        self.t_to_hit.clear()
        self.v_actual.clear(); self.v_desired.clear()
        self.cos_actual_desired.clear()
        self._last_t_to_hit = None
        self._impact_t = None
        self._impact_v_err = None
        self._impact_n_err = None
        self._impact_p_err = None
        self._impact_t_to_hit = None
        self._serve_t = None
        self._serve_planner_initial = None
        self._serve_arrived = False
        self._last_ball_x = None
        self._serve_events.clear()


class SafetyMonitor:
    """Fall-detect → damping mode FSM. Mirrors C++ deploy/FSM/State_Passive.h.

    Rule: when the body's projected gravity tilt exceeds ``limit_angle`` rad
    (= acos(-g_b.z)), switch the env into damping-only output until the user
    requests a fresh reset. While in damping mode the policy is bypassed —
    the env applies kp=0 + kd>0 to all joints with no commanded torque, so the
    robot settles softly instead of being whipped around by an actor that has
    lost its world model.

    Once latched, only `disarm()` clears the latch. The runtime calls
    `disarm()` from the reset-key handler so pressing R counts as user consent
    to retry. Auto-clearing on tilt recovery is intentionally NOT done — once
    the robot has fallen, even if the IMU happens to read upright again
    momentarily, the world state (ball, base velocity, joint history) is
    nowhere near in-distribution and re-engaging the policy would just trigger
    another fall.
    """

    def __init__(self, limit_angle_rad: float = 1.0):
        self.limit_angle = float(limit_angle_rad)
        self.fallen = False
        self._last_tilt = 0.0

    def update(self, base_quat_wxyz: np.ndarray) -> bool:
        # Same formula as deploy/include/isaaclab/envs/mdp/terminations.h
        # bad_orientation: tilt = acos(-g_b.z) where g_b is gravity in body frame.
        gravity_b = quat_rotate_inverse_wxyz(
            base_quat_wxyz, np.array([0.0, 0.0, -1.0], dtype=np.float32)
        )
        cos_tilt = -float(gravity_b[2])
        cos_tilt = max(-1.0, min(1.0, cos_tilt))
        tilt = math.acos(cos_tilt)
        self._last_tilt = tilt
        if not self.fallen and tilt > self.limit_angle:
            self.fallen = True
            print(
                f"[Safety] FALL DETECTED tilt={math.degrees(tilt):.1f}° > "
                f"{math.degrees(self.limit_angle):.1f}° → switching to DAMPING mode "
                "(press R to reset)"
            )
        return self.fallen

    def disarm(self) -> None:
        if self.fallen:
            print("[Safety] DAMPING cleared — re-arming policy.")
        self.fallen = False
        self._last_tilt = 0.0

    @property
    def tilt_deg(self) -> float:
        return math.degrees(self._last_tilt)


def apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    planner_cfg = cfg["planner"]
    if args.forward_npz is not None:
        planner_cfg["forward_motion_file"] = args.forward_npz
    if args.backward_npz is not None:
        planner_cfg["backward_motion_file"] = args.backward_npz
    if args.x_hit is not None:
        planner_cfg["x_hit_default"] = float(args.x_hit)
        if cfg.get("serve", {}).get("hit_x") is None:
            cfg["serve"]["hit_x"] = float(args.x_hit)
    if args.serve_hit_x is not None:
        cfg["serve"]["hit_x"] = float(args.serve_hit_x)
    if args.forehand_y_safety_clamp is not None:
        planner_cfg["forehand_y_safety_clamp"] = float(args.forehand_y_safety_clamp)
    if args.no_forehand_y_safety_clamp:
        planner_cfg["forehand_y_safety_clamp"] = None


def run(args: argparse.Namespace) -> None:
    cfg = yaml.safe_load(resolve_path(args.config).read_text())
    apply_cli_overrides(cfg, args)
    policies = PolicyManager(cfg["policy"])
    if args.print_policy_audit or args.check:
        policies.print_audit()
    serve_machine = ServeMachine(cfg)
    cmd_mgr = PingpongCommandManager(cfg)
    backend_is_unitree = args.network is not None

    # Safety FSM: tilt > limit_angle_rad → switch to damping mode (kp=0, kd>0).
    # Mirrors the C++ FSM bad_orientation gate that drops into State_Passive.
    safety_cfg = cfg.get("safety", {}) or {}
    safety = SafetyMonitor(
        limit_angle_rad=float(safety_cfg.get("fall_limit_angle_rad", 1.0))
    )
    warmup_s = float(safety_cfg.get("warmup_s", args.warmup_s))

    if backend_is_unitree:
        env = UnitreeSdkBackend(cfg, policies.active.joint_ids_map, args.network)
        virtual_ball = VirtualBall()
        env.reset(policies.active.default_pos)
    else:
        env = MujocoPingpongEnv(
            cfg, policies.active.joint_names, render=not args.no_render
        )
        virtual_ball = None
        env.reset(policies.active.default_pos)

    # Live viz + per-swing error plotting. Both are no-ops in unitree_real mode
    # (no MuJoCo viewer / no MuJoCo paddle pose) so the same code path runs.
    debug_vis = None
    if not backend_is_unitree and not args.no_render and not args.no_debug_viz:
        import mujoco as _mj  # local import to keep top-of-file lean
        debug_vis = DebugVisualizer(_mj, env.viewer)
    err_recorder = (
        None if args.no_err_plot
        else ErrorRecorder(
            control_dt=float(cfg["robot"]["control_dt"]),
            hit_plane_x=float(cfg["planner"].get("x_hit_default") or 0.5373),
        )
    )

    def serve() -> None:
        pos, vel = serve_machine.sample()
        cmd_mgr.reset()
        if backend_is_unitree:
            virtual_ball.reset(pos, vel)
        else:
            env.set_ball_state(pos, vel)
        print(f"[Serve] pos={pos.round(3).tolist()} vel={vel.round(3).tolist()}")

    def hard_reset() -> None:
        env.reset(policies.active.default_pos)
        policies.reset()
        safety.disarm()
        if err_recorder is not None:
            err_recorder.reset()
        serve()
        if err_recorder is not None:
            # err_recorder.reset() cleared serve tracking, so re-notify with
            # current runtime t so the per-serve plane-arrival measurement
            # works on the freshly reset run.
            err_recorder.notify_serve(runtime_time())

    serve()  # initial serve (pre-loop, t=0)
    if err_recorder is not None:
        err_recorder.notify_serve(0.0)
    policies.reset()

    if args.check:
        state = (
            env.get_state_with_ball(virtual_ball)
            if backend_is_unitree
            else env.get_state()
        )
        cmd, _ = cmd_mgr.update(0.0, state)
        pd = policies.act(state, cmd)
        print(
            "[Check] cmd "
            f"active={cmd.active} valid={cmd.planner_valid} mode={cmd.plan_mode} "
            f"t_to_hit={cmd.t_to_hit:.3f} "
            f"p_hit={np.round(cmd.p_hit_world, 3).tolist()} "
            f"v_racket={np.round(cmd.v_racket_hat_world, 3).tolist()}"
        )
        print(f"[Check] pd_target shape={pd.shape}, first3={pd[:3].round(3).tolist()}")
        env.shutdown()
        return

    print(
        f"[Runtime] warmup_s={warmup_s:.2f}  fall_limit={math.degrees(safety.limit_angle):.1f}°"
    )
    start = time.perf_counter()
    last = start
    last_serve_t = 0.0
    last_safety_log = 0.0

    def runtime_time() -> float:
        if backend_is_unitree:
            return time.perf_counter() - start
        return env.sim_time()

    try:
        while True:
            now_wall = time.perf_counter()
            t = runtime_time()
            dt = min(max(now_wall - last, 0.0), 0.05)
            last = now_wall

            for key in env.pop_keys():
                if key == "reset":
                    hard_reset()
                    t = runtime_time()
                    last_serve_t = t
                elif key == "serve":
                    if safety.fallen:
                        print(
                            "[Safety] ignoring serve while in DAMPING mode (press R to re-arm)."
                        )
                    else:
                        serve()
                        t = runtime_time()
                        last_serve_t = t
                        if err_recorder is not None:
                            err_recorder.notify_serve(t)
                elif key == "next_policy":
                    if not safety.fallen:
                        policies.next()
                elif key == "prev_policy":
                    if not safety.fallen:
                        policies.prev()
                elif key.startswith("policy:"):
                    if not safety.fallen:
                        try:
                            policies.switch_to(key.split(":", 1)[1])
                        except KeyError as exc:
                            print(f"[PolicyManager][WARN] {exc}")

            if backend_is_unitree:
                virtual_ball.step(dt)
                state = env.get_state_with_ball(virtual_ball)
            else:
                state = env.get_state()

            # ── Safety latch (post-warmup only) ──
            # During warmup we accept that the robot may shudder briefly while PD
            # closes the loop; tilt readings during the first ~0.5s aren't
            # meaningful for fall detection. After that, any tilt > limit_angle
            # latches the FSM into damping mode.
            if t > warmup_s:
                safety.update(state.base_quat_wxyz)

            if safety.fallen:
                # Skip planner / policy entirely. Just drain joint velocity with
                # kp=0 + kd>0 so the robot lays down softly. Hold last logged
                # output to ~1Hz so we don't spam the console.
                env.step_damping(policies.active.damping)
                if t - last_safety_log > 1.0:
                    last_safety_log = t
                    print(
                        f"[Safety] DAMPING active (tilt={safety.tilt_deg:.1f}°, press R to reset)"
                    )
                if args.duration is not None and t >= args.duration:
                    break
                if backend_is_unitree:
                    sleep_s = max(
                        0.0,
                        float(cfg["robot"]["control_dt"])
                        - (time.perf_counter() - now_wall),
                    )
                    time.sleep(sleep_s)
                continue

            cmd, serve_after_swing = cmd_mgr.update(t, state)

            # ── Warmup: hold default pose with full PD before engaging policy ──
            # Sends `default_pos` straight into PD so the actor sees a stable
            # in-distribution state on its first inference. Without this the
            # pelvis may still be settling from the reset write — the actor
            # then sees velocities/joint deltas it never trained on and the
            # MuJoCo dynamics diverge from IsaacLab's faster.
            if t < warmup_s:
                env.step(
                    policies.active.default_pos,
                    policies.active.stiffness,
                    policies.active.damping,
                )
                last_serve_t = t  # don't auto-reset during warmup
                if args.duration is not None and t >= args.duration:
                    break
                if backend_is_unitree:
                    sleep_s = max(
                        0.0,
                        float(cfg["robot"]["control_dt"])
                        - (time.perf_counter() - now_wall),
                    )
                    time.sleep(sleep_s)
                continue

            # Auto-serve: strict fixed cadence (config.serve.auto_interval_s,
            # default 3.0 s). Previously also fired on `serve_after_swing` or
            # `ball_dead`, which interrupted swings before the ball reached
            # the paddle. Now serves are time-locked so each swing has a full
            # cycle to play out.
            serve_interval = float(cfg["serve"].get("auto_interval_s", 3.0))
            if cfg["serve"].get("auto_serve", True) and (t - last_serve_t > serve_interval):
                serve()
                last_serve_t = t
                if err_recorder is not None:
                    err_recorder.notify_serve(t)
                state = (
                    env.get_state_with_ball(virtual_ball)
                    if backend_is_unitree
                    else env.get_state()
                )
                cmd, _ = cmd_mgr.update(t, state)

            pd_target = policies.act(state, cmd)

            # Pre-sync hook: refresh red/green markers + log per-swing error
            # using the AFTER-physics state, so the viewer renders them in the
            # same frame as the paddle's new pose.
            def _post_step():
                if debug_vis is None and err_recorder is None:
                    return
                state_post = (
                    env.get_state_with_ball(virtual_ball) if backend_is_unitree
                    else env.get_state()
                )
                if debug_vis is not None:
                    debug_vis.update(
                        state_post.paddle_pos_world,
                        state_post.paddle_quat_wxyz,
                        state_post.paddle_lin_vel_world,
                        cmd,
                    )
                if err_recorder is not None:
                    t_post = runtime_time()
                    contact = (
                        env.ball_paddle_contact() if not backend_is_unitree else False
                    )
                    err_recorder.step(
                        t_post, state_post.paddle_pos_world,
                        state_post.paddle_lin_vel_world,
                        state_post.paddle_quat_wxyz, cmd, contact,
                        ball_pos_w=state_post.ball_pos_world,
                    )

            env.step(
                pd_target, policies.active.stiffness, policies.active.damping,
                before_sync_cb=_post_step if not backend_is_unitree else None,
            )
            # Unitree backend: no MuJoCo viewer, just record error stats.
            if backend_is_unitree and err_recorder is not None:
                err_recorder.step(
                    runtime_time(), state.paddle_pos_world, state.paddle_lin_vel_world,
                    state.paddle_quat_wxyz, cmd, False,
                    ball_pos_w=state.ball_pos_world,
                )

            if args.duration is not None and t >= args.duration:
                break
            if backend_is_unitree:
                sleep_s = max(
                    0.0,
                    float(cfg["robot"]["control_dt"])
                    - (time.perf_counter() - now_wall),
                )
                time.sleep(sleep_s)
    finally:
        env.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="G1 23DoF pingpong MuJoCo sim2sim / Unitree SDK sim2real runner."
    )
    parser.add_argument(
        "--config", default="deploy/robots/g1_23dof_pingpong_mujoco/config/config.yaml"
    )
    parser.add_argument(
        "--forward-npz",
        default=None,
        help="Forehand/forward expert npz used to derive impact geometry.",
    )
    parser.add_argument(
        "--backward-npz",
        default=None,
        help="Backhand/backward expert npz used to derive impact geometry.",
    )
    parser.add_argument(
        "--x-hit",
        type=float,
        default=None,
        help="Override planner hit-plane x. Omit/null to derive from npz.",
    )
    parser.add_argument(
        "--serve-hit-x",
        type=float,
        default=None,
        help="Override synthetic serve target x. Omit/null to follow planner x.",
    )
    parser.add_argument(
        "--forehand-y-safety-clamp",
        type=float,
        default=None,
        help="Override forehand y safety clamp used by training geometry.",
    )
    parser.add_argument(
        "--no-forehand-y-safety-clamp",
        action="store_true",
        help="Disable forehand y safety clamp and use raw npz offsets.",
    )
    parser.add_argument(
        "--network",
        default=None,
        help="Unitree network interface, e.g. enp129s0. Omit for direct MuJoCo sim2sim.",
    )
    parser.add_argument(
        "--no-render", action="store_true", help="Disable MuJoCo viewer."
    )
    parser.add_argument(
        "--duration", type=float, default=None, help="Optional run duration in seconds."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Load config/model and execute one policy step, then exit.",
    )
    parser.add_argument(
        "--print-policy-audit",
        action="store_true",
        help="Print obs/action order, scale, clip, and PD mapping.",
    )
    parser.add_argument(
        "--warmup-s",
        type=float,
        default=0.1,
        help="Seconds at startup to hold default pose under PD before engaging the policy. Overridden by config.safety.warmup_s if set.",
    )
    parser.add_argument(
        "--no-debug-viz",
        action="store_true",
        help="Disable in-viewer markers (red=desired, green=actual). Default: enabled when MuJoCo viewer is on.",
    )
    parser.add_argument(
        "--no-err-plot",
        action="store_true",
        help="Disable the live matplotlib error window (v_err / n_err vs time, impact marked).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
