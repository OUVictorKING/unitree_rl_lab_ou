"""Replay G1 29DoF motion CSV(s) produced by GMR (gvhmr_to_robot.py) and emit
23DoF NPZ(s) for ping-pong mimic training.

Differences from ``scripts/mimic/csv_to_npz_for29to23.py``:

* **Directory mode only.** Single-file conversion stays in
  ``scripts/AMP/csv_to_npz_final.py``.
* **Header row auto-stripped.** GMR's CSV has a column-name header on line 1.
* **impact_frame lookup.** For each clip ``<stem>.csv`` we read the companion
  ``_clips_info.csv`` (written by ``cut_from_yaml.py``) at
  ``<INPUT_DIR>/../_clips_info.csv``, take ``impact_in_clip``, and rescale to
  the output fps: ``new_idx = round(impact_in_clip * output_fps / input_fps)``.
* **NPZ extras.** Output NPZ includes ``fps``, ``impact_frame``, ``clip_name``
  in addition to the usual ``joint_pos / joint_vel / body_*`` arrays.
* **Default fps.** input=30, output=50 (mimic policy control rate).

Usage::

    python scripts/pingpong_data_process/csv_to_npz_pingpong.py \
        --input  motion_datasets/pingpong/humanoid_data/final/forward_hand/csv \
        --output motion_datasets/pingpong/humanoid_data/final/forward_hand_npz \
        --input_fps 30 --output_fps 50 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv as _csv
import sys
import traceback
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Batch replay GMR-29DoF CSVs (with header) and emit 23DoF NPZs."
)
parser.add_argument("--input",  "-i", required=True, type=str,
                    help="Directory containing *.csv files produced by GMR gvhmr_to_robot.py.")
parser.add_argument("--output", "-o", required=True, type=str,
                    help="Directory to write *.npz files into.")
parser.add_argument("--input_fps",  type=int, default=30,
                    help="FPS of the source video / GMR csv (default 30).")
parser.add_argument("--output_fps", type=int, default=50,
                    help="Output FPS — must match deployed policy control rate (default 50).")
parser.add_argument("--clips_info", type=str, default=None,
                    help="Override path to _clips_info.csv. "
                         "Default: <INPUT_DIR>/../_clips_info.csv (then <INPUT_DIR>/_clips_info.csv).")
parser.add_argument("--stop_on_error", action="store_true",
                    help="Abort on first clip failure (default: continue with remaining clips).")
parser.add_argument("--overwrite", action="store_true",
                    help="Re-convert files even if the destination NPZ already exists.")
parser.add_argument("--paddle", action="store_true",
                    help="Spawn the robot with the paddle URDF (g1_23dof_rev_1_0_paddle.urdf), "
                         "so body_pos_w / body_quat_w / body_*_vel_w include the "
                         "right_paddle_blade body as the last index. The joint count stays 23 "
                         "(blade is fixed-joint).")
parser.add_argument("--task_name", type=str, default=None,
                    choices=["forward_hand", "backward_hand"],
                    help="Pingpong task name. Maps to swing_type in the npz: "
                         "forward_hand→0 (forehand), backward_hand→1 (backhand). "
                         "If omitted, swing_type is not written and you'll need to patch "
                         "the npz later with update_npz_metadata.py.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


def _resolve_clips_info(input_dir: Path, override: str | None) -> Path:
    if override:
        p = Path(override).expanduser().resolve()
        if not p.exists():
            parser.error(f"--clips_info not found: {p}")
        return p
    candidates = [input_dir.parent / "_clips_info.csv",
                  input_dir / "_clips_info.csv"]
    for c in candidates:
        if c.exists():
            return c
    parser.error(f"_clips_info.csv not found in {candidates}")
    raise SystemExit  # unreachable


def _load_impact_table(clips_info_path: Path) -> dict[str, int]:
    """Map clip stem (e.g. 'forward_001') → impact_in_clip (int, 0-based)."""
    table: dict[str, int] = {}
    with open(clips_info_path, newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            clip = row.get("clip", "").strip()
            if not clip:
                continue
            stem = Path(clip).stem  # forward_001.mp4 -> forward_001
            raw = row.get("impact_in_clip", "").strip()
            if raw == "" or raw.lower() == "none":
                continue
            try:
                table[stem] = int(raw)
            except ValueError:
                continue
    return table


def _resolve_io_pairs(input_dir: Path, output_dir: Path) -> list[tuple[Path, Path]]:
    csv_files = sorted(input_dir.glob("*.csv"))
    csv_files = [c for c in csv_files if c.name != "_clips_info.csv"]
    if not csv_files:
        parser.error(f"no .csv files found under {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return [(c, output_dir / (c.stem + ".npz")) for c in csv_files]


INPUT_DIR  = Path(args_cli.input).expanduser().resolve()
OUTPUT_DIR = Path(args_cli.output).expanduser().resolve()
if not INPUT_DIR.is_dir():
    parser.error(f"--input must be a directory: {INPUT_DIR}")

CLIPS_INFO_PATH = _resolve_clips_info(INPUT_DIR, args_cli.clips_info)
IMPACT_TABLE    = _load_impact_table(CLIPS_INFO_PATH)
IO_PAIRS        = _resolve_io_pairs(INPUT_DIR, OUTPUT_DIR)

print(f"[INFO] input  dir : {INPUT_DIR}")
print(f"[INFO] output dir : {OUTPUT_DIR}")
print(f"[INFO] clips info : {CLIPS_INFO_PATH}  ({len(IMPACT_TABLE)} entries)")
print(f"[INFO] found {len(IO_PAIRS)} csv file(s)")


# launch omniverse app (single instance reused across all CSVs)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_mul,
    quat_slerp,
)

##
# Pre-defined configs
##
from unitree_rl_lab.assets.robots.unitree import (
    UNITREE_G1_23DOF_CFG,
    UNITREE_G1_23DOF_PADDLE_CFG,
)

ROBOT_CFG = UNITREE_G1_23DOF_PADDLE_CFG if args_cli.paddle else UNITREE_G1_23DOF_CFG
print(f"[INFO] robot URDF : {ROBOT_CFG.spawn.asset_path}")


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        # GMR gvhmr_to_robot.py writes a header row on line 1 — skip it.
        motion = torch.from_numpy(np.loadtxt(
            self.motion_file, delimiter=",", skiprows=1
        ))
        motion = motion.to(torch.float32).to(self.device)
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7][:, [3, 0, 1, 2]]  # xyzw -> wxyz
        # GMR with `unitree_g1_23dof` already outputs the 23-DoF order expected by
        # the deploy URDF: legs(12) + waist_yaw(1) + left arm 5 + right arm 5.
        self.motion_dof_poss_input = motion[:, 7:]
        assert self.motion_dof_poss_input.shape[1] == 23, (
            f"expected 23 DoF columns from GMR (use --robot unitree_g1_23dof), "
            f"got {self.motion_dof_poss_input.shape[1]}"
        )

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(
            f"[INFO] motion loaded ({self.motion_file}), duration: {self.duration:.3f}s,"
            f" frames: {self.input_frames}"
        )

    def _interpolate_motion(self):
        times = torch.arange(
            0, self.duration, self.output_dt, device=self.device, dtype=torch.float32
        )
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"[INFO] interpolated to {self.output_frames} frames @ {self.output_fps} fps"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor):
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        self.motion_base_lin_vels = torch.gradient(
            self.motion_base_poss, spacing=self.output_dt, dim=0
        )[0]
        self.motion_dof_vels = torch.gradient(
            self.motion_dof_poss, spacing=self.output_dt, dim=0
        )[0]
        self.motion_base_ang_vels = self._so3_derivative(
            self.motion_base_rots, self.output_dt
        )

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        return omega

    def get_next_state(self):
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


# right_paddle_blade is glued to right_wrist_roll_rubber_hand by a fixed joint
# whose origin (xyz, rpy) is read from the paddle URDF below. IsaacLab merges
# fixed joints by default — disabling that would also expose 7 unrelated fixed
# children (head/imu/d435/mid360/contour/logo), which we don't want in the npz.
# Instead we keep merging on and synthesize the blade body analytically:
#   blade_pos  = wrist_pos + R_wrist · xyz_local
#   blade_quat = wrist_quat ⊗ q_local         (q_local from joint rpy)
#   blade_lin  = wrist_lin + ω × (R_wrist · xyz_local)
#   blade_ang  = wrist_ang                    (rigidly attached → same world ω)
def _parse_paddle_blade_joint_origin(urdf_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (xyz, q_wxyz) of right_paddle_blade_fixed_joint from the paddle URDF."""
    import math
    import xml.etree.ElementTree as ET

    root = ET.parse(urdf_path).getroot()
    joint = next(
        (j for j in root.findall("joint")
         if j.get("name") == "right_paddle_blade_fixed_joint"),
        None,
    )
    if joint is None:
        raise RuntimeError(
            f"right_paddle_blade_fixed_joint not found in {urdf_path}"
        )
    origin = joint.find("origin")
    xyz = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()],
                   dtype=np.float32)
    r, p, y = (float(v) for v in origin.get("rpy", "0 0 0").split())
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    q_wxyz = np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float32)
    return xyz, q_wxyz


PADDLE_OFFSET_LOCAL, PADDLE_QUAT_LOCAL = _parse_paddle_blade_joint_origin(
    UNITREE_G1_23DOF_PADDLE_CFG.spawn.asset_path
)


def _rotate_vec_by_quat_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v (..., 3) by quaternion q (..., 4) in wxyz convention."""
    w = q[..., 0:1]
    qv = q[..., 1:4]
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 ⊗ q2 in wxyz convention, broadcasting over leading dims."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def convert_one_csv(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    robot_joint_indexes,
    wrist_body_idx: int | None,
    body_names: list[str],
    input_csv: Path,
    output_npz: Path,
):
    """Run one full motion replay and write a single NPZ."""
    motion = MotionLoader(
        motion_file=str(input_csv),
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
    )

    clip_stem = input_csv.stem
    if clip_stem not in IMPACT_TABLE:
        raise RuntimeError(
            f"no impact_in_clip for {clip_stem} in {CLIPS_INFO_PATH}"
        )
    impact_in_clip = IMPACT_TABLE[clip_stem]
    new_impact_idx = int(round(
        impact_in_clip * args_cli.output_fps / args_cli.input_fps
    ))
    new_impact_idx = max(0, min(new_impact_idx, motion.output_frames - 1))
    print(f"[INFO] {clip_stem}: impact_in_clip={impact_in_clip} "
          f"(@{args_cli.input_fps}fps) → {new_impact_idx} (@{args_cli.output_fps}fps)")

    robot = scene["robot"]
    log = {
        "fps":          [args_cli.output_fps],
        "impact_frame": [new_impact_idx],
        "clip_name":    [clip_stem],
        # body_names ordered to match body_pos_w / body_quat_w / body_*_vel_w
        # second-axis indexing. paddle synthesis appends "right_paddle_blade"
        # as the final body, so it shows up in this list iff --paddle.
        "body_names":   np.array(body_names, dtype="U64"),
        "joint_pos":      [],
        "joint_vel":      [],
        "body_pos_w":     [],
        "body_quat_w":    [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    if args_cli.task_name is not None:
        # forward_hand → 0 (forehand), backward_hand → 1 (backhand)
        log["swing_type"] = np.array(
            [0 if args_cli.task_name == "forward_hand" else 1],
            dtype=np.int8,
        )
    file_saved = False

    while simulation_app.is_running() and not file_saved:
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()  # render-only, no physics step
        scene.update(sim.get_physics_dt())

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if reset_flag:
            for k in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ):
                log[k] = np.stack(log[k], axis=0)

            if args_cli.paddle:
                if wrist_body_idx is None:
                    raise RuntimeError(
                        "wrist body index unavailable; cannot synthesize paddle blade"
                    )
                wpos  = log["body_pos_w"][:, wrist_body_idx]    # (T, 3)
                wquat = log["body_quat_w"][:, wrist_body_idx]   # (T, 4) wxyz
                wlin  = log["body_lin_vel_w"][:, wrist_body_idx]
                wang  = log["body_ang_vel_w"][:, wrist_body_idx]
                offset_w = _rotate_vec_by_quat_wxyz(wquat, PADDLE_OFFSET_LOCAL)
                blade_pos  = wpos + offset_w
                blade_quat = _quat_mul_wxyz(wquat, np.broadcast_to(PADDLE_QUAT_LOCAL, wquat.shape))
                blade_lin  = wlin + np.cross(wang, offset_w)
                blade_ang  = wang.copy()
                log["body_pos_w"]     = np.concatenate([log["body_pos_w"],     blade_pos[:, None]],  axis=1)
                log["body_quat_w"]    = np.concatenate([log["body_quat_w"],    blade_quat[:, None]], axis=1)
                log["body_lin_vel_w"] = np.concatenate([log["body_lin_vel_w"], blade_lin[:, None]],  axis=1)
                log["body_ang_vel_w"] = np.concatenate([log["body_ang_vel_w"], blade_ang[:, None]],  axis=1)

            output_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez(str(output_npz), **log)
            file_saved = True

    return file_saved


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    valid_joint_names: list[str],
):
    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(valid_joint_names, preserve_order=True)[0]

    # body_names ordered to match the second axis of body_pos_w / body_quat_w /
    # body_*_vel_w. With --paddle, convert_one_csv() synthesizes the blade body
    # and appends it as the last index, so we mirror that here.
    body_names = list(robot.body_names)

    wrist_body_idx: int | None = None
    if args_cli.paddle:
        wrist_ids, wrist_names = robot.find_bodies(["right_wrist_roll_rubber_hand"])
        if not wrist_ids:
            raise RuntimeError(
                "right_wrist_roll_rubber_hand body not found in articulation; "
                "paddle URDF expected"
            )
        wrist_body_idx = int(wrist_ids[0])
        print(f"[INFO] wrist body : index={wrist_body_idx} name={wrist_names[0]}")
        body_names.append("right_paddle_blade")

    print(f"[INFO] body_names ({len(body_names)}): {body_names}")
    if args_cli.task_name is not None:
        swing_type = 0 if args_cli.task_name == "forward_hand" else 1
        print(f"[INFO] task_name = {args_cli.task_name} → swing_type = {swing_type}")
    else:
        print("[WARN] --task_name not given; swing_type will be omitted from npz")

    n_total = len(IO_PAIRS)
    n_ok = 0
    n_skipped = 0
    n_err = 0

    for idx, (input_csv, output_npz) in enumerate(IO_PAIRS, start=1):
        if not simulation_app.is_running():
            print("[WARN] Isaac Sim app stopped; aborting batch.")
            break

        rel_in = input_csv.relative_to(INPUT_DIR)
        rel_out = output_npz.relative_to(OUTPUT_DIR)

        if output_npz.exists() and not args_cli.overwrite:
            print(f"[SKIP] ({idx}/{n_total}) {rel_in} -> {rel_out} (output exists, use --overwrite)")
            n_skipped += 1
            continue

        print(f"[RUN ] ({idx}/{n_total}) {rel_in} -> {rel_out}")
        try:
            ok = convert_one_csv(
                sim=sim,
                scene=scene,
                robot_joint_indexes=robot_joint_indexes,
                wrist_body_idx=wrist_body_idx,
                body_names=body_names,
                input_csv=input_csv,
                output_npz=output_npz,
            )
            if ok:
                print(f"[OK  ] saved {output_npz}")
                n_ok += 1
            else:
                print(f"[ERR ] {rel_in} did not finish (sim exited?)")
                n_err += 1
                if args_cli.stop_on_error:
                    break
        except Exception as e:  # noqa: BLE001
            n_err += 1
            print(f"[ERR ] {rel_in}: {e}")
            traceback.print_exc()
            if args_cli.stop_on_error:
                break

    print(
        f"[DONE] ok={n_ok}  skipped={n_skipped}  errors={n_err}  total={n_total}"
    )
    simulation_app.close()


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO] Setup complete...")
    valid_joint_names = [
        name for name in scene_cfg.robot.joint_sdk_names if name != ""  # type: ignore[attr-defined]
    ]
    run_simulator(sim, scene, valid_joint_names)


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            simulation_app.close()
        except Exception:  # noqa: BLE001
            pass
        sys.exit(0)
