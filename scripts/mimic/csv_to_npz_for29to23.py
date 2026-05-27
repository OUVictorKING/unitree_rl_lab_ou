"""Replay G1 29DoF motion CSV(s) and emit 23DoF NPZ(s).

This script supports two modes:

* Single file::

    python scripts/mimic/csv_to_npz_for29to23.py \\
        --input  /path/to/one_motion.csv \\
        --output /path/to/one_motion.npz

  ``--input_file/-f`` and ``--output_name`` are still accepted for
  backwards compatibility.

* Directory (recursive)::

    python scripts/mimic/csv_to_npz_for29to23.py \\
        --input  /path/to/csv_dir \\
        --output /path/to/npz_dir

  Every ``*.csv`` under the input directory is converted to a matching
  ``*.npz`` under the output directory, preserving the relative path
  layout.  All conversions run inside a single Isaac Sim launch — Isaac
  startup is the dominant cost otherwise.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Replay motion from csv file(s) and output to npz file(s)."
)
# Backwards-compatible aliases.  --input/--output are the new canonical names
# but --input_file/-f and --output_name still work as before.
parser.add_argument(
    "--input",
    "-i",
    dest="input",
    type=str,
    default=None,
    help="Input path: a single .csv file or a directory containing .csv files.",
)
parser.add_argument(
    "--output",
    "-o",
    dest="output",
    type=str,
    default=None,
    help=(
        "Output path: a single .npz file (only valid for single-file input) or"
        " a directory (required for directory input)."
    ),
)
parser.add_argument(
    "--input_file",
    "-f",
    dest="input_file",
    type=str,
    default=None,
    help="(deprecated) Path to a single input CSV. Equivalent to --input.",
)
parser.add_argument(
    "--output_name",
    dest="output_name",
    type=str,
    default=None,
    help="(deprecated) Path to a single output NPZ. Equivalent to --output.",
)
parser.add_argument(
    "--input_fps", type=int, default=60, help="The fps of the input motion."
)
parser.add_argument(
    "--output_fps", type=int, default=50, help="The fps of the output motion."
)
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=(
        "frame range: START END (both inclusive). The frame index starts from 1. If not"
        " provided, all frames will be loaded.  Only meaningful for single-file mode."
    ),
)
parser.add_argument(
    "--continue_on_error",
    action="store_true",
    help=(
        "Directory mode only: when a single CSV fails, log the error and keep"
        " converting the remaining files instead of aborting."
    ),
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Re-convert files even if the destination NPZ already exists.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()


def _resolve_io_pairs(args) -> tuple[list[tuple[Path, Path]], bool]:
    """Resolve (input_csv, output_npz) pairs.

    Returns (pairs, is_directory_mode).  Validates inputs and creates output
    directories as needed.
    """
    raw_input = args.input or args.input_file
    raw_output = args.output or args.output_name
    if raw_input is None:
        parser.error("must provide --input (or legacy --input_file/-f)")

    input_path = Path(raw_input).expanduser().resolve()

    if input_path.is_file():
        if input_path.suffix.lower() != ".csv":
            parser.error(f"--input file must end in .csv (got {input_path.suffix})")
        if raw_output is None:
            output_npz = input_path.with_suffix(".npz")
        else:
            out = Path(raw_output).expanduser().resolve()
            if out.is_dir() or raw_output.endswith("/"):
                out.mkdir(parents=True, exist_ok=True)
                output_npz = out / input_path.with_suffix(".npz").name
            else:
                output_npz = out
                output_npz.parent.mkdir(parents=True, exist_ok=True)
        return [(input_path, output_npz)], False

    if input_path.is_dir():
        if raw_output is None:
            parser.error("--output is required when --input is a directory")
        out_dir = Path(raw_output).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_files = sorted(input_path.rglob("*.csv"))
        if not csv_files:
            parser.error(f"no .csv files found under {input_path}")
        pairs: list[tuple[Path, Path]] = []
        for csv_file in csv_files:
            rel = csv_file.relative_to(input_path)
            output_npz = out_dir / rel.with_suffix(".npz")
            pairs.append((csv_file, output_npz))
        if args.frame_range is not None:
            print(
                "[WARN] --frame_range is ignored when --input is a directory;"
                " all frames of every CSV will be used."
            )
        return pairs, True

    parser.error(f"--input does not exist: {input_path}")


IO_PAIRS, IS_DIR_MODE = _resolve_io_pairs(args_cli)

if IS_DIR_MODE:
    print(f"[INFO] input dir : {Path(args_cli.input or args_cli.input_file).resolve()}")
    print(f"[INFO] output dir: {Path(args_cli.output or args_cli.output_name).resolve()}")
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
    UNITREE_G1_23DOF_CFG as ROBOT_CFG,
)


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
        frame_range: tuple[int, int] | None,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        if self.frame_range is None:
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
        else:
            motion = torch.from_numpy(
                np.loadtxt(
                    self.motion_file,
                    delimiter=",",
                    skiprows=self.frame_range[0] - 1,
                    max_rows=self.frame_range[1] - self.frame_range[0] + 1,
                )
            )
        motion = motion.to(torch.float32).to(self.device)
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7][:, [3, 0, 1, 2]]  # xyzw -> wxyz
        motion_dof_29 = motion[:, 7:]
        keep_dof_indices = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            15, 16, 17, 18, 19, 22, 23, 24, 25, 26,
        ]
        self.motion_dof_poss_input = motion_dof_29[:, keep_dof_indices]

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


def convert_one_csv(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    robot_joint_indexes,
    input_csv: Path,
    output_npz: Path,
):
    """Run one full motion replay and write a single NPZ."""
    motion = MotionLoader(
        motion_file=str(input_csv),
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range if not IS_DIR_MODE else None,
    )

    robot = scene["robot"]
    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
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

    base = (
        Path(args_cli.input or args_cli.input_file).expanduser().resolve()
        if IS_DIR_MODE
        else None
    )

    n_total = len(IO_PAIRS)
    n_ok = 0
    n_skipped = 0
    n_err = 0

    for idx, (input_csv, output_npz) in enumerate(IO_PAIRS, start=1):
        if not simulation_app.is_running():
            print("[WARN] Isaac Sim app stopped; aborting batch.")
            break

        rel_in = input_csv.relative_to(base) if (IS_DIR_MODE and base is not None) else input_csv
        rel_out = output_npz.relative_to(Path(args_cli.output or args_cli.output_name).expanduser().resolve()) if IS_DIR_MODE else output_npz

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
                input_csv=input_csv,
                output_npz=output_npz,
            )
            if ok:
                print(f"[OK  ] saved {output_npz}")
                n_ok += 1
            else:
                print(f"[ERR ] {rel_in} did not finish (sim exited?)")
                n_err += 1
                if not args_cli.continue_on_error:
                    break
        except Exception as e:  # noqa: BLE001
            n_err += 1
            print(f"[ERR ] {rel_in}: {e}")
            traceback.print_exc()
            if not args_cli.continue_on_error:
                break

    print(
        f"[DONE] ok={n_ok}  skipped={n_skipped}  errors={n_err}  total={n_total}"
    )

    # Programmatic shutdown — preserves a clean exit even in single-file mode,
    # which previously required manual Ctrl-C after the save.
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
        # main() already calls simulation_app.close(); guard against early exits
        # so the second close is a no-op without erroring out the script.
        try:
            simulation_app.close()
        except Exception:  # noqa: BLE001
            pass
        sys.exit(0)
