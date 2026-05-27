"""Visualize an AMP motion NPZ in Isaac Sim.

Kinematic playback of the per-frame state stored in an AMP ``.npz`` — we
directly write root + joint states to a G1 articulation each step. No
physics step, so the robot exactly tracks the clip.

The AMP NPZ schema (produced by ``scripts/AMP/csv_to_npz_final.py`` and
``scripts/AMP/augment_motion_npz.py``):

    fps             (1,) int/float
    joint_pos       (T, J)           J = 23 for G1-23DoF, **articulation order**
    joint_vel       (T, J)
    body_pos_w      (T, N, 3)        N = 24 for G1-23DoF, **articulation order**
    body_quat_w     (T, N, 4)        wxyz
    body_lin_vel_w  (T, N, 3)
    body_ang_vel_w  (T, N, 3)

Important: the NPZ's ``joint_pos`` / ``body_pos_w`` columns are in
**articulation order** (they were recorded from ``robot.data.joint_pos[0]``
and ``robot.data.body_pos_w[0]`` by ``csv_to_npz_final.py``). This script
writes them back to the articulation with **no remapping** — exactly the
pattern used by :mod:`scripts/mimic/replay_npz.py`.

Usage
-----
    # Single-clip replay:
    python scripts/AMP/visualize_motion_npz.py -f motion_datasets/penguin/g1_qie_motion.npz

    # Side-by-side compare (original left, augmented right, 2m apart):
    python scripts/AMP/visualize_motion_npz.py \
        -f motion_datasets/penguin/g1_qie_motion.npz \
        --compare motion_datasets/penguin/g1_qie_motion_aug.npz
"""

from __future__ import annotations

import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Kinematically replay an AMP NPZ in Isaac Sim.")
parser.add_argument("--file", "-f", type=str, required=True, help="Input NPZ path.")
parser.add_argument("--compare", type=str, default=None,
                    help="Second NPZ to replay side-by-side (2 m offset).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim BEFORE importing any isaaclab.* modules that touch the stage.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# -----------------------------------------------------------------------------
# Imports that require the app to be launched first
# -----------------------------------------------------------------------------

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_23DOF_CFG as ROBOT_CFG


# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------


@configclass
class VizSceneCfg(InteractiveSceneCfg):
    """One or two G1-23DoF robots on a ground plane."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# -----------------------------------------------------------------------------
# NPZ loader
# -----------------------------------------------------------------------------

REQUIRED_KEYS = ("fps", "joint_pos", "joint_vel",
                 "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def load_amp_npz(path: str, device: str | torch.device) -> dict:
    raw = np.load(path, allow_pickle=False)
    missing = [k for k in REQUIRED_KEYS if k not in raw.files]
    if missing:
        raise KeyError(f"{path} missing required keys: {missing}")
    out = {
        "fps": float(np.asarray(raw["fps"]).reshape(-1)[0]),
        "joint_pos": torch.from_numpy(np.asarray(raw["joint_pos"], dtype=np.float32)).to(device),
        "joint_vel": torch.from_numpy(np.asarray(raw["joint_vel"], dtype=np.float32)).to(device),
        "body_pos_w": torch.from_numpy(np.asarray(raw["body_pos_w"], dtype=np.float32)).to(device),
        "body_quat_w": torch.from_numpy(np.asarray(raw["body_quat_w"], dtype=np.float32)).to(device),
        "body_lin_vel_w": torch.from_numpy(np.asarray(raw["body_lin_vel_w"], dtype=np.float32)).to(device),
        "body_ang_vel_w": torch.from_numpy(np.asarray(raw["body_ang_vel_w"], dtype=np.float32)).to(device),
    }
    # Optional self-describing name lists (written by csv_to_npz_final.py).
    if "joint_names" in raw.files:
        out["joint_names"] = [str(x) for x in np.asarray(raw["joint_names"]).reshape(-1).tolist()]
    if "body_names" in raw.files:
        out["body_names"] = [str(x) for x in np.asarray(raw["body_names"]).reshape(-1).tolist()]
    T = out["joint_pos"].shape[0]
    names_tag = (
        f"  joint_names={'yes' if 'joint_names' in out else 'no'}"
        f"  body_names={'yes' if 'body_names' in out else 'no'}"
    )
    print(f"[load]  {path}  T={T}  fps={out['fps']}  "
          f"J={out['joint_pos'].shape[1]}  N={out['body_pos_w'].shape[1]}{names_tag}")
    return out


# -----------------------------------------------------------------------------
# Simulation loop
# -----------------------------------------------------------------------------


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene,
                  clips: list[dict]) -> None:
    robot: Articulation = scene["robot"]
    num_envs = scene.num_envs
    assert len(clips) == num_envs, f"{len(clips)} clips vs {num_envs} envs"

    # Per-env articulation-joint and -body counts (read from the live robot
    # so we never out-of-bounds the articulation tensors).
    J_art = robot.data.default_joint_pos.shape[1]
    N_art = robot.data.body_pos_w.shape[1]
    art_joint_names = list(robot.data.joint_names)
    art_body_names = list(robot.data.body_names)
    for e, c in enumerate(clips):
        if c["joint_pos"].shape[1] != J_art:
            raise RuntimeError(
                f"clip {e}: J={c['joint_pos'].shape[1]} != articulation J={J_art}. "
                "The NPZ must be saved from the same articulation (G1-23DoF)."
            )
        if c["body_pos_w"].shape[1] != N_art:
            raise RuntimeError(
                f"clip {e}: N_bodies={c['body_pos_w'].shape[1]} != articulation N={N_art}."
            )
        # If the npz carries self-describing name lists, cross-check them
        # against the live articulation. This catches the "npz from a
        # different articulation order is being written directly back into
        # this one" bug loudly instead of silently producing a scrambled
        # playback.
        if "joint_names" in c and list(c["joint_names"]) != art_joint_names:
            raise RuntimeError(
                f"clip {e}: joint_names in npz do not match the live articulation.\n"
                f"  npz = {list(c['joint_names'])}\n"
                f"  art = {art_joint_names}\n"
                "Regenerate the npz via scripts/AMP/csv_to_npz_final.py against "
                "the same articulation."
            )
        if "body_names" in c and list(c["body_names"]) != art_body_names:
            raise RuntimeError(
                f"clip {e}: body_names in npz do not match the live articulation.\n"
                f"  npz = {list(c['body_names'])}\n"
                f"  art = {art_body_names}"
            )

    T_arr = torch.tensor([c["joint_pos"].shape[0] for c in clips],
                         dtype=torch.long, device=sim.device)
    current_idx = torch.zeros(num_envs, dtype=torch.long, device=sim.device)

    sim.set_camera_view([3.0, 3.0, 1.2], [0.0, 0.0, 0.8])
    sim_dt = sim.get_physics_dt()

    loop_counter = 0
    while simulation_app.is_running():
        # Wrap at end-of-clip per env (each clip may have its own T).
        current_idx = current_idx % T_arr

        # Root state: NPZ body[0] is the articulation root (pelvis) in
        # world frame. For multi-env, shift by env_origins so the clips
        # play side-by-side instead of overlapping.
        root_states = robot.data.default_root_state.clone()
        for e in range(num_envs):
            f = int(current_idx[e].item())
            c = clips[e]
            root_states[e, :3] = c["body_pos_w"][f, 0] + scene.env_origins[e]
            root_states[e, 3:7] = c["body_quat_w"][f, 0]
            root_states[e, 7:10] = c["body_lin_vel_w"][f, 0]
            root_states[e, 10:] = c["body_ang_vel_w"][f, 0]

        # Joint state: NPZ columns are ALREADY in articulation order — write
        # directly. No SDK-name lookup, no scatter.
        joint_pos_write = robot.data.default_joint_pos.clone()
        joint_vel_write = robot.data.default_joint_vel.clone()
        for e in range(num_envs):
            f = int(current_idx[e].item())
            c = clips[e]
            joint_pos_write[e] = c["joint_pos"][f]
            joint_vel_write[e] = c["joint_vel"][f]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(joint_pos_write, joint_vel_write)
        scene.write_data_to_sim()
        sim.render()           # kinematic playback — don't sim.step()
        scene.update(sim_dt)

        # Follow the first env's pelvis.
        cam_target = root_states[0, :3].detach().cpu().numpy()
        sim.set_camera_view(cam_target + np.array([2.2, 2.2, 0.6]), cam_target)

        current_idx += 1
        loop_counter += 1
        if loop_counter % 200 == 0:
            msg = "  ".join(
                f"env{e}: {int(current_idx[e].item()) % int(T_arr[e].item())}/{int(T_arr[e].item())}"
                for e in range(num_envs)
            )
            print(f"[play]  {msg}")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main() -> None:
    # Peek at the primary clip's fps so the sim paces real-time.
    probe = np.load(args_cli.file, allow_pickle=False)
    clip_fps = float(np.asarray(probe["fps"]).reshape(-1)[0])
    probe.close()

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / clip_fps)
    sim = SimulationContext(sim_cfg)

    num_envs = 2 if args_cli.compare else 1
    scene_cfg = VizSceneCfg(num_envs=num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    clips = [load_amp_npz(args_cli.file, sim.device)]
    if args_cli.compare:
        clips.append(load_amp_npz(args_cli.compare, sim.device))

    # If the two clips disagree on fps, warn — the sim paces at clip-0's rate.
    if len(clips) == 2 and abs(clips[0]["fps"] - clips[1]["fps"]) > 1e-3:
        print(f"[warn]  clip fps differ ({clips[0]['fps']} vs {clips[1]['fps']}); "
              f"sim paces at {clips[0]['fps']} Hz.")

    run_simulator(sim, scene, clips)


if __name__ == "__main__":
    main()
    simulation_app.close()
