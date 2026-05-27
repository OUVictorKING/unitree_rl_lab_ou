"""Select clips whose racket-impact x-offset (in pelvis frame) is close to paper's 0.4m.

Reads npz files from forward_hand_npz/ and backward_hand_npz/, computes the
forward (base-frame +x) distance from pelvis to right_paddle_blade at the
impact frame, and copies clips whose distance lies in a tolerance band around
0.40m to motion_datasets/pingpong/humanoid_data/final/expert/{forward,backward}/.

The user then manually picks the 2 best clips (1 forehand + 1 backhand) from
the expert/ folder as paper-aligned reference clips.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

ROOT = Path("/home/woan/HumanoidProject/unitree_rl_lab/motion_datasets/pingpong/humanoid_data/final")
PELVIS_IDX = 0
BLADE_IDX = 24
PAPER_X = 0.40


def yaw_from_wxyz(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def pelvis_frame_offset(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    imp = int(d["impact_frame"][0])
    p_pelv = d["body_pos_w"][imp, PELVIS_IDX]
    p_blade = d["body_pos_w"][imp, BLADE_IDX]
    q_pelv = d["body_quat_w"][imp, PELVIS_IDX]
    yaw = yaw_from_wxyz(q_pelv)
    diff = p_blade - p_pelv
    c, s = np.cos(-yaw), np.sin(-yaw)
    dx_local = c * diff[0] - s * diff[1]
    dy_local = s * diff[0] + c * diff[1]
    dz_local = diff[2]
    n_frames = d["joint_pos"].shape[0]
    fps = int(d["fps"][0])
    swing = int(d["swing_type"][0])
    return {
        "name": npz_path.stem,
        "dx": float(dx_local),
        "dy": float(dy_local),
        "dz": float(dz_local),
        "imp": imp,
        "n_frames": n_frames,
        "fps": fps,
        "swing": swing,
        "v_blade": float(np.linalg.norm(d["body_lin_vel_w"][imp, BLADE_IDX])),
    }


def process(src_dir: Path, out_dir: Path, lo: float, hi: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_files = sorted(src_dir.glob("*.npz"))
    rows = [pelvis_frame_offset(p) for p in npz_files]
    selected = [r for r in rows if lo <= r["dx"] <= hi]
    selected.sort(key=lambda r: abs(r["dx"] - PAPER_X))
    print(f"\n=== {src_dir.name}: {len(rows)} clips, {len(selected)} pass dx ∈ [{lo:.2f}, {hi:.2f}] ===")
    print(f"  copying to {out_dir}")
    for r in selected:
        src = src_dir / f"{r['name']}.npz"
        dst = out_dir / src.name
        shutil.copy2(src, dst)
    print(f"  {'name':<22} {'dx':>6} {'|dx-0.4|':>9} {'dy':>6} {'dz':>6} {'imp':>4} {'frames':>6} {'v_blade':>8}")
    for r in selected:
        print(f"  {r['name']:<22} {r['dx']:>6.3f} {abs(r['dx']-PAPER_X):>9.3f} "
              f"{r['dy']:>6.3f} {r['dz']:>6.3f} {r['imp']:>4d} {r['n_frames']:>6d} {r['v_blade']:>8.3f}")
    rejected = [r for r in rows if not (lo <= r["dx"] <= hi)]
    if rejected:
        print(f"\n  rejected ({len(rejected)} clips, dx outside band):")
        for r in sorted(rejected, key=lambda r: abs(r["dx"] - PAPER_X))[:10]:
            print(f"    {r['name']:<22} dx={r['dx']:>6.3f}  (Δ={r['dx']-PAPER_X:+.3f})")
        if len(rejected) > 10:
            print(f"    ... {len(rejected) - 10} more")
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=float, default=0.30, help="min forward dx (m)")
    ap.add_argument("--hi", type=float, default=0.55, help="max forward dx (m)")
    args = ap.parse_args()

    expert_root = ROOT / "expert"
    fwd = process(ROOT / "forward_hand_npz", expert_root / "forward", args.lo, args.hi)
    bwd = process(ROOT / "backward_hand_npz", expert_root / "backward", args.lo, args.hi)
    print(f"\n=== summary ===")
    print(f"  forward selected: {len(fwd)}  (paper x=0.40m, top-3 closest:")
    for r in fwd[:3]:
        print(f"    {r['name']:<22} dx={r['dx']:.3f}  v_blade={r['v_blade']:.2f} m/s")
    print(f"  backward selected: {len(bwd)}  top-3 closest:")
    for r in bwd[:3]:
        print(f"    {r['name']:<22} dx={r['dx']:.3f}  v_blade={r['v_blade']:.2f} m/s")
    print(f"\n  expert root: {expert_root}")
    print(f"  → 用户手动从 expert/forward 和 expert/backward 各挑 1 个最标准的动作")


if __name__ == "__main__":
    main()
