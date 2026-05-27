#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def _fmt_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    x = float(num_bytes)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{num_bytes} B"


def _to_scalar(x):
    arr = np.asarray(x)
    if arr.ndim == 0:
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return x


def print_npz_summary(path: Path, title: str) -> None:
    data = np.load(path, allow_pickle=True)
    print("\n" + "=" * 90)
    print(f"[{title}]")
    print("=" * 90)
    print(f"path: {path}")
    print(f"size: {_fmt_size(path.stat().st_size)}")
    print(f"num_keys: {len(data.files)}")
    print("keys:")
    for k in sorted(data.files):
        print(f"  - {k}")

    print("\nfield summary:")
    for k in sorted(data.files):
        v = data[k]
        if isinstance(v, np.ndarray):
            if v.dtype.kind in {"U", "S", "O"}:
                preview = v.tolist()
                if isinstance(preview, list) and len(preview) > 8:
                    preview = preview[:8] + ["..."]
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}, preview={preview}")
            else:
                print(
                    f"  {k}: shape={v.shape}, dtype={v.dtype}, "
                    f"min={float(np.min(v)):.5f}, max={float(np.max(v)):.5f}"
                )
        else:
            print(f"  {k}: type={type(v)}")

    if "fps" in data:
        print(f"\nfps: {int(_to_scalar(data['fps']))}")

    if "joint_names" in data:
        print(
            f"joint_names ({len(data['joint_names'])}): {data['joint_names'].tolist()}"
        )
    elif "dof_names" in data:
        print(f"dof_names ({len(data['dof_names'])}): {data['dof_names'].tolist()}")

    if "body_names" in data:
        print(f"body_names ({len(data['body_names'])}): {data['body_names'].tolist()}")

    if "joint_pos" in data:
        print(f"single/mimic frames: {data['joint_pos'].shape[0]}")
    if "dof_positions" in data:
        print(f"amp frames: {data['dof_positions'].shape[0]}")

    if "clip_names" in data:
        clip_names = data["clip_names"].tolist()
        clip_lengths = data["clip_lengths"].astype(int).tolist()
        clip_starts = data["clip_start_indices"].astype(int).tolist()
        print("\nmerged bank clip table:")
        for i, (name, start, length) in enumerate(
            zip(clip_names, clip_starts, clip_lengths)
        ):
            end = start + length - 1
            print(
                f"  [{i:02d}] {name:<35s} start={start:6d}  length={length:6d}  end={end:6d}"
            )

        total = sum(clip_lengths)
        print(f"\nmerged summary: {len(clip_names)} clips, total frames={total}")
        if "category" in data:
            print(f"category: {_to_scalar(data['category'])}")
        if "target_dof" in data:
            print(f"target_dof: {int(_to_scalar(data['target_dof']))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--single", type=str, default=None, help="Path to one single-clip NPZ"
    )
    parser.add_argument(
        "--merged", type=str, default=None, help="Path to one merged-bank NPZ"
    )
    args = parser.parse_args()
    if args.single is not None:
        print_npz_summary(Path(args.single).expanduser().resolve(), "SINGLE CLIP NPZ")
    if args.merged is not None:
        print_npz_summary(Path(args.merged).expanduser().resolve(), "MERGED BANK NPZ")


if __name__ == "__main__":
    main()
