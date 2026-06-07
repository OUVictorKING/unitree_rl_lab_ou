#!/usr/bin/env python3
"""Live plot Pingpong hit/racket tracking errors from the controller CSV."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path


DEFAULT_CSV = Path(
    "/home/woan/HumanoidProject/unitree_rl_lab/deploy/robots/g1_23dof_pingpong/logs/hit_error_trace.csv"
)


def read_rows(path: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    if not path.exists():
        return out
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                out.setdefault(key, [])
                try:
                    out[key].append(float(value))
                except (TypeError, ValueError):
                    out[key].append(float("nan"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--window", type=float, default=20.0, help="seconds of controller time to show; <=0 shows all")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    fig, (ax_norm, ax_xyz) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    fig.canvas.manager.set_window_title("Pingpong hit tracking error")

    while True:
        data = read_rows(args.csv)
        ax_norm.clear()
        ax_xyz.clear()

        t_all = data.get("controller_t", [])
        if t_all and args.window > 0.0:
            t_min = max(t_all) - args.window
            idx = [i for i, x in enumerate(t_all) if x >= t_min]
        else:
            idx = list(range(len(t_all)))

        def series(name: str) -> list[float]:
            values = data.get(name, [])
            return [values[i] for i in idx if i < len(values)]

        t = [t_all[i] for i in idx]
        if t:
            ax_norm.plot(t, series("racket_err_norm"), label="racket -> p_hit norm", color="tab:red", linewidth=1.8)
            ax_norm.plot(t, series("ball_hit_err_norm"), label="ball projected-to-hit -> p_hit norm", color="tab:blue", linewidth=1.5)
            ax_norm.plot(t, series("ball_now_err_norm"), label="ball now -> p_hit norm", color="tab:purple", linewidth=1.2, alpha=0.85)
            ax_norm.set_ylabel("error norm (m)")
            ax_norm.grid(True, alpha=0.3)
            ax_norm.legend(loc="upper right")

            ax_xyz.plot(t, series("racket_err_x"), label="racket err x", color="tab:red")
            ax_xyz.plot(t, series("racket_err_y"), label="racket err y", color="tab:green")
            ax_xyz.plot(t, series("racket_err_z"), label="racket err z", color="tab:blue")
            ax_xyz.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
            ax_xyz.set_xlabel("controller time in Pingpong (s)")
            ax_xyz.set_ylabel("racket - p_hit (m)")
            ax_xyz.grid(True, alpha=0.3)
            ax_xyz.legend(loc="upper right")
            latest = len(t) - 1
            ax_norm.set_title(
                f"{args.csv} | latest t_to_hit={series('t_to_hit')[latest]:.3f}s, "
                f"racket={series('racket_err_norm')[latest]:.3f}m, "
                f"ball_now={series('ball_now_err_norm')[latest]:.3f}m"
            )
        else:
            ax_norm.set_title(f"Waiting for CSV: {args.csv}")
            ax_norm.grid(True, alpha=0.3)
            ax_xyz.grid(True, alpha=0.3)

        plt.pause(0.001)
        if args.once:
            break
        time.sleep(max(args.interval, 0.05))


if __name__ == "__main__":
    main()
