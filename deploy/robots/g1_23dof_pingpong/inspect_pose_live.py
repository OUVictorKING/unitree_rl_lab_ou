#!/usr/bin/env python3
"""Live PoseStamped viewer — plot xyz position + RPY orientation in real time.

Use this to sanity-check a mocap stream BEFORE connecting the robot. No
bag recording, just a matplotlib live window showing the latest few seconds.

Usage::

    source /opt/ros/humble/setup.bash

    # single topic
    python3 deploy/robots/g1_23dof_pingpong/inspect_pose_live.py \
        --topic /vrpn_mocap/g1/pose

    # both robot base + ball, side-by-side
    python3 deploy/robots/g1_23dof_pingpong/inspect_pose_live.py \
        --topic /vrpn_mocap/g1/pose \
        --topic /vrpn_mocap/U_Tracker0/pose

    # different rolling window
    python3 deploy/robots/g1_23dof_pingpong/inspect_pose_live.py \
        --topic /vrpn_mocap/g1/pose --window-s 30

For each --topic the figure adds a column of two subplots:
    upper:   x / y / z              (m,    in topic frame)
    lower:   roll / pitch / yaw     (deg,  ZYX intrinsic Tait-Bryan, REP-103)

Quaternion → RPY conversion is done inline (no tf_transformations dependency)
so this works in any ROS2 Python install. The `xyzw` order from
`geometry_msgs/Pose.orientation` matches the formulas below.
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from collections import deque
from typing import Any


def _ensure_ros2_sourced():
    try:
        import rclpy  # noqa: F401
        from geometry_msgs.msg import PoseStamped  # noqa: F401
    except ImportError as exc:
        sys.exit(
            f"[inspect_pose_live] {exc}\n"
            "  ROS2 isn't sourced into this Python.\n"
            "  Run first:  source /opt/ros/humble/setup.bash\n"
            "  Then:       python3 deploy/robots/g1_23dof_pingpong/inspect_pose_live.py --topic <topic>"
        )


def quat_to_rpy(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float]:
    """Convert quaternion (xyzw, ROS convention) to Tait-Bryan ZYX intrinsic
    Euler angles (REP-103: roll about X, pitch about Y, yaw about Z).
    Returns (roll, pitch, yaw) in radians.

    Pitch is clamped to (-π/2, +π/2) — gimbal-lock degenerate cases will
    saturate `pitch` at ±90° but `roll`/`yaw` may swap sign rapidly there.
    """
    # roll (X axis)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # pitch (Y axis)
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = max(-1.0, min(1.0, sinp))   # clamp to safe asin domain
    pitch = math.asin(sinp)
    # yaw (Z axis)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def quat_mean_hemisphere_aligned(quats):
    """Hemisphere-aligned arithmetic mean of unit quaternions (xyzw order).

    Mirrors the C++ helper in State_Pingpong.cpp so the live plot matches
    what the controller's actor obs sees. Each input is a 4-tuple (qx, qy,
    qz, qw); output is the same form.

    The exact statistical mean is the leading eigenvector of M = Σ qᵢ qᵢᵀ
    (4×4 PSD); when intra-window angular spread is small it is well-
    approximated by hemisphere-aligning each qᵢ against the first sample
    (flip sign if dot < 0 — same rotation, double-cover sign flip), then
    averaging and renormalizing. At 110 Hz × 5 ≈ 45 ms the rigid body
    cannot rotate more than a few degrees, so this is essentially identical
    to the eigendecomposition solution.
    """
    if not quats:
        return (0.0, 0.0, 0.0, 1.0)
    qx0, qy0, qz0, qw0 = quats[0]
    sx = sy = sz = sw = 0.0
    for qx, qy, qz, qw in quats:
        dot = qw * qw0 + qx * qx0 + qy * qy0 + qz * qz0
        s = -1.0 if dot < 0.0 else 1.0
        sx += s * qx; sy += s * qy; sz += s * qz; sw += s * qw
    n = len(quats)
    sx /= n; sy /= n; sz /= n; sw /= n
    norm = math.sqrt(sx * sx + sy * sy + sz * sz + sw * sw)
    if norm < 1e-9:
        return (qx0, qy0, qz0, qw0)
    return (sx / norm, sy / norm, sz / norm, sw / norm)


class PoseSubscriber:
    """rclpy node that buffers the latest N seconds of a PoseStamped topic.

    Maintains TWO data streams:
      - raw           : every received sample, exactly as published
      - filtered      : sliding-mean over the last `filter_window` samples
                        (position arithmetic mean + quaternion hemisphere-
                        aligned mean), matching the C++ State_Pingpong
                        base-pose filter so this plot reflects what the
                        actor obs actually sees.
    """

    def __init__(self, node, topic: str, window_s: float, filter_window: int = 5,
                 frame_filter: str | None = None):
        from geometry_msgs.msg import PoseStamped
        from rclpy.qos import qos_profile_sensor_data
        self.node = node
        self.topic = topic
        self.frame_filter = frame_filter
        self.window_s = float(window_s)
        self.filter_window = max(1, int(filter_window))
        # Single shared buffer; matplotlib reads it under the lock.
        self.lock = threading.Lock()
        # Time-domain plot buffers (trimmed to window_s).
        self.t  = deque()
        # Raw (single-sample) trajectories.
        self.xs = deque(); self.ys = deque(); self.zs = deque()
        self.rs = deque(); self.ps = deque(); self.ws = deque()  # raw roll/pitch/yaw
        # Filtered trajectories (mean over filter_window past samples).
        self.fxs = deque(); self.fys = deque(); self.fzs = deque()
        self.frs = deque(); self.fps = deque(); self.fws = deque()
        # Internal sliding-window deques used to compute the filter mean.
        # Capped at filter_window — popleft when full.
        self._win_xyz = deque(maxlen=self.filter_window)
        self._win_quat = deque(maxlen=self.filter_window)   # tuples (qx,qy,qz,qw)
        self._t0: float | None = None
        self._n = 0
        self._n_bad_frame = 0
        # VRPN-mocap (and most high-rate sensor publishers) use BEST_EFFORT
        # reliability — RELIABLE subscribers don't match. SensorDataQoS profile
        # = BEST_EFFORT + KEEP_LAST(5) + VOLATILE, the right defaults for live
        # mocap streaming.
        self._sub = node.create_subscription(PoseStamped, topic, self._cb, qos_profile_sensor_data)
        node.get_logger().info(
            f"subscribed to {topic} (PoseStamped, sensor_data QoS), "
            f"window={window_s}s filter_window={self.filter_window}"
        )

    def _stamp_seconds(self, stamp) -> float:
        return float(stamp.sec) + 1.0e-9 * float(stamp.nanosec)

    def _cb(self, msg) -> None:
        if self.frame_filter and msg.header.frame_id and self.frame_filter != msg.header.frame_id:
            self._n_bad_frame += 1
            return

        t = self._stamp_seconds(msg.header.stamp)
        # If publisher didn't fill stamp (zero), fall back to wall-clock so the
        # plot still progresses — but warn once.
        if t == 0.0:
            t = time.time()
            if self._n == 0:
                self.node.get_logger().warn(
                    "msg.header.stamp is zero; falling back to wall-clock. "
                    "Mocap publisher should fill stamp = node->now()."
                )

        if self._t0 is None:
            self._t0 = t
        t_rel = t - self._t0

        p = msg.pose.position
        q = msg.pose.orientation
        # Raw (single sample) values.
        roll, pitch, yaw = quat_to_rpy(q.x, q.y, q.z, q.w)

        # Push into the sliding filter window, then compute mean. matches the
        # C++ base filter exactly (sample-count window, not time window).
        self._win_xyz.append((p.x, p.y, p.z))
        self._win_quat.append((q.x, q.y, q.z, q.w))
        n = len(self._win_xyz)
        fx = sum(v[0] for v in self._win_xyz) / n
        fy = sum(v[1] for v in self._win_xyz) / n
        fz = sum(v[2] for v in self._win_xyz) / n
        fqx, fqy, fqz, fqw = quat_mean_hemisphere_aligned(self._win_quat)
        froll, fpitch, fyaw = quat_to_rpy(fqx, fqy, fqz, fqw)

        with self.lock:
            self.t.append(t_rel)
            self.xs.append(p.x);  self.ys.append(p.y);  self.zs.append(p.z)
            self.rs.append(math.degrees(roll))
            self.ps.append(math.degrees(pitch))
            self.ws.append(math.degrees(yaw))
            self.fxs.append(fx); self.fys.append(fy); self.fzs.append(fz)
            self.frs.append(math.degrees(froll))
            self.fps.append(math.degrees(fpitch))
            self.fws.append(math.degrees(fyaw))
            # Trim anything older than window_s
            t_cut = t_rel - self.window_s
            while self.t and self.t[0] < t_cut:
                self.t.popleft()
                self.xs.popleft();  self.ys.popleft();  self.zs.popleft()
                self.rs.popleft();  self.ps.popleft();  self.ws.popleft()
                self.fxs.popleft(); self.fys.popleft(); self.fzs.popleft()
                self.frs.popleft(); self.fps.popleft(); self.fws.popleft()
        self._n += 1

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "t":  list(self.t),
                "x":  list(self.xs),  "y":  list(self.ys),  "z":  list(self.zs),
                "r":  list(self.rs),  "p":  list(self.ps),  "w":  list(self.ws),
                "fx": list(self.fxs), "fy": list(self.fys), "fz": list(self.fzs),
                "fr": list(self.frs), "fp": list(self.fps), "fw": list(self.fws),
                "n":  self._n,
                "n_bad_frame": self._n_bad_frame,
                "filter_window": self.filter_window,
            }


def run_live_plot(subs: list["PoseSubscriber"], refresh_hz: float) -> None:
    import matplotlib
    for backend in ("TkAgg", "QtAgg", "Qt5Agg", "GTK3Agg"):
        try:
            matplotlib.use(backend, force=True)
            break
        except Exception:  # noqa: BLE001
            continue
    import matplotlib.pyplot as plt
    plt.ion()

    n = len(subs)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 6), sharex="col", squeeze=False)
    try:
        title_txt = ", ".join(s.topic for s in subs)
        fig.canvas.manager.set_window_title(f"PoseStamped live — {title_txt}")
    except Exception:  # noqa: BLE001
        pass

    # One column per topic. Two rows: pos (top), rpy (bottom). For each
    # data channel we plot raw as a thin solid line and the sliding-mean
    # filter as a thicker dashed line — the filter line is what the C++
    # actor obs uses, the raw line shows mocap-side jitter.
    line_handles: list[dict] = []
    for ci, sub in enumerate(subs):
        ax_p = axes[0, ci]
        ax_r = axes[1, ci]
        h = {}
        # raw position (thin solid)
        h["x"],  = ax_p.plot([], [], color="tab:red",   label="x raw",  linewidth=0.8, alpha=0.6)
        h["y"],  = ax_p.plot([], [], color="tab:green", label="y raw",  linewidth=0.8, alpha=0.6)
        h["z"],  = ax_p.plot([], [], color="tab:blue",  label="z raw",  linewidth=0.8, alpha=0.6)
        # filtered position (thick dashed)
        h["fx"], = ax_p.plot([], [], color="tab:red",   label="x filt", linewidth=1.6, linestyle="--")
        h["fy"], = ax_p.plot([], [], color="tab:green", label="y filt", linewidth=1.6, linestyle="--")
        h["fz"], = ax_p.plot([], [], color="tab:blue",  label="z filt", linewidth=1.6, linestyle="--")
        ax_p.set_ylabel("position (m)")
        ax_p.grid(True, alpha=0.3); ax_p.legend(loc="upper left", fontsize=7, ncol=2)
        # raw rpy (thin solid)
        h["roll"],   = ax_r.plot([], [], color="tab:red",   label="roll raw",  linewidth=0.8, alpha=0.6)
        h["pitch"],  = ax_r.plot([], [], color="tab:green", label="pitch raw", linewidth=0.8, alpha=0.6)
        h["yaw"],    = ax_r.plot([], [], color="tab:blue",  label="yaw raw",   linewidth=0.8, alpha=0.6)
        # filtered rpy (thick dashed)
        h["froll"],  = ax_r.plot([], [], color="tab:red",   label="roll filt",  linewidth=1.6, linestyle="--")
        h["fpitch"], = ax_r.plot([], [], color="tab:green", label="pitch filt", linewidth=1.6, linestyle="--")
        h["fyaw"],   = ax_r.plot([], [], color="tab:blue",  label="yaw filt",   linewidth=1.6, linestyle="--")
        ax_r.set_ylabel("orientation (deg)")
        ax_r.set_xlabel("time since first message (s)")
        ax_r.grid(True, alpha=0.3); ax_r.legend(loc="upper left", fontsize=7, ncol=2)
        for y in (-180, -90, 0, 90, 180):
            ax_r.axhline(y, color="k", linewidth=0.3, linestyle=":", alpha=0.4)
        h["title"] = ax_p.set_title(f"{sub.topic}  (waiting for messages…)", fontsize=9)
        line_handles.append(h)

    fig.suptitle("PoseStamped live viewer  —  raw (thin) vs sliding-mean filter (dashed)")
    fig.tight_layout()
    plt.show(block=False)
    plt.pause(0.2)

    period = 1.0 / max(refresh_hz, 1.0)
    last_log = 0.0
    while plt.fignum_exists(fig.number):
        for ci, sub in enumerate(subs):
            snap = sub.snapshot()
            t = snap["t"]
            h = line_handles[ci]
            if t:
                h["x"].set_data(t, snap["x"])
                h["y"].set_data(t, snap["y"])
                h["z"].set_data(t, snap["z"])
                h["fx"].set_data(t, snap["fx"])
                h["fy"].set_data(t, snap["fy"])
                h["fz"].set_data(t, snap["fz"])
                h["roll"].set_data(t, snap["r"])
                h["pitch"].set_data(t, snap["p"])
                h["yaw"].set_data(t, snap["w"])
                h["froll"].set_data(t, snap["fr"])
                h["fpitch"].set_data(t, snap["fp"])
                h["fyaw"].set_data(t, snap["fw"])
                axes[0, ci].relim(); axes[0, ci].autoscale_view(scalex=True, scaley=True)
                axes[1, ci].relim(); axes[1, ci].autoscale_view(scalex=True, scaley=True)
                h["title"].set_text(
                    f"{sub.topic}    msgs={snap['n']:>5d}  filter_window={snap['filter_window']}\n"
                    f"raw  pos=({snap['x'][-1]:+.3f}, {snap['y'][-1]:+.3f}, {snap['z'][-1]:+.3f})  "
                    f"filt pos=({snap['fx'][-1]:+.3f}, {snap['fy'][-1]:+.3f}, {snap['fz'][-1]:+.3f})  "
                    f"raw rpy=({snap['r'][-1]:+.1f}, {snap['p'][-1]:+.1f}, {snap['w'][-1]:+.1f})°  "
                    f"filt rpy=({snap['fr'][-1]:+.1f}, {snap['fp'][-1]:+.1f}, {snap['fw'][-1]:+.1f})°"
                )
            else:
                h["title"].set_text(f"{sub.topic}    (waiting for messages …)")
        try:
            fig.canvas.draw_idle()
            plt.pause(period)
        except Exception:  # noqa: BLE001
            break

        now = time.time()
        if now - last_log > 2.0:
            last_log = now
            for sub in subs:
                snap = sub.snapshot()
                sub.node.get_logger().info(
                    f"  topic={sub.topic}  msgs={snap['n']}  "
                    f"buffer_n={len(snap['t'])}  bad_frame={snap['n_bad_frame']}"
                )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topic", required=True, action="append",
                    help="ROS2 topic name to subscribe (PoseStamped). "
                         "Repeat for multiple topics, e.g. --topic /vrpn_mocap/g1/pose "
                         "--topic /vrpn_mocap/U_Tracker0/pose")
    ap.add_argument("--window-s", type=float, default=10.0,
                    help="rolling window length shown in the plot (default 10 s)")
    ap.add_argument("--filter-window", type=int, default=5,
                    help="sliding-mean filter window in samples (default 5; "
                         "1 disables filtering; matches C++ ros.base_filter_window)")
    ap.add_argument("--refresh-hz", type=float, default=20.0,
                    help="plot redraw rate (default 20 Hz)")
    ap.add_argument("--frame", default=None,
                    help="optional header.frame_id filter — applied to ALL topics")
    args = ap.parse_args()

    _ensure_ros2_sourced()
    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    rclpy.init()
    node = rclpy.create_node("inspect_pose_live")
    subs = [PoseSubscriber(node, t, args.window_s, args.filter_window, args.frame)
            for t in args.topic]

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()

    try:
        run_live_plot(subs, args.refresh_hz)
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
