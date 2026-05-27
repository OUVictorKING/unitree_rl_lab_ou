"""HITTER ball-dynamics fitter & runtime planner (torch math layer).

Reference: HITTER (arXiv:2508.21043v2)
  Sec.IV-A "State Estimation":
    - per-axis 2nd-order polynomial LSQ over the latest 31 frames
    - buffer is cleared at every detected bounce (pre/post never mixed)
  Sec.IV-B "Ball Trajectory Prediction":
    - Eq.1a  a = -k ||v|| v + g                       (no Magnus, no spin)
    - Eq.1b  v+ = C v-,  C = diag(C_h, C_h, -C_v)     (no friction, no spin)
    - Eq.4   pooled LSQ over all trajectories for k, C_h, C_v
       k     = Σ ||a_i - g|| ||v_i||^2  /  Σ ||v_i||^4
       C_h   = Σ (|vx-||vx+| + |vy-||vy+|)  /  Σ (vx-^2 + vy-^2)
       C_v   = Σ |vz-| |vz+|  /  Σ vz-^2

Runtime functions accept torch tensors on any device/dtype.
The offline fit forces CPU + float64 for numerical stability of the
closed-form normal-equation solutions.
"""
from __future__ import annotations

import json
import os
from collections import deque
from datetime import date
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch

# ---------------- paper constants ----------------
GRAVITY = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64)
POLY_WIN = 31      # paper Sec.IV-A: nearest 31 frames
POLY_DEG = 2       # paper Sec.IV-A: 2nd-order polynomial

# ---------------- bounce detection (aligned with filter_to_final_v3.py) ----------------
VZ_THRESH = 0.4
TBL_Z = 0.060
TABLE_HX, TABLE_HY, ON_TABLE_PAD = 1.37, 0.7625, 0.05

# ---------------- offline-fit precision ----------------
_FIT_DEVICE = torch.device("cpu")
_FIT_DTYPE = torch.float64

# ---------------- default I/O paths ----------------
_BASE = "/home/woan/HumanoidProject/unitree_rl_lab/motion_datasets/pingpong/ball_data"
DEFAULT_AIR_DRAG_DIR = os.path.join(_BASE, "final_used_data", "air_drag")
DEFAULT_BOUNCE_DIR = os.path.join(_BASE, "final_used_data", "bounce_coef")
DEFAULT_OUT_DIR = os.path.join(_BASE, "fitted_params")
DEFAULT_PARAMS_PT = os.path.join(DEFAULT_OUT_DIR, "ball_params.pt")
DEFAULT_META_JSON = os.path.join(DEFAULT_OUT_DIR, "fit_metadata.json")


# ===================================================================
# I/O + low-level utilities  (adapted from filter_to_final_v3.py:80-178)
# ===================================================================
def _load_trajectory(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Read a 4-column space-delimited file (t x y z) into torch tensors."""
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None,
                     names=["t", "x", "y", "z"])
    arr = df.to_numpy()
    t = torch.from_numpy(arr[:, 0]).to(_FIT_DEVICE).to(_FIT_DTYPE)
    xyz = torch.from_numpy(arr[:, 1:]).to(_FIT_DEVICE).to(_FIT_DTYPE)
    return t, xyz


def _is_on_table(x: float, y: float) -> bool:
    return abs(x) <= TABLE_HX + ON_TABLE_PAD and abs(y) <= TABLE_HY + ON_TABLE_PAD


def _find_tbl_bounce(t: torch.Tensor, xyz: torch.Tensor) -> Optional[int]:
    """Return index of the (single) on-table bounce, or None if not found.
    Uses the same vz-reversal rule as filter_to_final_v3.py."""
    dt = torch.diff(t).clamp_min(1e-6)
    vz = torch.diff(xyz[:, 2]) / dt
    cands = []
    for i in range(len(vz) - 1):
        if vz[i] < -VZ_THRESH and vz[i + 1] > VZ_THRESH:
            j = i + 1
            zj, xj, yj = float(xyz[j, 2]), float(xyz[j, 0]), float(xyz[j, 1])
            if zj < TBL_Z and _is_on_table(xj, yj):
                cands.append(j)
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    # Multiple bounces shouldn't occur after v3 filtering, but be defensive:
    # take the lowest-z one.
    return min(cands, key=lambda j: float(xyz[j, 2]))


def _smooth_va(
    t: torch.Tensor,
    xyz: torch.Tensor,
    idx: int,
    win_lo: int,
    win_hi: int,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Per-axis 2nd-order polynomial LSQ fit on samples [win_lo, win_hi),
    evaluated at t[idx]. Returns (v[3], a[3]) torch tensors.

    Paper Sec.IV-A: latest 31 frames; pre/post-bounce frames must not be mixed,
    so the caller passes [win_lo, win_hi) restricted to one side of any bounce.

    The window is then truncated to the POLY_WIN samples nearest to idx.
    Returns None if fewer than POLY_DEG+1 (=3) samples are available.
    """
    win_lo = max(0, win_lo)
    win_hi = min(len(t), win_hi)
    # nearest POLY_WIN samples to idx, clipped to [win_lo, win_hi)
    half = POLY_WIN // 2
    lo = max(win_lo, idx - half)
    hi = min(win_hi, lo + POLY_WIN)
    lo = max(win_lo, hi - POLY_WIN)
    if hi - lo < POLY_DEG + 1:
        return None

    tau = (t[lo:hi] - t[idx]).reshape(-1, 1)         # local time, s
    A = torch.cat([torch.ones_like(tau), tau, tau ** 2], dim=1)  # (n,3)
    samples = xyz[lo:hi]                              # (n,3)

    # closed-form normal-equation solve:  (A^T A) c = A^T y
    AtA = A.T @ A
    Aty = A.T @ samples                               # (3,3)
    coef = torch.linalg.solve(AtA, Aty)               # rows: c0,c1,c2

    # at tau=0:  v = c1,  a = 2 * c2
    v = coef[1]
    a = 2.0 * coef[2]
    return v, a


# ===================================================================
# Eq.4 fit-1: drag coefficient  k
# ===================================================================
def fit_drag(air_drag_dir: str = DEFAULT_AIR_DRAG_DIR) -> dict:
    """Pooled LSQ over all in-flight frames: minimise Σ (||a-g|| - k ||v||²)².

    For every file the polynomial window is restricted to one side of any
    detected bounce (Sec.IV-A: buffer cleared at bounce).
    """
    files = sorted(f for f in os.listdir(air_drag_dir) if f.endswith(".txt"))
    ag_norms: list[float] = []     # ||a - g||  per usable frame
    v2s: list[float] = []          # ||v||²     per usable frame

    for name in files:
        t, xyz = _load_trajectory(os.path.join(air_drag_dir, name))
        n = len(t)
        bnc = _find_tbl_bounce(t, xyz)
        sides = [(0, n)] if bnc is None else [(0, bnc + 1), (bnc + 1, n)]
        for lo, hi in sides:
            if hi - lo < POLY_DEG + 1:
                continue
            for i in range(lo, hi):
                va = _smooth_va(t, xyz, i, lo, hi)
                if va is None:
                    continue
                v, a = va
                v2 = float((v * v).sum())
                if v2 < 1e-6:               # essentially static frame
                    continue
                ag_norms.append(float(torch.linalg.vector_norm(a - GRAVITY)))
                v2s.append(v2)

    if not v2s:
        raise RuntimeError("fit_drag: no usable frames")

    ag_t = torch.tensor(ag_norms, dtype=_FIT_DTYPE)
    v2_t = torch.tensor(v2s, dtype=_FIT_DTYPE)
    sum_av2 = float((ag_t * v2_t).sum())
    sum_v4 = float((v2_t * v2_t).sum())
    k = sum_av2 / sum_v4
    rms = float(torch.sqrt(((ag_t - k * v2_t) ** 2).mean()))

    return dict(
        k=k,
        n_frames=len(v2s),
        n_files=len(files),
        residual_rms=rms,           # m/s²
    )


# ===================================================================
# Eq.4 fit-2/3: restitution C_h, C_v
# ===================================================================
def fit_bounce(bounce_dir: str = DEFAULT_BOUNCE_DIR) -> dict:
    """Pooled LSQ for the diagonal restitution matrix.

    For each trajectory, fit one v⁻ at frame `bnc` using only the pre-bounce
    samples [0, bnc+1) and one v⁺ at frame `bnc` using only post-bounce
    samples [bnc+1, n) (paper: buffer cleared at bounce).
    """
    files = sorted(f for f in os.listdir(bounce_dir) if f.endswith(".txt"))
    samples = []         # list of (name, vminus[3], vplus[3])
    skipped = []

    for name in files:
        t, xyz = _load_trajectory(os.path.join(bounce_dir, name))
        n = len(t)
        bnc = _find_tbl_bounce(t, xyz)
        if bnc is None:
            skipped.append((name, "no on-table bounce"))
            continue

        # pre-side window: [0, bnc+1), evaluated at frame bnc
        va_pre = _smooth_va(t, xyz, bnc, 0, bnc + 1)
        # post-side window: [bnc, n), evaluated at frame bnc as well so we
        # measure v⁺ at the bounce instant (smoothed by post-side data only).
        va_post = _smooth_va(t, xyz, bnc, bnc, n)
        if va_pre is None or va_post is None:
            skipped.append((name, f"insufficient frames (pre={bnc+1}, post={n-bnc})"))
            continue
        v_minus = va_pre[0]
        v_plus = va_post[0]

        # paper enforces vz⁻<0, vz⁺>0; otherwise the bounce is malformed
        if float(v_minus[2]) >= 0.0 or float(v_plus[2]) <= 0.0:
            skipped.append((name, f"bad vz signs (vz-={float(v_minus[2]):.3f},vz+={float(v_plus[2]):.3f})"))
            continue
        samples.append((name, v_minus, v_plus))

    if not samples:
        raise RuntimeError("fit_bounce: no usable bounces")

    # closed-form pooled LSQ
    sum_xy_num = 0.0     # Σ (|vx-||vx+| + |vy-||vy+|)
    sum_xy_den = 0.0     # Σ (vx-² + vy-²)
    sum_z_num = 0.0      # Σ |vz-||vz+|
    sum_z_den = 0.0      # Σ vz-²
    for _, vm, vp in samples:
        vmx, vmy, vmz = float(vm[0]), float(vm[1]), float(vm[2])
        vpx, vpy, vpz = float(vp[0]), float(vp[1]), float(vp[2])
        sum_xy_num += abs(vmx) * abs(vpx) + abs(vmy) * abs(vpy)
        sum_xy_den += vmx * vmx + vmy * vmy
        sum_z_num += abs(vmz) * abs(vpz)
        sum_z_den += vmz * vmz
    Ch = sum_xy_num / sum_xy_den
    Cv = sum_z_num / sum_z_den

    # residuals
    sq_xy, n_xy = 0.0, 0
    sq_z, n_z = 0.0, 0
    per_traj = []
    for name, vm, vp in samples:
        vmx, vmy, vmz = float(vm[0]), float(vm[1]), float(vm[2])
        vpx, vpy, vpz = float(vp[0]), float(vp[1]), float(vp[2])
        sq_xy += (abs(vpx) - Ch * abs(vmx)) ** 2
        sq_xy += (abs(vpy) - Ch * abs(vmy)) ** 2
        n_xy += 2
        sq_z += (abs(vpz) - Cv * abs(vmz)) ** 2
        n_z += 1
        v_minus_norm = (vmx * vmx + vmy * vmy + vmz * vmz) ** 0.5
        v_plus_norm = (vpx * vpx + vpy * vpy + vpz * vpz) ** 0.5
        # per-traj individual ratios for sanity-check
        Ch_i = (abs(vpx) + abs(vpy)) / max(abs(vmx) + abs(vmy), 1e-9)
        Cv_i = abs(vpz) / max(abs(vmz), 1e-9)
        per_traj.append(dict(
            file=name,
            v_minus=[vmx, vmy, vmz],
            v_plus=[vpx, vpy, vpz],
            v_minus_norm=v_minus_norm,
            v_plus_norm=v_plus_norm,
            Ch_i=Ch_i,
            Cv_i=Cv_i,
        ))
    rms_xy = (sq_xy / max(n_xy, 1)) ** 0.5
    rms_z = (sq_z / max(n_z, 1)) ** 0.5

    return dict(
        Ch=Ch,
        Cv=Cv,
        n_traj=len(samples),
        n_files=len(files),
        n_skipped=len(skipped),
        skipped=skipped,
        residual_rms_xy=rms_xy,
        residual_rms_z=rms_z,
        per_traj=per_traj,
    )


# ===================================================================
# Top-level fit + persistence
# ===================================================================
def fit_all(
    air_drag_dir: str = DEFAULT_AIR_DRAG_DIR,
    bounce_dir: str = DEFAULT_BOUNCE_DIR,
    out_dir: str = DEFAULT_OUT_DIR,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    drag = fit_drag(air_drag_dir)
    bnc = fit_bounce(bounce_dir)

    pt_path = os.path.join(out_dir, "ball_params.pt")
    json_path = os.path.join(out_dir, "fit_metadata.json")

    torch.save(
        dict(
            k=torch.tensor(drag["k"], dtype=_FIT_DTYPE),
            Ch=torch.tensor(bnc["Ch"], dtype=_FIT_DTYPE),
            Cv=torch.tensor(bnc["Cv"], dtype=_FIT_DTYPE),
        ),
        pt_path,
    )

    meta = dict(
        paper="HITTER, arXiv:2508.21043v2",
        equations=dict(
            flight="a = -k ||v|| v + g  (Eq.1a)",
            bounce="v+ = diag(Ch, Ch, -Cv) v-  (Eq.1b/2)",
            losses=dict(
                k="min Σ (||a-g|| - k ||v||²)²  (Eq.4)",
                Ch="min Σ ((|vx+|-Ch|vx-|)² + (|vy+|-Ch|vy-|)²)  (Eq.4)",
                Cv="min Σ (|vz+|-Cv|vz-|)²  (Eq.4)",
            ),
        ),
        method=dict(
            v_a_estimator=f"per-axis {POLY_DEG}nd-order polynomial LSQ over latest {POLY_WIN} frames; "
                          "buffer cleared at bounce",
            solver="closed-form normal-equation",
            offline_dtype="float64",
        ),
        params=dict(k=drag["k"], Ch=bnc["Ch"], Cv=bnc["Cv"]),
        residuals=dict(
            drag_rms_m_per_s2=drag["residual_rms"],
            bounce_rms_xy_m_per_s=bnc["residual_rms_xy"],
            bounce_rms_z_m_per_s=bnc["residual_rms_z"],
        ),
        data=dict(
            air_drag_dir=air_drag_dir,
            bounce_dir=bounce_dir,
            n_drag_files=drag["n_files"],
            n_drag_frames=drag["n_frames"],
            n_bounce_files=bnc["n_files"],
            n_bounce_traj=bnc["n_traj"],
            n_bounce_skipped=bnc["n_skipped"],
            bounce_skipped=bnc["skipped"],
        ),
        per_traj=bnc["per_traj"],
        fit_date=str(date.today()),
    )
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    return dict(drag=drag, bnc=bnc, pt_path=pt_path, json_path=json_path, meta=meta)


def load_ball_params(
    path: Optional[str] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> dict:
    """Load fitted (k, Ch, Cv) for runtime use. Returns scalar tensors."""
    p = path or DEFAULT_PARAMS_PT
    raw = torch.load(p, map_location=device or "cpu")
    out = {}
    for key, val in raw.items():
        v = val
        if dtype is not None:
            v = v.to(dtype)
        if device is not None:
            v = v.to(device)
        out[key] = v
    return out


# ===================================================================
# Runtime: state estimator (Sec.IV-A)
# ===================================================================
def estimate_state(
    t_buf: torch.Tensor,
    xyz_buf: torch.Tensor,
    last_bounce_idx: int = -1,
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Per-axis 2nd-order polynomial LSQ over the latest 31 frames since the
    most recent bounce (paper Sec.IV-A: buffer cleared at bounce). Evaluated
    at the last buffer frame.

    Returns (p, v, a) at the latest frame, or None if too few post-bounce
    samples (< POLY_DEG+1 = 3 frames).
    """
    n = len(t_buf)
    if n == 0:
        return None
    win_lo = (last_bounce_idx + 1) if last_bounce_idx is not None and last_bounce_idx >= 0 else 0
    win_hi = n
    idx = n - 1
    va = _smooth_va(t_buf, xyz_buf, idx, win_lo, win_hi)
    if va is None:
        return None
    v, a = va
    p = xyz_buf[idx]
    return p, v, a


# ===================================================================
# Runtime: ball-trajectory prediction (Sec.IV-B)
# ===================================================================
def _integrate(
    p0: np.ndarray,
    v0: np.ndarray,
    *,
    k: float,
    Ch: float,
    Cv: float,
    dt: float = 0.002,
    max_t: float = 2.0,
    max_bounces: int = 1,
    table_z: float = 0.0,
    table_hx: float = TABLE_HX,
    table_hy: float = TABLE_HY,
    on_table_pad: float = ON_TABLE_PAD,
    stop_predicate: Optional[Callable] = None,
) -> dict:
    """RK2 (midpoint) integrator with up to ``max_bounces`` Eq.1b table bounces.

    The integrator runs in numpy float64 for numerical stability. Termination is
    governed by ``stop_predicate(p_cur, v_cur, p_next, v_next) -> (alpha, p_hit,
    v_hit) | None``, where ``alpha`` is the sub-step fraction in [0, 1] used to
    interpolate the hit time. If the predicate never fires within ``max_t`` the
    return dict has ``hit_pos = None``.

    Returns numpy-typed dict (callers convert to torch as needed):
      hit_pos:  np.ndarray[3] | None
      hit_vel:  np.ndarray[3] | None
      t_to_hit: float | None
      bounces:  int
      traj_t:   np.ndarray[N]
      traj_p:   np.ndarray[N, 3]
    """
    g = np.array([0.0, 0.0, -9.81], dtype=np.float64)
    p_cur = p0.astype(np.float64).copy()
    v_cur = v0.astype(np.float64).copy()
    t = 0.0
    bounces_done = 0
    traj_t = [0.0]
    traj_p = [p_cur.copy()]

    # already-at-stop fast path
    if stop_predicate is not None:
        early = stop_predicate(p_cur, v_cur, p_cur, v_cur)
        if early is not None:
            _, p_hit, v_hit = early
            return dict(
                hit_pos=p_hit.copy(), hit_vel=v_hit.copy(), t_to_hit=0.0,
                bounces=0, traj_t=np.asarray(traj_t), traj_p=np.stack(traj_p),
            )

    while t < max_t:
        v_norm = float(np.linalg.norm(v_cur))
        a_cur = -k * v_norm * v_cur + g
        v_half = v_cur + 0.5 * dt * a_cur
        v_half_norm = float(np.linalg.norm(v_half))
        a_half = -k * v_half_norm * v_half + g
        p_next = p_cur + dt * v_half
        v_next = v_cur + dt * a_half

        # bounce check: z crossing table_z from above, on-table
        if (bounces_done < max_bounces
                and p_cur[2] > table_z and p_next[2] <= table_z):
            on_tbl = (abs(p_next[0]) <= table_hx + on_table_pad
                      and abs(p_next[1]) <= table_hy + on_table_pad)
            if on_tbl:
                denom = max(p_cur[2] - p_next[2], 1e-9)
                alpha = (p_cur[2] - table_z) / denom
                p_b = p_cur + alpha * (p_next - p_cur)
                v_b = v_cur + alpha * (v_next - v_cur)
                # Eq.1b: v+ = diag(Ch, Ch, -Cv) v-
                v_post = np.array([Ch * v_b[0], Ch * v_b[1], -Cv * v_b[2]],
                                  dtype=np.float64)
                rest = (1.0 - alpha) * dt
                t_bounce = t + alpha * dt
                # advance the rest of dt drag-free (sub-step is tiny, ≪ 1 ms)
                p_cur = p_b + v_post * rest
                v_cur = v_post
                t = t_bounce + rest
                bounces_done += 1
                traj_t.append(t_bounce); traj_p.append(p_b.copy())
                traj_t.append(t); traj_p.append(p_cur.copy())
                continue

        # stop predicate evaluated on (p_cur, v_cur) -> (p_next, v_next)
        if stop_predicate is not None:
            stop_event = stop_predicate(p_cur, v_cur, p_next, v_next)
            if stop_event is not None:
                alpha, p_hit, v_hit = stop_event
                t_hit = t + alpha * dt
                traj_t.append(t_hit); traj_p.append(p_hit.copy())
                return dict(
                    hit_pos=p_hit.copy(), hit_vel=v_hit.copy(),
                    t_to_hit=float(t_hit), bounces=bounces_done,
                    traj_t=np.asarray(traj_t), traj_p=np.stack(traj_p),
                )

        p_cur = p_next
        v_cur = v_next
        t += dt
        traj_t.append(t); traj_p.append(p_cur.copy())

    # never matched
    return dict(
        hit_pos=None, hit_vel=None, t_to_hit=None,
        bounces=bounces_done,
        traj_t=np.asarray(traj_t), traj_p=np.stack(traj_p),
    )


def _make_x_plane_stop(x_hit: float):
    """Stop when ball x-coord crosses ``x_hit`` from the +x side."""
    def stop(p_cur, v_cur, p_next, v_next):
        if p_cur[0] > x_hit >= p_next[0]:
            denom = max(p_cur[0] - p_next[0], 1e-9)
            alpha = (p_cur[0] - x_hit) / denom
            p_hit = p_cur + alpha * (p_next - p_cur)
            v_hit = v_cur + alpha * (v_next - v_cur)
            return alpha, p_hit, v_hit
        return None
    return stop


def _make_shell_stop(base_pos: np.ndarray, r_min: float, r_max: float, z_min: float):
    """Stop on first frame inside spherical shell around ``base_pos`` with
    z ≥ z_min (i.e. above the lowest acceptable hit height)."""
    def stop(p_cur, v_cur, p_next, v_next):
        d_next = float(np.linalg.norm(p_next - base_pos))
        if r_min <= d_next <= r_max and p_next[2] >= z_min:
            return 1.0, p_next.copy(), v_next.copy()
        return None
    return stop


def predict_hit_plane(
    p: torch.Tensor,
    v: torch.Tensor,
    k: float,
    Ch: float,
    Cv: float,
    x_hit: float,
    table_z: float = 0.0,
    table_hx: float = TABLE_HX,
    table_hy: float = TABLE_HY,
    on_table_pad: float = ON_TABLE_PAD,
    dt: float = 0.002,
    max_t: float = 2.0,
    *,
    max_bounces: int = 3,
    allow_bounce: Optional[bool] = None,
) -> dict:
    """Forward-roll Eq.1a (drag) with up to ``max_bounces`` Eq.1b table bounces
    until the ball's x-coordinate crosses ``x_hit`` (paper Sec.IV-B). RK2
    integration; bounces are detected by z crossing ``table_z`` from above
    while the landing is on the table.

    The legacy boolean ``allow_bounce`` is retained for backward compatibility:
    when not None it overrides ``max_bounces`` (True → 1, False → 0).

    Returns dict (torch tensors on the input device/dtype):
      hit_pos:  Tensor[3] | None    ball pos when x = x_hit
      hit_vel:  Tensor[3] | None    ball vel at hit
      t_to_hit: float | None         elapsed integration time
      bounced:  bool                 whether ≥1 bounce occurred
      bounces:  int                  number of bounces actually applied
      traj_t:   Tensor[N]            predicted-path times (visualization)
      traj_p:   Tensor[N,3]          predicted-path positions
    """
    if allow_bounce is not None:
        max_bounces = 1 if allow_bounce else 0

    dtype = p.dtype
    device = p.device
    p0 = p.detach().cpu().numpy().astype(np.float64)
    v0 = v.detach().cpu().numpy().astype(np.float64)

    def _to_torch(arr):
        return torch.from_numpy(np.asarray(arr, dtype=np.float64)).to(dtype).to(device)

    # already past the hit plane → nothing to forward-roll
    if p0[0] <= x_hit:
        return dict(
            hit_pos=_to_torch(p0), hit_vel=_to_torch(v0), t_to_hit=0.0,
            bounced=False, bounces=0,
            traj_t=_to_torch(np.asarray([0.0])),
            traj_p=_to_torch(np.stack([p0])),
        )

    res = _integrate(
        p0, v0, k=k, Ch=Ch, Cv=Cv, dt=dt, max_t=max_t,
        max_bounces=max_bounces,
        table_z=table_z, table_hx=table_hx, table_hy=table_hy,
        on_table_pad=on_table_pad,
        stop_predicate=_make_x_plane_stop(x_hit),
    )
    return dict(
        hit_pos=_to_torch(res["hit_pos"]) if res["hit_pos"] is not None else None,
        hit_vel=_to_torch(res["hit_vel"]) if res["hit_vel"] is not None else None,
        t_to_hit=res["t_to_hit"],
        bounced=res["bounces"] > 0,
        bounces=res["bounces"],
        traj_t=_to_torch(res["traj_t"]),
        traj_p=_to_torch(res["traj_p"]),
    )


def _predict_sphere(
    p: torch.Tensor,
    v: torch.Tensor,
    k: float,
    Ch: float,
    Cv: float,
    base_pos: torch.Tensor,
    r_min: float,
    r_max: float,
    z_min: float,
    table_z: float = 0.0,
    table_hx: float = TABLE_HX,
    table_hy: float = TABLE_HY,
    on_table_pad: float = ON_TABLE_PAD,
    dt: float = 0.002,
    max_t: float = 2.0,
    max_bounces: int = 3,
) -> dict:
    """Forward-roll until the ball enters the spherical shell
    [r_min, r_max] around ``base_pos`` with z ≥ ``z_min``. Same return shape
    as :func:`predict_hit_plane`."""
    dtype = p.dtype
    device = p.device
    p0 = p.detach().cpu().numpy().astype(np.float64)
    v0 = v.detach().cpu().numpy().astype(np.float64)
    bp = base_pos.detach().cpu().numpy().astype(np.float64).reshape(3)

    def _to_torch(arr):
        return torch.from_numpy(np.asarray(arr, dtype=np.float64)).to(dtype).to(device)

    res = _integrate(
        p0, v0, k=k, Ch=Ch, Cv=Cv, dt=dt, max_t=max_t,
        max_bounces=max_bounces,
        table_z=table_z, table_hx=table_hx, table_hy=table_hy,
        on_table_pad=on_table_pad,
        stop_predicate=_make_shell_stop(bp, r_min, r_max, z_min),
    )
    return dict(
        hit_pos=_to_torch(res["hit_pos"]) if res["hit_pos"] is not None else None,
        hit_vel=_to_torch(res["hit_vel"]) if res["hit_vel"] is not None else None,
        t_to_hit=res["t_to_hit"],
        bounced=res["bounces"] > 0,
        bounces=res["bounces"],
        traj_t=_to_torch(res["traj_t"]),
        traj_p=_to_torch(res["traj_p"]),
    )


def solve_paddle_target(
    p_hit: torch.Tensor,
    v_ball_in: torch.Tensor,
    target_land: Optional[torch.Tensor] = None,
    flight_time: float = 0.45,
    paddle_cor: float = 0.85,
    g: float = 9.81,
) -> dict:
    """Solve for the desired paddle pose and velocity at impact.

    Paddle is modeled as a rigid frictionless plane that moves only along
    its own normal at the moment of contact (paper Sec.IV-C, simplified).
    The ball's tangential velocity is preserved; the normal component
    follows Newton's restitution law:

        v_out_n = (1+e) v_paddle_n - e v_in_n
      => v_paddle_n = (v_out_n + e v_in_n) / (1+e)

    The paddle normal n is chosen as the unit vector of (v_out - v_in),
    which is the only direction consistent with the frictionless model
    (i.e. the tangential constraint v_out_t = v_in_t is automatically met).

    The post-impact ball velocity v_out is solved from a drag-free
    ballistic to ``target_land`` over ``flight_time``:

        target_land = p_hit + v_out * T + 0.5 * (0,0,-g) * T²
      => v_out = (target_land - p_hit) / T + (0, 0, 0.5*g*T)

    Args:
      p_hit:        ball pos at hit (predict_hit_plane.hit_pos).
      v_ball_in:    ball vel at hit (predict_hit_plane.hit_vel).
      target_land:  desired ball-landing point on opponent's half-table.
                    Default = (+0.7, 0.0, 0.06) — opponent center, table-top z.
      flight_time:  ball flight time from impact to landing (s).
      paddle_cor:   paddle restitution e (~0.85 for rubber on rubber ball).

    Returns dict (torch tensors on the same device/dtype as ``p_hit``):
      paddle_normal:  unit vector — the direction the paddle face points
      v_paddle:       desired paddle COM velocity at impact (the planner output)
      v_ball_out:     post-impact ball velocity (sanity check)
      target_land:    landing target actually used
    """
    dtype = p_hit.dtype
    device = p_hit.device
    p = p_hit.detach().cpu().numpy().astype(np.float64)
    v_in = v_ball_in.detach().cpu().numpy().astype(np.float64)

    if target_land is None:
        tl = np.array([0.7, 0.0, 0.06], dtype=np.float64)
    else:
        tl = target_land.detach().cpu().numpy().astype(np.float64)

    T = float(flight_time)
    # ballistic with gravity only (no drag during return): solve v_out
    v_out = (tl - p) / T + np.array([0.0, 0.0, 0.5 * g * T], dtype=np.float64)

    delta_v = v_out - v_in
    n_norm = float(np.linalg.norm(delta_v))

    def _to_torch(arr):
        return torch.from_numpy(np.asarray(arr, dtype=np.float64)).to(dtype).to(device)

    if n_norm < 1e-9:
        return dict(
            paddle_normal=None, v_paddle=None,
            v_ball_out=_to_torch(v_out), target_land=_to_torch(tl),
        )

    n = delta_v / n_norm
    e = float(paddle_cor)
    v_in_n = float(np.dot(v_in, n))
    v_out_n = float(np.dot(v_out, n))
    v_pad_n = (v_out_n + e * v_in_n) / (1.0 + e)
    v_paddle = v_pad_n * n

    return dict(
        paddle_normal=_to_torch(n),
        v_paddle=_to_torch(v_paddle),
        v_ball_out=_to_torch(v_out),
        target_land=_to_torch(tl),
    )


# ===================================================================
# Runtime: HitterPlanner — workspace-adaptive hit-plane + hold-last-safe
# ===================================================================
class HitterPlanner:
    """Per-tick HITTER planner.

    update(t, ball_pos, base_pos, base_quat) -> packet | None

    Ladder per call:
      1. push (t, ball_pos) into rolling 31-frame buffer
      2. early-exit gates  → ``_hold_or_none(reason)`` (last-safe cache or None)
      3. ``_find_hit_point``: default plane → forward/backward shift along
         ``x_hit`` to recover from z violations; ``y_far`` / no recovery → fail
      4. ``solve_paddle_target`` (paper Eq.5+Eq.6)
      5. cache fresh packet, return

    The class does NOT reason about base-vs-table clearance — that is the WBC's
    job. It only enforces *paddle vs. workspace* (z_min/z_max for the paddle
    above the table, |hit_y - base_y| ≤ y_max_dev for arm reach in y). When the
    ball would land outside |y - base_y|, the planner gives up and lets the
    next tick's state estimate try again.
    """

    def __init__(
        self,
        params_path: Optional[str] = None,
        # --- workspace ---
        x_hit_default: float = -1.50,
        z_min_world: float = 0.10,
        z_max_world: float = 0.60,
        y_max_dev: float = 1.00,
        shift_step: float = 0.02,
        shift_max_forward: float = 0.10,    # toward the table (less negative x)
        shift_max_backward: float = 0.20,   # away from the table (more negative)
        # --- buffer ---
        buffer_size: int = POLY_WIN,
        min_frames_for_predict: int = 10,
        swing_lock_frames: int = 50,
        max_bounces_in_flight: int = 3,
        # --- table ---
        table_z: float = 0.0,
        # --- paddle target (paper Sec.IV-C inputs) ---
        target_land: tuple = (0.70, 0.00, 0.06),
        flight_time: float = 0.45,
        paddle_cor: float = 0.85,
        # --- integration ---
        dt_pred: float = 0.002,
        max_t_pred: float = 2.0,
        # --- runtime tensors ---
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ):
        # workspace + scheduling knobs
        self.x_hit_default = float(x_hit_default)
        self.z_min_world = float(z_min_world)
        self.z_max_world = float(z_max_world)
        self.y_max_dev = float(y_max_dev)
        self.shift_step = float(shift_step)
        self.shift_max_forward = float(shift_max_forward)
        self.shift_max_backward = float(shift_max_backward)

        # buffer
        self.buffer_size = int(buffer_size)
        self.min_frames_for_predict = int(min_frames_for_predict)
        self.swing_lock_frames = int(swing_lock_frames)
        self.max_bounces_in_flight = int(max_bounces_in_flight)

        # table + integration
        self.table_z = float(table_z)
        self.dt_pred = float(dt_pred)
        self.max_t_pred = float(max_t_pred)

        # paddle target
        self.flight_time = float(flight_time)
        self.paddle_cor = float(paddle_cor)
        self.dtype = dtype
        self.device = device or torch.device("cpu")
        self._target_land = torch.tensor(
            list(target_land), dtype=self.dtype, device=self.device,
        )

        # ball params (k, Ch, Cv) — single load, not re-read each tick
        params = load_ball_params(params_path, device=self.device, dtype=self.dtype)
        self._k = float(params["k"])
        self._Ch = float(params["Ch"])
        self._Cv = float(params["Cv"])

        # rolling buffer + last-safe cache
        self._t_buf: deque = deque(maxlen=self.buffer_size)
        self._xyz_buf: deque = deque(maxlen=self.buffer_size)
        self._last_safe_packet: Optional[dict] = None
        self._last_safe_t: Optional[float] = None
        # swing-type lock: latched once _frames_seen ≥ swing_lock_frames,
        # cleared on reset(). Buffer is a sliding 31-frame window so we keep a
        # separate monotonic counter to detect the lock threshold.
        self._frames_seen: int = 0
        self._swing_locked: Optional[str] = None

    # -----------------------------------------------------------------
    def reset(self) -> None:
        """Drop the rolling buffer and the hold-last-safe cache."""
        self._t_buf.clear()
        self._xyz_buf.clear()
        self._last_safe_packet = None
        self._last_safe_t = None
        self._frames_seen = 0
        self._swing_locked = None

    # -----------------------------------------------------------------
    def _as_tensor(self, x, n: int) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(dtype=self.dtype, device=self.device).reshape(n)
        return torch.as_tensor(x, dtype=self.dtype, device=self.device).reshape(n)

    # -----------------------------------------------------------------
    def update(
        self,
        t: float,
        ball_pos,
        base_pos,
        base_quat,
    ) -> Optional[dict]:
        """One tick. See class docstring for the ladder. Returns packet or None.

        ``base_quat`` is currently pass-through (not consumed inside the
        planner) — kept on the input signature so downstream code can express
        targets in base-local frame later.
        """
        t_f = float(t)
        bp = self._as_tensor(ball_pos, 3)
        base_p = self._as_tensor(base_pos, 3)
        base_q = self._as_tensor(base_quat, 4)

        self._t_buf.append(t_f)
        self._xyz_buf.append(bp.detach().cpu().numpy().astype(np.float64))
        self._frames_seen += 1

        # gate 1: enough frames?
        if len(self._t_buf) < self.min_frames_for_predict:
            return self._hold_or_none(t_f, base_p, base_q, "buf_short")

        # build buffer tensors
        t_buf_t = torch.tensor(list(self._t_buf), dtype=self.dtype, device=self.device)
        xyz_buf_t = torch.tensor(np.stack(list(self._xyz_buf)),
                                  dtype=self.dtype, device=self.device)

        # gate 2: state estimate (Sec.IV-A)
        last_bnc = _find_tbl_bounce(t_buf_t, xyz_buf_t)
        st = estimate_state(
            t_buf_t, xyz_buf_t,
            last_bounce_idx=last_bnc if last_bnc is not None else -1,
        )
        if st is None:
            return self._hold_or_none(t_f, base_p, base_q, "state_fail")
        p_now, v_now, _ = st

        # gate 3: ball must be moving toward the robot (-x)
        if float(v_now[0]) >= 0.0:
            return self._hold_or_none(t_f, base_p, base_q, "no_dir")

        # core: pick a hit point in the workspace
        pred, plan_mode, x_hit_used = self._find_hit_point(p_now, v_now, base_p)
        if pred is None:
            return self._hold_or_none(t_f, base_p, base_q, "workspace_fail")

        # paddle solver (paper Eq.5+Eq.6)
        sol = solve_paddle_target(
            pred["hit_pos"], pred["hit_vel"],
            target_land=self._target_land,
            flight_time=self.flight_time,
            paddle_cor=self.paddle_cor,
        )
        if sol["paddle_normal"] is None:
            return self._hold_or_none(t_f, base_p, base_q, "paddle_solve_fail")

        # base target + swing type (paper Sec V.B.3 deploy heuristic; user-defined)
        base_target_xy, swing_type = self._compute_base_target(
            pred["hit_pos"], base_p,
        )

        # fresh packet
        packet = dict(
            base_pos=base_p,
            base_quat=base_q,
            p_hit_world=pred["hit_pos"],
            t_to_hit=float(pred["t_to_hit"]),
            v_racket_hat_world=sol["v_paddle"],
            n_target_world=sol["paddle_normal"],
            v_ball_in_world=pred["hit_vel"],
            v_ball_out_world=sol["v_ball_out"],
            target_land_world=sol["target_land"],
            n_buf=len(self._t_buf),
            plan_mode=plan_mode,
            x_hit_used=x_hit_used,
            stale_age_s=0.0,
            hold_reason=None,
            bounced=pred["bounced"],
            bounces=pred["bounces"],
            traj_p=pred["traj_p"],
            p_base_xy_world=base_target_xy,
            swing_type=swing_type,
        )
        self._last_safe_packet = packet
        self._last_safe_t = t_f
        return packet

    # -----------------------------------------------------------------
    def _compute_base_target(
        self,
        hit_pos: torch.Tensor,
        base_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, str]:
        """Paper Sec V.B.3 deploy heuristic — *not specified* by paper.

        User-defined right-handed paddle policy under +y=LEFT world frame:
          x: base sits 0.4 m behind the racket   → p̂_base_x = p̂_racket_x − 0.4
          y:
            racket on base's right (racket_y < base_y) → forehand
                p̂_base_y = p̂_racket_y + 0.25  (base 25 cm to the left of racket)
            else → backhand
                p̂_base_y = p̂_racket_y         (base directly under racket in y)

        Swing-type lifecycle (gated by ``swing_lock_frames``):
          * frames_seen <  swing_lock_frames → recompute swing every tick
            (early-flight pose can still flip as the prediction firms up)
          * frames_seen >= swing_lock_frames → latch on first crossing and
            hold until ``reset()``; only ``base_target_xy`` keeps tracking
            the ball afterward.
        """
        hp = hit_pos.detach().cpu().numpy().reshape(3)
        bp = base_pos.detach().cpu().numpy().reshape(3)

        if self._swing_locked is not None:
            swing_type = self._swing_locked
        else:
            swing_type = "forehand" if hp[1] < bp[1] else "backhand"
            if self._frames_seen >= self.swing_lock_frames:
                self._swing_locked = swing_type

        if swing_type == "forehand":
            by = float(hp[1] + 0.25)
        else:
            by = float(hp[1])

        bx = float(hp[0] - 0.40)

        out = torch.tensor(
            [bx, by], dtype=hit_pos.dtype, device=hit_pos.device,
        )
        return out, swing_type

    # -----------------------------------------------------------------
    def _why_oow(self, pred: dict, base_pos: torch.Tensor) -> Optional[str]:
        """Classify why ``pred`` is out-of-workspace. ``None`` = OK."""
        if pred is None or pred.get("hit_pos") is None:
            return "no_cross"
        hp = pred["hit_pos"].detach().cpu().numpy().reshape(3)
        bp = base_pos.detach().cpu().numpy().reshape(3)
        if hp[2] < self.z_min_world:
            return "z_low"
        if hp[2] > self.z_max_world:
            return "z_high"
        if abs(hp[1] - bp[1]) > self.y_max_dev:
            return "y_far"
        return None

    # -----------------------------------------------------------------
    def _find_hit_point(
        self,
        p_now: torch.Tensor,
        v_now: torch.Tensor,
        base_pos: torch.Tensor,
    ) -> tuple[Optional[dict], str, Optional[float]]:
        """Default plane → adaptive forward/backward shift along x_hit.

        Only z-axis violations (and ``no_cross``) are recovered by shifting:
          z too low / no_cross  →  push the plane forward (less negative x_hit)
                                   so the ball is intercepted earlier and higher
          z too high            →  push the plane backward (more negative x_hit)
                                   so the ball has more time to drop

        ``y_far`` and other failures return None — the controller defers to
        ``_hold_or_none`` and waits for a better state estimate next tick.
        """
        # (1) default plane
        pred = predict_hit_plane(
            p_now, v_now, self._k, self._Ch, self._Cv,
            x_hit=self.x_hit_default,
            table_z=self.table_z, max_bounces=self.max_bounces_in_flight,
            dt=self.dt_pred, max_t=self.max_t_pred,
        )
        fail = self._why_oow(pred, base_pos)
        if fail is None:
            return pred, "plane_default", self.x_hit_default

        # (2) adaptive shift (only for z-axis / no_cross issues)
        if fail in ("z_low", "no_cross"):
            direction = +1
            max_shift = self.shift_max_forward
        elif fail == "z_high":
            direction = -1
            max_shift = self.shift_max_backward
        else:
            # y_far or other — give up; controller will hold-last-safe
            return None, "fail", None

        if self.shift_step <= 0.0 or max_shift <= 0.0:
            return None, "fail", None

        n_steps = int(round(max_shift / self.shift_step))
        for kstep in range(1, n_steps + 1):
            x_try = self.x_hit_default + direction * kstep * self.shift_step
            cand = predict_hit_plane(
                p_now, v_now, self._k, self._Ch, self._Cv,
                x_hit=x_try,
                table_z=self.table_z, max_bounces=self.max_bounces_in_flight,
                dt=self.dt_pred, max_t=self.max_t_pred,
            )
            if self._why_oow(cand, base_pos) is None:
                sign = "+" if direction > 0 else "-"
                return cand, f"plane_shift({sign}{kstep})", x_try

        return None, "fail", None

    # -----------------------------------------------------------------
    def _hold_or_none(
        self,
        t: float,
        base_pos: torch.Tensor,
        base_quat: torch.Tensor,
        reason: str,
    ) -> Optional[dict]:
        """Return last-safe packet (with refreshed mocap pass-through and
        ``stale_age_s``) or None if the cache is cold."""
        if self._last_safe_packet is None:
            return None
        pkt = dict(self._last_safe_packet)
        pkt["base_pos"] = base_pos
        pkt["base_quat"] = base_quat
        pkt["plan_mode"] = "held"
        pkt["stale_age_s"] = float(t - (self._last_safe_t or t))
        pkt["hold_reason"] = reason
        return pkt


# ===================================================================
# CLI + smoke tests
# ===================================================================
def _print_report(res: dict) -> None:
    drag, bnc = res["drag"], res["bnc"]
    bar = "=" * 78
    print(bar)
    print("HITTER ball-dynamics fit (planner.py)")
    print(bar)
    print("[paper formulas, strictly enforced]")
    print("  Eq.1a  flight :  a = -k ||v|| v + g       (g = (0,0,-9.81))")
    print("  Eq.1b  bounce :  v+ = diag(Ch, Ch, -Cv) v-")
    print("  Eq.4   losses (closed-form):")
    print("           k  = Σ ||a-g|| ||v||² / Σ ||v||⁴")
    print("           Ch = Σ(|vx-||vx+|+|vy-||vy+|) / Σ(vx-²+vy-²)")
    print("           Cv = Σ |vz-||vz+| / Σ vz-²")
    print(f"[v,a estimator] per-axis {POLY_DEG}nd-order poly LSQ over latest {POLY_WIN} frames, "
          "buffer cleared at bounce (Sec.IV-A)")
    print(bar)
    print("[fitted parameters]")
    print(f"  k   = {drag['k']:.6f}  m^-1")
    print(f"  Ch  = {bnc['Ch']:.6f}  (-)")
    print(f"  Cv  = {bnc['Cv']:.6f}  (-)")
    print()
    print("[residuals]")
    print(f"  drag    RMS = {drag['residual_rms']:.4f} m/s²  "
          f"(over {drag['n_frames']} frames in {drag['n_files']} files)")
    print(f"  bounce  RMS_xy = {bnc['residual_rms_xy']:.4f} m/s, "
          f"RMS_z = {bnc['residual_rms_z']:.4f} m/s  "
          f"(over {bnc['n_traj']}/{bnc['n_files']} bounces; skipped {bnc['n_skipped']})")
    print()
    print("[sanity check] expected ranges (paper does not publish values):")
    print("  k  in [0.10, 0.20] m^-1   (ping-pong drag with m=2.7g, r=20mm)")
    print("  Ch in [0.60, 0.85]")
    print("  Cv in [0.70, 0.95]")
    print(bar)
    print("[per-trajectory bounce table]  |v⁻|, |v⁺|, Ch_i, Cv_i")
    for r in bnc["per_traj"]:
        print(f"  {r['file']:60s}  |v-|={r['v_minus_norm']:5.2f}  |v+|={r['v_plus_norm']:5.2f}  "
              f"Ch_i={r['Ch_i']:.3f}  Cv_i={r['Cv_i']:.3f}")
    if bnc["skipped"]:
        print()
        print("[skipped bounces]")
        for f, why in bnc["skipped"]:
            print(f"  {f}: {why}")
    print(bar)
    print(f"[saved] params -> {res['pt_path']}")
    print(f"[saved] meta   -> {res['json_path']}")
    print(bar)


def _run_smoke_tests() -> None:
    """Quick correctness checks for HitterPlanner. Run via:
        python planner.py --test
    Exercises the 4 cases that don't need synthetic-trajectory engineering:
      (1) real-trajectory replay produces fresh packets
      (2) direction gate (vx>0 throughout) yields no fresh packet
      (3) hold-last-safe holds after fresh packets stop arriving
      (4) cold-start hold (vx>0 from frame 0) returns only None
    """
    print("=" * 60)
    print("HitterPlanner smoke tests")
    print("=" * 60)

    files = sorted(f for f in os.listdir(DEFAULT_BOUNCE_DIR) if f.endswith(".txt"))
    p_traj = os.path.join(DEFAULT_BOUNCE_DIR, files[0])
    t, xyz = _load_trajectory(p_traj)
    print(f"[setup] using {os.path.basename(p_traj)}  n={len(t)}")

    # base placed near the centerline so the ball's expected hit-y is in reach
    base_pos = torch.tensor([-1.6, 0.0, 0.0], dtype=torch.float64)
    base_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)

    # ---- (1) real-trajectory replay ----
    print("\n[1] real-trajectory replay")
    planner = HitterPlanner()
    counts = dict(none=0, fresh=0, held=0)
    fresh_modes: set = set()
    last_stale = -1.0
    last_fresh_pkt: Optional[dict] = None
    for i in range(len(t)):
        pkt = planner.update(float(t[i]), xyz[i], base_pos, base_quat)
        if pkt is None:
            counts["none"] += 1
            last_stale = -1.0
        elif pkt["plan_mode"] == "held":
            counts["held"] += 1
            assert pkt["stale_age_s"] >= 0.0
            if last_stale >= 0.0:
                assert pkt["stale_age_s"] >= last_stale - 1e-9, \
                    "stale_age_s should be monotonic within a held run"
            last_stale = pkt["stale_age_s"]
        else:
            counts["fresh"] += 1
            fresh_modes.add(pkt["plan_mode"])
            last_fresh_pkt = pkt
            last_stale = 0.0
    print(f"    counts={counts}  fresh_modes={sorted(fresh_modes)}")
    assert counts["fresh"] > 0, "expected ≥1 fresh packet on a real bounce trajectory"
    assert last_fresh_pkt is not None
    required_cmd_keys = {
        "p_hit_world",
        "v_ball_in_world",
        "v_racket_hat_world",
        "n_target_world",
        "v_ball_out_world",
        "target_land_world",
        "p_base_xy_world",
    }
    old_packet_keys = {
        "hit_pos",
        "v_paddle",
        "paddle_normal",
        "v_ball_in",
        "v_ball_out",
        "target_land",
        "base_target_xy",
    }
    assert required_cmd_keys.issubset(last_fresh_pkt.keys()), last_fresh_pkt.keys()
    assert old_packet_keys.isdisjoint(last_fresh_pkt.keys()), last_fresh_pkt.keys()
    # physical sanity on the last fresh packet
    hp = last_fresh_pkt["p_hit_world"].cpu().numpy()
    vp = last_fresh_pkt["v_racket_hat_world"].cpu().numpy()
    n_pad = last_fresh_pkt["n_target_world"].cpu().numpy()
    assert 0.0 <= last_fresh_pkt["t_to_hit"] < 1.0, last_fresh_pkt["t_to_hit"]
    assert float(np.linalg.norm(vp)) < 10.0, vp
    assert abs(float(np.linalg.norm(n_pad)) - 1.0) < 1e-6, n_pad
    if last_fresh_pkt["plan_mode"] == "plane_default" or \
       last_fresh_pkt["plan_mode"].startswith("plane_shift"):
        assert planner.z_min_world <= hp[2] <= planner.z_max_world, hp
    print("    PASS")

    # ---- (2) direction gate ----
    print("\n[2] direction gate (vx>0 only)")
    planner = HitterPlanner()
    n_pkt = 0; n_n = 0
    for i in range(20):
        t_i = i * 0.01
        p_i = np.array([0.0 + 2.0 * t_i, 0.0, 1.0])
        pkt = planner.update(t_i, torch.tensor(p_i), base_pos, base_quat)
        (n_pkt if pkt is not None else n_n)  # noqa
        if pkt is None: n_n += 1
        else: n_pkt += 1
    print(f"    None={n_n}  packet={n_pkt}")
    assert n_pkt == 0, "no fresh packet should ever exist with vx>0"
    print("    PASS")

    # ---- (3) hold-last-safe ----
    print("\n[3] hold-last-safe after fresh stops")
    planner = HitterPlanner()
    half = len(t) // 2 + 5
    last_real_pkt = None
    for i in range(half):
        pkt = planner.update(float(t[i]), xyz[i], base_pos, base_quat)
        if pkt is not None and pkt["plan_mode"] != "held":
            last_real_pkt = pkt
    if last_real_pkt is None:
        print("    SKIP — no fresh packet from this clip's first half")
    else:
        # feed enough fake +x frames to fully flush the 31-frame buffer
        t_last = float(t[half - 1])
        p_last = xyz[half - 1].cpu().numpy()
        n_flush = planner.buffer_size + 8
        held_packets = []
        for j in range(1, n_flush + 1):
            t_j = t_last + j * 0.02
            p_j = p_last + np.array([+2.0, 0.0, 0.0]) * (j * 0.02)
            pkt = planner.update(t_j, torch.tensor(p_j), base_pos, base_quat)
            assert pkt is not None, "cache exists -> should hold, not None"
            if pkt["plan_mode"] == "held":
                assert required_cmd_keys.issubset(pkt.keys()), pkt.keys()
                assert old_packet_keys.isdisjoint(pkt.keys()), pkt.keys()
                held_packets.append(pkt)
        # last 8 must be held (buffer has been flushed for several frames now)
        assert len(held_packets) >= 8, f"expected ≥8 held packets, got {len(held_packets)}"
        stales = [p["stale_age_s"] for p in held_packets[-8:]]
        for a, b in zip(stales, stales[1:]):
            assert b >= a - 1e-9, f"stale_age_s should be monotonic, got {stales}"
        print(f"    held packets={len(held_packets)}/{n_flush}  "
              f"final stale_age_s={stales[-1]:.3f}")
        assert stales[-1] > 0.0
        print("    PASS")

    # ---- (4) cold-start hold (no cache) ----
    print("\n[4] cold-start hold")
    planner = HitterPlanner()
    for i in range(15):
        t_i = i * 0.01
        p_i = np.array([0.0 + 2.0 * t_i, 0.0, 1.0])
        pkt = planner.update(t_i, torch.tensor(p_i), base_pos, base_quat)
        assert pkt is None, f"expected None on cold-start with vx>0, got {pkt}"
    print("    PASS")

    print("\n" + "=" * 60)
    print("all smoke tests passed")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        _run_smoke_tests()
    else:
        res = fit_all()
        _print_report(res)
