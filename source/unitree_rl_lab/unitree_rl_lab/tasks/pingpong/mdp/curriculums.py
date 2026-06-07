from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from .commands import PingpongCommand

# A/B ablation toggle (env var, read at import; default now = σ-ease ON).
# PINGPONG_SIGMA_ORI_FLOOR: goal_orientation σ_ori floor. 0.20 = σ-ease (default,
# eases the over-pressure that pushed face→0.83 but dropped hsr→~0.69); 0.15 = no σ-ease.
# Visible in TensorBoard as Curriculum/pingpong/std_g_ori (settles at this floor).
_SIGMA_ORI_FLOOR = float(os.environ.get("PINGPONG_SIGMA_ORI_FLOOR", "0.20"))

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Default imitation sub-term weight split (sums to 1.0). Multiplied by w_i at runtime.
# Variants:
#   "default" / "joint_dominant" : joint-pos 0.65, joint-vel 0.10, body-pos 0.25
#                                  (originalv5.7 — strong joint anchoring)
#   "body_dominant"              : joint-pos 0.30, joint-vel 0.10, body-pos 0.60
#                                  (plan C — flips emphasis to body keypoints when
#                                   joint imitation drives the policy into a paddle-
#                                   normal cheat basin)
_IMIT_SPLIT_PRESETS: dict[str, dict[str, float]] = {
    "default": {
        "imitation_joint_pos": 0.65,
        "imitation_joint_vel": 0.10,
        "imitation_body_pos": 0.25,
    },
    "joint_dominant": {
        "imitation_joint_pos": 0.65,
        "imitation_joint_vel": 0.10,
        "imitation_body_pos": 0.25,
    },
    "body_dominant": {
        "imitation_joint_pos": 0.30,
        "imitation_joint_vel": 0.10,
        "imitation_body_pos": 0.60,
    },
}
_IMIT_SPLIT = _IMIT_SPLIT_PRESETS["default"]


# Module-level EMA of episode length at termination, used to gate imit_anneal
# phase advancement on actual standing skill rather than wall-clock iter.
# Reset only on process restart; that's intentional — within a run, we want a
# slow drift so brief drops don't roll the policy back to phase 0.
_EP_LENGTH_EMA: dict = {"value": 0.0, "init": False}


# Monotone latch for the signed strike-window goal_orientation reward.
# Closed (open=False) at startup → weight forced to 0. Flips to True once
# EMA crosses min_ep_length_for_ori_advance, and never re-closes (so a brief
# EMA dip after standing won't zero out the weight that the window curriculum
# has since raised).
#
# Why we need this: signed-ori provides a clean "which face should point at
# n_target" gradient. Combined with goal_velocity (lenient std=0.5), it lets
# the policy collect meaningful task reward while falling — committing to a
# swing-while-flailing basin instead of learning to stand. Baseline 23-07-21
# used |dot| (ambiguous) and never collected hit_success>0 during stand-up,
# which is the FEATURE that forced it to learn balance first. Run
# 2026-05-25_11-01-44 had hit_success=0.21 / vel_fail=0.001 at iter 2000 with
# EL=40 (baseline same iter: hit_success=0.0, EL=251) — the swing-basin
# diagnosis in action.
#
# goal_orientation_pre_strike is NOT covered here — it lives in
# _POS_VEL_GATE_LATCH so all three pre_strike rewards (pos/vel/ori) open
# together with one threshold and one restore mechanism.
_ORI_GATE_LATCH: dict = {"open": False}


# Monotone latch for goal_position / goal_velocity (+ their pre_strike
# variants). Mirrors _ORI_GATE_LATCH but covers the much larger pos/vel
# rewards (initial weights 2.0 / 2.0 / 0.3 / 1.0). Before this latch was
# added, those rewards were gated only by |t_to_hit| ≤ strike_window — they
# stayed live even before the policy could stand. After the M1 RSI base-yaw
# fix made the blade align correctly in world frame, run 2026-05-25_14-51-08
# stalled at EL≈41 for 1680+ iter while goal_velocity reward sat at 14× the
# baseline value: policy was farming pos/vel reward by swing-while-falling,
# the same swing-basin pathology as run 2026-05-25_11-01-44 but driven by
# pos/vel instead of ori. Holding pos/vel at 0 until EMA(EL)≥250 forces the
# stand-up phase to learn from imitation + alive + regularization only —
# matching the reward landscape baseline 23-07-21 had during its EL=40→234
# breakthrough.
#
# `original_weights` captures env_cfg values on the first curriculum call
# (before any zeroing) so that pre_strike weights — which the window
# curriculum (2e) does NOT ratchet — can be restored when the latch opens.
# Without this restore, run 2026-05-25_15-53-45 stood up by iter 367 but
# stalled at hit_success=0.003 with strike_blade_hit_dist_min=0.71m: post-
# strike rewards reactivated via window ratchet but pre_strike (the
# "pull-paddle-toward-p_hit" wind-up signal) stayed at 0 forever.
_POS_VEL_GATE_LATCH: dict = {"open": False, "original_weights": None}


# goal_base smooth-ramp curriculum (Option B in design discussion).
# goal_base belongs to Layer-1 locomotion (getting into position), not
# Layer-2 manipulation (striking). It must be active from iter 0 — gating
# it binary would force the policy to learn static stance first, then
# relearn dynamic balance with lateral motion (negative transfer).
# Instead ramp the weight linearly from `start_weight` (gentle, ~alive
# scale) at low ep_len_ema to `target_weight` (the env_cfg value, full
# strength) once basic standing is established. `target_weight` is
# captured on first call from env_cfg so a single source of truth.
_GOAL_BASE_RAMP: dict = {
    "target_weight": None,
    "start_weight": 0.5,
    "ep_lo": 50.0,
    "ep_hi": 250.0,
}


# Module-level state for metric-driven imitation phase advancement
# (schedule="metric"). Tracks EMAs of the four success indicators and a
# monotone max-phase latch so a brief metric drop after advancing doesn't
# roll w_i back. Reset only on process restart (intentional — same drift
# semantics as _EP_LENGTH_EMA).
_IMIT_METRIC_EMA: dict = {
    "hit_success_rate": 0.0,
    "pos_success_rate": 0.0,
    "vel_success_rate": 0.0,
    "ori_success_rate": 0.0,
    "init": False,
}
_IMIT_PHASE_LATCH: dict = {"max_phase": 0}

# v61: 3-phase task curriculum (stand → imit → strike).
# Monotone latch — once advanced, never reverts. User-driven design from
# 2026-05-30 sim observation: policy was using cross-body backhand strokes
# for forehand commands (cheat). Solution: force policy to FIRST learn to
# stand robustly (Phase 0), THEN imitate both swings well (Phase 1, no
# strike rewards yet), THEN add strike rewards (Phase 2, paper-aligned).
# Each phase advancement is gated on EL EMA + minimum iter duration.
# Imit weight schedule: 0.10 (Phase 0) → 1.00 (Phase 1) → 0.30 (Phase 2).
# goal_* rewards are zeroed in Phase 0/1, restored to env_cfg baselines on
# Phase 2 entry (one-time reset to break the cos_sim_ratchet_freeze loop).
_TASK_PHASE_LATCH: dict = {"phase": 0, "phase_1_entry_iter": -1, "prev_phase": 0}

# v62: cross-curriculum cooldown to STAGGER shape_tier (which controls σ_pos/σ_vel/σ_ori
# tightening) and v_in_mag (which controls ball incoming speed). User feedback:
# if both tighten σ_vel AND increase ball speed in the same iter, policy gets
# shocked — reward range shifts faster than policy can adapt. Cooldown enforces
# at least N iter between any pair of (shape_tier ↑, v_in_mag ↑) events.
# Each curriculum tracks last upgrade iter; the OTHER curriculum checks before
# applying its own upgrade. If too recent, hold the upgrade for next iter.
_SHAPE_TIER_LATCH: dict = {"tier": 0, "last_change_iter": -10000}
_V_IN_TIER_LATCH: dict = {"high": 2.0, "last_change_iter": -10000}
_CROSS_CURRICULUM_COOLDOWN_ITERS = 500

_TASK_PHASE_IMIT_SPLIT: dict[str, float] = {
    "imitation_joint_pos": 0.40,
    "imitation_body_pos": 0.50,
    "imitation_joint_vel": 0.10,
}

_TASK_PHASE_GOAL_TERMS: tuple[str, ...] = (
    "goal_position",
    "goal_position_pre_strike",
    "goal_velocity",
    "goal_velocity_pre_strike",
    "goal_orientation",
    "goal_orientation_pre_strike",
)

# v61 baseline weights for goal_* terms — used on Phase 2 entry to restore
# weights from 0 (set during Phase 0/1) to env_cfg defaults. This bypasses
# the window curriculum freeze (cos_sim_ratchet_freeze) that would otherwise
# leave goal_* at 0 forever (run 2026-05-30_15-41-23 stuck at goal_*=0
# because cos_sim_ema=0.15 < freeze_threshold=0.45 → window curriculum
# never advanced past tier 0 → max(0, target) didn't fire).
_TASK_PHASE_2_BASELINE_WEIGHTS: dict[str, float] = {
    "goal_position": 2.0,
    "goal_position_pre_strike": 1.0,
    "goal_velocity": 2.0,
    "goal_velocity_pre_strike": 1.0,
    "goal_orientation": 4.0,  # RAISED 0.5→4.0: strike-instant face maintenance entering Phase 2 (pairs with w_ori tiers)
    "goal_orientation_pre_strike": 0.5,  # moot — task_phase overrides per-phase via face_prestrike_phase_weights
}

# Face-orienting joints whose imitation weight update_task_phase sets per-phase
# (early face learning). Cached joint_name -> local index in imitation_joint_names.
_FACE_IMIT_IDS: dict = {}


def update_task_phase(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    el_phase_0_to_1: float = 350.0,
    el_phase_1_to_2: float = 450.0,
    phase_1_min_iters: int = 2000,
    imit_w_phase0: float = 0.10,
    imit_w_phase1: float = 1.00,
    imit_w_phase2: float = 0.30,
    num_steps_per_env: int = 24,
    leg_reg_phase_weights: dict | None = None,
    command_name: str = "pingpong",
    face_prestrike_phase_weights: tuple | None = None,
    face_imit_phase_weights: dict | None = None,
    face_p1_ramp_frac: tuple = (0.4, 0.8),
    face_prestrike_p1_early: float = 0.2,
    face_imit_p1_early: float = 1.0,
) -> dict[str, float]:
    """v61 3-phase task curriculum with monotone latches.

    Phase 0 (stand): imit weight LOW (default 0.10), all goal_* rewards ZERO.
                     Policy learns to stand from RSI mid-swing pose using
                     stand rewards (alive/pelvis_*/base_*) plus weak imit prior.
                     Gate to advance: EL_ema ≥ el_phase_0_to_1 (default 350).
    Phase 1 (imit):  imit weight HIGH (default 1.00), goal_* still ZERO.
                     Policy must imitate both forehand and backhand demos
                     correctly. Gate to advance: BOTH EL_ema ≥ el_phase_1_to_2
                     AND phase_1_min_iters elapsed since Phase 1 entry.
                     v61.1 fix: previous version (no min duration) let Phase 0
                     skip to Phase 2 in ~50 iter when EL surged from 339 to 448
                     in one window — Phase 1 (heavy imit) never got to teach
                     forehand vs backhand differentiation. Min duration ensures
                     policy gets enough exposure to imit signal.
    Phase 2 (strike): imit weight PAPER (default 0.30), goal_* ENABLED via
                      one-time reset to env_cfg baselines (2.0/1.0/2.0/1.0/0.5/0.5
                      for pos/pos_pre/vel/vel_pre/ori/ori_pre). The reset is
                      necessary because window curriculum's cos_sim_ratchet_freeze
                      (cos_sim_ema < 0.45) would otherwise keep weights at 0
                      indefinitely, creating a chicken-and-egg loop (no goal_ori
                      signal → low cos_sim → freeze → no signal).

    Latches are MONOTONE — phase only advances, never reverts. This is the
    user's "single-direction valve" design: once policy proves competence at
    a phase, lock that competence in and move on.

    Reads: _EP_LENGTH_EMA (populated by update_imitation_weight which keeps
    schedule="metric" purely for EMA tracking).
    """
    el_ema = float(_EP_LENGTH_EMA["value"]) if _EP_LENGTH_EMA["init"] else 0.0
    iter_count = int(env.common_step_counter // max(num_steps_per_env, 1))

    cur_phase = int(_TASK_PHASE_LATCH["phase"])
    prev_phase = cur_phase

    # Phase 0 → 1
    if cur_phase < 1 and el_ema >= float(el_phase_0_to_1):
        cur_phase = 1
        _TASK_PHASE_LATCH["phase"] = 1
        _TASK_PHASE_LATCH["phase_1_entry_iter"] = iter_count

    # Phase 1 → 2 (requires BOTH EL gate AND min duration in Phase 1)
    if cur_phase < 2 and el_ema >= float(el_phase_1_to_2):
        phase_1_iters = iter_count - int(_TASK_PHASE_LATCH["phase_1_entry_iter"])
        if phase_1_iters >= int(phase_1_min_iters):
            cur_phase = 2
            _TASK_PHASE_LATCH["phase"] = 2

    imit_w_table = (float(imit_w_phase0), float(imit_w_phase1), float(imit_w_phase2))
    w_i = imit_w_table[cur_phase]

    # Override imit weights based on task phase.
    for term_name, share in _TASK_PHASE_IMIT_SPLIT.items():
        env.reward_manager.get_term_cfg(term_name).weight = share * w_i

    # Phase 0/1: zero out all goal_* rewards.
    if cur_phase < 2:
        for term_name in _TASK_PHASE_GOAL_TERMS:
            env.reward_manager.get_term_cfg(term_name).weight = 0.0

    # Phase 2 entry: one-time reset of goal_* weights to env_cfg baselines.
    # This breaks the cos_sim_ratchet_freeze deadlock (window curriculum stays
    # frozen because weights=0 → no goal_ori signal → cos_sim stays low → freeze).
    # After this reset, window curriculum's max(current, target) can ramp up
    # weights as success EMAs climb (or just leave them at baseline if frozen).
    if prev_phase < 2 and cur_phase == 2:
        for term_name, weight in _TASK_PHASE_2_BASELINE_WEIGHTS.items():
            env.reward_manager.get_term_cfg(term_name).weight = float(weight)

    # Phase-scaled lower-body (leg) regularizers. Set every tick like the imit
    # weights above. Gentle profile, Phase 2 slightly weaker so the leg terms
    # don't fight lateral repositioning / low-ball squats (goal_base — the
    # move-into-position reward — is active from Phase 0, so legs need stepping
    # room in every phase). `leg_reg_phase_weights` maps reward-term name ->
    # (phase0_w, phase1_w, phase2_w). None = config opted out (e.g. 29dof before
    # sync) -> no-op. Per-term try/except keeps it robust if a config lacks the
    # term entirely.
    leg_dev_w_now = 0.0
    if leg_reg_phase_weights is not None:
        for term_name, ws in leg_reg_phase_weights.items():
            try:
                env.reward_manager.get_term_cfg(term_name).weight = float(ws[cur_phase])
            except (KeyError, AttributeError, IndexError):
                continue
        ld = leg_reg_phase_weights.get("leg_joint_deviation")
        if ld is not None:
            leg_dev_w_now = float(ld[cur_phase])

    # ── Early face learning with intra-Phase-1 ramp (POSTURE-FIRST) ─────────
    # Split Phase 1 EARLY→LATE so the policy learns the distinct fh/bh POSTURES
    # first (high imit, LOW face reward — can't game the posture-agnostic face),
    # then ramp the face reward UP mid-late Phase 1 to refine the face WITHIN the
    # already-learned posture basin (the degenerate "forehand-posture + wrist-flip"
    # earns no extra face reward → no gradient pulls there). pos/vel stay 0 until
    # Phase 2. r1 goes 0→1 as phase_1_iters_elapsed crosses [f0,f1]·phase_1_min_iters;
    # Phase 0/2 use the tuple value directly. body_pos (right elbow+shoulder_yaw now
    # tracked) position-anchors the posture as a backstop; face stays on wrist_roll.
    _p1_entry = int(_TASK_PHASE_LATCH["phase_1_entry_iter"])
    _p1_elapsed = (iter_count - _p1_entry) if _p1_entry >= 0 else 0
    _P = max(int(phase_1_min_iters), 1)
    _lo, _hi = float(face_p1_ramp_frac[0]) * _P, float(face_p1_ramp_frac[1]) * _P
    r1 = min(1.0, max(0.0, (_p1_elapsed - _lo) / max(_hi - _lo, 1.0)))

    if face_prestrike_phase_weights is not None:
        if cur_phase == 1:
            _e = float(face_prestrike_p1_early)
            _fw = _e + r1 * (float(face_prestrike_phase_weights[1]) - _e)
        else:
            _fw = float(face_prestrike_phase_weights[cur_phase])
        try:
            env.reward_manager.get_term_cfg("goal_orientation_pre_strike").weight = _fw
        except (KeyError, AttributeError, IndexError):
            pass
    if face_imit_phase_weights is not None:
        _fcmd = env.command_manager.get_term(command_name)
        if not _FACE_IMIT_IDS:
            _names = list(_fcmd.cfg.imitation_joint_names)
            for _jn in face_imit_phase_weights:
                if _jn in _names:
                    _FACE_IMIT_IDS[_jn] = _names.index(_jn)
        for _jn, _ws in face_imit_phase_weights.items():
            _lid = _FACE_IMIT_IDS.get(_jn)
            if _lid is None:
                continue
            if cur_phase == 1:
                _e = float(face_imit_p1_early)
                _val = _e + r1 * (float(_ws[1]) - _e)
            else:
                _val = float(_ws[cur_phase])
            _fcmd.imit_joint_weights[_lid] = _val

    _TASK_PHASE_LATCH["prev_phase"] = cur_phase

    phase_1_iters_now = (
        iter_count - int(_TASK_PHASE_LATCH["phase_1_entry_iter"])
        if int(_TASK_PHASE_LATCH["phase_1_entry_iter"]) >= 0
        else 0
    )
    return {
        "task_phase": float(cur_phase),
        "task_phase_imit_w": float(w_i),
        "task_phase_el_ema": el_ema,
        "task_phase_el_gate_0_to_1": float(el_phase_0_to_1),
        "task_phase_el_gate_1_to_2": float(el_phase_1_to_2),
        "task_phase_iter_count": float(iter_count),
        "task_phase_1_iters_elapsed": float(phase_1_iters_now),
        "task_phase_1_min_iters": float(phase_1_min_iters),
        "task_phase_leg_dev_w": leg_dev_w_now,
        "task_phase_face_ramp_r1": float(r1),
    }


# Multi-metric reward-shaping curriculum (sigma_g_pos + goal_velocity std).
# Tier advances are gated on ALL FOUR imitation-tracked EMAs (hsr / pos / vel /
# ori success rates) crossing tier thresholds simultaneously. Rationale: prior
# single-hsr-gated sigma curriculum let sigma_g_pos drop to 0.15 at hsr=0.30
# even with vel_fail=0.62 and ori_fail=0.32 — pos_fail was already ~0.06 so
# tightening pos sigma was wasted while vel signal was starved. Multi-gate
# only tightens reward shaping when all four task channels are passing,
# preventing one-channel collapse.
#
# CRITICAL: hsr_thr is the cold-start gate. pos/vel/ori EMAs are computed as
# (1 - fail_rate) and seed from the first observation; before any episode
# triggers a hit, fail_rate=0 so the EMA initializes to 1.0 (run 19-44-44
# bug: shape_tier jumped to 4 at iter 52 because pos/vel/ori EMAs were 0.999).
# hsr starts at 0 and only climbs when real hits occur, so it acts as the
# "have we actually hit anything yet?" gate that blocks premature tightening.
#
# Format: (sigma_g_pos, std_g_vel, std_g_ori, hsr_thr, pos_thr, vel_thr, ori_thr).
# Tiers checked top-down; first tier whose all-4-EMAs pass wins.
# Floor (sigma_g_pos=0.06, std_g_vel=0.25, std_g_ori=0.20) is the paper-strict
# precision target.
# Defaults (tier 0): sigma_g_pos=0.30, std_g_vel=0.45, std_g_ori=0.40 — all
# loose so reward magnitudes start in the same order (~0.05–0.10 weighted).
# Why std_g_ori needs to be in the tier system (was missing pre run 19-48-49):
# goal_orientation uses cosine distance with `(1.0 - dot).clamp(min=0.0)`. With
# std=0.2 fixed, basin half-life is at ~37° error; outside the basin the
# gradient is FLAT ZERO (clamp + tiny exp(-large)). Run 19-48-49 iter 3909
# showed ori_ema stuck at 0.61 for 1100+ iter while pos/vel EMAs climbed past
# 0.95/0.86 — policy locked into "hit pos+vel correct, paddle face wrong"
# local optimum because no signal pulled paddle into the orientation basin.
# Putting std_g_ori in the tier ladder mirrors std_g_vel: start wide (0.40 →
# basin ~53°), tighten only after ori_ema demonstrates competence.
#
# 7-tier finer ladder (R10 expansion 2026-05-27): the 5-tier table had a
# 0.05 std_g_vel jump from tier 1→2 (0.33→0.28) which run 16-45-31 exposed
# as a vel-reward cliff: vel reward peaked 0.0053 at iter 5359 with std=0.30
# (tier 0) under the new linear-exp formula, then dropped to 0.0021 horizontal
# for 1500 iter once the monotone latch tightened std to 0.20 at tier 2.
# Splitting each std-vel hop into ~0.03 increments lets the policy spend
# more time consolidating at each width before the next squeeze. sigma_g_pos
# and std_g_ori columns also get intermediate stops at the same EMA gates so
# the 4-EMA AND check still graduates atomically. Top tier (6) preserves
# paper-strict targets (0.06 / 0.20 / 0.20).
_REWARD_SHAPE_TIERS: tuple[tuple[float, float, float, float, float, float, float], ...] = (
    # v62: std_g_vel column rescaled for Gaussian formula (squared norm).
    # Math: exp(-(2.0)²/σ²) — σ=0.50 → 0, σ=1.0 → 0.018, σ=1.5 → 0.17, σ=2.0 → 0.37.
    # Top tier σ=0.50 demands tight precision; tier 0 σ=1.50 lets policy
    # escape v61's "no gradient" trap (Laplacian σ=0.45 + ||Δv||=2 → reward 0.001).
    (0.06, 0.50, 0.15, 0.85, 0.95, 0.85, 0.85),  # tier 6 (v64: sharper ori σ 0.20→0.15)
    (0.08, 0.65, 0.18, 0.75, 0.92, 0.78, 0.80),  # tier 5 (v64: ori σ 0.22→0.18)
    (0.10, 0.80, 0.22, 0.65, 0.88, 0.70, 0.75),  # tier 4 (v64: ori σ 0.25→0.22)
    (0.13, 1.00, 0.28, 0.55, 0.82, 0.62, 0.70),  # tier 3
    (0.18, 1.20, 0.32, 0.40, 0.75, 0.55, 0.65),  # tier 2
    (0.24, 1.35, 0.36, 0.20, 0.60, 0.40, 0.55),  # tier 1
    (0.30, 1.50, 0.40, 0.00, 0.00, 0.00, 0.00),  # tier 0 (default, vel σ=1.50)
)


def _reward_shape_tier(
    hsr_ema: float, pos_ema: float, vel_ema: float, ori_ema: float
) -> tuple[int, float, float, float]:
    """Return (tier_index_from_top, sigma_g_pos_target, std_g_vel_target, std_g_ori_target).

    Tier index is 0=loosest (last in tuple) ... 6=tightest. Returned values
    are still subject to the monotone latch in the caller (max(min(cur, target), floor)).
    All four EMAs must exceed the tier's thresholds for that tier to be granted.
    """
    for idx, (sigma, std_vel, std_ori, hsr_thr, pos_thr, vel_thr, ori_thr) in enumerate(_REWARD_SHAPE_TIERS):
        if hsr_ema >= hsr_thr and pos_ema >= pos_thr and vel_ema >= vel_thr and ori_ema >= ori_thr:
            tier = len(_REWARD_SHAPE_TIERS) - 1 - idx
            return tier, sigma, std_vel, std_ori
    # Fallback (should never hit since tier 0 thresholds are 0): loosest.
    return 0, 0.30, 0.38, 0.40


# Strike-window curriculum: as success EMAs climb, shrink strike_window
# (the |t_to_hit| <= window gate used by goal_position / goal_velocity /
# goal_orientation in rewards.py) and ramp up the matching reward weights.
#
# Why couple them: the integrated per-episode reward through the gate is
# roughly weight × (window / dt). Holding that product constant keeps PPO's
# learning signal stable as the window narrows. The final tier intentionally
# bumps weight slightly faster than 1/window — once the policy has the gross
# motion, demand precision.
#
# CRITICAL — 4-EMA gate (D1 fix, run 2026-05-25_22-50-41 post-mortem):
# Prior version gated on instantaneous batch-mean success_rate, sampled
# from the just-reset env subset (50-200 envs per call). With true global
# hsr=0.27 at iter 1036 and 0.42 at iter 2424, the per-batch sample spiked
# to 0.60+ and 0.80+ respectively, tripping the monotone ratchet to top
# tier (window=0.01, w=12/12/4) before the policy was anywhere near ready.
# The aggressive reward landscape then locked the policy in a "swing-vel-
# priority, miss pos by 21cm" basin for the remaining 17k iter.
#
# New gate: every tier requires hsr_ema AND pos_ema AND vel_ema AND ori_ema
# all clear tier-specific thresholds (mirroring shape_tier's 4-channel AND).
# Uses _IMIT_METRIC_EMA (slow alpha=0.05 EMA) so single-batch noise can't
# trigger advancement. Prevents one-channel collapse from pulling the others
# into a higher tier they can't sustain.
#
# Format: (hsr_thr, pos_thr, vel_thr, ori_thr, strike_window_s, w_pos, w_vel, w_ori)
# Tiers checked top-down; first tier whose all-4-EMAs pass wins.
# dt = sim.dt * decimation = 0.005 * 4 = 0.02s, so:
#   window=0.10 -> ~5 frames active per swing  (baseline; original setting)
#   window=0.06 -> ~3 frames
#   window=0.04 -> ~2 frames
#   window=0.02 -> ~1 frame
#   window=0.01 -> 1 frame, strict (paper target)
_WINDOW_CURRICULUM_TIERS: tuple[tuple[float, float, float, float, float, float, float, float], ...] = (
    # cols: hsr_thr, pos_thr, vel_thr, ori_thr, strike_window, w_pos, w_vel, w_ori
    # w_ori RAISED (was 0.5→4) to HOLD the Phase-1-learned face against pos/vel in
    # Phase 2: cos_sim dipped 0.92→0.85 at Phase-2 entry (run 18-26-56) because the
    # strike-instant face weight was only 0.5 vs pos+vel 4. Now comparable from tier 0.
    (0.80, 0.85, 0.85, 0.85, 0.01, 12.0, 12.0, 10.0),
    (0.60, 0.75, 0.70, 0.75, 0.02,  8.0,  8.0,  8.0),
    (0.40, 0.65, 0.55, 0.65, 0.04,  5.0,  5.0,  6.0),
    (0.20, 0.50, 0.40, 0.55, 0.06,  3.0,  3.0,  5.0),
    (0.00, 0.00, 0.00, 0.00, 0.10,  2.0,  2.0,  4.0),
)


# Module-level state for D2 cos_sim guardrail. Tracks an EMA of the
# at-impact cos_sim metric so the window-curriculum ratchet can halt when
# the orientation channel is degrading. Run 2026-05-25_22-50-41 post-mortem:
# cos_sim 50i mean regressed 0.58 (iter 1569) → 0.50 (iter 19804) while
# strike_window kept ratcheting; 500i min hit 0.35 (below the cos basin
# half-width ~0.45). This guardrail freezes ratchet whenever cos_sim
# average is in the danger zone, preventing further tightening that would
# strand the orientation gradient.
_COS_SIM_EMA: dict = {"value": 0.0, "init": False}

# v64: monotone σ latch (re-added; removed in v60). Once a shape σ tightens it
# never loosens (min only). Breaks the shape_tier 4<->5 limit cycle where
# ori_success hovers at the tier-5 threshold (0.80 = backhand ori bar) and σ
# loosens back each dip, so the face never gets pushed past 0.80. Reset per
# process (resume starts fresh at inf and re-tightens monotonically).
_SIGMA_LATCH: dict = {"sigma_g_pos": float("inf"), "std_g_vel": float("inf"), "std_g_ori": float("inf")}

# v65: stall-driven down-weighting of the face-orienting joints' imitation. The
# full-joint imitation pins waist_yaw + right-arm distal to the (static-waist,
# ~0.80-face) demo, capping the strike face. This latch LOWERS (never frees) the
# imitation weight on those joints each time cos_sim_ema plateaus, so goal_orientation
# can recruit the waist twist + wrist to push the face past 0.80. Only acts in Phase 2.
_IMIT_ORIENT_LATCH: dict = {"w": 1.0, "best_cos": -1.0, "anchor_iter": -1, "local_ids": None}

# Module-level state for the pre_strike shaping anneal (update_prestrike_ramp_anneal):
# shrinks the pos/vel/ori pre_strike ramp_time as cos_sim_ema plateaus, then disables
# the pre_strike terms entirely (ramp→0 + weight→0). `off` latches once disabled.
_PRESTRIKE_LATCH: dict = {"ramp": None, "best_cos": -1.0, "anchor_iter": -1, "off": False}


def update_imit_orient_weight(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    command_name: str = "pingpong",
    orient_joint_names: tuple = (
        "waist_yaw_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
    ),
    active_phase: int = 2,
    stall_iters: int = 600,
    stall_decay: float = 0.6,
    floor: float = 0.05,
    improve_eps: float = 0.005,
    num_steps_per_env: int = 24,
) -> dict[str, float]:
    """Lower the imitation weight on the face-orienting joints (waist + right-arm
    distal) as the policy plateaus, freeing goal_orientation to push the paddle face
    past the demo's ~0.80 cap. Stall-driven: each time cos_sim_ema fails to improve by
    `improve_eps` for `stall_iters` iters, the weight steps down x`stall_decay` (to
    `floor`). Only acts in Phase >= active_phase (Phase 0/1 keep full imitation so the
    swing is learned). Reads _COS_SIM_EMA (set by update_pingpong_curriculum) and
    _TASK_PHASE_LATCH (set by update_task_phase) — register this term AFTER both."""
    cmd = env.command_manager.get_term(command_name)
    if _IMIT_ORIENT_LATCH["local_ids"] is None:
        names = list(cmd.cfg.imitation_joint_names)
        want = set(orient_joint_names)
        ids = [i for i, n in enumerate(names) if n in want]
        _IMIT_ORIENT_LATCH["local_ids"] = torch.tensor(ids, dtype=torch.long, device=cmd.device)
    phase = int(_TASK_PHASE_LATCH["phase"])
    iter_count = int(env.common_step_counter // max(num_steps_per_env, 1))
    if phase >= int(active_phase):
        cos = float(_COS_SIM_EMA["value"]) if _COS_SIM_EMA["init"] else 0.0
        if int(_IMIT_ORIENT_LATCH["anchor_iter"]) < 0:
            _IMIT_ORIENT_LATCH["anchor_iter"] = iter_count
            _IMIT_ORIENT_LATCH["best_cos"] = cos
        elif cos > float(_IMIT_ORIENT_LATCH["best_cos"]) + float(improve_eps):
            _IMIT_ORIENT_LATCH["best_cos"] = cos
            _IMIT_ORIENT_LATCH["anchor_iter"] = iter_count
        elif iter_count - int(_IMIT_ORIENT_LATCH["anchor_iter"]) >= int(stall_iters):
            _IMIT_ORIENT_LATCH["w"] = max(float(floor), float(_IMIT_ORIENT_LATCH["w"]) * float(stall_decay))
            _IMIT_ORIENT_LATCH["anchor_iter"] = iter_count
        lids = _IMIT_ORIENT_LATCH["local_ids"]
        if lids.numel() > 0:
            # Multiplicative: stacks this stall-driven decay ON TOP of the per-phase
            # face-joint seed update_task_phase writes each tick (it runs first), so
            # Phase-2 face-joint imit = seed[p2] x w (w decays 1.0 -> floor on stall).
            cmd.imit_joint_weights[lids] *= float(_IMIT_ORIENT_LATCH["w"])
    return {"imit_orient_weight": float(_IMIT_ORIENT_LATCH["w"])}


def update_prestrike_ramp_anneal(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    command_name: str = "pingpong",
    prestrike_terms: tuple = (
        "goal_position_pre_strike",
        "goal_velocity_pre_strike",
        "goal_orientation_pre_strike",
    ),
    active_phase: int = 2,
    initial_ramp: float = 0.2,
    stall_iters: int = 600,
    ramp_decay: float = 0.6,
    off_threshold: float = 0.05,
    improve_eps: float = 0.005,
    num_steps_per_env: int = 24,
) -> dict[str, float]:
    """Shrink the pre_strike shaping window (ramp_time) of the pos/vel/ori pre_strike
    rewards as the policy plateaus, then disable them so the policy relies on the true
    strike-instant reward (removes the dense-shaping crutch). Stall-driven: each time
    cos_sim_ema fails to improve by `improve_eps` for `stall_iters` iters, ramp_time
    steps down x`ramp_decay`; once it would drop below `off_threshold` it is set to 0
    (the pre_strike gate `0<t_to_hit<ramp_time` becomes empty → reward 0) and the three
    terms' weights are zeroed for good measure. Only acts in Phase >= active_phase
    (Phase 0/1 keep the full pre_strike window). Reads _COS_SIM_EMA (set by
    update_pingpong_curriculum) and _TASK_PHASE_LATCH (set by update_task_phase) —
    register this term AFTER both."""
    if _PRESTRIKE_LATCH["ramp"] is None:
        _PRESTRIKE_LATCH["ramp"] = float(initial_ramp)
    phase = int(_TASK_PHASE_LATCH["phase"])
    iter_count = int(env.common_step_counter // max(num_steps_per_env, 1))
    if phase >= int(active_phase) and not bool(_PRESTRIKE_LATCH["off"]):
        cos = float(_COS_SIM_EMA["value"]) if _COS_SIM_EMA["init"] else 0.0
        if int(_PRESTRIKE_LATCH["anchor_iter"]) < 0:
            _PRESTRIKE_LATCH["anchor_iter"] = iter_count
            _PRESTRIKE_LATCH["best_cos"] = cos
        elif cos > float(_PRESTRIKE_LATCH["best_cos"]) + float(improve_eps):
            _PRESTRIKE_LATCH["best_cos"] = cos
            _PRESTRIKE_LATCH["anchor_iter"] = iter_count
        elif iter_count - int(_PRESTRIKE_LATCH["anchor_iter"]) >= int(stall_iters):
            new_ramp = float(_PRESTRIKE_LATCH["ramp"]) * float(ramp_decay)
            _PRESTRIKE_LATCH["anchor_iter"] = iter_count
            if new_ramp < float(off_threshold):
                _PRESTRIKE_LATCH["ramp"] = 0.0
                _PRESTRIKE_LATCH["off"] = True
                for term_name in prestrike_terms:
                    env.reward_manager.get_term_cfg(term_name).weight = 0.0
            else:
                _PRESTRIKE_LATCH["ramp"] = new_ramp
        # Propagate the current ramp_time into each pre_strike term every tick.
        ramp_now = float(_PRESTRIKE_LATCH["ramp"])
        for term_name in prestrike_terms:
            env.reward_manager.get_term_cfg(term_name).params["ramp_time"] = ramp_now
    ramp_log = float(_PRESTRIKE_LATCH["ramp"]) if _PRESTRIKE_LATCH["ramp"] is not None else float(initial_ramp)
    return {"prestrike_ramp": ramp_log, "prestrike_off": float(bool(_PRESTRIKE_LATCH["off"]))}


def update_imitation_weight(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    schedule: str = "iter",
    num_steps_per_env: int = 24,
    iter_thresholds: tuple[int, int] = (3000, 8000),
    w_i_values: tuple[float, float, float] = (0.5, 0.3, 0.15),
    split: str | dict[str, float] | None = None,
    min_ep_length_for_phase_advance: float | None = None,
    ep_length_ema_alpha: float = 0.05,
    command_name: str = "pingpong",
    phase_thresholds: tuple[dict[str, float], ...] | None = None,
    metric_ema_alpha: float = 0.05,
) -> dict[str, float]:
    """Anneal the imitation top-level weight w_i in three phases.

    Rationale: starting w_i high (~0.5) gives PPO a strong shaping signal toward the
    reference posture before the policy collapses into a body-shift cheat basin
    (where it converts every ball into a backhand by translating the torso). Once
    the gross posture is locked in, drop w_i so task rewards (goal_pos / goal_vel)
    can drive precision.

    Schedule modes:
      - "iter"   : phase 0 → w[0], phase 1 → w[1], phase 2 → w[2]
                   boundaries are PPO iter counts.
      - "metric" : phase advances when EMA of (hit_success / pos_success /
                   vel_success / ori_success) plus ep_length_ema all clear
                   the per-phase thresholds in `phase_thresholds`. Monotone
                   via _IMIT_PHASE_LATCH — once advanced, never roll back.
                   Use this when iter-based timing diverges from actual
                   skill (the original motivation for min_ep_length_for_
                   phase_advance: a 33k-iter run that never stood up still
                   anneal'd w_i down at iter 8000, killing the only shaping
                   signal). With metric mode, w_i stays high until the
                   policy demonstrates competence, regardless of iter.
      - "off"    : do nothing (lets the static weights in env_cfg stand).

    From-scratch safety gate (min_ep_length_for_phase_advance):
      Only relevant in "iter" mode. When set, phase advancement is also gated
      on an EMA of the episode length at termination. Below the threshold,
      phase is clamped to 0 regardless of iter count; between [threshold,
      2*threshold), phase is clamped to ≤1. In "metric" mode this is folded
      into phase_thresholds[*]["min_ep_length"] and ignored.

    Metric mode `phase_thresholds` shape:
      Tuple of dicts, length = len(w_i_values) - 1. phase_thresholds[i] is
      the threshold to advance from phase i to phase i+1. Each dict accepts
      any subset of:
        "hit_success_rate", "pos_success_rate", "vel_success_rate",
        "ori_success_rate", "min_ep_length"
      Missing keys default to 0.0 (no constraint). All listed keys must be
      satisfied (logical AND) to advance.

    Implementation note: env.reward_manager.get_term_cfg(name) returns a live ref
    (RewardManager line ~195), and the per-step compute reads cfg.weight every step
    (line ~150), so mutating .weight here is sufficient — no set_term_cfg needed.
    """
    # Update episode-length EMA from just-terminated envs. CurriculumManager
    # passes env_ids = the envs being reset, and episode_length_buf[env_ids]
    # still holds their pre-reset terminal length at this point.
    if env_ids is not None:
        if isinstance(env_ids, slice):
            sample = env.episode_length_buf
        elif isinstance(env_ids, torch.Tensor):
            sample = env.episode_length_buf[env_ids]
        else:
            idx = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
            sample = env.episode_length_buf[idx]
        if sample.numel() > 0:
            cur = float(sample.float().mean().item())
            if not _EP_LENGTH_EMA["init"]:
                _EP_LENGTH_EMA["value"] = cur
                _EP_LENGTH_EMA["init"] = True
            else:
                a = float(ep_length_ema_alpha)
                _EP_LENGTH_EMA["value"] = (1.0 - a) * _EP_LENGTH_EMA["value"] + a * cur
    ep_length_ema = float(_EP_LENGTH_EMA["value"])

    if schedule == "off":
        return {"imit_ep_length_ema": ep_length_ema}

    iter_count = int(env.common_step_counter // max(num_steps_per_env, 1))

    if schedule == "iter":
        if iter_count < iter_thresholds[0]:
            base_phase = 0
        elif iter_count < iter_thresholds[1]:
            base_phase = 1
        else:
            base_phase = 2

        if min_ep_length_for_phase_advance is not None:
            thr = float(min_ep_length_for_phase_advance)
            if ep_length_ema < thr:
                phase = 0
            elif ep_length_ema < 2.0 * thr:
                phase = min(base_phase, 1)
            else:
                phase = base_phase
        else:
            phase = base_phase

    elif schedule == "metric":
        # Per-reset metric snapshot from the pingpong command. Same indexing
        # pattern as update_pingpong_curriculum so they see the same envs.
        command = env.command_manager.get_term(command_name)
        if isinstance(env_ids, slice):
            metric_ids = torch.arange(env.num_envs, device=env.device)
        elif isinstance(env_ids, torch.Tensor):
            metric_ids = env_ids.to(device=env.device, dtype=torch.long)
        else:
            metric_ids = torch.tensor(env_ids, dtype=torch.long, device=env.device)

        if metric_ids.numel() > 0:
            hsr_now = float(torch.mean(command.metrics["hit_success_rate"][metric_ids]).item())
            pos_fail_now = float(torch.mean(command.metrics["hit_success_pos_fail_rate"][metric_ids]).item())
            vel_fail_now = float(torch.mean(command.metrics["hit_success_vel_fail_rate"][metric_ids]).item())
            ori_fail_now = float(torch.mean(command.metrics["hit_success_ori_fail_rate"][metric_ids]).item())
            obs_now = {
                "hit_success_rate": hsr_now,
                "pos_success_rate": 1.0 - pos_fail_now,
                "vel_success_rate": 1.0 - vel_fail_now,
                "ori_success_rate": 1.0 - ori_fail_now,
            }
            if not _IMIT_METRIC_EMA["init"]:
                for k, v in obs_now.items():
                    _IMIT_METRIC_EMA[k] = v
                _IMIT_METRIC_EMA["init"] = True
            else:
                a = float(metric_ema_alpha)
                for k, v in obs_now.items():
                    _IMIT_METRIC_EMA[k] = (1.0 - a) * _IMIT_METRIC_EMA[k] + a * v

        if phase_thresholds is None:
            raise ValueError("schedule='metric' requires phase_thresholds")
        if len(phase_thresholds) != len(w_i_values) - 1:
            raise ValueError(
                f"phase_thresholds must have {len(w_i_values) - 1} entries "
                f"(one per phase transition); got {len(phase_thresholds)}"
            )

        # Walk the ladder: advance one phase per threshold dict that's met.
        # All listed keys (AND) must clear; missing keys default to 0.
        base_phase = 0
        for thr_dict in phase_thresholds:
            metric_ok = (
                _IMIT_METRIC_EMA["hit_success_rate"] >= float(thr_dict.get("hit_success_rate", 0.0))
                and _IMIT_METRIC_EMA["pos_success_rate"] >= float(thr_dict.get("pos_success_rate", 0.0))
                and _IMIT_METRIC_EMA["vel_success_rate"] >= float(thr_dict.get("vel_success_rate", 0.0))
                and _IMIT_METRIC_EMA["ori_success_rate"] >= float(thr_dict.get("ori_success_rate", 0.0))
                and ep_length_ema >= float(thr_dict.get("min_ep_length", 0.0))
            )
            if metric_ok:
                base_phase += 1
            else:
                break

        # Monotone latch: never regress.
        phase = max(base_phase, _IMIT_PHASE_LATCH["max_phase"])
        _IMIT_PHASE_LATCH["max_phase"] = phase

    else:
        raise ValueError(f"Unknown imitation schedule: {schedule}")

    w_i = float(w_i_values[phase])

    if split is None:
        split_map = _IMIT_SPLIT
    elif isinstance(split, str):
        if split not in _IMIT_SPLIT_PRESETS:
            raise ValueError(
                f"Unknown imitation split preset: {split!r}. "
                f"Available: {list(_IMIT_SPLIT_PRESETS)}"
            )
        split_map = _IMIT_SPLIT_PRESETS[split]
    else:
        split_map = split

    for term_name, share in split_map.items():
        env.reward_manager.get_term_cfg(term_name).weight = share * w_i

    return {
        "imit_w_i": w_i,
        "imit_phase": float(phase),
        "imit_base_phase": float(base_phase),
        "imit_iter": float(iter_count),
        "imit_ep_length_ema": ep_length_ema,
        "imit_min_ep_length_threshold": float(min_ep_length_for_phase_advance or 0.0),
        "imit_split_jp": float(split_map.get("imitation_joint_pos", 0.0)),
        "imit_split_bp": float(split_map.get("imitation_body_pos", 0.0)),
        "imit_metric_hsr_ema": float(_IMIT_METRIC_EMA["hit_success_rate"]),
        "imit_metric_pos_ema": float(_IMIT_METRIC_EMA["pos_success_rate"]),
        "imit_metric_vel_ema": float(_IMIT_METRIC_EMA["vel_success_rate"]),
        "imit_metric_ori_ema": float(_IMIT_METRIC_EMA["ori_success_rate"]),
        "imit_max_phase_latch": float(_IMIT_PHASE_LATCH["max_phase"]),
    }


def update_pingpong_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    command_name: str = "pingpong",
    enable_noise: bool = False,
    enable_range: bool = False,
    enable_y_curriculum: bool = True,
    enable_v_curriculum: bool = True,
    enable_window_curriculum: bool = True,
    min_ep_length_for_window_advance: float = 250.0,
    min_ep_length_for_ori_advance: float = 250.0,
    min_ep_length_for_pos_vel_advance: float = 250.0,
    cos_sim_freeze_threshold: float = 0.45,
    sequenced_curriculum: bool = True,
    v_unlock_shape_tier: int = 6,
    v_unlock_hsr: float = 0.85,
    v_unlock_cos_sim: float = 0.55,
    y_unlock_v_in_high: float = 3.5,
    y_unlock_hsr: float = 0.80,
    cos_sim_collapse_threshold: float = 0.35,
    cos_sim_collapse_retreat_v_in_high: float = 2.5,
    cos_sim_collapse_retreat_hit_y_half_w: float = 0.10,
) -> dict[str, float]:
    """Update v5.7 pingpong curricula from the command success metric.

    Range curriculum (when enable_range=True):
      - hit_y_range / hit_z_range driven by overall hit_success_rate
        (advances only when position+velocity+orientation all clear thresholds)
      - v_in_mag_range driven by BOTH hit_success_rate AND vel_success_rate
        (must clear both, i.e. min(hit_success, vel_success) >= threshold).
        This rules out the inflation case where vel_ok passes while the paddle
        is far from the ball, AND the case where pos is good but vel still loose.

    Switches:
      - enable_range: master toggle. False = no range adaptation.
      - enable_y_curriculum: only effective when enable_range=True.
      - enable_v_curriculum: only effective when enable_range=True.

    Window-curriculum stand-up gate (min_ep_length_for_window_advance):
      The window ratchet is monotone — once tier-1+ locks in (window 0.06,
      weights 3/3/1), there's no rollback. With signed-ori reward, hit_success
      can hit ~0.20 while ep_length is still ~40 (run 2026-05-25_10-08-03):
      the ratchet trips early, the stronger upper-body strike gradient
      out-competes the leg balance learning, and the EL=40→234 breakthrough
      that baseline 23-07-21 saw at iter ~1800 never happens. Holding tier-0
      until imit_ep_length_ema crosses the threshold preserves baseline-style
      shaping during stand-up. Reads the same EMA written by
      update_imitation_weight; 1-iter stale (CurriculumCfg order is pingpong→
      imit_anneal) but the EMA is slow (alpha=0.05) so the lag is invisible.

    Pos/Vel stand-up gate (min_ep_length_for_pos_vel_advance):
      Force goal_position / goal_velocity / goal_position_pre_strike /
      goal_velocity_pre_strike weights to 0 until EMA(EL) crosses threshold,
      monotone latch (once opened, never re-closes). Without it, those four
      rewards (initial weights 2.0/2.0/0.3/1.0) stay live during the stand-up
      phase, gated only by |t_to_hit| ≤ strike_window. After the M1 RSI
      base-yaw fix made blade orientation correct, run 2026-05-25_14-51-08
      stalled at EL≈41 for 1680+ iter — policy farmed pos/vel reward by
      swinging during falls instead of learning to stand. This gate restores
      the baseline 23-07-21 reward landscape (only imitation + alive +
      regularization) until the policy survives ~5s.
    """
    command: PingpongCommand = env.command_manager.get_term(command_name)
    command.finalize_partial_swings(env_ids)
    if isinstance(env_ids, slice):
        metric_ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, torch.Tensor):
        metric_ids = env_ids.to(device=env.device, dtype=torch.long)
    else:
        metric_ids = torch.tensor(env_ids, dtype=torch.long, device=env.device)
    success_rate = float(torch.mean(command.metrics["hit_success_rate"][metric_ids]).item())
    vel_fail_rate = float(torch.mean(command.metrics["hit_success_vel_fail_rate"][metric_ids]).item())
    vel_success_rate = 1.0 - vel_fail_rate

    # Multi-metric reward-shaping curriculum: tier advances only when ALL four
    # task EMAs (hsr / pos / vel / ori success rates) cross tier thresholds.
    # Replaces the prior single-hsr-gated sigma curriculum which let sigma_g_pos
    # drop to 0.15 at hsr=0.30 even with vel_fail=0.62 — pos was already passing
    # so tightening pos sigma was wasted while vel was starved (run
    # 2026-05-25_18-48-35 iter 1485: weighted ratio 117× wrong direction).
    # hsr_ema is the cold-start gate: pos/vel/ori EMAs init from first
    # observation (1.0 when no strikes happened yet) so without hsr gating,
    # tier jumps to 4 at iter ~50 (run 2026-05-25_19-44-44 bug).
    hsr_ema = float(_IMIT_METRIC_EMA["hit_success_rate"])
    pos_ema = float(_IMIT_METRIC_EMA["pos_success_rate"])
    vel_ema = float(_IMIT_METRIC_EMA["vel_success_rate"])
    ori_ema = float(_IMIT_METRIC_EMA["ori_success_rate"])
    new_shape_tier, sigma_target_new, std_vel_target_new, std_ori_target_new = _reward_shape_tier(hsr_ema, pos_ema, vel_ema, ori_ema)

    # v62 staggering: shape_tier and v_in_mag curricula must not advance in the
    # same iter window. If v_in_mag changed within the last cooldown period,
    # HOLD shape_tier at its previous tier (don't apply new tighter values).
    iter_count_now = int(env.common_step_counter // 24)  # approx PPO iter (matches imit_anneal default)
    prev_shape_tier = int(_SHAPE_TIER_LATCH["tier"])
    cooldown_active_for_shape = (iter_count_now - int(_V_IN_TIER_LATCH["last_change_iter"])) < int(_CROSS_CURRICULUM_COOLDOWN_ITERS)

    if new_shape_tier > prev_shape_tier and cooldown_active_for_shape:
        # Hold at previous tier — recompute targets at the previous tier index.
        # _REWARD_SHAPE_TIERS is ordered top→bottom, but `new_shape_tier` is index from top.
        # Recover prev tier values by passing through _reward_shape_tier with FORCED tier.
        # Simpler: use the tier table directly — table[len-1-prev_tier] gives prev tier params.
        held_idx = len(_REWARD_SHAPE_TIERS) - 1 - prev_shape_tier
        sigma_target = float(_REWARD_SHAPE_TIERS[held_idx][0])
        std_vel_target = float(_REWARD_SHAPE_TIERS[held_idx][1])
        std_ori_target = float(_REWARD_SHAPE_TIERS[held_idx][2])
        shape_tier = prev_shape_tier
    else:
        sigma_target = sigma_target_new
        std_vel_target = std_vel_target_new
        std_ori_target = std_ori_target_new
        shape_tier = new_shape_tier
        if new_shape_tier != prev_shape_tier:
            # Tier changed (up OR down — latch only ratchets up via max() below)
            if new_shape_tier > prev_shape_tier:
                _SHAPE_TIER_LATCH["last_change_iter"] = iter_count_now
            _SHAPE_TIER_LATCH["tier"] = new_shape_tier

    # v64: monotone σ latch (re-added, was removed in v60). Once tightened, never
    # loosen — see _SIGMA_LATCH. Breaks the shape_tier 4<->5 limit cycle so σ keeps
    # pushing the face past the 0.80 plateau. Floors unchanged.
    _SIGMA_LATCH["sigma_g_pos"] = min(_SIGMA_LATCH["sigma_g_pos"], max(sigma_target, 0.06))
    command.cfg.sigma_g_pos = _SIGMA_LATCH["sigma_g_pos"]

    gv_cfg = env.reward_manager.get_term_cfg("goal_velocity")
    _SIGMA_LATCH["std_g_vel"] = min(_SIGMA_LATCH["std_g_vel"], max(std_vel_target, 0.20))
    gv_cfg.params["std"] = _SIGMA_LATCH["std_g_vel"]

    # goal_orientation_pre_strike keeps its own fixed std (0.4).
    # σ_ori floor is an A/B toggle (_SIGMA_ORI_FLOOR, env PINGPONG_SIGMA_ORI_FLOOR):
    # 0.15 = no σ-ease (default control arm); 0.20 = σ-ease (eases the over-pressure
    # that pushed face to 0.83 but dropped hsr to ~0.69 / shape_tier 5→3).
    go_cfg = env.reward_manager.get_term_cfg("goal_orientation")
    _SIGMA_LATCH["std_g_ori"] = min(_SIGMA_LATCH["std_g_ori"], max(std_ori_target, _SIGMA_ORI_FLOOR))
    go_cfg.params["std"] = _SIGMA_LATCH["std_g_ori"]

    if enable_noise:
        if success_rate >= 0.50:
            command.cfg.noise_t_sigma = max(command.cfg.noise_t_sigma, 0.005)
        if success_rate >= 0.75:
            command.cfg.noise_p_sigma = max(command.cfg.noise_p_sigma, 0.005)
            command.cfg.noise_v_sigma = max(command.cfg.noise_v_sigma, 0.05)
            command.cfg.noise_base_sigma = max(command.cfg.noise_base_sigma, 0.015)

    # cos_sim EMA (must be computed BEFORE the y/v curriculum gates so the
    # sequenced-unlock logic can read it). Freeze + collapse retreat live further
    # below in the window-curriculum block.
    # v64: gate on the STRIKE-INSTANT signed cos (~0.8, the true contact face), not
    # the deflated current-frame cos (~0.46) that stalled Stage-2 unlock (needs
    # cos_sim_ema>=0.55) and flickered the freeze at 0.45.
    cos_sim_now = float(torch.mean(command.metrics["cos_sim_at_strike"][metric_ids]).item())
    if not _COS_SIM_EMA["init"]:
        _COS_SIM_EMA["value"] = cos_sim_now
        _COS_SIM_EMA["init"] = True
    else:
        _COS_SIM_EMA["value"] = 0.95 * _COS_SIM_EMA["value"] + 0.05 * cos_sim_now
    cos_sim_ema = float(_COS_SIM_EMA["value"])
    cos_sim_ratchet_freeze = cos_sim_ema < cos_sim_freeze_threshold

    # Sequenced curriculum gates (Stage 1 → 2 → 3): each layer unlocks only
    # after the prior layer proves healthy. Run 2026-05-26_20-52-38 collapsed
    # at iter ~14k because v_in_mag and hit_y advanced in parallel with the
    # window curriculum: while shape_tier was still at 1.3, v_in had already
    # been pushed to 2.71 and the policy hit a vel_fail=0.75 / hsr=0.17 reward-
    # hacking corner (only追 base_pos, abandoned vel/ori/imit). Stage gates:
    #   Stage 1 — only window curriculum runs. v_in & y locked at defaults.
    #   Stage 2 — v_in unlocks once shape_tier ≥ v_unlock_shape_tier AND
    #             hsr_ema ≥ v_unlock_hsr AND cos_sim_ema ≥ v_unlock_cos_sim.
    #   Stage 3 — y unlocks once Stage 2 has driven v_in_high to
    #             y_unlock_v_in_high AND hsr_ema ≥ y_unlock_hsr.
    # Each gate is checked every iter (no latch); if a metric regresses, the
    # tier curricula simply stop advancing — combined with the collapse-retreat
    # below, this gives the policy room to recover.
    v_curriculum_unlocked = (not sequenced_curriculum) or (
        shape_tier >= int(v_unlock_shape_tier)
        and hsr_ema >= float(v_unlock_hsr)
        and cos_sim_ema >= float(v_unlock_cos_sim)
    )
    v_in_mag_high_now = float(command.cfg.v_in_mag_range[1])
    y_curriculum_unlocked = (not sequenced_curriculum) or (
        v_curriculum_unlocked
        and v_in_mag_high_now >= float(y_unlock_v_in_high)
        and hsr_ema >= float(y_unlock_hsr)
    )
    effective_v_curriculum = bool(enable_v_curriculum) and bool(v_curriculum_unlocked)
    effective_y_curriculum = bool(enable_y_curriculum) and bool(y_curriculum_unlocked)

    # cos_sim collapse retreat — the sibling of cos_sim_ratchet_freeze. When
    # cos_sim_ema falls deep below the freeze threshold (default 0.35), the
    # policy is in a wrong-face basin; freezing alone is not enough because
    # v_in_mag and hit_y stay at their last-assigned tier, continuing to push
    # the policy. Actively roll v_in_high back to the safest tier and hit_y
    # back to the narrowest band so the policy can re-find the orientation
    # gradient. Reverse-ratchet (overrides the existing monotone tiers).
    cos_sim_collapsed = cos_sim_ema < float(cos_sim_collapse_threshold)
    if cos_sim_collapsed:
        retreat_v_high = float(cos_sim_collapse_retreat_v_in_high)
        if v_in_mag_high_now > retreat_v_high:
            command.cfg.v_in_mag_range = (float(command.cfg.v_in_mag_range[0]), retreat_v_high)
        # v60: retreat shrinks WORLD CAP to the initial (narrowest) value.
        # Tighten only if current cap is wider than initial.
        cap_initial = float(command.cfg.hit_y_world_cap_initial)
        if float(command.cfg.hit_y_world_cap) > cap_initial:
            command.cfg.hit_y_world_cap = cap_initial

    if enable_range:
        # y / z spatial curriculum — gated by overall hit_success_rate.
        # v60: WORLD CAP driven directly (env-local |hit_y_world - env.y| ≤ cap).
        # Cap grows from cap_initial (e.g. 0.30) to cap_max (e.g. 1.00) as the
        # policy succeeds. NO base-frame range curriculum — base half-widths
        # come from divider geometry inside _sample_new_swing.
        if effective_y_curriculum:
            cap_max = float(command.cfg.hit_y_world_cap_max)
            cap_initial = float(command.cfg.hit_y_world_cap_initial)
            # Tier ladder for hit_y_world_cap (interpolating cap_initial → cap_max):
            # tier 0 (init):       cap = cap_initial               (narrowest, demo-near)
            # tier 1 (hsr ≥ 0.30): cap = lerp(0.20)
            # tier 2 (hsr ≥ 0.50): cap = lerp(0.45)
            # tier 3 (hsr ≥ 0.75): cap = lerp(0.75)
            # tier 4 (hsr ≥ 0.90): cap = cap_max                   (widest)
            def _tier_cap(t: float) -> float:
                # t in [0, 1] interpolates between cap_initial and cap_max
                return cap_initial + t * (cap_max - cap_initial)

            if success_rate >= 0.90:
                command.cfg.hit_y_world_cap = cap_max
                command.cfg.hit_z_range = (0.85, 1.25)
            elif success_rate >= 0.75:
                command.cfg.hit_y_world_cap = _tier_cap(0.75)
                command.cfg.hit_z_range = (0.85, 1.25)
            elif success_rate >= 0.50:
                command.cfg.hit_y_world_cap = _tier_cap(0.45)
                command.cfg.hit_z_range = (0.88, 1.25)
            elif success_rate >= 0.30:
                command.cfg.hit_y_world_cap = _tier_cap(0.20)
                command.cfg.hit_z_range = (0.92, 1.25)

        # v_in_mag curriculum — gated by BOTH hit_success_rate AND vel_success_rate.
        # Must clear both metrics at each tier, equivalent to gating on
        # min(hit_success_rate, vel_success_rate). This avoids:
        #   (a) vel_success inflation while paddle is far from ball (pos_fail high)
        #   (b) advancing when pos is OK but velocity tracking is still loose
        if effective_v_curriculum:
            v_gate = min(success_rate, vel_success_rate)
            # v62 staggering: compute new v_in_high target, then check if shape_tier
            # changed within cooldown — if yes, HOLD at previous v_in_high to avoid
            # simultaneous σ_vel tightening + ball-speed increase.
            if v_gate >= 0.90:
                new_v_in_high = 4.0
            elif v_gate >= 0.75:
                new_v_in_high = 3.5
            elif v_gate >= 0.50:
                new_v_in_high = 3.0
            elif v_gate >= 0.30:
                new_v_in_high = 2.5
            else:
                new_v_in_high = float(command.cfg.v_in_mag_range[1])  # no change

            prev_v_in_high = float(_V_IN_TIER_LATCH["high"])
            cooldown_active_for_v_in = (iter_count_now - int(_SHAPE_TIER_LATCH["last_change_iter"])) < int(_CROSS_CURRICULUM_COOLDOWN_ITERS)

            if new_v_in_high > prev_v_in_high and cooldown_active_for_v_in:
                # Hold v_in at previous value (don't increase during shape cooldown)
                pass  # leave v_in_mag_range unchanged
            else:
                command.cfg.v_in_mag_range = (1.5, new_v_in_high)
                if new_v_in_high > prev_v_in_high:
                    _V_IN_TIER_LATCH["last_change_iter"] = iter_count_now
                    _V_IN_TIER_LATCH["high"] = new_v_in_high
                elif new_v_in_high < prev_v_in_high:
                    _V_IN_TIER_LATCH["high"] = new_v_in_high  # allow decrease without cooldown

    # Strike-window curriculum: shrink window + ramp weights together.
    # Gated on hit_success_rate AND ep_length EMA (stand-up gate).
    window_ep_ema = float(_EP_LENGTH_EMA["value"]) if _EP_LENGTH_EMA["init"] else 0.0
    window_gate_open = window_ep_ema >= float(min_ep_length_for_window_advance)

    # v61: removed swing_p_forehand warmup curriculum (90:10 → 50:50 latch).
    # The 3-phase task curriculum (update_task_phase) handles single-task →
    # dual-task progression via stand → imit → strike phases. swing_p_forehand
    # is now fixed at 0.50 (paper design) throughout.

    # Ori-reward stand-up gate (monotone latch): force goal_orientation.weight
    # to 0 until EMA crosses min_ep_length_for_ori_advance, then never re-close.
    # This removes the swing-while-falling basin: signed-ori provides a clean
    # paddle-direction gradient that — combined with goal_velocity (lenient
    # std=0.5) — lets the policy collect task reward while never standing.
    # Once the latch opens, the window curriculum's monotone max() raises the
    # weight back to whatever tier applies.
    #
    # NOTE: goal_orientation_pre_strike is gated by _POS_VEL_GATE_LATCH below,
    # NOT by this latch — pre_strike opening is unified across all three
    # task channels (pos/vel/ori) for structural consistency. This latch only
    # covers the strike-window goal_orientation, which has the specific
    # swing-while-falling rationale captured in the docstring above.
    if not _ORI_GATE_LATCH["open"]:
        if window_ep_ema >= float(min_ep_length_for_ori_advance):
            _ORI_GATE_LATCH["open"] = True
        else:
            env.reward_manager.get_term_cfg("goal_orientation").weight = 0.0
    ori_gate_open = bool(_ORI_GATE_LATCH["open"])

    # Pos/Vel stand-up gate (monotone latch). Mirrors ori_gate but covers the
    # much larger goal_position / goal_velocity rewards (initial weights 2.0
    # each) plus all three pre_strike variants (pos 0.3 / vel 1.0 / ori 0.5).
    # With M1 RSI base-yaw fix the blade is correctly oriented in world frame,
    # so any incidental swing during fall collects pos/vel reward — driving a
    # swing-while-falling basin (run 2026-05-25_14-51-08: EL=41 stuck for 1680+
    # iter, goal_velocity reward 14× baseline). Force these weights to 0 until
    # EMA(EL) crosses threshold; once opened, the window curriculum's monotone
    # max() ratchet raises goal_position / goal_velocity back, but the
    # pre_strike trio has no curriculum to manage them — restore those from
    # captured env_cfg values when the latch flips open.
    #
    # All three pre_strike rewards (pos/vel/ori) are gated by THIS latch (not
    # _ORI_GATE_LATCH) so that their opening is structurally identical: same
    # threshold, same restore mechanism, same monotone latch. Even though
    # min_ep_length_for_ori_advance == min_ep_length_for_pos_vel_advance == 250
    # makes them fire at the same EL, sharing one latch is the right contract:
    # "task pre_strike rewards open together when the policy can stand."
    pos_vel_terms = (
        "goal_position",
        "goal_velocity",
        "goal_position_pre_strike",
        "goal_velocity_pre_strike",
        "goal_orientation_pre_strike",
    )
    pre_strike_restore = (
        "goal_position_pre_strike",
        "goal_velocity_pre_strike",
        "goal_orientation_pre_strike",
    )
    if _POS_VEL_GATE_LATCH["original_weights"] is None:
        _POS_VEL_GATE_LATCH["original_weights"] = {
            name: float(env.reward_manager.get_term_cfg(name).weight) for name in pos_vel_terms
        }
    if not _POS_VEL_GATE_LATCH["open"]:
        if window_ep_ema >= float(min_ep_length_for_pos_vel_advance):
            _POS_VEL_GATE_LATCH["open"] = True
            for name in pre_strike_restore:
                env.reward_manager.get_term_cfg(name).weight = _POS_VEL_GATE_LATCH["original_weights"][name]
        else:
            for term_name in pos_vel_terms:
                env.reward_manager.get_term_cfg(term_name).weight = 0.0
    pos_vel_gate_open = bool(_POS_VEL_GATE_LATCH["open"])

    # goal_base smooth ramp: linear interpolation from start_weight to
    # target_weight as window_ep_ema grows from ep_lo to ep_hi. Captures
    # target_weight from env_cfg on first call. Always active (no binary
    # gate) — see _GOAL_BASE_RAMP docstring above for rationale.
    if _GOAL_BASE_RAMP["target_weight"] is None:
        _GOAL_BASE_RAMP["target_weight"] = float(env.reward_manager.get_term_cfg("goal_base").weight)
    _gb_lo = float(_GOAL_BASE_RAMP["ep_lo"])
    _gb_hi = float(_GOAL_BASE_RAMP["ep_hi"])
    _gb_start = float(_GOAL_BASE_RAMP["start_weight"])
    _gb_target = float(_GOAL_BASE_RAMP["target_weight"])
    _gb_ratio = max(0.0, min(1.0, (window_ep_ema - _gb_lo) / max(1e-6, _gb_hi - _gb_lo)))
    goal_base_weight_now = _gb_start + _gb_ratio * (_gb_target - _gb_start)
    env.reward_manager.get_term_cfg("goal_base").weight = goal_base_weight_now

    # D2 cos_sim guardrail: cos_sim_ema and cos_sim_ratchet_freeze are
    # computed earlier in this function (right before enable_range) so the
    # sequenced-curriculum gates can reference them. Original docstring:
    # When EMA dips below the danger threshold, freeze the window-curriculum
    # ratchet. The cos basin half-width is at cos≈0.45 (cos_dist=0.55,
    # std=0.30 → exp(-1) ≈ 0.37 satisfaction); going below that means the
    # policy is wandering into the flat-zero gradient zone and further window
    # tightening would strand the orientation channel. Same alpha as
    # _IMIT_METRIC_EMA so they respond on similar timescales.

    # D1 multi-EMA gate: each tier requires hsr/pos/vel/ori EMAs all clear
    # tier-specific thresholds. Replaces the prior batch-noise-driven ratchet
    # that took run 2026-05-25_22-50-41 to top tier (window=0.01, w=12/12/4)
    # at iter 2424 when global hsr was only 0.42 — see
    # _WINDOW_CURRICULUM_TIERS docstring for the full forensic trail.
    if enable_window_curriculum and window_gate_open and not cos_sim_ratchet_freeze:
        for hsr_thr, pos_thr, vel_thr, ori_thr, window_s, w_pos, w_vel, w_ori in _WINDOW_CURRICULUM_TIERS:
            if (
                hsr_ema >= hsr_thr
                and pos_ema >= pos_thr
                and vel_ema >= vel_thr
                and ori_ema >= ori_thr
            ):
                # Monotone shrink: never widen the gate once tightened.
                command.cfg.strike_window = min(command.cfg.strike_window, window_s)
                # Monotone weight ramp: never lower a tier's weight once raised.
                env.reward_manager.get_term_cfg("goal_position").weight = max(
                    env.reward_manager.get_term_cfg("goal_position").weight, w_pos
                )
                env.reward_manager.get_term_cfg("goal_velocity").weight = max(
                    env.reward_manager.get_term_cfg("goal_velocity").weight, w_vel
                )
                env.reward_manager.get_term_cfg("goal_orientation").weight = max(
                    env.reward_manager.get_term_cfg("goal_orientation").weight, w_ori
                )
                break

    return {
        "hit_success_rate": success_rate,
        "vel_success_rate": vel_success_rate,
        "v_curriculum_gate": min(success_rate, vel_success_rate),
        "sigma_g_pos": float(command.cfg.sigma_g_pos),
        "std_g_vel": float(env.reward_manager.get_term_cfg("goal_velocity").params["std"]),
        "std_g_ori": float(env.reward_manager.get_term_cfg("goal_orientation").params["std"]),
        "shape_tier": float(shape_tier),
        "shape_hsr_ema": hsr_ema,
        "shape_pos_ema": pos_ema,
        "shape_vel_ema": vel_ema,
        "shape_ori_ema": ori_ema,
        "noise_p_sigma": float(command.cfg.noise_p_sigma),
        "noise_v_sigma": float(command.cfg.noise_v_sigma),
        "noise_base_sigma": float(command.cfg.noise_base_sigma),
        "noise_t_sigma": float(command.cfg.noise_t_sigma),
        "hit_y_max": float(command.cfg.hit_y_world_cap),
        "hit_y_min": float(-command.cfg.hit_y_world_cap),
        "hit_y_world_cap": float(command.cfg.hit_y_world_cap),
        # DEBUG diagnostics for goal_base reward
        "diag_goal_base_err_mean": float(getattr(command, "_goal_base_err_diag", torch.zeros(1, device=command.device)).mean().item()),
        "diag_goal_base_err_max": float(getattr(command, "_goal_base_err_diag", torch.zeros(1, device=command.device)).max().item()),
        "diag_goal_base_gate_mean": float(getattr(command, "_goal_base_gate_diag", torch.zeros(1, device=command.device)).mean().item()),
        "diag_root_x_mean": float(getattr(command, "_goal_base_root_x_diag", torch.zeros(1, device=command.device)).mean().item()),
        "diag_root_y_mean": float(getattr(command, "_goal_base_root_y_diag", torch.zeros(1, device=command.device)).mean().item()),
        "diag_root_x_absmax": float(getattr(command, "_goal_base_root_x_diag", torch.zeros(1, device=command.device)).abs().max().item()),
        "diag_target_x_mean": float(getattr(command, "_goal_base_target_x_diag", torch.zeros(1, device=command.device)).mean().item()),
        "diag_target_y_mean": float(getattr(command, "_goal_base_target_y_diag", torch.zeros(1, device=command.device)).mean().item()),
        "diag_target_x_absmax": float(getattr(command, "_goal_base_target_x_diag", torch.zeros(1, device=command.device)).abs().max().item()),
        "diag_phit_x_mean": float(getattr(command, "_goal_base_phit_x_diag", torch.zeros(1, device=command.device)).mean().item()),
        "diag_phit_y_mean": float(getattr(command, "_goal_base_phit_y_diag", torch.zeros(1, device=command.device)).mean().item()),
        "diag_phit_x_absmax": float(getattr(command, "_goal_base_phit_x_diag", torch.zeros(1, device=command.device)).abs().max().item()),
        # KEY: per-env delta absmax — diagnoses coordinate-frame mismatch
        "diag_delta_x_absmax": float(getattr(command, "_goal_base_delta_x_absmax_diag", torch.zeros(1, device=command.device)).item()),
        "diag_delta_y_absmax": float(getattr(command, "_goal_base_delta_y_absmax_diag", torch.zeros(1, device=command.device)).item()),
        "diag_delta_x_mean": float(getattr(command, "_goal_base_delta_x_mean_diag", torch.zeros(1, device=command.device)).item()),
        "diag_delta_y_mean": float(getattr(command, "_goal_base_delta_y_mean_diag", torch.zeros(1, device=command.device)).item()),
        "diag_env_origin_x_absmax": float(getattr(command, "_goal_base_env_origin_x_absmax_diag", torch.zeros(1, device=command.device)).item()),
        "diag_env_origin_y_absmax": float(getattr(command, "_goal_base_env_origin_y_absmax_diag", torch.zeros(1, device=command.device)).item()),
        # KEY: per-axis absmax to localize the Y-axis bug
        "diag_root_x_absmax_v2": float(getattr(command, "_goal_base_root_x_absmax_diag", torch.zeros(1, device=command.device)).item()),
        "diag_root_y_absmax": float(getattr(command, "_goal_base_root_y_absmax_diag", torch.zeros(1, device=command.device)).item()),
        "diag_target_y_absmax": float(getattr(command, "_goal_base_target_y_absmax_diag", torch.zeros(1, device=command.device)).item()),
        "diag_phit_y_absmax": float(getattr(command, "_goal_base_phit_y_absmax_diag", torch.zeros(1, device=command.device)).item()),
        "hit_z_low": float(command.cfg.hit_z_range[0]),
        "v_in_mag_high": float(command.cfg.v_in_mag_range[1]),
        "strike_window": float(command.cfg.strike_window),
        "w_goal_pos": float(env.reward_manager.get_term_cfg("goal_position").weight),
        "w_goal_vel": float(env.reward_manager.get_term_cfg("goal_velocity").weight),
        "w_goal_ori": float(env.reward_manager.get_term_cfg("goal_orientation").weight),
        "w_goal_base": goal_base_weight_now,
        "w_goal_base_ramp_ratio": _gb_ratio,
        "window_gate_ep_ema": window_ep_ema,
        "window_gate_open": float(window_gate_open),
        "window_gate_threshold": float(min_ep_length_for_window_advance),
        "ori_gate_open": float(ori_gate_open),
        "ori_gate_threshold": float(min_ep_length_for_ori_advance),
        "pos_vel_gate_open": float(pos_vel_gate_open),
        "pos_vel_gate_threshold": float(min_ep_length_for_pos_vel_advance),
        "w_goal_pos_pre": float(env.reward_manager.get_term_cfg("goal_position_pre_strike").weight),
        "w_goal_vel_pre": float(env.reward_manager.get_term_cfg("goal_velocity_pre_strike").weight),
        "w_goal_ori_pre": float(env.reward_manager.get_term_cfg("goal_orientation_pre_strike").weight),
        "cos_sim_ema": cos_sim_ema,
        "cos_sim_ratchet_freeze": float(cos_sim_ratchet_freeze),
        "v_curriculum_unlocked": float(v_curriculum_unlocked),
        "y_curriculum_unlocked": float(y_curriculum_unlocked),
        "cos_sim_collapsed": float(cos_sim_collapsed),
    }


# ---------------------------------------------------------------------------
# Table guard curriculum (R8): hide the table during stand-up + swing-learning,
# then teleport it back when the policy can stand and hit reliably.
#
# Stage machine:
#   0  hidden     table sunk to z=-10 (cannot collide / cannot be used as
#                 mechanical-balance support); paddle/body table_contact
#                 weights forced to 0; non_paddle_table_stuck termination
#                 short-circuits to zeros via _pingpong_table_active flag.
#   1  unlocked  unlock condition met; flag flipped True. Each env's table is
#                 teleported to (1.77, 0, 0.735) on its NEXT reset by the
#                 reset_table_position_by_stage EventTerm. No active teleport
#                 to avoid slamming the table into a paddle mid-swing.
#   2  ramping   weights linearly grow from 0 to target over `ramp_iters`,
#                 giving the policy time to learn collision avoidance.
#   3  active    weights at target; non_paddle_table_stuck termination active;
#                 system equivalent to the standard HITTER from-scratch setup.
#
# Unlock conditions (ALL of, batch-mean EMAs):
#   hsr_ema           >= min_hsr_ema           (truly hitting, not just standing)
#   cos_sim_ema       >= min_cos_sim_ema       (paddle alignment learnt)
#   ep_length_ema     >= min_ep_length_ema     (robot stands stably)
#   iter              >= min_iter              (EMA noise floor)
#
# Run `update_table_guard_stage` AFTER `pingpong` in the CurriculumCfg, so the
# pingpong term has already refreshed _COS_SIM_EMA and _IMIT_METRIC_EMA this tick.
_TABLE_GUARD: dict = {
    "stage": 0,
    "iter_at_unlock": -1,
    # Default unlock thresholds (override via params= on the CurrTerm).
    "min_hsr_ema": 0.65,
    "min_cos_sim_ema": 0.50,
    "min_ep_length_ema": 400.0,
    "min_iter": 1500,
    # Ramp config (override via params=).
    "ramp_iters": 500,
    "target_paddle_weight": -10.0,
    "target_body_weight": -1.0,
}


def update_table_guard_stage(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    num_steps_per_env: int = 24,
    min_hsr_ema: float = 0.65,
    min_cos_sim_ema: float = 0.50,
    min_ep_length_ema: float = 400.0,
    min_iter: int = 1500,
    ramp_iters: int = 500,
    target_paddle_weight: float = -10.0,
    target_body_weight: float = -1.0,
    paddle_term_name: str = "paddle_table_contact",
    body_term_name: str = "body_table_contact",
) -> dict:
    """Advance the table-guard stage machine and ramp contact-penalty weights.

    See module docstring above for the stage machine.
    """
    # Sync runtime params into module dict (so the unlock thresholds can be
    # tuned from CurriculumCfg without restarting Python).
    state = _TABLE_GUARD
    state["min_hsr_ema"] = float(min_hsr_ema)
    state["min_cos_sim_ema"] = float(min_cos_sim_ema)
    state["min_ep_length_ema"] = float(min_ep_length_ema)
    state["min_iter"] = int(min_iter)
    state["ramp_iters"] = int(ramp_iters)
    state["target_paddle_weight"] = float(target_paddle_weight)
    state["target_body_weight"] = float(target_body_weight)

    iter_count = int(env.common_step_counter // max(int(num_steps_per_env), 1))

    hsr_ema = float(_IMIT_METRIC_EMA["hit_success_rate"]) if _IMIT_METRIC_EMA["init"] else 0.0
    cos_sim_ema = float(_COS_SIM_EMA["value"]) if _COS_SIM_EMA["init"] else 0.0
    ep_length_ema = float(_EP_LENGTH_EMA["value"]) if _EP_LENGTH_EMA["init"] else 0.0

    # Stage 0 -> 1 transition
    if state["stage"] == 0:
        unlock = (
            hsr_ema >= state["min_hsr_ema"]
            and cos_sim_ema >= state["min_cos_sim_ema"]
            and ep_length_ema >= state["min_ep_length_ema"]
            and iter_count >= state["min_iter"]
        )
        if unlock:
            state["stage"] = 1
            state["iter_at_unlock"] = iter_count
            env._pingpong_table_active = True

    # Stage 1: wait until ALL envs have reset at least once after unlock
    # (so the table is at active position everywhere). We approximate this by
    # checking that the smallest episode_length_buf is below one step_dt's worth
    # of steps after the unlock — i.e., at least one full reset cycle has
    # completed. Conservative fallback: hold Stage 1 for `ramp_iters / 4` iter.
    if state["stage"] == 1:
        iters_since_unlock = iter_count - int(state["iter_at_unlock"])
        # After ~ 1/4 of ramp window, every env should have reset at least once
        # (typical episode length is 500 steps = 24 iter @ num_steps_per_env=24,
        # so ramp_iters/4 = 125 iter is ample).
        if iters_since_unlock >= max(1, state["ramp_iters"] // 4):
            state["stage"] = 2

    # Stage 2: ramp weights
    if state["stage"] >= 2:
        iters_since_unlock = iter_count - int(state["iter_at_unlock"])
        progress = max(0.0, min(1.0, iters_since_unlock / max(1, state["ramp_iters"])))
        try:
            env.reward_manager.get_term_cfg(paddle_term_name).weight = progress * state["target_paddle_weight"]
            env.reward_manager.get_term_cfg(body_term_name).weight = progress * state["target_body_weight"]
        except (KeyError, AttributeError):
            progress = 0.0
        if progress >= 1.0 and state["stage"] == 2:
            state["stage"] = 3
    else:
        # Force weights to 0 in Stage 0/1 (defensive — env_cfg should also be 0).
        try:
            env.reward_manager.get_term_cfg(paddle_term_name).weight = 0.0
            env.reward_manager.get_term_cfg(body_term_name).weight = 0.0
        except (KeyError, AttributeError):
            pass

    # Diagnostics — easy to chart in TensorBoard.
    try:
        w_paddle = float(env.reward_manager.get_term_cfg(paddle_term_name).weight)
        w_body = float(env.reward_manager.get_term_cfg(body_term_name).weight)
    except (KeyError, AttributeError):
        w_paddle = 0.0
        w_body = 0.0

    return {
        "table_stage": float(state["stage"]),
        "table_iter_at_unlock": float(state["iter_at_unlock"]),
        "table_w_paddle": w_paddle,
        "table_w_body": w_body,
        "table_active_flag": float(bool(getattr(env, "_pingpong_table_active", False))),
        "table_unlock_hsr_ema": hsr_ema,
        "table_unlock_cos_sim_ema": cos_sim_ema,
        "table_unlock_ep_length_ema": ep_length_ema,
    }
