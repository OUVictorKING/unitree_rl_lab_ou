// Copyright (c) 2026 — see BallTrajFilter.h for design + integration notes.

#include "BallTrajFilter.h"

#include <algorithm>
#include <limits>

BallTrajFilter::BallTrajFilter(int   max_samples,
                               int   poly_deg,
                               float bounce_vz_thresh,
                               float table_top_z_world,
                               float ball_radius,
                               float gap_reset_s,
                               int   bounce_lookback)
    : max_samples_(std::max(3, max_samples)),
      poly_deg_(std::max(1, poly_deg)),
      bounce_vz_thresh_(bounce_vz_thresh),
      table_top_z_(table_top_z_world),
      ball_radius_(ball_radius),
      gap_reset_s_(std::max(0.0f, gap_reset_s)),
      bounce_lookback_(std::max(3, bounce_lookback))
{
}

void BallTrajFilter::push_sample(TimePoint t, const Eigen::Vector3f &p_world)
{
    std::lock_guard<std::mutex> lock(mtx_);

    // ── Out-of-order guard ──
    // If the new sample's timestamp is BEFORE the latest one already in the
    // buffer, the network/transport delivered packets out of order. LSQ on
    // a non-monotonic τ vector still works mathematically, but the bounce
    // detector and gap detector both assume monotonic time. Drop the sample
    // rather than corrupting the buffer.
    if (!buf_.empty() && t < buf_.back().t)
    {
        out_of_order_count_ += 1;
        return;
    }

    // ── Gap-reset guard ──
    // A large gap between the previous sample and this one (e.g. sensor
    // freeze, network buffering, ball briefly off-camera) means the OLD
    // 30 samples are stale relative to the new one. Polyfitting them all
    // together would give a wildly wrong (p, v, a). Clear the buffer so
    // the filter restarts from this fresh sample.
    if (!buf_.empty())
    {
        const float dt =
            std::chrono::duration<float>(t - buf_.back().t).count();
        if (dt > gap_reset_s_)
        {
            buf_.clear();
            last_bounce_idx_ = -1;
            gap_reset_count_ += 1;
        }
    }

    buf_.push_back({t, p_world});

    // Drop oldest if buffer overflows; track index shift.
    while (static_cast<int>(buf_.size()) > max_samples_)
    {
        buf_.pop_front();
        if (last_bounce_idx_ > 0)
            last_bounce_idx_ -= 1;
        else if (last_bounce_idx_ == 0)
            last_bounce_idx_ = -1;  // bounce frame just rolled out — fit window grows again
    }
    detect_bounce_locked_();
}

void BallTrajFilter::reset()
{
    std::lock_guard<std::mutex> lock(mtx_);
    buf_.clear();
    last_bounce_idx_ = -1;
    gap_reset_count_ = 0;
    out_of_order_count_ = 0;
}

int BallTrajFilter::sample_count() const
{
    std::lock_guard<std::mutex> lock(mtx_);
    return static_cast<int>(buf_.size());
}

int BallTrajFilter::last_bounce_index() const
{
    std::lock_guard<std::mutex> lock(mtx_);
    return last_bounce_idx_;
}

bool BallTrajFilter::has_recent_bounce() const
{
    std::lock_guard<std::mutex> lock(mtx_);
    return last_bounce_idx_ >= 0;
}

int BallTrajFilter::num_gap_resets() const
{
    std::lock_guard<std::mutex> lock(mtx_);
    return gap_reset_count_;
}

int BallTrajFilter::num_out_of_order() const
{
    std::lock_guard<std::mutex> lock(mtx_);
    return out_of_order_count_;
}

int BallTrajFilter::num_dropped_total() const
{
    std::lock_guard<std::mutex> lock(mtx_);
    return gap_reset_count_ + out_of_order_count_;
}

void BallTrajFilter::detect_bounce_locked_()
{
    // Robust bounce detection: scan the last `bounce_lookback_` (default 5)
    // samples for any window where vz reverses through ±bounce_vz_thresh
    // AND the local-minimum z is near the table. This survives a single
    // dropped frame at the bounce instant — if frame i+1 (the bounce frame)
    // is missing, the i→i+2 vz computation still flips sign, just with a
    // slightly wider gap. Without this widening the old 3-sample test missed
    // the bounce whenever the bounce-frame packet got dropped.
    const int n = static_cast<int>(buf_.size());
    if (n < 3)
        return;

    const int scan_start = std::max(0, n - bounce_lookback_);
    int       best_bounce_idx = -1;
    float     best_zj = std::numeric_limits<float>::max();

    for (int i = scan_start; i + 2 < n; ++i)
    {
        const float dt1 =
            std::chrono::duration<float>(buf_[i + 1].t - buf_[i].t).count();
        const float dt2 =
            std::chrono::duration<float>(buf_[i + 2].t - buf_[i + 1].t).count();
        if (dt1 < 1e-6f || dt2 < 1e-6f)
            continue;

        const float vz_a = (buf_[i + 1].p.z() - buf_[i].p.z()) / dt1;
        const float vz_b = (buf_[i + 2].p.z() - buf_[i + 1].p.z()) / dt2;

        // vz reversal through threshold AND local z minimum near table top.
        if (vz_a < -bounce_vz_thresh_ && vz_b > bounce_vz_thresh_)
        {
            const float zj = buf_[i + 1].p.z();
            if (zj < table_top_z_ + ball_radius_ + 0.05f && zj < best_zj)
            {
                best_bounce_idx = i + 1;
                best_zj = zj;
            }
        }
    }

    // Only OVERWRITE last_bounce_idx_ if the bounce we just found is NEWER
    // than the one already recorded — this avoids re-detecting the same
    // bounce on every push.
    if (best_bounce_idx > last_bounce_idx_)
        last_bounce_idx_ = best_bounce_idx;
}

std::optional<BallTrajFilter::Estimate> BallTrajFilter::estimate() const
{
    std::lock_guard<std::mutex> lock(mtx_);
    const int n = static_cast<int>(buf_.size());
    if (n < poly_deg_ + 1)
        return std::nullopt;  // filter warmup — fewer than 3 samples

    // Window restricted to post-bounce samples (paper Sec.IV-A: buffer cleared
    // at bounce). If no bounce, use everything.
    const int win_lo = std::max(0, last_bounce_idx_ + 1);
    const int win_hi = n;
    const int idx    = n - 1;  // we evaluate the polynomial at the latest sample

    if (win_hi - win_lo < poly_deg_ + 1)
        return std::nullopt;  // not enough post-bounce data yet

    // Centre the 31-sample window on idx (mirrors _smooth_va in planner.py
    // — "nearest POLY_WIN samples to idx", clipped to [win_lo, win_hi)).
    const int half = max_samples_ / 2;
    int lo = std::max(win_lo, idx - half);
    int hi = std::min(win_hi, lo + max_samples_);
    lo     = std::max(win_lo, hi - max_samples_);
    const int win_n = hi - lo;
    if (win_n < poly_deg_ + 1)
        return std::nullopt;

    // Build A (n × P) and Y (n × 3) for least-squares c = (AᵀA)⁻¹ AᵀY,
    // where P = poly_deg+1, columns are powers of τ = t - t[idx].
    // For poly_deg=2 → P=3: [1, τ, τ²].
    const int          P = poly_deg_ + 1;
    Eigen::MatrixXf    A(win_n, P);
    Eigen::MatrixX3f   Y(win_n, 3);
    const TimePoint    t_eval = buf_[idx].t;
    for (int i = 0; i < win_n; ++i)
    {
        const float tau = std::chrono::duration<float>(buf_[lo + i].t - t_eval).count();
        float       pwr = 1.0f;
        for (int k = 0; k < P; ++k)
        {
            A(i, k) = pwr;
            pwr *= tau;
        }
        Y.row(i) = buf_[lo + i].p.transpose();
    }

    // Solve normal equations:  AᵀA · c = AᵀY  (per-axis since Y has 3 columns).
    // ldlt is fine here because AᵀA is symmetric positive-definite when the τ
    // values are not all equal (which they aren't unless dt is zero, already
    // gated by detect_bounce_locked_).
    const Eigen::MatrixXf AtA  = A.transpose() * A;             // (P, P)
    const Eigen::MatrixXf AtY  = A.transpose() * Y;             // (P, 3)
    const Eigen::MatrixXf coef = AtA.ldlt().solve(AtY);         // (P, 3)

    // Coefficients at τ = 0 (the latest sample):
    //   c[0] = smoothed position
    //   c[1] = velocity
    //   c[2] = ½·a  → a = 2·c[2]
    Estimate est;
    est.p = coef.row(0).transpose();
    est.v = (P >= 2) ? Eigen::Vector3f(coef.row(1).transpose()) : Eigen::Vector3f::Zero();
    est.a = (P >= 3) ? Eigen::Vector3f(2.0f * coef.row(2).transpose()) : Eigen::Vector3f::Zero();
    est.t = t_eval;
    est.n_used = win_n;
    est.last_bounce_idx = last_bounce_idx_;
    return est;
}
