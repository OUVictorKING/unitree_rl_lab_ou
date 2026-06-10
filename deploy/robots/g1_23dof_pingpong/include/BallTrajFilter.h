// Copyright (c) 2026 — drop-in ball-state estimator for the C++ pingpong planner.
//
// Mirrors the runtime estimator in
//   source/.../tasks/pingpong/mdp/planner.py::estimate_state + _smooth_va,
// which is the paper (HITTER §IV-A) recipe:
//   - keep the LATEST 31 ball-position samples (one per ROS message)
//   - per-axis 2nd-order LSQ polynomial fit on those samples
//   - evaluate the polynomial at the most recent sample to get smoothed
//     (p, v, a) — paper says "buffer cleared at bounce", so when a vz-reversal
//     is detected we restrict the fit window to post-bounce samples only.
//
// The class is thread-safe (push/estimate may be called from different
// threads — typically push from the ROS callback, estimate from the 50 Hz
// policy loop). It owns its own mutex; the caller does NOT need ext_mtx_.
//
// Integration sketch (see State_Pingpong patch hints in the chat):
//   ball_odom_cb  →  filter.push_sample(now, p_world);
//   plan_once     →  auto est = filter.estimate();  if (!est) return waiting;
//   enter()       →  filter.reset();

#pragma once

#include <Eigen/Dense>

#include <chrono>
#include <deque>
#include <mutex>
#include <optional>

class BallTrajFilter
{
public:
    using TimePoint = std::chrono::steady_clock::time_point;

    struct Estimate
    {
        Eigen::Vector3f p;          // smoothed position at the latest sample
        Eigen::Vector3f v;          // smoothed velocity at the latest sample
        Eigen::Vector3f a;          // smoothed acceleration at the latest sample
        TimePoint       t;          // latest sample timestamp (for staleness checks)
        int             n_used;     // samples actually used in the fit
        int             last_bounce_idx;  // -1 if no bounce in the current window
    };

    // max_samples       paper Sec.IV-A: 31 frames
    // poly_deg          2 (gravity is constant, so quadratic in t is enough)
    // bounce_vz_thresh  m/s — vz reversal magnitude treated as a real bounce
    //                   (paper VZ_THRESH = 0.4 m/s, planner.py:37)
    // table_top_z_world world z of the table top surface (default 0.76 m)
    // ball_radius       used to gate bounces (only count bounces close to table)
    // gap_reset_s       if a new sample arrives more than this many seconds
    //                   after the previous one, the buffer is cleared. This
    //                   guards the polyfit against mixing 'old' and 'new'
    //                   tracking phases when the ball stream drops out for
    //                   an extended period (e.g. ball off-camera for 200 ms,
    //                   sensor freeze, dropped network burst). Default 0.05 s.
    // bounce_lookback   how many of the most recent samples to scan when
    //                   testing for a bounce (default 5). Larger values
    //                   tolerate single-packet drops in the bounce frame.
    explicit BallTrajFilter(int   max_samples       = 31,
                            int   poly_deg          = 2,
                            float bounce_vz_thresh  = 0.4f,
                            float table_top_z_world = 0.76f,
                            float ball_radius       = 0.02f,
                            float gap_reset_s       = 0.05f,
                            int   bounce_lookback   = 5);

    // Append the newest world-frame ball position (training-frame xyz, the
    // same coords plan_once consumes). Cheap; runs O(1) + bounce detection
    // on the last 3 samples.
    void push_sample(TimePoint t, const Eigen::Vector3f &p_world);

    // Drop everything (call from FSM enter()).
    void reset();

    // nullopt if fewer than (poly_deg + 1) post-bounce samples are
    // available — caller should treat this as "filter warming up" and
    // route to the waiting branch of update_command, which mirrors the
    // existing missing_ball_state path.
    std::optional<Estimate> estimate() const;

    int  sample_count() const;
    int  last_bounce_index() const;
    bool has_recent_bounce() const;  // true if last_bounce_idx_ >= 0

    // Diagnostics: counters incremented by push_sample.
    int  num_gap_resets() const;     // buffer cleared because of large gap
    int  num_out_of_order() const;   // sample with t < previous t — dropped
    int  num_dropped_total() const;  // num_gap_resets + num_out_of_order

private:
    int   max_samples_;
    int   poly_deg_;
    float bounce_vz_thresh_;
    float table_top_z_;
    float ball_radius_;
    float gap_reset_s_;
    int   bounce_lookback_;

    struct Sample { TimePoint t; Eigen::Vector3f p; };
    mutable std::mutex      mtx_;
    std::deque<Sample>      buf_;
    int                     last_bounce_idx_ = -1;
    int                     gap_reset_count_ = 0;
    int                     out_of_order_count_ = 0;

    // Caller must hold mtx_.
    void detect_bounce_locked_();
};
