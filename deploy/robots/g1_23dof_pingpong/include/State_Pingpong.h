#pragma once

#include "FSM/FSMState.h"
#include "isaaclab/algorithms/algorithms.h"
#include "unitree_articulation.h"

#include <Eigen/Dense>
#include <atomic>
#include <chrono>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <sys/types.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>

#include "BallTrajFilter.h"

class State_Pingpong : public FSMState
{
public:
    State_Pingpong(int state_mode, std::string state_string);
    ~State_Pingpong();

    static void set_use_local_sim_time(bool enabled);

    void enter();
    void run();
    void exit();

private:
    struct ObsTermCfg
    {
        std::string name;
        std::vector<float> scale;
        std::vector<float> clip;
        int history_length = 1;
    };

    struct ExternalState
    {
        Eigen::Vector3f ball_pos = Eigen::Vector3f::Zero();
        Eigen::Vector3f ball_vel = Eigen::Vector3f::Zero();
        Eigen::Vector3f base_pos = Eigen::Vector3f::Zero();
        Eigen::Quaternionf base_quat = Eigen::Quaternionf::Identity();
        Eigen::Vector3f blade_normal_world = Eigen::Vector3f::UnitY();
        bool has_ball = false;
        bool has_base = false;
        std::chrono::steady_clock::time_point ball_time;
        std::chrono::steady_clock::time_point ball_sample_time;
        std::chrono::steady_clock::time_point base_time;
        double ball_stamp_s = 0.0;
        double base_stamp_s = 0.0;
    };

    struct Command
    {
        Eigen::Vector3f p_hit_world = Eigen::Vector3f(0.54f, 0.0f, 1.05f);
        Eigen::Vector3f v_ball_in_world = Eigen::Vector3f(-3.0f, 0.0f, -0.5f);
        Eigen::Vector3f v_ball_out_world = Eigen::Vector3f::Zero();
        Eigen::Vector3f v_racket_hat_world = Eigen::Vector3f(-2.0f, 0.0f, 0.0f);
        Eigen::Vector3f n_target_world = Eigen::Vector3f(-1.0f, 0.0f, 0.0f);
        Eigen::Vector3f target_land_world = Eigen::Vector3f(2.45f, 0.0f, 0.78f);
        Eigen::Vector2f p_base_xy_world = Eigen::Vector2f::Zero();
        float t_to_hit = 0.60f;
        int swing_type = 0; // forehand=0, backhand=1
        bool planner_valid = false;
        bool active = false;
        bool waiting_only = false;
    };

    struct PlannerResult
    {
        Command cmd;
        float raw_t_to_hit = 0.0f;
        int table_bounces = 0;
        bool valid = false;
        bool force_waiting = false;
        std::string reject_reason;
    };

    struct FallbackReference
    {
        Eigen::Vector3f hit_offset_base = Eigen::Vector3f(0.5496f, -0.2879f, 0.3187f);
        Eigen::Vector3f racket_vel_base = Eigen::Vector3f(0.4f, 0.8f, 0.6f);
        Eigen::Vector3f normal_base = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
        int swing_type = 0;
        int frame = -1;
        bool valid = false;
    };

    struct PolicyIo
    {
        std::vector<int> joint_ids_map;
        std::vector<float> default_joint_pos;
        std::vector<float> action_scale;
        std::vector<float> action_offset;
        std::vector<std::vector<float>> action_clip;
        std::vector<float> stiffness;
        std::vector<float> damping;
        std::vector<ObsTermCfg> obs_terms;
        int obs_dim = 0;
        int action_dim = 0;
    };

    void load_config(const YAML::Node &cfg);
    void load_training_geometry_from_npz(const YAML::Node &planner);
    void load_policy(const YAML::Node &cfg);
    void apply_motor_gain_overrides(const YAML::Node &cfg);
    void start_ros_if_enabled(const YAML::Node &cfg);
    void stop_ros();
    void start_ros_bag_tools();
    void stop_ros_bag_tools();
    void start_ros_bag_record();
    void start_ros_bag_replay();
    void stop_ros_bag_process(
        pid_t *pid,
        const char *name,
        const std::string &output_path = std::string(),
        const std::string &log_path = std::string());
    void policy_loop();

    // Pull-side base filter: ROS callback only pushes raw samples into the
    // sliding window (cheap), the control loop calls this at 50 Hz when it
    // actually needs a base pose for the actor obs. Result is the
    // mean-of-window already mapped through input_point_to_training /
    // input_quat_to_training so callers don't need a second transform.
    // Caller MUST hold ext_mtx_; the deques live behind that mutex.
    // Returns reset defaults if the window is empty (defensive — practical
    // path is gated upstream by ext_.has_base).
    std::pair<Eigen::Vector3f, Eigen::Quaternionf>
    compute_base_mean_world_locked() const;

    double controller_time_seconds() const;
    void observe_sim_time_stamp(const builtin_interfaces::msg::Time &stamp);
    bool external_state_fresh(ExternalState *out) const;
    ExternalState latest_external_state_for_policy() const;
    void update_command(double now_s, const ExternalState &state);
    void hold_previous_or_seed_initial_command(double now_s, const ExternalState &state, const char *reason);
    PlannerResult plan_once(const ExternalState &state) const;
    Command make_fallback_command(const ExternalState *state = nullptr) const;
    Command make_waiting_command(const ExternalState &state, float t_to_hit) const;

    std::vector<float> build_obs(const ExternalState &state, const Command &cmd);
    std::vector<float> build_obs_term(const std::string &name, const ExternalState &state, const Command &cmd) const;
    std::vector<float> apply_obs_scale_clip(const ObsTermCfg &term, std::vector<float> value) const;
    std::vector<float> processed_action_from_raw(const std::vector<float> &raw) const;
    void set_safe_targets_locked();
    void apply_policy_gains_to_lowcmd();
    void maybe_log_hit_window(const Command &cmd, const ExternalState &state, float t_to_hit);
    void log_hit_trace_sample(const Command &cmd, const ExternalState &state, float t_to_hit);
    void debug_log_control_state(
        const char *tag,
        double elapsed_s,
        bool external_fresh,
        const std::vector<float> &reference_target,
        int detail_count = 6);

    float joint_pos_by_sdk_id(int sdk_id) const;
    float yaw_from_quat(const Eigen::Quaternionf &q) const;
    Eigen::Vector2f rotate_yaw_2d(const Eigen::Vector2f &v, float yaw) const;
    Eigen::Matrix3f rpy_matrix(float roll, float pitch, float yaw) const;
    Eigen::Matrix3f axis_angle_matrix(const Eigen::Vector3f &axis, float angle) const;
    Eigen::Affine3f joint_transform(const Eigen::Vector3f &xyz, const Eigen::Matrix3f &rpy, const Eigen::Vector3f &axis, float q) const;
    Eigen::Affine3f compute_blade_transform_from_fk(const Eigen::Vector3f &base_pos, const Eigen::Quaternionf &base_quat) const;
    Eigen::Vector3f compute_blade_position_from_fk(const Eigen::Vector3f &base_pos, const Eigen::Quaternionf &base_quat) const;
    Eigen::Vector3f compute_blade_normal_from_fk(const Eigen::Quaternionf &base_quat) const;
    Eigen::Vector3f solve_racket_target(const Eigen::Vector3f &p_hit, const Eigen::Vector3f &v_in, Command *cmd) const;
    Eigen::Vector3f input_point_to_training(const Eigen::Vector3f &p) const;
    Eigen::Vector3f input_vector_to_training(const Eigen::Vector3f &v) const;
    Eigen::Quaternionf input_quat_to_training(const Eigen::Quaternionf &q) const;

    static std::vector<float> yaml_float_vector(const YAML::Node &node, const std::string &name);
    static std::vector<int> yaml_int_vector_from_numeric(const YAML::Node &node, const std::string &name);
    static std::vector<float> remap_full_or_policy(const std::vector<float> &values, const std::vector<int> &ids);
    std::vector<float> make_switch_entry_target(const std::vector<float> &start_q) const;
    static std::vector<float> load_joint_pos_frame_from_npz(const std::string &path, int frame);
    static Eigen::Vector2f load_impact_offset_from_npz(const std::string &path);
    static FallbackReference load_fallback_reference_from_npz(const std::string &path, int frame_override);

    std::shared_ptr<unitree::BaseArticulation<LowState_t::SharedPtr>> robot_;
    std::unique_ptr<isaaclab::OrtRunner> actor_;
    PolicyIo io_;

    mutable std::mutex ext_mtx_;
    ExternalState ext_;

    mutable std::mutex cmd_mtx_;
    Command cmd_;
    std::vector<float> last_raw_action_;
    std::vector<float> current_pd_target_;
    std::vector<float> entry_joint_pos_;
    std::vector<float> switch_entry_joint_pos_;
    std::vector<float> switch_start_q_;
    std::vector<float> actor_blend_start_target_;
    std::string entry_joint_mode_ = "full";
    std::unordered_map<std::string, std::deque<std::vector<float>>> obs_history_;
    bool active_control_ = false;
    bool actor_output_ready_ = false;
    bool actor_blend_active_ = false;
    bool policy_gains_applied_ = false;
    bool keep_current_gains_during_switch_ = true;
    bool support_gain_override_enable_ = false;
    std::vector<int> support_gain_sdk_ids_;
    std::vector<float> support_gain_kp_;
    std::vector<float> support_gain_kd_;

    std::thread policy_thread_;
    std::atomic<bool> policy_thread_running_{false};

    Eigen::Vector3f reset_root_pos_ = Eigen::Vector3f(0.0f, 0.0f, 0.74f);
    Eigen::Quaternionf reset_root_quat_ = Eigen::Quaternionf::Identity();
    Eigen::Vector3f target_land_world_ = Eigen::Vector3f(2.45f, 0.0f, 0.78f);
    Eigen::Vector3f input_origin_in_training_world_ = Eigen::Vector3f::Zero();
    Eigen::Quaternionf input_to_training_quat_ = Eigen::Quaternionf::Identity();
    Eigen::Vector2f forehand_offset_base_ = Eigen::Vector2f(0.5496f, -0.2879f);
    Eigen::Vector2f backhand_offset_base_ = Eigen::Vector2f(0.5250f, 0.0164f);
    Eigen::Vector3f fallback_blade_normal_world_ = Eigen::Vector3f(0.0f, -1.0f, 0.0f);
    FallbackReference fallback_ref_;

    float y_mid_base_ = -0.1358f;
    float swing_y_sign_ = -1.0f;
    float x_hit_world_ = 0.5373f;
    float z_min_world_ = 0.85f;
    float z_max_world_ = 1.25f;
    float table_top_z_ = 0.76f;
    float ball_radius_ = 0.02f;
    float table_center_x_ = 1.77f;
    float table_center_y_ = 0.0f;
    float table_half_x_ = 1.37f;
    float table_half_y_ = 0.7625f;
    float planner_dt_ = 0.01f;
    float planner_max_time_ = 1.50f;
    float planner_drag_k_ = 0.10257265f;
    float planner_bounce_ch_ = 0.72700506f;
    float planner_bounce_cv_ = 0.90183574f;
    float planner_min_t_to_hit_ = 0.05f;
    float planner_max_t_to_hit_ = 1.20f;
    float gait_phase_min_t_hit0_ = 0.20f;
    float gait_phase_max_t_hit0_ = 0.90f;
    float planner_min_incoming_speed_x_ = 0.05f;
    float planner_min_ball_z_world_ = 0.74f;
    float freeze_time_before_hit_ = 0.20f;
    float post_swing_time_ = 0.60f;
    float flight_time_ = 0.45f;
    float paddle_cor_ = 0.85f;
    float topic_timeout_s_ = 0.20f;
    float max_ball_sample_age_s_ = 0.25f;
    float policy_dt_ = 0.02f;
    float switch_blend_s_ = 1.50f;
    float actor_blend_s_ = 0.0f;
    float waiting_initial_t_to_hit_ = -0.02f;
    float planner_hit_log_window_s_ = 0.05f;
    bool post_hit_imitation_ = true;
    bool debug_control_log_enable_ = false;
    bool debug_actor_log_enable_ = false;
    bool planner_hit_log_enable_ = true;
    bool hit_trace_csv_enable_ = false;
    bool use_ros_header_stamp_ = true;
    bool require_base_topic_ = true;
    // Real-robot debugging hatch (yaml: ros.enable_ball_input). When false the
    // ball callback still records CSV + prints 1-Hz info but does NOT push to
    // BallTrajFilter or update ext_, so planner permanently runs the fallback
    // path = first-frame forehand waiting cmd. See ros: section in config.yaml.
    bool ball_input_to_planner_enable_ = true;
    bool local_sim_time_active_ = false;
    bool ros_bag_record_enable_ = false;
    bool ros_bag_replay_enable_ = false;
    bool ros_bag_replay_loop_ = false;
    float ros_bag_replay_rate_ = 1.0f;
    std::string ros_bag_record_path_;
    std::string ros_bag_record_active_output_path_;
    std::string ros_bag_record_active_log_path_;
    std::string ros_bag_replay_path_;
    std::string hit_trace_csv_path_;
    std::string ball_topic_ = "/pingpong/ball_state";
    std::string base_topic_ = "/pingpong/base_pose";

    bool command_active_ = false;
    bool command_frozen_ = false;
    bool has_live_planner_cmd_ = false;
    bool gait_phase_latched_ = false;
    double gait_phase_start_time_s_ = 0.0;
    float gait_phase_t_hit0_ = 0.0f;
    double t_hit_abs_ = 0.0;
    double start_time_s_ = 0.0;
    double actor_blend_start_time_s_ = 0.0;
    double last_command_update_s_ = 0.0;
    int run_debug_counter_ = 0;
    double last_waiting_reason_log_s_ = -10.0;
    std::string last_waiting_reason_;
    bool last_waiting_held_previous_ = false;
    bool hit_window_logged_ = false;
    std::ofstream hit_trace_csv_;
    std::mutex hit_trace_mtx_;
    // Per-message ROS topic trace (one row per ball/base callback). Truncated
    // each time Pingpong is entered so each session is self-contained. Each
    // row is flushed immediately so Ctrl+C does not lose tail rows.
    bool ros_trace_enable_ = true;
    // Defaults are RELATIVE to the deploy-package root (= proj_dir, =
    // deploy/robots/g1_23dof_pingpong). resolve_project_path prepends proj_dir
    // unless the value is already absolute, so don't repeat the package prefix
    // here — config.yaml may override with absolute paths if you want logs
    // outside the package.
    std::string ros_ball_trace_path_ = "logs/ros_ball_trace.csv";
    std::string ros_base_trace_path_ = "logs/ros_base_trace.csv";
    std::ofstream ros_ball_trace_csv_;
    std::ofstream ros_base_trace_csv_;
    std::mutex ros_ball_trace_mtx_;
    std::mutex ros_base_trace_mtx_;
    // Per-policy-tick motor trace: every 50 Hz tick records actor raw output
    // + final motor q command + measured joint state, so leg jitter can be
    // root-caused offline (actor-side instability vs motor-side tracking).
    // yaml: logging.motor_trace_csv.{enable, output}.
    bool motor_trace_enable_ = true;
    std::string motor_trace_path_ = "logs/motor_trace.csv";
    std::ofstream motor_trace_csv_;
    std::mutex motor_trace_mtx_;
    // Per-tick observation trace: every 50 Hz tick records the full actor
    // input vector (post scale / clip / history concatenation). Use
    // for sim-vs-real obs diff to localize OOD dimensions when the actor
    // diverges on hardware. yaml: logging.obs_trace_csv.{enable, output}.
    bool obs_trace_enable_ = true;
    std::string obs_trace_path_ = "logs/obs_trace.csv";
    std::ofstream obs_trace_csv_;
    std::mutex obs_trace_mtx_;
    // Joint velocity obs source. Real-robot motor encoders report dq with
    // huge internal-estimator noise (std 1-10 rad/s when joints are static,
    // vs ~0 in sim mujoco.qvel during training). Two choices:
    //   "motor_dq"    : take robot_->data.joint_vel directly, then run it
    //                   through a sliding-mean filter of size
    //                   joint_vel_filter_window_ before exposing to obs.
    //                   Window=10 → 100ms latency, σ-noise ~ σ_raw / sqrt(10).
    //   "finite_diff" : compute LSQ slope of last (joint_vel_finite_diff_steps_+1)
    //                   q samples — replaces the noisy motor.dq with a
    //                   q-derived velocity, matches mujoco.qvel statistics.
    std::string joint_vel_obs_source_ = "motor_dq";
    // Sliding-mean filter window (in 50Hz samples) applied to the motor-side
    // dq before it goes into the obs vector. window=1 disables filtering.
    int joint_vel_filter_window_ = 10;
    mutable std::deque<std::vector<float>> motor_dq_window_;
    // Number of past q samples to span when finite-differencing (only used
    // when joint_vel_obs_source_ == "finite_diff"). Larger N averages over
    // more samples → noise floor scales like 1/N, but latency grows like
    // N * policy_dt_ / 2.
    int joint_vel_finite_diff_steps_ = 5;
    mutable std::deque<std::vector<float>> joint_pos_history_for_dq_;
    // Per-second ROS message rate counters. Both callbacks run in the same
    // SingleThreadedExecutor, so plain int + steady_clock time_point is
    // race-free without atomics.
    int ball_msg_count_window_ = 0;
    int base_msg_count_window_ = 0;
    std::chrono::steady_clock::time_point ball_rate_window_start_{};
    std::chrono::steady_clock::time_point base_rate_window_start_{};
    // Sliding-mean filter for the base PoseStamped stream. Mocap publishes
    // ~110Hz on a noisy rigid body; control runs at 50Hz. Averaging the last
    // N samples cuts pose jitter without adding meaningful latency. Position
    // uses arithmetic mean; quaternion uses hemisphere-aligned arithmetic mean
    // (Markley 2007 simplified — exact when intra-window angular spread is
    // small, which it always is at 110Hz × 5 ≈ 45ms). yaml override:
    // ros.base_filter_window. window=1 → deque holds 1 sample → mean = sample
    // = identical to the unfiltered code path.
    int base_filter_window_ = 5;
    std::deque<Eigen::Vector3f> base_pos_window_;
    std::deque<Eigen::Quaternionf> base_quat_window_;
    int planner_max_table_bounces_before_fallback_ = 4;
    std::atomic<int> policy_loop_heartbeat_{0};
    std::atomic<double> last_policy_loop_time_s_{-1.0};
    std::atomic<bool> sim_time_valid_{false};
    std::atomic<double> sim_time_s_{0.0};

    rclcpp::Node::SharedPtr ros2_node_;
    // VRPN-mocap publishes PoseStamped for every tracker (no twist), so the
    // ball subscription is PoseStamped too. Velocity / acceleration are
    // estimated by `ball_filter_` (31-frame 2nd-order polyfit, paper §IV-A).
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr ball_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr base_sub_;
    BallTrajFilter ball_filter_{31, 2, 0.4f, 0.76f, 0.02f, 0.05f, 5};
    std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> ros2_executor_;
    std::thread ros2_thread_;
    pid_t ros_bag_record_pid_ = -1;
    pid_t ros_bag_replay_pid_ = -1;

    void ball_pose_cb(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void base_pose_cb(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
};

REGISTER_FSM(State_Pingpong)
