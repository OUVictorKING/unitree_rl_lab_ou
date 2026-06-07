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
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>

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
    int planner_max_table_bounces_before_fallback_ = 4;
    std::atomic<int> policy_loop_heartbeat_{0};
    std::atomic<double> last_policy_loop_time_s_{-1.0};
    std::atomic<bool> sim_time_valid_{false};
    std::atomic<double> sim_time_s_{0.0};

    rclcpp::Node::SharedPtr ros2_node_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr ball_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr base_sub_;
    std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> ros2_executor_;
    std::thread ros2_thread_;
    pid_t ros_bag_record_pid_ = -1;
    pid_t ros_bag_replay_pid_ = -1;

    void ball_odom_cb(const nav_msgs::msg::Odometry::SharedPtr msg);
    void base_pose_cb(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
};

REGISTER_FSM(State_Pingpong)
