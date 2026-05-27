#pragma once

#include "FSM/State_RLBase.h"
#include <Eigen/Dense>
#include <array>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

class State_Mimic : public FSMState
{
public:
    State_Mimic(int state_mode, std::string state_string);

    void enter();
    void run();

    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable())
        {
            policy_thread.join();
        }
    }

    class MotionLoader_;

    // 供 REGISTER_OBSERVATION(...) 使用
    static std::shared_ptr<MotionLoader_> motion;

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    std::shared_ptr<MotionLoader_> motion_;

    std::thread policy_thread;
    bool policy_thread_running = false;
    std::array<float, 2> time_range_{0.0f, 0.0f};
};

class State_Mimic::MotionLoader_
{
public:
    MotionLoader_(const std::string &motion_file, float fps);

    void update(float time_s);
    void reset(const isaaclab::ArticulationData &data, float t = 0.0f);

    Eigen::VectorXf joint_pos() const;
    Eigen::VectorXf joint_vel() const;

    Eigen::Vector3f root_position() const;
    Eigen::Quaternionf root_quaternion() const;

    // 23DOF 下的 torso/anchor 姿态：
    // root_quat * yaw(waist_yaw)
    Eigen::Quaternionf anchor_quaternion() const;

    float dt = 0.02f;
    int num_frames = 0;
    float duration = 0.0f;

    std::vector<Eigen::Vector3f> root_positions;
    std::vector<Eigen::Quaternionf> root_quaternions;
    std::vector<Eigen::VectorXf> dof_positions;
    std::vector<Eigen::VectorXf> dof_velocities;

    // 初始 yaw 对齐用
    Eigen::Matrix3f world_to_init_ = Eigen::Matrix3f::Identity();

private:
    int index_0_ = 0;
    int index_1_ = 0;
    float blend_ = 0.0f;

    static std::vector<Eigen::VectorXf> compute_raw_derivative(
        const std::vector<Eigen::VectorXf> &data,
        float dt);
};

REGISTER_FSM(State_Mimic)

