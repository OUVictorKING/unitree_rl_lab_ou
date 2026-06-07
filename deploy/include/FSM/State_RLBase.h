// // Copyright (c) 2025, Unitree Robotics Co., Ltd.
// // All rights reserved.

// #pragma once

// #include "FSMState.h"
// #include "isaaclab/envs/mdp/actions/joint_actions.h"
// #include "isaaclab/envs/mdp/terminations.h"
// #include <vector>

// class State_RLBase : public FSMState
// {
// public:
//     //储存scale和offset 
//     std::vector<float> action_scale_;
//     std::vector<float> action_offset_;

//     State_RLBase(int state_mode, std::string state_string);
    
//     void enter()
//     {
//         // set gain
//         // for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
//         // {
//         //     lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
//         //     lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
//         //     lowcmd->msg_.motor_cmd()[i].dq() = 0;
//         //     lowcmd->msg_.motor_cmd()[i].tau() = 0;
//         // }

//         for (int i = 0; i < (int)env->robot->data.joint_ids_map.size(); ++i)
//         {
//             int sdk_id = env->robot->data.joint_ids_map[i];

//             lowcmd->msg_.motor_cmd()[sdk_id].kp() = env->robot->data.joint_stiffness[sdk_id];
//             lowcmd->msg_.motor_cmd()[sdk_id].kd() = env->robot->data.joint_damping[sdk_id];
//             lowcmd->msg_.motor_cmd()[sdk_id].dq() = 0.0;
//             lowcmd->msg_.motor_cmd()[sdk_id].tau() = 0.0;
//         }

//         env->robot->update();
//         // Start policy thread
//         policy_thread_running = true;
//         policy_thread = std::thread([this]{
//             using clock = std::chrono::high_resolution_clock;
//             const std::chrono::duration<double> desiredDuration(env->step_dt);
//             const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

//             // Initialize timing
//             auto sleepTill = clock::now() + dt;
//             env->reset();

//             while (policy_thread_running)
//             {
//                 env->step();

//                 // Sleep
//                 std::this_thread::sleep_until(sleepTill);
//                 sleepTill += dt;
//             }
//         });
//     }

//     void run();
    
//     void exit()
//     {
//         policy_thread_running = false;
//         if (policy_thread.joinable()) {
//             policy_thread.join();
//         }
//     }

// private:
//     std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

//     std::thread policy_thread;
//     bool policy_thread_running = false;
// };

// REGISTER_FSM(State_RLBase)

// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"
#include <atomic>
#include <chrono>
#include <vector>

// velocity模式的入口文件

class State_RLBase : public FSMState
{
public:
    std::vector<float> action_scale_;
    std::vector<float> action_offset_;

    State_RLBase(int state_mode, std::string state_string);

    void enter()
    {
        env->robot->update();
        enter_time_ = std::chrono::steady_clock::now();
        entry_hold_q_.assign(
            env->robot->data.joint_pos.data(),
            env->robot->data.joint_pos.data() + env->robot->data.joint_pos.size());
        policy_output_ready_.store(false);

        // 按 action idx -> sdk_id 的映射设置 PD
        for (int i = 0; i < (int)env->robot->data.joint_ids_map.size(); ++i)
        {
            int sdk_id = env->robot->data.joint_ids_map[i];

            // 关键修改：这里用 sdk_id，不要用 i
            lowcmd->msg_.motor_cmd()[sdk_id].kp() = env->robot->data.joint_stiffness[sdk_id];
            lowcmd->msg_.motor_cmd()[sdk_id].kd() = env->robot->data.joint_damping[sdk_id];
            lowcmd->msg_.motor_cmd()[sdk_id].dq() = 0.0f;
            lowcmd->msg_.motor_cmd()[sdk_id].tau() = 0.0f;

            // lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            // lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            // lowcmd->msg_.motor_cmd()[i].dq() = 0.0f;
            // lowcmd->msg_.motor_cmd()[i].tau() = 0.0f;

            // 进入 Velocity 时，先 hold 当前实测关节位姿，等第一帧 policy 输出后再接管。
            lowcmd->msg_.motor_cmd()[sdk_id].q() = entry_hold_q_[i];
            // lowcmd->msg_.motor_cmd()[sdk_id].q() = action_offset_[sdk_id];
        }

        policy_thread_running = true;
        policy_thread = std::thread([this]
                                    {
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            auto sleepTill = clock::now() + dt;
            env->reset();

            while (policy_thread_running)
            {
                env->step();
                policy_output_ready_.store(true);
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            } });
    }

    void run();

    void exit()
    {
        policy_thread_running = false;
        policy_output_ready_.store(false);
        if (policy_thread.joinable())
        {
            policy_thread.join();
        }
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    std::thread policy_thread;
    bool policy_thread_running = false;
    std::atomic<bool> policy_output_ready_{false};
    std::vector<float> entry_hold_q_;
    std::chrono::steady_clock::time_point enter_time_{};
    float bad_orientation_grace_s_ = 0.0f;
};

REGISTER_FSM(State_RLBase)
