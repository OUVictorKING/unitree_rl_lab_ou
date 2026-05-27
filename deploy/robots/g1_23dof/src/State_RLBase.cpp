// #include "FSM/State_RLBase.h"
// #include "unitree_articulation.h"
// #include "isaaclab/envs/mdp/observations/observations.h"
// #include "isaaclab/envs/mdp/actions/joint_actions.h"

// State_RLBase::State_RLBase(int state_mode, std::string state_string)
// : FSMState(state_mode, state_string) 
// {
//     std::cout << "[DEBUG] State_RLBase ctor: " << state_string << std::endl;

//     auto cfg = param::config["FSM"][state_string];
//     auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

//     env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
//         YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
//         std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
//     );
//     env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

//     this->registered_checks.emplace_back(
//         std::make_pair(
//             [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
//             FSMStringMap.right.at("Passive")
//         )
//     );
// }

// void State_RLBase::run()
// {
//     static bool printed = false;

//     auto action = env->action_manager->processed_actions();

//     if (!printed)
//     {
//         for (int i = 0; i < env->robot->data.joint_ids_map.size(); i++)
//         {
//             std::cout << "[joint_ids_map] action[" << i << "] -> sdk["
//                       << env->robot->data.joint_ids_map[i] << "]" << std::endl;
//         }
//         printed = true;
//     }

//     for (int i = 0; i < env->robot->data.joint_ids_map.size(); i++)
//     {
//         lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
//     }
// }


#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace
{
    std::vector<float> yaml_to_float_vector(const YAML::Node &node, const std::string &name)
    {
        if (!node || !node.IsSequence())
        {
            throw std::runtime_error("YAML field '" + name + "' is missing or not a sequence.");
        }

        std::vector<float> out;
        out.reserve(node.size());
        for (size_t i = 0; i < node.size(); ++i)
        {
            out.push_back(node[i].as<float>());
        }
        return out;
    }
    static const char *kActionNames[23] = {
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "waist_yaw_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_elbow_joint",
        "right_elbow_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "left_wrist_roll_joint",
        "right_wrist_roll_joint"};
}

// 一旦切到 Velocity，就会实例化 State_RLBase
State_RLBase::State_RLBase(int state_mode, std::string state_string)
    : FSMState(state_mode, state_string)
{
    std::cerr << "[DEBUG] State_RLBase ctor: " << state_string << std::endl;

    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());
    auto deploy_yaml_path = policy_dir / "params" / "deploy.yaml";

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(deploy_yaml_path),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate));
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    // -----------------------------
    // Read action scale / offset from deploy.yaml
    // -----------------------------
    YAML::Node deploy_cfg = YAML::LoadFile(deploy_yaml_path);

    // 优先使用 actions.JointPositionAction.offset
    // 如果没有，就退回 default_joint_pos
    if (deploy_cfg["actions"] &&
        deploy_cfg["actions"]["JointPositionAction"] &&
        deploy_cfg["actions"]["JointPositionAction"]["scale"])
    {
        action_scale_ = yaml_to_float_vector(
            deploy_cfg["actions"]["JointPositionAction"]["scale"],
            "actions.JointPositionAction.scale");
    }
    else
    {
        throw std::runtime_error("deploy.yaml missing actions.JointPositionAction.scale");
    }

    if (deploy_cfg["actions"] &&
        deploy_cfg["actions"]["JointPositionAction"] &&
        deploy_cfg["actions"]["JointPositionAction"]["offset"])
    {
        action_offset_ = yaml_to_float_vector(
            deploy_cfg["actions"]["JointPositionAction"]["offset"],
            "actions.JointPositionAction.offset");
    }
    else if (deploy_cfg["default_joint_pos"])
    {
        action_offset_ = yaml_to_float_vector(
            deploy_cfg["default_joint_pos"],
            "default_joint_pos");
    }
    else
    {
        throw std::runtime_error(
            "deploy.yaml missing both actions.JointPositionAction.offset and default_joint_pos");
    }

    const int action_dim = static_cast<int>(env->robot->data.joint_ids_map.size());

    if ((int)action_scale_.size() != action_dim)
    {
        std::cerr << "[ERROR] action_scale_.size()=" << action_scale_.size()
                  << " but joint_ids_map.size()=" << action_dim << std::endl;
        throw std::runtime_error("action_scale size mismatch");
    }

    if ((int)action_offset_.size() != action_dim)
    {
        std::cerr << "[ERROR] action_offset_.size()=" << action_offset_.size()
                  << " but joint_ids_map.size()=" << action_dim << std::endl;
        throw std::runtime_error("action_offset size mismatch");
    }

    std::cerr << "[DEBUG] loaded action_scale_/offset_ from deploy.yaml" << std::endl;
    int show_n = std::min<int>(6, action_dim);
    for (int i = 0; i < show_n; ++i)
    {
        std::cerr << "  idx[" << i << "]"
                  << " scale=" << action_scale_[i]
                  << " offset=" << action_offset_[i]
                  << " -> sdk[" << env->robot->data.joint_ids_map[i] << "]"
                  << std::endl;
    }

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]() -> bool
            { return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")));
}


// 这个是能够进行sim2sim的
// void State_RLBase::run()
// {
//     static int run_count = 0;

//     auto action = env->action_manager->processed_actions();

//     static bool printed_pd_once = false;
//     static const char *action_names[23] = {
//         "left_hip_pitch_joint",
//         "right_hip_pitch_joint",
//         "waist_yaw_joint",
//         "left_hip_roll_joint",
//         "right_hip_roll_joint",
//         "left_shoulder_pitch_joint",
//         "right_shoulder_pitch_joint",
//         "left_hip_yaw_joint",
//         "right_hip_yaw_joint",
//         "left_shoulder_roll_joint",
//         "right_shoulder_roll_joint",
//         "left_knee_joint",
//         "right_knee_joint",
//         "left_shoulder_yaw_joint",
//         "right_shoulder_yaw_joint",
//         "left_ankle_pitch_joint",
//         "right_ankle_pitch_joint",
//         "left_elbow_joint",
//         "right_elbow_joint",
//         "left_ankle_roll_joint",
//         "right_ankle_roll_joint",
//         "left_wrist_roll_joint",
//         "right_wrist_roll_joint"};

//     if (!printed_pd_once)
//     {
//         std::cerr << "\n[PD CHECK] actual kp/kd in Velocity state\n";
//         for (int i = 0; i < (int)env->robot->data.joint_ids_map.size(); ++i)
//         {
//             int sdk_id = env->robot->data.joint_ids_map[i];
//             auto &motor = lowcmd->msg_.motor_cmd()[sdk_id];

//             std::cerr << "  idx=" << i
//                       << " joint=" << action_names[i]
//                       << " sdk=" << sdk_id
//                       << " kp=" << motor.kp()
//                       << " kd=" << motor.kd()
//                       << std::endl;
//         }
//         std::cerr << std::endl;

//         printed_pd_once = true;
//     }

//     // 每 500 次打印一次前几个关节的信息
//     // if (run_count % 500 == 0)
//     // {
//     //     std::cerr << "[RUN DEBUG] run=" << run_count << std::endl;
//     //     for (int i = 0; i < std::min<int>(6, action.size()); ++i)
//     //     {
//     //         int sdk_id = env->robot->data.joint_ids_map[i];
//     //         float q_des = action_offset_[i] + action_scale_[i] * action[i];

//     //         std::cerr << "  idx=" << i
//     //                   << " joint=" << action_names[i]
//     //                   << " sdk=" << sdk_id
//     //                   << " action=" << action[i]
//     //                   << " offset=" << action_offset_[i]
//     //                   << " scale=" << action_scale_[i]
//     //                   << " q_des=" << q_des
//     //                   << std::endl;
//     //     }
//     // }

//     for (int i = 0; i < (int)env->robot->data.joint_ids_map.size(); i++)
//     {
//         int sdk_id = env->robot->data.joint_ids_map[i];

//         // 关键修改：恢复训练时的动作定义
//         float q_des = action[i];
//         float q_des_scaled = action_offset_[i] + action_scale_[i] * action[i];

//         lowcmd->msg_.motor_cmd()[sdk_id].q() = q_des_scaled;
//     }

//     run_count++;
// }

void State_RLBase::run()
{
    static int run_count = 0;

    auto action = env->action_manager->processed_actions();

    static bool printed_once = false;
    if (!printed_once)
    {
        std::cerr << "\n[JOINT MAP + PD CHECK]\n";
        for (int i = 0; i < (int)env->robot->data.joint_ids_map.size(); ++i)
        {
            int sdk_id = env->robot->data.joint_ids_map[i];
            auto &motor = lowcmd->msg_.motor_cmd()[sdk_id];

            std::cerr << "  idx=" << i
                      << " joint=" << kActionNames[i]
                      << " sdk=" << sdk_id
                      << " kp=" << motor.kp()
                      << " kd=" << motor.kd()
                      << " offset=" << action_offset_[i]
                      << " scale=" << action_scale_[i]
                      << std::endl;
        }
        std::cerr << std::endl;
        printed_once = true;

        std::cerr << "[RUN DEBUG] run=" << run_count << std::endl;
        for (int i = 0; i < std::min<int>(6, (int)action.size()); ++i)
        {
            int sdk_id = env->robot->data.joint_ids_map[i];
            float q_meas = lowstate->msg_.motor_state()[sdk_id].q();
            float dq_meas = lowstate->msg_.motor_state()[sdk_id].dq();
            float q_des_direct = action[i];
            float q_des_scaled = action_offset_[i] + action_scale_[i] * action[i];

            std::cerr << "  idx=" << i
                      << " joint=" << kActionNames[i]
                      << " sdk=" << sdk_id
                      << " action=" << action[i]
                      << " q_des_direct=" << q_des_direct
                      << " q_des_scaled=" << q_des_scaled
                      << " q=" << q_meas
                      << " dq=" << dq_meas
                      << std::endl;
        }
    }

    for (int i = 0; i < (int)env->robot->data.joint_ids_map.size(); ++i)
    {
        int sdk_id = env->robot->data.joint_ids_map[i];

        // 先保留你当前行为
        float q_des = action[i];

        lowcmd->msg_.motor_cmd()[sdk_id].q() = q_des;
    }

    run_count++;
}

// 测试关节顺序
// void State_RLBase::run()
// {
//     static int run_count = 0;
//     const int action_dim = (int)env->robot->data.joint_ids_map.size();

//     // ===== 只改这里：测试哪个 action =====
//     const int test_idx = 11; // 例如 11=left_knee_joint, 12=right_knee_joint
//     const float amp = 0.50f; // 测试幅度，可改 0.15 / 0.20 / 0.30

//     static const char *action_names[23] = {
//         "left_hip_pitch_joint",
//         "right_hip_pitch_joint",
//         "waist_yaw_joint",
//         "left_hip_roll_joint",
//         "right_hip_roll_joint",
//         "left_shoulder_pitch_joint",
//         "right_shoulder_pitch_joint",
//         "left_hip_yaw_joint",
//         "right_hip_yaw_joint",
//         "left_shoulder_roll_joint",
//         "right_shoulder_roll_joint",
//         "left_knee_joint",
//         "right_knee_joint",
//         "left_shoulder_yaw_joint",
//         "right_shoulder_yaw_joint",
//         "left_ankle_pitch_joint",
//         "right_ankle_pitch_joint",
//         "left_elbow_joint",
//         "right_elbow_joint",
//         "left_ankle_roll_joint",
//         "right_ankle_roll_joint",
//         "left_wrist_roll_joint",
//         "right_wrist_roll_joint"};

//     // 每 3 秒切一次符号
//     // 这里假设 run() 频率大约 1000 Hz
//     // 3000 次约等于 3 秒
//     const int switch_interval = 3000;

//     int phase = (run_count / switch_interval) % 2;
//     float delta = (phase == 0) ? (+amp) : (-amp);

//     for (int i = 0; i < action_dim; i++)
//     {
//         int sdk_id = env->robot->data.joint_ids_map[i];

//         // 所有关节默认回到 deploy 的 offset 姿态
//         float q_des = action_offset_[i];

//         // 只给一个测试关节加偏置
//         if (i == test_idx)
//         {
//             q_des += delta;
//         }

//         lowcmd->msg_.motor_cmd()[sdk_id].q() = q_des;
//     }

//     if (run_count % 500 == 0)
//     {
//         int sdk_id = env->robot->data.joint_ids_map[test_idx];
//         float q_des_test = action_offset_[test_idx] + delta;
//         float q = lowstate->msg_.motor_state()[sdk_id].q();
//         float dq = lowstate->msg_.motor_state()[sdk_id].dq();

//         std::cerr << "[ALT SIGN TEST] run=" << run_count
//                   << " test_idx=" << test_idx
//                   << " joint=" << action_names[test_idx]
//                   << " sdk_id=" << sdk_id
//                   << " delta=" << delta
//                   << " q_des=" << q_des_test
//                   << " q=" << q
//                   << " dq=" << dq
//                   << " err=" << (q_des_test - q)
//                   << std::endl;
//     }

//     run_count++;
// }