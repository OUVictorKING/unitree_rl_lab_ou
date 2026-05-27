
// #include "State_Mimic.h"
// #include "unitree_articulation.h"
// #include "isaaclab/envs/mdp/observations/observations.h"
// #include "isaaclab/envs/mdp/actions/joint_actions.h"

// #include <algorithm>
// #include <chrono>
// #include <cmath>
// #include <iostream>
// #include <stdexcept>
// #include <vector>

// #include "cnpy.h"

// std::shared_ptr<State_Mimic::MotionLoader_> State_Mimic::motion = nullptr;

// namespace
// {
//     template <typename T>
//     std::vector<float> copy_to_float_vec(const T *src, std::size_t n)
//     {
//         std::vector<float> out(n);
//         for (std::size_t i = 0; i < n; ++i)
//         {
//             out[i] = static_cast<float>(src[i]);
//         }
//         return out;
//     }

//     // 23dof deploy 顺序下，waist_yaw_joint 是第 3 个动作维度
//     constexpr int kWaistYawIndex = 2;

//     // SDK 槽位里 waist_yaw_joint 对应 motor[12]
//     constexpr int kSdkWaistYawIndex = 12;

//     Eigen::Quaternionf yaw_quat(float yaw)
//     {
//         return Eigen::Quaternionf(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()));
//     }

//     Eigen::Matrix<float, 6, 1> quat_to_6d(const Eigen::Quaternionf &q)
//     {
//         Eigen::Matrix3f mat = q.normalized().toRotationMatrix();
//         Eigen::Matrix<float, 6, 1> out;
//         out << mat(0, 0), mat(0, 1),
//             mat(1, 0), mat(1, 1),
//             mat(2, 0), mat(2, 1);
//         return out;
//     }

//     // 当前机器人 anchor 姿态：root_quat * yaw(waist_yaw)
//     // 这是 23dof 真机可得的近似 anchor，参考 29dof 部署版思路退化而来。:contentReference[oaicite:1]{index=1}
//     Eigen::Quaternionf robot_anchor_quat_w(isaaclab::ManagerBasedRLEnv *env)
//     {
//         using G1Type = unitree::BaseArticulation<LowState_t::SharedPtr>;
//         G1Type *robot = dynamic_cast<G1Type *>(env->robot.get());
//         if (!robot)
//         {
//             throw std::runtime_error("robot_anchor_quat_w: robot dynamic_cast failed");
//         }

//         const auto &q_root = env->robot->data.root_quat_w;
//         Eigen::Quaternionf root_q(q_root.w(), q_root.x(), q_root.y(), q_root.z());

//         const auto &motors = robot->lowstate->msg_.motor_state();
//         const float waist_yaw = motors[kSdkWaistYawIndex].q();

//         return (root_q * yaw_quat(waist_yaw)).normalized();
//     }

//     // 目标 motion anchor 姿态：motion_root_quat * yaw(motion_waist_yaw)
//     Eigen::Quaternionf motion_anchor_quat_w(std::shared_ptr<State_Mimic::MotionLoader_> loader)
//     {
//         Eigen::Quaternionf root_q = loader->root_quaternion();
//         Eigen::VectorXf q = loader->joint_pos();
//         const float waist_yaw = q(kWaistYawIndex);
//         return (root_q * yaw_quat(waist_yaw)).normalized();
//     }

// } // namespace

// namespace isaaclab
// {
//     namespace mdp
//     {

//         REGISTER_OBSERVATION(motion_joint_pos)
//         {
//             auto data = State_Mimic::motion->joint_pos();
//             return std::vector<float>(data.data(), data.data() + data.size());
//         }

//         REGISTER_OBSERVATION(motion_joint_vel)
//         {
//             auto data = State_Mimic::motion->joint_vel();
//             return std::vector<float>(data.data(), data.data() + data.size());
//         }

//         REGISTER_OBSERVATION(motion_command)
//         {
//             auto pos = State_Mimic::motion->joint_pos();
//             auto vel = State_Mimic::motion->joint_vel();

//             std::vector<float> data;
//             data.reserve(pos.size() + vel.size());
//             data.insert(data.end(), pos.data(), pos.data() + pos.size());
//             data.insert(data.end(), vel.data(), vel.data() + vel.size());
//             return data;
//         }

//         REGISTER_OBSERVATION(motion_anchor_ori_b)
//         {
//             Eigen::Quaternionf q_robot = robot_anchor_quat_w(env);
//             Eigen::Quaternionf q_motion = motion_anchor_quat_w(State_Mimic::motion);

//             // 训练里本质是“robot anchor frame 下 target anchor 的相对朝向”
//             // 这里用真机可得 anchor 近似复刻。
//             Eigen::Quaternionf rel_q = q_robot.conjugate() * q_motion;
//             auto data = quat_to_6d(rel_q);

//             return std::vector<float>(data.data(), data.data() + data.size());
//         }

//     } // namespace mdp
// } // namespace isaaclab

// State_Mimic::MotionLoader_::MotionLoader_(const std::string &motion_file, float fps)
// {
//     if (fps <= 0.0f)
//     {
//         throw std::runtime_error("MotionLoader_: fps must be > 0");
//     }

//     dt = 1.0f / fps;

//     cnpy::npz_t npz = cnpy::npz_load(motion_file);

//     if (!npz.count("joint_pos") ||
//         !npz.count("joint_vel") ||
//         !npz.count("body_pos_w") ||
//         !npz.count("body_quat_w"))
//     {
//         throw std::runtime_error("MotionLoader_: npz missing required keys");
//     }

//     std::vector<float> joint_pos_all;
//     std::vector<std::size_t> joint_pos_shape;
//     {
//         const auto &arr = npz["joint_pos"];
//         joint_pos_shape = arr.shape;
//         if (joint_pos_shape.size() != 2)
//         {
//             throw std::runtime_error("joint_pos must be [T, J]");
//         }

//         if (arr.word_size == sizeof(float))
//             joint_pos_all = copy_to_float_vec(arr.data<float>(), arr.num_vals);
//         else if (arr.word_size == sizeof(double))
//             joint_pos_all = copy_to_float_vec(arr.data<double>(), arr.num_vals);
//         else
//             throw std::runtime_error("joint_pos dtype unsupported");
//     }

//     std::vector<float> joint_vel_all;
//     std::vector<std::size_t> joint_vel_shape;
//     {
//         const auto &arr = npz["joint_vel"];
//         joint_vel_shape = arr.shape;
//         if (joint_vel_shape.size() != 2)
//         {
//             throw std::runtime_error("joint_vel must be [T, J]");
//         }

//         if (arr.word_size == sizeof(float))
//             joint_vel_all = copy_to_float_vec(arr.data<float>(), arr.num_vals);
//         else if (arr.word_size == sizeof(double))
//             joint_vel_all = copy_to_float_vec(arr.data<double>(), arr.num_vals);
//         else
//             throw std::runtime_error("joint_vel dtype unsupported");
//     }

//     std::vector<float> body_pos_all;
//     std::vector<std::size_t> body_pos_shape;
//     {
//         const auto &arr = npz["body_pos_w"];
//         body_pos_shape = arr.shape;
//         if (body_pos_shape.size() != 3 || body_pos_shape[2] != 3)
//         {
//             throw std::runtime_error("body_pos_w must be [T, B, 3]");
//         }

//         if (arr.word_size == sizeof(float))
//             body_pos_all = copy_to_float_vec(arr.data<float>(), arr.num_vals);
//         else if (arr.word_size == sizeof(double))
//             body_pos_all = copy_to_float_vec(arr.data<double>(), arr.num_vals);
//         else
//             throw std::runtime_error("body_pos_w dtype unsupported");
//     }

//     std::vector<float> body_quat_all;
//     std::vector<std::size_t> body_quat_shape;
//     {
//         const auto &arr = npz["body_quat_w"];
//         body_quat_shape = arr.shape;
//         if (body_quat_shape.size() != 3 || body_quat_shape[2] != 4)
//         {
//             throw std::runtime_error("body_quat_w must be [T, B, 4]");
//         }

//         if (arr.word_size == sizeof(float))
//             body_quat_all = copy_to_float_vec(arr.data<float>(), arr.num_vals);
//         else if (arr.word_size == sizeof(double))
//             body_quat_all = copy_to_float_vec(arr.data<double>(), arr.num_vals);
//         else
//             throw std::runtime_error("body_quat_w dtype unsupported");
//     }

//     if (joint_pos_shape[0] != joint_vel_shape[0] ||
//         joint_pos_shape[1] != joint_vel_shape[1])
//     {
//         throw std::runtime_error("joint_pos/joint_vel shape mismatch");
//     }

//     if (joint_pos_shape[0] != body_pos_shape[0] ||
//         joint_pos_shape[0] != body_quat_shape[0] ||
//         body_pos_shape[1] != body_quat_shape[1])
//     {
//         throw std::runtime_error("time/body shape mismatch");
//     }

//     num_frames = static_cast<int>(joint_pos_shape[0]);
//     const int dof_dim = static_cast<int>(joint_pos_shape[1]);
//     duration = (num_frames - 1) * dt;

//     std::cout << "[MIMIC] joint_pos shape = [" << joint_pos_shape[0] << ", " << joint_pos_shape[1] << "]" << std::endl;
//     std::cout << "[MIMIC] joint_vel shape = [" << joint_vel_shape[0] << ", " << joint_vel_shape[1] << "]" << std::endl;
//     std::cout << "[MIMIC] body_pos_w shape = [" << body_pos_shape[0] << ", " << body_pos_shape[1] << ", " << body_pos_shape[2] << "]" << std::endl;
//     std::cout << "[MIMIC] body_quat_w shape = [" << body_quat_shape[0] << ", " << body_quat_shape[1] << ", " << body_quat_shape[2] << "]" << std::endl;

//     // root 默认取第 0 个 body（通常是 pelvis/root）
//     root_positions.reserve(num_frames);
//     root_quaternions.reserve(num_frames);
//     dof_positions.reserve(num_frames);
//     dof_velocities.reserve(num_frames);

//     for (int t = 0; t < num_frames; ++t)
//     {
//         Eigen::Vector3f root_pos(
//             body_pos_all[(t * body_pos_shape[1] + 0) * 3 + 0],
//             body_pos_all[(t * body_pos_shape[1] + 0) * 3 + 1],
//             body_pos_all[(t * body_pos_shape[1] + 0) * 3 + 2]);
//         root_positions.push_back(root_pos);

//         Eigen::Quaternionf root_q(
//             body_quat_all[(t * body_quat_shape[1] + 0) * 4 + 0],
//             body_quat_all[(t * body_quat_shape[1] + 0) * 4 + 1],
//             body_quat_all[(t * body_quat_shape[1] + 0) * 4 + 2],
//             body_quat_all[(t * body_quat_shape[1] + 0) * 4 + 3]);
//         root_quaternions.push_back(root_q.normalized());

//         Eigen::VectorXf q = Eigen::VectorXf::Zero(dof_dim);
//         Eigen::VectorXf dq = Eigen::VectorXf::Zero(dof_dim);
//         for (int j = 0; j < dof_dim; ++j)
//         {
//             q(j) = joint_pos_all[t * dof_dim + j];
//             dq(j) = joint_vel_all[t * dof_dim + j];
//         }
//         dof_positions.push_back(q);
//         dof_velocities.push_back(dq);
//     }

//     update(0.0f);
// }

// void State_Mimic::MotionLoader_::update(float time_s)
// {
//     if (duration <= 0.0f)
//     {
//         index_0_ = 0;
//         index_1_ = 0;
//         blend_ = 0.0f;
//         return;
//     }

//     const float phase = std::clamp(time_s / duration, 0.0f, 1.0f);
//     const float frame_f = phase * (num_frames - 1);

//     index_0_ = static_cast<int>(std::floor(frame_f));
//     index_1_ = std::min(index_0_ + 1, num_frames - 1);
//     blend_ = frame_f - index_0_;
// }

// void State_Mimic::MotionLoader_::reset(const isaaclab::ArticulationData &data, float t)
// {
//     (void)data;
//     update(t);
// }

// Eigen::VectorXf State_Mimic::MotionLoader_::joint_pos() const
// {
//     return dof_positions[index_0_] * (1.0f - blend_) + dof_positions[index_1_] * blend_;
// }

// Eigen::VectorXf State_Mimic::MotionLoader_::joint_vel() const
// {
//     return dof_velocities[index_0_] * (1.0f - blend_) + dof_velocities[index_1_] * blend_;
// }

// Eigen::Vector3f State_Mimic::MotionLoader_::root_position() const
// {
//     return root_positions[index_0_] * (1.0f - blend_) + root_positions[index_1_] * blend_;
// }

// Eigen::Quaternionf State_Mimic::MotionLoader_::root_quaternion() const
// {
//     return root_quaternions[index_0_].slerp(blend_, root_quaternions[index_1_]).normalized();
// }

// State_Mimic::State_Mimic(int state_mode, std::string state_string)
//     : FSMState(state_mode, state_string)
// {
//     auto cfg = param::config["FSM"][state_string];
//     auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

//     auto articulation =
//         std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate);

//     std::filesystem::path motion_file = cfg["motion_file"].as<std::string>();
//     if (!motion_file.is_absolute())
//     {
//         motion_file = param::proj_dir / motion_file;
//     }

//     motion_ = std::make_shared<MotionLoader_>(motion_file.string(), cfg["fps"].as<float>());
//     motion = motion_;
//     spdlog::info("Loaded motion file '{}' with duration {:.2f}s",
//                  motion_file.stem().string(), motion_->duration);

//     if (cfg["time_start"])
//         time_range_[0] = std::clamp(cfg["time_start"].as<float>(), 0.0f, motion_->duration);
//     else
//         time_range_[0] = 0.0f;

//     if (cfg["time_end"])
//         time_range_[1] = std::clamp(cfg["time_end"].as<float>(), 0.0f, motion_->duration);
//     else
//         time_range_[1] = motion_->duration;

//     env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
//         YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
//         articulation);
//     env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

//     std::cout << "[MIMIC] deploy path: " << (policy_dir / "params" / "deploy.yaml") << std::endl;
//     std::cout << "[MIMIC] onnx path: " << (policy_dir / "exported" / "policy.onnx") << std::endl;
//     std::cout << "[MIMIC] action dim = " << env->action_manager->action().size() << std::endl;

//     this->registered_checks.emplace_back(
//         std::make_pair(
//             [&]() -> bool
//             { return (env->episode_length * env->step_dt) > time_range_[1]; },
//             FSMStringMap.right.at("Velocity")));

//     this->registered_checks.emplace_back(
//         std::make_pair(
//             [&]() -> bool
//             { return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
//             FSMStringMap.right.at("Passive")));
// }

// void State_Mimic::enter()
// {
//     for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
//     {
//         lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
//         lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
//         lowcmd->msg_.motor_cmd()[i].dq() = 0.0f;
//         lowcmd->msg_.motor_cmd()[i].tau() = 0.0f;
//     }

//     motion = motion_;
//     motion->reset(env->robot->data, time_range_[0]);
//     env->reset();

//     policy_thread_running = true;
//     policy_thread = std::thread([this]
//                                 {
//         using clock = std::chrono::high_resolution_clock;
//         const std::chrono::duration<double> desiredDuration(env->step_dt);
//         const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

//         auto sleepTill = clock::now() + dt;

//         while (policy_thread_running)
//         {
//             env->robot->update();
//             motion->update(env->episode_length * env->step_dt + time_range_[0]);
//             env->step();

//             std::this_thread::sleep_until(sleepTill);
//             sleepTill += dt;
//         } });
// }

// void State_Mimic::run()
// {
//     auto action = env->action_manager->processed_actions();
//     for (int i = 0; i < env->robot->data.joint_ids_map.size(); ++i)
//     {
//         lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
//     }
// }

#include "State_Mimic.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "cnpy.h"

// 与 29dof 部署版一致：进入 mimic 时记录一个 yaw 对齐四元数
static Eigen::Quaternionf init_quat = Eigen::Quaternionf::Identity();

std::shared_ptr<State_Mimic::MotionLoader_> State_Mimic::motion = nullptr;

namespace
{
    template <typename T>
    std::vector<float> copy_to_float_vec(const T *src, std::size_t n)
    {
        std::vector<float> out(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            out[i] = static_cast<float>(src[i]);
        }
        return out;
    }

    // 23dof mimic policy 顺序里，waist_yaw_joint 的索引
    constexpr int kWaistYawIndex = 2;

    // SDK motor_state 里 waist_yaw_joint 的槽位
    constexpr int kSdkWaistYawIndex = 12;

    Eigen::Quaternionf yaw_quat(float yaw)
    {
        return Eigen::Quaternionf(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()));
    }

    Eigen::Matrix<float, 6, 1> quat_to_6d_from_transposed_rotmat(const Eigen::Quaternionf &q)
    {
        // 与 29dof 部署版一致：toRotationMatrix().transpose() 后取前两列展开
        Eigen::Matrix3f rot = q.normalized().toRotationMatrix().transpose();

        Eigen::Matrix<float, 6, 1> data;
        data << rot(0, 0), rot(0, 1),
            rot(1, 0), rot(1, 1),
            rot(2, 0), rot(2, 1);
        return data;
    }

    // 当前机器人 anchor 姿态（23dof 真机可得近似）:
    // root_quat_w * yaw(waist_yaw)
    Eigen::Quaternionf robot_anchor_quat_w(isaaclab::ManagerBasedRLEnv *env)
    {
        using G1Type = unitree::BaseArticulation<LowState_t::SharedPtr>;
        G1Type *robot = dynamic_cast<G1Type *>(env->robot.get());
        if (!robot)
        {
            throw std::runtime_error("robot_anchor_quat_w: robot dynamic_cast failed");
        }

        const auto &q_root = env->robot->data.root_quat_w;
        Eigen::Quaternionf root_q(q_root.w(), q_root.x(), q_root.y(), q_root.z());

        const auto &motors = robot->lowstate->msg_.motor_state();
        const float waist_yaw = motors[kSdkWaistYawIndex].q();

        return (root_q * yaw_quat(waist_yaw)).normalized();
    }

    // 目标 motion anchor 姿态（23dof 参考近似）:
    // motion_root_quat * yaw(motion_waist_yaw)
    Eigen::Quaternionf motion_anchor_quat_w(std::shared_ptr<State_Mimic::MotionLoader_> loader)
    {
        Eigen::Quaternionf root_q = loader->root_quaternion();
        Eigen::VectorXf q = loader->joint_pos();

        const float waist_yaw = q(kWaistYawIndex);
        return (root_q * yaw_quat(waist_yaw)).normalized();
    }

} // namespace

namespace isaaclab
{
    namespace mdp
    {

        REGISTER_OBSERVATION(motion_joint_pos)
        {
            auto data = State_Mimic::motion->joint_pos();
            return std::vector<float>(data.data(), data.data() + data.size());
        }

        REGISTER_OBSERVATION(motion_joint_vel)
        {
            auto data = State_Mimic::motion->joint_vel();
            return std::vector<float>(data.data(), data.data() + data.size());
        }

        REGISTER_OBSERVATION(motion_command)
        {
            // 与训练定义一致：joint_pos + joint_vel
            auto pos = State_Mimic::motion->joint_pos();
            auto vel = State_Mimic::motion->joint_vel();

            std::vector<float> data;
            data.reserve(pos.size() + vel.size());
            data.insert(data.end(), pos.data(), pos.data() + pos.size());
            data.insert(data.end(), vel.data(), vel.data() + vel.size());
            return data;
        }

        REGISTER_OBSERVATION(motion_anchor_ori_b)
        {
            // 与 29dof 部署版一致：
            // 1) 取当前 anchor 姿态
            // 2) 取参考 anchor 姿态
            // 3) 用 enter-time 的 init_quat 把参考整体旋到当前机器人朝向系
            // 4) 再求相对旋转
            Eigen::Quaternionf real_quat_w = robot_anchor_quat_w(env);
            Eigen::Quaternionf ref_quat_w = motion_anchor_quat_w(State_Mimic::motion);

            Eigen::Quaternionf rot_ = (init_quat * ref_quat_w).conjugate() * real_quat_w;
            auto data = quat_to_6d_from_transposed_rotmat(rot_);

            return std::vector<float>(data.data(), data.data() + data.size());
        }

    } // namespace mdp
} // namespace isaaclab

State_Mimic::MotionLoader_::MotionLoader_(const std::string &motion_file, float fps)
{
    if (fps <= 0.0f)
    {
        throw std::runtime_error("MotionLoader_: fps must be > 0");
    }

    dt = 1.0f / fps;

    cnpy::npz_t npz = cnpy::npz_load(motion_file);

    if (!npz.count("joint_pos") ||
        !npz.count("joint_vel") ||
        !npz.count("body_pos_w") ||
        !npz.count("body_quat_w"))
    {
        throw std::runtime_error("MotionLoader_: npz missing required keys");
    }

    std::vector<float> joint_pos_all;
    std::vector<std::size_t> joint_pos_shape;
    {
        const auto &arr = npz["joint_pos"];
        joint_pos_shape = arr.shape;
        if (joint_pos_shape.size() != 2)
        {
            throw std::runtime_error("joint_pos must be [T, J]");
        }

        if (arr.word_size == sizeof(float))
            joint_pos_all = copy_to_float_vec(arr.data<float>(), arr.num_vals);
        else if (arr.word_size == sizeof(double))
            joint_pos_all = copy_to_float_vec(arr.data<double>(), arr.num_vals);
        else
            throw std::runtime_error("joint_pos dtype unsupported");
    }

    std::vector<float> joint_vel_all;
    std::vector<std::size_t> joint_vel_shape;
    {
        const auto &arr = npz["joint_vel"];
        joint_vel_shape = arr.shape;
        if (joint_vel_shape.size() != 2)
        {
            throw std::runtime_error("joint_vel must be [T, J]");
        }

        if (arr.word_size == sizeof(float))
            joint_vel_all = copy_to_float_vec(arr.data<float>(), arr.num_vals);
        else if (arr.word_size == sizeof(double))
            joint_vel_all = copy_to_float_vec(arr.data<double>(), arr.num_vals);
        else
            throw std::runtime_error("joint_vel dtype unsupported");
    }

    std::vector<float> body_pos_all;
    std::vector<std::size_t> body_pos_shape;
    {
        const auto &arr = npz["body_pos_w"];
        body_pos_shape = arr.shape;
        if (body_pos_shape.size() != 3 || body_pos_shape[2] != 3)
        {
            throw std::runtime_error("body_pos_w must be [T, B, 3]");
        }

        if (arr.word_size == sizeof(float))
            body_pos_all = copy_to_float_vec(arr.data<float>(), arr.num_vals);
        else if (arr.word_size == sizeof(double))
            body_pos_all = copy_to_float_vec(arr.data<double>(), arr.num_vals);
        else
            throw std::runtime_error("body_pos_w dtype unsupported");
    }

    std::vector<float> body_quat_all;
    std::vector<std::size_t> body_quat_shape;
    {
        const auto &arr = npz["body_quat_w"];
        body_quat_shape = arr.shape;
        if (body_quat_shape.size() != 3 || body_quat_shape[2] != 4)
        {
            throw std::runtime_error("body_quat_w must be [T, B, 4]");
        }

        if (arr.word_size == sizeof(float))
            body_quat_all = copy_to_float_vec(arr.data<float>(), arr.num_vals);
        else if (arr.word_size == sizeof(double))
            body_quat_all = copy_to_float_vec(arr.data<double>(), arr.num_vals);
        else
            throw std::runtime_error("body_quat_w dtype unsupported");
    }

    if (joint_pos_shape[0] != joint_vel_shape[0] ||
        joint_pos_shape[1] != joint_vel_shape[1])
    {
        throw std::runtime_error("joint_pos/joint_vel shape mismatch");
    }

    if (joint_pos_shape[0] != body_pos_shape[0] ||
        joint_pos_shape[0] != body_quat_shape[0] ||
        body_pos_shape[1] != body_quat_shape[1])
    {
        throw std::runtime_error("time/body shape mismatch");
    }

    num_frames = static_cast<int>(joint_pos_shape[0]);
    const int dof_dim = static_cast<int>(joint_pos_shape[1]);
    duration = (num_frames - 1) * dt;

    std::cout << "[MIMIC] joint_pos shape = [" << joint_pos_shape[0] << ", " << joint_pos_shape[1] << "]" << std::endl;
    std::cout << "[MIMIC] joint_vel shape = [" << joint_vel_shape[0] << ", " << joint_vel_shape[1] << "]" << std::endl;
    std::cout << "[MIMIC] body_pos_w shape = [" << body_pos_shape[0] << ", " << body_pos_shape[1] << ", " << body_pos_shape[2] << "]" << std::endl;
    std::cout << "[MIMIC] body_quat_w shape = [" << body_quat_shape[0] << ", " << body_quat_shape[1] << ", " << body_quat_shape[2] << "]" << std::endl;

    root_positions.reserve(num_frames);
    root_quaternions.reserve(num_frames);
    dof_positions.reserve(num_frames);
    dof_velocities.reserve(num_frames);

    for (int t = 0; t < num_frames; ++t)
    {
        // root 默认取 body[0]，通常是 pelvis/root
        Eigen::Vector3f root_pos(
            body_pos_all[(t * body_pos_shape[1] + 0) * 3 + 0],
            body_pos_all[(t * body_pos_shape[1] + 0) * 3 + 1],
            body_pos_all[(t * body_pos_shape[1] + 0) * 3 + 2]);
        root_positions.push_back(root_pos);

        Eigen::Quaternionf root_q(
            body_quat_all[(t * body_quat_shape[1] + 0) * 4 + 0],
            body_quat_all[(t * body_quat_shape[1] + 0) * 4 + 1],
            body_quat_all[(t * body_quat_shape[1] + 0) * 4 + 2],
            body_quat_all[(t * body_quat_shape[1] + 0) * 4 + 3]);
        root_quaternions.push_back(root_q.normalized());

        Eigen::VectorXf q = Eigen::VectorXf::Zero(dof_dim);
        Eigen::VectorXf dq = Eigen::VectorXf::Zero(dof_dim);

        for (int j = 0; j < dof_dim; ++j)
        {
            q(j) = joint_pos_all[t * dof_dim + j];
            dq(j) = joint_vel_all[t * dof_dim + j];
        }

        dof_positions.push_back(q);
        dof_velocities.push_back(dq);
    }

    update(0.0f);
}

void State_Mimic::MotionLoader_::update(float time_s)
{
    if (duration <= 0.0f)
    {
        index_0_ = 0;
        index_1_ = 0;
        blend_ = 0.0f;
        return;
    }

    const float phase = std::clamp(time_s / duration, 0.0f, 1.0f);
    const float frame_f = phase * (num_frames - 1);

    index_0_ = static_cast<int>(std::floor(frame_f));
    index_1_ = std::min(index_0_ + 1, num_frames - 1);
    blend_ = frame_f - index_0_;
}

void State_Mimic::MotionLoader_::reset(const isaaclab::ArticulationData &data, float t)
{
    (void)data;
    update(t);
}

Eigen::VectorXf State_Mimic::MotionLoader_::joint_pos() const
{
    return dof_positions[index_0_] * (1.0f - blend_) + dof_positions[index_1_] * blend_;
}

Eigen::VectorXf State_Mimic::MotionLoader_::joint_vel() const
{
    return dof_velocities[index_0_] * (1.0f - blend_) + dof_velocities[index_1_] * blend_;
}

Eigen::Vector3f State_Mimic::MotionLoader_::root_position() const
{
    return root_positions[index_0_] * (1.0f - blend_) + root_positions[index_1_] * blend_;
}

Eigen::Quaternionf State_Mimic::MotionLoader_::root_quaternion() const
{
    return root_quaternions[index_0_].slerp(blend_, root_quaternions[index_1_]).normalized();
}

State_Mimic::State_Mimic(int state_mode, std::string state_string)
    : FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    auto articulation =
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate);

    std::filesystem::path motion_file = cfg["motion_file"].as<std::string>();
    if (!motion_file.is_absolute())
    {
        motion_file = param::proj_dir / motion_file;
    }

    motion_ = std::make_shared<MotionLoader_>(motion_file.string(), cfg["fps"].as<float>());
    motion = motion_;
    spdlog::info("Loaded motion file '{}' with duration {:.2f}s",
                 motion_file.stem().string(), motion_->duration);

    if (cfg["time_start"])
        time_range_[0] = std::clamp(cfg["time_start"].as<float>(), 0.0f, motion_->duration);
    else
        time_range_[0] = 0.0f;

    if (cfg["time_end"])
        time_range_[1] = std::clamp(cfg["time_end"].as<float>(), 0.0f, motion_->duration);
    else
        time_range_[1] = motion_->duration;

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        articulation);
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    std::cout << "[MIMIC] deploy path: " << (policy_dir / "params" / "deploy.yaml") << std::endl;
    std::cout << "[MIMIC] onnx path: " << (policy_dir / "exported" / "policy.onnx") << std::endl;
    std::cout << "[MIMIC] action dim = " << env->action_manager->action().size() << std::endl;

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]() -> bool
            { return (env->episode_length * env->step_dt) > time_range_[1]; },
            FSMStringMap.right.at("Velocity")));

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]() -> bool
            { return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")));
}

void State_Mimic::enter()
{
    for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
    {
        lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
        lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
        lowcmd->msg_.motor_cmd()[i].dq() = 0.0f;
        lowcmd->msg_.motor_cmd()[i].tau() = 0.0f;
    }

    motion = motion_;
    env->reset();

    policy_thread_running = true;
    policy_thread = std::thread([this]
                                {
        using clock = std::chrono::high_resolution_clock;
        const std::chrono::duration<double> desiredDuration(env->step_dt);
        const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

        auto sleepTill = clock::now() + dt;

        // 与 29dof 部署版一致：进入 mimic 时做 yaw 对齐
        auto ref_yaw = isaaclab::yawQuaternion(motion->root_quaternion()).toRotationMatrix();
        auto robot_yaw = isaaclab::yawQuaternion(robot_anchor_quat_w(env.get())).toRotationMatrix();
        init_quat = Eigen::Quaternionf(robot_yaw * ref_yaw.transpose());

        motion->reset(env->robot->data, time_range_[0]);
        env->reset();

        while (policy_thread_running)
        {
            env->robot->update();
            motion->update(env->episode_length * env->step_dt + time_range_[0]);
            env->step();

            std::this_thread::sleep_until(sleepTill);
            sleepTill += dt;
        } });
}

void State_Mimic::run()
{
    auto action = env->action_manager->processed_actions();
    for (int i = 0; i < env->robot->data.joint_ids_map.size(); ++i)
    {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}