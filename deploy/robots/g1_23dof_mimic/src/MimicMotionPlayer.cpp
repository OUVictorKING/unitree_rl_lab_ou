#include "MimicMotionPlayer.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

#include "cnpy.h"

namespace
{
    template <typename T>
    std::vector<float> copy_to_float_vec(const T *src, std::size_t n)
    {
        std::vector<float> out(n);
        for (std::size_t i = 0; i < n; ++i)
            out[i] = static_cast<float>(src[i]);
        return out;
    }
}

MimicMotionPlayer::MimicMotionPlayer(
    const std::string &npz_path,
    const std::vector<std::string> &tracked_body_names)
{
    load(npz_path, tracked_body_names);
}

void MimicMotionPlayer::load(
    const std::string &npz_path,
    const std::vector<std::string> &tracked_body_names)
{
    tracked_body_names_ = tracked_body_names;
    tracked_body_indices_.clear();

    cnpy::npz_t npz = cnpy::npz_load(npz_path);

    if (!npz.count("fps") ||
        !npz.count("joint_pos") ||
        !npz.count("joint_vel") ||
        !npz.count("body_pos_w") ||
        !npz.count("body_quat_w"))
    {
        throw std::runtime_error("npz missing required keys");
    }

    // fps
    {
        const auto &arr = npz["fps"];
        if (arr.word_size == sizeof(float))
        {
            fps_ = *arr.data<float>();
        }
        else if (arr.word_size == sizeof(double))
        {
            fps_ = static_cast<float>(*arr.data<double>());
        }
        else
        {
            throw std::runtime_error("fps dtype unsupported");
        }
    }

    // joint_pos [T, J]
    {
        const auto &arr = npz["joint_pos"];
        if (arr.shape.size() != 2)
            throw std::runtime_error("joint_pos must be [T, J]");
        total_steps_ = arr.shape[0];
        joint_dim_ = arr.shape[1];

        if (joint_dim_ != 23)
        {
            throw std::runtime_error("joint_pos dim is not 23");
        }

        if (arr.word_size == sizeof(float))
        {
            joint_pos_all_ = copy_to_float_vec(arr.data<float>(), total_steps_ * joint_dim_);
        }
        else if (arr.word_size == sizeof(double))
        {
            joint_pos_all_ = copy_to_float_vec(arr.data<double>(), total_steps_ * joint_dim_);
        }
        else
        {
            throw std::runtime_error("joint_pos dtype unsupported");
        }
    }

    // joint_vel [T, J]
    {
        const auto &arr = npz["joint_vel"];
        if (arr.shape.size() != 2)
            throw std::runtime_error("joint_vel must be [T, J]");
        if (arr.shape[0] != total_steps_ || arr.shape[1] != joint_dim_)
        {
            throw std::runtime_error("joint_vel shape mismatch");
        }

        if (arr.word_size == sizeof(float))
        {
            joint_vel_all_ = copy_to_float_vec(arr.data<float>(), total_steps_ * joint_dim_);
        }
        else if (arr.word_size == sizeof(double))
        {
            joint_vel_all_ = copy_to_float_vec(arr.data<double>(), total_steps_ * joint_dim_);
        }
        else
        {
            throw std::runtime_error("joint_vel dtype unsupported");
        }
    }

    // body_pos_w [T, B, 3]
    {
        const auto &arr = npz["body_pos_w"];
        if (arr.shape.size() != 3 || arr.shape[2] != 3)
        {
            throw std::runtime_error("body_pos_w must be [T, B, 3]");
        }
        if (arr.shape[0] != total_steps_)
        {
            throw std::runtime_error("body_pos_w T mismatch");
        }

        full_body_count_ = arr.shape[1];

        if (arr.word_size == sizeof(float))
        {
            body_pos_w_all_ = copy_to_float_vec(arr.data<float>(), total_steps_ * full_body_count_ * 3);
        }
        else if (arr.word_size == sizeof(double))
        {
            body_pos_w_all_ = copy_to_float_vec(arr.data<double>(), total_steps_ * full_body_count_ * 3);
        }
        else
        {
            throw std::runtime_error("body_pos_w dtype unsupported");
        }
    }

    // body_quat_w [T, B, 4]
    {
        const auto &arr = npz["body_quat_w"];
        if (arr.shape.size() != 3 || arr.shape[2] != 4)
        {
            throw std::runtime_error("body_quat_w must be [T, B, 4]");
        }
        if (arr.shape[0] != total_steps_ || arr.shape[1] != full_body_count_)
        {
            throw std::runtime_error("body_quat_w shape mismatch");
        }

        if (arr.word_size == sizeof(float))
        {
            body_quat_w_all_ = copy_to_float_vec(arr.data<float>(), total_steps_ * full_body_count_ * 4);
        }
        else if (arr.word_size == sizeof(double))
        {
            body_quat_w_all_ = copy_to_float_vec(arr.data<double>(), total_steps_ * full_body_count_ * 4);
        }
        else
        {
            throw std::runtime_error("body_quat_w dtype unsupported");
        }
    }

    // 用 23DOF 机器人全 body 名，映射到训练 tracking body_names
    const auto full_body_names = get_g1_23dof_full_body_names();
    if (full_body_names.size() != full_body_count_)
    {
        throw std::runtime_error("full body name list size != npz body count");
    }

    std::unordered_map<std::string, std::size_t> body_name_to_idx;
    for (std::size_t i = 0; i < full_body_names.size(); ++i)
    {
        body_name_to_idx[full_body_names[i]] = i;
    }

    for (const auto &name : tracked_body_names_)
    {
        auto it = body_name_to_idx.find(name);
        if (it == body_name_to_idx.end())
        {
            throw std::runtime_error("tracked body not found in full body list: " + name);
        }
        tracked_body_indices_.push_back(it->second);
    }

    tracked_body_count_ = tracked_body_indices_.size();

    auto it_anchor = std::find(tracked_body_names_.begin(), tracked_body_names_.end(), "torso_link");
    if (it_anchor == tracked_body_names_.end())
    {
        throw std::runtime_error("torso_link not found in tracked_body_names");
    }
    anchor_body_local_index_ = std::distance(tracked_body_names_.begin(), it_anchor);

    reset(0);
}

void MimicMotionPlayer::reset(std::size_t start_step)
{
    if (total_steps_ == 0)
        throw std::runtime_error("motion not loaded");
    current_step_ = std::min(start_step, total_steps_ - 1);
    update_current_cache();
}

void MimicMotionPlayer::step()
{
    if (total_steps_ == 0)
        throw std::runtime_error("motion not loaded");
    current_step_ += 1;
    if (current_step_ >= total_steps_)
    {
        current_step_ = 0;
    }
    update_current_cache();
}

void MimicMotionPlayer::set_step(std::size_t step)
{
    if (total_steps_ == 0)
        throw std::runtime_error("motion not loaded");
    current_step_ = std::min(step, total_steps_ - 1);
    update_current_cache();
}

void MimicMotionPlayer::update_current_cache() const
{
    // joint
    joint_pos_cur_.resize(joint_dim_);
    joint_vel_cur_.resize(joint_dim_);
    for (std::size_t j = 0; j < joint_dim_; ++j)
    {
        joint_pos_cur_[j] = joint_pos_all_[current_step_ * joint_dim_ + j];
        joint_vel_cur_[j] = joint_vel_all_[current_step_ * joint_dim_ + j];
    }

    // tracked bodies only
    tracked_body_pos_cur_.resize(tracked_body_count_ * 3);
    tracked_body_quat_cur_.resize(tracked_body_count_ * 4);

    for (std::size_t bi = 0; bi < tracked_body_count_; ++bi)
    {
        const std::size_t full_idx = tracked_body_indices_[bi];

        for (std::size_t k = 0; k < 3; ++k)
        {
            tracked_body_pos_cur_[bi * 3 + k] =
                body_pos_w_all_[(current_step_ * full_body_count_ + full_idx) * 3 + k];
        }

        for (std::size_t k = 0; k < 4; ++k)
        {
            tracked_body_quat_cur_[bi * 4 + k] =
                body_quat_w_all_[(current_step_ * full_body_count_ + full_idx) * 4 + k];
        }
    }
}

const std::vector<float> &MimicMotionPlayer::joint_pos() const
{
    return joint_pos_cur_;
}

const std::vector<float> &MimicMotionPlayer::joint_vel() const
{
    return joint_vel_cur_;
}

const std::vector<float> &MimicMotionPlayer::tracked_body_pos_w_flat() const
{
    return tracked_body_pos_cur_;
}

const std::vector<float> &MimicMotionPlayer::tracked_body_quat_w_flat() const
{
    return tracked_body_quat_cur_;
}

std::array<float, 3> MimicMotionPlayer::anchor_pos_w() const
{
    const std::size_t i = anchor_body_local_index_;
    return {
        tracked_body_pos_cur_[i * 3 + 0],
        tracked_body_pos_cur_[i * 3 + 1],
        tracked_body_pos_cur_[i * 3 + 2]};
}

MimicMotionPlayer::QuatWXYZ MimicMotionPlayer::anchor_quat_w() const
{
    const std::size_t i = anchor_body_local_index_;
    return {
        tracked_body_quat_cur_[i * 4 + 0],
        tracked_body_quat_cur_[i * 4 + 1],
        tracked_body_quat_cur_[i * 4 + 2],
        tracked_body_quat_cur_[i * 4 + 3]};
}

std::vector<float> MimicMotionPlayer::motion_command() const
{
    std::vector<float> out;
    out.reserve(joint_dim_ * 2);
    out.insert(out.end(), joint_pos_cur_.begin(), joint_pos_cur_.end());
    out.insert(out.end(), joint_vel_cur_.begin(), joint_vel_cur_.end());
    return out;
}

std::vector<float> MimicMotionPlayer::motion_anchor_ori_b(const QuatWXYZ &robot_anchor_quat_w) const
{
    // 对应训练里的 subtract_frame_transforms(robot_anchor, target_anchor) 的相对姿态
    const QuatWXYZ target_anchor = anchor_quat_w();
    const QuatWXYZ rel = quat_mul(quat_inv(robot_anchor_quat_w), target_anchor);
    const auto R = quat_to_rotmat(rel);

    // 对应 Python:
    // mat[..., :2].reshape(...)
    return {
        R[0], R[1],
        R[3], R[4],
        R[6], R[7]};
}

MimicMotionPlayer::QuatWXYZ MimicMotionPlayer::quat_inv(const QuatWXYZ &q)
{
    const float n2 = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
    if (n2 < 1e-12f)
        throw std::runtime_error("quat_inv zero norm");
    return {q.w / n2, -q.x / n2, -q.y / n2, -q.z / n2};
}

MimicMotionPlayer::QuatWXYZ MimicMotionPlayer::quat_mul(const QuatWXYZ &a, const QuatWXYZ &b)
{
    return {
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w};
}

std::array<float, 9> MimicMotionPlayer::quat_to_rotmat(const QuatWXYZ &q0)
{
    const float n = std::sqrt(q0.w * q0.w + q0.x * q0.x + q0.y * q0.y + q0.z * q0.z);
    if (n < 1e-12f)
        throw std::runtime_error("quat_to_rotmat zero norm");

    const float w = q0.w / n;
    const float x = q0.x / n;
    const float y = q0.y / n;
    const float z = q0.z / n;

    return {
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - z * w), 2.f * (x * z + y * w),
        2.f * (x * y + z * w), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - x * w),
        2.f * (x * z - y * w), 2.f * (y * z + x * w), 1.f - 2.f * (x * x + y * y)};
}

std::vector<std::string> MimicMotionPlayer::get_g1_23dof_full_body_names()
{
    // 这里必须和 23DOF MuJoCo / IsaacLab 生成 npz 时 body 顺序一致
    // 你之前 IsaacLab 报错里 Available strings 已经给过这份 body 名列表。:contentReference[oaicite:7]{index=7}
    return {
        "pelvis",
        "left_hip_pitch_link",
        "right_hip_pitch_link",
        "torso_link",
        "left_hip_roll_link",
        "right_hip_roll_link",
        "left_shoulder_pitch_link",
        "right_shoulder_pitch_link",
        "left_hip_yaw_link",
        "right_hip_yaw_link",
        "left_shoulder_roll_link",
        "right_shoulder_roll_link",
        "left_knee_link",
        "right_knee_link",
        "left_shoulder_yaw_link",
        "right_shoulder_yaw_link",
        "left_ankle_pitch_link",
        "right_ankle_pitch_link",
        "left_elbow_link",
        "right_elbow_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_roll_rubber_hand",
        "right_wrist_roll_rubber_hand"};
}