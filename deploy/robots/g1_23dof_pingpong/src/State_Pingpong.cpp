#include "State_Pingpong.h"
#include "isaaclab/utils/utils.h"
#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "cnpy.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <ctime>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <cmath>
#include <iostream>
#include <sstream>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#include <zlib.h>

extern char **environ;

namespace
{
constexpr float kGravity = 9.81f;
std::atomic<bool> g_use_local_sim_time{false};

double now_seconds()
{
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    return std::chrono::duration<double>(clock::now() - t0).count();
}

double stamp_seconds(const builtin_interfaces::msg::Time &stamp)
{
    return static_cast<double>(stamp.sec) + 1.0e-9 * static_cast<double>(stamp.nanosec);
}

Eigen::Vector3f yaml_vec3(const YAML::Node &node, const Eigen::Vector3f &fallback)
{
    if (!node || !node.IsSequence() || node.size() < 3)
        return fallback;
    return Eigen::Vector3f(node[0].as<float>(), node[1].as<float>(), node[2].as<float>());
}

Eigen::Vector2f yaml_vec2(const YAML::Node &node, const Eigen::Vector2f &fallback)
{
    if (!node || !node.IsSequence() || node.size() < 2)
        return fallback;
    return Eigen::Vector2f(node[0].as<float>(), node[1].as<float>());
}

Eigen::Quaternionf yaml_quat_wxyz(const YAML::Node &node, const Eigen::Quaternionf &fallback)
{
    if (!node || !node.IsSequence() || node.size() < 4)
        return fallback;
    return Eigen::Quaternionf(node[0].as<float>(), node[1].as<float>(), node[2].as<float>(), node[3].as<float>()).normalized();
}

template <typename T>
T yaml_value(const YAML::Node &node, const std::string &key, T fallback)
{
    if (!node || !node[key])
        return fallback;
    return node[key].as<T>();
}

float clamp_scalar(float x, float lo, float hi)
{
    return std::max(lo, std::min(x, hi));
}

bool usable_path_value(const YAML::Node &node)
{
    if (!node || !node.IsScalar())
        return false;
    const std::string value = node.as<std::string>();
    return !value.empty() && value.rfind("REPLACE_WITH_", 0) != 0;
}

std::filesystem::path resolve_project_path(const std::string &value)
{
    std::filesystem::path path(value);
    return path.is_absolute() ? path : (param::proj_dir / path);
}

pid_t spawn_process(const std::vector<std::string> &args)
{
    if (args.empty())
        return -1;

    std::vector<char *> argv;
    argv.reserve(args.size() + 1);
    for (const auto &arg : args)
        argv.push_back(const_cast<char *>(arg.c_str()));
    argv.push_back(nullptr);

    posix_spawnattr_t attr;
    if (posix_spawnattr_init(&attr) != 0)
        return -1;
    short flags = POSIX_SPAWN_SETPGROUP;
    posix_spawnattr_setflags(&attr, flags);
    // pgroup=0 makes the child become the leader of a new process group, so
    // stop_ros_bag_process() can signal the whole shell/rosbag process tree.
    posix_spawnattr_setpgroup(&attr, 0);

    pid_t pid = -1;
    const int rc = posix_spawnp(&pid, args[0].c_str(), nullptr, &attr, argv.data(), environ);
    posix_spawnattr_destroy(&attr);
    if (rc != 0)
    {
        errno = rc;
        spdlog::error("posix_spawnp failed for '{}': {}", args[0], std::strerror(rc));
        return -1;
    }
    return pid;
}

std::string shell_quote(const std::string &value)
{
    std::string out = "'";
    for (const char c : value)
    {
        if (c == '\'')
            out += "'\\''";
        else
            out += c;
    }
    out += "'";
    return out;
}

pid_t spawn_shell_command(const std::string &command)
{
    return spawn_process({"/bin/bash", "-lc", command});
}

bool child_exited(pid_t pid, int *status_out = nullptr)
{
    if (pid <= 0)
        return true;
    int status = 0;
    const pid_t done = waitpid(pid, &status, WNOHANG);
    if (done == pid)
    {
        if (status_out)
            *status_out = status;
        return true;
    }
    if (done < 0 && errno == ECHILD)
    {
        if (status_out)
            *status_out = -1;
        return true;
    }
    return false;
}

std::string timestamp_suffix()
{
    using clock = std::chrono::system_clock;
    const auto now = clock::to_time_t(clock::now());
    std::tm tm{};
    localtime_r(&now, &tm);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
    return std::string(buf);
}

uint16_t read_le16(const unsigned char *p)
{
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

uint32_t read_le32(const unsigned char *p)
{
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

uint64_t read_le64(const unsigned char *p)
{
    uint64_t out = 0;
    for (int i = 7; i >= 0; --i)
        out = (out << 8) | static_cast<uint64_t>(p[i]);
    return out;
}

float yaw_from_wxyz_quat(const Eigen::Quaternionf &q_in)
{
    const Eigen::Quaternionf q = q_in.normalized();
    return std::atan2(2.0f * (q.w() * q.z() + q.x() * q.y()), 1.0f - 2.0f * (q.y() * q.y() + q.z() * q.z()));
}

float npy_value_as_float(const cnpy::NpyArray &arr, std::size_t idx)
{
    if (idx >= arr.num_vals)
        throw std::runtime_error("npz index out of range.");
    if (arr.word_size == sizeof(float))
        return arr.data<float>()[idx];
    if (arr.word_size == sizeof(double))
        return static_cast<float>(arr.data<double>()[idx]);
    throw std::runtime_error("npz array dtype must be float32 or float64.");
}

int npy_scalar_as_int(const cnpy::NpyArray &arr)
{
    if (arr.num_vals < 1)
        throw std::runtime_error("npz scalar array is empty.");
    if (arr.word_size == sizeof(std::int32_t))
        return static_cast<int>(arr.data<std::int32_t>()[0]);
    if (arr.word_size == sizeof(std::int64_t))
        return static_cast<int>(arr.data<std::int64_t>()[0]);
    if (arr.word_size == sizeof(float))
        return static_cast<int>(arr.data<float>()[0]);
    if (arr.word_size == sizeof(double))
        return static_cast<int>(arr.data<double>()[0]);
    throw std::runtime_error("impact_frame dtype must be int32/int64/float32/float64.");
}

std::string decode_numpy_string(const char *ptr, std::size_t word_size)
{
    std::string out;
    if (word_size >= 4)
    {
        bool looks_utf32 = (word_size % 4) == 0;
        for (std::size_t i = 0; looks_utf32 && i + 3 < std::min<std::size_t>(word_size, 64); i += 4)
        {
            if (ptr[i + 1] != 0 || ptr[i + 2] != 0 || ptr[i + 3] != 0)
            {
                looks_utf32 = false;
                break;
            }
            if (ptr[i] == 0)
                break;
        }
        if (looks_utf32)
        {
            for (std::size_t i = 0; i + 3 < word_size; i += 4)
            {
                const unsigned char c0 = static_cast<unsigned char>(ptr[i]);
                const unsigned char c1 = static_cast<unsigned char>(ptr[i + 1]);
                const unsigned char c2 = static_cast<unsigned char>(ptr[i + 2]);
                const unsigned char c3 = static_cast<unsigned char>(ptr[i + 3]);
                const std::uint32_t cp = c0 | (std::uint32_t(c1) << 8) | (std::uint32_t(c2) << 16) | (std::uint32_t(c3) << 24);
                if (cp == 0)
                    break;
                out.push_back(cp < 128 ? static_cast<char>(cp) : '?');
            }
            return out;
        }
    }
    for (std::size_t i = 0; i < word_size; ++i)
    {
        if (ptr[i] == '\0')
            break;
        out.push_back(ptr[i]);
    }
    return out;
}

std::vector<std::string> npy_string_vector(const cnpy::NpyArray &arr)
{
    std::vector<std::string> out;
    out.reserve(arr.num_vals);
    const char *raw = arr.data<char>();
    for (std::size_t i = 0; i < arr.num_vals; ++i)
        out.push_back(decode_numpy_string(raw + i * arr.word_size, arr.word_size));
    return out;
}

struct RawNpyArray
{
    std::vector<unsigned char> bytes;
    std::vector<std::size_t> shape;
    std::string descr;
    std::size_t word_size = 0;
    std::size_t data_offset = 0;
    std::size_t num_vals = 0;
};

std::size_t npy_word_size_from_descr(const std::string &descr)
{
    if (descr.size() < 2)
        throw std::runtime_error("bad npy descr: " + descr);
    const char kind = descr[0] == '<' || descr[0] == '>' || descr[0] == '|' ? descr[1] : descr[0];
    const std::size_t npos = descr.find_first_of("0123456789");
    if (npos == std::string::npos)
        throw std::runtime_error("bad npy descr without item size: " + descr);
    const std::size_t n = static_cast<std::size_t>(std::stoul(descr.substr(npos)));
    return kind == 'U' ? 4 * n : n;
}

RawNpyArray parse_raw_npy(std::vector<unsigned char> bytes)
{
    if (bytes.size() < 10 || std::memcmp(bytes.data(), "\x93NUMPY", 6) != 0)
        throw std::runtime_error("bad npy magic");
    const unsigned char major = bytes[6];
    std::size_t header_len = 0;
    std::size_t header_start = 0;
    if (major == 1)
    {
        header_len = read_le16(bytes.data() + 8);
        header_start = 10;
    }
    else if (major == 2 || major == 3)
    {
        if (bytes.size() < 12)
            throw std::runtime_error("bad npy v2/v3 header");
        header_len = read_le32(bytes.data() + 8);
        header_start = 12;
    }
    else
    {
        throw std::runtime_error("unsupported npy version");
    }
    if (header_start + header_len > bytes.size())
        throw std::runtime_error("npy header exceeds buffer");

    const std::string header(reinterpret_cast<const char *>(bytes.data() + header_start), header_len);
    auto parse_quoted_value = [&](const std::string &key) -> std::string {
        const std::size_t key_pos = header.find(key);
        if (key_pos == std::string::npos)
            throw std::runtime_error("npy header missing " + key);
        const std::size_t colon = header.find(':', key_pos);
        const std::size_t q1 = header.find('\'', colon);
        const std::size_t q2 = header.find('\'', q1 + 1);
        if (colon == std::string::npos || q1 == std::string::npos || q2 == std::string::npos)
            throw std::runtime_error("npy header malformed " + key);
        return header.substr(q1 + 1, q2 - q1 - 1);
    };

    RawNpyArray out;
    out.bytes = std::move(bytes);
    out.descr = parse_quoted_value("descr");
    out.word_size = npy_word_size_from_descr(out.descr);
    out.data_offset = header_start + header_len;

    const std::size_t shape_key = header.find("shape");
    const std::size_t lp = header.find('(', shape_key);
    const std::size_t rp = header.find(')', lp);
    if (shape_key == std::string::npos || lp == std::string::npos || rp == std::string::npos)
        throw std::runtime_error("npy header missing shape");
    const std::string shape_text = header.substr(lp + 1, rp - lp - 1);
    std::size_t pos = 0;
    while (pos < shape_text.size())
    {
        while (pos < shape_text.size() && !std::isdigit(static_cast<unsigned char>(shape_text[pos])))
            ++pos;
        if (pos >= shape_text.size())
            break;
        const std::size_t start = pos;
        while (pos < shape_text.size() && std::isdigit(static_cast<unsigned char>(shape_text[pos])))
            ++pos;
        out.shape.push_back(static_cast<std::size_t>(std::stoul(shape_text.substr(start, pos - start))));
    }
    out.num_vals = 1;
    for (const std::size_t dim : out.shape)
        out.num_vals *= dim;
    if (out.data_offset + out.num_vals * out.word_size > out.bytes.size())
        throw std::runtime_error("npy data exceeds buffer");
    return out;
}

RawNpyArray load_raw_npz_array(const std::string &path, const std::string &key)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot open npz: " + path);
    std::vector<unsigned char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const std::string wanted = key.size() >= 4 && key.substr(key.size() - 4) == ".npy" ? key : key + ".npy";

    std::size_t pos = 0;
    while (pos + 30 <= blob.size())
    {
        if (read_le32(blob.data() + pos) != 0x04034b50u)
        {
            ++pos;
            continue;
        }
        const uint16_t flags = read_le16(blob.data() + pos + 6);
        const uint16_t method = read_le16(blob.data() + pos + 8);
        uint64_t comp_size = read_le32(blob.data() + pos + 18);
        uint64_t uncomp_size = read_le32(blob.data() + pos + 22);
        const uint16_t name_len = read_le16(blob.data() + pos + 26);
        const uint16_t extra_len = read_le16(blob.data() + pos + 28);
        const std::size_t name_start = pos + 30;
        const std::size_t data_start = name_start + name_len + extra_len;
        if (data_start > blob.size())
            break;
        const std::string name(reinterpret_cast<const char *>(blob.data() + name_start), name_len);

        if (comp_size == 0xffffffffull || uncomp_size == 0xffffffffull)
        {
            std::size_t ep = name_start + name_len;
            const std::size_t extra_end = ep + extra_len;
            while (ep + 4 <= extra_end && ep + 4 <= blob.size())
            {
                const uint16_t header_id = read_le16(blob.data() + ep);
                const uint16_t data_size = read_le16(blob.data() + ep + 2);
                ep += 4;
                if (ep + data_size > extra_end || ep + data_size > blob.size())
                    break;
                if (header_id == 0x0001u)
                {
                    std::size_t zp = ep;
                    if (uncomp_size == 0xffffffffull && zp + 8 <= ep + data_size)
                    {
                        uncomp_size = read_le64(blob.data() + zp);
                        zp += 8;
                    }
                    if (comp_size == 0xffffffffull && zp + 8 <= ep + data_size)
                    {
                        comp_size = read_le64(blob.data() + zp);
                        zp += 8;
                    }
                    break;
                }
                ep += data_size;
            }
        }

        if (name == wanted)
        {
            if (flags & 0x0008u)
                throw std::runtime_error("npz entry uses data descriptor, unsupported for " + wanted);
            if (data_start + comp_size > blob.size())
                throw std::runtime_error("npz entry exceeds file for " + wanted);

            std::vector<unsigned char> raw;
            if (method == 0)
            {
                raw.assign(blob.begin() + static_cast<std::ptrdiff_t>(data_start),
                           blob.begin() + static_cast<std::ptrdiff_t>(data_start + static_cast<std::size_t>(comp_size)));
            }
            else if (method == 8)
            {
                raw.resize(static_cast<std::size_t>(uncomp_size));
                z_stream zs{};
                zs.next_in = const_cast<Bytef *>(reinterpret_cast<const Bytef *>(blob.data() + data_start));
                zs.avail_in = static_cast<uInt>(comp_size);
                zs.next_out = reinterpret_cast<Bytef *>(raw.data());
                zs.avail_out = static_cast<uInt>(uncomp_size);
                if (inflateInit2(&zs, -MAX_WBITS) != Z_OK)
                    throw std::runtime_error("inflateInit2 failed for " + wanted);
                const int ret = inflate(&zs, Z_FINISH);
                inflateEnd(&zs);
                if (ret != Z_STREAM_END)
                    throw std::runtime_error("inflate failed for " + wanted);
            }
            else
            {
                throw std::runtime_error("unsupported npz compression method for " + wanted);
            }
            return parse_raw_npy(std::move(raw));
        }
        if (flags & 0x0008u)
            throw std::runtime_error("npz entry data descriptor appears before requested key: " + wanted);
        pos = data_start + comp_size;
    }
    throw std::runtime_error("npz key not found: " + wanted);
}

float raw_npy_value_as_float(const RawNpyArray &arr, std::size_t idx)
{
    if (idx >= arr.num_vals)
        throw std::runtime_error("raw npy index out of range.");
    const char kind = arr.descr[0] == '<' || arr.descr[0] == '>' || arr.descr[0] == '|' ? arr.descr[1] : arr.descr[0];
    const unsigned char *p = arr.bytes.data() + arr.data_offset + idx * arr.word_size;
    if (kind == 'f' && arr.word_size == sizeof(float))
    {
        float v;
        std::memcpy(&v, p, sizeof(v));
        return v;
    }
    if (kind == 'f' && arr.word_size == sizeof(double))
    {
        double v;
        std::memcpy(&v, p, sizeof(v));
        return static_cast<float>(v);
    }
    throw std::runtime_error("raw npy dtype must be float32 or float64.");
}

int raw_npy_scalar_as_int(const RawNpyArray &arr)
{
    if (arr.num_vals < 1)
        throw std::runtime_error("raw npy scalar array is empty.");
    const char kind = arr.descr[0] == '<' || arr.descr[0] == '>' || arr.descr[0] == '|' ? arr.descr[1] : arr.descr[0];
    const unsigned char *p = arr.bytes.data() + arr.data_offset;
    if ((kind == 'i' || kind == 'u') && arr.word_size == sizeof(std::int32_t))
    {
        std::int32_t v;
        std::memcpy(&v, p, sizeof(v));
        return static_cast<int>(v);
    }
    if ((kind == 'i' || kind == 'u') && arr.word_size == sizeof(std::int64_t))
    {
        std::int64_t v;
        std::memcpy(&v, p, sizeof(v));
        return static_cast<int>(v);
    }
    if (kind == 'f')
        return static_cast<int>(raw_npy_value_as_float(arr, 0));
    throw std::runtime_error("raw npy scalar dtype must be int32/int64/float32/float64.");
}

std::vector<std::string> raw_npy_string_vector(const RawNpyArray &arr)
{
    std::vector<std::string> out;
    out.reserve(arr.num_vals);
    const char *raw = reinterpret_cast<const char *>(arr.bytes.data() + arr.data_offset);
    for (std::size_t i = 0; i < arr.num_vals; ++i)
        out.push_back(decode_numpy_string(raw + i * arr.word_size, arr.word_size));
    return out;
}

} // namespace

void State_Pingpong::set_use_local_sim_time(bool enabled)
{
    g_use_local_sim_time.store(enabled);
    spdlog::info(
        "Pingpong controller time source: {}",
        enabled ? "local simulation ROS header stamp (--network lo/lo0)" : "steady_clock / real robot");
}

State_Pingpong::State_Pingpong(int state_mode, std::string state_string)
    : FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    load_config(cfg);
    load_policy(cfg);
    start_ros_if_enabled(cfg);

    if (entry_joint_pos_.empty())
        entry_joint_pos_ = io_.action_offset;
    else
        entry_joint_pos_ = remap_full_or_policy(entry_joint_pos_, io_.joint_ids_map);
    if (entry_joint_pos_.size() != static_cast<size_t>(io_.action_dim))
        throw std::runtime_error("Pingpong entry_joint_pos dim mismatch.");
    switch_entry_joint_pos_ = entry_joint_pos_;

    cmd_ = make_fallback_command();
    last_raw_action_.assign(io_.action_dim, 0.0f);
    current_pd_target_ = switch_entry_joint_pos_;
    set_safe_targets_locked();

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]() -> bool {
                robot_->update();
                const float z = robot_->data.projected_gravity_b.z();
                const float tilt = std::acos(clamp_scalar(-z, -1.0f, 1.0f));
                const bool fallen = tilt > 1.0f;
                if (fallen)
                {
                    spdlog::warn(
                        "Pingpong fall check triggered: tilt={:.1f}deg projected_gravity_b=[{:.3f},{:.3f},{:.3f}]",
                        tilt * 180.0f / static_cast<float>(M_PI),
                        robot_->data.projected_gravity_b.x(),
                        robot_->data.projected_gravity_b.y(),
                        robot_->data.projected_gravity_b.z());
                }
                return fallen;
            },
            FSMStringMap.right.at("Passive")));
}

State_Pingpong::~State_Pingpong()
{
    policy_thread_running_ = false;
    if (policy_thread_.joinable())
        policy_thread_.join();
    stop_ros();
}

void State_Pingpong::load_config(const YAML::Node &cfg)
{
    const auto planner = cfg["planner"];
    const auto world = cfg["world"];
    const auto safety = cfg["safety"];
    const auto logging = cfg["logging"];

    reset_root_pos_ = yaml_vec3(world["reset_root_pos"], reset_root_pos_);
    reset_root_quat_ = yaml_quat_wxyz(world["reset_root_quat_wxyz"], reset_root_quat_);
    target_land_world_ = yaml_vec3(planner["target_land_world"], target_land_world_);
    forehand_offset_base_ = yaml_vec2(planner["forehand_offset_base"], forehand_offset_base_);
    backhand_offset_base_ = yaml_vec2(planner["backhand_offset_base"], backhand_offset_base_);
    load_training_geometry_from_npz(planner);
    const YAML::Node entry_node = planner["entry_motion_file"] ? planner["entry_motion_file"] : planner["forward_motion_file"];
    if (usable_path_value(entry_node))
    {
        std::filesystem::path entry_path = entry_node.as<std::string>();
        if (entry_path.is_relative())
            entry_path = param::proj_dir / entry_path;
        const int entry_frame = yaml_value<int>(planner, "entry_motion_frame", 0);
        entry_joint_pos_ = load_joint_pos_frame_from_npz(entry_path.string(), entry_frame);
        spdlog::info("Pingpong entry pose loaded from npz frame {}: {}", entry_frame, entry_path.string());
    }
    entry_joint_mode_ = yaml_value<std::string>(planner, "entry_joint_mode", entry_joint_mode_);
    const YAML::Node fallback_node = planner["fallback_motion_file"] ? planner["fallback_motion_file"] : planner["forward_motion_file"];
    if (usable_path_value(fallback_node))
    {
        std::filesystem::path fallback_path = fallback_node.as<std::string>();
        if (fallback_path.is_relative())
            fallback_path = param::proj_dir / fallback_path;
        const int fallback_frame = yaml_value<int>(planner, "fallback_motion_frame", -1);
        fallback_ref_ = load_fallback_reference_from_npz(fallback_path.string(), fallback_frame);
        spdlog::info(
            "Pingpong fallback reference loaded from npz frame {}: hit_offset_b=[{:.4f},{:.4f},{:.4f}], v_racket_b=[{:.4f},{:.4f},{:.4f}], n_b=[{:.4f},{:.4f},{:.4f}], swing={}",
            fallback_ref_.frame,
            fallback_ref_.hit_offset_base.x(), fallback_ref_.hit_offset_base.y(), fallback_ref_.hit_offset_base.z(),
            fallback_ref_.racket_vel_base.x(), fallback_ref_.racket_vel_base.y(), fallback_ref_.racket_vel_base.z(),
            fallback_ref_.normal_base.x(), fallback_ref_.normal_base.y(), fallback_ref_.normal_base.z(),
            fallback_ref_.swing_type);
    }

    const auto input = cfg["input_frame"];
    input_origin_in_training_world_ = yaml_vec3(input["origin_in_training_world"], input_origin_in_training_world_);
    input_to_training_quat_ = yaml_quat_wxyz(input["rotation_wxyz_to_training"], input_to_training_quat_);

    const float forehand_y = forehand_offset_base_.y();
    const float backhand_y = backhand_offset_base_.y();
    float forehand_y_eff = forehand_y;
    if (!planner["forehand_y_safety_clamp"] || !planner["forehand_y_safety_clamp"].IsNull())
    {
        const float cap = std::abs(yaml_value<float>(planner, "forehand_y_safety_clamp", 0.40f));
        forehand_y_eff = forehand_y < 0.0f ? std::max(forehand_y, -cap) : std::min(forehand_y, cap);
    }
    y_mid_base_ = (planner["y_mid_base"] && !planner["y_mid_base"].IsNull())
                      ? planner["y_mid_base"].as<float>()
                      : 0.5f * (forehand_y_eff + backhand_y);
    swing_y_sign_ = forehand_y > backhand_y ? 1.0f : -1.0f;
    x_hit_world_ = (planner["x_hit_world"] && !planner["x_hit_world"].IsNull())
                       ? planner["x_hit_world"].as<float>()
                       : 0.5f * (forehand_offset_base_.x() + backhand_offset_base_.x());

    z_min_world_ = yaml_value<float>(planner, "z_min_world", z_min_world_);
    z_max_world_ = yaml_value<float>(planner, "z_max_world", z_max_world_);
    table_top_z_ = yaml_value<float>(world, "table_top_z", table_top_z_);
    ball_radius_ = yaml_value<float>(world, "ball_radius", ball_radius_);
    table_center_x_ = yaml_vec3(world["table_center"], Eigen::Vector3f(table_center_x_, table_center_y_, 0.735f)).x();
    table_center_y_ = yaml_vec3(world["table_center"], Eigen::Vector3f(table_center_x_, table_center_y_, 0.735f)).y();
    const Eigen::Vector3f table_size = yaml_vec3(world["table_size"], Eigen::Vector3f(2.74f, 1.525f, 0.05f));
    table_half_x_ = 0.5f * table_size.x();
    table_half_y_ = 0.5f * table_size.y();

    planner_dt_ = yaml_value<float>(planner, "planner_dt", planner_dt_);
    planner_max_time_ = yaml_value<float>(planner, "planner_max_time", planner_max_time_);
    planner_drag_k_ = yaml_value<float>(planner, "planner_drag_k", planner_drag_k_);
    planner_bounce_ch_ = yaml_value<float>(planner, "planner_bounce_ch", planner_bounce_ch_);
    planner_bounce_cv_ = yaml_value<float>(planner, "planner_bounce_cv", planner_bounce_cv_);
    planner_min_t_to_hit_ = yaml_value<float>(planner, "planner_min_t_to_hit", planner_min_t_to_hit_);
    planner_max_t_to_hit_ = yaml_value<float>(planner, "planner_max_t_to_hit", planner_max_t_to_hit_);
    planner_min_incoming_speed_x_ = yaml_value<float>(planner, "min_incoming_speed_x", planner_min_incoming_speed_x_);
    planner_min_ball_z_world_ = yaml_value<float>(planner, "min_ball_z_world", table_top_z_ - ball_radius_);
    planner_max_table_bounces_before_fallback_ =
        yaml_value<int>(planner, "max_table_bounces_before_fallback", planner_max_table_bounces_before_fallback_);
    freeze_time_before_hit_ = yaml_value<float>(planner, "freeze_time_before_hit", freeze_time_before_hit_);
    post_swing_time_ = yaml_value<float>(planner, "post_swing_time", post_swing_time_);
    post_hit_imitation_ = yaml_value<bool>(planner, "post_hit_imitation", post_hit_imitation_);
    flight_time_ = yaml_value<float>(planner, "flight_time", flight_time_);
    paddle_cor_ = yaml_value<float>(planner, "paddle_cor", paddle_cor_);

    topic_timeout_s_ = yaml_value<float>(safety, "topic_timeout_s", topic_timeout_s_);
    max_ball_sample_age_s_ = yaml_value<float>(safety, "max_ball_sample_age_s", max_ball_sample_age_s_);
    switch_blend_s_ = yaml_value<float>(safety, "switch_blend_s", switch_blend_s_);
    switch_blend_s_ = yaml_value<float>(cfg, "switch_blend_s", switch_blend_s_);
    actor_blend_s_ = yaml_value<float>(planner, "actor_blend_s", actor_blend_s_);
    actor_blend_s_ = yaml_value<float>(safety, "actor_blend_s", actor_blend_s_);
    actor_blend_s_ = yaml_value<float>(cfg, "actor_blend_s", actor_blend_s_);
    policy_dt_ = yaml_value<float>(cfg, "policy_dt", policy_dt_);
    const YAML::Node motor_gains = cfg["motor_gains"];
    keep_current_gains_during_switch_ = yaml_value<bool>(motor_gains, "keep_current_during_switch", keep_current_gains_during_switch_);
    support_gain_override_enable_ = yaml_value<bool>(motor_gains["support_override"], "enable", support_gain_override_enable_);
    if (support_gain_override_enable_)
    {
        support_gain_sdk_ids_ = yaml_int_vector_from_numeric(motor_gains["support_override"]["sdk_ids"], "motor_gains.support_override.sdk_ids");
        support_gain_kp_ = yaml_float_vector(motor_gains["support_override"]["kp"], "motor_gains.support_override.kp");
        support_gain_kd_ = yaml_float_vector(motor_gains["support_override"]["kd"], "motor_gains.support_override.kd");
        if (support_gain_sdk_ids_.size() != support_gain_kp_.size() || support_gain_sdk_ids_.size() != support_gain_kd_.size())
            throw std::runtime_error("motor_gains.support_override sdk_ids/kp/kd size mismatch.");
    }
    use_ros_header_stamp_ = yaml_value<bool>(cfg["ros"], "use_header_stamp", use_ros_header_stamp_);
    require_base_topic_ = yaml_value<bool>(cfg["ros"], "require_base_topic", require_base_topic_);
    const YAML::Node bag_record = cfg["ros"]["bag_record"];
    ros_bag_record_enable_ = yaml_value<bool>(bag_record, "enable", ros_bag_record_enable_);
    ros_bag_record_path_ = yaml_value<std::string>(bag_record, "output", ros_bag_record_path_);
    const YAML::Node bag_replay = cfg["ros"]["bag_replay"];
    ros_bag_replay_enable_ = yaml_value<bool>(bag_replay, "enable", ros_bag_replay_enable_);
    ros_bag_replay_path_ = yaml_value<std::string>(bag_replay, "input", ros_bag_replay_path_);
    ros_bag_replay_loop_ = yaml_value<bool>(bag_replay, "loop", ros_bag_replay_loop_);
    ros_bag_replay_rate_ = yaml_value<float>(bag_replay, "rate", ros_bag_replay_rate_);
    waiting_initial_t_to_hit_ = yaml_value<float>(planner, "waiting_initial_t_to_hit", waiting_initial_t_to_hit_);
    if (waiting_initial_t_to_hit_ >= 0.0f)
        waiting_initial_t_to_hit_ = -std::max(waiting_initial_t_to_hit_, policy_dt_);
    debug_control_log_enable_ = yaml_value<bool>(logging, "debug_control", debug_control_log_enable_);
    debug_actor_log_enable_ = yaml_value<bool>(logging, "debug_actor", debug_actor_log_enable_);
    planner_hit_log_enable_ = yaml_value<bool>(logging, "hit_window", planner_hit_log_enable_);
    planner_hit_log_window_s_ = yaml_value<float>(logging, "hit_window_s", planner_hit_log_window_s_);
    const YAML::Node hit_trace_csv = logging["hit_trace_csv"];
    hit_trace_csv_enable_ = yaml_value<bool>(hit_trace_csv, "enable", hit_trace_csv_enable_);
    hit_trace_csv_path_ = yaml_value<std::string>(hit_trace_csv, "output", hit_trace_csv_path_);
    planner_hit_log_window_s_ = std::max(0.0f, planner_hit_log_window_s_);

    spdlog::info(
        "Pingpong geometry: x_hit={:.4f}, y_mid={:.4f}, swing_sign={:.1f}, fh=[{:.3f},{:.3f}], bh=[{:.3f},{:.3f}]",
        x_hit_world_, y_mid_base_, swing_y_sign_, forehand_offset_base_.x(), forehand_offset_base_.y(),
        backhand_offset_base_.x(), backhand_offset_base_.y());
    spdlog::info(
        "Pingpong command timing: post_hit_imitation={}, post_swing_time={:.2f}s",
        post_hit_imitation_, post_swing_time_);
    spdlog::info(
        "Pingpong planner guards: min_incoming_speed_x={:.3f}, min_ball_z_world={:.3f}, max_table_bounces_before_fallback={}",
        planner_min_incoming_speed_x_, planner_min_ball_z_world_, planner_max_table_bounces_before_fallback_);
    spdlog::info(
        "Pingpong logging: debug_control={}, debug_actor={}, hit_window={} abs(t_to_hit)<={:.3f}s, fallback warnings=reason_changes_only, hit_trace_csv={}",
        debug_control_log_enable_, debug_actor_log_enable_, planner_hit_log_enable_,
        planner_hit_log_window_s_,
        hit_trace_csv_enable_ ? hit_trace_csv_path_ : "disabled");
}

void State_Pingpong::load_training_geometry_from_npz(const YAML::Node &planner)
{
    if (!usable_path_value(planner["forward_motion_file"]) || !usable_path_value(planner["backward_motion_file"]))
        return;

    std::filesystem::path forward_path = planner["forward_motion_file"].as<std::string>();
    std::filesystem::path backward_path = planner["backward_motion_file"].as<std::string>();
    if (forward_path.is_relative())
        forward_path = param::proj_dir / forward_path;
    if (backward_path.is_relative())
        backward_path = param::proj_dir / backward_path;
    if (!std::filesystem::exists(forward_path))
        throw std::runtime_error("planner.forward_motion_file does not exist: " + forward_path.string());
    if (!std::filesystem::exists(backward_path))
        throw std::runtime_error("planner.backward_motion_file does not exist: " + backward_path.string());

    forehand_offset_base_ = load_impact_offset_from_npz(forward_path.string());
    backhand_offset_base_ = load_impact_offset_from_npz(backward_path.string());

    spdlog::info(
        "Pingpong npz geometry loaded: forehand_offset=[{:.4f},{:.4f}], backhand_offset=[{:.4f},{:.4f}]",
        forehand_offset_base_.x(), forehand_offset_base_.y(), backhand_offset_base_.x(), backhand_offset_base_.y());
}

void State_Pingpong::load_policy(const YAML::Node &cfg)
{
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());
    auto deploy_yaml_path = policy_dir / "params" / "deploy.yaml";
    YAML::Node deploy = YAML::LoadFile(deploy_yaml_path);

    io_.joint_ids_map = yaml_int_vector_from_numeric(deploy["joint_ids_map"], "joint_ids_map");
    io_.default_joint_pos = yaml_float_vector(deploy["default_joint_pos"], "default_joint_pos");
    io_.stiffness = remap_full_or_policy(yaml_float_vector(deploy["stiffness"], "stiffness"), io_.joint_ids_map);
    io_.damping = remap_full_or_policy(yaml_float_vector(deploy["damping"], "damping"), io_.joint_ids_map);
    apply_motor_gain_overrides(cfg);

    const auto action = deploy["actions"]["JointPositionAction"];
    io_.action_scale = yaml_float_vector(action["scale"], "actions.JointPositionAction.scale");
    io_.action_offset = yaml_float_vector(action["offset"], "actions.JointPositionAction.offset");
    if (action["clip"] && action["clip"].IsSequence())
    {
        for (size_t i = 0; i < action["clip"].size(); ++i)
        {
            io_.action_clip.push_back(yaml_float_vector(action["clip"][i], "actions.JointPositionAction.clip[]"));
        }
    }
    io_.action_dim = static_cast<int>(io_.joint_ids_map.size());
    if ((int)io_.action_scale.size() != io_.action_dim || (int)io_.action_offset.size() != io_.action_dim)
        throw std::runtime_error("Pingpong action scale/offset dim mismatch with joint_ids_map.");

    io_.obs_terms.clear();
    for (auto it = deploy["observations"].begin(); it != deploy["observations"].end(); ++it)
    {
        ObsTermCfg term;
        term.name = it->first.as<std::string>();
        term.history_length = it->second["history_length"] ? it->second["history_length"].as<int>() : 1;
        if (it->second["scale"] && !it->second["scale"].IsNull())
            term.scale = yaml_float_vector(it->second["scale"], "observations." + term.name + ".scale");
        if (it->second["clip"] && !it->second["clip"].IsNull())
            term.clip = yaml_float_vector(it->second["clip"], "observations." + term.name + ".clip");
        int term_dim = static_cast<int>(term.scale.size());
        if (term_dim == 0)
        {
            if (term.name == "base_ang_vel" || term.name == "projected_gravity" || term.name == "hit_pos" ||
                term.name == "racket_vel" || term.name == "active_face" || term.name == "target_normal")
                term_dim = 3;
            else if (term.name == "base_yaw" || term.name == "base_err")
                term_dim = 2;
            else if (term.name == "t_to_hit")
                term_dim = 1;
            else if (term.name == "joint_pos" || term.name == "joint_vel" || term.name == "last_action")
                term_dim = io_.action_dim;
            else
                throw std::runtime_error("Cannot infer obs dim for term: " + term.name);
        }
        io_.obs_dim += term_dim * term.history_length;
        io_.obs_terms.push_back(term);
    }
    std::string obs_order;
    for (size_t i = 0; i < io_.obs_terms.size(); ++i)
    {
        if (i > 0)
            obs_order += ", ";
        obs_order += io_.obs_terms[i].name;
    }

    robot_ = std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate);
    robot_->data.joint_ids_map.assign(io_.joint_ids_map.begin(), io_.joint_ids_map.end());
    robot_->data.joint_pos.resize(io_.joint_ids_map.size());
    robot_->data.joint_vel.resize(io_.joint_ids_map.size());
    robot_->data.default_joint_pos = Eigen::VectorXf::Map(io_.default_joint_pos.data(), io_.default_joint_pos.size());
    robot_->data.joint_stiffness = io_.stiffness;
    robot_->data.joint_damping = io_.damping;
    robot_->update();

    actor_ = std::make_unique<isaaclab::OrtRunner>((policy_dir / "exported" / "policy.onnx").string());
    spdlog::info("Pingpong deploy yaml: {}", deploy_yaml_path.string());
    spdlog::info("Pingpong obs_dim={}, action_dim={}", io_.obs_dim, io_.action_dim);
    spdlog::info("Pingpong obs order: {}", obs_order);

    // Pingpong now mirrors the Mimic/RoboJuDo handoff: the state transition
    // itself interpolates to the npz entry pose, then the HITTER actor owns
    // every control tick.  The Velocity FSM state still exists outside this
    // state, but there is no internal Velocity fallback controller here.
}

void State_Pingpong::apply_motor_gain_overrides(const YAML::Node &)
{
    if (!support_gain_override_enable_)
        return;

    int applied = 0;
    std::ostringstream detail;
    for (size_t j = 0; j < support_gain_sdk_ids_.size(); ++j)
    {
        const int sdk_id = support_gain_sdk_ids_[j];
        auto it = std::find(io_.joint_ids_map.begin(), io_.joint_ids_map.end(), sdk_id);
        if (it == io_.joint_ids_map.end())
        {
            spdlog::warn("Pingpong support gain override ignored unknown sdk id {}", sdk_id);
            continue;
        }
        const int policy_i = static_cast<int>(std::distance(io_.joint_ids_map.begin(), it));
        const float old_kp = io_.stiffness[policy_i];
        const float old_kd = io_.damping[policy_i];
        io_.stiffness[policy_i] = support_gain_kp_[j];
        io_.damping[policy_i] = support_gain_kd_[j];
        if (applied < 6)
        {
            detail << " sdk=" << sdk_id
                   << " kp " << old_kp << "->" << io_.stiffness[policy_i]
                   << " kd " << old_kd << "->" << io_.damping[policy_i] << ";";
        }
        applied += 1;
    }
    spdlog::info("Pingpong support motor gain override applied to {} joints.{}", applied, detail.str());
}

void State_Pingpong::start_ros_if_enabled(const YAML::Node &cfg)
{
    const auto ros_cfg = cfg["ros"];
    if (!yaml_value<bool>(ros_cfg, "enable", true))
        return;
    if (!rclcpp::ok())
    {
        int argc = 0;
        char const *const *argv = nullptr;
        rclcpp::init(argc, argv, rclcpp::InitOptions(), rclcpp::SignalHandlerOptions::None);
    }

    ros2_node_ = std::make_shared<rclcpp::Node>("g1_23dof_pingpong_deploy");
    const bool use_sim_time_for_replay =
        yaml_value<bool>(ros_cfg, "use_sim_time_for_replay", false) || ros_bag_replay_enable_;
    if (use_sim_time_for_replay)
    {
        if (!ros2_node_->has_parameter("use_sim_time"))
            ros2_node_->declare_parameter<bool>("use_sim_time", true);
        ros2_node_->set_parameter(rclcpp::Parameter("use_sim_time", true));
        spdlog::info("Pingpong ROS2 use_sim_time enabled for bag replay; make sure /clock is being played.");
    }
    ball_topic_ = yaml_value<std::string>(ros_cfg, "ball_state_topic", ball_topic_);
    base_topic_ = yaml_value<std::string>(ros_cfg, "base_pose_topic", base_topic_);

    auto qos = rclcpp::SensorDataQoS().keep_last(1);
    ball_sub_ = ros2_node_->create_subscription<nav_msgs::msg::Odometry>(
        ball_topic_, qos, [this](const nav_msgs::msg::Odometry::SharedPtr msg) { this->ball_odom_cb(msg); });
    base_sub_ = ros2_node_->create_subscription<geometry_msgs::msg::PoseStamped>(
        base_topic_, qos, [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) { this->base_pose_cb(msg); });

    ros2_executor_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>();
    ros2_executor_->add_node(ros2_node_);
    ros2_thread_ = std::thread([this]() { ros2_executor_->spin(); });
    spdlog::info("Pingpong ROS2 Humble subscribers started: ball='{}', base='{}'; paddle face normal is computed by FK.",
                 ball_topic_, base_topic_);
}

void State_Pingpong::stop_ros()
{
    stop_ros_bag_tools();
    if (ros2_executor_)
        ros2_executor_->cancel();
    if (ros2_thread_.joinable())
        ros2_thread_.join();
    if (ros2_executor_ && ros2_node_)
        ros2_executor_->remove_node(ros2_node_);
    base_sub_.reset();
    ball_sub_.reset();
    ros2_executor_.reset();
    ros2_node_.reset();
    if (rclcpp::ok())
        rclcpp::shutdown();
}

void State_Pingpong::start_ros_bag_tools()
{
    if (!ros2_node_)
        return;
    if (ros_bag_replay_enable_ && ros_bag_record_enable_)
        spdlog::warn("Both ros.bag_replay.enable and ros.bag_record.enable are true; starting both because config requested it.");
    start_ros_bag_replay();
    start_ros_bag_record();
}

void State_Pingpong::stop_ros_bag_tools()
{
    stop_ros_bag_process(
        &ros_bag_record_pid_,
        "ros2 bag record",
        ros_bag_record_active_output_path_,
        ros_bag_record_active_log_path_);
    ros_bag_record_active_output_path_.clear();
    ros_bag_record_active_log_path_.clear();
    stop_ros_bag_process(&ros_bag_replay_pid_, "ros2 bag play");
}

void State_Pingpong::start_ros_bag_record()
{
    if (!ros_bag_record_enable_ || ros_bag_record_pid_ > 0)
        return;
    if (ros_bag_record_path_.empty())
    {
        spdlog::error("Pingpong ROS bag record requested but ros.bag_record.output is empty.");
        return;
    }
    auto out = resolve_project_path(ros_bag_record_path_);
    if (std::filesystem::exists(out))
    {
        const auto renamed = out.string() + "_" + timestamp_suffix();
        spdlog::warn("ROS bag record output already exists: '{}'; using '{}' instead.", out.string(), renamed);
        out = renamed;
    }
    std::filesystem::create_directories(out.parent_path());
    const std::filesystem::path log_path = out.string() + ".record.log";
    ros_bag_record_active_output_path_ = out.string();
    ros_bag_record_active_log_path_ = log_path.string();
    const auto wrapper = param::proj_dir / "scripts/rosbag_record_wrapper.sh";
    if (!std::filesystem::exists(wrapper))
    {
        spdlog::error("ros2 bag record wrapper not found: {}", wrapper.string());
        ros_bag_record_pid_ = -1;
        ros_bag_record_active_output_path_.clear();
        ros_bag_record_active_log_path_.clear();
        return;
    }
    ros_bag_record_pid_ = spawn_process({
        wrapper.string(),
        out.string(),
        log_path.string(),
        "/clock",
        ball_topic_,
        base_topic_,
    });
    if (ros_bag_record_pid_ <= 0)
        spdlog::error("Failed to start ros2 bag record.");
    else
    {
        int status = 0;
        if (child_exited(ros_bag_record_pid_, &status))
        {
            spdlog::error(
                "ros2 bag record exited immediately with status {}. Check log: {}",
                status, log_path.string());
            ros_bag_record_pid_ = -1;
            ros_bag_record_active_output_path_.clear();
            ros_bag_record_active_log_path_.clear();
        }
        else
        {
            spdlog::info(
                "Started ros2 bag record pid={} output='{}' log='{}'",
                ros_bag_record_pid_, out.string(), log_path.string());
        }
    }
}

void State_Pingpong::start_ros_bag_replay()
{
    if (!ros_bag_replay_enable_ || ros_bag_replay_pid_ > 0)
        return;
    if (ros_bag_replay_path_.empty())
    {
        spdlog::error("Pingpong ROS bag replay requested but ros.bag_replay.input is empty.");
        return;
    }
    const auto in = resolve_project_path(ros_bag_replay_path_);
    if (!std::filesystem::exists(in))
    {
        spdlog::error("Pingpong ROS bag replay input does not exist: '{}'", in.string());
        return;
    }
    const std::filesystem::path log_path = in.string() + ".replay.log";
    std::string command =
        "source /opt/ros/humble/setup.bash && exec ros2 bag play " +
        shell_quote(in.string()) + " --topics " +
        shell_quote("/clock") + " " + shell_quote(ball_topic_) + " " + shell_quote(base_topic_);
    if (ros_bag_replay_loop_)
        command += " --loop";
    if (std::abs(ros_bag_replay_rate_ - 1.0f) > 1.0e-4f)
    {
        command += " --rate " + shell_quote(std::to_string(ros_bag_replay_rate_));
    }
    // ros2 bag play can try to read keyboard controls.  Because we spawn it
    // as a background process group from the controller terminal, inheriting
    // stdin may stop it with SIGTTIN before it publishes any messages.
    command += " < /dev/null > " + shell_quote(log_path.string()) + " 2>&1";
    ros_bag_replay_pid_ = spawn_shell_command(command);
    if (ros_bag_replay_pid_ <= 0)
        spdlog::error("Failed to start ros2 bag play.");
    else
    {
        int status = 0;
        if (child_exited(ros_bag_replay_pid_, &status))
        {
            spdlog::error(
                "ros2 bag play exited immediately with status {}. Check log: {}",
                status, log_path.string());
            ros_bag_replay_pid_ = -1;
        }
        else
        {
            spdlog::info(
                "Started ros2 bag play pid={} input='{}' log='{}'",
                ros_bag_replay_pid_, in.string(), log_path.string());
        }
    }
}

void State_Pingpong::stop_ros_bag_process(
    pid_t *pid,
    const char *name,
    const std::string &output_path,
    const std::string &log_path)
{
    if (!pid || *pid <= 0)
        return;
    const pid_t child = *pid;
    *pid = -1;

    const std::string process_name = name ? name : "process";
    std::thread([child, process_name, output_path, log_path]() {
        auto report_bag_status = [&]() {
            if (output_path.empty())
                return;
            const std::filesystem::path out(output_path);
            const std::filesystem::path metadata = out / "metadata.yaml";
            if (std::filesystem::exists(metadata))
            {
                spdlog::info("Saved {} output='{}'", process_name, out.string());
            }
            else
            {
                spdlog::warn(
                    "{} stopped, but no bag metadata found at '{}'. Check log='{}'",
                    process_name, metadata.string(), log_path);
            }
        };
        auto wait_for_exit = [&](int loops, int sleep_ms) -> bool {
            for (int i = 0; i < loops; ++i)
            {
                if (child_exited(child))
                    return true;
                std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
            }
            return false;
        };
        auto signal_group = [&](int sig) {
            if (kill(-child, sig) != 0)
                kill(child, sig);
        };

        signal_group(SIGINT);
        if (wait_for_exit(150, 100))
        {
            spdlog::info("Stopped {} pid={} with SIGINT", process_name, child);
            report_bag_status();
            return;
        }

        signal_group(SIGTERM);
        if (wait_for_exit(50, 100))
        {
            spdlog::warn("Stopped {} pid={} with SIGTERM", process_name, child);
            report_bag_status();
            return;
        }

        signal_group(SIGKILL);
        if (wait_for_exit(50, 100))
        {
            spdlog::warn("Force-stopped {} pid={} with SIGKILL", process_name, child);
            report_bag_status();
        }
        else
            spdlog::error("Failed to reap {} pid={} after SIGKILL; continuing shutdown.", process_name, child);
    }).detach();
    spdlog::info("Requested {} pid={} stop with SIGINT in background.", process_name, child);
}

double State_Pingpong::controller_time_seconds() const
{
    if (local_sim_time_active_ && sim_time_valid_.load())
        return sim_time_s_.load();
    return now_seconds();
}

void State_Pingpong::observe_sim_time_stamp(const builtin_interfaces::msg::Time &stamp)
{
    if (!g_use_local_sim_time.load())
        return;
    if (stamp.sec == 0 && stamp.nanosec == 0)
        return;
    const double t = stamp_seconds(stamp);
    const double prev = sim_time_s_.load();
    if (sim_time_valid_.load() && t + 1.0 < prev)
    {
        spdlog::info("Pingpong local sim time reset/jump detected: {:.3f}s -> {:.3f}s", prev, t);
    }
    sim_time_s_.store(t);
    sim_time_valid_.store(true);
}

void State_Pingpong::enter()
{
    robot_->update();
    switch_start_q_.assign(io_.action_dim, 0.0f);
    for (int i = 0; i < io_.action_dim; ++i)
        switch_start_q_[i] = robot_->data.joint_pos[i];

    switch_entry_joint_pos_ = make_switch_entry_target(switch_start_q_);
    if (switch_entry_joint_pos_.size() != static_cast<size_t>(io_.action_dim))
        switch_entry_joint_pos_ = switch_start_q_;

    auto linf_diff = [](const std::vector<float> &a, const std::vector<float> &b) -> float {
        if (a.size() != b.size())
            return 0.0f;
        float out = 0.0f;
        for (size_t i = 0; i < a.size(); ++i)
            out = std::max(out, std::abs(a[i] - b[i]));
        return out;
    };
    const float z = robot_->data.projected_gravity_b.z();
    const float tilt = std::acos(clamp_scalar(-z, -1.0f, 1.0f));
    spdlog::info(
        "Pingpong enter: tilt={:.1f}deg blend_s={:.2f} actor_blend_s={:.2f} entry=npz_frame mode={} full_entry_linf={:.3f} switch_entry_linf={:.3f}",
        tilt * 180.0f / static_cast<float>(M_PI),
        switch_blend_s_,
        actor_blend_s_,
        entry_joint_mode_,
        linf_diff(switch_start_q_, entry_joint_pos_),
        linf_diff(switch_start_q_, switch_entry_joint_pos_));

    for (int i = 0; i < io_.action_dim; ++i)
    {
        const int sdk_id = io_.joint_ids_map[i];
        if (!keep_current_gains_during_switch_)
        {
            lowcmd->msg_.motor_cmd()[sdk_id].kp() = io_.stiffness[i];
            lowcmd->msg_.motor_cmd()[sdk_id].kd() = io_.damping[i];
        }
        lowcmd->msg_.motor_cmd()[sdk_id].dq() = 0.0f;
        lowcmd->msg_.motor_cmd()[sdk_id].tau() = 0.0f;
        lowcmd->msg_.motor_cmd()[sdk_id].q() = switch_start_q_[i];
    }
    if (keep_current_gains_during_switch_)
        spdlog::info("Pingpong switch blend keeps previous FSM motor gains; HITTER gains will be applied after switch_blend_s.");
    else
        spdlog::info("Pingpong switch blend applies HITTER motor gains immediately.");

    {
        std::lock_guard<std::mutex> lock(cmd_mtx_);
        current_pd_target_ = switch_entry_joint_pos_;
        cmd_ = make_fallback_command();
        active_control_ = false;
        actor_output_ready_ = false;
        actor_blend_active_ = false;
        actor_blend_start_time_s_ = 0.0;
        actor_blend_start_target_ = switch_entry_joint_pos_;
        policy_gains_applied_ = !keep_current_gains_during_switch_;
        command_active_ = false;
        command_frozen_ = false;
        has_live_planner_cmd_ = false;
        last_command_update_s_ = 0.0;
        last_raw_action_.assign(io_.action_dim, 0.0f);
        obs_history_.clear();
    }

    run_debug_counter_ = 0;
    last_waiting_reason_log_s_ = -10.0;
    last_waiting_reason_.clear();
    last_waiting_held_previous_ = false;
    hit_window_logged_ = false;
    {
        std::lock_guard<std::mutex> lock(hit_trace_mtx_);
        if (hit_trace_csv_.is_open())
            hit_trace_csv_.close();
        if (hit_trace_csv_enable_ && !hit_trace_csv_path_.empty())
        {
            const auto trace_path = resolve_project_path(hit_trace_csv_path_);
            std::filesystem::create_directories(trace_path.parent_path());
            hit_trace_csv_.open(trace_path, std::ios::out | std::ios::trunc);
            if (hit_trace_csv_.is_open())
            {
                hit_trace_csv_
                    << "controller_t,t_to_hit,swing,waiting,p_hit_x,p_hit_y,p_hit_z,"
                    << "racket_x,racket_y,racket_z,racket_err_x,racket_err_y,racket_err_z,racket_err_norm,"
                    << "ball_x,ball_y,ball_z,ball_now_err_x,ball_now_err_y,ball_now_err_z,ball_now_err_norm,"
                    << "ball_at_hit_x,ball_at_hit_y,ball_at_hit_z,ball_hit_err_x,ball_hit_err_y,ball_hit_err_z,ball_hit_err_norm\n";
                spdlog::info("Pingpong hit trace CSV: {}", trace_path.string());
            }
            else
            {
                spdlog::warn("Failed to open Pingpong hit trace CSV: {}", trace_path.string());
            }
        }
    }

    // Bag record/replay is best-effort and must not block the FSM transition.
    // Blocking here starves lowcmd updates during Velocity -> Pingpong and can
    // trip the MuJoCo/robot lowcmd watchdog before the entry blend starts.
    if (g_use_local_sim_time.load() && ros_bag_replay_enable_)
        sim_time_valid_.store(false);
    start_ros_bag_tools();
    local_sim_time_active_ = g_use_local_sim_time.load() && sim_time_valid_.load();
    start_time_s_ = controller_time_seconds();
    if (g_use_local_sim_time.load() && !local_sim_time_active_)
        spdlog::warn("Pingpong local sim time requested, but no ROS header stamp has arrived at entry; using steady_clock for this entry to avoid blocking lowcmd.");
    else if (local_sim_time_active_)
        spdlog::info("Pingpong local sim time active for this entry: start_time={:.3f}s", start_time_s_);
    policy_thread_running_ = true;
    policy_thread_ = std::thread(&State_Pingpong::policy_loop, this);
}

void State_Pingpong::exit()
{
    policy_thread_running_ = false;
    if (policy_thread_.joinable())
        policy_thread_.join();
    {
        std::lock_guard<std::mutex> lock(hit_trace_mtx_);
        if (hit_trace_csv_.is_open())
            hit_trace_csv_.close();
    }
    stop_ros_bag_tools();
    // CtrlFSM caches state objects and calls enter()/exit() repeatedly. Keep
    // ROS subscribers alive across temporary exits; otherwise a second entry
    // into Pingpong would never receive fresh ball/base topics and the actor
    // would remain in safe-hold forever.
}

void State_Pingpong::run()
{
    std::vector<float> target;
    std::vector<float> actor_blend_start;
    bool active = false;
    bool actor_blend_active = false;
    bool policy_gains_applied = false;
    bool apply_policy_gains = false;
    double actor_blend_start_time = 0.0;
    {
        std::lock_guard<std::mutex> lock(cmd_mtx_);
        target = current_pd_target_;
        active = active_control_;
        actor_blend_active = actor_blend_active_;
        actor_blend_start_time = actor_blend_start_time_s_;
        actor_blend_start = actor_blend_start_target_;
        if (active && !policy_gains_applied_)
        {
            policy_gains_applied_ = true;
            apply_policy_gains = true;
        }
        policy_gains_applied = policy_gains_applied_;
    }
    if (apply_policy_gains)
    {
        apply_policy_gains_to_lowcmd();
        spdlog::info("Pingpong handoff: policy gains applied; HITTER actor is now controlling.");
    }
    if (target.empty())
        target = switch_entry_joint_pos_.empty() ? io_.action_offset : switch_entry_joint_pos_;
    if (actor_blend_start.size() != static_cast<size_t>(io_.action_dim))
        actor_blend_start = switch_entry_joint_pos_.empty() ? io_.action_offset : switch_entry_joint_pos_;

    run_debug_counter_ += 1;
    const double now_s = controller_time_seconds();
    const bool print_run_debug = debug_control_log_enable_ &&
                                 (run_debug_counter_ <= 5 ||
                                  (now_s - start_time_s_ < 2.0 && run_debug_counter_ % 100 == 0));

    const float elapsed = static_cast<float>(std::max(0.0, now_s - start_time_s_));
    float blend = switch_blend_s_ > 1.0e-4f ? clamp_scalar(elapsed / switch_blend_s_, 0.0f, 1.0f) : 1.0f;
    // Smoothstep avoids a torque impulse at both ends of the state transition.
    blend = blend * blend * (3.0f - 2.0f * blend);
    float actor_blend = 1.0f;
    if (active && actor_blend_active && actor_blend_s_ > 1.0e-4f)
    {
        actor_blend = clamp_scalar(static_cast<float>((now_s - actor_blend_start_time) / actor_blend_s_), 0.0f, 1.0f);
        actor_blend = actor_blend * actor_blend * (3.0f - 2.0f * actor_blend);
    }

    for (int i = 0; i < io_.action_dim; ++i)
    {
        const int sdk_id = io_.joint_ids_map[i];
        const float safe = (policy_gains_applied && target.size() == static_cast<size_t>(io_.action_dim))
                               ? target[i]
                               : ((switch_entry_joint_pos_.size() == static_cast<size_t>(io_.action_dim)) ? switch_entry_joint_pos_[i] : io_.action_offset[i]);
        float desired = active ? target[i] : safe;
        if (active && actor_blend_active && actor_blend_s_ > 1.0e-4f)
            desired = actor_blend_start[i] + actor_blend * (target[i] - actor_blend_start[i]);
        const float start = (switch_start_q_.size() == static_cast<size_t>(io_.action_dim)) ? switch_start_q_[i] : desired;
        lowcmd->msg_.motor_cmd()[sdk_id].q() = start + blend * (desired - start);
    }

    if (print_run_debug)
    {
        robot_->update();
        float q_err_linf = 0.0f;
        float dq_linf = 0.0f;
        for (int i = 0; i < io_.action_dim; ++i)
        {
            const int sdk_id = io_.joint_ids_map[i];
            q_err_linf = std::max(q_err_linf, std::abs(lowcmd->msg_.motor_cmd()[sdk_id].q() - robot_->data.joint_pos[i]));
            dq_linf = std::max(dq_linf, std::abs(robot_->data.joint_vel[i]));
        }
        std::ostringstream ss;
        ss << "\n[PINGPONG RUN DEBUG] run=" << run_debug_counter_
           << " t=" << (now_s - start_time_s_)
           << " active=" << active
           << " policy_gains=" << policy_gains_applied
           << " blend=" << blend
           << " heartbeat=" << policy_loop_heartbeat_.load()
           << " policy_t=" << last_policy_loop_time_s_.load()
           << " qerr_linf=" << q_err_linf
           << " dq_linf=" << dq_linf
           << " grav=[" << robot_->data.projected_gravity_b.x() << ","
           << robot_->data.projected_gravity_b.y() << ","
           << robot_->data.projected_gravity_b.z() << "]";
        const int n = std::min<int>(6, io_.action_dim);
        for (int i = 0; i < n; ++i)
        {
            const int sdk_id = io_.joint_ids_map[i];
            const auto &motor = lowcmd->msg_.motor_cmd()[sdk_id];
            ss << "\n  idx=" << i
               << " sdk=" << sdk_id
               << " q=" << robot_->data.joint_pos[i]
               << " dq=" << robot_->data.joint_vel[i]
               << " q_cmd=" << motor.q()
               << " kp=" << motor.kp()
               << " kd=" << motor.kd()
               << " target=" << (i < static_cast<int>(target.size()) ? target[i] : 0.0f);
        }
        spdlog::info("{}", ss.str());
    }
}

void State_Pingpong::policy_loop()
{
    using clock = std::chrono::steady_clock;
    const auto dt = std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(policy_dt_));
    auto sleep_till = clock::now() + dt;
    int debug_counter = 0;
    int actor_step_counter = 0;

    try
    {
        while (policy_thread_running_)
        {
            debug_counter += 1;
            policy_loop_heartbeat_.store(debug_counter);
            const double controller_now_s = controller_time_seconds();
            last_policy_loop_time_s_.store(controller_now_s - start_time_s_);
            robot_->update();
            const double t = controller_now_s - start_time_s_;
            const bool in_switch_blend = t < static_cast<double>(switch_blend_s_);
            std::vector<float> entry_target = in_switch_blend ? switch_entry_joint_pos_ : std::vector<float>{};
            ExternalState state;
            const bool fresh = external_state_fresh(&state);

            if (in_switch_blend)
            {
                {
                    std::lock_guard<std::mutex> lock(cmd_mtx_);
                    current_pd_target_ = switch_entry_joint_pos_;
                    active_control_ = false;
                    actor_output_ready_ = false;
                    actor_blend_active_ = false;
                    actor_blend_start_time_s_ = 0.0;
                    policy_gains_applied_ = !keep_current_gains_during_switch_;
                    last_raw_action_.assign(io_.action_dim, 0.0f);
                    obs_history_.clear();
                }
                if (debug_control_log_enable_ && (debug_counter <= 10 || debug_counter % 5 == 0))
                    debug_log_control_state("switch_entry_npz", t, fresh, entry_target);
                std::this_thread::sleep_until(sleep_till);
                sleep_till += dt;
                continue;
            }

            if (!fresh)
            {
                state = latest_external_state_for_policy();
                if (debug_control_log_enable_ && debug_counter % 25 == 0)
                    spdlog::warn("Pingpong external state stale/missing: holding previous cmd, or seeding the initial cmd if none exists.");
                hold_previous_or_seed_initial_command(t, state, "stale_or_missing_topic");
                if (debug_control_log_enable_ && (debug_counter <= 10 || debug_counter % 25 == 0))
                    debug_log_control_state("planner_waiting_stale_topic", t, fresh, switch_entry_joint_pos_);
            }
            else
            {
                update_command(t, state);
            }

            Command cmd_copy;
            {
                std::lock_guard<std::mutex> lock(cmd_mtx_);
                cmd_copy = cmd_;
            }
            if (debug_control_log_enable_ && !(cmd_copy.active && cmd_copy.planner_valid) && debug_counter % 25 == 0)
                spdlog::warn("Pingpong planner inactive: waiting for the first seeded or live command.");

            auto obs = build_obs(state, cmd_copy);
            auto raw = actor_->act({{"obs", obs}});
            if ((int)raw.size() != io_.action_dim)
                throw std::runtime_error("Pingpong actor action dim mismatch.");

            auto processed = processed_action_from_raw(raw);

            actor_step_counter += 1;

            float raw_linf = 0.0f;
            float pd_linf = 0.0f;
            float pd_step_linf = 0.0f;
            {
                std::lock_guard<std::mutex> lock(cmd_mtx_);
                const auto &prev_target = current_pd_target_;
                for (int i = 0; i < io_.action_dim; ++i)
                {
                    if (prev_target.size() == static_cast<size_t>(io_.action_dim))
                        pd_step_linf = std::max(pd_step_linf, std::abs(processed[i] - prev_target[i]));
                    raw_linf = std::max(raw_linf, std::abs(raw[i]));
                    pd_linf = std::max(pd_linf, std::abs(processed[i] - io_.action_offset[i]));
                }
            }
            if (debug_actor_log_enable_ && (actor_step_counter <= 8 || debug_counter % 25 == 0))
            {
                spdlog::info(
                    "Pingpong actor: step={} active={} valid={} waiting={} t_to_hit={:.3f} raw_linf={:.3f} pd_delta_linf={:.3f} pd_step_linf={:.3f} p_hit=[{:.3f},{:.3f},{:.3f}] v_racket=[{:.3f},{:.3f},{:.3f}]",
                    actor_step_counter,
                    cmd_copy.active, cmd_copy.planner_valid, cmd_copy.waiting_only, cmd_copy.t_to_hit, raw_linf, pd_linf, pd_step_linf,
                    cmd_copy.p_hit_world.x(), cmd_copy.p_hit_world.y(), cmd_copy.p_hit_world.z(),
                    cmd_copy.v_racket_hat_world.x(), cmd_copy.v_racket_hat_world.y(), cmd_copy.v_racket_hat_world.z());
            }
            {
                std::lock_guard<std::mutex> lock(cmd_mtx_);
                if (!actor_output_ready_)
                {
                    actor_blend_start_target_ = current_pd_target_.size() == static_cast<size_t>(io_.action_dim)
                                                    ? current_pd_target_
                                                    : switch_entry_joint_pos_;
                    actor_blend_start_time_s_ = controller_time_seconds();
                    actor_blend_active_ = actor_blend_s_ > 1.0e-4f;
                    actor_output_ready_ = true;
                    if (actor_blend_active_)
                        spdlog::info("Pingpong actor handoff: blending first policy target over {:.2f}s from npz entry target.", actor_blend_s_);
                    else
                        spdlog::info("Pingpong actor handoff: HITTER actor target applied directly after npz entry blend.");
                }
                last_raw_action_ = raw;
                current_pd_target_ = processed;
                // Match the working MuJoCo validation path: the HITTER actor
                // remains in control during waiting/follow-through commands.
                // Planner active/valid describes the command phase, not whether
                // policy targets should be applied.
                active_control_ = true;
            }

            std::this_thread::sleep_until(sleep_till);
            sleep_till += dt;
        }
    }
    catch (const std::exception &e)
    {
        spdlog::error("Pingpong policy_loop exception: {}", e.what());
    }
    catch (...)
    {
        spdlog::error("Pingpong policy_loop unknown exception.");
    }
}

bool State_Pingpong::external_state_fresh(ExternalState *out) const
{
    std::lock_guard<std::mutex> lock(ext_mtx_);
    const auto now = std::chrono::steady_clock::now();
    const bool ball_ok = ext_.has_ball && std::chrono::duration<float>(now - ext_.ball_time).count() <= topic_timeout_s_;
    const bool ball_sample_ok = ext_.has_ball && std::chrono::duration<float>(now - ext_.ball_sample_time).count() <= max_ball_sample_age_s_;
    const bool base_ok = (!require_base_topic_) || (ext_.has_base && std::chrono::duration<float>(now - ext_.base_time).count() <= topic_timeout_s_);
    if (!(ball_ok && ball_sample_ok && base_ok))
        return false;
    *out = ext_;
    if (!out->has_base)
    {
        out->base_pos = reset_root_pos_;
        out->base_quat = robot_->data.root_quat_w.normalized();
    }
    out->blade_normal_world = compute_blade_normal_from_fk(out->base_quat);
    return true;
}

State_Pingpong::ExternalState State_Pingpong::latest_external_state_for_policy() const
{
    ExternalState out;
    {
        std::lock_guard<std::mutex> lock(ext_mtx_);
        out = ext_;
    }

    const auto now = std::chrono::steady_clock::now();
    if (!out.has_ball)
    {
        out.ball_pos = Eigen::Vector3f(x_hit_world_ + 1.0f, 0.0f, table_top_z_ + 0.25f);
        out.ball_vel = Eigen::Vector3f::Zero();
        out.ball_time = now;
        out.ball_sample_time = now;
    }
    if (!out.has_base)
    {
        out.base_pos = reset_root_pos_;
        out.base_quat = robot_->data.root_quat_w.normalized();
        out.base_time = now;
    }
    else
    {
        out.base_quat.normalize();
    }
    out.blade_normal_world = compute_blade_normal_from_fk(out.base_quat);
    return out;
}

void State_Pingpong::update_command(double now_s, const ExternalState &state)
{
    auto policy_t_to_hit = [&](float raw_t_to_hit) -> float {
        if (post_hit_imitation_)
            return std::max(raw_t_to_hit, -post_swing_time_);
        if (raw_t_to_hit <= 0.0f)
            return -post_swing_time_;
        return raw_t_to_hit;
    };

    PlannerResult plan = plan_once(state);
    if (plan.force_waiting)
    {
        if (command_active_)
        {
            Command cmd_for_log;
            {
                std::lock_guard<std::mutex> lock(cmd_mtx_);
                cmd_for_log = cmd_;
            }
            const float raw_local_t_to_hit = static_cast<float>(t_hit_abs_ - now_s);
            maybe_log_hit_window(cmd_for_log, state, raw_local_t_to_hit);
            log_hit_trace_sample(cmd_for_log, state, raw_local_t_to_hit);
        }
        hold_previous_or_seed_initial_command(now_s, state, plan.reject_reason.c_str());
        return;
    }

    if (plan.valid)
    {
        last_command_update_s_ = now_s;
        has_live_planner_cmd_ = true;
        float sample_age = 0.0f;
        if (local_sim_time_active_ && state.ball_stamp_s > 0.0)
        {
            sample_age = clamp_scalar(
                static_cast<float>((start_time_s_ + now_s) - state.ball_stamp_s),
                0.0f,
                max_ball_sample_age_s_);
        }
        else
        {
            sample_age = clamp_scalar(
                std::chrono::duration<float>(std::chrono::steady_clock::now() - state.ball_sample_time).count(),
                0.0f,
                max_ball_sample_age_s_);
        }
        const float corrected_t_to_hit = plan.raw_t_to_hit - sample_age;
        const bool starting_new_command =
            !command_active_ || t_hit_abs_ - now_s <= -static_cast<double>(post_swing_time_);
        if (starting_new_command)
        {
            command_frozen_ = false;
            hit_window_logged_ = false;
            last_waiting_reason_.clear();
        }
        command_active_ = true;
        // Keep timing tied to the newest valid ball observation.  The ROS
        // callback updates the latest ball state at the publisher rate; this
        // 50 Hz policy/planner loop consumes the newest sample and refreshes
        // t_hit_abs_ every valid cycle.  Space may still freeze if explicitly
        // configured, but hit time is never latched.
        t_hit_abs_ = now_s + static_cast<double>(corrected_t_to_hit);
        const float raw_local_t_to_hit = static_cast<float>(t_hit_abs_ - now_s);
        const float local_t_to_hit = policy_t_to_hit(raw_local_t_to_hit);

        // Real deploy mode: external ball tracking is the source of truth.
        // Valid frames refresh both spatial targets and t_to_hit.  If
        // freeze_time_before_hit_ is set, only the spatial command can freeze;
        // timing continues to follow the newest valid planner result.
        const bool should_freeze_space =
            command_active_ && command_frozen_ && freeze_time_before_hit_ > 0.0f;
        if (!should_freeze_space)
        {
            std::lock_guard<std::mutex> lock(cmd_mtx_);
            cmd_ = plan.cmd;
            cmd_.t_to_hit = local_t_to_hit;
            cmd_.active = true;
            cmd_.planner_valid = true;
            cmd_.waiting_only = false;
        }
        else
        {
            std::lock_guard<std::mutex> lock(cmd_mtx_);
            cmd_.t_to_hit = local_t_to_hit;
            cmd_.active = true;
            cmd_.planner_valid = true;
            cmd_.waiting_only = false;
        }
        command_frozen_ = freeze_time_before_hit_ > 0.0f && local_t_to_hit <= freeze_time_before_hit_;
        Command cmd_for_log;
        {
            std::lock_guard<std::mutex> lock(cmd_mtx_);
            cmd_for_log = cmd_;
        }
        maybe_log_hit_window(cmd_for_log, state, raw_local_t_to_hit);
        log_hit_trace_sample(cmd_for_log, state, raw_local_t_to_hit);
    }
    else if (command_active_)
    {
        last_command_update_s_ = now_s;
        // After the ball passes the hit plane, or if the live planner briefly
        // becomes invalid, keep the last spatial command but continue the clock.
        // This is the important anti-twitch behavior: the actor sees the
        // follow-through/recovery phase instead of a stuck t_to_hit=0 command.
        const float raw_local_t_to_hit = static_cast<float>(t_hit_abs_ - now_s);
        const float local_t_to_hit = policy_t_to_hit(raw_local_t_to_hit);
        Command cmd_for_log;
        {
            std::lock_guard<std::mutex> lock(cmd_mtx_);
            cmd_.t_to_hit = local_t_to_hit;
            cmd_.active = true;
            cmd_.planner_valid = true;
            cmd_.waiting_only = false;
            cmd_for_log = cmd_;
        }
        maybe_log_hit_window(cmd_for_log, state, raw_local_t_to_hit);
        log_hit_trace_sample(cmd_for_log, state, raw_local_t_to_hit);
        if (local_t_to_hit <= -post_swing_time_)
        {
            std::lock_guard<std::mutex> lock(cmd_mtx_);
            command_active_ = false;
            command_frozen_ = false;
            cmd_.t_to_hit = -post_swing_time_;
            cmd_.active = true;
            cmd_.planner_valid = true;
            cmd_.waiting_only = true;
            spdlog::info(
                "Pingpong follow-through complete: holding previous cmd, t_to_hit={:.3f}; Pingpong stays on the HITTER actor.",
                cmd_.t_to_hit);
        }
    }
    else
    {
        if (has_live_planner_cmd_)
        {
            hold_previous_or_seed_initial_command(now_s, state, "planner_invalid_after_live_cmd");
            return;
        }

        hold_previous_or_seed_initial_command(now_s, state, "initial_or_no_valid_live_cmd");
        return;
    }
}

void State_Pingpong::hold_previous_or_seed_initial_command(double now_s, const ExternalState &state, const char *reason)
{
    const float dt = last_command_update_s_ > 0.0
                         ? static_cast<float>(std::max(0.0, now_s - last_command_update_s_))
                         : policy_dt_;
    last_command_update_s_ = now_s;

    Command out_cmd;
    bool held_previous = false;
    {
        std::lock_guard<std::mutex> lock(cmd_mtx_);
        const bool has_previous_cmd = cmd_.active && cmd_.planner_valid;
        if (has_previous_cmd && has_live_planner_cmd_)
        {
            // After a real planner command has existed, bad/missing ball frames
            // must not move the spatial target. Keep the last command geometry
            // and expose a completed-swing time until a new valid command arrives.
            cmd_.t_to_hit = -post_swing_time_;
            cmd_.active = true;
            cmd_.planner_valid = true;
            cmd_.waiting_only = true;
            out_cmd = cmd_;
            held_previous = true;
        }
        else
        {
            // First Pingpong command: there is no previous live-ball command to
            // hold, so seed the actor with the forehand npz-based waiting cmd.
            // Keep its spatial target fixed and let time decay to -post_swing.
            float next_t_to_hit = waiting_initial_t_to_hit_;
            if (has_previous_cmd && cmd_.t_to_hit < 0.0f)
                next_t_to_hit = cmd_.t_to_hit - dt;
            out_cmd = has_previous_cmd ? cmd_ : make_waiting_command(state, next_t_to_hit);
            out_cmd.t_to_hit = std::max(next_t_to_hit, -post_swing_time_);
            out_cmd.active = true;
            out_cmd.planner_valid = true;
            out_cmd.waiting_only = true;
            cmd_ = out_cmd;
        }
        command_active_ = false;
        command_frozen_ = false;
    }

    const std::string reason_str = reason ? reason : "unknown";
    const bool reason_changed =
        reason_str != last_waiting_reason_ || held_previous != last_waiting_held_previous_;
    if (reason_changed)
    {
        last_waiting_reason_log_s_ = now_s;
        last_waiting_reason_ = reason_str;
        last_waiting_held_previous_ = held_previous;
        spdlog::warn(
            "Pingpong planner fallback: reason={} -> {}, t_to_hit={:.3f}",
            reason_str,
            held_previous ? "holding previous cmd" : "seeding initial forehand cmd",
            out_cmd.t_to_hit);
    }
}

State_Pingpong::PlannerResult State_Pingpong::plan_once(const ExternalState &state) const
{
    PlannerResult out;
    auto reject_to_waiting = [&](const std::string &reason) {
        out.force_waiting = true;
        out.reject_reason = reason;
        return out;
    };

    if (!state.has_ball)
        return reject_to_waiting("missing_ball_state");
    if (!state.ball_pos.allFinite() || !state.ball_vel.allFinite())
        return reject_to_waiting("nonfinite_ball_state");
    if (state.ball_pos.z() < planner_min_ball_z_world_)
        return reject_to_waiting("ball_below_table");
    if (state.ball_vel.x() >= -planner_min_incoming_speed_x_)
        return reject_to_waiting("ball_not_flying_to_robot");

    Eigen::Vector3f p = state.ball_pos;
    Eigen::Vector3f v = state.ball_vel;
    Eigen::Vector3f prev_p = p;
    Eigen::Vector3f prev_v = v;
    const float center_z = table_top_z_ + ball_radius_;
    const int steps = std::max(1, static_cast<int>(planner_max_time_ / planner_dt_));
    int bounces = 0;

    for (int step = 1; step <= steps; ++step)
    {
        const float speed = v.norm();
        Eigen::Vector3f acc(0.0f, 0.0f, -kGravity);
        acc -= planner_drag_k_ * speed * v;
        Eigen::Vector3f v_next = v + acc * planner_dt_;
        Eigen::Vector3f p_next = p + v_next * planner_dt_;

        const bool on_table_xy =
            std::abs(p_next.x() - table_center_x_) <= table_half_x_ &&
            std::abs(p_next.y() - table_center_y_) <= table_half_y_;
        const bool bounced = p.z() > center_z && p_next.z() <= center_z && v_next.z() < 0.0f && on_table_xy;
        if (bounced)
        {
            p_next.z() = center_z;
            v_next.x() *= planner_bounce_ch_;
            v_next.y() *= planner_bounce_ch_;
            v_next.z() = -v_next.z() * planner_bounce_cv_;
            bounces += 1;
            if (planner_max_table_bounces_before_fallback_ > 0 &&
                bounces >= planner_max_table_bounces_before_fallback_)
            {
                out.table_bounces = bounces;
                return reject_to_waiting("too_many_table_bounces");
            }
        }

        const bool moving_to_robot = prev_v.x() < -planner_min_incoming_speed_x_;
        const bool crosses = prev_p.x() >= x_hit_world_ && p_next.x() <= x_hit_world_;
        if (moving_to_robot && crosses)
        {
            const float denom = std::min(p_next.x() - prev_p.x(), -1.0e-6f);
            const float alpha = clamp_scalar((x_hit_world_ - prev_p.x()) / denom, 0.0f, 1.0f);
            const Eigen::Vector3f p_hit = prev_p + alpha * (p_next - prev_p);
            const Eigen::Vector3f v_hit = prev_v + alpha * (v_next - prev_v);
            const float t_hit = (static_cast<float>(step - 1) + alpha) * planner_dt_;

            const bool z_ok = p_hit.z() >= z_min_world_ && p_hit.z() <= z_max_world_;
            const bool t_ok = t_hit >= planner_min_t_to_hit_ && t_hit <= planner_max_t_to_hit_;
            if (z_ok && t_ok)
            {
                Command cmd;
                cmd.p_hit_world = p_hit;
                cmd.v_ball_in_world = v_hit;
                cmd.target_land_world = target_land_world_;
                solve_racket_target(p_hit, v_hit, &cmd);

                const float yaw = yaw_from_quat(state.base_quat);
                const Eigen::Vector2f diff = p_hit.head<2>() - state.base_pos.head<2>();
                const Eigen::Vector2f hit_base = rotate_yaw_2d(diff, -yaw);
                const bool forehand = (hit_base.y() - y_mid_base_) * swing_y_sign_ > 0.0f;
                cmd.swing_type = forehand ? 0 : 1;
                const Eigen::Vector2f offset = cmd.swing_type == 0 ? forehand_offset_base_ : backhand_offset_base_;
                cmd.p_base_xy_world = p_hit.head<2>() - rotate_yaw_2d(offset, yaw);
                cmd.t_to_hit = t_hit;
                cmd.planner_valid = true;
                cmd.active = true;

                out.cmd = cmd;
                out.raw_t_to_hit = t_hit;
                out.table_bounces = bounces;
                out.valid = true;
                return out;
            }
        }

        prev_p = p;
        prev_v = v;
        p = p_next;
        v = v_next;
    }

    return out;
}

State_Pingpong::Command State_Pingpong::make_fallback_command(const ExternalState *state) const
{
    Command cmd;
    const Eigen::Vector3f base_pos = state != nullptr ? state->base_pos : reset_root_pos_;
    const Eigen::Quaternionf base_q = state != nullptr ? state->base_quat.normalized() : reset_root_quat_.normalized();

    Eigen::Vector3f hit_offset = fallback_ref_.hit_offset_base;
    if (!fallback_ref_.valid || !hit_offset.allFinite())
        hit_offset = Eigen::Vector3f(forehand_offset_base_.x(), forehand_offset_base_.y(), 1.05f - reset_root_pos_.z());

    Eigen::Vector3f racket_vel = fallback_ref_.racket_vel_base;
    if (!fallback_ref_.valid || !racket_vel.allFinite() || racket_vel.norm() < 1.0e-6f)
        racket_vel = Eigen::Vector3f(0.4f, 0.8f, 0.6f);

    Eigen::Vector3f target_normal = fallback_ref_.normal_base;
    if (!fallback_ref_.valid || !target_normal.allFinite() || target_normal.norm() < 1.0e-6f)
        target_normal = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
    target_normal.normalize();

    cmd.p_hit_world = base_pos + base_q * hit_offset;
    cmd.v_ball_in_world = Eigen::Vector3f::Zero();
    cmd.v_ball_out_world = Eigen::Vector3f::Zero();
    cmd.v_racket_hat_world = base_q * racket_vel;
    cmd.n_target_world = (base_q * target_normal).normalized();
    cmd.target_land_world = target_land_world_;
    cmd.p_base_xy_world = base_pos.head<2>();
    cmd.t_to_hit = waiting_initial_t_to_hit_;
    cmd.swing_type = fallback_ref_.valid ? fallback_ref_.swing_type : 0;
    cmd.planner_valid = false;
    cmd.active = false;
    cmd.waiting_only = false;
    return cmd;
}

State_Pingpong::Command State_Pingpong::make_waiting_command(const ExternalState &state, float t_to_hit) const
{
    Command cmd;
    const Eigen::Quaternionf base_q = state.base_quat.normalized();

    Eigen::Vector3f hit_offset = fallback_ref_.hit_offset_base;
    if (!fallback_ref_.valid || !hit_offset.allFinite())
        hit_offset = Eigen::Vector3f(forehand_offset_base_.x(), forehand_offset_base_.y(), 1.05f - reset_root_pos_.z());

    Eigen::Vector3f racket_vel = fallback_ref_.racket_vel_base;
    if (!fallback_ref_.valid || !racket_vel.allFinite() || racket_vel.norm() < 1.0e-6f)
        racket_vel = Eigen::Vector3f(0.4f, 0.8f, 0.6f);

    Eigen::Vector3f target_normal = fallback_ref_.normal_base;
    if (!fallback_ref_.valid || !target_normal.allFinite() || target_normal.norm() < 1.0e-6f)
        target_normal = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
    target_normal.normalize();

    cmd.p_hit_world = state.base_pos + base_q * hit_offset;
    cmd.v_ball_in_world = Eigen::Vector3f::Zero();
    cmd.v_ball_out_world = Eigen::Vector3f::Zero();
    cmd.v_racket_hat_world = base_q * racket_vel;
    cmd.n_target_world = (base_q * target_normal).normalized();
    cmd.target_land_world = target_land_world_;
    cmd.p_base_xy_world = state.base_pos.head<2>();
    const float waiting_t = t_to_hit < 0.0f ? t_to_hit : waiting_initial_t_to_hit_;
    cmd.t_to_hit = std::max(waiting_t, -post_swing_time_);
    cmd.swing_type = fallback_ref_.valid ? fallback_ref_.swing_type : 0;
    cmd.planner_valid = true;
    cmd.active = true;
    cmd.waiting_only = true;
    return cmd;
}

std::vector<float> State_Pingpong::build_obs(const ExternalState &state, const Command &cmd)
{
    std::vector<float> obs;
    for (const auto &term : io_.obs_terms)
    {
        auto current = apply_obs_scale_clip(term, build_obs_term(term.name, state, cmd));
        auto &hist = obs_history_[term.name];
        if (hist.empty())
        {
            for (int i = 0; i < std::max(1, term.history_length); ++i)
                hist.push_back(current);
        }
        else
        {
            hist.push_back(current);
            while ((int)hist.size() > std::max(1, term.history_length))
                hist.pop_front();
        }
        while ((int)hist.size() < std::max(1, term.history_length))
            hist.push_front(current);

        for (const auto &snap : hist)
            obs.insert(obs.end(), snap.begin(), snap.end());
    }
    return obs;
}

std::vector<float> State_Pingpong::build_obs_term(const std::string &name, const ExternalState &state, const Command &cmd) const
{
    if (name == "base_ang_vel")
        return {robot_->data.root_ang_vel_b.x(), robot_->data.root_ang_vel_b.y(), robot_->data.root_ang_vel_b.z()};
    if (name == "projected_gravity")
        return {robot_->data.projected_gravity_b.x(), robot_->data.projected_gravity_b.y(), robot_->data.projected_gravity_b.z()};
    if (name == "base_yaw")
    {
        const float yaw = yaw_from_quat(state.base_quat);
        return {std::cos(yaw), std::sin(yaw)};
    }
    if (name == "base_err")
    {
        const Eigen::Vector2f e = cmd.p_base_xy_world - state.base_pos.head<2>();
        return {e.x(), e.y()};
    }
    if (name == "hit_pos")
    {
        const Eigen::Vector3f v = state.base_quat.conjugate() * (cmd.p_hit_world - state.base_pos);
        return {v.x(), v.y(), v.z()};
    }
    if (name == "racket_vel")
        return {cmd.v_racket_hat_world.x(), cmd.v_racket_hat_world.y(), cmd.v_racket_hat_world.z()};
    if (name == "t_to_hit")
        return {cmd.t_to_hit};
    if (name == "active_face")
    {
        const float sign = 1.0f - 2.0f * static_cast<float>(cmd.swing_type);
        Eigen::Vector3f n = state.blade_normal_world;
        if (n.norm() < 1.0e-6f)
            n = fallback_blade_normal_world_;
        n.normalize();
        const Eigen::Vector3f b = state.base_quat.conjugate() * (sign * n);
        return {b.x(), b.y(), b.z()};
    }
    if (name == "target_normal")
    {
        const Eigen::Vector3f n = state.base_quat.conjugate() * cmd.n_target_world;
        return {n.x(), n.y(), n.z()};
    }
    if (name == "joint_pos")
    {
        std::vector<float> out(io_.action_dim);
        for (int i = 0; i < io_.action_dim; ++i)
            out[i] = robot_->data.joint_pos[i] - io_.action_offset[i];
        return out;
    }
    if (name == "joint_vel")
        return std::vector<float>(robot_->data.joint_vel.data(), robot_->data.joint_vel.data() + robot_->data.joint_vel.size());
    if (name == "last_action")
        return last_raw_action_;

    throw std::runtime_error("Pingpong obs term is not implemented: " + name);
}

std::vector<float> State_Pingpong::apply_obs_scale_clip(const ObsTermCfg &term, std::vector<float> value) const
{
    if (!term.clip.empty())
    {
        if (term.clip.size() == 2)
        {
            for (auto &v : value)
                v = clamp_scalar(v, term.clip[0], term.clip[1]);
        }
    }
    if (!term.scale.empty())
    {
        if (term.scale.size() != value.size())
            throw std::runtime_error("Obs scale dim mismatch for term: " + term.name);
        for (size_t i = 0; i < value.size(); ++i)
            value[i] *= term.scale[i];
    }
    return value;
}

std::vector<float> State_Pingpong::processed_action_from_raw(const std::vector<float> &raw) const
{
    std::vector<float> out(io_.action_dim, 0.0f);
    for (int i = 0; i < io_.action_dim; ++i)
    {
        out[i] = io_.action_offset[i] + raw[i] * io_.action_scale[i];
        if (io_.action_clip.size() == static_cast<size_t>(io_.action_dim) && io_.action_clip[i].size() >= 2)
            out[i] = clamp_scalar(out[i], io_.action_clip[i][0], io_.action_clip[i][1]);
    }
    return out;
}

void State_Pingpong::set_safe_targets_locked()
{
    if (policy_gains_applied_ && current_pd_target_.size() == static_cast<size_t>(io_.action_dim))
        return;
    current_pd_target_ = switch_entry_joint_pos_.empty()
                             ? (entry_joint_pos_.empty() ? io_.action_offset : entry_joint_pos_)
                             : switch_entry_joint_pos_;
}

void State_Pingpong::apply_policy_gains_to_lowcmd()
{
    for (int i = 0; i < io_.action_dim; ++i)
    {
        const int sdk_id = io_.joint_ids_map[i];
        lowcmd->msg_.motor_cmd()[sdk_id].kp() = io_.stiffness[i];
        lowcmd->msg_.motor_cmd()[sdk_id].kd() = io_.damping[i];
        lowcmd->msg_.motor_cmd()[sdk_id].dq() = 0.0f;
        lowcmd->msg_.motor_cmd()[sdk_id].tau() = 0.0f;
    }
}

void State_Pingpong::maybe_log_hit_window(const Command &cmd, const ExternalState &state, float t_to_hit)
{
    if (!planner_hit_log_enable_ || hit_window_logged_ || !cmd.active || !cmd.planner_valid || cmd.waiting_only)
        return;
    if (planner_hit_log_window_s_ <= 0.0f || t_to_hit > 0.0f || std::abs(t_to_hit) > planner_hit_log_window_s_)
        return;

    hit_window_logged_ = true;
    const float dt = t_to_hit;
    const Eigen::Vector3f gravity(0.0f, 0.0f, -kGravity);
    const Eigen::Vector3f ball_at_hit = state.ball_pos + state.ball_vel * dt + 0.5f * gravity * dt * dt;
    const Eigen::Vector3f err = ball_at_hit - cmd.p_hit_world;
    const Eigen::Vector3f err_now = state.ball_pos - cmd.p_hit_world;
    const Eigen::Vector3f racket_pos = compute_blade_position_from_fk(state.base_pos, state.base_quat);
    const Eigen::Vector3f racket_err = racket_pos - cmd.p_hit_world;
    spdlog::info(
        "Pingpong hit-position-error: t_to_hit={:.3f}s swing={} p_hit_world=[{:.3f},{:.3f},{:.3f}] racket_world=[{:.3f},{:.3f},{:.3f}] racket_err_world=racket-p_hit=[{:.3f},{:.3f},{:.3f}] racket_err_norm={:.3f} ball_now_world=[{:.3f},{:.3f},{:.3f}] ball_at_hit_est_world=[{:.3f},{:.3f},{:.3f}] ball_hit_err_world=ball_at_hit-p_hit=[{:.3f},{:.3f},{:.3f}] ball_hit_err_norm={:.3f} ball_now_err_world=[{:.3f},{:.3f},{:.3f}]",
        t_to_hit,
        cmd.swing_type == 0 ? "forehand" : "backhand",
        cmd.p_hit_world.x(), cmd.p_hit_world.y(), cmd.p_hit_world.z(),
        racket_pos.x(), racket_pos.y(), racket_pos.z(),
        racket_err.x(), racket_err.y(), racket_err.z(), racket_err.norm(),
        state.ball_pos.x(), state.ball_pos.y(), state.ball_pos.z(),
        ball_at_hit.x(), ball_at_hit.y(), ball_at_hit.z(),
        err.x(), err.y(), err.z(), err.norm(),
        err_now.x(), err_now.y(), err_now.z());
}

void State_Pingpong::log_hit_trace_sample(const Command &cmd, const ExternalState &state, float t_to_hit)
{
    if (!hit_trace_csv_enable_ || !cmd.active || !cmd.planner_valid)
        return;

    std::lock_guard<std::mutex> lock(hit_trace_mtx_);
    if (!hit_trace_csv_.is_open())
        return;

    const Eigen::Vector3f racket_pos = compute_blade_position_from_fk(state.base_pos, state.base_quat);
    const Eigen::Vector3f racket_err = racket_pos - cmd.p_hit_world;
    const Eigen::Vector3f ball_now_err = state.ball_pos - cmd.p_hit_world;
    const Eigen::Vector3f gravity(0.0f, 0.0f, -kGravity);
    const Eigen::Vector3f ball_at_hit = state.ball_pos + state.ball_vel * t_to_hit + 0.5f * gravity * t_to_hit * t_to_hit;
    const Eigen::Vector3f ball_hit_err = ball_at_hit - cmd.p_hit_world;

    hit_trace_csv_
        << (controller_time_seconds() - start_time_s_) << ","
        << t_to_hit << ","
        << cmd.swing_type << ","
        << (cmd.waiting_only ? 1 : 0) << ","
        << cmd.p_hit_world.x() << "," << cmd.p_hit_world.y() << "," << cmd.p_hit_world.z() << ","
        << racket_pos.x() << "," << racket_pos.y() << "," << racket_pos.z() << ","
        << racket_err.x() << "," << racket_err.y() << "," << racket_err.z() << "," << racket_err.norm() << ","
        << state.ball_pos.x() << "," << state.ball_pos.y() << "," << state.ball_pos.z() << ","
        << ball_now_err.x() << "," << ball_now_err.y() << "," << ball_now_err.z() << "," << ball_now_err.norm() << ","
        << ball_at_hit.x() << "," << ball_at_hit.y() << "," << ball_at_hit.z() << ","
        << ball_hit_err.x() << "," << ball_hit_err.y() << "," << ball_hit_err.z() << "," << ball_hit_err.norm()
        << "\n";
    hit_trace_csv_.flush();
}

void State_Pingpong::debug_log_control_state(
    const char *tag,
    double elapsed_s,
    bool external_fresh,
    const std::vector<float> &reference_target,
    int detail_count)
{
    robot_->update();

    Command cmd_copy;
    std::vector<float> target;
    bool active = false;
    bool actor_ready = false;
    bool policy_gains = false;
    bool cmd_lock_busy = false;
    {
        std::unique_lock<std::mutex> lock(cmd_mtx_, std::try_to_lock);
        if (lock.owns_lock())
        {
            cmd_copy = cmd_;
            target = current_pd_target_;
            active = active_control_;
            actor_ready = actor_output_ready_;
            policy_gains = policy_gains_applied_;
        }
        else
        {
            cmd_lock_busy = true;
        }
    }

    auto linf_vec = [](const std::vector<float> &v) -> float {
        float out = 0.0f;
        for (const float x : v)
            out = std::max(out, std::abs(x));
        return out;
    };
    auto linf_diff = [](const std::vector<float> &a, const std::vector<float> &b) -> float {
        if (a.size() != b.size())
            return -1.0f;
        float out = 0.0f;
        for (size_t i = 0; i < a.size(); ++i)
            out = std::max(out, std::abs(a[i] - b[i]));
        return out;
    };

    std::vector<float> q_meas(io_.action_dim, 0.0f);
    std::vector<float> dq_meas(io_.action_dim, 0.0f);
    std::vector<float> q_cmd(io_.action_dim, 0.0f);
    for (int i = 0; i < io_.action_dim; ++i)
    {
        const int sdk_id = io_.joint_ids_map[i];
        q_meas[i] = robot_->data.joint_pos[i];
        dq_meas[i] = robot_->data.joint_vel[i];
        q_cmd[i] = lowcmd->msg_.motor_cmd()[sdk_id].q();
    }

    const float tilt = std::acos(clamp_scalar(-robot_->data.projected_gravity_b.z(), -1.0f, 1.0f));
    const float q_target_err = linf_diff(target, q_meas);
    const float q_cmd_err = linf_diff(q_cmd, q_meas);
    const float ref_err = linf_diff(reference_target, q_meas);
    const float target_ref_diff = linf_diff(target, reference_target);
    const float ref_start_diff = linf_diff(reference_target, switch_start_q_);

    std::ostringstream ss;
    ss << "\n[PINGPONG CTRL DEBUG] tag=" << tag
       << " t=" << elapsed_s
       << " fresh=" << external_fresh
       << " cmd_mtx_busy=" << cmd_lock_busy
       << " reference_target=" << (!reference_target.empty())
       << " active=" << active
       << " actor_ready=" << actor_ready
       << " policy_gains=" << policy_gains
       << " cmd(active=" << cmd_copy.active
       << ",valid=" << cmd_copy.planner_valid
       << ",waiting=" << cmd_copy.waiting_only
       << ",t_to_hit=" << cmd_copy.t_to_hit << ")"
       << " tilt_deg=" << tilt * 180.0f / static_cast<float>(M_PI)
       << " grav=[" << robot_->data.projected_gravity_b.x() << ","
       << robot_->data.projected_gravity_b.y() << ","
       << robot_->data.projected_gravity_b.z() << "]"
       << " angvel=[" << robot_->data.root_ang_vel_b.x() << ","
       << robot_->data.root_ang_vel_b.y() << ","
       << robot_->data.root_ang_vel_b.z() << "]"
       << " q_target_err=" << q_target_err
       << " q_cmd_err=" << q_cmd_err
       << " ref_err=" << ref_err
       << " target_ref_diff=" << target_ref_diff
       << " ref_start_diff=" << ref_start_diff
       << " dq_linf=" << linf_vec(dq_meas)
       << " heartbeat=" << policy_loop_heartbeat_.load();

    const std::vector<int> important = {0, 1, 2, 3, 4, 7, 8, 11, 12, 15, 16, 19, 20, 5, 6, 17, 18, 21, 22};
    int printed = 0;
    for (const int i : important)
    {
        if (i < 0 || i >= io_.action_dim)
            continue;
        if (detail_count > 0 && printed >= std::max(detail_count, 6) && elapsed_s > 0.12)
            break;
        const int sdk_id = io_.joint_ids_map[i];
        const auto &motor = lowcmd->msg_.motor_cmd()[sdk_id];
        const float tgt = i < static_cast<int>(target.size()) ? target[i] : 0.0f;
        const float ref = i < static_cast<int>(reference_target.size()) ? reference_target[i] : 0.0f;
        ss << "\n  idx=" << i
           << " sdk=" << sdk_id
           << " q=" << q_meas[i]
           << " dq=" << dq_meas[i]
           << " q_cmd=" << q_cmd[i]
           << " tgt=" << tgt
           << " ref=" << ref
           << " kp=" << motor.kp()
           << " kd=" << motor.kd();
        printed += 1;
    }
    spdlog::info("{}", ss.str());
}

float State_Pingpong::joint_pos_by_sdk_id(int sdk_id) const
{
    for (int i = 0; i < io_.action_dim; ++i)
    {
        if (io_.joint_ids_map[i] == sdk_id)
            return robot_->data.joint_pos[i];
    }
    return 0.0f;
}

float State_Pingpong::yaw_from_quat(const Eigen::Quaternionf &q_in) const
{
    const Eigen::Quaternionf q = q_in.normalized();
    return std::atan2(2.0f * (q.w() * q.z() + q.x() * q.y()), 1.0f - 2.0f * (q.y() * q.y() + q.z() * q.z()));
}

Eigen::Vector2f State_Pingpong::rotate_yaw_2d(const Eigen::Vector2f &v, float yaw) const
{
    const float c = std::cos(yaw);
    const float s = std::sin(yaw);
    return Eigen::Vector2f(c * v.x() - s * v.y(), s * v.x() + c * v.y());
}

Eigen::Matrix3f State_Pingpong::rpy_matrix(float roll, float pitch, float yaw) const
{
    const Eigen::AngleAxisf rx(roll, Eigen::Vector3f::UnitX());
    const Eigen::AngleAxisf ry(pitch, Eigen::Vector3f::UnitY());
    const Eigen::AngleAxisf rz(yaw, Eigen::Vector3f::UnitZ());
    return (rz * ry * rx).toRotationMatrix();
}

Eigen::Matrix3f State_Pingpong::axis_angle_matrix(const Eigen::Vector3f &axis, float angle) const
{
    return Eigen::AngleAxisf(angle, axis.normalized()).toRotationMatrix();
}

Eigen::Affine3f State_Pingpong::joint_transform(
    const Eigen::Vector3f &xyz,
    const Eigen::Matrix3f &rpy,
    const Eigen::Vector3f &axis,
    float q) const
{
    Eigen::Affine3f t = Eigen::Affine3f::Identity();
    t.translate(xyz);
    t.rotate(rpy);
    if (axis.squaredNorm() > 1.0e-8f)
        t.rotate(Eigen::AngleAxisf(q, axis.normalized()));
    return t;
}

Eigen::Affine3f State_Pingpong::compute_blade_transform_from_fk(
    const Eigen::Vector3f &base_pos,
    const Eigen::Quaternionf &base_quat) const
{
    // URDF chain:
    // pelvis --waist_yaw--> torso --right shoulder/elbow/wrist--> paddle blade.
    // Joint axes are expressed in each joint frame. URDF origin rpy uses
    // R = Rz(yaw) * Ry(pitch) * Rx(roll). BLADE_NORMAL_LOCAL is local -Y.
    Eigen::Affine3f t = Eigen::Affine3f::Identity();
    t.translate(base_pos);
    t.rotate(base_quat.normalized());

    t = t * joint_transform(
                Eigen::Vector3f(-0.0039635f, 0.0f, 0.044f),
                Eigen::Matrix3f::Identity(),
                Eigen::Vector3f::UnitZ(),
                joint_pos_by_sdk_id(12)); // waist_yaw_joint
    t = t * joint_transform(
                Eigen::Vector3f(0.0039563f, -0.10021f, 0.24778f),
                rpy_matrix(-0.27931f, 5.4949e-05f, 0.00019159f),
                Eigen::Vector3f::UnitY(),
                joint_pos_by_sdk_id(22)); // right_shoulder_pitch_joint
    t = t * joint_transform(
                Eigen::Vector3f(0.0f, -0.038f, -0.013831f),
                rpy_matrix(0.27925f, 0.0f, 0.0f),
                Eigen::Vector3f::UnitX(),
                joint_pos_by_sdk_id(23)); // right_shoulder_roll_joint
    t = t * joint_transform(
                Eigen::Vector3f(0.0f, -0.00624f, -0.1032f),
                Eigen::Matrix3f::Identity(),
                Eigen::Vector3f::UnitZ(),
                joint_pos_by_sdk_id(24)); // right_shoulder_yaw_joint
    t = t * joint_transform(
                Eigen::Vector3f(0.015783f, 0.0f, -0.080518f),
                Eigen::Matrix3f::Identity(),
                Eigen::Vector3f::UnitY(),
                joint_pos_by_sdk_id(25)); // right_elbow_joint
    t = t * joint_transform(
                Eigen::Vector3f(0.100f, -0.00188791f, -0.010f),
                Eigen::Matrix3f::Identity(),
                Eigen::Vector3f::UnitX(),
                joint_pos_by_sdk_id(26)); // right_wrist_roll_joint
    t = t * joint_transform(
                Eigen::Vector3f(0.3105f, 0.0f, 0.0f),
                rpy_matrix(-2.3561944902f, 0.0f, 0.0f),
                Eigen::Vector3f::Zero(),
                0.0f); // right_paddle_blade_fixed_joint
    return t;
}

Eigen::Vector3f State_Pingpong::compute_blade_position_from_fk(
    const Eigen::Vector3f &base_pos,
    const Eigen::Quaternionf &base_quat) const
{
    return compute_blade_transform_from_fk(base_pos, base_quat).translation();
}

Eigen::Vector3f State_Pingpong::compute_blade_normal_from_fk(const Eigen::Quaternionf &base_quat) const
{
    const Eigen::Matrix3f r = compute_blade_transform_from_fk(Eigen::Vector3f::Zero(), base_quat).linear();
    Eigen::Vector3f n = r * Eigen::Vector3f(0.0f, -1.0f, 0.0f);
    if (n.norm() < 1.0e-6f)
        return fallback_blade_normal_world_;
    return n.normalized();
}

Eigen::Vector3f State_Pingpong::solve_racket_target(const Eigen::Vector3f &p_hit, const Eigen::Vector3f &v_in, Command *cmd) const
{
    const float t = std::max(flight_time_, 1.0e-3f);
    Eigen::Vector3f gravity_term(0.0f, 0.0f, 0.5f * kGravity * t);
    const Eigen::Vector3f v_out = (target_land_world_ - p_hit) / t + gravity_term;
    Eigen::Vector3f delta_v = v_out - v_in;
    Eigen::Vector3f n = Eigen::Vector3f(-1.0f, 0.0f, 0.0f);
    if (delta_v.norm() > 1.0e-6f)
        n = delta_v.normalized();
    const float v_in_n = v_in.dot(n);
    const float v_out_n = v_out.dot(n);
    const float v_pad_n = (v_out_n + paddle_cor_ * v_in_n) / (1.0f + paddle_cor_);

    cmd->v_ball_out_world = v_out;
    cmd->n_target_world = n;
    cmd->v_racket_hat_world = v_pad_n * n;
    return cmd->v_racket_hat_world;
}

Eigen::Vector3f State_Pingpong::input_point_to_training(const Eigen::Vector3f &p) const
{
    return input_to_training_quat_ * p + input_origin_in_training_world_;
}

Eigen::Vector3f State_Pingpong::input_vector_to_training(const Eigen::Vector3f &v) const
{
    return input_to_training_quat_ * v;
}

Eigen::Quaternionf State_Pingpong::input_quat_to_training(const Eigen::Quaternionf &q) const
{
    return (input_to_training_quat_ * q).normalized();
}

std::vector<float> State_Pingpong::yaml_float_vector(const YAML::Node &node, const std::string &name)
{
    if (!node || !node.IsSequence())
        throw std::runtime_error("YAML field '" + name + "' is missing or not a sequence.");
    std::vector<float> out;
    out.reserve(node.size());
    for (size_t i = 0; i < node.size(); ++i)
        out.push_back(node[i].as<float>());
    return out;
}

std::vector<int> State_Pingpong::yaml_int_vector_from_numeric(const YAML::Node &node, const std::string &name)
{
    if (!node || !node.IsSequence())
        throw std::runtime_error("YAML field '" + name + "' is missing or not a sequence.");
    std::vector<int> out;
    out.reserve(node.size());
    for (size_t i = 0; i < node.size(); ++i)
        out.push_back(node[i].as<int>());
    return out;
}

std::vector<float> State_Pingpong::remap_full_or_policy(const std::vector<float> &values, const std::vector<int> &ids)
{
    if (values.size() == ids.size())
        return values;
    const int max_id = *std::max_element(ids.begin(), ids.end());
    if (values.size() <= static_cast<size_t>(max_id))
        throw std::runtime_error("Cannot remap vector: not policy-order and not full SDK order.");
    std::vector<float> out(ids.size());
    for (size_t i = 0; i < ids.size(); ++i)
        out[i] = values[ids[i]];
    return out;
}

std::vector<float> State_Pingpong::make_switch_entry_target(const std::vector<float> &start_q) const
{
    if (entry_joint_pos_.size() != static_cast<size_t>(io_.action_dim))
        return start_q;
    if (entry_joint_mode_ == "full")
        return entry_joint_pos_;

    std::vector<float> target = start_q;
    if (entry_joint_mode_ == "hitter_lower_arms" || entry_joint_mode_ == "hitter_lower_waist_arms")
        target = io_.action_offset;
    if (target.size() != static_cast<size_t>(io_.action_dim))
        target.assign(io_.action_dim, 0.0f);

    auto should_use_entry = [this](int sdk_id) -> bool {
        const bool waist = sdk_id == 12;
        const bool left_arm = sdk_id >= 15 && sdk_id <= 19;
        const bool right_arm = sdk_id >= 22 && sdk_id <= 26;
        if (entry_joint_mode_ == "arms_only" || entry_joint_mode_ == "hitter_lower_arms")
            return left_arm || right_arm;
        if (entry_joint_mode_ == "waist_arms" || entry_joint_mode_ == "upper_body" || entry_joint_mode_ == "hitter_lower_waist_arms")
            return waist || left_arm || right_arm;
        return true;
    };

    for (int i = 0; i < io_.action_dim; ++i)
    {
        if (should_use_entry(io_.joint_ids_map[i]))
            target[i] = entry_joint_pos_[i];
    }
    return target;
}

std::vector<float> State_Pingpong::load_joint_pos_frame_from_npz(const std::string &path, int frame)
{
    const RawNpyArray joint_pos = load_raw_npz_array(path, "joint_pos");
    if (joint_pos.shape.size() != 2)
        throw std::runtime_error("joint_pos must have shape [T, J]: " + path);
    const int num_frames = static_cast<int>(joint_pos.shape[0]);
    const int num_joints = static_cast<int>(joint_pos.shape[1]);
    if (num_frames <= 0 || num_joints <= 0)
        throw std::runtime_error("joint_pos is empty: " + path);
    if (frame < 0)
        frame = 0;
    if (frame >= num_frames)
        throw std::runtime_error("entry_motion_frame out of range for joint_pos: " + path);

    std::vector<float> out(num_joints);
    const std::size_t base = static_cast<std::size_t>(frame) * static_cast<std::size_t>(num_joints);
    for (int j = 0; j < num_joints; ++j)
        out[j] = raw_npy_value_as_float(joint_pos, base + static_cast<std::size_t>(j));
    return out;
}

Eigen::Vector2f State_Pingpong::load_impact_offset_from_npz(const std::string &path)
{
    auto load_required = [&](const std::string &key) -> RawNpyArray {
        try
        {
            return load_raw_npz_array(path, key);
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("npz missing/unreadable required field '" + key + "': " + path + " (" + e.what() + ")");
        }
    };

    const RawNpyArray impact_arr = load_required("impact_frame");
    const RawNpyArray body_pos = load_required("body_pos_w");
    const RawNpyArray body_quat = load_required("body_quat_w");
    const int impact_frame = raw_npy_scalar_as_int(impact_arr);
    if (body_pos.shape.size() != 3 || body_pos.shape[2] != 3)
        throw std::runtime_error("body_pos_w must have shape [T, B, 3]: " + path);
    if (body_quat.shape.size() != 3 || body_quat.shape[2] != 4)
        throw std::runtime_error("body_quat_w must have shape [T, B, 4]: " + path);
    const int num_frames = static_cast<int>(body_pos.shape[0]);
    const int num_bodies = static_cast<int>(body_pos.shape[1]);

    int pelvis_id = 0;
    int blade_id = num_bodies - 1;
    try
    {
        const RawNpyArray body_names_arr = load_raw_npz_array(path, "body_names");
        const auto body_names = raw_npy_string_vector(body_names_arr);
        auto find_body = [&](const std::string &name) -> int {
            const auto it = std::find(body_names.begin(), body_names.end(), name);
            if (it == body_names.end())
                throw std::runtime_error("npz body_names missing '" + name + "': " + path);
            return static_cast<int>(std::distance(body_names.begin(), it));
        };
        pelvis_id = find_body("pelvis");
        blade_id = find_body("right_paddle_blade");
    }
    catch (const std::exception &e)
    {
        spdlog::warn(
            "npz body_names was not readable ({}); using fallback body indices pelvis=0, right_paddle_blade=last for {}",
            e.what(), path);
    }
    if (impact_frame < 0 || impact_frame >= num_frames || pelvis_id >= num_bodies || blade_id >= num_bodies)
        throw std::runtime_error("npz impact/body index out of range: " + path);

    auto pos_at = [&](int frame, int body, int k) {
        return raw_npy_value_as_float(body_pos, (static_cast<std::size_t>(frame) * num_bodies + body) * 3 + k);
    };
    auto quat_at = [&](int frame, int body, int k) {
        return raw_npy_value_as_float(body_quat, (static_cast<std::size_t>(frame) * num_bodies + body) * 4 + k);
    };

    const Eigen::Vector2f pelvis_xy(pos_at(impact_frame, pelvis_id, 0), pos_at(impact_frame, pelvis_id, 1));
    const Eigen::Vector2f blade_xy(pos_at(impact_frame, blade_id, 0), pos_at(impact_frame, blade_id, 1));
    const Eigen::Quaternionf pelvis_q(
        quat_at(impact_frame, pelvis_id, 0),
        quat_at(impact_frame, pelvis_id, 1),
        quat_at(impact_frame, pelvis_id, 2),
        quat_at(impact_frame, pelvis_id, 3));
    const float yaw = yaw_from_wxyz_quat(pelvis_q);
    const Eigen::Vector2f diff = blade_xy - pelvis_xy;
    const float c = std::cos(-yaw);
    const float s = std::sin(-yaw);
    return Eigen::Vector2f(c * diff.x() - s * diff.y(), s * diff.x() + c * diff.y());
}

State_Pingpong::FallbackReference State_Pingpong::load_fallback_reference_from_npz(const std::string &path, int frame_override)
{
    auto load_required = [&](const std::string &key) -> RawNpyArray {
        try
        {
            return load_raw_npz_array(path, key);
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("npz missing/unreadable required field '" + key + "': " + path + " (" + e.what() + ")");
        }
    };

    const RawNpyArray impact_arr = load_required("impact_frame");
    const RawNpyArray body_pos = load_required("body_pos_w");
    const RawNpyArray body_quat = load_required("body_quat_w");
    const RawNpyArray body_lin_vel = load_required("body_lin_vel_w");
    if (body_pos.shape.size() != 3 || body_pos.shape[2] != 3)
        throw std::runtime_error("body_pos_w must have shape [T, B, 3]: " + path);
    if (body_quat.shape.size() != 3 || body_quat.shape[2] != 4)
        throw std::runtime_error("body_quat_w must have shape [T, B, 4]: " + path);
    if (body_lin_vel.shape.size() != 3 || body_lin_vel.shape[2] != 3)
        throw std::runtime_error("body_lin_vel_w must have shape [T, B, 3]: " + path);

    const int num_frames = static_cast<int>(body_pos.shape[0]);
    const int num_bodies = static_cast<int>(body_pos.shape[1]);
    if (body_quat.shape[0] != body_pos.shape[0] || body_quat.shape[1] != body_pos.shape[1] ||
        body_lin_vel.shape[0] != body_pos.shape[0] || body_lin_vel.shape[1] != body_pos.shape[1])
        throw std::runtime_error("body_pos_w/body_quat_w/body_lin_vel_w shape mismatch: " + path);

    int pelvis_id = 0;
    int blade_id = num_bodies - 1;
    try
    {
        const RawNpyArray body_names_arr = load_raw_npz_array(path, "body_names");
        const auto body_names = raw_npy_string_vector(body_names_arr);
        auto find_body = [&](const std::string &name) -> int {
            const auto it = std::find(body_names.begin(), body_names.end(), name);
            if (it == body_names.end())
                throw std::runtime_error("npz body_names missing '" + name + "': " + path);
            return static_cast<int>(std::distance(body_names.begin(), it));
        };
        pelvis_id = find_body("pelvis");
        blade_id = find_body("right_paddle_blade");
    }
    catch (const std::exception &e)
    {
        spdlog::warn(
            "npz body_names was not readable for fallback ref ({}); using fallback body indices pelvis=0, right_paddle_blade=last for {}",
            e.what(), path);
    }

    int frame = frame_override >= 0 ? frame_override : raw_npy_scalar_as_int(impact_arr);
    if (frame < 0 || frame >= num_frames || pelvis_id >= num_bodies || blade_id >= num_bodies)
        throw std::runtime_error("npz fallback frame/body index out of range: " + path);

    auto pos_at = [&](int f, int body, int k) {
        return raw_npy_value_as_float(body_pos, (static_cast<std::size_t>(f) * num_bodies + body) * 3 + k);
    };
    auto vel_at = [&](int f, int body, int k) {
        return raw_npy_value_as_float(body_lin_vel, (static_cast<std::size_t>(f) * num_bodies + body) * 3 + k);
    };
    auto quat_at = [&](int f, int body, int k) {
        return raw_npy_value_as_float(body_quat, (static_cast<std::size_t>(f) * num_bodies + body) * 4 + k);
    };

    const Eigen::Vector3f pelvis_p(pos_at(frame, pelvis_id, 0), pos_at(frame, pelvis_id, 1), pos_at(frame, pelvis_id, 2));
    const Eigen::Vector3f blade_p(pos_at(frame, blade_id, 0), pos_at(frame, blade_id, 1), pos_at(frame, blade_id, 2));
    const Eigen::Vector3f blade_v(vel_at(frame, blade_id, 0), vel_at(frame, blade_id, 1), vel_at(frame, blade_id, 2));
    const Eigen::Quaternionf pelvis_q(
        quat_at(frame, pelvis_id, 0),
        quat_at(frame, pelvis_id, 1),
        quat_at(frame, pelvis_id, 2),
        quat_at(frame, pelvis_id, 3));
    const Eigen::Quaternionf blade_q(
        quat_at(frame, blade_id, 0),
        quat_at(frame, blade_id, 1),
        quat_at(frame, blade_id, 2),
        quat_at(frame, blade_id, 3));
    const Eigen::Matrix3f base_R = pelvis_q.normalized().toRotationMatrix();

    FallbackReference ref;
    ref.frame = frame;
    ref.hit_offset_base = base_R.transpose() * (blade_p - pelvis_p);
    ref.racket_vel_base = base_R.transpose() * blade_v;
    ref.normal_base = base_R.transpose() * (blade_q.normalized() * Eigen::Vector3f(0.0f, -1.0f, 0.0f));
    if (ref.normal_base.norm() > 1.0e-6f)
        ref.normal_base.normalize();
    else
        ref.normal_base = Eigen::Vector3f(0.0f, 1.0f, 0.0f);

    try
    {
        const RawNpyArray swing_type = load_raw_npz_array(path, "swing_type");
        ref.swing_type = raw_npy_scalar_as_int(swing_type) == 0 ? 0 : 1;
    }
    catch (const std::exception &)
    {
        ref.swing_type = ref.hit_offset_base.y() < 0.0f ? 0 : 1;
    }
    ref.valid = true;
    return ref;
}

void State_Pingpong::ball_odom_cb(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(ext_mtx_);
    const auto recv_time = std::chrono::steady_clock::now();
    observe_sim_time_stamp(msg->header.stamp);
    ext_.ball_pos = input_point_to_training(Eigen::Vector3f(
        msg->pose.pose.position.x,
        msg->pose.pose.position.y,
        msg->pose.pose.position.z));
    ext_.ball_vel = input_vector_to_training(Eigen::Vector3f(
        msg->twist.twist.linear.x,
        msg->twist.twist.linear.y,
        msg->twist.twist.linear.z));
    ext_.has_ball = true;
    ext_.ball_time = recv_time;
    ext_.ball_sample_time = recv_time;
    ext_.ball_stamp_s = stamp_seconds(msg->header.stamp);
    if (!g_use_local_sim_time.load() && use_ros_header_stamp_ &&
        (msg->header.stamp.sec != 0 || msg->header.stamp.nanosec != 0) && ros2_node_)
    {
        const double age_s = std::max(0.0, (ros2_node_->now() - rclcpp::Time(msg->header.stamp)).seconds());
        ext_.ball_sample_time = recv_time - std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                                std::chrono::duration<double>(age_s));
    }
}

void State_Pingpong::base_pose_cb(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(ext_mtx_);
    observe_sim_time_stamp(msg->header.stamp);
    ext_.base_pos = input_point_to_training(Eigen::Vector3f(
        msg->pose.position.x,
        msg->pose.position.y,
        msg->pose.position.z));
    ext_.base_quat = input_quat_to_training(Eigen::Quaternionf(
                         msg->pose.orientation.w,
                         msg->pose.orientation.x,
                         msg->pose.orientation.y,
                         msg->pose.orientation.z)
                         .normalized());
    ext_.has_base = true;
    ext_.base_time = std::chrono::steady_clock::now();
    ext_.base_stamp_s = stamp_seconds(msg->header.stamp);
}
