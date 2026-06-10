# G1 23DoF Pingpong 部署文档

这个目录把训练好的 HITTER 乒乓策略部署为 Unitree G1 23DoF 机器人的一个 FSM 任务. 控制器 C++ 实现, 加载导出的 ONNX 策略, 通过 Unitree DDS 跟机器人通信, 通过 ROS2 跟 mocap (VRPN) 通信, 跑完整的"等待 → 球轨迹预测 → 击球 → 跟随"循环.

可以同时跟**真机**跑 (网卡 `--network <iface>`) 或跟 **mujoco sim** 跑 (网卡 `--network lo`). 仿真侧的代码在 [`/home/woan/HumanoidProject/unitree_mujoco/simulate_python_pingpong/`](/home/woan/HumanoidProject/unitree_mujoco/simulate_python_pingpong/), 已配成跟真机 VRPN 完全一致的 topic + QoS, C++ 控制器无需改动即可对接.

---

## 1. 整体架构

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       g1_pingpong_ctrl  (C++ binary)                     │
│                                                                          │
│  main.cpp                                                                │
│    ├─ 解析 --network 网卡                                                 │
│    ├─ ChannelFactory::Init(0, network)        (Unitree DDS init)         │
│    ├─ FSMState::lowcmd / lowstate 创建        (机器人通信)                │
│    ├─ CtrlFSM ctor (读 config.yaml [FSM])                                │
│    ├─ ★ fsm->preinstantiate_state(10) ★      (Pingpong 提前实例化       │
│    │                                            → ROS subscribers 提前   │
│    │                                            做 DDS discovery)        │
│    └─ fsm->start() (跑 FSM 主循环, 默认 200Hz)                            │
│                                                                          │
│  CtrlFSM (deploy/include/FSM/CtrlFSM.h)                                  │
│    ├─ State_Passive       (机器人下电姿态)                                │
│    ├─ State_FixStand      (站立, 固定关节角)                              │
│    ├─ State_Velocity      (locomotion 策略)                              │
│    ├─ State_Mimic_*       (Dance / Style 模仿动作)                        │
│    └─ ★ State_Pingpong ★  (本目录核心)                                   │
│         ├─ ROS2 Node: 订阅 /vrpn_mocap/U_Tracker0/pose (球)              │
│         │              订阅 /vrpn_mocap/g1/pose       (机器人 base)      │
│         ├─ BallTrajFilter (31 帧 2 阶 LSQ 多项式重建球速度/加速度)         │
│         ├─ Planner (前向积分弹道 + 击球点求解)                            │
│         ├─ HITTER actor (ONNX, 92 维 obs → 23 维 action @ 50Hz)         │
│         └─ 5 个 CSV trace + 1Hz INFO 诊断输出                             │
└──────────────────────────────────────────────────────────────────────────┘
                            │                   ▲
                            ▼                   │
                   ┌────────────────┐  ┌────────────────────┐
                   │   Unitree DDS  │  │   ROS2 (humble)    │
                   │ rt/lowcmd      │  │ /vrpn_mocap/...    │
                   │ rt/lowstate    │  │ (PoseStamped       │
                   │ rt/wireless... │  │  BEST_EFFORT)      │
                   └────────────────┘  └────────────────────┘
                            │                   ▲
                   ┌────────┴────┐    ┌─────────┴────────┐
                   │ G1 真机     │    │ Mocap (Optitrack │
                   │ 或 mujoco   │    │ /VRPN), 或 sim   │
                   │ sim         │    │ publisher        │
                   └─────────────┘    └──────────────────┘
```

**关键设计要点**:

1. **FSM 由手柄按键驱动** (`L2+Up` → FixStand, `R1+X` → Velocity, `Up` → Pingpong, `Y` → Passive). 切换时 `enter()/exit()` 维护 state 资源.
2. **Pingpong state 在程序启动时就预实例化** ([main.cpp `preinstantiate_state(10)`](main.cpp)), 让 ROS subscribers 提前订阅, 避免第一次进 Pingpong 时头几秒没数据 (DDS discovery delay).
3. **ROS subscribers 跨 state 持久** (设计在 [State_Pingpong.cpp `exit()`](src/State_Pingpong.cpp) — 只关闭 csv, 不停 ROS), 多次进出 Pingpong 不需重新做 DDS handshake.
4. **三线程并发**:
   - **FSM 线程** (200Hz): `State_Pingpong::run()` 写 motor cmd
   - **policy_loop 线程** (50Hz): `policy_loop()` 跑 actor inference + 算 cmd
   - **ROS executor 线程**: `ros2_executor_->spin()` 跑 ball/base callback
5. **Pull-side base 滤波**: ROS callback 只 push 数据进 deque, 控制 tick 拉取时才算滑窗均值, 避免 ROS 高频 callback 浪费算力 (详见 §5.6 base_filter_window).

---

## 2. 目录结构

```
deploy/robots/g1_23dof_pingpong/
├── README.md                       本文件
├── CMakeLists.txt                  C++ 编译配置
├── main.cpp                        程序入口
├── include/
│   ├── State_Pingpong.h            Pingpong FSM state 类声明
│   └── BallTrajFilter.h            31 帧多项式球轨迹滤波器
├── src/
│   ├── State_Pingpong.cpp          Pingpong 主实现 (~3000 行)
│   └── BallTrajFilter.cpp          滤波器实现
├── config/
│   └── config.yaml                 全部运行时参数 (见 §5)
├── build/                          CMake build dir (gitignored)
├── data/                           历史数据
├── scripts/                        辅助脚本 (rosbag wrapper 等)
├── tools/                          C++ 辅助工具
├── bags/                           rosbag 记录目录 (可选)
├── logs/                           运行时 csv 输出 (每次进 Pingpong 重写)
│   ├── ros_ball_trace.csv          每条球 ROS callback 一行 (raw + transform)
│   ├── ros_base_trace.csv          每条 base ROS callback 一行 (raw + transform + filt)
│   ├── motor_trace.csv             每个 50Hz tick 一行 (raw_a + q_des + q_act + dq_act)
│   ├── obs_trace.csv               每个 50Hz tick 一行 (全 92 维 actor 输入)
│   └── hit_error_trace.csv         击球点误差 (planner 命中追踪)
├── logs_real_*/ logs_sim_*/        手动备份的历史 csv (跑 sim/real 切换前 mv)
├── inspect_pose_live.py            实时画 PoseStamped raw vs filt 曲线
├── inspect_bag.py                  rosbag sqlite3 schema 检查器
└── plot_hit_trace.py               离线画 hit_error_trace.csv
```

---

## 3. 编译 + 启动

### 3.1 编译

```bash
cd /home/woan/HumanoidProject/unitree_rl_lab/deploy/robots/g1_23dof_pingpong
mkdir -p build && cd build
cmake .. && cmake --build . -j
# 产物: build/g1_pingpong_ctrl
```

### 3.2 启动 (真机)

```bash
# 终端 1: 确认 mocap 在发数据
source /opt/ros/humble/setup.bash
ros2 topic list | grep vrpn   # 应该看到 /vrpn_mocap/g1/pose 等

# 终端 2: 控制器
cd build
./g1_pingpong_ctrl --network enx6c1ff76cb7d7   # 替换成你的网卡名
```

启动 log 关键行 (按时间顺序):
```
Pingpong state pre-instantiated; ROS mocap subscribers running ahead of first transition.
Pingpong ROS2 Humble subscribers started: ball='/vrpn_mocap/U_Tracker0/pose', base='/vrpn_mocap/g1/pose'
[JOINT MAP + PD CHECK]                                       ← 23 个关节的 idx ↔ sdk 映射
FSM: Start Passive
```

### 3.3 启动 (sim)

```bash
# 终端 1: sim
cd /home/woan/HumanoidProject/unitree_mujoco/simulate_python_pingpong
source /opt/ros/humble/setup.bash
python3 unitree_pingpong_mujoco.py

# 终端 2: 控制器, --network lo (sim 用 loopback)
cd /home/woan/HumanoidProject/unitree_rl_lab/deploy/robots/g1_23dof_pingpong/build
./g1_pingpong_ctrl --network lo
```

### 3.4 操作流程 (手柄)

| 按键 | 状态切换 |
|---|---|
| `L2 + Up` | Passive → FixStand (机器人锁站立) |
| `R1 + X` | FixStand → Velocity (locomotion 策略接管) |
| `Up` | Velocity → Pingpong (HITTER 策略接管) |
| `B` | Pingpong → Velocity (返回行走) |
| `Y` | Pingpong / Velocity → Passive (机器人下电) |
| `Right` | Velocity → Mimic_Dance_102 |
| `Left` | Velocity → Mimic_gangnanm_style |

---

## 4. 控制流程 (Pingpong state 的 lifecycle)

### 4.1 启动 → 进入 Pingpong

```
程序启动 (main.cpp)
   ├─ ChannelFactory init                         (Unitree DDS)
   ├─ wait_for_connection (等机器人连上)
   ├─ CtrlFSM ctor                                (创建 Passive)
   ├─ ★ preinstantiate_state(10) ★               (Pingpong ctor 跑)
   │     ├─ load_config(yaml)                     (读 config.yaml)
   │     ├─ load_training_geometry_from_npz       (读 expert npz)
   │     ├─ load_policy(yaml)                     (加载 ONNX + deploy.yaml)
   │     ├─ start_ros_if_enabled                  (创建 ROS2 node + subscribers)
   │     │     ├─ rclcpp::init
   │     │     ├─ create_subscription<PoseStamped>  ball_topic + base_topic
   │     │     │     QoS: SensorDataQoS().keep_last(50)  ← burst-tolerant
   │     │     ├─ ros2_executor_->spin() (独立线程开始 spin)
   │     │     └─ DDS discovery 开始 (~2-3s 完成)
   │     └─ ext_.has_ball = false, ext_.has_base = false (初始)
   │
   ├─ fsm->start()                                 (跑 Passive::enter)
   │
   ├─ [若干秒后, 用户按手柄切到 FixStand → Velocity → Pingpong]
   │
   └─ Pingpong::enter()
         ├─ 清空 base_pos_window_, base_quat_window_, motor_dq_window_, joint_pos_history_
         ├─ 截断打开 5 个 csv 文件 (logs/*.csv)
         ├─ start_ros_bag_tools (可选 rosbag 录制/回放)
         ├─ start_time_s_ = controller_time_seconds()
         └─ policy_thread_ = std::thread(policy_loop, this)  ← 50Hz inference 线程启动
```

### 4.2 50Hz inference 循环 (`policy_loop()`)

每 20ms 一个 tick:

```
1. robot_->update()                                  (从 LowState 读 IMU + encoder)
2. external_state_fresh(&state)                      (检查 ros 数据 fresh + base mean filter)
   - 内部: lock(ext_mtx_); 取 has_ball/has_base/timestamps
   - 调 compute_base_mean_world_locked() 算 base 滑窗均值
3. if (in_switch_blend) { hold pos, return }         (头 0.1s 入口插值, 见 §4.3)

4. update_command(t, state)
   - 走 plan_once() 用 BallTrajFilter 估算球状态 + 弹道前向积分 + 击球点求解
   - 返回 plan: valid / force_waiting / 空
   - if plan.valid → cmd_ = plan.cmd, has_live_planner_cmd_ = true
   - else → hold_previous_or_seed_initial_command (用 npz frame 0 兜底)

5. obs = build_obs(state, cmd_)                      (92 维)
   - base_ang_vel(3) projected_gravity(3) base_yaw(2) base_err(2)
   - hit_pos(3) racket_vel(3) t_to_hit(1) active_face(3) target_normal(3)
   - joint_pos(23) joint_vel(23) last_action(23)

6. 写 obs_trace.csv 一行                              (诊断)

7. raw_action = actor_->act({{"obs", obs}})          (ONNX 推理, 23 维)
8. processed = action_offset + scale * raw_action    (default_q + 0.25 * raw_action)

9. lock(cmd_mtx_); current_pd_target_ = processed; last_raw_action_ = raw_action;

10. 写 motor_trace.csv 一行                           (诊断)
11. sleep_until 下一个 50Hz tick
```

### 4.3 200Hz motor command 写出 (`run()`, FSM 主循环调用)

每 5ms 一个 tick:

```
1. lock(cmd_mtx_); target = current_pd_target_; (snapshot)
2. 计算 switch_blend (头 0.1s, 训练 reset → npz frame 64 入口姿势)
3. 计算 actor_blend (头 0.25s, npz 入口姿势 → actor 第一帧 target)
4. 对每个关节:
     desired = active ? target[i] : safe[i]
     desired = blend_smoothstep(actor_blend_start, target)  (actor_blend 期间)
     motor.q = blend_smoothstep(switch_start_q, desired)     (switch_blend 期间)
5. ROS bag 录制 (可选)
6. flush motor_cmd → DDS → 机器人
```

### 4.4 ROS callbacks (执行器线程)

```
ball_pose_cb(msg):                                    (~300Hz, VRPN U_Tracker0)
   ├─ raw_xyz, raw_q
   ├─ p_world = input_point_to_training(raw_xyz)     (M frame → W frame)
   ├─ 1Hz INFO print (累计帧数, 频率, 坐标)
   ├─ 写 ros_ball_trace.csv (raw + transform 单帧)
   ├─ if (!ball_input_to_planner_enable_) return     ← 调试开关, 见 §5.7 enable_ball_input
   ├─ ball_filter_.push_sample(t, p_world)            (31 帧多项式滤波器)
   └─ lock(ext_mtx_); ext_.ball_pos = p_world; ext_.has_ball = true

base_pose_cb(msg):                                    (~300Hz, VRPN g1)
   ├─ raw_xyz, raw_q_normed
   ├─ lock(ext_mtx_):
   │     ├─ push 进 base_pos_window_ / base_quat_window_ (sliding window)
   │     ├─ 同时算 mean (给 csv 用, 实际 actor 用 pull-side mean)
   │     └─ ext_.has_base = true; ext_.base_time = now
   ├─ 1Hz INFO print
   └─ 写 ros_base_trace.csv (raw + transform 单帧 + filt 滑窗均值)
```

### 4.5 退出 Pingpong (`exit()`)

```
1. policy_thread_running_ = false; policy_thread_.join()
2. 关闭 5 个 csv
3. stop_ros_bag_tools
4. (注意: 不关闭 ros2_node_ / subscribers, 跨 state 持久, 下次进 Pingpong 不重新订阅)
```

---

## 5. config.yaml 参数详解

[config/config.yaml](config/config.yaml) 整个 FSM 的运行时参数. 所有 state 共用同一个 yaml. 这一节详细解释 `Pingpong:` 块的所有字段 (其它 state 比如 Passive / FixStand / Velocity / Mimic 略, 跟 fusion / locomotion deploy 一致).

### 5.0 顶层 FSM 配置

```yaml
FSM:
  _:
    Passive:    {id: 1}
    FixStand:   {id: 2}
    Velocity:   {id: 3, type: RLBase}
    Mimic_*:    {id: 101/102, type: Mimic}
    Pingpong:   {id: 10, type: Pingpong}              ← 关键 id, main.cpp 提前实例化用
```

### 5.1 Pingpong 顶层

```yaml
Pingpong:
  transitions:
    Velocity: B.on_pressed                            # B 键回到 Velocity
    Passive:  Y.on_pressed                            # Y 键紧急下电

  policy_dir: ../../../logs/rsl_rl/.../<run_id>       # 训练产物目录
                                                       # 必须含 exported/policy.onnx + params/deploy.yaml
  policy_dt: 0.02                                     # actor 推理周期 = 50Hz
```

### 5.2 joint_vel obs source (sim2real gap 关键开关)

```yaml
  joint_vel_obs_source: motor_dq                      # motor_dq | finite_diff
  joint_vel_filter_window: 10                         # motor_dq 模式下的滑窗均值长度
  # joint_vel_finite_diff_steps: 5                    # finite_diff 模式 LSQ 窗口
```

| 选项 | 行为 | 适用 |
|---|---|---|
| `motor_dq` + window=1 | 直接用 `robot_->data.joint_vel` (= LowState `motor_state[].dq()`) 不滤波 | 训练已含 `Unoise(±0.5)` joint_vel 噪声的新策略 |
| `motor_dq` + window=10 | 真机 dq + 10 帧滑窗均值, latency ~100ms, 噪声压 √10≈3.16× | 老策略 (训练时没 noise injection) 的部署补丁 |
| `finite_diff` (steps=5) | LSQ 拟合最近 6 个 q, 取斜率 → q-derived dq, 噪声 ≈ q_quant/dt÷√(N³/12) | 紧急部署补丁, 用 q 重建速度避开 motor encoder dq 噪声 |

**背景**: 真机 motor encoder 的 dq 静止时 std 1-10 rad/s (内部速度估计器 artifact), 训练 sim 是 0. 这是 sim2real gap 的主因. 见 [TROUBLESHOOTING.md 第十七章 D9-D11](../../../source/unitree_rl_lab/unitree_rl_lab/tasks/pingpong/TROUBLESHOOTING.md).

### 5.3 logging — 5 个 CSV trace 开关

```yaml
  logging:
    debug_control: false                              # spdlog 内部 PD/target 详细 debug, 终端刷屏严重
    debug_actor:   false                              # actor obs/raw_action 详细 debug
    hit_window: true                                  # 打印 abs(t_to_hit) ≤ 0.05s 的击球瞬间几何
    hit_window_s: 0.05

    hit_trace_csv:                                    # 击球点 vs 实际 paddle 位置误差 (planner 性能)
      enable: true
      output: <绝对路径>/logs/hit_error_trace.csv

    ros_topic_trace:                                  # ball + base 每条 ROS callback 一行
      enable: true                                    # 列: controller_t, recv_steady_s, stamp_s, sample_age_s,
      ball_output: <abs>/logs/ros_ball_trace.csv      #     raw_xyz, transform_xyz (单帧, 不滤波)
      base_output: <abs>/logs/ros_base_trace.csv      # base csv 多: raw_q, transform_q, filt_xyz, filt_q, yaw_deg, filt_yaw_deg

    motor_trace_csv:                                  # 50Hz tick 一行: raw_a + q_des + q_act + dq_act
      enable: true                                    # 用来诊断"是 actor 输出抖 还是电机跟踪不上"
      output: <abs>/logs/motor_trace.csv

    obs_trace_csv:                                    # 50Hz tick 一行: 全 92 维 actor 输入 (post scale/clip)
      enable: true                                    # 用来 sim/real 对比定位 OOD 维度
      output: <abs>/logs/obs_trace.csv
```

### 5.4 入口插值 (Velocity → Pingpong)

```yaml
  switch_blend_s: 0.1                                 # FSM 切换瞬间, 头 0.1s 从当前 q 平滑过渡到 npz 入口姿势
                                                       # 0.0 = 硬切, 不推荐 (扭矩冲击); 0.15 + 偏长但稳
  actor_blend_s:  0.25                                # 入口姿势 → actor 第一帧 target 的二段平滑

  motor_gains:
    keep_current_during_switch: false                 # false = 切到 HITTER PD 立即生效 (推荐)
                                                       # true  = 头 switch_blend_s 内保持 Velocity PD
    support_override:                                  # 把 HITTER 软 PD 替换成硬 PD (debug 腿不稳用)
      enable: false                                    # 不动 obs/action, 仅改 motor.kp/kd
      sdk_ids: [...]
      kp:      [...]
      kd:      [...]
```

### 5.5 input_frame — Mocap 坐标系到训练世界系的变换

```yaml
  input_frame:
    origin_in_training_world: [1.77, 0.0, 0.815]      # ROS 世界 → 训练世界的平移
                                                       # 真机 z 校准: 让 base FixStand 静止时 raw_z 平均 → 期望 0.74
                                                       #   origin_z = 0.74 - mean(raw_z)
                                                       # 例: raw_z mean=-0.064 → origin_z = 0.804 (取保守 0.815)
    rotation_wxyz_to_training: [1.0, 0.0, 0.0, 0.0]   # 旋转, identity = 不旋转 (mocap 跟训练世界系同向)
```

### 5.6 base_filter — base mocap 滑窗均值

```yaml
  ros:
    base_filter_window: 10                            # base mocap 滑窗均值长度
                                                       # 1   = 不滤波 (= 用 mocap 单帧 raw)
                                                       # 5-10 = 推荐 (yaw jitter raw 0.4° → filt 0.1°)
                                                       # 实现: pull-side, ROS callback 只 push deque,
                                                       #       控制 tick 才算 mean
                                                       #       (节省算力 + sample-count window)
                                                       # quat 用 hemisphere-aligned 均值 (Markley 简化)
```

### 5.7 ros — ROS topic 配置

```yaml
  ros:
    enable: true                                       # 关掉则不订阅 ROS, planner 一直 fallback

    ball_state_topic: /vrpn_mocap/U_Tracker0/pose      # 球的 PoseStamped (VRPN-mocap 命名规则)
    base_pose_topic:  /vrpn_mocap/g1/pose              # 机器人 base 的 PoseStamped
    require_base_topic: true                           # true = 需要 base fresh; false = 用 reset_root_pos 兜底

    enable_ball_input: false                           # ★ 调试开关 ★
                                                       # false: ball 数据进 csv 但不进 BallTrajFilter
                                                       #        → has_ball 永远 false → planner 永远 fallback
                                                       #        → cmd 永远是 npz frame 0 (forehand waiting)
                                                       #        用来单独验证 actor 是否稳定
                                                       # true:  正常运行, 球数据进 planner

    use_header_stamp: true                             # 用 msg.header.stamp 算 sample_age 做延迟补偿
                                                       # cmd.t_to_hit = planner_t_to_hit_from_sample - sample_age
    use_sim_time_for_replay: false                     # rosbag replay 时设 true 让 controller 用 /clock

    bag_record:
      enable: false
      output: <abs>/bags/pingpong_sim_record           # 进 Pingpong 时自动启动 ros2 bag record
    bag_replay:
      enable: false
      input:  <abs>/bags/pingpong_sim_record           # 进 Pingpong 时自动启动 ros2 bag play
      loop:   false
      rate:   1.0
```

### 5.8 safety — 数据 fresh 阈值

```yaml
  safety:
    topic_timeout_s: 0.20                              # ball/base 距上次 callback > 0.2s 视为 stale, planner 走 fallback
    max_ball_sample_age_s: 0.25                        # ball header.stamp 距 controller now() > 0.25s 视为过老
```

### 5.9 world — 训练世界系常量

```yaml
  world:
    reset_root_pos: [-0.138, 0.0, 0.74]                # 机器人 base 训练 reset 位置 (训练 sim 跟这一致)
                                                        # x = -0.138 → 距桌沿 0.40 还有 0.538m, 是 demo 触手长度
    reset_root_quat_wxyz: [1.0, 0.0, 0.0, 0.0]         # identity, 朝 +x (面对桌面)
    table_center: [1.77, 0.0, 0.735]                   # 桌身中心 (z 是 5cm 桌面板的中心)
    table_size:   [2.74, 1.525, 0.05]
    table_top_z:  0.76                                  # 桌面顶
    ball_radius:  0.02
```

### 5.10 planner — 球轨迹预测 + 击球点求解

```yaml
  planner:
    forward_motion_file:  <abs>/.../forward_001_wristfix_rotated.npz     # 正手专家 motion clip
    backward_motion_file: <abs>/.../backward_001_rotated.npz             # 反手专家 motion clip
                                                                          # 这两个 npz 用来取 paddle offset / racket vel / normal,
                                                                          # 帮 planner 把击球点 → 球拍 6D 目标

    entry_motion_file:    <abs>/.../forward_001_wristfix_rotated.npz     # Velocity → Pingpong 切换时的入口姿势
    entry_motion_frame:   64                                              # 用这个 npz 的第 64 帧 (post-impact ready pose)
    entry_joint_mode: waist_arms                                          # full         = 23 关节都跟 npz
                                                                           # waist_arms   = 仅 waist + 双臂 (推荐, 腿保持当前 stand)
                                                                           # arms_only    = 仅双臂
                                                                           # hitter_lower_*  = 腿用 HITTER default + 上身 npz

    fallback_motion_file: <abs>/.../forward_001_wristfix_rotated.npz     # 没球数据时的 waiting cmd 几何来源
    fallback_motion_frame: 0
    waiting_initial_t_to_hit: -0.02                                       # 没球时 cmd 初始 t_to_hit, 之后衰减到 -post_swing_time
    forehand_y_safety_clamp: 0.40                                         # 反手击球时 hit_y 安全限位

    # 这两组在 planner 计算击球点时用作 base-frame paddle offset (从 npz impact 帧反算)
    forehand_offset_base: [0.5496, -0.2879]                               # forehand: base → blade 的 (Δx, Δy)
    backhand_offset_base: [0.5250, 0.0164]
    y_mid_base: null                                                       # null = 自动从 npz 算

    # 击球虚拟平面 + 接受窗口 (paper §V-A)
    x_hit_world: 0.40                                                      # 击球 x 平面 = 桌面近边 (1.77 - 1.37 = 0.4)
    z_min_world: 0.85                                                      # 击球点 z 下限
    z_max_world: 1.25                                                      # 击球点 z 上限
    target_land_world: [2.45, 0.0, 0.78]                                  # 期望对方半场落点 (中心)
    flight_time: 0.45                                                      # 出球后到落点的飞行时间 (秒)
    paddle_cor: 0.85                                                       # 球拍-球碰撞恢复系数

    # 弹道前向积分参数 (paper §IV-A 数值实测)
    planner_dt: 0.01                                                       # 积分步长
    planner_max_time: 1.50                                                 # 最长向前看 1.5s
    planner_drag_k: 0.10257265376884504                                    # 空气阻力系数
    planner_bounce_ch: 0.727005044772834                                   # 桌面反弹水平 COR
    planner_bounce_cv: 0.9018357357260598                                  # 桌面反弹垂直 COR

    planner_min_t_to_hit: 0.0                                              # 击球时间上下限 (太近来不及挥)
    planner_max_t_to_hit: 1.20

    min_incoming_speed_x: 0.05                                             # 球必须朝机器人 (vx ≤ -0.05) 否则 reject
    min_ball_z_world: 0.7                                                  # 球必须 > 0.7m (低于桌面认为不是飞行球)
    max_table_bounces_before_fallback: 4                                   # 弹道预测中 ≥4 次反弹 reject

    freeze_time_before_hit: 0.0                                            # >0 时, 击球前 N 秒空间 cmd 冻结, 只 t_to_hit 衰减
                                                                           # 0 = 持续追新球 (推荐)
    post_swing_time: 0.50                                                  # 击球后 cmd t_to_hit 走到 -0.5 后 hold
    post_hit_imitation: true                                               # true = post-swing 期间走 imitation (跟 npz follow-through)
                                                                           # false = post-swing 立即 jump 到 -post_swing_time
```

### 5.11 PD 增益来源

PD 增益不在 config.yaml 里直接写, 而是从 `policy_dir/params/deploy.yaml` 的 `actions.JointPositionAction.stiffness/damping` 读. 启动 log 的 `[JOINT MAP + PD CHECK]` 会打出每个关节的 kp/kd, 校验.

---

## 6. CSV 诊断工具

5 个 csv 都在 `logs/`, 按 `enter()` 时 truncate 重写 → 每次进 Pingpong 是一个独立 session. 每行 `flush()` 一次保证 Ctrl+C 安全.

### 6.1 ros_ball_trace.csv (~300 行/秒)

| 列 | 说明 |
|---|---|
| controller_t | 进 Pingpong 后秒数 |
| recv_steady_s | C++ steady_clock 时间戳 (epoch) |
| stamp_s | msg.header.stamp 秒 |
| sample_age_s | controller now − header.stamp (延迟) |
| raw_x/y/z | 球的原始 mocap 位置 (input frame) |
| world_x/y/z | transform 后的训练世界坐标 (单帧, **不滤波**) |
| filter_n | BallTrajFilter buffer 长度 |
| filter_bounce_idx | 上次反弹检测到的 buffer 索引, -1 表示没反弹 |

### 6.2 ros_base_trace.csv (~300 行/秒)

| 列组 | 说明 |
|---|---|
| meta (4 列) | controller_t, recv_steady_s, stamp_s, sample_age_s |
| raw_x/y/z, raw_qxyzw | 原始 mocap base pose (input frame) |
| world_x/y/z, world_qxyzw | transform 后单帧瞬时 (不滤波) |
| filt_x/y/z, filt_qxyzw | **滤波后** (size = base_filter_window 的滑窗均值, 喂给 actor 的就是这个) |
| yaw_deg, filt_yaw_deg | 单帧 yaw + 滤波后 yaw (度) |

**对比 raw vs filt 列直接看出滤波平滑了多少**.

### 6.3 motor_trace.csv (~50 行/秒)

| 列组 | dim | 说明 |
|---|---|---|
| controller_t, t_to_hit, active | 3 | meta + cmd 状态 |
| raw_a_0..22 | 23 | actor 输出 raw (在 scale/offset 之前) |
| q_des_0..22 | 23 | 实际写到 motor.q() 的 final 目标 (含 switch_blend / actor_blend) |
| q_act_0..22 | 23 | LowState 读到的实际关节角 |
| dq_act_0..22 | 23 | LowState 读到的关节速度 (= robot_->data.joint_vel) |

**诊断公式**:
- `q_des Δstd 大 + q_act Δstd 小` → actor 输出抖, 电机滤掉了 (问题在 actor / obs)
- `q_des Δstd 小 + q_act Δstd 大` → 策略平滑但电机跟踪不上 (问题在 PD / 通信)

### 6.4 obs_trace.csv (~50 行/秒)

`controller_t, t_to_hit, active, obs_0..91`. 列含义按 `Pingpong obs order` 启动 log:

| obs idx | 名称 | dim | 来源 |
|---|---|---|---|
| 0..2 | base_ang_vel | 3 | **IMU gyroscope** |
| 3..5 | projected_gravity | 3 | **IMU quat 投影** |
| 6..7 | base_yaw (cos, sin) | 2 | mocap base_quat |
| 8..9 | base_err | 2 | cmd.p_base_xy − state.base_xy (mocap) |
| 10..12 | hit_pos | 3 | base frame, R^T·(cmd.p_hit − state.base_pos) |
| 13..15 | racket_vel | 3 | base frame, R^T·cmd.v_racket_hat |
| 16 | t_to_hit | 1 | cmd.t_to_hit (秒) |
| 17..19 | active_face | 3 | base frame paddle 法向 (FK from base_quat + joint_pos) |
| 20..22 | target_normal | 3 | base frame, R^T·cmd.n_target |
| 23..45 | joint_pos | 23 | encoder − action_offset (relative-to-default) |
| 46..68 | joint_vel | 23 | motor.dq + filter (或 finite-diff, 见 §5.2) |
| 69..91 | last_action | 23 | 上一帧 actor raw_action |

**离线 sim/real 对比每维 std 直接定位 OOD 维度**.

### 6.5 hit_error_trace.csv

planner 命中追踪 (击球点 vs 实际 blade 位置). 仅在 `cmd.active && cmd.planner_valid` 时写. cmd 冻结模式下 (`enable_ball_input: false`) 通常空.

### 6.6 实时可视化

```bash
# 实时 raw vs filt mocap 曲线
python3 inspect_pose_live.py \
    --topic /vrpn_mocap/g1/pose \
    --topic /vrpn_mocap/U_Tracker0/pose \
    --filter-window 5

# 离线 hit_trace
python3 plot_hit_trace.py logs/hit_error_trace.csv
```

---

## 7. 常见调试任务

### 7.1 校准 mocap z origin

机器人 FixStand 站立 ~5 秒 → 进 Pingpong 跑 1-2 秒 → Ctrl+C → 用 csv 算:

```bash
awk -F, 'NR>1 {n++; s+=$7} END{printf "raw_z mean = %.4f\n  recommended origin_z = %.3f\n", s/n, 0.74-s/n}' \
    logs/ros_base_trace.csv
```

写回 [config.yaml `origin_in_training_world[2]`](config/config.yaml).

### 7.2 单独验证 actor 是否稳定 (cmd 冻结模式)

```yaml
ros:
  enable_ball_input: false   # ← 关掉球进 planner, cmd 永远是 npz frame 0
```

进 Pingpong 后机器人**站着不动**, 不应该腿/腕抖. 抖了说明 actor 自己有问题, 不是球数据.

### 7.3 切换 joint_vel obs 来源

| 期望行为 | yaml |
|---|---|
| 真机 motor.dq + 滑窗均值 (老 actor 的部署补丁) | `joint_vel_obs_source: motor_dq, joint_vel_filter_window: 10` |
| q 的差分速度 (q-derived dq, 避开 motor encoder 噪声) | `joint_vel_obs_source: finite_diff, joint_vel_finite_diff_steps: 5` |
| 真机原始 motor.dq, 不滤波 (新 actor 训练时已加 noise injection) | `joint_vel_obs_source: motor_dq, joint_vel_filter_window: 1` |

### 7.4 监测 ROS topic 真实频率 (不受 C++ 接收影响)

```bash
source /opt/ros/humble/setup.bash
ros2 topic info -v /vrpn_mocap/g1/pose       # 看 publisher QoS
# 标准 ros2 topic hz 不支持 BEST_EFFORT, 用下面 python 脚本:
python3 -c "
import rclpy, time
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
rclpy.init(); n=Node('hz'); c=[0]
def cb(_): c[0]+=1
n.create_subscription(PoseStamped,'/vrpn_mocap/g1/pose',cb,qos_profile_sensor_data)
t=time.time()
while time.time()-t<5: rclpy.spin_once(n,timeout_sec=0.05)
print(f'avg rate ≈ {c[0]/5:.1f} Hz')
"
```

期望 ~300Hz. 如果 C++ csv 频率 << ROS 频率, 是 C++ 端 DDS QoS 问题 (keep_last 太小). 已经修过, 见 [TROUBLESHOOTING.md D2](../../../source/unitree_rl_lab/unitree_rl_lab/tasks/pingpong/TROUBLESHOOTING.md).

### 7.5 安全: 紧急停止

任何时候按手柄 `Y` → Passive (机器人下电瘫倒). 控制器进程仍在跑, 可以再按 `L2+Up` 重新进 FixStand. 终止程序: Ctrl+C (`SIGINT`), 第二次 Ctrl+C 强制退出.

### 7.6 sim/real 切换时备份 csv

```bash
DIR=/home/woan/HumanoidProject/unitree_rl_lab/deploy/robots/g1_23dof_pingpong
TS=$(date +%Y-%m-%d_%H%M)
cp -r $DIR/logs $DIR/logs_real_$TS    # 切到 sim 之前
# 然后跑 sim, sim 跑完同样备份 logs_sim_$TS
```

---

## 8. 跟训练 / 仿真的对应关系

| 训练侧 (Isaac Lab) | 部署侧 |
|---|---|
| `tasks/pingpong/.../hitter/hitter_env_cfg.py PolicyCfg` | C++ `build_obs_term`, deploy.yaml `observations.policy.terms` |
| `mdp.JointPositionActionCfg.scale = UNITREE_G1_23DOF_PADDLE_MIMIC_ACTION_SCALE` | deploy.yaml `actions.JointPositionAction.scale`, C++ `processed_action_from_raw` |
| `mdp.PingpongCommandCfg` (训练时随机生成 cmd) | deploy 端 `plan_once` (从真实球轨迹算 cmd) + `make_waiting_command` (兜底) |
| `EventCfg.randomize_imu_offset / comm_delay` | 真机自然存在, 不需复现 |
| `EventCfg.PolicyCfg noise=Unoise(...)` (新加, 见 TROUBLESHOOTING D10) | 训练后 actor 鲁棒于真机 sensor 噪声, deploy 端可关 filter |

部署 yaml 加载顺序: `config.yaml [Pingpong]` → `policy_dir/params/deploy.yaml` (用来读 obs_dim, action_offset/scale, kp/kd, joint_ids_map). 两者**互补不重复**.

---

## 9. 相关文档

- [TROUBLESHOOTING.md](../../../source/unitree_rl_lab/unitree_rl_lab/tasks/pingpong/TROUBLESHOOTING.md) — 训练 + 部署遇到的所有问题汇总, 第十七章是 sim2real 部署诊断
- [BallTrajFilter.h](include/BallTrajFilter.h) — 31 帧多项式球轨迹滤波器算法注释 (paper §IV-A)
- [State_Pingpong.h / .cpp](src/State_Pingpong.cpp) — 主实现, 所有逻辑都在这里
- [main.cpp](main.cpp) — 程序入口
- [HITTER paper](https://arxiv.org/abs/2508.21043) — 训练算法原文

---

## 10. 维护提示

- **每次改 config.yaml 不需重新编译**, 重启控制器即可.
- **每次改 .h / .cpp 必须 `cmake --build . -j` 重新编译**.
- **改 sim 端 (unitree_mujoco/...) 不影响真机部署**, 反之亦然. 但 `INPUT_ORIGIN_IN_TRAINING_WORLD` 必须两侧一致, 否则 sim 输出的 raw 跟 deploy transform 对不上.
- **每个 csv 文件每次进 Pingpong 都被 truncate**, 想保留对比数据先 `mv logs logs_<ts>` 备份.
- **多次进出 Pingpong 不会重新做 DDS discovery** (subscribers 持久), 但**每次** truncate csv + 重新打开 + 清空滤波 deque.
