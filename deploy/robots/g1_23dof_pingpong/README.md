# G1 23DoF Pingpong C++ Deploy

这个目录把乒乓球策略作为一个 Unitree FSM 任务部署：

- `Passive`
- `FixStand`
- `Velocity`
- `Mimic_*`
- `Pingpong`

`Pingpong` 负责加载 HITTER 的 `exported/policy.onnx`、构造和训练一致的 obs、运行 planner、做 Velocity 到 Pingpong 的 npz 入口插值。入口插值结束后始终由 HITTER 策略接管；MuJoCo 侧只负责物理仿真、DDS lowstate/lowcmd 桥接、键盘模拟手柄、发布球和 pelvis/base 的 ROS2 topic。

## 坐标和几何

controller 内部统一使用 IsaacLab/training world `W`：

- `+x`: 机器人侧指向对方半桌。
- `+y`: 沿球网方向。
- `+z`: 向上。
- 球桌厚度中心: `(1.77, 0.0, 0.735)`。
- 桌面上表面: `z = 0.76`。
- 球网平面: `x = 1.77`。
- 近边缘: `x = 1.77 - 2.74 / 2 = 0.40`。

当前 sim2sim 几何采用：

```text
机器人 pelvis/root      近边缘 = 虚拟击球平面                  球桌中心
  x = -0.138              x = 0.40                            x = 1.77
     |------ 0.538 m ------>|=========== table ===============|
```

所以 `planner.x_hit_world` 是世界坐标，不是距离：

```yaml
world:
  reset_root_pos: [-0.138, 0.0, 0.74]
planner:
  x_hit_world: 0.40
```

这表示机器人 pelvis/root 离球桌近边缘约 `0.538 m`，虚拟击球平面正好在桌边。这样 planner 反推 base target 时，零前后位移的目标仍在 reset 附近，pre-strike 位置也不会扎进桌面。

## 输入坐标系

`config/config.yaml` 的 `input_frame` 定义 ROS topic 输入坐标 `M` 到训练世界 `W` 的转换：

```text
p_W = R_WM * p_M + origin_in_training_world
v_W = R_WM * v_M
q_W = q_WM * q_M
```

当前 sim2sim 使用的 `M` 原点是球桌 5 cm 桌板的厚度中心，轴向和 `W` 对齐：

```yaml
input_frame:
  origin_in_training_world: [1.77, 0.0, 0.735]
  rotation_wxyz_to_training: [1.0, 0.0, 0.0, 0.0]
```

因此 MuJoCo 发布：

```text
ball/base position in M = position in W - (1.77, 0.0, 0.735)
velocity in M = velocity in W
quat in M = quat in W
```

如果真实动捕原点是桌面上表面中心，而不是 5 cm 桌板厚度中心，只改：

```yaml
input_frame:
  origin_in_training_world: [1.77, 0.0, 0.76]
```

如果真实动捕轴向和训练世界不一致，再改 `rotation_wxyz_to_training`。仅平移不一致时不要改旋转。

## ROS2 Topic

controller 只订阅 ROS2 Humble topic：

- `/pingpong/ball_state`: `nav_msgs/Odometry`
  - `pose.pose.position`: 球心位置，坐标系为 `input_frame`。
  - `twist.twist.linear`: 球心速度，坐标系为 `input_frame`。
  - `header.stamp`: 绝对时间戳。`ros.use_header_stamp=true` 时，controller 会用它扣除采样延迟。
- `/pingpong/base_pose`: `geometry_msgs/PoseStamped`
  - `pose.position`: pelvis/root base 位置，坐标系为 `input_frame`。
  - `pose.orientation`: pelvis/root base 四元数，ROS 字段顺序 `xyzw`，controller 内部转成 `wxyz`。

controller 不订阅拍面法向量。右手拍面法向量由 pelvis/root quat 和 Unitree 关节编码器通过 FK 计算。

仿真器还会发布标准 ROS2 `/clock`，时间值等于 MuJoCo `data.time`。controller 在 `--network lo/lo0` 时用 topic stamp 驱动本地仿真命令时间；在 `--network enp...` 这种真机模式下，控制循环和 `t_to_hit` 递减仍然使用本机 steady clock。真机模式只有在 `ros.use_sim_time_for_replay=true` 时才让 ROS2 node 使用 `/clock` 做 bag replay 的 header 延迟修正。

## 录制和回放虚拟球

仿真下需要录的关键 ROS 信息就是三类：

- `/pingpong/ball_state`: 小球位置、速度和该样本的 header stamp。
- `/pingpong/base_pose`: pelvis/root base 的位置和姿态。
- `/clock`: MuJoCo 仿真时间。

方式一：手动录制：

```bash
cd /home/woan/HumanoidProject/unitree_rl_lab
source /opt/ros/humble/setup.bash
deploy/robots/g1_23dof_pingpong/tools/record_pingpong_ros2_bag.sh
```

也可以指定输出目录：

```bash
deploy/robots/g1_23dof_pingpong/tools/record_pingpong_ros2_bag.sh /tmp/pingpong_virtual_ball_001
```

真机无动捕、只做虚拟击球验证时，可以回放这个 bag 给 controller。先把 `config/config.yaml` 里的开关改成：

```yaml
FSM:
  Pingpong:
    ros:
      use_header_stamp: true
      use_sim_time_for_replay: true
```

然后启动真机 controller，`--network` 用真实网卡，不要用 `lo`：

```bash
cd /home/woan/HumanoidProject/unitree_rl_lab/deploy/robots/g1_23dof_pingpong/build
source /opt/ros/humble/setup.bash
./g1_pingpong_ctrl --network enp129s0
```

另一个终端回放：

```bash
cd /home/woan/HumanoidProject/unitree_rl_lab
source /opt/ros/humble/setup.bash
deploy/robots/g1_23dof_pingpong/tools/replay_pingpong_ros2_bag.sh /tmp/pingpong_virtual_ball_001
```

不要给这个回放命令额外加 `ros2 bag play --clock`，因为 bag 里已经录了仿真器发布的 `/clock`。如果不想使用 `/clock`，另一种办法是把 `use_header_stamp` 设成 `false`，让 controller 按接收时刻处理 replay 消息；但这不适合真实低延迟 tracking，只适合离线虚拟球测试。

方式二：让 C++ controller 自动录制/回放。自动录制和自动回放都在 FSM 进入 `Pingpong` 时启动，离开 `Pingpong` 时停止。controller 会用 `source /opt/ros/humble/setup.bash && ros2 bag ...` 启动子进程，并把 rosbag 输出写到日志文件。

仿真时自动录制：

```yaml
FSM:
  Pingpong:
    ros:
      bag_record:
        enable: true
        output: /home/woan/HumanoidProject/unitree_rl_lab/deploy/robots/g1_23dof_pingpong/bags/pingpong_sim_record
      bag_replay:
        enable: false
```

录制成功时终端会打印：

```text
Started ros2 bag record pid=... output='.../pingpong_sim_record' log='.../pingpong_sim_record.record.log'
```

`output` 是最终 bag 目录，里面应包含 `metadata.yaml` 和 rosbag 数据文件。如果 `output` 已存在，controller 会自动追加时间戳后缀，避免 `ros2 bag record` 因目录已存在而失败。如果录制子进程立刻退出，终端会提示查看 `.record.log`，优先看这个日志定位 ROS2/bag 插件问题。

结束自动录制的方法：

- 在 `Pingpong` 状态按 `B`：切回 `Velocity`，自动停止录制并 flush bag。
- 在 `Pingpong` 状态按 `Y`：切到 `Passive`，也会自动停止录制。
- 仿真键盘映射里通常是键盘 `b` 对应手柄 `B`，键盘 `y` 对应手柄 `Y`。

真机无动捕时自动回放：

```yaml
FSM:
  Pingpong:
    ros:
      use_header_stamp: true
      use_sim_time_for_replay: true
      bag_record:
        enable: false
      bag_replay:
        enable: true
        input: /home/woan/HumanoidProject/unitree_rl_lab/deploy/robots/g1_23dof_pingpong/bags/pingpong_sim_record
        loop: false
        rate: 1.0
```

`bag_replay.enable=true` 时，controller 会自动打开 ROS2 `use_sim_time`，并回放 bag 里的 `/clock`、`/pingpong/ball_state`、`/pingpong/base_pose`。`loop=false` 表示播完一次就停，`loop=true` 表示循环播放。`rate=1.0` 表示按录制速度播放，`0.5` 是半速，`2.0` 是两倍速。正式复现仿真来球建议保持 `loop: false, rate: 1.0`。

普通真机动捕/真实球 tracking：

```yaml
FSM:
  Pingpong:
    ros:
      use_header_stamp: true
      use_sim_time_for_replay: false
      bag_record:
        enable: false
      bag_replay:
        enable: false
```

这个流程只能验证“真机能否按虚拟球 cmd 做出策略动作”。它不会验证真实动捕、真实小球接触和真实落点。尤其 `/pingpong/base_pose` 是录下来的仿真 pelvis pose；如果真机实际站位、朝向或漂移和 bag 不一致，planner 的 base error/hit_pos 会有偏差。做安全测试时先低刚度/空场地/远离真实球桌确认动作幅度。

## 关闭 Controller

推荐关闭顺序：

1. 如果当前在 `Pingpong`，先按 `B` 回到 `Velocity`，让自动录制/回放正常停止。
2. 再按 `Ctrl+C` 关闭 `g1_pingpong_ctrl`。
3. 如果第一次 `Ctrl+C` 后卡住，第二次 `Ctrl+C` 会强制退出当前 controller 进程。

自动录制/回放子进程停止时，controller 会依次发送 `SIGINT -> SIGTERM -> SIGKILL`，不会无限等待 `ros2 bag record/play`。正常停止时终端会看到类似：

```text
Stopped ros2 bag record pid=... with SIGINT
```

如果意外还有残留 rosbag 进程，可以检查并手动清理：

```bash
pgrep -af "ros2 bag"
pkill -INT -f "ros2 bag"
```

## config.yaml 参数说明

下面只解释 `deploy/robots/g1_23dof_pingpong/config/config.yaml`，不是 Python sim 的 `config.py`。

| 字段 | 含义 | 常用改法 |
| --- | --- | --- |
| `FSM._.*.id` | 每个 FSM state 的 id。 | 不改。 |
| `FSM.Passive.transitions` | 从 `Passive` 进入 `FixStand` 的按键。 | 一般不改。 |
| `FSM.Passive.mode/kd` | 被动状态的电机模式和阻尼。 | 不改。 |
| `FSM.FixStand.transitions` | `B/Y/X` 等按键对应的切换。 | 一般不改。 |
| `FSM.FixStand.kp/kd/qs` | 起立/固定站立目标和 PD。 | 只有站姿明显不对时改。 |
| `FSM.Velocity.policy_dir` | 速度策略目录，供 `Velocity` state 使用。 | 换 velocity 策略时改。 |
| `FSM.Velocity.transitions.Pingpong` | 进入 `Pingpong` 的按键，当前是 `up.on_pressed`。 | 需要换按键时改。 |
| `FSM.Velocity.bad_orientation_grace_s` | 刚切回 Velocity 后，延迟多久再启用 bad-orientation 自动切 Passive。 | 当前 `0.50s`，避免 Pingpong -> Velocity 交接瞬间误触发 Passive；真摔倒仍会在宽限后进 Passive。 |
| `FSM.Mimic_*.switch_blend_s` | Velocity 切 Mimic 时插值到 npz 姿态的时间。 | 切换过硬就加大。 |
| `FSM.Mimic_*.motion_file` | mimic 参考 npz。 | 换动作时改。 |
| `FSM.Pingpong.policy_dir` | 乒乓策略目录，需要包含 `params/deploy.yaml` 和 `exported/policy.onnx`。 | 换 HITTER 策略时改。 |
| `FSM.Pingpong.policy_dt` | 乒乓策略/planner cmd 运行周期，当前 `0.02s`，即 50Hz。ROS 小球 topic 可以更高频发布；controller 只在 50Hz tick 取最新一帧生成 cmd。 | 必须和导出策略训练频率匹配。 |
| `FSM.Pingpong.switch_blend_s` | Velocity 到 Pingpong 时，从当前关节插值到 `entry_motion_file/frame` npz pose 的时间。 | 像 RoboJuDo/Mimic 一样用于切入；当前配置是 `0.15s`。 |
| `FSM.Pingpong.actor_blend_s` | npz 入口结束后，是否额外平滑第一帧 actor target。 | 当前配置是 `0.25s`。 |
| `FSM.Pingpong.motor_gains.keep_current_during_switch` | `switch_blend_s` 入口插值期间是否保持上一 FSM 的低层刚度。 | 当前为 `false`：Velocity -> Pingpong 插值期间也立刻使用 HITTER deploy.yaml 里的训练/部署 kp、kd。 |
| `FSM.Pingpong.motor_gains.support_override.enable` | 是否覆盖支撑关节 PD gain。只改 low-level PD，不改策略 obs/action。 | 当前为 `false`；若 HITTER 接管后腿/腰仍发软，再打开。 |
| `FSM.Pingpong.motor_gains.support_override.sdk_ids/kp/kd` | 要覆盖的 SDK 关节 id 及对应 `kp/kd`。当前列出双腿和 `waist_yaw`，手臂仍用 HITTER gains。 | 仅在 `support_override.enable=true` 时生效。 |
| `FSM.Pingpong.input_frame.origin_in_training_world` | 输入 topic 坐标原点在训练世界的位置。 | 动捕原点变了就改这里。 |
| `FSM.Pingpong.input_frame.rotation_wxyz_to_training` | 输入 topic 坐标到训练世界的旋转，内部 `wxyz`。 | 动捕轴向和训练轴向不一致时改。 |
| `FSM.Pingpong.ros.enable` | 是否启用 ROS2 topic。 | 真机/真实球 tracking 要开。 |
| `FSM.Pingpong.ros.ball_state_topic` | 球状态 topic 名。 | 和外部发布器保持一致。 |
| `FSM.Pingpong.ros.base_pose_topic` | pelvis/root pose topic 名。 | 和外部发布器保持一致。 |
| `FSM.Pingpong.ros.require_base_topic` | base topic 是否必须新鲜。若不新鲜，planner 不生成新 live cmd；已有 cmd 时保持上一条 cmd。 | 真机建议 `true`。 |
| `FSM.Pingpong.ros.use_header_stamp` | 是否用球状态绝对时间戳修正 `t_to_hit`。 | 多机同步后建议 `true`。 |
| `FSM.Pingpong.ros.use_sim_time_for_replay` | 是否让 ROS2 node 使用 `/clock`，用于真机回放仿真 bag。 | 普通真机 tracking 保持 `false`；虚拟球 bag 回放设为 `true`。 |
| `FSM.Pingpong.ros.bag_record.enable/output` | 进入 Pingpong 后自动调用 `ros2 bag record` 录制 `/clock`、ball、base；结束 Pingpong 时停止并 flush。 | 仿真采样虚拟球时打开；看终端 `output=` 和旁边 `.record.log`。 |
| `FSM.Pingpong.ros.bag_replay.enable/input/loop/rate` | 进入 Pingpong 后自动调用 `ros2 bag play` 回放指定 bag，固定回放 `/clock`、ball、base；`loop` 循环播放，`rate` 调播放速度。 | 真机无动捕虚拟击球时打开；正式复现建议 `loop=false, rate=1.0`。 |
| `FSM.Pingpong.safety.topic_timeout_s` | ball/base topic 超时阈值。 | tracking 频率低时可略增。 |
| `FSM.Pingpong.safety.max_ball_sample_age_s` | 绝对时间戳修正后的最大球样本年龄。 | 延迟大时可略增，但会降低命中可靠性。 |
| `FSM.Pingpong.world.reset_root_pos` | fallback/nominal pelvis/root 世界位置。当前 `[-0.138,0,0.74]`。 | 要改变机器人离桌距离时改。 |
| `FSM.Pingpong.world.reset_root_quat_wxyz` | fallback/nominal base 朝向。 | 通常 `[1,0,0,0]`。 |
| `FSM.Pingpong.world.table_center` | 训练世界里的桌板厚度中心。 | 必须和训练/仿真一致。 |
| `FSM.Pingpong.world.table_size` | 球桌长宽厚。 | 不改。 |
| `FSM.Pingpong.world.table_top_z` | 桌面上表面高度。 | 不改。 |
| `FSM.Pingpong.world.ball_radius` | 球半径。 | 不改。 |
| `FSM.Pingpong.planner.forward_motion_file` | 正手专家 npz，用于动态读取正手击球几何。 | 换策略对应 npz 时改。 |
| `FSM.Pingpong.planner.backward_motion_file` | 反手专家 npz，用于动态读取反手击球几何。 | 换策略对应 npz 时改。 |
| `FSM.Pingpong.planner.entry_motion_file/frame` | Velocity 到 Pingpong 的入口参考 pose；切换阶段会插值到这一帧。 | 入口姿态不合适时改。 |
| `FSM.Pingpong.planner.entry_joint_mode` | 入口插值时哪些关节跟随 npz。`waist_arms` 表示腿保持进入 Pingpong 瞬间的当前姿态，只切腰和双臂；`hitter_lower_waist_arms` 表示腿先切到 HITTER 默认腿姿态。 | 当前用 `waist_arms`，避免 Velocity 切入时腿快速下蹲。 |
| `FSM.Pingpong.planner.fallback_motion_file/frame` | Pingpong 刚进入、还没有上一条 cmd 时生成第一条 forehand 初始 cmd 的参考帧。 | 换 npz 时同步改。 |
| `FSM.Pingpong.planner.waiting_initial_t_to_hit` | 没有可打球/恢复结束后的 idle waiting `t_to_hit` 初值。 | 当前用 `-0.02s`，之后每 tick 持续减小。 |
| `FSM.Pingpong.planner.forehand_offset_base/backhand_offset_base` | npz 不可用时的 fallback 击球偏移。 | 正常由 npz 覆盖。 |
| `FSM.Pingpong.planner.y_mid_base` | 正反手切换边界。 | 正常由 npz 覆盖或自动计算。 |
| `FSM.Pingpong.planner.x_hit_world` | 虚拟击球平面世界 x。当前 `0.40`，即球桌近边。 | 想让击球面前后移动时改。 |
| `FSM.Pingpong.planner.z_min_world/z_max_world` | planner 可接受击球高度范围。 | 球太高/太低时调。 |
| `FSM.Pingpong.planner.target_land_world` | 期望回球落点。 | 训练目标变化时改。 |
| `FSM.Pingpong.planner.flight_time/paddle_cor` | 由目标落点反推期望拍速的飞行时间和球拍恢复系数。 | 回球速度不合理时调。 |
| `FSM.Pingpong.planner.planner_dt/planner_max_time` | 小球预测积分步长和最长预测时间。`planner_dt` 不是 cmd 频率；cmd 频率由 `policy_dt` 决定。 | 一般不改。 |
| `FSM.Pingpong.planner.planner_drag_k` | planner 空气阻力系数。 | 和训练 planner 对齐。 |
| `FSM.Pingpong.planner.planner_bounce_ch/cv` | planner 桌面反弹水平/竖直系数。 | 和训练 planner 对齐。 |
| `FSM.Pingpong.planner.planner_min_t_to_hit/max_t_to_hit` | 可接受击球时间窗口。当前 `min=0.0`，允许 `t_hit_abs` 随最新 valid planner 结果一直修正到击球平面。 | 来球太远时主要调 `max_t_to_hit`。 |
| `FSM.Pingpong.planner.min_incoming_speed_x` | 小球朝机器人飞来的最小 `-vx` 阈值。若 `vx >= -min_incoming_speed_x`，planner 保持上一条 cmd。 | 当前 `0.05`。 |
| `FSM.Pingpong.planner.min_ball_z_world` | 小球中心低于该世界 z 时视为在桌下/坏状态，planner 保持上一条 cmd。 | 当前 `0.74`。 |
| `FSM.Pingpong.planner.max_table_bounces_before_fallback` | 预测轨迹在到达击球平面前弹桌次数达到该值时放弃 swing。 | 当前 `4`，即 4 次及以上保持上一条 cmd；设为 `0` 可关闭这个保护。 |
| `FSM.Pingpong.planner.freeze_time_before_hit` | 临近击球前是否冻结 spatial target。真实 tracking 当前 `0.0`。即使将来打开 spatial freeze，`t_hit_abs/t_to_hit` 也会继续跟随最新 valid planner 结果，不再时间锁死。 | 真实球信息持续新鲜时保持 `0.0`。 |
| `FSM.Pingpong.planner.post_swing_time` | 击球后 follow-through/等待下限时间；坏球或无球且已有上一条 cmd 时，`t_to_hit` 固定为 `-post_swing_time`。 | 当前 `0.60s`。 |

## Sim2Sim 启动

Terminal 1: 启动 MuJoCo 仿真器。它会加载带球拍的 G1 XML，生成带球桌/球网/球的 scene，发布 DDS lowstate 和 ROS2 ball/base topic。

```bash
cd /home/woan/unitree_mujoco/simulate_python_pingpong
source /opt/ros/humble/setup.bash
conda activate unitree-mujoco
python unitree_pingpong_mujoco.py --network lo --auto-start
```

Terminal 2: 启动 C++ controller。它通过 DDS 接收 MuJoCo lowstate/wireless，通过 ROS2 接收 ball/base，通过 DDS 写回 lowcmd。

```bash
cd /home/woan/HumanoidProject/unitree_rl_lab/deploy/robots/g1_23dof_pingpong/build
source /opt/ros/humble/setup.bash
./g1_pingpong_ctrl --network lo
```

本机 sim2sim 用 `--network lo`。真机时把 controller 的 `--network` 改成真实网卡，例如 `--network enp129s0`，同时由真实 tracking 系统发布 `/pingpong/ball_state` 和 `/pingpong/base_pose`。

## 通信流程

```text
MuJoCo Python
  -> DDS rt/lowstate              -> C++ controller 读机器人状态
  -> DDS rt/sportmodestate        -> C++ controller 读高层状态
  -> DDS rt/wirelesscontroller    -> C++ controller 接收键盘模拟手柄
  -> ROS2 /pingpong/ball_state    -> C++ planner 预测击球 cmd
  -> ROS2 /pingpong/base_pose     -> C++ planner/obs 使用真实 pelvis pose

C++ controller
  -> DDS rt/lowcmd                -> MuJoCo Python 应用 PD/电机命令
```

MuJoCo sim2sim 中，小球和 base pose 从仿真直接读出。真机中，小球和 base pose 由你的动捕/tracking 设备发布到同名 ROS2 topic。controller 侧逻辑相同。

## 初始化流程

1. MuJoCo Python 读取 `g1_23dof_rev_1_0_paddle.xml`，生成含球桌、球网、小球的 scene。
2. MuJoCo reset 机器人到 `RESET_ROOT_POS_W=(-0.138,0,0.74)`，关节角使用策略 `params/deploy.yaml` 的默认 offset。
3. MuJoCo 采样一次 serve，使小球朝 `x=0.40` 的虚拟击球平面飞来。
4. MuJoCo 启动 DDS bridge，开始发布 lowstate/highstate/wireless，并开始发布 ROS2 ball/base。
5. C++ controller 启动后读取 `config.yaml`，加载 Pingpong policy、npz 几何和 ROS2 subscriber。
6. Python `--auto-start` 发送虚拟手柄 `Up -> X`，让 controller 从 `Passive -> FixStand -> Velocity`。
7. 你按 `u` 或 viewer 的 `U`，Python 发送 dpad up，controller 从 `Velocity -> Pingpong`。
8. `Pingpong` 先在 `switch_blend_s` 内把当前关节平滑插到 `entry_motion_file/frame` 的 npz pose。
9. planner 对球轨迹积分，找到穿过 `x_hit_world=0.40` 的时间和位置，生成 `p_hit_world/v_racket_hat_world/n_target_world/t_to_hit`。
10. 入口插值结束后，HITTER actor 每个 tick 都接管输出，controller 写 DDS lowcmd，MuJoCo 应用到机器人。
11. 如果球已经不可打、topic 超时、球在桌下、不朝机器人飞来、或预测弹桌次数过多，planner 不生成新空间目标；已有上一条 cmd 时保持上一条 cmd，只把 `t_to_hit` 固定为 `-post_swing_time`。只有刚进入 Pingpong 且没有上一条 cmd 时，才用 forehand npz 生成第一条初始 cmd。

## Build

```bash
cd /home/woan/HumanoidProject/unitree_rl_lab
source /opt/ros/humble/setup.bash
cmake -S deploy/robots/g1_23dof_pingpong -B deploy/robots/g1_23dof_pingpong/build
cmake --build deploy/robots/g1_23dof_pingpong/build -j
```




1. 仿真和 controller 几何统一为：

```
table near edge / hit plane: x = 0.40 robot reset pelvis/root:     x = -0.138 distance:                    0.538 m
```

1. MuJoCo viewer 现在会显示：

```
world_frame_*        仿真/训练世界 W 原点坐标轴，位于 (0,0,0) ros_topic_frame_*    ROS topic 输入坐标 M 原点坐标轴，位于 (1.77,0,0.735) hit_plane_marker     橙色线，x=0.40，球桌近边/虚拟击球平面 reset_root_marker    紫色球，机器人 reset pelvis/root 位置
```

颜色约定：红色 +x，绿色 +y，蓝色 +z。

1. ROS topic 坐标转换现在单独用：

```
INPUT_ORIGIN_IN_TRAINING_WORLD = TABLE_CENTER_W
```

ball 和 base 都统一为：

```
ball_M = ball_W - INPUT_ORIGIN_IN_TRAINING_WORLD base_M = base_W - INPUT_ORIGIN_IN_TRAINING_WORLD
```

所以当前 reset 后应该看到：

```
base_W = [-0.138, 0.0, 0.74] base_M = [-1.908, 0.0, 0.005]
```

我已经用 --check --no-render --ros-backend none 验证过，输出确实是：

```
base W pos=[-0.138, 0.0, 0.74] base M pos=[-1.908, 0.0, 0.005] expected reset base M=[-1.908, 0.0, 0.005]
```

**通信流程**

```
MuJoCo Python  -> DDS rt/lowstate / rt/sportmodestate / rt/wirelesscontroller  -> ROS2 /pingpong/ball_state, /pingpong/base_pose C++ controller  <- DDS lowstate / wireless  <- ROS2 ball/base  -> planner + obs + policy  -> DDS rt/lowcmd MuJoCo Python  <- DDS rt/lowcmd  -> 应用到 MuJoCo 机器人
```

**初始化流程**

1. Python 加载带球拍 G1 XML，生成球桌/球网/小球场景。
2. reset 到 base_W=(-0.138,0,0.74)。
3. 发球朝 hit_x=0.40。
4. Python 发布 DDS lowstate 和 ROS2 ball/base。
5. C++ 读取 config.yaml、策略、npz、ROS2 topic。
6. 自动进入 Velocity。
7. 你按 u 进入 Pingpong。
8. planner 在 W 中预测小球穿过 x=0.40，生成 cmd。
9. policy 输出 lowcmd，MuJoCo 执行。



# 查看误差

python deploy/robots/g1_23dof_pingpong/plot_hit_trace.py --window 20 --interval 0.1

