# Pingpong Real Rally Env v1 真实对打训练规划

本文档规划一个新的真实乒乓球对打训练任务：`Unitree-G1-23dof-Pingpong-HITTER-REAL`。

该任务不替换现有 `Unitree-G1-23dof-Pingpong-HITTER` 空挥拍任务。旧任务继续作为预训练、消融实验和几何跟踪稳定性基线；新任务在旧任务的 whole-body striking policy 结构上加入真实球、真实球桌、随机发球机 / 对手模拟器、训练专用 batched planner、真实击球接触和回球落点成功率判定。

核心设计原则：

- **Policy 仍跟踪 planner command**，第一版不把 raw ball state 直接喂给 Actor。
- **真实成功率来自球的物理 rollout**，不再用空挥拍几何判定作为主成功率。
- **旧几何奖励继续保留**，用于维持击球窗口内的末端位置、速度和拍面朝向学习信号。
- **`t_post_swing` 固定为 `0.60s`**，消除恢复段随机时长对 Actor 不可观测的问题；但 expert clip 的 pre/post 浮点帧插值和 slerp/lerp 机制完整保留。
- **IMU/base 语义锁定为 pelvis/root base frame**，不使用 URDF 中的 `imu_in_torso` 作为 policy base sensor。

参考依据：

- ITTF 桌面尺寸、高度和反弹规则：桌面 `2.74m x 1.525m`，台面高 `0.76m`，标准球从 `0.30m` 落下反弹约 `0.23m`。本文只把这些作为 PhysX 参数校准目标，不写死所谓“标准刚度 N/m”。参考：`https://cdn.megaspin.net/rules/pdf/2025/ittf-rules-2.pdf`
- DeepMimic / motion imitation 常用 phase/clock 解决参考动作相位同步问题。本文为了保持旧 obs 维度，选择固定 `t_post_swing=0.60s`，让 post 段负的 `t_to_hit` 唯一对应恢复 phase。参考：`https://xbpeng.github.io/projects/DeepMimic/DeepMimic_2018.pdf`

---

## 1. 任务边界

### 1.1 新旧任务关系

| 项 | 旧任务 `HITTER` | 新任务 `HITTER-REAL` |
|---|---|---|
| 主要目标 | 空挥拍跟踪 `p_hit / v_racket / n_target` | 真实球 rollout 下击球、过网、落点 |
| command 来源 | synthetic cmd sampling | ball state + `planner_for_training.py` |
| 成功率 | strike window 几何阈值 | ball-racket contact + return outcome |
| 球物理 | 无真实球参与训练 | PhysX 真实球 + contact event tracker |
| planner | 训练端 inline Eq.5/Eq.6 | batched torch planner |
| obs | Actor=86, Critic=213 | 第一版保持旧结构，不加入 raw ball state |
| expert tracking | 保留 | 保留 |
| 用途 | 预训练 / ablation | 真实对打训练 |

### 1.2 第一版明确不做

- 不实现完整 humanoid opponent；对手先用发球机 / opponent simulator 表示。
- 不把 raw ball position / velocity 加进 Actor obs。
- 不修改 runtime `mdp/planner.py` 的算法逻辑。
- 不用 reward 或 reset 手动改写球的反弹速度来假装物理正确。
- 不把 `imu_in_torso` 当作 base IMU。
- 不随机 `t_post_swing`。

---

## 2. 文件与任务注册

### 2.1 建议新增文件

| 文件 | 作用 |
|---|---|
| `source/unitree_rl_lab/unitree_rl_lab/tasks/pingpong/final_real_v1.md` | 本规划书 |
| `source/unitree_rl_lab/unitree_rl_lab/tasks/pingpong/mdp/planner_for_training.py` | batched torch planner |
| `source/unitree_rl_lab/unitree_rl_lab/tasks/pingpong/mdp/real_commands.py` | 真实球驱动 command term，或在现有 `commands.py` 中新增 real 版本 |
| `source/unitree_rl_lab/unitree_rl_lab/tasks/pingpong/mdp/real_events.py` | ball reset、serve machine、ball contact state machine |
| `source/unitree_rl_lab/unitree_rl_lab/tasks/pingpong/mdp/real_rewards.py` | ball outcome rewards，可复用旧 `rewards.py` |
| `source/unitree_rl_lab/unitree_rl_lab/tasks/pingpong/mdp/real_curriculums.py` | 真实任务 curriculum |
| `source/unitree_rl_lab/unitree_rl_lab/tasks/pingpong/robots/g1_23dof/hitter_real/hitter_real_env_cfg.py` | 新 env cfg |
| `source/unitree_rl_lab/unitree_rl_lab/tasks/pingpong/robots/g1_23dof/hitter_real/__init__.py` | gym task 注册 |

如果为了减少文件数量，`real_*` 可以合并进现有 mdp 文件，但必须保证旧 `HITTER` 行为不受影响。

### 2.2 新任务注册名

```text
Unitree-G1-23dof-Pingpong-HITTER-REAL
```

旧任务：

```text
Unitree-G1-23dof-Pingpong-HITTER
```

必须继续可运行，不允许被真实任务覆盖。

---

## 3. Scene / Sim

### 3.1 训练 world frame 唯一标准

| 量 | 数值 / 方向 | 说明 |
|---|---|---|
| +x | 从机器人侧指向对方半桌 | 回球方向 |
| +y | 球桌横向 | 右手坐标系 |
| +z | 竖直向上 | IsaacLab world up |
| table center | `(1.77, 0.0, 0.735)` | 5cm 薄桌面中心 |
| table size | `(2.74, 1.525, 0.05)` | ITTF 尺寸 + 5cm 厚度 |
| table top | `z = 0.76` | `0.735 + 0.025` |
| near edge | `x = 0.40` | 机器人侧桌沿 |
| net plane | `x = 1.77` | 球桌中心线 |
| far edge | `x = 3.14` | 对方侧桌沿 |
| half width | `|y| <= 0.7625` | `1.525 / 2` |
| ball radius | `0.02m` | 40mm 球 |
| ball mass | `0.0027kg` | 2.7g 球 |
| ball resting center | `z = 0.78` | table top + radius |
| net top | `z = 0.9125` | table top + 0.1525 |

### 3.2 球桌建模

直接照抄当前 hitter 工程的薄桌面，不再使用旧文档中的 0.76m 厚大方块：

```python
table = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Table",
    spawn=sim_utils.CuboidCfg(
        size=(2.74, 1.525, 0.05),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.9,
            dynamic_friction=0.8,
            restitution=0.2,  # 初值；真实任务中以落球测试重新校准
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.35, 0.55)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(1.77, 0.0, 0.735), rot=(1.0, 0.0, 0.0, 0.0)),
)
```

说明：

- 桌子允许底部悬空，这是更接近真实球桌的建模方式。
- 不用一个从地面到桌面的完整大方体代替球桌。
- robot-table contact penalty 继续只惩罚机器人撞桌，不作为 termination。

### 3.3 球网

第一版球网可以用 kinematic cuboid / mesh 建模：

| 项 | 数值 |
|---|---|
| prim path | `{ENV_REGEX_NS}/Net` |
| center x | `1.77` |
| center y | `0.0` |
| center z | `0.76 + 0.1525 / 2 = 0.83625` |
| size x | `0.02` |
| size y | `1.83`，略宽于球桌 |
| size z | `0.1525` |
| collision | 开启 |
| kinematic | 是 |

成功判定中，球过网必须满足：

```python
ball crosses x = 1.77 from robot side to opponent side
ball_center_z > net_top + ball_radius
```

其中：

```python
net_top = 0.9125
ball_radius = 0.02
clear_net_threshold_z = 0.9325
```

如果 PhysX 中球撞网，则该 swing 记为 `net_contact=1`，`return_success=0`。

### 3.4 乒乓球

建议用 `RigidObjectCfg` + sphere spawn：

| 项 | 数值 / 说明 |
|---|---|
| prim path | `{ENV_REGEX_NS}/Ball` |
| radius | `0.02m` |
| mass | `0.0027kg` |
| CCD | 必须开启 |
| gravity | 开启 |
| contact | table / net / racket / robot body |
| initial restitution | `sqrt(0.23 / 0.30) ≈ 0.876`，仅作落球校准初值 |
| material randomization | Stage 4 后再开启 |

不写死球或球桌的“标准 stiffness”。若 Isaac / PhysX 提供 contact stiffness / damping 参数，它们只是数值稳定性调参项；最终验收以落球反弹高度和接触事件稳定性为准。

### 3.5 传感器

| Sensor | 作用 |
|---|---|
| `contact_forces` | 旧机器人 contact sensor，服务 feet / undesired contact |
| `robot_table_contact` | 旧 robot-table filtered sensor，服务 `r_table_contact` |
| `ball_table_contact` | 新增，检测球与桌面接触 |
| `ball_racket_contact` | 新增，检测球与 `right_paddle_blade` 接触 |
| `ball_net_contact` | 新增，检测球与 net 接触 |
| `ball_robot_contact` | 新增，检测球与非拍面 robot body 接触 |

如果 IsaacLab `ContactSensorCfg` 难以直接 filter ball-vs-specific-body，可在 env step 中读取 ball / robot contact report 或用 ball state crossing + force proxy 实现事件 tracker。规划要求是事件语义必须稳定，不强制具体 sensor API。

---

## 4. IMU / Base Frame 硬约束

论文 Table I 语义是：

| 观测 | 含义 |
|---|---|
| `omega_base` | base frame 角速度 |
| `g_base` | 重力在 base frame 投影 |
| `e_base,x` | base forward vector / yaw encoding |

本工程中 base frame 锁定为 **pelvis/root frame**：

- URDF root link 是 `pelvis`。
- URDF 中 `imu_in_pelvis` 固定在 `pelvis` 上，`rpy=0 0 0`。
- `imu_in_torso` 存在，但不参与 policy obs。
- `root_pos_w / root_quat_w / root_ang_vel_b` 与 pelvis/root base 语义一致。

实现要求：

- Actor 的 `base_ang_vel / projected_gravity / base_yaw` 继续使用当前 `base_ang_vel_imu / projected_gravity_imu / base_yaw_encoding_imu`。
- Critic 继续使用 clean `mdp.base_ang_vel / mdp.projected_gravity / base_yaw_encoding`。
- 部署端如果传入 torso IMU，必须先转换到 pelvis/base frame，再喂 policy。
- 单元测试必须覆盖 identity quat、90deg yaw、projected gravity、pelvis/root 与 torso frame 防混用。

---

## 5. Command / Planner 接口

### 5.1 第一版 obs 不加 raw ball state

Actor obs 保持旧任务结构：

| # | Term | 维度 | 来源 |
|---|---|---:|---|
| 1 | `base_ang_vel` | 3 | pelvis/root IMU channel |
| 2 | `projected_gravity` | 3 | pelvis/root IMU channel |
| 3 | `base_yaw` | 2 | pelvis/root yaw encoding |
| 4 | `base_err` | 2 | `cmd.p_base_xy_world - root_xy` |
| 5 | `hit_pos` | 3 | `R_base^T (cmd.p_hit_world - root_pos)` |
| 6 | `racket_vel` | 3 | `cmd.v_racket_hat_world` |
| 7 | `t_to_hit` | 1 | command timing |
| 8 | `joint_pos` | 23 | delayed joint encoder |
| 9 | `joint_vel` | 23 | delayed joint encoder |
| 10 | `last_action` | 23 | previous action |

Actor 维度仍为 `86`。`ball_pos_world / ball_vel_world / raw planner internals` 不进 Actor obs。

Critic 第一版保持旧结构，仍为 `213`。如果后续加入 privileged ball state，必须作为 v2 文档单独记录，不能偷偷改变本版 obs 维度。

### 5.2 `planner_for_training.py`

新增训练专用 batched torch planner。它服务 RL 环境内大量 env 并行，不需要保留 runtime planner 的 Python object/state API，但输出字段必须完全对齐 runtime planner 当前 packet / `PingpongCommand` 字段。

#### 输入

| 字段 | shape | frame | 说明 |
|---|---:|---|---|
| `ball_pos_world` | `(N, 3)` | world | 当前球心位置 |
| `ball_vel_world` | `(N, 3)` | world | 当前球速度 |
| `robot_root_pos_world` | `(N, 3)` | world | robot root / pelvis 位置 |
| `robot_root_quat_world` | `(N, 4)` | wxyz | robot root / pelvis 姿态 |
| `target_land_world` | `(N, 3)` | world | 期望回球落点 |
| `table_top_z` | scalar or `(N,)` | world | 默认 `0.76` |
| `ball_radius` | scalar or `(N,)` | — | 默认 `0.02` |
| `valid_mask` | `(N,)` | — | env 是否需要规划 |

#### 输出

| 字段 | shape | 说明 |
|---|---:|---|
| `p_hit_world` | `(N, 3)` | 预测击球点 |
| `v_ball_in_world` | `(N, 3)` | 预测击球瞬间来球速度 |
| `v_ball_out_world` | `(N, 3)` | 由目标落点和飞行时间推导的出球速度 |
| `v_racket_hat_world` | `(N, 3)` | Eq.5/Eq.6 得到的目标拍速 |
| `n_target_world` | `(N, 3)` | 目标拍面法向 |
| `target_land_world` | `(N, 3)` | 实际使用的目标落点 |
| `p_base_xy_world` | `(N, 2)` | base target |
| `t_to_hit` | `(N,)` | 当前时刻到预测击球时刻的剩余时间 |
| `swing_type` | `(N,) long` | `0=backhand`, `1=forehand` |
| `planner_valid` | `(N,) bool` | 是否找到可用规划 |
| `plan_mode` | `(N,) int` | `0=invalid`, `1=fresh`, `2=held`, `3=frozen` |
| `bounce_count_pred` | `(N,)` | 预测到 hit 前球与桌面反弹次数 |
| `x_hit_used` | `(N,)` | 实际使用的击球平面或 hit x |
| `fallback_reason` | `(N,) int` | 失败原因诊断 |

### 5.3 Planner 物理模型

训练 planner 第一版可复用现有 runtime planner 的模型形式：

```python
a = -k * ||v|| * v + g
v_plus = diag(Ch, Ch, -Cv) @ v_minus
```

说明：

- `k / Ch / Cv` 使用现有拟合参数作为预测模型初值。
- 真实环境 success 以 PhysX rollout 为准，planner 模型只负责生成 command 和统计 prediction error。
- Stage 4 可以对 planner 输入加入 observation noise / delay / drop frame。

### 5.4 Eq.5 / Eq.6 回球目标求解

给定：

- `p_hit_world`
- `v_ball_in_world`
- `target_land_world`
- `flight_time`
- `paddle_cor`

先计算出球速度：

```python
v_ball_out_world = (target_land_world - p_hit_world) / flight_time + [0, 0, 0.5 * g * flight_time]
```

再计算拍面法向：

```python
delta_v = v_ball_out_world - v_ball_in_world
n_target_world = delta_v / ||delta_v||
```

退化分支：

```python
if ||delta_v|| < 1e-9:
    n_target_world = [-1, 0, 0]
```

最后计算法向目标拍速：

```python
v_in_n = dot(v_ball_in_world, n_target_world)
v_out_n = dot(v_ball_out_world, n_target_world)
v_pad_n = (v_out_n + paddle_cor * v_in_n) / (1 + paddle_cor)
v_racket_hat_world = v_pad_n * n_target_world
```

### 5.5 Base target 与 swing type

`swing_type` 继续使用当前 v5.7 规则：

```python
hit_y_base = (R_base^T @ (p_hit_world - root_pos_world))[1]
swing_type = forehand if hit_y_base > Y_MID_BASE else backhand
Y_MID_BASE = 0.157
```

`p_base_xy_world`：

```python
p_base_xy_world = p_hit_world.xy - R_yaw(root_quat) @ expert_offset_base[swing_type]
```

expert offset：

| swing | `expert_offset_base` |
|---|---|
| forehand | `(+0.496, +0.208)` |
| backhand | `(+0.428, +0.106)` |

### 5.6 Planner target freeze

为避免临近击球时 command 抖动：

- 当 `t_to_hit > 0.20s`：允许 planner 更新 spatial target。
- 当 `t_to_hit <= 0.20s`：冻结 `p_hit_world / v_racket_hat_world / n_target_world / swing_type / p_base_xy_world`。
- 冻结后只更新 `t_to_hit = t_hit_abs - now`。
- 如果 planner invalid，但已有 frozen target，则继续执行 frozen swing。
- 如果 planner invalid 且没有 active swing，则保持 idle / wait 状态，不给虚假击球 command。

### 5.7 Active swing 生命周期

```text
NoActiveSwing
    ├─ planner_valid -> ActivePreStrike
    └─ otherwise     -> WaitServe

ActivePreStrike
    ├─ t_to_hit > 0.20s     -> planner may update target
    ├─ 0 < t_to_hit <=0.20s -> freeze target
    ├─ abs(t_to_hit)<=0.06s -> strike window rewards/events active
    ├─ legal contact        -> ActivePostStrike
    └─ t_to_hit < -0.10s without contact -> Missed

ActivePostStrike
    ├─ t_to_hit >= -0.60s -> recovery ref playback
    ├─ new incoming valid planner with t_to_hit_raw <= 0.90s -> interrupt to ActivePreStrike
    └─ t_to_hit < -0.60s -> WaitServe / next ball

Missed
    ├─ record failure
    └─ wait 0.20s diagnostics then reset or next serve
```

`t_post_swing` 固定：

```python
t_post_swing = 0.60
```

但 ref interpolation 保留：

```python
frame_f = motion_loader.frame_from_step(
    cur_step=cmd.cur_step,
    t_pre=cmd.t_pre_initial,
    t_post=0.60,
    dt=env.step_dt,
)
```

也就是说，post 段仍从 impact frame 连续插值到 clip last frame，只是恢复总时长固定为 `0.60s`。

---

## 6. 发球机 / Opponent Simulator

### 6.1 Stage 1 单球中心发球

球从对方半桌上方或发球机口 reset：

| 参数 | 范围 |
|---|---|
| `serve_pos_x` | `2.55–3.05` |
| `serve_pos_y` | `-0.20–0.20` |
| `serve_pos_z` | `0.95–1.25` |
| `serve_speed` | `2.0–3.5 m/s` |
| `serve_target_x` | `0.35–0.55` |
| `serve_target_y` | `-0.35–0.35` |
| `serve_target_z_at_hit` | `0.30–0.55` |

要求：

- 球朝机器人侧运动。
- planner 有足够时间预测，初始 `t_to_hit` 建议 `0.45–0.90s`。
- 先不加入 spin。

### 6.2 Stage 2 单球随机发球

扩展范围：

| 参数 | 范围 |
|---|---|
| `serve_pos_y` | `-0.65–0.65` |
| `serve_speed` | `2.0–6.0 m/s` |
| `serve_target_y` | `-0.75–0.75` |
| `serve_pitch` | 覆盖上旋 / 下落来球 |
| `first_bounce_side` | 可配置：机器人侧一次反弹或空中来球 |

### 6.3 Stage 3 连续对打

成功回球后不 reset episode：

1. 记录当前球 `return_success / target_success`。
2. 等球到对方半桌或越过对方区域。
3. opponent simulator 生成下一球。
4. planner 重新创建 active swing。

为了避免无限等待：

```python
if time_since_last_valid_ball > 1.5s:
    force_next_serve_or_reset()
```

---

## 7. Rewards

### 7.1 旧 reward 保留

保留当前旧任务主体：

| 组 | Term | 说明 |
|---|---|---|
| imitation | `r^p`, `r^v`, `r^bp` | 跟踪 expert joint/body |
| goal | `r_g_pos`, `r_g_vel`, `r_g_ori`, `r_g_base` | 击球目标跟踪 |
| regularization | alive / action / torque / acc / feet / contact | 旧任务同款 |
| table | `r_table_contact` | robot 撞桌软惩罚 |

几何 sparse reward 仍只在：

```python
abs(t_to_hit) <= 0.06
```

### 7.2 新增 ball outcome reward

| Term | 公式 / 事件 | weight | gate |
|---|---|---:|---|
| `r_ball_contact` | 合法首次 ball-racket contact | `+2.0` | once per swing |
| `r_return_direction` | contact 后 `v_ball_x > 0` | `+0.5` | once after contact |
| `r_clear_net` | 过网时 `z > net_top + ball_radius` | `+1.0` | once per swing |
| `r_opponent_land` | contact 后第一落点在对方半桌内 | `+3.0` | once per swing |
| `r_target_land` | 第一落点到 target xy 距离 `< target_land_radius` | `+2.0` | once per swing |
| `r_illegal` | 非拍面接触 / 多次触球 / 出界 / 撞网 | `-2.0` | event |

### 7.3 合法接触定义

合法 ball-racket contact 必须同时满足：

```python
contact_pair == (ball, right_paddle_blade)
abs(t_to_hit) <= 0.08
first_robot_contact_body == right_paddle_blade
not already_contacted_this_swing
dot(v_ball_after - v_ball_before, n_target_world) > 0
```

如果球先接触 `right_wrist_roll_rubber_hand`、手臂、躯干、腿或地面，则：

```python
illegal_contact = True
hit_success = False
```

### 7.4 回球成功定义

`return_success=1` 当且仅当：

```python
legal_ball_racket_contact
ball crosses net plane x=1.77 after contact
ball_center_z_at_crossing > 0.9125 + 0.02
first_post_contact_table_bounce lies in opponent half:
    1.77 < bounce_x < 3.14
    abs(bounce_y) < 0.7625
```

`target_success=1`：

```python
return_success
norm(first_bounce_xy - target_land_world.xy) < target_land_radius
```

`planner_success=1`：

```python
planner_valid_at_swing_start
abs(actual_contact_time - predicted_hit_time) < 0.08
norm(actual_contact_pos - p_hit_world_frozen) < 0.10
```

---

## 8. Curriculum

### 8.1 Stage 表

| Stage | 名称 | 目标 | 推进条件 |
|---|---|---|---|
| 0 | Calibration | 落球 / 发球 / planner debug | 手动验收 |
| 1 | Center Serve Single Ball | 中心低速单球击中 | `hit_success > 0.60` |
| 2 | Random Serve Single Ball | 随机 y/速度/落点 | `return_success > 0.45` |
| 3 | Rally | 成功回球后继续下一球 | `return_success > 0.55` |
| 4 | Realization | material/noise/delay randomization | `target_success > 0.35` |

统计窗口建议：最近 `1000` 个 completed swing。

### 8.2 `target_land_radius`

| Stage | radius |
|---|---:|
| 1 | `0.45m` |
| 2 early | `0.45m -> 0.35m` |
| 2 late | `0.35m -> 0.30m` |
| 3 | `0.30m` |
| 4 | `0.30m`，可继续收紧到 `0.25m` |

### 8.3 Serve range curriculum

| 项 | Stage 1 | Stage 2 | Stage 3/4 |
|---|---|---|---|
| `serve_y` | `[-0.20, 0.20]` | `[-0.65, 0.65]` | `[-0.75, 0.75]` |
| speed | `[2.0, 3.5]` | `[2.0, 6.0]` | `[2.0, 6.0]` |
| `t_to_hit_raw` | `[0.55, 0.90]` | `[0.40, 0.90]` | `[0.35, 0.90]` |
| target land | fixed center | random around center | random opponent half |

---

## 9. Domain Randomization

### 9.1 Robot DR 使用当前代码实况

| Event | mode | 当前参数 |
|---|---|---|
| `physics_material` | startup | robot all bodies, static friction `(0.3,1.6)`, dynamic friction `(0.3,1.2)`, restitution `(0.0,0.5)`, `num_buckets=64` |
| `add_link_mass` | startup | all bodies mass scale `(0.9,1.1)` |
| `randomize_joint_friction` | startup | all joints friction scale `(0.5,1.5)` |
| `randomize_joint_damping` | startup | all joints damping scale `(0.7,1.3)` |
| `randomize_imu_offset` | startup | gaussian `sigma_deg=2.0` |
| `randomize_comm_delay` | startup | `max_delay_steps=1` |
| `add_joint_default_pos` | startup | all joints default pos add `(-0.01,0.01)` |
| `base_com` | startup | `torso_link`, x `(-0.025,0.025)`, y/z `(-0.05,0.05)` |
| `push_robot` | interval | every `1.0–3.0s`, velocity range from current `VELOCITY_RANGE` |

当前 `VELOCITY_RANGE`：

```python
{
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}
```

### 9.2 Observation DR

Actor：

- `base_ang_vel = DelayedObservation(base_ang_vel_imu)`
- `projected_gravity = DelayedObservation(projected_gravity_imu)`
- `base_yaw = DelayedObservation(base_yaw_encoding_imu)`
- `joint_pos = DelayedObservation(joint_pos_rel)`
- `joint_vel = DelayedObservation(joint_vel_rel)`
- command noise 只作用在 Actor command obs。

Critic：

- 使用 clean base / joint / command obs。
- 不使用 delayed/IMU-offset sensor wrapper。

### 9.3 Ball / table DR

Stage 0–3：

- 不开启 ball/table material randomization。
- 先完成落球校准和稳定接触事件。

Stage 4：

| 项 | 范围 |
|---|---|
| ball restitution scale | `[0.95, 1.05]` around calibrated value |
| table restitution scale | `[0.95, 1.05]` around calibrated value |
| ball mass scale | `[0.98, 1.02]` |
| serve velocity noise | gaussian, `sigma=0.05–0.15 m/s` |
| planner ball position noise | gaussian, `sigma=0.002–0.008m` |
| planner ball velocity noise | gaussian, `sigma=0.03–0.10m/s` |
| planner delay | `0–2` policy steps |
| planner dropout | max `5%` frames, hold-last-safe |

---

## 10. Termination

| Termination | 条件 |
|---|---|
| timeout | episode length `10s` |
| bad orientation | 复用旧任务 `bad_orientation limit_angle=0.8rad` |
| low root | 复用旧任务 `root_height < 0.30m` |
| severe undesired contact | 复用旧任务 body regex |
| ball dead | 球低于地面或离桌过远且当前 swing 已失败 |
| repeated illegal contact | 同一 episode 非法接触次数超过阈值 |

不 terminate：

- robot-table contact：只走 `r_table_contact` 软惩罚。
- 单次 miss：Stage 3/4 可以记录失败后发下一球，不必立即 reset。
- ball-net contact：记为回球失败，可继续或 reset，按 curriculum stage 决定。

---

## 11. Ref / Motion Playback

### 11.1 Expert clip 选择

- `swing_type=forehand` 使用当前 forward expert clip。
- `swing_type=backhand` 使用当前 backward expert clip。
- `swing_type` 不进 obs，只作为 ref selector 和 base target heuristic。

### 11.2 Impact 对齐

必须保持：

```python
t_to_hit == 0.0  <=>  ref_frame_f == impact_frame
```

允许浮点误差：

```python
abs(ref_frame_f - impact_frame) < 1e-4
```

### 11.3 Post-swing 固定时间但保留插值

锁定：

```python
t_post_swing = 0.60
```

保留当前 `motion_loader.frame_from_step` 逻辑：

```python
sim_t = cur_step * dt
pre_frame = (sim_t / t_pre).clamp(0,1) * impact
post_frame = impact + ((sim_t - t_pre) / 0.60).clamp(0,1) * (last - impact)
frame_f = where(sim_t <= t_pre, pre_frame, post_frame)
```

随后 joint/body/ref state 继续使用 lerp / slerp 浮点插值。

### 11.4 新球打断恢复

若处于 post recovery 且满足：

```python
planner_valid
t_to_hit_raw <= 0.90
ball moving toward robot side
```

则允许打断 post recovery，进入下一次 active swing。打断时：

- `cur_step` 重新置 0。
- `t_pre_initial = t_to_hit_raw`。
- `t_post_swing = 0.60`。
- 按新 `p_hit_world` 重新选 `swing_type` 和 ref clip。

---

## 12. Debug Draw / Monitor

### 12.1 Debug draw

必须支持单 env play 中绘制：

- 真实球当前轨迹。
- planner 预测轨迹。
- `p_hit_world`。
- `p_base_xy_world`。
- `n_target_world`。
- `v_racket_hat_world`。
- blade local +Y normal。
- net plane / net top。
- 实际第一落点。
- strike window 内 blade trajectory 到 `p_hit_world` 的距离曲线。

### 12.2 Monitor

| 指标 | 说明 |
|---|---|
| `real/hit_success_rate` | 合法 ball-racket contact |
| `real/return_success_rate` | 过网 + 对方半桌落点 |
| `real/target_success_rate` | 落入 target radius |
| `real/planner_valid_rate` | planner valid |
| `real/planner_time_error_mean` | 实际 contact time - predicted hit time |
| `real/planner_pos_error_mean` | 实际 contact pos - frozen `p_hit_world` |
| `real/net_clearance_mean` | 过网高度裕量 |
| `real/landing_error_mean` | 落点 xy error |
| `real/illegal_contact_rate` | 非拍面 / 多次触球 |
| `real/ball_net_contact_rate` | 撞网率 |
| `real/ball_table_bounce_count` | 每球桌面反弹数 |
| `real/post_recovery_interrupt_rate` | 恢复被新球打断比例 |
| `real/fall_rate` | 摔倒率 |
| `real/table_contact_penalty_mean` | robot-table 软惩罚均值 |

---

## 13. PPO / Training

第一版继续复用 mimic PPO cfg：

- actor/critic MLP `[512, 256, 128]`。
- PPO 超参不在本任务中重新发明，直接复用 `tasks/mimic/agents/rsl_rl_ppo_cfg.py` 中现有基础配置。
- 只覆盖 task entry、experiment name、最大迭代数和必要 env 数。

推荐训练顺序：

1. 使用旧 `HITTER` 训练或加载几何击球 policy。
2. 切到 `HITTER-REAL` Stage 1，冻结较简单发球分布。
3. 打开 Stage 2 随机发球。
4. 打开 Stage 3 连续对打。
5. 最后打开 Stage 4 真实化 DR。

---

## 14. 验证计划

### 14.1 静态 / 单元测试

| Test | 验收 |
|---|---|
| table geometry | size `(2.74,1.525,0.05)`, center `(1.77,0,0.735)`, top `0.76` |
| net geometry | net x `1.77`, top `0.9125` |
| ball geometry | radius `0.02`, mass `0.0027`, resting center `0.78` |
| quaternion | identity / 90deg yaw / wxyz convention |
| IMU frame | pelvis/root 与 torso 混用检测 |
| planner shapes | 所有输出 shape/device/dtype 正确 |
| planner field names | 输出字段对齐 `PingpongCommand` |
| phase | `t_post_swing==0.60` 且 ref post frame 连续 |
| impact alignment | `t_to_hit=0` 时 `frame_f=impact_frame` |

### 14.2 物理校准测试

| Test | 验收 |
|---|---|
| drop test | 30cm 落球反弹 `23cm ± 2cm` |
| table bounce | ball-table contact event 稳定 |
| net contact | ball-net event 可检测 |
| racket contact | ball-racket event 可检测 |
| illegal body contact | ball 先撞非拍面 body 可检测 |

### 14.3 Play smoke

单 env 可视化检查：

- 发球机能发球到机器人侧可击区域。
- planner 预测轨迹与真实球轨迹同向、误差可解释。
- frozen target 在 `t_to_hit<=0.20s` 后不抖。
- strike window 与 expert impact frame 对齐。
- 真实球能与球拍发生 contact。
- 成功回球可过网并落到对方半桌。

### 14.4 Training smoke

| Run | 验收 |
|---|---|
| 128 env rollout | 无 shape/device/contact sensor 崩溃 |
| 1k iter Stage 1 | `hit_success` 上升，摔倒率不过高 |
| Stage 2 smoke | planner valid rate 稳定 |
| Stage 3 smoke | episode 内可多 swing |
| Stage 4 smoke | DR 打开后不崩溃 |

---

## 15. 实现优先级

1. 新建 `hitter_real_env_cfg.py`：复制旧 hitter cfg，加入 ball / net / real command term。
2. 实现球桌、球、网 scene 与落球校准 play script。
3. 实现 `planner_for_training.py` batched 输出字段。
4. 实现 real command lifecycle：active swing、freeze、post recovery、interrupt。
5. 实现 ball contact event tracker。
6. 加入 ball outcome rewards 和 success metrics。
7. 加入 curriculum stage。
8. 完成 debug draw。
9. 跑单 env play smoke。
10. 跑 128 env rollout 与 1k iter train smoke。

---

## 16. Assumptions

- 第一版 opponent 是发球机 / opponent simulator，不是完整 humanoid 对手。
- 第一版不把 raw ball state 给 Actor。
- 第一版以 PhysX 球 rollout 作为真实 success 来源。
- 现有 analytic `k / Ch / Cv` 仅用于 planner 预测和误差监控。
- PPO 复用 mimic cfg，网络结构保持 `[512, 256, 128]`。
- 旧 `HITTER` 不被修改为真实球任务。
- `t_post_swing=0.60s` 是硬约束。
- IMU/base 使用 pelvis/root frame 是硬约束。
