# HITTER Pingpong (G1 23dof) — 当前完整工程设计 (final_1)

> 本文档**取代**冗长的 `final.md`(v57→v65 的补丁流水账)。这里只描述 **当前最优** 的一套强化学习环境:把整个 pingpong task 的 **Scene / Robot / Observation / Action / Command / Reward / Termination / Event / Curriculum / Sim / PPO** 一次性、干净地讲清楚,每一项都给出 **当时为什么这么设计**(以及它解决了我训练中遇到的什么问题)。
>
> 那些被推翻的差参数不再单独罗列;它们的教训直接写进对应参数的「设计原因」里(`⚠️ 教训` 段)。
>
> **关于「3 个版本」**:整套 RL env(reward/curriculum/obs/…)是**唯一的、共享的**。所谓「当前最优的 3 个版本」指的是 **3 套机器人 PD(执行器增益)配置**,它们跑的是**同一套 env_cfg**,只是腿/腰/臂的 kp/kd 不同——见 **§3**。这是 sim2real 腿部抖动问题逼出来的三个候选。
>
> 论文:HITTER (Su et al. 2025, arXiv:2508.21043v2)。仿真:IsaacLab 5.1 + IsaacSim 5.1。

---

## 0. 总览:这套环境在学什么

机器人站在球桌近边外 ~0.54 m 处,面朝球桌(+X)。每一拍,一个 **command**(由 planner / 合成器给出)告诉策略:

- **击球点** `p_hit_world`(虚拟击球平面 x=近边 固定,y/z 由球决定)
- **击球瞬间球拍速度** `v_racket_hat_world`(论文 Eq.6)
- **击球瞬间球拍法向** `n_target_world`(论文 Eq.5,只进 reward 不进 obs)
- **该站到的底盘位置** `p_base_xy_world`(= 击球点 − demo 在击球帧的 拍↔骨盆 偏移)
- **距离击球还有多久** `t_to_hit`
- **正手/反手** `swing_type`(只用来选 demo clip 和给 reward 定符号,不进 obs)

策略要在 `t_to_hit` 归零的那一刻,把球拍(右手拍面)**准确送到击球点、达到目标速度、且拍面法向对准 `n_target`**,同时**全程站稳、横向移动到位、并模仿 demo 的上半身姿态**。

**核心难点(本设计反复围绕的两件事)**:
1. **23dof 右臂只有 5 DOF**(shoulder pitch/roll/yaw + elbow + wrist_roll;腕只有 roll,腰只有 waist_yaw)。「拍面位置 3 + 法向 2 = 5 DOF」→ **零冗余**。要既到位又摆正拍面,**必须**靠 waist_yaw + 底盘横移制造冗余。
2. **奖励稠密度倒置**:唯一每步都在起作用的稠密信号是**模仿**(把脸部关节钉向 demo 的固定拍面),而纠正拍面的 `goal_orientation` 只在 ~1 帧击球窗里给。→ 策略容易停在「位置/速度对、拍面歪」的别扭局部最优。

整套 curriculum / reward 设计的主线就是**在不破坏站立的前提下,把拍面学出来**。

---

## 1. 任务三层结构 + Planner

HITTER 把问题拆成三层,本仓库严格对应:

| 层 | 做什么 | 在本仓库 |
|---|---|---|
| **L0 弹道预测** | 给来球状态 → 预测击球点 `p_hit` 与时刻 `t_to_hit` | 训练端**跳过**(直接合成击球点);部署/`hitter_real` 用 planner |
| **L1 Eq.5/Eq.6** | 给击球点 + 出球目标 → 反解 `v_racket`、`n_target` | `commands.py::_solve_paddle_target`(训练 inline);`planner_for_training.py`(eval) |
| **L2 RL 策略** | 把 base 移到位 + 把拍准确送到击球点 | 本文档主体 |

### 1.1 Eq.5 / Eq.6(球-拍碰撞反解)

```
球台抛物线(已知击球点 p_hit、落点 p_land、飞行时间 T,反推出射速度):
    v_out = (p_land − p_hit) / T + (0, 0, ½·g·T)

Eq.5 球拍法向:        n̂ = (v_out − v_in) / ‖v_out − v_in‖
Eq.6 法向速度(COR e): v_pad,n = (v_out·n̂ + e·v_in·n̂) / (1 + e)
球拍速度(只给法向分量,切向自由): v̂_racket = v_pad,n · n̂
```

实现见 [mdp/commands.py `_solve_paddle_target`](mdp/commands.py)。`e = paddle_cor`,DR 采样 `U(0.80, 0.90)`。退化时(`‖Δv‖<1e-9`)回退 `n̂=(-1,0,0)`,极少触发。

### 1.2 ⭐ 击球点 ↔ Base 位置(关键不变量)

球拍要打到 `p_hit_world`,但机器人不能瞬移,底盘该站哪?**用 demo 在击球帧的「拍↔骨盆」相对偏移,在 base frame 下复现**:

```python
# expert_offset_base[swing] 是一次性预处理:demo 击球帧 blade−pelvis 在 base frame 的 xy 偏移
offsets_world  = R(yaw_robot) @ expert_offset_base[swing_type]   # base→world
p_base_xy_world = p_hit_world.xy − offsets_world                  # base 站位 = 击球点 − 偏移
```

见 [mdp/commands.py `_compute_base_target`](mdp/commands.py) 与 [mdp/motion_loader.py](mdp/motion_loader.py)(`expert_offset_base` 在 clip 载入时算好)。

- **为什么 base frame**:demo 挥拍的转身靠 `waist_yaw` 关节角,骨盆基本朝 +X,所以 base frame≈world frame,offset 量级稳定,正反手统一。
- **为什么用 robot 当前 yaw 旋回 world**:启动有 ±10° yaw noise,训练中 base 会自由旋转,必须用当前 yaw 旋,否则 offset 与姿态错位。
- **关键不变量**:**击球点由球物理决定;base 站位是从击球点反推的从动量。** 策略学的是「在合理时间内把 base 移到这个目标 + 把拍准确送到击球点」。

---

## 2. Scene 完整规范 [robots/g1_23dof/hitter/hitter_env_cfg.py](robots/g1_23dof/hitter/hitter_env_cfg.py)

| # | 项 | 取值 | 设计原因 |
|---|---|---|---|
| 1 | `num_envs` / `env_spacing` | 4096 / 4.0 m | IsaacLab 标准并行 |
| 2 | `terrain` | plane,friction 1.0/1.0,`combine_mode="multiply"` | 室内硬地 |
| 3 | `robot` | `ROBOT_CFG`(= 当前选用的 PD 版本,见 §3) + paddle URDF | 右腕固连一块 `right_paddle_blade` |
| 4 | `table` | Cuboid 2.74×1.525×0.05,`kinematic_enabled=True`,friction 0.9/0.8,restitution 0.2,**init z=−10** | kinematic 不被撞动;初始**沉到地下**(见 table-guard §10.7) |
| 5 | `light` / `sky_light` | DistantLight 3000 / DomeLight 1000 | 标准 |
| 6 | `contact_forces` | `ContactSensor`,覆盖 `Robot/.*`,`force_threshold=10`,`track_air_time=True` | 服务 feet_slide / undesired_contacts / hard_contact |
| 7 | `robot_table_contact` | `ContactSensor`,`filter_prim_paths_expr=[Table]` | 只算 robot↔Table,服务桌接触罚 / 撞桌终止 |

**双 sensor 分工**:`contact_forces` 用 `body_names` 正则过滤(脚滑、非法接触、硬接触终止);`robot_table_contact` 过滤到 Table(拍撞桌、身体撞桌、撞桌终止)。

---

## 3. ⭐ 机器人与执行器 — 3 套 PD 版本(当前最优候选)

三套配置都在 [assets/robots/unitree.py](../../assets/robots/unitree.py),**共享同一套 env_cfg**(reward/curriculum/obs 完全一致),只差腿/腰/臂的 kp/kd。当前 `hitter_env_cfg.py` 默认 import 的是 **low_PD**(big-PD 那行被注释)。

### 3.0 执行器物理(理解三套差异的基础)

`ImplicitActuator`:`τ = kp·(q_des − q) − kd·q̇`,且 `τ` 被 clamp 到 `effort_limit_sim`。

- **动作幅度** `action_scale = 0.25 · effort_limit_sim / kp` → **只与 kp 有关,与 kd 无关**。所以改 kd **不改** action/obs 维度与尺度,可以做干净的 A/B。
- **关节最大速度** `dq_max ≈ effort / kd` → kd 越大,关节越慢(挥拍越被拖住)。
- **阻尼比** `ζ = kd / (2·√(kp·I))` → kd 太小欠阻尼(回弹/振荡),kd 太大过阻尼(迟钝)。

MIMIC 基线增益常量(来自电机参数,10 Hz 自然频率、阻尼比 2.0):
`STIFFNESS_5020≈14.25`、`_7520_14≈40.18`、`_7520_22≈99.10`;`DAMPING_5020≈0.907`、`_7520_14≈2.558`、`_7520_22≈6.309`。

### 3.1 三套配置对照

| 关节组 | (A) 软 MIMIC 基线 | (B) big-PD(腿硬) | (C) low-PD(腿硬·kd减半,**当前默认**) |
|---|---|---|---|
| hip pitch/roll/yaw `kp` | 40.18 / 99.10 / 40.18 | **100 / 100 / 100** | 100 / 100 / 100 |
| knee `kp` | 99.10 | **150** | 150 |
| hip `kd` | 2.56 / 6.31 / 2.56 | **2.0 / 2.0 / 2.0** | **1.0 / 1.0 / 1.0** |
| knee `kd` | 6.31 | **4.0** | **2.0** |
| feet(ankle)`kp/kd` | 28.5 / 1.81 | **40 / 2.0** | 40 / **1.0** |
| waist_yaw `kp/kd` | 40.18 / 2.56 | 40.18 / 2.56(软,不动) | 40.18 / 2.56(软,不动) |
| arms `kp/kd` | 14.25 / 0.907 | 14.25 / 0.907(软,不动) | 14.25 / 0.907(软,不动) |
| `action_scale` | (软基线) | 重算(kp 变) | **与 B 完全相同**(kp 不变) |
| 符号 | `UNITREE_G1_23DOF_MIMIC_CFG` | `UNITREE_G1_23DOF_PADDLE_MIMIC_CFG` | `..._PADDLE_MIMIC_CFG_low_PD` |

> 切换:在 [hitter_env_cfg.py:20-22](robots/g1_23dof/hitter/hitter_env_cfg.py#L20-L22) 改 `ROBOT_CFG` 的 import 即可;`UNITREE_G1_23DOF_PADDLE_MIMIC_ACTION_SCALE` 那行 B/C 通用(kp 一致)。

### 3.2 为什么是这三套(sim2real 腿抖动的演化)

- **(A) 软基线的问题**:早期 pingpong 策略用 MIMIC 软腿(locomotion 血统)训练,**sim2sim 正常,但 sim2real 时腿剧烈抖动**。根因:软腿(kp 低)+ 真机执行延迟 + 传感噪声,闭环增益不足以镇住高频;手那时反而能摆到 demo 那一帧,说明**不是手的问题,是支撑腿**。
  - ⚠️ 教训:`support_override`(只在部署端换硬增益)在 SIM 里也会抖——因为训练/部署动力学不匹配。**硬增益必须烘进训练**,不能只在部署换。

- **(B) big-PD = 只把腿(+脚)抬到 locomotion 级硬增益,腰和臂保持软**:
  - 腿硬(kp 100/100/150,脚 40)→ 真机支撑稳;
  - **腰保持软**:`waist_yaw` 是挥拍/拍面关节(扭腰发力 + 定拍面),硬腰(曾试 kp 200)会把它锁死;
  - **臂保持软**:⚠️ 曾把臂也调硬(kp 80 / kd 3),结果**反噬**——`dq_max = effort/kd` 从 ~28 掉到 ~8 rad/s,挥拍够不到 `v_racket_hat`,`vel_fail` 从 0.08 暴涨到 0.46、`hsr` 0.67→0.44。快速挥拍需要**低臂阻尼**,软臂本来就把拍面+速度跟得很好。所以**只有腿需要变硬**。
  - 必须 **从头重训**(软增益策略不能直接部署硬增益)。

- **(C) low-PD = 在 B 基础上把腿+脚 kd 减半(kp 不变)**:
  - 低 kd = 腿更「活」(挥拍/迈步时不被阻尼拖住)+ **对真机关节速度噪声的放大更小**(若 big-PD 真机仍有高频嗡,这版应更安静);
  - 代价 = 更欠阻尼 → 若 sim 里看到**腿低频回弹 / EL 下掉 / 站不稳**,说明 kd 减太多,往回加(如 hip 1.5 / knee 3 / ankle 1.5);
  - `action_scale` 与 B **完全一致**(只看 kp),obs/动作维度不变,是干净的 kd A/B。

> 一句话:**腿抖 = 整体增益(尤其 kp)偏软、被真机延迟/噪声激出来的**;修法是把**腿**烘成硬增益(B),再用(C)在「响应快 vs 欠阻尼」之间找平衡。腰、臂始终保持软以保住挥拍速度与拍面自由度。

---

## 4. Observation 完整规范 (Actor=92, Critic=212)

非对称 Actor-Critic:Actor 只看「可部署 + 带噪」的量;Critic 看全部 clean + 特权信息。

### 4.1 Actor (PolicyCfg) — 12 项,共 92 维

| # | obs | 维度 | 来源 / 公式 | 备注 |
|---|---|:---:|---|---|
| 1 | `base_ang_vel` | 3 | `DelayedObservation(base_ang_vel_imu)` | 过 IMU 偏置 + 通信延迟 |
| 2 | `projected_gravity` | 3 | `DelayedObservation(projected_gravity_imu)` | 同上 |
| 3 | `base_yaw` | 2 | `[cos yaw, sin yaw]`,过 IMU+延迟 | 朝向编码 |
| 4 | `base_err` | 2 | `p_base_xy_world(+noise_base) − root.xy` | 该往哪走 |
| 5 | `hit_pos` | 3 | `R_baseᵀ·(p_hit_world(+noise_p) − root)` | 击球点(base 系) |
| 6 | `racket_vel` | 3 | `v_racket_hat_world(+noise_v)` | 目标拍速(world) |
| 7 | `t_to_hit` | 1 | `t_to_hit(+noise_t)` | 倒计时 |
| 8 | **`active_face`** | 3 | `R_baseᵀ·(sign·n_blade_w)`,`sign=1−2·swing` | **v64 新增**,见下 |
| 9 | **`target_normal`** | 3 | `R_baseᵀ·n_target_world` | **v64 新增**,见下 |
| 10 | `joint_pos` | 23 | `DelayedObservation(joint_pos_rel)` | |
| 11 | `joint_vel` | 23 | `DelayedObservation(joint_vel_rel)` | |
| 12 | `last_action` | 23 | `last_action`(不延迟) | |

**⭐ #8/#9 为什么加(v64,直接针对拍面瓶颈)**:Actor 原来只能看到 `racket_vel`,**看不到自己当前的拍面朝向**,必须从原始关节角经 FK 自己推——这极可能就是拍面一直学不好的原因。`active_face` 是**当前拍面法向(按 swing 取符号,正手 +n_blade、反手 −n_blade)**,正是 `goal_orientation` 打分的那个向量;`target_normal` 是目标法向。两者都在 base 系,任务统一成「把 `active_face` 指向 `target_normal`」。
- ⚠️ 为什么是「带符号的拍面反馈」而不是 swing_type 标量:曾把 `swing_type` 当 ±1 标量喂 Actor(run 11-10-06),导致**模式坍缩**(forehand=0.97、EL 卡 125、cos_sim=−0.25)。`active_face` 是**对当前拍的反馈**(不是命令里的 swing 标签),不会重新引入那个坍缩。可部署(拍刚性固连腕,FK 可算)。

### 4.2 Critic (CriticCfg) — Actor 同款(clean,无噪无延迟)+ 特权信息

额外:`base_lin_vel`(3)、`ref_body_state`(跟踪 10 个 body 的 pos+quat = 10×7 = 70)、`time_left`(1)、`ref_joint_state`(ref clip 的 joint pos+vel = 23+23 = 46)。
**Critic = 92(同 Actor 项,clean) + 3 + 70 + 1 + 46 = 212**。

### 4.3 跟踪 body 集合(`tracked_body_names`,10 个)

`torso_link` + 左臂 5(shoulder pitch/roll/yaw、elbow、wrist_roll_rubber_hand)+ **右臂 4**(shoulder pitch/roll/**yaw**、**elbow**)。
- **为什么右臂加回 shoulder_yaw + elbow**(原来只 pitch/roll):`imitation_body_pos` **位置锚定**右臂姿态 → 挡住「正手用反手姿势 + 翻腕」的退化(run 18-26-56);
- **为什么不含右腕/拍**:位置锚定 ≠ 朝向锁定,**拍面自由度留给 wrist_roll**,腕和拍故意不跟踪。

### 4.4 模仿关节集合(`imitation_joint_names`,11 个全上半身)

`waist_yaw` + 左臂 5 + 右臂 5(shoulder pitch/roll/yaw、elbow、wrist_roll)。
- ⚠️ 演化:V1 把右臂 distal(shoulder_yaw/elbow/wrist_roll)从模仿里**砍掉**留给拍面自由 → 出现「正手用反手姿势」。v63 起**加回全 11 关节**模仿 demo 正手姿势;但拍面不会被锁死,因为 `task_phase` + `imit_orient_anneal` 会在 Phase 2 把这几个「脸部关节」的模仿权重**逐步降权**(见 §10.5),让 `goal_orientation` 把拍面顶过 demo 的 ~0.80 上限。

### 4.5 强约束(必须遵守)
- `swing_type` **不进任何 obs**(Actor/Critic 都不进)——靠 `cmd.swing_type` 路由 ref clip。
- 上游物理(`v_ball_in`、`target_land`、`flight_time`、`paddle_cor`、`n_target`、`v_ball_out`、各 `noise_*`)**全不进 obs**。
- Actor 的 4 项 cmd 字段(base_err/hit_pos/racket_vel/t_to_hit)走 **per-swing 冻结噪声**(§9.3);Critic 全 clean。
- IMU 偏置/通信延迟只 wrap Actor 的传感通道;`last_action` 不 wrap。

---

## 5. Action 完整规范

| 字段 | 维度 | 公式 |
|---|:---:|---|
| `JointPositionAction` | **23** | `q_target[j] = q_default[j] + scale[j]·a[j]` |

- `scale` = `UNITREE_G1_23DOF_PADDLE_MIMIC_ACTION_SCALE`(per-joint `0.25·effort/kp`,B/C 通用)。
- `use_default_offset=True`;控制频率 = `sim.dt 0.005 × decimation 4` → **50 Hz**(论文)。
- **`clip={".*": (-10, 10)}`**:⚠️ `action_l2/action_rate_l2` 是无界 `Σaction²`,actor 一旦发散(resume 课程跳变 iter 60002、v61 iter~35k)惩罚冲到 −1e22 → value 爆炸 → 训练崩。正常动作 ~±3,±10 永远不咬真实挥拍,但把发散惩罚封到有限可恢复值。配合 §8 的 `*_bounded` 惩罚函数双保险。

---

## 6. Command 完整规范 [mdp/commands.py](mdp/commands.py)

`PingpongCommand` 维护每个 env 的击球任务。重采样时机:`t_to_hit ≤ −t_post_swing`(打完 + 走完 follow-through)→ 采新一拍;episode reset 也采首拍。

### 6.1 关键 cfg(当前值)

| 字段 | 值 | 含义 / 原因 |
|---|---|---|
| `forward/backward_motion_file` | `new_3/.../forward_001_wristfix_rotated` / `backward_001_rotated` | ⚠️ 正手 clip 用 wristfix 版:击球肘角 ~121°/伸展 0.87,**脱离奇异**(原 forward_003 伸展 0.987 近奇异,逼策略在「拍面准 vs 拍速快」之间二选一) |
| `hit_x` | **0.40**(env_cfg 显式覆盖) | 虚拟击球平面 = 球桌**近边**(world x=1.77−1.37=0.40)。配合 `reset_root_pos.x=−0.138`(机器人 ~0.54 m 外),demo 伸展刚好落在边沿,**零强迫前后位移**;pre_strike 回投点 `p_hit−v̂·t` 落在 x<0.40 = 桌外,低球也不穿桌 |
| `reset_root_pos` | `(−0.138, 0, 0.74)` | 见上;0.74 是 pelvis 标称高度 |
| `hit_z_range` | `(0.95, 1.25)` | 上限 1.15→1.25 对齐新 clip 高接触点(fh~1.16/bh~1.26),否则命令逼策略低于 demo |
| `v_in_mag_range` | `(1.5, 2.0)` → 课程到 `(1.5, 4.0)` | 来球速度,先慢后快 |
| `target_land` | `(2.45, 0, 0.78)` | 对方半台正中(桌中 1.77→远边 3.14 的中点)+ 桌面 0.76 + 球半径 0.02 |
| `flight_time_range` | `(0.30, 0.65)` | Eq.5 输入 |
| `paddle_cor_range` | `(0.80, 0.90)` | 球-橡胶 COR,DR 模拟老化 |
| `swing_p_forehand` | **0.50** | 正反手 Bernoulli 50:50(论文 task input)。⚠️ v60 的 90:10→50:50 warmup 已删——3-phase 课程接管「单任务→双任务」 |
| `hit_y_world_cap` | `0.45` init → `1.00` max | 世界系 y 采样半宽(env-local),课程驱动。init 0.45 > max(|fh_y|,|bh_y|)≈0.40 才能覆盖两个 demo 点 |
| `forehand_y_safety_clamp` | `0.40` | 正手 reach 的右臂奇异安全夹;`forehand_y` 被夹到 ±0.40 |
| `reset_yaw_noise` | `±10°` | 防 yaw 锁死 |
| `strike_window` | `0.10` → `0.01` | 击球判定窗(课程收紧),见 §10.4 |
| `success_pos/vel/ori_thresh` | `0.15 m / 1.0 m·s⁻¹ / 0.25` | pos/vel/ori「成功」判据(固定阈,不随 σ 漂) |
| `success_ori_cos_dist_thresh_backhand` | **0.20** | ⚠️ v64:反手在共享 0.25 下用「凑合的拍面」也能算成功 → imit_w 降后拍面退化。反手单独收紧到 0.20(要求 signed-cos>0.80),把好拍面变成成功的**前提** |

数据驱动量(载入 clip 时 [motion_loader.py](mdp/motion_loader.py) 自动算,换 npz 自动重算):`expert_offset_base[2,2]`(正反手 拍↔骨盆 base-xy 偏移)、`y_mid_base`(正反手分界)、`_swing_y_sign`(forehand_y>backhand_y → +1,当前 23dof 为 −1,正手在 −y 侧)、`t_post_swing_fixed`(取两 clip 较长的 follow-through)。

### 6.2 `_sample_new_swing` — 世界系采样 + 锚定 divider(v61)

每拍按顺序(节选,完整见 [commands.py:463](mdp/commands.py#L463)):

1. **Bernoulli swing_target**(50:50),**提前写入** `self.swing_type`(保证 RSI 选对 clip);采 `hit_z`、来球速度(`yaw=π±40°`、`pitch=±75°`)、`target_land`、`flight_time`、`paddle_cor`。
2. **RSI 覆写 root_quat**(若 reset):取 clip 随机帧的 pelvis_yaw + `±10°` noise 写 root quat(在算 hit_y 前,保证用的是策略开局真实 yaw)。
3. **世界系采样 hit_y**:`hit_x_world = env.x + hit_x`(固定);`hit_y_world ∈ [env.y − cap, env.y + cap]`。
4. **⭐ divider 锚定 env_origin**:`divider_world = env.y + y_mid_base`(**不跟随 root**)。正手区 = divider 的固定一侧。
   - ⚠️ 这是反作弊核心:v60 让 divider 跟随 `root.y` 时,策略学会**把 base 移到一侧、把正手命令变成 cross-body 反手伸手**(用户实测)。锚定后 forehand 永远是 `world.y < env.y + y_mid_base`,与 base 漂移无关——机器人**必须站回 env_origin 附近**才能合法击球,`goal_base` 自然把它拉回来。
5. **边界 override**:若目标半区与 cap 交集为空,强制翻到另一半(再由 `goal_base` 把机器人拉回界内);记 `_dead_zone_count`。
6. 直接在有效半区采 `hit_y_world` → 写 `p_hit_world`(绝对世界系)。
7. `_solve_paddle_target`(Eq.5/Eq.6)→ `v_racket_hat_world`、`n_target_world`。
8. `_compute_base_target` → `p_base_xy_world`;采时间字段(`t_pre_initial`、`t_to_hit`);写 RSI 关节态;冻结 per-swing noise;更新 ref_state。

> ⚠️ v62 已**删除** v58 的 `_compute_swing_type` 后置重分类:swing 标签现在是采样时**构造保证**的(hit_y 直接在 swing_target 的半区里采),不再事后翻分类。

### 6.3 `_blade_target_cosine` — signed 拍面(reward/metric/success 共用一个定义)

```python
sign = 1 − 2·swing_type          # forehand=+1, backhand=−1
n_blade = quat_apply(blade_quat_w, BLADE_NORMAL_LOCAL)   # BLADE_NORMAL_LOCAL=(0,−1,0)
cos = (sign · (n_blade · n_target)).clamp(−1, 1)
```
- `BLADE_NORMAL_LOCAL=(0,−1,0)`:⚠️ URDF 把拍绕 X 转 −135°,实测局部 −Y 在击球帧指向球台 = 正手面。早期写成 `(0,1,0)` 导致「僵硬手臂 + 翻腕用反手面打正手」的姿态(V1 旧 best)。
- ⚠️ 为什么 signed 而不是 `|dot|`:对称 `|dot|` 让任意朝向都满足 ori_ok,策略干脆**把拍停住不挥**,`pos_fail` 卡 ~0.99(run 22-46-22)。signed 才给出「该哪个面对准」的挥拍梯度。

---

## 7. Reward 完整规范 [mdp/rewards.py](mdp/rewards.py)

权重分三类。**注意**:很多权重的 env_cfg 初值是「死值」,运行时被课程覆盖(下表标注「← 课程」)。

### 7.1 模仿 r_i(权重由 `task_phase` 按相位写:Phase split = jp 0.40 / bp 0.50 / jv 0.10 × `w_i`,`w_i` = 0.10→1.00→0.30)

| 子项 | 公式 | gate | 原因 |
|---|---|---|---|
| `imitation_joint_pos` | `exp(−2·Σⱼ wⱼ(qⱼ−q̂ⱼ)²)` | `gate_pre_strike=False`,`post_strike_scale=1.0`,`post_strike_delay=0.04` | 全程跟踪关节角;击球后留 ~2 帧接触缓冲,之后**满权重**跟 demo 收拍回中。`wⱼ` = `imit_joint_weights`(per-joint,被脸部降权课程改) |
| `imitation_joint_vel` | `exp(−0.1·Σⱼ wⱼ(q̇ⱼ−q̇̂ⱼ)²)` | 同上 | |
| `imitation_body_pos` | `exp(−10·Σ_b‖p_rel−p̂_rel‖²)`(相对 pelvis 锚) | `gate_pre_strike=False` | body_dominant split 占比最大,位置锚定上半身姿态 |

⚠️ 教训(为什么 follow-through 要 delay):窗口期结束后球还没飞远,策略会发现「微调拍面比继续模仿后段 reward 更高」→ 击球后不收拍。`post_strike_delay=0.04` 让击球后 ~2 帧才恢复满权重模仿,既保留接触缓冲又把收拍锚回 demo。
⚠️ 教训(为什么 body_pos 现在不 gate):早期 `body_dominant` split 让 body_pos(0.281/step)在击球帧压过 `goal_orientation`(+0.0002),把拍面拖向 demo 轨迹均值 → cos_sim 崩到 −0.76。当时的解是「body_pos 只在 pre-strike 开」;v62 后改用 **3-phase 课程**(Phase0/1 全关 goal_*),从根上消除了这个梯度抢夺,所以 body_pos 又可以全程开。

### 7.2 任务 goal r_g

| 子项 | 公式 | env_cfg 权重 | std | gate |
|---|---|---|---|---|
| `goal_position` | `exp(−‖p_blade−p_hit‖²_base/σ²)` | 2.0 ← 窗口课程 ramp 到 12 | `sigma_g_pos` 0.30→0.06(σ latch) | sparse `\|t_to_hit\|≤strike_window` |
| `goal_velocity` | **Gaussian** `exp(−‖Δv‖²_base/σ²)` | 2.0 ← ramp 到 12 | 1.50→0.50 | sparse 同上 |
| `goal_orientation` | signed `exp(−(1−sign·n_blade·n_target)²/σ²)` | 0.5 ← Phase2 入口重置 **4.0**,窗口 ramp `w_ori` 到 10 | 0.40→0.15 | sparse 同上 |
| `goal_position_pre_strike` | 沿 `v̂` 线性回投 `p_hit−v̂·t` 的位置 | 1.0 | 0.2 | dense `0<t_to_hit<ramp_time(0.2)` |
| `goal_velocity_pre_strike` | 目标 = `ramp·v̂`(逐渐拉到全速) | 1.0 | 0.6 | dense 同上 |
| `goal_orientation_pre_strike` | signed cos 距离 | 0.5 ← `task_phase` 按相位写 (0 / 1.8 / 2.5) | 0.4 | dense 同上 |
| `goal_base` | `exp(−‖root.xy − p_base_xy_world‖²/σ²)` | **1.5** ← ramp(0.5→1.5) | 0.3 | dense `t_to_hit>0`(打完关) |
| `goal_base_orientation` | `exp(−yaw²/σ²)` | 0.3 | 0.3 | dense `t_to_hit>0` |

⚠️ 为什么 `goal_velocity` 改 Gaussian(v62):原 Laplacian `exp(−‖Δv‖/σ)`,σ=0.45 时 `‖Δv‖=2` → reward≈0.001,**中等误差区梯度近零,策略无法从 2 学到 1**。Gaussian σ=1.5 时同样 `‖Δv‖=2` → 0.17(大 170×),梯度可感知。课程再把 σ 收到 0.50(论文精度)。
⚠️ 为什么 pre_strike 存在 + 后来要退火:窗口只 1~2 帧太稀疏,pre_strike 在击球前 `ramp_time` 内给**稠密**的位置/速度/拍面引导(教会腰+base 协调)。但它是拐杖——到瓶颈期由 `prestrike_ramp_anneal`(§10.6)逐步缩 `ramp_time` 直至关闭,逼策略靠真正的击球瞬间奖励。
⚠️ 为什么 `goal_base` 抬到 1.5:逼策略更主动**横移到位**,减少「横身用手硬够」;由 `_GOAL_BASE_RAMP` 从 0.5 平滑 ramp 到 env_cfg 的 1.5(站稳前不要全力横移)。
⚠️ 为什么加 `goal_base_orientation`:锚 base yaw 朝 +X,逼策略**横向平移**覆盖左右击球点,而不是转身侧covering(配合锚定 divider 一起反 cheat)。

### 7.3 正则 r_r

| 项 | 权重 | 原因 |
|---|---|---|
| `alive` | +0.04 | 防早结束 |
| `action_rate_l2` (bounded) | −0.001 | 平滑;击球需快变,弱罚。`*_bounded`:每关节平方差 clamp `max=4`,只封发散不咬真实挥拍 |
| `action_l2` (bounded) | −0.0005 | 同;clamp `max=25`(封 \|action\|>5) |
| `joint_torque` / `joint_acc` | −3e-6 / −1e-7 | 力矩/加速度 L2 |
| `energy` | −2e-5 | 节能软正则(locomotion 默认) |
| `joint_limit` | −5.0 | 越界软罚 |
| `pelvis_orientation` | −1.0 | base 倾斜 `proj_g_xy²` |
| `pelvis_ang_vel_xy` | −0.05 | 只罚 base roll/pitch 角速率(**yaw 自由**,挥拍要转);治击球后反作用力矩晃倒 |
| `pelvis_lin_vel_z` | −1.5 | 防移动时跳跃(locomotion 默认) |
| `pelvis_height` | −5.0(target 0.74) | base 高度 |
| `feet_slide` | −0.3 | 防拖脚(仅 ankle_roll 接触时罚水平速度) |
| **`leg_joint_deviation`** | −0.5(Phase2 −0.3)← `task_phase` | **只罚 hip_roll/yaw 偏离默认**(hip_pitch/knee/ankle 自由,保留迈步/下蹲);治腿向侧方叉开 |
| **`feet_contact_no_strike`** | +0.20(Phase2 +0.10)← `task_phase` | **待命时(`t_to_hit≤0`)**奖双脚着地;治击球后单脚站。门控:approach 时(`t_to_hit>0`)关闭,允许迈步横移 |
| **`feet_distance_no_strike`** | −0.5(Phase2 −0.3)← `task_phase` | 待命时罚站距偏离 0.20 m;**非对称**:交叉腿(过窄)全罚,叉开(过宽)×0.3(宽站有助低球下蹲) |
| `undesired_contacts` | −1.0 | 非足/非腕胶手/非拍 body 触地软罚 |
| `paddle_table_contact` | 0.0 ← table-guard ramp 到 −10 | 拍撞桌(stage-aware) |
| `body_table_contact` | 0.0 ← ramp 到 −1 | 身体撞桌(stage-aware) |

⚠️ 三条腿正则(`leg_joint_deviation` / `feet_contact_no_strike` / `feet_distance_no_strike`)为什么加:pingpong 的模仿集是**纯上半身**,腿完全无约束 → 击球后出现「抬一条腿当配重 / 单脚站 / 前后晃」。这三条从 locomotion 移植,门控到 pingpong 的 `t_to_hit`(「不接近击球」= 待命),Phase2 略弱以不挡横移/下蹲。权重纳入 `task_phase` 按相位写(env_cfg 里的初值是死值)。

---

## 8. Termination 完整规范 [mdp/terminations.py](mdp/terminations.py)

| # | 项 | 触发 | time_out | 原因 |
|---|---|---|:---:|---|
| 1 | `time_out` | `t ≥ 10 s`(500 步) | ✓ | 自然结束,GAE 用 `V(s_T)` bootstrap |
| 2 | `base_height` | pelvis < 0.30 m | ✗ | 摔地 |
| 3 | `bad_orientation` | `limit_angle=0.8 rad`(≈46°) | ✗ | 倾倒 |
| 4 | `hard_contact` | `pelvis/torso/head/.*_hip_pitch_link` 触地力 > 1.0 N | ✗ | 严重摔倒 |
| 5 | `non_paddle_table_stuck` | 非拍 body 持续撞桌 ≥ 0.3 s(力>3 N) | ✗ | 撞桌作弊;**stage-aware**:桌隐藏时(`_pingpong_table_active=False`)短路返回 zeros |

**强约束**:robot↔table 接触**永远只走 reward 软罚**,不作终止(除上面持续撞桌的 #5)。否则一次探索碰撞就丢失整段击球学习信号。

---

## 9. Events / RSI / Domain Randomization [mdp/events.py](mdp/events.py)

### 9.1 startup(一次冻结)

| 项 | 范围 | 原因 |
|---|---|---|
| `physics_material` | static `[0.3,1.6]`、dynamic `[0.3,1.2]`、restitution `[0,0.5]`,64 buckets | 地面摩擦 DR |
| `add_link_mass` | scale `[0.9,1.1]` | 连杆质量 ±10% |
| `randomize_joint_friction` | scale `[0.5,1.5]` | 关节摩擦 |
| `randomize_joint_damping` (`randomize_actuator_gains`) | scale `[0.7,1.3]` | 阻尼 ±30% |
| `randomize_imu_offset`(自定义) | gauss `σ=2°` | IMU 标定误差,写 `_pingpong_imu_offset_quat` |
| `randomize_comm_delay`(自定义) | uniform `{0,1}` step(0–20 ms) | 通信延迟,写 `_pingpong_obs_delay_steps` |
| `add_joint_default_pos` | add `[-0.01,0.01]` | 关节零位标定误差 |
| `base_com` | torso_link COM x`±0.025`/y`±0.05`/z`±0.05` | 质心不确定 |

IMU/延迟用 **startup 一次冻结**(硬件上装机后基本恒定);要 per-step 抖动把 `mode` 改 `interval` 即可,wrapper 不动。

### 9.2 interval / reset

- `push_robot`(interval `1–3 s`):base lin vel `±0.5 m/s`(z `±0.2`)、ang vel(roll/pitch `±0.52`、yaw `±0.78`),鲁棒性扰动。
- `reset_table`(reset):按 table-guard stage 把桌放到隐藏(z=−10)或激活(z=0.735)。

### 9.3 RSI(Reference State Initialization)三步约束

1. reset root 到 env_origin + `reset_root_pos`,yaw 加 `±10°` noise;
2. 用 `clip[swing_type]` 的随机帧写关节角 + 关节速度;
3. **用该帧 pelvis_yaw 覆写 root_quat**(否则 `n_blade` 整体被旋 60°+,reset 瞬间拍面就和 `n_target` 错位)。

### 9.4 cmd noise per-swing 冻结(非对称 AC 的关键契约)

每次重采样**一次性**采出 `noise_p/v/base/t`(`clip(gauss(0,σ), ±3σ)`),整拍不变。**只注入 Actor obs**;Critic、所有 reward、strike gate、重采样边界**一律用 clean cmd**。
- 终值 σ:`σ_t=0.005 s`(hsr≥50% 开)、`σ_p=0.005 m`、`σ_v=0.05 m/s`、`σ_base=0.015 m`(后三者 hsr≥75% 开)。当前 env_cfg 课程 `enable_noise=False`,即默认不加噪(等基本功能稳了再开)。

---

## 10. Curriculum 完整规范 [mdp/curriculums.py](mdp/curriculums.py)

**注册顺序敏感**(CurriculumCfg 内):`imit_anneal` → `pingpong` → `task_phase` → `table_guard` → `imit_orient_anneal` → `prestrike_ramp_anneal`。理由:后面的 term 读前面写的 EMA/相位,并**覆盖**它们的权重决定。

### 10.0 三种 σ(防混淆)

| σ 类型 | 含义 | 单调方向 |
|---|---|---|
| reward kernel σ | `exp(−d²/σ²)` 衰减半径 | 只**收紧**(任务变难) |
| 采样 σ(cmd noise) | `gauss(0,σ²)` 噪声幅度 | 只**升**(更鲁棒) |
| uniform 区间 | hit_y/z、v_in 范围 | 只**扩** |

### 10.1 ⭐ 3-phase 任务课程 `update_task_phase`(反作弊主干,单向阀)

把训练分三个**单向阀**相位,每个只学一件事;**只升不降**(monotone latch):

```
Phase 0 (stand):  imit_w=0.10,goal_* 全 0           → 先学站(从 RSI 中段姿态)
   ↓ EL_ema ≥ 350
Phase 1 (imit):   imit_w=1.00,goal_* 全 0           → 充分模仿正反手 demo
   ↓ EL_ema ≥ 450 且 进入 Phase1 已 ≥ phase_1_min_iters(2000)
Phase 2 (strike): imit_w=0.30,goal_* 一次性重置到 baseline → 正式击球任务
```

- **imit split**:jp 0.40 / bp 0.50 / jv 0.10 × `w_i`。
- **Phase 2 入口一次性重置** goal_* 权重到 baseline(pos 2.0 / pos_pre 1.0 / vel 2.0 / vel_pre 1.0 / **ori 4.0** / ori_pre 0.5)。⚠️ 必须重置:否则 `cos_sim_ratchet_freeze`(cos_sim<0.45 冻结窗口课程)会让 goal_* 永远卡在 0(「没 ori 信号→cos 低→冻结→没信号」死锁)。`ori` baseline 抬到 4.0 是因为进 Phase2 时拍面会从 0.92 跌到 0.85(strike-instant ori 权重只 0.5 vs pos+vel 4)。
- **`phase_1_min_iters=2000`**:⚠️ 防 EL 暴涨跳过 Phase1(run 15-41-23:EL 339→448 仅 100 iter),让重模仿阶段真的教会正反手区分。
- **腿正则按相位写**:`leg_reg_phase_weights`(三条腿正则的 Phase0/1/2 权重)。
- **⭐ 提前学拍面 + Phase1 内 ramp(posture-first)**:
  - `face_prestrike_phase_weights=(0.0, 1.8, 2.5)`:**Phase 1 就打开稠密拍面信号**(`goal_orientation_pre_strike`),而 pos/vel 还关着 → 拍面**无竞争**地先学(不和 pos/vel 抢梯度)。
  - `face_imit_phase_weights`:脸部关节(waist_yaw、wrist_roll 纯拍面 → 低 0.3;shoulder_yaw、elbow 兼挥拍+拍面 → 0.6)在 Phase1/2 的模仿权重。
  - **Phase1 内 posture-first ramp**(`face_p1_ramp_frac=(0.4,0.8)`、`face_*_p1_early`):Phase1 早期 = 高脸部模仿(1.0)+ 低拍面奖励(0.2)→ 先学清楚正反手**姿势**;中后期把拍面奖励 ramp 上去,在已学到的姿势 basin 里**精修**拍面(退化的「正手姿势+翻腕」拿不到额外拍面奖励 → 没梯度往那走)。

### 10.2 模仿退火 `update_imitation_weight`(metric 模式,现仅作 EMA 源)

`schedule="metric"`,`split="body_dominant"`,`w_i_values=(0.5,0.3,0.15)`,phase_thresholds 两段(放宽后:① hsr0.30/pos0.40/vel0.40/ori0.55/EL250;② hsr0.50/pos0.70/vel0.65/ori0.75/EL400)。
- 它维护 `_EP_LENGTH_EMA` 和四个成功率 EMA(hsr/pos/vel/ori),`task_phase` 和 `pingpong` 都读它们。
- ⚠️ 为什么 metric 而非 iter:纯 iter 退火会在「33k iter 还没站起来」时照样 iter 8000 把 w_i 砍到 0.15,杀掉唯一正向 shaping。metric 模式让模仿一直强到策略证明能力。⚠️ 为什么放宽 vel/ori 阈值:高 w_i → 策略抄 demo 关节角 → demo 的 vel 目标≠球物理 vel 目标 → vel_fail 高 → 相位卡死(imit feedback trap),放宽后相位能进、w_i 降、goal_velocity 相对增强。
- **注意**:`task_phase` 会**覆盖** imit 权重,所以实际相位由 `update_task_phase` 主导,本项主要作 EMA 提供者。

### 10.3 reward-shape σ 课程 `_REWARD_SHAPE_TIERS`(7 档,4-EMA AND 门控)

7 档收紧 `(σ_pos, σ_vel, σ_ori)`:tier0 `(0.30, 1.50, 0.40)` → tier6 `(0.06, 0.50, 0.15)`。每档要 **hsr/pos/vel/ori 四个 EMA 同时**过阈才升。
- ⚠️ hsr_ema 是冷启动门:pos/vel/ori EMA 从首次观测 init 成 1.0(还没击球时 fail=0),没 hsr 门控会在 iter~50 直接跳 tier4。
- **v64 σ 单调 latch `_SIGMA_LATCH`**:σ 一旦收紧不再放松(只 `min`)。⚠️ 破解 tier4↔5 极限环——ori_success 在 tier5 阈值(0.80=反手 ori 门)附近抖,σ 每次回弹拍面就顶不过 0.80。
- **σ_ori floor**(`PINGPONG_SIGMA_ORI_FLOOR`,默认 0.20):σ-ease,缓解「拍面被压到 0.83 但 hsr 掉到 0.69」的过压。

### 10.4 窗口课程 `_WINDOW_CURRICULUM_TIERS`(5 档,收窗 + 抬权重)

`strike_window` 0.10→0.01 s 的同时把 `(w_pos, w_vel, w_ori)` 从 `(2,2,4)` ratchet 到 `(12,12,10)`。
- 为什么耦合:窗口内积分奖励 ≈ `weight×(window/dt)`,收窗时同步抬权重保持 PPO 信号稳定。
- ⚠️ `w_ori` 抬到 4→10(原 0.5):进 Phase2 后要**顶住** Phase1 学到的拍面对抗 pos/vel(cos_sim 曾 0.92→0.85)。
- ⚠️ 4-EMA 门 + 慢 EMA(α=0.05):防单 batch 噪声把 ratchet 一把推到顶档(run 22-50-41:全局 hsr 0.42 时 batch 抽样 0.80+ 误触发顶档,锁死 17k iter)。
- 门控:`window_gate_open`(EL_ema≥250)且 **非** `cos_sim_ratchet_freeze`(cos_sim_ema≥0.45)才推进。

### 10.5 站立门闸 + σ/EMA(`update_pingpong_curriculum` 内)

- **POS_VEL_GATE / ORI_GATE**(单调 latch):EL_ema<250 时把 goal_position/velocity(+三个 pre_strike)和 goal_orientation 的权重强制 0。⚠️ M1 RSI 修好拍面朝向后,run 14-51-08 在 EL≈41 卡 1680+ iter——策略靠「边摔边挥」farm pos/vel 奖励(14× baseline)。站不稳不许学击球。开闸后窗口课程的 `max()` 再把权重抬回。(注:Phase 课程已是更强的同类闸,这套作为冗余保留。)
- **cos_sim EMA**:v64 起读 **strike-instant** signed cos(`cos_sim_at_strike`,真接触面 ~0.8)而非当前帧 cos(~0.46),修 Stage-2 解锁卡死 + 冻结抖动。
- **sequenced curriculum(Stage1→2→3)**:先让窗口课程毕业(Stage1),再解锁 `v_in_mag`(Stage2:shape_tier≥6 且 hsr≥0.85 且 cos≥0.55),再解锁 `hit_y` cap(Stage3:v_in≥3.5 且 hsr≥0.80)。⚠️ 防 σ 收紧、球加速、范围扩张同时压策略。
- **cross-curriculum cooldown(500 iter)**:shape_tier 和 v_in_mag 不能在 500 iter 内同时升。
- **cos_sim collapse retreat**:cos_sim_ema<0.35 时**反向**退档(v_in 退到 2.5、hit_y cap 退回 initial),给策略重找拍面梯度的路。
- **goal_base 平滑 ramp**(`_GOAL_BASE_RAMP`):权重随 EL_ema 从 0.5(ep_lo=50)线性到 env_cfg 的 1.5(ep_hi=250)。一直在线(不二值门),避免「先学静态站再重学动态横移」的负迁移。

### 10.5b 脸部关节降权 `update_imit_orient_weight`(v65,只在 Phase 2)

cos_sim_ema 每停滞 600 iter,就把脸部关节(waist_yaw、right shoulder_yaw/elbow/wrist_roll)的模仿权重 ×0.6(到 floor 0.05)。
- ⚠️ 全关节模仿把这几个关节钉向「静腰 + ~0.80 拍面」的 demo,**封顶了拍面**;停滞驱动地降权,让 `goal_orientation` 招募腰扭 + 腕把拍面顶过 0.80。**乘性**叠在 `task_phase` 每 tick 写的脸部 seed 之上。

### 10.6 pre_strike 退火 `update_prestrike_ramp_anneal`(只在 Phase 2)

cos_sim_ema 每停滞 600 iter,把 pos/vel/ori 三个 pre_strike 的 `ramp_time` ×0.6;降到 <0.05 时**彻底关闭**(ramp→0 + weight→0)。
- 目的:长 pre_strike 窗(`ramp_time=0.2`)早期教腰+base 协调;到瓶颈期撤掉这根拐杖,逼策略靠真正的击球瞬间奖励。

### 10.7 table-guard `update_table_guard_stage`(4 阶段)

```
Stage0 hidden  : 桌 z=−10,桌接触罚=0,撞桌终止短路。让策略先学站+挥拍,不被「不可能的接触」终止
Stage1 unlocked: hsr_ema≥0.65 且 cos_sim_ema≥0.45 且 EL_ema≥400 且 iter≥1500 → 翻 flag;桌靠各 env reset 时 EventTerm 自然搬回(不主动 teleport 砸拍)
Stage2 ramping : 桌接触罚 0→(−10,−1) 线性 ramp 500 iter
Stage3 active  : 罚到位 + 撞桌终止启用
```

### 10.8 整合视图

| # | 名称 | 控制对象 | 单调 | 函数 |
|---|---|---|---|---|
| 1 | imit_anneal | EMA 源 + imit 权重(被 task_phase 覆盖) | latch | `update_imitation_weight` |
| 2 | task_phase | 3 相位 imit/goal/腿正则/脸部 | latch | `update_task_phase` |
| 3 | shape σ | σ_pos/vel/ori 收紧 | latch | `update_pingpong_curriculum` |
| 4 | window | strike_window 收 + w_pos/vel/ori 抬 | latch | 同上 |
| 5 | sequenced | v_in / hit_y cap 分级解锁 | 门控 | 同上 |
| 6 | cos_sim freeze/retreat | 拍面崩时冻结/退档 | 反向 | 同上 |
| 7 | pos_vel/ori gate | 站不稳时关击球奖励 | latch | 同上 |
| 8 | goal_base ramp | goal_base 0.5→1.5 | ramp | 同上 |
| 9 | imit_orient_anneal | 脸部关节模仿降权 | latch | `update_imit_orient_weight` |
| 10 | prestrike_ramp_anneal | pre_strike 缩窗→关 | latch | `update_prestrike_ramp_anneal` |
| 11 | table_guard | 桌隐藏→激活 + 接触罚 ramp | 单调 | `update_table_guard_stage` |

---

## 11. Sim / PPO

**Sim**:`sim.dt=1/200 s`、`decimation=4` → 50 Hz;`episode_length_s=10.0`(500 步);`gravity=(0,0,−9.81)`;`gpu_max_rigid_patch_count=10·2¹⁵`。

**PPO**([agents/rsl_rl_ppo_cfg.py](agents/rsl_rl_ppo_cfg.py)):`num_steps_per_env=24`、`max_iterations=180000`、`save_interval=1000`、`empirical_normalization=False`;MLP `[512,256,128]`(actor+critic)、elu;`clip_param=0.2`、`entropy_coef=0.005`、`num_learning_epochs=5`、`num_mini_batches=4`、`lr=5e-4 adaptive`、`gamma=0.99`、`lam=0.95`、`desired_kl=0.01`、`max_grad_norm=1.0`。

---

## 12. 关键 convention 与实测

- **四元数全工程 wxyz**;任何 isaaclab 边界返回 xyzw 必须显式转。
- `BLADE_NORMAL_LOCAL=(0,−1,0)`(URDF paddle fixed joint rpy=−135° 绕 X)。
- `SWING_FOREHAND=0`、`SWING_BACKHAND=1`;`sign=1−2·swing`。
- demo 风格 = **step-and-reach**:挥拍靠右肩 pitch 大幅摆臂(~35–48°)+ base 跨步,`waist_yaw` 实测只 ~4.6°(几乎锁住)。这就是为什么 5-DOF 臂零冗余、拍面难——也是为什么要靠 base 横移 + 课程把拍面单独学出来。
- **TensorBoard 权威性**:判读 hit_success / cos_sim / fail rate 用 `Curriculum/pingpong/*`(step-level batch 均值);`Metrics/pingpong/*` 只在 episode 结束累计,早期 time_out 占比高时假性显 0。
- **拍面真值看 `cos_sim_at_strike_*`**(strike-instant signed cos,~0.8),不是 `cos_sim_*_only`(当前帧,~0.46)。

---

## 13. 启动训练 / 验证

### 13.1 启动(三套 PD 切换只改 `ROBOT_CFG` import)

```bash
python scripts/rsl_rl/train.py \
  --task Unitree-G1-23dof-Pingpong-HITTER \
  --headless --run_name <name>
```
- 换 PD 版本:[hitter_env_cfg.py:20-22](robots/g1_23dof/hitter/hitter_env_cfg.py#L20-L22) 改 `ROBOT_CFG` import(big-PD ↔ low_PD ↔ 软基线)。**换 PD 必须从头重训**(动力学不同,不能 resume 别套 PD 的 checkpoint)。
- 任务入口注册见 [robots/g1_23dof/hitter/__init__.py](robots/g1_23dof/hitter/__init__.py)。

### 13.2 验证清单(from-scratch,按相位看)

| 阶段 | 看什么 | 不达标含义 |
|---|---|---|
| 早期(Phase0/1) | `mean_episode_length` 升 + `hard_contact` 降,**EL>250 前别看拍面** | 站立没学会 |
| 进相位 | `task_phase` 0→1(EL 跨 350)→2(EL≥450 且 p1elapsed≥2000) | 相位卡住查 EL_ema |
| Phase2 后 | `cos_sim_at_strike_forehand/backhand` 升(主判据)、`hit_success_rate` 不跌、`*_fail_rate` 降、`prestrike_ramp` 阶梯下降最终归 0、`base_y_drift_meanabs` 不塌 | |
| cheat 检测 | `paddle_y_base_at_strike_forehand≈forehand_y`、`_backhand≈backhand_y`;`cos_sim_*_only` 正反手都 >0.5;`hsr_forehand` 与 `hsr_backhand` 差 ≤0.10 | 反手 paddle_y 跑到 −y 侧 / cos<0 = cross-body cheat |
| 目检(play) | 正手不再「反手姿势+翻腕」、绿箭头贴红箭头、击球后收拍回中、不抬腿/不跳/不拖脚 | |

### 13.3 停训判据(成功 → 切 hitter_real / 部署)
🟢 `hit_success_rate≥0.80` 持续 500 iter、`vel_fail≤0.15`、`cos_sim_at_strike` 500 iter 最低 ≥0.50、`ori_fail≤0.20`、`task_phase=2`、`table_stage=3`。
🔴 iter>12000 但 cos_sim_ema<0.40 或 hsr<0.30;actor std<0(PPO 崩);bad_orientation>50% 持续 1000 iter。

---

## 14. 文件索引(复现入口)

| 文件 | 关键内容 |
|---|---|
| [mdp/commands.py](mdp/commands.py) | `PingpongCommand`、世界系采样+锚定 divider、`_solve_paddle_target`(Eq.5/6)、`_blade_target_cosine`(signed)、per-swing 诊断 metric |
| [mdp/motion_loader.py](mdp/motion_loader.py) | 双 clip 载入、`expert_offset_base` 预处理、float-frame lerp/slerp、`frame_from_step`(自然 follow-through + 末帧锁定) |
| [mdp/observations.py](mdp/observations.py) | 12 项 obs、`active_face`/`target_normal`、`DelayedObservation`、IMU offset wrapper |
| [mdp/rewards.py](mdp/rewards.py) | imitation(post_strike_delay)、goal_*(Gaussian vel、signed ori、pre_strike 回投)、腿正则、bounded action 罚 |
| [mdp/terminations.py](mdp/terminations.py) | hard_contact、撞桌持续终止(stage-aware) |
| [mdp/events.py](mdp/events.py) | IMU offset / comm delay / table teleport / DR |
| [mdp/curriculums.py](mdp/curriculums.py) | task_phase(3 相位 + 脸部 ramp)、shape σ latch、window、sequenced、imit_orient_anneal、prestrike_ramp_anneal、table_guard |
| [robots/g1_23dof/hitter/hitter_env_cfg.py](robots/g1_23dof/hitter/hitter_env_cfg.py) | 全部 cfg(选 ROBOT_CFG PD 版本、reward/obs/curriculum 注册与顺序) |
| [assets/robots/unitree.py](../../assets/robots/unitree.py) | **3 套 PD 配置**:`UNITREE_G1_23DOF_MIMIC_CFG`(软) / `..._PADDLE_MIMIC_CFG`(big-PD) / `..._PADDLE_MIMIC_CFG_low_PD` |
| [agents/rsl_rl_ppo_cfg.py](agents/rsl_rl_ppo_cfg.py) | `BasePPORunnerCfg` |

---

**END OF final_1**
