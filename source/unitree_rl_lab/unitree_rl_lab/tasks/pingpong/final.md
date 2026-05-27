# Paper-Aligned EnvCfg 设计表 v5 — HITTER (arXiv:2508.21043v2)
首先详细的阅读文章/home/woan/文档/zotero/AllData/storage/7XEMW63Q/Su 等 - 2025 - HITTER A HumanoId Table TEnnis Robot via Hierarchical Planning and Learning.pdf

## Context

**v5 增量** (相对 v4): 5–10 全部按用户评论修订, 三表 (1–3) 锁定不动.

**用户决定累计 (v5 锁定, #14–#23 新增)**:
1. paper [35] = DeepMimic (kernel k 实参借用)
2. r_g_base 击球前 ON, 击球后 (`t ≥ t_strike`) **OFF**, 不切下一击目标
3. Strike window: **±3 帧 @ 50Hz, 共 7 帧, `abs(t_to_hit) ≤ 0.06s`**
4. 删除 r^e (与 r_g_pos / r_g_vel 击球目标冲突)
5. **r^c → r^bp**: 上半身各 body 的 anchor-relative position 跟踪 (排除 `right_paddle_blade`); body-level velocity 不在此. **权重不能太高**.
6. r_g_ori 目标从 cmd 推: `n̂_target = v̂_racket / ‖v̂_racket‖`, 无独立 cmd 字段
7. obs `t_strike` 锁定为 **time-to-strike (剩余击球时间)**, 不用绝对时间
8. **Cmd hit point sample 范围**: `x = 0.4 m fixed`, `y ∈ [-1, 1] m`, `z ∈ [0.08, 0.6] m`
9. **Cmd planner 与训练 cmd 解耦** (planner 上层只关心"击球点 ↔ base" 相对关系)
10. **r^p / r^v 关节集合 J**: 仅上半身, 排除 `right_wrist_roll_joint`. |J|=10
11. **Cmd `Δt_swing`** = `truncN(low=0.2, high=1.5, peak_low=0.4, peak_high=0.8)` (非均匀, [0.4,0.6] 太严苛, v5.4 user-decided 放宽到 [0.4,0.8])
12. **Cmd `v̂_racket`**: `v_mag~U[2,6]`, `v_yaw=base_yaw+π+U[-40°,+40°]`, `v_pitch~U[10°,60°]`
13. **Cmd 重新生成时机**: `t_to_hit ≤ 0` 立即重采样, 不再有 gap1 / opponent 0.1s
14. **5 Action scale [v5 user-decided]**: 不用全局 0.25, 用 `unitree.py` 中 mimic 专用的 **per-joint** 公式 `0.25 · effort_limit / stiffness`. 对应 `UNITREE_G1_23DOF_PADDLE_MIMIC_ACTION_SCALE` (unitree.py:957-968).
15. **6 Termination 表桌碰撞 [v5 OPEN]**: ⏳ 用户提问"是 termination 还是 reward 惩罚?". 待 6.4 用户决定后定 σ_g_pos 范围.
16. **7 Events `swing_type` 独立采样 [v5 user-decided]**: 不再从 hit_xy 几何反推 swing_type. swing_type 是独立 uniform({forehand, backhand}) 采样, 然后通过 `expert_offset[swing_type]` 计算 `base_target_xy`.
17. **7 Events cmd noise 数值大幅缩小 [v5 user-decided]**: σ_p 从 0.02 → **gauss 在 [-0.015, 0.015] 采样, clip 到 [-0.0015, 0.0015]** (xyz 独立). 其他 σ_v / σ_base / σ_t 同比缩小 (见 7.3).
18. **8 σ_g_pos curriculum [v5 user-decided]**: 由"击球成功率"驱动, **单调收紧只升不降** (防止与噪声耦合发散), `0.10 → 0.03` linear. **0.03 是最小值** (考虑部署噪声底).
19. **8 mimic_start_prob curriculum [v5 user-decided]**: 由"survival rate (= 1 − fail_termination_rate)" / "timeout 比例" 驱动, 即站得稳了才放 mimic 比例.
20. **8 cmd noise σ_p 触发 [v5 user-decided]**: 当 σ_g_pos 收紧到 **0.03** 时才开 σ_p curriculum (击球已稳定再加感知噪声).
21. **8 cmd noise σ_t 触发 [v5 user-decided]**: 当 σ_g_pos 收紧到 **≥ 0.05** 时即可开 σ_t curriculum (比 σ_p 更早).
22. **8 其他 cmd noise (σ_v / σ_base) 课程 [v5 user-decided]**: 全部参照 σ_p 同款触发模式 + 同款 gauss/clip 双层结构, 见 8.3.
23. **9 contact sensor 精简 [v5 user-decided]**: 不用 6 个 sensor; 参照 mimic / locomotion 任务做法 — **单 `ContactSensorCfg("Robot/.*")` + 在 reward 端用 `body_names` 正则过滤**.
24. **9 URDF 路径 [v5 user-decided]**: `/home/woan/HumanoidProject/unitree_rl_lab/unitree_ros/robots/g1_description/g1_23dof_rev_1_0_paddle.urdf`. 与 `unitree.py:951` `UNITREE_G1_23DOF_PADDLE_MIMIC_CFG` 一致.

来源标记: `✓ paper` = HITTER 原文; `△ DM` = paper 引 [35] DeepMimic; `[user-decided]` = 用户决定; `⚠️` = 数值 [我提案], paper 未给; `⏳` = 等用户决定.

---

## 1. Reward 总览表 [HITTER V-B] **[user-confirmed v4, 不变]**

| # | Reward | 属于 | 公式 | 计算 / 来源 (ref or target) | weight | σ / kernel | paper? | 时序 (gate) |
|---|---|:---:|---|---|:---:|---|:---:|---|
| 0 | `w_i, w_g, w_r` | 总 | `r = w_i·r_i + w_g·r_g + w_r·r_r` | Eq. 7 顶层 | **0.5 / 1.0 / 1.0** ⚠️ | — | ✓ V-B2 | — |
| 1 | `r^p` (joint pose) | r_i | `exp(-2·Σⱼ‖q̂ⱼ ⊖ qⱼ‖²)`, **J = upper-body \\ {right_wrist_roll_joint}** | sim `q[J]` vs ref `q̂[J]` (ref 跟随 cmd.swing_type 选 clip, 见 11.1) | **0.65** (DM 原值) | k=2 sum-form (DM) | △ DM | dense 全程 (clip 末尾冻结) |
| 2 | `r^v` (joint vel) | r_i | `exp(-0.1·Σⱼ‖q̇̂ⱼ − q̇ⱼ‖²)`, **J = upper-body \\ {right_wrist_roll_joint}** | sim `q̇[J]` vs ref `q̇̂[J]` (同上) | **0.10** (DM 原值) | k=0.1 sum-form (DM) | △ DM | dense 全程 (clip 末尾冻结) |
| 3 | `r^bp` (body pos, anchor-relative) | r_i | `exp(-k·Σ_b ‖p_rel[b] − p̂_rel[b]‖²)`, b ∈ ℬ_pos | sim: `p_rel[b] = (p_world[b].xy − pelvis.xy, p_world[b].z)` **[v5.2 #Q3: xy 减 z 保留, 防蹲下作弊]**; ref: 同公式 | **0.25** [user #5] | k=40 (DM r^e 风格) ⚠️ | [user-decided] | dense 全程 (clip 末尾自然冻结) |
| ~~r^e~~ | end-effector pos | — | **删除** | — | — | — | △ DM | (用户 #4) |
| ~~r^c~~ | COM pos | — | **替换为 r^bp** | — | — | — | △ DM | (用户 #5) |
| 4 | `r_g_pos` | r_g | `exp(-‖p_blade^base − p̂_racket^base‖²/σ²)` **base frame** [v5.5 A1] | sim: `p_blade^base = R_base^T·(p_blade_world − p_base_world)`; target: `p̂_racket^base = R_base^T·(cmd.p̂_racket_world − p_base_world)`. **数学等价于 world frame 计算 (旋转不变), base frame 写法更清晰, 与 obs #5 同一坐标** | **2.0** ⚠️ | σ=0.05 m (curriculum → 0.02, 8.1) | ✓ V-B2 | sparse `abs(t_to_hit) ≤ 0.06s` |
| 5 | `r_g_vel` | r_g | `exp(-‖v_blade^base − v̂_racket^base‖²/σ²)` **base frame** [v5.5 A1 一致] | sim: `v_blade^base = R_base^T · v_blade_world`; target: `v̂_racket^base = R_base^T · cmd.v̂_racket_world`. 旋转不变, base 写法清晰 | **1.0** ⚠️ | σ=0.5 m/s | ✓ V-B2 | sparse 同上 |
| 6 | `r_g_ori` | r_g | `exp(-(1 − n_blade · n̂)²/σ²)` (cos 内积本身坐标无关) | `n_blade` = blade local +Y 旋到 world; `n̂ = v̂_racket_world / ‖v̂_racket_world‖`. 两侧同一坐标 (world) 即可 | **0.5** ⚠️ | σ=0.2 (cos dist) | ✓ V-B2 + IV-C | sparse 同上 |
| 7 | `r_g_base` | r_g | `exp(-‖p_base_xy_world − p̂_base_xy_world‖²/σ²)` **world frame** (base 自己 xy 在 world 才有意义) | sim: pelvis world xy; target: cmd `p̂_base_xy_world` | **0.3** ⚠️ | σ=0.3 m | ✓ V-B2 | dense `t_to_hit > 0`, **OFF `t_to_hit ≤ 0`** |
| 8 | `r_r` | r_r | regularization (子项见 4.4) | sim 当前 state | 子项见 4.4 ⚠️ | — | ✓ Eq.7 | dense 全程 |

### 1.1 ℬ_pos for r^bp + T_B (Critic obs #12) [v5.5 A2: 二者**统一**, 都排除 paddle blade]

**11 个 body** (r^bp 和 T_B 共用相同集合):

| # | body name | 类别 |
|---|---|---|
| 1 | `torso_link` | 躯干 |
| 2 | `left_shoulder_pitch_link` | 左臂 |
| 3 | `left_shoulder_roll_link` | 左臂 |
| 4 | `left_shoulder_yaw_link` | 左臂 |
| 5 | `left_elbow_link` | 左臂 |
| 6 | `left_wrist_roll_rubber_hand` | 左末端 |
| 7 | `right_shoulder_pitch_link` | 右臂 |
| 8 | `right_shoulder_roll_link` | 右臂 |
| 9 | `right_shoulder_yaw_link` | 右臂 |
| 10 | `right_elbow_link` | 右臂 |
| 11 | `right_wrist_roll_rubber_hand` | 右末端 |

**排除**:
- `right_paddle_blade` — **不跟踪** (用户 A2 原话: "因为我的专家数据不算非常好, 因此不希望末端拍面的位置影响真正击球时候的拍面朝向"). 拍面朝向由 r_g_ori (任务信号) 主导, body-level 不重复跟踪.
- 整个下半身 12 个 body — paper V-B2 ℬ ⊆ upper body.
- `pelvis` — anchor 自身, 用于减去得到 anchor-relative 坐标, 不是被跟踪对象.

**注**: r^bp 用 (pos, lin_vel) 子集, T_B 用 (pos, quat) 子集 → T_B 维度 = 11×7 = 77.

### 1.2 J for r^p / r^v [v5.5 A3: 全部 joint 显式列出]

**G1 23-DoF 全部 23 joint** (URDF 验证, 见 9.3 路径):
- 下半身 (12, 排除): 左/右 × {`hip_pitch_joint`, `hip_roll_joint`, `hip_yaw_joint`, `knee_joint`, `ankle_pitch_joint`, `ankle_roll_joint`}
- **上半身 11 个** (候选):

| # | joint name | 类别 |
|---|---|---|
| 1 | `waist_yaw_joint` | 腰 |
| 2 | `left_shoulder_pitch_joint` | 左臂 |
| 3 | `left_shoulder_roll_joint` | 左臂 |
| 4 | `left_shoulder_yaw_joint` | 左臂 |
| 5 | `left_elbow_joint` | 左臂 |
| 6 | `left_wrist_roll_joint` | 左末端 |
| 7 | `right_shoulder_pitch_joint` | 右臂 |
| 8 | `right_shoulder_roll_joint` | 右臂 |
| 9 | `right_shoulder_yaw_joint` | 右臂 |
| 10 | `right_elbow_joint` | 右臂 |
| ~~11~~ | ~~`right_wrist_roll_joint`~~ | **排除** (拍面 quat 通过 r_g_ori 主导, joint-level 不重复约束) |

**最终 |J| = 10**: 上面 11 个减去 `right_wrist_roll_joint`. 与 ℬ_pos / T_B 排除 `right_paddle_blade` (在 right_wrist_roll 下游) 的策略一致.

⚠️ **DIVERGENCE D — 排除 right_wrist_roll**: paper V-B2 仅说 J ⊆ upper-body, **未列具体 joint**. 我们排除 right_wrist_roll_joint 是 paper-derived 推断 (右小臂 roll = 拍面 yaw 等价), 让 r_g_ori 主导.

---

## 2. Observation 总览表 [HITTER Table I] **[user-confirmed v4, 不变]**

| # | 符号 | 含义 | 维度 (23-dof) | 计算 / 来源 | Actor | Critic | paper? |
|---|---|---|---|---|:---:|:---:|:---:|
| 1 | `ω_base` | base 角速度 (base frame) | 3 | sim IMU `base_link` | ✓ | ✓ | ✓ |
| 2 | `g_base` | 重力在 base frame 投影 | 3 | `R_base^T · [0,0,-1]` | ✓ | ✓ | ✓ |
| 3 | `e_base,x` | base 朝向 yaw 编码 | 2 | `[cos(yaw), sin(yaw)]` | ✓ | ✓ | ✓ |
| 4 | `p̂_base,xy − p_base,xy` | base 位置误差 | 2 | cmd − sim | ✓ | ✓ | ✓ |
| 5 | `p̂_racket` | 球拍目标位置 (**obs 端转 base-relative**, 内部存 world) | 3 | obs = `R_base^T · (cmd.p̂_racket_world − p_base_world)` [v5.3 #31] | ✓ | ✓ | ✓ |
| 6 | `v̂_racket` | 球拍目标速度 (world frame) | 3 | cmd 字段 | ✓ | ✓ | ✓ |
| 7 | `t_to_hit` | **剩余击球时间 (秒)**, ✓ 命名统一 (cmd 内部和 obs 都叫 t_to_hit) | 1 | cmd.t_to_hit (initial = Δt_swing 采样, 每 step 递减 dt=0.02s) | ✓ | ✓ | ✓ |
| 8 | `q` | 关节位置 | **23** ⚠️ | sim joint encoder | ✓ | ✓ | ✓ |
| 9 | `q̇` | 关节速度 | **23** ⚠️ | sim joint encoder | ✓ | ✓ | ✓ |
| 10 | `a_last` | 上一 step 动作 | **23** ⚠️ | rollout buffer | ✓ | ✓ | ✓ |
| 11 | `v_base` | base 线速度 (privileged) | 3 | sim base lin vel | – | ✓ | ✓ |
| 12 | `T_B` | 跟踪 body pos+quat (**ref clip 由 cmd.swing_type 选定**, 见 11.1; **排除 `right_paddle_blade`** [v5.5 A2]) | 7·\|ℬ\|=11·7=**77** [v5.5 A2: 原 84, 减 paddle blade 的 7 维] | ref body world state at `clip[swing_type][ref_frame_f]` (浮点插值, 11.1.2) | – | ✓ | ✓ |
| 13 | `t_left` | episode 剩余时间 | 1 | `T_episode − t_now` | – | ✓ | ✓ |
| 14 | `[q̄, q̇̄]` | ref clip joint pos+vel (**clip 由 cmd.swing_type 选定**, 同上) | **46** ⚠️ | motion_loader at `clip[swing_type][ref_frame_f]` (浮点插值) | – | ✓ | ✓ |

**Actor = 86, Critic = 86 + 3 + 77 + 1 + 46 = 213** [v5.5 A2: 原 220, 减 7 维 paddle blade]

---

## 3. Command 总览表 **[user-confirmed v4 + v5 #16 swing_type 独立采样]**

### 3.1 Cmd 字段表

| # | 字段 | 维度 | frame | sample 时机 | 取值范围 | paper? |
|---|---|:---:|---|---|---|:---:|
| 1 | `swing_type` | 1 (cat) | — | swing 完成后 (**独立 uniform 采样**, 不依赖 hit_xy) [v5 user-decided #16] | uniform({forehand, backhand}) | ✓ V-B |
| 2 | `p̂_racket` | 3 | **world** [v5.3 user-decided #31] | swing 完成后 | `x=0.4` 固定 (= 桌沿世界 x), `y∈[-1,1]`, `z∈[0.08,0.6]` (世界 z) | ✓ V-B (x=0.4) |
| 3 | `v̂_racket` | 3 | world | swing 完成后 | `v_mag~U[2,6]` ∧ `v_yaw=world_frame[-40°,40°]` ∧ `v_pitch~U[10°,60°]` | ✓ V-B |
| 4 | `p̂_base,xy` | 2 | world | swing 完成后 | `hit_xy − expert_offset[swing_type]` (#16) | ✓ V-B |
| 5 | `t_to_hit` (剩余击球时间, scalar) | 1 | — | swing 完成后 (= `t_to_hit ≤ -t_post_swing` 那一刻立即重采样) [v5.4] | 重采样: `t_to_hit ← t_pre_initial`, `t_pre_initial ~ truncN[0.2,1.5] peak [0.4,0.8]`. 每 step: `t_to_hit ← t_to_hit − dt`. **t_to_hit 在 pre 阶段 →0, post 阶段 →负, 走到 -t_post_swing 触发重采样.** | ✓ V-B |
| 6 | `t_pre_initial` (pre-strike sim 时长, scalar) [v5.4 NEW] | 1 | — | 重采样时同 t_to_hit 采样, 保留作为缩放分母 | `truncN[0.2, 1.5] peak [0.4, 0.8]` 秒 [v5.4 user-decided 放宽: 原 [0.4,0.6] 太严苛] | [我提案 ⚠️ DIVERGENCE Q] |
| 7 | `t_post_swing` (post-strike sim 时长, scalar) [v5.4 NEW] | 1 | — | swing 完成后, **独立 truncN 采样**, 不进 obs | `truncN[0.2, 1.5] peak [0.4, 0.8]` 秒 [初版与 t_pre 同分布; v5.4 同步放宽] | [我提案 ⚠️ DIVERGENCE Q] |
| 8 | `cur_step` (sim step 计数器) [v5.4 替代 cur_frame, 因 ref_frame 变浮点] | 1 | — | swing 重采样时复位 0 | int, 每 step +1, 范围 [0, (t_pre_initial+t_post_swing)/dt] | (内部状态) |
| 9 | (隐含) `n̂_target` | 3 | world | — | `v̂_racket / ‖v̂_racket‖` | ✓ IV-C |

### 3.2 Cmd 生成代码 [v5.4: 双段时间比例插值 + cur_step]

**Planner (上层 robojudo, 一次性预处理 — 同时记录 expert clip 的 pre/post 时长, 见 11.1.1)** [v5.5 A9: 修正为 base frame, ref clip yaw ≠ 0]:
```python
for swing_type in [forehand, backhand]:
    clip = load_expert_clip(swing_type)
    imp = clip.impact_frame
    fps = clip.fps                                        # = 50
    # base-frame xy delta (= world delta 旋到 当时 pelvis frame, 见 11.4)
    expert_offset[swing_type]   = R(clip.pelvis_yaw[imp])^T · (clip.blade_xy[imp] − clip.pelvis_xy[imp])
    expert_pre_duration[swing_type]  = imp / fps                            # native pre 时长 (秒)
    expert_post_duration[swing_type] = (clip.length - 1 - imp) / fps        # native post 时长 (秒)
# forward_001:  pre=0.74s  post=0.88s,  expert_offset = (0.496, 0.208)  base frame [v5.5 A9 实测]
# backward_004: pre=0.40s  post=0.86s,  expert_offset = (0.428, 0.106)  base frame [v5.5 A9 实测]
# 旧 v5.3 假设 yaw≈0, 直接 world delta 是错的; 实测 forward yaw=63.6°, backward yaw=128.8°
```

**训练端 cmd 生成 (每次 swing 完成后, = `t_to_hit ≤ -t_post_swing` 那一 step 的 events.interval 阶段)** [v5.4 + v5.3 #31]:
```python
# (v5 #16: swing_type 独立, 不再从 hit_xy 几何反推)
swing_type = uniform_sample({forehand, backhand})

# (#8: hit point 采样, 全部 world frame [v5.3 #31])
hit_x_world = 0.4                                     # = 桌沿 world x, paper V-B
hit_y_world = uniform(-1.0, 1.0)
hit_z_world = uniform(0.08, 0.6)
p̂_racket_world = (hit_x_world, hit_y_world, hit_z_world)

# (#16: swing_type → expert_offset → base_target) [v5.5 A9: expert_offset 是 base frame, 须按 robot 当前 yaw 旋到 world]
yaw_robot = base_yaw_now                                                  # robot 当前 base yaw (= 0 ± reset noise)
c, s = np.cos(yaw_robot), np.sin(yaw_robot)
offset_world = np.array([                                                  # base frame → world frame
    c * expert_offset[swing_type][0] - s * expert_offset[swing_type][1],
    s * expert_offset[swing_type][0] + c * expert_offset[swing_type][1],
])
base_target_xy_world = (hit_x_world, hit_y_world) - offset_world           # world 减法 (offset 已旋到 world)
p̂_base_xy_world = base_target_xy_world

# (#12: v̂_racket world frame, paper Table I)
v_mag   = uniform(2.0, 6.0)
v_yaw   = base_yaw + pi + uniform(-deg2rad(40), deg2rad(40))
v_pitch = uniform(deg2rad(10), deg2rad(60))
v̂_racket_world = v_mag * np.array([cos(v_yaw)·cos(v_pitch), sin(v_yaw)·cos(v_pitch), sin(v_pitch)])

# (#11 + v5.4: pre/post 双段时长独立采样)
t_pre_initial = truncN(0.2, 1.5, 0.4, 0.8)        # pre-strike sim 时长 (秒) [v5.4 放宽 peak 0.6→0.8], 进 obs 作 t_to_hit 初值
t_post_swing  = truncN(0.2, 1.5, 0.4, 0.8)        # post-strike sim 时长 (秒) [v5.4 放宽 peak 0.6→0.8], 不进 obs (见 11.5)
t_to_hit      = t_pre_initial                      # cmd.t_to_hit 单调递减, 每 step -= dt
cur_step      = 0                                  # sim step 计数器 (取代 v5.3 cur_frame)
                                                   # ref_frame 由 cur_step 通过 11.1.2 双段插值映射得到 (浮点)

# (paper IV-C, 隐含)
n̂_target = v̂_racket_world / ‖v̂_racket_world‖     # world frame
```

**Obs 端坐标转换 (cmd → policy 输入)** [v5.3 #31 + v5.5 A10: Actor noisy / Critic clean]:
```python
# Actor obs (注入 cmd.noise_*, 一次冻结 per swing, 见 7.3.2):
obs_actor.p̂_racket  = R_base^T · ((cmd.p̂_racket_world + cmd.noise_p) − p_base_world)   # base-rel + noise
obs_actor.v̂_racket  =              cmd.v̂_racket_world + cmd.noise_v                     # world + noise
obs_actor.p̂_base_err = (cmd.p̂_base_xy_world + cmd.noise_base) − p_base_xy_world         # world delta + noise
obs_actor.t_to_hit   = cmd.t_to_hit + cmd.noise_t                                        # scalar + noise

# Critic obs (无噪声, asymmetric AC privileged critic):
obs_critic.p̂_racket  = R_base^T · (cmd.p̂_racket_world − p_base_world)   # → base-relative (2 #5)
obs_critic.v̂_racket  = cmd.v̂_racket_world                                # 保 world (2 #6, paper)
obs_critic.p̂_base_err = cmd.p̂_base_xy_world − p_base_xy_world           # base 位置误差 (2 #4 已是 world delta)
obs_critic.t_to_hit   = cmd.t_to_hit                                     # 标量 (2 #7)
# 注: t_post_swing / cur_step / cmd.noise_* 都不进 obs
# 注: r_g / r_g_base / strike window gate `abs(t_to_hit)<=0.06s` 全部用 cmd 真值 (无噪声), 见 7.3.3
```

**每 step 维护** [v5.4: 改用 t_post_swing 边界]:
```python
t_to_hit -= dt                                      # dt = 0.02s @ 50Hz, t_to_hit 单调递减
cur_step += 1                                       # sim step 计数器 +1
# (ref_frame 不再单独维护; 通过 11.1.2 get_ref_state(cmd, dt) 当场算)
if t_to_hit ≤ -t_post_swing:                        # post 段也走完 (= sim_pre_steps + sim_post_steps step)
    立即重采样 (回到上面 cmd 生成代码)              # swing_type / hit / v̂ / t_pre_initial / t_post_swing / t_to_hit / cur_step
```

### 3.3 Cmd 生命周期 [v5.4: 双段时间 + cur_step + post 段重采样边界]

```
═══ episode reset ═══
│ 生成首组 cmd (swing_type / p̂_racket / v̂_racket / p̂_base,xy / t_to_hit=t_pre_initial / t_post_swing / cur_step=0)
│ obs.t_to_hit = cmd.t_to_hit (= 首次 t_pre_initial 采样, t_post_swing 不进 obs)
│ ref state 起点: get_ref_state(cmd, dt) at cur_step=0 → ref_frame_f=0 (clip 第 0 帧)
│
│ 每 step:
│   cmd.t_to_hit  -= dt (= 0.02s @ 50Hz)         # pre 段 →0, post 段 →负
│   cmd.cur_step  += 1                             # 取代 v5.3 cur_frame
│   ref 计算: get_ref_state(cmd, dt) 当场算       # 11.1.2 双段独立线性插值缩放
│       pre 段:  cur_step ∈ [0, sim_pre_steps]     → ref_frame ∈ [0, impact]
│       post 段: cur_step ∈ (sim_pre_steps, total] → ref_frame ∈ (impact, clip_len-1]
│   abs(cmd.t_to_hit) ≤ 0.06s 时 r_g sparse 激活 (±3 帧窗, t_to_hit=0 = ref impact 帧, 击球帧对齐)
│   cmd.t_to_hit > 0 时 r_g_base dense 激活;  cmd.t_to_hit ≤ 0 时 r_g_base OFF (1 #7)
│
├──── cmd.t_to_hit ≤ -t_post_swing (post 段也走完) ────
│    立即重采样新 cmd; obs.t_to_hit ← 新的 t_pre_initial
│    swing_type 独立重采样 → ref clip 切换; cur_step=0 重新对齐
│    r_g_base 切到新 p̂_base,xy
│
═══ 重复, 直到: 10s timeout / 摔倒 termination / robot↔table contact (6.4 软惩罚不 terminate) ═══
```

---

## 4. Reward weights + σ baseline + r_r 完整套餐 [v5.5 A4: r_r 全表 inline]

### 4.1 Top-level weights

| | w_i | w_g | w_r |
|---|---|---|---|
| paper Eq. 7 顶层 | **0.5** ⚠️ | **1.0** ⚠️ | **1.0** ⚠️ |

⚠️ **DIVERGENCE A** — paper 公式 Eq. 7 形式正确, 数值未给. 我们 0.5 / 1.0 / 1.0 是工程选择.

### 4.2 r_i 子项 (DM kernel)

| sub | k | weight | 来源 |
|---|---|---|---|
| `r^p` (joint pose, |J|=10) | 2 | **0.65** | DM 原值 |
| `r^v` (joint vel, |J|=10) | 0.1 | **0.10** | DM 原值 |
| `r^bp` (body pos rel-to-pelvis, |ℬ|=11) | 40 | **0.25** | DM r^e 风格 (用户 #5: 权重不能太高) |

**spike ratio** ≈ 5.2× ✓ paper "relatively high" (peak r_g ≈ 5.2 × baseline r_i).

### 4.3 r_g 子项

| sub | weight | σ | gate |
|---|---|---|---|
| `r_g_pos` | **2.0** ⚠️ | 0.05 m (curriculum → 0.02, 8.1) | sparse `abs(t_to_hit) ≤ 0.06s` |
| `r_g_vel` | **1.0** ⚠️ | 0.5 m/s | sparse 同上 |
| `r_g_ori` | **0.5** ⚠️ | 0.2 (cos dist) | sparse 同上 |
| `r_g_base` | **0.3** ⚠️ | 0.3 m | dense `t_to_hit > 0`, OFF `t_to_hit ≤ 0` |

### 4.4 r_r 完整套餐 [v5.5 A4: 删除 pelvis_height_mimic / pelvis_height_free 双段]

⚠️ **DIVERGENCE I — r_r 整体**: paper V-B 完全没给 r_r 细节, 全部 [我提案], 沿用 IsaacLab humanoid locomotion 标准 reg 套餐.

| # | sub-term | 公式 | weight | 说明 |
|---|---|---|---|---|
| 1 | `alive_reward` | `+1` per step (dense) | **+0.1** | 防 policy 学早结束 episode 来最大化负 reg |
| 2 | `action_rate_l2` | `‖a_t − a_{t−1}‖²` | **-0.01** | 平滑, 防 jitter |
| 3 | `action_l2` | `‖a_t‖²` | **-0.0005** | 限幅, 防大动作 |
| 4 | `dof_torques_l2` | `‖τ‖²` | **-2e-5** | 节能 |
| 5 | `dof_acc_l2` | `‖q̈‖²` | **-2.5e-7** | 关节平滑 |
| 6 | `dof_pos_limits` | hinge 关节超限 (per joint) | **-5.0** | 防爆关节 |
| 7 | `pelvis_orientation_l2` | `‖proj_g_xy‖²` (倾倒度) | **-1.0** | 防摔倒 |
| 8 | `pelvis_height` | `(h − 0.74)²` (h = pelvis world z) | **-10.0** | 站立高度. **[v5.5 A4: 单一权重, 不分 mimic/free]** — 因每 episode 都有专家数据持续参与, ref 通过 r^bp 的 z 已约束高度, 这里只做轻 reg 防漂移 |
| 9 | `feet_slide` | 接触脚的水平速度 | **-0.05** | 防蹭脚 |
| 10 | `feet_air_time` | 摆动脚悬空时间 (target=0.4s, 越接近越奖) | **+0.5** | 鼓励正常 walking gait |
| 11 | `undesired_contacts` | 非足非腰非手腕的 body 触地 (per body indicator) | **-1.0** | 软惩罚 (膝/手肘等) |
| 12 | `r_table_contact` (6.4.3) | `−Σ_b 𝟙[‖force(b, table)‖ > 1.0 N]` (per-body indicator) | **-1.0** | robot ↔ table 任何接触都罚, 含球拍 |

**单 step 量级估算**:
- `action_l2`: ‖a‖²~10 × 5e-4 ≈ -5e-3
- `dof_torques_l2`: ‖τ‖²~1000 × 2e-5 ≈ -2e-2
- `dof_acc_l2`: ‖q̈‖²~1e5 × 2.5e-7 ≈ -2.5e-2
- `pelvis_height`: Δh²~0.04² × 10 ≈ -1.6e-2
- `alive_reward`: +0.1
- 总 r_r ≈ -0.05 ~ -0.2 per step, × `w_r=1.0` ≈ -0.05 ~ -0.2.

相对 r_g spike (sparse window 内 ≈ 1.5–3.5), reg/任务比 ≈ 5–15%, 不主导任务. ✓

⚠️ **DIVERGENCE I — pelvis_height 单段 (新)**: 老 v2 文档分 mimic/free 两段权重 (-10/-50). v5.5 用户决定 (A4): 整个 episode 都有专家数据参与, 不存在 "free play" 段, 单一权重 -10. **风险**: 若 r^bp 的 z 约束不够强 (k=40 已经较硬, 但 anchor-relative 残差累计后等效 z 约束强度需第一轮 monitor), 机器人可能逐步偏离 0.74 m. **第一轮训练监控**: pelvis_height 漂移 + 是否需要把权重收紧到 -20 ~ -30.

⚠️ **DIVERGENCE B — r_g_base cut-off 仍保留**: paper V-B 描述 base_pos reward "before strike". 我们对齐: dense `t_to_hit > 0`, OFF `t_to_hit ≤ 0` (1 #7). 不改回 dense 全程. paper-aligned ✓.

---

## 5. Actions 总览表 [v5 user-decided #14: 用 mimic per-joint scale]

### 5.1 Action 字段

| 字段 | 维度 | 类型 | 公式 |
|---|:---:|---|---|
| `JointPositionAction` | **23** | 关节目标位置 (offset from default) | `q_target[j] = q_default[j] + scale[j] · a[j]` |

### 5.2 Action scale [v5: per-joint, 不用全局 0.25]

参考 [unitree.py:957-968](source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py#L957-L968) 的 `UNITREE_G1_23DOF_PADDLE_MIMIC_ACTION_SCALE`:

```python
# 直接复用 unitree.py 已有常量:
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_23DOF_PADDLE_MIMIC_ACTION_SCALE
# 内部公式 (unitree.py:957-968):
#   for actuator in cfg.actuators.values():
#       e = actuator.effort_limit_sim       # 每关节 effort 上限 (N·m)
#       s = actuator.stiffness              # 每关节 P-gain
#       scale[joint] = 0.25 · e[joint] / s[joint]
```

**设计含义** (unitree.py 注释): action_scale ≈ 25% 最大可承受位置误差. mimic 动作幅度大, 小 scale 会成为限制 → 球类同样需要快速大幅动作, 复用 mimic 配置.

### 5.3 控制频率 [paper V "50Hz"]

- physics dt = 1/200 s
- decimation = **4**
- control = 50Hz ✓

### 5.4 Action clipping

不显式 clip action. `q_target` 在 sim 端被 joint pos limit 截断, 通过 `dof_pos_limits` (4.4) 软惩罚.

⚠️ **DIVERGENCE K — action scale**: paper 仅给 "joint pos action @ 50Hz", 未给具体 scale 公式. 我们用 mimic 配置是 [user-decided #14], 与 unitree_rl_lab 内部其他 mimic 任务一致.

---

## 6. Terminations 总览表

### 6.1 Termination 项 (确定项)

| # | 信号 | 触发条件 | time_out flag | 含义 | paper? |
|---|---|---|:---:|---|:---:|
| 1 | `time_out` | `t ≥ 10s` (= 500 steps @ 50Hz) | ✓ (bootstrap V) | episode 自然终止 | ✓ V-B1 |
| 2 | `bad_orientation` | `‖proj_g_xy‖ > 0.7` (~45°) | ✗ | 摔倒 / 倾倒 | [我提案] |
| 3 | `root_height_below_min` | `pelvis_height < 0.3` m | ✗ | 摔到地上 | [我提案] |
| 4 | `undesired_contact_terminate` | `pelvis` / `head_link` / `.*_hip_pitch_link` 触地 (URDF 验证, 见 9.3 路径) | ✗ | 严重摔倒 (头/胯/髋触地) | [我提案] |

### 6.2 与 mimic / locomotion 的对照 [explore 验证]

- **mimic** (`tracking_env_cfg.py:308-336`): `time_out` + `anchor_pos` (torso z<0.25m) + `anchor_ori` (倾角>0.8 rad) + `ee_body_pos`
- **locomotion** (`velocity_env_cfg.py:404-411`): `time_out` + `base_height` (root<0.2m) + `bad_orientation` (倾角>0.8 rad)
- **我们 (pingpong)**: 借 locomotion 套餐 (站立任务为主, 不强制 ee 高度), 加严重接触 terminate.

### 6.3 GAE bootstrap 含义

- `time_out=True` → `terminated=False` + 自然结束, GAE 用 `V(s_T)` bootstrap (避免 10s 末尾 reward 截断)
- `time_out=False` → `terminated=True`, GAE 不 bootstrap (失败状态 V≈0)

### 6.4 表桌碰撞 — RigidBody + ContactSensor (v5.1 修订, 用户 #25)

**用户决定 v5.1**: 不写复杂几何判断, 直接在 scene 里**生成静态长方体球桌**, 用 IsaacLab 物理引擎做碰撞检测. 通过 ContactSensor 读 robot ↔ table 接触力, reward 端简单粗暴.

**优点**:
- 无需手写球拍半径补偿 / arm z 阈值 / base x 阈值等几何判断
- 自动覆盖球拍 STL 真实几何 (球拍碰桌即触发, 半径自动包含)
- 复用 IsaacLab 标准 `ContactSensor` + `undesired_contacts` 风格 reward

#### 6.4.1 Table asset (9.1 #6 实现)

```python
# scene 端 (与 robot 同级, 静态 RigidBody)
table = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Table",
    spawn=sim_utils.CuboidCfg(
        size=(2.74, 1.525, 0.76),                     # 国际标准球桌 长×宽×高
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),  # 不动
        collision_props=sim_utils.CollisionPropertiesCfg(),                    # 启用碰撞
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.4, 0.7)),
    ),
    init_state=AssetBaseCfg.InitialStateCfg(pos=(1.77, 0.0, 0.38)),
    # x_center = 0.4 + 2.74/2 = 1.77 (近端桌沿 x=0.4 与 paper 击球点对齐)
    # z_center = 0.76/2 = 0.38 (桌面 top 在 z=0.76)
)
```

世界坐标约定 [user #15]: 机器人初始 base = world (0, 0, 0.76), 朝 +x 方向, 桌沿 x=0.4.

#### 6.4.2 ContactSensor (filter to table only)

```python
robot_table_contact = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*",
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"],     # 只检测 robot vs table 接触
    history_length=3,
)
```

#### 6.4.3 Reward 子项 (整合到 r_r 4.4)

| sub | 公式 | weight | 说明 |
|---|---|---|---|
| `r_table_contact` | `−Σ_b 𝟙[‖force(b, table)‖ > 1.0 N]` (per-body indicator) | **-1.0** | 任何 robot body 撞 table 都罚, 包括球拍 |

**单一 reward 项**, 不再分 base / arm / blade. 物理引擎自动给真值, σ_g_pos 课程不再依赖 6.4 选项.

⚠️ **DIVERGENCE N — 表桌建模**: paper V 未明 (paper 关注击球策略, 不强制建模球桌作为障碍). 我们加 RigidBody 球桌是工程选择, 用最自然的方式表达 "机器人不能撞桌". 风险: sim 性能略降 (多一个 collision body × 4096 envs); 但 IsaacLab 静态 RigidBody 性能开销可忽略.

#### 6.4.4 σ 不是半径 (用户提问 #15)

σ 是 Gaussian kernel std-dev (`exp(-d²/σ²)`), 不是硬半径. 在 d=σ 时 reward = e⁻¹ ≈ 0.37; d=2σ 时 ≈ 0.018. 工程上"有效命中半径" ≈ σ. **σ_g_pos 详细 curriculum 见 8.1, 终值锁 0.02m** (= 噪声极限, 用户 #28).

---

## 7. Events 总览表 (Domain Rand + RSI + Cmd Noise)

### 7.1 Mode `startup` (一次性, paper V-B3)

| # | 项 | 范围 | 单位 | paper? | 备注 |
|---|---|---|---|:---:|---|
| 1 | `add_link_mass` | uniform `±10%` | kg | ✓ V-B3 | 每 link 独立 |
| 2 | `randomize_link_friction` | uniform `[0.5, 1.5]` × | — | ✓ V-B3 | foot 重点 |
| 3 | `randomize_joint_friction` | uniform `[0.5, 1.5]` × | Nm·s | ✓ V-B3 | 23 dof 独立 |
| 4 | `randomize_joint_damping` | uniform `[0.7, 1.3]` × | — | ✓ V-B3 | 23 dof 独立 |
| 5 | `randomize_imu_offset` | gauss `σ=2°` | rad | ✓ V-B3 | base IMU |
| 6 | `comm_delay` | uniform `[0, 20]` ms | ms | ✓ V-B3 | obs 延迟 1 step |

### 7.2 Mode `reset` (每 episode, [v5 #16 swing_type 独立])

| # | 项 | 范围 | 备注 | paper? |
|---|---|---|---|:---:|
| 1 | `reset_robot_pose` (RSI) | 见 7.4 | 必从 ref clip 内随机帧起步 [v5.5 A7: 删 mimic_start_prob 分支] | [paper-derived] |
| 2 | `reset_joint_pos_noise` | gauss `σ=0.05` rad | 23 dof 独立 | [我提案] |
| 3 | `reset_base_yaw_noise` | uniform `±10°` | 防 yaw 锁死 | [我提案] |
| 4 | `sample_swing_type_initial` | discrete uniform({forehand, backhand}) | **独立采样** | ✓ V-B + #16 |
| 5 | `sample_first_cmd` | 3.2 | 立刻生成首组 cmd | [user-decided #13] |

### 7.3 Mode `interval` (周期性) + cmd noise [v5.5 A10 重写: cmd noise 改为 swing-resample-time 一次冻结, **不**进 interval, Actor 端注入 / Critic 端 clean]

#### 7.3.1 Mode `interval` (真正周期性的项, 仅 push)

| # | 项 | 间隔 | 公式 (终值, curriculum 触发后) | 备注 |
|---|---|---|---|---|
| 1 | `push_robot_velocity` | 每 5–10s | base lin vel `±0.3 m/s`, ang vel `±0.2 rad/s` | 鲁棒性扰动, 与 cmd 无关 |

[v5.5 A10 用户决定]: 把 v5 表里 #2–#5 的 `cmd_noise σ_*` **从 interval 移出**, 详见 7.3.2. 它们不是 "周期性 per-step 注入", 而是 "每次 cmd 重采时一次冻结到下次重采".

#### 7.3.2 cmd noise (一次冻结 per swing) [v5.5 A10]

**用户原话**: "critic 看到的是没加噪声的, 但是 actor 看到的是加入噪声的! 而且这个噪声每次在生成时才会加入一次, 而不是在每个 step 都加入噪声的."

**采样时机**: 每次 cmd 重采样 (= `t_to_hit ≤ -t_post_swing` 触发 + episode reset 的首组 cmd) 时, **一次性**采出 4 路噪声向量, 存到 cmd 内部字段 (`cmd.noise_p / noise_v / noise_base / noise_t`), 在整个 swing 持续期间**不变**, 直到下次重采才换新.

**注入位置**: 仅 Actor obs (asymmetric AC). Critic obs / r_g / r_g_base / strike window gate (`abs(t_to_hit)<=0.06s`) 全部用**真值 cmd** (无噪声), 这样 value head + reward signal 不被噪声扰动, 训练信号干净.

| # | 噪声项 | 采样时机 | 公式 (终值, curriculum 触发后) | 注入位置 |
|---|---|---|---|---|
| 1 | `cmd.noise_p` (xyz, 独立) | 重采 cmd 时 (= 每 swing 一次) | **gauss(σ=0.005 m), clip [-0.015, 0.015]** | 仅 Actor obs `obs_p̂_racket` (2 #5) |
| 2 | `cmd.noise_v` (xyz, 独立) | 同上 | **gauss(σ=0.05 m/s), clip [-0.15, 0.15]** | 仅 Actor obs `obs_v̂_racket` (2 #6) |
| 3 | `cmd.noise_base` (xy, 独立) | 同上 | **gauss(σ=0.015 m), clip [-0.045, 0.045]** | 仅 Actor obs `p̂_base − p_base` (2 #4) |
| 4 | `cmd.noise_t` (scalar) | 同上 | **gauss(σ=0.005 s), clip [-0.015, 0.015]** | 仅 Actor obs `t_to_hit` (2 #7) |

**统一原则**: 所有 4 路 noise 都按 "gauss σ : clip = 1 : 3" (= 3σ 截断, 工程标准). curriculum 控 σ 的 ramp-up (0 → terminal value) 见 8.3 / 8.5.

**实现伪代码** [v5.5 A10]:
```python
# (1) cmd 重采时 (commands.py.resample), 一次性冻结 noise (Mode reset 也走这条):
def resample(cmd):
    cmd.swing_type = uniform({forehand, backhand})
    cmd.p̂_racket_world = ...                                 # 见 3.2
    cmd.v̂_racket_world = ...
    cmd.p̂_base_xy_world = ...
    cmd.t_pre_initial   = truncN(...)
    cmd.t_post_swing    = truncN(...)
    cmd.t_to_hit        = cmd.t_pre_initial
    cmd.cur_step        = 0
    # ↓ A10 新增: 一次性采 noise, 整个 swing 不变
    cmd.noise_p    = clip(gauss(0, σ_p_now,    size=3), -3*σ_p_now,    3*σ_p_now)
    cmd.noise_v    = clip(gauss(0, σ_v_now,    size=3), -3*σ_v_now,    3*σ_v_now)
    cmd.noise_base = clip(gauss(0, σ_base_now, size=2), -3*σ_base_now, 3*σ_base_now)
    cmd.noise_t    = clip(gauss(0, σ_t_now,    size=1), -3*σ_t_now,    3*σ_t_now)

# (2) Actor obs (observations.py.actor_obs), 注入 noise:
obs_actor.p̂_racket  = R_base^T · ((cmd.p̂_racket_world + cmd.noise_p) − p_base_world)
obs_actor.v̂_racket  =              cmd.v̂_racket_world + cmd.noise_v
obs_actor.base_err  = (cmd.p̂_base_xy_world + cmd.noise_base) − p_base_xy_world
obs_actor.t_to_hit  = cmd.t_to_hit + cmd.noise_t

# (3) Critic obs / r_g / r_g_base / gate (rewards.py + observations.py.critic_obs):
obs_critic.p̂_racket = R_base^T · (cmd.p̂_racket_world − p_base_world)   # clean
r_g_pos = exp(-‖p_blade^base − cmd.p̂_racket^base‖² / σ²) · 𝟙[abs(cmd.t_to_hit)<=0.06]   # clean
r_g_base = exp(-‖p_base_xy_world − cmd.p̂_base_xy_world‖² / σ²)        # clean
```

⚠️ **DIVERGENCE R [新, v5.5 A10]**: paper §V-B3 列了 "obs noise" 但**未明** noise 是 per-step 还是 per-swing 冻结. 我们选 per-swing 冻结是用户决定 — 物理直觉: 感知系统给球的预测在一次击球内是相对稳定的 (球的轨迹外推一旦计算就不会大幅抖动), per-step 重采反而会让 Actor 学不到稳定的因果. Critic clean 是 asymmetric AC 标准做法 (privileged critic), 与 paper §V-B "value head sees full state" 一致.

#### 7.3.3 reward / gate 端的真值约定 [v5.5 A10 强约束]

实现端必须严格遵守:

- **r_g_pos / r_g_vel / r_g_ori** 计算用 `cmd.p̂_racket_world` / `cmd.v̂_racket_world` (无噪声真值)
- **r_g_base** 计算用 `cmd.p̂_base_xy_world` (无噪声真值)
- **strike window gate** `abs(cmd.t_to_hit) <= 0.06s` 用 `cmd.t_to_hit` (无噪声真值)
- **重采样边界** `cmd.t_to_hit <= -cmd.t_post_swing` 用真值 (无噪声), 防止 noise 把重采时间提前/推后
- **Critic obs (12 项 #4–#7 / #11–#14)** 全部 clean
- **Actor obs (10 项 #4–#7 + #1–#3 / #8–#10)** 仅 #4 / #5 / #6 / #7 用 noisy cmd, 其他 (`q`, `q̇`, `a_last`, IMU) 是 sensor 真值 (走 `randomize_imu_offset` + `comm_delay` 那条 startup DR, 不在 cmd noise 范畴)

⚠️ 如果实现端不小心让 r_g 看到 noisy cmd, reward landscape 会被噪声扰动, σ_g_pos curriculum 会失效. **代码 review 必须确认**: 任何 reward 函数读 cmd 时都不读 `cmd.noise_*`.

### 7.4 RSI [v5.5 A7 简化: 删除 mimic_start_prob 分支, 每个 episode 必从 ref clip 起步]

**v5.5 A7 用户决定**: 删除 `mimic_start_prob` 与 `pose_src ~ {ref_clip, default_standing}` 双分支. 整个工程**每个 episode 都有专家数据参与**, 不存在 "free 起步" 概念. RSI 直接从 ref clip 内随机帧起步.

```python
# Step 1: 独立采样 swing_type (cmd 字段, 7.2 #4)
swing_type ~ uniform({forehand, backhand})

# Step 2: 由 swing_type 决定 ref_clip — 不再独立采样
ref_clip = expert_clip[swing_type]      # forward_001 or backward_004 (见 11.4)

# Step 3: RSI — 必从 ref clip 内随机一帧起步 (无 mimic_start_prob 分支)
f_init ~ uniform(0, T_clip - 1)         # T_clip = ref_clip.length (forward=82, backward=64)
robot.set_state(ref_clip[f_init])       # joint pos / vel / base pose / base vel 全部从 ref 取
cur_step = 0                             # ref 时间从 cur_step=0 重新对齐 cmd 时间 (见 11.5 DIVERGENCE M)
                                         # 注意: 物理姿态 = ref[f_init], 但 ref 进度 = clip[0] 起算

# Step 4: 生成首组 cmd (含 t_to_hit, 详见 3.2)
cmd = sample_cmd_from_3.2(swing_type=swing_type)
```

⚠️ **DIVERGENCE M — RSI 解耦 (保留 v5.4 语义)**: paper / DM 标准 RSI 是 "物理姿态 = ref 同帧, ref 进度跟随". 我们解耦 "姿态采样帧 f_init" vs "ref 时间 cur_step". `cur_step=0` 强制 ref 进度从 clip 头开始, 因为 cmd.t_to_hit 也是从 0 起算, ref clip 进度需与 cmd 时间同步. 物理姿态 ref[f_init] 给出多样性起手位 (类似 DM RSI 的随机起步), ref 跟踪信号则统一从 clip[0] 开始.

⚠️ **DIVERGENCE M' [新, v5.5 A7]**: paper 没明指 RSI 概率 = 1.0 (paper §V-B 仅说 "we use reference state initialization"). 我们删除 mimic_start_prob 是 [user-decided]: 整个工程不存在 "free 起步" 阶段, 全部 episode 跟踪 ref. 风险: policy 可能学不到 "野生起步" 的鲁棒性 — 但 cmd 噪声 (7.3) + DR (7.1) 提供另一层鲁棒性, 训练第一轮 monitor episodes 在 mimic 跟踪段是否有过拟合到 ref 起步姿态的迹象.

---

## 8. Curriculum 总览表 [v5.1 修订, 用户 #26/27/28/29]

### 8.0 σ 参数语义说明 (用户 #29: 防混淆)

本 plan 出现 3 类 σ, 实现意义完全不同:

| σ 类型 | 含义 | 出现位置 | 示例 |
|---|---|---|---|
| **Reward kernel σ** | Gaussian kernel `exp(-d²/σ²)` 的衰减半径. σ 越小奖励越严. **不是采样基线**, 是 reward 公式参数. | r_g_pos, r_g_vel, r_g_ori, r_g_base (1, 4.3) | σ_g_pos = 0.02 m: 击球距离误差 d=2cm 时 reward = e⁻¹ ≈ 0.37 |
| **采样标准差 σ_sampling** | Gaussian 采样分布 `N(0, σ²)` 的 std-dev. 是采样基线 — **每次 swing resample 从此分布抽一次, 冻结到下次重采** (v5.5 A10, 不是每 step). | cmd noise σ_p, σ_v, σ_base, σ_t (7.3) | σ_p = 0.005 m: 高斯采样 `noise ~ N(0, 0.005²)`, 然后 clip ±0.015 |
| **DR 采样区间** | uniform / gauss 采样的上下界 (非 σ, 但常被叫 "noise level") | 关节摩擦/阻尼/质量 DR (7.1) | `add_link_mass: U(±10%)` 是区间, 不是 σ |

**特别提醒**: 8 表里的 σ_g_pos 是 **reward kernel σ** (公式参数, 课程收紧 = 任务变难), 而 σ_p/σ_v/σ_base/σ_t 是 **采样标准差** (噪声幅度, 课程放大 = 训练更鲁棒). 两者方向相反.

### 8.1 σ_g_pos curriculum [v5.2 修订: 加 0.03 中段防 0.04→0.02 跳幅过大]

**类型**: Reward kernel σ (`exp(-d²/σ²)` 公式参数). 单调**只收紧**.

**v5.2 合理性 review** (用户 #30): 原 v5.1 4 阶段 0.10→0.06→0.04→0.02. 问题: 0.04→0.02 是任务本质上变 4× 难 (kernel 衰减距离的平方反比), 且跨过 75% 阈值, policy 容易卡在该段. 加 0.03 中段, 让最后两段台阶更均匀.

| 阶段 | σ_g_pos | 触发条件 (击球成功率, 1k iter sliding window) | 备注 |
|---|---|---|---|
| 初始 | **0.10** m | iter 0 | warmup, 10cm 容差 |
| stage 1 | linear `0.10 → 0.06` | 击球成功率 ≥ 30% | 任务大致学会 |
| stage 2 | linear `0.06 → 0.04` | 击球成功率 ≥ 50% | 4cm 中段 [v5.1 #26] |
| stage 3 | linear `0.04 → 0.03` | 击球成功率 ≥ 65% | **新增 3cm 缓冲** [v5.2 #30] |
| stage 4 | linear `0.03 → 0.02` | 击球成功率 ≥ 80% | **终值 2cm = 噪声极限** [v5.1 #28] |
| 锁定 | **0.02** m | 达成 | 不再放宽 |

**为什么 0.02 是噪声极限** [#28]: cmd noise σ_p = gauss(0.005) clip(±0.015). 实际感知噪声 RMS ≈ 0.005 m. σ_g_pos < 4× σ_p (= 0.02m) 时, reward landscape 被噪声主导, policy 学不到稳定信号.

**击球成功率定义**: `r_g_pos > 0.5` (即 `‖p_blade − p̂_racket‖ < σ_g_pos · √(ln 2)` ≈ 0.83 σ) 在 strike window (`abs(t_to_hit) ≤ 0.06s`) 内至少 1 帧达成.

**单调性约束**: σ_g_pos **只升不降** (= 只收紧不放宽). 即使下一段成功率回落, σ_g_pos 也不返回上一段值. 防止与 8.3 噪声耦合发散.

### 8.2 [v5.5 A7 删除] mimic_start_prob curriculum

**v5.5 A7 用户决定**: 删除. 整个工程**每个 episode 都从 ref clip 起步** (见 7.4), 不存在 "free 起步" 概念, 该 curriculum 无意义. survival rate 仅用于 monitor (训练日志), 不再驱动任何课程.

### 8.3 cmd noise curriculum [v5.2 修订: 触发改用击球成功率, 删除 σ_g_pos 间接 trigger]

**类型**: 采样标准差 σ_sampling (`gauss(0, σ²)` 然后 clip ±3σ). 单调**只升**.

**触发原则** (用户原话: "在我能够稳定击球后添加噪声增强鲁棒性"): 直接用**击球成功率**作为 metric, 不再用 σ_g_pos 当中介.

| # | 噪声项 | 采样基线 σ (终值) | clip 范围 (终值, ±3σ) | 触发 (击球成功率) | 来源 |
|---|---|---|---|---|---|
| 1 | `σ_t` (时间) | gauss σ = **0.005** s | ±0.015 s | **≥ 50%** (最早开, 时间扰动是基础) | [#21 精神保留] |
| 2 | `σ_p` (位置 xyz) | gauss σ = **0.005** m | ±0.015 m | **≥ 75%** (击球已稳定再加感知噪声) | [#17 解读 B] |
| 3 | `σ_v` (速度 xyz) | gauss σ = **0.05** m/s | ±0.15 m/s | ≥ 75% (与 σ_p 同) | [#22] |
| 4 | `σ_base` (基座 xy) | gauss σ = **0.015** m | ±0.045 m | ≥ 75% (与 σ_p 同) | [#22] |

**调度形式**: 触发后 1k iter linear 从 0 → 终值. 触发后**单调不退**.

**注入时机 [v5.5 A10 重要修订]**: σ_*_now (curriculum 控) **仅在 swing resample 时读取并采一次, 冻结整个 swing**, 不在每 step 注入. 详见 [§7.3.2](#73-mode-interval-周期性--cmd-noise-v55-a10-重写-cmd-noise-改为-swing-resample-time-一次冻结-不进-interval-actor-端注入--critic-端-clean). 即:

```python
# (a) 每 iter 由 curriculum 更新 σ_*_now (训练循环外层):
σ_p_now    = curriculum.update(...)        # 每个 PPO iter 读击球成功率, 更新到下次 resample 用
σ_v_now    = curriculum.update(...)
σ_base_now = curriculum.update(...)
σ_t_now    = curriculum.update(...)

# (b) swing 重采样那一 step (= cmd.t_to_hit ≤ -cmd.t_post_swing 触发的 events.interval):
def resample_cmd(cmd, σ_p_now, σ_v_now, σ_base_now, σ_t_now):
    cmd.swing_type / cmd.p_racket_hat / ... = sample_§3.2(...)
    # 同时把噪声采一次冻结到 cmd 字段:
    cmd.noise_p    = clip(gauss(0, σ_p_now,    size=3), -3*σ_p_now,    3*σ_p_now)
    cmd.noise_v    = clip(gauss(0, σ_v_now,    size=3), -3*σ_v_now,    3*σ_v_now)
    cmd.noise_base = clip(gauss(0, σ_base_now, size=2), -3*σ_base_now, 3*σ_base_now)
    cmd.noise_t    = clip(gauss(0, σ_t_now,    size=1), -3*σ_t_now,    3*σ_t_now)

# (c) 每 step Actor obs 端用冻结的 noise 直接加 (不再每 step 重新 gauss):
obs_actor.p̂_racket = R_base^T · ((cmd.p̂_racket_world + cmd.noise_p) − p_base_world)
obs_actor.v̂_racket = cmd.v̂_racket_world + cmd.noise_v
obs_actor.p̂_base_err = (cmd.p̂_base_xy_world + cmd.noise_base) − p_base_xy_world
obs_actor.t_to_hit = cmd.t_to_hit + cmd.noise_t

# (d) Critic obs / r_g / strike_window_gate 全部用 cmd.* 真值, 不读 cmd.noise_*  (见 §7.3.3 truth-value contract)
```

⚠️ 与 v5.2 的差异: v5.2 写的"每 step 调用 cmd_noise_obs(cmd, σ_now)"被 [v5.5 A10] 否决. 关键原因: paper 没明确指定 noise 频率, 但物理上 perception (球桌 / 来球) 是按"事件" (= swing 任务下达) 一次性给定的, 不是每控制 step 重新采样. 把噪声跟 swing 绑定 (per-swing freeze) 比 per-step gauss 更接近真实部署. 详见 §7.3.2 DIVERGENCE R.

**为何 σ_t 触发更早 (≥50%)**: 时间扰动比位置更"基础" (perception 总有几 ms 偏差), 早开始训练对 t_to_hit 的鲁棒性.

### 8.4 Hit point / v_mag 范围扩展 curriculum [v5.2 修订: 改成击球成功率驱动]

**类型**: uniform 采样区间上下界. 范围 monotone **扩展** (不收紧).

**触发原则** (用户决定): 也用击球成功率, 与 8.1/8.3 统一 metric.

| 参数 | 初始 (≥0%) | ≥30% | ≥50% | ≥75% (终值) |
|---|---|---|---|---|
| `hit_y` | `±0.5` m | linear → `±0.7` | linear → `±0.85` | linear → **`±1.0`** m |
| `hit_z` | `[0.20, 0.45]` | → `[0.15, 0.50]` | → `[0.12, 0.55]` | → **`[0.08, 0.60]`** |
| `v_mag` | `[2.5, 4.5]` m/s | → `[2.2, 5.0]` | → `[2.0, 5.5]` | → **`[2.0, 6.0]`** m/s |

**调度形式**: 每个阶段触发后 1k iter linear 推到该阶段目标值. 单调扩展, 不收回.

**与 8.1 σ_g_pos 同步**: 击球成功率 ≥30% 同时触发 σ_g_pos stage1 + hit/v 范围扩展. 训练 metric 统一, 防止课程间错位.

### 8.5 Curriculum 整合视图 (单表) [v5.5 A7: 删除 #2 mimic_start_prob, 8 项]

| # | 名称 | σ 类型 | 起 → 终 | 触发 metric | 单调方向 | 实现 (curriculums.py 函数) |
|---|---|---|---|---|---|---|
| 1 | σ_g_pos | reward kernel σ | 0.10 → **0.02** m | 击球成功率 (30/50/65/80%) | 只收紧 (升) | `update_g_pos_sigma` |
| ~~2~~ | ~~mimic_start_prob~~ | — | **删除** [v5.5 A7] | — | — | (无, 见 8.2) |
| 2 | σ_t (cmd 时间) | 采样标准差 | 0 → **0.005** s | **击球成功率 ≥ 50%** (早) | 只升 | `update_cmd_noise_sigma_t` |
| 3 | σ_p (cmd 位置) | 采样标准差 | 0 → **0.005** m | 击球成功率 ≥ 75% | 只升 | `update_cmd_noise_sigma_p` |
| 4 | σ_v (cmd 速度) | 采样标准差 | 0 → **0.05** m/s | 击球成功率 ≥ 75% | 只升 | `update_cmd_noise_sigma_v` |
| 5 | σ_base (cmd 站位) | 采样标准差 | 0 → **0.015** m | 击球成功率 ≥ 75% | 只升 | `update_cmd_noise_sigma_base` |
| 6 | hit_y range | uniform 区间 | ±0.5 → **±1.0** m | 击球成功率 (30/50/75%) | 扩展 | `update_hit_y_range` |
| 7 | hit_z range | uniform 区间 | `[0.20,0.45]` → **`[0.08,0.60]`** | 击球成功率 (30/50/75%) | 扩展 | `update_hit_z_range` |
| 8 | v_mag range | uniform 区间 | `[2.5,4.5]` → **`[2.0,6.0]`** m/s | 击球成功率 (30/50/75%) | 扩展 | `update_v_mag_range` |

**统一原则** (v5.5): 所有 curriculum 都用**击球成功率**作为 metric (8.2 mimic_start_prob 删除后, 不再有 survival 驱动的课程), 阶梯阈值 30/50/65/75/80% 之间互相协调:
- 击球成功率 ≥30%: σ_g_pos stage1 (→0.06) + 范围 stage1 扩展
- ≥50%: σ_g_pos stage2 (→0.04) + 范围 stage2 + **σ_t 触发**
- ≥65%: σ_g_pos stage3 (→0.03)
- ≥75%: 范围全开 + **σ_p/σ_v/σ_base 触发**
- ≥80%: σ_g_pos stage4 (→0.02 终值)

⚠️ 8.4 三条**不**改单调 — 范围本来就是收紧→放宽, 与 σ_g_pos 收紧方向相反.

### 8.6 第一轮训练只开

只开 **#1 σ_g_pos** 一条 (核心难度) [v5.5 A7: 原来 #1 + #2, 删 #2 后只剩 #1]. 其他 (cmd noise + 范围扩展) 留作 ablation; 训练第一轮可保持 hit_y/hit_z/v_mag 终值 (`±1.0` / `[0.08, 0.60]` / `[2.0, 6.0]`) 直接训, 验证 baseline 稳定后再加 cmd noise.

---

## 9. Scene 总览表 [v5.1: 加 Table asset + 双 ContactSensor]

### 9.1 InteractiveSceneCfg

| # | 项 | 取值 / 配置 | 备注 |
|---|---|---|---|
| 1 | `num_envs` | 4096 (训练) / 1 (play) | IsaacLab 标准 |
| 2 | `env_spacing` | 4.0 m | env 间隔 |
| 3 | `terrain` | `flat_plane` | ✓ paper V (室内球场) |
| 4 | `robot` | `g1_23dof_paddle` (#24) | 见 9.3 |
| 5 | `table` (静态 RigidBody) | `AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Table", spawn=CuboidCfg(size=(2.74,1.525,0.76), rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True), collision_props=CollisionPropertiesCfg()), init_state=InitialStateCfg(pos=(1.77, 0.0, 0.38)))` | [v5.1 #25] 国际标准球桌; 近端桌沿 x=0.4 = paper 击球点; 桌面 top z=0.76; **kinematic 不动** |
| 6 | `contact_forces` (self-contacts) | `ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0)` [参 [tracking_env_cfg.py:75-81](source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/dance_102/tracking_env_cfg.py#L75-L81)] | **sensor #1**: 自接触, 服务于 feet_slide / feet_air_time / undesired_contacts / undesired_contact_terminate |
| 7 | `robot_table_contact` (robot↔table) | `ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"], history_length=3)` | **sensor #2**: 通过 `filter_prim_paths_expr` 只检测 robot vs Table 接触, 服务于 6.4.3 `r_table_contact` reward |

### 9.2 ContactSensor 用法 (双 sensor, 各司其职)

**sensor #1 `contact_forces`** (覆盖全 robot body, 用 `body_names` 正则过滤):
- `feet_slide` / `feet_air_time`: `SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"])`
- `undesired_contacts` (软惩罚): `body_names=["(?!.*ankle.*).*"]` (排除脚, locomotion 风格); 或 mimic 风格排除 `[ankle, wrist]` 后惩罚其他
- `undesired_contact_terminate` (硬终止, 6.1 #4): `body_names=["pelvis", "head_link", ".*_hip_pitch_link"]`

**sensor #2 `robot_table_contact`** (robot 全部 body 与 Table 的接触, 已被 `filter_prim_paths_expr` 限定):
- `r_table_contact` (6.4.3): `SceneEntityCfg("robot_table_contact", body_names=".*")` — 任何 robot body 撞 table 即罚, 含 `right_paddle_blade` (球拍 STL collision 已包含在 URDF 里, 物理引擎自动处理几何)

### 9.3 Robot asset (G1 23-dof + paddle) [v5 #24]

| # | 项 | 配置 |
|---|---|---|
| 1 | URDF | **`/home/woan/HumanoidProject/unitree_rl_lab/unitree_ros/robots/g1_description/g1_23dof_rev_1_0_paddle.urdf`** [v5 #24, 与 [unitree.py:951](source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py#L951) `UNITREE_G1_23DOF_PADDLE_MIMIC_CFG` 一致] |
| 2 | Asset cfg 复用 | `UNITREE_G1_23DOF_PADDLE_MIMIC_CFG` (含 stiffness / damping / armature / effort_limit, 与 5.2 action_scale 配套) |
| 3 | Base init pos | `(0, 0, 0.76)` (与 mimic 配置一致) |
| 4 | Base init quat | `(1, 0, 0, 0)` wxyz |
| 5 | Joint init values | 复用 `UNITREE_G1_23DOF_MIMIC_CFG.init_state.joint_pos` (unitree.py:751+) |
| 6 | Body 总数 | 25 (含 pelvis + paddle blade) |

⚠️ paddle URDF rpy: `right_paddle_blade_fixed_joint` 已嵌入 -45° rpy 到 body world quat, 不重复加.

---

## 10. Sim 总览表 **[v5 不变]**

| # | 项 | 取值 | paper? |
|---|---|---|:---:|
| 1 | `sim.dt` | `1/200` s | [我提案] (G1 标准) |
| 2 | `decimation` | **4** | ✓ paper V "50Hz" |
| 3 | `episode_length_s` | **10.0** s | ✓ paper V-B1 |
| 4 | `gravity` | `(0, 0, -9.81)` | ✓ |
| 5 | `solver_position_iter` | 4 | PhysX 默认 |
| 6 | `solver_velocity_iter` | 1 | PhysX 默认 |
| 7 | `friction_combine_mode` | `"multiply"` | 通用 |
| 8 | `restitution_combine_mode` | `"multiply"` | 通用 |

---

## 11. 歧义消解记录 [v5.2 user-decided Q1–Q4, 实现端必须严格遵守]

本节记录 v5.2 self-review 期间发现的 4 处文档歧义, 以及用户的最终裁决. **代码实现必须严格按本节规范, 否则 reward 信号会与 obs 不一致, 导致训练失败.**

### 11.1 [Q1] r_i ref clip 击球帧对齐 + 双段时间比例插值缩放 [v5.4 user-decided, 替代 v5.2/v5.3 的"末尾冻结"]

**问题**: r^p / r^v / r^bp 在 1 #1/#2/#3 写"dense 全程", 但 ref clip 长度有限 (forward_001=82 帧≈1.64s, backward_004=64 帧≈1.28s), 而 sim 端 cmd 采样的 swing 时长 (pre + post) 与 ref 自身的击球前后段时长不一致. 怎么对齐?

**用户决定 v5.4** (用户原话: "首先你需要记录正反手的专家数据在击球前后的时间, 然后这肯定和我们采样得到的时间有差距的, 那么就需要根据这个时间比例进行插值/抽帧了 (但是击球帧肯定要一致, 以击球帧重合为中心, 将时间分成击球前后两段时间)"):

1. cmd 加新字段 `t_post_swing` (post-strike sim 时长), **独立 truncN 采样**, 不进 obs
2. **击球帧对齐**: sim 端 t_to_hit=0 那一 step 必对齐 ref 的 impact_frame
3. **双段独立线性插值缩放**:
   - **pre-strike**: sim_step ∈ [0, sim_pre_steps] ↔ ref_frame ∈ [0, impact_frame] 线性插值
   - **post-strike**: sim_step ∈ [sim_pre_steps, sim_pre_steps + sim_post_steps] ↔ ref_frame ∈ [impact_frame, clip_len-1] 线性插值
4. 一次 swing 用完 (sim_pre_steps + sim_post_steps step) 后立即重采样 cmd (含 t_to_hit + t_post_swing 都重采), cur_step 复位 0
5. **重要 — ref state 用浮点 ref_frame 做线性插值** (joint pos/vel, body pos/lin_vel/ang_vel 用 lerp; body quat 用 slerp). 不再用整数 cur_frame 直接索引

#### 11.1.1 ref clip 击球前后时长 (一次性预处理, 写到 expert_clip dict)

| clip | clip_len | impact_frame | pre 帧数 | pre 时长 (50 fps) | post 帧数 | post 时长 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| forward_001 | 82 | 37 | 37 | **0.74 s** | 44 | **0.88 s** |
| backward_004 | 64 | 20 | 20 | **0.40 s** | 43 | **0.86 s** |

```python
# 一次性预处理 (motion_loader.__init__):
for swing_type, npz_path in [...]:
    d = np.load(npz_path)
    impact = int(d["impact_frame"][0])
    clip_len = d["joint_pos"].shape[0]
    fps = int(d["fps"][0])               # = 50
    expert_clip[swing_type] = MotionClip(
        joint_pos       = d["joint_pos"],
        joint_vel       = d["joint_vel"],
        body_pos_w      = d["body_pos_w"],
        body_quat_w     = d["body_quat_w"],
        body_lin_vel_w  = d["body_lin_vel_w"],
        body_ang_vel_w  = d["body_ang_vel_w"],
        impact_frame    = impact,
        clip_len        = clip_len,
        pre_duration    = impact / fps,           # native pre 时长
        post_duration   = (clip_len - 1 - impact) / fps,  # native post 时长
    )
```

#### 11.1.2 实现规范 (motion_loader, 每 step 调用)

```python
def get_ref_state(cmd: PingpongCommand, dt: float = 0.02):
    """计算 ref state at sim step cmd.cur_step, 用浮点 ref_frame + 线性插值."""
    clip = expert_clip[cmd.swing_type]
    impact = clip.impact_frame
    clip_len = clip.clip_len

    # sim 两段时长 (step 数, 浮点)
    sim_pre_steps  = cmd.t_pre_initial / dt    # 例如 1.5s / 0.02 = 75 step
    sim_post_steps = cmd.t_post_swing  / dt    # 例如 0.6s / 0.02 = 30 step

    # 当前 sim step 在两段中的位置, 映射到 ref_frame (浮点)
    if cmd.cur_step <= sim_pre_steps:
        # pre-strike: sim_step ∈ [0, sim_pre_steps] → ref_frame ∈ [0, impact]
        progress = cmd.cur_step / max(sim_pre_steps, 1.0)   # 防 div 0 (sim_pre_steps≥10 因 t≥0.2s)
        ref_frame_f = progress * impact
    else:
        # post-strike: sim_step ∈ (sim_pre_steps, sim_pre+sim_post] → ref_frame ∈ (impact, clip_len-1]
        sim_post_step = cmd.cur_step - sim_pre_steps
        progress = sim_post_step / max(sim_post_steps, 1.0)
        ref_frame_f = impact + progress * (clip_len - 1 - impact)

    ref_frame_f = float(np.clip(ref_frame_f, 0.0, clip_len - 1))   # 安全 clamp

    # 线性插值 ref state (lerp for vec, slerp for quat)
    f_low  = int(np.floor(ref_frame_f))
    f_high = min(f_low + 1, clip_len - 1)
    α      = ref_frame_f - f_low

    ref_q       = (1-α)*clip.joint_pos[f_low]      + α*clip.joint_pos[f_high]
    ref_qd      = (1-α)*clip.joint_vel[f_low]      + α*clip.joint_vel[f_high]
    ref_p_world = (1-α)*clip.body_pos_w[f_low]     + α*clip.body_pos_w[f_high]      # (12, 3)
    ref_v_world = (1-α)*clip.body_lin_vel_w[f_low] + α*clip.body_lin_vel_w[f_high]  # (12, 3)
    ref_w_world = (1-α)*clip.body_ang_vel_w[f_low] + α*clip.body_ang_vel_w[f_high]  # (11, 3)
    ref_q_world = quat_slerp(clip.body_quat_w[f_low], clip.body_quat_w[f_high], α)  # (11, 4) wxyz
    return ref_q, ref_qd, ref_p_world, ref_q_world, ref_v_world, ref_w_world
```

#### 11.1.3 cmd 重采样时机 [v5.4 替换 v5.2 "t_to_hit ≤ 0 立即重采样"]

```python
def cmd_step(cmd, dt=0.02):
    cmd.t_to_hit -= dt
    cmd.cur_step += 1
    # 一次 swing 完整 = sim_pre_steps + sim_post_steps 走完 (= t_to_hit ≤ -t_post_swing)
    if cmd.t_to_hit <= -cmd.t_post_swing:
        resample(cmd)              # swing_type / hit / v̂ / t_to_hit / t_pre_initial / t_post_swing 全部重采
        cmd.cur_step = 0           # ref 从 clip[0] 重新对齐
```

注意: cmd.t_to_hit 在 pre-strike 阶段从 t_pre_initial 单调减到 0, post-strike 阶段继续减到 -t_post_swing. **strike window gate `abs(cmd.t_to_hit) ≤ 0.06s` 不变** — 仍以 t_to_hit=0 为中心 ±3 帧, 自然落在 sim_pre_steps 与 sim_post_steps 交界处 (= ref 的 impact 帧附近).

#### 11.1.4 数字举例 [v5.4 重写, 替换 v5.3 末尾冻结举例]

**例 A — 长 pre + 短 post, backhand**:
```
swing_type      = backhand          (clip_len=64, impact=20, pre=0.40s, post=0.86s native)
t_pre_initial   = 1.50s             (sim_pre_steps = 75)
t_post_swing    = 0.30s             (sim_post_steps = 15)
total swing     = 1.80s             (90 step), 然后重采样

step  cur_step  t_to_hit  ref_frame_f                                  说明
─────────────────────────────────────────────────────────────────────────────────
   0       0     1.50    progress=0/75=0,    ref_f = 0×20 = 0.0       clip[0], 起手
  37      37     0.76    progress=37/75=0.49, ref_f = 0.49×20 = 9.87  clip[9]+α=0.87 lerp clip[10]
  75      75     0.00    progress=75/75=1.0,  ref_f = 1×20 = 20.0     clip[20] = impact 帧 (击球!)
  76      76    -0.02    progress=(76-75)/15=0.067, ref_f=20+0.067×(63-20)=22.87  clip[22]+α=0.87 lerp clip[23]
  90      90    -0.30    progress=15/15=1.0, ref_f=20+1.0×43=63       clip[63] = clip 末尾 (post 完)
─── t_to_hit ≤ -t_post_swing = -0.30, 立即重采样 cmd ───
  91       0     1.10*    重新采 swing_type / pre / post                cur_step 复位
*  新 t_pre_initial=1.10s, t_post_swing 重采, 都从 truncN 抽
```

机制: pre-strike 的 75 step (1.50s) 对应 ref pre 段的 0→20 帧, **ref 被 1.50/0.40 = 3.75× 慢放**. post-strike 的 15 step (0.30s) 对应 ref post 段的 20→63 帧, **ref 被 0.30/0.86 = 0.349× 加速**.

**例 B — 短 pre + 长 post, forehand**:
```
swing_type      = forehand         (clip_len=82, impact=37, pre=0.74s, post=0.88s native)
t_pre_initial   = 0.30s             (sim_pre_steps = 15)
t_post_swing    = 1.20s             (sim_post_steps = 60)
total swing     = 1.50s             (75 step)

step  cur_step  t_to_hit  ref_frame_f                                  说明
─────────────────────────────────────────────────────────────────────────────────
   0       0     0.30    ref_f=0                                       clip[0]
  15      15     0.00    ref_f=15/15×37=37                              clip[37] = impact (击球)
  16      16    -0.02    ref_f=37+(1/60)×(81-37)=37.73                  clip[37]+α=0.73 lerp clip[38]
  75      75    -1.20    ref_f=37+(60/60)×44=81                         clip[81] = clip 末尾
─── 重采样 ───
```

机制: pre 0.30s ref 被 0.30/0.74 = 0.405× 加速 (ref 跳着播); post 1.20s ref 被 1.20/0.88 = 1.36× 慢放.

**例 C — episode timeout 时**:
sim 在某次 swing 中途 (cur_step < total_steps) 遇到 step 500 (10s) timeout, GAE 用 V(s_500) bootstrap. 不需特殊处理 ref — 那一帧 ref 还在 swing 中段, dense 计算 r_i 即可.

#### 11.1.5 quat_slerp 实现说明

```python
def quat_slerp(q0, q1, α):
    # q0, q1 shape (..., 4), wxyz convention
    dot = (q0 * q1).sum(-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)        # 取近端
    dot = abs(dot).clip(-1.0, 1.0)
    θ = np.arccos(dot)
    # 小角度退化为 lerp 防数值不稳
    sin_θ = np.sin(θ).clip(1e-6, None)
    w0 = np.sin((1-α)*θ) / sin_θ
    w1 = np.sin(α*θ)     / sin_θ
    return w0*q0 + w1*q1
```

⚠️ **DIVERGENCE O 重写 [v5.4]**: paper 单 clip 设定下 t_strike 与 ref impact 自然对齐, 不需要时间插值. 我们 2-clip + cmd 独立采样 t_pre / t_post 与 ref pre/post native 时长不一致, **必须做双段独立线性插值缩放才能保证击球帧对齐**. 这是 2-clip pool + 自由 swing 时长设定的工程必要, 与 paper Sec V-B 不冲突 (paper 没禁止时间缩放, 只是单 clip 不需要).

---

### 11.2 [Q2] swing_type 在网络中的位置: **不进 obs, 作为 ref clip selector**

**问题**: swing_type 是独立 uniform 采样的隐变量 (7.2 #4), 2 obs 表里没列. 但 r^p/r^v/r^bp / T_B (Critic obs #12) / [q̄, q̇̄] (Critic obs #14) 都依赖 ref clip 内容, 而 ref clip 选择 = clip[swing_type]. policy 必须能区分 swing_type 才能匹配跟踪信号.

**用户决定**: 用户原话 "critic需要从专家动作中获得这个判别，因此需要swing的信息，但是不作为输入到critic的网络，而是先经过swing选择后输出需要的T". 即 swing_type **作为 ref clip selector 路由 T_B / [q̄, q̇̄], 不直接作为 obs 维度进网络**.

**实现规范**:
```python
# Critic obs #12 / #14 计算时:
selected_clip = expert_clip[cmd.swing_type]                          # 内部 dispatch
T_B  = compute_body_state(selected_clip, ref_frame_f)                # 11·7=77 维 [v5.5 A2: 排除 paddle blade]
qd_qdot_ref = compute_joint_state(selected_clip, ref_frame_f)        # 23+23=46 维
# Critic input = [..., T_B, t_left, qd_qdot_ref]   # 213 维 [v5.5 A2: 原 220, 减 7]
# Actor input  = [..., q, q_dot, a_last]            # 仍 86 维, 不变 (无 swing_type)

# Actor 通过 p̂_racket / v̂_racket 几何分布隐推 swing_type:
#   - 击球点 hit_y > 0 偏右: 多为 backhand (左手在右侧反手击)
#   - 击球点 hit_y < 0 偏左: 多为 forehand (右手在左侧正手击)
# 这是 paper V-B 的设定, Actor 不显式见 swing_type 是 paper-aligned.
```

**Actor / Critic 维度**: Actor=86, **Critic=213** [v5.5 A2: T_B 11·7=77 排除 paddle blade] (Q2 选项 C 的语义, swing_type 隐含).

**实现细节**:
- 维护 `expert_clip: Dict[str, MotionClip]` (key="forehand"/"backhand"), 加载 forward_001 + backward_004.
- 计算 T_B / [q̄, q̇̄] 时按 cmd.swing_type 索引 dict, 不进 obs tensor.
- r_i (r^p/r^v/r^bp) 计算同理: 按 swing_type 路由 ref. 

⚠️ **DIVERGENCE P — swing_type 隐含**: paper 的 single-clip 设定下 swing_type 是固定的, 不存在路由问题. 我们 2-clip 设定下用 swing_type 作 selector 是工程必要.

---

### 11.3 [Q3] r^bp 的 p_rel 计算: **xy 减 z 保留 (防蹲下作弊)**

**问题**: r^bp 公式 `exp(-k·Σ_b ‖p_rel[b] − p̂_rel[b]‖²)` 中 `p_rel[b] = p_world[b] − p_world[pelvis]` 写法不明确减是 3D 全减还是仅 xy 减. 老 v2 文档说 "xy 减 z 不减", v5 重写时丢失这条注解.

**用户决定**: **xy 减, z 保留绝对高度**.

**实现规范**:
```python
# 对每个 b ∈ ℬ_pos (11 bodies, 排除 right_paddle_blade):
# sim 端
p_rel[b] = np.array([
    p_world_sim[b].x - pelvis_world_sim.x,
    p_world_sim[b].y - pelvis_world_sim.y,
    p_world_sim[b].z                              # ← 保留绝对世界 z, 不减 pelvis.z
])

# ref 端 (motion_loader, 同样规则)
p̂_rel[b] = np.array([
    ref_p_world[b].x - ref_pelvis_world.x,
    ref_p_world[b].y - ref_pelvis_world.y,
    ref_p_world[b].z
])

# r_bp 计算
r_bp = exp(-40.0 * sum(np.linalg.norm(p_rel[b] - p̂_rel[b])**2 for b in B_pos))
```

**为什么 xy 减 z 保留**: 防止机器人通过整体蹲下来"作弊"减小 p_rel 距离. xy 减把 base 平移自由度吸收 (不同 episode pelvis xy 起点不同, 但 body 相对 pelvis 的 xy 偏移应一致); z 不减则强制机器人保持 body 的世界绝对高度跟 ref 一致 (ref 是机器人正常站立时录的, 站立高度 ≈ 0.74m 时 torso z ≈ 1.10m). 若机器人为了 r_bp 高分而蹲下, body z 会偏离 ref, r_bp 受罚.

⚠️ **跟 1 #3 公式一致**: 已 v5.2 patch 到 1 #3 的"计算/来源"列, 实现时按本节为准.

---

### 11.4 [expert_offset 与 expert_clip 来源]

为 11.1 / 11.2 / 3.2 expert_offset 的明确性补充:

**两个 expert clip 的 npz 路径**:
- forward (forehand): `motion_datasets/pingpong/humanoid_data/final/expert/forward/forward_001.npz`
- backward (backhand): `motion_datasets/pingpong/humanoid_data/final/expert/backward/backward_004.npz`
- 由 `scripts/pingpong_data_process/select_x04_clips.py` 从 GVHMR retarget 后的 npz 池筛选 (paper x=0.4 容差带), 然后用户手动挑出最标准的 1 个正手 + 1 个反手. 见 [REWARD_DESIGN.md](source/unitree_rl_lab/unitree_rl_lab/tasks/pingpong/REWARD_DESIGN.md) 3.5.

**expert_offset 一次性预处理代码 (见 3.2)** [v5.5 A9 修正: ref clip yaw ≠ 0, 改用 base-frame 计算]:
```python
expert_offset: Dict[str, np.ndarray] = {}
for swing_type, npz_path in [("forehand",  ".../forward_001.npz"),
                              ("backhand", ".../backward_004.npz")]:
    d = np.load(npz_path)
    imp = int(d["impact_frame"][0])
    blade_w  = d["body_pos_w"][imp, BLADE_IDX,  :2]   # BLADE_IDX = 24,  world xy (m)
    pelvis_w = d["body_pos_w"][imp, PELVIS_IDX, :2]   # PELVIS_IDX = 0,  world xy (m)
    pelvis_q = d["body_quat_w"][imp, PELVIS_IDX]      # wxyz
    yaw      = yaw_from_wxyz(pelvis_q)                # ref clip impact 帧 pelvis yaw (rad)

    # 世界 delta → 旋转到 base frame (= 当时 pelvis 朝向)
    diff_w = blade_w - pelvis_w                                      # (2,) world delta
    c, s = np.cos(-yaw), np.sin(-yaw)
    expert_offset[swing_type] = np.array([
        c * diff_w[0] - s * diff_w[1],                               # x in base frame
        s * diff_w[0] + c * diff_w[1],                               # y in base frame
    ])

# 实测值 (用户 A9 提供, base frame xy):
# expert_offset["forehand"]  = (0.496, 0.208)   forward_001  (impact=37,  pelvis yaw=63.6°)
# expert_offset["backhand"] = (0.428, 0.106)   backward_004 (impact=20, pelvis yaw=128.8°)
```

⚠️ **expert_offset frame 注释 [v5.5 A9 重写, 替代 v5.3 #31]**: offset 是 **base frame xy delta** (= ref clip impact 帧 blade 在当时 pelvis frame 下的 xy 偏移). v5.3 假设 "ref clip yaw ≈ 0" 是**错的** — 实测 forward_001 yaw=63.6°, backward_004 yaw=128.8° (GVHMR retarget 后 pelvis 朝向并非 +x). 因此**必须**通过 `R(yaw)^T` 把 world delta 旋到 base frame, 否则 hit_xy_world − expert_offset_world 算出来的 base_target 会偏离机器人真正应该站的位置.

⚠️ **训练端 cmd 生成代码相应修正 [v5.5 A9 cont., 涵盖 3.2]**:
```python
# 错误版本 (v5.3): base_target_xy_world = hit_xy_world − expert_offset_world  (偏差大)
# 正确版本 (v5.5):
yaw_robot = base_yaw_now                                       # 训练 robot 当前 base yaw (= 0 ± reset noise)
c, s = np.cos(yaw_robot), np.sin(yaw_robot)
offset_world_now = np.array([
    c * expert_offset[swing_type][0] - s * expert_offset[swing_type][1],
    s * expert_offset[swing_type][0] + c * expert_offset[swing_type][1],
])
base_target_xy_world = (hit_x_world, hit_y_world) - offset_world_now
```

实现端 (commands.py): 把 expert_offset 当作 base frame 常量, 训练时按 robot 当前 base_yaw 旋到 world frame, 再用 hit_xy_world 相减得到 p̂_base,xy. 这与 §3.2 中 "world 坐标减法" 注释要在 commands.py 实现端按本节修正 — §3.2 注释保留 "world delta" 视角是为了与 obs 表 (`p̂_base − p_base` world delta) 一致.

**clip 内部用到的字段** (motion_loader 加载时):
- `joint_pos` (T, 23): 关节位置
- `joint_vel` (T, 23): 关节速度
- `body_pos_w` (T, 25, 3): body 世界位置 (含 paddle_blade)
- `body_quat_w` (T, 25, 4): body 世界 quat (wxyz)
- `body_lin_vel_w` (T, 25, 3): body 世界线速度
- `body_ang_vel_w` (T, 25, 3): body 世界角速度
- `impact_frame` scalar: 击球瞬间帧索引
- `fps` scalar: 50 (csv_to_npz output_fps)
- `swing_type` scalar: 0=forehand 1=backhand

---

### 11.5 [Q4] 时间字段命名: **cmd 内部和 obs 都叫 t_to_hit (剩余击球时间)**

**问题**: v5 表里曾出现 `t_strike` (= absolute strike 时刻) 和 `t_to_hit` (= 剩余时间) 两个字段, 同名异义混淆 — paper Table I 写 `t_strike` 但语义是剩余时间.

**用户决定**: **统一用 `t_to_hit`** (剩余击球时间). cmd 内部不存绝对时刻, 直接存 t_to_hit 标量, 重采样时 t_to_hit ← Δt_swing (truncN 采样), 每 step `t_to_hit -= dt`.

**实现规范** [v5.4: 加 t_pre_initial / t_post_swing, cur_frame → cur_step]:
```python
# cmd dataclass:
@dataclass
class PingpongCommand:
    swing_type: str            # "forehand" / "backhand"  (7.2 #4 独立 uniform 采样)
    p_racket_hat: np.ndarray   # (3,) **world frame** [v5.3 #31]
    v_racket_hat: np.ndarray   # (3,) world frame
    p_base_xy_hat: np.ndarray  # (2,) world frame
    t_to_hit: float            # 标量, 剩余击球时间 (秒). pre 段从 t_pre_initial 减到 0, post 段减到 -t_post_swing
    t_pre_initial: float       # [v5.4 NEW] 重采样时采的 pre-strike sim 时长 (秒); 进 obs 作 t_to_hit 初值
    t_post_swing: float        # [v5.4 NEW] 重采样时独立采的 post-strike sim 时长 (秒); **不进 obs**, 仅控 ref 缩放 + 重采边界
    cur_step: int              # [v5.4 取代 cur_frame] sim step 计数器, 重采样复位 0
    # [v5.5 A10 NEW] noise 一次冻结 per swing, 仅 Actor obs 注入, Critic / r_g / gate 全部用真值 (无噪声)
    noise_p:    np.ndarray     # (3,) gauss(σ_p_now)    clip ±3σ; obs_actor.p̂_racket  端注入
    noise_v:    np.ndarray     # (3,) gauss(σ_v_now)    clip ±3σ; obs_actor.v̂_racket  端注入
    noise_base: np.ndarray     # (2,) gauss(σ_base_now) clip ±3σ; obs_actor.base_err  端注入
    noise_t:    float          # (1,) gauss(σ_t_now)    clip ±3σ; obs_actor.t_to_hit  端注入
    # ref_frame_f 不存! 通过 11.1.2 get_ref_state(cmd, dt) 当场 cur_step → ref_frame_f 双段插值映射
    # 不存 t_strike_absolute! 只存剩余时间.

# 每 step [v5.4: 用 -t_post_swing 边界; v5.5 A10: noise 仅在 resample 时换, 每 step 不动]:
def cmd_step(cmd, dt=0.02):
    cmd.t_to_hit -= dt                                         # 用真值递减, noise 不参与
    cmd.cur_step += 1
    if cmd.t_to_hit <= -cmd.t_post_swing:                      # post 段也走完
        resample(cmd)                                          # 重采 swing_type/p̂/v̂/t/... + noise_* 一次冻结 (见 7.3.2)
        cmd.cur_step = 0                                       # ref 从 clip[0] 重新对齐 (11.1.2)

# obs 端 [v5.5 A10: Actor noisy / Critic clean]:
obs_actor.p̂_racket  = R_base^T · ((cmd.p_racket_hat + cmd.noise_p) − p_base_world)
obs_actor.v̂_racket  =              cmd.v_racket_hat + cmd.noise_v
obs_actor.base_err  = (cmd.p_base_xy_hat + cmd.noise_base) − p_base_xy_world
obs_actor.t_to_hit  = cmd.t_to_hit + cmd.noise_t

obs_critic.p̂_racket  = R_base^T · (cmd.p_racket_hat − p_base_world)   # clean
obs_critic.v̂_racket  =             cmd.v_racket_hat                    # clean
obs_critic.base_err  = cmd.p_base_xy_hat − p_base_xy_world             # clean
obs_critic.t_to_hit  = cmd.t_to_hit                                    # clean
# t_pre_initial / t_post_swing / cur_step / noise_* 都不进 obs (内部状态)
```

**与 paper 的关系**: paper Table I 字段名 `t_strike` 在 paper 中实际就是剩余时间 (paper V-B "the time until the strike"). 我们改名为 `t_to_hit` 是为了消除"绝对/相对时刻"的歧义 — 与 paper 语义一致, 命名更清楚.

**Strike window gate**: 仍用 `abs(cmd.t_to_hit) ≤ 0.06s` (= ±3 帧 @ 50Hz). cmd.t_to_hit 在 pre 段从 t_pre_initial 单调减到 0, post 段继续减到 -t_post_swing 才重采样, 所以"负数 t_to_hit" 在 [−t_post_swing, 0] 区间内停留多个 step (= sim_post_steps 个), strike window 仍以 t_to_hit=0 ± 3 帧自然落在 ref impact 帧附近.

---

## End. v5.4 状态 (全部锁定)

| | 状态 |
|---|:---:|
| 1 Reward (含 v5.2 #Q3: r^bp 的 p_rel xy 减 z 保留) | ✓ |
| 2 Obs (含 v5.2 #Q4 t_to_hit 命名统一; #Q2 swing_type 不进 obs; v5.3 #31: p̂_racket 在 obs 端转 base-relative) | ✓ |
| 3 Cmd (#16 swing_type 独立采样 + v5.3 #31: cmd 内部全部 world frame + **v5.4 双段时长 t_pre_initial / t_post_swing + cur_step**) | ✓ v5.4 |
| 4 Weights + σ baseline | ✓ user-confirmed v4 |
| 5 Actions (mimic per-joint scale, #14) | ✓ |
| 6 Terminations (基础 4 项) | ✓ |
| 6.4 表桌 (RigidBody + ContactSensor + r_table_contact reward) | ✓ v5.1 #25 |
| 7 Events (#16 swing_type 独立 + 7.3 1:3 比例噪声 + v5.2 #Q1 RSI cur_step=0) | ✓ |
| 8 Curriculum (8.0 σ 语义 + 8.1–8.6, σ_g_pos 终值 0.02m, 5 阶段 0.10→0.06→0.04→0.03→0.02) | ✓ |
| 9 Scene (Table asset + 双 ContactSensor) | ✓ v5.1 |
| 10 Sim | ✓ |
| 11 歧义消解记录 (Q1 v5.4 重写: 击球帧对齐 + 双段插值缩放; Q2/Q3/Q4 + 11.4 expert_offset world frame 注释) | ✓ v5.4 |

### v5.4 锁定决定 (本轮 Q1 重写 + 累计)

#### v5.4 本轮新决定 (用户 Q1 重写 — 否定 v5.3 末尾冻结):

**用户原话**: "首先你需要记录正反手的专家数据在击球前后的时间, 然后这肯定和我们采样得到的时间有差距的, 那么就需要根据这个时间比例进行插值/抽帧了 (但是击球帧肯定要一致, 以击球帧重合为中心, 将时间分成击球前后两段时间)..."

- **11.1 Q1 完全重写**: 删除 v5.3 "末尾冻结" 机制. 改为**击球帧对齐 + 双段独立线性插值缩放**:
  - **击球帧对齐**: sim 的 t_to_hit=0 那一 step 必对齐 ref clip 的 impact_frame
  - **pre 段**: sim_step ∈ [0, sim_pre_steps] ↔ ref_frame ∈ [0, impact] 线性插值 (浮点 ref_frame_f)
  - **post 段**: sim_step ∈ [sim_pre_steps, total] ↔ ref_frame ∈ [impact, clip_len-1] 线性插值
  - ref state lerp (vec) / slerp (quat) 取浮点帧, 不再用整数 cur_frame 直接索引
- **11.1.1 expert clip pre/post 时长** 一次性预处理 (forward pre=0.74s post=0.88s, backward pre=0.40s post=0.86s)
- **11.1.2 get_ref_state(cmd, dt)** 完整算法 (浮点 ref_frame_f + lerp/slerp)
- **11.1.3 cmd 重采样时机**: `t_to_hit ≤ -t_post_swing` (post 段也走完), 不再 `t_to_hit ≤ 0`
- **11.1.4 数字举例**: 例 A 长 pre + 短 post (backhand, t_pre=1.5s + t_post=0.3s); 例 B 短 pre + 长 post (forehand, t_pre=0.3s + t_post=1.2s); 例 C timeout 处理. 每例步进表展示 cur_step / t_to_hit / ref_frame_f
- **11.1.5 quat_slerp 实现** (numpy, 处理近端 + 小角度退化为 lerp)
- **3.1 cmd 字段表** 新增 3 个字段:
  - `t_pre_initial` (truncN[0.2,1.5] peak [0.4,0.8] 秒, 进 obs 作 t_to_hit 初值) [v5.4 放宽 peak]
  - `t_post_swing` (独立 truncN[0.2,1.5] peak [0.4,0.8] 秒, **不进 obs**, 仅控 ref 缩放 + 重采边界) [v5.4 放宽 peak]
  - `cur_step` (取代 v5.3 `cur_frame`, 因 ref_frame 变浮点)
- **3.2 cmd 生成代码** 重写: 加 t_pre_initial / t_post_swing 独立采样; cur_step=0 复位; 重采边界 `t_to_hit ≤ -t_post_swing`
- **3.3 cmd 生命周期** 重写: ref state 计算改为 get_ref_state(cmd, dt) 当场算 (不存 ref_frame_f); 重采边界更新
- **11.5 PingpongCommand dataclass** 加 t_pre_initial / t_post_swing, cur_frame → cur_step
- **DIVERGENCE O 重写**: 不再"末尾冻结", 改为"双段时间比例插值缩放" — paper 单 clip 不需此操作, 我们 2-clip + 自由 swing 时长设定的工程必要
- **新增 DIVERGENCE Q**: cmd 字段扩展 (t_pre_initial / t_post_swing / cur_step), paper 仅 1 个 t_strike 字段 (= 我们的 t_to_hit), 我们多出 3 个内部字段是双段缩放的实现需要

#### v5.3 锁定决定 (前一轮, 已并入):
- **#31 cmd 内部全部存 world frame** (球桌 fixed in world):
  - 3.1 #2 字段 `p̂_racket` frame: `base-relative` → **`world`** (`x=0.4` = 桌沿 world x)
  - 3.2 cmd 生成代码全部 world 量命名, 加 "Obs 端坐标转换" 子段 `R_base^T · (cmd.world − p_base_world)`
  - 11.4 expert_offset 注释 "2D world delta" (假设 ref clip yaw ≈ 0)

#### v5.2 锁定决定 (前两轮, 已并入):
- **11.2 [Q2] swing_type 隐含路由**: 不进 Actor / Critic obs (维度不变), 作 ref clip selector 路由 T_B / [q̄, q̇̄] / r_i ref. ⚠️ DIVERGENCE P
- **11.3 [Q3] r^bp p_rel xy 减 z 保留**: 防蹲下作弊
- **11.4 expert 数据来源**: forward_001 + backward_004, BLADE_IDX=24, PELVIS_IDX=0
- **11.5 [Q4] t_to_hit 命名统一**: 删除 t_strike 绝对时刻
- **8.1 σ_g_pos 加 0.03 中段**: 5 阶段 0.10→0.06→0.04→0.03→0.02

#### v5.1 锁定决定 (更早, 已并入):
- **6.4 表桌建模**: RigidBody Cuboid kinematic + 第二个 ContactSensor `filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"]` + 单一 reward `r_table_contact`, weight -1.0
- **8.1 σ_g_pos 终值收紧**: 0.03 → 0.02m
- **8.0 σ 语义说明**: reward kernel σ / 采样标准差 σ_sampling / DR 区间 三类
- **8.5 课程统一表**: 9 项 curriculum 单表

#### v5 累计决定 (更早, 已并入):
- **5 Action scale**: per-joint `0.25 · effort_limit / stiffness` (复用 `UNITREE_G1_23DOF_PADDLE_MIMIC_ACTION_SCALE`)
- **7.3 cmd noise 解读**: gauss σ : clip = 1 : 3 (±3σ 截断)
- **7.2 #4 swing_type**: 独立 uniform({forehand, backhand}) 采样
- **9.3 URDF 路径**: `unitree_ros/robots/g1_description/g1_23dof_rev_1_0_paddle.urdf`
- **世界坐标约定**: 机器人初始 base = world (0, 0, 0.76); cmd 在 world 坐标; Table 近端桌沿 x=0.4

### 记录目标

- 不再分散写到 REWARD_DESIGN.md / OBSERVATION_DESIGN.md / COMMAND_DESIGN.md
- **统一写入 `source/unitree_rl_lab/unitree_rl_lab/tasks/pingpong/final.md`** (新建)
- final.md 是代码编写的唯一参考依据

### 实现 Phase

#### Phase 1: 写 final.md
将 1–11 全部内容整合进 `source/unitree_rl_lab/unitree_rl_lab/tasks/pingpong/final.md` (新建). **11 歧义消解记录 (Q1 v5.4 击球帧对齐 + 双段插值缩放, 含 11.1.1 expert pre/post 时长 + 11.1.2 get_ref_state + 11.1.3 重采时机 + 11.1.4 数字举例 3 个 + 11.1.5 quat_slerp; Q2 / Q3 / Q4 + 11.4 expert_offset world delta 注释) 必须完整保留**, 含每项的实现规范代码. 这是代码编写时消除歧义的唯一参考.

#### Phase 2: 实现 mdp/ 代码
关键文件 (新建):
- `mdp/commands.py` — `PingpongCommand` 类: 3.2 cmd 生成 + 3.3 lifecycle (`t_to_hit ≤ -t_post_swing` 重采样 + swing_type 独立 + cur_step → ref_frame_f 双段插值)
- `mdp/motion_loader.py` — expert_clip dict (forehand / backhand) + `get_ref_state(cmd, dt)` (11.1.2 双段插值 + lerp/slerp)
- `mdp/observations.py` — 2 表的 14 项 (Actor 86 + Critic 213) [v5.5 A2: T_B 排除 paddle blade, 11·7=77], Critic 端用 swing_type dispatch ref state
- `mdp/rewards.py`:
  - r_i: `r^p` / `r^v` / `r^bp` (DM kernel, J 排除 right_wrist_roll_joint, ℬ_pos 排除 right_paddle_blade); ref state 来自 motion_loader.get_ref_state(cmd, dt)
  - r_g: `r_g_pos` / `r_g_vel` / `r_g_ori` / `r_g_base` (sparse abs(t_to_hit)≤0.06s 门控, base 击球后 OFF)
  - r_r: 4.4 IsaacLab 标准 + 6.4.3 单一 `r_table_contact` (sensor #2 `robot_table_contact`)
- `mdp/events.py` — 7.1 startup DR + 7.2 reset (含 RSI 7.4) + 7.3.1 interval (push_robot only) + **cmd noise 走 commands.py resample 一次冻结路径, 不在 interval** [v5.5 A10]
- `mdp/terminations.py` — 6.1 四项 (time_out / bad_orientation / root_height / undesired_contact_terminate)
- `mdp/curriculums.py` — 8.5 表 9 项, 第一轮训练只开 #1 + #2

#### Phase 3: env_cfg
`tasks/pingpong/pingpong_env_cfg.py` (新建): 装配 5 actions + 9 scene (含 Table asset + 双 sensor) + 10 sim + 上面 mdp/

复用 unitree.py 资产: `UNITREE_G1_23DOF_PADDLE_MIMIC_CFG` + `UNITREE_G1_23DOF_PADDLE_MIMIC_ACTION_SCALE`

### 验证 (end-to-end)

1. **Plan-driven sanity**: dry-run env reset, 打印 cmd / obs 维度 (Actor=86, Critic=213) [v5.5 A2], 确认双段 ref_frame_f 插值在 cur_step=sim_pre_steps 时刚好 = impact_frame.
2. **ref state 击球帧对齐 unit test**: 单 env 单 swing 跑完, 验证 t_to_hit=0 那一 step 的 ref_frame_f == impact_frame (浮点等价), ref body pos/quat 与 clip[impact] 完全一致.
3. **Single-env play smoke test**: `--num_envs=1 --headless=False`, 看 reward dashboards, strike window 内 r_g sparse 激活, ref body 跟随 sim cur_step 双段缩放推进 (慢 swing → ref 慢放, 快 swing → ref 加速).
4. **Train**: 4096 envs 训 1k iter, 监控 8.6 第一轮指标 (击球成功率 ↑, survival rate ↑), 确认 σ_g_pos curriculum 推进到 stage 1 (≥30%) 触发.
