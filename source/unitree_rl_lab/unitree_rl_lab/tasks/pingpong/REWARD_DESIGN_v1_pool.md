# Reward Design — G1 23-DoF Paddle WBC

> 仅记录 reward 相关内容: 公式, 激活/不激活条件, 参数, 设计原因.
> Command / Observation / Termination / Event / Curriculum 详细机制见各自独立文档.
> 来源标记: `[paper]` / `[paper-derived]` / `[我提案]` / `[user-decided]`
> Paper: HITTER, arXiv:2508.21043v2.

---

## 0. 总公式 [paper Eq. 7]

$$
\begin{aligned}
r_t  &=  w_i · r_i  +  w_g · r_g  +  w_r · r_r \\
     &=  0.5 · r_i  +  1.0 · r_g  +  0.1 · r_r       [我提案的权重]\\
     r_i &= \text{imitation reward}\\
     r_g &= \text{goal reward}\\
     r_r &= \text{普通机器人运动的reward}
\end{aligned}
$$

权重设计原则: r_g (任务信号) > r_i (引导) > r_r (平滑约束). r_i 不到 r_g 一半, 因为我们要让 task 主导, mimic 只是 warm-up bias.

---

## 1. 激活条件 (Reward Gating)

reward 各 sub-term 的开关由 **3** 个 flag 控制. flag 生成机制属于 event/command 设计, 详见 [COMMAND_DESIGN.md](COMMAND_DESIGN.md) §5; 这里只列 reward 关心的 gate 行为.

| Flag | 含义 | gate 哪些 reward |
|---|---|---|
| `mimic_active` (per env, bool) | 当前 step 是否在 "对照 clip 模仿" 阶段 | 全部 r_i sub-terms; 部分 r_r sub-term 的双段权重 |
| `strike_completed` (per env, bool) | 当前 cmd 对应的击球**事件**是否已发生 (单调切换, 直到 next cmd 才重置) | r_g_pos / r_g_vel / r_g_ori 强制关闭 (即使 strike_window 内) |
| `strike_window` (per env, bool) | `\|t_to_hit\| ≤ 0.1s` (±5 帧 @ 50Hz) [paper Sec V-B2] | r_g_pos / r_g_vel / r_g_ori (3 项 sparse, 仅在 `!strike_completed` 时检查) |

最终 gate: `r_g_sparse_active = strike_window AND (NOT strike_completed)`. 实现上 `t_to_hit` 在击球后冻结为 `-1.0` sentinel, strike_window 自然 = False, 所以两者部分冗余 — 但保留 `strike_completed` 显式 flag 是因为它也喂给 actor/critic obs (见 COMMAND_DESIGN §5.3 占位策略).

### 1.1 阶段 × Reward 激活矩阵

| mimic_active | strike_completed | 阶段 | r_i | r_g sparse | r_g_base | r_r |
|:---:|:---:|---|:---:|:---:|:---:|:---:|
| T | F | mimic pre-swing | ✓ | strike_window 内 ✓ | ✓ | ✓ |
| T | T | mimic follow-through + return-to-ready | ✓ | ✗ | ✓ | ✓ |
| F | T | no-cmd gap (球在空中 + 对手击球) | ✗ | ✗ | ✓ | ✓ |
| F | F | free play pre-strike | ✗ | strike_window 内 ✓ | ✓ | ✓ |
| F | T | free play follow-through | ✗ | ✗ | ✓ | ✓ |

### 1.2 设计原因

- **`mimic_active` hard cutoff**: GVHMR retarget 数据只在 clip 内可信. clip 播放结束后 mimic_active → False, r_i 立刻归零, 由 r_g + r_r 接管. 不做软过渡, 因为 critic obs 端的 mask + valid flag 已能让 value head 区分两阶段.
- **`strike_completed`**: paper 没显式定义但**隐含需要** — paper Sec V-B2 sparse window 只覆盖击球前后, 击完之后到 follow-through 结束这段时间应当不再 active. 用一个独立 flag 比"靠 t_to_hit 自然超出 window"更清晰可靠 (后者依赖 sentinel 的具体数值约定).
- **`strike_window`**: paper Sec V-B2 引文 — *"the tracking rewards for the racket position, velocity, and orientation are only activated during a short window around the hitting time"*. ±5 帧由 50Hz 控制频率 + 业余球员级击球时间窗推断.
- **r_g_base 和 r_r 全程 dense**: 提供 free play 段的引导信号, 避免 sparse-reward 摆烂局部最优 (机器人 mimic 完后冻住不动以最小化 r_r). gap 期 base_target 保持当前击球点站位 (cmd 不变, 见 COMMAND_DESIGN §4.2).

---

## 2. r_i — Imitation Reward (mimic 段 dense)

[paper] 仅说明 ℬ ⊆ upper body 且 r_i 用于 imitation; sub-term 列表 / 公式形式 / σ / weight 全部 [我提案].

### 2.1 公式

```
r_i = mimic_active · ( w_jp·r_jp + w_jv·r_jv + w_bp·r_bp + w_bq·r_bq + w_blv·r_blv + w_bav·r_bav )
```

| sub-term | 含义 | σ | weight |
|---|---|---|---|
| `r_jp`  | 关节位置 (22 dof, J)         | 0.3 rad   | **0.30** |
| `r_jv`  | 关节速度 (22 dof, J)         | 2.0 rad/s | **0.10** |
| `r_bp`  | body 位置 (12 bodies, ℬ_pos) | 0.05 m    | **0.30** |
| `r_bq`  | body 朝向 (11 bodies, ℬ_quat)| 0.2       | **0.20** |
| `r_blv` | body 线速度 (12 bodies)      | 0.5 m/s   | **0.05** |
| `r_bav` | body 角速度 (11 bodies)      | 2.0 rad/s | **0.05** |

权重比例: pos > vel (位置信号比速度更稳); joint 与 body 各占 0.4 (joint-level 准确性 + body-level 全局对齐).

### 2.2 公式细节 (DeepMimic 风格)

```python
# Joint
r_jp = exp(-mean((q[J] - q̂[J])²) / σ_jp²)
r_jv = exp(-mean((q̇[J] - q̇̂[J])²) / σ_jv²)

# Body world pose, 减去 base xy 偏移以对齐多 clip
p_rel[b] = p_world[b] - p_world[pelvis]    # xy 减, z 不减
r_bp  = exp(-mean(‖p_rel[b]  - p̂_rel[b]‖²) / σ_bp²)   for b in ℬ_pos
r_bq  = exp(-mean((1 - |q[b] · q̂[b]|)²)    / σ_bq²)   for b in ℬ_quat   # quat dist
r_blv = exp(-mean(‖v_world[b]- v̂_world[b]‖²)/ σ_blv²) for b in ℬ_pos
r_bav = exp(-mean(‖ω_world[b]- ω̂_world[b]‖²)/ σ_bav²) for b in ℬ_quat
```

### 2.3 参考量来源

`q̂ / q̇̂ / p̂_rel / q̂_b / v̂ / ω̂` 来自 npz 在 `(clip_id, t_offset + step_in_episode)` 帧. RSI 的 `t_offset` 让 mimic 段 ref 与 robot 状态在 reset 时对齐, 之后随 step 推进.

具体加载机制 (motion_loader, 多 clip 池采样, RSI) 见 `EVENT_DESIGN.md` (待写).

### 2.4 跟踪集合 ℬ — 排除规则 + 原因

针对 GVHMR retarget 数据的末端噪声特征, 不全 body / 全 joint mimic, 而是**选择性跟踪**:

**`ℬ_pos`** (位置 + 线速度跟踪 — 12 bodies):
```
[pelvis, torso_link,
 left_shoulder_pitch_link, left_shoulder_roll_link, left_shoulder_yaw_link,
 left_elbow_link, left_wrist_roll_rubber_hand,
 right_shoulder_pitch_link, right_shoulder_roll_link, right_shoulder_yaw_link,
 right_elbow_link, right_wrist_roll_rubber_hand]
```
**排除** `right_paddle_blade`. 原因: 拍面位置噪声大 + 拍面位置应由 r_g (task signal) 主导.

**`ℬ_quat`** (朝向 + 角速度跟踪 — 11 bodies): 同 ℬ_pos 减去 `right_wrist_roll_rubber_hand`.
**排除** `right_paddle_blade` + `right_wrist_roll_rubber_hand`. 原因: 右小臂 quat = 拍面朝向, 与 blade 同样噪声敏感, 留给 r_g_ori.

**Joint J** (22 dof): 全部 23 dof 中**排除** `right_wrist_roll_joint`. 原因: 与 `right_wrist_roll_rubber_hand` 的 quat 不跟踪保持一致 — body-level 不跟踪的部位, joint-level 也不强加约束.

排除策略总原则: GVHMR retarget 数据末端 (右小臂 + 拍) 噪声最大, 不喂给 mimic 反而能让 r_g 主导这部分动作的语义.

---

## 3. r_g — Goal Reward

[paper Sec V-B2] sub-term 命名 + sparse/dense 划分; [我提案] σ + weight + 公式形式.

### 3.1 公式

```
r_g = 1.0 · r_g_pos + 0.5 · r_g_vel + 0.3 · r_g_ori + 0.3 · r_g_base
```

权重比例: 击球点 (1.0) > 速度 (0.5) > 朝向 / 站位 (0.3 / 0.3). 击球点最关键 — 没打到位置就谈不上速度和朝向.

### 3.2 sub-terms

```python
n̂_target = v̂_racket / ‖v̂_racket‖
# [paper Sec IV-C]: "We assume that, at impact, the racket plane is perpendicular
#                    to its velocity vector."
# n̂_target NOT 一个独立 cmd 字段, 必从 v̂_racket 推导.

n_blade = quat_rotate(q_blade, [0, 1, 0])
# blade local +Y 是拍面法向, URDF 中 right_paddle_blade_fixed_joint 的 -45° rpy
# 已经嵌入到 q_blade (body world quat) 里, 这里不要重复加.

# σ_vel 自适应: 慢拍 cmd 容差小, 快拍 cmd 容差按比例放大
# (应对 backward npz v_blade 中位 2 m/s vs forward 4 m/s 的失衡, 见 §3.6)
σ_vel = max(0.3, 0.2 · ‖v̂_racket‖)

r_g_pos  = exp(-‖p_blade - p̂_racket‖² / 0.05²)         · 𝟙[strike_window ∧ ¬strike_completed]
r_g_vel  = exp(-‖v_blade - v̂_racket‖² / σ_vel²)         · 𝟙[strike_window ∧ ¬strike_completed]
r_g_ori  = exp(-(1 - n_blade · n̂_target)² / 0.2²)       · 𝟙[strike_window ∧ ¬strike_completed]
r_g_base = exp(-‖base_xy - base_target_xy‖² / 0.3²)     # 全程 dense, 无 gate
```

`p_blade / v_blade / q_blade`: `right_paddle_blade` body 的 world state.
`base_xy`: pelvis world xy.

### 3.3 σ 选取依据

| sub | σ | 物理意义 |
|---|---|---|
| pos | 0.05 m | 拍面直径 0.15 m 的 1/3 容差; 命中 "拍面有效区" |
| vel | **自适应** `max(0.3, 0.2·‖v̂‖)` | 默认 σ=0.5 (对应 ‖v̂‖=2.5 m/s); forward clip 中位 4 m/s → σ=0.8; backward 中位 2 m/s → σ=0.4. 慢拍场景 σ 不会塌缩到 ε, 快拍场景 σ 按比例放大. 详见 §3.6 |
| ori | 0.2 (cos dist) | 对应 ~11°, 业余球员级精度 |
| base | 0.3 m | 站位精度比拍面宽松, 鼓励 robot 朝目标移动而非追求精确站定 |

### 3.4 r_g_base DENSE 的设计原因

paper Sec V-B 说明 base_pos reward 用于 "before strike". 在我们的两阶段 episode 里, r_g_base 是 phase 2 (`mimic_active=False`) 期间唯一 dense 的 task signal — 没有它, phase 2 reward 几乎全是 r_r 的负惩罚 + 击球瞬间一个 spike, sparse 程度太高, policy 容易陷入 "冻住不动" 局部最优.

### 3.5 关于 n̂_target 的常见误区

不要把 `n̂_target` 当独立 cmd 字段. paper 定的关系是:
- planner 内部先算法向 `u = (v_o − v_i) / ‖v_o − v_i‖` (出球方向 - 来球方向归一化)
- 然后用 `v̂_racket = ((v_o·u + C_r·v_i·u) / (1 + C_r)) · u` 算拍速大小, 整个 v̂ 沿 u 方向
- 所以 `v̂_racket` 已经携带了法向信息, `n̂ = v̂/‖v̂‖` 直接得到, 物理意义上正确

cmd 数据结构里只存 `v̂_racket`, **不要**额外存 `n̂_target` (除非未来要训练侧旋等非垂直击球, 那时 paper 假设不再成立).

### 3.6 σ_vel 自适应的来源 (D7 测量)

D7 在所有 npz 上跑了 `body_lin_vel_w[impact_frame, blade_idx]` 的统计:

| 子集 | n | 中位 ‖v_blade‖@impact | ±5fr 窗最大 | 量级判断 |
|---|---|---|---|---|
| forward (forehand) | 85 | 4.10 m/s | 4.99 m/s | 与 paper 2-6 m/s 对得上 ✓ |
| backward (backhand) | 75 | **1.99 m/s** | 2.72 m/s | 偏低, GVHMR 末端速度可能被滤平 ⚠️ |

**问题**: 若 r_g_vel 用统一 σ=0.5, backward clip 的 ‖v_blade − v̂‖ 天然小 (因为 v̂ 本身就小), r_g_vel 容易高分 → backward 数据被过度奖励, forward / backward 学习信号失衡.

**应对** (D7 用户决定): `σ_vel = max(0.3, 0.2·‖v̂_racket‖)`.
- backward clip ‖v̂‖=2 m/s → σ_vel=0.4 (容差 20%, 实际仍鼓励 robot 把速度做出来)
- forward clip ‖v̂‖=4 m/s → σ_vel=0.8 (容差 20%, 比统一 0.5 更宽松)
- floor `0.3`: 防止 ‖v̂‖→0 时 σ→0 触发数值爆炸 (e.g., gap 期占位 cmd, ‖v_racket‖ 可能很小)

数据集层面的根本修复 (TODO): backward 数据集后续若效果不达预期, 重新采集大幅度反手 mp4 重训.

---

## 4. r_r — Regularization Reward (dense, 全程激活)

[paper 完全没给, 全部 [我提案]]. IsaacLab humanoid locomotion 标准 reg 套餐.

| sub-term | 公式 | weight |
|---|---|---|
| `action_rate_l2`        | ‖a_t − a_{t-1}‖²                | **-0.01** |
| `action_l2`             | ‖a_t‖²                          | **-0.0005** |
| `dof_torques_l2`        | ‖τ‖²                            | **-2e-5** |
| `dof_acc_l2`            | ‖q̈‖²                            | **-2.5e-7** |
| `dof_pos_limits`        | hinge 关节超限惩罚              | **-5.0** |
| `pelvis_orientation_l2` | ‖proj_g_xy‖² (倾倒)             | **-1.0** |
| `pelvis_height_mimic`   | (h - 0.74)² · 𝟙[mimic_active]   | **-10.0** |
| `pelvis_height_free`    | (h - 0.74)² · 𝟙[!mimic_active]  | **-50.0** |
| `feet_slide`            | 接触脚的水平速度                | **-0.05** |
| `feet_air_time`         | 摆动脚悬空时间 (target=0.4s)    | **+0.5** |
| `undesired_contacts`    | 膝/手/胯触地 (per body)         | **-1.0** |

### 4.1 设计要点

- **`pelvis_height` 双段制**: free play 段权重加大 (-50 vs -10), 防止 mimic 完后 robot 慢慢蹲下放弃站立. 因为 mimic 段 r_i 已强约束 base 高度通过 ref tracking; free 段没有 r_i 约束, reg 必须更强.
- **`action_rate` vs `action_l2`**: 前者管平滑, 后者管动作幅度, 两者协同避免 jitter; 单用 action_rate 会让 policy 学到大幅但平滑的动作.
- **`feet_slide` + `feet_air_time` 组合**: 鼓励正常 walking gait. 站着不动 → feet_air_time 给负值; 蹭脚 → feet_slide 惩罚; 跳跃 → 双脚 air_time 同时大, gait 不稳, ang_vel_xy 罚.
- **`undesired_contacts`**: 膝/手/胯触地是摔倒的早期信号, 比 termination 更细粒度.

---

## 5. Reward 相关 Curriculum [我提案]

只列影响 reward 公式的参数调度. 其他 curriculum (cmd 速度范围, phase 2 时长占比等) 见 `CURRICULUM_DESIGN.md` (待写).

| Iter range | r_g_pos σ | 备注 |
|---|---|---|
| 0–8k    | 0.10 (loose)        | warmup, reward landscape 平滑, policy 容易爬出局部最优 |
| 8k–25k  | 0.06 → 0.05 (linear)| 渐进收紧到 default |
| 25k+    | 0.05 (or 0.03)      | optional fine-tune 到 3 cm 精度 |

其他 sub-term 的 σ 不做 curriculum, 全程 default. 原因: r_g_pos 是任务最关键信号, 难度 curriculum 的边际收益最大; 其他项收紧 σ 影响小.

---

## 6. Reward 设计决定记录

| 决定 | 选择 | 原因 / 引用 |
|---|---|---|
| Total weights | 0.5 / 1.0 / 0.1 | 我提案; r_g (任务) > r_i (引导) > r_r (平滑) |
| `r_i` sub-terms 数 | 6 (jp/jv/bp/bq/blv/bav) | DeepMimic 标准分解, 各自独立调 σ |
| `r_g` σ 紧度 | pos=0.05 / vel=自适应 / ori=0.2 / base=0.3 | 拍面 1/3 容差; vel 适应 backward 量级失衡 (D7) |
| σ_vel 自适应公式 | `max(0.3, 0.2·‖v̂‖)` | D7 测量: backward clip ‖v_blade‖ 中位 2 m/s, forward 4 m/s, 统一 σ 失衡 |
| `n̂_target` 推导 | `v̂_racket / ‖v̂_racket‖` | paper Sec IV-C 物理假设, 不存独立字段 |
| Strike gating flags | `mimic_active + strike_completed + strike_window` | strike_completed 是事件 flag (击球后单调 True), 与 mimic_active 正交, 也喂 obs |
| Strike window | ±5 帧 (±0.1s @ 50Hz) | paper Sec V-B2 "short window" |
| ℬ 跟踪排除 | blade / wrist_roll quat / wrist_roll_joint | GVHMR 数据末端噪声, 让 r_g 主导 |
| `pelvis_height` 双段制 | mimic -10, free -50 | 防 free play 段 robot 蹲下 |
| Mimic cutoff | 硬切 (1 step 内 w_i 归零) | reward gate 简洁; critic 端用 mask + flag 区分 |
| `r_i` ref state 帧 | `clip[t_offset + step]` | 复用 RSI offset, mimic 段始终对齐 clip 进度 |
| `r_r` 套餐 | IsaacLab humanoid locomotion 标准 | paper 无, 复用 unitree_rl_lab 已 validate 的 reg 配置 |

---

## 7. Paper 引用索引

| 内容 | paper 章节 |
|---|---|
| Total reward 公式 (Eq. 7) | Sec V-B |
| ℬ ⊆ upper body | Sec V-B2 |
| Strike window 描述 | Sec V-B2 |
| `n̂_target ≡ v̂_racket / ‖v̂_racket‖` | Sec IV-C |
| Episode = 10s | Sec V-B1 |
| 50Hz 控制 + joint pos action | Sec V |

paper 未给 (全部 [我提案]):
- 各 sub-term 具体公式 / σ / weight 数值
- r_r 任何细节
- ℬ 的具体 body 列表
- Curriculum 具体 schedule
