# Reward Design v2 — G1 23-DoF Paddle WBC (Paper-Aligned 2-Clip)

> **v2 改动**: paper-aligned 2-clip setup (forward_001 + backward_004 expert clips). 删除 strike flag 状态机, 用 `abs(t_to_hit) <= 0.1` 直接 gate sparse reward.
> 旧版 (clip pool + strike flag 状态机) 见 [REWARD_DESIGN_v1_pool.md](REWARD_DESIGN_v1_pool.md).
>
> Doc 范围: reward 公式, 激活条件, σ / weight, 设计原因.
> Command / Observation / Termination / Event / Curriculum 详细机制见各自独立文档.
> 来源标记: `[paper]` / `[paper-derived]` / `[我提案]` / `[user-decided]` / **⚠️ DIVERGENCE**
> Paper: HITTER, arXiv:2508.21043v2.

---

## §0. v2 关键变更点 (相对 v1)

| v1 | v2 (paper-aligned 2-clip) | 原因 |
|---|---|---|
| Sparse gate: `strike_window AND ¬strike_completed` (双 flag) | **直接 `abs(t_to_hit) <= 0.1`** | 删除 flag 状态机, t_to_hit 自然信号 (cmd 设计 v2 决定) |
| `r_alive` 在 v1 §11.1 cross-doc TODO | **正式整合到 r_r** | alive_reward = +0.1 per step |
| ℬ 排除 paper-derived (含 blade + wrist_roll) | **保留**, 但 ⚠️ DIVERGENCE 标注 (paper 只说 ⊆ upper body) | expert 数据噪声小, 仍排除 blade (r_g 主导) |
| σ_vel 自适应 `max(0.3, 0.2·‖v̂‖)` | **保留, 但 TODO 第一轮训练后重测** | 应对 backward_004 v=1.99m/s, paper 未指定 σ |
| `[我提案]` 的 weight + σ | **全部重新审视, 标注 ⚠️ DIVERGENCE 来源** | paper 没给具体数值, 需要明确这是我们的工程选择 |

---

## §1. 总公式 [paper Eq. 7]

$$
r_t  =  w_i · r_i  +  w_g · r_g  +  w_r · r_r
$$

**Weights** [我提案, paper 未给]:

| 权重 | 数值 | 设计原则 |
|---|---|---|
| `w_i` | 0.5 | 引导项, 不主导 |
| `w_g` | 1.0 | 任务项 (击球), 主导 |
| `w_r` | 0.1 | 平滑约束, 防干扰主任务 |

设计原则: `w_g (任务) > w_i (引导) > w_r (平滑)`. r_i 不到 r_g 一半, 让 task 主导, mimic 只是 warm-up bias.

⚠️ **DIVERGENCE A — weights 数值**: paper 公式 Eq. 7 形式正确, 但 `w_i / w_g / w_r` 具体数值未给. 我们 0.5/1.0/0.1 是工程选择, 训练后可调.

---

## §2. 激活条件 (Reward Gating)

### 2.1 Gate 信号 (v2 简化)

| 信号 | 含义 | gate 哪些 reward |
|---|---|---|
| `mimic_active` (per env, bool) — **episode-internal state, NOT cmd 字段** | 当前 step 是否在 "mimic clip 跟踪" 阶段 | 全部 r_i sub-terms; 部分 r_r sub-term 的双段权重 |
| `abs(cmd.t_to_hit) <= 0.1` (per env, bool) — **从 cmd.t_to_hit 直接推, 无 flag** | 击球时间窗口 (±5 帧 @ 50Hz, 11 帧总宽) | r_g_pos / r_g_vel / r_g_ori (3 项 sparse) |

⚠️ v1 的 `strike_window_reward_passed` 和 `hit_actually_landed` flag **已删除**, 详见 [COMMAND_DESIGN.md](COMMAND_DESIGN.md) §0.

### 2.2 阶段 × Reward 激活矩阵

| mimic_active | abs(t_to_hit)<=0.1 | 阶段 | r_i | r_g sparse | r_g_base | r_r |
|:---:|:---:|---|:---:|:---:|:---:|:---:|
| T | F | mimic pre-swing 前段 / follow-through | ✓ | ✗ | ✓ | ✓ |
| T | T | mimic 击球瞬间 (cur_frame ∈ [impact-5, impact+5]) | ✓ | ✓ | ✓ | ✓ |
| F | F | gap / free pre-strike 前段 / follow-through | ✗ | ✗ | ✓ | ✓ |
| F | T | free 击球瞬间 | ✗ | ✓ | ✓ | ✓ |

### 2.3 设计原因

- **`mimic_active` hard cutoff**: clip 播完那一刻 mimic_active → False, r_i 立刻归零. 不做软过渡, 因为 critic obs 端 (待 OBSERVATION_DESIGN.md 定) 可以用 mask 让 value head 区分两阶段.
- **直接用 `abs(t_to_hit) <= 0.1`**: paper Sec V-B2 — *"the tracking rewards for the racket position, velocity, and orientation are only activated during a short window around the hitting time"*. v1 用 flag 状态机实现, v2 发现 t_to_hit 单调减 + 不冻结后, abs() 直接表达 ±5 帧窗口, 等价但实现简单.
- **r_g_base 和 r_r 全程 dense**: 提供 free play 段的引导信号, 避免 sparse-reward 摆烂局部最优 (机器人 mimic 完后冻住不动以最小化 r_r). gap 期 base_target 保持击球完瞬间 base_xy frozen ([COMMAND_DESIGN.md](COMMAND_DESIGN.md) §4.1).

⚠️ **DIVERGENCE B — r_g_base dense 全程**: paper Sec V-B 描述 base_pos reward 为 "before strike". 我们 dense 全程 (含 follow-through + gap). 理由: 防止 free play 段反弹到局部最优. 留 ablation TODO.

---

## §3. r_i — Imitation Reward (mimic 段 dense)

### 3.1 公式

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

⚠️ **DIVERGENCE C — r_i sub-term 分解**: paper Sec V-B2 仅说 *"ℬ ⊆ upper body 且 r_i 用于 imitation"*, **sub-term 列表 / 公式形式 / σ / weight 全部 [我提案]**. 我们沿用 DeepMimic 标准 6 项分解 (jp/jv/bp/bq/blv/bav).

### 3.2 公式细节 (DeepMimic 风格)

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

### 3.3 参考量来源

`q̂ / q̇̂ / p̂_rel / q̂_b / v̂ / ω̂` 来自 npz 在 `(clip_id, t_offset + step_in_episode)` 帧.

**v2 简化 (RSI 解耦)**: `t_offset = 0` 始终. mimic 段 ref 从 clip 第 0 帧开始, 跟着 step 推进, 直到 clip 末则 mimic_active = False.

`clip_id ∈ {0, 1}` 由 episode reset 时随机采 (50/50 forward / backward, 见 [COMMAND_DESIGN.md](COMMAND_DESIGN.md) §2.3).

具体加载机制 (motion_loader, 2-clip pool, RSI) 见 EVENT_DESIGN.md (待写).

### 3.4 跟踪集合 ℬ — 排除规则 + 原因

针对 GVHMR retarget 数据的末端噪声特征 (即使 expert clip 也仍有), 不全 body / 全 joint mimic, 而是**选择性跟踪**:

**`ℬ_pos`** (位置 + 线速度跟踪 — 12 bodies):
```
[pelvis, torso_link,
 left_shoulder_pitch_link, left_shoulder_roll_link, left_shoulder_yaw_link,
 left_elbow_link, left_wrist_roll_rubber_hand,
 right_shoulder_pitch_link, right_shoulder_roll_link, right_shoulder_yaw_link,
 right_elbow_link, right_wrist_roll_rubber_hand]
```
**排除** `right_paddle_blade`. 原因: 拍面位置应由 r_g (task signal) 主导, 不与 r_i 重复约束.

**`ℬ_quat`** (朝向 + 角速度跟踪 — 11 bodies): 同 ℬ_pos 减去 `right_wrist_roll_rubber_hand`.
**排除** `right_paddle_blade` + `right_wrist_roll_rubber_hand`. 原因: 右小臂 quat ≈ 拍面朝向 (rigid 连接), 留给 r_g_ori 主导.

**Joint J** (22 dof): 全部 23 dof 中**排除** `right_wrist_roll_joint`. 原因: 与 `right_wrist_roll_rubber_hand` quat 不跟踪保持一致 — body-level 不跟踪的部位, joint-level 也不强加约束.

⚠️ **DIVERGENCE D — ℬ 排除策略**: paper Sec V-B2 仅说 `ℬ ⊆ upper body`, **未列具体 body**. 我们的排除 (blade / wrist_roll quat / wrist_roll_joint) 是 paper-derived 推断:
- "blade 应由 r_g 主导" 是合理工程判断 (避免 r_i / r_g 重复约束)
- "wrist_roll quat = blade quat" 是几何事实 (rigid)

**v2 是否回退**: 因为 expert clip 噪声比 v1 pool 小, 理论上可以放宽排除. 但**保守选择 v1 排除**, 训练后看 r_i 各项 mean 值再决定. 第一轮训练 monitor:
- 若 r_bp 持续低, 可能是 ℬ_pos 排除太严; 但 expert clip 应该不会
- 若 r_bq 在 wrist_roll 重新加入后仍 > 0.7, 可考虑放宽

### 3.5 v2 数据特性 — 2 expert clips

| clip | swing | frames | impact | ratio | v_blade@imp | 关键问题 |
|---|---|---|---|---|---|---|
| forward_001 | 0 | 82 | 37 | **0.451** | 4.42 m/s | impact ratio paper-aligned ✓; dy=0.21 偏侧 |
| backward_004 | 1 | 64 | 20 | **0.313** ⚠ | 1.99 m/s ⚠ | impact ratio 偏前 (paper 0.46); v 偏低 |

⚠️ **DIVERGENCE E — clip 长度 + ratio**: paper 单 clip 94 帧 impact 43 ratio 0.46. 我们 forward 接近 paper, backward 偏离. mimic 段时长跟 clip 走 (forward 1.64s, backward 1.28s vs paper 1.88s).

⚠️ **DIVERGENCE F — backward v_blade 量级**: paper 没明确给反手击球速度数值. backward_004 = 1.99m/s, 是 GVHMR 末端被滤平结果. 详见 §4.6 σ_vel 自适应.

---

## §4. r_g — Goal Reward

### 4.1 公式

```
r_g = 1.0 · r_g_pos + 0.5 · r_g_vel + 0.3 · r_g_ori + 0.3 · r_g_base
```

权重比例: 击球点 (1.0) > 速度 (0.5) > 朝向 / 站位 (0.3 / 0.3). 击球点最关键 — 没打到位置就谈不上速度和朝向.

⚠️ **DIVERGENCE G — r_g sub-term weights**: paper Sec V-B2 列 sub-term 命名 + sparse/dense 划分, **σ + weight + 公式形式 [我提案]**.

### 4.2 sub-terms

```python
# strike window gate (v2 直接用 t_to_hit, 无 flag):
strike_window_active = (abs(cmd.t_to_hit) <= 0.1)

# 拍面法向推导 [paper Sec IV-C]:
# "We assume that, at impact, the racket plane is perpendicular to its velocity vector."
n̂_target = v̂_racket / ‖v̂_racket‖     # NOT 独立 cmd 字段, 必从 v̂_racket 推导
n_blade  = quat_rotate(q_blade, [0, 1, 0])
# blade local +Y 是拍面法向. URDF 中 right_paddle_blade_fixed_joint 的 -45° rpy
# 已经嵌入到 q_blade (body world quat) 里, 这里不要重复加.

# σ_vel 自适应 [DIVERGENCE H]:
σ_vel = max(0.3, 0.2 · ‖v̂_racket‖)

r_g_pos  = exp(-‖p_blade - p̂_racket‖² / 0.05²)         · 𝟙[strike_window_active]
r_g_vel  = exp(-‖v_blade - v̂_racket‖² / σ_vel²)         · 𝟙[strike_window_active]
r_g_ori  = exp(-(1 - n_blade · n̂_target)² / 0.2²)       · 𝟙[strike_window_active]
r_g_base = exp(-‖base_xy - base_target_xy‖² / 0.3²)     # 全程 dense, 无 gate
```

`p_blade / v_blade / q_blade`: `right_paddle_blade` body 的 world state.
`base_xy`: pelvis world xy.

### 4.3 σ 选取依据

| sub | σ | 物理意义 |
|---|---|---|
| pos | 0.05 m | 拍面直径 0.15 m 的 1/3 容差; 命中 "拍面有效区" |
| vel | **自适应** `max(0.3, 0.2·‖v̂‖)` ⚠️ DIVERGENCE H | forward (‖v̂‖=4.42): σ=0.884; backward (‖v̂‖=1.99): σ=0.398 |
| ori | 0.2 (cos dist) | 对应 ~11°, 业余球员级精度 |
| base | 0.3 m | 站位精度比拍面宽松, 鼓励 robot 朝目标移动而非追求精确站定 |

⚠️ **DIVERGENCE H — σ_vel 自适应**: paper 公式 `exp(-‖·‖²/σ²)` 但 σ 数值未指定. 我们用自适应是为了应对 backward_004 v=1.99m/s vs paper 隐含的 ~3-5 m/s 不平衡.
- 若 σ_vel 用统一 σ=0.5, backward clip ‖v_blade − v̂‖ 天然小 → r_g_vel 容易高分 → backward 数据被过度奖励
- 自适应 floor 0.3 防止 ‖v̂‖→0 时 σ→0 触发数值爆炸

**v2 决策**: 保留自适应, 训练第一轮后 monitor:
- r_g_vel 在 forward / backward 上的 mean / variance 是否平衡 (差距 < 30% 算可接受)
- 若失衡 → 选项: (a) 切回 paper 风格固定 σ=0.5, 接受 forward / backward 不对称; (b) 替换 backward 数据 (重新拍 mp4)

### 4.4 r_g_base DENSE 的设计原因 (DIVERGENCE B)

paper Sec V-B 说明 base_pos reward 用于 "before strike". 我们的两阶段 episode 里, r_g_base 是 phase 2 (`mimic_active=False`) 期间唯一 dense 的 task signal — 没有它, phase 2 reward 几乎全是 r_r 的负惩罚 + 击球瞬间一个 spike, sparse 程度太高, policy 容易陷入 "冻住不动" 局部最优.

**Ablation TODO**: 跑两个 baseline:
- A: r_g_base dense 全程 (本设计)
- B: r_g_base 仅 strike_window 内 active (paper-aligned)

对比 free play 段的"行动量" + 击球率, 决定保留哪个.

### 4.5 关于 n̂_target 的常见误区

不要把 `n̂_target` 当独立 cmd 字段. paper Sec IV-C 关系:
- planner 内部先算法向 `u = (v_o − v_i) / ‖v_o − v_i‖` (出球方向 - 来球方向归一化)
- 然后用 `v̂_racket = ((v_o·u + C_r·v_i·u) / (1 + C_r)) · u` 算拍速大小, 整个 v̂ 沿 u 方向
- 所以 `v̂_racket` 已经携带了法向信息, `n̂ = v̂/‖v̂‖` 直接得到, 物理意义上正确

cmd 数据结构里只存 `v̂_racket`, **不要**额外存 `n̂_target` (除非未来要训练侧旋等非垂直击球, 那时 paper 假设不再成立).

### 4.6 σ_vel 自适应的来源 (D7 测量, v2 expert clip 复测)

v1 D7 在 pool (85+75 clips) 上 measure:
| 子集 | n | 中位 ‖v_blade‖@impact | ±5fr 窗最大 |
|---|---|---|---|
| forward | 85 | 4.10 m/s | 4.99 m/s |
| backward | 75 | 1.99 m/s | 2.72 m/s |

v2 expert clips (单 clip):
| clip | ‖v_blade‖@impact | ‖v_blade‖ peak (±5fr 窗) |
|---|---|---|
| forward_001 | **4.42 m/s** | 5.80 m/s (peak_off=-2 帧) |
| backward_004 | **1.99 m/s** | 2.72 m/s |

forward/backward 不对称依然存在. 自适应公式 `σ = max(0.3, 0.2·‖v̂‖)` 给 σ_fwd=0.884, σ_bwd=0.398, 两者 r_g_vel 期望 reward 在 0.3-0.5 区间均衡.

---

## §5. r_r — Regularization Reward (dense, 全程激活)

### 5.1 公式

| sub-term | 公式 | weight |
|---|---|---|
| `alive_reward`          | +1 per step (dense)             | **+0.1** |
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

⚠️ **DIVERGENCE I — r_r 整体**: paper Sec V-B 完全没给 r_r 细节, **全部 [我提案]**. 沿用 IsaacLab humanoid locomotion 标准 reg 套餐.

### 5.2 设计要点

- **`alive_reward = +0.1`** [v1 cross-doc → v2 整合]: 防止 policy 学"早结束 episode"作 reward 最大化策略 (因为 r_r 全负). 信号 dense, IsaacLab humanoid locomotion 标准做法.
- **`pelvis_height` 双段制**: free play 段权重加大 (-50 vs -10), 防止 mimic 完后 robot 慢慢蹲下放弃站立. 因为 mimic 段 r_i 已强约束 base 高度通过 ref tracking; free 段没有 r_i 约束, reg 必须更强.
- **`action_rate` vs `action_l2`**: 前者管平滑, 后者管动作幅度, 两者协同避免 jitter; 单用 action_rate 会让 policy 学到大幅但平滑的动作.
- **`feet_slide` + `feet_air_time` 组合**: 鼓励正常 walking gait. 站着不动 → feet_air_time 给负值; 蹭脚 → feet_slide 惩罚; 跳跃 → 双脚 air_time 同时大, 步态不稳, ang_vel_xy 罚.
- **`undesired_contacts`**: 膝/手/胯触地是摔倒早期信号, 比 termination 更细粒度.

### 5.3 r_r 数值原则

`w_r = 0.1` 总权重压制 r_r 影响 (相对 r_i 0.5 + r_g 1.0 总 1.5). r_r 绝对值通过单项 weight 已经压到很小:
- `action_l2 -5e-4` × ‖a‖²~10 = -5e-3 量级
- `dof_torques_l2 -2e-5` × ‖τ‖²~1000 = -2e-2 量级
- `pelvis_height_free -50` × (Δh)²~0.04² = -8e-2 量级 (h=0.7m 时)

总 r_r 单 step 量级 -0.1 ~ -0.3, ×w_r=0.1 后 -0.01 ~ -0.03. 远小于 r_g spike (0.5-1.5).

---

## §6. Reward 相关 Curriculum [我提案]

只列影响 reward 公式的参数调度. 其他 curriculum (cmd 速度范围, mimic_start_prob 等) 见 [COMMAND_DESIGN.md](COMMAND_DESIGN.md) §4.4 / `CURRICULUM_DESIGN.md` (待写).

| Iter range | r_g_pos σ | 备注 |
|---|---|---|
| 0–8k    | 0.10 (loose)        | warmup, reward landscape 平滑, policy 容易爬出局部最优 |
| 8k–25k  | 0.06 → 0.05 (linear)| 渐进收紧到 default |
| 25k+    | 0.05 (or 0.03)      | optional fine-tune 到 3 cm 精度 |

其他 sub-term 的 σ 不做 curriculum, 全程 default. 原因: r_g_pos 是任务最关键信号, 难度 curriculum 的边际收益最大; 其他项收紧 σ 影响小.

⚠️ **DIVERGENCE J — σ curriculum**: paper 没说 σ curriculum, 我提案. 训练第一轮可不开启, 验证默认 σ 是否够稳.

---

## §7. ⚠️ Paper Divergence 索引 (v2 集中审视)

| # | 项 | paper | 我们 | 风险 / TODO |
|---|---|---|---|---|
| A | r_i / r_g / r_r weights | 公式给, 数值未给 | 0.5 / 1.0 / 0.1 | 训练后可调 |
| B | r_g_base 全程 dense | "before strike" | dense 全程 | 留 ablation TODO §4.4 |
| C | r_i sub-term 分解 | 公式形式只说 ⊆ upper body | DeepMimic 6 项 (jp/jv/bp/bq/blv/bav) | 标准做法, 风险低 |
| D | ℬ 排除 (blade / wrist_roll / wrist_roll_joint) | paper 未列具体 body | 排除 3 个末端 | 训练第一轮 monitor r_i 各项 mean |
| E | clip 长度 / ratio | 94 帧 imp=43 ratio=0.46 | fwd 82/37/0.45 ✓; bwd 64/20/0.31 ⚠ | backward ratio 偏离, mimic 段时长跟 clip 走 |
| F | backward v_blade | 数值未给 | 1.99 m/s (vs forward 4.42) | GVHMR 滤平, 通过 σ_vel 自适应应对 |
| G | r_g sub-term weights | 数值未给 | pos 1.0 / vel 0.5 / ori 0.3 / base 0.3 | 训练后可调 |
| H | σ_vel 自适应 | 数值未给 | `max(0.3, 0.2·‖v̂‖)` | 训练第一轮后重测决定保留 |
| I | r_r 整体 | 完全未给 | IsaacLab 标准套餐 + alive_reward | 标准做法, 风险低 |
| J | σ_pos curriculum | 未提 | 0.10 → 0.05 | 第一轮可不开启 |

⚠️ **训练第一轮 monitor checklist**:
- [ ] DIVERGENCE D: r_i 各 sub-term mean 值 (期望: r_jp / r_bp ≈ 0.7-0.9 表示跟踪好; 若持续 < 0.5 表示 ℬ 排除可能太严)
- [ ] DIVERGENCE E + F: r_i / r_g_vel 在 forward / backward 两 clip 上的均衡度
- [ ] DIVERGENCE H: r_g_vel 在两 clip 上的 mean / variance — 决定是否切回固定 σ
- [ ] DIVERGENCE B: free play 段 r_g_base dense vs sparse-only ablation (留作训练后实验)

---

## §8. Reward 设计决定记录 (v2 lock)

| 决定 | 选择 | 来源 / 引用 |
|---|---|---|
| Total weights | 0.5 / 1.0 / 0.1 | [我提案 ⚠️ DIVERGENCE A]; r_g (任务) > r_i (引导) > r_r (平滑) |
| `r_i` sub-terms 数 | 6 (jp/jv/bp/bq/blv/bav) | [我提案 ⚠️ DIVERGENCE C]; DeepMimic 标准分解 |
| `r_g` σ | pos=0.05 / vel=自适应 / ori=0.2 / base=0.3 | [我提案 ⚠️ DIVERGENCE G+H]; pos 拍面 1/3 容差 |
| σ_vel 自适应公式 | `max(0.3, 0.2·‖v̂‖)` | [我提案 ⚠️ DIVERGENCE H]; 应对 backward 慢拍量级失衡 |
| `n̂_target` 推导 | `v̂_racket / ‖v̂_racket‖` | paper Sec IV-C 物理假设 |
| Strike sparse gate | **`abs(t_to_hit) <= 0.1`** (无 flag 状态机) | [v2 user-decided + paper Sec V-B2 "short window"] |
| Strike window 宽度 | ±0.1s = ±5 帧 @ 50Hz | paper Sec V-B2 |
| ℬ 跟踪排除 | blade / wrist_roll quat / wrist_roll_joint | [paper-derived ⚠️ DIVERGENCE D]; r_g 主导末端 |
| `pelvis_height` 双段制 | mimic -10, free -50 | [我提案]; 防 free play 蹲下 |
| `alive_reward` | +0.1 per step | [v2 user-decided ⚠️ DIVERGENCE I]; 防早结束 episode |
| Mimic cutoff | 硬切 (1 step 内 w_i 归零) | reward gate 简洁 |
| `r_i` ref state 帧 | `clip[0 + step]` (RSI t_offset = 0) | [v2 user-decided] |
| `r_r` 套餐 | IsaacLab humanoid locomotion 标准 | [我提案 ⚠️ DIVERGENCE I]; paper 无, 复用 unitree_rl_lab 已 validate 的 reg 配置 |
| r_g_base dense | 全程 (含 follow-through + gap) | [我提案 ⚠️ DIVERGENCE B]; 防 free play sparse 摆烂 |

---

## §9. Paper 引用索引

| 内容 | paper 章节 |
|---|---|
| Total reward 公式 (Eq. 7) | Sec V-B |
| ℬ ⊆ upper body | Sec V-B2 |
| Strike window 描述 ("short window") | Sec V-B2 |
| `n̂_target ≡ v̂_racket / ‖v̂_racket‖` | Sec IV-C |
| Episode = 10s | Sec V-B1 |
| 50Hz 控制 + joint pos action | Sec V |

paper 未给 (全部 [我提案], 详见 §7 DIVERGENCE 索引):
- 各 sub-term 具体公式 / σ / weight 数值
- r_r 任何细节
- ℬ 的具体 body 列表
- σ curriculum
- r_g_base "before strike only" 的具体实现 (我们 dense 全程)

---

## §10. v2 设计完整性 review checkpoint

读完 9 节, 实现前需要确认:
- [x] 删除 v1 strike_window_reward_passed / hit_actually_landed 引用
- [x] sparse gate 直接用 `abs(t_to_hit) <= 0.1` (无 flag)
- [x] alive_reward 整合到 r_r (从 v1 cross-doc TODO 升级)
- [x] DIVERGENCE 索引集中 §7 — 训练第一轮 monitor 列表
- [ ] **σ_vel 自适应保留 vs 回 paper 风格固定 σ** 待第一轮训练后决定
- [ ] **r_g_base dense vs sparse-only** 留作 ablation
- [ ] **ℬ 排除是否放宽** (expert 数据噪声小) 第一轮 monitor r_i 各项 mean
- [ ] **σ_pos curriculum 0.10 → 0.05** 第一轮可不开启验证默认是否够稳
