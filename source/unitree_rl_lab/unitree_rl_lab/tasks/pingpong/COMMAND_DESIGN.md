# Command Design v2 — G1 23-DoF Paddle WBC (Paper-Aligned 2-Clip)

> **v2 改动**: 切换到 paper-aligned 2-clip setup (1 forehand + 1 backhand expert).
> 旧版 (clip pool 采样 + multi-clip chaining + 双 strike flag) 见 [COMMAND_DESIGN_v1_pool.md](COMMAND_DESIGN_v1_pool.md).
>
> Doc 范围: cmd 数据结构, 各阶段 cmd 来源, 时序逻辑, 与 planner 对齐, 噪声 curriculum.
> Reward / Observation / Event / Termination 详细机制见各自独立文档.
> 来源标记: `[paper]` / `[paper-derived]` / `[我提案]` / `[user-decided]` / **⚠️ DIVERGENCE**
> Paper: HITTER, arXiv:2508.21043v2.

---

## §0. v2 关键变更点 (相对 v1)

| v1 (clip pool) | v2 (paper-aligned 2-clip) | 原因 |
|---|---|---|
| 85 forward + 75 backward npz pool, 随机采样 | **forward_001 + backward_004** 各 1 个 expert | paper 只用 2 个 reference clip, 我们数据质量参差不齐, 收敛风险高 |
| Multi-clip chaining (CHAIN_PROB curriculum) | **不做串接** | paper 没有, 2-clip 数据量太小串不起来 |
| `strike_window_reward_passed` flag (单调切换) | **删除**, 直接用 `abs(t_to_hit) <= 0.1` gate | flag 状态机冗余, t_to_hit 自然信号 |
| `hit_actually_landed` flag (几何检测) | **删除** | paper 没有, 部署侧 perception 不可信, 用处低 |
| `is_mimic_phase` 在 cmd struct 里 | **移到 episode-internal state**, 不进 cmd | paper actor obs 不含 mimic flag, 我们也不该让 actor 看到 (训练/部署不一致) |
| Free 段 x 自由采样 [-0.30, +0.30] | **truncated Gaussian** mean=0.4, std=0.08, clip [0.25, 0.6] | x=0.4 是 paper 标准, 我们让 deploy 时 perception 给非 0.4 也能泛化 |
| Cmd 噪声**一次性** (cmd 生成时加) | **每 step 重新加** (持续性) | 模拟 vision/perception 每帧抖动 |

⚠️ **DIVERGENCE 索引** — paper 的偏离点集中在 §9, 实现时回看核对.

---

## §1. Cmd 数据结构

```python
@dataclass
class HitCommand:
    # === paper Table I 4 大核心字段 (world frame) ===
    p_racket:       torch.Tensor  # (E, 3)   期望击球点 world pos
    v_racket:       torch.Tensor  # (E, 3)   期望拍速 world (大小+方向)
                                  # n̂_target = v_racket / ‖v_racket‖, NOT 独立字段
                                  #   [paper Sec IV-C: "racket plane perpendicular to its velocity"]
    t_to_hit:       torch.Tensor  # (E,)     距离击球还有多少秒 (相对时间)
                                  #   击球后继续递减到负值, 不冻结 [user-decided]
                                  #   ⚠️ paper Table I 是 t_strike (单调减), 我们语义一致 (t_to_hit 即 t_strike - t_now)
    base_target_xy: torch.Tensor  # (E, 2)   期望脚下站位 world xy

    # === 内部状态 (sampler / cmd manager 用, 不全部进 actor obs) ===
    swing_type:     torch.Tensor  # (E,)     int8. 0=forehand, 1=backhand
                                  #   仅用于 base_target_xy 几何 (forehand 偏 +y) + 选 mimic clip
                                  #   ⚠️ 不进 actor obs (paper Table I 没有, 隐式由 p_racket 几何决定)
                                  #   ⚠️ 是否进 critic obs 待定 (取决于 critic obs 设计)
    clip_id:        torch.Tensor  # (E,)     int8. 0=forward_001, 1=backward_004 (mimic 段才有意义)
                                  #   ⚠️ 编码方式 / 是否进 obs 留待 critic obs 设计后定
```

### 1.1 已删除字段 (相对 v1)

| 字段 | 删除原因 |
|---|---|
| `is_mimic_phase` | 移到 episode-internal `mimic_active` state, 不暴露给 actor (deploy=False, 训练含 True 会让 actor 学到训练捷径). r_i gate 直接读 episode state. |
| `strike_window_reward_passed` | 删除. r_g sparse gate 直接用 `abs(t_to_hit) <= 0.1`. |
| `hit_actually_landed` | 删除. paper 没有, 部署 perception 测不准, 训练侧也只是 diagnostic. |

### 1.2 字段单位与 planner 输出对齐

```python
# planner.update() 返回 19 字段, cmd 只挑 5 个 (free 段使用):
cmd.p_racket         <-  planner.hit_pos
cmd.v_racket         <-  planner.v_paddle
cmd.t_to_hit         <-  planner.t_to_hit
cmd.base_target_xy   <-  planner.base_target_xy
cmd.swing_type       <-  planner.swing_type    # 0=forehand, 1=backhand

# planner 不输出的字段:
cmd.clip_id          <-  swing_type 决定 (0→0=forward_001; 1→1=backward_004)
```

planner 输出但 cmd 不存的 12 字段 (planner 内部诊断): `base_pos, base_quat, paddle_normal, v_ball_in, v_ball_out, target_land, n_buf, plan_mode, x_hit_used, stale_age_s, hold_reason, bounced, bounces, traj_p`.

---

## §2. Mimic 段 cmd 来源 (2 expert clips)

mimic 段 cmd **全部从 npz 合成**, 与 r_i 的参考动作严格对齐.

### 2.1 Reference clips (paper-aligned)

```
expert/forward/forward_001.npz    swing_type=0  frames=82  impact=37  ratio=0.451  v_blade=4.42 m/s
expert/backward/backward_004.npz  swing_type=1  frames=64  impact=20  ratio=0.313  v_blade=1.99 m/s
```

⚠️ **DIVERGENCE 1 — clip 长度**: paper Sec V 说 "94 frames @ 50Hz, impact at frame 43" (1.88s, ratio=0.46).
- forward_001 ratio=0.451 ✓ 接近
- backward_004 ratio=0.313 ⚠️ 偏离 (impact 在 clip 31% 处, follow-through 比 paper 长)
- frames 都比 paper 短 (82/64 vs 94)

**应对**: mimic 段时长跟 clip 帧数走, 不强对齐 paper 94 帧. clip 末了就 mimic_active = False, 不重复播放.

⚠️ **DIVERGENCE 2 — backward v_blade**: paper 没明确给反手击球速度. backward_004 v_blade=1.99 m/s 比 forward 4.42 m/s 慢一倍 (GVHMR 末端被滤平). 应对见 §2.4.

### 2.2 字段合成

```python
# Episode reset 时 sample 一个 clip:
clip_id = sample({0: 0.5, 1: 0.5})           # 50/50 forward / backward [user-decided]
clip = REF_CLIPS[clip_id]                     # forward_001 or backward_004
impact = clip.impact_frame                    # 37 or 20

# Static cmd 字段 (整个 mimic 段不变):
p_racket   = clip.body_pos_w[impact, BLADE_IDX]      # blade body world pos
v_racket   = clip.body_lin_vel_w[impact, BLADE_IDX]  # blade body world lin vel
swing_type = clip.swing_type                         # 0 or 1

# Dynamic cmd 字段 (随 cur_frame 推进):
cur_frame  = step_in_episode                         # 0, 1, 2, ...
t_to_hit   = (impact - cur_frame) / fps              # 单调下降, 击球后继续负
                                                     # ⚠️ 击球后不冻结, 一直减下去 [user-decided]

# base_target_xy: 分段 [user-decided]
if cur_frame <= impact + 5:                          # pre-swing + strike_window
    base_target_xy = clip.body_pos_w[impact, PELVIS_IDX, :2]
else:                                                # follow-through + return-to-ready
    base_target_xy = clip.body_pos_w[-1, PELVIS_IDX, :2]
```

`BLADE_IDX = 24`, `PELVIS_IDX = 0` (npz body_pos_w shape = (T, 25, 3)). 通过 `body_names` 字段按 name 解析, 避免硬编码漂移.

### 2.3 RSI (Reference State Init)

**只随机 robot 起始物理姿态, 不随机 mimic ref 时间** [user-decided]:

```python
# Reset:
ref_clip_id, ref_clip = sample({0, 1}, p=[0.5, 0.5])         # 这条 episode 跟踪的 clip
pose_src_clip, pose_src_frame = sample_pose_source()         # 起始姿态来源 (任意 clip 任意帧)

# robot 物理状态 = pose_src 的 (joint_pos, joint_vel, base_pose, base_lin_vel, base_ang_vel)
robot.set_state(pose_src_clip[pose_src_frame])

# mimic 跟踪 ref 始终从 ref_clip 第 0 帧开始
t_offset = 0
cur_frame = 0
```

⚠️ **DIVERGENCE 3 — RSI**: paper / DeepMimic 标准 RSI 是 robot 物理状态 = ref 同帧, 我们解耦了"姿态采样"和"ref 时间采样". 实现简单 (不需要 mid-episode 对齐), 训练增加噪声鲁棒性. paper 没说 RSI 实现细节, 这个偏差预计无害.

**`pose_src` 采样池**:
- 50% 从 ref_clip 任意帧 (i.e., 标准 RSI 行为)
- 50% 从 default standing pose (joint=default, base_pos=spawn) — 增强"任意起点收敛到 ref"鲁棒性

### 2.4 σ_vel 自适应 (是否保留待重测)

backward_004 v_blade@impact = 1.99 m/s. 若 r_g_vel 用统一 σ=0.5:
- forward (‖v̂‖=4.42): σ=0.5, ‖v_blade − v̂‖ 容差仅 11% — 太严
- backward (‖v̂‖=1.99): σ=0.5, ‖v_blade − v̂‖ 容差 25% — 偏松, robot 学到的反手速度可能学不到位

v1 方案: `σ_vel = max(0.3, 0.2·‖v̂_racket‖)`:
- forward: σ = max(0.3, 0.884) = 0.884
- backward: σ = max(0.3, 0.398) = 0.398

⚠️ **DIVERGENCE 4 — σ_vel 自适应**: paper 公式形式 `exp(-‖·‖²/σ²)` 但**没指定 σ 数值**, 也未提自适应. 我们的自适应是为了应对 GVHMR backward 末端速度被滤平的问题. 详见 [REWARD_DESIGN.md](REWARD_DESIGN.md) §3.

**TODO**: 训练第一轮跑出来后 measure r_g_vel 在 forward / backward 两 clip 上的 mean / variance, 决定保留自适应还是回到 paper 风格的固定 σ.

---

## §3. Free 段 cmd 来源 (统一 sampler, paper-aligned)

free 段 (`mimic_active=False`) 由**单一 sampler** 生成. 不调用 `HitterPlanner.update()`, 复用 planner 内部的几何关系.

### 3.1 Sampler 流程

```python
# === 1) 采样击球点 (pelvis frame, 然后转 world) ===
# x: paper 标准 0.4m, 我们 truncated Gaussian (DIVERGENCE 5)
hit_x_local = clip(N(mean=0.40, std=0.08), low=0.25, high=0.60)
# y, z: world rectangle, 与 pelvis_y 相对
hit_y_local = uniform(-Y_DEV, +Y_DEV)            # Y_DEV = 0.5  (paper 没给, 我提案)
hit_z       = uniform(Z_MIN, Z_MAX)              # [0.10, 0.60]  (球台高 + 上下沿)

# 转 world (用当前 base 朝向)
hit_x_w, hit_y_w = base_xy + R_yaw(base_yaw) @ (hit_x_local, hit_y_local)
p_racket = (hit_x_w, hit_y_w, hit_z)

# === 2) swing_type 由 hit_y_local 几何决定 ===
swing_type = 0 if hit_y_local < 0 else 1         # 0=forehand (击球点在 base 右侧), 1=backhand
                                                  # ⚠️ +y_local = LEFT (相对 robot), 右手持拍正手击球 → 球点在 right (-y_local)

# === 3) base_target_xy (与 planner._compute_base_target 一致几何) ===
base_target_x_local = hit_x_local - 0.40         # base 永远在击球点后方 0.40m (paper 标准 stance)
base_target_y_local = hit_y_local + 0.25 if swing_type == 0 else hit_y_local
base_target_xy_w    = base_xy + R_yaw(base_yaw) @ (base_target_x_local, base_target_y_local)

# === 4) v_racket 采样 (大小 + 方向) ===
v_mag   = uniform(2.0, 6.0)                       # paper "amateur 2-6 m/s"
v_yaw   = base_yaw + π + uniform(-40°, +40°)     # 朝向 robot 反向 (球出去) ± 40°
v_pitch = uniform(10°, 60°)                       # 上扬角
v_racket = v_mag * (cos(v_yaw)·cos(v_pitch), sin(v_yaw)·cos(v_pitch), sin(v_pitch))

# === 5) t_to_hit 初值 [user-decided] ===
t_to_hit = truncN(low=0.2, high=1.5, peak_low=0.4, peak_high=0.6)

# === 6) clip_id (free 段也要给, 用于 critic 可能需要的 ref 信息) ===
clip_id = swing_type                              # 几何决定的 swing 选 同方向 clip

# 不需要的字段:
#   is_mimic_phase, strike_window_reward_passed, hit_actually_landed → 全部已删除
```

### 3.2 工作空间约束 [v1 §3.2 沿用]

Sampler 输出**必须**满足 planner workspace:
- `hit_x_local ∈ [0.25, 0.60]` (truncated Gaussian 已保证)
- `hit_y_local ∈ [-0.5, +0.5]` (右臂 + 左臂可达)
- `hit_z ∈ [0.10, 0.60]` (球台高 0.76m? — 需要核 robot 站立时 base_z + hit_z 实际高度)
- `‖v_racket‖ ∈ [2, 6]` m/s

不满足 → reject + resample (truncated Gaussian 已保证, 主要拒绝点是 v_yaw / v_pitch 极端组合).

⚠️ Workspace 常量集中在 `PADDLE_WORKSPACE_CONFIG` dict, planner.py 和此 sampler 都引用.

### 3.3 ⚠️ DIVERGENCE 5 — x sampling

**paper**: x 固定 = 0.4m (论文 striking plane 平面假设).
**我们**: truncated Gaussian mean=0.4 std=0.08 clip [0.25, 0.60].

**理由 (我提案 → user 确认)**:
- deploy 时 perception 给的 hit_x 不会刚好 = 0.4 (有估计误差 + 球路径不同)
- 只在 x=0.4 训练 → 0.5m 时 robot 不会动
- 0.08 std 让 mean ± 1σ ≈ [0.32, 0.48] 占 ~68% 样本, mean ± 2σ ≈ [0.24, 0.56] 占 ~95%, x=0.4 仍然是高频值
- clip [0.25, 0.60] 防止极端: 0.25m 太近不舒展, 0.60m 太远 robot 触碰不到

⚠️ **TODO**: 如果训练发现 x=0.4 vs 0.6 reward 表现差异极大, 考虑 hybrid (80% fixed 0.4 + 20% uniform [0.25, 0.6]).

---

## §4. Cmd 时序

### 4.1 完整流程 (单 clip mimic + 多次 rally)

```
═══ episode reset (RSI) ═══
│ curriculum 决定起点 (§4.4):
│   - 早期: 100% mimic 起步 (Phase M)
│   - 后期: 30% free 起步 (Phase F)
│
═══ Path A: mimic 起步 ═══
│
│  Phase M1: pre-swing  (cur_frame ∈ [0, impact+5])
│    [mimic_active=True]
│    cmd 来自 npz, 详见 §2.2
│    t_to_hit 从 (impact/fps) 单调下降到 -0.1
│
├──── strike_window 关闭 (cur_frame > impact + 5) ────
│    base_target_xy 切换: clip[impact].pelvis → clip[-1].pelvis
│    t_to_hit 继续递减 (无冻结)
│
│  Phase M2: follow-through  (cur_frame ∈ [impact+6, T_clip])
│    [mimic_active=True]
│    r_i 仍激活 (跟踪 clip 后半段 ref)
│    r_g sparse 关闭 (abs(t_to_hit) > 0.1), r_g_base dense
│
═══ mimic clip 结束 (cur_frame = T_clip) ═══
│ ⚠️ v2 不做 multi-clip chain. 直接进 free play.
│
═══ Free play (mimic_active=False, 多次 rally 重复) ═══
│
│  Phase F0: gap1 (球飞向对手)
│    duration = truncN(low=0.2, high=1.5, peak_low=0.4, peak_high=0.6)  [user]
│    cmd 字段 hold last value, t_to_hit 仍递减 (变更负)
│    base_target_xy = 击球完那一刻 base_xy (frozen) [user]
│    r_i = 0, r_g sparse = 0, r_g_base dense, r_r 全程
│
│  Phase F0.5: 对手击球 (固定 0.1s)
│    [同上, 没有新 cmd]
│
├──── new cmd 到达 (free sampler) ────
│    cmd 字段由 §3 sampler 生成
│    cmd.t_to_hit = truncN[0.2, 1.5] peak [0.4, 0.6]
│    swing_type / clip_id 由 sampler 几何决定
│
│  Phase F2: pre-strike  (new cmd 到达 → strike_window 关闭)
│    cmd.t_to_hit 随仿真时间每 step 减 dt=0.02s
│    Cmd 噪声每 step 在 world frame 注入 (§8)
│    abs(t_to_hit) <= 0.1 时 r_g sparse 激活
│
├──── strike_window 关闭 (t_to_hit < -0.1) ────
│    base_target_xy 切换为击球完那一刻 base_xy (frozen)
│    t_to_hit 继续递减 (不冻结)
│
│  Phase F3: free follow-through
│    准备下一 rally
│
═══ 重复 (Phase F0 + F0.5 + new cmd + F2 + F3) 直到 ═══
   - 10s timeout (paper Sec V-B1)
   - 摔倒 termination
   - episode 内击球次数无上限
```

### 4.2 关键决定汇总

| 时序点 | 行为 | 来源 |
|---|---|---|
| RSI | 解耦姿态 vs ref 时间, t_offset = 0 | [user-decided] |
| Mimic 段 base_target | M1: clip[impact].pelvis_xy; M2: clip[-1].pelvis_xy | [user-decided] |
| Mimic clip 结束行为 | 直接进 free play (无 chaining) | [v2 user-decided] |
| t_to_hit 击球后行为 | **继续递减到负值, 不冻结** | [v2 user-decided] |
| gap1 (球飞向对手) | truncN[0.2, 1.5] peak [0.4, 0.6] | [user] |
| 对手击球 | 固定 0.1s | [user, paper-derived] |
| New cmd 到达时机 | 对手击球完那一刻立刻到达 | [user] |
| New cmd t_to_hit 初值 | truncN[0.2, 1.5] peak [0.4, 0.6] | [user] |
| Cmd 切换跳变 | **硬切** (无平滑) — 兼作 mimic 稳定性训练 | [v2 user-decided] |
| Free 段击球后 base_target | 击球完那一刻的 base_xy (frozen) | [user] |
| Episode 击球次数 | 无上限, 10s timeout 自然结束 | [user] |

### 4.3 Episode 终止

- 10s timeout (paper Sec V-B1) — episode-level
- 摔倒 (pelvis_height < 0.4 等) — termination
- cmd 本身**不**触发 termination

### 4.4 Episode 起点 curriculum

| Iter range | mimic_start_prob | 说明 |
|---|---|---|
| 0 – 8k    | 1.00 | warmup, 全部 mimic 起步 |
| 8k – 25k  | 1.00 → 0.7 | 渐增 free 起步 |
| 25k+      | 0.7 | 30% episode 直接 free 起步 |

**Free 起步细节**:
- robot 物理状态 = default standing pose
- 立刻调用 §3 free sampler 生成首 cmd
- 跳过 Phase M1/M2

---

## §5. Strike timing (无 flag 状态机)

### 5.1 v2 简化: 直接用 t_to_hit gate

```python
# v1: 双 flag 状态机 (strike_window_reward_passed + hit_actually_landed) → 删除
# v2: r_g sparse gate 直接读 t_to_hit:

strike_window_active = (abs(cmd.t_to_hit) <= 0.1)   # 11 帧 @ 50Hz, ±5 帧 [paper Sec V-B2]
r_g_sparse = strike_window_active * (...)
```

**优点**:
- 无状态切换 — t_to_hit 是数值, abs() 比较实现一行
- 无 off-by-one (flag flip 时机争议消失)
- t_to_hit 还可以继续给 actor obs (`obs.t_to_hit = cmd.t_to_hit`), 击球后值变负, actor 自然学到"过期"

**缺点 (相比 v1 flag)**:
- gap 期 cmd hold last value, 此时 t_to_hit 仍在递减, 可能跨越 -0.1 → r_g sparse 误激活? **不会**, 因为 cmd hold = old cmd, 此时 cmd 已经 stale, abs(t_to_hit) > 0.1 自然 (旧 cmd 的 t_to_hit 早就 < -0.1).
- 但若 episode 第一拍非 mimic 起 (Phase F 起), 初始 t_to_hit 可能在 [0.2, 1.5], 这没问题.

### 5.2 mimic_active 与 strike_window 的正交关系

| mimic_active | abs(t_to_hit)<=0.1 | 阶段 | r_i | r_g sparse | r_g_base |
|:---:|:---:|---|:---:|:---:|:---:|
| T | F | mimic pre-swing 前段 / follow-through | ✓ | ✗ | ✓ (target=clip[impact].pelvis 或 clip[-1].pelvis) |
| T | T | mimic 击球瞬间 (cur_frame ∈ [impact-5, impact+5]) | ✓ | ✓ | ✓ |
| F | F | gap / free pre-strike 前段 / follow-through | ✗ | ✗ | ✓ (target=sampler 给 或 击球完 frozen) |
| F | T | free 击球瞬间 | ✗ | ✓ | ✓ |

⚠️ **DIVERGENCE 6 — r_g_base dense 全程**: paper Sec V-B 说 base_pos reward "before strike", 我们 dense 全程 (含 follow-through + gap). 理由: free 段 phase 2 reward 太 sparse, robot 容易"冻住不动"陷局部最优. 详见 [REWARD_DESIGN.md](REWARD_DESIGN.md) §3.4.

---

## §6. Cmd → Reward Gating 总览

cmd 字段控制 reward sub-term 的激活 (详细公式见 [REWARD_DESIGN.md](REWARD_DESIGN.md)):

| cmd / state 字段 | gate 哪些 reward | 作用 |
|---|---|---|
| `mimic_active` (episode-state, NOT cmd) | r_i 全部 sub-terms | mimic clip 内 r_i 才有意义 |
| `abs(t_to_hit) <= 0.1` (从 cmd.t_to_hit 推) | r_g_pos, r_g_vel, r_g_ori | sparse, ±5 帧窗内才计 |
| (none, dense 全程) | r_g_base | 防止 free play reward 太 sparse |
| `cmd.swing_type` (内部) | (无直接 gate) | 影响 r_i 选哪个 ref clip / r_g_base 几何 |
| `cmd.clip_id` (内部) | (无直接 gate) | mimic 段决定 r_i 用哪个 clip 的 q̂ |

---

## §7. 部署对齐 (HitterPlanner → cmd)

### 7.1 World frame 存储 + base-relative adapter

cmd 内部全部用 **world frame** 存储 (与 planner 输出一致). actor obs 经过 adapter 转 base-relative:

```python
def world_to_base_rel(p_world, base_pos, base_quat):
    return quat_rotate_inverse(base_quat, p_world - base_pos)

obs.p_racket_rel       = world_to_base_rel(cmd.p_racket, base_pos, base_quat)
obs.v_racket_rel       = quat_rotate_inverse(base_quat, cmd.v_racket)
obs.base_target_dxy    = (cmd.base_target_xy - base_pos[..., :2])    # planar
obs.t_to_hit           = cmd.t_to_hit                                 # scalar
# ⚠️ swing_type / clip_id 是否进 obs 待 critic obs 设计后定 (§9 DIVERGENCE 7)
```

### 7.2 部署时 cmd 切换跳变 — 硬切

```
理由:
- planner 内部已有 _hold_or_none + swing_lock_frames=50 稳定机制
- 真实 rally 中 cmd 之间本来就是离散事件
- 任何平滑都引入"假"中间 cmd, 训练 vs 部署不一致
- ⭐ 硬切兼作 mimic 稳定性训练: actor 必须学会从任意 cmd 突变恢复
```

---

## §8. Cmd 噪声 (per-step persistent, 小)

### 8.1 注入方式

**每 step 重新加噪 (持续性)**, 在 world-frame cmd 上加, 之后再走 obs adapter [user-decided]:
- reward 计算用每 step 的 noisy cmd
- obs 也用同一份 noisy cmd
- 物理意义: 模拟 vision/perception 估计噪声 (每帧测量都不同)
- ⚠️ **量级一定要小** (用户原话). 见 §8.3

```python
# 每 step 在 cmd 上重新加噪 (mimic 段 + free 段都加, 在 obs adapter 之前):
def apply_cmd_noise(cmd, σ):
    cmd.p_racket       += gauss(0, σ_p,    shape=(E, 3))
    cmd.v_racket       += gauss(0, σ_v,    shape=(E, 3))
    cmd.base_target_xy += gauss(0, σ_base, shape=(E, 2))
    cmd.t_to_hit       += gauss(0, σ_t,    shape=(E,))
    # swing_type / clip_id 不加噪 (离散字段)
    return cmd
```

⚠️ 噪声只在 obs / reward 计算前加, 不修改 cmd manager 的 underlying state — underlying cmd 仅在切换 cmd 时变化:
- `cmd_buffer = sampler_output` (整个 cmd 期间不变)
- `cmd_observed = cmd_buffer + per_step_noise()` (每 step 重新采样)
- reward 用 `cmd_observed`, obs 用 `cmd_observed`

### 8.2 Curriculum schedule (草案)

| Iter range | σ_p | σ_v | σ_base | σ_t | 备注 |
|---|---|---|---|---|---|
| 0 – 8k    | 0       | 0       | 0       | 0       | warmup, 干净 cmd 学核心动作 |
| 8k – 25k  | 0 → 0.02 | 0 → 0.2 | 0 → 0.05 | 0 → 0.02 | 渐进引入 |
| 25k+      | 0.02    | 0.2     | 0.05    | 0.02    | 部署级噪声 (planner 测量误差量级) |

详细 schedule 见 [CURRICULUM_DESIGN.md](待写).

### 8.3 噪声量级原则

- **不要太大** (用户原话): 太大噪声直接破坏击球观测, policy 学不到精准动作
- 上表数值已按此原则 (0.02 m / 0.2 m/s / 0.05 m / 0.02 s 都是较小级别)
- **课程开启**, 不一开始就加 (干净 cmd 帮助 policy 先收敛)

---

## §9. ⚠️ Paper Divergence 索引 (核心审视点)

这一节集中列出我们与 paper 不一致的设计选择. **每个 ⚠️ 都要在训练初期 monitor reward 曲线判断是否需要回退**.

| # | 项 | paper | 我们 | 理由 / 风险 |
|---|---|---|---|---|
| 1 | clip 长度 | 94 frames @ 50Hz, impact at 43 (1.88s) | forward 82帧 imp=37 / backward 64帧 imp=20 | 数据可得限制. backward ratio=0.31 偏离风险大 |
| 2 | backward v_blade | (paper 无明确数值) | 1.99 m/s (vs forward 4.42 m/s) | GVHMR 末端速度被滤平. 应对: σ_vel 自适应 |
| 3 | RSI 设计 | (DeepMimic 标准: 物理姿态 = ref 同帧) | 解耦 (姿态 vs ref 时间), t_offset=0 | 实现简单, 训练增加噪声鲁棒性. 预计无害 |
| 4 | σ_vel 自适应 | 公式 `exp(-‖·‖²/σ²)` 但 σ 数值未指定 | `max(0.3, 0.2·‖v̂‖)` | 应对 backward 慢拍. 训练第一轮后重测决定保留 |
| 5 | x sampling | fixed 0.4m | truncated Gaussian mean=0.4 std=0.08 clip [0.25, 0.60] | deploy perception 给非 0.4 时仍泛化 |
| 6 | r_g_base dense 全程 | "before strike" only | dense 全程 (含 follow-through + gap) | 防止 free play reward 太 sparse 摆烂 |
| 7 | swing_type / clip_id 是否进 obs | paper actor obs 不含 (Table I) | **TBD** (待 critic obs 设计) | actor 应该不含 (符合 paper); critic 是否含待定 |
| 8 | Multi-clip chaining | 无 | v1 有, **v2 移除** | paper-aligned (移除 OK) |
| 9 | Strike flag 状态机 | (paper 用 sparse window 描述, 没明确 flag) | v2 直接用 abs(t_to_hit)<=0.1 | 实现简单, 等价 |
| 10 | hit_actually_landed flag | 无 | v2 移除 | paper-aligned (移除 OK) |
| 11 | is_mimic_phase 进 actor obs | Table I 不含 | v2 不进 actor obs | paper-aligned ✓ |

⚠️ **训练监控 checklist**:
- DIVERGENCE 1: 看 mimic 段 r_i 在 backward_004 的 ratio=0.31 处 (impact 在 31%) 是否反常 (mimic 段时长分布不对称)
- DIVERGENCE 2 + 4: r_g_vel 在 backward / forward 上的 mean / variance 是否平衡
- DIVERGENCE 5: hit_x ∈ [0.25, 0.32] 区间的 reward (1σ 外侧) 是否显著低于 [0.32, 0.48] (mean ± σ)
- DIVERGENCE 6: free play 段 r_g_base 累积分布 (dense 全程 vs sparse-only 对照实验) — 留作 ablation

---

## §10. 已锁定的设计决定 (汇总表)

| 项 | 决定 | 来源 |
|---|---|---|
| `n̂_target` 不存独立字段 | 从 `v̂_racket` 推导 | paper Sec IV-C |
| Cmd 字段命名 | `v_racket` (paper) 而非 `v_paddle` | [v1 D2] |
| 时间字段 | `t_to_hit` (相对, 单调减), 击球后继续负, **不冻结** | [v2 user-decided] |
| `paddle_normal` 字段 | 不暴露 | [v1 D2] |
| `is_mimic_phase` 字段 | **不进 cmd struct**, episode-internal state | [v2 user-decided] |
| `strike_window_reward_passed` flag | **删除**, 用 abs(t_to_hit)<=0.1 替代 | [v2 user-decided] |
| `hit_actually_landed` flag | **删除** | [v2 user-decided] |
| `swing_type` cmd 字段 | 保留 (内部用), 不进 actor obs | [v2 user-decided + paper Table I] |
| `clip_id` cmd 字段 | 保留 (内部用), 是否进 obs 待 critic obs 设计 | [v2 TBD] |
| `base_target_xy` 显式提供 | 不从其他字段推导 | [v1 user] |
| Mimic 段 cmd | 全部从 npz 合成, 2 expert clips (forward_001 + backward_004) | [v2 user-decided] |
| Multi-clip chain | **移除** | [v2 paper-aligned] |
| Free 段 sampler | 单一 sampler, 用 planner 几何, 不调 update() | [v1 user] |
| 工作空间约束 | sampler 必满足 planner workspace | [v1 D6] |
| RSI 设计 | 只采姿态, t_offset 强制 = 0 | [v1 user] |
| x sampling | truncated Gaussian mean=0.4 std=0.08 clip [0.25, 0.60] ⚠️ DIVERGENCE 5 | [v2 user-decided] |
| Episode 起点 | curriculum: mimic 起 → free 起占比上升 | [v1 user] |
| Episode 击球次数 | 无上限 | [v1 user] |
| Cmd timing 模型 | gap1 + 0.1 + new cmd 立刻到达 (无 gap2) | [v1 user] |
| gap1 采样 | truncN[0.2, 1.5] peak [0.4, 0.6] | [v1 user] |
| 对手击球 | 固定 0.1s | [v1 user] |
| New cmd t_to_hit 初值 | 独立 truncN[0.2, 1.5] peak [0.4, 0.6] | [v1 user] |
| Strike sparse window | abs(t_to_hit) <= 0.1 (±5 帧 @ 50Hz, 11 帧) | [paper Sec V-B2] |
| Mimic base_target | M1: clip[impact].pelvis; M2: clip[-1].pelvis | [v1 user] |
| Free base_target | F0/F1/F3: 击球完 base_xy frozen; F2: sampler 给 | [v1 user] |
| Cmd 切换跳变 | **硬切** (无平滑) — 兼作 mimic 稳定性训练 | [v2 user-decided] |
| World frame 存储 + adapter | reward 用 world, obs 转 base-rel | [v1 D1] |
| Cmd 噪声注入位置 | world frame, **每 step 重新加** (持续性) | [v2 user-decided] |
| Cmd 噪声量级 | 小 (σ_p=0.02 等), curriculum 开启, **绝不能太大** | [v2 user-decided] |
| σ_vel 自适应 | `max(0.3, 0.2·‖v̂‖)` ⚠️ DIVERGENCE 4 | [v1 D7, 训练后重测] |
| Free 段 v_racket 方向采样 | yaw=robot反向±40°, pitch=10°-60° | [v1 user] |
| Termination 信号 | `alive_reward = +0.1` per step | [v1 user] |

---

## §11. 实现 TODO

### 11.1 csv_to_npz_pingpong.py ✅ 已完成
- [x] `body_names: list[str]` 字段
- [x] `swing_type: int8` 字段
- [x] `--task_name` CLI arg

### 11.2 Expert clip 选择 ✅ 已完成
- [x] `expert/forward/forward_001.npz` (paper-aligned ratio=0.451, v=4.42 m/s)
- [x] `expert/backward/backward_004.npz` (ratio=0.313 ⚠ DIVERGENCE 1, v=1.99 m/s ⚠ DIVERGENCE 2)
- [x] `scripts/pingpong_data_process/select_x04_clips.py` (筛选脚本)

### 11.3 mdp/commands.py (待写)
- [ ] `HitCommand` dataclass + manager (按 §1, **不含** is_mimic_phase / strike_window_reward_passed / hit_actually_landed)
- [ ] Mimic 段 sampler (从 2 expert npz, 按 §2.2)
- [ ] RSI 解耦 (姿态 vs ref 时间, 按 §2.3)
- [ ] Free 段 unified sampler (按 §3, 用 planner 几何, **truncated Gaussian x**)
- [ ] Workspace 约束 reject + resample
- [ ] Gap timing (gap1 + 0.1, new cmd 到达逻辑, 按 §4)
- [ ] Episode 起点 curriculum (mimic vs free 起步)
- [ ] base_target_xy 分段切换 (按 §10)
- [ ] World-frame cmd 噪声注入 (按 §8.1, **每 step 重新加**)

### 11.4 mdp/observations.py (待写, 等 OBSERVATION_DESIGN.md 定稿)
- [ ] World → base-relative adapter (按 §7.1)
- [ ] Actor obs 字段 (paper Table I 对齐, **不含** swing_type / clip_id / mimic_active / hit_actually_landed)
- [ ] Critic privileged obs 字段 (TBD)

### 11.5 mdp/curriculums.py (待写)
- [ ] σ_pos 调度 (REWARD_DESIGN §5)
- [ ] Cmd 噪声 σ 调度 (§8.2)
- [ ] Mimic 起步 prob 调度 (§4.4)

### 11.6 工程层面常量集中
- [ ] `PADDLE_WORKSPACE_CONFIG` dict (X mean/std/clip, Y_DEV, Z_MIN/MAX, V range, gap range)
- [ ] planner.py + commands.py 都引用这个 config
- [ ] 避免 sampler 和 planner 走偏

### 11.7 Sanity tests
- [ ] gap 期 obs 平滑 (t_to_hit 不会 NaN, 持续递减)
- [ ] backward clip cmd.v_racket 量级 + r_g_vel 有意义 (σ_vel 自适应)
- [ ] cmd 切换跳变在 actor obs 上的 magnitude (是否需要 obs clipping)
- [ ] x sampling 分布 (truncated Gaussian 经验分布 vs 理论)
- [ ] strike sparse gate 边界 (abs(t_to_hit) = 0.1 那一帧 on/off)
- [ ] Free play 起步 episode (no mimic) reward landscape 没有断崖

---

## §12. Cross-doc 联动

### 12.1 REWARD_DESIGN.md 同步修订点 (v2)
1. 删除 `strike_window_reward_passed` / `hit_actually_landed` 引用
2. r_g sparse gate 表达式: `strike_window AND ¬strike_completed` → `abs(t_to_hit) <= 0.1`
3. ℬ 排除策略**重新审视**: expert 数据噪声小, 是否还需要排除 blade / wrist?
4. σ_vel 自适应**保留 vs 回 paper 固定 σ** 待第一轮训练后决定
5. r_g_base dense vs paper "before strike only" — DIVERGENCE 6, 留 ablation TODO

### 12.2 OBSERVATION_DESIGN.md (待写)
- Actor obs paper Table I 严格对齐 (不含 swing_type / clip_id / mimic_active / hit_actually_landed)
- Critic privileged obs 设计 (待用户决策)
- clip_id 编码方式确定 (取决于 critic obs 是否需要)

### 12.3 EVENT_DESIGN.md (待写)
- RSI pose source sampling
- Mimic vs free 起步决策
- Domain randomization (paper Sec V-B3)

### 12.4 CURRICULUM_DESIGN.md (待写)
- σ_pos 紧度调度
- Cmd 噪声调度 (本 doc §8.2)
- Mimic 起步 prob 调度 (本 doc §4.4)

---

## §End. v2 设计完整性 review checkpoint

读完 12 节, 实现前需要确认:
- [x] 2 expert clips 已选定 (forward_001, backward_004)
- [x] 删除 v1 三个 flag (strike_window_reward_passed, hit_actually_landed, is_mimic_phase from cmd)
- [x] Multi-clip chaining 移除
- [x] x sampling truncated Gaussian (DIVERGENCE 5)
- [x] Cmd 噪声 per-step persistent
- [x] 硬切 cmd 切换 (兼 mimic 稳定性训练)
- [x] DIVERGENCE 索引集中 §9 — 训练初期 monitor 列表
- [ ] **clip_id 编码 / 是否进 obs** 待 critic obs 设计后定 (§9 DIVERGENCE 7)
- [ ] **σ_vel 自适应 vs 固定** 待第一轮训练后决定
- [ ] **r_g_base dense vs sparse** 留作 ablation TODO
