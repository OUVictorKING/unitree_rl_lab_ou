# Command Design — G1 23-DoF Paddle WBC

> Doc 范围: cmd 数据结构, 各阶段 cmd 来源, 时序逻辑, 与 planner 对齐, 噪声 curriculum, 多 clip 串接.
> Reward / Observation / Event / Termination 详细机制见各自独立文档.
> 来源标记: `[paper]` / `[paper-derived]` / `[我提案]` / `[user-decided]`
> Paper: HITTER, arXiv:2508.21043v2.

---

## §0. Cmd 用途定位

WBC policy 接收的 "任务指令". 训练时由 sampler 合成, 部署时由上层 [HitterPlanner](mdp/planner.py) 输出. 包含 **击球点 + 拍速 + 击球时刻 + 站位 + 正反手 + 阶段标志** 6 类核心信息.

[paper Table I 列出 cmd-side observation]: `p̂_racket, v̂_racket, t_strike, p̂_base diff`.

设计要点:
- cmd 是 reward (r_g) 计算的输入, 也是 actor obs 的输入 — cmd 接口一变, reward + obs 都要同步改
- 训练 cmd ↔ 部署 cmd 接口必须一致 (即和 `HitterPlanner` 输出对齐), 否则 sim2real 无法复用 policy
- cmd 存储在 world frame; obs 端有 adapter 转 base-relative 供 actor 使用
- noise 在 world-frame cmd 上注入 (cmd 生成那一刻就加), reward + obs 都用 noisy cmd, 保证 train/eval 一致

---

## §1. Cmd 数据结构 (最终)

```python
@dataclass
class HitCommand:
    # === paper Table I 4 大核心字段 (world frame) ===
    p_racket:         torch.Tensor  # (E, 3)   期望击球点 world pos
    v_racket:         torch.Tensor  # (E, 3)   期望拍速 world (大小+方向)
                                    # NOTE: paddle_normal NOT exposed —
                                    #   r_g_ori 现场推导 n̂ = v_racket / ‖v_racket‖
                                    #   [paper Sec IV-C: "racket plane perpendicular to its velocity"]
    t_to_hit:         torch.Tensor  # (E,)     距离击球还有多少秒 (相对时间, 不是绝对 episode 秒)
                                    #   strike_window_reward_passed=True 时 = -1.0 sentinel (frozen)
    base_target_xy:   torch.Tensor  # (E, 2)   期望脚下站位 world xy

    # === 阶段标志 / sampler 内部状态 ===
    is_mimic_phase:   torch.Tensor  # (E,)     bool. True = 当前 cmd 对应 mimic clip 段
                                    #   部署时该字段 = False (上层 planner 不知道 mimic)
                                    #   训练时由 episode 内 mimic_active 自动同步
    strike_window_reward_passed:    # (E,)     bool. True = 此 cmd 的 r_g 时间窗(±3 帧 = ±0.06s)已关闭
        torch.Tensor                #   gate r_g 的 sparse 项 (pos/vel/ori)
                                    #   ⚠️ 改名自 strike_completed (语义更准: 是"窗口已过", 不是"成功击中")
    hit_actually_landed:            # (E,)     bool. True = 在 strike_window 内确实命中拍面
        torch.Tensor                #   几何检测 (见 §5.2): in-plane d<0.05m + normal d<0.015m
                                    #   仅用于 obs / diagnostic / curriculum, 不 gate reward
    swing_type:       torch.Tensor  # (E,)     int8. 0=forehand, 1=backhand.
                                    #   影响 base_target_xy 几何 (见 §3 sampler)
```

### 1.1 关于 `n̂_target = v̂_racket / ‖v̂_racket‖`

paper Sec IV-C 关系:
- planner 内部先算法向 `u = (v_o − v_i) / ‖v_o − v_i‖` (出球 - 来球归一化)
- 然后用反弹模型 `v̂_racket = ((v_o·u + C_r·v_i·u) / (1 + C_r)) · u` 解出拍速大小, 整个 `v̂` 沿 `u` 方向
- 因此 `v̂_racket` 已携带法向信息, `n̂ = v̂/‖v̂‖` 直接得到

cmd **不另存** `paddle_normal`. r_g_ori 计算时现场推导. 例外: 未来训练侧旋等非垂直击球, paper 假设不再成立, 那时再加.

### 1.2 字段命名 / 单位与 planner 输出对齐

planner.py update() 返回 19 字段, cmd 只挑其中 5 个 + 加 4 个 episode-state 字段:

| cmd 字段 | planner 字段 | 单位 / 备注 |
|---|---|---|
| `p_racket` | `hit_pos` | world m |
| `v_racket` | `v_paddle` | world m/s (planner 字段名是 v_paddle, cmd 用 v_racket 保持 paper 命名) |
| `t_to_hit` | `t_to_hit` | sec, 相对 (与 planner 一致, **不用** paper 的 absolute t_strike) |
| `base_target_xy` | `base_target_xy` | world m, 见 [planner._compute_base_target](mdp/planner.py#L942-L984) |
| `swing_type` | `swing_type` | 0=forehand, 1=backhand. planner 在 swing_lock_frames=50 后锁定 |

planner 不输出的 cmd 字段 (训练侧补):
- `is_mimic_phase`: episode-level state, 部署 = False
- `strike_window_reward_passed`: 由 t_to_hit 跨过 -0.1 自动推导, 部署 = (t_to_hit < -0.1)
- `hit_actually_landed`: 训练侧几何检测, 部署侧不需要 (但接口保留以对齐 obs shape)

planner 输出但 cmd 不用的 12 字段 (planner 内部诊断 / 中间量): `base_pos, base_quat, paddle_normal, v_ball_in, v_ball_out, target_land, n_buf, plan_mode, x_hit_used, stale_age_s, hold_reason, bounced, bounces, traj_p`.

---

## §2. Mimic 段 cmd 来源 (npz 直接合成)

mimic 段 (`is_mimic_phase=True`) cmd **全部从 npz 合成**, 与 r_i 的参考动作严格对齐.

### 2.1 字段合成

```python
clip_id      = sampled at episode reset (随机从 motion pool 抽)
t_offset     = 0   # ⚠️ 始终从 clip 第 0 帧开始播 [GAPG user-decided]
impact       = clip.impact_frame             # npz 字段, 已标注

# Static cmd 字段 (整个 mimic 段 episode-life 不变)
p_racket         = clip.body_pos_w[impact, BLADE_IDX]       # blade body world pos
v_racket         = clip.body_lin_vel_w[impact, BLADE_IDX]   # blade body world lin vel
swing_type       = clip.swing_type                          # 0=forward_hand, 1=backward_hand
is_mimic_phase   = True

# Dynamic cmd 字段 (随 cur_frame 推进)
cur_frame        = step_in_episode                          # 0, 1, 2, ...
t_to_hit         = (impact - cur_frame) / fps               # 单调下降至 0 再继续负
strike_window_reward_passed = (cur_frame > impact + 5)      # ⚠️ strike_window 关闭后 flip (±5 帧)
hit_actually_landed         = geometric_check(blade, ball)  # 见 §5.2

# base_target_xy: 分段切换 [GAPJ user-decided]
if cur_frame <= impact + 5:                  # mimic pre-swing + strike_window (±5 帧)
    base_target_xy = clip.body_pos_w[impact, PELVIS_IDX, :2]
else:                                        # mimic follow-through + return-to-ready
    base_target_xy = clip.body_pos_w[-1, PELVIS_IDX, :2]    # clip 末帧 pelvis xy
```

`BLADE_IDX = 24`, `PELVIS_IDX = 0` (npz body_pos_w shape = (T, 25, 3)). 如 npz 含 `body_names` 字段, 优先按 name 找 idx 避免漂移.

### 2.2 RSI 修订 [GAPG user-decided]

**RSI 只随机 robot 起始物理姿态, 不随机 mimic ref 时间**:

```python
# Reset:
ref_clip_id, ref_clip = sample_motion_pool()         # 当前 episode 跟踪的 clip
pose_src_clip, pose_src_frame = sample_pose_source() # 起始姿态来源 (可以来自任意 clip 任意帧)

# robot 物理状态 = pose_src 的 (joint_pos, joint_vel, base_pose, base_lin_vel, base_ang_vel)
robot.set_state(pose_src_clip[pose_src_frame])

# mimic 跟踪 ref 始终从 ref_clip 第 0 帧开始
t_offset = 0
cur_frame = 0
```

**含义**:
- robot 初始姿态 ≠ ref 第 0 帧姿态 (有 mismatch, 强迫 policy 学"从任意起点收敛到 ref")
- 跟 paper / DeepMimic 标准 RSI 略有不同 — 我们解耦了"姿态采样"和"ref 时间采样"
- 减少了实现复杂度: 不需要在 episode 中段对齐 ref 时间和 cur_frame
- 训练 robust 度: 起始姿态噪声让 policy 不依赖完美 reset

### 2.3 backward clip 的 v_racket 量级失真 [D7 测量]

| 子集 | n | 中位 ‖v_blade‖@impact | ±5fr 窗最大 |
|---|---|---|---|
| forward (forehand) | 85 | **4.10 m/s** | 4.99 m/s |
| backward (backhand) | 75 | **1.99 m/s** | 2.72 m/s |

forward clip 的 v_blade 跟 paper "业余-中级 2-6 m/s" 对得上, 可信. backward clip 慢一倍, 接近 GVHMR 噪声底.

**应对**: r_g_vel 的 σ 自适应 — `σ_vel = max(0.3, 0.2·‖v̂_racket‖)`. 跑慢拍 cmd 容差小, 跑快拍 cmd 容差按比例放大. 详见 [REWARD_DESIGN.md](REWARD_DESIGN.md) §3.

⚠️ 数据集层面的 TODO: backward 数据集后续若发现训练效果不达预期 (反手击球速度上不去), 重新采集大幅度反手 mp4 重训.

### 2.4 多 mimic clip 串接 (新, GAPL user-decided)

**目的**: 让 policy 学"动作衔接" (e.g., forehand 击球完 → 直接接 backhand 击球, 不经过 free-play stance).

```python
# Per-env at reset, decide whether to use multi-clip chaining
if random() < CHAIN_PROB:                    # CHAIN_PROB 由 curriculum 控制 (见下)
    n_clips = random.randint(2, 3)           # 串接 2-3 个 clip
    clip_chain = [sample_motion_pool() for _ in range(n_clips)]
else:
    clip_chain = [sample_motion_pool()]      # 单 clip

current_clip_idx = 0
```

**串接行为**:
- 上一 clip 播完最后一帧, 立刻切换到下一 clip 第 0 帧 (无 transition / blending)
- mimic ref 跟着切, cmd 也跟着切 (新 clip 的 impact_frame, p_racket, v_racket 重新计算)
- robot 物理状态保持连续 (不 teleport)
- 单个串接 episode 内会触发多次 mimic 段 cmd 切换

**Curriculum schedule** (草案):

| Iter range | CHAIN_PROB | 说明 |
|---|---|---|
| 0 – 8k    | 0.3 (高)  | 早期强调串接学习 |
| 8k – 25k  | 0.3 → 0.1 | 渐降 |
| 25k+      | 0.1       | 保持非零, 维持衔接能力 |

⚠️ 串接 episode 比单 clip episode 长 (2-3 倍), 与 10s timeout 可能冲突. 实现时若 chain 总长 > 10s, **截断**到 10s timeout 自然结束.

---

## §3. Free 段 cmd 来源 (统一 sampler)

free 段 (`is_mimic_phase=False`) 由**单一 sampler** 生成. 不再分 80/20 planner-style + virtual [GAPC user-decided] — 击球点在 workspace 内随机采样已经天然打散了 swing_type ↔ base_y 的相关性.

### 3.1 Sampler 流程 [GAPB user-decided]

**关键**: 不调用 `HitterPlanner.update()`. 只**复用 planner 内部的几何关系** (击球点 ↔ base 站位 + swing_type 决策).

```python
# 1) 采样击球点 (在 planner workspace 内)
hit_x = X_HIT + uniform(-0.30, +0.30)        # X_HIT = -1.50 (planner.x_hit_default)
hit_y = uniform(base_y_current - Y_DEV, base_y_current + Y_DEV)  # Y_DEV = 1.0
hit_z = uniform(Z_MIN, Z_MAX)                # Z_MIN=0.10, Z_MAX=0.60

p_racket = (hit_x, hit_y, hit_z)

# 2) swing_type 由 hit_y vs base_y 决定 (与 planner._compute_base_target 一致)
#    +y=LEFT 世界, robot 右手持拍:
swing_type = 0 if hit_y < base_y_current else 1   # 0=forehand, 1=backhand

# 3) base_target_xy 由几何关系算 (与 planner 同公式)
base_target_x = hit_x - 0.40                  # 击球点前方 40 cm 站定
base_target_y = hit_y + 0.25 if swing_type == 0 else hit_y
base_target_xy = (base_target_x, base_target_y)

# 4) v_racket 采样 (大小 + 方向)
v_mag       = uniform(2.0, 6.0)               # paper 2-6 m/s
v_yaw       = uniform(robot_facing_dir ± 40°) # 朝 robot 反向 ± 40°
v_pitch     = uniform(10°, 60°)               # 上扬角
v_racket    = mag * (cos_yaw·cos_pitch, sin_yaw·cos_pitch, sin_pitch)

# 5) t_to_hit 独立采样 [GAPA user-decided]
t_to_hit = truncN(low=0.2, high=1.5, peak_low=0.4, peak_high=0.6)

# 6) 阶段 flag
is_mimic_phase              = False
strike_window_reward_passed = False
hit_actually_landed         = False
```

### 3.2 工作空间约束 [D6, 仍生效]

Sampler 输出**必须**满足 planner workspace constraints:
- `hit_x` 在 `X_HIT ± 0.30` (球台对侧 robot 一侧)
- `hit_y` 在 `base_y_current ± Y_DEV` (右臂可达半径)
- `hit_z` 在 `[Z_MIN, Z_MAX]` (球台高度附近)
- `‖v_racket‖` 在 `[2, 6]` m/s

不满足 → reject + resample. 这避免 policy 学到对不可达 cmd 的无意义动作.

⚠️ Workspace 常量集中在一处 (e.g., `PADDLE_WORKSPACE_CONFIG` dict), planner.py 和此 sampler 都引用, 避免分歧.

---

## §4. Cmd 时序 (重写)

每个 episode 包含**多次击球 (rally)**. 每次击球之间有一段 "no-cmd gap" — 球飞向对手 + 对手击球 + 球飞回我方.

### 4.1 完整时序 (按用户最新流程 GAPA)

```
═══════ episode reset (RSI 起始姿态, t_offset=0) ═══════
│
│ 路径分支:
│   - 早期 / 高 mimic_prob: 起始 cmd 来自 mimic clip (Phase M)
│   - 后期 / 低 mimic_prob: 起始 cmd 来自 free sampler (Phase F)
│   (curriculum 控制比例, 详见 §4.4)
│
═══════ Path A: mimic 起步 ═══════
│
│  Phase M1: pre-swing  (cur_frame ∈ [0, impact+5])
│    [is_mimic_phase=True, strike_window_reward_passed=False]
│    cmd 来自 npz, 详见 §2.1
│    t_to_hit 从 (impact/fps) 单调下降到 -0.1
│
├──── strike_window 关闭 (cur_frame = impact + 6) ────
│    strike_window_reward_passed = True            ⚠️ 这一刻 flip
│    t_to_hit                    = -1.0 sentinel    ⚠️ 此时冻结
│    base_target_xy 切换: clip[impact].pelvis → clip[-1].pelvis
│
│  Phase M2: follow-through + return-to-ready  (cur_frame ∈ [impact+6, T_clip])
│    [is_mimic_phase=True, strike_window_reward_passed=True]
│    r_i 仍然 active (跟踪 clip 后半段 ref, 至 clip 末)
│    r_g_pos/vel/ori 关闭, r_g_base 仍 dense (target = clip 末帧 pelvis xy)
│
═══════ mimic clip 结束 ═══════
│
│  ⚠️ 多 clip 串接分支 (§2.4):
│   - 若启用串接 + 还有下一 clip: 直接切换到下一 clip 第 0 帧, 重新进入 Phase M1
│   - 否则进入 free play
│
═══════ Free play (mimic_active=False) ═══════
│
│  Phase F0: gap1 (球飞向对手)
│    duration = truncN(low=0.2, high=1.5, peak_low=0.4, peak_high=0.6)  [user]
│    [is_mimic_phase=False, strike_window_reward_passed=True, cmd 字段全部 hold last value]
│    base_target_xy = 击球完那一刻的 robot base_xy (frozen)             [GAPJ]
│    r_i = 0, r_g sparse = 0, r_g_base dense, r_r 全程
│
│  Phase F0.5: 对手击球 (固定 0.1s)
│    [同上, 没有 cmd]
│
├──── new cmd 到达 (free sampler) ────
│    [is_mimic_phase=False, strike_window_reward_passed=False]
│    cmd 字段由 §3 sampler 生成
│    cmd.t_to_hit = truncN[0.2, 1.5] peak [0.4, 0.6]   ← 独立采样作为初始值 [GAPA]
│
│  Phase F2: pre-strike  (new cmd 到达 → strike_window 关闭)
│    cmd.t_to_hit 随仿真时间每 step 减 dt=0.02s
│    Cmd 噪声每 step 在 world frame 注入 (§8)
│
├──── strike_window 关闭 (类似 Phase M1 → M2 切换) ────
│    strike_window_reward_passed = True
│    t_to_hit                    = -1.0 sentinel
│    base_target_xy 切换为击球完那一刻的 robot base_xy   [GAPJ free 段]
│
│  Phase F3: free follow-through  (strike_window 关闭 → 下一 cmd 到达)
│    [类似 Phase F0/F0.5/F1, 准备下一 rally]
│
═══════ 重复 (gap1 + 0.1 + new cmd + Phase F2 + Phase F3) 直到 ═══════
   - 10s timeout (paper Sec V-B1) — episode-level
   - 摔倒 termination
   - episode 内击球次数无上限 [GAPE]
```

### 4.2 关键决定汇总

| 时序点 | 行为 | 来源 |
|---|---|---|
| RSI | 只随机 robot 物理姿态, t_offset 强制 = 0 | [GAPG] |
| Mimic clip 内击球 | strike_window_reward_passed 在 cur_frame=impact+6 切 True | [GAPI 修订] |
| `t_to_hit` 在击球后 | 在 cur_frame=impact+6 那一刻冻结到 -1.0 sentinel | [GAPI 修订] |
| Mimic 段 base_target | pre-swing/strike: clip[impact].pelvis_xy; follow-through: clip[-1].pelvis_xy | [GAPJ] |
| Mimic clip 末 | mimic_active 切 False; cmd 字段全部 hold | [D4] |
| Mimic 串接分支 | 一定比例 envs 启用, curriculum 减少占比 (但保留) | [GAPL] |
| gap1 (球飞向对手) | truncN[0.2, 1.5] peak [0.4, 0.6] | [user] |
| 对手击球 | 固定 0.1s | [user, paper-derived] |
| New cmd 到达时机 | 对手击球完那一刻立刻到达 (不再有"gap2") | [GAPA 修订] |
| New cmd t_to_hit 初值 | 独立采样 truncN[0.2, 1.5] peak [0.4, 0.6] | [GAPA] |
| Cmd 切换跳变 | accept jump (no smoothing) | [D5] |
| Free 段击球后 base_target | 击球完那一刻的 base_xy (frozen) | [GAPJ free 段] |
| 不提前给 next cmd | 遵循部署不可预测性 | [D4 sub] |
| Episode 击球次数 | 无上限, 直到 10s timeout 或摔倒 | [GAPE] |
| Sampler 选择颗粒度 | per-cmd (但目前只有一种 free sampler) | [GAPF] |

### 4.3 Episode 终止条件 (与 cmd 相关)

- 10 秒 timeout (paper Sec V-B1) — episode-level, 与 cmd 无关
- 摔倒 (pelvis_height < 0.4 等) — termination, 与 cmd 无关
- cmd 本身**不**触发 termination (击空 / 击不到都不算 fail, 由 reward 自然惩罚)

### 4.4 Episode 起点 (curriculum, GAPD)

| Iter range | mimic_start_prob | 说明 |
|---|---|---|
| 0 – 8k    | 1.00 | 100% mimic 起步, warmup 阶段强 ref 引导 |
| 8k – 25k  | 1.00 → 0.7 | 渐增 free play 起步比例 |
| 25k+      | 0.7  | 30% episode 直接从 default pose + free sampler 起步 |

**Free play 起步细节**:
- robot 物理状态 = default standing pose (joint_pos=default, base_pos=default)
- 立刻调用 §3 free sampler 生成首 cmd
- 跳过 mimic 段 (Phase M1/M2), 直接进入 Phase F2
- t_to_hit 起始 = sampler 给的 truncN 采样

---

## §5. Strike 双标志详细语义

### 5.1 `strike_window_reward_passed` (时间 flag) [GAPI rename]

**定义**: r_g sparse 项的时间窗口已关闭. 用于 gate r_g_pos / r_g_vel / r_g_ori.

```python
# 每个环境 step 末尾:
t_to_hit_curr = (impact - cur_frame) / fps
if t_to_hit_prev >= -0.1 and t_to_hit_curr < -0.1:
    strike_window_reward_passed = True
    t_to_hit                    = -1.0   # freeze sentinel
```

**单调切换**: True 之后不再回到 False. 直到 next cmd 到达, flag 才被新 cmd 重置 (新 cmd 自带 strike_window_reward_passed=False).

**为什么 ±0.1s (±5 帧 @ 50Hz) 而不是击球瞬间切**:
- paper Sec V-B2 的 sparse window = "short window around hitting time", 我们取 ±5 帧 (cur_frame ∈ [impact-5, impact+5], 共 11 帧)
- 在 impact 瞬间立刻关闭 r_g 等于"只有一帧机会拿满分", 信号过 sparse
- 让 r_g 在整个 11 帧窗口都激活, robot 学到的是"在窗口内任意时刻命中"
- 与 paper 描述对齐, 比击球瞬间宽容

### 5.2 `hit_actually_landed` (几何 flag, 新, GAPI)

**定义**: 在 strike_window 内, blade 几何上是否真正命中目标击球点.

⚠️ 关键: **不能用纯 L1 距离**. 距离 5cm 但在拍面前方 5cm (而不是拍面平面内) 算误判命中. 必须分两个方向阈值:

```python
# 在 strike_window 内任意 step 计算:
n_blade   = quat_rotate(blade_quat, [0, 1, 0])     # blade 法向 (Y 轴)
delta     = ball_pos_w - blade_pos_w               # 拍面到球
d_normal  = abs(delta · n_blade)                   # 平面外法向距离
d_inplane = sqrt(‖delta‖² - d_normal²)             # 平面内距离

hit_actually_landed_step = (d_inplane < 0.05) and (d_normal < 0.015)
# 0.05 m:  拍面有效区半径 (拍 r=0.075, 容差 5cm)
# 0.015 m: 法向必须紧贴 (球径 4cm 标准球, 拍厚 1cm, 半厚+余量 = 0.015) [user-decided]
```

`hit_actually_landed` 是**整个 strike_window 内的 OR 累积**:

```python
if strike_window_active and hit_actually_landed_step:
    cmd.hit_actually_landed = True              # 一旦 True 不再 reset 直到 next cmd
```

**用途**:
- **不 gate reward** (r_g 已经用 strike_window_reward_passed gate)
- 暴露给 obs (actor + critic): policy 知道是否真的击中, 利于 value head 区分"击中但 reward 低 (动作差)" vs "完全没碰到"
- diagnostic: 训练曲线监控 hit-rate 趋势, 评估 policy 击球准确性
- curriculum 信号: hit_rate 高时收紧 σ_pos (REWARD_DESIGN §5)

⚠️ "球" 的位置在训练时是 cmd.p_racket (即 sampler / mimic 给的目标点); 部署时是真实球 perception. 接口一致.

### 5.3 mimic_active 与 strike_window_reward_passed 的正交关系

| mimic_active | strike_window_reward_passed | 阶段 | r_i | r_g sparse | r_g_base |
|:---:|:---:|---|:---:|:---:|:---:|
| T | F | mimic pre-swing + strike_window 内 (Phase M1) | ✓ | ✓ | ✓ (target=clip[impact].pelvis) |
| T | T | mimic follow-through (Phase M2) | ✓ | ✗ | ✓ (target=clip[-1].pelvis) |
| F | T | no-cmd gap (Phase F0/F0.5/F1) | ✗ | ✗ | ✓ (target=击球完 base_xy frozen) |
| F | F | free play pre-strike + strike_window 内 (Phase F2) | ✗ | ✓ | ✓ (target=sampler 给) |
| F | T | free play follow-through (Phase F3) | ✗ | ✗ | ✓ (target=击球完 base_xy frozen) |

### 5.4 Adapter 占位策略

新 cmd 还没到达时 (Phase F0/F0.5/F1), observation adapter 给 actor 什么?

```python
# adapter pseudo:
if cmd is None or cmd.is_stale:
    obs_cmd = last_known_cmd          # hold 最后一个 cmd 全部字段
    obs_cmd.t_to_hit                    = -1.0
    obs_cmd.strike_window_reward_passed = True
else:
    obs_cmd = cmd
```

**为什么用占位 + flag, 不用 zero / sentinel cmd**:
- zero 占位 obs 突变 → policy 看到 (大值) → (0, 0, 0) jump, 训练抖动
- 占位 + flag: obs 平滑, policy 通过 flag 显式知道"cmd 已过期"
- critic 同 obs, value head 学到 "True 时回退 / 等下一个 cmd"

---

## §6. Cmd → Reward Gating 总览

cmd 字段控制 reward sub-term 的激活. 详细公式见 [REWARD_DESIGN.md](REWARD_DESIGN.md), 这里只列 cmd-side 的 gate 关系:

| cmd 字段 | gate 哪些 reward | 作用 |
|---|---|---|
| `is_mimic_phase` (== mimic_active) | r_i 全部 sub-terms | mimic clip 内 r_i 才有意义 |
| `strike_window_reward_passed` | r_g_pos, r_g_vel, r_g_ori | sparse, 时间窗内才计 |
| (none, 全程激活) | r_g_base | dense, 防止 free play 段 reward 太 sparse |
| `t_to_hit` | strike_window 判定 (内部用; flag 由 t_to_hit 推导) | 不直接 gate, 但驱动 flag 切换 |
| `hit_actually_landed` | **不 gate reward** | 仅 obs / diagnostic |

---

## §7. 部署对齐 (HitterPlanner → cmd) [D1, D5]

### 7.1 World frame 存储 + base-relative adapter

cmd 内部全部用 **world frame** 存储 (与 planner 输出一致). actor obs 之前经过 adapter 转 base-relative:

```python
# obs adapter (训练 + 部署都用同一份)
def world_to_base_rel(p_world, base_pos, base_quat):
    return quat_rotate_inverse(base_quat, p_world - base_pos)

obs.p_racket_rel              = world_to_base_rel(cmd.p_racket, base_pos, base_quat)
obs.v_racket_rel              = quat_rotate_inverse(base_quat, cmd.v_racket)
obs.base_target_dxy           = (cmd.base_target_xy - base_pos[..., :2])   # planar relative
obs.t_to_hit                  = cmd.t_to_hit                                # scalar, 不变
obs.is_mimic_phase            = cmd.is_mimic_phase.float()
obs.strike_window_reward_passed = cmd.strike_window_reward_passed.float()
obs.hit_actually_landed       = cmd.hit_actually_landed.float()
obs.swing_type_onehot         = one_hot(cmd.swing_type, 2)
```

设计原因 [D1]:
- world frame 存储 → reward 直接用 world-frame body state 做差, 不需要 frame transform
- adapter 在 obs 边界做转换, 训练 + 部署接口统一 (planner 和 sim 都给 world frame)

### 7.2 部署时 cmd 切换跳变 [D5 = A]

接受 cmd 切换的 obs jump. 不做平滑滤波. 理由:
- planner 内部已经有 `_hold_or_none` + `swing_lock_frames=50` 的稳定机制
- 真实 rally 中 cmd 之间本来就是离散事件 (球击出 → 等飞行 → 新 cmd)
- 任何平滑都会引入"假"中间 cmd, 训练 vs 部署不一致

---

## §8. Cmd 噪声 (curriculum) [D3, GAPK]

### 8.1 注入方式 [GAPK 修订: per-step]

**每 step 都重新加噪 (持续性注入)**, 在 world-frame cmd 上加, 之后再走 obs adapter [user-decided]:
- reward 计算用每 step 的 noisy cmd (训练学到的"目标"每帧抖一点)
- obs 也用同一份 noisy cmd (与 reward 一致, 没有 train/eval 不对称)
- 物理意义: 模拟 vision/perception 估计噪声 — 每帧测量都不同, robot 看到的就是带帧间噪声的球状态预测
- ⚠️ **量级一定要小**, 否则会直接破坏击球观测 (用户原话). 见 §8.3

```python
# 每 step 在 cmd 上重新加噪 (mimic 段 / free 段都加, 在 obs adapter 之前):
def apply_cmd_noise(cmd, σ):
    cmd.p_racket       += gauss(0, σ_p, shape=(E, 3))
    cmd.v_racket       += gauss(0, σ_v, shape=(E, 3))
    cmd.base_target_xy += gauss(0, σ_base, shape=(E, 2))
    cmd.t_to_hit       += gauss(0, σ_t, shape=(E,))
    # swing_type 不加噪 (离散字段)
    # strike_window_reward_passed / hit_actually_landed 不加噪 (flag)
    return cmd
```

⚠️ 噪声只在 obs / reward 计算前加, 不修改 cmd manager 的 underlying state — underlying cmd (sampler 给的 ground truth) 仅在切换 cmd 时变化. 即:
- `cmd_buffer = sampler_output` (整个 cmd 期间不变)
- `cmd_observed = cmd_buffer + per_step_noise()` (每 step 重新采样)
- reward 用 `cmd_observed`, obs 用 `cmd_observed`

### 8.2 Curriculum schedule (草案)

| Iter range | σ_p | σ_v | σ_base | σ_t | 备注 |
|---|---|---|---|---|---|
| 0 – 8k    | 0       | 0       | 0      | 0      | warmup, 干净 cmd 学习核心动作 |
| 8k – 25k  | 0 → 0.02 | 0 → 0.2 | 0 → 0.05 | 0 → 0.02 | 渐进引入 |
| 25k+      | 0.02    | 0.2     | 0.05   | 0.02   | 部署级噪声 (planner 测量误差量级) |

详细 schedule 见 [CURRICULUM_DESIGN.md](待写).

### 8.3 噪声量级原则 [GAPK]

- **不要太大** (用户原话): 太大的噪声会直接破坏击球观测, policy 学不到精准动作
- 上表数值已按此原则 (0.02 m / 0.2 m/s / 0.05 m / 0.02 s 都是较小的级别)
- **课程开启**, 不一开始就加 (干净 cmd 帮助 policy 先收敛)

---

## §9. D7 测量结果

(已在 §2.3 列出, 此处不重复)

**关键决定**: σ_vel 自适应 `σ_vel = max(0.3, 0.2·‖v̂_racket‖)`. 详见 [REWARD_DESIGN.md](REWARD_DESIGN.md) §3.

---

## §10. base_target_xy 在不同阶段的取值汇总 [GAPJ]

整理一张表, 实现时直接对照:

| 阶段 | base_target_xy 取值 | 备注 |
|---|---|---|
| Mimic Phase M1 (pre-swing + strike_window) | `clip[impact].pelvis_xy` | 引导 robot 走到 impact 时的站位 |
| Mimic Phase M2 (follow-through) | `clip[-1].pelvis_xy` | 引导 robot 回到 ready stance |
| Free Phase F0/F0.5/F1 (no-cmd gap) | 击球完那一刻的 robot base_xy (frozen) | 保持站位, 等下一 rally |
| Free Phase F2 (pre-strike) | sampler 输出 (`hit_x - 0.4, hit_y + 0.25 if forehand else hit_y`) | 几何 |
| Free Phase F3 (follow-through) | 击球完那一刻的 robot base_xy (frozen) | 同 F0/F0.5/F1 |

**切换时机**: 都跟 `strike_window_reward_passed` flip 同步 (cur_frame=impact+6 那一刻, 即 t_to_hit 从 -0.1 跨到 -0.12 的 step). [user-decided]

**设计原因**:
- mimic 段用 clip 末帧 pelvis: paper / DeepMimic 风格 ref tracking, ref 自带 ready stance
- free 段用击球完瞬间 base_xy: 没有 ref, 只能 freeze 当前位置, 防止 robot 漂走

---

## §11. Cross-doc notes (其他 doc 待同步更新)

### 11.1 REWARD_DESIGN.md 待更新

1. `strike_completed` → `strike_window_reward_passed` 全文 rename
2. r_g sub-term 的 gate 表达式更新: `strike_window ∧ ¬strike_completed` → `¬strike_window_reward_passed`
3. **新增**: `r_alive` (Gap M) [user-decided]
   - 当前 r_r 全负 → policy 可能倾向"早结束 episode"作为 reward 最大化策略
   - **决定**: `alive_reward = +0.1` (每 step), 而非 termination_penalty
   - 理由: 信号 dense, IsaacLab humanoid locomotion 标准做法, 实现简单
   - 更新 REWARD_DESIGN.md §4 r_r 表格

### 11.2 EVENT_DESIGN.md (待写) 应包含

- RSI 详细机制: pose source sampling (任意 clip 任意帧) + ref source (当前 clip frame 0)
- Multi-clip chaining 实现细节: clip queue 管理, transition handling
- Mimic 起步 vs Free 起步的 episode-level 决策 (curriculum-controlled probability)
- Domain randomization (paper Sec V-B3)

### 11.3 CURRICULUM_DESIGN.md (待写) 应包含

- σ_pos 紧度调度 (REWARD_DESIGN §5)
- Cmd 噪声调度 (本 doc §8.2)
- Mimic 起步 prob 调度 (本 doc §4.4)
- Multi-clip chain prob 调度 (本 doc §2.4)

---

## §12. 已锁定的设计决定 (汇总表)

| 项 | 决定 | 来源 |
|---|---|---|
| `n̂_target` 不存为独立字段 | 从 `v̂_racket` 推导 | paper Sec IV-C |
| Cmd 字段命名 | `v_racket` (paper) 而非 `v_paddle` | [D2] |
| 时间字段 | `t_to_hit` (相对) 而非 `t_strike` (绝对) | [D2 + planner 对齐] |
| `paddle_normal` 字段 | 不暴露 | [D2] |
| `is_mimic_phase` 字段 | 暴露, 默认 False | [D1] |
| `base_target_xy` 显式提供 | 不从其他字段推导 | [user] |
| Mimic 段 cmd | 全部从 npz 合成 | [D7-Q2] |
| Free 段 sampler | 单一 sampler (无 80/20), 用 planner 几何但不调 update() | [GAPB / GAPC user] |
| 工作空间约束 | sampler 必满足 planner workspace | [D6] |
| RSI 设计 | 只采姿态, t_offset 强制 = 0 | [GAPG user] |
| Mimic clip 串接 | 一定比例 envs, curriculum 减少 | [GAPL user] |
| Episode 起点 | curriculum: mimic 起 → free 起占比上升 | [GAPD user] |
| Episode 击球次数 | 无上限 | [GAPE user] |
| Sampler 颗粒度 | per-cmd 独立掷骰子 (但目前只有一种) | [GAPF user] |
| Cmd timing 模型 | gap1 + 0.1 + new cmd 立刻到达 (无 gap2) | [GAPA 修订] |
| gap1 采样 | truncN[0.2, 1.5] peak [0.4, 0.6] | [user] |
| 对手击球 | 固定 0.1s | [user, paper-derived] |
| New cmd t_to_hit 初值 | 独立 truncN[0.2, 1.5] peak [0.4, 0.6] | [GAPA user] |
| Strike flag 命名 | `strike_window_reward_passed` (rename) | [GAPI] |
| Strike flag flip 时机 | strike_window 关闭那一刻 (cur_frame=impact+6, ±5 帧 @50Hz) | [GAPI 修订] |
| `hit_actually_landed` flag | 几何检测 (in-plane + normal 双阈值), 不 gate reward, 仅 obs/diagnostic | [GAPI user] |
| 几何检测阈值 | d_inplane<0.05, d_normal<0.015 (拍厚 1cm + 标准球) | [user-decided] |
| Mimic base_target | M1: clip[impact].pelvis; M2: clip[-1].pelvis | [GAPJ user] |
| Free base_target | F0-F1/F3: 击球完 base_xy frozen; F2: sampler 给 | [GAPJ user] |
| Cmd 切换跳变 | **硬切** (无平滑) — 兼作 mimic 稳定性训练 | [D5 + user] |
| World frame 存储 + adapter | reward 用 world, obs 转 base-rel | [D1] |
| Cmd 噪声注入位置 | world frame, **每 step 重新加** (持续性, 不是一次性) | [GAPK user 修订] |
| Cmd 噪声量级 | 小 (σ_p=0.02 等), curriculum 开启, 量级**绝不能太大** | [GAPK user, D3] |
| σ_vel 自适应 | `max(0.3, 0.2·‖v̂‖)` 应对 backward 慢拍 | [D7] |
| Free 段 v_racket 方向采样 | yaw=robot反向±40°, pitch=10°-60° | [user-decided] |
| 击球 strike_window 宽度 | ±5 帧 (cur_frame ∈ [impact-5, impact+5], 11 帧) [回到 paper Sec V-B2] | [user-decided] |
| Termination 信号 | `alive_reward = +0.1` per step (非 termination_penalty) | [user-decided] |

---

## §13. 实现 TODO (代码层面 checklist)

### 13.1 csv_to_npz_pingpong.py ✅ 已完成
- [x] `body_names: list[str]` 字段
- [x] `swing_type: int8` 字段
- [x] `--task_name` CLI arg

### 13.2 mdp/commands.py (待写)
- [ ] `HitCommand` dataclass + manager (按 §1)
- [ ] Mimic 段 sampler (从 npz, 按 §2.1)
- [ ] RSI 解耦 (姿态 vs ref 时间, 按 §2.2)
- [ ] Multi-clip chaining (按 §2.4, curriculum prob)
- [ ] Free 段 unified sampler (按 §3, 用 planner 几何)
- [ ] Workspace 约束 reject + resample
- [ ] Gap timing (gap1 + 0.1, new cmd 到达逻辑, 按 §4)
- [ ] Episode 起点 (mimic 起 vs free 起, curriculum)
- [ ] Strike flag flip 逻辑 (按 §5.1, cur_frame=impact+6)
- [ ] hit_actually_landed 几何检测 (按 §5.2)
- [ ] base_target_xy 分段切换 (按 §10)
- [ ] World-frame cmd 噪声注入 (按 §8.1, 在 cmd 生成时加)

### 13.3 mdp/observations.py (待写)
- [ ] World → base-relative adapter (按 §7.1)
- [ ] cmd 字段写到 actor obs
- [ ] hit_actually_landed 写到 obs

### 13.4 mdp/curriculums.py (待写)
- [ ] σ_pos 调度 (REWARD_DESIGN §5)
- [ ] Cmd 噪声 σ 调度 (§8.2)
- [ ] Mimic 起步 prob 调度 (§4.4)
- [ ] Multi-clip chain prob 调度 (§2.4)

### 13.5 工程层面常量集中
- [ ] `PADDLE_WORKSPACE_CONFIG` dict (X_HIT, Y_DEV, Z_MIN/MAX, V range, gap range)
- [ ] planner.py + commands.py 都引用这个 config
- [ ] 避免 sampler 和 planner 走偏

### 13.6 Sanity tests
- [ ] gap 期 obs 平滑 (t_to_hit 不会 NaN, flag 切换正确)
- [ ] backward clip cmd.v_racket 量级 + r_g_vel 有意义 (σ_vel 自适应)
- [ ] cmd 切换跳变在 actor obs 上的 magnitude (是否需要 obs clipping)
- [ ] Multi-clip chaining 串接帧上 robot 物理状态连续
- [ ] strike flag flip 时机 (cur_frame=impact+6 那一帧, off-by-one verify)
- [ ] hit_actually_landed 几何检测: 拍前 5cm 不算命中 (d_normal=0.05 > 0.015), 拍面内 5cm 算命中 (d_inplane=0.05 边界)
- [ ] Free play 起步 episode (no mimic) reward landscape 没有断崖

---

## §End. 设计完整性 review checkpoint

读完这 13 节, 实现前需要再确认的点:
- [x] §3.1 v_racket 方向采样 — yaw=robot反向±40°, pitch=10°-60° [user-decided]
- [x] §5.2 几何阈值 d_inplane=0.05 / d_normal=0.015 — 拍厚 1cm + 标准球 (球径 4cm), d_normal=0.015 = 半厚+少量余量 [user-decided]
- [x] §10 base_target 切换 — 跟 strike_window_reward_passed flip 严格同步 (cur_frame=impact+6, ±5 帧之后) [user-decided]
- [x] §11.1 alive_reward = +0.1 per step (不用 termination_penalty) [user-decided]
