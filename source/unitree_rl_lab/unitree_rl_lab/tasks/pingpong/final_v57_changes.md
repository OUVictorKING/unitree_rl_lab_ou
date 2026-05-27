# final.md → v5.7 修订差分清单

> 目的: 记录从当前 final.md (v5.4 / 部分 v5.5) 整合到 v5.7 完整版本所需的全部修改点.
> 每条修改包含: **位置**, **原内容 (摘要)**, **新内容 (v5.7)**, **修改原因**, **来源 (plan §N?)**.
> 整合时按本清单逐条对照执行, 不应再出现歧义.
>
> Plan 来源: `/home/woan/.claude/plans/frolicking-weaving-lighthouse.md` §N1–§N13.
> 目标文件: `source/unitree_rl_lab/unitree_rl_lab/tasks/pingpong/final.md` (整文件覆盖).

---

## 0. 总览 — v5.7 锁定的 12 条用户决议 (相对 v5.5)

整合时, header 顶部"用户决定累计"列表必须把这 12 条加进去 (与原有 24 条 v5 决议合并, 去重为 ~30 条).

| 用户编号 | 决议 (v5.7) | 影响章节 |
|:---:|---|---|
| ① | **不 import planner.solve_paddle_target**, 把同一套公式 (paper Eq.5+Eq.6) 直接写在 `mdp/commands.py` 内 | §3.2 (cmd 生成) |
| ② | v_ball_in / target_land / flight_time / paddle_cor 4 项进 cmd 字段, **全部不进 obs** (Actor 也不进, Critic 也不进) | §2 / §3.1 |
| ③ | obs Q5 决议: 选项 A (paper Table I 严格); Critic 维度保持 213, 不加 7 维 privileged | §2 |
| ④ | `v_in_pitch ∈ [-deg2rad(75), +deg2rad(75)]` (允许下落球, 替代 v5.5 的 [10°, 60°]) | §3.2 Step 2 |
| ⑤ | `target_land` **不 sample**, 用常量 `(2.45, 0, 0.78)` (= planner 默认值平移到训练 world frame) | §3.2 Step 3 |
| ⑥ | `flight_time` 用 `uniform[0.30, 0.65]` 秒, 替代原 truncN | §3.2 Step 4 |
| ⑦ | `t_pre_initial` 改为 `truncN(low=0.20, high=0.90, peak_low=0.30, peak_high=0.65)` 秒 | §3.2 Step 7 |
| ⑧ | `t_post_swing` 改为 `truncN(low=0.40, high=1.10, peak_low=0.40, peak_high=0.75)` 秒 | §3.2 Step 7 |
| ⑨ | `expert_offset` 重新实测确认: **base frame 才稳定** (forehand=(+0.496, +0.208), backhand=(+0.428, +0.106)); world frame 不可用 (clip yaw 差 65°) | §11.4 |
| ⑩ | `r_g_ori` 公式中 `n̂_target` 直接用 `cmd.n_target_world` (paper Eq.5 直给), **不再** 用 `v̂_racket / ‖v̂_racket‖` | §1 #6 |
| ⑪ | swing_type forehand:backhand 偏斜**只监控不修 sample 逻辑** | §12 monitor #1 |
| ⑫ | swing_type 改为 BASE frame 阈值 `y_mid_base = +0.157 m`; **每次 swing 重采时算一次, pre-strike 期间允许变更 1 次后锁定** | §3.2 Step 6 + §3.3 |

**Q-N1 / Q-N2 / Q-N3 已锁** (无 OPEN 项):
- Q-N1: target_land 用 (2.45, 0, 0.78) 世界坐标
- Q-N2: 每次 swing 重采都重算 swing_type, 加 1-change lock
- Q-N3: paddle_cor = 0.85 写死 (不进 DR)

---

## 1. Header / Context 块 — 整体替换

### 位置
final.md 第 1–36 行 (header + "用户决定累计" 列表).

### 原内容
- 标题: `# Paper-Aligned EnvCfg 设计表 v5 — HITTER (arXiv:2508.21043v2)`
- "用户决定累计 (v5 锁定, #14–#23)" 24 条

### 新内容
- 标题改为: `# Paper-Aligned EnvCfg v5.7 — HITTER (arXiv:2508.21043v2)`
- "v5.7 增量" 段落引用 §N1 12 条决议
- 合并 v5 → v5.7 决议为统一列表 (~30 条, 去重)
- **删除矛盾的 v5 旧决定**:
  - 原 v5 #12 "Cmd `v̂_racket`: `v_mag~U[2,6] / v_yaw=base_yaw+π+U[-40°,+40°] / v_pitch~U[10°,60°]`" — 整行删除, 因 v5.7 用户 ② 改成 planner 物理推导
  - 原 v5 #16 "Events `swing_type` 独立采样 — 不再从 hit_xy 几何反推" — 整行删除, 因 v5.7 用户 ⑫ 改成 base-frame 阈值 + 1-change lock

### 原因
v5.7 颠覆了 v5 关于 v̂_racket 来源 + swing_type 决定方式的两个核心决定. 不删旧条目会造成读者矛盾.

### 来源
plan §N1, §N12.

---

## 2. §1 Reward 总览表 — r_g_ori 行修订

### 位置
final.md 约第 50 行, §1 表的 #6 r_g_ori 行 "计算 / 来源" 列.

### 原内容
```
| 6 | `r_g_ori` | r_g | `exp(-(1 − n_blade · n̂)²/σ²)` | `n_blade` = blade local +Y; `n̂ = v̂_racket / ‖v̂_racket‖` | **0.5** ⚠️ | σ=0.2 (cos dist) | ✓ §V-B2 + §IV-C | sparse 同上 |
```

### 新内容
```
| 6 | `r_g_ori` | r_g | `exp(-(1 − n_blade · n̂_target)²/σ²)` | `n_blade` = blade local +Y; **`n̂_target = cmd.n_target_world` (= paper Eq.5 paddle_normal, 不再是 v̂/‖v̂‖)** | **0.5** ⚠️ | σ=0.2 (cos dist) | ✓ §V-B2 + §IV-C | sparse 同上 |
```

### 原因
v5.7 用户 ⑩: paper §IV-C Eq.5 把 paddle_normal 物理定义为 `Δv / ‖Δv‖`, 不是 `v̂/‖v̂‖`. 在 frictionless 模型下两者方向数值等价 (v_paddle ∥ n̂), 但物理来源不同 — 直接用 cmd 中已计算的 `n_target_world` 更精确, 避免在 reward 端重复 normalize.

### 来源
plan §N5.

---

## 3. §2 Observation 总览表 — obs #6 v̂_racket 来源注

### 位置
final.md 约第 110–115 行, §2 表的 #6 v̂_racket 行 "计算 / 来源" 列.

### 原内容
```
| 6 | `v̂_racket` | 球拍目标速度 (world frame) | 3 | cmd 字段 | ✓ | ✓ | ✓ |
```

### 新内容
```markdown
| 6 | `v̂_racket` | 球拍目标速度 (world frame) | 3 | **cmd.v_racket_hat_world (= planner 物理推导, paper Eq.5+Eq.6, 见 §3.2 Step 5)** | ✓ | ✓ | ✓ |
```

### 原因
v5.7 用户 ① + ②: v̂_racket 不再由 cmd 直接采样, 而是由 cmd 中的 (p_hit, v_ball_in, target_land, flight_time, paddle_cor) 五项**通过 paper Eq.5+Eq.6 物理推导**. 维度 / frame 不变, 但来源描述需要更新, 否则读者会以为还在 sample.

### 重要
obs 维度保持: **Actor=86, Critic=213**. v_ball_in / target_land / flight_time / paddle_cor 四项**绝对不进 obs** (用户 ② + ③ 双重确认).

### 来源
plan §N4.

---

## 4. §3.1 Cmd 字段表 — 整体替换 (9 项 → 14 项)

### 位置
final.md 约第 130–155 行 (§3.1 节整张表).

### 原内容 (v5.4 / v5.5 残留, 9 项)
| swing_type, p̂_racket, v̂_racket, p̂_base,xy, t_to_hit, t_pre_initial, t_post_swing, cur_step, n̂_target |

### 新内容 (v5.7, 14 项)

### 3.1 Cmd 字段表 (v5.7, 14 项)

| #    | 字段                              |  维度   | frame | sample 时机 / 来源                           | 取值范围 / 公式                                           |       进 Actor obs?        | 进 Critic obs? |            paper?            |
| ---- | --------------------------------- | :-----: | ----- | -------------------------------------------- | --------------------------------------------------------- | :------------------------: | :------------: | :--------------------------: |
| 1    | `swing_type`                      | 1 (cat) | —     | swing 重采 + pre-strike 1-change lock (§3.3) | 见 §3.2 Step 6 base-frame 阈值                            | ✗ (隐含 ref clip selector) |       ✗        |      △ deploy heuristic      |
| 2    | `swing_change_remaining`          | 1 (int) | —     | resample 时设 1, 首次变更后置 0              | {0, 1}                                                    |        ✗ (内部状态)        |       ✗        | [我提案 ⚠️ DIVERGENCE P 修订] |
| 3    | `p_hit_world`                     |    3    | world | swing 重采时 sample                          | x=0.4 固定; y∈U(-1,1); z∈U(0.08,0.6)                      |   ✓ obs #5 (转 base-rel)   |       ✓        |        ✓ §V-B (x=0.4)        |
| 4    | `v_ball_in_world`                 |    3    | world | swing 重采时 sample                          | mag∈U(2,6); yaw=π+U(-40°,+40°); **pitch∈U(-75°,+75°)**    |             ✗              |       ✗        |       ⚠️ DIVERGENCE R-1       |
| 5    | `target_land_world`               |    3    | world | 常量 (Q-N1 锁定)                             | **(2.45, 0, 0.78)** = 对方桌中心 + 桌面 + 球半径 0.02     |             ✗              |       ✗        |       ⚠️ DIVERGENCE R-2       |
| 6    | `flight_time`                     |    1    | —     | swing 重采时 sample                          | **uniform[0.30, 0.65]** 秒                                |             ✗              |       ✗        |       ⚠️ DIVERGENCE R-3       |
| 7    | `paddle_cor`                      |    1    | —     | 常量 (Q-N3 锁定)                             | **0.85** (= paper Eq.6 e)                                 |             ✗              |       ✗        |           ✓ §IV-C            |
| 8    | `v_racket_hat_world` (= v̂_racket) |    3    | world | **§3.2 Step 5 推导** (paper Eq.6)            | `v_pad_n · n̂`                                             |          ✓ obs #6          |       ✓        |        ✓ §V-B + §IV-C        |
| 9    | `n_target_world` (= n̂_target)     |    3    | world | **§3.2 Step 5 推导** (paper Eq.5)            | `(v_out − v_in) / ‖·‖`                                    |   ✗ (经 r_g_ori 间接用)    |       ✗        |           ✓ §IV-C            |
| 10   | `v_ball_out_world`                |    3    | world | §3.2 Step 5 副产品 (sanity 监控)             | `(target_land − p_hit)/T + (0,0,0.5gT)`                   |             ✗              |       ✗        |            (内部)            |
| 11   | `p_base_xy_world`                 |    2    | world | swing 重采时计算                             | `hit_xy_world − R(yaw_robot)·expert_offset_b[swing_type]` | ✓ obs #4 (转 base-rel err) |       ✓        |            ✓ §V-B            |
| 12   | `t_to_hit`                        |    1    | —     | resample 时 = t_pre_initial; 每 step −= dt   | 范围 [-t_post_swing, t_pre_initial]                       |          ✓ obs #7          |       ✓        |            ✓ §V-B            |
| 13   | `t_pre_initial`                   |    1    | —     | resample 时 sample                           | **truncN(0.20, 0.90, 0.30, 0.65)** 秒                     |             ✗              |       ✗        |   [我提案 ⚠️ DIVERGENCE Q]    |
| 14   | `t_post_swing`                    |    1    | —     | resample 时独立 sample                       | **truncN(0.40, 1.10, 0.40, 0.75)** 秒                     |             ✗              |       ✗        |   [我提案 ⚠️ DIVERGENCE Q]    |
| 15   | `cur_step`                        |    1    | —     | resample 时复位 0; 每 step +1                | int [0, (t_pre+t_post)/dt]                                |          ✗ (内部)          |       ✗        |            (内部)            |

(注: 表项 15 算 14 项 + 1 计数器, 也可合并到 14. 行数无关紧要, 内容必齐.)

### 原因
- v5.7 用户 ① + ②: cmd 字段从原 9 项扩为 14+1 项, 把"上游物理输入 (v_ball_in/target_land/flight_time/paddle_cor)" 与 "下游 planner 推导输出 (v_racket_hat/n_target/v_ball_out)" 显式分开.
- 用户 ③: 4 项上游全部不进 obs, 1 项 sanity 监控也不进 obs.
- 用户 ④/⑤/⑥/⑦/⑧: 取值范围与原表完全不同, 必须更新.
- 用户 ⑫: 加 `swing_change_remaining` 字段.

### 来源
plan §N3, §N12.

---

## 5. §3.2 Cmd 生成代码 — 整体替换 (Path B 7 步)

### 位置
final.md 约第 160–230 行, §3.2 节整段代码.

### 原内容
- "v_mag~U[2,6] / v_yaw=π+U[-40°,40°] / v_pitch~U[10°,60°]" 直采 v̂_racket 的 Python 代码块
- swing_type 用 `uniform_sample({forehand, backhand})` 独立采样

### 新内容 (v5.7 §N2 7 步, 完整代码块)

````markdown
### 3.2 Cmd 生成代码 (v5.7 Path B 7 步, 物理推导 inline 不 import planner)

**Planner (上层 robojudo, 一次性预处理 — 同时记录 expert clip 的 pre/post 时长 + base-frame offset, 见 §11.1.1 / §11.4)**:
```python
expert_offset_base: Dict[str, np.ndarray] = {}      # base-frame xy (v5.7 §N6 重测确认)
expert_pre_duration: Dict[str, float] = {}
expert_post_duration: Dict[str, float] = {}

for swing_type, npz_path in [("forehand",  ".../forward_001.npz"),
                              ("backhand", ".../backward_004.npz")]:
    d = np.load(npz_path)
    imp = int(d["impact_frame"][0])
    fps = int(d["fps"][0])                          # = 50

    # base-frame xy (yaw 旋转 world → base; v5.7 §N6 实测两 clip yaw ≠ 0, world frame 不稳定)
    p_pelv_w  = d["body_pos_w"][imp, PELVIS_IDX]    # PELVIS_IDX=0
    p_blade_w = d["body_pos_w"][imp, BLADE_IDX]     # BLADE_IDX=24
    q_pelv_w  = d["body_quat_w"][imp, PELVIS_IDX]   # wxyz
    yaw       = yaw_from_wxyz(q_pelv_w)
    diff_xy   = p_blade_w[:2] - p_pelv_w[:2]
    c, s      = np.cos(-yaw), np.sin(-yaw)
    expert_offset_base[swing_type] = np.array([
        c*diff_xy[0] - s*diff_xy[1],     # = +0.496 (fwd) / +0.428 (bwd)
        s*diff_xy[0] + c*diff_xy[1],     # = +0.208 (fwd) / +0.106 (bwd)
    ])

    expert_pre_duration[swing_type]  = imp / fps
    expert_post_duration[swing_type] = (d["joint_pos"].shape[0] - 1 - imp) / fps

# y_mid_base = (+0.208 + +0.106) / 2 = +0.157  → §3.2 Step 6 阈值
Y_MID_BASE = 0.157

# expert_pre / expert_post (s):
#   forward_001:  pre=0.74 / post=0.88
#   backward_004: pre=0.40 / post=0.86
```

**训练端 cmd 生成 (resample, = `t_to_hit ≤ -t_post_swing` 那一 step, paper Eq.5+Eq.6 inline)**:

```python
import numpy as np

def resample_cmd(env_id, cmd, robot, dt=0.02):
    # ── Step 1: hit point sample (world frame) ──────────────────────────────
    hit_x_world = 0.4                                          # ✓ paper §V-B
    hit_y_world = np.random.uniform(-1.0, 1.0)
    hit_z_world = np.random.uniform(0.08, 0.6)
    p_hit_world = np.array([hit_x_world, hit_y_world, hit_z_world])

    # ── Step 2: incoming ball velocity sample [v5.7 用户 ④ pitch ±75°] ──────
    v_in_mag   = np.random.uniform(2.0, 6.0)                                 # m/s
    v_in_yaw   = np.pi + np.random.uniform(-np.deg2rad(40), np.deg2rad(40))   # 世界 -x 为 0° 基线
    v_in_pitch = np.random.uniform(-np.deg2rad(75), np.deg2rad(75))           # ⚠️ 允许下落球
    v_ball_in_world = v_in_mag * np.array([
        np.cos(v_in_yaw) * np.cos(v_in_pitch),
        np.sin(v_in_yaw) * np.cos(v_in_pitch),
        np.sin(v_in_pitch),
    ])

    # ── Step 3: target land 常量 [v5.7 Q-N1 锁定: 平移到训练 world frame] ──
    target_land_world = np.array([2.45, 0.0, 0.78])
    # = 对方桌半区中心 (网 x=1.77, 远端 x=3.14, 中点≈2.45) + 桌面 z=0.76 + 球半径 0.02

    # ── Step 4: flight_time sample [v5.7 用户 ⑥ uniform] ────────────────────
    flight_time = np.random.uniform(0.30, 0.65)                # 秒

    # ── Step 5: 物理推导 v̂_racket + n̂_target [v5.7 用户 ① + ⑩ 公式 inline] ─
    paddle_cor = 0.85                                          # paper Eq.6 e (Q-N3 写死)
    g, T = 9.81, flight_time

    # Eq.5 弹道反解 (无空气阻力 + 重力):
    v_out_world = (target_land_world - p_hit_world) / T + np.array([0.0, 0.0, 0.5 * g * T])

    # paddle_normal (frictionless 假设, paper §IV-C):
    delta_v = v_out_world - v_ball_in_world
    norm    = np.linalg.norm(delta_v)
    if norm < 1e-9:
        # 退化情况 — fallback (§N8 / 见歧义消解 Q5):
        n_target_world     = np.array([-1.0, 0.0, 0.0])
        v_racket_hat_world = 2.0 * n_target_world
        # 监控 §12 #6: pingpong/solve_paddle_degenerate_rate
    else:
        n_target_world = delta_v / norm                        # ✓ paper Eq.5
        v_in_n  = float(v_ball_in_world @ n_target_world)
        v_out_n = float(v_out_world @ n_target_world)
        v_pad_n = (v_out_n + paddle_cor * v_in_n) / (1.0 + paddle_cor)   # ✓ paper Eq.6
        v_racket_hat_world = v_pad_n * n_target_world          # 沿 n̂ 方向 (frictionless)

    # ── Step 6: swing_type 几何推导 [v5.7 用户 ⑫: base-frame 阈值] ──────────
    yaw_robot   = quat_to_yaw(robot.data.root_quat_w[env_id])
    base_xy_w   = robot.data.root_pos_w[env_id, :2].cpu().numpy()
    c, s        = np.cos(-yaw_robot), np.sin(-yaw_robot)
    R_inv       = np.array([[c, -s], [s, c]])
    hit_xy_base = R_inv @ (np.array([hit_x_world, hit_y_world]) - base_xy_w)
    hit_y_base  = hit_xy_base[1]
    swing_type  = "forehand" if hit_y_base > Y_MID_BASE else "backhand"
    # ⚠️ 注意: 与 planner.py:970 字面方向相反, 因为 expert clip 录制时 paddle 都在 base 左前方
    #          (forehand y_base=+0.208, backhand y_base=+0.106), 阈值取中点 +0.157.
    swing_change_remaining = 1                                 # pre-strike 期间允许变更 1 次, §3.3

    # ── Step 7: ref clip + base_target + 时间字段 ──────────────────────────
    expert_offset_b     = expert_offset_base[swing_type]       # base-frame, §11.4
    R_b2w               = np.array([[ np.cos(yaw_robot), -np.sin(yaw_robot)],
                                    [ np.sin(yaw_robot),  np.cos(yaw_robot)]])
    expert_offset_world = R_b2w @ expert_offset_b
    p_base_xy_world     = np.array([hit_x_world, hit_y_world]) - expert_offset_world

    # 时间字段 [v5.7 用户 ⑦ + ⑧]:
    t_pre_initial = truncN(low=0.20, high=0.90, peak_low=0.30, peak_high=0.65)   # 秒
    t_post_swing  = truncN(low=0.40, high=1.10, peak_low=0.40, peak_high=0.75)   # 秒

    # ── 写入 cmd ──────────────────────────────────────────────────────────
    cmd.swing_type             = swing_type
    cmd.swing_change_remaining = swing_change_remaining
    cmd.p_hit_world            = p_hit_world
    cmd.v_ball_in_world        = v_ball_in_world
    cmd.target_land_world      = target_land_world
    cmd.flight_time            = flight_time
    cmd.paddle_cor             = paddle_cor
    cmd.v_racket_hat_world     = v_racket_hat_world
    cmd.n_target_world         = n_target_world
    cmd.v_ball_out_world       = v_out_world
    cmd.p_base_xy_world        = p_base_xy_world
    cmd.t_pre_initial          = t_pre_initial
    cmd.t_post_swing           = t_post_swing
    cmd.t_to_hit               = t_pre_initial
    cmd.cur_step               = 0
```
````

### 原因
- v5.7 用户 ①: 不能 import planner 模块, 公式必须 inline 写在 commands.py
- 用户 ②/③: 只 sample 4 项上游 (v_in / target_land / flight_time / paddle_cor), v̂/n̂ 走 Eq.5+Eq.6 推导
- 用户 ④: pitch 范围 [-75°, +75°] (含下落球)
- 用户 ⑤: target_land 不 sample, 用常量 (2.45, 0, 0.78)
- 用户 ⑥: flight_time uniform 不 truncN
- 用户 ⑦/⑧: t_pre / t_post 范围窄于 v5.4
- 用户 ⑨: expert_offset 是 base-frame (实测 §N6 确认)
- 用户 ⑫: swing_type 用 base-frame y_mid=+0.157 阈值

### 来源
plan §N2 (Step 1–7), §N6 (expert_offset 数值), §N8 (退化处理).

---

## 6. §3.3 Cmd 生命周期 — 加 swing_type lock 状态机

### 位置
final.md 约第 232–249 行, §3.3 段落.

### 原内容
- swing_type 在 reset / resample 时独立 uniform 采样, episode 内不变
- 没有 swing_change_remaining 概念

### 新内容 (v5.7 §N7)

````markdown
### 3.3 Cmd 生命周期 (v5.7: 双段时间 + cur_step + swing_type 1-change lock)

```
═══ episode reset ═══
│ 生成首组 cmd (resample_cmd, §3.2 Step 1–7 全 7 步)
│ obs.t_to_hit = cmd.t_pre_initial
│ ref state 起点: get_ref_state(cmd, dt) at cur_step=0 → ref_frame_f=0 (clip 第 0 帧)
│
│ 每 step 先 cmd 维护 (events.interval 阶段):
│   cmd.t_to_hit -= dt  (= 0.02s @ 50Hz)
│   cmd.cur_step += 1
│
│   ─── swing_type 1-change lock 状态机 [v5.7 §N7] ───
│   if cmd.t_to_hit > 0 and cmd.swing_change_remaining > 0:
│       new_swing = compute_swing_type(current_hit_y_base)   # 用当前 base 重算
│       if new_swing != cmd.swing_type:
│           cmd.swing_type             = new_swing            # 同时切 ref_clip
│           cmd.swing_change_remaining = 0                    # latch, 用掉机会
│   if cmd.t_to_hit <= 0:
│       pass    # 击球瞬间 + post 阶段, swing_type 锁死, 不再变
│   ─────────────────────────────────────────────────
│
│   ref 计算 (§11.1.2): get_ref_state(cmd, dt) 当场算
│       pre 段:  cur_step ∈ [0, sim_pre_steps]   → ref_frame_f ∈ [0, impact]
│       post 段: cur_step ∈ (sim_pre_steps, total] → ref_frame_f ∈ (impact, clip_len-1]
│
│   reward gate:
│       abs(cmd.t_to_hit) ≤ 0.06s 时 r_g sparse 激活 (±3 帧 @ 50Hz, t_to_hit=0 = ref impact 帧对齐)
│       cmd.t_to_hit > 0  时 r_g_base dense 激活
│       cmd.t_to_hit ≤ 0  时 r_g_base OFF (§1 #7)
│
├──── cmd.t_to_hit ≤ -t_post_swing (post 段也走完) ────
│    立即 resample_cmd; obs.t_to_hit ← 新 t_pre_initial
│    swing_type 重新计算 (按当前 base + new hit), swing_change_remaining 重置 1
│    cur_step=0 重新对齐
│    r_g_base 切到新 p̂_base,xy
│
═══ 重复, 直到: 10s timeout / 摔倒 termination / robot↔table contact (§6.4 软惩罚不 terminate) ═══
```

**state machine 设计动机** (v5.7 用户 ⑫): pre-strike 期间 base 可能漂移, 给 1 次容错重新决定 swing_type 防止首次决策因 base 微移而锁错; 但只允许 1 次, 防止 thrash. 击球瞬间 onwards 锁死.
````

### 原因
v5.7 用户 ⑫ 决定 swing_type 改用动态 base-frame 阈值 + 1-change lock. 原 v5 静态 swing_type 不再适用.

### 来源
plan §N7.

---

## 7. §11.4 expert_offset / 阈值数值 — 整体增补

### 位置
final.md 约第 950–980 行 (§11.4 节).

### 原内容
- v5.5 A9 已记录 base-frame Δ 数值, 但缺少 swing_type 阈值 +0.157 的明示
- 缺少 world-frame 不可用的解释 (yaw 65° 差异)

### 新内容 (v5.7 §N6 完整数据 + 阈值)

````markdown
### 11.4 expert_offset 实测复算 (v5.7 §N6 锁定) + swing_type 阈值

#### 11.4.1 实测脚本

`/home/woan/miniforge/envs/env_isaaclab_51/bin/python` 读两个 npz:
- `motion_datasets/pingpong/humanoid_data/final/expert/forward/forward_001.npz`
- `motion_datasets/pingpong/humanoid_data/final/expert/backward/backward_004.npz`

#### 11.4.2 实测数据

| clip | swing_type | impact 帧 | clip_len | pelvis yaw | Δ_world (xy, m) | **Δ_base (xy, m)** | ‖v_blade‖@imp (m/s) |
|---|---|:---:|:---:|:---:|---|---|---|
| forward_001 | "forehand" | 37 | 82 | +63.57° | (+0.035, +0.536) | **(+0.496, +0.208)** | 4.418 |
| backward_004 | "backhand" | 20 | 64 | +128.77° | (-0.351, +0.268) | **(+0.428, +0.106)** | 1.995 |

#### 11.4.3 关键发现

1. **WORLD frame Δ 不可用**: 两 clip pelvis yaw 差 65.2°, world frame 下 forehand Δ 在 +y 大, backhand Δ 略偏 -x — **物理上不一致**, 不能作为统一 expert_offset.
2. **BASE frame Δ 稳定**: 两 clip 在 base 系下 paddle 都在前方稍左 (x≈0.46m, y∈[+0.106, +0.208]) — 物理一致, 可以作为 expert_offset.

#### 11.4.4 swing_type 阈值 [v5.7 用户 ⑫]

由 base-frame y 数值得:
- y_fwd_base = +0.208 m (forehand 击球点 y_base)
- y_bwd_base = +0.106 m (backhand 击球点 y_base)
- **Y_MID_BASE = (+0.208 + +0.106) / 2 = +0.157 m** ← swing_type 切换阈值

```python
swing_type = "forehand" if hit_y_base > 0.157 else "backhand"
```

#### 11.4.5 ⚠️ 与 planner.py:970 方向不同的说明

planner.py:970 (deploy 端) 写: `forehand if hp[1] < bp[1] else backhand` (世界 frame, 阈值 0).

我们训练端: `forehand if hit_y_base > +0.157 else backhand` (base frame, 阈值 +0.157).

**方向看似相反**, 但**物理含义一致**: 两个 expert clip 录制时机器人 yaw 各自 63.6° / 128.8° 让 paddle 都飞到 base 左前方; 训练数据由这两个 clip 决定, 故 base-frame y_mid 是正值 +0.157 而不是 0. 部署到右手球员真机时 paddle 应在右侧 (hit_y_base < 0), expert 数据需要重录 — 当前 plan 假设 expert clip 不变.

#### 11.4.6 expert_pre / post 时长 (v5.4 §11.1.1 一致)

| clip | clip_len | impact_frame | pre 时长 (s) | post 时长 (s) |
|---|:---:|:---:|:---:|:---:|
| forward_001 | 82 | 37 | **0.74** | **0.88** |
| backward_004 | 64 | 20 | **0.40** | **0.86** |

#### 11.4.7 实现端要点

- 一次性预处理时存 `expert_offset_base[swing_type]` (base frame), 不存 world.
- 训练每 step 用 `R(yaw_robot) @ expert_offset_base[swing_type]` 旋转回 world 得 base_target_world (§3.2 Step 7).
- swing_type 切换时 expert_offset 也跟着切 (= 切 ref_clip).
````

### 原因
- v5.7 用户 ⑨: 重新实测确认 expert_offset 是 base-frame, 给具体数值
- 用户 ⑫: 阈值 +0.157 必须文档化, 否则实现端不知道该用 0 还是别的值
- 必须解释为何与 planner.py 字面方向相反, 防止后续 review 误判为 bug

### 来源
plan §N6, §N2 Step 6.

---

## 8. §11.5 PingpongCommand dataclass — 加新字段

### 位置
final.md 约第 1037–1075 行, §11.5 (Q4) 的 dataclass 段.

### 原内容 (v5.4)
```python
@dataclass
class PingpongCommand:
    swing_type: str
    p_racket_hat: np.ndarray   # (3,) world
    v_racket_hat: np.ndarray   # (3,) world
    p_base_xy_hat: np.ndarray  # (2,) world
    t_to_hit: float
    t_pre_initial: float
    t_post_swing: float
    cur_step: int
```

### 新内容 (v5.7)

```python
@dataclass
class PingpongCommand:
    # ── swing 决策 (v5.7 §N7 1-change lock) ─────────────────────────
    swing_type: str                           # "forehand" / "backhand"
    swing_change_remaining: int               # = 1 at resample, → 0 on first flip in pre-strike

    # ── 上游物理输入 (v5.7 用户 ② cmd 字段, 不进 obs) ───────────────
    p_hit_world: np.ndarray                   # (3,) world frame, ✓ paper §V-B
    v_ball_in_world: np.ndarray               # (3,) world frame, ⚠️ DIVERGENCE R-1
    target_land_world: np.ndarray             # (3,) world frame = (2.45, 0, 0.78), ⚠️ DIVERGENCE R-2
    flight_time: float                        # 秒, ⚠️ DIVERGENCE R-3
    paddle_cor: float                         # = 0.85, ✓ paper Eq.6 e

    # ── 下游 planner 推导 (v5.7 用户 ① inline 公式输出) ─────────────
    v_racket_hat_world: np.ndarray            # (3,) world, paper Eq.6 输出
    n_target_world: np.ndarray                # (3,) world, paper Eq.5 输出 (r_g_ori 用)
    v_ball_out_world: np.ndarray              # (3,) world, sanity 监控

    # ── base 站位目标 ─────────────────────────────────────────────
    p_base_xy_world: np.ndarray               # (2,) world

    # ── 时间字段 (v5.4 §11.1 双段) ────────────────────────────────
    t_to_hit: float                           # pre 段 t_pre→0, post 段 0→-t_post
    t_pre_initial: float                      # truncN[0.20, 0.90, 0.30, 0.65] 秒 (v5.7 ⑦)
    t_post_swing: float                       # truncN[0.40, 1.10, 0.40, 0.75] 秒 (v5.7 ⑧)
    cur_step: int                             # sim step 计数, resample 时复位 0

    # 注意: 不存 t_strike_absolute! 不存 ref_frame_f! (后者通过 §11.1.2 当场算)
```

### 原因
- v5.7 用户 ②: 字段从 8 个扩到 14 个, 含 4 项上游输入 + 3 项下游输出 + 1 项 swing lock 状态
- 用户 ⑫: 加 swing_change_remaining
- 用户 ⑦/⑧: t_pre / t_post 范围注释更新

### 来源
plan §N3, §N7, §N12.

---

## 9. 新增 §12 Monitor 指标 (整段新增)

### 位置
final.md 末尾, §11 后, §End 前 (约第 1080 行处插入).

### 新内容 (整段新增, v5.7 §N9)

````markdown
## §12. v5.7 训练 Monitor 指标 [user-decided ⑪]

记录这些指标到 rsl_rl `extras` dict, episode end 时 push wandb. 第一轮训练 1k iter 后回看, 必要时调 cmd sample 范围 / 加死区.

| # | 指标 | 频次 | 阈值 (alarm) | 含义 / 用途 |
|---|---|---|---|---|
| 1 | `pingpong/swing_ratio_forehand` | per 1k iter sliding window | < 0.30 or > 0.70 | forehand:backhand 不平衡 → ref clip 训练样本偏斜. **保留监控不修 sample 逻辑** [用户 ⑪] |
| 2 | `pingpong/dead_zone_trigger_rate` | per swing | > 5% | \|hit_y_base − 0.157\| < 0.01 时 swing 决策接近随机 (1mm 决定 ref clip 切换) — 高频则加 ε 死区 |
| 3 | `pingpong/swing_flip_rate_per_episode` | per episode | > 4 | episode 内 swing_type 翻转次数 (一次 swing ≤ 1, 一个 episode 5–8 swing). 表征 base 漂移多寡 |
| 4 | `pingpong/base_y_drift_meanabs` | per episode | abs > 0.5 | base y 累计平均漂移 (与 swing 偏斜直接相关) |
| 5 | `pingpong/v_racket_hat_mag_mean` / `_std` | per 1k iter | mean ∉ [1.5, 5.0], std > 1.5 | planner 输出 v̂ 量级 sanity (与 expert 4.42/1.99 比较) |
| 6 | `pingpong/solve_paddle_degenerate_rate` | per swing | > 0.001 | ‖Δv‖<1e-9 fallback 触发率 (§3.2 Step 5 退化分支) |
| 7 | `pingpong/cos_sim_n_blade_n_target_at_impact` | per swing (impact 帧) | mean < 0.85 | 击球瞬间 paddle 法向对齐 (= r_g_ori 物理意义) |
| 8 | `pingpong/swing_change_remaining_used_rate` | per swing | > 50% | swing_type 锁定机制使用率 (高频用 = base 漂移多, 训练初期可能正常) |

### 12.1 实现要点

```python
# 在 mdp/curriculums.py 或 mdp/observations.py 中:
def write_monitor(env, extras: dict):
    cmd = env.command_manager.get_term("pingpong_cmd")

    # 例: #1 forehand ratio
    forehand_mask = (cmd.swing_type_id == FOREHAND_ID)
    extras["pingpong/swing_ratio_forehand"] = forehand_mask.float().mean().item()

    # 例: #2 dead zone
    dz_mask = (cmd.hit_y_base - 0.157).abs() < 0.01
    extras["pingpong/dead_zone_trigger_rate"] = dz_mask.float().mean().item()

    # 例: #6 degenerate
    extras["pingpong/solve_paddle_degenerate_rate"] = cmd.last_resample_was_degenerate.float().mean().item()

    # ... 其他类似
```

### 12.2 Alarm 触发后处理建议

- #1 偏斜 > ±20%: 加 reward 项强制 base_y 回归 0 (= r_g_base sigma 收紧), 或采样时强制 base_y 重置 (留用户决定)
- #2 dead zone > 5%: 加 ε 死区 0.05m (5cm 内强制保留上次 swing_type)
- #4 base_y 漂移 > 0.5: 同 #1 处理
- #6 degenerate > 0.001: 调 v_in / target_land 采样范围
````

### 原因
v5.7 用户 ⑪ 决议保留 swing 偏斜监控但不修 sample 逻辑. 8 项 metric 是 §M9 → §N9 锁定列表, 实现端必须有, 否则训练第一轮无法判断分布是否异常.

### 来源
plan §N9.

---

## 10. §13 (新建) Paper DIVERGENCE 索引 — 加 R-1 / R-2 / R-3 + 移除 v5.5 v̂ 直采条目

### 位置
final.md §End 之前 (新建 §13 节).

### 现状
原 final.md 有 DIVERGENCE A–J + O–Q 索引 (§7 / §End 散落). v5.7 需要集中并新增 R 系列.

### 新内容 (v5.7 完整索引)

````markdown
## §13. ⚠️ Paper Divergence 完整索引 (v5.7)

| # | 项 | paper | 我们 | 风险 / TODO |
|---|---|---|---|---|
| A | r_i / r_g / r_r 顶层 weights | Eq.7 给, 数值未给 | 0.5 / 1.0 / 1.0 | 训练后可调 |
| B | r_g_base 全程 dense | "before strike" | dense 全程 | 留 ablation TODO |
| C | r_i sub-term 分解 | 仅 ⊆ upper body | DM 6 项 (jp/jv/bp/bq/blv/bav, 删 e/c, 加 bp) | 标准做法 |
| D | ℬ 排除 (blade / wrist_roll / wrist_roll_joint) | paper 未列 | 排除 3 个末端 | 训练第一轮 monitor r_i 各项 mean |
| E | clip 长度 / ratio | 94 帧 imp=43 ratio=0.46 | fwd 82/37/0.45 ✓; bwd 64/20/0.31 ⚠ | bwd ratio 偏离 |
| F | backward v_blade | 未给 | 1.99 m/s (vs forward 4.42) | GVHMR 末端被滤平 |
| G | r_g sub-term weights | 未给 | pos 2.0 / vel 1.0 / ori 0.5 / base 0.3 | 训练后可调 |
| H | σ_vel 自适应 (v5 删) | 未给 | v5+ 改固定 σ=0.5 | — |
| I | r_r 整体 | 未给 | IsaacLab 标准 + alive_reward | 标准做法 |
| J | σ_pos curriculum | 未提 | 0.10 → 0.02 m, 5 阶段 | 第一轮启用 |
| K | action scale | 未给具体数值 | mimic per-joint `0.25·effort/stiffness` | 复用 mimic 配置 |
| L | (空, 历史编号) | — | — | — |
| M | RSI 解耦 (姿态 vs cur_frame) | DM 标准 RSI 同帧 | cur_frame=0 强制从 clip 头 | 与 cmd.t_to_hit 同步 |
| N | 表桌建模 (RigidBody + ContactSensor) | 未明 | 加静态 table | 工程必要 |
| O | ref clip 双段时间比例插值 (v5.4) | 单 clip 不需 | 2-clip + 自由 swing 时长 → 双段 lerp/slerp | 击球帧对齐工程必要 |
| P | swing_type 几何推导 (v5.7 修订) | 未明 | base-frame y_mid=+0.157 + 1-change lock | 与 planner.py:970 物理一致, 字面方向相反 |
| Q | cmd 字段扩展 (t_pre_initial / t_post_swing / cur_step / swing_change_remaining) | paper 仅 1 个 t_strike | 我们扩 4 个内部字段 | 双段缩放 + lock 实现需要 |
| **R-1** [v5.7 新] | v_ball_in 采样 | paper 假设有真实弹道 | uniform sample (mag U(2,6), yaw π+U±40°, **pitch U(-75°,+75°)**) | 训练 cmd 是合成的, paper 没说 |
| **R-2** [v5.7 新] | target_land 采样 | paper 用真实回球场景 | **常量 (2.45, 0, 0.78)** = planner 默认平移 | 避免 cmd 维度爆炸 |
| **R-3** [v5.7 新] | flight_time 采样 | paper 用预测弹道 | uniform[0.30, 0.65] | 训练 cmd 没真实弹道 |
| ~~R-4~~ | (拟项: v_in/target/flight_time 进 critic privileged) | paper Table I 不含 | **不进 obs** [用户 ③] | DIVERGENCE 撤销, paper-aligned 严格 |

### 13.1 v5.7 净变化 (相对 v5.5)

| 项 | v5.5 状态 | v5.7 状态 | paper 对齐? |
|---|---|---|---|
| v̂_racket 来源 | sample (DIVERGENCE) | **planner Eq.6 推导** | ✓ paper-aligned |
| n̂_target 来源 | `v̂/‖v̂‖` (DIVERGENCE) | **paper Eq.5 直给** | ✓ paper-aligned |
| swing_type 决定 | uniform sample (DIVERGENCE) | **base-frame 几何 + 1-lock** | △ deploy heuristic |
| v_ball_in / target / flight_time / paddle_cor obs | N/A | 全部不进 (用户 ③) | ✓ paper-aligned |
| 新增 R-1/R-2/R-3 | 无 | 上游 sample 工程必要 | ⚠️ DIVERGENCE 但 paper 未说 |

**净效果**: v5.7 比 v5.5 paper-aligned 改善 4 项 (v̂ / n̂ / obs 严格 / r_g_ori 公式), 偏离 3 项 (R-1/R-2/R-3). 偏离的 3 项都是 paper 没说的工程选择 (paper 假设上游有真实弹道, 我们没有), 不是与 paper 直接冲突.

### 13.2 删除原 v5.5 中的 DIVERGENCE 条目

原 final.md / 原 plan v5.5 中含:
- "DIVERGENCE: cmd 直接 sample v̂_racket via v_mag/v_yaw/v_pitch"  ← **整条删除** (因 v5.7 已改 planner 推导, 不再 divergence)
- "swing_type 独立 uniform sample"  ← **整条删除** (因 v5.7 已改几何推导)

### 13.3 第一轮训练 monitor checklist

- [ ] DIVERGENCE A: r_i / r_g / r_r 顶层 weights 是否需要调
- [ ] DIVERGENCE B: r_g_base dense 全程 vs sparse-only ablation
- [ ] DIVERGENCE D: r_i 各 sub-term mean 值
- [ ] DIVERGENCE E + F: r_i / r_g_vel 在 forward / backward 两 clip 上的均衡度
- [ ] DIVERGENCE J: σ_pos curriculum 是否平滑收紧
- [ ] DIVERGENCE P: swing_type forehand:backhand 比例 (§12 #1)
- [ ] DIVERGENCE R-1: v_in pitch 范围 [-75°, +75°] 是否产生不合理 v̂_racket (§12 #5)
- [ ] DIVERGENCE R-2: target_land 常量是否限制 v̂ 多样性
- [ ] DIVERGENCE R-3: flight_time 范围是否合理
````

### 原因
- 原文 DIVERGENCE 索引散落, 不利于代码 review
- v5.7 新增 R-1/R-2/R-3 必须显式记录 (paper 没说的工程选择)
- v5.5 的 v̂ 直采 + swing_type 独立 uniform 两条 DIVERGENCE 被 v5.7 撤销, 必须删除避免误导

### 来源
plan §N7 (DIVERGENCE 总结), §M7 (paper alignment 状态对比).

---

## 11. §End Changelog — 追加 v5.5 / v5.6 / v5.7

### 位置
final.md 末尾 §End 节.

### 原内容
原 changelog 至 v5.4.

### 新内容 (追加)

```markdown
### v5.5 锁定决定 (合并自前轮, 已并入 §1–§11):
- **A2**: T_B 维度 11×7=77 (排除 right_paddle_blade), Critic obs 总 213 维
- **A7**: 删除 mimic_start_prob curriculum (每 episode RSI 从 ref clip), 课程项剩 8 项
- **A9**: expert_offset 标记为 base-frame, 一次性预处理
- **A10**: cmd noise per-swing freeze — resample 时采样, swing 内冻结, 仅注入 Actor obs (Critic 看真值)

### v5.6 (草案, 已被 v5.7 替代, 不并入 final.md):
- 提出用 planner.solve_paddle_target 推导 v̂/n̂ 替代直采 (用户 ① 改为 inline)
- 提出 swing_type 用 hit_y_world < base_y_world (用户 ⑫ 改为 base-frame y_mid=+0.157)
- 草拟 7 项 monitor 指标 (用户 ⑪ + 新增 1 项 → 8 项 §12)

### v5.7 锁定决定 (本版本, 已全部并入 §1–§13):
- **用户 ①**: planner 公式 inline 写在 mdp/commands.py, 不 import planner 模块 (§3.2)
- **用户 ②**: cmd 字段从 9 项扩到 14 项 (含 v_ball_in / target_land / flight_time / paddle_cor 4 项上游输入), 上游 4 项不进 obs (§3.1)
- **用户 ③**: obs 维度 Actor=86 / Critic=213 不变, paper Table I 严格对齐
- **用户 ④**: v_in_pitch 范围 [-75°, +75°] (允许下落球) (§3.2 Step 2)
- **用户 ⑤ + Q-N1**: target_land 常量 (2.45, 0, 0.78) world frame (§3.2 Step 3)
- **用户 ⑥**: flight_time uniform[0.30, 0.65] 秒 (§3.2 Step 4)
- **用户 ⑦**: t_pre_initial truncN[0.20, 0.90, 0.30, 0.65] 秒 (§3.2 Step 7)
- **用户 ⑧**: t_post_swing truncN[0.40, 1.10, 0.40, 0.75] 秒 (§3.2 Step 7)
- **用户 ⑨**: expert_offset base-frame 实测 (§11.4): forehand (+0.496, +0.208), backhand (+0.428, +0.106), y_mid=+0.157
- **用户 ⑩**: r_g_ori 用 cmd.n_target_world (paper Eq.5 直给), 不再 v̂/‖v̂‖ (§1 #6)
- **用户 ⑪**: forehand:backhand 偏斜监控不修 sample 逻辑 (§12 #1)
- **用户 ⑫ + Q-N2**: swing_type base-frame y_mid=+0.157 阈值 + 每 swing 1-change lock (§3.2 Step 6 + §3.3)
- **Q-N3**: paddle_cor = 0.85 写死 (不进 DR) (§3.2 Step 5)
- **DIVERGENCE 净变化**: 撤销 v̂ 直采 / swing_type uniform 两项 paper-divergence, 新增 R-1/R-2/R-3 三项工程必要 divergence
```

### 原因
版本演进必须显式追加, 否则后续维护看不出 v5.4 → v5.7 之间发生的变化.

### 来源
plan §N1, §N12, §N13.

---

## 12. 必须删除的内容 (整体清理)

| 位置 | 原内容 | 删除原因 |
|---|---|---|
| 原 §3 / §3.2 / §3.3 中所有 `v_mag~U[2,6] ∧ v_yaw=base_yaw+π+U[-40°,+40°] ∧ v_pitch~U[10°,60°]` 直采 v̂_racket 的代码块 / 表行 | v5.5 写法 | v5.7 用户 ① + ② 改为 planner Eq.5+Eq.6 推导 |
| 原 v5 累计 #12 "Cmd v̂_racket 直采" 决定项 | header 列表 | v5.7 撤销 |
| 原 v5 累计 #16 "swing_type 独立 uniform 采样" 决定项 | header 列表 | v5.7 用户 ⑫ 改为几何推导 + lock |
| 原 §11.4 expert_offset 注释 "world delta + 假设 yaw≈0" | §11.4 旧文 | v5.7 §N6 实测发现 yaw 65° 差异, world frame 不可用 |
| v5.5 → v5.6 → v5.7 之间任何 "DIVERGENCE 待 ablation" 过渡子句 | 散落各处 | 仅保留终态 |
| 末尾 "Q-N1 / Q-N2 / Q-N3 待定" / "OPEN" / "⏳" 标记 | 任何残留 | 已锁定无 OPEN 项 |
| v5.6 §M1–§M13 整段 (如果 final.md 此前曾被部分写入过) | 任何残留 | 用户已审 → v5.7 已并入, M 系列是过渡草案 |

---

## 13. 必须保留不变的内容 (整体保留, 验证完整)

| 章节 | 状态 |
|---|---|
| §1.1 ℬ_pos (11 bodies, 排除 right_paddle_blade) | 不变 |
| §1.2 J (10 joints, 排除 right_wrist_roll_joint) | 不变 |
| §2 obs 表 14 项 (维度 Actor=86 / Critic=213) | 仅 #6 v̂_racket 来源标注更新 (§3 修订项), 其他不变 |
| §4 weights + r_r 12 项套餐 | 不变 |
| §5 actions (mimic per-joint scale, K) | 不变 |
| §6 terminations 4 项 + §6.4 table (RigidBody + 双 ContactSensor + r_table_contact) | 不变 |
| §7 events (DR + RSI + per-swing-freeze cmd noise) | 仅 #4 swing_type 改为几何推导 (§3.2 Step 6 引), 其他不变 |
| §8 curriculum 8 项 (mimic_start_prob 已删 in v5.5 A7) | 不变 |
| §9 scene + §10 sim | 不变 |
| §11.1 Q1 双段插值 (含 §11.1.1–§11.1.5 完整代码) | 不变 — v5.4 锁定核心机制 |
| §11.2 Q2 swing_type 不进 obs (作 ref clip selector) | 不变 |
| §11.3 Q3 r^bp p_rel xy 减 z 保留 | 不变 |
| §11.6 (旧 §11.5) Q4 t_to_hit 命名 | 不变, 仅 dataclass 字段加新项 (§8 修订项) |

---

## 14. 整合执行顺序建议

1. **先备份原 final.md** (cp 一份 final_v5pre.md.bak, 防整合出错)
2. **整文件覆盖**写入新 v5.7 版本 (不要做散弹 Edit, 否则整合不彻底)
3. 写完后按 §13 (本清单) 验证保留章节完整
4. 按 §12 (本清单) 验证删除章节无残留
5. **关键交叉验证**: §3.2 Step 7 中 expert_offset_base 数值与 §11.4.2 表中数值一致 (forehand (+0.496, +0.208), backhand (+0.428, +0.106))
6. **关键交叉验证**: §3.2 Step 6 中 Y_MID_BASE = 0.157 与 §11.4.4 推导一致
7. **关键交叉验证**: §1 #6 r_g_ori 公式中 n̂_target 来源与 §3.2 Step 5 输出一致 (都指 `cmd.n_target_world`)
8. **关键交叉验证**: §11.5 dataclass 字段与 §3.1 cmd 字段表 14 项一一对应
9. 整合后**进入 Phase 2** (mdp/ 代码), final.md 是唯一参考依据

---

## 15. 整合后实现优先级 (Phase 2)

按 §N10 给出的顺序:

1. `mdp/commands.py` — `PingpongCommand` class: §3.2 七步 + §3.3 lifecycle (含 swing lock 状态机) + §3.2 Step 5 inline 物理公式 + §3.2 Step 5 退化 fallback
2. `mdp/motion_loader.py` — expert_clip dict + `get_ref_state(cmd, dt)` (§11.1.2 双段 lerp/slerp) + §11.4 expert_offset_base 一次性预处理
3. `mdp/observations.py` — §2 表 14 项, swing_type dispatch ref state (Critic T_B / [q̄, q̇̄] 路由)
4. `mdp/rewards.py` — r_i (DM kernel, J 排除 right_wrist_roll_joint, ℬ_pos 排除 right_paddle_blade) + r_g (含 §1 #6 r_g_ori 用 `cmd.n_target_world`) + r_r 套餐
5. `mdp/events.py` — §7 全套
6. `mdp/terminations.py` — §6 全套
7. `mdp/curriculums.py` — §8 全套 + §12 monitor (extras dict 输出)
8. `tasks/pingpong/hitter_env_cfg.py` (或 pingpong_env_cfg.py) — 装配

---

## 16. 整合后 sanity test

按 plan §N13.5:

1. **Syntax review**: 通读, 确认无 v5.5 残留代码 / Q-N 待定项
2. **DIVERGENCE 索引完整性**: §13 表覆盖 A–R (含 R-1/R-2/R-3)
3. **Cross-reference**: 数值一致 (§3.2 / §11.4 / §11.5)
4. **Code-ready**: §3.2 + §11.1 代码可直接复制到实现
5. **Phase 2 触发**: final.md 完成后才能开 mdp/ 实现

---

## 附录 A. 关键文件路径

- **目标**: `source/unitree_rl_lab/unitree_rl_lab/tasks/pingpong/final.md` (整文件覆盖)
- **本清单**: `source/unitree_rl_lab/unitree_rl_lab/tasks/pingpong/final_v57_changes.md` (你正在读)
- **plan 来源**: `/home/woan/.claude/plans/frolicking-weaving-lighthouse.md` §N1–§N13
- **历史 v2 doc** (保留, 不删): `source/unitree_rl_lab/unitree_rl_lab/tasks/pingpong/REWARD_DESIGN.md`
- **expert clip 数据**: `motion_datasets/pingpong/humanoid_data/final/expert/{forward/forward_001.npz, backward/backward_004.npz}`
- **Robot asset 复用**: `unitree.py:951` `UNITREE_G1_23DOF_PADDLE_MIMIC_CFG` + `unitree.py:957-968` `UNITREE_G1_23DOF_PADDLE_MIMIC_ACTION_SCALE`
- **物理参考但不 import**: `mdp/planner.py:652-732` `solve_paddle_target` (公式直接复制到 commands.py)

## 附录 B. 关键常量速查

| 常量 | 值 | 出处 |
|---|---|---|
| Y_MID_BASE (swing_type 阈值) | +0.157 m | §11.4.4 |
| target_land_world (Q-N1) | (2.45, 0, 0.78) | §3.2 Step 3 |
| paddle_cor (Q-N3) | 0.85 | §3.2 Step 5 |
| g | 9.81 m/s² | §3.2 Step 5 |
| dt | 0.02 s @ 50Hz | §3.3 |
| strike window | abs(t_to_hit) ≤ 0.06 s = ±3 帧 | §1 / §3.3 |
| degenerate threshold | ‖Δv‖ < 1e-9 | §3.2 Step 5 fallback |
| expert_offset["forehand"] (base) | (+0.496, +0.208) | §11.4.2 |
| expert_offset["backhand"] (base) | (+0.428, +0.106) | §11.4.2 |
| expert_pre_duration["forehand"] | 0.74 s | §11.4.6 |
| expert_post_duration["forehand"] | 0.88 s | §11.4.6 |
| expert_pre_duration["backhand"] | 0.40 s | §11.4.6 |
| expert_post_duration["backhand"] | 0.86 s | §11.4.6 |
| BLADE_IDX | 24 | §11.4.1 select_x04_clips.py:21 |
| PELVIS_IDX | 0 | §11.4.1 |

---

## 附录 C. 详细数字举例 (整合后用于 sanity check)

> 本附录补充几个完整数字 walkthrough, 用于代码实现完成后逐 step 比对 cmd / ref state, 防止整合时数值出现 off-by-one 或方向错误.

### C.1 cmd resample 示例 (forehand 长 pre)

**输入** (假设 RNG 抽到):
- hit point: hit_x=0.4, hit_y=-0.30, hit_z=0.40 (hit_y < 0 表示 robot 左手前方)
- v_in_mag=4.0, v_in_yaw=π+10°=190°, v_in_pitch=-30° (来球下降中)
- target_land=(2.45, 0, 0.78) [常量]
- flight_time=0.45 s
- paddle_cor=0.85 [常量]
- robot base: pos=(0, 0, 0.76), yaw=0 (RSI 后初始)

**Step 2 计算 v_ball_in_world**:
```
v_ball_in = 4.0 · [cos(190°)cos(-30°), sin(190°)cos(-30°), sin(-30°)]
         = 4.0 · [-0.853, -0.150, -0.500]
         = [-3.413, -0.601, -2.000]   m/s
```
合理性: x < 0 → 朝 -x 飞 (机器人在 x=0, 球从对方桌 x>0 飞过来) ✓; z < 0 → 下降中 ✓.

**Step 5 物理推导** (g=9.81, T=0.45):
```
v_out = (target_land - p_hit) / T + (0, 0, 0.5·g·T)
     = ((2.45-0.4, 0-(-0.30), 0.78-0.40))/0.45 + (0, 0, 0.5·9.81·0.45)
     = (2.05/0.45, 0.30/0.45, 0.38/0.45) + (0, 0, 2.207)
     = (4.556, 0.667, 0.844) + (0, 0, 2.207)
     = (4.556, 0.667, 3.051)   m/s

delta_v = v_out - v_ball_in
       = (4.556-(-3.413), 0.667-(-0.601), 3.051-(-2.000))
       = (7.969, 1.268, 5.051)
‖delta_v‖ = sqrt(7.969² + 1.268² + 5.051²) = sqrt(63.50 + 1.61 + 25.51) = sqrt(90.62)
         ≈ 9.519
n_target = (7.969, 1.268, 5.051) / 9.519
        ≈ (0.837, 0.133, 0.531)   单位向量

v_in · n  = -3.413·0.837 + (-0.601)·0.133 + (-2.000)·0.531
         = -2.857 - 0.080 - 1.062
         = -3.999

v_out · n = 4.556·0.837 + 0.667·0.133 + 3.051·0.531
         = 3.813 + 0.089 + 1.620
         = 5.522

v_pad_n = (5.522 + 0.85·(-3.999)) / (1 + 0.85)
       = (5.522 - 3.399) / 1.85
       = 2.123 / 1.85
       ≈ 1.148   m/s

v_racket_hat = 1.148 · (0.837, 0.133, 0.531)
            ≈ (0.961, 0.153, 0.609)   m/s
‖v_racket_hat‖ ≈ 1.148   m/s
```
合理性 sanity:
- ‖v_racket_hat‖ ≈ 1.15 m/s 偏小 (低于 expert forehand 4.42 m/s), 因为 v_in 已经有较大的 -z 分量 (球下降)
- 监控 §12 #5: per-iter mean ‖v̂‖ 期望在 [1.5, 5.0], 本例略低于阈, 单点不报警

**Step 6 swing_type**:
```
hit_xy_world = (0.4, -0.30)
base_xy_world = (0, 0)
yaw_robot = 0
R_inv(0) = I
hit_xy_base = (0.4, -0.30)
hit_y_base = -0.30
0.30 < 0.157? NO (因为 -0.30 < 0.157)  → -0.30 < 0.157 ✓
swing_type = "backhand"
```
等等 — hit_y_base = -0.30 < +0.157, 所以 swing_type = "backhand". 但 hit_y < 0 在 deploy 端 (planner.py:970, world frame) 通常表示"球在右侧 → forehand". **这就是 §11.4.5 描述的方向反转**: 因为 expert forward_001 录制时 paddle 在 base +y 侧 (+0.208), backward_004 也在 +y 侧但更靠中线 (+0.106), 训练阈值 +0.157 把所有 hit_y_base < +0.157 都归入 backhand (含负的 hit_y_base). **整合时务必引用 §11.4.5 解释清楚, 否则 reviewer 会以为是 bug**.

**Step 7 base_target**:
```
expert_offset_b["backhand"] = (+0.428, +0.106)
yaw_robot = 0, R_b2w = I
expert_offset_world = (+0.428, +0.106)
p_base_xy_world = (0.4, -0.30) - (+0.428, +0.106)
              = (-0.028, -0.406)
```

**Step 7 时间字段** (假设 truncN 抽到):
- t_pre_initial = 0.50 s → sim_pre_steps = 25 step
- t_post_swing = 0.60 s → sim_post_steps = 30 step
- t_to_hit = 0.50, cur_step = 0

### C.2 ref state 双段插值示例 (用 C.1 的 cmd, swing 中段)

ref clip = backward_004 (因 swing_type = "backhand"), impact=20, clip_len=64.

```
sim_pre_steps  = t_pre_initial / dt = 0.50 / 0.02 = 25
sim_post_steps = t_post_swing  / dt = 0.60 / 0.02 = 30
total_steps    = 55
```

**第 0 step** (resample 后):
- cur_step=0, t_to_hit=0.50
- progress (pre) = 0/25 = 0
- ref_frame_f = 0 · 20 = 0.0
- ref state = clip[0] (lerp α=0)

**第 12 step**:
- cur_step=12, t_to_hit = 0.50 - 12·0.02 = 0.26
- progress (pre) = 12/25 = 0.48
- ref_frame_f = 0.48 · 20 = 9.60
- ref state = lerp(clip[9], clip[10], α=0.60); quat 用 slerp

**第 25 step (击球瞬间)**:
- cur_step=25, t_to_hit = 0.50 - 25·0.02 = 0.00 ← 击球
- progress (pre) = 25/25 = 1.0
- ref_frame_f = 1.0 · 20 = 20.0
- ref state = clip[20] (= impact 帧)
- **strike window 激活**: abs(t_to_hit) ≤ 0.06s, 帧 [22, 28] = ±3 step 都在窗内
- r_g_pos / r_g_vel / r_g_ori 此 step sparse 激活

**第 26 step**:
- cur_step=26, t_to_hit = -0.02
- progress (post) = (26-25)/30 = 0.0333
- ref_frame_f = 20 + 0.0333·(63-20) = 20 + 1.43 = 21.43
- ref state = lerp(clip[21], clip[22], α=0.43)

**第 55 step (post 段末)**:
- cur_step=55, t_to_hit = 0.50 - 55·0.02 = -0.60
- progress (post) = 30/30 = 1.0
- ref_frame_f = 20 + 1.0·43 = 63 = clip_len-1
- ref state = clip[63] (= clip 末)

**第 56 step (resample 触发)**:
- t_to_hit = -0.62 ≤ -t_post_swing = -0.60? 是 (-0.62 ≤ -0.60 等价 0.62 ≥ 0.60 ✓)
- 触发 resample_cmd, cur_step → 0, swing_change_remaining → 1

### C.3 swing_type 1-change lock 示例

**场景**: episode 内, robot base 从 (0,0) 漂移到 (0, +0.3), 中途 hit_y_base 越过阈值.

**Step A** (resample 时, robot @ (0,0,0.76), base_y=0):
- hit_y_world = 0.10
- hit_y_base = 0.10 - 0 = 0.10
- 0.10 > 0.157? NO → swing_type = "backhand"
- swing_change_remaining = 1

**Step B** (10 step 后, robot 漂移 base_y=0.30):
- 每 step 重算 hit_y_base:
- hit_y_base = 0.10 - 0.30 = -0.20
- -0.20 > 0.157? NO → swing_type 保持 "backhand", 不变

**Step C** (20 step 后, robot 又漂移 base_y=-0.10):
- hit_y_base = 0.10 - (-0.10) = 0.20
- 0.20 > 0.157? YES → 新 swing_type = "forehand"
- new_swing != cmd.swing_type ("backhand" → "forehand")?  YES
- 切换:
  - cmd.swing_type = "forehand"
  - cmd.swing_change_remaining = 0  ← 用掉机会
  - ref_clip 也跟着切换 (backward_004 → forward_001)
  - **可能产生 r_i 跳变**, 监控 §12 #3 swing_flip_rate

**Step D** (再 5 step 后, robot 又漂移 base_y=-0.30):
- hit_y_base = 0.10 - (-0.30) = 0.40
- 0.40 > 0.157? YES → 新 swing_type = "forehand"
- swing_change_remaining = 0, 不允许再变, swing_type 锁死 "forehand"

**Step E** (击球后 t_to_hit = -0.05 < 0, post 阶段):
- 不再判断 swing_type, 锁死

**Step F** (t_to_hit = -t_post_swing 触发 resample):
- swing_change_remaining 重置为 1, swing_type 按当前 base 重算

### C.4 退化情况示例 (Step 5 fallback)

**场景**: v_ball_in 与 v_out 几乎相等 (球擦边而过).

**输入**:
- p_hit = (0.4, 0, 0.4)
- v_ball_in = (-3.0, 0, 0)  (水平来球)
- target_land = (2.45, 0, 0.78)
- flight_time = 0.6

**计算**:
```
v_out = ((2.45-0.4, 0, 0.78-0.4))/0.6 + (0, 0, 0.5·9.81·0.6)
     = (3.417, 0, 0.633) + (0, 0, 2.943)
     = (3.417, 0, 3.576)
delta_v = (3.417-(-3.0), 0, 3.576-0) = (6.417, 0, 3.576)
‖delta_v‖ = 7.347   ← 远大于 1e-9, 不退化
```

(实际退化非常罕见 — 物理上需要 v_out ≈ v_in, 即 hit_pos 距 target_land 的位移恰好等于 v_in·T. 训练 sample 范围内基本不会触发.)

**真退化 fallback** (理论):
```
n_target = (-1, 0, 0)
v_racket_hat = 2.0 · (-1, 0, 0) = (-2, 0, 0)
监控 §12 #6 计数 +1
```

实现端 retry 5 次微扰 v_in ± gauss(0, 0.05) 后仍退化才 fallback (§N8). 第一轮训练观察 fallback rate < 0.1%.

---

## 附录 D. 整合时常见错误防御清单

> 这一节是"整合时容易踩的坑"汇总, 写到 final.md 时**不必复制本节**, 但整合执行时**必须自检**.

### D.1 swing_type 方向反转

**陷阱**: `swing_type = "forehand" if hit_y_base > 0.157 else "backhand"` 与 planner.py:970 `forehand if hp[1] < bp[1] else backhand` (hit_y 小为 forehand) 字面相反.

**原因**: 两个 expert clip 录制时 paddle 都在 base +y 侧 (+0.208 / +0.106), 阈值是中点 +0.157 而非 0.

**自检**: 实现端读 §11.4.5 注释; reward log 第一轮中两 swing_type 出现频率应大致平衡.

### D.2 expert_offset frame 混淆

**陷阱**: 一次性预处理时, 容易直接存 `p_blade_w[imp, :2] - p_pelv_w[imp, :2]` (= world delta), 但因两 clip yaw 不同 (63.6° vs 128.8°), 训练时旋转回 world 会得到错误目标.

**自检**: `expert_offset_base["forehand"]` 的 x 分量必为 +0.496 (有 yaw 校正), 若得到 +0.035 (= world dx) 说明缺少 R(-yaw) 旋转.

### D.3 r_g_ori 公式 n̂_target 来源

**陷阱**: 常见写法 `n_target = v_racket_hat / np.linalg.norm(v_racket_hat)` — 数值上正确 (因 v_pad ∥ n̂), 但**不是 paper Eq.5 直定义**, 等价是巧合, 不是物理来源.

**自检**: rewards.py 中读 `cmd.n_target_world` 即可, 不应该出现 `v_racket_hat` 的 normalize.

### D.4 strike window 边界

**陷阱**: `abs(t_to_hit) ≤ 0.06` 对应 ±3 step (因 dt=0.02), 共 7 step (-3, -2, -1, 0, +1, +2, +3). 不是 ±0.06s = 6 step.

**自检**: 单 swing 跑一遍, 统计 strike_window 激活的 step 数应为 7.

### D.5 ref_frame_f 浮点 vs 整数

**陷阱**: 直接 `clip[int(ref_frame_f)]` 会丢失插值精度, 击球帧对齐失败.

**自检**: cur_step = sim_pre_steps 那一刻, ref_frame_f 应严格 = impact_frame (浮点等于整数, 无插值误差); 此 step ref body pos 应 = clip[impact] 完全相等.

### D.6 swing_change_remaining 重置时机

**陷阱**: resample 时忘了重置 swing_change_remaining = 1, 一直是 0, 之后 episode 内再也不能变 swing_type.

**自检**: 每次 resample_cmd 函数结束前显式赋值 swing_change_remaining = 1.

### D.7 t_to_hit 与 cur_step 同步

**陷阱**: 单独维护 t_to_hit 和 cur_step, 出现 t_to_hit = 0.05 但 cur_step = 30 (不匹配) 的情况.

**自检**: 永远 `t_to_hit = t_pre_initial - cur_step · dt` 关系成立; 实现端只维护 cur_step, t_to_hit 由它当场算更安全 (但要注意 t_to_hit 还要进 obs, 不能省).

### D.8 v_ball_in pitch 范围下界

**陷阱**: v5.5 写 `[10°, 60°]` 是上升球, v5.7 改为 `[-75°, +75°]` 是含下降球. 整合时容易复制旧范围.

**自检**: rewards.py 中 v_in_pitch sample 行必须含负数边界.

### D.9 target_land frame 选错

**陷阱**: 直接照 planner.py:777 默认 (0.7, 0, 0.06) 复制, 在我们 frame 下落到桌面以下 70cm.

**自检**: target_land 必须是 (2.45, 0, 0.78), 不是 (0.7, 0, 0.06).

### D.10 obs 维度 paper 严格

**陷阱**: 看到 v_ball_in / target_land / flight_time 4 项 cmd, 觉得 "应该让 critic 看到 privileged" 加进 Critic obs.

**自检**: 用户 ③ 决议明确 — 4 项**全不进 obs**, Critic 维度严格 213.

---

## 附录 E. 整合后 Phase 2 实现端 Checklist (commands.py)

> 本附录给整合后写 mdp/commands.py 的开发者一个 checklist, 防止漏功能.

```python
class PingpongCommand:
    # [ ] 14 cmd 字段 (§3.1) 全部 init
    # [ ] resample_cmd 实现 §3.2 Step 1–7
    #     [ ] Step 1: hit_x=0.4 写死, hit_y/z uniform
    #     [ ] Step 2: v_in_pitch [-75°, +75°]
    #     [ ] Step 3: target_land (2.45, 0, 0.78) 写死
    #     [ ] Step 4: flight_time uniform[0.30, 0.65]
    #     [ ] Step 5: paddle_cor=0.85; 物理公式 inline (Eq.5+Eq.6); ‖Δv‖<1e-9 fallback
    #     [ ] Step 6: hit_y_base 计算; Y_MID_BASE=0.157 阈值; 注意 +y_base 是 forehand
    #     [ ] Step 7: t_pre/t_post truncN; expert_offset_base 旋转回 world
    # [ ] _update_command (每 step):
    #     [ ] t_to_hit -= dt
    #     [ ] cur_step += 1
    #     [ ] swing_type 1-change lock 状态机 (§3.3):
    #         [ ] t_to_hit > 0 且 swing_change_remaining > 0 时重算 swing_type
    #         [ ] 若变更, 同时切 ref_clip + 把 swing_change_remaining 置 0
    #         [ ] t_to_hit ≤ 0 时不再变
    #     [ ] resample 触发: t_to_hit ≤ -t_post_swing
    # [ ] 退化 fallback retry 5 次后再用 (-1, 0, 0)
    # [ ] 监控 §12 8 项 metric 输出到 extras dict
    # [ ] cmd noise per-swing freeze (v5.5 A10): 注意 cmd 真值 vs Actor obs 噪声值分开
```

---

## 附录 F. 整合时关键的 `Edit` 替换字符串提示

> 整合方案是"整文件覆盖" (Write 工具写入), 但若整合时想用 Edit 工具做替换 (例如 Phase 2 实现期间局部更新 final.md), 以下是几个关键 anchor 的 grep 提示, 帮助快速定位.

| 修改点 | grep anchor (find_string) | 期望命中行 |
|---|---|---|
| §1 #6 r_g_ori | `n̂ = v̂_racket / ‖v̂_racket‖` | r_g_ori 行 (约 #50) |
| §3.1 cmd field 旧表 | `swing_type` 第一次出现的表行 | §3.1 表头 (约 #130) |
| §3.2 v_mag 直采代码 | `v_mag   = uniform(2.0, 6.0)` | §3.2 旧代码 (约 #170) |
| §11.4 expert_offset | `expert_offset["forehand"]` | §11.4 旧注释 (约 #960) |
| §11.5 dataclass | `class PingpongCommand:` | §11.5 旧 dataclass (约 #1037) |
| §End changelog | `## §End. v5.4 状态` 或 `### v5.4 锁定` | 末尾 (约 #1100) |

每个 anchor 都是 unique 字符串, Edit 时不会误命中其他位置.

---

## 附录 G. 一句话总结 (整合时心法)

> **v5.7 的核心不是改公式, 是改 cmd 的"上下游分工"**:
>
> - **上游** (v_ball_in / target_land / flight_time / paddle_cor) 是 cmd 真实输入, **不暴露给 policy**
> - **下游** (v̂_racket / n̂_target) 由 paper Eq.5+Eq.6 推导, **暴露给 policy**
> - **swing_type** 由 hit 与 base 的几何关系决定 (base-frame y 阈值), 不再 sample
>
> 整合时最容易出错: ① 把 v_in 错抄成 v̂_racket; ② swing_type 方向反转; ③ target_land frame 选错.
>
> 三道防线: (a) 按 §11.4.2 / §11.4.5 验数值, (b) 按附录 D 自检, (c) 按附录 E 实现 checklist.
