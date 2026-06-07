# HITTER 23dof Pingpong — 当前完整工程设计 (v58)

> 本文档替代旧 `final.md`。基于 `final_v57.md` 的设计骨架，整合 v57→v58 之间的实证修正：M4 paddle face convention 修复 + R11 swing-first 撤销 + R12 Plan B 半 gate + R13 base 稳定 reward 套餐。
>
> **当前 baseline**：`logs/rsl_rl/unitree_g1_23dof_pingpong_hitter/2026-05-29_14-54-15_v3_swing_first_base_ori`（命名沿用 V3 但实际是 v58 配置）。iter 1000 已达成 cos_sim=+0.585、EL=462、hard_contact=0.20 — 全面超越 V1 21-04-08 历史最佳。
>
> 配套阅读：[TROUBLESHOOTING.md](TROUBLESHOOTING.md)（问题/方案流程记录）、[REWARD_DESIGN.md](REWARD_DESIGN.md) / [COMMAND_DESIGN.md](COMMAND_DESIGN.md)（设计推导）。

---

## 0. Hitter Planner 完整设计 — 训练端 / 部署端 双 planner 对照

> **为什么需要 planner？** HITTER 论文 §IV-C 把 cmd 的物理推导分两部分：(a) 弹道预测——给定来球状态，预测击球点 `p_hit`、击球时刻 `t_to_hit`；(b) Eq.5/Eq.6 正/逆解——给定击球点 + 出球目标，反推球拍速度 `v_racket` 和法向 `n_target`。这两步必须一致：训练端用合成 cmd 跳过弹道预测，但 hitter_real / 部署端必须真做。
>
> 工程上拆成两个文件：
> - **`mdp/planner.py`**（部署专用，1262 行）：numpy + torch 混合，HitterPlanner 类，含 RK2 积分、bounce 检测、kf 拟合、参数标定。**训练时不 import**。
> - **`mdp/planner_for_training.py`**（训练 + hitter_real eval 用，272 行）：纯 torch batched 实现，单函数 `plan_pingpong_hits()`。**hitter / hitter_real 都可调用**。

### 0.1 角色矩阵

| Pipeline | 来球状态来源 | 调用 planner？ | cmd 字段填充方式 |
|---|---|---|---|
| **hitter (训练 v58)** | 合成（`v_in_mag/yaw/pitch ~ U(...)`） | ❌ 不调用 | [commands.py `_sample_new_swing`](mdp/commands.py) 直接 inline Eq.5/Eq.6（`_solve_paddle_target`）+ 几何 swing 分类 |
| **hitter_real (eval 续训)** | 真实物理球 rollout | ✅ `plan_pingpong_hits()` | [real_commands.py:231-275](mdp/real_commands.py) 调 planner，把 `TrainingPlannerOutput` 字段直接拷贝到 cmd buffer |
| **deploy (硬件)** | sensor / 视觉跟踪 + EKF | ✅ `HitterPlanner` (planner.py) | runtime bridge 把 `predict_hit_plane()` + `solve_paddle_target()` 输出转换到训练 world frame 后送 policy |

### 0.2 训练端 `plan_pingpong_hits()` 完整签名

```python
def plan_pingpong_hits(
    # ====== 必传 (来球 + robot 状态 + 目标落点) ======
    ball_pos_world: torch.Tensor,            # (N, 3) 当前球位置 world frame
    ball_vel_world: torch.Tensor,            # (N, 3) 当前球速度 world frame
    robot_root_pos_world: torch.Tensor,      # (N, 3) robot pelvis world pos
    robot_root_quat_world: torch.Tensor,     # (N, 4) wxyz convention!
    target_land_world: torch.Tensor,         # (N, 3) 出球目标落点 (e.g. 对方半台中心 (2.45, 0, 0.78))
    # ====== 必传 (从 cmd cfg auto-derive) ======
    expert_offset_base: torch.Tensor,        # (2, 2) [forehand_xy, backhand_xy] 在 base frame
    y_mid_base: float,                       # forehand/backhand 分界
    # ====== 可选 (含合理默认) ======
    table_top_z: float | Tensor = 0.76,
    ball_radius: float | Tensor = 0.02,
    valid_mask: torch.Tensor | None = None,  # (N,) bool, 哪些 env 要算
    x_hit_world: float | Tensor = 0.4,       # 击球平面 x 坐标
    table_center_x_world: float | Tensor | None = None,  # default = x_hit + table_half_x
    table_center_y_world: float | Tensor | None = None,  # default = 0
    table_half_x: float = 1.37,              # 半张桌长 (国际标准 2.74m / 2)
    table_half_y: float = 0.7625,            # 半张桌宽 (1.525m / 2)
    flight_time: float | Tensor = 0.45,      # paper Eq.5 输入
    paddle_cor: float | Tensor = 0.85,       # paper Eq.6 e (橡胶 + 球)
    dt: float = 0.01,                        # 物理积分步长 (10ms)
    max_time: float = 1.50,                  # 最长向前预测时间
    drag_k: float = 0.10257265376884504,     # 球-空气拖曳 (从真球轨迹拟合)
    bounce_ch: float = 0.727005044772834,    # 桌反弹切向衰减 (xy 分量)
    bounce_cv: float = 0.9018357357260598,   # 桌反弹法向衰减 (z 分量)
    min_t_to_hit: float = 0.05,              # 太近的球丢弃
    max_t_to_hit: float = 1.20,              # 太远的球丢弃
    hit_z_range: tuple[float, float] = (0.85, 1.25),  # 击球 z 区间
) -> TrainingPlannerOutput
```

### 0.3 `TrainingPlannerOutput` — 16 字段完整表

每个字段：含义 / shape / 生成方式 / 消费者。

| # | 字段 | shape | 含义 | 生成方式 | 消费者 |
|---|---|---|---|---|---|
| 1 | `p_hit_world` | (N, 3) | 预测击球点（world frame） | **Step 2** ball rollout 找到 x=x_hit_world 平面 crossing 时刻的球位置（线性插值） | obs `pingpong_hit_position_b`, `r_g_pos`, `_compute_swing_type` |
| 2 | `v_ball_in_world` | (N, 3) | 击球瞬间来球速度 | **Step 2** 同样 alpha 线性插值 prev_v 和 next_v | Eq.5/Eq.6 输入；不进 obs |
| 3 | `v_ball_out_world` | (N, 3) | 期望出球速度 (球台抛物线方程倒推) | **Step 3a** `(target_land - p_hit) / T + (0,0,0.5*g*T)` | sanity monitor；不进 obs |
| 4 | `v_racket_hat_world` | (N, 3) | 球拍击球瞬间速度（paper Eq.6） | **Step 3c** `v_pad_n * n_target_world` (= `v_pad_n` 沿法向) | obs `pingpong_racket_velocity_w`, `r_g_vel` |
| 5 | `n_target_world` | (N, 3) | 球拍目标法向（paper Eq.5） | **Step 3b** `delta_v / ‖delta_v‖`，degenerate 时 fallback `(-1,0,0)` | `r_g_ori`（**唯一消费者**，不进 obs） |
| 6 | `target_land_world` | (N, 3) | 出球目标落点 | 调用者直接传入（hitter 用常量 `(2.45, 0, 0.78)`，hitter_real 可任意） | sanity；不进 obs |
| 7 | `p_base_xy_world` | (N, 2) | base 应该站到的位置（world xy） | **Step 5** `p_hit_xy - R(yaw_robot) @ expert_offset_base[swing_type]` ⬅ **关键关系，见 §0.6** | obs `pingpong_base_position_error`, `r_g_base` |
| 8 | `t_to_hit` | (N,) | 距离击球还有多久（秒） | **Step 2** `(step-1+alpha) * dt` | obs `pingpong_t_to_hit`，cmd 时序 gate |
| 9 | `swing_type` | (N,) long | 0=forehand, 1=backhand | **Step 4** base-frame `hit_y_base` 与 `y_mid_base` 比，符号由 `swing_y_sign` 决定 | ref clip 选择（routing）；不进 obs |
| 10 | `planner_valid` | (N,) bool | 这个 env 是否成功预测到击球点 | True ↔ Step 2 找到了合法 crossing | caller 判断是否用新 cmd vs 保持上一个 |
| 11 | `plan_mode` | (N,) long | `PLAN_INVALID=0` / `PLAN_FRESH=1` / `PLAN_HELD=2` / `PLAN_FROZEN=3` | Step 2 找到 → FRESH；其他状态由 caller 写 | obs／调试 |
| 12 | `bounce_count_pred` | (N,) long | 直到击球瞬间预测会 bounce 几次 | rollout 中累积 | hitter_real curriculum 监控 |
| 13 | `x_hit_used` | (N,) | 实际用的击球平面 x | clone 输入 `x_hit_world` | 调试 / 可视化 |
| 14 | `fallback_reason` | (N,) long | 0=ok / 1=no_valid_mask / 2=z_out_range / 3=t_out_range / 4=ball_not_moving_to_robot | rollout 内分支 | 调试 |
| 15 | `traj_p` | (N, max_steps+1, 3) | 完整 forward roll 轨迹 (含 bounce) | 每步存一帧 | debug visualizer |
| 16 | `traj_valid` | (N, max_steps+1) | 轨迹哪些帧有效 | rollout 中 mask | debug visualizer |

> **不在 plan output 但 cmd 需要的字段**：`flight_time` / `paddle_cor` / `t_pre_initial` / `t_post_swing` / `cur_step` / `swing_change_remaining` / `hit_y_base` / `noise_*`。这些由 caller（[real_commands.py](mdp/real_commands.py)）独立采样或维护，planner 不管。

### 0.4 算法 Step-by-step

```
═══ Step 1: 准备 ════════════════════════════════════════════
- broadcast scalar 参数到 (N,) tensor
- 默认 valid_mask = ones
- 默认 table_center_x = x_hit + table_half_x（站机器人侧远端往后半张台）
- 初始化 _empty_output（fallback_reason=1, planner_valid=False）

═══ Step 2: Ball forward-roll ═══════════════════════════════
for step in 1..max_steps:
    speed = ‖v‖
    acc = (0, 0, -9.81) - drag_k · speed · v       # 二次拖曳
    v_next = v + acc · dt
    p_next = p + v_next · dt

    # bounce 检测 (从台面上方下穿 + 落点在桌内 + z 速度向下)
    on_table_xy = |p_next.x - table_center_x| ≤ table_half_x
                  AND |p_next.y - table_center_y| ≤ table_half_y
    bounced = (p.z > center_z) AND (p_next.z ≤ center_z) AND (v_next.z < 0) AND on_table_xy
    if bounced:
        p_next.z = center_z
        v_next.xy *= bounce_ch
        v_next.z = -v_next.z * bounce_cv
        bounce_count += 1

    # 击球平面 crossing 检测（球向机器人方向 + x 穿越 x_hit）
    moving_to_robot = prev_v.x < -0.05
    crosses = (prev_p.x ≥ x_hit) AND (p_next.x ≤ x_hit)
    eligible = valid_mask AND not_found AND moving_to_robot AND crosses
    if eligible:
        alpha = (x_hit - prev_p.x) / (p_next.x - prev_p.x)   # 线性插值系数
        p_hit = prev_p + alpha · (p_next - prev_p)
        v_hit = prev_v + alpha · (v_next - prev_v)
        t_hit = (step-1 + alpha) · dt
        z_ok  = hit_z_range[0] ≤ p_hit.z ≤ hit_z_range[1]
        t_ok  = min_t_to_hit ≤ t_hit ≤ max_t_to_hit
        if z_ok AND t_ok:
            写入 out.p_hit_world / v_ball_in_world / t_to_hit
            out.planner_valid = True
            out.plan_mode = PLAN_FRESH
            out.fallback_reason = 0
        else:
            out.fallback_reason = 2 (z bad) or 3 (t bad)

═══ Step 3: paper Eq.5 + Eq.6 (solve_paddle_targets_batched) ══
仅对 planner_valid 的 env：
  v_ball_out = (target_land - p_hit) / flight_time + (0, 0, 0.5·g·flight_time)
  delta_v = v_ball_out - v_ball_in
  if ‖delta_v‖ < 1e-9:
      n_target = (-1, 0, 0)              # fallback (向 robot)
      v_racket_hat = 2 · n_target
  else:
      n_target = delta_v / ‖delta_v‖
      v_in_n  = v_ball_in · n_target
      v_out_n = v_ball_out · n_target
      v_pad_n = (v_out_n + cor · v_in_n) / (1 + cor)    # paper Eq.6 反推
      v_racket_hat = v_pad_n · n_target

═══ Step 4: swing_type 几何分类 ═════════════════════════════
yaw = yaw_from_wxyz(robot_root_quat_world)
diff_xy = p_hit_world.xy - robot_root_pos_world.xy
hit_base_xy = R(-yaw) · diff_xy                       # world → base frame

# swing_y_sign 由 expert_offset_base 自动推导（forehand_y > backhand_y → +1）
swing_y_sign = +1 if expert_offset_base[0,1] > expert_offset_base[1,1] else -1
forehand = (hit_base_xy[1] - y_mid_base) · swing_y_sign > 0
swing_type = 0 if forehand else 1

═══ Step 5: hit → base position 关系 ═══════════════════════════
offsets = expert_offset_base[swing_type]              # base frame (xy)
offsets_world = R(yaw) · offsets                       # base → world
p_base_xy_world = p_hit_world.xy - offsets_world       # base 站位
```

### 0.5 paper Eq.5 / Eq.6 推导（HITTER §IV-C 复现）

```
Eq.5 (球拍法向方向):
    n̂ = (v_out - v_in) / ‖v_out - v_in‖

Eq.6 (球拍法向速度，COR e = paddle_cor):
    v_pad,n = (v_out · n̂ + e · v_in · n̂) / (1 + e)

球拍速度（仅给法向分量，切向自由）:
    v̂_racket = v_pad,n · n̂

球台抛物线（已知击球点 + 落点 + 飞行时间，反推出射速度）:
    v_out = (p_land - p_hit) / T + (0, 0, ½·g·T)
```

注意：
- `e = 0.85` 是球-橡胶碰撞 COR，paper §IV-C 标定。`paddle_cor_range=(0.80, 0.90)` DR 加扰动模拟橡胶老化
- `flight_time` 在训练端 `U(0.30, 0.65)` 采样，在 hitter_real 端由 planner 选定
- `target_land_world` 训练用常量 `(2.45, 0, 0.78)`（对方半台中心 + 桌面 + 球半径），hitter_real 可任意
- degenerate fallback（`‖delta_v‖ < 1e-9`）极少触发（paper-strict 配置下），但留 safety net；`solve_paddle_degenerate_rate` 是 §12 monitor 之一

### 0.6 ⭐ 击球点 ↔ Base 位置关系（用户特别关注的设计点）

**问题**：球拍要击中 `p_hit_world`，但 robot 不能瞬移；base 该站到哪？

**答案**：用 expert clip 在击球瞬间的 paddle ↔ pelvis 几何，**在 base frame 下**复现。

```
expert_offset_base = {
    forehand:  (Δx_fh_base, Δy_fh_base),   # 例如 (+0.496, +0.208)
    backhand:  (Δx_bh_base, Δy_bh_base),   # 例如 (+0.428, +0.106)
}
```

这两个 offset 是**一次性预处理**（[motion_loader.py](mdp/motion_loader.py)）从 NPZ 击球瞬间拍 - pelvis 在 base frame 的相对偏移。

**计算流程**（[planner_for_training.py:262-270](mdp/planner_for_training.py#L262-L270) 与 [commands.py `_compute_base_target`](mdp/commands.py) 完全一致）：

```python
# Step A: 当前 robot yaw
yaw_robot = yaw_from_wxyz(robot_root_quat_world)        # (N,)

# Step B: 选这个 swing_type 对应的 offset（base frame）
offsets_base = expert_offset_base[swing_type]            # (N, 2) 在 base frame

# Step C: base → world 旋转
offsets_world = R(yaw_robot) @ offsets_base              # (N, 2) 在 world frame

# Step D: 已知击球点 + offset，求 base 站位
p_base_xy_world = p_hit_world.xy - offsets_world         # (N, 2) world frame
```

**为什么必须 base frame**？两条 expert clip 在击球时虽然 pelvis_yaw 都接近 0（rotated NPZ 实测 forward -5°，backward +11°，详见 §4.7），但 paddle 在 world frame 的位置相对 pelvis 有不同的偏移方向；用 base-frame Δ 提取 offset 后两者都在「身前 +x 侧」，offset 量级一致，可作为统一规则。**关键**：demo 挥拍的 yaw 旋转靠 `waist_yaw_joint` 关节角（不是 pelvis 整体旋转），所以 pelvis 始终大致朝 +X，base frame 接近 world frame 而不是绕一大角度。

**为什么必须用 robot 当前 yaw 旋回 world**？policy 启动时 base yaw 有 ±10° noise，且训练中 base 会自由旋转；不用当前 yaw 旋，offset 就和当前姿态错位。

**关键不变量**：base 站位 = 击球点 - offset。**击球点本身由 ball 物理决定**（hitter 是合成，hitter_real 是 planner 预测），base 站位是从击球点反推的从动量。policy 学的是「在合理时间内把 base 移动到这个目标 + 把 paddle 准确送到击球点」。

**新版（v58）planner 已经包含这条关系**（[planner_for_training.py:262-270](mdp/planner_for_training.py#L262-L270)），无需改动。

### 0.7 Fallback / validity 状态机

```
plan_pingpong_hits() 执行后：

planner_valid=True ─→ plan_mode = PLAN_FRESH (1)
                      所有 16 个字段都填好；caller 直接拷贝到 cmd

planner_valid=False ─→ caller (real_commands.py:259-275) 选择：
                       (a) force=True → 即便 invalid 也用（serve 模式）
                       (b) force=False → 保持上一次 cmd，写 plan_mode = PLAN_HELD (2) 或 PLAN_FROZEN (3)

fallback_reason 详细码：
  0  = ok                              (planner_valid=True)
  1  = valid_mask=False                (调用者标记不需要算)
  2  = z_out_range                     (击球 z 不在 [0.85, 1.25])
  3  = t_out_range                     (t_to_hit 太近 < 0.05 或太远 > 1.2)
  4  = ball_not_moving_to_robot        (ball_vel.x ≥ -0.05, 球远离机器人)
```

### 0.8 调用约定（Caller 必传 + 默认）

| 必传字段 | 来源 | 备注 |
|---|---|---|
| `ball_pos_world / ball_vel_world` | sim 球物理 / 真实球感知 | shape (N, 3)，world frame |
| `robot_root_pos_world / robot_root_quat_world` | `robot.data.root_pos_w / root_quat_w` | quat **wxyz** convention |
| `target_land_world` | 调用者决定（训练用 `(2.45, 0, 0.78)`） | 不能用 planner.py `solve_paddle_target` 的旧默认 `(0.7, 0, 0.06)` |
| `expert_offset_base` | `cmd.cfg.expert_offset_base`（auto-derive in `__init__`） | shape (2, 2)，行 0=forehand 行 1=backhand |
| `y_mid_base` | `cmd.cfg.y_mid_base`（auto-derive） | scalar，分界 |

> **`expert_offset_base` 和 `y_mid_base` 没有默认值** — 必须从 `PingpongCommand.cfg` 读取（[planner_for_training.py:249-259](mdp/planner_for_training.py#L249-L259) 显式 `raise ValueError`）。这避免了硬编码 fallback 在 NPZ 换数据后失配。

### 0.9 v58 兼容性确认

| v58 变更 | 影响 planner？ | 解释 |
|---|---|---|
| F1 `BLADE_NORMAL_LOCAL=(0,-1,0)` | ❌ | planner 只算 `n_target_world`（来自 ball impulse `delta_v`），不接触 paddle 局部法向 |
| F2 swing-first 撤销 | ❌ | planner Step 4 用的就是 V1 风格 base-frame 后置分类，从来不是 swing-first |
| F3 Plan B gate | ❌ | gate 是 reward 参数 |
| F4-F7 base 稳定 reward | ❌ | RewardsCfg 项 |
| F8 train.log | ❌ | 训练脚本 |
| F9 29dof 同步 | ❌ | planner 是 robot-agnostic（接 root state + expert_offset_base 由 caller 传） |

**结论：planner_for_training.py 在 v58 下完全可用，无需修改。**

### 0.10 训练 ↔ 部署 接口对账（与 deploy planner.py 的差异）

| 字段/步骤 | `planner_for_training.py` | `planner.py (deploy)` | 对接约束 |
|---|---|---|---|
| 实现 | 纯 torch batched | numpy + torch 混合 | 单元测试需保证两者输出一致到 1e-5 |
| 弹道积分 | Euler 1阶 + 二次拖曳 | RK2 + 二次拖曳 | 选 RK2 是 deploy 精度需求；训练 Euler 速度更快 |
| Bounce 检测 | z 下穿 + on_table | 同 | 同 |
| `predict_hit_plane` 等价 | Step 2 | `predict_hit_plane()` | x 平面 crossing |
| Eq.5/Eq.6 | `solve_paddle_targets_batched` | `solve_paddle_target` | 数值一致 |
| swing_type 分类 | base-frame Y_MID_BASE 阈值 | deploy heuristic `forehand if hp[1] < bp[1] else backhand`（world frame） | ⚠️ 部署若用 deploy heuristic，必须确保 expert clip 数据匹配 |
| `target_land` 默认 | 必传无默认 | `(0.7, 0, 0.06)` 旧默认 | ⚠️ 部署不可用旧默认，必须传训练 world `(2.45, 0, 0.78)` |
| `expert_offset_base` | 必传 | 部署端可硬编码（机器人固定后） | 训练值来自当前 NPZ |

---

## 1. v58 关键变更（vs v57）

| # | 变更 | 文件 | 关联 |
|---|---|---|---|
| **F1** | `BLADE_NORMAL_LOCAL = (0, 1, 0) → (0, -1, 0)` | [commands.py:38](mdp/commands.py#L38) | M4 — URDF 实测正手面是局部 -Y |
| **F2** | swing-first 采样撤销，回 V1 uniform sample + 后置分类 | [commands.py:341-419](mdp/commands.py#L341-L419) | R11 — 50/50 强制采样 + 去 critic swing 让 cos_sim 永远负值 |
| **F3** | Plan B 半 gate：`imit_joint_*: gate_pre_strike=False`，`imit_body_pos: gate_pre_strike=True` | [hitter_env_cfg.py:289-303](robots/g1_23dof/hitter/hitter_env_cfg.py#L289-L303) | R12 — body_pos 全开污染 strike 帧 paddle 朝向梯度 |
| **F4** | 加 `pelvis_ang_vel_xy = -0.05` | [hitter_env_cfg.py:350](robots/g1_23dof/hitter/hitter_env_cfg.py#L350) | R13 — 防 base 摆动倒地（仅 roll/pitch，yaw 自由） |
| **F5** | 加 `pelvis_lin_vel_z = -0.8` | [hitter_env_cfg.py:354](robots/g1_23dof/hitter/hitter_env_cfg.py#L354) | R13 — 防移动时跳跃 |
| **F6** | 加 `energy = -2e-5`（locomotion 默认） | [hitter_env_cfg.py:340](robots/g1_23dof/hitter/hitter_env_cfg.py#L340) | R13 — 节能软 reg；需在 [pingpong/mdp/__init__.py](mdp/__init__.py) 加 import |
| **F7** | `feet_slide` weight `-0.08 → -0.20` | [hitter_env_cfg.py:357](robots/g1_23dof/hitter/hitter_env_cfg.py#L357) | R13 — 防止拖脚移动 |
| **F8** | `train.py --log_redirect` 默认 OFF（旧 `--no_log_redirect` 默认 ON） | [scripts/rsl_rl/train.py:507,706](../../../../../scripts/rsl_rl/train.py#L507) | 用户：训练 log 占空间太大 |
| **F9** | 29dof 同步：把 F3-F7 全部移植到 29dof | [g1_29dof/hitter/hitter_env_cfg.py](robots/g1_29dof/hitter/hitter_env_cfg.py) | R14 — 29dof imit 信号饥饿 |

### v58.1 增量（基于 run 14-54-15 iter 2437 实测：fh_share=0.003 反手模式锁）

| # | 变更 | 文件 | 关联 |
|---|---|---|---|
| **F10** | swing-first 采样：Bernoulli(0.5) 引导 hit_y 半区，**solve 后用 `_compute_swing_type` 重分类**确保 label 与 base-frame 一致 | [commands.py:361-388](mdp/commands.py#L361-L388) | 破解反手 cheat basin；**和 V3 swing-first 关键差异**：V3 直接信任 Bernoulli sample 当 swing_type 标签，v58.1 重分类后用真值 |
| **F11** | 加 `goal_base_orientation = RewTerm(weight=0.3, std=0.3)` — base yaw 朝 +X 软锚定 | [hitter_env_cfg.py:333-339](robots/g1_23dof/hitter/hitter_env_cfg.py#L333-L339) | 配合 F10 — 强制 policy 横向移动而非侧身覆盖左右击球点；reward 公式 `exp(-yaw² / std²) × (t_to_hit > 0)` 仅 pre-strike |
| **F12** | `pelvis_lin_vel_z` -0.8 → **-1.5**（locomotion 默认） | [hitter_env_cfg.py:363](robots/g1_23dof/hitter/hitter_env_cfg.py#L363) | 用户加重抗跳 |
| **F13** | `feet_slide` -0.20 → **-0.30** | [hitter_env_cfg.py:367](robots/g1_23dof/hitter/hitter_env_cfg.py#L367) | 用户加重防拖脚 |
| F14（**未做**，记录意图） | 后续 fine-tune 阶段把 `goal_velocity` 公式改回 sharp Gaussian (`exp(-err²/std²)`) | [rewards.py:97-107](mdp/rewards.py#L97-L107) | 提精度，但 R10 forensic 表明 from-scratch 不能用；待 hsr ≥ 0.65 + shape_tier ≥ 4 后再做 |

---

## 2. Reward 总览（v58 实测 reward 配置）

### 2.1 r_i (imitation, w_i 由 imit_anneal curriculum 控制)

| sub | 公式 | env_cfg weight | 实际 split (body_dominant) | gate_pre_strike | 备注 |
|---|---|---|---|---|---|
| `imitation_joint_pos` | `exp(-2·Σⱼ(qⱼ-q̂ⱼ)²)` | `0.65 * w_i` | `0.30 * w_i`（curriculum 覆盖） | **False** ✅ | 全程跟踪关节角，post-strike 自然回 ready |
| `imitation_joint_vel` | `exp(-0.1·Σⱼ(q̇ⱼ-q̇̂ⱼ)²)` | `0.10 * w_i` | `0.10 * w_i` | **False** ✅ | 同上 |
| `imitation_body_pos` | `exp(-10·Σ_b ‖p_rel-p̂_rel‖²)` | `0.25 * w_i` | `0.60 * w_i`（curriculum 覆盖） | **True** ⚠️ | Plan B：仅 pre-strike，避免 strike 帧污染 paddle 朝向 |

`w_i_values = (0.5, 0.3, 0.15)`（V1 baseline，phase 0/1/2）。`split = "body_dominant"` → curriculum 在 [curriculums.py:416-429](mdp/curriculums.py#L416-L429) 用 `(0.30, 0.10, 0.60)` 覆盖 env_cfg 权重。

**Plan B 半 gate 推导**（R12）：
- `body_pos` 用 link 位置奖励，"位置近似就给奖"——strike 帧和 `goal_orientation` 的"朝向精确"奖励抢梯度。`body_dominant` split 0.60 让它成为最大正向 reward（实测 0.281/step），1400× 压过 `goal_orientation_pre_strike` (+0.0002)
- `joint_pos/vel` 用关节角，关节角直接决定 paddle 朝向（FK 一一对应），不会和 `goal_orientation` 冲突
- 故仅 `body_pos` gate 到 pre-strike，joint_*  保持全程开火

### 2.2 r_g (goal, w_g curriculum 控制)

| sub | 公式 | weight | std (tier 0 / tier 6) | gate |
|---|---|---|---|---|
| `goal_position` | `exp(-‖p_blade^base - p_hit^base‖²/σ²)` (base frame) | **2.0** | 0.30 → 0.06 | sparse `\|t_to_hit\| ≤ 0.10s` (curriculum 控) |
| `goal_velocity` | `exp(-‖v_blade^base - v_hat^base‖/σ)` (linear-exp, R10) | **2.0** | 0.45 → 0.20 | sparse 同上 |
| `goal_orientation` | `exp(-(1-sign·n_blade·n_target)²/σ²)` signed | **0.5** | 0.40 → 0.20 | sparse 同上 |
| `goal_position_pre_strike` | 同 pos，linear back-projection target | **1.0** | 0.20 fixed | dense `t_to_hit ∈ (0, 0.20s)` |
| `goal_velocity_pre_strike` | 同 vel，ramp target = ramp · v_hat | **1.0** | 0.60 fixed | dense `t_to_hit ∈ (0, 0.10s)` |
| `goal_orientation_pre_strike` | signed cos dist | **0.5** | 0.40 fixed | dense `t_to_hit ∈ (0, 0.20s)` |
| `goal_base` | `exp(-‖p_base_xy_world - p̂_base_xy_world‖²/σ²)` | **0.8** | 0.3 fixed | dense `t_to_hit > 0`（OFF post-strike） |

**signed cos formula 详细**（R1+M4 配套）：
```
sign = 1.0 - 2.0 * swing_type        # forehand=+1, backhand=-1
n_blade = quat_apply(blade_quat_w, BLADE_NORMAL_LOCAL)  # 现在 BLADE_NORMAL_LOCAL=(0,-1,0)
dot = sign * (n_blade · n_target)    # forehand 奖励 +正手面对齐, backhand 奖励 -正手面=+反手面对齐
```

### 2.3 r_r (regularization)

| 项 | weight | 类别 | 备注 |
|---|---|---|---|
| `alive` | **+0.04** | 存活 | 防 policy 学早结束 |
| `action_rate_l2` | **-0.001** | 动作平滑 | 击球需快速变化，弱罚 |
| `action_l2` | **-0.0005** | 动作幅度 | 同 |
| `joint_torque` | **-3e-6** | 关节力矩 L2 | |
| `joint_acc` | **-1e-7** | 关节加速度 L2 | |
| **`energy`** | **-2e-5** | 节能 | F6 新增；func 来自 `unitree_rl_lab.tasks.locomotion.mdp.rewards.energy` |
| `joint_limit` | **-5.0** | 关节限位 | hinge 越界罚 |
| `pelvis_orientation` | **-1.0** | base 倾斜 | `proj_g_xy²` |
| **`pelvis_ang_vel_xy`** | **-0.05** | base 摇晃率 | F4 新增；roll/pitch only，yaw 自由（挥拍要 yaw） |
| **`pelvis_lin_vel_z`** | **-0.8** | base 跳跃 | F5 新增；locomotion 默认 -1.5 的 0.5×（留余量给击球腾跃） |
| `pelvis_height` | **-5.0**（target=0.74 m） | base 高度 | NPZ pelvis Z 实测 0.760，但用户决定保留 0.74 |
| **`feet_slide`** | **-0.20** | 拖脚 | F7 升级（旧 -0.08）；仅 ankle_roll 接触时罚水平速度 |
| `undesired_contacts` | **-1.0** | 非足非腕非拍接触 | 软罚（脚踝/腕胶手/拍排除） |
| `paddle_table_contact` | **-10.0**（curriculum） | 拍撞桌 | R8 stage-aware 启用 |
| `body_table_contact` | **-1.0**（curriculum） | 身体撞桌 | R8 stage-aware 启用 |

---

## 3. Observation 完整规范 (Actor=86, Critic=213)

### 3.1 14 项 obs 完整表（HITTER Table I 复现）

| # | 符号 | 含义 | 维度 (23dof) | 计算 / 来源 | Actor | Critic | paper |
|---|---|---|:---:|---|:---:|:---:|:---:|
| 1 | `ω_base` | base 角速度 (base frame) | 3 | sim IMU `base_link` | ✓ (DelayedObs+IMU offset) | ✓ clean | ✓ |
| 2 | `g_base` | 重力在 base frame 投影 | 3 | `R_base^T · [0,0,-1]` | ✓ (DelayedObs+IMU offset) | ✓ clean | ✓ |
| 3 | `e_base` | base 朝向 yaw 编码 | 2 | `[cos(yaw), sin(yaw)]` | ✓ (DelayedObs+IMU offset) | ✓ clean | ✓ |
| 4 | `p̂_base − p_base` | base 位置误差 | 2 | cmd `p_base_xy_world` − robot xy | ✓ (cmd noise) | ✓ clean | ✓ |
| 5 | `p̂_racket` | 球拍目标位置 (base-relative) | 3 | `R_base^T · (cmd.p_hit_world + noise_p − p_base_world)` | ✓ (cmd noise) | ✓ clean | ✓ |
| 6 | `v̂_racket` | 球拍目标速度 (world frame) | 3 | `cmd.v_racket_hat_world (+noise_v)` | ✓ (cmd noise) | ✓ clean | ✓ |
| 7 | `t_to_hit` | 剩余击球时间 (秒) | 1 | `cmd.t_to_hit (+ noise_t)` | ✓ (cmd noise) | ✓ clean | ✓ |
| 8 | `q` | 关节位置 | **23** | sim joint encoder | ✓ (DelayedObs) | ✓ clean | ✓ |
| 9 | `q̇` | 关节速度 | **23** | sim joint encoder | ✓ (DelayedObs) | ✓ clean | ✓ |
| 10 | `a_last` | 上一 step 动作 | **23** | rollout buffer | ✓ no delay | ✓ | ✓ |
| 11 | `v_base` | base 线速度 (privileged) | 3 | sim base lin vel | – | ✓ | ✓ |
| 12 | `T_B` | 跟踪 body pos+quat | 11×7=**77** | ref body world state at clip[swing_type] frame_f (浮点插值, §11.4) **排除 right_paddle_blade** | – | ✓ | ✓ |
| 13 | `t_left` | episode 剩余时间 | 1 | `T_episode − t_now` | – | ✓ | ✓ |
| 14 | `[q̄, q̇̄]` | ref clip joint pos+vel | **46** | motion_loader at clip[swing_type] frame_f | – | ✓ | ✓ |

**Actor 总维度 = 3+3+2+2+3+3+1+23+23+23 = 86**
**Critic 总维度 = 86 + 3(v_base) + 77(T_B) + 1(t_left) + 46(qd_qdot) = 213**

### 3.2 ℬ_pos / T_B 跟踪 body 集合（11 个，统一）

| # | body name | 类别 |
|---|---|---|
| 1 | `torso_link` | 躯干 |
| 2-7 | `left_shoulder_pitch / roll / yaw / elbow / wrist_roll_rubber_hand` | 左臂 6 |
| 7-11 | `right_shoulder_pitch / roll`（注：v55 起删 yaw/elbow/wrist 给右臂自由） | 右臂 2 |

**排除**：`right_paddle_blade` (v5.5 A2 — paddle 朝向交给 r_g_ori 主导)，全下半身 (paper V-B2 ℬ ⊆ upper body)，`pelvis` (anchor 自身)。

### 3.3 imitation 关节集合 J（10 个，原版 V1 23dof）

```
waist_yaw_joint
left_shoulder_pitch / roll / yaw / elbow / wrist_roll_joint  (5)
right_shoulder_pitch / roll                                  (2)
# 排除: right_shoulder_yaw / right_elbow / right_wrist_roll
#       (paddle-orientation freedom, V1 baseline 已排除)
```

最终 |J| = 10。和 ℬ_pos 排除 `right_paddle_blade` (在 right_wrist_roll 下游) 的策略一致。

### 3.4 Actor / Critic obs 路由总表 (v5.8 DR 完整后)

| obs 项 | Actor (PolicyCfg) | Critic (CriticCfg) |
|---|---|---|
| `base_ang_vel` | `DelayedObservation(base_ang_vel_imu)` | `mdp.base_ang_vel` (clean) |
| `projected_gravity` | `DelayedObservation(projected_gravity_imu)` | `mdp.projected_gravity` (clean) |
| `base_yaw` | `DelayedObservation(base_yaw_encoding_imu)` | `mdp.base_yaw_encoding` (clean) |
| `base_err` | noisy cmd | clean cmd |
| `hit_pos` | noisy cmd | clean cmd |
| `racket_vel` | noisy cmd | clean cmd |
| `t_to_hit` | noisy cmd | clean cmd |
| `joint_pos` | `DelayedObservation(joint_pos_rel)` | `mdp.joint_pos_rel` (clean) |
| `joint_vel` | `DelayedObservation(joint_vel_rel)` | `mdp.joint_vel_rel` (clean) |
| `last_action` | `mdp.last_action` (no delay) | `mdp.last_action` |
| `base_lin_vel` | — | `mdp.base_lin_vel` |
| `ref_body_state (T_B)` | — | `mdp.pingpong_ref_body_state` |
| `time_left` | — | `mdp.episode_time_left` |
| `ref_joint_state` | — | `mdp.pingpong_ref_joint_state` |

### 3.5 强约束

- `swing_type` **不进 obs**（不论 actor 还是 critic）— 通过 `cmd.swing_type` 路由 ref clip / T_B（critic 走 ref 选择路径）
- 上游物理 `v_ball_in_world / target_land_world / flight_time / paddle_cor / n_target_world / v_ball_out_world / t_pre_initial / t_post_swing / cur_step / swing_change_remaining / cmd.noise_*` 全部不进 obs
- Actor 4 项 cmd 字段走 per-swing frozen noise（详见 §7.3）；Critic 全 clean

---

## 4. Command 完整规范

### 4.1 PingpongCommand 内部 17 字段

| # | 字段 | 维度 | frame | 来源 / 公式 | 进 obs? | paper |
|---|---|:---:|---|---|:---:|:---:|
| 1 | `swing_type` | 1 (cat) | — | F10 swing-first 引导 + `_compute_swing_type` 重分类 | ✗ (隐含 routes ref clip) | △ |
| 2 | `swing_change_remaining` | 1 (int) | — | resample 时设 1（v58.1 也设 0，保留向后兼容字段）；首次变更后置 0 | ✗ | ⚠️ |
| 3 | `p_hit_world` | 3 | world | F10 sample（见 §4.2） | ✓ obs #5 (转 base-rel) | ✓ V-B |
| 4 | `v_ball_in_world` | 3 | world | `mag∈U(2,6); yaw=π+U(±40°); pitch∈U(±75°)` | ✗ | ⚠️ R-1 |
| 5 | `target_land_world` | 3 | world | 常量 `env_origin + (2.45, 0, 0.78)` | ✗ | ⚠️ R-2 |
| 6 | `flight_time` | 1 | — | `U(0.30, 0.65)` 秒 | ✗ | ⚠️ R-3 |
| 7 | `paddle_cor` | 1 | — | `U(0.80, 0.90)` (v5.8 DR 扩展) | ✗ | ✓ IV-C |
| 8 | `v_racket_hat_world` (= v̂_racket) | 3 | world | `_solve_paddle_target` Eq.6 推导 | ✓ obs #6 | ✓ V-B + IV-C |
| 9 | `n_target_world` (= n̂_target) | 3 | world | `_solve_paddle_target` Eq.5 推导 | ✗ (reward 用) | ✓ IV-C |
| 10 | `v_ball_out_world` | 3 | world | Eq.5 副产品 | ✗ | 内部 |
| 11 | `p_base_xy_world` | 2 | world | `hit_xy - R(yaw_robot) @ expert_offset_base[swing_type]` | ✓ obs #4 | ✓ V-B |
| 12 | `t_to_hit` | 1 | — | resample 时 = `t_pre_initial`；每 step `-= dt` | ✓ obs #7 | ✓ V-B |
| 13 | `t_pre_initial` | 1 | — | `_sample_peak_uniform(0.20, 0.90, 0.30, 0.65)` 秒 | ✗ | ⚠️ |
| 14 | `t_post_swing` | 1 | — | `cfg.t_post_swing_fixed` (从 clips post_durations max 取) | ✗ | ⚠️ |
| 15 | `cur_step` | 1 | — | resample=0；每 step +1 | ✗ | 内部 |
| 16 | `noise_p / v / base / t` | 3+3+2+1 | — | per-swing freeze: `clip(gauss(0, σ_now), ±3σ)` | 仅 Actor obs 加噪用 | △ R |
| 17 | `hit_y_base` | 1 | base | `_compute_swing_type` 副产品 | ✗ (debug) | 内部 |

### 4.2 `_sample_new_swing` 完整流程（v58.1 swing-first 版）

```python
def _sample_new_swing(self, ids, reset_robot, root_pos_override=None, root_quat_override=None):
    root_pos = self.robot.data.root_pos_w[ids] if root_pos_override is None else root_pos_override
    root_quat = self.robot.data.root_quat_w[ids] if root_quat_override is None else root_quat_override
    env_origins = self._env.scene.env_origins[ids]

    # ═══ Step 1 (v58.1): swing-first 引导 hit_y 采样 ═══
    swing_target = (torch.rand(len(ids), device=self.device) < 0.5).long()  # Bernoulli(0.5)
    y_mid = float(self.cfg.y_mid_base)
    y_lo, y_hi = float(self.cfg.hit_y_range[0]), float(self.cfg.hit_y_range[1])
    if self._swing_y_sign > 0:                                              # forehand_y > backhand_y
        fh_lo, fh_hi = max(y_mid, y_lo), y_hi
        bh_lo, bh_hi = y_lo, min(y_mid, y_hi)
    else:
        fh_lo, fh_hi = y_lo, min(y_mid, y_hi)
        bh_lo, bh_hi = max(y_mid, y_lo), y_hi
    fh_lo, fh_hi = min(fh_lo, fh_hi), max(fh_lo, fh_hi)                     # 防 curriculum 退化
    bh_lo, bh_hi = min(bh_lo, bh_hi), max(bh_lo, bh_hi)
    is_forehand_target = swing_target == SWING_FOREHAND
    fh_y = sample_uniform(fh_lo, fh_hi, (len(ids),), device=self.device)
    bh_y = sample_uniform(bh_lo, bh_hi, (len(ids),), device=self.device)
    hit_y = torch.where(is_forehand_target, fh_y, bh_y)
    hit_z = sample_uniform(self.cfg.hit_z_range[0], self.cfg.hit_z_range[1], (len(ids),), device=self.device)
    local_hit = torch.stack((torch.full_like(hit_y, self.cfg.hit_x), hit_y, hit_z), dim=-1)
    self.p_hit_world[ids] = env_origins + local_hit

    # ═══ Step 2: incoming ball velocity ═══
    v_mag = sample_uniform(self.cfg.v_in_mag_range[0], self.cfg.v_in_mag_range[1], (len(ids),), device=self.device)
    v_yaw = math.pi + sample_uniform(-math.radians(40.0), math.radians(40.0), (len(ids),), device=self.device)
    v_pitch = sample_uniform(-math.radians(75.0), math.radians(75.0), (len(ids),), device=self.device)
    self.v_ball_in_world[ids] = v_mag.unsqueeze(-1) * torch.stack(
        (torch.cos(v_yaw) * torch.cos(v_pitch), torch.sin(v_yaw) * torch.cos(v_pitch), torch.sin(v_pitch)), dim=-1
    )

    # ═══ Step 3: target land + flight_time + paddle_cor ═══
    local_target = torch.tensor(self.cfg.target_land, dtype=torch.float32, device=self.device).unsqueeze(0)
    self.target_land_world[ids] = env_origins + local_target
    self.flight_time[ids] = sample_uniform(self.cfg.flight_time_range[0], self.cfg.flight_time_range[1], (len(ids),), device=self.device)
    self.paddle_cor[ids] = sample_uniform(self.cfg.paddle_cor_range[0], self.cfg.paddle_cor_range[1], (len(ids),), device=self.device)

    # ═══ Step 4: paper Eq.5/Eq.6 inline (_solve_paddle_target) ═══
    self._solve_paddle_target(ids)

    # ═══ Step 5 (v58.1): swing_type 重分类 (label 与 base-frame 一致) ═══
    new_swing, hit_y_base = self._compute_swing_type(ids, root_pos[:, :2], root_quat)
    self.swing_type[ids] = new_swing                                        # 真值，非 swing_target
    self.hit_y_base[ids] = hit_y_base
    in_dead_zone = (hit_y_base - self.cfg.y_mid_base).abs() < self.cfg.swing_dead_zone
    self._dead_zone_count[ids] += in_dead_zone.float()
    self.swing_change_remaining[ids] = 0

    # ═══ Step 6: RSI (root yaw + 关节状态) ═══
    rsi_frames: torch.Tensor | None = None
    if reset_robot and not self.cfg.disable_rsi:
        rsi_frames, pelvis_yaws = self._sample_rsi_frames(ids)
        yaw_noise = sample_uniform(self.cfg.reset_yaw_noise[0], self.cfg.reset_yaw_noise[1], (len(ids),), device=self.device)
        final_yaw = pelvis_yaws + yaw_noise                                 # clip pelvis_yaw + ±10°
        zeros = torch.zeros_like(final_yaw)
        new_root_quat = quat_from_euler_xyz(zeros, zeros, final_yaw)
        root_lin = torch.zeros(len(ids), 3, device=self.device)
        root_ang = torch.zeros(len(ids), 3, device=self.device)
        self.robot.write_root_state_to_sim(torch.cat((root_pos, new_root_quat, root_lin, root_ang), dim=-1), env_ids=ids)
        root_quat = new_root_quat                                            # 新 yaw 用于后续 base target

    # ═══ Step 7: base target + 时间字段 ═══
    self.p_base_xy_world[ids] = self._compute_base_target(ids, root_quat)
    self.t_pre_initial[ids] = _sample_peak_uniform(0.20, 0.90, 0.30, 0.65, (len(ids),), self.device)
    self.t_post_swing[ids] = float(self.cfg.t_post_swing_fixed)
    self.t_to_hit[ids] = self.t_pre_initial[ids]
    self.cur_step[ids] = 0

    if reset_robot and not self.cfg.disable_rsi:
        self._write_rsi_joint_state(ids, frames=rsi_frames)

    self._reset_window_flags(ids)
    self._freeze_noise(ids)                                                  # per-swing noise 一次冻结
    self._update_ref_state(ids)
```

### 4.3 `_solve_paddle_target` (paper Eq.5/Eq.6 inline)

```python
def _solve_paddle_target(self, ids):
    g = 9.81
    t = self.flight_time[ids].unsqueeze(-1)
    gravity_term = torch.tensor((0.0, 0.0, 0.5 * g), device=self.device).unsqueeze(0) * t
    v_out = (self.target_land_world[ids] - self.p_hit_world[ids]) / t + gravity_term
    delta_v = v_out - self.v_ball_in_world[ids]
    norm = torch.linalg.norm(delta_v, dim=-1, keepdim=True)
    degenerate = norm.squeeze(-1) < 1.0e-9
    n_target = delta_v / norm.clamp_min(1.0e-9)
    fallback_n = torch.tensor((-1.0, 0.0, 0.0), device=self.device).expand_as(n_target)
    n_target = torch.where(degenerate.unsqueeze(-1), fallback_n, n_target)
    v_in_n = torch.sum(self.v_ball_in_world[ids] * n_target, dim=-1)
    v_out_n = torch.sum(v_out * n_target, dim=-1)
    cor = self.paddle_cor[ids]
    v_pad_n = (v_out_n + cor * v_in_n) / (1.0 + cor)                         # paper Eq.6
    v_racket = v_pad_n.unsqueeze(-1) * n_target
    v_racket = torch.where(degenerate.unsqueeze(-1), 2.0 * fallback_n, v_racket)
    self.v_ball_out_world[ids] = v_out
    self.n_target_world[ids] = n_target
    self.v_racket_hat_world[ids] = v_racket
    self.last_resample_was_degenerate[ids] = degenerate
```

### 4.4 `_compute_swing_type` + `_compute_base_target`

```python
def _compute_swing_type(self, ids, root_xy, root_quat):
    yaw = yaw_from_wxyz(root_quat)
    diff = self.p_hit_world[ids, :2] - root_xy
    hit_base = _rotate_yaw_2d(diff, -yaw)                                    # world → base
    hit_y_base = hit_base[:, 1]
    is_forehand = (hit_y_base - self.cfg.y_mid_base) * self._swing_y_sign > 0
    swing = torch.where(is_forehand, SWING_FOREHAND, SWING_BACKHAND).long()
    return swing, hit_y_base

def _compute_base_target(self, ids, root_quat):
    yaw = yaw_from_wxyz(root_quat)
    offsets = self.expert_offset_base[self.swing_type[ids]]                  # (N, 2) base frame
    offsets_world = _rotate_yaw_2d(offsets, yaw)                              # base → world
    return self.p_hit_world[ids, :2] - offsets_world                          # base 站位 = hit - offset
```

### 4.5 Cmd 重采样时机

```python
def _update_command(self):
    self.t_to_hit -= self.dt
    self.cur_step += 1

    # Pre-strike 1-change lock（v5.7 设计；v58.1 swing_change_remaining=0 故此处实际不触发，保留兼容）
    pre_ids = torch.nonzero((self.t_to_hit > 0.0) & (self.swing_change_remaining > 0), as_tuple=False).flatten()
    if len(pre_ids) > 0:
        new_swing, hit_y_base = self._compute_swing_type(pre_ids, self.robot.data.root_pos_w[pre_ids, :2], self.robot.data.root_quat_w[pre_ids])
        changed = new_swing != self.swing_type[pre_ids]
        if torch.any(changed):
            ids = pre_ids[changed]
            self.swing_type[ids] = new_swing[changed]
            self.swing_change_remaining[ids] = 0
            self.hit_y_base[ids] = hit_y_base[changed]
            self.p_base_xy_world[ids] = self._compute_base_target(ids, self.robot.data.root_quat_w[ids])
            self._swing_change_used_count[ids] += 1.0

    # 重采样：post 段也走完
    done_ids = torch.nonzero(self.t_to_hit <= -self.t_post_swing, as_tuple=False).flatten()
    if len(done_ids) > 0:
        self._complete_swing(done_ids)                                       # 累计 hit_success metrics
        self._sample_new_swing(done_ids, reset_robot=False)
        self.command_counter[done_ids] += 1
        self.time_left[done_ids] = self.cfg.resampling_time_range[1]

    self._update_ref_state()
```

### 4.6 expert_offset_base / y_mid_base 一次性预处理（`__init__` 中调用 motion_loader）

```python
# 一次性预处理（motion_loader.__init__）
expert_offset_base: dict[str, np.ndarray] = {}
expert_pre_duration: dict[str, float] = {}
expert_post_duration: dict[str, float] = {}

for swing_name, npz_path in [
    ("forehand", DEFAULT_EXPERT_ROOT / "new" / "forward" / "npz" / "forward_003_rotated.npz"),
    ("backhand", DEFAULT_EXPERT_ROOT / "new" / "backward" / "npz" / "backward_001_rotated.npz"),
]:
    d = np.load(npz_path)
    imp = int(d["impact_frame"][0])
    fps = int(d["fps"][0])                                                    # =50
    pelvis_w = d["body_pos_w"][imp, 0, :2]                                    # PELVIS_IDX=0
    blade_w  = d["body_pos_w"][imp, 24, :2]                                   # BLADE_IDX=24
    pelvis_q = d["body_quat_w"][imp, 0]                                       # wxyz
    yaw      = yaw_from_wxyz(pelvis_q)
    diff_xy  = blade_w - pelvis_w                                             # world frame
    c, s     = np.cos(-yaw), np.sin(-yaw)
    expert_offset_base[swing_name] = np.array([                               # 旋到 base frame
        c * diff_xy[0] - s * diff_xy[1],
        s * diff_xy[0] + c * diff_xy[1],
    ])
    expert_pre_duration[swing_name]  = imp / fps                              # 击球前秒数
    expert_post_duration[swing_name] = (d["joint_pos"].shape[0] - 1 - imp) / fps

# y_mid_base auto-derive
if cfg.y_mid_base is None:
    cfg.y_mid_base = 0.5 * (forehand_y_eff + backhand_y)                      # ≈ +0.157 (rotated 数据)
```

### 4.7 实测 expert clip 数据（rotated NPZ）

| clip | swing_type | impact_frame | clip_len | clip pelvis_yaw 范围 | impact 时 pelvis_yaw | **Δ_base (xy, m)** | ‖v_blade‖@imp | pre/post 时长 (s) |
|---|---|:---:|:---:|---|---:|---|---:|---|
| `forward_003_rotated.npz` | 0 | 50 | 138 | **[-8.3°, -4.7°]**（漂 3.6°）| **-5.3°** | **(~+0.45, +0.18)** | ~4.2 m/s | 1.00 / 1.76 |
| `backward_001_rotated.npz` | 1 | 32 | 116 | **[+10.8°, +11.9°]**（漂 1.1°）| **+11.3°** | **(~+0.40, +0.10)** | ~2.0 m/s | 0.64 / 1.68 |

`y_mid_base = 0.5 * (forehand_y_eff + backhand_y) ≈ +0.157`
`_swing_y_sign = +1`（forehand_y > backhand_y, 标准方向）
`t_post_swing_fixed`：取两个 clips 的 max post duration ≈ 1.76 s

**关键观察（v58 Q&A 修正 + 实测复核）**：rotated NPZ 的 demo 是**"step-and-reach"风格**——pelvis 和 waist 都几乎不动，挥拍靠**右肩 pitch 大幅摆臂**（forward range 48°，backward range 35°）+ **base xy 跨步移动**完成左右覆盖。

**实测各关节 range**（forward_003 / backward_001）：
- `waist_yaw_joint`：4.6° / 2.5°（**锁住**）
- `right_shoulder_pitch`：**48.4°** / **34.7°**（主要击球动作）
- `right_shoulder_roll/yaw / elbow`：< 10°（小幅辅助）
- `right_wrist_roll`（free，不受 imit）：15.4° / 16.2°（调 paddle face）

**这就是为什么 v58.1 设计自洽**：
- `goal_base_orientation` 锁 pelvis_yaw≈0（demo: ±10° 以内）
- `imit_joint_pos` 含 `waist_yaw_joint`，自然把 waist 拉向 ~0°（demo: ≤5° range）
- `imit_joint_pos` 含 `right_shoulder_pitch`，跟踪 demo 的 35-48° 摆臂
- 右臂 distal `shoulder_yaw / elbow / wrist_roll` 全 free，policy 自由调 paddle face
- base xy 通过 `goal_base` 跟踪 `expert_offset_base` 完成横向移动

**v57 旧文档错误数据**（已废弃）：v57 §11.4 写的 "pelvis_yaw +63.6° (forehand) / +128.8° (backhand)" 是基于**旧 clip `forward_001.npz / backward_004.npz`**（未 rotated）。v58 切换到 `forward_003_rotated / backward_001_rotated` 后实测如上表，pelvis_yaw 接近 0，waist_yaw 也接近 0。

**后期 imit 降权时的预期行为**（phase 2, w_i=0.15）：
- imit_joint_pos 单帧 max ~0.0165（很弱）vs goal_position 单帧 max ~4.4（curriculum ratchet 后），相差 ~80×
- policy 可能学会**轻微 waist 旋转 (±10-20°)** 来换 paddle 朝向更精确，但不会大幅偏离 demo
- 这是健康的"用 6 DOF 精修击球"行为，符合用户期待

### 4.8 训练 / 部署 frame 红线

| 量 | 训练 world 标准值 | 说明 |
|---|---|---|
| +z | 竖直向上 | IsaacLab world up |
| +x | 从机器人指向对方半桌 | 回球方向 |
| table top | `z=0.76` | table cuboid center z=0.38, 高 0.76 |
| 击球平面 | `cfg.hit_x = 0.4` (env-local) | `p_hit_world.x = env_origin.x + 0.4` |
| `target_land` | `(2.45, 0.0, 0.78)` (env-local) | 对方半台中心 + 桌面 + 球半径 |
| robot reset | env origin 附近朝 +X，含 ±10° yaw noise | 不照抄 expert root xy/yaw |

---

## 5. Action 完整规范

### 5.1 Action 字段

| 字段 | 维度 | 类型 | 公式 |
|---|:---:|---|---|
| `JointPositionAction` | **23** (23dof) / **29** (29dof) | 关节目标位置 (offset from default) | `q_target[j] = q_default[j] + scale[j] * a[j]` |

### 5.2 Action scale (per-joint, 复用 mimic 配置)

```python
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_23DOF_PADDLE_MIMIC_ACTION_SCALE

# unitree.py:957-968 内部公式：
#   for actuator in cfg.actuators.values():
#       e = actuator.effort_limit_sim       # 每关节 effort 上限 (N·m)
#       s = actuator.stiffness              # 每关节 P-gain
#       scale[joint] = 0.25 * e[joint] / s[joint]
```

设计含义：`action_scale ≈ 25%` 最大可承受位置误差；mimic / pingpong 都需要快速大幅动作，复用同一 per-joint scale 表。

### 5.3 控制频率

- `physics dt = 1/200 s` (5ms)
- `decimation = 4`
- 最终控制频率 = **50 Hz** ✓ paper V

### 5.4 Action clip / 限位

不显式 clip action。`q_target` 在 sim 端被 joint pos limit 截断；越界由 `dof_pos_limits` reward (-5.0) 软惩罚。

---

## 6. Termination 完整规范

### 6.1 5 项 termination

| # | 信号 | 触发条件 | time_out flag | 含义 | paper |
|---|---|---|:---:|---|:---:|
| 1 | `time_out` | `t ≥ 10s` (= 500 steps @ 50Hz) | ✓ (bootstrap V) | episode 自然终止 | ✓ V-B1 |
| 2 | `bad_orientation` | `limit_angle = 0.8 rad` (≈46°)，等价 `‖proj_g_xy‖ > sin(0.8)=0.717` | ✗ | 摔倒 / 倾倒 | [我提案] |
| 3 | `base_height` | `pelvis < 0.30 m` | ✗ | 摔到地上 | [我提案] |
| 4 | `hard_contact` | `pelvis / torso_link / head_link / .*_hip_pitch_link` 触地力 > 1.0 N | ✗ | 严重摔倒 (头/胯/髋触地) | [我提案] |
| 5 | `non_paddle_table_stuck` | 非拍 body 持续撞桌 | ✗ | 撞桌（仅 R8 stage 3 启用，stage 0-2 短路返回 zeros） | [我提案] |

### 6.2 Termination 落地参数

| env_cfg term | func | params |
|---|---|---|
| `time_out` | `mdp.time_out` | `time_out=True` |
| `bad_orientation` | `mdp.bad_orientation` | `limit_angle=0.8` |
| `base_height` | `mdp.root_height_below_minimum` | `minimum_height=0.30` |
| `hard_contact` | `mdp.illegal_contact` | `threshold=1.0`, `sensor_cfg=SceneEntityCfg("contact_forces", body_names=["pelvis", "torso_link", "head_link", ".*_hip_pitch_link"])` |
| `non_paddle_table_stuck` | local `body_table_contact_sustained` (R8 stage-aware) | stage 0-2 短路 zeros，stage 3 启用 |

**强约束**：robot ↔ table 接触永远只走 reward 软罚（`paddle_table_contact / body_table_contact`），**不**作为 termination；否则一次探索碰撞就丢失击球窗口学习信号。

### 6.3 GAE bootstrap 含义

- `time_out=True` → `terminated=False` + 自然结束，GAE 用 `V(s_T)` bootstrap（避免 10s 末尾 reward 截断）
- `time_out=False` → `terminated=True`，GAE 不 bootstrap（失败状态 V≈0）

---

## 7. Events / RSI / Domain Randomization 完整规范

### 7.1 `startup` mode (一次性，paper §V-B3)

| # | 项 | 范围 | 单位 | 落地 | paper |
|---|---|---|---|---|:---:|
| 1 | `add_link_mass` | uniform `±10%` | kg | `mdp.randomize_rigid_body_mass`, `body_names=".*"`, `mass_distribution_params=(0.9, 1.1)`, `operation="scale"` | ✓ |
| 2 | `physics_material` | static_friction `[0.3, 1.6]`, dynamic `[0.3, 1.2]`, restitution `[0.0, 0.5]` | — | `mdp.randomize_rigid_body_material` | ✓ |
| 3 | `randomize_joint_friction` | `[0.5, 1.5]×` | Nm·s | `mdp.randomize_joint_parameters`, `friction_distribution_params=(0.5, 1.5)`, `operation="scale"` | ✓ |
| 4 | `randomize_joint_damping` | `[0.7, 1.3]×` | — | `mdp.randomize_actuator_gains`, `damping_distribution_params=(0.7, 1.3)`, `operation="scale"` | ✓ |
| 5 | `randomize_imu_offset` (custom) | gauss `σ=2°` | rad | local fn 写 `env._pingpong_imu_offset_quat[num_envs, 4]` (wxyz) | ✓ |
| 6 | `randomize_comm_delay` (custom) | uniform `{0, 1}` step (= 0–20 ms @ 50Hz) | step | local fn 写 `env._pingpong_obs_delay_steps[num_envs]` | ✓ |

DR 5/6 是 **startup 一次冻结**（不是 per-step / interval）。理由：IMU 标定误差和通信链路在硬件上是装机后基本恒定的偏移；per-step 抖动会让 Actor 学不到稳定因果关系。复现方需要 per-step 抖动时把 EventTerm `mode="startup"` 改成 `"interval"` 即可，wrapper 不动。

### 7.2 `reset` mode (每 episode, RSI + 首组 cmd)

| # | 项 | 范围 | 备注 |
|---|---|---|---|
| 1 | `reset_robot_pose` (RSI) | 见 §7.4 | 必从 ref clip 内随机帧起步 (v5.5 A7: 删 mimic_start_prob 分支) |
| 2 | `reset_joint_pos_noise` | gauss `σ=0.05` rad | 23 dof 独立 |
| 3 | `reset_base_yaw_noise` | uniform `±10°` (`reset_yaw_noise=(-pi/18, pi/18)`) | 防 yaw 锁死 |
| 4 | `sample_first_cmd` | §7.4 顺序 + §4.2 swing-first | reset nominal root 后生成首组 clean cmd；noise 在 ref/RSI consistency 确认后冻结 |

### 7.3 `interval` mode + cmd noise per-swing freeze (v5.5 A10)

#### 7.3.1 真正周期性 (仅 push)

| # | 项 | 间隔 | 公式 | 备注 |
|---|---|---|---|---|
| 1 | `push_robot_velocity` | 每 5–10s | base lin vel `±0.3 m/s`, ang vel `±0.2 rad/s` | 鲁棒性扰动，与 cmd 无关 |

#### 7.3.2 cmd noise (一次冻结 per swing) — 关键约定

**采样时机**：每次 cmd 重采样 (= `t_to_hit ≤ -t_post_swing` 触发 + episode reset 的首组 cmd) 时**一次性**采出 4 路噪声向量，存到 cmd 内部字段（`cmd.noise_p / noise_v / noise_base / noise_t`），整个 swing 持续期间**不变**，直到下次重采才换新。

**注入位置**：仅 Actor obs（asymmetric AC）。Critic obs / r_g / r_g_base / strike window gate 全部用真值 cmd（无噪声），这样 value head + reward signal 不被噪声扰动，训练信号干净。

| # | 噪声项 | 终值 σ | clip 范围 (±3σ) | 触发 (击球成功率) | 注入位置 |
|---|---|---|---|---|---|
| 1 | `cmd.noise_t` (scalar) | gauss `σ=0.005 s` | ±0.015 s | **≥ 50%** (最早开) | Actor obs `t_to_hit` |
| 2 | `cmd.noise_p` (xyz, 独立) | gauss `σ=0.005 m` | ±0.015 m | ≥ 75% | Actor obs `p̂_racket` |
| 3 | `cmd.noise_v` (xyz, 独立) | gauss `σ=0.05 m/s` | ±0.15 m/s | ≥ 75% | Actor obs `v̂_racket` |
| 4 | `cmd.noise_base` (xy, 独立) | gauss `σ=0.015 m` | ±0.045 m | ≥ 75% | Actor obs `p̂_base − p_base` |

**调度形式**：触发后 1k iter linear 从 0 → 终值。触发后**单调不退**。

**实现伪代码**：

```python
# (a) curriculum 每 iter 更新 σ_*_now (训练循环外层):
σ_p_now    = curriculum.update_cmd_noise_sigma_p()
σ_v_now    = curriculum.update_cmd_noise_sigma_v()
σ_base_now = curriculum.update_cmd_noise_sigma_base()
σ_t_now    = curriculum.update_cmd_noise_sigma_t()

# (b) swing 重采样那一 step (commands.py.resample_cmd):
def resample_cmd(cmd, σ_p, σ_v, σ_base, σ_t):
    # 先做 §4.2 swing-first 七步生成 clean cmd，然后一次性冻结 noise:
    cmd.noise_p    = clip(gauss(0, σ_p,    size=3), -3*σ_p,    3*σ_p)
    cmd.noise_v    = clip(gauss(0, σ_v,    size=3), -3*σ_v,    3*σ_v)
    cmd.noise_base = clip(gauss(0, σ_base, size=2), -3*σ_base, 3*σ_base)
    cmd.noise_t    = clip(gauss(0, σ_t,    size=1), -3*σ_t,    3*σ_t)

# (c) 每 step Actor obs 端用冻结的 noise 直接加 (不再每 step 重新 gauss):
obs_actor.p̂_racket = R_base^T · ((cmd.p_hit_world + cmd.noise_p) − p_base_world)
obs_actor.v̂_racket = cmd.v_racket_hat_world + cmd.noise_v
obs_actor.base_err = (cmd.p_base_xy_world + cmd.noise_base) − p_base_xy_world
obs_actor.t_to_hit = cmd.t_to_hit + cmd.noise_t

# (d) Critic obs / r_g / strike_window_gate 全部用 cmd.* 真值:
obs_critic.p̂_racket = R_base^T · (cmd.p_hit_world − p_base_world)
r_g_pos = exp(-‖p_blade^base − cmd.p_hit^base‖² / σ²) · 𝟙[abs(cmd.t_to_hit)<=strike_window]
r_g_base = exp(-‖p_base_xy_world − cmd.p_base_xy_world‖² / σ²)
```

#### 7.3.3 reward / gate 端真值约定（**强约束**）

实现端**严禁**让 reward 函数读 `cmd.noise_*`：
- `r_g_pos / r_g_vel / r_g_ori` 用 clean `cmd.p_hit_world / cmd.v_racket_hat_world / cmd.n_target_world`
- `r_g_base / r_g_base_orientation` 用 clean
- strike window gate `|cmd.t_to_hit| ≤ strike_window` 用 clean
- 重采样边界 `cmd.t_to_hit ≤ -cmd.t_post_swing` 用 clean
- Critic obs 全 clean

### 7.4 RSI 完整流程 (v5.7 锁定)

```python
# Step 0: reset root 到训练 nominal pose (不照抄 expert world root xy/yaw)
root_xy_world  = env.scene.env_origins[ids, :2]                              # env origin xy
root_z_world   = 0.74                                                         # pelvis 标准高度
root_yaw_pre   = uniform(-pi/18, pi/18)                                       # ±10° noise (placeholder)
robot.write_root_pose(root_xy_world, root_z_world, root_yaw_pre)

# Step 1: generate clean cmd (§4.2 swing-first; 此时 yaw 还未对齐 clip)
cmd = sample_clean_cmd(root_pose=robot.root_pose)

# Step 2: choose ref clip from cmd.swing_type (post-classify 后)
ref_clip = expert_clip[cmd.swing_type]

# Step 3: 用 clip 采样帧的 pelvis_yaw 覆写 root yaw
rsi_frames, pelvis_yaws = self._sample_rsi_frames(ids)                        # clip 内随机帧
final_yaw = pelvis_yaws + uniform(-pi/18, pi/18)                              # clip yaw + ±10°
new_root_quat = quat_from_euler_xyz(0, 0, final_yaw)
robot.write_root_state_to_sim(root_pos + new_root_quat + zeros, ids)          # 覆写 quat

# Step 4: 关节状态从 clip[f_init] 拷贝
robot.write_joint_state(ref_clip[rsi_frames].joint_pos, ref_clip[rsi_frames].joint_vel)

# Step 5: ref 时间从 clip 头开始
cmd.cur_step = 0

# Step 6: 重新计算依赖 root_quat 的 cmd 字段 (p_base_xy_world)
cmd.p_base_xy_world = self._compute_base_target(ids, new_root_quat)

# Step 7: freeze per-swing command noise
freeze_cmd_noise(cmd)
```

**关键不变量**：物理姿态 = ref 采样帧；ref 进度 = 0（从 clip 头开始）；root yaw = clip pelvis_yaw + noise。这样保证 `n_blade_world` 在 reset 瞬间就和 `n_target_world` 大致对齐（M1 修复）。

### 7.5 IMU offset / comm delay 实现要点

```python
# pingpong/mdp/events.py
def randomize_imu_offset(env, env_ids, asset_cfg, sigma_deg=2.0):
    rx, ry, rz = gauss(0, deg2rad(sigma_deg), 3)
    q = quat_from_euler_xyz(rx, ry, rz)
    env._pingpong_imu_offset_quat[env_ids] = q                                # buffer

def randomize_comm_delay(env, env_ids, max_delay_steps=1):
    env._pingpong_obs_delay_steps[env_ids] = uniform(0, max_delay_steps + 1)  # int 0/1

# pingpong/mdp/observations.py: Actor wrapper
class DelayedObservation(ManagerTermBase):
    """Per-env one-step delay using env._pingpong_obs_delay_steps buffer.
    First step buffer initialized to current obs (delay 0)；env reset 清零下次自动重填。"""
    def __call__(self, env, ...):
        cur = inner_func(env, ...)
        delayed = where(delay > 0, prev_buffer, cur)
        prev_buffer = cur
        return delayed
```

Actor wrap 项：`base_ang_vel_imu / projected_gravity_imu / base_yaw_encoding_imu / joint_pos_rel / joint_vel_rel`。
Critic 全部走 clean `mdp.*`（不 wrap）。`last_action` 不 wrap（无通信链路）。

---

## 8. Curriculum 完整规范 (v5.1 + R7 + R8 + R10)

### 8.0 σ 类型说明（防混淆）

| σ 类型 | 含义 | 出现位置 | 单调方向 |
|---|---|---|---|
| **Reward kernel σ** | `exp(-d²/σ²)` 衰减半径 | r_g_pos / vel / ori / base 公式参数 | 只**收紧**（升档=任务变难）|
| **采样标准差 σ_sampling** | `gauss(0, σ²)` 噪声幅度 | cmd noise σ_p/v/base/t | 只**升**（更鲁棒）|
| **uniform 区间** | sample 上下界 | hit_y/z range, v_in_mag range | 只**扩展**（更难）|

### 8.1 σ_g_pos curriculum (击球成功率驱动 monotone tighten)

| 阶段 | σ_g_pos | 触发 (击球成功率, 1k iter sliding window) |
|---|---|---|
| 初始 | 0.10 m | iter 0 |
| stage 1 | linear `0.10 → 0.06` | hit_success_rate ≥ 30% |
| stage 2 | linear `0.06 → 0.04` | ≥ 50% |
| stage 3 | linear `0.04 → 0.03` | ≥ 65% |
| stage 4 | linear `0.03 → 0.02` | ≥ 80% |
| 锁定 | 0.02 m (= 噪声极限 4×σ_p) | 达成后单调不退 |

### 8.2 hit_y / hit_z / v_in_mag range curriculum (扩展)

| 参数 | 初始 (≥0%) | ≥30% | ≥50% | ≥75% (终值) |
|---|---|---|---|---|
| `hit_y` | `±0.5` m | linear → `±0.7` | → `±0.85` | → **`±1.0`** m |
| `hit_z` | `[0.20, 0.45]` | → `[0.15, 0.50]` | → `[0.12, 0.55]` | → **`[0.08, 0.60]`** |
| `v_in_mag` | `[2.5, 4.5]` m/s | → `[2.2, 5.0]` | → `[2.0, 5.5]` | → **`[2.0, 6.0]`** m/s |

### 8.3 cmd noise σ curriculum

详见 §7.3.2 表格（σ_t / σ_p / σ_v / σ_base 触发与终值）。

### 8.4 imit_anneal (R5 metric 模式)

```python
imit_anneal = CurrTerm(
    func=mdp.update_imitation_weight,
    params={
        "schedule": "metric",                                                # 不是 iter
        "command_name": "pingpong",
        "num_steps_per_env": 24,
        "w_i_values": (0.5, 0.3, 0.15),                                       # phase 0/1/2
        "split": "body_dominant",                                             # jp:0.30, jv:0.10, bp:0.60
        "min_ep_length_for_phase_advance": 250,                               # EL_ema<250 强制 phase=0
        "phase_thresholds": [
            {"hsr_ema": 0.30, "cos_sim_ema": 0.45, "ep_length_ema": 250},     # phase 0→1
            {"hsr_ema": 0.65, "cos_sim_ema": 0.55, "ep_length_ema": 400},     # phase 1→2
        ],
    },
)
```

curriculum 每 iter 调用 `update_imitation_weight()` 重写三个 imit term 的 weight：
```python
weight[imitation_joint_pos] = split[0] * w_i_values[phase]                    # = 0.30 * w_i
weight[imitation_joint_vel] = split[1] * w_i_values[phase]                    # = 0.10 * w_i
weight[imitation_body_pos]  = split[2] * w_i_values[phase]                    # = 0.60 * w_i
```

env_cfg 内 `weight=0.65 * w_i / 0.10 * w_i / 0.25 * w_i` 是 dead 值（curriculum 第一 tick 后被覆盖），保留为可读性。

### 8.5 Window curriculum (strike_window 收紧 + r_g 权重 ratchet)

R10 的 7 档 ladder：

```
shape_tier 0 → 6:
  std_pos: 0.30 → 0.06    (七档: 0.30/0.24/0.18/0.13/0.10/0.08/0.06)
  std_vel: 0.45 → 0.20    (七档: 0.38/0.35/0.32/0.29/0.26/0.23/0.20)
  std_ori: 0.40 → 0.20    (七档: 0.40/0.36/0.32/0.28/0.25/0.22/0.20)
  strike_window: 0.10 → 0.01 s (七档: 0.10/0.08/0.06/0.04/0.03/0.02/0.01)
  weight ratchet: pos 2 → 12, vel 2 → 12, ori 0.5 → 4

每档触发: hit_success_rate ≥ {0.10, 0.20, 0.40, 0.55, 0.65, 0.75}
单调: 只升不退；但有 cos_sim_collapse_retreat (R7) 反向逻辑
```

### 8.6 Sequenced curriculum (R7)

```
Stage 1 (window-only): 启用 window curriculum 推进 shape_tier
  锁定:   v_in_mag_range = (2, 2)             (固定低速)
          hit_y_range = (y_mid_base ± 0.10)   (窄)

Stage 2 (v_in unlock): 触发 shape_tier ≥ 6 AND hsr_ema ≥ 0.85 AND cos_sim_ema ≥ 0.55
  解锁:   v_in_mag_range 扩展到 (2, 3.5)

Stage 3 (hit_y unlock): 触发 v_in_high ≥ 3.5 AND hsr_ema ≥ 0.80
  解锁:   hit_y_range 扩展到 cap (0.5, 1.0) 等
```

env_cfg 暴露 9 个开关：
```python
"sequenced_curriculum": True,
"v_unlock_shape_tier": 6,                                                    # paper-strict 顶档
"v_unlock_hsr": 0.85,
"v_unlock_cos_sim": 0.55,
"v_in_high_max": 6.0,
"y_unlock_v_in_high": 3.5,
"y_unlock_hsr": 0.80,
"hit_y_max_cap": 1.0,
"hit_y_min_cap": -1.0,
```

### 8.7 cos_sim collapse retreat (R7 反向 ratchet)

```
if cos_sim_ema < 0.35:
    v_in_high = max(2.5, v_in_high - 0.05)                                    # 退档
    hit_y_range 退到 (y_mid_base ± 0.10)                                      # 收回
    cos_sim_collapsed flag = True

注: shape_tier 自身不退档 (monotone-only ratchet)；仅 v_in / hit_y 退。
```

### 8.8 R8 table_guard 4 阶段

```
Stage 0  hidden     桌子 z=-10，weight=0，flag=False
Stage 1  unlocked   解锁条件 (AND, 全 batch 均值):
                    hsr_ema ≥ 0.65, cos_sim_ema ≥ 0.50, EL_ema ≥ 400, iter ≥ 1500
                    翻 flag=True；不主动 teleport，靠每 env reset 时 EventTerm 自然搬桌子
                    保持 ramp_iters/4 iter 静默期 (让所有 env reset 完)
Stage 2  ramping    weight 0→target 线性 ramp 500 iter
                    paddle_table_contact: 0 → -10
                    body_table_contact:   0 → -1
Stage 3  active     non_paddle_table_stuck termination 启用
```

### 8.9 cos_sim_ratchet_freeze (M1 配套, 跨 tier 朝向准入)

```
if cos_sim_ema < cos_sim_freeze_threshold (=0.45):
    cos_sim_ratchet_freeze = True
    禁止 shape_tier 升档（即便 hit_success_rate 达标）
else:
    cos_sim_ratchet_freeze = False
    shape_tier 可升

理由: shape_tier 升档收紧 std 后, 若 paddle 朝向不对, sigma 变小直接零信号; 
      所以拍面对齐没起来前不允许更严标准。
```

### 8.10 POS_VEL_GATE_LATCH / ORI_GATE_LATCH (S1+S2 stand-up gate)

```python
# 单调 latch (一开就不关):
_POS_VEL_GATE_LATCH = {"opened": False, "min_ep_length": 250, "original_weights": {}}
_ORI_GATE_LATCH    = {"opened": False, "min_ep_length": 250}

# 每 iter 检查:
if EL_ema >= min_ep_length AND not opened:
    opened = True
    恢复 env_cfg 原始权重: goal_position, goal_velocity, goal_*_pre_strike (POS_VEL gate)
                         goal_orientation (ORI gate)

# opened=False 时强制 weight=0 (即便 env_cfg 写了非零值)
```

理由：站不稳时（EL_ema<250）不允许学击球（避免 R2 swing-while-falling basin）。

### 8.11 Curriculum 整合视图 (单表)

| # | 名称 | σ 类型 | 起 → 终 | 触发 metric | 单调方向 | 实现 (curriculums.py) |
|---|---|---|---|---|---|---|
| 1 | σ_g_pos | reward kernel σ | 0.10 → 0.02 m | hit_success (30/50/65/80%) | 收紧 | `update_g_pos_sigma` |
| 2 | σ_t (cmd 时间) | sampling σ | 0 → 0.005 s | ≥ 50% | 升 | `update_cmd_noise_sigma_t` |
| 3 | σ_p (cmd 位置) | sampling σ | 0 → 0.005 m | ≥ 75% | 升 | `update_cmd_noise_sigma_p` |
| 4 | σ_v (cmd 速度) | sampling σ | 0 → 0.05 m/s | ≥ 75% | 升 | `update_cmd_noise_sigma_v` |
| 5 | σ_base (cmd 站位) | sampling σ | 0 → 0.015 m | ≥ 75% | 升 | `update_cmd_noise_sigma_base` |
| 6 | hit_y range | uniform 区间 | ±0.5 → ±1.0 m | hit_success (30/50/75%) | 扩展 | `update_hit_y_range` |
| 7 | hit_z range | uniform 区间 | `[0.20,0.45] → [0.08,0.60]` | hit_success (30/50/75%) | 扩展 | `update_hit_z_range` |
| 8 | v_in_mag range | uniform 区间 | `[2.5,4.5] → [2.0,6.0]` m/s | hit_success (30/50/75%) | 扩展 | `update_v_in_mag_range` |
| 9 | imit_anneal | w_i 三段 | (0.5, 0.3, 0.15) | hsr/cos_sim/EL EMA 三联 | 降（学好后降）| `update_imitation_weight` |
| 10 | window curriculum | shape_tier ladder | 0 → 6 | hit_success_rate | 收紧（升档）| `update_window_curriculum` |
| 11 | sequenced curriculum | stage 1→3 | — | shape_tier+hsr+cos_sim | 单调 | `update_sequenced_curriculum` |
| 12 | cos_sim collapse retreat | v_in/hit_y 退档 | — | cos_sim_ema < 0.35 | 反向 | `update_cos_sim_retreat` |
| 13 | cos_sim_ratchet_freeze | shape_tier 准入 | — | cos_sim_ema < 0.45 | 锁定 | `update_cos_sim_freeze` |
| 14 | POS_VEL_GATE_LATCH | 击球 weight 0/原值 | — | EL_ema ≥ 250 | latch | `update_pos_vel_gate` |
| 15 | ORI_GATE_LATCH | goal_orientation weight | — | EL_ema ≥ 250 | latch | `update_ori_gate` |
| 16 | table_guard | 4 阶段 | hidden→active | hsr/cos_sim/EL+iter | 单调 | `update_table_guard_stage` |

### 8.12 Curriculum order 强约束 (S3 修复)

[hitter_env_cfg.py CurriculumCfg](robots/g1_23dof/hitter/hitter_env_cfg.py)：**`imit_anneal` 必须在 `pingpong` 之前**。理由：imit_anneal 写 `_EP_LENGTH_EMA`，pingpong 读它做 window-advance gate；顺序反了产生 1-tick 滞后。

---

## 9. Scene 完整规范

### 9.1 InteractiveSceneCfg

| # | 项 | 取值 / 配置 | 备注 |
|---|---|---|---|
| 1 | `num_envs` | 4096 (训练) / 1 (play) | IsaacLab 标准 |
| 2 | `env_spacing` | 4.0 m | env 间隔 |
| 3 | `terrain` | `TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, friction_combine_mode="multiply", restitution_combine_mode="multiply"), visual_material=MdlFileCfg(...))` | ✓ paper V (室内) |
| 4 | `robot` | 23dof: `g1_23dof_rev_1_0_paddle.urdf` / 29dof: `g1_29dof_rev_1_0_paddle.urdf` | 见 §9.3 |
| 5 | `table` | `RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Table", spawn=CuboidCfg(size=(2.74, 1.525, 0.05), rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True), collision_props=CollisionPropertiesCfg(), physics_material=RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.8, restitution=0.2)), init_state=InitialStateCfg(pos=(1.77, 0.0, **-10.0**)))` | R8 stage 0/1: pos.z=-10 (地下)；stage 2+: teleport to z=-0.005 (桌面 top 在 ~0.76 m) |
| 6 | `light` | `AssetBaseCfg(prim_path="/World/light", spawn=DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0))` | IsaacLab 标准 |
| 7 | `sky_light` | `AssetBaseCfg(prim_path="/World/skyLight", spawn=DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0))` | 同 |
| 8 | `contact_forces` | `ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0)` | sensor #1: 服务 feet_slide / undesired_contacts / hard_contact termination |
| 9 | `robot_table_contact` | `ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"], history_length=3)` | sensor #2: filtered，仅 robot vs Table，服务 paddle_table_contact / body_table_contact / non_paddle_table_stuck |

### 9.2 ContactSensor 用法（双 sensor 各司其职）

**sensor #1 `contact_forces`**（覆盖全 robot body，用 `body_names` 正则过滤）：

| 用途 | body_names regex |
|---|---|
| `feet_slide` / `feet_air_time` | `[".*ankle_roll_link"]` |
| `undesired_contacts` (软罚) | regex 排除 `[ankle, wrist_roll_rubber_hand (23dof) / rubber_hand (29dof), paddle_blade]` 后的所有 body |
| `hard_contact` (终止) | `["pelvis", "torso_link", "head_link", ".*_hip_pitch_link"]` |

**sensor #2 `robot_table_contact`**（filter 到 Table，所有 robot body 与 Table 的接触）：

| 用途 | body_names |
|---|---|
| `paddle_table_contact` (R8) | `["right_paddle_blade"]` |
| `body_table_contact` (R8) | 排除 paddle 后所有 body |
| `non_paddle_table_stuck` (终止, R8 stage 3) | 排除 paddle |

### 9.3 Robot asset

| 项 | 23dof | 29dof |
|---|---|---|
| URDF | `g1_23dof_rev_1_0_paddle.urdf` | `g1_29dof_rev_1_0_paddle.urdf` |
| Asset cfg | `UNITREE_G1_23DOF_PADDLE_MIMIC_CFG` | `UNITREE_G1_29DOF_PADDLE_MIMIC_CFG` |
| Action scale | `UNITREE_G1_23DOF_PADDLE_MIMIC_ACTION_SCALE` | `UNITREE_G1_29DOF_PADDLE_MIMIC_ACTION_SCALE` |
| init pos | `(0, 0, 0.76)` | 同 |
| init quat | `(1, 0, 0, 0)` wxyz | 同 |
| Joint init | clip frame 0 stance (override 自 NPZ, [unitree.py:989-1019](../../assets/robots/unitree.py#L989-L1019)) | 同 |
| Body 总数 | 25（含 pelvis + paddle blade） | 33（额外 8 个中间 link：waist_yaw/roll, wrist_roll/pitch/yaw ×2） |
| `enabled_self_collisions` | True | **False** ⚠️ (29dof 自碰撞会 step 0 触发 hard_contact，造成 EpLen=1 collapse) |
| paddle fixed joint rpy | `-2.3561944902, 0, 0` (= -3π/4 = -135° 绕 X) | 同 |

⚠️ paddle fixed joint rpy 之前有文档误写为 `-π/4` (-45°)，**实际是 -135°**，已嵌入 `body_quat_w` 由 `quat_apply` 自动吸收，但和 `BLADE_NORMAL_LOCAL = (0, -1, 0)` 配合才得到正手面（M4）。

### 9.4 训练 world frame 红线

| 量 | 训练 world 标准 | 说明 |
|---|---|---|
| +z | 竖直向上 | IsaacLab world up |
| +x | 从机器人指向对方半桌 | 回球方向 |
| table top | `z=0.76` | (table cuboid 高度 0.05，center.z 在 stage 2+ 取 -0.005，等效 top 在 robot world 标准 z 系) |
| 击球平面 | `cfg.hit_x = 0.4` (env-local) | `p_hit_world.x = env_origin.x + 0.4` |
| `target_land` | `(2.45, 0.0, 0.78)` (env-local) | 对方半台中心 + 桌面 + 球半径 |
| robot reset | env origin 附近朝 +X，含 ±10° yaw noise | 不照抄 expert root xy/yaw |

---

## 10. Sim 完整规范

| # | 项 | 取值 | paper |
|---|---|---|:---:|
| 1 | `sim.dt` | `1/200 s` (5ms) | [我提案] (G1 标准) |
| 2 | `decimation` | **4** | ✓ paper V "50Hz" |
| 3 | `episode_length_s` | **10.0 s** = 500 steps | ✓ paper V-B1 |
| 4 | `gravity` | `(0, 0, -9.81)` | ✓ |
| 5 | `solver_position_iter` | 4 | PhysX 默认 |
| 6 | `solver_velocity_iter` | 1 | PhysX 默认 |
| 7 | `friction_combine_mode` | `"multiply"` | 通用 |
| 8 | `restitution_combine_mode` | `"multiply"` | 通用 |

---

## 11. 关键 convention 与实测数据

### 11.1 Quaternion: `wxyz` 全工程统一

任何 isaaclab / asset 边界返回 `xyzw` 必须显式转换。

### 11.2 BLADE_NORMAL_LOCAL: `(0, -1, 0)` (M4)

```python
BLADE_NORMAL_LOCAL = (0.0, -1.0, 0.0)
```

URDF `g1_23dof_rev_1_0_paddle.urdf` 的 fixed joint rpy=`-2.356 0 0` (-135° around X) 把腕部 +Y 旋到 paddle 局部 "身后偏下" 方向。NPZ 实测 forward_003 impact 帧：
- 局部 +Y 在世界 = (-0.652, -0.719, -0.241) 朝身后下
- **局部 -Y 在世界 = (+0.652, +0.719, +0.241) 朝球台 ← 这是正手面**

故 `BLADE_NORMAL_LOCAL=(0,-1,0)` 指向正手面。signed reward 公式：
- forehand sign=+1: reward `+正手面 · n_target`
- backhand sign=-1: reward `-正手面 · n_target` = `+反手面 · n_target`

### 11.3 swing_type: integer 编码

```python
SWING_FOREHAND = 0
SWING_BACKHAND = 1
```

`y_mid_base = 0.5 * (forehand_y_eff + backhand_y) ≈ +0.14 ~ +0.16`（auto-derived from clips at `__init__`）；`_swing_y_sign` 由 forehand_y 与 backhand_y 的相对位置自动推导（forehand_y > backhand_y → sign=+1，反之 -1）。

### 11.4 RSI 三步约束

1. reset root xy/yaw 到训练 nominal pose（**不**复制 expert world root），加 ±10° yaw noise
2. 用 `clip[swing_type]` 的 sampled frame 写关节角 + 关节速度
3. 用 clip pelvis_yaw 覆写 root_quat（保证关节角和 base 朝向一致，否则 n_blade 整体被旋转 60°+）

### 11.5 Reward gate truth-value contract

实现端**严禁**让 reward 函数读 `cmd.noise_*`：
- `r_g_pos / r_g_vel / r_g_ori` 用 clean `cmd.p_hit_world / cmd.v_racket_hat_world / cmd.n_target_world`
- `r_g_base` 用 clean `cmd.p_base_xy_world`
- strike window gate `|cmd.t_to_hit| ≤ strike_window` 用 clean `cmd.t_to_hit`
- 重采样边界 `cmd.t_to_hit ≤ -cmd.t_post_swing` 用 clean
- Critic obs 全 clean
- Actor obs 仅 4 项 cmd 字段加 noise

### 11.6 Tensorboard 权威性

判读 hit_success / cos_sim / fail rate **一律用 `Curriculum/pingpong/*` namespace**（step-level batch 均值）。`Metrics/pingpong/*` 仅在 episode 结束时累计，早期 time_out 占比高时会假性显示 0。

---

## 12. PPO 配置（复用 mimic `BasePPORunnerCfg`）

复用 `tasks/mimic/agents/rsl_rl_ppo_cfg.py:BasePPORunnerCfg`，只改 task entry / experiment_name。

| 字段 | 取值 |
|---|---|
| num_steps_per_env | 24 |
| max_iterations | 180000 |
| save_interval | 1000 |
| empirical_normalization | False |
| init_noise_std | 1.0 |
| MLP | [512, 256, 128] |
| activation | elu |
| value_loss_coef | 1.0 |
| use_clipped_value_loss | True |
| clip_param | 0.2 |
| entropy_coef | 0.005 |
| num_learning_epochs | 5 |
| num_mini_batches | 4 |
| learning_rate | 5e-4 |
| schedule | adaptive |
| gamma | 0.99 |
| lam | 0.95 |
| desired_kl | 0.01 |
| max_grad_norm | 1.0 |

---

## 13. 启动训练

### 23dof V1 v58 from-scratch

```bash
python scripts/rsl_rl/train.py \
  --task Unitree-G1-23dof-Pingpong-HITTER \
  --headless \
  --run_name v1_planB_paddleface_fixed
```

注：
- 不加 `--log_redirect` → 不生成 train.log（F8 默认 OFF），TB events 仍正常输出
- 任务入口 `Unitree-G1-23dof-Pingpong-HITTER` 在 [robots/g1_23dof/__init__.py](robots/g1_23dof/__init__.py) 注册

### 29dof v58 from-scratch

```bash
python scripts/rsl_rl/train.py \
  --task Unitree-G1-29dof-Pingpong-HITTER \
  --headless \
  --run_name v1_planB_29dof_v2
```

29dof 配置已完全同步 23dof v58（F1-F7 全部移植 + R14 imit gate 修正），但保留 29dof 特有的：
- IMITATION_JOINT_NAMES_29DOF: 12 个（V1 paddle-freedom 模式：右臂 distal-to-shoulder_roll 砍掉 5 个）
- TRACKED_BODY_NAMES_29DOF: 10 个（同步右臂 5 个 body 砍掉）
- B7 forensic: `w_i_values=(2.0, 1.0, 0.5)` 的 4× boost + `pre_strike weights=0.0`
- 29dof URDF: `g1_29dof_rev_1_0_paddle.urdf` + `enabled_self_collisions=False`（避免 8 个 extra link 的自碰撞）

---

## 14. 验证清单（新 baseline）

### 14.1 启动 sanity（前 200 iter）

| 指标 | 期望 | 不达标含义 |
|---|---|---|
| AST 通过 | OK | 配置文件语法错误 |
| 没有 `BLADE_NORMAL_LOCAL` 旧值残留 | grep `(0.0, 1.0, 0.0)` 空 | F1 没生效 |
| `Episode_Reward/imitation_body_pos` 在 pre-strike 帧 > 0 | 是 | gate 配置错 |
| `Episode_Reward/energy` 出现 | 是 | F6 import 没加 |
| `Episode_Reward/pelvis_ang_vel_xy` 量级 -0.005 ~ -0.02 | 是 | 权重过大或过小 |

### 14.2 早期训练（iter 200-2000）

| iter | 23dof v58 期望（比 V1 21-04-08 显著好） |
|---|---|
| 200 | EL ≥ 30, cos_sim ≥ 0（V1 历史: 17, -0.4） |
| 500 | EL ≥ 200, cos_sim ≥ 0（V1 历史: 47, -0.4） |
| 1000 | EL ≥ 400, cos_sim ≥ +0.4, hsr 出现首批非零（V1 历史: 47, -0.4, 0） |
| 2000 | EL ≥ 480, cos_sim ≥ +0.5（V1 历史: 52, -0.6, 0） |

实测 run 2026-05-29_14-54-15 iter 1000：EL=462, cos_sim=+0.585, hsr=0.004 ✅

### 14.3 中后期（iter 5000-15000）

期望（v58）：
- iter 5000: EL ≥ 480 plateau, hsr ≥ 0.30, cos_sim_ema ≥ +0.50, shape_tier ≥ 1
- iter 8000: hsr ≥ 0.60, cos_sim_ema ≥ +0.55, shape_tier ≥ 2
- iter 15000: hsr ≥ 0.70 不退步（V1 历史在此段开始衰退至 0.32，因为旧 cos_sim 是反的）

### 14.4 视频 ground truth（最强信号）

播放 `model_8000.pt`：
- ✅ paddle 朝向自然，**不再扭手腕 180°**（V1 旧 model 的 rigid 姿态来自 cos_sim 反 convention）
- ✅ 用腰 + 臂自然挥拍（不是僵硬照搬 demo）
- ✅ 击球后回到 ready pose（imit_joint_pos 全程跟踪生效）
- ✅ 移动时抬脚不拖，不跳

---

## 15. 历史 baseline 对比表

| 版本 | 配置 | iter 1000 cos_sim | iter 8000 cos_sim | 实战感受 |
|---|---|---|---|---|
| V1 21-04-08（旧 best） | `BLADE=(0,1,0)`, gate=True/True/True | -0.40 | +0.48（**反 convention**） | 手臂打直照搬 demo，扭手腕用反手面打正手 |
| V2 15-28-09 | + critic swing_type | -0.06 | +0.21 | cos_sim 学习落后 V1 ~1500 iter |
| V3 19-52-00 | + swing-first 50/50 + 去 critic swing + base_orientation reward | +0.28 → -0.10 | -0.13 | 强制 50/50 把 cos_sim 拖到长期负值，hsr=0.18 卡死 |
| V3 12-07-26 | + gate_pre_strike=False 三个全开 | -0.16 → -0.76 | (kill) | imit_body_pos 主导污染 strike 帧 |
| **v58 14-54-15（当前）** | F1-F7 全部应用 | **+0.585** | (待训) | **真物理正确，不再扭手腕** |

---

## 16. 可选继续探索方向（**未实施**，记录意图）

| # | 想法 | 触发条件 |
|---|---|---|
| 1 | post-strike `_update_ref_state` 用 clip frame 0 作 ref（让 robot 击球后回 ready pose 而不是 follow-through） | 若 v58 跑出来 follow-through 不自然 |
| 2 | 严格按论文重写 cmd 生成逻辑（paper-strict 复现对照） | 用户指定要复现 |
| 3 | 切 split 从 `body_dominant` → `default` (0.65/0.10/0.25)，body_pos 占比 ÷ 2.4 | 若 v58 iter 8000 cos_sim 仍 < +0.4 |
| 4 | bad_orientation 阈值松动（0.8 → 1.0 rad）助 29dof 站立 | 若 29dof v58 iter 3000 EL 仍 < 100 |
| 5 | 加 `pelvis_orientation_l2` 加重到 -2.0（locomotion 用 -5.0） | 若 base 还是不够稳 |
| 6 | post-strike cmd 提前生成下一击目标（paper-style immediate resample） | 若 EL 卡 500 不再涨 |

---

## 17. 当前实施进度（2026-05-29 更新）

> 当前 run：`logs/rsl_rl/unitree_g1_23dof_pingpong_hitter/2026-05-29_14-54-15_v3_swing_first_base_ori`（命名是历史遗留，配置是 v58）
> 当前 iter：~1100（仍在跑）
> 状态：🟢 突破性进展，所有指标全面超越 V1 21-04-08 历史最佳

### 17.1 关键决策时间线

| 日期 | 决策 | 触发证据 | 当前评判 |
|---|---|---|---|
| 2026-05-25 | M1 RSI 写 root_quat（覆盖 expert pelvis_yaw + ±10°） | 站起来后 cos_sim≈0 不动 | 🟢 |
| 2026-05-26 | R8 table 4 阶段 stage-aware curriculum | 早期 paddle 撞桌作弊 | 🟢 |
| 2026-05-27 | R10 vel reward 改 linear-exp + 7-tier ladder | 跨档崩 + 早期无梯度 | 🟢 |
| 2026-05-28 早 | V2 加 critic swing_type | paper Table I 提示 | 🔴 cos_sim 落后 V1 |
| 2026-05-28 晚 | **V3** swing-first 50/50 + 去 critic swing + base_orientation reward | V1/V2 都不稳 | 🔴 cos_sim 永远负值 |
| 2026-05-29 早 | **F2** 撤销 V3 swing-first，回 V1 uniform sampling | V3 26000 iter cos_sim=-0.10 | 🟢 |
| 2026-05-29 早 | **F4-F7** 加 base 稳定 reward 套餐（+ang_vel_xy +lin_vel_z +energy +feet_slide↑） | 用户播放 model_3000 看到跳跃 | 🟢 |
| 2026-05-29 中 | **F3** Plan B 半 gate（body_pos True，joint_* False） | run 12-07-26 全开 gate=False 让 cos_sim 崩到 -0.76 | 🟢 |
| 2026-05-29 中 | **F1** `BLADE_NORMAL_LOCAL` (0,1,0) → (0,-1,0) | NPZ 实测 + URDF rpy=-135° 验证 | 🟢 |
| 2026-05-29 晚 | **F9** 全部移植到 29dof | 11-13-14 run 8735 iter 没站起来 | ⚪ 待验证 |
| 2026-05-29 晚 | **F8** train.log 默认关闭 | 用户：占空间太大 | 🟢 |

### 17.2 v58 第一次 from-scratch run（14-54-15）实测

iter 1000 reward breakdown：
```
imitation_body_pos        : +0.07   ← Plan B 后从旧 0.28 砍 4×
imitation_joint_pos       : +0.11
imitation_joint_vel       : +0.001
imitation total           : +0.18   ← V1 baseline 健康量级

goal_position             : +0.16   ← curriculum 已解锁，发奖中
goal_position_pre_strike  : +0.04
goal_orientation_pre_strike: +0.012  ← 不再是 0.0002 死信号
goal_orientation (terminal): +0.009
goal_base                 : +0.17

base 稳定:
pelvis_ang_vel_xy         : -0.014  ← F4 在预估区间
pelvis_lin_vel_z          : -0.010  ← F5 抑制跳
feet_slide                : -0.048  ← F7 提到 -0.20 后 reward 量级
energy                    : -0.005  ← F6 微小符合预期
```

EL=462, cos_sim=+0.585, hard_contact 从 iter 200 的 1.00 降到 0.20（5× 改善）

### 17.3 停训判据

🟢 **正常停训（成功 → 切 hitter_real）**：
- `hit_success_rate` ≥ 0.80 持续 500 iter
- `vel_fail_rate` ≤ 0.15
- `cos_sim_ema` 500 iter 最低 ≥ 0.50
- `ori_fail_rate` ≤ 0.20
- `imit_phase` ≥ 1
- `table_stage` = 3 active

🔴 **异常停训**：
- iter > 12000 但 `cos_sim_ema` 50i 均值仍 < 0.40
- iter > 12000 但 hsr < 0.30
- actor std < 0（PPO 数值崩）
- bad_orientation > 50% 持续 1000 iter（base 稳定 reward 失效）

🟡 **观察停**：
- shape_tier 升 2 后 cos_sim_ema 跌回 < 0.40（触发 R7 反向 ratchet）
- forehand_share 漂到 [0.30, 0.70] 之外 1000 iter（cheat basin）

---

## 18. 实现 phase 索引（供 AI 复现代码）

| Phase | 文件 | 关键内容 |
|---|---|---|
| 1 | [mdp/commands.py](mdp/commands.py) | `BLADE_NORMAL_LOCAL=(0,-1,0)`，PingpongCommand + `_sample_new_swing` 7 步流程，`_compute_swing_type` 后置分类，`_blade_target_cosine` signed |
| 2 | [mdp/motion_loader.py](mdp/motion_loader.py) | expert clip dict, get_ref_state 双段 lerp/slerp, expert_offset_base 预处理, yaw_from_wxyz |
| 3 | [mdp/observations.py](mdp/observations.py) | 14 项 (Actor 86 / Critic 213)，DelayedObservation wrapper, IMU offset wrapper |
| 4 | [mdp/rewards.py](mdp/rewards.py) | `imitation_*`（gate_pre_strike 参数）、`goal_*`（_strike_gate）、`goal_*_pre_strike`、`pelvis_orientation_l2`、`feet_air_time_no_command`、`robot_table_contact_penalty` |
| 5 | [mdp/__init__.py](mdp/__init__.py) | `from unitree_rl_lab.tasks.locomotion.mdp.rewards import energy` ⬅ F6 必加 |
| 6 | [mdp/events.py](mdp/events.py) | `randomize_imu_offset`, `randomize_comm_delay`, RSI reset, table teleport by stage |
| 7 | [mdp/terminations.py](mdp/terminations.py) | bad_orientation, base_height, hard_contact, non_paddle_table_stuck (stage-aware short-circuit) |
| 8 | [mdp/curriculums.py](mdp/curriculums.py) | imit_anneal (metric mode + body_dominant split), pingpong_curriculum (window/sequenced/cos_sim retreat), update_table_guard_stage |
| 9 | [robots/g1_23dof/hitter/hitter_env_cfg.py](robots/g1_23dof/hitter/hitter_env_cfg.py) | RewardsCfg (含 F4/F5/F6/F7)，ObservationsCfg (Plan B gate)，CurriculumCfg (imit_anneal 在前) |
| 10 | [robots/g1_29dof/hitter/hitter_env_cfg.py](robots/g1_29dof/hitter/hitter_env_cfg.py) | 29dof 同步配置（IMITATION_JOINT_NAMES_29DOF + TRACKED_BODY_NAMES_29DOF + B7 w_i ×4 + 全部 base 稳定 reward） |
| 11 | [agents/rsl_rl_ppo_cfg.py](agents/rsl_rl_ppo_cfg.py) | 复用 `BasePPORunnerCfg`，只改 entry name |
| 12 | [scripts/rsl_rl/train.py](../../../../../scripts/rsl_rl/train.py) | `--log_redirect` 默认 False (F8) |

---

## 19. 引用与依赖

- HITTER 论文：Su 等 2025, arXiv:2508.21043v2
- IsaacLab 5.1 + IsaacSim 5.1 (代码 ready)
- DeepMimic kernel：r^p k=2, r^v k=0.1, r^bp k=10
- expert clips: `motion_datasets/pingpong/humanoid_data/final/expert/new/{forward,backward}/npz/{forward_003,backward_001}_rotated.npz` (Rz(-90°) 已应用)
- URDF: `unitree_ros/robots/g1_description/g1_23dof_rev_1_0_paddle.urdf` (paddle fixed joint rpy=-2.356,0,0)
- robot cfg: `unitree.py:UNITREE_G1_23DOF_PADDLE_MIMIC_CFG` + `UNITREE_G1_23DOF_PADDLE_MIMIC_ACTION_SCALE`

依赖 23dof / 29dof 共享：
- `mdp/commands.py`（BLADE_NORMAL_LOCAL 共享）
- `mdp/rewards.py`（imitation_*, goal_*, goal_*_pre_strike 共享）
- `mdp/motion_loader.py`（NPZ 路径独立配置）

---

**END OF v58 DESIGN**

---

# v59 → v62 架构更新（2026-05-29 / 30）

> **本节是 v62 当前实现的完整描述**，足以独立指导从零复现。v58 部分（前 19 节）作为历史保留 —— v62 在此基础上做了 sampling、curriculum、reward 三大维度的重构。

## 20. v62 架构总览 — 三大改动

| 维度 | v58.1（旧）| **v62（当前）**| 解决了什么 |
|---|---|---|---|
| 击球点采样 | env-local 1D 采样 + 后置 `_compute_swing_type` 重分类 | **世界系采样 + env_origin 锚定 divider**（删除重分类）| backhand cheat basin（fh_share=0.005）|
| 课程结构 | 2 段 imit_anneal × shape_tier 单 ratchet | **3-phase 任务课程**（stand → imit → strike）+ σ 双向跟随 + 课程错开 cooldown | imit feedback trap、σ-deadlock、Phase 1 跳过 |
| Velocity reward | Laplacian `exp(-‖Δv‖/σ)` σ=0.45 | **Gaussian** `exp(-‖Δv‖²/σ²)` σ=1.5 → 0.5 | 中等误差区无梯度（reward=0.001 卡死）|

### v62 完整 file-level 映射

| 文件 | v58→v62 关键改动 |
|---|---|
| `mdp/commands.py` | 删除 `_compute_swing_type` 重分类逻辑；`_sample_new_swing` 完全重写为**世界系采样 + 锚定 divider**；新增 cfg 字段 `swing_p_forehand`、`hit_y_world_cap*`；加 per-swing 诊断 metric（hsr_fh/bh, cos_fh/bh, py_fh/bh）|
| `mdp/curriculums.py` | 新增 `update_task_phase` 函数（3-phase 课程）；删除 σ monotone latch；改 `_REWARD_SHAPE_TIERS` σ_vel 列（Gaussian-scale 1.5→0.5）；加 `_SHAPE_TIER_LATCH` + `_V_IN_TIER_LATCH` cross-curriculum cooldown；删除 `_SWING_RATIO_LATCH`（v62 移除 swing_warmup）|
| `mdp/rewards.py` | `goal_velocity` 公式 Laplacian → Gaussian；其他 reward 不变 |
| `mdp/motion_loader.py` | `frame_from_step` 用 `clip.post_duration` 做插值除数（自然速度 + 末帧锁定）|
| `robots/g1_23dof/hitter/hitter_env_cfg.py` | 加 `task_phase` CurrTerm；`goal_velocity.std` 0.45→1.5；`imitation_body_pos.gate_pre_strike` True→False；imit_anneal `phase_thresholds` 放宽（vel/ori 0.70→0.40/0.55，第 2 段 0.85→0.65/0.75）|

---

## 21. 核心数据：固定常量 + 自动推导

### 21.1 NPZ 数据决定的 demo 几何（自动推导，不可手设）

`PingpongMotionLoader` 读两个 NPZ（forehand/backhand），在 `commands.py PingpongCommand.__init__` 中算出：

```python
# 从 NPZ 读 expert_offset_base (paddle 在 pelvis frame 的 base-y at impact frame)
forehand_y = float(self.motion.clips["forehand"].expert_offset_base[1])  # 23dof 当前: -0.462
backhand_y = float(self.motion.clips["backhand"].expert_offset_base[1])  # 当前: +0.024

# 应用右臂 singularity safety clamp
if cfg.forehand_y_safety_clamp is not None:  # default 0.40
    if forehand_y < 0:
        forehand_y_eff = max(forehand_y, -0.40)  # = -0.40 (clamped from -0.462)
    else:
        forehand_y_eff = min(forehand_y, 0.40)
self._forehand_y_eff = forehand_y_eff  # -0.40

# y_mid_base 是分类阈值
cfg.y_mid_base = 0.5 * (forehand_y_eff + backhand_y)  # = 0.5 * (-0.40 + 0.024) = -0.188

# _swing_y_sign 判数据布局方向
self._swing_y_sign = 1.0 if forehand_y > backhand_y else -1.0  # 当前: -1（forehand 在 -y 侧）

# expert_offset_base[2,2]: forehand=(+0.498,-0.462), backhand=(+0.556,+0.024)
self.expert_offset_base = self.motion.expert_offset_base.to(self.device)

# hit_x（base-frame x of hit point）= 两 demo 的均值
cfg.hit_x = 0.5 * (forehand_x + backhand_x)  # ≈ 0.527
```

**当前 23dof 实际值（forward_003_rotated.npz / backward_001_rotated.npz）**：
```
forehand_y = -0.462,  forehand_y_eff = -0.40  (safety clamp 触发)
backhand_y = +0.024
y_mid_base = -0.188
_swing_y_sign = -1   (forehand 在 -y 侧)
hit_x = +0.527
```

### 21.2 cfg 字段（PingpongCommandCfg）

```python
@configclass
class PingpongCommandCfg(CommandTermCfg):
    # === 数据驱动（auto-derived in __init__）===
    y_mid_base: float | None = None          # 当前 = -0.188
    hit_x: float | None = None               # 当前 = 0.527
    hit_y_cap_low: float | None = None       # legacy sanity, 当前 = -0.40
    hit_y_cap_high: float | None = None      # legacy sanity, 当前 = +0.024
    forehand_y_safety_clamp: float | None = 0.40

    # === v62 世界系采样 ===
    hit_y_world_cap: float = 0.45            # current cap (curriculum mutates)
    hit_y_world_cap_initial: float = 0.45    # tier 0 (covers ±0.40 demo + 5cm)
    hit_y_world_cap_max: float = 1.00        # tier 4 (paper widest)

    # === Bernoulli swing 采样 ===
    swing_p_forehand: float = 0.50           # 固定 50:50 (v62 删除 warmup)

    # === legacy 字段（保留但 v62 不再使用）===
    hit_y_base_range: tuple | None = None              # legacy, dead code
    hit_y_base_initial_half_width: float = 0.10        # legacy
    hit_y_base_max_half_width: float = 0.50            # legacy

    # === 球物理 ===
    hit_z_range: tuple = (0.95, 1.15)
    v_in_mag_range: tuple = (1.5, 2.0)       # v_in 课程会推到 (1.5, 4.0)
    target_land: tuple = (2.45, 0.0, 0.78)
    flight_time_range: tuple = (0.85, 1.05)
    paddle_cor_range: tuple = (0.85, 0.95)

    # === reset / RSI ===
    reset_root_pos: tuple = (0.0, 0.0, 0.74)
    reset_yaw_noise: tuple = (-radians(10), radians(10))
    disable_rsi: bool = False                # set True 跳过 RSI（仅 debug）

    # === reward shape ===
    sigma_g_pos: float = 0.30                # curriculum mutates 0.30→0.06
    strike_window: float = 0.10              # curriculum mutates 0.10→0.01

    # === noise sigmas (cmd 端注入) ===
    noise_p_sigma: float = 0.0
    noise_v_sigma: float = 0.0
    noise_base_sigma: float = 0.0
    noise_t_sigma: float = 0.0

    # === post-strike timing ===
    t_post_swing_fixed: float | None = None          # auto = max post_duration
    t_post_swing_mode: str = "max"

    # === success thresholds（pos_ok/vel_ok/ori_ok 的判定）===
    success_pos_thresh: float = 0.10
    success_vel_thresh: float = 0.5
    success_ori_cos_dist_thresh: float = 0.20
```

---

## 22. 世界系采样完整逻辑（v61 锚定 divider）

### 22.1 几何核心

**关键设计**：`divider_world` **永远锚定在 env_origin**（不跟随 robot 移动）：
```python
divider_world = env_origins[:, 1] + y_mid_base   # 例: 100 + (-0.188) = 99.812
```

为什么锚定？`v60` 让 divider 跟随 `root.y` 时，policy 学会**移动 base 到 +y 让 forehand 命令变成 cross-body 反手伸手**。锚定 env_origin 关闭这个几何 cheat：robot 必须**站在 env_origin 附近**才能合法 hit。

### 22.2 `_sample_new_swing` 完整流程（v62）

```python
def _sample_new_swing(ids, reset_robot, root_pos_override=None, root_quat_override=None):
    # 11 个步骤

    # ════ Step 1: yaw-independent 采样 ════
    # swing_target Bernoulli 50:50（paper Table I task input）
    p_fh = float(self.cfg.swing_p_forehand)  # 固定 0.50
    swing_target = (torch.rand(n) >= p_fh).long()  # 0=forehand, 1=backhand
    self.swing_type[ids] = swing_target  # ★ EARLY 写入（RSI consistency）
    
    # 球物理参数
    hit_z = uniform(hit_z_range)
    v_mag = uniform(v_in_mag_range)
    v_yaw = π + uniform(±40°)            # 球对面射来
    v_pitch = uniform(±75°)
    self.v_ball_in_world[ids] = v_mag * (cos(v_yaw)*cos(v_pitch), sin(v_yaw)*cos(v_pitch), sin(v_pitch))
    self.target_land_world[ids] = env_origins + cfg.target_land
    self.flight_time[ids] = uniform(flight_time_range)
    self.paddle_cor[ids] = uniform(paddle_cor_range)

    # ════ Step 2: RSI override root_quat（reset_robot=True）════
    rsi_frames = None
    if reset_robot and not cfg.disable_rsi:
        rsi_frames, pelvis_yaws = self._sample_rsi_frames(ids)  # 用 self.swing_type 选 clip
        yaw_noise = uniform(reset_yaw_noise)
        final_yaw = pelvis_yaws + yaw_noise   # ★ 用 demo 的 pelvis yaw + noise
        new_root_quat = quat_from_euler_xyz(0, 0, final_yaw)
        self.robot.write_root_state_to_sim(...)
        root_quat = new_root_quat

    # ════ Step 3-7 (v61 重写): 世界系采样 + 锚定 divider ════
    yaw = yaw_from_wxyz(root_quat)
    cos_y = cos(yaw); sin_y = sin(yaw)
    cos_y_safe = where(abs(cos_y) < 0.05, sign-clamped 0.05, cos_y)  # defensive

    hit_x_world = env_origins[:, 0] + cfg.hit_x   # per-env, fixed
    diff_x_world = hit_x_world - root_pos[:, 0]

    # 世界 cap 范围（env-local: env.y ± cap）
    cap = float(cfg.hit_y_world_cap)
    world_y_lo = env_origins[:, 1] - cap
    world_y_hi = env_origins[:, 1] + cap

    # ★★ 核心: divider 锚定 env_origin (不跟随 root) ★★
    divider_world = env_origins[:, 1] + y_mid_base

    # 按 _swing_y_sign 决定 forehand 在 divider 哪侧
    if self._swing_y_sign > 0:
        # forehand 在 +y 侧
        fh_lo_eff = max(divider_world, world_y_lo)
        fh_hi_eff = world_y_hi
        bh_lo_eff = world_y_lo
        bh_hi_eff = min(divider_world, world_y_hi)
    else:  # current case: _swing_y_sign = -1
        # forehand 在 -y 侧 (forehand demo y_base = -0.40 < y_mid -0.188)
        fh_lo_eff = world_y_lo                              # = env.y - 0.45
        fh_hi_eff = min(divider_world, world_y_hi)          # = env.y - 0.188
        bh_lo_eff = max(divider_world, world_y_lo)          # = env.y - 0.188
        bh_hi_eff = world_y_hi                              # = env.y + 0.45

    # 检查范围有效性（boundary case）
    fh_valid = (fh_hi_eff > fh_lo_eff + 1e-4)
    bh_valid = (bh_hi_eff > bh_lo_eff + 1e-4)

    # ════ Step 5: boundary OVERRIDE swing_target ════
    is_forehand_target = swing_target == SWING_FOREHAND
    force_to_bh = is_forehand_target & ~fh_valid & bh_valid
    force_to_fh = ~is_forehand_target & ~bh_valid & fh_valid
    swing_target = where(force_to_bh, SWING_BACKHAND, swing_target)
    swing_target = where(force_to_fh, SWING_FOREHAND, swing_target)
    is_forehand_target = swing_target == SWING_FOREHAND
    self._dead_zone_count[ids] += (force_to_bh | force_to_fh).float()  # diagnostic

    # Step 5b: 重写 swing_type 反映 override
    self.swing_type[ids] = swing_target

    # 极端 fallback（理论上不会触发，因为 divider 锚定 env_origin）
    both_invalid = ~fh_valid & ~bh_valid
    if any(both_invalid):
        fallback_lo = env_origins[:, 1].clone()
        fh_lo_eff = where(both_invalid, fallback_lo, fh_lo_eff)
        # ... fh_hi/bh_lo/bh_hi 同 fallback

    # ════ Step 6: 直接在世界 y 采样 hit_y_world ════
    rand_fh = rand(n); rand_bh = rand(n)
    fh_y_world = rand_fh * (fh_hi_eff - fh_lo_eff) + fh_lo_eff
    bh_y_world = rand_bh * (bh_hi_eff - bh_lo_eff) + bh_lo_eff
    hit_y_world = where(is_forehand_target, fh_y_world, bh_y_world)

    # ════ Step 7: 仅诊断（hit_y_base derived from world for metric/debug）════
    hit_y_base = -sin_y * diff_x_world + cos_y * (hit_y_world - root_pos[:, 1])

    # ════ Step 8: 写 p_hit_world (绝对世界系) ════
    p_hit_world_new = stack((hit_x_world, hit_y_world, env_origins[:, 2] + hit_z), dim=-1)
    self.p_hit_world[ids] = p_hit_world_new

    # ════ Step 9: solve paper Eq.5/Eq.6 (depends on p_hit_world) ════
    self._solve_paddle_target(ids)

    # ════ Step 10: 记录 hit_y_base 诊断值 ════
    self.hit_y_base[ids] = hit_y_base
    self.swing_change_remaining[ids] = 0  # legacy

    # ════ Step 11: base target + time fields ════
    self.p_base_xy_world[ids] = self._compute_base_target(ids, root_quat)
    self.t_pre_initial[ids] = _sample_peak_uniform(0.20, 0.90, 0.30, 0.65)
    self.t_post_swing[ids] = float(cfg.t_post_swing_fixed)
    self.t_to_hit[ids] = self.t_pre_initial[ids]
    self.cur_step[ids] = 0

    if reset_robot and not cfg.disable_rsi:
        self._write_rsi_joint_state(ids, frames=rsi_frames)

    self._reset_window_flags(ids)
    self._freeze_noise(ids)
    self._update_ref_state(ids)
```

### 22.3 `_compute_base_target` (v62 不变)

```python
def _compute_base_target(ids, root_quat):
    yaw = yaw_from_wxyz(root_quat)
    offsets = self.expert_offset_base[self.swing_type[ids]]   # (n, 2)
    offsets_world = _rotate_yaw_2d(offsets, yaw)              # 转世界系
    return self.p_hit_world[ids, :2] - offsets_world          # 世界系 base target
```

返回的 `p_base_xy_world` 是绝对世界坐标，反向给 `goal_base_position` 用。

### 22.4 boundary override 几何举例（说明 divider 锚定后行为）

**场景: robot 在 base.y = +0.4m（向 +y 漂移）**：
- `divider_world = env.y - 0.188`（不变，锚定 env_origin）
- `forehand 范围 = [env.y - 0.45, env.y - 0.188]` (width 0.262)
- `backhand 范围 = [env.y - 0.188, env.y + 0.45]` (width 0.638)
- 命令 forehand → 采样 hit_y_world ∈ [env.y - 0.45, env.y - 0.188]
- robot 在 env.y + 0.4，要 hit env.y - 0.3（左方 0.7m）→ 需要**先移回**才能击中
- `goal_base_position` 自然把 robot 拉回 → cheat path closed

**场景: robot 大幅漂到 +1.5m（极端）**：
- `world_y_hi = env.y + 0.45` → robot 已超出 hit 范围 1.0m+
- forehand 范围仍存在，hit_y 在 robot 左方 1.7m+
- robot 物理上够不到 → goal_position reward 不发 → policy 学着回到 env_origin

---

## 23. 3-phase 任务课程（v61，核心反作弊设计）

### 23.1 设计动机

v60 训练发现：即便锚定 divider，policy 在击球阶段仍倾向**单一 swing 姿态（forehand pose）做所有击球**。Reward landscape 太复杂（imit + goal_pos + goal_vel + goal_ori 同时存在），policy 选最容易的局部最优。

**用户的解法**：把训练分 3 个**单向阀门** phase，每个 phase 只学一件事：

```
Phase 0 (stand):    站立（imit shaping prior 0.10, goal_* 全关）
   ↓ EL_ema ≥ 350
Phase 1 (imit):     正反手 demo 充分 imit（imit weight 1.0, goal_* 仍关）
   ↓ EL_ema ≥ 450 AND phase_1_iters_elapsed ≥ 2000
Phase 2 (strike):   击球任务（imit weight 0.30 paper-like, goal_* baseline 启动）
```

**关键约束**：每个 phase 只升不降（monotone latch）；Phase 1 必须**至少跑 2000 iter**；Phase 2 入口**一次性 reset goal_* 权重到 baseline**。

### 23.2 实现细节

#### `_TASK_PHASE_LATCH` (curriculums.py)

```python
_TASK_PHASE_LATCH: dict = {
    "phase": 0,                    # 当前 phase (0/1/2)
    "phase_1_entry_iter": -1,      # 进入 Phase 1 的 iter
    "prev_phase": 0,               # 上一次的 phase（用于检测 Phase 1→2 跨界）
}

# Phase 2 入口要 reset 的 weight
_TASK_PHASE_2_BASELINE_WEIGHTS = {
    "goal_position": 2.0,
    "goal_position_pre_strike": 1.0,
    "goal_velocity": 2.0,
    "goal_velocity_pre_strike": 1.0,
    "goal_orientation": 0.5,
    "goal_orientation_pre_strike": 0.5,
}

_TASK_PHASE_IMIT_SPLIT = {
    "imitation_joint_pos": 0.40,
    "imitation_body_pos": 0.50,
    "imitation_joint_vel": 0.10,
}

_TASK_PHASE_GOAL_TERMS = (
    "goal_position", "goal_position_pre_strike",
    "goal_velocity", "goal_velocity_pre_strike",
    "goal_orientation", "goal_orientation_pre_strike",
)
```

#### `update_task_phase` 完整逻辑

```python
def update_task_phase(env, env_ids,
                     el_phase_0_to_1: float = 350.0,
                     el_phase_1_to_2: float = 450.0,
                     phase_1_min_iters: int = 2000,
                     imit_w_phase0: float = 0.10,
                     imit_w_phase1: float = 1.00,
                     imit_w_phase2: float = 0.30,
                     num_steps_per_env: int = 24):
    
    el_ema = float(_EP_LENGTH_EMA["value"]) if _EP_LENGTH_EMA["init"] else 0.0
    iter_count = int(env.common_step_counter // max(num_steps_per_env, 1))
    
    cur_phase = int(_TASK_PHASE_LATCH["phase"])
    prev_phase = cur_phase
    
    # Phase 0 → 1
    if cur_phase < 1 and el_ema >= el_phase_0_to_1:
        cur_phase = 1
        _TASK_PHASE_LATCH["phase"] = 1
        _TASK_PHASE_LATCH["phase_1_entry_iter"] = iter_count
    
    # Phase 1 → 2 (需要 EL gate AND min duration)
    if cur_phase < 2 and el_ema >= el_phase_1_to_2:
        phase_1_iters = iter_count - int(_TASK_PHASE_LATCH["phase_1_entry_iter"])
        if phase_1_iters >= int(phase_1_min_iters):
            cur_phase = 2
            _TASK_PHASE_LATCH["phase"] = 2
    
    # Set imit weights
    imit_w_table = (imit_w_phase0, imit_w_phase1, imit_w_phase2)
    w_i = imit_w_table[cur_phase]
    for term, share in _TASK_PHASE_IMIT_SPLIT.items():
        env.reward_manager.get_term_cfg(term).weight = share * w_i
    
    # Phase 0/1: zero out goal_* (override window curriculum)
    if cur_phase < 2:
        for term in _TASK_PHASE_GOAL_TERMS:
            env.reward_manager.get_term_cfg(term).weight = 0.0
    
    # Phase 2 entry: ONE-TIME reset goal_* to baseline (突破 cos_sim_ratchet_freeze 死锁)
    if prev_phase < 2 and cur_phase == 2:
        for term, weight in _TASK_PHASE_2_BASELINE_WEIGHTS.items():
            env.reward_manager.get_term_cfg(term).weight = float(weight)
    
    _TASK_PHASE_LATCH["prev_phase"] = cur_phase
    return {"task_phase": float(cur_phase), "task_phase_imit_w": float(w_i), ...}
```

### 23.3 Phase 各阶段 reward 总结

| Reward 项 | Phase 0 (stand) | Phase 1 (imit) | Phase 2 (strike) |
|---|---|---|---|
| imitation_joint_pos | 0.04 (0.10×0.40) | 0.40 (1.00×0.40) | 0.12 (0.30×0.40) |
| imitation_body_pos | 0.05 (0.10×0.50) | 0.50 (1.00×0.50) | 0.15 (0.30×0.50) |
| imitation_joint_vel | 0.01 (0.10×0.10) | 0.10 (1.00×0.10) | 0.03 (0.30×0.10) |
| goal_position | 0 | 0 | **2.0** (baseline，window curriculum 后续 ramp) |
| goal_position_pre_strike | 0 | 0 | 1.0 |
| goal_velocity | 0 | 0 | 2.0 |
| goal_velocity_pre_strike | 0 | 0 | 1.0 |
| goal_orientation | 0 | 0 | 0.5 |
| goal_orientation_pre_strike | 0 | 0 | 0.5 |
| goal_base | 0.8（不变） | 0.8 | 0.8 |
| goal_base_orientation | 0.3 | 0.3 | 0.3 |
| alive | 0.04 | 0.04 | 0.04 |
| pelvis_orientation/height/lin_vel_z/ang_vel_xy | 不变 | 不变 | 不变 |
| undesired_contacts/joint_*/feet_slide | 不变 | 不变 | 不变 |

### 23.4 Phase 进入条件实证（v61.1 run 2026-05-30_18-59-49）

```
iter 3001: phase 0 → 1   (EL_ema 跨过 350)
iter 5000: 仍在 phase 1（EL=497, p1elapsed=1999, gate 未达 2000）
iter 5008: phase 1 → 2   (EL=498 ≥ 450 AND p1elapsed=2007 ≥ 2000)
iter 5100: w_goal_pos=2.0 (baseline 重置成功) → window curriculum 接力 ramp 到 3.0
iter 6000: w_goal_pos=5.0 (tier 2 ramp 完成)
```

---

## 24. v62 Velocity Reward 重写（Gaussian + 错开课程）

### 24.1 公式改动

```python
# v58/59/60/61: Laplacian (linear)
def goal_velocity(env, command_name, std=0.45, half_width=None):
    err = torch.norm(v_blade_b - v_hat_b, dim=-1)        # L2 norm
    return torch.exp(-err / std) * gate                   # /σ

# v62: Gaussian (squared)
def goal_velocity(env, command_name, std=1.50, half_width=None):
    err = torch.sum(torch.square(v_blade_b - v_hat_b), dim=-1)  # ||·||²
    return torch.exp(-err / (std**2)) * gate                     # /σ²
```

### 24.2 数学：为什么改公式

v61 现象：`||Δv|| ≈ 2 m/s` 时 Laplacian σ=0.45 → reward = exp(-2/0.45) ≈ 0.012（接近零）。
**Gradient 为零 → policy 无法从 ‖Δv‖=2 学到 ‖Δv‖=1**。

Gaussian σ=1.5 → reward = exp(-(2)²/2.25) = 0.169（**14× 更大**）。Gradient ≈ -0.30（vs Laplacian -0.026，**12× 更强**）。

### 24.3 σ_vel 课程表（v62 新值）

```python
# curriculums.py:_REWARD_SHAPE_TIERS 第 2 列（σ_vel）改成 Gaussian-scale
_REWARD_SHAPE_TIERS = (
    # (σ_pos, σ_vel, σ_ori, hsr_thr, pos_thr, vel_thr, ori_thr)
    (0.06, 0.50, 0.20, 0.85, 0.95, 0.85, 0.85),  # tier 6 (paper-strict, σ_vel=0.50)
    (0.08, 0.65, 0.22, 0.75, 0.92, 0.78, 0.80),  # tier 5
    (0.10, 0.80, 0.25, 0.65, 0.88, 0.70, 0.75),  # tier 4
    (0.13, 1.00, 0.28, 0.55, 0.82, 0.62, 0.70),  # tier 3
    (0.18, 1.20, 0.32, 0.40, 0.75, 0.55, 0.65),  # tier 2
    (0.24, 1.35, 0.36, 0.20, 0.60, 0.40, 0.55),  # tier 1
    (0.30, 1.50, 0.40, 0.00, 0.00, 0.00, 0.00),  # tier 0 (default, σ_vel=1.50)
)
```

env_cfg `goal_velocity` 默认 `std=1.50`（match tier 0）。

### 24.4 σ 双向 follow EMAs（v60 删除 monotone latch）

```python
# v60 fix: 删除 max(min(cur, target), floor) latch
# σ 现在跟 EMAs 双向变化（升/降都 OK）
command.cfg.sigma_g_pos = max(sigma_target, 0.06)             # floor 0.06
gv_cfg.params["std"] = max(std_vel_target, 0.20)              # floor 0.20
go_cfg.params["std"] = max(std_ori_target, 0.20)              # floor 0.20
```

### 24.5 cross-curriculum cooldown (shape_tier ↔ v_in_mag)

**问题**：shape_tier 收紧 σ_vel **AND** v_in_mag 增加球速**同时发生**会让 policy 同时面对"reward 更严 + 任务更难"，policy 来不及适应 → 退步。

**Fix**：500 iter cooldown，两个课程不能在 500 iter 内同时升级。

```python
# curriculums.py
_SHAPE_TIER_LATCH = {"tier": 0, "last_change_iter": -10000}
_V_IN_TIER_LATCH = {"high": 2.0, "last_change_iter": -10000}
_CROSS_CURRICULUM_COOLDOWN_ITERS = 500
```

shape_tier 升级时检查 v_in_mag 上次升级 iter：
```python
prev_shape_tier = int(_SHAPE_TIER_LATCH["tier"])
cooldown_active_for_shape = (iter_count_now - _V_IN_TIER_LATCH["last_change_iter"]) < 500

if new_shape_tier > prev_shape_tier and cooldown_active_for_shape:
    # HOLD shape_tier (don't advance, use previous tier values)
    held_idx = len(_REWARD_SHAPE_TIERS) - 1 - prev_shape_tier
    sigma_target = _REWARD_SHAPE_TIERS[held_idx][0]
    std_vel_target = _REWARD_SHAPE_TIERS[held_idx][1]
    std_ori_target = _REWARD_SHAPE_TIERS[held_idx][2]
    shape_tier = prev_shape_tier
else:
    if new_shape_tier > prev_shape_tier:
        _SHAPE_TIER_LATCH["last_change_iter"] = iter_count_now
    _SHAPE_TIER_LATCH["tier"] = new_shape_tier
```

v_in_mag 升级类似（互相检查对方的 last_change_iter）。

---

## 25. Per-swing 诊断 metrics（验证 cheat 用）

### 25.1 6 个新 metric

```python
# commands.py 新增 state
self._paddle_y_base_at_strike = torch.zeros(n)   # 每 env 在 strike 帧 paddle 在 base 系 y
self._cos_sim_at_strike = torch.zeros(n)         # 每 env 在 strike 帧的 signed cos_sim

# _update_success_window 中（每个 strike 帧捕获）
paddle_in_base = quat_apply_inverse(root_quat_w, blade_pos_w - pelvis_pos_w)
self._paddle_y_base_at_strike = where(in_window, paddle_in_base[:, 1], previous)
self._cos_sim_at_strike = where(in_window, _blade_target_cosine(), previous)

# _refresh_metrics_from_counts 中：split by swing_type
fh_mask = (self.swing_type == SWING_FOREHAND).float()
bh_mask = (self.swing_type == SWING_BACKHAND).float()
fh_count = fh_mask.sum().clamp_min(1.0)
bh_count = bh_mask.sum().clamp_min(1.0)

hsr_fh = (hsr_per_env * fh_mask).sum() / fh_count
hsr_bh = (hsr_per_env * bh_mask).sum() / bh_count
# (类似 cos_sim, paddle_y_base 也 split)

# Broadcast scalar 给 IsaacLab logger（mean_over_envs 给回正确值）
self.metrics["hsr_forehand_only"] = hsr_fh.expand(num_envs).clone()
self.metrics["hsr_backhand_only"] = hsr_bh.expand(num_envs).clone()
self.metrics["cos_sim_forehand_only"] = cos_fh.expand(num_envs).clone()
self.metrics["cos_sim_backhand_only"] = cos_bh.expand(num_envs).clone()
self.metrics["paddle_y_base_at_strike_forehand"] = py_fh.expand(num_envs).clone()
self.metrics["paddle_y_base_at_strike_backhand"] = py_bh.expand(num_envs).clone()
```

### 25.2 解读这些 metric（cheat detection）

**正确学习的 signature**：
- `hsr_forehand_only ≈ hsr_backhand_only`（差 ≤ 0.10）
- `paddle_y_base_at_strike_forehand` ≈ `forehand_y_eff = -0.40`
- `paddle_y_base_at_strike_backhand` ≈ `backhand_y = +0.024`
- `cos_sim_forehand_only > 0.5` AND `cos_sim_backhand_only > 0.5`（都正）

**Cheat 现象**：
- `paddle_y_base_at_strike_backhand ≈ -0.30`（应该是 +0.024 但跟 forehand 一样在 -y 侧）
- `cos_sim_backhand_only < 0`（拍面方向反了）
- `hsr_forehand_only - hsr_backhand_only > 0.30`（一边强一边弱）

---

## 26. 实施顺序（从 v58 复现到 v62）

如果从 v58 baseline 开始重新实现 v62，建议顺序（按 dependency）：

### Step 1: motion_loader frame interpolation 修复
```python
# motion_loader.py frame_from_step
clip_post = max(self.post_duration, dt)  # 用 clip 自然 duration
post_frame = impact + ((sim_t - pre) / clip_post).clamp(0, 1) * (last - impact)
```

### Step 2: 重写 commands.py `_sample_new_swing` 为世界系采样
- 删除 `_compute_swing_type` 重分类调用
- Step 3-7 改为世界系采样 + `divider_world = env_origins[:, 1] + y_mid_base`
- swing_target Bernoulli 50:50 + boundary override

### Step 3: 加 cfg 字段
- `hit_y_world_cap`, `hit_y_world_cap_initial=0.45`, `hit_y_world_cap_max=1.00`
- `swing_p_forehand=0.50`

### Step 4: rewards.py `goal_velocity` 改 Gaussian
```python
err = torch.sum(torch.square(v_blade_b - v_hat_b), dim=-1)  # squared
return torch.exp(-err / (std**2)) * gate                     # /σ²
```

### Step 5: 加 per-swing 诊断 metric
- `_paddle_y_base_at_strike` / `_cos_sim_at_strike` state
- 6 个 metric 在 `_refresh_metrics_from_counts` 中 split

### Step 6: env_cfg 改动
- `imitation_body_pos.gate_pre_strike: True → False`
- `goal_velocity.std: 0.45 → 1.50`
- imit_anneal `phase_thresholds` vel/ori 阈值放宽（0.70→0.40, 0.85→0.65）

### Step 7: curriculums.py 大改
- 删除 `_SWING_RATIO_LATCH` + 对应 curriculum block
- 删除 σ monotone latch
- 改 `_REWARD_SHAPE_TIERS` σ_vel 列（Gaussian-scale）
- 加 `_TASK_PHASE_LATCH` + `update_task_phase` 函数
- 加 `_SHAPE_TIER_LATCH` + `_V_IN_TIER_LATCH` + cross-curriculum cooldown
- pingpong curriculum 改为驱动 `hit_y_world_cap`（不再驱动 `hit_y_base_range`）
- retreat block 也改为收 `hit_y_world_cap` 到 initial
- metric writes 加 task_phase 系列、6 个 per-swing metric

### Step 8: env_cfg 加 `task_phase` CurrTerm
```python
task_phase = CurrTerm(
    func=mdp.update_task_phase,
    params={
        "el_phase_0_to_1": 350.0,
        "el_phase_1_to_2": 450.0,
        "phase_1_min_iters": 2000,
        "imit_w_phase0": 0.10,
        "imit_w_phase1": 1.00,
        "imit_w_phase2": 0.30,
    },
)
```
**位置**：必须放在 `imit_anneal` 和 `pingpong` 之后（顺序敏感，task_phase 需要覆盖它们的 weight 设置）。

### Step 9: 验证 AST + 数学 sanity
- 4 个文件 AST 通过
- 验证 Gaussian σ=1.5 时 ‖Δv‖=2 reward = 0.169
- 验证 divider_world = env.y - 0.188（锚定）
- 验证 task_phase 0→1→2 latch 正确

### Step 10: from-scratch 训练（约 8000-15000 iter）
预期里程碑：
- iter 0-3000: Phase 0 stand 学习，EL 慢爬到 350
- iter ~3000: phase 0→1，imit_w 跳到 1.0
- iter ~5000: phase 1→2（满足 EL≥450 + p1elapsed≥2000），goal_* baseline 启动
- iter 5000+: 击球任务正式开始，hsr 应快速从 0 涨到 0.30+
- iter 8000-10000: hsr 0.50+，正反手 paddle_y_base 都接近各自 demo

---

## 27. 当前 cfg 默认值速查表（v62）

| 字段 | 文件 | 当前值 | 含义 |
|---|---|---|---|
| `swing_p_forehand` | commands.py | 0.50 | Bernoulli p(forehand)，固定 50:50 |
| `hit_y_world_cap` | commands.py | 0.45 | 当前世界 cap（curriculum mutates）|
| `hit_y_world_cap_initial` | commands.py | 0.45 | tier 0 cap |
| `hit_y_world_cap_max` | commands.py | 1.00 | tier 4 max cap |
| `forehand_y_safety_clamp` | commands.py | 0.40 | forehand_y abs 上限 |
| `reset_yaw_noise` | commands.py | (-10°, +10°) | RSI yaw noise |
| `t_post_swing_mode` | commands.py | "max" | follow-through 长度模式 |
| `goal_velocity.std` | env_cfg.py | 1.50 | Gaussian σ for vel reward (tier 0) |
| `goal_velocity.weight` | env_cfg.py | 2.0 | weight init (curriculum ramp 到 8.0+) |
| `goal_position.std` | env_cfg.py | (n/a, uses sigma_g_pos) | reads cfg.sigma_g_pos |
| `goal_position.weight` | env_cfg.py | 2.0 | weight init |
| `goal_orientation.std` | env_cfg.py | 0.5→0.20 | curriculum tightens |
| `goal_orientation.weight` | env_cfg.py | 0.5 | weight init |
| `goal_base.weight` | env_cfg.py | 0.8 | (target after ramp from 0.5) |
| `imitation_body_pos.gate_pre_strike` | env_cfg.py | **False** | v62: 启用 post-strike body imit |
| `imit_anneal.phase_thresholds[0]` | env_cfg.py | hsr=0.30, pos=0.40, **vel=0.40, ori=0.55**, EL=250 | v62 放宽 |
| `imit_anneal.phase_thresholds[1]` | env_cfg.py | hsr=0.50, pos=0.70, **vel=0.65, ori=0.75**, EL=400 | v62 放宽 |
| `imit_anneal.w_i_values` | env_cfg.py | (0.5, 0.3, 0.15) | 3-phase weights（task_phase override 这些）|
| `task_phase.el_phase_0_to_1` | env_cfg.py | 350.0 | Phase 0→1 EL gate |
| `task_phase.el_phase_1_to_2` | env_cfg.py | 450.0 | Phase 1→2 EL gate |
| `task_phase.phase_1_min_iters` | env_cfg.py | 2000 | Phase 1 minimum duration |
| `task_phase.imit_w_phase0` | env_cfg.py | 0.10 | Phase 0 imit (low shaping) |
| `task_phase.imit_w_phase1` | env_cfg.py | 1.00 | Phase 1 imit (heavy) |
| `task_phase.imit_w_phase2` | env_cfg.py | 0.30 | Phase 2 imit (paper) |
| `_CROSS_CURRICULUM_COOLDOWN_ITERS` | curriculums.py | 500 | shape ↔ v_in cooldown |

---

## 28. v62 启动训练（建议）

### From-scratch（推荐，新设计需要 fresh policy）

```bash
python scripts/rsl_rl/train.py \
    --task Unitree-G1-23dof-Pingpong-HITTER \
    --headless \
    --run_name v62_full_design
```

### Resume（如果有 v61 ckpt 可继承站立基础）

```bash
python scripts/rsl_rl/train.py \
    --task Unitree-G1-23dof-Pingpong-HITTER \
    --headless \
    --resume \
    --load_run <v61_run_name> \
    --checkpoint model_3000.pt \
    --run_name v62_resume_from_v61
```

注意：resume 时 module-level state（`_TASK_PHASE_LATCH` 等）会**重置为初始值**（Python restart），phase 重新从 0 开始爬。policy 权重保留，所以 EL 应该比 from-scratch 涨得快。

### 关键监控（前 5000 iter）

| iter | 期望状态 |
|---|---|
| 0-100 | task_phase=0, imit_w=0.10, goal_*=0, σ_vel=1.50, EL 慢爬 |
| 1000 | EL 应在 50-100 范围 |
| 2500-3500 | EL 跨过 350，phase 0→1 触发 |
| 3500-5500 | phase=1, imit_w=1.0，imit reward 飙升（jp~0.40, bp~0.50）|
| ~5000 | phase 1→2 触发（EL≥450 + p1elapsed≥2000），goal_* baseline 启动 |
| 5000+ | hsr 从 0 开始涨，per-swing diagnostics 应分化（py_fh→-0.40, py_bh→+0.024）|
| 8000+ | hsr 期望 0.40+，cos_sim 0.50+，正反手平衡 |

---

**END OF v62 DESIGN UPDATE (2026-05-30)**

---

# v63 更新 (2026-06-01)

> 本节是 v62 之后的当前状态与勘误。详细问题/方案记录见 [TROUBLESHOOTING.md](TROUBLESHOOTING.md) §十二~十六（R24-R27 / W16-W18 / N1-N3）。

## 29. v63 关键变更

| # | 变更 | 文件 | 关联 |
|---|---|---|---|
| **G1** | 加 3 个下半身腿正则：`leg_joint_deviation`(hip_roll/yaw 偏离默认,常开,复用 `joint_deviation_l1`)、`feet_contact_no_strike`(待命 `t_to_hit≤0` 奖双脚着地)、`feet_distance_no_strike`(待命双向罚脚间距,叉开×0.3)。权重纳入 `update_task_phase` 做 phase 课程(Phase0/1/2 = 强→弱)| [rewards.py](mdp/rewards.py)、[curriculums.py update_task_phase](mdp/curriculums.py)、[g1_23dof/hitter/hitter_env_cfg.py](robots/g1_23dof/hitter/hitter_env_cfg.py) | R24 — 治"击球后抬腿/单脚站/晃" |
| **G2** | 23dof `imitation_joint_names` override 为**全 11 个上半身关节**(加回右臂 distal shoulder_yaw/elbow/wrist_roll);`tracked_body_names` 保持 8 不变(obs 维度不变可 resume)| [g1_23dof/hitter/hitter_env_cfg.py CommandsCfg](robots/g1_23dof/hitter/hitter_env_cfg.py) | R27 — 正手姿势对照实验 |
| **G3** | 换 motion clip：`new/forward_003`+`new/backward_001` → **`new_new/forward_001_rotated` + `new_new/backward_001_rotated`** | [commands.py:847-848](mdp/commands.py) | R25 — 旧正手 clip follow-through 下扎撞桌;新 clip follow-through 全程在台面之上 |
| **G4** | `hit_z_range` 上限 1.15→**1.25**;课程 z 档上限统一 1.25 | [commands.py:945](mdp/commands.py)、[curriculums.py](mdp/curriculums.py) | R26 — 对齐新 clip 高接触点(fh 1.16/bh 1.26)|
| **G5** | 29dof 从 v58/B7 一步同步到 v62+腿正则(3-phase、Gaussian vel std=1.5、leg regs、去 B7 ×4 boost)| [g1_29dof/hitter/hitter_env_cfg.py](robots/g1_29dof/hitter/hitter_env_cfg.py) | N1-N3 — 29dof **首次站起来** |

## 30. 记录勘误（修正旧文档错误）

- **§4.7 expert clip 实测表**：描述的是**旧 clip**(`new/forward_003_rotated` / `new/backward_001_rotated`,impact frame 50/32,击球 z≈0.78/0.92)。**v63 起已换 `new_new/forward_001_rotated` + `backward_001_rotated`**(impact frame 20/56,击球 z≈1.16/1.26,follow-through 高位不撞桌)。§4.7 旧数值仅历史参考。其中 **waist_yaw "4.6°/锁住" 对旧 clip 是正确的** —— demo 不靠扭腰发力(靠肩 pitch 48° + 迈步),这点新旧 clip 一致(W16 勘误:曾误用 torso-pelvis body-quat yaw 算出"18°扭腰",实为 pelvis roll/pitch 污染,真值看 `joint_pos[waist_yaw]` ≈ 4.6°)。
- **§19 引用与依赖**：当前权威 clip 路径 = `motion_datasets/pingpong/humanoid_data/final/expert/new_new/{forward,backward}/npz/{forward_001,backward_001}_rotated.npz`(替代旧 `new/.../{forward_003,backward_001}`)。
- **§3 / §4 imitation 集合**：23dof v63 起 `imitation_joint_names` = 全 11 上半身关节(含右臂 distal);`tracked_body_names` 仍为 8(不含右臂 distal)。
- **`target_land`**：(2.45, 0, 0.78) = 对方半台正中心(球台中心 1.77→远边 3.14 的中点 2.455,差 0.5cm;y=0 中线;z=台面 0.76+球半径 0.02)。已确认无误。
- **29dof 配置 docstring**:旧写 "imitation 全 17 关节无排除" **是错的**,实际 12 关节(右臂 distal 5 个排除,同 23dof 自由臂)。已在 [g1_29dof/hitter/hitter_env_cfg.py](robots/g1_29dof/hitter/hitter_env_cfg.py) 顶部修正。

## 31. 当前未决 / 待验证

- **正手姿势**:R27 全关节模仿把 cos_sim 抬上去了(FH 0.53/BH 0.50 平衡)、hsr 0.81,但视觉姿势仍不够自然 → 残因在 motion clip 风格(step-and-reach,不扭腰)+ 5-DOF 臂可达性,非 reward bug。换 new_new clip(R25)后待重验。
- **new_new clip from-scratch**:换 clip 后 `expert_offset_base/hit_x/y_mid_base` 重算,需 from-scratch 重训(勿从旧 clip checkpoint resume)。验证:击球后不再抵桌/抽搐、`pos_fail` 正常。
- **29dof 退步**(N2):iter8180 峰值 0.86/tier6 → iter30920 退到 0.705/tier4,待查(疑似 see-saw / EMA 退档)。
- **R23 崩溃**:旧 v61 run iter35083 PPO std→NaN(plateau + 无界 action 惩罚放大),v62 已修根因;无界 `action_rate_l2/action_l2` 放大器仍是 latent 风险(未硬化)。

---

**END OF v63 UPDATE (2026-06-01)**
