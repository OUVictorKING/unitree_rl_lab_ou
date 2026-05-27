# Paper-Aligned EnvCfg v5.7 — HITTER (arXiv:2508.21043v2)
首先详细阅读文章 `/home/woan/文档/zotero/AllData/storage/7XEMW63Q/Su 等 - 2025 - HITTER A HumanoId Table TEnnis Robot via Hierarchical Planning and Learning.pdf`。

这个大纲只是训练机器人跟踪击球的动作以及速度、拍面朝向，并没有真正的击球物理仿真。

## Context

**v5.7 增量**: 以当前 `final.md` 为母版完整保留 reward / obs / action / termination / event / curriculum / scene / sim / 歧义消解的详细表格和实现规范；仅将 v5.7 锁定修改直接并入正文。本文档不保留未决项或过渡草案。

**用户决定累计 (v5.7 锁定)**:
1. paper [35] = DeepMimic (kernel k 实参借用)。
2. `r_g_base` 击球前 ON，击球后 (`t_to_hit <= 0`) OFF，不提前切下一击目标。
3. Strike window: **±3 帧 @ 50Hz，共 7 帧，`abs(t_to_hit) <= 0.06s`**。
4. 删除 `r^e`，避免与 `r_g_pos / r_g_vel` 的击球任务目标冲突。
5. **`r^c -> r^bp`**: 上半身各 body 的 anchor-relative position 跟踪，排除 `right_paddle_blade`；body-level velocity 不在此项，权重不能太高。
6. **`r_g_ori` 目标来自 `cmd.n_target_world`**: `n_target_world = (v_ball_out_world - v_ball_in_world) / ||v_ball_out_world - v_ball_in_world||`，即 paper Eq.5 的 paddle normal，不再由 `v_racket_hat_world / ||v_racket_hat_world||` 在 reward 端重新定义。
7. Obs `t_strike` 统一命名为 **`t_to_hit`** (剩余击球时间)，不存绝对击球时刻。
8. **Cmd hit point sample 初始范围**: `x = 0.4 m fixed`, `y in [0.05, 0.25] m`, `z in [0.95, 1.15] m`，均为 world frame；该范围由当前 expert impact blade z 与 forehand/backhand base-frame offset 反推，必须高于 table top `z=0.76`，不得再使用 `[0.08,0.60]` 的旧地下/桌下范围。
9. **Cmd planner 与训练 cmd 解耦**: planner 上层只关心“击球点 ↔ base”相对关系；训练端不 import `planner.solve_paddle_target`，只在 `commands.py` 内 inline 同一套 paper Eq.5/Eq.6 公式。
10. **`r^p / r^v` 关节集合 J**: 仅上半身，排除 `right_wrist_roll_joint`，|J|=10。
11. **`v_racket_hat_world` 不再直接采样**: 由 `p_hit_world / v_ball_in_world / target_land_world / flight_time / paddle_cor` 按 paper Eq.5+Eq.6 推导得到。
12. **`v_ball_in_world` 初始采样**: `v_in_mag ~ U[2,4]`, `v_in_yaw = pi + U[-40 deg, +40 deg]`, `v_in_pitch ~ U[-75 deg, +75 deg]`，允许下落球；课程后可扩展到 `U[2,5.5]`。
13. **`target_land_world` 常量**: `(2.45, 0.0, 0.78)`，即对方半桌中心 + 桌面高度 + 球半径。
14. **`flight_time` 采样**: `uniform[0.30, 0.65]` 秒。
15. **`paddle_cor` 常量**: `0.85`，不做 domain randomization。
16. **`t_pre_initial`**: `truncN(low=0.20, high=0.90, peak_low=0.30, peak_high=0.65)` 秒。
17. **`t_post_swing`**: 固定为 `0.60s`。这是第一版 HITTER 的硬约束，用负的 `t_to_hit` 唯一表示 post recovery phase，避免 Actor 在不可观测随机 post 时长下学习恢复段。
18. **Cmd 重采样边界**: `t_to_hit <= -t_post_swing`，即 pre 段与 post 段都走完后立即重采样。
19. **Action scale**: 不用全局 0.25，复用 `UNITREE_G1_23DOF_PADDLE_MIMIC_ACTION_SCALE`，即 `0.25 * effort_limit / stiffness`。
20. **表桌碰撞**: 在 scene 中生成静态长方体 table，用 filtered `ContactSensor` 读 robot-table 接触，作为 `r_table_contact` 软惩罚，不作为 termination。
21. **`swing_type` 几何推导**: 不再独立采样；每次 swing resample 时由 base-frame `hit_y_base` 与 `Y_MID_BASE = +0.157 m` 决定：`forehand if hit_y_base > 0.157 else backhand`。
22. **`swing_type` 1-change lock**: pre-strike (`t_to_hit > 0`) 期间允许因 base 漂移重新判断并变更 1 次，变更后锁定；击球后 (`t_to_hit <= 0`) 不再变。
23. **expert offset 使用 base frame**: forehand `(+0.496, +0.208)`，backhand `(+0.428, +0.106)`；world frame offset 不可用，因为两条 clip impact yaw 相差约 65 度。
24. **cmd noise per-swing freeze**: 每次 resample 时一次性采样并冻结到下次 resample；只注入 Actor obs，Critic / reward / gate / resample boundary 全部用 clean cmd。
25. **cmd noise 数值**: `noise_p sigma=0.005m clip±0.015`, `noise_v sigma=0.05m/s clip±0.15`, `noise_base sigma=0.015m clip±0.045`, `noise_t sigma=0.005s clip±0.015`。
26. **`sigma_g_pos` curriculum**: 击球成功率驱动，单调收紧不回退，`0.10 -> 0.06 -> 0.04 -> 0.03 -> 0.02`。
27. **删除 `mimic_start_prob` curriculum**: 每个 episode 都从 expert clip RSI 起步，不存在 free 起步分支。
28. **hit / incoming velocity range curriculum**: 使用击球成功率驱动扩展；原 `v_mag range` 语义在 v5.7 中改为 `v_in_mag range`。
29. **ContactSensor**: 使用 `contact_forces` 处理一般接触 / feet / undesired contact，使用 `robot_table_contact` 专门过滤 Table。
30. **URDF 路径**: `/home/woan/HumanoidProject/unitree_rl_lab/unitree_ros/robots/g1_description/g1_23dof_rev_1_0_paddle.urdf`，与 `unitree.py` 中 `UNITREE_G1_23DOF_PADDLE_MIMIC_CFG` 一致。
31. **Actor / Critic obs 维度严格保持**: Actor=86，Critic=213；`v_ball_in_world / target_land_world / flight_time / paddle_cor` 等上游物理输入全部不进 obs。
32. **训练 monitor**: forehand/backhand 偏斜只监控，不修采样逻辑；新增 §12 的 10 项指标。
33. **curriculum success 定义锁定**: 无真实 ball rollout 时，`hit_success` 由 strike window 内任意一帧同时满足 pos / vel / ori 三个几何阈值判定，不从 reward 数值或物理球碰撞反推。
34. **success threshold 与 reward σ 分离**: `success_pos_thresh=max(0.06, 2*sigma_g_pos_now)`，`success_vel_thresh=1.0 m/s`，`success_ori_cos_dist_thresh=0.25`；curriculum 统计用这组三个阈值。
35. **Quaternion convention 硬约束**: 全工程内部统一 `wxyz`；任何 IsaacLab / asset / utility 边界若给 `xyzw`，必须显式转换，并写 identity / 90deg yaw / blade normal 三个 unit test。
36. **blade local normal 验证**: 默认 `right_paddle_blade` local +Y 为拍面法向；必须用 FK 或 debug draw 实测确认。若反向，只能在 asset normal 定义处统一修正，不允许在 reward 内临时取负。
37. **reset / RSI / cmd 顺序锁定**: reset root 到训练 nominal pose → 生成 clean cmd → 由 `cmd.swing_type` 选 ref clip → RSI 写入专家关节/速度但不照搬专家世界 root xy/yaw → 必要时重算 clean cmd 几何 → 冻结 noise。
38. **expert root reset 选择**: 不直接复制专家世界 root xy/yaw 到仿真 robot root；训练 world 中 root xy/yaw 使用 env nominal pose + reset yaw noise，专家数据只提供关节/速度和 ref 跟踪目标。
39. **IsaacLab r_r mapping**: `r_r` 每一项必须在 §4.5 映射到具体 `mdp` reward 函数、`SceneEntityCfg`、body/joint regex 和 sensor 依赖。
40. **termination 数值化**: `root_height_below_min=0.30m`，`bad_orientation limit_angle=0.8rad`，严重 ground contact body regex 固定；table contact 永远只是 soft penalty，不作为 termination。
41. **train / deploy command 接口分离**: 训练端 synthetic sample；部署端接 `planner.py`，且 `solve_paddle_target` 默认 `target_land=(0.7,0,0.06)` 不可直接使用，必须显式传工程坐标目标点。
42. **PPO cfg**: 第一版直接复用 `tasks/mimic/agents/rsl_rl_ppo_cfg.py:BasePPORunnerCfg` 的 PPO 超参；pingpong 只覆盖 task entry / experiment name，actor/critic MLP 继续 `[512,256,128]`。
43. **训练 world frame 红线**: `p_hit_world.x=0.4`、`target_land_world=(2.45,0,0.78)`、table top `z=0.76` 是训练 world 的唯一标准；planner runtime frame 进入训练/部署接口前必须显式转换。
44. **debug draw 必做**: play smoke 中必须画 `p_hit_world / p_base_xy_world / n_target_world / v_racket_hat_world / blade normal`，并画 strike window 内 blade trajectory 到 hit point 的距离曲线。

来源标记: `✓ paper` = HITTER 原文；`△ DM` = paper 引 [35] DeepMimic；`[user-decided]` = 用户决定；`⚠️` = paper 未给、工程选择或 divergence。

---

## 1. Reward 总览表 [HITTER V-B] **[user-confirmed v4, 不变]**

| # | Reward | 属于 | 公式 | 计算 / 来源 (ref or target) | weight | σ / kernel | paper? | 时序 (gate) |
|---|---|:---:|---|---|:---:|---|:---:|---|
| 0 | `w_i, w_g, w_r` | 总 | `r = w_i·r_i + w_g·r_g + w_r·r_r` | Eq. 7 顶层 | **0.5 / 1.0 / 1.0** ⚠️ | — | ✓ V-B2 | — |
| 1 | `r^p` (joint pose) | r_i | `exp(-2·Σⱼ‖q̂ⱼ ⊖ qⱼ‖²)`, **J = upper-body \\ {right_wrist_roll_joint}** | sim `q[J]` vs ref `q̂[J]` (ref 跟随 cmd.swing_type 选 clip, 见 11.1) | **0.65** (DM 原值) | k=2 sum-form (DM) | △ DM | dense 全程 (ref 由 §11.1 双段插值，末端安全 clamp) |
| 2 | `r^v` (joint vel) | r_i | `exp(-0.1·Σⱼ‖q̇̂ⱼ − q̇ⱼ‖²)`, **J = upper-body \\ {right_wrist_roll_joint}** | sim `q̇[J]` vs ref `q̇̂[J]` (同上) | **0.10** (DM 原值) | k=0.1 sum-form (DM) | △ DM | dense 全程 (ref 由 §11.1 双段插值，末端安全 clamp) |
| 3 | `r^bp` (body pos, anchor-relative) | r_i | `exp(-k·Σ_b ‖p_rel[b] − p̂_rel[b]‖²)`, b ∈ ℬ_pos | sim: `p_rel[b] = (p_world[b].xy − pelvis.xy, p_world[b].z)` **[v5.2 #Q3: xy 减 z 保留, 防蹲下作弊]**; ref: 同公式 | **0.25** [user #5] | k=40 (DM r^e 风格) ⚠️ | [user-decided] | dense 全程 (ref 由 §11.1 双段插值，末端安全 clamp) |
| ~~r^e~~ | end-effector pos | — | **删除** | — | — | — | △ DM | (用户 #4) |
| ~~r^c~~ | COM pos | — | **替换为 r^bp** | — | — | — | △ DM | (用户 #5) |
| 4 | `r_g_pos` | r_g | `exp(-‖p_blade^base − p̂_racket^base‖²/σ²)` **base frame** [v5.5 A1] | sim: `p_blade^base = R_base^T·(p_blade_world − p_base_world)`; target: `p̂_racket^base = R_base^T·(cmd.p_hit_world − p_base_world)`. **数学等价于 world frame 计算 (旋转不变), base frame 写法更清晰, 与 obs #5 同一坐标** | **2.0** ⚠️ | σ=0.05 m (curriculum → 0.02, 8.1) | ✓ V-B2 | sparse `abs(t_to_hit) ≤ 0.06s` |
| 5 | `r_g_vel` | r_g | `exp(-‖v_blade^base − v̂_racket^base‖²/σ²)` **base frame** [v5.5 A1 一致] | sim: `v_blade^base = R_base^T · v_blade_world`; target: `v̂_racket^base = R_base^T · cmd.v_racket_hat_world`. 旋转不变, base 写法清晰 | **1.0** ⚠️ | σ=0.5 m/s | ✓ V-B2 | sparse 同上 |
| 6 | `r_g_ori` | r_g | `exp(-(1 − n_blade · n̂_target)²/σ²)` (cos 内积本身坐标无关) | `n_blade` = blade local +Y 旋到 world; **`n̂_target = cmd.n_target_world` (= paper Eq.5 paddle normal, 不再是 `v̂/‖v̂‖`)** | **0.5** ⚠️ | σ=0.2 (cos dist) | ✓ V-B2 + IV-C | sparse 同上 |
| 7 | `r_g_base` | r_g | `exp(-‖p_base_xy_world_sim − p_base_xy_world_cmd‖²/σ²)` **world frame** (base 自己 xy 在 world 才有意义) | sim: pelvis/root world xy; target: `cmd.p_base_xy_world` | **0.3** ⚠️ | σ=0.3 m | ✓ V-B2 | dense `t_to_hit > 0`, **OFF `t_to_hit ≤ 0`** |
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
| 5 | `p̂_racket` | 球拍目标位置 (**obs 端转 base-relative**, 内部存 world) | 3 | obs = `R_base^T · (cmd.p_hit_world − p_base_world)` [v5.3 #31] | ✓ | ✓ | ✓ |
| 6 | `v̂_racket` | 球拍目标速度 (world frame) | 3 | **cmd.v_racket_hat_world (= §3.2 Step 5 按 paper Eq.5+Eq.6 物理推导)** | ✓ | ✓ | ✓ |
| 7 | `t_to_hit` | **剩余击球时间 (秒)**, ✓ 命名统一 (cmd 内部和 obs 都叫 t_to_hit) | 1 | cmd.t_to_hit (initial = Δt_swing 采样, 每 step 递减 dt=0.02s) | ✓ | ✓ | ✓ |
| 8 | `q` | 关节位置 | **23** ⚠️ | sim joint encoder | ✓ | ✓ | ✓ |
| 9 | `q̇` | 关节速度 | **23** ⚠️ | sim joint encoder | ✓ | ✓ | ✓ |
| 10 | `a_last` | 上一 step 动作 | **23** ⚠️ | rollout buffer | ✓ | ✓ | ✓ |
| 11 | `v_base` | base 线速度 (privileged) | 3 | sim base lin vel | – | ✓ | ✓ |
| 12 | `T_B` | 跟踪 body pos+quat (**ref clip 由 cmd.swing_type 选定**, 见 11.1; **排除 `right_paddle_blade`** [v5.5 A2]) | 7·\|ℬ\|=11·7=**77** [v5.5 A2: 原 84, 减 paddle blade 的 7 维] | ref body world state at `clip[swing_type][ref_frame_f]` (浮点插值, 11.1.2) | – | ✓ | ✓ |
| 13 | `t_left` | episode 剩余时间 | 1 | `T_episode − t_now` | – | ✓ | ✓ |
| 14 | `[q̄, q̇̄]` | ref clip joint pos+vel (**clip 由 cmd.swing_type 选定**, 同上) | **46** ⚠️ | motion_loader at `clip[swing_type][ref_frame_f]` (浮点插值) | – | ✓ | ✓ |

**Actor = 86, Critic = 86 + 3 + 77 + 1 + 46 = 213** [v5.5 A2: 原 220, 减 7 维 paddle blade]

**v5.7 obs 强约束**: `v_ball_in_world` / `target_land_world` / `flight_time` / `paddle_cor` / `n_target_world` / `v_ball_out_world` / `t_pre_initial` / `t_post_swing` / `cur_step` / `swing_change_remaining` / `cmd.noise_*` 全部不作为网络输入。Actor 只看 Table I 的命令输出项，其中 #4–#7 可注入 frozen cmd noise；Critic 看 clean Table I，不额外加 4 项上游物理输入。

---

## 3. Command 总览表 **[v5.7: 物理推导 v̂ + base-frame swing_type + 1-change lock]**

### 3.0 训练 world frame 红线

训练端只认一个 world frame；所有 `cmd.*_world` 字段、reward sparse gate、debug draw、planner 部署接口都必须先落到这个 frame 再计算。paper 的桌面 frame 与 `planner.py` runtime frame 只在边界处转换，不允许在 `commands.py / rewards.py / observations.py` 内混用。

| 量 | 训练 world frame 标准值 / 方向 | 说明 |
|---|---|---|
| +z | 竖直向上 | IsaacLab world up |
| +x | 从机器人侧指向对方半桌 | 回球目标在 +x 方向 |
| table top | `z=0.76` | table cuboid center z=0.38, 高 0.76 |
| 近端桌沿 / 击球平面 | `x=0.4` | `p_hit_world.x` 固定为 0.4 |
| `p_hit_world` | `(0.4, hit_y, hit_z)` | 初始 `hit_y∈[0.05,0.25]`, `hit_z∈[0.95,1.15]`；课程最多扩展到 `hit_y∈[-0.65,0.65]`, `hit_z∈[0.85,1.25]` |
| `target_land_world` | `(2.45, 0.0, 0.78)` | 对方半桌中心 + 桌面高度 + 球半径 |
| robot reset nominal | env origin 附近, 朝 +x | root xy/yaw 不照搬 expert 世界坐标；见 §7.4 |

**实现红线**: `planner.py` 的默认 `target_land=(0.7,0,0.06)` 属于该 planner 的局部/旧默认语义，不能直接传进训练 world。部署时必须显式把 planner 输出的 `hit_pos / hit_vel / t_to_hit` 和目标落点转换到上表的训练 world 语义，再调用 Eq.5/Eq.6 或等价 `solve_paddle_target`。

### 3.1 Cmd 字段表 (内部 15 项，只有 Table I 对应项进 obs)

| # | 字段 | 维度 | frame | sample 时机 / 来源 | 取值范围 / 公式 | 进 Actor obs? | 进 Critic obs? | paper? |
|---|---|:---:|---|---|---|:---:|:---:|:---:|
| 1 | `swing_type` | 1 (cat) | — | swing 重采 + pre-strike 1-change lock (§3.3) | `forehand if hit_y_base > 0.157 else backhand` | ✗ (隐含 ref clip selector) | ✗ | △ deploy heuristic |
| 2 | `swing_change_remaining` | 1 (int) | — | resample 时设 1，首次变更后置 0 | `{0, 1}` | ✗ | ✗ | ⚠️ |
| 3 | `p_hit_world` | 3 | world | swing 重采时 sample | `x=0.4` 固定；初始 `y∈U[0.05,0.25]`、`z∈U[0.95,1.15]`；课程最多扩展到 `y∈U[-0.65,0.65]`、`z∈U[0.85,1.25]` | ✓ obs #5 (转 base-rel) | ✓ | ✓ V-B |
| 4 | `v_ball_in_world` | 3 | world | swing 重采时 sample | 初始 `mag∈U[2,4]`; 课程最多 `U[2,5.5]`; `yaw=π+U[-40°,40°]`; `pitch∈U[-75°,75°]` | ✗ | ✗ | ⚠️ R-1 |
| 5 | `target_land_world` | 3 | world | 常量 | `(2.45, 0.0, 0.78)` | ✗ | ✗ | ⚠️ R-2 |
| 6 | `flight_time` | 1 | — | swing 重采时 sample | `uniform[0.30, 0.65]` 秒 | ✗ | ✗ | ⚠️ R-3 |
| 7 | `paddle_cor` | 1 | — | 常量 | `0.85` (= paper Eq.6 restitution) | ✗ | ✗ | ✓ IV-C |
| 8 | `v_racket_hat_world` (= `v̂_racket`) | 3 | world | §3.2 Step 5 推导 | `v_pad_n * n_target_world` | ✓ obs #6 | ✓ | ✓ V-B + IV-C |
| 9 | `n_target_world` (= `n̂_target`) | 3 | world | §3.2 Step 5 推导 | `(v_ball_out_world - v_ball_in_world) / ||·||` | ✗ (reward 用) | ✗ | ✓ IV-C |
| 10 | `v_ball_out_world` | 3 | world | §3.2 Step 5 副产品 | `(target_land - p_hit)/T + (0,0,0.5*g*T)` | ✗ | ✗ | 内部 |
| 11 | `p_base_xy_world` | 2 | world | swing 重采或 swing_type 变更时计算 | `hit_xy_world - R(yaw_robot) @ expert_offset_base[swing_type]` | ✓ obs #4 | ✓ | ✓ V-B |
| 12 | `t_to_hit` | 1 | — | resample 时 = `t_pre_initial`; 每 step `-= dt` | `[-t_post_swing, t_pre_initial]` | ✓ obs #7 | ✓ | ✓ V-B |
| 13 | `t_pre_initial` | 1 | — | resample 时 sample | `truncN(0.20, 0.90, 0.30, 0.65)` 秒 | ✗ | ✗ | ⚠️ Q |
| 14 | `t_post_swing` | 1 | — | 常量 | 固定 `0.60s` | ✗ | ✗ | ⚠️ Q |
| 15 | `cur_step` | 1 | — | resample 时复位 0；每 step +1 | int `[0, (t_pre+t_post)/dt]` | ✗ | ✗ | 内部 |

**命名约定**:
- `p_hit_world` 就是 policy Table I 的 `p̂_racket` 目标位置，obs 端转成 base-relative。
- `v_racket_hat_world` 就是 Table I 的 `v̂_racket`，但它由 paper Eq.5+Eq.6 推导，不再直接采样。
- `n_target_world` 不进 obs，只供 `r_g_ori` 使用。
- `v_ball_in_world / target_land_world / flight_time / paddle_cor` 是上游物理输入，全部不进 obs。

### 3.2 Cmd 生成代码 (v5.7 Path B: Eq.5+Eq.6 inline，不 import planner)

**一次性 expert 预处理** (见 §11.4 的实测复算):
```python
expert_offset_base: Dict[str, np.ndarray] = {}
expert_pre_duration: Dict[str, float] = {}
expert_post_duration: Dict[str, float] = {}

for swing_type, npz_path in [("forehand",  ".../forward_001.npz"),
                              ("backhand", ".../backward_004.npz")]:
    d = np.load(npz_path)
    imp = int(d["impact_frame"][0])
    fps = int(d["fps"][0])                          # = 50

    p_pelv_w  = d["body_pos_w"][imp, PELVIS_IDX]    # PELVIS_IDX=0
    p_blade_w = d["body_pos_w"][imp, BLADE_IDX]     # BLADE_IDX=24
    q_pelv_w  = d["body_quat_w"][imp, PELVIS_IDX]   # wxyz
    yaw       = yaw_from_wxyz(q_pelv_w)
    diff_xy   = p_blade_w[:2] - p_pelv_w[:2]
    c, s      = np.cos(-yaw), np.sin(-yaw)
    expert_offset_base[swing_type] = np.array([
        c * diff_xy[0] - s * diff_xy[1],
        s * diff_xy[0] + c * diff_xy[1],
    ])

    expert_pre_duration[swing_type]  = imp / fps
    expert_post_duration[swing_type] = (d["joint_pos"].shape[0] - 1 - imp) / fps

# 实测: forehand (+0.496, +0.208), backhand (+0.428, +0.106)
Y_MID_BASE = 0.157
```

**训练端 `resample_cmd`**:
```python
def resample_cmd(env_id, cmd, robot, sigma_p_now=0.0, sigma_v_now=0.0, sigma_base_now=0.0, sigma_t_now=0.0):
    # Step 1: hit point sample (world frame)
    hit_x_world = 0.4
    hit_y_world = uniform(-1.0, 1.0)
    hit_z_world = uniform(0.08, 0.6)
    p_hit_world = np.array([hit_x_world, hit_y_world, hit_z_world])

    # Step 2: incoming ball velocity sample (world frame)
    v_in_mag   = uniform(2.0, 6.0)
    v_in_yaw   = np.pi + uniform(-deg2rad(40), deg2rad(40))
    v_in_pitch = uniform(-deg2rad(75), deg2rad(75))
    v_ball_in_world = v_in_mag * np.array([
        np.cos(v_in_yaw) * np.cos(v_in_pitch),
        np.sin(v_in_yaw) * np.cos(v_in_pitch),
        np.sin(v_in_pitch),
    ])

    # Step 3: target land 常量 (训练 world frame)
    target_land_world = np.array([2.45, 0.0, 0.78])
    # = 对方桌半区中心 (网 x=1.77, 远端 x=3.14, 中点≈2.45) + 桌面 z=0.76 + 球半径 0.02

    # Step 4: post-impact flight time
    flight_time = uniform(0.30, 0.65)

    # Step 5: paper Eq.5 + Eq.6 inline 推导 v_racket_hat_world 与 n_target_world
    paddle_cor = 0.85
    g, T = 9.81, flight_time
    v_ball_out_world = (target_land_world - p_hit_world) / T + np.array([0.0, 0.0, 0.5 * g * T])

    delta_v = v_ball_out_world - v_ball_in_world
    norm = np.linalg.norm(delta_v)
    if norm < 1e-9:
        # 退化极少见；实现端先 retry 5 次微扰 v_ball_in_world，仍失败才 fallback。
        n_target_world = np.array([-1.0, 0.0, 0.0])
        v_racket_hat_world = 2.0 * n_target_world
        last_resample_was_degenerate = True
    else:
        n_target_world = delta_v / norm
        v_in_n = float(v_ball_in_world @ n_target_world)
        v_out_n = float(v_ball_out_world @ n_target_world)
        v_pad_n = (v_out_n + paddle_cor * v_in_n) / (1.0 + paddle_cor)
        v_racket_hat_world = v_pad_n * n_target_world
        last_resample_was_degenerate = False

    # Step 6: swing_type base-frame 几何推导
    yaw_robot = quat_to_yaw(robot.data.root_quat_w[env_id])
    base_xy_w = robot.data.root_pos_w[env_id, :2].cpu().numpy()
    c, s = np.cos(-yaw_robot), np.sin(-yaw_robot)
    R_w2b = np.array([[c, -s], [s, c]])
    hit_xy_base = R_w2b @ (np.array([hit_x_world, hit_y_world]) - base_xy_w)
    hit_y_base = hit_xy_base[1]
    swing_type = "forehand" if hit_y_base > Y_MID_BASE else "backhand"
    swing_change_remaining = 1

    # Step 7: base target + 两段时间
    c, s = np.cos(yaw_robot), np.sin(yaw_robot)
    R_b2w = np.array([[c, -s], [s, c]])
    expert_offset_world = R_b2w @ expert_offset_base[swing_type]
    p_base_xy_world = np.array([hit_x_world, hit_y_world]) - expert_offset_world

    t_pre_initial = truncN(low=0.20, high=0.90, peak_low=0.30, peak_high=0.65)
    t_post_swing = 0.60

    cmd.swing_type = swing_type
    cmd.swing_change_remaining = swing_change_remaining
    cmd.p_hit_world = p_hit_world
    cmd.v_ball_in_world = v_ball_in_world
    cmd.target_land_world = target_land_world
    cmd.flight_time = flight_time
    cmd.paddle_cor = paddle_cor
    cmd.v_racket_hat_world = v_racket_hat_world
    cmd.n_target_world = n_target_world
    cmd.v_ball_out_world = v_ball_out_world
    cmd.p_base_xy_world = p_base_xy_world
    cmd.t_pre_initial = t_pre_initial
    cmd.t_post_swing = t_post_swing
    cmd.t_to_hit = t_pre_initial
    cmd.cur_step = 0
    cmd.last_resample_was_degenerate = last_resample_was_degenerate

    # v5.5 A10: cmd noise 每次 resample 采一次并冻结，整个 swing 不变。
    cmd.noise_p = clip(gauss(0, sigma_p_now, size=3), -3 * sigma_p_now, 3 * sigma_p_now)
    cmd.noise_v = clip(gauss(0, sigma_v_now, size=3), -3 * sigma_v_now, 3 * sigma_v_now)
    cmd.noise_base = clip(gauss(0, sigma_base_now, size=2), -3 * sigma_base_now, 3 * sigma_base_now)
    cmd.noise_t = clip(gauss(0, sigma_t_now, size=1), -3 * sigma_t_now, 3 * sigma_t_now)
```

reset 首帧有一个顺序特例: §7.4 先生成 clean cmd，再根据 `cmd.swing_type` 选 ref clip / 写 RSI 姿态；若 root pose 被实现路径改动，则重算 clean cmd 几何，最后再冻结 noise。普通 swing 重采样可以直接执行上面的 end-to-end `resample_cmd`。

**Obs 端坐标转换 (cmd → policy 输入)**:
```python
# Actor obs: noisy command, per-swing frozen
obs_actor.p_hit = R_base.T @ ((cmd.p_hit_world + cmd.noise_p) - p_base_world)
obs_actor.v_racket_hat_world = cmd.v_racket_hat_world + cmd.noise_v
obs_actor.p_base_err = (cmd.p_base_xy_world + cmd.noise_base) - p_base_xy_world
obs_actor.t_to_hit = cmd.t_to_hit + cmd.noise_t

# Critic obs: clean command
obs_critic.p_hit = R_base.T @ (cmd.p_hit_world - p_base_world)
obs_critic.v_racket_hat_world = cmd.v_racket_hat_world
obs_critic.p_base_err = cmd.p_base_xy_world - p_base_xy_world
obs_critic.t_to_hit = cmd.t_to_hit
```

**每 step 维护**:
```python
def cmd_step(cmd, robot, env_id, dt=0.02):
    cmd.t_to_hit -= dt
    cmd.cur_step += 1

    # v5.7: pre-strike 允许 1 次 swing_type 变更，防止 base 漂移导致首次判断锁错。
    if cmd.t_to_hit > 0.0 and cmd.swing_change_remaining > 0:
        new_swing = compute_swing_type_from_current_base(cmd.p_hit_world, robot, env_id)
        if new_swing != cmd.swing_type:
            cmd.swing_type = new_swing
            cmd.swing_change_remaining = 0
            cmd.p_base_xy_world = compute_base_target(cmd.p_hit_world, robot, env_id, new_swing)

    # 击球后 swing_type 锁死；post 段走完后立即重采。
    if cmd.t_to_hit <= -cmd.t_post_swing:
        resample_cmd(env_id, cmd, robot)
```

### 3.3 Cmd 生命周期 (双段时间 + cur_step + swing_type 1-change lock)

```
═══ episode reset ═══
│ 按 §7.4 生成首组 clean cmd → 选 ref/RSI → 必要时重算几何 → freeze per-swing noise
│ obs.t_to_hit = cmd.t_pre_initial
│ ref state 起点: get_ref_state(cmd, dt) at cur_step=0 → ref_frame_f=0
│
│ 每 step:
│   cmd.t_to_hit -= dt
│   cmd.cur_step += 1
│
│   if cmd.t_to_hit > 0 and cmd.swing_change_remaining > 0:
│       new_swing = compute_swing_type_from_current_base(cmd.p_hit_world, robot, env_id)
│       if new_swing != cmd.swing_type:
│           cmd.swing_type = new_swing
│           cmd.swing_change_remaining = 0
│           cmd.p_base_xy_world = recompute_base_target(new_swing)
│
│   ref 计算: get_ref_state(cmd, dt)
│       pre 段:  cur_step ∈ [0, sim_pre_steps]      → ref_frame_f ∈ [0, impact]
│       post 段: cur_step ∈ (sim_pre_steps, total]  → ref_frame_f ∈ (impact, clip_len-1]
│
│   abs(cmd.t_to_hit) <= 0.06s 时 r_g sparse 激活
│   cmd.t_to_hit > 0  时 r_g_base dense 激活
│   cmd.t_to_hit <= 0 时 r_g_base OFF，swing_type 锁死
│
├──── cmd.t_to_hit <= -cmd.t_post_swing ────
│    立即 resample_cmd；cur_step=0；swing_change_remaining=1
│
═══ 重复，直到 10s timeout / 摔倒 termination ═══
```

### 3.4 训练 / 部署 command 接口差异

训练端 `commands.py` 不 import `planner.solve_paddle_target`，但部署端可以复用 planner 的弹道预测和 Eq.5/Eq.6 求解；两者的接口边界必须写清楚，防止坐标系和默认参数偷渡。

| 字段 / 步骤 | train (`commands.py`) | deploy (`planner.py` / runtime bridge) | 约束 |
|---|---|---|---|
| `p_hit_world` | synthetic sample: `(0.4, hit_y, hit_z)` | from `HitterPlanner.predict_hit_plane(...).hit_pos` | 进入 policy 前必须转换到 §3.0 训练 world |
| `v_ball_in_world` | synthetic sample: `v_in_mag/yaw/pitch` | from planner predicted `hit_vel` | 不进 obs，只用于 Eq.5/Eq.6 |
| `target_land_world` | fixed `(2.45,0,0.78)` | runtime 明确传入，禁止用 `solve_paddle_target` 默认值 | 默认 `(0.7,0,0.06)` 不可用 |
| `flight_time` | `uniform[0.30,0.65]` | planner / strategy 给定；若无则沿用训练范围 sample | 单位秒 |
| `t_to_hit` | sample `t_pre_initial`，每 step 递减 | from planner prediction | obs 字段仍叫 `t_to_hit` |
| `v_racket_hat_world` | inline Eq.5/Eq.6 | `solve_paddle_target` 或同公式 inline | 输出必须在训练 world |
| `n_target_world` | `(v_ball_out_world-v_ball_in_world)/||·||` | 同左 | `r_g_ori` 只读此字段 |
| `swing_type` | base-frame `Y_MID_BASE=0.157` + 1-change lock | 仍用当前 robot base-frame heuristic | 不使用 planner 的旧 world-y 方向判断训练 policy |

部署 bridge 的最低单元测试: 给定同一组 `p_hit_world / v_ball_in_world / target_land_world / flight_time / paddle_cor`，部署路径和训练路径输出的 `v_racket_hat_world / n_target_world / v_ball_out_world` 必须逐元素一致到 `1e-5`。

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

### 4.5 IsaacLab term mapping (r_r 落地表)

实现时 `r_g_* / r_i` 为 pingpong local reward；`r_r` 尽量复用 IsaacLab / 现有 locomotion-mimic mdp term。下表是第一版落地映射，若上游 IsaacLab 函数名在本机版本不同，只允许在 `pingpong/mdp/rewards.py` 写同名薄 wrapper，保持 env_cfg 语义不变。

| r_r sub-term | env_cfg term name | func | params / SceneEntityCfg | sensor | 备注 |
|---|---|---|---|---|---|
| `alive_reward` | `alive` | `mdp.is_alive` 或 local `alive_reward` | — | — | 每 step +1，再乘 weight `+0.1` |
| `action_rate_l2` | `action_rate_l2` | `mdp.action_rate_l2` | — | — | IsaacLab 标准项 |
| `action_l2` | `action_l2` | `mdp.action_l2` | — | — | IsaacLab 标准项 |
| `dof_torques_l2` | `joint_torque` | `mdp.joint_torques_l2` | `asset_cfg=SceneEntityCfg("robot", joint_names=[".*"])` | — | 与现有 mimic/locomotion 命名对齐 |
| `dof_acc_l2` | `joint_acc` | `mdp.joint_acc_l2` | `asset_cfg=SceneEntityCfg("robot", joint_names=[".*"])` | — | 与现有 mimic/locomotion 命名对齐 |
| `dof_pos_limits` | `joint_limit` | `mdp.joint_pos_limits` | `asset_cfg=SceneEntityCfg("robot", joint_names=[".*"])` | — | hinge 超限惩罚 |
| `pelvis_orientation_l2` | `pelvis_orientation_l2` | local `pelvis_orientation_l2` | `asset_cfg=SceneEntityCfg("robot")` | — | `proj_g_x^2 + proj_g_y^2` |
| `pelvis_height` | `pelvis_height` | `mdp.base_height_l2` 或 local wrapper | `target_height=0.74`, `asset_cfg=SceneEntityCfg("robot")` | — | 单段 weight `-10.0` |
| `feet_slide` | `feet_slide` | `mdp.feet_slide` | `asset_cfg=SceneEntityCfg("robot", body_names=".*ankle_roll.*")` | `contact_forces` 同 body regex | 只在脚接触时罚水平速度 |
| `feet_air_time` | `feet_air_time` | `mdp.feet_air_time_positive_biped` 或 local no-command wrapper | `threshold=0.4`, `sensor_cfg=SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")` | `contact_forces` | 若内置函数强依赖 locomotion command，pingpong wrapper 去掉 command gate |
| `undesired_contacts` | `undesired_contacts` | `mdp.undesired_contacts` | `threshold=1.0`, `sensor_cfg=SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*)(?!.*wrist_roll_rubber_hand.*)(?!right_paddle_blade$).*"])` | `contact_forces` | 软惩罚；脚、手腕胶手、paddle blade 排除 |
| `r_table_contact` | `table_contact` | local `robot_table_contact_penalty` | `threshold=1.0`, `sensor_cfg=SceneEntityCfg("robot_table_contact", body_names=".*")` | `robot_table_contact` | soft penalty，永不 terminate |

`contact_forces` 用于 feet / undesired ground contact；`robot_table_contact` 只过滤 Table。两套 sensor 不混用，避免 table contact 被 ground-contact regex 误吞。

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
| 2 | `bad_orientation` | `limit_angle=0.8 rad`，等价近似 `‖proj_g_xy‖ > sin(0.8)=0.717` | ✗ | 摔倒 / 倾倒 | [我提案] |
| 3 | `root_height_below_min` | `pelvis_height < 0.3` m | ✗ | 摔到地上 | [我提案] |
| 4 | `undesired_contact_terminate` | `pelvis` / `torso_link` / `head_link` / `.*_hip_pitch_link` 触地且力 `>1.0N` (URDF 验证, 见 9.3 路径) | ✗ | 严重摔倒 (头/胯/髋触地) | [我提案] |

#### 6.1.1 Termination 落地参数

| env_cfg term | func | params | 说明 |
|---|---|---|---|
| `time_out` | `mdp.time_out` | `time_out=True` | 10s episode length 由 env cfg 控制 |
| `bad_orientation` | `mdp.bad_orientation` | `limit_angle=0.8` | 与 locomotion baseline 对齐；不要再叠加第二套 roll/pitch hard threshold |
| `root_height_below_min` | `mdp.root_height_below_minimum` | `minimum_height=0.30` | G1 23DoF + paddle 第一版取 0.30m |
| `undesired_contact_terminate` | `mdp.illegal_contact` 或 local `hard_undesired_contact` | `threshold=1.0`, `sensor_cfg=SceneEntityCfg("contact_forces", body_names=["pelvis", "torso_link", "head_link", ".*_hip_pitch_link"])` | 只处理严重摔倒；膝/肘/腕等一般接触留给 `r_r.undesired_contacts` 软惩罚 |

**table contact 永不 termination**: robot ↔ table 接触只进入 `r_table_contact` soft penalty。即使球拍碰桌，也不触发 `DoneTerm`，否则 policy 容易因为一次探索碰撞就失去击球窗口学习信号。

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

#### 7.1.1 落地实现表 (v5.7 → v5.8 DR 完整化)

每条 paper §V-B3 DR 在代码中的具体落点；目标文件 `tasks/pingpong/robots/g1_23dof/hitter/hitter_env_cfg.py::EventCfg`。前两条直接调用 IsaacLab 内置 API，后四条需要额外说明。

| # | 项 | EventTerm 名 | `func` | 关键 params | 落地文件 |
|---|---|---|---|---|---|
| 1 | add_link_mass | `add_link_mass` | `mdp.randomize_rigid_body_mass` | `body_names=".*"`, `mass_distribution_params=(0.9, 1.1)`, `operation="scale"`, `distribution="uniform"` | IsaacLab |
| 2 | link friction | `physics_material` (已存在) | `mdp.randomize_rigid_body_material` | `static_friction_range=(0.3, 1.6)`, `dynamic_friction_range=(0.3, 1.2)`, `restitution_range=(0.0, 0.5)` | IsaacLab |
| 3 | joint friction | `randomize_joint_friction` | `mdp.randomize_joint_parameters` | `joint_names=[".*"]`, `friction_distribution_params=(0.5, 1.5)`, `operation="scale"`, `distribution="uniform"` | IsaacLab |
| 4 | joint damping | `randomize_joint_damping` | `mdp.randomize_actuator_gains` | `joint_names=[".*"]`, `damping_distribution_params=(0.7, 1.3)`, `operation="scale"`, `distribution="uniform"` | IsaacLab |
| 5 | IMU offset | `randomize_imu_offset` | `mdp.randomize_imu_offset` (custom) | `sigma_deg=2.0`, `distribution="gaussian"` | `pingpong/mdp/events.py` |
| 6 | comm delay | `randomize_comm_delay` | `mdp.randomize_comm_delay` (custom) | `max_delay_steps=1` (与 step_dt=20 ms 配合 → 整型 0/1 step) | `pingpong/mdp/events.py` |

> 工程注 (#3 vs #4): IsaacLab 把 "joint friction (静/动/粘性)" 暴露在 `randomize_joint_parameters.friction_distribution_params`，把 "joint damping (PD damping)" 暴露在 `randomize_actuator_gains.damping_distribution_params`。两者互不重叠：前者改 PhysX joint friction coefficient，后者改 actuator PD damping。命名与 paper 一致即可。
>
> 工程注 (#1 amplitude): paper 没明确给数值，我们采纳常见 humanoid 训练实践 `±10%`；如复现指标偏弱可放宽到 `±20%` (`(0.8, 1.2)`)。
>
> 工程注 (#5/#6 自定义事件): IsaacLab 5.1 不内置 IMU calibration offset 和 obs-side communication delay。本节落地要求新增两条 startup event；实现见 §7.1.2。

#### 7.1.2 自定义 startup event (`pingpong/mdp/events.py`) [新, v5.8]

两条 startup event 在 env 上写入持久 buffer，由 `pingpong/mdp/observations.py` 的 wrapper 函数读取并作用到 **Actor obs only**, Critic obs 保持 clean (与 §7.3.2 一致的 asymmetric AC 约定)。

##### `randomize_imu_offset(env, env_ids, asset_cfg, sigma_deg=2.0, distribution="gaussian")`

- 行为: 每 env 采一个体素小角 Euler `(rx, ry, rz) ~ N(0, σ_deg)`，转 `quat_from_euler_xyz(rx, ry, rz)` 写到 `env._pingpong_imu_offset_quat[num_envs, 4]` (wxyz)。
- 默认 identity: 若事件未配置，wrapper helper `get_imu_offset_quat(env)` 返回 (1, 0, 0, 0) 单位四元数 buffer，行为退化为无随机化；不影响 Critic / r_g / gate。
- 注入位置 (Actor only):
  - `obs.base_yaw = base_yaw_encoding_imu(env)` (perceived quat = `quat_mul(root_quat_w, q_offset)`)
  - `obs.projected_gravity = projected_gravity_imu(env)` (`quat_apply_inverse(perceived_quat, GRAVITY_VEC_W)`)
  - `obs.base_ang_vel = base_ang_vel_imu(env)` (`quat_apply_inverse(q_offset, root_ang_vel_b)`)
- 不变: `pingpong_hit_position_b` 仍读 `cmd.robot.data.root_quat_w` 真值 — `p̂_racket` 是 cmd 几何，IMU offset 不污染 cmd 噪声 channel。

##### `randomize_comm_delay(env, env_ids, max_delay_steps=1)`

- 行为: 每 env 采 `delay_steps ~ Uniform{0, ..., max_delay_steps}`，写到 `env._pingpong_obs_delay_steps[num_envs]`。
- 与 step_dt=20 ms 配合: `max_delay_steps=1` ⇒ 一半 env 当前步 (delay 0)，一半 env 上一步 (delay 1, 即 20 ms 通信延迟); 物理意义对应 paper "0–20 ms" 区间的两端。
- 注入位置 (Actor only): 由 `DelayedObservation(ManagerTermBase)` 类包裹任一 inner ObsTerm。每 step 先调 inner 拿当前值，再按 `delay > 0` mask 与上一步 buffer 切换；返回后更新 buffer。
- Actor 包裹清单 (Critic 全部直读真值):
  - `base_ang_vel_imu`, `projected_gravity_imu`, `base_yaw_encoding_imu` (sensor channel)
  - `joint_pos_rel`, `joint_vel_rel` (encoder channel)
  - **不包裹** `last_action` (action 是 actor 自己的输出, 无通信链路)；**不包裹** cmd 几何 4 项 (`base_err`/`hit_pos`/`racket_vel`/`t_to_hit`, 它们走 §7.3.2 cmd noise 通道)
- 边界: 第一帧 buffer 初始化为当前值 (delay 0 即起步无延迟)；env reset 时由 `DelayedObservation.reset(env_ids)` 把对应 env buffer 清零下次自动重填。

##### Actor / Critic obs 路由总表 (v5.8 DR 完整后)

| obs 项 | Actor (PolicyCfg) | Critic (CriticCfg) |
|---|---|---|
| `base_ang_vel` | `DelayedObservation(base_ang_vel_imu)` | `mdp.base_ang_vel` (clean) |
| `projected_gravity` | `DelayedObservation(projected_gravity_imu)` | `mdp.projected_gravity` (clean) |
| `base_yaw` | `DelayedObservation(base_yaw_encoding_imu)` | `mdp.base_yaw_encoding` (clean) |
| `base_err` | noisy cmd (§7.3.2) | clean cmd |
| `hit_pos` | noisy cmd (§7.3.2) | clean cmd |
| `racket_vel` | noisy cmd (§7.3.2) | clean cmd |
| `t_to_hit` | noisy cmd (§7.3.2) | clean cmd |
| `joint_pos` | `DelayedObservation(joint_pos_rel)` | `mdp.joint_pos_rel` (clean) |
| `joint_vel` | `DelayedObservation(joint_vel_rel)` | `mdp.joint_vel_rel` (clean) |
| `last_action` | `mdp.last_action` (no delay, 见上) | `mdp.last_action` |
| `base_lin_vel` | — | `mdp.base_lin_vel` |
| `ref_body_state` | — | `mdp.pingpong_ref_body_state` |
| `time_left` | — | `mdp.episode_time_left` |
| `ref_joint_state` | — | `mdp.pingpong_ref_joint_state` |

⚠️ **DIVERGENCE Q [新, v5.8 DR]**: paper §V-B3 仅列 6 项 DR 范围、未指定 IMU offset 是 per-step 抖动还是 startup 一次冻结、也未指定 comm_delay 是 per-step 抽样还是 startup 一次冻结。本工程**两者都做 startup 一次冻结**，理由: (a) IMU 标定误差在硬件上是装机 / 上电后基本恒定的偏移；(b) 通信链路在一次 deployment 内是稳定的拓扑，per-step 抖动会让 Actor 学不到稳定的因果关系，与 §7.3.2 cmd noise 的 per-swing 冻结理由一致。如复现侧后续要 per-step 噪声，仅需把 EventTerm `mode="startup"` 改成 `mode="reset"` 或 `mode="interval"` 即可，wrapper / buffer 结构不动。

#### 7.1.3 工程附加 DR (v5.8 在 paper §V-B3 之外加的, 风险可控小幅)

| # | 项 | 范围 | 落地 | 说明 |
|---|---|---|---|---|
| 7 | `paddle_cor` 每 swing 重采 | uniform `U(0.80, 0.90)` | `commands.py::_resample_command_internal`, `cfg.paddle_cor_range=(0.80, 0.90)` | 模拟橡胶老化 / 球速度不同导致的 e 漂移；不做 curriculum, 一开始就开 |

**实现说明**: `cfg.paddle_cor: float = 0.85` 仍保留为初始 buffer 值 (constructor 用), 每次 cmd 重采样 (= 每个 swing) 时从 `cfg.paddle_cor_range` 采新值, 写入 `cmd.paddle_cor[ids]` 后再调 `_solve_paddle_target`。这一项**不进 obs** (与 v5.7 Q-N3 / 用户 ② / ③ 的 "上游物理输入不进 obs" 约定一致), 但因 `paddle_cor` 是 Eq.6 的物理参数, 它的扰动会直接影响 `v_racket_hat_world`, 从而让 policy 学到对球拍 e 漂移鲁棒的击球速度。

> **回顾 Q-N3**: v5.7 锁定 "paddle_cor = 0.85 写死 (不进 DR)" 是基于 paper §IV-C 的常量假设; v5.8 在不破坏 Eq.5 / Eq.6 物理一致性的前提下放开为小区间 `±0.05` 绝对值 (≈ ±5.9% 相对), 仍远小于 paper §IV-C 给的 e 标定不确定度区间, 不会影响 deploy bridge 单元测试 (deploy 只需 cor 是已知数值, 区间内任何值都满足 paper Eq.5 / Eq.6)。

### 7.2 Mode `reset` (每 episode, RSI + 首组 cmd 几何推导 swing_type)

| # | 项 | 范围 | 备注 | paper? |
|---|---|---|---|:---:|
| 1 | `reset_robot_pose` (RSI) | 见 7.4 | 必从 ref clip 内随机帧起步 [v5.5 A7: 删 mimic_start_prob 分支] | [paper-derived] |
| 2 | `reset_joint_pos_noise` | gauss `σ=0.05` rad | 23 dof 独立 | [我提案] |
| 3 | `reset_base_yaw_noise` | uniform `±10°` | 防 yaw 锁死 | [我提案] |
| 4 | `sample_first_cmd` | §7.4 顺序 + §3.2 clean cmd | reset nominal root 后生成首组 clean cmd；`swing_type` 由 §3.2 Step 6 的 base-frame 几何阈值推导；noise 在 ref/RSI consistency 确认后冻结 | [user-decided v5.7] |

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
# (1) cmd 重采时 (commands.py.resample_cmd), 先执行 §3.2 七步生成 clean cmd，随后一次性冻结 noise:
def resample_cmd(cmd, σ_p_now, σ_v_now, σ_base_now, σ_t_now):
    # §3.2 Step 1–7:
    #   p_hit_world / v_ball_in_world / target_land_world / flight_time / paddle_cor
    #   → v_racket_hat_world / n_target_world / v_ball_out_world
    #   → swing_type(base-frame Y_MID_BASE) / p_base_xy_world / t_pre_initial / t_post_swing / cur_step
    cmd.noise_p    = clip(gauss(0, σ_p_now,    size=3), -3*σ_p_now,    3*σ_p_now)
    cmd.noise_v    = clip(gauss(0, σ_v_now,    size=3), -3*σ_v_now,    3*σ_v_now)
    cmd.noise_base = clip(gauss(0, σ_base_now, size=2), -3*σ_base_now, 3*σ_base_now)
    cmd.noise_t    = clip(gauss(0, σ_t_now,    size=1), -3*σ_t_now,    3*σ_t_now)

# (2) Actor obs (observations.py.actor_obs), 注入 noise:
obs_actor.p̂_racket  = R_base^T · ((cmd.p_hit_world + cmd.noise_p) − p_base_world)
obs_actor.v̂_racket  =              cmd.v_racket_hat_world + cmd.noise_v
obs_actor.base_err  = (cmd.p_base_xy_world + cmd.noise_base) − p_base_xy_world
obs_actor.t_to_hit  = cmd.t_to_hit + cmd.noise_t

# (3) Critic obs / r_g / r_g_base / gate (rewards.py + observations.py.critic_obs):
obs_critic.p̂_racket = R_base^T · (cmd.p_hit_world − p_base_world)   # clean
r_g_pos = exp(-‖p_blade^base − cmd.p_hit^base‖² / σ²) · 𝟙[abs(cmd.t_to_hit)<=0.06]   # clean
r_g_base = exp(-‖p_base_xy_world − cmd.p_base_xy_world‖² / σ²)        # clean
```

⚠️ **DIVERGENCE R [新, v5.5 A10]**: paper §V-B3 列了 "obs noise" 但**未明** noise 是 per-step 还是 per-swing 冻结. 我们选 per-swing 冻结是用户决定 — 物理直觉: 感知系统给球的预测在一次击球内是相对稳定的 (球的轨迹外推一旦计算就不会大幅抖动), per-step 重采反而会让 Actor 学不到稳定的因果. Critic clean 是 asymmetric AC 标准做法 (privileged critic), 与 paper §V-B "value head sees full state" 一致.

#### 7.3.3 reward / gate 端的真值约定 [v5.5 A10 强约束]

实现端必须严格遵守:

- **r_g_pos / r_g_vel / r_g_ori** 计算用 `cmd.p_hit_world` / `cmd.v_racket_hat_world` (无噪声真值)
- **r_g_base** 计算用 `cmd.p_base_xy_world` (无噪声真值)
- **strike window gate** `abs(cmd.t_to_hit) <= 0.06s` 用 `cmd.t_to_hit` (无噪声真值)
- **重采样边界** `cmd.t_to_hit <= -cmd.t_post_swing` 用真值 (无噪声), 防止 noise 把重采时间提前/推后
- **Critic obs (12 项 #4–#7 / #11–#14)** 全部 clean
- **Actor obs (10 项 #4–#7 + #1–#3 / #8–#10)** 仅 #4 / #5 / #6 / #7 用 noisy cmd, 其他 (`q`, `q̇`, `a_last`, IMU) 是 sensor 真值 (走 `randomize_imu_offset` + `comm_delay` 那条 startup DR, 不在 cmd noise 范畴)

⚠️ 如果实现端不小心让 r_g 看到 noisy cmd, reward landscape 会被噪声扰动, σ_g_pos curriculum 会失效. **代码 review 必须确认**: 任何 reward 函数读 cmd 时都不读 `cmd.noise_*`.

### 7.4 RSI [v5.5 A7 简化: 删除 mimic_start_prob 分支, 每个 episode 必从 ref clip 起步]

**v5.5 A7 用户决定**: 删除 `mimic_start_prob` 与 `pose_src ~ {ref_clip, default_standing}` 双分支. 整个工程**每个 episode 都有专家数据参与**, 不存在 "free 起步" 概念. RSI 直接从 ref clip 内随机帧取专家关节/速度作为物理初态多样性。

**v5.7 root reset 锁定**: 不直接复制专家世界 root xy/yaw 到仿真 robot root。训练 world 的 root xy/yaw 使用 env nominal pose + reset yaw noise；专家 clip 只提供 joint pos / joint vel / 可选 root z 与 root lin/ang vel 的相对初始化参考，以及后续 `r_i / T_B / q_ref` 的 ref target。这样 `p_hit_world / p_base_xy_world` 的几何关系始终由训练 world command 定义，而不是被专家 npz 的世界坐标污染。

```python
# Step 0: reset root to training nominal pose, not expert world root xy/yaw
root_xy_world = env_cfg.training_nominal_root_xy          # usually near env origin
root_yaw_world = env_cfg.training_nominal_yaw + uniform(-10deg, +10deg)
root_z_world = env_cfg.training_nominal_root_z            # pelvis standing height, around 0.74
robot.write_root_pose(root_xy_world, root_z_world, root_yaw_world)

# Step 1: generate clean cmd using current root pose (§3.2)
# No command noise is frozen yet; this phase only computes clean task geometry.
cmd = sample_clean_cmd_from_section_3_2(root_pose=robot.root_pose)

# Step 2: choose ref clip from cmd.swing_type
ref_clip = expert_clip[cmd.swing_type]  # forward_001 or backward_004 (见 11.4)

# Step 3: RSI pose diversity from selected clip, but keep training-world root xy/yaw
f_init ~ uniform(0, T_clip - 1)         # T_clip = ref_clip.length (forward=82, backward=64)
robot.write_joint_state(ref_clip[f_init].joint_pos, ref_clip[f_init].joint_vel)
robot.write_root_velocity(ref_clip[f_init].root_lin_vel_rel, ref_clip[f_init].root_ang_vel_rel)  # yaw-aligned if used
# root xy/yaw remain the training nominal values from Step 0.

# Step 4: if any implementation path touched root pose, recompute clean cmd-dependent geometry once
if root_pose_changed_after_step1:
    recompute_hit_y_base_and_swing_type(cmd, root_pose=robot.root_pose)
    recompute_p_base_xy_world(cmd, root_pose=robot.root_pose)

# Step 5: ref time starts at clip head so cmd.t_to_hit and ref progress align
cmd.cur_step = 0

# Step 6: freeze per-swing command noise after clean cmd/ref consistency is final
freeze_cmd_noise(cmd)
```

⚠️ **DIVERGENCE M — RSI 解耦 (保留 v5.4 语义, v5.7 明确 root 规则)**: paper / DM 标准 RSI 是 "物理姿态 = ref 同帧, ref 进度跟随". 我们解耦 "姿态采样帧 f_init" vs "ref 时间 cur_step". `cur_step=0` 强制 ref 进度从 clip 头开始, 因为 cmd.t_to_hit 也是从 0 起算, ref clip 进度需与 cmd 时间同步. 物理关节姿态 ref[f_init] 给出多样性起手位 (类似 DM RSI 的随机起步), ref 跟踪信号则统一从 clip[0] 开始；root xy/yaw 则服从训练 world nominal pose，不服从专家 npz world pose。

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

#### 8.1.1 击球成功率定义 (无 ball rollout 版)

训练环境第一版没有真实乒乓球 rollout / 碰撞结果，因此 curriculum 的 `hit_success` 只由机器人末端在 strike window 内的几何达成度定义，不从 reward 值、event flag 或物理球碰撞结果反推。

每个 swing 的 strike window 为 `abs(cmd.t_to_hit) <= 0.06s`。若窗口内存在任意一帧同时满足以下三个条件，则该 swing 记为 `hit_success=1`；否则 `hit_success=0`:

```python
pos_ok = norm(p_blade_base - p_hit_base) < success_pos_thresh
vel_ok = norm(v_blade_base - v_racket_hat_base) < success_vel_thresh
ori_ok = (1.0 - dot(n_blade_world, cmd.n_target_world)) < success_ori_cos_dist_thresh
hit_success = any_over_strike_window(pos_ok & vel_ok & ori_ok)
```

success 判据使用独立阈值，不复用 reward kernel σ:

| 阈值 | 数值 | 说明 |
|---|---|---|
| `success_pos_thresh` | `max(0.06, 2 * sigma_g_pos_now)` | σ 收紧时仍保留 6cm 统计容差，避免 success rate 抖到不可用 |
| `success_vel_thresh` | `1.0 m/s` | 与 `r_g_vel σ=0.5` 解耦 |
| `success_ori_cos_dist_thresh` | `0.25` | 即 `dot(n_blade_world, n_target_world) > 0.75` |

1k iter sliding window 的“击球成功率”定义为该窗口内所有完成 swing 的 `mean(hit_success)`。如果一个 episode timeout 时最后一个 swing 尚未走完 post 段，但已经经历过 strike window，则仍计入统计；若尚未进入 strike window，则不计入分母。

**单调性约束**: σ_g_pos **只升不降** (= 只收紧不放宽). 即使下一段成功率回落, σ_g_pos 也不返回上一段值. 防止与 8.3 噪声耦合发散.

### 8.2 [v5.5 A7 删除] mimic_start_prob curriculum（已删除）

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
    cmd.swing_type / cmd.p_hit_world / ... = sample_§3.2(...)
    # 同时把噪声采一次冻结到 cmd 字段:
    cmd.noise_p    = clip(gauss(0, σ_p_now,    size=3), -3*σ_p_now,    3*σ_p_now)
    cmd.noise_v    = clip(gauss(0, σ_v_now,    size=3), -3*σ_v_now,    3*σ_v_now)
    cmd.noise_base = clip(gauss(0, σ_base_now, size=2), -3*σ_base_now, 3*σ_base_now)
    cmd.noise_t    = clip(gauss(0, σ_t_now,    size=1), -3*σ_t_now,    3*σ_t_now)

# (c) 每 step Actor obs 端用冻结的 noise 直接加 (不再每 step 重新 gauss):
obs_actor.p̂_racket = R_base^T · ((cmd.p_hit_world + cmd.noise_p) − p_base_world)
obs_actor.v̂_racket = cmd.v_racket_hat_world + cmd.noise_v
obs_actor.p̂_base_err = (cmd.p_base_xy_world + cmd.noise_base) − p_base_xy_world
obs_actor.t_to_hit = cmd.t_to_hit + cmd.noise_t

# (d) Critic obs / r_g / strike_window_gate 全部用 cmd.* 真值, 不读 cmd.noise_*  (见 §7.3.3 truth-value contract)
```

⚠️ 与 v5.2 的差异: v5.2 写的"每 step 调用 cmd_noise_obs(cmd, σ_now)"被 [v5.5 A10] 否决. 关键原因: paper 没明确指定 noise 频率, 但物理上 perception (球桌 / 来球) 是按"事件" (= swing 任务下达) 一次性给定的, 不是每控制 step 重新采样. 把噪声跟 swing 绑定 (per-swing freeze) 比 per-step gauss 更接近真实部署. 详见 §7.3.2 DIVERGENCE R.

**为何 σ_t 触发更早 (≥50%)**: 时间扰动比位置更"基础" (perception 总有几 ms 偏差), 早开始训练对 t_to_hit 的鲁棒性.

### 8.4 Hit point / v_in_mag 范围扩展 curriculum [v5.2 修订: 改成击球成功率驱动]

**类型**: uniform 采样区间上下界. 范围 monotone **扩展** (不收紧).

**触发原则** (用户决定): 也用击球成功率, 与 8.1/8.3 统一 metric.

| 参数 | 初始 (≥0%) | ≥30% | ≥50% | ≥75% (终值) |
|---|---|---|---|---|
| `hit_y` | `[0.05, 0.25]` m | linear → `[-0.15, 0.35]` | linear → `[-0.35, 0.55]` | linear → **`[-0.65, 0.65]`** m |
| `hit_z` | `[0.95, 1.15]` | → `[0.92, 1.18]` | → `[0.88, 1.22]` | → **`[0.85, 1.25]`** |
| `v_in_mag` | `[2.0, 4.0]` m/s | → `[2.0, 4.5]` | → `[2.0, 5.0]` | → **`[2.0, 5.5]`** m/s |

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
| 6 | hit_y range | uniform 区间 | `[0.05,0.25]` → **`[-0.65,0.65]`** m | 击球成功率 (30/50/75%) | 扩展 | `update_hit_y_range` |
| 7 | hit_z range | uniform 区间 | `[0.95,1.15]` → **`[0.85,1.25]`** | 击球成功率 (30/50/75%) | 扩展 | `update_hit_z_range` |
| 8 | v_in_mag range | uniform 区间 | `[2.0,4.0]` → **`[2.0,5.5]`** m/s | 击球成功率 (30/50/75%) | 扩展 | `update_v_in_mag_range` |

**统一原则** (v5.5): 所有 curriculum 都用**击球成功率**作为 metric (8.2 mimic_start_prob 删除后, 不再有 survival 驱动的课程), 阶梯阈值 30/50/65/75/80% 之间互相协调:
- 击球成功率 ≥30%: σ_g_pos stage1 (→0.06) + 范围 stage1 扩展
- ≥50%: σ_g_pos stage2 (→0.04) + 范围 stage2 + **σ_t 触发**
- ≥65%: σ_g_pos stage3 (→0.03)
- ≥75%: 范围全开 + **σ_p/σ_v/σ_base 触发**
- ≥80%: σ_g_pos stage4 (→0.02 终值)

⚠️ 8.4 三条**不**改单调 — 范围本来就是收紧→放宽, 与 σ_g_pos 收紧方向相反.

### 8.6 第一轮训练只开

只开 **#1 σ_g_pos** 一条 (核心难度) [v5.5 A7: 原来 #1 + #2, 删 #2 后只剩 #1]. 其他 (cmd noise + 范围扩展) 留作 ablation; 训练第一轮使用初始可学范围 (`[0.05,0.25]` / `[0.95, 1.15]` / `[2.0, 4.0]`) 直接训，验证 baseline 稳定后再逐步开启范围扩展和 cmd noise。不得直接使用旧终值 (`±1.0` / `[0.08,0.60]` / `[2.0,6.0]`)。

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

1. cmd 加新字段 `t_post_swing` (post-strike sim 时长), **固定为 `0.60s`**, 不进 obs
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
        resample_cmd(cmd)          # p_hit / v_ball_in / v̂ / n̂ / swing_type / t_pre_initial / t_post_swing / noise 全部重采
        cmd.cur_step = 0           # ref 从 clip[0] 重新对齐
```

注意: cmd.t_to_hit 在 pre-strike 阶段从 t_pre_initial 单调减到 0, post-strike 阶段继续减到 -t_post_swing. **strike window gate `abs(cmd.t_to_hit) ≤ 0.06s` 不变** — 仍以 t_to_hit=0 为中心 ±3 帧, 自然落在 sim_pre_steps 与 sim_post_steps 交界处 (= ref 的 impact 帧附近).

#### 11.1.4 数字举例 [v5.7: 使用新 t_pre/t_post 范围]

**例 A — 长 pre + 短 post，backhand**:
```
swing_type      = backhand          (clip_len=64, impact=20, pre=0.40s, post=0.86s native)
t_pre_initial   = 0.90s             (sim_pre_steps = 45)
t_post_swing    = 0.40s             (sim_post_steps = 20)
total swing     = 1.30s             (65 step)，然后重采样

step  cur_step  t_to_hit  ref_frame_f                                  说明
─────────────────────────────────────────────────────────────────────────────────
   0       0     0.90    progress=0/45=0,      ref_f = 0.0            clip[0], 起手
  22      22     0.46    progress=22/45=0.489, ref_f = 9.78           clip[9]+α=0.78 lerp clip[10]
  45      45     0.00    progress=45/45=1.0,   ref_f = 20.0           clip[20] = impact 帧
  46      46    -0.02    progress=1/20=0.05,   ref_f = 22.15          clip[22]+α=0.15 lerp clip[23]
  65      65    -0.40    progress=20/20=1.0,   ref_f = 63.0           clip[63] = clip 末尾
─── t_to_hit <= -t_post_swing = -0.40，立即重采样 cmd ───
```

机制: pre-strike 的 45 step (0.90s) 对应 ref pre 段 0→20 帧，ref 被 0.90/0.40 = 2.25× 慢放；post-strike 的 20 step (0.40s) 对应 ref post 段 20→63 帧，ref 被 0.40/0.86 = 0.465× 加速。

**例 B — 短 pre + 长 post，forehand**:
```
swing_type      = forehand         (clip_len=82, impact=37, pre=0.74s, post=0.88s native)
t_pre_initial   = 0.20s             (sim_pre_steps = 10)
t_post_swing    = 1.10s             (sim_post_steps = 55)
total swing     = 1.30s             (65 step)

step  cur_step  t_to_hit  ref_frame_f                                  说明
─────────────────────────────────────────────────────────────────────────────────
   0       0     0.20    ref_f=0                                       clip[0]
  10      10     0.00    ref_f=10/10×37=37                              clip[37] = impact
  11      11    -0.02    ref_f=37+(1/55)×(81-37)=37.80                  clip[37]+α=0.80 lerp clip[38]
  65      65    -1.10    ref_f=37+(55/55)×44=81                         clip[81] = clip 末尾
─── 重采样 ───
```

机制: pre 0.20s ref 被 0.20/0.74 = 0.270× 加速；post 1.10s ref 被 1.10/0.88 = 1.25× 慢放。

**例 C — episode timeout 时**:
sim 在某次 swing 中途 (cur_step < total_steps) 遇到 step 500 (10s) timeout，GAE 用 `V(s_500)` bootstrap。不需特殊处理 ref；那一帧 ref 还在 swing 中段，dense 计算 `r_i` 即可。
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

**问题**: `swing_type` 是由 §3.2 Step 6 base-frame 几何阈值推导出的 cmd 内部变量，2 obs 表里没列。但 `r^p/r^v/r^bp` / `T_B` (Critic obs #12) / `[q̄, q̇̄]` (Critic obs #14) 都依赖 ref clip 内容，而 ref clip 选择 = `clip[cmd.swing_type]`。policy 需要通过 `p_hit / v_racket_hat_world / base_err / t_to_hit` 等任务命令隐式区分动作类型，而不是显式读取 swing_type。

**用户决定**: 用户原话 "critic需要从专家动作中获得这个判别，因此需要swing的信息，但是不作为输入到critic的网络，而是先经过swing选择后输出需要的T". 即 swing_type **作为 ref clip selector 路由 T_B / [q̄, q̇̄], 不直接作为 obs 维度进网络**.

**实现规范**:
```python
# Critic obs #12 / #14 计算时:
selected_clip = expert_clip[cmd.swing_type]                          # 内部路由
T_B  = compute_body_state(selected_clip, ref_frame_f)                # 11·7=77 维 [v5.5 A2: 排除 paddle blade]
qd_qdot_ref = compute_joint_state(selected_clip, ref_frame_f)        # 23+23=46 维
# Critic input = [..., T_B, t_left, qd_qdot_ref]   # 213 维 [v5.5 A2: 原 220, 减 7]
# Actor input  = [..., q, q_dot, a_last]            # 仍 86 维, 不变 (无 swing_type)

# Actor 通过 p̂_racket / v̂_racket 几何分布隐推 swing_type:
# Actor 通过 p_hit / v_racket_hat_world / base_err 的几何分布隐式推断需要的动作模式。
# v5.7 中 swing_type 由 hit_y_base 与 Y_MID_BASE=0.157 推导，不显式进入网络。
```

**Actor / Critic 维度**: Actor=86, **Critic=213** [v5.5 A2: T_B 11·7=77 排除 paddle blade] (Q2 选项 C 的语义, swing_type 隐含).

**实现细节**:
- 维护 `expert_clip: Dict[str, MotionClip]` (key="forehand"/"backhand"), 加载 forward_001 + backward_004.
- 计算 T_B / [q̄, q̇̄] 时按 cmd.swing_type 索引 dict, 不进 obs tensor.
- r_i (r^p/r^v/r^bp) 计算同理: 按 swing_type 路由 ref. 

⚠️ **DIVERGENCE P — swing_type 隐含 + 几何推导**: paper 的 single-clip 设定下 swing_type 是固定的，不存在路由问题。我们 2-clip 设定下用 `cmd.swing_type` 作 selector 是工程必要；v5.7 进一步改为 base-frame `Y_MID_BASE=0.157` 几何推导并加 1-change lock。

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

⚠️ **跟 1 #3 公式一致**: 1 #3 的"计算/来源"列已同步此规则，实现时按本节为准。

---

### 11.4 expert_offset 实测复算 (v5.7 锁定) + swing_type 阈值

为 11.1 / 11.2 / 3.2 expert_offset 的明确性补充。

**两个 expert clip 的 npz 路径**:
- forehand: `motion_datasets/pingpong/humanoid_data/final/expert/forward/forward_001.npz`
- backhand: `motion_datasets/pingpong/humanoid_data/final/expert/backward/backward_004.npz`

**clip 内部字段**:
- `joint_pos` (T, 23): 关节位置
- `joint_vel` (T, 23): 关节速度
- `body_names` (25,): body 名称
- `body_pos_w` (T, 25, 3): body 世界位置，含 `right_paddle_blade`
- `body_quat_w` (T, 25, 4): body 世界 quat，wxyz
- `body_lin_vel_w` (T, 25, 3): body 世界线速度
- `body_ang_vel_w` (T, 25, 3): body 世界角速度
- `impact_frame` scalar: 击球瞬间帧索引
- `fps` scalar: 50
- `swing_type` scalar: 0=forehand, 1=backhand

**body index**:
- `PELVIS_IDX = 0`
- `BLADE_IDX = 24` (`right_paddle_blade`)

#### 11.4.1 body_names 实测

两条 expert clip 的 body_names 一致:

| idx | body name | idx | body name |
|---:|---|---:|---|
| 0 | `pelvis` | 13 | `right_knee_link` |
| 1 | `left_hip_pitch_link` | 14 | `left_shoulder_yaw_link` |
| 2 | `right_hip_pitch_link` | 15 | `right_shoulder_yaw_link` |
| 3 | `torso_link` | 16 | `left_ankle_pitch_link` |
| 4 | `left_hip_roll_link` | 17 | `right_ankle_pitch_link` |
| 5 | `right_hip_roll_link` | 18 | `left_elbow_link` |
| 6 | `left_shoulder_pitch_link` | 19 | `right_elbow_link` |
| 7 | `right_shoulder_pitch_link` | 20 | `left_ankle_roll_link` |
| 8 | `left_hip_yaw_link` | 21 | `right_ankle_roll_link` |
| 9 | `right_hip_yaw_link` | 22 | `left_wrist_roll_rubber_hand` |
| 10 | `left_shoulder_roll_link` | 23 | `right_wrist_roll_rubber_hand` |
| 11 | `right_shoulder_roll_link` | 24 | `right_paddle_blade` |
| 12 | `left_knee_link` |  |  |

#### 11.4.2 实测数据

| clip | swing_type | impact 帧 | clip_len | pelvis yaw | Δ_world (xy, m) | **Δ_base (xy, m)** | ‖v_blade‖@imp (m/s) | pre/post 时长 |
|---|---|:---:|:---:|:---:|---|---|---:|---|
| `forward_001` | `forehand` | 37 | 82 | +63.57° | (+0.035, +0.536) | **(+0.496, +0.208)** | 4.418 | 0.74 / 0.88 s |
| `backward_004` | `backhand` | 20 | 64 | +128.77° | (-0.351, +0.268) | **(+0.428, +0.106)** | 1.995 | 0.40 / 0.86 s |

#### 11.4.3 expert_offset 计算代码

```python
expert_offset_base: Dict[str, np.ndarray] = {}
for swing_type, npz_path in [("forehand",  ".../forward_001.npz"),
                              ("backhand", ".../backward_004.npz")]:
    d = np.load(npz_path)
    imp = int(d["impact_frame"][0])
    blade_w  = d["body_pos_w"][imp, BLADE_IDX,  :2]
    pelvis_w = d["body_pos_w"][imp, PELVIS_IDX, :2]
    pelvis_q = d["body_quat_w"][imp, PELVIS_IDX]
    yaw      = yaw_from_wxyz(pelvis_q)

    diff_w = blade_w - pelvis_w
    c, s = np.cos(-yaw), np.sin(-yaw)
    expert_offset_base[swing_type] = np.array([
        c * diff_w[0] - s * diff_w[1],
        s * diff_w[0] + c * diff_w[1],
    ])
```

#### 11.4.4 关键结论

1. **world frame Δ 不可用**: 两条 clip 的 pelvis yaw 相差约 65.2°，world frame 下 forehand / backhand 的 delta 方向不一致，不能作为统一 expert_offset。
2. **base frame Δ 稳定**: 两条 clip 在 pelvis/base frame 下 paddle 都在前方稍左，x≈0.46m，y∈[+0.106,+0.208]，可作为 base target 的稳定 offset。
3. **训练端必须存 `expert_offset_base`**，每次根据 robot 当前 yaw 旋回 world:

```python
yaw_robot = base_yaw_now
R_b2w = np.array([[ np.cos(yaw_robot), -np.sin(yaw_robot)],
                  [ np.sin(yaw_robot),  np.cos(yaw_robot)]])
offset_world_now = R_b2w @ expert_offset_base[cmd.swing_type]
p_base_xy_world = hit_xy_world - offset_world_now
```

#### 11.4.5 swing_type 阈值

由 base-frame y 数值得:
- `y_forehand_base = +0.208 m`
- `y_backhand_base = +0.106 m`
- **`Y_MID_BASE = (+0.208 + +0.106) / 2 = +0.157 m`**

```python
swing_type = "forehand" if hit_y_base > 0.157 else "backhand"
```

#### 11.4.6 与 `planner.py` deploy heuristic 的方向差异

`planner.py` 的 deploy 端 heuristic 是 `forehand if hp[1] < bp[1] else backhand`，使用 world frame 且阈值为 0。训练端 v5.7 使用 `forehand if hit_y_base > +0.157 else backhand`，使用 base frame 且阈值来自 expert 数据。

二者字面方向不同，原因是当前 expert clip 的 pelvis yaw 分别为 63.6° / 128.8°，paddle 在两个 clip 中都落在 base +y 侧。训练端必须服从当前 expert 数据的 base-frame 几何；否则 `r_i` ref clip selector 与 `p_base_xy_world` 会不一致。部署端若改成真正右手球员几何，需重录或重标 expert 数据。

---

### 11.5 [Q4] 时间字段命名: **cmd 内部和 obs 都叫 t_to_hit (剩余击球时间)**

**问题**: v5 表里曾出现 `t_strike` (= absolute strike 时刻) 和 `t_to_hit` (= 剩余时间) 两个字段, 同名异义混淆 — paper Table I 写 `t_strike` 但语义是剩余时间.

**用户决定**: **统一用 `t_to_hit`** (剩余击球时间). cmd 内部不存绝对时刻, 直接存 t_to_hit 标量, 重采样时 t_to_hit ← Δt_swing (truncN 采样), 每 step `t_to_hit -= dt`.

**实现规范** [v5.4: 加 t_pre_initial / t_post_swing, cur_frame → cur_step]:
```python
# cmd dataclass:
@dataclass
class PingpongCommand:
    # swing 决策 (v5.7 1-change lock)
    swing_type: str                           # "forehand" / "backhand"
    swing_change_remaining: int               # resample 时 =1；pre-strike 首次变更后置 0

    # 上游物理输入 (cmd 内部字段，不进 obs)
    p_hit_world: np.ndarray                   # (3,) world, Table I 的 p_racket target
    v_ball_in_world: np.ndarray               # (3,) world, 合成 incoming ball velocity
    target_land_world: np.ndarray             # (3,) world = (2.45, 0.0, 0.78)
    flight_time: float                        # uniform[0.30, 0.65] 秒
    paddle_cor: float                         # = 0.85

    # Eq.5 / Eq.6 推导输出
    v_racket_hat_world: np.ndarray            # (3,) world, Table I 的 v_racket target
    n_target_world: np.ndarray                # (3,) world, r_g_ori 使用的 paddle normal
    v_ball_out_world: np.ndarray              # (3,) world, sanity monitor

    # base 站位目标
    p_base_xy_world: np.ndarray               # (2,) world

    # 时间字段
    t_to_hit: float                           # pre 段 t_pre→0，post 段 0→-t_post
    t_pre_initial: float                      # truncN[0.20,0.90,0.30,0.65]
    t_post_swing: float                       # fixed 0.60s
    cur_step: int                             # sim step 计数，resample 时复位 0

    # cmd noise: 每次 resample 采一次并冻结，仅 Actor obs 注入
    noise_p: np.ndarray                       # (3,) gauss(σ_p_now) clip ±3σ
    noise_v: np.ndarray                       # (3,) gauss(σ_v_now) clip ±3σ
    noise_base: np.ndarray                    # (2,) gauss(σ_base_now) clip ±3σ
    noise_t: float                            # scalar gauss(σ_t_now) clip ±3σ

    # monitor
    last_resample_was_degenerate: bool        # Eq.5/Eq.6 delta_v 退化 fallback 是否触发
    hit_y_base: float                         # 最近一次 swing_type 判断使用的 base-frame y

    # 不存 t_strike_absolute；不存 ref_frame_f；ref_frame_f 由 §11.1.2 get_ref_state(cmd, dt) 当场算。

# 每 step:
def cmd_step(cmd, robot, env_id, dt=0.02):
    cmd.t_to_hit -= dt
    cmd.cur_step += 1

    if cmd.t_to_hit > 0.0 and cmd.swing_change_remaining > 0:
        new_swing = compute_swing_type_from_current_base(cmd.p_hit_world, robot, env_id)
        if new_swing != cmd.swing_type:
            cmd.swing_type = new_swing
            cmd.swing_change_remaining = 0
            cmd.p_base_xy_world = compute_base_target(cmd.p_hit_world, robot, env_id, new_swing)

    if cmd.t_to_hit <= -cmd.t_post_swing:
        resample_cmd(cmd)

# obs 端 [v5.5 A10 + v5.7: Actor noisy / Critic clean]:
obs_actor.p_hit = R_base.T @ ((cmd.p_hit_world + cmd.noise_p) - p_base_world)
obs_actor.v_racket_hat_world = cmd.v_racket_hat_world + cmd.noise_v
obs_actor.base_err = (cmd.p_base_xy_world + cmd.noise_base) - p_base_xy_world
obs_actor.t_to_hit = cmd.t_to_hit + cmd.noise_t

obs_critic.p_hit = R_base.T @ (cmd.p_hit_world - p_base_world)
obs_critic.v_racket_hat_world = cmd.v_racket_hat_world
obs_critic.base_err = cmd.p_base_xy_world - p_base_xy_world
obs_critic.t_to_hit = cmd.t_to_hit
# t_pre_initial / t_post_swing / cur_step / noise_* / v_ball_in / target_land / flight_time / paddle_cor 都不进 obs。
```

**与 paper 的关系**: paper Table I 字段名 `t_strike` 在 paper 中实际就是剩余时间 (paper V-B "the time until the strike"). 我们改名为 `t_to_hit` 是为了消除"绝对/相对时刻"的歧义 — 与 paper 语义一致, 命名更清楚.

**Strike window gate**: 仍用 `abs(cmd.t_to_hit) ≤ 0.06s` (= ±3 帧 @ 50Hz). cmd.t_to_hit 在 pre 段从 t_pre_initial 单调减到 0, post 段继续减到 -t_post_swing 才重采样, 所以"负数 t_to_hit" 在 [−t_post_swing, 0] 区间内停留多个 step (= sim_post_steps 个), strike window 仍以 t_to_hit=0 ± 3 帧自然落在 ref impact 帧附近.

### 11.6 Quaternion convention 与 blade normal 硬约束

全工程内部 quaternion convention 统一为 **`wxyz`**。专家 npz 的 `body_quat_w / root_quat_w` 按 `wxyz` 读取；`motion_loader.py`、`observations.py`、`rewards.py`、`events.py` 内部张量也都保持 `wxyz`。任何 IsaacLab API、asset utility、可视化工具或第三方函数若返回 / 接收 `xyzw`，必须在边界显式转换，转换函数名要写清楚，例如 `xyzw_to_wxyz()` / `wxyz_to_xyzw()`，不得靠注释或调用者记忆。

**必须写的 quaternion unit tests**:

| case | 输入 | 期望 |
|---|---|---|
| identity | `q_wxyz=(1,0,0,0)` | `yaw_from_wxyz=0`，任意向量旋转不变 |
| 90deg yaw | `q_wxyz=(cos45,0,0,sin45)` | local +x 旋到 world +y；`yaw_from_wxyz=pi/2` |
| blade normal | 使用 asset 初始站姿 FK | `right_paddle_blade` local +Y 旋到 world 后与 debug draw 方向一致 |

`right_paddle_blade` 的拍面法向默认定义为 local **+Y**:

```python
n_blade_world = quat_rotate(blade_quat_wxyz, torch.tensor([0.0, 1.0, 0.0], device=device))
r_g_ori = exp(-((1.0 - dot(n_blade_world, cmd.n_target_world)) ** 2) / sigma_ori**2)
```

**实测验证要求**: 初始站姿或单 env play 中必须画出 blade local +Y 箭头。箭头应从拍面正面指向来球/出球目标侧；若方向相反，只允许在 asset normal 定义或统一 helper 中修正一次，例如把 `BLADE_NORMAL_LOCAL` 改为 `[0,-1,0]`，不允许在 `r_g_ori`、success 判据或某个 debug draw 里临时取负。这样可以保证 `r_g_ori`、§8.1.1 success ori 判据、debug draw 和部署检查使用同一个法向定义。

---

## 12. v5.7 训练 Monitor 指标

记录这些指标到 rsl_rl `extras` dict，episode end 时 push wandb。第一轮训练 1k iter 后回看，必要时再决定是否调整 cmd sample 范围或加入死区。v5.7 决定: forehand/backhand 偏斜**只监控，不改 sample 逻辑**。

| # | 指标 | 频次 | 阈值 (alarm) | 含义 / 用途 |
|---|---|---|---|---|
| 1 | `pingpong/hit_success_rate` | per 1k iter sliding window | curriculum 主指标 | §8.1.1 的 swing-level `mean(hit_success)`，驱动 σ_g_pos / noise / range |
| 2 | `pingpong/hit_success_pos_fail_rate` / `_vel_fail_rate` / `_ori_fail_rate` | per 1k iter sliding window | 任一项 > 70% | 拆解 success 失败原因，避免只看总成功率 |
| 3 | `pingpong/swing_ratio_forehand` | per 1k iter sliding window | < 0.30 or > 0.70 | forehand:backhand 不平衡 → ref clip 训练样本偏斜 |
| 4 | `pingpong/dead_zone_trigger_rate` | per swing | > 5% | `abs(hit_y_base - 0.157) < 0.01` 时 swing 决策接近边界 |
| 5 | `pingpong/swing_flip_rate_per_episode` | per episode | > 4 | episode 内 swing_type 翻转次数，一次 swing 最多 1 次 |
| 6 | `pingpong/base_y_drift_meanabs` | per episode | abs > 0.5 | base y 累计平均漂移，与 swing 偏斜直接相关 |
| 7 | `pingpong/v_racket_hat_world_mag_mean` / `_std` | per 1k iter | mean 不在 [1.5, 5.0] 或 std > 1.5 | Eq.5/Eq.6 输出球拍速度量级 sanity |
| 8 | `pingpong/solve_paddle_degenerate_rate` | per swing | > 0.001 | `||delta_v|| < 1e-9` fallback 触发率 |
| 9 | `pingpong/cos_sim_n_blade_n_target_at_impact` | per swing (impact 帧) | mean < 0.85 | 击球瞬间 paddle 法向与目标法向对齐程度 |
| 10 | `pingpong/swing_change_remaining_used_rate` | per swing | > 50% | 1-change lock 的使用率，高说明 base 漂移或边界附近样本多 |

实现端可以在 `mdp/curriculums.py` 或 command term 的 metrics 中写入这些标量。注意所有 monitor 读取 clean cmd 真值；不要把 Actor noisy obs 反推回来做统计。

---

## 13. Paper Divergence 完整索引 (v5.7)

| # | 项 | paper | 我们 | 风险 / TODO |
|---|---|---|---|---|
| A | r_i / r_g / r_r 顶层 weights | Eq.7 给形式，未给数值 | 0.5 / 1.0 / 1.0 | 训练后可调 |
| B | r_g_base 时序 | 描述为 before strike | `t_to_hit > 0` dense，击球后 OFF | paper-aligned，留 ablation |
| C | r_i sub-term 分解 | 仅说 upper-body imitation | `r^p/r^v/r^bp`，删 `r^e/r^c` | 工程选择 |
| D | ℬ / J 排除末端 | paper 未列具体 body/joint | 排除 `right_paddle_blade` 与 `right_wrist_roll_joint` | 让任务信号主导拍面 |
| E | clip 长度 / impact ratio | paper clip 94 帧、impact 43 | forehand 82/37；backhand 64/20 | backhand ratio 偏低 |
| F | backhand blade speed | paper 未给 | 1.995 m/s vs forehand 4.418 m/s | 监控 r_g_vel 均衡 |
| G | r_g sub-term weights | 未给 | pos 2.0 / vel 1.0 / ori 0.5 / base 0.3 | 训练后可调 |
| H | σ_vel 自适应 | 未给 | 固定 σ=0.5 m/s | 简化 |
| I | r_r 整体 | 未给细节 | IsaacLab 标准 reg + alive + table contact | 工程必要 |
| J | σ_g_pos curriculum | 未提 | 0.10 → 0.02 m，5 阶段 | 第一轮启用 |
| K | action scale | 未给具体数值 | mimic per-joint `0.25*effort/stiffness` | 复用高速 mimic 配置 |
| M | RSI 解耦 | DM 标准 RSI 同帧 | 物理姿态随机帧，ref 时间 `cur_step=0` | 与 cmd.t_to_hit 同步 |
| N | 表桌建模 | 未明 | 静态 table + filtered ContactSensor | 工程必要 |
| O | ref 双段时间比例插值 | 单 clip 不需 | pre/post 分段映射，保证 impact 对齐 | 2-clip + 自由时长需要 |
| P | swing_type | paper deploy heuristic 未详述 | base-frame `Y_MID_BASE=0.157` + 1-change lock | 来自当前 expert 数据 |
| Q | cmd 内部字段扩展 | Table I 只列 policy 输入 | 增加 t_pre/t_post/cur_step/swing lock/物理上游字段 | 实现需要，不进 obs |
| R-1 | incoming ball velocity | paper 部署时来自真实弹道预测 | 训练中合成 `v_ball_in_world` | sample 范围需 monitor |
| R-2 | target landing point | paper 设期望落点 | 固定 `(2.45,0,0.78)` | 可能限制回球多样性 |
| R-3 | flight time | paper 可由规划给定 | `uniform[0.30,0.65]` | 可能影响 v̂ 分布 |
| S | curriculum success | paper 真实任务可看球是否回到目标区域 | 无 ball rollout 时用 strike-window 几何三条件 | 需 monitor fail reason |
| T | root reset | DM RSI 可整帧复制 root | 不复制专家世界 root xy/yaw，只用训练 nominal root | 减少坐标污染，但少了 root pose 多样性 |
| U | quaternion / blade normal | paper 未涉及实现 convention | 内部 `wxyz` + blade local +Y 实测 | 单测 / debug draw 必做 |
| V | train/deploy frame bridge | paper deploy 系统有感知规划闭环 | 训练 world frame 固定，planner 输出必须显式转换 | 部署默认 `target_land` 风险 |
| W | PPO 超参 | paper 只给 MLP | 复用 mimic `BasePPORunnerCfg` | 非论文公开值 |

v5.7 净变化: 撤销“直接采样 `v_racket_hat_world`”和“独立采样 `swing_type`”两项旧 divergence；新增 R-1/R-2/R-3/S/T/U/V/W 作为训练合成命令、reset、接口与训练配置的工程选择。

---

## 14. 实现 Phase

### Phase 1: 本文档

`final_v57.md` 是代码编写的 v5.7 参考副本；原 `final.md` 保持不变以便追溯。

### Phase 2: 实现 `mdp/` 代码

1. `mdp/commands.py` — `PingpongCommand` / `PingpongCommandCfg`: §3.2 七步 resample、Eq.5/Eq.6 inline、swing 1-change lock、per-swing noise freeze、command metrics。
2. `mdp/motion_loader.py` — expert clip dict、`get_ref_state(cmd, dt)`、双段 lerp/slerp、`expert_offset_base` 预处理。
3. `mdp/observations.py` — Table I 的 14 项，Actor=86，Critic=213，Actor noisy / Critic clean。
4. `mdp/rewards.py` — `r_i` / `r_g` / `r_r`，其中 `r_g_ori` 使用 `cmd.n_target_world`。
5. `mdp/events.py` — startup DR、RSI reset、interval push；cmd noise 不在 interval per-step 重采。
6. `mdp/terminations.py` — timeout、bad orientation、root height、严重 undesired contact。
7. `mdp/curriculums.py` — σ_g_pos、cmd noise、hit/v_in range curriculum 与 §12 monitor。
8. `robots/g1_23dof/hitter/hitter_env_cfg.py` — 装配 scene/action/obs/reward/event/termination/curriculum。
9. `agents/rsl_rl_ppo_cfg.py` — 第一版继承 mimic `BasePPORunnerCfg`，actor/critic MLP `[512, 256, 128]` 不变。

### Phase 3: env_cfg

复用:
- `UNITREE_G1_23DOF_PADDLE_MIMIC_CFG`
- `UNITREE_G1_23DOF_PADDLE_MIMIC_ACTION_SCALE`

装配:
- table cuboid
- `contact_forces`
- `robot_table_contact`
- 10s episode
- 50Hz control

### Phase 4: PPO / Runner cfg

论文只公开 WBC 使用 PPO 和 actor/critic MLP `[512, 256, 128]`，没有给完整 PPO 超参。第一版 pingpong 训练直接复用 [mimic rsl_rl_ppo_cfg.py](/home/woan/HumanoidProject/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/agents/rsl_rl_ppo_cfg.py:15) 的 `BasePPORunnerCfg`，不要重新发明一套超参。

| 字段 | 取值 |
|---|---|
| `num_steps_per_env` | `48` |
| `max_iterations` | `80000` |
| `save_interval` | `1000` |
| `empirical_normalization` | `False` |
| `policy.init_noise_std` | `1.0` |
| `policy.actor_hidden_dims` | `[512, 256, 128]` |
| `policy.critic_hidden_dims` | `[512, 256, 128]` |
| `policy.activation` | `"elu"` |
| `algorithm.value_loss_coef` | `1.0` |
| `algorithm.use_clipped_value_loss` | `True` |
| `algorithm.clip_param` | `0.2` |
| `algorithm.entropy_coef` | `0.005` |
| `algorithm.num_learning_epochs` | `5` |
| `algorithm.num_mini_batches` | `4` |
| `algorithm.learning_rate` | `5.0e-4` |
| `algorithm.schedule` | `"adaptive"` |
| `algorithm.gamma` | `0.99` |
| `algorithm.lam` | `0.95` |
| `algorithm.desired_kl` | `0.01` |
| `algorithm.max_grad_norm` | `1.0` |

实现方式: `tasks/pingpong/agents/rsl_rl_ppo_cfg.py` 可以 subclass mimic `BasePPORunnerCfg`，只覆盖 `experiment_name` / run naming / task entry 需要的字段；PPO 核心字段保持上表。若第一轮训练出现 reward scale 不稳，先调 reward 和 curriculum，不优先改 PPO。

## 15. 验证计划

1. **Markdown sanity**: `final_v57.md` 不包含旧未决项或过渡修改口吻；原 `final.md` 未修改。
2. **Cmd unit test**: 固定 RNG 后检查 `v_ball_out_world / n_target_world / v_racket_hat_world` 与 Eq.5/Eq.6 手算一致。
3. **Degenerate test**: 构造 `||delta_v|| < 1e-9`，确认 retry / fallback 与 monitor 生效。
4. **Expert data test**: 验证 `impact_frame`、`pre/post`、`expert_offset_base` 与 §11.4 表一致。
5. **Ref alignment test**: `cur_step = t_pre_initial / dt` 时 `ref_frame_f == impact_frame`。
6. **Obs dimension test**: Actor=86，Critic=213；上游物理字段不进 obs。
7. **Reward gate test**: strike window 共 7 帧；`r_g_base` 击球后 OFF；reward 不读 `cmd.noise_*`。
8. **Scene smoke test**: table collision 触发 `r_table_contact`；一般 contact sensor 仍服务 feet / undesired contact。
9. **Single-env play smoke**: `--num_envs=1 --headless=False`，可视化检查 base target / hit target / ref clip progression。
10. **1k iter train monitor**: 检查 `hit_success_rate`、σ_g_pos curriculum、swing ratio、v_racket_hat_world magnitude、degenerate rate、impact cos alignment。
11. **Success metric test**: 构造 strike window 内 pos/vel/ori 三条件分别成功/失败的样例，确认 `hit_success_rate` 只由 §8.1.1 阈值驱动，不读 reward 标量。
12. **Quaternion convention test**: identity、90deg yaw、blade +Y normal 三个 case 全部通过；所有 loader / reward / obs 边界保持 `wxyz`。
13. **Blade normal FK/debug test**: 初始站姿画出 `right_paddle_blade` local +Y；若反向，在统一 normal 定义处修正，不在 reward 端取负。
14. **Reset/RSI consistency test**: reset 后 `cmd.swing_type`、`selected_ref_clip`、`p_base_xy_world` 一致；专家 root world xy/yaw 没有覆盖训练 nominal root。
15. **Train/deploy bridge test**: 同一组 `p_hit_world / v_ball_in_world / target_land_world / flight_time / paddle_cor` 下，训练 inline Eq.5/Eq.6 与 deploy `solve_paddle_target` 输出一致；部署显式传 `target_land_world=(2.45,0,0.78)`。
16. **Debug draw 1**: 每个 env 绘制 `p_hit_world`、`p_base_xy_world`、`n_target_world`、`v_racket_hat_world`、`blade normal`，检查坐标系和 normal 方向。
17. **Debug draw 2**: strike window 内绘制 blade trajectory 与 `p_hit_world` 的距离曲线，确认 `t_to_hit=0` 时 `ref_frame_f==impact_frame` 且距离最低点落在窗口内。
