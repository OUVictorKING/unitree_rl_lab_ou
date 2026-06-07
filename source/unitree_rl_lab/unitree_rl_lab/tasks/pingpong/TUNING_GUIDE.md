# Pingpong (HITTER / HITTER-REAL) 参数调节指南

> 适用：`robots/g1_23dof/hitter/hitter_env_cfg.py`（训练主配置）+ 它引用的 `mdp/commands.py`、`mdp/rewards.py`、`mdp/curriculums.py`。
> HITTER-REAL 继承本文件，额外加真实球奖励/课程（见末尾）。

---

## 0. 怎么用这份文档
- 想改某个行为 → 先看 **§1 架构** 知道它归哪个课程管，再看 **§2 死/活权重**确认改的地方是不是被课程覆盖了，最后到对应分区表查"调大/调小"。
- 急用 → 直接看 **§10 常见场景速查**。
- 控制频率：`decimation=4`、`sim.dt=0.005` → **仿真 200Hz、策略 50Hz、控制 dt=0.02s**。`episode_length_s=10` → 每回合 500 步。`num_envs=4096`。

---

## 1. 训练架构总览

**3 个任务相位（task_phase 课程，单调闭锁，只进不退）**
| Phase | 何时 | imit 权重(imit_w) | goal_* 击球奖励 | 目的 |
|---|---|---|---|---|
| 0 站立 | 起步 | 0.10（弱） | **0** | 先学会站 + 横移到位 |
| 1 模仿 | EL_ema≥350 | 1.00（强） | **0** | 学正/反手挥拍姿态 |
| 2 击球 | EL_ema≥450 且 phase1 跑满 `phase_1_min_iters` | 0.30 | **开**（按 window 课程逐级 ramp） | 打准/打到位 |

**5 个课程（每 tick 按顺序跑，顺序重要）**
1. `imit_anneal` — track EL/命中 EMA + 相位指标（**它设的模仿权重会被 task_phase 覆盖**，主要作用是喂 EMA）。
2. `pingpong`（window/shape/v_in/hit_y 总课程）— 收紧 σ、收紧击球窗口、ramp goal_* 权重、放大球速/落点范围；含冻结/回退保护。
3. `task_phase` — 设相位 + **覆盖**模仿权重、腿正则权重、Phase2 开 goal_*。
4. `table_guard` — 站稳+会打后把桌子放回来 + ramp 桌面碰撞惩罚。
5. `imit_orient_anneal`（v65 新）— 瓶颈时把腰+右手 distal 的模仿权重往下降。

---

## 2. ⚠️ 最重要：哪些"初始权重/值"是死的（被课程每 tick 覆盖）

**改这些 RewTerm 的初始 weight 没用**——课程每步都会重写。要改就改课程里的源头：

| 你看到的初始值（位置） | 实际由谁控制 | 真正的旋钮 |
|---|---|---|
| `imitation_joint_pos/vel/body` 的 `0.65/0.1/0.25 * w_i`（rewards 区 `w_i=0.5`） | **task_phase 每 tick 覆盖** = `_TASK_PHASE_IMIT_SPLIT × imit_w_phase` | `imit_w_phase0/1/2`（task_phase 参数）+ `_TASK_PHASE_IMIT_SPLIT`(curriculums.py，默认 joint0.40/body0.50/vel0.10) |
| `goal_position/velocity/orientation(+pre_strike)` 的 2.0/1.0/… | Phase0/1=0；Phase2 = **window 课程 tier ramp** | `_WINDOW_CURRICULUM_TIERS`(§7) 的 w_pos/w_vel/w_ori |
| `goal_*` 的 `std`（0.2/1.5/0.4…） | **shape 课程每 tick 覆盖** σ | `_REWARD_SHAPE_TIERS`(§7) 的 σ 列 + curriculums.py 的 floor |
| `leg_joint_deviation / feet_contact_no_strike / feet_distance_no_strike` 的 -0.5/0.2/-0.5 | **task_phase 覆盖** | `leg_reg_phase_weights`（task_phase 参数，3 相位元组） |
| `paddle_table_contact / body_table_contact` 的 0.0 | **table_guard ramp** 到 -10/-1 | `target_paddle_weight / target_body_weight`（table_guard） |
| `command.cfg.sigma_g_pos / strike_window / hit_z_range / v_in_mag_range / hit_y_*` | **pingpong 课程驱动** | §7 tier 表 + §6 pingpong 的 unlock 阈值 |
| `imit_joint_weights[waist+右distal]` | **imit_orient_anneal 驱动** | stall_iters/stall_decay/floor（§6） |

> **真正"静态可调、不被覆盖"的**：所有正则项权重（alive/action_*/joint_*/pelvis_*/feet_slide/undesired_contacts）、`success_*_thresh` 命中判据、`swing_p_forehand`、动作 clip、imitation 的 `gate_pre_strike/post_strike_scale`、PPO 超参、各课程的"阈值/速率"本身。

---

## 3. 命令 / 任务几何（`CommandsCfg.pingpong` + `PingpongCommandCfg` 默认）

| 参数 | 位置 | 当前值 | 作用 | 调大↑ / 调小↓ |
|---|---|---|---|---|
| `strike_window` | CommandsCfg | 0.10（初始，window课程→0.01） | 判定"在击球窗口内"的 \|t_to_hit\| 阈值；goal_* 只在窗口内给分 | 大=窗口宽、好学但糙；课程会收 |
| `success_pos_thresh` | cfg默认 | 0.15 m | 位置命中判据 | 小=更严，hsr 降 |
| `success_vel_thresh` | cfg默认 | 1.0 m/s | 速度命中判据 | 小=要求挥得更准速 |
| `success_ori_cos_dist_thresh` | cfg默认 | 0.25（正手 cos>0.75=41°） | 正手拍面命中判据 | 小=拍面要更准 |
| `success_ori_cos_dist_thresh_backhand` | **CommandsCfg 覆盖** | **0.20**（反手 cos>0.80=37°） | 反手拍面判据（更严，逼反手对面） | 小=更逼拍面但 hsr_bh 降、可能卡课程；大=放松离开刀刃 |
| `hit_z_range` | cfg默认 | (0.95,1.25) | 击球点高度范围（pingpong课程会调 0.85~0.92 下限） | — |
| `v_in_mag_range` | cfg默认 | (1.5,2.0) | 来球速度；高端被 v_in 课程 ramp 到 3.5 | 大=更快球、更难 |
| `hit_y_base_initial_half_width` / `..._max_half_width` | cfg默认 | 0.10 / 0.50 | 击球点 y(base) 范围，课程从窄到宽 | — |
| `hit_y_world_cap(_initial/_max)` | cfg默认 | 0.45 / 0.45 / 1.00 | 世界系 \|y\| 硬上限，课程放宽 | — |
| `swing_p_forehand` | cfg默认 | 0.50 | 正手 vs 反手采样比例 | 想多练反手就调小 |
| `imitation_joint_names` | **CommandsCfg 覆盖** | 全 11 关节（含腰+右distal） | 哪些关节进模仿；删掉=完全放开（默认版放开右distal） | 见 §10「不用腰/拍面卡」 |
| `forward/backward_motion_file` | cfg默认 | new_3 wristfix/rotated | 专家 demo 片段（换 npz 在此改） | — |
| `sigma_g_pos` | cfg默认 | 0.30（shape课程→0.06） | goal_position 高斯宽度（被课程覆盖，见§7） | — |

---

## 4. 动作（`ActionsCfg.JointPositionAction`）
| 参数 | 当前值 | 作用 | 备注 |
|---|---|---|---|
| `scale` | `UNITREE_G1_23DOF_PADDLE_MIMIC_ACTION_SCALE`(=0.25·effort/stiffness) | 每关节动作→目标角的缩放 | 右臂≈25%物理力矩上限；想更"灵敏"可在 asset 里加右臂 scale |
| `clip` | `{".*": (-10,10)}` | 夹住原始动作 | **防发散崩溃**（action²惩罚爆 −1e22 的护栏）；正常动作 ±3，不咬 |

---

## 5. 奖励权重（`RewardsCfg`，**先看 §2 哪些是死的**）

**模仿（有效权重见 §2，这里是 func 参数）**
| 参数 | 位置 | 当前值 | 作用 |
|---|---|---|---|
| `imitation_joint_pos/vel` `gate_pre_strike` | params | False | True=击球后完全不模仿 |
| `imitation_joint_pos/vel` `post_strike_scale` | params | **0.0**（=击球后模仿关掉）| 0~1：击球后模仿 = 击球前的这个比例。0.3=保留3成、收拍仍被锚住；1.0=全程满模仿 |
| `imitation_*` `k` | params | pos 2.0 / vel 0.1 | 误差→exp 的锐度 |

**任务 goal（Phase2 才开，权重被 window 课程 ramp）**
| 项 | 当前 std | 作用 |
|---|---|---|
| `goal_position(_pre_strike)` | 0.2 | 拍到击球点 |
| `goal_velocity(_pre_strike)` | 1.50 / 0.6 | 拍速对齐 v_racket_hat |
| `goal_orientation(_pre_strike)` | 0.4（shape课程→0.15 floor） | 拍面对齐 n_target（**调小 std=更锐、逼精度**） |
| `goal_base(_orientation)` | 0.3 / 0.3 | 横移到位 + 基座朝 +X（防扭身体作弊） |

**正则（这些是静态、随便调）**
| 项 | 权重 | 作用 / 调参 |
|---|---|---|
| `alive` | +0.04 | 存活；太大→赖着不动 |
| `action_rate_l2 / action_l2` | -0.001 / -0.0005 | **已是 bounded 版**（防爆）；动作平滑 |
| `joint_torque / joint_acc / energy` | -3e-6 / -1e-7 / -2e-5 | 省力/平滑；太大→不敢发力 |
| `joint_limit` | -5.0 | 关节限位 |
| `pelvis_orientation` | -1.0 | 基座姿态稳 |
| `pelvis_ang_vel_xy` | -0.05 | 罚 roll/pitch 速率（不罚 yaw，留扭腰） |
| `pelvis_lin_vel_z` | -1.5 | 罚竖直弹跳 |
| `pelvis_height` | -5.0（target 0.74） | 站高 |
| `feet_slide` | -0.3 | 脚打滑 |
| `leg_joint_deviation/feet_contact_no_strike/feet_distance_no_strike` | 死值，task_phase 覆盖 | 腿待命姿态（见 leg_reg_phase_weights） |
| `undesired_contacts` | -1.0 | 非脚/腕/拍的碰撞 |
| `paddle/body_table_contact` | 0→table_guard ramp -10/-1 | 拍/身体压桌 |

---

## 6. 课程参数（`CurriculumCfg`）

### 6.1 `imit_anneal`（update_imitation_weight）
| 参数 | 当前值 | 作用 |
|---|---|---|
| `w_i_values` | (0.5,0.3,0.15) | 三相位模仿基权重（**注意：被 task_phase 的 imit_w_phase 覆盖**，主要还在喂 EMA + 相位推进） |
| `split` | "body_dominant" | 模仿三项的分配（同样被 task_phase 的 `_TASK_PHASE_IMIT_SPLIT` 覆盖） |
| `phase_thresholds` | 两段 dict（hsr/pos/vel/ori/min_ep_length） | 相位推进门槛；**卡相位时放松这里** |
| `metric_ema_alpha` | 0.05 | EMA 慢速 |

### 6.2 `pingpong`（update_pingpong_curriculum，总课程）
| 参数 | 当前值 | 作用 |
|---|---|---|
| `min_ep_length_for_window/ori/pos_vel_advance` | 250 | 站稳门：EL<250 时不开 goal_*/不收窗口（防站不稳就farm击球分） |
| `sequenced_curriculum` | True | Stage1(窗口/权重)→Stage2(球速)→Stage3(落点) 顺序解锁 |
| `v_unlock_shape_tier / v_unlock_hsr / v_unlock_cos_sim` | 6 / 0.85 / 0.55 | **Stage2(加球速)解锁门**——hsr 要 0.85、cos_sim_ema 要 0.55 |
| `y_unlock_v_in_high / y_unlock_hsr` | 3.5 / 0.80 | Stage3(放宽落点)解锁门 |
| `cos_sim_collapse_threshold` | 0.35 | cos_sim_ema 跌破此值→回退球速/落点（反向保护） |
| (代码内)`cos_sim_freeze_threshold` | 0.45 | cos_sim_ema<0.45→冻结 window 课程 |

> ⚠️ **cos_sim_ema 现在读"击球瞬间 cos"**（v64 修，原来读被稀释的 current-frame）。`std_g_ori` floor 已压到 0.15（v64），σ 单调闸已加（收紧不回松）。

### 6.3 `task_phase`（update_task_phase）—— 模仿/相位主旋钮
| 参数 | 当前值 | 作用 |
|---|---|---|
| `el_phase_0_to_1 / _1_to_2` | 350 / 450 | 相位推进 EL 门 |
| `phase_1_min_iters` | **0**（v65 快验证；原 2000） | Phase1 最少跑多少 iter。**从头练要改回 ~2000**（防跳过模仿）；resume 验证用小值 |
| `imit_w_phase0/1/2` | 0.10/1.00/0.30 | **模仿强度主旋钮**（× `_TASK_PHASE_IMIT_SPLIT`） |
| `leg_reg_phase_weights` | 3 项×3 相位 | 腿正则相位权重 |

### 6.4 `table_guard`（update_table_guard_stage）
| 参数 | 当前值 | 作用 |
|---|---|---|
| `min_hsr_ema/min_cos_sim_ema/min_ep_length_ema/min_iter` | 0.65/0.45/400/1500 | 放回桌子的门槛 |
| `ramp_iters` | 500 | 桌面惩罚 0→满 的 ramp 步数 |
| `target_paddle/body_weight` | -10 / -1 | 桌面碰撞最终惩罚 |

### 6.5 `imit_orient_anneal`（update_imit_orient_weight，v65 新）
| 参数 | 当前值 | 作用 |
|---|---|---|
| `orient_joint_names` | 腰yaw+右shoulder_yaw/elbow/wrist_roll | 哪些关节降模仿权重 |
| `active_phase` | 2 | 只在 Phase2 生效 |
| `stall_iters` | 600 | cos_sim_ema 多少 iter 不涨判为瓶颈 |
| `stall_decay` | 0.6 | 每次瓶颈把这些关节模仿权重 ×0.6 |
| `floor` | 0.05 | 降到的下限（不为 0，不放开） |

---

## 7. 课程 tier 表（curriculums.py，σ + window 权重）

**`_REWARD_SHAPE_TIERS`**：格式 `(σ_pos, σ_vel, σ_ori, hsr门, pos门, vel门, ori门)`，从下往上（tier0→6）随 4 个成功率 EMA 全部达标逐级收紧 σ：
```
tier6 (0.06, 0.50, 0.15, |0.85,0.95,0.85,0.85)  ← 最严（paper）
tier5 (0.08, 0.65, 0.18, |0.75,0.92,0.78,0.80)
tier4 (0.10, 0.80, 0.22, |0.65,0.88,0.70,0.75)
tier3 (0.13, 1.00, 0.28, |0.55,0.82,0.62,0.70)
tier2 (0.18, 1.20, 0.32, |0.40,0.75,0.55,0.65)
tier1 (0.24, 1.35, 0.36, |0.20,0.60,0.40,0.55)
tier0 (0.30, 1.50, 0.40, |0,0,0,0)              ← 起步（宽，有梯度）
```
σ 越小奖励越锐、越逼精度。floor：σ_pos≥0.06、σ_vel≥0.20、σ_ori≥**0.15**。**σ 单调闸**：收紧后不回松。

**`_WINDOW_CURRICULUM_TIERS`**：格式 `(hsr门,pos门,vel门,ori门, strike_window, w_pos, w_vel, w_ori)`，hsr 越高→窗口越窄、goal_* 权重越大：
```
(0.80,0.85,0.85,0.85, 0.01, 12, 12, 4.0)
(0.60,0.75,0.70,0.75, 0.02,  8,  8, 2.5)
(0.40,0.65,0.55,0.65, 0.04,  5,  5, 1.5)
(0.20,0.50,0.40,0.55, 0.06,  3,  3, 1.0)
(0.00,0.00,0.00,0.00, 0.10,  2,  2, 0.5)   ← 起步
```

---

## 8. 终止（`TerminationsCfg`）
| 项 | 阈值 | 作用 |
|---|---|---|
| `time_out` | 10s | 回合超时 |
| `base_height` | <0.30m | 摔了 |
| `bad_orientation` | >0.8 rad | 基座倾太多 |
| `hard_contact` | 力>1.0（pelvis/torso/head/hip_pitch） | 硬碰撞 |
| `non_paddle_table_stuck` | 力>3.0 持续 0.3s | 非拍身体压桌（防作弊） |

---

## 9. 环境 / PPO 超参
| 参数 | 位置 | 当前值 | 作用 |
|---|---|---|---|
| `decimation / sim.dt / episode_length_s` | env `__post_init__` | 4 / 0.005 / 10 | 50Hz 控制、200Hz 仿真、500 步/回合 |
| `num_envs` | RobotSceneCfg | 4096 | 并行环境 |
| `learning_rate` | agents/rsl_rl_ppo_cfg.py | 5e-4 adaptive | 学习率（续训防崩可降到 1e-4） |
| `max_iterations` | 同上 | 180000 | 训练上限 |
| `num_steps_per_env` | 同上 | 24 | rollout 长度 |
| `entropy_coef / desired_kl` | 同上 | 0.005 / 0.01 | 探索 / 自适应 LR 目标 KL |
| `experiment_name` | 同上 | ""（=任务名） | log 目录名 |

---

## 10. 常见场景速查（recipe）

| 症状 | 改哪里 |
|---|---|
| **拍面卡 0.80 / 反手过不了** | ① imit_orient_anneal（降腰+右手模仿，§6.5）② `success_ori_cos_dist_thresh_backhand` 0.20→0.22 先离开刀刃 ③ `_REWARD_SHAPE_TIERS` 的 σ_ori 再压 |
| **不用腰调拍面** | imit_orient_anneal 把 `waist_yaw` 权重降（已含）；或临时调小 stall_iters 让它更快降 |
| **击球无力（HITTER）** | 本质是 v_racket_hat 目标温柔（vel_fail 已很低）。真要力量→HITTER-REAL 续训调 planner（return_flight_time↓/target↑/paddle_cor↓）+ 加出球速度奖励 |
| **hsr 涨不动 / shape_tier 卡** | 多半被 cos_sim_ema 或某成功率 EMA 卡在 tier 门；看是哪条 EMA，放松对应 `_REWARD_SHAPE_TIERS` 门 或 `phase_thresholds` |
| **Stage2(球速)解锁不了** | `v_unlock_hsr`(0.85)/`v_unlock_cos_sim`(0.55) 太高且 hsr/cos 卡；先解决拍面让 hsr 上去，或临时降门 |
| **cos_sim_ema 贴 0.45 反复冻结** | 已修为读 strike-instant；若仍冻结，确认 `cos_sim_collapse_threshold` 没误触 |
| **崩溃（reward→−1e22）** | action 惩罚已 bounded + clip；若仍崩，查是否新加了无界惩罚项 |
| **resume 续训** | LR 降到 1e-4 防 fresh critic 砸坏 actor；`phase_1_min_iters` 用小值快进 Phase2 |
| **play 自碰撞/看不清** | 那是 play 的 `post_outcome_hold_time` OOD（已隔离到 RobotPlayEnvCfg）；训练不受影响 |

---

## 11. 关键 TensorBoard 曲线 ↔ 参数
| 曲线 | 反映 | 关联旋钮 |
|---|---|---|
| `Metrics/pingpong/cos_sim_at_strike_forehand/backhand` | **真实击球拍面**（v64新） | success_ori_thresh、σ_ori、imit_orient_anneal |
| `Metrics/pingpong/hit_success_rate` + `hsr_*_only` | 命中率 | 全部 success_thresh + 课程门 |
| `Metrics/pingpong/hit_success_pos/vel/ori_fail_rate` | 哪种失败为主 | 对应 thresh / σ |
| `Curriculum/pingpong/cos_sim_ema` | 课程门控的拍面 EMA（守 0.45冻结/0.35回退） | v_unlock_cos_sim、collapse_threshold |
| `Curriculum/pingpong/shape_tier / std_g_ori / std_g_vel` | σ 收紧进度 | `_REWARD_SHAPE_TIERS` |
| `Curriculum/pingpong/v_in_mag_high / hit_y_max` | Stage2/3 是否解锁 | v_unlock_*/y_unlock_* |
| `Curriculum/task_phase/task_phase / task_phase_imit_w` | 当前相位 + 模仿强度 | el_phase_*、imit_w_phase、phase_1_min_iters |
| `Curriculum/.../imit_orient_weight` | 腰+右手模仿降到多少（v65） | imit_orient_anneal |
| `Metrics/pingpong/rarm_torque_sat_*` | 右臂力矩饱和（是否torque-limited） | action scale |
| `Train/mean_reward / mean_episode_length` | 总体健康（崩溃/站立） | — |

---

## HITTER-REAL 额外（继承本文件 + 真实球）
- 额外奖励：`ball_contact(2)/return_direction(0.5)/clear_net(1)/opponent_land(3)/target_land(2)/illegal(-2)`（`mdp/real_rewards.py`，权重在 hitter_real RewardsCfg）。
- 额外终止：`ball_dead`。
- 真实球"力量"旋钮（`RealPingpongCommandCfg`）：`return_flight_time(0.45)`、`target_land((2.45,0,0.78))`、`paddle_cor(0.85)`、`serve_max_speed(6.0)`——降 flight_time / 推远 target / 降 cor 都会抬高所需拍速。
- 真实球课程 `update_real_pingpong_curriculum`：按 hit→return→target 成功率收紧 `target_land_radius`、放宽 serve/hit_y。
- 续训要点：actor 可从 HITTER ckpt 直接 load（obs 一致）；critic 重新初始化（reward 变了）→ 建议 LR 降到 1e-4 预热。play-only 旋钮（hold_time/ball_dead/traj_len/reset_table）都隔离在 `RobotPlayEnvCfg`。
