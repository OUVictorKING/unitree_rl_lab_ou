# 训练问题与解决方案流程记录

记录 G1 23-DOF pingpong (HITTER 复现) 训练中遇到的问题、诊断证据、解决方案、有效性。按问题域分组。

状态图例：🟢 已验证有效 / 🟡 部分有效或需观察 / 🔴 无效已回滚 / ⚪ 已落地待验证 / 🔵 已确认非 bug

---

## 一、Reward / Curriculum 设计类

| # | 问题 | 触发现象 | 根因 | 解决方案（代码位置） | 验证方式 | 状态 |
|---|---|---|---|---|---|---|
| R1 | `goal_orientation` 用 `\|dot\|` 杀掉摆动方向梯度 | 早期 baseline 永远 ori_fail≈1，policy 不知道往哪个方向转 | 对称 \|dot\| 让 +n_target 和 -n_target 都给同样 reward，没有「正手 vs 反手」方向信号 | 改为 `sign(swing_type) × (n_blade · n_target)`，正手奖 +n、反手奖 -n （[rewards.py:129-139](mdp/rewards.py#L129-L139)） | run 2026-05-25_10-08-03 hit_success_rate 第一次涨过 0 | 🟢 |
| R2 | 但 R1 引入了 swing-while-falling basin | run 2026-05-25_10-08-03/11-01-44 出现 `hit_success=0.21 / vel_fail=0.001 / hard_contact=0.999` 的怪组合 | 信号干净了，policy 可以「边摔边挥」收 reward；window curriculum 早期就 ratchet 到 tier-1，进一步扼杀站立学习 | 加 EL-based 单调闩 latch 见 S1/S2 | run 2026-05-25_11-59-27 站起阶段不再"挥拍倒地" | 🟢 |
| R3 | sigma_g_pos 太紧 → policy 在击球瞬间被迫减速保位置，杀速度跟踪 | goal_velocity reward 一直涨不上去 | 收紧 sigma_g_pos 让 pos reward 主导，policy 牺牲速度精度 | sigma_g_pos 设地板 0.04（[curriculums.py:271](mdp/curriculums.py#L271-L272)） | vel reward 起得来 | 🟢 |
| R4 | 击球后姿势怪异，policy 不收尾 | body_dominant run 2026-05-23_18-16-51 出现"尾随期 weird pose" | imitation reward 在 t_to_hit ≤ 0 仍然要求跟 clip → 限制了击球后自由调整 | 加 `gate_pre_strike=True`，t_to_hit≤0 时 imitation reward 置 0（[rewards.py:36-38](mdp/rewards.py#L36-L38)，[hitter_env_cfg.py:268-282](robots/g1_23dof/hitter/hitter_env_cfg.py#L268-L282)） | 后续 run 击球后姿态自然 | 🟢 |
| R5 | imit_anneal 过早把 w_i 砍到 0.15 | run 2026-05-24_07-52-04 跑 33k iter EL≈30 仍未站起，但 iter 8000 时 anneal 已切到 phase 2 | 三段 anneal 只看 iter 数，不管 policy 是否站起；从 scratch 时早期需要强 imit 信号 | 加 `min_ep_length_for_phase_advance=250`：EMA(EL)<thr 时强制 phase=0；<2thr 时 ≤1（[curriculums.py:168-177](mdp/curriculums.py#L168-L177)） | 后续 from-scratch run 在站起前 w_i 保持 0.5 | 🟢 |
| R6 | 切换到非对称 npz 后，goal_base weight 校准失败：跑了 3 个 run 才找到对的值 | 三连 fail（同 npz forward_003/backward_001，要求 base 横移 ~0.46m）：<br>① **w=0.3** (run 14-04-32)：`base_y_drift_meanabs≈0.10m` 远不够，`hsr=0` 持续 3000 iter（base 不动）；<br>② **w=2.0 直接** (run 15-34-31)：`EL=40→269`、`hard_contact≈0.99`、`cos_sim_ema 0.25→-0.01`、`hsr=0` 全程（站不起来）；<br>③ **ramp 0.5→2.0** (run 16-45-17)：站立 OK (EL=470)，但 `cos_sim_ema 从 +0.13 跌到 -0.08 长期不回升`；**`pos_vel_gate_open=1` 但 `w_goal_pos=w_goal_vel=w_goal_ori=0` 永远**；`shape_tier=0` 永远；`Episode_Reward/goal_position/velocity/orientation` 全为 0；hsr 卡 0.014~0.032 共 6000+ iter | **量级失衡**——reward 之间梯度争夺：goal_base 太小 (0.011 vs imitation_body_pos 0.074, 差 6.7×) 时 policy 忽略 base；goal_base 太大 (ramp 完成后 0.317 vs pre_strike rewards 0.02~0.07, 差 5~150×) 时 policy 抛弃 paddle 朝向。<br>**+ Layer 2 chicken-and-egg**：strike-window 权重只由 tier ratchet 抬升，前置 `cos_sim_ema ≥ 0.50` 否则 `cos_sim_ratchet_freeze=True`（[curriculums.py:683](mdp/curriculums.py#L683)）→ goal_base 量级失衡把 cos_sim 压到 < 0.50 → ratchet 永久冻结 → strike-window 权重永远 0 → 永远拿不到「击球质量」直接信号 | (a) 平滑 ramp 课程 `_GOAL_BASE_RAMP`：window_ep_ema ∈ [50,250] 线性从 0.5 升到 target；ep_ema<50 用 0.5（站立先学），ep_ema≥250 saturate（[curriculums.py:655-667](mdp/curriculums.py#L655-L667)） + (b) target 设 **0.8** 而非 2.0（[hitter_env_cfg.py:307](robots/g1_23dof/hitter/hitter_env_cfg.py#L307)）。**校准原则**：让 goal_base 量级 ≈ goal_position_pre_strike ≈ imitation_body_pos（即 0.10~0.12），各 reward 不互相抢梯度 | run 2026-05-26_20-52-38（ramp 0.5→0.8）：iter 1787 cos_sim_ema 突破 +0.50 解冻，iter 2500 `shape_tier=2.0`，`w_goal_pos/vel/ori` ratchet 升到 8/8/2.5；iter 3227 `hsr=0.524`、`cos_sim_ema=0.45`、`base_y_drift=0.62m`（base 跟踪与击球同时学） | 🟢 |
| R7 | window/v_in/y 三个课程并行推进，policy 在 7-14k 后陷入 reward-hacking 直至 actor `std<0` 崩溃 | run 2026-05-26_20-52-38 自闭环 28k：iter 7-14k 峰值 `hsr=0.476 / cos_sim_ema=0.448`；iter 14k 后系统性退步——`hsr 0.476→0.167`、`vel_fail 0.40→0.75`、`shape_tier 1.32→0.00`、`cos_sim_ratchet_freeze` 永久锁 1.0；iter 28000+ 训练 `RuntimeError: normal expects all elements of std >= 0.0` 自动停止（PPO actor std 崩负） | **三课程并行 + monotone-only ratchet**：14k 时 shape_tier 才 1.3，但 v_in_mag 已被推到 2.71、hit_y_max 推到 -0.067；policy 找到 sub-optimal corner（只追 base_pos，放弃 vel/ori/imit）→ `goal_velocity reward 0.007→0.0017`、`imitation_joint_pos 0.008→0.0007`；cos_sim 持续侵蚀但 v_in/hit_y 没有反向退档机制 → 难度只升不降 → policy 越走越偏 → actor std 数值崩溃 | (a) **Sequenced curriculum**（[curriculums.py:535-580](mdp/curriculums.py#L535-L580)）：Stage 1 只推 window；Stage 2 解锁条件 `shape_tier ≥ 4 AND hsr_ema ≥ 0.85 AND cos_sim_ema ≥ 0.55`；Stage 3 解锁条件 `v_in_high ≥ 3.5 AND hsr_ema ≥ 0.80`<br>(b) **cos_sim 崩溃反向 ratchet**：`cos_sim_ema < 0.35` 时主动把 v_in_high 退到 2.5、hit_y_range 退到 ±0.10（[curriculums.py:582-600](mdp/curriculums.py#L582-L600)）<br>(c) 在 [hitter_env_cfg.py:460-490](robots/g1_23dof/hitter/hitter_env_cfg.py#L460-L490) 暴露 9 个新开关 | 待新 run 验证：从 model_10000.pt 续训 + 应用 R7 课程后，预期 v_in_high 在 hsr 未到 0.85 前停在 2.5（不再被推到 2.71）、cos_sim_ema 跌到 0.35 时 v_in 自动退档 | ⚪ |
| R8 | paddle / body 一直把桌子当机械支撑 ⇒ stand-up reward 被作弊路径绑架 | from-scratch 早期 `non_paddle_table_stuck` 起作用前，policy 已学会拍面/手肘搁桌面保平衡（contact 力小但持续）。一旦后期把 weight ramp 起来，policy 的「不撑桌」与「学站」两个梯度冲突，hsr 与 EL 同步崩；R7 sequenced curriculum 也无法兜底因为 cos_sim_ema 会被「拍面贴桌」直接污染 | 桌子在 stand-up 阶段是噪声源：i) 给反馈早（policy 还没站稳就开始撞桌），ii) 给信号杂（与 imitation/goal_base 抢梯度），iii) 即使设小 weight 仍能机械学到「靠在桌子上」 | **Stage-wise table curriculum**（4 阶段状态机）：<br>**Stage 0 hidden**：`table.init_state.pos=(1.77,0,-10)` 沉地下；`paddle_table_contact.weight=0`、`body_table_contact.weight=0`；`non_paddle_table_stuck` 通过 `env._pingpong_table_active=False` 早返回 zeros（[terminations.py:42-44](mdp/terminations.py#L42-L44)）<br>**Stage 1 unlocked**：解锁条件 `hsr_ema≥0.65 AND cos_sim_ema≥0.50 AND ep_length_ema≥400 AND iter≥1500`（全 batch 均值，[curriculums.py update_table_guard_stage](mdp/curriculums.py)）；翻 `_pingpong_table_active=True` flag。**不主动 teleport**——靠 `reset_table_position_by_stage` EventTerm 在每个 env 下次 reset 时自然搬桌子上来（[events.py reset_table_position_by_stage](mdp/events.py)）<br>**Stage 2 ramping**：等 `ramp_iters/4` iter 让所有 env 都 reset 过一遍；之后 weight 0→target 线性 ramp 500 iter<br>**Stage 3 active**：weight=(-10,-1)，`non_paddle_table_stuck` 启用，等价于原版 HITTER from-scratch | run 2026-05-27_12-07-21（R8 from-scratch 首跑）：iter 0-5337 站立先行（EL 25→442），iter 5337-6746 cos_sim 突破 0.50 一次解冻 → shape_tier 0→1，hsr 0→0.47，w_goal_pos/vel/ori ratchet 启动 0→3/3/1（详见 R9）；table_stage 仍 0（cos_sim_ema=0.46 还差 0.04），符合预期 | 🟡 |
| R9 | shape_tier 升档时 fail-rate metric 反向上升，被误判为「policy 退步」 | run 2026-05-27_12-07-21 iter 5337→6746：`shape_pos_ema(fail) 0.79→0.97` 看似 pos_fail 暴涨，但同期 hsr 0→0.47、shape_hsr_ema 0→0.45 都在涨 | shape_tier 0→1 升档后 sigma_g_pos / std_g_vel / std_g_ori 全部收紧（tier-1 标准更严），用同一阈值算 fail rate 自然反向上升——这是「升档代价」不是退步。**真实进步看 hit_success_rate / shape_hsr_ema，而非 shape_*_ema(fail)** | （非 bug，写为约定）判读规则：<br>- `shape_*_ema` 是「相对 tier 标准」的 fail rate，跨 tier 比较无意义<br>- 跨 tier 比较只用：`hit_success_rate`、`mean_episode_length`、`vel_success_rate`<br>- shape_tier 升档后 fail rate 反升 ≤ 0.20 是预期；> 0.30 才需警惕 | run 2026-05-27_12-07-21 iter 6746：shape_pos_ema(fail) 0.97 但 hsr=0.47——确认是升档代价，不需干预 | 🔵 |
| R10 | `goal_velocity` Gaussian-on-squared-L2 公式 + 5 档过粗 std 表 → vel reward landscape 早期全平、跨档崩 | **两阶段实证**：<br>**阶段 1**（run 2026-05-27_12-07-21 iter 8000+）：plateau 期 `goal_velocity` reward 卡死 **0.003**（vs pos=0.175, ori=0.047），cos_sim_ema 卡 0.46-0.48 摸不到 freeze 阈值 0.50，hard_contact 翻倍 0.020→0.038（policy 用力过猛追 pos 唯一活信号）。<br>**阶段 2**（run 2026-05-27_16-45-31 from-scratch ×1.2 退档表）：iter 5359（std=0.30 tier 0）vel reward 历史峰 **0.0053**（公式生效证据），但 monotone latch 收紧到 std=0.20（tier 2，0.05 跳）后 iter 5359→6900 vel reward 跌到 **0.0021 横盘 1500 iter**——证明 5 档表的 0.05 std 步长在 linear-exp 下仍造成 60% reward 衰减 | (a) **公式问题**：`exp(-err²/std²)`（squared L2 + Gaussian）等价 `exp(-(d/std)²)`，d=1.0 m/s std=0.35 时 reward=3e-4，d=1.5 时 3e-9——早期 from-scratch policy vel error 普遍 1.0-2.0 m/s 区间**完全无梯度**。<br>(b) **跨档崩问题**：5 档表 std_vel 步长 0.05/0.05/0.05/0.05，每档 reward 衰减 ~50-60%，policy 没有充分时间消化每档精度要求 | (a) **公式改 linear-exp**（[rewards.py:97-107](mdp/rewards.py#L97-L107)）：`err = ‖v_blade - v_hat‖`（去掉 square），`return torch.exp(-err / std) * gate`（去掉 std²）。**仅 strike-window**——`goal_velocity_pre_strike` 保持原 Gaussian-on-squared 公式（pre_strike 不需精确跟踪）。<br>(b) **5 档 → 7 档**（[curriculums.py:158-177](mdp/curriculums.py#L158-L177)）：std_vel 列改 `0.20/0.23/0.26/0.29/0.32/0.35/0.38`（步长 0.03，每档衰减 ≤ 38%）；sigma_pos 列 `0.06/0.08/0.10/0.13/0.18/0.24/0.30`；std_ori 列 `0.20/0.22/0.25/0.28/0.32/0.36/0.40`。tier 6 保持 paper-strict (0.06/0.20/0.20)；tier 0 fallback 不变。<br>(c) **顶档常量同步** [curriculums.py:449](mdp/curriculums.py#L449) `v_unlock_shape_tier: 4 → 6`、[hitter_env_cfg.py:525](robots/g1_23dof/hitter/hitter_env_cfg.py#L525) 同步 → R7 sequenced curriculum Stage 2 解锁仍要求 paper-strict 顶档。<br>(d) **freeze 阈值松动** [curriculums.py:447](mdp/curriculums.py#L447) `cos_sim_freeze_threshold: 0.50 → 0.45`、[hitter_env_cfg.py:556](robots/g1_23dof/hitter/hitter_env_cfg.py#L556) `min_cos_sim_ema: 0.50 → 0.45`，避免 EMA 在 0.46-0.48 反复 freeze 1↔0 震荡 | 待 from-scratch 重训验证：预期 (i) tier 升档时 vel reward 衰减 ≤ 38%（不再跨档崩）、(ii) 每 1500-2500 iter 升 1 档（之前 tier 2 卡 1500 iter）、(iii) shape_tier 不再回退 | ⚪ |
| R11 | V3 swing-first 50/50 强制采样 + 去 critic swing_type → cos_sim 长期负值，hsr 永久卡 0.18 | run 2026-05-28_19-52-00（V3）跑 26000 iter：站立超快（iter 5000 EL=470），但 cos_sim 长期 -0.10 ~ -0.15、shape_tier 全程 0、hsr=0.18 plateau。对比 V1 21-04-08 同 iter cos_sim=+0.46/hsr=0.51 | 两个改动叠加：(a) `swing-first` 在 hit_y 采样前先 Bernoulli(0.5) 选 swing_type → 强制 50:50 fh:bh 分布；(b) critic obs 移除 swing_type → critic 失去判别 swing 类型的特权信息。结果：50% 反手 episode 把 cos_sim 拖到负值，policy 没法靠 forehand 早期专精→突破→泛化的路径 | 全部撤销，回到 V1 uniform sampling + 后置 `_compute_swing_type` 分类（[commands.py:341-419](mdp/commands.py#L341-L419)）；critic obs swing_type 保持注释掉（V1 baseline 不需要）；同时删除 `goal_base_orientation` reward（V3 配套引入的实验项）。**关键经验**：用户的"track 早期再 free"假设被自身数据反驳——iter 33000 V1（free + 低 shape_tier）实战 hsr 优于 iter 8000 V1（free + 高 shape_tier rigid pull），说明 LESS imit pressure 才是对的方向 | run 2026-05-29_14-54-15 同 iter 1000：cos_sim=+0.585 EL=462 — 超过 V1 21-04-08 任意 iter 历史最高（+0.48 at iter 8000） | 🟢 |
| R12 | `gate_pre_strike: True → False` 三个 imit 项全开 → strike 帧 imit_body_pos 污染 paddle 朝向梯度，cos_sim 崩到 -0.76 | run 2026-05-29_12-07-26 跑到 iter 3000：站立超快（EL=498），但 cos_sim 单调下跌 -0.16 → -0.76 → -0.52；reward 拆解显示 imit_body_pos=+0.281（最大正向，34% 总信号）vs goal_orientation_pre_strike=+0.0002（几乎死信号）。比例 1400:1，policy 必然先优化 body_pos | (a) **量级失衡**：gate=False 让 body_pos 在所有帧开火（之前只 pre-strike），episode 平均贡献 ×2；(b) **strike 帧污染**：`body_pos = exp(-k·Σ‖p_rel - p̂_rel‖²)` 用 link xy 位置（pelvis-anchored），位置近似就给奖；strike 帧 goal_orientation 要求 paddle 法向精确对齐 — 两者在同一 1-2 帧上抢梯度，body_pos 用"位置近似"压过 goal_orientation 的"朝向精确"。joint_pos/vel 没此问题（关节角直接决定 paddle 朝向，且权重小） | **Plan B 半 gate**：`imitation_joint_pos / imitation_joint_vel: gate_pre_strike=False`（保留 post-strike 关节级跟踪——demo 的 follow-through 关节角自然回 ready）；`imitation_body_pos: gate_pre_strike=True`（仅 pre-strike，让 strike 帧 paddle 朝向只受 goal_orientation 主导）（[hitter_env_cfg.py:289-303](robots/g1_23dof/hitter/hitter_env_cfg.py#L289-L303)） | run 2026-05-29_14-54-15 同 iter 1000：imit_body_pos 从 0.281 砍到 0.07，goal_orientation_pre_strike 从 0.0002 涨到 +0.012（活了），cos_sim 翻为 +0.58 | 🟢 |
| R13 | V1 23dof 基座移动时跳动 + 后期摆动倒地，缺 base 稳定 reward | (a) 用户播放 V1 model_3000.pt 观察"机器人移动时是跳过去的"；(b) 29dof run 2026-05-29_11-13-14 `bad_orientation` 99.9% 终止 8000 iter；(c) reward 表对比 locomotion baseline，缺 `lin_vel_z_l2` / `ang_vel_xy_l2` / `energy` 三项；feet_slide 量级偏小 | pingpong 长期只用 `pelvis_orientation_l2 (-1.0)` + `pelvis_height_l2 (-5.0)`，缺 vertical bounce 抑制（lin_vel_z）、roll/pitch rate 抑制（ang_vel_xy）、和 energy reg。挥拍反作用力会激发 base 摆动→倒地，没有 rate-damping 信号 | (a) `pelvis_ang_vel_xy = -0.05`（roll/pitch only，不罚 yaw — 挥拍要 yaw）（[hitter_env_cfg.py:350](robots/g1_23dof/hitter/hitter_env_cfg.py#L350)）；(b) `pelvis_lin_vel_z = -0.8`（locomotion 默认 -1.5 的 0.5×，留余量给击球时合法腾跃）（[hitter_env_cfg.py:354](robots/g1_23dof/hitter/hitter_env_cfg.py#L354)）；(c) `energy = -2e-5`（locomotion 默认值，需在 [pingpong/mdp/__init__.py](mdp/__init__.py) 加 `from unitree_rl_lab.tasks.locomotion.mdp.rewards import energy`）（[hitter_env_cfg.py:340](robots/g1_23dof/hitter/hitter_env_cfg.py#L340)）；(d) `feet_slide` 权重 -0.08 → -0.20（防止拖脚移动，鼓励抬脚换位）（[hitter_env_cfg.py:357-359](robots/g1_23dof/hitter/hitter_env_cfg.py#L357-L359)） | run 2026-05-29_14-54-15 同 iter 1000：hard_contact 从 V1 同期 0.99 → 0.20 (5× 改善)，bad_or=0.005，pelvis_ang_vel_xy reward -0.014/step、pelvis_lin_vel_z -0.010/step（量级符合预期），feet_slide -0.048/step（说明在抬脚） | 🟢 |
| R14 | 29dof imit `gate_pre_strike=True` + 短 episode → imit 信号饥饿，PPO 学不到站立姿态 | run 2026-05-29_11-13-14 跑 8735 iter EL 永远 32-34，bad_or=99.9%，imit total 仅 0.09（23dof V1 同期 0.18，差 2×）；fh_share=0.50（无模式偏好，纯随机摔） | 鸡生蛋死循环：站不住→episode 短(EL=33)→pre-strike 帧极少(t_to_hit > 0 帧 ≈ 10)→`gate_pre_strike=True` 把 imit 在剩下 90% 帧全部清零→imit reward 几乎不发→没姿态信号→站不住 | 把 23dof Plan B 的 `gate_pre_strike` 配置移植到 29dof：`imit_joint_pos / imit_joint_vel: True → False`（让 imit 在所有帧开火，包括摔倒前的早期帧），`imit_body_pos: True` 保留（同 23dof Plan B 防 strike 帧污染）（[g1_29dof/hitter/hitter_env_cfg.py:387,392,397](robots/g1_29dof/hitter/hitter_env_cfg.py#L387-L397)）；同时移植 R13 的全部 4 项 base 稳定 reward | 待新 from-scratch 验证；预期 imit 早期信号 ≥ 0.10/step（不再饥饿）、EL iter 1500 ≥ 50 | ⚪ |

## 二、Stand-up gate 类

| # | 问题 | 触发现象 | 根因 | 解决方案 | 验证方式 | 状态 |
|---|---|---|---|---|---|---|
| S1 | window curriculum ratchet 过早跳 tier-1 | run 2026-05-25_10-08-03 iter 500 时 hit_success≈0.20 但 EL=40，window 已被 ratchet 到 0.06，weight 拉到 3/3/1 | hit_success_rate 单一指标无法区分"站着击中"和"摔倒过程中蒙到" | 加 `min_ep_length_for_window_advance=250` 闩，EMA(EL)<thr 时不允许 ratchet 推进（[curriculums.py:316-348](mdp/curriculums.py#L316-L348)） | run 2026-05-25_11-59-27 stand-up 阶段 window 一直是 0.10 | 🟢 |
| S2 | signed-ori reward 在站不稳时也给信号 | run 2026-05-25_11-01-44 EL=40 stuck 2000 iter 不突破 | signed-ori 干净的方向梯度 + 宽 vel std=0.5 → swing-while-falling basin 自洽 | 加 `min_ep_length_for_ori_advance=250` 单调 latch：EMA(EL)<thr 时强制 w_goal_ori=0；开了不再关（[curriculums.py:64](mdp/curriculums.py#L64)，[curriculums.py:326-331](mdp/curriculums.py#L326-L331)） | 修复后站起阶段 goal_orientation reward 强制为 0，policy 必须先学站 | 🟢 |
| S3 | EMA 与 curriculum 顺序导致 1-tick 滞后 | imit_anneal 写 EMA → pingpong 读 EMA，跨 tick 老 1 帧 | CurriculumCfg 字段顺序敏感 | 把 `imit_anneal` 放在 `pingpong` 之前（[hitter_env_cfg.py:367-419](robots/g1_23dof/hitter/hitter_env_cfg.py#L367-L419)） | 不再观察到 1-tick 错位 | 🟢 |

## 三、Motion clip / RSI 类（最关键）

| # | 问题 | 触发现象 | 根因 | 解决方案 | 验证方式 | 状态 |
|---|---|---|---|---|---|---|
| M1 | **RSI 只写 joint_pos 不写 root_quat** ← 主 bug | run 2026-05-25_11-59-27 站起后 (EL>250) `cos_sim_n_blade_n_target_at_impact ≈ 0`、`ori_fail ≈ 0.49` 长期不动；imit 实现率正常但 ori reward 拿不到 | motion clip 在 impact 帧 pelvis_yaw=+63.6°(forehand) / +128.8°(backhand) 是侧身姿态；`_write_rsi_joint_state` 只复制关节角，base 由 `_write_nominal_root` 写随机 ±10° → 关节"按侧身姿态摆"但 base 朝前 → 世界系 n_blade 整体被反向旋转 60°+，与 n_target 永远偏离 | 1) `motion_loader.py` 加 `pelvis_yaw_at_frame(frames)`<br>2) `commands.py` 加 `_sample_rsi_frames(ids)`<br>3) `_sample_new_swing(reset_robot=True)` 在 swing_type 决定后用 clip pelvis_yaw + ±10° noise 覆写 root_quat<br>4) `_write_rsi_joint_state` 接受预采 frames 保证 root 与 joint 同帧（[commands.py:311-340](mdp/commands.py#L311-L340)，[commands.py:393-425](mdp/commands.py#L393-L425)，[motion_loader.py:104-110](mdp/motion_loader.py#L104-L110)） | run 2026-05-25_14-51-08 iter 10 cos_sim 立刻跳到 0.45，iter 100 峰 0.79，last-50 mean 0.68（≈ forward clip 本征值 0.67）；vs 基线同期 cos_sim≈0 | 🟡 几何已确认；reward 端待 EL>250 latch 开启后验证 |
| M2 | `p_base_xy_world` 用 reset 时 yaw 而非 impact yaw 算 | 一直如此，但被 M1 掩盖 | `_compute_base_target` 拿 `expert_offset_base`（pelvis 系）转到世界系时用了「当前 yaw」而非「clip impact 帧 yaw」 | 暂未修。M1 修复后用「采样帧 yaw」相比之前「±10° 随机」已强相关于 impact yaw（[commands.py:333](mdp/commands.py#L333)，[commands.py:377-380](mdp/commands.py#L377-L380)） | M1 验证完看 `goal_base_position` reward 实现率 | ⚪ 待 M1 完整验证后判断是否需补 |
| M3 | imitation_joint_names 包含 right_shoulder_yaw + right_elbow → fine-tune 时拍面方向被卡死 | fine-tune run cos_sim 卡 0.52 不涨 | 这两个关节直接控制拍面法向，clip 关节角与 ball-physics 决定的 n_target 不一致时无解 | 用户手动从 imitation_joint_names 移除（[velocity_env_cfg.py 等](robots/)） | run 2026-05-23_15-51-30 拍面解锁 | 🟢 |
| M4 | **`BLADE_NORMAL_LOCAL` convention 反了** ← 根本性 bug | V1 21-04-08 训练 cos_sim_metric=+0.48 看似"成功"，但 iter 8000 用户播放时观察到 "手臂打直照搬 demo + 拍面正确" 但姿态僵硬不自然；iter 33000 shape_tier 退到 0 反而比 8000 实战更好 | URDF `g1_23dof_rev_1_0_paddle.urdf` 的 fixed joint rpy=`-2.356 0 0`（-135° 绕 X），把腕部 +Y 旋到了 paddle frame 的"身后偏下"方向。**实测 NPZ 数据**（`forward_003_rotated.npz` impact_frame=50）：paddle 局部 -Y 方向在世界 (+0.652, +0.719, +0.241) 朝向球台，是真正的正手击球面；而代码用 `BLADE_NORMAL_LOCAL = (0, 1, 0)` 即 +Y 当正手面，**和 URDF 反**。结果 V1 policy 学会的是"扭手腕 180° 用反手面打正手"——cos_sim 数值正确但物理姿态扭曲，所以僵硬；iter 33000 退档后 policy 放弃错误奖励反而趋向真实击球姿态 | 1 字符修复 [commands.py:38](mdp/commands.py#L38)：`BLADE_NORMAL_LOCAL = (0, 1, 0) → (0, -1, 0)`。语义校正：`n_blade_world` 现在指向正手面（URDF -Y）；正手 sign=+1 reward `+正手面 · n_target` ✓；反手 sign=-1 reward `-正手面 · n_target` = `+反手面 · n_target` ✓。`rewards.py:125` / `commands.py:551,641` 三处都通过 `from .commands import BLADE_NORMAL_LOCAL` 自动同步 | run 2026-05-29_14-54-15 同 iter 1000：cos_sim=+0.585（V1 历史 iter 8000 才达到 +0.48），且**这是物理正确的对齐**（不再扭手腕） | 🟢 |
| M5 | **历史 cos_sim 数值反号警告** | V1 21-04-08 / V2 / V3 等所有用 `BLADE_NORMAL_LOCAL=(0,1,0)` 时期的 cos_sim 数值，含义是「`+正手面（URDF 反手面）· n_target`」——和物理意义相反 | M4 修正前所有训练都在用反着的 convention，cos_sim_ema 的"高低"是反的 | 跨 M4 边界比较 cos_sim 时，旧值 × (-1) 才是物理意义的真值。**新 baseline 数值要重新建立**，不要直接和历史 V1/V2/V3 数字横向对比 | 用户实证：V1 iter 33000（旧 cos_sim "退步"）实战击球率比 iter 8000（旧 cos_sim "峰值"）更好，说明 policy 在 cos_sim 反号下的最优解和 cos_sim 信号不一致；M4 修正后 cos_sim 才是物理 ground truth | 🔵 |

## 四、诊断过程中的误判（走过的弯路）

| # | 错误判断 | 反驳证据 | 修正 |
|---|---|---|---|
| W1 | "5-DOF 机械臂不可能在整个 strike window 保持拍面对齐" | 用户：腰部 yaw 自由度可补偿 | 改查 reward 实现 → 找到 M1 |
| W2 | 优先建议增大 `goal_orientation.std` | 用户：先看 reward 是不是写错的 | 严格按"先排查实现 → 再调参"的顺序 |
| W3 | 错查 `g1_23dof_paddle.urdf`（rpy=0） | 用户：URDF 加了一定角度 | 通过 [unitree.py:940-944](../../assets/robots/unitree.py#L940-L944) 确认实际加载的是 `g1_23dof_rev_1_0_paddle.urdf`（rpy=`-0.7854 0 0`，-45°） |
| W4 | 假设标准右手球员惯例：-Y=正手 | 用户：+Y=正手，-Y=反手 | 与 `y_mid_base=+0.157` 一致；motion clip forehand blade Y=+0.208 也佐证 |
| W5 | 看 `Metrics/pingpong/hit_success_rate=0` 就判 "policy 5337 iter 完全没击中过球" | 同 run 同 iter 的 `Curriculum/pingpong/hit_success_rate=0.122`、`shape_hsr_ema=0.118` 都在涨；用户提示「3 个失败率也在下降」 | 两个 namespace 含义不同：`Metrics/...` 只在 episode 终结时累计（time_out=0.84 时大量样本被 censored）、`Curriculum/...` 是 step-level batch 均值，**真值看 Curriculum/pingpong namespace**；rotated NPZ 不是 hsr=0 的根因（被这个误判带偏一回） |
| W6 | 看 `shape_pos_ema(fail) 0.79→0.97` 判 "policy 在退步" | 同期 `hit_success_rate 0→0.47`、`shape_tier 0→1`、`mean_EL 442→487` 全部上升 | shape_*_ema(fail) 是「相对当前 tier 的 fail rate」，升 tier 后 sigma 收紧自然反弹；详见 R9 |
| W7 | 把 V1 21-04-08 cos_sim_metric=+0.48 判作"V1 已经成功" | 用户实测 model_8000 (cos_sim 峰值) vs model_33000 (cos_sim 退步)：8000 手臂僵硬照搬 demo，33000 反而用腰+臂自由挥拍，实战击中率更高 | M4 揭示 `BLADE_NORMAL_LOCAL` convention 反了——V1 时代的 cos_sim 数值方向是错的，policy 在 cos_sim_metric 高时实际是"用反手面打正手"，姿态扭曲。视觉判断"姿态不自然"才是真 ground truth |
| W8 | 推荐 `gate_pre_strike: True → False` 三个 imit 项全开时低估副作用 | 我评估"strike window 只 1-2 帧 imit 总贡献小"——实际 run 2026-05-29_12-07-26 跑出 cos_sim 崩到 -0.76 | 漏算了 imit_body_pos 在 body_dominant split=0.60×w_i 下是**最大正向 reward**（0.281/step），而 strike 帧 goal_orientation 信号仅 0.0002，比例 1400:1。即便 strike 只 1-2 帧，PPO advantage 也会被 body_pos 的位置近似奖励主导。修正：仅 joint_pos/vel gate=False，body_pos 保留 gate=True（Plan B，详见 R12） |
| W9 | 推荐"先模仿右臂再开放"（用户最初想法） | 用户自身数据反驳：iter 33000（右臂 free + 低 imit pressure）实战 hsr 略好于 iter 8000（右臂 free + 高 imit pressure rigid pull） | 直觉是"先把动作教对"，但 V1 已经把右臂 free 了（commands.py:679-691 V1 默认 imit 排除 `right_shoulder_yaw / right_elbow / right_wrist_roll`），iter 8000 看到的 rigid 不是 imit 不够强而是 cos_sim 反号导致 policy 在扭手腕。正确方向是修 cos_sim convention（M4），不是加 imit |

## 五、调试流程模板（应对类似问题）

1. **现象分类**：reward 拿不到 → 是 gate 问题、weight 问题、还是几何/物理问题？
2. **基线对照**：找一个稳定 baseline run，定位某个 metric 偏离基线多少 / 在哪个 iter 段开始偏。
3. **看几何指标**：
   - `cos_sim_n_blade_n_target_at_impact`（应该 ≥ clip 本征 0.67~0.92）
   - `imit_*_rate`（应该 ≥ 30%）
   - `mean_episode_length`（< 250 时所有打球指标都不可信，先看站立）
4. **排除非 bug 路径**：
   - 是 curriculum gate / latch 设计的预期 0 吗？
   - 是 ep_length < 阈值导致的 stand-up phase 锁死吗？
5. **再查实现**：从 motion clip → RSI → reward 函数，每一步验证「clip 数据 vs sim 数据」是否一致（用 NPZ 直接读，不靠 tensorboard）。
6. **最后才调超参**：std / weight / 阈值。

---

## 六、其他关键约定（避免重复踩坑）

- **swing_type 约定**：`+Y(body left) = SWING_FOREHAND(0)`，`-Y = SWING_BACKHAND(1)`，分界 `y_mid_base=+0.157`（不是标准右手球员惯例）
- **Blade 法向 (M4 修正后)**：`BLADE_NORMAL_LOCAL = (0, -1, 0)` 在 blade 局部系——指向**正手面**。URDF `g1_23dof_rev_1_0_paddle.urdf` 的 fixed joint rpy=`-2.356 0 0`（-135° 绕 X），导致腕部 +Y 旋到 paddle frame "身后偏下"方向；NPZ 实测正手击球瞬间 -Y 方向 (+0.652, +0.719, +0.241) 朝向球台。**M4 修正前用 (0, 1, 0) 是反的，所有历史 cos_sim 数值要 × (-1) 才是真值**
- **imit 半 gate 约定 (Plan B)**：`imit_joint_pos / imit_joint_vel: gate_pre_strike=False`（全程跟踪关节角，post-strike 自然回 ready）；`imit_body_pos: gate_pre_strike=True`（只 pre-strike，避免 strike 帧 body_pos "位置近似" 信号压过 goal_orientation "朝向精确" 信号）。详见 R12
- **base 稳定 reward 套餐**：`pelvis_orientation_l2 (-1.0)` + `pelvis_height_l2 (-5.0)` + `pelvis_ang_vel_xy (-0.05)`（仅 roll/pitch，yaw 自由）+ `pelvis_lin_vel_z (-0.8)`（防跳）+ `feet_slide (-0.20)`（防拖脚）+ `energy (-2e-5)`。详见 R13
- **`energy` 函数 import**：`unitree_rl_lab.tasks.locomotion.mdp.rewards.energy` 不在 `isaaclab.envs.mdp` 里，必须在 `pingpong/mdp/__init__.py` 加 `from unitree_rl_lab.tasks.locomotion.mdp.rewards import energy`
- **swing-first 采样禁用**：V3 实验证明 forced 50:50 fh:bh + 移除 critic swing_type 让 cos_sim 长期负值（R11）。保持 V1 uniform sample + 后置 `_compute_swing_type` 分类
- **RSI 范围**：`reset_yaw_noise ±10°` 是叠加在「clip 采样帧 pelvis_yaw」之上的扰动，不是绝对范围
- **Motion clip 加载**：`g1_23dof_rev_1_0_paddle.urdf`（带 -135° 拍面安装角，**注意不是 -45°**），不是 `g1_23dof_paddle.urdf`
- **从 scratch 训练阶段**：EL>250 之前所有打球指标都被 latch 锁住（这是设计），看 EL / hard_contact 才有意义；详见 [memory/project-pingpong-from_scratch-phases](../../../../../../../root/.claude/projects/-mnt-workspace-unitree-rl-lab/memory/project_pingpong_from_scratch_phases.md)
- **Tensorboard namespace 权威性**：`Metrics/pingpong/*` 只在 episode 终结时累计（早期 time_out 主导 → 大量样本 censored，会假性显示 0）；**判读 hit_success / cos_sim / fail rate 一律用 `Curriculum/pingpong/*` namespace**（step-level batch 均值，更可信）。两者数值偏差大时按 Curriculum 为准。
- **跨 tier 比较 metric**：shape_tier 升档时 `shape_*_ema(fail)` 会反向上升（sigma 收紧代价）；要判 policy 是否真的在进步，看 `hit_success_rate` / `mean_episode_length` / `vel_success_rate`，不要看 shape_*_ema 跨 tier 比较。
- **R8 table-guard 解锁条件**：`hsr_ema≥0.65 AND cos_sim_ema≥0.50 AND ep_length_ema≥400 AND iter≥1500`（全 batch 均值）。Stage 0→1 不主动 teleport，靠每个 env 下次 reset 自然搬桌子。预留 `ramp_iters/4` iter 让所有 env reset 一遍后再开始 weight ramp。
- **train.log 默认关闭 (R13 配套)**：`scripts/rsl_rl/train.py` 的 `--log_redirect` 默认 False（旧 `--no_log_redirect` 默认 True，已替换）。需要 stdout/stderr 落盘时手动加 `--log_redirect`，否则 `events.out.tfevents` 是唯一日志源
- **29dof 配置同步**：23dof 上验证好的 reward / gate 套餐通过 [g1_29dof/hitter/hitter_env_cfg.py](robots/g1_29dof/hitter/hitter_env_cfg.py) 同步；commands.py / rewards.py / curriculums.py 共享。`BLADE_NORMAL_LOCAL` 修正自动 inherit
- **历史 V1 21-04-08 baseline 的真实地位**：cos_sim_metric=+0.48 是**反 convention 下的虚假成功**（W7）。不要把它作为 cos_sim 高低的参考；当前 baseline 重新建立从 run 2026-05-29_14-54-15 起算

---

# v59 → v62 新增问题 (2026-05-29 / 30)

> v58.1 训练发现 backhand cheat basin 后开始的全面重构。新问题主要围绕：sampling 设计、curriculum 阀门、reward formula。

## 七、Sampling / 几何设计类（v59 → v62 重构）

| # | 问题 | 触发现象 | 根因 | 解决方案（代码位置）| 验证方式 | 状态 |
|---|---|---|---|---|---|---|
| **R15** | **v58.1 backhand cheat basin（fh_share=0.005）** | run 2026-05-29_16-24-48 iter 5000 实测：`hit_success_rate=0.70`、`cos_sim=0.75` 看似优秀，但 `swing_ratio_forehand=0.005`（**99.5% 反手**）。`base_y_drift_meanabs=0.38m` 持续 +y 偏 | `_sample_new_swing` 在 env-local 切 `cfg.hit_y_range` 两半区采样 hit_y_local（50:50 Bernoulli）；然后 `_compute_swing_type(root_pos, root_quat)` **用当前 base 位置重分类**。policy 学会把 base.y 移到 +0.4m，让 `hit_y_base = hit_y_world - base.y` 全部落到 backhand 半区 → 100% backhand 重分类 → policy 专精反手 | **v59 swing-first base-frame 采样**：(a) Bernoulli 抢先决定 swing_type（不再后置分类）；(b) 直接在 base frame 采样 `hit_y_base ∈ swing 半区`；(c) 用当前 root.xy + yaw 反推 `hit_y_world`；(d) 加 `cfg.hit_y_world_cap=1.0` 强制世界系硬边界（一边为空时翻转 swing_target）；(e) 删除 `_compute_swing_type` 重分类逻辑（[commands.py:347-526 v59](mdp/commands.py)）| run 2026-05-29_20-51-13 swing_ratio 严格 50:50（fh=0.501）；构造性等于 sampling 概率 → cheat basin 关闭 | 🟢 |
| **R16** | **v59 RSI 顺序 CUDA crash「Failed to set DOF positions」** | run 2026-05-29_20-36-36 启动几秒后 CUDA assert：joint pos OOB array access | `_sample_rsi_frames` 和 `_write_rsi_joint_state` 都读 `self.swing_type[ids]` 选 clip。v59 初版把 `swing_type = swing_target` 写在 RSI 块**之后** → RSI 用 STALE swing_type 选了 forehand clip 的帧，但 joint_pos 写入用了 NEW swing_target = backhand 的 clip 索引 → 帧索引 OOB | 把 `self.swing_type[ids] = swing_target` 移到 **Step 1b**（RSI 之前）；boundary override 后再 Step 5b 重写一次（[commands.py:367-373](mdp/commands.py#L367)+[commands.py:486-492](mdp/commands.py#L486)）| 后续 run 启动正常无 CUDA assert | 🟢 |
| **R17** | **v59 `hit_y_world_cap` 当成绝对世界坐标，env grid 上的 robot 全报废** | run 2026-05-30_00-43-25 全 4096 envs `Episode_Reward/goal_base = 0.0000` from iter 0；诊断 metric `diag_goal_base_err_mean=5457`（距离≈74m）、`diag_delta_y_absmax=127`（远超 0.5m 预期）、`phit_y_absmax=1.0`（被 clamp 到 cap） | Step 3 cap 公式 `cos*(±cap - root_pos[:, 1])` 把 cap=1.0 当**绝对世界 y**：`|hit_y_world| ≤ 1`。对 env at world y=126 的 robot 来说，要求 hit_y_world ∈ [-1, 1] → 与 forehand 半区 [-0.426, -0.188]（base frame）反推后**无交集** → both_invalid fallback 触发 → hit_y_base ≈ cap_lo=-127 → hit_y_world = root.y - 127 ≈ -1（被 cap 锁死）| **v60 锚定 divider 到 env_origin**：`divider_world = env_origins[:, 1] + y_mid_base`（不再是 `root.y + ...`）；同时 cap 解释改为**env-local**：`world_y_lo = env.y - cap, world_y_hi = env.y + cap`（[commands.py:443-453](mdp/commands.py#L443)）| run 2026-05-30_00-43-25 修复后 `delta_y_absmax=0.518`（预期 0.5m offset），`errMean=0.041`（5 万倍下降），goal_base reward 从 0 涨到 0.018 | 🟢 |
| **R18** | **v60 forehand-as-backhand 几何 cheat（policy 移 base 让正手命令做反手伸手）** | sim play model_5000 iter 9000：用户观察「机器人只会反手，正手命令通过移基座来反手击球」。Per-swing diagnostic 实证 iter 5000 `paddle_y_base_at_strike_backhand=-0.27`（demo 应是 +0.024，差 0.30m！），`cos_sim_backhand_only=-0.16`（拍面反向）| `divider_world = root.y + y_mid_base` 跟随 robot：policy 移 base.y 到 +0.4m → divider 跟着移到 env.y + 0.212 → forehand 范围变成 [env.y - 1, env.y + 0.212] → forehand 命令时 hit_y 在 robot 左方 0.5m+ → 物理上需要 cross-body 反手伸手 → policy 用 backhand pose 做"forehand"命令 | **v61 锚定 divider 到 env_origin**：`divider_world = env_origins[:, 1] + y_mid_base`（per-env 固定，不跟 root）。policy 移 base 不再能改变正反手分界线 → 移 base 只让 hit point 更难到达 → cheat 收益消失（[commands.py:443-453](mdp/commands.py#L443)）。结合 R20 / R21 的 3-phase 课程 + Phase 2 baseline reset 一起部署 | run 2026-05-30_18-59-49 iter 5500（Phase 2 启动后 +500 iter）：`paddle_y_base_at_strike_backhand=+0.017`（demo +0.024 ✓），`cos_sim_backhand_only=+0.39`（正向，拍面对了）| 🟢 |
| **R19** | **v60 σ monotone latch 死锁：EMA 跌 σ 不松，vel reward 卡 0.001** | run 2026-05-30_08-34-18 iter 4000 EMA 峰值时 σ_vel 收到 tier 3 (0.29)，iter 4000-7000 EMA 跌回 0.60（应该 fallback 到 tier 2 σ=0.32），但 σ_vel 卡死 0.29 → ‖Δv‖=2 时 reward = exp(-2/0.29) = 0.001 → 几乎零梯度 → vel_fail 卡死 0.40+ | `command.cfg.sigma_g_pos = max(min(cur, target), 0.06)` 是单调收紧 ratchet——只允许 σ 缩小不允许扩大。EMA 暂时升到 tier 3 阈值就把 σ 永久收到 tier 3 值；之后 EMA 跌回 tier 2 区间，σ 卡 tier 3 不退 → policy 看到的是「reward 永远很难拿」 | **删除 monotone latch**，改为 σ 双向跟随 EMA：`command.cfg.sigma_g_pos = max(sigma_target, 0.06)`（[curriculums.py:700-714](mdp/curriculums.py#L700)）；同样改 `gv_cfg.params["std"]` 和 `go_cfg.params["std"]`。EMA 升 σ 收紧、EMA 降 σ 松开。Floor (0.06 / 0.20 / 0.20) 保留作 paper-strict 顶档 | run 2026-05-30_10-57-57 σ_vel 在 0.32 ↔ 0.29 之间动态振荡（之前永远卡 0.29），shape_tier 在 1.94 ↔ 2.86 浮动 | 🟢 |
| **R20** | **v61 Phase 1 → 2 跳过（EL_ema 飙升把 phase 1 撑过整个区间）** | run 2026-05-30_15-41-23 iter 3500-3700：EL_ema 从 339 → 448 共 100 iter，phase 从 0 → **2**（跳过 1）。Phase 1 实际生效 ~50 iter，imit_w=1.0 没机会真的训练正反手 | 原 latch 只检查 `EL_ema >= threshold`：(a) Phase 0→1 阈值 350，(b) Phase 1→2 阈值 450。两个 if 在同一次 curriculum 调用里**顺序执行**。当 EL_ema 一帧之内跨过 350+450（policy 突然站稳），两个 if 都触发 → cur_phase 从 0 直接到 2 | **加 Phase 1 minimum duration**：cur_phase < 2 升级时 `phase_1_iters = iter_count - phase_1_entry_iter`，必须 ≥ `phase_1_min_iters = 2000` 才允许进 Phase 2（[curriculums.py:222-227](mdp/curriculums.py#L222)）。同时 latch 加 `phase_1_entry_iter` 字段记录进入时间 | run 2026-05-30_18-59-49 iter 3001 进 Phase 1，iter 5008 进 Phase 2（满足 EL≥450 + p1elapsed≥2007 ≥ 2000） — Phase 1 实际跑 2000+ iter | 🟢 |
| **R21** | **v61 Phase 2 后 goal_* weight 卡死 0（cos_sim_ratchet_freeze chicken-and-egg）** | run 2026-05-30_15-41-23 iter 4000-5000：phase=2, ori_gate=1, pv_gate=1, win_gate=1（全开），但 `w_goal_pos = w_goal_vel = w_goal_ori = 0` 全程不动；hsr=0；reward landscape 没有击球信号 | (a) Phase 0/1 中 `update_task_phase` 把所有 goal_* weight 设 0；(b) Phase 2 时 task_phase **不再**touch goal_*（让 pingpong window curriculum 接管）；(c) window curriculum 用 `max(current, target)` 单调 ratchet，但被 `cos_sim_ratchet_freeze` 锁住（cos_sim_ema=0.13 < freeze threshold 0.45）；(d) cos_sim 低是因为 Phase 0/1 没 goal_orientation 信号 → cos_sim 永远低 → freeze 永远锁 → goal_* 永远 0 | **Phase 2 入口一次性 reset goal_* 到 baseline**：`if prev_phase < 2 and cur_phase == 2: env.reward_manager.get_term_cfg(term).weight = baseline`（[curriculums.py:246-248](mdp/curriculums.py#L246)）。baseline 值 = env_cfg defaults: pos/vel=2.0, pre_strike=1.0, ori=0.5。一次性写入跳出 ratchet 死锁；之后 window curriculum 的 max() 只升不降，保持 ≥ baseline | run 2026-05-30_18-59-49 iter 5100：w_goal_pos=2.0（baseline 重置成功），iter 5200 ramp 到 3.0，iter 6000 ramp 到 5.0；hsr 从 0 涨到 0.50+ | 🟢 |
| **R22** | **v61 vel reward Laplacian 公式中等误差区无梯度（plateau hsr=0.45 长期不动）** | run 2026-05-30_18-59-49 iter 9000+：hsr 0.50 → 0.45 → 0.40 持续退步，`Episode_Reward/goal_velocity = 0.009/step`（其他 goal reward 0.18），`vel_fail` 卡 0.40-0.49。诊断算 ‖Δv‖ ≈ 2 m/s 时 Laplacian σ=0.29 给 reward = exp(-6.9) = 0.001 → policy 在 ‖Δv‖=[1, 4] 范围**完全无梯度** | (a) **公式问题**：`exp(-‖Δv‖/σ)` 在大误差时极陡（梯度趋零）；(b) **σ 太小**：σ=0.45 (default) 或 σ=0.29 (curriculum tightened)，对实测 ‖Δv‖=2 来说 reward 接近 0；(c) **paper 没讨论**：HITTER paper 没给 vel reward 具体公式或讨论 vel 学习困难；(d) `goal_velocity_pre_strike` 已经是 Gaussian 但 σ=0.6 仍然太紧 + ramp_time 0.1s 5 帧贡献小 | **v62 三件套修复**：(a) **公式 Laplacian → Gaussian**：`err = sum_squares(v_blade - v_hat); reward = exp(-err / σ²)`（[rewards.py:98-117](mdp/rewards.py#L98)）；(b) **σ 表 Gaussian-scale**：tier 0 σ=1.50（exp(-(2)²/2.25)=0.169 vs Laplacian 0.001 → **170× larger**），tier 6 σ=0.50（[curriculums.py:312-323](mdp/curriculums.py#L312)）；(c) **cross-curriculum cooldown 500 iter**：shape_tier 升级与 v_in_mag 升级**至少隔 500 iter**，防止 policy 同时面对"reward 更严+任务更难"双重打击（[curriculums.py:147-156](mdp/curriculums.py#L147)+[curriculums.py:692-722](mdp/curriculums.py#L692)+[curriculums.py:837-865](mdp/curriculums.py#L837)）| run 2026-05-30_23-43-23 iter 48 σ_vel=1.5000 正确生效；待跑到 iter 5000+ Phase 2 启动后验证 vel reward 是否真涨 | ⚪ |
| **R23** | **v61 长 plateau 后 PPO actor 数值发散 → `RuntimeError: normal expects all elements of std >= 0.0`（R7 复发）** | run 2026-05-30_18-59-49 **iter 35083 crash**。发散轨迹（TB 实证）：iter 35072-074 正常（`mean_reward=5.4`, `action_rate_l2=-0.06`）→ **iter 35075 突变**（reward -425, action_rate_l2 -12）→ iter 35078 `action_rate_l2=-2.8e10` → **iter 35083 `action_rate_l2=-1.21e26` / `action_l2=-6.1e25`** → `Mean value loss: nan` → 下一次 `actor.sample()` 时 std=NaN，`torch.normal` 抛错。崩时状态：shape_tier 卡 **1**（35k iter 没推进）、`goal_velocity=0.007`、`vel_fail=0.52`、`cos_sim_fh=0.06`/`cos_sim_bh=0.62`（正反手反复互换坏） | **双重**：(a) **长 plateau 是温床**——此 run 用**旧 Laplacian vel 公式**（`std_g_vel=0.35` 恒定，非 v62 Gaussian-scale 1.5），vel 通道被饿死（R22），shape_tier 卡 1、cos_sim 在 fh↔bh 间反复，policy 在平坦/冲突 landscape 上随机游走（entropy 28.3 高），某个 bad minibatch 把 actor mean 推进发散区；(b) **`action_rate_l2`/`action_l2` 是无界二次惩罚**——actions 一旦爆掉，这两项 reward 冲到 ~1e26（其他 reward 都被 exp() 限幅），value target 溢出 → value loss NaN → 反传 NaN 污染 `log_std` → std=NaN | **(a) 根因已被 v62 修掉**：Gaussian vel 公式（R22）让 vel 通道活、shape_tier 能推进，policy 不再长期 plateau（deviation run iter 4713 已到 shape_tier 4 / vel_success 0.76 / 无发散）→ 远离发散态。**v61 是被取代的旧设计，不必续训**。**(b) 无界惩罚是放大器**（未硬化）：若要兜底可加 action clip 或 NaN-guard（rsl_rl 已有 `max_grad_norm=1.0`，但拦不住单 rollout 的巨额 reward）。**注意：此 crash 与腿正则（deviation）改动无关** —— v61 进程启动于 05-30 18:59，模块快照早于腿改动 | deviation run（v62+腿正则）iter 4713 shape_tier=4 / vel_success 0.76 / action_rate_l2=-0.06 正常 —— 证明 v62 路线避开 v61 的 plateau-then-diverge；无界惩罚放大器为已知 latent 风险 | 🟡 |


## 八、3-Phase 任务课程设计（v61）

| # | 问题 | 触发现象 | 根因 | 解决方案 | 验证方式 | 状态 |
|---|---|---|---|---|---|---|
| **P1** | **v60 单一 phase 训练让 policy 选择简单的局部最优**（只学一种 swing pose） | run 2026-05-30_00-43-25 sim play 观察：双 swing 命令下 policy 只用 forehand pose，backhand 命令时拍面错向 | imit + goal_pos + goal_vel + goal_ori 同时存在的 reward landscape 里，policy 选最容易的 pose 满足大部分项；backhand demo 信号被 forehand-style stroke 的 reward 总和压过 | **3-phase 任务课程（单向阀门 monotone latch）**：<br>- **Phase 0 (stand)**: imit_w=0.10（弱 shaping），goal_*=0；EL_ema 涨到 350 后进 Phase 1<br>- **Phase 1 (imit)**: imit_w=1.0（重 imit），goal_*=0；EL_ema≥450 + Phase 1 跑 ≥ 2000 iter 后进 Phase 2<br>- **Phase 2 (strike)**: imit_w=0.30（paper-balanced），goal_* baseline 启动（[curriculums.py:142-282 update_task_phase](mdp/curriculums.py#L142))<br>- task_phase CurrTerm 必须放在 imit_anneal + pingpong **之后**（顺序敏感）| run 2026-05-30_18-59-49 iter 5008 进 Phase 2 后：iter 5500 hsr 0.28，iter 6000 hsr 0.38，iter 9000 hsr 0.52；`paddle_y_base_at_strike_backhand=+0.017`（demo +0.024 ✓） | 🟢 |
| **P2** | **swing_p_forehand_warmup 跟 task_phase 设计冲突** | v59-v60 期间 swing_warmup 90:10 → 50:50（EL_ema=250 触发）；v61 加 task_phase 后两个 latch 各自工作可能错位（swing 切 50:50 但 imit 仍在 Phase 0 弱权重） | swing_warmup 是 v59 时代的 single-task warmup 补丁；task_phase 已经做更彻底的 stand → imit → strike 分阶段。冗余且可能冲突 | **v62 完全删除 swing_warmup**：`swing_p_forehand` 固定 0.50（paper design），删除 `_SWING_RATIO_LATCH` + 对应 curriculum block + cfg 字段（`swing_p_forehand_warmup` / `swing_p_forehand_steady` / `swing_warmup_ep_length_threshold`） | run 2026-05-30_23-43-23 iter 48 `swing_ratio_forehand=0.499`（严格 50:50）| 🟢 |

## 九、Per-swing 诊断 metric 设计（验证 cheat 用）

为了精确检测 backhand cheat 是否在新设计下复发，加 6 个 split-by-swing-type metric。这些 metric 在 `_update_success_window` 中捕获 strike-instant 数据，在 `_refresh_metrics_from_counts` 中 split：

| Metric 名 | 计算方式 | 期望（正确学习）| Cheat signature |
|---|---|---|---|
| `Metrics/pingpong/hsr_forehand_only` | hsr per-env × fh_mask, sum / fh_count | ≈ hsr_backhand_only（差 ≤ 0.10）| 单边 hsr > 0.30 而另一边 < 0.10 |
| `Metrics/pingpong/hsr_backhand_only` | hsr per-env × bh_mask, sum / bh_count | 同上 | 同上 |
| `Metrics/pingpong/cos_sim_forehand_only` | signed cos_sim × fh_mask 取均值 | > 0.5（拍面对齐）| < 0（拍面反向）|
| `Metrics/pingpong/cos_sim_backhand_only` | signed cos_sim × bh_mask 取均值 | > 0.5 | < 0 |
| `Metrics/pingpong/paddle_y_base_at_strike_forehand` | strike 帧 paddle_pos 在 base frame 的 y，限 forehand cmd | ≈ `forehand_y_eff = -0.40` | 跟 backhand 一样 ≈ +0.024 或更接近 0（policy 抄反手位置）|
| `Metrics/pingpong/paddle_y_base_at_strike_backhand` | 同上限 backhand cmd | ≈ `backhand_y = +0.024` | 跟 forehand 一样 ≈ -0.40（policy 抄正手位置）|

实现要点：
- 在 `_update_success_window` 中 `in_window` 触发时捕获 `paddle_y_base` 和 `cos_sim_at_strike` 到 module state
- `_refresh_metrics_from_counts` 中按 `swing_type` mask 求均值，broadcast scalar 到全 env_count tensor（IsaacLab logger 按 env mean 取值时给回原 scalar）
- forehand_y_eff / backhand_y 是 `__init__` 阶段从 NPZ 自动算出来的（demo 数据决定，不可手设）

## 十、调试过程中的新误判（v59-v62 期间）

| # | 错误判断 | 反驳证据 | 修正 |
|---|---|---|---|
| W10 | "EL=51 plateau 是因为 goal_velocity 信号弱" | 加 diag metric 后发现 `delta_y_absmax = 127m`、`errMean=5457m²` —— `p_base_xy_world` 和 `root_pos_w` 不在同一 frame | 不是 reward 问题，是 frame mismatch (R17)。修 cap 解释（绝对世界 → env-local）后 errMean 从 5457 降到 0.04 |
| W11 | "v59 vel reward 弱是因为 σ_vel curriculum 收得太紧" | 删除 monotone latch（让 σ 双向跟随 EMA）后 hsr 从 0.45 仅涨到 0.56 又跌回 0.45 | σ 不是唯一原因。真正问题是 (a) Laplacian 公式中等误差区无梯度（R22）+ (b) policy 单 swing pose cheat（R18）+ (c) goal_* 在 Phase 2 卡 0 (R21)。三个一起修 |
| W12 | "task_phase 跳过 Phase 1 是 EL_ema 计算错" | EL_ema 数据正常，问题在两个 if 顺序执行让 100 iter 内连续触发 Phase 0→1→2 | 加 `phase_1_min_iters=2000` 强制 Phase 1 跑满（R20）|
| W13 | "swing_p_forehand=0.90 warmup 帮助单 task 学站立" | 实测 v60_no_swing_warmup vs v60_swing_warmup 早期 EL 几乎一致（28 vs 29 at iter 50）；3-phase 课程的 Phase 0 已经是 single-task focus（imit_w=0.10）| swing_warmup 跟 task_phase 重复，删除（P2）。Phase 0 给的 imit shaping 信号比 swing_warmup 90:10 更"对症"（结构上限制 reward landscape，不是采样比例上）|
| W14 | "保留 v_in_mag curriculum 跟 shape_tier 同时升级也没问题（反正都基于 EMAs）" | 实证：v60-v61 多次出现 shape_tier 升 σ_vel 收紧 + v_in_mag 增加球速同步发生 → policy 同时 face "reward 更严 + task 更难" → reward 突然崩 | 加 cross-curriculum cooldown 500 iter（R22 fix c），任意一个升级后另一个必须等 500 iter 才能升 |
| W15 | "Phase 2 启动后 goal_* 自然会被 window curriculum ramp 起来" | 实测 Phase 2 后 goal_* weight 永远 0（cos_sim_ratchet_freeze + Phase 0/1 已写 0 → window 用 max() 不更新）| 必须**显式 reset to baseline** at Phase 2 entry（R21 fix）。让 window 后续接管时基准是 baseline 不是 0 |

## 十一、v62 启动检查清单

新 from-scratch 训练前，确认以下都正确：

```bash
# 1. AST 检查
python -c "
import ast
ast.parse(open('source/unitree_rl_lab/unitree_rl_lab/tasks/pingpong/mdp/commands.py').read())
ast.parse(open('source/unitree_rl_lab/unitree_rl_lab/tasks/pingpong/mdp/curriculums.py').read())
ast.parse(open('source/unitree_rl_lab/unitree_rl_lab/tasks/pingpong/mdp/rewards.py').read())
ast.parse(open('source/unitree_rl_lab/unitree_rl_lab/tasks/pingpong/mdp/motion_loader.py').read())
ast.parse(open('source/unitree_rl_lab/unitree_rl_lab/tasks/pingpong/robots/g1_23dof/hitter/hitter_env_cfg.py').read())
print('OK AST')
"

# 2. 关键代码点 grep 验证
grep -n "divider_world = env_origins\[:, 1\] + y_mid" source/unitree_rl_lab/unitree_rl_lab/tasks/pingpong/mdp/commands.py
# 应该匹配 commands.py:443+

grep -n "torch.sum(torch.square(v_blade_b - v_hat_b)" source/unitree_rl_lab/unitree_rl_lab/tasks/pingpong/mdp/rewards.py
# 应该匹配 rewards.py vel reward Gaussian 公式

grep -n "_TASK_PHASE_LATCH\|update_task_phase" source/unitree_rl_lab/unitree_rl_lab/tasks/pingpong/mdp/curriculums.py
# 应该匹配 ~10+ 行

grep -n "task_phase = CurrTerm" source/unitree_rl_lab/unitree_rl_lab/tasks/pingpong/robots/g1_23dof/hitter/hitter_env_cfg.py
# 应该匹配 1 行（task_phase CurrTerm 注册）

grep -n "phase_1_min_iters" source/unitree_rl_lab/unitree_rl_lab/tasks/pingpong/robots/g1_23dof/hitter/hitter_env_cfg.py
# 应该匹配 1 行（值 2000）

grep -n "imitation_body_pos.*gate_pre_strike.*: False" source/unitree_rl_lab/unitree_rl_lab/tasks/pingpong/robots/g1_23dof/hitter/hitter_env_cfg.py
# 应该匹配 1 行（v62 改成 False）

# 3. 数学 sanity
python -c "
import math
# Gaussian σ=1.5 at ||Δv||=2 应该给 0.169
r = math.exp(-(2.0)**2 / 1.5**2)
assert abs(r - 0.1690) < 0.001, f'expect 0.169, got {r}'
print(f'Gaussian σ=1.5, ||Δv||=2: reward={r:.4f} ✓')

# Phase 2 baseline weights
assert 2.0 == 2.0  # goal_position
assert 1.0 == 1.0  # pre_strike
assert 0.5 == 0.5  # goal_orientation

# divider 锚定: env at world y=100, divider should be 100 + (-0.188) = 99.812
divider = 100 + (-0.188)
assert abs(divider - 99.812) < 0.001
print(f'divider_world for env at y=100: {divider} ✓')
print('ALL OK')
"
```

启动训练命令：
```bash
python scripts/rsl_rl/train.py --task Unitree-G1-23dof-Pingpong-HITTER --headless --run_name v62_full_design
```

预期前 5000 iter 进度（监控这些 metric）：

```
iter 0-100:    task_phase=0, imit_w=0.10, goal_*=0, σ_vel=1.50, EL 慢爬到 30+
iter 1000:     EL ~50-100 (Phase 0 持续)
iter 2500-3500: EL_ema 跨 350，phase 0→1，imit_w 跳到 1.0
iter 3500-5000: phase=1, imit reward 大涨（imit_jp 0.40+, imit_bp 0.50+）
iter ~5000:    phase 1→2 触发（满足 EL≥450 + p1elapsed≥2000）
iter 5000-6000: goal_* baseline 启动（w_goal_pos=2.0），window curriculum 后续 ramp 到 5.0
iter 5000+:    hsr 从 0 涨到 0.30+，per-swing diagnostic 应分化
iter 8000-15000: hsr 0.50+，cos_sim 0.50+，paddle_y_base 正反手分别接近各自 demo
```

警戒线：
- iter 8000 hsr 仍 < 0.30 → 检查 σ_vel curriculum / cooldown 是否过严
- iter 5000+ phase 仍 < 2 → 检查 EL_ema 是否真在涨 / phase_1_min_iters 是否应缩短
- `paddle_y_base_at_strike_backhand < -0.10`（应该 > 0）→ 反手 cheat 复发，回查 R18 fix 是否生效
- goal_velocity reward 仍 < 0.01 → Gaussian 公式可能没生效，回查 rewards.py

---

# v63 更新 (2026-06-01)

> 本轮围绕：下半身腿部正则、右臂全关节模仿、follow-through 撞桌的 motion-clip 根因、29dof 同步。

## 十二、下半身 / 击球后姿势类（v63）

| # | 问题 | 触发现象 | 根因 | 解决方案（代码位置）| 验证 | 状态 |
|---|---|---|---|---|---|---|
| **R24** | 击球后"抬右腿/单脚站/前后晃" | play model_5000/7000 观察待命期怪姿势 | **下半身完全无约束**：imit 只含上半身，腿没有默认姿态锚定、没有"双脚着地"奖励、悬空脚零成本（feet_slide 只罚着地滑动）。pelvis 高 0.74+不倒即可，腿随便摆 | 加 3 个**腿正则**（[rewards.py](mdp/rewards.py)：`feet_contact_no_strike`、`feet_distance_no_strike`；复用 `joint_deviation_l1`）：`leg_joint_deviation`(hip_roll/yaw 偏离默认,常开)、`feet_contact_no_strike`(待命 `t_to_hit≤0` 奖双脚着地)、`feet_distance_no_strike`(待命双向罚脚间距,叉开惩罚小×0.3)。权重纳入 `update_task_phase` 做 phase 课程(Phase2 弱)。`feet_contact`/`feet_distance` gate 到 `t_to_hit≤0` 以保留接球期侧移 | deviation run iter4713：base_y_drift 0.13(横移保住)、hard_contact 0.03、3 项量级 ±(0.05~0.3) 不压 goal | 🟢 |
| **R25** | **击球后拍子"抵桌" + 抽搐** ← 真根因在 motion clip | play 观察击球后拍子贴桌面 + 抖动 | **不是桌面接触惩罚问题**（训练 `paddle_table_contact` reward 实测=0，桌惩罚已满 -10，机器人训练中并没碰桌）。真因：**旧正手 clip `forward_003` 的 follow-through 把拍子带到台面以下**（NPZ 实测击球后 9 帧落入桌体积，z 最深 0.70 低于台面 6cm，clip 末尾锁在 z=0.62）。post-strike 模仿开着 → 策略复刻这段下扎轨迹 → 与桌子物理碰撞 → 抵桌+抽搐 | **换新 clip `new_new/forward_001` + `backward_001`**：follow-through 全程在台面之上(z 1.12~1.34)、0 帧进桌体积、收尾拍子抬高收回。([commands.py:847-848](mdp/commands.py)) | new_new NPZ 轨迹分析：正/反手 post-impact 均 0 帧进桌；反手旧 clip 本就干净（仅正手旧 clip 下扎）| ⚪ 待新 clip from-scratch 验证 |
| **R26** | 新 clip 击球点偏高，与 `hit_z_range` 不匹配 | new_new 正手 impact z=1.16、反手 z=1.26，而 `hit_z_range=(0.95,1.15)` → demo 高于命令上限 | 旧 clip 反而偏低(0.78<0.95)；新 clip 偏高,demo 与命令高度不重叠会让策略总在 demo 之下击球 | `hit_z_range` 上限 **1.15→1.25**（[commands.py:945](mdp/commands.py)）+ 课程 z 档上限统一到 1.25（[curriculums.py:848,851](mdp/curriculums.py)，原 1.18/1.22）| grep 确认四档 z 上限均 1.25 | ⚪ |

## 十三、右臂全关节模仿对照实验（v63）

| # | 问题 | 触发现象 | 根因 | 方案 | 结果 | 状态 |
|---|---|---|---|---|---|---|
| **R27** | 正手姿势别扭（"用反手姿势 + 肘外撇凑正手"）| play deviation/v63 观察 | 击球奖励只打**拍面/位置/速度**(strike 1 帧)，**不约束手臂姿势**；而右臂 distal(shoulder_yaw/elbow/wrist_roll)故意不模仿(M3 拍面自由)→ 同一击球点正反手只差 180° 翻面，自由腕子可"一套姿势翻腕通吃"→ 姿势丑 | **对照实验**：23dof 把右臂全 11 个上半身关节加回 `imitation_joint_names`(保持 `tracked_body_names`=8 不变,obs 维度不变可 resume)([hitter_env_cfg.py CommandsCfg override](robots/g1_23dof/hitter/hitter_env_cfg.py))| iter39565：cos_FH 0.48→**0.53**、cos_BH 0.23→**0.50**(平衡)、hsr 0.68→**0.81**、**无 M3 拍面锁(cos 反升)、无抢梯度(hsr 没掉)**。但**视觉正手姿势仍不够自然** → 残余原因在 motion clip(R25)+5DOF 臂可达性,非 reward bug | 🟡 |

## 十四、v63 期间的误判（走过的弯路）

| # | 错误判断 | 反驳证据 | 修正 |
|---|---|---|---|
| **W16** | "demo 正手扭腰约 18°"（我用 torso-pelvis 的 body-quat yaw 差算出 18°）| 直接读 `joint_pos` 的 `waist_yaw`(=joint[12])**实测 span 仅 4.6°、静态 ~20°** | **body-quat yaw 差被 pelvis 自身 roll/pitch 污染了**（动态动作下 naive yaw 提取不可靠）。**用户是对的:demo 不扭腰**，靠肩 pitch(joint[18]=48°)抡 + 迈步(joint[6]=69°)发力,腰锁死。`final.md §4.7` 原写的 "4.6°/锁住" **本就正确**(描述旧 clip)|
| **W17** | "29dof 远超 23dof(hsr0.86/tier6 vs 0.68/tier4)→ 23dof 是机械极限"| 23dof v63 训久后 iter39565 反超：hsr **0.81** > 29dof 同期 0.705；且 29dof **从峰值退步**(0.86/tier6 → 0.705/tier4)| 正手**不是 23dof 硬机械极限**；23dof 全关节模仿训久了能做好。29dof 退步是另一个待查的不稳定信号 |
| **W18** | "击球后抵桌 → 加大桌面接触惩罚" | 训练 `paddle_table_contact` reward 实测 = 0(已满 -10、零接触)| 加惩罚无用,真因是 follow-through 的 motion 把拍子往桌里带(R25)。改 clip 才对症 |

## 十五、29dof 同步结果（v63）

| # | 项 | 结果 | 状态 |
|---|---|---|---|
| **N1** | 29dof 从 v58/B7 一步同步到 v62+腿正则(3-phase 课程、Gaussian vel、leg regs、去 B7 ×4 imit boost、imit_w_phase 照搬 23dof) | run 2026-05-31_14-23-13：**首次站起来**(EL 222→500,hard_contact 0,bad_or 0.96→0.015,iter~1300 站稳)。历史上 29dof 从没站起来(run 11-13-14 卡 EL=33)| 🟢 站立 |
| **N2** | 29dof 击球进度 | iter8180 峰值 hsr 0.86/shape_tier 6 → iter30920 退到 hsr 0.705/tier4(cos_FH 0.43/BH 0.30)| 🟡 冲顶后退步,待查 |
| **N3** | `goal_velocity` std 同步 bug | 29dof 旧值 std=0.45 配 v62 Gaussian 公式 → vel reward≈0(exp(-4/0.45²)) | 同步改 std=1.5 | 🟢 |

## 十六、记录勘误

- **`final.md §4.7`**：描述的是**旧 clip**(`new/forward_003_rotated` / `new/backward_001_rotated`)的实测(impact frame 50/32、waist 4.6° 等)。v63 起**已换 `new_new/forward_001_rotated` + `backward_001_rotated`**(impact frame 20/56、击球 z 1.16/1.26、follow-through 高位不撞桌)。§4.7 的旧数值仅作历史参考。其中 waist "4.6°/锁住" **对旧 clip 是正确的**(对照 W16)。
- **`final.md §19` / 依赖清单**：当前权威 clip 路径为 `motion_datasets/.../expert/new_new/{forward,backward}/npz/{forward_001,backward_001}_rotated.npz`。
- **23dof 当前 `imitation_joint_names`**：v63 起在 [hitter_env_cfg.py CommandsCfg](robots/g1_23dof/hitter/hitter_env_cfg.py) override 为**全 11 个上半身关节**(含右臂 distal),`tracked_body_names` 保持 8 不变。
- **`target_land`**：(2.45, 0, 0.78) = 对方半台正中心(球台中心 1.77→远边 3.14 中点 2.455,差 0.5cm;y=0 中线;z=台面 0.76+球半径 0.02)。已确认无误。

---

## 十七、Sim2Real 部署诊断 (2026-06-09 → 2026-06-10)

**Sim2sim 稳, 真机 cmd 冻结模式仍腿/腕剧烈抖动**：sim 中机器人静止 actor `raw_a Δstd ~0.008`；真机同样 `enable_ball_input=false` (cmd 永远 = make_waiting_command 第一帧, t_to_hit 衰到 -0.5)，`raw_a Δstd ~0.29`，**两者差 36 倍**。诊断链如下表。所有 deploy-side 修复都在 [`deploy/robots/g1_23dof_pingpong/`](../../../deploy/robots/g1_23dof_pingpong/)，训练侧改动在 [hitter_env_cfg.py](robots/g1_23dof/hitter/hitter_env_cfg.py)。

### 17.1 ROS / Mocap 数据通路问题

| # | 问题 | 触发现象 | 根因 | 方案 | 验证 | 状态 |
|---|---|---|---|---|---|---|
| **D1** | VRPN topic 名错配 | C++ 订阅 `/pingpong/ball_state`, `/pingpong/base_pose`，但 VRPN-mocap 在发 `/vrpn_mocap/U_Tracker0/pose`, `/vrpn_mocap/g1/pose`，0 帧进 callback | sim 端旧 topic 名遗留；真机 VRPN 用自己默认前缀 | config.yaml `ros.ball_state_topic / base_pose_topic` 改成 `/vrpn_mocap/...`；message type 由 `Odometry` 改 `PoseStamped` (VRPN 不发 twist)；同步把 sim 端 [unitree_pingpong_mujoco.py](../../../../../unitree_mujoco/simulate_python_pingpong/unitree_pingpong_mujoco.py) 的 publisher 类型也改 PoseStamped + sensor_data QoS | 终端 1Hz INFO 出现 `[ball ...] ~300 msg/s`、`[base ...] ~300 msg/s` | 🟢 |
| **D2** | C++ DDS QoS `keep_last(1)` 在 VRPN burst-send 下丢 2/3 包 | `ros2 topic hz` 实测 ~292Hz, 但 C++ ros_base_trace.csv 录到 ~96Hz, 比例正好 1/3 | VRPN 把多帧打包 burst (median inter-arrival 0.16ms, max 50ms idle), `KEEP_LAST(1)` buffer 太小, 来一帧 burst 第二第三帧到来时第一帧已被覆盖丢弃 | `auto qos = SensorDataQoS().keep_last(50)` (deploy `start_ros_if_enabled`)；执行器有足够 buffer 漏空一次 burst | C++ csv 频率 96Hz → **300Hz**, 跟 monitor 吻合 | 🟢 |
| **D3** | DDS discovery delay → 进 Pingpong 头 ~2.88 秒 base 一帧没收 | 第一次进 Pingpong 时 `Pingpong ROS2 Humble subscribers started` 才打印 (= subscribers 那时才创建), 头几秒 ext_.has_base=false → fallback path 用 `reset_root_pos = (-0.138, 0, 0.74)` 算 cmd, **此时 mocap 实际 base 跟 reset 差几十厘米 + yaw 几十度**, 严重 OOD | CtrlFSM lazy init: state ctor 只在第一次切到该 state 时调；start_ros_if_enabled 在 ctor 里, 跟 enter() 同时刻才真正订阅 | 加 `CtrlFSM::preinstantiate_state(int)` (idempotent)；main.cpp 在 `fsm->start()` 之前调 `preinstantiate_state(10)` 让 Pingpong state 启动时就实例化, ROS subscribers 早 2-3 秒做完 DDS discovery | 启动 log 里 `Pingpong ROS2 Humble subscribers started` 在 main 里就打印；进 Pingpong 后 csv 第一行 controller_t = 0.000s 就有数据 | 🟢 |
| **D4** | csv 路径相对路径双拼 | C++ 默认 `deploy/robots/g1_23dof_pingpong/logs/...`, `resolve_project_path` 又 prepend `proj_dir = deploy/robots/g1_23dof_pingpong/` → 写到 `deploy/robots/g1_23dof_pingpong/deploy/robots/g1_23dof_pingpong/logs/...` 双拼 | `proj_dir` 是包根不是仓库根；默认值假设错 | C++ 默认改 `logs/ros_*_trace.csv` 包内相对；config.yaml 里给绝对路径覆盖 | 跑后 `logs/` 一级目录有 csv | 🟢 |

### 17.2 Cmd 生成几何 + 滤波

| # | 问题 | 触发现象 | 根因 | 方案 | 验证 | 状态 |
|---|---|---|---|---|---|---|
| **D5** | 第一帧 cmd 永远基于 reset_root_pos, 即使 mocap 后到也不重算 | `enable_ball_input=false` 模式下：第一个 50Hz tick (t=0.020s) 因 `has_ball=false` 走 fallback → `make_waiting_command(state, ...)`; 但此刻 ext_.has_base 也 false → state.base = reset_root_pos. 之后 mocap 来了 has_base=true, 但 update_command 看到 `has_previous_cmd=true` → 直接复用上一帧 cmd 几何, 不重算 | hold_previous_or_seed_initial_command 设计：fallback path 内, has_previous_cmd → 复用 cmd_, 不重新调 make_waiting_command | D3 把 subscribers 提前到程序启动 → 进 Pingpong 时 ext_.has_base 已 true → 第一帧 make_waiting_command 直接用真实 mocap base | csv `world_*` mean 跟 mocap 实测对齐 | 🟢 |
| **D6** | base mocap z origin 标定偏 0.08m | csv `world_z mean = 0.661` (训练 reset z = 0.74, 偏 -0.08) | config.yaml `input_frame.origin_in_training_world = (1.77, 0, 0.735)` 是从训练 sim table center 直接复制, 没考虑真机 mocap 原点 z 偏移 | 改成 `(1.77, 0, 0.815)` (= 0.74 - raw_z_mean(-0.064)). 校准方法写进 yaml 注释 | 改后 csv `world_z mean = 0.758` (target ~0.74, 仍偏 +0.018, 接受) | 🟢 |
| **D7** | base mocap 滤波: push-side mean → pull-side mean (对应 D8 quat 半球均值) | mocap 110-300Hz, 控制 50Hz, 不滤波则 actor 看每帧 jitter | (a) 高频 mocap 信号需要平滑；(b) push-side (callback 里算) 浪费算力, ROS 频率 ≫ 控制频率 | 加 `compute_base_mean_world_locked()` helper：callback 只 push deque + 写时间戳；control loop pull 时算 sample-count 滑窗均值 (default window=5)。Quaternion mean 用 hemisphere-aligned 简化 Markley (小角度差等价于 SVD 解, 110Hz×5=45ms 内不可能转过几度). yaml: `ros.base_filter_window: 5` | csv 加 `filt_*` 列, raw vs filt 对比 yaw Δstd 0.378° → 0.100° (4× 缩减), z Δstd 0.0008m → 0.0005m | 🟢 |
| **D8** | quat 简单算术平均 → 数值崩溃 | quaternion 两半球 (q 和 -q 同一旋转, double cover), 直接平均会 cancel | hemisphere-align: 所有 q_i 跟 q_ref dot < 0 时翻号, 再平均, 再 normalize. 等价 SVD 主特征向量解在小角度差时 | [State_Pingpong.cpp `quat_mean_hemisphere_aligned()`](../../../../../deploy/robots/g1_23dof_pingpong/src/State_Pingpong.cpp) | 算法层面验证 (单元测试两个 (id, -id, id) 输入返回 (0,0,0,1)) | 🟢 |

### 17.3 Actor 抖动 — sensor 高频噪声 OOD (核心问题)

| # | 问题 | 触发现象 | 根因 | 方案 | 验证 | 状态 |
|---|---|---|---|---|---|---|
| **D9** | obs_trace.csv 暴露真机 sensor 高频噪声 | cmd 冻结+机器人静止站立, q_act std=0.001 (机器人真不动), 但：<br>obs_46..68 (joint_vel = motor.dq) Δstd **mean 2.66, max 10 rad/s**(R_wrist)<br>obs_0..2 (base_ang_vel = IMU gyro) Δstd **mean 0.62, max 0.82 rad/s** (range ±3.6)<br>训练 sim 静止时这两组 Δstd ≈ 0 | (a) **motor encoder 内部速度估计器 artifact**：Unitree G1 motor controller 输出的 dq 是 short-window finite-diff + estimator filter, 静止时 std 0.5-2 rad/s, 腕关节小电机更糟 ±10 rad/s<br>(b) **MEMS IMU gyro 静态噪声**：典型 ±0.05 rad/s 高频 + 5-10° tilt 后投影偏差<br>(c) **训练分布外 100×**, actor 看到的 obs[joint_vel] / obs[base_ang_vel] 完全 OOD → 输出乱 → 反馈到下一帧 last_action obs → 自激震荡 | 加 `obs_trace.csv` (92 维 actor 输入 wide-format)、`motor_trace.csv` (raw_a + q_des + q_act + dq_act) trace 工具；离线 pandas 算 per-dim Δstd 直接定位 OOD 维度 | obs[46..68] 真机 std 2.66, sim 0；obs[0..2] 真机 std 0.62, sim 0；定位完成 | 🟢 |
| **D10** | 训练 EventCfg 没 sensor 高频噪声 randomization | `randomize_imu_offset` 是 startup 模式 episode-level **静态 ±2°** offset (= IMU 校准误差模拟), 没有 per-step 高频噪声; `randomize_comm_delay` 也是 startup 0-1 step (= 0-20ms) 静态延迟。**没有 joint_pos / joint_vel / IMU gyro / IMU accel 的 per-step Unoise 注入** | actor 训练时 obs 完美 (mujoco.qvel ≈ 0, IMU gyro 静止 ≈ 0), 部署时被高频噪声打脸 | (a) PolicyCfg 4 个 sensor obs 加 `noise=Unoise(...)` (mimic / locomotion 标准值): `base_ang_vel ±0.2`, `projected_gravity ±0.05`, `joint_pos_rel ±0.01`, `joint_vel_rel ±0.5`<br>(b) `enable_corruption = True` (打开 IsaacLab corruption 管道, 否则 noise= 不生效)<br>(c) Critic 保持 `enable_corruption = False` + 各 obs term 不写 noise → 看清洁 ground truth (privileged value estimation)<br>([hitter_env_cfg.py PolicyCfg](robots/g1_23dof/hitter/hitter_env_cfg.py)) | 待重训验证；预期 actor 训练时见过 ±0.5 rad/s joint_vel 噪声 → deploy 时对真机 motor.dq 噪声鲁棒 | ⚪ |
| **D11** | Deploy 端短期 fix (训练完成前): `joint_vel` obs 改用 q 的 finite-diff(N) | motor.dq 噪声 1-10 rad/s, 但 q 编码器 std ~0.001 rad. `dq = (q[k]-q[k-1])/dt` → ~0.05 rad/s, 比 motor.dq 干净 100×。LSQ slope on N samples 噪声进一步 ÷ √(N³/12). N=5 latency 50ms (训练 delay 上限 20ms, 偏多但比 motor.dq 好得多) | motor controller 内部估计器对静止状态的速度估计有 quantization + estimator artifact, 但 q 自己 quantization 小 | (a) build_obs_term("joint_vel") 加 `joint_vel_obs_source: motor_dq | finite_diff` 切换 yaml; (b) finite_diff 路径用 LSQ slope 公式 `dq = Σᵢ(i-mean_i)·q[i] / ((n³-n)/12·dt)`, 全 N 个点参与, 抗 outlier; (c) motor_dq 路径加 sliding mean filter window=10 (用户当前选用) | 1-step finite-diff: obs joint_vel Δstd 2.66 → 0.69 (3.9×↓), raw_a Δstd 0.29 → 0.107 (2.7×↓). LSQ N=5 进一步降; motor_dq+window=10 latency 100ms 但用户接受 | 🟡 (短期 fix；D10 重训完后可关) |

### 17.4 走过的弯路

| # | 错误判断 | 反驳证据 | 修正 |
|---|---|---|---|
| **W19** | "base_yaw 训练用 IMU(`base_yaw_encoding_imu`), 部署用 mocap → yaw 来源不一致是 actor 抖根因" | sim2sim (sim 端 base_yaw 也走 mocap 等价) actor 完全稳定；用户主动澄清"IMU 没磁力计, 不能给可靠 yaw, 故意改用 mocap" | mocap base_yaw 是 deliberate 工程决定。yaw 数值 sim/real 接近 (12° vs 16°), 不是抖动根因。撤回嫌疑 |
| **W20** | "Pingpong 0.96s 退到 Passive 是 actor 失控自动 fallback" | 用户解释：是手动按 Y 关停的, "看到策略失控了手动关掉" | Pingpong → Passive 唯一路径是 Y.on_pressed; 是 user 干预, 不是策略 bug。要看实际策略行为需要让它跑久 |
| **W21** | "球频率从监测 292Hz 掉到 csv 95Hz 是 mocap 软件 occlude" | 跟 base 对照：base 在两次跑里频率从 95Hz → 300Hz, 唯一改动是 `keep_last(1) → keep_last(50)`. mocap 端没动 | 频率丢失全部出在 C++ 接收端 DDS QoS, 不是 mocap (D2) |
| **W22** | "fusion 的 obs 实现就是 pingpong 的参考, 我们应该完全跟它一致" | 用户：fusion 是 mimic task 的 obs, pingpong 自己有专门的 obs term (active_face / hit_pos / target_normal 等), 不能直接照搬 | 对照标准应该是 pingpong 训练 [hitter_env_cfg.py PolicyCfg](robots/g1_23dof/hitter/hitter_env_cfg.py), 不是 fusion. 改去看 obs term name → DelayedObservation wrapper → inner_func, 找到训练时 actor 真正看到的 obs |

### 17.5 Deploy 端新增诊断工具 (留档)

部署阶段加的所有 csv trace + 监测都在 [deploy/robots/g1_23dof_pingpong/](../../../../../deploy/robots/g1_23dof_pingpong/), config.yaml `Pingpong.logging` 块控制开关：

| 工具 | 文件 | 作用 |
|---|---|---|
| `ros_ball_trace.csv` / `ros_base_trace.csv` | logs/ | 每个 ROS callback 写一行 (raw + transform + filt), 跟 `ros2 topic echo` 对比 frame / unit / sign / timestamp |
| `motor_trace.csv` | logs/ | 50Hz 一行: actor raw_a + final motor.q (含 blend) + measured q_act + dq_act, 23 关节 |
| `obs_trace.csv` | logs/ | 50Hz 一行: 全 92 维 actor 输入 (post scale/clip), sim/real 对比定位 OOD 维度 |
| 1Hz `[ball ...]` / `[base ...]` INFO | terminal | callback 频率 + 单帧 raw + transform 瞬时值, 诊断 mocap 是否在线 + transform 是否对 |
| `inspect_pose_live.py --topic` | python | rclpy + matplotlib 画 raw vs filtered (虚线) 对比, 实时看滤波效果 + mocap 抖动 |

### 17.6 Sim 端同步改动 (sim2sim → C++ 接口完全对齐 mocap)

[unitree_mujoco/simulate_python_pingpong/](../../../../../unitree_mujoco/simulate_python_pingpong/) 改三处使 sim 跟真机 VRPN 完全对齐, C++ 控制器无法分辨 sim publisher 还是真 mocap:
- `config.py`: `INPUT_FRAME_ID = "world"`, `BALL_STATE_TOPIC = /vrpn_mocap/U_Tracker0/pose`, `BASE_POSE_TOPIC = /vrpn_mocap/g1/pose`, `INPUT_ORIGIN_IN_TRAINING_WORLD = (1.77, 0, 0.815)` (跟 deploy 校准一致)
- `unitree_pingpong_mujoco.py Ros2Publisher`: ball publisher 类型 `Odometry` → `PoseStamped` (VRPN 不发 twist；C++ BallTrajFilter 从 PoseStamped 序列重建 v); QoS `default(RELIABLE, depth=10)` → `qos_profile_sensor_data` (BEST_EFFORT + KEEP_LAST(5) + VOLATILE)

跑 sim 前先 `mv logs logs_sim_<ts>` 备份, 防止 sim/real 同样 csv 路径互相覆盖。

### 17.7 关键判读约定

- **Actor 抖动来源判读** (用 motor_trace + obs_trace):
  - `q_des Δstd 大 + q_act Δstd 小`：策略输出抖, 电机 PD 滤掉了 → actor / obs 问题, 不是电机
  - `q_des Δstd 小 + q_act Δstd 大`：策略平滑但电机跟不上 → 电机 / 通信延迟问题
  - `obs_X Δstd ≫ obs_X_sim Δstd`：第 X 维 OOD, 元凶维度
- **frequency 损失判读**：先 `ros2 topic hz --qos-profile sensor_data <topic>` 确认 publisher 真实频率, 再看 csv 实际频率, 差距来自 C++ 订阅端 DDS QoS / executor latency, 不是 mocap
- **mocap origin 校准**：让机器人 FixStand 静止录 1s base csv, `origin_z = 0.74 - mean(raw_z)`. 公式同样套到 origin_x/y (但通常 x=1.77, y=0 跟桌面中心一致, 不需校)
