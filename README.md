# Unitree RL Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)
[![Discord](https://img.shields.io/badge/-Discord-5865F2?style=flat&logo=Discord&logoColor=white)](https://discord.gg/ZwcVwxv5rq)


## Overview

This project provides a set of reinforcement learning environments for Unitree robots, built on top of [IsaacLab](https://github.com/isaac-sim/IsaacLab).

Currently supports Unitree **Go2**, **H1** and **G1-29dof** robots.

<div align="center">

| <div align="center"> Isaac Lab </div> | <div align="center">  Mujoco </div> |  <div align="center"> Physical </div> |
|--- | --- | --- |
| [<img src="https://oss-global-cdn.unitree.com/static/d879adac250648c587d3681e90658b49_480x397.gif" width="240px">](g1_sim.gif) | [<img src="https://oss-global-cdn.unitree.com/static/3c88e045ab124c3ab9c761a99cb5e71f_480x397.gif" width="240px">](g1_mujoco.gif) | [<img src="https://oss-global-cdn.unitree.com/static/6c17c6cf52ec4e26bbfab1fbf591adb2_480x270.gif" width="240px">](g1_real.gif) |

</div>

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
- Install the Unitree RL IsaacLab standalone environments.

  - Clone or copy this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

    ```bash
    git clone https://github.com/unitreerobotics/unitree_rl_lab.git
    ```
  - Use a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    conda activate env_isaaclab
    ./unitree_rl_lab.sh -i
    # restart your shell to activate the environment changes.
    ```
- Download unitree robot description files

  *Method 1: Using USD Files*
  - Download unitree usd files from [unitree_model](https://huggingface.co/datasets/unitreerobotics/unitree_model/tree/main), keeping folder structure
    ```bash
    git clone https://huggingface.co/datasets/unitreerobotics/unitree_model
    ```
  - Config `UNITREE_MODEL_DIR` in `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`.

    ```bash
    UNITREE_MODEL_DIR = "</home/user/projects/unitree_usd>"
    ```

  *Method 2: Using URDF Files [Recommended]* Only for Isaacsim >= 5.0
  -  Download unitree robot urdf files from [unitree_ros](https://github.com/unitreerobotics/unitree_ros)
      ```
      git clone https://github.com/unitreerobotics/unitree_ros.git
      ```
  - Config `UNITREE_ROS_DIR` in `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`.
    ```bash
    UNITREE_ROS_DIR = "</home/user/projects/unitree_ros/unitree_ros>"
    ```
  - [Optional]: change *robot_cfg.spawn* if you want to use urdf files



- Verify that the environments are correctly installed by:

  - Listing the available tasks:

    ```bash
    ./unitree_rl_lab.sh -l # This is a faster version than isaaclab
    ```
  - Running a task:

    ```bash
    ./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity # support for autocomplete task-name
    # same as
    python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity
    ```
  - Inference with a trained agent:

    ```bash
    ./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity # support for autocomplete task-name
    # same as
    python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity
    ```

## Deploy

After the model training is completed, we need to perform sim2sim on the trained strategy in Mujoco to test the performance of the model.
Then deploy sim2real.

### Setup

```bash
# Install dependencies
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
# Install unitree_sdk2
git clone git@github.com:unitreerobotics/unitree_sdk2.git
cd unitree_sdk2
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF # Install on the /usr/local directory
sudo make install
# Compile the robot_controller
cd unitree_rl_lab/deploy/robots/g1_29dof # or other robots
mkdir build && cd build
cmake .. && make
```

### Sim2Sim

Installing the [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco?tab=readme-ov-file#installation).

- Set the `robot` at `/simulate/config.yaml` to g1
- Set `domain_id` to 0
- Set `enable_elastic_hand` to 1
- Set `use_joystck` to 1.

```bash
# start simulation
cd unitree_mujoco/simulate/build
./unitree_mujoco
# ./unitree_mujoco -i 0 -n eth0 -r g1 -s scene_29dof.xml # alternative
```

```bash
cd unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl
# 1. press [L2 + Up] to set the robot to stand up
# 2. Click the mujoco window, and then press 8 to make the robot feet touch the ground.
# 3. Press [R1 + X] to run the policy.
# 4. Click the mujoco window, and then press 9 to disable the elastic band.
```

### Sim2Real

You can use this program to control the robot directly, but make sure the on-borad control program has been closed.

```bash
./g1_ctrl --network eth0 # eth0 is the network interface name.
```

## Personal Extensions

This fork extends `unitreerobotics/unitree_rl_lab` with two research lines targeting the Unitree **G1 23-DoF** platform: an **AMP-style training stack** and a reproduction of the **HITTER** table-tennis paper. Upstream tasks remain untouched.

### 1. Adversarial Motion Priors (AMP) for G1

Custom AMP-PPO training stack on top of `rsl_rl`, with a reference-motion discriminator that injects style reward into the PPO update. The full algorithm/storage/runner triplet is forked off `rsl_rl` so changes don't leak into the upstream-shared training scripts.

| Component | Path |
| --- | --- |
| AMP-PPO + curriculum | `source/unitree_rl_lab/unitree_rl_lab/rsl_rl_amp/algorithms/{amp_ppo,amp_curriculum}.py` |
| Motion buffer + AMP rollout storage | `source/unitree_rl_lab/unitree_rl_lab/rsl_rl_amp/storage/{motion_dataset,amp_rollout_storage}.py` |
| On-policy AMP runner | `source/unitree_rl_lab/unitree_rl_lab/rsl_rl_amp/runners/on_policy_amp_runner.py` |
| Env / MDP | `source/unitree_rl_lab/unitree_rl_lab/tasks/amp/` |
| Motion data pipeline | `source/unitree_rl_lab/unitree_rl_lab/utils/amp_data_tools/` |
| Pre-processing scripts | `scripts/AMP/` |

Registered tasks (all `Isaac-Lab` style env IDs):

- `Unitree-G1-23dof-Velocity-AMP` — locomotion velocity tracking + AMP style reward
- `Unitree-G1-23dof-Penguin-AMP` / `-V2` / `-V3` — penguin-gait imitation, three iterated reward / curriculum revisions

Motion data tooling (LAFAN / Ember motion bank → 23-DoF G1):

- `download_and_merge_ember_lafan_g1.py` — fetch and merge motion banks into a unified `.npz`
- `recompute_fk_ember_23dof.py`, `recompute_fk_npz_flexible.py` — re-derive end-effector and joint trajectories via forward kinematics
- `replay_23dof_npz.py` — playback in MuJoCo for sanity check
- `inspect_ember_npz.py` — schema inspection

Train an AMP policy:

```bash
./unitree_rl_lab.sh -t --task Unitree-G1-23dof-Penguin-AMP-V3
```

### 2. HITTER: Humanoid Table Tennis Reproduction

Reproduction of *Su et al. 2025, "HITTER: A HumanoId Table TEnnis Robot via Hierarchical Planning and Learning"* on the Unitree G1 23-DoF. The hierarchical structure separates a **high-level striking-pose planner** (ball trajectory → target pose / contact timing) from a **low-level RL whole-body controller**.

Training side — `source/unitree_rl_lab/unitree_rl_lab/tasks/pingpong/`:

| Module | Role |
| --- | --- |
| `mdp/planner.py` | Runtime planner: ball-state → striking pose target |
| `mdp/planner_for_training.py` | Vectorized planner variant for parallel sim |
| `mdp/motion_loader.py` | Reference striking motion loader |
| `mdp/commands.py` / `mdp/real_commands.py` | Sim and real-world command spaces |
| `mdp/{rewards,terminations,curriculums,events,observations}.py` | HITTER-specific MDP wiring |
| `robots/g1_23dof/hitter/` | Sim training env config |
| `robots/g1_23dof/hitter_real/` | Real-world deployment env config |

Registered tasks:

- `Unitree-G1-23dof-Pingpong-HITTER` — sim training
- `Unitree-G1-23dof-Pingpong-HITTER-REAL` — real-world / sim2real inference config

Train:

```bash
./unitree_rl_lab.sh -t --task Unitree-G1-23dof-Pingpong-HITTER
```

Deployment side — `deploy/robots/g1_23dof_pingpong/` (C++ FSM + ONNX inference):

- `src/State_Pingpong.cpp` — FSM state that runs the trained ONNX policy plus the runtime planner on the robot
- `src/BallTrajFilter.cpp` (+ `include/BallTrajFilter.h`) — ball trajectory state estimator (filter + future-pose prediction)
- `inspect_bag.py`, `inspect_pose_live.py`, `plot_hit_trace.py` — ROS bag / live pose analysis tools
- `data/{velocity, dance102_mimic, gangnanstyle_mimic, pingpong}/` — exported `policy.onnx` and `params/env.yaml` per skill
- `bags/` — recorded sim + real interaction traces
- A separate Mujoco sim2sim variant: `deploy/robots/g1_23dof_pingpong_mujoco/`

Build & run on the robot (or in MuJoCo with `unitree_mujoco`):

```bash
cd deploy/robots/g1_23dof_pingpong
mkdir -p build && cd build
cmake .. && make -j
./g1_pingpong_ctrl                  # sim2sim (with unitree_mujoco running)
./g1_pingpong_ctrl --network eth0   # sim2real on the robot
```

Design / debugging notes live alongside the task code:

- `final.md`, `final_1.md` — design iterations
- `TROUBLESHOOTING.md` — failure-mode log
- `TUNING_GUIDE.md` — hyperparameter tuning notes

> Reference: Su, et al. *HITTER: A HumanoId Table TEnnis Robot via Hierarchical Planning and Learning*, 2025.

## Acknowledgements

This repository is built upon the support and contributions of the following open-source projects. Special thanks to:

- [IsaacLab](https://github.com/isaac-sim/IsaacLab): The foundation for training and running codes.
- [mujoco](https://github.com/google-deepmind/mujoco.git): Providing powerful simulation functionalities.
- [robot_lab](https://github.com/fan-ziqi/robot_lab): Referenced for project structure and parts of the implementation.
- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking): Versatile humanoid control framework for motion tracking.
