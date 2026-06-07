# G1 23DoF Pingpong MuJoCo Deploy

This folder is a Python sim2sim/sim2real runner for the pingpong HITTER policy.

It uses:

- `assets/g1_23dof_pingpong_scene.xml`: G1 23DoF paddle robot, thin table, net, and a free pingpong ball.
- `config/config.yaml`: policy list, deploy.yaml/checkpoint paths, planner/world/serve parameters.
- `run_pingpong_mujoco.py`: MuJoCo backend, Unitree SDK backend, HITTER planner bridge, IsaacLab-style observation assembly, and policy switching.

## Sim2Sim

```bash
conda run -n unitree-mujoco python deploy/robots/g1_23dof_pingpong_mujoco/run_pingpong_mujoco.py
```

Useful options:

```bash
conda run -n unitree-mujoco python deploy/robots/g1_23dof_pingpong_mujoco/run_pingpong_mujoco.py --check --no-render
conda run -n unitree-mujoco python deploy/robots/g1_23dof_pingpong_mujoco/run_pingpong_mujoco.py --no-render --duration 5
```

Expert geometry is not hard-coded. Pass a different forehand/backhand npz pair and the runner derives `x_hit_default`, `y_mid_base`, `swing_type`, and `p_base_xy_world` from the npz impact frames:

```bash
conda run -n unitree-mujoco python deploy/robots/g1_23dof_pingpong_mujoco/run_pingpong_mujoco.py \
  --forward-npz motion_datasets/pingpong/humanoid_data/final/expert/new_3/forward/npz/forward_001_wristfix_rotated.npz \
  --backward-npz motion_datasets/pingpong/humanoid_data/final/expert/new_3/backward/npz/backward_001_rotated.npz
```

Viewer keys:

- `Space`: resample serve
- `R`: reset robot and ball
- `N` / `P`: next / previous policy
- `H`: HITTER policy
- `F`: fixed stand policy

## Sim2Real

```bash
conda run -n unitree-mujoco python deploy/robots/g1_23dof_pingpong_mujoco/run_pingpong_mujoco.py --network enp129s0 --no-render
```

The Unitree backend reads low state and publishes low command through `unitree_sdk2py`.
The current first version uses the same synthetic serve/ball state source for the planner on real hardware; replace `VirtualBall` with a camera ball source when the vision pipeline is ready.

## Observation Contract

The HITTER policy obs is assembled in the same order as `params/deploy.yaml`:

`base_ang_vel, projected_gravity, base_yaw, base_err, hit_pos, racket_vel, t_to_hit, active_face, target_normal, joint_pos, joint_vel, last_action`

For `model_4000.pt` this is `92` dims and `23` actions. Joint order is derived from `joint_ids_map`, then mapped to MuJoCo actuators or Unitree motor IDs.
