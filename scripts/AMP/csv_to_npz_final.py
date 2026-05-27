"""
Usage:
    python csv_to_npz_final.py -f /home/woan/下载/g1_qie_motion.csv --input_fps 30 --output_fps 30
from
    fps=30
    root_pos (111,3) 机器人 Pelvis（盆骨）在三维空间中的 [x,y,z] 位置。从数据看 z≈0.78m，这是 G1 的标准站立高度
    root_rot (111,4) 机器人 Pelvis 的旋转四元数 [x,y,z,w]，描述了机器人在空间中的朝向
    dof_pos (111,29)

form:
    Key: fps                | Shape: (1,)
    Key: joint_pos          | Shape: (110, 23)
    Key: joint_vel          | Shape: (110, 23)
    Key: body_pos_w         | Shape: (110, 24, 3)
    Key: body_quat_w        | Shape: (110, 24, 4) in wxyz
    Key: body_lin_vel_w     | Shape: (110, 24, 3)
    Key: body_ang_vel_w     | Shape: (110, 24, 3)
    Key: joint_names        | Shape: (23,)   dtype='<U...'  articulation joint order
    Key: body_names         | Shape: (24,)   dtype='<U...'  articulation body order
"""

import argparse
import numpy as np
import torch
from isaaclab.app import AppLauncher

# 1. 参数解析
parser = argparse.ArgumentParser(description="将 G1 29DOF CSV 转换为 23DOF NPZ")
parser.add_argument(
    "-f",
    "--input_file",
    type=str,
    default="/home/woan/HumanoidProject/unitree_rl_lab/motion_datasets/penguin/g1_qie_motion.csv",
    help="输入 CSV 路径",
)
parser.add_argument("--input_fps", type=int, default=30, help="原始数据 FPS")
parser.add_argument("--output_fps", type=int, default=30, help="输出 NPZ 的 FPS")
parser.add_argument("--output_name", type=str, help="输出文件名")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 2. 启动 Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_mul,
    quat_slerp,
)

# 导入机器人配置
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_23DOF_CFG as ROBOT_CFG


@configclass
class ReplaySceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class MotionLoader:
    def __init__(self, motion_file, input_fps, output_fps, device):
        self.motion_file = motion_file
        self.input_dt = 1.0 / input_fps
        self.output_dt = 1.0 / output_fps
        self.device = device
        self.current_idx = 0
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        # 加载数据 [N, 36] (3 pos + 4 rot + 29 joints)
        motion = (
            torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
            .to(torch.float32)
            .to(self.device)
        )
        self.motion_base_poss_input = motion[:, :3]
        # CSV 是 xyzw -> Isaac 需要 wxyz [3, 0, 1, 2]
        self.motion_base_rots_input = motion[:, 3:7][:, [3, 0, 1, 2]]

        # 核心：29 映射到 23
        keep_dof_indices = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            15,
            16,
            17,
            18,
            19,
            22,
            23,
            24,
            25,
            26,
        ]
        motion_dof_29 = motion[:, 7:]
        self.motion_dof_poss_input = motion_dof_29[:, keep_dof_indices]

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt

    def _interpolate_motion(self):
        # 时间轴插值
        times = torch.arange(0, self.duration, self.output_dt, device=self.device)
        self.output_frames = times.shape[0]

        phase = times / self.duration
        idx0 = (phase * (self.input_frames - 1)).floor().long()
        idx1 = torch.minimum(idx0 + 1, torch.tensor(self.input_frames - 1))
        blend = (phase * (self.input_frames - 1)) - idx0

        self.motion_base_poss = self.motion_base_poss_input[idx0] * (
            1 - blend.unsqueeze(1)
        ) + self.motion_base_poss_input[idx1] * blend.unsqueeze(1)
        self.motion_dof_poss = self.motion_dof_poss_input[idx0] * (
            1 - blend.unsqueeze(1)
        ) + self.motion_dof_poss_input[idx1] * blend.unsqueeze(1)

        # 四元数 SLERP 插值
        slerped_rots = torch.zeros((self.output_frames, 4), device=self.device)
        for i in range(self.output_frames):
            slerped_rots[i] = quat_slerp(
                self.motion_base_rots_input[idx0[i]],
                self.motion_base_rots_input[idx1[i]],
                blend[i],
            )
        self.motion_base_rots = slerped_rots

    def _compute_velocities(self):
        # 差分得到速度
        self.motion_base_lin_vels = torch.gradient(
            self.motion_base_poss, spacing=self.output_dt, dim=0
        )[0]
        self.motion_dof_vels = torch.gradient(
            self.motion_dof_poss, spacing=self.output_dt, dim=0
        )[0]

        # 旋转差分得到角速度
        q_prev, q_next = self.motion_base_rots[:-2], self.motion_base_rots[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * self.output_dt)
        self.motion_base_ang_vels = torch.cat([omega[:1], omega, omega[-1:]], dim=0)

    def get_next_state(self):
        flag = False
        if self.current_idx >= self.output_frames:
            flag = True
            self.current_idx = 0  # 循环回放
        state = (
            self.motion_base_poss[self.current_idx],
            self.motion_base_rots[self.current_idx],
            self.motion_base_lin_vels[self.current_idx],
            self.motion_base_ang_vels[self.current_idx],
            self.motion_dof_poss[self.current_idx],
            self.motion_dof_vels[self.current_idx],
        )
        self.current_idx += 1
        return state, flag


def main():
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device, dt=1.0 / args_cli.output_fps
    )
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(ReplaySceneCfg(num_envs=1, env_spacing=2.0))
    robot = scene["robot"]
    sim.reset()

    motion = MotionLoader(
        args_cli.input_file, args_cli.input_fps, args_cli.output_fps, sim.device
    )

    # 记录字典
    # 名称列表与 joint_pos / body_pos_w 的列顺序严格对应（即
    # robot.data.joint_names / robot.data.body_names 的原生顺序）。
    # 下游 loader（MotionDataset / visualize_motion_npz / augment_motion_npz）
    # 优先读取 npz 自带的 joint_names / body_names，避免再靠调用方传入。
    joint_names_np = np.asarray(list(robot.data.joint_names))
    body_names_np = np.asarray(list(robot.data.body_names))
    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
        "joint_names": joint_names_np,
        "body_names": body_names_np,
    }

    # 预先找到关节索引
    valid_joint_names = [n for n in scene.cfg.robot.joint_sdk_names if n != ""]
    robot_idx = robot.find_joints(valid_joint_names, preserve_order=True)[0]

    print("[INFO]: 开始回放并记录数据...")
    saved = False
    while True:
        state, is_end = motion.get_next_state()
        if is_end:
            # 保存 NPZ
            saved = True
            save_path = args_cli.output_name or args_cli.input_file.replace(
                ".csv", ".npz"
            )
            # 仅 stack 逐帧追加的列；fps / joint_names / body_names 保持原样。
            _skip_stack = {"fps", "joint_names", "body_names"}
            for k in log:
                if k not in _skip_stack:
                    log[k] = np.stack(log[k])
            np.savez(save_path, **log)
            print(
                f"[INFO]: 数据已成功保存至 {save_path} "
                f"(J={len(joint_names_np)}, N_bodies={len(body_names_np)})"
            )
            # break

        (pos, rot, lin_vel, ang_vel, dof_p, dof_v) = state

        # 写入 Root 状态
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3], root_states[:, 3:7] = pos, rot
        root_states[:, 7:10], root_states[:, 10:] = lin_vel, ang_vel
        robot.write_root_state_to_sim(root_states)

        # 写入 Joint 状态
        full_dof_p = robot.data.default_joint_pos.clone()
        full_dof_v = robot.data.default_joint_vel.clone()
        full_dof_p[:, robot_idx] = dof_p
        full_dof_v[:, robot_idx] = dof_v
        robot.write_joint_state_to_sim(full_dof_p, full_dof_v)

        sim.render()
        scene.update(sim.get_physics_dt())

        if not saved:
            # 记录数据（保持与你的 npz 格式一致）
            log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0].cpu().numpy().copy())
            log["body_lin_vel_w"].append(
                robot.data.body_lin_vel_w[0].cpu().numpy().copy()
            )
            log["body_ang_vel_w"].append(
                robot.data.body_ang_vel_w[0].cpu().numpy().copy()
            )


if __name__ == "__main__":
    main()
