"""
原始npy格式数据
fps=30
root_pos (111,3) 机器人 Pelvis（盆骨）在三维空间中的 [x,y,z] 位置。从数据看 z≈0.78m，这是 G1 的标准站立高度
root_rot (111,4) 机器人 Pelvis 的旋转四元数 [x,y,z,w]，描述了机器人在空间中的朝向
dof_pos (111,29)
"""

import numpy as np
import os

# 配置路径
input_path = "/home/woan/下载/g1_qie_motion.npy"
output_path = "/home/woan/HumanoidProject/unitree_rl_lab/motion_datasets/penguin/g1_qie_motion.csv"


def convert_npy_to_csv():
    # 1. 加载高版本 npy
    data_dict = np.load(input_path, allow_pickle=True)
    print(data_dict)

    root_pos = data_dict["root_pos"]  # (N, 3)
    root_rot = data_dict["root_rot"]  # (N, 4)  已经是 [x, y, z, w]
    dof_pos = data_dict["dof_pos"]  # (N, 29)

    # 2. 合并数据：位置(3) + 旋转(4) + 关节(29) = 36列
    combined_data = np.hstack((root_pos, root_rot, dof_pos))

    # 3. 保存为 CSV
    np.savetxt(output_path, combined_data, delimiter=",")

    print(f"--- 转换成功 ---")
    print(f"输入帧数: {combined_data.shape[0]}")
    print(f"输出路径: {output_path}")
    print(f"数据格式: 前3列位移, 4-7列四元数(xyzw), 后29列关节角度")


if __name__ == "__main__":
    convert_npy_to_csv()
