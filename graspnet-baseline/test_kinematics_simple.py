import sys
import os
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

# ================= 路径修复 =================
# 获取当前脚本所在的目录绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上一级目录 (即项目根目录)
project_root = os.path.dirname(current_dir)
# 将根目录加入 Python 搜索路径
sys.path.append(project_root)
sys.path.extend([
    project_root,                            
    os.path.join(project_root, 'arm_CR'),    
    os.path.join(project_root, 'loophand'),  
])

from arm_CR.cr5 import CR5

# ================= 1. 填入你的实际标定矩阵 =================

HOME_POSE = [-84.6480479, -400.69554335, 357.43467829, -176.9688317, -0.77024222, -88.71960534]
TEST_POSE = [-213.51800155046612, -812.1209797440353, 3.610127514056714, 175.4517131714603, -53.44449229288822, 55.21804889143199]

# 手眼标定 (Flange -> Camera)
T_FLANGE_CAM = np.array([
    [-0.08991802, -0.99594771,  0.00170803, -0.03926148],
    [ 0.99594155, -0.08991049,  0.00406658, -0.01666193],
    [-0.00389653,  0.00206676,  0.99999027,  0.02159971],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
])

# 法兰到TCP (Flange -> TCP)
T_FLANGE_TCP = np.array([
    [0.0, -1.0, 0.0,  0.125],
    [1.0, 0.0, 0.0,  0.000],
    [0.0, 0.0, 1.0,  0.090],  
    [0.0, 0.0, 0.0,  1.000]
])

def pose_to_matrix(pose_vec):
    """[x,y,z,rx,ry,rz] (mm, deg) -> 4x4 Matrix (m)"""
    x, y, z, rx, ry, rz = pose_vec
    t = np.array([x, y, z]) / 1000.0  # mm 转 m
    r = R.from_euler('xyz', [rx, ry, rz], degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = r
    T[:3, 3] = t
    return T

def matrix_to_pose(matrix):
    """4x4 Matrix (m) -> [x,y,z,rx,ry,rz] (mm, deg)"""
    t = matrix[:3, 3] * 1000.0        # m 转 mm
    r = R.from_matrix(matrix[:3, :3]).as_euler('xyz', degrees=True)
    return np.concatenate([t, r]).tolist()

def main():
    print("[System] 连接机械臂...")
    robot = CR5()
    robot.connect()
    robot.dashboard.SpeedFactor(2) # 安全速度
    time.sleep(1)

    # 1. 获取当前基座到法兰的矩阵 (Base -> Flange_Current)
    current_pose_vec = robot.current_actual
    T_base_flange_curr = pose_to_matrix(current_pose_vec)
    print(f"[Info] 当前位姿: {current_pose_vec}")

    # 2. 定义目标：相机坐标系下的 (0, 0, 0.1) 即前方10cm
    # 目标：TCP 能够到达 相机前方10cm 的位置
    target_pos_cam = np.array([0, 0, 0.1]) 
    
    # 目标旋转：这里假设 TCP 的姿态要和相机姿态一致 (即平移过去)
    target_rot_cam = np.eye(3) 

    T_cam_target = np.eye(4)
    T_cam_target[:3, :3] = target_rot_cam
    T_cam_target[:3, 3] = target_pos_cam

    # 3. 核心计算链
    # 逻辑：我们要计算 目标法兰(Flange_Target) 在 基座(Base) 下的位置
    
    # 第一步：计算相机在基座下的位置 (Base -> Flange_Curr -> Cam)
    T_base_cam = T_base_flange_curr @ T_FLANGE_CAM
    
    # 第二步：计算 目标TCP点 在基座下的位置 (Base -> Cam -> Target_TCP)
    T_base_tcp_target = T_base_cam @ T_cam_target
    
    # 第三步：回推 目标法兰 的位置 (已知 Flange -> TCP，求 Flange)
    # Flange_Target = TCP_Target * inv(Flange -> TCP)
    T_base_flange_target = T_base_tcp_target @ np.linalg.inv(T_FLANGE_TCP)

    # 4. 转换回机械臂指令格式
    init_pose =HOME_POSE
    target_pose_vec = matrix_to_pose(T_base_flange_target)
    
    print("-" * 40)
    print(f"[Check] 你的 TCP 设置: X偏移={T_FLANGE_TCP[0,3]}m, Z偏移={T_FLANGE_TCP[2,3]}m")
    print("-" * 40)
    print(f"[Plan] 目标法兰位姿: {np.round(target_pose_vec, 2)}")
    
    user_input = input("按 'y' 开始移动 (请手放在急停上): ")
    if user_input.lower() == 'y':
        robot.MoveCartesian(HOME_POSE)
        time.sleep(3)
        print("移动完成。")
        print("验证方法：")
        print("  如果你设定的 T_FLANGE_TCP 是准确的（指尖位置），")
        print("  那么现在【指尖】应该正好位于【相机视野中心前方10cm】处。")
    else:
        print("已取消")

if __name__ == "__main__":
    main()