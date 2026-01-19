import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from scipy.spatial.transform import Rotation as R
from PIL import Image
import cv2

import torch
from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
cfgs = parser.parse_args()

#手眼标定矩阵（眼在手上）
T_FLANGE_CAM = np.array([
    [-0.08991802, -0.99594771,  0.00170803, -0.03926148],
    [ 0.99594155, -0.08991049,  0.00406658, -0.01666193],
    [-0.00389653,  0.00206676,  0.99999027,  0.02159971],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
    ])

#手中心坐标系到末端法兰盘坐标系的变换矩阵
# 法兰到TCP (Flange -> TCP)
T_FLANGE_TCP = np.array([
    [0.0, -1.0, 0.0,  0.160],
    [1.0, 0.0, 0.0,  0.015],
    [0.0, 0.0, 1.0,  0.055],  
    [0.0, 0.0, 0.0,  1.000]
])


def pose2_to_matrix(pose):
    """
    输入：CR5 机械臂姿态 [x, y, z, rx, ry, rz]
    单位：mm, degrees
    输出：4x4 变换矩阵 (单位：米)
    """
    x,y,z,rx,ry,rz = pose
    t = np.array([x,y,z])/1000.0
    r = R.from_euler('xyz',[rx,ry,rz],degrees=True).as_matrix()
    T = np.eye(4)
    T[:3,:3] = r
    T[:3,3] = t
    return T

def matrix_to_pose(T):
    """
    输入：4x4 变换矩阵 (单位：米)
    输出：CR5 机械臂姿态 [x, y, z, rx, ry, rz]
    单位：mm, degrees
    """
    r_matrix = T[:3,:3]
    t_matrix = T[:3,3]
    r_euler = R.from_matrix(r_matrix).as_euler('xyz',degrees=True)
    t = t_matrix * 1000.0
    return np.concatenate([t,r_euler])


def get_net():
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net

def get_and_process_data(color_img,depth_img,mask_path):
    # load data
    if color_img.shape[-1]==3:
        color = cv2.cvtColor(color_img,cv2.COLOR_BGR2RGB)
    else:
        color = color_img
    color = np.array(color, dtype=np.float32) / 255.0
    depth = depth_img
    img_h, img_w = depth.shape 
    print(f"Debug: 实际读取到的深度图尺寸为: {img_w}x{img_h}")
    # workspace_mask = np.array(Image.open(mask_path))
    workspace_mask =mask_path
    intrinsic =np.array([
        [692.683594, 0.,         645.506226],
        [0.,         692.615234, 358.246460],
        [0.,         0.,         1.]
    ])
    factor_depth = 1000.0
    # generate cloud
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    mask = (workspace_mask & (depth > 0))
    cloud_masked = cloud[mask]
    color_masked = color[mask]
    # sample points
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # convert data
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" )
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud

def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg

def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    return gg


def pose_filter(gg, cloud):
    # --- 1. 碰撞检测 (基础) ---
    if cfgs.collision_thresh > 0:
        gg = collision_detection(gg, np.array(cloud.points))

    # --- 2. 角度筛选 (进给方向 & 旋转方向) ---
    # A. 进给方向 (Approach) - 保证垂直向下
    vector_axis = np.array([0, 0, 1], dtype=np.float32)
    approach_vectors = gg.rotation_matrices[:, :, 0] 
    dots_app = np.dot(approach_vectors, vector_axis)
    dots_app = np.clip(dots_app, -1.0, 1.0)
    app_angles = np.degrees(np.arccos(dots_app))
    
    app_mask = app_angles < 30.0 
    gg = gg[app_mask] # 先过滤，减少计算量

    # B. 平面旋转 (Closing) - 保证手腕不扭曲
    reference_axis = np.array([1, 0, 0], dtype=np.float32) # 根据实际安装调整 [1,0,0] 或 [0,1,0]
    closing_vectors = gg.rotation_matrices[:, :, 1] 
    dots_rot = np.dot(closing_vectors, reference_axis)
    dots_rot = np.clip(dots_rot, -1.0, 1.0)
    rot_angles = np.degrees(np.arccos(dots_rot))
    
    rot_mask = rot_angles < 90.0
    gg = gg[rot_mask]

    # --- 3. 向心筛选 (Center Bias) - 新增部分 ---
    if len(gg) > 0:
        # Step A: 计算物体点云的质心
        points = np.asarray(cloud.points)
        if len(points) > 0:
            object_center = np.mean(points, axis=0) # [x, y, z] 质心坐标
        else:
            object_center = np.array([0, 0, 0])

        # Step B: 计算所有抓取点到质心的欧氏距离
        # gg.translations 是抓取点的位置 (N, 3)
        grasp_centers = gg.translations
        # 计算距离: sqrt((x1-x2)^2 + ...)
        dists = np.linalg.norm(grasp_centers - object_center, axis=1)

        # Step C: 分数重加权 (这一步最关键)
        # 我们不直接删除边缘抓取，而是降低它们的分数，让它们排在后面。
        # 权重系数: 比如 2.0，意味着每偏离中心 10cm (0.1m)，分数就扣掉 0.2
        dist_penalty_weight = 2.0 
        
        # 新分数 = 旧分数 - (距离 * 权重)
        # 这样距离越小(越靠中间)，扣分越少，排名越靠前
        gg.scores = gg.scores - (dists * dist_penalty_weight)

        # (可选) 如果你非常激进，想直接删掉距离中心太远(比如超过5cm)的：
        # dist_mask = dists < 0.05 
        # gg = gg[dist_mask]

    # --- 4. 后处理 ---
    gg.nms()            # 非极大值抑制 (会根据新的 scores 进行筛选，所以会自动保留中间的)
    gg.sort_by_score()  # 再次排序
    gg = gg[:20]        # 取前20个
    
    return gg

def grasp_generation(color_img,depth_img,mask_path):
    net = get_net()
    end_points, cloud = get_and_process_data(color_img,depth_img,mask_path)
    gg = get_grasps(net, end_points)
    grasp_poseture = pose_filter(gg, cloud)

    return grasp_poseture, cloud

def targetpose_generation(color_img,depth_img,mask_path,current_pose):
    # grasp_poseture = grasp_generation(color_img,depth_img,mask_path)
    # if len(grasp_poseture)==0:
    #     print("没有检测到合适的抓取姿态！")
    #     return None
    
    # best_grasp = grasp_poseture[0]
    grasp_poseture, cloud = grasp_generation(color_img, depth_img, mask_path)
    
    if len(grasp_poseture) == 0:
        print("没有检测到合适的抓取姿态！")
        return None
    
    # 取出分数最高的那个抓取
    best_grasp = grasp_poseture[0]

    print(f"Debug: 正在显示最佳抓取 (Score: {best_grasp.score:.4f})... 请关闭窗口以继续机械臂移动")
    
    # 技巧：切片 [0:1] 可以保留 GraspGroup 类型，方便转几何体
    best_grasp_gg = grasp_poseture[0:1] 
    
    # 转为 Open3D 几何体
    best_gripper_geo = best_grasp_gg.to_open3d_geometry_list()
    
    # 把选中的夹爪涂成红色 [R, G, B] -> [1, 0, 0]
    for geo in best_gripper_geo:
        geo.paint_uniform_color([1, 0, 0]) 

    other_grasps_geo = grasp_poseture[1:].to_open3d_geometry_list()
    for geo in other_grasps_geo:
        geo.paint_uniform_color([0, 1, 0]) # 绿色
    
    # 创建坐标轴辅助观看
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    # 渲染：点云 + 红色最佳抓取 + 坐标轴
    o3d.visualization.draw_geometries([cloud, *best_gripper_geo, *other_grasps_geo, coord_frame], 
                                      window_name="Grasp Visualization")

    T_cam_grasp =np.eye(4)
    T_cam_grasp[:3,:3] = best_grasp.rotation_matrix
    T_cam_grasp[:3,3] = best_grasp.translation

    rotation_corr = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0]
    ])
    T_corr = np.eye(4)
    T_corr[:3,:3] = rotation_corr

    T_cam_target = T_cam_grasp @ T_corr

    T_base_flange = pose2_to_matrix(current_pose)
    T_base_targetObj= T_base_flange @ T_FLANGE_CAM @ T_cam_target #这里计算出来的是物体目标坐标系在机械臂基座坐标系下的位姿
    T_base_targetArm = T_base_targetObj @ np.linalg.inv(T_FLANGE_TCP) #但是我们实际可以控制命令的是机械臂末端法兰盘的位姿，所以需要转换一下
    arm_target_pose = matrix_to_pose(T_base_targetArm)

    return arm_target_pose

# if __name__ == '__main__':