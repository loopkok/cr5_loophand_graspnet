""" Demo to show prediction results.
    Author: chenxi-wang
"""

import glob
import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
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
from vlm_process import segment_image

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
cfgs = parser.parse_args()


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

def get_and_process_data(data_dir):
    # load data
    color_dir = os.path.join(data_dir, "color")
    depth_dir = os.path.join(data_dir, "depth")

    # 自动选取最新的一帧彩色图，并按同一时间戳匹配原始深度图
    color_files = sorted(glob.glob(os.path.join(color_dir, "*_color.png")))
    if not color_files:
        raise FileNotFoundError(f"未在 {color_dir} 找到 *_color.png 图像")
    color_path = color_files[-1]
    timestamp = os.path.basename(color_path).split("_")[1]
    depth_path = os.path.join(depth_dir, f"capture_{timestamp}_depth_raw.png")
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"未找到与 {color_path} 匹配的深度图: {depth_path}")
    print(f"Debug: 使用彩色图 {color_path}")
    print(f"Debug: 使用深度图 {depth_path}")

    color_img = cv2.imread(color_path)
    color = np.array(Image.open(color_path), dtype=np.float32) / 255.0
    depth = np.array(Image.open(depth_path))
    img_h, img_w = depth.shape  # 读出来的可能是 (400, 640)
    print(f"Debug: 实际读取到的深度图尺寸为: {img_w}x{img_h}")
    #depth_raw
    #seg_mask= np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
    seg_mask = segment_image(color_img, output_mask='./data/debug_data/mask_output.png')
    
    if seg_mask is not None:
        # 转换成bool类型，给到graspnet进行深度点云索引，只要像素值大于 0，就变成 True，否则是 False
        workspace_mask = seg_mask > 0  
    # meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
    # intrinsic = meta['intrinsic_matrix']
    # factor_depth = meta['factor_depth']
    intrinsic =np.array([
        [692.683594, 0.,         645.506226],
        [0.,         692.615234, 358.246460],
        [0.,         0.,         1.]
    ])
    factor_depth = 1000.0
    # generate cloud
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    print(f"Debug: 使用的相机内参为: \n{intrinsic}")
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    print(f"Debug: 原始点云形状: {cloud.shape}")
    # get valid points
    mask = (workspace_mask & (depth > 0))
    print(f"Debug: 有效点云数量: {np.sum(mask)}")
    cloud_masked = cloud[mask]
    color_masked = color[mask]
    print(f"Debug: 有效点云形状: {cloud_masked.shape}")
    # sample points
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]
    print(f"Debug: Sampled point cloud shape: {cloud_sampled.shape}")

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

def vis_grasps(gg, cloud):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:10]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])

def demo(data_dir):
    net = get_net()
    end_points, cloud = get_and_process_data(data_dir)
    gg = get_grasps(net, end_points)
    if cfgs.collision_thresh > 0:
        gg = collision_detection(gg, np.array(cloud.points))
    vis_grasps(gg, cloud)
if __name__=='__main__':
    # data_dir = 'doc/example_data/debug_data'
    data_dir = os.path.join(ROOT_DIR, 'data', 'apple_data')
    demo(data_dir)
