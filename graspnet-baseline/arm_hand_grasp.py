import os
import sys
import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image 

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

# 导入硬件模块
from arm_CR.cr5 import CR5
from gemini335 import Gemini335
# [新增] 导入触觉手控制器
from loophand.tactile_grasping import ShearGraspController 
from process import targetpose_generation
from vlm_process import segment_image

# ================= 配置区域 =================

# 1. 机械臂点位 [x, y, z, rx, ry, rz] (笛卡尔坐标)
HOME_POSE = [-84.6480479, -600.69554335, 357.43467829, -176.9688317, -0.77024222, -88.71960534]
PLACE_POSE = [-84.6480479, -700.69554335, 357.43467829, -176.9688317, -0.77024222, -88.71960534]

# 2. 抓取参数
PRE_GRASP_DIST = 100.0   # 预抓取距离 10cmn

# 3. 调试路径（相对 graspnet-baseline/ 运行）
SAVE_DIR = "./data/debug_data"
MASK_PATH = os.path.join(SAVE_DIR, "workspace_mask.png")

# 4. 运动参数
VEL_RATIO = 15 
ACC_RATIO = 8

# ===========================================

class GraspSystem:
    def __init__(self):
        self.init_hardware()
        self.ensure_dirs()
        
    def ensure_dirs(self):
        os.makedirs(SAVE_DIR, exist_ok=True)
        
    def init_hardware(self):
        print("[System] 初始化全系统硬件...")
        
        # 1. 机械臂 CR5
        try:
            # 使用你提供的 CR5 类
            self.robot = CR5()
            self.robot.connect()
            
            # 设置速度 (CR5类没有封装这个，只能调dashboard，这部分保持原样)
            self.robot.dashboard.SpeedFactor(VEL_RATIO)
            self.robot.dashboard.AccJ(ACC_RATIO)
            self.robot.dashboard.AccL(ACC_RATIO)
            
            # CR5类中封装了 ClearRobotError 的线程，这里确保状态继续
            self.robot.dashboard.Continue()
            print("[Robot] CR5 就绪")
        except Exception as e:
            print(f"[Robot] 连接失败: {e}")
            sys.exit(1)

        # 2. 深度相机 Gemini335
        try:
            self.camera = Gemini335()
            for _ in range(10): self.camera.capture_frame()
            print("[Camera] Gemini335 就绪")
        except Exception as e:
            print(f"[Camera] 启动失败: {e}")
            sys.exit(1)

        # 3. 触觉灵巧手初始化
        try:
            print("[Hand] 正在启动触觉手控制器...")
            hand_urdf = os.path.join(project_root, "loophand", "urdf", "aero_hand_open_left.urdf")
            self.hand_controller = ShearGraspController(urdf_path=hand_urdf)
            self.hand_controller.start()
            print("[Hand] 触觉控制线程已在后台运行")
        except Exception as e:
            print(f"[Hand] 启动失败: {e}")
            sys.exit(1)

    # --- 辅助功能 ---
    def wait_until_arrival(self, target_pose, threshold_xyz=10.0, timeout=40.0):
        """
        机械臂到位等待
        注意：ServoP/MoveCartesian 是非阻塞的，必须用这个函数卡住程序
        """
        print(f"    ... 机械臂运动中 (目标: {target_pose[:3]})")
        start_time = time.time()
        target_xyz = np.array(target_pose[:3])
        
        while time.time() - start_time < timeout:
            # 使用 cr5.py 中定义的 current_actual (由 GetFeed 线程更新)
            curr = self.robot.current_actual
            
            # 数据有效性检查
            if curr is None or len(curr) < 6 or curr[0] == -1:
                time.sleep(0.05)
                continue
                
            # 计算距离
            curr_xyz = np.array(curr[:3])
            dist = np.linalg.norm(curr_xyz - target_xyz)
            
            if dist < threshold_xyz:
                return True
            time.sleep(0.1)
            
        print("[Warning] 机械臂运动超时")
        return False

    def calculate_pre_grasp(self, grasp_pose):
        """计算预抓取点"""
        pre_pose = np.array(grasp_pose, dtype=float).copy()
        rx, ry, rz = grasp_pose[3:]
        r = R.from_euler('xyz', [rx, ry, rz], degrees=True)
        z_axis = r.as_matrix()[:, 2]
        pre_pose[:3] -= z_axis * PRE_GRASP_DIST
        return pre_pose.tolist()

    def generate_simple_mask(self, img_shape):
        h, w = img_shape[:2]
        # 直接创建布尔类型的掩码 
        mask = np.zeros((h, w), dtype=bool)
        
        # 将感兴趣区域 (ROI) 设为 True
        mask[100:620, 200:1080] = True 
        
        return mask

    # --- 主逻辑 ---
    def run(self):
        print("================ 触觉智能抓取开始 ================")
        
        # 初始状态：确保手是松开的
        self.hand_controller.set_command("RELEASE")
        time.sleep(2.0) 
        
        while True:
            try:
                # ---------------- Step 1: 初始复位 ----------------
                print("\n>>> Step 1: 机械臂归位 & 手部复位")
                
                # 【修改】使用 cr5.py 中的接口 MoveCartesian
                # 注意：HOME_POSE 是列表，不需要加 *
                self.robot.MoveCartesian(HOME_POSE)
                
                self.hand_controller.set_command("IDLE") 
                
                self.wait_until_arrival(HOME_POSE)
                time.sleep(1.0) 

                # ---------------- Step 2: 视觉感知 ----------------
                print("\n>>> Step 2: 视觉识别")
                color, raw_depth, _, _ = self.camera.capture_frame()
                if color is None: continue
                
                cv2.imwrite(f"{SAVE_DIR}/latest_color.png", color)
                cv2.imwrite(f"{SAVE_DIR}/latest_depth.png", raw_depth)
                # mask = self.generate_simple_mask(color.shape)
                seg_mask = segment_image(color, output_mask=os.path.join(SAVE_DIR, "mask_output.png"))
                
                if seg_mask is not None:
                    # 转换成bool类型，给到graspnet进行深度点云索引，只要像素值大于 0，就变成 True，否则是 False
                    seg_mask_bool = seg_mask > 0  
                else:
                    print("[警告] 掩码生成失败，跳过此次抓取规划")
                    continue 

                current_pose = self.robot.current_actual
                if current_pose is None or len(current_pose) < 6: continue
                
                print(f"[Info] 拍照位姿: {list(current_pose)}")

                # 算法生成目标
                target_pose = targetpose_generation(color, raw_depth, seg_mask_bool, current_pose)
                if target_pose is None:
                    print("[Info] 无目标，重试...")
                    time.sleep(1)
                    continue

                print(f"\n[Check] 计算出的目标位姿: {target_pose}")
                user_input = input(">>> 确认执行抓取吗？(输入 y 执行，输入 n 跳回原点): ")

                if user_input.lower() != 'y':
                    print("[Info] 用户取消抓取，正在重新开始循环...")
                    # 这里不需要额外动作，因为 continue 会直接跳到下一次循环的开头
                    # 循环开头会执行 Step 1: 机械臂归位，符合你的要求
                    continue

                # ---------------- Step 3: 预抓取准备 ----------------
                pre_grasp_pose = self.calculate_pre_grasp(target_pose)
                print(f"[Info] 预抓取点: {pre_grasp_pose}")
                # if target_pose[2] < 5.0:
                #     print(f"[Warning] 目标点太低 ({target_pose[2]})，跳过")
                #     continue
                
                print("\n>>> Step 3: 机械臂前往预抓取点 & 灵巧手预姿态")
                
                print("    [Hand] 切换到预姿态 (Pre-Shape)")
                self.hand_controller.set_command("PRE_SHAPE")
                
                # 
                self.robot.MoveCartesian(pre_grasp_pose)
                self.wait_until_arrival(pre_grasp_pose, threshold_xyz=5.0)
                
                time.sleep(0.5)

                # ---------------- Step 4: 执行抓取 ----------------
                print("\n>>> Step 4: 插入并执行自适应抓取")
                
                # 机械臂慢速插入
                self.robot.dashboard.SpeedFactor(2)
                
                # 
                self.robot.MoveCartesian(target_pose)
                self.wait_until_arrival(target_pose, threshold_xyz=2.0)
                
                # 启动触觉自适应抓取
                print("    [Hand] 启动自适应抓取 (Active Grasp)...")
                self.hand_controller.set_command("GRASP")
                
                print("    ... 等待力闭合稳定 (3s)")
                time.sleep(3.0)
                
                # ---------------- Step 5: 搬运与放置 ----------------
                print("\n>>> Step 5: 搬运放置")
                
                # 抬起 (手保持 GRASP 状态)
                self.robot.dashboard.SpeedFactor(VEL_RATIO)
                
                #  (抬起)
                self.robot.MoveCartesian(pre_grasp_pose)
                self.wait_until_arrival(pre_grasp_pose)
                
                # (去放置点)
                # PLACE_POSE 也是笛卡尔坐标，所以用 MoveCartesian
                self.robot.MoveCartesian(PLACE_POSE)
                self.wait_until_arrival(PLACE_POSE)
                
                # 松开物体
                print("    [Hand] 松开 (Releasing)...")
                self.hand_controller.set_command("RELEASE")
                time.sleep(2.0) 
                
                # ---------------- Step 6: 循环 ----------------
                print("\n>>> Step 6: 本轮结束")
                time.sleep(1.0)

            except KeyboardInterrupt:
                print("\n[System] 停止")
                # self.hand_controller.stop_thread()
                self.hand_controller.set_command("RELEASE")
                self.robot.dashboard.SpeedFactor(20)
                self.robot.MoveCartesian(HOME_POSE)
                break
            except Exception as e:
                print(f"[Error] {e}")
                time.sleep(3)

if __name__ == "__main__":
    system = GraspSystem()
    system.run()