import os
import time
import logging
import threading
import numpy as np
import pinocchio as pin
from collections import deque
from enum import Enum
from typing import List

_LOOPHAND_ROOT = os.path.dirname(os.path.abspath(__file__))

# ================= 引入依赖 (根据你的实际项目路径调整) =================
# 假设你的目录结构支持以下导入，如果报错请调整路径
from aero_open_sdk.aero_hand import AeroHand
from extern_driver.tactile_driver import TactileSensor
from controller.pid import BoundedPID
from controller.admittancecontroller import AdmittanceController

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ShearControl")

# ================= 1. 硬件配置 & 常量 (源自代码1) =================
SENSOR_TO_LINK_MAP = {
    "Thumb-Tip": "left_thumb_tip_link",
    "Index-Tip": "left_index_tip_link",
    "Index-Mid": "left_index_proximal_link", 
    "Middle-Tip": "left_middle_tip_link",
    "Middle-Mid": "left_middle_proximal_link", 
    "Ring-Tip": "left_ring_tip_link",
    "Ring-Mid": "left_ring_proximal_link", 
    "Pinky-Tip": "left_pinky_tip_link",
    "Pinky-Mid": "left_pinky_proximal_link" 
}

JOINT_ORDER_FULL = [
    "left_thumb_cmc_abd", "left_thumb_cmc_flex", "left_thumb_mcp", "left_thumb_ip",
    "left_index_mcp_flex", "left_index_pip", "left_index_dip",
    "left_middle_mcp_flex", "left_middle_pip", "left_middle_dip",
    "left_ring_mcp_flex", "left_ring_pip", "left_ring_dip",
    "left_pinky_mcp_flex", "left_pinky_pip", "left_pinky_dip"
]

ACTIVE_JOINT_NAMES = [
    "left_thumb_cmc_abd", "left_thumb_cmc_flex", "left_thumb_mcp",
    "left_index_mcp_flex", "left_middle_mcp_flex", "left_ring_mcp_flex", "left_pinky_mcp_flex"
]

class GraspState(Enum):
    IDLE = -1          # 待机
    PRE_SHAPE = 0      # 预姿态 (调整拇指)
    ACTIVE_GRASP = 1   # 接触并自适应抓取
    RELEASING = 2      # 松开物体 (来自代码2的新增状态)

# ================= 2. 信号处理 (源自代码1，保持不变) =================
class SignalProcessor:
    def __init__(self, alpha=0.4, history_len=5, derivative_alpha=0.1):
        self.alpha = alpha
        self.derivative_alpha = derivative_alpha
        self.history_len = history_len
        self.epsilon = 1e-3
        self.lowpass = np.zeros(3)
        self.history = deque(maxlen=history_len)
        self.shear_prev = np.zeros(3)
        
    def process(self, raw_force: np.ndarray, dt: float):
        self.lowpass = self.alpha * raw_force + (1.0 - self.alpha) * self.lowpass
        self.history.append(self.lowpass.copy())
        
        if len(self.history) >= 2:
            prev = self.history[0]
            shear_delta = (self.lowpass - prev) / max(dt * (len(self.history) - 1), self.epsilon)
        else:
            shear_delta = np.zeros(3)
            
        self.shear_prev = (self.derivative_alpha * shear_delta + (1.0 - self.derivative_alpha) * self.shear_prev)
        
        tangential_force = np.linalg.norm(self.lowpass[:2])
        normal_force = self.lowpass[2]
        slip_ratio = tangential_force / max(abs(normal_force), self.epsilon)
        
        return {
            "force_filtered": self.lowpass.copy(),
            "shear_rate": self.shear_prev.copy(),
            "normal_force": normal_force,
            "tangential_force": tangential_force,
            "slip_ratio": slip_ratio
        }

# ================= 3. 控制器主类 (Thread化) =================
class ShearGraspController(threading.Thread):
    def __init__(self, urdf_path=os.path.join(_LOOPHAND_ROOT, "urdf", "aero_hand_open_left.urdf"), port_hand="/dev/ttyACM1", port_tactile="/dev/ttyACM0"):
        super().__init__()
        self.daemon = True # 守护线程
        
        logger.info("Init Hardware & Model...")
        self.hand = AeroHand(port=port_hand) 
        self.tactile = TactileSensor(port=port_tactile)

        time.sleep(1.0) 
        self.tactile.tare(duration=2.0)
        
        # 模型初始化
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        
        # 索引映射
        self.pin_active_indices = []
        self.user_active_indices = []
        self.finger_map = {"Thumb": [], "Index": [], "Middle": [], "Ring": [], "Pinky": []}
        self.thumb_abd_index = None    

        # 初始化索引 (逻辑源自代码1)
        for name in ACTIVE_JOINT_NAMES:
            if self.model.existJointName(name):
                joint_id = self.model.getJointId(name)
                idx_v = self.model.idx_vs[joint_id]
                self.pin_active_indices.append(idx_v)
            
            if name in JOINT_ORDER_FULL:
                idx = JOINT_ORDER_FULL.index(name)
                self.user_active_indices.append(idx)
                
        for idx, name in enumerate(JOINT_ORDER_FULL):
            if "abd" in name: 
                if "thumb" in name: self.thumb_abd_index = idx
                continue
            if "thumb" in name:  self.finger_map["Thumb"].append(idx)
            elif "index" in name: self.finger_map["Index"].append(idx)
            elif "middle" in name: self.finger_map["Middle"].append(idx)
            elif "ring" in name:   self.finger_map["Ring"].append(idx)
            elif "pinky" in name:  self.finger_map["Pinky"].append(idx)

        # 参数初始化
        self.dt = 0.01
        self._gravity_world = np.array([0.0, -1.0, 0.0], dtype=float)
        
        self.modules = {}
        PID_PARAMS = (0.1, 0.0, 0.0, 5.0) 
        
        # 物理参数配置 (源自代码1)
        ADM_M = np.array([0.11, 0.11, 0.11]) 
        ADM_D = np.array([12.0, 12.0, 12.0])    
        ADM_K = np.array([0.0, 0.0, 0.0]) 

        for sensor_name, link_name in SENSOR_TO_LINK_MAP.items():
            self.modules[sensor_name] = {
                "link_name": link_name,
                "processor": SignalProcessor(),
                "pid": BoundedPID(*PID_PARAMS),
                "contact_active": False,
                "has_anchored": False,
                "admittance": AdmittanceController(ADM_M, ADM_K, ADM_D, self.dt, max_vel=0.5, max_acc=10.0),
            }
        
        self.current_q = np.zeros(len(JOINT_ORDER_FULL))
        self.running = False
        
        # 线程同步用的锁和状态
        self.lock = threading.Lock()
        self.current_state = GraspState.IDLE
        
        self.params = {
            "mu_est": 0.4,           
            "safety_margin": 1.5,    
            "min_force": 3.0,        
            "approach_thumb_abd_speed": 0.03,
            "approach_thumb_speed": 0.01,
            "approach_other_speed": 0.03,
            "contact_thresh": 0.1,  
            "thumb_abd_target": 1.65, 
            "pre_shape_tol": 0.05,
            "limit_speed": 0.06,
        }

    # ================= 4. 辅助函数 (严格复刻代码1) =================
    def _map_7_to_16(self, compact_deg: List[float]) -> np.ndarray:
        q_7 = np.deg2rad(np.array(compact_deg))
        q_16 = np.zeros(len(JOINT_ORDER_FULL))
        q_16[0] = q_7[0]; q_16[1] = q_7[1]; q_16[2] = q_7[2]; q_16[3] = q_7[2] * 1.0 
        val_idx = q_7[3]; q_16[4] = val_idx; q_16[5] = val_idx * 0.8; q_16[6] = val_idx * 0.6 
        val_mid = q_7[4]; q_16[7] = val_mid; q_16[8] = val_mid * 0.8; q_16[9] = val_mid * 0.6
        val_rng = q_7[5]; q_16[10] = val_rng; q_16[11] = val_rng * 0.8; q_16[12] = val_rng * 0.6
        val_pnk = q_7[6]; q_16[13] = val_pnk; q_16[14] = val_pnk * 0.8; q_16[15] = val_pnk * 0.6
        return np.array(q_16, dtype=np.float64)

    def _update_kinematics(self, q_user: np.ndarray, link_name: str):
        q_pin = pin.neutral(self.model)
        for user_i, name in enumerate(JOINT_ORDER_FULL):
            if self.model.existJointName(name):
                pin_id = self.model.getJointId(name)
                pin_q_idx = self.model.idx_qs[pin_id]
                q_pin[pin_q_idx] = q_user[user_i]
        
        pin.forwardKinematics(self.model, self.data, q_pin)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, q_pin)
        
        frame_id = self.model.getFrameId(link_name)
        J_spatial = pin.getFrameJacobian(self.model, self.data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        
        J_lin_origin = np.array(J_spatial[:3, :], copy=False)
        J_ang = np.array(J_spatial[3:, :], copy=False)
        rot = np.array(self.data.oMf[frame_id].rotation, copy=False)
        
        if "proximal" in link_name: local_offset = np.array([0.0, 0.0, 0.025]) 
        else: local_offset = np.array([0.0, 0.0, 0.0])
        
        r_world = rot @ local_offset
        current_pos_sensor = self.data.oMf[frame_id].translation + r_world
        
        rx, ry, rz = r_world
        skew_r = np.array([[0, -rz, ry], [rz, 0, -rx], [-ry, rx, 0]])
        J_lin_sensor = J_lin_origin - skew_r @ J_ang
        return rot, J_lin_sensor, current_pos_sensor

    def _gravity_scaling(self, rotation_ws: np.ndarray) -> np.ndarray:
        def axis_gain(axis):
            axis_unit = axis / max(np.linalg.norm(axis), 1e-9)
            cos_theta = np.clip(abs(float(axis_unit.dot(self._gravity_world))), 0.0, 1.0)
            theta = float(np.arccos(cos_theta))
            return float(np.clip(1.0 - (2.0 / np.pi) * theta, 0.0, 1.0))
        return np.array([axis_gain(rotation_ws[:, 0]), axis_gain(rotation_ws[:, 1])], dtype=float)

    def _solve_ik_underactuated(self, J_full: np.ndarray, dx: np.ndarray):
        J_active = J_full[:, self.pin_active_indices] 
        dls_lambda = 0.05
        try:
            inv_term = np.linalg.inv(J_active @ J_active.T + dls_lambda**2 * np.eye(3))
            dq_active = J_active.T @ inv_term @ dx 
        except np.linalg.LinAlgError:
            return np.zeros(len(JOINT_ORDER_FULL))
        dq_full = np.zeros(len(JOINT_ORDER_FULL))
        for i, val in enumerate(dq_active):
            user_idx = self.user_active_indices[i]
            dq_full[user_idx] = val
            if user_idx in [4, 7, 10, 13]: 
                if user_idx + 1 < len(dq_full): dq_full[user_idx+1] = val * 0.8
                if user_idx + 2 < len(dq_full): dq_full[user_idx+2] = val * 0.6
        return dq_full

    def _apply_and_send(self, total_dq, limit_val):
        total_dq = np.clip(total_dq, -limit_val, limit_val)
        target_q_16 = np.rad2deg(self.current_q + total_dq) 
        try:
            target_actuations = target_q_16[self.user_active_indices]
            self.hand.set_joint_positions(target_actuations.tolist())
        except Exception: pass

    # ================= 5. 外部控制接口 =================
    def set_command(self, cmd_str):
        """主程序调用此函数来改变手的状态"""
        with self.lock:
            if cmd_str == "PRE_SHAPE":
                self.current_state = GraspState.PRE_SHAPE
                logger.info("[Hand] 状态切换 -> PRE_SHAPE (调整拇指)")
            elif cmd_str == "GRASP":
                self.current_state = GraspState.ACTIVE_GRASP
                logger.info("[Hand] 状态切换 -> ACTIVE_GRASP (闭合控制)")
            elif cmd_str == "RELEASE":
                self.current_state = GraspState.RELEASING
                logger.info("[Hand] 状态切换 -> RELEASING (松开)")
            elif cmd_str == "IDLE":
                self.current_state = GraspState.IDLE
                logger.info("[Hand] 状态切换 -> IDLE")

    def stop_thread(self):
        self.running = False

    # ================= 6. 主循环 (移植逻辑核心) =================
    def run(self):
        self.running = True
        logger.info("Hand Control Thread Started.")
        
        while self.running:
            start_time = time.time()
            
            # --- 读取硬件 ---
            try:
                joint_deg_7 = self.hand.get_joint_positions_compact()
                if joint_deg_7 is None: continue
                self.current_q = self._map_7_to_16(joint_deg_7)
                tactile_data = self.tactile.get_latest_data()
            except Exception: continue
            
            # 获取当前状态 (线程安全)
            with self.lock:
                state = self.current_state

            # --- 状态机逻辑 ---
            
            # 状态: IDLE
            if state == GraspState.IDLE:
                time.sleep(0.05)
                continue
            
            # 状态: RELEASING (原代码1没有，沿用代码2的简单逻辑)
            if state == GraspState.RELEASING:
                target_q_7 = [0.0] * 7
                target_q_7[0] = 1.65 # 保持拇指张开
                try:
                    self.hand.set_joint_positions(target_q_7)
                except: pass
                time.sleep(0.02)
                continue

            total_dq = np.zeros(len(self.current_q)) # 初始化 dq

            # 状态: PRE_SHAPE (逻辑来自代码1)
            if state == GraspState.PRE_SHAPE:
                if self.thumb_abd_index is None:
                    # 自动切换到 ACTIVE (如果没法调拇指)
                    with self.lock: self.current_state = GraspState.ACTIVE_GRASP
                else:
                    current_val = self.current_q[self.thumb_abd_index]
                    target_val = self.params['thumb_abd_target']
                    diff = target_val - current_val
                    step_move = np.clip(diff, -self.params['approach_thumb_abd_speed'], self.params['approach_thumb_abd_speed'])
                    total_dq[self.thumb_abd_index] += step_move
                
                self._apply_and_send(total_dq, 0.5)
                # 循环结束，等待下一次 tick
                time.sleep(max(0, self.dt - (time.time() - start_time)))
                continue

            # 状态: ACTIVE_GRASP (核心逻辑，完全来自代码1)
            if state == GraspState.ACTIVE_GRASP:
                all_finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
                ADM_DIRECTION = -1.0 
                
                for finger_name in all_finger_names:
                    related_sensors = [s for s in SENSOR_TO_LINK_MAP.keys() if finger_name in s]
                    finger_in_contact = False
                    active_sensor_entries = []
                    
                    # --- 4.1 遍历传感器判断接触 ---
                    for s_name in related_sensors:
                        if s_name not in tactile_data: continue
                        module = self.modules[s_name]
                        
                        # 运动学更新
                        rot, J, pos = self._update_kinematics(self.current_q, module['link_name'])
                        gravity_scale = self._gravity_scaling(rot) # 这一步很关键，原代码2没有
                        
                        # 信号处理
                        raw = tactile_data[s_name]
                        features = module['processor'].process(np.array([raw['fx'], raw['fy'], raw['fz']]), self.dt)
                        features['shear_rate'][:2] *= gravity_scale

                        # 映射修正
                        f_filtered_sensor = features['force_filtered']
                        f_control_link = np.zeros(3)
                        f_control_link[0] = f_filtered_sensor[0] 
                        f_control_link[1] = f_filtered_sensor[2] # 正向
                        f_control_link[2] = f_filtered_sensor[1] 

                        # 迟滞判断
                        is_touching = False
                        if features['normal_force'] > self.params['contact_thresh']: is_touching = True
                        elif module['contact_active'] and features['normal_force'] > 5: is_touching = True
                        
                        if is_touching:
                            finger_in_contact = True
                            module['contact_active'] = True
                            active_sensor_entries.append({
                                "name": s_name, "features": features, "f_control_link": f_control_link,
                                "module": module, "rotation": rot, "jacobian": J, "pos": pos
                            })
                        else:
                            module['contact_active'] = False
                            module['has_anchored'] = False 
                            module['pid'].reset()
                            module['admittance'].reset() 

                    # --- 4.2 独立控制决策 ---
                    
                    # [分支 A] 未接触：速度控制 (Open Loop)
                    if not finger_in_contact:
                        indices = self.finger_map.get(finger_name, [])
                        speed = self.params['approach_thumb_speed'] if finger_name == "Thumb" else self.params['approach_other_speed']
                        for idx in indices: total_dq[idx] += speed 
                        
                    # [分支 B] 已接触：锚定导纳控制 (Closed Loop)
                    else:
                        tangential_forces = np.array([e["features"]["tangential_force"] + 1e-3 for e in active_sensor_entries])
                        weights = tangential_forces / tangential_forces.sum()
                        finger_dq_accum = np.zeros(len(self.current_q))
                        
                        for weight, entry in zip(weights, active_sensor_entries):
                            features = entry["features"]
                            module = entry["module"]
                            J = entry["jacobian"]
                            
                            # 1. 锚定
                            if not module.get('has_anchored', False):
                                module['admittance'].reset()
                                module['has_anchored'] = True
                                logger.debug(f"[{entry['name']}] ANCHOR SET")
                            
                            # 2. 目标力计算
                            SENSOR_NOISE_FLOOR = 0.2  
                            f_min_safe = max(self.params['min_force'], SENSOR_NOISE_FLOOR)
                            f_physics_need = (features['tangential_force'] / self.params['mu_est']) * self.params['safety_margin']                  
                            f_target = max(f_min_safe, f_physics_need)
                            
                            if features['slip_ratio'] > self.params['mu_est']:
                                f_target += (features['slip_ratio'] - self.params['mu_est']) * 5.0
                            
                            # PID
                            shear_err = -features['shear_rate'][:2]
                            f_tangent = module['pid'].update(shear_err, self.dt)

                            # 3. 构造指令
                            f_cmd_local = np.array([f_tangent[0], f_target, f_tangent[1]])
                            rot_np = np.array(entry["rotation"], copy=False)
                            f_meas_world = rot_np @ entry["f_control_link"]
                            f_cmd_world = rot_np @ f_cmd_local
                            
                            force_error = f_cmd_world - f_meas_world
                            if np.linalg.norm(force_error) < 0.02: force_error[:] = 0.0
                            
                            # 4. 导纳计算
                            module['admittance'].update(force_error)
                            dx = module['admittance'].vel_error * self.dt * ADM_DIRECTION
                            
                            # 5. 防回退棘轮
                            SAFE_LIMIT = 8.0 
                            curr_force = features['normal_force']
                            vel_y = module['admittance'].vel_error[1] * ADM_DIRECTION
                            
                            if vel_y < 0 and curr_force < SAFE_LIMIT:
                                    dx[:] = 0.0

                            # 6. IK求解
                            dq_sensor = self._solve_ik_underactuated(J, dx)
                            finger_dq_accum += weight * dq_sensor
                        
                        total_dq += finger_dq_accum

                # 发送指令
                self._apply_and_send(total_dq, self.params['limit_speed'])

            # 频率控制
            time.sleep(max(0, self.dt - (time.time() - start_time)))

if __name__ == "__main__":
    # 测试代码
    hand_ctrl = ShearGraspController()
    hand_ctrl.start() # 启动线程
    
    try:
        print("指令列表: PRE_SHAPE, GRASP, RELEASE, IDLE")
        while True:
            cmd = input("请输入指令: ").strip()
            if cmd in ["PRE_SHAPE", "GRASP", "RELEASE", "IDLE"]:
                hand_ctrl.set_command(cmd)
            elif cmd == "exit":
                break
            else:
                print("无效指令")
    except KeyboardInterrupt:
        hand_ctrl.hand.set_actuations([0.0]*7)
        pass
    
    hand_ctrl.stop_thread()
    print("退出程序")