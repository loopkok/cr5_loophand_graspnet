import os
import sys
import time
import logging
import threading
import numpy as np
import pinocchio as pin
from collections import deque
from enum import Enum

# 本文件位于 loophand/examples/，把 loophand/ 加入模块搜索路径
_LOOPHAND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _LOOPHAND_ROOT not in sys.path:
    sys.path.insert(0, _LOOPHAND_ROOT)

# 引入硬件驱动
from aero_open_sdk.aero_hand import AeroHand
from extern_driver.tactile_driver import TactileSensor
from controller.pid import BoundedPID
from controller.admittancecontroller import AdmittanceController

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ShearControl")

# ================= 配置常量  =================
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
    IDLE = -1          # 新增: 待机/空闲
    PRE_SHAPE = 0      # 预姿态 (调整拇指)
    ACTIVE_GRASP = 1   # 接触并自适应抓取
    RELEASING = 2      # 新增: 松开物体

# ================= 信号处理类 (保持不变) =================
class SignalProcessor:
    def __init__(self, alpha=0.4, history_len=5, derivative_alpha=0.1):
        self.alpha = alpha
        self.derivative_alpha = derivative_alpha
        self.history = deque(maxlen=history_len)
        self.lowpass = np.zeros(3)
        self.shear_prev = np.zeros(3)
        self.epsilon = 1e-3
        
    def process(self, raw_force: np.ndarray, dt: float):
        self.lowpass = self.alpha * raw_force + (1.0 - self.alpha) * self.lowpass
        self.history.append(self.lowpass.copy())
        if len(self.history) >= 2:
            shear_delta = (self.lowpass - self.history[0]) / max(dt * (len(self.history) - 1), self.epsilon)
        else:
            shear_delta = np.zeros(3)
        self.shear_prev = (self.derivative_alpha * shear_delta + (1.0 - self.derivative_alpha) * self.shear_prev)
        return {
            "force_filtered": self.lowpass.copy(),
            "shear_rate": self.shear_prev.copy(),
            "normal_force": self.lowpass[2],
            "tangential_force": np.linalg.norm(self.lowpass[:2]),
            "slip_ratio": np.linalg.norm(self.lowpass[:2]) / max(abs(self.lowpass[2]), self.epsilon)
        }

# ================= 触觉手控制器主类 =================
class ShearGraspController(threading.Thread):
    def __init__(self, urdf_path=os.path.join(_LOOPHAND_ROOT, "urdf", "aero_hand_open_left.urdf"), port_hand="/dev/ttyACM1", port_tactile="/dev/ttyACM0"):
        super().__init__()
        self.daemon = True  # 设置为守护线程，主程序退出时自动退出
        
        logger.info("Initializing Hand Hardware & Kinematics...")
        self.hand = AeroHand(port=port_hand) 
        self.tactile = TactileSensor(port=port_tactile)
        time.sleep(1.0)
        self.tactile.tare(duration=3.0)
        
        # Pinocchio 模型初始化
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        
        # 索引映射初始化
        self.pin_active_indices = []
        self.user_active_indices = []
        self.finger_map = {"Thumb": [], "Index": [], "Middle": [], "Ring": [], "Pinky": []}
        self.thumb_abd_index = None
        
        self._init_indices() # 辅助函数初始化索引
        
        # 控制参数
        self.dt = 0.01
        self._gravity_world = np.array([0.0, -1.0, 0.0])
        self.running = False
        self.current_state = GraspState.IDLE
        self.lock = threading.Lock()
        
        # 模块初始化 (PID + Admittance)
        self.modules = {}
        PID_PARAMS = (0.1, 0.0, 0.0, 5.0) 
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
        
        self.params = {
            "mu_est": 0.4, "safety_margin": 1.5, "min_force": 8.0,        
            "approach_thumb_abd_speed": 0.03, "approach_thumb_speed": 0.01,
            "approach_other_speed": 0.03, "contact_thresh": 0.1,  
            "thumb_abd_target": 1.65, "pre_shape_tol": 0.05, "limit_speed": 0.06,
        }

    def _init_indices(self):
        for name in ACTIVE_JOINT_NAMES:
            if self.model.existJointName(name):
                self.pin_active_indices.append(self.model.idx_vs[self.model.getJointId(name)])
            if name in JOINT_ORDER_FULL:
                self.user_active_indices.append(JOINT_ORDER_FULL.index(name))
        
        for idx, name in enumerate(JOINT_ORDER_FULL):
            if "abd" in name and "thumb" in name: self.thumb_abd_index = idx
            elif "thumb" in name: self.finger_map["Thumb"].append(idx)
            elif "index" in name: self.finger_map["Index"].append(idx)
            elif "middle" in name: self.finger_map["Middle"].append(idx)
            elif "ring" in name:   self.finger_map["Ring"].append(idx)
            elif "pinky" in name:  self.finger_map["Pinky"].append(idx)

    # --- 外部调用接口 ---
    def set_command(self, cmd_str):
        """主程序调用此函数来改变手的状态"""
        with self.lock:
            if cmd_str == "PRE_SHAPE":
                self.current_state = GraspState.PRE_SHAPE
                logger.info("[Hand] 状态切换 -> 预姿态 (调整拇指)")
            elif cmd_str == "GRASP":
                self.current_state = GraspState.ACTIVE_GRASP
                logger.info("[Hand] 状态切换 -> 自适应抓取 (闭合)")
            elif cmd_str == "RELEASE":
                self.current_state = GraspState.RELEASING
                logger.info("[Hand] 状态切换 -> 松开")
            elif cmd_str == "IDLE":
                self.current_state = GraspState.IDLE

    def stop_thread(self):
        self.running = False

    # --- 内部逻辑函数 (保持你的核心算法) ---
    def _map_7_to_16(self, compact_deg):
        # 你的原始映射逻辑
        q_7 = np.deg2rad(np.array(compact_deg))
        q_16 = np.zeros(len(JOINT_ORDER_FULL))
        q_16[0:3] = q_7[0:3]; q_16[3] = q_7[2] * 1.0 
        for i, start_idx in enumerate([4, 7, 10, 13]):
            val = q_7[3+i]
            q_16[start_idx:start_idx+3] = [val, val*0.8, val*0.6]
        return q_16

    def _update_kinematics(self, q_user, link_name):
        q_pin = pin.neutral(self.model)
        for user_i, name in enumerate(JOINT_ORDER_FULL):
            if self.model.existJointName(name):
                q_pin[self.model.idx_qs[self.model.getJointId(name)]] = q_user[user_i]
        
        pin.forwardKinematics(self.model, self.data, q_pin)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, q_pin)
        
        frame_id = self.model.getFrameId(link_name)
        J_spatial = pin.getFrameJacobian(self.model, self.data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        
        J_lin = J_spatial[:3, :]
        J_ang = J_spatial[3:, :]
        rot = self.data.oMf[frame_id].rotation
        current_pos = self.data.oMf[frame_id].translation
        
        offset = np.array([0.0, 0.0, 0.025]) if "proximal" in link_name else np.zeros(3)
        r_world = rot @ offset
        
        rx, ry, rz = r_world
        skew_r = np.array([
            [0, -rz, ry],
            [rz, 0, -rx],
            [-ry, rx, 0]
        ])
        
        J_lin_sensor = J_lin - skew_r @ J_ang
        return rot, J_lin_sensor, current_pos + r_world

    def _solve_ik(self, J_full, dx):
        J_active = J_full[:, self.pin_active_indices]
        try:
            dq_active = J_active.T @ np.linalg.inv(J_active @ J_active.T + 0.0025 * np.eye(3)) @ dx
        except: return np.zeros(len(JOINT_ORDER_FULL))
        
        dq_full = np.zeros(len(JOINT_ORDER_FULL))
        for i, val in enumerate(dq_active):
            u_idx = self.user_active_indices[i]
            dq_full[u_idx] = val
            if u_idx in [4, 7, 10, 13]: # 欠驱动联动
                if u_idx+1 < 16: dq_full[u_idx+1] = val * 0.8
                if u_idx+2 < 16: dq_full[u_idx+2] = val * 0.6
        return dq_full

    def run(self):
        """线程主循环"""
        self.running = True
        logger.info("[Hand] 控制线程启动")
        
        while self.running:
            start_t = time.time()
            
            # 1. 硬件读取
            try:
                raw_q = self.hand.get_joint_positions_compact()
                if raw_q is None: continue
                self.current_q = self._map_7_to_16(raw_q)
                tactile_data = self.tactile.get_latest_data()
            except Exception: continue

            # 2. 状态机逻辑
            total_dq = np.zeros(len(self.current_q))
            
            with self.lock:
                state = self.current_state

            if state == GraspState.IDLE:
                time.sleep(0.05)
                continue

            elif state == GraspState.RELEASING:
                # 简单开环：所有电机归零或设为张开位置
                # 假设 0 度是张开
                target_q_7 = [0.0] * 7 
                # 稍微保留一点拇指位置
                target_q_7[0] = 1.5 # 拇指外展保持
                self.hand.set_joint_positions(target_q_7)
                time.sleep(0.02)
                continue

            elif state == GraspState.PRE_SHAPE:
                # 仅调整拇指外展
                if self.thumb_abd_index is not None:
                    curr = self.current_q[self.thumb_abd_index]
                    targ = self.params['thumb_abd_target']
                    step = np.clip(targ - curr, -0.03, 0.03)
                    total_dq[self.thumb_abd_index] += step
                    
                    # 同时让其他四指稍微张开一点 (负向运动)
                    for f in ["Index", "Middle", "Ring", "Pinky"]:
                         for idx in self.finger_map[f]:
                             # 如果当前弯曲了，就往回退一点
                             if self.current_q[idx] > 0.1:
                                 total_dq[idx] -= 0.02

            elif state == GraspState.ACTIVE_GRASP:
                # 你的核心核心导纳+PID逻辑
                for finger_name in ["Thumb", "Index", "Middle", "Ring", "Pinky"]:
                    sensors = [s for s in SENSOR_TO_LINK_MAP if finger_name in s]
                    in_contact = False
                    active_entries = []
                    
                    # 接触检测
                    for s_name in sensors:
                        if s_name not in tactile_data: continue
                        mod = self.modules[s_name]
                        raw_f = tactile_data[s_name]
                        feats = mod['processor'].process(np.array([raw_f['fx'], raw_f['fy'], raw_f['fz']]), self.dt)
                        
                        # 坐标变换
                        rot, J, _ = self._update_kinematics(self.current_q, mod['link_name'])
                        
                        # Z->Y 映射 (正向)
                        f_ctrl = np.array([feats['force_filtered'][0], feats['force_filtered'][2], feats['force_filtered'][1]])
                        
                        # 接触判断
                        if feats['normal_force'] > self.params['contact_thresh'] or (mod['contact_active'] and feats['normal_force']>5):
                            in_contact = True
                            mod['contact_active'] = True
                            active_entries.append({"mod": mod, "feat": feats, "J": J, "f_ctrl": f_ctrl, "rot": rot})
                        else:
                            mod['contact_active'] = False
                            mod['has_anchored'] = False
                            mod['pid'].reset(); mod['admittance'].reset()

                    # 控制分支
                    if not in_contact:
                        # 接近阶段: 匀速闭合
                        spd = self.params['approach_thumb_speed'] if finger_name == "Thumb" else self.params['approach_other_speed']
                        for idx in self.finger_map[finger_name]: total_dq[idx] += spd
                    else:
                        # 接触阶段: 导纳控制
                        dq_accum = np.zeros(16)
                        weights = [e['feat']['tangential_force']+1e-3 for e in active_entries]
                        w_sum = sum(weights)
                        
                        for w, entry in zip(weights, active_entries):
                            mod = entry['mod']
                            # 锚定逻辑
                            if not mod['has_anchored']:
                                mod['has_anchored'] = True; mod['admittance'].reset()
                            
                            # 力目标计算
                            f_min = max(self.params['min_force'], 0.2)
                            f_phys = (entry['feat']['tangential_force'] / self.params['mu_est']) * self.params['safety_margin']
                            f_curr = entry['feat']['normal_force']
                            f_target = max(f_min, f_phys)
                            if f_curr > f_target: f_target = f_curr * 0.99
                            if entry['feat']['slip_ratio'] > self.params['mu_est']:
                                f_target += (entry['feat']['slip_ratio'] - self.params['mu_est']) * 5.0
                                
                            # PID 抗剪切
                            f_tan = mod['pid'].update(-entry['feat']['shear_rate'][:2], self.dt)
                            
                            # 导纳输入误差
                            f_cmd_world = entry['rot'] @ np.array([f_tan[0], f_target, f_tan[1]])
                            f_meas_world = entry['rot'] @ entry['f_ctrl']
                            err = f_cmd_world - f_meas_world
                            if np.linalg.norm(err) < 0.02: err[:] = 0.0
                            
                            mod['admittance'].update(err)
                            
                            # 棘轮逻辑 (ADM_DIRECTION = -1.0)
                            ADM_DIR = -1.0
                            dx = mod['admittance'].vel_error * self.dt * ADM_DIR
                            if mod['admittance'].vel_error[1] * ADM_DIR < 0 and f_curr < 8.0:
                                dx[:] = 0.0 # 禁止在低力下回退
                            
                            dq_accum += (w/w_sum) * self._solve_ik(entry['J'], dx)
                        
                        total_dq += dq_accum

            # 3. 发送指令
            if state in [GraspState.PRE_SHAPE, GraspState.ACTIVE_GRASP]:
                limit = self.params['limit_speed']
                total_dq = np.clip(total_dq, -limit, limit)
                # 仅将变化量应用到当前位置
                targ_q_deg = np.rad2deg(self.current_q + total_dq)
                try:
                    self.hand.set_joint_positions(targ_q_deg[self.user_active_indices].tolist())
                except: pass

            # 频控制
            elapsed = time.time() - start_t
            time.sleep(max(0, self.dt - elapsed))

if __name__ == "__main__":
    # 测试代码
    hand = ShearGraspController()
    hand.start()
    try:
        input("按回车预姿态...")
        hand.set_command("PRE_SHAPE")
        input("按回车抓取...")
        hand.set_command("GRASP")
        input("按回车松开...")
        hand.set_command("RELEASE")
        time.sleep(2)
        hand.stop_thread()
    except KeyboardInterrupt:
        hand.stop_thread()
        hand.hand.set_actuations([0.0]*7)
        print("退出测试")