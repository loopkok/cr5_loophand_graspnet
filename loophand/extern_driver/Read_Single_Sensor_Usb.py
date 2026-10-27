import serial
import time
import struct
import threading
import logging
import numpy as np
import serial.tools.list_ports
from typing import Dict, List, Optional

class PalmTactileSensor:
    def __init__(self, port: Optional[str] = None, baudrate: int = 921600, sensor_id: int = 1):
        self.baudrate = baudrate
        self.sensor_id = sensor_id
        self.ser: Optional[serial.Serial] = None
        
        # 掌心传感器配置 (MC-M2020-Elite)
        self.NUM_POINTS = 9
        self.DATA_LEN = self.NUM_POINTS * 3  # 9个点 * 3维 = 27 字节
        self.REG_DISTRIBUTED = 1038          # 分布力起始地址 0x040E
        
        # 数据存储
        self.resultant_force = {"fx": 0.0, "fy": 0.0, "fz": 0.0}
        self.distributed_force: List[Dict[str, float]] = [{"fx": 0.0, "fy": 0.0, "fz": 0.0} for _ in range(self.NUM_POINTS)]
        
        self._offset_res = np.zeros(3)
        self._offset_dist = np.zeros((self.NUM_POINTS, 3))
        self._is_tared = False
        
        # 线程控制
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        
        self.RESP_HEAD = b"\xaa\x55"
        # 帧长度 = 头(2) + 长度(2) + 地址(1) + 预留(1) + 功能码(1) + 寄存器(4) + 返回字节数(2) + 状态(1) + 数据(27) + LRC(1) = 42字节
        self.EXPECTED_FRAME_LEN = 15 + self.DATA_LEN 
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger("PalmSensor")

        self.port = port or self._find_available_port()
        if not self.port:
            raise RuntimeError("未检测到可用串口，请检查 USB 连接。")

        self._connect_and_initialize()

    def _find_available_port(self) -> Optional[str]:
        ports = list(serial.tools.list_ports.comports())
        return ports[0].device if ports else None

    def _calc_lrc(self, data: bytes) -> int:
        """LRC 计算：累加所有传入字节，取反加一"""
        lrc = sum(data) & 0xFF
        return ((~lrc) + 1) & 0xFF

    def _build_read_frame(self, reg_addr: int, read_len: int) -> bytes:
        """动态构建符合协议的串口读请求帧"""
        # 数据域: Length(2 bytes) + Addr(1) + Res(1) + Func(1) + RegAddr(4) + ReadLen(2) = 11 bytes
        body = struct.pack('<HBBBIH', 9, self.sensor_id, 0x00, 0xFB, reg_addr, read_len)
        
        # 【核心修复】：LRC 校验必须包含帧头 \x55\xaa
        frame_without_lrc = b'\x55\xaa' + body
        lrc = self._calc_lrc(frame_without_lrc)
        
        return frame_without_lrc + bytes([lrc])

    def _connect_and_initialize(self):
        try:
            self.ser = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=0.1)
            self.ser.reset_input_buffer()
            
            # 构建并缓存读取分布力的指令 (读取27字节)
            self.cmd_get_distributed = self._build_read_frame(self.REG_DISTRIBUTED, self.DATA_LEN)
            
            self._running = True
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
            self.logger.info("掌心传感器 (9点阵列) 初始化完成，开始接收流数据...")

        except Exception as e:
            self.logger.error(f"初始化失败: {e}")
            raise

    def tare(self, duration: float = 1.0):
        """重新标定零点 (去皮)"""
        self.logger.info("开始去皮，请勿触碰传感器...")
        start_time = time.time()
        samples_dist = []
        
        with self._lock:
            self._is_tared = False 
            self._offset_res = np.zeros(3)
            self._offset_dist = np.zeros((self.NUM_POINTS, 3))
        
        while time.time() - start_time < duration:
            time.sleep(0.05)
            with self._lock:
                current_dist = [[pt['fx'], pt['fy'], pt['fz']] for pt in self.distributed_force]
                samples_dist.append(current_dist)

        with self._lock:
            if samples_dist:
                arr = np.array(samples_dist)
                mean_dist = np.mean(arr, axis=0) 
                self._offset_dist = mean_dist
                self._offset_res = np.sum(mean_dist, axis=0) # 合力偏置 = 9个点偏置之和
            self._is_tared = True
        self.logger.info("去皮完成。后续数据将自动扣除偏置。")

    def get_latest_data(self) -> Dict:
        """返回合力与分布力阵列"""
        with self._lock:
            return {
                "Resultant": self.resultant_force.copy(),
                "Distributed": [pt.copy() for pt in self.distributed_force]
            }

    def close(self):
        self._running = False
        if self._thread: self._thread.join(timeout=1.0)
        if self.ser: self.ser.close()

    def _read_loop(self):
        buffer = b""
        while self._running and self.ser.is_open:
            try:
                # 持续发送查询指令
                self.ser.write(self.cmd_get_distributed)
                
                if self.ser.in_waiting:
                    buffer += self.ser.read(self.ser.in_waiting)
                    
                    while self.RESP_HEAD in buffer:
                        head_idx = buffer.find(self.RESP_HEAD)
                        if head_idx > 0: buffer = buffer[head_idx:]
                            
                        if len(buffer) >= self.EXPECTED_FRAME_LEN:
                            frame = buffer[:self.EXPECTED_FRAME_LEN]
                            
                            # 解析帧长度字段
                            resp_len = int.from_bytes(frame[2:4], byteorder='little')
                            lrc_pos = 4 + resp_len
                            
                            # 【核心修复】：验证端 LRC 也必须包含头部的 AA 55
                            if lrc_pos < len(frame):
                                calc_lrc = self._calc_lrc(frame[:lrc_pos])
                                if calc_lrc == frame[lrc_pos]:
                                    self._parse_packet(frame)
                                else:
                                    pass # 可在此添加 logger.debug("LRC不匹配") 记录误码
                                
                            buffer = buffer[self.EXPECTED_FRAME_LEN:]
                        else:
                            break
                time.sleep(0.01) # 控制约 100Hz 轮询
            except Exception as e:
                time.sleep(0.1)

    def _parse_packet(self, frame: bytes):
        try:
            # 协议规定：第13字节是状态码(0x00)，真实数据从第14字节开始
            data_start = 14
            data_bytes = frame[data_start:data_start + self.DATA_LEN]
            
            temp_dist = []
            sum_fx, sum_fy, sum_fz = 0.0, 0.0, 0.0
            
            # 解析 9 个测点
            for i in range(self.NUM_POINTS):
                pt_bytes = data_bytes[i*3 : i*3+3]
                
                # Fx, Fy 为有符号('b'), Fz 为无符号('B')
                raw_fx, raw_fy, raw_fz = struct.unpack('bbB', pt_bytes)
                
                real_fx = raw_fx - self._offset_dist[i][0]
                real_fy = raw_fy - self._offset_dist[i][1]
                real_fz = raw_fz - self._offset_dist[i][2]
                
                # 滤除底噪
                if abs(real_fx) < 1.5: real_fx = 0.0
                if abs(real_fy) < 1.5: real_fy = 0.0
                if abs(real_fz) < 1.5: real_fz = 0.0
                
                temp_dist.append({"fx": real_fx, "fy": real_fy, "fz": real_fz})
                
                sum_fx += real_fx
                sum_fy += real_fy
                sum_fz += real_fz

            with self._lock:
                self.distributed_force = temp_dist
                self.resultant_force = {"fx": sum_fx, "fy": sum_fy, "fz": sum_fz}
                
        except Exception as e:
            pass

# if __name__ == "__main__":
#     sensor_left = PalmTactileSensor('/dev/ttyACM0',sensor_id=1)
#     sensor_right = PalmTactileSensor('/dev/ttyACM1',sensor_id=1)
#     time.sleep(1.0) # 等待流稳定
#     sensor_left.tare(duration=1.5)
#     sensor_right.tare(duration=1.5)
    
#     try:
#         while True:
#             data_left = sensor_left.get_latest_data()
#             data_right = sensor_right.get_latest_data()
#             res_left = data_left["Resultant"]
#             res_right = data_right["Resultant"]
#             # 打印自己加和出来的 3D 合力
#             print(
#                 f"\r[左手] Fx: {res_left['fx']:5.1f} | Fy: {res_left['fy']:5.1f} | Fz: {res_left['fz']:5.1f}    "
#                 f"||    [右手] Fx: {res_right['fx']:5.1f} | Fy: {res_right['fy']:5.1f} | Fz: {res_right['fz']:5.1f}   ", 
#                 end="", flush=True
#             )
#             time.sleep(0.05)
#     except KeyboardInterrupt:
#         print("\n程序结束。")
#     finally:
#         if 'sensor_left' in locals() and sensor_left:
#             sensor_left.close()
#         if 'sensor_right' in locals() and sensor_right:
#             sensor_right.close()
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt

#     print("正在连接左右手掌心传感器...")
#     sensor_left = PalmTactileSensor('/dev/ttyACM0', sensor_id=1)
#     sensor_right = PalmTactileSensor('/dev/ttyACM1', sensor_id=1)
    
#     time.sleep(1.0) # 等待流稳定
    
#     print("准备去皮...")
#     sensor_left.tare(duration=1.5)
#     sensor_right.tare(duration=1.5)
    
#     # ==========================================
#     # 初始化 Matplotlib 实时可视化面板
#     # ==========================================
#     plt.ion()  # 开启交互模式，程序不会被画图阻塞
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
#     # 初始全 0 的 3x3 矩阵
#     dummy_matrix = np.zeros((3, 3))
    
#     # 配置左手画布 (带平滑，vmin和vmax控制颜色映射的力度范围，你可以根据实际按压力度调整 vmax)
#     im_left = ax1.imshow(dummy_matrix, cmap='jet', interpolation='bicubic', vmin=0, vmax=20)
#     ax1.set_title("Left Palm (Smoothed)")
#     fig.colorbar(im_left, ax=ax1, label="Force Fz (N)")
    
#     # 配置右手画布
#     im_right = ax2.imshow(dummy_matrix, cmap='jet', interpolation='bicubic', vmin=0, vmax=20)
#     ax2.set_title("Right Palm (Smoothed)")
#     fig.colorbar(im_right, ax=ax2, label="Force Fz (N)")
    
#     plt.tight_layout()

#     print("\n===== 开始持续读取并可视化双臂数据 =====")
#     print("请在弹出的图形窗口中观察触觉热力图。按 Ctrl+C 终止程序。")
    
#     try:
#         while True:
#             # 1. 获取最新数据
#             data_left = sensor_left.get_latest_data()
#             data_right = sensor_right.get_latest_data()
            
#             # 2. 提取 9 个点的 Fz，并重塑为 3x3 矩阵
#             fz_matrix_left = np.array([pt["fz"] for pt in data_left["Distributed"]]).reshape(3, 3)
#             fz_matrix_right = np.array([pt["fz"] for pt in data_right["Distributed"]]).reshape(3, 3)
            
#             # 3. 【高性能刷新核心】只更新像素数据，不重建坐标轴
#             im_left.set_data(fz_matrix_left)
#             im_right.set_data(fz_matrix_right)
            
#             # 4. 刷新 GUI 事件
#             fig.canvas.draw()
#             fig.canvas.flush_events()
            
#             # 5. 终端保持合力数值打印
#             res_left = data_left["Resultant"]
#             res_right = data_right["Resultant"]
#             print(
#                 f"\r[左手] Fx: {res_left['fx']:5.1f} | Fy: {res_left['fy']:5.1f} | Fz: {res_left['fz']:5.1f}    "
#                 f"||    [右手] Fx: {res_right['fx']:5.1f} | Fy: {res_right['fy']:5.1f} | Fz: {res_right['fz']:5.1f}   ", 
#                 end="", flush=True
#             )
            
#             # 控制更新帧率 (约 20FPS)，过快会导致 matplotlib 卡顿
#             time.sleep(0.05)
            
#     except KeyboardInterrupt:
#         print("\n\n用户终止程序。")
#     finally:
#         # 安全释放资源
#         sensor_left.close()
#         sensor_right.close()
#         plt.ioff() # 关闭交互模式
#         plt.close() # 关闭画板

if __name__ == "__main__":
    import cv2
    import numpy as np
    import time

    print("正在连接左右手掌心传感器...")
    # 假设都在同一台机器的两个不同USB口，默认模组号均为1
    sensor_left = PalmTactileSensor('/dev/ttyACM0', sensor_id=1)
    sensor_right = PalmTactileSensor('/dev/ttyACM1', sensor_id=1)
    
    time.sleep(1.0) # 等待底层数据流稳定
    
    print("准备去皮...")
    sensor_left.tare(duration=1.5)
    sensor_right.tare(duration=1.5)
    
    print("\n===== 开始持续读取双臂数据 (OpenCV 高频渲染) =====")
    print("在弹出的 OpenCV 窗口中按 'q' 键或 'Esc' 键退出程序。")
    
    # 可视化参数配置
    MAX_FORCE = 20.0  # 假设最大下压力为 20N，超过此值显示最亮（黄色）
    DISPLAY_SIZE = 300 # 将 3x3 的矩阵放大到 300x300 像素显示

    def matrix_to_heatmap(fz_matrix, max_val, size):
        """将 3x3 的物理受力矩阵转换为 OpenCV 热力图特征图像"""
        # 1. 截断最大值，并归一化到 0 ~ 255 的图像像素区间
        norm_matrix = np.clip(fz_matrix, 0, max_val) / max_val * 255.0
        norm_matrix = norm_matrix.astype(np.uint8)
        
        # 2. 应用伪彩色映射：COLORMAP_VIRIDIS 最符合深度学习 Feature Map 风格 (暗紫 -> 亮黄)
        heatmap = cv2.applyColorMap(norm_matrix, cv2.COLORMAP_VIRIDIS)
        
        # 3. 放大图像：INTER_NEAREST 保持马赛克/离散像素感 (CNN 真实输入的样子)
        # 如果你想看平滑效果，可以改成 cv2.INTER_CUBIC
        enlarged = cv2.resize(heatmap, (size, size), interpolation=cv2.INTER_NEAREST)
        return enlarged

    try:
        while True:
            # 获取最新数据
            data_left = sensor_left.get_latest_data()
            data_right = sensor_right.get_latest_data()
            
            # 提取 9 个点的 Fz 并重塑为 3x3 矩阵
            fz_matrix_left = np.array([pt["fz"] for pt in data_left["Distributed"]]).reshape(3, 3)
            fz_matrix_right = np.array([pt["fz"] for pt in data_right["Distributed"]]).reshape(3, 3)
            
            # 生成 OpenCV 热力图
            img_left = matrix_to_heatmap(fz_matrix_left, MAX_FORCE, DISPLAY_SIZE)
            img_right = matrix_to_heatmap(fz_matrix_right, MAX_FORCE, DISPLAY_SIZE)
            
            # 为图像添加文字标签 (可选，提升监控体验)
            cv2.putText(img_left, "Left Palm (3x3)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img_right, "Right Palm (3x3)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # 将左右手图像水平拼接在一起 (hstack)
            combined_img = np.hstack((img_left, img_right))
            
            # 高性能显示
            cv2.imshow("Tactile Spatial Feature Map", combined_img)
            
            # 终端保持合力数值打印
            res_left = data_left["Resultant"]
            res_right = data_right["Resultant"]
            print(
                f"\r[左手] Fx: {res_left['fx']:5.1f} | Fy: {res_left['fy']:5.1f} | Fz: {res_left['fz']:5.1f}    "
                f"||    [右手] Fx: {res_right['fx']:5.1f} | Fy: {res_right['fy']:5.1f} | Fz: {res_right['fz']:5.1f}   ", 
                end="", flush=True
            )
            
            # waitKey(1) 是 OpenCV 刷新的核心，1ms 的延迟满足 100Hz 以上的超高帧率
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # 按 Esc 或 q 退出
                break
                
            # 控制整体轮询帧率，避免死循环榨干 CPU（0.02s 即 50Hz 刷新率）
            time.sleep(0.02)
            
    except KeyboardInterrupt:
        print("\n\n用户终止程序。")
    finally:
        # 安全释放所有资源
        if 'sensor_left' in locals() and sensor_left:
            sensor_left.close()
        if 'sensor_right' in locals() and sensor_right:
            sensor_right.close()
        cv2.destroyAllWindows()