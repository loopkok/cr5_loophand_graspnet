# force_plotter.py
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


class ForceDataPlotter:
    """独立的外部力数据绘图器，不修改原有代码"""

    def __init__(self, robot, max_data_points=200, update_interval=0.1):
        """
        初始化绘图器

        Args:
            robot: CR5机器人实例
            max_data_points: 最大数据点数
            update_interval: 数据更新间隔(秒)
        """
        self.robot = robot
        self.max_data_points = max_data_points
        self.update_interval = update_interval

        # 数据存储
        self.time_data = deque(maxlen=max_data_points)
        self.force_x_data = deque(maxlen=max_data_points)
        self.force_y_data = deque(maxlen=max_data_points)
        self.force_z_data = deque(maxlen=max_data_points)
        self.force_mag_data = deque(maxlen=max_data_points)

        # 线程控制
        self.is_running = False
        self.plot_thread = None

        # 初始化时间基准
        self.start_time = time.time()

        # 初始化图表
        self.setup_plot()

    def setup_plot(self):
        """设置绘图窗口"""
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # 力分量图
        self.ax1.set_title('实时力传感器数据 - 分量')
        self.ax1.set_ylabel('力 (N)')
        self.ax1.grid(True)
        self.line_x, = self.ax1.plot([], [], 'r-', label='Fx', linewidth=2)
        self.line_y, = self.ax1.plot([], [], 'g-', label='Fy', linewidth=2)
        self.line_z, = self.ax1.plot([], [], 'b-', label='Fz', linewidth=2)
        self.ax1.legend()

        # 力大小图
        self.ax2.set_title('实时力传感器数据 - 大小')
        self.ax2.set_ylabel('力大小 (N)')
        self.ax2.set_xlabel('时间 (s)')
        self.ax2.grid(True)
        self.line_mag, = self.ax2.plot([], [], 'k-', label='|F|', linewidth=2)
        self.ax2.legend()

        plt.tight_layout()
        plt.ion()  # 交互模式
        plt.show()

    def update_data(self):
        """更新力数据"""
        try:
            raw_force = self.robot.get_TCPForce()[0:3]
            current_time = time.time() - self.start_time
            force_magnitude = np.linalg.norm(raw_force)

            self.time_data.append(current_time)
            self.force_x_data.append(raw_force[0])
            self.force_y_data.append(raw_force[1])
            self.force_z_data.append(raw_force[2])
            self.force_mag_data.append(force_magnitude)

            return True
        except Exception as e:
            print(f"获取力数据失败: {e}")
            return False

    def update_plot(self):
        """更新图表"""
        if len(self.time_data) == 0:
            return

        # 更新力分量图
        self.line_x.set_data(self.time_data, self.force_x_data)
        self.line_y.set_data(self.time_data, self.force_y_data)
        self.line_z.set_data(self.time_data, self.force_z_data)

        # 更新力大小图
        self.line_mag.set_data(self.time_data, self.force_mag_data)

        # 自动调整坐标轴
        if len(self.time_data) > 1:
            time_range = max(1, self.time_data[-1] - self.time_data[0])

            # X轴范围
            self.ax1.set_xlim(self.time_data[0], self.time_data[0] + time_range)
            self.ax2.set_xlim(self.time_data[0], self.time_data[0] + time_range)

            # Y轴范围 - 动态调整
            all_forces = list(self.force_x_data) + list(self.force_y_data) + list(self.force_z_data)
            if all_forces:
                force_min = min(all_forces)
                force_max = max(all_forces)
                margin = max(1, (force_max - force_min) * 0.1)
                self.ax1.set_ylim(force_min - margin, force_max + margin)

            if self.force_mag_data:
                mag_max = max(self.force_mag_data)
                self.ax2.set_ylim(0, mag_max * 1.1)

        # 更新统计信息
        if len(self.force_mag_data) > 10:
            recent_mag = list(self.force_mag_data)[-10:]
            avg_force = np.mean(recent_mag)
            std_force = np.std(recent_mag)
            self.ax2.set_title(f'实时力传感器数据 - 大小 (最近10点: 平均={avg_force:.2f}N, 噪声={std_force:.2f}N)')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _plot_loop(self):
        """绘图循环"""
        while self.is_running:
            if self.update_data():
                self.update_plot()
            time.sleep(self.update_interval)

    def start(self):
        """开始绘图"""
        if self.is_running:
            print("绘图器已经在运行")
            return

        self.is_running = True
        self.plot_thread = threading.Thread(target=self._plot_loop)
        self.plot_thread.daemon = True
        self.plot_thread.start()
        print("力数据绘图器已启动")

    def stop(self):
        """停止绘图"""
        self.is_running = False
        if self.plot_thread:
            self.plot_thread.join(timeout=1.0)
        plt.close(self.fig)
        print("力数据绘图器已停止")

    def get_statistics(self):
        """获取统计信息"""
        if len(self.time_data) == 0:
            return None

        stats = {
            'data_points': len(self.time_data),
            'force_x_range': (min(self.force_x_data), max(self.force_x_data)),
            'force_y_range': (min(self.force_y_data), max(self.force_y_data)),
            'force_z_range': (min(self.force_z_data), max(self.force_z_data)),
            'force_mag_range': (min(self.force_mag_data), max(self.force_mag_data)),
            'noise_level': (
                np.std(self.force_x_data),
                np.std(self.force_y_data),
                np.std(self.force_z_data)
            )
        }
        return stats