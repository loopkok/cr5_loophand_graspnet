import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from collections import deque
import numpy as np
import threading
import time

class RealTimePlotter:
    """
    通用实时绘图类
    可以同时绘制多个变量的实时变化曲线
    """
    
    def __init__(self, title="实时数据监控", window_size=500, update_interval=50):
        """
        初始化实时绘图器
        
        参数:
            title: 窗口标题
            window_size: 显示的数据点数量
            update_interval: 更新间隔(毫秒)
        """
        self.window_size = window_size
        self.update_interval = update_interval
        self.running = False
        self.data_lock = threading.Lock()
        
        # 存储数据: {变量名: deque数据}
        self.data_queues = {}
        # 存储时间戳
        self.time_queue = deque(maxlen=window_size)
        # 变量配置: {变量名: {'color':颜色, 'label':标签, 'visible':是否可见}}
        self.variable_configs = {}
        
        # 初始化图形界面
        self.setup_ui(title)
        
    def setup_ui(self, title):
        """设置用户界面"""
        # 创建主窗口
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("800x600")
        
        # 创建matplotlib图形
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_xlabel('time (s)')
        self.ax.set_ylabel('value')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_title('plot')
        
        # 创建画布
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # 创建控制面板
        self.setup_control_panel()
        
        # 初始化线条字典
        self.lines = {}
        
    def setup_control_panel(self):
        """设置控制面板"""
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 开始/停止按钮
        self.start_button = tk.Button(control_frame, text="开始", command=self.start)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.stop_button = tk.Button(control_frame, text="停止", command=self.stop, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # 清空数据按钮
        clear_button = tk.Button(control_frame, text="清空数据", command=self.clear_data)
        clear_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # 状态标签
        self.status_label = tk.Label(control_frame, text="状态: 未启动")
        self.status_label.pack(side=tk.RIGHT, padx=5, pady=5)
        
    def add_variable(self, name, color=None, label=None, visible=True):
        """
        添加要绘制的变量
        
        参数:
            name: 变量名称
            color: 线条颜色
            label: 显示标签(默认为变量名)
            visible: 是否可见
        """
        if color is None:
            # 自动分配颜色
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            color = colors[len(self.variable_configs) % len(colors)]
        
        if label is None:
            label = name
            
        with self.data_lock:
            self.data_queues[name] = deque(maxlen=self.window_size)
            self.variable_configs[name] = {
                'color': color,
                'label': label,
                'visible': visible
            }
            
        # 在图形上添加对应的线条
        line, = self.ax.plot([], [], color=color, label=label, linewidth=1.5)
        self.lines[name] = line
        
        # 更新图例
        self.ax.legend(loc='upper right')
        
    def update_data(self, variable_name, value, timestamp=None):
        """
        更新变量数据
        
        参数:
            variable_name: 变量名称
            value: 变量值
            timestamp: 时间戳(默认为当前时间)
        """
        if not self.running:
            return
            
        if timestamp is None:
            timestamp = time.time()
            
        with self.data_lock:
            if variable_name in self.data_queues:
                self.data_queues[variable_name].append(value)
                
                # 如果是第一个变量，用它来更新时间队列
                if len(self.time_queue) == 0 or timestamp != self.time_queue[-1]:
                    self.time_queue.append(timestamp)
    
    def start(self):
        """开始实时绘图"""
        if not self.running:
            self.running = True
            self.start_time = time.time()
            
            # 启动动画
            self.animation = animation.FuncAnimation(
                self.fig, self._update_plot, interval=self.update_interval, blit=False
            )
            
            # 更新按钮状态
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="状态: 运行中")
            
            print("实时绘图已启动")
    
    def stop(self):
        """停止实时绘图"""
        if self.running:
            self.running = False
            if hasattr(self, 'animation'):
                self.animation.event_source.stop()
                
            # 更新按钮状态
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_label.config(text="状态: 已停止")
            
            print("实时绘图已停止")
    
    def clear_data(self):
        """清空所有数据"""
        with self.data_lock:
            for queue in self.data_queues.values():
                queue.clear()
            self.time_queue.clear()
            
        print("数据已清空")
    
    def _update_plot(self, frame):
        """更新绘图（动画回调函数）"""
        with self.data_lock:
            if len(self.time_queue) == 0:
                return self.lines.values()
                
            # 转换为相对时间（从开始时间算起）
            current_time = time.time()
            relative_times = [t - self.start_time for t in self.time_queue]
            
            # 更新每个变量的线条数据
            for var_name, line in self.lines.items():
                if (var_name in self.data_queues and 
                    self.variable_configs[var_name]['visible']):
                    
                    data_queue = self.data_queues[var_name]
                    
                    # 确保数据长度与时间长度匹配
                    if len(data_queue) == len(relative_times):
                        line.set_data(relative_times, list(data_queue))
                    elif len(data_queue) > 0:
                        # 如果数据长度不匹配，使用最近的数据点
                        recent_data = list(data_queue)[-len(relative_times):]
                        line.set_data(relative_times[-len(recent_data):], recent_data)
            
            # 自动调整坐标轴范围
            if len(relative_times) > 0:
                self.ax.set_xlim(0, max(relative_times) + 1)
                
                # 计算所有可见数据的Y轴范围
                y_min = float('inf')
                y_max = float('-inf')
                for var_name, config in self.variable_configs.items():
                    if config['visible'] and var_name in self.data_queues:
                        data = list(self.data_queues[var_name])
                        if len(data) > 0:
                            y_min = min(y_min, min(data))
                            y_max = max(y_max, max(data))
                
                if y_min != float('inf') and y_max != float('-inf'):
                    y_range = y_max - y_min
                    if y_range == 0:
                        y_range = 1
                    self.ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            
            self.fig.tight_layout()
            
        return self.lines.values()
    
    def show(self):
        """显示绘图窗口（阻塞式）"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.stop()
    
    def close(self):
        """关闭绘图窗口"""
        self.stop()
        if hasattr(self, 'root'):
            self.root.quit()
            self.root.destroy()

# 简化的实时绘图器（如果上面的太复杂）
class SimpleRealTimePlotter:
    """
    简化版实时绘图器
    """
    
    def __init__(self, variables=None, window_size=500):
        """
        初始化
        
        参数:
            variables: 变量列表，如 ['force_x', 'force_y', 'position_z']
            window_size: 显示的数据点数量
        """
        self.window_size = window_size
        self.data = {}
        self.timestamps = deque(maxlen=window_size)
        self.start_time = time.time()
        
        # 设置图形
        plt.ion()  # 开启交互模式
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.lines = {}
        
        # 初始化变量
        if variables:
            for i, var in enumerate(variables):
                color = plt.cm.tab10(i % 10)
                line, = self.ax.plot([], [], color=color, label=var, linewidth=1.5)
                self.lines[var] = line
                self.data[var] = deque(maxlen=window_size)
        
        self.ax.set_xlabel('time (s)')
        self.ax.set_ylabel('value')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.legend()
        self.ax.set_title('plot')
        
        plt.tight_layout()
    
    def update(self, **kwargs):
        """
        更新数据
        
        用法:
            plotter.update(force_x=1.5, force_y=2.3, position_z=0.8)
        """
        current_time = time.time() - self.start_time
        self.timestamps.append(current_time)
        
        for var_name, value in kwargs.items():
            if var_name in self.data:
                self.data[var_name].append(value)
                
                # 更新对应的线条
                if len(self.timestamps) == len(self.data[var_name]):
                    self.lines[var_name].set_data(self.timestamps, self.data[var_name])
        
        # 自动调整坐标轴
        if len(self.timestamps) > 0:
            self.ax.set_xlim(0, max(self.timestamps) + 1)
            
            # 计算Y轴范围
            all_data = []
            for data_queue in self.data.values():
                if len(data_queue) > 0:
                    all_data.extend(data_queue)
            
            if all_data:
                y_min, y_max = min(all_data), max(all_data)
                y_range = y_max - y_min
                if y_range == 0:
                    y_range = 1
                self.ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def add_variable(self, name, color=None):
        """添加新变量"""
        if color is None:
            color = plt.cm.tab10(len(self.lines) % 10)
        
        line, = self.ax.plot([], [], color=color, label=name, linewidth=1.5)
        self.lines[name] = line
        self.data[name] = deque(maxlen=self.window_size)
        
        # 更新图例
        self.ax.legend()
        self.fig.canvas.draw()
    
    def close(self):
        """关闭绘图"""
        plt.close(self.fig)
        plt.ioff()