# virtualforcecontroller.py 的替代版本
from pynput import keyboard
import numpy as np

class VirtualForceController:
    """虚拟力键盘控制器 - 使用 pynput"""
    def __init__(self):
        self.force = np.array([0.0, 0.0, 0.0])
        self.force_increment = 0.2
        self.max_force = 20.0
        self.decay_rate = 0.9
        
        # 按键状态跟踪
        self.key_state = {
            'up': False,      # Y+ 方向
            'down': False,    # Y- 方向  
            'left': False,    # X- 方向
            'right': False,   # X+ 方向
            'page_up': False, # Z+ 方向
            'page_down': False, # Z- 方向
            '0': False        # 重置键
        }
        
        # 方向向量映射
        self.direction_mapping = {
            'up': np.array([1.0, 0.0, 0.0]),
            'down': np.array([-1.0, -1.0, 0.0]),
            'left': np.array([0.0, -1.0, 0.0]),
            'right': np.array([0.0, 1.0, 0.0]),
            'page_up': np.array([0.0, 0.0, 1.0]),
            'page_down': np.array([0.0, 0.0, -1.0])
        }
        
        # 启动键盘监听
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()
        
    def on_press(self, key):
        """按键按下时的回调"""
        try:
            key_name = key.char
            if key_name == '0' or key_name == 'r' or key_name == 'R':
                self.key_state['0'] = True
                self.force = np.array([0.0, 0.0, 0.0])
                print("力已重置为0")
        except AttributeError:
            # 特殊键（方向键等）
            key_name = str(key)
            if key_name == 'Key.up':
                self.key_state['up'] = True
            elif key_name == 'Key.down':
                self.key_state['down'] = True
            elif key_name == 'Key.left':
                self.key_state['left'] = True
            elif key_name == 'Key.right':
                self.key_state['right'] = True
            elif key_name == 'Key.page_up':
                self.key_state['page_up'] = True
            elif key_name == 'Key.page_down':
                self.key_state['page_down'] = True
                
    def on_release(self, key):
        """按键释放时的回调"""
        try:
            key_name = key.char
            if key_name == '0' or key_name == 'r' or key_name == 'R':
                self.key_state['0'] = False
        except AttributeError:
            key_name = str(key)
            if key_name == 'Key.up':
                self.key_state['up'] = False
            elif key_name == 'Key.down':
                self.key_state['down'] = False
            elif key_name == 'Key.left':
                self.key_state['left'] = False
            elif key_name == 'Key.right':
                self.key_state['right'] = False
            elif key_name == 'Key.page_up':
                self.key_state['page_up'] = False
            elif key_name == 'Key.page_down':
                self.key_state['page_down'] = False
    
    def update_force(self):
        """根据按键状态更新虚拟力"""
        active_force = np.array([0.0, 0.0, 0.0])
        any_key_pressed = False
        
        for key_name in ['up', 'down', 'left', 'right', 'page_up', 'page_down']:
            if self.key_state[key_name]:
                active_force += self.direction_mapping[key_name]
                any_key_pressed = True
        
        if any_key_pressed:
            # 归一化方向并增加力
            if np.linalg.norm(active_force) > 0:
                active_force = active_force / np.linalg.norm(active_force)
                self.force += active_force * self.force_increment
                
                # 限制最大力
                force_magnitude = np.linalg.norm(self.force)
                if force_magnitude > self.max_force:
                    self.force = self.force / force_magnitude * self.max_force
        else:
            # 没有按键时，力逐渐衰减
            self.force *= self.decay_rate
            
        # 如果力很小，直接设为0
        if np.linalg.norm(self.force) < 0.1:
            self.force = np.array([0.0, 0.0, 0.0])
            
    def get_force(self):
        """获取当前虚拟力"""
        return self.force.copy()
    
    def get_force_info(self):
        """获取力的详细信息"""
        magnitude = np.linalg.norm(self.force)
        return {
            'force_vector': self.force.copy(),
            'magnitude': magnitude,
            'direction': self.force / magnitude if magnitude > 0 else np.array([0.0, 0.0, 0.0])
        }
    
    def stop(self):
        """停止监听"""
        self.listener.stop()