import threading
from spatialmath import SE3
from cr5 import *
from admittanccontroller import *
import time
from virtualforcecontroller import *
from real_plot import *


def AdmittanceControlCR5():
    robot = CR5()
    robot.connect()
    robot.set_logging(False)
    time.sleep(2)

    # # 初始化实时绘图器
    # plotter = SimpleRealTimePlotter([
    #     'raw_force_x', 'raw_force_y', 'raw_force_z',
    #     'filtered_force_x', 'filtered_force_y', 'filtered_force_z',
    # ])

     # 初始化虚拟力控制器
    virtual_force_controller = VirtualForceController()

    CR5_AdmittanceController = AdmittanceController(
        _mass=0.05,      # 质量 10 0.05
        _stiffness=30.0, # 刚度 5 30.0 
        _damping=4.0,   # 阻尼 5  4.0
        dt=0.01         # 控制周期 0.01 
    )

    Force_constant = np.array([0.0,0.0,0.0])

    CR5_AdmittanceController.set_Force_constant(Force_constant)

    force_history = []
    force_low_pass = None
    filter_alpha = 0.15

    T_init = robot.get_cartesian()

    print("=" * 50)
    print("虚拟力键盘控制已启动")
    print("控制说明:")
    print("  上下箭头: Y方向力")
    print("  左右箭头: X方向力") 
    print("  PageUp/PageDown: Z方向力")
    print("  R键: 重置力为0")
    print("  Ctrl+C: 退出程序")
    print("=" * 50)

    force_deadzone = 5.0  # 25N死区，只有超过此值才响应
    # 运动阈值
    movement_threshold = 0.2  # 只有调整量超过3mm才运动
    position_init = robot.TConvertL(T_init)
    # print(f"计算出来的初始位置为：{position_init}")
    # print(f"机械臂反馈的实际位置为{robot.current_actual}")
    
    try:
        while True:
            # 检查键盘中断
            try:
                raw_force = robot.get_TCPForce()[0:3] 
                # print(f"原始力为{raw_force}")          
            except KeyboardInterrupt:
                print("用户中断")
                break        
            # 简单的移动平均滤波
            force_history.append(raw_force)
            if len(force_history) > 5:
                force_history.pop(0)
            filtered_force = np.mean(force_history, axis=0)

            #实现一个一阶低通滤波
            vec = np.asarray(filtered_force,dtype=float)
            if force_low_pass is None:
                force_low_pass = vec.copy()
            else:
                force_low_pass = filter_alpha* vec + (1 - filter_alpha) * force_low_pass

            # print(f"一阶低通滤波后的力位{force_low_pass}")

            # 更新虚拟力
            virtual_force_controller.update_force()
            virtual_force = virtual_force_controller.get_force()
            # print(f"键盘虚拟力为：{virtual_force}")

            # # 计算力大小
            # force_magnitude = np.linalg.norm(filtered_force)
            # # # 死区处理
            # if force_magnitude < force_deadzone:
            #     # 力小于死区，不进行控制
            #     print(f"力太小 ({force_magnitude:.1f}N < {force_deadzone}N)，跳过控制")
            #     CR5_AdmittanceController.reset()
            #     continue
            
            # print(f"检测到有效力: {filtered_force}, 大小: {force_magnitude:.1f}N")

            Tt = robot.get_cartesian()
            Rt = Tt.R
            Force_ext = Rt @ virtual_force 
            
            position_adj = CR5_AdmittanceController.update(Force_ext)

            # # 检查运动量
            # adj_norm = np.linalg.norm(position_adj)
            # if adj_norm < movement_threshold:
            #     print(f"调整量太小 ({adj_norm:.1f}mm)，跳过运动")
            #     time.sleep(0.03)
            #     continue

            Te =SE3.Trans(position_adj*1000) * T_init
            target_position = robot.TConvertL(Te)

            #             # 更新实时绘图器
            # plotter.update(
            #     raw_force_x=raw_force[0],
            #     raw_force_y=raw_force[1], 
            #     raw_force_z=raw_force[2],
            #     filtered_force_x=force_low_pass[0],
            #     filtered_force_y=force_low_pass[1],
            #     filtered_force_z=force_low_pass[2],
            # )
            # print(f"目标位置为{target_position}")
            # print("实际位置为",robot.current_actual)

            # target_joints = robot.Ikine(target_position)
            # print("目标各关节角度",target_joints)

            robot.MoveCartesian(target_position)

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"发生错误: {e}")
    # finally:
    #     # 新增：停止绘图器
    #     plotter.stop()
        
    #     # 打印统计信息
    #     stats = plotter.get_statistics()
    #     if stats:
    #         print("\n=== 力数据统计 ===")
    #         print(f"数据点数: {stats['data_points']}")
    #         print(f"X力范围: {stats['force_x_range'][0]:.2f} ~ {stats['force_x_range'][1]:.2f} N")
    #         print(f"Y力范围: {stats['force_y_range'][0]:.2f} ~ {stats['force_y_range'][1]:.2f} N")
    #         print(f"Z力范围: {stats['force_z_range'][0]:.2f} ~ {stats['force_z_range'][1]:.2f} N")
    #         print(f"力大小范围: {stats['force_mag_range'][0]:.2f} ~ {stats['force_mag_range'][1]:.2f} N")
    #         print(f"噪声水平: X={stats['noise_level'][0]:.2f}N, Y={stats['noise_level'][1]:.2f}N, Z={stats['noise_level'][2]:.2f}N")

if __name__ == '__main__':
    AdmittanceControlCR5()
        
    
    
    

    
