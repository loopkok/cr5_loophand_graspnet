import threading
from dobot_api import DobotApiDashboard, DobotApi, DobotApiMove,DobotApiFeedBack, MyType, alarmAlarmJsonFile
from time import sleep
import numpy as np
import re
from spatialmath import SE3
from scipy.spatial.transform import Rotation as R

# 全局变量(当前坐标)
current_actual = np.array([-1])
TCPForce = [-1]
Qactual = np.array([-1])
actual_quaternion = [-1]
Tt = [-1]
algorithm_queue = -1
enableStatus_robot = -1
robotErrorState = False
robotMode = 0   
globalLockValue = threading.Lock()


def ConnectRobot():
    try:
        ip = "192.168.5.1"
        dashboardPort = 29999
        movePort = 30003
        feedPort = 30004
        print("正在建立连接...")
        dashboard = DobotApiDashboard(ip, dashboardPort)
        move = DobotApiMove(ip, movePort)
        feed = DobotApi(ip, feedPort)
        feedFour = DobotApiFeedBack(ip,feedPort)
        print(">.<连接成功>!<")
        return dashboard, move, feed,feedFour
    except Exception as e:
        print(":(连接失败:(")
        raise e


def RunPoint(move: DobotApiMove, point_list: list):
    move.MovL(point_list[0], point_list[1], point_list[2],
              point_list[3], point_list[4], point_list[5])


def Fkine(q):
    Tt = dashboard.PositiveSolution(*q, 0, 0)
    # 解析返回的字符串，提取花括号内的笛卡尔坐标
    # 格式: 0,{129.872670,-260.502409,1012.281628,-24.563281,-49.766709,140.695290},...
    match = re.search(r'\{([^}]+)\}', Tt)
    if match:
        # 提取花括号内的内容并转换为浮点数列表
        coords_str = match.group(1)
        coords = [float(x.strip()) for x in coords_str.split(',')]
        return coords
    else:
        return []

def get_cartesian_from_feedback():
    """
    使用直接反馈的笛卡尔坐标构建 SE3 变换对象
    """
    global current_actual
    if current_actual is not None and len(current_actual) >= 6 and current_actual[0] != -1:
        x, y, z = current_actual[0], current_actual[1], current_actual[2]
        rx, ry, rz = current_actual[3], current_actual[4], current_actual[5]
        
        try:
                    # 创建平移变换
                    T_trans = SE3.Trans(x, y, z)
                    
                    # 创建旋转变换 - 使用固定轴ZYX顺序
                    # 注意: spatialmath 使用弧度，所以需要将度转换为弧度
                    T_rot = SE3.Rz(np.radians(rz)) * SE3.Ry(np.radians(ry)) * SE3.Rx(np.radians(rx))
                    
                    # 组合平移和旋转
                    T = T_trans * T_rot
                    
                    print("使用固定轴ZYX顺序的变换矩阵:")
                    print(T)
                    return T
        except Exception as e:
                    print(f"方法1失败: {e}")
    else:
        print("反馈: 坐标数据不足或为初始值")
        return SE3()

def get_cartesian():
    """
    返回当前笛卡尔坐标的 SE3 变换对象
    返回格式: SE3 变换矩阵，可以通过 .R 访问旋转矩阵，通过 .t 访问平移向量
    """
    coords = Fkine(Qactual)
    if len(coords) >= 6:
        # coords = [x, y, z, rx, ry, rz] (单位：mm, 度)
        x, y, z = coords[0], coords[1], coords[2]
        rx, ry, rz = coords[3], coords[4], coords[5]
        
        # 位置向量（单位：mm）
        position = np.array([x, y, z])
        
        # 将欧拉角从度转换为弧度，并创建旋转矩阵
        # 注意：这里假设是 XYZ 欧拉角顺序（roll-pitch-yaw），根据实际机器人调整
        euler_angles = np.array([np.radians(rx), np.radians(ry), np.radians(rz)])
        rotation = R.from_euler('xyz', euler_angles, degrees=False)
        rotation_matrix = rotation.as_matrix()
        
        # 创建 SE3 变换对象：先平移，再旋转
        T = SE3.Trans(x, y, z) * SE3(rotation_matrix)
        return T
    else:
        # 如果坐标不完整，返回单位变换
        return SE3()

def GetFeed(feedFour: DobotApiFeedBack):
    global current_actual
    global algorithm_queue
    global enableStatus_robot
    global robotErrorState
    global robotMode
    global TCPForce
    global Qactual
    global actual_quaternion
# 获取机器人状态
    while True:
        with globalLockValue:
            feedInfo = feedFour.feedBackData()
            if hex((feedInfo['test_value'][0])) == '0x123456789abcdef':
                # Refresh Properties
                robotMode=feedInfo['robot_mode'][0]
                current_actual = feedInfo["tool_vector_actual"][0]
                algorithm_queue = feedInfo['run_queued_cmd'][0]
                enableStatus_robot = feedInfo['enable_status'][0]
                robotErrorState = feedInfo['error_status'][0]
                TCPForce = feedInfo['TCP_force'][0]
                Qactual = feedInfo['q_actual'][0]
                actual_quaternion = feedInfo['actual_quaternion'][0]
                # 自定义添加所需反馈数据
            sleep(0.001)

def WaitArrive(point_list):
    while True:
        is_arrive = True
        globalLockValue.acquire()
        if current_actual is not None:
            for index in range(4):
                if (abs(current_actual[index] - point_list[index]) > 1):
                    is_arrive = False
            if is_arrive:
                globalLockValue.release()
                return
        globalLockValue.release()
        sleep(0.001)


def ClearRobotError(dashboard: DobotApiDashboard):
    global robotErrorState
    dataController, dataServo = alarmAlarmJsonFile()    # 读取控制器和伺服告警码
    while True:
        globalLockValue.acquire()
        if robotErrorState:
            numbers = re.findall(r'-?\d+', dashboard.GetErrorID())
            numbers = [int(num) for num in numbers]
            if (numbers[0] == 0):
                if (len(numbers) > 1):
                    for i in numbers[1:]:
                        alarmState = False
                        if i == -2:
                            print("机器告警 机器碰撞 ", i)
                            alarmState = True
                        if alarmState:
                            continue
                        for item in dataController:
                            if i == item["id"]:
                                print("机器告警 Controller errorid", i,
                                      item["zh_CN"]["description"])
                                alarmState = True
                                break
                        if alarmState:
                            continue
                        for item in dataServo:
                            if i == item["id"]:
                                print("机器告警 Servo errorid", i,
                                      item["zh_CN"]["description"])
                                break

                    choose = input("输入1, 将清除错误, 机器继续运行: ")
                    if int(choose) == 1:
                        dashboard.ClearError()
                        sleep(0.01)
                        dashboard.Continue()

        else:
            if int(enableStatus_robot) == 1 and int(algorithm_queue) == 0:
                dashboard.Continue()
        globalLockValue.release()
        sleep(5)


if __name__ == '__main__':
    dashboard, move, feed,feedFour = ConnectRobot()
    feed_thread = threading.Thread(target=GetFeed, args=(feedFour,))
    feed_thread.daemon = True
    feed_thread.start()
    feed_thread1 = threading.Thread(target=ClearRobotError, args=(dashboard,))
    feed_thread1.daemon = True
    feed_thread1.start()
    print("开始使能...")
    dashboard.EnableRobot()
    print("完成使能:)")
    while True:
        if int(enableStatus_robot) == 1:
            print("机器人使能成功，可以运行程序")
        
