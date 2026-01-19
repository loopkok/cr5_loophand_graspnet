from ast import List
import numpy as np
import threading
from dobot_api import DobotApiDashboard, DobotApi, DobotApiMove,DobotApiFeedBack, MyType, alarmAlarmJsonFile
from time import sleep
import re
from spatialmath import SE3
from scipy.spatial.transform import Rotation as R
from force_plotter import ForceDataPlotter


class DobotLogController:
    """Dobot API日志控制器"""
    
    def __init__(self):
        self.original_log_methods = {}
        self.logging_enabled = True
    
    def disable_logging(self, dobot_objects):
        """禁用Dobot对象的日志输出"""
        for obj in dobot_objects:
            # 保存原始log方法
            self.original_log_methods[id(obj)] = obj.log
            # 替换为静默方法
            obj.log = self._silent_log
    
    def enable_logging(self, dobot_objects):
        """启用Dobot对象的日志输出"""
        for obj in dobot_objects:
            obj_id = id(obj)
            if obj_id in self.original_log_methods:
                obj.log = self.original_log_methods[obj_id]
    
    def _silent_log(self, text):
        """静默日志方法"""
        pass


class CR5():
    def __init__(self):
        self.dashboard = DobotApiDashboard("192.168.5.1", 29999)
        self.control = DobotApiMove("192.168.5.1", 30003) 
        self.feed = DobotApi("192.168.5.1", 30004)
        self.feedFour = DobotApiFeedBack("192.168.5.1", 30004)
        self.myType = MyType
        self.alarmAlarmJsonFile = alarmAlarmJsonFile()
        self.lock = threading.Lock()
        self.current_actual = [-1]
        self.TCPForce = [-1]
        self.QActual = [-1]
        self.algorithm_queue = -1
        self.enableStatus_robot = -1
        self.six_force_online = 0
        self.sixForce = np.zeros(6, dtype=np.float64)
 
            # 初始化日志控制器并禁用日志
        self.log_controller = DobotLogController()
        self.set_logging(False)
    def set_logging(self, enabled):
        """控制Dobot API的日志输出"""
        dobot_objects = [self.dashboard, self.control, self.feed, self.feedFour]
        if enabled:
            self.log_controller.enable_logging(dobot_objects)
        else:
            self.log_controller.disable_logging(dobot_objects)


    def LConvertT(self,coords):
        if len(coords) >= 6:
            # coords = [x, y, z, rx, ry, rz] (单位：mm, 度)
            x, y, z = coords[0], coords[1], coords[2]
            rx, ry, rz = coords[3], coords[4], coords[5]
            
            # 创建平移变换
            T_trans = SE3.Trans(x, y, z)
            
            # 创建旋转变换 - 使用固定轴ZYX顺序
            # 注意: spatialmath 使用弧度，所以需要将度转换为弧度
            T_rot = SE3.Rz(np.radians(rz)) * SE3.Ry(np.radians(ry)) * SE3.Rx(np.radians(rx))
            
            # 组合平移和旋转
            T = T_trans * T_rot
            return T
        else:
            # 如果坐标不完整，返回单位变换
            return SE3()

    def TConvertL(self, Tt):
        """将SE3对象转换回Dobot位姿列表 (ZYX顺序)"""
        # 提取平移向量 (单位: 毫米)
        position = Tt.t
        
        # 提取旋转矩阵
        rotation_matrix = Tt.R
        
        # 将旋转矩阵转换为ZYX欧拉角
        # 使用scipy的Rotation类，指定ZYX顺序
        r = R.from_matrix(rotation_matrix)
        euler_zyx = r.as_euler('zyx', degrees=True)  # 返回 [rz, ry, rx] 度
        
        # 重新排序为 [rx, ry, rz] 以匹配Dobot格式
        # euler_zyx = [rz, ry, rx]，我们需要 [rx, ry, rz]
        euler_dobot = [
            euler_zyx[2],  # rx (绕X轴旋转)
            euler_zyx[1],  # ry (绕Y轴旋转) 
            euler_zyx[0]   # rz (绕Z轴旋转)
        ]
        
        # 组合位置和姿态
        dobot_pose = [
            position[0],    # x (mm)
            position[1],    # y (mm)
            position[2],    # z (mm)
            euler_dobot[0], # rx (度)
            euler_dobot[1], # ry (度)
            euler_dobot[2]  # rz (度)
        ]
        
        return dobot_pose


    def Ikine(self,x):
        Joints = self.dashboard.InverseSolution(*x,0,0)
        match = re.search(r'\{([^}]+)\}', Joints)
        if match:
            # 提取花括号内的内容并转换为浮点数列表
            coords_str = match.group(1)
            coords = [float(x.strip()) for x in coords_str.split(',')]
            return coords
        else:
            return []

    def Fkine(self,q):
        
        Tt = self.dashboard.PositiveSolution(*q,0,0)
        match = re.search(r'\{([^}]+)\}', Tt)
        if match:
            # 提取花括号内的内容并转换为浮点数列表
            coords_str = match.group(1)
            coords = [float(x.strip()) for x in coords_str.split(',')]
            return coords
        else:
            return []

    def GetFeed(self):
        while True:
            with self.lock:
                feedInfo = self.feedFour.feedBackData()
                if hex((feedInfo['test_value'][0])) == '0x123456789abcdef':
                    # Refresh Properties
                    self.robotMode=feedInfo['robot_mode'][0]
                    self.current_actual = feedInfo["tool_vector_actual"][0]
                    self.algorithm_queue = feedInfo['run_queued_cmd'][0]
                    self.enableStatus_robot = feedInfo['enable_status'][0]
                    self.robotErrorState = feedInfo['error_status'][0]
                    self.TCPForce = feedInfo['TCP_force'][0]
                    self.QActual = feedInfo['q_actual'][0]
                    self.six_force_online = int(feedInfo['six_force_online'][0])
                    self.sixForce = feedInfo['six_force_value'][0]
                                  # 自定义添加所需反馈数据
            sleep(0.001)

    def get_cartesian(self):
        """
        返回当前笛卡尔坐标的 SE3 变换对象
        返回格式: SE3 变换矩阵，可以通过 .R 访问旋转矩阵，通过 .t 访问平移向量
        """
        coords = self.Fkine(self.QActual)
        return self.LConvertT(coords)

    def get_sixForce(self):
        if self.six_force_online == 1:
            return self.sixForce.copy()
        else:
            print("未安装六维力传感器")
            return None
    def get_TCPForce(self):
        return self.TCPForce.copy()

    def MoveJoints(self,point_joinsts):
        self.control.ServoJ(
            point_joinsts[0],
            point_joinsts[1],
            point_joinsts[2],
            point_joinsts[3],
            point_joinsts[4],
            point_joinsts[5])

    def MoveCartesian(self,ponit_cartesian):
        self.control.ServoP(
            ponit_cartesian[0],
            ponit_cartesian[1],
            ponit_cartesian[2],
            ponit_cartesian[3],
            ponit_cartesian[4],
            ponit_cartesian[5])
    

    def ClearRobotError(self):
        dataController, dataServo = alarmAlarmJsonFile()    # 读取控制器和伺服告警码
        while True:
            self.lock.acquire()
            if  self.robotErrorState:
                numbers = re.findall(r'-?\d+', self.dashboard.GetErrorID())
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
                            self.dashboard.ClearError()
                            sleep(0.01)
                            self.dashboard.Continue()

            else:
                if int( self.enableStatus_robot) == 1 and int(self.algorithm_queue) == 0:
                    self.dashboard.Continue()
            self.lock.release()
            sleep(5)

    def connect(self):       
        feed_thread = threading.Thread(target=self.GetFeed)
        feed_thread.daemon = True
        feed_thread.start()

        ClearError_thread  = threading.Thread(target=self.ClearRobotError)
        ClearError_thread.daemon = True
        ClearError_thread.start()

        result = self.dashboard.SetArmOrientation(1,-1,1,1)
        print("设置机械臂配置:", result)

        print("开始使能...")
        self.dashboard.EnableRobot()
        print("完成使能:)")

