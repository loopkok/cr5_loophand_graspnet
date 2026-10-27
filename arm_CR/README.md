# arm_CR —— Dobot CR5 机械臂控制模块

本模块封装 Dobot CR5 的 TCP/IP 通信，向上提供简洁的运动与状态接口，并附带一组**笛卡尔导纳控制**实验代码。

- Dobot 官方 SDK 文档（中文）：[README_Dobot_API.md](README_Dobot_API.md)
- Dobot 官方 SDK 文档（英文）：[README-EN.md](README-EN.md)
- 项目整体说明：[../README.md](../README.md)

---

## 1. 文件分类

### 1.1 项目自定义

| 文件 | 作用 |
| --- | --- |
| `cr5.py` | **核心封装**：连接、使能、反馈线程、错误处理、`ServoP`/`ServoJ`、SE3 ↔ 位姿转换 |
| `admittanccontroller.py` | 机械臂笛卡尔导纳控制器（三轴） |
| `virtualforcecontroller.py` | 键盘虚拟力输入（方向键/PageUp/PageDown/R） |
| `test_admittance_control.py` | 导纳控制示例：虚拟力 → 导纳 → `MoveCartesian` |
| `force_plotter.py` | 基于 `matplotlib` 的六维力实时曲线 |
| `real_plot.py` | 通用 Tkinter 实时绘图组件 |

### 1.2 Dobot 官方 / 示例

| 文件 | 作用 |
| --- | --- |
| `dobot_api.py` | Dobot TCP/IP Python SDK（Dashboard / Move / FeedBack） |
| `main.py` | 官方连接与反馈示例 |
| `main_UI.py`、`ui.py` | 官方 Tkinter 控制界面 |
| `PythonExample.py` | 官方指令示例 |
| `files/` | 控制器/伺服告警码表（`cr5.py` 错误处理使用） |
| `images/` | 官方文档图片 |
| `README_Dobot_API.md`、`README-EN.md` | 官方 SDK 文档 |

---

## 2. 连接与配置

CR5 通过以太网 TCP/IP 通信：

| 通道 | 端口 | 用途 |
| --- | --- | --- |
| Dashboard | 29999 | 使能、模式、速度、错误清除 |
| Move | 30003 | `ServoP` / `ServoJ` / `MovL` 等运动指令 |
| FeedBack | 30004 | 机器人状态、位姿、力传感器数据 |

默认 IP 为 `192.168.5.1`，硬编码在 `cr5.py::CR5.__init__()` 和 `main.py::ConnectRobot()` 中。现场网段不同时请修改：

```python
self.dashboard = DobotApiDashboard("192.168.5.1", 29999)
self.control   = DobotApiMove("192.168.5.1", 30003)
self.feed      = DobotApi("192.168.5.1", 30004)
```

连接前确认：

```bash
ping 192.168.5.1
```

---

## 3. CR5 封装接口

```python
from cr5 import CR5

robot = CR5()
robot.connect()                      # 启动反馈/清错线程，使能机械臂

robot.MoveCartesian([x, y, z, rx, ry, rz])   # ServoP，单位 mm/度，非阻塞
robot.MoveJoints([j1, ..., j6])              # ServoJ，单位度，非阻塞

pose6 = robot.current_actual         # 最新反馈位姿（由后台线程更新）
q6    = robot.QActual                # 最新关节角
force = robot.get_sixForce()         # 六维力（未安装时返回 None）
tcp_f = robot.get_TCPForce()         # TCP 力

T = robot.get_cartesian()            # 由关节角正解得到的 SE3
pose = robot.TConvertL(T)            # SE3 → [x,y,z,rx,ry,rz]（mm/度）

robot.dashboard.SpeedFactor(20)      # 速度百分比
robot.dashboard.AccJ(10)             # 关节加速度
robot.dashboard.AccL(10)             # 笛卡尔加速度
robot.dashboard.Continue()           # 解除暂停
```

### 3.1 后台线程

`connect()` 会启动两个 daemon 线程：

| 线程 | 作用 |
| --- | --- |
| `GetFeed` | 每 ~1 ms 读取 `DobotApiFeedBack.feedBackData()`，刷新 `current_actual`、`QActual`、`TCPForce`、`sixForce`、错误状态等 |
| `ClearRobotError` | 每 5 s 检查告警；检测到告警时在终端提示并等待输入 `1` 后清除 |

注意 `ClearRobotError` 在检测到告警时会调用 `input()`，会阻塞该线程，等待人工确认。

### 3.2 运动到位判断

`ServoP` / `ServoJ` 都是**非阻塞**指令。主流程通过轮询 `robot.current_actual` 判断是否到位，示例：

```python
def wait_until_arrival(robot, target_xyz, threshold_mm=10.0, timeout=40.0):
    ...
```

`arm_hand_grasp.py::GraspSystem.wait_until_arrival()` 就是这样实现的。

### 3.3 坐标与欧拉角约定

- 位置单位 **mm**，角度单位 **度**。
- `pose2_to_matrix` / `matrix_to_pose` 使用 `scipy` 的 `xyz` 欧拉角顺序。
- `LConvertT` / `TConvertL` 使用 `spatialmath.SE3` 的固定轴 `ZYX` 旋转组合。
- 两套约定在项目中同时存在，跨模块传递位姿时务必确认顺序，详见 [../docs/architecture.md](../docs/architecture.md)。

---

## 4. 导纳控制实验

`test_admittance_control.py` 演示了机械臂末端的柔顺控制：

```bash
cd arm_CR
python test_admittance_control.py
```

操作方式：

| 按键 | 作用 |
| --- | --- |
| ↑ / ↓ | Y 方向虚拟力 |
| ← / → | X 方向虚拟力 |
| PageUp / PageDown | Z 方向虚拟力 |
| `R` 或 `0` | 虚拟力清零 |
| Ctrl+C | 退出 |

控制链：

```text
键盘虚拟力 + 六维力反馈（当前版本主要使用虚拟力）
        │
        ▼
  AdmittanceController.update()
        │  位移增量
        ▼
  SE3.Trans(adj) * T_init
        │
        ▼
  robot.MoveCartesian(target)
```

相关文件：

- `admittanccontroller.py`：`M/K/D` 参数可调，默认质量 0.05、刚度 30、阻尼 4。
- `virtualforcecontroller.py`：虚拟力增量 0.2 N/次，上限 20 N。
- `force_plotter.py` / `real_plot.py`：可视化力曲线（示例中部分绘图代码被注释）。

> 该示例是实验代码：六维力反馈已被读取和滤波，但当前控制主要使用键盘虚拟力；死区、运动阈值等逻辑部分被注释。

---

## 5. 已知限制

- IP、端口硬编码，尚未抽成配置文件。
- `CR5` 类未封装 `SpeedFactor` / `AccJ` / `AccL`，需要直接访问 `robot.dashboard`。
- `current_actual` 由后台线程写入、其他线程无锁读取；高实时场景建议增加同步。
- `get_cartesian()` 走 Dashboard 正解接口，频率较低；高频控制建议使用反馈中的 `current_actual`。
- `ClearRobotError` 中的 `input()` 会阻塞告警处理线程，不适合无人值守。
- 未实现断线重连；通信异常时需重启进程。
- 导纳示例中的参数与机械臂负载、安装姿态强相关，换场景需重新整定。
