# loophand —— Aero Hand Open 触觉灵巧手模块

本模块负责左手的**串口驱动、触觉读取、运动学求解和自适应抓取控制**。它向上只暴露一个 `ShearGraspController` 线程和 `set_command()` 接口，不依赖机械臂或相机。

项目整体说明见 [../README.md](../README.md)，控制算法细节见 [../docs/architecture.md](../docs/architecture.md)。

---

## 1. 目录结构

```text
loophand/
├── README.md
├── tactile_grasping.py          # ★ 触觉自适应抓取控制器（主入口类）
├── aero_open_sdk/               # Aero Hand Open 官方 SDK（TetherIA，Apache-2.0）
│   ├── aero_hand.py             # AeroHand：串口协议、关节/执行器控制
│   ├── aero_hand_constants.py   # 关节限位、执行器参数
│   ├── actuations_to_joints.py  # 执行器 → 关节角
│   ├── joints_to_actuations.py  # 关节角 → 执行器
│   ├── gui.py / __main__.py     # 官方 GUI 示例（python -m aero_open_sdk）
│   └── Read_Single_Sensor_Hand.py # 独立的 GEN3 触觉传感器串口读取工具
├── controller/
│   ├── pid.py                   # BoundedPID（二维力误差，带积分限幅）
│   └── admittancecontroller.py  # 三维导纳控制器（带速度/加速度限幅）
├── extern_driver/
│   ├── tactile_driver.py        # 多点触觉传感器总线驱动（主流程使用）
│   └── Read_Single_Sensor_Usb.py# 掌心 9 点阵列传感器独立驱动（未接入主流程）
├── examples/
│   └── test_hand.py             # 早期独立手部测试脚本（历史版本）
└── urdf/
    └── aero_hand_open_left.urdf # 左手指运动学模型
```

标 ★ 的是本项目核心控制代码；`aero_open_sdk/` 为上游 SDK。

---

## 2. 硬件与连接

| 设备 | 默认端口 | 波特率 |
| --- | --- | --- |
| Aero Hand Open（左手） | `/dev/ttyACM1` | 921600 |
| 触觉传感器总线 | `/dev/ttyACM0` | 921600 |
| 掌心传感器（可选，独立工具） | 自动选择 `/dev/ttyACM0`、`/dev/ttyACM1` | 921600 |

```bash
ls -l /dev/ttyACM*
sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1
```

> 端口号可能随插拔顺序变化。若握持/触觉读取异常，先用 `ls -l /dev/ttyACM*` 确认设备对应关系，或修改 `ShearGraspController` 的 `port_hand` / `port_tactile` 参数。
>
> `Read_Single_Sensor_Usb.py` 会同时占用 `ttyACM0` 和 `ttyACM1`，**不能与手部控制器同时运行**。

---

## 3. 依赖

```bash
pip install pyserial numpy scipy
# Pinocchio（运动学）
conda install -c conda-forge pinocchio
# 或 pip install pin
```

- `numpy`、`pyserial`：驱动与信号处理。
- `pinocchio`：URDF 解析、正运动学、雅可比。
- `tactile_grasping.py` 的交互测试只需上述依赖，不依赖相机/机械臂。

---

## 4. 快速使用

### 4.1 交互测试

```bash
cd loophand
python tactile_grasping.py
```

程序会初始化手和触觉传感器、去皮 2 秒，然后进入命令循环：

```text
PRE_SHAPE   # 拇指外展，准备包围物体
GRASP       # 接触后进入触觉闭环，自适应闭合
RELEASE     # 张开
IDLE        # 停止动作
exit        # 退出
```

### 4.2 在其他程序中调用

```python
from tactile_grasping import ShearGraspController

hand = ShearGraspController(
    urdf_path="loophand/urdf/aero_hand_open_left.urdf",
    port_hand="/dev/ttyACM1",
    port_tactile="/dev/ttyACM0",
)
hand.start()                 # 启动 100 Hz 控制线程

hand.set_command("PRE_SHAPE")
hand.set_command("GRASP")
hand.set_command("RELEASE")
hand.set_command("IDLE")

hand.stop_thread()
```

主流程 [`../graspnet-baseline/arm_hand_grasp.py`](../graspnet-baseline/arm_hand_grasp.py) 就是这样调用的，`urdf_path` 通过项目根目录拼接为绝对路径。

`ShearGraspController` 继承 `threading.Thread` 且为 daemon 线程；主程序退出时线程随之结束。

---

## 5. 控制架构

### 5.1 状态机

```text
IDLE ──PRE_SHAPE──► PRE_SHAPE ──GRASP──► ACTIVE_GRASP
  ▲                                          │
  └──────────── RELEASE ◄────────────────────┘
```

| 状态 | 动作 |
| --- | --- |
| `IDLE` | 保持不动 |
| `PRE_SHAPE` | 拇指外展到 `thumb_abd_target = 1.65 rad` |
| `ACTIVE_GRASP` | 未接触手指恒速闭合；接触手指进入力闭环 |
| `RELEASING` | 目标关节置零，保留拇指外展 |

### 5.2 每根手指的控制流程

1. 读取该手指对应的 1–2 个触觉传感器（Tip / Mid）。
2. `SignalProcessor` 低通滤波，计算法向力、切向力、剪切率和滑移比。
3. 判断接触（`normal_force > 0.1 N`，带滞回）。
4. **未接触**：开环恒速闭合（拇指 0.01，其余 0.03 rad/tick）。
5. **已接触**：
   - 根据摩擦系数估计目标法向力 `F_t / μ × safety_margin`；
   - 滑移比超阈值时增加目标力；
   - PID 抑制切向剪切；
   - 导纳控制器把力误差转成末端位移；
   - Pinocchio 雅可比 + 阻尼最小二乘求 7 个主动关节增量；
   - 欠驱动关节按 `1.0 / 0.8 / 0.6` 联动；
   - 防回退棘轮、速度限幅后下发。

关键参数位于 `tactile_grasping.py::__init__` 的 `self.params` 和 `ADM_*` 常量：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `mu_est` | 0.4 | 估计摩擦系数 |
| `safety_margin` | 1.5 | 目标力安全系数 |
| `min_force` | 3.0 N | 最小法向力 |
| `contact_thresh` | 0.1 N | 接触阈值 |
| `thumb_abd_target` | 1.65 rad | 预姿态拇指外展 |
| `limit_speed` | 0.06 rad/tick | 关节增量限幅 |
| `ADM_M/D/K` | 0.11 / 12.0 / 0.0 | 导纳质量/阻尼/刚度 |

### 5.3 触觉传感器映射

`SENSOR_TO_LINK_MAP` 把 9 个传感器映射到 URDF link：

```text
Thumb-Tip  → left_thumb_tip_link
Index-Tip  → left_index_tip_link
Index-Mid  → left_index_proximal_link
Middle-Tip → left_middle_tip_link
Middle-Mid → left_middle_proximal_link
Ring-Tip   → left_ring_tip_link
Ring-Mid   → left_ring_proximal_link
Pinky-Tip  → left_pinky_tip_link
Pinky-Mid  → left_pinky_proximal_link
```

`tactile_driver.py` 支持的总线设备名更多（含 Prox/Nail/Palm），但控制器只订阅上述 9 个。更换手型或传感器布局时，需要同步修改该映射。

---

## 6. 独立工具

| 工具 | 命令 | 说明 |
| --- | --- | --- |
| 官方 GUI | `python -m aero_open_sdk` | 扫描串口、连接手、手动控制关节 |
| 手部触觉传感器读取 | `python aero_open_sdk/Read_Single_Sensor_Hand.py` | 交互式选择串口，读取 GEN3 传感器自动回传帧 |
| 掌心阵列传感器 | `python extern_driver/Read_Single_Sensor_Usb.py` | 左右掌心各 9 点，OpenCV 热力图显示 |
| 早期测试脚本 | `python examples/test_hand.py` | 与 `tactile_grasping.py` 类似，参数略有不同，仅作参考 |

---

## 7. 已知限制

- 控制器只针对**左手**和 `aero_hand_open_left.urdf` 调试；换右手需调整 link 名称、关节方向和 URDF。
- 主流程只使用 7 个主动执行器；手指远端关节由欠驱动耦合近似。
- `examples/test_hand.py` 是 `tactile_grasping.py` 的早期版本，`min_force` 等参数与当前版本不同，不建议作为生产入口。
- `extern_driver/Read_Single_Sensor_Usb.py`（掌心传感器）尚未接入 `ShearGraspController`。
- 串口为阻塞读写；设备掉线时 `run()` 中的异常会被吞掉并继续循环，不会主动退出。
- 触觉阈值与导纳参数与具体物体/手型相关，换场景需要重新整定。
