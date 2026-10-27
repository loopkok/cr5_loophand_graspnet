# 架构与实现说明

本文档描述系统各模块的边界、数据流、坐标变换、线程模型和关键算法，便于二次开发与排障。快速上手请先看根目录 [README.md](../README.md)。

---

## 1. 模块边界

```text
┌───────────────────────────────────────────────────────────────┐
│                        应用层（编排）                          │
│  graspnet-baseline/arm_hand_grasp.py                           │
│  - 硬件初始化、流程控制、人工确认、异常处理                     │
└──────────────┬──────────────────┬─────────────────────────────┘
               │                  │
   ┌───────────▼──────────┐   ┌───▼────────────────────────────┐
   │ 感知与规划层          │   │ 执行层                          │
   │ graspnet-baseline/    │   │ arm_CR/ + loophand/             │
   │ - gemini335.py        │   │ - cr5.py（机械臂）              │
   │ - vlm_process.py      │   │ - tactile_grasping.py（手）     │
   │ - process.py          │   │ - controller/（PID/导纳）       │
   └───────────┬──────────┘   │ - extern_driver/（触觉）        │
               │              └───┬────────────────────────────┘
               ▼                  ▼
        RGB-D / mask / pose   关节/力/坐标指令
```

依赖方向是单向的：应用层依赖硬件层；硬件层不反向依赖应用层。`loophand/tactile_grasping.py` 只通过 `set_command()` 接收高层状态，不感知机械臂或相机。

### 目录职责速查

| 路径 | 职责 | 是否可直接复用 |
| --- | --- | --- |
| `graspnet-baseline/arm_hand_grasp.py` | 系统主流程 | 项目专用 |
| `graspnet-baseline/process.py` | 点云 → GraspNet → 基座系位姿 | 项目专用 |
| `graspnet-baseline/gemini335.py` | Gemini 335 采集与保存 | 可独立使用 |
| `graspnet-baseline/vlm_process.py` | Qwen-VL + SAM 分割 | 可独立使用 |
| `graspnet-baseline/vlm_plan.py` | VLM 技能序列生成 | 实验性 |
| `graspnet-baseline/planner_client.py` | socket 任务客户端 | 实验性 |
| `graspnet-baseline/legacy/vlm_planner/` | 早期 VLM 规划原型 | 仅参考 |
| `loophand/tactile_grasping.py` | 触觉自适应抓取状态机 | 项目专用 |
| `loophand/controller/` | PID、导纳 | 通用 |
| `loophand/extern_driver/` | 触觉串口协议 | 可独立使用 |
| `loophand/aero_open_sdk/` | 手部 SDK | 上游 |
| `arm_CR/cr5.py` | CR5 封装 | 可复用 |
| `arm_CR/dobot_api.py` | Dobot SDK | 上游 |
| `arm_CR/admittanccontroller.py` | 机械臂导纳 | 实验用 |

---

## 2. 主流程数据流

`arm_hand_grasp.py::GraspSystem.run()` 的每轮循环：

```text
1. CR5.MoveCartesian(HOME_POSE)            # 归位
2. hand.set_command("IDLE")
3. Gemini335.capture_frame()
       → color (BGR uint8), raw_depth (uint16 mm)
4. segment_image(color, output_mask=...)
       → Qwen-VL 输出 bbox → SAM 输出 mask (uint8, 0/255)
       → 转 bool: mask > 0
5. targetpose_generation(color, depth, mask_bool, current_pose)
       → process.py:
          a. 用内参把深度图反投影为 organized 点云
          b. mask & (depth > 0) 筛选点云与颜色
          c. 采样 num_point=20000 点
          d. GraspNet 前向 → GraspGroup
          e. 碰撞/方向/向心筛选 → NMS → top 20
          f. Open3D 可视化，取第 0 个最佳抓取
          g. 相机系 → 基座系 → 法兰目标位姿
6. input("确认执行抓取吗？")                # 人工确认
7. calculate_pre_grasp(target_pose)         # 沿工具 Z 轴回退 100 mm
8. hand.set_command("PRE_SHAPE")
   CR5.MoveCartesian(pre_grasp_pose)
9. CR5 低速 MoveCartesian(target_pose)
   hand.set_command("GRASP")
   sleep(3.0)                               # 等待力闭合稳定
10. 抬起 → MoveCartesian(PLACE_POSE)
    hand.set_command("RELEASE")
```

### 关键函数契约

| 函数 | 输入 | 输出 |
| --- | --- | --- |
| `Gemini335.capture_frame()` | 无 | `(color_bgr, raw_depth_uint16, depth_viz, overlay)`，失败返回四个 `None` |
| `segment_image(image, output_mask)` | BGR 图像、掩码保存路径 | `uint8` 掩码（0/255）或 `None` |
| `targetpose_generation(color, depth, mask_bool, current_pose)` | BGR、深度 mm、bool 掩码、CR5 当前位姿 `[x,y,z,rx,ry,rz]`（mm/度） | 基座系法兰目标位姿 `list[6]` 或 `None` |
| `ShearGraspController.set_command(cmd)` | `"IDLE" / "PRE_SHAPE" / "GRASP" / "RELEASE"` | 无（异步改变状态） |
| `CR5.MoveCartesian(pose6)` | `[x,y,z,rx,ry,rz]`（mm/度） | 无（`ServoP`，非阻塞） |

> `CR5.MoveCartesian` 调用的是 `ServoP`，属于**非阻塞**指令。主流程用 `wait_until_arrival()` 轮询 `robot.current_actual` 判断是否到位。

---

## 3. 坐标帧与手眼变换

系统中涉及五个坐标系：

| 符号 | 含义 |
| --- | --- |
| `base` | CR5 基座坐标系 |
| `flange` | 机械臂末端法兰坐标系（`current_actual` / `ServoP` 的控制对象） |
| `cam` | 相机坐标系（GraspNet 输出抓取位姿的坐标系） |
| `grasp` | GraspNet 定义的抓取坐标系 |
| `tcp` | 灵巧手掌心工具坐标系（实际希望抓取点到达的位置） |

`process.py` 中的变换链：

```python
T_cam_grasp = [R_grasp | t_grasp]              # GraspNet 输出
T_cam_target = T_cam_grasp @ T_corr            # 调整到本项目约定的工具朝向
T_base_flange = pose2_to_matrix(current_pose)  # 当前法兰位姿（mm/度 → m/弧度）
T_base_targetObj = T_base_flange @ T_FLANGE_CAM @ T_cam_target
T_base_targetArm = T_base_targetObj @ inv(T_FLANGE_TCP)
arm_target_pose = matrix_to_pose(T_base_targetArm)
```

其中：

- `T_FLANGE_CAM`：眼在手上标定得到的 **相机 → 法兰**（cam2gripper）变换，即 `cv.calibrateHandEye()` 的输出 `RT_cam2gripper`。注意变量名容易误读，这里左乘 `T_base_flange` 后把相机系点变换到基座系。
- `T_FLANGE_TCP`：**法兰 → 掌心工具中心** 的固定安装变换（含 160 mm 前伸、15 mm 侧偏、55 mm 高度）。
- `T_corr`：把 GraspNet 抓取坐标系旋转到夹爪闭合/进给约定：

```text
T_corr = [[ 0, 0, 1],
          [ 0, 1, 0],
          [-1, 0, 0]]
```

单位约定：

- CR5 位姿：位置 **mm**，欧拉角 **度**，`scipy` 的 `xyz` 顺序。
- GraspNet / 点云 / 手眼矩阵：**米**。
- 深度图：**毫米**，`factor_depth = 1000`。

更换相机安装、末端手爪或重新标定后，必须同步更新 `T_FLANGE_CAM`、`T_FLANGE_TCP` 和相机内参。

---

## 4. GraspNet 抓取筛选

`process.py::pose_filter()` 在原始 GraspGroup 上依次执行：

1. **碰撞检测**（`ModelFreeCollisionDetector`）
   - `voxel_size = 0.01 m`
   - `approach_dist = 0.05 m`
   - `collision_thresh = 0.01`
2. **进给方向过滤**：抓取接近向量与 `[0,0,1]` 夹角 `< 30°`，保证自上而下抓取。
3. **闭合方向过滤**：闭合向量与 `[1,0,0]` 夹角 `< 90°`，避免手腕大角度扭转。
4. **向心惩罚**：`score -= 2.0 * ||grasp_translation - object_centroid||`，让靠近物体质心的抓取排前。
5. **NMS + 排序**，保留前 20 个。

可视化时最佳抓取为红色，其余候选为绿色。Open3D 窗口是**阻塞**的，关闭窗口后才会继续机械臂运动。

---

## 5. 触觉手控制

`loophand/tactile_grasping.py::ShearGraspController` 是一个 100 Hz（`dt = 0.01 s`）的守护线程。

### 5.1 状态机

```text
IDLE ──PRE_SHAPE──► PRE_SHAPE ──GRASP──► ACTIVE_GRASP
  ▲                                          │
  └──────────── RELEASE ◄────────────────────┘
```

| 状态 | 行为 |
| --- | --- |
| `IDLE` | 不发送指令，50 ms 轮询一次 |
| `PRE_SHAPE` | 拇指外展到 `thumb_abd_target = 1.65 rad`，速度为 0.03 rad/tick |
| `ACTIVE_GRASP` | 每根手指独立判断接触；未接触则恒速闭合，接触后进入力闭环 |
| `RELEASING` | 7 个执行器目标置零，仅保留拇指外展 1.65 |

### 5.2 触觉信号处理

9 个传感器通过 `SENSOR_TO_LINK_MAP` 映射到 URDF link：

```text
Thumb-Tip, Index-Tip, Index-Mid, Middle-Tip, Middle-Mid,
Ring-Tip, Ring-Mid, Pinky-Tip, Pinky-Mid
```

`SignalProcessor` 对原始 `(fx, fy, fz)`：

- 一阶低通：`alpha = 0.4`
- 滑移/剪切率：对低通信号做差分，再低通 `derivative_alpha = 0.1`
- 输出：`force_filtered`、`shear_rate`、`normal_force`、`tangential_force`、`slip_ratio = |F_t| / |F_n|`

接触判定：

- `normal_force > 0.1 N` 进入接触；
- 已接触时 `normal_force > 5 N` 保持接触（滞回）。

### 5.3 接触后的力闭环

对每个接触传感器：

```text
f_physics_need = tangential_force / mu_est * safety_margin
f_target       = max(min_force, f_physics_need)
若 slip_ratio > mu_est：f_target += (slip_ratio - mu_est) * 5.0
```

- `mu_est = 0.4`，`safety_margin = 1.5`。
- 切向误差经 `BoundedPID(kp=0.1, ki=0, kd=0, integral_limit=5.0)` 得到切向力指令。
- 力误差送入导纳控制器（`M = 0.11`，`D = 12`，`K = 0`，`max_vel = 0.5`，`max_acc = 10`）。
- 导纳输出位移经 Pinocchio 雅可比 + 阻尼最小二乘（`lambda = 0.05`）求关节增量；只驱动 7 个主动关节，欠驱动关节按 `1.0 / 0.8 / 0.6` 联动。
- 防回退棘轮：当法向力低于 8 N 且导纳速度方向为“回退”时，位移清零。
- 每个 tick 的关节增量限幅 `0.06 rad`。

### 5.4 7 → 16 关节映射

硬件只回传 7 个执行器角度，控制器将其映射为 16 个 URDF 关节角：

```text
q16[0:3] = q7[0:3]          # 拇指 abd / flex / mcp
q16[3]   = q7[2]            # 拇指 ip
每个手指 i（index/middle/ring/pinky）：
  q16[base]   = q7[3+i]
  q16[base+1] = q7[3+i] * 0.8
  q16[base+2] = q7[3+i] * 0.6
```

---

## 6. 线程与并发

| 线程 | 创建位置 | 频率 | 作用 |
| --- | --- | --- | --- |
| 主线程 | `arm_hand_grasp.py` | 流程驱动 | 采集、规划、运动、确认 |
| `GetFeed` | `CR5.connect()` | ~1 ms | 读取 `feedBackData()`，更新 `current_actual`、`TCPForce`、`QActual`、错误状态 |
| `ClearRobotError` | `CR5.connect()` | 5 s | 检查告警，必要时提示并 `ClearError/Continue` |
| `ShearGraspController` | `start()` | 100 Hz | 触觉读取、状态机、导纳控制、串口下发 |

共享状态：

- `CR5.current_actual` 由反馈线程写、主线程读；代码未加锁读取（Python GIL 下列表赋值是原子引用替换，但并非严格同步）。
- `ShearGraspController.current_state` 由 `set_command()` 与 `run()` 通过 `self.lock` 保护。

---

## 7. 数据与日志目录

| 路径 | 内容 | 是否入库 |
| --- | --- | --- |
| `graspnet-baseline/data/apple_data/` | 相机采集，`color/ depth/ viz/` 三套子目录，文件名含 Unix 时间戳 | 忽略 |
| `graspnet-baseline/data/debug_data/` | `latest_color.png`、`latest_depth.png`、`mask_output.png` 等运行调试中间结果 | 忽略 |
| `graspnet-baseline/data/cab_data/` | 手眼标定板图像与 `calibration.txt` | 部分入库 |
| `graspnet-baseline/data/cap_data/` | 抓取数据采集、LabelMe JSON、掩码 | 部分入库 |
| `graspnet-baseline/data/capture_test/` | 两张格式样例图 | 入库 |
| `graspnet-baseline/logs/` | GraspNet 权重与训练日志 | 忽略 |
| `graspnet-baseline/Log/` | Orbbec SDK 运行日志 | 忽略 |

所有相对路径都以 **`graspnet-baseline/` 为工作目录** 解析。

---

## 8. 扩展指引

### 8.1 替换 VLM / 分割模型

- `vlm_process.py::generate_robot_actions()`：修改 `OpenAI(base_url=...)`、`model` 与 Prompt 即可接入其他多模态模型；只需保持返回 `{"response", "coordinates": {"bbox": [x1,y1,x2,y2]}}`。
- SAM 在 `choose_model()` 中初始化，权重路径已按文件位置解析到 `graspnet-baseline/sam_b.pt`。

### 8.2 接入掌心触觉传感器

`loophand/extern_driver/Read_Single_Sensor_Usb.py::PalmTactileSensor` 已实现掌心 9 点阵列读取。接入步骤：

1. 在 `tactile_grasping.py` 中实例化 `PalmTactileSensor`；
2. 在 `SENSOR_TO_LINK_MAP` 中增加掌心 link（如 `left_palm_link`）；
3. 确认 URDF 中存在该 link，并在 `_update_kinematics()` 中处理偏移。

### 8.3 增加任务步骤

主流程目前是硬编码状态序列。若要扩展（例如开门、抽屉），建议新增一个任务调度层，复用已有原语：

- `CR5.MoveCartesian` / `MoveJoints`
- `hand.set_command(...)`
- `targetpose_generation(...)`

实验性的 `vlm_plan.py` + `planner_client.py` 提供了“自然语言 → 技能序列 → socket 任务”的雏形，但服务端不在本仓库。

### 8.4 重新标定

1. 采集标定板图像到 `graspnet-baseline/data/cab_data/<case>/`；
2. 修改 `hand_eye_calibration.py` 的 `cal_path`、棋盘格内角点数和边长；
3. 运行 `python hand_eye_calibration.py`；
4. 将输出的 `RT_cam2gripper` 填入 `process.py::T_FLANGE_CAM`；
5. 更新相机内参与 `T_FLANGE_TCP`。

---

## 9. 常见坑

- `process.py` 在 **import 时**解析命令行参数，`--checkpoint_path` 是必填项。任何新入口脚本都必须传入该参数，否则导入 `process` 会直接退出。
- `process.py` 的内参固定为 1280×720；相机分辨率或对齐方式变化后必须修改。
- `torch.load(cfgs.checkpoint_path)` 未指定 `map_location`，跨设备加载权重时可能需要手动处理。
- `Open3D` 可视化是阻塞的，自动化运行时需要关闭窗口或改为无界面模式。
- `/dev/ttyACM0` 与 `/dev/ttyACM1` 的编号可能随插拔顺序变化，运行前用 `ls -l /dev/ttyACM*` 确认。
- `CR5.current_actual` 初始为 `[-1]`；刚连接时 `wait_until_arrival()` 会等待反馈有效。
- 主流程中 `input()` 与 Open3D 窗口都需要人工交互，不适合无人值守运行。
