# CR5 + Aero Hand 触觉智能抓取系统

基于 **Dobot CR5 机械臂**、**Orbbec Gemini 335 深度相机** 和 **Aero Hand Open 触觉灵巧手** 的视觉引导抓取系统。系统使用 **Qwen-VL + SAM** 从自然语言指令中定位目标物体，使用 **GraspNet** 生成 6-DoF 抓取位姿，并通过触觉反馈 + 导纳控制完成自适应闭合抓取。

> 本仓库把三套硬件/算法栈组合成一个完整应用：
> 机械臂控制（`arm_CR/`）+ 视觉抓取规划（`graspnet-baseline/`）+ 触觉灵巧手控制（`loophand/`）。

---

## 1. 系统组成

| 模块 | 目录 | 职责 |
| --- | --- | --- |
| 机械臂控制 | [`arm_CR/`](arm_CR/) | 封装 Dobot CR5 TCP/IP 通信、实时反馈、笛卡尔/关节运动；提供导纳控制实验 |
| 视觉抓取 | [`graspnet-baseline/`](graspnet-baseline/) | 相机采集、VLM 语义定位、SAM 分割、GraspNet 抓取位姿生成、手眼变换、主流程编排 |
| 触觉灵巧手 | [`loophand/`](loophand/) | Aero Hand Open SDK、触觉传感器驱动、PID + 导纳 + Pinocchio 逆运动学的自适应抓取控制器 |

各模块的详细说明见对应目录下的 `README.md`。

---

## 2. 整体架构

```text
┌──────────────────────────────────────────────────────────────────────┐
│                          arm_hand_grasp.py                           │
│                     主流程编排（GraspSystem）                         │
└───────┬───────────────────┬────────────────────┬─────────────────────┘
        │                   │                    │
        ▼                   ▼                    ▼
┌───────────────┐   ┌───────────────────┐   ┌───────────────────────┐
│  gemini335.py │   │   vlm_process.py  │   │ tactile_grasping.py   │
│  RGB-D 采集   │   │ Qwen-VL + SAM     │   │ 触觉自适应抓取控制    │
└───────┬───────┘   └─────────┬─────────┘   └──────────┬────────────┘
        │                     │                        │
        │  color / depth      │  object mask           │ force / joint
        ▼                     ▼                        ▼
┌─────────────────────────────────────────┐   ┌───────────────────────┐
│              process.py                 │   │  AeroHand + Tactile   │
│ 点云构建 → GraspNet 推理 → 位姿筛选      │   │  Sensor 驱动          │
│ → 相机系抓取 → 基座系目标位姿（手眼变换）│   └───────────────────────┘
└───────────────────┬─────────────────────┘
                    │ target pose (base frame)
                    ▼
        ┌───────────────────────┐
        │        cr5.py         │
        │ ServoP / ServoJ 控制  │
        └───────────┬───────────┘
                    ▼
              Dobot CR5 机械臂
```

### 单次抓取时序

1. **归位**：CR5 移动到 `HOME_POSE`，灵巧手切换 `IDLE`。
2. **感知**：Gemini 335 采集对齐后的 RGB-D 图像。
3. **语义定位**：Qwen-VL 根据用户输入返回目标 `bbox`，SAM（`sam_b.pt`）生成像素级掩码。
4. **抓取规划**：掩码内点云送入 GraspNet，经碰撞检测、进给方向/闭合方向/向心筛选后得到最佳抓取。
5. **坐标变换**：相机系抓取位姿 → 基座系末端目标位姿（使用手眼矩阵、法兰-TCP 矩阵和当前机械臂位姿）。
6. **人工确认**：Open3D 可视化最佳抓取，终端输入 `y` 后执行。
7. **预抓取**：移动到目标点上方 `PRE_GRASP_DIST` 处，手切换到 `PRE_SHAPE`。
8. **插入 + 抓取**：低速移动到目标位姿，手切换 `GRASP`，触觉闭环稳定 3 秒。
9. **搬运/放置**：抬起、移动到 `PLACE_POSE`、`RELEASE`，进入下一轮。

更详细的坐标变换、线程模型和状态机说明见 [docs/architecture.md](docs/architecture.md)。

---

## 3. 目录结构

```text
cr5_loophand_graspnet/
├── README.md                    # 本文件：项目总览与快速开始
├── CHANGELOG.md                 # 变更记录
├── docs/
│   └── architecture.md          # 详细架构、数据流、坐标帧与扩展点
├── arm_CR/                      # 机械臂控制模块
│   ├── README.md                # 模块说明（项目自定义部分）
│   ├── README_Dobot_API.md      # Dobot 官方 SDK 文档（中文，原 README）
│   ├── README-EN.md             # Dobot 官方 SDK 文档（英文）
│   ├── cr5.py                   # CR5 封装：连接、反馈线程、ServoP/ServoJ、SE3 转换
│   ├── dobot_api.py             # Dobot TCP/IP 官方 Python SDK
│   ├── admittanccontroller.py   # 机械臂笛卡尔导纳控制器
│   ├── virtualforcecontroller.py# 键盘虚拟力输入
│   ├── force_plotter.py         # 六维力实时曲线
│   ├── real_plot.py             # 通用 Tkinter 实时绘图
│   ├── test_admittance_control.py # 机械臂导纳控制示例
│   ├── main.py / main_UI.py / ui.py # 官方连接与 GUI 示例
│   └── files/ images/           # 告警码表、文档图片
├── graspnet-baseline/           # 视觉感知与抓取规划模块
│   ├── README.md                # 模块说明（项目集成部分）
│   ├── README_GraspNet.md       # GraspNet 官方文档（安装/训练/评测）
│   ├── arm_hand_grasp.py        # ★ 系统主流程
│   ├── process.py               # ★ 点云 + GraspNet + 手眼变换
│   ├── gemini335.py             # ★ Gemini 335 相机采集
│   ├── vlm_process.py           # ★ Qwen-VL + SAM 目标分割
│   ├── vlm_plan.py              # VLM 技能序列生成（实验性）
│   ├── planner_client.py        # 技能序列 socket 客户端（实验性）
│   ├── hand_eye_calibration.py  # 眼在手标定
│   ├── check_depth.py           # 深度图诊断工具
│   ├── demo.py                  # 离线抓取可视化示例（已接入 SAM）
│   ├── data/                    # 采集与调试数据（见 data/README.md）
│   ├── legacy/vlm_planner/      # 早期 VLM 规划原型（见 legacy/README.md）
│   ├── models/ dataset/ utils/  # GraspNet 网络与数据
│   ├── pointnet2/ knn/          # 需编译的 CUDA 算子
│   ├── graspnetAPI/             # GraspNet 评测 API
│   ├── labelme2mask/            # LabelMe 标注转掩码工具
│   ├── doc/                     # 官方示例数据与图片
│   ├── logs/log_kn/checkpoint.tar # GraspNet 权重（需自行放置）
│   └── sam_b.pt                 # SAM 权重（需自行放置）
├── loophand/                    # 触觉灵巧手模块
│   ├── README.md                # 模块说明
│   ├── tactile_grasping.py      # ★ 触觉自适应抓取控制器
│   ├── aero_open_sdk/           # Aero Hand Open 官方 SDK
│   ├── controller/              # PID 与导纳控制器
│   ├── extern_driver/           # 触觉传感器串口驱动
│   ├── examples/test_hand.py    # 早期独立手部测试脚本
│   └── urdf/                    # Aero Hand 左手指 URDF
```

> 标 ★ 的是本项目自行开发/集成的核心文件；其余多为上游 SDK 或算法原始代码。

---

## 4. 硬件与连接

| 硬件 | 连接方式 | 默认地址/端口 |
| --- | --- | --- |
| Dobot CR5 机械臂 | 以太网 TCP/IP | `192.168.5.1`；Dashboard `29999`，运动 `30003`，反馈 `30004` |
| Orbbec Gemini 335 | USB 3.0 | 由 `pyorbbecsdk` 枚举；深度对齐到彩色流 |
| Aero Hand Open（左手） | USB 串口 | `/dev/ttyACM1`，921600 baud |
| 手部触觉传感器总线 | USB 串口 | `/dev/ttyACM0`，921600 baud |
| 掌心触觉传感器（可选） | USB 串口 | 由 `Read_Single_Sensor_Usb.py` 自动枚举，未接入主流程 |
| CR5 六维力传感器（可选） | 机械臂反馈 | 用于 `arm_CR/` 导纳实验 |

使用前请确认：

```bash
ping 192.168.5.1
ls /dev/ttyACM*
```

串口权限不足时执行：

```bash
sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1
```

> CR5 的 IP、端口目前硬编码在 `arm_CR/cr5.py` 与 `arm_CR/main.py` 中；如现场网段不同，需要修改。

---

## 5. 环境与依赖

推荐使用 **Python 3.8 + CUDA** 环境（GraspNet 原始代码基于 PyTorch 1.x 编写）。

### 5.1 GraspNet 核心依赖

```bash
cd graspnet-baseline
pip install -r requirements.txt
```

### 5.2 需要编译的算子

```bash
cd graspnet-baseline/pointnet2 && python setup.py install
cd ../knn && python setup.py install
cd ../graspnetAPI && pip install .
```

### 5.3 本系统额外依赖

```bash
pip install openai ultralytics torch torchvision \
            open3d pyserial numpy scipy pillow tqdm \
            sounddevice soundfile pydub pyttsx3 \
            spatialmath-python pynput matplotlib
# Pinocchio（用于手部运动学）：
conda install -c conda-forge pinocchio
# 或 pip install pin
```

`pyorbbecsdk` 需按 Orbbec 官方 SDK 文档安装（通常随相机 SDK 提供，不在 PyPI 上直接安装）。

> 注意：`vlm_process.py` 中的语音识别（Whisper）代码默认被注释，当前使用键盘输入目标描述；TTS 使用 `pyttsx3` 播报 VLM 的自然语言回复。

### 5.4 模型权重

以下文件体积较大，未纳入版本管理，需要手动放置：

| 文件 | 用途 |
| --- | --- |
| `graspnet-baseline/logs/log_kn/checkpoint.tar` | GraspNet Kinect 预训练权重 |
| `graspnet-baseline/sam_b.pt` | Ultralytics SAM-B 权重 |

GraspNet 预训练权重下载方式见 [graspnet-baseline/README_GraspNet.md](graspnet-baseline/README_GraspNet.md)。

### 5.5 API Key

VLM 调用使用阿里云 DashScope 的 OpenAI 兼容接口，必须通过环境变量提供 Key：

```bash
export DASHSCOPE_API_KEY="sk-xxxxxxxx"
```

代码中不再硬编码 API Key。

---

## 6. 快速开始

### 6.1 运行完整抓取流程

```bash
export DASHSCOPE_API_KEY="sk-xxxxxxxx"
cd graspnet-baseline
bash armhand_start.sh
```

等价于：

```bash
CUDA_VISIBLE_DEVICES=0 python arm_hand_grasp.py \
    --checkpoint_path logs/log_kn/checkpoint.tar
```

运行后：

1. 按提示输入要抓的物体描述（例如“抓那个红色的苹果”）。
2. 程序调用 Qwen-VL + SAM 生成掩码。
3. 弹出 Open3D 窗口显示点云与最佳抓取（红色为最佳，绿色为候选），**关闭窗口**后继续。
4. 终端输入 `y` 执行抓取，输入其他内容则回到归位步骤。

> 必须在 `graspnet-baseline/` 目录下运行：代码中的 `./data/...`、`logs/...`、`models/` 等路径均相对于该目录。

### 6.2 分模块调试

| 目的 | 命令（在对应目录下运行） |
| --- | --- |
| 相机实时预览 / 保存数据 | `cd graspnet-baseline && python gemini335.py`（按 `s` 保存到 `data/apple_data`） |
| 深度图检查 | `cd graspnet-baseline && python check_depth.py [depth_raw.png]` |
| 离线抓取可视化 | `cd graspnet-baseline && python demo.py --checkpoint_path logs/log_kn/checkpoint.tar` |
| 手眼标定 | `cd graspnet-baseline && python hand_eye_calibration.py` |
| 灵巧手交互测试 | `cd loophand && python tactile_grasping.py` |
| 灵巧手早期测试脚本 | `cd loophand && python examples/test_hand.py` |
| 机械臂导纳控制 | `cd arm_CR && python test_admittance_control.py` |
| 掌心触觉传感器 | `cd loophand && python extern_driver/Read_Single_Sensor_Usb.py` |

---

## 7. 关键配置项

运行前务必根据现场情况检查以下参数：

| 位置 | 参数 | 说明 |
| --- | --- | --- |
| `graspnet-baseline/arm_hand_grasp.py` | `HOME_POSE` / `PLACE_POSE` | 机械臂归位点、放置点（mm + 度） |
| `graspnet-baseline/arm_hand_grasp.py` | `PRE_GRASP_DIST` | 预抓取距离（mm） |
| `graspnet-baseline/arm_hand_grasp.py` | `VEL_RATIO` / `ACC_RATIO` | 机械臂速度/加速度百分比 |
| `graspnet-baseline/process.py` | `T_FLANGE_CAM` | 眼在手上的手眼矩阵（需标定） |
| `graspnet-baseline/process.py` | `T_FLANGE_TCP` | 法兰盘 → 工具中心点（掌心）变换 |
| `graspnet-baseline/process.py` | 相机内参 / `factor_depth` | 默认 1280×720、深度单位 mm |
| `graspnet-baseline/process.py` | 抓取筛选阈值 | 进给角 < 30°、闭合角 < 90°、向心惩罚权重 |
| `graspnet-baseline/vlm_process.py` | `model="qwen-vl-plus"` | VLM 模型名 |
| `loophand/tactile_grasping.py` | `port_hand` / `port_tactile` | 手与触觉传感器串口 |
| `loophand/tactile_grasping.py` | `mu_est` / `min_force` / 导纳参数 | 抓取力与柔顺参数 |
| `arm_CR/cr5.py` | IP 与端口 | CR5 网络配置 |

---

## 8. 已知限制与实验性内容

- **主流程只保留一条固定抓取序列**：归位 → 抓取 → 放置，暂无多物体/多步任务调度。
- `vlm_plan.py` + `planner_client.py` 是 VLM 技能序列规划的实验性链路，依赖外部 socket 服务（默认 `127.0.0.1:12345`），仓库内未包含该服务，未接入主流程。
- `legacy/vlm_planner/` 是早期原型，与 `vlm_process.py` 有重复代码，仅作参考。
- `loophand/examples/test_hand.py` 是 `tactile_grasping.py` 的早期独立版本，参数略有差异（例如 `min_force`）。
- `loophand/extern_driver/Read_Single_Sensor_Usb.py`（掌心传感器）目前是独立工具，尚未接入触觉抓取控制器。
- 相机内参、手眼矩阵、TCP 变换均为当前设备的标定结果，更换相机/手/安装方式后必须重新标定。
- `vlm_process.py` 的语音识别分支默认关闭，当前为键盘输入。
- `arm_hand_grasp.py` 中保留了一个未使用的 `generate_simple_mask()` 历史函数。

---

## 9. 文档与变更

- [docs/architecture.md](docs/architecture.md)：模块边界、数据流、坐标帧、线程与状态机、扩展指引。
- [graspnet-baseline/README.md](graspnet-baseline/README.md)：视觉与抓取模块。
- [loophand/README.md](loophand/README.md)：触觉手模块。
- [arm_CR/README.md](arm_CR/README.md)：机械臂模块。
- [CHANGELOG.md](CHANGELOG.md)：变更记录。

---

## 10. 致谢与许可

- GraspNet-1Billion / graspnet-baseline：MVIG, SJTU（见 `graspnet-baseline/README_GraspNet.md` 与 `LICENSE`）。
- Dobot TCP-IP-Python-V3：Dobot 官方 SDK（见 `arm_CR/README_Dobot_API.md` 与 `arm_CR/LICENSE`）。
- Aero Hand Open SDK：TetherIA, Inc.（Apache-2.0，见 `loophand/aero_open_sdk/` 文件头）。

各上游组件的许可协议以对应目录中的 `LICENSE` 为准。
