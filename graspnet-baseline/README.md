# graspnet-baseline —— 视觉感知与抓取规划模块

本目录以 [GraspNet-1Billion 官方 baseline](https://github.com/graspnet/graspnet-baseline) 为基础，增加了本项目所需的**相机采集、VLM 语义定位、SAM 分割、手眼变换和整机抓取流程**。

- 上游算法（安装、训练、评测）请见 [README_GraspNet.md](README_GraspNet.md)。
- 项目整体架构与硬件说明请见 [../README.md](../README.md) 和 [../docs/architecture.md](../docs/architecture.md)。

> 说明：项目自定义脚本与上游代码保留在同一层，是为了复用 `process.py` 对 `models/`、`dataset/`、`utils/` 的 `sys.path` 依赖，以及 `./data/`、`logs/` 等相对路径约定。若要迁移到子目录，需要同时调整导入路径和工作目录假设。

---

## 1. 文件分类

### 1.1 本项目新增/修改的核心文件

| 文件 | 作用 |
| --- | --- |
| `arm_hand_grasp.py` | **系统主入口**：初始化 CR5、Gemini 335、触觉手，编排“归位→感知→规划→抓取→放置”流程 |
| `process.py` | 点云构建、GraspNet 推理、抓取筛选、相机系 → 基座系位姿变换（含手眼矩阵） |
| `gemini335.py` | Orbbec Gemini 335 采集封装：彩色/原始深度/热力图/对齐叠加，按 `color/ depth/ viz/` 保存 |
| `vlm_process.py` | Qwen-VL 目标定位 + SAM 分割，输出工作空间掩码；包含可选语音/TTS 代码 |
| `vlm_plan.py` | 用 Qwen-VL 把自然语言任务转成技能编号序列（实验性） |
| `planner_client.py` | 技能序列 → socket 任务的客户端（实验性，服务端不在本仓库） |
| `hand_eye_calibration.py` | 眼在手上（eye-in-hand）标定，输出 `RT_cam2gripper` |
| `check_depth.py` | 深度图格式/范围诊断与可视化 |
| `test_kinematics_simple.py` | 机械臂运动 + 手眼变换的独立验证脚本（会实际移动 CR5，注意安全） |
| `utils.py` | Orbbec 帧格式转换等相机工具函数（注意与上游 `utils/` 目录同名） |
| `demo.py` | 上游 demo 的项目化版本：读取 `data/apple_data` 最新一帧，接入 `segment_image` |

### 1.2 上游 GraspNet 代码

| 路径 | 作用 |
| --- | --- |
| `models/` | GraspNet 网络结构、损失、模块 |
| `dataset/` | 数据集加载、容差标签生成 |
| `utils/` | 碰撞检测、点云工具、标签生成 |
| `pointnet2/`、`knn/` | 需要编译的 CUDA 算子 |
| `graspnetAPI/` | GraspNet 评测 API |
| `train.py`、`test.py` | 训练与评测入口 |
| `command_train.sh`、`command_test.sh` | 上游训练/评测示例命令 |
| `doc/` | 官方示例数据与图片 |

### 1.3 辅助与遗留

| 路径 | 说明 |
| --- | --- |
| `data/` | 采集、标定、调试数据，见 [data/README.md](data/README.md) |
| `legacy/vlm_planner/` | 早期 VLM 规划原型，见 [legacy/README.md](legacy/README.md) |
| `labelme2mask/` | LabelMe 标注转二值掩码 |
| `logs/log_kn/checkpoint.tar` | GraspNet 权重（需自行放置） |
| `sam_b.pt` | SAM-B 权重（需自行放置） |
| `armhand_start.sh` | 主流程启动脚本 |

---

## 2. 安装

完整的 GraspNet 环境（PyTorch、CUDA 算子、graspnetAPI）见 [README_GraspNet.md](README_GraspNet.md)。简要步骤：

```bash
cd graspnet-baseline
pip install -r requirements.txt

cd pointnet2 && python setup.py install
cd ../knn && python setup.py install
cd ../graspnetAPI && pip install .
```

此外还需要：

```bash
pip install openai ultralytics pyserial \
            sounddevice soundfile pydub pyttsx3 \
            open3d numpy scipy pillow tqdm
# 相机 SDK
# pyorbbecsdk：按 Orbbec 官方文档安装
```

### 权重与 Key

```bash
# GraspNet 权重
ls logs/log_kn/checkpoint.tar

# SAM 权重
ls sam_b.pt

# VLM Key
export DASHSCOPE_API_KEY="sk-xxxxxxxx"
```

`sam_b.pt` 的路径在 `vlm_process.py::choose_model()` 中基于文件位置解析，因此从其他目录调用也能找到；GraspNet 权重仍通过命令行 `--checkpoint_path` 传入。

---

## 3. 运行

> 所有命令都在 **`graspnet-baseline/`** 目录下执行。代码中的 `data/`、`logs/`、`models/` 等路径均相对此目录。

### 3.1 整机抓取主流程

```bash
export DASHSCOPE_API_KEY="sk-xxxxxxxx"
bash armhand_start.sh
```

脚本内容：

```bash
CUDA_VISIBLE_DEVICES=0 python arm_hand_grasp.py \
    --checkpoint_path logs/log_kn/checkpoint.tar
```

交互步骤：

1. 终端输入目标描述（如“抓那个红色苹果”）；
2. 等待 Qwen-VL + SAM 生成掩码；
3. 关闭 Open3D 抓取可视化窗口；
4. 输入 `y` 执行抓取，其他输入回到归位。

### 3.2 离线抓取可视化

```bash
python demo.py --checkpoint_path logs/log_kn/checkpoint.tar
```

`demo.py` 会自动选择 `data/apple_data/color/` 中时间戳最新的一张彩色图，并匹配 `data/apple_data/depth/` 下同时间戳的 `*_depth_raw.png`。

### 3.3 相机预览与采集

```bash
python gemini335.py
```

四个窗口分别显示彩色、原始深度、深度热力图、对齐叠加图。按 `s` 保存一帧到 `data/apple_data/{color,depth,viz}/`，按 `q`/`Esc` 退出。

### 3.4 深度图诊断

```bash
python check_depth.py [path/to/depth_raw.png]
```

不传参数时检查 `data/debug_data/latest_depth.png`。

### 3.5 手眼标定

```bash
python hand_eye_calibration.py
```

默认读取 `data/cab_data/cab1`。请先根据实际棋盘格修改 `size=(11, 8)`、`size_step`、`img_num` 和 `cal_path`。标定结果需手动填入 `process.py::T_FLANGE_CAM`。

### 3.6 运动学/手眼变换验证

```bash
python test_kinematics_simple.py
```

该脚本会连接 CR5、依次移动到 `HOME_POSE` 和 `TEST_POSE`，并验证手眼变换链。**会真实驱动机械臂**，运行前请确认工作空间安全。注意其 `T_FLANGE_TCP` 当前值与 `process.py` 中的值不一致，使用前请按实际工具标定统一。

### 3.7 实验性 VLM 任务规划

```bash
python planner_client.py
```

该客户端默认连接 `127.0.0.1:12345` 的任务服务，并在 `12346` 等待执行完成信号；服务端不在本仓库，因此该脚本目前不能独立跑通，仅保留作后续扩展参考。

---

## 4. 主流程模块关系

```text
arm_hand_grasp.py
 ├─ gemini335.Gemini335          # RGB-D
 ├─ vlm_process.segment_image    # Qwen-VL + SAM → mask
 ├─ process.targetpose_generation# GraspNet + 手眼变换 → 基座系位姿
 ├─ arm_CR.cr5.CR5               # ServoP/ServoJ
 └─ loophand.tactile_grasping    # 触觉自适应抓取
```

`arm_hand_grasp.py` 通过以下方式把仓库根目录、`arm_CR/`、`loophand/` 加入 `sys.path`：

```python
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.extend([project_root, .../arm_CR, .../loophand])
```

因此跨模块导入不依赖安装成包。

---

## 5. 关键配置

| 位置 | 参数 | 说明 |
| --- | --- | --- |
| `arm_hand_grasp.py` | `HOME_POSE` / `PLACE_POSE` | 归位点、放置点，`[x,y,z,rx,ry,rz]`（mm/度） |
| `arm_hand_grasp.py` | `PRE_GRASP_DIST` | 预抓取回退距离（mm） |
| `arm_hand_grasp.py` | `VEL_RATIO` / `ACC_RATIO` | CR5 速度/加速度百分比 |
| `arm_hand_grasp.py` | `SAVE_DIR` | 调试输出目录，默认 `./data/debug_data` |
| `process.py` | `T_FLANGE_CAM` | 手眼矩阵（相机 → 法兰，即 `calibrateHandEye` 的 cam2gripper 输出） |
| `process.py` | `T_FLANGE_TCP` | 法兰 → 掌心 TCP |
| `process.py` | 内参矩阵 / `factor_depth` | 默认 1280×720，深度 mm |
| `process.py` | `--num_point` / `--collision_thresh` / `--voxel_size` | GraspNet 推理与碰撞参数 |
| `process.py` | 进给角 `< 30°`、闭合角 `< 90°`、向心权重 `2.0` | 抓取筛选 |
| `vlm_process.py` | `model="qwen-vl-plus"` | VLM 模型名 |
| `vlm_process.py` | `DASHSCOPE_API_KEY` | 环境变量，不写入源码 |

> `process.py` 在模块导入时就会解析 `--checkpoint_path`（必填）。新增入口脚本时也必须传入该参数。

---

## 6. 数据目录

| 路径 | 内容 |
| --- | --- |
| `data/apple_data/` | 相机采集：`color/`、`depth/`、`viz/` |
| `data/debug_data/` | 运行中间结果：`latest_color.png`、`latest_depth.png`、`mask_output.png` |
| `data/cab_data/` | 手眼标定板图像与 `calibration.txt` |
| `data/cap_data/` | 数据采集、LabelMe JSON、掩码 |
| `data/capture_test/` | 格式样例（`capture_color.png` / `capture_depth.png`） |

详见 [data/README.md](data/README.md)。

---

## 7. 已知问题

- `arm_hand_grasp.py` 中保留未使用的 `generate_simple_mask()`；当前实际使用 `segment_image()`。
- `process.py` 的相机内参与 `T_FLANGE_CAM` 是当前设备的标定值，换设备必须重标。
- `vlm_process.py` 默认使用键盘输入目标描述，语音识别分支（Whisper）被注释；TTS 依赖 `pyttsx3`，失败时仅打印警告。
- `demo.py` 依赖 `data/apple_data/color`、`data/apple_data/depth` 的命名约定；缺少成对文件会抛 `FileNotFoundError`。
- `vlm_plan.py` / `planner_client.py` 依赖外部任务服务，未接入主流程。
- Open3D 可视化与 `input()` 都会阻塞流程，不适合无人值守。
