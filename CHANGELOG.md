# Changelog

本项目所有重要变更都会记录在此文件中。格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 新增 (Added)

- 新增根目录 `README.md`（项目总览、架构、快速开始）与 `docs/architecture.md`（数据流、坐标帧、线程模型、扩展指引）。
- 新增各模块 README：`arm_CR/README.md`、`graspnet-baseline/README.md`、`loophand/README.md`、`graspnet-baseline/data/README.md`、`graspnet-baseline/legacy/README.md`。
- 新增 VLM + SAM 图像分割模块 `graspnet-baseline/vlm_process.py`：通过自然语言指令定位目标物体，输出分割掩码供 GraspNet 抓取规划使用。
- 新增 VLM 任务规划模块 `graspnet-baseline/vlm_plan.py`（`QwenVLRequester`）与规划器客户端 `graspnet-baseline/planner_client.py`，用于将用户指令转化为技能序列。
- 新增掌心触觉传感器 USB 读取脚本 `loophand/extern_driver/Read_Single_Sensor_Usb.py`（`PalmTactileSensor`）。
- 新增 `graspnet-baseline/legacy/vlm_planner/` 目录（原 `vlm_plan_test/`），保存 VLM 规划流程的历史原型。

### 变更 (Changed)

- **目录结构整理**：相机采集、标定与调试数据统一收拢到 `graspnet-baseline/data/`（`apple_data/`、`cab_data/`、`cap_data/`、`capture_test/`、`debug_data/`）。
- **遗留代码归档**：`graspnet-baseline/vlm_plan_test/` 移动到 `graspnet-baseline/legacy/vlm_planner/`。
- **测试脚本归位**：根目录 `test_hand.py` 移动到 `loophand/examples/test_hand.py`，并补充 `sys.path` 引导，可在任意工作目录运行。
- **上游文档保留**：GraspNet 官方 README 重命名为 `graspnet-baseline/README_GraspNet.md`；Dobot 官方 README 重命名为 `arm_CR/README_Dobot_API.md`，新 `README.md` 用于描述项目自定义部分。
- 抓取主流程 `graspnet-baseline/arm_hand_grasp.py` 改用 VLM/SAM 分割掩码（`segment_image`）替代原有的简单掩码生成 `generate_simple_mask`。
- `graspnet-baseline/demo.py` 数据读取路径改为 `color/`、`depth/` 子目录，并接入 `segment_image` 生成工作空间掩码。
- `graspnet-baseline/gemini335.py` 相机采集保存逻辑改为按 `color/`、`depth/`、`viz/` 分目录存储。
- `graspnet-baseline/requirements.txt` 将 numpy 版本从 1.23.0 升级到 1.23.4。

### 修复 (Fixed)

- 移除 `vlm_plan.py`、`vlm_process.py` 等文件中硬编码的 DashScope API Key，改为通过环境变量 `DASHSCOPE_API_KEY` 读取。
- 数据目录移动后，同步更新 `arm_hand_grasp.py`、`demo.py`、`gemini335.py`、`vlm_process.py`、`planner_client.py`、`hand_eye_calibration.py`、`labelme2mask.py` 中的路径引用。
- `arm_hand_grasp.py`、`tactile_grasping.py`、`examples/test_hand.py` 的 URDF 路径改为基于文件位置解析，避免依赖当前工作目录。
- `check_depth.py` 支持命令行传入深度图路径，默认检查 `data/debug_data/latest_depth.png`。
- `demo.py` 自动选择 `data/apple_data/color/` 中最新一帧，并匹配同时间戳的原始深度图，不再硬编码文件名。
- `vlm_process.py` 与 `legacy/vlm_planner/segment_process.py` 的 `sam_b.pt` 路径改为基于文件位置解析。

### 安全 (Security)

- API Key 不再写入源码，统一由环境变量 `DASHSCOPE_API_KEY` 提供。

### 杂项 (Chore)

- `.gitignore` 同步 `graspnet-baseline/data/` 新布局，继续忽略相机采集、调试掩码等生成物。
- 停止跟踪 `loophand/aero_open_sdk/sensor_comm.log`（约 82 MB 运行日志）。文件保留在本地，但不再进入版本库。
