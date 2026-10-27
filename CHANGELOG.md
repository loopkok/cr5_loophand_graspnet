# Changelog

本项目所有重要变更都会记录在此文件中。格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 新增 (Added)
- 新增 VLM + SAM 图像分割模块 `graspnet-baseline/vlm_process.py`：通过自然语言指令定位目标物体，输出分割掩码供 GraspNet 抓取规划使用。
- 新增 VLM 任务规划模块 `graspnet-baseline/vlm_plan.py`（`QwenVLRequester`）与规划器客户端 `graspnet-baseline/planner_client.py`，用于将用户指令转化为技能序列。
- 新增掌心触觉传感器 USB 读取脚本 `loophand/extern_driver/Read_Single_Sensor_Usb.py`（`PalmTactileSensor`）。
- 新增 `graspnet-baseline/vlm_plan_test/` 目录，用于 VLM 规划流程的测试脚本。

### 变更 (Changed)
- 抓取主流程 `graspnet-baseline/arm_hand_grasp.py` 改用 VLM/SAM 分割掩码（`segment_image`）替代原有的简单掩码生成 `generate_simple_mask`。
- `graspnet-baseline/demo.py` 数据读取路径改为 `color/`、`depth/` 子目录，并接入 `segment_image` 生成工作空间掩码。
- `graspnet-baseline/gemini335.py` 相机采集保存逻辑改为按 `color/`、`depth/`、`viz/` 分目录存储。
- `graspnet-baseline/requirements.txt` 将 numpy 版本从 1.23.0 升级到 1.23.4。

### 修复 (Fixed)
- 移除 `vlm_plan.py`、`vlm_process.py` 等文件中硬编码的 DashScope API Key，改为通过环境变量 `DASHSCOPE_API_KEY` 读取。

### 安全 (Security)
- API Key 不再写入源码，统一由环境变量 `DASHSCOPE_API_KEY` 提供。
