# legacy —— 历史原型

这里保存不再参与主流程、但仍有参考价值的早期实现。

## `vlm_planner/`

早期版本的“VLM 技能序列规划”原型，与当前的 `vlm_process.py` + `arm_hand_grasp.py` 并存，但**不被主流程引用**。

| 文件 | 说明 |
| --- | --- |
| `vlm_plan.py` | `QwenVLRequester`：向 Qwen-VL 请求技能编号序列（`##skill01(...)##skill02##...`） |
| `planner.py` | `GraspPlanner`：把技能映射为分割、抓取检测、移动、抓/放、回零等动作；大量依赖外部 ROS/服务，部分代码已注释 |
| `planner_client.py` | socket 客户端：连接外部任务服务（`12345`），等待执行完成信号（`12346`） |
| `segment_process.py` | 早期版本的 VLM + SAM 分割脚本，是当前 `vlm_process.py` 的前身 |

### 与当前实现的差异

- 当前主流程直接在 Python 进程内顺序调用感知与执行函数，不经过 socket 服务。
- 当前的语义定位只返回 `bbox` 并交给 SAM；早期 `vlm_plan.py` 还要求模型输出技能序列。
- 早期 `segment_process.py` 与实际主流程的掩码接口基本一致，但保留下载/调试用的独立入口。

### 注意

- 这些脚本引用的外部服务、IP、ROS 环境和数据路径都需要按实际情况修改后才能运行。
- 依赖路径已更新为新的 `data/debug_data/` 目录；`sam_b.pt` 按文件位置解析到 `graspnet-baseline/sam_b.pt`。
- 新功能请勿继续在此目录开发，优先扩展 `vlm_process.py`。
