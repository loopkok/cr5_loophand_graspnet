# data —— 采集、标定与调试数据

所有路径均相对 `graspnet-baseline/` 目录。该目录中的大部分内容体积较大或属于运行产物，已在 `.gitignore` 中忽略。

| 子目录 | 产生方式 | 内容与命名 |
| --- | --- | --- |
| `apple_data/` | `gemini335.py` 按 `s` 保存 | `color/capture_<ts>_color.png`、`depth/capture_<ts>_depth_raw.png`（uint16，mm）、`viz/capture_<ts>_depth_viz.png`；`<ts>` 为 Unix 时间戳 |
| `debug_data/` | `arm_hand_grasp.py` / `vlm_process.py` 运行时写入 | `latest_color.png`、`latest_depth.png`、`mask_output.png` 等最新一帧中间结果 |
| `cab_data/` | 手眼标定采集 | 每场景一个子目录（如 `cab0/`），含 `corner_*.png`、`img_*.png`、`calibration.txt` |
| `cap_data/` | 抓取数据采集 / LabelMe 标注 | 原始采集图、`labelme_jsons/`、`masks/`、`test/` |
| `capture_test/` | 手动放置 | `capture_color.png`、`capture_depth.png` 格式样例 |

## 使用约定

- 主流程读取的实时帧写入 `debug_data/`；`demo.py` 读取 `apple_data/` 中最新一帧（要求 `color/` 与 `depth/` 下存在同时间戳的 `*_color.png` 和 `*_depth_raw.png`）。
- 深度图必须保存为 **16 位 PNG**（`uint16`，单位 mm），不要用 JPEG 或有损压缩，否则精度会被破坏。
- 手眼标定输出 `data.npy` 与 `calibration.txt` 位于 `cab_data/<case>/`；标定结果需要手动填回 `process.py`。
- `labelme2mask` 默认读取 `cap_data/labelme_jsons/` 并写入 `cap_data/masks/`。

重新采集数据时，保持上述目录与文件名约定即可直接复用现有代码。
