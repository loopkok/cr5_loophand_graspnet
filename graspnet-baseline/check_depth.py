import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

def inspect_depth(depth_path):
    # 1. 读取图片 (必须以 -1 或 cv2.IMREAD_UNCHANGED 读取)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
    if depth is None:
        print("❌ 错误：无法读取图片，请检查路径。")
        return

    # 2. 检查数据类型 (最关键！)
    print(f"--- 深度图体检报告 ---")
    print(f"文件路径: {depth_path}")
    print(f"尺寸 (H, W): {depth.shape}")
    print(f"数据类型: {depth.dtype}")  # 必须是 uint16
    
    # 3. 检查数值范围
    # 过滤掉 0 (无效点) 后再看统计
    valid_pixels = depth[depth > 0]
    if len(valid_pixels) == 0:
        print("❌ 警告：图片全是黑的 (所有值为0)！")
        return

    min_val = np.min(valid_pixels)
    max_val = np.max(valid_pixels)
    mean_val = np.mean(valid_pixels)
    
    print(f"有效深度最小值: {min_val} (代表 {min_val}mm)")
    print(f"有效深度最大值: {max_val} (代表 {max_val}mm)")
    print(f"有效深度平均值: {mean_val:.1f}")

    # 4. 判定标准
    print(f"--- 诊断结果 ---")
    if depth.dtype != np.uint16:
        print("❌ 格式错误：不是 uint16 格式！你可能保存成了 8位图 (jpg/png)。")
        print("   -> 导致原因：cv2.imwrite 时没存对，或者被其他软件转码了。")
    elif max_val < 255:
        print("⚠️ 数值可疑：最大值小于 255。")
        print("   -> 可能原因：虽然是16位图，但数值被压缩到了 0-255 范围，单位丢失。")
    elif max_val > 10000:
        print("⚠️ 存在飞点：最大值超过 10米 (10000mm)。")
        print("   -> 建议：这就是导致你内存爆炸的原因，需要进行去噪处理。")
    else:
        print("✅ 格式正常：是 16位图，且数值在合理范围 (桌面场景通常 200-2000mm)。")

    # 5. 可视化 (带颜色条，方便鼠标取点查看)
    plt.figure(figsize=(10, 6))
    plt.title(f"Depth Map Visualization (Min:{min_val}mm, Max:{max_val}mm)")
    # 限制显示范围在 0~2000mm，超过的会变色，方便看细节
    plt.imshow(depth, cmap='jet', vmin=0, vmax=2000) 
    plt.colorbar(label='Depth (mm)')
    plt.show()

# 运行检查：python check_depth.py [depth_raw.png]
if __name__ == '__main__':
    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'debug_data', 'latest_depth.png')
    inspect_depth(sys.argv[1] if len(sys.argv) > 1 else default_path)