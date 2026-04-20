import os
import numpy as np
import cv2
from tqdm import tqdm

# ================= 配置区 =================
# 深度图所在路径
DEPTH_DIR = "./golden_dataset/depth"
# 可视化输出路径
OUT_DIR = "./inference_results/vis_depth_check"

os.makedirs(OUT_DIR, exist_ok=True)
# ==========================================

def main():
    npy_files = [f for f in os.listdir(DEPTH_DIR) if f.endswith('.npy')]
    print(f"🔍 正在转换 {len(npy_files)} 张深度图进行灰度可视化 (近亮远暗)...")

    for f in tqdm(npy_files):
        # 读取深度图 (float16 或 float32 均可)
        depth = np.load(os.path.join(DEPTH_DIR, f))
        
        # 1. 创建掩码（深度大于0的有效区域）
        mask = depth > 0
        if not np.any(mask):
            continue
            
        # 2. 获取有效区域的最值
        d_min = depth[mask].min()
        d_max = depth[mask].max()
        
        # 3. 创建全黑的 8 位灰度底图（背景保持纯黑）
        depth_gray = np.zeros_like(depth, dtype=np.uint8)
        
        # 4. 仅对有牙齿的区域进行归一化映射 (近亮远暗)
        if d_max > d_min:
            # 核心修改：深度越小(近)，减去d_min后越接近0，255减去0就越接近255(白)
            # 深度越大(远)，减去d_min后越接近d_max，算出来接近255，255减去255就越接近0(黑)
            normalized_values = 255 - ((depth[mask] - d_min) / (d_max - d_min) * 255)
            
            depth_gray[mask] = normalized_values.astype(np.uint8)
            
        # 5. 保存灰度图片 (单通道)
        cv2.imwrite(os.path.join(OUT_DIR, f.replace('.npy', '.png')), depth_gray)

    print(f"✅ 灰度可视化完成！请检查文件夹: {OUT_DIR}")

if __name__ == "__main__":
    main()