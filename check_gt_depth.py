import os
import numpy as np

def analyze_per_frame_gt_depth():
    # 替换为你实际的 wxb 数据集深度图目录
    gt_depth_dir = "/root/lanyun-tmp/golden_dataset/depth"
    
    if not os.path.exists(gt_depth_dir):
        print(f"❌ 找不到目录: {gt_depth_dir}")
        return
        
    depth_files = [f for f in os.listdir(gt_depth_dir) if f.endswith('.npy')]
    # 确保按帧号顺序排列，方便对齐比对
    depth_files.sort()
    
    print(f"📂 找到 {len(depth_files)} 个 GT 深度文件，开始逐帧物理范围分析...\n")
    print(f"{'帧名称 (Frame)':<15} | {'最小深度 (Min)':<15} | {'最大深度 (Max)':<15} | {'平均深度 (Mean)':<15}")
    print("-" * 70)
    
    for f in depth_files:
        d_np = np.load(os.path.join(gt_depth_dir, f))
        valid_mask = d_np > 0
        
        if valid_mask.sum() > 0:
            valid_depths = d_np[valid_mask]
            d_min = valid_depths.min()
            d_max = valid_depths.max()
            d_mean = valid_depths.mean()
            
            # 逐帧打印输出
            print(f"{f:<15} | {d_min:<15.4f} | {d_max:<15.4f} | {d_mean:<15.4f}")
        else:
            print(f"{f:<15} | {'无有效深度 (全黑)':<15} | {'-':<15} | {'-':<15}")

if __name__ == "__main__":
    analyze_per_frame_gt_depth()