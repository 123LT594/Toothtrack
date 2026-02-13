"""
深度图 Ground Truth 验证
方法：
1. 用2D标注点做PnP得到位姿
2. 采样牙齿模型上的点，用该位姿投影到2D
3. 读取深度图在投影位置的深度值
4. 比较：投影点的深度 vs 深度图深度
"""

import os
import json
import numpy as np
import cv2
import trimesh

# 配置
DATA_DIR = "./demo_data/tooth_gt"
DEPTH_DIR = "/workspace/output_depth"
N_SAMPLES = 100  # 每帧采样点数

def main():
    # 加载数据
    K = np.loadtxt(f"{DATA_DIR}/cam_K.txt", dtype=np.float64)
    
    with open(f"{DATA_DIR}/annotations.json", 'r') as f:
        data = json.load(f)
    annotations = data.get("annotations", data)
    
    with open(f"{DATA_DIR}/keypoints_3d_model.json", 'r') as f:
        keypoints_3d = np.array(json.load(f)['keypoints_3d_original'], dtype=np.float64)
    
    # 加载牙齿模型
    mesh = trimesh.load(f"{DATA_DIR}/mesh/tooth.obj")
    vertices = mesh.vertices
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # 收集深度误差
    depth_errors = []
    reproj_errors = []
    valid_count = 0
    
    for depth_file in sorted(os.listdir(DEPTH_DIR)):
        if not depth_file.endswith('.npy'):
            continue
        
        frame_name = depth_file.replace('.npy', '.png')
        if frame_name not in annotations:
            continue
        
        # 加载深度图
        depth = np.load(os.path.join(DEPTH_DIR, depth_file))
        h, w = depth.shape
        
        # PnP解算位姿
        ball_2d = annotations[frame_name]
        points_2d = np.array([ball_2d[f'ball_{i}'] for i in range(1, 5)], dtype=np.float64)
        
        result = cv2.solvePnP(keypoints_3d, points_2d, K, None, flags=cv2.SOLVEPNP_AP3P)
        if len(result) == 4:
            _, rvec, tvec = result
        else:
            _, rvec, tvec = result
        
        # 计算位姿矩阵
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.flatten()
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = t
        
        # 采样牙齿模型顶点
        indices = np.random.choice(len(vertices), min(N_SAMPLES, len(vertices)), replace=False)
        samples = vertices[indices]
        
        # 投影到2D
        x, y, z = samples[:, 0], samples[:, 1], samples[:, 2]
        u = x / z * fx + cx
        v = y / z * fy + cy
        
        # 筛选在图像范围内且深度有效的点
        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (z > 0)
        
        u_valid = u[valid]
        v_valid = v[valid]
        z_valid = z[valid]
        
        for i in range(len(u_valid)):
            ui, vi = int(round(u_valid[i])), int(round(v_valid[i]))
            depth_at = depth[vi, ui]
            
            if depth_at > 0:
                error = abs(depth_at - z_valid[i]) * 1000  # mm
                depth_errors.append(error)
                valid_count += 1
        
        valid_frames = 1
    
    # 输出结果
    print(f"有效帧数: {valid_frames}")
    print(f"总检测点数: {valid_count}")
    print()
    
    if depth_errors:
        errors = np.array(depth_errors)
        print(f"深度误差:")
        print(f"  平均: {np.mean(errors):.4f} mm")
        print(f"  中位数: {np.median(errors):.4f} mm")
        print(f"  标准差: {np.std(errors):.4f} mm")
        print(f"  最大值: {np.max(errors):.4f} mm")

if __name__ == "__main__":
    main()
