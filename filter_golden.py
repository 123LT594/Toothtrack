import os
import cv2
import json
import numpy as np
import trimesh
import shutil
from tqdm import tqdm

# ================= 配置区 =================
SRC_DIR = "./demo_data/tooth_gt"
DST_DIR = "../lanyun-tmp/golden_dataset"

MESH_DIR = os.path.join(SRC_DIR, "mesh")
K_PATH = os.path.join(SRC_DIR, "cam_K.txt")
ANN_PATH = os.path.join(SRC_DIR, "annotations.json")
RGB_SRC = os.path.join(SRC_DIR, "rgb")
DEPTH_SRC = os.path.join(SRC_DIR, "depth")
POSE_SRC = os.path.join(SRC_DIR, "pose")

# 筛选阈值 (单位: 像素, 540p 尺度)
ERROR_THRESHOLD = 1.5

# 只创建训练真正需要的 rgb 和 depth 目录！坚决不要 pose！
for sub in ["rgb", "depth"]:
    os.makedirs(os.path.join(DST_DIR, sub), exist_ok=True)
# ==========================================

def get_ball_centroids():
    """保留你原汁原味的按顺序读取逻辑"""
    centroids = []
    for i in range(1, 5):
        mesh_path = os.path.join(MESH_DIR, f"{i}.obj")
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"找不到钢珠模型: {mesh_path}")
        m = trimesh.load(mesh_path, process=False)
        centroids.append(m.vertices.mean(axis=0))
    return np.array(centroids, dtype=np.float64)

def calculate_reprojection_error(pts_3d, pts_2d, pose, K):
    """理想小孔成像，移除畸变参数"""
    R = pose[:3, :3]
    t = pose[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    
    # 畸变参数设为 None
    proj_pts, _ = cv2.projectPoints(pts_3d, rvec, t, K, None)
    proj_pts = proj_pts.squeeze()
    
    errors = np.linalg.norm(proj_pts - pts_2d, axis=1)
    return np.mean(errors)

def main():
    pts_3d = get_ball_centroids()
    K = np.loadtxt(K_PATH)
    
    with open(ANN_PATH, 'r') as f:
        full_data = json.load(f)
    
    annotations = full_data["annotations"] if "annotations" in full_data else full_data
    img_names = sorted([k for k in annotations.keys() if k.endswith(('.png', '.jpg'))])
    
    golden_count = 0
    total_valid = 0

    print(f"🔍 正在严格筛选误差 <= {ERROR_THRESHOLD}px 的黄金数据...")

    for img_name in tqdm(img_names):
        base_name = os.path.splitext(img_name)[0]
        pose_path = os.path.join(POSE_SRC, f"{base_name}.npy")
        depth_path = os.path.join(DEPTH_SRC, f"{base_name}.npy")
        
        if not (os.path.exists(pose_path) and os.path.exists(depth_path)):
            continue
            
        frame_data = annotations[img_name]
        try:
            pts_2d_orig = np.array([frame_data[f"ball_{i}"] for i in range(1, 5)], dtype=np.float64)
            # 🌟 保留这个数学命门：JSON 是 1080p，K 矩阵是 540p，必须降维对齐！
            pts_2d_540 = pts_2d_orig * 0.5
        except (KeyError, TypeError):
            continue

        pose = np.load(pose_path)
        error = calculate_reprojection_error(pts_3d, pts_2d_540, pose, K)
        total_valid += 1

        # 筛选与转移
        if error <= ERROR_THRESHOLD:
            golden_count += 1
            shutil.copy2(os.path.join(RGB_SRC, img_name), os.path.join(DST_DIR, "rgb", img_name))
            shutil.copy2(depth_path, os.path.join(DST_DIR, "depth", f"{base_name}.npy"))

    print("\n" + "="*40)
    print(f"📊 黄金训练集提取完毕！")
    print(f"   - 共提取: {golden_count} 帧 (仅含 rgb 与 depth)")
    print(f"   - 目的地: {DST_DIR}")
    print("="*40)

if __name__ == "__main__":
    main()