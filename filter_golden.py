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

# 你的标准阈值：1.5px（540p纯正尺度）
ERROR_THRESHOLD = 1.5

# 创建三个目录
for sub in ["rgb", "depth", "pose"]:
    os.makedirs(os.path.join(DST_DIR, sub), exist_ok=True)
# ==========================================

def get_ball_centroids():
    centroids = []
    for i in range(1, 5):
        mesh_path = os.path.join(MESH_DIR, f"{i}.obj")
        m = trimesh.load(mesh_path, process=False)
        centroids.append(m.vertices.mean(axis=0))
    return np.array(centroids, dtype=np.float64)

def main():
    pts_3d = get_ball_centroids()
    K = np.loadtxt(K_PATH, dtype=np.float64)
    
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
            pts_2d = np.array([frame_data[f"ball_{i}"] for i in range(1, 5)], dtype=np.float64)
        except (KeyError, TypeError):
            continue

        # 读取pose
        pose = np.load(pose_path)
        R = pose[:3, :3]
        t = pose[:3, 3]
        rvec, _ = cv2.Rodrigues(R)
        
        # 正确重投影
        proj_pts, _ = cv2.projectPoints(pts_3d, rvec, t, K, None)
        proj_pts = proj_pts.squeeze()
        error = np.mean(np.linalg.norm(proj_pts - pts_2d, axis=1))
        
        total_valid += 1

        # 筛选通过，复制三个文件
        if error <= ERROR_THRESHOLD:
            golden_count += 1
            shutil.copy2(os.path.join(RGB_SRC, img_name), os.path.join(DST_DIR, "rgb", img_name))
            shutil.copy2(depth_path, os.path.join(DST_DIR, "depth", f"{base_name}.npy"))
            shutil.copy2(pose_path, os.path.join(DST_DIR, "pose", f"{base_name}.npy"))

    print("\n" + "="*50)
    print(f"✅ 黄金训练集提取完毕！")
    print(f"   有效总帧数：{total_valid}")
    print(f"   筛选黄金帧：{golden_count}")
    print(f"   输出目录：{DST_DIR}")
    print("   包含：rgb / depth / pose")
    print("="*50)

if __name__ == "__main__":
    main()