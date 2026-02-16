# 质量核验  将 3D ID 标签（ID_1 到 ID_4）投影回图像
import os
import cv2
import json
import numpy as np

# 获取项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# ================= 配置 =================
DATA_DIR = os.path.join(PROJECT_ROOT, "demo_data/tooth_gt")
DEPTH_DIR = os.path.join(PROJECT_ROOT, "demo_data/tooth_gt/gt_depth")  # 深度图目录
OUTPUT_DEBUG_DIR = os.path.join(PROJECT_ROOT, "debug_id_check")
# =======================================

def main():
    os.makedirs(OUTPUT_DEBUG_DIR, exist_ok=True)
    
    # 1. 加载数据
    K = np.loadtxt(f"{DATA_DIR}/cam_K.txt", dtype=np.float64)
    with open(f"{DATA_DIR}/keypoints_3d_model.json", 'r') as f:
        kpts_3d = np.array(json.load(f)['keypoints_3d_original'], dtype=np.float64)
    with open(f"{DATA_DIR}/annotations.json", 'r') as f:
        data = json.load(f)
    annotations = data.get("annotations", data)

    # 2. 获取已生成的有效帧（仅检查优化后的数据）
    valid_files = sorted([f for f in os.listdir(DEPTH_DIR) if f.endswith('.npy')])
    # 抽取前 20 帧进行 ID 对比
    check_files = valid_files[:20]

    print(f"正在生成 {len(check_files)} 帧的 ID 对应核验图...")
    
    for depth_file in check_files:
        frame_name = depth_file.replace('.npy', '.png').replace('.npy', '.jpg')
        
        rgb_path = os.path.join(DATA_DIR, "rgb", frame_name)
        if not os.path.exists(rgb_path):
            rgb_path = os.path.join(DATA_DIR, frame_name)
        
        img = cv2.imread(rgb_path)
        if img is None: continue
            
        if frame_name not in annotations: continue
        pts_2d_gt = []
        try:
            for i in range(1, 5):
                pts_2d_gt.append(annotations[frame_name][f'ball_{i}'])
            pts_2d_gt = np.array(pts_2d_gt, dtype=np.float64)
        except: continue

        # 3. 运行优化后的 PnP 逻辑
        _, rvec_init, tvec_init = cv2.solvePnP(kpts_3d, pts_2d_gt, K, None, flags=cv2.SOLVEPNP_EPNP)
        success, rvec, tvec = cv2.solvePnP(
            kpts_3d, pts_2d_gt, K, None, 
            flags=cv2.SOLVEPNP_ITERATIVE, 
            useExtrinsicGuess=True, 
            rvec=rvec_init, 
            tvec=tvec_init
        )
        
        if success:
            # 获取投影点
            proj_pts, _ = cv2.projectPoints(kpts_3d, rvec, tvec, K, None)
            proj_pts = proj_pts.squeeze()
            
            for i in range(4):
                # A. 绘制 3D 投影点 (红叉)
                pr_x, pr_y = int(proj_pts[i][0]), int(proj_pts[i][1])
                cv2.drawMarker(img, (pr_x, pr_y), (0, 0, 255), cv2.MARKER_CROSS, 10, 2)
                
                # B. 在红叉旁边标注 3D 模型的点序号 (ID)
                cv2.putText(img, f"ID_{i+1}", (pr_x + 5, pr_y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # C. 绘制原始 2D 标注点 (绿圆)
                gt_x, gt_y = int(pts_2d_gt[i][0]), int(pts_2d_gt[i][1])
                cv2.circle(img, (gt_x, gt_y), 8, (0, 255, 0), 1)

        save_path = os.path.join(OUTPUT_DEBUG_DIR, f"id_check_{frame_name}")
        cv2.imwrite(save_path, img)

    print(f"完成！核验图保存在: {OUTPUT_DEBUG_DIR}")

if __name__ == "__main__":
    main()