import time
import os
from utils.estimater import *
from utils.datareader import *
import argparse
from utils.tools import (
    create_centered_mesh, convert_pose_for_render, compute_mesh_diameter,
    compute_mesh_center, quick_verify_coordinate_system, evaluate_pose_fast,
    evaluate_metrics_fast, evaluate_metrics
)
import numpy as np
import cv2
import torch
import json
import trimesh
from utils.render_3d import create_visualization
from datetime import datetime as dt 
import pytz
from scipy.spatial.transform import Rotation as R
import nvdiffrast.torch as dr

# 🌟 深度估计与掩码估计 (独立模型)
from utils.depth_model import SimpleUNet, DepthUNet

SAVE_VIDEO = True  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    # 保持一致：追踪物理引擎使用带钢珠的真实轮廓模型
    parser.add_argument("--mesh_file", type=str, default=f"{code_dir}/demo_data/tooth/mesh/teeth.obj")
    parser.add_argument("--test_scene_dir", type=str, default=f"{code_dir}/demo_data/tooth")
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=1) 
    parser.add_argument("--debug", type=int, default=1)
    parser.add_argument("--mode", type=int, default=1)
    
    parser.add_argument("--mask_weight", type=str, default="/root/lanyun-tmp/models/models0/mask_best.pth")
    parser.add_argument("--depth_weight", type=str, default="/root/lanyun-tmp/models/models0/depth_best.pth")
    
    args = parser.parse_args()

    output_root = "/root/lanyun-tmp/output"
    beijing_tz = pytz.timezone('Asia/Shanghai')
    output_dir = os.path.join(output_root, f"{dt.now(beijing_tz).strftime('%m%d_%H%M')}_joint_track")
    img_output_dir = os.path.join(output_dir, "img")
    os.makedirs(img_output_dir, exist_ok=True)

    set_logging_format(); set_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. 初始化深度/掩码网络 (独立)
    mask_net = SimpleUNet().to(device)
    mask_net.load_state_dict(torch.load(args.mask_weight, map_location=device))
    mask_net.eval()
    
    depth_net = DepthUNet().to(device)
    depth_net.load_state_dict(torch.load(args.depth_weight, map_location=device))
    depth_net.eval()

    # 加载 GT 标注数据
    ann_path = os.path.join(args.test_scene_dir + "_gt", "annotations_540p.json")
    if not os.path.exists(ann_path): 
        ann_path = os.path.join(args.test_scene_dir, "annotations_540p.json")
    with open(ann_path, 'r') as f:
        gt_data = json.load(f).get("annotations", {})
    all_frame_errors = []

    # 2. 坐标系与几何准备
    mesh = trimesh.load(args.mesh_file)
    model_center = compute_mesh_center(mesh) 
    mesh_centered = create_centered_mesh(mesh, model_center) # FoundationPose 内部使用
    mesh_dir = os.path.dirname(args.mesh_file)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    ball_centroids = [] 
    for ball_id in [1, 2, 3, 4]:
        p = os.path.join(mesh_dir, f"{ball_id}.obj")
        if os.path.exists(p): ball_centroids.append(trimesh.load(p).vertices.mean(0))
    ball_centroids = np.array(ball_centroids, dtype=np.float32) if ball_centroids else None

    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, 
                         scorer=ScorePredictor(), refiner=PoseRefinePredictor(), glctx=dr.RasterizeCudaContext())
    reader = YcbineoatReader(video_dir=args.test_scene_dir, zfar=np.inf)

    video_writer = cv2.VideoWriter(os.path.join(output_dir, "track_independent.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 30, (reader.W, reader.H))
    memory_bank = {}; history_poses = []

    try:
        for i in range(len(reader.color_files)):
            color = reader.get_color(i)
            frame_name = reader.id_strs[i] + ".png"
            
            # --- 实时独立预测 (跑两次网络) ---
            rgb_tensor = torch.from_numpy(np.transpose(color.astype(np.float32)/255.0, (2,0,1))).unsqueeze(0).to(device)
            rgb_tensor[:, :, 13:75, 829:940] = 0.0 
            
            with torch.no_grad():
                m_probs = torch.sigmoid(mask_net(rgb_tensor)).squeeze().cpu().numpy()
                mask_u8 = (m_probs > 0.5).astype(np.uint8)*255
                num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
                if num > 1: mask_u8 = np.where(labels == (1+np.argmax(stats[1:,4])), 255, 0).astype(np.uint8)
                mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
                current_mask = mask_u8.astype(bool)
                
                current_depth = (depth_net(rgb_tensor) * (torch.from_numpy(mask_u8).float().to(device)/255.0)).squeeze().cpu().numpy()

            # --- 追踪核心---
            t1 = time.time()
            if i == 0:
                pose = est.register(K=reader.K, rgb=color, depth=current_depth, ob_mask=current_mask, iteration=5)
            else:
                pose = est.track_one_new(rgb=color, depth=current_depth, K=reader.K, iteration=args.track_refine_iter, mask=current_mask)
                
                # 仅监控，绝不复位干预！
                if current_mask.sum() > 500:
                    y_idx, x_idx = np.where(current_mask)
                    mask_cx, mask_cy = np.mean(x_idx), np.mean(y_idx)
                    
                    curr_pose_np = est.pose_last.detach().cpu().numpy().reshape(4, 4)
                    rvec, _ = cv2.Rodrigues(curr_pose_np[:3, :3]); tvec = curr_pose_np[:3, 3]
                    
                    proj_pts, _ = cv2.projectPoints(np.array([[0,0,0]], dtype=np.float32), rvec, tvec, reader.K, None)
                    mesh_cx, mesh_cy = proj_pts.squeeze()
                    drift_dist = np.sqrt((mask_cx - mesh_cx)**2 + (mask_cy - mesh_cy)**2)
                    
             
            t2 = time.time()
            history_poses.append(pose)

            # --- 可视化与 🌟 对齐后的精度计算 ---
            exact_center_pose = pose @ np.linalg.inv(to_origin)
            vis = create_visualization(color, pose, to_origin, reader.K, bbox, fps=1/(t2-t1),
                    render_3d=True, mesh_dir=mesh_dir, main_mesh=mesh, center_pose=exact_center_pose)
            
            frame_error = None
            if ball_centroids is not None and i < 800 and frame_name in gt_data:
                try:
                    # ✅ 1. 彻底移除 model_center 的错误减法，使用原位投影
                    rvec_v, _ = cv2.Rodrigues(pose[:3,:3]); tvec_v = pose[:3,3]
                    pts_2d_proj, _ = cv2.projectPoints(ball_centroids, rvec_v, tvec_v, reader.K, None)
                    pts_2d_proj = pts_2d_proj.squeeze()

                    # 获取真值
                    pts_2d_gt = np.array([gt_data[frame_name][f'ball_{j}'] for j in range(1, 5)], dtype=np.float32)

                    # ✅ 2. 引入匈牙利算法，解决点序错乱导致的冤枉误差
                    from scipy.spatial.distance import cdist
                    from scipy.optimize import linear_sum_assignment
                    dist_matrix = cdist(pts_2d_proj, pts_2d_gt)
                    row_ind, col_ind = linear_sum_assignment(dist_matrix)
                    
                    frame_error = dist_matrix[row_ind, col_ind].mean()
                    all_frame_errors.append(frame_error)
                    
                    # ✅ 3. 对齐可视化表现形式：红点预测，青圈真值
                    for j, (r, c) in enumerate(zip(row_ind, col_ind)):
                        p_proj = tuple(pts_2d_proj[r].astype(int))
                        p_gt = tuple(pts_2d_gt[c].astype(int))
                        cv2.circle(vis, p_proj, 5, (255, 0, 0), -1) 
                        cv2.circle(vis, p_gt, 6, (0, 255, 255), 2) 
                        
                    cv2.putText(vis, f"Indep Err: {frame_error:.2f}px", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                except Exception as e:
                    pass

            cv2.imwrite(os.path.join(img_output_dir, f"{reader.id_strs[i]}.png"), vis[..., ::-1])
            video_writer.write(vis[..., ::-1])
            
            # 🌟 实时输出：对齐 PNG 帧号 (i 从 0 开始，真实图片从 1 开始)
            real_frame_id = i + 1
            err_str = f" | Err: {frame_error:.2f}px" if frame_error is not None else (" | 无GT (跳过)" if i < 800 else " | 仅追踪 (>800)")
            print(f"✅ 帧 {real_frame_id:04d} ({frame_name}) 完成 | 耗时: {(t2-t1)*1000:.1f}ms{err_str}")

    except KeyboardInterrupt:
        print("\n⚠️ 收到手动中断信号 (Ctrl+C)！提前结束追踪...")

    finally:
        video_writer.release()
    
        # 🌟 输出全套硬核统计指标
        if all_frame_errors:
            err_arr = np.array(all_frame_errors)
            mean_err = np.mean(err_arr)
            median_err = np.median(err_arr) # 中位数
            rate_2px = np.sum(err_arr < 2.0) / len(err_arr) * 100
            rate_3px = np.sum(err_arr < 3.0) / len(err_arr) * 100 # <3px 比例
            
            print("\n" + "="*50)
            print(f"🏆 裸奔模式追踪精度评估 (无复位干预)")
            print(f"📊 平均像素误差 (Mean):   {mean_err:.4f} px")
            print(f"📈 中位数误差 (Median): {median_err:.4f} px")
            print(f"✨ 极高精度比例 (<2px):  {rate_2px:.2f}%")
            print(f"✨ 优秀精度比例 (<3px):  {rate_3px:.2f}%")
            print("="*50)