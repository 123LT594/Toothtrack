import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import time
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

# 🌟 导入解耦双头网络
from utils.depth_model_joint import MultiTaskUNet

SAVE_VIDEO = True  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    
    # 🌟 统一使用带钢珠的模型，提供物理轮廓锚点
    parser.add_argument("--mesh_file", type=str, default=f"{code_dir}/demo_data/tooth/mesh/tooth.obj")
    parser.add_argument("--test_scene_dir", type=str, default=f"{code_dir}/demo_data/tooth")
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=2) 
    parser.add_argument("--debug", type=int, default=1)
    parser.add_argument("--mode", type=int, default=1)
    
    # 🌟 确保加载你刚才【全量数据】训练出的新权重
    parser.add_argument("--joint_weight", type=str, default="/root/lanyun-tmp/models/models_joint/joint_best.pth")
    parser.add_argument("--temporal_weight", type=str, default="/root/lanyun-tmp/models/temporal_refiner/temporal_refiner_best.pth")
    args = parser.parse_args()

    output_root = "/root/lanyun-tmp/output"
    beijing_tz = pytz.timezone('Asia/Shanghai')
    output_dir = os.path.join(output_root, f"{dt.now(beijing_tz).strftime('%m%d_%H%M')}_joint_track")
    img_output_dir = os.path.join(output_dir, "img")
    os.makedirs(img_output_dir, exist_ok=True)

    set_logging_format(); set_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 🌟 提前抢占渲染通道，防止死锁
    glctx = dr.RasterizeCudaContext()
    joint_net = MultiTaskUNet().to(device)

    # 1. 初始化双头网络 (只需加载一次)
    joint_net = MultiTaskUNet().to(device)
    joint_net.load_state_dict(torch.load(args.joint_weight, map_location=device))
    joint_net.eval()

    # 加载 540p 标注数据
    ann_path = os.path.join(args.test_scene_dir + "_gt", "annotations.json")
    if not os.path.exists(ann_path): 
        ann_path = os.path.join(args.test_scene_dir, "annotations.json")
    with open(ann_path, 'r') as f:
        gt_data = json.load(f).get("annotations", {})
    all_frame_errors = []

    # 2. 坐标系与几何准备
    mesh = trimesh.load(args.mesh_file)
    model_center = compute_mesh_center(mesh) 
    mesh_centered = create_centered_mesh(mesh, model_center)
    mesh_dir = os.path.dirname(args.mesh_file)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    ball_centroids = [] 
    for ball_id in [1, 2, 3, 4]:
        p = os.path.join(mesh_dir, f"{ball_id}.obj")
        if os.path.exists(p): ball_centroids.append(trimesh.load(p).vertices.mean(0))
    ball_centroids = np.array(ball_centroids, dtype=np.float32) if ball_centroids else None

    # 3. 🌟 强制排队初始化：拆解嵌套，防止 CUDA 显存算子死锁
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    torch.tensor([0.0]).cuda()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, glctx=glctx)
    reader = YcbineoatReader(video_dir=args.test_scene_dir, zfar=np.inf)

    video_writer = cv2.VideoWriter(os.path.join(output_dir, "track_joint.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 30, (reader.W, reader.H))
    history_poses = []

    memory_bank = {}
    try:
        for i in range(len(reader.color_files)):
            color = reader.get_color(i)
            frame_name = reader.id_strs[i] + ".png"
            
            # --- 实时联合预测 ---
            rgb_tensor = torch.from_numpy(np.transpose(color.astype(np.float32)/255.0, (2,0,1))).unsqueeze(0).to(device)
            rgb_tensor[:, :, 13:75, 829:940] = 0.0  
            
            with torch.no_grad():
                m_logits, d_preds = joint_net(rgb_tensor)
                m_probs = torch.sigmoid(m_logits).squeeze().cpu().numpy()
                mask_u8 = (m_probs > 0.5).astype(np.uint8)*255
                num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
                if num > 1: mask_u8 = np.where(labels == (1+np.argmax(stats[1:,4])), 255, 0).astype(np.uint8)
                mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
                current_mask = mask_u8.astype(bool)
                current_depth = (d_preds.squeeze().cpu().numpy()) * (mask_u8.astype(np.float32) / 255.0)
                
                # 获取当前帧掩码面积
                current_area = current_mask.sum()

            # --- 追踪核心 ---
            t1 = time.time()
            if i == 0:
                pose = est.register(K=reader.K, rgb=color, depth=current_depth, ob_mask=current_mask, iteration=5)
                
                # ================= 🌟 [时序模块保留区] 权重注入 =================
                # print("\n⏳ 注入时序模块权重...")
                # refiner_net = getattr(est.refiner, 'model', getattr(est.refiner, 'net', None))
                # checkpoint = torch.load(args.temporal_weight, map_location=device)
                # if 'model' in checkpoint: checkpoint = checkpoint['model']
                # temporal_weights = {k: v for k, v in checkpoint.items() if 'temporal_attn' in k}
                # refiner_net.load_state_dict(temporal_weights, strict=False)
                # 
                # est.refiner.use_temporal = True  
                # print(f"✅ 时序大脑就绪！")
                # ==================================================================

            else:
                ys, xs = np.nonzero(current_mask)
                
                if len(ys) > 50:
                    # ================= 🌟 现实 vs 虚拟 实时对撞检测 =================
                    # 1. 现实世界 (UNet Mask) 的几何中心与跨度
                    real_cx = (xs.max() + xs.min()) / 2.0
                    real_cy = (ys.max() + ys.min()) / 2.0
                    real_span = max(xs.max() - xs.min(), ys.max() - ys.min())

                    # 2. 虚拟世界 (网络历史位姿) 的投影中心与跨度
                    pose_last_np = est.pose_last.detach().cpu().numpy().reshape(4, 4)
                    pts_cam = (pose_last_np[:3, :3] @ mesh_centered.vertices.T + pose_last_np[:3, 3:4]).T
                    pts_img = pts_cam @ reader.K.T
                    pts_img = pts_img[:, :2] / np.maximum(pts_img[:, 2:3], 1e-5)
                    
                    proj_min = pts_img.min(axis=0)
                    proj_max = pts_img.max(axis=0)
                    proj_span = max(proj_max[0] - proj_min[0], proj_max[1] - proj_min[1])

                    # 3. 计算尺度偏差
                    span_ratio = real_span / proj_span if proj_span > 0 else 1.0
                    iters = args.track_refine_iter

                    # ================= 🛡️ 终极护航：动态视野 + 深度与位姿双轨纠偏 =================
                    if 'initial_crop_ratio' not in memory_bank:
                        memory_bank['initial_crop_ratio'] = float(est.refiner.cfg['crop_ratio'])
                    base_crop = memory_bank['initial_crop_ratio']

                    # 只要偏差超过 5%，立马介入，绝不让它有累积误差的机会
                    if span_ratio > 1.05 or span_ratio < 0.95:
                        # 1. 动态撑大视野：保证网络绝对能看到边缘
                        dynamic_crop = base_crop * max(span_ratio, 1.0) * 1.1 
                        est.refiner.cfg['crop_ratio'] = float(min(dynamic_crop, 2.0)) 
                        
                        # 2. 计算物理拉伸系数 (Mask变大 -> span_ratio>1 -> 距离变近 -> 系数<1)
                        correction_factor = 1.0 / span_ratio
                        
                        # 3. 🌟 核心救星：强行按比例缩放 UNet 的深度图！
                        # 这一步让 FoundationPose 看到的 3D 点云瞬间逼近，彻底打消它往回跑的念头！
                        current_depth[current_mask] = current_depth[current_mask] * correction_factor
                        
                        # 4. 同步缩放上一帧的先验位姿，送到正确的 Z 轴起跑线
                        pose_last_np = est.pose_last.detach().cpu().numpy().reshape(4, 4)
                        new_pose = pose_last_np.copy()
                        new_pose[:3, 3] *= correction_factor
                        est.pose_last = torch.tensor(new_pose, device=device, dtype=torch.float32).unsqueeze(0)
                        
                        status_tag = f"🟡 变焦介入 (拉伸:{span_ratio:.2f}x) - 深度与位姿强行同步"
                        iters = 5 
                    else:
                        est.refiner.cfg['crop_ratio'] = base_crop
                        status_tag = "🟢 平稳追踪"
                    # =======================================================================

                    # ================= 🗡️ 网络精修 =================
                    # 此时无论是平稳还是被我们物理推拉过，位姿已经处于绝对安全区
                    centered_pose = est.track_one(rgb=color, depth=current_depth, K=reader.K, iteration=iters)
                    pose = centered_pose @ est.get_tf_to_centered_mesh().data.cpu().numpy().reshape(4, 4)

                else:
                    status_tag = "🔴 严重遮挡"
                    pose = est.pose_last.detach().cpu().numpy().reshape(4, 4) @ est.get_tf_to_centered_mesh().data.cpu().numpy().reshape(4, 4)
            t2 = time.time()
            history_poses.append(pose)

            # --- 可视化与精度计算 ---
            exact_center_pose = pose @ np.linalg.inv(to_origin)
            vis = create_visualization(color, pose, to_origin, reader.K, bbox, fps=1/(t2-t1),
                    render_3d=True, mesh_dir=mesh_dir, main_mesh=mesh, center_pose=exact_center_pose)
            
            frame_error = None
            if ball_centroids is not None and frame_name in gt_data and i < 800:
                try:
                    rvec_v, _ = cv2.Rodrigues(pose[:3,:3])
                    tvec_v = pose[:3,3]
                    pts_2d_proj, _ = cv2.projectPoints(ball_centroids, rvec_v, tvec_v, reader.K, None)
                    pts_2d_proj = pts_2d_proj.squeeze()

                    pts_2d_gt = np.array([gt_data[frame_name][f'ball_{j}'] for j in range(1, 5)], dtype=np.float32)

                    from scipy.spatial.distance import cdist
                    from scipy.optimize import linear_sum_assignment
                    dist_matrix = cdist(pts_2d_proj, pts_2d_gt)
                    row_ind, col_ind = linear_sum_assignment(dist_matrix)
                    
                    frame_error = dist_matrix[row_ind, col_ind].mean()
                    all_frame_errors.append(frame_error)
                    
                    for j, (r, c) in enumerate(zip(row_ind, col_ind)):
                        p_proj = tuple(pts_2d_proj[r].astype(int))
                        p_gt = tuple(pts_2d_gt[c].astype(int))
                        cv2.circle(vis, p_proj, 5, (255, 0, 0), -1)
                        cv2.circle(vis, p_gt, 6, (0, 255, 255), 2)
                        
                    cv2.putText(vis, f"Joint Err: {frame_error:.2f}px", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                except Exception as e:
                    pass
            
            cv2.imwrite(os.path.join(img_output_dir, f"{reader.id_strs[i]}.png"), vis[..., ::-1])
            video_writer.write(vis[..., ::-1])
            
            real_frame_id = i + 1
            err_str = f" | Err: {frame_error:.2f}px" if frame_error is not None else " | 无GT"
            current_tag = status_tag if i > 0 else "🟢 初始化"  # 🌟 统一用 status_tag
            print(f"✅ 帧 {real_frame_id:04d} ({frame_name}) 完成 | 耗时: {(t2-t1)*1000:.1f}ms{err_str} | {current_tag}")

    except KeyboardInterrupt:
        print("\n⚠️ 收到手动中断信号 (Ctrl+C)！提前结束追踪...")

    finally:
        video_writer.release()

        # 🌟 输出全套硬核统计指标
        if all_frame_errors:
            err_arr = np.array(all_frame_errors)
            mean_err = np.mean(err_arr)
            median_err = np.median(err_arr)
            rate_2px = np.sum(err_arr < 2.0) / len(err_arr) * 100
            rate_3px = np.sum(err_arr < 3.0) / len(err_arr) * 100
            
            print("\n" + "="*50)
            print(f"🏆 Joint 双头模型裸奔测试")
            print(f"📊 平均像素误差 (Mean):   {mean_err:.4f} px")
            print(f"📈 中位数误差 (Median): {median_err:.4f} px")
            print(f"✨ 极高精度比例 (<2px):  {rate_2px:.2f}%")
            print(f"✨ 优秀精度比例 (<3px):  {rate_3px:.2f}%")
            print("="*50)