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
import imageio
from utils.render_3d import create_visualization
from datetime import datetime
import pytz
from scipy.spatial.transform import Rotation as R

# 🌟 1. 导入你提取出来的双头深度预测网络 🌟
from utils.depth_model import PureDualHeadUNet

SAVE_VIDEO = True  # 可视化结果保存为MP4视频

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument("--mesh_file", type=str, default=f"{code_dir}/demo_data/tooth/mesh/teeth.obj")
    parser.add_argument("--test_scene_dir", type=str, default=f"{code_dir}/demo_data/tooth")
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=1)
    parser.add_argument("--debug", type=int, default=1, help="0=无输出, 1=可视化+保存")
    parser.add_argument("--debug_dir", type=str, default=f"{code_dir}/debug")
    
    parser.add_argument("--mode", type=int, default=1, help="Depth mode: 0=fake depth, 1=unet_depth_and_mask")
    # 🌟 2. 模型权重路径参数 (请确保名字是你最终满血版的 pth) 🌟
    parser.add_argument("--unet_weight", type=str, default=f"{code_dir}/../lanyun-tmp/models/pro_tooth_refine_e50.pth", help="双头网络权重路径")
    
    parser.add_argument("--no_render_3d", action="store_false", dest="render_3d", help="禁用3D模型渲染")
    parser.set_defaults(render_3d=True)
    parser.add_argument("--eval_full", action="store_true", help="计算完整误差指标")
    args = parser.parse_args()

    beijing_tz = pytz.timezone('Asia/Shanghai')
    beijing_time = datetime.now(beijing_tz)
    output_dir = os.path.join(code_dir, "output", f"{beijing_time.strftime('%m%d_%H%M')}_mode{args.mode}")
    img_output_dir = os.path.join(output_dir, "img")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(img_output_dir, exist_ok=True)

    set_logging_format()
    set_seed(0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    depth_net = None
    if args.mode == 1:
        depth_net = PureDualHeadUNet().to(device)
        depth_net.load_state_dict(torch.load(args.unet_weight, map_location=device))
        depth_net.eval()
        print("✅ 双头预测网络 (Mode 1: 实时 牙齿Depth + 牙齿Mask) 加载成功！")

    mesh = trimesh.load(args.mesh_file)
    model_center = compute_mesh_center(mesh)
    mesh_centered = create_centered_mesh(mesh, model_center)
    mesh_dir = os.path.dirname(args.mesh_file)
    
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    ball_centroids = []
    for ball_id in [1, 2, 3, 4]:
        ball_path = os.path.join(mesh_dir, f"{ball_id}.obj")
        if os.path.exists(ball_path):
            m = trimesh.load(ball_path, process=False)
            ball_centroids.append(m.vertices.mean(axis=0))
    if len(ball_centroids) == 4:
        ball_centroids = np.array(ball_centroids, dtype=np.float32)
        print("✅ 成功加载 4 个钢珠模型，准备进行联合刚体投影！")

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=args.debug_dir,
        debug=args.debug,
        glctx=glctx,
    )

    reader = YcbineoatReader(video_dir=args.test_scene_dir, zfar=np.inf)

    pose_centered = None
    use_rt_init = False
    last_depth = None
    last_mask = None
    pose = None
    history_poses = []

    video_writer = None
    if SAVE_VIDEO and args.debug >= 1:
        output_video_path = os.path.join(output_dir, "tracking_video.mp4")
        video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (reader.W, reader.H))

    for i in range(len(reader.color_files)):
        color = reader.get_color(i)
        
        # 🌟 4. 网络实时预测 🌟
        current_depth = None
        current_mask = None
        
        if args.mode == 1:
            rgb_input = color.astype(np.float32) / 255.0
            rgb_tensor = torch.from_numpy(np.transpose(rgb_input, (2, 0, 1))).unsqueeze(0).to(device)
            
            with torch.no_grad():
                depth_pred, mask_logits = depth_net(rgb_tensor)
                
                pred_mask_binary = (torch.sigmoid(mask_logits) > 0.5).float()
                mask_uint8 = (pred_mask_binary.squeeze().cpu().numpy() * 255).astype(np.uint8)
                
                # ================= 🌟 后处理护城河升级 =================
                # 1. 输出端物理遮罩：涂黑右上角 Logo
                mask_uint8[13:75, 829:940] = 0 
                
                # 2. 连通域过滤：剔除杂质
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
                if num_labels > 1:
                    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    mask_uint8 = np.where(labels == largest_label, 255, 0).astype(np.uint8)
                
                # 3. 🌟 物理抹除：如果钢珠导致漂移，启用此逻辑抠除钢珠 (可选)
                # gray_img = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
                # _, ball_mask = cv2.threshold(gray_img, 80, 255, cv2.THRESH_BINARY_INV)
                # holes = cv2.bitwise_and(ball_mask, mask_uint8)
                # mask_uint8[holes == 255] = 0
                    
                # 4. 边缘腐蚀：斩断“拉丝”悬崖，保证 ICP 不被打滑扯飞
                kernel_erode = np.ones((5, 5), np.uint8)
                mask_uint8 = cv2.erode(mask_uint8, kernel_erode, iterations=1)
                # =========================================================
                
                current_mask = mask_uint8.astype(bool)
                clean_mask_tensor = torch.from_numpy(mask_uint8).float().to(device) / 255.0
                current_depth = (depth_pred * clean_mask_tensor).squeeze().cpu().numpy()

        # =========== 第一帧：初始化 ===========
        if i == 0:
            disk_mask = reader.get_mask(0)
            if disk_mask is not None:
                mask = disk_mask.astype(bool)
                print("🌟 发现硬盘 GT Mask！采用完美 Mask 初始化")
            else:
                mask = current_mask if args.mode == 1 else reader.get_mask(0).astype(bool)
                print("⚠️ 使用网络预测的 Mask 初始化")
                
            last_mask = mask
            t1 = time.time()

            initial_pose_original = reader.get_gt_pose(0) if len(reader.gt_pose_files) > 0 else None

            if initial_pose_original is not None:
                print("✓ 使用 GT pose 初始化")
                pose = initial_pose_original.copy()
                
                tf_to_center = est.get_tf_to_centered_mesh().cpu().numpy()
                pose_centered_init = pose @ np.linalg.inv(tf_to_center)

                est.pose_last = torch.as_tensor(pose_centered_init, device="cuda", dtype=torch.float)
                est.xyz = est.pose_last[:3, 3]
                est.mask_last = mask
                est.track_good = True
                est.H, est.W = mask.shape[:2]
                est.K = reader.K

                euler_angles = R.from_matrix(pose_centered_init[:3, :3]).as_euler("xyz").reshape(3, 1)
                est.tracker.initialize(est.xyz.detach().cpu().numpy().reshape(3, 1), euler_angles)
                last_depth = current_depth if args.mode == 1 else render_cad_depth(convert_pose_for_render(pose, model_center), mesh, reader.K, w=reader.W, h=reader.H)
            else:
                print("✓ 使用网络预测的 Depth 和 Mask 进行全图初始化")
                pose = est.register(K=reader.K, rgb=color, depth=current_depth, ob_mask=mask, iteration=args.est_refine_iter)
                last_depth = current_depth
            t2 = time.time()

        # =========== 后续帧：追踪 ===========
        else:
            t1 = time.time()
            if args.mode == 0:
                last_depth = np.zeros_like(last_mask)
                pose = est.track_one(rgb=color, depth=last_depth, K=reader.K, iteration=args.track_refine_iter)
            elif args.mode == 1:
                last_depth = current_depth
                last_mask = current_mask
                pose = est.track_one_new(rgb=color, depth=last_depth, K=reader.K, iteration=args.track_refine_iter, mask=current_mask)
            t2 = time.time()

        print(f"✅ 帧 {i:04d} 追踪完成")
        history_poses.append(pose)

        # ================= 可视化与多目标投影 =================
        if args.debug >= 1:
            fps_val = 1 / (t2 - t1) if (t2 - t1) > 0 else 0
            
            exact_center_pose = pose @ np.linalg.inv(to_origin)
            
            vis = create_visualization(color, pose, to_origin, reader.K, bbox, fps=fps_val,
                render_3d=args.render_3d, mesh_dir=mesh_dir if args.render_3d else None,
                main_mesh=mesh if args.render_3d else None, 
                center_pose=exact_center_pose,
                pose_centered=None,
                use_rt_init=False)
            
            if len(ball_centroids) == 4:
                vis = vis.copy()
                rvec, _ = cv2.Rodrigues(pose[:3, :3])
                tvec = pose[:3, 3]
                pts_2d, _ = cv2.projectPoints(ball_centroids, rvec, tvec, reader.K, None)
                pts_2d = pts_2d.squeeze().astype(int)
                
                for pt in pts_2d:
                    cv2.circle(vis, tuple(pt), 4, (255, 0, 0), -1) 
                    cv2.circle(vis, tuple(pt), 2, (0, 255, 255), -1) 

            cv2.imwrite(os.path.join(img_output_dir, f"{reader.id_strs[i]}.png"), vis[..., ::-1])
            if SAVE_VIDEO and video_writer is not None:
                video_writer.write(vis[..., ::-1])

    if SAVE_VIDEO and video_writer is not None:
        video_writer.release()

    # ==================== 误差分析 ====================
    if len(reader.gt_pose_files) > 0 and len(history_poses) > 0:
        try:
            avg_metrics, frame_metrics = evaluate_metrics_fast(history_poses, reader, mesh_centered, traj=True, model_center=model_center)

            if isinstance(frame_metrics, dict):
                num_frames = len(frame_metrics.get('ADD', []))
                frame_metrics = [{key: frame_metrics[key][i] for key in frame_metrics} for i in range(num_frames)]

            if args.eval_full:
                avg_metrics_full, frame_metrics_full = evaluate_metrics(history_poses, reader, mesh_centered, traj=True)
                for key in ['mspd', 'mssd', 'AR_vsd', 'AR_mspd', 'AR_mssd', 'recall']:
                    if key in avg_metrics_full:
                        avg_metrics[key] = avg_metrics_full[key]
                if isinstance(frame_metrics_full, dict):
                    num_frames = len(frame_metrics_full.get('ADD', []))
                    frame_metrics_full = [{key: frame_metrics_full[key][i] for key in frame_metrics_full} for i in range(num_frames)]
                    for i, fm_full in enumerate(frame_metrics_full):
                        if i < len(frame_metrics) and frame_metrics[i] is not None:
                            for key in ['mspd', 'mssd', 'AR_vsd', 'AR_mspd', 'AR_mssd', 'recall']:
                                if key in fm_full and fm_full[key] is not None:
                                    frame_metrics[i][key] = fm_full[key]

            add_values = [f['ADD'] for f in frame_metrics if f and f.get('ADD') is not None]
            adds_values = [f['ADD-S'] for f in frame_metrics if f and f.get('ADD-S') is not None]
            rot_errors = [f['rotation_error_deg'] for f in frame_metrics if f and f.get('rotation_error_deg') is not None]
            trans_errors = [f['translation_error'] for f in frame_metrics if f and f.get('translation_error') is not None]

            eval_results = {
                'summary': {
                    'total_frames': len(history_poses),
                    'average_metrics': {k: avg_metrics[k] for k in ['ADD', 'ADD-S', 'rotation_error_deg', 'translation_error']},
                    'statistics': {
                        'ADD': {'mean': float(np.mean(add_values)), 'median': float(np.median(add_values))},
                        'ADD-S': {'mean': float(np.mean(adds_values)), 'median': float(np.median(adds_values))},
                        'rotation_error_deg': {'mean': float(np.mean(rot_errors)), 'median': float(np.median(rot_errors))},
                        'translation_error': {'mean': float(np.mean(trans_errors)), 'median': float(np.median(trans_errors))},
                    }
                },
                'frame_metrics': [
                    {'frame_id': reader.id_strs[i] if i < len(reader.id_strs) else f"frame_{i:04d}", **f}
                    for i, f in enumerate(frame_metrics)
                ]
            }

            eval_output_file = os.path.join(output_dir, "evaluation_results.json")
            with open(eval_output_file, 'w', encoding='utf-8') as f:
                json.dump(eval_results, f, indent=2, ensure_ascii=False)

            print(f"\n✓ 误差分析完成！结果已保存到: {eval_output_file}")

        except Exception as e:
            print(f"误差分析失败: {e}")