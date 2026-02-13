import time
import os
from utils.estimater import *
from utils.datareader import *
import argparse
from utils.tools import (
    create_centered_mesh, convert_pose_for_render, compute_mesh_diameter,
    compute_mesh_center, quick_verify_coordinate_system, evaluate_pose_fast,
    evaluate_metrics_fast
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


SAVE_VIDEO = True  # 可视化结果保存为MP4视频


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument("--mesh_file", type=str, default=f"{code_dir}/demo_data/tooth/mesh/tooth.obj")
    parser.add_argument("--test_scene_dir", type=str, default=f"{code_dir}/demo_data/tooth")
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=1)
    parser.add_argument("--debug", type=int, default=1, help="0=无输出, 1=可视化+保存")
    parser.add_argument("--debug_dir", type=str, default=f"{code_dir}/debug")
    parser.add_argument("--mode", type=int, default=0, help="Depth mode: 0=fake depth, 1=render_cad_depth")
    parser.add_argument("--no_render_3d", action="store_false", dest="render_3d", help="禁用3D模型渲染")
    parser.set_defaults(render_3d=True)
    parser.add_argument("--eval_full", action="store_true", help="计算完整误差指标")
    args = parser.parse_args()

    # 创建输出目录
    beijing_tz = pytz.timezone('Asia/Shanghai')
    beijing_time = datetime.now(beijing_tz)
    output_dir = os.path.join(code_dir, "output", f"{beijing_time.strftime('%m%d_%H%M')}_mode{args.mode}")
    img_output_dir = os.path.join(output_dir, "img")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(img_output_dir, exist_ok=True)

    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(args.mesh_file)
    model_center = compute_mesh_center(mesh)
    mesh_centered = create_centered_mesh(mesh, model_center)
    mesh_diameter = compute_mesh_diameter(mesh_centered)
    mesh_dir = os.path.dirname(args.mesh_file)

    os.makedirs(f"{args.debug_dir}/ob_in_cam", exist_ok=True)

    # 计算bbox
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

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

    reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=480, zfar=np.inf)

    # 初始化变量
    pose_centered = None
    use_rt_init = False
    last_depth = None
    last_mask = None
    pose = None
    history_poses = []

    # 初始化视频写入器
    video_writer = None
    if SAVE_VIDEO and args.debug >= 1:
        output_video_path = os.path.join(output_dir, "tracking_video.mp4")
        video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (reader.W, reader.H))
        if not video_writer.isOpened():
            logging.error(f"无法创建视频文件: {output_video_path}")
            video_writer = None

    for i in range(len(reader.color_files)):
        color = reader.get_color(i)
        if i == 0:
            mask = reader.get_mask(0)
            if mask is None:
                raise ValueError("第一帧的mask文件不存在")
            mask = mask.astype(bool)
            last_mask = mask
            t1 = time.time()

            # 优先使用GT pose初始化
            initial_pose = reader.get_gt_pose(0) if len(reader.gt_pose_files) > 0 else None

            if initial_pose is not None:
                print("✓ 使用GT pose初始化")
                pose_centered = initial_pose
                use_rt_init = True
                pose = pose_centered.copy()

                est.pose_last = torch.as_tensor(pose_centered, device="cuda", dtype=torch.float)
                est.xyz = est.pose_last[:3, 3]
                est.mask_last = mask
                est.track_good = True
                est.H, est.W = mask.shape[:2]
                est.K = reader.K

                euler_angles = R.from_matrix(pose_centered[:3, :3]).as_euler("xyz").reshape(3, 1)
                est.tracker.initialize(est.xyz.detach().cpu().numpy().reshape(3, 1), euler_angles)

                last_depth = render_cad_depth(convert_pose_for_render(pose, model_center), mesh, reader.K, w=reader.W, h=reader.H)
            else:
                pose = est.register(K=reader.K, rgb=color, depth=np.zeros_like(mask), ob_mask=mask,
                                   iteration=args.est_refine_iter, rough_depth_guess=0.123)
                use_rt_init = False
                pose_centered = None
                last_depth = render_cad_depth(convert_pose_for_render(pose, model_center), mesh, reader.K, w=reader.W, h=reader.H)

            history_poses.append(pose.copy())

            # 第一帧验证
            if len(reader.gt_pose_files) > 0:
                gt_pose = reader.get_gt_pose(0)
                if gt_pose is not None:
                    verify_result = quick_verify_coordinate_system(gt_pose, pose, mesh_centered, model_center)
                    print("=" * 70)
                    print("🔍 第一帧坐标系快速验证")
                    print("=" * 70)
                    print(f"点云中心距离: {verify_result['center_distance_mm']:.2f} mm")
                    print(f"ADD值: {verify_result['add_mm']:.2f} mm")
                    print(f"GT平移: {verify_result['gt_translation_norm']*1000:.2f} mm")
                    print(f"EST平移: {verify_result['est_translation_norm']*1000:.2f} mm")
                    print(f"模型直径: {mesh_diameter*1000:.2f}mm")
                    print("=" * 70)

            t2 = time.time()

        else:
            t1 = time.time()
            # 后续帧追踪
            if args.mode == 0:
                last_depth = np.zeros_like(last_mask)
            elif args.mode == 1:
                last_depth = render_cad_depth(convert_pose_for_render(pose, model_center), mesh, reader.K, w=reader.W, h=reader.H)

            pose = est.track_one(rgb=color, depth=last_depth, K=reader.K, iteration=args.track_refine_iter)
            t2 = time.time()

        # 保存位姿
        np.savetxt(f"{args.debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt", pose.reshape(4, 4))
        history_poses.append(pose.copy())

        # 实时误差计算
        if len(reader.gt_pose_files) > 0:
            gt_pose = reader.get_gt_pose(i)
            if gt_pose is not None and i > 0:
                verify_result = quick_verify_coordinate_system(gt_pose, pose, mesh_centered, model_center)
                print(f"帧 {i:04d} ✓ ADD={verify_result['add_mm']:.2f}mm")

        # 可视化
        if args.debug >= 1:
            fps_val = 1 / (t2 - t1) if (t2 - t1) > 0 else 0
            vis = create_visualization(color, pose, to_origin, reader.K, bbox, fps=fps_val,
                render_3d=args.render_3d, mesh_dir=mesh_dir if args.render_3d else None,
                main_mesh=mesh if args.render_3d else None, center_pose=None,
                pose_centered=pose_centered if (i == 0 and use_rt_init) else None,
                use_rt_init=(i == 0 and use_rt_init))

            # cv2.imshow 需要图形界面，无头模式下跳过
            # cv2.imshow("3D Model Overlay", vis[..., ::-1])
            # cv2.waitKey(1)
            cv2.imwrite(os.path.join(img_output_dir, f"{reader.id_strs[i]}.png"), vis[..., ::-1])

            if SAVE_VIDEO and video_writer is not None:
                video_writer.write(vis[..., ::-1])

    if SAVE_VIDEO and video_writer is not None:
        video_writer.release()

    # ==================== 误差分析 ====================
    if len(reader.gt_pose_files) > 0 and len(history_poses) > 0:
        try:
            avg_metrics, frame_metrics = evaluate_metrics_fast(history_poses, reader, mesh_centered, traj=True, model_center=model_center)

            # 转换格式
            if isinstance(frame_metrics, dict):
                num_frames = len(frame_metrics.get('ADD', []))
                frame_metrics = [{key: frame_metrics[key][i] for key in frame_metrics} for i in range(num_frames)]

            # 完整评估
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

            # 统计计算
            add_values = [f['ADD'] for f in frame_metrics if f and f.get('ADD') is not None]
            adds_values = [f['ADD-S'] for f in frame_metrics if f and f.get('ADD-S') is not None]
            rot_errors = [f['rotation_error_deg'] for f in frame_metrics if f and f.get('rotation_error_deg') is not None]
            trans_errors = [f['translation_error'] for f in frame_metrics if f and f.get('translation_error') is not None]

            eval_results = {
                'summary': {
                    'total_frames': len(history_poses),
                    'average_metrics': {k: avg_metrics[k] for k in ['ADD', 'ADD-S', 'rotation_error_deg', 'translation_error']},
                    'statistics': {
                        'ADD': {'mean': np.mean(add_values), 'median': np.median(add_values), 'std': np.std(add_values), 'min': np.min(add_values), 'max': np.max(add_values)},
                        'ADD-S': {'mean': np.mean(adds_values), 'median': np.median(adds_values), 'std': np.std(adds_values), 'min': np.min(adds_values), 'max': np.max(adds_values)},
                        'rotation_error_deg': {'mean': np.mean(rot_errors), 'median': np.median(rot_errors), 'std': np.std(rot_errors), 'min': np.min(rot_errors), 'max': np.max(rot_errors)},
                        'translation_error': {'mean': np.mean(trans_errors), 'median': np.median(trans_errors), 'std': np.std(trans_errors), 'min': np.min(trans_errors), 'max': np.max(trans_errors)},
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
            logging.error(f"误差分析失败: {e}")
            import traceback
            traceback.print_exc()
