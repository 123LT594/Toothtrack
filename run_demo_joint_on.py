import os
import sys
os.environ['OMP_NUM_THREADS'] = '1'
import time
import argparse
import numpy as np
import cv2
import torch
import json
import trimesh
import pytz
import nvdiffrast.torch as dr
from datetime import datetime as dt 

from utils.estimater import *
from utils.datareader import *
from utils.tools import *
from utils.render_3d import create_visualization

from learning.models.student_depth_net import StudentDepthNet
from learning.training.training_config import DISTILL_PHYSICAL_WIDTH

# ==========================================================
# 🌟 引入我们全新的纯几何首帧粗筛模块
from utils.zero_shot_geometry_matcher import FastZeroShotMatcher
# ==========================================================


def calc_te(pose_pred, pose_gt):
    return np.linalg.norm(pose_pred[:3, 3] - pose_gt[:3, 3]) * 1000.0

def calc_re(pose_pred, pose_gt):
    R_pred, R_gt = pose_pred[:3, :3], pose_gt[:3, :3]
    trace = np.trace(R_pred @ R_gt.T)
    return np.rad2deg(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))

def calc_add(pose_pred, pose_gt, vertices):
    pts_pred = (pose_pred[:3, :3] @ vertices.T + pose_pred[:3, 3:4]).T
    pts_gt = (pose_gt[:3, :3] @ vertices.T + pose_gt[:3, 3:4]).T
    return np.linalg.norm(pts_pred - pts_gt, axis=1).mean() * 1000.0

def get_bbox_from_pose(pose, K, vertices):
    rvec, _ = cv2.Rodrigues(pose[:3, :3])
    tvec = pose[:3, 3]
    pts_2d, _ = cv2.projectPoints(vertices, rvec, tvec, K, None)
    pts_2d = pts_2d.squeeze()
    x_min, x_max = pts_2d[:, 0].min(), pts_2d[:, 0].max()
    y_min, y_max = pts_2d[:, 1].min(), pts_2d[:, 1].max()
    w = max(x_max - x_min, 10)
    h = max(y_max - y_min, 10)
    return x_min, y_min, w, h

# 深度图可视化上色函数
def get_auto_color(d_np, mask_uint8):
    m_np = mask_uint8 > 127
    vis = np.zeros_like(d_np, dtype=np.uint8)
    if m_np.sum() > 0:
        valid = d_np[m_np]
        p_min, p_max = valid.min(), valid.max()
        if p_max - p_min > 1e-4:
            norm = np.clip((d_np - p_min) / (p_max - p_min), 0, 1)
            vis = (norm * 255).astype(np.uint8)
        else:
            vis[m_np] = 127 
    color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    color[~m_np] = 0
    return color

class SuppressPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument("--mesh_file", type=str, default=f"{code_dir}/demo_data/ztooth/mesh/tooth.obj")
    parser.add_argument("--test_scene_dir", type=str, default=f"{code_dir}/demo_data/ztooth")
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=1) 
    parser.add_argument("--weight_student", type=str, default="/root/lanyun-tmp/models/stage1_distill/models/student_stage1_ep99.pth")
    
    parser.add_argument("--use_pred_init", default=True, type=bool, help="默认开启首帧预测")
    parser.add_argument("--no_eval", default=False, type=bool, help="默认不开启验证")
    
    args = parser.parse_args()

    output_root = "/root/lanyun-tmp/output"
    beijing_tz = pytz.timezone('Asia/Shanghai')
    output_dir = os.path.join(output_root, f"{dt.now(beijing_tz).strftime('%m%d_%H%M')}_distill_track")
    img_output_dir = os.path.join(output_dir, "img")
    mask_depth_output_dir = os.path.join(output_dir, "mask+depth") 
    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(mask_depth_output_dir, exist_ok=True)
    
    error_txt_path = os.path.join(output_dir, "frame_errors.txt")
    txt_file = open(error_txt_path, "w", encoding="utf-8")

    set_logging_format(); set_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    glctx = dr.RasterizeCudaContext()

    print("🚀 加载 3D 蒸馏基座 (StudentDepthNet)...")
    model_expert = StudentDepthNet().to(device)
    model_expert.load_state_dict(torch.load(args.weight_student, map_location=device))
    model_expert.eval()

    gt_data = {}
    ball_centroids = None
    all_frame_errors = []
    all_track_times = []
    eval_metrics = {'TE': [], 'RE': [], 'ADD': [], 'IoU': [], 'MAE': []}
    
    if not args.no_eval:
        ann_path = os.path.join(args.test_scene_dir, "annotations.json")
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f: gt_data = json.load(f).get("annotations", {})
        
        gt_pose_dir = "/root/lanyun-tmp/golden_dataset/pose"
        gt_depth_dir = "/root/lanyun-tmp/golden_dataset/depth"

        ball_centroids_list = [] 
        for j in [1, 2, 3, 4]:
            p = os.path.join(os.path.dirname(args.mesh_file), f"{j}.obj")
            if os.path.exists(p): ball_centroids_list.append(trimesh.load(p).vertices.mean(0))
        if ball_centroids_list: ball_centroids = np.array(ball_centroids_list, dtype=np.float32)

    all_frame_times = []
    mesh = trimesh.load(args.mesh_file)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    # ==========================================================
    # 🌟 核心修改点：
    # 1. 注释掉原版极耗显存的 RGB Scorer 模型
    # 2. 将 scorer 传入 None
    # 3. 初始化极速全数学几何匹配器 (FastZeroShotMatcher)
    # ==========================================================
    # scorer = ScorePredictor() 
    refiner = PoseRefinePredictor() 
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=None, refiner=refiner, glctx=glctx)
    
    print("🚀 加载 3D 零样本纯几何粗筛器 (FastZeroShotMatcher)...")
    matcher_pkl_path = "/root/Toothtrack/demo_data/ztooth/zero_shot_db.pkl"
    matcher = FastZeroShotMatcher(pkl_path=matcher_pkl_path, alpha=0.5)
    # ==========================================================

    reader = YcbineoatReader(video_dir=args.test_scene_dir, zfar=np.inf)
    video_writer = cv2.VideoWriter(os.path.join(output_dir, "track_distill.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 30, (reader.W, reader.H))
    previous_pose = None

    try:
        for i in range(len(reader.color_files)):
            color, frame_name, H, W = reader.get_color(i), reader.id_strs[i] + ".png", reader.get_color(i).shape[0], reader.get_color(i).shape[1]
            torch.cuda.synchronize()
            t1 = time.time()
            
            with torch.no_grad():
                if i == 0:
                    if args.use_pred_init:
                        initial_mask_path = os.path.join(args.test_scene_dir, "mask", reader.id_strs[i] + ".png")
                        initial_mask = cv2.imread(initial_mask_path, cv2.IMREAD_GRAYSCALE) > 0 
                        ys, xs = np.where(initial_mask)
                        x_min, x_max = xs.min(), xs.max()
                        y_min, y_max = ys.min(), ys.max()
                        w = max(x_max - x_min, 10)
                        h = max(y_max - y_min, 10)
                    else:
                        pose_gt = np.load(os.path.join(args.test_scene_dir, "annotated_pose", reader.id_strs[i] + ".npy"))
                        previous_pose = pose_gt
                        x_min, y_min, w, h = get_bbox_from_pose(previous_pose, reader.K, mesh.vertices)
                else:
                    x_min, y_min, w, h = get_bbox_from_pose(previous_pose, reader.K, mesh.vertices)
                
                c_x, c_y = x_min + w / 2.0, y_min + h / 2.0
                crop_size = max(w, h) * 1.2 
                
                M = cv2.getRotationMatrix2D((c_x, c_y), 0, 160.0 / crop_size)
                M[0, 2] += 80.0 - c_x
                M[1, 2] += 80.0 - c_y
                
                rgb_crop = cv2.warpAffine(color, M, (160, 160), flags=cv2.INTER_LINEAR, borderValue=(0,0,0))
                rgb_tensor = torch.from_numpy(rgb_crop).float().permute(2,0,1) / 255.0
                
                M_3x3 = np.vstack([M, [0, 0, 1]])
                K_crop = M_3x3 @ reader.K
                K_inv = np.linalg.inv(K_crop)
                u, v = np.meshgrid(np.arange(160), np.arange(160))
                uv1 = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3)
                unnorm_rays = (K_inv @ uv1.T).T.reshape(160, 160, 3)
                ray_map = unnorm_rays / np.linalg.norm(unnorm_rays, axis=-1, keepdims=True)
                ray_tensor = torch.from_numpy(ray_map).float().permute(2,0,1)
                
                inputs_6c = torch.cat([rgb_tensor, ray_tensor], dim=0).unsqueeze(0).to(device)
                Z_base = reader.K[0, 0] * (DISTILL_PHYSICAL_WIDTH / crop_size)
                
                shape_weight_raw, mask_pred, delta_z_scalar = model_expert(inputs_6c)
                
                MAX_Z_CORRECTION = 0.03
                THICKNESS_FACTOR = 0.0075
                shape_weight = torch.tanh(shape_weight_raw)
                delta_z = torch.tanh(delta_z_scalar.view(-1, 1, 1, 1)) * MAX_Z_CORRECTION
                D_pred = Z_base + delta_z + shape_weight * THICKNESS_FACTOR
                
                # 这里的 mask 和 depth 是精准裁剪到 160x160 的特征
                mask_crop_np = (mask_pred[0, 0] > 0.5).cpu().numpy().astype(np.uint8)
                depth_crop_np = D_pred[0, 0].cpu().numpy()
                depth_crop_np = depth_crop_np * mask_crop_np
                
                M_inv = cv2.invertAffineTransform(M)
                full_mask = cv2.warpAffine(mask_crop_np, M_inv, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
                full_depth = cv2.warpAffine(depth_crop_np, M_inv, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
                
                current_mask = full_mask.astype(bool)
                current_depth = full_depth

            with SuppressPrint(): 
                torch.cuda.synchronize()
                t_track_start = time.time()
                
                if i == 0:
                    if args.use_pred_init:
                        # ==========================================================
                        # 🌟 终极位姿数学修正：解耦旋转(R)与平移(T)
                        # ==========================================================
                        # 使用 sys.stderr.write 绕过 SuppressPrint，让终端能看到提示
                        sys.stderr.write("\n🚀 启动零样本纯几何粗筛定位...\n")
                        
                        # 1. 送入匹配器 (返回离线空间下的 cam2world 位姿)
                        initial_pose_cam2world = matcher.match(mask_crop_np, depth_crop_np)
                        
                        if initial_pose_cam2world is not None:
                            sys.stderr.write("✅ 几何粗筛成功！利用预测深度图解算真实物理平移(T)...\n")
                            
                            # 2. 坐标系求逆：把 相机到世界(cam2world) 变成 物体到相机(obj2cam)
                            obj2cam_template = np.linalg.inv(initial_pose_cam2world)
                            
                            # 3. 取出模板最核心的资产：正确的 3D 旋转姿态 (R)
                            R_pred = obj2cam_template[:3, :3]
                            
                            # 4. 动态计算真实平移：获取当前画面牙齿的绝对物理深度 (tz)
                            valid_depth = current_depth[current_mask]
                            real_tz = np.median(valid_depth) if len(valid_depth) > 0 else 0.1 
                            
                            # 5. 针孔相机极线反投影：根据 2D BBox 中心点推算真实的真实 tx, ty
                            # 公式：X = (u - cx) * Z / fx
                            real_tx = (c_x - reader.K[0, 2]) * real_tz / reader.K[0, 0]
                            real_ty = (c_y - reader.K[1, 2]) * real_tz / reader.K[1, 1]
                            
                            # 6. 重新拼装出真正属于当前帧画面的完美初始位姿
                            real_initial_pose = np.eye(4, dtype=np.float32)
                            real_initial_pose[:3, :3] = R_pred
                            real_initial_pose[:3, 3] = [real_tx, real_ty, real_tz]
                            
                            # 直接喂给 FoundationPose，此时初始重叠率绝对超过 80%！
                            est.pose_last = torch.tensor(real_initial_pose, device=device, dtype=torch.float32).unsqueeze(0)
                            
                            pose_centered = est.track_one(rgb=color, depth=current_depth, K=reader.K, iteration=args.est_refine_iter)
                            pose = pose_centered @ est.get_tf_to_centered_mesh().data.cpu().numpy().reshape(4, 4)
                        else:
                            sys.stderr.write("❌ 警告：未匹配到任何有效模板！\n")
                            pose = np.eye(4)
                        # ==========================================================

                    else:
                        est.pose_last = torch.tensor(pose_gt @ np.linalg.inv(est.get_tf_to_centered_mesh().data.cpu().numpy().reshape(4, 4)), device=device, dtype=torch.float32).unsqueeze(0)
                        est.track_one(rgb=color, depth=current_depth, K=reader.K, iteration=1)
                        pose = pose_gt
                else:
                    pose = est.track_one(rgb=color, depth=current_depth, K=reader.K, iteration=args.track_refine_iter) @ est.get_tf_to_centered_mesh().data.cpu().numpy().reshape(4, 4)
                
                previous_pose = pose
                
                torch.cuda.synchronize()
                t_track_end = time.time()
                track_time_ms = (t_track_end - t_track_start) * 1000.0
                all_track_times.append(track_time_ms)

            torch.cuda.synchronize()
            t2 = time.time()
            frame_time_ms = (t2 - t1) * 1000.0
            all_frame_times.append(frame_time_ms)
            
            # (移出计时区) 可视化深度拼图
            mask_crop_255 = mask_crop_np * 255
            vis_rgb = cv2.cvtColor(rgb_crop.astype(np.uint8), cv2.COLOR_RGB2BGR)
            vis_mask_3c = cv2.cvtColor(mask_crop_255, cv2.COLOR_GRAY2BGR)
            vis_depth = get_auto_color(depth_crop_np, mask_crop_255)
            
            concat_img = np.hstack([vis_rgb, vis_mask_3c, vis_depth])
            cv2.putText(concat_img, 'Crop RGB', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(concat_img, 'Pred Mask', (170, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(concat_img, 'Pred Depth', (330, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(os.path.join(mask_depth_output_dir, f"{reader.id_strs[i]}.png"), concat_img)
            
            vis = create_visualization(color, pose, to_origin, reader.K, bbox, fps=1/(t2-t1), render_3d=True, mesh_dir=os.path.dirname(args.mesh_file), main_mesh=mesh, center_pose=pose @ np.linalg.inv(to_origin))
            
            frame_error = None
            err_str = " | 纯推理模式"
            val_str = "N/A"
            
            if not args.no_eval:
                if ball_centroids is not None and frame_name in gt_data:
                    try:
                        rvec_v, _ = cv2.Rodrigues(pose[:3,:3]); tvec_v = pose[:3,3]
                        pts_proj, _ = cv2.projectPoints(ball_centroids, rvec_v, tvec_v, reader.K, None)
                        pts_gt = np.array([gt_data[frame_name][f'ball_{j}'] for j in range(1, 5)], dtype=np.float32)
                        from scipy.spatial.distance import cdist; from scipy.optimize import linear_sum_assignment
                        dist_matrix = cdist(pts_proj.squeeze(), pts_gt)
                        row_ind, col_ind = linear_sum_assignment(dist_matrix)
                        frame_error = dist_matrix[row_ind, col_ind].mean()
                        all_frame_errors.append(frame_error)
                        cv2.putText(vis, f"Err: {frame_error:.2f}px", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)
                    except: pass
                
                pose_gt_path = os.path.join(gt_pose_dir, f"{reader.id_strs[i]}.npy")
                depth_gt_path = os.path.join(gt_depth_dir, f"{reader.id_strs[i]}.npy")
                
                if os.path.exists(pose_gt_path) and os.path.exists(depth_gt_path):
                    pose_gt_val = np.load(pose_gt_path)
                    depth_gt_val = np.load(depth_gt_path) 
                    
                    te_val = calc_te(pose, pose_gt_val)
                    re_val = calc_re(pose, pose_gt_val)
                    add_val = calc_add(pose, pose_gt_val, mesh.vertices)
                    
                    eval_metrics['TE'].append(te_val)
                    eval_metrics['RE'].append(re_val)
                    eval_metrics['ADD'].append(add_val)
                    
                    if depth_gt_val.shape != current_mask.shape:
                        depth_gt_val = cv2.resize(depth_gt_val, (W, H), interpolation=cv2.INTER_NEAREST)
                    
                    gt_mask_val = depth_gt_val > 0
                    intersection = np.logical_and(gt_mask_val, current_mask).sum()
                    union = np.logical_or(gt_mask_val, current_mask).sum()
                    iou_val = intersection / union if union > 0 else 0
                    eval_metrics['IoU'].append(iou_val)
                    
                    if gt_mask_val.sum() > 0:
                        mae_val = np.abs(current_depth[gt_mask_val] - depth_gt_val[gt_mask_val]).mean() * 1000.0
                        eval_metrics['MAE'].append(mae_val)
                    else:
                        mae_val = 0.0
                        
                    err_str = f" | Err: {frame_error:.4f}px | ADD:{add_val:.2f}mm | IoU:{iou_val:.2f}" if frame_error else f" | ADD:{add_val:.2f}mm | IoU:{iou_val:.2f}"
                    val_str = f"{frame_error:.4f},{te_val:.4f},{re_val:.4f},{add_val:.4f},{iou_val:.4f},{mae_val:.4f}" if frame_error else f"N/A,{te_val:.4f},{re_val:.4f},{add_val:.4f},{iou_val:.4f},{mae_val:.4f}"
                else:
                    err_str = f" | Err: {frame_error:.4f}px" if frame_error else " | 无GT数据"
                    val_str = f"{frame_error:.4f},N/A,N/A,N/A,N/A,N/A" if frame_error else "N/A,N/A,N/A,N/A,N/A,N/A"
                
            cv2.imwrite(os.path.join(img_output_dir, f"{reader.id_strs[i]}.png"), vis[..., ::-1])
            video_writer.write(vis[..., ::-1])
            
            real_frame_id = i + 1
            print(f"🔹 帧 {real_frame_id:04d} 完成 | Track: {track_time_ms:.1f}ms | Total: {frame_time_ms:.1f}ms{err_str}")
            txt_file.write(f"{real_frame_id},{val_str}\n")
            txt_file.flush() 

            if not args.no_eval and real_frame_id == 800 and all_frame_errors:
                err_arr = np.array(all_frame_errors)
                print("\n" + "="*50)
                print("⏳ 前 800 帧 追踪统计")
                print(f"📊 平均像素误差 (Mean):   {np.mean(err_arr):.4f} px")
                print(f"📈 中位数误差 (Median): {np.median(err_arr):.4f} px")
                print(f"✨ 极高精度比例 (<2px):  {np.sum(err_arr < 2.0) / len(err_arr) * 100:.2f}%")
                print(f"✨ 优秀精度比例 (<3px):  {np.sum(err_arr < 3.0) / len(err_arr) * 100:.2f}%")
                print("="*50 + "\n")

    finally:
        video_writer.release()
        if 'txt_file' in locals() and not txt_file.closed:
            if not args.no_eval:
                if all_frame_errors:
                    err_arr = np.array(all_frame_errors)
                    txt_file.write("\n" + "="*60 + "\n")
                    txt_file.write("🏆 [视觉特征能力] 大盘留档统计\n")
                    txt_file.write(f"📊 2D 重投影误差 (Mean):  {np.mean(err_arr):.4f} px\n")
                    txt_file.write(f"✨ PCK @ 2px (微雕精度): {np.sum(err_arr < 2.0) / len(err_arr) * 100:.2f}%\n")
                    txt_file.write(f"✨ PCK @ 3px (极高精度): {np.sum(err_arr < 3.0) / len(err_arr) * 100:.2f}%\n")
                    txt_file.write(f"✨ PCK @ 10px (抗脱轨率): {np.sum(err_arr < 10.0) / len(err_arr) * 100:.2f}%\n")
                    txt_file.write("="*60 + "\n")
                    
                if len(eval_metrics['TE']) > 0:
                    txt_file.write("\n" + "="*60 + "\n")
                    txt_file.write("🏆 [物理位姿级] 大盘留档统计\n")
                    txt_file.write(f"📍 平移误差 (TE): {np.mean(eval_metrics['TE']):.4f} mm\n")
                    txt_file.write(f"🔄 旋转误差 (RE): {np.mean(eval_metrics['RE']):.4f} °\n")
                    txt_file.write(f"🦷 表面误差 (ADD): {np.mean(eval_metrics['ADD']):.4f} mm\n")
                    txt_file.write("="*60 + "\n")
                    txt_file.write("\n" + "="*60 + "\n")
                    txt_file.write("🏆 [前端感知能力] 大盘留档统计\n")
                    txt_file.write(f"🎯 掩码重合度 (Mask IoU): {np.mean(eval_metrics['IoU']):.4f}\n")
                    txt_file.write(f"📏 深度预测误差 (Depth MAE): {np.mean(eval_metrics['MAE']):.4f} mm\n")
                    txt_file.write("="*60 + "\n")
            
            if len(all_frame_times) > 0:
                avg_time_ms = np.mean(all_frame_times)
                avg_track_time_ms = np.mean(all_track_times) if len(all_track_times) > 0 else 0
                fps = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0
                txt_file.write("\n" + "="*60 + "\n")
                txt_file.write("🏆 [系统推理性能] 大盘留档统计\n")
                txt_file.write(f"🎞️ 总处理帧数 (Total Frames): {len(all_frame_times)} 帧\n")
                txt_file.write(f"⏱️ 平均 Track 耗时 (Avg Track Latency): {avg_track_time_ms:.2f} ms\n")
                txt_file.write(f"⏱️ 平均单帧总耗时 (Avg Total Latency): {avg_time_ms:.2f} ms\n")
                txt_file.write(f"🚀 等效运行帧率 (Equivalent FPS): {fps:.2f} FPS\n")
                txt_file.write("="*60 + "\n")
                
            txt_file.close()