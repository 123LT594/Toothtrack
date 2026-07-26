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
from utils.depth_model_joint import SwinMultiTaskUNet

# =====================================================================
# 🌟 新增：顶会级 3D 几何评估计算器 (TE, RE, ADD)
# =====================================================================
def calc_te(pose_pred, pose_gt):
    """计算平移误差 (Translation Error)，返回毫米 (mm)"""
    return np.linalg.norm(pose_pred[:3, 3] - pose_gt[:3, 3]) * 1000.0

def calc_re(pose_pred, pose_gt):
    """计算旋转误差 (Rotation Error)，返回度数 (°)"""
    R_pred, R_gt = pose_pred[:3, :3], pose_gt[:3, :3]
    trace = np.trace(R_pred @ R_gt.T)
    # clip 防越界产生 nan
    return np.rad2deg(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))

def calc_add(pose_pred, pose_gt, vertices):
    """计算表面模型平均距离 (ADD)，返回毫米 (mm)"""
    pts_pred = (pose_pred[:3, :3] @ vertices.T + pose_pred[:3, 3:4]).T
    pts_gt = (pose_gt[:3, :3] @ vertices.T + pose_gt[:3, 3:4]).T
    return np.linalg.norm(pts_pred - pts_gt, axis=1).mean() * 1000.0

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
    parser.add_argument("--mesh_file", type=str, default=f"{code_dir}/demo_data/tooth/mesh/tooth.obj")
    parser.add_argument("--test_scene_dir", type=str, default=f"{code_dir}/demo_data/tooth")
    parser.add_argument("--track_refine_iter", type=int, default=1) 
    
    parser.add_argument("--weight_physical", type=str, default="/root/lanyun-tmp/models/models_joint_physical/joint_best.pth")
    args = parser.parse_args()

    output_root = "/root/lanyun-tmp/output"
    beijing_tz = pytz.timezone('Asia/Shanghai')
    output_dir = os.path.join(output_root, f"{dt.now(beijing_tz).strftime('%m%d_%H%M')}_baseline_track")
    img_output_dir = os.path.join(output_dir, "img")
    os.makedirs(img_output_dir, exist_ok=True)
    # 🌟 新增：初始化帧误差归档文本
    error_txt_path = os.path.join(output_dir, "frame_errors.txt")
    txt_file = open(error_txt_path, "w", encoding="utf-8")

    set_logging_format(); set_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    glctx = dr.RasterizeCudaContext()

    print("🚀 加载纯物理版大一统前端专家 (SwinMultiTaskUNet)...")
    model_expert = SwinMultiTaskUNet().to(device)
    model_expert.load_state_dict(torch.load(args.weight_physical, map_location=device))
    model_expert.eval()

    # ImageNet 归一化常量 (与 albumentations 严格一致)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    ann_path = os.path.join(args.test_scene_dir, "annotations.json")
    with open(ann_path, 'r') as f: gt_data = json.load(f).get("annotations", {})
    all_frame_errors = []
    all_frame_times = []
    # 🌟 新增：初始化 GT 路径和多维评测容器
    gt_pose_dir = "/root/lanyun-tmp/golden_dataset/pose"
    gt_depth_dir = "/root/lanyun-tmp/golden_dataset/depth"
    # 这里存储我们算出的所有高级指标
    eval_metrics = {'TE': [], 'RE': [], 'ADD': [], 'IoU': [], 'MAE': []}

    mesh = trimesh.load(args.mesh_file)
    model_center = compute_mesh_center(mesh) 
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    ball_centroids = [] 
    for j in [1, 2, 3, 4]:
        p = os.path.join(os.path.dirname(args.mesh_file), f"{j}.obj")
        if os.path.exists(p): ball_centroids.append(trimesh.load(p).vertices.mean(0))
    ball_centroids = np.array(ball_centroids, dtype=np.float32) if ball_centroids else None

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor() 
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, glctx=glctx)
    reader = YcbineoatReader(video_dir=args.test_scene_dir, zfar=np.inf)

    video_writer = cv2.VideoWriter(os.path.join(output_dir, "track_baseline.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 30, (reader.W, reader.H))

    try:
        for i in range(len(reader.color_files)):
            color, frame_name, H, W = reader.get_color(i), reader.id_strs[i] + ".png", reader.get_color(i).shape[0], reader.get_color(i).shape[1]
            # 🌟 1. 同步 GPU 并计时起点
            torch.cuda.synchronize()
            t1 = time.time()
            
            with torch.no_grad():
                pad_h, pad_w = (32 - H % 32) % 32, (32 - W % 32) % 32
                img_padded = cv2.copyMakeBorder(color, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
                
                # 🌟 核心修复 1：严格执行 ImageNet 归一化
                tensor_expert = torch.from_numpy(cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)).float().permute(2,0,1).unsqueeze(0).to(device) / 255.0
                tensor_expert = (tensor_expert - mean) / std
                
                m_logits, d_preds = model_expert(tensor_expert)
                
                m_np = (torch.sigmoid(m_logits) > 0.5).squeeze().cpu().numpy()[:H, :W]
                d_np = d_preds.squeeze().cpu().numpy()[:H, :W]
                
                # 🌟 核心修复 2：极简 Mask 清洗，杜绝形态学污染
                binary_mask = m_np.astype(np.uint8)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
                
                if num_labels > 1:
                    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    current_mask = (labels == largest_label).astype(bool)
                else:
                    current_mask = binary_mask.astype(bool)

                # 严格按照 test_joint_physical.py：直接裁剪，绝不引入背景的垃圾深度
                current_depth = d_np * current_mask

            # 🌟 核心修复 3：移除兜底逻辑，百分百信任物理回归模型
            with SuppressPrint(): 
                # 1. 严格同步 GPU 并记录 track 起点
                torch.cuda.synchronize()
                t_track_start = time.time()
                
                if i == 0:
                    pose_gt = np.load(os.path.join(args.test_scene_dir, "annotated_pose", reader.id_strs[i] + ".npy"))
                    est.pose_last = torch.tensor(pose_gt @ np.linalg.inv(est.get_tf_to_centered_mesh().data.cpu().numpy().reshape(4, 4)), device=device, dtype=torch.float32).unsqueeze(0)
                    est.track_one(rgb=color, depth=current_depth, K=reader.K, iteration=1)
                    pose = pose_gt
                else:
                    pose = est.track_one(rgb=color, depth=current_depth, K=reader.K, iteration=args.track_refine_iter) @ est.get_tf_to_centered_mesh().data.cpu().numpy().reshape(4, 4)
                
                # 2. 严格同步 GPU 并记录 track 终点
                torch.cuda.synchronize()
                t_track_end = time.time()
                track_time_ms = (t_track_end - t_track_start) * 1000.0
            # ==========================================================
            # 🌟 3. 同步 GPU 并计时终点
            torch.cuda.synchronize()
            t2 = time.time()
            # 🌟 新增：记录当前帧计算耗时 (ms)
            frame_time_ms = (t2 - t1) * 1000.0
            all_frame_times.append(frame_time_ms)
            vis = create_visualization(color, pose, to_origin, reader.K, bbox, fps=1/(t2-t1), render_3d=True, mesh_dir=os.path.dirname(args.mesh_file), main_mesh=mesh, center_pose=pose @ np.linalg.inv(to_origin))
            
            frame_error = None
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
            # =========================================================
            # 🌟 新增：全维度 GT 数据拉取与测评 (TE, RE, ADD, IoU, MAE)
            # =========================================================
            err_str = f" | Err: {frame_error:.4f} px" if frame_error is not None else " | Err: N/A"
            val_str = f"{frame_error:.4f}" if frame_error is not None else "N/A"
            
            # 拼接读取路径，这里用 id_strs 作为文件名
            pose_gt_path = os.path.join(gt_pose_dir, f"{reader.id_strs[i]}.npy")
            depth_gt_path = os.path.join(gt_depth_dir, f"{reader.id_strs[i]}.npy")
            
            if os.path.exists(pose_gt_path) and os.path.exists(depth_gt_path):
                pose_gt_val = np.load(pose_gt_path)
                
                # 读取深度 GT 
                depth_gt_val = np.load(depth_gt_path) 
                # 🚨 如果你的 depth 是 PNG 格式，请注销上一行，使用下面这行(假设归一化系数为1000):
                # depth_gt_val = cv2.imread(depth_gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
                
                # --- 1. 计算 3D 物理误差 ---
                te_val = calc_te(pose, pose_gt_val)
                re_val = calc_re(pose, pose_gt_val)
                add_val = calc_add(pose, pose_gt_val, mesh.vertices)
                
                eval_metrics['TE'].append(te_val)
                eval_metrics['RE'].append(re_val)
                eval_metrics['ADD'].append(add_val)
                
                # --- 2. 计算 2D Mask & Depth 误差 ---
                # 统一尺寸防崩溃
                if depth_gt_val.shape != current_mask.shape:
                    depth_gt_val = cv2.resize(depth_gt_val, (W, H), interpolation=cv2.INTER_NEAREST)
                
                gt_mask_val = depth_gt_val > 0
                
                # 计算 IoU
                intersection = np.logical_and(gt_mask_val, current_mask).sum()
                union = np.logical_or(gt_mask_val, current_mask).sum()
                iou_val = intersection / union if union > 0 else 0
                eval_metrics['IoU'].append(iou_val)
                
                # 计算 Depth MAE (仅在 GT 有效范围内计算，转换成毫米)
                if gt_mask_val.sum() > 0:
                    mae_val = np.abs(current_depth[gt_mask_val] - depth_gt_val[gt_mask_val]).mean() * 1000.0
                    eval_metrics['MAE'].append(mae_val)
                else:
                    mae_val = 0.0
                    
                # 丰富控制台日志
                err_str += f" | ADD:{add_val:.2f}mm | IoU:{iou_val:.2f}"
                val_str += f",{te_val:.4f},{re_val:.4f},{add_val:.4f},{iou_val:.4f},{mae_val:.4f}"
            else:
                # 缺失真值时留空占位
                val_str += ",N/A,N/A,N/A,N/A,N/A"
            # =========================================================
            cv2.imwrite(os.path.join(img_output_dir, f"{reader.id_strs[i]}.png"), vis[..., ::-1])
            video_writer.write(vis[..., ::-1])
            
            err_str = f" | Err: {frame_error:.4f} px" if frame_error is not None else " | Err: N/A"
            real_frame_id = i + 1
            # 🌟 打印时把 track 耗时单独列出来！
            print(f"🔹 帧 {real_frame_id:04d} 完成 | Track 耗时: {track_time_ms:.1f}ms | 总耗时: {frame_time_ms:.1f}ms{err_str}")
            # 🌟 新增：结构化写入 txt（帧号,误差），若无 GT 则记录为 N/A
            val_str = f"{frame_error:.4f}" if frame_error is not None else "N/A"
            txt_file.write(f"{real_frame_id},{val_str}\n")
            txt_file.flush()  # 强行刷新缓冲区，防止意外中断时数据丢失

            # =================================================================
            # 🌟 新增：第 800 帧时的中期大盘数据播报
            # =================================================================
            if real_frame_id == 800 and all_frame_errors:
                err_arr = np.array(all_frame_errors)
                mean_err = np.mean(err_arr)
                median_err = np.median(err_arr)
                rate_2px = np.sum(err_arr < 2.0) / len(err_arr) * 100
                rate_3px = np.sum(err_arr < 3.0) / len(err_arr) * 100
                
                print("\n" + "="*50)
                print("⏳ 前 800 帧 追踪统计")
                print(f"📊 平均像素误差 (Mean):   {mean_err:.4f} px")
                print(f"📈 中位数误差 (Median): {median_err:.4f} px")
                print(f"✨ 极高精度比例 (<2px):  {rate_2px:.2f}%")
                print(f"✨ 优秀精度比例 (<3px):  {rate_3px:.2f}%")
                print("="*50 + "\n")

    finally:
        video_writer.release()
        if 'txt_file' in locals() and not txt_file.closed:
            if all_frame_errors:
                err_arr = np.array(all_frame_errors)
                
                # 计算各种阈值下的 PCK (Percentage of Correct Keypoints)
                pck_2px = np.sum(err_arr < 2.0) / len(err_arr) * 100
                pck_3px = np.sum(err_arr < 3.0) / len(err_arr) * 100
                pck_10px = np.sum(err_arr < 10.0) / len(err_arr) * 100
                
                txt_file.write("\n" + "="*60 + "\n")
                txt_file.write("🏆 [视觉特征能力] 大盘留档统计\n")
                txt_file.write(f"📊 2D 重投影误差 (Mean):  {np.mean(err_arr):.4f} px\n")
                txt_file.write(f"✨ PCK @ 2px (微雕精度): {pck_2px:.2f}%\n")
                txt_file.write(f"✨ PCK @ 3px (极高精度): {pck_3px:.2f}%\n")
                txt_file.write(f"✨ PCK @ 10px (抗脱轨率): {pck_10px:.2f}%\n")
                txt_file.write("="*60 + "\n")
                
                # 只有当成功读取到物理位姿真值时，才打印以下报告
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
                # =========================================================
                # 🌟 新增：系统推理与时间性能统计
                # =========================================================
                if len(all_frame_times) > 0:
                    avg_time_ms = np.mean(all_frame_times)
                    fps = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0
                    
                    txt_file.write("\n" + "="*60 + "\n")
                    txt_file.write("🏆 [系统推理性能] 大盘留档统计\n")
                    txt_file.write(f"🎞️ 总处理帧数 (Total Frames): {len(all_frame_times)} 帧\n")
                    txt_file.write(f"⏱️ 平均单帧耗时 (Avg Latency): {avg_time_ms:.2f} ms\n")
                    txt_file.write(f"🚀 等效运行帧率 (Equivalent FPS): {fps:.2f} FPS\n")
                    txt_file.write("="*60 + "\n")
                
            txt_file.close()