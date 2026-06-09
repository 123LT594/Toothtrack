import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import time
import argparse
import numpy as np
import cv2
import torch
import json
import trimesh
import timm  
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime as dt 
import pytz
from scipy.spatial.transform import Rotation as R
import nvdiffrast.torch as dr

from utils.estimater import *
from utils.datareader import *
from utils.tools import (
    create_centered_mesh, convert_pose_for_render, compute_mesh_diameter,
    compute_mesh_center, quick_verify_coordinate_system, evaluate_pose_fast,
    evaluate_metrics_fast, evaluate_metrics
)
from utils.render_3d import create_visualization

MAX_DEPTH = 0.3

# ================= 1. 你的终极双头 Transformer 网络 =================
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class SwinMultiTaskUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用你训练时的 convnext_tiny
        self.encoder = timm.create_model('convnext_tiny', pretrained=False, features_only=True)
        
        # 双头解码器
        self.m_dec4 = DecoderBlock(768, 384, 384); self.m_dec3 = DecoderBlock(384, 192, 192)
        self.m_dec2 = DecoderBlock(192, 96, 96);   self.m_dec1 = DecoderBlock(96, 0, 32)
        self.m_dec0 = DecoderBlock(32, 0, 16);     self.mask_head = nn.Conv2d(16, 1, 1)

        self.d_dec4 = DecoderBlock(768, 384, 384); self.d_dec3 = DecoderBlock(384, 192, 192) 
        self.d_dec2 = DecoderBlock(192, 96, 96);   self.d_dec1 = DecoderBlock(96, 0, 32)     
        self.d_dec0 = DecoderBlock(32, 0, 16);     self.depth_head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        features = self.encoder(x)
        e1, e2, e3, e4 = features[0], features[1], features[2], features[3]
        
        m4 = self.m_dec4(e4, e3); m3 = self.m_dec3(m4, e2); m2 = self.m_dec2(m3, e1)
        m1 = self.m_dec1(m2); m0 = self.m_dec0(m1)
        
        d4 = self.d_dec4(e4, e3); d3 = self.d_dec3(d4, e2); d2 = self.d_dec2(d3, e1)
        d1 = self.d_dec1(d2); d0 = self.d_dec0(d1)
        
        return self.mask_head(m0), torch.sigmoid(self.depth_head(d0)) * MAX_DEPTH

# ================= 2. 极致纯粹的端到端测试逻辑 =================
SAVE_VIDEO = True  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    
    parser.add_argument("--mesh_file", type=str, default=f"{code_dir}/demo_data/tooth/mesh/tooth.obj")
    parser.add_argument("--test_scene_dir", type=str, default=f"{code_dir}/demo_data/tooth")
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=2) 
    
    parser.add_argument("--full_weight", type=str, default="/root/lanyun-tmp/models/models_joint_transformer/joint_best.pth")
    args = parser.parse_args()

    output_root = "/root/lanyun-tmp/output"
    beijing_tz = pytz.timezone('Asia/Shanghai')
    output_dir = os.path.join(output_root, f"{dt.now(beijing_tz).strftime('%m%d_%H%M')}_joint_track_pure_dualhead")
    img_output_dir = os.path.join(output_dir, "img")
    os.makedirs(img_output_dir, exist_ok=True)

    set_logging_format(); set_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    glctx = dr.RasterizeCudaContext()

    print("🚀 加载终极双头专家 (ConvNeXt-Transformer)...")
    model_full = SwinMultiTaskUNet().to(device)
    model_full.load_state_dict(torch.load(args.full_weight, map_location=device))
    model_full.eval()

    ann_path = os.path.join(args.test_scene_dir + "_gt", "annotations.json")
    if not os.path.exists(ann_path): 
        ann_path = os.path.join(args.test_scene_dir, "annotations.json")
    with open(ann_path, 'r') as f:
        gt_data = json.load(f).get("annotations", {})
    all_frame_errors = []

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

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    torch.tensor([0.0]).cuda()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, glctx=glctx)
    reader = YcbineoatReader(video_dir=args.test_scene_dir, zfar=np.inf)

    video_writer = cv2.VideoWriter(os.path.join(output_dir, "track_dualhead.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 30, (reader.W, reader.H))
    history_poses = []

    MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3).astype(np.float32)
    STD = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3).astype(np.float32)

    try:
        for i in range(len(reader.color_files)):
            color = reader.get_color(i)
            frame_name = reader.id_strs[i] + ".png"
            H, W = color.shape[:2]
            t1 = time.time()
            
            with torch.no_grad():
                # =======================================================================
                # 🌟 端到端双头预测
                # =======================================================================
                pad_h = (32 - H % 32) % 32
                pad_w = (32 - W % 32) % 32
                img_padded = cv2.copyMakeBorder(color, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
                
                img_norm = (img_padded.astype(np.float32) / 255.0 - MEAN) / STD
                tensor_full = torch.from_numpy(img_norm).float().permute(2,0,1).unsqueeze(0).to(device)
                
                m_logits_full, d_preds_full = model_full(tensor_full)
                
                depth_full_raw = d_preds_full.squeeze().cpu().numpy()[:H, :W]
                m_full_np = (torch.sigmoid(m_logits_full) > 0.5).squeeze().cpu().numpy()[:H, :W].astype(np.uint8)
                
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m_full_np, connectivity=8)
                if num_labels > 1:
                    largest_mask = (labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(bool)
                else:
                    largest_mask = m_full_np.astype(bool)

                current_depth = depth_full_raw * largest_mask
                current_mask = largest_mask

            # =======================================================================
            # 🛡️ FoundationPose 追踪逻辑
            # =======================================================================
            if i == 0:
                annotated_pose_dir = os.path.join(args.test_scene_dir, "annotated_pose")
                npy_path = os.path.join(annotated_pose_dir, reader.id_strs[i] + ".npy")
                if os.path.exists(npy_path):
                    pose_gt = np.load(npy_path)
                    tf_to_center = est.get_tf_to_centered_mesh().data.cpu().numpy().reshape(4, 4)
                    centered_pose = pose_gt @ np.linalg.inv(tf_to_center)
                    est.pose_last = torch.tensor(centered_pose, device=device, dtype=torch.float32).unsqueeze(0)
                    
                    est.track_one(rgb=color, depth=current_depth, K=reader.K, iteration=1)
                    pose = pose_gt
                    status_tag = "🟢 GT初始化"
                else:
                    pose = est.register(K=reader.K, rgb=color, depth=current_depth, ob_mask=current_mask, iteration=args.est_refine_iter)
                    status_tag = "🟢 盲搜初始化"
            else:
                if current_mask.sum() > 50:
                    centered_pose = est.track_one(rgb=color, depth=current_depth, K=reader.K, iteration=args.track_refine_iter)
                    pose = centered_pose @ est.get_tf_to_centered_mesh().data.cpu().numpy().reshape(4, 4)
                    status_tag = "🟢 平稳追踪"
                else:
                    pose = est.pose_last.detach().cpu().numpy().reshape(4, 4) @ est.get_tf_to_centered_mesh().data.cpu().numpy().reshape(4, 4)
                    status_tag = "🔴 严重遮挡"
                    
            t2 = time.time()
            history_poses.append(pose)

            # =======================================================================
            # 📊 精度计算与容错处理 (完美适配钢珠缺失)
            # =======================================================================
            exact_center_pose = pose @ np.linalg.inv(to_origin)
            vis = create_visualization(color, pose, to_origin, reader.K, bbox, fps=1/(t2-t1),
                    render_3d=True, mesh_dir=mesh_dir, main_mesh=mesh, center_pose=exact_center_pose)
            
            frame_error = None
            if ball_centroids is not None and frame_name in gt_data and i < 800:
                try:
                    rvec_v, _ = cv2.Rodrigues(pose[:3,:3])
                    tvec_v = pose[:3,3]
                    # 投影 3D CAD 模型上的全部钢珠到 2D
                    pts_2d_proj, _ = cv2.projectPoints(ball_centroids, rvec_v, tvec_v, reader.K, None)
                    pts_2d_proj = pts_2d_proj.squeeze()

                    # 🌟 智能非对称解析：只读取该帧存在的 GT 钢珠
                    valid_gt_balls = []
                    for j in range(1, 5):
                        ball_key = f'ball_{j}'
                        if ball_key in gt_data[frame_name]:
                            valid_gt_balls.append(gt_data[frame_name][ball_key])
                    
                    # 只要至少看到1颗钢珠，就能算误差！
                    if len(valid_gt_balls) > 0:
                        pts_2d_gt = np.array(valid_gt_balls, dtype=np.float32)
                        from scipy.spatial.distance import cdist
                        from scipy.optimize import linear_sum_assignment
                        
                        # 匈牙利算法天生支持非对称矩阵 (例如 4个预测点 匹配 3个真值点)
                        dist_matrix = cdist(pts_2d_proj, pts_2d_gt)
                        row_ind, col_ind = linear_sum_assignment(dist_matrix)
                        
                        frame_error = dist_matrix[row_ind, col_ind].mean()
                        all_frame_errors.append(frame_error)
                        
                        for r, c in zip(row_ind, col_ind):
                            cv2.circle(vis, tuple(pts_2d_proj[r].astype(int)), 5, (255, 0, 0), -1)
                            cv2.circle(vis, tuple(pts_2d_gt[c].astype(int)), 6, (0, 255, 255), 2)
                        cv2.putText(vis, f"Joint Err: {frame_error:.2f}px", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                except Exception as e:
                    pass # 如果出现极端的矩阵无法解算等情况，安全跳过
            
            cv2.imwrite(os.path.join(img_output_dir, f"{reader.id_strs[i]}.png"), vis[..., ::-1])
            video_writer.write(vis[..., ::-1])
            
            # 🌟 安全打印日志 (解决 TypeError NoneType 报错)
            real_frame_id = i + 1
            err_str = f" | Err: {frame_error:.2f}px" if frame_error is not None else " | 无GT"
            print(f"✅ 帧 {real_frame_id:04d} ({frame_name}) 完成 | 耗时: {(t2-t1)*1000:.1f}ms{err_str} | {status_tag}")

    except KeyboardInterrupt: 
        print("\n⚠️ 收到手动中断信号 (Ctrl+C)！提前结束追踪...")
        
    finally: 
        video_writer.release()

        # 🌟 输出全套统计大盘指标
        if all_frame_errors:
            err_arr = np.array(all_frame_errors)
            mean_err = np.mean(err_arr)
            median_err = np.median(err_arr)
            rate_2px = np.sum(err_arr < 2.0) / len(err_arr) * 100
            rate_3px = np.sum(err_arr < 3.0) / len(err_arr) * 100
            
            print("\n" + "="*50)
            print(f"🏆 终极双头 Transformer 原生追踪测试")
            print(f"📊 平均像素误差 (Mean):   {mean_err:.4f} px")
            print(f"📈 中位数误差 (Median): {median_err:.4f} px")
            print(f"✨ 极高精度比例 (<2px):  {rate_2px:.2f}%")
            print(f"✨ 优秀精度比例 (<3px):  {rate_3px:.2f}%")
            print("="*50)