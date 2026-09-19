import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset_synthetic import SyntheticPretrainDataset 
from learning.models.student_depth_net import StudentDepthNet
from learning.training.training_config import MAX_Z_RATIO, SHAPE_SCALE_RATIO

def verify_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 锁定最新保存的权重路径
    ckpt_path = "/root/lanyun-tmp/models/stage0_pretrain/models/student_stage0_ep49.pth"
    out_dir = "/root/lanyun-tmp/models/stage0_pretrain/verify_results"
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到权重文件: {ckpt_path}，请确认是否跑完了第49个Epoch！")

    print("⏳ 正在加载预训练模型...")
    model = StudentDepthNet().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval() # 💥 开启评估模式，冻结 BatchNorm 和 Dropout
    
    # 2. 加载测试数据集 (is_training=False 关闭仿射缩放和颜色破坏)
    print("⏳ 正在初始化测试数据...")
    dataset = SyntheticPretrainDataset(is_training=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    
    print(f"🚀 开始推理验证！结果将保存在: {out_dir}")
    
    # 我们只抽取 10 张图进行可视化验证
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 10:
                break
                
            inputs_6c = batch["inputs_6c"].to(device).float()
            rgb_crop = batch["rgb_crop"].to(device).float()
            depth_gt = batch["depth_gt"].to(device).float()
            mask_gt = batch["mask_gt"].to(device).float()
            Z_base = batch["Z_base"].to(device).view(-1, 1, 1, 1).float()
            
            # 🌟 修复拼写错误: ynamic_width -> dynamic_width
            dynamic_width = batch["mesh_width"].to(device).view(-1, 1, 1, 1).float()

            # 模型推理！
            shape_weight_raw, mask_pred, delta_z_scalar = model(inputs_6c)

            # 🌟 修复漏网之鱼: 补上缺失的 SHAPE_SCALE_RATIO，完全对齐训练端
            shape_weight = torch.tanh(shape_weight_raw) * dynamic_width * SHAPE_SCALE_RATIO
            
            delta_z_rel = torch.tanh(delta_z_scalar.view(-1, 1, 1, 1)) * MAX_Z_RATIO
            z_global = Z_base * (1.0 + delta_z_rel)
            
            # 终极深度计算
            D_pred = z_global + shape_weight
            
            # =============== 📊 核心诊断：让数据说话 ===============
            valid_gt_mask = mask_gt[0, 0] > 0.5
            if valid_gt_mask.sum() > 0:
                print(f"\n--- 🦷 Batch {i} 物理深度诊断 ---")
                print(f"📌 Z_base 基准距离 : {Z_base.item():.4f}")
                print(f"🎯 真实 GT (中位数): {np.median(depth_gt[0,0][valid_gt_mask].cpu().numpy()):.4f}")
                print(f"🤖 预测 Pred(中位数): {np.median(D_pred[0,0][valid_gt_mask].cpu().numpy()):.4f}")
                
                # 1. 监控网络内部的活跃度 (无量纲)
                sw_min = shape_weight[0,0][valid_gt_mask].min().item()
                sw_max = shape_weight[0,0][valid_gt_mask].max().item()
                print(f"⚙️ 神经元活跃度 (Shape): [{sw_min:.4f} 到 {sw_max:.4f}]")
                if abs(sw_max - sw_min) < 0.001:
                    print("⚠️ 警告：模型预测的是一个没有凹凸的纯平平面！")
                
                # 2. 💥 绝对物理起伏极值对比 (苹果比苹果)
                gt_min = depth_gt[0,0][valid_gt_mask].min().item()
                gt_max = depth_gt[0,0][valid_gt_mask].max().item()
                pred_min = D_pred[0,0][valid_gt_mask].min().item()
                pred_max = D_pred[0,0][valid_gt_mask].max().item()
                
                gt_drop = gt_max - gt_min
                pred_drop = pred_max - pred_min
                
                print(f"⛰️ 真值 (GT) 物理起伏: [{gt_min:.4f} 到 {gt_max:.4f}] (最大落差: {gt_drop:.4f} m)")
                print(f"⛰️ 预测 (Pred)物理起伏: [{pred_min:.4f} 到 {pred_max:.4f}] (最大落差: {pred_drop:.4f} m)")

            # =============== 🎨 独立自适应可视化 ===============
            vis_rgb = (rgb_crop[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR) 
            vis_mask_pred = (mask_pred[0, 0].cpu().numpy() * 255).astype(np.uint8)
            vis_mask_gt = (mask_gt[0, 0].cpu().numpy() * 255).astype(np.uint8)
            vis_mask_pred_3c = cv2.cvtColor(vis_mask_pred, cv2.COLOR_GRAY2BGR)
            vis_mask_gt_3c = cv2.cvtColor(vis_mask_gt, cv2.COLOR_GRAY2BGR)
            
            def colorize_depth(depth_tensor, mask_tensor, vmin=None, vmax=None):
                """绝对自适应拉伸：无视整体偏差，只要有形状，就给你画出彩虹！"""
                d_np = depth_tensor[0, 0].cpu().numpy()
                m_np = mask_tensor[0, 0].cpu().numpy() > 0.5 
                vis = np.zeros_like(d_np, dtype=np.uint8)
                
                if m_np.sum() > 0:
                    valid_d = d_np[m_np]
                    if vmin is None or vmax is None:
                        vmin = np.percentile(valid_d, 2)
                        vmax = np.percentile(valid_d, 98)
                        
                    if vmax <= vmin:
                        vmax = vmin + 1e-5
                        
                    norm_d = np.clip((d_np - vmin) / (vmax - vmin), 0, 1)
                    vis = (norm_d * 255).astype(np.uint8)
                
                color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                color[~m_np] = 0
                return color

            # 🌟 修复：使用 GT 的绝对尺度作为统一着色标尺，才能进行像素级颜色比对！
            gt_np = depth_gt[0, 0].cpu().numpy()
            m_gt = mask_gt[0, 0].cpu().numpy() > 0.5
            if m_gt.sum() > 0:
                vmin_shared = np.percentile(gt_np[m_gt], 2)
                vmax_shared = np.percentile(gt_np[m_gt], 98)
            else:
                vmin_shared, vmax_shared = None, None
            
            # 都用同一张 GT Mask 确保边缘形状对齐，强制同标尺
            vis_depth_pred = colorize_depth(D_pred, mask_gt, vmin_shared, vmax_shared)
            vis_depth_gt = colorize_depth(depth_gt, mask_gt, vmin_shared, vmax_shared)
            
            # 拼图并打上文字标签
            concat_img = np.hstack([vis_bgr, vis_mask_pred_3c, vis_mask_gt_3c, vis_depth_pred, vis_depth_gt])
            cv2.putText(concat_img, 'RGB', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(concat_img, 'Pred Mask', (160 + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(concat_img, 'GT Mask', (320 + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(concat_img, 'Pred Depth', (480 + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(concat_img, 'GT Depth', (640 + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            save_path = os.path.join(out_dir, f"test_result_{i:02d}.png")
            cv2.imwrite(save_path, concat_img)

if __name__ == "__main__":
    verify_model()