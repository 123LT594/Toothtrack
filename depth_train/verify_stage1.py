import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset_distill import DualDistillDataset 
from learning.models.student_depth_net import StudentDepthNet
# 💥 引入配置中的常量
from learning.training.training_config import DISTILL_PHYSICAL_WIDTH, DISTILL_PHYSICAL_THICKNESS, DISTILL_K_BASE

def verify_stage1_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt_path = "/root/lanyun-tmp/models/stage1_distill/models/student_stage1_ep99.pth"
    out_dir = "/root/lanyun-tmp/models/stage1_distill/verify_results_all"
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到权重文件: {ckpt_path}")

    model = StudentDepthNet().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval() 
    # ========== 加载教师网络用于特征统计 ==========
    from learning.models.refine_network import RefineNet
    teacher = RefineNet(c_in=6).to(device)
    teacher_ckpt = "/root/Toothtrack/weights/2023-10-28-18-33-37/model_best.pth"
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device), strict=False)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    # =====================================================
    
    dataset = DualDistillDataset(is_training=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0) 
    
    MAX_Z_CORRECTION = 0.03
    
    print(f"🚀 开始全量数据集测试！图像将保存在 {out_dir}")
    print("=" * 50)
    
    # 用于记录所有帧指标的列表
    all_ious = []
    all_maes = []
    loss_feat_list = []          # 训练等价 loss_feat
    teacher_std_list = []        # 教师特征标准差
    rmse_list = []               # 有效区域特征 RMSE

    # 最终统计输出函数
    def print_final_stats():
        if loss_feat_list:
            arr_loss = np.array(loss_feat_list)
            arr_rmse = np.array(rmse_list)
            arr_teacher_std = np.array(teacher_std_list)
            
            mean_loss = arr_loss.mean()
            mean_rmse = arr_rmse.mean()
            mean_teacher_std = arr_teacher_std.mean()
            ratio_percent = (mean_rmse / mean_teacher_std * 100) if mean_teacher_std > 0 else 0.0
            
            print("\n" + "="*60)
            print("📊 特征蒸馏损失 (全帧统计)")
            print(f"  帧数: {len(arr_loss)}")
            print(f"  loss_feat 均值: {mean_loss:.4f}")
            print(f"  loss_feat 标准差: {arr_loss.std():.4f}")
            print(f"  RMSE 均值: {mean_rmse:.3f}")
            print(f"  教师特征 std 均值: {mean_teacher_std:.3f}")
            print(f"  🔥 RMSE / 教师 std = {ratio_percent:.2f}%")
            print(f"  (该比例越低，蒸馏越紧密，<50% 即表示有效蒸馏)")
            print("="*60)

    # 注册 Ctrl+C 信号处理
    import signal
    def signal_handler(sig, frame):
        print("\n⚠️ 收到中断信号，正在输出已处理帧的统计...")
        print_final_stats()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            frame_name = dataset.frames[i]
            
            inputs_6c = batch["inputs_6c"].to(device).float()
            rgb_crop = batch["rgb_crop"].to(device).float()
            depth_gt = batch["depth_gt"].to(device).float()
            mask_gt = batch["mask_gt"].to(device).float()
            Z_base = batch["Z_base"].to(device).view(-1, 1, 1, 1).float()
            
            shape_weight_raw, mask_pred, delta_z_scalar = model(inputs_6c)
            
            shape_weight = torch.tanh(shape_weight_raw)
            delta_z = torch.tanh(delta_z_scalar.view(-1, 1, 1, 1)) * MAX_Z_CORRECTION
            # 使用导入的真实物理厚度常量
            D_pred = Z_base + delta_z + shape_weight * DISTILL_PHYSICAL_THICKNESS

            # ========== 每帧计算训练等价 loss_feat ==========
            unnorm_rays = batch["unnorm_rays"].to(device).float()
            XYZ_scale = DISTILL_PHYSICAL_WIDTH
            XYZ_pred_norm = (D_pred * unnorm_rays) / XYZ_scale
            XYZ_gt_norm = (depth_gt * unnorm_rays) / XYZ_scale
            A_pred = torch.cat([rgb_crop, XYZ_pred_norm], dim=1)
            A_gt = torch.cat([rgb_crop, XYZ_gt_norm], dim=1)

            F_s = teacher.extract_distill_feature(A_pred)
            F_t = teacher.extract_distill_feature(A_gt)
            mask_feat_down = F.interpolate(mask_gt, size=F_t.shape[2:], mode='nearest')
            mse_masked = (F_s - F_t).pow(2) * mask_feat_down
            loss_feat_val = mse_masked.sum() / (mask_feat_down.sum() * F_s.shape[1] + 1e-8)
            loss_feat_list.append(loss_feat_val.item())
            teacher_std_list.append(F_t.std().item())
            rmse_list.append(torch.sqrt(loss_feat_val).item())

            # 第一帧额外打印详细特征统计
            if i == 0:
                rmse_feat = torch.sqrt(loss_feat_val)
                print(f"\n🔍 首帧教师特征 std: {F_t.std().item():.3f}")
                print(f"🔍 首帧训练等价 loss_feat: {loss_feat_val.item():.3f}")
                print(f"🔍 首帧有效区域特征 RMSE: {rmse_feat.item():.3f}")
                print(f"🔍 RMSE / 教师std = {rmse_feat.item() / F_t.std().item():.2%}")
            # ===================================================

            valid_gt_mask = mask_gt[0, 0] > 0.5
            if valid_gt_mask.sum() > 0:
                print(f"\n--- {frame_name} ---")
                print(f"📌 Z_base 基准距离 : {Z_base.item():.4f}")
                print(f"🎯 真实 GT (中位数): {np.median(depth_gt[0,0][valid_gt_mask].cpu().numpy()):.4f}")
                print(f"🤖 预测 Pred(中位数): {np.median(D_pred[0,0][valid_gt_mask].cpu().numpy()):.4f}")
                
                sw_min = shape_weight[0,0][valid_gt_mask].min().item()
                sw_max = shape_weight[0,0][valid_gt_mask].max().item()
                print(f"⚙️ 神经元活跃度 (Tanh): [{sw_min:.3f} 到 {sw_max:.3f}]")
                
                gt_min = depth_gt[0,0][valid_gt_mask].min().item()
                gt_max = depth_gt[0,0][valid_gt_mask].max().item()
                pred_min = D_pred[0,0][valid_gt_mask].min().item()
                pred_max = D_pred[0,0][valid_gt_mask].max().item()
                
                gt_drop = gt_max - gt_min
                pred_drop = pred_max - pred_min
                
                print(f"⛰️ 真值 (GT) 物理起伏: [{gt_min:.4f} 到 {gt_max:.4f}] (最大落差: {gt_drop:.4f})")
                print(f"⛰️ 预测 (Pred)物理起伏: [{pred_min:.4f} 到 {pred_max:.4f}] (最大落差: {pred_drop:.4f})")

                # 计算 Mask IoU 和 Depth MAE
                pred_mask_bool = (mask_pred[0, 0] > 0.5)
                intersection = (pred_mask_bool & valid_gt_mask).sum().item()
                union = (pred_mask_bool | valid_gt_mask).sum().item()
                iou = intersection / union if union > 0 else 0.0
                all_ious.append(iou)
                
                mae_mm = torch.abs(D_pred[0,0][valid_gt_mask] - depth_gt[0,0][valid_gt_mask]).mean().item() * 1000.0
                all_maes.append(mae_mm)
                
                print(f"🎯 掩码重合度 (Mask IoU): {iou:.4f}")
                print(f"📏 深度平均误差 (MAE): {mae_mm:.4f} mm")

            # 可视化保存
            vis_rgb = (rgb_crop[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR) 
            vis_mask_pred = (mask_pred[0, 0].cpu().numpy() * 255).astype(np.uint8)
            vis_mask_gt = (mask_gt[0, 0].cpu().numpy() * 255).astype(np.uint8)
            vis_mask_pred_3c = cv2.cvtColor(vis_mask_pred, cv2.COLOR_GRAY2BGR)
            vis_mask_gt_3c = cv2.cvtColor(vis_mask_gt, cv2.COLOR_GRAY2BGR)
            
            def get_auto_color(depth_t, mask_t):
                d_np = depth_t[0, 0].cpu().numpy()
                m_np = mask_t[0, 0].cpu().numpy() > 0.5 
                vis = np.zeros_like(d_np, dtype=np.uint8)
                if m_np.sum() > 0:
                    valid = d_np[m_np]
                    p_min, p_max = np.percentile(valid, 2), np.percentile(valid, 98)
                    if p_max - p_min > 1e-4:
                        norm = np.clip((d_np - p_min) / (p_max - p_min), 0, 1)
                        vis = (norm * 255).astype(np.uint8)
                    else:
                        vis[m_np] = 127 
                color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                color[~m_np] = 0
                return color

            vis_depth_pred = get_auto_color(D_pred, mask_gt)
            vis_depth_gt = get_auto_color(depth_gt, mask_gt)
            
            concat_img = np.hstack([vis_bgr, vis_mask_pred_3c, vis_mask_gt_3c, vis_depth_pred, vis_depth_gt])
            cv2.putText(concat_img, 'RGB', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(concat_img, 'Pred Mask', (160 + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(concat_img, 'GT Mask', (320 + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(concat_img, 'Pred Depth', (480 + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(concat_img, 'GT Depth', (640 + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            save_path = os.path.join(out_dir, f"{frame_name}.png")
            cv2.imwrite(save_path, concat_img)

    # 正常结束后输出特征蒸馏统计
    print_final_stats()

    # 原有的全量总结
    if len(all_ious) > 0:
        print("\n" + "=" * 50)
        print("🏆 [全量测试大盘总结] 🏆")
        print(f"📊 总计测试帧数: {len(all_ious)} 帧")
        print(f"🎯 平均掩码重合度 (mIoU): {np.mean(all_ious):.4f}")
        print(f"📏 平均深度误差 (MAE):   {np.mean(all_maes):.4f} mm")
        print("=" * 50 + "\n")

if __name__ == "__main__":
    verify_stage1_all()