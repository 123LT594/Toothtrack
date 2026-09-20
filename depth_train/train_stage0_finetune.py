import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import psutil
from dataset_synthetic import SyntheticPretrainDataset 
from learning.models.student_depth_net import StudentDepthNet
from learning.training.training_config import MAX_Z_RATIO, SHAPE_SCALE_RATIO

# ---------- 微调超参数 ----------
# 🌟 微调模式：用阶段0预训练权重初始化，小学习率适应MAX_Z_RATIO从0.07改到0.15
FINETUNE_LR = 1e-5              # 微调学习率（是预训练2e-4的1/20，防止破坏已学好的shape/mask）
FINETUNE_EPOCHS = 15            # 微调轮数（15轮足够让delta_z头适应新的输出范围）
COSINE_T_MAX = 20               # 余弦退火周期（20轮，比微调轮数略大，保证学习率持续下降）
COSINE_ETA_MIN = 1e-6           # 最低学习率

# 预训练权重路径（阶段0训练完成的权重）
PRETRAIN_CKPT = "/root/lanyun-tmp/models/stage0_pretrain/models/student_stage0_ep49.pth"

# 损失权重（和预训练保持一致）
DEEPTH_LOSS_WEIGHT = 10.0
LOSS_Z_WEIGHT = 5.0
SHAPE_REG_WEIGHT = 2.0
SHAPE_SMOOTH_WEIGHT = 0.1
# ----------------------------

def compute_l1_ssim(pred, target, mask, window_size=11):
    """
    深度图结构化损失：相对 L1 (AbsRel) + 局部 SSIM
    """
    abs_diff = torch.abs(pred - target) * mask
    rel_diff = abs_diff / (target.clamp(min=1e-3))
    l1_loss = rel_diff.sum() / (mask.sum() + 1e-8)
    
    max_depth = (target * mask).max().detach() + 1e-8
    p_norm = pred / max_depth
    t_norm = target / max_depth
    
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pad = window_size // 2
    weight = mask
    
    sum_weight = F.avg_pool2d(weight, window_size, stride=1, padding=pad) + 1e-8
    mu_x = F.avg_pool2d(p_norm * weight, window_size, stride=1, padding=pad) / sum_weight
    mu_y = F.avg_pool2d(t_norm * weight, window_size, stride=1, padding=pad) / sum_weight
    
    sigma_x_sq = F.avg_pool2d((p_norm - mu_x)**2 * weight, window_size, stride=1, padding=pad) / sum_weight
    sigma_y_sq = F.avg_pool2d((t_norm - mu_y)**2 * weight, window_size, stride=1, padding=pad) / sum_weight
    sigma_xy = F.avg_pool2d((p_norm - mu_x) * (t_norm - mu_y) * weight, window_size, stride=1, padding=pad) / sum_weight
    
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x_sq + sigma_y_sq + C2))
    ssim_loss = 1.0 - (ssim_map * mask).sum() / (mask.sum() + 1e-8)
    
    return l1_loss + 2.0 * ssim_loss

def compute_bce_dice(pred, target):
    """二值交叉熵 + Dice loss"""
    bce = F.binary_cross_entropy(pred, target)
    intersection = (pred * target).sum()
    dice = 1 - (2. * intersection + 1e-5) / (pred.sum() + target.sum() + 1e-5)
    return bce + dice

def compute_shape_mean_reg(shape_weight, mask):
    """shape均值正则：惩罚shape_weight在mask内的均值偏离0"""
    shape_masked = shape_weight * mask
    shape_mean = shape_masked.sum(dim=[1,2,3]) / (mask.sum(dim=[1,2,3]) + 1e-8)
    return torch.mean(shape_mean ** 2)

def compute_shape_smooth_loss(shape_weight, mask):
    """shape平滑性损失（Total Variation）"""
    diff_h = torch.abs(shape_weight[:, :, :, 1:] - shape_weight[:, :, :, :-1])
    mask_h = mask[:, :, :, 1:] * mask[:, :, :, :-1]
    diff_v = torch.abs(shape_weight[:, :, 1:, :] - shape_weight[:, :, :-1, :])
    mask_v = mask[:, :, 1:, :] * mask[:, :, :-1, :]
    smooth_h = (diff_h * mask_h).sum() / (mask_h.sum() + 1e-8)
    smooth_v = (diff_v * mask_v).sum() / (mask_v.sum() + 1e-8)
    return smooth_h + smooth_v

def train_stage0_finetune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_name = "stage0_finetune"
    base_out_dir = f"/root/lanyun-tmp/models/{model_name}"
    
    model_dir = os.path.join(base_out_dir, "models")
    log_dir = os.path.join(base_out_dir, "logs")
    vis_dir = os.path.join(base_out_dir, "vis")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    dataset = SyntheticPretrainDataset(is_training=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=True)
    
    student = StudentDepthNet().to(device)
    
    # 🌟 加载阶段0预训练权重（只加载model_state_dict，不加载optimizer和scheduler）
    # 因为学习率从2e-4改成1e-5，优化器状态需要重新初始化
    if os.path.exists(PRETRAIN_CKPT):
        print(f"🔄 加载阶段0预训练权重: {PRETRAIN_CKPT}")
        ckpt = torch.load(PRETRAIN_CKPT, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            student.load_state_dict(ckpt['model_state_dict'])
            print(f"✅ 加载完整checkpoint中的model_state_dict（epoch={ckpt.get('epoch', '?')}）")
        else:
            student.load_state_dict(ckpt)
            print("✅ 加载旧格式checkpoint（纯模型权重）")
        print(f"⚠️  不加载optimizer和scheduler状态（学习率已改为{FINETUNE_LR}，重新初始化）")
    else:
        raise FileNotFoundError(f"找不到预训练权重: {PRETRAIN_CKPT}，请先完成阶段0预训练！")
    
    # 🌟 微调优化器：小学习率，防止破坏已学好的shape/mask
    optimizer = torch.optim.AdamW(student.parameters(), lr=FINETUNE_LR, weight_decay=1e-4)
    # 🌟 余弦退火：从FINETUNE_LR平滑降到COSINE_ETA_MIN
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=COSINE_T_MAX, eta_min=COSINE_ETA_MIN)
    
    writer = SummaryWriter(log_dir=log_dir)
    print(f"📁 微调数据保存至: {base_out_dir}")
    print(f"🎯 微调配置: lr={FINETUNE_LR}, epochs={FINETUNE_EPOCHS}, T_max={COSINE_T_MAX}, eta_min={COSINE_ETA_MIN}")
    print(f"📐 MAX_Z_RATIO={MAX_Z_RATIO}（从0.07改到0.15，让大偏差样本也能修正）")
    
    # 🌟 微调不支持断点续训（每次都从预训练权重开始）
    # 如果需要继续微调，可以手动修改PRETRAIN_CKPT指向微调后的权重
    
    for epoch in range(FINETUNE_EPOCHS):
        student.train()
        epoch_loss_total = 0.0
        epoch_loss_depth = 0.0
        epoch_loss_mask = 0.0
        epoch_loss_shape_reg = 0.0
        epoch_delta_z = 0.0
        epoch_delta_z_gt = 0.0
        epoch_delta_z_error = 0.0
        epoch_shape_std = 0.0
        epoch_shape_max = 0.0
        epoch_shape_mean = 0.0
        epoch_mae = 0.0
        epoch_iou = 0.0
        epoch_loss_shape_smooth = 0.0
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Finetune Epoch [{epoch}/{FINETUNE_EPOCHS}] lr={current_lr:.2e}")
        for batch_idx, batch in pbar:
            inputs_6c = batch["inputs_6c"].to(device).float()
            rgb_crop = batch["rgb_crop"].to(device).float()
            depth_gt = batch["depth_gt"].to(device).float()
            mask_gt = batch["mask_gt"].to(device).float()
            Z_base = batch["Z_base"].to(device).view(-1, 1, 1, 1).float()
            
            # 网络前向
            shape_weight_raw, mask_pred, delta_z_scalar = student(inputs_6c)
            
            # 动态获取当前 batch 中真实模型的物理跨度
            dynamic_width = batch["mesh_width"].to(device).view(-1, 1, 1, 1).float()
            
            shape_weight = torch.tanh(shape_weight_raw) * dynamic_width * SHAPE_SCALE_RATIO
            
            # 硬约束：强制减掉shape在mask内的均值
            shape_mean_val = (shape_weight * mask_gt).sum(dim=[1,2,3], keepdim=True) / (mask_gt.sum(dim=[1,2,3], keepdim=True) + 1e-8)
            shape_weight = shape_weight - shape_mean_val
            
            # 全局偏移：MAX_Z_RATIO=0.15
            delta_z_rel = torch.tanh(delta_z_scalar.view(-1, 1, 1, 1)) * MAX_Z_RATIO
            z_global = Z_base * (1.0 + delta_z_rel)
            
            # 最终深度图
            D_pred = z_global + shape_weight

            # 硬核验证：只在第0个Batch打印一次
            if batch_idx == 0:
                valid_mask = mask_gt[0, 0] > 0.5
                if valid_mask.sum() > 0:
                    print(f"\n[🔍 单位验证] Z_base基准: {Z_base[0,0,0,0].item():.4f}")
                    print(f"[🔍 单位验证] 相对推拉比: {delta_z_rel[0,0,0,0].item()*100:.2f}%")
                    print(f"[🔍 单位验证] 预测 Pred 范围: {D_pred[0,0][valid_mask].min().item():.4f} ~ {D_pred[0,0][valid_mask].max().item():.4f}")
                    print(f"[🔍 单位验证] 真实 GT  范围: {depth_gt[0,0][valid_mask].min().item():.4f} ~ {depth_gt[0,0][valid_mask].max().item():.4f}")
            
            # 损失计算
            loss_mask = compute_bce_dice(mask_pred, mask_gt)
            loss_depth = compute_l1_ssim(D_pred, depth_gt, mask_gt)

            # delta_z显式监督（用GT中位数）
            batch_size = depth_gt.shape[0]
            depth_gt_median = torch.zeros(batch_size, 1, 1, 1, device=device)
            for b in range(batch_size):
                valid = depth_gt[b, 0][mask_gt[b, 0] > 0.5]
                if valid.numel() > 0:
                    depth_gt_median[b, 0, 0, 0] = torch.median(valid)
            delta_z_gt = (depth_gt_median - Z_base) / Z_base
            loss_z = F.l1_loss(delta_z_rel, delta_z_gt)
            
            # shape均值正则
            loss_shape_reg = compute_shape_mean_reg(shape_weight, mask_gt)
            
            # shape平滑性损失
            loss_shape_smooth = compute_shape_smooth_loss(shape_weight, mask_gt)

            loss_total = (loss_mask + DEEPTH_LOSS_WEIGHT * loss_depth + LOSS_Z_WEIGHT * loss_z 
                          + SHAPE_REG_WEIGHT * loss_shape_reg + SHAPE_SMOOTH_WEIGHT * loss_shape_smooth)
            
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            
            epoch_loss_total += loss_total.item()
            epoch_loss_depth += loss_depth.item()
            epoch_loss_mask += loss_mask.item()
            epoch_loss_shape_reg += loss_shape_reg.item()
            epoch_loss_shape_smooth += loss_shape_smooth.item()
            
            # 计算核心监控指标（整个batch平均）
            with torch.no_grad():
                batch_delta_z = delta_z_rel.mean().item() * 100
                batch_delta_z_gt = delta_z_gt.mean().item() * 100
                batch_delta_z_error = (delta_z_rel - delta_z_gt).abs().mean().item() * 100
                
                batch_shape_std = 0.0
                batch_shape_max = 0.0
                batch_shape_mean = 0.0
                batch_mae = 0.0
                batch_iou = 0.0
                valid_count = 0
                for b in range(batch_size):
                    valid_mask_b = mask_gt[b, 0] > 0.5
                    if valid_mask_b.sum() > 0:
                        shape_vals_b = shape_weight[b, 0][valid_mask_b]
                        batch_shape_std += shape_vals_b.std().item() * 1000
                        batch_shape_max += shape_vals_b.abs().max().item() * 1000
                        batch_shape_mean += shape_vals_b.mean().item() * 1000
                        batch_mae += torch.abs(D_pred[b,0] - depth_gt[b,0])[valid_mask_b].mean().item() * 1000
                        pred_mask_bin_b = mask_pred[b,0] > 0.5
                        intersection = torch.logical_and(pred_mask_bin_b, valid_mask_b).sum().item()
                        union = torch.logical_or(pred_mask_bin_b, valid_mask_b).sum().item()
                        batch_iou += intersection / (union + 1e-8)
                        valid_count += 1
                if valid_count > 0:
                    batch_shape_std /= valid_count
                    batch_shape_max /= valid_count
                    batch_shape_mean /= valid_count
                    batch_mae /= valid_count
                    batch_iou /= valid_count
                
                epoch_delta_z += batch_delta_z
                epoch_delta_z_gt += batch_delta_z_gt
                epoch_delta_z_error += batch_delta_z_error
                epoch_shape_std += batch_shape_std
                epoch_shape_max += batch_shape_max
                epoch_shape_mean += batch_shape_mean
                epoch_mae += batch_mae
                epoch_iou += batch_iou
                
                # 进度条显示用第0个样本
                delta_z_val = delta_z_rel[0].item() * 100
                std_depth = D_pred[0, 0][mask_gt[0, 0] > 0.5].std().item() if (mask_gt[0, 0] > 0.5).sum() > 0 else 0.0
            
            # 进程内存
            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / 1024 / 1024
            
            pbar.set_postfix({
                'Loss': f"{loss_total.item():.3f}",
                'DepthL': f"{loss_depth.item():.3f}",
                'MaskL': f"{loss_mask.item():.3f}",
                'Std': f"{std_depth:.3f}",
                'ΔZ': f"{delta_z_val:.2f}",
                'RAM': f"{mem_mb:.0f}MB"
            })
            
            # 可视化前10个batch
            if batch_idx < 10:
                with torch.no_grad():
                    vis_rgb = (rgb_crop[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
                    vis_mask_pred = (mask_pred[0, 0].cpu().numpy() * 255).astype(np.uint8)
                    vis_mask_gt = (mask_gt[0, 0].cpu().numpy() * 255).astype(np.uint8)
                    vis_mask_pred_3c = cv2.cvtColor(vis_mask_pred, cv2.COLOR_GRAY2BGR)
                    vis_mask_gt_3c = cv2.cvtColor(vis_mask_gt, cv2.COLOR_GRAY2BGR)
                    
                    def colorize_depth(depth_tensor, mask_tensor, vmin=None, vmax=None):
                        d = depth_tensor[0, 0].cpu().numpy()
                        m = mask_tensor[0, 0].cpu().numpy() > 0.5
                        if m.sum() == 0:
                            return np.zeros_like(d, dtype=np.uint8)
                        valid_d = d[m]
                        if vmin is None or vmax is None:
                            vmin = np.percentile(valid_d, 2)
                            vmax = np.percentile(valid_d, 98)
                        if vmax <= vmin:
                            vmax = vmin + 1e-5
                        norm_d = np.clip((d - vmin) / (vmax - vmin), 0, 1)
                        vis = (norm_d * 255).astype(np.uint8)
                        color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                        color[~m] = 0
                        return color
                    
                    gt_np = depth_gt[0, 0].cpu().numpy()
                    m_gt = mask_gt[0, 0].cpu().numpy() > 0.5
                    if m_gt.sum() > 0:
                        vmin_shared = np.percentile(gt_np[m_gt], 2)
                        vmax_shared = np.percentile(gt_np[m_gt], 98)
                    else:
                        vmin_shared, vmax_shared = None, None
                    
                    vis_depth_pred = colorize_depth(D_pred, mask_gt, vmin_shared, vmax_shared)
                    vis_depth_gt = colorize_depth(depth_gt, mask_gt, vmin_shared, vmax_shared)
                    
                    concat_img = np.hstack([vis_bgr, vis_mask_pred_3c, vis_mask_gt_3c, vis_depth_pred, vis_depth_gt])
                    
                    for i, (x, label) in enumerate([(10, 'RGB'), (170, 'Pred Mask'), (330, 'GT Mask'), (490, 'Pred Depth'), (650, 'GT Depth')]):
                        cv2.putText(concat_img, label, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    cv2.imwrite(os.path.join(vis_dir, f"epoch_{epoch:03d}_{batch_idx:02d}.png"), concat_img)
                    writer.add_image(f"Finetune_Combined_Vis/Batch_{batch_idx}", cv2.cvtColor(concat_img, cv2.COLOR_BGR2RGB), epoch, dataformats='HWC')
        
        # Epoch 记录
        num_batches = len(dataloader)
        writer.add_scalar("Loss/Total", epoch_loss_total / num_batches, epoch)
        writer.add_scalar("Loss/Depth", epoch_loss_depth / num_batches, epoch)
        writer.add_scalar("Loss/Mask", epoch_loss_mask / num_batches, epoch)
        writer.add_scalar("Loss/ShapeReg", epoch_loss_shape_reg / num_batches, epoch)
        writer.add_scalar("Loss/ShapeSmooth", epoch_loss_shape_smooth / num_batches, epoch)
        writer.add_scalar("LR/current", current_lr, epoch)

        # 核心业务指标曲线
        writer.add_scalar("Metrics/delta_z_rel(%)", epoch_delta_z / num_batches, epoch)
        writer.add_scalar("Metrics/delta_z_gt(%)", epoch_delta_z_gt / num_batches, epoch)
        writer.add_scalar("Metrics/delta_z_error(%)", epoch_delta_z_error / num_batches, epoch)
        writer.add_scalar("Metrics/shape_std(mm)", epoch_shape_std / num_batches, epoch)
        writer.add_scalar("Metrics/shape_max_abs(mm)", epoch_shape_max / num_batches, epoch)
        writer.add_scalar("Metrics/shape_mean(mm)", epoch_shape_mean / num_batches, epoch)
        writer.add_scalar("Metrics/depth_MAE(mm)", epoch_mae / num_batches, epoch)
        writer.add_scalar("Metrics/mask_IoU", epoch_iou / num_batches, epoch)
        
        # 每5轮保存一次，最后一轮也保存
        if (epoch + 1) % 5 == 0 or epoch == FINETUNE_EPOCHS - 1:
            save_path = os.path.join(model_dir, f"student_stage0_finetune_ep{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'finetune_lr': FINETUNE_LR,
                'pretrain_ckpt': PRETRAIN_CKPT,
            }, save_path)
            print(f"💾 保存微调checkpoint: {save_path}")
        
        # 每个epoch结束后更新学习率
        scheduler.step()
    
    writer.close()
    print(f"\n✅ 微调完成！最终权重保存在: {model_dir}")
    print(f"📊 建议用verify_stage0.py验证微调后的效果，确认大偏差样本是否能修正")

if __name__ == "__main__":
    train_stage0_finetune()
