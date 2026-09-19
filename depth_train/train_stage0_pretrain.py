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

# ---------- 超参数 ----------
DEEPTH_LOSS_WEIGHT = 30.0      # 深度（相对形状）损失加权系数
LOSS_Z_WEIGHT = 0.5          # delta_z 小范围微调监督权重（解耦后大幅降低）
SHAPE_REG_WEIGHT = 0.3        # shape均值正则：强制shape只表达局部起伏，不携带全局偏移
# ----------------------------

def compute_l1_ssim(pred, target, mask, window_size=11):
    """
    深度图结构化损失：🌟 相对 L1 (AbsRel) + 局部 SSIM
    彻底解决微距/远景尺度差异。让 30mm 处的 1mm 误差权重等于 150mm 处的 5mm 误差。
    """
    abs_diff = torch.abs(pred - target) * mask
    rel_diff = abs_diff / (target.clamp(min=1e-3)) # clamp 防止除 0
    l1_loss = rel_diff.sum() / (mask.sum() + 1e-8)
    
    # 实例级归一化，使数值进入[0,1]区间，保证SSIM常数有效
    max_depth = (target * mask).max().detach() + 1e-8
    p_norm = pred / max_depth
    t_norm = target / max_depth
    
    # 局部SSIM
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
    
    return l1_loss + 2.0 * ssim_loss  # 适当提高SSIM权重，强化结构约束

def compute_bce_dice(pred, target):
    """二值交叉熵 + Dice loss，pred 应为概率值[0,1]"""
    bce = F.binary_cross_entropy(pred, target)
    intersection = (pred * target).sum()
    dice = 1 - (2. * intersection + 1e-5) / (pred.sum() + target.sum() + 1e-5)
    return bce + dice

def compute_affine_invariant_loss(pred, target, mask, window_size=11):
    """
    🌟 形状-尺度解耦核心：仿射不变深度损失
    只监督相对形状（近远比例、表面起伏），不监督全局尺度和偏移。
    先将预测和GT都归一化到「零均值、单位方差」，再计算L1+SSIM。
    """
    valid_mask = mask > 0.5
    pred_valid = pred[valid_mask]
    target_valid = target[valid_mask]
    
    if pred_valid.numel() < 10:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    
    # 仿射归一化：零均值、单位方差（仅在有效区域内计算）
    pred_mean = pred_valid.mean()
    pred_std = pred_valid.std() + 1e-8
    target_mean = target_valid.mean()
    target_std = target_valid.std() + 1e-8
    
    pred_norm = (pred - pred_mean) / pred_std
    target_norm = (target - target_mean) / target_std
    
    # 相对L1损失
    abs_diff = torch.abs(pred_norm - target_norm) * mask
    l1_loss = abs_diff.sum() / (mask.sum() + 1e-8)
    
    # 局部SSIM（归一化后计算，只对比形状结构）
    max_val = 1.0
    p_norm = pred_norm / max_val
    t_norm = target_norm / max_val
    
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

def compute_shape_mean_reg(shape_weight, mask):
    """
    🌟 shape均值正则：惩罚shape_weight在mask内的均值偏离0
    强制shape只表达局部起伏，不携带全局偏移（解耦的关键约束）
    """
    shape_masked = shape_weight * mask
    shape_mean = shape_masked.sum(dim=[1,2,3]) / (mask.sum(dim=[1,2,3]) + 1e-8)
    return torch.mean(shape_mean ** 2)

def train_stage0():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_name = "stage0_pretrain"
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
    optimizer = torch.optim.AdamW(student.parameters(), lr=2e-4, weight_decay=1e-4)
    
    writer = SummaryWriter(log_dir=log_dir)
    print(f"📁 所有数据保存至: {base_out_dir}")
    
    # 自动断点续训
    start_epoch = 0
    if os.path.exists(model_dir):
        checkpoints = [f for f in os.listdir(model_dir) if f.startswith("student_stage0_ep") and f.endswith(".pth")]
        if checkpoints:
            epochs = [int(f.split('ep')[-1].split('.pth')[0]) for f in checkpoints]
            latest_epoch = max(epochs)
            latest_ckpt = os.path.join(model_dir, f"student_stage0_ep{latest_epoch}.pth")
            print(f"🔄 加载权重: {latest_ckpt}")
            student.load_state_dict(torch.load(latest_ckpt, map_location=device))
            start_epoch = latest_epoch + 1
            print(f"🚀 从 Epoch {start_epoch} 继续训练")
    
    for epoch in range(start_epoch, 50):
        student.train()
        epoch_loss_total = 0.0
        epoch_loss_depth = 0.0
        epoch_loss_mask = 0.0
        epoch_loss_shape_reg = 0.0
        # 🌟 新增：核心指标epoch累加器
        epoch_delta_z = 0.0
        epoch_shape_std = 0.0
        epoch_shape_max = 0.0
        epoch_shape_mean = 0.0
        epoch_mae = 0.0
        epoch_iou = 0.0
        
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch [{epoch}/50]")
        for batch_idx, batch in pbar:
            inputs_6c = batch["inputs_6c"].to(device).float()
            rgb_crop = batch["rgb_crop"].to(device).float()
            depth_gt = batch["depth_gt"].to(device).float()
            mask_gt = batch["mask_gt"].to(device).float()
            Z_base = batch["Z_base"].to(device).view(-1, 1, 1, 1).float()
            
            # 网络前向
            shape_weight_raw, mask_pred, delta_z_scalar = student(inputs_6c)
            
            # 🌟 动态获取当前 batch 中真实模型的物理跨度
            dynamic_width = batch["mesh_width"].to(device).view(-1, 1, 1, 1).float()
            
            shape_weight = torch.tanh(shape_weight_raw) * dynamic_width * SHAPE_SCALE_RATIO
            
            # 🌟 全局偏移：动态计算相对推拉比例 (如 ±15%)
            delta_z_rel = torch.tanh(delta_z_scalar.view(-1, 1, 1, 1)) * MAX_Z_RATIO
            z_global = Z_base * (1.0 + delta_z_rel)
            
            # 最终深度图：完美解耦
            D_pred = z_global + shape_weight

            # ================= 🕵️ 硬核验证：只在第0个Batch打印一次 =================
            if batch_idx == 0:
                valid_mask = mask_gt[0, 0] > 0.5
                if valid_mask.sum() > 0:
                    print(f"\n[🔍 单位验证] Z_base基准: {Z_base[0,0,0,0].item():.4f}")
                    print(f"[🔍 单位验证] 相对推拉比: {delta_z_rel[0,0,0,0].item()*100:.2f}%")
                    print(f"[🔍 单位验证] 预测 Pred 范围: {D_pred[0,0][valid_mask].min().item():.4f} ~ {D_pred[0,0][valid_mask].max().item():.4f}")
                    print(f"[🔍 单位验证] 真实 GT  范围: {depth_gt[0,0][valid_mask].min().item():.4f} ~ {depth_gt[0,0][valid_mask].max().item():.4f}")
            # =======================================================================
            
            # 损失计算
            loss_mask = compute_bce_dice(mask_pred, mask_gt)
            # 🌟 形状-尺度解耦：使用仿射不变损失，只监督相对形状
            loss_depth = compute_affine_invariant_loss(D_pred, depth_gt, mask_gt)

            # =========================================================
            # 🌟 核心修复：新增阶段 0 的 delta_z 显式监督
            # 强制全局偏移头承担基准微调职责，杜绝 shape_weight "篡位"做全局平移
            # =========================================================
            depth_gt_mean = (depth_gt * mask_gt).sum(dim=[1,2,3]) / (mask_gt.sum(dim=[1,2,3]) + 1e-8)
            delta_z_gt = (depth_gt_mean.view(-1, 1, 1, 1) - Z_base) / Z_base
            loss_z = F.mse_loss(delta_z_rel, delta_z_gt)

            # 🌟 新增：shape均值正则，强制shape只表达局部起伏
            loss_shape_reg = compute_shape_mean_reg(shape_weight, mask_gt)

            loss_total = loss_mask + DEEPTH_LOSS_WEIGHT * loss_depth + LOSS_Z_WEIGHT * loss_z + SHAPE_REG_WEIGHT * loss_shape_reg
            
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            
            epoch_loss_total += loss_total.item()
            epoch_loss_depth += loss_depth.item()
            epoch_loss_mask += loss_mask.item()
            epoch_loss_shape_reg += loss_shape_reg.item()
            
            # 计算核心监控指标
            with torch.no_grad():
                valid_mask = mask_gt[0, 0] > 0.5
                if valid_mask.sum() > 0:
                    std_depth = D_pred[0, 0][valid_mask].std().item()
                    min_depth = D_pred[0, 0][valid_mask].min().item()
                    max_depth = D_pred[0, 0][valid_mask].max().item()
                    delta_z_val = delta_z_rel[0].item() * 100
                    
                    # 🌟 新增：形状与精度指标计算
                    shape_vals = shape_weight[0, 0][valid_mask]
                    shape_std = shape_vals.std().item() * 1000  # 转毫米
                    shape_max_abs = shape_vals.abs().max().item() * 1000  # 转毫米
                    shape_mean_val = shape_vals.mean().item() * 1000  # 转毫米，理想值趋近0
                    mae = torch.abs(D_pred[0,0] - depth_gt[0,0])[valid_mask].mean().item() * 1000  # 转毫米
                    
                    # 掩码IoU
                    pred_mask_bin = mask_pred[0,0] > 0.5
                    intersection = torch.logical_and(pred_mask_bin, valid_mask).sum().item()
                    union = torch.logical_or(pred_mask_bin, valid_mask).sum().item()
                    iou = intersection / (union + 1e-8)
                else:
                    std_depth = 0.0
                    min_depth = max_depth = 0.0
                    delta_z_val = 0.0
                    shape_std = 0.0
                    shape_max_abs = 0.0
                    shape_mean_val = 0.0
                    mae = 0.0
                    iou = 0.0
                
                # 累加到epoch级
                epoch_delta_z += delta_z_val
                epoch_shape_std += shape_std
                epoch_shape_max += shape_max_abs
                epoch_shape_mean += shape_mean_val
                epoch_mae += mae
                epoch_iou += iou
            
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
                        """将深度图转为JET伪彩色，可指定范围"""
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
                    
                    # 使用GT深度的范围作为统一标尺（便于比较）
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
                    writer.add_image(f"Stage0_Combined_Vis/Batch_{batch_idx}", cv2.cvtColor(concat_img, cv2.COLOR_BGR2RGB), epoch, dataformats='HWC')
        
        # Epoch 记录
        num_batches = len(dataloader)
        writer.add_scalar("Loss/Total", epoch_loss_total / num_batches, epoch)
        writer.add_scalar("Loss/Depth", epoch_loss_depth / num_batches, epoch)
        writer.add_scalar("Loss/Mask", epoch_loss_mask / num_batches, epoch)
        writer.add_scalar("Loss/ShapeReg", epoch_loss_shape_reg / num_batches, epoch)

        # 🌟 新增：核心业务指标曲线
        writer.add_scalar("Metrics/delta_z_rel(%)", epoch_delta_z / num_batches, epoch)
        writer.add_scalar("Metrics/shape_std(mm)", epoch_shape_std / num_batches, epoch)
        writer.add_scalar("Metrics/shape_max_abs(mm)", epoch_shape_max / num_batches, epoch)
        writer.add_scalar("Metrics/shape_mean(mm)", epoch_shape_mean / num_batches, epoch)
        writer.add_scalar("Metrics/depth_MAE(mm)", epoch_mae / num_batches, epoch)
        writer.add_scalar("Metrics/mask_IoU", epoch_iou / num_batches, epoch)
        
        if (epoch + 1) % 5 == 0 or epoch == 49:
            save_path = os.path.join(model_dir, f"student_stage0_ep{epoch}.pth")
            torch.save(student.state_dict(), save_path)
            print(f"💾 保存权重: {save_path}")
    
    writer.close()

if __name__ == "__main__":
    train_stage0()
