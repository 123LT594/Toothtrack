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
DEEPTH_LOSS_WEIGHT = 10.0      # 深度（相对形状）损失加权系数（从30降到10：避免delta_z梯度被淹没，保持约15:1的平衡比例）
LOSS_Z_WEIGHT = 5.0          # delta_z 全局尺度修正监督权重（L1损失，平衡点：既有足够梯度又不干扰shape/mask学习）
SHAPE_REG_WEIGHT = 2.0        # shape均值正则：强制shape只表达局部起伏，不携带全局偏移（硬约束后恒为0，保留作为保险）
SHAPE_SMOOTH_WEIGHT = 0.1     # shape平滑性损失（Total Variation）：惩罚相邻像素剧烈变化，提升表面光滑度
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
    ⚠️ [已弃用] 仿射不变深度损失 —— 阶段0不使用，原因如下：
    完全仿射不变（减均值除标准差）对全局缩放和平移完全无惩罚，
    在阶段0没有特征蒸馏等其他全局约束的情况下，网络会走捷径：
    把全局偏移藏到shape_weight里（shape_mean不趋近0），导致
    Loss/Depth下降但绝对深度MAE上升，验证效果差。
    阶段1有FoundationPose特征蒸馏做隐含全局约束，可尝试使用。
    保留此函数用于消融实验对比。
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

def compute_shape_smooth_loss(shape_weight, mask):
    """
    🌟 shape平滑性损失（Total Variation）：惩罚相邻像素的剧烈变化
    提升牙齿表面的光滑度，减少噪声状的形状输出
    """
    # 水平方向梯度
    diff_h = torch.abs(shape_weight[:, :, :, 1:] - shape_weight[:, :, :, :-1])
    mask_h = mask[:, :, :, 1:] * mask[:, :, :, :-1]
    # 垂直方向梯度
    diff_v = torch.abs(shape_weight[:, :, 1:, :] - shape_weight[:, :, :-1, :])
    mask_v = mask[:, :, 1:, :] * mask[:, :, :-1, :]
    smooth_h = (diff_h * mask_h).sum() / (mask_h.sum() + 1e-8)
    smooth_v = (diff_v * mask_v).sum() / (mask_v.sum() + 1e-8)
    return smooth_h + smooth_v

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
    # 🌟 余弦退火学习率调度：从2e-4平滑降到1e-6，减少后期震荡
    # 输入通道改为7通道（加入Z_base），需从头训练，T_max=50对应总训练轮数
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    writer = SummaryWriter(log_dir=log_dir)
    print(f"📁 所有数据保存至: {base_out_dir}")
    
    # 自动断点续训（兼容旧格式：只存模型权重；新格式：存完整checkpoint）
    start_epoch = 0
    if os.path.exists(model_dir):
        checkpoints = [f for f in os.listdir(model_dir) if f.startswith("student_stage0_ep") and f.endswith(".pth")]
        if checkpoints:
            epochs = [int(f.split('ep')[-1].split('.pth')[0]) for f in checkpoints]
            latest_epoch = max(epochs)
            latest_ckpt = os.path.join(model_dir, f"student_stage0_ep{latest_epoch}.pth")
            print(f"🔄 加载权重: {latest_ckpt}")
            ckpt = torch.load(latest_ckpt, map_location=device)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                # 🌟 新格式：完整checkpoint，同时加载模型、优化器、调度器
                student.load_state_dict(ckpt['model_state_dict'])
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                if 'scheduler_state_dict' in ckpt:
                    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                start_epoch = ckpt.get('epoch', latest_epoch) + 1
                print(f"✅ 加载完整checkpoint（含优化器+调度器状态）")
            else:
                # 旧格式：只有模型权重，优化器保持初始化状态
                student.load_state_dict(ckpt)
                start_epoch = latest_epoch + 1
                print(f"⚠️ 旧格式checkpoint（仅模型权重），优化器重新初始化")
            print(f"🚀 从 Epoch {start_epoch} 继续训练")
    
    for epoch in range(start_epoch, 50):
        student.train()
        epoch_loss_total = 0.0
        epoch_loss_depth = 0.0
        epoch_loss_mask = 0.0
        epoch_loss_shape_reg = 0.0
        # 🌟 新增：核心指标epoch累加器
        epoch_delta_z = 0.0
        epoch_delta_z_gt = 0.0       # 🌟 真实delta_z（用于验证网络是否学会）
        epoch_delta_z_error = 0.0    # 🌟 |pred - gt|，最关键的验证指标
        epoch_shape_std = 0.0
        epoch_shape_max = 0.0
        epoch_shape_mean = 0.0
        epoch_mae = 0.0
        epoch_iou = 0.0
        epoch_loss_shape_smooth = 0.0  # 🌟 shape平滑损失
        
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
            
            # 🌟 硬约束：强制减掉shape在mask内的均值，shape只表达局部起伏，不能携带全局偏移
            # 这比软正则（shape_mean_reg）更直接、更有效，彻底杜绝网络走捷径
            shape_mean_val = (shape_weight * mask_gt).sum(dim=[1,2,3], keepdim=True) / (mask_gt.sum(dim=[1,2,3], keepdim=True) + 1e-8)
            shape_weight = shape_weight - shape_mean_val  # 广播减法，mask内均值恒为0
            
            # 🌟 全局偏移：动态计算相对推拉比例 (如 ±7%)
            delta_z_rel = torch.tanh(delta_z_scalar.view(-1, 1, 1, 1)) * MAX_Z_RATIO
            z_global = Z_base * (1.0 + delta_z_rel)
            
            # 最终深度图：全局尺度由Z_base+delta_z决定，局部形状由shape决定（shape均值恒为0）
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
            # 🌟 相对L1+SSIM：对全局缩放有弱惩罚，防止网络把全局偏移藏到shape里
            loss_depth = compute_l1_ssim(D_pred, depth_gt, mask_gt)

            # =========================================================
            # 🌟 核心修复：新增阶段 0 的 delta_z 显式监督
            # 强制全局偏移头承担基准微调职责，杜绝 shape_weight "篡位"做全局平移
            # 用GT中位数计算delta_z_gt，和Z_base的定义（基于中位数）一致
            # =========================================================
            batch_size = depth_gt.shape[0]
            depth_gt_median = torch.zeros(batch_size, 1, 1, 1, device=device)
            for b in range(batch_size):
                valid = depth_gt[b, 0][mask_gt[b, 0] > 0.5]
                if valid.numel() > 0:
                    depth_gt_median[b, 0, 0, 0] = torch.median(valid)
            delta_z_gt = (depth_gt_median - Z_base) / Z_base
            # 用L1而非MSE：梯度更稳定，对大偏差样本修正更有效
            loss_z = F.l1_loss(delta_z_rel, delta_z_gt)
            
            # 🌟 累加delta_z验证指标（用于TensorBoard曲线判断是否学会）
            with torch.no_grad():
                epoch_delta_z += delta_z_rel[0].item() * 100  # 转百分比
                epoch_delta_z_gt += delta_z_gt[0].item() * 100
                epoch_delta_z_error += abs(delta_z_rel[0].item() - delta_z_gt[0].item()) * 100

            # 🌟 新增：shape均值正则，强制shape只表达局部起伏
            loss_shape_reg = compute_shape_mean_reg(shape_weight, mask_gt)
            
            # 🌟 新增：shape平滑性损失，提升表面光滑度
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
            
            # 计算核心监控指标（🌟 改为整个batch平均，不再只取第0个样本，曲线更可信）
            with torch.no_grad():
                # delta_z：整个batch平均
                batch_delta_z = delta_z_rel.mean().item() * 100
                batch_delta_z_gt = delta_z_gt.mean().item() * 100
                batch_delta_z_error = (delta_z_rel - delta_z_gt).abs().mean().item() * 100
                
                # 形状与精度指标：整个batch平均（遍历每个样本）
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
                
                # 累加到epoch级
                epoch_delta_z += batch_delta_z
                epoch_delta_z_gt += batch_delta_z_gt
                epoch_delta_z_error += batch_delta_z_error
                epoch_shape_std += batch_shape_std
                epoch_shape_max += batch_shape_max
                epoch_shape_mean += batch_shape_mean
                epoch_mae += batch_mae
                epoch_iou += batch_iou
                
                # 进度条显示用第0个样本（保持简洁）
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
        writer.add_scalar("Loss/ShapeSmooth", epoch_loss_shape_smooth / num_batches, epoch)

        # 🌟 新增：核心业务指标曲线
        writer.add_scalar("Metrics/delta_z_rel(%)", epoch_delta_z / num_batches, epoch)
        writer.add_scalar("Metrics/delta_z_gt(%)", epoch_delta_z_gt / num_batches, epoch)
        writer.add_scalar("Metrics/delta_z_error(%)", epoch_delta_z_error / num_batches, epoch)  # 🌟 最关键：持续下降说明学会了
        writer.add_scalar("Metrics/shape_std(mm)", epoch_shape_std / num_batches, epoch)
        writer.add_scalar("Metrics/shape_max_abs(mm)", epoch_shape_max / num_batches, epoch)
        writer.add_scalar("Metrics/shape_mean(mm)", epoch_shape_mean / num_batches, epoch)
        writer.add_scalar("Metrics/depth_MAE(mm)", epoch_mae / num_batches, epoch)
        writer.add_scalar("Metrics/mask_IoU", epoch_iou / num_batches, epoch)
        
        if (epoch + 1) % 5 == 0 or epoch == 49:
            save_path = os.path.join(model_dir, f"student_stage0_ep{epoch}.pth")
            # 🌟 保存完整checkpoint：模型+优化器+调度器+epoch，支持完美断点续训
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, save_path)
            print(f"💾 保存完整checkpoint: {save_path}")
        
        # 🌟 每个epoch结束后更新学习率
        scheduler.step()
    
    writer.close()

if __name__ == "__main__":
    train_stage0()
