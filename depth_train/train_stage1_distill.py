import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset_distill import DualDistillDataset
from learning.models.student_depth_net import StudentDepthNet
from learning.models.refine_network import RefineNet
from learning.training.training_config import MAX_Z_RATIO, SHAPE_SCALE_RATIO

def compute_l1_ssim(pred, target, mask, window_size=11):
    # =========================================================
    # 🌟 核心改进：相对 L1 损失 (AbsRel) + 局部滑窗 SSIM
    # =========================================================
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

def train_stage1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_name = "stage1_distill"
    base_out_dir = f"/root/lanyun-tmp/models/{model_name}"
    model_dir = os.path.join(base_out_dir, "models")
    log_dir = os.path.join(base_out_dir, "logs")
    vis_dir = os.path.join(base_out_dir, "vis")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_old = DualDistillDataset(data_dir=os.path.abspath(os.path.join(_current_dir, "../../lanyun-tmp/golden_dataset")), is_training=True)
    dataset_new = DualDistillDataset(data_dir=os.path.abspath(os.path.join(_current_dir, "../../lanyun-tmp/golden_dataset_wxb")), is_training=True)
    mixed_dataset = ConcatDataset([dataset_old, dataset_new])
    len_old = len(dataset_old)
    len_new = len(dataset_new)
    sample_weights = [1.0] * len_old + [1.5] * len_new
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    dataloader = DataLoader(mixed_dataset, batch_size=16, sampler=sampler, num_workers=4)
    
    student = StudentDepthNet().to(device)
    
    stage0_ckpt = "/root/lanyun-tmp/models/stage0_pretrain/models/student_stage0_ep49.pth"
    if os.path.exists(stage0_ckpt):
        student.load_state_dict(torch.load(stage0_ckpt, map_location=device))
        print(f"✅ 成功注入 Stage 0 物理先验权重: {stage0_ckpt}")
    else:
        print(f"⚠️ 警告: 未找到 Stage 0 权重 ({stage0_ckpt})，网络将盲人摸象！")
        
    student_ema = copy.deepcopy(student)
    for param in student_ema.parameters():
        param.requires_grad = False
    print("🛡️ 已开启 EMA (指数移动平均) 权重保护，大幅增强小样本泛化性。")
    
    teacher = RefineNet(c_in=6).to(device)
    teacher_ckpt = "/root/Toothtrack/weights/2023-10-28-18-33-37/model_best.pth" 
    
    if os.path.exists(teacher_ckpt):
        teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device), strict=False)
        print(f"🎓 成功注入 FoundationPose 教师网络灵魂: {teacher_ckpt}")
    else:
        raise FileNotFoundError(f"🚨 致命错误: 找不到教师网络权重 {teacher_ckpt}！")
    
    teacher.eval() 
    for param in teacher.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=5e-5, weight_decay=1e-4)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"📁 蒸馏日志定向至: {log_dir}")
    
    for epoch in range(100):
        student.train()
        
        epoch_loss = 0.0
        epoch_loss_depth = 0.0
        epoch_loss_mask = 0.0
        epoch_loss_feat = 0.0
        epoch_loss_shape_reg = 0.0
        # 🌟 新增：核心指标epoch累加器（和阶段0命名完全一致）
        epoch_delta_z = 0.0
        epoch_shape_std = 0.0
        epoch_shape_max = 0.0
        epoch_shape_mean = 0.0
        epoch_mae = 0.0
        epoch_iou = 0.0
        
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Stage1 Epoch [{epoch}/100]")
        for batch_idx, batch in pbar:
            inputs_6c = batch["inputs_6c"].to(device).float()     
            rgb_crop = batch["rgb_crop"].to(device).float()       
            unnorm_rays = batch["unnorm_rays"].to(device).float() 
            depth_gt = batch["depth_gt"].to(device).float()
            mask_gt = batch["mask_gt"].to(device).float()
            Z_base = batch["Z_base"].to(device).view(-1, 1, 1, 1).float()
            
            shape_weight_raw, mask_pred, delta_z_scalar = student(inputs_6c)
            
            dynamic_width = batch["mesh_width"].to(device).view(-1, 1, 1, 1)
            shape_weight = torch.tanh(shape_weight_raw) * dynamic_width * SHAPE_SCALE_RATIO
            delta_z_rel = torch.tanh(delta_z_scalar.view(-1, 1, 1, 1)) * MAX_Z_RATIO
            z_global = Z_base * (1.0 + delta_z_rel)
            
            D_pred = z_global + shape_weight
            
            # XYZ 绝对空间域归一化
            XYZ_scale = dynamic_width
            XYZ_pred_norm = (D_pred * unnorm_rays) / XYZ_scale
            XYZ_gt_norm = (depth_gt * unnorm_rays) / XYZ_scale
            
            A_pred = torch.cat([rgb_crop, XYZ_pred_norm], dim=1)
            A_gt = torch.cat([rgb_crop, XYZ_gt_norm], dim=1)
            
            F_s = teacher.extract_distill_feature(A_pred)
            with torch.no_grad():
                F_t = teacher.extract_distill_feature(A_gt)
                
            # 🌟 形状-尺度解耦：使用仿射不变损失，只监督相对形状
            loss_depth = compute_affine_invariant_loss(D_pred, depth_gt, mask_gt)
            loss_mask = compute_bce_dice(mask_pred, mask_gt)
            
            mask_feat = F.interpolate(mask_gt, size=F_t.shape[2:], mode='nearest')
            loss_feat = F.mse_loss(F_s * mask_feat, F_t * mask_feat, reduction='sum') / (mask_feat.sum() * F_s.shape[1] + 1e-8)
            
            depth_gt_mean = (depth_gt * mask_gt).sum(dim=[1,2,3]) / (mask_gt.sum(dim=[1,2,3]) + 1e-8)
            delta_z_gt = (depth_gt_mean.view(-1, 1, 1, 1) - Z_base) / Z_base
            loss_z = F.mse_loss(delta_z_rel, delta_z_gt)
            
            # 🌟 新增：shape均值正则，强制shape只表达局部起伏
            loss_shape_reg = compute_shape_mean_reg(shape_weight, mask_gt)
            
            # 🌟 形状-尺度解耦：w_z 大幅降低（小范围微调），新增 shape_reg 权重
            if epoch < 30:
                w_depth, w_mask, w_feat, w_z, w_shape_reg = 10.0, 1.0, 1.0, 0.5, 0.3
            elif epoch < 60:
                w_depth, w_mask, w_feat, w_z, w_shape_reg = 5.0, 1.0, 5.0, 0.5, 0.3
            else:
                w_depth, w_mask, w_feat, w_z, w_shape_reg = 1.0, 0.5, 10.0, 0.5, 0.3
                
            loss_total = w_depth * loss_depth + w_mask * loss_mask + w_feat * loss_feat + w_z * loss_z + w_shape_reg * loss_shape_reg
            
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            
            with torch.no_grad():
                m = 0.999
                for param_q, param_k in zip(student.parameters(), student_ema.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            
            epoch_loss += loss_total.item()
            epoch_loss_depth += loss_depth.item()
            epoch_loss_mask += loss_mask.item()
            epoch_loss_feat += loss_feat.item()
            epoch_loss_shape_reg += loss_shape_reg.item()

            # 🌟 新增：计算核心监控指标（取batch第0个样本，和阶段0逻辑一致）
            with torch.no_grad():
                valid_mask = mask_gt[0, 0] > 0.5
                if valid_mask.sum() > 0:
                    delta_z_val = delta_z_rel[0].item() * 100
                    
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
                    delta_z_val = 0.0
                    shape_std = 0.0
                    shape_max_abs = 0.0
                    shape_mean_val = 0.0
                    mae = 0.0
                    iou = 0.0
                
                epoch_delta_z += delta_z_val
                epoch_shape_std += shape_std
                epoch_shape_max += shape_max_abs
                epoch_shape_mean += shape_mean_val
                epoch_mae += mae
                epoch_iou += iou
            
            pbar.set_postfix({'Tot': f"{loss_total.item():.3f}", 'F': f"{loss_feat.item():.3f}", 'D': f"{loss_depth.item():.3f}"})
            if batch_idx < 10:
                with torch.no_grad():
                    # =========================================================
                    # 🌟 修复可视化：构建 5 图拼接，并推送到 TensorBoard
                    # =========================================================
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
                    
                    for i, (x, label) in enumerate([(10, 'Crop RGB'), (170, 'Pred Mask'), (330, 'GT Mask'), (490, 'Pred Depth'), (650, 'GT Depth')]):
                        cv2.putText(concat_img, label, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # 1. 保存到本地目录
                    cv2.imwrite(os.path.join(vis_dir, f"epoch_{epoch:03d}_{batch_idx:02d}.png"), concat_img)
                    
                    # 2. 🌟 转换 BGR 到 RGB 并推送到 TensorBoard 大盘
                    writer.add_image(f"Stage1_Combined_Vis/Batch_{batch_idx}", cv2.cvtColor(concat_img, cv2.COLOR_BGR2RGB), epoch, dataformats='HWC')
            
        num_batches = len(dataloader)
        writer.add_scalar("Loss/Total", epoch_loss / num_batches, epoch)
        writer.add_scalar("Loss/Depth", epoch_loss_depth / num_batches, epoch)
        writer.add_scalar("Loss/Mask", epoch_loss_mask / num_batches, epoch)
        writer.add_scalar("Loss/Distill_Feat", epoch_loss_feat / num_batches, epoch)
        writer.add_scalar("Loss/ShapeReg", epoch_loss_shape_reg / num_batches, epoch)

        # 🌟 新增：核心业务指标曲线（和阶段0完全对齐，方便对比）
        writer.add_scalar("Metrics/delta_z_rel(%)", epoch_delta_z / num_batches, epoch)
        writer.add_scalar("Metrics/shape_std(mm)", epoch_shape_std / num_batches, epoch)
        writer.add_scalar("Metrics/shape_max_abs(mm)", epoch_shape_max / num_batches, epoch)
        writer.add_scalar("Metrics/shape_mean(mm)", epoch_shape_mean / num_batches, epoch)
        writer.add_scalar("Metrics/depth_MAE(mm)", epoch_mae / num_batches, epoch)
        writer.add_scalar("Metrics/mask_IoU", epoch_iou / num_batches, epoch)
        
        save_path = os.path.join(model_dir, f"student_stage1_ep{epoch}.pth")
        torch.save(student.state_dict(), save_path)
        ema_save_path = os.path.join(model_dir, f"student_stage1_ema_ep{epoch}.pth")
        torch.save(student_ema.state_dict(), ema_save_path)
        
    writer.close()

if __name__ == "__main__":
    train_stage1()
