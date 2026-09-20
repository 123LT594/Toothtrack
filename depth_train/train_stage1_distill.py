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

def compute_shape_smooth_loss(shape_weight, mask):
    """
    🌟 shape平滑性损失（Total Variation）：惩罚相邻像素的剧烈变化
    提升牙齿表面的光滑度，减少噪声状的形状输出
    """
    diff_h = torch.abs(shape_weight[:, :, :, 1:] - shape_weight[:, :, :, :-1])
    mask_h = mask[:, :, :, 1:] * mask[:, :, :, :-1]
    diff_v = torch.abs(shape_weight[:, :, 1:, :] - shape_weight[:, :, :-1, :])
    mask_v = mask[:, :, 1:, :] * mask[:, :, :-1, :]
    smooth_h = (diff_h * mask_h).sum() / (mask_h.sum() + 1e-8)
    smooth_v = (diff_v * mask_v).sum() / (mask_v.sum() + 1e-8)
    return smooth_h + smooth_v

def channel_normalize(feat, mask):
    """
    🌟 特征通道归一化：每个通道在有效区域内减均值除标准差
    防止大值通道主导特征蒸馏损失，使各通道贡献更均衡
    feat: [B, C, H, W], mask: [B, 1, H, W]
    """
    mask_expanded = mask.expand_as(feat)
    valid_count = mask_expanded.sum(dim=[0, 2, 3], keepdim=True) + 1e-8
    mean = (feat * mask_expanded).sum(dim=[0, 2, 3], keepdim=True) / valid_count
    var = ((feat - mean) ** 2 * mask_expanded).sum(dim=[0, 2, 3], keepdim=True) / valid_count
    std = torch.sqrt(var + 1e-8)
    return (feat - mean) / std

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
    dataset_old = DualDistillDataset(data_dir=os.path.abspath(os.path.join(_current_dir, "../../lanyun-tmp/golden_dataset")), is_training=True, dataset_id=0)
    dataset_new = DualDistillDataset(data_dir=os.path.abspath(os.path.join(_current_dir, "../../lanyun-tmp/golden_dataset_wxb")), is_training=True, dataset_id=1)
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
    
    # 🌟 用阶段0微调后的权重初始化（MAX_Z_RATIO=0.15，大偏差样本也能修正）
    # 微调后效果：delta_z_error=0.6%, MAE=0.33mm，远好于预训练的2.6%/1.28mm
    stage0_ckpt = "/root/lanyun-tmp/models/stage0_finetune/models/student_stage0_finetune_ep14.pth"
    if os.path.exists(stage0_ckpt):
        ckpt = torch.load(stage0_ckpt, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            student.load_state_dict(ckpt['model_state_dict'])
        else:
            student.load_state_dict(ckpt)
        print(f"✅ 成功注入 Stage 0 微调权重: {stage0_ckpt}")
    else:
        # 兜底：如果微调权重不存在，用预训练权重
        stage0_pretrain_ckpt = "/root/lanyun-tmp/models/stage0_pretrain/models/student_stage0_ep49.pth"
        if os.path.exists(stage0_pretrain_ckpt):
            student.load_state_dict(torch.load(stage0_pretrain_ckpt, map_location=device))
            print(f"⚠️ 未找到微调权重，使用预训练权重: {stage0_pretrain_ckpt}")
        else:
            print(f"⚠️ 警告: 未找到 Stage 0 权重，网络将盲人摸象！")
        
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
    # 🌟 余弦退火学习率调度：从5e-5平滑降到1e-6，减少后期震荡（和阶段0一致）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"📁 蒸馏日志定向至: {log_dir}")
    
    # 🌟 自动断点续训（兼容完整checkpoint和纯模型权重）
    start_epoch = 0
    if os.path.exists(model_dir):
        checkpoints = [f for f in os.listdir(model_dir) if f.startswith("student_stage1_ep") and f.endswith(".pth") and "ema" not in f]
        if checkpoints:
            epochs = [int(f.split('ep')[-1].split('.pth')[0]) for f in checkpoints]
            latest_epoch = max(epochs)
            latest_ckpt = os.path.join(model_dir, f"student_stage1_ep{latest_epoch}.pth")
            print(f"🔄 加载权重: {latest_ckpt}")
            ckpt = torch.load(latest_ckpt, map_location=device)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                # 完整checkpoint：同时加载模型、优化器、调度器
                student.load_state_dict(ckpt['model_state_dict'])
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                if 'scheduler_state_dict' in ckpt:
                    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                start_epoch = ckpt.get('epoch', latest_epoch) + 1
                print(f"✅ 加载完整checkpoint（含优化器+调度器状态）")
            else:
                # 旧格式：只有模型权重
                student.load_state_dict(ckpt)
                start_epoch = latest_epoch + 1
                print(f"⚠️ 旧格式checkpoint（仅模型权重），优化器重新初始化")
            print(f"🚀 从 Epoch {start_epoch} 继续训练")
    
    for epoch in range(start_epoch, 100):
        student.train()
        
        epoch_loss = 0.0
        epoch_loss_depth = 0.0
        epoch_loss_mask = 0.0
        epoch_loss_feat = 0.0
        epoch_loss_shape_reg = 0.0
        epoch_loss_shape_smooth = 0.0  # 🌟 shape平滑损失
        # 🌟 新增：核心指标epoch累加器（和阶段0命名完全一致）
        epoch_delta_z = 0.0
        epoch_delta_z_gt = 0.0       # 🌟 真实delta_z（用于验证网络是否学会）
        epoch_delta_z_error = 0.0    # 🌟 |pred - gt|，最关键的验证指标
        epoch_shape_std = 0.0
        epoch_shape_max = 0.0
        epoch_shape_mean = 0.0
        epoch_mae = 0.0
        epoch_iou = 0.0
        # 🌟 新增：双数据集分别监控的累加器
        # ds0 = 旧数据集(2800长焦, 10mm), ds1 = 新数据集(500短焦, <5mm)
        epoch_ds0_count = 0
        epoch_ds1_count = 0
        epoch_ds0_loss_depth = 0.0
        epoch_ds1_loss_depth = 0.0
        epoch_ds0_delta_z = 0.0
        epoch_ds1_delta_z = 0.0
        epoch_ds0_delta_z_error = 0.0  # 🌟 新增：双数据集delta_z_error
        epoch_ds1_delta_z_error = 0.0
        epoch_ds0_shape_mean = 0.0
        epoch_ds1_shape_mean = 0.0
        epoch_ds0_mae = 0.0
        epoch_ds1_mae = 0.0
        epoch_ds0_iou = 0.0
        epoch_ds1_iou = 0.0
        
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Stage1 Epoch [{epoch}/100]")
        for batch_idx, batch in pbar:
            inputs_6c = batch["inputs_6c"].to(device).float()     
            rgb_crop = batch["rgb_crop"].to(device).float()       
            unnorm_rays = batch["unnorm_rays"].to(device).float() 
            depth_gt = batch["depth_gt"].to(device).float()
            mask_gt = batch["mask_gt"].to(device).float()
            Z_base = batch["Z_base"].to(device).view(-1, 1, 1, 1).float()
            dataset_id = batch["dataset_id"].to(device)  # [B], 0=旧, 1=新
            
            # 双数据集mask：用于分别计算loss和指标
            ds0_mask = (dataset_id == 0)  # 旧数据集(2800长焦)
            ds1_mask = (dataset_id == 1)  # 新数据集(500短焦)
            has_ds0 = ds0_mask.sum().item() > 0
            has_ds1 = ds1_mask.sum().item() > 0
            
            shape_weight_raw, mask_pred, delta_z_scalar = student(inputs_6c)
            
            dynamic_width = batch["mesh_width"].to(device).view(-1, 1, 1, 1)
            shape_weight = torch.tanh(shape_weight_raw) * dynamic_width * SHAPE_SCALE_RATIO
            
            # 🌟 硬约束：强制减掉shape在mask内的均值，和阶段0完全一致
            # 彻底杜绝网络把全局偏移藏到shape里，delta_z必须承担全局修正
            shape_mean_val = (shape_weight * mask_gt).sum(dim=[1,2,3], keepdim=True) / (mask_gt.sum(dim=[1,2,3], keepdim=True) + 1e-8)
            shape_weight = shape_weight - shape_mean_val
            
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
            loss_depth = compute_l1_ssim(D_pred, depth_gt, mask_gt)
            loss_mask = compute_bce_dice(mask_pred, mask_gt)
            
            mask_feat = F.interpolate(mask_gt, size=F_t.shape[2:], mode='nearest')
            # 🌟 特征通道归一化：防止大值通道主导蒸馏损失，各通道贡献更均衡
            F_s_norm = channel_normalize(F_s, mask_feat)
            F_t_norm = channel_normalize(F_t, mask_feat)
            loss_feat = F.mse_loss(F_s_norm * mask_feat, F_t_norm * mask_feat, reduction='sum') / (mask_feat.sum() * F_s.shape[1] + 1e-8)
            
            # 🌟 delta_z监督：用GT中位数计算（和Z_base定义一致），L1损失（梯度更稳定）
            batch_size = depth_gt.shape[0]
            depth_gt_median = torch.zeros(batch_size, 1, 1, 1, device=device)
            for b in range(batch_size):
                valid = depth_gt[b, 0][mask_gt[b, 0] > 0.5]
                if valid.numel() > 0:
                    depth_gt_median[b, 0, 0, 0] = torch.median(valid)
            delta_z_gt = (depth_gt_median - Z_base) / Z_base
            loss_z = F.l1_loss(delta_z_rel, delta_z_gt)
            
            # 🌟 新增：shape均值正则，强制shape只表达局部起伏
            loss_shape_reg = compute_shape_mean_reg(shape_weight, mask_gt)
            
            # 🌟 新增：shape平滑性损失，提升表面光滑度
            loss_shape_smooth = compute_shape_smooth_loss(shape_weight, mask_gt)
            
            # 🌟 形状-尺度解耦：w_z提高（L1损失，需要足够梯度），新增shape_smooth权重
            if epoch < 30:
                w_depth, w_mask, w_feat, w_z, w_shape_reg, w_shape_smooth = 10.0, 1.0, 1.0, 5.0, 0.3, 0.1
            elif epoch < 60:
                w_depth, w_mask, w_feat, w_z, w_shape_reg, w_shape_smooth = 5.0, 1.0, 5.0, 5.0, 0.3, 0.1
            else:
                w_depth, w_mask, w_feat, w_z, w_shape_reg, w_shape_smooth = 1.0, 0.5, 10.0, 5.0, 0.3, 0.1
                
            loss_total = (w_depth * loss_depth + w_mask * loss_mask + w_feat * loss_feat 
                          + w_z * loss_z + w_shape_reg * loss_shape_reg + w_shape_smooth * loss_shape_smooth)
            
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
            epoch_loss_shape_smooth += loss_shape_smooth.item()

            # 🌟 计算核心监控指标（改为整个batch平均，不再只取第0个样本，曲线更可信）
            with torch.no_grad():
                # delta_z：整个batch平均
                batch_delta_z = delta_z_rel.mean().item() * 100
                batch_delta_z_gt = delta_z_gt.mean().item() * 100
                batch_delta_z_error = (delta_z_rel - delta_z_gt).abs().mean().item() * 100
                
                # 形状与精度指标：整个batch平均
                batch_shape_std = 0.0
                batch_shape_max = 0.0
                batch_shape_mean = 0.0
                batch_mae = 0.0
                batch_iou = 0.0
                valid_count = 0
                for b in range(inputs_6c.shape[0]):
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
                
                # 🌟 新增：双数据集分别计算指标（遍历batch中每个样本）
                for b_idx in range(inputs_6c.shape[0]):
                    ds_id = dataset_id[b_idx].item()
                    valid_mask_b = mask_gt[b_idx, 0] > 0.5
                    if valid_mask_b.sum() > 0:
                        dz_b = delta_z_rel[b_idx].item() * 100
                        dz_gt_b = delta_z_gt[b_idx].item() * 100
                        dz_err_b = abs(dz_b - dz_gt_b)
                        shape_vals_b = shape_weight[b_idx, 0][valid_mask_b]
                        sm_b = shape_vals_b.mean().item() * 1000
                        mae_b = torch.abs(D_pred[b_idx,0] - depth_gt[b_idx,0])[valid_mask_b].mean().item() * 1000
                        pred_mask_b = mask_pred[b_idx,0] > 0.5
                        inter_b = torch.logical_and(pred_mask_b, valid_mask_b).sum().item()
                        union_b = torch.logical_or(pred_mask_b, valid_mask_b).sum().item()
                        iou_b = inter_b / (union_b + 1e-8)
                    else:
                        dz_b = 0.0
                        dz_err_b = 0.0
                        sm_b = 0.0
                        mae_b = 0.0
                        iou_b = 0.0
                    
                    if ds_id == 0:
                        epoch_ds0_count += 1
                        epoch_ds0_delta_z += dz_b
                        epoch_ds0_delta_z_error += dz_err_b
                        epoch_ds0_shape_mean += sm_b
                        epoch_ds0_mae += mae_b
                        epoch_ds0_iou += iou_b
                    else:
                        epoch_ds1_count += 1
                        epoch_ds1_delta_z += dz_b
                        epoch_ds1_delta_z_error += dz_err_b
                        epoch_ds1_shape_mean += sm_b
                        epoch_ds1_mae += mae_b
                        epoch_ds1_iou += iou_b
            
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
        writer.add_scalar("Loss/ShapeSmooth", epoch_loss_shape_smooth / num_batches, epoch)

        # 🌟 新增：核心业务指标曲线（和阶段0完全对齐，方便对比）
        writer.add_scalar("Metrics/delta_z_rel(%)", epoch_delta_z / num_batches, epoch)
        writer.add_scalar("Metrics/delta_z_gt(%)", epoch_delta_z_gt / num_batches, epoch)
        writer.add_scalar("Metrics/delta_z_error(%)", epoch_delta_z_error / num_batches, epoch)  # 🌟 最关键：持续下降说明学会了
        writer.add_scalar("Metrics/shape_std(mm)", epoch_shape_std / num_batches, epoch)
        writer.add_scalar("Metrics/shape_max_abs(mm)", epoch_shape_max / num_batches, epoch)
        writer.add_scalar("Metrics/shape_mean(mm)", epoch_shape_mean / num_batches, epoch)
        writer.add_scalar("Metrics/depth_MAE(mm)", epoch_mae / num_batches, epoch)
        writer.add_scalar("Metrics/mask_IoU", epoch_iou / num_batches, epoch)
        
        # 🌟 新增：双数据集分别监控（及时发现适配问题）
        # 用"/"分组，让TensorBoard像Metrics一样把同一数据集的曲线放在一行显示
        # ds0 = 旧数据集(2800长焦, 10mm工作距离), ds1 = 新数据集(500短焦, <5mm工作距离)
        if epoch_ds0_count > 0:
            writer.add_scalar("DS0_Old/delta_z(%)", epoch_ds0_delta_z / epoch_ds0_count, epoch)
            writer.add_scalar("DS0_Old/delta_z_error(%)", epoch_ds0_delta_z_error / epoch_ds0_count, epoch)
            writer.add_scalar("DS0_Old/shape_mean(mm)", epoch_ds0_shape_mean / epoch_ds0_count, epoch)
            writer.add_scalar("DS0_Old/depth_MAE(mm)", epoch_ds0_mae / epoch_ds0_count, epoch)
            writer.add_scalar("DS0_Old/mask_IoU", epoch_ds0_iou / epoch_ds0_count, epoch)
        if epoch_ds1_count > 0:
            writer.add_scalar("DS1_New/delta_z(%)", epoch_ds1_delta_z / epoch_ds1_count, epoch)
            writer.add_scalar("DS1_New/delta_z_error(%)", epoch_ds1_delta_z_error / epoch_ds1_count, epoch)
            writer.add_scalar("DS1_New/shape_mean(mm)", epoch_ds1_shape_mean / epoch_ds1_count, epoch)
            writer.add_scalar("DS1_New/depth_MAE(mm)", epoch_ds1_mae / epoch_ds1_count, epoch)
            writer.add_scalar("DS1_New/mask_IoU", epoch_ds1_iou / epoch_ds1_count, epoch)
        
        # 🌟 学习率曲线
        writer.add_scalar("LR/current", optimizer.param_groups[0]['lr'], epoch)
        
        # 🌟 每5轮保存一次完整checkpoint（含优化器+调度器，支持断点续训）
        if (epoch + 1) % 5 == 0 or epoch == 99:
            save_path = os.path.join(model_dir, f"student_stage1_ep{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, save_path)
            ema_save_path = os.path.join(model_dir, f"student_stage1_ema_ep{epoch}.pth")
            torch.save(student_ema.state_dict(), ema_save_path)
            print(f"💾 保存完整checkpoint: {save_path}")
        
        # 🌟 每个epoch结束后更新学习率
        scheduler.step()
        
    writer.close()

if __name__ == "__main__":
    train_stage1()
