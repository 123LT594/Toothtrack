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

from dataset_distill import DualDistillDataset
from learning.models.student_depth_net import StudentDepthNet
from learning.models.refine_network import RefineNet

# 💥 修复隐患1: 引入 DISTILL_PHYSICAL_WIDTH 用于 XYZ 归一化
from learning.training.training_config import DISTILL_PHYSICAL_THICKNESS, DISTILL_PHYSICAL_WIDTH

# 💥 修复Bug2: 换回 Stage0 极其成功的局部滑窗 SSIM 与实例归一化
def compute_l1_ssim(pred, target, mask, window_size=11):
    max_depth = (target * mask).max().detach() + 1e-8
    p_norm = pred / max_depth
    t_norm = target / max_depth
    
    l1_loss = F.l1_loss(p_norm * mask, t_norm * mask, reduction='sum') / (mask.sum() + 1e-8)
    
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

    dataset = DualDistillDataset(is_training=True)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    
    student = StudentDepthNet().to(device)
    
    # 💥 修复Bug1: 强制加载阶段0的神装先验权重！不加载必崩！
    stage0_ckpt = "/root/lanyun-tmp/models/stage0_pretrain/models/student_stage0_ep49.pth"
    if os.path.exists(stage0_ckpt):
        student.load_state_dict(torch.load(stage0_ckpt, map_location=device))
        print(f"✅ 成功注入 Stage 0 物理先验权重: {stage0_ckpt}")
    else:
        print(f"⚠️ 警告: 未找到 Stage 0 权重 ({stage0_ckpt})，网络将盲人摸象！")
    
    teacher = RefineNet(c_in=6).to(device) 
    teacher_ckpt = "/root/Toothtrack/weights/2023-10-28-18-33-37/model_best.pth" 
    
    if os.path.exists(teacher_ckpt):
        # 如果模型键值对不完全匹配（比如去掉了末尾的某些头），可以使用 strict=False
        teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device), strict=False)
        print(f"🎓 成功注入 FoundationPose 教师网络灵魂: {teacher_ckpt}")
    else:
        raise FileNotFoundError(f"🚨 致命错误: 找不到教师网络权重 {teacher_ckpt}！老师脑子空空，蒸馏毫无意义！请立刻检查路径。")
    
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
        
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Stage1 Epoch [{epoch}/100]")
        for batch_idx, batch in pbar:
            inputs_6c = batch["inputs_6c"].to(device).float()     
            rgb_crop = batch["rgb_crop"].to(device).float()       
            unnorm_rays = batch["unnorm_rays"].to(device).float() 
            depth_gt = batch["depth_gt"].to(device).float()
            mask_gt = batch["mask_gt"].to(device).float()
            Z_base = batch["Z_base"].to(device).view(-1, 1, 1, 1).float()
            
            shape_weight_raw, mask_pred, delta_z_scalar = student(inputs_6c)
            
            MAX_Z_CORRECTION = 0.03
            THICKNESS_FACTOR = 0.0075
            
            shape_weight = torch.tanh(shape_weight_raw)
            delta_z = torch.tanh(delta_z_scalar.view(-1, 1, 1, 1)) * MAX_Z_CORRECTION
            
            D_pred = Z_base + delta_z + shape_weight * THICKNESS_FACTOR
            
            # 💥 修复隐患1: XYZ 绝对空间域归一化，对齐 Teacher 视角！
            XYZ_scale = DISTILL_PHYSICAL_WIDTH
            XYZ_pred_norm = (D_pred * unnorm_rays) / XYZ_scale
            XYZ_gt_norm = (depth_gt * unnorm_rays) / XYZ_scale
            
            A_pred = torch.cat([rgb_crop, XYZ_pred_norm], dim=1)
            A_gt = torch.cat([rgb_crop, XYZ_gt_norm], dim=1)
            
            F_s = teacher.extract_distill_feature(A_pred)
            with torch.no_grad():
                F_t = teacher.extract_distill_feature(A_gt)
                
            loss_depth = compute_l1_ssim(D_pred, depth_gt, mask_gt)
            loss_mask = compute_bce_dice(mask_pred, mask_gt)
            
            mask_feat = F.interpolate(mask_gt, size=F_t.shape[2:], mode='nearest')
            loss_feat = F.mse_loss(F_s * mask_feat, F_t * mask_feat, reduction='sum') / (mask_feat.sum() * F_s.shape[1] + 1e-8)
            loss_total = 10.0 * loss_depth + 1.0 * loss_mask + 1.0 * loss_feat   # 特征权重 1.0
            
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            
            epoch_loss += loss_total.item()
            epoch_loss_depth += loss_depth.item()
            epoch_loss_mask += loss_mask.item()
            epoch_loss_feat += loss_feat.item()

            pbar.set_postfix({'Tot': f"{loss_total.item():.3f}", 'F': f"{loss_feat.item():.3f}", 'D': f"{loss_depth.item():.3f}"})

            if batch_idx < 10:
                with torch.no_grad():
                    writer.add_image(f"Stage1_Vis/1_RGB_Dirty_{batch_idx}", rgb_crop[0], epoch)
                    writer.add_image(f"Stage1_Vis/2_Mask_Pred_{batch_idx}", mask_pred[0], epoch)
                    writer.add_image(f"Stage1_Vis/3_Mask_GT_{batch_idx}", mask_gt[0], epoch)
                    
                    vis_rgb = (rgb_crop[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
                    vis_mask_pred = (mask_pred[0, 0].cpu().numpy() * 255).astype(np.uint8)
                    vis_mask_gt = (mask_gt[0, 0].cpu().numpy() * 255).astype(np.uint8)
                    vis_mask_pred_3c = cv2.cvtColor(vis_mask_pred, cv2.COLOR_GRAY2BGR)
                    vis_mask_gt_3c = cv2.cvtColor(vis_mask_gt, cv2.COLOR_GRAY2BGR)
                    
                    # 💥 新增：引入阶段 0 的深度着色函数
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
                    
                    # 使用 GT 深度范围作为统一标尺
                    gt_np = depth_gt[0, 0].cpu().numpy()
                    m_gt = mask_gt[0, 0].cpu().numpy() > 0.5
                    if m_gt.sum() > 0:
                        vmin_shared = np.percentile(gt_np[m_gt], 2)
                        vmax_shared = np.percentile(gt_np[m_gt], 98)
                    else:
                        vmin_shared, vmax_shared = None, None
                    
                    # 生成彩色深度图
                    vis_depth_pred = colorize_depth(D_pred, mask_gt, vmin_shared, vmax_shared)
                    vis_depth_gt = colorize_depth(depth_gt, mask_gt, vmin_shared, vmax_shared)
                    
                    # 💥 将原来的 3 张图水平拼接，改为 5 张图拼接！
                    concat_img = np.hstack([vis_bgr, vis_mask_pred_3c, vis_mask_gt_3c, vis_depth_pred, vis_depth_gt])
                    
                    # 打上文字标签
                    for i, (x, label) in enumerate([(10, 'RGB'), (170, 'Pred Mask'), (330, 'GT Mask'), (490, 'Pred Depth'), (650, 'GT Depth')]):
                        cv2.putText(concat_img, label, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    cv2.imwrite(os.path.join(vis_dir, f"epoch_{epoch:03d}_{batch_idx:02d}.png"), concat_img)
            
        num_batches = len(dataloader)
        writer.add_scalar("Loss/Total", epoch_loss / num_batches, epoch)
        writer.add_scalar("Loss/Depth", epoch_loss_depth / num_batches, epoch)
        writer.add_scalar("Loss/Mask", epoch_loss_mask / num_batches, epoch)
        writer.add_scalar("Loss/Distill_Feat", epoch_loss_feat / num_batches, epoch)
        
        save_path = os.path.join(model_dir, f"student_stage1_ep{epoch}.pth")
        torch.save(student.state_dict(), save_path)
        
    writer.close()

if __name__ == "__main__":
    train_stage1()