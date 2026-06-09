import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from tqdm import tqdm
import sys

# ==============================================================================
# 🌟 核心控制台 🌟
# ==============================================================================
# 1. 调试模式开关：设为 True 时，只遍历数据计算虚拟深度常数；算完后设为 False 进行训练。
FIND_CONSTANT_MODE = False

# 2. 你的专属常数配置区 (等 FIND_CONSTANT_MODE 跑完后，把算出的值填到这里)
MAX_DEPTH = 0.194  # 建议填入：打印出来的平均虚拟深度的 2 倍，确保完美落在 Sigmoid(0.5)

# 数据集路径
DATASET_ROOT = '/root/lanyun-tmp/golden_dataset'
# ==============================================================================


# ================= 1. 动态权重模块 =================
class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num_tasks=2):
        super(AutomaticWeightedLoss, self).__init__()
        self.params = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 * torch.exp(-self.params[i]) * loss + 0.5 * self.params[i]
        return loss_sum

# ================= 2. 解耦网络架构 (保持不变) =================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class DecoupledMultiTaskUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = DoubleConv(3, 32); self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128); self.enc4 = DoubleConv(128, 256)
        self.bot = DoubleConv(256, 512)
        self.m_dec4 = DoubleConv(512 + 256, 256); self.m_dec3 = DoubleConv(256 + 128, 128)
        self.m_dec2 = DoubleConv(128 + 64, 64); self.m_dec1 = DoubleConv(64 + 32, 32)
        self.mask_head = nn.Conv2d(32, 1, 1)
        self.d_dec4 = DoubleConv(512 + 256, 256); self.d_dec3 = DoubleConv(256 + 128, 128)
        self.d_dec2 = DoubleConv(128 + 64, 64); self.d_dec1 = DoubleConv(64 + 32, 32)
        self.depth_head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2)); e4 = self.enc4(self.pool(e3))
        b = self.bot(self.pool(e4))
        
        m_u4 = F.interpolate(b, size=e4.shape[2:], mode='bilinear', align_corners=True)
        m_d4 = self.m_dec4(torch.cat([m_u4, e4], dim=1))
        m_u3 = F.interpolate(m_d4, size=e3.shape[2:], mode='bilinear', align_corners=True)
        m_d3 = self.m_dec3(torch.cat([m_u3, e3], dim=1))
        m_u2 = F.interpolate(m_d3, size=e2.shape[2:], mode='bilinear', align_corners=True)
        m_d2 = self.m_dec2(torch.cat([m_u2, e2], dim=1))
        m_u1 = F.interpolate(m_d2, size=e1.shape[2:], mode='bilinear', align_corners=True)
        m_d1 = self.m_dec1(torch.cat([m_u1, e1], dim=1))
        
        d_u4 = F.interpolate(b, size=e4.shape[2:], mode='bilinear', align_corners=True)
        d_d4 = self.d_dec4(torch.cat([d_u4, e4], dim=1))
        d_u3 = F.interpolate(d_d4, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d_d3 = self.d_dec3(torch.cat([d_u3, e3], dim=1))
        d_u2 = F.interpolate(d_d3, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d_d2 = self.d_dec2(torch.cat([d_u2, e2], dim=1))
        d_u1 = F.interpolate(d_d2, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d_d1 = self.d_dec1(torch.cat([d_u1, e1], dim=1))
        
        return self.mask_head(m_d1), torch.sigmoid(self.depth_head(d_d1)) * MAX_DEPTH

# ================= 3. 无损的梯度损失计算 =================
class SafeGradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]).view(1,1,3,3).cuda() / 4.0
        ky = torch.tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]]).view(1,1,3,3).cuda() / 4.0
        self.register_buffer('kx', kx); self.register_buffer('ky', ky)
        
    def forward(self, pred, target, mask):
        px = F.conv2d(pred, self.kx, padding=1)
        py = F.conv2d(pred, self.ky, padding=1)
        tx = F.conv2d(target, self.kx, padding=1)
        ty = F.conv2d(target, self.ky, padding=1)
        
        eroded_mask = F.max_pool2d(1 - mask, kernel_size=3, stride=1, padding=1)
        safe_mask = 1 - eroded_mask
        
        valid_pixels = safe_mask.sum() + 1e-5
        loss_x = F.l1_loss(px * safe_mask, tx * safe_mask, reduction='sum') / valid_pixels
        loss_y = F.l1_loss(py * safe_mask, ty * safe_mask, reduction='sum') / valid_pixels
        return loss_x + loss_y

# ================= 4. 数据集 (完美解耦) =================
class CropJointDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_size=512, crop_margin=1.5, is_debug=False):
        self.rgb_dir = os.path.join(root_dir, 'rgb')
        self.depth_dir = os.path.join(root_dir, 'depth')
        
        # 1. 获取所有文件名
        all_files = sorted(os.listdir(self.rgb_dir))
        
        # 2. 🌟 强制物理锁定：只保留编号 <= 765 的文件
        # 假设文件名格式为 frame_XXXX.png 或 XXXX.png
        valid_files = []
        for f in all_files:
            # 提取文件名中的数字
            try:
                # 去掉扩展名，提取数字部分（例如 'frame_0001' -> '0001'）
                num_str = ''.join(filter(str.isdigit, os.path.splitext(f)[0]))
                if num_str and int(num_str) <= 765:
                    valid_files.append(f)
            except:
                continue
        
        self.files = valid_files
        self.transform = transform
        self.target_size = target_size
        self.crop_margin = crop_margin
        self.is_debug = is_debug
        
        print(f"📊 [Dataset Check] 编号765以前的文件共: {len(self.files)} 个")
        
    def __len__(self): return len(self.files)
    
    def __getitem__(self, idx):
        img_name = self.files[idx]; base_name = os.path.splitext(img_name)[0]
        
        image = cv2.imread(os.path.join(self.rgb_dir, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = np.load(os.path.join(self.depth_dir, base_name + '.npy')).astype(np.float32)
        depth = np.nan_to_num(depth, nan=0.0)
        mask = (depth > 0.0001).astype(np.float32)
        
        ys, xs = np.where(mask > 0)
        if len(ys) < 50:
            if self.is_debug: return 0.0, 0.0
            return torch.zeros((3, self.target_size, self.target_size)), \
                   torch.zeros((1, self.target_size, self.target_size)), \
                   torch.zeros((1, self.target_size, self.target_size))
                   
        real_cx = (xs.max() + xs.min()) / 2.0
        real_cy = (ys.max() + ys.min()) / 2.0
        real_span = max(xs.max() - xs.min(), ys.max() - ys.min())
        
        L = self.target_size
        # 计算放大系数
        s_factor = L / (real_span * self.crop_margin)
        
        # ================= 🌟 调试模式：只计算虚拟深度常数 =================
        if self.is_debug:
            # 获取牙齿表面深度的近似最大值（或者平均值）
            valid_depths = depth[mask > 0]
            # 为了过滤飞点噪点，取 95% 分位数作为该帧的真实代表深度
            real_depth_repr = np.percentile(valid_depths, 95)
            # 核心物理公式：虚拟深度 = 真实深度 / 放大系数
            virtual_depth = real_depth_repr / s_factor
            return virtual_depth, real_depth_repr
        # ===================================================================

        # 🌟 正常训练模式：数据增强坚决去掉缩放 Scale！保留平移和旋转！
        theta = np.random.uniform(-180, 180)
        cx = real_cx + np.random.uniform(-0.05, 0.05) * real_span
        cy = real_cy + np.random.uniform(-0.05, 0.05) * real_span
        
        M = cv2.getRotationMatrix2D((cx, cy), theta, s_factor)
        M[0, 2] += (L / 2) - cx
        M[1, 2] += (L / 2) - cy
        
        rgb_crop = cv2.warpAffine(image, M, (L, L), flags=cv2.INTER_LINEAR)
        mask_crop = cv2.warpAffine(mask, M, (L, L), flags=cv2.INTER_NEAREST)
        depth_crop = cv2.warpAffine(depth, M, (L, L), flags=cv2.INTER_LINEAR)
        
        # 🌟 物理深度归一化 (核心解耦：让所有输入的牙齿深度变成一个常数模板！)
        depth_crop = depth_crop * (1.0 / s_factor) * mask_crop
        
        if self.transform:
            augmented = self.transform(image=rgb_crop)
            rgb_crop = augmented['image'] 
            
        mask_tensor = torch.from_numpy(mask_crop).unsqueeze(0).float()
        depth_tensor = torch.from_numpy(depth_crop).unsqueeze(0).float()
        
        return rgb_crop, depth_tensor, mask_tensor

# ================= 5. 主程序入口 =================
if __name__ == "__main__":
    if FIND_CONSTANT_MODE:
        print("🔍 启动【物理常数扫描模式】...")
        print(f"📂 正在扫描数据集: {DATASET_ROOT} (仅限前 765 帧)")
        
        debug_ds = CropJointDataset(DATASET_ROOT, target_size=512, crop_margin=1.5, is_debug=True)
        virtual_depths = []
        real_depths = []
        
        for i in tqdm(range(len(debug_ds)), desc="扫描计算中"):
            v_depth, r_depth = debug_ds[i]
            if v_depth > 0:
                virtual_depths.append(v_depth)
                real_depths.append(r_depth)
                
        avg_v_depth = np.mean(virtual_depths)
        avg_r_depth = np.mean(real_depths)
        
        print("\n" + "="*50)
        print("🎯 扫描完成！为你算出的绝对物理参数如下：")
        print(f"📍 前765帧原图的平均真实深度 (Z_real) 约为: {avg_r_depth:.4f} 米")
        print(f"👑 网络将看到的【归一化虚拟深度】恒定在: {avg_v_depth:.4f} 米")
        print("-" * 50)
        print(f"🚀 【下一步操作指南】：")
        print(f"1. 请将本代码顶部的 MAX_DEPTH 设置为虚拟深度的 2 倍。")
        print(f"   => 建议设置: MAX_DEPTH = {avg_v_depth * 2.0:.3f}")
        print(f"2. 将 FIND_CONSTANT_MODE 修改为 False。")
        print(f"3. 重新运行本脚本，开启正式训练！")
        print("="*50)
        sys.exit(0)
        
    # ================= 下面是正常训练逻辑 =================
    print(f"🚀 启动正式训练模式！当前配置的 MAX_DEPTH = {MAX_DEPTH}")
    device = torch.device('cuda')
    MODEL_SAVE_DIR = '/root/lanyun-tmp/models/models_joint_box'  
    LOG_DIR = '/root/lanyun-tmp/logs/joint_exp_box_full'
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_DIR), exist_ok=True)
    
    writer = SummaryWriter(LOG_DIR)

    train_tf = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.ColorJitter(saturation=(0.6, 1.4), hue=0, p=0.5),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 7)), A.MotionBlur(blur_limit=5)], p=0.3),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
        A.Normalize(mean=(0,0,0), std=(1,1,1)), 
        ToTensorV2()
    ])

    train_ds = CropJointDataset(DATASET_ROOT, transform=train_tf, target_size=512, crop_margin=1.5, is_debug=False)
    print(f"🔥 数据加载完毕，共 {len(train_ds)} 帧")
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)

    model = DecoupledMultiTaskUNet().to(device)
    awl = AutomaticWeightedLoss(num_tasks=2).to(device)
    
    optimizer = optim.AdamW([
        {'params': model.parameters()},
        {'params': awl.parameters(), 'weight_decay': 0}
    ], lr=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    
    bce_crit = nn.BCEWithLogitsLoss()
    grad_crit = SafeGradientLoss().to(device)

    for epoch in range(1, 101):
        model.train()
        epoch_m_loss = 0; epoch_d_loss = 0; epoch_total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/100", unit="batch")
        
        for i, (imgs, depths, masks) in enumerate(train_loader):
            imgs, depths, masks = imgs.to(device), depths.to(device), masks.to(device)
            optimizer.zero_grad()
            
            m_logits, d_preds = model(imgs)
            
            loss_m = bce_crit(m_logits, masks)
            valid_pixels = masks.sum() + 1e-5
            
            # L1 Loss (使用还原掩码确保干净)
            l1_loss = F.l1_loss(d_preds * masks, depths * masks, reduction='sum') / valid_pixels
            # 安全梯度 Loss (内部已处理掩码)
            g_loss = grad_crit(d_preds, depths, masks) 
            
            loss_d = l1_loss + 1.5 * g_loss
            
            total_loss = awl(loss_m, loss_d)
            total_loss.backward()
            optimizer.step()
            
            epoch_m_loss += loss_m.item()
            epoch_d_loss += loss_d.item()
            epoch_total += total_loss.item()
            
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({'Total': f"{total_loss.item():.4f}"})

        scheduler.step()
        print(f"\n✅ Epoch {epoch} | Total: {epoch_total/len(train_loader):.6f} | D-Loss: {epoch_d_loss/len(train_loader):.6f}")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, f"joint_epoch_{epoch}.pth"))
        if epoch == 100:
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "joint_best.pth"))

    writer.close()