import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # 强制走国内镜像
os.environ["HF_HUB_OFFLINE"] = "0"

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
import timm # 🌟 引入最强 Transformer 武器库

MAX_DEPTH = 0.2

# ================= 1. 动态权重模块 (保持原版精髓) =================
class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num_tasks=2):
        super(AutomaticWeightedLoss, self).__init__()
        self.params = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 * torch.exp(-self.params[i]) * loss + 0.5 * self.params[i]
        return loss_sum

# ================= 2. 边缘自适应平滑损失 (噪点杀手) =================
class EdgeAwareSmoothnessLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, depth, img):
        # 计算深度图的 X, Y 方向梯度
        depth_dx = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
        depth_dy = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])
        
        # 计算原图 RGB 的 X, Y 方向梯度 (求平均通道)
        img_dx = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]).mean(dim=1, keepdim=True)
        img_dy = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]).mean(dim=1, keepdim=True)
        
        # 核心公式：深度梯度 * exp(-边缘强度)
        weight_x = torch.exp(-img_dx * 10.0)
        weight_y = torch.exp(-img_dy * 10.0)
        
        loss_x = (depth_dx * weight_x).mean()
        loss_y = (depth_dy * weight_y).mean()
        
        return loss_x + loss_y

# ================= 3. Transformer 解耦多任务网络架构 (AACNet 魔改版) =================
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
        
        # Swin Transformer Encoder
        self.encoder = timm.create_model('convnext_tiny', pretrained=True, features_only=True)
        
        # --- 独立的 Mask Decoder ---
        self.m_dec4 = DecoderBlock(768, 384, 384)
        self.m_dec3 = DecoderBlock(384, 192, 192)
        self.m_dec2 = DecoderBlock(192, 96, 96)  
        self.m_dec1 = DecoderBlock(96, 0, 32)    
        self.m_dec0 = DecoderBlock(32, 0, 16)    
        self.mask_head = nn.Conv2d(16, 1, 1)

        # --- 独立的 Depth Decoder ---
        self.d_dec4 = DecoderBlock(768, 384, 384) 
        self.d_dec3 = DecoderBlock(384, 192, 192) 
        self.d_dec2 = DecoderBlock(192, 96, 96)   
        self.d_dec1 = DecoderBlock(96, 0, 32)     
        self.d_dec0 = DecoderBlock(32, 0, 16)     
        self.depth_head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        features = self.encoder(x)
        e1, e2, e3, e4 = features[0], features[1], features[2], features[3]
        
        # --- Mask 分支 (先计算) ---
        m4 = self.m_dec4(e4, e3)
        m3 = self.m_dec3(m4, e2)
        m2 = self.m_dec2(m3, e1)
        m1 = self.m_dec1(m2)
        m0 = self.m_dec0(m1)
        
        mask_logits = self.mask_head(m0)
        
        # 🌟 核心升级 1：SDMAA 解剖感知注意力 (Mask 引导 Depth)
        # 强迫特征顺着牙齿几何流形汇聚，屏蔽背景骨头噪声
        mask_attention = torch.sigmoid(mask_logits)
        
        # --- Depth 分支 ---
        d4 = self.d_dec4(e4, e3)
        d3 = self.d_dec3(d4, e2)
        d2 = self.d_dec2(d3, e1)
        d1 = self.d_dec1(d2)
        d0 = self.d_dec0(d1)
        
        # 运用解剖注意力权重过滤深度特征
        d0_guided = d0 * mask_attention
        
        return mask_logits, torch.sigmoid(self.depth_head(d0_guided)) * MAX_DEPTH

# ================= 4. 数据集与辅助类 =================
class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]).view(1,1,3,3).cuda() / 4.0
        ky = torch.tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]]).view(1,1,3,3).cuda() / 4.0
        self.register_buffer('kx', kx); self.register_buffer('ky', ky)
    def forward(self, pred, target):
        px = F.conv2d(pred, self.kx, padding=1); py = F.conv2d(pred, self.ky, padding=1)
        tx = F.conv2d(target, self.kx, padding=1); ty = F.conv2d(target, self.ky, padding=1)
        return F.l1_loss(px, tx) + F.l1_loss(py, ty)

class JointDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.rgb_dir = os.path.join(root_dir, 'rgb')
        self.depth_dir = os.path.join(root_dir, 'depth')
        self.files = sorted(os.listdir(self.rgb_dir))
        self.transform = transform
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        img_name = self.files[idx]; base_name = os.path.splitext(img_name)[0]
        image = cv2.imread(os.path.join(self.rgb_dir, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        depth = np.load(os.path.join(self.depth_dir, base_name + '.npy')).astype(np.float32)
        depth = np.nan_to_num(depth, nan=0.0)
        mask = (depth > 0.0001).astype(np.float32) # 严格的 0/1 Mask
        
        if self.transform:
            augmented = self.transform(image=image, masks=[depth, mask])
            image, (depth, mask) = augmented['image'], augmented['masks']
            
        return image, depth.unsqueeze(0), mask.unsqueeze(0)

# ================= 5. 训练主循环 =================
if __name__ == "__main__":
    device = torch.device('cuda')
    MODEL_SAVE_DIR = '/root/lanyun-tmp/models/models_joint_transformer_aacnet'
    LOG_DIR = '/root/lanyun-tmp/logs/joint_exp_transformer_aacnet'
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_DIR), exist_ok=True)
    
    writer = SummaryWriter(LOG_DIR)

    # 🌟 核心升级 2：绝对几何安全的数据增强
    # 彻底剔除 Shift, Scale, Rotate, Flip，确保与相机内参 100% 绑定
    train_tf = A.Compose([
        A.PadIfNeeded(min_height=544, min_width=960, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.RandomBrightnessContrast(p=0.5),
        A.ColorJitter(saturation=(0.6, 1.4), hue=0, p=0.5),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 7)), A.MotionBlur(blur_limit=5)], p=0.3),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
        ToTensorV2()
    ])

    train_ds = JointDataset('/root/lanyun-tmp/golden_dataset', train_tf)
    print(f"🔥 AACNet魔改版 Transformer 降维打击启动！全量数据：共 {len(train_ds)} 帧")
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=8, pin_memory=True) 

    model = SwinMultiTaskUNet().to(device)
    awl = AutomaticWeightedLoss(num_tasks=2).to(device)
    
    optimizer = optim.AdamW([
        {'params': model.parameters()},
        {'params': awl.parameters(), 'weight_decay': 0}
    ], lr=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    
    bce_crit = nn.BCEWithLogitsLoss(reduction='none') # 注意这里改为了 none，为了做门控
    l1_crit = nn.L1Loss()
    grad_crit = GradientLoss()
    smooth_crit = EdgeAwareSmoothnessLoss() 

    for epoch in range(1, 101):
        model.train()
        epoch_m_loss = 0; epoch_d_loss = 0; epoch_total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/100", unit="batch")
        
        for i, (imgs, depths, masks) in enumerate(train_loader):
            imgs, depths, masks = imgs.to(device), depths.to(device), masks.to(device)
            optimizer.zero_grad()
            
            m_logits, d_preds = model(imgs)
            
            # 🌟 核心升级 3：AGBR 模糊门控边界特征细化
            # 1. 基础 BCE 损失
            loss_m_base = bce_crit(m_logits, masks).mean()
            
            # 2. 计算基尼系数模糊场 (找茬：寻找网络最犹豫的边缘)
            m_probs = torch.sigmoid(m_logits)
            ambiguity_field = 4.0 * m_probs * (1.0 - m_probs)
            
            # 3. 物理上锁 (Tau 阈值设为 0.8，仅截取极度模糊的体素)
            gating_mask = (ambiguity_field > 0.8).float()
            
            # 4. 对模糊区域进行强烈的二次惩罚
            loss_m_boundary = (bce_crit(m_logits, masks) * gating_mask).mean()
            
            # 最终 Mask 损失：大盘基础学习 + 边缘重点打击
            loss_m = loss_m_base + 2.0 * loss_m_boundary
            
            # --- Depth 终极复合损失 ---
            loss_d = l1_crit(d_preds * masks, depths * masks) \
                     + 1.5 * grad_crit(d_preds * masks, depths * masks) \
                     + 0.5 * smooth_crit(d_preds * masks, imgs) 
            
            total_loss = awl(loss_m, loss_d)
            total_loss.backward()
            optimizer.step()
            
            epoch_m_loss += loss_m.item()
            epoch_d_loss += loss_d.item()
            epoch_total += total_loss.item()
            
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Total': f"{total_loss.item():.4f}",
                'LR': f"{current_lr:.6f}"
            })

        scheduler.step()

        writer.add_scalar('Loss/Train_Total', epoch_total/len(train_loader), epoch)
        writer.add_scalar('Loss/Task_Mask_Raw', epoch_m_loss/len(train_loader), epoch)
        writer.add_scalar('Loss/Task_Depth_Raw', epoch_d_loss/len(train_loader), epoch)
        
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, f"joint_epoch_{epoch}.pth"))
            
        if epoch == 100:
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "joint_best.pth"))
            print(f"🏆 AACNet赋能版模型已保存至 {MODEL_SAVE_DIR}/joint_best.pth")

    writer.close()