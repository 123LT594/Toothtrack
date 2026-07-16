#最终版，利用真实的相机内参 K，直接回归物理度量
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
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
import timm 

MAX_DEPTH = 0.2 

# ================= 1. 基础 Loss 模块 =================
class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num_tasks=2):
        super(AutomaticWeightedLoss, self).__init__()
        self.params = nn.Parameter(torch.zeros(num_tasks))
    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 * torch.exp(-self.params[i]) * loss + 0.5 * self.params[i]
        return loss_sum

class EdgeAwareSmoothnessLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, depth, img):
        depth_dx = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
        depth_dy = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])
        img_dx = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]).mean(dim=1, keepdim=True)
        img_dy = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]).mean(dim=1, keepdim=True)
        weight_x = torch.exp(-img_dx * 10.0)
        weight_y = torch.exp(-img_dy * 10.0)
        return (depth_dx * weight_x).mean() + (depth_dy * weight_y).mean()

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

# ================= 2. 物理反投影 Loss =================
class ProjectionLoss(nn.Module):
    def __init__(self, K, device):
        super().__init__()
        self.K = K.to(device)
        self.fx, self.fy = self.K[0, 0], self.K[1, 1]
        self.cx, self.cy = self.K[0, 2], self.K[1, 2]

    def backproject(self, depth):
        B, _, H, W = depth.shape
        y, x = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(W, device=depth.device), indexing='ij')
        u = (x.float() - self.cx) / self.fx
        v = (y.float() - self.cy) / self.fy
        z = depth.squeeze(1) 
        pts_x = u.unsqueeze(0) * z
        pts_y = v.unsqueeze(0) * z
        return torch.stack([pts_x, pts_y, z], dim=-1) 

    def forward(self, pred_depth, gt_depth, mask):
        pts_pred = self.backproject(pred_depth)
        pts_gt = self.backproject(gt_depth)
        mask_3d = mask.squeeze(1).unsqueeze(-1) 
        pts_pred_masked = pts_pred * mask_3d
        pts_gt_masked = pts_gt * mask_3d
        return F.l1_loss(pts_pred_masked, pts_gt_masked)

# ================= 3. 彻底解耦的网络架构 =================
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
        self.encoder = timm.create_model('convnext_tiny', pretrained=True, features_only=True)
        
        self.m_dec4 = DecoderBlock(768, 384, 384); self.m_dec3 = DecoderBlock(384, 192, 192)
        self.m_dec2 = DecoderBlock(192, 96, 96); self.m_dec1 = DecoderBlock(96, 0, 32)    
        self.m_dec0 = DecoderBlock(32, 0, 16); self.mask_head = nn.Conv2d(16, 1, 1)

        self.d_dec4 = DecoderBlock(768, 384, 384); self.d_dec3 = DecoderBlock(384, 192, 192) 
        self.d_dec2 = DecoderBlock(192, 96, 96); self.d_dec1 = DecoderBlock(96, 0, 32)     
        self.d_dec0 = DecoderBlock(32, 0, 16); self.depth_head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        features = self.encoder(x)
        e1, e2, e3, e4 = features[0], features[1], features[2], features[3]
        
        m4 = self.m_dec4(e4, e3); m3 = self.m_dec3(m4, e2); m2 = self.m_dec2(m3, e1)
        m1 = self.m_dec1(m2); m0 = self.m_dec0(m1)
        mask_logits = self.mask_head(m0)
        
        d4 = self.d_dec4(e4, e3); d3 = self.d_dec3(d4, e2); d2 = self.d_dec2(d3, e1)
        d1 = self.d_dec1(d2); d0 = self.d_dec0(d1)
        
        # 🚀 同步解耦：控制变量，除了 Loss 不加基尼，架构完全一致
        depth_physical = torch.sigmoid(self.depth_head(d0)) * MAX_DEPTH
        
        return mask_logits, depth_physical

# ================= 4. 数据集 =================
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
        mask = (depth > 0.0001).astype(np.float32) 
        
        if self.transform:
            augmented = self.transform(image=image, masks=[depth, mask])
            image, (depth, mask) = augmented['image'], augmented['masks']
        return image, depth.unsqueeze(0), mask.unsqueeze(0)

# ================= 5. 主循环 =================
if __name__ == "__main__":
    device = torch.device('cuda')
    
    MODEL_SAVE_DIR = '/root/lanyun-tmp/models/models_joint_physical'
    LOG_DIR = '/root/lanyun-tmp/logs/joint_exp_physical'
    
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_DIR), exist_ok=True)
    writer = SummaryWriter(LOG_DIR)

    K_matrix = torch.tensor([[2866.3146, 0.0, 480.0],
                             [0.0, 2866.3146, 270.0],
                             [0.0, 0.0, 1.0]], dtype=torch.float32)

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
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=8, pin_memory=True) 

    model = SwinMultiTaskUNet().to(device)
    awl = AutomaticWeightedLoss(num_tasks=2).to(device)
    
    optimizer = optim.AdamW([
        {'params': model.parameters()},
        {'params': awl.parameters(), 'weight_decay': 0}
    ], lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    
    bce_crit = nn.BCEWithLogitsLoss(reduction='none') 
    l1_crit = nn.L1Loss()
    grad_crit = GradientLoss()
    smooth_crit = EdgeAwareSmoothnessLoss() 
    proj_crit = ProjectionLoss(K_matrix, device) 

    for epoch in range(1, 101):
        model.train()
        epoch_m_loss = 0; epoch_d_loss = 0; epoch_total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/100 [Ablation: No Gini]", unit="batch")
        
        for i, (imgs, depths, masks) in enumerate(train_loader):
            imgs, depths, masks = imgs.to(device), depths.to(device), masks.to(device)
            optimizer.zero_grad()
            
            m_logits, d_preds = model(imgs)
            
            # 🌟 唯一变量区：无基尼门控，纯朴素全图 BCE 损失
            loss_m = bce_crit(m_logits, masks).mean()
            
            loss_d_l1 = l1_crit(d_preds * masks, depths * masks)
            loss_d_grad = grad_crit(d_preds * masks, depths * masks)
            loss_d_smooth = smooth_crit(d_preds * masks, imgs)
            loss_d_proj = proj_crit(d_preds, depths, masks) 
            
            loss_d = loss_d_l1 + 1.5 * loss_d_grad + 0.5 * loss_d_smooth + 0.5 * loss_d_proj
            
            total_loss = awl(loss_m, loss_d)
            total_loss.backward()
            optimizer.step()
            
            epoch_m_loss += loss_m.item(); epoch_d_loss += loss_d.item(); epoch_total += total_loss.item()
            pbar.set_postfix({'Total': f"{total_loss.item():.4f}"})

        scheduler.step()
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, f"joint_epoch_{epoch}.pth"))
        if epoch == 100:
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "joint_best.pth"))

    writer.close()