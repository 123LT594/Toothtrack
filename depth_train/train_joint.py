import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from tqdm import tqdm
MAX_DEPTH = 0.3

# ================= 1. 动态权重模块 =================
class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num_tasks=2):
        super(AutomaticWeightedLoss, self).__init__()
        # 初始权重参数为 0
        self.params = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            # 1/2 * exp(-s) * Loss + 1/2 * s
            loss_sum += 0.5 * torch.exp(-self.params[i]) * loss + 0.5 * self.params[i]
        return loss_sum

# ================= 2. 解耦的多任务网络架构 (Y型网络) =================
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
        
        # 🌟 共享的 Encoder (提取通用特征：边缘、结构、纹理)
        self.enc1 = DoubleConv(3, 32); self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128); self.enc4 = DoubleConv(128, 256)
        self.bot = DoubleConv(256, 512)
        
        # 🌟 独立的 Mask Decoder (专心重构锐利边缘)
        self.m_dec4 = DoubleConv(512 + 256, 256); self.m_dec3 = DoubleConv(256 + 128, 128)
        self.m_dec2 = DoubleConv(128 + 64, 64); self.m_dec1 = DoubleConv(64 + 32, 32)
        self.mask_head = nn.Conv2d(32, 1, 1)

        # 🌟 独立的 Depth Decoder (专心雕刻平滑起伏，不受 Mask 高频信号干扰)
        self.d_dec4 = DoubleConv(512 + 256, 256); self.d_dec3 = DoubleConv(256 + 128, 128)
        self.d_dec2 = DoubleConv(128 + 64, 64); self.d_dec1 = DoubleConv(64 + 32, 32)
        self.depth_head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        # --- 共享特征提取 ---
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2)); e4 = self.enc4(self.pool(e3))
        b = self.bot(self.pool(e4))
        
        # --- Mask 解码分支 ---
        m_u4 = F.interpolate(b, size=e4.shape[2:], mode='bilinear', align_corners=True)
        m_d4 = self.m_dec4(torch.cat([m_u4, e4], dim=1))
        m_u3 = F.interpolate(m_d4, size=e3.shape[2:], mode='bilinear', align_corners=True)
        m_d3 = self.m_dec3(torch.cat([m_u3, e3], dim=1))
        m_u2 = F.interpolate(m_d3, size=e2.shape[2:], mode='bilinear', align_corners=True)
        m_d2 = self.m_dec2(torch.cat([m_u2, e2], dim=1))
        m_u1 = F.interpolate(m_d2, size=e1.shape[2:], mode='bilinear', align_corners=True)
        m_d1 = self.m_dec1(torch.cat([m_u1, e1], dim=1))
        
        # --- Depth 解码分支 ---
        d_u4 = F.interpolate(b, size=e4.shape[2:], mode='bilinear', align_corners=True)
        d_d4 = self.d_dec4(torch.cat([d_u4, e4], dim=1))
        d_u3 = F.interpolate(d_d4, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d_d3 = self.d_dec3(torch.cat([d_u3, e3], dim=1))
        d_u2 = F.interpolate(d_d3, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d_d2 = self.d_dec2(torch.cat([d_u2, e2], dim=1))
        d_u1 = F.interpolate(d_d2, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d_d1 = self.d_dec1(torch.cat([d_u1, e1], dim=1))
        
        return self.mask_head(m_d1), torch.sigmoid(self.depth_head(d_d1)) * MAX_DEPTH

# ================= 3. 数据集与辅助类 =================
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
        mask = (depth > 0.0001).astype(np.float32)
        
        if self.transform:
            augmented = self.transform(image=image, masks=[depth, mask])
            image, (depth, mask) = augmented['image'], augmented['masks']
        return image, depth.unsqueeze(0), mask.unsqueeze(0)

if __name__ == "__main__":
    device = torch.device('cuda')
    MODEL_SAVE_DIR = '/root/lanyun-tmp/models/models_joint'
    LOG_DIR = '/root/lanyun-tmp/logs/joint_exp_full_data'
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_DIR), exist_ok=True)
    
    writer = SummaryWriter(LOG_DIR)

    # 数据准备 (原封不动)
    train_tf = A.Compose([
        A.PadIfNeeded(min_height=544, min_width=960, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.RandomBrightnessContrast(p=0.5),
        A.ColorJitter(saturation=(0.6, 1.4), hue=0, p=0.5),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 7)), A.MotionBlur(blur_limit=5)], p=0.3),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=(-0.2, 0.2), rotate_limit=30, p=0.5),
        A.Normalize(mean=(0,0,0), std=(1,1,1)), ToTensorV2()
    ])

    # 🌟 核心修改 1：100% 全量数据出击！干掉验证集划分
    train_ds = JointDataset('/root/lanyun-tmp/golden_dataset', train_tf)
    print(f"🔥 使用 100% 全量数据：共 {len(train_ds)} 帧")
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True) 

    model = DecoupledMultiTaskUNet().to(device)
    awl = AutomaticWeightedLoss(num_tasks=2).to(device)
    
    optimizer = optim.AdamW([
        {'params': model.parameters()},
        {'params': awl.parameters(), 'weight_decay': 0}
    ], lr=1e-4)
    
    # 🌟 核心修改 2：加入余弦退火学习率调度器
    # 让学习率在前 100 个 epoch 里从 1e-4 平滑降到 1e-6，完美沉淀
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    
    bce_crit = nn.BCEWithLogitsLoss()
    l1_crit = nn.L1Loss(); grad_crit = GradientLoss()

    for epoch in range(1, 101):
        model.train()
        epoch_m_loss = 0; epoch_d_loss = 0; epoch_total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/100", unit="batch")
        
        for i, (imgs, depths, masks) in enumerate(train_loader):
            imgs, depths, masks = imgs.to(device), depths.to(device), masks.to(device)
            optimizer.zero_grad()
            
            m_logits, d_preds = model(imgs)
            
            loss_m = bce_crit(m_logits, masks)
            loss_d = l1_crit(d_preds * masks, depths * masks) + 1.5 * grad_crit(d_preds * masks, depths * masks)
            
            total_loss = awl(loss_m, loss_d)
            total_loss.backward()
            optimizer.step()
            
            epoch_m_loss += loss_m.item()
            epoch_d_loss += loss_d.item()
            epoch_total += total_loss.item()
            
            # 记录当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Total': f"{total_loss.item():.4f}",
                'LR': f"{current_lr:.6f}"
            })

        # 🌟 核心修改 3：每个 Epoch 结束时更新学习率
        scheduler.step()

        writer.add_scalar('Loss/Train_Total', epoch_total/len(train_loader), epoch)
        writer.add_scalar('Loss/Task_Mask_Raw', epoch_m_loss/len(train_loader), epoch)
        writer.add_scalar('Loss/Task_Depth_Raw', epoch_d_loss/len(train_loader), epoch)
        writer.add_scalar('Weights/Mask_Dynamic_Weight', torch.exp(-awl.params[0]).item(), epoch)
        writer.add_scalar('Weights/Depth_Dynamic_Weight', torch.exp(-awl.params[1]).item(), epoch)
        writer.add_scalar('HyperParams/Learning_Rate', current_lr, epoch)

        print(f"\n✅ Epoch {epoch} 总结 | Total: {epoch_total/len(train_loader):.6f} | D-Loss: {epoch_d_loss/len(train_loader):.6f}")

        # 🌟 核心修改 4：全量数据盲跑保存策略
        # 没有验证集了，我们直接保存最后的模型。为防万一，每 10 个 Epoch 存一个检查点
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, f"joint_epoch_{epoch}.pth"))
            
        # 最后一个 Epoch 的模型就是我们的最终兵器
        if epoch == 100:
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "joint_best.pth"))
            print(f"🏆 最终模型已保存至 {MODEL_SAVE_DIR}/joint_best.pth")

    writer.close()