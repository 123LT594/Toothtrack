import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from train_mask0 import SimpleUNet, MaskDataset, train_transform, val_transform

MAX_DEPTH = 0.3

# 1. 深度网络定义
class DepthUNet(SimpleUNet):
    def forward(self, x):
        return torch.sigmoid(super().forward(x)) * MAX_DEPTH

# 2. 梯度损失定义
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

# 3. 深度数据集定义
class DepthDataset(MaskDataset):
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

# 4. 训练循环
if __name__ == "__main__":
    device = torch.device('cuda')
    MODEL_SAVE_DIR = '/root/lanyun-tmp/models/models0'
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # 🌟 数据集路径修正
    full_ds = MaskDataset('../golden_dataset', train_transform)
    train_size = int(0.9 * len(full_ds))
    train_indices, val_indices = random_split(range(len(full_ds)), [train_size, len(full_ds)-train_size])
    
    train_loader = DataLoader(DepthDataset('golden_dataset', train_transform), batch_size=4, sampler=torch.utils.data.SubsetRandomSampler(train_indices))
    val_loader = DataLoader(DepthDataset('golden_dataset', val_transform), batch_size=4, sampler=torch.utils.data.SubsetRandomSampler(val_indices))

    model = DepthUNet().to(device)
    l1_crit = nn.L1Loss(); grad_crit = GradientLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')

    for epoch in range(1, 101):
        model.train(); t_loss = 0
        for imgs, depths, masks in train_loader:
            imgs, depths, masks = imgs.to(device), depths.to(device), masks.to(device)
            optimizer.zero_grad(); preds = model(imgs) * masks
            loss = l1_crit(preds, depths * masks) + 1.5 * grad_crit(preds, depths * masks)
            loss.backward(); optimizer.step(); t_loss += loss.item()
        
        print(f"Epoch {epoch}/100 | Depth Loss: {t_loss/len(train_loader):.6f}")

        # 每 5 轮执行一次最优模型检测
        if epoch % 5 == 0:
            model.eval(); v_loss = 0
            with torch.no_grad():
                for imgs, depths, masks in val_loader:
                    imgs, depths, masks = imgs.to(device), depths.to(device), masks.to(device)
                    preds = model(imgs) * masks
                    v_loss += (l1_crit(preds, depths * masks) + 1.5 * grad_crit(preds, depths * masks)).item()
            
            avg_v_loss = v_loss / len(val_loader)
            print(f"📊 Validation Loss: {avg_v_loss:.6f}")

            if avg_v_loss < best_val_loss:
                best_val_loss = avg_v_loss
                torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "depth_best.pth"))