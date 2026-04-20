#使用无钢珠数据集将mask、depth分开训练
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

# ================= 1. 网络架构 =================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = DoubleConv(3, 32); self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128); self.enc4 = DoubleConv(128, 256)
        self.bot = DoubleConv(256, 512)
        self.dec4 = DoubleConv(512 + 256, 256); self.dec3 = DoubleConv(256 + 128, 128)
        self.dec2 = DoubleConv(128 + 64, 64); self.dec1 = DoubleConv(64 + 32, 32)
        self.out_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2)); e4 = self.enc4(self.pool(e3))
        b = self.bot(self.pool(e4))
        u4 = F.interpolate(b, size=e4.shape[2:], mode='bilinear', align_corners=True)
        d4 = self.dec4(torch.cat([u4, e4], dim=1))
        u3 = F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self.out_conv(d1)

# ================= 2. 数据集逻辑 =================
class MaskDataset(Dataset):
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
        mask = (np.nan_to_num(depth, nan=0.0) > 0.0001).astype(np.float32)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        return image, mask.unsqueeze(0)

# ================= 3. 数据增强 =================
aug_params = [
    A.PadIfNeeded(min_height=544, min_width=960, border_mode=cv2.BORDER_CONSTANT, value=0, position='top_left'),
    A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.15), contrast_limit=(-0.2, 0.2), p=0.5),
    A.ColorJitter(saturation=(0.6, 1.4), hue=0, p=0.5),
    A.OneOf([A.GaussianBlur(blur_limit=(3, 7)), A.MotionBlur(blur_limit=5)], p=0.3),
    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=(-0.2, 0.2), rotate_limit=30, p=0.5),
    A.Normalize(mean=(0,0,0), std=(1,1,1)),
    ToTensorV2()
]
train_transform = A.Compose(aug_params)
val_transform = A.Compose([
    A.PadIfNeeded(min_height=544, min_width=960, border_mode=cv2.BORDER_CONSTANT, value=0, position='top_left'),
    A.Normalize(mean=(0,0,0), std=(1,1,1)), ToTensorV2()
])

# ================= 4. 训练循环 =================
if __name__ == "__main__":
    device = torch.device('cuda')
    MODEL_SAVE_DIR = '/root/lanyun-tmp/models/models0'
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    full_ds = MaskDataset('../golden_dataset', train_transform)
    train_size = int(0.9 * len(full_ds))
    train_indices, val_indices = random_split(range(len(full_ds)), [train_size, len(full_ds)-train_size])
    
    train_loader = DataLoader(MaskDataset('golden_dataset', train_transform), batch_size=4, sampler=torch.utils.data.SubsetRandomSampler(train_indices))
    val_loader = DataLoader(MaskDataset('golden_dataset', val_transform), batch_size=4, sampler=torch.utils.data.SubsetRandomSampler(val_indices))

    model = SimpleUNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')

    for epoch in range(1, 51):
        model.train(); t_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad(); preds = model(imgs)
            loss = criterion(preds, masks); loss.backward(); optimizer.step()
            t_loss += loss.item()
        
        print(f"Epoch {epoch}/50 | Train Loss: {t_loss/len(train_loader):.4f}")

        # 每 5 轮执行一次深度抽查
        if epoch % 5 == 0:
            model.eval(); v_loss = 0
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    v_loss += criterion(model(imgs), masks).item()
            
            avg_v_loss = v_loss / len(val_loader)
            print(f"📊 Validation Loss: {avg_v_loss:.4f}")

            if avg_v_loss < best_val_loss:
                best_val_loss = avg_v_loss
                torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "mask_best.pth"))