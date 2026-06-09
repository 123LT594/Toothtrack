import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

MAX_DEPTH = 0.3

# ================= 1. 解耦网络架构 (与最新训练脚本保持一致) =================
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
        
        # 🌟 共享的 Encoder (提取通用特征)
        self.enc1 = DoubleConv(3, 32); self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128); self.enc4 = DoubleConv(128, 256)
        self.bot = DoubleConv(256, 512)
        
        # 🌟 独立的 Mask Decoder
        self.m_dec4 = DoubleConv(512 + 256, 256); self.m_dec3 = DoubleConv(256 + 128, 128)
        self.m_dec2 = DoubleConv(128 + 64, 64); self.m_dec1 = DoubleConv(64 + 32, 32)
        self.mask_head = nn.Conv2d(32, 1, 1)

        # 🌟 独立的 Depth Decoder
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

# ================= 2. 主测试逻辑 =================
def test_joint():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dir = "../demo_data/tooth/rgb"
    out_mask = "test_results_joint_decoupled/mask"   
    out_depth = "test_results_joint_decoupled/depth"
    os.makedirs(out_mask, exist_ok=True)
    os.makedirs(out_depth, exist_ok=True)

    # 🌟 使用新的解耦模型实例化
    model = DecoupledMultiTaskUNet().to(device)
    model.load_state_dict(torch.load("/root/lanyun-tmp/models/models_joint/joint_best.pth", map_location=device))
    model.eval()

    files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
    print(f"🚀 解耦架构联合推理启动，带绝对深度监控，共 {len(files)} 帧...")

    with torch.no_grad():
        for f in tqdm(files, desc="Joint 推理中", unit="帧"):
            img_bgr = cv2.imread(os.path.join(input_dir, f))
            h, w = img_bgr.shape[:2]
            
            # 预处理：按训练尺度补齐黑边
            img_padded = np.zeros((544, 960, 3), dtype=np.float32)
            img_padded[:h, :w, :] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) / 255.0
            img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).unsqueeze(0).to(device)
            
            # 核心：一次推理解锁两个能力
            mask_logits, depth_pred = model(img_tensor)
            
            # ================= [处理 Mask] =================
            pred_mask = torch.sigmoid(mask_logits).squeeze().cpu().numpy()[:h, :w]
            binary_mask = (pred_mask > 0.5).astype(np.uint8)
            
            # CCA 最大连通域净化
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            if num_labels > 1:
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                final_mask_bin = (labels == largest_label).astype(np.uint8)
            else:
                final_mask_bin = binary_mask
                
            cv2.imwrite(os.path.join(out_mask, f), final_mask_bin * 255)
            
            # ================= [处理 Depth & 终端监控] =================
            depth_crop = depth_pred.squeeze().cpu().numpy()[:h, :w]
            
            if final_mask_bin.max() == 0:
                cv2.imwrite(os.path.join(out_depth, f), np.zeros((h, w), dtype=np.uint8))
                tqdm.write(f"⚠️ {f} | 绝对深度范围: 未检测到目标")
                continue
            
            depth_crop = depth_crop * final_mask_bin
            
            # 🌟 核心：为了获取真实的中心有效深度，侵蚀掉 Mask 边缘 5 个像素，防止拿到平滑过渡值
            eroded_mask = cv2.erode(final_mask_bin, np.ones((5,5), np.uint8))
            v_valid = depth_crop[eroded_mask > 0]
            
            if len(v_valid) > 0 and v_valid.max() > v_valid.min():
                # 安全输出终端深度范围
                d_min = v_valid.min()
                d_max = v_valid.max()
                tqdm.write(f"✅ {f} | 绝对深度范围: {d_min:.4f}m ~ {d_max:.4f}m")
                
                # 黑坑白牙归一化 -> 反转为白坑黑牙 -> 背景归零
                v = depth_crop[final_mask_bin > 0] # 可视化拉伸还是用全量有效像素
                norm = (depth_crop - v.min()) / (v.max() - v.min()) * 255
                vis = (norm * final_mask_bin).astype(np.uint8)
                depth_grayscale_final = ((255 - vis) * final_mask_bin).astype(np.uint8)
                cv2.imwrite(os.path.join(out_depth, f), depth_grayscale_final)
            else:
                cv2.imwrite(os.path.join(out_depth, f), np.zeros((h, w), dtype=np.uint8))
                tqdm.write(f"⚠️ {f} | 绝对深度范围: 深度值异常")

    print(f"\n✅ 测试完成！去对比全景模型和裁剪模型的物理深度差异吧！")

if __name__ == "__main__":
    test_joint()