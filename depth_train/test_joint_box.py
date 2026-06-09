import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# ================= 1. 解耦网络架构 (支持动态 MAX_DEPTH) =================
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
    def __init__(self, max_depth):
        super().__init__()
        self.max_depth = max_depth
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
        
        return self.mask_head(m_d1), torch.sigmoid(self.depth_head(d_d1)) * self.max_depth

# ================= 2. 主测试逻辑 =================
def test_joint_box():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dir = "/root/Toothtrack/demo_data/tooth/rgb"
    out_mask = "test_results_box/mask"
    out_depth_masked = "test_results_box/depth"
    out_depth_full = "test_results_box/depth_full"
    
    os.makedirs(out_mask, exist_ok=True)
    os.makedirs(out_depth_masked, exist_ok=True)
    os.makedirs(out_depth_full, exist_ok=True)

    # 🌟 核心：加载两位专属领域的专家！
    print("加载全景定位专家 (model_full)...")
    model_full = DecoupledMultiTaskUNet(max_depth=0.3).to(device)
    model_full.load_state_dict(torch.load("/root/lanyun-tmp/models/models_joint/joint_best.pth", map_location=device))
    model_full.eval()

    print("加载局部高精专家 (model_box)...")
    model_box = DecoupledMultiTaskUNet(max_depth=0.194).to(device)
    model_box.load_state_dict(torch.load("/root/lanyun-tmp/models/models_joint_box/joint_best.pth", map_location=device))
    model_box.eval()

    files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
    print(f"🚀 双剑合璧推理启动：全景精准定位 + 局部物理还原，共 {len(files)} 帧...")

    target_size = 512
    crop_margin = 1.5

    with torch.no_grad():
        for f in tqdm(files, desc="推理中", unit="帧"):
            img_bgr = cv2.imread(os.path.join(input_dir, f))
            H, W = img_bgr.shape[:2]
            
            # =======================================================================
            # ✅ 阶段一：全景专家原生定位 (绝对无畸变)
            # =======================================================================
            pad_h = (32 - H % 32) % 32
            pad_w = (32 - W % 32) % 32
            img_padded = cv2.copyMakeBorder(img_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
            
            tensor_full = torch.from_numpy(cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)).float().permute(2,0,1).unsqueeze(0).to(device) / 255.0
            
            # 全景专家在它最熟悉的原始分辨率下进行预测
            m_logits_full, _ = model_full(tensor_full)
            m_full_np = (torch.sigmoid(m_logits_full) > 0.5).squeeze().cpu().numpy()
            
            m_full_np = m_full_np[:H, :W] # 去除边缘 Pad
            ys, xs = np.nonzero(m_full_np)
            
            if len(ys) < 100: 
                cx, cy, span = W / 2.0, H / 2.0, min(W, H) / 2.0
            else:
                cx = (xs.max() + xs.min()) / 2.0
                cy = (ys.max() + ys.min()) / 2.0
                span = max(xs.max() - xs.min(), ys.max() - ys.min())

            # =======================================================================
            # ✅ 阶段二：高精专家仿射预测与物理还原
            # =======================================================================
            s_factor = target_size / (span * crop_margin)
            M = np.zeros((2, 3), dtype=np.float32)
            M[0, 0] = s_factor; M[1, 1] = s_factor
            
            M[0, 2] = 256.0 - cx * s_factor
            M[1, 2] = 256.0 - cy * s_factor
            
            color_crop = cv2.warpAffine(img_bgr, M, (target_size, target_size), flags=cv2.INTER_LINEAR)
            tensor_box = torch.from_numpy(cv2.cvtColor(color_crop, cv2.COLOR_BGR2RGB)).float().permute(2,0,1).unsqueeze(0).to(device) / 255.0
            
            # 局部高精专家出马
            m_logits_box, d_preds_box = model_box(tensor_box)
            
            mask_crop_soft = torch.sigmoid(m_logits_box).squeeze().cpu().numpy().astype(np.float32)
            depth_crop = d_preds_box.squeeze().cpu().numpy()
            
            # 🌟 物理补偿：将虚拟深度还原为极其精确的真实物理深度
            depth_crop = depth_crop * s_factor
            
            # --- 逆变换与边缘处理 ---
            valid_mask = (mask_crop_soft > 0.5).astype(np.uint8)
            depth_dilated = cv2.dilate(depth_crop, np.ones((5,5), np.uint8), iterations=3)
            depth_crop_bleed = np.where(valid_mask > 0, depth_crop, depth_dilated)

            M_inv = cv2.invertAffineTransform(M)
            
            full_mask_soft = cv2.warpAffine(mask_crop_soft, M_inv, (W, H), flags=cv2.INTER_LINEAR)
            full_depth = cv2.warpAffine(depth_crop_bleed, M_inv, (W, H), flags=cv2.INTER_LINEAR)
            
            binary_mask = (full_mask_soft > 0.5).astype(np.uint8)

            # --- 拓扑修复 ---
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            if num_labels > 1:
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                largest_mask = (labels == largest_label).astype(np.uint8)
            else:
                largest_mask = binary_mask

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
            largest_mask = cv2.morphologyEx(largest_mask, cv2.MORPH_CLOSE, kernel)

            im_padded = cv2.copyMakeBorder(largest_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
            h_pad, w_pad = im_padded.shape
            mask_flood = np.zeros((h_pad + 2, w_pad + 2), np.uint8)
            cv2.floodFill(im_padded, mask_flood, (0, 0), 1)
            im_floodfill_inv = 1 - im_padded
            hole_mask = im_floodfill_inv[1:-1, 1:-1] 
            
            final_mask_bin = largest_mask | hole_mask
                
            # ================= [输出保存与终端监控] =================
            cv2.imwrite(os.path.join(out_mask, f), final_mask_bin * 255)
            
            if final_mask_bin.max() == 0:
                blank = np.zeros((H, W), dtype=np.uint8)
                cv2.imwrite(os.path.join(out_depth_masked, f), blank)
                cv2.imwrite(os.path.join(out_depth_full, f), blank)
                tqdm.write(f"⚠️ {f} | 绝对深度范围: 未检测到目标")
                continue

            eroded_mask = cv2.erode(final_mask_bin, np.ones((5,5), np.uint8))
            v_valid = full_depth[eroded_mask > 0]
            
            if len(v_valid) > 0 and v_valid.max() > v_valid.min():
                # 🌟 获取核心有效区域的极值并安全打印
                d_min = v_valid.min()
                d_max = v_valid.max()
                tqdm.write(f"✅ {f} | 绝对深度范围: {d_min:.4f}m ~ {d_max:.4f}m")
                
                vmin, vmax = np.percentile(v_valid, 1), np.percentile(v_valid, 99)
                norm_full = np.clip((full_depth - vmin) / (vmax - vmin + 1e-5), 0, 1) * 255
                vis_full = 255 - norm_full
                
                depth_full_grayscale = vis_full.copy()
                depth_full_grayscale[full_depth == 0] = 0
                cv2.imwrite(os.path.join(out_depth_full, f), depth_full_grayscale.astype(np.uint8))
                
                depth_masked_grayscale = (vis_full * final_mask_bin).astype(np.uint8)
                cv2.imwrite(os.path.join(out_depth_masked, f), depth_masked_grayscale)
            else:
                blank = np.zeros((H, W), dtype=np.uint8)
                cv2.imwrite(os.path.join(out_depth_masked, f), blank)
                cv2.imwrite(os.path.join(out_depth_full, f), blank)
                tqdm.write(f"⚠️ {f} | 绝对深度范围: 深度值异常")

if __name__ == "__main__":
    test_joint_box()