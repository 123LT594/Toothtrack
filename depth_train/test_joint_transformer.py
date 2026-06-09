import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import timm  
from tqdm import tqdm

# ⚠️ 必须与你训练时的设定严格保持一致
MAX_DEPTH = 0.2

# ================= 1. 终极双头 Transformer (AACNet魔改版) =================
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
        self.encoder = timm.create_model('convnext_tiny', pretrained=False, features_only=True)
        
        self.m_dec4 = DecoderBlock(768, 384, 384)
        self.m_dec3 = DecoderBlock(384, 192, 192)
        self.m_dec2 = DecoderBlock(192, 96, 96)
        self.m_dec1 = DecoderBlock(96, 0, 32)
        self.m_dec0 = DecoderBlock(32, 0, 16)
        self.mask_head = nn.Conv2d(16, 1, 1)

        self.d_dec4 = DecoderBlock(768, 384, 384) 
        self.d_dec3 = DecoderBlock(384, 192, 192) 
        self.d_dec2 = DecoderBlock(192, 96, 96)   
        self.d_dec1 = DecoderBlock(96, 0, 32)     
        self.d_dec0 = DecoderBlock(32, 0, 16)     
        self.depth_head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        features = self.encoder(x)
        e1, e2, e3, e4 = features[0], features[1], features[2], features[3]
        
        # --- Mask 分支 ---
        m4 = self.m_dec4(e4, e3); m3 = self.m_dec3(m4, e2); m2 = self.m_dec2(m3, e1)
        m1 = self.m_dec1(m2); m0 = self.m_dec0(m1)
        mask_logits = self.mask_head(m0)
        
        # 🌟 SDMAA 解剖感知注意力
        mask_attention = torch.sigmoid(mask_logits)
        
        # --- Depth 分支 ---
        d4 = self.d_dec4(e4, e3); d3 = self.d_dec3(d4, e2); d2 = self.d_dec2(d3, e1)
        d1 = self.d_dec1(d2); d0 = self.d_dec0(d1)
        
        # 🌟 深度特征被 Mask 门控引导
        d0_guided = d0 * mask_attention
        
        return mask_logits, torch.sigmoid(self.depth_head(d0_guided)) * MAX_DEPTH

# ================= 2. 极简极速对标推理逻辑 =================
def test_pure_transformer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- 路径配置 ---
    input_dir = "/root/Toothtrack/demo_data/tooth/rgb"
    gt_depth_dir = "/root/lanyun-tmp/golden_dataset/depth"
    
    out_mask = "test_results_pure_aacnet/mask"
    out_depth_masked = "test_results_pure_aacnet/depth"
    
    os.makedirs(out_mask, exist_ok=True)
    os.makedirs(out_depth_masked, exist_ok=True)

    print("🚀 加载纯净版 Transformer 统一模型...")
    model = SwinMultiTaskUNet().to(device)
    model.load_state_dict(torch.load("/root/lanyun-tmp/models/models_joint_transformer_aacnet/joint_best.pth", map_location=device))
    model.eval()

    files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
    print(f"\n🔥 终极单模型推理启动：共 {len(files)} 帧...\n")

    # ImageNet 归一化参数
    MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3).astype(np.float32)
    STD = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3).astype(np.float32)

    with torch.no_grad():
        for f in tqdm(files, desc="推理中", unit="帧"):
            
            # ================= [前置拦截 GT] =================
            base_name = os.path.splitext(f)[0]
            gt_path = os.path.join(gt_depth_dir, base_name + '.npy')
            if not os.path.exists(gt_path):
                # 找不到 GT 文件直接跳过，连图都不读
                continue

            # ================= [前向推理] =================
            img_bgr = cv2.imread(os.path.join(input_dir, f))
            H, W = img_bgr.shape[:2]
            
            pad_h = (32 - H % 32) % 32
            pad_w = (32 - W % 32) % 32
            img_padded = cv2.copyMakeBorder(img_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
            img_rgb_padded = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
            
            img_norm = (img_rgb_padded.astype(np.float32) / 255.0 - MEAN) / STD
            tensor_in = torch.from_numpy(img_norm).float().permute(2,0,1).unsqueeze(0).to(device)
            
            m_logits, d_preds = model(tensor_in)
            
            final_mask = (torch.sigmoid(m_logits) > 0.5).squeeze().cpu().numpy().astype(np.uint8)[:H, :W]
            depth_raw = d_preds.squeeze().cpu().numpy()[:H, :W]
            
            # 保存 Mask
            cv2.imwrite(os.path.join(out_mask, f), final_mask * 255)
            
            if final_mask.max() == 0:
                cv2.imwrite(os.path.join(out_depth_masked, f), np.zeros((H, W), dtype=np.uint8))
                tqdm.write(f"⚠️ {f} | 未检测到目标")
                continue

            # ================= [对标输出] =================
            depth_masked = depth_raw * final_mask
            v_valid_pred = depth_masked[final_mask > 0]
            
            if os.path.exists(gt_path):
                gt_depth = np.load(gt_path)
                v_valid_gt = gt_depth[gt_depth > 0.0001]
                
                if len(v_valid_pred) > 0 and len(v_valid_gt) > 0:
                    pred_min, pred_max = v_valid_pred.min(), v_valid_pred.max()
                    gt_min, gt_max = v_valid_gt.min(), v_valid_gt.max()
                    
                    # 🌟 打印出物理对齐的震撼效果！
                    tqdm.write(f"✅ {f} | 预测深度: {pred_min:.4f}m ~ {pred_max:.4f}m | 🎯 GT深度: {gt_min:.4f}m ~ {gt_max:.4f}m")

                # ================= [终极渲染逻辑] =================
                # 1. 向内腐蚀，避开边缘软过渡跌落区，找到真正的中心物理极值
                kernel = np.ones((5, 5), np.uint8)
                eroded_mask = cv2.erode(final_mask, kernel)
                v_core = depth_masked[eroded_mask > 0]
                
                if len(v_core) > 0 and v_core.max() > v_core.min():
                    true_vmin = v_core.min()
                    true_vmax = v_core.max()
                    
                    # 2. 将深度限制在真实极值内，防止溢出
                    depth_clipped = np.clip(depth_masked, true_vmin, true_vmax)
                    
                    # 3. 线性映射至 0~255 (近处0，远处255)
                    norm = (depth_clipped - true_vmin) / (true_vmax - true_vmin + 1e-5)
                    vis = norm * 255.0
                    
                    # 4. 颜色翻转 (近白远黑) 并乘上 Mask 清除背景
                    vis_inverted = (255.0 - vis) * final_mask
                    
                    # 5. 🔫 终极白边杀手 (Halo Killer)！
                    # 将所有低于真实极值（被软过渡坑成白边的边缘像素）强行涂黑
                    vis_inverted[depth_masked < true_vmin] = 0
                    
                    depth_grayscale_final = vis_inverted.astype(np.uint8)
                    cv2.imwrite(os.path.join(out_depth_masked, f), depth_grayscale_final)
                else:
                    cv2.imwrite(os.path.join(out_depth_masked, f), np.zeros((H, W), dtype=np.uint8))
                    tqdm.write(f"⚠️ {f} | 核心深度极值异常")
            else:
                cv2.imwrite(os.path.join(out_depth_masked, f), np.zeros((H, W), dtype=np.uint8))
                tqdm.write(f"⚠️ {f} | 深度值异常")

if __name__ == "__main__":
    test_pure_transformer()