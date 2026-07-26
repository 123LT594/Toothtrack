import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import timm  
from tqdm import tqdm

MAX_DEPTH = 0.2

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
        
        # 🚀 已同步修改：前向完全解耦
        depth_physical = torch.sigmoid(self.depth_head(d0)) * MAX_DEPTH
        return mask_logits, depth_physical

def test_ablation_transformer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    input_dir = "/root/lanyun-tmp/resized_frames"
    gt_depth_dir = "/root/lanyun-tmp/golden_dataset/depth"
    
    out_mask = "test_results_physical/mask"
    out_depth_masked_png = "test_results_physical/depth_png"
    out_depth_masked_npy = "test_results_physical/depth_npy" 
    out_heatmap = "test_results_physical/heatmap" # 🌟 新增：消融实验热力图
    
    os.makedirs(out_mask, exist_ok=True)
    os.makedirs(out_depth_masked_png, exist_ok=True)
    os.makedirs(out_depth_masked_npy, exist_ok=True)
    os.makedirs(out_heatmap, exist_ok=True)

    print("🚀 加载【无基尼消融版】物理对齐模型...")
    model = SwinMultiTaskUNet().to(device)
    model.load_state_dict(torch.load("/root/lanyun-tmp/models/models_joint_physical/joint_best.pth", map_location=device))
    model.eval()

    files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
    MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3).astype(np.float32)
    STD = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3).astype(np.float32)

    with torch.no_grad():
        for f in tqdm(files, desc="消融推理中"):
            img_bgr = cv2.imread(os.path.join(input_dir, f))
            H, W = img_bgr.shape[:2]
            
            pad_h = (32 - H % 32) % 32; pad_w = (32 - W % 32) % 32
            img_padded = cv2.copyMakeBorder(img_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
            img_rgb_padded = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
            
            img_norm = (img_rgb_padded.astype(np.float32) / 255.0 - MEAN) / STD
            tensor_in = torch.from_numpy(img_norm).float().permute(2,0,1).unsqueeze(0).to(device)
            
            m_logits, d_preds = model(tensor_in)
            
            # 🌟 新增：导出消融实验的热力图
            m_probs = torch.sigmoid(m_logits).squeeze().cpu().numpy()[:H, :W]
            hesitant_count = ((m_probs > 0.1) & (m_probs < 0.9)).sum()
            heatmap = (m_probs * 255).astype(np.uint8)
            
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(out_heatmap, f), heatmap_color)
            
            final_mask = (m_probs > 0.5).astype(np.uint8)
            depth_raw = d_preds.squeeze().cpu().numpy()[:H, :W] 
            depth_masked = depth_raw * final_mask
            
            cv2.imwrite(os.path.join(out_mask, f), final_mask * 255)
            
            if final_mask.max() == 0:
                cv2.imwrite(os.path.join(out_depth_masked_png, f), np.zeros((H, W), dtype=np.uint8))
                np.save(os.path.join(out_depth_masked_npy, f.replace('.png', '.npy')), np.zeros((H, W), dtype=np.float32))
                continue

            np.save(os.path.join(out_depth_masked_npy, f.replace('.png', '.npy')), depth_masked)

            base_name = os.path.splitext(f)[0]
            gt_path = os.path.join(gt_depth_dir, base_name + '.npy')
            v_valid_pred = depth_masked[final_mask > 0]
            
            if os.path.exists(gt_path):
                gt_depth = np.load(gt_path)
                v_valid_gt = gt_depth[gt_depth > 0.0001]
                if len(v_valid_pred) > 0 and len(v_valid_gt) > 0:
                    pred_min, pred_max = v_valid_pred.min(), v_valid_pred.max()
                    gt_min, gt_max = v_valid_gt.min(), v_valid_gt.max()
                    tqdm.write(f"✅ {f} | 预测: {pred_min:.4f}m~{pred_max:.4f}m | 🎯 GT: {gt_min:.4f}m~{gt_max:.4f}m | 👻 犹豫像素: {hesitant_count} 个")

            kernel = np.ones((5, 5), np.uint8)
            eroded_mask = cv2.erode(final_mask, kernel)
            v_core = depth_masked[eroded_mask > 0]
            
            if len(v_core) > 0 and v_core.max() > v_core.min():
                true_vmin = v_core.min(); true_vmax = v_core.max()
                depth_clipped = np.clip(depth_masked, true_vmin, true_vmax)
                norm = (depth_clipped - true_vmin) / (true_vmax - true_vmin + 1e-5)
                vis = norm * 255.0
                vis_inverted = (255.0 - vis) * final_mask
                vis_inverted[depth_masked < true_vmin] = 0 
                cv2.imwrite(os.path.join(out_depth_masked_png, f), vis_inverted.astype(np.uint8))
            else:
                cv2.imwrite(os.path.join(out_depth_masked_png, f), np.zeros((H, W), dtype=np.uint8))

if __name__ == "__main__":
    test_ablation_transformer()