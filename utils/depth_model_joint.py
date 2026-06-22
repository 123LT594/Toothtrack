import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

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
        # 底层唯一共享的特质提取引擎
        self.encoder = timm.create_model('convnext_tiny', pretrained=False, features_only=True)
        
        # 独立的 Mask 分割解码分支
        self.m_dec4 = DecoderBlock(768, 384, 384); self.m_dec3 = DecoderBlock(384, 192, 192)
        self.m_dec2 = DecoderBlock(192, 96, 96); self.m_dec1 = DecoderBlock(96, 0, 32)    
        self.m_dec0 = DecoderBlock(32, 0, 16); self.mask_head = nn.Conv2d(16, 1, 1)

        # 独立的 Depth 物理测绘解码分支
        self.d_dec4 = DecoderBlock(768, 384, 384); self.d_dec3 = DecoderBlock(384, 192, 192) 
        self.d_dec2 = DecoderBlock(192, 96, 96); self.d_dec1 = DecoderBlock(96, 0, 32)     
        self.d_dec0 = DecoderBlock(32, 0, 16); self.depth_head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        features = self.encoder(x)
        e1, e2, e3, e4 = features[0], features[1], features[2], features[3]
        
        # Mask 独立流
        m4 = self.m_dec4(e4, e3); m3 = self.m_dec3(m4, e2); m2 = self.m_dec2(m3, e1)
        m1 = self.m_dec1(m2); m0 = self.m_dec0(m1)
        mask_logits = self.mask_head(m0)
        
        # Depth 独立流 (完全剔除了任何 mask 概率的污染，保证曲率纯净)
        d4 = self.d_dec4(e4, e3); d3 = self.d_dec3(d4, e2); d2 = self.d_dec2(d3, e1)
        d1 = self.d_dec1(d2); d0 = self.d_dec0(d1)
        depth_physical = torch.sigmoid(self.depth_head(d0)) * MAX_DEPTH
        
        return mask_logits, depth_physical