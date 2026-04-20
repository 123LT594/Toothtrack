import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_DEPTH = 0.3

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

class MultiTaskUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        
        # --- 共享 Encoder ---
        self.enc1 = DoubleConv(3, 32); self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128); self.enc4 = DoubleConv(128, 256)
        self.bot = DoubleConv(256, 512)
        
        # --- 独立 Mask 支路 ---
        self.m_dec4 = DoubleConv(512 + 256, 256); self.m_dec3 = DoubleConv(256 + 128, 128)
        self.m_dec2 = DoubleConv(128 + 64, 64); self.m_dec1 = DoubleConv(64 + 32, 32)
        self.mask_head = nn.Conv2d(32, 1, 1)
        
        # --- 独立 Depth 支路 ---
        self.d_dec4 = DoubleConv(512 + 256, 256); self.d_dec3 = DoubleConv(256 + 128, 128)
        self.d_dec2 = DoubleConv(128 + 64, 64); self.d_dec1 = DoubleConv(64 + 32, 32)
        self.depth_head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        # 1. 共享特征提取
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bot(self.pool(e4))
        
        # 2. Mask 独立逐层解码 (坚决不搞一行流嵌套)
        m_u4 = F.interpolate(b, size=e4.shape[2:], mode='bilinear', align_corners=True)
        m_d4 = self.m_dec4(torch.cat([m_u4, e4], dim=1))
        
        m_u3 = F.interpolate(m_d4, size=e3.shape[2:], mode='bilinear', align_corners=True)
        m_d3 = self.m_dec3(torch.cat([m_u3, e3], dim=1))
        
        m_u2 = F.interpolate(m_d3, size=e2.shape[2:], mode='bilinear', align_corners=True)
        m_d2 = self.m_dec2(torch.cat([m_u2, e2], dim=1))
        
        m_u1 = F.interpolate(m_d2, size=e1.shape[2:], mode='bilinear', align_corners=True)
        m_d1 = self.m_dec1(torch.cat([m_u1, e1], dim=1))
        
        # 3. Depth 独立逐层解码
        d_u4 = F.interpolate(b, size=e4.shape[2:], mode='bilinear', align_corners=True)
        d_d4 = self.d_dec4(torch.cat([d_u4, e4], dim=1))
        
        d_u3 = F.interpolate(d_d4, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d_d3 = self.d_dec3(torch.cat([d_u3, e3], dim=1))
        
        d_u2 = F.interpolate(d_d3, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d_d2 = self.d_dec2(torch.cat([d_u2, e2], dim=1))
        
        d_u1 = F.interpolate(d_d2, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d_d1 = self.d_dec1(torch.cat([d_u1, e1], dim=1))
        
        return self.mask_head(m_d1), torch.sigmoid(self.depth_head(d_d1)) * MAX_DEPTH