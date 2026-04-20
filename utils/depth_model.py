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

class DepthUNet(SimpleUNet):
    def forward(self, x): return torch.sigmoid(super().forward(x)) * MAX_DEPTH