import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

MAX_DEPTH = 0.3 # 必须和训练时保持一致

class PureDualHeadUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 🌟 核心修正：使用 InstanceNorm2d 适配小 Batch 训练的模型，并提升通道数
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), 
                nn.InstanceNorm2d(out_c, affine=True), 
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1), 
                nn.InstanceNorm2d(out_c, affine=True), 
                nn.ReLU(inplace=True)
            )
        self.enc1 = conv_block(3, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.pool = nn.MaxPool2d(2)
        
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = conv_block(512, 256) 
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = conv_block(256, 128) 
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = conv_block(128, 64) 
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = conv_block(32, 32) 
        
        self.depth_head = nn.Conv2d(32, 1, 1)
        self.mask_head = nn.Conv2d(32, 1, 1)

    def forward(self, rgb):
        # 推理时不需要 checkpoint，直接调用
        return self._forward_impl(rgb)

    def _forward_impl(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        
        def up_and_concat(up_layer, prev_feature, skip_feature):
            x_up = up_layer(prev_feature)
            if x_up.shape[2:] != skip_feature.shape[2:]:
                x_up = F.interpolate(x_up, size=skip_feature.shape[2:], mode='bilinear', align_corners=True)
            return torch.cat([x_up, skip_feature], dim=1)

        d4 = self.dec4(up_and_concat(self.up4, s4, s3))
        d3 = self.dec3(up_and_concat(self.up3, d4, s2))
        d2 = self.dec2(up_and_concat(self.up2, d3, s1))
        
        d1 = self.up1(d2)
        if d1.shape[2:] != x.shape[2:]:
            d1 = F.interpolate(d1, size=x.shape[2:], mode='bilinear', align_corners=True)
        d1 = self.dec1(d1)
        
        return torch.sigmoid(self.depth_head(d1)) * MAX_DEPTH, self.mask_head(d1)