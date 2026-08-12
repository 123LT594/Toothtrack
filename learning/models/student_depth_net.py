import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

class StudentDepthNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载 ConvNeXt-Tiny
        self.backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        
        # 修改第一层卷积以接受 6 通道输入 (RGB + RayMap)
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            6, original_conv.out_channels, 
            kernel_size=original_conv.kernel_size, 
            stride=original_conv.stride, 
            padding=original_conv.padding
        )
        # 初始化新增通道的权重
        nn.init.kaiming_normal_(self.backbone.features[0][0].weight[:, 3:, :, :])
        
        # 深度解码头 (输出原始形状系数，不做 tanh，由训练脚本处理)
        self.depth_head = nn.Sequential(
            nn.Conv2d(768, 256, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 64, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 1, 3, padding=1)
        )
        
        # 掩膜解码头
        self.mask_head = nn.Sequential(
            nn.Conv2d(768, 256, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),   # 32/4=8，即先到 1/8
            nn.Conv2d(256, 64, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),   # 再到原图 1/1
            nn.Conv2d(64, 1, 3, padding=1)
        )
        
        # ★ 全局偏移头：从最深特征图提取全局深度修正量
        self.global_offset_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),        # 将 768×H×W 压缩成 768×1×1
            nn.Flatten(),
            nn.Linear(768, 1)               # 输出一个标量 Δz_raw
        )

    def forward(self, x):
        # 提取 backbone 最深特征 (下采样32倍，通道768)
        feat = self.backbone.features(x)
        
        # 形状系数原始输出 (不激活，后续由 tanh 约束)
        shape_weight_raw = self.depth_head(feat)
        
        # 掩膜概率
        mask_pred = torch.sigmoid(self.mask_head(feat))
        
        # 全局偏移标量 (不激活，后续由 tanh 约束)
        delta_z_scalar = self.global_offset_head(feat)   # shape: (B, 1)
        
        return shape_weight_raw, mask_pred, delta_z_scalar