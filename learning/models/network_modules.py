# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os,sys,copy,math,tqdm
import numpy as np
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
import torch.nn.functional as F
import torch
import torch.nn as nn
import time
import cv2
sys.path.append(f'{dir_path}/../../../../')




class ConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, groups=1, bias=True,dilation=1,):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, groups=groups, bias=bias,dilation=dilation),
            nn.BatchNorm2d(C_out),
        )

    def forward(self, x):
        return self.net(x)


class ConvBNReLU(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, groups=1, bias=True,dilation=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        padding = (kernel_size - 1) // 2
        layers = [
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, groups=groups, bias=bias,dilation=dilation),
        ]
        if norm_layer is not None:
          layers.append(norm_layer(C_out))
        layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConvPadding(nn.Module):
  def __init__(self,C_in, C_out, kernel_size=3, stride=1, groups=1, bias=True,dilation=1):
    super(ConvPadding, self).__init__()
    padding = (kernel_size - 1) // 2
    self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride, padding, groups=groups, bias=bias,dilation=dilation)

  def forward(self,x):
    return self.conv(x)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

class ResnetBasicBlock(nn.Module):
  __constants__ = ['downsample']

  def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=nn.BatchNorm2d, bias=False):
    super().__init__()
    self.norm_layer = norm_layer
    if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
    if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
    # Both self.conv1 and self.downsample layers downsample the input when stride != 1
    self.conv1 = conv3x3(inplanes, planes, stride,bias=bias)
    if self.norm_layer is not None:
      self.bn1 = norm_layer(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes,bias=bias)
    if self.norm_layer is not None:
      self.bn2 = norm_layer(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    if self.norm_layer is not None:
      out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    if self.norm_layer is not None:
      out = self.bn2(out)

    if self.downsample is not None:
      identity = self.downsample(x)
    out += identity
    out = self.relu(out)

    return out

class DeformableCrossAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dim = channels
        self.offset_net = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 2, kernel_size=3, padding=1)
        )
        self.attn = nn.MultiheadAttention(embed_dim=self.dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(self.dim)
        self.gamma = nn.Parameter(torch.tensor([0.0]))
        
    def forward(self, curr_feat, ref_feat):
        B, C, H, W = curr_feat.shape
        cat_feat = torch.cat([curr_feat, ref_feat], dim=1)
        offsets = self.offset_net(cat_feat) 
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H, device=curr_feat.device), 
                                        torch.linspace(-1, 1, W, device=curr_feat.device), indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        deformed_grid = grid + offsets.permute(0, 2, 3, 1) 
        aligned_ref = F.grid_sample(ref_feat, deformed_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        query = curr_feat.view(B, C, -1).permute(0, 2, 1)
        key_value = aligned_ref.view(B, C, -1).permute(0, 2, 1)
        attn_out, attn_w = self.attn(query, key_value, key_value)
        res = self.norm(attn_out).permute(0, 2, 1).view(B, C, H, W)
        return curr_feat + self.gamma * res, offsets, attn_w


class PositionalEmbedding(nn.Module):
  def __init__(self, d_model, max_len=512):
    super().__init__()

    # Compute the positional encodings once in log space.
    pe = torch.zeros(max_len, d_model).float()
    pe.require_grad = False

    position = torch.arange(0, max_len).float().unsqueeze(1)  #(N,1)
    div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()[None]

    pe[:, 0::2] = torch.sin(position * div_term)  #(N, d_model/2)
    pe[:, 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)  #(1, max_len, D)


  def forward(self, x):
    '''
    @x: (B,N,D)
    '''
    return x + self.pe[:, :x.size(1)]

