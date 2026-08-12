import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
import random

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from learning.training.training_config import DISTILL_K_BASE, DISTILL_PHYSICAL_WIDTH

class DualDistillDataset(Dataset):
    def __init__(self, data_dir=None, is_training=True):
        super().__init__()
        if data_dir is None:
            _current_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.abspath(os.path.join(_current_dir, "../../lanyun-tmp/golden_dataset"))
        self.rgb_dir = os.path.join(data_dir, "rgb")
        self.depth_dir = os.path.join(data_dir, "depth")
        self.pose_dir = os.path.join(data_dir, "pose")
        self.frames = [f.split('.')[0] for f in os.listdir(self.pose_dir) if f.endswith('.npy')]
        self.is_training = is_training
        
        self.K_base = np.array(DISTILL_K_BASE, dtype=np.float32)
        
        # 光度增强 (仅针对裁剪后的 160x160)
        self.color_jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)

    def __len__(self):
        return len(self.frames)

    def _get_gt_bbox(self, depth_img):
        """利用 GT 深度图快速提取初始紧密 BBox"""
        y_indices, x_indices = np.where(depth_img > 0)
        if len(x_indices) == 0:
            return 480, 270, 100, 100
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        
        # 💥 同样保护
        w = max(x_max - x_min, 10)
        h = max(y_max - y_min, 10)
        return x_min, y_min, w, h

    def _apply_extreme_photometric_aug(self, rgb_crop):
        """完全对齐方案：追加失焦、噪声、遮挡，以及 💥新增：极端人造高光"""
        import random
        # 1. 运动模糊 (残影) 或 失焦模糊 (高斯模糊)
        if random.random() < 0.4:
            if random.random() < 0.5: # 运动模糊
                k_size = random.choice([5, 7, 9])
                kernel = np.zeros((k_size, k_size))
                kernel[int((k_size-1)/2), :] = np.ones(k_size)
                M_rot = cv2.getRotationMatrix2D((k_size/2, k_size/2), random.uniform(0, 180), 1)
                kernel = cv2.warpAffine(kernel, M_rot, (k_size, k_size))
                kernel = kernel / np.sum(kernel)
                rgb_crop = cv2.filter2D(rgb_crop, -1, kernel)
            else: # 失焦模糊
                k_size = random.choice([3, 5, 7])
                rgb_crop = cv2.GaussianBlur(rgb_crop, (k_size, k_size), 0)

        # 2. 传感器高斯噪声 (模拟低光照 ISO 噪点)
        if random.random() < 0.3:
            mean, std = 0, random.uniform(5, 15)
            noise = np.random.normal(mean, std, rgb_crop.shape).astype(np.float32)
            rgb_crop = np.clip(rgb_crop + noise, 0, 255).astype(np.uint8)

        # 3. 探针/血液遮挡 (Cutout)
        if random.random() < 0.4:
            h, w = rgb_crop.shape[:2]
            cx, cy = random.randint(0, w), random.randint(0, h)
            length, thickness = random.randint(20, 60), random.randint(5, 15)
            angle = random.uniform(0, 180)
            color = (0, 0, random.randint(100, 200)) if random.random() < 0.5 else (180, 180, 180)
            rect = ((cx, cy), (length, thickness), angle)
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.fillPoly(rgb_crop, [box], color)

        # 💥 修复隐患2: 极端人造高光 (模拟唾液强反射，逼迫网络学会局部 3D 脑补)
        if random.random() < 0.3:
            h, w = rgb_crop.shape[:2]
            num_highlights = random.randint(1, 3)
            for _ in range(num_highlights):
                cx, cy = random.randint(0, w), random.randint(0, h)
                ax1, ax2 = random.randint(5, 15), random.randint(2, 6) # 椭圆长短轴
                angle = random.uniform(0, 180)
                # 绘制纯白色高亮斑块
                cv2.ellipse(rgb_crop, (cx, cy), (ax1, ax2), angle, 0, 360, (255, 255, 255), -1)
                
        return rgb_crop

    def __getitem__(self, idx):
        frame = self.frames[idx]
        
        rgb_img = cv2.imread(os.path.join(self.rgb_dir, f"{frame}.png"))
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        depth_gt = np.load(os.path.join(self.depth_dir, f"{frame}.npy")).astype(np.float32)
        
        x_min, y_min, w, h = self._get_gt_bbox(depth_gt)
        c_x, c_y = x_min + w / 2.0, y_min + h / 2.0
        
        if self.is_training:
            dx = np.random.uniform(-0.25, 0.25) * w
            dy = np.random.uniform(-0.25, 0.25) * h
            scale = np.random.uniform(0.8, 1.2)
            angle = np.random.uniform(-30, 30) 
        else:
            dx, dy, scale, angle = 0, 0, 1.0, 0
            
        c_x_new, c_y_new = c_x + dx, c_y + dy
        crop_size = max(w, h) * scale * 1.2 

        M = cv2.getRotationMatrix2D((c_x_new, c_y_new), angle, 160.0 / crop_size)
        M[0, 2] += 80.0 - c_x_new
        M[1, 2] += 80.0 - c_y_new

        rgb_crop = cv2.warpAffine(rgb_img, M, (160, 160), flags=cv2.INTER_LINEAR, borderValue=(0,0,0))
        depth_crop = cv2.warpAffine(depth_gt, M, (160, 160), flags=cv2.INTER_NEAREST, borderValue=0)
        mask_crop = (depth_crop > 0).astype(np.float32)

        if self.is_training:
            import PIL.Image as Image
            rgb_pil = Image.fromarray(rgb_crop)
            rgb_crop = np.array(self.color_jitter(rgb_pil))
            rgb_crop = self._apply_extreme_photometric_aug(rgb_crop)

        rgb_crop = rgb_crop.astype(np.float32) / 255.0

        # 5. K_crop 更新与 Ray Map 映射
        M_3x3 = np.vstack([M, [0, 0, 1]])
        K_crop = M_3x3 @ self.K_base   # 去掉 np.linalg.inv
        K_inv = np.linalg.inv(K_crop)
        
        u, v = np.meshgrid(np.arange(160), np.arange(160))
        uv1 = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3)
        
        # 核心修正：保留 unnorm_rays 用于 Z 还原为 XYZ
        unnorm_rays = (K_inv @ uv1.T).T.reshape(160, 160, 3) 
        # 归一化后的 ray_map 喂给学生网络
        ray_map = unnorm_rays / np.linalg.norm(unnorm_rays, axis=-1, keepdims=True)

        Z_base = self.K_base[0, 0] * (DISTILL_PHYSICAL_WIDTH / crop_size)

        rgb_t = torch.from_numpy(rgb_crop).permute(2, 0, 1)
        ray_t = torch.from_numpy(ray_map).permute(2, 0, 1)
        inputs_6c = torch.cat([rgb_t, ray_t], dim=0) 

        return {
            "inputs_6c": inputs_6c, # RGB + 归一化 RayMap (给学生)
            "rgb_crop": rgb_t,
            "depth_gt": torch.from_numpy(depth_crop).unsqueeze(0),
            "unnorm_rays": torch.from_numpy(unnorm_rays).permute(2, 0, 1).float(), # 用于还原 XYZ
            "mask_gt": torch.from_numpy(mask_crop).unsqueeze(0),
            "Z_base": torch.tensor(Z_base, dtype=torch.float32)
        }