import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
import random
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class DualDistillDataset(Dataset):
    def __init__(self, data_dir=None, is_training=True, dataset_id=0):
        super().__init__()
        if data_dir is None:
            _current_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.abspath(os.path.join(_current_dir, "../../lanyun-tmp/golden_dataset"))
        self.rgb_dir = os.path.join(data_dir, "rgb")
        self.depth_dir = os.path.join(data_dir, "depth")
        self.pose_dir = os.path.join(data_dir, "pose")
        self.frames = [f.split('.')[0] for f in os.listdir(self.pose_dir) if f.endswith('.npy')]
        self.is_training = is_training
        self.data_dir = data_dir
        self.dataset_id = dataset_id  # 0=旧数据集(2800长焦), 1=新数据集(500短焦)
        
        k_path = os.path.join(self.data_dir, "cam_K.txt")
        if not os.path.exists(k_path):
            raise FileNotFoundError(f"找不到专属内参文件: {k_path}")
        self.K_base = np.loadtxt(k_path, dtype=np.float32)
        
        self.color_jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)
        import trimesh
        mesh_path = os.path.join(self.data_dir, "mesh", "tooth.obj")
        mesh = trimesh.load(mesh_path, process=True)
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        self.dynamic_physical_width = max(extents)

    def __len__(self):
        return len(self.frames)

    def _get_gt_bbox(self, depth_img):
        y_indices, x_indices = np.where(depth_img > 0)
        if len(x_indices) == 0:
            return 480, 270, 100, 100
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        
        w = max(x_max - x_min, 10)
        h = max(y_max - y_min, 10)
        return x_min, y_min, w, h

    def _apply_extreme_photometric_aug(self, rgb_crop):
        import random
        if random.random() < 0.4:
            if random.random() < 0.5: 
                k_size = random.choice([5, 7, 9])
                kernel = np.zeros((k_size, k_size))
                kernel[int((k_size-1)/2), :] = np.ones(k_size)
                M_rot = cv2.getRotationMatrix2D((k_size/2, k_size/2), random.uniform(0, 180), 1)
                kernel = cv2.warpAffine(kernel, M_rot, (k_size, k_size))
                kernel = kernel / np.sum(kernel)
                rgb_crop = cv2.filter2D(rgb_crop, -1, kernel)
            else: 
                k_size = random.choice([3, 5, 7])
                rgb_crop = cv2.GaussianBlur(rgb_crop, (k_size, k_size), 0)
        if random.random() < 0.5: 
            mean, std = 0, random.uniform(10, 30) 
            noise = np.random.normal(mean, std, rgb_crop.shape).astype(np.float32)
            rgb_crop = np.clip(rgb_crop.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if random.random() < 0.4:
            h, w = rgb_crop.shape[:2]
            cx, cy = random.randint(0, w), random.randint(0, h)
            length, thickness = random.randint(20, 60), random.randint(5, 15)
            angle = random.uniform(0, 180)
            color = (0, 0, random.randint(100, 200)) if random.random() < 0.5 else (180, 180, 180)
            rect = ((cx, cy), (length, thickness), angle)
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.fillPoly(rgb_crop, [box], color)
        if random.random() < 0.4: 
            h, w = rgb_crop.shape[:2]
            num_highlights = random.randint(1, 4)
            bloom_layer = np.zeros_like(rgb_crop, dtype=np.float32)
            for _ in range(num_highlights):
                cx, cy = random.randint(0, w), random.randint(0, h)
                ax1, ax2 = random.randint(8, 25), random.randint(4, 10) 
                angle = random.uniform(0, 180)
                cv2.ellipse(bloom_layer, (cx, cy), (ax1, ax2), angle, 0, 360, (255, 255, 255), -1)
            bloom_layer = cv2.GaussianBlur(bloom_layer, (15, 15), 5)
            rgb_crop = np.clip(rgb_crop.astype(np.float32) + bloom_layer, 0, 255).astype(np.uint8)
                
        return rgb_crop

    def __getitem__(self, idx):
        frame = self.frames[idx]
        rgb_img = cv2.imread(os.path.join(self.rgb_dir, f"{frame}.png"))
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        depth_gt = np.load(os.path.join(self.depth_dir, f"{frame}.npy")).astype(np.float32)
        
        # 🌟 加载对应的 GT 位姿矩阵，用于角度补偿
        pose_gt = np.load(os.path.join(self.pose_dir, f"{frame}.npy")).astype(np.float32)
        # pose_gt[2, 2] 是相机光轴与物体 Z 轴的点积，绝对值即为倾斜角的 cos(theta)
        cos_theta = max(abs(pose_gt[2, 2]), 0.5)
        
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
        M_3x3 = np.vstack([M, [0, 0, 1]])
        K_crop = M_3x3 @ self.K_base   
        K_inv = np.linalg.inv(K_crop)
        
        u, v = np.meshgrid(np.arange(160), np.arange(160))
        uv1 = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3)
        
        unnorm_rays = (K_inv @ uv1.T).T.reshape(160, 160, 3) 
        ray_map = unnorm_rays / np.linalg.norm(unnorm_rays, axis=-1, keepdims=True)
        valid_depths = depth_gt[depth_gt > 0]
        z_val = np.median(valid_depths) if len(valid_depths) > 0 else 0.1
        # =========================================================
        # 🌟 形状-尺度解耦：Z_base = GT深度中位数 × 随机噪声(±15%)
        # 废除公式法的固定系统偏差，两个数据集统一围绕各自GT真值波动
        # =========================================================
        valid_depth_crop = depth_crop[depth_crop > 0]
        gt_z_median = np.median(valid_depth_crop) if len(valid_depth_crop) > 0 else 0.1
        noise_scale = random.uniform(0.85, 1.15)
        Z_base = float(gt_z_median * noise_scale)
        Z_base_np = np.array(Z_base, dtype=np.float32)
        rgb_t = torch.from_numpy(rgb_crop).permute(2, 0, 1)
        ray_t = torch.from_numpy(ray_map).permute(2, 0, 1)
        # 🌟 Z_base作为第7通道输入：归一化到0~1（除以0.2m），让网络能"看到"基准深度
        z_base_norm = torch.full((1, rgb_t.shape[1], rgb_t.shape[2]), Z_base / 0.2, dtype=torch.float32)
        inputs_6c = torch.cat([rgb_t, ray_t, z_base_norm], dim=0)  # 实际7通道：RGB(3)+Ray(3)+Z_base(1)
        return {
            "inputs_6c": inputs_6c,
            "rgb_crop": rgb_t,
            "depth_gt": torch.from_numpy(depth_crop).unsqueeze(0),
            "unnorm_rays": torch.from_numpy(unnorm_rays).permute(2, 0, 1).float(),
            "mask_gt": torch.from_numpy(mask_crop).unsqueeze(0),
            "Z_base": torch.tensor(Z_base, dtype=torch.float32),
            "mesh_width": torch.tensor(self.dynamic_physical_width, dtype=torch.float32),
            "dataset_id": torch.tensor(self.dataset_id, dtype=torch.long)
        }
