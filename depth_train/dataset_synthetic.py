import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
import random

# 强制使用 OSMesa 离屏渲染，防止 DataLoader 多进程死锁
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import pyrender
import trimesh

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from learning.training.training_config import DISTILL_K_BASE, DISTILL_PHYSICAL_WIDTH

class SyntheticPretrainDataset(Dataset):
    def __init__(self, data_dir=None, is_training=True):
        super().__init__()
        if data_dir is None:
            _current_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.abspath(os.path.join(_current_dir, "../demo_data/synthetic"))
        self.bg_dir = os.path.join(data_dir, "background")
        self.mesh_dir = os.path.join(data_dir, "mesh")
        self.is_training = is_training
        
        if not os.path.exists(self.bg_dir):
            raise FileNotFoundError(f"找不到背景图目录，请检查路径: {self.bg_dir}")
        if not os.path.exists(self.mesh_dir):
            raise FileNotFoundError(f"找不到模型目录，请检查路径: {self.mesh_dir}")
        
        self.K_base = np.array(DISTILL_K_BASE, dtype=np.float32)
        self.color_jitter = T.ColorJitter(brightness=0.5, hue=0.1)

        # 1. 加载背景图
        self.bgs = []
        for bg_file in os.listdir(self.bg_dir):
            if bg_file.endswith('.png'):
                bg_path = os.path.join(self.bg_dir, bg_file)
                bg_img = cv2.imread(bg_path)
                bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
                bg_img = cv2.resize(bg_img, (960, 540))
                self.bgs.append(bg_img)
                
        # 2. 加载 obj 模型数据
        self.meshes = []
        for mesh_file in os.listdir(self.mesh_dir):
            if mesh_file.endswith('.obj'):
                mesh_path = os.path.join(self.mesh_dir, mesh_file)
                tm = trimesh.load(mesh_path, process=True)
                
                tm.vertices -= tm.bounding_box.centroid
                max_extent = max(tm.extents)
                tm.vertices *= (DISTILL_PHYSICAL_WIDTH / max_extent)
                
                material = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.05, 
                    roughnessFactor=0.6, 
                    baseColorFactor=(0.75, 0.70, 0.65, 1.0)
                )
                self.meshes.append(pyrender.Mesh.from_trimesh(tm, smooth=True, material=material))
                
        print(f"✅ 成功加载 {len(self.bgs)} 张背景图与 {len(self.meshes)} 个 3D 模型。")
        
        # 💥 核心修改：主进程绝对不碰 PyRender 实例化！全部设为 None。
        self.renderer = None
        self.scenes = []
        self.camera_nodes = []
        self.light_nodes = []
        print("⏳ 图形渲染引擎已配置为在多进程 Worker 中进行懒加载保护...")

    def _init_pyrender(self):
        """💥 新增：在子进程中独立创建渲染环境，与主进程彻底物理隔离！"""
        safe_margin = DISTILL_PHYSICAL_WIDTH
        camera = pyrender.IntrinsicsCamera(
            fx=self.K_base[0,0], fy=self.K_base[1,1], 
            cx=self.K_base[0,2], cy=self.K_base[1,2], 
            znear=0.01 * safe_margin, zfar=1000.0 * safe_margin
        )
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 0.95], intensity=1.5)

        for mesh in self.meshes:
            scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.4, 0.4, 0.4])
            scene.add(mesh, pose=np.eye(4)) # 牙齿定死在原点
            
            cam_node = scene.add(camera, pose=np.eye(4))
            light_node = scene.add(light, pose=np.eye(4))
            
            self.scenes.append(scene)
            self.camera_nodes.append(cam_node)
            self.light_nodes.append(light_node)
            
        self.renderer = pyrender.OffscreenRenderer(960, 540)

    def __len__(self):
        return 10000 if self.is_training else 1000

    def _render_random_mesh(self):
        # 💥 懒加载：如果当前进程还没有渲染器，立刻初始化
        if self.renderer is None:
            self._init_pyrender()
            
        idx = random.randint(0, len(self.scenes) - 1)
        scene = self.scenes[idx]
        cam_node = self.camera_nodes[idx]
        light_node = self.light_nodes[idx]

        target_pixel_width = random.uniform(150.0, 400.0)
        z_val = self.K_base[0, 0] * (DISTILL_PHYSICAL_WIDTH / target_pixel_width)
        
        mesh_pose = np.eye(4)
        mesh_pose[2, 3] = -z_val 
        
        base_rot = cv2.Rodrigues(np.array([0.0, 0.0, 0.0]))[0] 
        rot_x = cv2.Rodrigues(np.array([random.uniform(-0.4, 0.4), 0, 0]))[0]
        rot_y = cv2.Rodrigues(np.array([0, random.uniform(-0.4, 0.4), 0]))[0]
        rot_z = cv2.Rodrigues(np.array([0, 0, random.uniform(-np.pi, np.pi)]))[0]
        mesh_pose[:3, :3] = rot_z @ rot_y @ rot_x @ base_rot
        
        # 逆矩阵法则移动相机
        cam_pose = np.linalg.inv(mesh_pose)
        scene.set_pose(cam_node, pose=cam_pose)
        
        tilt_rot = np.eye(4)
        tilt_rot[:3, :3] = cv2.Rodrigues(np.array([-0.3, 0.3, 0]))[0] 
        light_pose = cam_pose @ tilt_rot
        scene.set_pose(light_node, pose=light_pose)
        
        color, depth = self.renderer.render(scene)
        return color, depth

    def __getitem__(self, idx):
        render_rgb, depth_gt = self._render_random_mesh()
        mask_gt = (depth_gt > 0).astype(np.float32)
        
        bg_img = random.choice(self.bgs).copy()
        
        composite_rgb = bg_img.copy()
        valid_idx = mask_gt > 0
        composite_rgb[valid_idx] = render_rgb[valid_idx]
        
        y_indices, x_indices = np.where(mask_gt > 0)
        if len(x_indices) == 0:
            x_min, y_min, w, h = 480, 270, 100, 100
        else:
            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()
            w = max(x_max - x_min, 10)
            h = max(y_max - y_min, 10)
            
        c_x, c_y = x_min + w / 2.0, y_min + h / 2.0
        
        if self.is_training:
            dx = np.random.uniform(-0.2, 0.2) * w
            dy = np.random.uniform(-0.2, 0.2) * h
            scale = np.random.uniform(0.8, 1.2)
            angle = np.random.uniform(-30, 30) 
        else:
            dx, dy, scale, angle = 0, 0, 1.0, 0
            
        c_x_new, c_y_new = c_x + dx, c_y + dy
        crop_size = max(w, h) * scale * 1.2 

        M = cv2.getRotationMatrix2D((c_x_new, c_y_new), angle, 160.0 / crop_size)
        M[0, 2] += 80.0 - c_x_new
        M[1, 2] += 80.0 - c_y_new

        rgb_crop = cv2.warpAffine(composite_rgb, M, (160, 160), flags=cv2.INTER_LINEAR, borderValue=(0,0,0))
        depth_crop = cv2.warpAffine(depth_gt, M, (160, 160), flags=cv2.INTER_NEAREST, borderValue=0)
        mask_crop = (depth_crop > 0).astype(np.float32)

        if self.is_training:
            import PIL.Image as Image
            rgb_pil = Image.fromarray(rgb_crop)
            rgb_crop = np.array(self.color_jitter(rgb_pil))

        rgb_crop = rgb_crop.astype(np.float32) / 255.0

        M_3x3 = np.vstack([M, [0, 0, 1]])
        # K_crop = np.linalg.inv(M_3x3) @ self.K_base
        # 💥 修复：直接相乘！绝对不能用逆矩阵！
        K_crop = M_3x3 @ self.K_base
        K_inv = np.linalg.inv(K_crop)
        
        u, v = np.meshgrid(np.arange(160), np.arange(160))
        uv1 = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3)
        
        unnorm_rays = (K_inv @ uv1.T).T.reshape(160, 160, 3) 
        ray_map = unnorm_rays / np.linalg.norm(unnorm_rays, axis=-1, keepdims=True)

        Z_base = self.K_base[0, 0] * (DISTILL_PHYSICAL_WIDTH / max(w, h))

        rgb_t = torch.from_numpy(rgb_crop).permute(2, 0, 1).float()
        ray_t = torch.from_numpy(ray_map).permute(2, 0, 1).float()
        inputs_6c = torch.cat([rgb_t, ray_t], dim=0) 

        # 💥 暴君式释放内存：赶在返回给主进程之前，强制销毁局部高清大图废料！
        del render_rgb, depth_gt, bg_img, composite_rgb

        return {
            "inputs_6c": inputs_6c, 
            "rgb_crop": rgb_t,
            "depth_gt": torch.from_numpy(depth_crop).unsqueeze(0),
            "mask_gt": torch.from_numpy(mask_crop).unsqueeze(0),
            "Z_base": torch.tensor(Z_base, dtype=torch.float32)
        }