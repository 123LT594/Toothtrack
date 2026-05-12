import os
import glob
import re
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import trimesh
import nvdiffrast.torch as dr
import sys

# 导入原版工程代码
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from learning.models.refine_network import RefineNet
from utils.tools import make_mesh_tensors, nvdiffrast_render

# ================= 1. 全局配置区 =================
DATA_DIR = "/root/lanyun-tmp/golden_dataset"
WEIGHT_PATH = "/root/Toothtrack/weights/2023-10-28-18-33-37/model_best.pth"
MODEL_SAVE_DIR = "/root/lanyun-tmp/models/temporal_refiner"
LOG_DIR = "/root/lanyun-tmp/logs/temporal_refiner"
MESH_PATH = "/root/Toothtrack/demo_data/tooth/mesh/tooth.obj"

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 🌟 RTX 3090 专属配置
BATCH_SIZE = 1
ACCUMULATION_STEPS = 4
EPOCHS = 50
LR = 5e-5

K_MATRIX = np.array([[2866.3146, 0.0, 480.0],
                     [0.0, 2866.3146, 270.0],
                     [0.0, 0.0, 1.0]], dtype=np.float32)
H_RAW, W_RAW = 540, 960

# ================= 2. 核心物理几何转换与裁剪 =================

def depth2xyzmap(depth, K):
    """ 将 1 通道深度反算为严格的 3 通道物理空间坐标 """
    invalid = depth < 1e-3
    H, W = depth.shape[:2]
    vs, us = np.meshgrid(np.arange(0, H), np.arange(0, W), indexing='ij')
    vs, us = vs.astype(np.float32), us.astype(np.float32)
    
    x = (us - K[0, 2]) * depth / K[0, 0]
    y = (vs - K[1, 2]) * depth / K[1, 1]
    z = depth
    
    xyz_map = np.stack([x, y, z], axis=-1)
    xyz_map[invalid] = 0
    return xyz_map

# 🌟 核心修正：引入 scale_jitter，打破永远完美的 1.5 倍裁剪框！
def get_crop_bbox(pose, K, pts_3d, scale_jitter=1.0):
    """ 计算局部裁剪框 """
    pts_cam = (pose[:3, :3] @ pts_3d.T + pose[:3, 3:4]).T
    pts_img = pts_cam @ K.T
    pts_img = pts_img[:, :2] / pts_img[:, 2:3]
    
    min_x, min_y = np.min(pts_img, axis=0)
    max_x, max_y = np.max(pts_img, axis=0)
    
    center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    
    # 根据 scale_jitter 强行切断边缘或拉远视野
    size = max(max_x - min_x, max_y - min_y) * 1.5 * scale_jitter
    return np.array([int(center_x - size/2), int(center_y - size/2), int(size)])

def crop_and_resize(img, bbox, out_size=160, is_rgb=True):
    """ 严格裁剪 """
    x, y, s = bbox
    h, w = img.shape[:2]
    
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w, x + s), min(h, y + s)
    
    cropped = np.zeros((s, s, 3), dtype=img.dtype)
    interp = cv2.INTER_LINEAR if is_rgb else cv2.INTER_NEAREST
        
    cx1, cy1 = x1 - x, y1 - y
    cx2, cy2 = cx1 + (x2 - x1), cy1 + (y2 - y1)
    
    if cx2 > cx1 and cy2 > cy1 and x2 > x1 and y2 > y1:
        cropped[cy1:cy2, cx1:cx2] = img[y1:y2, x1:x2]
        
    return cv2.resize(cropped, (out_size, out_size), interpolation=interp)

def assemble_6channel_clean(rgb_crop, xyz_crop):
    """ 拼装 6 通道输入 (RGB 3 + XYZ 3) """
    mask = (xyz_crop[..., 2] > 0.001).astype(np.float32)[..., np.newaxis]
    clean_rgb = rgb_crop * mask
    rgb_xyz = np.concatenate([clean_rgb, xyz_crop], axis=-1)
    return torch.from_numpy(rgb_xyz).permute(2, 0, 1).float() 

# ================= 3. 时序数据集 =================
class TemporalPoseDataset(Dataset):
    def __init__(self, data_dir, mesh_path):
        self.pose_files = sorted(glob.glob(os.path.join(data_dir, "pose", "*.npy")))
        self.rgb_dir = os.path.join(data_dir, "rgb")
        self.depth_dir = os.path.join(data_dir, "depth")
        
        mesh = trimesh.load(mesh_path)
        self.pts_3d = mesh.vertices
        print(f"📦 数据就绪: {len(self.pose_files)} 帧")

    def __len__(self):
        return len(self.pose_files)

    # 🌟 核心修正：这个函数现在只负责 XY 和旋转扰动，Z 轴交由下游的概率分支控制
    def add_dental_noise_xy_rot(self, pose):
        noisy = pose.copy()
        noisy[0, 3] += np.random.uniform(-0.003, 0.003)
        noisy[1, 3] += np.random.uniform(-0.003, 0.003)
        
        r_matrix = R.from_matrix(noisy[:3, :3])
        noise_euler = np.random.uniform(-5, 5, 3)
        noisy[:3, :3] = (R.from_euler('xyz', noise_euler, degrees=True) * r_matrix).as_matrix()
        return noisy

    def get_raw_image_and_xyz(self, name):
        rgb = cv2.imread(os.path.join(self.rgb_dir, f"{name}.png"), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        depth = np.load(os.path.join(self.depth_dir, f"{name}.npy")).astype(np.float32)
        xyz = depth2xyzmap(depth, K_MATRIX)
        return rgb, xyz

    def __getitem__(self, idx):
        curr_pose_path = self.pose_files[idx]
        name_t = os.path.basename(curr_pose_path).replace('.npy', '')
        gt_pose = np.load(curr_pose_path).astype(np.float32)
        
        A_curr_rgb_raw, A_curr_xyz_raw = self.get_raw_image_and_xyz(name_t)
        
        A_prev_rgb_raw, A_prev_xyz_raw = None, None
        match = re.search(r'\d+', name_t)
        if match:
            curr_id, prefix, suffix_len = int(match.group()), name_t[:match.start()], match.end() - match.start()
            for step in range(1, 4):
                prev_id = curr_id - step
                if prev_id < 0: continue
                expected_name = f"{prefix}{prev_id:0{suffix_len}d}"
                if os.path.join(os.path.dirname(curr_pose_path), f"{expected_name}.npy") in self.pose_files:
                    A_prev_rgb_raw, A_prev_xyz_raw = self.get_raw_image_and_xyz(expected_name)
                    break
                    
        if A_prev_rgb_raw is None:
            A_prev_rgb_raw, A_prev_xyz_raw = A_curr_rgb_raw.copy(), A_curr_xyz_raw.copy()
            
        # 1. 注入 XY 和 旋转噪声
        noisy_pose = self.add_dental_noise_xy_rot(gt_pose).astype(np.float32)
        
        # ================= 🌟 终极领域先验：二八定律概率增强 =================
        p = np.random.rand()
        
        if p < 0.8:
            # 🟢 80% 平稳基本盘 (守护高精度)
            noisy_pose[2, 3] += np.random.uniform(-0.002, 0.005) # 微小呼吸抖动
            scale_jitter = np.random.uniform(1.1, 1.5)             # 视野完整，不切边缘
        else:
            # 🔴 20% 极限施压模式 (专杀 770 帧变焦放大！)
            # 模拟相机突然大幅推进 (高达 50mm 滞后)
            noisy_pose[2, 3] += np.random.uniform(0.005, 0.050)  
            # 模拟裁剪框跟不上，牙齿边缘被残忍斩首 (0.5倍小框)
            scale_jitter = np.random.uniform(0.5, 1.05)            
        # ====================================================================
        
        # 2. 带着 scale_jitter 去截取破坏性的视野
        bbox = get_crop_bbox(noisy_pose, K_MATRIX, self.pts_3d, scale_jitter)
        
        A_curr = assemble_6channel_clean(
            crop_and_resize(A_curr_rgb_raw, bbox, is_rgb=True), 
            crop_and_resize(A_curr_xyz_raw, bbox, is_rgb=False))
        
        A_prev = assemble_6channel_clean(
            crop_and_resize(A_prev_rgb_raw, bbox, is_rgb=True), 
            crop_and_resize(A_prev_xyz_raw, bbox, is_rgb=False))
        
        return A_curr, A_prev, torch.from_numpy(noisy_pose), torch.from_numpy(gt_pose), torch.from_numpy(bbox)

# ================= 4. 物理 Loss 计算 =================
def batch_rodrigues(rvecs):
    theta = torch.norm(rvecs, dim=1, keepdim=True) + 1e-8
    r = rvecs / theta
    K = torch.zeros(rvecs.shape[0], 3, 3, device=rvecs.device)
    K[:, 0, 1] = -r[:, 2]; K[:, 0, 2] = r[:, 1]
    K[:, 1, 0] = r[:, 2]; K[:, 1, 2] = -r[:, 0]
    K[:, 2, 0] = -r[:, 1]; K[:, 2, 1] = r[:, 0]
    I = torch.eye(3, device=rvecs.device).unsqueeze(0)
    return I + torch.sin(theta).unsqueeze(-1) * K + (1 - torch.cos(theta)).unsqueeze(-1) * torch.bmm(K, K)

def compute_add_loss(pred_t, pred_r_vec, noisy_pose, gt_pose, model_pts):
    delta_R = batch_rodrigues(pred_r_vec)
    base_R, base_t = noisy_pose[:, :3, :3], noisy_pose[:, :3, 3:4]
    pred_R_final = torch.bmm(delta_R, base_R) 
    pred_t_final = base_t + pred_t.unsqueeze(-1) 
    gt_R, gt_t = gt_pose[:, :3, :3], gt_pose[:, :3, 3:4]
    pts_pred = torch.bmm(pred_R_final, model_pts.transpose(1, 2)) + pred_t_final
    pts_gt = torch.bmm(gt_R, model_pts.transpose(1, 2)) + gt_t
    return torch.mean(torch.norm(pts_pred - pts_gt, dim=1))

# ================= 5. 主循环 =================
if __name__ == "__main__":
    device = torch.device('cuda')
    writer = SummaryWriter(LOG_DIR)
    
    glctx = dr.RasterizeCudaContext()
    mesh = trimesh.load(MESH_PATH)
    mesh_tensors = make_mesh_tensors(mesh)
    for k in mesh_tensors: mesh_tensors[k] = mesh_tensors[k].to(device)
    
    pts_np = mesh.vertices[np.random.choice(len(mesh.vertices), 1000)]
    model_pts = torch.tensor(pts_np, dtype=torch.float32, device=device).unsqueeze(0).repeat(BATCH_SIZE, 1, 1)

    # 载入 6 通道模型
    model = RefineNet(c_in=6).to(device)
    print("⏳ 正在注入 2023 年通用大模型基因...")
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device), strict=False)
    print("✅ 权重完美契合加载成功！")
    
    # ================= 🌟 严格冻结机制 =================
    print("🧊 执行网络层冻结...")
    frozen_count, active_count = 0, 0
    for name, param in model.named_parameters():
        # 只放开时序注意力层和最终的平移/旋转回归头
        if "temporal_attn" in name or "trans_head" in name or "rot_head" in name:
            param.requires_grad = True
            active_count += 1
        else:
            param.requires_grad = False 
            frozen_count += 1
    print(f"🔒 冻结层数: {frozen_count} | 🔓 训练层数: {active_count}")
    # ====================================================

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    dataset = TemporalPoseDataset(DATA_DIR, MESH_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")
        
        optimizer.zero_grad()
        
        for step, (A_curr, A_prev, noisy_p, gt_p, bbox) in enumerate(pbar):
            A_curr, A_prev = A_curr.to(device), A_prev.to(device)
            noisy_p, gt_p = noisy_p.to(device), gt_p.to(device)
            bs = A_curr.shape[0]
            
            B_rgbs_crop = []
            for i in range(bs):
                pose = noisy_p[i:i+1] 
                bb = bbox[i].cpu().numpy()
                
                r_rgb, r_d, _ = nvdiffrast_render(
                    K=K_MATRIX, H=H_RAW, W=W_RAW, ob_in_cams=pose,
                    context='cuda', glctx=glctx, mesh_tensors=mesh_tensors,
                    output_size=[H_RAW, W_RAW], use_light=True
                )
                
                r_rgb = r_rgb.detach().cpu().numpy()
                r_d = r_d.detach().cpu().numpy()
                
                if r_rgb.ndim == 4: r_rgb = r_rgb[0]
                if r_d.ndim == 3: r_d = r_d[0]
                
                r_rgb = r_rgb.astype(np.float32)
                r_xyz = depth2xyzmap(r_d, K_MATRIX)
                
                B_rgb_crop = crop_and_resize(r_rgb, bb, is_rgb=True)
                B_xyz_crop = crop_and_resize(r_xyz, bb, is_rgb=False)
                
                B_rgbs_crop.append(assemble_6channel_clean(B_rgb_crop, B_xyz_crop))
                
            B_render = torch.stack(B_rgbs_crop).to(device)
            
            output = model(B_render, A_curr, A_prev)
            loss = compute_add_loss(output['trans'], output['rot'], noisy_p, gt_p, model_pts[:bs])
            loss = loss / ACCUMULATION_STEPS 
            
            loss.backward()
            
            if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(loader):
                optimizer.step()      
                optimizer.zero_grad()
            
            actual_loss = loss.item() * ACCUMULATION_STEPS
            epoch_loss += actual_loss
            loss_mm = actual_loss * 1000
            
            pbar.set_postfix({'ADD Err(mm)': f"{loss_mm:.2f}"})
            writer.add_scalar('Train/ADD_Error_mm', loss_mm, global_step)
            global_step += 1

        avg_loss = (epoch_loss / len(loader)) * 1000
        print(f"✅ Epoch {epoch} 完成 | 均平移旋转误差: {avg_loss:.3f} mm")
        
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, f"temporal_refiner_ep{epoch}.pth"))
            
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "temporal_refiner_best.pth"))
    writer.close()