import os
# 强制使用纯CPU或EGL离线渲染平台（配合 xvfb 运行）
import cv2
import numpy as np
import trimesh
import pyrender
import pickle
from tqdm import tqdm

# ================= 配置区 =================
OBJ_PATH = "/root/Toothtrack/demo_data/ztooth/mesh/tooth.obj"          
CAM_K_PATH = "/root/Toothtrack/demo_data/ztooth/cam_K.txt"   
OUTPUT_PKL = "/root/Toothtrack/demo_data/ztooth/zero_shot_db.pkl" 
DEBUG_DIR = "/root/lanyun-tmp/prerender"      

TARGET_SIZE = 160       
CROWN_DIR = np.array([0.0, 0.0, 1.0])  # 牙冠朝向 Z 轴
MAX_ANGLE = 60          
NUM_VIEWS = 252         
# ==========================================


class MedicalGeometryFeatureExtractor:
    """法向与轮廓特征提取器"""
    def __init__(self, target_size=160, grid_size=16):
        self.target_size = target_size
        self.grid_size = grid_size
        self.pool_kernel = target_size // grid_size  # 160 / 16 = 10

    def extract_fourier_descriptor(self, mask, num_harmonics=15):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return np.zeros(num_harmonics, dtype=np.float32)
        contour = max(contours, key=cv2.contourArea).squeeze()
        if len(contour.shape) < 2 or len(contour) < num_harmonics:
            return np.zeros(num_harmonics, dtype=np.float32)

        contour_complex = np.empty(contour.shape[0], dtype=complex)
        contour_complex.real = contour[:, 0]
        contour_complex.imag = contour[:, 1]

        fourier_result = np.fft.fft(contour_complex)
        descriptors = np.abs(fourier_result[1:num_harmonics+1])
        descriptors = descriptors / (descriptors[0] + 1e-6)
        return descriptors.astype(np.float32)

    def extract_grid_normal_feature(self, depth, mask_bin):
        depth_smoothed = cv2.GaussianBlur(depth.astype(np.float32), (3, 3), 0)
        dzdx = cv2.Sobel(depth_smoothed, cv2.CV_32F, 1, 0, ksize=3)
        dzdy = cv2.Sobel(depth_smoothed, cv2.CV_32F, 0, 1, ksize=3)
        
        normal = np.dstack((-dzdx, -dzdy, np.ones_like(depth_smoothed)))
        norm = np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-6
        normal = normal / norm

        k = self.pool_kernel
        normal_grid = normal.reshape(self.grid_size, k, self.grid_size, k, 3)
        mask_grid = mask_bin.reshape(self.grid_size, k, self.grid_size, k)

        grid_mask_ratio = mask_grid.mean(axis=(1, 3))
        grid_normal_sum = (normal_grid * mask_grid[..., None]).sum(axis=(1, 3))
        grid_normal_count = mask_grid.sum(axis=(1, 3))[..., None] + 1e-6
        grid_normal_mean = grid_normal_sum / grid_normal_count
        
        grid_norm = np.linalg.norm(grid_normal_mean, axis=-1, keepdims=True) + 1e-6
        grid_normal_mean = grid_normal_mean / grid_norm

        return grid_normal_mean.astype(np.float32), grid_mask_ratio.astype(np.float32)


def get_crown_oriented_opencv_poses(radius, crown_dir, max_angle_deg, num_points):
    """
    生成 OpenCV 坐标系下的牙冠朝向位姿，并强制包含 360 度平面内自转 (Roll) 遍历！
    彻底解决内窥镜旋转导致的网格特征错位问题。
    """
    poses = []
    cos_thresh = np.cos(np.radians(max_angle_deg))
    crown_dir = crown_dir / np.linalg.norm(crown_dir)

    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_points)
    theta = np.pi * (1 + 5**0.5) * indices

    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    points = np.stack([x, y, z], axis=1)

    for p in points:
        if np.dot(p, crown_dir) < cos_thresh:
            continue
            
        cam_pos = p * radius
        
        # 基础坐标系构建
        forward = -cam_pos / np.linalg.norm(cam_pos) 
        world_up = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(forward, world_up)) > 0.99:
            world_up = np.array([0.0, 0.0, 1.0])
            
        right_base = np.cross(world_up, forward)
        right_base /= np.linalg.norm(right_base)
        down_base = np.cross(forward, right_base)
        down_base /= np.linalg.norm(down_base)
        
        # ==========================================================
        # 🌟 核心新增：自转角度遍历 (In-Plane Rotation)
        # 每 30 度生成一个自转模板，彻底覆盖所有手腕扭转情况
        # ==========================================================
        for roll_deg in range(0, 360, 30):
            roll_rad = np.radians(roll_deg)
            cos_r, sin_r = np.cos(roll_rad), np.sin(roll_rad)
            
            # 绕着 forward (相机Z轴) 旋转 right 和 down 轴
            right = cos_r * right_base + sin_r * down_base
            down = -sin_r * right_base + cos_r * down_base
            
            pose = np.eye(4)
            pose[:3, 0] = right
            pose[:3, 1] = down
            pose[:3, 2] = forward
            pose[:3, 3] = cam_pos
            poses.append(pose)
            
    return poses

def load_intrinsics(txt_path):
    try:
        K = np.loadtxt(txt_path)
        return K
    except:
        print(f"未能加载 {txt_path}，使用默认内参")
        return np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])

def crop_and_resize(image, mask, target_size=160):
    """等比例正方形裁剪与缩放，彻底杜绝牙齿边缘被切断或拉伸变形"""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
        
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    
    # 计算宽高
    h = y_max - y_min
    w = x_max - x_min
    
    # 取最大边长，构建正方形 BBox，防止局部被切
    max_len = max(h, w)
    center_y = (y_min + y_max) // 2
    center_x = (x_min + x_max) // 2
    
    half_len = int(max_len * 0.6) # 留出 20% 的安全边距
    
    img_h, img_w = image.shape[:2]
    ymin = max(0, center_y - half_len)
    ymax = min(img_h, center_y + half_len)
    xmin = max(0, center_x - half_len)
    xmax = min(img_w, center_x + half_len)
    
    cropped = image[ymin:ymax, xmin:xmax]
    
    # 统一 Resize 到 160x160
    resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    return resized

def main():
    os.makedirs(DEBUG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_PKL), exist_ok=True)

    K = load_intrinsics(CAM_K_PATH)
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    global_w, global_h = 640, 480 

    # 加载网格并自动中心化
    mesh = trimesh.load(OBJ_PATH)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    vertices = mesh.vertices
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
    mesh.vertices -= center  

    max_body_diameter = np.linalg.norm(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0))
    sphere_radius = float(max_body_diameter * 6.0)
    print(f"-> 模型已居中。最大直径: {max_body_diameter:.4f}, 渲染半径: {sphere_radius:.4f}")

    scene = pyrender.Scene(bg_color=[0, 0, 0, 0])
    scene.add(pyrender.Mesh.from_trimesh(mesh))
    
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.001, zfar=10.0)
    cam_node = scene.add(camera, pose=np.eye(4))
    r = pyrender.OffscreenRenderer(viewport_width=global_w, viewport_height=global_h)

    poses_opencv = get_crown_oriented_opencv_poses(sphere_radius, CROWN_DIR, MAX_ANGLE, NUM_VIEWS)
    print(f"已生成 {len(poses_opencv)} 个有效视角。")

    extractor = MedicalGeometryFeatureExtractor(target_size=TARGET_SIZE, grid_size=16)
    zero_shot_db = []

    # OpenCV 转换到 Pyrender(OpenGL) 的矩阵
    cv_to_gl = np.diag([1, -1, -1, 1])

    print("开始渲染与特征入库...")
    for i, pose_cv in enumerate(tqdm(poses_opencv)):
        # 转换成 OpenGL 姿态给 Pyrender 渲染用
        pose_gl = pose_cv @ cv_to_gl
        scene.set_pose(cam_node, pose=pose_gl)
        
        color, depth = r.render(scene)
        
        mask = (depth > 0).astype(np.uint8)
        if np.sum(mask) < 10:
            continue
            
        depth_160 = crop_and_resize(depth, mask, target_size=TARGET_SIZE)
        mask_160 = crop_and_resize(mask, mask, target_size=TARGET_SIZE)
        
        if depth_160 is None:
            continue
            
        mask_255 = mask_160 * 255
        cv2.imwrite(os.path.join(DEBUG_DIR, f"view_{i:04d}_mask.png"), mask_255)
        depth_vis = depth_160.copy()
        valid_depths = depth_vis[depth_vis > 0]
        if len(valid_depths) > 0:
            d_min, d_max = valid_depths.min(), valid_depths.max()
            # 归一化有效深度到 50~255 之间，背景保持 0
            depth_vis[depth_vis > 0] = ((depth_vis[depth_vis > 0] - d_min) / (d_max - d_min + 1e-6)) * 205 + 50
        depth_vis = depth_vis.astype(np.uint8)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"view_{i:04d}_depth.png"), depth_vis)

        fourier = extractor.extract_fourier_descriptor(mask_255)
        grid_normal, grid_mask = extractor.extract_grid_normal_feature(depth_160, mask_160)

        # 存入数据库：保存原汁原味的 OpenCV 位姿矩阵（给 Refiner 用）
        zero_shot_db.append({
            'pose': pose_cv.astype(np.float32),   
            'fourier': fourier,
            'grid_normal': grid_normal,
            'grid_mask': grid_mask,
            'mask_160': mask_160
        })

    final_dict = {
        "zero_shot_db": zero_shot_db,
        "metadata": {
            "obj_path": OBJ_PATH,
            "target_size": TARGET_SIZE,
            "num_views_saved": len(zero_shot_db)
        }
    }
    
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(final_dict, f)

    r.delete()
    print(f"\n全部完成！成功保存 {len(zero_shot_db)} 个模板至: {OUTPUT_PKL}")
    print(f"检查示例 Mask 和 Depth: {DEBUG_DIR}")

if __name__ == "__main__":
    main()