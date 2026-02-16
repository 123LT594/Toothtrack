"""
高精度深度图生成脚本
特性：
1. 双重 PnP 策略: EPNP 初始化 + Iterative LM 优化
2. 严格的误差过滤: 只有重投影误差 < 3.0 像素的帧才会被保存
"""

import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa' # 强制 CPU
import json
import cv2
import numpy as np
import trimesh
import pyrender
from tqdm import tqdm
import logging

# 获取项目根目录（prepare_gt_depth 的父目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # prepare_gt_depth 的父目录才是项目根

# ================= 配置区域 =================
DATA_DIR = os.path.join(PROJECT_ROOT, "demo_data/tooth_gt")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "demo_data/tooth_gt/gt_depth")  # 输出到 gt_depth 目录
ERROR_THRESHOLD = 3.0   # 容忍的最大重投影误差 (像素)
                       
# ===========================================

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_opencv_to_opengl(matrix):
    # 修正后的坐标转换逻辑
    cv_to_gl = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]], dtype=np.float64)
    c2w_cv = np.linalg.inv(matrix)
    matrix_gl = c2w_cv @ cv_to_gl 
    return matrix_gl

class DepthRenderer:
    def __init__(self, mesh_path, K, h=1080, w=1920):
        self.scene = pyrender.Scene(bg_color=[0, 0, 0])
        self.mesh = trimesh.load(mesh_path)
        self.pyrender_mesh = pyrender.Mesh.from_trimesh(self.mesh, smooth=False)
        self.mesh_node = self.scene.add(self.pyrender_mesh, pose=np.eye(4))
        
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        self.camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.001, zfar=2.0)
        self.cam_node = self.scene.add(self.camera, pose=np.eye(4))
        
        # 强制 EGL 或 OSMesa
        if os.environ.get('PYOPENGL_PLATFORM') is None:
             os.environ['PYOPENGL_PLATFORM'] = 'egl' # 尝试 GPU

        try:
            self.renderer = pyrender.OffscreenRenderer(w, h)
        except Exception as e:
            logger.warning(f"EGL 初始化失败，尝试 OSMesa (CPU): {e}")
            os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
            self.renderer = pyrender.OffscreenRenderer(w, h)

    def render(self, rvec, tvec):
        # 构建 OpenCV 4x4 矩阵
        matrix_cv = np.eye(4)
        R, _ = cv2.Rodrigues(rvec)
        matrix_cv[:3, :3] = R
        matrix_cv[:3, 3] = tvec.flatten()
        
        # 转换并应用 Pose
        pose_gl = convert_opencv_to_opengl(matrix_cv)
        self.scene.set_pose(self.cam_node, pose_gl)
        
        # 渲染
        depth = self.renderer.render(self.scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
        return depth

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 加载数据
    logger.info("正在加载数据...")
    K = np.loadtxt(f"{DATA_DIR}/cam_K.txt", dtype=np.float64)
    with open(f"{DATA_DIR}/annotations.json", 'r') as f:
        annotations = json.load(f)['annotations']
    with open(f"{DATA_DIR}/keypoints_3d_model.json", 'r') as f:
        kpts_3d = np.array(json.load(f)['keypoints_3d_original'], dtype=np.float64)
    
    # 初始化渲染器
    renderer = DepthRenderer(f"{DATA_DIR}/mesh/tooth.obj", K)
    
    total_frames = len(annotations)
    success_count = 0
    discard_count = 0
    
    logger.info(f"开始处理 {total_frames} 帧... 阈值: {ERROR_THRESHOLD} px")
    
    for frame_name, data in tqdm(annotations.items()):
        # 提取 2D 点
        pts_2d = []
        try:
            for i in range(1, 5):
                pts_2d.append(data[f'ball_{i}'])
            pts_2d = np.array(pts_2d, dtype=np.float64)
        except KeyError:
            continue # 标注不全
            
        # ==========================================
        # 核心算法升级：两步法 PnP
        # ==========================================
        
        # 步骤 1: 使用 EPNP 进行初始化 (稳定，不依赖初值)
        ret_init, rvec_init, tvec_init = cv2.solvePnP(
            kpts_3d, pts_2d, K, None, 
            flags=cv2.SOLVEPNP_EPNP
        )
        
        if not ret_init:
            continue
            
        # 步骤 2: 使用 ITERATIVE 进行非线性优化 (Refinement)
        # 这一步会微调位姿，使重投影误差最小化
        ret_final, rvec_final, tvec_final = cv2.solvePnP(
            kpts_3d, pts_2d, K, None, 
            flags=cv2.SOLVEPNP_ITERATIVE,
            useExtrinsicGuess=True, # 关键：使用上一步的结果作为初值
            rvec=rvec_init,
            tvec=tvec_init
        )
        
        if not ret_final:
            continue

        # ==========================================
        # 质量控制：计算重投影误差
        # ==========================================
        proj_pts, _ = cv2.projectPoints(kpts_3d, rvec_final, tvec_final, K, None)
        proj_pts = proj_pts.squeeze()
        
        # 计算 4 个点与 GT 2D 点的欧氏距离平均值
        errors = np.linalg.norm(pts_2d - proj_pts, axis=1)
        mean_error = np.mean(errors)
        
        # 判定
        if mean_error > ERROR_THRESHOLD:
            discard_count += 1
            # logger.warning(f"丢弃 {frame_name}: 误差 {mean_error:.2f} px > 阈值")
            continue
            
        # ==========================================
        # 渲染与保存
        # ==========================================
        depth_map = renderer.render(rvec_final, tvec_final)
        
        save_path = os.path.join(OUTPUT_DIR, frame_name.replace('.png', '.npy').replace('.jpg', '.npy'))
        np.save(save_path, depth_map)
        success_count += 1

    logger.info("="*30)
    logger.info(f"处理完成")
    logger.info(f"保留帧数 (Success): {success_count}")
    logger.info(f"丢弃帧数 (Discard): {discard_count}")
    logger.info(f"保留率: {success_count/total_frames*100:.1f}%")
    logger.info(f"结果保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()