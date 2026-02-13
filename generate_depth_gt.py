"""
全自动深度图 Ground Truth 生成脚本

功能：
1. 读取3D模型、相机内参、3D关键点坐标和2D标注
2. 通过PnP解算每帧的相机位姿
3. 利用PyRender渲染出高精度深度图
4. 保存为.npy格式

依赖：
    pip install numpy opencv-python trimesh pyrender scipy

作者：CV/Graphics Expert
日期：2026-01-27
"""

import os
import sys
import json
import glob
import logging
import warnings
import numpy as np
import cv2
import trimesh
import pyrender
from scipy.spatial.transform import Rotation as R

# ============== 配置参数 ==============
class Config:
    # 数据目录
    DATA_DIR = "./demo_data/tooth_gt"
    MESH_DIR = os.path.join(DATA_DIR, "mesh")
    
    # 输入文件路径
    TOOTH_OBJ_PATH = os.path.join(MESH_DIR, "tooth.obj")
    CAM_K_PATH = os.path.join(DATA_DIR, "cam_K.txt")
    ANNOTATIONS_PATH = os.path.join(DATA_DIR, "annotations.json")
    KEYPOINTS_3D_PATH = os.path.join(DATA_DIR, "keypoints_3d_model.json")
    
    # 输出目录
    OUTPUT_DIR = "./output_depth"
    
    # 渲染参数
    Z_NEAR = 0.001      # 近裁剪面（米）
    Z_FAR = 2.0         # 远裁剪面（米）
    RENDER_FLAGS = pyrender.RenderFlags.DEPTH_ONLY  # 只渲染深度
    
    # PnP参数
    PNP_METHOD = cv2.SOLVEPNP_ITERATIVE
    
    # 日志级别
    LOG_LEVEL = logging.INFO


# ============== 日志配置 ==============
logging.basicConfig(
    level=Config.LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============== 工具函数 ==============
def load_camera_intrinsics(filepath):
    """
    加载相机内参矩阵
    
    Args:
        filepath: cam_K.txt 文件路径
        
    Returns:
        K: 3x3 相机内参矩阵 (numpy array)
    """
    logger.info(f"加载相机内参: {filepath}")
    
    K = np.loadtxt(filepath, dtype=np.float64)
    
    if K.shape != (3, 3):
        raise ValueError(f"相机内参矩阵形状错误，期望 (3, 3)，实际 {K.shape}")
    
    logger.info(f"相机内参矩阵:\n{K}")
    logger.info(f"  fx={K[0,0]:.2f}, fy={K[1,1]:.2f}, cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")
    
    return K


def load_annotations(filepath):
    """
    加载2D标注文件
    
    Args:
        filepath: annotations.json 文件路径
        
    Returns:
        annotations: dict, {"frame_name.png": {"ball_1": [u, v], ...}, ...}
    """
    logger.info(f"加载2D标注: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # 处理两种可能的格式：
    # 1. {"frame_name.png": {...}, ...}
    # 2. {"annotations": {"frame_name.png": {...}, ...}}
    if "annotations" in data:
        annotations = data["annotations"]
    else:
        annotations = data
    
    logger.info(f"共加载 {len(annotations)} 帧标注")
    
    return annotations


def load_keypoints_3d(filepath):
    """
    加载3D关键点坐标
    
    Args:
        filepath: keypoints_3d_model.json 文件路径
        
    Returns:
        keypoints_3d: 4x3 numpy array, 4个钢珠的3D坐标
    """
    logger.info(f"加载3D关键点: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # 使用原始坐标（obj文件中的坐标）
    keypoints_3d = np.array(data['keypoints_3d_original'], dtype=np.float64)
    
    logger.info(f"3D关键点坐标 (原始坐标系):\n{keypoints_3d}")
    logger.info(f"  坐标系: {data['coordinate_system_original']}")
    
    return keypoints_3d


def load_tooth_mesh(filepath):
    """
    加载牙齿3D模型
    
    Args:
        filepath: tooth.obj 文件路径
        
    Returns:
        mesh: trimesh.Trimesh 对象
    """
    logger.info(f"加载牙齿模型: {filepath}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"模型文件不存在: {filepath}")
    
    mesh = trimesh.load(filepath)
    
    logger.info(f"  顶点数: {len(mesh.vertices)}")
    logger.info(f"  面数: {len(mesh.faces)}")
    
    # 计算边界框中心（用于后续处理）
    bbox_center = mesh.vertices.mean(axis=0)
    logger.info(f"  边界框中心: {bbox_center}")
    
    return mesh


def project_points_to_image(points_3d, rvec, tvec, K, dist_coeffs=None):
    """
    将3D点投影到2D图像平面
    
    Args:
        points_3d: Nx3 array, 3D点坐标
        rvec: 旋转向量 (3,)
        tvec: 平移向量 (3,)
        K: 相机内参 (3x3)
        dist_coeffs: 畸变系数
        
    Returns:
        points_2d: Nx2 array, 2D图像坐标
    """
    points_2d, _ = cv2.projectPoints(
        points_3d, rvec, tvec, K, dist_coeffs
    )
    return points_2d.reshape(-1, 2)


def solve_pnp(keypoints_3d, keypoints_2d, K, method=cv2.SOLVEPNP_AP3P):
    """
    使用PnP解算相机位姿
    
    Args:
        keypoints_3d: 4x3 array, 3D关键点坐标
        keypoints_2d: 4x2 array, 2D关键点坐标
        K: 3x3 相机内参矩阵
        
    Returns:
        rvec: 旋转向量 (3,)
        tvec: 平移向量 (3,)
        success: bool, 是否成功
    """
    # AP3P算法支持4个点，是最优选择
    result = cv2.solvePnP(
        keypoints_3d,
        keypoints_2d,
        K,
        distCoeffs=None,
        flags=method,
        useExtrinsicGuess=False
    )
    
    # 处理不同OpenCV版本的返回值
    if len(result) == 4:
        success, rvec, tvec, inliers = result
    elif len(result) == 3:
        success, rvec, tvec = result
        inliers = None
    else:
        logger.warning("PnP返回值格式未知")
        return None, None, False
    
    if not success:
        logger.warning("AP3P解算失败，尝试使用P3P...")
        # 备选方案：使用P3P
        result = cv2.solvePnP(
            keypoints_3d,
            keypoints_2d,
            K,
            distCoeffs=None,
            flags=cv2.SOLVEPNP_P3P,
            useExtrinsicGuess=False
        )
        if len(result) == 4:
            success, rvec, tvec, inliers = result
        elif len(result) == 3:
            success, rvec, tvec = result
            inliers = None
        else:
            logger.warning("所有PnP方法均失败")
            return None, None, False
    
    # 验证重投影误差
    projected_2d = project_points_to_image(keypoints_3d, rvec, tvec, K)
    reproj_error = np.linalg.norm(projected_2d - keypoints_2d)
    
    logger.debug(f"PnP重投影误差: {reproj_error:.4f} pixels")
    
    return rvec, tvec, True


def convert_rvec_tvec_to_matrix(rvec, tvec):
    """
    将旋转向量和平移向量转换为4x4变换矩阵
    
    Args:
        rvec: 旋转向量 (3,)
        tvec: 平移向量 (3,)
        
    Returns:
        matrix: 4x4 变换矩阵 (World-to-Camera, OpenCV坐标系)
    """
    # 旋转向量 -> 旋转矩阵
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    
    # 构建4x4变换矩阵
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = tvec.flatten()
    
    return matrix


def convert_opencv_to_opengl(matrix):
    """
    将OpenCV坐标系转换为OpenGL坐标系
    
    输入：World-to-Camera 矩阵 (OpenCV坐标系)
    输出：Camera-to-World 矩阵 (OpenGL坐标系)
    
    OpenCV: +X right, +Y down, +Z forward (指向场景)
    OpenGL: +X right, +Y up, +Z backward (指向相机)
    
    步骤：
    1. 求逆得到 Camera-to-World (OpenCV)
    2. 坐标系转换：Y和Z轴取反
    """
    # OpenCV到OpenGL的转换矩阵（绕X轴旋转180度）
    cv_to_gl = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ], dtype=np.float64)
    
    # 求逆得到 Camera-to-World (OpenCV)
    c2w_cv = np.linalg.inv(matrix)
    
    # 转换到 OpenGL 坐标系
    matrix_gl = cv_to_gl @ c2w_cv @ cv_to_gl
    
    return matrix_gl


class DepthRenderer:
    """
    基于PyRender的深度图渲染器
    """
    
    def __init__(self, mesh, K, z_near=Config.Z_NEAR, z_far=Config.Z_FAR):
        """
        初始化渲染器
        
        Args:
            mesh: trimesh.Trimesh 对象
            K: 相机内参矩阵
            z_near: 近裁剪面距离（米）
            z_far: 远裁剪面距离（米）
        """
        self.mesh = mesh
        self.z_near = z_near
        self.z_far = z_far
        
        # 从K中提取焦距和主点
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        
        logger.info(f"初始化渲染器: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        logger.info(f"  z_near={z_near}, z_far={z_far}")
        
        # 创建PyRender场景
        self.scene = pyrender.Scene(
            ambient_light=[1.0, 1.0, 1.0],  # 环境光，确保深度图渲染正常
            bg_color=[0, 0, 0]
        )
        
        # 创建相机（使用IntrinsicsCamera）
        self.camera = pyrender.IntrinsicsCamera(
            fx=fx, fy=fy, cx=cx, cy=cy,
            znear=z_near, zfar=z_far
        )
        
        # 添加相机节点
        self.cam_node = self.scene.add(self.camera, pose=np.eye(4), name='camera')
        
        # 将trimesh转换为pyrender Mesh
        self.pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        
        # 添加物体节点
        self.mesh_node = self.scene.add(
            self.pyrender_mesh, 
            pose=np.eye(4), 
            name='object'
        )
        
        # 创建离屏渲染器
        # 从mesh边界框推断图像尺寸
        bbox = mesh.bounding_box.vertices
        h = int(2 * max(abs(bbox[:, 1].max()), abs(bbox[:, 1].min()))) + 100
        w = int(2 * max(abs(bbox[:, 0].max()), abs(bbox[:, 0].min()))) + 100
        
        # 根据相机内参和场景尺度估算图像尺寸
        # 使用主点位置和合理的视野范围
        h = max(int(2 * cy + 100), 480)
        w = max(int(2 * cx + 100), 640)
        
        logger.info(f"渲染图像尺寸: {w} x {h}")
        self.renderer = pyrender.OffscreenRenderer(w, h)
        
        logger.info("渲染器初始化完成")
    
    def set_camera_pose(self, pose_matrix):
        """
        设置相机位姿
        
        Args:
            pose_matrix: 4x4 相机位姿矩阵 (Camera-to-World, OpenGL坐标系)
        """
        self.scene.set_pose(self.cam_node, pose_matrix)
    
    def render_depth(self):
        """
        渲染深度图
        
        Returns:
            depth: numpy array, 深度图（float32，米为单位）
        """
        result = self.renderer.render(
            self.scene, 
            flags=Config.RENDER_FLAGS
        )
        
        # DEPTH_ONLY标志可能返回单个值或元组
        if isinstance(result, tuple):
            _, depth = result
        else:
            depth = result
        
        return depth
    
    def get_image_size(self):
        """获取渲染图像尺寸"""
        return self.renderer.viewport_width, self.renderer.viewport_height
    
    def cleanup(self):
        """清理资源"""
        self.renderer.delete()


def validate_frame_annotations(annotations, frame_name):
    """
    验证帧标注是否有效
    
    Args:
        annotations: 标注字典
        frame_name: 帧名称
        
    Returns:
        valid: bool, 是否有效
        ball_coords: dict, {ball_name: [u, v], ...}
    """
    if frame_name not in annotations:
        logger.warning(f"帧 {frame_name} 不在标注文件中")
        return False, None
    
    ball_coords = annotations[frame_name]
    
    # 检查是否有4个球的标注
    required_balls = ['ball_1', 'ball_2', 'ball_3', 'ball_4']
    missing_balls = [b for b in required_balls if b not in ball_coords]
    
    if missing_balls:
        logger.warning(f"帧 {frame_name} 缺少球的标注: {missing_balls}")
        return False, None
    
    # 检查标注点数量
    if len(ball_coords) < 4:
        logger.warning(f"帧 {frame_name} 标注点不足: {len(ball_coords)} < 4")
        return False, None
    
    return True, ball_coords


def extract_2d_points(ball_coords):
    """
    提取2D点坐标，确保顺序与3D关键点一致
    
    Args:
        ball_coords: dict, {ball_name: [u, v], ...}
        
    Returns:
        points_2d: 4x2 numpy array, 按 ball_1, ball_2, ball_3, ball_4 顺序
    """
    points_2d = []
    
    for ball_name in ['ball_1', 'ball_2', 'ball_3', 'ball_4']:
        if ball_name in ball_coords:
            u, v = ball_coords[ball_name]
            points_2d.append([float(u), float(v)])
        else:
            raise ValueError(f"缺少 {ball_name} 的标注")
    
    return np.array(points_2d, dtype=np.float64)


def main():
    """
    主函数：生成深度图Ground Truth
    """
    logger.info("=" * 60)
    logger.info("深度图 Ground Truth 生成程序")
    logger.info("=" * 60)
    
    # ============== 1. 初始化与加载 ==============
    logger.info("\n[步骤1] 初始化与加载数据...")
    
    # 创建输出目录
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    logger.info(f"输出目录: {Config.OUTPUT_DIR}")
    
    # 加载相机内参
    K = load_camera_intrinsics(Config.CAM_K_PATH)
    
    # 加载3D关键点
    keypoints_3d = load_keypoints_3d(Config.KEYPOINTS_3D_PATH)
    
    # 加载2D标注
    annotations = load_annotations(Config.ANNOTATIONS_PATH)
    
    # 加载3D模型
    mesh = load_tooth_mesh(Config.TOOTH_OBJ_PATH)
    
    # ============== 2. 初始化渲染器 ==============
    logger.info("\n[步骤2] 初始化深度图渲染器...")
    renderer = DepthRenderer(mesh, K)
    
    # ============== 3. 逐帧处理 ==============
    logger.info("\n[步骤3] 开始逐帧处理...")
    
    frame_names = sorted(annotations.keys())
    total_frames = len(frame_names)
    success_count = 0
    skip_count = 0
    
    for idx, frame_name in enumerate(frame_names):
        logger.info(f"\n处理帧 {idx+1}/{total_frames}: {frame_name}")
        
        # 验证标注
        valid, ball_coords = validate_frame_annotations(annotations, frame_name)
        if not valid:
            skip_count += 1
            continue
        
        # 提取2D点
        try:
            points_2d = extract_2d_points(ball_coords)
        except ValueError as e:
            logger.warning(f"提取2D点失败: {e}")
            skip_count += 1
            continue
        
        logger.debug(f"2D点坐标:\n{points_2d}")
        
        # PnP解算
        rvec, tvec, success = solve_pnp(keypoints_3d, points_2d, K)
        if not success or rvec is None:
            logger.warning(f"PnP解算失败: {frame_name}")
            skip_count += 1
            continue
        
        # 转换为4x4变换矩阵
        w2c_matrix = convert_rvec_tvec_to_matrix(rvec, tvec)
        logger.debug(f"World-to-Camera矩阵 (OpenCV):\n{w2c_matrix}")
        
        # 转换为OpenGL坐标系
        cam_pose_gl = convert_opencv_to_opengl(w2c_matrix)
        logger.debug(f"Camera-to-World矩阵 (OpenGL):\n{cam_pose_gl}")
        
        # 设置相机位姿并渲染
        renderer.set_camera_pose(cam_pose_gl)
        depth = renderer.render_depth()
        
        # 验证深度图
        valid_depth = depth > 0
        if not np.any(valid_depth):
            logger.warning(f"深度图全为0，可能渲染问题: {frame_name}")
        
        # 保存深度图
        output_path = os.path.join(
            Config.OUTPUT_DIR, 
            frame_name.replace('.png', '.npy').replace('.jpg', '.npy')
        )
        np.save(output_path, depth.astype(np.float32))
        
        logger.info(f"  深度图已保存: {output_path}")
        logger.info(f"  深度范围: [{depth[valid_depth].min():.4f}, {depth[valid_depth].max():.4f}] m")
        
        success_count += 1
    
    # ============== 4. 完成 ==============
    logger.info("\n" + "=" * 60)
    logger.info("处理完成!")
    logger.info(f"  总帧数: {total_frames}")
    logger.info(f"  成功: {success_count}")
    logger.info(f"  跳过: {skip_count}")
    logger.info(f"  输出目录: {Config.OUTPUT_DIR}")
    logger.info("=" * 60)
    
    # 清理资源
    renderer.cleanup()
    
    return success_count > 0


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
