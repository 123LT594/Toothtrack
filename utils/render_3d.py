#3D渲染和可视化

import numpy as np
import cv2
import trimesh
import os
from typing import Tuple, Optional, List


def project_points_to_image(points_3d: np.ndarray, K: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """
    将3D点投影到图像平面（使用统一的投影函数）
    
    Args:
        points_3d: Nx3 的3D点坐标（物体坐标系）
        K: 3x3 相机内参矩阵
        pose: 4x4 物体到相机的变换矩阵（OpenCV坐标系）
        
    Returns:
        Nx2 的图像坐标和深度值
    """
    from .tools import project_points_to_2d
    return project_points_to_2d(points_3d, K, pose)


def render_mesh_shaded(rgb_image: np.ndarray, 
                       mesh: trimesh.Trimesh,
                       pose: np.ndarray,
                       K: np.ndarray,
                       color: Tuple[int, int, int] = (180, 180, 230),
                       opacity: float = 0.6,
                       light_direction: np.ndarray = np.array([0, 0, -1])) -> np.ndarray:
    """
    渲染带阴影的3D网格模型（优化版本）
    
    Args:
        rgb_image: 输入RGB图像
        mesh: trimesh网格对象
        pose: 4x4 姿态矩阵
        K: 3x3 相机内参
        color: 网格颜色
        opacity: 透明度
        light_direction: 光源方向
        
    Returns:
        渲染后的图像
    """
    result = rgb_image.copy()
    h, w = rgb_image.shape[:2]
    
    # 投影所有顶点（向量化）
    vertices_2d, depths = project_points_to_image(mesh.vertices, K, pose)
    vertices_2d = np.round(vertices_2d).astype(np.int32)
    
    # 将所有顶点变换到相机坐标系（向量化）
    vertices_homo = np.hstack([mesh.vertices, np.ones((mesh.vertices.shape[0], 1))])
    vertices_cam = (pose @ vertices_homo.T).T[:, :3]
    
    # 向量化计算所有面的属性
    faces = mesh.faces
    n_faces = len(faces)
    
    # 获取每个面的顶点索引
    v0_indices = faces[:, 0]
    v1_indices = faces[:, 1]
    v2_indices = faces[:, 2]
    
    # 获取每个面的顶点（向量化）
    v0_cam = vertices_cam[v0_indices]
    v1_cam = vertices_cam[v1_indices]
    v2_cam = vertices_cam[v2_indices]
    
    # 计算所有面的法向量（向量化）
    edge1 = v1_cam - v0_cam
    edge2 = v2_cam - v0_cam
    normals = np.cross(edge1, edge2)
    normals_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals_norm[normals_norm < 1e-10] = 1.0  # 避免除零
    normals = normals / normals_norm
    
    # 背面剔除（向量化）
    front_facing = normals[:, 2] < 0  # 法向量Z分量小于0表示面向相机
    
    # 计算每个面的深度（向量化）
    face_depths = depths[v0_indices] + depths[v1_indices] + depths[v2_indices]
    face_depths = face_depths / 3.0
    
    # 检查深度有效性（向量化）
    valid_depth = (depths[v0_indices] > 0) & (depths[v1_indices] > 0) & (depths[v2_indices] > 0)
    
    # 计算光照强度（向量化）
    light_dir_norm = light_direction / np.linalg.norm(light_direction)
    dot_products = -np.dot(normals, light_dir_norm)
    light_intensities = np.maximum(0, dot_products)
    light_intensities = 0.3 + 0.7 * light_intensities  # 环境光 + 漫反射
    
    # 获取每个面的2D投影点
    pts_2d_0 = vertices_2d[v0_indices]
    pts_2d_1 = vertices_2d[v1_indices]
    pts_2d_2 = vertices_2d[v2_indices]
    
    # 检查是否在图像边界内（向量化）
    min_x = np.minimum(np.minimum(pts_2d_0[:, 0], pts_2d_1[:, 0]), pts_2d_2[:, 0])
    max_x = np.maximum(np.maximum(pts_2d_0[:, 0], pts_2d_1[:, 0]), pts_2d_2[:, 0])
    min_y = np.minimum(np.minimum(pts_2d_0[:, 1], pts_2d_1[:, 1]), pts_2d_2[:, 1])
    max_y = np.maximum(np.maximum(pts_2d_0[:, 1], pts_2d_1[:, 1]), pts_2d_2[:, 1])
    
    in_bounds = (min_x >= 0) & (max_x < w) & (min_y >= 0) & (max_y < h)
    
    # 综合所有条件
    valid_faces = front_facing & valid_depth & in_bounds
    
    # 只处理有效的面
    valid_indices = np.where(valid_faces)[0]
    
    if len(valid_indices) == 0:
        return result
    
    # 优化：直接使用numpy数组，避免Python循环
    # 准备渲染数据（向量化）
    valid_pts_2d_0 = pts_2d_0[valid_indices]
    valid_pts_2d_1 = pts_2d_1[valid_indices]
    valid_pts_2d_2 = pts_2d_2[valid_indices]
    valid_depths = face_depths[valid_indices]
    valid_intensities = light_intensities[valid_indices]
    
    # 按深度排序（画家算法）
    sort_indices = np.argsort(valid_depths)[::-1]
    valid_pts_2d_0 = valid_pts_2d_0[sort_indices]
    valid_pts_2d_1 = valid_pts_2d_1[sort_indices]
    valid_pts_2d_2 = valid_pts_2d_2[sort_indices]
    valid_intensities = valid_intensities[sort_indices]
    
    # 创建渲染层
    render_layer = np.zeros_like(rgb_image, dtype=np.uint8)
    
    # 优化：量化光照强度到8个级别，减少fillPoly调用
    intensity_quantized = np.round(valid_intensities * 8).astype(np.int32)
    unique_intensities = np.unique(intensity_quantized)
    
    # 按强度级别批量渲染
    for intensity_level in unique_intensities:
        mask_level = intensity_quantized == intensity_level
        intensity = intensity_level / 8.0
        face_color = tuple(int(c * intensity) for c in color)
        
        level_indices = np.where(mask_level)[0]
        polygons = []
        for idx in level_indices:
            polygons.append(np.array([
                valid_pts_2d_0[idx],
                valid_pts_2d_1[idx],
                valid_pts_2d_2[idx]
            ], dtype=np.int32))
        
        if len(polygons) > 0:
            cv2.fillPoly(render_layer, polygons, face_color)
    
    # 混合
    mask = np.any(render_layer > 0, axis=2)
    if np.any(mask):
        result = result.astype(np.float32)
        render_layer = render_layer.astype(np.float32)
        result[mask] = (1 - opacity) * result[mask] + opacity * render_layer[mask]
    
    return result.astype(np.uint8)


def render_mesh_wireframe(rgb_image: np.ndarray,
                         mesh: trimesh.Trimesh,
                         pose: np.ndarray,
                         K: np.ndarray,
                         color: Tuple[int, int, int] = (0, 255, 0),
                         thickness: int = 1) -> np.ndarray:
    """
    渲染网格线框
    
    Args:
        rgb_image: 输入RGB图像
        mesh: trimesh网格对象
        pose: 4x4 姿态矩阵
        K: 3x3 相机内参
        color: 线条颜色
        thickness: 线条粗细
        
    Returns:
        渲染后的图像
    """
    result = rgb_image.copy()
    
    # 投影顶点
    vertices_2d, depths = project_points_to_image(mesh.vertices, K, pose)
    vertices_2d = np.round(vertices_2d).astype(np.int32)
    
    # 获取所有边
    edges = set()
    for face in mesh.faces:
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i+1)%3]]))
            edges.add(edge)
    
    # 绘制边
    for v1_idx, v2_idx in edges:
        # 检查深度
        if depths[v1_idx] <= 0 or depths[v2_idx] <= 0:
            continue
            
        pt1 = tuple(vertices_2d[v1_idx])
        pt2 = tuple(vertices_2d[v2_idx])
        
        cv2.line(result, pt1, pt2, color, thickness, cv2.LINE_AA)
    
    return result


def render_mesh_points(rgb_image: np.ndarray,
                      mesh: trimesh.Trimesh,
                      pose: np.ndarray,
                      K: np.ndarray,
                      color: Tuple[int, int, int] = (255, 0, 0),
                      radius: int = 2) -> np.ndarray:
    """
    渲染网格顶点
    
    Args:
        rgb_image: 输入RGB图像
        mesh: trimesh网格对象  
        pose: 4x4 姿态矩阵
        K: 3x3 相机内参
        color: 点颜色
        radius: 点半径
        
    Returns:
        渲染后的图像
    """
    result = rgb_image.copy()
    
    # 投影顶点
    vertices_2d, depths = project_points_to_image(mesh.vertices, K, pose)
    vertices_2d = np.round(vertices_2d).astype(np.int32)
    
    # 绘制顶点
    for i, (pt, depth) in enumerate(zip(vertices_2d, depths)):
        if depth <= 0:
            continue
            
        cv2.circle(result, tuple(pt), radius, color, -1, cv2.LINE_AA)
    
    return result


def render_mesh_combined(rgb_image: np.ndarray,
                        mesh: trimesh.Trimesh,
                        pose: np.ndarray,
                        K: np.ndarray,
                        render_shaded: bool = True,
                        render_wireframe: bool = False,
                        render_points: bool = False,
                        shaded_color: Tuple[int, int, int] = (180, 180, 230),
                        shaded_opacity: float = 0.5,
                        wireframe_color: Tuple[int, int, int] = (0, 255, 0),
                        wireframe_thickness: int = 1,
                        points_color: Tuple[int, int, int] = (255, 0, 0),
                        points_radius: int = 2) -> np.ndarray:
    result = rgb_image.copy()
    
    if render_shaded:
        result = render_mesh_shaded(
            result, mesh, pose, K,
            shaded_color, shaded_opacity
        )
    
    if render_wireframe:
        result = render_mesh_wireframe(
            result, mesh, pose, K,
            wireframe_color, wireframe_thickness
        )
    
    if render_points:
        result = render_mesh_points(
            result, mesh, pose, K,
            points_color, points_radius
        )
    
    return result


def get_mesh_colors_by_filename(mesh_files: list) -> list:
    """
    根据mesh文件名自动分配颜色（与vtk_renderer.py相同的颜色顺序）
    
    Args:
        mesh_files: mesh文件路径列表
        
    Returns:
        颜色列表（BGR格式，0-255范围）
    """
    # 与vtk_renderer.py相同的颜色（BGR格式）
    # VTK: (1.0,0.0,0.0)红 -> BGR(0,0,255), (0.0,1.0,0.0)绿 -> BGR(0,255,0)
    #      (0.0,0.0,1.0)蓝 -> BGR(255,0,0), (0.0,1.0,1.0)青 -> BGR(255,255,0)
    steel_colors = [
        (0, 0, 255),      # 1.obj - 红色（与verify_visualization.py一致）
        (0, 255, 0),      # 2.obj - 绿色
        (255, 0, 0),      # 3.obj - 蓝色
        (0, 255, 255),    # 4.obj - 黄色（改为更明显的黄色）
    ]
    
    # tooth.obj使用灰色（BGR格式）
    gray_color = (128, 128, 128)  # 灰色 BGR(128, 128, 128) = RGB(128, 128, 128)
    default_color = (179, 179, 230)  # 其他模型的默认颜色（浅蓝色）
    
    colors = []
    import logging
    for mesh_file in mesh_files:
        filename = os.path.basename(mesh_file) if isinstance(mesh_file, str) else str(mesh_file)
        
        # 检查是否是1-4.obj（使用更严格的匹配）
        if filename == '1.obj' or filename.endswith('/1.obj') or filename.endswith('\\1.obj'):
            colors.append(steel_colors[0])  # 红色
        elif filename == '2.obj' or filename.endswith('/2.obj') or filename.endswith('\\2.obj'):
            colors.append(steel_colors[1])  # 绿色
        elif filename == '3.obj' or filename.endswith('/3.obj') or filename.endswith('\\3.obj'):
            colors.append(steel_colors[2])  # 蓝色
        elif filename == '4.obj' or filename.endswith('/4.obj') or filename.endswith('\\4.obj'):
            colors.append(steel_colors[3])  # 黄色
        elif filename == 'tooth.obj' or filename.endswith('/tooth.obj') or filename.endswith('\\tooth.obj'):
            colors.append(gray_color)  # 灰色
        else:
            colors.append(default_color)  # 默认颜色
    
    return colors


def render_multiple_meshes(rgb_image: np.ndarray,
                           meshes: list,
                           poses: list,
                           K: np.ndarray,
                           colors: list = None,
                           opacity: float = 0.5,
                           mesh_files: list = None) -> np.ndarray:
    """
    在一张图像上渲染多个3D模型（优化版本）
    
    Args:
        rgb_image: 输入RGB图像
        meshes: mesh对象列表
        poses: 姿态矩阵列表（4x4）
        K: 3x3 相机内参
        colors: 颜色列表，如果None则根据mesh_files自动分配颜色
        opacity: 透明度
        mesh_files: mesh文件路径列表（用于自动分配颜色）
        
    Returns:
        渲染后的图像
    """
    result = rgb_image.copy()
    
    if colors is None:
        if mesh_files is not None:
            # 确保mesh_files顺序与meshes顺序一致
            if len(mesh_files) != len(meshes):
                import logging
                logging.warning(f"mesh_files长度({len(mesh_files)})与meshes长度({len(meshes)})不匹配，将截断或填充")
            colors = get_mesh_colors_by_filename(mesh_files)
        else:
            colors = [(179, 179, 230)] * len(meshes)
    
    # 确保colors长度匹配
    if len(colors) != len(meshes):
        import logging
        logging.warning(f"colors长度({len(colors)})与meshes长度({len(meshes)})不匹配，将截断或填充")
        colors = colors[:len(meshes)] if len(colors) > len(meshes) else colors + [(179, 179, 230)] * (len(meshes) - len(colors))
    
    # 确保meshes和poses长度匹配
    min_len = min(len(meshes), len(poses), len(colors))
    if min_len < len(meshes):
        import logging
        logging.warning(f"截断到min_len={min_len} (meshes={len(meshes)}, poses={len(poses)}, colors={len(colors)})")
    meshes = meshes[:min_len]
    poses = poses[:min_len]
    colors = colors[:min_len]
    
    # 合并所有面的渲染数据（优化：减少fillPoly调用）
    all_faces_data = []
    
    for i, (mesh, pose, color) in enumerate(zip(meshes, poses, colors)):
        vertices_2d, depths = project_points_to_image(mesh.vertices, K, pose)
        vertices_2d = np.round(vertices_2d).astype(np.int32)
        
        vertices_homo = np.hstack([mesh.vertices, np.ones((mesh.vertices.shape[0], 1))])
        vertices_cam = (pose @ vertices_homo.T).T[:, :3]
        
        faces = mesh.faces
        v0_indices = faces[:, 0]
        v1_indices = faces[:, 1]
        v2_indices = faces[:, 2]
        
        v0_cam = vertices_cam[v0_indices]
        v1_cam = vertices_cam[v1_indices]
        v2_cam = vertices_cam[v2_indices]
        
        edge1 = v1_cam - v0_cam
        edge2 = v2_cam - v0_cam
        normals = np.cross(edge1, edge2)
        normals_norm = np.linalg.norm(normals, axis=1, keepdims=True)
        normals_norm[normals_norm < 1e-10] = 1.0
        normals = normals / normals_norm
        
        front_facing = normals[:, 2] < 0
        face_depths = (depths[v0_indices] + depths[v1_indices] + depths[v2_indices]) / 3.0
        valid_depth = (depths[v0_indices] > 0) & (depths[v1_indices] > 0) & (depths[v2_indices] > 0)
        
        # 计算光照效果（Phong shading：环境光 + 漫反射 + 高光）
        # 光源方向：从左上角照射，增强立体感
        light_direction = np.array([0.4, 0.4, -1.0])
        light_dir_norm = light_direction / np.linalg.norm(light_direction)
        
        # 漫反射（Lambertian）
        dot_products = -np.dot(normals, light_dir_norm)
        diffuse = np.maximum(0, dot_products)
        
        # 高光反射（Blinn-Phong）
        # 视线方向（从相机看向物体）
        view_direction = np.array([0, 0, -1])
        view_dir_norm = view_direction / np.linalg.norm(view_direction)
        # 半角向量（对每个面计算）
        # light_dir_norm 和 view_dir_norm 都是1D，需要广播到每个面
        half_vector = light_dir_norm + view_dir_norm  # (3,)
        half_vector_norm = half_vector / (np.linalg.norm(half_vector) + 1e-10)  # (3,)
        # 高光强度：计算每个面的法向量与半角向量的点积
        specular_dot = np.maximum(0, np.dot(normals, half_vector_norm))  # (n_faces,)
        specular = np.power(specular_dot, 32.0)  # 高光指数，值越大高光越集中
        
        # 组合光照：环境光(0.4) + 漫反射(0.4) + 高光(0.2)，整体更亮
        light_intensities = 0.4 + 0.4 * diffuse + 0.2 * specular
        # 进一步增加整体亮度，避免过暗
        light_intensities = light_intensities * 1.4
        light_intensities = np.clip(light_intensities, 0.0, 1.0)
        
        pts_2d_0 = vertices_2d[v0_indices]
        pts_2d_1 = vertices_2d[v1_indices]
        pts_2d_2 = vertices_2d[v2_indices]
        
        h, w = rgb_image.shape[:2]
        min_x = np.minimum(np.minimum(pts_2d_0[:, 0], pts_2d_1[:, 0]), pts_2d_2[:, 0])
        max_x = np.maximum(np.maximum(pts_2d_0[:, 0], pts_2d_1[:, 0]), pts_2d_2[:, 0])
        min_y = np.minimum(np.minimum(pts_2d_0[:, 1], pts_2d_1[:, 1]), pts_2d_2[:, 1])
        max_y = np.maximum(np.maximum(pts_2d_0[:, 1], pts_2d_1[:, 1]), pts_2d_2[:, 1])
        in_bounds = (min_x >= 0) & (max_x < w) & (min_y >= 0) & (max_y < h)
        
        valid_faces = front_facing & valid_depth & in_bounds
        valid_indices = np.where(valid_faces)[0]
        
        if len(valid_indices) > 0:
            valid_pts_0 = pts_2d_0[valid_indices]
            valid_pts_1 = pts_2d_1[valid_indices]
            valid_pts_2 = pts_2d_2[valid_indices]
            valid_depths = face_depths[valid_indices]
            valid_intensities = light_intensities[valid_indices]
            
            # 量化光照强度到32个级别，平衡性能和纹理感
            intensity_quantized = np.round(valid_intensities * 32).astype(np.int32) / 32.0
            
            for idx in range(len(valid_indices)):
                # 根据光照强度调整颜色，增强纹理感
                intensity = intensity_quantized[idx]
                shaded_color = tuple(int(c * intensity) for c in color)
                all_faces_data.append({
                    'pts': np.array([valid_pts_0[idx], valid_pts_1[idx], valid_pts_2[idx]], dtype=np.int32),
                    'depth': valid_depths[idx],
                    'color': shaded_color  # 使用光照调整后的颜色
                })
    
    # 按深度排序所有面（从远到近）
    all_faces_data.sort(key=lambda x: x['depth'], reverse=True)
    
    # 创建渲染层
    render_layer = np.zeros_like(rgb_image, dtype=np.uint8)
    
    # 按颜色分组渲染（减少fillPoly调用）
    color_groups = {}
    for face_data in all_faces_data:
        color_key = tuple(face_data['color'])
        if color_key not in color_groups:
            color_groups[color_key] = []
        color_groups[color_key].append(face_data['pts'])
    
    # 批量渲染
    # 注意：rgb_image是RGB格式，需要将BGR颜色转换为RGB颜色
    for color, polygons in color_groups.items():
        # 将BGR颜色转换为RGB颜色（因为rgb_image是RGB格式）
        rgb_color = (color[2], color[1], color[0])  # BGR to RGB
        cv2.fillPoly(render_layer, polygons, tuple(int(c) for c in rgb_color))
    
    # 混合
    mask = np.any(render_layer > 0, axis=2)
    if np.any(mask):
        result = result.astype(np.float32)
        render_layer = render_layer.astype(np.float32)
        result[mask] = (1 - opacity) * result[mask] + opacity * render_layer[mask]
    
    return result.astype(np.uint8)


def create_depth_map(mesh: trimesh.Trimesh,
                    pose: np.ndarray,
                    K: np.ndarray,
                    width: int,
                    height: int) -> np.ndarray:
    """
    创建深度图
    
    Args:
        mesh: trimesh网格对象
        pose: 4x4 姿态矩阵
        K: 3x3 相机内参
        width: 图像宽度
        height: 图像高度
        
    Returns:
        深度图 (H, W)
    """
    # 初始化深度图
    depth_map = np.full((height, width), np.inf)
    
    # 投影顶点
    vertices_2d, depths = project_points_to_image(mesh.vertices, K, pose)
    vertices_2d = np.round(vertices_2d).astype(np.int32)
    
    # 渲染每个面
    for face in mesh.faces:
        pts_2d = vertices_2d[face]
        face_depths = depths[face]
        
        # 检查深度
        if np.any(face_depths <= 0):
            continue
        
        # 创建面的掩码
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [pts_2d], 1)
        
        # 获取面内的像素坐标
        y_coords, x_coords = np.where(mask > 0)
        
        if len(y_coords) == 0:
            continue
        
        # 计算重心坐标插值深度
        # 简化处理：使用平均深度
        avg_depth = np.mean(face_depths)
        
        # 更新深度图
        for y, x in zip(y_coords, x_coords):
            if 0 <= x < width and 0 <= y < height:
                depth_map[y, x] = min(depth_map[y, x], avg_depth)
    
    # 将无穷大值设为0
    depth_map[depth_map == np.inf] = 0
    
    return depth_map


def prepare_mesh_files_for_rendering(mesh_dir: str) -> List[str]:
    """
    准备用于渲染的mesh文件列表，按特定顺序排序
    
    Args:
        mesh_dir: mesh文件目录路径
        
    Returns:
        排序后的mesh文件路径列表（1.obj, 2.obj, 3.obj, 4.obj, tooth.obj, 其他...）
    """
    all_mesh_files = [f for f in os.listdir(mesh_dir) if f.endswith('.obj')]
    
    # 排序逻辑：1-4.obj优先，然后是tooth.obj，最后是其他文件
    def sort_key(f):
        basename = os.path.basename(f)
        if basename in ['1.obj', '2.obj', '3.obj', '4.obj']:
            return (0, int(basename[0]))  # 数字排序
        elif basename == 'tooth.obj':
            return (1, 0)  # tooth.obj排在第二位
        else:
            return (2, basename)  # 其他文件按字母排序
    
    sorted_files = sorted([os.path.join(mesh_dir, f) for f in all_mesh_files], key=lambda x: sort_key(x))
    return sorted_files


def filter_mesh_files_for_rendering(all_mesh_files: List[str], 
                                    include_balls: bool = True,
                                    include_tooth: bool = True) -> List[str]:
    """
    筛选用于渲染的mesh文件（只保留1-4.obj和tooth.obj）
    
    Args:
        all_mesh_files: 所有mesh文件路径列表
        include_balls: 是否包含1-4.obj
        include_tooth: 是否包含tooth.obj
        
    Returns:
        筛选后的mesh文件路径列表
    """
    filtered = []
    for mesh_file in all_mesh_files:
        basename = os.path.basename(mesh_file)
        if include_balls and basename in ['1.obj', '2.obj', '3.obj', '4.obj']:
            filtered.append(mesh_file)
        elif include_tooth and basename == 'tooth.obj':
            filtered.append(mesh_file)
    
    # 确保1-4.obj按数字顺序
    ball_files = [f for f in filtered if os.path.basename(f) in ['1.obj', '2.obj', '3.obj', '4.obj']]
    ball_files = sorted(ball_files, key=lambda x: int(os.path.basename(x).replace('.obj', '')))
    
    # 添加tooth.obj
    tooth_files = [f for f in filtered if os.path.basename(f) == 'tooth.obj']
    
    return ball_files + tooth_files


def compute_center_pose_for_visualization(pose: np.ndarray,
                                           to_origin: np.ndarray,
                                           mesh: trimesh.Trimesh = None,
                                           pose_centered: np.ndarray = None,
                                           use_rt_init: bool = False) -> np.ndarray:
    """
    计算用于可视化的center_pose（oriented_bounds中心化坐标系）
    
    注意：输入的pose是简单中心化坐标系（边界框中心）的，需要转换到oriented_bounds中心化坐标系
    
    Args:
        pose: 简单中心化坐标系的位姿（边界框中心，与FoundationPose一致）
        to_origin: oriented_bounds中心化变换矩阵（从原始坐标系到oriented_bounds中心化坐标系）
        mesh: 主mesh对象（用于计算model_center）
        pose_centered: 简单中心化坐标系的位姿（RT矩阵，如果使用RT初始化，与pose相同）
        use_rt_init: 是否使用RT初始化
        
    Returns:
        center_pose: oriented_bounds中心化坐标系的位姿
    """
    if mesh is not None:
        # 计算model_center（边界框中心）
        # 统一使用tools.py中的compute_mesh_center函数，确保一致性
        from .tools import compute_mesh_center
        model_center = compute_mesh_center(mesh)
        
        # 构建从原始坐标系到简单中心化坐标系的变换矩阵
        # 简单中心化：P_simple = P_original - model_center
        tf_to_simple = np.eye(4)
        tf_to_simple[:3, 3] = -model_center
        
        # 从简单中心化坐标系到oriented_bounds中心化坐标系的变换
        # P_oriented = to_origin @ P_original
        # P_original = inv(tf_to_simple) @ P_simple
        # 所以：P_oriented = to_origin @ inv(tf_to_simple) @ P_simple
        simple_to_oriented = to_origin @ np.linalg.inv(tf_to_simple)
        
        # 将pose从简单中心化坐标系转换到oriented_bounds中心化坐标系
        # pose是将简单中心化坐标系的点变换到相机坐标系
        # center_pose应该将oriented_bounds中心化坐标系的点变换到相机坐标系
        # 所以：center_pose = pose @ inv(simple_to_oriented)
        center_pose = pose @ np.linalg.inv(simple_to_oriented)
    else:
        # 如果没有mesh，假设pose已经是oriented_bounds中心化坐标系的
        center_pose = pose.copy()

    # 检查并修正Z轴方向（确保Z轴指向相机前方，符合OpenCV坐标系）
    # 在OpenCV坐标系中，物体应该在相机前方，所以平移向量的Z分量应该为正
    # 如果Z分量为负，说明物体在相机后面，需要翻转Z轴
    if center_pose[2, 3] < 0:
        # 翻转Z轴：旋转矩阵的Z轴取反，同时翻转平移的Z分量
        center_pose[:3, 2] = -center_pose[:3, 2]  # 翻转旋转矩阵的Z轴
        center_pose[2, 3] = -center_pose[2, 3]  # 翻转平移的Z分量
        # 为了保持右手坐标系，需要翻转X轴或Y轴之一
        # 这里选择翻转X轴以保持右手坐标系
        center_pose[:3, 0] = -center_pose[:3, 0]
    
    return center_pose


def prepare_multi_mesh_rendering(main_mesh: trimesh.Trimesh,
                                  all_mesh_files: List[str],
                                  center_pose: np.ndarray) -> Tuple[List[trimesh.Trimesh], List[np.ndarray], List[str]]:
    """
    准备多模型渲染：加载所有mesh并计算对应的位姿
    
    Args:
        main_mesh: 主模型（用于计算偏移参考）
        all_mesh_files: 所有mesh文件路径列表
        center_pose: 主模型的中心化位姿（oriented_bounds中心化物体坐标系到相机坐标系）
    
    Returns:
        render_meshes: 中心化后的mesh列表（oriented_bounds中心化）
        render_poses: 对应的位姿列表
        render_mesh_files: 对应的mesh文件路径列表（保持顺序一致）
    """
    render_meshes = []
    render_poses = []
    render_mesh_files = []
    
    # 主模型的oriented_bounds变换
    to_origin_main, extents = trimesh.bounds.oriented_bounds(main_mesh)
    
    # 主模型中心在原始坐标系中的位置（统一使用边界框中心，与FoundationPose一致）
    max_xyz_main = main_mesh.vertices.max(axis=0)
    min_xyz_main = main_mesh.vertices.min(axis=0)
    main_mesh_center = (min_xyz_main + max_xyz_main) / 2
    
    for idx, mesh_file in enumerate(all_mesh_files):
        m = trimesh.load(mesh_file)
        
        # 统一使用主模型的to_origin变换，确保所有模型都在同一个oriented_bounds中心化坐标系中
        render_mesh = m.copy()
        render_mesh.apply_transform(to_origin_main)  # 使用主模型的to_origin中心化每个模型
        render_meshes.append(render_mesh)
        render_mesh_files.append(mesh_file)  # 保持文件路径顺序
        
        # 由于系统已经统一使用中心化坐标系，所有mesh使用相同的pose
        render_poses.append(center_pose.copy())
    
    return render_meshes, render_poses, render_mesh_files


def create_visualization(rgb_image: np.ndarray,
                        pose: np.ndarray,
                        to_origin: np.ndarray,
                        K: np.ndarray,
                        bbox: np.ndarray,
                        fps: float = 0,
                        render_3d: bool = False,
                        render_meshes: List[trimesh.Trimesh] = None,
                        render_poses: List[np.ndarray] = None,
                        mesh_files: List[str] = None,
                        mesh_dir: str = None,
                        main_mesh: trimesh.Trimesh = None,
                        bbox_color: Tuple[int, int, int] = (0, 255, 0),
                        axis_scale: float = 0.1,
                        center_pose: np.ndarray = None,
                        pose_centered: np.ndarray = None,
                        use_rt_init: bool = False) -> np.ndarray:
    """
    创建完整的可视化图像（包括3D渲染、3D box、坐标轴等）
    
    Args:
        rgb_image: 输入RGB图像
        pose: 位姿（原始坐标系）
        to_origin: 中心化变换矩阵
        K: 相机内参
        bbox: 边界框
        fps: 帧率（用于显示）
        render_3d: 是否启用3D渲染
        render_meshes: 渲染的mesh列表（如果启用3D渲染，可选）
        render_poses: 对应的位姿列表（如果启用3D渲染，可选）
        mesh_files: mesh文件路径列表（用于自动颜色分配，可选）
        mesh_dir: mesh文件目录（如果提供，会自动加载和准备mesh，可选）
        main_mesh: 主mesh对象（如果提供mesh_dir，需要提供主mesh用于计算偏移，可选）
        bbox_color: 3D box颜色
        axis_scale: 坐标轴缩放
        center_pose: 中心化坐标系的位姿（如果未提供则自动计算）
        pose_centered: 简单中心化坐标系的位姿（RT矩阵，用于RT初始化时的坐标转换）
        use_rt_init: 是否使用RT初始化
    
    Returns:
        可视化图像
    """
    # 验证相机内参K和图像尺寸是否匹配
    h, w = rgb_image.shape[:2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    # 相机内参检查信息不输出
    
    # 转换为中心化坐标系（如果未提供则计算）
    if center_pose is None:
        # 确保to_origin和main_mesh一致
        if main_mesh is not None:
            # 重新计算to_origin以确保一致性
            to_origin_recalc, _ = trimesh.bounds.oriented_bounds(main_mesh)
            center_pose = compute_center_pose_for_visualization(
                pose, to_origin_recalc, main_mesh, pose_centered, use_rt_init
            )
        else:
            center_pose = compute_center_pose_for_visualization(
                pose, to_origin, main_mesh, pose_centered, use_rt_init
            )
    
    # 如果启用了3D渲染但没有提供mesh数据，尝试从mesh_dir自动加载
    if render_3d and (render_meshes is None or render_poses is None):
        if mesh_dir is not None and main_mesh is not None:
            # 自动加载和准备mesh文件
            try:
                all_mesh_files = prepare_mesh_files_for_rendering(mesh_dir)
                filtered_mesh_files = filter_mesh_files_for_rendering(
                    all_mesh_files, include_balls=True, include_tooth=True
                )
                if len(filtered_mesh_files) > 0:
                    render_meshes, render_poses, mesh_files = prepare_multi_mesh_rendering(
                        main_mesh, filtered_mesh_files, center_pose
                    )
                else:
                    # 如果没有找到合适的mesh文件，禁用3D渲染
                    logging.warning(f"在mesh_dir={mesh_dir}中未找到合适的mesh文件（需要1.obj, 2.obj, 3.obj, 4.obj或tooth.obj），禁用3D渲染")
                    render_3d = False
            except Exception as e:
                logging.error(f"加载mesh文件时出错: {e}")
                render_3d = False
        else:
            # 如果提供了mesh_dir或main_mesh但缺少另一个，禁用3D渲染
            if mesh_dir is not None or main_mesh is not None:
                import logging
                logging.warning(f"render_3d=True但缺少mesh_dir或main_mesh (mesh_dir={mesh_dir}, main_mesh={'provided' if main_mesh is not None else 'None'})，禁用3D渲染")
            render_3d = False
    
    vis = rgb_image.copy()
    
    # 显示FPS
    if fps > 0:
        vis = cv2.putText(vis, f"fps {int(fps)}", (10, 30), 
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # 3D模型渲染
    if render_3d and render_meshes is not None and render_poses is not None:
        if len(render_meshes) == 0:
            # 没有模型需要渲染，跳过
            logging.warning("render_meshes为空，跳过3D渲染")
            pass
        elif len(render_meshes) > 1:
            # 多模型渲染
            vis = render_multiple_meshes(
                vis,
                render_meshes,
                render_poses,
                K,
                colors=None,  # 自动根据文件名分配颜色
                opacity=0.5,
                mesh_files=mesh_files
            )
        else:
            # 单模型渲染
            vis = render_mesh_combined(
                vis,
                render_meshes[0],
                render_poses[0],
                K,
                render_shaded=True,
                render_wireframe=False,
                render_points=False,
                shaded_color=(179, 179, 230),
                shaded_opacity=0.5,
                wireframe_color=(0, 255, 0),
                wireframe_thickness=1
            )
    
    # 渲染3D box（如果提供了bbox）- 仍然关闭（避免画框干扰）
    # if bbox is not None:
    #     try:
    #         from .utils import draw_posed_3d_box
    #         vis = draw_posed_3d_box(K, vis, center_pose, bbox, line_color=bbox_color, linewidth=2)
    #     except ImportError:
    #         import logging
    #         logging.warning("无法导入draw_posed_3d_box函数，跳过3D box渲染")

    # 渲染坐标轴（如果提供了axis_scale且大于0）——关闭
    # if axis_scale > 0:
    #     try:
    #         from .utils import draw_xyz_axis
    #         # 创建坐标轴方向变换矩阵，将坐标轴映射到期望的方向：
    #         # 红色（X轴）-> 向外（+Z方向，相机前方）
    #         # 绿色（Y轴）-> 向上（-Y方向，图像上方）
    #         # 蓝色（Z轴）-> 向左（-X方向，图像左侧）
    #         axis_transform = np.array([
    #             [0, 0, 1, 0],   # X轴（红色）-> Z方向（向外）
    #             [0, -1, 0, 0],  # Y轴（绿色）-> -Y方向（向上）
    #             [-1, 0, 0, 0],  # Z轴（蓝色）-> -X方向（向左）
    #             [0, 0, 0, 1]
    #         ], dtype=np.float32)
    #         # 应用坐标轴变换到pose
    #         axis_pose = center_pose @ axis_transform
    #         vis = draw_xyz_axis(
    #             vis, axis_pose, scale=axis_scale, K=K,
    #             thickness=3, transparency=0, is_input_rgb=False
    #         )
    #     except ImportError:
    #         import logging
    #         logging.warning("无法导入draw_xyz_axis函数，跳过坐标轴渲染")

    return vis
