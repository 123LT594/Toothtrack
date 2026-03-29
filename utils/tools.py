#工具函数（深度渲染、坐标转换等）

import numpy as np
import trimesh
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
from .utils import *
from scipy.spatial.distance import cdist
import torch
from scipy.spatial import ConvexHull
from filterpy.kalman import KalmanFilter

from scipy.spatial.transform import Rotation
from bop_toolkit_lib import pose_error


from bop_toolkit_lib import misc
from bop_toolkit_lib import visibility


# ==================== 统一的坐标变换和投影函数 ====================

def compute_mesh_center(mesh) -> np.ndarray:
    """
    统一计算mesh中心（边界框中心）

    重要：所有地方都应该使用这个函数计算mesh中心，确保一致性。
    与FoundationPose和prepare_data.py的处理方式一致。

    Args:
        mesh: trimesh网格对象或顶点数组 (N, 3)

    Returns:
        center: 3D数组，边界框中心坐标
    """
    if hasattr(mesh, 'vertices'):
        vertices = mesh.vertices
    else:
        vertices = np.array(mesh)

    max_xyz = vertices.max(axis=0)
    min_xyz = vertices.min(axis=0)
    return (min_xyz + max_xyz) / 2


def create_centered_mesh(mesh: trimesh.Trimesh, model_center: np.ndarray) -> trimesh.Trimesh:
    """
    创建中心化的mesh（统一函数）

    将mesh的顶点坐标减去中心点，使mesh以原点为中心。

    Args:
        mesh: 原始mesh
        model_center: mesh中心坐标

    Returns:
        mesh_centered: 中心化的mesh副本
    """
    mesh_centered = mesh.copy()
    mesh_centered.vertices = mesh.vertices - model_center
    return mesh_centered


def convert_pose_for_render(pose: np.ndarray, model_center: np.ndarray) -> np.ndarray:
    """
    将中心化坐标系的pose转换为原始坐标系用于渲染（统一函数）

    注意：render_cad_depth需要原始坐标系的pose，但track_one返回的是中心化坐标系的pose。
    这个函数用于在两者之间进行转换。

    Args:
        pose: 中心化坐标系的pose (4x4)
        model_center: mesh中心坐标

    Returns:
        pose_for_render: 原始坐标系的pose（用于render_cad_depth）
    """
    pose_for_render = pose.copy()
    pose_for_render[:3, 3] = pose[:3, 3] - pose[:3, :3] @ model_center
    return pose_for_render


def compute_mesh_diameter(mesh: trimesh.Trimesh) -> float:
    """
    计算mesh直径（米）

    Args:
        mesh: mesh对象（可以是中心化的或原始坐标系）

    Returns:
        mesh_diameter: mesh直径（米）
    """
    mesh_center = compute_mesh_center(mesh)
    mesh_diameter = np.max(np.linalg.norm(mesh.vertices - mesh_center, axis=1)) * 2
    return mesh_diameter


def transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """
    统一的3D点变换函数：将物体坐标系中的点变换到相机坐标系
    
    使用齐次坐标方法，与矩阵运算一致，更直观。
    
    Args:
        points: (N, 3) 物体坐标系中的3D点
        pose: (4, 4) 物体到相机的变换矩阵
    
    Returns:
        transformed_points: (N, 3) 相机坐标系中的3D点
    """
    if points.shape[0] == 0:
        return points
    # 转换为齐次坐标
    points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
    # 应用变换矩阵
    transformed_homo = (pose @ points_homo.T).T
    # 返回前3列（去除齐次坐标的1）
    return transformed_homo[:, :3]


def project_points_to_2d(points_3d: np.ndarray, K: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """
    统一的3D点投影函数：将物体坐标系中的点投影到图像平面
    
    Args:
        points_3d: (N, 3) 物体坐标系中的3D点
        K: (3, 3) 相机内参矩阵
        pose: (4, 4) 物体到相机的变换矩阵
    
    Returns:
        points_2d: (N, 2) 图像平面上的2D点
        depths: (N,) 深度值（可选）
    """
    # 先变换到相机坐标系
    points_cam = transform_points(points_3d, pose)
    # 投影到图像平面
    points_proj = (K @ points_cam.T).T
    points_2d = points_proj[:, :2] / points_proj[:, 2:3]
    return points_2d, points_cam[:, 2]


def render_rgbd(cad_model, object_pose, K, W, H):
    if type(object_pose) is np.ndarray:
        pose_tensor = torch.from_numpy(object_pose).float().to("cuda")
    else:
        pose_tensor = object_pose
    # pose_tensor= torch.from_numpy(object_pose).float().to("cuda")
    mesh_tensors = make_mesh_tensors(cad_model)
    glctx =  dr.RasterizeCudaContext()
    rgb_r, depth_r, normal_r = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=pose_tensor, context='cuda', get_normal=False, glctx=glctx, mesh_tensors=mesh_tensors, output_size=[H,W], use_light=True)
    rgb_r = rgb_r.squeeze()
    depth_r = depth_r.squeeze()
    mask_r = (depth_r > 0)
    return rgb_r, depth_r, mask_r





class PoseTracker:
    def __init__(self, dt=0.001):
        """
        Initialize a 6D pose tracker (position + orientation) with Kalman Filter
        dt: time step between measurements
        """
        # State: [x, y, z, vx, vy, vz, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
        # 12-dimensional state: position, velocity, orientation, angular rates
        self.kf = KalmanFilter(dim_x=12, dim_z=6)  
        
        # State transition matrix
        self.kf.F = np.zeros((12, 12))
        # Position and velocity
        self.kf.F[0:3, 0:3] = np.eye(3)
        self.kf.F[0:3, 3:6] = np.eye(3) * dt
        self.kf.F[3:6, 3:6] = np.eye(3)
        # Orientation and angular rates
        self.kf.F[6:9, 6:9] = np.eye(3)
        self.kf.F[6:9, 9:12] = np.eye(3) * dt
        self.kf.F[9:12, 9:12] = np.eye(3)
        
        # Measurement matrix (we measure position and orientation)
        self.kf.H = np.zeros((6, 12))
        self.kf.H[0:3, 0:3] = np.eye(3)  # Position measurements
        self.kf.H[3:6, 6:9] = np.eye(3)  # Orientation measurements
        
        # Measurement noise
        # self.kf.R = np.eye(6)
        # self.kf.R[0:3, 0:3] *= 0.1  # Position measurement noise
        # self.kf.R[3:6, 3:6] *= 0.2  # Orientation measurement noise
        
        # Process noise
        # pos_vel_noise = Q_discrete_white_noise(dim=2, dt=dt, var=0.1)
        # angle_rate_noise = Q_discrete_white_noise(dim=2, dt=dt, var=0.2)
        
        # self.kf.Q = np.zeros((12, 12))
        # # Apply noise to position-velocity states
        # for i in range(3):
        #     self.kf.Q[i*2:(i+1)*2, i*2:(i+1)*2] = pos_vel_noise
        # # Apply noise to orientation-angular rate states
        # for i in range(3):
        #     self.kf.Q[(i*2+6):(i*2+8), (i*2+6):(i*2+8)] = angle_rate_noise
        
        # Initial state covariance
        self.kf.P = np.eye(12) * 100000
        
        self.is_initialized = False
        
    def normalize_angles(self, angles):
        """
        Normalize angles to [-pi, pi]
        """
        return np.mod(angles + np.pi, 2 * np.pi) - np.pi
        
    def initialize(self, position, orientation):
        """
        Initialize the tracker with first pose measurement
        position: [x, y, z]
        orientation: [roll, pitch, yaw]
        """

        self.kf.x[0:3] = position
        self.kf.x[6:9] = self.normalize_angles(orientation)
    
        self.is_initialized = True

    def get_current_pose(self):
        pose=np.eye(4)
        pose[:3, 3] = np.squeeze(self.kf.x[0:3])
        pose[:3, :3] = Rotation.from_euler("xyz", np.squeeze(self.kf.x[6:9])).as_matrix()
        return pose

    def predict_next_pose(self):
        """
        Predict the next pose without updating the Kalman filter state.
        Returns: predicted position, orientation, velocity, and angular rates
        """
        if not self.is_initialized:
            raise ValueError("Tracker not initialized!")
        
        # Compute the predicted state using the state transition matrix
        predicted_state = self.kf.F @ self.kf.x
        
        # Normalize the predicted orientation angles
        predicted_state[6:9] = self.normalize_angles(predicted_state[6:9])
        
        return {
            'position': predicted_state[0:3],
            'orientation': predicted_state[6:9],
            'velocity': predicted_state[3:6],
            'angular_rates': predicted_state[9:12]
        }
        
    def update(self, measurement=None):
        """
        Update the state estimate. If measurement is None, predict without update
        measurement: [x, y, z, roll, pitch, yaw] or None during occlusion
        Returns: position and orientation estimates
        """
        if not self.is_initialized:
            raise ValueError("Tracker not initialized!")
            
        # Predict next state
        self.kf.predict()
        # Normalize orientation states
        self.kf.x[6:9] = self.normalize_angles(self.kf.x[6:9])
    
        # Update with measurement if available
        if measurement is not None:
            # Normalize measured orientation angles
            measurement[3:6] = self.normalize_angles(measurement[3:6])

            # Handle angle wrapping in measurement update
            innovation = measurement - self.kf.H @ self.kf.x
        
            innovation[3:6] = self.normalize_angles(innovation[3:6])
            
            # Custom update to handle angle wrapping
            PHT = self.kf.P @ self.kf.H.T
            S = self.kf.H @ PHT + self.kf.R
            K = PHT @ np.linalg.inv(S)
    
            self.kf.x = self.kf.x + K @ innovation
            assert self.kf.x.shape == (12,1)
            self.kf.P = (np.eye(12) - K @ self.kf.H) @ self.kf.P
            
        # Return current pose estimate
    
        return {
            'position': self.kf.x[0:3],
            'orientation': self.normalize_angles(self.kf.x[6:9]),
            'velocity': self.kf.x[3:6],
            'angular_rates': self.kf.x[9:12]
        }
    
    def get_uncertainty(self):
        """
        Return the uncertainty in position and orientation estimates
        """
        return {
            'position_std': np.sqrt(np.diag(self.kf.P)[0:3]),
            'orientation_std': np.sqrt(np.diag(self.kf.P)[6:9])
        }
    
def evaluate_metrics_fast(history_poses, reader, mesh, traj=False, **kwargs):
    """
    快速评估追踪性能，只计算不需要深度图的指标。
    history_poses: List of estimated poses at each frame (应该是中心化坐标系的)
    reader: DatasetReader object
    mesh: Trimesh mesh object (应该是中心化的mesh)
    traj: If True, return per-frame metrics
    **kwargs: 其他参数（model_center: 原始mesh的中心，用于转换GT pose）
    Returns: Average metrics and optionally per-frame metrics
    """
    data = {"ADD": 0, "ADD-S": 0, "rotation_error_deg": 0, "translation_error": 0}
    if traj:
        data2 = {"ADD": [], "ADD-S": [], "rotation_error_deg": [], "translation_error": []}
    
    # 检查mesh是否中心化（统一使用边界框中心，与FoundationPose一致）
    vertices = mesh.vertices if hasattr(mesh, 'vertices') else mesh
    max_xyz = vertices.max(axis=0)
    min_xyz = vertices.min(axis=0)
    mesh_center_check = (min_xyz + max_xyz) / 2
    is_centered = np.linalg.norm(mesh_center_check) < 0.001
    
    if not is_centered:
        logging.warning("⚠️  mesh可能不是中心化的，评估结果可能不准确")
    
    # 获取model_center（如果提供）
    model_center = kwargs.get('model_center', None)
    
    valid_count = 0
    for i, pose in enumerate(history_poses):
        gt_pose = reader.get_gt_pose(i)
        if gt_pose is None:
            if traj:
                # 对于没有GT的帧，添加None标记
                for key in data2:
                    data2[key].append(None)
            continue
        
        # 重要：GT pose已经是中心化坐标系的！
        # 根据 prepare_data.py 的逻辑：
        # 1. keypoints_3d 来自 keypoints_3d_model.json，这是中心化的（已减去 tooth_center）
        # 2. PnP 求解使用中心化的 keypoints_3d：cv2.solvePnP(keypoints_3d, keypoints_2d, K, None, ...)
        # 3. 验证时使用 project_3d_to_2d(keypoints_3d, RT, K, tooth_center=None)，说明RT是中心化坐标系的
        # 4. tooth_center = (min_xyz + max_xyz) / 2，与评估时使用的 model_center 计算方式完全相同
        #
        # 因此，GT pose 已经是中心化坐标系的，不需要转换！
        # 如果GT pose和EST pose的平移差异很大，可能是追踪误差本身很大，而不是坐标系不匹配
        gt_translation_norm = np.linalg.norm(gt_pose[:3, 3])
        est_translation_norm = np.linalg.norm(pose[:3, 3])
        
        # 仅记录警告，不进行转换
        # 注意：GT pose已经是中心化坐标系的（由prepare_data.py生成），不需要转换
        # 如果GT pose和EST pose的平移差异很大，可能是追踪误差本身很大
        if is_centered and abs(gt_translation_norm - est_translation_norm) > 0.03:
            if i == 0:  # 只在第一帧打印一次
                # 计算旋转误差和平移误差
                gt_R = gt_pose[:3, :3]
                est_R = pose[:3, :3]
                R_diff = np.dot(est_R, gt_R.T)
                trace_value = np.trace(R_diff)
                rotation_error_deg = np.degrees(np.arccos(np.clip((trace_value - 1) / 2, -1.0, 1.0)))
                translation_error = np.linalg.norm(gt_pose[:3, 3] - pose[:3, 3])
                
                logging.warning(f"[evaluate_metrics_fast()] ⚠️  第一帧分析：GT pose和EST pose差异较大")
                logging.warning(f"    平移差异: {abs(gt_translation_norm - est_translation_norm)*1000:.2f}mm")
                logging.warning(f"    GT pose平移: {gt_pose[:3, 3]*1000} mm (模长={gt_translation_norm*1000:.2f}mm)")
                logging.warning(f"    EST pose平移: {pose[:3, 3]*1000} mm (模长={est_translation_norm*1000:.2f}mm)")
                logging.warning(f"    旋转误差: {rotation_error_deg:.2f}°")
                logging.warning(f"    平移误差: {translation_error*1000:.2f}mm")
                logging.warning(f"    注意：GT pose已经是中心化坐标系的（由prepare_data.py生成）")
                logging.warning(f"    如果差异很大，可能是追踪误差本身较大，而不是坐标系问题")
        
        # 所有输入都应该是中心化坐标系的
        # history_poses已经是中心化坐标系的
        # mesh也是中心化的
        # GT pose现在也应该是中心化坐标系的
        tmp_data = evaluate_pose_fast(gt_pose, pose, mesh)
        if tmp_data is None:
            if traj:
                for key in data2:
                    data2[key].append(None)
            continue
        
        valid_count += 1
        for key in tmp_data:
            data[key] += tmp_data[key]
            if traj:
                data2[key].append(tmp_data[key])
    
    if valid_count == 0:
        logging.warning("没有有效的ground truth位姿用于评估")
        if traj:
            return data, data2
        return data
    
    for key in data:
        data[key] /= valid_count
    
    if traj:
        return data, data2
    return data


def evaluate_metrics(history_poses,reader, mesh, traj=False):
    """
    Evaluate the tracking performance using the ground truth poses.
    history_poses: List of estimated poses at each frame
    reader: DatasetReader object
    Returns: List of errors (rotation, translation) for each frame
    """
    # errors = []
    vertices = mesh.vertices
    pairwise_distances = cdist(vertices, vertices)  # Use scipy.spatial.distance.cdist
    diameter_exact = np.max(pairwise_distances)
    data={"ADD":0, "ADD-S":0,  "rotation_error_deg":0, "translation_error":0, "mspd":0,"mssd":0, "recall":0, "AR_mspd":0, "AR_mssd":0, "AR_vsd":0}
    if traj:
        data2={"ADD":[], "ADD-S":[],  "rotation_error_deg":[], "translation_error":[], "mspd":[],"mssd":[], "recall":[], "AR_mspd":[], "AR_mssd":[], "AR_vsd":[]}
    valid_count = 0
    for i, pose in enumerate(history_poses):
        gt_pose = reader.get_gt_pose(i)
        if gt_pose is None:
            if traj:
                # 对于没有GT的帧，添加None标记
                for key in data2:
                    data2[key].append(None)
            continue
        tmp_data= evaluate_pose(gt_pose, pose, mesh, diameter_exact, reader.K)
        valid_count += 1
        for key in tmp_data:
            data[key]+=tmp_data[key]
            if traj:
                data2[key].append(tmp_data[key])
    
    if valid_count == 0:
        logging.warning("没有有效的ground truth位姿用于评估")
        if traj:
            return data, data2
        return data
    
    for key in data:
        data[key]/=valid_count
    if traj:
        return data,data2
    return data
    
def demo_tracking():
    """
    Demonstrate tracker usage with simulated occlusion
    """
    # Create tracker
    tracker = PoseTracker()
    
    # Initialize with first position
    initial_pos = np.array([0., 0., 0.])
    tracker.initialize(initial_pos)
    
    # Simulate some measurements with occlusion
    measurements = [
        [0.1, 0.1, 0.1],    # Visible
        [0.2, 0.2, 0.2],    # Visible
        None,               # Occluded
        None,               # Occluded
        [0.5, 0.5, 0.5]     # Visible again
    ]
    
    positions = []
    uncertainties = []
    
    for measurement in measurements:
        pos = tracker.update(measurement)
        uncertainty = tracker.get_position_uncertainty()
        
        positions.append(pos)
        uncertainties.append(uncertainty)
        
        status = "OCCLUDED" if measurement is None else "VISIBLE"
        print(f"Status: {status}")
        print(f"Estimated position: {pos}")
        print(f"Position uncertainty: {uncertainty}\n")
        
    return positions, uncertainties
import numpy as np




def render_cad_depth_nvidia(pose, mesh_model, K, w=640, h=480):
    """
    Render depth image from a CAD model using the given camera pose and intrinsic matrix.

    Args:
        pose (np.ndarray): 4x4 camera pose matrix (world to camera transformation).
        mesh_model (np.ndarray): Nx3 array of 3D mesh vertices.
        K (np.ndarray): 3x3 camera intrinsic matrix.
        w (int): Width of the output depth image.
        h (int): Height of the output depth image.

    Returns:
        np.ndarray: Depth image of size (h, w).
    """
    pose_tensor= torch.from_numpy(pose).float().to("cuda")
    mesh_tensors = make_mesh_tensors(mesh_model)
    glctx =  dr.RasterizeCudaContext()
    depth_r= nvdiffrast_render_depthonly(K=K, H=h, W=w, ob_in_cams=pose_tensor, context='cuda', glctx=glctx, mesh_tensors=mesh_tensors, output_size=[h,w])
    depth_r = depth_r.squeeze()
    return depth_r.cpu().numpy()

def render_cad_depth(pose, mesh_model, K, w=640, h=480):
    """
    Render a depth map using a CAD model and its pose.
    使用 nvdiffrast 进行基于面片的可靠渲染，而不是只投影顶点。
    
    重要：pose和mesh_model必须使用相同的坐标系！
    - 如果pose是中心化坐标系的，mesh_model也必须是中心化的
    - 如果pose是原始坐标系的，mesh_model也必须是原始坐标系的
    
    Parameters:
    pose: 4x4 numpy array - Transformation matrix（物体坐标系到相机坐标系）
    mesh_model: Trimesh object - CAD model（必须与pose使用相同的坐标系）
    K: 3x3 numpy array - Camera intrinsic matrix
    w, h: int - Width and height of the depth image

    Returns:
    depth_map: numpy array of shape (h, w) - Rendered depth map
    
    Raises:
    ValueError: 如果检测到坐标系不匹配（如果mesh中心距离原点较远，但pose是中心化坐标系的）
    """
    import logging
    
    # 检查图像尺寸：nvdiffrast要求宽高必须能被8整除
    w_original, h_original = w, h
    w_adjusted = ((w + 7) // 8) * 8  # 向上取整到8的倍数
    h_adjusted = ((h + 7) // 8) * 8  # 向上取整到8的倍数
    
    # 如果尺寸需要调整，同时调整K矩阵以匹配新尺寸
    K_adjusted = K.copy() if (w_adjusted != w or h_adjusted != h) else K
    if w_adjusted != w or h_adjusted != h:
        scale_x = w_adjusted / w_original
        scale_y = h_adjusted / h_original
        K_adjusted = K.copy()
        K_adjusted[0, 0] *= scale_x  # fx
        K_adjusted[1, 1] *= scale_y  # fy
        K_adjusted[0, 2] *= scale_x  # cx
        K_adjusted[1, 2] *= scale_y  # cy
        # 图像尺寸调整信息不输出
        w, h = w_adjusted, h_adjusted
    
    # 移除运行时检查，避免误报
    
    try:
        # 使用 nvdiffrast 进行可靠的面片渲染（GPU加速）
        # 如果图像尺寸被调整了，使用调整后的K矩阵
        depth = render_cad_depth_nvidia(pose, mesh_model, K_adjusted, w, h)
        # 如果图像尺寸被调整了，需要调整回原始尺寸
        if w != w_original or h != h_original:
            import cv2
            depth = cv2.resize(depth, (w_original, h_original), interpolation=cv2.INTER_LINEAR)
        return depth
    except Exception as e:
        # 如果 GPU 渲染失败，使用 CPU 回退方法（基于三角形光栅化）
        logging.warning(f"[render_cad_depth] GPU渲染失败，使用CPU回退方法: {e}")
        # CPU方法不需要调整尺寸（不依赖nvdiffrast），使用原始尺寸
        return _render_cad_depth_cpu_triangles(pose, mesh_model, K, w_original, h_original)


def _render_cad_depth_cpu_triangles(pose, mesh_model, K, w=640, h=480):
    """
    使用CPU进行基于三角形的深度图渲染（回退方法）
    通过扫描线算法渲染每个三角形面片
    """
    import cv2
    
    vertices = np.array(mesh_model.vertices)
    faces = np.array(mesh_model.faces)
    
    # Transform vertices to camera space (使用统一的变换函数)
    transformed_vertices = transform_points(vertices, pose)
    
    # Project vertices to 2D (使用统一的投影函数)
    projected_2d, _ = project_points_to_2d(vertices, K, pose)
    depths = transformed_vertices[:, 2]
    
    # Initialize depth map with infinity
    depth_map = np.full((h, w), np.inf, dtype=np.float32)
    
    # Render each triangle
    for face in faces:
        v0, v1, v2 = face
        
        # Skip triangles behind camera
        if depths[v0] <= 0 or depths[v1] <= 0 or depths[v2] <= 0:
            continue
        
        # Get 2D coordinates and depths of triangle vertices
        pts_2d = np.array([
            projected_2d[v0],
            projected_2d[v1],
            projected_2d[v2]
        ]).astype(np.float32)
        
        tri_depths = np.array([depths[v0], depths[v1], depths[v2]])
        
        # Get bounding box
        min_x = max(0, int(np.floor(pts_2d[:, 0].min())))
        max_x = min(w - 1, int(np.ceil(pts_2d[:, 0].max())))
        min_y = max(0, int(np.floor(pts_2d[:, 1].min())))
        max_y = min(h - 1, int(np.ceil(pts_2d[:, 1].max())))
        
        if min_x > max_x or min_y > max_y:
            continue
        
        # Compute barycentric coordinates for each pixel in bounding box
        # Using vectorized approach for efficiency
        v0_2d = pts_2d[0]
        v1_2d = pts_2d[1]
        v2_2d = pts_2d[2]
        
        # Edge vectors
        v0v1 = v1_2d - v0_2d
        v0v2 = v2_2d - v0_2d
        
        # Precompute denominator
        denom = v0v1[0] * v0v2[1] - v0v1[1] * v0v2[0]
        if abs(denom) < 1e-8:
            continue  # Degenerate triangle
        
        inv_denom = 1.0 / denom
        
        # Iterate over pixels in bounding box
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                p = np.array([x + 0.5, y + 0.5])
                v0p = p - v0_2d
                
                # Barycentric coordinates
                u = (v0p[0] * v0v2[1] - v0p[1] * v0v2[0]) * inv_denom
                v = (v0v1[0] * v0p[1] - v0v1[1] * v0p[0]) * inv_denom
                
                # Check if point is inside triangle
                if u >= 0 and v >= 0 and (u + v) <= 1:
                    # Interpolate depth
                    w0 = 1 - u - v
                    depth = w0 * tri_depths[0] + u * tri_depths[1] + v * tri_depths[2]
                    
                    # Z-buffer test
                    if depth < depth_map[y, x]:
                        depth_map[y, x] = depth
    
    # Replace inf with 0 (background)
    depth_map[depth_map == np.inf] = 0
    return depth_map
    
def render_cad_mask(pose, mesh_model, K, w=640, h=480):
    """
    Renders the binary mask of the object based on its pose, CAD model, and camera parameters.

    Args:
        pose (np.ndarray): 4x4 transformation matrix of the object's pose.
        mesh_model: Mesh object containing vertices of the CAD model.
        K (np.ndarray): 3x3 intrinsic matrix of the camera.
        w (int): Image width.
        h (int): Image height.

    Returns:
        np.ndarray: Binary mask of the object (1 for object pixels, 0 for background).
    """
    # Load the vertices from the mesh model
    vertices = np.array(mesh_model.vertices)
    sample_indices = np.random.choice(len(vertices), size=500, replace=False)
    vertices = vertices[sample_indices]

    # Transform vertices with the object pose (使用统一的变换函数)
    transformed_vertices = transform_points(vertices, pose)

    # Project vertices to the 2D plane using the intrinsic matrix K (使用统一的投影函数)
    projected_points, _ = project_points_to_2d(vertices, K, pose)

    # Create a polygon from the projected 2D points
    polygon = np.int32(projected_points).reshape((-1, 1, 2))

    # Initialize a blank mask and draw the polygon
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], color=1)

    return mask



def to_homo(pts):
    '''
    @pts: (N,3 or 2) will homogeneliaze the last dimension
    '''
    assert len(pts.shape)==2, f'pts.shape: {pts.shape}'
    homo = np.concatenate((pts, np.ones((pts.shape[0],1))),axis=-1)
    return homo

def compute_iou(mask1, mask2):
    """
    Compute the Intersection over Union (IoU) between two binary masks.

    Parameters:
    - mask1: np.ndarray, first binary mask.
    - mask2: np.ndarray, second binary mask.

    Returns:
    - iou: float, the IoU value.
    """
    # Ensure the masks are binary
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    # Compute intersection and union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    # Calculate IoU
    iou = intersection / union if union > 0 else 0.0
    return iou


def compute_error(pose1, pose2):
    """
    Computes the rotation error (in degrees) and translation error (in meters) between two poses.
    
    Parameters:
    - pose1: (4x4 numpy array) Transformation matrix representing pose 1.
    - pose2: (4x4 numpy array) Transformation matrix representing pose 2.
    
    Returns:
    - rotation_error: The angular difference in degrees between the two poses.
    - translation_error: The Euclidean distance between the translations of the two poses.
    """
    # Extract rotation matrices (upper-left 3x3 submatrix)
    R1 = pose1[:3, :3]
    R2 = pose2[:3, :3]
    
    # Extract translation vectors (rightmost 3 elements of the 4th column)
    t1 = pose1[:3, 3]
    t2 = pose2[:3, 3]
    
    # Compute the relative rotation matrix R_rel = R1_inv * R2
    R_rel = np.dot(R1.T, R2)
    
    # Compute the rotation error as the angle of the relative rotation
    # trace(R_rel) = 1 + 2*cos(theta), where theta is the rotation angle
    trace_R_rel = np.trace(R_rel)
    theta = np.arccos(np.clip((trace_R_rel - 1) / 2.0, -1.0, 1.0))  # theta in radians
    
    # Convert the rotation error to degrees
    rotation_error = np.degrees(theta)
    
    # Compute the translation error as the Euclidean distance between t1 and t2
    translation_error = np.linalg.norm(t1 - t2)
    
    return rotation_error, translation_error

        
def get_3d_points(depth_image, keypoints, camera_matrix):
    points_3d = []
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    for kp in keypoints:
        try:
            u, v = int(kp.pt[0]), int(kp.pt[1])
        except:
            u,v=int(kp[0]), int(kp[1])
        z = depth_image[v, u] # assuming depth is in millimeters
        # z = depth_image[u, v] # assuming depth is in millimeters
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
    
        points_3d.append([x, y, z])
    
    return np.array(points_3d)



def get_pose_icp(self, pointcloud1, pointcloud2):
    """
    Perform ICP (Iterative Closest Point) registration to compute the relative transformation 
    from pointcloud1 to pointcloud2.

    :param pointcloud1: Source point cloud as a numpy array of shape (N, 3)
    :param pointcloud2: Target point cloud as a numpy array of shape (M, 3)
    :return: 4x4 transformation matrix representing the transformation from pointcloud1 to pointcloud2
    """
    
    # Convert numpy arrays to Open3D PointCloud objects
    
    print("Pointcloud1: ", pointcloud1)
    print("Pointcloud2: ", pointcloud2)
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    
    source.points = o3d.utility.Vector3dVector(pointcloud1)
    target.points = o3d.utility.Vector3dVector(pointcloud2)
    
    # Estimate normals (required for point-to-plane ICP)
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # Perform point-to-plane ICP (this usually gives better results than point-to-point)
    threshold = 0.02  # Distance threshold for matching points
    initial_transformation = np.eye(4)  # No initial transformation
    
    icp_result = o3d.pipelines.registration.registration_icp(
        source, 
        target, 
        threshold, 
        initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    
    # Get the transformation matrix from the ICP result
    transformation = icp_result.transformation

    return transformation



#"vsd_taus": list(np.arange(0.05, 0.51, 0.05)),
def vsd(
    depth_gt,
    depth_est,
    K,
    delta,
    taus,
    diameter,
    cost_type="step",
):
    """Visible Surface Discrepancy -- by Hodan, Michel et al. (ECCV 2018).

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param depth_test: hxw ndarray with the test depth image.
    :param K: 3x3 ndarray with an intrinsic camera matrix.
    :param delta: Tolerance used for estimation of the visibility masks.
    :param taus: A list of misalignment tolerance values.
    :param normalized_by_diameter: Whether to normalize the pixel-wise distances
        by the object diameter.
    :param diameter: Object diameter.
    :param renderer: Instance of the Renderer class (see renderer.py).
    :param obj_id: Object identifier.
    :param cost_type: Type of the pixel-wise matching cost:
        'tlinear' - Used in the original definition of VSD in:
            Hodan et al., On Evaluation of 6D Object Pose Estimation, ECCVW'16
        'step' - Used for SIXD Challenge 2017 onwards.
    :return: List of calculated errors (one for each misalignment tolerance).
    """
    # Render depth images of the model in the estimated and the ground-truth pose.
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # Convert depth images to distance images.
    dist_gt = misc.depth_im_to_dist_im_fast(depth_gt, K)
    dist_est = misc.depth_im_to_dist_im_fast(depth_est, K)

    # Visibility mask of the model in the ground-truth pose.
    visib_gt = visibility.estimate_visib_mask_gt(
        dist_gt,dist_gt, delta, visib_mode="bop19"
    )

    # Visibility mask of the model in the estimated pose.
    visib_est = visibility.estimate_visib_mask_est(
        dist_gt, dist_est, visib_gt, delta, visib_mode="bop19"
    )

    # Intersection and union of the visibility masks.
    visib_inter = np.logical_and(visib_gt, visib_est)
    visib_union = np.logical_or(visib_gt, visib_est)
    

    visib_union_count = visib_union.sum()
    visib_comp_count = visib_union_count - visib_inter.sum()

    # Pixel-wise distances.
    dists = np.abs(dist_gt[visib_inter] - dist_est[visib_inter])
    normalized_by_diameter = True
    # Normalization of pixel-wise distances by object diameter.
    if normalized_by_diameter:
        dists /= diameter

    # Calculate VSD for each provided value of the misalignment tolerance.
    if visib_union_count == 0:
        errors = [1.0] * len(taus)
    else:
        errors = []
        for tau in taus:
            # Pixel-wise matching cost.
            if cost_type == "step":
                costs = dists >= tau
            elif cost_type == "tlinear":  # Truncated linear function.
                costs = dists / tau
                costs[costs > 1.0] = 1.0
            else:
                raise ValueError("Unknown pixel matching cost.")

            e = (np.sum(costs) + visib_comp_count) / float(visib_union_count)
            errors.append(e)

    return errors


#   "vsd": [0.3],
#         "mssd": [0.2],
#         "mspd": [10],

# thresholds=np.linspace(0.05, 0.5, 10)
def quick_verify_coordinate_system(gt_pose, est_pose, mesh, model_center=None):
    """
    快速验证GT pose和EST pose是否使用相同的坐标系（仅用于第一帧快速检查）
    
    注意：现在整个系统统一使用中心化坐标系，GT pose和EST pose都应该是中心化坐标系的。
    这个函数主要用于验证坐标系一致性，不再尝试多种转换方法。
    
    Args:
        gt_pose: GT pose矩阵（应该是中心化坐标系的）
        est_pose: EST pose矩阵（应该是中心化坐标系的）
        mesh: mesh对象（应该是中心化的）
        model_center: 原始mesh的中心（已废弃，保留仅用于兼容性）
    
    Returns:
        dict: 包含验证结果的字典
    """
    vertices = mesh.vertices if hasattr(mesh, 'vertices') else mesh
    # 统一使用边界框中心，与FoundationPose一致
    max_xyz = vertices.max(axis=0)
    min_xyz = vertices.min(axis=0)
    mesh_center = (min_xyz + max_xyz) / 2
    is_centered = np.linalg.norm(mesh_center) < 0.001
    
    # 变换点云
    gt_transformed = transform_points(vertices, gt_pose)
    est_transformed = transform_points(vertices, est_pose)
    
    # 计算点云中心距离
    gt_center = gt_transformed.mean(axis=0)
    est_center = est_transformed.mean(axis=0)
    center_distance = np.linalg.norm(gt_center - est_center)
    
    # 计算ADD（用于对比）
    add_value = np.mean(np.linalg.norm(gt_transformed - est_transformed, axis=1))
    
    result = {
        'is_centered': is_centered,
        'center_distance_mm': center_distance * 1000,
        'add_mm': add_value * 1000,
        'gt_center': gt_center,
        'est_center': est_center,
        'gt_translation_norm': np.linalg.norm(gt_pose[:3, 3]),
        'est_translation_norm': np.linalg.norm(est_pose[:3, 3]),
    }
    
    # 注意：不再尝试多种转换方法，因为系统已统一使用中心化坐标系
    # 如果GT pose和EST pose都是中心化坐标系的，ADD应该很小
    
    return result


def evaluate_pose_fast(gt_pose, est_pose, mesh):
    """
    快速评估位姿误差，只计算不需要深度图的指标（ADD, ADD-S, 旋转误差, 平移误差）。
    不计算VSD、MSPD、MSSD等需要深度图的指标，速度更快。
    
    Args:
        gt_pose (np.ndarray): 4x4 ground truth pose matrix
        est_pose (np.ndarray): 4x4 estimated pose matrix
        mesh: Trimesh mesh object
    
    Returns:
        dict: Dictionary containing fast evaluation metrics
    """
    if gt_pose is None or est_pose is None:
        return None
    
    def compute_add(gt_points, est_points):
        """Compute ADD metric (average distance between corresponding points)."""
        return np.mean(np.linalg.norm(gt_points - est_points, axis=1))

    def compute_adds_fast(gt_points, est_points):
        """Compute ADD-S metric using KDTree for fast nearest neighbor search."""
        from scipy.spatial import cKDTree
        tree = cKDTree(est_points)
        distances, _ = tree.query(gt_points, k=1)
        return np.mean(distances)

    # Transform model points using ground truth and estimated poses
    vertices = mesh.vertices if hasattr(mesh, 'vertices') else mesh
    
    # 检查mesh是否中心化（统一使用边界框中心，与FoundationPose一致）
    max_xyz = vertices.max(axis=0)
    min_xyz = vertices.min(axis=0)
    mesh_center = (min_xyz + max_xyz) / 2
    is_centered = np.linalg.norm(mesh_center) < 0.001  # 如果中心接近原点，认为是中心化的
    
    # 如果mesh是中心化的，但GT pose和EST pose的平移差异很大，可能是坐标系不匹配
    # 需要确保GT pose和EST pose都使用相同的坐标系
    gt_translation_norm = np.linalg.norm(gt_pose[:3, 3])
    est_translation_norm = np.linalg.norm(est_pose[:3, 3])
    # 计算mesh直径（使用边界框中心作为参考点）
    mesh_diameter = np.max(np.linalg.norm(vertices - mesh_center, axis=1)) * 2
    
    # 重要：GT pose已经是中心化坐标系的（由prepare_data.py生成）
    # 如果mesh是中心化的，但GT pose和EST pose的平移差异很大（>30mm），
    # 可能是追踪误差本身很大，而不是坐标系不匹配
    # 仅记录警告，不进行转换
    if is_centered and abs(gt_translation_norm - est_translation_norm) > 0.03:
        # 使用静态变量确保只打印一次
        if not hasattr(evaluate_pose_fast, '_coordinate_warning_printed'):
            logging.warning(f"[evaluate_pose_fast()] ⚠️  GT pose和EST pose的平移差异较大（{abs(gt_translation_norm - est_translation_norm)*1000:.2f}mm）")
            logging.warning(f"    mesh直径={mesh_diameter*1000:.2f}mm")
            logging.warning(f"    GT pose平移: {gt_pose[:3, 3]*1000} mm (模长={gt_translation_norm*1000:.2f}mm)")
            logging.warning(f"    EST pose平移: {est_pose[:3, 3]*1000} mm (模长={est_translation_norm*1000:.2f}mm)")
            logging.warning(f"    注意：GT pose已经是中心化坐标系的（由prepare_data.py生成），不需要转换")
            logging.warning(f"    如果差异很大，可能是追踪误差本身较大，而不是坐标系问题")
            evaluate_pose_fast._coordinate_warning_printed = True
    
    gt_transformed = transform_points(vertices, gt_pose)
    est_transformed = transform_points(vertices, est_pose)
    
    # 调试信息：检查变换后的点云（仅在检测到坐标系不匹配时打印）
    if is_centered and abs(gt_translation_norm - est_translation_norm) > 0.03:
        # 使用静态变量确保只打印一次
        if not hasattr(evaluate_pose_fast, '_debug_printed'):
            logging.warning(f"[evaluate_pose_fast()] 变换后的点云统计（坐标系不匹配时）:")
            logging.warning(f"    GT变换后点云: min={gt_transformed.min(axis=0)*1000}, max={gt_transformed.max(axis=0)*1000}, mean={gt_transformed.mean(axis=0)*1000} mm")
            logging.warning(f"    EST变换后点云: min={est_transformed.min(axis=0)*1000}, max={est_transformed.max(axis=0)*1000}, mean={est_transformed.mean(axis=0)*1000} mm")
            logging.warning(f"    点云中心差异: {(gt_transformed.mean(axis=0) - est_transformed.mean(axis=0))*1000} mm")
            logging.warning(f"    点云中心距离: {np.linalg.norm(gt_transformed.mean(axis=0) - est_transformed.mean(axis=0))*1000:.2f} mm")
            evaluate_pose_fast._debug_printed = True

    # Compute ADD and ADD-S
    add_value = compute_add(gt_transformed, est_transformed)
    adds_value = compute_adds_fast(gt_transformed, est_transformed)

    # Extract rotation error
    gt_R = gt_pose[:3, :3]
    est_R = est_pose[:3, :3]
    R_diff = np.dot(est_R, gt_R.T)
    trace_value = np.trace(R_diff)
    theta_rad = np.arccos(np.clip((trace_value - 1) / 2, -1.0, 1.0))  # Avoid numerical errors
    rotation_error = np.degrees(theta_rad)  # Convert to degrees

    # Extract translation error
    translation_error = np.linalg.norm(gt_pose[:3, 3] - est_pose[:3, 3])
    
    return {
        'ADD': add_value,
        'ADD-S': adds_value,
        'rotation_error_deg': rotation_error,
        'translation_error': translation_error,
    }


def evaluate_pose(gt_pose, est_pose, mesh, diameter,K ):
    """
    Evaluate 6D pose estimation performance using ADD and ADD-S metrics.
    
    Args:
        gt_pose (np.ndarray): 4x4 ground truth pose matrix
        est_pose (np.ndarray): 4x4 estimated pose matrix
        model_points (np.ndarray): Nx3 array of 3D model points
        diameter (float): diameter of the object
        thresholds (np.ndarray): distance thresholds for AUC computation
    
    Returns:
        dict: Dictionary containing various evaluation metrics
    """

    def compute_add(gt_points, est_points):
        """Compute ADD metric (average distance between corresponding points)."""
        return np.mean(np.linalg.norm(gt_points - est_points, axis=1))

    def compute_adds(gt_points, est_points):
        """Compute ADD-S metric (average distance to nearest neighbor)."""
        distances = np.zeros((gt_points.shape[0],))
        for i, gt_point in enumerate(gt_points):
            distances[i] = np.min(np.linalg.norm(gt_point - est_points, axis=1))
        return np.mean(distances)

    # Transform model points using ground truth and estimated poses
    gt_transformed = transform_points(mesh.vertices , gt_pose)
    est_transformed = transform_points(mesh.vertices , est_pose)

    # Compute ADD and ADD-S
    add_value = compute_add(gt_transformed, est_transformed)
    adds_value = compute_adds(gt_transformed, est_transformed)

    # # Compute success rates and AUC for different thresholds
    # add_success_rates = []
    # adds_success_rates = []
    # # print("thresholds: ", thresholds)
    # # print("add_value: ", add_value)
    # # print("adds_value: ", adds_value)
    
    # for threshold in thresholds:
    #     # Normalize threshold by object diameter

    #     normalized_threshold = threshold * diameter
        
    #     # Compute ADD success rate
    #     add_success = (add_value < normalized_threshold)
    #     add_success_rates.append(float(add_success))
        
    #     # Compute ADD-S success rate
    #     adds_success = (adds_value < normalized_threshold)
    #     adds_success_rates.append(float(adds_success))
    # print("add_success_rates: ", add_success_rates)
    # print("diameter: ", diameter)
    # Compute AUC (normalize thresholds to 0-1 range for AUC computation)
    # normalized_thresholds = thresholds / thresholds[-1]
    # add_auc = auc(normalized_thresholds, add_success_rates)
    # adds_auc = auc(normalized_thresholds, adds_success_rates)

    # Extract rotation error
    gt_R = gt_pose[:3, :3]
    est_R = est_pose[:3, :3]
    R_diff = np.dot(est_R, gt_R.T)
    trace_value = np.trace(R_diff)
    theta_rad = np.arccos(np.clip((trace_value - 1) / 2, -1.0, 1.0))  # Avoid numerical errors
    rotation_error = np.degrees(theta_rad)  # Convert to degrees
    depth_gt=render_cad_depth_nvidia(gt_pose, mesh, K)
    depth_est=render_cad_depth_nvidia(est_pose, mesh, K)

    translation_error = np.linalg.norm(gt_pose[:3, 3] - est_pose[:3, 3])
    taus=list(np.arange(0.05, 0.51, 0.05))
    delta=15
    # taus=[0.5]
    theta=np.arange(0.05, 0.51, 0.05)

    vsd_errors=vsd(depth_gt, depth_est, K, delta, taus, diameter, cost_type="step")
    AR_vsd =0
    for err in vsd_errors:
        if err<=0.3:
            AR_vsd+=1
    AR_vsd/=10
    # vsd_error=np.mean(vsd_errors)
    error_mspd=pose_error.mspd(est_pose[:3, :3], est_pose[:3, 3].reshape(3,1),gt_pose[:3, :3], gt_pose[:3, 3].reshape(3,1), K, mesh.vertices, [{"R": np.eye(3), "t": np.array([[0, 0, 0]]).T}])
    error_mssd=pose_error.mssd(est_pose[:3, :3], est_pose[:3, 3].reshape(3,1),gt_pose[:3, :3], gt_pose[:3, 3].reshape(3,1), mesh.vertices, [{"R": np.eye(3), "t": np.array([[0, 0, 0]]).T}])
    AR_mspd =0 
    AR_mssd =0
    for th in np.arange(5,51,5):
        if error_mspd<=th:
            AR_mspd+=1
    AR_mspd/=10
    for th in theta*diameter:
        if error_mssd<=th:
            AR_mssd+=1
    AR_mssd/=10
    
    # Extract translation error
    translation_error = np.linalg.norm(gt_pose[:3, 3] - est_pose[:3, 3])
    res={
        'ADD': add_value,
        'ADD-S': adds_value,
        'rotation_error_deg': rotation_error,
        'translation_error': translation_error,
        "recall": (AR_mspd+AR_mssd+AR_vsd)/3,
        "mspd": error_mspd,
        "mssd": error_mssd,
        "AR_vsd": AR_vsd,
        "AR_mspd": AR_mspd,
        "AR_mssd": AR_mssd    
        # 'add_success_rates': add_success_rates,
        # 'adds_success_rates': adds_success_rates,
        # 'thresholds': thresholds
    }
    print("res: ", res)
    return res
    

# def evaluate_pose_bop(gt_pose, est_pose, K, model_points):
#     syms=[{"R": np.eye(3), "t": np.array([[0, 0, 0]]).T}]
#     Rgt=gt_pose[:3, :3]
#     tgt=gt_pose[:3, 3]
#     Rest=est_pose[:3, :3]
#     test=est_pose[:3, 3]
#     err_mspd = pose_error.compute_mspd( Rest, test,Rgt, tgt,K, model_points, syms)
#     err_mssd = pose_error.compute_mssd( Rest, test,Rgt, tgt, model_points, syms)

def save_poses_to_txt(file_path, poses):
    """
    Save a list of 4x4 np.matrix to a text file.
    
    Parameters:
    - poses: A list of np.matrix (4x4 matrices).
    - file_path: The file path where the matrices will be saved.
    """
    # Verify that all poses are 4x4 matrices
    for pose in poses:
        if pose.shape != (4, 4):
            raise ValueError(f"Each pose must be a 4x4 matrix. Found shape: {pose.shape}")
    
    # Convert np.matrix objects to np.ndarray (for easier handling)
    poses_array = [np.array(pose) for pose in poses]
    
    # Stack them into a single numpy array
    stacked_poses = np.stack(poses_array)
    
    # Save to a text file (flatten the matrices into rows of 16 values)
    np.savetxt(file_path, stacked_poses.reshape(-1, 16), delimiter=' ')
    print(f"Saved {len(poses)} poses to {file_path}")

def read_poses_from_txt(file_path):
    """
    Read a list of 4x4 np.matrix from a text file.
    
    Parameters:
    - file_path: The file path from which the matrices will be read.
    
    Returns:
    - A list of np.matrix (4x4 matrices).
    """
    # Load the flattened array from the text file
    loaded_poses = np.loadtxt(file_path)
    
    # Verify that the number of elements is a multiple of 16
    if loaded_poses.size % 16 != 0:
        raise ValueError("The file does not contain a valid number of elements for 4x4 matrices.")
    
    # Reshape the array into 4x4 matrices
    poses = []
    for i in range(0, loaded_poses.shape[0], 16):
        pose = loaded_poses[i:i+16].reshape(4, 4)
        poses.append(np.matrix(pose))  # Convert back to np.matrix
    
    print(f"Loaded {len(poses)} poses from {file_path}")
    return poses




def binary_search_depth(est,mesh, rgb, mask, K, depth_min=0.5, depth_max=2,w=640, h=480, debug=False, ycb=False, depth_input=None, iteration=5):
    low=depth_min
    high=depth_max
    last_depth=np.inf
    while low <= high:
        mid= (low+high)/2
        depth_gueess=mid
        depth_zero=np.zeros_like(mask)
        # depth= np.ones_like(mask)*mid
        pose= est.register(K, rgb, depth_zero, mask, iteration, rough_depth_guess=depth_gueess)
        
        mask_r= render_cad_mask( pose, mesh, K, w, h)
        current_depth= pose[2,3]
        if np.abs(current_depth-last_depth)<1e-2:
            print("depth not change")
            break
        last_depth=current_depth
        if debug:
            rgb_r, depth_r, mask_r2= render_rgbd(mesh, pose, K, w, h)
            
            plt.subplot(1, 2, 1)
            plt.imshow(rgb_r.cpu().numpy())
            plt.axis('off')  # Turn off the axes for the first subplot

            plt.subplot(1, 2, 2)
            rgb_copy = rgb.copy()
            # rgb_copy[mask == 0] = 0
            plt.imshow(rgb_copy)
            plt.axis('off')  # Turn off the axes for the second subplot
            # rgb_save=rgb_r.cpu().numpy()*255
            # #rgbtobgr
            # rgb_save= rgb_save[...,::-1]
            # cv2.imwrite(f"tmp/debug_{mid}.png", rgb_save)
            plt.savefig(f"tmp/debug_{mid}.png")
            plt.close()  # Close the figure to free resources
        if abs(high-low)<0.001:
            break
        
        # for ycb dataset
        if ycb:     
            bounding_box= cv2.boundingRect(mask_r)
            area= bounding_box[2]*bounding_box[3]
        else:
            area=np.sum(mask_r)
        
        # if  area-np.sum(mask)<10:
        #     print("area close")
        #     return pose
        if area>np.sum(mask):
            low=mid
        elif area<np.sum(mask):
            high=mid
    # depth_zero= np.ones_like(mask)
    # for i in range(10):
    #     print(i)
    #     pose=est.track_one(rgb, depth_zero, K,1)
    return pose

            

def binary_search_scale(est,mesh, rgb,depth, mask, K, scale_min=0.2, scale_max=5,w=640, h=480, debug=False):
    low=scale_min
    high=scale_max
    while low<=high:
        mid= (low+high)/2
        mesh_c=mesh.copy()
        mesh_c.apply_scale(mid)
        est.reset_object(model_pts=mesh_c.vertices.copy(), model_normals=mesh_c.vertex_normals.copy(), mesh=mesh_c)
        pose= est.register(K, rgb,depth, mask, 5)
        # rgb_r, depth_r, mask_r= render_rgbd(mesh_c, pose, K, 640, 480)
        mask_r= render_cad_mask( pose, mesh_c, K, w, h)
        binary_mask = (mask_r > 0).astype(np.uint8)
    
        # Calculate the bounding box
        x, y, width, height = cv2.boundingRect(binary_mask)
        
        # Calculate the area of the bounding box
        # area = width * height
        area= np.sum(mask_r)
        if debug:
            rgb_r, depth_r, mask_r= render_rgbd(mesh_c, pose, K, 640, 480)
            plt.subplot(1, 2, 1)
            plt.imshow(rgb_r)
            plt.subplot(1, 2, 2)
            rgb_copy= rgb.copy()
            rgb_copy[mask==0]=0
            plt.imshow(rgb_copy)
            plt.savefig(f"tmp/debug_{mid}.png")
        if abs(high-low)<0.01:
            break
        if  abs(area-np.sum(mask))<20:
            break
        if area>np.sum(mask):
            high=mid
        elif area<np.sum(mask):
            low=mid
    return pose, mid

