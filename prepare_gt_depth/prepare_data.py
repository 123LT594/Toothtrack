#!/usr/bin/env python3
"""
数据准备工具：合并了三个数据准备脚本的功能
- extract_keypoints: 从obj文件提取3D关键点坐标
- generate_rt: 从2D关键点标注生成RT矩阵
- prepare_inference: 准备推理数据

使用方法:
  python prepare_data.py extract_keypoints [选项]
  python prepare_data.py generate_rt [选项]
  python prepare_data.py prepare_inference [选项]
"""

import sys
import os
import json
import argparse
import shutil
import glob
import trimesh
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional

# 允许直接以脚本运行
CUR_DIR = os.path.abspath(os.path.dirname(__file__))
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)

SCRIPT_DIR = CUR_DIR
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # 父目录为项目根目录
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 默认数据目录
DEFAULT_DATA_ROOT = os.path.join(PROJECT_ROOT, "demo_data/tooth_train")


# ==================== extract_keypoints 相关函数 ====================

def load_mesh_and_keypoints_for_extract(mesh_dir, tooth_file="tooth.obj", ball_files=None):
    """从obj文件加载4个钢珠的中心位置"""
    if ball_files is None:
        ball_files = ["1.obj", "2.obj", "3.obj", "4.obj"]
    
    # 加载牙齿模型
    tooth_path = os.path.join(mesh_dir, tooth_file)
    tooth_mesh = trimesh.load(tooth_path)
    
    # 计算牙齿模型中心（与FoundationPose一致，使用统一函数）
    tooth_center = compute_tooth_center(tooth_mesh)
    
    # 加载4个钢珠并计算中心位置
    keypoints_3d_original = []
    keypoints_3d_centered = []
    
    for i, ball_file in enumerate(ball_files, 1):
        ball_path = os.path.join(mesh_dir, ball_file)
        ball_mesh = trimesh.load(ball_path)
        # 统一使用边界框中心，与FoundationPose一致
        max_xyz_ball = ball_mesh.vertices.max(axis=0)
        min_xyz_ball = ball_mesh.vertices.min(axis=0)
        center_original = (min_xyz_ball + max_xyz_ball) / 2
        
        # 中心化（减去牙齿模型中心）
        center_centered = center_original - tooth_center
        
        keypoints_3d_original.append(center_original.tolist())
        keypoints_3d_centered.append(center_centered.tolist())
    
    return (
        np.array(keypoints_3d_centered),
        np.array(keypoints_3d_original),
        tooth_center,
    )


def compute_tooth_center(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    统一计算tooth_center的函数（边界框中心）
    
    Args:
        mesh: trimesh网格对象
        
    Returns:
        tooth_center: 3D数组，边界框中心坐标
    """
    max_xyz = mesh.vertices.max(axis=0)
    min_xyz = mesh.vertices.min(axis=0)
    return (min_xyz + max_xyz) / 2


def project_3d_to_2d_for_verify(keypoints_3d, rt, K, tooth_center=None, check_coordinate_system=True):
    """
    将3D点投影到2D（用于验证）
    
    重要：RT矩阵是从中心化坐标系生成的，因此keypoints_3d必须是中心化的。
    如果传入了tooth_center，会抛出错误，因为这表示坐标系不匹配。
    
    Args:
        keypoints_3d: 3D关键点坐标（必须是中心化的，即已减去tooth_center）
        rt: RT矩阵（从中心化坐标系到相机坐标系）
        K: 相机内参矩阵
        tooth_center: 牙齿模型中心（如果提供，会抛出错误，因为RT是中心化坐标系的）
        check_coordinate_system: 是否检查坐标系一致性
        
    Returns:
        keypoints_2d: 2D投影坐标
        
    Raises:
        ValueError: 如果传入了tooth_center，表示坐标系不匹配
    """
    import warnings
    
    # 检查坐标系一致性：RT是中心化坐标系的，不应该传入tooth_center
    if check_coordinate_system and tooth_center is not None:
        raise ValueError(
            "错误：RT矩阵是从中心化坐标系生成的，但传入了tooth_center参数。\n"
            "这表示坐标系不匹配。RT是中心化坐标系的，应该使用tooth_center=None。\n"
            "如果keypoints_3d是原始坐标系的，请先将其中心化：keypoints_3d_centered = keypoints_3d - tooth_center"
        )
    
    # RT是从中心化坐标系生成的，所以keypoints_3d必须是中心化的
    # 直接使用keypoints_3d（不添加tooth_center）
        keypoints_3d_homo = np.hstack([keypoints_3d, np.ones((len(keypoints_3d), 1))])
    
    keypoints_3d_cam = (rt @ keypoints_3d_homo.T).T[:, :3]
    keypoints_2d_homo = (K @ keypoints_3d_cam.T).T
    keypoints_2d = keypoints_2d_homo[:, :2] / keypoints_2d_homo[:, 2:3]
    return keypoints_2d


def verify_keypoints(keypoints_3d_centered, rt_file, annotations_file, frame_id, K_file, tooth_center):
    """
    验证3D坐标是否正确
    
    重要：RT矩阵是从中心化坐标系生成的，因此只验证中心化坐标系。
    不再尝试原始坐标系，避免掩盖坐标系不匹配的问题。
    
    Args:
        keypoints_3d_centered: 中心化的3D关键点坐标（已减去tooth_center）
        rt_file: RT矩阵文件路径（从中心化坐标系到相机坐标系）
        annotations_file: 2D标注文件路径
        frame_id: 帧ID
        K_file: 相机内参文件路径
        tooth_center: 牙齿模型中心（仅用于信息显示，不用于计算）
        
    Returns:
        bool: 验证是否通过
    """
    print("\n" + "=" * 60)
    print("验证3D坐标是否正确")
    print("=" * 60)
    print("注意：RT矩阵是从中心化坐标系生成的，只验证中心化坐标系")
    print("=" * 60)
    
    # 加载RT矩阵
    rt = np.loadtxt(rt_file)
    if rt.shape != (4, 4):
        raise ValueError(f"RT矩阵应该是4x4，但得到{rt.shape}")
    
    # 加载相机内参
    K = np.loadtxt(K_file)
    if K.shape != (3, 3):
        raise ValueError(f"相机内参应该是3x3，但得到{K.shape}")
    
    # 运行时检查：验证keypoints是否中心化（中心应该接近原点）
    keypoints_center = keypoints_3d_centered.mean(axis=0)
    center_norm = np.linalg.norm(keypoints_center)
    if center_norm > 0.01:  # 如果中心距离原点超过1cm，可能不是中心化的
        import warnings
        warnings.warn(
            f"警告：keypoints_3d_centered的中心距离原点较远（{center_norm*1000:.2f}mm），"
            f"可能不是中心化的。期望中心接近原点（< 1mm）。",
            UserWarning
        )
    
    # 加载2D标注
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    annotations = data.get("annotations", data)
    frame_key = None
    for key in annotations.keys():
        key_base = key.replace('.png', '')
        if frame_id in key_base or key_base in frame_id:
            frame_key = key
            break
    
    if frame_key is None:
        raise KeyError(f"找不到帧 {frame_id} 的标注")
    
    frame_data = annotations[frame_key]
    keypoints_2d_gt = []
    for i in range(1, 5):
        ball_key = f'ball_{i}'
        if ball_key not in frame_data:
            raise KeyError(f"帧 {frame_key} 中缺少 {ball_key}")
        keypoints_2d_gt.append(frame_data[ball_key])
    keypoints_2d_gt = np.array(keypoints_2d_gt)
    
    # 只使用中心化坐标系进行验证（RT是中心化坐标系的）
    keypoints_2d_pred = project_3d_to_2d_for_verify(
        keypoints_3d_centered, rt, K, 
        tooth_center=None,  # RT是中心化坐标系的，必须使用None
        check_coordinate_system=True
    )
    
    errors = np.linalg.norm(keypoints_2d_pred - keypoints_2d_gt, axis=1)
    mean_error = errors.mean()
    max_error = errors.max()
    
    print(f"使用的坐标系: 中心化坐标系（与RT矩阵一致）")
    print(f"keypoints中心距离原点: {center_norm*1000:.2f} mm")
    print(f"平均误差: {mean_error:.2f} 像素")
    print(f"最大误差: {max_error:.2f} 像素")
    
    if mean_error < 5.0:
        print("✓ 验证通过！坐标正确（平均误差 < 5像素）")
        return True
    elif mean_error < 10.0:
        print("⚠ 坐标基本正确，但误差稍大（5-10像素）")
        return True
    else:
        print("✗ 验证失败！误差较大（> 10像素）")
        print("   可能的原因：")
        print("   1. keypoints_3d_centered不是中心化的")
        print("   2. RT矩阵的坐标系不匹配")
        print("   3. 2D标注或相机内参有误")
        return False


def extract_keypoints_main():
    """提取3D关键点坐标的主函数"""
    parser = argparse.ArgumentParser(description="提取3D关键点坐标，并可选择性地验证")
    parser.add_argument("--mesh-dir", type=str, default=os.path.join(DEFAULT_DATA_ROOT, "mesh"), help="mesh文件目录")
    parser.add_argument("--output", type=str, default=os.path.join(DEFAULT_DATA_ROOT, "keypoints_3d_model.json"), help="输出JSON文件")
    parser.add_argument("--verify", action="store_true", help="是否验证坐标（需要提供RT矩阵和2D标注）")
    parser.add_argument("--rt-file", type=str, default=os.path.join(DEFAULT_DATA_ROOT, "annotated_poses/000000.txt"), help="RT矩阵文件（用于验证）")
    parser.add_argument("--annotations", type=str, default=os.path.join(DEFAULT_DATA_ROOT, "annotations.json"), help="2D标注文件（用于验证）")
    parser.add_argument("--frame-id", type=str, default="frame_0001", help="用于验证的帧ID")
    parser.add_argument("--K-file", type=str, default=os.path.join(DEFAULT_DATA_ROOT, "cam_K.txt"), help="相机内参文件（用于验证）")
    args = parser.parse_args()
    
    print("=" * 60)
    print("提取4个钢珠的3D坐标（用于训练）")
    print("=" * 60)
    
    # 1. 提取3D坐标
    print(f"\n1. 从obj文件加载3D坐标: {args.mesh_dir}")
    keypoints_3d_centered, keypoints_3d_original, tooth_center = load_mesh_and_keypoints_for_extract(args.mesh_dir)
    
    print(f"   ✓ 已加载，形状: {keypoints_3d_centered.shape}")
    print(f"   牙齿模型中心: [{tooth_center[0]:.6f}, {tooth_center[1]:.6f}, {tooth_center[2]:.6f}]")
    
    # 2. 可选：验证坐标
    if args.verify:
        if not os.path.exists(args.rt_file):
            print(f"\n⚠ 警告: RT文件不存在 ({args.rt_file})，跳过验证")
        elif not os.path.exists(args.annotations):
            print(f"\n⚠ 警告: 标注文件不存在 ({args.annotations})，跳过验证")
        else:
            verify_success = verify_keypoints(
                keypoints_3d_centered,
                args.rt_file,
                args.annotations,
                args.frame_id,
                args.K_file,
                tooth_center,
            )
            if not verify_success:
                print("\n⚠ 验证失败，但继续保存坐标（请检查数据）")
    
    # 3. 保存为JSON
    print(f"\n2. 保存到JSON文件: {args.output}")
    data = {
        "keypoints_3d_model": keypoints_3d_centered.tolist(),
        "coordinate_system": "centered_model_coords",
        "description": "4个钢珠在中心化模型坐标系中的3D坐标（用于训练）",
        "note": "这些坐标已经减去牙齿模型的中心，与FoundationPose的处理方式一致",
        "tooth_center": tooth_center.tolist(),
        "keypoints_3d_original": keypoints_3d_original.tolist(),
        "coordinate_system_original": "original_obj_coords",
        "description_original": "4个钢珠在原始obj文件坐标系中的3D坐标（仅用于参考）",
        "mapping": {
            "1.obj": "ball_1",
            "2.obj": "ball_2",
            "3.obj": "ball_3",
            "4.obj": "ball_4"
        },
        "usage": {
            "for_training": "使用 keypoints_3d_model（中心化后的坐标）",
            "for_visualization": "使用 keypoints_3d_original（原始坐标）"
        }
    }
    
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"   ✓ 已保存")
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print(f"\n重要提示:")
    print(f"1. 训练时使用: keypoints_3d_model（中心化后的坐标）")
    print(f"2. 这些坐标已经与FoundationPose的模型坐标系对齐")
    print(f"3. JSON文件: {args.output}")


# ==================== generate_rt 相关函数 ====================

def solve_pnp_from_keypoints(
    keypoints_3d: np.ndarray,
    keypoints_2d: np.ndarray,
    K: np.ndarray,
    use_ransac: bool = True,
    try_multiple_algorithms: bool = True,
    initial_rt: Optional[np.ndarray] = None,
    check_centered: bool = True,
) -> Tuple[Optional[np.ndarray], float]:
    """
    使用PnP算法从2D-3D对应关系求解位姿
    
    重要：keypoints_3d必须是中心化的（已减去tooth_center）。
    生成的RT矩阵是从中心化坐标系到相机坐标系的变换。
    
    Args:
        keypoints_3d: 3D关键点坐标（必须是中心化的）
        keypoints_2d: 2D关键点坐标
        K: 相机内参矩阵
        use_ransac: 是否使用RANSAC（当前未使用）
        try_multiple_algorithms: 是否尝试多种算法
        initial_rt: 初始RT矩阵（可选）
        check_centered: 是否检查keypoints是否中心化
        
    Returns:
        Tuple[RT, reproj_error]: RT矩阵和重投影误差
        
    Raises:
        ValueError: 如果keypoints_3d不是中心化的（中心距离原点太远）
    """
    if len(keypoints_3d) < 2:
        raise ValueError(f"至少需要2个点，但只有{len(keypoints_3d)}个")
    
    # 运行时检查：验证keypoints是否中心化
    if check_centered:
        keypoints_center = keypoints_3d.mean(axis=0)
        center_norm = np.linalg.norm(keypoints_center)
        if center_norm > 0.01:  # 如果中心距离原点超过1cm，可能不是中心化的
            import warnings
            warnings.warn(
                f"警告：keypoints_3d的中心距离原点较远（{center_norm*1000:.2f}mm），"
                f"可能不是中心化的。期望中心接近原点（< 1mm）。\n"
                f"如果keypoints_3d是原始坐标系的，请先将其中心化：keypoints_3d_centered = keypoints_3d - tooth_center",
                UserWarning
            )
    
    num_points = len(keypoints_3d)
    best_RT = None
    best_error = float("inf")
    
    # 处理2个点的情况
    if num_points == 2:
        if initial_rt is None:
            return None, float("inf")
        R_init = initial_rt[:3, :3]
        t_init = initial_rt[:3, 3]
        rvec_init, _ = cv2.Rodrigues(R_init)
        success, rvec, tvec = cv2.solvePnP(
            keypoints_3d, keypoints_2d, K, None,
            rvec=rvec_init, tvec=t_init.reshape(3, 1),
            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if success:
            R, _ = cv2.Rodrigues(rvec)
            RT = np.eye(4)
            RT[:3, :3] = R
            RT[:3, 3] = tvec.flatten()
            pred_2d = project_3d_to_2d(keypoints_3d, RT, K, tooth_center=None)
            error = compute_reproj_err(pred_2d, keypoints_2d)
            if error < best_error:
                best_RT = RT
                best_error = error
        if best_RT is None:
            return None, float("inf")
        return best_RT, best_error
    
    # 处理3个点的情况
    if num_points == 3:
        if initial_rt is None:
            center_3d = keypoints_3d.mean(axis=0)
            center_2d = keypoints_2d.mean(axis=0)
            z_guess = 0.2
            center_3d_cam = np.linalg.inv(K) @ np.array([center_2d[0], center_2d[1], 1.0]) * z_guess
            t_init = center_3d_cam - center_3d
            R_init = np.eye(3)
            rvec_init, _ = cv2.Rodrigues(R_init)
        else:
            R_init = initial_rt[:3, :3]
            t_init = initial_rt[:3, 3]
            rvec_init, _ = cv2.Rodrigues(R_init)
        success, rvec, tvec = cv2.solvePnP(
            keypoints_3d, keypoints_2d, K, None,
            rvec=rvec_init, tvec=t_init.reshape(3, 1),
            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if success:
            R, _ = cv2.Rodrigues(rvec)
            RT = np.eye(4)
            RT[:3, :3] = R
            RT[:3, 3] = tvec.flatten()
            pred_2d = project_3d_to_2d(keypoints_3d, RT, K, tooth_center=None)
            error = compute_reproj_err(pred_2d, keypoints_2d)
            if error < best_error:
                best_RT = RT
                best_error = error
        if best_RT is None:
            return None, float("inf")
        return best_RT, best_error
    
    # 4个或更多点的情况
    if try_multiple_algorithms:
        success_init, rvec_init, tvec_init = cv2.solvePnP(
            keypoints_3d, keypoints_2d, K, None, flags=cv2.SOLVEPNP_EPNP,
        )
        if success_init:
            success, rvec, tvec = cv2.solvePnP(
                keypoints_3d, keypoints_2d, K, None,
                rvec=rvec_init, tvec=tvec_init,
                useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if success:
                R, _ = cv2.Rodrigues(rvec)
                RT = np.eye(4)
                RT[:3, :3] = R
                RT[:3, 3] = tvec.flatten()
                pred_2d = project_3d_to_2d(keypoints_3d, RT, K, tooth_center=None)
                error = compute_reproj_err(pred_2d, keypoints_2d)
                if error < best_error:
                    best_RT = RT
                    best_error = error
    
    success, rvec, tvec = cv2.solvePnP(
        keypoints_3d, keypoints_2d, K, None, flags=cv2.SOLVEPNP_EPNP,
    )
    if success:
        R, _ = cv2.Rodrigues(rvec)
        RT = np.eye(4)
        RT[:3, :3] = R
        RT[:3, 3] = tvec.flatten()
        pred_2d = project_3d_to_2d(keypoints_3d, RT, K, tooth_center=None)
        error = compute_reproj_err(pred_2d, keypoints_2d)
        if error < best_error:
            best_RT = RT
            best_error = error
    
    if best_error > 20.0 and len(keypoints_3d) == 4 and best_RT is not None:
        pred_2d_all = project_3d_to_2d(keypoints_3d, best_RT, K, tooth_center=None)
        errors_per_point = np.linalg.norm(pred_2d_all - keypoints_2d, axis=1)
        worst_idx = np.argmax(errors_per_point)
        indices_3 = [i for i in range(4) if i != worst_idx]
        kp3d_3 = keypoints_3d[indices_3]
        kp2d_3 = keypoints_2d[indices_3]
        R_init = best_RT[:3, :3]
        t_init = best_RT[:3, 3]
        rvec_init, _ = cv2.Rodrigues(R_init)
        success, rvec, tvec = cv2.solvePnP(
            kp3d_3, kp2d_3, K, None,
            rvec=rvec_init, tvec=t_init.reshape(3, 1),
            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if success:
            R, _ = cv2.Rodrigues(rvec)
            RT = np.eye(4)
            RT[:3, :3] = R
            RT[:3, 3] = tvec.flatten()
            pred_2d = project_3d_to_2d(keypoints_3d, RT, K, tooth_center=None)
            error = compute_reproj_err(pred_2d, keypoints_2d)
            if error < best_error:
                best_RT = RT
                best_error = error
    
    if best_RT is None:
        return None, float("inf")
    return best_RT, best_error


def generate_rt_for_all_frames(
    data_root: str = "demo_data/tooth_train",
    output_dir: str = "demo_data/tooth_train/annotated_poses",
    max_reproj_error: float = 10.0,
    max_frames: int = None,
) -> Dict[str, Tuple[float, bool]]:
    """为所有有2D标注的帧生成RT矩阵"""
    ann_path = os.path.join(data_root, "annotations.json")
    K_path = os.path.join(data_root, "cam_K.txt")
    mesh_dir = os.path.join(data_root, "mesh")
    keypoints_3d_path = os.path.join(data_root, "keypoints_3d_model.json")
    
    with open(ann_path, "r") as f:
        ann = json.load(f)
    annotations: Dict[str, Dict[str, List[float]]] = ann.get("annotations", ann)
    
    K = np.loadtxt(K_path)
    
    with open(keypoints_3d_path, "r") as f:
        kp_data = json.load(f)
    keypoints_3d = np.array(kp_data["keypoints_3d_model"], dtype=np.float32)
    
    _, _, _, _, tooth_center = load_mesh_and_keypoints(mesh_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    frame_ids = sorted(annotations.keys())
    if max_frames is not None:
        frame_ids = frame_ids[:max_frames]
    
    results = {}
    success_count = 0
    prev_rt = None
    
    print(f"开始为 {len(frame_ids)} 帧生成RT矩阵...")
    print("=" * 60)
    
    for i, frame_id in enumerate(frame_ids, 1):
        try:
            kp2d_list = []
            kp3d_indices = []
            for j in range(1, 5):
                ball_key = f"ball_{j}"
                if ball_key in annotations[frame_id]:
                    kp2d_list.append(annotations[frame_id][ball_key])
                    kp3d_indices.append(j - 1)
            
            if len(kp2d_list) < 2:
                print(f"[{i}/{len(frame_ids)}] {frame_id}: 可见关键点不足（{len(kp2d_list)} < 2），跳过")
                results[frame_id] = (float("inf"), False)
                continue
            
            if len(kp2d_list) == 2 and prev_rt is None:
                print(f"[{i}/{len(frame_ids)}] {frame_id}: 只有2个点且无初始值，跳过")
                results[frame_id] = (float("inf"), False)
                continue
            
            keypoints_2d = np.array(kp2d_list, dtype=np.float32)
            keypoints_3d_subset = keypoints_3d[kp3d_indices]
            initial_rt = prev_rt if len(kp2d_list) == 3 else None
            
            RT, reproj_error = solve_pnp_from_keypoints(
                keypoints_3d_subset, keypoints_2d, K,
                use_ransac=False, initial_rt=initial_rt
            )
            
            if RT is None:
                print(f"[{i}/{len(frame_ids)}] {frame_id}: PnP求解失败")
                results[frame_id] = (float("inf"), False)
                continue
            
            if len(kp2d_list) == 2:
                error_threshold = max_reproj_error * 2.0
            elif len(kp2d_list) == 3:
                error_threshold = max_reproj_error * 1.5
            else:
                error_threshold = max_reproj_error
            
            if reproj_error > error_threshold:
                print(f"[{i}/{len(frame_ids)}] {frame_id}: 重投影误差过大 ({reproj_error:.2f} > {error_threshold:.2f}), 跳过")
                results[frame_id] = (reproj_error, False)
                continue
            
            frame_num = int(frame_id.replace("frame_", "").replace(".png", ""))
            rt_idx = frame_num - 1
            rt_path = os.path.join(output_dir, f"{rt_idx:06d}.txt")
            np.savetxt(rt_path, RT, fmt="%.18e")
            
            prev_rt = RT
            success_count += 1
            results[frame_id] = (reproj_error, True)
            
            if len(kp2d_list) == 2:
                point_info = "2个点（微调）"
            elif len(kp2d_list) == 3:
                point_info = "3个点"
            else:
                point_info = "4个点"
            if i % 50 == 0 or i == len(frame_ids):
                print(f"[{i}/{len(frame_ids)}] {frame_id}: ✓ 误差={reproj_error:.2f}px ({point_info}), 已保存 -> {rt_path}")
        
        except Exception as e:
            print(f"[{i}/{len(frame_ids)}] {frame_id}: 错误 - {e}")
            results[frame_id] = (float("inf"), False)
    
    print("=" * 60)
    print(f"完成！成功生成 {success_count}/{len(frame_ids)} 个RT矩阵")
    
    errors = [err for err, success in results.values() if success]
    if errors:
        print(f"重投影误差统计:")
        print(f"  平均: {np.mean(errors):.2f} 像素")
        print(f"  中位数: {np.median(errors):.2f} 像素")
        print(f"  最小: {np.min(errors):.2f} 像素")
        print(f"  最大: {np.max(errors):.2f} 像素")
    
    return results


def generate_rt_main():
    """生成RT矩阵的主函数"""
    parser = argparse.ArgumentParser(description="从2D关键点生成RT矩阵")
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT, help="数据根目录")
    parser.add_argument("--output-dir", type=str, default=os.path.join(DEFAULT_DATA_ROOT, "annotated_poses"), help="RT矩阵输出目录")
    parser.add_argument("--max-reproj-error", type=float, default=10.0, help="最大重投影误差（像素）")
    parser.add_argument("--max-frames", type=int, default=None, help="最多处理N帧（用于测试）")
    args = parser.parse_args()
    
    generate_rt_for_all_frames(
        data_root=args.data_root,
        output_dir=args.output_dir,
        max_reproj_error=args.max_reproj_error,
        max_frames=args.max_frames,
    )


# ==================== prepare_inference 相关函数 ====================

def prepare_inference_data(
    source_dir: str = "demo_data/tooth_train",
    target_dir: str = "demo_data/tooth",
    start_frame: int = 799,
    copy_mesh: bool = True,
    copy_cam_k: bool = True,
):
    """从训练数据中提取从指定帧开始的数据作为推理测试数据"""
    print("=" * 60)
    print("准备推理数据")
    print("=" * 60)
    
    source_dir = os.path.abspath(source_dir)
    target_dir = os.path.abspath(target_dir)
    
    if source_dir == target_dir:
        print("⚠️  错误: 源目录和目标目录不能相同！")
        return False
    
    if not os.path.exists(source_dir):
        print(f"⚠️  错误: 源目录不存在: {source_dir}")
        return False
    
    print(f"\n源目录: {source_dir}")
    print(f"目标目录: {target_dir}")
    print(f"起始帧: {start_frame} (frame_{start_frame:04d}.png)")
    
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(os.path.join(target_dir, "rgb"), exist_ok=True)
    
    source_rgb_dir = os.path.join(source_dir, "rgb")
    target_rgb_dir = os.path.join(target_dir, "rgb")
    
    if not os.path.exists(source_rgb_dir):
        print(f"⚠️  错误: RGB目录不存在: {source_rgb_dir}")
        return False
    
    rgb_files = sorted(glob.glob(os.path.join(source_rgb_dir, "*.png")))
    if not rgb_files:
        print(f"⚠️  错误: 在 {source_rgb_dir} 中找不到RGB图像")
        return False
    
    frame_files = []
    for rgb_file in rgb_files:
        basename = os.path.basename(rgb_file)
        if basename.startswith("frame_"):
            frame_num = int(basename.replace("frame_", "").replace(".png", ""))
        else:
            frame_num = int(basename.replace(".png", ""))
        if frame_num >= start_frame:
            frame_files.append((frame_num, rgb_file))
    
    if not frame_files:
        print(f"⚠️  错误: 没有找到 >= {start_frame} 的帧")
        return False
    
    print(f"\n找到 {len(frame_files)} 帧（从第 {start_frame} 帧开始）")
    
    copied_rgb = 0
    for new_idx, (old_frame_num, rgb_file) in enumerate(frame_files):
        new_filename = f"frame_{new_idx:04d}.png"
        target_path = os.path.join(target_rgb_dir, new_filename)
        shutil.copy2(rgb_file, target_path)
        copied_rgb += 1
        if (new_idx + 1) % 100 == 0:
            print(f"  已复制 {new_idx + 1}/{len(frame_files)} 帧...")
    
    print(f"✓ 已复制 {copied_rgb} 个RGB图像")
    
    if copy_mesh:
        source_mesh_dir = os.path.join(source_dir, "mesh")
        target_mesh_dir = os.path.join(target_dir, "mesh")
        if os.path.exists(source_mesh_dir):
            if os.path.exists(target_mesh_dir):
                shutil.rmtree(target_mesh_dir)
            shutil.copytree(source_mesh_dir, target_mesh_dir)
            print(f"✓ 已复制mesh目录")
        else:
            print(f"⚠️  警告: mesh目录不存在: {source_mesh_dir}")
    
    if copy_cam_k:
        source_cam_k = os.path.join(source_dir, "cam_K.txt")
        target_cam_k = os.path.join(target_dir, "cam_K.txt")
        if os.path.exists(source_cam_k):
            shutil.copy2(source_cam_k, target_cam_k)
            print(f"✓ 已复制相机内参")
        else:
            print(f"⚠️  警告: 相机内参文件不存在: {source_cam_k}")
    
    target_masks_dir = os.path.join(target_dir, "masks")
    if os.path.exists(os.path.join(source_dir, "masks")):
        os.makedirs(target_masks_dir, exist_ok=True)
        source_masks_dir = os.path.join(source_dir, "masks")
        for new_idx, (old_frame_num, _) in enumerate(frame_files):
            old_mask_patterns = [
                f"frame_{old_frame_num:04d}.png",
                f"{old_frame_num:06d}.png",
            ]
            new_mask_name = f"frame_{new_idx:04d}.png"
            for pattern in old_mask_patterns:
                old_mask_path = os.path.join(source_masks_dir, pattern)
                if os.path.exists(old_mask_path):
                    new_mask_path = os.path.join(target_masks_dir, new_mask_name)
                    shutil.copy2(old_mask_path, new_mask_path)
                    break
        print(f"✓ 已复制masks（如果存在）")
    
    readme_path = os.path.join(target_dir, "README_INFERENCE_DATA.txt")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("推理测试数据目录说明\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"此目录包含用于推理测试的数据\n")
        f.write(f"从训练数据目录提取: {source_dir}\n")
        f.write(f"起始帧: {start_frame} (frame_{start_frame:04d}.png)\n")
        f.write(f"总帧数: {len(frame_files)}\n\n")
        f.write("目录结构:\n")
        f.write("  - rgb/: RGB图像（从第0帧开始重新编号）\n")
        f.write("  - mesh/: 3D模型文件\n")
        f.write("  - cam_K.txt: 相机内参\n")
        f.write("  - masks/: mask文件（如果存在）\n\n")
        f.write("注意:\n")
        f.write("  - 此目录仅用于推理测试\n")
        f.write("  - RGB图像已重新编号（从frame_0000.png开始）\n")
        f.write("  - 推理脚本使用 --test_scene_dir 参数指定此目录\n")
    
    print(f"\n✓ 已创建说明文件: {readme_path}")
    print("\n" + "=" * 60)
    print("推理数据准备完成！")
    print("=" * 60)
    print(f"\n推理数据目录: {target_dir}")
    print(f"包含 {len(frame_files)} 帧（从原始第 {start_frame} 帧开始）")
    print(f"\n使用方法:")
    print(f"  python run_demo_without_depth.py \\")
    print(f"    --test_scene_dir {target_dir} \\")
    print(f"    --mesh_file {target_dir}/mesh/tooth.obj")
    
    return True


def prepare_inference_main():
    """准备推理数据的主函数"""
    parser = argparse.ArgumentParser(description="准备推理数据，从训练数据中提取指定帧范围")
    parser.add_argument("--source-dir", type=str, default=DEFAULT_DATA_ROOT, help="源数据目录（训练数据）")
    parser.add_argument("--target-dir", type=str, default=os.path.join(PROJECT_ROOT, "demo_data/tooth"), help="目标数据目录（推理数据，新建）")
    parser.add_argument("--start-frame", type=int, default=799, help="起始帧号（从0开始，即frame_0799.png对应799）")
    parser.add_argument("--no-mesh", action="store_true", help="不复制mesh目录")
    parser.add_argument("--no-cam-k", action="store_true", help="不复制相机内参")
    args = parser.parse_args()
    
    success = prepare_inference_data(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        start_frame=args.start_frame,
        copy_mesh=not args.no_mesh,
        copy_cam_k=not args.no_cam_k,
    )
    
    if not success:
        print("\n⚠️  准备推理数据失败，请检查错误信息")
        return 1
    
    return 0


# ==================== 主入口 ====================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]  # 移除子命令，保留其他参数
    
    if command == "extract_keypoints":
        extract_keypoints_main()
    elif command == "generate_rt":
        generate_rt_main()
    elif command == "prepare_inference":
        exit(prepare_inference_main())
    else:
        print(f"未知命令: {command}")
        print(__doc__)
        sys.exit(1)
