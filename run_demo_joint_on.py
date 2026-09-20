# # ① DAv2 预测深度，全图无门控
# python run_demo_joint_on.py --depth_src dav2  --dav2_ckpt /root/Toothtrack/checkpoints/depth_anything_v2_metric_golden_vits.pth --dav2_max_depth 0.2

# # ② DAv2 预测深度 + GT mask
# python run_demo_joint_on.py --depth_src dav2 --use_gt_mask  --dav2_ckpt /root/Toothtrack/checkpoints/depth_anything_v2_metric_golden_vits.pth --dav2_max_depth 0.2

# # ③ GT 深度 + GT mask（oracle 参考结果）
# python run_demo_joint_on.py --depth_src gt --use_gt_mask

# # ④ 旧蒸馏 StudentDepthNet + 预测 mask
# python run_demo_joint_on.py --depth_src distill --weight_student /root/lanyun-tmp/models/stage1_distill/models/student_stage1_ema_ep99.pth
import os
import sys
os.environ['OMP_NUM_THREADS'] = '1'
import time
import argparse
import numpy as np
import cv2
import torch
import json
import hashlib
import re
import trimesh
import pytz
import nvdiffrast.torch as dr
from datetime import datetime as dt

from utils.estimater import *
from utils.datareader import *
from utils.tools import *
from utils.render_3d import create_visualization
from utils.utils import make_mesh_tensors, nvdiffrast_render_depthonly

from learning.models.student_depth_net import StudentDepthNet
from learning.training.training_config import MAX_Z_RATIO, SHAPE_SCALE_RATIO
from utils.zero_shot_geometry_matcher import FastZeroShotMatcher

def calc_te(pose_pred, pose_gt):
    return np.linalg.norm(pose_pred[:3, 3] - pose_gt[:3, 3]) * 1000.0

def calc_re(pose_pred, pose_gt):
    R_pred, R_gt = pose_pred[:3, :3], pose_gt[:3, :3]
    trace = np.trace(R_pred @ R_gt.T)
    return np.rad2deg(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))

def calc_add(pose_pred, pose_gt, vertices):
    pts_pred = (pose_pred[:3, :3] @ vertices.T + pose_pred[:3, 3:4]).T
    pts_gt = (pose_gt[:3, :3] @ vertices.T + pose_gt[:3, 3:4]).T
    return np.linalg.norm(pts_pred - pts_gt, axis=1).mean() * 1000.0

def get_bbox_from_pose(pose, K, vertices):
    rvec, _ = cv2.Rodrigues(pose[:3, :3])
    tvec = pose[:3, 3]
    pts_2d, _ = cv2.projectPoints(vertices, rvec, tvec, K, None)
    pts_2d = pts_2d.squeeze()
    x_min, x_max = pts_2d[:, 0].min(), pts_2d[:, 0].max()
    y_min, y_max = pts_2d[:, 1].min(), pts_2d[:, 1].max()
    w = max(x_max - x_min, 10)
    h = max(y_max - y_min, 10)
    return x_min, y_min, w, h

def bbox_rect_mask(pose, K, vertices, H, W):
    """用上一帧位姿投影 bbox 画矩形（DAv2 无门控时仅用于 IoU/可视化参考，不参与追踪）。"""
    m = np.zeros((H, W), dtype=bool)
    if pose is None:
        return m
    x_min, y_min, w, h = get_bbox_from_pose(pose, K, vertices)
    y0, y1 = int(max(y_min, 0)), int(min(y_min + h, H))
    x0, x1 = int(max(x_min, 0)), int(min(x_min + w, W))
    if y1 > y0 and x1 > x0:
        m[y0:y1, x0:x1] = True
    return m

def render_gt_depth_map(pose, K, H, W, glctx, mesh_tensors):
    """
    用 nvdiffrast 按【原始坐标系 GT 位姿 + 原始 mesh】面片渲染一张全图深度（背景=0）。
    depth>0 即精确的牙齿 silhouette（GT mask），比凸包/点投影准确。
    处理 nvdiffrast 对宽高 8 对齐的要求。
    """
    Wp = int(np.ceil(W / 8.0) * 8)
    Hp = int(np.ceil(H / 8.0) * 8)
    Kp = K.copy()
    if Wp != W or Hp != H:
        Kp[0, 0] *= Wp / W
        Kp[0, 2] *= Wp / W
        Kp[1, 1] *= Hp / H
        Kp[1, 2] *= Hp / H
    pose_t = torch.as_tensor(np.asarray(pose), dtype=torch.float32, device='cuda').reshape(1, 4, 4)
    with torch.no_grad():
        d = nvdiffrast_render_depthonly(
            K=Kp, H=Hp, W=Wp, ob_in_cams=pose_t, glctx=glctx,
            mesh_tensors=mesh_tensors, output_size=[Hp, Wp])
    d = d.detach().reshape(Hp, Wp).cpu().numpy().astype(np.float32)
    if Wp != W or Hp != H:
        d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
    return d

def get_auto_color(d_np, mask_uint8):
    m_np = mask_uint8 > 127
    vis = np.zeros_like(d_np, dtype=np.uint8)
    if m_np.sum() > 0:
        valid = d_np[m_np]
        p_min, p_max = valid.min(), valid.max()
        if p_max - p_min > 1e-4:
            norm = np.clip((d_np - p_min) / (p_max - p_min), 0, 1)
            vis = (norm * 255).astype(np.uint8)
        else:
            vis[m_np] = 127
    color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    color[~m_np] = 0
    return color

def save_full_triplet(path, color_rgb, mask_bool, depth, W):
    """全图三联：RGB | 目标Mask | 深度(JET)。"""
    rgb_bgr = cv2.cvtColor(color_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    m255 = (mask_bool.astype(np.uint8) * 255)
    m3 = cv2.cvtColor(m255, cv2.COLOR_GRAY2BGR)
    dcol = get_auto_color(depth.astype(np.float32), m255)
    cat = np.hstack([rgb_bgr, m3, dcol])
    cv2.putText(cat, 'RGB', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(cat, 'Mask', (W + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(cat, 'Depth', (2 * W + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(path, cat)

# ===== Evaluation only: never feeds labels/diagnostics back into tracking =====
DEPTH_VALID_MIN = 0.001  # meters; matches the backend depth validity threshold

METRIC_GROUPS = {
    '位姿误差（误差越低越好；平移分量有符号）': [
        'Err2D_px', 'Err2D_ID_px', 'TE_mm', 'TxError_mm', 'TyError_mm',
        'TzError_mm', 'RE_deg', 'ADD_mm', 'PoseIoU', 'PoseBBoxIoU'],
    '输入深度与分割（覆盖率越高越好；Bias有符号）': [
        'InputMaskIoU', 'PredMaskIoU', 'DepthMAE_all_mm', 'DepthMAE_valid_mm',
        'DepthCoverage_pct', 'DepthBias_mm', 'DepthShapeMAE_mm',
        'BoundaryMAE_all_mm', 'InteriorMAE_all_mm',
        'BoundaryMAE_valid_mm', 'InteriorMAE_valid_mm',
        'BoundaryCoverage_pct', 'InteriorCoverage_pct',
        'RawDepthMAE_valid_mm', 'RawDepthCoverage_pct'],
    '实际后端几何覆盖（首轮为主；完整XYZ，不是任意非零分量）': [
        'PostFilterCoverage_pct', 'CropCoverage_pct', 'CropDepthCoverage_pct',
        'BackendCoverage_pct', 'BackendOffTargetRatio_pct',
        'BackendCoverageLast_pct', 'BackendOffTargetRatioLast_pct'],
    '计时（诊断采集启用时包含采集开销，不代表纯模型或视频端到端速度）': [
        'Track_ms', 'FrontendTrack_ms'],
}
METRIC_NAMES = [name for group in METRIC_GROUPS.values() for name in group]
FRAME_COLUMNS = [
    'sequence_index', 'frame_id', 'frame_gap', 'is_init', 'pose_updated',
    'has_gt_pose', 'has_gt_depth', 'gt_mask_fallback', 'input_mask_kind',
    'GTToothPixels', 'DepthValidPixels', 'BoundaryPixels', 'InteriorPixels',
    'Err2DPoints', 'BackendIterations', 'BackendStatus',
] + METRIC_NAMES + ['notes']


def valid_pose_for_eval(pose):
    return pose is not None and np.shape(pose) == (4, 4) and np.isfinite(pose).all()


def frame_number_from_id(frame_id):
    """Return the trailing numeric frame number, e.g. frame_0223 -> 223."""
    match = re.search(r'(\d+)$', str(frame_id))
    return int(match.group(1)) if match else None


def valid_depth_mask(depth):
    return np.isfinite(depth) & (depth >= DEPTH_VALID_MIN)


def mask_iou_for_eval(mask_a, mask_b):
    union = np.logical_or(mask_a, mask_b).sum()
    return float(np.logical_and(mask_a, mask_b).sum() / union) if union else np.nan


def split_tooth_regions(reference_mask, boundary_px):
    """Inner boundary band: r iterations of 3x3 erosion, image exterior=False."""
    interior = np.asarray(reference_mask, dtype=bool).copy()
    height, width = interior.shape
    for _ in range(boundary_px):
        padded = np.pad(interior, 1, mode='constant', constant_values=False)
        interior = np.logical_and.reduce([
            padded[dy:dy + height, dx:dx + width]
            for dy in range(3) for dx in range(3)
        ])
    return reference_mask & ~interior, interior


def depth_region_metrics(depth, reference_depth, region):
    """All-region MAE penalizes missing predictions as zero; valid MAE excludes them."""
    total = int(np.count_nonzero(region))
    valid = region & valid_depth_mask(depth)
    count = int(np.count_nonzero(valid))
    if not total:
        return np.nan, np.nan, np.nan, np.nan, np.nan, count
    # Evaluation copy only. No values written back to the tracking depth.
    evaluated_depth = np.where(valid_depth_mask(depth), depth, 0.0)
    mae_all = float(np.mean(np.abs(evaluated_depth[region] - reference_depth[region])) * 1000)
    if not count:
        return mae_all, np.nan, 0.0, np.nan, np.nan, count
    error_mm = (depth[valid].astype(np.float64) - reference_depth[valid]) * 1000
    return (mae_all, float(np.mean(np.abs(error_mm))), 100.0 * count / total,
            float(np.mean(error_mm)),
            float(np.mean(np.abs(error_mm - np.median(error_mm)))), count)


def calculate_depth_metrics(depth, reference_depth, boundary_px, raw_depth=None):
    reference_mask = valid_depth_mask(reference_depth)
    boundary, interior = split_tooth_regions(reference_mask, boundary_px)
    all_mae, valid_mae, coverage, bias, shape_mae, count = depth_region_metrics(
        depth, reference_depth, reference_mask)
    result = dict(DepthMAE_all_mm=all_mae, DepthMAE_valid_mm=valid_mae,
                  DepthCoverage_pct=coverage, DepthBias_mm=bias, DepthShapeMAE_mm=shape_mae,
                  GTToothPixels=int(reference_mask.sum()), DepthValidPixels=count,
                  BoundaryPixels=int(boundary.sum()), InteriorPixels=int(interior.sum()))
    for prefix, region in [('Boundary', boundary), ('Interior', interior)]:
        values = depth_region_metrics(depth, reference_depth, region)
        result[prefix + 'MAE_all_mm'] = values[0]
        result[prefix + 'MAE_valid_mm'] = values[1]
        result[prefix + 'Coverage_pct'] = values[2]
    if raw_depth is not None:
        values = depth_region_metrics(raw_depth, reference_depth, reference_mask)
        result['RawDepthMAE_valid_mm'] = values[1]
        result['RawDepthCoverage_pct'] = values[2]
    return result


def project_reference_support(reference_mask, tf_to_crop, raw_xyz, final_xyz,
                              translation, radius, normalize_xyz):
    """Project original reference pixel centers onto the actual nearest-sampled crop.

    Measures support coverage, not preservation of the original full-resolution points.
    A point is complete only if all three XYZ components survive normalization/clipping.
    """
    ys, xs = np.nonzero(reference_mask)
    if not len(xs):
        return dict(CropCoverage_pct=np.nan, CropDepthCoverage_pct=np.nan,
                    BackendCoverage_pct=np.nan, BackendOffTargetRatio_pct=np.nan)
    height, width = raw_xyz.shape[1:]
    projected = tf_to_crop @ np.stack([xs, ys, np.ones_like(xs)], axis=0)
    denom = projected[2]
    good = np.isfinite(projected).all(axis=0) & (np.abs(denom) > 1e-12)
    u = np.divide(projected[0], denom, out=np.full(len(xs), np.nan), where=good)
    v = np.divide(projected[1], denom, out=np.full(len(xs), np.nan), where=good)
    # Match nearest-neighbor sampling support, including half-pixel border cells.
    xi = np.rint(np.where(good, u, -1)).astype(np.int64)
    yi = np.rint(np.where(good, v, -1)).astype(np.int64)
    in_crop = good & (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    raw_valid = np.isfinite(raw_xyz).all(axis=0) & (raw_xyz[2] >= DEPTH_VALID_MIN)
    expected = raw_xyz.astype(np.float32) - np.asarray(translation, dtype=np.float32).reshape(3, 1, 1)
    if normalize_xyz:
        expected *= np.float32(1.0) / np.float32(radius)
    # Compare with captured output, rather than treating zero coordinates as invalid.
    kept = raw_valid & np.isfinite(final_xyz).all(axis=0) & np.isclose(
        final_xyz, expected, rtol=1e-5, atol=1e-7).all(axis=0)
    if normalize_xyz:
        kept &= (np.abs(expected) < 2).all(axis=0)

    # Classify every complete backend XYZ by the original-image GT tooth support.
    # This diagnoses extra geometry entering the refiner in an ungated run.
    crop_y, crop_x = np.indices((height, width))
    crop_h = np.stack([crop_x.ravel(), crop_y.ravel(), np.ones(height * width)], axis=0)
    try:
        crop_to_image = np.linalg.inv(tf_to_crop)
        original = crop_to_image @ crop_h
        original_denom = original[2]
        mapped = np.isfinite(original).all(axis=0) & (np.abs(original_denom) > 1e-12)
        original_x = np.rint(np.divide(
            original[0], original_denom, out=np.full(height * width, -1.0), where=mapped)).astype(np.int64)
        original_y = np.rint(np.divide(
            original[1], original_denom, out=np.full(height * width, -1.0), where=mapped)).astype(np.int64)
        image_h, image_w = reference_mask.shape
        mapped &= ((original_x >= 0) & (original_x < image_w) &
                   (original_y >= 0) & (original_y < image_h))
        mapped_to_target = np.zeros(height * width, dtype=bool)
        mapped_to_target[mapped] = reference_mask[original_y[mapped], original_x[mapped]]
        complete_mapped = kept.ravel() & mapped
        complete_count = int(complete_mapped.sum())
        off_target_ratio = (100.0 * np.count_nonzero(complete_mapped & ~mapped_to_target) / complete_count
                            if complete_count else np.nan)
    except np.linalg.LinAlgError:
        off_target_ratio = np.nan

    xc, yc = xi[in_crop], yi[in_crop]
    return dict(CropCoverage_pct=100.0 * in_crop.sum() / len(xs),
                CropDepthCoverage_pct=100.0 * raw_valid[yc, xc].sum() / len(xs),
                BackendCoverage_pct=100.0 * kept[yc, xc].sum() / len(xs),
                BackendOffTargetRatio_pct=off_target_ratio)


class BackendCoverageProbe:
    """Read-only snapshots from existing calls. Original methods/outputs are unchanged.

    Captures filtered depth at refiner entry and actual crop XYZ before/after dataset
    normalization. Evaluation happens after the tracking timer; tensor-copy overhead
    remains in that timer. No extra refiner pass and no GT input to the tracker.
    """
    def __init__(self, refiner):
        self.refiner = refiner
        self.original_predict = refiner.predict
        self.original_transform = refiner.dataset.transform_batch
        self.reset()
        refiner.predict = self._predict
        refiner.dataset.transform_batch = self._transform

    def reset(self):
        self.filtered_depth = None
        self.snapshots = []
        self.error = ''

    def _predict(self, *args, **kwargs):
        try:
            depth = kwargs.get('depth', args[1] if len(args) > 1 else None)
            self.filtered_depth = depth.detach().clone()
        except Exception as exc:
            self.error = 'capture_depth:' + str(exc)
        return self.original_predict(*args, **kwargs)

    def _transform(self, *args, **kwargs):
        snapshot = None
        try:
            batch = kwargs.get('batch', args[0] if args else None)
            snapshot = dict(raw=batch.xyz_mapBs.detach().clone(),
                            tf=batch.tf_to_crops.detach().clone(),
                            pose=batch.poseA.detach().clone(),
                            diameters=batch.mesh_diameters.detach().clone(),
                            normalize=bool(self.refiner.cfg.get('normalize_xyz', False)))
        except Exception as exc:
            self.error = 'capture_crop:' + str(exc)
        result = self.original_transform(*args, **kwargs)
        if snapshot is not None:
            try:
                snapshot['final'] = result.xyz_mapBs.detach().clone()
                self.snapshots.append(snapshot)
            except Exception as exc:
                self.error = 'capture_normalized:' + str(exc)
        return result

    def evaluate(self, reference_mask):
        result = dict(BackendIterations=len(self.snapshots), BackendStatus='ok')
        if self.error:
            result['BackendStatus'] = 'capture_error:' + self.error
        elif not self.snapshots:
            result['BackendStatus'] = 'not_run'
        total = int(reference_mask.sum())
        if self.filtered_depth is not None and total:
            filtered = self.filtered_depth.detach().cpu().numpy()
            result['PostFilterCoverage_pct'] = 100.0 * (reference_mask & valid_depth_mask(filtered)).sum() / total
        if not total:
            result['BackendStatus'] = 'empty_reference'
        for index, snapshot in enumerate(self.snapshots):
            if snapshot['raw'].shape[0] != 1:
                result['BackendStatus'] = 'unsupported_pose_batch'
                continue
            values = project_reference_support(
                reference_mask, snapshot['tf'][0].detach().cpu().numpy(),
                snapshot['raw'][0].detach().cpu().numpy(),
                snapshot['final'][0].detach().cpu().numpy(),
                snapshot['pose'][0, :3, 3].detach().cpu().numpy(),
                float(snapshot['diameters'][0].detach().cpu().numpy()) / 2,
                snapshot['normalize'])
            if index == 0:
                result.update(values)
            if index == len(self.snapshots) - 1:
                result['BackendCoverageLast_pct'] = values['BackendCoverage_pct']
                result['BackendOffTargetRatioLast_pct'] = values['BackendOffTargetRatio_pct']
        return result

    def close(self):
        self.refiner.predict = self.original_predict
        self.refiner.dataset.transform_batch = self.original_transform


def format_report_value(value):
    if value is None:
        return 'N/A'
    if isinstance(value, (float, np.floating)):
        return f'{value:.6f}' if np.isfinite(value) else 'N/A'
    return str(value).replace('\t', ' ').replace('\r', ' ').replace('\n', ' ')


def write_metric_definitions(handle, boundary_px):
    definitions = [
        '评估协议：不筛帧、不新增位姿重置；使用用户准备的RGB序列。所有汇总排除首帧。',
        'GT初始化首帧仅写入位姿，不调用refiner。逐帧表仍保留首帧，is_init=1。',
        'GT深度是模型渲染参考；以下牙齿区域V=finite(GTdepth)且GTdepth>=0.001m，不自动代表真实遮挡后的可见区域。',
        'P=finite(input_depth)且input_depth>=0.001m；深度指标在门控后、后端滤波前评价。单位：深度/平移/ADD为mm，RE为度，Err2D为px。',
        'DepthMAE_all_mm=mean_V(abs(D_eval-Dgt))*1000；仅评估副本将无效预测置0，包含漏预测惩罚。',
        'DepthMAE_valid_mm=mean_(V&P)(abs(D-Dgt))*1000；DepthCoverage_pct=100*|V&P|/|V|。二者必须联合阅读。',
        'DepthBias_mm=mean_(V&P)(D-Dgt)*1000；正值偏远、负值偏近。DepthShapeMAE_mm为有效误差减去其中位数后的MAE，只诊断形状，不能替代米制MAE。',
        f'Boundary为V内部{boundary_px}px边界带（3x3方形核腐蚀{boundary_px}次；图像外部视为背景），Interior为剩余V。分区_all惩罚遗漏，_valid仅评价交集。',
        'RawDepthMAE_valid_mm/RawDepthCoverage_pct评价门控前深度；distill只在其真实160crop映射覆盖范围内有预测，不填补crop外。',
        'Tx/Ty/TzError=(t_pred-t_gt)*1000，为相机坐标系中原始mesh原点的有符号平移误差。TE是其L2范数；RE是旋转测地角；ADD是同一mesh顶点对应距离均值。',
        'Err2D_px保留旧的最优置换匹配定义；Err2D_ID_px按ball_j固定身份对应（仅当标注身份一致才有意义）。阈值比例按每帧平均Err2D统计，不称关键点PCK。',
        'PoseIoU=当前输出pose渲染silhouette与V的IoU，四版本可比。PoseBBoxIoU=当前输出pose投影矩形与V的IoU；不是refiner真实crop。',
        'InputMaskIoU=实际深度门控mask与V的IoU；无门控为N/A，GT门控版本仅为一致性自检。distill使用预测mask，因此InputMaskIoU与PredMaskIoU相同。所有IoU范围0~1。',
        'PostFilterCoverage_pct=真实腐蚀/双边滤波后，V中有效深度的比例。',
        'CropCoverage_pct=V像素中心投影到实际首轮crop的覆盖比例；CropDepthCoverage_pct再要求该crop位置有有效观测深度。',
        'BackendCoverage_pct再要求首轮实际归一化输出保留完整XYZ；任意坐标被筛除不计完整保留。以原图V像素中心映射后最近采样统计，分母始终|V|；这是几何支持覆盖，不表示160crop保留原分辨率细节。',
        'BackendOffTargetRatio_pct=首轮实际保留的完整XYZ中，映射到GT牙齿区域V之外的比例；越低表示进入Refiner的目标外几何越少。它衡量混入比例，不衡量这些点造成的误差。',
        'BackendCoverageLast_pct和BackendOffTargetRatioLast_pct为最后一轮的同口径值；首轮/末轮不同反映精炼中crop变化。首次初始化无后端调用，后端指标N/A。',
        '空区域/无有效预测/无标签/未执行项记N/A，不填0。每个汇总指标单独列N；比较版本时仍需相同frame_id集合。',
        'Track_ms为原追踪代码段耗时；FrontendTrack_ms为原前端+追踪计时范围，含GT读取/调试打印，不含读RGB、评估渲染和保存视频。后端快照采集有开销，纯测速可加--no_backend_metrics。计时汇总只统计首帧之后实际更新的帧。',
        '四版本解释：①vs②是理想门控收益；②vs③是GT门控下DAv2深度与参考深度的差距；④使用Student深度与预测mask，与①②比较属于前端系统对比，不能只归因于深度；③是oracle参照，不是理论性能上限。',
    ]
    handle.write('# Metric report v2 (UTF-8; per-frame table is TAB-separated)\n')
    for definition in definitions:
        handle.write('# ' + definition + '\n')


def write_metric_summary(handle, rows):
    evaluated = [row for row in rows if not row['is_init']]
    handle.write('\n# ================= 汇总（排除首帧） =================\n')
    handle.write(f'# 总处理帧={len(rows)}；汇总候选帧={len(evaluated)}；实际更新帧={sum(r["pose_updated"] for r in evaluated)}\n')
    handle.write(f'# GT pose可用帧={sum(r["has_gt_pose"] for r in evaluated)}；GT depth可用帧={sum(r["has_gt_depth"] for r in evaluated)}；GT mask回退帧={sum(r["gt_mask_fallback"] for r in evaluated)}\n')
    handle.write('# Signed误差均值可能相互抵消，另列AbsP95；覆盖率请关注Min/P05，误差关注P95/Max。\n')
    for group, names in METRIC_GROUPS.items():
        handle.write('# ' + group + '\n')
        handle.write('# metric\tN\tMean\tMedian\tMin\tP05\tP95\tMax\tAbsP95\n')
        for name in names:
            selected = [r for r in evaluated if r['pose_updated']] if name.endswith('_ms') else evaluated
            values = np.asarray([r.get(name, np.nan) for r in selected], dtype=float)
            values = values[np.isfinite(values)]
            if len(values):
                stats = [values.mean(), np.median(values), values.min(), np.percentile(values, 5),
                         np.percentile(values, 95), values.max(), np.percentile(np.abs(values), 95)]
                handle.write('# ' + name + '\t' + str(len(values)) + '\t' + '\t'.join(format_report_value(x) for x in stats) + '\n')
            else:
                handle.write('# ' + name + '\t0\tN/A\tN/A\tN/A\tN/A\tN/A\tN/A\tN/A\n')
    for name in ['Err2D_px', 'Err2D_ID_px']:
        values = np.asarray([r.get(name, np.nan) for r in evaluated], dtype=float)
        values = values[np.isfinite(values)]
        for threshold in [2, 3, 10]:
            rate = 100.0 * np.mean(values < threshold) if len(values) else np.nan
            handle.write(f'# FrameSuccess({name}<{threshold}px)={format_report_value(rate)}%, N={len(values)}\n')
    # Rank by original frame_id so every outlier can be located in the source sequence.
    for name, reverse in [('ADD_mm', True), ('DepthMAE_all_mm', True),
                          ('BackendCoverage_pct', False), ('BackendOffTargetRatio_pct', True)]:
        candidates = [r for r in evaluated if np.isfinite(r.get(name, np.nan))]
        ranked = sorted(candidates, key=lambda r: r[name], reverse=reverse)[:10]
        handle.write(f'# 异常帧Top10: {name}\n')
        for row in ranked:
            fields = ['frame_id', 'frame_gap', name, 'TE_mm', 'RE_deg', 'DepthMAE_valid_mm',
                      'DepthCoverage_pct', 'DepthBias_mm', 'BackendCoverage_pct',
                      'BackendOffTargetRatio_pct']
            fields = list(dict.fromkeys(fields))
            handle.write('# ' + ', '.join(key + '=' + format_report_value(row.get(key)) for key in fields) + '\n')


class SuppressPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument("--mesh_file", type=str, default=f"{code_dir}/demo_data/tooth/mesh/tooth.obj")
    parser.add_argument("--test_scene_dir", type=str, default=f"{code_dir}/demo_data/tooth")
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=1)
    parser.add_argument("--weight_student", type=str, default="/root/lanyun-tmp/models/stage1_distill/models/student_stage1_ema_ep99.pth")

    parser.add_argument("--use_pred_init", action='store_true', help="首帧用预测初始化（仅 distill 有效；默认 GT 位姿初始化）")
    parser.add_argument("--no_eval", action='store_true', help="不开启验证")

    # ===== 消融：深度来源 + 是否用 GT mask 门控 =====
    # depth_src:
    #   dav2    -> Depth Anything V2 全图米制深度
    #   gt      -> GT 渲染深度（oracle）
    #   distill -> 旧 StudentDepthNet（160 crop，固定使用网络预测mask门控）
    # use_gt_mask: 用 GT 位姿渲染的精确牙齿 silhouette 把深度背景清零（门控）。
    #   四版本：
    #     dav2（无flag）          = DAv2 预测深度，全图无门控（现状 baseline）
    #     dav2 + --use_gt_mask    = DAv2 预测深度 + GT mask（门控上界，路A决定性实验）
    #     gt   + --use_gt_mask    = GT 深度 + GT mask（oracle 参考输入）
    #     distill                = 旧蒸馏 StudentDepthNet 深度 + 预测mask
    parser.add_argument("--depth_src", type=str, default=None, choices=["dav2", "gt", "distill"])
    parser.add_argument("--use_dav2", action='store_true', help="[兼容旧命令] 等价 --depth_src dav2")
    parser.add_argument("--use_gt_mask", action='store_true', help="用 GT 位姿渲染的 mask 把深度背景清零（门控）")
    parser.add_argument("--gt_pose_dir", type=str, default=None, help="GT 位姿目录（默认 test_scene_dir/pose）")
    parser.add_argument("--gt_depth_dir", type=str, default=None, help="GT 深度目录（默认 test_scene_dir/depth）")

    # ===== DAv2 参数 =====
    parser.add_argument("--dav2_ckpt", type=str,
                        default="/root/Toothtrack/checkpoints/depth_anything_v2_metric_golden_vits.pth")
    parser.add_argument("--dav2_encoder", type=str, default="vits", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--dav2_input_size", type=int, default=518)
    parser.add_argument("--dav2_max_depth", type=float, default=0.2, help="必须和微调时 --max-depth 一致")

    parser.add_argument('--eval_boundary_px', type=int, default=3,
                        help='评估用牙齿内边界带宽度（原图像素，不改变追踪）')
    parser.add_argument('--no_backend_metrics', action='store_true',
                        help='关闭后端张量快照，仅用于减少诊断计时开销；不改变追踪结果')
    args = parser.parse_args()
    if args.eval_boundary_px < 1:
        parser.error('--eval_boundary_px 必须 >= 1')

    # 解析深度来源（兼容旧 --use_dav2 / 无参默认 distill）
    if args.depth_src is None:
        args.depth_src = "dav2" if args.use_dav2 else "distill"
    if args.depth_src == "distill" and args.use_gt_mask:
        print("ℹ️  distill 固定使用预测mask，已忽略 --use_gt_mask。")
        args.use_gt_mask = False
    if args.use_pred_init and args.depth_src != "distill":
        print("⚠️  --use_pred_init 仅 distill 基线支持，已忽略，首帧改用 GT 位姿初始化。")
        args.use_pred_init = False

    gt_pose_dir = args.gt_pose_dir or os.path.join(args.test_scene_dir, "pose")
    gt_depth_dir = args.gt_depth_dir or os.path.join(args.test_scene_dir, "depth")
    annotated_pose_dir = os.path.join(args.test_scene_dir, "annotated_pose")

    if args.depth_src == "distill":
        tag = "distill_track"
    elif args.depth_src == "dav2":
        tag = "dav2_gtmask" if args.use_gt_mask else "dav2_nomask"
    else:
        tag = "gtdepth_gtmask" if args.use_gt_mask else "gtdepth_nomask"

    output_root = "/root/lanyun-tmp/output"
    beijing_tz = pytz.timezone('Asia/Shanghai')
    output_dir = os.path.join(output_root, f"{dt.now(beijing_tz).strftime('%m%d_%H%M')}_{tag}")
    img_output_dir = os.path.join(output_dir, "img")
    mask_depth_output_dir = os.path.join(output_dir, "mask+depth")
    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(mask_depth_output_dir, exist_ok=True)

    error_txt_path = os.path.join(output_dir, "frame_errors.txt")
    txt_file = open(error_txt_path, "w", encoding="utf-8")
    txt_file.write(f"# config: depth_src={args.depth_src} use_gt_mask={args.use_gt_mask} "
                   f"gt_pose_dir={gt_pose_dir} gt_depth_dir={gt_depth_dir}\n")

    set_logging_format(); set_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    glctx = dr.RasterizeCudaContext()

    print(f"🧪 消融版本: {tag}")
    print(f"   GT pose : {gt_pose_dir}")
    print(f"   GT depth: {gt_depth_dir}")

    # ===== 加载深度模型（按需）=====
    dav2 = None
    model_expert = None
    student_in_channels = None
    if args.depth_src == "dav2":
        print("🚀 加载 DAv2 微调深度模型 (Depth Anything V2 metric)...")
        sys.path.insert(0, code_dir)
        from depth_anything_v2.dpt import DepthAnythingV2
        dav2_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        dav2 = DepthAnythingV2(**{**dav2_configs[args.dav2_encoder], 'max_depth': args.dav2_max_depth})
        ckpt = torch.load(args.dav2_ckpt, map_location='cpu', weights_only=False)
        state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        dav2.load_state_dict(state)
        dav2 = dav2.to(device).eval()
        print(f"   ✅ DAv2 已加载: {args.dav2_ckpt} (max_depth={args.dav2_max_depth})")
    elif args.depth_src == "distill":
        print("🚀 加载 3D 蒸馏基座 (StudentDepthNet)...")
        model_expert = StudentDepthNet().to(device)
        model_expert.load_state_dict(torch.load(args.weight_student, map_location=device))
        model_expert.eval()
        student_in_channels = int(model_expert.backbone.features[0][0].in_channels)
        if student_in_channels not in (6, 7):
            raise RuntimeError(f"StudentDepthNet首层输入通道应为6或7，实际为{student_in_channels}")
        print(f"   ✅ StudentDepthNet 输入通道: {student_in_channels} "
              f"({'RGB+Ray' if student_in_channels == 6 else 'RGB+Ray+Z_base'})")

    gt_data = {}
    ball_centroids = None
    metric_rows = []
    ball_ids = []

    if not args.no_eval:
        ann_path = os.path.join(args.test_scene_dir, "annotations.json")
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f: gt_data = json.load(f).get("annotations", {})

        ball_centroids_list = []
        for j in [1, 2, 3, 4]:
            p = os.path.join(os.path.dirname(args.mesh_file), f"{j}.obj")
            if os.path.exists(p):
                ball_centroids_list.append(trimesh.load(p).vertices.mean(0))
                ball_ids.append(j)
        if ball_centroids_list: ball_centroids = np.array(ball_centroids_list, dtype=np.float32)

    mesh = trimesh.load(args.mesh_file)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    # 用于按 GT 位姿渲染精确 silhouette（原始 mesh / 原始位姿同系）
    gt_mesh_tensors = make_mesh_tensors(mesh) if (args.use_gt_mask or not args.no_eval) else None

    refiner = PoseRefinePredictor()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=None, refiner=refiner, glctx=glctx)

    backend_probe = (BackendCoverageProbe(refiner)
                     if not args.no_eval and not args.no_backend_metrics else None)

    # 仅 distill + 预测初始化才需要几何粗筛器
    matcher = None
    if args.depth_src == "distill" and args.use_pred_init:
        print("🚀 加载 3D 零样本纯几何粗筛器 (FastZeroShotMatcher)...")
        matcher_pkl_path = "/root/Toothtrack/demo_data/ztooth/zero_shot_db.pkl"
        matcher = FastZeroShotMatcher(pkl_path=matcher_pkl_path, alpha=0.5)

    reader = YcbineoatReader(video_dir=args.test_scene_dir, zfar=np.inf)
    video_writer = cv2.VideoWriter(os.path.join(output_dir, f"track_{tag}.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 30, (reader.W, reader.H))
    write_metric_definitions(txt_file, args.eval_boundary_px)
    sequence_digest = hashlib.sha256('\n'.join(reader.id_strs).encode('utf-8')).hexdigest()
    txt_file.write(f'# frame_count={len(reader.id_strs)} frame_id_sha256={sequence_digest}\n')
    txt_file.write(f'# backend_capture={backend_probe is not None}; track_refine_iter={args.track_refine_iter}; '
                   f'init={"predicted" if args.use_pred_init else "GT"}; metric_boundary_px={args.eval_boundary_px}\n')
    txt_file.write('\t'.join(FRAME_COLUMNS) + '\n')
    previous_pose = None
    previous_mask = None

    # distill 可视化占位
    mask_crop_np = np.zeros((160, 160), dtype=np.uint8)
    rgb_crop = np.zeros((160, 160, 3), dtype=np.uint8)
    depth_crop_np = np.zeros((160, 160), dtype=np.float32)
    c_x = c_y = 0.0

    try:
        for i in range(len(reader.color_files)):
            color, frame_name, H, W = reader.get_color(i), reader.id_strs[i] + ".png", reader.get_color(i).shape[0], reader.get_color(i).shape[1]
            torch.cuda.synchronize()
            t1 = time.time()

            # ---------- 该帧 GT 位姿（oracle mask / 首帧初始化 / 评估都用它）----------
            gt_pose_path = os.path.join(gt_pose_dir, f"{reader.id_strs[i]}.npy")
            frame_gt_pose = np.load(gt_pose_path) if os.path.exists(gt_pose_path) else None
            ann_path_i = os.path.join(annotated_pose_dir, f"{reader.id_strs[i]}.npy")
            if frame_gt_pose is None and os.path.exists(ann_path_i):
                frame_gt_pose = np.load(ann_path_i)

            if i == 0 and not args.use_pred_init:
                assert frame_gt_pose is not None, f"首帧需要 GT 位姿，未找到: {gt_pose_path}"
                previous_pose = frame_gt_pose
                pose_gt = frame_gt_pose

            have_obs = True  # 保留原有缺观测行为；数据筛选由用户在运行前完成。
            pose_updated = False
            raw_depth_eval = None
            predicted_mask_eval = None
            if backend_probe is not None:
                backend_probe.reset()

            with torch.no_grad():
                if args.depth_src == "distill":
                    # ==========================================================
                    # 旧 StudentDepthNet 流程（160x160 crop + ray_map，自带预测 mask）
                    # ==========================================================
                    if i == 0:
                        if args.use_pred_init:
                            initial_mask_path = os.path.join(args.test_scene_dir, "mask", reader.id_strs[i] + ".png")
                            initial_mask = cv2.imread(initial_mask_path, cv2.IMREAD_GRAYSCALE) > 0
                            ys, xs = np.where(initial_mask)
                            x_min, x_max = xs.min(), xs.max()
                            y_min, y_max = ys.min(), ys.max()
                            w = max(x_max - x_min, 10)
                            h = max(y_max - y_min, 10)
                        else:
                            x_min, y_min, w, h = get_bbox_from_pose(pose_gt, reader.K, mesh.vertices)
                    else:
                        ys, xs = np.where(previous_mask)
                        if len(xs) > 10:
                            x_min, x_max = xs.min(), xs.max()
                            y_min, y_max = ys.min(), ys.max()
                            w = max(x_max - x_min, 10)
                            h = max(y_max - y_min, 10)
                        else:
                            x_min, y_min, w, h = get_bbox_from_pose(previous_pose, reader.K, mesh.vertices)

                    c_x, c_y = x_min + w / 2.0, y_min + h / 2.0
                    crop_size = max(w, h) * 1.2

                    M = cv2.getRotationMatrix2D((c_x, c_y), 0, 160.0 / crop_size)
                    M[0, 2] += 80.0 - c_x
                    M[1, 2] += 80.0 - c_y

                    rgb_crop = cv2.warpAffine(color, M, (160, 160), flags=cv2.INTER_LINEAR, borderValue=(0,0,0))
                    rgb_tensor = torch.from_numpy(rgb_crop).float().permute(2,0,1) / 255.0

                    M_3x3 = np.vstack([M, [0, 0, 1]])
                    K_crop = M_3x3 @ reader.K
                    K_inv = np.linalg.inv(K_crop)
                    u, v = np.meshgrid(np.arange(160), np.arange(160))
                    uv1 = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3)
                    unnorm_rays = (K_inv @ uv1.T).T.reshape(160, 160, 3)
                    ray_map = unnorm_rays / np.linalg.norm(unnorm_rays, axis=-1, keepdims=True)
                    ray_tensor = torch.from_numpy(ray_map).float().permute(2,0,1)

                    dynamic_physical_width = extents.max()
                    if i == 0 or previous_pose is None:
                        if args.use_pred_init:
                            cos_theta = 1.0
                        else:
                            cos_theta = max(abs(pose_gt[2, 2]), 0.5)
                        Z_base = reader.K[0, 0] * (dynamic_physical_width / max(w, h)) * cos_theta
                    else:
                        Z_base = previous_pose[2, 3]

                    # 旧权重按训练协议使用6通道RGB+Ray；同时兼容明确训练过的7通道模型。
                    student_input_parts = [rgb_tensor, ray_tensor]
                    if student_in_channels == 7:
                        z_base_norm = torch.full(
                            (1, 160, 160), float(Z_base) / 0.2, dtype=torch.float32)
                        student_input_parts.append(z_base_norm)
                    student_inputs = torch.cat(student_input_parts, dim=0).unsqueeze(0).to(device)
                    if student_inputs.shape[1] != student_in_channels:
                        raise RuntimeError(
                            f"Student输入构造为{student_inputs.shape[1]}通道，模型要求{student_in_channels}通道")

                    shape_weight_raw, mask_pred, delta_z_scalar = model_expert(student_inputs)

                    shape_weight = torch.tanh(shape_weight_raw) * dynamic_physical_width * SHAPE_SCALE_RATIO
                    delta_z_rel = torch.tanh(delta_z_scalar.view(-1, 1, 1, 1)) * MAX_Z_RATIO
                    z_global = Z_base * (1.0 + delta_z_rel)

                    D_pred = z_global + shape_weight

                    mask_crop_np = (mask_pred[0, 0] > 0.5).cpu().numpy().astype(np.uint8)
                    depth_crop_raw_np = D_pred[0, 0].cpu().numpy()
                    depth_crop_np = depth_crop_raw_np * mask_crop_np

                    M_inv = cv2.invertAffineTransform(M)
                    full_mask = cv2.warpAffine(mask_crop_np, M_inv, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
                    full_depth = cv2.warpAffine(depth_crop_np, M_inv, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
                    full_depth_raw = cv2.warpAffine(
                        depth_crop_raw_np, M_inv, (W, H),
                        flags=cv2.INTER_NEAREST, borderValue=0)

                    if not args.no_eval:
                        # 保留网络原始深度和预测mask，用于区分深度与门控误差。
                        raw_depth_eval = full_depth_raw.copy()
                        predicted_mask_eval = full_mask.astype(bool)

                    # 原始Student深度先乘预测mask，再映射回全图供追踪。
                    current_mask = full_mask.astype(bool)
                    current_depth = full_depth
                    previous_mask = current_mask

                else:
                    # ==========================================================
                    # 全图深度来源：dav2（预测）或 gt（oracle）
                    # ==========================================================
                    if args.depth_src == "dav2":
                        color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                        current_depth = dav2.infer_image(color_bgr, input_size=args.dav2_input_size).astype(np.float32)
                    else:  # gt
                        gt_depth_path_i = os.path.join(gt_depth_dir, f"{reader.id_strs[i]}.npy")
                        if os.path.exists(gt_depth_path_i):
                            gd = np.load(gt_depth_path_i).astype(np.float32)
                            if gd.shape != (H, W):
                                gd = cv2.resize(gd, (W, H), interpolation=cv2.INTER_NEAREST)
                            current_depth = gd
                        else:
                            current_depth = np.zeros((H, W), dtype=np.float32)
                            have_obs = False
                            print(f"⚠️ 帧{i:04d} 缺 GT 深度，该帧不更新位姿")

                    if not args.no_eval:
                        raw_depth_eval = current_depth.copy()
                    if args.use_gt_mask:
                        # GT mask：有 GT 位姿用 GT（oracle），否则用上一帧预测位姿渲染
                        mask_pose = frame_gt_pose if frame_gt_pose is not None else previous_pose
                        rd = render_gt_depth_map(mask_pose, reader.K, H, W, glctx, gt_mesh_tensors)
                        current_mask = rd > 0.001
                        current_depth = current_depth * current_mask  # 背景清零 -> crop 后无背景点云
                    else:
                        if args.depth_src == "gt":
                            current_mask = current_depth > 0.001      # GT 深度本身只渲染牙齿
                        else:
                            current_mask = bbox_rect_mask(previous_pose, reader.K, mesh.vertices, H, W)  # 仅参考，不门控
                    previous_mask = current_mask

                # 深度统计（门控后 / 目标区域内）
                valid_depth_vals = current_depth[current_mask]
                if len(valid_depth_vals) > 0:
                    print(f"🔍 帧{i:04d} [{tag}] 深度 | 均值: {np.mean(valid_depth_vals):.4f} | "
                          f"最小: {np.min(valid_depth_vals):.4f} | 最大: {np.max(valid_depth_vals):.4f}")

            with SuppressPrint():
                torch.cuda.synchronize()
                t_track_start = time.time()

                if i == 0:
                    if args.use_pred_init:
                        sys.stderr.write("\n🚀 启动零样本纯几何粗筛定位...\n")
                        initial_pose_cam2world = matcher.match(mask_crop_np, depth_crop_np)
                        if initial_pose_cam2world is not None:
                            sys.stderr.write("✅ 几何粗筛成功！利用预测深度图解算真实物理平移(T)...\n")
                            obj2cam_template = np.linalg.inv(initial_pose_cam2world)
                            R_pred = obj2cam_template[:3, :3]
                            valid_depth = current_depth[current_mask]
                            real_tz = np.median(valid_depth) if len(valid_depth) > 0 else 0.1
                            real_tx = (c_x - reader.K[0, 2]) * real_tz / reader.K[0, 0]
                            real_ty = (c_y - reader.K[1, 2]) * real_tz / reader.K[1, 1]
                            real_initial_pose = np.eye(4, dtype=np.float32)
                            real_initial_pose[:3, :3] = R_pred
                            real_initial_pose[:3, 3] = [real_tx, real_ty, real_tz]
                            est.pose_last = torch.tensor(real_initial_pose, device=device, dtype=torch.float32).unsqueeze(0)
                            pose_updated = True
                            pose_centered = est.track_one(rgb=color, depth=current_depth, K=reader.K, iteration=args.est_refine_iter)
                            pose = pose_centered @ est.get_tf_to_centered_mesh().data.cpu().numpy().reshape(4, 4)
                        else:
                            sys.stderr.write("❌ 警告：未匹配到任何有效模板！\n")
                            pose = np.eye(4)
                    else:
                        tf_c = est.get_tf_to_centered_mesh().data.cpu().numpy().reshape(4, 4)
                        est.pose_last = torch.tensor(pose_gt @ np.linalg.inv(tf_c), device=device, dtype=torch.float32).unsqueeze(0)
                        # GT首帧仅初始化；不精炼、不改变刚设定的内部位姿。
                        pose = pose_gt.copy()
                else:
                    if args.depth_src in ("dav2", "gt") and not have_obs:
                        pose = previous_pose  # 缺观测，沿用上一帧
                    else:
                        pose_updated = True
                        pose = est.track_one(rgb=color, depth=current_depth, K=reader.K, iteration=args.track_refine_iter) \
                                @ est.get_tf_to_centered_mesh().data.cpu().numpy().reshape(4, 4)

                previous_pose = pose

                torch.cuda.synchronize()
                t_track_end = time.time()
                track_time_ms = (t_track_end - t_track_start) * 1000.0

            torch.cuda.synchronize()
            t2 = time.time()
            frame_time_ms = (t2 - t1) * 1000.0

            # ---------- mask+depth 可视化 ----------
            if args.depth_src == "distill":
                mask_crop_255 = mask_crop_np * 255
                vis_rgb = cv2.cvtColor(rgb_crop.astype(np.uint8), cv2.COLOR_RGB2BGR)
                vis_mask_3c = cv2.cvtColor(mask_crop_255, cv2.COLOR_GRAY2BGR)
                vis_depth = get_auto_color(depth_crop_np, mask_crop_255)
                concat_img = np.hstack([vis_rgb, vis_mask_3c, vis_depth])
                cv2.putText(concat_img, 'Crop RGB', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(concat_img, 'Pred Mask', (170, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(concat_img, 'Pred Depth', (330, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imwrite(os.path.join(mask_depth_output_dir, f"{reader.id_strs[i]}.png"), concat_img)
            else:
                # 全图三联：门控版用 current_mask（GT silhouette）着色；
                # 无门控版用【当前帧追踪窗口 bbox】着色（此时 previous_pose 已更新为当前帧），
                # 深度只对窗口内 JET 着色、窗口外黑——既能看牙齿深度，也能看到窗口内混入的背景点。
                if args.use_gt_mask or args.depth_src == "gt":
                    vis_mask = current_mask
                else:
                    vis_mask = bbox_rect_mask(previous_pose, reader.K, mesh.vertices, H, W)
                save_full_triplet(os.path.join(mask_depth_output_dir, f"{reader.id_strs[i]}.png"),
                                  color, vis_mask, current_depth, W)

            vis = create_visualization(color, pose, to_origin, reader.K, bbox, fps=1/(t2-t1), render_3d=True, mesh_dir=os.path.dirname(args.mesh_file), main_mesh=mesh, center_pose=pose @ np.linalg.inv(to_origin))

            # ---------- Read-only evaluation, after the original tracking timer ----------
            frame_id = reader.id_strs[i]
            current_frame_number = frame_number_from_id(frame_id)
            previous_frame_number = frame_number_from_id(reader.id_strs[i - 1]) if i > 0 else None
            frame_gap = (current_frame_number - previous_frame_number
                         if current_frame_number is not None and previous_frame_number is not None else None)
            row = {name: np.nan for name in METRIC_NAMES}
            row.update(sequence_index=i + 1, frame_id=frame_id, frame_gap=frame_gap,
                       is_init=int(i == 0), pose_updated=int(pose_updated),
                       has_gt_pose=int(valid_pose_for_eval(frame_gt_pose)), has_gt_depth=0,
                       gt_mask_fallback=int(args.use_gt_mask and frame_gt_pose is None),
                       input_mask_kind=('gt_pose_render' if args.use_gt_mask and frame_gt_pose is not None else
                                        'previous_pose_render' if args.use_gt_mask else
                                        'predicted' if args.depth_src == 'distill' else
                                        'gt_depth_support' if args.depth_src == 'gt' else 'none'),
                       GTToothPixels=np.nan, DepthValidPixels=np.nan,
                       BoundaryPixels=np.nan, InteriorPixels=np.nan, Err2DPoints=0,
                       BackendIterations=0, BackendStatus='disabled' if backend_probe is None else 'not_run',
                       Track_ms=track_time_ms, FrontendTrack_ms=frame_time_ms, notes='')
            notes = []
            frame_error = None
            err_str = ' | 纯推理模式' if args.no_eval else ''

            if not args.no_eval:
                pose_is_valid = valid_pose_for_eval(pose)
                if not pose_is_valid:
                    notes.append('invalid_predicted_pose')
                if ball_centroids is not None and frame_name in gt_data and pose_is_valid:
                    try:
                        # Fixed IDs and the legacy unordered-point metric are both reported.
                        selected = [(index, j) for index, j in enumerate(ball_ids)
                                    if gt_data[frame_name].get(f'ball_{j}') is not None]
                        selected = [(index, j) for index, j in selected
                                    if np.asarray(gt_data[frame_name][f'ball_{j}']).shape == (2,)
                                    and np.isfinite(gt_data[frame_name][f'ball_{j}']).all()]
                        if selected:
                            model_balls = ball_centroids[[index for index, _ in selected]]
                            pts_gt = np.asarray([gt_data[frame_name][f'ball_{j}'] for _, j in selected], dtype=np.float32)
                            rvec_v, _ = cv2.Rodrigues(pose[:3, :3])
                            pts_proj, _ = cv2.projectPoints(model_balls, rvec_v, pose[:3, 3], reader.K, None)
                            pts_proj = pts_proj.reshape(-1, 2)
                            row['Err2D_ID_px'] = float(np.linalg.norm(pts_proj - pts_gt, axis=1).mean())
                            from scipy.optimize import linear_sum_assignment
                            distances = np.linalg.norm(pts_proj[:, None, :] - pts_gt[None, :, :], axis=2)
                            indices_a, indices_b = linear_sum_assignment(distances)
                            frame_error = float(distances[indices_a, indices_b].mean())
                            row['Err2D_px'] = frame_error
                            row['Err2DPoints'] = len(selected)
                            cv2.putText(vis, f'Err(set): {frame_error:.2f}px', (20, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                    except Exception as exc:
                        notes.append('Err2D:' + str(exc))

                # Pose metrics do not require a depth label.
                if row['has_gt_pose'] and pose_is_valid:
                    row['TE_mm'] = calc_te(pose, frame_gt_pose)
                    row['RE_deg'] = calc_re(pose, frame_gt_pose)
                    row['ADD_mm'] = calc_add(pose, frame_gt_pose, mesh.vertices)
                    translation_error = (pose[:3, 3] - frame_gt_pose[:3, 3]) * 1000.0
                    for name, value in zip(['TxError_mm', 'TyError_mm', 'TzError_mm'], translation_error):
                        row[name] = float(value)

                depth_gt_path = os.path.join(gt_depth_dir, f'{frame_id}.npy')
                if os.path.exists(depth_gt_path):
                    try:
                        depth_gt_val = np.load(depth_gt_path).astype(np.float32)
                        if depth_gt_val.ndim != 2:
                            raise ValueError(f'GT depth must be HxW, got {depth_gt_val.shape}')
                        if depth_gt_val.shape != (H, W):
                            depth_gt_val = cv2.resize(depth_gt_val, (W, H), interpolation=cv2.INTER_NEAREST)
                        gt_mask_val = valid_depth_mask(depth_gt_val)
                        row['has_gt_depth'] = int(gt_mask_val.any())
                        row.update(calculate_depth_metrics(current_depth, depth_gt_val,
                                                           args.eval_boundary_px, raw_depth_eval))
                        if gt_mask_val.any():
                            if row['input_mask_kind'] != 'none':
                                row['InputMaskIoU'] = mask_iou_for_eval(current_mask, gt_mask_val)
                            if predicted_mask_eval is not None:
                                row['PredMaskIoU'] = mask_iou_for_eval(predicted_mask_eval, gt_mask_val)
                            if pose_is_valid:
                                projected_depth = render_gt_depth_map(pose, reader.K, H, W, glctx, gt_mesh_tensors)
                                row['PoseIoU'] = mask_iou_for_eval(valid_depth_mask(projected_depth), gt_mask_val)
                                # Current output pose, not the previous-frame display bbox.
                                output_bbox = bbox_rect_mask(pose, reader.K, mesh.vertices, H, W)
                                row['PoseBBoxIoU'] = mask_iou_for_eval(output_bbox, gt_mask_val)
                            if backend_probe is not None:
                                row.update(backend_probe.evaluate(gt_mask_val))
                        else:
                            notes.append('empty_GT_depth_support')
                    except Exception as exc:
                        notes.append('depth_or_backend_eval:' + str(exc))
                        if backend_probe is not None:
                            row['BackendStatus'] = 'evaluation_error'
                elif backend_probe is not None:
                    row['BackendStatus'] = 'no_reference_depth'
                if i == 0 and not args.use_pred_init and backend_probe is not None:
                    row['BackendStatus'] = 'GT_init_no_refinement'
                elif not pose_updated and backend_probe is not None:
                    row['BackendStatus'] = 'no_pose_update'
                err_str = (f' | Err(set): {format_report_value(row["Err2D_px"])}px'
                           f' | ADD: {format_report_value(row["ADD_mm"])}mm'
                           f' | MAE(all/valid): {format_report_value(row["DepthMAE_all_mm"])}/'
                           f'{format_report_value(row["DepthMAE_valid_mm"])}mm'
                           f' | Coverage: {format_report_value(row["DepthCoverage_pct"])}%')
            row['notes'] = '; '.join(notes)
            if notes:
                print(f'⚠️ 评估提示 [{frame_id}]: {row["notes"]}')
            metric_rows.append(row)

            cv2.imwrite(os.path.join(img_output_dir, f"{reader.id_strs[i]}.png"), vis[..., ::-1])
            video_writer.write(vis[..., ::-1])

            print(f"🔹 帧 {i + 1:04d} [{frame_id}] 完成 | Track: {track_time_ms:.1f}ms | Total: {frame_time_ms:.1f}ms{err_str}")
            txt_file.write('\t'.join(format_report_value(row.get(name)) for name in FRAME_COLUMNS) + '\n')
            txt_file.flush()

    finally:
        if backend_probe is not None:
            backend_probe.close()
        video_writer.release()
        if 'txt_file' in locals() and not txt_file.closed:
            write_metric_summary(txt_file, metric_rows)
            txt_file.close()
