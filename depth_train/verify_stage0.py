import os
import sys
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset_synthetic import SyntheticPretrainDataset
from learning.models.student_depth_net import StudentDepthNet
from learning.training.training_config import MAX_Z_RATIO, SHAPE_SCALE_RATIO


def verify_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 权重路径和输出目录
    ckpt_path = "/root/lanyun-tmp/models/stage0_pretrain/models/student_stage0_ep49.pth"
    out_dir = "/root/lanyun-tmp/models/stage0_pretrain/verify_results"
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到权重文件: {ckpt_path}，请确认是否跑完了第49个Epoch！")

    print("⏳ 正在加载预训练模型...")
    model = StudentDepthNet().to(device)

    # 🌟 兼容两种checkpoint格式：
    # 1. 新格式：完整checkpoint（包含model_state_dict、optimizer等）
    # 2. 旧格式：直接是模型state_dict
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ 加载完整checkpoint（epoch={checkpoint.get('epoch', '?')}）")
    else:
        model.load_state_dict(checkpoint)
        print("✅ 加载旧格式checkpoint（纯模型权重）")

    model.eval()

    # 2. 加载测试数据集 (is_training=False 关闭数据增强)
    print("⏳ 正在初始化测试数据...")
    dataset = SyntheticPretrainDataset(is_training=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    print(f"🚀 开始推理验证！结果将保存在: {out_dir}")

    # 🌟 统计指标累加器（跑50个样本算平均，更有统计意义）
    num_eval = 50
    total_mae = 0.0
    total_delta_z_error = 0.0
    total_mask_iou = 0.0
    total_shape_std = 0.0
    valid_count = 0

    # 只保存前10张可视化图
    num_vis = 10

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_eval:
                break

            inputs_6c = batch["inputs_6c"].to(device).float()
            rgb_crop = batch["rgb_crop"].to(device).float()
            depth_gt = batch["depth_gt"].to(device).float()
            mask_gt = batch["mask_gt"].to(device).float()
            Z_base = batch["Z_base"].to(device).view(-1, 1, 1, 1).float()
            dynamic_width = batch["mesh_width"].to(device).view(-1, 1, 1, 1).float()

            # 模型推理
            shape_weight_raw, mask_pred, delta_z_scalar = model(inputs_6c)

            # 🌟 和训练完全一致：shape缩放
            shape_weight = torch.tanh(shape_weight_raw) * dynamic_width * SHAPE_SCALE_RATIO

            # 🌟 硬约束：减掉shape在GT mask内的均值，和训练完全一致
            shape_mean_val = (shape_weight * mask_gt).sum(dim=[1, 2, 3], keepdim=True) / (
                mask_gt.sum(dim=[1, 2, 3], keepdim=True) + 1e-8
            )
            shape_weight = shape_weight - shape_mean_val

            # delta_z和最终深度
            delta_z_rel = torch.tanh(delta_z_scalar.view(-1, 1, 1, 1)) * MAX_Z_RATIO
            z_global = Z_base * (1.0 + delta_z_rel)
            D_pred = z_global + shape_weight

            # =============== 📊 核心诊断 ===============
            valid_gt_mask = mask_gt[0, 0] > 0.5
            if valid_gt_mask.sum() > 0:
                # GT中位数和delta_z_gt
                gt_median = torch.median(depth_gt[0, 0][valid_gt_mask]).item()
                delta_z_gt = (gt_median - Z_base.item()) / Z_base.item() * 100
                delta_z_pred = delta_z_rel[0, 0, 0, 0].item() * 100
                dz_error = abs(delta_z_pred - delta_z_gt)

                # MAE
                mae = torch.abs(D_pred[0, 0] - depth_gt[0, 0])[valid_gt_mask].mean().item() * 1000

                # mask IoU
                pred_mask_bin = mask_pred[0, 0] > 0.5
                intersection = torch.logical_and(pred_mask_bin, valid_gt_mask).sum().item()
                union = torch.logical_or(pred_mask_bin, valid_gt_mask).sum().item()
                iou = intersection / (union + 1e-8)

                # shape标准差
                shape_vals = shape_weight[0, 0][valid_gt_mask]
                shape_std = shape_vals.std().item() * 1000

                # 累加统计
                total_mae += mae
                total_delta_z_error += dz_error
                total_mask_iou += iou
                total_shape_std += shape_std
                valid_count += 1

                # 打印前10个样本的详细诊断
                if i < num_vis:
                    print(f"\n--- 🦷 Sample {i:02d} 深度诊断 ---")
                    print(f"📌 Z_base 基准距离 : {Z_base.item():.4f} m")
                    print(f"🎯 GT 中位数      : {gt_median:.4f} m")
                    print(f"🤖 Pred 中位数    : {torch.median(D_pred[0, 0][valid_gt_mask]).item():.4f} m")
                    print(f"📐 delta_z_gt     : {delta_z_gt:+.2f}%")
                    print(f"📐 delta_z_pred   : {delta_z_pred:+.2f}%")
                    print(f"❌ delta_z_error  : {dz_error:.2f}%")
                    print(f"📏 depth MAE      : {mae:.2f} mm")
                    print(f"🎭 mask IoU       : {iou:.4f}")
                    print(f"⛰️ shape std      : {shape_std:.2f} mm")

                    # 物理起伏对比
                    gt_min = depth_gt[0, 0][valid_gt_mask].min().item()
                    gt_max = depth_gt[0, 0][valid_gt_mask].max().item()
                    pred_min = D_pred[0, 0][valid_gt_mask].min().item()
                    pred_max = D_pred[0, 0][valid_gt_mask].max().item()
                    print(f"⛰️ GT 起伏范围    : [{gt_min:.4f}, {gt_max:.4f}] (落差: {gt_max-gt_min:.4f})")
                    print(f"⛰️ Pred起伏范围   : [{pred_min:.4f}, {pred_max:.4f}] (落差: {pred_max-pred_min:.4f})")

            # =============== 🎨 可视化（前10个样本） ===============
            if i < num_vis:
                vis_rgb = (rgb_crop[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
                vis_mask_pred = (mask_pred[0, 0].cpu().numpy() * 255).astype(np.uint8)
                vis_mask_gt = (mask_gt[0, 0].cpu().numpy() * 255).astype(np.uint8)
                vis_mask_pred_3c = cv2.cvtColor(vis_mask_pred, cv2.COLOR_GRAY2BGR)
                vis_mask_gt_3c = cv2.cvtColor(vis_mask_gt, cv2.COLOR_GRAY2BGR)

                def get_auto_color(depth_t, mask_t):
                    """绝对自适应拉伸：只要有形状，就画出彩虹"""
                    d_np = depth_t[0, 0].cpu().numpy()
                    m_np = mask_t[0, 0].cpu().numpy() > 0.5
                    vis = np.zeros_like(d_np, dtype=np.uint8)
                    if m_np.sum() > 0:
                        valid = d_np[m_np]
                        p_min, p_max = np.percentile(valid, 2), np.percentile(valid, 98)
                        if p_max - p_min > 1e-4:
                            norm = np.clip((d_np - p_min) / (p_max - p_min), 0, 1)
                            vis = (norm * 255).astype(np.uint8)
                        else:
                            vis[m_np] = 127
                    color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                    color[~m_np] = 0
                    return color

                def get_error_color(pred_t, gt_t, mask_t, max_err=0.01):
                    """误差图：蓝色=误差小，红色=误差大"""
                    err = np.abs(pred_t[0, 0].cpu().numpy() - gt_t[0, 0].cpu().numpy())
                    m_np = mask_t[0, 0].cpu().numpy() > 0.5
                    vis = np.zeros_like(err, dtype=np.uint8)
                    if m_np.sum() > 0:
                        norm = np.clip(err / max_err, 0, 1)
                        vis = (norm * 255).astype(np.uint8)
                    color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                    color[~m_np] = 0
                    return color

                # 都用同一张 GT Mask 确保边缘形状对齐
                vis_depth_pred = get_auto_color(D_pred, mask_gt)
                vis_depth_gt = get_auto_color(depth_gt, mask_gt)
                vis_error = get_error_color(D_pred, depth_gt, mask_gt, max_err=0.01)

                # 拼图：RGB | Pred Mask | GT Mask | Pred Depth | GT Depth | Error
                concat_img = np.hstack([
                    vis_bgr, vis_mask_pred_3c, vis_mask_gt_3c,
                    vis_depth_pred, vis_depth_gt, vis_error
                ])

                # 文字标签
                labels = ['RGB', 'Pred Mask', 'GT Mask', 'Pred Depth', 'GT Depth', 'Error']
                for j, label in enumerate(labels):
                    cv2.putText(concat_img, label, (j * 160 + 10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                # 在图上标注关键指标
                info_text = f"dz_gt={delta_z_gt:+.1f}% dz_pred={delta_z_pred:+.1f}% err={dz_error:.1f}% MAE={mae:.1f}mm"
                cv2.putText(concat_img, info_text, (10, concat_img.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

                save_path = os.path.join(out_dir, f"test_result_{i:02d}.png")
                cv2.imwrite(save_path, concat_img)

    # =============== 📊 整体统计结果 ===============
    print("\n" + "=" * 60)
    print("📊 阶段0验证整体统计结果（{}个有效样本）".format(valid_count))
    print("=" * 60)
    if valid_count > 0:
        print(f"📏 平均 depth MAE       : {total_mae / valid_count:.2f} mm")
        print(f"❌ 平均 delta_z_error  : {total_delta_z_error / valid_count:.2f} %")
        print(f"🎭 平均 mask IoU        : {total_mask_iou / valid_count:.4f}")
        print(f"⛰️ 平均 shape std       : {total_shape_std / valid_count:.2f} mm")
    print("=" * 60)
    print(f"🎨 可视化结果保存在: {out_dir}")
    print("✅ 验证完成！")


if __name__ == "__main__":
    verify_model()
