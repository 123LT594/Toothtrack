import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset_distill import DualDistillDataset
from learning.models.student_depth_net import StudentDepthNet
from learning.training.training_config import MAX_Z_RATIO, SHAPE_SCALE_RATIO


def verify_stage1(ckpt_path, dataset_path, dataset_name, out_dir, num_eval=50, num_vis=10):
    """
    验证阶段1蒸馏模型在单个真实数据集上的效果
    Args:
        ckpt_path: 模型权重路径
        dataset_path: 数据集路径
        dataset_name: 数据集名称（用于打印和保存）
        out_dir: 输出目录
        num_eval: 验证样本数
        num_vis: 可视化样本数
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*60}")
    print(f"📊 验证数据集: {dataset_name}")
    print(f"📁 数据路径: {dataset_path}")
    print(f"🔍 验证样本数: {num_eval}")
    print(f"{'='*60}")
    
    # 1. 加载模型
    model = StudentDepthNet().to(device)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到权重文件: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"✅ 加载完整checkpoint（epoch={ckpt.get('epoch', '?')}）")
    else:
        model.load_state_dict(ckpt)
        print("✅ 加载旧格式checkpoint（纯模型权重）")
    model.eval()
    
    # 2. 加载真实数据集（is_training=False 关闭数据增强）
    dataset = DualDistillDataset(data_dir=dataset_path, is_training=False, dataset_id=0)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    
    # 3. 统计指标累加器
    total_mae = 0.0
    total_delta_z_error = 0.0
    total_mask_iou = 0.0
    total_shape_std = 0.0
    total_delta_z_gt = 0.0
    total_delta_z_pred = 0.0
    valid_count = 0
    
    # 输出子目录
    dataset_out_dir = os.path.join(out_dir, dataset_name)
    os.makedirs(dataset_out_dir, exist_ok=True)
    
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
            
            # shape缩放（和训练完全一致）
            shape_weight = torch.tanh(shape_weight_raw) * dynamic_width * SHAPE_SCALE_RATIO
            
            # 硬约束：减掉shape在GT mask内的均值（和训练完全一致）
            shape_mean_val = (shape_weight * mask_gt).sum(dim=[1,2,3], keepdim=True) / (
                mask_gt.sum(dim=[1,2,3], keepdim=True) + 1e-8
            )
            shape_weight = shape_weight - shape_mean_val
            
            # delta_z和最终深度
            delta_z_rel = torch.tanh(delta_z_scalar.view(-1, 1, 1, 1)) * MAX_Z_RATIO
            z_global = Z_base * (1.0 + delta_z_rel)
            D_pred = z_global + shape_weight
            
            # =============== 📊 指标计算 ===============
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
                total_delta_z_gt += delta_z_gt
                total_delta_z_pred += delta_z_pred
                valid_count += 1
                
                # 打印前num_vis个样本的详细诊断
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
            
            # =============== 🎨 可视化（前num_vis个样本） ===============
            if i < num_vis:
                vis_rgb = (rgb_crop[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
                vis_mask_pred = (mask_pred[0, 0].cpu().numpy() * 255).astype(np.uint8)
                vis_mask_gt = (mask_gt[0, 0].cpu().numpy() * 255).astype(np.uint8)
                vis_mask_pred_3c = cv2.cvtColor(vis_mask_pred, cv2.COLOR_GRAY2BGR)
                vis_mask_gt_3c = cv2.cvtColor(vis_mask_gt, cv2.COLOR_GRAY2BGR)
                
                def get_auto_color(depth_t, mask_t):
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
                    err = np.abs(pred_t[0, 0].cpu().numpy() - gt_t[0, 0].cpu().numpy())
                    m_np = mask_t[0, 0].cpu().numpy() > 0.5
                    vis = np.zeros_like(err, dtype=np.uint8)
                    if m_np.sum() > 0:
                        norm = np.clip(err / max_err, 0, 1)
                        vis = (norm * 255).astype(np.uint8)
                    color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                    color[~m_np] = 0
                    return color
                
                vis_depth_pred = get_auto_color(D_pred, mask_gt)
                vis_depth_gt = get_auto_color(depth_gt, mask_gt)
                vis_error = get_error_color(D_pred, depth_gt, mask_gt, max_err=0.01)
                
                # 拼图：RGB | Pred Mask | GT Mask | Pred Depth | GT Depth | Error
                concat_img = np.hstack([
                    vis_bgr, vis_mask_pred_3c, vis_mask_gt_3c,
                    vis_depth_pred, vis_depth_gt, vis_error
                ])
                
                labels = ['RGB', 'Pred Mask', 'GT Mask', 'Pred Depth', 'GT Depth', 'Error']
                for j, label in enumerate(labels):
                    cv2.putText(concat_img, label, (j * 160 + 10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                # 标注关键指标
                info_text = f"dz_gt={delta_z_gt:+.1f}% dz_pred={delta_z_pred:+.1f}% err={dz_error:.1f}% MAE={mae:.1f}mm"
                cv2.putText(concat_img, info_text, (10, concat_img.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
                
                save_path = os.path.join(dataset_out_dir, f"test_result_{i:02d}.png")
                cv2.imwrite(save_path, concat_img)
    
    # =============== 📊 整体统计结果 ===============
    print(f"\n{'='*60}")
    print(f"📊 {dataset_name} 验证整体统计结果（{valid_count}个有效样本）")
    print(f"{'='*60}")
    results = {}
    if valid_count > 0:
        avg_mae = total_mae / valid_count
        avg_dz_error = total_delta_z_error / valid_count
        avg_iou = total_mask_iou / valid_count
        avg_shape_std = total_shape_std / valid_count
        avg_dz_gt = total_delta_z_gt / valid_count
        avg_dz_pred = total_delta_z_pred / valid_count
        
        print(f"📏 平均 depth MAE       : {avg_mae:.2f} mm")
        print(f"❌ 平均 delta_z_error  : {avg_dz_error:.2f} %")
        print(f"📐 平均 delta_z_gt     : {avg_dz_gt:+.2f} %")
        print(f"📐 平均 delta_z_pred   : {avg_dz_pred:+.2f} %")
        print(f"🎭 平均 mask IoU        : {avg_iou:.4f}")
        print(f"⛰️ 平均 shape std       : {avg_shape_std:.2f} mm")
        
        results = {
            'mae': avg_mae,
            'dz_error': avg_dz_error,
            'dz_gt': avg_dz_gt,
            'dz_pred': avg_dz_pred,
            'iou': avg_iou,
            'shape_std': avg_shape_std,
            'valid_count': valid_count,
        }
    else:
        print("⚠️ 没有有效样本！")
    print(f"{'='*60}")
    print(f"🎨 可视化结果保存在: {dataset_out_dir}")
    
    return results


def main():
    # ========== 配置 ==========
    # 阶段1权重路径（用最后一轮的权重，也可以改成EMA权重）
    ckpt_path = "/root/lanyun-tmp/models/stage1_distill/models/student_stage1_ep99.pth"
    
    # 两个真实数据集路径
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_old_path = os.path.abspath(os.path.join(_current_dir, "../../lanyun-tmp/golden_dataset"))
    dataset_new_path = os.path.abspath(os.path.join(_current_dir, "../../lanyun-tmp/golden_dataset_wxb"))
    
    # 输出目录
    out_dir = "/root/lanyun-tmp/models/stage1_distill/verify_results"
    os.makedirs(out_dir, exist_ok=True)
    
    # 验证参数
    num_eval = 50   # 每个数据集验证50个样本
    num_vis = 10    # 每个数据集可视化10个样本
    
    # ========== 验证两个数据集 ==========
    print("⏳ 正在加载阶段1蒸馏模型...")
    print(f"🔍 权重路径: {ckpt_path}")
    
    # 验证旧数据集（2800长焦，10mm工作距离）
    results_old = verify_stage1(
        ckpt_path=ckpt_path,
        dataset_path=dataset_old_path,
        dataset_name="DS0_Old_2800mm_focal",
        out_dir=out_dir,
        num_eval=num_eval,
        num_vis=num_vis
    )
    
    # 验证新数据集（500短焦，<5mm工作距离）
    results_new = verify_stage1(
        ckpt_path=ckpt_path,
        dataset_path=dataset_new_path,
        dataset_name="DS1_New_500mm_focal",
        out_dir=out_dir,
        num_eval=num_eval,
        num_vis=num_vis
    )
    
    # ========== 两个数据集对比总结 ==========
    print(f"\n\n{'#'*60}")
    print(f"# 📊 阶段1蒸馏验证 — 双数据集对比总结")
    print(f"{'#'*60}")
    print(f"{'指标':<25} {'旧数据集(2800长焦)':<20} {'新数据集(500短焦)':<20} {'差异':<10}")
    print(f"{'-'*75}")
    
    if results_old and results_new:
        print(f"{'depth MAE (mm)':<25} {results_old['mae']:<20.2f} {results_new['mae']:<20.2f} {abs(results_old['mae']-results_new['mae']):<10.2f}")
        print(f"{'delta_z_error (%)':<25} {results_old['dz_error']:<20.2f} {results_new['dz_error']:<20.2f} {abs(results_old['dz_error']-results_new['dz_error']):<10.2f}")
        print(f"{'delta_z_gt (%)':<25} {results_old['dz_gt']:<20.2f} {results_new['dz_gt']:<20.2f} {abs(results_old['dz_gt']-results_new['dz_gt']):<10.2f}")
        print(f"{'delta_z_pred (%)':<25} {results_old['dz_pred']:<20.2f} {results_new['dz_pred']:<20.2f} {abs(results_old['dz_pred']-results_new['dz_pred']):<10.2f}")
        print(f"{'mask IoU':<25} {results_old['iou']:<20.4f} {results_new['iou']:<20.4f} {abs(results_old['iou']-results_new['iou']):<10.4f}")
        print(f"{'shape std (mm)':<25} {results_old['shape_std']:<20.2f} {results_new['shape_std']:<20.2f} {abs(results_old['shape_std']-results_new['shape_std']):<10.2f}")
        print(f"{'有效样本数':<25} {results_old['valid_count']:<20d} {results_new['valid_count']:<20d}")
    
    print(f"{'#'*60}")
    
    # 判断是否达标
    if results_old and results_new:
        avg_dz_error = (results_old['dz_error'] + results_new['dz_error']) / 2
        avg_mae = (results_old['mae'] + results_new['mae']) / 2
        
        print(f"\n📋 达标判断：")
        print(f"  - 平均 delta_z_error: {avg_dz_error:.2f}% (目标: <2%) {'✅' if avg_dz_error < 2 else '⚠️'}")
        print(f"  - 平均 depth MAE: {avg_mae:.2f}mm (目标: <1mm) {'✅' if avg_mae < 1 else '⚠️'}")
        print(f"  - 双数据集差异(delta_z_error): {abs(results_old['dz_error']-results_new['dz_error']):.2f}% (目标: <1%) {'✅' if abs(results_old['dz_error']-results_new['dz_error']) < 1 else '⚠️'}")
        
        if avg_dz_error < 2 and avg_mae < 1:
            print(f"\n✅ 验证通过！可以开始run推理测试了！")
        else:
            print(f"\n⚠️ 验证未完全达标，建议继续训练或调整参数后再推理。")
    
    print(f"\n📁 所有验证结果保存在: {out_dir}")


if __name__ == "__main__":
    main()
