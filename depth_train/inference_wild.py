import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import torch
import numpy as np
import argparse

from learning.models.student_depth_net import StudentDepthNet
from learning.training.training_config import DISTILL_PHYSICAL_WIDTH, DISTILL_PHYSICAL_THICKNESS, DISTILL_K_BASE

def inference_on_wild_image(img_path, roi_box=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型
    ckpt_path = "/root/lanyun-tmp/models/stage1_distill/models/student_stage1_ep99.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到权重文件: {ckpt_path}")

    model = StudentDepthNet().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval() 
    
    # 2. 加载野生图像
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"无法读取图像: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_bgr.shape[:2]
    
    K_base = np.array(DISTILL_K_BASE, dtype=np.float32)
    
    # ========================================================
    # 🌟 终端输入或命令行参数获取 BBox
    # ========================================================
    if roi_box is not None:
        x_min, y_min, w, h = roi_box
    else:
        print(f"📷 当前图像尺寸: {W}x{H}")
        print("💡 请输入牙齿在图中的边界框坐标 (x_min, y_min, w, h)")
        print("   (如果不确定，直接回车将自动截取图片正中心区域测试)")
        user_input = input("👉 请输入 (例如: 403,177,248,206 或直接回车): ").strip()
        
        if user_input == "":
            w, h = 200, 200
            x_min, y_min = (W - w) // 2, (H - h) // 2
            print(f"⚙️ 已自动采用中心区域默认 BBox: x={x_min}, y={y_min}, w={w}, h={h}")
        else:
            try:
                x_min, y_min, w, h = [int(v.strip()) for v in user_input.split(',')]
            except:
                print("⚠️ 输入格式错误！将自动采用中心默认区域。")
                w, h = 200, 200
                x_min, y_min = (W - w) // 2, (H - h) // 2

    # ========================================================
    # 🎯 绝对严格听从你的指令：直接以你输入的 x_min, y_min, w, h 为裁剪窗口
    # ========================================================
    side = max(w, h)
    scale = side / 160.0

    # 构造仿射矩阵：确保原图的 (x_min, y_min) 严格对应输出 160x160 的 (0, 0)
    M = np.array([
        [1.0 / scale, 0, -x_min / scale],
        [0, 1.0 / scale, -y_min / scale]
    ], dtype=np.float32)

    crop_size = side
    
    rgb_crop = cv2.warpAffine(img_rgb, M, (160, 160), flags=cv2.INTER_LINEAR, borderValue=(0,0,0))
    rgb_tensor = torch.from_numpy(rgb_crop).float().permute(2,0,1) / 255.0
    
    # 生成射线地图
    M_3x3 = np.vstack([M, [0, 0, 1]])
    K_crop = M_3x3 @ K_base
    K_inv = np.linalg.inv(K_crop)
    u, v = np.meshgrid(np.arange(160), np.arange(160))
    uv1 = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3)
    unnorm_rays = (K_inv @ uv1.T).T.reshape(160, 160, 3)
    ray_map = unnorm_rays / np.linalg.norm(unnorm_rays, axis=-1, keepdims=True)
    ray_tensor = torch.from_numpy(ray_map).float().permute(2,0,1)
    
    inputs_6c = torch.cat([rgb_tensor, ray_tensor], dim=0).unsqueeze(0).to(device)
    Z_base = K_base[0, 0] * (DISTILL_PHYSICAL_WIDTH / crop_size)
    
    # ========================================================
    # 模型推理！
    # ========================================================
    with torch.no_grad():
        shape_weight_raw, mask_pred, delta_z_scalar = model(inputs_6c)
        
        MAX_Z_CORRECTION = 0.03
        shape_weight = torch.tanh(shape_weight_raw)
        delta_z = torch.tanh(delta_z_scalar.view(-1, 1, 1, 1)) * MAX_Z_CORRECTION
        D_pred = Z_base + delta_z + shape_weight * DISTILL_PHYSICAL_THICKNESS
        
        mask_crop_np = (mask_pred[0, 0].cpu().numpy() * 255).astype(np.uint8)
        depth_crop_np = D_pred[0, 0].cpu().numpy()
        
    # --- 渲染出图 ---
    vis_rgb = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR)
    vis_mask_3c = cv2.cvtColor(mask_crop_np, cv2.COLOR_GRAY2BGR)
    
    def get_auto_color(d_np, mask_uint8):
        m_np = mask_uint8 > 127
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

    vis_depth = get_auto_color(depth_crop_np, mask_crop_np)
    
    concat_img = np.hstack([vis_rgb, vis_mask_3c, vis_depth])
    cv2.putText(concat_img, 'Crop RGB', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(concat_img, 'Pred Mask', (170, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(concat_img, 'Pred Depth', (330, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    out_path = "result.png"
    cv2.imwrite(out_path, concat_img)
    print(f"✅ 野生图像推理完成！结果已保存至: {out_path}")
    print(f"📌 模型推断该牙齿距相机的 Z_base: {Z_base:.4f} 米")
    print(f"⛰️ 预测最大物理起伏落差: {depth_crop_np[mask_crop_np>127].max() - depth_crop_np[mask_crop_np>127].min():.4f} 米")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="输入测试图像路径")
    parser.add_argument("--bbox", type=int, nargs=4, default=None, help="可选：直接传入 x_min y_min w h")
    args = parser.parse_args()
    
    if os.path.exists(args.image):
        inference_on_wild_image(args.image, roi_box=args.bbox)
    else:
        print(f"🚨 找不到指定的图像路径: {args.image}")