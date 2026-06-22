准备训练用的cad渲染图
import os
import sys
import cv2
import numpy as np
import trimesh
import nvdiffrast.torch as dr
from tqdm import tqdm
import tempfile
import shutil

code_dir = "/root/Toothtrack"
sys.path.append(code_dir)
from utils.render_3d import create_visualization

def render_golden_dataset():
    dataset_dir = "/root/lanyun-tmp/golden_dataset"
    pose_dir = os.path.join(dataset_dir, "pose")
    rgb_dir = os.path.join(dataset_dir, "rgb")
    rendered_dir = os.path.join(dataset_dir, "rendered_rgb")
    os.makedirs(rendered_dir, exist_ok=True)
    
    # 你原始真实的 mesh 目录 (包含 tooth.obj 和 1,2,3,4.obj)
    orig_mesh_dir = os.path.join(code_dir, "demo_data/tooth/mesh")
    mesh_file = os.path.join(orig_mesh_dir, "tooth.obj")
    
    K = np.array([[2866.3146, 0.0, 480.0],
                  [0.0, 2866.3146, 270.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    W, H = 960, 540

    glctx = dr.RasterizeCudaContext()
    mesh = trimesh.load(mesh_file)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.png', '.jpg'))])
    print(f"🔥 开始渲染 (共 {len(files)} 帧)...")

    # ====================================================================
    # 🌟 绝杀：创建沙盒，克隆全套模型，把钢珠“流放”到外太空！
    # ====================================================================
    with tempfile.TemporaryDirectory() as temp_root:
        temp_mesh_dir = os.path.join(temp_root, "mesh")
        shutil.copytree(orig_mesh_dir, temp_mesh_dir)

        # 遍历 4 个小球，修改它们的三维顶点坐标，让它们飞出摄像机视野！
        for j in [1, 2, 3, 4]:
            ball_file = os.path.join(temp_mesh_dir, f"{j}.obj")
            if os.path.exists(ball_file):
                ball_mesh = trimesh.load(ball_file)
                # 平移 9999 米！它依然存在，依然占着颜色名额，但绝对看不见！
                ball_mesh.apply_translation([9999.0, 9999.0, 9999.0])
                ball_mesh.export(ball_file)

        for f in tqdm(files, desc="Rendering A-Images"):
            base_name = os.path.splitext(f)[0]
            pose_file = os.path.join(pose_dir, base_name + ".npy")
            if not os.path.exists(pose_file): continue
                
            pose = np.load(pose_file)
            color_dummy = np.zeros((H, W, 3), dtype=np.uint8) 
            
            # 传入这个被动了手脚的沙盒目录
            vis = create_visualization(
                color_dummy, 
                pose, 
                to_origin, 
                K, 
                bbox, 
                fps=0, 
                render_3d=True, 
                mesh_dir=temp_mesh_dir, # <-- 这里！
                main_mesh=mesh, 
                center_pose=pose @ np.linalg.inv(to_origin)
            )
            
            cv2.imwrite(os.path.join(rendered_dir, f), vis)

    print("\n✅ 纯净版 A 图渲染完毕！文件数量对上了，颜色对了，而且画面里绝对没有钢珠！")

if __name__ == "__main__":
    render_golden_dataset()