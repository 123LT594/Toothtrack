import os
import cv2
import json
import numpy as np
import trimesh
# 强制使用 OSMesa 离屏渲染
os.environ['PYOPENGL_PLATFORM'] = 'osmesa' 
import pyrender
from tqdm import tqdm

# ================= 配置：路径直接指向 tooth_gt =================
DATA_DIR = "./demo_data/tooth_gt"

MESH_TOOTH = os.path.join(DATA_DIR, "mesh/tooth.obj")
BALL_FILES = [os.path.join(DATA_DIR, f"mesh/{i}.obj") for i in [1, 2, 3, 4]]
K_PATH = os.path.join(DATA_DIR, "cam_K.txt") 
ANN_PATH = os.path.join(DATA_DIR, "annotations.json")
RGB_DIR = os.path.join(DATA_DIR, "rgb")

# 输出目录
DEPTH_540_OUT = os.path.join(DATA_DIR, "depth")
POSE_540_OUT = os.path.join(DATA_DIR, "pose")
VIS_540_OUT = os.path.join(DATA_DIR, "vis_check")
# =============================================================

def get_ball_centroids():
    centroids = []
    for f in BALL_FILES:
        m = trimesh.load(f, process=False)
        centroids.append(m.vertices.mean(axis=0))
    return np.array(centroids, dtype=np.float64)

def setup_renderer(mesh_path, K, h, w):
    mesh = trimesh.load(mesh_path)
    scene = pyrender.Scene(bg_color=[0, 0, 0])
    pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(pyrender_mesh, pose=np.eye(4))
    
    camera = pyrender.IntrinsicsCamera(
        fx=K[0,0], fy=K[1,1], cx=K[0,2], cy=K[1,2],
        znear=0.001, zfar=1000.0
    )
    cam_node = scene.add(camera, pose=np.eye(4))
    return scene, pyrender.OffscreenRenderer(w, h), cam_node

def main():
    # 1. 创建输出文件夹
    for d in [DEPTH_540_OUT, POSE_540_OUT, VIS_540_OUT]: 
        os.makedirs(d, exist_ok=True)
    
    kpts_3d = get_ball_centroids()
    
    # 2. 直接读取 540p 的内参
    K_540 = np.loadtxt(K_PATH, dtype=np.float64)
    
    with open(ANN_PATH, 'r') as f:
        annotations = json.load(f).get("annotations", {})
    
    h_540, w_540 = 540, 960
    scene, renderer, cam_node = setup_renderer(MESH_TOOTH, K_540, h_540, w_540)

    tooth_mesh = trimesh.load(MESH_TOOTH, process=False)
    tooth_v_samples, _ = trimesh.sample.sample_surface(tooth_mesh, 3000)
    tooth_v_samples = tooth_v_samples.astype(np.float32)

    gold_errors = [] 
    bad_count = 0
    total_processed = 0

    print(f"\n🚀 开始处理数据 (输入图片、内参、标注均已是纯正的 540p)...")
    pbar = tqdm(annotations.items(), desc="Generating 540p GT")
    
    for frame_name, ball_anns in pbar:
        img_path = os.path.join(RGB_DIR, frame_name)
        img_540 = cv2.imread(img_path)
        if img_540 is None: continue
        
        # 简单安全校验
        if img_540.shape[0] != 540 or img_540.shape[1] != 960:
            print(f"⚠️ 警告: 图片 {frame_name} 的分辨率不是 960x540！实际是 {img_540.shape[1]}x{img_540.shape[0]}")
        
        try:
            # 🌟 核心修改 2：因为读取的已经是 540p 的 JSON，直接使用，不再乘以 0.5！
            pts_2d_540 = np.array([ball_anns[f'ball_{i}'] for i in range(1, 5)], dtype=np.float64)
        except: continue

        # ================= PnP 匹配与渲染 =================
        ret_i, r_i, t_i = cv2.solvePnP(kpts_3d, pts_2d_540, K_540, None, flags=cv2.SOLVEPNP_EPNP)
        ret, rvec, tvec = cv2.solvePnP(kpts_3d, pts_2d_540, K_540, None, flags=cv2.SOLVEPNP_ITERATIVE, 
                                      useExtrinsicGuess=True, rvec=r_i, tvec=t_i)
        
        if ret:
            proj_b, _ = cv2.projectPoints(kpts_3d, rvec, tvec, K_540, None)
            error = np.mean(np.linalg.norm(pts_2d_540 - proj_b.squeeze(), axis=1))
            
            if error > 2.0:
                tqdm.write(f"⚠️  High Error: {frame_name} | {error:.2f} px (540p scale)")
                bad_count += 1
            elif error <= 2.0:
                gold_errors.append(error)
            
            # ✅ 位姿：保存为安全的 float32
            rmat, _ = cv2.Rodrigues(rvec)
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3], pose[:3, 3] = rmat, tvec.ravel()
            np.save(os.path.join(POSE_540_OUT, frame_name.replace('.png', '.npy')), pose)

            # ✅ 深度：保存为省空间的 float16
            cv_to_gl = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])
            cam_pose_gl = np.linalg.inv(pose) @ cv_to_gl
            scene.set_pose(cam_node, cam_pose_gl)
            depth = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
            np.save(os.path.join(DEPTH_540_OUT, frame_name.replace('.png', '.npy')), depth.astype(np.float16))

            # 画图可视化验证
            proj_m, _ = cv2.projectPoints(tooth_v_samples, rvec, tvec, K_540, None)
            for p in proj_m.squeeze():
                cv2.circle(img_540, (int(p[0]), int(p[1])), 1, (0, 255, 0), -1)
            for i, p in enumerate(proj_b.squeeze()):
                cv2.drawMarker(img_540, (int(p[0]), int(p[1])), (0, 0, 255), cv2.MARKER_CROSS, 8, 2)
                cv2.circle(img_540, (int(pts_2d_540[i][0]), int(pts_2d_540[i][1])), 5, (0, 255, 255), 1)
            
            cv2.imwrite(os.path.join(VIS_540_OUT, frame_name), img_540)
            total_processed += 1

    # 计算详细统计信息
    gold_count = len(gold_errors)
    avg_gold_err = np.mean(gold_errors) if gold_count > 0 else 0
    med_gold_err = np.median(gold_errors) if gold_count > 0 else 0

    print("\n" + "="*50)
    print(f"✅ 处理完毕！")
    print(f"📊 总处理帧数: {total_processed} | 黄金帧 (误差 <= 2px): {gold_count}")
    print(f"✨ 黄金帧平均误差: {avg_gold_err:.4f} px (540p尺度)")
    print(f"📏 黄金帧中位数误差: {med_gold_err:.4f} px (540p尺度)")
    print(f"❌ 高误差帧 (误差 > 2px): {bad_count}")
    print("="*50)

if __name__ == "__main__":
    main()