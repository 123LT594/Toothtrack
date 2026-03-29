import trimesh
import os

def merge_dental_meshes(mesh_dir, output_path):
    # 1. 定义所有组件路径
    parts = ["tooth.obj", "1.obj", "2.obj", "3.obj", "4.obj"]
    scene = trimesh.Scene()

    print("🚀 开始合并模型组件...")
    for part in parts:
        path = os.path.join(mesh_dir, part)
        if os.path.exists(path):
            # process=False 保证不改变原始坐标系
            mesh = trimesh.load(path, process=False)
            scene.add_geometry(mesh)
            print(f"✅ 已加入: {part}")
        else:
            print(f"⚠️ 找不到组件: {path}")

    # 2. 导出合并后的单体模型
    combined = scene.dump(concatenate=True)
    combined.export(output_path)
    print(f"🎉 任务完成！新模型已保存至: {output_path}")

# 使用示例
mesh_dir = "./demo_data/tooth/mesh"
merge_dental_meshes(mesh_dir, "./demo_data/tooth_gt/mesh/teeth.obj")