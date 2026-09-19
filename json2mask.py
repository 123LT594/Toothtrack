import os
import json
import numpy as np
import cv2

def convert_labelme_json_to_mask(json_dir, output_dir, target_label=None):
    """
    将 LabelMe 格式的 JSON 批量转化为追踪脚本所需的黑白 Mask 图片
    目标 = 255 (纯白), 背景 = 0 (纯黑)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for file_name in os.listdir(json_dir):
        if not file_name.endswith('.json'):
            continue
            
        json_path = os.path.join(json_dir, file_name)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 1. 动态获取标注图的真实分辨率
        img_h = data.get('imageHeight', 480) # 默认保底值，可按需修改
        img_w = data.get('imageWidth', 640)
        
        # 2. 创建一张全黑的空画布
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        
        # 3. 遍历 JSON 中的所有多边形区块
        for shape in data.get('shapes', []):
            label = shape.get('label', '')
            
            # 如果你的图里标注了多个牙齿，可以通过 target_label 过滤（比如 target_label="tooth_1"）
            if target_label and label != target_label:
                continue
                
            # 提取多边形顶点坐标 [[x1, y1], [x2, y2]...] 并转为 int32
            points = np.array(shape['points'], dtype=np.int32)
            
            # 4. 核心：将多边形区域内部填充为 255（纯白）
            cv2.fillPoly(mask, [points], color=255)
            
        # 5. 保存为同名的 .png 图片（供跑批读取）
        save_name = file_name.replace('.json', '.png')
        save_path = os.path.join(output_dir, save_name)
        cv2.imwrite(save_path, mask)
        print(f"✅ 生成 Mask: {save_path}")

if __name__ == "__main__":
    # ==========================================================
    # ⚙️ 配置路径
    # ==========================================================
    # 你的 .json 标注文件存放目录
    INPUT_JSON_DIR = "/root/Toothtrack/demo_data/wxb" 
    
    # run_demo_joint_on.py 需要读取的首帧 mask 目录
    OUTPUT_MASK_DIR = "/root/Toothtrack/demo_data/wxb/mask" 
    
    # 执行转换
    convert_labelme_json_to_mask(INPUT_JSON_DIR, OUTPUT_MASK_DIR, target_label=None)