import numpy as np
import cv2
import pickle

class MedicalGeometryFeatureExtractor:
    """
    线上线下绝对对齐的几何特征提取器
    输出: 傅里叶轮廓描述子 (用于粗筛) + 16x16网格法向 (用于精筛)
    """
    def __init__(self, target_size=160, grid_size=16):
        self.target_size = target_size
        self.grid_size = grid_size
        self.pool_kernel = target_size // grid_size  # 例如 160/16 = 10

    def extract_fourier_descriptor(self, mask, num_harmonics=15):
        # 寻找最大轮廓
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return np.zeros(num_harmonics, dtype=np.float32)
        
        contour = max(contours, key=cv2.contourArea).squeeze()
        if len(contour.shape) < 2 or len(contour) < num_harmonics:
            return np.zeros(num_harmonics, dtype=np.float32)

        # 转换为复数坐标 x + iy 用于傅里叶变换
        contour_complex = np.empty(contour.shape[0], dtype=complex)
        contour_complex.real = contour[:, 0]
        contour_complex.imag = contour[:, 1]

        fourier_result = np.fft.fft(contour_complex)
        # 取低频分量，跳过第0项(平移)，截取指定的谐波数量
        descriptors = np.abs(fourier_result[1:num_harmonics+1])
        # 尺度归一化
        descriptors = descriptors / (descriptors[0] + 1e-6)
        return descriptors.astype(np.float32)

    def extract_grid_normal_feature(self, depth, mask_bin):
        # 1. 深度图轻微高斯平滑抗噪，后计算 Sobel 梯度
        depth_smoothed = cv2.GaussianBlur(depth.astype(np.float32), (3, 3), 0)
        dzdx = cv2.Sobel(depth_smoothed, cv2.CV_32F, 1, 0, ksize=3)
        dzdy = cv2.Sobel(depth_smoothed, cv2.CV_32F, 0, 1, ksize=3)
        
        # 构造表面法向 (假设相机朝向Z正方向)
        normal = np.dstack((-dzdx, -dzdy, np.ones_like(depth_smoothed)))
        norm = np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-6
        normal = normal / norm

        # 2. 16x16 强效低通网格池化 (Grid Pooling)
        k = self.pool_kernel
        # Reshape: [16, 10, 16, 10, 3]
        normal_grid = normal.reshape(self.grid_size, k, self.grid_size, k, 3)
        mask_grid = mask_bin.reshape(self.grid_size, k, self.grid_size, k)

        # 计算每个网格的前景覆盖率
        grid_mask_ratio = mask_grid.mean(axis=(1, 3))
        
        # 仅对网格内真实属于前景的法向求平均
        grid_normal_sum = (normal_grid * mask_grid[..., None]).sum(axis=(1, 3))
        grid_normal_count = mask_grid.sum(axis=(1, 3))[..., None] + 1e-6
        grid_normal_mean = grid_normal_sum / grid_normal_count
        
        # 再次归一化
        grid_norm = np.linalg.norm(grid_normal_mean, axis=-1, keepdims=True) + 1e-6
        grid_normal_mean = grid_normal_mean / grid_norm

        return grid_normal_mean.astype(np.float32), grid_mask_ratio.astype(np.float32)


class FastZeroShotMatcher:
    """
    在线端纯数学检索引擎
    流程: 傅里叶粗筛 -> MNN精筛 -> 手性防翻转 -> WAE'打分
    """
    def __init__(self, pkl_path, alpha=0.5):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            self.db = data['zero_shot_db'] # 读取新增的特征库
        
        self.alpha = alpha
        self.extractor = MedicalGeometryFeatureExtractor()

    def check_chirality(self, pt_o, pt_t):
        """标量三重积(手性)校验，直接剔除 180°/90° 镜像伪影"""
        if len(pt_o) < 3:
            return True
        # 取首、中、尾三个匹配点构成拓扑三角形
        v1_o, v2_o, v3_o = pt_o[0], pt_o[len(pt_o)//2], pt_o[-1]
        v1_t, v2_t, v3_t = pt_t[0], pt_t[len(pt_t)//2], pt_t[-1]
        # 计算 2D 叉乘符号 (顺时针/逆时针)
        cross_o = np.cross(v2_o - v1_o, v3_o - v1_o)
        cross_t = np.cross(v2_t - v1_t, v3_t - v1_t)
        # 符号一致则未发生翻转
        return (cross_o * cross_t) >= 0

    def match(self, pred_mask_160, pred_depth_160):
        # --- 1. 提取在线特征 ---
        mask_255 = (pred_mask_160 * 255).astype(np.uint8)
        fourier_online = self.extractor.extract_fourier_descriptor(mask_255)
        normal_online, mask_ratio_online = self.extractor.extract_grid_normal_feature(pred_depth_160, pred_mask_160)
        
        valid_online = mask_ratio_online > 0.5
        pts_online = np.argwhere(valid_online)
        feats_online = normal_online[valid_online]

        if len(pts_online) == 0:
            return None

        # --- 2. 傅里叶轮廓粗筛 (漏斗过滤) ---
        fourier_dists = []
        for i, template in enumerate(self.db):
            dist = np.linalg.norm(fourier_online - template['fourier'])
            fourier_dists.append((dist, i))
        fourier_dists.sort(key=lambda x: x[0])
        top_candidates = fourier_dists[:50]  # 只保留形貌最接近的50个模板
        # # 临床 Mask 边缘不稳定，2D轮廓不可靠，直接对库中所有模板进行 3D 硬匹配！
        # top_candidates = [(0, i) for i in range(len(self.db))]

        best_score = float('inf')
        best_pose = None

        # --- 3. 法向 MNN 精筛 & 混合 WAE' 打分 ---
        for _, idx in top_candidates:
            template = self.db[idx]
            
            valid_t = template['grid_mask'] > 0.5
            pts_t = np.argwhere(valid_t)
            feats_t = template['grid_normal'][valid_t]

            if len(feats_t) == 0: continue

            # 余弦相似度矩阵点乘
            sim_matrix = np.dot(feats_online, feats_t.T)

            # 寻找互为最近邻 (Mutual Nearest Neighbor)
            max_online = np.argmax(sim_matrix, axis=1)
            max_template = np.argmax(sim_matrix, axis=0)
            
            cosine_errors = []
            matched_pts_o, matched_pts_t = [], []

            for i_o, i_t in enumerate(max_online):
                if max_template[i_t] == i_o:  # 互为双向奔赴
                    cosine_errors.append(1.0 - sim_matrix[i_o, i_t])
                    matched_pts_o.append(pts_online[i_o])
                    matched_pts_t.append(pts_t[i_t])

            if len(cosine_errors) < 5:
                continue

            # 手性翻转校验
            if not self.check_chirality(np.array(matched_pts_o), np.array(matched_pts_t)):
                continue

            # 混合 WAE' 打分计算
            mean_normal_error = np.mean(cosine_errors)
            
            # 直接计算 2D 投影 IoU (因为已裁剪对齐)
            intersection = np.logical_and(pred_mask_160, template['mask_160']).sum()
            union = np.logical_or(pred_mask_160, template['mask_160']).sum() + 1e-6
            iou = intersection / union

            # 空间支持度 (匹配上网格的覆盖率)
            coverage = len(cosine_errors) / (len(pts_online) + 1e-6)

            # 最终得分：误差与不重合度越小越好，覆盖率越大越好
            wae_score = (self.alpha * mean_normal_error + (1 - self.alpha) * (1 - iou)) / coverage

            if wae_score < best_score:
                best_score = wae_score
                best_pose = template['pose']

        return best_pose