import os,sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple,Union
import numpy as np
import omegaconf
import torch

#⭐新增==========================================================
import trimesh
# 真理内参 (从 1100+ 黄金帧反推出的最强物理基准)
DISTILL_K_BASE = [
    [2866.3146, 0.0, 480.0],
    [0.0, 2866.3146, 270.0],
    [0.0, 0.0, 1.0]
]
# 获取当前脚本所在目录，向上回退两级到达根目录，再指向 demo_data
_current_dir = os.path.dirname(os.path.abspath(__file__))
_mesh_path = os.path.abspath(os.path.join(_current_dir, "../../demo_data/tooth_gt/mesh/tooth.obj"))

# 💥 零容忍校验：找不到 CAD 模型直接引发系统崩溃！
if not os.path.exists(_mesh_path):
    raise FileNotFoundError(
        f"\n{'='*60}\n"
        f"【致命错误】找不到核心 CAD 模型！\n"
        f"路径: {_mesh_path}\n"
        f"此模型是物理锚点和虚拟绝对深度的唯一标尺，缺少此文件训练无从谈起，请立即检查数据目录！\n"
        f"{'='*60}"
    )

# 动态提取绝对物理尺度
_mesh = trimesh.load(_mesh_path, process=False)
_extents = _mesh.extents

# 物理锚点参数 (保持原生数据类型体系自洽)
DISTILL_PHYSICAL_WIDTH = float(max(_extents[0], _extents[1]))  # X或Y的最大跨度
DISTILL_PHYSICAL_THICKNESS = float(_extents[2])                # Z轴最大起伏厚度
# ==========================================================
@dataclass
class TrainingConfig(omegaconf.dictconfig.DictConfig):
    input_resize: tuple = (160, 160)
    normalize_xyz:Optional[bool] = True
    use_mask:Optional[bool] = False
    crop_ratio:Optional[float] = None
    split_objects_across_gpus: bool = True
    max_num_key: Optional[int] = None
    use_normal:bool = False
    n_view:int = 1
    zfar:float = np.inf
    c_in:int = 6
    train_num_pair:Optional[int] = None
    make_pair_online:Optional[bool] = False
    render_backend:Optional[str] = 'nvdiffrast'

    # Run management
    run_id: Optional[str] = None
    exp_name:Optional[str] = None
    resume_run_id: Optional[str] = None
    save_dir: Optional[str] = None
    batch_size: int = 64
    epoch_size: int = 115200
    val_size: int = 1280
    n_epochs: int = 25
    save_epoch_interval: int = 100
    n_dataloader_workers: int = 20
    n_rendering_workers: int = 1
    gradient_max_norm:float = np.inf
    max_step_per_epoch: Optional[int] = 25000

    # Network
    use_BN:bool = True
    loss_type:Optional[str] = 'pairwise_valid'

    # Optimizer
    optimizer: str = "adam"
    weight_decay: float = 0.0
    clip_grad_norm: float = np.inf
    lr: float = 0.0001
    warmup_step: int = -1   # -1 means disable
    n_epochs_warmup: int = 1

    # Visualization
    vis_interval: Optional[int] = 1000

    debug: Optional[bool] = None



@dataclass
class TrainRefinerConfig:
    # Datasets
    input_resize: tuple = (160, 160)  #(W,H)
    crop_ratio:Optional[float] = None
    max_num_key: Optional[int] = None
    use_normal:bool = False
    use_mask:Optional[bool] = False
    normal_uint8:bool = False
    normalize_xyz:Optional[bool] = True
    trans_normalizer:Optional[list] = None
    rot_normalizer:Optional[float] = None
    c_in:int = 6
    n_view:int = 1
    zfar:float = np.inf
    trans_rep:str = 'tracknet'  # tracknet/deepim
    rot_rep:Optional[str] = 'axis_angle'  # 6d/axis_angle
    save_dir: Optional[str] = None

    # Run management
    run_id: Optional[str] = None
    exp_name:Optional[str] = None
    batch_size: int = 64
    use_BN:bool = True
    optimizer: str = "adam"
    weight_decay: float = 0.0
    clip_grad_norm: float = np.inf
    lr: float = 0.0001
    warmup_step: int = -1
    loss_type:str = 'l2'   # l1/l2/add

    vis_interval: Optional[int] = 1000
    debug: Optional[bool] = None


