#!/usr/bin/env bash
# chmod +x run_3_modes.sh 
# ./run_3_modes.sh

# 文件名沿用旧入口；当前DAv2诊断实验实际运行3个模式 x 5个数据集，共15次。
set -uo pipefail

DATA_ROOT="${1:-/root/Toothtrack/demo_data}"
MESH_FILE="${2:-/root/Toothtrack/demo_data/tooth/mesh/tooth.obj}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${RUN_SCRIPT:-${SCRIPT_DIR}/run_demo_dav2.py}"

DAV2_CKPT="${DAV2_CKPT:-/root/Toothtrack/checkpoints/depth_anything_v2_metric_golden_vits.pth}"
DAV2_MAX_DEPTH="${DAV2_MAX_DEPTH:-0.2}"
DAV2_INPUT_SIZE="${DAV2_INPUT_SIZE:-518}"
HIGHLIGHT_THRESHOLD="${HIGHLIGHT_THRESHOLD:-240}"
BALL_RADIUS_PX="${BALL_RADIUS_PX:-12}"
ERROR_VIS_LIMIT_MM="${ERROR_VIS_LIMIT_MM:-5}"
BACKEND_METRICS_FLAG="${BACKEND_METRICS_FLAG:-}"

# 可通过环境变量覆盖，例如：DATASETS="tooth-1 tooth-3" ./run_four_modes.sh
read -r -a DATASET_LIST <<< "${DATASETS:-tooth-1 tooth-2 tooth-3 tooth-4 tooth-5}"

BATCH_ID="$(date +%m%d_%H%M%S)"
OUTPUT_ROOT="${BATCH_OUTPUT_ROOT:-/root/lanyun-tmp/output1/dav2_diagnosis_${BATCH_ID}}"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "$LOG_DIR"

failures=()

run_one() {
    local dataset="$1"
    local mode="$2"
    shift 2
    local scene_dir="${DATA_ROOT}/${dataset}"
    local result_dir="${OUTPUT_ROOT}/${dataset}/${mode}"
    local log_file="${LOG_DIR}/${dataset}_${mode}.log"

    echo
    echo "========== 开始 ${dataset} / ${mode} =========="
    if "$PYTHON_BIN" -u "$RUN_SCRIPT" \
        --test_scene_dir "$scene_dir" \
        --mesh_file "$MESH_FILE" \
        --output_dir "$result_dir" \
        --highlight_threshold "$HIGHLIGHT_THRESHOLD" \
        --ball_radius_px "$BALL_RADIUS_PX" \
        --error_vis_limit_mm "$ERROR_VIS_LIMIT_MM" \
        ${BACKEND_METRICS_FLAG:+"$BACKEND_METRICS_FLAG"} \
        "$@" 2>&1 | tee "$log_file"; then
        echo "========== 完成 ${dataset} / ${mode} =========="
    else
        local status=$?
        failures+=("${dataset}/${mode}(exit=${status})")
        echo "========== 失败 ${dataset} / ${mode}，继续下一项 =========="
    fi
}

if [[ ! -f "$RUN_SCRIPT" ]]; then
    echo "错误：找不到运行脚本 $RUN_SCRIPT" >&2
    exit 2
fi
if [[ ! -f "$MESH_FILE" ]]; then
    echo "错误：找不到mesh $MESH_FILE" >&2
    exit 2
fi
if [[ ! -f "$DAV2_CKPT" ]]; then
    echo "错误：找不到DAv2权重 $DAV2_CKPT" >&2
    exit 2
fi

for dataset in "${DATASET_LIST[@]}"; do
    scene_dir="${DATA_ROOT}/${dataset}"
    if [[ ! -d "$scene_dir" ]]; then
        failures+=("${dataset}(missing_directory)")
        echo "警告：跳过不存在的数据集目录 $scene_dir" >&2
        continue
    fi
    for required_dir in rgb pose depth; do
        if [[ ! -d "${scene_dir}/${required_dir}" ]]; then
            echo "警告：${dataset} 缺少 ${required_dir}/，对应流程可能失败或指标为N/A" >&2
        fi
    done
    if [[ ! -f "${scene_dir}/annotations.json" ]]; then
        echo "警告：${dataset} 缺少 annotations.json，钢珠误差和钢珠邻域指标将为N/A" >&2
    fi

    # ① 当前系统基线：全图DAv2深度，不做门控。
    run_one "$dataset" "01_dav2_nomask" \
        --depth_src dav2 \
        --dav2_ckpt "$DAV2_CKPT" \
        --dav2_max_depth "$DAV2_MAX_DEPTH" \
        --dav2_input_size "$DAV2_INPUT_SIZE"

    # ② 理想门控诊断：同一张DAv2预测，只在送入FoundationPose前乘GT mask。
    run_one "$dataset" "02_dav2_gtmask" \
        --depth_src dav2 \
        --use_gt_mask \
        --dav2_ckpt "$DAV2_CKPT" \
        --dav2_max_depth "$DAV2_MAX_DEPTH" \
        --dav2_input_size "$DAV2_INPUT_SIZE"

    # ③ Oracle参照：GT深度 + GT mask。
    run_one "$dataset" "03_gtdepth_gtmask" \
        --depth_src gt \
        --use_gt_mask
done

echo
echo "全部结果：$OUTPUT_ROOT"
if ((${#failures[@]} > 0)); then
    echo "失败或缺失项：${failures[*]}"
    exit 1
fi

echo "15项实验均已完成。"
