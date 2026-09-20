#!/usr/bin/env bash

set -uo pipefail

SCENE_DIR="${1:-/root/Toothtrack/demo_data/tooth}"
MESH_FILE="${2:-/root/Toothtrack/demo_data/tooth/mesh/tooth.obj}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_SCRIPT="${RUN_SCRIPT:-run_demo_joint_on.py}"

DAV2_CKPT="${DAV2_CKPT:-/root/Toothtrack/checkpoints/depth_anything_v2_metric_golden_vits.pth}"
STUDENT_CKPT="${STUDENT_CKPT:-/root/lanyun-tmp/models/stage1_distill/models/student_stage1_ema_ep99.pth}"
DAV2_MAX_DEPTH="${DAV2_MAX_DEPTH:-0.2}"

BATCH_ID="$(date +%m%d_%H%M%S)"
LOG_DIR="${BATCH_LOG_DIR:-/root/lanyun-tmp/output/batch_${BATCH_ID}}"
mkdir -p "$LOG_DIR"

failures=()

run_one() {
    local name="$1"
    shift
    echo
    echo "========== 开始 ${name} =========="
    if "$PYTHON_BIN" -u "$RUN_SCRIPT" \
        --test_scene_dir "$SCENE_DIR" \
        --mesh_file "$MESH_FILE" \
        "$@" 2>&1 | tee "$LOG_DIR/${name}.log"; then
        echo "========== 完成 ${name} =========="
    else
        local status=$?
        failures+=("${name}(exit=${status})")
        echo "========== 失败 ${name}，继续下一组 =========="
    fi
}

# run_one "01_dav2_nomask" \
#     --depth_src dav2 \
#     --dav2_ckpt "$DAV2_CKPT" \
#     --dav2_max_depth "$DAV2_MAX_DEPTH"

run_one "02_dav2_gtmask" \
    --depth_src dav2 \
    --use_gt_mask \
    --dav2_ckpt "$DAV2_CKPT" \
    --dav2_max_depth "$DAV2_MAX_DEPTH"

run_one "03_gtdepth_gtmask" \
    --depth_src gt \
    --use_gt_mask

run_one "04_distill_gtmask" \
    --depth_src distill \
    --use_gt_mask \
    --weight_student "$STUDENT_CKPT"

echo
echo "批量日志目录：$LOG_DIR"
if ((${#failures[@]} > 0)); then
    echo "失败实验：${failures[*]}"
    exit 1
fi

echo "四组实验均已完成。"
