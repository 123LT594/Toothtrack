#!/usr/bin/env bash

# 第二阶段：共同支持下 原始DAv2 / 只去整体偏差 / GT参照，3模式x5片段。
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
NO_VIDEO="${NO_VIDEO:-1}"
SNAPSHOT_FRAMES="${SNAPSHOT_FRAMES-219:221,365:366,613:616,760:790}"
TRACK_REFINE_ITER="${TRACK_REFINE_ITER:-1}"
extra_args=()
if [[ "$NO_VIDEO" == "1" ]]; then extra_args+=(--no_video); fi
if [[ -n "$SNAPSHOT_FRAMES" ]]; then extra_args+=(--snapshot_frames "$SNAPSHOT_FRAMES"); fi
if [[ -n "$BACKEND_METRICS_FLAG" ]]; then extra_args+=("$BACKEND_METRICS_FLAG"); fi

# 可通过环境变量覆盖，例如：DATASETS="tooth-1 tooth-3" bash run_3_modes.sh
read -r -a DATASET_LIST <<< "${DATASETS:-tooth-1 tooth-2 tooth-3 tooth-4 tooth-5}"

BATCH_ID="$(date +%m%d_%H%M%S)"
OUTPUT_ROOT="${BATCH_OUTPUT_ROOT:-/root/lanyun-tmp/output/dav2_bias_experiment_${BATCH_ID}}"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "$LOG_DIR"

failures=()
completed=0

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
        --track_refine_iter "$TRACK_REFINE_ITER" \
        --dav2_ckpt "$DAV2_CKPT" \
        --dav2_max_depth "$DAV2_MAX_DEPTH" \
        --dav2_input_size "$DAV2_INPUT_SIZE" \
        "${extra_args[@]}" \
        "$@" 2>&1 | tee "$log_file"; then
        echo "========== 完成 ${dataset} / ${mode} =========="
        completed=$((completed + 1))
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
    missing_dirs=0
    for required_dir in rgb pose depth; do
        if [[ ! -d "${scene_dir}/${required_dir}" ]]; then
            echo "错误：${dataset} 缺少 ${required_dir}/，受控实验需要完整GT" >&2
            missing_dirs=1
        fi
    done
    if [[ "$missing_dirs" == "1" ]]; then
        failures+=("${dataset}(missing_rgb_pose_depth)")
        continue
    fi
    if [[ ! -f "${scene_dir}/annotations.json" ]]; then
        echo "警告：${dataset} 缺少 annotations.json，钢珠误差和钢珠邻域指标将为N/A" >&2
    fi

    failures_before=${#failures[@]}
    # 三组均自动应用共同支持S；oracle也计算DAv2以严格匹配S。
    run_one "$dataset" "01_shared_raw" --depth_experiment raw

    run_one "$dataset" "02_shared_bias_corrected" --depth_experiment bias_corrected

    run_one "$dataset" "03_shared_oracle" --depth_experiment oracle
    if ((${#failures[@]} > failures_before)); then
        echo "${dataset} 有模式运行失败，不读取可能残留的旧报告进行比较" >&2
        continue
    fi

    # 使用标准库读取报告，核对逐帧输入/支持的一致性并生成跨模式汇总。
    if "$PYTHON_BIN" - "${OUTPUT_ROOT}/${dataset}" <<'PY'
import csv, io, math, statistics, sys
from pathlib import Path
root = Path(sys.argv[1])
names = ['01_shared_raw', '02_shared_bias_corrected', '03_shared_oracle']
expected = ['raw', 'bias_corrected', 'oracle']
runs = []
for name, mode in zip(names, expected):
    text = (root / name / 'frame_errors.txt').read_text(encoding='utf-8-sig')
    if '# run_completed=True' not in text:
        raise ValueError(name + ': incomplete run')
    lines = [line for line in text.splitlines() if line and not line.startswith('#')]
    rows = list(csv.DictReader(io.StringIO('\n'.join(lines)), delimiter='\t'))
    if len(rows) < 2 or any(r['depth_experiment'] != mode for r in rows):
        raise ValueError(name + ': empty/incorrect experiment')
    if rows[0]['is_init'] != '1' or rows[0]['pose_updated'] != '0':
        raise ValueError(name + ': first frame must initialize only')
    for row in rows:
        if row['has_gt_pose'] != '1' or row['has_gt_depth'] != '1' or row['notes'].strip():
            raise ValueError(name + ': invalid labels/evaluation at ' + row['frame_id'])
        if row['is_init'] == '0' and row['pose_updated'] != '1':
            raise ValueError(name + ': missing tracking update at ' + row['frame_id'])
    runs.append(rows)
if not all([r['frame_id'] for r in rows] == [r['frame_id'] for r in runs[0]] for rows in runs):
    raise ValueError('Frame order differs across modes')
for triplet in zip(*runs):
    for key in ['support_sha256', 'prediction_sha256', 'gt_depth_sha256']:
        values = [r[key] for r in triplet]
        if any(len(v) != 64 for v in values) or len(set(values)) != 1:
            raise ValueError(triplet[0]['frame_id'] + ': ' + key + ' mismatch; comparison invalid')
    corrected_bias = float(triplet[1]['SupportInputBias_mm'])
    oracle_mae = float(triplet[2]['SupportInputMAE_mm'])
    if not math.isfinite(corrected_bias) or abs(corrected_bias) > 0.0001:
        raise ValueError(triplet[0]['frame_id'] + ': corrected mean error not zero')
    if not math.isfinite(oracle_mae) or abs(oracle_mae) > 0.0001:
        raise ValueError(triplet[0]['frame_id'] + ': oracle error not zero')
def quantile(values, q):
    a = sorted(values); p = (len(a) - 1) * q; lo = int(p); hi = min(lo + 1, len(a) - 1)
    return a[lo] + (a[hi] - a[lo]) * (p - lo)
lines = ['受控实验比较：帧号、原始DAv2、GT深度和共同支持逐帧哈希一致。',
         '统计排除首帧；GT辅助诊断，非可部署校正。',
         'mode\tN\tADD_mean_mm\tADD_P95_mm\tRE_mean_deg\tErr2D_ID_mean_px\tSupportMAE_mm\tADD_gt1mm_pct\tlongest_ADD_gt1mm_run']
means = []
for name, rows in zip(names, runs):
    ev = [r for r in rows if r['is_init'] == '0']
    def values(key):
        v = [float(r[key]) for r in ev]
        if not all(math.isfinite(x) for x in v):
            raise ValueError(name + ': nonfinite ' + key)
        return v
    add = values('ADD_mm'); means.append(statistics.mean(add))
    streak = longest = 0
    for v in add:
        streak = streak + 1 if v > 1 else 0; longest = max(longest, streak)
    # 1mm is an explicitly descriptive threshold, not a clinical failure criterion.
    err = [float(r['Err2D_ID_px']) for r in ev if r['Err2D_ID_px'] not in ('N/A', '')]
    row = [name, str(len(ev)), f'{means[-1]:.6f}', f'{quantile(add,.95):.6f}',
           f'{statistics.mean(values("RE_deg")):.6f}', f'{statistics.mean(err):.6f}' if err else 'N/A',
           f'{statistics.mean(values("SupportInputMAE_mm")):.6f}',
           f'{100*sum(v>1 for v in add)/len(add):.3f}', str(longest)]
    lines.append('\t'.join(row))
lines.append('ADD>1mm仅为描述性分组；连续长度按处理帧计，保留原始帧号间隔。')
lines.append(f'原始到校正的ADD均值变化(改善为正): {means[0]-means[1]:.6f} mm')
if means[0] > means[2]:
    lines.append(f'相对GT参照缩小的ADD差距: {100*(means[0]-means[1])/(means[0]-means[2]):.2f}%')
(root / 'mode_comparison.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('\n'.join(lines))
PY
    then
        echo "${dataset} 三模式一致性核对完成"
    else
        failures+=("${dataset}(comparison_failed)")
    fi
done

echo
echo "全部结果：$OUTPUT_ROOT"
if ((${#failures[@]} > 0)); then
    echo "失败或缺失项：${failures[*]}"
    exit 1
fi

echo "${completed}项实验均已完成，三模式逐帧一致性核对通过。"
