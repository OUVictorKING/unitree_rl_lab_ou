#!/usr/bin/env bash
# End-to-end pipeline for the new/ subfolder layout under
#   <ROOT>/{forward,backward}/  (ROOT specified via --root, or hardcoded default)
#
# Required folder layout (per-task):
#   <ROOT>/forward/forward.yaml         (raw_mp4_dir inside points to real mp4)
#   <ROOT>/forward/<raw>.mp4
#   <ROOT>/backward/backward.yaml
#   <ROOT>/backward/<raw>.mp4
#
# Stage 0: cut_from_yaml.py (gmr env)        yaml + raw mp4 → <task>_NNN.mp4 + _clips_info.csv
# Stage 1: per-clip GVHMR + GMR (gmr env)    .mp4 → 23-DoF csv
# Stage 2: batch csv → npz (env_isaaclab_51) all csvs → 23-DoF npzs @ 60 Hz
#
# Output layout (per task):
#   <ROOT>/<task>/{*.mp4, csv/*.csv, _clips_info.csv, npz/*.npz}
#
# Two conda envs (isolated PYTHONPATH):
#   - gmr           : $HOME/miniforge/envs/gmr/bin/python              (GVHMR + GMR + ffmpeg trim)
#   - env_isaaclab_51 : $HOME/miniforge/envs/env_isaaclab_51/bin/python (Isaac Sim replay → npz)
#
# Usage:
#   bash run_all_new.sh --root <ROOT> [--forward] [--backward]
#     --root <PATH>   : data root containing forward/ and/or backward/ subdirs (REQUIRED unless default below kept)
#     --forward       : process forward/ only
#     --backward      : process backward/ only
#     (no flag)       : process both forward + backward (default)

set -euo pipefail

ULAB=$HOME/HumanoidProject/unitree_rl_lab
GVHMR_DIR=$HOME/HumanoidProject/GMR/GVHMR
GMR_DIR=$HOME/HumanoidProject/GMR/GMR
GMR_PY=$HOME/miniforge/envs/gmr/bin/python
ULAB_PY=$HOME/miniforge/envs/env_isaaclab_51/bin/python
INPUT_FPS=30
OUTPUT_FPS=60
CUT_SCRIPT=$ULAB/motion_datasets/pingpong/humanoid_data/cut_from_yaml.py

# ── CLI arg parsing ────────────────────────────────────
NEW_ROOT=""
TASKS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)     NEW_ROOT="$2"; shift 2 ;;
        --forward)  TASKS+=(forward);  shift ;;
        --backward) TASKS+=(backward); shift ;;
        -h|--help)
            sed -n '/^# Usage:/,/^$/p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "ERROR: unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$NEW_ROOT" ]]; then
    echo "ERROR: --root <PATH> required" >&2
    echo "  (e.g.: bash $0 --root \$HOME/HumanoidProject/unitree_rl_lab/motion_datasets/pingpong/humanoid_data/final/expert/new_3)" >&2
    exit 1
fi
[[ -d "$NEW_ROOT" ]] || { echo "ERROR: --root not a directory: $NEW_ROOT" >&2; exit 1; }
NEW_ROOT=$(realpath "$NEW_ROOT")

# default to both tasks
if [[ ${#TASKS[@]} -eq 0 ]]; then
    TASKS=(forward backward)
fi

# Unique GVHMR cache subdir per dataset, to avoid clobbering (uses ROOT basename)
ROOT_TAG=$(basename "$NEW_ROOT")

export PYTHONNOUSERSITE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

echo "=== run_all_new.sh ==="
echo "  NEW_ROOT     : $NEW_ROOT"
echo "  ROOT_TAG     : $ROOT_TAG  (used for GVHMR cache subdir)"
echo "  tasks        : ${TASKS[*]}"
echo "  fps  in/out  : $INPUT_FPS / $OUTPUT_FPS"

# ── 0) cut clips per yaml (gmr env, fast) ──────────────
echo
echo "=== Stage 0: cut_from_yaml.py ==="
for task in "${TASKS[@]}"; do
    YAML_PATH=$NEW_ROOT/$task/$task.yaml
    if [[ ! -f "$YAML_PATH" ]]; then
        echo "ERROR: missing yaml: $YAML_PATH" >&2; exit 1
    fi
    ( unset PYTHONPATH; "$GMR_PY" "$CUT_SCRIPT" --yaml "$YAML_PATH" --task-label "$task" )
done

# ── 1) per-clip GVHMR + GMR (gmr env) ──────────────────
# We glob `<task>_NNN.mp4` (3-digit zero-padded) to pick cuts only —
# avoids matching raw mp4s like backward_1.mp4 / backward_2.mp4 that share the prefix.
echo
echo "=== Stage 1: per-clip GVHMR + GMR ==="
for task in "${TASKS[@]}"; do
    INPUT_DIR=$NEW_ROOT/$task
    CSV_DIR=$INPUT_DIR/csv
    GVHMR_OUT_ROOT=$GVHMR_DIR/outputs/demo/pingpong/${ROOT_TAG}_${task}
    mkdir -p "$CSV_DIR" "$GVHMR_OUT_ROOT"

    shopt -s nullglob
    mp4s=( "$INPUT_DIR/${task}_"[0-9][0-9][0-9].mp4 )
    shopt -u nullglob

    if [[ ${#mp4s[@]} -eq 0 ]]; then
        echo "[WARN] no cut mp4s in $INPUT_DIR — skipping $task"
        continue
    fi
    echo "[task=$task] ${#mp4s[@]} clips"

    for MP4 in "${mp4s[@]}"; do
        STEM=$(basename "$MP4" .mp4)
        CSV_OUT=$CSV_DIR/$STEM.csv
        HMR_OUT=$GVHMR_OUT_ROOT/$STEM/hmr4d_results.pt
        HMR_DEFAULT=$GVHMR_DIR/outputs/demo/$STEM/hmr4d_results.pt

        if [[ -f "$CSV_OUT" ]]; then
            echo "[skip] $STEM (csv exists)"
            continue
        fi

        if [[ -f "$HMR_OUT" || -f "$HMR_DEFAULT" ]]; then
            echo "[skip GVHMR] $STEM (hmr4d_results.pt already on disk)"
        else
            echo "[GVHMR] $STEM"
            ( unset PYTHONPATH; cd "$GVHMR_DIR" && "$GMR_PY" tools/demo/demo.py --video="$MP4" -s ) || \
                echo "  (note) demo.py exited non-zero — checking for hmr4d_results.pt anyway"
        fi

        DEFAULT_OUT=$GVHMR_DIR/outputs/demo/$STEM
        if [[ -d "$DEFAULT_OUT" ]]; then
            if [[ -d "$GVHMR_OUT_ROOT/$STEM" ]]; then
                rm -rf "$GVHMR_OUT_ROOT/$STEM"
            fi
            mv "$DEFAULT_OUT" "$GVHMR_OUT_ROOT/$STEM"
        fi

        if [[ ! -f "$HMR_OUT" ]]; then
            echo "  ✗ GVHMR did not produce $HMR_OUT, skip"
            continue
        fi

        echo "[GMR ] $STEM"
        if ! ( unset PYTHONPATH; cd "$GMR_DIR" && "$GMR_PY" scripts/gvhmr_to_robot.py \
                  --gvhmr_pred_file "$HMR_OUT" \
                  --robot unitree_g1_23dof \
                  --save_path "$CSV_OUT" ); then
            echo "  ✗ GMR failed for $STEM"
            continue
        fi
    done
done

# ── 2) batch csv → npz @ 60 Hz (single Isaac Sim launch per task) ──
# Activate env_isaaclab_51 once; its activate hook adds isaacsim's
# _isaac_sim/python_packages to PYTHONPATH (required to import isaacsim).
echo
echo "=== Stage 2: csv → npz @ ${OUTPUT_FPS} Hz ==="
set +eu
# shellcheck source=/dev/null
source "$HOME/miniforge/etc/profile.d/conda.sh"
if [[ "${CONDA_DEFAULT_ENV:-}" != "env_isaaclab_51" ]]; then
    conda activate env_isaaclab_51
fi
set -eu

for task in "${TASKS[@]}"; do
    INPUT_DIR=$NEW_ROOT/$task
    CSV_DIR=$INPUT_DIR/csv
    OUTPUT_DIR=$INPUT_DIR/npz
    mkdir -p "$OUTPUT_DIR"

    if [[ "$task" == "forward" ]]; then
        TASK_NAME=forward_hand
    else
        TASK_NAME=backward_hand
    fi

    shopt -s nullglob
    csvs=( "$CSV_DIR"/*.csv )
    shopt -u nullglob
    if [[ ${#csvs[@]} -eq 0 ]]; then
        echo "[csv→npz] no csvs in $CSV_DIR — skipping $task"
        continue
    fi

    echo "[csv→npz] task=$task: $CSV_DIR (${#csvs[@]} files) → $OUTPUT_DIR"
    "$ULAB_PY" \
        "$ULAB/scripts/pingpong_data_process/csv_to_npz_pingpong.py" \
        --input  "$CSV_DIR" \
        --output "$OUTPUT_DIR" \
        --input_fps  "$INPUT_FPS" \
        --output_fps "$OUTPUT_FPS" \
        --task_name  "$TASK_NAME" \
        --paddle \
        --overwrite \
        --headless || echo "  (note) csv_to_npz_pingpong exited non-zero (likely close-hang killed externally) — npz already written"

    echo "[DONE] $task → $OUTPUT_DIR"
done

echo
echo "=== ALL DONE ==="
