#!/usr/bin/env bash
# End-to-end pipeline for the new/ subfolder layout under
#   motion_datasets/pingpong/humanoid_data/final/expert/new/{forward,backward}/
#
# Stage 0: cut_from_yaml.py (gmr env)        yaml + raw mp4 → <task>_NNN.mp4 + _clips_info.csv
# Stage 1: per-clip GVHMR + GMR (gmr env)    .mp4 → 23-DoF csv
# Stage 2: batch csv → npz (env_isaaclab_51) all csvs → 23-DoF npzs @ 60 Hz
#
# Output layout (per user choice):
#   .../new/forward/{*.mp4, csv/*.csv, _clips_info.csv, npz/*.npz}
#   .../new/backward/{*.mp4, csv/*.csv, _clips_info.csv, npz/*.npz}
#
# Two conda envs (isolated PYTHONPATH):
#   - gmr           : $HOME/miniforge/envs/gmr/bin/python              (GVHMR + GMR + ffmpeg trim)
#   - env_isaaclab_51 : $HOME/miniforge/envs/env_isaaclab_51/bin/python (Isaac Sim replay → npz)

set -euo pipefail

ULAB=$HOME/HumanoidProject/unitree_rl_lab
GVHMR_DIR=$HOME/HumanoidProject/GMR/GVHMR
GMR_DIR=$HOME/HumanoidProject/GMR/GMR
GMR_PY=$HOME/miniforge/envs/gmr/bin/python
ULAB_PY=$HOME/miniforge/envs/env_isaaclab_51/bin/python
INPUT_FPS=30
OUTPUT_FPS=60
NEW_ROOT=$ULAB/motion_datasets/pingpong/humanoid_data/final/expert/new
CUT_SCRIPT=$ULAB/motion_datasets/pingpong/humanoid_data/cut_from_yaml.py

export PYTHONNOUSERSITE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

echo "=== run_all_new.sh ==="
echo "  NEW_ROOT     : $NEW_ROOT"
echo "  fps  in/out  : $INPUT_FPS / $OUTPUT_FPS"

# ── 0) cut clips per yaml (gmr env, fast) ──────────────
echo
echo "=== Stage 0: cut_from_yaml.py ==="
( unset PYTHONPATH; "$GMR_PY" "$CUT_SCRIPT" forward_new backward_new )

# ── 1) per-clip GVHMR + GMR (gmr env) ──────────────────
# Source mp4 and cut clips share the same directory after Stage 0:
#   source : forward_hand_1.mp4, forward_hand_2.mp4, backward_hand.mp4
#   cuts   : forward_001.mp4, forward_002.mp4, ... , backward_001.mp4, ...
# We glob `<task>_[0-9]*.mp4` to pick cuts only (source has `_hand` infix).
echo
echo "=== Stage 1: per-clip GVHMR + GMR ==="
for task in forward backward; do
    INPUT_DIR=$NEW_ROOT/$task
    CSV_DIR=$INPUT_DIR/csv
    GVHMR_OUT_ROOT=$GVHMR_DIR/outputs/demo/pingpong/${task}_new
    mkdir -p "$CSV_DIR" "$GVHMR_OUT_ROOT"

    shopt -s nullglob
    mp4s=( "$INPUT_DIR/${task}_"[0-9]*.mp4 )
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
set +u
# shellcheck source=/dev/null
source "$HOME/miniforge/etc/profile.d/conda.sh"
conda activate env_isaaclab_51
set -u

for task in forward backward; do
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
        --headless

    echo "[DONE] $task → $OUTPUT_DIR"
done

echo
echo "=== ALL DONE ==="
