#!/usr/bin/env bash
# End-to-end ping-pong mp4 → 23-DoF npz pipeline.
#
# Per clip:
#   1) GVHMR demo.py    (gmr conda env)   .mp4 → hmr4d_results.pt
#   2) gvhmr_to_robot   (gmr conda env)   .pt  → 29-DoF csv (with header row)
# Batch:
#   3) csv_to_npz_pingpong.py (env_isaaclab_51) all csvs → 23-DoF npzs at output_fps
#
# User edits TASK_NAME / INPUT_DIR / OUTPUT_DIR for each pass (forward / backward).
set -euo pipefail

# ============== USER EDITS THESE 3 VARS ==============
TASK_NAME=${TASK_NAME:-forward_hand}   # forward_hand, backward_hand (override via env var)
INPUT_DIR=$HOME/HumanoidProject/unitree_rl_lab/motion_datasets/pingpong/humanoid_data/final/${TASK_NAME}
OUTPUT_DIR=$HOME/HumanoidProject/unitree_rl_lab/motion_datasets/pingpong/humanoid_data/final/${TASK_NAME}_npz
# =====================================================

# ============== fixed config ==============
GVHMR_DIR=$HOME/HumanoidProject/GMR/GVHMR
GMR_DIR=$HOME/HumanoidProject/GMR/GMR
ULAB=$HOME/HumanoidProject/unitree_rl_lab
ISAACLAB_DIR=$HOME/HumanoidProject/IsaacLab
GMR_PY=$HOME/miniforge/envs/gmr/bin/python                # gmr conda env
ULAB_PY=$HOME/miniforge/envs/env_isaaclab_51/bin/python   # isaaclab env
INPUT_FPS=30
OUTPUT_FPS=50

GVHMR_OUT_ROOT=$GVHMR_DIR/outputs/demo/pingpong/${TASK_NAME}
CSV_DIR=$INPUT_DIR/csv

mkdir -p "$CSV_DIR" "$OUTPUT_DIR" "$GVHMR_OUT_ROOT"

# Each conda env needs its own PYTHONPATH:
#   - gmr env: must NOT see IsaacLab's _isaac_sim PYTHONPATH (its bundled torch
#     breaks gmr's torch). Stage 1 runs gmr commands inside subshells that
#     `unset PYTHONPATH` locally so the env var stays clean upstream.
#   - env_isaaclab_51: REQUIRES PYTHONPATH set by `conda activate` to find
#     isaacsim (its `etc/conda/activate.d/setenv.sh` adds _isaac_sim paths).
#     Stage 2 sources conda and activates that env explicitly.
#
# ~/.local/lib/python3.10/site-packages is auto-prepended to sys.path by
# default and can override conda-env packages (e.g. an old user-site numpy
# breaks chumpy in GVHMR's render_incam, and rich/transforms3d-via-user-site
# vanish if PYTHONNOUSERSITE=1). Disable user site-packages so each conda env
# stays self-contained and we can install missing deps directly into gmr.
export PYTHONNOUSERSITE=1

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

echo "=== task = $TASK_NAME ==="
echo "  input  mp4 dir : $INPUT_DIR"
echo "  csv    dir     : $CSV_DIR"
echo "  npz    dir     : $OUTPUT_DIR"
echo "  GVHMR  dir     : $GVHMR_OUT_ROOT"
echo "  fps  in/out    : $INPUT_FPS / $OUTPUT_FPS"

# ── 1) per-clip GVHMR + GMR (gmr env) ─────────────────
shopt -s nullglob
mp4s=( "$INPUT_DIR"/*.mp4 )
if [[ ${#mp4s[@]} -eq 0 ]]; then
  echo "[WARN] no *.mp4 under $INPUT_DIR — skipping stage 1."
fi

for MP4 in "${mp4s[@]}"; do
  STEM=$(basename "$MP4" .mp4)
  CSV_OUT=$CSV_DIR/$STEM.csv
  HMR_OUT=$GVHMR_OUT_ROOT/$STEM/hmr4d_results.pt
  HMR_DEFAULT=$GVHMR_DIR/outputs/demo/$STEM/hmr4d_results.pt

  if [[ -f "$CSV_OUT" ]]; then
    echo "[skip] $STEM (csv exists at $CSV_OUT)"
    continue
  fi

  # Skip GVHMR if the .pt is already on disk (prior partial run, render failed
  # but prediction succeeded). We'll move it / run GMR on it below.
  if [[ -f "$HMR_OUT" || -f "$HMR_DEFAULT" ]]; then
    echo "[skip GVHMR] $STEM (hmr4d_results.pt already on disk)"
  else
    echo "[GVHMR] $STEM"
    # demo.py runs preprocess → predict (saves hmr4d_results.pt) → render_incam
    # → render_global. The render steps are visualization only; an old user-site
    # numpy can break chumpy in render_incam even after the .pt is on disk. So
    # we don't `continue` on demo.py's exit code — we only require the .pt to
    # exist below.
    ( unset PYTHONPATH; cd "$GVHMR_DIR" && "$GMR_PY" tools/demo/demo.py --video="$MP4" -s ) || \
      echo "  (note) demo.py exited non-zero — checking for hmr4d_results.pt anyway"
  fi

  # GVHMR's default output dir = $GVHMR_DIR/outputs/demo/<stem>/.
  # Move it under the per-task subfolder so the tree stays organised.
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

# ── 2) batch csv → npz (single Isaac Sim launch) ──────
shopt -s nullglob
csvs=( "$CSV_DIR"/*.csv )
if [[ ${#csvs[@]} -eq 0 ]]; then
  echo "[csv→npz] no csv files in $CSV_DIR — skipping stage 2."
  echo "[DONE] $TASK_NAME (no npz produced)"
  exit 0
fi

echo "[csv→npz] $CSV_DIR (${#csvs[@]} files) → $OUTPUT_DIR  @ ${OUTPUT_FPS} Hz"
# env_isaaclab_51's `etc/conda/activate.d/setenv.sh` adds the IsaacLab
# `_isaac_sim/python_packages` paths to PYTHONPATH (isaacsim isn't a pip
# package). We must `conda activate` that env explicitly — `isaaclab.sh -p`
# would also work, but it requires a TTY (calls `tput`), which fails in
# background shells with `'ansi+tabs': unknown terminal type` and silently
# falls back to /usr/bin/python.
# shellcheck source=/dev/null
# `set -u` aborts on any unbound var read inside the activate hook
# (IsaacLab's _isaac_sim/setup_conda_env.sh checks $ZSH_VERSION). Relax
# nounset around `conda activate`, then restore.
set +u
source "$HOME/miniforge/etc/profile.d/conda.sh"
conda activate env_isaaclab_51
set -u
"$ULAB_PY" \
  "$ULAB/scripts/pingpong_data_process/csv_to_npz_pingpong.py" \
  --input  "$CSV_DIR" \
  --output "$OUTPUT_DIR" \
  --input_fps  "$INPUT_FPS" \
  --output_fps "$OUTPUT_FPS" \
  --task_name "$TASK_NAME" \
  --paddle \
  --overwrite \
  --headless

echo "[DONE] $TASK_NAME → $OUTPUT_DIR"
