#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-}"
if [[ -z "${OUT}" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  OUT="/home/woan/HumanoidProject/unitree_rl_lab/deploy/robots/g1_23dof_pingpong/bags/pingpong_sim_${TS}"
fi

mkdir -p "$(dirname "${OUT}")"

echo "[record] output: ${OUT}"
echo "[record] topics:"
echo "  /clock"
echo "  /pingpong/ball_state"
echo "  /pingpong/base_pose"

exec ros2 bag record \
  -o "${OUT}" \
  /clock \
  /pingpong/ball_state \
  /pingpong/base_pose
