#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <bag_dir> [extra ros2 bag play args...]" >&2
  exit 2
fi

BAG="$1"
shift

echo "[replay] bag: ${BAG}"
echo "[replay] playing recorded /clock from the bag; do not add ros2 bag play --clock here."
echo "[replay] controller config must set FSM.Pingpong.ros.use_sim_time_for_replay=true for real-robot virtual replay."

exec ros2 bag play "${BAG}" \
  --topics /clock /pingpong/ball_state /pingpong/base_pose \
  "$@"
