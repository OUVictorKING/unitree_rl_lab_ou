#!/usr/bin/env bash

if [[ $# -lt 3 ]]; then
  echo "usage: rosbag_record_wrapper.sh OUTPUT_DIR LOG_FILE TOPIC..." >&2
  exit 2
fi

OUT="$1"
LOG="$2"
shift 2

mkdir -p "$(dirname "$LOG")"
exec >"$LOG" 2>&1

echo "[wrapper] start $(date --iso-8601=seconds)"
echo "[wrapper] output: $OUT"
echo "[wrapper] topics: $*"

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "[wrapper] missing /opt/ros/humble/setup.bash"
  exit 3
fi

source /opt/ros/humble/setup.bash

echo "[wrapper] ROS_DISTRO=${ROS_DISTRO:-unset}"
echo "[wrapper] RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-unset}"
echo "[wrapper] ros2=$(command -v ros2 || true)"
echo "[wrapper] run: ros2 bag record --storage sqlite3 -o '$OUT' $*"
trap 'echo "[wrapper] signal INT $(date --iso-8601=seconds)"; kill -INT "$child" 2>/dev/null; wait "$child"; rc=$?; echo "[wrapper] ros2 bag record exited rc=$rc"; exit "$rc"' INT
trap 'echo "[wrapper] signal TERM $(date --iso-8601=seconds)"; kill -TERM "$child" 2>/dev/null; wait "$child"; rc=$?; echo "[wrapper] ros2 bag record exited rc=$rc"; exit "$rc"' TERM

ros2 bag record --storage sqlite3 -o "$OUT" "$@" &
child=$!
echo "[wrapper] child pid=$child"
wait "$child"
rc=$?
echo "[wrapper] ros2 bag record exited rc=$rc"
exit "$rc"
