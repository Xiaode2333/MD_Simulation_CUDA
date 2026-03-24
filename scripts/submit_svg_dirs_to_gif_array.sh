#!/bin/bash

set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 RESULT_ROOT [GIF_FPS] [JOB_NAME]" >&2
    exit 1
fi

RESULT_ROOT="$1"
GIF_FPS="${2:-4}"
JOB_NAME="${3:-$(basename "$RESULT_ROOT")_gif}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULT_ROOT="$(realpath "$RESULT_ROOT")"

if [ ! -d "$RESULT_ROOT" ]; then
    echo "[ERROR] Result root '$RESULT_ROOT' does not exist." >&2
    exit 2
fi

LOG_DIR="${RESULT_ROOT}/logs"
FRAME_DIR_LIST="${LOG_DIR}/pics2gif_dirs.txt"
mkdir -p "$LOG_DIR"

mapfile -t frame_dirs < <(find "$RESULT_ROOT" -type d -path '*/frames/*' | sort)

: > "$FRAME_DIR_LIST"
for frame_dir in "${frame_dirs[@]}"; do
    if find "$frame_dir" -maxdepth 1 -type f -name '*.svg' -print -quit | grep -q .; then
        printf '%s\n' "$frame_dir" >> "$FRAME_DIR_LIST"
    fi
done

task_count="$(wc -l < "$FRAME_DIR_LIST")"
if [ "$task_count" -eq 0 ]; then
    echo "[ERROR] No frame directories containing SVG files were found under '$RESULT_ROOT'." >&2
    exit 3
fi

array_end=$((task_count - 1))

echo "Submitting GIF conversion array for ${RESULT_ROOT}"
echo "  tasks: ${task_count}"
echo "  fps:   ${GIF_FPS}"
echo "  list:  ${FRAME_DIR_LIST}"

sbatch \
    --job-name="$JOB_NAME" \
    --array="0-${array_end}" \
    --output="${LOG_DIR}/pics2gif-%A_%a.out" \
    --error="${LOG_DIR}/pics2gif-%A_%a.err" \
    --export=ALL,REPO_ROOT="${REPO_ROOT}",FRAME_DIR_LIST="${FRAME_DIR_LIST}",GIF_FPS="${GIF_FPS}" \
    "${SCRIPT_DIR}/run_svg_dirs_to_gif_array.sh"
