#!/bin/bash
#SBATCH --partition=day
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G

set -euo pipefail

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "[ERROR] SLURM_ARRAY_TASK_ID is not set." >&2
    exit 1
fi

if [ -z "${FRAME_DIR_LIST:-}" ]; then
    echo "[ERROR] FRAME_DIR_LIST is not set." >&2
    exit 2
fi

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
GIF_FPS="${GIF_FPS:-4}"
REGENERATE_SNAPSHOTS="${REGENERATE_SNAPSHOTS:-0}"

if ! type module >/dev/null 2>&1; then
    if [ -r /etc/profile.d/modules.sh ]; then
        source /etc/profile.d/modules.sh
    elif [ -r /usr/share/Modules/init/bash ]; then
        source /usr/share/Modules/init/bash
    elif [ -r /usr/share/lmod/lmod/init/bash ]; then
        source /usr/share/lmod/lmod/init/bash
    else
        echo "[ERROR] 'module' command is unavailable and module init scripts were not found." >&2
        exit 3
    fi
fi

module load miniconda/24.11.3
module load FFmpeg/7.0.2-GCCcore-13.3.0
module list

auto_conda_sh=""
if command -v conda >/dev/null 2>&1; then
    auto_conda_sh="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
fi
if [ -n "$auto_conda_sh" ] && [ -r "$auto_conda_sh" ]; then
    source "$auto_conda_sh"
elif [ -r /apps/software/2022b/software/miniconda/24.11.3/etc/profile.d/conda.sh ]; then
    source /apps/software/2022b/software/miniconda/24.11.3/etc/profile.d/conda.sh
else
    echo "[ERROR] Could not find conda.sh to activate py3." >&2
    exit 4
fi
set +u
conda activate py3
set -u

frame_dir="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$FRAME_DIR_LIST")"
if [ -z "$frame_dir" ]; then
    echo "[ERROR] No frame directory found for array index ${SLURM_ARRAY_TASK_ID}." >&2
    exit 5
fi

if [ ! -d "$frame_dir" ]; then
    echo "[ERROR] Frame directory '$frame_dir' does not exist." >&2
    exit 6
fi

svg_count="$(find "$frame_dir" -maxdepth 1 -type f -name '*.svg' | wc -l)"
if [ "$svg_count" -eq 0 ]; then
    echo "[ERROR] No SVG frames found in '$frame_dir'." >&2
    exit 7
fi

folder_name="$(basename "$frame_dir")"
output_gif="$frame_dir/${folder_name}.gif"
gif_input_dir="$frame_dir"
gif_frame_count="$svg_count"

if [ "$REGENERATE_SNAPSHOTS" = "1" ]; then
    echo "[INFO] Regenerating snapshot SVGs in '$frame_dir' from trajectory data."
    cd "$REPO_ROOT"
    python python/regenerate_phase_snapshots.py \
        --frame-dir "$frame_dir"
    gif_input_dir="$frame_dir/png"
    gif_frame_count="$(find "$gif_input_dir" -maxdepth 1 -type f -name '*.png' | wc -l)"
    if [ "$gif_frame_count" -eq 0 ]; then
        echo "[ERROR] No PNG frames were regenerated in '$gif_input_dir'." >&2
        exit 8
    fi
fi

echo "[INFO] Converting ${gif_frame_count} frame(s) from '$gif_input_dir' to '$output_gif' at ${GIF_FPS} fps."

cd "$REPO_ROOT"
python python/build_gif.py \
    --figure-dir "$gif_input_dir" \
    --output "$output_gif" \
    --fps "$GIF_FPS"
