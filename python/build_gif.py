#!/usr/bin/env python3
"""Build a looping GIF from a naturally sorted sequence of PNG or SVG frames."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

VALID_EXTENSIONS = {".png", ".svg"}
_NUMBER_GROUP = re.compile(r"(\d+)")


def _natural_key(path: Path):
    # Natural sort so img2.svg comes before img10.svg.
    return [int(token) if token.isdigit() else token.lower() for token in _NUMBER_GROUP.split(path.name)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a directory of PNG/SVG frames into a looping GIF.")
    parser.add_argument("--figure-dir", type=Path, required=True, help="Directory containing figure frames.")
    parser.add_argument("--start-filename", default=None, help="First filename (inclusive) to include.")
    parser.add_argument("--end-filename", default=None, help="Last filename (inclusive) to include.")
    parser.add_argument("--fps", type=float, default=4.0, help="Frames per second for the output GIF.")
    parser.add_argument("--output", type=Path, required=True, help="Output GIF path.")
    return parser.parse_args()


def collect_frames(figure_dir: Path, start: str | None, end: str | None) -> List[Path]:
    if not figure_dir.is_dir():
        raise ValueError(f"Figure directory does not exist or is not a directory: {figure_dir}")

    frames = sorted(
        (path for path in figure_dir.iterdir() if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS),
        key=_natural_key,
    )
    if not frames:
        raise ValueError(f"No PNG or SVG files found in {figure_dir}")

    names = [frame.name for frame in frames]
    start_idx = 0
    end_idx = len(frames)

    if start is not None:
        if start not in names:
            raise ValueError(f"start-filename not found in directory: {start}")
        start_idx = names.index(start)
    if end is not None:
        if end not in names:
            raise ValueError(f"end-filename not found in directory: {end}")
        end_idx = names.index(end) + 1
    if start_idx >= end_idx:
        raise ValueError("start-filename comes after end-filename in sorted order.")

    return frames[start_idx:end_idx]


def rasterize_frames(frames: List[Path], temp_dir: Path) -> str:
    try:
        import cairosvg  # type: ignore
    except ImportError:
        raise ImportError(
            "cairosvg is required to rasterize SVG frames. Install it or activate an environment that provides it."
        ) from None

    for idx, frame in enumerate(frames):
        raster_path = temp_dir / f"frame_{idx:06d}.png"
        if frame.suffix.lower() == ".svg":
            cairosvg.svg2png(url=str(frame), write_to=str(raster_path))
        else:
            shutil.copy2(frame, raster_path)

    return str(temp_dir / "frame_%06d.png")


def build_gif(frames: List[Path], fps: float, output: Path) -> None:
    if fps <= 0:
        raise ValueError("fps must be positive.")
    if output.suffix.lower() != ".gif":
        raise ValueError("output must end in .gif")

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        pattern = rasterize_frames(frames, temp_dir)
        palette_path = temp_dir / "palette.png"

        palette_cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            pattern,
            "-vf",
            "palettegen=stats_mode=diff",
            str(palette_path),
        ]
        gif_cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            pattern,
            "-i",
            str(palette_path),
            "-lavfi",
            "paletteuse=dither=sierra2_4a",
            "-loop",
            "0",
            str(output),
        ]

        try:
            subprocess.run(palette_cmd, check=True)
            subprocess.run(gif_cmd, check=True)
        except FileNotFoundError:
            raise FileNotFoundError("ffmpeg not found. Please install ffmpeg or load the module that provides it.")
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"ffmpeg failed with exit code {exc.returncode}") from exc


def main() -> None:
    args = parse_args()
    try:
        frames = collect_frames(args.figure_dir, args.start_filename, args.end_filename)
        build_gif(frames, args.fps, args.output)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Wrote {len(frames)} frame(s) to {args.output.resolve()}")


if __name__ == "__main__":
    main()
