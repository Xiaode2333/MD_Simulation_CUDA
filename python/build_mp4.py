#!/usr/bin/env python3
"""Build an MP4 video from a sequence of PNG or SVG figures."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

VALID_EXTENSIONS = {".png", ".svg"}
_NUMBER_GROUP = re.compile(r"(\d+)")


def _natural_key(path: Path):
    # Natural sort so img2.png comes before img10.png.
    return [int(token) if token.isdigit() else token.lower() for token in _NUMBER_GROUP.split(path.name)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a directory of PNG/SVG frames into an MP4.")
    parser.add_argument("--figure-dir", type=Path, required=True, help="Directory containing figure frames.")
    parser.add_argument("--start-filename", default=None, help="First filename (inclusive) to include.")
    parser.add_argument("--end-filename", default=None, help="Last filename (inclusive) to include.")
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second for the output video.")
    parser.add_argument("--output", type=Path, required=True, help="Output MP4 path.")
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


def write_concat_file(frames: List[Path]) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
    with handle as tmp:
        for frame in frames:
            resolved = frame.resolve()
            safe_path = str(resolved).replace("'", r"'\''")
            tmp.write(f"file '{safe_path}'\n")
    return Path(handle.name)


def ensure_raster_frames(frames: List[Path]) -> Tuple[List[Path], tempfile.TemporaryDirectory | None]:
    """Convert SVG frames to PNG in a temp directory so ffmpeg can ingest them."""
    temp_dir: tempfile.TemporaryDirectory | None = None
    raster_frames: List[Path] = []
    for idx, frame in enumerate(frames):
        if frame.suffix.lower() != ".svg":
            raster_frames.append(frame)
            continue
        if temp_dir is None:
            temp_dir = tempfile.TemporaryDirectory()
        raster_path = Path(temp_dir.name) / f"frame_{idx:06d}.png"
        try:
            import cairosvg  # type: ignore
        except ImportError:
            # Clean up immediately to avoid dangling temp dir if rasterization failed.
            temp_dir.cleanup()
            raise ImportError(
                "cairosvg is required to rasterize SVG frames. Install it (pip install cairosvg) "
                "or load the module that provides it."
            ) from None
        cairosvg.svg2png(url=str(frame), write_to=str(raster_path))
        raster_frames.append(raster_path)
    # Preserve the original ordering.
    return raster_frames, temp_dir


def build_mp4(frames: List[Path], fps: float, output: Path) -> None:
    if fps <= 0:
        raise ValueError("fps must be positive.")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    temp_dir: tempfile.TemporaryDirectory | None = None
    list_file: Path | None = None
    raster_frames, temp_dir = ensure_raster_frames(frames)
    list_file = write_concat_file(raster_frames)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-vf",
        f"fps={fps}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise FileNotFoundError("ffmpeg not found. Please install ffmpeg or load the module that provides it.")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed with exit code {exc.returncode}") from exc
    finally:
        if list_file is not None:
            list_file.unlink(missing_ok=True)
        if temp_dir is not None:
            temp_dir.cleanup()


def main() -> None:
    args = parse_args()
    try:
        frames = collect_frames(args.figure_dir, args.start_filename, args.end_filename)
        build_mp4(frames, args.fps, args.output)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Wrote {len(frames)} frame(s) to {args.output.resolve()}")


if __name__ == "__main__":
    main()
