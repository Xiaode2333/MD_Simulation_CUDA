#!/usr/bin/env python3
"""Regenerate snapshot SVGs for one frames/<PHASE> directory from trajectory data."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "python"))

from particle_csv import _finalize_plot, _scatter_particles


SNAPSHOT_RE = re.compile(r"^snapshot_(?P<index>\d+)_step_(?P<global_step>\d+)\.svg$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate snapshot SVGs for a single frames/<PHASE> directory."
    )
    parser.add_argument(
        "--frame-dir",
        type=Path,
        required=True,
        help="Path to a frames/<PHASE> directory containing snapshot_*.svg files.",
    )
    parser.add_argument(
        "--dpi",
        type=float,
        default=110.0,
        help="Output DPI for regenerated SVGs.",
    )
    return parser.parse_args()


def parse_metadata_line(line: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for token in line.strip().split():
        key, value = token.split("=", 1)
        parsed[key] = value
    return parsed


def infer_trajectory_path(temp_dir: Path, phase_name: str) -> Path:
    if phase_name == "NPH":
        return temp_dir / "trajectory.xyz"
    if phase_name == "NPH_PISTON":
        return temp_dir / "trajectory_piston.xyz"
    if phase_name == "NVT":
        for candidate in ("trajectory.xyz", "trajectory_piston.xyz"):
            path = temp_dir / candidate
            if path.exists():
                return path
        raise FileNotFoundError(f"Could not infer NVT trajectory under {temp_dir}")
    if phase_name in {"NVT_100", "NVT_500"}:
        return temp_dir / "trajectory_nvt.xyz"
    raise ValueError(f"Unsupported phase directory name: {phase_name}")


def infer_config_path(temp_dir: Path, phase_name: str) -> Path:
    config_dir = temp_dir / "configs"
    if phase_name == "NPH":
        return config_dir / "config_nph.json"
    if phase_name == "NPH_PISTON":
        return config_dir / "config_nph_piston.json"
    if phase_name == "NVT":
        return config_dir / "config_nvt.json"
    if phase_name == "NVT_100":
        return config_dir / "config_nvt_100.json"
    if phase_name == "NVT_500":
        return config_dir / "config_nvt_500.json"
    raise ValueError(f"Unsupported phase directory name: {phase_name}")


def load_sigmas(config_path: Path) -> tuple[float, float]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return float(config["SIGMA_AA"]), float(config["SIGMA_BB"])


def collect_targets(frame_dir: Path) -> dict[int, Path]:
    targets: dict[int, Path] = {}
    for path in sorted(frame_dir.glob("snapshot_*_step_*.svg")):
        match = SNAPSHOT_RE.match(path.name)
        if not match:
            continue
        global_step = int(match.group("global_step"))
        targets[global_step] = path
    if not targets:
        raise ValueError(f"No snapshot_*.svg targets found under {frame_dir}")
    return targets


def parse_particle_lines(particle_lines: list[str]) -> tuple[np.ndarray, np.ndarray]:
    positions = np.empty((len(particle_lines), 2), dtype=np.float64)
    types = np.empty((len(particle_lines),), dtype=np.int32)
    for idx, line in enumerate(particle_lines):
        tag, x_str, y_str, *_ = line.split()
        positions[idx, 0] = float(x_str)
        positions[idx, 1] = float(y_str)
        types[idx] = 0 if tag == "A" else 1
    return positions, types


def figure_size_for_bounds(
    x_left: float,
    x_right: float,
    y_bottom: float,
    y_top: float,
    *,
    l_ref: float = 50.0,
) -> tuple[float, float]:
    width = max(x_right - x_left, 1.0)
    height = max(y_top - y_bottom, 1.0)
    fig_width_in = 10.0
    if width > l_ref:
        fig_width_in *= width / l_ref
        fig_width_in = min(fig_width_in, 50.0)
    fig_height_in = fig_width_in * (height / width)
    return fig_width_in, fig_height_in


def render_frame(
    *,
    positions: np.ndarray,
    types: np.ndarray,
    sigma_aa: float,
    sigma_bb: float,
    frame_box_w: float,
    frame_box_h: float,
    canvas_bounds: tuple[float, float, float, float],
    output_svg: Path,
    output_png: Path,
    dpi: float,
) -> None:
    x_left, x_right, y_bottom, y_top = canvas_bounds
    fig_width_in, fig_height_in = figure_size_for_bounds(x_left, x_right, y_bottom, y_top)
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=dpi)
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bottom, y_top)

    metadata = {
        "sigma_aa": sigma_aa,
        "sigma_bb": sigma_bb,
        "draw_box": True,
    }
    _scatter_particles(ax, positions, types, metadata, fig_width_in, x_right - x_left)
    _finalize_plot(ax, True, (0.0, frame_box_w, 0.0, frame_box_h))
    fig.savefig(output_svg, dpi=dpi, bbox_inches=None)
    fig.savefig(output_png, dpi=dpi, bbox_inches=None)
    plt.close(fig)


def regenerate_snapshots(frame_dir: Path, dpi: float) -> int:
    frame_dir = frame_dir.resolve()
    phase_name = frame_dir.name
    temp_dir = frame_dir.parent.parent
    if frame_dir.parent.name != "frames":
        raise ValueError(f"Expected a frames/<PHASE> directory, got {frame_dir}")

    trajectory_path = infer_trajectory_path(temp_dir, phase_name)
    config_path = infer_config_path(temp_dir, phase_name)
    sigma_aa, sigma_bb = load_sigmas(config_path)
    targets = collect_targets(frame_dir)
    remaining_steps = set(targets)
    png_dir = frame_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    for stale_png in png_dir.glob("snapshot_*_step_*.png"):
        stale_png.unlink()

    frames: dict[int, dict[str, object]] = {}
    with trajectory_path.open("r", encoding="utf-8") as handle:
        while remaining_steps:
            header = handle.readline()
            if not header:
                break

            header = header.strip()
            if not header:
                continue

            n_particles = int(header)
            metadata = parse_metadata_line(handle.readline())
            phase = metadata["phase"]
            global_step = int(metadata["global_step"])

            particle_lines = [handle.readline().strip() for _ in range(n_particles)]
            if phase != phase_name or global_step not in remaining_steps:
                continue

            positions, types = parse_particle_lines(particle_lines)
            frames[global_step] = {
                "positions": positions,
                "types": types,
                "box_w": float(metadata["Lx"]),
                "box_h": float(metadata["Ly"]),
            }
            remaining_steps.remove(global_step)

    if remaining_steps:
        missing = ", ".join(str(step) for step in sorted(remaining_steps))
        raise RuntimeError(
            f"Failed to find {len(remaining_steps)} target frame(s) in {trajectory_path}: {missing}"
        )

    all_left = min(0.0, *(float(frame["positions"][:, 0].min()) for frame in frames.values()))
    all_right = max(
        *(max(float(frame["box_w"]), float(frame["positions"][:, 0].max())) for frame in frames.values())
    )
    all_bottom = min(0.0, *(float(frame["positions"][:, 1].min()) for frame in frames.values()))
    all_top = max(
        *(max(float(frame["box_h"]), float(frame["positions"][:, 1].max())) for frame in frames.values())
    )
    canvas_bounds = (all_left, all_right, all_bottom, all_top)

    for global_step, output_svg in sorted(targets.items()):
        frame = frames[global_step]
        render_frame(
            positions=frame["positions"],
            types=frame["types"],
            sigma_aa=sigma_aa,
            sigma_bb=sigma_bb,
            frame_box_w=float(frame["box_w"]),
            frame_box_h=float(frame["box_h"]),
            canvas_bounds=canvas_bounds,
            output_svg=output_svg,
            output_png=png_dir / output_svg.with_suffix(".png").name,
            dpi=dpi,
        )

    return len(targets)


def main() -> None:
    args = parse_args()
    try:
        written = regenerate_snapshots(args.frame_dir, args.dpi)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Regenerated {written} snapshot SVG(s) under {args.frame_dir.resolve()}")


if __name__ == "__main__":
    main()
