from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRIANGLE_TYPES = ("AAB", "ABB")
NEIGHBOR_PLOT_SPECS = (
    ("near_AAB", "Na", "near AAB"),
    ("near_AAA", "NA", "near AAA"),
    ("near_BBB", "NB", "near BBB"),
    ("near_ABB", "Nb", "near ABB / BBA"),
)
TEMPERATURE_PATTERN = re.compile(r"T=(?P<temperature>\d+(?:\.\d+)?)")
CSV_NAME = "aab_abb_triangle_neighbors.csv"


def find_repo_root(start: Path | None = None) -> Path:
    start = start or Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "python" / "particle_csv.py").exists():
            return candidate
    raise RuntimeError("Could not locate the repository root from the current working directory.")


def discover_triangle_neighbor_results(
    result_root: Path,
    temperatures: list[float] | None = None,
    max_temperatures: int | None = None,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    allowed = None if temperatures is None else {round(float(value), 8) for value in temperatures}

    for csv_path in sorted(result_root.glob(f"T=*/{CSV_NAME}")):
        match = TEMPERATURE_PATTERN.search(csv_path.as_posix())
        if match is None:
            continue
        temperature = float(match.group("temperature"))
        if allowed is not None and round(temperature, 8) not in allowed:
            continue
        records.append(
            {
                "temperature": temperature,
                "temperature_label": f"T={temperature:.1f}",
                "csv_path": csv_path.resolve(),
            }
        )

    if not records:
        return pd.DataFrame(columns=["temperature", "temperature_label", "csv_path"])

    manifest = pd.DataFrame.from_records(records).sort_values("temperature").reset_index(drop=True)
    if max_temperatures is not None and not manifest.empty:
        manifest = manifest.head(max_temperatures).reset_index(drop=True)
    return manifest


def _iter_csv_chunks(csv_path: Path, usecols: list[str], chunksize: int) -> Any:
    dtype_map = {
        "frame_index": np.int32,
        "triangle_type": "category",
        "area": np.float64,
        "near_AAA": np.int8,
        "near_AAB": np.int8,
        "near_ABB": np.int8,
        "near_BBB": np.int8,
    }
    return pd.read_csv(
        csv_path,
        usecols=usecols,
        dtype={column: dtype_map[column] for column in usecols if column in dtype_map},
        chunksize=chunksize,
    )


def compute_global_area_bin_edges(
    csv_paths: list[Path],
    area_bin_count: int = 120,
    chunksize: int = 1_000_000,
) -> np.ndarray:
    area_min = np.inf
    area_max = -np.inf
    for csv_path in csv_paths:
        for chunk in _iter_csv_chunks(csv_path, usecols=["area"], chunksize=chunksize):
            values = chunk["area"].to_numpy(dtype=np.float64, copy=False)
            if values.size == 0:
                continue
            area_min = min(area_min, float(np.min(values)))
            area_max = max(area_max, float(np.max(values)))

    if not np.isfinite(area_min) or not np.isfinite(area_max):
        raise RuntimeError("Could not determine the area range from the available CSV files.")

    if np.isclose(area_min, area_max):
        area_max = area_min + 1.0

    return np.linspace(area_min, area_max, area_bin_count + 1, dtype=np.float64)


def _empty_state(area_bin_count: int, neighbor_bin_count: int) -> dict[str, Any]:
    return {
        "frame_count": 0,
        "mixed_frame_count": 0,
        "area_hist_sum": np.zeros(area_bin_count, dtype=np.float64),
        "frame_mixed_area_sum_values": [],
        "count_sum": {triangle_type: 0.0 for triangle_type in TRIANGLE_TYPES},
        "count_sq_sum": {triangle_type: 0.0 for triangle_type in TRIANGLE_TYPES},
        "active_frame_count": {triangle_type: 0 for triangle_type in TRIANGLE_TYPES},
        "mean_area_sum": {triangle_type: 0.0 for triangle_type in TRIANGLE_TYPES},
        "mean_area_sq_sum": {triangle_type: 0.0 for triangle_type in TRIANGLE_TYPES},
        "neighbor_hist_sum": {
            triangle_type: {
                column: np.zeros(neighbor_bin_count, dtype=np.float64)
                for column, _, _ in NEIGHBOR_PLOT_SPECS
            }
            for triangle_type in TRIANGLE_TYPES
        },
        "neighbor_mean_sum": {
            triangle_type: {column: 0.0 for column, _, _ in NEIGHBOR_PLOT_SPECS}
            for triangle_type in TRIANGLE_TYPES
        },
    }


def _safe_mean(sum_value: float, count: int) -> float:
    return float(sum_value / count) if count else np.nan


def _safe_std(sum_sq_value: float, mean_value: float, count: int) -> float:
    if not count:
        return np.nan
    variance = max(sum_sq_value / count - mean_value * mean_value, 0.0)
    return float(np.sqrt(variance))


def _process_frame(
    frame_df: pd.DataFrame,
    area_bin_edges: np.ndarray,
    neighbor_bin_edges: np.ndarray,
    area_bin_widths: np.ndarray,
    state: dict[str, Any],
) -> None:
    state["frame_count"] += 1

    if not frame_df.empty:
        area_values = frame_df["area"].to_numpy(dtype=np.float64, copy=False)
        state["frame_mixed_area_sum_values"].append(float(np.sum(area_values)))
        area_counts, _ = np.histogram(area_values, bins=area_bin_edges)
        if area_counts.sum() > 0:
            state["area_hist_sum"] += area_counts / (area_counts.sum() * area_bin_widths)
            state["mixed_frame_count"] += 1
    else:
        state["frame_mixed_area_sum_values"].append(0.0)

    for triangle_type in TRIANGLE_TYPES:
        type_df = frame_df.loc[frame_df["triangle_type"] == triangle_type]
        n_triangles = int(len(type_df))
        state["count_sum"][triangle_type] += n_triangles
        state["count_sq_sum"][triangle_type] += n_triangles * n_triangles
        if n_triangles == 0:
            continue

        state["active_frame_count"][triangle_type] += 1

        mean_area = float(type_df["area"].mean())
        state["mean_area_sum"][triangle_type] += mean_area
        state["mean_area_sq_sum"][triangle_type] += mean_area * mean_area

        for column, _, _ in NEIGHBOR_PLOT_SPECS:
            values = type_df[column].to_numpy(dtype=np.float64, copy=False)
            counts, _ = np.histogram(values, bins=neighbor_bin_edges)
            state["neighbor_hist_sum"][triangle_type][column] += counts / n_triangles
            state["neighbor_mean_sum"][triangle_type][column] += float(np.mean(values))


def summarize_temperature_csv(
    csv_path: Path,
    area_bin_edges: np.ndarray,
    neighbor_bin_edges: np.ndarray | None = None,
    chunksize: int = 1_000_000,
    verbose: bool = False,
) -> dict[str, Any]:
    neighbor_bin_edges = (
        np.asarray(neighbor_bin_edges, dtype=np.float64)
        if neighbor_bin_edges is not None
        else np.arange(-0.5, 4.5, 1.0, dtype=np.float64)
    )
    area_bin_widths = np.diff(area_bin_edges)
    state = _empty_state(len(area_bin_edges) - 1, len(neighbor_bin_edges) - 1)

    usecols = ["frame_index", "triangle_type", "area", *[column for column, _, _ in NEIGHBOR_PLOT_SPECS]]
    carry_df: pd.DataFrame | None = None

    if verbose:
        print(f"Processing {csv_path}")

    for chunk in _iter_csv_chunks(csv_path, usecols=usecols, chunksize=chunksize):
        if carry_df is not None and not carry_df.empty:
            chunk = pd.concat([carry_df, chunk], ignore_index=True)
            carry_df = None

        if chunk.empty:
            continue

        last_frame_index = int(chunk["frame_index"].iloc[-1])
        carry_df = chunk.loc[chunk["frame_index"] == last_frame_index].copy()
        body_df = chunk.loc[chunk["frame_index"] != last_frame_index]

        if not body_df.empty:
            for _, frame_df in body_df.groupby("frame_index", sort=False):
                _process_frame(frame_df, area_bin_edges, neighbor_bin_edges, area_bin_widths, state)

    if carry_df is not None and not carry_df.empty:
        _process_frame(carry_df, area_bin_edges, neighbor_bin_edges, area_bin_widths, state)

    match = TEMPERATURE_PATTERN.search(csv_path.as_posix())
    temperature = float(match.group("temperature")) if match else np.nan

    summary: dict[str, Any] = {
        "temperature": temperature,
        "temperature_label": f"T={temperature:.1f}",
        "csv_path": csv_path.resolve(),
        "frame_count": state["frame_count"],
        "mixed_frame_count": state["mixed_frame_count"],
        "frame_mixed_area_sum_values": np.asarray(state["frame_mixed_area_sum_values"], dtype=np.float64),
        "area_hist": (
            state["area_hist_sum"] / state["mixed_frame_count"]
            if state["mixed_frame_count"]
            else np.zeros_like(state["area_hist_sum"])
        ),
        "active_frame_count": dict(state["active_frame_count"]),
        "mean_count": {},
        "std_count": {},
        "mean_area": {},
        "std_area": {},
        "neighbor_hist": {triangle_type: {} for triangle_type in TRIANGLE_TYPES},
        "neighbor_mean": {triangle_type: {} for triangle_type in TRIANGLE_TYPES},
    }

    for triangle_type in TRIANGLE_TYPES:
        frame_count = state["frame_count"]
        active_frame_count = state["active_frame_count"][triangle_type]

        mean_count = _safe_mean(state["count_sum"][triangle_type], frame_count)
        summary["mean_count"][triangle_type] = mean_count
        summary["std_count"][triangle_type] = _safe_std(
            state["count_sq_sum"][triangle_type],
            0.0 if np.isnan(mean_count) else mean_count,
            frame_count,
        )

        mean_area = _safe_mean(state["mean_area_sum"][triangle_type], active_frame_count)
        summary["mean_area"][triangle_type] = mean_area
        summary["std_area"][triangle_type] = _safe_std(
            state["mean_area_sq_sum"][triangle_type],
            0.0 if np.isnan(mean_area) else mean_area,
            active_frame_count,
        )

        for column, _, _ in NEIGHBOR_PLOT_SPECS:
            if active_frame_count:
                summary["neighbor_hist"][triangle_type][column] = (
                    state["neighbor_hist_sum"][triangle_type][column] / active_frame_count
                )
            else:
                summary["neighbor_hist"][triangle_type][column] = np.zeros(
                    len(neighbor_bin_edges) - 1, dtype=np.float64
                )
            summary["neighbor_mean"][triangle_type][column] = _safe_mean(
                state["neighbor_mean_sum"][triangle_type][column], active_frame_count
            )

    return summary


def build_triangle_neighbor_dataset(
    result_root: Path,
    area_bin_count: int = 120,
    chunksize: int = 1_000_000,
    temperatures: list[float] | None = None,
    max_temperatures: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    manifest = discover_triangle_neighbor_results(
        result_root=result_root,
        temperatures=temperatures,
        max_temperatures=max_temperatures,
    )
    if manifest.empty:
        raise RuntimeError(f"No '{CSV_NAME}' files were found under '{result_root}'.")

    area_bin_edges = compute_global_area_bin_edges(
        csv_paths=manifest["csv_path"].tolist(),
        area_bin_count=area_bin_count,
        chunksize=chunksize,
    )
    neighbor_bin_edges = np.arange(-0.5, 4.5, 1.0, dtype=np.float64)

    summaries = [
        summarize_temperature_csv(
            csv_path=row.csv_path,
            area_bin_edges=area_bin_edges,
            neighbor_bin_edges=neighbor_bin_edges,
            chunksize=chunksize,
            verbose=verbose,
        )
        for row in manifest.itertuples(index=False)
    ]

    return {
        "result_root": result_root.resolve(),
        "manifest": manifest.copy(),
        "area_bin_edges": area_bin_edges,
        "area_bin_centers": 0.5 * (area_bin_edges[:-1] + area_bin_edges[1:]),
        "neighbor_bin_edges": neighbor_bin_edges,
        "neighbor_bin_centers": np.arange(0, len(neighbor_bin_edges) - 1, dtype=np.int32),
        "summaries": sorted(summaries, key=lambda summary: summary["temperature"]),
    }


def make_summary_table(analysis: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary in analysis["summaries"]:
        rows.append(
            {
                "temperature": summary["temperature"],
                "frames_total": summary["frame_count"],
                "frames_with_mixed": summary["mixed_frame_count"],
                "frames_with_AAB": summary["active_frame_count"]["AAB"],
                "frames_with_ABB": summary["active_frame_count"]["ABB"],
                "mean_AAB_per_frame": summary["mean_count"]["AAB"],
                "mean_ABB_per_frame": summary["mean_count"]["ABB"],
                "mean_AAB_area": summary["mean_area"]["AAB"],
                "mean_ABB_area": summary["mean_area"]["ABB"],
            }
        )
    return pd.DataFrame(rows).sort_values("temperature").reset_index(drop=True)


def _temperature_palette(analysis: dict[str, Any]) -> dict[float, Any]:
    temperatures = [summary["temperature"] for summary in analysis["summaries"]]
    color_positions = [0.55] if len(temperatures) == 1 else np.linspace(0.12, 0.92, len(temperatures))
    cmap = plt.colormaps["viridis"]
    return {temperature: cmap(position) for temperature, position in zip(temperatures, color_positions)}


def _save_figure(fig: Any, figure_dir: Path | None, filename: str) -> Path | None:
    if figure_dir is None:
        return None
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / filename
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    return figure_path


def plot_area_overlay(analysis: dict[str, Any], figure_dir: Path | None = None) -> tuple[Any, Any]:
    palette = _temperature_palette(analysis)
    x_values = analysis["area_bin_centers"]

    fig, ax = plt.subplots(figsize=(9, 6))
    for summary in analysis["summaries"]:
        ax.plot(
            x_values,
            summary["area_hist"],
            linewidth=2.0,
            color=palette[summary["temperature"]],
            label=summary["temperature_label"],
        )

    ax.set_xlabel("Triangle area")
    ax.set_ylabel("Mean per-frame probability density")
    ax.set_title("Mixed-Triangle Area Distribution (AAB + ABB)")
    ax.legend(title="Temperature", ncols=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    _save_figure(fig, figure_dir, "mixed_triangle_area_overlay.png")
    return fig, ax


def plot_frame_mixed_area_sum_histogram(
    analysis: dict[str, Any],
    figure_dir: Path | None = None,
    bin_count: int = 80,
) -> tuple[Any, Any]:
    palette = _temperature_palette(analysis)
    filename = "mixed_triangle_area_sum_per_frame_histogram_overlay.png"

    fig, ax = plt.subplots(figsize=(9, 6))

    finite_values_by_temperature: dict[float, np.ndarray] = {}
    finite_values: list[np.ndarray] = []
    for summary in analysis["summaries"]:
        values = np.asarray(summary["frame_mixed_area_sum_values"], dtype=np.float64)
        values = values[np.isfinite(values)]
        finite_values_by_temperature[summary["temperature"]] = values
        if values.size > 0:
            finite_values.append(values)

    if not finite_values:
        ax.text(0.5, 0.5, "No plottable data", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        _save_figure(fig, figure_dir, filename)
        return fig, ax

    value_min = min(float(np.min(values)) for values in finite_values)
    value_max = max(float(np.max(values)) for values in finite_values)
    if np.isclose(value_min, value_max):
        value_max = value_min + 1.0
    bin_edges = np.linspace(value_min, value_max, bin_count + 1, dtype=np.float64)

    for summary in analysis["summaries"]:
        values = finite_values_by_temperature[summary["temperature"]]
        if values.size == 0:
            continue
        ax.hist(
            values,
            bins=bin_edges,
            density=True,
            histtype="step",
            linewidth=2.0,
            color=palette[summary["temperature"]],
            label=summary["temperature_label"],
        )

    ax.set_xlabel("Sum of AAB + ABB area per frame")
    ax.set_ylabel("Probability density")
    ax.set_title("Per-Frame Mixed-Triangle Area-Sum Histogram")
    ax.legend(title="Temperature", ncols=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    _save_figure(fig, figure_dir, filename)
    return fig, ax


def plot_neighbor_histograms(
    analysis: dict[str, Any],
    triangle_type: str,
    figure_dir: Path | None = None,
) -> tuple[Any, Any]:
    if triangle_type not in TRIANGLE_TYPES:
        raise ValueError(f"Unsupported triangle type '{triangle_type}'. Expected one of {TRIANGLE_TYPES}.")

    palette = _temperature_palette(analysis)
    x_values = analysis["neighbor_bin_centers"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for ax, (column, short_label, long_label) in zip(axes.flat, NEIGHBOR_PLOT_SPECS):
        for summary in analysis["summaries"]:
            ax.plot(
                x_values,
                summary["neighbor_hist"][triangle_type][column],
                marker="o",
                linewidth=1.8,
                color=palette[summary["temperature"]],
                label=summary["temperature_label"],
            )
        ax.set_title(f"{triangle_type}: {short_label} ({long_label})")
        ax.set_xlabel("Neighbor count")
        ax.set_ylabel("Mean per-frame probability")
        ax.set_xticks(x_values)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Temperature", loc="upper center", ncols=3, frameon=False)
    fig.suptitle(f"{triangle_type} Neighbor-Count Histograms", y=1.02)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    _save_figure(fig, figure_dir, f"{triangle_type.lower()}_neighbor_histograms.png")
    return fig, axes


def plot_abundance_vs_temperature(
    analysis: dict[str, Any],
    figure_dir: Path | None = None,
) -> tuple[Any, Any]:
    summary_df = make_summary_table(analysis)
    temperatures = summary_df["temperature"].to_numpy(dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        temperatures,
        summary_df["mean_AAB_per_frame"].to_numpy(dtype=np.float64),
        marker="o",
        linewidth=2.0,
        label="AAB",
    )
    ax.plot(
        temperatures,
        summary_df["mean_ABB_per_frame"].to_numpy(dtype=np.float64),
        marker="s",
        linewidth=2.0,
        label="ABB",
    )

    ax.set_xlabel("Temperature")
    ax.set_ylabel("Mean triangles per frame")
    ax.set_title("Mixed-Triangle Abundance vs Temperature")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    _save_figure(fig, figure_dir, "mixed_triangle_abundance_vs_temperature.png")
    return fig, ax


def plot_mean_area_vs_temperature(
    analysis: dict[str, Any],
    figure_dir: Path | None = None,
) -> tuple[Any, Any]:
    summary_df = make_summary_table(analysis)
    temperatures = summary_df["temperature"].to_numpy(dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        temperatures,
        summary_df["mean_AAB_area"].to_numpy(dtype=np.float64),
        marker="o",
        linewidth=2.0,
        label="AAB",
    )
    ax.plot(
        temperatures,
        summary_df["mean_ABB_area"].to_numpy(dtype=np.float64),
        marker="s",
        linewidth=2.0,
        label="ABB",
    )

    ax.set_xlabel("Temperature")
    ax.set_ylabel("Mean triangle area")
    ax.set_title("Mean Mixed-Triangle Area vs Temperature")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    _save_figure(fig, figure_dir, "mixed_triangle_mean_area_vs_temperature.png")
    return fig, ax


def plot_neighbor_heatmaps(
    analysis: dict[str, Any],
    figure_dir: Path | None = None,
) -> tuple[Any, Any]:
    temperatures = [summary["temperature"] for summary in analysis["summaries"]]
    y_labels = [f"{temperature:.1f}" for temperature in temperatures]
    x_labels = [f"{short_label}\n({long_label})" for _, short_label, long_label in NEIGHBOR_PLOT_SPECS]

    panels: dict[str, np.ndarray] = {}
    for triangle_type in TRIANGLE_TYPES:
        panels[triangle_type] = np.array(
            [
                [summary["neighbor_mean"][triangle_type][column] for column, _, _ in NEIGHBOR_PLOT_SPECS]
                for summary in analysis["summaries"]
            ],
            dtype=np.float64,
        )

    vmax = max(float(np.nanmax(panel)) for panel in panels.values())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, triangle_type in zip(axes, TRIANGLE_TYPES):
        panel = panels[triangle_type]
        image = ax.imshow(panel, aspect="auto", cmap="magma", vmin=0.0, vmax=vmax)
        ax.set_title(f"{triangle_type} mean neighbor counts")
        ax.set_xlabel("Neighbor category")
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels)
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels)
        if triangle_type == "AAB":
            ax.set_ylabel("Temperature")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Mean neighbors")

    fig.tight_layout()
    _save_figure(fig, figure_dir, "mixed_triangle_neighbor_heatmaps.png")
    return fig, axes


def save_default_triangle_neighbor_figures(
    analysis: dict[str, Any],
    figure_dir: Path,
) -> dict[str, Path | None]:
    saved_paths = {
        "area_overlay": _save_figure(plot_area_overlay(analysis)[0], figure_dir, "mixed_triangle_area_overlay.png"),
        "frame_area_sum_histogram": _save_figure(
            plot_frame_mixed_area_sum_histogram(analysis)[0],
            figure_dir,
            "mixed_triangle_area_sum_per_frame_histogram_overlay.png",
        ),
        "aab_histograms": _save_figure(
            plot_neighbor_histograms(analysis, triangle_type="AAB")[0],
            figure_dir,
            "aab_neighbor_histograms.png",
        ),
        "abb_histograms": _save_figure(
            plot_neighbor_histograms(analysis, triangle_type="ABB")[0],
            figure_dir,
            "abb_neighbor_histograms.png",
        ),
        "abundance": _save_figure(
            plot_abundance_vs_temperature(analysis)[0],
            figure_dir,
            "mixed_triangle_abundance_vs_temperature.png",
        ),
        "mean_area": _save_figure(
            plot_mean_area_vs_temperature(analysis)[0],
            figure_dir,
            "mixed_triangle_mean_area_vs_temperature.png",
        ),
        "heatmaps": _save_figure(
            plot_neighbor_heatmaps(analysis)[0],
            figure_dir,
            "mixed_triangle_neighbor_heatmaps.png",
        ),
    }
    plt.close("all")
    return saved_paths


__all__ = [
    "CSV_NAME",
    "NEIGHBOR_PLOT_SPECS",
    "TRIANGLE_TYPES",
    "build_triangle_neighbor_dataset",
    "compute_global_area_bin_edges",
    "discover_triangle_neighbor_results",
    "find_repo_root",
    "make_summary_table",
    "plot_abundance_vs_temperature",
    "plot_area_overlay",
    "plot_frame_mixed_area_sum_histogram",
    "plot_mean_area_vs_temperature",
    "plot_neighbor_heatmaps",
    "plot_neighbor_histograms",
    "save_default_triangle_neighbor_figures",
    "summarize_temperature_csv",
]
