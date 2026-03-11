"""Helpers for reading & plotting particle CSV snapshots produced by print_particles."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any

import matplotlib.pyplot as plt
import numpy as np


Metadata = Dict[str, Union[float, int, bool]]


def _parse_metadata_row(row: list[str]) -> Metadata:
    tokens = [token.strip() for token in row if token.strip() != ""]
    if len(tokens) % 2 != 0:
        raise ValueError("Metadata row must contain key,value pairs")

    metadata: Metadata = {}
    for key, value in zip(tokens[::2], tokens[1::2]):
        if key == "n_particles":
            metadata[key] = int(float(value))
        elif key == "draw_box":
            metadata[key] = bool(int(float(value)))
        elif key == "n_triangles":
            metadata[key] = int(float(value))
        elif key == "n_interfaces":
            metadata[key] = int(float(value))
        else:
            metadata[key] = float(value)
    return metadata


def _next_tokens(reader: csv.reader) -> list[str] | None:
    for row in reader:
        tokens = [token.strip() for token in row if token.strip() != ""]
        if tokens:
            return tokens
    return None


def _parse_particle_tokens(tokens: list[str], line_no: int | None = None) -> Tuple[float, float, int]:
    if tokens[0] != "x" or tokens[2] != "y":
        location = f" at line {line_no}" if line_no is not None else ""
        raise ValueError(f"Malformed particle row{location}: {tokens}")

    try:
        pos_x = float(tokens[1])
        pos_y = float(tokens[3])
    except ValueError as exc:
        location = f" at line {line_no}" if line_no is not None else ""
        raise ValueError(f"Invalid coordinate{location}: {tokens}") from exc

    particle_type = 0
    if len(tokens) == 6 and tokens[4] == "type":
        try:
            particle_type = int(float(tokens[5]))
        except ValueError as exc:
            location = f" at line {line_no}" if line_no is not None else ""
            raise ValueError(f"Invalid type{location}: {tokens}") from exc
    elif len(tokens) != 4:
        location = f" at line {line_no}" if line_no is not None else ""
        raise ValueError(f"Malformed particle row{location}: {tokens}")

    return pos_x, pos_y, particle_type


def load_particle_csv(
    path: Union[str, Path], *, return_types: bool = False
) -> Union[Tuple[Metadata, np.ndarray], Tuple[Metadata, np.ndarray, np.ndarray]]:
    """Load metadata + positions from the CSV file emitted by ``print_particles``.

    Parameters
    ----------
    path:
        Path to the CSV snapshot.
    return_types:
        When True, also return the particle-type array read from the CSV (defaults to False).
    """

    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            metadata_row = next(reader)
        except StopIteration as exc:
            raise ValueError("Particle CSV is empty") from exc

        metadata = _parse_metadata_row(metadata_row)

        positions = []
        ptypes = []
        for line_no, row in enumerate(reader, start=2):
            tokens = [token.strip() for token in row if token.strip() != ""]
            if not tokens:
                continue
            pos_x, pos_y, particle_type = _parse_particle_tokens(tokens, line_no)
            positions.append((pos_x, pos_y))
            ptypes.append(particle_type)

    pos_array = np.array(positions, dtype=np.float64)
    if pos_array.size == 0:
        pos_array = np.zeros((0, 2), dtype=np.float64)

    type_array = (
        np.array(ptypes, dtype=np.int32) if ptypes else np.zeros((0,), dtype=np.int32)
    )

    expected = int(metadata.get("n_particles", pos_array.shape[0]))
    if expected != pos_array.shape[0]:
        raise ValueError(
            f"n_particles ({expected}) does not match number of rows ({pos_array.shape[0]})"
        )

    if return_types:
        return metadata, pos_array, type_array
    return metadata, pos_array


def load_triangulation_csv(
    path: Union[str, Path]
) -> Tuple[Metadata, np.ndarray, np.ndarray, np.ndarray]:
    """Load metadata, particles, and triangle mesh from a triangulation CSV."""

    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            metadata_row = next(reader)
        except StopIteration as exc:
            raise ValueError("Triangulation CSV is empty") from exc

        metadata = _parse_metadata_row(metadata_row)

        n_particles = int(metadata.get("n_particles", 0))
        n_triangles = int(metadata.get("n_triangles", 0))

        positions = []
        ptypes = []
        for idx in range(n_particles):
            tokens = _next_tokens(reader)
            if tokens is None:
                raise ValueError(
                    f"Unexpected end of file while reading particle rows (expected {n_particles})"
                )
            pos_x, pos_y, particle_type = _parse_particle_tokens(tokens)
            positions.append((pos_x, pos_y))
            ptypes.append(particle_type)

        triangles: list[list[float]] = []
        for tri_idx in range(n_triangles):
            tokens = _next_tokens(reader)
            if tokens is None:
                raise ValueError(
                    f"Unexpected end of file while reading triangle rows (expected {n_triangles})"
                )
            if (
                len(tokens) != 12
                or tokens[0] != "x0"
                or tokens[2] != "y0"
                or tokens[4] != "x1"
                or tokens[6] != "y1"
                or tokens[8] != "x2"
                or tokens[10] != "y2"
            ):
                raise ValueError(
                    f"Malformed triangle row at index {tri_idx}: {tokens}"
                )

            try:
                tri_vals = [
                    float(tokens[1]),
                    float(tokens[3]),
                    float(tokens[5]),
                    float(tokens[7]),
                    float(tokens[9]),
                    float(tokens[11]),
                ]
            except ValueError as exc:
                raise ValueError(
                    f"Invalid triangle coordinate at index {tri_idx}: {tokens}"
                ) from exc

            triangles.append(tri_vals)

    pos_array = np.array(positions, dtype=np.float64)
    if pos_array.size == 0:
        pos_array = np.zeros((0, 2), dtype=np.float64)

    type_array = (
        np.array(ptypes, dtype=np.int32) if ptypes else np.zeros((0,), dtype=np.int32)
    )

    if not triangles:
        tri_array = np.zeros((0, 3, 2), dtype=np.float64)
    else:
        tri_array = np.array(triangles, dtype=np.float64).reshape(-1, 3, 2)

    return metadata, pos_array, type_array, tri_array


def load_interface_csv(
    path: Union[str, Path]
) -> Tuple[Metadata, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Load metadata, particles, and interface segments from an interface CSV."""

    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            metadata_row = next(reader)
        except StopIteration as exc:
            raise ValueError("Interface CSV is empty") from exc

        metadata = _parse_metadata_row(metadata_row)
        n_particles = int(metadata.get("n_particles", 0))

        positions = []
        ptypes = []
        for idx in range(n_particles):
            tokens = _next_tokens(reader)
            if tokens is None:
                raise ValueError(
                    f"Unexpected end of file while reading particle rows (expected {n_particles})"
                )
            pos_x, pos_y, particle_type = _parse_particle_tokens(tokens)
            positions.append((pos_x, pos_y))
            ptypes.append(particle_type)

        interfaces = []
        while True:
            tokens = _next_tokens(reader)
            if tokens is None:
                break
            
            # Format: iface_idx, I, x1, X1, y1, Y1, x2, X2, y2, Y2
            if len(tokens) != 10 or tokens[0] != "iface_idx":
                # Fallback or strict check; ignoring malformed lines for robustness or raising error
                continue
                
            try:
                seg = {
                    'idx': int(tokens[1]),
                    'x1': float(tokens[3]),
                    'y1': float(tokens[5]),
                    'x2': float(tokens[7]),
                    'y2': float(tokens[9]),
                }
                interfaces.append(seg)
            except ValueError:
                continue

    pos_array = np.array(positions, dtype=np.float64)
    if pos_array.size == 0:
        pos_array = np.zeros((0, 2), dtype=np.float64)

    type_array = (
        np.array(ptypes, dtype=np.int32) if ptypes else np.zeros((0,), dtype=np.int32)
    )

    return metadata, pos_array, type_array, interfaces


def load_ab_network_csv(
    path: Union[str, Path]
) -> Tuple[
    Metadata, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]
]:
    """Load metadata, particles, and AB midpoint networks from a CSV."""

    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            metadata_row = next(reader)
        except StopIteration as exc:
            raise ValueError("AB network CSV is empty") from exc

        metadata = _parse_metadata_row(metadata_row)
        n_particles = int(metadata.get("n_particles", 0))
        n_networks = int(metadata.get("n_networks", 0))

        positions: list[tuple[float, float]] = []
        ptypes: list[int] = []
        for idx in range(n_particles):
            tokens = _next_tokens(reader)
            if tokens is None:
                raise ValueError(
                    f"Unexpected end of file while reading particle rows (expected {n_particles})"
                )
            pos_x, pos_y, particle_type = _parse_particle_tokens(tokens)
            positions.append((pos_x, pos_y))
            ptypes.append(particle_type)

        nodes: list[list[tuple[float, float]]] = [[] for _ in range(n_networks)]
        edges: list[list[tuple[float, float, float, float]]] = [[] for _ in range(n_networks)]

        while True:
            tokens = _next_tokens(reader)
            if tokens is None:
                break
            if len(tokens) < 4 or tokens[0] != "net_idx":
                continue

            try:
                net_idx = int(float(tokens[1]))
            except ValueError:
                continue
            if net_idx < 0:
                continue

            while len(nodes) <= net_idx:
                nodes.append([])
                edges.append([])

            if (
                len(tokens) == 8
                and tokens[2] == "node_idx"
                and tokens[4] == "x"
                and tokens[6] == "y"
            ):
                try:
                    node_x = float(tokens[5])
                    node_y = float(tokens[7])
                except ValueError:
                    continue
                nodes[net_idx].append((node_x, node_y))
            elif (
                len(tokens) == 12
                and tokens[2] == "edge_idx"
                and tokens[4] == "x0"
                and tokens[6] == "y0"
                and tokens[8] == "x1"
                and tokens[10] == "y1"
            ):
                try:
                    x0 = float(tokens[5])
                    y0 = float(tokens[7])
                    x1 = float(tokens[9])
                    y1 = float(tokens[11])
                except ValueError:
                    continue
                edges[net_idx].append((x0, y0, x1, y1))

    pos_array = np.array(positions, dtype=np.float64)
    if pos_array.size == 0:
        pos_array = np.zeros((0, 2), dtype=np.float64)

    type_array = (
        np.array(ptypes, dtype=np.int32) if ptypes else np.zeros((0,), dtype=np.int32)
    )

    node_arrays: List[np.ndarray] = []
    for net_nodes in nodes:
        if net_nodes:
            node_arrays.append(np.array(net_nodes, dtype=np.float64).reshape(-1, 2))
        else:
            node_arrays.append(np.zeros((0, 2), dtype=np.float64))

    edge_arrays: List[np.ndarray] = []
    for net_edges in edges:
        if net_edges:
            edge_arrays.append(
                np.array(net_edges, dtype=np.float64).reshape(-1, 2, 2)
            )
        else:
            edge_arrays.append(np.zeros((0, 2, 2), dtype=np.float64))

    return metadata, pos_array, type_array, node_arrays, edge_arrays


def _setup_plot(
    metadata: Metadata,
    positions: np.ndarray,
    dpi: float,
    l_ref: float,
    strict_limits: bool = False,
):
    if positions.size == 0:
        raise ValueError("Particle CSV contains no particles to plot")

    min_x = float(positions[:, 0].min())
    max_x = float(positions[:, 0].max())
    min_y = float(positions[:, 1].min())
    max_y = float(positions[:, 1].max())

    box_w = float(metadata.get("box_w", max_x - min_x))
    box_h = float(metadata.get("box_h", max_y - min_y))
    if box_w <= 0.0:
        box_w = max(max_x - min_x, 1.0)
    if box_h <= 0.0:
        box_h = max(max_y - min_y, 1.0)

    draw_box = bool(metadata.get("draw_box", False))

    if strict_limits:
        x_left = 0.0
        x_right = box_w
        y_bottom = 0.0
        y_top = box_h
    else:
        x_left = min_x
        x_right = x_left + box_w
        y_bottom = min_y
        y_top = y_bottom + box_h

    fig_width_in = 10.0
    if box_w > l_ref:
        fig_width_in *= box_w / l_ref
        fig_width_in = min(fig_width_in, 50.0)

    fig_height_in = fig_width_in * (box_h / box_w) if box_w > 0 else fig_width_in
    if fig_height_in <= 0.0:
        fig_height_in = fig_width_in

    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=dpi)
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bottom, y_top)

    return fig, ax, fig_width_in, box_w, box_h, draw_box, (x_left, x_right, y_bottom, y_top)


def _scatter_particles(
    ax: plt.Axes,
    positions: np.ndarray,
    types: np.ndarray,
    metadata: Metadata,
    fig_width_in: float,
    box_w: float,
):
    sigma_a = float(metadata.get("sigma_aa", 1.0))
    sigma_b = float(metadata.get("sigma_bb", 1.0))
    width_scale = fig_width_in / max(box_w, 1e-12)

    radius_a_pts = max((sigma_a * 1.12) * width_scale * 72.0, 1.0)
    radius_b_pts = max((sigma_b * 1.12) * width_scale * 72.0, 1.0)
    size_a = radius_a_pts * radius_a_pts
    size_b = radius_b_pts * radius_b_pts

    mask_a = types == 0
    mask_b = ~mask_a

    if np.any(mask_a):
        ax.scatter(
            positions[mask_a, 0],
            positions[mask_a, 1],
            s=size_a,
            facecolors="red",
            edgecolors="black",
            linewidths=0.1,
        )

    if np.any(mask_b):
        ax.scatter(
            positions[mask_b, 0],
            positions[mask_b, 1],
            s=size_b,
            facecolors="blue",
            edgecolors="black",
            linewidths=0.1,
        )


def _finalize_plot(ax: plt.Axes, draw_box: bool, bounds: Tuple[float, float, float, float]):
    x_left, x_right, y_bottom, y_top = bounds
    if draw_box:
        ax.plot(
            [x_left, x_right, x_right, x_left, x_left],
            [y_bottom, y_bottom, y_top, y_top, y_bottom],
            c="black",
            linestyle="--",
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")


def plot_particle_csv(
    csv_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    dpi: float = 100.0,
    l_ref: float = 50.0,
    strict_box_limits: bool = False,
) -> None:
    """Recreate the legacy matplotlibcpp scatter plot from a particle CSV."""

    metadata, positions, types = load_particle_csv(csv_path, return_types=True)
    fig, ax, fig_width_in, box_w, _, draw_box, bounds = _setup_plot(
        metadata, positions, dpi, l_ref, strict_box_limits
    )
    _scatter_particles(ax, positions, types, metadata, fig_width_in, box_w)
    ax.set_title("Particles")
    _finalize_plot(ax, draw_box, bounds)
    fig.tight_layout()

    output_path = Path(output_path)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_triangulation_csv(
    csv_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    dpi: float = 100.0,
    l_ref: float = 50.0,
    strict_box_limits: bool = False,
) -> None:
    """Render triangulation scatter + mesh from a CSV."""

    metadata, positions, types, triangles = load_triangulation_csv(csv_path)
    fig, ax, fig_width_in, box_w, _, draw_box, bounds = _setup_plot(
        metadata, positions, dpi, l_ref, strict_box_limits
    )
    _scatter_particles(ax, positions, types, metadata, fig_width_in, box_w)

    for tri in triangles:
        ax.plot(
            [tri[0, 0], tri[1, 0], tri[2, 0], tri[0, 0]],
            [tri[0, 1], tri[1, 1], tri[2, 1], tri[0, 1]],
            c="black",
            linewidth=0.05,
        )

    ax.set_title("Triangulation mesh")
    _finalize_plot(ax, draw_box, bounds)
    fig.tight_layout()

    output_path = Path(output_path)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_interface_csv(
    csv_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    dpi: float = 100.0,
    l_ref: float = 50.0,
    strict_box_limits: bool = False,
) -> None:
    """Render interface scatter + segments from a CSV."""

    metadata, positions, types, interfaces = load_interface_csv(csv_path)
    fig, ax, fig_width_in, box_w, _, draw_box, bounds = _setup_plot(
        metadata, positions, dpi, l_ref, strict_box_limits
    )
    _scatter_particles(ax, positions, types, metadata, fig_width_in, box_w)

    for seg in interfaces:
        ax.plot(
            [seg['x1'], seg['x2']],
            [seg['y1'], seg['y2']],
            c="black",
            linewidth=2.0,
            alpha=1.0
        )

    ax.set_title("Interfaces")
    _finalize_plot(ax, draw_box, bounds)
    fig.tight_layout()

    output_path = Path(output_path)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_ab_network_csv(
    csv_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    dpi: float = 100.0,
    l_ref: float = 50.0,
    strict_box_limits: bool = False,
) -> None:
    """Render AB midpoint networks and particles from a CSV."""

    metadata, positions, types, nodes, edges = load_ab_network_csv(csv_path)
    fig, ax, fig_width_in, box_w, _, draw_box, bounds = _setup_plot(
        metadata, positions, dpi, l_ref, strict_box_limits
    )
    _scatter_particles(ax, positions, types, metadata, fig_width_in, box_w)

    n_networks = max(len(nodes), len(edges))
    if n_networks > 0:
        colors = plt.get_cmap("tab10")
        for idx in range(n_networks):
            color = colors(idx % 10)
            node_arr = nodes[idx] if idx < len(nodes) else np.zeros((0, 2))
            edge_arr = edges[idx] if idx < len(edges) else np.zeros((0, 2, 2))

            if edge_arr.size > 0:
                for seg in edge_arr:
                    ax.plot(
                        [seg[0, 0], seg[1, 0]],
                        [seg[0, 1], seg[1, 1]],
                        c=color,
                        linewidth=1.0,
                        alpha=0.9,
                    )

            if node_arr.size > 0:
                ax.scatter(
                    node_arr[:, 0],
                    node_arr[:, 1],
                    s=12.0,
                    facecolors=color,
                    edgecolors="black",
                    linewidths=0.2,
                    zorder=3,
                )

    ax.set_title("A–B midpoint networks")
    _finalize_plot(ax, draw_box, bounds)
    fig.tight_layout()

    output_path = Path(output_path)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


__all__ = [
    "load_particle_csv",
    "plot_particle_csv",
    "load_triangulation_csv",
    "plot_triangulation_csv",
    "load_interface_csv",
    "plot_interface_csv",
    "load_ab_network_csv",
    "plot_ab_network_csv",
]
