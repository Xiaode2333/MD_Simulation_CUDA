#!/usr/bin/env python3
"""Plot C(q) = k_BT / (L S(q)) versus q^2 from capillary wave analysis CSV lines."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def _parse_row(tokens: List[str]) -> Dict[str, float]:
    if len(tokens) % 2 != 0:
        raise ValueError(f"Malformed row with tokens: {tokens}")
    data: Dict[str, float] = {}
    for key, value in zip(tokens[::2], tokens[1::2]):
        if key == "step":
            data[key] = float(value)
        else:
            data[key] = float(value)
    return data


def load_latest_entry(path: Path) -> Dict[str, float]:
    rows: List[Dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            tokens = [token.strip() for token in row if token.strip()]
            if not tokens:
                continue
            rows.append(_parse_row(tokens))
    if not rows:
        raise ValueError(f"No data rows found in {path}")
    return rows[-1]


def _extract_inverse_spectrum(data: Dict[str, float]) -> tuple[list[int], list[float], list[float]]:
    """Return modes, q^2, and C(q) if present in the newer format."""
    q_prefix = "q_sq_"
    c_prefix = "Cq_"
    q_modes = {
        int(key[len(q_prefix) :]) for key in data if key.startswith(q_prefix)
    }
    c_modes = {
        int(key[len(c_prefix) :]) for key in data if key.startswith(c_prefix)
    }
    indices = sorted(q_modes & c_modes)
    if not indices:
        return [], [], []
    q_vals = [data[f"{q_prefix}{idx}"] for idx in indices]
    c_vals = [data[f"{c_prefix}{idx}"] for idx in indices]
    return indices, q_vals, c_vals


def _extract_legacy_spectrum(data: Dict[str, float]) -> tuple[list[int], list[float], list[float]]:
    """Fallback for legacy rows containing inv_q_sq_* and hq*_sq."""
    inv_prefix = "inv_q_sq_"
    h_prefix = "hq"
    indices = sorted(
        {
            int(key[len(inv_prefix) :])
            for key in data
            if key.startswith(inv_prefix)
        }
        & {
            int(key[len(h_prefix) : key.find("_sq")])
            for key in data
            if key.startswith(h_prefix) and key.endswith("_sq")
        }
    )
    if not indices:
        return [], [], []
    q_vals = [data[f"{inv_prefix}{idx}"] for idx in indices]
    c_vals = [data[f"{h_prefix}{idx}_sq"] for idx in indices]
    return indices, q_vals, c_vals


def plot_entry(data: Dict[str, float], output: Path) -> None:
    indices, x_vals, y_vals = _extract_inverse_spectrum(data)
    y_label = r"$C(q) = k_B T / (L S(q))$"
    x_label = r"$q^2$"
    title_gamma = data.get("gamma", float("nan"))
    title_c0 = data.get("C0", float("nan"))

    if not indices:
        # Try legacy format for backward compatibility
        indices, x_vals, y_vals = _extract_legacy_spectrum(data)
        y_label = r"$|h_q|^2$"
        x_label = r"$1 / q^2$"
        title_c0 = float("nan")

    if not indices:
        raise ValueError("No q-mode entries found in data row")

    step = int(data.get("step", 0.0))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_vals, y_vals, marker="o", linestyle="", label="data")

    if x_vals and y_vals and x_label == r"$q^2$" and math.isfinite(title_gamma):
        x_min, x_max = min(x_vals), max(x_vals)
        x_line = [x_min, x_max]
        y_line = [title_gamma * x + title_c0 for x in x_line]
        ax.plot(x_line, y_line, linestyle="-", color="tab:orange", label="fit")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"Step {step}, gamma={title_gamma:.4f}, C0={title_c0:.3e}")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot capillary wave data from CSV.")
    parser.add_argument("--csv_path", required=True, help="Input CSV file with CWA rows.")
    parser.add_argument("--output", required=True, help="Output image file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    output_path = Path(args.output)
    data = load_latest_entry(csv_path)
    plot_entry(data, output_path)


if __name__ == "__main__":
    main()
