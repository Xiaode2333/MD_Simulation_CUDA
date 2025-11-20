#!/usr/bin/env python3
"""Average CWA snapshots and plot aggregated C(q) fit and gamma time series.

Reads a CSV like tests/cwa_test/sample_csv/cwa_instant.csv (rows of key,value pairs),
computes mode-wise averages of C(q) and |h_q|^2, performs a linear regression of
average C(q) vs q^2 to estimate gamma, and plots both the averaged fit and the
per-step gamma values.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def parse_row(tokens: List[str]) -> Dict[str, float]:
    if len(tokens) % 2 != 0:
        raise ValueError(f"Malformed row with tokens: {tokens}")
    data: Dict[str, float] = {}
    for key, value in zip(tokens[::2], tokens[1::2]):
        data[key] = float(value)
    return data


def load_rows(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            tokens = [token.strip() for token in row if token.strip()]
            if not tokens:
                continue
            rows.append(parse_row(tokens))
    if not rows:
        raise ValueError(f"No data rows found in {path}")
    return rows


def linear_regression(x: List[float], y: List[float]) -> Tuple[float, float]:
    valid = [
        (xi, yi)
        for xi, yi in zip(x, y)
        if math.isfinite(xi) and math.isfinite(yi)
    ]
    if len(valid) < 2:
        return float("nan"), float("nan")
    xs, ys = zip(*valid)
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in valid)
    den = sum((xi - mean_x) ** 2 for xi, _ in valid)
    if den <= 1e-16:
        return float("nan"), float("nan")
    slope = num / den
    intercept = mean_y - slope * mean_x
    return slope, intercept


def aggregate_modes(rows: List[Dict[str, float]]):
    mode_data: Dict[int, Dict[str, float]] = {}
    for row in rows:
        for key, val in row.items():
            if not key.startswith("q_sq_"):
                continue
            mode = int(key[len("q_sq_") :])
            entry = mode_data.setdefault(
                mode,
                {
                    "q2": float("nan"),
                    "hq2_sum": 0.0,
                    "hq2_count": 0.0,
                    "cq_sum": 0.0,
                    "cq_count": 0.0,
                },
            )
            if math.isfinite(val):
                entry["q2"] = val
            h_key = f"hq_sq_{mode}"
            s_key = f"S_q_{mode}"
            for hq_key in (h_key, s_key):
                h_val = row.get(hq_key)
                if h_val is not None and math.isfinite(h_val) and h_val > 0.0:
                    entry["hq2_sum"] += h_val
                    entry["hq2_count"] += 1.0
                    break
            c_key = f"Cq_{mode}"
            c_val = row.get(c_key)
            if c_val is not None and math.isfinite(c_val):
                entry["cq_sum"] += c_val
                entry["cq_count"] += 1.0
    modes = sorted(mode_data)
    q2_vals: List[float] = []
    avg_hq2: List[float] = []
    avg_cq: List[float] = []
    for mode in modes:
        entry = mode_data[mode]
        q2_vals.append(entry["q2"])
        avg_hq2.append(
            entry["hq2_sum"] / entry["hq2_count"] if entry["hq2_count"] > 0 else float("nan")
        )
        avg_cq.append(
            entry["cq_sum"] / entry["cq_count"] if entry["cq_count"] > 0 else float("nan")
        )
    return modes, q2_vals, avg_hq2, avg_cq


def plot_average_cwa(
    input_csv: Path,
    output_svg: Path,
    output_csv: Path,
    rows: List[Dict[str, float]],
    modes: List[int],
    q2: List[float],
    avg_hq2: List[float],
    avg_cq: List[float],
) -> None:
    gamma_avg, c0_avg = linear_regression(q2, avg_cq)
    steps = [row.get("step", float("nan")) for row in rows]
    gammas = [row.get("gamma", float("nan")) for row in rows]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4.5))

    ax_left.plot(q2, avg_cq, "o", label="avg C(q)")
    if math.isfinite(gamma_avg):
        x_min, x_max = min(q2), max(q2)
        x_line = [x_min, x_max]
        y_line = [gamma_avg * x + (c0_avg if math.isfinite(c0_avg) else 0.0) for x in x_line]
        ax_left.plot(x_line, y_line, "-", label=f"fit (gamma={gamma_avg:.4f})")
    ax_left.set_xlabel(r"$q^2$")
    ax_left.set_ylabel(r"$C(q)$")
    ax_left.set_title("Averaged C(q) vs $q^2$")
    ax_left.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax_left.legend()

    ax_right.plot(steps, gammas, marker="o", linestyle="-", label="gamma per step")
    if math.isfinite(gamma_avg):
        ax_right.axhline(gamma_avg, color="tab:orange", linestyle="--", label="avg C(q) fit gamma")
    ax_right.set_xlabel("step")
    ax_right.set_ylabel(r"$\gamma$")
    ax_right.set_title("Gamma time series")
    ax_right.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax_right.legend()

    fig.suptitle(f"Average CWA from {input_csv}")
    fig.tight_layout()
    fig.savefig(output_svg, dpi=150)
    plt.close(fig)

    if math.isfinite(gamma_avg):
        print(f"Average gamma from C(q) fit: {gamma_avg:.6f}, C0={c0_avg:.6e}")
    else:
        print("Unable to determine average gamma (insufficient data)")

    for mode, q2_val, hq2_val, cq_val in zip(modes, q2, avg_hq2, avg_cq):
        print(
            f"mode {mode:2d}: q^2={q2_val:.6f}, avg |h_q|^2={hq2_val:.6e}, avg C(q)={cq_val:.6e}"
        )

    # Write summary CSV for downstream analysis
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8") as handle:
        handle.write("index,value\n")
        handle.write(f"gamma,{gamma_avg}\n")
        handle.write(f"C0,{c0_avg}\n")
        handle.write("mode,q_sq,avg_hq_sq,avg_Cq\n")
        for mode, q2_val, hq2_val, cq_val in zip(modes, q2, avg_hq2, avg_cq):
            handle.write(f"{mode},{q2_val},{hq2_val},{cq_val}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Average CWA snapshots and plot results.")
    parser.add_argument(
        "--input",
        default="./tests/cwa_test/sample_csv/cwa_instant.csv",
        help="Input CSV with append-only CWA rows (default: tests/cwa_test/sample_csv/cwa_instant.csv)",
    )
    parser.add_argument(
        "--out-svg-path",
        default="./tests/cwa_test/sample_csv/average_cwa.svg",
        help="Output plot file (default: tests/cwa_test/sample_csv/average_cwa.svg)",
    )
    parser.add_argument(
        "--out-csv-path",
        default="./tests/cwa_test/sample_csv/average_cwa.csv",
        help="Output CSV summary (default: tests/cwa_test/sample_csv/average_cwa.csv)",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_svg_path = Path(args.out_svg_path).expanduser()
    output_csv_path = Path(args.out_csv_path).expanduser()

    rows = load_rows(input_path)
    modes, q2_vals, avg_hq2, avg_cq = aggregate_modes(rows)
    if not modes:
        raise ValueError("No modes found in input for averaging")

    plot_average_cwa(input_path, output_svg_path, output_csv_path, rows, modes, q2_vals, avg_hq2, avg_cq)


if __name__ == "__main__":
    main()
