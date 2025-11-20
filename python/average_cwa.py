#!/usr/bin/env python3
"""Average CWA snapshots and plot aggregated C(q) fit and gamma time series.

Reads a CSV like tests/cwa_test/sample_csv/cwa_instant.csv (rows of key,value pairs),
computes mode-wise averages of S(q)=|h_q|^2, reconstructs C(q)=k_B T/(L_y S(q))
from the averaged S(q) (instead of averaging C(q) directly), performs a linear
regression of this reconstructed C(q) vs q^2 to estimate gamma, and plots both
the averaged fit and the per-step gamma values.
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
    """Linear regression with intercept forced to zero.

    Fits y ≈ slope * x, i.e., the regression line is constrained to pass
    through (0, 0). Returns (slope, intercept) where intercept is always 0.
    """
    valid = [
        (xi, yi)
        for xi, yi in zip(x, y)
        if math.isfinite(xi) and math.isfinite(yi)
    ]
    if len(valid) < 1:
        return float("nan"), float("nan")
    num = sum(xi * yi for xi, yi in valid)
    den = sum(xi * xi for xi, _ in valid)
    if den <= 1e-16:
        return float("nan"), float("nan")
    slope = num / den
    intercept = 0.0
    return slope, intercept


def estimate_prefactor(rows: List[Dict[str, float]]) -> float:
    """Estimate k_B T / L_y from instantaneous C(q) and S(q) data.

    From the simulation, C(q) is computed as C(q) = T / (L_y * S(q)) (k_B=1),
    so T / L_y = C(q) * S(q). We average C(q) * S(q) over all available
    modes and snapshots to get a robust estimate of this prefactor.
    """
    prefactor_sum = 0.0
    prefactor_count = 0.0
    for row in rows:
        for key, c_val in row.items():
            if not key.startswith("Cq_"):
                continue
            if not math.isfinite(c_val):
                continue
            mode = int(key[len("Cq_") :])
            s_key = f"S_q_{mode}"
            h_key = f"hq_sq_{mode}"
            s_val = row.get(s_key)
            if s_val is None:
                s_val = row.get(h_key)
            if s_val is None or not math.isfinite(s_val) or s_val <= 0.0:
                continue
            prefactor_sum += c_val * s_val
            prefactor_count += 1.0
    if prefactor_count <= 0.0:
        return float("nan")
    return prefactor_sum / prefactor_count


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

    ax_left.plot(q2, avg_cq, "o", label="C(q) from avg S(q)")
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
        ax_right.axhline(
            gamma_avg,
            color="tab:orange",
            linestyle="--",
            label="C(q) from avg S(q) fit",
        )
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
        print(f"Average gamma from C(q) (via avg S(q)) fit: {gamma_avg:.6f}, C0={c0_avg:.6e}")
    else:
        print("Unable to determine average gamma (insufficient data)")

    for mode, q2_val, hq2_val, cq_val in zip(modes, q2, avg_hq2, avg_cq):
        print(
            f"mode {mode:2d}: q^2={q2_val:.6f}, avg |h_q|^2={hq2_val:.6e}, C(q) from avg S(q)={cq_val:.6e}"
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
    prefactor = estimate_prefactor(rows)
    modes, q2_vals, avg_hq2, _avg_cq_raw = aggregate_modes(rows)
    if not modes:
        raise ValueError("No modes found in input for averaging")

    # Reconstruct C(q) from the averaged S(q) using C(q) = (T / L_y) / <S(q)>,
    # where T / L_y is estimated from instantaneous C(q) and S(q).
    if math.isfinite(prefactor):
        avg_cq_from_s = [
            (prefactor / hq2 if math.isfinite(hq2) and hq2 > 0.0 else float("nan"))
            for hq2 in avg_hq2
        ]
    else:
        avg_cq_from_s = [float("nan")] * len(avg_hq2)

    plot_average_cwa(
        input_path,
        output_svg_path,
        output_csv_path,
        rows,
        modes,
        q2_vals,
        avg_hq2,
        avg_cq_from_s,
    )


if __name__ == "__main__":
    main()
