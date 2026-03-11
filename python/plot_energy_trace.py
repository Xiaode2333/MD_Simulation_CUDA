#!/usr/bin/env python3
"""
Plot energy vs FIRE steps from a CSV emitted by tests/run_test.

Usage:
  python plot_energy_trace.py --csv_path path/to/is_energy_step_XXXX.csv \
                              --figure_path path/to/is_energy_step_XXXX.svg
"""

import argparse
import csv
from pathlib import Path
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def load_energy(csv_path: Path):
    steps = []
    energies = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["fire_step"]))
            energies.append(float(row["energy"]))
    return steps, energies


def plot_energy(steps, energies, figure_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, energies, lw=1.5, color="#0B6E99")
    ax.set_xlabel("FIRE step")
    ax.set_ylabel("Potential energy")
    ax.set_title("Energy vs. FIRE steps")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--figure_path", required=True)
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    fig_path = Path(args.figure_path)

    steps, energies = load_energy(csv_path)
    plot_energy(steps, energies, fig_path)


if __name__ == "__main__":
    main()
