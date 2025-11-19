#!/usr/bin/env python3
"""CLI wrapper that renders interface CSV snapshots via matplotlib."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from particle_csv import plot_interface_csv  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MD Interface CSV snapshots")
    parser.add_argument("--filename", required=True, help="Output image path (PNG)")
    parser.add_argument("--csv_path", required=True, help="Interface CSV path")
    parser.add_argument("--dpi", type=float, default=100.0, help="Matplotlib DPI")
    parser.add_argument(
        "--l_ref",
        type=float,
        default=50.0,
        help="Reference box length used to scale figure width",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_interface_csv(
        csv_path=args.csv_path,
        output_path=args.filename,
        dpi=args.dpi,
        l_ref=args.l_ref,
    )


if __name__ == "__main__":
    main()