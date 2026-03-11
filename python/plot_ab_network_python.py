#!/usr/bin/env python3
"""CLI wrapper to render AB midpoint network CSV snapshots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from particle_csv import plot_ab_network_csv  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot AB midpoint network CSV snapshots"
    )
    parser.add_argument("--csv_name", required=True, help="AB network CSV path")
    parser.add_argument("--output_name", required=True, help="Output image path")
    parser.add_argument("--dpi", type=float, default=100.0, help="Matplotlib DPI")
    parser.add_argument(
        "--l_ref",
        type=float,
        default=50.0,
        help="Reference box length used to scale figure width",
    )
    parser.add_argument(
        "--strict-box-limits",
        action="store_true",
        help="Force x/y limits to [0, box_w] and [0, box_h] from metadata",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_ab_network_csv(
        csv_path=args.csv_name,
        output_path=args.output_name,
        dpi=args.dpi,
        l_ref=args.l_ref,
        strict_box_limits=args.strict_box_limits,
    )


if __name__ == "__main__":
    main()
