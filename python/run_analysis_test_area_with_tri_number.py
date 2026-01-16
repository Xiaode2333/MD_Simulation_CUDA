#!/usr/bin/env python
"""
Execute the analysis_test_area_with_tri_number.ipynb notebook for a given temperature.

The runner injects T_value_primary and RUN_ROOT_NAME before execution so the notebook
analyzes the chosen temperature without modifying the source notebook in-place.
"""

from __future__ import annotations

import argparse
import nbformat
from nbclient import NotebookClient
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute analysis notebook for a given temperature.")
    parser.add_argument(
        "-T",
        "--temperature",
        type=float,
        required=True,
        help="Temperature value, e.g., 0.5",
    )
    parser.add_argument(
        "-b",
        "--base-root",
        type=Path,
        default=Path("results/20260113_test_area_with_num_tri_types"),
        help="Base results directory containing T_<T> or T_<T>.tar",
    )
    parser.add_argument(
        "-i",
        "--input-nb",
        type=Path,
        default=Path("python/analysis_test_area_with_tri_number.ipynb"),
        help="Path to the source analysis notebook.",
    )
    parser.add_argument(
        "-o",
        "--output-nb",
        type=Path,
        default=None,
        help="Path to write the executed notebook. Defaults to <base-root>/T_<T>_analysis/analysis_test_area_with_tri_number_T_<T>.ipynb",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_root = args.base_root.resolve()
    if not base_root.exists():
        raise FileNotFoundError(f"Base root not found: {base_root}")

    input_nb = args.input_nb.resolve()
    if not input_nb.exists():
        raise FileNotFoundError(f"Input notebook not found: {input_nb}")

    t_label = f"{args.temperature:g}"
    default_out = base_root / f"T_{t_label}_analysis" / f"analysis_test_area_with_tri_number_T_{t_label}.ipynb"
    output_nb = args.output_nb.resolve() if args.output_nb is not None else default_out
    output_nb.parent.mkdir(parents=True, exist_ok=True)

    nb = nbformat.read(input_nb, as_version=4)

    # Inject parameter overrides at the top without altering the source notebook.
    injected_source = [
        "from pathlib import Path",
        "import sys",
        "repo_root = Path('.').resolve()",
        "python_dir = repo_root / 'python'",
        "if str(python_dir) not in sys.path:",
        "    sys.path.insert(0, str(python_dir))",
        f'T_value_primary = {args.temperature}',
        f"RUN_ROOT_NAME = r'{base_root}'",
    ]
    nb.cells.insert(0, nbformat.v4.new_code_cell("\n".join(injected_source)))

    client = NotebookClient(
        nb,
        timeout=None,
        kernel_name="python3",
        resources={"metadata": {"path": str(input_nb.parent.parent)}},  # repo root
    )
    client.execute()

    nbformat.write(nb, output_nb)
    print(f"Executed notebook written to {output_nb}")


if __name__ == "__main__":
    main()
