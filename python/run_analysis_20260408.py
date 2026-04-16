#!/usr/bin/env python3
"""
Execute python/analysis/analysis_20260408.ipynb and save an executed copy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute analysis_20260408.ipynb.")
    parser.add_argument(
        "-i",
        "--input-nb",
        type=Path,
        default=Path("python/analysis/analysis_20260408.ipynb"),
        help="Path to the source notebook.",
    )
    parser.add_argument(
        "-o",
        "--output-nb",
        type=Path,
        default=Path("results/analysis_20260408_local_rho/analysis_20260408.executed.ipynb"),
        help="Path to write the executed notebook.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=None,
        help="Override N_BINS inside the notebook before the analysis driver runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_nb = args.input_nb.resolve()
    if not input_nb.exists():
        raise FileNotFoundError(f"Input notebook not found: {input_nb}")

    output_nb = args.output_nb.resolve()
    output_nb.parent.mkdir(parents=True, exist_ok=True)

    nb = nbformat.read(input_nb, as_version=4)
    if args.n_bins is not None:
        if args.n_bins <= 0:
            raise ValueError("--n-bins must be positive")

        override_source = "\n".join(
            [
                f"N_BINS = {args.n_bins}",
                "RHO_BIN_EDGES = np.linspace(-1.0, 1.0, N_BINS + 1, dtype=np.float64)",
                "print(f'Overriding N_BINS to {N_BINS}')",
            ]
        )
        nb.cells.insert(2, nbformat.v4.new_code_cell(override_source, id="override-n-bins"))

    client = NotebookClient(
        nb,
        timeout=None,
        kernel_name="python3",
        resources={"metadata": {"path": str(input_nb.parent.parent.parent)}},
    )

    exec_error = None
    try:
        nb = client.execute()
    except Exception as exc:
        exec_error = exc
    finally:
        tmp_output = output_nb.with_suffix(output_nb.suffix + ".tmp")
        nbformat.write(nb, tmp_output)
        tmp_output.replace(output_nb)
        print(f"Executed notebook written to {output_nb}")

    if exec_error is not None:
        raise exec_error


if __name__ == "__main__":
    main()
