#!/usr/bin/env python3
"""
Thin out .svg plot files in simulation results to save disk space.

For each subdirectory under a given root (default: ./results), if the directory
contains more than a specified number of .svg files (default: 20), this script
deletes most of them and keeps only the requested number of files. The kept
files are chosen to be evenly distributed in the sorted name order.

Example:
    If a folder contains 1000 plots named like "*_step={i}.svg" (i=1..1000),
    then with --keep 20 we will keep 20 files with indices spaced roughly
    uniformly between the first and last in the filename-sorted order.

Run from the project root, e.g.:

    ~/.conda/envs/py3/bin/python python/thin_svg_plots.py --root results --keep 20

Use --dry-run to see what would be deleted without actually removing files.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Set


def choose_evenly_spaced(items: List[Path], keep: int) -> Set[Path]:
    """
    Given a non-empty list of Paths sorted by name and a keep count (keep >= 1),
    return a set of items to keep, spaced as evenly as possible across the list.
    """
    n = len(items)
    if keep >= n:
        return set(items)

    if keep == 1:
        # Pick the middle item.
        return {items[n // 2]}

    # Evenly spaced indices from 0 to n-1 inclusive, similar to numpy.linspace.
    indices = [
        int(round(i * (n - 1) / (keep - 1)))
        for i in range(keep)
    ]

    # In theory, with n >= keep and the formula above, indices should be unique,
    # but we guard just in case.
    unique_indices = sorted(set(indices))
    selected = {items[idx] for idx in unique_indices}

    # If for some pathological reason we ended up with fewer than `keep` items,
    # fill in from neighbors.
    if len(selected) < keep:
        for idx in range(n):
            if items[idx] not in selected:
                selected.add(items[idx])
                if len(selected) == keep:
                    break

    return selected


def thin_svgs_in_dir(
    directory: Path,
    keep: int,
    dry_run: bool = False,
    verbose: bool = True,
) -> None:
    """
    In a single directory, keep at most `keep` .svg files, deleting the rest.
    Files kept are evenly distributed in the filename-sorted order.
    """
    svg_files = sorted(
        (directory / name for name in os.listdir(directory)),
        key=lambda p: p.name,
    )
    svg_files = [p for p in svg_files if p.is_file() and p.suffix.lower() == ".svg"]

    total = len(svg_files)
    if total <= keep:
        return

    to_keep = choose_evenly_spaced(svg_files, keep)
    to_delete = [p for p in svg_files if p not in to_keep]

    if verbose:
        print(f"[{directory}] total .svg files: {total}, keeping: {keep}, deleting: {len(to_delete)}")

    for path in to_delete:
        if verbose:
            print(f"  DELETE: {path}")
        if not dry_run:
            try:
                path.unlink()
            except OSError as exc:
                print(f"    Failed to delete {path}: {exc}")


def thin_svgs_recursive(
    root: Path,
    keep: int,
    dry_run: bool = False,
    verbose: bool = True,
) -> None:
    """
    Walk `root` recursively, thinning .svg files in every subdirectory.
    """
    if not root.is_dir():
        print(f"Root path {root} is not a directory; nothing to do.")
        return

    for dirpath, dirnames, filenames in os.walk(root):
        directory = Path(dirpath)
        # Quick check: only bother if there is at least one .svg file.
        if not any(name.lower().endswith(".svg") for name in filenames):
            continue
        thin_svgs_in_dir(directory, keep=keep, dry_run=dry_run, verbose=verbose)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Thin .svg plots under a results/ directory to save space.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results"),
        help="Root directory to scan recursively (default: ./results).",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=20,
        help="Maximum number of .svg files to keep in each directory (default: 20).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually removing any files.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.keep <= 0:
        print("Nothing to keep (--keep must be positive). Exiting.")
        return

    verbose = not args.quiet
    thin_svgs_recursive(
        root=args.root,
        keep=args.keep,
        dry_run=args.dry_run,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()

