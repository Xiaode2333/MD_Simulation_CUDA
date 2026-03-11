#!/usr/bin/env python3
"""
Delete all *.bin files recursively under a given root directory.

By default, starts from the current working directory.

Examples (run from project root):

  # Dry-run: show which .bin files would be removed
  ~/.conda/envs/py3/bin/python python/delete_bin_files.py --dry-run

  # Actually delete all .bin files under the current directory
  ~/.conda/envs/py3/bin/python python/delete_bin_files.py

  # Only clean within the results/ tree
  ~/.conda/envs/py3/bin/python python/delete_bin_files.py --root results
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively delete *.bin files under a root directory.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("./results"),
        help="Root directory to scan recursively (default: ./results).",
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


def delete_bin_files(root: Path, dry_run: bool = False, verbose: bool = True) -> None:
    if not root.is_dir():
        print(f"Root path {root} is not a directory; nothing to do.")
        return

    removed_count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        directory = Path(dirpath)
        for name in filenames:
            if not name.lower().endswith(".bin"):
                continue
            path = directory / name
            if verbose:
                print(("DELETE" if not dry_run else "WOULD DELETE") + f": {path}")
            if not dry_run:
                try:
                    path.unlink()
                    removed_count += 1
                except OSError as exc:
                    print(f"  Failed to delete {path}: {exc}")

    if verbose:
        if dry_run:
            print(f"Dry run complete under {root}.")
        else:
            print(f"Deleted {removed_count} *.bin files under {root}.")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    verbose = not args.quiet
    delete_bin_files(root=args.root, dry_run=args.dry_run, verbose=verbose)


if __name__ == "__main__":
    main()

