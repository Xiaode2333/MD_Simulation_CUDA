#!/usr/bin/env python3
"""Compute entropy and free energy from triangulation CSVs across lambda directories."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from particle_csv import load_triangulation_csv

k_B = 1.0
step_min_entropy = 100000
DEFAULT_RESULTS_ROOT = Path("results/20260108_test_area_with_num_tri_types")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Entropy & free energy with triangulation correlations "
            "(self-contained, per-frame)"
        )
    )
    parser.add_argument(
        "-T",
        "--temperature",
        type=float,
        required=True,
        help="Temperature to analyze (same units as k_B).",
    )
    parser.add_argument(
        "-b",
        "--base-dir",
        type=Path,
        help=(
            "Base directory containing lambda_* subdirectories. "
            "Defaults to searching results/20260108_test_area_with_num_tri_types/"
            "T_<temperature> upward from the current working directory."
        ),
    )
    return parser.parse_args()


def locate_base_dir(T_value: float, base_dir_arg: Path | None) -> Path:
    """Resolve the base directory containing lambda_* folders for a temperature."""
    if base_dir_arg is not None:
        base_dir = base_dir_arg.expanduser()
        if not base_dir.exists():
            raise RuntimeError(f"Provided base_dir does not exist: {base_dir}")
        return base_dir

    target_rel = DEFAULT_RESULTS_ROOT / f"T_{T_value}"
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        candidate = parent / target_rel
        if candidate.exists():
            return candidate

    raise RuntimeError(f"Could not find base_dir for {target_rel} starting from {cwd}")


def parse_kv_csv(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",") if p.strip()]
            if len(parts) < 2:
                continue
            mapping = {}
            for i in range(0, len(parts) - 1, 2):
                key = parts[i]
                if key:
                    mapping[key] = parts[i + 1]
            if mapping:
                rows.append(mapping)
    return pd.DataFrame(rows)


def masked_mean(series):
    if series is None:
        return np.nan
    arr = pd.to_numeric(series, errors="coerce").to_numpy()
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else np.nan


def vertex_key(x: float, y: float, Lx: float, Ly: float, ndigits: int = 6):
    return (round(np.mod(x, Lx), ndigits), round(np.mod(y, Ly), ndigits))


def compute_entropy_from_triangulation(csv_path: Path) -> tuple[int, float] | None:
    metadata, positions, types, tri_coords = load_triangulation_csv(csv_path)
    if tri_coords.size == 0 or positions.size == 0:
        return None

    Lx = float(metadata.get("box_w", 0.0))
    Ly = float(metadata.get("box_h", 0.0))
    if Lx <= 0 or Ly <= 0:
        return None

    # Map wrapped vertex coords to particle types
    pt_map = {vertex_key(x, y, Lx, Ly): int(t) for (x, y), t in zip(positions, types)}

    tri_counts = {"AAA": 0, "AAB": 0, "ABB": 0, "BBB": 0}
    edge_to_types: dict[tuple[tuple[float, float], tuple[float, float]], list[str]] = {}

    for tri in tri_coords:
        verts = [vertex_key(tri[i, 0], tri[i, 1], Lx, Ly) for i in range(3)]
        try:
            v_types = [pt_map[v] for v in verts]
        except KeyError:
            continue  # skip if any vertex type is missing

        nA = sum(1 for t in v_types if t == 0)
        nB = sum(1 for t in v_types if t == 1)
        if nA + nB != 3:
            continue
        if nA == 3:
            tri_type = "AAA"
        elif nB == 3:
            tri_type = "BBB"
        elif nA == 2:
            tri_type = "AAB"
        elif nB == 2:
            tri_type = "ABB"
        else:
            continue

        tri_counts[tri_type] += 1

        edges = [(verts[0], verts[1]), (verts[1], verts[2]), (verts[2], verts[0])]
        for a, b in edges:
            edge_key = tuple(sorted((a, b)))
            edge_to_types.setdefault(edge_key, []).append(tri_type)

    total_tri = sum(tri_counts.values())
    if total_tri == 0:
        return None

    p_x = {k: v / total_tri for k, v in tri_counts.items() if v > 0}

    edge_counts: dict[tuple[str, str], int] = {}
    for types in edge_to_types.values():
        if len(types) < 2:
            continue
        seen_pairs = set()
        for i in range(len(types)):
            for j in range(i + 1, len(types)):
                pair = tuple(sorted((types[i], types[j])))
                seen_pairs.add(pair)
        for pair in seen_pairs:
            edge_counts[pair] = edge_counts.get(pair, 0) + 1

    total_edges = sum(edge_counts.values())

    S1 = -sum(p * np.log(p) for p in p_x.values())
    S2 = 0.0
    if total_edges > 0:
        for (a, b), cnt in edge_counts.items():
            pxy = cnt / total_edges
            pa = p_x.get(a, 0.0)
            pb = p_x.get(b, 0.0)
            if pxy > 0 and pa > 0 and pb > 0:
                S2 += pxy * np.log(pxy / (pa * pb))

    S_val = k_B * (S1 - S2)

    # Extract step from filename (e.g., triangulation_step_250000.csv)
    try:
        step_val = int(csv_path.stem.split("_")[-1])
    except Exception:
        step_val = -1

    return step_val, S_val


def main() -> None:
    args = parse_args()
    T_entropy = args.temperature
    base_dir = locate_base_dir(T_entropy, args.base_dir)

    output_dir = base_dir / "data_analysis"
    output_dir.mkdir(exist_ok=True)

    records = []
    for lambda_dir in sorted(base_dir.glob("lambda_*/")):
        try:
            lam_val = float(lambda_dir.name.split("lambda_")[1])
        except Exception:
            continue

        # --- U and Ly per frame ---
        df_uk = pd.DataFrame()
        uk_path = lambda_dir / "sample_csv" / "U_K_tot_log.csv"
        if uk_path.exists():
            df_uk = parse_kv_csv(uk_path)
            if not df_uk.empty:
                df_uk["step"] = pd.to_numeric(df_uk.get("step"), errors="coerce")
                df_uk["U_tot"] = pd.to_numeric(df_uk.get("U_tot"), errors="coerce")
                df_uk["Ly"] = pd.to_numeric(df_uk.get("Ly"), errors="coerce")
                df_uk = df_uk.dropna(subset=["step"])
                df_uk = df_uk[df_uk["step"] > step_min_entropy]

        # --- AB pair length per frame ---
        df_ab = pd.DataFrame()
        ab_path = lambda_dir / "sample_csv" / "ab_pair_length_log.csv"
        if ab_path.exists():
            df_ab = parse_kv_csv(ab_path)
            if not df_ab.empty:
                df_ab["step"] = pd.to_numeric(df_ab.get("step"), errors="coerce")
                df_ab["AB_pair_length"] = pd.to_numeric(
                    df_ab.get("AB_pair_length"), errors="coerce"
                )
                df_ab = df_ab.dropna(subset=["step"])
                df_ab = df_ab[df_ab["step"] > step_min_entropy]

        # --- Entropy per frame from triangulation CSV ---
        df_S = []
        tri_dir = lambda_dir / "triangulation" / "csv"
        if tri_dir.exists():
            for csv_path in sorted(tri_dir.glob("triangulation_step_*.csv")):
                res = compute_entropy_from_triangulation(csv_path)
                if res is None:
                    continue
                step_val, S_val = res
                if step_val > step_min_entropy:
                    df_S.append({"step": step_val, "S": S_val})
        df_S = pd.DataFrame(df_S)

        # --- Merge per-frame data on common steps ---
        df_frame = df_S.copy()
        if not df_uk.empty:
            df_frame = (
                pd.merge(df_frame, df_uk[["step", "U_tot", "Ly"]], on="step", how="inner")
                if not df_frame.empty
                else df_uk[["step", "U_tot", "Ly"]]
            )
        if not df_ab.empty:
            df_frame = (
                pd.merge(
                    df_frame, df_ab[["step", "AB_pair_length"]], on="step", how="inner"
                )
                if not df_frame.empty
                else df_ab[["step", "AB_pair_length"]]
            )

        if not df_frame.empty and "S" in df_frame and "U_tot" in df_frame:
            df_frame["F"] = df_frame["U_tot"] - T_entropy * df_frame["S"]
            S_mean = masked_mean(df_frame.get("S"))
            U_mean = masked_mean(df_frame.get("U_tot"))
            F_mean = masked_mean(df_frame.get("F"))
            Ly_mean = masked_mean(df_frame.get("Ly"))
            AB_mean = masked_mean(df_frame.get("AB_pair_length"))
            n_frames = len(df_frame)
        else:
            S_mean = U_mean = F_mean = Ly_mean = AB_mean = np.nan
            n_frames = 0

        records.append(
            {
                "lambda_deform": lam_val,
                "n_frames": n_frames,
                "Ly_mean": Ly_mean,
                "entropy_S_mean": S_mean,
                "U_mean": U_mean,
                "F_mean": F_mean,
                "AB_pair_length_mean": AB_mean,
            }
        )

    entropy_df = (
        pd.DataFrame(records).sort_values("lambda_deform").reset_index(drop=True)
    )
    entropy_df.to_csv(
        output_dir / "entropy_free_energy_over_lambda_joint.csv", index=False
    )
    print(
        "Saved entropy/free energy summary to",
        output_dir / "entropy_free_energy_over_lambda_joint.csv",
    )

    # Plot S vs Ly
    mask_S = np.isfinite(entropy_df["entropy_S_mean"]) & np.isfinite(
        entropy_df["Ly_mean"]
    )
    fig_s, ax_s = plt.subplots()
    ax_s.plot(entropy_df.loc[mask_S, "Ly_mean"], entropy_df.loc[mask_S, "entropy_S_mean"], "o-")
    ax_s.set_xlabel("Ly_mean")
    ax_s.set_ylabel("S_mean (per-frame, with correlations)")
    ax_s.set_title("Entropy vs Ly (joint)")
    fig_s.tight_layout()
    fig_s.savefig(output_dir / "fig_entropy_vs_Ly_joint.png", dpi=300)

    # Plot F vs Ly
    mask_F_Ly = np.isfinite(entropy_df["F_mean"]) & np.isfinite(entropy_df["Ly_mean"])
    fig_fly, ax_fly = plt.subplots()
    ax_fly.plot(
        entropy_df.loc[mask_F_Ly, "Ly_mean"],
        entropy_df.loc[mask_F_Ly, "F_mean"],
        "o-",
        label="data",
    )
    ax_fly.set_xlabel("Ly_mean")
    ax_fly.set_ylabel("F_mean = <U - T*S>")
    ax_fly.set_title("Free energy vs Ly (joint)")

    # Plot F vs AB pair length (scatter only)
    mask_F_ab = np.isfinite(entropy_df["F_mean"]) & np.isfinite(
        entropy_df["AB_pair_length_mean"]
    )
    fig_fab, ax_fab = plt.subplots()
    ax_fab.scatter(
        entropy_df.loc[mask_F_ab, "AB_pair_length_mean"],
        entropy_df.loc[mask_F_ab, "F_mean"],
        label="data",
    )
    ax_fab.set_xlabel("<AB_pair_length> (per-frame mean)")
    ax_fab.set_ylabel("F_mean = <U - T*S>")
    ax_fab.set_title("Free energy vs AB pair length (joint)")

    # Linear regression for gamma (Ly)
    slope_Ly = intercept_Ly = gamma_Ly = np.nan
    if mask_F_Ly.sum() >= 2:
        slope_Ly, intercept_Ly = np.polyfit(
            entropy_df.loc[mask_F_Ly, "Ly_mean"], entropy_df.loc[mask_F_Ly, "F_mean"], 1
        )
        gamma_Ly = slope_Ly / 2.0
        x_line = np.linspace(
            entropy_df.loc[mask_F_Ly, "Ly_mean"].min(),
            entropy_df.loc[mask_F_Ly, "Ly_mean"].max(),
            100,
        )
        y_line = slope_Ly * x_line + intercept_Ly
        ax_fly.plot(x_line, y_line, "r--", label="linear fit")
        ax_fly.legend()
        print(
            f"Linear fit F vs Ly (joint): slope={slope_Ly:.6g}, intercept={intercept_Ly:.6g}, gamma={gamma_Ly:.6g}"
        )

    # Linear regression for gamma (AB pair length)
    slope_AB = intercept_AB = gamma_AB = np.nan
    if mask_F_ab.sum() >= 2:
        slope_AB, intercept_AB = np.polyfit(
            entropy_df.loc[mask_F_ab, "AB_pair_length_mean"],
            entropy_df.loc[mask_F_ab, "F_mean"],
            1,
        )
        gamma_AB = slope_AB
        x_ab_line = np.linspace(
            entropy_df.loc[mask_F_ab, "AB_pair_length_mean"].min(),
            entropy_df.loc[mask_F_ab, "AB_pair_length_mean"].max(),
            100,
        )
        y_ab_line = slope_AB * x_ab_line + intercept_AB
        ax_fab.plot(x_ab_line, y_ab_line, "r--", label="linear fit")
        ax_fab.legend()
        print(
            f"Linear fit F vs <AB_pair_length> (joint): slope={slope_AB:.6g}, intercept={intercept_AB:.6g}, gamma={gamma_AB:.6g}"
        )

    fig_fly.tight_layout()
    fig_fly.savefig(output_dir / "fig_F_vs_Ly_joint.png", dpi=300)
    fig_fab.tight_layout()
    fig_fab.savefig(output_dir / "fig_F_vs_AB_pair_length_joint.png", dpi=300)

    # Save gamma summary
    gamma_df = pd.DataFrame(
        {
            "fit": ["Ly", "AB_pair_length"],
            "slope": [slope_Ly, slope_AB],
            "intercept": [intercept_Ly, intercept_AB],
            "gamma": [gamma_Ly, gamma_AB],
        }
    )
    gamma_df.to_csv(output_dir / "gamma_from_entropy_F_joint.csv", index=False)
    print("Saved gamma fits to", output_dir / "gamma_from_entropy_F_joint.csv")


if __name__ == "__main__":
    main()
