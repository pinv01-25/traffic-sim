#!/usr/bin/env python3
"""
Agrega resultados del experimento multi-seed.

Lee tripinfo.xml de todos los results/seed_*/sim_{A,B}/ y calcula:
  - mean ± std de duration, waitingTime, timeLoss por condición y seed
  - Wilcoxon pareado sobre la mejora% entre seeds
  - Violin plot (duration por seed, A vs B)

Output:
  - diag_multiseed_effect.png
  - multiseed_summary.csv

Uso:
    uv run python scripts/aggregate_seeds.py --results-dir results/
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from visualization.parsers import parse_tripinfo


def load_seed_results(results_dir: Path) -> pd.DataFrame:
    """
    Carga tripinfo.xml de todos los results/seed_*/sim_{A,B}/.

    Returns DataFrame con columnas: seed, condition, id, duration, waitingTime, timeLoss
    """
    rows = []
    seed_dirs = sorted(results_dir.glob("seed_*"))

    if not seed_dirs:
        print(f"WARNING: No se encontraron directorios seed_* en {results_dir}")
        return pd.DataFrame()

    for seed_dir in seed_dirs:
        seed_str = seed_dir.name.replace("seed_", "")
        try:
            seed = int(seed_str)
        except ValueError:
            continue

        for condition in ("sim_A", "sim_B"):
            tripinfo_path = seed_dir / condition / "tripinfo.xml"
            if not tripinfo_path.exists():
                print(f"  WARNING: No encontrado {tripinfo_path}")
                continue

            df = parse_tripinfo(str(tripinfo_path))
            if df.empty:
                print(f"  WARNING: tripinfo vacío en {tripinfo_path}")
                continue

            df["seed"] = seed
            df["condition"] = condition
            rows.append(df[["seed", "condition", "id", "duration", "waitingTime", "timeLoss"]])
            print(f"  Cargado seed={seed} {condition}: {len(df)} viajes")

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def compute_seed_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula estadísticas por (seed, condition) para las métricas principales.
    """
    metrics = ["duration", "waitingTime", "timeLoss"]
    records = []

    for (seed, cond), grp in df.groupby(["seed", "condition"]):
        rec = {"seed": seed, "condition": cond, "n_trips": len(grp)}
        for m in metrics:
            if m in grp.columns:
                rec[f"{m}_mean"] = grp[m].mean()
                rec[f"{m}_median"] = grp[m].median()
                rec[f"{m}_std"] = grp[m].std()
        records.append(rec)

    return pd.DataFrame(records).sort_values(["seed", "condition"])


def compute_improvement_per_seed(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada seed, calcula el % de mejora (A→B) en duration, waitingTime, timeLoss.
    Retorna DataFrame con columns: seed, duration_pct, waitingTime_pct, timeLoss_pct.
    """
    pivot = stats_df.pivot(index="seed", columns="condition")
    records = []
    for seed, row in pivot.iterrows():
        rec = {"seed": seed}
        for metric in ["duration", "waitingTime", "timeLoss"]:
            mean_a = row.get((f"{metric}_mean", "sim_A"), np.nan)
            mean_b = row.get((f"{metric}_mean", "sim_B"), np.nan)
            if pd.notna(mean_a) and pd.notna(mean_b) and mean_a > 0:
                rec[f"{metric}_pct"] = (mean_b - mean_a) / mean_a * 100
            else:
                rec[f"{metric}_pct"] = np.nan
        records.append(rec)
    return pd.DataFrame(records)


def wilcoxon_test(improvements: pd.Series, label: str) -> None:
    """Wilcoxon signed-rank test: ¿es la mejora consistentemente distinta de 0?"""
    valid = improvements.dropna()
    if len(valid) < 4:
        print(f"  {label}: n={len(valid)} — muestra insuficiente para test")
        return
    try:
        from scipy import stats
        stat, pval = stats.wilcoxon(valid, alternative="less")
        direction = "mejora" if valid.mean() < 0 else "empeora"
        print(f"  {label}: n={len(valid)}, mean={valid.mean():.2f}%, "
              f"Wilcoxon W={stat:.0f}, p={pval:.4f} ({direction})")
    except ImportError:
        print(f"  {label}: n={len(valid)}, mean={valid.mean():.2f}% (scipy no disponible)")


def plot_violin_by_seed(df: pd.DataFrame, out_path: Path) -> None:
    """Violin plot de duration por seed y condición (A vs B)."""
    seeds = sorted(df["seed"].unique())
    n_seeds = len(seeds)

    if n_seeds == 0:
        print("  WARNING: Sin datos para violin plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        f"Experimento Multi-Seed: distribución de métricas por seed\n"
        f"(sim_A = fixed-time, sim_B = IA optimizado; n={n_seeds} seeds)",
        fontsize=12, fontweight="bold",
    )

    metrics = [
        ("duration", "Duración del viaje (s)", "duration"),
        ("waitingTime", "Tiempo de espera (s)", "waitingTime"),
        ("timeLoss", "Tiempo perdido (s)", "timeLoss"),
    ]

    colors = {"sim_A": "steelblue", "sim_B": "tomato"}
    label_map = {"sim_A": "A (fixed)", "sim_B": "B (IA)"}

    for ax, (metric, ylabel, _) in zip(axes, metrics):
        positions_a = []
        positions_b = []
        data_a = []
        data_b = []
        xticks = []
        xticklabels = []

        for i, seed in enumerate(seeds):
            base = i * 3
            positions_a.append(base + 0.5)
            positions_b.append(base + 1.5)
            xticks.append(base + 1.0)
            xticklabels.append(f"s={seed}")

            grp_a = df[(df["seed"] == seed) & (df["condition"] == "sim_A")][metric].dropna()
            grp_b = df[(df["seed"] == seed) & (df["condition"] == "sim_B")][metric].dropna()
            data_a.append(grp_a.values if len(grp_a) > 0 else np.array([np.nan]))
            data_b.append(grp_b.values if len(grp_b) > 0 else np.array([np.nan]))

        # Plot violins
        for pos, data, cond in [
            (positions_a, data_a, "sim_A"),
            (positions_b, data_b, "sim_B"),
        ]:
            valid = [(p, d) for p, d in zip(pos, data) if not np.all(np.isnan(d)) and len(d) > 1]
            if valid:
                vp = ax.violinplot(
                    [d for _, d in valid],
                    positions=[p for p, _ in valid],
                    widths=0.8, showmedians=True, showextrema=False,
                )
                for body in vp["bodies"]:
                    body.set_facecolor(colors[cond])
                    body.set_alpha(0.6)
                vp["cmedians"].set_color(colors[cond])
                vp["cmedians"].set_linewidth(2)

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=30, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3, axis="y")

        # Legend proxy
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors["sim_A"], alpha=0.6, label=label_map["sim_A"]),
            Patch(facecolor=colors["sim_B"], alpha=0.6, label=label_map["sim_B"]),
        ]
        ax.legend(handles=legend_elements, fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Violin plot guardado: {out_path}")


def plot_improvement_scatter(impr_df: pd.DataFrame, out_path: Path) -> None:
    """Scatter plot de % mejora por seed para las 3 métricas."""
    if impr_df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    colors_m = {"duration_pct": "steelblue", "waitingTime_pct": "tomato", "timeLoss_pct": "forestgreen"}
    markers_m = {"duration_pct": "o", "waitingTime_pct": "s", "timeLoss_pct": "^"}
    labels_m = {"duration_pct": "duration", "waitingTime_pct": "waitingTime", "timeLoss_pct": "timeLoss"}

    x = np.arange(len(impr_df))
    for col in ["duration_pct", "waitingTime_pct", "timeLoss_pct"]:
        if col not in impr_df.columns:
            continue
        ax.scatter(x, impr_df[col].values, color=colors_m[col],
                   marker=markers_m[col], s=80, label=labels_m[col], zorder=3)
        ax.plot(x, impr_df[col].values, color=colors_m[col], alpha=0.4, lw=1)

    ax.axhline(0, color="black", lw=1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed={s}" for s in impr_df["seed"]], rotation=30, fontsize=9)
    ax.set_ylabel("Mejora B vs A (%)\n(negativo = B mejor que A)")
    ax.set_title("Mejora de IA (sim_B) vs fixed-time (sim_A) por seed")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Scatter plot guardado: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Agrega resultados multi-seed y genera violin plots"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Directorio con resultados multi-seed (default: results/)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directorio de salida de gráficas (default: results/multiseed_analysis/)",
    )
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"ERROR: {args.results_dir} no existe. Corre run_multiseed.py primero.")
        return 1

    out_dir = args.out_dir or (args.results_dir / "multiseed_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cargando resultados de: {args.results_dir}")
    df = load_seed_results(args.results_dir)

    if df.empty:
        print("ERROR: No se encontraron datos. Verifica que run_multiseed.py haya completado.")
        return 1

    n_seeds = df["seed"].nunique()
    print(f"\nTotal viajes cargados: {len(df)} ({n_seeds} seeds × 2 condiciones)")
    print(f"Condiciones: {sorted(df['condition'].unique())}")

    # --- Compute per-seed stats ---
    stats_df = compute_seed_stats(df)
    stats_csv = out_dir / "multiseed_stats_by_seed.csv"
    stats_df.to_csv(stats_csv, index=False)
    print(f"\nEstadísticas por seed guardadas: {stats_csv}")

    # --- Aggregate across seeds ---
    print("\n--- Resumen agregado (mean ± std a través de seeds) ---")
    for cond in ("sim_A", "sim_B"):
        sub = stats_df[stats_df["condition"] == cond]
        for metric in ["duration", "waitingTime", "timeLoss"]:
            col = f"{metric}_mean"
            if col in sub.columns:
                vals = sub[col].dropna()
                print(f"  {cond} {metric}: {vals.mean():.2f} ± {vals.std():.2f} s "
                      f"(across {len(vals)} seeds)")

    # --- Improvement per seed ---
    impr_df = compute_improvement_per_seed(stats_df)
    impr_csv = out_dir / "multiseed_improvement.csv"
    impr_df.to_csv(impr_csv, index=False)

    print("\n--- Mejora % (B vs A) por seed ---")
    for _, row in impr_df.iterrows():
        parts = [f"seed={int(row['seed'])}"]
        for m in ["duration_pct", "waitingTime_pct", "timeLoss_pct"]:
            if m in row and pd.notna(row[m]):
                parts.append(f"{m.replace('_pct', '')}={row[m]:+.1f}%")
        print("  " + "  ".join(parts))

    # --- Wilcoxon tests ---
    print("\n--- Tests estadísticos (Wilcoxon: mejora consistente entre seeds?) ---")
    for metric in ["duration_pct", "waitingTime_pct", "timeLoss_pct"]:
        if metric in impr_df.columns:
            wilcoxon_test(impr_df[metric], metric.replace("_pct", ""))

    # --- Plots ---
    print("\nGenerando gráficas...")
    plot_violin_by_seed(df, out_dir / "diag_multiseed_effect.png")
    plot_improvement_scatter(impr_df, out_dir / "diag_multiseed_improvement.png")

    print(f"\nAnálisis completado. Resultados en: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
