#!/usr/bin/env python3
"""
Comparación three-way: sim_A (fixed-time naïf) vs sim_B (IA) vs sim_C (Webster óptimo).

Lee tripinfo.xml de sim_A, sim_B, sim_C y genera:
  - Box plot de duration y waitingTime para las 3 condiciones
  - Tabla: mean, std, mejora% vs A, p-value Mann-Whitney (A-B, A-C, B-C)
  - diag_three_way.png

Uso:
    uv run python scripts/compare_three_runs.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._sim_utils import find_sim_dir  # noqa: E402
from visualization.parsers import parse_tripinfo  # noqa: E402

_sim_a = find_sim_dir("A")
_sim_b = find_sim_dir("B")
_sim_c = find_sim_dir("C") or (PROJECT_ROOT / "sim_C")

SIM_DIRS = {
    "A (fixed-time)": (_sim_a / "logs" / "sumo_output" / "tripinfo.xml") if _sim_a else None,
    "B (IA)":         (_sim_b / "logs" / "sumo_output" / "tripinfo.xml") if _sim_b else None,
    "C (Webster)":    (_sim_c / "logs" / "sumo_output" / "tripinfo.xml"),
}
# Remove missing entries
SIM_DIRS = {k: v for k, v in SIM_DIRS.items() if v is not None}

# Output goes next to the B visualizations (or CWD/limitaciones as fallback)
_out_base = (_sim_b / "logs" / "visualizations" / "ab_test" / "limitaciones") if _sim_b \
    else (PROJECT_ROOT / "limitaciones")
OUT_DIR = _out_base


def load_all(sim_dirs: dict) -> dict:
    """Load tripinfo DataFrames for all conditions."""
    dfs = {}
    for label, path in sim_dirs.items():
        if not path.exists():
            print(f"  WARNING: {path} no encontrado — omitiendo {label}")
            continue
        df = parse_tripinfo(str(path))
        if df.empty:
            print(f"  WARNING: tripinfo vacío para {label}")
            continue
        dfs[label] = df
        print(f"  {label}: {len(df)} viajes completados")
    return dfs


def mann_whitney(a: pd.Series, b: pd.Series, alt: str = "two-sided") -> tuple:
    """Mann-Whitney U test. Returns (U, p)."""
    try:
        from scipy import stats
        result = stats.mannwhitneyu(a.dropna(), b.dropna(), alternative=alt)
        return result.statistic, result.pvalue
    except ImportError:
        return float("nan"), float("nan")


def format_pval(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "<0.001***"
    if p < 0.01:
        return f"{p:.3f}**"
    if p < 0.05:
        return f"{p:.3f}*"
    if p < 0.1:
        return f"{p:.3f}†"
    return f"{p:.3f}"


def print_summary_table(dfs: dict, metric: str) -> None:
    """Print mean ± std, improvement % vs A, and pairwise p-values."""
    labels = list(dfs.keys())
    ref_label = labels[0]  # sim_A is the reference

    print(f"\n--- {metric} ---")
    print(f"{'Condición':<20} {'n':>5} {'mean':>8} {'std':>8} {'Δ vs A':>10} {'p vs A':>12}")
    print("-" * 65)

    ref_mean = dfs[ref_label][metric].mean() if ref_label in dfs else 1.0

    for label, df in dfs.items():
        data = df[metric].dropna()
        mean = data.mean()
        std = data.std()
        pct = (mean - ref_mean) / ref_mean * 100 if ref_mean > 0 else float("nan")
        if label == ref_label:
            pval_str = "—"
        else:
            _, pval = mann_whitney(dfs[ref_label][metric], data, alt="two-sided")
            pval_str = format_pval(pval)
        pct_str = f"{pct:+.1f}%" if not np.isnan(pct) and label != ref_label else "—"
        print(f"{label:<20} {len(data):>5} {mean:>8.1f} {std:>8.1f} {pct_str:>10} {pval_str:>12}")

    # Pairwise B vs C
    if len(labels) >= 3:
        b_label = [l for l in labels if "B" in l]
        c_label = [l for l in labels if "C" in l]
        if b_label and c_label:
            bl, cl = b_label[0], c_label[0]
            _, pbc = mann_whitney(dfs[bl][metric], dfs[cl][metric])
            print(f"\n  B vs C: p={format_pval(pbc)}")


def plot_three_way(dfs: dict, out_path: Path) -> None:
    """Box plot of duration and waitingTime for all conditions."""
    if not dfs:
        print("  WARNING: Sin datos para graficar")
        return

    metrics = [
        ("duration", "Duración del viaje (s)"),
        ("waitingTime", "Tiempo de espera (s)"),
        ("timeLoss", "Tiempo perdido (s)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(
        "Comparación Three-Way: Fixed-Time vs IA vs Webster\n"
        "(sim_A = naïf, sim_B = IA-optimizado, sim_C = Webster óptimo teórico)",
        fontsize=12, fontweight="bold",
    )

    labels = list(dfs.keys())
    colors = ["steelblue", "tomato", "forestgreen"][:len(labels)]
    short_labels = [l.split()[0] for l in labels]  # "A", "B", "C"

    for ax, (metric, ylabel) in zip(axes, metrics):
        data_list = []
        valid_labels = []
        valid_colors = []

        for label, color in zip(labels, colors):
            if label not in dfs:
                continue
            data = dfs[label][metric].dropna()
            if len(data) == 0:
                continue
            data_list.append(data.values)
            valid_labels.append(label.split()[0])  # short label
            valid_colors.append(color)

        if not data_list:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(
            data_list,
            tick_labels=valid_labels,
            patch_artist=True,
            notch=False,
            showfliers=True,
            flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
        )

        for patch, color in zip(bp["boxes"], valid_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)

        # Add mean markers
        for i, (data, color) in enumerate(zip(data_list, valid_colors), 1):
            ax.scatter([i], [np.mean(data)], marker="D", color=color,
                       s=40, zorder=5, edgecolors="white", linewidths=0.5)

        ax.set_ylabel(ylabel)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3, axis="y")

        # Annotate means
        for i, data in enumerate(data_list, 1):
            ax.text(i, np.mean(data) * 1.02, f"{np.mean(data):.0f}",
                    ha="center", va="bottom", fontsize=7, color="black")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Box plot guardado: {out_path}")


def save_summary_csv(dfs: dict, out_path: Path) -> None:
    """Save summary statistics to CSV."""
    records = []
    labels = list(dfs.keys())
    ref_label = labels[0] if labels else None
    metrics = ["duration", "waitingTime", "timeLoss"]

    for label, df in dfs.items():
        rec = {"condition": label, "n_trips": len(df)}
        for metric in metrics:
            if metric not in df.columns:
                continue
            data = df[metric].dropna()
            rec[f"{metric}_mean"] = data.mean()
            rec[f"{metric}_std"] = data.std()
            rec[f"{metric}_median"] = data.median()
            # improvement vs A
            if ref_label and ref_label in dfs and label != ref_label:
                ref_mean = dfs[ref_label][metric].dropna().mean()
                rec[f"{metric}_pct_vs_A"] = (data.mean() - ref_mean) / ref_mean * 100
                _, pval = mann_whitney(dfs[ref_label][metric], data)
                rec[f"{metric}_p_vs_A"] = pval
            else:
                rec[f"{metric}_pct_vs_A"] = 0.0
                rec[f"{metric}_p_vs_A"] = float("nan")
        records.append(rec)

    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"  Summary CSV guardado: {out_path}")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Cargando tripinfo de las 3 simulaciones...")
    dfs = load_all(SIM_DIRS)

    if len(dfs) < 2:
        print("ERROR: Se necesitan al menos 2 simulaciones para comparar.")
        if not (PROJECT_ROOT / "sim_C").exists():
            print("  sim_C no existe. Ejecuta:")
            print("    uv run python scripts/generate_webster_timing.py")
            print("    uv run python scripts/run_sim_c.py")
        return 1

    # --- Print summary tables ---
    print("\n" + "=" * 65)
    print("COMPARACIÓN THREE-WAY: sim_A vs sim_B vs sim_C")
    print("=" * 65)
    for metric in ["duration", "waitingTime", "timeLoss"]:
        print_summary_table(dfs, metric)

    # --- Plot ---
    print("\nGenerando gráficas...")
    out_plot = OUT_DIR / "diag_three_way.png"
    plot_three_way(dfs, out_plot)

    # --- CSV ---
    csv_out = OUT_DIR / "three_way_summary.csv"
    save_summary_csv(dfs, csv_out)

    print(f"\nResultados en: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
