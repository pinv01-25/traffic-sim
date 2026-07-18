"""Comprehensive A/B test utilities for comparing two SUMO simulation runs.

Provides statistical analysis and visualizations for comparing baseline vs
optimized traffic light timing. Generates:
 - Multiple distribution comparisons (histograms, CDFs, violins, boxplots)
 - Time series comparisons with confidence intervals
 - Summary metric comparisons
 - FCD-based speed analysis
 - Statistical tests (permutation, bootstrap, Mann-Whitney U)
 - Improvement summary
 - Detailed CSV/JSON reports
"""
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .parsers import (
    compute_summary_statistics,
    compute_trip_statistics,
    parse_fcd,
    parse_summary,
    parse_tripinfo,
)
from .plots import (
    plot_boxplot_two,
    plot_congestion_timeline,
    plot_correlation_heatmap,
    plot_efficiency_comparison,
    plot_fcd_comparison,
    # Basic plots
    plot_histogram_cdf_two,
    plot_improvement_summary,
    # Advanced comparison plots
    plot_metric_comparison_bars,
    plot_multi_metric_violin,
    plot_percentile_comparison,
    plot_speed_distribution_comparison,
    plot_summary_comparison,
    plot_time_series_comparison,
    plot_time_series_mean,
    plot_violin_comparison,
    plot_waiting_time_analysis,
)


def _find_file(run_dir: str, filename: str) -> Optional[Path]:
    """Find a file in common SUMO output locations."""
    candidates = [
        Path(run_dir) / 'logs' / 'sumo_output' / filename,
        Path(run_dir) / filename
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# =============================================================================
# INCOMPLETE TRIPS ANALYSIS
# =============================================================================

def _get_last_fcd_times(run_dir: str) -> Dict[str, float]:
    """Stream fcd.xml and return the last observed timestep per vehicle ID."""
    import xml.etree.ElementTree as ET
    p = _find_file(run_dir, 'fcd.xml')
    if p is None:
        return {}
    last_time: Dict[str, float] = {}
    current_time = 0.0
    for _event, elem in ET.iterparse(str(p), events=['end']):
        if elem.tag == 'timestep':
            current_time = float(elem.get('time', 0.0))
        elif elem.tag == 'vehicle':
            vid = elem.get('id')
            if vid:
                last_time[vid] = current_time
            elem.clear()
    return last_time


def _get_depart_times(run_dir: str) -> Dict[str, float]:
    """Parse routes.rou.xml and return vehicle depart times."""
    import xml.etree.ElementTree as ET
    p = _find_file(run_dir, 'routes.rou.xml')
    if p is None:
        # also look in parent of logs/sumo_output
        p = Path(run_dir) / 'routes.rou.xml'
        if not p.exists():
            return {}
    depart: Dict[str, float] = {}
    for _event, elem in ET.iterparse(str(p)):
        if elem.tag == 'vehicle':
            vid = elem.get('id')
            d = elem.get('depart')
            if vid and d is not None:
                try:
                    depart[vid] = float(d)
                except ValueError:
                    pass
            elem.clear()
    return depart


def _compute_incomplete_trips(run_dir: str, df_trip: pd.DataFrame) -> pd.DataFrame:
    """
    Identify vehicles present in FCD but absent in tripinfo (incomplete trips).

    Returns DataFrame with columns:
        id, last_fcd_time, depart, time_in_network
    """
    last_fcd = _get_last_fcd_times(run_dir)
    if not last_fcd:
        return pd.DataFrame()

    completed_ids = set(df_trip['id'].tolist()) if not df_trip.empty else set()
    incomplete_ids = set(last_fcd.keys()) - completed_ids

    if not incomplete_ids:
        return pd.DataFrame()

    depart_times = _get_depart_times(run_dir)

    rows = []
    for vid in incomplete_ids:
        last_t = last_fcd[vid]
        depart = depart_times.get(vid)
        rows.append({
            'id': vid,
            'last_fcd_time': last_t,
            'depart': depart if depart is not None else float('nan'),
            'time_in_network': (last_t - depart) if depart is not None else float('nan'),
        })
    return pd.DataFrame(rows).sort_values('time_in_network').reset_index(drop=True)


def _plot_incomplete_histogram(
    completed_dur: np.ndarray,
    time_in_network: np.ndarray,
    out_dir: str,
    label: str,
    filename: str = '21_incomplete_histogram.png',
) -> None:
    """Histogram: duration of completed trips vs time_in_network of incomplete ones."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    p99_c = float(np.percentile(completed_dur, 99)) if len(completed_dur) else 1
    p99_i = float(np.percentile(time_in_network, 99)) if len(time_in_network) else 1
    ax.hist(np.clip(completed_dur, 0, p99_c), bins=30, alpha=0.55,
            color='steelblue', density=True,
            label=f'Completados — {label} (n={len(completed_dur)})')
    if len(time_in_network):
        ax.hist(np.clip(time_in_network, 0, p99_i), bins=20, alpha=0.55,
                color='tomato', density=True,
                label=f'Incompletos time_in_network (n={len(time_in_network)})')
        ax.axvline(float(np.mean(time_in_network)), color='tomato', ls='--', lw=1.5,
                   label=f'μ incompletos={np.mean(time_in_network):.0f}s')
    ax.axvline(float(np.mean(completed_dur)), color='steelblue', ls='--', lw=1.5,
               label=f'μ completados={np.mean(completed_dur):.0f}s')
    ax.set_xlabel('Tiempo en red (s)')
    ax.set_ylabel('Densidad')
    ax.set_title(f'Distribución de tiempos — viajes incompletos vs completados\n({label})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_incomplete_cdf(
    completed_dur: np.ndarray,
    time_in_network: np.ndarray,
    out_dir: str,
    label: str,
    filename: str = '22_incomplete_cdf.png',
) -> None:
    """CDF: completed duration vs incomplete time_in_network."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for data, color, lbl in [
        (np.sort(completed_dur), 'steelblue', 'Completados (duration)'),
        (np.sort(time_in_network), 'tomato', 'Incompletos (time_in_network)'),
    ]:
        if len(data):
            cdf = np.arange(1, len(data) + 1) / len(data)
            ax.plot(data, cdf, color=color, lw=2, label=lbl)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('CDF')
    ax.set_title(f'Función de distribución acumulada — incompletos vs completados\n({label})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_incomplete_boxplot(
    completed_dur: np.ndarray,
    time_in_network: np.ndarray,
    out_dir: str,
    label: str,
    filename: str = '23_incomplete_boxplot.png',
) -> None:
    """Box plot: completed duration vs incomplete time_in_network."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    data_list = [d for d in [completed_dur, time_in_network] if len(d) > 0]
    tick_labels = ['Completados\n(duration)', 'Incompletos\n(time_in_network)'][:len(data_list)]
    bp = ax.boxplot(data_list, tick_labels=tick_labels, patch_artist=True,
                    showfliers=True,
                    flierprops={'marker': '.', 'markersize': 3, 'alpha': 0.4})
    colors = ['steelblue', 'tomato']
    for patch, color in zip(bp['boxes'], colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    for i, data in enumerate(data_list, 1):
        ax.scatter([i], [np.mean(data)], marker='D', color=colors[i - 1],
                   s=50, zorder=5, edgecolors='white', linewidths=0.8)
        ax.text(i, np.mean(data) * 1.02, f'μ={np.mean(data):.0f}s',
                ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Tiempo (s)')
    ax.set_title(f'Box plot comparativo — viajes incompletos vs completados\n({label})')
    ax.grid(True, alpha=0.25, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_incomplete_scatter(
    df_incomplete: pd.DataFrame,
    completed_dur: np.ndarray,
    out_dir: str,
    label: str,
    filename: str = '24_incomplete_scatter.png',
) -> None:
    """Scatter: depart time vs time_in_network for incomplete trips."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    valid = df_incomplete.dropna(subset=['time_in_network', 'depart'])
    if not valid.empty:
        ax.scatter(valid['depart'], valid['time_in_network'],
                   alpha=0.6, s=35, color='tomato', label='Viaje incompleto', zorder=3)
    if len(completed_dur):
        ax.axhline(float(np.mean(completed_dur)), color='steelblue', ls='--', lw=1.8,
                   label=f'μ duration completados ({np.mean(completed_dur):.0f}s)')
    ax.set_xlabel('Tiempo de salida (depart, s)')
    ax.set_ylabel('time_in_network (s)')
    ax.set_title(f'Viajes incompletos: tiempo de salida vs tiempo en red\n({label})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    n = len(valid)
    ax.annotate(f'n={n} incompletos', xy=(0.02, 0.96), xycoords='axes fraction',
                fontsize=9, va='top')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), dpi=150, bbox_inches='tight')
    plt.close(fig)


def analyze_incomplete_trips(
    run_dir: str,
    df_trip: pd.DataFrame,
    out_dir: str,
    label: str = 'A',
    tag: str = 'A',
    file_start: int = 21,
) -> List[str]:
    """
    Analyze incomplete trips (vehicles in FCD but not in tripinfo) and generate
    4 separate diagnostic plots.

    Called automatically from compare_runs() for both runs (A: files 21-24,
    B: files 25-28).

    Args:
        run_dir:    Simulation directory (must contain fcd.xml and routes.rou.xml)
        df_trip:    Parsed tripinfo DataFrame for this run
        out_dir:    Output directory for the 4 PNGs
        label:      Run label shown in plot titles
        tag:        Short tag used in filenames ('A' or 'B')
        file_start: Number of the first generated plot

    Returns:
        List of generated filenames.
    """
    print(f'  Analyzing incomplete trips for {label}...')
    df_incomplete = _compute_incomplete_trips(run_dir, df_trip)

    if df_incomplete.empty:
        print('  No incomplete trips data available (fcd.xml missing?)')
        return []

    completed_dur = df_trip['duration'].dropna().values if not df_trip.empty else np.array([])
    time_in_network = df_incomplete['time_in_network'].dropna().values

    n_complete = len(completed_dur)
    n_incomplete = len(df_incomplete)
    print(f'  Completados: {n_complete} | Incompletos: {n_incomplete} '
          f'| time_in_network mean: {time_in_network.mean():.1f}s' if len(time_in_network) else
          f'  Completados: {n_complete} | Incompletos: {n_incomplete}')

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    generated = []

    names = [
        f'{file_start:02d}_incomplete_histogram_{tag}.png',
        f'{file_start + 1:02d}_incomplete_cdf_{tag}.png',
        f'{file_start + 2:02d}_incomplete_boxplot_{tag}.png',
        f'{file_start + 3:02d}_incomplete_scatter_{tag}.png',
    ]

    _plot_incomplete_histogram(completed_dur, time_in_network, out_dir, label, filename=names[0])
    _plot_incomplete_cdf(completed_dur, time_in_network, out_dir, label, filename=names[1])
    _plot_incomplete_boxplot(completed_dur, time_in_network, out_dir, label, filename=names[2])
    _plot_incomplete_scatter(df_incomplete, completed_dur, out_dir, label, filename=names[3])
    generated.extend(names)

    # Save CSV
    csv_path = Path(out_dir) / f'incomplete_trips_{tag}.csv'
    df_incomplete.to_csv(str(csv_path), index=False)

    return generated


def _load_tripinfo(run_dir: str) -> pd.DataFrame:
    """Load tripinfo.xml from a run directory."""
    p = _find_file(run_dir, 'tripinfo.xml')
    if p is None:
        return pd.DataFrame()
    return parse_tripinfo(str(p))


def _load_summary(run_dir: str) -> pd.DataFrame:
    """Load summary.xml from a run directory."""
    p = _find_file(run_dir, 'summary.xml')
    if p is None:
        return pd.DataFrame()
    return parse_summary(str(p))


def _load_fcd(run_dir: str, sample_rate: int = 5) -> pd.DataFrame:
    """Load fcd.xml from a run directory with sampling."""
    p = _find_file(run_dir, 'fcd.xml')
    if p is None:
        return pd.DataFrame()
    return parse_fcd(str(p), sample_rate=sample_rate)


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def permutation_test_mean(x: np.ndarray, y: np.ndarray, n_iter: int = 5000, seed: int = 0) -> float:
    """Two-sample permutation test for difference in means.

    Tests H0: mean(x) = mean(y)
    Returns p-value (two-sided).
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) == 0 or len(y) == 0:
        return np.nan

    obs_diff = abs(x.mean() - y.mean())
    pooled = np.concatenate([x, y])
    n_x = len(x)

    count = 0
    for _ in range(n_iter):
        rng.shuffle(pooled)
        new_diff = abs(pooled[:n_x].mean() - pooled[n_x:].mean())
        if new_diff >= obs_diff:
            count += 1

    return (count + 1) / (n_iter + 1)


def bootstrap_ci_diff(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 1
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for difference in means.

    Returns (lower_ci, upper_ci, point_estimate).
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) == 0 or len(y) == 0:
        return (np.nan, np.nan, np.nan)

    diffs = []
    for _ in range(n_boot):
        sx = rng.choice(x, size=len(x), replace=True)
        sy = rng.choice(y, size=len(y), replace=True)
        diffs.append(sx.mean() - sy.mean())

    lo = np.percentile(diffs, 100 * (alpha / 2))
    hi = np.percentile(diffs, 100 * (1 - alpha / 2))
    point = x.mean() - y.mean()

    return (lo, hi, point)


def mann_whitney_test(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Mann-Whitney U test (non-parametric).

    Returns (statistic, p-value).
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) < 2 or len(y) < 2:
        return (np.nan, np.nan)

    stat, pval = scipy_stats.mannwhitneyu(x, y, alternative='two-sided')
    return (stat, pval)


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) < 2 or len(y) < 2:
        return np.nan

    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * x.std(ddof=1)**2 + (ny - 1) * y.std(ddof=1)**2) / (nx + ny - 2))

    if pooled_std == 0:
        return 0.0

    return (x.mean() - y.mean()) / pooled_std


# =============================================================================
# PAIRED PER-VEHICLE ANALYSIS
# =============================================================================
# La demanda es idéntica en A y B (mismos vehículos, mismas rutas, mismos
# departs), así que cada vehículo puede compararse consigo mismo. Esta es la
# evidencia estadística más fuerte del experimento.

def _paired_frame(df_a: pd.DataFrame, df_b: pd.DataFrame, metric: str = 'duration') -> pd.DataFrame:
    """Join per-vehicle metric of both runs on vehicle id.

    Returns DataFrame with columns id, <metric>_a, <metric>_b, delta (B - A).
    Empty DataFrame if pairing is not possible.
    """
    for df in (df_a, df_b):
        if df.empty or 'id' not in df.columns or metric not in df.columns:
            return pd.DataFrame()
    merged = pd.merge(
        df_a[['id', metric]].dropna(),
        df_b[['id', metric]].dropna(),
        on='id', suffixes=('_a', '_b'),
    )
    if merged.empty:
        return merged
    merged['delta'] = merged[f'{metric}_b'] - merged[f'{metric}_a']
    return merged


def paired_statistics(df_a: pd.DataFrame, df_b: pd.DataFrame, metric: str = 'duration') -> Dict:
    """Per-vehicle paired statistics (same vehicle in A and B).

    Returns dict with n_paired, mean_delta, median_delta, pct_improved
    (share of vehicles with lower metric in B) and wilcoxon_pvalue
    (two-sided signed-rank test on the deltas). Empty dict if no pairs.
    """
    merged = _paired_frame(df_a, df_b, metric)
    if merged.empty:
        return {}
    delta = merged['delta'].values
    try:
        _stat, wilcoxon_pvalue = scipy_stats.wilcoxon(delta, alternative='two-sided')
        wilcoxon_pvalue = float(wilcoxon_pvalue)
    except ValueError:  # e.g. todas las diferencias son cero
        wilcoxon_pvalue = float('nan')
    return {
        'metric': metric,
        'n_paired': int(len(delta)),
        'mean_delta': float(np.mean(delta)),
        'median_delta': float(np.median(delta)),
        'pct_improved': float((delta < 0).mean() * 100),
        'wilcoxon_pvalue': wilcoxon_pvalue,
    }


def _plot_qq_comparison(dur_a, dur_b, out_dir, labels, filename='29_qq_duration.png'):
    """QQ plot: quantiles of B vs quantiles of A with y=x reference."""
    import matplotlib.pyplot as plt

    qs = np.linspace(1, 99, 99)
    qa = np.percentile(dur_a, qs)
    qb = np.percentile(dur_b, qs)

    fig, ax = plt.subplots(figsize=(7, 7))
    lim = [0, max(qa.max(), qb.max()) * 1.05]
    ax.plot(lim, lim, color='gray', ls='--', lw=1.5, label='y = x (sin cambio)')
    ax.scatter(qa, qb, s=18, color='steelblue', alpha=0.8, zorder=3)
    ax.fill_between(lim, [0, lim[1]], [0, 0], color='seagreen', alpha=0.06)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f'Cuantiles {labels[0]} (s)')
    ax.set_ylabel(f'Cuantiles {labels[1]} (s)')
    ax.set_title(f'QQ plot de duración de viaje — {labels[1]} vs {labels[0]}\n'
                 'Puntos bajo la diagonal = mejora en ese cuantil')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_paired_scatter(merged, metric, out_dir, labels, filename='30_paired_scatter.png'):
    """Per-vehicle scatter: metric in A vs metric in B, colored by outcome."""
    import matplotlib.pyplot as plt

    x = merged[f'{metric}_a'].values
    y = merged[f'{metric}_b'].values
    improved = y < x

    fig, ax = plt.subplots(figsize=(8, 8))
    lim = [0, max(x.max(), y.max()) * 1.05]
    ax.plot(lim, lim, color='gray', ls='--', lw=1.5, label='y = x (sin cambio)')
    ax.scatter(x[improved], y[improved], s=8, alpha=0.35, color='seagreen',
               label=f'Mejora (n={improved.sum()})')
    ax.scatter(x[~improved], y[~improved], s=8, alpha=0.35, color='tomato',
               label=f'Empeora (n={(~improved).sum()})')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f'{metric} en {labels[0]} (s)')
    ax.set_ylabel(f'{metric} en {labels[1]} (s)')
    pct = improved.mean() * 100
    ax.set_title(f'Comparación pareada por vehículo (mismo vehículo en ambas corridas)\n'
                 f'{pct:.1f}% de los {len(merged)} vehículos mejora en {labels[1]}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_paired_delta_hist(delta, metric, out_dir, labels, filename='31_paired_delta_hist.png'):
    """Histogram of per-vehicle deltas (B - A); negative = improvement."""
    import matplotlib.pyplot as plt

    lo, hi = np.percentile(delta, [1, 99])
    clipped = np.clip(delta, lo, hi)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(clipped, bins=40, color='steelblue', alpha=0.75)
    ax.axvline(0, color='gray', ls='-', lw=1.5, label='Sin cambio')
    ax.axvline(float(np.mean(delta)), color='seagreen' if np.mean(delta) < 0 else 'tomato',
               ls='--', lw=2, label=f'Δ medio = {np.mean(delta):+.1f}s')
    ax.axvline(float(np.median(delta)), color='darkorange', ls=':', lw=2,
               label=f'Δ mediana = {np.median(delta):+.1f}s')
    pct = (delta < 0).mean() * 100
    ax.set_xlabel(f'Δ {metric} por vehículo: {labels[1]} − {labels[0]} (s)')
    ax.set_ylabel('Vehículos')
    ax.set_title(f'Diferencia pareada por vehículo (negativo = mejora)\n'
                 f'{pct:.1f}% de vehículos mejora')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_throughput_timeline(df_trip_a, df_trip_b, out_dir, labels,
                              filename='32_throughput_timeline.png'):
    """Cumulative completed trips over time for both runs."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    finals = []
    for df, color, label in [(df_trip_a, 'steelblue', labels[0]),
                             (df_trip_b, 'seagreen', labels[1])]:
        if df.empty or 'arrival' not in df.columns:
            continue
        arrivals = np.sort(df['arrival'].dropna().values)
        if len(arrivals) == 0:
            continue
        ax.step(arrivals, np.arange(1, len(arrivals) + 1), where='post',
                color=color, lw=2, label=f'{label} (total={len(arrivals)})')
        finals.append(len(arrivals))
    ax.set_xlabel('Tiempo de simulación (s)')
    ax.set_ylabel('Viajes completados (acumulado)')
    title = 'Throughput: viajes completados acumulados en el tiempo'
    if len(finals) == 2 and finals[0] > 0:
        title += f'\n{labels[1]} completa {(finals[1] - finals[0]) / finals[0] * 100:+.1f}% de viajes'
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), dpi=150, bbox_inches='tight')
    plt.close(fig)


def throughput_statistics(df_trip_a: pd.DataFrame, df_trip_b: pd.DataFrame) -> Dict:
    """Completed-trip counts per run and relative change."""
    n_a = int(len(df_trip_a)) if not df_trip_a.empty else 0
    n_b = int(len(df_trip_b)) if not df_trip_b.empty else 0
    return {
        'completed_a': n_a,
        'completed_b': n_b,
        'throughput_change_pct': float((n_b - n_a) / n_a * 100) if n_a > 0 else float('nan'),
    }


def system_time_statistics(df_summary_a: pd.DataFrame, df_summary_b: pd.DataFrame) -> Dict:
    """Métrica primaria insesgada: tiempo total de sistema (vehículo-segundos).

    `percent_improvement` (media de `duration` en tripinfo) solo ve viajes
    COMPLETADOS: si un lado colapsa (gridlock), su media baja artificialmente
    porque promedia solo los viajes rápidos que lograron escapar (sesgo de
    supervivencia). El summary.xml de SUMO, en cambio, reporta por cada paso
    de simulación cuántos vehículos están `running` (circulando, incluidos
    los atrapados) — su suma es `system_total_time`, insesgada porque cuenta
    a TODOS los vehículos en red mientras están en red, hayan llegado o no.

    Se asume un paso de summary de 1s (el caso normal); si el intervalo real
    difiere, la comparación relativa entre A y B sigue siendo válida siempre
    que ambas corridas usen el mismo intervalo.

    `inserted` es una columna acumulada (SUMO reporta el total insertado
    hasta ese paso) → se usa su máximo como total de vehículos que entraron
    a la red. Si A y B insertan cantidades distintas (backlog de inserción
    por gridlock: los vehículos esperan en el borde y no llegan a
    contabilizarse como circulando), `mean_time_per_vehicle` normaliza el
    tiempo total por vehículo insertado, para que la comparación no
    favorezca artificialmente a la corrida que menos vehículos logró meter
    en la red.
    """
    def _totals(df):
        if df.empty or 'running' not in df.columns:
            return (float('nan'), float('nan'))
        total_time = float(df['running'].dropna().sum())
        if 'inserted' in df.columns and not df['inserted'].dropna().empty:
            inserted = float(df['inserted'].dropna().max())
        else:
            inserted = float('nan')
        return total_time, inserted

    time_a, ins_a = _totals(df_summary_a)
    time_b, ins_b = _totals(df_summary_b)

    result = {
        'system_time_a': time_a,
        'system_time_b': time_b,
        'inserted_a': ins_a,
        'inserted_b': ins_b,
    }

    if time_a == time_a and time_a > 0 and time_b == time_b:
        result['system_time_improvement_pct'] = float((time_a - time_b) / time_a * 100)
    else:
        result['system_time_improvement_pct'] = float('nan')

    mtpv_a = time_a / max(ins_a, 1) if time_a == time_a and ins_a == ins_a else float('nan')
    mtpv_b = time_b / max(ins_b, 1) if time_b == time_b and ins_b == ins_b else float('nan')
    result['mean_time_per_vehicle_a'] = mtpv_a
    result['mean_time_per_vehicle_b'] = mtpv_b

    if mtpv_a == mtpv_a and mtpv_a > 0 and mtpv_b == mtpv_b:
        result['mean_time_per_vehicle_improvement_pct'] = float((mtpv_a - mtpv_b) / mtpv_a * 100)
    else:
        result['mean_time_per_vehicle_improvement_pct'] = float('nan')

    return result


# =============================================================================
# VEREDICTO COMPUESTO (insesgado por supervivencia)
# =============================================================================
#
# `percent_improvement` (media de duration en tripinfo) sufre sesgo de
# supervivencia: solo ve viajes completados, así que un colapso (gridlock)
# hace bajar la media porque solo promedia los viajes rápidos que
# escaparon. Esta función combina throughput (guardia dura), tiempo de
# sistema (métrica primaria insesgada) y significancia estadística en un
# único veredicto, usado por el JSON, el CSV, el HTML y el resumen de
# campaña — para que los cuatro nunca se contradigan entre sí.

#: Umbral de cambio de throughput (%) que se considera "grande": por encima
#: decide el veredicto sin importar la duración (guardia dura).
_THR_BIG_PCT = 10.0
#: Debajo de este |headline_pct| (tiempo de sistema, o duration si no hay
#: tiempo de sistema disponible) el resultado se considera empate.
_HEADLINE_TIE_PCT = 2.0


def compute_verdict(statistical_results: Dict) -> Dict:
    """Veredicto A/B único e insesgado.

    Devuelve un dict con:
        verdict: 'improvement' | 'regression' | 'tie' |
                 'misleading_improvement' | 'misleading_regression'
        reason: explicación en español.
        headline_pct: porcentaje a mostrar como titular (tiempo de sistema
            cuando está disponible; si no, la media sesgada de duration).

    Reglas (en orden):
      1. Colapso a 0 viajes completados en alguna corrida → 'tie' (no hay
         base de comparación).
      2. |throughput_change_pct| > 10%: el throughput manda.
         - Cae > 10%: 'regression' (o 'misleading_improvement' si la media
           de duration mostraba una mejora — sesgo de supervivencia).
         - Sube > 10%: favorable — 'improvement' (o 'misleading_regression'
           si la media de duration mostraba un empeoramiento porque el
           baseline colapsó).
      3. Throughput dentro de ±10%: decide `system_time_improvement_pct`
         (con `percent_improvement` como respaldo si no hay tiempo de
         sistema), apoyado en significancia estadística (permutación o
         Wilcoxon pareado). |headline| < 2% o no significativo → 'tie'.
    """
    thr = statistical_results.get('throughput') or {}
    thr_pct = thr.get('throughput_change_pct')
    completed_a = thr.get('completed_a')
    completed_b = thr.get('completed_b')
    pct = statistical_results.get('percent_improvement')

    st = statistical_results.get('system_time') or {}
    st_pct = st.get('system_time_improvement_pct')
    if st_pct is not None and st_pct != st_pct:  # NaN
        st_pct = None
    headline = st_pct if st_pct is not None else pct

    p = statistical_results.get('permutation_test_pvalue')
    paired = statistical_results.get('paired') or {}
    p_paired = paired.get('wilcoxon_pvalue')
    significant = (
        (p is not None and p == p and p < 0.05) or
        (p_paired is not None and p_paired == p_paired and p_paired < 0.05)
    )

    # 1. Colapso total: sin base de comparación.
    if completed_a == 0 or completed_b == 0:
        return {
            'verdict': 'tie',
            'reason': ('Colapso total de viajes completados (0 en alguna corrida); '
                       'no hay base de comparación.'),
            'headline_pct': pct,
        }

    # 2. Guardia dura de throughput.
    if thr_pct is not None and thr_pct == thr_pct:
        if thr_pct < -_THR_BIG_PCT:
            if pct is not None and pct > 0:
                return {
                    'verdict': 'misleading_improvement',
                    'reason': (
                        f'La media de duración "mejora" {pct:+.1f}% pero el throughput cae '
                        f'{thr_pct:+.1f}% ({completed_a} → {completed_b} viajes): sesgo de '
                        'supervivencia — los viajes lentos no llegan a completarse.'
                    ),
                    'headline_pct': headline,
                }
            return {
                'verdict': 'regression',
                'reason': (
                    f'El throughput cae {thr_pct:+.1f}% ({completed_a} → {completed_b} viajes); '
                    'la corrida optimizada empeora el sistema.'
                ),
                'headline_pct': headline,
            }
        if thr_pct > _THR_BIG_PCT:
            if pct is not None and pct < 0:
                return {
                    'verdict': 'misleading_regression',
                    'reason': (
                        f'La media de duración "empeora" {pct:+.1f}% pero el throughput sube '
                        f'{thr_pct:+.1f}% ({completed_a} → {completed_b} viajes): el baseline '
                        'colapsó y su media solo promedia los viajes rápidos que escaparon.'
                    ),
                    'headline_pct': headline,
                }
            return {
                'verdict': 'improvement',
                'reason': (
                    f'El throughput sube {thr_pct:+.1f}% ({completed_a} → {completed_b} viajes) '
                    'y el tiempo de sistema mejora.'
                ),
                'headline_pct': headline,
            }

    # 3. Throughput estable: decide el tiempo de sistema (o duration como
    #    respaldo), apoyado en significancia estadística.
    if headline is None:
        return {'verdict': 'tie', 'reason': 'Sin datos suficientes para veredicto.', 'headline_pct': None}
    if abs(headline) < _HEADLINE_TIE_PCT or not significant:
        return {
            'verdict': 'tie',
            'reason': f'La diferencia observada ({headline:+.1f}%) no alcanza significancia o es marginal.',
            'headline_pct': headline,
        }
    if headline > 0:
        return {
            'verdict': 'improvement',
            'reason': f'Mejora de {headline:+.1f}% con significancia estadística y throughput estable.',
            'headline_pct': headline,
        }
    return {
        'verdict': 'regression',
        'reason': f'Empeoramiento de {headline:+.1f}% con significancia estadística y throughput estable.',
        'headline_pct': headline,
    }


# =============================================================================
# MAIN COMPARISON FUNCTION
# =============================================================================

def compare_runs(
    run_a: str,
    run_b: str,
    out_dir: str,
    labels: Tuple[str, str] = ('Baseline', 'Optimized'),
    time_bin: int = 30,
    generate_all_plots: bool = True,
    use_sumo_tools: bool = False,
    extra_info: Optional[Dict] = None,
) -> Dict:
    """Comprehensive comparison of two simulation runs.

    Generates all available visualizations and statistical analyses.

    Args:
        run_a: Directory for baseline simulation
        run_b: Directory for optimized simulation
        out_dir: Output directory for plots and reports
        labels: Labels for the two runs
        time_bin: Time bin size for aggregations (seconds)
        generate_all_plots: Whether to generate all available plots
        use_sumo_tools: Whether to also run native SUMO visualization tools
        extra_info: Run configuration (seed, steps, mode…) embedded in the
            JSON and HTML reports for traceability

    Returns:
        Dictionary with analysis results and file paths
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {run_a} and {run_b}...")
    df_trip_a = _load_tripinfo(run_a)
    df_trip_b = _load_tripinfo(run_b)
    df_summary_a = _load_summary(run_a)
    df_summary_b = _load_summary(run_b)
    df_fcd_a = _load_fcd(run_a, sample_rate=5)
    df_fcd_b = _load_fcd(run_b, sample_rate=5)

    # Extract key arrays
    dur_a = df_trip_a['duration'].dropna().values if not df_trip_a.empty and 'duration' in df_trip_a.columns else np.array([])
    dur_b = df_trip_b['duration'].dropna().values if not df_trip_b.empty and 'duration' in df_trip_b.columns else np.array([])

    # Compute statistics
    print("Computing statistics...")
    stats_a = compute_trip_statistics(df_trip_a)
    stats_b = compute_trip_statistics(df_trip_b)
    summary_stats_a = compute_summary_statistics(df_summary_a)
    summary_stats_b = compute_summary_statistics(df_summary_b)

    generated_files = []

    if generate_all_plots:
        print("Generating visualizations...")

        # 1. Basic distribution comparisons
        plot_histogram_cdf_two(dur_a, dur_b, out_dir, filename='01_duration_hist_cdf.png')
        generated_files.append('01_duration_hist_cdf.png')

        plot_boxplot_two(dur_a, dur_b, out_dir, filename='02_duration_boxplot.png')
        generated_files.append('02_duration_boxplot.png')

        # 2. Violin plots
        plot_violin_comparison(dur_a, dur_b, out_dir, labels=labels,
                              metric_name='Travel Time (s)', filename='03_duration_violin.png')
        generated_files.append('03_duration_violin.png')

        if 'timeLoss' in df_trip_a.columns and 'timeLoss' in df_trip_b.columns:
            tl_a = df_trip_a['timeLoss'].dropna().values
            tl_b = df_trip_b['timeLoss'].dropna().values
            plot_violin_comparison(tl_a, tl_b, out_dir, labels=labels,
                                  metric_name='Time Loss (s)', filename='04_timeloss_violin.png')
            generated_files.append('04_timeloss_violin.png')

        # 3. Multi-metric violin
        plot_multi_metric_violin(df_trip_a, df_trip_b,
                                 ['duration', 'timeLoss', 'waitingTime', 'departDelay'],
                                 out_dir, labels=labels, filename='05_multi_metric_violin.png')
        generated_files.append('05_multi_metric_violin.png')

        # 4. Metric comparison bars
        if stats_a and stats_b:
            plot_metric_comparison_bars(stats_a, stats_b,
                                        ['duration', 'timeLoss', 'waitingTime', 'departDelay'],
                                        out_dir, labels=labels, filename='06_metric_comparison_bars.png')
            generated_files.append('06_metric_comparison_bars.png')

        # 5. Time series comparisons
        plot_time_series_comparison(df_trip_a, df_trip_b, 'depart', 'duration',
                                    out_dir, labels=labels, bin_size=time_bin,
                                    filename='07_duration_time_series.png')
        generated_files.append('07_duration_time_series.png')

        plot_time_series_comparison(df_trip_a, df_trip_b, 'depart', 'timeLoss',
                                    out_dir, labels=labels, bin_size=time_bin,
                                    filename='08_timeloss_time_series.png')
        generated_files.append('08_timeloss_time_series.png')

        # 6. Summary comparison (running, halting, etc.)
        plot_summary_comparison(df_summary_a, df_summary_b, out_dir, labels=labels,
                               filename='09_summary_comparison.png')
        generated_files.append('09_summary_comparison.png')

        # 7. Congestion timeline
        plot_congestion_timeline(df_summary_a, df_summary_b, out_dir, labels=labels,
                                filename='10_congestion_timeline.png')
        generated_files.append('10_congestion_timeline.png')

        # 8. Efficiency comparison
        plot_efficiency_comparison(df_trip_a, df_trip_b, out_dir, labels=labels,
                                   filename='11_efficiency_comparison.png')
        generated_files.append('11_efficiency_comparison.png')

        # 9. Speed distribution
        plot_speed_distribution_comparison(df_trip_a, df_trip_b, out_dir, labels=labels,
                                           filename='12_speed_distribution.png')
        generated_files.append('12_speed_distribution.png')

        # 10. Waiting time analysis
        plot_waiting_time_analysis(df_trip_a, df_trip_b, out_dir, labels=labels,
                                   filename='13_waiting_time_analysis.png')
        generated_files.append('13_waiting_time_analysis.png')

        # 11. Percentile comparison
        plot_percentile_comparison(dur_a, dur_b, out_dir, labels=labels,
                                   metric_name='duration', filename='14_percentile_comparison.png')
        generated_files.append('14_percentile_comparison.png')

        # 12. Correlation heatmaps
        plot_correlation_heatmap(df_trip_a, out_dir, label=labels[0],
                                filename='15_correlation_heatmap_A.png')
        generated_files.append('15_correlation_heatmap_A.png')
        plot_correlation_heatmap(df_trip_b, out_dir, label=labels[1],
                                filename='16_correlation_heatmap_B.png')
        generated_files.append('16_correlation_heatmap_B.png')

        # 13. FCD comparison (if available)
        if not df_fcd_a.empty or not df_fcd_b.empty:
            plot_fcd_comparison(df_fcd_a, df_fcd_b, out_dir, labels=labels,
                               filename='17_fcd_comparison.png', time_bin=time_bin)
            generated_files.append('17_fcd_comparison.png')

        # 14. Improvement summary
        if stats_a and stats_b:
            plot_improvement_summary(stats_a, stats_b, out_dir, labels=labels,
                                     filename='18_improvement_summary.png')
            generated_files.append('18_improvement_summary.png')

        # 15. Individual time series
        if not df_trip_a.empty and 'depart' in df_trip_a.columns:
            plot_time_series_mean(df_trip_a, 'depart', 'duration', out_dir,
                                 filename=f'19_time_series_mean_{labels[0]}.png', bin_size=time_bin)
            generated_files.append(f'19_time_series_mean_{labels[0]}.png')
        if not df_trip_b.empty and 'depart' in df_trip_b.columns:
            plot_time_series_mean(df_trip_b, 'depart', 'duration', out_dir,
                                 filename=f'20_time_series_mean_{labels[1]}.png', bin_size=time_bin)
            generated_files.append(f'20_time_series_mean_{labels[1]}.png')

        # 16. Incomplete trips analysis for BOTH runs (A: 21–24, B: 25–28)
        generated_files.extend(analyze_incomplete_trips(
            run_a, df_trip_a, out_dir, label=labels[0], tag='A', file_start=21))
        generated_files.extend(analyze_incomplete_trips(
            run_b, df_trip_b, out_dir, label=labels[1], tag='B', file_start=25))

        # 17. Paired per-vehicle evidence (29–32)
        if len(dur_a) > 0 and len(dur_b) > 0:
            _plot_qq_comparison(dur_a, dur_b, out_dir, labels, filename='29_qq_duration.png')
            generated_files.append('29_qq_duration.png')

        merged = _paired_frame(df_trip_a, df_trip_b, 'duration')
        if not merged.empty:
            _plot_paired_scatter(merged, 'duration', out_dir, labels,
                                 filename='30_paired_scatter.png')
            generated_files.append('30_paired_scatter.png')
            _plot_paired_delta_hist(merged['delta'].values, 'duration', out_dir, labels,
                                    filename='31_paired_delta_hist.png')
            generated_files.append('31_paired_delta_hist.png')

        _plot_throughput_timeline(df_trip_a, df_trip_b, out_dir, labels,
                                  filename='32_throughput_timeline.png')
        generated_files.append('32_throughput_timeline.png')

    # Statistical tests
    print("Running statistical tests...")
    statistical_results = {}

    if len(dur_a) > 0 and len(dur_b) > 0:
        # Permutation test
        pval_perm = permutation_test_mean(dur_a, dur_b, n_iter=5000)
        statistical_results['permutation_test_pvalue'] = pval_perm

        # Bootstrap CI
        ci_lo, ci_hi, point_est = bootstrap_ci_diff(dur_a, dur_b, n_boot=5000)
        statistical_results['bootstrap_ci_95'] = {'lower': ci_lo, 'upper': ci_hi, 'point_estimate': point_est}

        # Mann-Whitney U
        mw_stat, mw_pval = mann_whitney_test(dur_a, dur_b)
        statistical_results['mann_whitney_u'] = {'statistic': mw_stat, 'pvalue': mw_pval}

        # Effect size
        d = cohens_d(dur_a, dur_b)
        statistical_results['cohens_d'] = d

        # Basic descriptive comparison
        mean_diff = float(dur_a.mean() - dur_b.mean())
        pct_improvement = (mean_diff / dur_a.mean() * 100) if dur_a.mean() > 0 else 0
        statistical_results['mean_difference'] = mean_diff
        statistical_results['percent_improvement'] = pct_improvement

    # Paired per-vehicle statistics (strongest evidence: identical fleets)
    paired = paired_statistics(df_trip_a, df_trip_b, 'duration')
    statistical_results['paired'] = paired
    throughput = throughput_statistics(df_trip_a, df_trip_b)
    statistical_results['throughput'] = throughput
    # Tiempo total de sistema (métrica primaria insesgada por supervivencia)
    system_time = system_time_statistics(df_summary_a, df_summary_b)
    statistical_results['system_time'] = system_time
    # Veredicto compuesto único: JSON, CSV y HTML delegan aquí.
    verdict = compute_verdict(statistical_results)
    statistical_results['verdict'] = verdict

    # Use native SUMO tools if requested
    if use_sumo_tools:
        try:
            from .sumo_tools import check_sumo_tools_available, generate_all_sumo_plots
            if check_sumo_tools_available():
                sumo_out_dir = os.path.join(out_dir, 'sumo_native')
                sumo_files = generate_all_sumo_plots(run_a, run_b, sumo_out_dir, labels=labels)
                generated_files.extend([f'sumo_native/{os.path.basename(f)}' for f in sumo_files])
        except Exception as e:
            print(f"Warning: Could not generate SUMO native plots: {e}")

    # Generate reports
    print("Generating reports...")

    # CSV summary
    csv_path = Path(out_dir) / 'ab_summary.csv'
    _write_csv_summary(csv_path, stats_a, stats_b, labels, statistical_results)
    generated_files.append('ab_summary.csv')

    # JSON report
    json_path = Path(out_dir) / 'ab_report.json'
    report = {
        'labels': labels,
        'run_config': extra_info or {},
        'statistics': {
            labels[0]: stats_a,
            labels[1]: stats_b,
        },
        'summary_statistics': {
            labels[0]: summary_stats_a,
            labels[1]: summary_stats_b,
        },
        'statistical_tests': statistical_results,
        'generated_files': generated_files,
        'data_info': {
            labels[0]: {
                'tripinfo_count': len(df_trip_a),
                'summary_rows': len(df_summary_a),
                'fcd_rows': len(df_fcd_a),
            },
            labels[1]: {
                'tripinfo_count': len(df_trip_b),
                'summary_rows': len(df_summary_b),
                'fcd_rows': len(df_fcd_b),
            },
        }
    }
    _write_json_report(json_path, report)
    generated_files.append('ab_report.json')

    # HTML report (autocontenido, referencias relativas a los PNG)
    html_path = Path(out_dir) / 'ab_report.html'
    _write_html_report(
        html_path,
        labels=labels,
        stats_a=stats_a,
        stats_b=stats_b,
        statistical_results=statistical_results,
        paired=paired,
        throughput=throughput,
        run_config=extra_info or {},
        generated_files=generated_files,
    )
    generated_files.append('ab_report.html')

    print(f"Analysis complete! Generated {len(generated_files)} files in {out_dir}")

    return {
        'out_dir': str(out_dir),
        'csv': str(csv_path),
        'json': str(json_path),
        'html': str(html_path),
        'statistical_results': statistical_results,
        'stats_a': stats_a,
        'stats_b': stats_b,
        'generated_files': generated_files,
    }


def _write_csv_summary(
    path: Path,
    stats_a: Dict,
    stats_b: Dict,
    labels: Tuple[str, str],
    statistical_results: Dict
):
    """Write CSV summary of comparison results."""
    with open(path, 'w', newline='') as cf:
        writer = csv.writer(cf)

        # Header
        writer.writerow(['A/B Test Summary Report'])
        writer.writerow([])

        # Per-run statistics
        writer.writerow(['Metric Statistics'])
        writer.writerow(['label', 'metric', 'count', 'mean', 'median', 'std', 'min', 'max', 'q25', 'q75', 'q95'])

        for label, stats in [(labels[0], stats_a), (labels[1], stats_b)]:
            for metric, values in stats.items():
                if isinstance(values, dict) and 'mean' in values:
                    writer.writerow([
                        label, metric,
                        values.get('count', ''),
                        f"{values.get('mean', ''):.4f}" if values.get('mean') is not None else '',
                        f"{values.get('median', ''):.4f}" if values.get('median') is not None else '',
                        f"{values.get('std', ''):.4f}" if values.get('std') is not None else '',
                        f"{values.get('min', ''):.4f}" if values.get('min') is not None else '',
                        f"{values.get('max', ''):.4f}" if values.get('max') is not None else '',
                        f"{values.get('q25', ''):.4f}" if values.get('q25') is not None else '',
                        f"{values.get('q75', ''):.4f}" if values.get('q75') is not None else '',
                        f"{values.get('q95', ''):.4f}" if values.get('q95') is not None else '',
                    ])

        writer.writerow([])

        # Statistical test results
        writer.writerow(['Statistical Test Results'])

        if 'permutation_test_pvalue' in statistical_results:
            writer.writerow(['Permutation Test (H0: equal means)', f"p={statistical_results['permutation_test_pvalue']:.6f}"])

        if 'mann_whitney_u' in statistical_results:
            mw = statistical_results['mann_whitney_u']
            writer.writerow(['Mann-Whitney U Test', f"U={mw['statistic']:.2f}, p={mw['pvalue']:.6f}"])

        if 'cohens_d' in statistical_results:
            d = statistical_results['cohens_d']
            effect_size = 'small' if abs(d) < 0.5 else 'medium' if abs(d) < 0.8 else 'large'
            writer.writerow(['Cohen\'s d Effect Size', f"d={d:.4f} ({effect_size})"])

        if 'bootstrap_ci_95' in statistical_results:
            ci = statistical_results['bootstrap_ci_95']
            writer.writerow(['Bootstrap 95% CI for Mean Diff', f"[{ci['lower']:.4f}, {ci['upper']:.4f}]"])
            writer.writerow(['Point Estimate (A - B)', f"{ci['point_estimate']:.4f}"])

        if 'percent_improvement' in statistical_results:
            pct = statistical_results['percent_improvement']
            writer.writerow(['Percent Improvement', f"{pct:+.2f}%"])

        writer.writerow([])

        # Interpretation
        writer.writerow(['Interpretation'])
        if 'permutation_test_pvalue' in statistical_results:
            p = statistical_results['permutation_test_pvalue']
            if p < 0.01:
                writer.writerow(['', 'Strong evidence of difference (p < 0.01)'])
            elif p < 0.05:
                writer.writerow(['', 'Moderate evidence of difference (p < 0.05)'])
            else:
                writer.writerow(['', 'Insufficient evidence of difference (p >= 0.05)'])

        # Veredicto compuesto insesgado (throughput + tiempo de sistema)
        writer.writerow([])
        writer.writerow(['Veredicto (insesgado por supervivencia)'])
        verdict = statistical_results.get('verdict') or compute_verdict(statistical_results)
        writer.writerow(['verdict', verdict.get('verdict')])
        headline = verdict.get('headline_pct')
        writer.writerow(['headline_pct', f"{headline:+.2f}%" if isinstance(headline, (int, float)) and headline == headline else ''])
        writer.writerow(['reason', verdict.get('reason')])


def _write_json_report(path: Path, report: Dict):
    """Write JSON report with full analysis results."""

    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        elif pd.isna(obj):
            return None
        return obj

    serializable_report = convert_to_serializable(report)

    with open(path, 'w') as f:
        json.dump(serializable_report, f, indent=2)


# =============================================================================
# HTML REPORT
# =============================================================================

_HTML_SECTIONS = [
    ('Evidencia pareada por vehículo (la más fuerte)', range(29, 33)),
    ('Distribuciones', range(1, 6)),
    ('Comparación de métricas', [6, 11, 12, 13, 14, 18]),
    ('Series temporales', [7, 8, 9, 10, 19, 20]),
    ('Correlaciones y FCD', [15, 16, 17]),
    ('Viajes incompletos — corrida A', range(21, 25)),
    ('Viajes incompletos — corrida B', range(25, 29)),
]


def _fmt(v, spec='.2f', default='—'):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return format(v, spec)
    except (TypeError, ValueError):
        return default


_VERDICT_CSS = {
    'improvement': 'green',
    'misleading_regression': 'green',
    'regression': 'red',
    'misleading_improvement': 'red',
    'tie': 'gray',
}


def _verdict(statistical_results: Dict, paired: Dict) -> Tuple[str, str, str]:
    """Return (css_class, title, detail) for the verdict banner.

    Delega en `compute_verdict`, la función única de veredicto compuesto
    (throughput + tiempo de sistema insesgado + significancia) que también
    usan el JSON, el CSV y el resumen de campaña.
    """
    pct = statistical_results.get('percent_improvement')
    if pct is None:
        return ('gray', 'Sin datos suficientes',
                'No hay viajes completados en ambas corridas para comparar.')

    merged = dict(statistical_results)
    if paired:
        merged['paired'] = paired
    v = compute_verdict(merged)

    headline = v['headline_pct']
    hp = f'{headline:+.1f}%' if isinstance(headline, (int, float)) and headline == headline else '—'

    titles = {
        'improvement': f'Mejora significativa: {hp} en tiempo de sistema',
        'misleading_regression': f'Mejora real (el baseline colapsó): {hp} en tiempo de sistema',
        'regression': f'Empeoramiento: {hp} en tiempo de sistema',
        'misleading_improvement': f'Resultado engañoso: media {pct:+.1f}% pero throughput cae',
        'tie': f'Resultado no concluyente ({hp})',
    }

    css_class = _VERDICT_CSS[v['verdict']]
    title = titles[v['verdict']]
    detail = v['reason']
    if paired and v['verdict'] in ('improvement', 'misleading_regression'):
        detail += (f' {paired["pct_improved"]:.1f}% de los {paired["n_paired"]} vehículos '
                   f'pareados mejora (Wilcoxon p={_fmt(paired["wilcoxon_pvalue"], ".2e")}).')
    return (css_class, title, detail)


def _write_html_report(
    path: Path,
    *,
    labels: Tuple[str, str],
    stats_a: Dict,
    stats_b: Dict,
    statistical_results: Dict,
    paired: Dict,
    throughput: Dict,
    run_config: Dict,
    generated_files: List[str],
):
    """Write a self-contained HTML report (relative <img> refs, no external deps)."""
    from datetime import datetime

    css_class, verdict_title, verdict_detail = _verdict(statistical_results, paired)

    # --- comparabilidad de muestras (sesgo de supervivencia) ---
    completed_a = throughput.get('completed_a') if throughput else None
    completed_b = throughput.get('completed_b') if throughput else None
    samples_noncomparable = False
    if completed_a is not None and completed_b is not None and max(completed_a, completed_b) > 0:
        count_diff_pct = abs(completed_a - completed_b) / max(completed_a, completed_b) * 100
        samples_noncomparable = count_diff_pct > 10

    header_n_a = f' (n={completed_a})' if completed_a is not None else ''
    header_n_b = f' (n={completed_b})' if completed_b is not None else ''

    survivorship_note = ''
    if samples_noncomparable:
        survivorship_note = (
            f'<p class="warn">⚠️ Las métricas por viaje solo incluyen viajes completados: '
            f'{labels[0]} completó {completed_a} y {labels[1]} completó {completed_b} '
            f'(Δ throughput {_fmt(throughput.get("throughput_change_pct"), "+.1f")}%). '
            f'El lado que colapsa promedia solo los viajes rápidos que escaparon.</p>'
        )

    # --- tabla de métricas clave A vs B ---
    metric_rows = []
    for metric in ('duration', 'timeLoss', 'waitingTime', 'departDelay'):
        sa, sb = stats_a.get(metric), stats_b.get(metric)
        if not (isinstance(sa, dict) and isinstance(sb, dict)):
            continue
        mean_a, mean_b = sa.get('mean'), sb.get('mean')
        median_a, median_b = sa.get('median'), sb.get('median')
        if samples_noncomparable and median_a:
            pct = (median_a - median_b) / median_a * 100
            tooltip = (
                f'title="muestras no comparables (A completó {completed_a}, '
                f'B completó {completed_b}): mejora calculada sobre medianas; '
                f'la media está dominada por sesgo de supervivencia"'
            )
            improvement_cell = f'<td class="{{color}}" {tooltip}>~{_fmt(pct, "+.2f")}%</td>'
        else:
            pct = ((mean_a - mean_b) / mean_a * 100) if mean_a else None
            improvement_cell = '<td class="{color}">' + f'{_fmt(pct, "+.2f")}%</td>'
        color = 'pos' if (pct or 0) > 0 else 'neg'
        improvement_cell = improvement_cell.format(color=color)
        metric_rows.append(
            f'<tr><td>{metric}</td>'
            f'<td>{_fmt(mean_a)}</td><td>{_fmt(mean_b)}</td>'
            f'<td>{_fmt(sa.get("median"))}</td><td>{_fmt(sb.get("median"))}</td>'
            f'<td>{_fmt(sa.get("q95"))}</td><td>{_fmt(sb.get("q95"))}</td>'
            f'{improvement_cell}</tr>'
        )

    # --- tabla de tests estadísticos ---
    test_rows = []
    p = statistical_results.get('permutation_test_pvalue')
    if p is not None:
        interp = ('Evidencia fuerte de diferencia real' if p < 0.01 else
                  'Evidencia moderada de diferencia real' if p < 0.05 else
                  'Diferencia no significativa')
        test_rows.append(('Test de permutación (medias)', f'p = {_fmt(p, ".2e")}', interp))
    mw = statistical_results.get('mann_whitney_u')
    if mw:
        test_rows.append(('Mann-Whitney U', f'U = {_fmt(mw.get("statistic"), ".0f")}, '
                          f'p = {_fmt(mw.get("pvalue"), ".2e")}',
                          'Test no paramétrico sobre distribuciones completas'))
    d = statistical_results.get('cohens_d')
    if d is not None:
        size = 'pequeño' if abs(d) < 0.5 else 'mediano' if abs(d) < 0.8 else 'grande'
        test_rows.append(("Cohen's d", f'd = {_fmt(d, ".3f")}', f'Tamaño de efecto {size}'))
    ci = statistical_results.get('bootstrap_ci_95')
    if ci:
        test_rows.append(('Bootstrap IC 95% (A − B)',
                          f'[{_fmt(ci.get("lower"))}, {_fmt(ci.get("upper"))}] s',
                          'Si el intervalo no cruza 0, la diferencia es robusta'))
    if paired:
        test_rows.append(('Wilcoxon pareado por vehículo',
                          f'p = {_fmt(paired.get("wilcoxon_pvalue"), ".2e")}',
                          f'{_fmt(paired.get("pct_improved"), ".1f")}% de {paired.get("n_paired")} '
                          f'vehículos mejora; Δ mediana = {_fmt(paired.get("median_delta"), "+.1f")}s'))
    test_html = ''.join(f'<tr><td>{n}</td><td>{v}</td><td>{i}</td></tr>' for n, v, i in test_rows)

    # --- throughput ---
    thr_html = ''
    if throughput and throughput.get('completed_a') is not None:
        thr_html = (
            f'<p><b>Throughput:</b> {labels[0]} completó {throughput["completed_a"]} viajes, '
            f'{labels[1]} completó {throughput["completed_b"]} '
            f'({_fmt(throughput.get("throughput_change_pct"), "+.1f")}%).</p>'
        )

    # --- run config ---
    config_html = ''
    if run_config:
        rows = ''.join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in run_config.items())
        config_html = (f'<h2>Configuración del experimento</h2>'
                       f'<table><tr><th>Parámetro</th><th>Valor</th></tr>{rows}</table>')

    # --- imágenes agrupadas por sección ---
    pngs = [f for f in generated_files if f.endswith('.png')]

    def _num(f):
        try:
            return int(f.split('_')[0])
        except ValueError:
            return -1

    sections_html = []
    used = set()
    for title, nums in _HTML_SECTIONS:
        files = [f for f in pngs if _num(f) in nums]
        if not files:
            continue
        used.update(files)
        imgs = ''.join(f'<figure><img src="{f}" alt="{f}"><figcaption>{f}</figcaption></figure>'
                       for f in sorted(files, key=_num))
        sections_html.append(f'<h2>{title}</h2><div class="grid">{imgs}</div>')
    leftover = [f for f in pngs if f not in used]
    if leftover:
        imgs = ''.join(f'<figure><img src="{f}" alt="{f}"><figcaption>{f}</figcaption></figure>'
                       for f in sorted(leftover))
        sections_html.append(f'<h2>Otros gráficos</h2><div class="grid">{imgs}</div>')

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Reporte A/B — {labels[0]} vs {labels[1]}</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 1100px;
        color: #222; line-height: 1.5; padding: 0 1rem; }}
 h1 {{ font-size: 1.6rem; }} h2 {{ margin-top: 2.2rem; border-bottom: 1px solid #ddd; }}
 .banner {{ padding: 1rem 1.4rem; border-radius: 8px; margin: 1rem 0; }}
 .banner.green {{ background: #e6f6ec; border: 1px solid #34a853; }}
 .banner.red {{ background: #fdecea; border: 1px solid #ea4335; }}
 .banner.gray {{ background: #f1f3f4; border: 1px solid #9aa0a6; }}
 .banner h2 {{ margin: 0 0 .4rem; border: none; }}
 table {{ border-collapse: collapse; width: 100%; margin: .8rem 0; font-size: .92rem; }}
 th, td {{ border: 1px solid #ddd; padding: .45rem .6rem; text-align: left; }}
 th {{ background: #f6f8fa; }}
 td.pos {{ color: #188038; font-weight: 600; }} td.neg {{ color: #c5221f; font-weight: 600; }}
 p.warn {{ background: #fff8e1; border: 1px solid #f9ab00; border-radius: 6px;
           padding: .6rem .9rem; }}
 .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 1rem; }}
 figure {{ margin: 0; }} img {{ max-width: 100%; border: 1px solid #eee; border-radius: 4px; }}
 figcaption {{ font-size: .78rem; color: #777; text-align: center; }}
 footer {{ margin-top: 3rem; font-size: .8rem; color: #999; }}
</style>
</head>
<body>
<h1>Reporte automatizado A/B — {labels[0]} vs {labels[1]}</h1>
<p>Generado el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Ambas corridas usan
demanda, red, semilla y configuración idénticas; la única diferencia es la optimización
de semáforos aplicada en «{labels[1]}».</p>

<div class="banner {css_class}">
<h2>{verdict_title}</h2>
<p>{verdict_detail}</p>
</div>

{thr_html}

<h2>Métricas clave por viaje</h2>
{survivorship_note}
<table>
<tr><th>Métrica</th><th>Media {labels[0]}{header_n_a}</th><th>Media {labels[1]}{header_n_b}</th>
<th>Mediana {labels[0]}</th><th>Mediana {labels[1]}</th>
<th>p95 {labels[0]}</th><th>p95 {labels[1]}</th><th>Mejora</th></tr>
{''.join(metric_rows)}
</table>

<h2>Tests estadísticos</h2>
<table>
<tr><th>Test</th><th>Resultado</th><th>Interpretación</th></tr>
{test_html}
</table>

{config_html}

{''.join(sections_html)}

<footer>Datos completos en <code>ab_report.json</code> y <code>ab_summary.csv</code>
(mismo directorio). Generado por traffic-sim.</footer>
</body>
</html>
"""
    Path(path).write_text(html, encoding='utf-8')


# =============================================================================
# QUICK COMPARISON FUNCTIONS
# =============================================================================

def quick_compare(run_a: str, run_b: str, metric: str = 'duration') -> Dict:
    """Quick comparison returning just key statistics.

    Useful for programmatic access without generating plots.
    """
    df_a = _load_tripinfo(run_a)
    df_b = _load_tripinfo(run_b)

    if df_a.empty or df_b.empty or metric not in df_a.columns or metric not in df_b.columns:
        return {'error': 'Data not available'}

    data_a = df_a[metric].dropna().values
    data_b = df_b[metric].dropna().values

    return {
        'A': {
            'count': len(data_a),
            'mean': float(np.mean(data_a)),
            'median': float(np.median(data_a)),
            'std': float(np.std(data_a)),
        },
        'B': {
            'count': len(data_b),
            'mean': float(np.mean(data_b)),
            'median': float(np.median(data_b)),
            'std': float(np.std(data_b)),
        },
        'difference': {
            'mean_diff': float(np.mean(data_a) - np.mean(data_b)),
            'percent_change': float((np.mean(data_a) - np.mean(data_b)) / np.mean(data_a) * 100) if np.mean(data_a) > 0 else 0,
        },
        'tests': {
            'permutation_pvalue': permutation_test_mean(data_a, data_b, n_iter=2000),
            'cohens_d': cohens_d(data_a, data_b),
        }
    }


def compare_multiple_metrics(run_a: str, run_b: str) -> pd.DataFrame:
    """Compare all available metrics and return as DataFrame.

    Useful for quick tabular comparison.
    """
    df_a = _load_tripinfo(run_a)
    df_b = _load_tripinfo(run_b)

    metrics = ['duration', 'timeLoss', 'waitingTime', 'departDelay', 'routeLength']
    results = []

    for metric in metrics:
        if metric not in df_a.columns or metric not in df_b.columns:
            continue

        data_a = df_a[metric].dropna().values
        data_b = df_b[metric].dropna().values

        if len(data_a) == 0 or len(data_b) == 0:
            continue

        mean_a = np.mean(data_a)
        mean_b = np.mean(data_b)
        pct_change = ((mean_a - mean_b) / mean_a * 100) if mean_a > 0 else 0

        results.append({
            'metric': metric,
            'mean_A': mean_a,
            'mean_B': mean_b,
            'diff': mean_a - mean_b,
            'pct_improvement': pct_change,
            'p_value': permutation_test_mean(data_a, data_b, n_iter=1000),
        })

    return pd.DataFrame(results)
