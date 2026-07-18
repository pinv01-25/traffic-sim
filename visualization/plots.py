"""Plotting helpers for SUMO visualization outputs.

Comprehensive matplotlib-based functions for single runs and A/B comparisons.
Accepts DataFrames from parsers and writes PNG files to an output directory.
"""
import os
from typing import Dict, List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style defaults
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# =============================================================================
# SURVIVORSHIP-BIAS ANNOTATION HELPERS
# =============================================================================
# tripinfo.xml solo contiene viajes COMPLETADOS. Cuando un lado del A/B
# colapsa (gridlock o bloqueo de inserción), su muestra queda dominada por
# los viajes rápidos que escaparon temprano: cualquier histograma, CDF,
# boxplot, violín o serie temporal que compare A vs B sin mostrar los n por
# lado está sesgado a favor del lado colapsado. Estos helpers centralizan la
# anotación mínima (n en leyenda + nota en título) para todos los gráficos
# que comparan muestras de tripinfo.

#: Umbral de diferencia relativa de conteos (%) por encima del cual las
#: muestras se consideran "no comparables" y se anotan.
SAMPLE_NONCOMPARABLE_THRESHOLD_PCT = 10.0

#: Bins con menos de esta cantidad de observaciones se consideran poco
#: fiables en series temporales (posible cola vacía por colapso).
MIN_BIN_SAMPLES = 5


def samples_noncomparable(n_a: int, n_b: int,
                           threshold_pct: float = SAMPLE_NONCOMPARABLE_THRESHOLD_PCT) -> bool:
    """True si los conteos de dos muestras difieren lo suficiente como para
    que una comparación directa de distribuciones esté sesgada por
    supervivencia (un lado colapsado solo aporta los viajes rápidos que
    escaparon)."""
    if n_a is None or n_b is None:
        return False
    m = max(n_a, n_b)
    if m == 0:
        return False
    return abs(n_a - n_b) / m * 100 > threshold_pct


def legend_label_with_n(label: str, n: int) -> str:
    """Legend label annotated with its sample size, e.g. 'A (n=349)'."""
    return f'{label} (n={n})'


def title_with_sample_note(title: str, n_a: int, n_b: int) -> str:
    """Append a non-comparability note to a plot title when the two sample
    sizes differ by more than SAMPLE_NONCOMPARABLE_THRESHOLD_PCT.

    La nota apunta a 32_throughput_timeline.png (y no a "ver nota" a secas)
    a propósito: este gráfico puede abrirse solo, fuera del HTML, y ese es
    el único gráfico insesgado que cuenta a TODOS los vehículos — el que de
    verdad decide el resultado cuando las distribuciones de tripinfo no son
    comparables entre A y B.
    """
    if samples_noncomparable(n_a, n_b):
        return (f'{title}\nn={n_a} vs n={n_b} — muestras no comparables\n'
                '(resultado real: ver 32_throughput_timeline.png)')
    return title


def plot_summary_timeline(df: pd.DataFrame, out_dir: str):
    _ensure_dir(out_dir)
    if df.empty:
        return
    # Try to pick a sensible time column
    time_col = None
    for candidate in ('begin', 'time', 't'):
        if candidate in df.columns:
            time_col = candidate
            break
    if time_col is None and 'time' not in df.columns:
        # attempt to use first numeric column as index
        numeric_cols = df.select_dtypes('number').columns.tolist()
        if numeric_cols:
            time_col = numeric_cols[0]
    try:
        df_plot = df.set_index(time_col) if time_col is not None else df
    except Exception:
        df_plot = df

    # Plot a few columns if present
    candidates = [c for c in ['running', 'halting', 'entered', 'left', 'teleported'] if c in df_plot.columns]
    if not candidates:
        # fallback: plot first 3 numeric columns
        candidates = df_plot.select_dtypes('number').columns[:3].tolist()

    if not candidates:
        return

    ax = df_plot[candidates].plot(figsize=(10, 4), title='Summary timeline')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('value')
    fig = ax.get_figure()
    out_file = os.path.join(out_dir, 'summary_timeline.png')
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


def plot_tripinfo_histograms(df: pd.DataFrame, out_dir: str):
    _ensure_dir(out_dir)
    if df.empty:
        return
    # Plot duration and timeLoss histograms side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if 'duration' in df.columns:
        axes[0].hist(df['duration'].dropna(), bins=30)
        axes[0].set_title('Travel time (s)')
        axes[0].set_xlabel('duration (s)')
    if 'timeLoss' in df.columns:
        axes[1].hist(df['timeLoss'].dropna(), bins=30)
        axes[1].set_title('Time loss (s)')
        axes[1].set_xlabel('timeLoss (s)')
    out_file = os.path.join(out_dir, 'tripinfo_distributions.png')
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


def plot_depart_delay_scatter(df: pd.DataFrame, out_dir: str):
    _ensure_dir(out_dir)
    if df.empty:
        return
    if 'depart' not in df.columns or 'departDelay' not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df['depart'], df['departDelay'], s=8, alpha=0.6)
    ax.set_xlabel('Depart time (s)')
    ax.set_ylabel('Depart delay (s)')
    ax.set_title('Depart time vs Depart delay')
    out_file = os.path.join(out_dir, 'depart_delay_scatter.png')
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


def plot_network_traffic_lights(net_file: str, out_dir: str):
    """Plot simple scatter of traffic light junctions from a .net.xml file.

    This is a lightweight fallback when no SUMO outputs are available.
    """
    _ensure_dir(out_dir)
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(net_file)
        root = tree.getroot()
        coords = []
        for junction in root.findall('junction'):
            if junction.get('type') == 'traffic_light':
                try:
                    x = float(junction.get('x', 0.0))
                    y = float(junction.get('y', 0.0))
                    jid = junction.get('id')
                    coords.append((x, y, jid))
                except Exception:
                    continue
        if not coords:
            return
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        labels = [c[2] for c in coords]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(xs, ys, c='red', s=20)
        for xi, yi, lab in zip(xs, ys, labels, strict=False):
            ax.text(xi, yi, lab, fontsize=6)
        ax.set_title('Traffic lights (network)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        out_file = os.path.join(out_dir, 'net_traffic_lights.png')
        fig.tight_layout()
        fig.savefig(out_file)
        plt.close(fig)
    except Exception:
        # don't raise — fallback is best-effort
        return


def plot_histogram_cdf_two(dur_a, dur_b, out_dir: str, filename: str = 'duration_hist_cdf.png'):
    _ensure_dir(out_dir)
    import numpy as _np
    if len(dur_a) == 0 and len(dur_b) == 0:
        return
    n_a, n_b = len(dur_a), len(dur_b)
    label_a, label_b = legend_label_with_n('A', n_a), legend_label_with_n('B', n_b)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    # Histogram (overlaid)
    ax[0].hist(dur_a, bins=40, density=True, alpha=0.5, label=label_a)
    ax[0].hist(dur_b, bins=40, density=True, alpha=0.5, label=label_b)
    ax[0].set_title(title_with_sample_note('Travel time distribution (density)', n_a, n_b))
    ax[0].set_xlabel('duration (s)')
    ax[0].legend()

    # Empirical CDFs
    def _ecdf(x):
        x = _np.sort(_np.asarray(x))
        y = _np.arange(1, len(x) + 1) / len(x) if len(x) > 0 else _np.array([])
        return x, y

    xa, ya = _ecdf(dur_a)
    xb, yb = _ecdf(dur_b)
    if len(xa):
        ax[1].step(xa, ya, where='post', label=label_a)
    if len(xb):
        ax[1].step(xb, yb, where='post', label=label_b)
    ax[1].set_title(title_with_sample_note('Empirical CDF of travel time', n_a, n_b))
    ax[1].set_xlabel('duration (s)')
    ax[1].set_ylabel('CDF')
    ax[1].legend()

    out_file = os.path.join(out_dir, filename)
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


def plot_boxplot_two(dur_a, dur_b, out_dir: str, filename: str = 'duration_boxplot.png'):
    _ensure_dir(out_dir)
    import numpy as _np
    data = []
    labels = []
    if len(dur_a):
        data.append(_np.asarray(dur_a))
        labels.append(legend_label_with_n('A', len(dur_a)))
    if len(dur_b):
        data.append(_np.asarray(dur_b))
        labels.append(legend_label_with_n('B', len(dur_b)))
    if not data:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_title(title_with_sample_note('Travel time boxplot', len(dur_a), len(dur_b)))
    ax.set_ylabel('duration (s)')
    out_file = os.path.join(out_dir, filename)
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


def plot_time_series_mean(df: 'pd.DataFrame', time_col: str, value_col: str, out_dir: str, filename: str = 'time_series_mean.png', bin_size: int = 60):
    _ensure_dir(out_dir)
    if df is None or df.empty:
        return
    if time_col not in df.columns or value_col not in df.columns:
        return
    # Bin by integer time windows
    times = (df[time_col].fillna(0).astype(float) // bin_size).astype(int) * bin_size
    df2 = df.copy()
    df2['time_bin'] = times
    grouped = df2.groupby('time_bin')[value_col].agg(['mean', 'count'])
    fig, ax = plt.subplots(figsize=(10, 4))
    thin = grouped['count'] < MIN_BIN_SAMPLES
    ax.plot(grouped.index, grouped['mean'].where(~thin))
    title = f'Mean {value_col} per {bin_size}s (n={len(df)})'
    if thin.any():
        # Bins tardíos con pocos viajes TERMINADOS ahí (posible colapso) se
        # marcan punteados para no leerse como una mejora real.
        ax.plot(grouped.index, grouped['mean'].where(thin), linestyle=':', alpha=0.7,
                color=ax.lines[0].get_color())
        title += f'\n(punteado = bin con <{MIN_BIN_SAMPLES} viajes)'
    ax.set_title(title)
    ax.set_xlabel('time (s)')
    ax.set_ylabel(value_col)
    out_file = os.path.join(out_dir, filename)
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


# =============================================================================
# A/B COMPARISON PLOTS
# =============================================================================

def plot_metric_comparison_bars(
    stats_a: Dict,
    stats_b: Dict,
    metrics: List[str],
    out_dir: str,
    labels: Tuple[str, str] = ('A', 'B'),
    filename: str = 'metric_comparison_bars.png'
):
    """Bar chart comparing multiple metrics between two runs.

    Args:
        stats_a, stats_b: Dict with metric names as keys, each containing 'mean', 'std'
        metrics: List of metric names to compare
        out_dir: Output directory
        labels: Labels for the two runs
        filename: Output filename
    """
    _ensure_dir(out_dir)

    available_metrics = [m for m in metrics if m in stats_a and m in stats_b]
    if not available_metrics:
        return

    x = np.arange(len(available_metrics))
    width = 0.35

    means_a = [stats_a[m].get('mean', 0) for m in available_metrics]
    means_b = [stats_b[m].get('mean', 0) for m in available_metrics]
    stds_a = [stats_a[m].get('std', 0) for m in available_metrics]
    stds_b = [stats_b[m].get('std', 0) for m in available_metrics]

    fig, ax = plt.subplots(figsize=(max(10, len(available_metrics) * 2), 6))

    # 'count' puede diferir por métrica (NaNs distintos); se usa 'duration'
    # como referencia del tamaño de muestra por corrida si está disponible.
    n_a = stats_a.get('duration', {}).get('count') or (available_metrics and stats_a[available_metrics[0]].get('count'))
    n_b = stats_b.get('duration', {}).get('count') or (available_metrics and stats_b[available_metrics[0]].get('count'))

    ax.bar(x - width/2, means_a, width, yerr=stds_a, label=legend_label_with_n(labels[0], n_a),
                    capsize=5, color='steelblue', alpha=0.8)
    ax.bar(x + width/2, means_b, width, yerr=stds_b, label=legend_label_with_n(labels[1], n_b),
                    capsize=5, color='coral', alpha=0.8)

    ax.set_ylabel('Value')
    ax.set_title(title_with_sample_note('Metric Comparison: A vs B', n_a, n_b))
    ax.set_xticks(x)
    ax.set_xticklabels(available_metrics, rotation=45, ha='right')
    ax.legend()

    # Add percentage difference annotations. Igual que en
    # plot_improvement_summary: cuando el conteo por métrica difiere >10%
    # (lado colapsado), la media está dominada por sesgo de supervivencia
    # y se reemplaza por la mediana — si no, departDelay puede mostrar un
    # "+281%" que en realidad es una mejora real leída al revés (el
    # baseline solo insertó los pocos vehículos con poca espera de origen).
    for i, metric in enumerate(available_metrics):
        sa, sb = stats_a[metric], stats_b[metric]
        noncomparable = samples_noncomparable(sa.get('count'), sb.get('count'))
        ma, mb = (sa.get('median', 0), sb.get('median', 0)) if noncomparable else (means_a[i], means_b[i])
        if ma != 0:
            pct_diff = ((mb - ma) / ma) * 100
            color = 'green' if pct_diff < 0 else 'red'
            prefix = '~' if noncomparable else ''
            ax.annotate(f'{prefix}{pct_diff:+.1f}%', xy=(i, max(means_a[i], means_b[i])),
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', fontsize=8, color=color)

    out_file = os.path.join(out_dir, filename)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_violin_comparison(
    data_a: np.ndarray,
    data_b: np.ndarray,
    out_dir: str,
    labels: Tuple[str, str] = ('A', 'B'),
    metric_name: str = 'duration',
    filename: str = 'violin_comparison.png'
):
    """Violin plot comparing distributions between two runs.

    Shows the full distribution shape, not just summary statistics.
    """
    _ensure_dir(out_dir)

    if len(data_a) == 0 and len(data_b) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    data_to_plot = []
    positions = []
    if len(data_a) > 0:
        data_to_plot.append(data_a)
        positions.append(1)
    if len(data_b) > 0:
        data_to_plot.append(data_b)
        positions.append(2)

    parts = ax.violinplot(data_to_plot, positions=positions, showmeans=True,
                          showmedians=True, showextrema=True)

    # Color the violins
    colors = ['steelblue', 'coral']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    n_a, n_b = len(data_a), len(data_b)
    tick_a = legend_label_with_n(labels[0], n_a) if 1 in positions else ''
    tick_b = legend_label_with_n(labels[1], n_b) if 2 in positions else ''
    ax.set_xticks(positions)
    ax.set_xticklabels([tick_a, tick_b][:len(positions)])
    ax.set_ylabel(metric_name)
    ax.set_title(title_with_sample_note(
        f'{metric_name.replace("_", " ").title()} Distribution Comparison', n_a, n_b))

    # Add statistics text
    stats_text = []
    if len(data_a) > 0:
        stats_text.append(f"{labels[0]} (n={n_a}): μ={np.mean(data_a):.2f}, σ={np.std(data_a):.2f}")
    if len(data_b) > 0:
        stats_text.append(f"{labels[1]} (n={n_b}): μ={np.mean(data_b):.2f}, σ={np.std(data_b):.2f}")
    ax.text(0.02, 0.98, '\n'.join(stats_text), transform=ax.transAxes,
            verticalalignment='top', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    out_file = os.path.join(out_dir, filename)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_multi_metric_violin(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    metrics: List[str],
    out_dir: str,
    labels: Tuple[str, str] = ('A', 'B'),
    filename: str = 'multi_metric_violin.png'
):
    """Violin plots for multiple metrics side by side."""
    _ensure_dir(out_dir)

    available_metrics = [m for m in metrics if m in df_a.columns and m in df_b.columns]
    if not available_metrics:
        return

    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))

    if n_metrics == 1:
        axes = [axes]

    n_a_overall = len(df_a) if not df_a.empty else 0
    n_b_overall = len(df_b) if not df_b.empty else 0

    for i, metric in enumerate(available_metrics):
        ax = axes[i]
        data_a = df_a[metric].dropna().values
        data_b = df_b[metric].dropna().values

        label_a = legend_label_with_n(labels[0], len(data_a))
        label_b = legend_label_with_n(labels[1], len(data_b))

        data_combined = []
        group_labels = []
        if len(data_a) > 0:
            data_combined.extend(data_a)
            group_labels.extend([label_a] * len(data_a))
        if len(data_b) > 0:
            data_combined.extend(data_b)
            group_labels.extend([label_b] * len(data_b))

        if data_combined:
            plot_df = pd.DataFrame({'value': data_combined, 'group': group_labels})
            sns.violinplot(data=plot_df, x='group', y='value', ax=ax,
                          palette=['steelblue', 'coral'])
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_xlabel('')
            ax.set_ylabel('')

    suptitle = title_with_sample_note('Distribution Comparison Across Metrics',
                                       n_a_overall, n_b_overall)
    fig.suptitle(suptitle, fontsize=12)
    out_file = os.path.join(out_dir, filename)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_time_series_comparison(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    time_col: str,
    value_col: str,
    out_dir: str,
    labels: Tuple[str, str] = ('A', 'B'),
    bin_size: int = 30,
    filename: str = 'time_series_comparison.png',
    show_ci: bool = True
):
    """Time series comparison with confidence intervals.

    Plots mean values over time for both runs with optional confidence bands.
    """
    _ensure_dir(out_dir)

    if df_a.empty and df_b.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    # Los bins tardíos del lado que colapsa (gridlock o bloqueo de
    # inserción) reciben pocos o ningún viaje TERMINADO en ese intervalo:
    # se marcan con línea punteada para no leerse como "mejora" cuando en
    # realidad es ausencia de datos.
    n_a = int(df_a[value_col].dropna().shape[0]) if not df_a.empty and value_col in df_a.columns else 0
    n_b = int(df_b[value_col].dropna().shape[0]) if not df_b.empty and value_col in df_b.columns else 0

    for df, label, color in [(df_a, legend_label_with_n(labels[0], n_a), 'steelblue'),
                              (df_b, legend_label_with_n(labels[1], n_b), 'coral')]:
        if df.empty or time_col not in df.columns or value_col not in df.columns:
            continue

        df_temp = df.copy()
        df_temp['time_bin'] = (df_temp[time_col] // bin_size) * bin_size

        grouped = df_temp.groupby('time_bin')[value_col].agg(['mean', 'std', 'count'])
        grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])
        grouped['ci_lo'] = grouped['mean'] - 1.96 * grouped['se']
        grouped['ci_hi'] = grouped['mean'] + 1.96 * grouped['se']

        thin = grouped['count'] < MIN_BIN_SAMPLES
        # Segmento normal (línea sólida)
        solid = grouped['mean'].where(~thin)
        ax.plot(grouped.index, solid, label=label, color=color, linewidth=2)
        # Bins con <5 muestras: línea punteada, sin entrar en la leyenda
        if thin.any():
            dashed = grouped['mean'].where(thin)
            ax.plot(grouped.index, dashed, color=color, linewidth=2, linestyle=':',
                    alpha=0.7)

        if show_ci and len(grouped) > 0:
            ax.fill_between(grouped.index, grouped['ci_lo'], grouped['ci_hi'],
                           color=color, alpha=0.2)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(value_col.replace('_', ' ').title())
    title = f'{value_col.replace("_", " ").title()} Over Time'
    title = title_with_sample_note(title, n_a, n_b)
    title += f'\n(punteado = bin con <{MIN_BIN_SAMPLES} viajes)'
    ax.set_title(title)
    ax.legend()

    out_file = os.path.join(out_dir, filename)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_summary_comparison(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    out_dir: str,
    labels: Tuple[str, str] = ('A', 'B'),
    filename: str = 'summary_comparison.png'
):
    """Compare summary metrics (running, halting, etc.) over time."""
    _ensure_dir(out_dir)

    if df_a.empty and df_b.empty:
        return

    # Find time column
    time_col = None
    for candidate in ('begin', 'time', 't'):
        if (not df_a.empty and candidate in df_a.columns) or \
           (not df_b.empty and candidate in df_b.columns):
            time_col = candidate
            break

    if time_col is None:
        return

    metrics = ['running', 'halting', 'entered', 'left']
    available_metrics = []
    for m in metrics:
        if (not df_a.empty and m in df_a.columns) or (not df_b.empty and m in df_b.columns):
            available_metrics.append(m)

    if not available_metrics:
        return

    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 3 * n_metrics), sharex=True)

    if n_metrics == 1:
        axes = [axes]

    colors = {'A': 'steelblue', 'B': 'coral'}

    for i, metric in enumerate(available_metrics):
        ax = axes[i]

        for df, label in [(df_a, labels[0]), (df_b, labels[1])]:
            if df.empty or metric not in df.columns:
                continue
            ax.plot(df[time_col], df[metric], label=label,
                   color=colors.get(label[0], 'gray'), linewidth=1.5, alpha=0.8)

        ax.set_ylabel(metric.title())
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Simulation Summary Comparison', fontsize=12)

    out_file = os.path.join(out_dir, filename)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_efficiency_comparison(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    out_dir: str,
    labels: Tuple[str, str] = ('A', 'B'),
    filename: str = 'efficiency_comparison.png'
):
    """Compare time efficiency (useful travel time / total time)."""
    _ensure_dir(out_dir)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Calculate efficiency for each run
    efficiencies = {}
    for df, label in [(df_a, labels[0]), (df_b, labels[1])]:
        if df.empty or 'duration' not in df.columns or 'timeLoss' not in df.columns:
            continue
        valid = df['duration'] > 0
        eff = (df.loc[valid, 'duration'] - df.loc[valid, 'timeLoss']) / df.loc[valid, 'duration']
        efficiencies[label] = eff.values

    if not efficiencies:
        plt.close(fig)
        return

    n_a = len(efficiencies.get(labels[0], []))
    n_b = len(efficiencies.get(labels[1], []))

    # Histogram
    ax = axes[0]
    for label, eff in efficiencies.items():
        color = 'steelblue' if label == labels[0] else 'coral'
        ax.hist(eff, bins=30, alpha=0.5, label=legend_label_with_n(label, len(eff)),
                color=color, density=True)
    ax.set_xlabel('Time Efficiency')
    ax.set_ylabel('Density')
    ax.set_title(title_with_sample_note('Time Efficiency Distribution', n_a, n_b))
    ax.legend()
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

    # Box plot
    ax = axes[1]
    data = list(efficiencies.values())
    box_labels = [legend_label_with_n(lbl, len(d)) for lbl, d in efficiencies.items()]
    bp = ax.boxplot(data, labels=box_labels, patch_artist=True)
    colors = ['steelblue', 'coral']
    for patch, color in zip(bp['boxes'], colors[:len(data)], strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Time Efficiency')
    ax.set_title(title_with_sample_note('Time Efficiency Comparison', n_a, n_b))

    # Add mean values as text
    for i, (_label, eff) in enumerate(efficiencies.items()):
        mean_eff = np.mean(eff)
        ax.annotate(f'μ={mean_eff:.3f}', xy=(i + 1, mean_eff),
                   xytext=(10, 0), textcoords='offset points',
                   fontsize=9, va='center')

    out_file = os.path.join(out_dir, filename)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_speed_distribution_comparison(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    out_dir: str,
    labels: Tuple[str, str] = ('A', 'B'),
    filename: str = 'speed_distribution.png'
):
    """Compare speed distributions from tripinfo (calculated from routeLength/duration)."""
    _ensure_dir(out_dir)

    speeds = {}
    for df, label in [(df_a, labels[0]), (df_b, labels[1])]:
        if df.empty or 'duration' not in df.columns or 'routeLength' not in df.columns:
            continue
        valid = (df['duration'] > 0) & (df['routeLength'] > 0)
        speed_ms = df.loc[valid, 'routeLength'] / df.loc[valid, 'duration']
        speed_kmh = speed_ms * 3.6
        speeds[label] = speed_kmh.values

    if not speeds:
        return

    n_a = len(speeds.get(labels[0], []))
    n_b = len(speeds.get(labels[1], []))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Overlaid histogram
    ax = axes[0]
    for label, speed in speeds.items():
        color = 'steelblue' if label == labels[0] else 'coral'
        ax.hist(speed, bins=30, alpha=0.5, label=legend_label_with_n(label, len(speed)),
                color=color, density=True)
    ax.set_xlabel('Average Speed (km/h)')
    ax.set_ylabel('Density')
    ax.set_title(title_with_sample_note('Speed Distribution', n_a, n_b))
    ax.legend()

    # CDF
    ax = axes[1]
    for label, speed in speeds.items():
        color = 'steelblue' if label == labels[0] else 'coral'
        sorted_speed = np.sort(speed)
        cdf = np.arange(1, len(sorted_speed) + 1) / len(sorted_speed)
        ax.plot(sorted_speed, cdf, label=legend_label_with_n(label, len(speed)),
                color=color, linewidth=2)
    ax.set_xlabel('Average Speed (km/h)')
    ax.set_ylabel('CDF')
    ax.set_title(title_with_sample_note('Cumulative Speed Distribution', n_a, n_b))
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_file = os.path.join(out_dir, filename)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_waiting_time_analysis(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    out_dir: str,
    labels: Tuple[str, str] = ('A', 'B'),
    filename: str = 'waiting_time_analysis.png'
):
    """Detailed waiting time analysis."""
    _ensure_dir(out_dir)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    n_a_wait = len(df_a['waitingTime'].dropna()) if not df_a.empty and 'waitingTime' in df_a.columns else 0
    n_b_wait = len(df_b['waitingTime'].dropna()) if not df_b.empty and 'waitingTime' in df_b.columns else 0
    n_a_tl = len(df_a['timeLoss'].dropna()) if not df_a.empty and 'timeLoss' in df_a.columns else 0
    n_b_tl = len(df_b['timeLoss'].dropna()) if not df_b.empty and 'timeLoss' in df_b.columns else 0
    n_a_dd = len(df_a['departDelay'].dropna()) if not df_a.empty and 'departDelay' in df_a.columns else 0
    n_b_dd = len(df_b['departDelay'].dropna()) if not df_b.empty and 'departDelay' in df_b.columns else 0

    # Waiting time histogram
    ax = axes[0, 0]
    for df, label in [(df_a, labels[0]), (df_b, labels[1])]:
        if df.empty or 'waitingTime' not in df.columns:
            continue
        data = df['waitingTime'].dropna().values
        color = 'steelblue' if label == labels[0] else 'coral'
        ax.hist(data, bins=30, alpha=0.5, label=legend_label_with_n(label, len(data)),
                color=color, density=True)
    ax.set_xlabel('Waiting Time (s)')
    ax.set_ylabel('Density')
    ax.set_title(title_with_sample_note('Waiting Time Distribution', n_a_wait, n_b_wait))
    ax.legend()

    # Waiting time vs depart time
    ax = axes[0, 1]
    for df, label in [(df_a, labels[0]), (df_b, labels[1])]:
        if df.empty or 'waitingTime' not in df.columns or 'depart' not in df.columns:
            continue
        color = 'steelblue' if label == labels[0] else 'coral'
        pts = df[['depart', 'waitingTime']].dropna()
        ax.scatter(pts['depart'], pts['waitingTime'], alpha=0.3, s=10,
                   label=legend_label_with_n(label, len(pts)), color=color)
    ax.set_xlabel('Departure Time (s)')
    ax.set_ylabel('Waiting Time (s)')
    ax.set_title(title_with_sample_note('Waiting Time vs Departure Time', n_a_wait, n_b_wait))
    ax.legend()

    # Time loss histogram
    ax = axes[1, 0]
    for df, label in [(df_a, labels[0]), (df_b, labels[1])]:
        if df.empty or 'timeLoss' not in df.columns:
            continue
        data = df['timeLoss'].dropna().values
        color = 'steelblue' if label == labels[0] else 'coral'
        ax.hist(data, bins=30, alpha=0.5, label=legend_label_with_n(label, len(data)),
                color=color, density=True)
    ax.set_xlabel('Time Loss (s)')
    ax.set_ylabel('Density')
    ax.set_title(title_with_sample_note('Time Loss Distribution', n_a_tl, n_b_tl))
    ax.legend()

    # Depart delay histogram
    ax = axes[1, 1]
    for df, label in [(df_a, labels[0]), (df_b, labels[1])]:
        if df.empty or 'departDelay' not in df.columns:
            continue
        data = df['departDelay'].dropna().values
        color = 'steelblue' if label == labels[0] else 'coral'
        ax.hist(data, bins=30, alpha=0.5, label=legend_label_with_n(label, len(data)),
                color=color, density=True)
    ax.set_xlabel('Departure Delay (s)')
    ax.set_ylabel('Density')
    ax.set_title(title_with_sample_note('Departure Delay Distribution', n_a_dd, n_b_dd))
    ax.legend()

    out_file = os.path.join(out_dir, filename)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    out_dir: str,
    label: str = '',
    filename: str = 'correlation_heatmap.png'
):
    """Correlation heatmap for tripinfo metrics.

    Es un diagnóstico INTERNO de una sola corrida (qué tan acopladas están
    duration/timeLoss/waitingTime/routeLength/departDelay entre sí), no una
    comparación A vs B: la correlación no tiene una dirección "buena" o
    "mala" en sí misma — duration y timeLoss correlacionan cerca de 1.0 casi
    siempre porque timeLoss es, por definición, casi todo el contenido de
    duration en una red congestionada, no porque el sistema esté mejor o
    peor. Además, como cada heatmap se calcula solo sobre los viajes
    COMPLETADOS de esa corrida, si las corridas tienen conteos muy distintos
    (colapso de un lado) tampoco son directamente comparables entre sí — una
    diferencia entre el heatmap de A y el de B puede deberse a qué
    subconjunto de vehículos sobrevivió, no a un cambio real de
    comportamiento. No usar este gráfico para argumentar cuál corrida es
    superior; para eso están throughput/tiempo de sistema/pareado.
    """
    _ensure_dir(out_dir)

    if df.empty:
        return

    numeric_cols = ['duration', 'routeLength', 'departDelay', 'timeLoss', 'waitingTime']
    available_cols = [c for c in numeric_cols if c in df.columns]

    if len(available_cols) < 2:
        return

    corr_df = df[available_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    # Diverging rojo-verde centrado en 0 y con rango fijo [-1, 1]: así el
    # signo se lee de un vistazo (verde = correlación positiva, rojo =
    # negativa) en vez de RdYlBu_r auto-escalado, que con datos siempre
    # positivos (típico en estas 5 métricas) desperdiciaba la mitad de la
    # escala y hacía ver "rojo" (extremo alto) lo que solo era correlación
    # fuerte, no algo negativo. El signo NO implica "bueno/malo": una
    # correlación fuerte entre duration y timeLoss es estructural, no una
    # señal de desempeño.
    sns.heatmap(corr_df, annot=True, cmap='RdYlGn', center=0, vmin=-1, vmax=1,
                square=True, ax=ax, fmt='.2f')
    # Correlación calculada solo sobre viajes completados (n=len(df)); no es
    # una comparación A vs B, pero se anota el n para que quede claro que
    # excluye los viajes atrapados/no insertados.
    title = (f'Metric Correlations{" - " + label if label else ""} (n={len(df)})'
             '\n(diagnóstico interno — no compara A vs B)')
    ax.set_title(title)

    out_file = os.path.join(out_dir, filename)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_fcd_speed_heatmap(
    df_fcd: pd.DataFrame,
    out_dir: str,
    label: str = '',
    filename: str = 'fcd_speed_heatmap.png',
    time_bin: float = 30.0
):
    """Heatmap of average speed over time and position from FCD data."""
    _ensure_dir(out_dir)

    if df_fcd.empty:
        return

    df = df_fcd.copy()
    df['time_bin'] = (df['time'] // time_bin) * time_bin

    # Aggregate by time bin and edge
    pivot = df.groupby(['time_bin', 'edge'])['speed'].mean().unstack(fill_value=0)

    if pivot.empty or pivot.shape[1] < 2:
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(pivot.T, cmap='RdYlGn', ax=ax, cbar_kws={'label': 'Speed (m/s)'})
    ax.set_xlabel('Time Bin (s)')
    ax.set_ylabel('Edge')
    title = f'Speed Heatmap{" - " + label if label else ""}'
    ax.set_title(title)

    out_file = os.path.join(out_dir, filename)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_fcd_comparison(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    out_dir: str,
    labels: Tuple[str, str] = ('A', 'B'),
    filename: str = 'fcd_comparison.png',
    time_bin: float = 30.0
):
    """Compare FCD aggregated metrics between two runs."""
    _ensure_dir(out_dir)

    if df_a.empty and df_b.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Aggregate by time for both
    agg_data = {}
    for df, label in [(df_a, labels[0]), (df_b, labels[1])]:
        if df.empty:
            continue
        df_temp = df.copy()
        df_temp['time_bin'] = (df_temp['time'] // time_bin) * time_bin
        agg = df_temp.groupby('time_bin').agg({
            'id': 'nunique',
            'speed': ['mean', 'std']
        }).reset_index()
        agg.columns = ['time_bin', 'vehicle_count', 'avg_speed', 'speed_std']
        agg_data[label] = agg

    # Plot 1: Vehicle count over time
    ax = axes[0, 0]
    for label, agg in agg_data.items():
        color = 'steelblue' if label == labels[0] else 'coral'
        ax.plot(agg['time_bin'], agg['vehicle_count'], label=label, color=color, linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Active Vehicles')
    ax.set_title('Active Vehicle Count Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Average speed over time
    ax = axes[0, 1]
    for label, agg in agg_data.items():
        color = 'steelblue' if label == labels[0] else 'coral'
        ax.plot(agg['time_bin'], agg['avg_speed'], label=label, color=color, linewidth=2)
        ax.fill_between(agg['time_bin'],
                       agg['avg_speed'] - agg['speed_std'],
                       agg['avg_speed'] + agg['speed_std'],
                       alpha=0.2, color=color)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Average Speed Over Time (±1 std)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Speed distribution from FCD
    ax = axes[1, 0]
    for df, label in [(df_a, labels[0]), (df_b, labels[1])]:
        if df.empty or 'speed' not in df.columns:
            continue
        color = 'steelblue' if label == labels[0] else 'coral'
        ax.hist(df['speed'].dropna(), bins=40, alpha=0.5, label=label, color=color, density=True)
    ax.set_xlabel('Instantaneous Speed (m/s)')
    ax.set_ylabel('Density')
    ax.set_title('Instantaneous Speed Distribution (FCD)')
    ax.legend()

    # Plot 4: Speed by edge comparison (top 10 edges by traffic)
    ax = axes[1, 1]
    edge_speeds = {}
    for df, label in [(df_a, labels[0]), (df_b, labels[1])]:
        if df.empty:
            continue
        by_edge = df.groupby('edge')['speed'].mean()
        edge_speeds[label] = by_edge

    if len(edge_speeds) == 2:
        # Find common edges with most data
        common_edges = set(edge_speeds[labels[0]].index) & set(edge_speeds[labels[1]].index)
        if common_edges:
            counts = df_a.groupby('edge').size() + df_b.groupby('edge').size()
            top_edges = counts.loc[list(common_edges)].nlargest(10).index.tolist()

            x = np.arange(len(top_edges))
            width = 0.35
            speeds_a = [edge_speeds[labels[0]].get(e, 0) for e in top_edges]
            speeds_b = [edge_speeds[labels[1]].get(e, 0) for e in top_edges]

            ax.bar(x - width/2, speeds_a, width, label=labels[0], color='steelblue', alpha=0.8)
            ax.bar(x + width/2, speeds_b, width, label=labels[1], color='coral', alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(top_edges, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Avg Speed (m/s)')
            ax.set_title('Speed by Edge (Top 10 by Traffic)')
            ax.legend()

    out_file = os.path.join(out_dir, filename)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_improvement_summary(
    stats_a: Dict,
    stats_b: Dict,
    out_dir: str,
    labels: Tuple[str, str] = ('Baseline', 'Optimized'),
    filename: str = 'improvement_summary.png'
):
    """Summary visualization showing improvement percentages for key metrics."""
    _ensure_dir(out_dir)

    metrics_to_compare = {
        'duration': 'Travel Time',
        'timeLoss': 'Time Loss',
        'waitingTime': 'Waiting Time',
        'departDelay': 'Depart Delay',
    }

    improvements = []
    metric_labels = []
    used_median = []

    for metric, display_name in metrics_to_compare.items():
        if metric in stats_a and metric in stats_b:
            sa, sb = stats_a[metric], stats_b[metric]
            # La media está dominada por sesgo de supervivencia cuando los
            # conteos por lado difieren >10% (un lado colapsó y solo
            # promedia los viajes rápidos que escaparon o, en el caso de
            # departDelay, los pocos que alcanzaron a insertarse). En ese
            # caso se usa la mediana, igual que en la tabla "Métricas clave
            # por viaje" del HTML — así el gráfico no contradice al reporte.
            noncomparable = samples_noncomparable(sa.get('count'), sb.get('count'))
            if noncomparable:
                base_a, base_b = sa.get('median', 0), sb.get('median', 0)
            else:
                base_a, base_b = sa.get('mean', 0), sb.get('mean', 0)
            if base_a > 0:
                pct_change = ((base_b - base_a) / base_a) * 100
                improvements.append(-pct_change)  # Negative = improvement
                metric_labels.append(display_name + (' (mediana)' if noncomparable else ''))
                used_median.append(noncomparable)

    if not improvements:
        return

    n_a = stats_a.get('duration', {}).get('count')
    n_b = stats_b.get('duration', {}).get('count')

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax.barh(metric_labels, improvements, color=colors, alpha=0.7)

    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('Improvement (%)')
    title = title_with_sample_note(
        f'Performance Improvement: {labels[1]} vs {labels[0]}', n_a, n_b)
    if any(used_median):
        title += '\n(~ = mediana usada por sesgo de supervivencia; resultado real: ver 32_throughput_timeline.png)'
    ax.set_title(title)

    # Add value labels
    for bar, imp, is_median in zip(bars, improvements, used_median, strict=False):
        width = bar.get_width()
        prefix = '~' if is_median else ''
        ax.annotate(f'{prefix}{imp:+.1f}%',
                   xy=(width, bar.get_y() + bar.get_height() / 2),
                   xytext=(5 if width >= 0 else -5, 0),
                   textcoords='offset points',
                   va='center', ha='left' if width >= 0 else 'right',
                   fontweight='bold')

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='green', alpha=0.7, label='Improvement'),
        mpatches.Patch(facecolor='red', alpha=0.7, label='Degradation'),
    ]
    if any(used_median):
        legend_elements.append(mpatches.Patch(facecolor='none', edgecolor='none',
                                              label='~ = mediana (muestras no comparables)'))
    ax.legend(handles=legend_elements, loc='lower right')

    out_file = os.path.join(out_dir, filename)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_percentile_comparison(
    data_a: np.ndarray,
    data_b: np.ndarray,
    out_dir: str,
    labels: Tuple[str, str] = ('A', 'B'),
    metric_name: str = 'duration',
    filename: str = 'percentile_comparison.png'
):
    """Compare percentiles between two distributions."""
    _ensure_dir(out_dir)

    if len(data_a) == 0 and len(data_b) == 0:
        return

    percentiles = [5, 10, 25, 50, 75, 90, 95]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(percentiles))
    width = 0.35

    n_a, n_b = len(data_a), len(data_b)
    if n_a > 0:
        pcts_a = [np.percentile(data_a, p) for p in percentiles]
        ax.bar(x - width/2, pcts_a, width, label=legend_label_with_n(labels[0], n_a),
              color='steelblue', alpha=0.8)

    if n_b > 0:
        pcts_b = [np.percentile(data_b, p) for p in percentiles]
        ax.bar(x + width/2, pcts_b, width, label=legend_label_with_n(labels[1], n_b),
              color='coral', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f'P{p}' for p in percentiles])
    ax.set_xlabel('Percentile')
    ax.set_ylabel(metric_name.replace('_', ' ').title())
    ax.set_title(title_with_sample_note(
        f'{metric_name.replace("_", " ").title()} Percentile Comparison', n_a, n_b))
    ax.legend()

    out_file = os.path.join(out_dir, filename)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_congestion_timeline(
    df_summary_a: pd.DataFrame,
    df_summary_b: pd.DataFrame,
    out_dir: str,
    labels: Tuple[str, str] = ('A', 'B'),
    filename: str = 'congestion_timeline.png'
):
    """Plot congestion ratio (halting/running) over time."""
    _ensure_dir(out_dir)

    fig, ax = plt.subplots(figsize=(12, 5))

    time_col = None
    for candidate in ('begin', 'time', 't'):
        if (not df_summary_a.empty and candidate in df_summary_a.columns) or \
           (not df_summary_b.empty and candidate in df_summary_b.columns):
            time_col = candidate
            break

    if time_col is None:
        plt.close(fig)
        return

    for df, label in [(df_summary_a, labels[0]), (df_summary_b, labels[1])]:
        if df.empty or 'halting' not in df.columns or 'running' not in df.columns:
            continue

        color = 'steelblue' if label == labels[0] else 'coral'
        running = df['running'].replace(0, np.nan)
        congestion = df['halting'] / running

        ax.plot(df[time_col], congestion, label=label, color=color, linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Congestion Ratio (Halting/Running)')
    ax.set_title('Congestion Timeline Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    out_file = os.path.join(out_dir, filename)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
