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
from pathlib import Path
import os
import csv
import json
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .parsers import (
    parse_tripinfo, parse_summary, parse_fcd, parse_fcd_aggregated,
    get_available_outputs, compute_trip_statistics, compute_summary_statistics
)
from .plots import (
    # Basic plots
    plot_histogram_cdf_two, plot_boxplot_two, plot_time_series_mean,
    # Advanced comparison plots
    plot_metric_comparison_bars, plot_violin_comparison, plot_multi_metric_violin,
    plot_time_series_comparison, plot_summary_comparison, plot_efficiency_comparison,
    plot_speed_distribution_comparison, plot_waiting_time_analysis,
    plot_correlation_heatmap, plot_fcd_comparison, plot_improvement_summary,
    plot_percentile_comparison, plot_congestion_timeline
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
        (np.sort(completed_dur), 'steelblue', f'Completados (duration)'),
        (np.sort(time_in_network), 'tomato', f'Incompletos (time_in_network)'),
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
    for patch, color in zip(bp['boxes'], colors):
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
) -> List[str]:
    """
    Analyze incomplete trips (vehicles in FCD but not in tripinfo) and generate
    4 separate diagnostic plots.

    Called automatically from compare_runs() for run_A.

    Args:
        run_dir:  Simulation directory (must contain fcd.xml and routes.rou.xml)
        df_trip:  Parsed tripinfo DataFrame for this run
        out_dir:  Output directory for the 4 PNGs
        label:    Run label shown in plot titles

    Returns:
        List of generated filenames.
    """
    print(f'  Analyzing incomplete trips for {label}...')
    df_incomplete = _compute_incomplete_trips(run_dir, df_trip)

    if df_incomplete.empty:
        print(f'  No incomplete trips data available (fcd.xml missing?)')
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

    _plot_incomplete_histogram(completed_dur, time_in_network, out_dir, label)
    generated.append('21_incomplete_histogram.png')

    _plot_incomplete_cdf(completed_dur, time_in_network, out_dir, label)
    generated.append('22_incomplete_cdf.png')

    _plot_incomplete_boxplot(completed_dur, time_in_network, out_dir, label)
    generated.append('23_incomplete_boxplot.png')

    _plot_incomplete_scatter(df_incomplete, completed_dur, out_dir, label)
    generated.append('24_incomplete_scatter.png')

    # Save CSV
    csv_path = Path(out_dir) / 'incomplete_trips.csv'
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

        # 16. Incomplete trips analysis (runs 21–24)
        incomplete_files = analyze_incomplete_trips(
            run_a, df_trip_a, out_dir, label=labels[0]
        )
        generated_files.extend(incomplete_files)

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

    # Use native SUMO tools if requested
    if use_sumo_tools:
        try:
            from .sumo_tools import generate_all_sumo_plots, check_sumo_tools_available
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

    print(f"Analysis complete! Generated {len(generated_files)} files in {out_dir}")

    return {
        'out_dir': str(out_dir),
        'csv': str(csv_path),
        'json': str(json_path),
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
